# model/routers/fcvr.py

import torch
import torch.nn as nn
import os
from .base import MoERouter

# --- Component 1: The Full-Covariance Variational Router Class ---
class FullCovarianceVariationalRouter(MoERouter):
    """
    Implements Full-Covariance Variational Bayes on the logit space,
    adapting the user's residual MFVR design.
    """
    def __init__(self, config, existing_router):
        super(MoERouter, self).__init__() # Call grandparent's init
        
        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        
        # 1. Base Mean Network (Frozen prior mean) - Unchanged
        self.mean_base = nn.Linear(self.input_size, self.num_experts, bias=False)
        self.mean_base.load_state_dict(existing_router.layer.state_dict())
        for param in self.mean_base.parameters():
            param.requires_grad = False

        # 2. Shared Uncertainty Backbone (Trainable) - paper Fig 5 / App C.2.
        #    A single trunk extracts features feeding BOTH the residual-mean head
        #    (shift) and the Cholesky head (correlation).
        self.hidden_size = config.hidden_size // 4
        self.backbone = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=False),
            nn.ReLU()
        )

        # 3. Heads (Trainable)
        # Head A: Residual Mean (shift)
        self.mean_head = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Head B: Cholesky Factor (correlation). Output is the flattened
        # lower-triangular matrix.
        num_cholesky_elements = self.num_experts * (self.num_experts + 1) // 2
        self.cholesky_head = nn.Linear(self.hidden_size, num_cholesky_elements, bias=False)

        # Init Cholesky head near zero so we start with (near-)identity covariance.
        nn.init.normal_(self.cholesky_head.weight, mean=0.0, std=1e-3)
        
        self.num_mc_samples_inference = 35
        self.last_mu_residual = None
        self.last_cholesky_factor = None
        # When True (eval only), route with the deterministic posterior mean
        # (mu_final) instead of MC-sampling the logit distribution. The Cholesky
        # factor is still computed and stored, so the Inf-Logit-Var read-out is
        # unaffected -- but routing becomes reproducible and ~S times faster.
        # Used for per-token signal analysis (Step 1); default off so normal
        # FCVR evaluation is unchanged.
        self.deterministic_readout = False

    def _build_cholesky(self, flat_cholesky_elements):
        """Helper to build a batch of lower-triangular matrices."""
        batch_size = flat_cholesky_elements.shape[0]
        L = torch.zeros(batch_size, self.num_experts, self.num_experts, device=flat_cholesky_elements.device)
        
        tril_indices = torch.tril_indices(row=self.num_experts, col=self.num_experts, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = flat_cholesky_elements
        
        # Exponentiate diagonal to ensure positivity for a valid Cholesky factor
        diag_indices = torch.arange(self.num_experts)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        
        return L

    def forward(self, hidden_states, **kwargs):
        with torch.no_grad():
            mu_base = self.mean_base(hidden_states)

        # Shared backbone feeds both heads.
        features = self.backbone(hidden_states)
        mu_residual = self.mean_head(features)
        mu_final = mu_base + mu_residual

        # Predict and build the Cholesky factor from the same features.
        flat_cholesky = self.cholesky_head(features)
        cholesky_factor = self._build_cholesky(flat_cholesky)
        
        # MODIFIED: Use a MultivariateNormal distribution
        logit_dist = torch.distributions.MultivariateNormal(mu_final, scale_tril=cholesky_factor)
        
        # Store parameters needed for the KL divergence
        self.last_mu_residual = mu_residual
        self.last_cholesky_factor = cholesky_factor

        # The rest of the sampling and routing logic is unchanged
        if self.training:
            logits = logit_dist.rsample()
        elif self.deterministic_readout:
            # Deterministic posterior-mean routing (no MC sampling). The
            # covariance is already stored above for read-out.
            logits = mu_final
        else:
            # Draw the S MC samples one at a time: rsample([S]) broadcasts the
            # [N, E, E] scale_tril to [S, N, E, E] inside matmul, which OOMs
            # for E=256 on long-prompt batches. Sequential draws keep the peak
            # at the [N, E, E] Cholesky itself; the estimator is unchanged.
            mean_probs = None
            for _ in range(self.num_mc_samples_inference):
                probs = torch.softmax(logit_dist.rsample(), dim=-1)
                mean_probs = probs if mean_probs is None else mean_probs + probs
            mean_probs = mean_probs / self.num_mc_samples_inference
            logits = torch.log(mean_probs.clamp(min=1e-9))
        
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        
        batch_size = hidden_states.shape[0]
        zeros = torch.zeros((batch_size, self.num_experts), dtype=torch.long, device=logits.device)
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()
        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]
        
        return index_sorted_experts, batch_index, batch_gates, expert_size, logits

    def kl_divergence_per_token(self):
        """Per-position KL[N(mu, LL^T) || N(0, I)] over the last forward's tokens.
        Shape [B*T] (the router sees hidden states flattened row-major over
        (batch, position), see modeling_granitemoe.GraniteMoeMoE.forward)."""
        mu = self.last_mu_residual
        L = self.last_cholesky_factor
        E = self.num_experts

        # KL[N(mu, LL^T) || N(0, I)] = 0.5 * (Tr(LL^T) + mu^T mu - E - log_det(LL^T))
        trace_term = (L**2).sum(dim=(-1, -2))
        mu_sq_term = (mu**2).sum(-1)
        log_det_term = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return 0.5 * (trace_term + mu_sq_term - E - log_det_term)

    def kl_divergence(self, mask=None):
        """MODIFIED: KL divergence for the multivariate case, per-token MEAN
        (matches the mean reduction of the reconstruction loss, so the ELBO's two
        terms share the same normalisation as in the paper).

        mask: optional [B*T] {0,1} tensor (flattened row-major like the router
        input). With `attention_mask.flatten()` the mean runs over REAL tokens only
        -- padding positions are not part of the model and must not be
        regularised (they made the effective beta depend on batch composition).
        With `(labels != -100).flatten()` the KL is restricted to answer/target
        positions (an ablation of the weighting; the router latent does exist at
        prompt positions, so prompt-KL is part of the ELBO proper).
        mask=None keeps the previous behaviour (mean over every position, pads
        included) for backward comparability."""
        kl = self.kl_divergence_per_token()
        if mask is None:
            return kl.mean()
        mask = mask.to(kl.dtype).reshape(-1)
        if mask.shape[0] != kl.shape[0]:
            raise ValueError(f"kl mask has {mask.shape[0]} positions but the router saw {kl.shape[0]}")
        denom = mask.sum().clamp(min=1.0)
        return (kl * mask).sum() / denom

    def save_weights(self, path: str):
        """MODIFIED: Saves the new trainable components."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'backbone': self.backbone.state_dict(),
            'mean_head': self.mean_head.state_dict(),
            'cholesky_head': self.cholesky_head.state_dict()
        }, path)
        print(f"Saved FCVR weights to {path}")

    def load_weights(self, path: str, device=None):
        """MODIFIED: Loads the new trainable components (shared backbone + heads)."""
        state_dicts = torch.load(path, map_location=device)
        self.backbone.load_state_dict(state_dicts['backbone'])
        self.mean_head.load_state_dict(state_dicts['mean_head'])
        self.cholesky_head.load_state_dict(state_dicts['cholesky_head'])
        print(f"Loaded FCVR weights from {path}")