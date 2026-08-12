# model/routers/fcvr.py

import math
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
    def __init__(self, config, existing_router, prior_std=1.0, input_layernorm=False,
                 separate_trunks=False):
        super(MoERouter, self).__init__() # Call grandparent's init

        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.top_k = config.num_experts_per_tok

        # Std of the isotropic KL prior N(0, prior_std^2 I) on the residual.
        # 1.0 reproduces the original N(l_det, I) prior exactly.
        self.prior_std = float(prior_std)

        # Optional parameter-free normalization of the hidden state on its way
        # into the variational trunk(s) ONLY (mean_base sees raw hidden states).
        # elementwise_affine=False => no parameters, checkpoint format unchanged.
        self.input_layernorm = (
            nn.LayerNorm(self.input_size, elementwise_affine=False) if input_layernorm else None
        )

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

        # Optional separate trunk for the Cholesky head (the thesis' original
        # two-trunk design). When set, cholesky_head reads its own features
        # instead of sharing the mean head's backbone.
        if separate_trunks:
            self.cholesky_backbone = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size, bias=False),
                nn.ReLU()
            )
        else:
            self.cholesky_backbone = None

        self.num_mc_samples_inference = 35
        self.last_mu_residual = None
        self.last_cholesky_factor = None

    def _build_cholesky(self, flat_cholesky_elements):
        """Helper to build a batch of lower-triangular matrices."""
        batch_size = flat_cholesky_elements.shape[0]
        L = torch.zeros(batch_size, self.num_experts, self.num_experts,
                        device=flat_cholesky_elements.device,
                        dtype=flat_cholesky_elements.dtype)
        
        tril_indices = torch.tril_indices(row=self.num_experts, col=self.num_experts, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = flat_cholesky_elements
        
        # Exponentiate diagonal to ensure positivity for a valid Cholesky factor
        diag_indices = torch.arange(self.num_experts)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        
        return L

    def forward(self, hidden_states, **kwargs):
        with torch.no_grad():
            mu_base = self.mean_base(hidden_states)

        trunk_input = hidden_states
        if self.input_layernorm is not None:
            trunk_input = self.input_layernorm(trunk_input)

        # Shared backbone feeds both heads (unless a separate Cholesky trunk is used).
        features = self.backbone(trunk_input)
        mu_residual = self.mean_head(features)
        mu_final = mu_base + mu_residual

        # Predict and build the Cholesky factor.
        if self.cholesky_backbone is not None:
            cholesky_features = self.cholesky_backbone(trunk_input)
        else:
            cholesky_features = features
        flat_cholesky = self.cholesky_head(cholesky_features)
        cholesky_factor = self._build_cholesky(flat_cholesky)
        
        # MODIFIED: Use a MultivariateNormal distribution
        logit_dist = torch.distributions.MultivariateNormal(mu_final, scale_tril=cholesky_factor)
        
        # Store parameters needed for the KL divergence
        self.last_mu_residual = mu_residual
        self.last_cholesky_factor = cholesky_factor

        # The rest of the sampling and routing logic is unchanged
        if self.training:
            logits = logit_dist.rsample()
        else:
            logit_samples = logit_dist.rsample(sample_shape=torch.Size([self.num_mc_samples_inference]))
            probs_samples = torch.softmax(logit_samples, dim=-1)
            mean_probs = probs_samples.mean(dim=0)
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

    def kl_divergence(self):
        """MODIFIED: KL divergence for the multivariate case."""
        mu = self.last_mu_residual
        L = self.last_cholesky_factor
        E = self.num_experts
        s2 = self.prior_std ** 2

        # KL[N(mu, LL^T) || N(0, s^2 I)]
        #   = 0.5 * ((Tr(LL^T) + mu^T mu)/s^2 - E + 2E*log(s) - log_det(LL^T))
        # At prior_std=1 this reduces to the original N(0, I) expression.
        trace_term = (L**2).sum(dim=(-1, -2)) / s2
        mu_sq_term = (mu**2).sum(-1) / s2
        log_det_term = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)

        kl = 0.5 * (trace_term + mu_sq_term - E + 2 * E * math.log(self.prior_std) - log_det_term)
        # Per-token MEAN (matches the mean reduction of the reconstruction loss),
        # so the ELBO's two terms share the same normalisation as in the paper.
        return kl.mean()

    def save_weights(self, path: str):
        """MODIFIED: Saves the new trainable components."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'backbone': self.backbone.state_dict(),
            'mean_head': self.mean_head.state_dict(),
            'cholesky_head': self.cholesky_head.state_dict()
        }
        if self.cholesky_backbone is not None:
            payload['cholesky_backbone'] = self.cholesky_backbone.state_dict()
        torch.save(payload, path)
        print(f"Saved FCVR weights to {path}")

    def load_weights(self, path: str, device=None):
        """MODIFIED: Loads the new trainable components (shared backbone + heads)."""
        state_dicts = torch.load(path, map_location=device)
        self.backbone.load_state_dict(state_dicts['backbone'])
        self.mean_head.load_state_dict(state_dicts['mean_head'])
        self.cholesky_head.load_state_dict(state_dicts['cholesky_head'])
        if self.cholesky_backbone is not None:
            if 'cholesky_backbone' not in state_dicts:
                raise KeyError(
                    f"Checkpoint {path} has no 'cholesky_backbone' but the router was "
                    f"built with separate_trunks=True - flag mismatch between train and eval."
                )
            self.cholesky_backbone.load_state_dict(state_dicts['cholesky_backbone'])
        elif 'cholesky_backbone' in state_dicts:
            raise KeyError(
                f"Checkpoint {path} contains 'cholesky_backbone' but the router was "
                f"built with separate_trunks=False - flag mismatch between train and eval."
            )
        print(f"Loaded FCVR weights from {path}")