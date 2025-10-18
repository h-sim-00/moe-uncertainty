# model/routers/mfvr.py

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb


from .base import MoERouter

# --- Component 1: The Mean-Field Variational Router Class ---
class MeanFieldVariationalRouter(MoERouter):
    """
    Implements MFVI on the logit space with a residual mean and a separate
    network for the variance (VAE-style).
    """
    def __init__(self, config, existing_router):
        super(MoERouter, self).__init__() # Call grandparent's init
        
        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        
        # 1. Base Mean Network (Frozen) - Our MAP estimate
        self.mean_base = nn.Linear(self.input_size, self.num_experts, bias=False)
        self.mean_base.load_state_dict(existing_router.layer.state_dict())

        # 2. Residual Mean Network (Trainable)
        self.mean_residual_net = nn.Linear(self.input_size, self.num_experts, bias=False)

        # 3. Log Variance Network (Trainable)
        self.log_var_net = nn.Sequential(
            nn.Linear(self.input_size, config.hidden_size // 4, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, self.num_experts, bias=False)
        )

        # Initialize the last layer weights to small values so the output is near zero
        nn.init.normal_(self.log_var_net[2].weight, mean=0.0, std=1e-3)
        
        self.num_mc_samples_inference = 35
        self.last_mu_residual = None
        self.last_log_var = None


    def forward(self, hidden_states, **kwargs):
        with torch.no_grad():
            mu_base = self.mean_base(hidden_states)
        mu_residual = self.mean_residual_net(hidden_states)
        mu_final = mu_base + mu_residual
        log_var = self.log_var_net(hidden_states)
        
        logit_dist = torch.distributions.Normal(mu_final, torch.exp(0.5 * log_var))
        self.last_mu_residual = mu_residual
        self.last_log_var = log_var

        if self.training:
            logits = logit_dist.rsample()
        else:
            # Strategy A: Robust Estimation via internal MC sampling
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
        """KL divergence between N(mu_residual, exp(log_var)) and N(0, I) for the last forward pass."""
        mu = self.last_mu_residual
        log_var = self.last_log_var
        # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        kl = 0.5 * (torch.exp(log_var) + mu ** 2 - 1 - log_var)
        return kl.sum()

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'mean_residual_net': self.mean_residual_net.state_dict(),
            'log_var_net': self.log_var_net.state_dict()
        }, path)
        print(f"Saved MFVR weights to {path}")

    def load_weights(self, path: str, device=None):
        state_dicts = torch.load(path, map_location=device)
        self.mean_residual_net.load_state_dict(state_dicts['mean_residual_net'])
        self.log_var_net.load_state_dict(state_dicts['log_var_net'])
        print(f"Loaded MFVR weights from {path}")
