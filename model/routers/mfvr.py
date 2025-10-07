# model/routers/mfvr.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import math
import os
from tqdm import tqdm
import wandb

from model.adapters.granite_adapter import prepare_granite_bayesian_routers, save_granite_bayesian_routers

from .base import MoERouter
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

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

# --- Component 2: The Custom Training Function ---
def train_mfvr_router(model, tokenizer, train_loader, val_loader, args):
    """Custom training loop for the MeanFieldVariationalRouter using the ELBO loss."""
    
    run_name = f"mfvr-{args.model_shortcode}-{args.dataset_shortcode}"

    # === 1. Prepare Model for Training ===
    # Freeze all parameters in the entire model first
    print("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    causal_model = model.base_model.model.model

    model = prepare_granite_bayesian_routers(model, method="mfvr", args=args)

    # === 2. Create Optimizer ===
    
    # Get the list of all currently trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # === 3. Run Custom Training Loop ===

    project_name = "bayesian-router-finetuning"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    num_training_batches = len(train_loader)
    
    print("--- Starting MFVR Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            reconstruction_loss = outputs.loss
            
            total_kl_div = 0
            for layer_idx in args.train_layers:
                router = causal_model.layers[layer_idx].block_sparse_moe.router
                total_kl_div += router.kl_divergence()
            
            kl_term = (args.beta / num_training_batches) * total_kl_div
            loss = reconstruction_loss + kl_term
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            
        print(f"Epoch {epoch+1} average training loss: {total_epoch_loss / num_training_batches:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

    print("--- MFVR Fine-tuning complete ---")
    
    save_granite_bayesian_routers(model, method="mfvr", args=args)