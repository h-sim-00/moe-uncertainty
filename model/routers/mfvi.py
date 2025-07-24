import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
import wandb

from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

# --- Component 1: The Custom Variational Layer ---
class VariationalLinear(nn.Module):
    """
    A Linear layer with a learnable mean and variance for its weights and biases,
    implementing the Bayes by Backprop algorithm.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Variational parameters for the weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Variational parameters for the bias (if applicable)
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_var', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize means with a standard method
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize log variances to a large negative value (small variance)
        nn.init.constant_(self.weight_log_var, -10.0)
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_log_var, -10.0)

    def forward(self, x: torch.Tensor):
        # Use the reparameterization trick to sample weights and biases
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight_eps = torch.randn_like(weight_std)
        sampled_weight = self.weight_mu + (weight_eps * weight_std)

        if self.has_bias:
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias_eps = torch.randn_like(bias_std)
            sampled_bias = self.bias_mu + (bias_eps * bias_std)
        else:
            sampled_bias = None
        
        # --- KL Divergence Calculation ---
        # This is the complexity penalty part of the ELBO loss: log q(w) - log p(w)
        # We assume a standard Normal prior N(0,1) for p(w)
        
        # Log probability of the sampled weights under the variational posterior q(w)
        log_q_weight = -0.5 * (self.weight_log_var + (sampled_weight - self.weight_mu)**2 / torch.exp(self.weight_log_var)).sum()
        # Log probability of the sampled weights under the prior p(w)
        log_p_weight = -0.5 * (sampled_weight**2).sum()
        
        self.kl_div = log_q_weight - log_p_weight
        
        if self.has_bias:
            log_q_bias = -0.5 * (self.bias_log_var + (sampled_bias - self.bias_mu)**2 / torch.exp(self.bias_log_var)).sum()
            log_p_bias = -0.5 * (sampled_bias**2).sum()
            self.kl_div += (log_q_bias - log_p_bias)

        return F.linear(x, sampled_weight, sampled_bias)


# --- Component 2: The MFVI Router Class ---
class MFVI_Router(nn.Module):
    """Router that uses a VariationalLinear layer."""
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        # Replace nn.Linear with our custom VariationalLinear
        self.layer = VariationalLinear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
        logits = self.layer(hidden_states).float()
        # The rest of the routing logic is identical to the deterministic version
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

    def get_kl_divergence(self):
        return self.layer.kl_div


# --- Component 3: Model Modification and Loading ---
def add_mfvi_routers_to_model(model):
    """Performs the 'Lego swap' for MFVI routers."""
    print("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    device = model.device
    causal_model = model.base_model.model
    config = causal_model.config

    print("Swapping routers and initializing MFVI parameters...")
    for layer in causal_model.layers:
        old_router = layer.block_sparse_moe.router
        new_router = MFVI_Router(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        ).to(device)
        
        # Initialize the mean of the new router with the old router's weights
        new_router.layer.weight_mu.data.copy_(old_router.layer.state_dict()['weight'])
        
        layer.block_sparse_moe.router = new_router
        for param in layer.block_sparse_moe.router.parameters():
            param.requires_grad = True
            
    return model

def load_mfvi_routers_into_model(model, router_state_dicts):
    """Loads pre-trained MFVI router weights for evaluation."""
    # This function is similar to add_mfvi_routers_to_model but also loads weights
    print("Loading trained MFVI router weights into the model...")
    device = model.device
    causal_model = model.base_model.model
    config = causal_model.config

    for i, layer in enumerate(causal_model.layers):
        new_router = MFVI_Router(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        ).to(device)
        state_dict = router_state_dicts[f"layer_{i}"]
        new_router.load_state_dict(state_dict)
        layer.block_sparse_moe.router = new_router
        
    return model


# --- Component 4: The Custom Training Function ---
def train_mfvi_router(model, tokenizer, train_loader, val_loader, args):
    """Custom training loop for MFVI using the ELBO loss."""
    project_name = "bayesian-router-finetuning"
    run_name = f"MFVI_{args.model_shortcode}_seed-{args.seed}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    # Collect only the trainable router parameters for the optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    num_training_batches = len(train_loader)
    
    print("--- Starting MFVI Router Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            # Move data to the correct device
            inputs = {k: v.to(model.device) for k, v in data.items()}
            
            # Standard forward pass
            outputs = model(**inputs)
            
            # --- ELBO Loss Calculation ---
            # 1. Reconstruction Loss (Standard Cross-Entropy)
            # Shift logits and labels for Causal LM loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs['labels'][..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            reconstruction_loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            
            # 2. KL Divergence
            total_kl_div = 0
            for layer in model.base_model.model.layers:
                total_kl_div += layer.block_sparse_moe.router.get_kl_divergence()
            
            # Scale KL term by the number of batches (common practice)
            kl_term = total_kl_div / num_training_batches
            
            # 3. Total ELBO loss
            loss = reconstruction_loss + kl_term
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "reconstruction_loss": reconstruction_loss.item(), "kl_term": kl_term.item()})

        print(f"Epoch {epoch+1} average training loss: {total_loss / num_training_batches:.4f}")
        # Add validation loop here if desired

    print("--- MFVI Fine-tuning complete ---")
    
    final_router_states = {
        f"layer_{i}": layer.block_sparse_moe.router.state_dict()
        for i, layer in enumerate(model.base_model.model.layers)
    }
    
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "router_weights.pt")
    print(f"Saving the final router weights to {final_save_path}")
    torch.save(final_router_states, final_save_path)


# --- Component 5: The Evaluation Function ---
def evaluate_mfvi_router(model, tokenizer, dataset, dataset_name, num_samples=10, batch_size=8):
    """Orchestrates model evaluation using Monte Carlo sampling."""
    print(f"--- Evaluating on {dataset_name} with {num_samples} MFVI samples ---")
    model.eval()
    
    all_probs = []
    for i in tqdm(range(num_samples), desc="MC Samples"):
        _, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
        all_probs.append(probs)
    
    stacked_probs = torch.stack(all_probs)
    mean_probs = stacked_probs.mean(dim=0)
    final_preds = torch.argmax(mean_probs, dim=1)

    acc = calculate_accuracy(final_preds, labels)
    nll = calculate_nll(mean_probs, labels)
    ece, mce = calculate_ece_mce(mean_probs, labels)
    
    results = {
        'dataset': dataset_name, 'ACC': acc.item(), 'NLL': nll.item(),
        'ECE': ece.item(), 'MCE': mce.item(),
    }
    print(results)
    return results