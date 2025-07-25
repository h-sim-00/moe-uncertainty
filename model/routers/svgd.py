# TODO:
# MODIFICATION:
# N router particles: N set of paramters, N separate tensors.
# Remember each model have L layers, so each tensor is N * (L * weight_matrix)

# PROBLEM:
# Can we train all the routers together in this scheme?
# Answer is yes for now.
# Train all routers together, during inference time, do MC sampling averaging across all layers.

# TRAIN PROCESS:
# 1) Loop through the N router particles, load each one into the main LLM, and calculate the gradient of the log-posterior.
# 2) Compute the kernel-based repulsive force between all particles.
# 3) Combine the forces to get the final SVGD update and apply it to all N particle tensors.

# model/routers/svgd.py

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

from utils import (get_model_predictions, 
                   calculate_accuracy, calculate_ece_mce, calculate_nll)

# --- Component 1: Helper Functions ---
def _rbf_kernel(X, h=-1.):
    """
    Computes the RBF kernel matrix and its gradient.
    X: A tensor of particles, shape (N, D)
    h: bandwidth. If -1, use the median heuristic.
    """
    XY = X @ X.t()
    X2 = (X**2).sum(1).view(-1, 1)
    D2 = X2 + X2.t() - 2 * XY

    if h == -1:
        # Median heuristic for bandwidth selection
        h = torch.sqrt(0.5 * D2.median() / torch.log(torch.tensor(X.size(0), dtype=torch.float32)))

    K = torch.exp(-D2 / h**2 / 2)
    
    # Analytical gradient of the kernel
    grad_K = -torch.matmul(K, X)
    sum_K = K.sum(1, keepdim=True)
    grad_K += sum_K * X
    grad_K /= (h**2)
    
    return K, grad_K

def _get_all_router_weights(model):
    """Flattens and concatenates all router weights into a single vector."""
    causal_model = model.base_model.model
    all_weights = []
    for layer in causal_model.model.layers:
        for param in layer.block_sparse_moe.router.layer.parameters():
            all_weights.append(param.view(-1))
    return torch.cat(all_weights)

def _set_all_router_weights(model, particle_vector):
    """Takes a flattened vector and loads it back into the model's routers."""
    causal_model = model.base_model.model
    offset = 0
    for layer in causal_model.model.layers:
        for param in layer.block_sparse_moe.router.layer.parameters():
            numel = param.numel()
            param.data.copy_(particle_vector[offset:offset + numel].view_as(param))
            offset += numel

# --- Component 2: The Custom Training Function ---
def train_svgd_router(model, tokenizer, train_loader, args):
    """
    Custom training loop for SVGD.
    """
    project_name = "bayesian-router-finetuning"
    run_name = f"svgd_{args.model_shortcode}_seed-{args.seed}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    # 1. Initialize Particles
    device = model.device
    map_weights = _get_all_router_weights(model).detach()
    num_params = map_weights.numel()
    
    # Initialize N particles around the MAP estimate with small noise
    particles = map_weights.unsqueeze(0).repeat(args.num_particles, 1)
    particles += torch.randn_like(particles) * 0.01
    particles.requires_grad = True

    # 2. Setup Optimizer for the particle tensor
    optimizer = torch.optim.Adam([particles], lr=args.lr)
    
    # 3. Freeze all non-router parameters in the main model
    for param in model.parameters():
        param.requires_grad = False

    print("--- Starting SVGD Router Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            all_log_post_grads = []
            # Step A: Calculate log-posterior gradient for each particle
            for i in range(args.num_particles):
                # Load particle i into the model's routers
                _set_all_router_weights(model, particles[i])
                # Forward pass to get end-to-end loss
                outputs = model(**inputs)
                neg_log_likelihood = outputs.loss
                # L2 regularization (negative log-prior)
                l2_term = (args.weight_decay / 2.0) * torch.sum(particles[i]**2)
                neg_log_posterior = neg_log_likelihood + l2_term
                # BackProps
                neg_log_posterior.backward()
                
                grad_vec = []
                for layer in model.base_model.model.model.layers:
                    for param in layer.block_sparse_moe.router.layer.parameters():
                        grad_vec.append(param.grad.view(-1))
                        param.grad.zero_() # Clear grads for the next particle
                
                all_log_post_grads.append(torch.cat(grad_vec))

            stacked_grads = torch.stack(all_log_post_grads)

            # Step B: Kernel and Repulsive Force
            kernel_matrix, kernel_grad = _rbf_kernel(particles.detach())
            
            # Step C: SVGD Update
            # We want to ascend the log-posterior, so we use the positive gradient
            phi = (torch.matmul(kernel_matrix, -stacked_grads) + kernel_grad) / args.num_particles
            
            # Manually set the gradient for the optimizer and step
            particles.grad = -phi
            optimizer.step()
            
            wandb.log({"train_loss": neg_log_posterior.item()})

    print("--- SVGD Fine-tuning complete ---")
    
    # Save the final particle cloud
    save_dir = os.path.join("./adapters", run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "router_particles.pt")
    print(f"Saving the final SVGD particles to {final_save_path}")
    torch.save(particles.detach(), final_save_path)

# --- Component 3: The Evaluation Function ---
def evaluate_svgd_router(model, tokenizer, particles, dataset, dataset_name, batch_size=8):
    """Orchestrates model evaluation using the trained SVGD particles."""
    num_samples = particles.shape[0]
    print(f"--- Evaluating on {dataset_name} with {num_samples} SVGD particles ---")
    model.eval()
    
    all_probs = []
    for i in tqdm(range(num_samples), desc="SVGD MC Samples"):
        # Load the i-th particle into the model's routers
        with torch.no_grad():
            _set_all_router_weights(model, particles[i])
        
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