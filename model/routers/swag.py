import torch
import torch.nn as nn
from torch.optim.swa_utils import SWAG
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb
import os
from tqdm import tqdm

from utils import (get_model_predictions, 
                   calculate_accuracy, calculate_ece_mce, calculate_nll)


# --- Component 1: The Standard Router Class ---
class GraniteMoeDeterministicRouter(nn.Module):
    """A standard, non-stochastic router."""
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states, mode="top_k", temp=1.0):
        logits = self.layer(hidden_states).float()
        # --- The rest of the routing logic remains identical ---
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


# --- Component 2: The SWAG Training Function ---
def train_swag_router(model, tokenizer, train_dataset, val_dataset, args):
    """
    Fine-tunes the router and collects SWAG statistics.
    """
    causal_model = model.base_model.model
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    
    # 1. Freeze params and swap in deterministic routers
    print("Freezing all model parameters and swapping in deterministic routers...")
    # TODO: Implement freezing and substitute routers logic
    #       Don't forget to load the weights

    # 2. Train the routers to get a good starting point using the Trainer
    # TODO: Load the Laplace MAP weights, no need to retrain
    
    # 3. SWAG collection phase
    # TODO: Implement the details using torch.optim.swa_utils
    #       Everything is conseptual here.
    #       Decide what to store and how to use swag weights.
    print("--- Starting SWAG collection phase ---")
    
    swag_models = {}
    for i, layer in enumerate(causal_model.model.layers):
        router = layer.block_sparse_moe.router
        swag_models[f"layer_{i}"] = SWAG(router, swag_lr=args.swa_lr, max_num_models=20)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.swa_lr) # Use SGD for SWAG
    loss_fn = nn.CrossEntropyLoss()

    # Continue training for a few more epochs to collect weights
    for epoch in range(args.swa_epochs):
        model.train()
        for data, labels in tqdm(train_loader, desc=f"SWAG Epoch {epoch+1}/{args.swa_epochs}"):
            # Standard training step
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs.logits, labels) # Assuming model output is an object with logits
            loss.backward()
            optimizer.step()
        
        # Update SWAG statistics for each router
        for i, layer in enumerate(model.base_model.model.layers):
            swag_models[f"layer_{i}"].update_parameters(layer.block_sparse_moe.router)

    print("--- SWAG collection complete ---")
    
    # Update BN statistics for all SWAG models
    for swag_model in swag_models.values():
        swag_model.update_bn(train_loader)
        
    # 4. Save the fitted SWAG objects
    run_name = f"swag_{args.model_shortcode}_seed-{args.seed}"
    save_dir = os.path.join("./adapters", run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "swag_routers.pt")
    print(f"Saving the fitted SWAG objects to {final_save_path}")
    torch.save(swag_models, final_save_path)


# --- Component 3: The Evaluation Function ---
def evaluate_swag_router(model, tokenizer, swag_objects, dataset, dataset_name, num_samples=10, batch_size=8):
    """Orchestrates model evaluation using Monte Carlo sampling from SWAG."""
    causal_model = model.base_model.model
    print(f"--- Evaluating on {dataset_name} with {num_samples} SWAG samples ---")
    model.eval()
    all_probs = []

    _, _, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    for i in tqdm(range(num_samples), desc="SWAG MC Samples"):
        # For each sample, draw and apply new weights for all routers
        with torch.no_grad():
            for layer_idx, layer in enumerate(causal_model.model.layers):
                # TODO: How to use swag objects? Need to research
                swag_model = swag_objects[f"layer_{layer_idx}"]
                # Sample and apply weights
                swag_model.sample(scale=0.5)
                # The weights are loaded into the original router module that SWAG wraps
        
        _, probs, _ = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
        all_probs.append(probs)
    
    # Average probabilities and calculate metrics
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