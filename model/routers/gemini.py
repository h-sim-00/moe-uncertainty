# model/routers/swag.py

import torch
import torch.nn as nn
from torch.optim.swa_utils import SWAG
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import wandb
import os
from tqdm import tqdm

from ...utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll


# --- Component 1: The Standard Router Class ---
class GraniteMoeDeterministicRouter(nn.Module):
    """A standard, non-stochastic router."""
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
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


# --- Component 2: Helper for Model Preparation ---
def _prepare_model_for_swag(model, map_weights_path):
    """
    Performs the 'Lego swap', loads pre-trained MAP weights, and prepares
    the model for router-only SWAG training.
    """
    print("Freezing all model parameters and swapping in deterministic routers...")
    for param in model.parameters():
        param.requires_grad = False
    
    device = model.device
    causal_model = model.base_model.model
    config = causal_model.config

    for layer in causal_model.layers:
        new_router = GraniteMoeDeterministicRouter(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        ).to(device)
        layer.block_sparse_moe.router = new_router

    print(f"Loading pre-trained MAP router weights from {map_weights_path}...")
    map_state_dicts = torch.load(map_weights_path, map_location=device)
    for i, layer in enumerate(causal_model.layers):
        layer.block_sparse_moe.router.load_state_dict(map_state_dicts[f"layer_{i}"])

    print("Unfreezing all new router parameters for training...")
    for layer in causal_model.layers:
        for param in layer.block_sparse_moe.router.parameters():
            param.requires_grad = True
            
    return model


# --- Component 3: The SWAG Training Function ---
def train_swag_router(model, tokenizer, train_dataset, args):
    """
    Fine-tunes the router and collects SWAG statistics using a custom loop.
    """
    # 1. Prepare model by loading MAP weights into new deterministic routers
    model = _prepare_model_for_swag(model, args.map_weights_path)
    
    # 2. Initialize SWAG objects for each router
    print("Initializing SWAG for each router...")
    causal_model = model.base_model.model
    swag_objects = {}
    trainable_params = []
    for i, layer in enumerate(causal_model.layers):
        router = layer.block_sparse_moe.router
        trainable_params.extend(router.parameters())
        swag_objects[f"layer_{i}"] = SWAG(router, swag_lr=args.swa_lr, max_num_models=20)

    # 3. Setup optimizer and data loader for SWAG phase
    optimizer = torch.optim.SGD(trainable_params, lr=args.swa_lr, weight_decay=args.weight_decay)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)

    project_name = "bayesian-router-finetuning"
    run_name = f"SWAG_{args.model_shortcode}_seed-{args.seed}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    # 4. Custom training loop for SWAG collection
    print("--- Starting SWAG collection phase ---")
    for epoch in range(args.swa_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"SWAG Epoch {epoch+1}/{args.swa_epochs}"):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss # End-to-end loss
            loss.backward()
            optimizer.step()
        
        # Update SWAG statistics at the end of each epoch
        print(f"Epoch {epoch+1}: Updating SWAG statistics...")
        for swag_model in swag_objects.values():
            swag_model.update_parameters(swag_model.base_model)
    
    print("--- SWAG collection complete ---")
    
    # 5. Update BN statistics for all SWAG models
    print("Updating Batch Norm statistics for SWAG models...")
    for swag_model in swag_objects.values():
        swag_model.update_bn(train_loader)
        
    # 6. Save the fitted SWAG objects
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "swag_routers.pt")
    print(f"Saving the fitted SWAG objects to {final_save_path}")
    torch.save(swag_objects, final_save_path)


# --- Component 4: The Evaluation Function ---
def evaluate_swag_router(model, tokenizer, swag_objects, dataset, dataset_name, num_samples=10, batch_size=8):
    """Orchestrates model evaluation using Monte Carlo sampling from SWAG."""
    print(f"--- Evaluating on {dataset_name} with {num_samples} SWAG samples ---")
    model.eval()
    all_probs = []
    
    # get_model_predictions is assumed to be in the utils file
    _, _, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    for i in tqdm(range(num_samples), desc="SWAG MC Samples"):
        # For each sample, draw and apply new weights for all routers
        with torch.no_grad():
            for swag_model in swag_objects.values():
                # This call modifies the underlying router weights in-place
                swag_model.sample(scale=0.5)
        
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