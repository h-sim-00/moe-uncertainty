# model/routers/laplace.py

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from laplace import Laplace
import wandb
import os
from tqdm import tqdm

# Import the user-specified metric calculation utilities
from ...utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll


# --- Component 1: The Standard Router Class ---
# For Laplace, we start with a standard, deterministic router.
class GraniteMoeDeterministicRouter(nn.Module):
    """A standard, non-stochastic router to find the MAP estimate."""
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
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


# --- Component 2: Function to Modify and Train the Router ---
def train_and_fit_laplace(model, tokenizer, train_dataset, val_dataset, train_loader_for_laplace, args):
    """
    Full pipeline: Swaps in deterministic routers, trains them to find the MAP estimate,
    and then fits the Laplace approximation.
    """
    # 1. Freeze all model parameters and swap in the deterministic routers
    print("Freezing all model parameters and swapping in deterministic routers...")
    for param in model.parameters():
        param.requires_grad = False
    
    config = model.model.config
    device = model.device
    for layer in model.model.layers:
        new_router = GraniteMoeDeterministicRouter(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        ).to(device)
        layer.block_sparse_moe.router = new_router
        for param in layer.block_sparse_moe.router.parameters():
            param.requires_grad = True

    # 2. Train the routers to find the MAP estimate using the Trainer
    project_name = "bayesian-router-finetuning"
    run_name = f"Laplace-MAP_{args.model_shortcode}_seed-{args.seed}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    training_args = TrainingArguments(
        output_dir=f"./intermediate_checkpoints/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    print("--- Starting MAP Fine-tuning for Laplace ---")
    trainer.train()
    print("--- MAP Fine-tuning complete ---")
    
    # 3. Fit the Laplace approximation on each router
    print("--- Fitting Laplace Approximation for each router ---")
    laplace_approximations = {}
    for i, layer in enumerate(model.model.layers):
        print(f"Fitting Laplace for router in layer {i}...")
        # We specify which part of the model to make Bayesian
        la = Laplace(layer.block_sparse_moe.router, 'classification',
                     subset_of_weights='all',
                     hessian_structure='kfac')
        # Fit the approximation using the training data
        la.fit(train_loader_for_laplace)
        # Optional: Tune the prior precision on a validation set
        la.optimize_prior_precision(method='marglik')
        laplace_approximations[f"layer_{i}"] = la
    
    # 4. Save the fitted Laplace objects
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "laplace_routers.pkl")
    print(f"Saving the fitted Laplace objects to {final_save_path}")
    torch.save(laplace_approximations, final_save_path)


# --- Component 3: The Evaluation Function ---
def evaluate_laplace_router(model, tokenizer, laplace_objects, dataset, dataset_name, num_samples=10, batch_size=8):
    """Orchestrates model evaluation using Monte Carlo sampling from the Laplace approximation."""
    print(f"--- Evaluating on {dataset_name} with {num_samples} Laplace samples ---")
    
    model.eval()
    all_probs = []

    # Get the base predictions and labels once
    _, _, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    # Perform N Monte Carlo forward passes
    for i in tqdm(range(num_samples), desc="Laplace MC Samples"):
        # For each sample, draw new weights for all routers
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.model.layers):
                la = laplace_objects[f"layer_{layer_idx}"]
                # Sample and apply weights in-place
                la.sample_and_apply()

        # Get predictions with the new sampled router weights
        _, probs, _ = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
        all_probs.append(probs)
    
    # Restore the original MAP weights after sampling is done
    for la in laplace_objects.values():
        la.load_state_dict(la.map_state_dict)

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