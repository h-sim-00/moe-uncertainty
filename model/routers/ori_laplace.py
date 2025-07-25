import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from laplace import Laplace
import wandb
import os
from tqdm import tqdm

# Import the user-specified metric calculation utilities
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

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

# --- Component 2: Function to Modify and Train the Router ---
def train_and_fit_laplace(model, tokenizer, train_dataset, val_dataset, args):
    """
    Full pipeline: Swaps in deterministic routers, trains them to find the MAP estimate,
    and then fits the Laplace approximation.
    """
    # 1. Freeze all model parameters and swap in the deterministic routers
    print("Freezing all model parameters and swapping in deterministic routers...")
    for param in model.parameters():
        param.requires_grad = False
    
    device = model.device
    causal_model = model.base_model.model

    for layer in causal_model.model.layers:
        old_router = layer.block_sparse_moe.router
        new_router = GraniteMoeDeterministicRouter(
            input_size=old_router.input_size,
            num_experts=old_router.num_experts,
            top_k=old_router.top_k,
        ).to(device)
        new_router.layer.load_state_dict(old_router.layer.state_dict())
        layer.block_sparse_moe.router = new_router
        for param in layer.block_sparse_moe.router.parameters():
            param.requires_grad = True

    # 2. Train the routers to find the MAP estimate using the Trainer
    project_name = "bayesian-router-finetuning"
    run_name = f"Laplace_{args.model_shortcode}_seed-{args.seed}"

    map_save_dir = os.path.join("./adapters", run_name)
    os.makedirs(map_save_dir, exist_ok=True)
    map_weights_path = os.path.join(map_save_dir, "map_router_weights.pt")

    if os.path.exists(map_weights_path):
        print(f"--- Found pre-trained MAP weights at {map_weights_path}. Loading... ---")
        map_state_dicts = torch.load(map_weights_path, map_location=device)
        for i, layer in enumerate(causal_model.model.layers):
            layer.block_sparse_moe.router.load_state_dict(map_state_dicts[f"layer_{i}"])
    else: 
        print("--- No pre-trained MAP weights found. Starting fine-tuning... ---")
        wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
        training_args = TrainingArguments(
            output_dir=f"./intermediate_checkpoints/{run_name}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            report_to="wandb",
            logging_steps=10,
            eval_strategy="epoch",
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

        print(f"--- Saving MAP router weights to {map_weights_path} ---")
        map_router_states = {
            f"layer_{i}": layer.block_sparse_moe.router.state_dict()
            for i, layer in enumerate(causal_model.model.layers)
        }
        torch.save(map_router_states, map_weights_path)

    # 3. Fit the Laplace approximation on each router using a CORRECT CUSTOM LOOP
    print("--- Fitting Laplace Approximation for each router ---")
    from torch.utils.data import DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader_for_laplace = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    # Instantiate all Laplace objects first
    laplace_approximations = {}
    for i, layer in enumerate(causal_model.model.layers):
        router_module = layer.block_sparse_moe.router
        la = Laplace(router_module, 'classification',
                     subset_of_weights='all',
                     hessian_structure='kron')
        laplace_approximations[f"layer_{i}"] = la

    # Manually iterate to accumulate the Hessian approximation for ALL routers
    model.eval() # Ensure the main model is in eval mode
    for batch in tqdm(train_loader_for_laplace, desc="Fitting Laplace on all Layers"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        for la in laplace_approximations.values():
            la.backward(loss)

    # Optional: Tune the prior precision for each router
    print("Optimizing prior precision for all routers...")
    for la in laplace_approximations.values():
        la.optimize_prior_precision(method='marglik')
    
    # 4. Save the fitted Laplace objects
    save_dir = os.path.join("./adapters", run_name)
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "laplace_routers.pkl")
    print(f"Saving the fitted Laplace objects to {final_save_path}")
    torch.save(laplace_approximations, final_save_path)

# --- Component 3: The Evaluation Function ---
def evaluate_laplace_router(model, tokenizer, laplace_objects, dataset, dataset_name, num_samples=10, batch_size=8):
    """Orchestrates model evaluation using Monte Carlo sampling from the Laplace approximation."""
    print(f"--- Evaluating on {dataset_name} with {num_samples} Laplace samples ---")
    
    model.eval()
    causal_model = model.base_model.model

    all_probs = []

    # Get the base predictions and labels once
    _, _, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    # Perform N Monte Carlo forward passes
    for i in tqdm(range(num_samples), desc="Laplace MC Samples"):
        # For each sample, draw new weights for all routers
        with torch.no_grad():
            for layer_idx, layer in enumerate(causal_model.model.layers):
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