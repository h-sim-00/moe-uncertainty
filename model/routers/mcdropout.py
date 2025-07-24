import torch
from torch import nn
from ...utils import (get_model_predictions,
                      calculate_accuracy, calculate_ece_mce, calculate_nll)

class GraniteMoeMCDropoutRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int, dropout_rate: float = 0.1):
        """
        Initialize the gating mechanism.
        Args:
            input_size (`int`): Size of the input.
            num_experts (`int`): Number of experts.
            top_k (`int`): Number of top experts to select in "top_k" mode.
        """
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        """
        Forward method for the gating mechanism.
        Args:
            hidden_states (`torch.Tensor`): Input hidden states of shape [batch_size * seq_len, input_size].
        Returns:
            index_sorted_experts (`torch.Tensor`): Indices of selected experts, sorted for efficient processing.
            batch_index (`torch.Tensor`): Batch index for grouped inputs.
            batch_gates (`torch.Tensor`): Gate values for grouped inputs.
            expert_size (`List[int]`): Number of tokens assigned to each expert.
            logits (`torch.Tensor`): Original (pre-dropout) logits from the gating layer.
        """

        stochastic_hidden_states = self.dropout(hidden_states)
        logits = self.layer(stochastic_hidden_states).float()

        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        batch_size = hidden_states.shape[0]
        zeros = torch.zeros(
            (batch_size, self.num_experts), dtype=torch.long, device=logits.device
        )
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()
        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]

        return index_sorted_experts, batch_index, batch_gates, expert_size, logits

def add_mcdropout_routers_to_model(model, dropout_rate):
    """
    Performs the 'Lego swap' to replace original routers with MCDropoutRouters
    and prepares the model for router-only training.
    """
    print("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    device = model.device

    print("Swapping original routers with MCDropoutRouters...")
    for layer in model.model.layers:
        old_router = layer.block_sparse_moe.router
        new_router = GraniteMoeMCDropoutRouter(
            input_size=old_router.input_size,
            num_experts=old_router.num_experts,
            top_k=old_router.top_k,
            dropout_rate=dropout_rate
        )
        new_router.layer.load_state_dict(old_router.layer.state_dict())
        new_router.to(device)
        layer.block_sparse_moe.router = new_router

    print("Unfreezing all new router parameters for training...")
    for layer in model.model.layers:
        for param in layer.block_sparse_moe.router.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Successfully modified model. Trainable parameters: {trainable_params}")
    
    return model

# --- Component 3: The Training Function ---
def train_router(model, tokenizer, train_dataset, val_dataset, args):
    """
    Takes a prepared model and runs the Hugging Face Trainer.
    args: model_shortcode, seed, epochs, batch_size
    """
    project_name = "bayesian-router-finetuning"
    run_name = f"mcdropout_ft_{args.model_shortcode}_dor-{args.dropout_rate}_seed-{args.seed}"

    import wandb
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
    training_args = TrainingArguments(
        output_dir=f"./intermediate_checkpoints/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        report_to="wandb",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("--- Starting Router Fine-tuning (End-to-End) ---")
    trainer.train()
    print("--- Router Fine-tuning complete ---")

    final_router_states = {
        f"layer_{i}": layer.block_sparse_moe.router.state_dict()
        for i, layer in enumerate(model.model.layers)
    }
    
    import os
    output_dir = f"./adapters/" + run_name
    os.makedirs(output_dir, exist_ok=True)
    final_save_path = os.path.join(output_dir, f"router_weights.pt")
    
    print(f"Saving the final router weights to {final_save_path}")
    torch.save(final_router_states, final_save_path)

# --- Component 4: The Router Weights Loader ---
def load_router_weights(model, router_weights_path):
    """
    Loads the router weights from the specified path into the model.
    """

    print(f"Loading router weights from {router_weights_path}")
    router_state_dicts = torch.load(router_weights_path, map_location=model.device)
    
    for i, layer in enumerate(model.model.layers):
        state_dict = router_state_dicts[f"layer_{i}"]
        layer.block_sparse_moe.router.load_state_dict(state_dict)
    
    print("Router weights loaded successfully.")
    model.eval()
    return model

# --- Component 5: The Evaluation Function ---
def evaluate_router(model, tokenizer, dataset, dataset_name, num_samples=10, batch_size=8):
    """
    Orchestrates model evaluation using Monte Carlo Dropout.
    """
    print(f"--- Evaluating on {dataset_name} with {num_samples} MC samples ---")
    
    model.eval()
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            layer.block_sparse_moe.router.dropout.train()

    from tqdm import tqdm
    all_probs = []
    for i in tqdm(range(num_samples), desc="MC Samples"):
        # This call now works because the function is imported in this file.
        _, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
        all_probs.append(probs)
    
    stacked_probs = torch.stack(all_probs)
    mean_probs = stacked_probs.mean(dim=0)
    final_preds = torch.argmax(mean_probs, dim=1)

    acc = calculate_accuracy(final_preds, labels)
    nll = calculate_nll(mean_probs, labels)
    ece, mce = calculate_ece_mce(mean_probs, labels)
    
    results = {
        'dataset': dataset_name,
        'ACC': acc.item(),
        'NLL': nll.item(),
        'ECE': ece.item(),
        'MCE': mce.item(),
    }
    print(results)
    return results