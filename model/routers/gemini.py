# model/routers/mcdropout.py

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb
import os
from tqdm import tqdm

# --- CORRECTED IMPORT LOCATION ---
# We import the necessary evaluation functions here, where they are used.
# The relative path '...' assumes 'utils.py' is in the root directory.
from ...utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

# ... (GraniteMoeMCDropoutRouter class, add_mcdropout_routers_to_model, train_router) ...


# --- Evaluation Function ---
def evaluate_mcdropout_router(model, tokenizer, dataset, dataset_name, num_samples=10):
    """
    Orchestrates model evaluation using Monte Carlo Dropout.
    """
    print(f"--- Evaluating on {dataset_name} with {num_samples} MC samples ---")
    
    model.eval()
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            layer.block_sparse_moe.router.dropout.train()

    all_probs = []
    for i in tqdm(range(num_samples), desc="MC Samples"):
        # This call now works because the function is imported in this file.
        _, probs, labels = get_model_predictions(model, tokenizer, dataset)
        all_probs.append(probs)
    
    # ... (rest of the function is the same) ...
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
        'MCE': mce.item()
    }
    print(results)
    return results