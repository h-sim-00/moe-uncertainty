# filename: exp2-4-laplace-evaluation.py
import os
import argparse
import wandb
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils import vector_to_parameters
from torchmetrics.classification import MulticlassCalibrationError
from typing import Dict

# Local imports
from data import multiple_choice_prompt_engineer, load_classification_dataset
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for Laplace inference."""
    parser = argparse.ArgumentParser(description="Run inference with a fitted Laplace model.")
    parser.add_argument("--laplace_path", type=str, required=True, help="Path to the saved/fitted Laplace object.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the corresponding MAP adapter model directory (for tokenizer).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to use for evaluation (e.g., arc_easy, medmcqa).")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of Monte Carlo samples per input.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()

    # --- Load Dependencies ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading fitted Laplace object from: {args.laplace_path} onto {device}...")
    
    # Important: weights_only=False allows loading the model object and its state
    la = torch.load(args.laplace_path, map_location=device, weights_only=False)
    model = la.model
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and Laplace object loaded successfully.")

    # --- Prepare Data ---
    _, _, test_raw = load_classification_dataset(args.dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_raw]
    questions = [x['question'] for x in test_dataset]
    true_answers = [x['answer'] for x in test_dataset]
    true_answer_ids = [tokenizer.convert_tokens_to_ids(ans) for ans in true_answers]
    
    # --- Configure WandB ---
    project_name = "phase2-evaluation"
    laplace_filename = os.path.basename(args.laplace_path)
    exp_name = f"{os.path.splitext(laplace_filename)[0]}_on_{args.dataset_name}"
    wandb.init(project=project_name, name=exp_name, config=vars(args))

    # --- Run Manual Laplace Inference ---
    all_final_probs = []
    
    # Identify the parameters that Laplace was fitted on
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the loaded model for Laplace.")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc=f"Laplace Inference on {args.dataset_name}"):
            batch_questions = questions[i:i+args.batch_size]
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Sample all N weight vectors at once for the batch
            sampled_param_vectors = la.sample(n_samples=args.num_samples)

            batch_mc_probs = []
            # Loop through the collected weight samples
            for single_sampled_vector in sampled_param_vectors:
                # Load one sampled weight vector into the model
                vector_to_parameters(single_sampled_vector, trainable_params)
                
                # Perform a standard forward pass with the new weights
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                batch_mc_probs.append(probs)

            # Average the probabilities across all MC samples for the batch
            avg_probs = torch.stack(batch_mc_probs, dim=0).mean(dim=0)
            all_final_probs.append(avg_probs.cpu())

    final_probs_tensor = torch.cat(all_final_probs, dim=0)
    
    # --- Calculate Metrics ---
    pred_indices = torch.argmax(final_probs_tensor, dim=-1)
    pred_ids = pred_indices.tolist()
    
    correct_count = sum([p_id == t_id for p_id, t_id in zip(pred_ids, true_answer_ids)])
    accuracy = correct_count / len(true_answer_ids)
    
    targets_tensor = torch.tensor(true_answer_ids)
    nlls = -torch.log(final_probs_tensor[torch.arange(len(targets_tensor)), targets_tensor] + 1e-9)
    nll_avg, nll_std = nlls.mean().item(), nlls.std().item()
    
    entropies = -torch.sum(final_probs_tensor * torch.log(final_probs_tensor + 1e-9), dim=-1)
    ent_avg, ent_std = entropies.mean().item(), entropies.std().item()
    
    metrics_to_log = {"accuracy": accuracy, "nll_avg": nll_avg, "nll_std": nll_std, "entropy_avg": ent_avg, "entropy_std": ent_std}
    
    for n_bins in [10, 15, 20]:
        ece_metric = MulticlassCalibrationError(num_classes=model.config.vocab_size, n_bins=n_bins, norm='l1')
        ece = ece_metric(final_probs_tensor, targets_tensor).item()
        metrics_to_log[f"ece_{n_bins}"] = ece

    print(f"\n--- Final Metrics for {args.dataset_name} ---")
    for key, value in metrics_to_log.items():
        print(f"{key}: {value:.4f}")
    
    wandb.log(metrics_to_log)
    wandb.finish()
    print("Inference and evaluation complete.")

if __name__ == "__main__":
    main()
