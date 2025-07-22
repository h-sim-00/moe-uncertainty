# filename: exp3-id-ood-inference.py
import os
import argparse
import wandb
import torch
import transformers
import numpy as np
from tqdm import tqdm
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torchmetrics.classification import MulticlassCalibrationError
from typing import Tuple, List, Dict

# Assuming 'model.py' and 'data.py' are in the same directory or accessible
from model import load_peft_model_and_adapter
from data import multiple_choice_prompt_engineer, load_classification_dataset
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run MC Dropout inference on a fine-tuned MoE model.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained PEFT adapter directory.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for evaluation.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of Monte Carlo samples per input.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--router_dropout_rate", type=float, default=0.1, help="Dropout probability for the gating mechanism's internal dropout layer.")
    parser.add_argument("--target_layer", type=int, required=True, help="The specific MoE layer index where the intervention was applied during training.")
    return parser.parse_args()

def main():
    """Main function to orchestrate the MC Dropout inference and evaluation process."""
    args = parse_args()
    setup_environment()

    # --- Load Dependencies ---
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b"\
                else "ibm-granite/granite-3.1-1b-a400m-instruct"

    model = load_peft_model_and_adapter(model_id, args.adapter_path)
    # Set the router mode for stochastic inference
    model.change_router_mode_for_layers(
        mode="mc_dropout_stochastic_logits", 
        layer_indices=[args.target_layer] 
    )
    model.change_router_dropout_rate(args.router_dropout_rate)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prepare Data ---
    _, _, test_raw = load_classification_dataset(args.dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_raw] 
    questions = [x['question'] for x in test_dataset]
    true_answers = [x['answer'] for x in test_dataset]
    true_answer_ids = [tokenizer.convert_tokens_to_ids(ans) for ans in true_answers]
    
    # --- Configure WandB ---
    project_name = "phase2-evaluation"
    adapter_name = os.path.basename(os.path.normpath(args.adapter_path))
    exp_name = f"mcdropout_{adapter_name}_on_{args.dataset_name}_mcdropout_samples_{args.num_samples}"
    wandb.init(project=project_name, name=exp_name, config=vars(args))

    # --- Run MC Dropout Inference ---
    all_final_probs = []
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc=f"MC Inference on {args.dataset_name}"):
            batch_questions = questions[i:i+args.batch_size]
            
            inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            batch_mc_probs = []
            for _ in range(args.num_samples):
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                batch_mc_probs.append(probs)

            avg_probs = torch.stack(batch_mc_probs, dim=0).mean(dim=0)
            all_final_probs.append(avg_probs.cpu())

    final_probs_tensor = torch.cat(all_final_probs, dim=0)
    
    # --- Calculate Metrics ---
    pred_indices = torch.argmax(final_probs_tensor, dim=-1)
    preds = [tokenizer.decode(idx) for idx in pred_indices.tolist()]
    
    accuracy = sum([p == t for p, t in zip(preds, true_answers)]) / len(true_answers)
    
    nlls = (-torch.log(final_probs_tensor[torch.arange(len(true_answer_ids)), true_answer_ids])).tolist()
    nll_avg = np.mean(nlls)
    nll_std = np.std(nlls)
    
    # Add a small epsilon to prevent log(0)
    entropies = (-torch.sum(final_probs_tensor * torch.log(final_probs_tensor + 1e-9), dim=-1)).tolist()
    ent_avg = np.mean(entropies)
    ent_std = np.std(entropies)
    
    pseudo_logits = torch.log(final_probs_tensor + 1e-9)
    targets_tensor = torch.tensor(true_answer_ids)
    
    metrics_to_log = {
        "accuracy": accuracy,
        "nll_avg": nll_avg,
        "nll_std": nll_std,
        "entropy_avg": ent_avg,
        "entropy_std": ent_std,
    }
    
    for n_bins in [10, 15, 20]:
        ece_metric = MulticlassCalibrationError(num_classes=model.config.vocab_size, n_bins=n_bins, norm='l1')
        ece = ece_metric(pseudo_logits, targets_tensor).item()
        metrics_to_log[f"ece_{n_bins}"] = ece

    print(f"\n--- Final Metrics for {args.dataset_name} ---")
    for key, value in metrics_to_log.items():
        print(f"{key}: {value:.4f}")
    
    wandb.log(metrics_to_log)
    wandb.finish()
    print("Inference and evaluation complete.")

if __name__ == "__main__":
    main()
