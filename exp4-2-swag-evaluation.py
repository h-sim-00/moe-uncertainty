# filename: exp4-2-swag-evaluation.py

from utils import setup_environment
setup_environment()

import os
import argparse
import wandb
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torchmetrics.classification import MulticlassCalibrationError
from typing import Dict

# Local imports
from data import multiple_choice_prompt_engineer, load_classification_dataset

class SWAGModel:
    """A helper class to manage the SWAG model parameters and sampling."""
    def __init__(self, base_model: PeftModel, swag_weights_list: list, max_rank=20):
        self.base_model = base_model
        self.device = base_model.device

        # Get the names of trainable parameters directly from the live model
        self.param_names = [name for name, p in base_model.named_parameters() if p.requires_grad]
        if not self.param_names:
            raise ValueError("SWAG initialisation failed: No trainable parameters found. Ensure the PEFT model was loaded with is_trainable=True.")
        print(f"SWAG initialised for parameters: {self.param_names}")

        # Initialize the mean tensor on the correct device
        self.mean = {name: torch.zeros_like(p.data) for name, p in base_model.named_parameters() if p.requires_grad}

        # Move weights to the correct device before operation
        for weights in swag_weights_list:
            for name in self.param_names:
                if name in weights:
                    self.mean[name] += weights[name].to(self.device)
        
        for name in self.param_names:
            self.mean[name] /= len(swag_weights_list)

        # Compute deviation columns for covariance
        dev_matrix = []
        for weights in swag_weights_list:
            dev = torch.cat([(weights[name].to(self.device).flatten() - self.mean[name].flatten()) for name in self.param_names])
            dev_matrix.append(dev)
        
        dev_matrix = torch.stack(dev_matrix, dim=1)

        # Low-rank SVD for covariance
        U, S, _ = torch.svd(dev_matrix)
        self.U = U[:, :max_rank].to(self.device)
        self.S_diag = S[:max_rank].to(self.device)
        self.num_snapshots = len(swag_weights_list)

    def sample(self, scale=0.5):
        # Sample from the SWAG posterior on the correct device
        z1 = torch.randn(self.U.shape[1], device=self.device)
        
        # Unbiased estimate of variance requires K-1
        cov_factor = (scale / (2 * (self.num_snapshots - 1))**0.5) if self.num_snapshots > 1 else scale
        cov_sample = cov_factor * self.S_diag * z1
        
        sampled_dev = self.U @ cov_sample
        
        # Reconstruct the sampled weights
        sampled_weights = {}
        current_pos = 0
        for name in self.param_names:
            p_shape = self.mean[name].shape
            p_size = self.mean[name].numel()
            dev_slice = sampled_dev[current_pos : current_pos + p_size].view(p_shape)
            sampled_weights[name] = self.mean[name] + dev_slice
            current_pos += p_size
            
        return sampled_weights

    def set_model_params(self, sampled_weights):
        # Load a set of sampled weights into the model
        with torch.no_grad():
            for name, p in self.base_model.named_parameters():
                if name in sampled_weights:
                    p.data.copy_(sampled_weights[name])

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with SWAG.")
    parser.add_argument("--param_num", type=str, default="3b", help="Model parameter size.")
    parser.add_argument("--swag_fit_dir", type=str, required=True, help="Directory containing swag_weights.pt and the final adapter.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to evaluate on (e.g., arc_easy or medmcqa).")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of SWAG samples.")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--max_rank", type=int, default=10, help="Max rank for SWAG covariance matrix.")
    parser.add_argument("--target_layer", type=int, required=True, help="The specific MoE layer that SWAG was applied to.")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Load Model, Tokenizer, and SWAG weights ---
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b" else "ibm-granite/granite-3.1-1b-a400m-instruct"
    
    # Load the base model first
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # --- FIX: Load the adapter in TRAINABLE mode ---
    # This ensures that the LoRA parameters have `requires_grad=True`, which is
    # necessary for the SWAGModel class to identify them.
    print(f"Loading adapter from {args.swag_fit_dir} in trainable mode...")
    peft_model = PeftModel.from_pretrained(base_model, args.swag_fit_dir, is_trainable=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device)
    peft_model.eval() # Set to eval mode for inference (this does not affect requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(args.swag_fit_dir)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    swag_weights_path = os.path.join(args.swag_fit_dir, 'swag_weights.pt')
    print(f"Loading SWAG weights from {swag_weights_path}")
    swag_weights_list = torch.load(swag_weights_path, map_location='cpu')

    # --- Initialize SWAG Model ---
    # Now peft_model has parameters with requires_grad=True, so this will succeed.
    swag_model = SWAGModel(peft_model, swag_weights_list, max_rank=args.max_rank)
    
    # --- Prepare Data ---
    _, _, test_raw = load_classification_dataset(args.dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_raw]
    questions = [x['question'] for x in test_dataset]
    true_answers = [x['answer'] for x in test_dataset]
    true_answer_ids = [tokenizer.convert_tokens_to_ids(ans) for ans in true_answers]

    # --- Configure WandB ---
    project_name = "phase2-evaluation"
    exp_name = f"eval_swag_layer_{args.target_layer}_on_{args.dataset_name}"
    wandb.init(project=project_name, name=exp_name, config=vars(args))

    # --- Run SWAG Inference ---
    all_final_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc=f"SWAG Inference on {args.dataset_name}"):
            batch_questions = questions[i:i+args.batch_size]
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            batch_swag_probs = []
            for _ in range(args.num_samples):
                sampled_weights = swag_model.sample()
                swag_model.set_model_params(sampled_weights)
                
                outputs = peft_model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                batch_swag_probs.append(probs)

            avg_probs = torch.stack(batch_swag_probs, dim=0).mean(dim=0)
            all_final_probs.append(avg_probs.cpu())
            
    final_probs_tensor = torch.cat(all_final_probs, dim=0)
    
    # --- Calculate and Log Metrics ---
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
        ece_metric = MulticlassCalibrationError(num_classes=peft_model.config.vocab_size, n_bins=n_bins, norm='l1')
        ece = ece_metric(final_probs_tensor, targets_tensor).item()
        metrics_to_log[f"ece_{n_bins}"] = ece

    print(f"\n--- Final Metrics for {args.dataset_name} ---")
    for key, value in metrics_to_log.items():
        print(f"{key}: {value:.4f}")
    
    wandb.log(metrics_to_log)
    wandb.finish()

if __name__ == "__main__":
    main()
