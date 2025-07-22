# filename: exp3-laplace-fit.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from laplace import Laplace
from tqdm import tqdm
import numpy as np
from torch.nn.utils import parameters_to_vector

# Local imports
from model import load_peft_model_and_adapter
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

setup_environment()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Laplace Approximation to a fine-tuned PEFT model.")
    parser.add_argument("--param_num", type=str, default="3b", help="Model parameter size.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained PEFT adapter (MAP estimate).")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for fitting the Hessian.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the fitted Laplace object.")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- THE FIX: A robust, sequential loading process ---
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # 1. Load the base model to CPU. This avoids accelerate's device_map.
    print("Loading base model to CPU...")
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b" else "ibm-granite/granite-3.1-1b-a400m-instruct"
    # Loading with float32 for stability during fitting
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # 2. Apply the PEFT adapter to the CPU model.
    print(f"Loading PEFT adapter from {args.adapter_path} in TRAINABLE mode.")
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)

    # 3. Now move the entire, correctly configured PEFT model to the GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving the PEFT model to {device}...")
    model.to(device)
    print("Model successfully moved to GPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_raw, _, _ = load_classification_dataset(args.dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw][:100]
    train_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)

    # --- Fit Laplace Approximation ---
    # 4. Initialize Laplace with the clean, single-device model.
    print("Initializing Laplace...")
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='diag')
    
    # This check should now pass because la.params was set correctly on init.
    if not la.params or la.n_params > 10000000: # Sanity check for LoRA params
        raise ValueError(f"Incorrect number of trainable parameters found: {la.n_params}. Loading failed.")
    print(f"Laplace correctly initialized with {la.n_params} parameters.")

    # --- The rest of the script, which we know works ---
    print("Starting manual fitting loop to compute the Hessian...")
    model.train()
    for batch in tqdm(train_loader, desc="Fitting Hessian"):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

    print("Hessian fitting complete.")
    
    la.mean = parameters_to_vector(la.params).detach()

    print("Optimizing prior precision with manual grid search...")
    prec_candidates = np.logspace(-4, 2, 20)
    best_prior_prec, best_log_marglik = None, -np.inf

    for prior_prec_scalar in tqdm(prec_candidates, desc="Grid Searching Prior Precision"):
        log_marglik = la.log_marginal_likelihood(prior_precision=prior_prec_scalar)
        if log_marglik > best_log_marglik:
            best_log_marglik = log_marglik
            best_prior_prec = prior_prec_scalar
            
    print(f"Found best prior precision: {best_prior_prec:.4f} (log-marglik: {best_log_marglik:.2f})")
    la.prior_precision = best_prior_prec

    model.eval()

    print(f"Saving fitted Laplace object to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(la, args.output_file)
    print("Laplace model saved successfully.")


if __name__ == "__main__":
    main()
