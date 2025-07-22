# filename: exp2-3b-laplace-fit.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel
from laplace import Laplace
from tqdm import tqdm
import numpy as np
from torch.nn.utils import parameters_to_vector

# Local imports
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the Laplace fitting process."""
    parser = argparse.ArgumentParser(description="Fit Laplace Approximation to a fine-tuned PEFT model.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the pre-trained PEFT adapter (the MAP estimate).")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for fitting the Hessian.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final fitted Laplace object.")
    return parser.parse_args()

def main():
    """Main function to load a MAP model and fit the Laplace approximation."""
    args = parse_args()
    setup_environment()

    # --- Load Model and Adapter ---
    # This robust loading process is critical for the Laplace library to work correctly.
    from transformers import AutoModelForCausalLM

    # 1. Load the base model to CPU first to avoid device map issues.
    print("Loading base model to CPU...")
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b" else "ibm-granite/granite-3.1-1b-a400m-instruct"
    # Load with float32 for stable fitting.
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # 2. Apply the pre-trained PEFT adapter to the CPU model.
    print(f"Loading PEFT adapter from {args.adapter_path} in TRAINABLE mode.")
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)

    # 3. Move the entire, correctly configured PEFT model to the target device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving the PEFT model to {device}...")
    model.to(device)
    model.eval() # Set to eval mode before passing to Laplace
    print("Model successfully loaded and moved.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- Prepare Data for Fitting ---
    # Use a subset of the training data to fit the Hessian.
    train_raw, _, _ = load_classification_dataset(args.dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw][:200]
    fit_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    fit_loader = DataLoader(fit_dataset, batch_size=1, collate_fn=data_collator)

    # --- Fit Laplace Approximation ---
    print("Initializing Laplace...")
    # Initialize Laplace on the model. 'subset_of_weights'='all' is correct because
    # PEFT ensures that only the LoRA parameters have requires_grad=True.
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='diag')
    
    # Sanity check to ensure Laplace has found the correct LoRA parameters.
    if not la.params or la.n_params > 10000000: # Adjust sanity check number if needed
        raise ValueError(f"Incorrect number of trainable parameters found: {la.n_params}. Check PEFT setup.")
    print(f"Laplace correctly initialized with {la.n_params} parameters.")

    # Manually fit the Hessian using the known-working procedure.
    print("Starting manual fitting loop to compute the Hessian...")
    model.train() # Hessian calculation requires gradients
    for batch in tqdm(fit_loader, desc="Fitting Hessian"):
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
    
    # Save the final fitted Laplace object
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving fitted Laplace object to {args.output_file}...")
    torch.save(la, args.output_file)
    print("Laplace model saved successfully.")

if __name__ == "__main__":
    main()
