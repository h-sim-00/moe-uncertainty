# filename: exp2-3-laplace-train-and-fit.py
import os
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from laplace import Laplace
from tqdm import tqdm
import numpy as np
from torch.nn.utils import parameters_to_vector

# Local imports
from model import load_peft_model
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune for MAP and fit Laplace Approximation on a target layer.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for fine-tuning and fitting.")
    parser.add_argument("--output_dir_base", type=str, default="./laplace_fits", help="Base directory to save the final Laplace object.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    # --- MODIFICATION: Added target_layer argument ---
    parser.add_argument("--target_layer", type=int, required=True, help="The specific MoE layer index to apply Laplace to.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    
    # --- DYNAMICALLY SET PATHS AND NAMES ---
    # The output will be a single file, named after the layer
    output_filename = f"laplace_layer_{args.target_layer}.pt"
    adapter_save_dir = f"./peft_adapters/laplace_map_layer_{args.target_layer}"
    final_laplace_output_path = os.path.join(args.output_dir_base, output_filename)
    os.makedirs(args.output_dir_base, exist_ok=True)
    os.makedirs(adapter_save_dir, exist_ok=True)


    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b" else "ibm-granite/granite-3.1-1b-a400m-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data once for both training and fitting
    train_raw, val_raw, _ = load_classification_dataset(args.dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_raw]
    
    train_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)
    eval_dataset = preprocess_mask_question_for_training(val_engineered, tokenizer)

    # =================================================================================
    #                           PART 1: MAP FINE-TUNING
    # =================================================================================
    
    print(f"--- Starting Part 1: MAP Fine-tuning for Layer {args.target_layer} ---")

    # --- MODIFICATION: Pass target_layer to load_peft_model ---
    # Your load_peft_model function MUST be modified to handle this, as we discussed.
    peft_model = load_peft_model(
        model_id,
        finetune_mode="router",
        r=64,
        lora_dropout=0.0, # No dropout for MAP estimate
        target_layer=args.target_layer
    )
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    peft_model.print_trainable_parameters()
    
    project_name = f"phase2_bayesian_finetuning"
    exp_name = f"laplace_train_fit_layer_{args.target_layer}"
    wandb.init(project=project_name, name=exp_name, config=vars(args), reinit=True)

    training_args = TrainingArguments(
        output_dir=adapter_save_dir, # Temporary directory for trainer checkpoints
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=10,
        fp16=True, # Use fp16 for faster training
        report_to="wandb",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    print(f"MAP fine-tuning for layer {args.target_layer} complete.")
    
    # Save the MAP adapter and tokenizer for future reference and for the evaluation script
    peft_model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    print(f"MAP adapter for layer {args.target_layer} saved to {adapter_save_dir}")

    # =================================================================================
    #                      PART 2: LAPLACE FITTING
    # =================================================================================
     
    print(f"\n--- Starting Part 2: Laplace Fitting for Layer {args.target_layer} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device).to(torch.float32) # Ensure model is on GPU and in float32 for stable fitting
    
    # Use a subset of training data for fitting the Hessian
    fit_dataset = preprocess_mask_question_for_training(train_engineered[:200], tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    fit_loader = DataLoader(fit_dataset, batch_size=1, collate_fn=data_collator)
    
    # Initialize Laplace on the already trained model in memory
    la = Laplace(peft_model, 'classification',
                 subset_of_weights='all', # Use 'all' as PEFT ensures only LoRA params are trainable
                 hessian_structure='diag')
    
    # Sanity check the number of parameters Laplace has identified
    if not la.params or la.n_params > 10000000:
        raise ValueError(f"Incorrect number of trainable parameters found: {la.n_params}. Check PEFT setup.")
    print(f"Laplace correctly initialized with {la.n_params} parameters.")
    
    # --- Start of Corrected Fitting Logic ---
    # Replace la.fit() with the robust manual loop
    print("Starting manual fitting loop to compute the Hessian...")
    peft_model.train() # Set to train mode for gradients
    for batch in tqdm(fit_loader, desc="Fitting Hessian"):
        batch = {k: v.to(device) for k, v in batch.items()}
        peft_model.zero_grad()
        # The ** is crucial for unpacking the dictionary into keyword arguments
        outputs = peft_model(**batch) 
        loss = outputs.loss
        loss.backward()
    print("Hessian fitting complete.")
    
    # Manually set the mean of the Laplace approximation to the MAP estimate
    la.mean = parameters_to_vector(la.params).detach()
    
    # Manually optimize the prior precision using grid search
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
    
    # --- End of Corrected Fitting Logic ---
    
    # Save the final fitted Laplace object
    print(f"Saving fitted Laplace object to {final_laplace_output_path}...")
    torch.save(la, final_laplace_output_path)
    print("Laplace model saved successfully.")
    wandb.finish()
if __name__ == "__main__":
    main()