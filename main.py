from utils import setup_environment
import argparse
from model import load_peft_model, load_tokenizer
from utils import (load_classification_dataset, 
                   load_and_prepare_train_and_val_data,
                   get_model_predictions, 
                   calculate_accuracy, calculate_ece_mce, calculate_nll)
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="0-shot base-baseline setup.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Name of the model to use.")
    return parser.parse_args()


def evaluate(model, tokenizer, dataset, dataset_name):
    """Orchestrates model evaluation."""
    print(f"--- Evaluating on {dataset_name} ---")
    preds, probs, labels = get_model_predictions(model, tokenizer, dataset)

    # Calculate metrics
    acc = calculate_accuracy(preds, labels)
    nll = calculate_nll(probs, labels)
    ece, mce = calculate_ece_mce(probs, labels)
    
    results = {
        'dataset': dataset_name,
        'ACC': acc.item(),
        'NLL': nll.item(),
        'ECE': ece.item(),
        'MCE': mce.item()
    }
    print(results)
    return results

def train(model, tokenizer, 
          train_dataset, val_dataset, model_name, args=None,
          epochs=5, batch_size=4, early_stopping_patience=3, save_adapter_path="./adapters/new_finetuned_model"):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/checkpoints_{model_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}",
        logging_steps=10,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,# Load the best model found during training
        metric_for_best_model="eval_loss", # Use validation loss to determine the best model
        greater_is_better=False,    # Lower validation loss is better
        save_total_limit=2,          # Only keep the best and the most recent checkpoints
    )

    # Define the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # This is for Causal LM, not Masked LM
    )

    # Define the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # Save the best adapter
    print(f"Saving the best adapter weights to {save_adapter_path}")
    model.save_pretrained(save_adapter_path)


def main():
    # 1. Prepare the setup: wandb, transformers, parsers
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 2. Load the model and tokenizer, put on the peft jacket (KVQ LoRA)
    print(f"Loading PEFT model: {args.model_shortcode}")
    model = load_peft_model(args.model_shortcode, finetune_mode="qkv", device_map="auto")
    tokenizer = load_tokenizer(args.model_shortcode)

    # 3. Load the dataset and prepare for training
    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # 4. Train the model (Finetuning on ID dataset)
    output_adapter_path = f"./adapters/finetuned_{args.model_shortcode}"
    train(model, tokenizer, 
          train_dataset, val_dataset, 
          model_name=args.model_shortcode,
          epochs=5, batch_size=4, early_stopping_patience=2,
          save_adapter_path=output_adapter_path)

    # 5. Post-training evaluation
    print("\n--- Starting Post-Finetuning Evaluation ---")
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    for shortcode in dataset_shortcodes:
        print(f"\nLoading and processing dataset: {shortcode}")
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate(model, tokenizer, test_raw_dataset, dataset_name=shortcode)
        all_results.append(results)

    # 6. Save all results to a CSV
    output_csv_path = f"./results/finetuned_results_{args.model_shortcode}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved all evaluation results to {output_csv_path}")

if __name__ == "__main__":
    main()
    