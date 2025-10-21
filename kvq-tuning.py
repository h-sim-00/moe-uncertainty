import argparse
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from utils import setup_environment
from model import load_peft_model, load_tokenizer
from utils import load_and_prepare_train_and_val_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on a specific MMLU subject.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the MMLU subject to fine-tune on.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def train(model, tokenizer,
          train_dataset, val_dataset,
          model_name, dataset_name, args,
          early_stopping_patience=2,
          save_adapter_path="./adapters"):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    import wandb
    project_name = "kvq-finetuning"
    # Updated run name format
    run_name = f"{model_name}-{dataset_name}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
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
        save_total_limit=1, # Only keep the best model
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    print("--- Starting fine-tuning ---")
    trainer.train()
    print("--- Fine-tuning complete ---")

    # Updated final save path format
    final_save_path = f"{save_adapter_path}/{run_name}"
    print(f"Saving the best adapter weights to {final_save_path}")
    model.save_pretrained(final_save_path)

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    print(f"Loading PEFT model: {args.model_shortcode}")
    model = load_peft_model(args.model_shortcode, finetune_mode="qkv", device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)

    # Use the new argument to load a single dataset
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    train(
        model, tokenizer,
        train_dataset, val_dataset,
        model_name=args.model_shortcode,
        dataset_name=args.dataset_shortcode,
        args=args,
        early_stopping_patience=2
    )

if __name__ == "__main__":
    main()