import argparse
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from utils import setup_environment
from model import load_peft_model, load_tokenizer
from utils import load_and_prepare_train_and_val_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on MMLU subjects.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training and evaluation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def train(model, tokenizer, 
          train_dataset, val_dataset, model_name, args=None,
          epochs=5, batch_size=4, early_stopping_patience=2, 
          save_adapter_path="./adapters", seed=42):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    import wandb
    project_name = "finetuning"
    wandb.init(project=project_name, name=f"{model_name}-kvq_ft-seed_{seed}", config=vars(args), reinit=True)
    
    training_args = TrainingArguments(
        output_dir=f"./intermediate_checkpoints/kvq_ft_{model_name}_seed-{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        report_to="wandb", # Tell the Trainer to log to wandb
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=seed,
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

    final_save_path = f"{save_adapter_path}/kvq_ft_{model_name}_seed-{seed}"
    print(f"Saving the best adapter weights to {final_save_path}")
    model.save_pretrained(final_save_path)

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
    train(
        model, tokenizer,
        train_dataset, val_dataset,
        model_name=args.model_shortcode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=2,
        save_adapter_path="./adapters",
        seed=args.seed,
        args=args
    )

if __name__ == "__main__":
    main()