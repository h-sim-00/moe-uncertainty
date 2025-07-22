# filename: exp2-finetuning-laplace-map.py
import os
import argparse
import wandb
import torch
import transformers
import numpy as np
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Tuple, List, Dict

# Assuming 'model.py' and 'data.py' are in the same directory or accessible
from model import load_peft_model
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Granite MoE model with LoRA to get a MAP estimate for Laplace.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size.")
    parser.add_argument("--finetune_mode", type=str, default="router", choices=["router", "qkv"], help="LoRA target modules.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="./peft_adapters/laplace_map_router", help="Directory to save the trained LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    return parser.parse_args()

def load_and_prepare_data(tokenizer: AutoTokenizer, dataset_name: str) -> Tuple[Dataset, Dataset]:
    train_raw, val_raw, _ = load_classification_dataset(dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_raw]
    train_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)
    eval_dataset = preprocess_mask_question_for_training(val_engineered, tokenizer)
    return train_dataset, eval_dataset

def compute_metrics(eval_pred: transformers.EvalPrediction) -> Dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    labels, preds = labels.flatten(), preds.flatten()
    mask = labels != -100
    labels, preds = labels[mask], preds[mask]
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def main():
    args = parse_args()
    setup_environment()
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b"\
                else "ibm-granite/granite-3.1-1b-a400m-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = load_and_prepare_data(tokenizer, args.dataset_name)
    
    # Load model with lora_dropout=0.0 to get a deterministic model for MAP
    peft_model = load_peft_model(model_id, args.finetune_mode, r=64, lora_dropout=0.0)
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    peft_model.print_trainable_parameters()

    project_name = f"exp3-1_laplace_finetuning_epoch_{args.epochs}"
    exp_name = f"{args.param_num}_{args.finetune_mode}_map_{args.dataset_name}"
    wandb.init(project=project_name, name=exp_name, config=vars(args))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        fp16=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # compute_metrics=compute_metrics,
    )

    print(f"Starting MAP fine-tuning for experiment: {exp_name}")
    trainer.train()

    print(f"Training complete. Saving final adapter to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
