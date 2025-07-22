import os
import argparse
import wandb
import torch
import transformers
import numpy as np
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from typing import Tuple, List, Dict

# Assuming 'model.py' and 'data.py' are in the same directory or accessible
from model import load_peft_model
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

def parse_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Granite MoE model with LoRA for MC Dropout.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size.")
    parser.add_argument("--finetune_mode", type=str, default="router", choices=["router", "qkv"], help="LoRA target modules.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for fine-tuning.")
    parser.add_argument("--output_dir_base", type=str, default="./peft_adapters", help="Base directory to save the trained LoRA adapters.")
    
    # New arguments for hyperparameter sweeping
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate for AdamW.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers.")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension (rank).")
    parser.add_argument("--router_dropout_rate", type=float, default=0.1, help="Dropout probability for the gating mechanism's internal dropout layer.")
    
    parser.add_argument("--target_layer", type=int, default=None, help="The specific MoE layer index to apply the intervention to. If None, applies globally.")

    return parser.parse_args()


def load_and_prepare_data(tokenizer: AutoTokenizer, dataset_name: str) -> Tuple[Dataset, Dataset]:
    """Loads and preprocesses the dataset for causal language modeling."""
    train_raw, val_raw, _ = load_classification_dataset(dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_raw]

    train_dataset = preprocess_mask_question_for_training(train_engineered + val_engineered, tokenizer)
    eval_dataset = preprocess_mask_question_for_training(val_engineered, tokenizer)

    print(f"Dataset '{dataset_name}' loaded and preprocessed.")
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset[:200]

def compute_metrics(eval_pred: transformers.EvalPrediction) -> Dict[str, float]:
    """
    Computes accuracy on a per-token basis for the evaluation set.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    labels = labels.flatten()
    preds = preds.flatten()
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def main():
    """Main function to orchestrate the fine-tuning process."""
    args = parse_args()
    setup_environment()

    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b"\
                else "ibm-granite/granite-3.1-1b-a400m-instruct"

    # Load dependencies
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, _ = load_and_prepare_data(tokenizer, args.dataset_name)

    # Pass LoRA hyperparameters to the model creation function
    peft_model = load_peft_model(
        model_id,
        args.finetune_mode,
        r=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_layer=args.target_layer,
    )

    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    peft_model.print_trainable_parameters()

    # --- Set up the model for MC Dropout training ---
    # print("Setting model router mode to 'mc_dropout_stochastic_logits' for training...")
    # peft_model.change_router_mode("mc_dropout_stochastic_logits")
    # # Set the router dropout rate based on the new hyperparameter
    # print(f"Setting router dropout rate to: {args.router_dropout_rate}")
    # peft_model.change_router_dropout_rate(args.router_dropout_rate)

    # --- Set up the model for MC Dropout training ---
    print(f"Setting router mode for layer {args.target_layer} to 'mc_dropout_stochastic_logits'...")
    peft_model.change_router_mode_for_layers(
        mode="mc_dropout_stochastic_logits", 
        layer_indices=[args.target_layer] # Pass the target layer
    )

    print(f"Setting router dropout rate for all layers to: {args.router_dropout_rate}")
    peft_model.change_router_dropout_rate(
        args.router_dropout_rate,
    )

    # Configure training run name and output directory dynamically
    project_name = f"phase2_bayesian_finetuning"
    exp_name = f"mcdropout_{args.param_num}_{args.finetune_mode}_lr_{args.learning_rate}_ldrop_{args.lora_dropout}_rdrop_{args.router_dropout_rate}_r_{args.lora_r}_target_layer_{args.target_layer}"
    output_dir = os.path.join(args.output_dir_base, exp_name)

    wandb.init(project=project_name, name=exp_name, config=vars(args))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True, 
        learning_rate=args.learning_rate, # Use argument
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fp16=True,
        report_to="wandb",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Starting fine-tuning for experiment: {exp_name}")
    trainer.train()

    print(f"Training complete. Saving final adapter to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Adapter and tokenizer saved successfully.")
    wandb.finish()

if __name__ == "__main__":
    main()
