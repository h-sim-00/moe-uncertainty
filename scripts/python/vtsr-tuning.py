import argparse
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb

from model.adapters.granite_adapter import load_granite_map_routers, prepare_granite_bayesian_routers, save_granite_bayesian_routers
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the Variational Temperature Router (VTSR) layer by layer.")
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True)
    parser.add_argument("--base_adapter_path", type=str, required=True)
    parser.add_argument("--temperature_mode", type=str, required=True, choices=['per_expert', 'shared'])
    
    # New arguments for layer-wise control
    parser.add_argument("--swap_layers", type=int, nargs='+', required=True, help="All layers that should be VTSRs.")
    parser.add_argument("--load_layers", type=int, nargs='*', default=[])     
    parser.add_argument("--train_layers", type=int, nargs='+', required=True, help="Subset of swap_layers to unfreeze and train.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # Define internal paths and run name
    run_name = f"vtsr_{args.temperature_mode}-{args.model_shortcode}-{args.dataset_shortcode}"

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load the pre-trained MAP routers into the model as a starting point
    model = load_granite_map_routers(model, args=args)

    # 3. Perform the flexible "Lego Swap" for VTSRs
    model = prepare_granite_bayesian_routers(model, method="vtsr", args=args)

    # 4. Load data and train
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])

    project_name = "bayesian-router-finetuning"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    training_args = TrainingArguments(
        output_dir=f"./intermediate_checkpoints/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print(f"--- Starting Fine-tuning for VTSR layers: {args.train_layers} ---")
    trainer.train()
    print("--- Fine-tuning complete ---")
    
    # 5. Save the final weights for ALL swapped VTSR layers
    save_granite_bayesian_routers(model, method="vtsr", args=args)

if __name__ == "__main__":
    main()