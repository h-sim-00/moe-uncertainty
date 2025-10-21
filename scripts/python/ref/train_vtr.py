# train_vtr.py

import argparse
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.archives.vtr import VariationalTemperatureRouter

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the Variational Temperature Router (VTR).")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--base_adapter_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./models/routers")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # Freeze all parameters and perform the "Lego Swap"
    print("Freezing model and swapping in new VariationalTemperatureRouter instances...")
    for param in model.parameters():
        param.requires_grad = False
    
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        new_router = VariationalTemperatureRouter(
            config=causal_model.config,
            existing_router=layer.block_sparse_moe.router
        )
        new_router.to(model.device)
        layer.block_sparse_moe.router = new_router
        # Unfreeze only the parameters of the new temperature network
        for param in new_router.temperature_net.parameters():
            param.requires_grad = True

    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)

    project_name = "bayesian-router-finetuning"
    run_name = f"VTR_{args.model_shortcode}_seed-{args.seed}"
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
    
    print("--- Starting VTR Fine-tuning ---")
    trainer.train()
    print("--- Fine-tuning complete ---")
    
    print("--- Saving final VTR weights ---")
    save_dir = os.path.join(args.output_dir, run_name)
    for i, layer in enumerate(causal_model.layers):
        save_path = os.path.join(save_dir, f"layer_{i}_weights.pt")
        layer.block_sparse_moe.router.save_weights(save_path)

if __name__ == "__main__":
    main()