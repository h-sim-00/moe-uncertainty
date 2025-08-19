import argparse
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.base import MoERouter

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the deterministic MoE router (MAP baseline).")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the single dataset to train on (e.g., 'obqa').")
    parser.add_argument("--base_adapter_path", type=str, required=True, help="Path to the pre-trained Stage 1 adapter.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # Define internal paths
    output_dir = "./router_weights/base"

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Freeze all parameters and perform the "Lego Swap"
    print("Freezing model and swapping in new MoERouter instances...")
    for param in model.parameters():
        param.requires_grad = False
    
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        new_router = MoERouter(config=causal_model.config, existing_router=layer.block_sparse_moe.router)
        new_router.to(model.device)
        layer.block_sparse_moe.router = new_router
        for param in new_router.parameters():
            param.requires_grad = True

    # 3. Load data using the specified dataset_shortcode
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])

    # 4. Set up Trainer and run fine-tuning
    project_name = "router-finetuning"
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
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
    
    print("--- Starting MAP Router Fine-tuning ---")
    trainer.train()
    print("--- MAP Fine-tuning complete ---")
    
    # 5. Save the final router weights using our new API
    print("--- Saving final MAP router weights ---")
    for i, layer in enumerate(causal_model.layers):
        save_path = os.path.join(output_dir, run_name, f"layer_{i}_weights.pt")
        layer.block_sparse_moe.router.save_weights(save_path)

if __name__ == "__main__":
    main()