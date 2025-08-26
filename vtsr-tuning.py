import argparse
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.vtsr import VariationalTemperatureRouter
from model.routers.base import MoERouter

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
    output_root_dir = f"./router_weights/vtsr_{args.temperature_mode}"
    run_name = f"vtsr_{args.temperature_mode}-{args.model_shortcode}-{args.dataset_shortcode}"
    output_dir = os.path.join(output_root_dir, run_name)

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load the pre-trained MAP routers into the model as a starting point
    print("--- Loading base MAP routers ---")
    causal_model = model.base_model.model.model
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    
    for i, layer in enumerate(causal_model.layers):
        map_router = MoERouter(config=causal_model.config)
        map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        map_router.load_weights(map_weights_path, device=model.device)
        layer.block_sparse_moe.router = map_router

    # 3. Perform the flexible "Lego Swap" for VTSRs
    print("--- Swapping in VariationalTemperatureRouters ---")
    for layer_idx in args.swap_layers:
        target_layer = causal_model.layers[layer_idx]
        
        new_router = VariationalTemperatureRouter(
            config=causal_model.config,
            existing_router=target_layer.block_sparse_moe.router,
            temperature_mode=args.temperature_mode
        )
        
        if layer_idx in args.load_layers:
            print(f"Loading pre-trained VTSR for layer {layer_idx}...")
            weights_path = os.path.join(output_dir, f"layer_{layer_idx}_weights.pt")
            new_router.load_weights(weights_path, device=model.device)
        
        target_layer.block_sparse_moe.router = new_router.to(model.device)

    # 4. Freeze all parameters, then unfreeze only the target training layers
    print("--- Setting trainable parameters ---")
    for param in model.parameters():
        param.requires_grad = False
    
    for layer_idx in args.train_layers:
        print(f"Unfreezing router in layer {layer_idx} for training.")
        for param in causal_model.layers[layer_idx].block_sparse_moe.router.temperature_net.parameters():
            param.requires_grad = True

    # 5. Load data and train
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
    
    # 6. Save the final weights for ALL swapped VTSR layers
    print("--- Saving final weights for all swapped VTSRs ---")
    for layer_idx in args.swap_layers:
        save_path = os.path.join(output_dir, f"layer_{layer_idx}_weights.pt")
        causal_model.layers[layer_idx].block_sparse_moe.router.save_weights(save_path)

if __name__ == "__main__":
    main()