import argparse
import os
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.mfvr import MeanFieldVariationalRouter, train_mfvr_router
from model.routers.base import MoERouter

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an MoE router with Mean-Field VI on the logit space.")
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True)
    parser.add_argument("--base_adapter_path", type=str, required=True)
    
    parser.add_argument("--swap_layers", type=int, nargs='+', required=True)
    parser.add_argument("--load_layers", type=int, nargs='*', default=[]) 
    parser.add_argument("--train_layers", type=int, nargs='+', required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    output_root_dir = "./router_weights/mfvr"
    run_name = f"mfvr-{args.model_shortcode}-{args.dataset_shortcode}"
    
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 1. Load base MAP routers
    print("--- Loading base MAP routers ---")
    causal_model = model.base_model.model.model
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    
    for i, layer in enumerate(causal_model.layers):
        map_router = MoERouter(config=causal_model.config)
        map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        map_router.load_weights(map_weights_path, device=model.device)
        layer.block_sparse_moe.router = map_router

    # 2. Perform flexible "Lego Swap" for MFVR
    print("--- Swapping in MeanFieldVariationalRouters ---")
    for layer_idx in args.swap_layers:
        target_layer = causal_model.layers[layer_idx]
        new_router = MeanFieldVariationalRouter(
            config=causal_model.config,
            existing_router=target_layer.block_sparse_moe.router
        )
        if layer_idx in args.load_layers:
            print(f"Loading pre-trained MFVR for layer {layer_idx}...")
            weights_path = os.path.join(output_root_dir, run_name, f"layer_{layer_idx}_weights.pt")
            new_router.load_weights(weights_path, device=model.device)
        
        target_layer.block_sparse_moe.router = new_router.to(model.device)

    # 3. Freeze/Unfreeze parameters
    print("--- Setting trainable parameters ---")
    for param in model.parameters():
        param.requires_grad = False
    
    for layer_idx in args.train_layers:
        print(f"Unfreezing router in layer {layer_idx} for training.")
        for param in causal_model.layers[layer_idx].block_sparse_moe.router.parameters():
            if param.requires_grad: # Only unfreeze trainable parts
                param.requires_grad = True

    # 4. Load data and start training
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    train_mfvr_router(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()