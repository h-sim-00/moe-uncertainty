# train_swag_router.py

import argparse
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.swag import train_swag_router

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a router with SWAG.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--epochs", type=int, default=5, help="Initial training epochs before SWAG.")
    parser.add_argument("--swa_epochs", type=int, default=5, help="Epochs for SWAG weight collection.")
    parser.add_argument("--swa_lr", type=float, default=0.01, help="Learning rate for SWAG phase.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)
    
    # 2. Load data for Trainer and a separate DataLoader for SWAG
    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    
    # 3. Call the dedicated training and fitting function
    train_swag_router(
        model, tokenizer, train_dataset, val_dataset, args
    )

if __name__ == "__main__":
    main()