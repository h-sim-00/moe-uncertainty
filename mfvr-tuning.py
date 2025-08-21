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

    # 1. Load the base model and attach the Stage 1 adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load data
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    # 3. Call the dedicated training function
    # It now handles all the model setup and training internally.
    train_mfvr_router(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()