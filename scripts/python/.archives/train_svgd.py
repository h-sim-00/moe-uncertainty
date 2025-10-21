# train_svgd_router.py

import argparse
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.svgd import train_svgd_router

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an MoE router with SVGD.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the SVGD optimizer.")
    parser.add_argument("--num_particles", type=int, default=20, help="Number of particles for SVGD.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization strength (prior strength).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load data and create a DataLoader
    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, _ = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    
    print(f"Train loader size: {len(train_loader)}")

    # 3. Call the dedicated training function
    train_svgd_router(model, tokenizer, train_loader, args)

if __name__ == "__main__":
    main()