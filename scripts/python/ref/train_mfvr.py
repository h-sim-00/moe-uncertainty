# train_mfvi_logit_router.py

import argparse
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
from model.routers.mfvr import add_mfvi_logit_routers_to_model, train_mfvi_logit_router

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an MoE router with Logit-Space MFVI.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--base_adapter_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the custom optimizer.")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for the KL divergence term in the ELBO loss.")
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
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    model = add_mfvi_logit_routers_to_model(model)

    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    print(f"Train loader size: {len(train_loader)}, Validation loader size: {len(val_loader)}")

    train_mfvi_logit_router(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()