import argparse
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
# Import the new, modular functions
from model.routers.mcdropout import add_mcdropout_routers_to_model, train_router

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an MoE router with MC Dropout.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    print(f"Loading base model '{args.model_shortcode}' and attaching adapter from '{base_adapter_path}'")
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Modify the model for Bayesian training
    model = add_mcdropout_routers_to_model(model, args.dropout_rate)

    # 3. Load data
    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # 4. Call the dedicated training function with the now-modified model
    train_router(model, tokenizer, train_dataset, val_dataset, args)

if __name__ == "__main__":
    main()