import argparse
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import torch
import wandb

from model.adapters import granite_adapter, qwen_adapter, deepseek_adapter

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data

ADAPTER_MAP = {
    "granite": {
        "prepare": granite_adapter.swap_granite_moe_blocks,
        "save": granite_adapter.save_granite_map_routers,
    },
    "qwen": {
        "prepare": qwen_adapter.swap_qwen_moe_blocks,
        "save": qwen_adapter.save_qwen_map_routers,
    },
    "deepseek": {
        "prepare": deepseek_adapter.swap_deepseek_moe_blocks,
        "save": deepseek_adapter.save_deepseek_map_routers,
    }
}

def train(model, tokenizer, train_loader, val_loader, args):
    # === 0: Make sure we're using the currect swap, save functions ===
    adapter = ADAPTER_MAP[args.model_shortcode]
    prepare_for_tuning_func = adapter["prepare"]
    save_map_routers_func = adapter["save"]
    
    run_name = f"map-router-{args.model_shortcode}-{args.dataset_shortcode}"

    # === 1. Prepare Model for Training ===
    model = prepare_for_tuning_func(model)

    # === 2. Create Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # === 3. Run Custom Training Loop ===
    project_name = "moe-uncertainty"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    num_training_batches = len(train_loader)

    print("--- Starting MAP Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            wandb.log({
                "train_loss": loss.item()
            })
            
        print(f"Epoch {epoch+1} average training loss: {total_epoch_loss / num_training_batches:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

    print("--- MAP Fine-tuning complete ---")
    
    save_map_routers_func(model, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the deterministic MoE router (MAP baseline).")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the single dataset to train on (e.g., 'obqa').")
    parser.add_argument("--base_adapter_path", type=str, required=True, help="Path to the pre-trained Stage 1 adapter.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="cuda:1"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    train(model, tokenizer, train_loader, val_loader, args)


if __name__ == "__main__":
    main()