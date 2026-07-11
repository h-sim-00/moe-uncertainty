import argparse
import os, torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
import wandb
from tqdm import tqdm

from model.adapters import granite_adapter, qwen_adapter, deepseek_adapter

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data

ADAPTER_MAP = {
    "granite": {
        "load": granite_adapter.load_granite_map_routers,
        "prepare": granite_adapter.prepare_granite_bayesian_routers,
        "save": granite_adapter.save_granite_bayesian_routers,
    },
    "qwen": {
        "load": qwen_adapter.load_qwen_map_routers,
        "prepare": qwen_adapter.prepare_qwen_bayesian_routers,
        "save": qwen_adapter.save_qwen_bayesian_routers,
    },
    "deepseek": {
        "load": deepseek_adapter.load_deepseek_map_routers,
        "prepare": deepseek_adapter.prepare_deepseek_bayesian_routers,
        "save": deepseek_adapter.save_deepseek_bayesian_routers,
    },
}

def train(model, tokenizer, train_loader, val_loader, args):
    run_name = f"vtsr-{args.model_shortcode}-{args.dataset_shortcode}"

    # === 0: Make sure we're using the currect swap, save functions ===
    adapter = ADAPTER_MAP[args.model_shortcode]
    load_map_routers = adapter["load"]
    prepare_bayesian_routers = adapter["prepare"]
    save_bayesian_routers = adapter["save"]

    # === 1. Prepare Model for Training ===
    model = load_map_routers(model, args=args)
    model = prepare_bayesian_routers(model, method="vtsr", args=args)
    causal_model = model.base_model.model.model

    # === 2. Create Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # === 3. Run Custom Training Loop ===
    project_name = "moe-uncertainty"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    num_training_batches = len(train_loader)
    
    print("--- Starting VTSR Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            # 1. Get the standard language modeling loss
            reconstruction_loss = outputs.loss
            
            # 2. Calculate the temperature penalty
            total_temp_penalty = torch.tensor(0.0, device=model.device)
            for layer_idx in args.train_layers:
                if args.model_shortcode == "granite":
                    router = causal_model.layers[layer_idx].block_sparse_moe.router
                elif args.model_shortcode == "qwen":
                    router = causal_model.layers[layer_idx].mlp.router
                elif args.model_shortcode == "deepseek":
                    router = causal_model.layers[layer_idx].mlp.router
                temperature = router.last_temperature
                penalty = -torch.log(temperature).sum()
                total_temp_penalty += penalty
                
            # 3. Combine the losses
            final_loss = reconstruction_loss + args.temp_penalty_weight * total_temp_penalty

            final_loss.backward()
            optimizer.step()
            total_epoch_loss += final_loss.item()
            wandb.log({
                "train_loss": final_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "temp_penalty": total_temp_penalty.item(),
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

    print("--- VTSR Fine-tuning complete ---")

    save_bayesian_routers(model, method="vtsr", args=args)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the Variational Temperature Router (VTSR) layer by layer.")
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True)
    parser.add_argument("--base_adapter_path", type=str, required=True)
    
    # New arguments for layer-wise control
    parser.add_argument("--swap_layers", type=int, nargs='+', required=True, help="All layers that should be VTSRs.")
    parser.add_argument("--load_layers", type=int, nargs='*', default=[])     
    parser.add_argument("--train_layers", type=int, nargs='+', required=True, help="Subset of swap_layers to unfreeze and train.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--temp_penalty_weight", type=float, default=1e-3, help="Weight for the -log(T) penalty term to prevent temperature collapse.")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="cuda:1"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load data
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    # 3. Call the dedicated training function
    train(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()