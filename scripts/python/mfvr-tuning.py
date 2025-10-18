import argparse
import os, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data
import wandb

from model.adapters.granite_adapter import load_granite_map_routers, prepare_granite_bayesian_routers, save_granite_bayesian_routers


def train_mfvr_router(model, tokenizer, train_loader, val_loader, args):
    """Custom training loop for the MeanFieldVariationalRouter using the ELBO loss."""
    
    run_name = f"mfvr-{args.model_shortcode}-{args.dataset_shortcode}"

    # === 1. Prepare Model for Training ===
    # Freeze all parameters in the entire model first
    print("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    model = load_granite_map_routers(model, args=args)
    model = prepare_granite_bayesian_routers(model, method="mfvr", args=args)
    causal_model = model.base_model.model.model

    # === 2. Create Optimizer ===
    # Get the list of all currently trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # === 3. Run Custom Training Loop ===
    project_name = "bayesian-router-finetuning"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    num_training_batches = len(train_loader)
    
    print("--- Starting MFVR Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            reconstruction_loss = outputs.loss
            
            total_kl_div = 0
            for layer_idx in args.train_layers:
                router = causal_model.layers[layer_idx].block_sparse_moe.router
                total_kl_div += router.kl_divergence()
            
            kl_term = (args.beta / num_training_batches) * total_kl_div
            loss = reconstruction_loss + kl_term
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            
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

    print("--- MFVR Fine-tuning complete ---")

    save_granite_bayesian_routers(model, method="mfvr", args=args)

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
    train_mfvr_router(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()