import argparse
import wandb
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from utils import setup_environment
from model import load_peft_model, load_tokenizer
from utils import load_and_prepare_train_and_val_data

def train(model, tokenizer, train_loader, val_loader, args):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    project_name = "kvq-finetuning"
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    num_training_batches = len(train_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

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


    # Updated final save path format
    final_save_path = f"./adapters/{run_name}"
    print(f"Saving the best adapter weights to {final_save_path}")
    model.save_pretrained(final_save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on a specific MMLU subject.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the MMLU subject to fine-tune on.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    device = "cuda:0"

    model = load_peft_model(
        args.model_shortcode, 
        finetune_mode="qkv", 
        device_map=device
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # Use the new argument to load a single dataset
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    train(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()