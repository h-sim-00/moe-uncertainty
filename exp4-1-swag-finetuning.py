# filename: exp2-4-swag-train.py
import os
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from tqdm import tqdm
import copy

# Local imports
from model import load_peft_model
from data import multiple_choice_prompt_engineer, load_classification_dataset, preprocess_mask_question_for_training
from utils import setup_environment

def parse_args():
    """Parses command-line arguments for SWAG training."""
    parser = argparse.ArgumentParser(description="Fine-tune a PEFT model and collect weights for SWAG on a target layer.")
    parser.add_argument("--param_num", type=str, default="3b", help="Model parameter size.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset for fine-tuning.")
    parser.add_argument("--output_dir_base", type=str, default="./swag_fits", help="Base directory to save the collected SWAG weights and adapter.")
    parser.add_argument("--warmup_epochs", type=int, default=15, help="Number of initial epochs to warm up.")
    parser.add_argument("--swag_epochs", type=int, default=10, help="Number of epochs for SWAG weight collection.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate for warm-up.")
    parser.add_argument("--swag_learning_rate", type=float, default=1e-5, help="Cyclical/high learning rate for SWAG phase.")
    
    # --- MODIFICATION: Added target_layer argument ---
    parser.add_argument("--target_layer", type=int, required=True, help="The specific MoE layer index to apply SWAG to.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()

    # --- Dynamically Set Paths and Names ---
    output_dir = os.path.join(args.output_dir_base, f"swag_layer_{args.target_layer}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Dependencies ---
    model_id = "ibm-granite/granite-3.1-3b-a800m-instruct" if args.param_num == "3b" else "ibm-granite/granite-3.1-1b-a400m-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare data
    train_raw, _, _ = load_classification_dataset(args.dataset_name)
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw]
    train_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    # --- MODIFICATION: Load PEFT model targeting the specific layer ---
    print(f"Applying LoRA to router of layer {args.target_layer} for SWAG training.")
    peft_model = load_peft_model(
        model_id,
        finetune_mode="router",
        r=64,
        lora_dropout=0.1,
        target_layer=args.target_layer # Ensure this is handled by your model loading function
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device)

    # --- Training Setup ---
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * (args.warmup_epochs + args.swag_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    project_name = "phase2_bayesian_finetuning"
    exp_name = f"swag_train_layer_{args.target_layer}"
    wandb.init(project=project_name, name=exp_name, config=vars(args))

    # --- Custom Training Loop for SWAG ---
    swag_weights_list = []
    
    for epoch in range(args.warmup_epochs + args.swag_epochs):
        peft_model.train()
        
        # Adjust learning rate for SWAG phase
        if epoch >= args.warmup_epochs:
            if epoch == args.warmup_epochs: # Print only on the first SWAG epoch
                 print(f"\n--- Starting SWAG Collection Epoch {epoch - args.warmup_epochs + 1}/{args.swag_epochs} ---")
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.swag_learning_rate
        else:
            if epoch == 0:
                print(f"\n--- Starting Warm-up Epoch {epoch + 1}/{args.warmup_epochs} ---")

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = peft_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Collect weight snapshots during the SWAG phase
        if epoch >= args.warmup_epochs:
            # We only care about the trainable LoRA parameters for the target layer
            trainable_params = {name: p.cpu().clone() for name, p in peft_model.named_parameters() if p.requires_grad}
            swag_weights_list.append(trainable_params)
            print(f"Collected weight snapshot #{len(swag_weights_list)}")

    # --- Save Collected Weights ---
    print(f"\nSaving {len(swag_weights_list)} SWAG weight snapshots to {output_dir}...")
    torch.save(swag_weights_list, os.path.join(output_dir, "swag_weights.pt"))
    
    # Save the final model adapter and tokenizer for loading the architecture during inference
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("SWAG weight collection finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
