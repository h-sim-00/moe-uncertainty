"""
Stage 1: deterministic MAP adaptation with LoRA (paper Appendix D.2).

The paper applies LoRA to "the attention modules (Q/K/V projections) and the
Expert networks". `--finetune_mode qkv_experts` (the default on this branch)
does both: PEFT adapts Q/K/V, and model.expert_lora adapts every expert matrix
of every MoE layer. `--finetune_mode qkv` reproduces the original Q/K/V-only
behaviour.
"""

import argparse
import wandb
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
from utils import setup_environment
from model import load_peft_model, load_tokenizer
from model.expert_lora import save_expert_lora, expert_lora_path
from utils import load_and_prepare_train_and_val_data, is_generation_dataset

def train(model, tokenizer, train_loader, val_loader, args):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    project_name = "moe-uncertainty"
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    if args.adapter_suffix:
        run_name = f"{run_name}_{args.adapter_suffix}"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)
    
    num_training_batches = len(train_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Paper D.2: "All models were trained using the AdamW optimiser", lr with
    # cosine decay and warmup_ratio warmup.
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    total_optim_steps = num_training_batches * args.epochs
    warmup_steps = int(args.warmup_ratio * total_optim_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )
    print(f"--- Optim: AdamW lr={args.lr} cosine warmup={warmup_steps}/{total_optim_steps} steps ---")

    final_save_path = f"./adapters/{args.model_shortcode}-{args.dataset_shortcode}"
    if args.adapter_suffix:
        final_save_path = f"{final_save_path}-{args.adapter_suffix}"

    def save_adapter():
        model.save_pretrained(final_save_path)
        # PEFT only serialises the Q/K/V adapters; the expert-LoRA factors live in
        # a custom module and are written next to them inside the same directory.
        if args.finetune_mode == "qkv_experts":
            save_expert_lora(model, expert_lora_path(final_save_path))

    # Best-val checkpointing + early stopping on val loss (mirrors fcvr-tuning.py):
    # the adapter on disk is always the best epoch, never the (overfit) last one.
    best_val_loss = float("inf")
    epochs_no_improve = 0

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
            scheduler.step()
            total_epoch_loss += loss.item()
            wandb.log({
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
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

        # Best-val checkpointing: only persist the adapter when validation
        # improves (re-save overwrites the previous best, so final_save_path is
        # always the best epoch); early-stop after `early_stop_patience` epochs
        # without improvement.
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"  New best val loss {best_val_loss:.4f} -> saving adapter to {final_save_path}")
            save_adapter()
        else:
            epochs_no_improve += 1
            print(f"  No val-loss improvement ({epochs_no_improve}/{args.early_stop_patience}); "
                  f"keeping best checkpoint (val {best_val_loss:.4f}).")
            if epochs_no_improve >= args.early_stop_patience:
                print(f"--- Early stopping at epoch {epoch+1} (best val loss {best_val_loss:.4f}) ---")
                break

    # Safety net: if val loss never improved (best checkpoint never written),
    # persist the final state so the adapter dir is not empty.
    if best_val_loss == float("inf"):
        print(f"--- Val loss never improved; saving final state to {final_save_path} as a fallback ---")
        save_adapter()

    print(f"--- MAP Fine-tuning complete (best val loss {best_val_loss:.4f}); best adapter at {final_save_path} ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on a specific MMLU subject.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the MMLU subject to fine-tune on.")
    parser.add_argument("--finetune_mode", type=str, default="qkv_experts", choices=["qkv", "qkv_experts"],
                        help="'qkv_experts' (paper D.2): LoRA on Q/K/V AND the expert networks. 'qkv': Q/K/V only.")
    parser.add_argument("--expert_lora_r", type=int, default=64,
                        help="LoRA rank for the expert matrices (the paper does not state a rank; 64 matches the Q/K/V rank used here). Lower it if the run OOMs.")
    parser.add_argument("--adapter_suffix", type=str, default=None,
                        help="Suffix on ./adapters/<model>-<dataset>; keeps this run from overwriting an existing adapter.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total optimizer steps used for LR warmup (paper D.2: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Epochs of no val-loss improvement before early stopping (best checkpoint kept).")
    return parser.parse_args()


def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda:0"

    model = load_peft_model(
        args.model_shortcode,
        finetune_mode=args.finetune_mode,
        expert_lora_r=args.expert_lora_r,
        device_map=device
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # Answer-only loss for MCQA (`answer_only=True`) and explanation-only loss for
    # generation datasets (prompt-masked labels + EOS, see utils/data.py): in both
    # cases labels mask the prompt (-100), so the collator must preserve them
    # instead of rebuilding labels from input_ids. Right padding keeps real-token
    # positions correct during training (eval keeps left).
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode], answer_only=True)
    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    print(f"--- Loss mode: {'explanation-only (generation)' if is_generation_dataset([args.dataset_shortcode]) else 'answer-only (MCQA)'} ---")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    train(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()