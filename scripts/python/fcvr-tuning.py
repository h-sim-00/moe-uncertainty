import argparse
import math
import os, torch, wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup

from model.adapters import granite_adapter, qwen_adapter, deepseek_adapter

from utils import setup_environment, seed_everything
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_and_prepare_train_and_val_data, is_generation_dataset

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

def train_fcvr_router(model, tokenizer, train_loader, val_loader, args):
    """Custom training loop for the FCVR using the ELBO loss."""
    run_name = f"fcvr-{args.model_shortcode}-{args.dataset_shortcode}"
    if getattr(args, "run_suffix", None):
        run_name = f"{run_name}-{args.run_suffix}"

    # === 0: Make sure we're using the currect swap, save functions ===
    adapter = ADAPTER_MAP[args.model_shortcode]
    load_map_routers = adapter["load"]
    prepare_bayesian_routers = adapter["prepare"]
    save_bayesian_routers = adapter["save"]

    # === 1. Prepare Model for Training ===
    # Seed the FCVR prior mean (mean_base) either from the fine-tuned MAP
    # routers (inherited pipeline) or from the pre-trained Granite routers
    # (paper-faithful: paper freezes the pre-trained Wr and never MAP-tunes it).
    if args.prior_source == "map":
        model = load_map_routers(model, args=args)
    else:
        print("--- prior_source=pretrained: skipping MAP load; FCVR mean_base seeds from the pre-trained Granite router ---")
    model = prepare_bayesian_routers(model, method="fcvr", args=args)
    causal_model = model.base_model.model.model

    # === 2. Create Optimizer + Cosine Schedule (paper D.2) ===
    # AdamW + cosine decay with warmup_ratio warmup. Gradient accumulation lifts
    # the per-device batch to the paper's effective batch of 16.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    num_training_batches = len(train_loader)
    grad_accum = max(1, args.grad_accum_steps)
    steps_per_epoch = math.ceil(num_training_batches / grad_accum)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_optim_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )
    print(f"--- Optim: AdamW lr={args.lr} cosine warmup={warmup_steps}/{total_optim_steps} steps "
          f"| per-device batch={args.batch_size} x grad_accum={grad_accum} = eff batch {args.batch_size * grad_accum} ---")

    # === 3. Run Custom Training Loop ===
    project_name = "moe-uncertainty"
    wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    # Early stopping on validation NLL (the LM reconstruction loss).
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("--- Starting FCVR Fine-tuning (Custom Loop) ---")
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)

            reconstruction_loss = outputs.loss

            # KL position mask (supervisor point 4): 'attention' = real tokens only
            # (padding excluded -- bug fix, default); 'answer' = target tokens only
            # (weighting ablation); 'none' = every position incl. pads (legacy).
            if args.kl_mask == "attention":
                kl_mask = inputs["attention_mask"].reshape(-1)
            elif args.kl_mask == "answer":
                kl_mask = (inputs["labels"] != -100).reshape(-1)
            else:
                kl_mask = None
            total_kl_div = 0
            for layer_idx in args.train_layers:
                if args.model_shortcode == "granite":
                    router = causal_model.layers[layer_idx].block_sparse_moe.router
                elif args.model_shortcode == "qwen":
                    router = causal_model.layers[layer_idx].mlp.router
                elif args.model_shortcode == "deepseek":
                    router = causal_model.layers[layer_idx].mlp.router
                total_kl_div += router.kl_divergence(mask=kl_mask)

            # Paper-faithful ELBO weighting: loss = L_task + beta * sum_layers KL_layer,
            # where each KL_layer is a per-token mean (see fcvr.py kl_divergence).
            # No /num_training_batches: that was a global-latent recipe misapplied to
            # this amortised per-token latent, and it made the effective beta depend
            # on batch/dataset size.
            kl_term = args.beta * total_kl_div
            loss = reconstruction_loss + kl_term

            # Scale for gradient accumulation, then step every grad_accum
            # micro-batches (and on the final micro-batch of the epoch).
            (loss / grad_accum).backward()
            is_step = ((i + 1) % grad_accum == 0) or ((i + 1) == num_training_batches)
            if is_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_epoch_loss += loss.item()
            wandb.log({
                "train_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "kl_term": kl_term.item(),
                "kl_tokens_in_batch": int(kl_mask.sum().item()) if kl_mask is not None else int(inputs["input_ids"].numel()),
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
        print(f"Epoch {epoch+1} validation loss (NLL): {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # Early stopping on val NLL: keep the best checkpoint on disk.
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"  New best val NLL {best_val_loss:.4f} -> saving FCVR weights.")
            save_bayesian_routers(model, method="fcvr", args=args)
        else:
            epochs_no_improve += 1
            print(f"  No val-NLL improvement ({epochs_no_improve}/{args.early_stop_patience}).")
            if epochs_no_improve >= args.early_stop_patience:
                print(f"--- Early stopping at epoch {epoch+1} (best val NLL {best_val_loss:.4f}) ---")
                break

    # Safety net: if val NLL never improved (best checkpoint never written),
    # persist the final state so the weights dir is not empty.
    if best_val_loss == float("inf"):
        print("--- Val NLL never improved; saving final state as a fallback ---")
        save_bayesian_routers(model, method="fcvr", args=args)

    print(f"--- FCVR Fine-tuning complete (best val NLL {best_val_loss:.4f}) ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an MoE router with Full-Covariance VI.")
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True)
    parser.add_argument("--base_adapter_path", type=str, required=True)
    
    parser.add_argument("--swap_layers", type=int, nargs='+', required=True)
    parser.add_argument("--load_layers", type=int, nargs='*', default=[]) 
    parser.add_argument("--train_layers", type=int, nargs='+', required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device micro-batch; paper uses {2,4,8} with grad-accum to eff batch 16.")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps; batch_size * grad_accum_steps = effective batch (paper: 16).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total optimizer steps used for LR warmup (paper D.2: 0.05).")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Stop after this many epochs without val-NLL improvement (paper: early stop on val NLL).")
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--kl_mask", type=str, default="attention", choices=["none", "attention", "answer"],
                        help="Positions the per-token KL is averaged over: attention = real tokens only "
                             "(padding excluded; default), answer = target tokens only (ablation), "
                             "none = every position incl. padding (legacy; what the existing weights used).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_suffix", type=str, default=None,
                        help="Optional suffix on the FCVR weights dir to avoid overwriting other runs.")
    parser.add_argument("--map_suffix", type=str, default=None,
                        help="[prior_source=map] Suffix of the MAP router weights dir (router_weights/base/<model>_<dataset>-<suffix>).")
    parser.add_argument("--prior_source", type=str, default="map", choices=["map", "pretrained"],
                        help="Seed FCVR mean_base from fine-tuned MAP routers ('map') or the pre-trained Granite router ('pretrained', paper-faithful).")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    seed_everything(args.seed)  # random (dataset shuffles/splits) + numpy + torch

    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # Answer-only loss for MCQA (`answer_only=True`) / explanation-only loss for
    # generation datasets (prompt-masked labels + EOS, see utils/data.py): labels
    # mask the prompt (-100), so the collator must preserve them instead of
    # rebuilding labels from input_ids (DataCollatorForLanguageModeling would train
    # on the prompt too). Seq2Seq pads input_ids with pad_token and labels with
    # -100 dynamically per batch; right padding keeps real-token positions correct
    # during training (eval keeps left).
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode], seed=args.seed, answer_only=True)
    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    print(f"--- Loss mode: {'explanation-only (generation)' if is_generation_dataset([args.dataset_shortcode]) else 'answer-only (MCQA)'} ---")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    train_fcvr_router(model, tokenizer, train_loader, val_loader, args)

if __name__ == "__main__":
    main()