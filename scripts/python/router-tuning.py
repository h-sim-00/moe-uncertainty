"""Stage 2a: deterministic MAP router tuning (all routers unfrozen, everything else frozen).

NOT part of the ICML paper's protocol (App. D.2 freezes the pre-trained router
weights W_r; the VGLR prior mean is l_det = u W_r). This stage exists in the
inherited pipeline and is kept as the `prior_source=map` ABLATION arm: the FCVR
mean_base / prior is then seeded from these MAP-tuned routers instead of the
pre-trained ones. Both arms are run and reported (codex-recom-iter1-notes.md).

iter1 changes: seeds applied; prompt-masked (answer-/explanation-only) labels via
the same collator path as kvq-/fcvr-tuning; AdamW; best-val checkpoint +
early stopping; --map_suffix so a run never overwrites an earlier one.

Multi-GPU (branch OBQA-qwen): `torchrun --nproc_per_node=N` -> data-parallel
DDP, --batch_size is per rank (effective = batch_size x N x --grad_accum_steps);
without torchrun this is the single-process script that produced every Granite
MAP router.
"""
import argparse
import math
import os
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
import torch
import wandb

from model.adapters import granite_adapter, qwen_adapter, deepseek_adapter, qwen36_adapter, gemma4_adapter

from utils import setup_environment, seed_everything
from model import load_peft_model_and_adapter, load_tokenizer
from utils import (load_and_prepare_train_and_val_data, loss_mode_label, add_target_mode_arg,
                   add_system_prompt_arg, eligible_tag_for)
from utils.dist import (init_distributed, make_loader, set_epoch, wrap_ddp, unwrap, reduce_mean,
                        barrier, cleanup)

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
    },
    "qwen36": {
        "prepare": qwen36_adapter.swap_qwen36_moe_blocks,
        "save": qwen36_adapter.save_qwen36_map_routers,
    },
    "gemma4": {
        "prepare": gemma4_adapter.swap_gemma4_moe_blocks,
        "save": gemma4_adapter.save_gemma4_map_routers,
    },
}

def train(model, tokenizer, train_loader, val_loader, args, info):
    # === 0: Make sure we're using the currect swap, save functions ===
    adapter = ADAPTER_MAP[args.model_shortcode]
    save_map_routers_func = adapter["save"]

    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    if args.map_suffix:
        run_name = f"{run_name}_{args.map_suffix}"

    # === 1. (model already prepared by main: swap happens BEFORE the DDP wrap) ===

    # === 2. Create Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    grad_accum = max(1, args.grad_accum_steps)

    # === 3. Run Custom Training Loop ===
    project_name = os.environ.get("WANDB_PROJECT", "moe-uncertainty")
    if info.is_main:
        wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    num_training_batches = len(train_loader)
    print(f"--- Optim: AdamW lr={args.lr} | per-rank batch={args.batch_size} x grad_accum={grad_accum} "
          f"x ranks={info.world} = eff batch {args.batch_size * grad_accum * info.world} ---")
    # Best-val checkpoint + early stopping (the weights on disk are always the
    # best checkpoint; re-save overwrites the previous best of THIS run only).
    # Validation at every epoch end and, with --eval_every N > 0, every N steps;
    # patience counted in evaluations (== epochs when --eval_every 0).
    best_val_loss = float("inf")
    evals_no_improve = 0
    global_step = 0

    def save_routers():
        if info.is_main:
            save_map_routers_func(unwrap(model), args)
        barrier(info)

    def run_validation():
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(info.device) for k, v in batch.items()}
                outputs = model(**inputs)
                total_val_loss += outputs.loss.item()
        model.train()
        return reduce_mean(total_val_loss / len(val_loader), info)

    def check_and_save(avg_val_loss, where, epoch):
        nonlocal best_val_loss, evals_no_improve
        print(f"{where} validation loss: {avg_val_loss:.4f}")
        if info.is_main:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch, "global_step": global_step})
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            evals_no_improve = 0
            print(f"  New best val loss {best_val_loss:.4f} -> saving MAP routers.")
            save_routers()
            return False
        evals_no_improve += 1
        print(f"  No val-loss improvement ({evals_no_improve}/{args.early_stop_patience}).")
        if evals_no_improve >= args.early_stop_patience:
            print(f"--- Early stopping at {where} (best val loss {best_val_loss:.4f}) ---")
            return True
        return False

    print("--- Starting MAP router fine-tuning (Custom Loop) ---")
    if args.eval_every:
        print(f"--- Validation every {args.eval_every} steps AND at every epoch end; "
              f"patience {args.early_stop_patience} evaluations ---")
    stop = False
    for epoch in range(args.epochs):
        set_epoch(train_loader, epoch)
        model.train()
        total_epoch_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not info.is_main)):
            inputs = {k: v.to(info.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            (loss / grad_accum).backward()
            is_step = ((i + 1) % grad_accum == 0) or ((i + 1) == num_training_batches)
            if is_step:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            total_epoch_loss += loss.item()
            if info.is_main:
                wandb.log({
                    "train_loss": loss.item()
                })
            # Mid-epoch validation (skipped on the last batch: the epoch-end validation follows).
            if is_step and args.eval_every and global_step % args.eval_every == 0 and (i + 1) < num_training_batches:
                if check_and_save(run_validation(), f"Epoch {epoch+1} step {global_step}", epoch):
                    stop = True
                    break
        if stop:
            break

        print(f"Epoch {epoch+1} average training loss: {total_epoch_loss / num_training_batches:.4f}")

        if check_and_save(run_validation(), f"Epoch {epoch+1}", epoch):
            break

    if best_val_loss == float("inf"):
        print("--- Val loss never improved; saving final state as a fallback ---")
        save_routers()

    print(f"--- MAP router fine-tuning complete (best val loss {best_val_loss:.4f}) ---")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the deterministic MoE router (MAP baseline).")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="Shortcode for the single dataset to train on (e.g., 'obqa').")
    parser.add_argument("--base_adapter_path", type=str, required=True, help="Path to the pre-trained Stage 1 adapter.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-rank batch size under torchrun.")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps; batch_size x grad_accum_steps x ranks = effective batch. 1 = legacy.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Evaluations (== epochs unless --eval_every > 0) of no val-loss improvement before early stopping (best checkpoint kept).")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="Also validate (and checkpoint on improvement) every N steps; 0 = epoch end only (legacy).")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Drop (never truncate) train/val rows longer than this many tokens; 0 = keep all.")
    parser.add_argument("--map_suffix", type=str, default=None,
                        help="Suffix on router_weights/base/<model>_<dataset>; keeps this run from overwriting an earlier MAP run.")
    add_target_mode_arg(parser, "Must match the Stage-1 adapter's mode.")
    add_system_prompt_arg(parser)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()
    info = init_distributed()
    seed_everything(args.seed)  # random (dataset shuffles/splits) + numpy + torch

    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map=info.device
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # Same loss/collator path as kvq-tuning.py / fcvr-tuning.py: prompt-masked
    # labels (answer-only for MCQA, explanation-only + EOS for generation),
    # right padding, labels-preserving collator. Rank 0 builds the data first.
    if not info.is_main:
        barrier(info)
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(
        tokenizer, [args.dataset_shortcode], seed=args.seed, answer_only=True, max_seq_len=args.max_seq_len or None,
        target_mode=args.target_mode, system_prompt=args.system_prompt,
        eligible_tag=eligible_tag_for(args.model_shortcode))
    if info.is_main:
        barrier(info)
    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    print(f"--- Loss mode: {loss_mode_label([args.dataset_shortcode], args.target_mode)} ---")
    train_loader = make_loader(train_dataset, args.batch_size, data_collator, shuffle=True, seed=args.seed, info=info)
    val_loader = make_loader(val_dataset, args.batch_size, data_collator, shuffle=False, seed=args.seed, info=info)

    # Router swap / unfreeze BEFORE the DDP wrap (DDP registers the trainable params at construction).
    model = ADAPTER_MAP[args.model_shortcode]["prepare"](model)
    model = wrap_ddp(model, info)
    if info.distributed:
        torch.manual_seed(args.seed + info.rank)

    train(model, tokenizer, train_loader, val_loader, args, info)
    cleanup(info)


if __name__ == "__main__":
    main()
