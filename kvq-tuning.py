"""
Stage 1: deterministic MAP adaptation with LoRA (paper Appendix D.2).

The paper applies LoRA to "the attention modules (Q/K/V projections) and the
Expert networks". `--finetune_mode qkv_experts` (the default on this branch)
does both: PEFT adapts Q/K/V, and model.expert_lora adapts every expert matrix
of every MoE layer. `--finetune_mode qkv` reproduces the original Q/K/V-only
behaviour.

Multi-GPU (branch OBQA-qwen): launch with `torchrun --nproc_per_node=N` for
data-parallel DDP -- every rank holds a full replica, --batch_size is PER RANK,
so the effective batch is batch_size x N x --grad_accum_steps. Without torchrun
the script is the single-process one that produced every Granite adapter.
"""

import argparse
import math
import os
import wandb
import torch

from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
from utils import setup_environment, seed_everything
from model import load_peft_model, load_tokenizer
from model.expert_lora import save_expert_lora, expert_lora_path
from utils import (load_and_prepare_train_and_val_data, loss_mode_label, add_target_mode_arg,
                   add_system_prompt_arg, eligible_tag_for)
from utils.dist import (init_distributed, make_loader, set_epoch, wrap_ddp, unwrap, reduce_mean,
                        barrier, cleanup)

def train(model, tokenizer, train_loader, val_loader, args, info):
    """
    Fine-tunes a model using the Hugging Face Trainer API.
    """
    project_name = os.environ.get("WANDB_PROJECT", "moe-uncertainty")
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    if args.adapter_suffix:
        run_name = f"{run_name}_{args.adapter_suffix}"
    if info.is_main:
        wandb.init(project=project_name, name=run_name, config=vars(args), reinit=True)

    num_training_batches = len(train_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Paper D.2: "All models were trained using the AdamW optimiser", lr with
    # cosine decay and warmup_ratio warmup.
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    grad_accum = max(1, args.grad_accum_steps)
    steps_per_epoch = math.ceil(num_training_batches / grad_accum)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_optim_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )
    print(f"--- Optim: AdamW lr={args.lr} cosine warmup={warmup_steps}/{total_optim_steps} steps "
          f"| per-rank batch={args.batch_size} x grad_accum={grad_accum} x ranks={info.world} "
          f"= eff batch {args.batch_size * grad_accum * info.world} ---")

    final_save_path = f"./adapters/{args.model_shortcode}-{args.dataset_shortcode}"
    if args.adapter_suffix:
        final_save_path = f"{final_save_path}-{args.adapter_suffix}"

    def save_adapter():
        if info.is_main:
            unwrap(model).save_pretrained(final_save_path)
            # PEFT only serialises the Q/K/V adapters; the expert-LoRA factors live in
            # a custom module and are written next to them inside the same directory.
            if args.finetune_mode == "qkv_experts":
                save_expert_lora(unwrap(model), expert_lora_path(final_save_path))
        barrier(info)

    # Best-val checkpointing + early stopping on val loss (mirrors fcvr-tuning.py):
    # the adapter on disk is always the best checkpoint, never the (overfit) last
    # one. Evaluation happens at every epoch end and -- if --eval_every N > 0 --
    # additionally every N optimizer steps (large datasets such as medmcqa_gen,
    # where one epoch is thousands of steps). Patience is counted in
    # EVALUATIONS (== epochs when --eval_every 0, the legacy behaviour).
    best_val_loss = float("inf")
    evals_no_improve = 0
    global_step = 0

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
        """-> True if training should stop (patience exhausted)."""
        nonlocal best_val_loss, evals_no_improve
        print(f"{where} validation loss: {avg_val_loss:.4f}")
        if info.is_main:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch, "global_step": global_step})
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            evals_no_improve = 0
            print(f"  New best val loss {best_val_loss:.4f} -> saving adapter to {final_save_path}")
            save_adapter()
            return False
        evals_no_improve += 1
        print(f"  No val-loss improvement ({evals_no_improve}/{args.early_stop_patience}); "
              f"keeping best checkpoint (val {best_val_loss:.4f}).")
        if evals_no_improve >= args.early_stop_patience:
            print(f"--- Early stopping at {where} (best val loss {best_val_loss:.4f}) ---")
            return True
        return False

    print("--- Starting MAP Fine-tuning (Custom Loop) ---")
    if args.eval_every:
        print(f"--- Validation every {args.eval_every} optimizer steps AND at every epoch end; "
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
            # Scale for gradient accumulation, then step every grad_accum
            # micro-batches (and on the final micro-batch of the epoch).
            (loss / grad_accum).backward()
            is_step = ((i + 1) % grad_accum == 0) or ((i + 1) == num_training_batches)
            if is_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            total_epoch_loss += loss.item()
            if info.is_main:
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })
            # Mid-epoch validation (skipped on the last batch: the epoch-end validation follows).
            if is_step and args.eval_every and global_step % args.eval_every == 0 and (i + 1) < num_training_batches:
                if check_and_save(run_validation(), f"Epoch {epoch+1} step {global_step}", epoch):
                    stop = True
                    break
        if stop:
            break

        print(f"Epoch {epoch+1} average training loss: {total_epoch_loss / num_training_batches:.4f}")

        # Epoch-end validation (always).
        if check_and_save(run_validation(), f"Epoch {epoch+1}", epoch):
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
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size (per rank under torchrun).")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps; batch_size x grad_accum_steps x ranks = effective batch. 1 = legacy.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total optimizer steps used for LR warmup (paper D.2: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Evaluations (== epochs unless --eval_every > 0) of no val-loss improvement before early stopping (best checkpoint kept).")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="Also validate (and checkpoint on improvement) every N optimizer steps; 0 = epoch end only (legacy). "
                             "Use for large train sets (medmcqa_gen) so early stopping is not epoch-coarse.")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Drop (never truncate) train/val rows longer than this many tokens; 0 = keep all. Memory guard for long explanations.")
    add_target_mode_arg(parser)
    add_system_prompt_arg(parser)
    return parser.parse_args()


def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()
    info = init_distributed()

    seed_everything(args.seed)  # random (dataset shuffles/splits) + numpy + torch

    device = info.device

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
    # Rank 0 builds the data first (it may write the frozen eligible-ID list /
    # dataset cache); the others then read and verify the same files.
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

    model = wrap_ddp(model, info)
    if info.distributed:
        torch.manual_seed(args.seed + info.rank)   # per-rank dropout noise; init already identical (seeded before wrap)

    train(model, tokenizer, train_loader, val_loader, args, info)
    cleanup(info)

if __name__ == "__main__":
    main()
