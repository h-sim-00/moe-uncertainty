"""
Pre-flight check for Stage 1 with expert LoRA.

Runs the EXACT model/optimizer/data path that kvq-tuning.py uses, for a handful
of steps, and reports peak GPU memory + an estimated wall-clock per epoch. Use
it to decide whether the real run will fit before committing hours to it.

  python expert-lora-memory-check.py --dataset_shortcode obqa --batch_size 8 --expert_lora_r 64

Nothing is saved and no weights are written, so it cannot clobber a run.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from utils import setup_environment, load_and_prepare_train_and_val_data
from model import load_peft_model, load_tokenizer


GiB = 1024 ** 3


def parse_args():
    parser = argparse.ArgumentParser(description="Measure peak GPU memory for Stage-1 expert-LoRA training.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, default="obqa")
    parser.add_argument("--finetune_mode", type=str, default="qkv_experts", choices=["qkv", "qkv_experts"])
    parser.add_argument("--expert_lora_r", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3, help="Only used to extrapolate the time estimate.")
    parser.add_argument("--steps", type=int, default=4,
                        help="Training steps to run. Must be >=2: Adam allocates its state during the first step.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def report(tag, device):
    alloc = torch.cuda.memory_allocated(device) / GiB
    peak = torch.cuda.max_memory_allocated(device) / GiB
    reserved = torch.cuda.max_memory_reserved(device) / GiB
    print(f"  [{tag}] allocated {alloc:6.2f} GiB | peak allocated {peak:6.2f} GiB | peak reserved {reserved:6.2f} GiB")
    return peak, reserved


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- run this on the GPU host.")

    device = args.device
    idx = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory / GiB
    try:
        # On unified-memory parts (e.g. GB10) nvidia-smi reports "Not Supported";
        # this may also be unavailable or report the shared CPU+GPU pool.
        free_now = f"{torch.cuda.mem_get_info(idx)[0] / GiB:.1f} GiB"
    except Exception as e:  # noqa: BLE001 - informational only
        free_now = f"unavailable ({type(e).__name__})"

    print("=" * 70)
    print(f"GPU {idx}: {props.name} | total {total:.1f} GiB | free right now {free_now}")
    print(f"torch {torch.__version__} | mode={args.finetune_mode} r={args.expert_lora_r} "
          f"batch={args.batch_size} lr={args.lr}")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats(idx)

    print("\n--- Loading model ---")
    model = load_peft_model(
        args.model_shortcode,
        finetune_mode=args.finetune_mode,
        expert_lora_r=args.expert_lora_r,
        device_map=device,
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    param_dtype = next(model.parameters()).dtype
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    bytes_per = torch.finfo(param_dtype).bits // 8
    print(f"\nModel dtype: {param_dtype} ({bytes_per} bytes/param)")
    print(f"Total params     : {n_total/1e9:.3f} B  ({n_total*bytes_per/GiB:.2f} GiB)")
    print(f"Trainable params : {n_train/1e6:.1f} M   "
          f"(grads {n_train*bytes_per/GiB:.2f} GiB + Adam state {2*n_train*bytes_per/GiB:.2f} GiB)")
    report("after model load", idx)

    print("\n--- Building data ---")
    train_dataset, _ = load_and_prepare_train_and_val_data(tokenizer, [args.dataset_shortcode])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    batches_per_epoch = len(train_loader)
    print(f"{len(train_dataset)} train examples -> {batches_per_epoch} batches/epoch at batch_size={args.batch_size}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    steps = max(2, args.steps)
    print(f"\n--- Running {steps} training steps (same loop as kvq-tuning.py) ---")
    model.train()
    step_times = []
    try:
        for i, batch in enumerate(train_loader):
            if i >= steps:
                break
            torch.cuda.synchronize(idx)
            t0 = time.time()

            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            outputs.loss.backward()
            optimizer.step()

            torch.cuda.synchronize(idx)
            dt = time.time() - t0
            step_times.append(dt)
            print(f"  step {i+1}/{steps}: loss {outputs.loss.item():.4f} | "
                  f"seq_len {inputs['input_ids'].shape[1]} | {dt:.2f}s")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        peak, reserved = report("at OOM", idx)
        print("\n" + "=" * 70)
        print("VERDICT: OUT OF MEMORY -- this configuration will NOT train.")
        print("Try, in order of least damage to the protocol:")
        print("  1. --expert_lora_r 16   (fewer optimizer states; the paper states no rank)")
        print("  2. --batch_size 4       (changes the Stage-1 protocol -- note it if you use it)")
        print("=" * 70)
        return

    peak, reserved = report("after steps", idx)

    # Steady-state timing: skip step 1 (warmup, Adam state allocation).
    steady = sum(step_times[1:]) / max(1, len(step_times) - 1)
    epoch_min = steady * batches_per_epoch / 60
    headroom = total - reserved

    print("\n" + "=" * 70)
    print(f"Peak reserved: {reserved:.2f} GiB of {total:.1f} GiB  ->  {headroom:.2f} GiB headroom")
    print(f"Steady step time: {steady:.2f}s  ->  ~{epoch_min:.1f} min/epoch, "
          f"~{epoch_min*args.epochs/60:.1f} h for {args.epochs} epochs")
    if headroom < 2:
        print("VERDICT: IT FITS, BUT BARELY (<2 GiB spare). A long sequence in a later batch")
        print("         could still OOM mid-run. Consider --expert_lora_r 16.")
    else:
        print("VERDICT: FITS. Safe to launch bash kvq-tuning-granite-obqa-experts.sh")
    print("Note: peak here is measured over a few batches; the longest batch in the")
    print("      dataset may push it slightly higher.")
    print("=" * 70)


if __name__ == "__main__":
    main()
