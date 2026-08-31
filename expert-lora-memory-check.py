"""
Pre-flight check for Stage 1 (LoRA) and -- with --stage fcvr -- Stage 2 (FCVR).

Runs the EXACT model/optimizer/data path that kvq-tuning.py (or fcvr-tuning.py)
uses, for a handful of steps, and reports peak GPU memory + an estimated
wall-clock per epoch. Use it to decide whether the real run will fit before
committing hours to it.

  python expert-lora-memory-check.py --dataset_shortcode obqa --batch_size 8 --expert_lora_r 64
  python expert-lora-memory-check.py --dataset_shortcode medmcqa_gen --target_mode answer_explanation \
         --max_seq_len 768 --batch_size 8          # MedMCQA-comparison arm B (worst-case batch first)
  python expert-lora-memory-check.py --model_shortcode qwen36 --dataset_shortcode obqa_gen --finetune_mode qkv \
         --target_mode answer_explanation --max_seq_len 768 --batch_size 2            # Qwen3.6 Stage-1, per-rank batch
  python expert-lora-memory-check.py --model_shortcode qwen36 --dataset_shortcode obqa_gen --stage fcvr \
         --target_mode answer_explanation --max_seq_len 768 --batch_size 1            # Qwen3.6 FCVR (fresh heads)
  python expert-lora-memory-check.py --model_shortcode gemma4 --dataset_shortcode obqa_gen --stage fcvr \
         --swap_layers 5 6 7 8 18 19 26 27 28 29 --target_mode answer_explanation --max_seq_len 768 --batch_size 1
                                                                                       # Gemma 4 FCVR ('depth' layers; the
                                                                                       # Granite default set exceeds 30 layers)

Uses the same loader / collator path as kvq-tuning.py (answer-only or
target_mode labels, right padding, DataCollatorForSeq2Seq) and runs the batch of
the LONGEST rows first, so the reported peak is the real worst case.
Nothing is saved and no weights are written, so it cannot clobber a run.
GRADIENT_CHECKPOINTING=1 is honoured exactly as in the training scripts.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from utils import (setup_environment, load_and_prepare_train_and_val_data, add_target_mode_arg,
                   add_system_prompt_arg, eligible_tag_for)
from model import load_peft_model, load_peft_model_and_adapter, load_tokenizer


GiB = 1024 ** 3
FCVR_DEFAULT_LAYERS = [5, 6, 7, 8, 19, 20, 28, 29, 30, 31]


def parse_args():
    parser = argparse.ArgumentParser(description="Measure peak GPU memory for Stage-1 LoRA / Stage-2 FCVR training.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, default="obqa")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "fcvr"],
                        help="stage1 (default): LoRA fine-tuning path of kvq-tuning.py. fcvr: identity LoRA + fresh FCVR "
                             "heads on --swap_layers, ELBO step of fcvr-tuning.py (kl_mask attention, --beta).")
    parser.add_argument("--finetune_mode", type=str, default="qkv_experts", choices=["qkv", "qkv_experts"])
    parser.add_argument("--expert_lora_r", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8, help="PER-RANK batch (what one GPU sees under torchrun).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3, help="Only used to extrapolate the time estimate.")
    parser.add_argument("--steps", type=int, default=4,
                        help="Training steps to run. Must be >=2: Adam allocates its state during the first step.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swap_layers", type=int, nargs="+", default=FCVR_DEFAULT_LAYERS, help="[fcvr]")
    parser.add_argument("--beta", type=float, default=0.01, help="[fcvr]")
    add_target_mode_arg(parser)
    add_system_prompt_arg(parser)
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Drop rows longer than this (same as kvq-tuning.py --max_seq_len); 0 = keep all.")
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
    print(f"torch {torch.__version__} | stage={args.stage} mode={args.finetune_mode} r={args.expert_lora_r} "
          f"batch={args.batch_size} lr={args.lr}")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats(idx)

    print("\n--- Loading model ---")
    fcvr_routers = []
    if args.stage == "stage1":
        model = load_peft_model(
            args.model_shortcode,
            finetune_mode=args.finetune_mode,
            expert_lora_r=args.expert_lora_r,
            device_map=device,
        )
    else:
        # Same path as fcvr-tuning.py with --prior_source pretrained and no
        # --load_layers: identity LoRA wrapper (the Stage-1 adapter is not needed
        # for a memory probe) + fresh FCVR heads on the swap layers.
        from model.adapters import get_adapter, moe_router
        model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=None, device_map=device)
        fargs = argparse.Namespace(model_shortcode=args.model_shortcode, dataset_shortcode=args.dataset_shortcode,
                                   swap_layers=list(args.swap_layers), load_layers=[], train_layers=list(args.swap_layers),
                                   run_suffix="memcheck-never-saved")
        model = get_adapter(args.model_shortcode).prepare(model, method="fcvr", args=fargs)
        causal_model = model.base_model.model.model
        fcvr_routers = [moe_router(causal_model.layers[l]) for l in args.swap_layers]
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

    print("\n--- Building data (same loader/collator path as the training scripts) ---")
    train_dataset, _ = load_and_prepare_train_and_val_data(
        tokenizer, [args.dataset_shortcode], seed=args.seed, answer_only=True,
        max_seq_len=args.max_seq_len or None, target_mode=args.target_mode,
        system_prompt=args.system_prompt, eligible_tag=eligible_tag_for(args.model_shortcode))
    tokenizer.padding_side = "right"
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    batches_per_epoch = len(train_loader)
    print(f"{len(train_dataset)} train examples -> {batches_per_epoch} batches/epoch at batch_size={args.batch_size}")
    # Worst case first: the batch made of the LONGEST rows (per-batch padding means
    # a random batch is not the memory peak -- this one is).
    lengths = [len(x) for x in train_dataset["input_ids"]]
    longest = sorted(range(len(lengths)), key=lambda i: -lengths[i])[:args.batch_size]
    worst_batch = collator([train_dataset[i] for i in longest])
    print(f"worst-case batch: {len(longest)} longest rows, padded seq_len {worst_batch['input_ids'].shape[1]}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    steps = max(2, args.steps)
    print(f"\n--- Running {steps} training steps (same loop as the training script; step 1 = worst-case batch) ---")
    model.train()
    step_times = []

    def _batches():
        yield worst_batch
        for b in train_loader:
            yield b
    try:
        for i, batch in enumerate(_batches()):
            if i >= steps:
                break
            torch.cuda.synchronize(idx)
            t0 = time.time()

            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            if fcvr_routers:
                kl_mask = inputs["attention_mask"].reshape(-1)
                loss = loss + args.beta * sum(r.kl_divergence(mask=kl_mask) for r in fcvr_routers)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(idx)
            dt = time.time() - t0
            step_times.append(dt)
            print(f"  step {i+1}/{steps}: loss {loss.item():.4f} | "
                  f"seq_len {inputs['input_ids'].shape[1]} | {dt:.2f}s")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        peak, reserved = report("at OOM", idx)
        print("\n" + "=" * 70)
        print("VERDICT: OUT OF MEMORY -- this configuration will NOT train.")
        print("Try, in order of least damage to the protocol:")
        print("  1. GRADIENT_CHECKPOINTING=1  (activation recompute; mathematically identical)")
        if args.stage == "stage1" and args.finetune_mode == "qkv_experts":
            print("  2. --expert_lora_r 16   (fewer optimizer states; the paper states no rank)")
        print("  3. halve --batch_size and double --grad_accum_steps (same effective batch; note it if you use it)")
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
          f"~{epoch_min*args.epochs/60:.1f} h for {args.epochs} epochs (single process; /N ranks under torchrun)")
    if headroom < 2:
        print("VERDICT: IT FITS, BUT BARELY (<2 GiB spare). A long sequence in a later batch")
        print("         could still OOM mid-run. Consider GRADIENT_CHECKPOINTING=1 / a smaller per-rank batch.")
    else:
        print("VERDICT: FITS. Safe to launch the training driver.")
    print("Note: peak here is measured over a few batches; the longest batch in the")
    print("      dataset may push it slightly higher.")
    print("=" * 70)


if __name__ == "__main__":
    main()
