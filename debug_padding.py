"""
Quick padding diagnostic -- NO GPU / no model forward needed (just the tokenizer).

Tokenizes a real batch EXACTLY like evaluate_vtsr.py does (multiple_choice_prompt_engineer
-> tokenizer(padding=True)), then reports whether the `[:, -1, :]` final-token readout used
in the eval lands on each sequence's REAL last token or on a padding token.

Run it in seconds instead of waiting for a full eval:

    python debug_padding.py                                  # granite tokenizer, obqa, batch 8
    python debug_padding.py --dataset_shortcode arc_c --batch_size 4

How to read the output:
  - "rows where `-1` == real last token: N/N"  -> ALL rows OK, `-1` extraction is CORRECT.
  - anything less than N/N                      -> some rows are padded at position -1, so the
                                                   eval's `-1` readout is WRONG for those rows
                                                   (right-padded) and should use a mask-based readout.
"""

import argparse
import torch

from utils import setup_environment, load_exp_dataset, multiple_choice_prompt_engineer
from model import load_tokenizer


def report_padding(tokenizer, inputs):
    """Report whether `[:, -1, :]` grabs the real final token for every row."""
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    bsz, seq_len = ids.shape

    # Last index whose mask == 1, per row. Works for LEFT or RIGHT padding:
    #   left-padded  [0 0 1 1 1] -> real last = index 4 (== seq_len-1)
    #   right-padded [1 1 1 0 0] -> real last = index 2 (!= seq_len-1)
    idx = torch.arange(seq_len)
    last_real = (mask * idx).argmax(dim=1)      # [bsz]
    minus1_idx = seq_len - 1                     # what `[:, -1]` actually grabs
    n_ok = int((last_real == minus1_idx).sum().item())

    print("\n" + "=" * 66)
    print("[PADDING CHECK]")
    print(f"  tokenizer.padding_side = {tokenizer.padding_side!r}")
    print(f"  pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  batch shape (rows x seq_len) = {bsz} x {seq_len}")
    print(f"  `-1` grabs index {minus1_idx} in every row")
    print(f"  real last-token index per row = {last_real.tolist()}")
    print(f"  rows where `-1` == real last token: {n_ok}/{bsz}")
    for r in range(min(3, bsz)):
        lr = last_real[r].item()
        tok_minus1 = tokenizer.decode(ids[r, -1:].tolist())
        tok_real = tokenizer.decode(ids[r, lr:lr + 1].tolist())
        flag = "OK" if lr == minus1_idx else "MISMATCH"
        print(f"    row {r}: [-1] -> {tok_minus1!r}   real-last(idx {lr}) -> {tok_real!r}   [{flag}]")
    if n_ok == bsz:
        print("  VERDICT: `-1` is the real final token for EVERY row -> extraction CORRECT.")
    else:
        print("  VERDICT: `-1` hits a PAD token on some rows -> extraction WRONG (right-padded).")
        print("           The eval should use a mask-based readout (gather at real last-token index).")
    print("=" * 66 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose tokenizer padding for the final-token readout.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, default="obqa")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    setup_environment()
    tokenizer = load_tokenizer(args.model_shortcode)

    dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    processed = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x["question"] for x in processed][: args.batch_size]
    print(f"Loaded {len(questions)} '{args.dataset_shortcode}' questions (batch_size={args.batch_size}).")

    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    report_padding(tokenizer, inputs)


if __name__ == "__main__":
    main()
