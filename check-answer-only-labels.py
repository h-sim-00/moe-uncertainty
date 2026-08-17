"""Sanity check for answer-only / explanation-only training labels (run on quail-1 before training).

Builds the loaders exactly as kvq-tuning.py / fcvr-tuning.py / router-tuning.py
now do (`load_and_prepare_train_and_val_data(..., answer_only=True)` + right
padding + DataCollatorForSeq2Seq) and verifies that the only positions
contributing to the loss are the target tokens:
  1. every row has at least one label != -100
  2. pad positions (attention_mask == 0) all have label -100
  3. every unmasked label equals the input_id at the same position
  4. the unmasked positions form ONE contiguous block at the end of the real
     tokens (prompt fully masked, no stray prompt positions in the loss)
  5. the decoded unmasked tokens equal the example's target:
       MCQA       -> exactly the answer letter
       generation -> the gold explanation followed by EOS (the model must learn
                     to stop), checked against the prompt-engineered examples,
                     in order, on the unshuffled val split
Prints one decoded example showing the prompt/target mask boundary.

Usage:
    python check-answer-only-labels.py                            # granite/obqa (MCQA)
    python check-answer-only-labels.py --dataset_shortcode medexqa  # generation
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from utils import setup_environment
from model import load_tokenizer
from utils.data import load_exp_dataset, load_and_prepare_train_and_val_data, is_generation_dataset
from utils.prompt import multiple_choice_prompt_engineer, generation_prompt_engineer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="obqa")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=4, help="How many val batches to check (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    setup_environment()
    tokenizer = load_tokenizer(args.model_shortcode)
    generation = is_generation_dataset([args.dataset_shortcode])
    engineer = generation_prompt_engineer if generation else multiple_choice_prompt_engineer

    # Same call the training scripts make (val split is unshuffled -> row order matches).
    _, val_dataset = load_and_prepare_train_and_val_data(
        tokenizer, [args.dataset_shortcode], seed=args.seed, answer_only=True)
    _, val_raw, _ = load_exp_dataset(args.dataset_shortcode, seed=args.seed)
    val_engineered = [engineer(x, tokenizer=tokenizer) for x in val_raw]
    assert len(val_engineered) == len(val_dataset), "val length mismatch between raw and preprocessed"

    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    eos = tokenizer.eos_token or ""
    n_checked, n_target_tokens, n_pad_total = 0, 0, 0
    printed_example = False
    for b, batch in enumerate(val_loader):
        if args.num_batches and b >= args.num_batches:
            break
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        for r in range(input_ids.shape[0]):
            row_idx = b * args.batch_size + r
            unmasked = (labels[r] != -100).nonzero(as_tuple=True)[0]
            real = (attention_mask[r] == 1).nonzero(as_tuple=True)[0]
            assert len(unmasked) >= 1, f"row {row_idx}: no unmasked label positions"
            assert (labels[r][attention_mask[r] == 0] == -100).all(), \
                f"row {row_idx}: pad position with a real label"
            assert (labels[r][unmasked] == input_ids[r][unmasked]).all(), \
                f"row {row_idx}: unmasked labels differ from input_ids"
            # Contiguous block ending at the last real token.
            assert int(unmasked[-1]) == int(real[-1]), \
                f"row {row_idx}: loss block does not end at the last real token"
            assert int(unmasked[-1] - unmasked[0]) + 1 == len(unmasked), \
                f"row {row_idx}: loss positions are not contiguous"
            decoded = tokenizer.decode(labels[r][unmasked])
            expected = val_engineered[row_idx]["answer"] + (eos if generation else "")
            assert decoded == expected, \
                f"row {row_idx}: loss tokens decode to {decoded[:120]!r}..., expected {expected[:120]!r}..."
            n_checked += 1
            n_target_tokens += len(unmasked)
            n_pad_total += int((attention_mask[r] == 0).sum())

            if not printed_example:
                prompt_part = tokenizer.decode(input_ids[r][: unmasked[0]])
                print("=== example row 0 ===")
                print(f"prompt (masked, {unmasked[0].item()} tokens): ...{prompt_part[-200:]}")
                print(f"target (in loss, {len(unmasked)} token(s)): {decoded[:200]!r}{'...' if len(decoded) > 200 else ''}")
                print(f"padding: {(attention_mask[r] == 0).sum().item()} tokens, all labeled -100")
                print("=====================")
                printed_example = True

    kind = "explanation(+EOS)" if generation else "answer"
    print(f"OK: {n_checked} rows checked, {n_target_tokens} {kind} tokens in loss "
          f"({n_target_tokens / n_checked:.2f} per row), {n_pad_total} pad positions all masked; "
          f"prompt fully masked; loss block contiguous at sequence end.")


if __name__ == "__main__":
    main()
