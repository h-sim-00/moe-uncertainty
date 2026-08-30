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
       --target_mode letter              -> exactly the gold letter, no EOS (arm A)
       --target_mode answer_explanation  -> letter + "\nExplanation:" + explanation
                                            + EOS (arm B)
  6. (comparison arms only) the FIRST loss token is the bare letter token id
     (tokenizer.convert_tokens_to_ids(gold_letter)) -- the token the letter
     read-out (evaluate_letter.py) scores. This is what makes the two arms
     comparable; it must hold for every row.
Prints one decoded example showing the prompt/target mask boundary.

Usage:
    python check-answer-only-labels.py                            # granite/obqa (MCQA)
    python check-answer-only-labels.py --dataset_shortcode medexqa  # generation
    python check-answer-only-labels.py --dataset_shortcode medmcqa_gen  # generation (MedMCQA explanations)
    python check-answer-only-labels.py --dataset_shortcode medmcqa_gen --target_mode letter              # arm A
    python check-answer-only-labels.py --dataset_shortcode medmcqa_gen --target_mode answer_explanation  # arm B
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from utils import setup_environment
from model import load_tokenizer
from utils.data import (load_exp_dataset, load_and_prepare_train_and_val_data, is_generation_dataset,
                        build_target_mode_example, comparison_eligible_indices, add_target_mode_arg,
                        add_system_prompt_arg, eligible_tag_for)
from utils.prompt import multiple_choice_prompt_engineer, generation_prompt_engineer, SYSTEM_INSTRUCTIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="obqa")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=4, help="How many val batches to check (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    add_target_mode_arg(p)
    add_system_prompt_arg(p)
    p.add_argument("--max_seq_len", type=int, default=0,
                   help="Same as the training scripts' --max_seq_len (comparison arms: applies the shared eligible-ID list).")
    return p.parse_args()


def main():
    args = parse_args()
    setup_environment()
    tokenizer = load_tokenizer(args.model_shortcode)
    generation = is_generation_dataset([args.dataset_shortcode])
    arm = args.target_mode != "explanation"          # MedMCQA-comparison arm A / B
    if arm:
        system_instruction = SYSTEM_INSTRUCTIONS[args.system_prompt]
        engineer = lambda x, tokenizer: build_target_mode_example(x, tokenizer, args.target_mode, system_instruction)  # noqa: E731
    else:
        engineer = generation_prompt_engineer if generation else multiple_choice_prompt_engineer
    # EOS is part of the target for the generation recipes (explanation / answer_explanation),
    # never for the letter-only ones (MCQA, target_mode=letter).
    target_has_eos = generation and args.target_mode != "letter"

    # Same call the training scripts make (val split is unshuffled -> row order matches).
    _, val_dataset = load_and_prepare_train_and_val_data(
        tokenizer, [args.dataset_shortcode], seed=args.seed, answer_only=True,
        max_seq_len=args.max_seq_len or None, target_mode=args.target_mode,
        system_prompt=args.system_prompt, eligible_tag=eligible_tag_for(args.model_shortcode))
    _, val_raw, _ = load_exp_dataset(args.dataset_shortcode, seed=args.seed)
    if arm and args.max_seq_len:
        # the loader kept only the shared eligible rows -> mirror that here so row order matches
        keep = comparison_eligible_indices(val_raw, tokenizer, args.max_seq_len)
        val_raw = [val_raw[i] for i in keep]
    elif args.max_seq_len:
        raise SystemExit("--max_seq_len is only supported with a comparison --target_mode in this check "
                         "(the legacy path drops rows after tokenisation; run without it).")
    val_engineered = [engineer(x, tokenizer=tokenizer) for x in val_raw]
    assert len(val_engineered) == len(val_dataset), "val length mismatch between raw and preprocessed"

    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    eos = tokenizer.eos_token or ""
    n_checked, n_target_tokens, n_pad_total, n_junction_diff = 0, 0, 0, 0
    printed_example = False

    def _first_diff(a: str, b: str):
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return None if len(a) == len(b) else n
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
            expected = val_engineered[row_idx]["answer"] + (eos if target_has_eos else "")
            label_ids = labels[r][unmasked].tolist()
            if arm:
                # 6. first loss token == bare letter token (what evaluate_letter.py scores)
                gold = val_raw[row_idx]["gold_letter"]
                letter_id = tokenizer.convert_tokens_to_ids(gold)
                assert label_ids[0] == letter_id, (
                    f"row {row_idx}: first loss token id {label_ids[0]} "
                    f"({tokenizer.convert_ids_to_tokens(label_ids[0])!r}) != letter token id {letter_id} "
                    f"({gold!r}); the letter read-out would score a different token than the one trained on")
            expected_ids = tokenizer(expected, add_special_tokens=False).input_ids
            if label_ids != expected_ids:
                # Token ids can legitimately differ at the prompt/target junction
                # (joint vs separate tokenisation); the TEXT must still be identical.
                # Compare decoded text with tokenizer clean-up disabled on both sides
                # (Granite's decode otherwise rewrites " ," -> "," , " 's" -> "'s" ...).
                decoded = tokenizer.decode(label_ids, clean_up_tokenization_spaces=False)
                expected_rt = tokenizer.decode(expected_ids, clean_up_tokenization_spaces=False)
                if decoded != expected_rt:
                    i = _first_diff(decoded, expected_rt)
                    lo = max(0, (i or 0) - 40)
                    raise AssertionError(
                        f"row {row_idx}: loss tokens differ from target text at char {i} "
                        f"(len decoded={len(decoded)}, expected={len(expected_rt)}, "
                        f"n_ids {len(label_ids)} vs {len(expected_ids)}).\n"
                        f"  decoded : {decoded[lo:(i or 0) + 40]!r}\n"
                        f"  expected: {expected_rt[lo:(i or 0) + 40]!r}")
                n_junction_diff += 1
            else:
                decoded = tokenizer.decode(label_ids, clean_up_tokenization_spaces=False)
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

    kind = {"letter": "letter (arm A)", "answer_explanation": "letter+explanation(+EOS) (arm B)"}.get(
        args.target_mode, "explanation(+EOS)" if generation else "answer")
    print(f"OK: {n_checked} rows checked, {n_target_tokens} {kind} tokens in loss "
          f"({n_target_tokens / n_checked:.2f} per row), {n_pad_total} pad positions all masked; "
          f"prompt fully masked; loss block contiguous at sequence end.")
    if arm:
        print(f"OK: first loss token is the bare gold-letter token in all {n_checked} rows "
              f"(target_mode={args.target_mode}, system_prompt={args.system_prompt}, "
              f"eos={tokenizer.eos_token!r}, pad={tokenizer.pad_token!r}).")
    if n_junction_diff:
        print(f"note: {n_junction_diff} row(s) had a different token split at the prompt/target junction "
              f"(joint vs separate tokenisation) but identical target text -- harmless.")


if __name__ == "__main__":
    main()
