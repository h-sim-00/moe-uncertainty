"""Inspect the medmcqa_gen split BEFORE spending GPU time (CPU only, ~1-2 min).

Prints, for train / val / test:
  * sizes, subject_name histogram (stratification check), gold-letter balance
  * explanation word counts and Granite token counts (prompt / target / total)
    with percentiles -> justifies --max_seq_len (training) and --max_new_tokens
    (readouts) and shows how many gold explanations would exceed the generation cap
  * how many gold explanations open with a MedMCQA-style "Ans. (c) ..." commitment
    (what label_generation_correctness.py::committed_option will see after training)
  * a few sample rows (prompt inner text + target)
Also asserts train/val/test are pairwise disjoint on the question text.
Writes results/data/medmcqa_gen_inspect.json.

Usage (quail, moe_env):  python medmcqa-gen-inspect.py [--n_samples 3] [--max_new_tokens 256] [--max_seq_len 768]
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from utils import setup_environment
from utils.data import load_exp_dataset, MEDMCQA_GEN_MIN_EXP_WORDS, MEDMCQA_GEN_MAX_EXP_WORDS
from utils.prompt import generation_prompt_engineer
from model import load_tokenizer

_ANS = re.compile(r"^\s*\(?\s*ans(?:wer)?\b", re.I)


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q)) if len(a) else float("nan")


def summarize(name, rows, tokenizer, max_new_tokens, max_seq_len):
    words = [len(r["answer"].split()) for r in rows]
    prompt_tok, target_tok = [], []
    for r in rows:
        prompt = generation_prompt_engineer(r, tokenizer=tokenizer)["question"]
        prompt_tok.append(len(tokenizer(prompt, add_special_tokens=False).input_ids))
        target_tok.append(len(tokenizer(r["answer"] + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids))
    total_tok = [p + t for p, t in zip(prompt_tok, target_tok)]
    subj = Counter(r.get("subject_name", "?") for r in rows)
    letters = Counter(r.get("gold_letter", "?") for r in rows)
    ans_style = sum(1 for r in rows if _ANS.match(r["answer"]))
    out = {
        "n": len(rows),
        "subjects": dict(sorted(subj.items(), key=lambda kv: -kv[1])),
        "gold_letters": dict(sorted(letters.items())),
        "exp_words": {q: pct(words, q) for q in (5, 50, 95, 99)},
        "prompt_tokens": {q: pct(prompt_tok, q) for q in (5, 50, 95, 99, 100)},
        "target_tokens": {q: pct(target_tok, q) for q in (5, 50, 95, 99, 100)},
        "total_tokens": {q: pct(total_tok, q) for q in (5, 50, 95, 99, 100)},
        "frac_target_over_max_new_tokens": float(np.mean([t > max_new_tokens for t in target_tok])) if rows else None,
        "frac_total_over_max_seq_len": float(np.mean([t > max_seq_len for t in total_tok])) if rows else None,
        "frac_exp_starts_with_Ans": ans_style / max(1, len(rows)),
    }
    print(f"\n=== {name}: n={out['n']} ===")
    print("  subjects:", ", ".join(f"{k}={v}" for k, v in out["subjects"].items()))
    print("  gold letters:", out["gold_letters"])
    print(f"  exp words   p5/p50/p95/p99 = {out['exp_words'][5]:.0f}/{out['exp_words'][50]:.0f}/{out['exp_words'][95]:.0f}/{out['exp_words'][99]:.0f}"
          f"   (filter {MEDMCQA_GEN_MIN_EXP_WORDS}..{MEDMCQA_GEN_MAX_EXP_WORDS})")
    print(f"  prompt tok  p50/p95/p99/max = {out['prompt_tokens'][50]:.0f}/{out['prompt_tokens'][95]:.0f}/{out['prompt_tokens'][99]:.0f}/{out['prompt_tokens'][100]:.0f}")
    print(f"  target tok  p50/p95/p99/max = {out['target_tokens'][50]:.0f}/{out['target_tokens'][95]:.0f}/{out['target_tokens'][99]:.0f}/{out['target_tokens'][100]:.0f}"
          f"   -> {100*out['frac_target_over_max_new_tokens']:.1f}% exceed max_new_tokens={max_new_tokens}")
    print(f"  total tok   p50/p95/p99/max = {out['total_tokens'][50]:.0f}/{out['total_tokens'][95]:.0f}/{out['total_tokens'][99]:.0f}/{out['total_tokens'][100]:.0f}"
          f"   -> {100*out['frac_total_over_max_seq_len']:.1f}% exceed max_seq_len={max_seq_len} (would be DROPPED at training)")
    print(f"  gold explanations opening with 'Ans...': {100*out['frac_exp_starts_with_Ans']:.1f}%")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--out", default="results/data/medmcqa_gen_inspect.json")
    args = p.parse_args()

    setup_environment()
    tokenizer = load_tokenizer(args.model_shortcode)
    train, val, test = load_exp_dataset("medmcqa_gen", seed=args.seed)

    def qk(r):
        return " ".join(r["question"].lower().split())
    st, sv, ss = {qk(r) for r in train}, {qk(r) for r in val}, {qk(r) for r in test}
    assert not (st & sv) and not (st & ss) and not (sv & ss), "train/val/test overlap on question text!"
    print("Disjointness OK (train/val/test share no question text).")

    report = {"seed": args.seed}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        report[name] = summarize(name, rows, tokenizer, args.max_new_tokens, args.max_seq_len)

    print("\n=== sample rows (train) ===")
    for r in train[: args.n_samples]:
        print("-" * 70)
        print(r["question"])
        print(f"[gold {r['gold_letter']}] TARGET:{r['answer']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
