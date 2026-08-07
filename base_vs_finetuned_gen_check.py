"""
Ablation check: does the BASE granite-3.1-instruct model already generate coherent
MedExQA explanations WITHOUT the Stage-1 (KVQ) adapter?

Runs the SAME neutral prompt (utils.prompt.generation_prompt_engineer) and the SAME
MedExQA test split (utils.data.load_exp_dataset, seed 42 -> first 175 pooled rows) on:
  (A) the base model            -- load_model()                    [no adapter]
  (B) the Stage-1 fine-tuned    -- load_peft_model_and_adapter()   [./adapters/granite-medexqa]
and prints, per example: question / gold Explanation 1 / (A) base gen / (B) finetuned gen.

If (A) is already fluent and on-topic, Stage-1 SFT is contributing little and may be
distorting the router-uncertainty signal rather than enabling the task. If (B) is
clearly better (more faithful / better-formatted), Stage-1 is pulling its weight.

Run on quail-1 (conda env moe_env), from the repo root:
    conda activate moe_env
    python base_vs_finetuned_gen_check.py                 # 4 examples, both models
    python base_vs_finetuned_gen_check.py --n 8           # more examples
    python base_vs_finetuned_gen_check.py --base_only     # skip the adapter
    python base_vs_finetuned_gen_check.py --device cuda:0 # override GPU
"""
import argparse
import textwrap

import torch

from model import load_model, load_tokenizer, load_peft_model_and_adapter
from utils.prompt import generation_prompt_engineer
from utils.data import load_exp_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--dataset_shortcode", default="medexqa")
    p.add_argument("--adapter_path", default="./adapters/granite-medexqa")
    p.add_argument("--n", type=int, default=4, help="number of test examples")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_only", action="store_true", help="skip the fine-tuned model")
    return p.parse_args()


@torch.no_grad()
def generate_for(model, tokenizer, examples, device, max_new_tokens):
    """Greedy (deterministic) generation of the explanation region for each example."""
    model.config.use_cache = True
    model.eval()
    outs = []
    for ex in examples:
        prompt = generation_prompt_engineer(ex, tokenizer)["question"]
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        gen_ids = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # deterministic, matches "mean routing" spirit
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen_ids[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.strip())
    return outs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_shortcode)

    # Same test split the training carved out (first 175 pooled rows, seed 42).
    print(f"[data] loading {args.dataset_shortcode} test split ...", flush=True)
    test_dataset, _ = load_exp_dataset(args.dataset_shortcode, seed=args.seed)
    examples = list(test_dataset)[: args.n]
    print(f"[data] using {len(examples)} test examples", flush=True)

    # (A) BASE model -- no adapter.
    print(f"\n[base] loading base {args.model_shortcode} (no adapter) ...", flush=True)
    base = load_model(args.model_shortcode, device_map=args.device)
    base_gens = generate_for(base, tokenizer, examples, args.device, args.max_new_tokens)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (B) Stage-1 FINE-TUNED model -- base + trained LoRA adapter.
    ft_gens = None
    if not args.base_only:
        print(f"\n[finetuned] loading base + adapter {args.adapter_path} ...", flush=True)
        try:
            ft = load_peft_model_and_adapter(
                args.model_shortcode, args.adapter_path, eval_mode=True, device_map=args.device
            )
            ft_gens = generate_for(ft, tokenizer, examples, args.device, args.max_new_tokens)
            del ft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[finetuned] SKIPPED -- could not load adapter: {e!r}")

    # ---- report ----
    W = 100
    for i, ex in enumerate(examples):
        print("=" * W)
        print(f"EXAMPLE {i + 1}")
        print("-" * W)
        print("QUESTION:\n" + textwrap.fill(ex["question"], W - 4))
        print("\nGOLD Explanation 1:\n" + textwrap.fill(ex["answer"].strip(), W - 4))
        print("\n(A) BASE MODEL (no Stage-1 adapter):\n" + textwrap.fill(base_gens[i], W - 4))
        if ft_gens is not None:
            print("\n(B) STAGE-1 FINE-TUNED:\n" + textwrap.fill(ft_gens[i], W - 4))
        print()
    print("=" * W)
    print("[done] Eyeball (A) vs GOLD: is the base model already fluent + on-topic?")
    print("       If yes, Stage-1 SFT is contributing little to task competence.")


if __name__ == "__main__":
    main()
