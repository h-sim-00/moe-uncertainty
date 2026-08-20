"""Letter read-out evaluation for the MedMCQA comparison (branch MedMCQA-comparison).

Question answered: does training on the explanation AS WELL as the answer
(arm B, target_mode=answer_explanation) change the uncertainty signal on the
ANSWER compared with answer-only training (arm A, target_mode=letter)?

So everything here is measured on the answer letter and nothing is generated:
  * prompt  = the comparison prompt, IDENTICAL for both arms (COMPARISON_SYSTEM_
              INSTRUCTION + the `letter_question` inner text, ends in "Answer:")
              + assistant header; --system_prompt mcq selects the original MCQ
              instruction for models trained with it (OBQA reference rows)
  * read-out= next-token logits at the last prompt position, restricted to the
              four bare letter tokens A/B/C/D, softmax -> 4-way distribution
              (identical to utils.get_model_predictions / Albus's evaluate.py)
  * metrics = ACC / NLL / ECE / MCE (utils.metrics) on that distribution
  * signals, per example, at the same position:
      letter_entropy     Shannon entropy of the 4-way distribution
      one_minus_maxprob  1 - max prob
      gate_entropy_last  entropy of the routing distribution actually used
                         (softmax of the router logits), averaged over ALL MoE
                         layers (and, for FCVR, also over the FCVR layers only)
      ilv_last           [fcvr only] Inf-Logit-Var = tr(L L^T) = ||L||_F^2 of the
                         FCVR posterior Cholesky factor, mean over FCVR layers
    and the correct-vs-wrong AUROC (label 1 = wrong; higher signal => wrong) of
    each signal with a bootstrap CI (uq_stats).
Per-example rows are written to <out_dir>/<tag>_<split>_perexample.jsonl and the
aggregate to <out_dir>/<tag>_<split>.json (refuses to overwrite unless --overwrite).

Methods (rows of the comparison table):
  zero_shot   untuned base model (no adapter)
  kvq_ft      Stage-1 adapter only (stock routers)
  det         Stage-1 adapter + MAP-tuned routers (router-tuning.py, --map_suffix)
  fcvr        Stage-1 adapter + FCVR routers on --swap_layers (--run_suffix,
              --prior_source, S=--num_samples MC samples), as evaluate_fcvr.py

Usage:
  python evaluate_letter.py --dataset_shortcode medmcqa_gen --split val --method zero_shot --tag zero-shot --n 50
  python evaluate_letter.py --dataset_shortcode medmcqa_gen --split test --method fcvr --kvq_adapter_path adapters/granite-medmcqa_gen-armB-ansexp \
      --swap_layers 5 6 7 8 19 20 28 29 30 31 --run_suffix armB-ansexp-pretrained-prior-beta0.01 \
      --prior_source pretrained --tag armB-ansexp_fcvr_S35-s42
Works for plain MCQA datasets too (obqa, ...: prompt = example['question'],
answer = example['answer']) -- used for the OBQA-trained reference rows.
"""
import argparse
import json
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import setup_environment, seed_everything, load_exp_dataset
from utils import calculate_accuracy, calculate_ece_mce, calculate_nll
from utils.prompt import multiple_choice_prompt_engineer, SYSTEM_INSTRUCTIONS
from model import load_peft_model_and_adapter, load_tokenizer
from model.adapters import granite_adapter
from uq_stats import auroc, bootstrap_ci

CHOICES = ["A", "B", "C", "D"]
SIGNALS = ["letter_entropy", "one_minus_maxprob", "gate_entropy_last", "gate_entropy_last_fcvr", "ilv_last"]


def parse_args():
    p = argparse.ArgumentParser(description="Letter read-out (ACC/NLL/ECE/MCE + UQ signals + AUROC(wrong)).")
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="medmcqa_gen")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--system_prompt", type=str, default="comparison", choices=sorted(SYSTEM_INSTRUCTIONS),
                   help="'comparison' (default): the shared comparison-arm instruction; 'mcq': the original "
                        "letter-only instruction (for models trained with it, e.g. the OBQA reference).")
    p.add_argument("--method", type=str, required=True, choices=["zero_shot", "kvq_ft", "det", "fcvr"])
    p.add_argument("--kvq_adapter_path", type=str, default=None, help="Stage-1 adapter (kvq_ft / det / fcvr).")
    p.add_argument("--map_suffix", type=str, default=None,
                   help="[det, or fcvr with --prior_source map] suffix of router_weights/base/<model>_<dataset>-<suffix>.")
    p.add_argument("--swap_layers", type=int, nargs="*", default=None, help="[fcvr] layers carrying trained FCVR routers.")
    p.add_argument("--run_suffix", type=str, default=None, help="[fcvr] suffix of the FCVR weights dir.")
    p.add_argument("--prior_source", type=str, default="pretrained", choices=["map", "pretrained"],
                   help="[fcvr] must match the training run.")
    p.add_argument("--weights_dataset_shortcode", type=str, default=None,
                   help="[det/fcvr] dataset the routers were TRAINED on, if different from --dataset_shortcode "
                        "(e.g. obqa weights evaluated on medmcqa_gen); selects router_weights/.../*-<this>-*.")
    p.add_argument("--num_samples", type=int, default=35, help="[fcvr] MC samples at inference (S).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n", type=int, default=0, help="Evaluate only the first n examples (0 = all). Smoke tests.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--tag", type=str, required=True, help="Output file stem (e.g. armA-letter_fcvr_S35-s42).")
    p.add_argument("--out_dir", type=str, default="results/letter_eval")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# model preparation
# ---------------------------------------------------------------------------
def prepare(args):
    adapter = None if args.method == "zero_shot" else args.kvq_adapter_path
    if args.method != "zero_shot" and not adapter:
        raise SystemExit(f"--method {args.method} needs --kvq_adapter_path")
    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=adapter, device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)
    fcvr_layers = []
    # The router loaders build their paths from args.dataset_shortcode; for the
    # reference rows (OBQA-trained weights evaluated on MedMCQA) point them at the
    # dataset the weights were trained on.
    wargs = argparse.Namespace(**vars(args))
    wargs.dataset_shortcode = args.weights_dataset_shortcode or args.dataset_shortcode
    if args.method == "det":
        if not args.map_suffix:
            print("WARNING: --method det without --map_suffix -> loading the UNSUFFIXED MAP routers "
                  f"router_weights/base/{wargs.model_shortcode}_{wargs.dataset_shortcode}")
        model = granite_adapter.load_granite_map_routers(model, args=wargs)
    elif args.method == "fcvr":
        if not args.swap_layers or not args.run_suffix:
            raise SystemExit("--method fcvr needs --swap_layers and --run_suffix")
        from evaluate_fcvr import prepare_model_fcvr   # faithful reconstruction (MAP/pretrained prior + FCVR weights + S)
        model = prepare_model_fcvr(model, wargs)
        fcvr_layers = sorted(args.swap_layers)
    model.eval()
    return model, tokenizer, fcvr_layers


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def build_prompts(dataset, tokenizer, system_prompt):
    """-> (prompt_texts, gold_letters, metas). medmcqa_gen/medexqa rows carry
    `letter_question`/`gold_letter`; plain MCQA rows use question/answer."""
    system = SYSTEM_INSTRUCTIONS[system_prompt]
    prompts, golds, metas = [], [], []
    for ex in dataset:
        if ex.get("letter_question") and ex.get("gold_letter"):
            inner, gold = ex["letter_question"], ex["gold_letter"]
        else:
            inner, gold = ex["question"], ex["answer"]
        if gold not in CHOICES:
            raise ValueError(f"gold answer {gold!r} not in {CHOICES} (id={ex.get('id')})")
        eng = multiple_choice_prompt_engineer({"question": inner, "answer": gold, "id": ex.get("id")},
                                              tokenizer=tokenizer, system_instruction=system)
        prompts.append(eng["question"])
        golds.append(gold)
        metas.append({"id": ex.get("id"), "subject_name": ex.get("subject_name"), "topic_name": ex.get("topic_name")})
    return prompts, golds, metas


# ---------------------------------------------------------------------------
# router hooks: routing-distribution entropy at every position of every layer
# ---------------------------------------------------------------------------
class GateEntropyRecorder:
    """Forward hooks on every MoE router. The router is called on the flattened
    [bsz*seq, hidden] stream and returns (..., logits) with logits [bsz*seq, E]
    (the routing logits actually used: MC-averaged log-mean-probs for FCVR in
    stochastic read-out, the deterministic logits otherwise). We keep, per
    layer, the entropy of softmax(logits) at the LAST position of each sequence."""

    def __init__(self, causal_model):
        self.layers = list(range(len(causal_model.layers)))
        self.bsz = None
        self.last = {}
        self.handles = [causal_model.layers[l].block_sparse_moe.router.register_forward_hook(self._hook(l))
                        for l in self.layers]

    def _hook(self, l):
        def hook(module, inputs, output):
            logits = output[-1].detach().float()                  # [bsz*seq, E]
            E = logits.shape[-1]
            logits = logits.view(self.bsz, -1, E)[:, -1, :]       # last position per sequence (left padding)
            p = torch.softmax(logits, dim=-1)
            self.last[l] = (-(p * torch.log(p.clamp(min=1e-12))).sum(-1)).cpu()
        return hook

    def begin(self, bsz):
        self.bsz = bsz
        self.last = {}

    def per_layer(self):
        """-> tensor [n_layers, bsz] (layers in order)."""
        return torch.stack([self.last[l] for l in self.layers], dim=0)

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------
@torch.no_grad()
def readout(model, tokenizer, prompts, fcvr_layers, batch_size):
    causal_model = model.base_model.model.model
    device = model.device
    choice_ids = [tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
    if any(i is None or i == tokenizer.unk_token_id for i in choice_ids):
        raise ValueError(f"letter tokens {CHOICES} not single tokens in this tokenizer: {choice_ids}")
    choice_ids_t = torch.tensor(choice_ids, device=device)

    tokenizer.padding_side = "left"     # so position -1 is the last real token for every row
    rec = GateEntropyRecorder(causal_model)
    all_probs, gate_all, gate_fcvr, ilv = [], [], [], []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="letter read-out"):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                               max_length=2048, add_special_tokens=False).to(device)
            bsz, seq_len = inputs["input_ids"].shape
            assert bool(inputs["attention_mask"][:, -1].all()), "left padding expected: last column must be real tokens"
            rec.begin(bsz)
            logits = model(**inputs).logits
            choice_logits = logits[:, -1, :][:, choice_ids_t].float()
            all_probs.append(F.softmax(choice_logits, dim=1).cpu())

            ge = rec.per_layer()                                   # [L, bsz]
            gate_all.append(ge.mean(dim=0))
            if fcvr_layers:
                gate_fcvr.append(ge[fcvr_layers].mean(dim=0))
                per_layer = []
                for l in fcvr_layers:
                    router = causal_model.layers[l].block_sparse_moe.router
                    L = router.last_cholesky_factor                # [bsz*seq, E, E]
                    E = L.shape[-1]
                    L_last = L.view(bsz, seq_len, E, E)[:, -1, :, :]
                    per_layer.append((L_last.float() ** 2).sum(dim=(-1, -2)))   # tr(LL^T)
                ilv.append(torch.stack(per_layer, dim=0).mean(dim=0).cpu())
    finally:
        rec.remove()

    probs = torch.cat(all_probs)
    out = {
        "probs": probs,
        "gate_entropy_last": torch.cat(gate_all).numpy(),
        "gate_entropy_last_fcvr": torch.cat(gate_fcvr).numpy() if fcvr_layers else None,
        "ilv_last": torch.cat(ilv).numpy() if fcvr_layers else None,
    }
    return out


def main():
    setup_environment()
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.tag}_{args.split}")
    agg_path, per_path = stem + ".json", stem + "_perexample.jsonl"
    for pth in (agg_path, per_path):
        if os.path.exists(pth) and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {pth} (pass --overwrite)")

    dataset = load_exp_dataset(args.dataset_shortcode, seed=args.seed, split=args.split)
    if args.n:
        dataset = dataset[:args.n]
    print(f"--- {args.dataset_shortcode}/{args.split}: {len(dataset)} examples | method={args.method} "
          f"| system_prompt={args.system_prompt} | seed={args.seed} ---")

    model, tokenizer, fcvr_layers = prepare(args)
    prompts, golds, metas = build_prompts(dataset, tokenizer, args.system_prompt)
    print(f"--- prompt (first example, last 300 chars): ...{prompts[0][-300:]!r}")

    r = readout(model, tokenizer, prompts, fcvr_layers, args.batch_size)
    probs = r["probs"]
    labels = torch.tensor([CHOICES.index(g) for g in golds])
    preds = probs.argmax(dim=1)
    correct = (preds == labels)
    wrong = (~correct).numpy().astype(int)

    acc = calculate_accuracy(preds, labels).item()
    nll = calculate_nll(probs, labels).item()
    ece, mce = calculate_ece_mce(probs, labels)
    ece, mce = float(ece), float(mce)

    letter_entropy = torch.distributions.Categorical(probs=probs).entropy().numpy()
    one_minus_maxprob = (1.0 - probs.max(dim=1).values).numpy()
    signals = {
        "letter_entropy": letter_entropy,
        "one_minus_maxprob": one_minus_maxprob,
        "gate_entropy_last": r["gate_entropy_last"],
        "gate_entropy_last_fcvr": r["gate_entropy_last_fcvr"],
        "ilv_last": r["ilv_last"],
    }
    auroc_wrong = {}
    for name in SIGNALS:
        s = signals.get(name)
        if s is None:
            continue
        ci = bootstrap_ci(auroc, wrong, s, n_boot=args.n_boot, seed=args.seed)
        auroc_wrong[name] = {"auroc": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "n_boot": ci["n_boot"],
                             "mean_correct": float(np.mean(s[wrong == 0])) if (wrong == 0).any() else float("nan"),
                             "mean_wrong": float(np.mean(s[wrong == 1])) if (wrong == 1).any() else float("nan")}

    # per-example rows
    with open(per_path, "w") as f:
        for i, m in enumerate(metas):
            row = dict(m)
            row.update({
                "gold_letter": golds[i], "pred_letter": CHOICES[int(preds[i])], "correct": bool(correct[i]),
                "probs": [float(x) for x in probs[i]],
                "p_gold": float(probs[i, labels[i]]), "maxprob": float(probs[i].max()),
                "letter_entropy": float(letter_entropy[i]),
                "gate_entropy_last": float(r["gate_entropy_last"][i]),
                "gate_entropy_last_fcvr": float(r["gate_entropy_last_fcvr"][i]) if fcvr_layers else None,
                "ilv_last": float(r["ilv_last"][i]) if fcvr_layers else None,
            })
            f.write(json.dumps(row) + "\n")

    summary = {
        "config": vars(args),
        "n": int(len(dataset)),
        "n_wrong": int(wrong.sum()),
        "ACC": acc, "NLL": nll, "ECE": ece, "MCE": mce,
        "mean_maxprob": float(probs.max(dim=1).values.mean()),
        "pred_letter_hist": {c: int((preds == k).sum()) for k, c in enumerate(CHOICES)},
        "gold_letter_hist": {c: int((labels == k).sum()) for k, c in enumerate(CHOICES)},
        "auroc_wrong": auroc_wrong,
        "fcvr_layers": fcvr_layers,
        "per_example_path": per_path,
    }
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.tag} [{args.split}] n={len(dataset)} ===")
    print(f"ACC {acc:.4f} | NLL {nll:.4f} | ECE {ece:.4f} | MCE {mce:.4f} | mean maxprob {summary['mean_maxprob']:.3f}")
    for name, v in auroc_wrong.items():
        flag = "" if v["auroc"] >= 0.5 or math.isnan(v["auroc"]) else "  (INVERTED: lower on wrong)"
        print(f"  AUROC(wrong) {name:24s} {v['auroc']:.4f} [{v['lo']:.3f}, {v['hi']:.3f}]{flag}")
    print(f"saved {agg_path}\n      {per_path}")


if __name__ == "__main__":
    main()
