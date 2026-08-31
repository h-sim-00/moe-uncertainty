"""Test whether exp4-train-ans final-position ILV ranks wrong answers on OOD data.

This is NOT the usual ID-vs-OOD experiment. Every example is already drawn
from one of Albus's OOD evaluation datasets. Within each dataset we ask:

    does a larger final answer-predictive ILV indicate that the model's
    predicted answer letter is wrong?

The exp4-train-ans model is reconstructed from:

* Stage-1 adapter: ``adapters/granite-obqa-ansmask``
* Original beta-0.01 FCVR: ``ansmask-pretrained-prior-beta0.01``
* FCVR layers: 5,6,7,8,19,20,28,29,30,31
* inference: S=35, original MCQ prompt, final prompt position

The original exp4 FCVR is intentionally used here. Its KL reduction included
padding, so the report labels this provenance explicitly; the later
``ansmask-klattn`` weights are a different experiment.

Primary analysis
----------------
For each dataset independently:

* label 1 = the predicted answer letter is wrong;
* score = final-position ILV, averaged over the ten FCVR layers;
* AUROC > 0.5 and Spearman rho > 0 mean high ILV is associated with error;
* score direction is fixed before evaluation and is never flipped;
* 95% CIs use class-stratified example bootstrap resampling.

An equal-weight macro AUROC/rho is also reported across the OOD datasets. It is
preferred to a pooled correlation because pooling can confound correctness with
dataset-specific ILV scale. Per-dataset results remain the primary evidence.

Default Albus OOD sets
----------------------
``arc_e arc_c medmcqa_med mmlu_law``. These are the same processed 500-example
test sets used by ``evaluate_fcvr.py`` and the exp4 OOD reports.

Quail usage (repository root, ``moe_env`` active)::

    python evaluate_exp4_wrong_answer_ood.py

Smoke test::

    python evaluate_exp4_wrong_answer_ood.py \
      --n_per_dataset 25 --n_boot 100 --tag smoke-exp4-wrong-ood

Outputs
-------
``results/exp4_wrong_answer_ood/<tag>_test_data-s42_mc-s42.{json,md}``
and a matching ``_perexample.jsonl``. Existing outputs are never overwritten
unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from evaluate_fcvr import prepare_model_fcvr
from model import load_peft_model_and_adapter, load_tokenizer
from utils import (
    calculate_accuracy,
    calculate_ece_mce,
    calculate_nll,
    load_exp_dataset,
    multiple_choice_prompt_engineer,
    setup_environment,
)


DEFAULT_DATASETS = ["arc_e", "arc_c", "medmcqa_med", "mmlu_law"]
DEFAULT_LAYERS = [5, 6, 7, 8, 19, 20, 28, 29, 30, 31]
DEFAULT_ADAPTER = "adapters/granite-obqa-ansmask"
DEFAULT_RUN_SUFFIX = "ansmask-pretrained-prior-beta0.01"
CHOICES = ["A", "B", "C", "D"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Does exp4-train-ans final-position ILV rank wrong answers within Albus OOD datasets?"
    )
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                   help="Evaluation datasets. Defaults to the four Albus OOD sets.")
    p.add_argument("--include_obqa_id", action="store_true",
                   help="Also report the OBQA test anchor. It is excluded from the OOD macro.")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--n_per_dataset", type=int, default=0,
                   help="First n processed examples per dataset; 0 = all (normally 500).")
    p.add_argument("--data_seed", type=int, default=42,
                   help="Dataset construction/shuffle seed only. Keep fixed across MC repeats.")
    p.add_argument("--sampling_seed", type=int, default=42,
                   help="FCVR stochastic-routing seed only.")
    p.add_argument("--num_samples", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    p.add_argument("--run_suffix", default=DEFAULT_RUN_SUFFIX)
    p.add_argument("--weights_dataset_shortcode", default="obqa",
                   help="Dataset shortcode embedded in the exp4 FCVR checkpoint path.")
    p.add_argument("--swap_layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    p.add_argument("--output_dir", default="results/exp4_wrong_answer_ood")
    p.add_argument("--tag", default="exp4-wrong-answer-albus-ood")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_args(args):
    if args.n_per_dataset < 0:
        raise SystemExit("--n_per_dataset must be >= 0")
    if args.n_boot < 1:
        raise SystemExit("--n_boot must be >= 1")
    if args.num_samples < 1:
        raise SystemExit("--num_samples must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if not args.swap_layers:
        raise SystemExit("--swap_layers cannot be empty")
    args.datasets = list(dict.fromkeys(args.datasets))
    if not args.datasets:
        raise SystemExit("No datasets requested")


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def output_paths(args):
    stem = os.path.join(
        args.output_dir,
        f"{args.tag}_{args.split}_data-s{args.data_seed}_mc-s{args.sampling_seed}",
    )
    return stem + ".json", stem + ".md", stem + "_perexample.jsonl"


def refuse_collisions(paths, overwrite):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing output(s):\n  "
            + "\n  ".join(existing)
            + "\nPass --overwrite only if replacement is intentional."
        )


def expected_weight_paths(args):
    weight_dir = Path("router_weights") / "fcvr" / (
        f"fcvr-{args.model_shortcode}-{args.weights_dataset_shortcode}-{args.run_suffix}"
    )
    return weight_dir, [weight_dir / f"layer_{layer}_weights.pt" for layer in args.swap_layers]


def check_weights(args):
    if not Path(args.adapter_path).is_dir():
        raise SystemExit(
            f"Missing exp4 Stage-1 adapter: {args.adapter_path}\n"
            "Run this on quail-1 (where the exp4 artifacts are stored), or pass --adapter_path."
        )
    weight_dir, files = expected_weight_paths(args)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(
            f"Missing exp4 FCVR weights under {weight_dir}:\n  "
            + "\n  ".join(missing)
            + "\nCheck --run_suffix, --weights_dataset_shortcode, and --swap_layers."
        )


def prepare_args(args):
    """Namespace consumed by evaluate_fcvr.prepare_model_fcvr()."""
    return argparse.Namespace(
        model_shortcode=args.model_shortcode,
        dataset_shortcode=args.weights_dataset_shortcode,
        swap_layers=list(args.swap_layers),
        run_suffix=args.run_suffix,
        prior_source="pretrained",
        num_samples=args.num_samples,
    )


def domain_sampling_seed(base_seed, domain_index):
    """Distinct reproducible MC seed per dataset; independent of data loading."""
    return int(base_seed + domain_index * 1_000_003)


def seed_everything(seed):
    """Seed data-independent inference RNGs on old and new repository branches."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auroc(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(metric, labels, scores, n_boot, seed, stratified=True):
    """Example-level percentile CI; class-stratified for correctness metrics."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    point = metric(y, s)
    if y.size < 3 or not math.isfinite(point):
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "n_boot": 0,
            "n": int(y.size),
        }
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    values = []
    for _ in range(n_boot):
        if stratified and len(neg) and len(pos):
            idx = np.concatenate([
                rng.choice(neg, len(neg), replace=True),
                rng.choice(pos, len(pos), replace=True),
            ])
        else:
            idx = rng.integers(0, len(y), len(y))
        value = metric(y[idx], s[idx])
        if math.isfinite(value):
            values.append(float(value))
    if values:
        lo, hi = np.percentile(values, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": len(values),
        "n": int(y.size),
    }


@torch.no_grad()
def readout_exp4(model, tokenizer, prompts, fcvr_layers, batch_size):
    """Answer probabilities and analytic ILV at the final real prompt token.

    Left padding makes position -1 the final real token for every sequence.
    ILV is tr(LL^T) = ||L||_F^2, averaged across the trained FCVR layers.
    """
    causal_model = model.base_model.model.model
    device = model.device
    choice_ids = [tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
    if any(i is None or i == tokenizer.unk_token_id for i in choice_ids):
        raise ValueError(f"Letters {CHOICES} are not valid single tokens: {choice_ids}")
    choice_ids = torch.tensor(choice_ids, device=device)
    tokenizer.padding_side = "left"

    all_probs, all_ilv = [], []
    for start in tqdm(range(0, len(prompts), batch_size), desc="exp4 final-position read-out"):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)
        bsz, seq_len = inputs["input_ids"].shape
        if not bool(inputs["attention_mask"][:, -1].all()):
            raise RuntimeError("Left-padding invariant failed: final column must contain real tokens")
        logits = model(**inputs).logits
        probs = F.softmax(logits[:, -1, :][:, choice_ids].float(), dim=1)
        all_probs.append(probs.cpu())

        per_layer = []
        for layer_index in fcvr_layers:
            router = causal_model.layers[layer_index].block_sparse_moe.router
            L = router.last_cholesky_factor
            experts = L.shape[-1]
            L_last = L.view(bsz, seq_len, experts, experts)[:, -1, :, :]
            per_layer.append((L_last.float() ** 2).sum(dim=(-1, -2)))
        all_ilv.append(torch.stack(per_layer, dim=0).mean(dim=0).cpu())

    return {
        "probs": torch.cat(all_probs),
        "ilv_last": torch.cat(all_ilv).numpy(),
    }


def average_ranks(values):
    """One-based average ranks with exact tie handling, dependency-free."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman_binary(labels, scores):
    """Spearman correlation of binary wrong label with score ranks."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 3 or np.unique(y).size < 2 or np.unique(s).size < 2:
        return float("nan")
    ry, rs = average_ranks(y), average_ranks(s)
    return float(np.corrcoef(ry, rs)[0, 1])


def mean_wrong_minus_correct(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if not np.any(y == 0) or not np.any(y == 1):
        return float("nan")
    return float(np.mean(s[y == 1]) - np.mean(s[y == 0]))


def metric_ci(metric, labels, scores, args, seed_offset=0):
    return bootstrap_ci(
        metric,
        labels,
        scores,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed + seed_offset,
        stratified=True,
    )


def signal_summary(labels_wrong, scores, args, seed_offset=0):
    y = np.asarray(labels_wrong, dtype=int)
    s = np.asarray(scores, dtype=float)
    roc = metric_ci(auroc, y, s, args, seed_offset)
    pr = metric_ci(auprc, y, s, args, seed_offset + 1)
    rho = metric_ci(spearman_binary, y, s, args, seed_offset + 2)
    diff = metric_ci(mean_wrong_minus_correct, y, s, args, seed_offset + 3)
    return {
        "auroc_wrong": roc,
        "auprc_wrong": pr,
        "spearman_wrong": rho,
        "rank_biserial": {
            "point": float(2 * roc["point"] - 1) if math.isfinite(roc["point"]) else float("nan"),
            "lo": float(2 * roc["lo"] - 1) if math.isfinite(roc["lo"]) else float("nan"),
            "hi": float(2 * roc["hi"] - 1) if math.isfinite(roc["hi"]) else float("nan"),
        },
        "mean_correct": float(np.mean(s[y == 0])) if np.any(y == 0) else float("nan"),
        "mean_wrong": float(np.mean(s[y == 1])) if np.any(y == 1) else float("nan"),
        "mean_wrong_minus_correct": diff,
    }


def verdict(ilv):
    roc = ilv["auroc_wrong"]
    rho = ilv["spearman_wrong"]
    if roc["lo"] > 0.5 and rho["lo"] > 0:
        return "SUPPORTS high ILV -> wrong"
    if roc["hi"] < 0.5 and rho["hi"] < 0:
        return "INVERTED: high ILV -> correct"
    return "INCONCLUSIVE"


def evaluate_dataset(args, model, tokenizer, fcvr_layers, code, domain_index):
    rows = list(load_exp_dataset(code, seed=args.data_seed, split=args.split))
    if args.n_per_dataset:
        rows = rows[: args.n_per_dataset]
    if not rows:
        raise RuntimeError(f"{code}/{args.split} produced no examples")

    processed = [multiple_choice_prompt_engineer(row, tokenizer=tokenizer) for row in rows]
    prompts = [row["question"] for row in processed]
    golds = [row["answer"] for row in processed]
    metas = [{"id": row.get("id")} for row in rows]
    if any(gold not in CHOICES for gold in golds):
        raise ValueError(f"{code}: gold answers must be in {CHOICES}")
    prompt_lengths = np.asarray(
        [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts], dtype=int
    )

    mc_seed = domain_sampling_seed(args.sampling_seed, domain_index)
    torch.manual_seed(mc_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mc_seed)
    raw = readout_exp4(model, tokenizer, prompts, fcvr_layers, args.batch_size)

    probs = raw["probs"]
    labels = torch.tensor([CHOICES.index(gold) for gold in golds])
    preds = probs.argmax(dim=1)
    correct = (preds == labels).numpy()
    wrong = (~correct).astype(int)
    letter_entropy = torch.distributions.Categorical(probs=probs).entropy().numpy()
    one_minus_maxprob = (1.0 - probs.max(dim=1).values).numpy()
    ilv = np.asarray(raw["ilv_last"], dtype=float)

    for name, values in {
        "ilv_last": ilv,
        "letter_entropy": letter_entropy,
        "one_minus_maxprob": one_minus_maxprob,
    }.items():
        if values.shape != (len(rows),):
            raise RuntimeError(f"{code}/{name}: shape {values.shape}, expected {(len(rows),)}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{code}/{name}: non-finite values")

    acc = float(calculate_accuracy(preds, labels))
    nll = float(calculate_nll(probs, labels))
    ece, mce = calculate_ece_mce(probs, labels)

    summary = {
        "n": len(rows),
        "n_wrong": int(wrong.sum()),
        "wrong_rate": float(wrong.mean()),
        "ACC": acc,
        "NLL": nll,
        "ECE": float(ece),
        "MCE": float(mce),
        "mc_seed": mc_seed,
        "signals": {
            "ilv_last": signal_summary(wrong, ilv, args, domain_index * 100),
            "letter_entropy": signal_summary(wrong, letter_entropy, args, domain_index * 100 + 10),
            "one_minus_maxprob": signal_summary(wrong, one_minus_maxprob, args, domain_index * 100 + 20),
        },
    }
    summary["verdict"] = verdict(summary["signals"]["ilv_last"])

    per_example = []
    for i, meta in enumerate(metas):
        source_id = meta.get("id")
        per_example.append({
            "dataset": code,
            "row_index": i,
            "source_id": str(source_id if source_id is not None else f"{code}_{i}"),
            "gold_letter": golds[i],
            "pred_letter": CHOICES[int(preds[i])],
            "correct": bool(correct[i]),
            "wrong": int(wrong[i]),
            "probs": [float(x) for x in probs[i]],
            "p_gold": float(probs[i, labels[i]]),
            "maxprob": float(probs[i].max()),
            "letter_entropy": float(letter_entropy[i]),
            "one_minus_maxprob": float(one_minus_maxprob[i]),
            "ilv_last": float(ilv[i]),
            "prompt_tokens": int(prompt_lengths[i]),
            "mc_seed": mc_seed,
        })

    arrays = {
        "wrong": wrong,
        "ilv_last": ilv,
        "letter_entropy": letter_entropy,
        "one_minus_maxprob": one_minus_maxprob,
    }
    return summary, per_example, arrays


def macro_metric_ci(blocks, metric, args, seed_offset=0):
    """Equal-weight macro metric; resample correct/wrong within each dataset."""
    points = [metric(y, s) for y, s in blocks]
    finite_points = [x for x in points if math.isfinite(x)]
    point = float(np.mean(finite_points)) if finite_points else float("nan")
    rng = np.random.default_rng(args.bootstrap_seed + seed_offset)
    values = []
    for _ in range(args.n_boot):
        sampled = []
        for y, s in blocks:
            y = np.asarray(y, dtype=int)
            s = np.asarray(s, dtype=float)
            neg = np.flatnonzero(y == 0)
            pos = np.flatnonzero(y == 1)
            if not len(neg) or not len(pos):
                continue
            idx = np.concatenate([
                rng.choice(neg, len(neg), replace=True),
                rng.choice(pos, len(pos), replace=True),
            ])
            value = metric(y[idx], s[idx])
            if math.isfinite(value):
                sampled.append(value)
        if sampled:
            values.append(float(np.mean(sampled)))
    if values:
        lo, hi = np.percentile(values, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": len(values),
        "n_domains": len(finite_points),
    }


def macro_summary(args, arrays_by_dataset, ood_datasets):
    blocks_by_signal = {}
    for signal in ("ilv_last", "letter_entropy", "one_minus_maxprob"):
        blocks_by_signal[signal] = [
            (arrays_by_dataset[code]["wrong"], arrays_by_dataset[code][signal])
            for code in ood_datasets
        ]
    out = {}
    for idx, (signal, blocks) in enumerate(blocks_by_signal.items()):
        out[signal] = {
            "macro_auroc_wrong": macro_metric_ci(blocks, auroc, args, idx * 10),
            "macro_spearman_wrong": macro_metric_ci(blocks, spearman_binary, args, idx * 10 + 1),
        }
    ilv_roc = out["ilv_last"]["macro_auroc_wrong"]
    ilv_rho = out["ilv_last"]["macro_spearman_wrong"]
    if ilv_roc["lo"] > 0.5 and ilv_rho["lo"] > 0:
        out["verdict"] = "SUPPORTS high ILV -> wrong across OOD datasets"
    elif ilv_roc["hi"] < 0.5 and ilv_rho["hi"] < 0:
        out["verdict"] = "INVERTED across OOD datasets"
    else:
        out["verdict"] = "INCONCLUSIVE across OOD datasets"
    return out


def f3(value):
    return "nan" if value is None or not math.isfinite(float(value)) else f"{float(value):.3f}"


def ci_text(result):
    return f"{f3(result['point'])} [{f3(result['lo'])}, {f3(result['hi'])}]"


def build_markdown(args, summary):
    lines = [
        "# exp4-train-ans: does final-position ILV indicate a wrong answer on OOD data?",
        "",
        "This is a within-dataset correctness test, not an ID-vs-OOD test. "
        "Positive label = wrong answer; larger score is fixed in advance to mean more likely wrong; "
        "scores are never flipped.",
        "",
        f"Weights: `{args.adapter_path}` + original exp4 FCVR `{args.run_suffix}`; "
        f"layers {args.swap_layers}; S={args.num_samples}; split={args.split}; "
        f"data seed={args.data_seed}; sampling seed={args.sampling_seed}; n_boot={args.n_boot}.",
        "",
        "Important provenance: these are the original exp4 weights whose FCVR KL reduction included padding, "
        "not the later `ansmask-klattn` retrain.",
        "",
        "## Per-dataset results",
        "",
        "| Dataset | n | wrong | ACC | mean ILV correct | mean ILV wrong | ILV AUROC(wrong) [95% CI] | ILV Spearman(wrong) [95% CI] | Entropy AUROC(wrong) | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for code in summary["evaluated_order"]:
        d = summary["datasets"][code]
        ilv = d["signals"]["ilv_last"]
        ent = d["signals"]["letter_entropy"]
        lines.append(
            f"| {code} | {d['n']} | {d['n_wrong']} | {f3(d['ACC'])} | "
            f"{f3(ilv['mean_correct'])} | {f3(ilv['mean_wrong'])} | "
            f"{ci_text(ilv['auroc_wrong'])} | {ci_text(ilv['spearman_wrong'])} | "
            f"{ci_text(ent['auroc_wrong'])} | {d['verdict']} |"
        )

    macro = summary["ood_macro"]
    lines.extend([
        "",
        "## Equal-weight macro across the Albus OOD datasets",
        "",
        "| Signal | Macro AUROC(wrong) [95% CI] | Macro Spearman(wrong) [95% CI] |",
        "|---|---:|---:|",
    ])
    for signal in ("ilv_last", "letter_entropy", "one_minus_maxprob"):
        block = macro[signal]
        lines.append(
            f"| {signal} | {ci_text(block['macro_auroc_wrong'])} | "
            f"{ci_text(block['macro_spearman_wrong'])} |"
        )
    lines.extend([
        "",
        f"**Macro verdict:** {macro['verdict']}.",
        "",
        "## Interpretation rules",
        "",
        "- ILV AUROC above 0.5 means a randomly selected wrong answer tends to have higher ILV than a randomly selected correct answer from the same dataset.",
        "- Positive Spearman means larger ILV ranks are associated with the binary wrong-answer label.",
        "- A dataset supports the hypothesis only when the AUROC CI is above 0.5 and the Spearman CI is above 0. Otherwise it is labelled inconclusive or inverted.",
        "- Per-dataset results are primary. The macro gives every dataset equal weight. Pooling all rows was deliberately avoided because dataset-level ILV scale can create a spurious correctness correlation.",
        "- Bootstrap intervals cover examples from these fixed checkpoints and seeds. They do not measure independent training-seed uncertainty.",
        "",
        f"Per-example values: `{summary['per_example_path']}`.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    validate_args(args)
    paths = output_paths(args)
    refuse_collisions(paths, args.overwrite)
    check_weights(args)
    os.makedirs(args.output_dir, exist_ok=True)

    setup_environment()
    seed_everything(args.sampling_seed)
    model = load_peft_model_and_adapter(
        args.model_shortcode, adapter_path=args.adapter_path, device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)
    model = prepare_model_fcvr(model, prepare_args(args))
    fcvr_layers = sorted(args.swap_layers)

    requested = list(args.datasets)
    evaluated_order = (["obqa"] if args.include_obqa_id and "obqa" not in requested else []) + requested
    datasets_summary = {}
    arrays_by_dataset = {}
    per_example = []
    try:
        for domain_index, code in enumerate(evaluated_order):
            print("\n" + "=" * 80)
            print(f"Evaluating {code}/{args.split} with exp4 weights")
            print("=" * 80)
            d_summary, d_rows, d_arrays = evaluate_dataset(
                args, model, tokenizer, fcvr_layers, code, domain_index
            )
            datasets_summary[code] = d_summary
            arrays_by_dataset[code] = d_arrays
            per_example.extend(d_rows)
            ilv = d_summary["signals"]["ilv_last"]
            print(
                f"{code}: n={d_summary['n']} wrong={d_summary['n_wrong']} "
                f"ACC={d_summary['ACC']:.3f} ILV AUROC={ci_text(ilv['auroc_wrong'])} "
                f"rho={ci_text(ilv['spearman_wrong'])} -> {d_summary['verdict']}"
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ood_datasets = [code for code in requested if code in arrays_by_dataset]
    macro = macro_summary(args, arrays_by_dataset, ood_datasets)
    json_path, md_path, per_path = paths
    summary = {
        "question": "Within each OOD dataset, does higher final-position ILV indicate a wrong answer?",
        "positive_label": "wrong answer = 1",
        "score_direction": "higher score fixed a priori to mean more likely wrong; never flipped",
        "config": vars(args),
        "git_revision": git_revision(),
        "weights": {
            "stage1_adapter": args.adapter_path,
            "fcvr_run_suffix": args.run_suffix,
            "weights_dataset_shortcode": args.weights_dataset_shortcode,
            "fcvr_layers": list(fcvr_layers),
            "original_exp4_padding_inclusive_kl": True,
        },
        "evaluated_order": evaluated_order,
        "albus_ood_datasets": ood_datasets,
        "datasets": datasets_summary,
        "ood_macro": macro,
        "per_example_path": per_path,
    }

    with open(per_path, "w") as stream:
        for row in per_example:
            stream.write(json.dumps(row) + "\n")
    with open(json_path, "w") as stream:
        json.dump(summary, stream, indent=2)
    with open(md_path, "w") as stream:
        stream.write(build_markdown(args, summary))

    print("\n" + "=" * 80)
    print(f"Macro ILV AUROC: {ci_text(macro['ilv_last']['macro_auroc_wrong'])}")
    print(f"Macro ILV rho:   {ci_text(macro['ilv_last']['macro_spearman_wrong'])}")
    print(f"Verdict:         {macro['verdict']}")
    print(f"Saved: {json_path}")
    print(f"       {md_path}")
    print(f"       {per_path}")


if __name__ == "__main__":
    main()
