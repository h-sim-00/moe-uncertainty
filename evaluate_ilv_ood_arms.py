"""Compare Arm A vs Arm B for Albus-style input-level OoD detection.

Scientific question
-------------------
Both FCVR models were trained on the SAME MedMCQA rows and prompt. Arm A's
training loss covered only the answer letter; Arm B's covered the answer letter
plus the gold explanation. Does that target change make the learned
Inf-Logit-Var (ILV) signal better, worse, or indistinguishable for detecting
inputs from unseen domains?

This is deliberately separate from ``evaluate_letter.py``:

* ``evaluate_letter.py`` labels a MedMCQA example WRONG vs CORRECT.
* this script labels a prompt ID (MedMCQA) vs OoD (another dataset), matching
  the use of ILV in VMoER/Albus Table 8.

Protocol
--------
* ID: a registered comparison anchor (``--id_dataset``): frozen ``medmcqa_gen``
  (default) or ``obqa_gen`` (branch OBQA-comparison). The per-dataset arm
  registry (``ARM_SETUP``) carries each arm's adapter/FCVR paths, the dataset
  shortcode its FCVR weights were trained under, and its system prompt.
* OoD: distinct-source datasets. Same-corpus shortcodes (``medmcqa_med`` for a
  MedMCQA anchor, ``obqa`` for an OBQA anchor) are refused because they overlap
  the training distribution.
* Prompt: each arm's own training-time system prompt. For medmcqa_gen both
  arms share the comparison prompt; for obqa_gen arm A keeps the FROZEN
  exp4-train-ans Stage-1 adapter (granite-obqa-ansmask, plain MCQ prompt) with
  FCVR routers retrained under the corrected KL mask (exp4's originals averaged
  padding into the KL; suffix ansmask-klattn-...), so the arms differ in prompt
  as well as target -- the accepted confound of branch OBQA-comparison -- and
  the cross-arm prompt-length guard is only enforced when the arms share a
  system prompt.
* Read-out: final predictive position, averaged across the ten FCVR layers.
* Primary signal: ``ilv_last = mean_layer tr(L L^T) = mean_layer ||L||_F^2``.
* Paper baseline: routing Gate-Entropy on the same FCVR layers.
* Metrics: balanced ID-vs-OoD AUROC and AUPRC. Higher signal is fixed a priori
  to mean OoD; inverted scores are never flipped.
* Comparison: paired, class-stratified example bootstrap of Arm B - Arm A.
  The same ID/OoD prompts and the same resampled indices are used in both arms.

The script writes a JSON summary, a Markdown report, and per-example JSONL.
It refuses to overwrite unless ``--overwrite`` is passed.

Important seed fix
------------------
``--data_seed`` controls dataset construction only. ``--sampling_seed``
controls FCVR Monte Carlo routing only. Never use one flag for both: doing so
silently changes the examples when trying to repeat stochastic inference.

Typical Quail command (from the repository root, ``moe_env`` active)::

    python evaluate_ilv_ood_arms.py \
      --ood_datasets medexqa obqa arc_e arc_c sciq mmlu_law \
      --split test --n_per_domain 500 \
      --data_seed 42 --sampling_seed 42

Quick smoke test::

    python evaluate_ilv_ood_arms.py \
      --ood_datasets medexqa obqa --n_per_domain 25 --n_boot 100 \
      --tag smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from evaluate_letter import prepare, readout
from uq_stats import auprc, auroc, bootstrap_ci
from utils import load_exp_dataset, seed_everything, setup_environment
from utils.prompt import SYSTEM_INSTRUCTIONS, multiple_choice_prompt_engineer


DEFAULT_LAYERS = [5, 6, 7, 8, 19, 20, 28, 29, 30, 31]

# Per-ID-dataset registry of the two comparison arms. Each arm records where its
# Stage-1 adapter and FCVR weights live, which dataset shortcode the FCVR
# weights were trained under (None = the ID dataset itself; selects
# router_weights/fcvr/fcvr-<model>-<that>-<run_suffix>), and which system
# prompt the arm was trained with (key into utils.prompt.SYSTEM_INSTRUCTIONS).
# obqa_gen arm A = FROZEN exp4-train-ans Stage-1 adapter (legacy `obqa`
# shortcode, plain MCQ prompt) + FCVR routers retrained with --kl_mask attention
# (the klattn suffix; exp4's originals averaged padding into the KL).
ARM_SETUP = {
    "medmcqa_gen": {
        "tag": "medmcqa-arms",
        "default_ood": ["medexqa", "obqa", "arc_e", "arc_c", "sciq", "mmlu_law"],
        "ood_description": {
            "medexqa": "near-domain: medical/allied-health, distinct source",
            "obqa": "non-medical commonsense science",
            "arc_e": "non-medical primary science (easy)",
            "arc_c": "non-medical primary science (challenge)",
            "sciq": "non-medical broad science",
            "mmlu_law": "far-domain professional law",
        },
        "arms": {
            "armA-letter": {
                "label": "Arm A: answer-only",
                "adapter": "adapters/granite-medmcqa_gen-armA-letter",
                "run_suffix": "armA-letter-pretrained-prior-beta0.01",
                "weights_dataset": None,
                "system_prompt": "comparison",
            },
            "armB-ansexp": {
                "label": "Arm B: answer + explanation",
                "adapter": "adapters/granite-medmcqa_gen-armB-ansexp",
                "run_suffix": "armB-ansexp-pretrained-prior-beta0.01",
                "weights_dataset": None,
                "system_prompt": "comparison",
            },
        },
    },
    "obqa_gen": {
        "tag": "obqa-arms",
        "default_ood": ["medexqa", "medmcqa_gen", "arc_e", "arc_c", "sciq", "mmlu_law"],
        "ood_description": {
            "medexqa": "far-domain medical/allied-health",
            "medmcqa_gen": "far-domain medical MCQA",
            "arc_e": "near-domain primary science (easy)",
            "arc_c": "near-domain primary science (challenge)",
            "sciq": "near-domain broad science",
            "mmlu_law": "far-domain professional law",
        },
        "arms": {
            "armA-letter": {
                "label": "Arm A: answer-only (frozen exp4 Stage-1, KL-mask-fixed FCVR)",
                "adapter": "adapters/granite-obqa-ansmask",
                "run_suffix": "ansmask-klattn-pretrained-prior-beta0.01",
                "weights_dataset": "obqa",
                "system_prompt": "mcq",
            },
            "armB-ansexp": {
                "label": "Arm B: answer + explanation (fact1)",
                "adapter": "adapters/granite-obqa_gen-armB-ansexp",
                "run_suffix": "armB-ansexp-pretrained-prior-beta0.01",
                "weights_dataset": None,
                "system_prompt": "comparison",
            },
        },
    },
}

# Shortcodes drawn from the same upstream corpus as an ID anchor: not unseen to
# a model trained on that anchor, so refused as OoD.
SAME_SOURCE = {"medmcqa_gen": {"medmcqa_med", "medmcqa"},
               "obqa_gen": {"obqa", "openbookqa"}}

SIGNALS = ("ilv_last", "gate_entropy_last_fcvr", "letter_entropy", "one_minus_maxprob")


@dataclass
class DomainData:
    code: str
    ids: list[str]
    rows: list[dict]


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired Arm A/B comparison of final-token ILV for ID-vs-OoD detection."
    )
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--id_dataset", default="medmcqa_gen", choices=sorted(ARM_SETUP),
                   help="Training distribution and ID anchor; selects the ARM_SETUP registry entry.")
    p.add_argument("--ood_datasets", nargs="+", default=None,
                   help="Distinct-source OoD datasets. Default: the registry list for --id_dataset. "
                        "Same-corpus shortcodes (SAME_SOURCE) are refused.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n_per_domain", type=int, default=500,
                   help="Maximum examples loaded per domain. Each ID/OoD comparison is balanced to the smaller side; 0 = all.")
    p.add_argument("--data_seed", type=int, default=42,
                   help="ONLY dataset construction/split seed. Keep fixed across stochastic repeats.")
    p.add_argument("--sampling_seed", type=int, default=42,
                   help="ONLY FCVR Monte Carlo seed. Vary this while keeping --data_seed fixed.")
    p.add_argument("--num_samples", type=int, default=35, help="FCVR Monte Carlo samples S (paper: 35).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--swap_layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--prior_source", choices=["pretrained", "map"], default="pretrained")
    p.add_argument("--arm_a_adapter", default=None, help="Default: ARM_SETUP entry for --id_dataset.")
    p.add_argument("--arm_b_adapter", default=None, help="Default: ARM_SETUP entry for --id_dataset.")
    p.add_argument("--arm_a_run_suffix", default=None, help="Default: ARM_SETUP entry for --id_dataset.")
    p.add_argument("--arm_b_run_suffix", default=None, help="Default: ARM_SETUP entry for --id_dataset.")
    p.add_argument("--map_suffix", default=None,
                   help="Only used with --prior_source map; must match the router checkpoint.")
    p.add_argument("--output_dir", default="results/ilv_ood_arms")
    p.add_argument("--tag", default=None,
                   help="Output file stem. Default: the registry tag for --id_dataset "
                        "(medmcqa-arms / obqa-arms).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_arm_setup(args):
    """Fill registry defaults for --id_dataset and apply the per-arm CLI
    overrides. Returns (registry entry, {arm_key: arm cfg dict})."""
    setup = ARM_SETUP[args.id_dataset]
    if args.ood_datasets is None:
        args.ood_datasets = list(setup["default_ood"])
    if args.tag is None:
        args.tag = setup["tag"]
    arm_cfgs = {}
    for arm_key, cli in (("armA-letter", "a"), ("armB-ansexp", "b")):
        cfg = dict(setup["arms"][arm_key])
        adapter = getattr(args, f"arm_{cli}_adapter")
        run_suffix = getattr(args, f"arm_{cli}_run_suffix")
        if adapter:
            cfg["adapter"] = adapter
        if run_suffix:
            cfg["run_suffix"] = run_suffix
        arm_cfgs[arm_key] = cfg
    return setup, arm_cfgs


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def validate_args(args):
    if args.model_shortcode != "granite":
        raise SystemExit(
            "The two saved comparison arms and evaluate_fcvr.prepare_model_fcvr are Granite-specific; "
            "--model_shortcode must be granite."
        )
    if args.id_dataset not in ARM_SETUP:
        raise SystemExit(
            f"--id_dataset must be a registered arm comparison {sorted(ARM_SETUP)}. "
            "Use fcvr_input_level_ood_check.py for a generic single-arm anchor."
        )
    if args.n_per_domain < 0:
        raise SystemExit("--n_per_domain must be >= 0")
    if args.n_boot < 1:
        raise SystemExit("--n_boot must be >= 1")
    if not args.swap_layers:
        raise SystemExit("--swap_layers must contain the FCVR-modified layers")

    ood = []
    for code in args.ood_datasets:
        if code == args.id_dataset:
            print(f"SKIP {code}: it is the ID anchor")
            continue
        if code in SAME_SOURCE.get(args.id_dataset, set()):
            raise SystemExit(
                f"ERROR: {code} is sampled from the same upstream corpus as {args.id_dataset}; "
                "it overlaps the training distribution and is not a valid OoD domain."
            )
        if code not in ood:
            ood.append(code)
    if not ood:
        raise SystemExit("No valid --ood_datasets remain")
    args.ood_datasets = ood


def output_paths(args):
    stem = os.path.join(
        args.output_dir,
        f"{args.tag}_{args.split}_data-s{args.data_seed}_mc-s{args.sampling_seed}",
    )
    return stem + ".json", stem + ".md", stem + "_perexample.jsonl"


def check_output_collisions(paths, overwrite):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing output(s):\n  " + "\n  ".join(existing) +
            "\nPass --overwrite only if replacement is intentional."
        )


def load_domains(args) -> dict[str, DomainData]:
    """Load every dataset ONCE with the fixed data seed; both arms reuse rows."""
    out = {}
    for code in [args.id_dataset, *args.ood_datasets]:
        rows = list(load_exp_dataset(code, seed=args.data_seed, split=args.split))
        if args.n_per_domain:
            rows = rows[:args.n_per_domain]
        if not rows:
            raise RuntimeError(f"{code}/{args.split} produced no examples")
        ids = [str(r.get("id", f"{code}_{i}")) for i, r in enumerate(rows)]
        if len(set(ids)) != len(ids):
            # Some legacy loaders create short random IDs and can collide. The
            # fixed row index is the actual pairing key; retain the source ID as
            # readable metadata while making that key unique.
            print(f"WARNING: {code}/{args.split} has duplicate source ids; appending stable row indices")
            ids = [f"{source_id}#row{i}" for i, source_id in enumerate(ids)]
        out[code] = DomainData(code=code, ids=ids, rows=rows)
        print(f"Loaded {code}/{args.split}: {len(rows)} rows (data_seed={args.data_seed})")
    return out


def prompts_for_rows(rows: list[dict], tokenizer, system_instruction) -> list[str]:
    """Render one arm's prompt for ID and every OoD set (the same instruction
    is used for every domain within an arm).

    Gold labels are intentionally unnecessary for OoD detection. Generation
    datasets use their MCQA ``letter_question`` field when available; ordinary
    MCQA datasets already expose the canonical ``Question/Choices/Answer:``
    string in ``question``.
    """
    prompts = []
    for i, ex in enumerate(rows):
        inner = ex.get("letter_question") or ex.get("question")
        if not inner:
            raise ValueError(f"row {i} id={ex.get('id')!r} has no question text")
        engineered = multiple_choice_prompt_engineer(
            {"question": inner, "answer": "A", "id": ex.get("id", str(i))},
            tokenizer=tokenizer,
            system_instruction=system_instruction,
        )
        prompts.append(engineered["question"])
    return prompts


def make_prepare_args(args, adapter_path, run_suffix, weights_dataset=None):
    """Namespace expected by evaluate_letter.prepare/evaluate_fcvr."""
    return argparse.Namespace(
        model_shortcode=args.model_shortcode,
        dataset_shortcode=args.id_dataset,
        method="fcvr",
        kvq_adapter_path=adapter_path,
        map_suffix=args.map_suffix,
        swap_layers=list(args.swap_layers),
        run_suffix=run_suffix,
        prior_source=args.prior_source,
        weights_dataset_shortcode=weights_dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


def domain_sampling_seed(base_seed: int, domain_index: int) -> int:
    """Stable distinct seed per domain; identical across arms."""
    return int(base_seed + domain_index * 1_000_003)


def evaluate_arm(args, domains, arm_key, cfg):
    """Load one arm (registry cfg dict), score every fixed domain, then
    release GPU memory."""
    print("\n" + "=" * 80)
    print(f"{cfg['label']} | adapter={cfg['adapter']} | suffix={cfg['run_suffix']} | "
          f"weights_ds={cfg['weights_dataset'] or args.id_dataset} | system_prompt={cfg['system_prompt']}")
    print("=" * 80)

    seed_everything(args.sampling_seed)
    model, tokenizer, fcvr_layers = prepare(
        make_prepare_args(args, cfg["adapter"], cfg["run_suffix"], cfg["weights_dataset"]))
    if sorted(fcvr_layers) != sorted(args.swap_layers):
        raise RuntimeError(f"loaded FCVR layers {fcvr_layers} != requested {sorted(args.swap_layers)}")

    system_instruction = SYSTEM_INSTRUCTIONS[cfg["system_prompt"]]
    arm = {}
    try:
        for domain_index, (code, d) in enumerate(domains.items()):
            prompts = prompts_for_rows(d.rows, tokenizer, system_instruction)
            lengths = np.asarray([
                len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts
            ], dtype=float)

            # Common random numbers: reset to the same domain-specific seed for
            # each arm. Data never depends on this seed.
            mc_seed = domain_sampling_seed(args.sampling_seed, domain_index)
            torch.manual_seed(mc_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(mc_seed)
            raw = readout(model, tokenizer, prompts, fcvr_layers, args.batch_size)

            probs = raw["probs"].numpy()
            letter_entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
            scores = {
                "ilv_last": np.asarray(raw["ilv_last"], dtype=float),
                "gate_entropy_last_fcvr": np.asarray(raw["gate_entropy_last_fcvr"], dtype=float),
                "letter_entropy": letter_entropy.astype(float),
                "one_minus_maxprob": (1.0 - probs.max(axis=1)).astype(float),
                "prompt_len": lengths,
            }
            for name, values in scores.items():
                if values.shape != (len(d.rows),):
                    raise RuntimeError(f"{arm_key}/{code}/{name}: shape {values.shape}, expected {(len(d.rows),)}")
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{arm_key}/{code}/{name}: non-finite values found")
            arm[code] = {"ids": list(d.ids), "scores": scores, "mc_seed": mc_seed}
            print(
                f"{arm_key} {code}: n={len(d.rows)} "
                f"ILV={scores['ilv_last'].mean():.4f} "
                f"GateEnt={scores['gate_entropy_last_fcvr'].mean():.4f} "
                f"prompt_tokens={scores['prompt_len'].mean():.1f}"
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return arm


def paired_arrays(arm, id_code, ood_code, signal):
    """Balanced, aligned arrays for one arm and one ID/OoD comparison."""
    id_block, ood_block = arm[id_code], arm[ood_code]
    n = min(len(id_block["ids"]), len(ood_block["ids"]))
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    scores = np.concatenate([
        id_block["scores"][signal][:n],
        ood_block["scores"][signal][:n],
    ])
    ids = [f"{id_code}:{x}" for x in id_block["ids"][:n]] + [
        f"{ood_code}:{x}" for x in ood_block["ids"][:n]
    ]
    return labels, scores, ids, n


def residualize_length(arm, id_code, ood_code, signal):
    labels, scores, ids, n = paired_arrays(arm, id_code, ood_code, signal)
    _, lengths, length_ids, _ = paired_arrays(arm, id_code, ood_code, "prompt_len")
    if ids != length_ids:
        raise RuntimeError("length/signal row alignment failed")
    x = np.log(np.clip(lengths, 1.0, None))
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
    residual = scores - X @ beta
    var = float(np.var(scores))
    return labels, residual, ids, n, {
        "slope_per_log_token": float(beta[1]),
        "r2": float(1.0 - np.var(residual) / var) if var > 0 else float("nan"),
    }


def metric_summary(labels, scores, metric: Callable, n_boot, seed):
    ci = bootstrap_ci(metric, labels, scores, n_boot=n_boot, seed=seed, stratified=True)
    return {
        "point": float(ci["point"]),
        "lo": float(ci["lo"]),
        "hi": float(ci["hi"]),
        "n_boot": int(ci["n_boot"]),
    }


def paired_bootstrap_delta(labels, scores_a, scores_b, metric: Callable, n_boot=2000, seed=0):
    """Paired class-stratified bootstrap of metric(B) - metric(A)."""
    y = np.asarray(labels, dtype=int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if not (y.shape == a.shape == b.shape):
        raise ValueError(f"shape mismatch labels={y.shape}, A={a.shape}, B={b.shape}")
    if y.size < 4 or np.unique(y).size != 2:
        raise ValueError("paired delta needs at least two classes and four examples")
    point = float(metric(y, b) - metric(y, a))
    rng = np.random.default_rng(seed)
    cls = [np.flatnonzero(y == k) for k in (0, 1)]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
        d = metric(y[idx], b[idx]) - metric(y[idx], a[idx])
        if np.isfinite(d):
            values.append(float(d))
    if not values:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(values, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": len(values)}


def delta_verdict(delta):
    """Direction of B-A; overlap with zero is inconclusive, not proof of equality."""
    if not all(np.isfinite(delta[k]) for k in ("point", "lo", "hi")):
        return "N/A"
    if delta["lo"] > 0:
        return "BETTER with explanations"
    if delta["hi"] < 0:
        return "WORSE with explanations"
    return "NO CLEAR DIFFERENCE"


def signal_quality(summary):
    """Human-readable direction check without ever flipping an inverted AUROC."""
    if summary["hi"] < 0.5:
        return "INVERTED"
    if summary["lo"] > 0.5:
        return "detects OoD"
    return "chance/inconclusive"


def compare_domain(args, arms, ood_code):
    out = {"n_per_class": None, "signals": {}, "length_control": {}}
    for signal in SIGNALS:
        ya, sa, ids_a, n = paired_arrays(arms["armA-letter"], args.id_dataset, ood_code, signal)
        yb, sb, ids_b, n_b = paired_arrays(arms["armB-ansexp"], args.id_dataset, ood_code, signal)
        if n != n_b or ids_a != ids_b or not np.array_equal(ya, yb):
            raise RuntimeError(f"Arm A/B examples are not exactly aligned for {ood_code}/{signal}")
        out["n_per_class"] = n
        a_sum = {
            "auroc": metric_summary(ya, sa, auroc, args.n_boot, args.bootstrap_seed),
            "auprc": metric_summary(ya, sa, auprc, args.n_boot, args.bootstrap_seed),
            "id_mean": float(sa[:n].mean()),
            "ood_mean": float(sa[n:].mean()),
        }
        b_sum = {
            "auroc": metric_summary(yb, sb, auroc, args.n_boot, args.bootstrap_seed),
            "auprc": metric_summary(yb, sb, auprc, args.n_boot, args.bootstrap_seed),
            "id_mean": float(sb[:n].mean()),
            "ood_mean": float(sb[n:].mean()),
        }
        d_roc = paired_bootstrap_delta(ya, sa, sb, auroc, args.n_boot, args.bootstrap_seed)
        d_pr = paired_bootstrap_delta(ya, sa, sb, auprc, args.n_boot, args.bootstrap_seed)
        out["signals"][signal] = {
            "arm_a": a_sum,
            "arm_b": b_sum,
            "delta_b_minus_a": {"auroc": d_roc, "auprc": d_pr},
            "verdict_auroc": delta_verdict(d_roc),
        }

    # Length-residual ILV is secondary: the paper reports raw ILV, while this
    # diagnostic asks whether separation remains after removing a linear trend
    # with log prompt length.
    residual = {}
    arrays = {}
    for arm_key in ("armA-letter", "armB-ansexp"):
        y, s, ids, n, fit = residualize_length(arms[arm_key], args.id_dataset, ood_code, "ilv_last")
        arrays[arm_key] = (y, s, ids)
        residual[arm_key] = {
            "fit": fit,
            "auroc": metric_summary(y, s, auroc, args.n_boot, args.bootstrap_seed),
        }
    ya, sa, ia = arrays["armA-letter"]
    yb, sb, ib = arrays["armB-ansexp"]
    if ia != ib or not np.array_equal(ya, yb):
        raise RuntimeError(f"Arm A/B residual examples are not aligned for {ood_code}")
    residual_delta = paired_bootstrap_delta(ya, sa, sb, auroc, args.n_boot, args.bootstrap_seed)
    residual["delta_b_minus_a"] = residual_delta
    residual["verdict_auroc"] = delta_verdict(residual_delta)
    out["length_control"]["ilv_residual_log_prompt_len"] = residual

    # Reproduce Albus's central comparison: does ILV beat router Gate-Entropy?
    for arm_key in ("armA-letter", "armB-ansexp"):
        y_ilv, s_ilv, ids_ilv, _ = paired_arrays(arms[arm_key], args.id_dataset, ood_code, "ilv_last")
        y_gate, s_gate, ids_gate, _ = paired_arrays(
            arms[arm_key], args.id_dataset, ood_code, "gate_entropy_last_fcvr"
        )
        if ids_ilv != ids_gate or not np.array_equal(y_ilv, y_gate):
            raise RuntimeError(f"ILV/Gate-Entropy rows are not aligned for {arm_key}/{ood_code}")
        d = paired_bootstrap_delta(y_ilv, s_gate, s_ilv, auroc, args.n_boot, args.bootstrap_seed)
        out.setdefault("ilv_minus_gate_entropy", {})[arm_key] = {
            "auroc_delta": d,
            "verdict": (
                "ILV BETTER than Gate-Entropy" if d["lo"] > 0 else
                "ILV WORSE than Gate-Entropy" if d["hi"] < 0 else
                "NO CLEAR DIFFERENCE"
            ),
        }
    return out


def macro_delta(args, arms, ood_codes, signal, metric):
    """Macro-average per-domain paired B-A deltas with joint bootstrap."""
    blocks = []
    points = []
    for code in ood_codes:
        y, a, ids_a, _ = paired_arrays(arms["armA-letter"], args.id_dataset, code, signal)
        yb, b, ids_b, _ = paired_arrays(arms["armB-ansexp"], args.id_dataset, code, signal)
        if ids_a != ids_b or not np.array_equal(y, yb):
            raise RuntimeError(f"macro alignment failed for {code}/{signal}")
        blocks.append((y, a, b))
        points.append(metric(y, b) - metric(y, a))
    rng = np.random.default_rng(args.bootstrap_seed)
    boot = []
    for _ in range(args.n_boot):
        domain_deltas = []
        for y, a, b in blocks:
            cls = [np.flatnonzero(y == k) for k in (0, 1)]
            idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
            domain_deltas.append(metric(y[idx], b[idx]) - metric(y[idx], a[idx]))
        boot.append(float(np.mean(domain_deltas)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = {"point": float(np.mean(points)), "lo": float(lo), "hi": float(hi), "n_boot": len(boot)}
    return {"delta_b_minus_a": d, "verdict": delta_verdict(d)}


def f3(x):
    return "nan" if x is None or not math.isfinite(float(x)) else f"{float(x):.3f}"


def ci_text(d):
    return f"{f3(d['point'])} [{f3(d['lo'])}, {f3(d['hi'])}]"


def build_report(args, summary):
    lines = [
        "# ILV OoD comparison: answer-only vs answer + explanation",
        "",
        f"ID = `{args.id_dataset}`; split = `{args.split}`; data seed = {args.data_seed}; "
        f"FCVR sampling seed = {args.sampling_seed}; S = {args.num_samples}.",
        "",
        "Primary protocol: final predictive position, ILV averaged across FCVR layers; balanced ID/OoD "
        "examples; higher score is fixed a priori to mean OoD. Arm B - Arm A uses a paired, "
        "class-stratified example bootstrap. A CI crossing zero means **no clear difference**, not proof "
        "that the arms are equivalent.",
        "",
        "## Primary result: ILV OoD AUROC",
        "",
        "| OoD domain | n/class | Arm A answer-only | Arm B answer+explanation | Δ B-A [95% CI] | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for code in args.ood_datasets:
        d = summary["domains"][code]
        s = d["signals"]["ilv_last"]
        a, b = s["arm_a"]["auroc"], s["arm_b"]["auroc"]
        delta = s["delta_b_minus_a"]["auroc"]
        lines.append(
            f"| {code} | {d['n_per_class']} | {f3(a['point'])} [{f3(a['lo'])},{f3(a['hi'])}] "
            f"({signal_quality(a)}) | {f3(b['point'])} [{f3(b['lo'])},{f3(b['hi'])}] "
            f"({signal_quality(b)}) | {ci_text(delta)} | **{s['verdict_auroc']}** |"
        )

    macro = summary["macro"]["ilv_last"]["auroc"]
    lines += [
        "",
        f"**Macro-average Arm B - Arm A ILV AUROC:** {ci_text(macro['delta_b_minus_a'])} "
        f"-> **{macro['verdict']}**.",
        "",
        "## Does ILV beat Gate-Entropy?",
        "",
        "| OoD domain | Arm A: ΔAUROC ILV-Gate [95% CI] | Arm A verdict | Arm B: ΔAUROC ILV-Gate [95% CI] | Arm B verdict |",
        "|---|---:|---|---:|---|",
    ]
    for code in args.ood_datasets:
        g = summary["domains"][code]["ilv_minus_gate_entropy"]
        ga, gb = g["armA-letter"], g["armB-ansexp"]
        lines.append(
            f"| {code} | {ci_text(ga['auroc_delta'])} | {ga['verdict']} | "
            f"{ci_text(gb['auroc_delta'])} | {gb['verdict']} |"
        )

    lines += [
        "",
        "## Diagnostics",
        "",
        "The JSON contains AUPRC, answer-letter entropy, 1-max-probability, prompt-length AUROC, and "
        "ILV AUROC after residualising a linear relationship with log prompt length. Raw ILV is primary "
        "because it matches the paper; the length control tests whether simple prompt length may explain separation.",
        "",
        "## Interpretation rule",
        "",
        "- `BETTER with explanations`: the 95% CI for Arm B - Arm A is entirely above zero.",
        "- `WORSE with explanations`: the entire CI is below zero.",
        "- `NO CLEAR DIFFERENCE`: the CI crosses zero. This is inconclusive, not an equivalence proof.",
        "- AUROC below 0.5 is an inverted signal and is never flipped.",
        "",
        f"Git commit: `{summary['config']['git_commit']}`.",
    ]
    return "\n".join(lines) + "\n"


def jsonable_scores(arms):
    out = {}
    for arm_key, domains in arms.items():
        out[arm_key] = {}
        for code, block in domains.items():
            out[arm_key][code] = {
                "ids": block["ids"],
                "mc_seed": block["mc_seed"],
                "scores": {k: v.tolist() for k, v in block["scores"].items()},
            }
    return out


def write_perexample(path, args, arms):
    with open(path, "w", encoding="utf-8") as f:
        for arm_key, domains in arms.items():
            for code, block in domains.items():
                is_id = code == args.id_dataset
                for i, example_id in enumerate(block["ids"]):
                    row = {
                        "arm": arm_key,
                        "dataset": code,
                        "is_ood": not is_id,
                        "id": example_id,
                        "sampling_seed": args.sampling_seed,
                        **{name: float(values[i]) for name, values in block["scores"].items()},
                    }
                    f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    setup, arm_cfgs = resolve_arm_setup(args)
    validate_args(args)
    paths = output_paths(args)
    check_output_collisions(paths, args.overwrite)
    setup_environment()

    print("#" * 80)
    print(f"# Paired {args.id_dataset} arm ILV OoD evaluation")
    print(f"# DATA seed={args.data_seed} (fixed rows) | MC seed={args.sampling_seed} | S={args.num_samples}")
    print(f"# ID={args.id_dataset} | OoD={args.ood_datasets} | split={args.split}")
    print("#" * 80)

    domains = load_domains(args)
    arms = {
        arm_key: evaluate_arm(args, domains, arm_key, arm_cfgs[arm_key])
        for arm_key in ("armA-letter", "armB-ansexp")
    }

    # Absolute pairing guard: raw example IDs must agree; prompt lengths can
    # only be required to match when the arms share a system prompt.
    same_prompt = arm_cfgs["armA-letter"]["system_prompt"] == arm_cfgs["armB-ansexp"]["system_prompt"]
    for code in domains:
        a, b = arms["armA-letter"][code], arms["armB-ansexp"][code]
        if a["ids"] != b["ids"]:
            raise RuntimeError(f"Arm A/B ID ordering differs for {code}")
        if same_prompt and not np.array_equal(a["scores"]["prompt_len"], b["scores"]["prompt_len"]):
            raise RuntimeError(f"Arm A/B prompts differ for {code}")
    if not same_prompt:
        print("NOTE: arms use different system prompts "
              f"(A={arm_cfgs['armA-letter']['system_prompt']}, B={arm_cfgs['armB-ansexp']['system_prompt']}); "
              "cross-arm prompt_len equality not enforced.")

    summary = {
        "config": {
            "git_commit": git_revision(),
            "model_shortcode": args.model_shortcode,
            "id_dataset": args.id_dataset,
            "ood_datasets": args.ood_datasets,
            "ood_description": {c: setup["ood_description"].get(c, "unclassified shift") for c in args.ood_datasets},
            "split": args.split,
            "n_per_domain_cap": args.n_per_domain,
            "data_seed": args.data_seed,
            "sampling_seed": args.sampling_seed,
            "num_samples": args.num_samples,
            "swap_layers": sorted(args.swap_layers),
            "prior_source": args.prior_source,
            "arm_a_adapter": arm_cfgs["armA-letter"]["adapter"],
            "arm_b_adapter": arm_cfgs["armB-ansexp"]["adapter"],
            "arm_a_run_suffix": arm_cfgs["armA-letter"]["run_suffix"],
            "arm_b_run_suffix": arm_cfgs["armB-ansexp"]["run_suffix"],
            "arm_a_weights_dataset": arm_cfgs["armA-letter"]["weights_dataset"],
            "arm_b_weights_dataset": arm_cfgs["armB-ansexp"]["weights_dataset"],
            "arm_a_system_prompt": arm_cfgs["armA-letter"]["system_prompt"],
            "arm_b_system_prompt": arm_cfgs["armB-ansexp"]["system_prompt"],
            "n_boot": args.n_boot,
            "bootstrap_seed": args.bootstrap_seed,
            "prompt": "per-arm system instruction (see arm_*_system_prompt) + MCQA inner prompt; "
                      "identical across domains within an arm",
            "primary_signal": "ilv_last = mean over FCVR layers of tr(LL^T) at final predictive position",
            "sign": "higher score => OoD; never flip inverted AUROC",
        },
        "domains": {},
        "macro": {},
    }
    for code in args.ood_datasets:
        summary["domains"][code] = compare_domain(args, arms, code)

    summary["macro"]["ilv_last"] = {
        "auroc": macro_delta(args, arms, args.ood_datasets, "ilv_last", auroc),
        "auprc": macro_delta(args, arms, args.ood_datasets, "ilv_last", auprc),
    }

    json_path, md_path, per_path = paths
    os.makedirs(args.output_dir, exist_ok=True)
    payload = dict(summary)
    payload["per_example_scores"] = jsonable_scores(arms)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(args, summary))
    write_perexample(per_path, args, arms)

    print("\n" + build_report(args, summary))
    print(f"Saved: {json_path}\n       {md_path}\n       {per_path}")


if __name__ == "__main__":
    main()
