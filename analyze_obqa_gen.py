"""Diagnose the arm-B ILV inversion on OBQA-comparison (branch OBQA-comparison).

Background
----------
evaluate_ilv_ood_arms.py found arm B (answer + fact1 explanation loss) INVERTED
on every OoD domain (macro Delta B-A = -0.475; mmlu_law A 0.943 vs B 0.032) and
letter_eval_report.py found the same flip for wrong-answer detection (ilv_last
AUROC ~.77 vs ~.39). The arms differ in system prompt (mcq vs comparison) as
well as target, and the stored length controls show ILV~log(prompt_len) slopes
of OPPOSITE sign between the arms. This script runs the two diagnostics that
disentangle model from prompt without retraining:

Eval 1 -- prompt crossover (4 cells)
    {arm A, arm B} x {mcq, comparison} system prompts, ILV at the final
    predictive position, ID = obqa_gen test vs the 6 registry OoD domains.
    Per cell x domain: AUROC/AUPRC (+ stratified bootstrap CIs) for the four
    standard signals, wrong-answer AUROC on ID, and length controls
    (pooled log-length residualisation + per-dataset OLS slopes).
    Paired deltas isolate the prompt effect within each model and the model
    effect within each prompt (common MC random numbers per domain).

Eval 2 -- teacher-forced full-sequence ILV trace (2 cells, own training prompt)
    armA x mcq and armB x comparison. Sequence = arm prompt + gold target
    "{letter}\\nExplanation: {explanation}" + EOS, one batch-1 forward per
    example (no padding). ID = obqa_gen test; OoD = the explanation-bearing
    domains (medexqa, medmcqa_gen) traced with their OWN gold letter +
    explanation. Per position (row t predicts token t+1) the NINE metrics:
      ilv_mean / ilv_max / ilv_secondmax (over the FCVR layers), ilv_runmean10
      (mean of ilv_mean over rows t-9..t), ilv_per_layer (full vector, NOT
      stripped), predictive entropy, surprisal of the next token,
      answer_correct (example-level letter probe at the final prompt row),
      rel_expl_pos (0..1 inside the explanation span).
    Sequence-level OoD/wrong-answer scores are aggregates OVER THE EXPLANATION
    REGION (mean / max / second-max / mean of the last 10 explanation rows);
    the final-prompt-position ILV is kept only as a sanity cross-check vs Eval 1.

Conventions copied from evaluate_ilv_ood_arms.py: --data_seed fixes the rows,
--sampling_seed fixes the FCVR Monte Carlo draws, per-domain (Eval 1) and
per-example (Eval 2) seeds are identical across cells (common random numbers),
higher score is fixed a priori to mean OoD / wrong and inverted AUROCs are
never flipped, and the script refuses to overwrite existing outputs.

Typical Quail command (repo root, moe_env active)::

    python analyze_obqa_gen.py --split test --data_seed 42 --sampling_seed 42

Quick smoke test::

    python analyze_obqa_gen.py --n_per_domain 6 --trace_n_id 4 --trace_n_ood 2 \
        --n_boot 50 --tag smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_ilv_ood_arms import (
    ARM_SETUP,
    DEFAULT_LAYERS,
    SAME_SOURCE,
    check_model_registered,
    check_output_collisions,
    ci_text,
    default_tag,
    domain_sampling_seed,
    f3,
    git_revision,
    load_domains,
    make_prepare_args,
    metric_summary,
    paired_arrays,
    paired_bootstrap_delta,
    prompts_for_rows,
    residualize_length,
    resolve_arm_setup,
    signal_quality,
)
from evaluate_letter import CHOICES, prepare, readout
from model.adapters import moe_router
from uq_stats import auprc, auroc, bootstrap_ci, bootstrap_mean_ci
from utils import seed_everything, setup_environment
from utils.data import EXPLANATION_MARKER, build_target_mode_example
from utils.prompt import SYSTEM_INSTRUCTIONS, multiple_choice_prompt_engineer

ARM_KEYS = ("armA-letter", "armB-ansexp")
ARM_SHORT = {"armA-letter": "armA", "armB-ansexp": "armB"}
PROMPT_KEYS = ("mcq", "comparison")
SIGNALS = ("ilv_last", "gate_entropy_last_fcvr", "letter_entropy", "one_minus_maxprob")
# Explanation-bearing datasets: the only ones whose gold letter + explanation
# support a full teacher-forced trace.
TRACE_OOD_CHOICES = ("medexqa", "medmcqa_gen")
CATEGORIES = ("prompt", "prompt_final", "letter", "marker", "explanation", "eos")
# Sequence-level trace scores. Per user decision the ILV aggregates are over the
# EXPLANATION region; prompt_final_ilv is the Eval-1 sanity cross-check only.
TRACE_OOD_AGGS = (
    "expl_ilv_mean", "expl_ilv_max", "expl_ilv_secondmax", "expl_ilv_last10_mean",
    "expl_entropy_mean", "expl_surprisal_mean", "prompt_final_ilv",
)
TRACE_WRONG_AGGS = TRACE_OOD_AGGS + ("letter_ilv", "eos_ilv", "letter_entropy")


def cell_name(arm_key, prompt_key):
    return f"{ARM_SHORT[arm_key]}-{prompt_key}"


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r)
    except Exception:
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])


def parse_args():
    p = argparse.ArgumentParser(
        description="OBQA arm-B ILV inversion diagnostics: prompt crossover + teacher-forced token trace."
    )
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--id_dataset", default="obqa_gen", choices=["obqa_gen"],
                   help="This script is the OBQA-comparison diagnosis; the arm registry entry is obqa_gen.")
    p.add_argument("--ood_datasets", nargs="+", default=None,
                   help="Eval-1 OoD domains. Default: the registry list for obqa_gen "
                        "(medexqa medmcqa_gen arc_e arc_c sciq mmlu_law). Same-corpus shortcodes refused.")
    p.add_argument("--trace_ood_datasets", nargs="+", default=list(TRACE_OOD_CHOICES),
                   choices=list(TRACE_OOD_CHOICES),
                   help="Eval-2 OoD domains (must carry gold letters AND gold explanations).")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n_per_domain", type=int, default=500,
                   help="Eval-1 cap per domain (0 = all); comparisons are balanced to the smaller side.")
    p.add_argument("--trace_n_id", type=int, default=500, help="Eval-2 cap on ID examples (0 = all loaded).")
    p.add_argument("--trace_n_ood", type=int, default=500, help="Eval-2 cap per OoD domain (0 = all loaded).")
    p.add_argument("--stages", default="both", choices=["both", "crossover", "trace"])
    p.add_argument("--data_seed", type=int, default=42,
                   help="ONLY dataset construction/split seed. Keep fixed across stochastic repeats.")
    p.add_argument("--sampling_seed", type=int, default=42,
                   help="ONLY FCVR Monte Carlo seed. Vary this while keeping --data_seed fixed.")
    p.add_argument("--num_samples", type=int, default=35, help="FCVR MC samples S (paper: 35).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--routing", choices=["stochastic", "deterministic"], default="stochastic",
                   help="deterministic = posterior-mean routing (router.deterministic_readout); "
                        "covariance/ILV still recorded. Ablation only; stochastic matches all saved evals.")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--swap_layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--prior_source", choices=["pretrained", "map"], default="pretrained")
    p.add_argument("--map_suffix", default=None)
    p.add_argument("--arm_a_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_a_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--output_dir", default="results/analyze_obqa_gen")
    p.add_argument("--tag", default=None,
                   help="Output stem. Default: obqa-gen-analysis (granite) / obqa-gen-analysis-<model_shortcode> otherwise.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_args(args):
    check_model_registered(args)
    if args.tag is None:
        args.tag = default_tag("obqa-gen-analysis", args.model_shortcode)
    if args.n_per_domain < 0 or args.trace_n_id < 0 or args.trace_n_ood < 0:
        raise SystemExit("--n_per_domain/--trace_n_id/--trace_n_ood must be >= 0")
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
            raise SystemExit(f"ERROR: {code} overlaps the {args.id_dataset} training corpus; not a valid OoD domain.")
        if code not in ood:
            ood.append(code)
    if not ood:
        raise SystemExit("No valid --ood_datasets remain")
    args.ood_datasets = ood
    missing = [c for c in args.trace_ood_datasets if c not in args.ood_datasets]
    if missing and args.stages != "crossover":
        raise SystemExit(f"--trace_ood_datasets {missing} must also be in --ood_datasets (rows are loaded once).")


def output_paths(args):
    stem = os.path.join(
        args.output_dir,
        f"{args.tag}_{args.split}_data-s{args.data_seed}_mc-s{args.sampling_seed}",
    )
    return {
        "json": stem + ".json",
        "md": stem + ".md",
        "crossover": stem + "_crossover_perexample.jsonl",
        "pertoken": stem + "_trace_pertoken.jsonl",
        "perexample": stem + "_trace_perexample.jsonl",
    }


def set_routing_mode(model, fcvr_layers, routing):
    causal_model = model.base_model.model.model
    for l in fcvr_layers:
        moe_router(causal_model.layers[l]).deterministic_readout = (routing == "deterministic")


# ---------------------------------------------------------------------------
# Eval 1: prompt crossover
# ---------------------------------------------------------------------------
def gold_letter_of(ex):
    g = ex.get("gold_letter") or ex.get("answer")
    return g if g in CHOICES else None


def score_cell(model, tokenizer, fcvr_layers, domains, system_instruction, args):
    """One (loaded model, system prompt) cell: final-position readout on every
    fixed domain. Mirrors evaluate_ilv_ood_arms.evaluate_arm's inner loop so a
    single model load can serve both prompts."""
    cell = {}
    for domain_index, (code, d) in enumerate(domains.items()):
        prompts = prompts_for_rows(d.rows, tokenizer, system_instruction)
        lengths = np.asarray(
            [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts], dtype=float)

        # Common random numbers: the same per-domain seed in every cell.
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
                raise RuntimeError(f"{code}/{name}: shape {values.shape}, expected {(len(d.rows),)}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"{code}/{name}: non-finite values found")

        golds = [gold_letter_of(ex) for ex in d.rows]
        preds = [CHOICES[k] for k in probs.argmax(axis=1)]
        correct = [None if g is None else (p == g) for g, p in zip(golds, preds)]
        cell[code] = {
            "ids": list(d.ids), "scores": scores, "mc_seed": mc_seed,
            "gold": golds, "pred": preds, "correct": correct,
        }
        print(f"  {code}: n={len(d.rows)} ILV={scores['ilv_last'].mean():.4f} "
              f"prompt_tokens={scores['prompt_len'].mean():.1f}")
    return cell


def ols_log_length(scores, lengths):
    """Within-dataset OLS of score on log(prompt_len): slope + r^2."""
    x = np.log(np.clip(np.asarray(lengths, dtype=float), 1.0, None))
    y = np.asarray(scores, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    var = float(np.var(y))
    return {"slope_per_log_token": float(beta[1]),
            "r2": float(1.0 - np.var(resid) / var) if var > 0 else float("nan"),
            "n": int(y.size)}


def cell_pair_delta(cells, name_a, name_b, id_code, ood_code, signal, metric, n_boot, seed):
    ya, sa, ids_a, _ = paired_arrays(cells[name_a], id_code, ood_code, signal)
    yb, sb, ids_b, _ = paired_arrays(cells[name_b], id_code, ood_code, signal)
    if ids_a != ids_b or not np.array_equal(ya, yb):
        raise RuntimeError(f"cells {name_a}/{name_b} are not aligned for {ood_code}/{signal}")
    return paired_bootstrap_delta(ya, sa, sb, metric, n_boot, seed)


def macro_cell_delta(cells, name_a, name_b, id_code, ood_codes, signal, metric, n_boot, seed):
    """Macro-average per-domain paired (name_b - name_a) deltas with a joint bootstrap
    (generic-cell version of evaluate_ilv_ood_arms.macro_delta)."""
    blocks, points = [], []
    for code in ood_codes:
        ya, sa, ids_a, _ = paired_arrays(cells[name_a], id_code, code, signal)
        yb, sb, ids_b, _ = paired_arrays(cells[name_b], id_code, code, signal)
        if ids_a != ids_b or not np.array_equal(ya, yb):
            raise RuntimeError(f"macro alignment failed for {code}/{signal}")
        blocks.append((ya, sa, sb))
        points.append(metric(ya, sb) - metric(ya, sa))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        deltas = []
        for y, a, b in blocks:
            cls = [np.flatnonzero(y == k) for k in (0, 1)]
            idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
            deltas.append(metric(y[idx], b[idx]) - metric(y[idx], a[idx]))
        boot.append(float(np.mean(deltas)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(np.mean(points)), "lo": float(lo), "hi": float(hi), "n_boot": len(boot)}


def crossover_stats(args, cells):
    out = {"cells": {}, "deltas": {}, "wrong_answer_id": {}}
    for name, cell in cells.items():
        cell_out = {"domains": {}, "length_slopes": {}}
        for code in args.ood_datasets:
            dom = {"n_per_class": None, "signals": {}}
            for signal in SIGNALS:
                y, s, _, n = paired_arrays(cell, args.id_dataset, code, signal)
                dom["n_per_class"] = n
                summ = {
                    "auroc": metric_summary(y, s, auroc, args.n_boot, args.bootstrap_seed),
                    "auprc": metric_summary(y, s, auprc, args.n_boot, args.bootstrap_seed),
                    "id_mean": float(s[:n].mean()), "ood_mean": float(s[n:].mean()),
                }
                summ["quality"] = signal_quality(summ["auroc"])
                dom["signals"][signal] = summ
            yr, sr, _, _, fit = residualize_length(cell, args.id_dataset, code, "ilv_last")
            dom["length_control"] = {
                "fit_pooled": fit,
                "auroc_residual": metric_summary(yr, sr, auroc, args.n_boot, args.bootstrap_seed),
            }
            cell_out["domains"][code] = dom
        # Within-dataset ILV~log(len) slopes (the direct opposite-sign test).
        for code in [args.id_dataset, *args.ood_datasets]:
            cell_out["length_slopes"][code] = ols_log_length(
                cell[code]["scores"]["ilv_last"], cell[code]["scores"]["prompt_len"])
        # Wrong-answer detection on ID.
        idb = cell[args.id_dataset]
        keep = [i for i, c in enumerate(idb["correct"]) if c is not None]
        wrong = np.asarray([0 if idb["correct"][i] else 1 for i in keep], dtype=int)
        wa = {"n": int(len(keep)), "n_wrong": int(wrong.sum()),
              "acc": float(1.0 - wrong.mean()) if len(keep) else float("nan"), "signals": {}}
        for signal in SIGNALS:
            s = np.asarray([idb["scores"][signal][i] for i in keep], dtype=float)
            ci = bootstrap_ci(auroc, wrong, s, n_boot=args.n_boot, seed=args.bootstrap_seed)
            wa["signals"][signal] = {
                "auroc": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "n_boot": ci["n_boot"],
                "mean_correct": float(s[wrong == 0].mean()) if (wrong == 0).any() else float("nan"),
                "mean_wrong": float(s[wrong == 1].mean()) if (wrong == 1).any() else float("nan"),
            }
        out["cells"][name] = cell_out
        out["wrong_answer_id"][name] = wa

    # Paired deltas on the primary signal: prompt effect within model, model
    # effect within prompt.
    comparisons = {
        "armA_prompt_effect(comparison-mcq)": ("armA-mcq", "armA-comparison"),
        "armB_prompt_effect(comparison-mcq)": ("armB-mcq", "armB-comparison"),
        "model_effect_at_mcq(B-A)": ("armA-mcq", "armB-mcq"),
        "model_effect_at_comparison(B-A)": ("armA-comparison", "armB-comparison"),
    }
    for comp_name, (na, nb) in comparisons.items():
        per_domain = {}
        for code in args.ood_datasets:
            per_domain[code] = cell_pair_delta(
                cells, na, nb, args.id_dataset, code, "ilv_last", auroc, args.n_boot, args.bootstrap_seed)
        out["deltas"][comp_name] = {
            "per_domain_auroc": per_domain,
            "macro_auroc": macro_cell_delta(
                cells, na, nb, args.id_dataset, args.ood_datasets, "ilv_last", auroc,
                args.n_boot, args.bootstrap_seed),
        }
    return out


# ---------------------------------------------------------------------------
# Eval 2: teacher-forced full-sequence trace
# ---------------------------------------------------------------------------
def trace_rows_for_domain(d, cap):
    """Rows usable for the teacher-forced trace: need a valid gold letter,
    `letter_question`, and a non-empty explanation (obqa_gen test is unfiltered,
    so empty fact1 rows exist; medexqa rows can lack the gold letter)."""
    rows, ids, skipped = [], [], 0
    for i, ex in enumerate(d.rows):
        g = ex.get("gold_letter")
        expl = str(ex.get("answer") or "").strip()
        if g not in CHOICES or not ex.get("letter_question") or not expl:
            skipped += 1
            continue
        rows.append(ex)
        ids.append(d.ids[i])
        if cap and len(rows) >= cap:
            break
    return rows, ids, skipped


@torch.no_grad()
def trace_example(model, tokenizer, causal_model, fcvr_layers, choice_ids_t,
                  system_instruction, ex, seed):
    """One teacher-forced forward over prompt + gold '{letter}\\nExplanation: ...'
    + EOS (batch 1, no padding). Row t describes 'position t, about to predict
    token t+1'. Returns (per-position records, example-level aggregates)."""
    device = model.device
    # Target string via the canonical builder (letter + marker + " " + explanation);
    # the PROMPT is rebuilt with this arm's system instruction (the builder
    # hardwires the comparison instruction).
    target = build_target_mode_example(ex, tokenizer, "answer_explanation")["answer"]
    prompt = multiple_choice_prompt_engineer(
        {"question": ex["letter_question"], "answer": ex["gold_letter"], "id": ex.get("id")},
        tokenizer=tokenizer, system_instruction=system_instruction)["question"]
    gold_letter = ex["gold_letter"]

    full = prompt + target
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)

    # Character boundaries of the four segments.
    c0 = len(prompt)
    c1 = c0 + len(gold_letter)
    c2 = c1 + len(EXPLANATION_MARKER)

    # Prefix cross-checks: does tokenizing each prefix reproduce a prefix of the
    # full ids? Failures are recorded, never fatal (offset spans still assign).
    boundary_clean = True
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if ids[:len(p_ids)] == p_ids:
        n_prompt = len(p_ids)
    else:
        boundary_clean = False
        n_prompt = sum(1 for s, _ in offsets if s < c0)
    for prefix_text in (prompt + gold_letter, prompt + gold_letter + EXPLANATION_MARKER):
        pref = tokenizer(prefix_text, add_special_tokens=False).input_ids
        if ids[:len(pref)] != pref:
            boundary_clean = False

    letter_span_len = sum(1 for s, _ in offsets if c0 <= s < c1)
    if letter_span_len != 1:
        boundary_clean = False

    ids.append(tokenizer.eos_token_id)
    offsets.append(None)  # appended EOS: no character span
    seq_len = len(ids)

    def category_of(t):
        if t < n_prompt - 1:
            return "prompt"
        if t == n_prompt - 1:
            return "prompt_final"
        if offsets[t] is None:
            return "eos"
        s = offsets[t][0]
        if s < c1:
            return "letter"
        if s < c2:
            return "marker"
        return "explanation"

    cats = [category_of(t) for t in range(seq_len)]
    expl_pos = [t for t, c in enumerate(cats) if c == "explanation"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    logits = model(input_ids=input_ids.unsqueeze(0)).logits[0].float()   # [seq, vocab]

    logp = F.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)                            # [seq]
    next_ids = input_ids[1:]
    surprisal = -logp[torch.arange(seq_len - 1, device=device), next_ids]  # [seq-1]

    per_layer = []
    for l in fcvr_layers:
        L = moe_router(causal_model.layers[l]).last_cholesky_factor
        E = L.shape[-1]
        L = L.view(1, seq_len, E, E)[0].float()
        per_layer.append((L ** 2).sum(dim=(-1, -2)))                       # tr(LL^T) [seq]
    ilv_layers = torch.stack(per_layer, dim=0)                             # [n_layers, seq]
    ilv_sorted = ilv_layers.sort(dim=0, descending=True).values
    ilv_mean = ilv_layers.mean(dim=0)
    ilv_max = ilv_sorted[0]
    ilv_secondmax = ilv_sorted[1] if len(fcvr_layers) > 1 else ilv_sorted[0]

    # Letter probe at the final prompt row (same forward): 4-way distribution.
    choice_logits = logits[n_prompt - 1, choice_ids_t]
    choice_probs = F.softmax(choice_logits, dim=0)
    pred_letter = CHOICES[int(choice_probs.argmax())]
    answer_correct = pred_letter == gold_letter
    letter_entropy = float(-(choice_probs * choice_probs.clamp(min=1e-12).log()).sum())

    ilv_mean_np = ilv_mean.cpu().numpy()
    ilv_layers_np = ilv_layers.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    surprisal_np = surprisal.cpu().numpy()
    tok_strs = [tokenizer.decode([i]) for i in ids]

    records = []
    for t in range(seq_len):
        rel = None
        if cats[t] == "explanation" and expl_pos:
            rel = (expl_pos.index(t)) / max(1, len(expl_pos) - 1)
        lo = max(0, t - 9)
        records.append({
            "t": t,
            "category": cats[t],
            "tok": tok_strs[t],
            "next_tok": tok_strs[t + 1] if t + 1 < seq_len else None,
            "ilv_mean": float(ilv_mean_np[t]),
            "ilv_max": float(ilv_max[t]),
            "ilv_secondmax": float(ilv_secondmax[t]),
            "ilv_runmean10": float(ilv_mean_np[lo:t + 1].mean()),
            "ilv_per_layer": [float(v) for v in ilv_layers_np[:, t]],
            "entropy": float(entropy_np[t]),
            "surprisal": float(surprisal_np[t]) if t < seq_len - 1 else None,
            "answer_correct": bool(answer_correct),
            "rel_expl_pos": rel,
        })

    def rows_of(cat):
        return [t for t, c in enumerate(cats) if c == cat]

    expl = np.asarray([ilv_mean_np[t] for t in expl_pos], dtype=float)
    expl_sorted = np.sort(expl)[::-1]
    letter_rows = rows_of("letter")
    marker_rows = rows_of("marker")
    aggs = {
        "prompt_len": float(n_prompt),
        "target_len": float(seq_len - n_prompt),
        "n_expl_tokens": float(len(expl_pos)),
        "expl_ilv_mean": float(expl.mean()) if expl.size else float("nan"),
        "expl_ilv_max": float(expl_sorted[0]) if expl.size else float("nan"),
        "expl_ilv_secondmax": float(expl_sorted[1]) if expl.size > 1 else
                              (float(expl_sorted[0]) if expl.size else float("nan")),
        "expl_ilv_last10_mean": float(expl[-10:].mean()) if expl.size else float("nan"),
        "expl_entropy_mean": float(np.mean([entropy_np[t] for t in expl_pos])) if expl_pos else float("nan"),
        "expl_surprisal_mean": float(np.mean([surprisal_np[t] for t in expl_pos if t < seq_len - 1]))
                               if expl_pos else float("nan"),
        "prompt_final_ilv": float(ilv_mean_np[n_prompt - 1]),
        "prompt_ilv_mean": float(ilv_mean_np[:n_prompt].mean()),
        "letter_ilv": float(np.mean([ilv_mean_np[t] for t in letter_rows])) if letter_rows else float("nan"),
        "marker_ilv_mean": float(np.mean([ilv_mean_np[t] for t in marker_rows])) if marker_rows else float("nan"),
        "eos_ilv": float(ilv_mean_np[-1]),
        "letter_entropy": letter_entropy,
        "seq_ilv_max": float(ilv_mean_np.max()),
    }
    meta = {
        "seq_ilv_argmax_category": cats[int(ilv_mean_np.argmax())],
        "pred_letter": pred_letter,
        "gold_letter": gold_letter,
        "answer_correct": bool(answer_correct),
        "boundary_clean": bool(boundary_clean),
        "letter_span_len": int(letter_span_len),
        "category_counts": {c: cats.count(c) for c in CATEGORIES},
        # Per-category per-layer means (n_layers floats each) for the layer diagnostics.
        "per_layer_category_mean": {
            c: [float(v) for v in ilv_layers_np[:, rows_of(c)].mean(axis=1)]
            for c in CATEGORIES if rows_of(c)
        },
    }
    return records, aggs, meta


def run_trace_cell(model, tokenizer, fcvr_layers, args, cell, arm_key, prompt_key,
                   trace_domains, pertoken_fh):
    """All trace forwards for one cell. Streams per-token rows to `pertoken_fh`;
    returns {code: {ids, scores{agg: np.array}, meta rows}}."""
    causal_model = model.base_model.model.model
    device = model.device
    choice_ids = [tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
    if any(i is None or i == tokenizer.unk_token_id for i in choice_ids):
        raise ValueError(f"letter tokens {CHOICES} are not single tokens: {choice_ids}")
    choice_ids_t = torch.tensor(choice_ids, device=device)
    system_instruction = SYSTEM_INSTRUCTIONS[prompt_key]

    out = {}
    audit_printed = False
    for domain_index, (code, (rows, ids, skipped)) in enumerate(trace_domains.items()):
        agg_lists, meta_rows = {}, []
        n_unclean = 0
        for ex_i, (ex, ex_id) in enumerate(tqdm(list(zip(rows, ids)),
                                                desc=f"trace {cell} {code}")):
            # Stable per-(domain, example) MC seed, identical across cells.
            seed = args.sampling_seed * 100003 + domain_index * 100000 + ex_i
            records, aggs, meta = trace_example(
                model, tokenizer, causal_model, fcvr_layers, choice_ids_t,
                system_instruction, ex, seed)
            if not audit_printed:
                audit_printed = True
                boundary = [r for r in records
                            if r["category"] in ("prompt_final", "letter", "marker", "eos")
                            or (r["category"] == "explanation" and r["rel_expl_pos"] in (0.0,))]
                print(f"  [{cell}] boundary audit (first example, id={ex_id}):")
                for r in boundary[:8]:
                    print(f"    t={r['t']:4d} {r['category']:12s} tok={r['tok']!r} -> next={r['next_tok']!r}")
            if not meta["boundary_clean"]:
                n_unclean += 1
            for k, v in aggs.items():
                agg_lists.setdefault(k, []).append(v)
            meta_rows.append({**meta, "id": ex_id, "mc_seed": seed, **aggs})
            for r in records:
                pertoken_fh.write(json.dumps({
                    "cell": cell, "arm": arm_key, "system_prompt": prompt_key,
                    "dataset": code, "is_ood": code != args.id_dataset,
                    "example_id": ex_id, "example_index": ex_i, **r}) + "\n")
        scores = {k: np.asarray(v, dtype=float) for k, v in agg_lists.items()}
        out[code] = {"ids": ids, "scores": scores, "meta": meta_rows,
                     "n_skipped_rows": skipped, "n_boundary_unclean": n_unclean}
        print(f"  {cell} {code}: n={len(ids)} (skipped {skipped} rows without letter/explanation; "
              f"{n_unclean} boundary-unclean) expl_ILV={np.nanmean(scores['expl_ilv_mean']):.4f} "
              f"promptfinal_ILV={scores['prompt_final_ilv'].mean():.4f}")
    return out


def trace_stats(args, traces):
    """OoD + wrong-answer AUROC/AUPRC from the explanation-region aggregates,
    category profiles, and per-layer diagnostics, per trace cell."""
    out = {}
    for cell, blocks in traces.items():
        cell_out = {"n": {c: len(b["ids"]) for c, b in blocks.items()},
                    "n_skipped_rows": {c: b["n_skipped_rows"] for c, b in blocks.items()},
                    "n_boundary_unclean": {c: b["n_boundary_unclean"] for c, b in blocks.items()},
                    "ood": {}, "wrong_answer_id": {}, "category_profile": {},
                    "per_layer_category_mean": {}}
        # OoD detection from explanation-region aggregates.
        for code in args.trace_ood_datasets:
            if code not in blocks:
                continue
            dom = {"n_per_class": None, "aggregates": {}}
            for agg in TRACE_OOD_AGGS:
                y, s, _, n = paired_arrays(blocks, args.id_dataset, code, agg)
                dom["n_per_class"] = n
                summ = {
                    "auroc": metric_summary(y, s, auroc, args.n_boot, args.bootstrap_seed),
                    "auprc": metric_summary(y, s, auprc, args.n_boot, args.bootstrap_seed),
                    "id_mean": float(np.nanmean(s[:n])), "ood_mean": float(np.nanmean(s[n:])),
                }
                summ["quality"] = signal_quality(summ["auroc"])
                dom["aggregates"][agg] = summ
            cell_out["ood"][code] = dom
        # Wrong-answer detection on ID.
        idb = blocks[args.id_dataset]
        wrong = np.asarray([0 if m["answer_correct"] else 1 for m in idb["meta"]], dtype=int)
        wa = {"n": int(wrong.size), "n_wrong": int(wrong.sum()),
              "acc": float(1.0 - wrong.mean()) if wrong.size else float("nan"), "signals": {}}
        for agg in TRACE_WRONG_AGGS:
            s = idb["scores"][agg]
            ci = bootstrap_ci(auroc, wrong, s, n_boot=args.n_boot, seed=args.bootstrap_seed)
            wa["signals"][agg] = {"auroc": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "n_boot": ci["n_boot"]}
        cell_out["wrong_answer_id"] = wa
        # Category profile + per-layer means, per dataset.
        for code, b in blocks.items():
            prof = {}
            layer_acc = {}
            for m in b["meta"]:
                for c, vec in m["per_layer_category_mean"].items():
                    acc = layer_acc.setdefault(c, [np.zeros(len(vec)), 0])
                    acc[0] += np.asarray(vec)
                    acc[1] += 1
            for c in CATEGORIES:
                key = {"prompt": "prompt_ilv_mean", "prompt_final": "prompt_final_ilv",
                       "letter": "letter_ilv", "marker": "marker_ilv_mean",
                       "explanation": "expl_ilv_mean", "eos": "eos_ilv"}[c]
                vals = [v for v in b["scores"][key] if np.isfinite(v)]
                prof[c] = bootstrap_mean_ci(vals, n_boot=args.n_boot, seed=args.bootstrap_seed) if vals else None
            cell_out["category_profile"][code] = prof
            cell_out["per_layer_category_mean"][code] = {
                c: [float(v) for v in (total / count)] for c, (total, count) in layer_acc.items()
            }
        out[cell] = cell_out
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report(args, summary):
    lines = [
        "# OBQA arm-B ILV inversion diagnostics: prompt crossover + teacher-forced trace",
        "",
        f"ID = `{args.id_dataset}`; split = `{args.split}`; data seed = {args.data_seed}; "
        f"MC seed = {args.sampling_seed}; S = {args.num_samples}; routing = {args.routing}.",
        "",
        "Sign convention: higher score is fixed a priori to mean OoD / wrong; AUROC below 0.5 is an",
        "inverted signal and is never flipped. A CI crossing zero is inconclusive, not equivalence.",
        "",
    ]
    xo = summary.get("crossover")
    if xo:
        lines += [
            "## Eval 1: prompt-crossover ILV OoD AUROC (final predictive position)",
            "",
            "| OoD domain | " + " | ".join(xo["cells"]) + " |",
            "|---|" + "---:|" * len(xo["cells"]),
        ]
        for code in args.ood_datasets:
            row = [f"| {code} "]
            for name in xo["cells"]:
                a = xo["cells"][name]["domains"][code]["signals"]["ilv_last"]["auroc"]
                q = xo["cells"][name]["domains"][code]["signals"]["ilv_last"]["quality"]
                row.append(f"| {f3(a['point'])} [{f3(a['lo'])},{f3(a['hi'])}] ({q}) ")
            lines.append("".join(row) + "|")
        lines += [
            "",
            "### Paired ILV AUROC deltas (macro over OoD domains)",
            "",
            "| Comparison | macro Δ [95% CI] |",
            "|---|---:|",
        ]
        for comp, d in xo["deltas"].items():
            lines.append(f"| {comp} | {ci_text(d['macro_auroc'])} |")
        lines += [
            "",
            "### ILV ~ log(prompt_len) slopes (within dataset)",
            "",
            "| dataset | " + " | ".join(f"{n} slope (r2)" for n in xo["cells"]) + " |",
            "|---|" + "---:|" * len(xo["cells"]),
        ]
        for code in [args.id_dataset, *args.ood_datasets]:
            row = [f"| {code} "]
            for name in xo["cells"]:
                sl = xo["cells"][name]["length_slopes"][code]
                row.append(f"| {sl['slope_per_log_token']:+.3f} ({f3(sl['r2'])}) ")
            lines.append("".join(row) + "|")
        lines += [
            "",
            "### Wrong-answer detection on ID (ilv_last)",
            "",
            "| cell | ACC | AUROC(wrong) ilv_last [95% CI] |",
            "|---|---:|---:|",
        ]
        for name, wa in xo["wrong_answer_id"].items():
            v = wa["signals"]["ilv_last"]
            lines.append(f"| {name} | {f3(wa['acc'])} | {f3(v['auroc'])} [{f3(v['lo'])},{f3(v['hi'])}] |")
    tr = summary.get("trace")
    if tr:
        lines += [
            "",
            "## Eval 2: teacher-forced trace (explanation-region ILV aggregates)",
            "",
            "OoD AUROC per aggregate (ID vs each explanation-bearing domain):",
            "",
        ]
        for cell, c in tr.items():
            lines.append(f"### {cell}")
            lines.append("")
            lines.append("| aggregate | " + " | ".join(args.trace_ood_datasets) + " | AUROC(wrong) on ID |")
            lines.append("|---|" + "---:|" * (len(args.trace_ood_datasets) + 1))
            for agg in TRACE_OOD_AGGS:
                row = [f"| {agg} "]
                for code in args.trace_ood_datasets:
                    a = c["ood"].get(code, {}).get("aggregates", {}).get(agg)
                    row.append(f"| {f3(a['auroc']['point'])} ({a['quality']}) " if a else "| n/a ")
                w = c["wrong_answer_id"]["signals"].get(agg)
                row.append(f"| {f3(w['auroc'])} " if w else "| n/a ")
                lines.append("".join(row) + "|")
            lines.append("")
            lines.append("Category-profile mean ILV (ID vs OoD):")
            lines.append("")
            codes = list(c["category_profile"])
            lines.append("| category | " + " | ".join(codes) + " |")
            lines.append("|---|" + "---:|" * len(codes))
            for cat in CATEGORIES:
                row = [f"| {cat} "]
                for code in codes:
                    p = c["category_profile"][code].get(cat)
                    row.append(f"| {f3(p['mean'])} " if p else "| n/a ")
                lines.append("".join(row) + "|")
            lines.append("")
    lines += [
        "## Interpretation rules",
        "",
        "- Eval 1 separates 'inversion follows the model' (arm B inverted under BOTH prompts)",
        "  from 'inversion follows the prompt' (each model flips when the prompt changes).",
        "- Opposite-sign length slopes between cells sharing a model but not a prompt implicate the prompt;",
        "  opposite signs between models under the SAME prompt implicate training.",
        "- Never flip an inverted AUROC; a CI crossing zero is inconclusive.",
        "",
        f"Git commit: `{summary['config']['git_commit']}`.",
    ]
    return "\n".join(lines) + "\n"


def write_crossover_perexample(path, args, cells):
    with open(path, "w", encoding="utf-8") as f:
        for name, cell in cells.items():
            for code, block in cell.items():
                is_id = code == args.id_dataset
                for i, example_id in enumerate(block["ids"]):
                    row = {
                        "cell": name, "dataset": code, "is_ood": not is_id,
                        "id": example_id, "mc_seed": block["mc_seed"],
                        "gold_letter": block["gold"][i], "pred_letter": block["pred"][i],
                        "correct": block["correct"][i],
                        **{k: float(v[i]) for k, v in block["scores"].items()},
                    }
                    f.write(json.dumps(row) + "\n")


def write_trace_perexample(path, args, traces):
    with open(path, "w", encoding="utf-8") as f:
        for cell, blocks in traces.items():
            for code, block in blocks.items():
                is_id = code == args.id_dataset
                for m in block["meta"]:
                    row = {"cell": cell, "dataset": code, "is_ood": not is_id, **m}
                    f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    # Registry defaults (adapters/run_suffixes/system prompts) + CLI overrides.
    setup, arm_cfgs = resolve_arm_setup(args)
    validate_args(args)
    paths = output_paths(args)
    check_output_collisions(list(paths.values()), args.overwrite)
    setup_environment()

    do_crossover = args.stages in ("both", "crossover")
    do_trace = args.stages in ("both", "trace")

    print("#" * 80)
    print(f"# OBQA arm-B ILV inversion diagnostics ({args.stages})")
    print(f"# DATA seed={args.data_seed} | MC seed={args.sampling_seed} | S={args.num_samples} | routing={args.routing}")
    print(f"# Eval-1 OoD={args.ood_datasets} | Eval-2 trace OoD={args.trace_ood_datasets}")
    for arm_key in ARM_KEYS:
        cfg = arm_cfgs[arm_key]
        print(f"# {arm_key}: adapter={cfg['adapter']} suffix={cfg['run_suffix']} "
              f"weights_ds={cfg['weights_dataset'] or args.id_dataset} training_prompt={cfg['system_prompt']}")
    print("#" * 80)

    domains = load_domains(args)

    # Trace row lists (fixed BEFORE any model is loaded so both cells see the
    # same rows; unusable rows -- no letter / no explanation -- are skipped).
    trace_domains = {}
    if do_trace:
        trace_domains[args.id_dataset] = trace_rows_for_domain(domains[args.id_dataset], args.trace_n_id)
        for code in args.trace_ood_datasets:
            trace_domains[code] = trace_rows_for_domain(domains[code], args.trace_n_ood)
        for code, (rows, ids, skipped) in trace_domains.items():
            if not rows:
                raise SystemExit(f"trace: no usable rows for {code} (all lacked gold letter/explanation)")
            print(f"trace rows {code}: {len(rows)} kept, {skipped} skipped")

    os.makedirs(args.output_dir, exist_ok=True)
    cells, traces = {}, {}
    pertoken_fh = open(paths["pertoken"], "w", encoding="utf-8") if do_trace else None
    try:
        for arm_key in ARM_KEYS:
            cfg = arm_cfgs[arm_key]
            print("\n" + "=" * 80)
            print(f"{cfg['label']}")
            print("=" * 80)
            seed_everything(args.sampling_seed)
            model, tokenizer, fcvr_layers = prepare(
                make_prepare_args(args, cfg["adapter"], cfg["run_suffix"], cfg["weights_dataset"]))
            if sorted(fcvr_layers) != sorted(args.swap_layers):
                raise RuntimeError(f"loaded FCVR layers {fcvr_layers} != requested {sorted(args.swap_layers)}")
            set_routing_mode(model, fcvr_layers, args.routing)
            try:
                if do_crossover:
                    # All (left-padded) readout work first ...
                    for prompt_key in PROMPT_KEYS:
                        name = cell_name(arm_key, prompt_key)
                        print(f"\n-- Eval 1 cell {name} --")
                        cells[name] = score_cell(
                            model, tokenizer, fcvr_layers, domains,
                            SYSTEM_INSTRUCTIONS[prompt_key], args)
                if do_trace:
                    # ... then the un-padded batch-1 trace under the arm's OWN prompt.
                    prompt_key = cfg["system_prompt"]
                    name = cell_name(arm_key, prompt_key)
                    print(f"\n-- Eval 2 trace cell {name} --")
                    traces[name] = run_trace_cell(
                        model, tokenizer, fcvr_layers, args, name, arm_key, prompt_key,
                        trace_domains, pertoken_fh)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if pertoken_fh:
            pertoken_fh.close()

    # Alignment guards.
    if do_crossover:
        names = list(cells)
        for code in domains:
            id_lists = [cells[n][code]["ids"] for n in names]
            if any(l != id_lists[0] for l in id_lists[1:]):
                raise RuntimeError(f"example-ID ordering differs across cells for {code}")
        for prompt_key in PROMPT_KEYS:
            same_prompt = [n for n in names if n.endswith(f"-{prompt_key}")]
            for code in domains:
                lens = [cells[n][code]["scores"]["prompt_len"] for n in same_prompt]
                if any(not np.array_equal(l, lens[0]) for l in lens[1:]):
                    raise RuntimeError(f"prompt_len differs across same-prompt cells for {code}/{prompt_key}")

    summary = {
        "config": {
            "git_commit": git_revision(),
            "stages": args.stages,
            "model_shortcode": args.model_shortcode,
            "id_dataset": args.id_dataset,
            "ood_datasets": args.ood_datasets,
            "trace_ood_datasets": args.trace_ood_datasets,
            "ood_description": {c: setup["ood_description"].get(c, "unclassified shift")
                                for c in args.ood_datasets},
            "split": args.split,
            "n_per_domain_cap": args.n_per_domain,
            "trace_n_id": args.trace_n_id,
            "trace_n_ood": args.trace_n_ood,
            "data_seed": args.data_seed,
            "sampling_seed": args.sampling_seed,
            "num_samples": args.num_samples,
            "routing": args.routing,
            "swap_layers": sorted(args.swap_layers),
            "prior_source": args.prior_source,
            "arms": {k: {kk: arm_cfgs[k][kk] for kk in
                         ("adapter", "run_suffix", "weights_dataset", "system_prompt")}
                     for k in ARM_KEYS},
            "n_boot": args.n_boot,
            "bootstrap_seed": args.bootstrap_seed,
            "crossover_cells": [cell_name(a, p) for a in ARM_KEYS for p in PROMPT_KEYS] if do_crossover else [],
            "trace_cells": [cell_name(a, arm_cfgs[a]["system_prompt"]) for a in ARM_KEYS] if do_trace else [],
            "primary_signal": "ilv_last (Eval 1, final predictive position) / "
                              "explanation-region ILV aggregates (Eval 2)",
            "sign": "higher score => OoD / wrong; never flip inverted AUROC",
        },
    }
    if do_crossover:
        print("\nComputing crossover statistics ...")
        summary["crossover"] = crossover_stats(args, cells)
    if do_trace:
        print("Computing trace statistics ...")
        summary["trace"] = trace_stats(args, traces)
        # Sanity cross-check: trace prompt_final ILV vs Eval-1 ilv_last (same
        # rows, same cell) should agree in rank (exactly under deterministic routing).
        if do_crossover:
            checks = {}
            for arm_key in ARM_KEYS:
                name = cell_name(arm_key, arm_cfgs[arm_key]["system_prompt"])
                if name not in cells or name not in traces:
                    continue
                tb = traces[name][args.id_dataset]
                xo_block = cells[name][args.id_dataset]
                pos = {ex_id: i for i, ex_id in enumerate(xo_block["ids"])}
                pairs = [(tb["scores"]["prompt_final_ilv"][j], xo_block["scores"]["ilv_last"][pos[ex_id]])
                         for j, ex_id in enumerate(tb["ids"]) if ex_id in pos]
                if len(pairs) >= 3:
                    a, b = zip(*pairs)
                    rho = _spearman(a, b)
                    checks[name] = {"n": len(pairs), "spearman_trace_vs_eval1": rho}
                    print(f"  sanity {name}: Spearman(trace promptfinal ILV, Eval-1 ilv_last) = {rho:.3f} "
                          f"(n={len(pairs)})")
            summary["trace_vs_crossover_sanity"] = checks

    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(build_report(args, summary))
    if do_crossover:
        write_crossover_perexample(paths["crossover"], args, cells)
    if do_trace:
        write_trace_perexample(paths["perexample"], args, traces)

    print("\n" + build_report(args, summary))
    print("Saved:")
    for k, p in paths.items():
        if os.path.exists(p):
            print(f"  {p}")


if __name__ == "__main__":
    main()
