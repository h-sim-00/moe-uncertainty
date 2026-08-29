"""OoD detection from FCVR ILV at the answer position AND over the explanation
(gold or self-generated), arm A (answer-only) vs arm B (answer + fact1), ID =
OpenBookQA (branch OBQA-comparison).

Why
---
analyze_obqa_gen.py showed that arm B's ILV is INVERTED at the paper-protocol
read-out position (final prompt token, just before the answer letter) but
correctly oriented over the explanation region -- so far only against
medexqa/medmcqa_gen and only with teacher-forced GOLD explanations. This script
(a) re-runs the paper-protocol read-out on arc_c / arc_e / medexqa, and (b) on
explanation-bearing datasets (medexqa, scienceqa, ecqa, aqua_rat) reads the
nine example-level explanation scores for BOTH a teacher-forced gold
explanation and the model's OWN generated explanation, each as an OoD score
(ID = obqa_gen). Only ID-vs-OoD AUROC/AUPRC is reported (wrong-answer AUROC is
out of scope; correctness is recorded for stratification only).

Stages (one per invocation, each with its own output stem)
----------------------------------------------------------
stage1  final-position ilv_last (evaluate_letter.readout protocol, 5-choice
        aware) on the ID set and --ood_datasets (default arc_c arc_e medexqa);
        paired B-A deltas via evaluate_ilv_ood_arms.compare_domain/macro_delta.
tf      teacher-forced trace: prompt + "<gold>\\nExplanation: <gold expl>" +
        EOS, one batch-1 forward per example; the nine scores below + the
        final-position read-out on the same rows (default OoD: medexqa
        scienceqa ecqa aqua_rat).
gen     generated trace: prompt-only forward -> predicted letter (masked over
        the example's valid choices) -> forced prefix "<pred>\\nExplanation:"
        -> greedy decode (KV cache, <= --max_new_tokens or EOS) with the FCVR
        routers hooked (online ILV per generated token) -> one post-hoc
        batch-1 forward over the finished sequence = PRIMARY nine scores
        (identical code path / seed protocol as `tf`, so the only difference
        between the conditions is the explanation text). Online values are
        kept alongside (`*_online`) with an agreement block.

The nine explanation scores (per example)
-----------------------------------------
  expl_ilv_mean / expl_ilv_max / expl_ilv_secondmax / expl_ilv_last10_mean
      aggregates of the per-position ILV (mean over FCVR layers of tr(LL^T))
      over the explanation positions
  expl_entropy_mean / expl_surprisal_mean
      predictive entropy / next-token surprisal averaged over the explanation
  prompt_final_ilv   ILV at the final prompt position (paper protocol)
  letter_ilv         ILV at the answer-letter position
  eos_ilv            ILV at the EOS position (NaN when generation hit the cap)
`n_expl_tokens` is tabulated as a length baseline.

Choices: one canonical inner prompt ("Question/Choices/A. .../Answer:") for
every dataset; answer probabilities are masked to each example's valid letters
(4-choice rows never see E); normalised entropy = H / log(n_choices).

Seeds: --data_seed fixes rows, --sampling_seed fixes the FCVR Monte Carlo
draws; DOMAIN_SEED_INDEX pins the per-domain seed offsets to the shortcode so
the stage1 and tf numbers reproduce evaluate_ilv_ood_arms.py /
analyze_obqa_gen.py exactly. Outputs refuse to overwrite unless --overwrite.

Typical (quail-1, repo root, moe_env)::

    python evaluate_ood_expl_readout.py --stage stage1
    python evaluate_ood_expl_readout.py --stage tf
    python evaluate_ood_expl_readout.py --stage gen

Smoke::

    python evaluate_ood_expl_readout.py --stage gen --n_per_domain 12 --trace_n_id 6 \\
        --trace_n_ood 4 --n_boot 50 --max_new_tokens 24 --tag smoke
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_ilv_ood_arms import (
    ARM_SETUP,
    DEFAULT_LAYERS,
    SAME_SOURCE,
    check_output_collisions,
    ci_text,
    compare_domain,
    delta_verdict,
    domain_sampling_seed,
    f3,
    git_revision,
    load_domains,
    macro_delta,
    make_prepare_args,
    metric_summary,
    paired_arrays,
    paired_bootstrap_delta,
    prompts_for_rows,
    resolve_arm_setup,
    signal_quality,
)
from analyze_obqa_gen import ARM_KEYS, ARM_SHORT, CATEGORIES, _spearman, cell_name, set_routing_mode
from analyze_token_signals import OnlineILVRecorder
from evaluate_letter import GateEntropyRecorder, prepare
from uq_stats import auprc, auroc, bootstrap_mean_ci
from utils import seed_everything, setup_environment
from utils.data import EXPLANATION_MARKER
from utils.prompt import SYSTEM_INSTRUCTIONS, multiple_choice_prompt_engineer

LETTERS = ["A", "B", "C", "D", "E"]
STAGES = ("stage1", "tf", "gen")
DEFAULT_OOD = {
    "stage1": ["arc_c", "arc_e", "medexqa"],
    "tf": ["medexqa", "scienceqa", "ecqa", "aqua_rat"],
    "gen": ["medexqa", "scienceqa", "ecqa", "aqua_rat"],
}
# Per-shortcode seed offsets (NOT list positions) -> identical MC draws to the
# saved evaluate_ilv_ood_arms.py (obqa_gen medexqa medmcqa_gen arc_e arc_c sciq
# mmlu_law) and analyze_obqa_gen.py (obqa_gen medexqa medmcqa_gen) runs.
DOMAIN_SEED_INDEX = {
    "obqa_gen": 0, "medexqa": 1, "medmcqa_gen": 2, "arc_e": 3, "arc_c": 4, "sciq": 5,
    "mmlu_law": 6, "scienceqa": 7, "ecqa": 8, "aqua_rat": 9,
}
NINE_SCORES = (
    "expl_ilv_mean", "expl_ilv_max", "expl_ilv_secondmax", "expl_ilv_last10_mean",
    "expl_entropy_mean", "expl_surprisal_mean", "prompt_final_ilv", "letter_ilv", "eos_ilv",
)
TRACE_SCORES = NINE_SCORES + ("n_expl_tokens",)          # + length baseline
ONLINE_SCORES = ("expl_ilv_mean_online", "expl_ilv_max_online", "expl_ilv_secondmax_online",
                 "expl_ilv_last10_mean_online", "expl_entropy_mean_online",
                 "expl_surprisal_mean_online", "prompt_final_ilv_online", "letter_ilv_online",
                 "expl_gate_entropy_mean_online")
READOUT_SIGNALS = ("ilv_last", "gate_entropy_last_fcvr", "letter_entropy", "letter_entropy_norm",
                   "one_minus_maxprob")
PROFILE_KEY = {"prompt": "prompt_ilv_mean", "prompt_final": "prompt_final_ilv", "letter": "letter_ilv",
               "marker": "marker_ilv_mean", "explanation": "expl_ilv_mean", "eos": "eos_ilv"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--id_dataset", default="obqa_gen", choices=["obqa_gen"])
    p.add_argument("--ood_datasets", nargs="+", default=None,
                   help=f"Default per stage: {DEFAULT_OOD}")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n_per_domain", type=int, default=500, help="Rows loaded per domain (0 = all).")
    p.add_argument("--trace_n_id", type=int, default=500, help="[tf/gen] cap on ID trace rows (0 = all).")
    p.add_argument("--trace_n_ood", type=int, default=500, help="[tf/gen] cap per OoD domain (0 = all).")
    p.add_argument("--max_new_tokens", type=int, default=256, help="[gen] explanation generation cap.")
    p.add_argument("--max_seq_tokens", type=int, default=2048,
                   help="[tf/gen] rows whose teacher-forced sequence exceeds this are skipped (counted).")
    p.add_argument("--pertoken_scope", choices=["target", "all"], default="target",
                   help="[tf/gen] per-token JSONL rows: from prompt_final onward (default) or every position.")
    p.add_argument("--data_seed", type=int, default=42, help="ONLY dataset construction/split seed.")
    p.add_argument("--sampling_seed", type=int, default=42, help="ONLY FCVR Monte Carlo seed.")
    p.add_argument("--num_samples", type=int, default=35, help="FCVR MC samples S (paper: 35).")
    p.add_argument("--batch_size", type=int, default=8, help="Batched final-position read-out only.")
    p.add_argument("--routing", choices=["stochastic", "deterministic"], default="stochastic",
                   help="deterministic = posterior-mean routing (ablation / online-vs-posthoc alignment test).")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--swap_layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--prior_source", choices=["pretrained", "map"], default="pretrained")
    p.add_argument("--map_suffix", default=None)
    p.add_argument("--arm_a_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_a_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--output_dir", default="results/ood_expl_readout")
    p.add_argument("--tag", default="obqa-ood-expl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stats_only", default=None, metavar="PEREXAMPLE_JSONL",
                   help="Recompute this stage's statistics/report from a streamed per-example file (no GPU).")
    p.add_argument("--crosscheck", default=None, metavar="REF_PEREXAMPLE_JSONL",
                   help="Join a saved evaluate_ilv_ood_arms / analyze_obqa_gen per-example file on "
                        "(arm, dataset, id) and print agreement statistics.")
    return p.parse_args(argv)


def validate_args(args):
    if args.model_shortcode != "granite":
        raise SystemExit("The saved comparison arms are Granite-specific; --model_shortcode must be granite.")
    for k in ("n_per_domain", "trace_n_id", "trace_n_ood"):
        if getattr(args, k) < 0:
            raise SystemExit(f"--{k} must be >= 0")
    if args.n_boot < 1 or args.max_new_tokens < 1 or args.max_seq_tokens < 16:
        raise SystemExit("--n_boot >= 1, --max_new_tokens >= 1, --max_seq_tokens >= 16 required")
    if not args.swap_layers:
        raise SystemExit("--swap_layers must contain the FCVR-modified layers")
    if args.ood_datasets is None:
        args.ood_datasets = list(DEFAULT_OOD[args.stage])
    ood = []
    for code in args.ood_datasets:
        if code == args.id_dataset:
            print(f"SKIP {code}: it is the ID anchor")
            continue
        if code in SAME_SOURCE.get(args.id_dataset, set()):
            raise SystemExit(f"ERROR: {code} overlaps the {args.id_dataset} training corpus; not a valid OoD domain.")
        if code not in DOMAIN_SEED_INDEX:
            raise SystemExit(f"ERROR: {code} has no DOMAIN_SEED_INDEX entry (add one; never reuse another's).")
        if code not in ood:
            ood.append(code)
    if not ood:
        raise SystemExit("No valid --ood_datasets remain")
    args.ood_datasets = ood


def output_paths(args):
    stem = os.path.join(args.output_dir,
                        f"{args.tag}_{args.split}_data-s{args.data_seed}_mc-s{args.sampling_seed}_{args.stage}")
    paths = {"json": stem + ".json", "md": stem + ".md", "perexample": stem + "_perexample.jsonl"}
    if args.stage in ("tf", "gen"):
        paths["pertoken"] = stem + "_pertoken.jsonl"
    if args.stage == "gen":
        paths["texts"] = stem + "_texts.jsonl"
    return paths


def sibling_stage_path(args, stage, kind="perexample"):
    a = argparse.Namespace(**vars(args))
    a.stage = stage
    return output_paths(a)[kind]


# ---------------------------------------------------------------------------
# choice handling (pure)
# ---------------------------------------------------------------------------
def choice_mask(n_choices):
    if not (1 <= int(n_choices) <= len(LETTERS)):
        raise ValueError(f"n_choices must be in 1..{len(LETTERS)}, got {n_choices}")
    m = np.zeros(len(LETTERS), dtype=bool)
    m[:int(n_choices)] = True
    return m


def masked_choice_probs(logits5, n_choices):
    """numpy: softmax over the first n_choices letters; exact zeros elsewhere."""
    z = np.asarray(logits5, dtype=float)
    m = choice_mask(n_choices)
    z = np.where(m, z, -np.inf)
    z = z - z[m].max()
    e = np.exp(z)
    return e / e.sum()


def letter_entropy_of(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def normalized_entropy(h, n_choices):
    return float(h / math.log(int(n_choices))) if int(n_choices) > 1 else float("nan")


def n_choices_of(ex):
    return int(ex.get("n_choices") or 4)


def gold_letter_of(ex):
    g = ex.get("gold_letter") or ex.get("answer")
    return g if g in LETTERS[:n_choices_of(ex)] else None


# ---------------------------------------------------------------------------
# span / aggregate bookkeeping (pure)
# ---------------------------------------------------------------------------
def build_spans(n_prompt, n_letter, n_marker, n_expl, has_eos):
    """Token-index spans (input-token convention: row t = position whose input
    is token t). Layout: prompt | letter | marker | explanation | [eos]."""
    if n_prompt < 1 or n_letter < 0 or n_marker < 0 or n_expl < 0:
        raise ValueError("negative span lengths")
    p = n_prompt
    letter = list(range(p, p + n_letter)); p += n_letter
    marker = list(range(p, p + n_marker)); p += n_marker
    expl = list(range(p, p + n_expl)); p += n_expl
    eos = [p] if has_eos else []
    return {"n_prompt": int(n_prompt), "letter": letter, "marker": marker, "explanation": expl,
            "eos": eos, "seq_len": p + (1 if has_eos else 0)}


def categories_from_spans(spans, seq_len):
    if spans["seq_len"] != seq_len:
        raise ValueError(f"spans cover {spans['seq_len']} positions but sequence has {seq_len}")
    cats = ["prompt"] * seq_len
    cats[spans["n_prompt"] - 1] = "prompt_final"
    for name in ("letter", "marker", "explanation", "eos"):
        for t in spans[name]:
            cats[t] = name
    return cats


def explanation_aggregates(ilv_mean, entropy, surprisal, spans, cats, prefix=""):
    """The nine scores (+ auxiliaries) from per-position arrays. `surprisal` has
    one entry per position with NaN where undefined (last row)."""
    ilv_mean = np.asarray(ilv_mean, dtype=float)
    entropy = np.asarray(entropy, dtype=float)
    surprisal = np.asarray(surprisal, dtype=float)
    n_prompt = spans["n_prompt"]
    expl_pos = spans["explanation"]
    expl = ilv_mean[expl_pos]
    expl_sorted = np.sort(expl[np.isfinite(expl)])[::-1]

    def nanmean(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else float("nan")

    def rows_of(cat):
        return [t for t, c in enumerate(cats) if c == cat]

    out = {
        "expl_ilv_mean": nanmean(expl),
        "expl_ilv_max": float(expl_sorted[0]) if expl_sorted.size else float("nan"),
        "expl_ilv_secondmax": float(expl_sorted[1]) if expl_sorted.size > 1 else
                              (float(expl_sorted[0]) if expl_sorted.size else float("nan")),
        "expl_ilv_last10_mean": nanmean(expl[-10:]) if expl.size else float("nan"),
        "expl_entropy_mean": nanmean(entropy[expl_pos]) if expl_pos else float("nan"),
        "expl_surprisal_mean": nanmean(surprisal[expl_pos]) if expl_pos else float("nan"),
        "prompt_final_ilv": float(ilv_mean[n_prompt - 1]),
        "letter_ilv": nanmean(ilv_mean[spans["letter"]]) if spans["letter"] else float("nan"),
        "eos_ilv": float(ilv_mean[spans["eos"][0]]) if spans["eos"] else float("nan"),
        "prompt_ilv_mean": nanmean(ilv_mean[:n_prompt]),
        "marker_ilv_mean": nanmean(ilv_mean[spans["marker"]]) if spans["marker"] else float("nan"),
        "seq_ilv_max": float(np.nanmax(ilv_mean)) if np.isfinite(ilv_mean).any() else float("nan"),
    }
    return {prefix + k: v for k, v in out.items()}


# ---------------------------------------------------------------------------
# tokenisation helpers
# ---------------------------------------------------------------------------
def arm_prompt(tokenizer, ex, system_instruction):
    inner = ex.get("letter_question") or ex.get("question")
    return multiple_choice_prompt_engineer(
        {"question": inner, "answer": ex.get("gold_letter") or "A", "id": ex.get("id")},
        tokenizer=tokenizer, system_instruction=system_instruction)["question"]


def marker_token_ids(tokenizer):
    return list(tokenizer(EXPLANATION_MARKER, add_special_tokens=False).input_ids)


def letter_token_ids(tokenizer):
    ids = [tokenizer.convert_tokens_to_ids(c) for c in LETTERS]
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError(f"letter tokens {LETTERS} are not single tokens: {ids}")
    return ids


def check_prefix_property(tokenizer, prompt, marker_ids, letter_ids):
    """The gen prefix is built by id concatenation; assert that equals string
    tokenisation for every letter so tf and gen prefixes are token-identical."""
    p_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    for L, lid in zip(LETTERS, letter_ids):
        s_ids = list(tokenizer(prompt + L + EXPLANATION_MARKER, add_special_tokens=False).input_ids)
        if s_ids != p_ids + [lid] + marker_ids:
            return False
    return True


def build_tf_sequence(tokenizer, prompt, gold_letter, explanation, marker_ids):
    """Teacher-forced ids + spans for prompt + gold + marker + explanation + EOS,
    tokenised as a STRING exactly like analyze_obqa_gen.trace_example (so its
    numbers reproduce); spans from prefix lengths, cross-checked."""
    exp = explanation if explanation.startswith(" ") else " " + explanation
    full = prompt + gold_letter + EXPLANATION_MARKER + exp
    ids = list(tokenizer(full, add_special_tokens=False).input_ids)
    p_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    pl_ids = list(tokenizer(prompt + gold_letter, add_special_tokens=False).input_ids)
    plm_ids = list(tokenizer(prompt + gold_letter + EXPLANATION_MARKER, add_special_tokens=False).input_ids)
    clean = (ids[:len(p_ids)] == p_ids and ids[:len(pl_ids)] == pl_ids and ids[:len(plm_ids)] == plm_ids
             and len(pl_ids) - len(p_ids) == 1 and plm_ids[len(pl_ids):] == marker_ids)
    if clean:
        n_prompt, n_letter, n_marker = len(p_ids), 1, len(marker_ids)
    else:
        # Offset-mapping fallback (analyze_obqa_gen.trace_example L446-458).
        enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        offs = list(enc.offset_mapping)
        c0, c1 = len(prompt), len(prompt) + len(gold_letter)
        c2 = c1 + len(EXPLANATION_MARKER)
        n_prompt = sum(1 for s, _ in offs if s < c0)
        n_letter = sum(1 for s, _ in offs if c0 <= s < c1)
        n_marker = sum(1 for s, _ in offs if c1 <= s < c2)
    n_expl = len(ids) - n_prompt - n_letter - n_marker
    ids.append(tokenizer.eos_token_id)
    spans = build_spans(n_prompt, n_letter, n_marker, n_expl, has_eos=True)
    return ids, spans, bool(clean)


# ---------------------------------------------------------------------------
# model read-outs
# ---------------------------------------------------------------------------
@torch.no_grad()
def readout_masked(model, tokenizer, prompts, n_choices_list, fcvr_layers, batch_size):
    """evaluate_letter.readout with a per-example choice mask (A..E). Same left
    padding, same ILV/gate-entropy blocks, so ilv_last is bit-comparable."""
    causal_model = model.base_model.model.model
    device = model.device
    choice_ids_t = torch.tensor(letter_token_ids(tokenizer), device=device)
    tokenizer.padding_side = "left"
    rec = GateEntropyRecorder(causal_model)
    probs_all, gate_fcvr, ilv = [], [], []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="letter read-out"):
            batch = prompts[i:i + batch_size]
            nch = torch.tensor(n_choices_list[i:i + batch_size], device=device)
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                               max_length=2048, add_special_tokens=False).to(device)
            bsz, seq_len = inputs["input_ids"].shape
            assert bool(inputs["attention_mask"][:, -1].all()), "left padding expected: last column must be real tokens"
            rec.begin(bsz)
            logits = model(**inputs).logits
            choice_logits = logits[:, -1, :][:, choice_ids_t].float()             # [bsz, 5]
            mask = torch.arange(len(LETTERS), device=device)[None, :] < nch[:, None]
            choice_logits = choice_logits.masked_fill(~mask, float("-inf"))
            probs_all.append(F.softmax(choice_logits, dim=1).cpu())
            ge = rec.per_layer()
            gate_fcvr.append(ge[fcvr_layers].mean(dim=0))
            per_layer = []
            for l in fcvr_layers:
                L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
                E = L.shape[-1]
                L_last = L.view(bsz, seq_len, E, E)[:, -1, :, :]
                per_layer.append((L_last.float() ** 2).sum(dim=(-1, -2)))          # tr(LL^T)
            ilv.append(torch.stack(per_layer, dim=0).mean(dim=0).cpu())
    finally:
        rec.remove()
    probs = torch.cat(probs_all).numpy()
    n_arr = np.asarray(n_choices_list, dtype=int)
    ent = -(probs * np.log(np.clip(probs, 1e-12, None)) * (probs > 0)).sum(axis=1)
    return {
        "probs": probs,
        "pred_letter": [LETTERS[k] for k in probs.argmax(axis=1)],
        "letter_entropy": ent.astype(float),
        "letter_entropy_norm": np.asarray([normalized_entropy(h, n) for h, n in zip(ent, n_arr)], dtype=float),
        "one_minus_maxprob": (1.0 - probs.max(axis=1)).astype(float),
        "gate_entropy_last_fcvr": torch.cat(gate_fcvr).numpy().astype(float),
        "ilv_last": torch.cat(ilv).numpy().astype(float),
    }


def score_readout(model, tokenizer, fcvr_layers, domains, system_instruction, args, codes):
    """Final-position read-out per domain (evaluate_ilv_ood_arms.evaluate_arm
    inner loop; blocks are compare_domain/macro_delta-compatible)."""
    out = {}
    for code in codes:
        d = domains[code]
        prompts = prompts_for_rows(d.rows, tokenizer, system_instruction)
        lengths = np.asarray([len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts], dtype=float)
        nch = [n_choices_of(ex) for ex in d.rows]
        mc_seed = domain_sampling_seed(args.sampling_seed, DOMAIN_SEED_INDEX[code])
        torch.manual_seed(mc_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(mc_seed)
        raw = readout_masked(model, tokenizer, prompts, nch, fcvr_layers, args.batch_size)
        scores = {k: np.asarray(raw[k], dtype=float) for k in READOUT_SIGNALS}
        scores["prompt_len"] = lengths
        scores["n_choices"] = np.asarray(nch, dtype=float)
        for name, values in scores.items():
            if values.shape != (len(d.rows),):
                raise RuntimeError(f"{code}/{name}: shape {values.shape}, expected {(len(d.rows),)}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"{code}/{name}: non-finite values found")
        golds = [gold_letter_of(ex) for ex in d.rows]
        preds = raw["pred_letter"]
        correct = [None if g is None else (p == g) for g, p in zip(golds, preds)]
        out[code] = {"ids": list(d.ids), "scores": scores, "mc_seed": mc_seed,
                     "gold": golds, "pred": preds, "correct": correct}
        print(f"  readout {code}: n={len(d.rows)} ILV={scores['ilv_last'].mean():.4f} "
              f"GateEnt={scores['gate_entropy_last_fcvr'].mean():.4f} prompt_tokens={lengths.mean():.1f} "
              f"n_choices={int(scores['n_choices'][0])}")
    return out


@torch.no_grad()
def trace_sequence(model, causal_model, tokenizer, fcvr_layers, ids, spans, choice_ids_t, n_choices,
                   gold_letter, seed, online=None):
    """One batch-1 forward over `ids` (no padding); per-position records and
    the example-level aggregates. Row t = 'position t, about to predict t+1'.
    `online` (gen only): dict of per-position arrays from the decoding-time hooks."""
    device = model.device
    seq_len = len(ids)
    cats = categories_from_spans(spans, seq_len)
    n_prompt = spans["n_prompt"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    logits = model(input_ids=input_ids.unsqueeze(0)).logits[0].float()             # [seq, vocab]
    logp = F.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)                                      # [seq]
    surprisal = torch.full((seq_len,), float("nan"), device=device)
    surprisal[:-1] = -logp[torch.arange(seq_len - 1, device=device), input_ids[1:]]

    per_layer = []
    for l in fcvr_layers:
        L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
        E = L.shape[-1]
        L = L.view(1, seq_len, E, E)[0].float()
        per_layer.append((L ** 2).sum(dim=(-1, -2)))                                # tr(LL^T) [seq]
    ilv_layers = torch.stack(per_layer, dim=0)                                      # [n_layers, seq]
    ilv_sorted = ilv_layers.sort(dim=0, descending=True).values
    ilv_mean = ilv_layers.mean(dim=0)
    ilv_max = ilv_sorted[0]
    ilv_secondmax = ilv_sorted[1] if len(fcvr_layers) > 1 else ilv_sorted[0]

    # Letter probe at the final prompt row, masked to the example's choices.
    choice_logits = logits[n_prompt - 1, choice_ids_t]
    mask = torch.arange(len(LETTERS), device=device) < n_choices
    choice_probs = F.softmax(choice_logits.masked_fill(~mask, float("-inf")), dim=0)
    pred_letter = LETTERS[int(choice_probs.argmax())]
    probs_np = choice_probs.cpu().numpy()
    h = letter_entropy_of(probs_np)

    ilv_mean_np = ilv_mean.cpu().numpy()
    ilv_layers_np = ilv_layers.cpu().numpy()
    ilv_max_np, ilv_2nd_np = ilv_max.cpu().numpy(), ilv_secondmax.cpu().numpy()
    entropy_np, surprisal_np = entropy.cpu().numpy(), surprisal.cpu().numpy()
    tok_strs = [tokenizer.decode([i]) for i in ids]
    expl_pos = spans["explanation"]
    answer_correct = pred_letter == gold_letter

    records = []
    for t in range(seq_len):
        rel = (expl_pos.index(t)) / max(1, len(expl_pos) - 1) if cats[t] == "explanation" else None
        lo = max(0, t - 9)
        r = {
            "t": t, "category": cats[t], "tok": tok_strs[t],
            "next_tok": tok_strs[t + 1] if t + 1 < seq_len else None,
            "ilv_mean": float(ilv_mean_np[t]), "ilv_max": float(ilv_max_np[t]),
            "ilv_secondmax": float(ilv_2nd_np[t]),
            "ilv_runmean10": float(ilv_mean_np[lo:t + 1].mean()),
            "ilv_per_layer": [float(v) for v in ilv_layers_np[:, t]],
            "entropy": float(entropy_np[t]),
            "surprisal": float(surprisal_np[t]) if t < seq_len - 1 else None,
            "answer_correct": bool(answer_correct),
            "rel_expl_pos": rel,
        }
        if online is not None:
            for k in ("ilv_mean", "entropy", "surprisal", "gate_entropy"):
                v = online[k][t]
                r[f"{k}_online"] = float(v) if np.isfinite(v) else None
        records.append(r)

    aggs = explanation_aggregates(ilv_mean_np, entropy_np, surprisal_np, spans, cats)
    aggs.update({
        "letter_entropy": h, "letter_entropy_norm": normalized_entropy(h, n_choices),
        "n_choices": float(n_choices), "prompt_len": float(n_prompt),
        "target_len": float(seq_len - n_prompt), "n_expl_tokens": float(len(expl_pos)),
    })
    if online is not None:
        oa = explanation_aggregates(online["ilv_mean"], online["entropy"], online["surprisal"], spans, cats,
                                    prefix="")
        for k in ("expl_ilv_mean", "expl_ilv_max", "expl_ilv_secondmax", "expl_ilv_last10_mean",
                  "expl_entropy_mean", "expl_surprisal_mean", "prompt_final_ilv", "letter_ilv"):
            aggs[k + "_online"] = oa[k]
        ge = np.asarray(online["gate_entropy"], dtype=float)[expl_pos] if expl_pos else np.array([])
        aggs["expl_gate_entropy_mean_online"] = float(np.nanmean(ge)) if ge.size and np.isfinite(ge).any() else float("nan")
        on, po = np.asarray(online["ilv_mean"], dtype=float)[expl_pos], ilv_mean_np[expl_pos]
        ok = np.isfinite(on) & np.isfinite(po)
        aggs["online_posthoc_expl_spearman"] = _spearman(on[ok], po[ok]) if ok.sum() >= 3 else float("nan")
        aggs["online_posthoc_expl_maxabsdiff"] = float(np.abs(on[ok] - po[ok]).max()) if ok.any() else float("nan")

    def rows_of(cat):
        return [t for t, c in enumerate(cats) if c == cat]

    meta = {
        "pred_letter": pred_letter, "gold_letter": gold_letter, "answer_correct": bool(answer_correct),
        "seq_ilv_argmax_category": cats[int(np.nanargmax(ilv_mean_np))],
        "category_counts": {c: cats.count(c) for c in CATEGORIES},
        "per_layer_category_mean": {c: [float(v) for v in ilv_layers_np[:, rows_of(c)].mean(axis=1)]
                                    for c in CATEGORIES if rows_of(c)},
    }
    return records, aggs, meta


@torch.no_grad()
def letter_probe(model, causal_model, fcvr_layers, prompt_ids, choice_ids_t, n_choices, seed):
    """Prompt-only forward: masked predicted letter + final-position ILV."""
    device = model.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    logits = model(input_ids=input_ids.unsqueeze(0)).logits[0, -1].float()
    mask = torch.arange(len(LETTERS), device=device) < n_choices
    probs = F.softmax(logits[choice_ids_t].masked_fill(~mask, float("-inf")), dim=0).cpu().numpy()
    per_layer = []
    for l in fcvr_layers:
        L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
        per_layer.append(float((L[-1].float() ** 2).sum()))
    h = letter_entropy_of(probs)
    return {"pred_letter": LETTERS[int(probs.argmax())], "probs": probs, "letter_entropy": h,
            "letter_entropy_norm": normalized_entropy(h, n_choices),
            "one_minus_maxprob": float(1.0 - probs.max()),
            "prompt_final_ilv_promptonly": float(np.mean(per_layer))}


@torch.no_grad()
def generate_explanation(model, tokenizer, recorder, prefix_ids, max_new_tokens, seed):
    device = model.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    recorder.reset()
    inp = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    kw = dict(attention_mask=torch.ones_like(inp), max_new_tokens=max_new_tokens, do_sample=False,
              num_beams=1, use_cache=True, eos_token_id=eos_id, pad_token_id=pad_id,
              return_dict_in_generate=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        gen = model.generate(inp, output_logits=True, **kw)
        step_logits = gen.logits
    except TypeError:
        recorder.reset()
        torch.manual_seed(seed)
        gen = model.generate(inp, output_scores=True, **kw)
        step_logits = gen.scores                                    # greedy: scores == logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_ids = [int(x) for x in gen.sequences[0, len(prefix_ids):].tolist()]
    G = len(gen_ids)
    ended_with_eos = bool(G > 0 and gen_ids[-1] == eos_id)
    return {"gen_ids": gen_ids, "G": G, "ended_with_eos": ended_with_eos,
            "truncated": bool(not ended_with_eos and G >= max_new_tokens),
            "gen_seconds": time.perf_counter() - t0,
            "step_logits": [s[0].float() for s in step_logits]}


def online_rows(recorder, P, G, step_logits=None, gen_ids=None):
    """Align the decoding-time hook calls to sequence rows (length P+G): call 0
    = prefill (P rows), call j>=1 = the single row P+j-1. The last generated
    token (row P+G-1) is never a forward input -> NaN. Returns None if the
    call pattern does not match (e.g. cache-less generate)."""
    n = P + G
    ilv_layers, ge_layers = [], []
    for l in recorder.layers:
        calls, ge = recorder.calls[l], recorder.gate_ent[l]
        if len(calls) != G or calls[0].shape[0] != P or any(c.shape[0] != 1 for c in calls[1:]):
            return None
        row = np.full(n, np.nan); grow = np.full(n, np.nan)
        row[:P] = calls[0]; grow[:P] = ge[0]
        for j in range(1, G):
            row[P + j - 1] = calls[j][0]; grow[P + j - 1] = ge[j][0]
        ilv_layers.append(row); ge_layers.append(grow)
    ilv_layers = np.stack(ilv_layers, 0)
    out = {"ilv_mean": ilv_layers.mean(0), "ilv_per_layer": ilv_layers, "gate_entropy": np.stack(ge_layers, 0).mean(0),
           "entropy": np.full(n, np.nan), "surprisal": np.full(n, np.nan)}
    if step_logits is not None and gen_ids is not None:
        for s, (lg, tok) in enumerate(zip(step_logits, gen_ids)):          # step s -> row P-1+s
            lp = F.log_softmax(lg, dim=-1)
            out["entropy"][P - 1 + s] = float(-(lp.exp() * lp).sum())
            out["surprisal"][P - 1 + s] = float(-lp[tok])
    return out


# ---------------------------------------------------------------------------
# row selection
# ---------------------------------------------------------------------------
def trace_rows_for_domain(d, cap, tokenizer, marker_ids, max_seq_tokens):
    """Rows usable for BOTH conditions in BOTH arms: valid gold letter,
    letter_question, non-empty explanation, and a teacher-forced sequence that
    fits under either arm's system prompt (the longer 'comparison' prompt is
    checked)."""
    rows, ids, funnel = [], [], Counter()
    sys_long = SYSTEM_INSTRUCTIONS["comparison"]
    for i, ex in enumerate(d.rows):
        g = gold_letter_of(ex)
        expl = str(ex.get("answer") or "").strip()
        if g is None or not ex.get("letter_question") or not expl:
            funnel["no_letter_or_explanation"] += 1
            continue
        tf_ids, _, _ = build_tf_sequence(tokenizer, arm_prompt(tokenizer, ex, sys_long), g, expl, marker_ids)
        if len(tf_ids) > max_seq_tokens:
            funnel["too_long"] += 1
            continue
        rows.append(ex); ids.append(d.ids[i]); funnel["kept"] += 1
        if cap and len(rows) >= cap:
            break
    return rows, ids, dict(funnel)


# ---------------------------------------------------------------------------
# stage runners (per loaded arm)
# ---------------------------------------------------------------------------
def load_arm(args, cfg):
    seed_everything(args.sampling_seed)
    model, tokenizer, fcvr_layers = prepare(
        make_prepare_args(args, cfg["adapter"], cfg["run_suffix"], cfg["weights_dataset"]))
    if sorted(fcvr_layers) != sorted(args.swap_layers):
        raise RuntimeError(f"loaded FCVR layers {fcvr_layers} != requested {sorted(args.swap_layers)}")
    set_routing_mode(model, fcvr_layers, args.routing)
    model.config.use_cache = True
    return model, tokenizer, fcvr_layers


def release_arm(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_trace_arm(model, tokenizer, fcvr_layers, args, arm_key, cfg, trace_domains, fhs):
    """tf / gen condition for one arm. Streams per-token / per-example / text
    rows; returns {code: list of per-example rows}."""
    causal_model = model.base_model.model.model
    device = model.device
    condition = args.stage
    system_instruction = SYSTEM_INSTRUCTIONS[cfg["system_prompt"]]
    cell = cell_name(arm_key, cfg["system_prompt"])
    marker_ids = marker_token_ids(tokenizer)
    letter_ids = letter_token_ids(tokenizer)
    choice_ids_t = torch.tensor(letter_ids, device=device)
    recorder = OnlineILVRecorder(causal_model, fcvr_layers) if condition == "gen" else None
    out, audit_printed = {}, False
    try:
        for code, (rows, ids, funnel) in trace_domains.items():
            ex_rows = []
            is_ood = code != args.id_dataset
            for ex_i, (ex, ex_id) in enumerate(tqdm(list(zip(rows, ids)), desc=f"{condition} {cell} {code}")):
                seed = args.sampling_seed * 100003 + DOMAIN_SEED_INDEX[code] * 100000 + ex_i
                gold = gold_letter_of(ex)
                nch = n_choices_of(ex)
                prompt = arm_prompt(tokenizer, ex, system_instruction)
                prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
                if ex_i == 0 and not check_prefix_property(tokenizer, prompt, marker_ids, letter_ids):
                    print(f"  WARNING [{cell} {code}]: string tokenisation of prompt+letter+marker differs "
                          f"from id concatenation; boundary_clean will flag affected rows")
                extra = {}
                online = None
                if condition == "tf":
                    ids_full, spans, clean = build_tf_sequence(
                        tokenizer, prompt, gold, str(ex["answer"]), marker_ids)
                    records, aggs, meta = trace_sequence(
                        model, causal_model, tokenizer, fcvr_layers, ids_full, spans, choice_ids_t, nch, gold, seed)
                    extra["boundary_clean"] = bool(clean)
                else:
                    probe = letter_probe(model, causal_model, fcvr_layers, prompt_ids, choice_ids_t, nch, seed)
                    pred = probe["pred_letter"]
                    prefix = prompt_ids + [letter_ids[LETTERS.index(pred)]] + marker_ids
                    g = generate_explanation(model, tokenizer, recorder, prefix, args.max_new_tokens, seed)
                    P, G = len(prefix), g["G"]
                    n_expl = G - 1 if g["ended_with_eos"] else G
                    spans = build_spans(len(prompt_ids), 1, len(marker_ids), n_expl, g["ended_with_eos"])
                    ids_full = prefix + g["gen_ids"]
                    online = online_rows(recorder, P, G, g["step_logits"], g["gen_ids"])
                    records, aggs, meta = trace_sequence(
                        model, causal_model, tokenizer, fcvr_layers, ids_full, spans, choice_ids_t, nch, gold, seed,
                        online=online)
                    gen_text = tokenizer.decode(g["gen_ids"], skip_special_tokens=True)
                    extra.update({
                        "boundary_clean": True, "online_ok": online is not None,
                        "posthoc_pred_letter": meta["pred_letter"],
                        "pred_letter": pred, "answer_correct": bool(pred == gold),
                        "letter_entropy_promptonly": probe["letter_entropy"],
                        "letter_entropy_norm_promptonly": probe["letter_entropy_norm"],
                        "prompt_final_ilv_promptonly": probe["prompt_final_ilv_promptonly"],
                        "gen_len": float(G), "ended_with_eos": g["ended_with_eos"], "truncated": g["truncated"],
                        "gen_seconds": g["gen_seconds"],
                        "tokens_per_second": G / max(g["gen_seconds"], 1e-9),
                    })
                    meta["pred_letter"], meta["answer_correct"] = pred, bool(pred == gold)
                    for r in records:
                        r["answer_correct"] = bool(pred == gold)
                    fhs["texts"].write(json.dumps({
                        "arm": arm_key, "cell": cell, "dataset": code, "is_ood": is_ood, "id": ex_id,
                        "pred_letter": pred, "gold_letter": gold, "n_choices": nch, "gen_text": gen_text,
                        "n_gen_tokens": G, "ended_with_eos": g["ended_with_eos"], "truncated": g["truncated"],
                        "gold_explanation": str(ex["answer"]).strip()}) + "\n")
                if not audit_printed:
                    audit_printed = True
                    print(f"  [{cell}] boundary audit (first example, id={ex_id}):")
                    for r in [r for r in records if r["category"] in ("prompt_final", "letter", "marker", "eos")
                              or (r["category"] == "explanation" and r["rel_expl_pos"] == 0.0)][:8]:
                        print(f"    t={r['t']:4d} {r['category']:12s} tok={r['tok']!r} -> next={r['next_tok']!r}")
                row = {"arm": arm_key, "cell": cell, "system_prompt": cfg["system_prompt"],
                       "condition": condition, "dataset": code, "is_ood": is_ood, "id": ex_id,
                       "mc_seed": seed, **meta, **aggs, **extra}
                ex_rows.append(row)
                fhs["perexample"].write(json.dumps(row) + "\n")
                lo_t = 0 if args.pertoken_scope == "all" else spans["n_prompt"] - 1
                for r in records:
                    if r["t"] >= lo_t:
                        fhs["pertoken"].write(json.dumps({
                            "arm": arm_key, "cell": cell, "condition": condition, "dataset": code,
                            "is_ood": is_ood, "example_id": ex_id, "example_index": ex_i, **r}) + "\n")
            for fh in fhs.values():
                fh.flush()
            out[code] = ex_rows
            v = np.asarray([r["expl_ilv_mean"] for r in ex_rows], dtype=float)
            pf = np.asarray([r["prompt_final_ilv"] for r in ex_rows], dtype=float)
            print(f"  {condition} {cell} {code}: n={len(ex_rows)} funnel={funnel} "
                  f"expl_ILV={np.nanmean(v) if np.isfinite(v).any() else float('nan'):.4f} "
                  f"promptfinal_ILV={pf.mean():.4f}"
                  + (f" empty_expl={int((v != v).sum())}" if condition == "gen" else ""))
    finally:
        if recorder is not None:
            recorder.remove()
    return out


# ---------------------------------------------------------------------------
# blocks + statistics
# ---------------------------------------------------------------------------
def blocks_from_rows(rows):
    """per-example rows -> {arm: {code: {ids, scores{k: np.array}, meta: [rows]}}}
    (paired_arrays-compatible). Every finite/None numeric field becomes a score."""
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {}).setdefault(r["dataset"], []).append(r)
    blocks = {}
    for arm, doms in by.items():
        blocks[arm] = {}
        for code, rs in doms.items():
            keys = set()
            for r in rs:
                keys |= {k for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            scores = {k: np.asarray([float(r[k]) if r.get(k) is not None else np.nan for r in rs], dtype=float)
                      for k in keys}
            blocks[arm][code] = {"ids": [r["id"] for r in rs], "scores": scores, "meta": rs}
    return blocks


def score_table(args, blocks, codes, score_names):
    """{code: {n_per_class, scores{name: {auroc, auprc, id_mean, ood_mean, quality, n_valid_id, n_valid_ood}}}}"""
    out = {}
    for code in codes:
        if code not in blocks or args.id_dataset not in blocks:
            continue
        dom = {"n_per_class": None, "scores": {}}
        for name in score_names:
            if name not in blocks[code]["scores"]:
                continue
            y, s, _, n = paired_arrays(blocks, args.id_dataset, code, name)
            dom["n_per_class"] = n
            summ = {"auroc": metric_summary(y, s, auroc, args.n_boot, args.bootstrap_seed),
                    "auprc": metric_summary(y, s, auprc, args.n_boot, args.bootstrap_seed),
                    "id_mean": float(np.nanmean(s[:n])) if np.isfinite(s[:n]).any() else float("nan"),
                    "ood_mean": float(np.nanmean(s[n:])) if np.isfinite(s[n:]).any() else float("nan"),
                    "n_valid_id": int(np.isfinite(s[:n]).sum()), "n_valid_ood": int(np.isfinite(s[n:]).sum())}
            summ["quality"] = signal_quality(summ["auroc"]) if np.isfinite(summ["auroc"]["point"]) else "N/A"
            summ["low_valid_fraction"] = bool(min(summ["n_valid_id"], summ["n_valid_ood"]) < 0.8 * n)
            dom["scores"][name] = summ
        out[code] = dom
    return out


def aligned_pair(block_x, block_y, id_code, ood_code, score):
    """Balanced ID/OoD arrays for two blocks sharing example ids (order of x)."""
    y_ids = {c: {i: k for k, i in enumerate(block_y[c]["ids"])} for c in (id_code, ood_code)}
    keep = {c: [k for k, i in enumerate(block_x[c]["ids"]) if i in y_ids[c]] for c in (id_code, ood_code)}
    n = min(len(keep[id_code]), len(keep[ood_code]))
    if n < 2:
        return None
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    xs, ys = [], []
    for c in (id_code, ood_code):
        kx = keep[c][:n]
        xs.append(block_x[c]["scores"][score][kx])
        ys.append(block_y[c]["scores"][score][[y_ids[c][block_x[c]["ids"][k]] for k in kx]])
    return labels, np.concatenate(xs), np.concatenate(ys), n


def paired_delta_table(args, blocks_x, blocks_y, codes, score_names):
    """Per domain x score: metric(y) - metric(x) paired bootstrap (+ macro)."""
    out = {"per_domain": {}, "macro": {}}
    for name in score_names:
        per, blocks_for_macro = {}, []
        for code in codes:
            if code not in blocks_x or code not in blocks_y or name not in blocks_x[code]["scores"] \
                    or name not in blocks_y[code]["scores"]:
                continue
            ap = aligned_pair(blocks_x, blocks_y, args.id_dataset, code, name)
            if ap is None:
                continue
            labels, sx, sy, n = ap
            try:
                d = paired_bootstrap_delta(labels, sx, sy, auroc, args.n_boot, args.bootstrap_seed)
            except ValueError:
                continue
            per[code] = {**d, "n_per_class": n, "verdict": delta_verdict(d)}
            blocks_for_macro.append((labels, sx, sy))
        out["per_domain"][name] = per
        if blocks_for_macro:
            rng = np.random.default_rng(args.bootstrap_seed)
            points = [auroc(y, b) - auroc(y, a) for y, a, b in blocks_for_macro]
            boot = []
            for _ in range(args.n_boot):
                ds = []
                for y, a, b in blocks_for_macro:
                    cls = [np.flatnonzero(y == k) for k in (0, 1)]
                    idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
                    ds.append(auroc(y[idx], b[idx]) - auroc(y[idx], a[idx]))
                if np.all(np.isfinite(ds)):
                    boot.append(float(np.mean(ds)))
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
            d = {"point": float(np.nanmean(points)), "lo": float(lo), "hi": float(hi), "n_boot": len(boot)}
            out["macro"][name] = {**d, "verdict": delta_verdict(d)}
    return out


def category_profiles(args, arm_blocks):
    prof, layers = {}, {}
    for code, b in arm_blocks.items():
        prof[code] = {}
        acc = {}
        for m in b["meta"]:
            for c, vec in (m.get("per_layer_category_mean") or {}).items():
                a = acc.setdefault(c, [np.zeros(len(vec)), 0])
                a[0] += np.asarray(vec, dtype=float); a[1] += 1
        for c in CATEGORIES:
            key = PROFILE_KEY[c]
            vals = [v for v in b["scores"].get(key, []) if np.isfinite(v)]
            prof[code][c] = bootstrap_mean_ci(vals, n_boot=args.n_boot, seed=args.bootstrap_seed) if vals else None
        layers[code] = {c: [float(v) for v in (tot / cnt)] for c, (tot, cnt) in acc.items()}
    return prof, layers


def gen_health(arm_blocks):
    out = {}
    for code, b in arm_blocks.items():
        rs = b["meta"]
        n = len(rs)
        if not n:
            continue
        gl = np.asarray([r.get("gen_len", np.nan) for r in rs], dtype=float)
        ne = np.asarray([r.get("n_expl_tokens", np.nan) for r in rs], dtype=float)
        out[code] = {
            "n": n,
            "n_valid_expl": int((ne > 0).sum()),
            "frac_empty_explanation": float((ne == 0).mean()),
            "frac_ended_with_eos": float(np.mean([bool(r.get("ended_with_eos")) for r in rs])),
            "frac_truncated": float(np.mean([bool(r.get("truncated")) for r in rs])),
            "gen_len_mean": float(np.nanmean(gl)) if np.isfinite(gl).any() else float("nan"),
            "gen_len_median": float(np.nanmedian(gl)) if np.isfinite(gl).any() else float("nan"),
            "n_expl_tokens_mean": float(np.nanmean(ne)) if np.isfinite(ne).any() else float("nan"),
            "letter_accuracy": float(np.mean([bool(r.get("answer_correct")) for r in rs])),
            "pred_letter_hist": dict(Counter(r.get("pred_letter") for r in rs)),
            "frac_pred_E": float(np.mean([r.get("pred_letter") == "E" for r in rs])),
            "online_ok_frac": float(np.mean([bool(r.get("online_ok", True)) for r in rs])),
            "boundary_clean_frac": float(np.mean([bool(r.get("boundary_clean", True)) for r in rs])),
            "tokens_per_second_mean": float(np.nanmean([r.get("tokens_per_second", np.nan) for r in rs]))
            if any("tokens_per_second" in r for r in rs) else float("nan"),
        }
    return out


def online_agreement(arm_blocks):
    out = {}
    for code, b in arm_blocks.items():
        s = b["scores"]
        if "expl_ilv_mean_online" not in s:
            continue
        a, p = s["expl_ilv_mean_online"], s["expl_ilv_mean"]
        ok = np.isfinite(a) & np.isfinite(p)
        per_tok = s.get("online_posthoc_expl_spearman", np.array([]))
        out[code] = {
            "n": int(ok.sum()),
            "perexample_spearman_expl_ilv_mean": _spearman(a[ok], p[ok]) if ok.sum() >= 3 else float("nan"),
            "pertoken_spearman_mean": float(np.nanmean(per_tok)) if np.isfinite(per_tok).any() else float("nan"),
            "pertoken_maxabsdiff_max": float(np.nanmax(s.get("online_posthoc_expl_maxabsdiff", np.array([np.nan]))))
            if np.isfinite(s.get("online_posthoc_expl_maxabsdiff", np.array([np.nan]))).any() else float("nan"),
        }
    return out


def stage1_stats(args, blocks):
    arms = {k: blocks[k] for k in ARM_KEYS}
    summary = {"domains": {}, "macro": {}, "extra_signals": {}}
    for code in args.ood_datasets:
        summary["domains"][code] = compare_domain(args, arms, code)
        summary["extra_signals"][code] = {
            k: {"arm_a": score_table(args, arms["armA-letter"], [code], [k])[code]["scores"].get(k),
                "arm_b": score_table(args, arms["armB-ansexp"], [code], [k])[code]["scores"].get(k)}
            for k in ("letter_entropy_norm",)
        }
    summary["macro"]["ilv_last"] = {
        "auroc": macro_delta(args, arms, args.ood_datasets, "ilv_last", auroc),
        "auprc": macro_delta(args, arms, args.ood_datasets, "ilv_last", auprc),
    }
    return summary


def trace_stats(args, blocks, readout_blocks=None):
    summary = {"arms": {}, "delta_b_minus_a": {}, "readout": {}}
    for arm_key in ARM_KEYS:
        ab = blocks.get(arm_key, {})
        if not ab:
            continue
        codes = [c for c in args.ood_datasets if c in ab]
        prof, layers = category_profiles(args, ab)
        arm_out = {
            "n": {c: len(b["ids"]) for c, b in ab.items()},
            "ood": score_table(args, ab, codes, TRACE_SCORES),
            "category_profile": prof, "per_layer_category_mean": layers,
            "gen_health": gen_health(ab),
        }
        if args.stage == "gen":
            arm_out["ood_online"] = score_table(args, ab, codes, ONLINE_SCORES)
            arm_out["online_vs_posthoc"] = online_agreement(ab)
        if readout_blocks and arm_key in readout_blocks:
            arm_out["readout_signals"] = score_table(args, readout_blocks[arm_key], codes, READOUT_SIGNALS)
        summary["arms"][arm_key] = arm_out
    if all(k in blocks for k in ARM_KEYS):
        codes = [c for c in args.ood_datasets if all(c in blocks[k] for k in ARM_KEYS)]
        summary["delta_b_minus_a"] = paired_delta_table(args, blocks["armA-letter"], blocks["armB-ansexp"],
                                                        codes, TRACE_SCORES)
    if readout_blocks and all(k in readout_blocks for k in ARM_KEYS):
        arms = {k: readout_blocks[k] for k in ARM_KEYS}
        codes = [c for c in args.ood_datasets if all(c in arms[k] for k in ARM_KEYS)]
        summary["readout"] = {"domains": {c: compare_domain(args, arms, c) for c in codes}}
        if codes:
            summary["readout"]["macro_ilv_last_auroc"] = macro_delta(args, arms, codes, "ilv_last", auroc)
    return summary


def id_list_sha1(ids):
    return hashlib.sha1("\n".join(map(str, ids)).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _cell(summ):
    if not summ or not np.isfinite(summ["auroc"]["point"]):
        return "n/a"
    a = summ["auroc"]
    flag = " ⚠" if summ.get("low_valid_fraction") else ""
    return f"{f3(a['point'])} [{f3(a['lo'])},{f3(a['hi'])}] ({summ['quality']}){flag}"


def build_report(args, summary):
    cfg = summary["config"]
    lines = [
        f"# OoD detection from ILV: answer position vs explanation region -- stage `{args.stage}`",
        "",
        f"ID = `{args.id_dataset}`; split = `{args.split}`; data seed = {args.data_seed}; MC seed = "
        f"{args.sampling_seed}; S = {args.num_samples}; routing = {args.routing}; OoD = {args.ood_datasets}.",
        "",
        "Higher score is fixed a priori to mean OoD; AUROC < 0.5 is an inverted signal and is never flipped;",
        "a CI crossing 0.5 (or a delta CI crossing 0) is inconclusive. ⚠ marks cells where < 80 % of one class",
        "had a finite score (e.g. empty generated explanations); NaNs are dropped, never imputed.",
        "",
    ]
    if args.stage == "stage1":
        s1 = summary["stage1"]
        lines += ["## Final-position ILV OoD AUROC (paper protocol)", "",
                  "| OoD domain | n/class | Arm A answer-only | Arm B answer+explanation | Δ B-A [95% CI] | Verdict |",
                  "|---|---:|---:|---:|---:|---|"]
        for code in args.ood_datasets:
            d = s1["domains"][code]; s = d["signals"]["ilv_last"]
            a, b = s["arm_a"]["auroc"], s["arm_b"]["auroc"]
            lines.append(f"| {code} | {d['n_per_class']} | {f3(a['point'])} [{f3(a['lo'])},{f3(a['hi'])}] "
                         f"({signal_quality(a)}) | {f3(b['point'])} [{f3(b['lo'])},{f3(b['hi'])}] "
                         f"({signal_quality(b)}) | {ci_text(s['delta_b_minus_a']['auroc'])} | **{s['verdict_auroc']}** |")
        m = s1["macro"]["ilv_last"]["auroc"]
        lines += ["", f"**Macro Arm B - Arm A ILV AUROC:** {ci_text(m['delta_b_minus_a'])} -> **{m['verdict']}**.", "",
                  "### Baseline signals (AUROC, arm A / arm B)", "",
                  "| OoD domain | " + " | ".join(READOUT_SIGNALS) + " |", "|---|" + "---:|" * len(READOUT_SIGNALS)]
        for code in args.ood_datasets:
            row = [f"| {code} "]
            for sig in READOUT_SIGNALS:
                if sig == "letter_entropy_norm":
                    e = s1["extra_signals"][code][sig]
                    row.append(f"| {f3(e['arm_a']['auroc']['point']) if e['arm_a'] else 'n/a'} / "
                               f"{f3(e['arm_b']['auroc']['point']) if e['arm_b'] else 'n/a'} ")
                else:
                    s = s1["domains"][code]["signals"][sig]
                    row.append(f"| {f3(s['arm_a']['auroc']['point'])} / {f3(s['arm_b']['auroc']['point'])} ")
            lines.append("".join(row) + "|")
    else:
        tr = summary["trace"]
        cond = "teacher-forced gold explanation" if args.stage == "tf" else "model-generated explanation (post-hoc read-out primary)"
        lines += [f"## Explanation-region scores as OoD detectors -- condition: {cond}", ""]
        for arm_key, a in tr["arms"].items():
            codes = [c for c in args.ood_datasets if c in a["ood"]]
            lines += [f"### {arm_key}", "", "| score | " + " | ".join(codes) + " |", "|---|" + "---:|" * len(codes)]
            for name in TRACE_SCORES:
                row = [f"| {name} "]
                for code in codes:
                    row.append(f"| {_cell(a['ood'][code]['scores'].get(name))} ")
                lines.append("".join(row) + "|")
            if "readout_signals" in a:
                lines += ["", "Final-position (batched, paper-protocol) read-out on the same rows:", "",
                          "| signal | " + " | ".join(codes) + " |", "|---|" + "---:|" * len(codes)]
                for sig in READOUT_SIGNALS:
                    row = [f"| {sig} "]
                    for code in codes:
                        row.append(f"| {_cell(a['readout_signals'].get(code, {}).get('scores', {}).get(sig))} ")
                    lines.append("".join(row) + "|")
            if args.stage == "gen":
                lines += ["", "Online (decoding-time hook) variants:", "",
                          "| score | " + " | ".join(codes) + " |", "|---|" + "---:|" * len(codes)]
                for name in ONLINE_SCORES:
                    row = [f"| {name} "]
                    for code in codes:
                        row.append(f"| {_cell(a['ood_online'][code]['scores'].get(name))} ")
                    lines.append("".join(row) + "|")
                lines += ["", "Generation health:", "",
                          "| dataset | n | empty expl | ended EOS | truncated | gen_len mean/med | letter acc | E-rate | online ok | tok/s |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
                for code, h in a["gen_health"].items():
                    lines.append(f"| {code} | {h['n']} | {f3(h['frac_empty_explanation'])} | {f3(h['frac_ended_with_eos'])} "
                                 f"| {f3(h['frac_truncated'])} | {h['gen_len_mean']:.1f}/{h['gen_len_median']:.0f} "
                                 f"| {f3(h['letter_accuracy'])} | {f3(h['frac_pred_E'])} | {f3(h['online_ok_frac'])} "
                                 f"| {h['tokens_per_second_mean']:.1f} |")
                lines += ["", "Online vs post-hoc agreement (explanation rows):", "",
                          "| dataset | n | per-example Spearman(expl_ilv_mean) | mean per-token Spearman | max |Δ| |",
                          "|---|---:|---:|---:|---:|"]
                for code, o in a["online_vs_posthoc"].items():
                    lines.append(f"| {code} | {o['n']} | {f3(o['perexample_spearman_expl_ilv_mean'])} "
                                 f"| {f3(o['pertoken_spearman_mean'])} | {f3(o['pertoken_maxabsdiff_max'])} |")
            else:
                lines += ["", "Rows / explanation length:", "", "| dataset | n | n_expl_tokens mean | letter acc |",
                          "|---|---:|---:|---:|"]
                for code, h in a["gen_health"].items():
                    lines.append(f"| {code} | {h['n']} | {h['n_expl_tokens_mean']:.1f} | {f3(h['letter_accuracy'])} |")
            lines += ["", "Category-profile mean ILV (ID vs OoD):", "",
                      "| category | " + " | ".join(a["category_profile"]) + " |", "|---|" + "---:|" * len(a["category_profile"])]
            for cat in CATEGORIES:
                row = [f"| {cat} "]
                for code in a["category_profile"]:
                    p = a["category_profile"][code].get(cat)
                    row.append(f"| {f3(p['mean'])} " if p else "| n/a ")
                lines.append("".join(row) + "|")
            lines.append("")
        d = tr.get("delta_b_minus_a", {})
        if d.get("per_domain"):
            codes = sorted({c for per in d["per_domain"].values() for c in per})
            lines += ["### Paired Δ AUROC (arm B - arm A)", "", "| score | " + " | ".join(codes) + " | macro |",
                      "|---|" + "---:|" * (len(codes) + 1)]
            for name in TRACE_SCORES:
                per = d["per_domain"].get(name, {})
                row = [f"| {name} "]
                for code in codes:
                    row.append(f"| {ci_text(per[code])} " if code in per else "| n/a ")
                m = d["macro"].get(name)
                row.append(f"| {ci_text(m)} ({m['verdict']}) " if m else "| n/a ")
                lines.append("".join(row) + "|")
            lines.append("")
        g = summary.get("gen_minus_tf")
        if g:
            lines += ["### Paired Δ AUROC (generated - teacher-forced), within arm", ""]
            for arm_key, dd in g.items():
                codes = sorted({c for per in dd["per_domain"].values() for c in per})
                lines += [f"**{arm_key}**", "", "| score | " + " | ".join(codes) + " | macro |",
                          "|---|" + "---:|" * (len(codes) + 1)]
                for name in TRACE_SCORES:
                    per = dd["per_domain"].get(name, {})
                    row = [f"| {name} "]
                    for code in codes:
                        row.append(f"| {ci_text(per[code])} " if code in per else "| n/a ")
                    m = dd["macro"].get(name)
                    row.append(f"| {ci_text(m)} " if m else "| n/a ")
                    lines.append("".join(row) + "|")
                lines.append("")
        rd = tr.get("readout", {}).get("domains")
        if rd:
            lines += ["### Final-position ILV, paired B-A (same trace rows)", "",
                      "| OoD domain | n/class | Arm A | Arm B | Δ B-A [95% CI] | Verdict |", "|---|---:|---:|---:|---:|---|"]
            for code, dom in rd.items():
                s = dom["signals"]["ilv_last"]; a, b = s["arm_a"]["auroc"], s["arm_b"]["auroc"]
                lines.append(f"| {code} | {dom['n_per_class']} | {f3(a['point'])} ({signal_quality(a)}) | "
                             f"{f3(b['point'])} ({signal_quality(b)}) | {ci_text(s['delta_b_minus_a']['auroc'])} | {s['verdict_auroc']} |")
            lines.append("")
    lines += ["## Caveats", "",
              "- Four-choice (obqa_gen, medexqa, scienceqa, arc) and five-choice (ecqa, aqua_rat) prompts differ in",
              "  option count; letter entropy is masked to valid choices and `letter_entropy_norm` = H/log(n).",
              "- ECQA explanation = `taskA_pos` (positives); AQuA-RAT rationales have terminal answer phrases",
              "  removed; ScienceQA = text-only / no-hint / 4-choice rows, all subjects.",
              "- `expl_ilv_max`/`secondmax` grow with explanation length; `n_expl_tokens` is the length baseline.",
              "- gen: arm A never trained past the letter, so empty/truncated generations are reported, not imputed;",
              "  `eos_ilv` is NaN when generation hit the cap; online `eos_ilv` is undefined by construction.",
              "", f"Git commit: `{cfg['git_commit']}`."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# cross-check against saved runs
# ---------------------------------------------------------------------------
def crosscheck(args, rows, ref_path):
    """Join on (arm, dataset, id); report Spearman / max|Δ| per shared numeric key
    and AUROC deltas for the primary keys."""
    ref = {}
    with open(ref_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            arm = r.get("arm")
            if arm is None and r.get("cell"):
                arm = {v: k for k, v in ARM_SHORT.items()}.get(r["cell"].split("-")[0])
            ref[(arm, r.get("dataset"), str(r.get("id")))] = r
    keys = ["ilv_last", "prompt_final_ilv", "expl_ilv_mean", "expl_ilv_max", "letter_entropy", "gate_entropy_last_fcvr"]
    print(f"\n== cross-check vs {ref_path} ({len(ref)} reference rows) ==")
    out = {}
    for arm_key in ARM_KEYS:
        for code in [args.id_dataset, *args.ood_datasets]:
            mine = [r for r in rows if r["arm"] == arm_key and r["dataset"] == code]
            pairs = [(r, ref[(arm_key, code, str(r["id"]))]) for r in mine if (arm_key, code, str(r["id"])) in ref]
            if len(pairs) < 3:
                continue
            for k in keys:
                a = np.asarray([p[0].get(k, np.nan) for p in pairs], dtype=float)
                b = np.asarray([p[1].get(k, np.nan) for p in pairs], dtype=float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 3:
                    continue
                rho, mad = _spearman(a[ok], b[ok]), float(np.abs(a[ok] - b[ok]).max())
                out[f"{arm_key}/{code}/{k}"] = {"n": int(ok.sum()), "spearman": rho, "max_abs_diff": mad}
                print(f"  {arm_key:12s} {code:12s} {k:22s} n={ok.sum():4d} spearman={rho:.4f} max|Δ|={mad:.4g}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def write_summary(args, paths, summary):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    report = build_report(args, summary)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print("Saved:")
    for p in paths.values():
        if os.path.exists(p):
            print(f"  {p}")


def base_config(args, setup, arm_cfgs):
    return {
        "git_commit": git_revision(), "stage": args.stage, "model_shortcode": args.model_shortcode,
        "id_dataset": args.id_dataset, "ood_datasets": args.ood_datasets,
        "ood_description": {c: setup["ood_description"].get(c, "unclassified shift") for c in args.ood_datasets},
        "split": args.split, "n_per_domain_cap": args.n_per_domain, "trace_n_id": args.trace_n_id,
        "trace_n_ood": args.trace_n_ood, "max_new_tokens": args.max_new_tokens,
        "max_seq_tokens": args.max_seq_tokens, "pertoken_scope": args.pertoken_scope,
        "data_seed": args.data_seed, "sampling_seed": args.sampling_seed, "num_samples": args.num_samples,
        "routing": args.routing, "swap_layers": sorted(args.swap_layers), "prior_source": args.prior_source,
        "arms": {k: {kk: arm_cfgs[k][kk] for kk in ("adapter", "run_suffix", "weights_dataset", "system_prompt")}
                 for k in ARM_KEYS},
        "n_boot": args.n_boot, "bootstrap_seed": args.bootstrap_seed,
        "domain_seed_index": DOMAIN_SEED_INDEX, "nine_scores": list(NINE_SCORES),
        "primary_readout": "posthoc" if args.stage == "gen" else "teacher_forced",
        "generation": ("predicted letter (masked) + forced '\\nExplanation:' + greedy decode; "
                       "post-hoc batch-1 forward over the finished sequence is primary") if args.stage == "gen" else None,
        "sign": "higher score => OoD; never flip inverted AUROC",
    }


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    args = parse_args(argv)
    setup, arm_cfgs = resolve_arm_setup(args)
    validate_args(args)
    paths = output_paths(args)

    if args.stats_only:
        check_output_collisions([paths["json"], paths["md"]], args.overwrite)
        rows = read_rows(args.stats_only)
        readout_rows = [r for r in rows if r.get("kind") == "readout"]
        trace_rows = [r for r in rows if r.get("kind") != "readout"]
        summary = {"config": {**base_config(args, setup, arm_cfgs), "stats_only_from": args.stats_only}}
        if args.stage == "stage1":
            summary["stage1"] = stage1_stats(args, blocks_from_rows(trace_rows))
        else:
            summary["trace"] = trace_stats(args, blocks_from_rows(trace_rows),
                                           blocks_from_rows(readout_rows) if readout_rows else None)
        if args.crosscheck:
            summary["crosscheck"] = crosscheck(args, trace_rows, args.crosscheck)
        write_summary(args, paths, summary)
        return

    check_output_collisions(list(paths.values()), args.overwrite)
    setup_environment()
    print("#" * 80)
    print(f"# OoD explanation read-out -- stage {args.stage}")
    print(f"# DATA seed={args.data_seed} | MC seed={args.sampling_seed} | S={args.num_samples} | routing={args.routing}")
    print(f"# ID={args.id_dataset} | OoD={args.ood_datasets} | split={args.split}")
    for arm_key in ARM_KEYS:
        cfg = arm_cfgs[arm_key]
        print(f"# {arm_key}: adapter={cfg['adapter']} suffix={cfg['run_suffix']} "
              f"weights_ds={cfg['weights_dataset'] or args.id_dataset} prompt={cfg['system_prompt']}")
    print("#" * 80)

    domains = load_domains(args)
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {"config": base_config(args, setup, arm_cfgs)}
    summary["config"]["domain_id_sha1"] = {c: id_list_sha1(d.ids) for c, d in domains.items()}
    all_rows = []

    if args.stage == "stage1":
        blocks = {}
        for arm_key in ARM_KEYS:
            cfg = arm_cfgs[arm_key]
            print("\n" + "=" * 80 + f"\n{cfg['label']}\n" + "=" * 80)
            model, tokenizer, fcvr_layers = load_arm(args, cfg)
            try:
                blocks[arm_key] = score_readout(model, tokenizer, fcvr_layers, domains,
                                                SYSTEM_INSTRUCTIONS[cfg["system_prompt"]], args, list(domains))
            finally:
                release_arm(model)
        with open(paths["perexample"], "w", encoding="utf-8") as f:
            for arm_key, doms in blocks.items():
                for code, b in doms.items():
                    for i, ex_id in enumerate(b["ids"]):
                        row = {"arm": arm_key, "cell": cell_name(arm_key, arm_cfgs[arm_key]["system_prompt"]),
                               "dataset": code, "is_ood": code != args.id_dataset, "id": ex_id,
                               "mc_seed": b["mc_seed"], "gold_letter": b["gold"][i], "pred_letter": b["pred"][i],
                               "correct": b["correct"][i],
                               **{k: float(v[i]) for k, v in b["scores"].items()}}
                        all_rows.append(row)
                        f.write(json.dumps(row) + "\n")
        for code in domains:
            if blocks["armA-letter"][code]["ids"] != blocks["armB-ansexp"][code]["ids"]:
                raise RuntimeError(f"Arm A/B ID ordering differs for {code}")
        summary["stage1"] = stage1_stats(args, blocks)
    else:
        # Trace rows fixed BEFORE any model load (identical across arms/conditions).
        from model import load_tokenizer
        tok0 = load_tokenizer(args.model_shortcode)
        marker_ids = marker_token_ids(tok0)
        trace_domains = {}
        trace_domains[args.id_dataset] = trace_rows_for_domain(domains[args.id_dataset], args.trace_n_id, tok0,
                                                               marker_ids, args.max_seq_tokens)
        for code in args.ood_datasets:
            trace_domains[code] = trace_rows_for_domain(domains[code], args.trace_n_ood, tok0, marker_ids,
                                                        args.max_seq_tokens)
        for code, (rows, ids, funnel) in trace_domains.items():
            if not rows:
                raise SystemExit(f"trace: no usable rows for {code}: {funnel}")
            print(f"trace rows {code}: {len(rows)} kept, funnel={funnel}, n_choices={n_choices_of(rows[0])}")
        summary["config"]["trace_funnel"] = {c: v[2] for c, v in trace_domains.items()}
        summary["config"]["marker_token_ids"] = marker_ids
        trace_sub = {c: argparse.Namespace(code=c, ids=ids, rows=rows) for c, (rows, ids, _) in trace_domains.items()}

        fhs = {k: open(paths[k], "w", encoding="utf-8") for k in ("perexample", "pertoken", "texts") if k in paths}
        blocks, readout_blocks = {}, {}
        try:
            for arm_key in ARM_KEYS:
                cfg = arm_cfgs[arm_key]
                print("\n" + "=" * 80 + f"\n{cfg['label']}\n" + "=" * 80)
                model, tokenizer, fcvr_layers = load_arm(args, cfg)
                try:
                    # Batched final-position read-out on the trace rows FIRST (left padding) ...
                    rb = score_readout(model, tokenizer, fcvr_layers, trace_sub,
                                       SYSTEM_INSTRUCTIONS[cfg["system_prompt"]], args, list(trace_sub))
                    readout_blocks[arm_key] = rb
                    for code, b in rb.items():
                        for i, ex_id in enumerate(b["ids"]):
                            row = {"kind": "readout", "arm": arm_key, "dataset": code, "is_ood": code != args.id_dataset,
                                   "id": ex_id, "mc_seed": b["mc_seed"], "gold_letter": b["gold"][i],
                                   "pred_letter": b["pred"][i], "correct": b["correct"][i],
                                   **{k: float(v[i]) for k, v in b["scores"].items()}}
                            all_rows.append(row)
                            fhs["perexample"].write(json.dumps(row) + "\n")
                    # ... then the un-padded batch-1 traces.
                    per_code = run_trace_arm(model, tokenizer, fcvr_layers, args, arm_key, cfg, trace_domains, fhs)
                    rows_arm = [r for rs in per_code.values() for r in rs]
                    all_rows += rows_arm
                    blocks[arm_key] = blocks_from_rows(rows_arm)[arm_key]
                finally:
                    release_arm(model)
        finally:
            for fh in fhs.values():
                fh.close()
        summary["trace"] = trace_stats(args, blocks, readout_blocks=readout_blocks)
        if args.stage == "gen":
            tf_path = sibling_stage_path(args, "tf")
            if os.path.exists(tf_path):
                tf_blocks = blocks_from_rows([r for r in read_rows(tf_path) if r.get("kind") != "readout"])
                summary["gen_minus_tf"] = {
                    k: paired_delta_table(args, tf_blocks[k], blocks[k],
                                          [c for c in args.ood_datasets if c in tf_blocks[k] and c in blocks[k]],
                                          TRACE_SCORES)
                    for k in ARM_KEYS if k in tf_blocks and k in blocks}
                print(f"gen - tf paired deltas computed against {tf_path}")
            else:
                print(f"NOTE: no tf per-example file at {tf_path}; gen-tf deltas skipped "
                      "(run --stage tf first, or --stats_only later).")

    if args.crosscheck:
        summary["crosscheck"] = crosscheck(args, [r for r in all_rows if r.get("kind") != "readout"], args.crosscheck)
    write_summary(args, paths, summary)


if __name__ == "__main__":
    main()
