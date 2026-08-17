"""
Input-level ID-vs-OoD bridge test for the FCVR Inf-Logit-Var signal
(codex-recom-iter1 rewrite of the new-exp2 script; supervisor points 5, 6, 8).

The VMoER paper's validated Inf-Logit-Var result (Table 8) is input-level: one
score per prompt (final prompt token, averaged over the Bayesian layers)
separates ID from OoD prompts. This script asks whether the MedExQA-generation-
trained FCVR still shows that separation -- and whether what it separates is
CONTENT rather than surface form.

Protocol
--------
ID  = MedExQA prompts (the FCVR's training distribution), split = --split.
OoD = --ood_datasets, same split, same number of examples. Default
      obqa (far), medmcqa_med (NEAR: same medical domain, MCQA task shift --
      the interesting one for "semantic novelty vs law-vs-medicine"), mmlu_law (far).
Every prompt is wrapped in the same outer chat template AND, by default, its
INNER text is re-rendered in one canonical format (--inner_format generation =
"Question/Options/Explain the reasoning..." = the FCVR's training format) so
'Options' vs 'Choices' / the trailing instruction cannot drive the score.
--inner_format native reproduces the old (confounded) comparison; mcqa renders
everything as "Question/Choices/Answer:".

Controls
--------
* prompt length: AUROC(OoD) of prompt_len alone; a length-matched AUROC (ID and
  OoD examples paired within token-length bins); a residual AUROC after
  regressing each signal on log(prompt_len) (fit on the pooled ID+OoD set).
* --format_control: the SAME ID content rendered in the two inner formats
  (generation vs mcqa) is scored as if it were an ID/OoD pair. AUROC far from
  0.5 here means the score is reading the template, not the content.

Signals per prompt (batch size 1, no padding):
    ilv_last     tr(posterior cov) = ||L||_F^2 at the final prompt token, mean over FCVR layers
    ilv_mean     same, averaged over all prompt positions
    entropy_last predictive entropy at the final prompt token (baseline)
    prompt_len   token length (confound / baseline)

Reporting: AUROC/AUPRC (OoD = 1) with example-level bootstrap CIs; the sign is
fixed A PRIORI (higher ILV => OoD). AUROC < 0.5 is a NEGATIVE result -- it is
never flipped and |AUROC-0.5| is not a success criterion.

Routing: default = paper's stochastic S=35. --routing deterministic is an
ABLATION (earlier-layer routing changes later hidden states/covariances).
--arm: fcvr (trained) | untrained (fresh FCVR heads) | det (stock routers,
entropy/length only) for the baseline ladder.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import setup_environment, load_exp_dataset, generation_prompt_engineer
from utils.prompt import canonicalise_inner
from model import load_peft_model_and_adapter, load_tokenizer
from evaluate_fcvr import prepare_model_by_arm
from uq_stats import auroc, auprc, bootstrap_ci

OOD_DOMAIN = {"medmcqa_med": "near (medical domain, MCQA task shift)",
              "obqa": "far", "arc_c": "far", "arc_e": "far", "sciq": "far", "mmlu_law": "far"}


def parse_args():
    p = argparse.ArgumentParser(description="Input-level ID-vs-OoD Inf-Logit-Var bridge test (FCVR).")
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="medexqa",
                   help="ID dataset the FCVR was trained on (drives weight paths AND is the ID anchor).")
    p.add_argument("--kvq_adapter_path", type=str, required=True)
    p.add_argument("--arm", type=str, default="fcvr", choices=["fcvr", "untrained", "det"],
                   help="Baseline ladder: fcvr (trained), untrained (fresh FCVR heads), det (stock routers; entropy only).")
    p.add_argument("--swap_layers", type=int, nargs="+", default=[], help="Required unless --arm det.")
    p.add_argument("--run_suffix", type=str, default=None, help="Must match the training run's --run_suffix.")
    p.add_argument("--prior_source", type=str, default="pretrained", choices=["map", "pretrained"],
                   help="Must match the training run.")
    p.add_argument("--map_suffix", type=str, default=None,
                   help="[prior_source=map] Suffix of the MAP router weights dir used at training.")
    p.add_argument("--num_samples", type=int, default=35,
                   help="MC samples S for stochastic routing (paper: 35). Ignored if --routing deterministic.")
    p.add_argument("--routing", type=str, default="stochastic", choices=["stochastic", "deterministic"],
                   help="stochastic = paper inference (PRIMARY); deterministic = posterior-mean routing (ABLATION).")
    p.add_argument("--ood_datasets", type=str, nargs="+", default=["obqa", "medmcqa_med", "mmlu_law"],
                   help="OoD sets (load_exp_dataset shortcodes). medmcqa_med = near-domain task shift.")
    p.add_argument("--inner_format", type=str, default="generation", choices=["generation", "mcqa", "native"],
                   help="Canonical inner prompt format for ID AND OoD (default generation = FCVR training "
                        "format). native = each dataset's own format (confounded; comparison only).")
    p.add_argument("--format_control", action="store_true",
                   help="Also score ID content rendered as 'generation' vs 'mcqa' (same content, different "
                        "template) as if it were an ID/OoD pair.")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"],
                   help="Split used for the ID anchor AND every OoD set. PROTOCOL: 'val' = selection; "
                        "'test' = evaluated once per frozen configuration (pass explicitly).")
    p.add_argument("--num_examples", type=int, default=175, help="Examples per dataset.")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--n_len_bins", type=int, default=6, help="Token-length bins for the length-matched AUROC.")
    p.add_argument("--output_dir", type=str, default="results/input_level_ood")
    p.add_argument("--tag", type=str, default=None, help="Extra tag appended to the output filename.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_signals(model, tokenizer, dataset_code, num_examples, fcvr_layers, causal_model, device,
                    split="val", inner_format="generation", seed=42):
    """One prompt-only forward per example (batch 1) -> dict of np arrays."""
    ds = load_exp_dataset(dataset_code, split=split)[:num_examples]
    out = {"entropy_last": [], "prompt_len": []}
    if fcvr_layers:
        out.update({"ilv_last": [], "ilv_mean": []})
    for ex_i, ex in enumerate(tqdm(ds, desc=f"{dataset_code}[{inner_format}]")):
        inner = canonicalise_inner(ex["question"], inner_format)
        prompt = generation_prompt_engineer({"question": inner, "answer": ex.get("answer", ""), "id": ex.get("id", "")},
                                            tokenizer=tokenizer)["question"]
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048
        ).input_ids.to(device)
        seq_len = ids.shape[1]
        torch.manual_seed(seed * 100003 + ex_i)   # reproducible stochastic routing per example
        logits = model(input_ids=ids).logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        out["entropy_last"].append(float(-(logp.exp() * logp).sum().item()))
        out["prompt_len"].append(float(seq_len))

        if fcvr_layers:
            per_layer_last, per_layer_mean = [], []
            for l in fcvr_layers:
                L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
                E = L.shape[-1]
                tr = (L.view(seq_len, E, E).float() ** 2).sum(dim=(-1, -2))  # [seq] = tr(LL^T)
                per_layer_last.append(tr[-1])
                per_layer_mean.append(tr.mean())
            out["ilv_last"].append(float(torch.stack(per_layer_last).mean().item()))
            out["ilv_mean"].append(float(torch.stack(per_layer_mean).mean().item()))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _pair(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores), int), np.ones(len(ood_scores), int)])
    return labels, scores


def auc_block(id_scores, ood_scores, n_boot):
    labels, scores = _pair(id_scores, ood_scores)
    ci = bootstrap_ci(auroc, labels, scores, n_boot=n_boot)
    return {"auroc": ci["point"], "auroc_ci": [ci["lo"], ci["hi"]], "auprc": auprc(labels, scores),
            "id_mean": float(np.mean(id_scores)), "ood_mean": float(np.mean(ood_scores)),
            "direction": "OoD>ID (a-priori sign)" if np.mean(ood_scores) > np.mean(id_scores) else "OoD<ID (INVERTED)"}


def length_matched_auroc(id_sig, ood_sig, key, n_bins, seed=0):
    """Pair ID and OoD examples within token-length bins (bins over the pooled
    length distribution); AUROC on the balanced subsample. Removes the length
    confound at the price of n."""
    rng = np.random.default_rng(seed)
    lens = np.concatenate([id_sig["prompt_len"], ood_sig["prompt_len"]])
    edges = np.unique(np.quantile(lens, np.linspace(0, 1, n_bins + 1)))
    id_idx, ood_idx = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = np.flatnonzero((id_sig["prompt_len"] >= lo) & (id_sig["prompt_len"] <= hi))
        b = np.flatnonzero((ood_sig["prompt_len"] >= lo) & (ood_sig["prompt_len"] <= hi))
        k = min(a.size, b.size)
        if k == 0:
            continue
        id_idx += rng.choice(a, k, replace=False).tolist()
        ood_idx += rng.choice(b, k, replace=False).tolist()
    if len(id_idx) < 5:
        return {"auroc": float("nan"), "n_per_class": len(id_idx), "note": "too few length-overlapping examples"}
    labels, scores = _pair(id_sig[key][id_idx], ood_sig[key][ood_idx])
    return {"auroc": auroc(labels, scores), "n_per_class": len(id_idx),
            "len_id_mean": float(id_sig["prompt_len"][id_idx].mean()), "len_ood_mean": float(ood_sig["prompt_len"][ood_idx].mean())}


def residual_auroc(id_sig, ood_sig, key):
    """Regress the signal on log(prompt_len) over the pooled set; AUROC of the residual."""
    labels, scores = _pair(id_sig[key], ood_sig[key])
    x = np.log(np.concatenate([id_sig["prompt_len"], ood_sig["prompt_len"]]))
    X = np.stack([np.ones_like(x), x], 1)
    beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
    resid = scores - X @ beta
    return {"auroc": auroc(labels, resid), "slope_per_log_len": float(beta[1]),
            "r2": float(1 - resid.var() / scores.var()) if scores.var() > 0 else float("nan")}


def verdict_for(auc, lo, hi):
    if np.isnan(auc):
        return "n/a"
    if lo > 0.5 and auc >= 0.60:
        return "separation, correct direction (CI excludes 0.5)"
    if hi < 0.5 and auc <= 0.40:
        return "INVERTED -- negative result (higher ILV on ID; CI excludes 0.5)"
    return "chance / inconclusive (CI includes 0.5 or |effect| small)"


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.arm != "det" and not args.swap_layers:
        raise SystemExit("--swap_layers is required unless --arm det")
    ood_codes = [c for c in args.ood_datasets if c != args.dataset_shortcode]
    if len(ood_codes) != len(args.ood_datasets):
        print(f"SKIP {args.dataset_shortcode}: identical to the ID anchor")

    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)
    model, fcvr_layers = prepare_model_by_arm(model, args)
    print(f"Arm: {args.arm}")

    causal_model = model.base_model.model.model
    for l in fcvr_layers:
        causal_model.layers[l].block_sparse_moe.router.deterministic_readout = (args.routing == "deterministic")
    print(f"Routing mode: {'DETERMINISTIC posterior-mean (ABLATION)' if args.routing == 'deterministic' else f'STOCHASTIC S={args.num_samples} (paper inference; PRIMARY)'}")
    print(f"Inner prompt format: {args.inner_format} (ID and OoD alike)")

    if args.split == "test":
        print("#" * 72 + "\n# TEST SPLIT: evaluate ONCE per frozen configuration (selection happens on val).\n" + "#" * 72)
    print(f"\nID anchor: {args.dataset_shortcode} ({args.num_examples} {args.split} prompts)")
    id_sig = collect_signals(model, tokenizer, args.dataset_shortcode, args.num_examples, fcvr_layers,
                             causal_model, model.device, split=args.split, inner_format=args.inner_format, seed=args.seed)
    signals = [k for k in ("ilv_last", "ilv_mean", "entropy_last", "prompt_len") if k in id_sig]

    results = {
        "config": {
            "arm": args.arm, "id_dataset": args.dataset_shortcode, "ood_datasets": ood_codes,
            "ood_domain": {c: OOD_DOMAIN.get(c, "?") for c in ood_codes},
            "num_examples": args.num_examples, "split": args.split, "fcvr_layers": fcvr_layers,
            "run_suffix": args.run_suffix, "prior_source": args.prior_source, "map_suffix": args.map_suffix,
            "routing": args.routing, "num_samples": None if args.routing == "deterministic" else args.num_samples,
            "seed": args.seed, "inner_format": args.inner_format,
            "template": "generation_prompt_engineer outer chat template; inner text canonicalised per --inner_format",
            "sign_convention": "a priori: higher score => OoD. AUROC<0.5 is a negative result, never flipped.",
            "n_boot": args.n_boot,
        },
        "id_means": {k: float(v.mean()) for k, v in id_sig.items()},
        "ood": {},
        "format_control": None,
    }

    for code in ood_codes:
        print(f"\nOoD set: {code}  [{OOD_DOMAIN.get(code, '?')}]")
        ood_sig = collect_signals(model, tokenizer, code, args.num_examples, fcvr_layers, causal_model,
                                  model.device, split=args.split, inner_format=args.inner_format, seed=args.seed)
        block = {"n_id": int(len(id_sig["prompt_len"])), "n_ood": int(len(ood_sig["prompt_len"])),
                 "domain": OOD_DOMAIN.get(code, "?")}
        for sig in signals:
            b = auc_block(id_sig[sig], ood_sig[sig], args.n_boot)
            if sig != "prompt_len":
                b["length_matched"] = length_matched_auroc(id_sig, ood_sig, sig, args.n_len_bins, seed=args.seed)
                b["length_residual"] = residual_auroc(id_sig, ood_sig, sig)
            block[sig] = b
        results["ood"][code] = block

    if args.format_control:
        other = "mcqa" if args.inner_format != "mcqa" else "generation"
        print(f"\nFORMAT CONTROL: ID content in '{args.inner_format}' vs '{other}' (same questions)")
        alt = collect_signals(model, tokenizer, args.dataset_shortcode, args.num_examples, fcvr_layers,
                              causal_model, model.device, split=args.split, inner_format=other, seed=args.seed)
        fc = {"formats": [args.inner_format, other]}
        for sig in signals:
            fc[sig] = auc_block(id_sig[sig], alt[sig], args.n_boot)
        results["format_control"] = fc

    # ---- report ----
    print("\n" + "=" * 78)
    id_m = results["id_means"]
    print("ID means: " + "  ".join(f"{k}={v:.4f}" for k, v in id_m.items()))
    primary = "ilv_last" if "ilv_last" in signals else "entropy_last"
    verdict_lines = []
    for code in ood_codes:
        r = results["ood"][code]
        line = f"OoD {code:12s}"
        for sig in signals:
            b = r[sig]
            line += f"  {sig}={b['auroc']:.3f}[{b['auroc_ci'][0]:.2f},{b['auroc_ci'][1]:.2f}]"
        print(line)
        if primary != "prompt_len":
            b = r[primary]
            v = verdict_for(b["auroc"], *b["auroc_ci"])
            lm, rs = b["length_matched"], b["length_residual"]
            print(f"   {primary}: {v}; length-matched AUROC={lm['auroc']:.3f} (n/class={lm['n_per_class']}), "
                  f"length-residual AUROC={rs['auroc']:.3f}; prompt_len alone AUROC={r['prompt_len']['auroc']:.3f}")
            verdict_lines.append(f"{code}: {v}")
    if results["format_control"] is not None:
        fc = results["format_control"]
        print("FORMAT CONTROL (same content, two templates): " +
              "  ".join(f"{sig}={fc[sig]['auroc']:.3f}" for sig in signals))
        if primary != "prompt_len" and abs(fc[primary]["auroc"] - 0.5) >= 0.15:
            verdict_lines.append(f"format_control: {primary} separates the two TEMPLATES (AUROC {fc[primary]['auroc']:.3f}) "
                                 f"-- surface-form sensitivity; interpret OoD AUROCs with caution")
    results["verdict"] = verdict_lines
    print("VERDICT: " + " | ".join(verdict_lines))
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.run_suffix or "nosuffix"
    tag_str = f"_{args.tag}" if args.tag else ""
    split_str = f"_{args.split}" if args.split != "test" else ""   # historical test files carry no split token
    out_path = os.path.join(args.output_dir, f"input_ood_{args.dataset_shortcode}{split_str}_{suffix}{tag_str}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
