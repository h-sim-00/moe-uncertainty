"""
Bridge test (new-exp2): does the paper's INPUT-LEVEL OoD claim survive
generation fine-tuning?

The VMoER paper's validated Inf-Logit-Var result (Table 8) is input-level:
one score per input (read at the final predictive token, averaged over the
Bayesian-modified layers) separates ID prompts from OoD prompts. Nothing on
this branch had checked whether the MedExQA-generation-trained FCVR still
shows that separation. If it does not, the per-token generation analysis
(analyze_token_signals.py) has no support from the paper's mechanism -- run
this alongside it and read both verdicts together.

Protocol
--------
ID  = MedExQA test prompts (the FCVR's training distribution).
OoD = other datasets' test prompts (default: obqa near-ish, mmlu_law far).
Every prompt -- ID and OoD alike -- is wrapped in the SAME
generation_prompt_engineer chat template, so the separation cannot be a
prompt-template artefact; only the question content differs. (The MCQA
questions keep their native "Question/Choices/Answer:" inner format.)

Signals per prompt (prompt-only forward, batch size 1, so there is no padding
and the "last token" is genuinely the last token):
    ilv_last     : tr(posterior cov) = ||L||_F^2 at the final prompt token,
                   averaged over the FCVR layers   (paper Table 8 protocol)
    ilv_mean     : same, averaged over ALL prompt positions
    entropy_last : predictive entropy at the final prompt token  (baseline)

Reports AUROC/AUPRC (OoD = 1) per signal per OoD set, plus the ID/OoD means
with an OoD>ID / INVERTED direction note, mirroring evaluate_fcvr.py. FCVR
routers run in deterministic posterior-mean mode (the Cholesky factor, hence
the signal, is computed identically either way).

Prereq: trained FCVR weights (fcvr-tuning-granite-medexqa.sh) + the Stage-1
adapter. Run on quail-1 (moe_env); fcvr-eval-granite-medexqa.sh calls this
after the token-level readouts.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import setup_environment, load_exp_dataset, generation_prompt_engineer
from model import load_peft_model_and_adapter, load_tokenizer
from evaluate_fcvr import prepare_model_fcvr  # same faithful reconstruction as analyze_token_signals


def parse_args():
    p = argparse.ArgumentParser(description="Input-level ID-vs-OoD Inf-Logit-Var bridge test (FCVR).")
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="medexqa",
                   help="ID dataset the FCVR was trained on (drives weight paths AND is the ID anchor).")
    p.add_argument("--kvq_adapter_path", type=str, required=True)
    p.add_argument("--swap_layers", type=int, nargs="+", required=True)
    p.add_argument("--run_suffix", type=str, default=None,
                   help="Must match the training run's --run_suffix.")
    p.add_argument("--prior_source", type=str, default="pretrained", choices=["map", "pretrained"],
                   help="Must match the training run.")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Unused (deterministic readout); kept for prepare_model_fcvr compatibility.")
    p.add_argument("--ood_datasets", type=str, nargs="+", default=["obqa", "mmlu_law"],
                   help="OoD test sets (load_exp_dataset shortcodes).")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"],
                   help="Split used for the ID anchor AND every OoD set. PROTOCOL: 'val' = selection; "
                        "'test' = evaluated once per frozen configuration (pass explicitly).")
    p.add_argument("--num_examples", type=int, default=175, help="Examples per dataset.")
    p.add_argument("--output_dir", type=str, default="results/input_level_ood")
    p.add_argument("--tag", type=str, default=None, help="Extra tag appended to the output filename.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_signals(model, tokenizer, dataset_code, num_examples, fcvr_layers, causal_model, device, split="val"):
    """One prompt-only forward per example (batch 1) -> dict of np arrays."""
    ds = load_exp_dataset(dataset_code, split=split)[:num_examples]
    out = {"ilv_last": [], "ilv_mean": [], "entropy_last": []}
    for ex in tqdm(ds, desc=dataset_code):
        prompt = generation_prompt_engineer(ex, tokenizer=tokenizer)["question"]
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048
        ).input_ids.to(device)
        seq_len = ids.shape[1]

        logits = model(input_ids=ids).logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        out["entropy_last"].append(float(-(logp.exp() * logp).sum().item()))

        per_layer_last, per_layer_mean = [], []
        for l in fcvr_layers:
            L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
            E = L.shape[-1]
            tr = (L.view(seq_len, E, E).float() ** 2).sum(dim=(-1, -2))  # [seq] = tr(LL^T)
            per_layer_last.append(tr[-1])
            per_layer_mean.append(tr.mean())
        out["ilv_last"].append(float(torch.stack(per_layer_last).mean().item()))
        out["ilv_mean"].append(float(torch.stack(per_layer_mean).mean().item()))
    return {k: np.asarray(v) for k, v in out.items()}


def _auc(id_scores, ood_scores):
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


# Datasets sharing the medical-exam domain: none of these may serve as an OoD
# set when the ID anchor (the training dataset) is one of them.
MEDICAL_DATASETS = {"medexqa", "medmcqa_med"}


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    invalid = [c for c in args.ood_datasets
               if c == args.dataset_shortcode
               or (args.dataset_shortcode in MEDICAL_DATASETS and c in MEDICAL_DATASETS)]
    if invalid:
        print(f"SKIP {invalid}: same domain as ID anchor '{args.dataset_shortcode}' -- not valid OoD sets")
        args.ood_datasets = [c for c in args.ood_datasets if c not in invalid]
    if not args.ood_datasets:
        raise SystemExit("No valid OoD datasets left after same-domain filtering.")

    model = load_peft_model_and_adapter(
        args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)
    model = prepare_model_fcvr(model, args)

    fcvr_layers = sorted(args.swap_layers)
    causal_model = model.base_model.model.model
    for l in fcvr_layers:
        causal_model.layers[l].block_sparse_moe.router.deterministic_readout = True
    print("Routing mode: DETERMINISTIC posterior-mean (signal unaffected)")

    if args.split == "test":
        print("#" * 72 + "\n# TEST SPLIT: evaluate ONCE per frozen configuration (selection happens on val).\n" + "#" * 72)
    print(f"\nID anchor: {args.dataset_shortcode} ({args.num_examples} {args.split} prompts)")
    id_sig = collect_signals(model, tokenizer, args.dataset_shortcode, args.num_examples,
                             fcvr_layers, causal_model, model.device, split=args.split)

    results = {
        "config": {
            "id_dataset": args.dataset_shortcode, "ood_datasets": args.ood_datasets,
            "num_examples": args.num_examples, "split": args.split, "fcvr_layers": fcvr_layers,
            "run_suffix": args.run_suffix, "prior_source": args.prior_source,
            "template": "generation_prompt_engineer (uniform across ID and OoD)",
        },
        "id_means": {k: float(v.mean()) for k, v in id_sig.items()},
        "ood": {},
    }

    for code in args.ood_datasets:
        print(f"\nOoD set: {code}")
        ood_sig = collect_signals(model, tokenizer, code, args.num_examples,
                                  fcvr_layers, causal_model, model.device, split=args.split)
        results["ood"][code] = {}
        for sig in ("ilv_last", "ilv_mean", "entropy_last"):
            auroc, auprc = _auc(id_sig[sig], ood_sig[sig])
            results["ood"][code][sig] = {
                "auroc": auroc, "auprc": auprc, "ood_mean": float(ood_sig[sig].mean()),
            }

    # ---- verdict ----
    print("\n" + "=" * 70)
    id_m = results["id_means"]
    print(f"ID ({args.dataset_shortcode}):  ilv_last={id_m['ilv_last']:.4f}  "
          f"ilv_mean={id_m['ilv_mean']:.4f}  entropy_last={id_m['entropy_last']:.4f}")
    best_dev = 0.0
    for code in args.ood_datasets:
        r = results["ood"][code]
        direction = "OoD>ID" if r["ilv_last"]["ood_mean"] > id_m["ilv_last"] else "OoD<ID (INVERTED)"
        print(f"OoD {code:12s} ilv_last AUROC={r['ilv_last']['auroc']:.3f} ({direction})  "
              f"ilv_mean AUROC={r['ilv_mean']['auroc']:.3f}  "
              f"entropy_last AUROC={r['entropy_last']['auroc']:.3f}")
        best_dev = max(best_dev, abs(r["ilv_last"]["auroc"] - 0.5))
    if best_dev >= 0.10:
        tag = ("SEPARATION PRESENT: |AUROC - 0.5| >= 0.10 on at least one OoD set "
               "(direction noted above; AUROC < 0.5 = inverted, matching the OBQA repro). "
               "The paper's input-level mechanism survived generation training, so the "
               "token-level analysis rests on a live signal.")
    else:
        tag = ("SEPARATION GONE: ilv_last AUROC within 0.10 of chance on every OoD set. "
               "The paper's input-level claim did NOT survive generation training -- "
               "treat the token-level analysis as unsupported by the paper's mechanism.")
    results["verdict"] = tag
    print("VERDICT: " + tag)
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.run_suffix or "nosuffix"
    tag_str = f"_{args.tag}" if args.tag else ""
    # Historical (test-split) files carry no split token; val outputs are marked.
    split_str = f"_{args.split}" if args.split != "test" else ""
    out_path = os.path.join(args.output_dir, f"input_ood_{args.dataset_shortcode}{split_str}_{suffix}{tag_str}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
