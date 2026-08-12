"""
FCVR (VGLR-FC) evaluation -- faithful reconstruction.

Loads the fine-tuned MAP routers into ALL 32 layers, then swaps the selected
layers to FCVR and loads their trained variational weights. Because the MAP
routers are loaded first, each FCVR layer's frozen `mean_base` is seeded from
the fine-tuned MAP router (matching how fcvr-tuning.py trained), and every
non-FCVR layer stays as fine-tuned MAP.

OoD signals extracted:
  - answer_entropy : Shannon entropy of the predictive softmax over {A,B,C,D}
                     at the final token.
  - inf_log_var    : tr(posterior cov) = ||L||_F^2 of the FCVR router's Cholesky
                     factor at the final token, averaged over the FCVR layers.
                     (The FCVR-specific "Inf-Logit-Var" signal.)
  - decomposition  : diagonal mass (sum_i L_ii^2), off-diagonal mass, and
                     log-determinant of the posterior covariance -- which
                     component carries the ID-vs-OoD ordering?
  - per-layer      : inf_log_var AUROC per FCVR layer (not just the mean).
  - familiarity    : ID-train examples scored alongside ID-test; a monotone
                     train > test > OoD ILV gradient means the variance head
                     learned input familiarity (the inversion mechanism).

token-debug additions: --untrained_control (random-init heads), --prior_std /
--input_layernorm / --separate_trunks (must mirror training), per-example JSONL
dump, and wandb logging of all final metrics.

This file is dedicated to FCVR and leaves the generic evaluate.py untouched.
"""

import argparse
import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import setup_environment, load_exp_dataset, multiple_choice_prompt_engineer
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll
from model import load_peft_model_and_adapter, load_tokenizer
from model.adapters import granite_adapter

# Paper Table 8 OoD targets (OBQA is the fixed ID anchor). Edit if needed.
OOD_DATASETS = {
    "arc_c": "near",
    "arc_e": "near",
    "medmcqa_med": "far",
    "mmlu_law": "far",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful FCVR evaluation (answer_entropy + inf_log_var).")
    parser.add_argument("--task", type=str, required=True, choices=["id_calibration", "ood_detection"])
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, required=True,
                        help="The ID dataset the model was trained on (e.g. 'obqa').")
    parser.add_argument("--kvq_adapter_path", type=str, required=True,
                        help="Path to the Stage-1 fine-tuned KVQ LoRA adapter.")
    parser.add_argument("--swap_layers", type=int, nargs="+", required=True,
                        help="Layers that carry a trained FCVR router.")
    parser.add_argument("--run_suffix", type=str, default=None,
                        help="Suffix on the FCVR weights dir; must match the training run's --run_suffix.")
    parser.add_argument("--prior_source", type=str, default="map", choices=["map", "pretrained"],
                        help="Must match the training run: FCVR mean_base from fine-tuned MAP ('map') or pre-trained Granite ('pretrained').")
    parser.add_argument("--output_json_path", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=35, help="MC samples for FCVR inference.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # --- token-debug ablation knobs (must mirror the training run's flags) ---
    parser.add_argument("--prior_std", type=float, default=1.0,
                        help="Must match training. Affects only the KL, but kept for config faithfulness.")
    parser.add_argument("--input_layernorm", action="store_true",
                        help="Must match training: parameter-free LayerNorm on the FCVR trunk input.")
    parser.add_argument("--separate_trunks", action="store_true",
                        help="Must match training: separate trunk for the Cholesky head.")
    parser.add_argument("--untrained_control", action="store_true",
                        help="Skip loading FCVR weights; evaluate the seed-governed random init.")
    parser.add_argument("--per_example_jsonl_path", type=str, default=None,
                        help="ood_detection only: dump one JSON line per example (signals + components).")
    parser.add_argument("--wandb_project", type=str, default="moe-uncertainty-debug")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    return parser.parse_args()


def prepare_model_fcvr(model, args):
    """Reconstruct the trained model. Non-FCVR layers stay as the chosen prior
    routers; FCVR layers get their trained variational weights (mean_base seeded
    from the same prior). Must mirror how the run was trained."""
    print(f"--- Preparing FCVR model (prior_source={args.prior_source}) ---")
    # 1. Set the deterministic routers on all 32 layers. For prior_source=map,
    #    load the fine-tuned MAP routers (also seeds FCVR mean_base); for
    #    prior_source=pretrained, leave the pre-trained Granite routers in place.
    if args.prior_source == "map":
        model = granite_adapter.load_granite_map_routers(model, args=args)
    else:
        print("--- prior_source=pretrained: keeping pre-trained Granite routers; FCVR mean_base seeds from them ---")
    # 2. Swap the selected layers to FCVR and load their trained weights.
    #    load_granite_bayesian_routers reads swap_layers + run_suffix and loads
    #    ./router_weights/fcvr/fcvr-<model>-<dataset>-<suffix>/layer_<i>_weights.pt
    #    With --untrained_control the heads stay at their seed-governed random
    #    init (the exact state training starts from); missing weight files
    #    otherwise raise instead of silently falling back to random init.
    model = granite_adapter.load_granite_bayesian_routers(
        model, method="fcvr", args=args, load_weights=not args.untrained_control
    )

    # 3. Set the number of MC samples used at inference on each FCVR router.
    causal_model = model.base_model.model.model
    for i in args.swap_layers:
        router = causal_model.layers[i].block_sparse_moe.router
        if hasattr(router, "num_mc_samples_inference"):
            router.num_mc_samples_inference = args.num_samples
        if args.untrained_control:
            print(f"[untrained-init] layer {i}: |cholesky_head.W|={router.cholesky_head.weight.norm():.6e} "
                  f"|mean_head.W|={router.mean_head.weight.norm():.6e}")

    model.eval()
    print(f"FCVR layers: {sorted(args.swap_layers)} | MC samples: {args.num_samples}")
    return model


def compute_signals(model, tokenizer, dataset, fcvr_layers, args):
    """Single forward pass per batch. Returns a dict of numpy arrays:
        ans_entropy [N]         entropy over {A,B,C,D} at the final token
        ilv         [Lyr, N]    ||L||_F^2 per FCVR layer (mean over layers = the
                                paper's Inf-Logit-Var signal)
        ilv_diag    [N]         sum_i L_ii^2, meaned over layers (diagonal mass)
        ilv_offdiag [N]         ||L||_F^2 - sum_i L_ii^2, meaned over layers
        log_det     [N]         2*sum_i log L_ii, meaned over layers
    """
    model.eval()
    causal_model = model.base_model.model.model

    choices = ["A", "B", "C", "D"]
    choice_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(c) for c in choices], device=model.device
    )

    processed = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x["question"] for x in processed]

    ans_entropy = []
    ilv_layers = [[] for _ in fcvr_layers]
    ilv_diag, ilv_offdiag, log_det = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc="Signals"):
            batch_q = questions[i:i + args.batch_size]
            inputs = tokenizer(
                batch_q, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            bsz, seq_len = inputs["input_ids"].shape

            logits = model(**inputs).logits

            # --- answer entropy over {A,B,C,D} at the final token ---
            choice_logits = logits[:, -1, :][:, choice_ids]
            probs = F.softmax(choice_logits, dim=1)
            ent = torch.distributions.Categorical(probs=probs).entropy()
            ans_entropy.append(ent.cpu())

            # --- inf-log-var (+ decomposition) at the final token, per FCVR layer ---
            batch_diag, batch_offdiag, batch_logdet = [], [], []
            for j, l in enumerate(fcvr_layers):
                router = causal_model.layers[l].block_sparse_moe.router
                L = router.last_cholesky_factor  # [bsz*seq, E, E], (batch, seq) row-major
                E = L.shape[-1]
                L_last = L.view(bsz, seq_len, E, E)[:, -1, :, :]  # final token per sequence
                trace = (L_last ** 2).sum(dim=(-1, -2))           # trace(LL^T) = ||L||_F^2
                diag = torch.diagonal(L_last, dim1=-2, dim2=-1)   # [bsz, E]
                diag_sq = (diag ** 2).sum(-1)
                ilv_layers[j].append(trace.cpu())
                batch_diag.append(diag_sq)
                batch_offdiag.append(trace - diag_sq)
                batch_logdet.append(2 * torch.log(diag).sum(-1))
            ilv_diag.append(torch.stack(batch_diag).mean(dim=0).cpu())
            ilv_offdiag.append(torch.stack(batch_offdiag).mean(dim=0).cpu())
            log_det.append(torch.stack(batch_logdet).mean(dim=0).cpu())

    return {
        "ans_entropy": torch.cat(ans_entropy).numpy(),
        "ilv": np.stack([torch.cat(x).numpy() for x in ilv_layers]),
        "ilv_diag": torch.cat(ilv_diag).numpy(),
        "ilv_offdiag": torch.cat(ilv_offdiag).numpy(),
        "log_det": torch.cat(log_det).numpy(),
    }


def run_id_calibration(model, tokenizer, args):
    print("\n--- Task: ID Calibration ---")
    test_dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    _, probs, labels = get_model_predictions(model, tokenizer, test_dataset, batch_size=args.batch_size)

    preds = torch.argmax(probs, dim=1)
    acc = calculate_accuracy(preds, labels).item()
    nll = calculate_nll(probs, labels).item()
    ece, mce = calculate_ece_mce(probs, labels)
    result = {"ACC": acc, "NLL": nll, "ECE": ece.item(), "MCE": mce.item()}
    print(f"{args.dataset_shortcode}: {result}")
    return {args.dataset_shortcode: result}


def _auc(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def _per_example_rows(dataset_name, role, shift_type, sig, fcvr_layers):
    """One JSON-serializable row per example from a compute_signals dict."""
    rows = []
    n = sig["ans_entropy"].shape[0]
    ilv_mean = sig["ilv"].mean(axis=0)
    for i in range(n):
        rows.append({
            "dataset": dataset_name,
            "role": role,  # id_train | id_test | ood
            "shift_type": shift_type,
            "index": i,
            "ans_entropy": float(sig["ans_entropy"][i]),
            "ilv_mean": float(ilv_mean[i]),
            "ilv_diag": float(sig["ilv_diag"][i]),
            "ilv_offdiag": float(sig["ilv_offdiag"][i]),
            "log_det": float(sig["log_det"][i]),
            "ilv_per_layer": {str(l): float(sig["ilv"][j, i]) for j, l in enumerate(fcvr_layers)},
        })
    return rows


def run_ood_detection(model, tokenizer, args):
    print("\n--- Task: OOD Detection (answer_entropy + inf_log_var + decomposition) ---")
    fcvr_layers = sorted(args.swap_layers)

    print(f"ID anchor: {args.dataset_shortcode} (test)")
    id_dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    id_sig = compute_signals(model, tokenizer, id_dataset, fcvr_layers, args)
    id_ent = id_sig["ans_entropy"]
    id_ilv = id_sig["ilv"].mean(axis=0)
    print(f"  ID mean  answer_entropy={id_ent.mean():.4f}  inf_log_var={id_ilv.mean():.4f}  "
          f"(diag {id_sig['ilv_diag'].mean():.4f} / offdiag {id_sig['ilv_offdiag'].mean():.4f})")

    # --- Familiarity gradient: score ID TRAIN examples too. If the variance
    # head learned "familiarity", ILV should be monotone train > test > OoD. ---
    print(f"Familiarity gradient: {args.dataset_shortcode} (train, first 500)")
    train_dataset = load_exp_dataset(args.dataset_shortcode, split="train")[:500]
    train_sig = compute_signals(model, tokenizer, train_dataset, fcvr_layers, args)
    train_ilv = train_sig["ilv"].mean(axis=0)
    # AUROC of ILV separating train (positive) from test: >0.5 means the model
    # assigns HIGHER variance to the examples it was trained on.
    fam_auroc, _ = _auc(id_ilv, train_ilv)
    print(f"  train mean inf_log_var={train_ilv.mean():.4f} vs test {id_ilv.mean():.4f} "
          f"| train-vs-test AUROC={fam_auroc:.4f} ({'train>test' if fam_auroc > 0.5 else 'test>train'})")

    dump_rows = _per_example_rows(args.dataset_shortcode, "id_test", "id", id_sig, fcvr_layers)
    dump_rows += _per_example_rows(args.dataset_shortcode, "id_train", "id", train_sig, fcvr_layers)

    results = {
        "familiarity_gradient": {
            "id_train_mean_ilv": float(train_ilv.mean()),
            "id_test_mean_ilv": float(id_ilv.mean()),
            "train_vs_test_auroc_ilv": float(fam_auroc),
        },
    }
    for ood_code, shift_type in OOD_DATASETS.items():
        print(f"OOD vs {ood_code} ({shift_type})")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_sig = compute_signals(model, tokenizer, ood_dataset, fcvr_layers, args)
        ood_ent = ood_sig["ans_entropy"]
        ood_ilv = ood_sig["ilv"].mean(axis=0)
        dump_rows += _per_example_rows(ood_code, "ood", shift_type, ood_sig, fcvr_layers)

        ae_auroc, ae_auprc = _auc(id_ent, ood_ent)
        iv_auroc, iv_auprc = _auc(id_ilv, ood_ilv)
        # Decomposition: which component of the covariance carries the ordering?
        diag_auroc, diag_auprc = _auc(id_sig["ilv_diag"], ood_sig["ilv_diag"])
        off_auroc, off_auprc = _auc(id_sig["ilv_offdiag"], ood_sig["ilv_offdiag"])
        ld_auroc, ld_auprc = _auc(id_sig["log_det"], ood_sig["log_det"])
        results[ood_code] = {
            "shift_type": shift_type,
            "answer_entropy": {"auroc": ae_auroc, "auprc": ae_auprc},
            "inf_log_var": {"auroc": iv_auroc, "auprc": iv_auprc,
                            "id_mean": float(id_ilv.mean()), "ood_mean": float(ood_ilv.mean())},
            "ilv_diag": {"auroc": diag_auroc, "auprc": diag_auprc},
            "ilv_offdiag": {"auroc": off_auroc, "auprc": off_auprc},
            "log_det": {"auroc": ld_auroc, "auprc": ld_auprc},
            "inf_log_var_per_layer": {
                str(l): {"auroc": float(_auc(id_sig["ilv"][j], ood_sig["ilv"][j])[0])}
                for j, l in enumerate(fcvr_layers)
            },
            "familiarity": {"ood_mean_ilv": float(ood_ilv.mean())},
        }
        print(f"  answer_entropy AUROC={ae_auroc:.4f} AUPRC={ae_auprc:.4f} "
              f"(OoD mean {ood_ent.mean():.4f}, {'OoD>ID' if ood_ent.mean() > id_ent.mean() else 'OoD<ID INVERTED'})")
        print(f"  inf_log_var    AUROC={iv_auroc:.4f} AUPRC={iv_auprc:.4f} "
              f"(OoD mean {ood_ilv.mean():.4f}, {'OoD>ID' if ood_ilv.mean() > id_ilv.mean() else 'OoD<ID INVERTED'})")
        print(f"    components: diag AUROC={diag_auroc:.4f} | offdiag AUROC={off_auroc:.4f} | log_det AUROC={ld_auroc:.4f}")
        per_layer_str = " ".join(
            f"L{l}={results[ood_code]['inf_log_var_per_layer'][str(l)]['auroc']:.3f}" for l in fcvr_layers
        )
        print(f"    per-layer ILV AUROC: {per_layer_str}")

    if args.per_example_jsonl_path:
        os.makedirs(os.path.dirname(args.per_example_jsonl_path) or ".", exist_ok=True)
        with open(args.per_example_jsonl_path, "w") as f:
            for row in dump_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Per-example dump: {len(dump_rows)} rows -> {args.per_example_jsonl_path}")

    return results


def _flatten_metrics(d, prefix=""):
    """Flatten a nested dict of metrics into {'a/b/c': value} for wandb.log,
    keeping only numeric leaves."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = v
    return flat


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"eval-{args.task}-fcvr-{args.model_shortcode}-{args.dataset_shortcode}"
        if args.run_suffix:
            run_name += f"-{args.run_suffix}"
        if args.untrained_control:
            run_name += "-UNTRAINED"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
               tags=args.wandb_tags, reinit=True)

    model = load_peft_model_and_adapter(
        args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    model = prepare_model_fcvr(model, args)

    if args.task == "id_calibration":
        final_results = run_id_calibration(model, tokenizer, args)
    else:
        final_results = run_ood_detection(model, tokenizer, args)

    final_results["_meta"] = {
        "task": args.task,
        "run_suffix": args.run_suffix,
        "prior_source": args.prior_source,
        "prior_std": args.prior_std,
        "input_layernorm": args.input_layernorm,
        "separate_trunks": args.separate_trunks,
        "untrained_control": args.untrained_control,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "swap_layers": sorted(args.swap_layers),
    }

    os.makedirs(os.path.dirname(args.output_json_path) or ".", exist_ok=True)
    with open(args.output_json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nSaved results to {args.output_json_path}")

    wandb.log(_flatten_metrics({k: v for k, v in final_results.items() if k != "_meta"}))
    wandb.finish()


if __name__ == "__main__":
    main()
