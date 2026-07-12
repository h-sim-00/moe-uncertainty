import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# --- Assumed Utility and Model Imports ---
from utils import setup_environment, load_exp_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll
from utils import multiple_choice_prompt_engineer

# --- Import All Router Classes ---
from model.routers.base import MoERouter
from model.routers.mcdr import MCDropoutRouter
from model.routers.mfvr import MeanFieldVariationalRouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from model.routers.vtsr import VariationalTemperatureRouter

# =============================================================================
# This script MIRRORS Albus' original unified evaluate script
# (scripts/python/evaluate.py) as closely as possible:
#   - the same non-faithful `prepare_model` "Lego swap" (FCVR mean_base is seeded from
#     the untuned Granite router, NOT the Stage-2a MAP router; weights come from
#     --router_weights_path);
#   - the same ID calibration report (ACC / NLL / ECE / MCE);
#   - the same OOD framing: predictive ANSWER entropy via Categorical(probs).entropy(),
#     OOD set {arc_c, medmcqa_med, mmlu_law, sciq} (sciq included, no arc_e, no gate-ent).
#
# The ONLY addition over the original is a second OOD signal: FCVR's Inf-Logit-Var
# = trace of the posterior covariance = ||L||_F^2 (sum of squares of the predicted
# Cholesky factor), averaged over the FCVR layers. Everything else is his logic.
# =============================================================================

def parse_args():
    """Parses command-line arguments for the unified evaluation script."""
    parser = argparse.ArgumentParser(description="Unified evaluation script (Albus' original logic + Inf-Logit-Var).")

    # --- Core Arguments ---
    parser.add_argument("--method", type=str, required=True,
                        choices=['zero_shot', 'kvq_ft', 'det', 'temp_sampling', 'mcdr',
                                 'mfvr', 'fcvr', 'vtsr'])
    parser.add_argument("--task", type=str, required=True, choices=['id_calibration', 'ood_detection'])
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="The primary ID dataset the model was trained on (e.g., 'obqa').")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the final results JSON file.")

    # --- Path Arguments ---
    parser.add_argument("--kvq_adapter_path", type=str, help="Path to the Stage 1 fine-tuned KVQ LoRA adapter.")
    parser.add_argument("--router_weights_path", type=str, help="Path to the directory or file of saved Bayesian router weights/objects.")

    # --- Method-Specific Hyperparameters ---
    parser.add_argument("--swap_layers", type=int, nargs='*', default=None, help="Specific layers to swap with a Bayesian router. Defaults to all.")
    parser.add_argument("--num_samples", type=int, default=35, help="Number of MC samples for stochastic methods.")
    parser.add_argument("--dropout_rate", type=float, default=0.05, help="Dropout rate for re-instantiating MCDR.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for temperature sampling.")
    parser.add_argument("--vtsr_mode", type=str, choices=['per_expert', 'shared'], help="Mode for VTSR.")

    # --- General ---
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def prepare_model(model, args):
    """
    Dispatcher function to perform the correct 'Lego swap' and load weights
    based on the specified method and swap_layers. (Albus' original — unchanged.)
    """
    print(f"--- Preparing model for method: {args.method} ---")
    causal_model = model.base_model.model.model
    device = model.device
    num_layers = len(causal_model.layers)
    swap_layers = args.swap_layers if args.swap_layers is not None else range(num_layers)
    print(f"Targeting layers for swap: {list(swap_layers)}")

    # Define a mapping from method string to router class
    router_class_map = {
        'det': MoERouter,
        'temp_sampling': MoERouter,
        'mcdr': MCDropoutRouter,
        'mfvr': MeanFieldVariationalRouter,
        'fcvr': FullCovarianceVariationalRouter,
        'vtsr': VariationalTemperatureRouter,
    }

    if args.method not in router_class_map:
        return model  # No swap needed for zero_shot or kvq_ft

    # Perform the swap for the specified layers
    for i in swap_layers:
        layer = causal_model.layers[i]
        RouterClass = router_class_map[args.method]

        init_kwargs = {'config': causal_model.config, 'existing_router': layer.block_sparse_moe.router}
        if args.method == 'temp_sampling':
            init_kwargs['mode'] = 'sample_k'
            init_kwargs['temp'] = args.temperature
        elif args.method == 'mcdr':
            init_kwargs['dropout_rate'] = args.dropout_rate
        elif args.method == 'vtsr':
            init_kwargs['temperature_mode'] = args.vtsr_mode

        new_router = RouterClass(**init_kwargs).to(device)

        if args.router_weights_path:
            weights_path = os.path.join(args.router_weights_path, f"layer_{i}_weights.pt")
            if os.path.exists(weights_path):
                new_router.load_weights(weights_path, device=device)
            else:
                print(f"Warning: No weights file found for layer {i} at {weights_path}")

        layer.block_sparse_moe.router = new_router

    model.eval()
    print("Model preparation complete.")
    return model

def get_predictions(model, tokenizer, dataset, args):
    """
    Dispatcher for inference. Calls the correct evaluation logic based on the method.
    (Albus' original — unchanged.)
    """
    # --- Configure Routers Before Inference ---
    if args.method in ['mcdr', 'mfvr', 'fcvr']:
        for layer in model.base_model.model.model.layers:
            router = layer.block_sparse_moe.router
            if hasattr(router, 'num_mc_samples_inference'):
                router.num_mc_samples_inference = args.num_samples
            if args.method == 'mcdr' and hasattr(router, 'dropout'):
                router.dropout.train()

    _, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=args.batch_size)

    return probs, labels


def compute_inf_log_var(model, tokenizer, dataset, args):
    """
    ADDITION over Albus' original: FCVR Inf-Logit-Var per example.

    Runs one forward pass over `dataset` and, at the final real token, reads each FCVR
    layer's predicted Cholesky factor L (router.last_cholesky_factor) and forms
    tr(Sigma) = ||L||_F^2 = sum of squares of L (since Sigma = L L^T). Averaged over the
    FCVR layers (args.swap_layers). Returns a (num_examples,) numpy array.

    Prompting mirrors get_model_predictions via multiple_choice_prompt_engineer, so the
    examples line up 1:1 with the answer-entropy scores from get_predictions.
    """
    causal_model = model.base_model.model.model
    num_layers = len(causal_model.layers)

    # Ensure the eval-time MC sample count is set on FCVR routers.
    for layer in causal_model.layers:
        router = getattr(getattr(layer, 'block_sparse_moe', None), 'router', None)
        if router is not None and hasattr(router, 'num_mc_samples_inference'):
            router.num_mc_samples_inference = args.num_samples

    fcvr_layers = list(args.swap_layers) if args.swap_layers is not None else list(range(num_layers))

    processed_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x['question'] for x in processed_dataset]

    per_layer_ilv = {}
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(questions), args.batch_size), desc="Inf-Logit-Var"):
            batch_questions = questions[start:start + args.batch_size]
            inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            outputs = model(**inputs)  # noqa: F841 (populates router.last_cholesky_factor)

            attn = inputs["attention_mask"]  # (B, L)
            B, L = attn.shape
            positions = torch.arange(L, device=attn.device)
            last_pos = (attn * positions).argmax(dim=1)  # (B,) last real token, padding-side robust

            for i in fcvr_layers:
                router = getattr(getattr(causal_model.layers[i], 'block_sparse_moe', None), 'router', None)
                chol = getattr(router, 'last_cholesky_factor', None) if router is not None else None
                if chol is None:
                    continue
                chol = chol.float()  # (B*L, N, N)
                N = chol.shape[-1]
                chol = chol.reshape(B, L, N, N)  # row-major flatten, matches Granite
                gather_idx = last_pos.view(B, 1, 1, 1).expand(B, 1, N, N)
                final_chol = chol.gather(1, gather_idx).squeeze(1)  # (B, N, N)
                ilv = (final_chol ** 2).sum(dim=(-1, -2))  # (B,)  == tr(L L^T)
                per_layer_ilv.setdefault(i, []).append(ilv.cpu())

    if not per_layer_ilv:
        raise ValueError(
            "No Cholesky factors captured — none of the requested swap_layers is an FCVR "
            "layer. Inf-Logit-Var requires --method fcvr and FCVR-occupied --swap_layers."
        )

    per_layer = {i: torch.cat(v).numpy() for i, v in per_layer_ilv.items()}
    return np.mean([per_layer[i] for i in per_layer], axis=0)


def run_id_calibration(model, tokenizer, args):
    """Runs the In-Distribution Calibration task. (Albus' original — unchanged.)"""
    print("\n--- Running Task: ID Calibration ---")
    results = {}
    id_datasets = [args.dataset_shortcode]
    for dataset_code in id_datasets:
        print(f"Evaluating on: {dataset_code}")
        test_dataset = load_exp_dataset(dataset_code, split="test")
        probs, labels = get_predictions(model, tokenizer, test_dataset, args)

        preds = torch.argmax(probs, dim=1)
        acc = calculate_accuracy(preds, labels).item()
        nll = calculate_nll(probs, labels).item()
        ece, mce = calculate_ece_mce(probs, labels)

        results[dataset_code] = {'ACC': acc, 'NLL': nll, 'ECE': ece.item(), 'MCE': mce.item()}
        print(results[dataset_code])
    return results

def run_ood_detection(model, tokenizer, args):
    """Runs the Out-of-Distribution Detection task.

    Mirrors Albus' original: predictive ANSWER entropy via Categorical(probs).entropy()
    over the OOD set {arc_c, medmcqa_med, mmlu_law, sciq}. ADDS a second signal,
    FCVR Inf-Logit-Var, scored the same way (ID vs OOD AUROC/AUPRC).
    """
    print("\n--- Running Task: OOD Detection ---")
    results = {}

    fcvr_layers = list(args.swap_layers) if args.swap_layers is not None else "all"
    print(f"Inf-Logit-Var layer set (FCVR layers): {fcvr_layers}")

    # --- ID (obqa) reference scores for both signals ---
    id_dataset = load_exp_dataset("obqa", split="test")
    id_probs, _ = get_predictions(model, tokenizer, id_dataset, args)
    id_answer_entropy = torch.distributions.Categorical(probs=id_probs).entropy().numpy()
    id_inf_log_var = compute_inf_log_var(model, tokenizer, id_dataset, args)
    id_scores = {'answer_entropy': id_answer_entropy, 'inf_log_var': id_inf_log_var}
    for name in id_scores:
        print(f"[diag] ID mean {name:16s}= {np.mean(id_scores[name]):.4f}")

    ood_datasets = {"arc_c": "small", "medmcqa_med": "large", "mmlu_law": "large", "sciq": "large"}
    for ood_code, shift_type in ood_datasets.items():
        print(f"\nEvaluating OOD against: {ood_code} ({shift_type} shift)")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_probs, _ = get_predictions(model, tokenizer, ood_dataset, args)
        ood_answer_entropy = torch.distributions.Categorical(probs=ood_probs).entropy().numpy()
        ood_inf_log_var = compute_inf_log_var(model, tokenizer, ood_dataset, args)
        ood_scores = {'answer_entropy': ood_answer_entropy, 'inf_log_var': ood_inf_log_var}

        entry = {'shift_type': shift_type}
        for name in id_scores:
            scores = np.concatenate([id_scores[name], ood_scores[name]])
            labels = np.concatenate([np.zeros_like(id_scores[name]), np.ones_like(ood_scores[name])])
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            entry[name] = {'auroc': auroc, 'auprc': auprc}

            id_mean, ood_mean = np.mean(id_scores[name]), np.mean(ood_scores[name])
            direction = "OoD>ID (expected)" if ood_mean > id_mean else "OoD<ID (INVERTED)"
            print(f"[diag]   {name:16s} ID={id_mean:.4f} OoD={ood_mean:.4f} "
                  f"-> {direction}, AUROC={auroc:.4f}")

        results[ood_code] = entry
        print(results[ood_code])
    return results

def main():
    """Main orchestration script."""
    setup_environment()
    args = parse_args()

    if args.method in ['zero_shot']:
        model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=None, device_map="cuda:0")
    else:
        model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0")

    tokenizer = load_tokenizer(args.model_shortcode)

    model = prepare_model(model, args)

    if args.task == 'id_calibration':
        final_results = run_id_calibration(model, tokenizer, args)
    elif args.task == 'ood_detection':
        final_results = run_ood_detection(model, tokenizer, args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nSaved final results to {args.output_json_path}")

if __name__ == "__main__":
    main()
