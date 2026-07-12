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

# Faithful FCVR reconstruction uses the SAME adapter helpers that fcvr-tuning.py used,
# so the eval-time model is bit-for-bit the train-time stack (see prepare_model_fcvr).
from model.adapters.granite_adapter import (
    load_granite_map_routers,
    load_granite_bayesian_routers,
)

def parse_args():
    """Parses command-line arguments for the FCVR evaluation script."""
    parser = argparse.ArgumentParser(description="FCVR-dedicated evaluation script for Granite MoE.")

    # --- Core Arguments ---
    parser.add_argument("--method", type=str, default='fcvr',
                        choices=['zero_shot', 'kvq_ft', 'det', 'temp_sampling', 'mcdr',
                                 'mfvr', 'fcvr', 'vtsr'])
    parser.add_argument("--task", type=str, required=True, choices=['id_calibration', 'ood_detection'])
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="The primary ID dataset the model was trained on (e.g., 'obqa').")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the final results JSON file.")

    # --- Path Arguments ---
    parser.add_argument("--kvq_adapter_path", type=str, help="Path to the Stage 1 fine-tuned KVQ LoRA adapter.")
    # NOTE: for FCVR the MAP-router dir (./router_weights/base/<model>_<dataset>) and the
    # FCVR-router dir (./router_weights/fcvr/fcvr-<model>-<dataset>) are DERIVED from the
    # shortcodes inside the adapter helpers, exactly as fcvr-tuning.py did. This flag is
    # kept only for non-fcvr methods and is ignored on the fcvr path.
    parser.add_argument("--router_weights_path", type=str, help="Path to saved router weights (non-fcvr methods only).")

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


def prepare_model_fcvr(model, args):
    """
    Faithfully reconstructs the FCVR training-time stack for evaluation.

    fcvr-tuning.py builds the model as:
      1. KVQ LoRA adapter (loaded upstream in main()),
      2. load_granite_map_routers(): ALL 32 layers get their Stage-2a MAP routers,
      3. prepare/load Bayesian routers: the `swap_layers` are replaced by FCVR whose
         frozen `mean_base` is copied FROM the MAP router in (2), with the trained
         variational nets (mean_residual_net + cholesky_net) loaded on top.

    We mirror (2) and (3) exactly using the same helpers, so layers 0..26 stay MAP and
    layers 27..31 are FCVR seeded on MAP — not the original Granite router. This is the
    difference from the generic evaluate.py, which never loads MAP and seeds mean_base
    from the untuned Granite router.
    """
    print("--- Preparing FCVR model: faithful MAP + FCVR reconstruction ---")

    # (2) Load Stage-2a MAP routers into ALL 32 layers.
    #     Path derived by the helper: ./router_weights/base/<model>_<dataset>
    model = load_granite_map_routers(model, args=args)

    # (3) Swap `swap_layers` to FCVR (mean_base seeded from the MAP routers above) and
    #     load their trained variational nets.
    #     Path derived by the helper: ./router_weights/fcvr/fcvr-<model>-<dataset>
    model = load_granite_bayesian_routers(model, method="fcvr", args=args)

    # Set the eval-time MC sample count on the FCVR layers.
    causal_model = model.base_model.model.model
    fcvr_layers = args.swap_layers if args.swap_layers is not None else range(len(causal_model.layers))
    for i in fcvr_layers:
        router = causal_model.layers[i].block_sparse_moe.router
        if hasattr(router, "num_mc_samples_inference"):
            router.num_mc_samples_inference = args.num_samples

    model.eval()
    print(f"FCVR model ready. FCVR layers: {list(fcvr_layers)} | MC samples (eval): {args.num_samples}")
    return model


def prepare_model(model, args):
    """
    Generic 'Lego swap' dispatcher for non-FCVR methods (kept for completeness; the
    fcvr path uses prepare_model_fcvr instead).
    """
    print(f"--- Preparing model for method: {args.method} ---")
    causal_model = model.base_model.model.model
    device = model.device
    num_layers = len(causal_model.layers)
    swap_layers = args.swap_layers if args.swap_layers is not None else range(num_layers)
    print(f"Targeting layers for swap: {list(swap_layers)}")

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

# Susceptible ("most brittle") MoE layers for Granite-3B identified via the paper's
# brittleness analysis (Appendix B.1): transition regions 5-8, 19-20, 28-31.
SUSCEPTIBLE_LAYERS_GRANITE = [5, 6, 7, 8, 19, 20, 28, 29, 30, 31]


def compute_uncertainty_signals(model, tokenizer, dataset, args):
    """
    Runs a single forward pass over `dataset` and extracts, at the final predictive token:
      - answer_entropy: Shannon entropy of the predictive distribution over the A/B/C/D
        answer choices (output-level uncertainty).
      - per_layer_gate_entropy: dict {layer_idx: (num_examples,)} of Gate-Ent (paper Eq. 29),
        the entropy of each router's expert-gating distribution p = softmax(router_logits)
        over the N experts.
      - per_layer_inf_log_var: dict {layer_idx: (num_examples,)} of the FCVR-specific
        Inf-Logit-Var signal (paper Table 8) = trace of the posterior covariance over
        expert logits = ||L||_F^2 = sum of squares of the predicted Cholesky factor L
        (since Sigma = L L^T => tr(Sigma) = ||L||_F^2). Only populated for FCVR layers,
        which are the only ones carrying a `last_cholesky_factor`.

    All signals come from the same forward pass.
    """
    causal_model = model.base_model.model.model
    num_layers = len(causal_model.layers)

    # Answer-choice token ids (mirrors get_model_predictions).
    choices = ['A', 'B', 'C', 'D']
    choice_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(c) for c in choices], device=model.device
    )

    # Hook every router to capture all layers' expert logits in one pass.
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            captured[layer_idx] = output[-1] if isinstance(output, (tuple, list)) else output
        return hook

    for i in range(num_layers):
        router = getattr(getattr(causal_model.layers[i], 'block_sparse_moe', None), 'router', None)
        if router is not None:
            handles.append(router.register_forward_hook(make_hook(i)))

    processed_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x['question'] for x in processed_dataset]

    answer_entropies = []
    per_layer_gate = {i: [] for i in range(num_layers)}
    per_layer_ilv = {}  # populated lazily, only for FCVR layers
    model.eval()
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(questions), args.batch_size), desc="Uncertainty"):
                batch_questions = questions[start:start + args.batch_size]
                inputs = tokenizer(
                    batch_questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(model.device)

                captured.clear()
                outputs = model(**inputs)

                attn = inputs["attention_mask"]  # (B, L)
                B, L = attn.shape
                # Last real token per row (robust to both left- and right-padding).
                positions = torch.arange(L, device=attn.device)
                last_pos = (attn * positions).argmax(dim=1)  # (B,)

                # --- Answer entropy over A/B/C/D at the final token ---
                batch_idx = torch.arange(B, device=model.device)
                final_lm_logits = outputs.logits[batch_idx, last_pos, :]  # (B, vocab)
                choice_logits = final_lm_logits[:, choice_ids]  # (B, 4)
                ans_p = torch.softmax(choice_logits.float(), dim=-1)
                ans_ent = -(ans_p * torch.log(ans_p + 1e-12)).sum(dim=-1)  # (B,)
                answer_entropies.append(ans_ent.cpu())

                # --- Per-layer Gate-Ent over N experts at the final token ---
                idx_expand = last_pos.view(B, 1, 1)
                for i in captured:
                    router_logits = captured[i].float()  # (B*L, N)
                    N = router_logits.shape[-1]
                    gate_logits = router_logits.reshape(B, L, N)  # row-major, matches Granite flatten
                    final_gate = gate_logits.gather(1, idx_expand.expand(B, 1, N)).squeeze(1)  # (B, N)
                    p = torch.softmax(final_gate, dim=-1)
                    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1)  # (B,)
                    per_layer_gate[i].append(ent.cpu())

                # --- Per-FCVR-layer Inf-Logit-Var = tr(Sigma) = ||L||_F^2 at final token ---
                for i in range(num_layers):
                    router = getattr(getattr(causal_model.layers[i], 'block_sparse_moe', None), 'router', None)
                    chol = getattr(router, 'last_cholesky_factor', None) if router is not None else None
                    if chol is None:
                        continue
                    chol = chol.float()  # (B*L, N, N)
                    N = chol.shape[-1]
                    chol = chol.reshape(B, L, N, N)  # same row-major flatten as the gate path
                    gather_idx = last_pos.view(B, 1, 1, 1).expand(B, 1, N, N)
                    final_chol = chol.gather(1, gather_idx).squeeze(1)  # (B, N, N)
                    ilv = (final_chol ** 2).sum(dim=(-1, -2))  # (B,)  == tr(L L^T)
                    per_layer_ilv.setdefault(i, []).append(ilv.cpu())
    finally:
        for h in handles:
            h.remove()

    answer_entropy = torch.cat(answer_entropies).numpy()
    per_layer_gate_entropy = {
        i: torch.cat(v).numpy() for i, v in per_layer_gate.items() if v
    }
    per_layer_inf_log_var = {
        i: torch.cat(v).numpy() for i, v in per_layer_ilv.items() if v
    }
    return answer_entropy, per_layer_gate_entropy, per_layer_inf_log_var


def _mean_over_layers(per_layer, layers):
    """Averages a per-layer signal dict over the given layer indices -> (num_examples,)."""
    available = [i for i in layers if i in per_layer]
    if not available:
        raise ValueError(f"No captured signal for requested layers {layers}.")
    return np.mean([per_layer[i] for i in available], axis=0)


def _build_signals(ans, gate, ilv, all_layers, susceptible_layers, fcvr_layers):
    """Assembles the named 1-D score arrays from the raw per-layer signals."""
    signals = {
        'answer_entropy': ans,
        'gate_ent_all': _mean_over_layers(gate, all_layers),
        'gate_ent_susceptible': _mean_over_layers(gate, susceptible_layers),
    }
    # Inf-Logit-Var only exists where FCVR routers were placed.
    if ilv and fcvr_layers:
        available = [i for i in fcvr_layers if i in ilv]
        if available:
            signals['inf_log_var'] = _mean_over_layers(ilv, available)
    return signals


def run_id_calibration(model, tokenizer, args):
    """Runs the In-Distribution Calibration task (ACC / NLL / ECE / MCE)."""
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

    For every OOD dataset we score four uncertainty signals, all extracted from the same
    forward pass, and store AUROC/AUPRC for each:
      - answer_entropy:         predictive entropy over the A/B/C/D answer choices.
      - gate_ent_all:           Gate-Ent averaged over all MoE layers.
      - gate_ent_susceptible:   Gate-Ent averaged over the 10 susceptible Granite layers.
      - inf_log_var:            FCVR-specific tr(posterior cov) averaged over the FCVR layers.
    """
    print("\n--- Running Task: OOD Detection ---")
    results = {}

    num_layers = len(model.base_model.model.model.layers)
    all_layers = list(range(num_layers))
    susceptible_layers = [i for i in SUSCEPTIBLE_LAYERS_GRANITE if i < num_layers]
    fcvr_layers = list(args.swap_layers) if args.swap_layers is not None else all_layers
    print(f"Gate-Ent layer sets -> all: {all_layers} | susceptible: {susceptible_layers}")
    print(f"Inf-Logit-Var layer set (FCVR layers): {fcvr_layers}")

    id_dataset = load_exp_dataset("obqa", split="test")
    id_ans, id_gate, id_ilv = compute_uncertainty_signals(model, tokenizer, id_dataset, args)
    id_scores = _build_signals(id_ans, id_gate, id_ilv, all_layers, susceptible_layers, fcvr_layers)

    # --- Diagnostics: mean signals for ID to reveal each signal's direction ---
    print("\n[diag] ID (obqa) mean Gate-Ent per layer:")
    print("       " + "  ".join(f"L{i}:{np.mean(id_gate[i]):.3f}" for i in sorted(id_gate)))
    if id_ilv:
        print("[diag] ID (obqa) mean Inf-Logit-Var per FCVR layer:")
        print("       " + "  ".join(f"L{i}:{np.mean(id_ilv[i]):.3f}" for i in sorted(id_ilv)))
    for name in id_scores:
        print(f"[diag] ID mean {name:22s}= {np.mean(id_scores[name]):.4f}")

    ood_datasets = {"arc_e": "small", "arc_c": "small", "medmcqa_med": "large", "mmlu_law": "large"}
    for ood_code, shift_type in ood_datasets.items():
        print(f"\nEvaluating OOD against: {ood_code} ({shift_type} shift)")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_ans, ood_gate, ood_ilv = compute_uncertainty_signals(model, tokenizer, ood_dataset, args)
        ood_scores = _build_signals(ood_ans, ood_gate, ood_ilv, all_layers, susceptible_layers, fcvr_layers)

        entry = {'shift_type': shift_type}
        for name in id_scores:
            if name not in ood_scores:
                continue
            scores = np.concatenate([id_scores[name], ood_scores[name]])
            labels = np.concatenate([np.zeros_like(id_scores[name]), np.ones_like(ood_scores[name])])
            auroc = roc_auc_score(labels, scores)
            entry[name] = {'auroc': auroc, 'auprc': average_precision_score(labels, scores)}

            # Direction check: OoD should score HIGHER than ID for AUROC > 0.5.
            id_mean, ood_mean = np.mean(id_scores[name]), np.mean(ood_scores[name])
            direction = "OoD>ID (expected)" if ood_mean > id_mean else "OoD<ID (INVERTED)"
            print(f"[diag]   {name:22s} ID={id_mean:.4f} OoD={ood_mean:.4f} "
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

    if args.method == 'fcvr':
        model = prepare_model_fcvr(model, args)
    else:
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
