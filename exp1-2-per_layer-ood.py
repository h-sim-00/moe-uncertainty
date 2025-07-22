import os
import argparse
import torch
import wandb
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from torchmetrics.classification import MulticlassCalibrationError
from huggingface_hub import login as hf_login
from typing import Dict, Any, List, Tuple

# Local imports - ensure these files are in the same directory or accessible in the python path
from data import multiple_choice_prompt_engineer, load_classification_dataset
from model import load_model

def setup_arg_parser() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run MoE model evaluation with specified parameters.")
    parser.add_argument("--param_num", type=str, default="3b", choices=["1b", "3b"], help="Model parameter size (e.g., '1b', '3b').")
    parser.add_argument("--mode", type=str, default="sample_k", help="Routing mode for the MoE layer.")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for sampling mode.")
    parser.add_argument("--dataset_name", type=str, default="medmcqa", help="Name of the dataset to use.")
    return parser.parse_args()

def setup_environment():
    """Configures HuggingFace cache and logs into services."""
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if not IN_COLAB:
        HF_HOME = "/vol/bitbucket/al1624/.cache/huggingface"
        HF_DATASETS_CACHE = "/vol/bitbucket/al1624/.cache/huggingface/datasets"
        os.environ['HF_HOME'] = HF_HOME
        os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
    
    WANDB_KEY = "8d44174f1416d56dc5470b57deb50339b19f22e7"
    HF_TOKEN = "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"
    wandb.login(key=WANDB_KEY)
    hf_login(token=HF_TOKEN)

def load_dependencies(param_num: str, dataset_name: str) -> Tuple[AutoTokenizer, Any, List[Dict[str, Any]]]:
    """Loads the tokenizer, model, and dataset."""
    model_id = "ibm-granite/granite-3.1-1b-a400m-instruct" if param_num == "1b" else "ibm-granite/granite-3.1-3b-a800m-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id)
    
    _, _, test_dataset_raw = load_classification_dataset(dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_dataset_raw]
    
    print(f"Loaded model: {model_id}")
    print(f"Loaded dataset: {dataset_name} with {len(test_dataset)} test samples.")
    
    return tokenizer, model, test_dataset

def evaluate_layer(model: Any, tokenizer: AutoTokenizer, questions: List[str], true_answers: List[str], batch_size: int, layer_idx: int, mode: str, temp: float) -> Tuple[List[str], List[List[float]], List[float], List[float], List[List[float]]]:
    """Runs the inference loop for a single layer and returns raw results."""
    preds, all_logits, all_nlls, all_entropies, all_abcd_probs = [], [], [], [], []
    abcd_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in ['A', 'B', 'C', 'D']]

    model.change_router_mode("top_k")  # Set default mode
    model.change_router_mode_for_layers(mode, [layer_idx], temp=temp)  # Override for the target layer
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating Layer {layer_idx}"):
            batch_q = questions[i:i+batch_size]
            batch_true_answers = true_answers[i:i+batch_size]
            
            inputs = tokenizer(batch_q, return_tensors="pt", padding=True, truncation=True).to(model.device)
            logits = model(**inputs).logits
            assert not torch.isnan(logits).any(), "NaN detected in logits tensor"
            next_token_logits = logits[:, -1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            log_probs_for_entropy = torch.log_softmax(next_token_logits, dim=-1)
            batch_entropies = -torch.sum(probs * log_probs_for_entropy, dim=-1)
            all_entropies.extend(batch_entropies.cpu().tolist())

            for j, true_ans in enumerate(batch_true_answers):
                idx = tokenizer.convert_tokens_to_ids(true_ans)
                log_probs = torch.log_softmax(next_token_logits[j], dim=0)
                nll = -log_probs[idx].item()
                all_nlls.append(nll)

            pred_indices = torch.argmax(next_token_logits, dim=1).cpu().tolist()
            pred_letters = [tokenizer.decode(idx) for idx in pred_indices]
            preds.extend(pred_letters)
            all_logits.extend(next_token_logits.cpu().tolist())

            abcd_probs_batch = probs[:, abcd_token_ids]
            all_abcd_probs.extend(abcd_probs_batch.cpu().tolist())
            
    return preds, all_logits, all_nlls, all_entropies, all_abcd_probs

def calculate_and_log_metrics(run: wandb.wandb_sdk.wandb_run.Run, tokenizer: AutoTokenizer, preds: List[str], true_answers: List[str], all_logits: List[List[float]], all_nlls: List[float], all_entropies: List[float], vocab_size: int):
    """Calculates and logs all metrics to wandb."""
    # Accuracy and NLL
    accuracy = sum([p == t for p, t in zip(preds, true_answers)]) / len(true_answers)
    nll_avg = sum(all_nlls) / len(all_nlls) if all_nlls else 0
    nll_std = torch.tensor(all_nlls).std(unbiased=False).item() if all_nlls else 0
    print(f"Accuracy: {accuracy:.4f} | Avg NLL: {nll_avg:.4f}")
    run.log({"acc": accuracy, "nll_avg": nll_avg, "nll_std": nll_std})

    # ECE
    logits_tensor = torch.tensor(all_logits)
    targets_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in true_answers])
    for n_bins in [10, 15, 20]:
        ece_metric = MulticlassCalibrationError(num_classes=vocab_size, n_bins=n_bins, norm='l1')
        ece = ece_metric(logits_tensor, targets_tensor)
        print(f"ECE (n_bins={n_bins}): {ece:.4f}")
        run.log({f"ece-{n_bins}": ece.item()})

    # Entropy
    ent_avg = sum(all_entropies) / len(all_entropies) if all_entropies else 0
    ent_std = torch.tensor(all_entropies).std(unbiased=False).item() if all_entropies else 0
    print(f"Average Entropy: {ent_avg:.4f}")
    run.log({"ent_avg": ent_avg, "ent_std": ent_std})

def save_results_to_json(exp_name: str, ids: List[str], true_answers: List[str], preds: List[str], all_abcd_probs: List[List[float]], all_entropies: List[float]):
    """Saves the per-datapoint results to a JSON file."""
    output_data = [
        {"id": id_val, "label": t, "pred": p, "abcd_probs": abcd_p, "entropy": e}
        for id_val, t, p, abcd_p, e in zip(ids, true_answers, preds, all_abcd_probs, all_entropies)
    ]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_filename = f"{results_dir}/{exp_name}.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Output data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving JSON output: {e}")

def main():
    """Main function to orchestrate the evaluation."""
    args = setup_arg_parser()
    setup_environment()

    tokenizer, model, test_dataset = load_dependencies(args.param_num, args.dataset_name)
    
    assert args.mode == "sample_k", "This script is configured for 'sample_k' mode only."

    layer_num = len(model.model.layers)
    questions = [x['question'] for x in test_dataset]
    true_answers = [x['answer'] for x in test_dataset]
    ids = [x.get('id', i) for i, x in enumerate(test_dataset)]
    
    project_name = "exp1_evaluation"
    batch_size = 16

    for layer_idx in range(layer_num):
        exp_name = f"{args.param_num}_layer_{layer_idx}_{args.mode}_temp_{args.temp}"
        
        run = wandb.init(
            project=project_name,
            name=exp_name,
            config={
                "model_id": model.config._name_or_path,
                "param_num": args.param_num,
                "dataset_name": args.dataset_name,
                "exp_type": "per-layer",
                "router_mode": args.mode,
                "layer_idx": layer_idx,
                "temp": args.temp
            }
        )
        
        preds, all_logits, all_nlls, all_entropies, all_abcd_probs = evaluate_layer(
            model, tokenizer, questions, true_answers, batch_size, layer_idx, args.mode, args.temp
        )
        
        calculate_and_log_metrics(
            run, tokenizer, preds, true_answers, all_logits, all_nlls, all_entropies, model.config.vocab_size
        )
        
        save_results_to_json(
            exp_name, ids, true_answers, preds, all_abcd_probs, all_entropies
        )

        run.finish()


if __name__ == "__main__":
    main()
