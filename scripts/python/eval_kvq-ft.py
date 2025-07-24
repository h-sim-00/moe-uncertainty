import argparse
import torch
import pandas as pd
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer
from utils import (load_classification_dataset,
                   get_model_predictions, 
                   calculate_accuracy, calculate_ece_mce, calculate_nll)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned PEFT model.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the base model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def evaluate(model, tokenizer, dataset, dataset_name):
    """Orchestrates model evaluation."""
    print(f"--- Evaluating on {dataset_name} ---")
    preds, probs, labels = get_model_predictions(model, tokenizer, dataset)

    acc = calculate_accuracy(preds, labels)
    nll = calculate_nll(probs, labels)
    ece, mce = calculate_ece_mce(probs, labels)
    
    results = {
        'dataset': dataset_name,
        'adapter': model.active_adapter,
        'ACC': acc.item(),
        'NLL': nll.item(),
        'ECE': ece.item(),
        'MCE': mce.item()
    }
    print(results)
    return results

def main():
    # 1. Prepare the setup: wandb, transformers, parsers
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()
    adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"

    # 2. Load peft model with adapter and tokenizer
    print(f"Loading base model '{args.model_shortcode}' and attaching adapter from '{adapter_path}'")
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=adapter_path,
        device_map="auto",
        eval_mode=True
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 3. Load the test datasets and evaluate
    print("\n--- Starting Post-Finetuning Evaluation ---")
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    for shortcode in dataset_shortcodes:
        print(f"\nLoading test dataset: {shortcode}")
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate(model, tokenizer, test_raw_dataset, dataset_name=shortcode)
        all_results.append(results)

    # 4. Save all results to a CSV file
    output_csv_path = f"./results/eval_kvq-ft_results_{args.model_shortcode}_seed-{args.seed}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved all evaluation results to {output_csv_path}")

if __name__ == "__main__":
    main()