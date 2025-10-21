# 3. Load and preprocess the dataset;
# 4. Evaluation first, start finetuning, evaluation again at each epoch, when result is stable, save the model;
# 5. Evalaution: 
#    Only finetuning, without calibrating;
#    Performance: ACC, NLL, ECE, MCE;

from utils import setup_environment
import argparse
from model import load_model, load_tokenizer
from utils import (load_classification_dataset, 
                   get_model_predictions, 
                   calculate_accuracy, calculate_ece_mce, calculate_nll)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="0-shot base-baseline setup.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Name of the model to use.")
    return parser.parse_args()


def evaluate(model, tokenizer, dataset, dataset_name):
    """Orchestrates model evaluation."""
    print(f"--- Evaluating on {dataset_name} ---")
    preds, probs, labels = get_model_predictions(model, tokenizer, dataset)

    # Calculate metrics
    acc = calculate_accuracy(preds, labels)
    nll = calculate_nll(probs, labels)
    ece, mce = calculate_ece_mce(probs, labels)
    
    results = {
        'Dataset': dataset_name,
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


    # 2. Load the model and tokenizer (no need to put on the peft jacket)
    print(f"Loading model: {args.model_shortcode}")
    model = load_model(args.model_shortcode, device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)

    # 3. Load the dataset & zero-shot evaluation
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []

    for shortcode in dataset_shortcodes:
        print(f"\nLoading and processing dataset: {shortcode}")
        test_raw_dataset = load_classification_dataset(shortcode)[2]
        results = evaluate(model, tokenizer, test_raw_dataset, dataset_name=shortcode)
        all_results.append(results)

    # 4. Save all results to a CSV
    df = pd.DataFrame(all_results)
    df.to_csv(f"./results/eval_0-shot_results_{args.model_shortcode}.csv", index=False)
    print(f"\nSaved all evaluation results to zero_shot_results_{args.model_shortcode}.csv")

if __name__ == "__main__":
    main()
    