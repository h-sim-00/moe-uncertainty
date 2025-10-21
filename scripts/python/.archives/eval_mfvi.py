# evaluate_mfvi_router.py

import argparse
import torch
import pandas as pd
from utils import setup_environment, load_classification_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from model.routers.mfvr import load_mfvi_routers_into_model, evaluate_mfvi_router

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned MFVI MoE router.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--base_adapter_path", type=str, required=True, help="Path to the Stage 1 KVQ LoRA adapter.")
    parser.add_argument("--router_weights_path", type=str, required=True, help="Path to the saved MFVI router weights (.pt file).")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--results_csv_path", type=str, default="./results/evaluation_mfvi.csv")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load the trained Bayesian router weights
    router_state_dicts = torch.load(args.router_weights_path, map_location=model.device)
    model = load_mfvi_routers_into_model(model, router_state_dicts)
    model.eval()
    print("Model is ready for evaluation.")

    # 3. Run evaluation across all specified datasets
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    
    for shortcode in dataset_shortcodes:
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate_mfvi_router(
            model, tokenizer, test_raw_dataset,
            dataset_name=shortcode, num_samples=args.num_samples,
            batch_size=args.batch_size
        )
        all_results.append(results)

    # 4. Save results to a CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.results_csv_path, index=False)
    print(f"\nSaved all evaluation results to {args.results_csv_path}")

if __name__ == "__main__":
    main()