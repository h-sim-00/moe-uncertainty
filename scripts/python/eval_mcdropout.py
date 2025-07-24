import argparse
import torch
import pandas as pd
from utils import setup_environment, load_classification_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from model.routers.mcdropout import add_mcdropout_routers_to_model, evaluate_router, load_router_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned MC Dropout MoE router.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of Monte Carlo samples for inference.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--dropout_rate", type=float, default=0.05, help="Dropout rate used during training, for re-instantiating the router.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    print(f"Loading base model '{args.model_shortcode}' and attaching adapter from '{base_adapter_path}'")
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. "Lego Swap": Replace original routers and load the trained weights
    router_weights_path = f"./adapters/mcdropout_ft_{args.model_shortcode}_dor-{args.dropout_rate}_seed-{args.seed}/router_weights.pt"
    model = add_mcdropout_routers_to_model(model, args.dropout_rate)
    model = load_router_weights(model, router_weights_path)
    model.eval()
    print("Model is ready for evaluation.")

    # 3. Run evaluation across all specified datasets
    print("\n--- Starting Evaluation ---")
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []

    for shortcode in dataset_shortcodes:
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate_router(model, tokenizer, test_raw_dataset, dataset_name=shortcode, num_samples=args.num_samples, batch_size=args.batch_size)
        all_results.append(results)

    # 4. Save results to a CSV
    results_csv_path = f"./results/eval_mcdropout-ft_{args.model_shortcode}_dor-{args.dropout_rate}_n-{args.num_samples}_seed-{args.seed}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all evaluation results to {results_csv_path}")

if __name__ == "__main__":
    main()