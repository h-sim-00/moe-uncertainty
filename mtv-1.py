import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import json

# Assume these are in your project structure
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_classification_dataset 

# --- Global list to hold hook data ---
captured_logits = []

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the refined router brittleness experiment.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--adapter_path", type=str, default="./adapters/kvq_ft_granite_seed-42", help="Path to the saved MAP baseline adapter.")
    parser.add_argument("--sentences_csv", type=str, default="ref/sentences.csv", help="Path to the CSV file with sentences.")
    parser.add_argument("--output_dir", type=str, default="./draw/mtv-1", help="Directory to save all results.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    return parser.parse_args()

def calibrate_and_prepare(model, tokenizer, sentences, device):
    """Calculates L_bar and attaches hooks to the model's routers."""
    global captured_logits
    print("--- Stage 1: Calibrating Noise Level & Preparing Model ---")

    # --- Calibrate Noise ---
    all_norms = []
    # Correct path to the embedding layer
    embed_layer = model.base_model.model.model.embed_tokens
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Calibrating Noise"):
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            embeddings = embed_layer(inputs.input_ids)
            norms = torch.linalg.norm(embeddings.view(-1, embeddings.shape[-1]).float(), ord=2, dim=1)
            all_norms.append(norms.cpu())
    l_bar = torch.cat(all_norms).mean().item()
    print(f"Average L2 Norm of Embeddings (L_bar): {l_bar:.4f}")

    # --- Attach Hooks ---
    def router_capture_hook(module, input, output):
        captured_logits.append(output[-1].detach().cpu())

    # Correct path to the decoder layers
    for layer in model.base_model.model.model.layers:
        layer.block_sparse_moe.router.register_forward_hook(router_capture_hook)
    print("Attached capture hooks to all routers.")
    
    return l_bar, model

def collect_routing_data(model, tokenizer, sentences, l_bar, gammas, batch_size, device, output_dir):
    """Runs inference for each noise level and saves the raw logits."""
    global captured_logits
    
    print("\n--- Stage 2: Collecting Raw Routing Data ---")
    
    perturbation_noise = None
    def embedding_perturb_hook(module, input, output):
        if perturbation_noise is not None:
            noise = perturbation_noise[:output.shape[1], :].to(output.device)
            return output + noise.unsqueeze(0)
        return output
    
    # Correct path to the embedding layer
    hook_handle = model.base_model.model.model.embed_tokens.register_forward_hook(embedding_perturb_hook)

    for gamma in gammas:
        sigma = gamma * l_bar
        gamma_str = f"{gamma:.3f}".replace('.', 'p')
        output_path = os.path.join(output_dir, f"raw_logits_gamma_{gamma_str}.pt")
        
        if os.path.exists(output_path):
            print(f"Data for gamma={gamma:.3f} already exists. Skipping.")
            continue

        print(f"\nCollecting data for gamma = {gamma:.3f} (sigma = {sigma:.6f})")
        
        # --- Original Pass ---
        perturbation_noise = None
        captured_logits.clear()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc=f"Original Pass (gamma={gamma:.3f})"):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                model(inputs.input_ids)
        original_logits = list(captured_logits)

        # --- Perturbed Pass ---
        max_len = model.config.max_position_embeddings
        embed_dim = model.config.hidden_size
        perturbation_noise = torch.randn(max_len, embed_dim) * sigma
        captured_logits.clear()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc=f"Perturbed Pass (gamma={gamma:.3f})"):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                model(inputs.input_ids)
        perturbed_logits = list(captured_logits)
        
        torch.save({
            "gamma": gamma,
            "sigma": sigma,
            "original": original_logits,
            "perturbed": perturbed_logits
        }, output_path)
        print(f"Saved data to {output_path}")

    hook_handle.remove()

def compute_jaccard_scores(output_dir, top_k, num_layers):
    """Processes raw logits files to compute and save Jaccard scores."""
    print("\n--- Stage 3: Computing Jaccard Scores ---")
    all_scores = []
    
    data_files = [f for f in os.listdir(output_dir) if f.startswith("raw_logits_gamma_") and f.endswith(".pt")]
    
    for filename in tqdm(data_files, desc="Processing Raw Data Files"):
        data = torch.load(os.path.join(output_dir, filename))
        gamma = data['gamma']
        
        num_batches = len(data['original']) // num_layers
        
        original_by_layer = [torch.cat([data['original'][i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]
        perturbed_by_layer = [torch.cat([data['perturbed'][i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]
        
        for layer_idx in range(num_layers):
            orig_indices = torch.topk(original_by_layer[layer_idx], k=top_k, dim=-1).indices
            pert_indices = torch.topk(perturbed_by_layer[layer_idx], k=top_k, dim=-1).indices
            
            for token_orig, token_pert in zip(orig_indices, pert_indices):
                set_orig = set(token_orig.tolist())
                set_pert = set(token_pert.tolist())
                intersection = len(set_orig.intersection(set_pert))
                union = len(set_orig.union(set_pert))
                score = intersection / union if union > 0 else 0
                all_scores.append({"gamma": gamma, "layer": layer_idx, "jaccard_score": score})
                
    return all_scores

def main():
    """Orchestrates the entire experiment."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.adapter_path, device_map=device)
    model.eval()
    tokenizer = load_tokenizer(args.model_shortcode)


    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_raw_dataset = load_classification_dataset(train_dataset_shortcodes[1], split="test")
    sentences = [ex["question"] for ex in train_raw_dataset][:50]

    # Correct path to the decoder layers
    num_layers = len(model.base_model.model.model.layers)
    top_k = model.config.num_experts_per_tok
    gammas = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05]
    
    # --- Stage 1: Calibrate & Prepare ---
    l_bar, model = calibrate_and_prepare(model, tokenizer, sentences, device)
    
    # --- Stage 2: Collect Data ---
    collect_routing_data(model, tokenizer, sentences, l_bar, gammas, args.batch_size, device, args.output_dir)
    
    # --- Stage 3: Compute Scores and Save Final JSON ---
    final_json_path = os.path.join(args.output_dir, "all_jaccard_scores.json")
    if not os.path.exists(final_json_path):
        all_scores_list = compute_jaccard_scores(args.output_dir, top_k, num_layers)
        with open(final_json_path, 'w') as f:
            json.dump(all_scores_list, f)
        print(f"\nSaved final Jaccard scores to {final_json_path}")
    else:
        print(f"\nFinal Jaccard scores file already exists at {final_json_path}. Skipping computation.")

    print("\nExperiment data collection complete.")

if __name__ == "__main__":
    main()