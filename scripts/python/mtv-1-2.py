import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import json
import shutil

# Assume these are in your project structure
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_classification_dataset # Assuming this can load the sentences

# --- A class to manage hook data cleanly ---
class HookManager:
    def __init__(self):
        self.captured_data = {}

    def create_hook(self, layer_idx):
        def hook(module, input, output):
            if layer_idx not in self.captured_data:
                self.captured_data[layer_idx] = {'inputs': [], 'logits': []}
            # input[0] is the hidden_states tensor
            self.captured_data[layer_idx]['inputs'].append(input[0].detach().cpu())
            # output[-1] is the logits tensor
            self.captured_data[layer_idx]['logits'].append(output[-1].detach().cpu())
        return hook

    def clear(self):
        self.captured_data.clear()

    def get_data(self):
        """Concatenates the collected batch tensors for each layer."""
        processed_data = {}
        for layer_idx, data in self.captured_data.items():
            processed_data[layer_idx] = {
                'inputs': torch.cat(data['inputs']),
                'logits': torch.cat(data['logits'])
            }
        return processed_data

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
    print("--- Stage 1: Calibrating Noise Level & Preparing Model ---")

    all_norms = []
    embed_layer = model.base_model.model.model.embed_tokens
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Calibrating Noise"):
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            embeddings = embed_layer(inputs.input_ids)
            norms = torch.linalg.norm(embeddings.view(-1, embeddings.shape[-1]).float(), ord=2, dim=1)
            all_norms.append(norms.cpu())
    l_bar = torch.cat(all_norms).mean().item()
    print(f"Average L2 Norm of Embeddings (L_bar): {l_bar:.4f}")

    hook_manager = HookManager()
    for i, layer in enumerate(model.base_model.model.model.layers):
        layer.block_sparse_moe.router.register_forward_hook(hook_manager.create_hook(i))
    print("Attached capture hooks to all routers.")
    
    return l_bar, model, hook_manager

def collect_routing_data(model, tokenizer, sentences, hook_manager, l_bar, gammas, batch_size, device, output_dir):
    """Runs inference once to get original data, then runs perturbed passes for each layer."""
    print("\n--- Stage 2: Collecting Raw Routing Data ---")
    
    original_data_path = os.path.join(output_dir, "original_router_data.pt")

    # --- 2.1: Original Pass ---
    if not os.path.exists(original_data_path):
        print("Running inference on original sentences to capture inputs and logits...")
        hook_manager.clear()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Original Inference Pass"):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                model(inputs.input_ids)
        
        original_data = hook_manager.get_data()
        torch.save(original_data, original_data_path)
        print(f"Saved original router inputs and logits to {original_data_path}")
    else:
        print(f"Found existing original data at {original_data_path}. Loading.")
        original_data = torch.load(original_data_path)

    # --- 2.2: Perturbed Passes ---
    causal_model = model.base_model.model.model
    for gamma in gammas:
        sigma = gamma * l_bar
        gamma_str = f"{gamma:.3f}".replace('.', 'p')
        output_path = os.path.join(output_dir, f"perturbed_logits_gamma_{gamma_str}.pt")

        if os.path.exists(output_path):
            print(f"Perturbed data for gamma={gamma:.3f} already exists. Skipping.")
            continue

        print(f"\nGenerating perturbed logits for gamma = {gamma:.3f} (sigma = {sigma:.6f})")
        perturbed_logits_by_layer = {}
        with torch.no_grad():
            for layer_idx in tqdm(range(len(causal_model.layers)), desc=f"Perturbing Layers (gamma={gamma:.3f})"):
                layer = causal_model.layers[layer_idx]
                router = layer.block_sparse_moe.router
                
                # Get the original hidden states for this layer
                original_hidden_states = original_data[layer_idx]['inputs'].to(device)
                
                # Add fresh noise
                noise = torch.randn_like(original_hidden_states) * sigma
                perturbed_hidden_states = original_hidden_states + noise
                
                # Pass only through this layer's router
                perturbed_logits = router(perturbed_hidden_states)[-1]
                perturbed_logits_by_layer[layer_idx] = perturbed_logits.cpu()

        torch.save(perturbed_logits_by_layer, output_path)
        print(f"Saved perturbed logits to {output_path}")

def compute_jaccard_scores(output_dir, top_k, num_layers):
    """Processes raw data files to compute and save final Jaccard scores."""
    print("\n--- Stage 3: Computing Jaccard Scores ---")
    
    original_data_path = os.path.join(output_dir, "original_router_data.pt")
    if not os.path.exists(original_data_path):
        print("Error: Original router data not found. Please run Stage 2 first.")
        return None
        
    original_data = torch.load(original_data_path)
    all_scores = []
    
    data_files = [f for f in os.listdir(output_dir) if f.startswith("perturbed_logits_gamma_") and f.endswith(".pt")]
    
    for filename in tqdm(data_files, desc="Processing Perturbed Data Files"):
        perturbed_data = torch.load(os.path.join(output_dir, filename))
        gamma = float(filename.split('_')[-1].replace('.pt', '').replace('p', '.'))
        
        for layer_idx in range(num_layers):
            original_logits = original_data[layer_idx]['logits']
            pert_logits = perturbed_data[layer_idx]
            
            orig_indices = torch.topk(original_logits, k=top_k, dim=-1).indices
            pert_indices = torch.topk(pert_logits, k=top_k, dim=-1).indices
            
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
    df_sentences = pd.read_csv(args.sentences_csv)
    sentences = df_sentences['sentence'].tolist()
    
    num_layers = len(model.base_model.model.model.layers)
    top_k = model.config.num_experts_per_tok
    gammas = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05]
    
    # --- Stage 1: Calibrate & Prepare ---
    l_bar, model, hook_manager = calibrate_and_prepare(model, tokenizer, sentences, device)
    
    # --- Stage 2: Collect Data ---
    collect_routing_data(model, tokenizer, sentences, hook_manager, l_bar, gammas, args.batch_size, device, args.output_dir)
    
    # --- Stage 3: Compute Scores and Save Final JSON ---
    final_json_path = os.path.join(args.output_dir, "all_jaccard_scores.json")
    if not os.path.exists(final_json_path):
        all_scores_list = compute_jaccard_scores(args.output_dir, top_k, num_layers)
        if all_scores_list:
            with open(final_json_path, 'w') as f:
                json.dump(all_scores_list, f)
            print(f"\nSaved final Jaccard scores to {final_json_path}")
    else:
        print(f"\nFinal Jaccard scores file already exists at {final_json_path}. Skipping computation.")

    print("\nExperiment data collection complete.")

if __name__ == "__main__":
    main()
