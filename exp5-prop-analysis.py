# filename: exp5-prop-analysis.py
# =============================================================================
# This script is designed to analyse how a stochastic perturbation at one layer
# in a Mixture of Experts (MoE) model affects the routing decisions of all
# subsequent layers. It provides a quantitative map of the model's internal
# sensitivity and the flow of uncertainty.
# =============================================================================

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

# --- Local Imports ---
# You will need your own custom modules for loading data and models.
from data import multiple_choice_prompt_engineer, load_classification_dataset
from utils import setup_environment

from model import load_model

# =============================================================================
#                           PART 1: THE HOOKING MECHANISM
# Rationale: LLMs perform a single forward pass. To see what's happening
# inside the model (e.g., at the output of a router in layer 15), we can't
# just `return` that value. We need to "hook" into the model's execution flow.
# A forward hook is a function that PyTorch runs automatically every time a
# specific module (like a router) completes its forward pass. We use it to
# non-invasively capture the output of each router.
# =============================================================================

# This global dictionary will act as a temporary storage for the outputs
# captured by our hooks during a single forward pass.
router_outputs = {}

def get_router_output_hook(name):
    """
    This is a "hook factory". It creates a unique hook function for each layer.
    The `name` argument (e.g., 'layer_15') is used as the key in our global dictionary.
    """
    def hook(model, input, output):
        """
        This is the actual hook function. PyTorch calls it with the module, its input, and its output.
        - The router's raw output is often a tuple. Based on your model's architecture,
          the logits are the first element, so we take `output[0]`.
        - We apply softmax here to get a clean probability distribution.
        - We move the tensor to the CPU to avoid filling up GPU memory, as we will
          accumulate many of these from different layers.
        """
        global router_outputs
        router_outputs[name] = torch.softmax(output[-1], dim=-1).cpu()
    return hook

def register_hooks(model):
    """
    This function finds all the router modules in your model and attaches a hook to each one.
    """
    handles = []
    # INSTRUCTION: You must verify this path matches your model's architecture exactly.
    # For GraniteMoE, the path to a router in layer `i` is `model.layers[i].block_sparse_moe.router`.
    for i, layer in enumerate(model.model.layers):
        # Create a unique hook for this layer, e.g., get_router_output_hook('layer_15')
        handle = layer.block_sparse_moe.router.register_forward_hook(
            get_router_output_hook(f'layer_{i}')
        )
        handles.append(handle)
    print(f"Registered {len(handles)} hooks on MoE routers.")
    return handles

def remove_hooks(handles):
    """
    After a forward pass, it's good practice to remove the hooks to prevent
    unintended side effects. This function iterates through the handles returned
    by `register_hooks` and removes them.
    """
    for handle in handles:
        handle.remove()

# =============================================================================
#                           PART 2: SCRIPT EXECUTION LOGIC
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse heuristic uncertainty propagation in an MoE model.")
    parser.add_argument("--model_id", type=str, default="ibm-granite/granite-3.1-3b-a800m-instruct", help="Base model identifier.")
    # parser.add_argument("--adapter_path", type=str, default=None, help="Optional path to a PEFT adapter if using a fine-tuned model.")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset to use for analysis.")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of prompts to average results over. A higher number gives more stable results.")
    parser.add_argument("--intervention_temp", type=float, default=1.5, help="The temperature to use for the stochastic routing intervention.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model and Data ---
    print(f"Loading model: {args.model_id}")
    model = load_model(model_id=args.model_id, device_map=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    _, _, test_raw = load_classification_dataset(args.dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_raw][:args.num_prompts]
    questions = [x['question'] for x in test_dataset]
    
    # --- Analysis Loop ---
    num_model_layers = model.config.num_hidden_layers
    # We intervene on all robust layers, skipping the first two which are often too sensitive.
    intervention_layers = range(1, num_model_layers) 
    
    # This matrix will store our final results for the heatmap.
    jsd_heatmap_data = np.full((num_model_layers, num_model_layers), np.nan)

    # Register hooks once before the loops begin.
    handles = register_hooks(model)

    print(f"Current intervention_temp: {args.intervention_temp}")
    print(f"Beginning analysis over {len(intervention_layers)} intervention layers and {args.num_prompts} prompts.")

    # Outer loop: This loop decides WHICH layer to apply the intervention to.
    for intervention_layer in tqdm(intervention_layers, desc="Iterating Interventions"):
        print(f"\n--- Intervention at layer {intervention_layer} ---")
        # This temporary matrix stores all JSD results for this specific intervention layer
        # before we average them. Shape: (num_layers, num_prompts)
        jsd_matrix_for_layer = np.zeros((num_model_layers, args.num_prompts))

        with torch.no_grad():
            # Inner loop: We repeat the experiment for many different prompts to get a reliable average.
            for i, question in enumerate(questions):
                if i % 10 == 0:
                    print(f"  Processing prompt {i+1}/{len(questions)} for intervention layer {intervention_layer}...")

                inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

                # --- 1. Baseline Pass (all layers deterministic) ---
                global router_outputs
                router_outputs = {} # Clear the global storage before the pass.
                
                # INSTRUCTION: You need to implement a method on your model class
                # that sets all routers to the deterministic 'top_k' mode.
                model.change_router_mode('top_k') 
                
                _ = model(**inputs) # Perform the forward pass. Hooks will populate `router_outputs`.
                baseline_outputs = router_outputs.copy()

                # --- 2. Intervention Pass (one layer stochastic) ---
                router_outputs = {} # Clear the storage again.
                
                # INSTRUCTION: You need to implement a method that can set the routing
                # mode and temperature for a SINGLE layer, while leaving others as default.
                model.change_router_mode_for_layers(
                    mode='sample_k', 
                    layer_indices=[intervention_layer], 
                    temp=args.intervention_temp
                )
                
                _ = model(**inputs) # Perform the intervention pass.
                perturbed_outputs = router_outputs.copy()

                # --- 3. Calculate JSD for subsequent layers ---
                # We only care about the "ripple effect" on layers AFTER the intervention.
                for j in range(intervention_layer + 1, num_model_layers):
                    layer_name = f'layer_{j}'
                    if layer_name in baseline_outputs and layer_name in perturbed_outputs:
                        p_base = baseline_outputs[layer_name].numpy().flatten()
                        p_pert = perturbed_outputs[layer_name].numpy().flatten()
                        # Jensen-Shannon Divergence is a symmetric way to measure the "distance"
                        # between the two probability distributions. A score of 0 means they are identical.
                        jsd = jensenshannon(p_base, p_pert, base=2.0)
                        jsd_matrix_for_layer[j, i] = jsd
                        if i == 0 and (j == intervention_layer + 1 or j == num_model_layers - 1):
                            print(f"    JSD (layer {j}) for first prompt: {jsd:.4f}")
        
        # Average the JSD scores over all prompts for this intervention layer.
        # This gives us a stable estimate of the impact.
        jsd_heatmap_data[:, intervention_layer] = np.mean(jsd_matrix_for_layer, axis=1)
        print(f"  Mean JSD for intervention at layer {intervention_layer}: {np.nanmean(jsd_heatmap_data[:, intervention_layer]):.4f}")
        
    # Clean up hooks after all loops are done.
    remove_hooks(handles)

    # --- Dump the JSD heatmap data to a file for further analysis ---
    os.makedirs("./figs/phase-3", exist_ok=True)
    np.save(f"./figs/phase-3/jsd_heatmap_data_T={args.intervention_temp}.npy", jsd_heatmap_data)
    print(f"JSD heatmap data saved to ./figs/phase-3/jsd_heatmap_data_T={args.intervention_temp}.npy")

if __name__ == "__main__":
    main()
