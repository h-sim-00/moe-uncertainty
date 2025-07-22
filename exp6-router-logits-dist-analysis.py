# filename: exp6-router-logits-dist-analysis.py
# =============================================================================
# This script performs the data collection for our observational study.
# Its sole purpose is to run a model over a dataset and save the raw
# router logits from every layer for every prompt. This raw data will be
# processed and analysed by a separate script.
# =============================================================================

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Local imports
from data import multiple_choice_prompt_engineer, load_classification_dataset
from utils import setup_environment
# This assumes your model.py has a class with the methods to dynamically
# change router modes if needed, though for this script, we use the default.
from model import load_model

# --- Hooking Mechanism to Capture Intermediate Outputs ---
# This global list will store the logit tensors for each layer FOR A SINGLE prompt.
# It will be cleared after each prompt is processed.
prompt_layer_logits = []

def get_router_output_hook():
    """
    This hook captures the output of a router module during a forward pass.
    """
    def hook(model, input, output):
        """
        INSTRUCTION: Verify the output index. For GraniteMoE, the logits are
        the 5th element (index 4) of the router's output tuple.
        We store the raw logits on the CPU to avoid accumulating GPU memory.
        """
        global prompt_layer_logits
        # We append to the list, so the order corresponds to the layer order.
        prompt_layer_logits.append(output[-1].cpu())
    return hook

def register_hooks(model):
    """
    This function finds all the router modules in the model and attaches a hook.
    """
    handles = []
    # INSTRUCTION: Verify this path matches your model's architecture.
    for i, layer in enumerate(model.model.layers):
        handle = layer.block_sparse_moe.router.register_forward_hook(get_router_output_hook())
        handles.append(handle)
    print(f"Registered {len(handles)} hooks on MoE routers.")
    return handles

def remove_hooks(handles):
    """Remove all registered hooks to clean up."""
    for handle in handles:
        handle.remove()

# --- Main Script ---
def parse_args():
    parser = argparse.ArgumentParser(description="Collect router logits from an MoE model.")
    parser.add_argument("--model_id", type=str, default="ibm-granite/granite-3.1-3b-a800m-instruct", help="Base model identifier.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional path to a PEFT adapter.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to run analysis on (e.g., arc_easy, medmcqa).")
    parser.add_argument("--num_prompts", type=int, default=500, help="Number of prompts to process from the dataset.")
    parser.add_argument("--output_dir", type=str, default="./figs/phase-4", help="Directory to save the collected data.")
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
    
    # --- Data Collection Loop ---
    num_model_layers = model.config.num_hidden_layers
    # This list will store the results for all prompts.
    # Structure: [ [layer_0_logits, layer_1_logits, ...],  # Prompt 1
    #              [layer_0_logits, layer_1_logits, ...],  # Prompt 2 ... ]
    all_prompts_data = []

    # Register hooks once before the loop.
    handles = register_hooks(model)
    
    # Ensure the model is in deterministic mode for this observational study.
    # INSTRUCTION: You need a method on your model class to set all routers to top_k.
    model.change_router_mode('top_k') 

    with torch.no_grad():
        for question in tqdm(questions, desc=f"Processing prompts for {args.dataset_name}"):
            # IMPORTANT: Clear the global list for each new prompt.
            global prompt_layer_logits
            prompt_layer_logits = []
            
            inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _ = model(**inputs)
            
            # --- MODIFICATION: Process logits to ensure consistent shape ---
            # `prompt_layer_logits` is a list of tensors, where each tensor has shape (num_tokens, num_experts).
            # We average over the token dimension to get a single representative logit vector for the prompt for each layer.
            avg_logits_for_prompt = []
            for layer_logits_tensor in prompt_layer_logits:
                # Average across the token dimension (dim=0)
                avg_layer_logits = layer_logits_tensor.mean(dim=0)
                avg_logits_for_prompt.append(avg_layer_logits)
            
            # Now, avg_logits_for_prompt is a list of tensors of consistent shape (num_experts,),
            # which can be safely stored.
            all_prompts_data.append(avg_logits_for_prompt)

    # Clean up hooks after processing all prompts.
    remove_hooks(handles)

    # --- Save Data ---
    # Convert the list of lists of tensors into a more manageable format.
    # We will have a list of numpy arrays, where each array corresponds to a layer.
    # Each array will have shape (num_prompts, num_experts)
    num_experts = all_prompts_data[0][0].shape[-1]
    # Restructure the data: one list per layer
    layer_data = [[] for _ in range(num_model_layers)]
    for prompt_data in all_prompts_data:
        for layer_idx, logits in enumerate(prompt_data):
            # Ensure logits are squeezed to 1D vector of shape [num_experts]
            layer_data[layer_idx].append(logits.squeeze().numpy())

    # Convert lists to numpy arrays
    final_layer_data = [np.array(data) for data in layer_data]

    # Save the data to a .npy file for the analysis script.
    output_path = os.path.join(args.output_dir, f"{args.dataset_name}_router_logits.npy")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving collected logit data to: {output_path}")
    np.save(output_path, final_layer_data, allow_pickle=True) # allow_pickle is needed for list of arrays
    
    print("Data collection complete.")

if __name__ == "__main__":
    main()
