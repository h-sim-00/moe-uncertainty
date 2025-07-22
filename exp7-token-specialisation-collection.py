# filename: exp7-token-specialisation-collection.py
# =============================================================================
# This script is designed for the "Phase 4.5: Expert Specialisation" analysis.
# Its purpose is to perform a single forward pass for each prompt in a dataset
# and save a detailed record of each token's journey. This includes its
# linguistic features (POS, NER) and which MoE experts were chosen to process
# it at every layer. This granular data is essential for the subsequent analysis.
# =============================================================================

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy

# Local imports
from data import multiple_choice_prompt_engineer, load_classification_dataset
from utils import setup_environment
from model import load_model # Using the new, simpler model loader

# --- INSTRUCTION: You may need to download the spaCy model first ---
# In your terminal, run: python -m spacy download en_core_web_sm
NLP_PROCESSOR = None

# --- Hooking Mechanism to Capture Router's Chosen Experts ---
# This global list will store the chosen expert indices for each layer FOR A SINGLE prompt.
prompt_layer_expert_indices = []

def get_router_output_hook():
    """
    This hook captures the router's output. We deduce the top_k indices
    from the returned logits, as this is safer than modifying model code.
    """
    def hook(model, input, output):
        global prompt_layer_expert_indices
        # The output tuple is (index_sorted_experts, ..., logits)
        # We grab the logits, which are the 5th element (index 4).
        logits = output[4] 
        
        # Get the value of 'k' from the router's configuration.
        # This assumes your router module has a `top_k` attribute.
        k = model.top_k
        
        # Re-run the topk operation to deduce the chosen expert indices.
        _, top_k_indices = torch.topk(logits, k, dim=-1)
        
        prompt_layer_expert_indices.append(top_k_indices.cpu())
    return hook

def register_hooks(model):
    """Finds all router modules and attaches a forward hook."""
    handles = []
    # INSTRUCTION: Verify this path matches your model's architecture.
    for i, layer in enumerate(model.model.layers):
        handle = layer.block_sparse_moe.router.register_forward_hook(get_router_output_hook())
        handles.append(handle)
    return handles

def remove_hooks(handles):
    """Removes all registered hooks."""
    for handle in handles:
        handle.remove()

def parse_args():
    parser = argparse.ArgumentParser(description="Collect per-token expert choices and linguistic features.")
    parser.add_argument("--model_id", type=str, default="ibm-granite/granite-3.1-3b-a800m-instruct")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to analyse (e.g., arc_easy, medmcqa).")
    parser.add_argument("--num_prompts", type=int, default=200, help="Number of prompts to process.")
    parser.add_argument("--output_dir", type=str, default="./figs/phase-4.5")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    
    # --- Load spaCy model for linguistic features ---
    global NLP_PROCESSOR
    try:
        NLP_PROCESSOR = spacy.load("en_core_web_sm")
    except IOError:
        print("spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model and Data ---
    print(f"Loading model: {args.model_id}")
    model = load_model(model_id=args.model_id, device_map=device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    _, _, test_raw = load_classification_dataset(args.dataset_name)
    test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_raw][:args.num_prompts]
    
    # --- Data Collection Loop ---
    all_prompts_data = []
    handles = register_hooks(model)
    
    if hasattr(model, 'change_router_mode'):
        model.change_router_mode(mode='top_k')

    with torch.no_grad():
        for prompt_idx, sample in enumerate(tqdm(test_dataset, desc=f"Processing prompts for {args.dataset_name}")):
            global prompt_layer_expert_indices
            prompt_layer_expert_indices = []
            
            inputs = tokenizer(sample['question'], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _ = model(**inputs)
            
            token_ids = inputs.input_ids[0]
            tokens_text = tokenizer.convert_ids_to_tokens(token_ids)
            word_ids = inputs.word_ids(batch_index=0)
            doc = NLP_PROCESSOR(sample['question'])
            
            # --- MODIFICATION: Added try-except block for debugging ---
            try:
                token_data_list = []
                for i, word_id in enumerate(word_ids):
                    if word_id is None:
                        pos_tag = "SPECIAL"
                        ner_tag = "SPECIAL"
                    else:
                        # The line that causes the error is now inside the try block
                        spacy_token = doc[word_id]
                        pos_tag = spacy_token.pos_
                        ner_tag = spacy_token.ent_type_ if spacy_token.ent_type_ else "O"
                    
                    token_data_list.append({
                        "token": tokens_text[i],
                        "pos_tag": pos_tag,
                        "ner_tag": ner_tag
                    })

                expert_choices_per_layer = torch.stack(prompt_layer_expert_indices, dim=0).squeeze(1).numpy()
                
                for token_idx, _ in enumerate(token_data_list):
                     token_data_list[token_idx]["expert_choices"] = expert_choices_per_layer[:, token_idx, :]

                all_prompts_data.append({
                    "prompt_id": prompt_idx,
                    "full_text": sample['question'],
                    "token_data": token_data_list
                })

            except IndexError as e:
                # This block will execute if an IndexError occurs, printing detailed debug info.
                print("\n\n--- CAUGHT AN INDEX ERROR! ---")
                print(f"Error occurred at prompt_idx: {prompt_idx}")
                print(f"Original Question: {sample['question']}")
                print(f"spaCy Doc Length: {len(doc)}")
                print("\n--- Tokenization Mismatch Details ---")
                print("HF Tokens | HF word_id | spaCy Token (at word_id)")
                print("--------------------------------------------------")
                for i, word_id in enumerate(word_ids):
                    error_marker = "<-- ERROR HERE" if word_id is not None and word_id >= len(doc) else ""
                    spacy_token_text = "N/A"
                    if word_id is not None and word_id < len(doc):
                        spacy_token_text = doc[word_id].text
                    print(f"{tokens_text[i]:<15} | {str(word_id):<10} | {spacy_token_text} {error_marker}")
                
                print("\n--- Raw Data ---")
                print(f"Full word_ids list: {word_ids}")
                print(f"spaCy doc tokens: {[tok.text for tok in doc]}")
                print("-----------------------------------")
                # We will stop on the first error to analyse it.
                # Remove `exit()` to continue and see all errors.
                exit(1)

    remove_hooks(handles)

    # --- Save Data ---
    output_path = os.path.join(args.output_dir, f"{args.dataset_name}_token_specialisation_data.npy")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving collected per-token data to: {output_path}")
    np.save(output_path, all_prompts_data, allow_pickle=True)
    
    print("Data collection complete.")

if __name__ == "__main__":
    main()
