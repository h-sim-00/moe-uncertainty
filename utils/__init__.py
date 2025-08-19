import os
import wandb
from huggingface_hub import login as hf_login
import torch
from tqdm import tqdm
import torch.nn.functional as F

def setup_environment():
    """Configures HuggingFace cache and logs into services."""
    # These paths are specific to your cluster environment
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if not IN_COLAB:
        print("Setting up new cache paths for huggingface and Wandb...")
        
        HF_HOME = "/vol/bitbucket/al1624/.cache/huggingface"
        HF_DATASETS_CACHE = "/vol/bitbucket/al1624/.cache/huggingface/datasets"
        HF_HUB_CACHE="/vol/bitbucket/al1624/.cache/huggingface/hub"
        WANDB_STORAGE_DIR = "/vol/bitbucket/al1624/.cache/wandb"

        os.makedirs(WANDB_STORAGE_DIR, exist_ok=True)

        os.environ['HF_HOME'] = HF_HOME
        os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
        os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
        os.environ['WANDB_DIR'] = WANDB_STORAGE_DIR

    # Login to HuggingFace
    print("Logging into HuggingFace...")
    HF_TOKEN = "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"
    hf_login(token=HF_TOKEN)
    print("Login successful.")

    # Login to Wandb
    WANDB_KEY = "8d44174f1416d56dc5470b57deb50339b19f22e7"
    wandb.login(key=WANDB_KEY)

from .data import (
    load_generation_dataset,
    load_classification_dataset,
    batchify,
    preprocess_mask_question_for_training,
    load_and_prepare_train_and_val_data,
    load_exp_dataset,
)

from .prompt import (
    multi_shot_prompt_engineer,
    multiple_choice_prompt_engineer,
)

def get_model_predictions(model, tokenizer, dataset, batch_size=8):
    """Performs inference and returns predictions, probabilities, and labels."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    # The choices are fixed for MMLU style questions
    choices = ['A', 'B', 'C', 'D']
    choice_token_ids = {c: tokenizer.convert_tokens_to_ids(c) for c in choices}
    
    # Check if any choice token is unknown
    if any(v is None for v in choice_token_ids.values()):
        raise ValueError("One of the choice tokens 'A', 'B', 'C', 'D' is not in the tokenizer's vocabulary.")
        
    choice_ids_tensor = torch.tensor(list(choice_token_ids.values()), device=model.device)

    # Pre-process the entire dataset
    processed_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x['question'] for x in processed_dataset]
    correct_answers = [x['answer'] for x in processed_dataset]
    label_map = {label: i for i, label in enumerate(choices)}
    labels_numeric = [label_map[ans] for ans in correct_answers]

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), batch_size), desc="Evaluating"):
            batch_questions = questions[i:i+batch_size]
            inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)

            logits = model(**inputs).logits
            next_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            # Restrict logits to only the valid choice tokens (A, B, C, D)
            choice_logits = next_token_logits[:, choice_ids_tensor] # Shape: (batch_size, num_choices)
            
            # Apply softmax to get probabilities over choices
            probs = F.softmax(choice_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
    
    all_labels = torch.tensor(labels_numeric)
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)

    return all_preds, all_probs, all_labels

from .metrics import *