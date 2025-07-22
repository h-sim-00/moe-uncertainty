PARAM_NUM="3b"

# Set HuggingFace cache directories
import os
HF_HOME = "/vol/bitbucket/al1624/.cache/huggingface"
HF_DATASETS_CACHE = "/vol/bitbucket/al1624/.cache/huggingface/datasets"
HF_HOME = "/vol/bitbucket/al1624/.cache/huggingface"
HF_DATASETS_CACHE = "/vol/bitbucket/al1624/.cache/huggingface/datasets"

# Set wandb and HuggingFace tokens
WANDB_KEY = "8d44174f1416d56dc5470b57deb50339b19f22e7"
HF_TOKEN = "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"
import os
import wandb
from huggingface_hub import login as hf_login
wandb.login(key=WANDB_KEY)
hf_login(token=HF_TOKEN)
import torch
from transformers import AutoTokenizer
from data import multiple_choice_prompt_engineer, load_classification_dataset, batchify
from model import load_model
from torchmetrics.classification import MulticlassCalibrationError
from datasets import Dataset
import json
from tqdm import tqdm

BATCH_SIZE = 16
MAX_NEW_TOKENS = 10

# Model and Tokenizer
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct" if PARAM_NUM == "1b" else "ibm-granite/granite-3.1-3b-a800m-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = load_model(MODEL_ID)

# Dataset
DATASET_NAME = "medmcqa"
train_dataset, val_dataset, test_dataset = load_classification_dataset(DATASET_NAME)
train_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_dataset]
val_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_dataset]
test_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in test_dataset]

print(len(train_dataset), len(val_dataset), len(test_dataset))

# set project name
project_name = "exp1_evaluation"
exp_name = f"{PARAM_NUM}_baseline"

# Load post training peft model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Prepare dataset and record containers
questions = [x['question'] for x in test_dataset]
true_answers = [x['answer'] for x in test_dataset]
ids = [x['id'] for x in test_dataset]
abcd_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in ['A', 'B', 'C', 'D']]

preds = []
all_logits = []
all_nlls = []
all_entropies = [] # (NEW) For storing entropy of each prediction
all_abcd_probs = [] # (NEW) For storing probabilities of each choice
output_data_for_json = [] # (NEW) For storing data to be dumped

run = wandb.init(
    project=project_name,
    name=exp_name,
    config={
        "model_id": MODEL_ID,
        "param_num": PARAM_NUM,
        "dataset_name": DATASET_NAME,
        "exp_type": "baseline",
    }
)

## 2. Inference
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(questions), BATCH_SIZE)):
        batch_q = questions[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_true_answers = true_answers[i:i+BATCH_SIZE]
        # Infernence and get last-token logits
        inputs = tokenizer(
            batch_q,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        logits = model(**inputs).logits
        assert not torch.isnan(logits).any(), "NaN detected in logits tensor"
        next_token_logits = logits[:, -1, :] # batch_size * 1 * vocab_size

        # Calculate probabilities and entropies for the batch
        probs = torch.softmax(next_token_logits, dim=-1) # Shape: [batch_size, vocab_size]
        log_probs_for_entropy = torch.log_softmax(next_token_logits, dim=-1) # Using log_softmax for numerical stability
        # Calculate entropy for each sample in the batch
        # More stable: - sum(probs * log_probs_for_entropy)
        batch_entropies = -torch.sum(probs * log_probs_for_entropy, dim=-1) # Shape: [batch_size]
        all_entropies.extend(batch_entropies.cpu().tolist())

        # Compute negative log-likelihood (NLL) for each sample in the batch
        for j, true_ans in enumerate(batch_true_answers):
            idx = tokenizer.convert_tokens_to_ids(true_ans)
            log_probs = torch.log_softmax(next_token_logits[j], dim=0)
            nll = -log_probs[idx].item()
            all_nlls.append(nll)

        pred_indices = torch.argmax(next_token_logits, dim=1).cpu().tolist()
        pred_letters = [tokenizer.decode(idx) for idx in pred_indices]
        preds.extend(pred_letters)
        all_logits.extend(next_token_logits.cpu().tolist())

        abcd_probs = probs[:, abcd_token_ids]      # Shape: [batch_size, 4]
        all_abcd_probs.extend(abcd_probs.cpu().tolist())


# 3. Post-Processing & ACC, NLL calculation
vocab_size = model.lm_head.weight.shape[0]
logits_tensor = torch.tensor(all_logits)  # shape: [num_samples, vocab_size]
targets_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in true_answers])
accuracy = sum([p == t for p, t in zip(preds, true_answers)]) / len(true_answers)
print(f"Accuracy: {accuracy:.4f}")
nll_avg = sum(all_nlls) / len(all_nlls)
nll_std = torch.tensor(all_nlls).std(unbiased=False).item()
print(f"Average NLL: {nll_avg:.4f}")
print(f"Std Deviation NLL: {nll_std:.4f}")
wandb.log({"acc": accuracy,
           "nll_avg": nll_avg,
           "nll_std": nll_std})


# 4. Sweep the ECE config and store them all
for n_bins in [10, 15, 20]:
    ece_metric = MulticlassCalibrationError(num_classes=vocab_size, n_bins=n_bins, norm='l1')
    ece = ece_metric(logits_tensor, targets_tensor)
    print(f"ECE-{n_bins}: {ece:.4f}")
    wandb.log({f"ece-{n_bins}": ece.item()})

# 5. ENT calculation
ent_avg = sum(all_entropies) / len(all_entropies)
ent_std = torch.tensor(all_entropies).std(unbiased=False).item()
print(f"Average Entropy: {ent_avg:.4f}")
wandb.log({"ent_avg": ent_avg, "ent_std": ent_std})

# 6. Dump per datapoint information
output_data_for_json = [
    {
        "id": id,
        "label": t,
        "pred": p,
        "abcd_probs": abcd_probs,
        "entropy": e
    } for id, t, p, abcd_probs, e in zip(ids, true_answers, preds, all_abcd_probs, all_entropies)
]

os.makedirs("results", exist_ok=True)
output_filename = f"results/{exp_name}.json"
try:
    with open(output_filename, 'w') as f:
        json.dump(output_data_for_json, f, indent=4)
    print(f"Output data saved to {output_filename}")
except Exception as e:
    print(f"Error saving JSON output: {e}")

run.finish()


###############################################################################################################################