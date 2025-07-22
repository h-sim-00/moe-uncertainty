import argparse
import os
import wandb
from huggingface_hub import login as hf_login
import torch
from transformers import AutoTokenizer
from data import multiple_choice_prompt_engineer, load_classification_dataset, batchify
from model import load_model
import json
from tqdm import tqdm
from torchmetrics.classification import MulticlassCalibrationError

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline script with argument parsing.")
    parser.add_argument("--hf_home", type=str, default="/vol/bitbucket/al1624/.cache/huggingface", help="HuggingFace cache home directory")
    parser.add_argument("--hf_datasets_cache", type=str, default="/vol/bitbucket/al1624/.cache/huggingface/datasets", help="HuggingFace datasets cache directory")
    parser.add_argument("--wandb_key", type=str, default="8d44174f1416d56dc5470b57deb50339b19f22e7", help="Weights & Biases API key")
    parser.add_argument("--hf_token", type=str, default="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", help="HuggingFace API token")
    parser.add_argument("--dataset_name", type=str, default="arc_easy", help="Dataset name")
    parser.add_argument("--model_id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct", help="Model ID")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    # parser.add_argument("--max_new_tokens", type=int, default=10, help="Max new tokens")
    return parser.parse_args()

def main():
    args = parse_args()
    # 1. Initialise HuggingFace and Weights & Biases
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    hf_login(token=args.hf_token)
    wandb.login(key=args.wandb_key)
    run = wandb.init(
        project="exp1-baseline",
        name=f"{args.model_id}_{args.dataset_name}",
        config={
            "model_id": args.model_id,
            "dataset_name": args.dataset_name,
            "args.batch_size": args.batch_size,
            # "max_new_tokens": args.max_new_tokens
        }
    )

    # 2. Initialise model and tokeniser
    # You can now use args.dataset_name, args.model_id, etc.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = load_model(args.model_id)

    # 3. Load dataset (arc_easy)
    _, val_dataset = load_classification_dataset(args.dataset_name)
    val_dataset = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_dataset]

    # 4. Inference
    ## 4-1. Set up containers before inference
    questions = [x['question'] for x in val_dataset]
    true_answers = [x['answer'] for x in val_dataset]
    ids = [x.get('id', i) for i, x in enumerate(val_dataset)]
    preds = []
    all_logits = []
    all_nlls = []
    

    ## 4-2. Inference loop
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size)):
            batch_q = questions[i:i+args.batch_size]
            batch_ids = ids[i:i+args.batch_size]
            batch_true_answers = true_answers[i:i+args.batch_size]

            # Infernence and get last-token logits
            inputs = tokenizer(
                batch_q,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            logits = model(**inputs).logits
            assert not torch.isnan(logits).any(), "NaN detected in logits tensor"
            next_token_logits = logits[:, -1, :] # token_num * 1 * vocab_size

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
            
     
    # 5. Compute global metrics
    accuracy = sum([p == t for p, t in zip(preds, true_answers)]) / len(true_answers)
    print(f"Accuracy: {accuracy:.4f}")
    wandb.log({"accuracy": accuracy})

    nll_avg = sum(all_nlls) / len(all_nlls)
    print(f"Average NLL: {nll_avg:.4f}")
    wandb.log({"average_nll": nll_avg})

    # 6. Save id-predict-answer and logits to local folder
    save_dir = f"./results/{args.model_id.replace('/', '_')}_{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    results = [
        {
            "id": id_,
            "question": q,
            "true_answer": t,
            "pred": p,
            "logits": l
        }
        for id_, q, t, p, l in zip(ids, questions, true_answers, preds, all_logits)
    ]
    with open(os.path.join(save_dir, "predictions.jsonl"), "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    ###############################################################################################
    # Compute calibration error (ECE) over vocab_size classes
    vocab_size = tokenizer.vocab_size
    ece_metric = MulticlassCalibrationError(num_classes=vocab_size, n_bins=15, norm='l1')

    # Prepare predictions and targets for torchmetrics
    # Map each predicted token and true answer to its token id
    pred_token_ids = [tokenizer.convert_tokens_to_ids(p) for p in preds]
    target_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in true_answers]

    # Convert logits to tensor
    logits_tensor = torch.tensor(all_logits)  # shape: [num_samples, vocab_size]
    targets_tensor = torch.tensor(target_token_ids)

    ece = ece_metric(logits_tensor, targets_tensor)
    print(f"ECE: {ece:.4f}")
    wandb.log({"ece": ece.item()})
    ################################################################################################

    wandb.save(os.path.join(save_dir, "predictions.jsonl"))
    run.finish()




if __name__ == "__main__":
    main()