"""Sanity check for answer-only training labels (run on quail-1 before training).

Builds the granite/obqa loaders exactly as kvq-tuning.py / fcvr-tuning.py now do
(answer_only=True + right padding + DataCollatorForSeq2Seq) and verifies that
the only positions contributing to the loss are the answer tokens:
  1. every row has at least one label != -100
  2. pad positions (attention_mask == 0) all have label -100
  3. every unmasked label equals the input_id at the same position
  4. the decoded unmasked tokens are exactly the example's answer letter
     (checked against the prompt-engineered examples, in order, on the
     unshuffled val split)
Prints one decoded example showing the prompt/answer mask boundary.
"""
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from utils import setup_environment
from model import load_tokenizer
from utils.data import load_exp_dataset, preprocess_answer_only_for_training
from utils.prompt import multiple_choice_prompt_engineer

MODEL_SHORTCODE = "granite"
DATASET_SHORTCODE = "obqa"
BATCH_SIZE = 8
NUM_BATCHES = 4  # how many val batches to check

def main():
    setup_environment()
    tokenizer = load_tokenizer(MODEL_SHORTCODE)

    _, val_raw, _ = load_exp_dataset(DATASET_SHORTCODE, seed=42)
    val_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_raw]
    val_dataset = preprocess_answer_only_for_training(val_engineered, tokenizer)

    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    n_checked, n_answer_tokens = 0, 0
    printed_example = False
    for b, batch in enumerate(val_loader):
        if b >= NUM_BATCHES:
            break
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        for r in range(input_ids.shape[0]):
            row_idx = b * BATCH_SIZE + r
            unmasked = (labels[r] != -100).nonzero(as_tuple=True)[0]
            assert len(unmasked) >= 1, f"row {row_idx}: no unmasked label positions"
            assert (labels[r][attention_mask[r] == 0] == -100).all(), \
                f"row {row_idx}: pad position with a real label"
            assert (labels[r][unmasked] == input_ids[r][unmasked]).all(), \
                f"row {row_idx}: unmasked labels differ from input_ids"
            decoded = tokenizer.decode(labels[r][unmasked])
            expected = val_engineered[row_idx]["answer"]
            assert decoded == expected, \
                f"row {row_idx}: loss tokens decode to {decoded!r}, expected {expected!r}"
            n_checked += 1
            n_answer_tokens += len(unmasked)

            if not printed_example:
                prompt_part = tokenizer.decode(input_ids[r][: unmasked[0]])
                print("=== example row 0 ===")
                print(f"prompt (masked, {unmasked[0].item()} tokens): ...{prompt_part[-200:]}")
                print(f"answer (in loss, {len(unmasked)} token(s)): {decoded!r}")
                print(f"padding: {(attention_mask[r] == 0).sum().item()} tokens, all labeled -100")
                print("=====================")
                printed_example = True

    print(f"OK: {n_checked} rows checked, {n_answer_tokens} answer tokens in loss "
          f"({n_answer_tokens / n_checked:.2f} per row); prompt and padding fully masked.")

if __name__ == "__main__":
    main()
