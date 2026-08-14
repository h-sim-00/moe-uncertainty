python3 - <<'PY'
from model import load_tokenizer

tokenizer = load_tokenizer("granite")

print("padding_side:", tokenizer.padding_side)
print("pad_token:", repr(tokenizer.pad_token))
print("pad_token_id:", tokenizer.pad_token_id)

batch = tokenizer(
    ["This is a considerably longer example sentence.", "Short."],
    padding=True,
    return_tensors="pt",
)

print("input_ids:")
print(batch["input_ids"])
print("attention_mask:")
print(batch["attention_mask"])
PY