import os
import json
import datasets
import hashlib
from datasets import Dataset
from typing import List, Tuple
from transformers import AutoTokenizer
import random
from .prompt import multiple_choice_prompt_engineer, generation_prompt_engineer

# Datasets whose fine-tuning target is FREE-TEXT GENERATION (not a single MCQA
# option letter). These route through generation_prompt_engineer, get an EOS
# appended to the target, and use a label-preserving (Seq2Seq) collator so the
# prompt is masked out of the loss. Everything else stays multiple-choice.
GENERATION_DATASETS = {"medexqa"}


def is_generation_dataset(shortcodes):
    """True if any requested dataset is an open-generation dataset."""
    return any(s in GENERATION_DATASETS for s in shortcodes)

MMLU_SHORTCODE2NAME = {
    # ID
    'hs_us_his': 'high_school_us_history',
    'hs_gp': 'high_school_government_and_politics',
    'hs_psy': 'high_school_psychology',
    'soc': 'sociology',
    # Small OoD
    'jp': 'jurisprudence',
    'phi': 'philosophy',
    'pro_law': 'professional_law',
    # Big OoD
    "abs_alg": "abstract_algebra",
    "cs": "college_computer_science",
    "med_gen": "medical_genetics",
}

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
    'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_reasoning', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions'
]

def load_classification_dataset(dataset_shortcode, seed=42, split=None):
    """
    Load and reformat a multiple-choice question answering dataset.
    Now supports MMLU by numerical category index (e.g., 'mmlu_5').
    
    Args:
        dataset_name (str): The name of the dataset to load.
        seed (int): Random seed for splits if applicable.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets
               in the format [{"id": str, "question": str, "answer": str}].
    """
    
    if "arc" in dataset_shortcode:
        # Handles "arc_easy" and "arc_challenge"
        config_name = "ARC-Challenge" if "challenge" in dataset_shortcode else "ARC-Easy"
        dataset = datasets.load_dataset("ai2_arc", config_name)
        
        def reformat(example):
            # Handles cases where labels might not be A,B,C,D
            valid_labels = [str(i) for i in range(1, 5)] + ["A", "B", "C", "D"]
            if not all(lbl in valid_labels for lbl in example["choices"]["label"]):
                 return None
            
            answer_map = {label: choice for label, choice in zip(example["choices"]["label"], example["choices"]["text"])}
            answer_text = answer_map.get(example["answerKey"])

            if answer_text is None: return None # Skip if answerKey is not in labels

            return {
                'question': f"Question: {example['question']}\nChoices:\n" 
                            + "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])]) + "\nAnswer:",
                'answer': example["answerKey"],
                'id': example["id"],
            }
        
        train_dataset = [reformatted for example in dataset["train"] if (reformatted := reformat(example)) is not None]
        validation_dataset = [reformatted for example in dataset["validation"] if (reformatted := reformat(example)) is not None]
        test_dataset = [reformatted for example in dataset["test"] if (reformatted := reformat(example)) is not None]

    elif dataset_shortcode == "medmcqa":
        # Load MedMCQA dataset
        dataset = datasets.load_dataset("openlifescienceai/medmcqa")
        
        def reformat(example):
            labels = ["A", "B", "C", "D"]
            # Check if cop is valid index
            if example["cop"] is None or not (0 <= example["cop"] < 4):
                return None
            
            if example["choice_type"] != "single":
                return None
                
            choices = [example["opa"], example["opb"], example["opc"], example["opd"]]
            answer_key = labels[example["cop"]]
            
            return {
                'question': f"Question: {example['question']}\nChoices:\n" 
                            + "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)]) + "\nAnswer:",
                'answer': answer_key,
                'id': example["id"],
            }
        
        train_dataset = [reformatted for example in dataset["train"] if (reformatted := reformat(example)) is not None]
        validation_dataset = [reformatted for example in dataset["validation"] if (reformatted := reformat(example)) is not None]
        test_dataset = validation_dataset # MedMCQA doesn't have a standard test set, use validation

    elif dataset_shortcode == "openbookqa":
        dataset = datasets.load_dataset("openbookqa", "main")
        
        def reformat(example):
            return {
                'question': f"Question: {example['question_stem']}\nChoices:\n" 
                            + "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])]) + "\nAnswer:",
                'answer': example["answerKey"],
                'id': example["id"],
            }
            
        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        test_dataset = [reformat(d) for d in dataset["test"]]

    elif "winogrande" in dataset_shortcode: 
        dataset = datasets.load_dataset("winogrande", dataset_shortcode)
        
        def reformat(example):
            return {
                'question': f"Question: {example['sentence']}\nChoices:\n" 
                            + "\n".join([f"{label}. {choice}" for label, choice in zip(['1','2'], [example['option1'], example['option2']])]),
                'answer': example["answer"],
                'id': example.get("id", ""), # Add id if it exists
            }

        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        test_dataset = [reformat(d) for d in dataset["test"]]

    elif dataset_shortcode == "boolq":
        dataset = datasets.load_dataset("boolq")
        def reformat(example):
            # Reformat for a multiple-choice style
            question = f"Context: {example['passage']}\nQuestion: {example['question']}?\nChoices:\nA. True\nB. False\nAnswer:"
            answer_key = "A" if example["answer"] else "B"
            return {
                'question': question,
                'answer': answer_key
            }

        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        test_dataset = validation_dataset

    elif dataset_shortcode in MMLU_SHORTCODE2NAME:
        global MMLU_SUBJECTS
        subject_name = MMLU_SHORTCODE2NAME[dataset_shortcode]
        print(f"Loading MMLU subject: {subject_name}")
        dataset = datasets.load_dataset("cais/mmlu", subject_name)
        def reformat(example):
            labels = ["A", "B", "C", "D"]
            answer_key = labels[example["answer"]]
            return {
                'question': f"Question: {example['question']}\nChoices:\n" 
                            + "\n".join([f"{label}. {choice}" for label, choice in zip(labels, example['choices'])]) + "\nAnswer:",
                'answer': answer_key,
                'id': f"mmlu_{subject_name}_{example.get('id', 'no_id')}"
            }
        dev_split = [reformat(d) for d in dataset["dev"]]
        val_split = [reformat(d) for d in dataset["validation"]]
        test_split = [reformat(d) for d in dataset["test"]]
        random.seed(seed)
        random.shuffle(dev_split)
        random.shuffle(val_split)
        random.shuffle(test_split)
        # print(f"Loaded {len(dev_split)} dev, {len(val_split)} val, and {len(test_split)} test samples for MMLU subject '{dataset_shortcode}'.")
        # Construct train, validation, and test datasets
        # train & val
        if dataset_shortcode in ['hs_us_his', 'hs_gp', 'hs_psy', 'soc']:
            train_size = 200
            if dataset_shortcode == 'hs_psy':
                train_size = 400
            train_dataset = dev_split + val_split + test_split[-(train_size-len(dev_split)-len(val_split)):]
            validation_dataset = val_split
        else:
            train_dataset, validation_dataset = [], []
        # test
        thresholds = [500, 300, 200, 100]
        for threshold in thresholds:
            if len(test_split) >= threshold:
                test_dataset = test_split[:threshold]
                break
        if not test_dataset:
            test_dataset = test_split
    else:
        raise ValueError(f"Dataset '{dataset_shortcode}' not supported by load_classification_dataset.")
    
    print(f"Dataset '{dataset_shortcode}' loaded.")
    if test_dataset:
        print(f"Test dataset example: {test_dataset[0]}")
        
    if split == "train":
        return train_dataset
    elif split == "validation" or split == "val":
        return validation_dataset
    elif split == "test":
        return test_dataset
    
    # print(f"{dataset_shortcode}: Train samples: {len(train_dataset)}, Validation samples: {len(validation_dataset)}, Test samples: {len(test_dataset)}")
    return train_dataset, validation_dataset, test_dataset

def load_generation_dataset(dataset_name, seed=42):
    """
    Load dataset for generation tasks.
    Input: dataset_name;
    Output: train_dataset, validation_dataset;
    """

    train_dataset, validation_dataset = None, None

    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        dataset = datasets.load_dataset("albusli/bioasq-training11b")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    else:
        raise ValueError

    return train_dataset, validation_dataset

def batchify(data, batch_size, tokenizer, device="cuda:0"):
    """
    Create batches of data and handle padding.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        # Extract input texts and answers
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        ids = [item["id"] for item in batch]

        # Tokenize and pad
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,  # Pad to the longest sequence in the batch, CRUCIAL
            truncation=True
        ).to(device)

        yield inputs, answers, ids

def preprocess_mask_question_for_training(
    dataset_list: List[dict],
    tokenizer: AutoTokenizer,
    append_eos: bool = False,
    pad: bool = True,
) -> Dataset:
    """Tokenizes and masks the dataset for supervised fine-tuning.

    Two modes:
      * MCQA (append_eos=False, pad=True): unchanged legacy behaviour -- pads to
        the global longest sequence, masks the prompt prefix. NOTE: the MCQA
        training scripts now use `preprocess_answer_only_for_training` (exact
        answer-only labels) via `answer_only=True`; this legacy path is kept for
        the other router-tuning scripts that still call it.
      * GENERATION (append_eos=True, pad=False): appends EOS to the target so the
        model learns to stop, tokenizes WITHOUT re-adding special tokens (the
        chat template already injected them, so prompt-length masking is exact),
        masks the prompt prefix AND any padding to IGNORE_INDEX, and leaves rows
        un-padded for a dynamic label-preserving (Seq2Seq) collator.
    """
    IGNORE_INDEX = -100
    prompts = [item['question'] for item in dataset_list]

    if append_eos:
        eos = tokenizer.eos_token or ""
        full_texts = [item['question'] + item['answer'] + eos for item in dataset_list]
        model_inputs = tokenizer(
            full_texts, padding=("longest" if pad else False),
            truncation=False, add_special_tokens=False,
        )
        prompt_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    else:
        full_texts = [item['question'] + item['answer'] for item in dataset_list]
        # Tokenize the full texts to get input_ids
        model_inputs = tokenizer(full_texts, padding="longest", truncation=False)
        # Tokenize prompts separately to find their lengths for masking
        # We don't add special tokens here because we only care about the length of the prompt text itself
        prompt_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]

    attn = model_inputs.get("attention_mask")
    labels_list = []
    for i, input_id_row in enumerate(model_inputs['input_ids']):
        prompt_len = prompt_lengths[i]
        # The label is a copy of the input_ids
        label_row = list(input_id_row)

        # Mask the prompt part by setting it to IGNORE_INDEX
        # The first token is often BOS, which should also be masked.
        # We mask up to the length of the tokenized prompt.
        for j in range(min(prompt_len, len(label_row))):
            label_row[j] = IGNORE_INDEX
        # Generation path: also mask padding positions so pad tokens never
        # contribute to the loss (matters once answers vary in length).
        if append_eos and attn is not None:
            for j, m in enumerate(attn[i]):
                if m == 0:
                    label_row[j] = IGNORE_INDEX
        labels_list.append(label_row)

    model_inputs["labels"] = labels_list
    return Dataset.from_dict(model_inputs)

def preprocess_answer_only_for_training(dataset_list: List[dict], tokenizer: AutoTokenizer) -> Dataset:
    """Tokenizes for supervised fine-tuning with loss on the answer tokens only.

    Prompt and answer are tokenized separately and concatenated, so the prompt
    mask cannot be knocked out of alignment by padding side, model-specific
    special tokens, or BPE merges at the prompt/answer boundary. Rows are left
    unpadded; use a labels-aware collator (e.g. DataCollatorForSeq2Seq with
    label_pad_token_id=-100) to pad per batch.
    """
    IGNORE_INDEX = -100
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for item in dataset_list:
        prompt_ids = tokenizer(item['question']).input_ids
        answer_ids = tokenizer(item['answer'], add_special_tokens=False).input_ids
        input_ids_list.append(prompt_ids + answer_ids)
        attention_mask_list.append([1] * (len(prompt_ids) + len(answer_ids)))
        labels_list.append([IGNORE_INDEX] * len(prompt_ids) + answer_ids)
    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })

def load_and_prepare_train_and_val_data(tokenizer: AutoTokenizer, train_dataset_shortcodes: List, seed=42, answer_only=False) -> Tuple[Dataset, Dataset]:
    """Loads and preprocesses the dataset for causal language modeling.

    Dispatch (three mutually exclusive paths):
      * GENERATION datasets (see GENERATION_DATASETS): generation prompt template,
        `preprocess_mask_question_for_training(append_eos=True, pad=False)` --
        prompt-masked labels + EOS on the explanation, rows un-padded. This is
        the exact recipe that produced the existing MedExQA adapters/weights;
        `answer_only` is irrelevant here (the loss is already target-only).
      * MCQA + `answer_only=True`: `preprocess_answer_only_for_training` (prompt
        and answer tokenized separately -> exact answer-only labels).
      * MCQA + `answer_only=False`: legacy full-sequence-padded masking.
    All paths return prompt-masked labels; pair with a labels-preserving collator
    (DataCollatorForSeq2Seq, label_pad_token_id=-100).
    """
    generation = is_generation_dataset(train_dataset_shortcodes)
    engineer = generation_prompt_engineer if generation else multiple_choice_prompt_engineer

    train_raw, val_raw = [], []
    for dataset_shortcode in train_dataset_shortcodes:
        train_raw_curr, val_raw_curr, _ = load_exp_dataset(dataset_shortcode, seed=seed)
        train_raw.extend(train_raw_curr)
        val_raw.extend(val_raw_curr)

    train_engineered = [engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [engineer(x, tokenizer=tokenizer) for x in val_raw]

    if generation:
        # Generation: append EOS + keep rows un-padded (dynamic Seq2Seq collator).
        train_dataset = preprocess_mask_question_for_training(
            train_engineered, tokenizer, append_eos=True, pad=False)
        val_dataset = preprocess_mask_question_for_training(
            val_engineered, tokenizer, append_eos=True, pad=False)
    else:
        preprocess = preprocess_answer_only_for_training if answer_only else preprocess_mask_question_for_training
        train_dataset = preprocess(train_engineered, tokenizer)
        val_dataset = preprocess(val_engineered, tokenizer)

    print(f"Datasets '{train_dataset_shortcodes}' loaded and preprocessed.")
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(val_dataset)}")

    return train_dataset, val_dataset

def medexqa_split_manifest_path(seed=42):
    """Repo-relative path of the frozen MedExQA derived-split manifest."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "splits", f"medexqa-derived-seed{seed}.csv")


def medexqa_example_hash(example):
    """Content hash identifying a MedExQA example independently of the random id."""
    return hashlib.sha1(example["question"].encode("utf-8")).hexdigest()[:16]


def write_medexqa_split_manifest(train, val, test, path):
    """Freeze the derived split: one row per example (qhash, split, id, question_prefix)."""
    import csv as _csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["qhash", "split", "id", "question_prefix"])
        for split_name, rows in (("train", train), ("val", val), ("test", test)):
            for ex in rows:
                prefix = " ".join(ex["question"].split())[:80]
                w.writerow([medexqa_example_hash(ex), split_name, ex["id"], prefix])
    print(f"  MedExQA: wrote split manifest -> {path} "
          f"(train={len(train)}, val={len(val)}, test={len(test)})")


def verify_medexqa_split_manifest(train, val, test, seed=42, path=None):
    """Check the in-memory derived split against the frozen manifest.

    * manifest present  -> every example must be in the recorded split, and the
      per-split counts must match; otherwise raise (the split moved: RNG order,
      dedup, or upstream TSVs changed). Set MEDEXQA_SKIP_MANIFEST=1 to bypass.
    * manifest absent   -> print how to create it (write-medexqa-split-manifest.py).
    """
    import csv as _csv
    path = path or medexqa_split_manifest_path(seed)
    if os.environ.get("MEDEXQA_SKIP_MANIFEST") == "1":
        print("  MedExQA: split-manifest check SKIPPED (MEDEXQA_SKIP_MANIFEST=1)")
        return
    if not os.path.exists(path):
        print(f"  MedExQA: no split manifest at {path}; run "
              f"`python write-medexqa-split-manifest.py` once and commit it to freeze the split.")
        return
    recorded = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            recorded[row["qhash"]] = row["split"]
    mismatches, counts = [], {"train": 0, "val": 0, "test": 0}
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        for ex in rows:
            counts[split_name] += 1
            got = recorded.get(medexqa_example_hash(ex))
            if got != split_name:
                mismatches.append((split_name, got, ex["question"][:60]))
    rec_counts = {s: sum(1 for v in recorded.values() if v == s) for s in counts}
    if mismatches or rec_counts != counts:
        head = "\n".join(f"    in-memory={a} manifest={b}: {q!r}" for a, b, q in mismatches[:5])
        raise RuntimeError(
            f"MedExQA derived split does not match the frozen manifest {path}: "
            f"{len(mismatches)} misassigned example(s); counts in-memory={counts} manifest={rec_counts}.\n"
            f"{head}\nThe split moved (RNG order / dedup / upstream TSV change). Do NOT proceed with "
            f"val/test-dependent runs; investigate, or set MEDEXQA_SKIP_MANIFEST=1 to bypass knowingly."
        )
    print(f"  MedExQA: split matches frozen manifest ({path}); "
          f"train={counts['train']} val={counts['val']} test={counts['test']}")


def load_exp_dataset(dataset_shortcode, seed=42, split=None):
    """
    Loads and processes one of the six specified experimental datasets with
    custom train/validation/test splits for fine-tuning and evaluation.

    Args:
        dataset_shortcode (str): One of 'obqa', 'arc_c', 'arc_e', 
                                 'sciq', 'mmlu_law', 'medmcqa_med'.
        seed (int): Random seed for shuffling and sampling.
        split (str, optional): If specified, returns only the 'train', 
                               'validation', or 'test' split. 
                               Otherwise, returns all three.

    Returns:
        tuple or list: A tuple containing the (train, validation, test) datasets,
                       or a single list for the specified split. Each dataset is
                       a list of dictionaries.
    """
    
    random.seed(seed)

    def reformat_arc(example):
        labels = example['choices']['label']
        if labels != ['A', 'B', 'C', 'D']: return None
        question = example.get('question')
        choices = example['choices']['text']
        answer_key = example['answerKey']
        if answer_key not in labels: return None
        return {
            'question': f"Question: {question}\nChoices:\n" + "\n".join([f"{label}. {text}" for label, text in zip(labels, choices)]) + "\nAnswer:",
            'answer': answer_key, 
            'id': example["id"],
        }
    
    def reformat_obqa(example):
        labels = example['choices']['label']
        question = example.get('question_stem')
        choices = example['choices']['text']
        answer_key = example['answerKey']
        if answer_key not in labels: return None
        return {
            'question': f"Question: {question}\nChoices:\n" + "\n".join([f"{label}. {text}" for label, text in zip(labels, choices)]) + "\nAnswer:",
            'answer': answer_key, 
            'id': example["id"],
        }

    def reformat_sciq(example):
        question = example['question']
        choices = [example['distractor1'], example['distractor2'], example['distractor3'], example['correct_answer']]
        random.shuffle(choices)
        labels = ["A", "B", "C", "D"]
        answer_key = labels[choices.index(example['correct_answer'])]
        return {
            'question': f"Question: {question}\nChoices:\n" + "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)]) + "\nAnswer:",
            'answer': answer_key, 
            'id': f"sciq_{random.randint(1000, 9999)}",
        }

    def reformat_mmlu(example):
        labels = ["A", "B", "C", "D"]
        answer_key = labels[example["answer"]]
        return {
            'question': f"Question: {example['question']}\nChoices:\n" + "\n".join([f"{label}. {choice}" for label, choice in zip(labels, example['choices'])]) + "\nAnswer:",
            'answer': answer_key, 
            'id': f"mmlu_law_{random.randint(1000, 9999)}"
        }

    def reformat_medmcqa(example):
        if example.get("cop") is None or not (0 <= example["cop"] < 4) or example.get("choice_type") != "single":
            return None
        labels = ["A", "B", "C", "D"]
        choices = [example.get("opa"), example.get("opb"), example.get("opc"), example.get("opd")]
        if any(c is None for c in choices): return None
        answer_key = labels[example["cop"]]
        return {
            'question': f"Question: {example['question']}\nChoices:\n" + "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)]) + "\nAnswer:",
            'answer': answer_key, 'id': example["id"],
        }

    # --- Dataset Loading and Splitting Logic ---
    train_dataset, validation_dataset, test_dataset = [], [], []

    if dataset_shortcode == "obqa":
        # 5000 train, 500 test
        dataset = datasets.load_dataset("openbookqa", "main")
        train_pool = [reformat_obqa(ex) for ex in dataset["train"]] + [reformat_obqa(ex) for ex in dataset["validation"]]
        train_dataset = [ex for ex in train_pool if ex is not None][:5050]
        test_dataset = [reformat_obqa(ex) for ex in dataset["test"] if ex is not None]

    elif dataset_shortcode == "arc_c":
        dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
        test_pool = [reformatted for ex in dataset["test"] if (reformatted := reformat_arc(ex)) is not None]
        random.shuffle(test_pool)
        test_dataset = test_pool[:500]
        extra_train_from_test = test_pool[500:]
        train_pool = [reformatted for ex in dataset["train"] if (reformatted := reformat_arc(ex)) is not None] + \
                     [reformatted for ex in dataset["validation"] if (reformatted := reformat_arc(ex)) is not None]
        train_dataset = train_pool + extra_train_from_test

    elif dataset_shortcode == "arc_e":
        dataset = datasets.load_dataset("ai2_arc", "ARC-Easy")
        test_pool = [reformatted for ex in dataset["test"] if (reformatted := reformat_arc(ex)) is not None]
        random.shuffle(test_pool)
        test_dataset = test_pool[:500]
        train_pool = [reformatted for ex in dataset["train"] if (reformatted := reformat_arc(ex)) is not None] + \
                     [reformatted for ex in dataset["validation"] if (reformatted := reformat_arc(ex)) is not None]
        train_dataset = train_pool

    elif dataset_shortcode == "sciq":
        dataset = datasets.load_dataset("sciq")
        train_pool = [reformat_sciq(ex) for ex in dataset["train"]] + [reformat_sciq(ex) for ex in dataset["validation"]]
        train_dataset = [ex for ex in train_pool if ex is not None][:5050]
        test_dataset = [reformat_sciq(ex) for ex in dataset["test"] if ex is not None][:500]

    elif dataset_shortcode == "mmlu_law":
        dataset = datasets.load_dataset("cais/mmlu", "professional_law")
        full_pool = [reformat_mmlu(ex) for ex in dataset["validation"]] + [reformat_mmlu(ex) for ex in dataset["test"]]
        full_pool = [ex for ex in full_pool if ex is not None]
        random.shuffle(full_pool)
        test_dataset = full_pool[:500]
        train_dataset = full_pool[500:]

    elif dataset_shortcode == "medmcqa_med":
        dataset = datasets.load_dataset("medmcqa")
        # Filter for "Medicine" subject and combine train/val splits
        medicine_pool = []
        for ex in dataset["train"]:
            if ex.get('subject_name') == 'Medicine':
                reformatted = reformat_medmcqa(ex)
                if reformatted: medicine_pool.append(reformatted)
        for ex in dataset["validation"]:
            if ex.get('subject_name') == 'Medicine':
                reformatted = reformat_medmcqa(ex)
                if reformatted: medicine_pool.append(reformatted)
        
        random.shuffle(medicine_pool)
        
        # Create train and test sets from the filtered pool
        if len(medicine_pool) < 10500:
            raise ValueError(f"Not enough 'Medicine' samples in MedMCQA. Found {len(medicine_pool)}, need 10500.")
        
        train_dataset = medicine_pool[500:5550]
        test_dataset = medicine_pool[:500]

    elif dataset_shortcode == "medexqa":
        # MedExQA (bluesky333/MedExQA): an open-generation medical benchmark, NOT
        # a train corpus. 5 specialty configs, splits dev(5/specialty)+test(940).
        # The TSVs are HEADER-LESS with columns:
        #   [0] Question  [1..4] Choice A-D  [5] Explanation 1  [6] Explanation 2
        #   [7] Correct Answer
        # We download the raw TSVs directly (hf_hub_download) and parse them with
        # csv.reader(delimiter='\t') -- load_dataset()'s CSV builder assumes a
        # header row and a comma sep, which mangles these files (0 rows). We
        # fine-tune OPEN GENERATION with target = the free-text 'Explanation 1'.
        # No native train split -> pool every row across all specialties/splits,
        # shuffle (seed), carve a held-out test set; the generic tail below then
        # carves 50 val from the remainder.
        import csv as _csv
        from huggingface_hub import hf_hub_download

        repo = "bluesky333/MedExQA"
        specialties = [
            "biomedical_engineer", "clinical_laboratory_scientist",
            "clinical_psychologist", "occupational_therapist", "speech_pathologist",
        ]
        rel_paths = []
        for sp in specialties:
            rel_paths.append(f"dev/{sp}_dev.tsv")
            rel_paths.append(f"test/{sp}_test.tsv")

        def reformat_medexqa(cols):
            # Header-less row: at least [question, A, B, C, D, explanation1].
            # Cols 6/7 (Explanation 2, correct answer letter) are now kept when
            # present: the letter gives a cheap per-example correctness label
            # for the abstention readout, and the second explanation gives a
            # two-reference quality target. Training consumes only
            # question/answer, so these extra fields change nothing upstream.
            # NOTE: this function must consume the global RNG exactly once per
            # row (the id) -- the shuffle below depends on it, so adding RNG
            # calls here would silently change the train/val/test split.
            if len(cols) < 6:
                return None
            q = str(cols[0]).strip()
            choices = [str(c).strip() for c in cols[1:5]]
            e1 = str(cols[5]).strip()
            if not q or not e1:
                return None
            e2 = str(cols[6]).strip() if len(cols) > 6 else ""
            gold = str(cols[7]).strip().upper() if len(cols) > 7 else ""
            if gold not in {"A", "B", "C", "D"}:
                gold = ""
            opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(["A", "B", "C", "D"], choices) if txt)
            return {
                "question": f"Question: {q}\nOptions:\n{opts}\n\nExplain the reasoning for the correct answer.",
                # Free-text generation target = the gold explanation. Leading
                # space for a clean sub-word split at the prompt/answer boundary.
                "answer": " " + e1,
                "id": f"medexqa_{random.randint(100000, 999999)}",
                "explanation_2": e2,
                "gold_letter": gold,
                # MCQA-style prompt for the letter-probe correctness label.
                "letter_question": f"Question: {q}\nChoices:\n{opts}\nAnswer:",
            }

        pool = []
        for rel in rel_paths:
            try:
                local = hf_hub_download(repo_id=repo, filename=rel, repo_type="dataset")
            except Exception as e:
                print(f"  (MedExQA: could not fetch {rel} -- {e!r})")
                continue
            with open(local, newline="", encoding="utf-8") as f:
                for row in _csv.reader(f, delimiter="\t"):
                    if not row:
                        continue
                    ex = reformat_medexqa(row)
                    if ex is not None:
                        pool.append(ex)

        # Dedup defensively (the same question shouldn't appear twice).
        seen, deduped = set(), []
        for ex in pool:
            key = (ex["question"], ex["answer"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ex)
        pool = deduped
        random.shuffle(pool)

        if len(pool) < 200:
            raise ValueError(
                f"MedExQA: expected ~965 rows but resolved {len(pool)}. "
                "Check that the TSVs downloaded and are tab-separated (see reformat_medexqa)."
            )
        print(f"  MedExQA: resolved {len(pool)} rows from {len(rel_paths)} TSVs.")

        # DERIVED SPLIT (not the official MedExQA benchmark split): the official
        # release has 25 dev + 940 test examples; we pool both (965), seed-shuffle
        # (seed above), and carve 175 test / 50 val / ~740 train. The assignment is
        # frozen in splits/medexqa-derived-seed<seed>.csv (see
        # verify_medexqa_split_manifest below) so it can be cited and checked.
        test_dataset = pool[:175]
        train_dataset = pool[175:]   # generic tail carves 50 val from this -> ~740 train

    else:
        raise ValueError(f"Dataset '{dataset_shortcode}' not supported by load_exp_dataset.")

    validation_dataset = train_dataset[-50:]
    train_dataset = train_dataset[:-50]

    if dataset_shortcode == "medexqa":
        verify_medexqa_split_manifest(train_dataset, validation_dataset, test_dataset, seed=seed)

    print(f"Dataset '{dataset_shortcode}' processed: Train={len(train_dataset)}, Val={len(validation_dataset)}, Test={len(test_dataset)}")
    
    if split == "train":
        return train_dataset
    elif split == "validation" or split == "val":
        return validation_dataset
    elif split == "test":
        return test_dataset
    
    return train_dataset, validation_dataset, test_dataset