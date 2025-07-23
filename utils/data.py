import os
import json
import datasets
import hashlib
from datasets import Dataset
from typing import List, Tuple
from transformers import AutoTokenizer
import random
from .prompt import multiple_choice_prompt_engineer

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

def batchify(data, batch_size, tokenizer, device="auto"):
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

def preprocess_mask_question_for_training(dataset_list: List[dict], tokenizer: AutoTokenizer) -> Dataset:
    """Correctly tokenizes and masks the dataset for supervised fine-tuning."""
    IGNORE_INDEX = -100
    prompts = [item['question'] for item in dataset_list]
    full_texts = [item['question'] + item['answer'] for item in dataset_list]
    # Tokenize the full texts to get input_ids
    model_inputs = tokenizer(full_texts, padding="longest", truncation=False)
    # Tokenize prompts separately to find their lengths for masking
    # We don't add special tokens here because we only care about the length of the prompt text itself
    prompt_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    
    labels_list = []
    for i, input_id_row in enumerate(model_inputs['input_ids']):
        prompt_len = prompt_lengths[i]
        # The label is a copy of the input_ids
        label_row = list(input_id_row)
        
        # Mask the prompt part by setting it to IGNORE_INDEX
        # The first token is often BOS, which should also be masked.
        # We mask up to the length of the tokenized prompt.
        for j in range(prompt_len):
            label_row[j] = IGNORE_INDEX
        labels_list.append(label_row)
    
    model_inputs["labels"] = labels_list
    return Dataset.from_dict(model_inputs)

def load_and_prepare_train_and_val_data(tokenizer: AutoTokenizer, train_dataset_shortcodes: List, seed=42) -> Tuple[Dataset, Dataset]:
    """Loads and preprocesses the dataset for causal language modeling."""
    train_raw, val_raw = [], []
    for dataset_shortcode in train_dataset_shortcodes:
        train_raw_curr, val_raw_curr, _ = load_classification_dataset(dataset_shortcode, seed=seed)
        train_raw.extend(train_raw_curr)
        val_raw.extend(val_raw_curr)
    
    train_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in val_raw]
    
    train_dataset = preprocess_mask_question_for_training(train_engineered, tokenizer)
    val_dataset = preprocess_mask_question_for_training(val_engineered, tokenizer)

    print(f"Datasets '{train_dataset_shortcodes}' loaded and preprocessed.")
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(val_dataset)}")

    return train_dataset, val_dataset
