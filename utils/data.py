import os
import json
import datasets
import hashlib
import numbers
import re
from collections import Counter
from datasets import Dataset
from typing import List, Tuple
from transformers import AutoTokenizer
import random
from .prompt import (multiple_choice_prompt_engineer, generation_prompt_engineer, COMPARISON_SYSTEM_INSTRUCTION,
                     SYSTEM_INSTRUCTIONS)

# Datasets whose fine-tuning target is FREE-TEXT GENERATION (not a single MCQA
# option letter). These route through generation_prompt_engineer, get an EOS
# appended to the target, and use a label-preserving (Seq2Seq) collator so the
# prompt is masked out of the loss. Everything else stays multiple-choice.
GENERATION_DATASETS = {"medexqa", "medmcqa_gen", "obqa_gen"}

# ---------------------------------------------------------------------------
# target_mode (branch MedMCQA-comparison): what the fine-tuning TARGET of a
# generation dataset (medmcqa_gen) is. Selected by --target_mode on the three
# training scripts (add_target_mode_arg); "explanation" is the default and
# reproduces the existing behaviour bit-for-bit.
#   explanation          generation prompt -> " " + gold explanation + EOS
#                        (explanation-only loss; what every existing
#                        MedExQA/MedMCQA adapter and router was trained with)
#   letter               arm A: comparison prompt (`letter_question`, ends
#                        "Answer:", COMPARISON_SYSTEM_INSTRUCTION) -> bare gold
#                        letter, NO EOS -- the corrected answer-only recipe of
#                        branch exp4-train-ans
#   answer_explanation   arm B: the IDENTICAL prompt -> gold letter +
#                        "\nExplanation:" + " " + gold explanation + EOS; loss on
#                        letter AND explanation
# The prompt is the same in both arms and the letter is the FIRST target token,
# so only the target differs and the letter read-out is identical across arms.
# With max_seq_len, BOTH arms keep exactly the rows whose answer_explanation
# sequence fits (comparison_eligible_indices), so they train on the same rows.
# ---------------------------------------------------------------------------
TARGET_MODES = ("explanation", "letter", "answer_explanation")
COMPARISON_MODES = ("letter", "answer_explanation")
EXPLANATION_MARKER = "\nExplanation:"
# Artefact / result-file suffix of each comparison arm (adapters, router weights,
# evaluate_letter.py tags). The single place this mapping lives in python; the
# overnight driver mirrors it in bash (arm_sfx).
TARGET_MODE_SUFFIX = {"letter": "armA-letter", "answer_explanation": "armB-ansexp"}
TARGET_MODE_HELP = ("[generation datasets only] fine-tuning target: 'explanation' (default; explanation-only "
                    "loss, the existing recipe), 'letter' (MedMCQA-comparison arm A: comparison prompt -> gold "
                    "letter, answer-only loss), 'answer_explanation' (arm B: same prompt -> letter + "
                    "'\\nExplanation:' + explanation, loss on both).")


def add_target_mode_arg(parser, extra_help=""):
    """The one --target_mode argparse definition shared by every script."""
    parser.add_argument("--target_mode", type=str, default="explanation", choices=list(TARGET_MODES),
                        help=TARGET_MODE_HELP + (" " + extra_help if extra_help else ""))
    return parser


def add_system_prompt_arg(parser):
    """--system_prompt for the TRAINING side of the comparison arms (branch
    OBQA-qwen): which system instruction wraps `letter_question`. 'comparison'
    (default) reproduces every existing run; 'mcq' is the letter-only
    instruction (Qwen arm A is trained with it, mirroring how the Granite OBQA
    arm A was trained). Evaluation selects the same key via evaluate_letter.py
    --system_prompt / the ARM_SETUP registry."""
    parser.add_argument("--system_prompt", type=str, default="comparison", choices=sorted(SYSTEM_INSTRUCTIONS),
                        help="[comparison target modes] system instruction used to build the training prompt: "
                             "'comparison' (default, the shared letter+explanation instruction) or 'mcq' "
                             "(letter-only instruction). Ignored for target_mode=explanation.")
    return parser


def eligible_tag_for(model_shortcode):
    """Suffix that keys the frozen eligible-ID lists to the tokenizer. Granite
    (the model every existing list was written with) keeps the unsuffixed
    filenames; any other model gets its own list (a different tokenizer gives
    a different keep-list and would otherwise trip the frozen-list check)."""
    return None if model_shortcode == "granite" else model_shortcode


def loss_mode_label(dataset_shortcodes, target_mode="explanation"):
    """Human-readable description of which tokens carry the training loss
    (printed by the training scripts; mirrors the dispatch in
    load_and_prepare_train_and_val_data)."""
    if not is_generation_dataset(dataset_shortcodes):
        return "answer-only (MCQA)"
    return {
        "explanation": "explanation-only (generation)",
        "letter": "answer-only letter (target_mode=letter -- comparison arm A)",
        "answer_explanation": "letter + explanation + EOS (target_mode=answer_explanation -- comparison arm B)",
    }[target_mode]


def build_target_mode_example(example, tokenizer, target_mode, system_instruction=COMPARISON_SYSTEM_INSTRUCTION):
    """Turn one raw generation example (medmcqa_gen / medexqa layout: `question`,
    `answer` = " " + explanation, `gold_letter`, `letter_question`, `id`) into the
    prompt-engineered {question, answer, id} dict of the requested comparison arm.

    The prompt is `letter_question` wrapped with `system_instruction` (default
    COMPARISON_SYSTEM_INSTRUCTION -- the SAME for both arms in the MedMCQA /
    Granite-OBQA comparisons; the Qwen OBQA arm A passes MCQ_SYSTEM_INSTRUCTION);
    the target differs by arm:
      letter              -> answer = gold letter (bare, e.g. "C")
      answer_explanation  -> answer = gold letter + "\\nExplanation:" + explanation
    The caller decides about EOS (answer_explanation gets one, letter does not).
    Raises if the example does not carry the MCQA fields (non-generation datasets
    must keep using the plain MCQA path)."""
    if target_mode not in COMPARISON_MODES:
        raise ValueError(f"build_target_mode_example: target_mode must be one of {COMPARISON_MODES}, got {target_mode!r}")
    gold_letter = example.get("gold_letter")
    letter_question = example.get("letter_question")
    if not gold_letter or not letter_question:
        raise ValueError(f"target_mode={target_mode!r} needs `gold_letter` and `letter_question` on every "
                         f"example (generation datasets only); missing on id={example.get('id')!r}")
    engineered = multiple_choice_prompt_engineer(
        {"question": letter_question, "answer": gold_letter, "id": example["id"]},
        tokenizer=tokenizer, system_instruction=system_instruction,
    )
    if target_mode == "letter":
        target = gold_letter
    else:
        exp = example["answer"]                      # " " + gold explanation (loader convention)
        if not exp.startswith(" "):
            exp = " " + exp
        target = f"{gold_letter}{EXPLANATION_MARKER}{exp}"
    return {"question": engineered["question"], "answer": target, "id": engineered["id"]}


def comparison_eligible_indices(raw_rows, tokenizer, max_seq_len):
    """Indices of `raw_rows` whose LONGER (answer_explanation) training sequence --
    prompt + letter + "\\nExplanation:" + explanation + EOS, tokenized exactly as the
    generation preprocessor does -- has <= max_seq_len tokens. Both comparison arms
    apply this one list, so arm A never keeps a row arm B drops. None/0 = all rows."""
    if not max_seq_len:
        return list(range(len(raw_rows)))
    eos = tokenizer.eos_token or ""
    keep = []
    for i, ex in enumerate(raw_rows):
        e = build_target_mode_example(ex, tokenizer, "answer_explanation")
        n = len(tokenizer(e["question"] + e["answer"] + eos, add_special_tokens=False).input_ids)
        if n <= max_seq_len:
            keep.append(i)
    return keep


def eligible_ids_path(dataset_shortcode, max_seq_len, split_name, seed=42, eligible_tag=None):
    """`eligible_tag` (see eligible_tag_for): None -> the original filename (all
    Granite lists); otherwise "-<tag>" is appended so another tokenizer's list
    never collides with (or is checked against) the Granite one."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = f"-{eligible_tag}" if eligible_tag else ""
    return os.path.join(repo_root, "splits",
                        f"{dataset_shortcode}-comparison-eligible-{split_name}-maxseq{max_seq_len}-seed{seed}{tag}.txt")


def _freeze_eligible_ids(dataset_shortcode, split_name, raw_rows, keep, max_seq_len, seed, eligible_tag=None):
    """Write the eligible-ID list once (splits/...txt); on later calls verify the
    in-memory list is identical (so arm A and arm B provably used the same rows)."""
    path = eligible_ids_path(dataset_shortcode, max_seq_len, split_name, seed, eligible_tag)
    ids = [str(raw_rows[i]["id"]) for i in keep]
    if os.path.exists(path):
        frozen = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
        if frozen != ids:
            raise RuntimeError(f"eligible-ID list for {split_name} differs from the frozen one at {path} "
                               f"({len(ids)} vs {len(frozen)} ids). Both arms must use the same rows; "
                               f"delete the file only if you MEAN to move the list.")
        print(f"  [{split_name}] eligible-ID list verified against {path} ({len(ids)} ids)")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))
        print(f"  [{split_name}] eligible-ID list written -> {path} ({len(ids)} ids)")

# ---------------------------------------------------------------------------
# medmcqa_gen (branch MedMCQA): MedMCQA questions whose gold `exp` explanation is
# used as the FREE-TEXT generation target. Split protocol (frozen by manifest):
#   train  = MEDMCQA_GEN_N_TRAIN rows from the OFFICIAL train split, stratified
#            (proportional) by subject_name;
#   val    = MEDMCQA_GEN_N_VAL rows, also from official train, stratified,
#            disjoint from train  -> the ONLY selection set;
#   test   = MEDMCQA_GEN_N_TEST rows from the OFFICIAL validation split (the
#            set MedMCQA papers report on; test-split answers are withheld),
#            stratified  -> evaluated once per frozen configuration.
# Row filter (identical for every split): choice_type == "single", valid cop,
# four non-empty options, non-empty `exp` with MEDMCQA_GEN_MIN_EXP_WORDS <=
# words <= MEDMCQA_GEN_MAX_EXP_WORDS (drops "Ans. C i.e. Mite"-style stubs and
# textbook dumps that would exceed the generation cap), exact-duplicate
# questions removed (a question already in test is never re-used in train/val).
# ---------------------------------------------------------------------------
MEDMCQA_GEN_N_TRAIN = 30000
MEDMCQA_GEN_N_VAL = 1000
MEDMCQA_GEN_N_TEST = 1000
MEDMCQA_GEN_MIN_EXP_WORDS = 10
MEDMCQA_GEN_MAX_EXP_WORDS = 160   # ~<=256 Granite tokens incl. medical sub-words; see max_new_tokens
# Same inner prompt format as MedExQA (utils/prompt.py::render_inner 'generation'),
# so the two generation datasets can be evaluated with one template and the
# canonicalisation / format-control logic in the OoD bridge test applies to both.
GENERATION_INNER_TEMPLATE = "Question: {q}\nOptions:\n{opts}\n\nExplain the reasoning for the correct answer."
MCQA_INNER_TEMPLATE = "Question: {q}\nChoices:\n{opts}\nAnswer:"

# obqa_gen (branch OBQA-comparison): OpenBookQA with the gold science fact
# (`fact1`, HF "additional" config) as the explanation target. fact1 is short
# ("the sun is the source of energy for physical cycles on Earth" ~10 words),
# so the MedMCQA bounds (10-160) would reject nearly every row.
OBQA_GEN_MIN_EXP_WORDS = 3
OBQA_GEN_MAX_EXP_WORDS = 60


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

def _drop_overlong(dataset: Dataset, max_seq_len, name):
    """Drop rows whose tokenized length exceeds max_seq_len (None/0 = keep all).
    Rows are DROPPED, never truncated: truncating a generation target would cut
    the EOS off and teach the model to stop mid-sentence."""
    if not max_seq_len:
        return dataset
    lengths = [len(x) for x in dataset["input_ids"]]
    keep = [i for i, n in enumerate(lengths) if n <= max_seq_len]
    n_drop = len(lengths) - len(keep)
    if lengths:
        srt = sorted(lengths)
        print(f"  [{name}] token lengths: max={srt[-1]} p50={srt[len(srt)//2]} p95={srt[int(0.95*(len(srt)-1))]} "
              f"p99={srt[int(0.99*(len(srt)-1))]}; max_seq_len={max_seq_len} -> dropped {n_drop}/{len(lengths)} rows")
    return dataset.select(keep) if n_drop else dataset


def load_and_prepare_train_and_val_data(tokenizer: AutoTokenizer, train_dataset_shortcodes: List, seed=42,
                                        answer_only=False, max_seq_len=None,
                                        target_mode="explanation", system_prompt="comparison",
                                        eligible_tag=None) -> Tuple[Dataset, Dataset]:
    """Loads and preprocesses the dataset for causal language modeling.

    `system_prompt` (comparison target modes only; key into SYSTEM_INSTRUCTIONS,
    default 'comparison' = every existing run) selects the system instruction of
    the training prompt. `eligible_tag` (see eligible_tag_for) keys the frozen
    eligible-ID list to the tokenizer. The shared eligible-row list is ALWAYS
    computed from the comparison-prompt answer_explanation sequence, whatever
    `system_prompt` is, so both arms keep the same rows.
    MOE_SMOKE_N_ROWS=<n> (env) truncates train/val to n rows AFTER the
    eligibility list is frozen -- smoke runs only (use *_suffix smoke).

    Dispatch (mutually exclusive paths):
      * GENERATION datasets (see GENERATION_DATASETS), `target_mode="explanation"`
        (default): generation prompt template,
        `preprocess_mask_question_for_training(append_eos=True, pad=False)` --
        prompt-masked labels + EOS on the explanation, rows un-padded. This is
        the exact recipe that produced the existing MedExQA/MedMCQA
        adapters/weights; `answer_only` is irrelevant here (the loss is already
        target-only).
      * GENERATION datasets, `target_mode="letter"` (comparison arm A):
        comparison prompt (`letter_question` + COMPARISON_SYSTEM_INSTRUCTION)
        -> bare gold letter, answer-only preprocessor, no EOS (== the corrected
        exp4-train-ans MCQA recipe).
      * GENERATION datasets, `target_mode="answer_explanation"` (arm B): the
        IDENTICAL prompt -> letter + "\\nExplanation:" + explanation + EOS,
        generation preprocessor (loss on letter AND explanation).
        With `max_seq_len`, both arms keep the same rows: those whose
        answer_explanation sequence fits (comparison_eligible_indices; the ID
        list is frozen under splits/ and verified on every later call).
        See TARGET_MODES / build_target_mode_example.
      * MCQA + `answer_only=True`: `preprocess_answer_only_for_training` (prompt
        and answer tokenized separately -> exact answer-only labels).
      * MCQA + `answer_only=False`: legacy full-sequence-padded masking.
      (`target_mode` other than "explanation" on a non-generation dataset raises.)
    All paths return prompt-masked labels; pair with a labels-preserving collator
    (DataCollatorForSeq2Seq, label_pad_token_id=-100).
    `max_seq_len` (optional): drop (never truncate) train/val rows longer than
    this many tokens -- a memory guard for large generation sets (medmcqa_gen);
    the number of dropped rows is printed. None/0 = keep everything (default,
    identical to the behaviour that produced the MedExQA weights).
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    generation = is_generation_dataset(train_dataset_shortcodes)
    if target_mode != "explanation" and not generation:
        raise ValueError(f"target_mode={target_mode!r} is only defined for generation datasets "
                         f"{sorted(GENERATION_DATASETS)} (they carry gold_letter/letter_question); "
                         f"got {train_dataset_shortcodes}")
    engineer = generation_prompt_engineer if generation else multiple_choice_prompt_engineer
    if target_mode != "explanation":
        if system_prompt not in SYSTEM_INSTRUCTIONS:
            raise ValueError(f"system_prompt must be one of {sorted(SYSTEM_INSTRUCTIONS)}, got {system_prompt!r}")
        system_instruction = SYSTEM_INSTRUCTIONS[system_prompt]
        engineer = lambda x, tokenizer: build_target_mode_example(x, tokenizer, target_mode, system_instruction)  # noqa: E731

    train_raw, val_raw = [], []
    for dataset_shortcode in train_dataset_shortcodes:
        train_raw_curr, val_raw_curr, _ = load_exp_dataset(dataset_shortcode, seed=seed)
        train_raw.extend(train_raw_curr)
        val_raw.extend(val_raw_curr)

    comparison = target_mode in COMPARISON_MODES
    if comparison and max_seq_len:
        # ONE eligible-ID list for both arms, computed from the LONGER
        # (answer_explanation) sequence and frozen on disk, so arm A and arm B
        # train/validate on exactly the same rows whatever the target.
        ds_name = "+".join(train_dataset_shortcodes)
        keep_tr = comparison_eligible_indices(train_raw, tokenizer, max_seq_len)
        keep_va = comparison_eligible_indices(val_raw, tokenizer, max_seq_len)
        print(f"--- comparison eligibility (answer+explanation sequence <= {max_seq_len} tokens): "
              f"train {len(keep_tr)}/{len(train_raw)}, val {len(keep_va)}/{len(val_raw)} rows kept ---")
        _freeze_eligible_ids(ds_name, "train", train_raw, keep_tr, max_seq_len, seed, eligible_tag)
        _freeze_eligible_ids(ds_name, "val", val_raw, keep_va, max_seq_len, seed, eligible_tag)
        train_raw = [train_raw[i] for i in keep_tr]
        val_raw = [val_raw[i] for i in keep_va]

    smoke_n = int(os.environ.get("MOE_SMOKE_N_ROWS", "0") or 0)
    if smoke_n > 0:
        print(f"!!! MOE_SMOKE_N_ROWS={smoke_n}: truncating train/val to {smoke_n} rows each (SMOKE RUN ONLY) !!!")
        train_raw, val_raw = train_raw[:smoke_n], val_raw[:smoke_n]

    train_engineered = [engineer(x, tokenizer=tokenizer) for x in train_raw]
    val_engineered = [engineer(x, tokenizer=tokenizer) for x in val_raw]
    if target_mode != "explanation":
        print(f"--- comparison training prompt: system_prompt={system_prompt!r} ---")

    if target_mode == "letter":
        # Arm A: exact answer-only labels on the bare letter, no EOS (exp4 recipe).
        train_dataset = preprocess_answer_only_for_training(train_engineered, tokenizer)
        val_dataset = preprocess_answer_only_for_training(val_engineered, tokenizer)
        print("--- target_mode=letter: comparison prompt -> gold letter (answer-only loss, no EOS) ---")
    elif generation:
        # Generation (explanation / answer_explanation): append EOS + keep rows
        # un-padded (dynamic Seq2Seq collator).
        train_dataset = preprocess_mask_question_for_training(
            train_engineered, tokenizer, append_eos=True, pad=False)
        val_dataset = preprocess_mask_question_for_training(
            val_engineered, tokenizer, append_eos=True, pad=False)
        if target_mode == "answer_explanation":
            print("--- target_mode=answer_explanation: comparison prompt -> letter + '\\nExplanation:' + explanation + EOS "
                  "(loss on letter AND explanation) ---")
    else:
        preprocess = preprocess_answer_only_for_training if answer_only else preprocess_mask_question_for_training
        train_dataset = preprocess(train_engineered, tokenizer)
        val_dataset = preprocess(val_engineered, tokenizer)

    if not comparison:
        # Legacy per-arm length filter (explanation / MCQA). The comparison arms
        # were already filtered above on the shared list; _drop_overlong would be
        # a no-op for answer_explanation and must NOT run for letter.
        train_dataset = _drop_overlong(train_dataset, max_seq_len, "train")
        val_dataset = _drop_overlong(val_dataset, max_seq_len, "val")

    print(f"Datasets '{train_dataset_shortcodes}' loaded and preprocessed.")
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(val_dataset)}")

    return train_dataset, val_dataset

# ---------------------------------------------------------------------------
# Frozen derived-split manifests (medexqa, medmcqa_gen). One CSV per dataset:
#   splits/<dataset>-derived-seed<seed>.csv   (qhash, split, id, question_prefix)
# The loader verifies its in-memory split against the manifest on every call
# and raises if the split moved. The medexqa_* names below are kept as thin
# wrappers so existing callers / the MedExQA manifest file are unchanged.
# ---------------------------------------------------------------------------
_MANIFEST_LABEL = {"medexqa": "MedExQA", "medmcqa_gen": "MedMCQA-gen", "obqa_gen": "OBQA-gen"}
_MANIFEST_WRITER = {"medexqa": "write-medexqa-split-manifest.py",
                    "medmcqa_gen": "write-medmcqa-split-manifest.py",
                    "obqa_gen": "write-obqa-split-manifest.py"}


def split_manifest_path(dataset_shortcode, seed=42):
    """Repo-relative path of the frozen derived-split manifest for a dataset."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "splits", f"{dataset_shortcode}-derived-seed{seed}.csv")


def example_hash(example):
    """Content hash identifying an example independently of its (random) id."""
    return hashlib.sha1(example["question"].encode("utf-8")).hexdigest()[:16]


def _skip_manifest_env(dataset_shortcode):
    return f"{dataset_shortcode.upper()}_SKIP_MANIFEST"   # MEDEXQA_SKIP_MANIFEST / MEDMCQA_GEN_SKIP_MANIFEST


def write_split_manifest(dataset_shortcode, train, val, test, path):
    """Freeze the derived split: one row per example (qhash, split, id, question_prefix)."""
    import csv as _csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["qhash", "split", "id", "question_prefix"])
        for split_name, rows in (("train", train), ("val", val), ("test", test)):
            for ex in rows:
                prefix = " ".join(ex["question"].split())[:80]
                w.writerow([example_hash(ex), split_name, ex["id"], prefix])
    print(f"  {_MANIFEST_LABEL.get(dataset_shortcode, dataset_shortcode)}: wrote split manifest -> {path} "
          f"(train={len(train)}, val={len(val)}, test={len(test)})")


def verify_split_manifest(dataset_shortcode, train, val, test, seed=42, path=None):
    """Check the in-memory derived split against the frozen manifest.


        * manifest present  -> the multiset of (qhash, split) pairs must match the
      manifest exactly; otherwise raise (the split moved: RNG order, dedup, or
      upstream data changed). Set <DATASET>_SKIP_MANIFEST=1 to bypass.
    * manifest absent   -> print how to create it (write-<dataset>-split-manifest.py).

    Duplicate question texts are tolerated: rows are compared as a multiset of
    (qhash, split) pairs, so datasets whose loader does not dedup (obqa_gen
    mirrors the legacy obqa pool byte-for-byte, repeats included) verify as
    long as every copy stays in its recorded split with the same multiplicity.
    """
    import csv as _csv
    from collections import Counter
    label = _MANIFEST_LABEL.get(dataset_shortcode, dataset_shortcode)
    env = _skip_manifest_env(dataset_shortcode)
    path = path or split_manifest_path(dataset_shortcode, seed)
    if os.environ.get(env) == "1":
        print(f"  {label}: split-manifest check SKIPPED ({env}=1)")
        return
    if not os.path.exists(path):
        print(f"  {label}: no split manifest at {path}; run "
              f"`python {_MANIFEST_WRITER.get(dataset_shortcode, 'write-<dataset>-split-manifest.py')}` "
              f"once and commit it to freeze the split.")
        return
    recorded = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            recorded[(row["qhash"], row["split"])] += 1
    inmem, snippet = Counter(), {}
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        for ex in rows:
            h = example_hash(ex)
            inmem[(h, split_name)] += 1
            snippet.setdefault(h, ex["question"][:60])
    if inmem != recorded:
        counts = {s: sum(n for (_, sp), n in inmem.items() if sp == s) for s in ("train", "val", "test")}
        rec_counts = {s: sum(n for (_, sp), n in recorded.items() if sp == s) for s in ("train", "val", "test")}
        diffs = [(h, sp, inmem.get((h, sp), 0), recorded.get((h, sp), 0))
                 for (h, sp) in sorted(set(inmem) | set(recorded))
                 if inmem.get((h, sp), 0) != recorded.get((h, sp), 0)]
        head = "\n".join(f"    qhash={h} split={sp}: in-memory x{a} vs manifest x{b}: {snippet.get(h, '?')!r}"
                         for h, sp, a, b in diffs[:5])
        raise RuntimeError(
            f"{label} derived split does not match the frozen manifest {path}: "
            f"{len(diffs)} differing (qhash, split) entries; counts in-memory={counts} manifest={rec_counts}.\n"
            f"{head}\nThe split moved (RNG order / dedup / upstream data change). Do NOT proceed with "
            f"val/test-dependent runs; investigate, or set {env}=1 to bypass knowingly."
        )
    counts = {s: sum(n for (_, sp), n in inmem.items() if sp == s) for s in ("train", "val", "test")}
    n_dup = sum(n - 1 for n in inmem.values() if n > 1)
    splits_by_hash = {}
    for (h, sp), _n in inmem.items():
        splits_by_hash.setdefault(h, set()).add(sp)
    n_leak = sum(1 for ss in splits_by_hash.values() if len(ss) > 1)
    dup_note = f"; {n_dup} duplicate-question row(s) tolerated" if n_dup else ""
    if n_leak:
        dup_note += f"; WARNING: {n_leak} question text(s) appear in more than one split (upstream data; frozen as-is)"
    print(f"  {label}: split matches frozen manifest ({path}); "
          f"train={counts['train']} val={counts['val']} test={counts['test']}{dup_note}")


# Backward-compatible MedExQA wrappers (write-medexqa-split-manifest.py imports these).
def medexqa_split_manifest_path(seed=42):
    return split_manifest_path("medexqa", seed)


def medexqa_example_hash(example):
    return example_hash(example)


def write_medexqa_split_manifest(train, val, test, path):
    return write_split_manifest("medexqa", train, val, test, path)


def verify_medexqa_split_manifest(train, val, test, seed=42, path=None):
    return verify_split_manifest("medexqa", train, val, test, seed=seed, path=path)


# ---------------------------------------------------------------------------
# medmcqa_gen loader
# ---------------------------------------------------------------------------
def _medmcqa_gen_reformat(example, split_name):
    """MedMCQA row -> generation example, or None if it fails the row filter.

    Same field layout as the MedExQA examples so every downstream script
    (training, readouts, labels, OoD bridge) works unchanged:
      question        generation prompt inner text (GENERATION_INNER_TEMPLATE)
      answer          " " + gold explanation (`exp`)   -- the free-text target
      id              medmcqa_gen_<split>_<official id>  (deterministic, no RNG)
      gold_letter     A-D
      letter_question MCQA-style prompt for the letter-probe label
      explanation_2   ""  (MedMCQA has a single reference explanation)
      subject_name    kept for stratification / per-subject analysis
    Returns (example, None) if kept, else (None, reason) for the filter funnel.
    """
    if example.get("choice_type") != "single":
        return None, "not_single"
    cop = example.get("cop")
    if cop is None or not (0 <= cop < 4):          # HF cop is 0-indexed (checked 2026-08-17)
        return None, "bad_cop"
    q = " ".join(str(example.get("question") or "").split())
    choices = [" ".join(str(example.get(k) or "").split()) for k in ("opa", "opb", "opc", "opd")]
    exp = " ".join(str(example.get("exp") or "").split())
    if not q or any(not c for c in choices):
        return None, "empty_field"
    if not exp:
        return None, "no_explanation"
    n_words = len(exp.split())
    if n_words < MEDMCQA_GEN_MIN_EXP_WORDS:
        return None, "exp_too_short"
    if n_words > MEDMCQA_GEN_MAX_EXP_WORDS:
        return None, "exp_too_long"
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, choices))
    return {
        "question": GENERATION_INNER_TEMPLATE.format(q=q, opts=opts),
        "answer": " " + exp,
        "id": f"medmcqa_gen_{split_name}_{example.get('id')}",
        "explanation_2": "",
        "gold_letter": labels[cop],
        "letter_question": MCQA_INNER_TEMPLATE.format(q=q, opts=opts),
        "subject_name": str(example.get("subject_name") or "Unknown"),
        "topic_name": str(example.get("topic_name") or ""),
    }, None


def _stratified_take(rows, n, key, rng):
    """Proportional stratified sample of n rows (deterministic given rng).

    Rows are grouped by `key`, each group is shuffled with `rng`, and every group
    contributes round(n * group_frac) rows (largest-remainder rounding so the total
    is exactly min(n, len(rows))). Returns (taken, remaining) with each group's
    remaining rows kept in their shuffled order (so a second call on `remaining`
    yields a disjoint stratified sample)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    total = len(rows)
    n = min(n, total)
    quotas = {k: (n * len(groups[k])) / total for k in keys}
    take = {k: int(quotas[k]) for k in keys}
    short = n - sum(take.values())
    for k in sorted(keys, key=lambda k: quotas[k] - int(quotas[k]), reverse=True)[:short]:
        take[k] += 1
    taken, remaining = [], []
    for k in keys:
        taken.extend(groups[k][:take[k]])
        remaining.extend(groups[k][take[k]:])
    rng.shuffle(taken)
    return taken, remaining


def _load_medmcqa_gen(seed):
    """Build the medmcqa_gen (train, val, test) split. See the constants at the top."""
    rng = random.Random(seed)   # private RNG: independent of the global stream

    dataset = datasets.load_dataset("openlifescienceai/medmcqa")

    def build(split_name):
        raw = dataset[split_name]
        funnel = {"rows": len(raw), "kept": 0}
        out = []
        for ex in raw:
            r, reason = _medmcqa_gen_reformat(ex, split_name)
            if r is None:
                funnel[reason] = funnel.get(reason, 0) + 1
                continue
            out.append(r); funnel["kept"] += 1
        print(f"  MedMCQA-gen [{split_name}] filter funnel: {funnel}")
        return out

    def qkey(r):
        return " ".join(r["question"].lower().split())

    # ---- test: official validation split (answers + explanations available) ----
    dev_pool = build("validation")
    seen, dedup = set(), []
    for r in dev_pool:
        k = qkey(r)
        if k in seen: continue
        seen.add(k); dedup.append(r)
    test_dataset, _ = _stratified_take(dedup, MEDMCQA_GEN_N_TEST, "subject_name", rng)
    test_keys = {qkey(r) for r in test_dataset}

    # ---- train / val: official train split, deduped, disjoint from test ----
    train_pool = build("train")
    seen, dedup, n_dup, n_in_test = set(), [], 0, 0
    for r in train_pool:
        k = qkey(r)
        if k in test_keys:
            n_in_test += 1; continue
        if k in seen:
            n_dup += 1; continue
        seen.add(k); dedup.append(r)
    print(f"  MedMCQA-gen [train] after dedup: {len(dedup)} rows "
          f"(dropped {n_dup} duplicate questions, {n_in_test} overlapping the test set)")
    validation_dataset, rest = _stratified_take(dedup, MEDMCQA_GEN_N_VAL, "subject_name", rng)
    train_dataset, _ = _stratified_take(rest, MEDMCQA_GEN_N_TRAIN, "subject_name", rng)

    if len(train_dataset) < 5000 or len(validation_dataset) < 100 or len(test_dataset) < 100:
        raise ValueError(f"MedMCQA-gen: unexpectedly small split after filtering: "
                         f"train={len(train_dataset)} val={len(validation_dataset)} test={len(test_dataset)}")

    def subj_hist(rows):
        from collections import Counter
        c = Counter(r["subject_name"] for r in rows)
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
    print(f"  MedMCQA-gen subjects (train): {subj_hist(train_dataset)}")
    print(f"  MedMCQA-gen subjects (val):   {subj_hist(validation_dataset)}")
    print(f"  MedMCQA-gen subjects (test):  {subj_hist(test_dataset)}")
    return train_dataset, validation_dataset, test_dataset


def _load_obqa_gen(seed):
    """OBQA with the gold science fact (`fact1`) as the free-text explanation
    target (branch OBQA-comparison). Loads the HF "additional" config -- same
    rows/order as "main", plus `fact1` -- and replicates the legacy `obqa`
    split protocol byte-for-byte (pool = train split + validation split, drop
    only invalid answerKey, [:5050], last 50 = val, first 5000 = train, test =
    official test), so the train pool matches what the frozen exp4-train-ans
    arm-A adapter (adapters/granite-obqa-ansmask) was trained on and the 50 val
    rows are unseen by both arms. Deterministic: no RNG (`seed` only names the
    manifest). `letter_question` is byte-identical to the legacy reformat_obqa
    prompt. The fact1 word-count filter ([OBQA_GEN_MIN_EXP_WORDS,
    OBQA_GEN_MAX_EXP_WORDS]) runs AFTER the split carve and only on train/val:
    fact1 is unused at eval, and filtering before the [:5050] slice would shift
    the pool relative to exp4. write-obqa-split-manifest.py freezes the split
    and cross-checks it against the legacy `obqa` loader."""
    dataset = datasets.load_dataset("openbookqa", "additional")

    def reformat(example, split_name):
        labels = example["choices"]["label"]
        question = example.get("question_stem")
        choices = example["choices"]["text"]
        answer_key = example["answerKey"]
        if answer_key not in labels:
            return None
        opts = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
        return {
            "question": GENERATION_INNER_TEMPLATE.format(q=question, opts=opts),
            "answer": " " + str(example.get("fact1") or "").strip(),
            "id": f"obqa_gen_{split_name}_{example['id']}",
            "explanation_2": "",
            "gold_letter": answer_key,
            "letter_question": MCQA_INNER_TEMPLATE.format(q=question, opts=opts),
        }

    train_pool = ([reformat(ex, "train") for ex in dataset["train"]]
                  + [reformat(ex, "validation") for ex in dataset["validation"]])
    train_pool = [r for r in train_pool if r is not None][:5050]
    validation_dataset = train_pool[-50:]
    train_dataset = train_pool[:-50]
    test_dataset = [r for ex in dataset["test"] if (r := reformat(ex, "test")) is not None]

    def fact1_filter(rows, split_name):
        funnel = {"rows": len(rows), "kept": 0, "no_fact1": 0, "fact1_too_short": 0, "fact1_too_long": 0}
        out = []
        for r in rows:
            n_words = len(r["answer"].split())
            if not r["answer"].strip():
                funnel["no_fact1"] += 1
            elif n_words < OBQA_GEN_MIN_EXP_WORDS:
                funnel["fact1_too_short"] += 1
            elif n_words > OBQA_GEN_MAX_EXP_WORDS:
                funnel["fact1_too_long"] += 1
            else:
                out.append(r); funnel["kept"] += 1
        print(f"  OBQA-gen [{split_name}] fact1 filter funnel: {funnel}")
        return out

    train_dataset = fact1_filter(train_dataset, "train")
    validation_dataset = fact1_filter(validation_dataset, "val")
    print(f"  OBQA-gen [test] fact1 filter NOT applied ({len(test_dataset)} rows; fact1 unused at eval)")

    if len(train_dataset) < 4000 or len(validation_dataset) < 20 or len(test_dataset) < 400:
        raise ValueError(f"OBQA-gen: unexpectedly small split after filtering: "
                         f"train={len(train_dataset)} val={len(validation_dataset)} test={len(test_dataset)}")
    return train_dataset, validation_dataset, test_dataset


# ---------------------------------------------------------------------------
# Explanation-bearing OoD EVAL sets (branch OBQA-comparison,
# evaluate_ood_expl_readout.py): scienceqa / ecqa / aqua_rat. Eval-only --
# nothing trains on them, they are NOT in GENERATION_DATASETS and have no split
# manifest. Every record uses the generation-dataset schema (question, answer =
# " " + explanation, id, explanation_2, gold_letter, letter_question) plus
#   n_choices  4 or 5 (ECQA / AQuA-RAT have five options; E is never dropped)
#   meta       dataset-specific provenance (subject, conflict flags, raw text)
# and ONE canonical inner prompt (MCQA_INNER_TEMPLATE, "A. text" options).
# Raw downloads are cached under <repo>/data/ood_raw/<dataset>/ (override with
# MOE_RAW_DATA_DIR); the row converters are pure so they can be unit-tested.
# ---------------------------------------------------------------------------
LETTERS5 = ["A", "B", "C", "D", "E"]
ECQA_RAW_URL = "https://raw.githubusercontent.com/dair-iitd/ECQA-Dataset/main/"
ECQA_URLS = {
    "ecqa.jsonl": ECQA_RAW_URL + "ecqa.jsonl",
    "train_ids.txt": ECQA_RAW_URL + "author_split/train_ids.txt",
    "val_ids.txt": ECQA_RAW_URL + "author_split/val_ids.txt",
    "test_ids.txt": ECQA_RAW_URL + "author_split/test_ids.txt",
}
# Official CommonsenseQA v1.11 release (the files ECQA's generate_data.py joins to).
CSQA_URLS = {
    "train_rand_split.jsonl": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
    "dev_rand_split.jsonl": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
}


def raw_data_dir(name):
    root = os.environ.get("MOE_RAW_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ood_raw")
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def _download(url, path):
    """Fetch `url` to `path` once (temp file + rename; refuses empty files)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    import urllib.request
    tmp = path + ".part"
    print(f"  downloading {url} -> {path}")
    urllib.request.urlretrieve(url, tmp)
    if os.path.getsize(tmp) == 0:
        os.remove(tmp)
        raise RuntimeError(f"empty download: {url}")
    os.replace(tmp, path)
    return path


def mcqa_eval_record(code, split_name, source_id, question, choices, gold_letter, explanation,
                     explanation_2="", meta=None):
    """Canonical eval record. `choices` are the bare option texts (labels are
    assigned A.. in order); `explanation` gets the loader-convention leading space."""
    labels = LETTERS5[:len(choices)]
    if gold_letter not in labels:
        raise ValueError(f"{code}: gold letter {gold_letter!r} outside {labels}")
    opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, choices))
    return {
        "question": GENERATION_INNER_TEMPLATE.format(q=question, opts=opts),
        "answer": " " + str(explanation).strip(),
        "id": f"{code}_{split_name}_{source_id}",
        "explanation_2": str(explanation_2 or "").strip(),
        "gold_letter": gold_letter,
        "letter_question": MCQA_INNER_TEMPLATE.format(q=question, opts=opts),
        "n_choices": len(choices),
        "meta": dict(meta or {}),
    }


# ---- ScienceQA ---------------------------------------------------------------
def scienceqa_row_to_example(ex, idx, split_name):
    """Keep text-only (no image), context-free (no hint), exactly-four-choice
    rows with an authored `solution`; the 0-based integer `answer` becomes a
    letter. Returns (record, reject_reason)."""
    if ex.get("image") is not None:
        return None, "has_image"
    if str(ex.get("hint") or "").strip():
        return None, "has_hint"
    choices = [str(c).strip() for c in (ex.get("choices") or [])]
    if len(choices) != 4 or any(not c for c in choices):
        return None, "not_4_choices"
    solution = str(ex.get("solution") or "").strip()
    if not solution:
        return None, "no_solution"
    ans = ex.get("answer")
    if isinstance(ans, bool) or not isinstance(ans, numbers.Integral) or not (0 <= int(ans) < 4):
        return None, "bad_answer"
    question = str(ex.get("question") or "").strip()
    if not question:
        return None, "no_question"
    meta = {k: ex.get(k) for k in ("subject", "topic", "category", "grade", "task", "skill")}
    meta["has_lecture"] = bool(str(ex.get("lecture") or "").strip())
    return mcqa_eval_record("scienceqa", split_name, idx, question, choices, "ABCD"[int(ans)],
                            solution, meta=meta), None


def _load_scienceqa(seed):
    ds = datasets.load_dataset("derek-thomas/ScienceQA")
    out = {}
    for split_name, hf_split in (("val", "validation"), ("test", "test")):
        d = ds[hf_split].cast_column("image", datasets.Image(decode=False))   # never decode PIL
        rows, funnel = [], Counter()
        for idx, ex in enumerate(d):
            r, why = scienceqa_row_to_example(ex, idx, split_name)
            if r is None:
                funnel[why] += 1
            else:
                rows.append(r)
                funnel["kept"] += 1
        random.Random(seed).shuffle(rows)
        print(f"  ScienceQA [{split_name}] filter funnel (text-only, no hint, 4 choices, solution): {dict(funnel)}")
        out[split_name] = rows
    if len(out["test"]) < 100:
        raise ValueError(f"ScienceQA: unexpectedly few usable test rows ({len(out['test'])})")
    return [], out["val"], out["test"]


# ---- ECQA (annotations joined BY ID to CommonsenseQA) --------------------------
def _csqa_normalize(row):
    """One CommonsenseQA row (official jsonl layout or HF tau/commonsense_qa
    layout) -> (id, stem, labels, texts, answerKey, concept)."""
    q = row.get("question")
    if isinstance(q, dict):                      # official jsonl
        stem = q.get("stem")
        labels = [c["label"] for c in q.get("choices", [])]
        texts = [c["text"] for c in q.get("choices", [])]
        concept = q.get("question_concept")
    else:                                        # HF layout
        stem = q
        ch = row.get("choices") or {}
        labels, texts = list(ch.get("label", [])), list(ch.get("text", []))
        concept = row.get("question_concept")
    return str(row["id"]), str(stem or "").strip(), labels, [str(t).strip() for t in texts], \
        str(row.get("answerKey") or "").strip().upper(), concept


def ecqa_join(ecqa_rows, csqa_rows, split_ids, split_name):
    """Join ECQA annotations (id, positives, negatives, explanation) to the exact
    CommonsenseQA release BY ID -- never by row order -- restricted to the
    author-split ids. Preserves CommonsenseQA's answerKey as the gold letter
    (no text matching). Explanation = taskA_pos = positives joined with a
    newline (as the official generate_data.py writes it); the free-flow
    explanation is kept in explanation_2. Raises on any id that fails to
    resolve. Returns (records in split_ids order, funnel)."""
    csqa_by_id = {}
    for row in csqa_rows:
        cid, stem, labels, texts, key, concept = _csqa_normalize(row)
        csqa_by_id[cid] = (stem, labels, texts, key, concept)
    # ecqa.jsonl repeats some ids; the official generate_data.py assigns
    # data[id][...] per line, so the LAST occurrence wins (it only counts them
    # in a "twice" counter). Mirror that exactly and report the count.
    ecqa_by_id, n_dup = {}, 0
    for row in ecqa_rows:
        eid = str(row.get("id") if row.get("id") is not None else row.get("q_no"))
        if eid in ecqa_by_id:
            n_dup += 1
        ecqa_by_id[eid] = row
    missing_in_csqa = [i for i in ecqa_by_id if i not in csqa_by_id]
    if missing_in_csqa:
        raise ValueError(f"ECQA: {len(missing_in_csqa)} annotation ids not found in CommonsenseQA "
                         f"(first: {missing_in_csqa[:3]}) -- wrong CommonsenseQA release?")
    records, funnel = [], Counter()
    funnel["duplicate_annotation_ids_last_wins"] = n_dup
    for sid in split_ids:
        sid = str(sid).strip()
        if not sid:
            continue
        if sid not in ecqa_by_id or sid not in csqa_by_id:
            raise ValueError(f"ECQA [{split_name}]: author-split id {sid!r} missing from "
                             f"{'ecqa.jsonl' if sid not in ecqa_by_id else 'CommonsenseQA'}")
        ann = ecqa_by_id[sid]
        stem, labels, texts, key, concept = csqa_by_id[sid]
        if labels != LETTERS5 or len(texts) != 5 or any(not t for t in texts):
            funnel["bad_choices"] += 1
            continue
        if key not in LETTERS5 or not stem:
            funnel["bad_answerkey"] += 1
            continue
        pos = ann.get("positives")
        pos_text = "\n".join(str(p).strip() for p in pos if str(p).strip()) if isinstance(pos, list) \
            else str(pos or "").strip()
        if not pos_text:
            funnel["no_positives"] += 1
            continue
        neg = ann.get("negatives")
        meta = {
            "taskA_neg": "\n".join(str(n).strip() for n in neg) if isinstance(neg, list) else str(neg or ""),
            "concept": concept,
        }
        records.append(mcqa_eval_record("ecqa", split_name, sid, stem, texts, key, pos_text,
                                        explanation_2=str(ann.get("explanation") or "").strip(),
                                        meta=meta))
        funnel["kept"] += 1
    return records, funnel


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_ecqa(seed):
    d = raw_data_dir("ecqa")
    local = {name: _download(url, os.path.join(d, name)) for name, url in ECQA_URLS.items()}
    csqa_rows = []
    try:
        for name, url in CSQA_URLS.items():
            csqa_rows += _read_jsonl(_download(url, os.path.join(d, name)))
        print(f"  ECQA: CommonsenseQA train+dev from the official release ({len(csqa_rows)} rows)")
    except Exception as e:
        print(f"  ECQA: official CommonsenseQA download failed ({e!r}); falling back to HF tau/commonsense_qa")
        hf = datasets.load_dataset("tau/commonsense_qa")
        csqa_rows = list(hf["train"]) + list(hf["validation"])
    ecqa_rows = _read_jsonl(local["ecqa.jsonl"])
    out = {}
    for split_name, ids_file in (("train", "train_ids.txt"), ("val", "val_ids.txt"), ("test", "test_ids.txt")):
        with open(local[ids_file], encoding="utf-8") as f:
            split_ids = [line.strip() for line in f if line.strip()]
        rows, funnel = ecqa_join(ecqa_rows, csqa_rows, split_ids, split_name)
        random.Random(seed).shuffle(rows)
        print(f"  ECQA [{split_name}] joined by id: {dict(funnel)} (author split: {len(split_ids)} ids)")
        out[split_name] = rows
    if len(out["test"]) < 100:
        raise ValueError(f"ECQA: unexpectedly few test rows ({len(out['test'])})")
    return out["train"], out["val"], out["test"]


# ---- AQuA-RAT ---------------------------------------------------------------
AQUA_OPTION_RE = re.compile(r"^\s*([A-E])\s*[\)\.:]\s*(.*)$", re.S)
# Terminal answer-conclusion phrases, matched ONLY at the very end of the
# rationale (high precision; algebraic single letters mid-text are untouched).
# Group `pre` is the delimiter kept in place, `letter` the concluded option.
AQUA_TAIL_PATTERNS = [
    ("answer_phrase", re.compile(
        r"(?P<pre>^|[\s\.\,;:])[\(\[]?(?:so\s+|hence\s+|thus\s+|therefore\s+)?(?:the\s+)?(?:correct\s+)?"
        r"(?:answer|option|choice|ans)\s*(?:(?:is|:|-|=|\.|will\s+be|should\s+be)\s*)?"
        r"(?:option\s+|choice\s+)?[\(\[]?(?P<letter>[A-E])[\)\]]?\s*[.!]?\s*$", re.I)),
    ("bare_letter", re.compile(
        r"(?P<pre>^|\n|[.;:,!?]\s*)[\(\[]?(?P<letter>[A-E])[\)\]]?\s*[.!]?\s*$")),
]


def aqua_strip_option_prefix(options):
    """["A)125", "B)150", ...] -> ["125", "150", ...]; None unless the labels are
    exactly A..E in order and every option text is non-empty."""
    options = list(options or [])
    if len(options) != 5:
        return None
    texts = []
    for i, opt in enumerate(options):
        m = AQUA_OPTION_RE.match(str(opt))
        if not m or m.group(1) != LETTERS5[i]:
            return None
        text = m.group(2).strip()
        if not text or AQUA_OPTION_RE.match(text):      # double-labelled "A)A)125"
            return None
        texts.append(text)
    return texts if len(texts) == 5 else None


def aqua_normalize_rationale(raw, max_passes=3):
    """Remove a TERMINAL answer-conclusion phrase ("Answer: C", "Correct answer -
    A", "CORRECT OPTION: OPTION E", "Choice B", a bare final "D", ...) from the
    rationale so the teacher-forced target does not restate the label.
    Returns (normalized_text, extracted_letter or None, pattern_name or None)."""
    text = str(raw or "").strip()
    letter, name = None, None
    for _ in range(max_passes):
        hit = None
        for pat_name, pat in AQUA_TAIL_PATTERNS:
            m = pat.search(text)
            if m:
                hit = (pat_name, m)
                break
        if hit is None:
            break
        pat_name, m = hit
        if letter is None:
            letter, name = m.group("letter").upper(), pat_name
        text = (text[:m.start()] + m.group("pre")).rstrip()
        text = re.sub(r"[\s\-–—:;,]+$", "", text).rstrip()
    return text, letter, name


def aqua_row_to_example(ex, idx, split_name):
    """One raw AQuA-RAT row -> (record, reject_reason)."""
    correct = str(ex.get("correct") or "").strip().upper()
    if correct not in LETTERS5:
        return None, "bad_correct"
    texts = aqua_strip_option_prefix(ex.get("options"))
    if texts is None:
        return None, "bad_options"
    question = str(ex.get("question") or "").strip()
    if not question:
        return None, "no_question"
    raw = str(ex.get("rationale") or "")
    norm, extracted, pat = aqua_normalize_rationale(raw)
    if not norm.strip():
        return None, "empty_after_strip"
    meta = {
        "rationale_raw": raw,
        "extracted_letter": extracted,
        "strip_pattern": pat,
        "rationale_label_conflict": bool(extracted is not None and extracted != correct),
    }
    return mcqa_eval_record("aqua_rat", split_name, idx, question, texts, correct, norm, meta=meta), None


def _load_aqua_rat(seed):
    ds = datasets.load_dataset("deepmind/aqua_rat", "raw")

    def convert(hf_split, split_name):
        rows, funnel = [], Counter()
        for idx, ex in enumerate(ds[hf_split]):
            r, why = aqua_row_to_example(ex, f"{hf_split}{idx}", split_name)
            if r is None:
                funnel[why] += 1
            else:
                rows.append(r)
                funnel["kept"] += 1
                funnel["label_conflict"] += int(r["meta"]["rationale_label_conflict"])
                funnel["stripped_tail"] += int(r["meta"]["extracted_letter"] is not None)
        print(f"  AQuA-RAT [{hf_split} -> {split_name}]: {dict(funnel)}")
        return rows

    val = convert("validation", "val")
    test = convert("validation", "test") + convert("test", "test")   # official held-out pool (508)
    random.Random(seed).shuffle(val)
    random.Random(seed).shuffle(test)
    if len(test) < 100:
        raise ValueError(f"AQuA-RAT: unexpectedly few test rows ({len(test)})")
    return [], val, test


def load_exp_dataset(dataset_shortcode, seed=42, split=None):
    """
    Loads and processes one of the six specified experimental datasets with
    custom train/validation/test splits for fine-tuning and evaluation.

    Args:
        dataset_shortcode (str): One of 'obqa', 'arc_c', 'arc_e', 'sciq',
                                 'mmlu_law', 'medmcqa_med' (MCQA) or the
                                 generation sets 'medexqa', 'medmcqa_gen',
                                 'obqa_gen'.
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

    elif dataset_shortcode == "medmcqa_gen":
        train_dataset, validation_dataset, test_dataset = _load_medmcqa_gen(seed)

    elif dataset_shortcode == "obqa_gen":
        train_dataset, validation_dataset, test_dataset = _load_obqa_gen(seed)

    # Eval-only explanation-bearing OoD sets (own val/test, empty/unused train).
    elif dataset_shortcode == "scienceqa":
        train_dataset, validation_dataset, test_dataset = _load_scienceqa(seed)

    elif dataset_shortcode == "ecqa":
        train_dataset, validation_dataset, test_dataset = _load_ecqa(seed)

    elif dataset_shortcode == "aqua_rat":
        train_dataset, validation_dataset, test_dataset = _load_aqua_rat(seed)

    else:
        raise ValueError(f"Dataset '{dataset_shortcode}' not supported by load_exp_dataset.")

    if dataset_shortcode not in ("medmcqa_gen", "obqa_gen", "scienceqa", "ecqa", "aqua_rat"):
        # Generic tail: the last 50 train rows become the val set. medmcqa_gen
        # builds its own (larger, stratified) val set above; obqa_gen carves its
        # own last-50 val BEFORE the fact1 filter (exp4 pool parity).
        validation_dataset = train_dataset[-50:]
        train_dataset = train_dataset[:-50]

    if dataset_shortcode in ("medexqa", "medmcqa_gen", "obqa_gen"):
        verify_split_manifest(dataset_shortcode, train_dataset, validation_dataset, test_dataset, seed=seed)

    print(f"Dataset '{dataset_shortcode}' processed: Train={len(train_dataset)}, Val={len(validation_dataset)}, Test={len(test_dataset)}")
    
    if split == "train":
        return train_dataset
    elif split == "validation" or split == "val":
        return validation_dataset
    elif split == "test":
        return test_dataset
    
    return train_dataset, validation_dataset, test_dataset