from transformers import PreTrainedTokenizerBase

def multiple_choice_prompt_engineer(
        example, 
        tokenizer: PreTrainedTokenizerBase
        # Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`]
):
    """
    Preprocess a single example to form the input prompt.
    Returns a dictionary with input text and the correct answer.
    """

    SYSTEM_INSTRUCTION = (
    "You are a multiple-choice quiz answer generator. "
    "Respond with ONLY the letter of the correct option, for example, 'A', 'B', 'C', or 'D'."
)

    chat = [
        {
            "role": "system", 
            "content": SYSTEM_INSTRUCTION
        },
        {
            "role": "user",
            "content": example["question"]
        }
    ]

    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_eos=False)

    return {
        "question": input_text,
        "answer": example["answer"],
        "id": example["id"]
    }

def generation_prompt_engineer(
        example,
        tokenizer: PreTrainedTokenizerBase
):
    """
    Prompt engineer for OPEN TEXT GENERATION datasets (e.g. MedExQA).

    Unlike `multiple_choice_prompt_engineer`, the model is asked to produce a
    free-text explanation, NOT a single option letter, so the system prompt
    invites reasoning instead of constraining the output to 'A'..'D'. The target
    (`answer`) is the free-text explanation string, carried through unchanged.
    """

    SYSTEM_INSTRUCTION = (
        "Read the multiple-choice question and its options, then give a clear, "
        "concise explanation of the reasoning behind the correct answer."
    )

    chat = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": example["question"]},
    ]

    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, add_eos=False
    )

    return {
        "question": input_text,
        "answer": example["answer"],
        "id": example["id"],
    }

def multi_shot_prompt_engineer(
        example, 
        tokenizer: PreTrainedTokenizerBase
        # Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`]
):
    """
    Preprocess a single example to form the input prompt.
    Returns a dictionary with input text and the correct answer.
    """
    q_list = ...
    a_list = ...
    context = ...



    chat = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"Question: {question}\nChoices:\n" + "\n".join(
                [f"{label}. {choice}" for label, choice in zip(labels, choices)]
            )
        }
    ]

    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return {
        "input_text": input_text,
        "answerKey": answer
    }

