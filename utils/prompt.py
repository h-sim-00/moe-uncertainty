from transformers import PreTrainedTokenizerBase

# System instruction of the MCQA (letter-only) prompt. Unchanged from the
# original code; every MCQA run (OBQA, ...) and the arm-A ("letter") MedMCQA
# comparison runs use it.
MCQ_SYSTEM_INSTRUCTION = (
    "You are a multiple-choice quiz answer generator. "
    "Respond with ONLY the letter of the correct option, for example, 'A', 'B', 'C', or 'D'."
)

# System instruction of the MedMCQA comparison (branch MedMCQA-comparison),
# used by BOTH arms (letter-only and letter+explanation) at training and at
# evaluation, so the prompt is identical across arms and only the TARGET differs.
# The user turn is the MCQA prompt (ends in "Answer:"); the assistant target is
# "<letter>" (arm A) or "<letter>\nExplanation: <gold explanation>" (arm B).
# Supervisor's wording: the prompt asks for an answer and an explanation.
COMPARISON_SYSTEM_INSTRUCTION = (
    "You are a multiple-choice quiz answer generator. "
    "First respond with ONLY the letter of the correct option, for example, 'A', 'B', 'C', or 'D'. "
    "Then, on a new line beginning with 'Explanation:', explain the reasoning behind that answer."
)

# Named system instructions selectable at evaluation time (evaluate_letter.py
# --system_prompt): 'comparison' for the comparison arms, 'mcq' for models that
# were trained with the original MCQ instruction (e.g. the OBQA reference).
SYSTEM_INSTRUCTIONS = {
    "comparison": COMPARISON_SYSTEM_INSTRUCTION,
    "mcq": MCQ_SYSTEM_INSTRUCTION,
}


def multiple_choice_prompt_engineer(
        example,
        tokenizer: PreTrainedTokenizerBase,
        # Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`]
        system_instruction: str = None,
):
    """
    Preprocess a single example to form the input prompt.
    Returns a dictionary with input text and the correct answer.
    `system_instruction` (optional) overrides the default MCQ system prompt
    (the comparison arms pass COMPARISON_SYSTEM_INSTRUCTION); None = unchanged behaviour.
    """

    SYSTEM_INSTRUCTION = system_instruction if system_instruction is not None else MCQ_SYSTEM_INSTRUCTION

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



# ---------------------------------------------------------------------------
# Inner-prompt canonicalisation (codex-recom-iter1, supervisor point 5).
# Every MCQA loader in utils/data.py renders the same inner format
#   "Question: <q>\nChoices:\nA. ..\nB. ..\n...\nAnswer:"
# while MedExQA renders
#   "Question: <q>\nOptions:\nA. ..\n...\n\nExplain the reasoning for the correct answer."
# An ID-vs-OoD score can therefore separate the two on SURFACE cues (Options vs
# Choices, the trailing instruction, length). parse_mcqa_question() recovers
# (question, options) from either, render_inner() re-renders in ONE format.
# ---------------------------------------------------------------------------
import re as _re

_INNER_RE = _re.compile(
    r"^\s*(?:Context:\s*(?P<context>.*?)\n)?Question:\s*(?P<q>.*?)\n(?:Choices|Options):\s*\n(?P<opts>.*?)"
    r"(?:\n\s*\nExplain the reasoning for the correct answer\.\s*|\nAnswer:\s*)?$",
    _re.S,
)


def parse_mcqa_question(inner_text):
    """-> dict(question, options{letter: text}, context) or None if the text is
    not in a recognised inner format."""
    m = _INNER_RE.match(inner_text or "")
    if not m:
        return None
    opts = {}
    for om in _re.finditer(r"(?m)^\s*([A-Z])\.\s*(.*?)\s*$", m.group("opts")):
        opts[om.group(1)] = om.group(2)
    if not opts:
        return None
    return {"question": m.group("q").strip(), "options": opts, "context": (m.group("context") or "").strip()}


def render_inner(parsed, fmt):
    """Render (question, options) in one canonical inner format.
    fmt='generation' -> MedExQA training format (Options + explain instruction)
    fmt='mcqa'       -> Choices + 'Answer:'"""
    opts = "\n".join(f"{k}. {v}" for k, v in parsed["options"].items())
    ctx = f"Context: {parsed['context']}\n" if parsed.get("context") else ""
    if fmt == "generation":
        return f"{ctx}Question: {parsed['question']}\nOptions:\n{opts}\n\nExplain the reasoning for the correct answer."
    if fmt == "mcqa":
        return f"{ctx}Question: {parsed['question']}\nChoices:\n{opts}\nAnswer:"
    raise ValueError(f"unknown inner format {fmt!r}")


def canonicalise_inner(inner_text, fmt):
    """Re-render an example's inner prompt in `fmt`; fmt='native' returns it
    unchanged. Raises if the text cannot be parsed (so a silent fall-back can
    never re-introduce the format confound)."""
    if fmt == "native":
        return inner_text
    parsed = parse_mcqa_question(inner_text)
    if parsed is None:
        raise ValueError(f"cannot parse inner prompt for canonicalisation: {inner_text[:120]!r}")
    return render_inner(parsed, fmt)
