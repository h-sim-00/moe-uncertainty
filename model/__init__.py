from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_SHORTCODE2ID = {
    "granite": "ibm-granite/granite-3.1-3b-a800m-instruct",
    "deepseek": "deepseek-ai/deepseek-moe-16b-chat",
    "qwen": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
}

def load_tokenizer(model_shortcode: str):
    """
    Load a tokenizer based on the provided model_shortcode.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode])

def load_model(model_shortcode: str, device_map: str = "cuda:0"):
    """
    Load a model basmodel_shortcodee provided model_id.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."

    if "granite" in model_shortcode:
        from .granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
        return GraniteMoeForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map)
    elif "deepseek" in model_shortcode:
        from .deepseekmoe.modeling_deepseek import DeepseekForCausalLM
        return DeepseekForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map, trust_remote_code=True)
    elif "qwen" in model_shortcode:
        from .qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
        return Qwen2MoeForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map, trust_remote_code=True)
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map)

def load_peft_model(model_shortcode: str, finetune_mode: str, r: int = 64, lora_dropout: float = 0.01, target_layer: int | None = None, device_map="cuda:0", expert_lora_r: int | None = None) -> PreTrainedModel:
    """Loads the base model and applies LoRA configuration.

    finetune_mode='qkv_experts' is the paper-faithful Stage-1 setting (App. D.2:
    "LoRA adapters are applied to the attention modules (Q/K/V projections) and
    the Expert networks"). PEFT handles Q/K/V; the experts live in a custom
    3-D-parameter module that PEFT cannot wrap, so they are adapted by
    model.expert_lora after the PEFT wrap.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."

    base_model = load_model(model_shortcode, device_map=device_map)
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Define LoRA target modules based on the finetune mode
    if finetune_mode == 'router':
        if target_layer is not None:
            print(f"Applying LoRA specifically to the router of layer: {target_layer}")
            target_modules = [f"model.layers.{target_layer}.block_sparse_moe.router.layer"]
        else:
            print("Applying LoRA to routers of ALL layers.")
            target_modules = ["router.layer"] # This uses regex-like matching for module names
    elif finetune_mode in ('qkv', 'qkv_experts'):
        # Your existing logic for targeting attention layers
        target_modules = ["q_proj", "k_proj", "v_proj"]
    else:
        raise ValueError(f"Invalid finetune_mode: {finetune_mode}")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=lora_dropout,
        r=r,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )

    peft_model = get_peft_model(base_model, peft_config)

    if finetune_mode == 'qkv_experts':
        assert "granite" in model_shortcode, \
            "finetune_mode='qkv_experts' is only implemented for Granite MoE."
        from .expert_lora import inject_expert_lora
        # Injected after get_peft_model, which freezes every base parameter;
        # the newly created expert-LoRA factors are trainable by default.
        inject_expert_lora(
            peft_model,
            r=expert_lora_r if expert_lora_r is not None else r,
            lora_alpha=16,
            lora_dropout=lora_dropout,
        )

    peft_model.print_trainable_parameters()

    return peft_model

def load_peft_model_and_adapter(model_shortcode: str, adapter_path: str, eval_mode: bool = True, device_map="cuda:0") -> PeftModel:
    """Loads the base model and applies the trained LoRA adapter.

    If the adapter directory also contains expert-LoRA weights (Stage 1 trained
    with finetune_mode='qkv_experts'), they are injected and loaded too, so
    every downstream stage sees the same Stage-1 model.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."
    print(f"Loading base model: {model_shortcode}")
    base_model = load_model(model_shortcode, device_map=device_map)

    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

    from .expert_lora import EXPERT_LORA_FILENAME, has_expert_lora, expert_lora_path, load_expert_lora
    if has_expert_lora(adapter_path):
        print(f"Found expert-LoRA weights in {adapter_path}; loading them.")
        load_expert_lora(peft_model, expert_lora_path(adapter_path))
    else:
        print(f"No {EXPERT_LORA_FILENAME} in {adapter_path}; experts stay at their pre-trained weights.")

    if eval_mode:
        peft_model.eval()

    return peft_model