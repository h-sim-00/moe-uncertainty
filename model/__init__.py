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
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map)

def load_peft_model(model_shortcode: str, finetune_mode: str, r: int = 64, lora_dropout: float = 0.01, target_layer: int | None = None, device_map="cuda:0") -> PreTrainedModel:
    """Loads the base model and applies LoRA configuration."""
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
    elif finetune_mode == 'qkv':
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
    peft_model.print_trainable_parameters()
    
    return peft_model

def load_peft_model_and_adapter(model_shortcode: str, adapter_path: str, eval_mode: bool = True, device_map="cuda:0") -> PeftModel:
    """Loads the base model and applies the trained LoRA adapter."""
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."
    print(f"Loading base model: {model_shortcode}")
    base_model = load_model(model_shortcode, device_map=device_map)
    
    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    
    if eval_mode:
        peft_model.eval()

    return peft_model