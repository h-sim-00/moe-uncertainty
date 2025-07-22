from .granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def load_model(model_id: str, device_map: str = "auto"):
    """
    Load a model based on the provided model_id.
    """
    if "granite" in model_id:
        return GraniteMoeForCausalLM.from_pretrained(model_id, device_map=device_map)
    else:
        return AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)

def load_peft_model(model_id: str, finetune_mode: str, r: int = 64, lora_dropout: float = 0.1, target_layer: int | None = None) -> PreTrainedModel:
    """Loads the base model and applies LoRA configuration."""
    base_model = load_model(model_id)
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

def load_peft_model_and_adapter(model_id: str, adapter_path: str, eval_mode: bool = True) -> PeftModel:
    """Loads the base model and applies the trained LoRA adapter."""

    print(f"Loading base model: {model_id}")
    base_model = load_model(model_id)
    
    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    
    if eval_mode:
        peft_model.eval()

    return peft_model