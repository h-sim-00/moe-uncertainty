import os

from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_SHORTCODE2ID = {
    "granite": "ibm-granite/granite-3.1-3b-a800m-instruct",
    "deepseek": "deepseek-ai/deepseek-moe-16b-chat",
    "qwen": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    # branch OBQA-qwen: Qwen3.6-35B-A3B (transformers>=5 `Qwen3_5MoeForCausalLM`;
    # hybrid Gated-DeltaNet/attention, 40 layers, 256 experts top-8 + shared expert).
    "qwen36": "Qwen/Qwen3.6-35B-A3B",
    # branch OBQA-gemma: Gemma 4 26B-A4B instruction-tuned (transformers>=5.5
    # `Gemma4ForCausalLM`; 30 layers, MoE in every layer: 128 experts top-8 +
    # a dense MLP in parallel; multimodal checkpoint loaded text-only).
    "gemma4": "google/gemma-4-26B-A4B-it",
}

# Stage-1 LoRA targets per model for finetune_mode in ('qkv', 'qkv_experts').
# Qwen3.6: q/k/v_proj exist only in the 10 full-attention layers; the 30 Gated
# DeltaNet layers project Q/K/V through the fused `in_proj_qkv` Linear, so it is
# included so every layer's QKV projection is adapted. Granite list unchanged.
# Gemma 4: q/k/v_proj on the 25 sliding-window layers; the 5 global layers have
# attention_k_eq_v (no v_proj module, V = K), so they get q/k LoRA only.
LORA_QKV_TARGETS = {
    "default": ["q_proj", "k_proj", "v_proj"],
    "qwen36": ["q_proj", "k_proj", "v_proj", "in_proj_qkv"],
    "gemma4": ["q_proj", "k_proj", "v_proj"],
}

# Gemma 4's tokenizer eos is "<eos>", but its chat template closes every turn
# with "<turn|>" (id 106) and the instruct model emits that token to end a
# turn (generation_config eos = [<eos>, <turn|>, ...]). Training targets and
# generation stops use tokenizer.eos_token, so the Gemma tokenizer is loaded
# with eos_token="<turn|>" -- the exact analogue of Qwen's "<|im_end|>".
GEMMA4_EOS_TOKEN = "<turn|>"


def _enable_gradient_checkpointing(peft_model):
    """Opt-in via GRADIENT_CHECKPOINTING=1: recomputes activations in backward,
    cutting activation memory (arm B on a 44-46 GiB card; every Qwen3.6 stage)
    at ~25-35% speed cost. Mathematically identical training."""
    if os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1":
        peft_model.config.use_cache = False
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("--- Gradient checkpointing ENABLED (GRADIENT_CHECKPOINTING=1) ---")

def load_tokenizer(model_shortcode: str):
    """
    Load a tokenizer based on the provided model_shortcode.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."
    from transformers import AutoTokenizer
    if model_shortcode == "gemma4":
        tok = AutoTokenizer.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], eos_token=GEMMA4_EOS_TOKEN)
        eos_id = tok.convert_tokens_to_ids(GEMMA4_EOS_TOKEN)
        assert tok.eos_token == GEMMA4_EOS_TOKEN and tok.eos_token_id == eos_id and eos_id != tok.unk_token_id, \
            f"gemma4 tokenizer: could not set eos to {GEMMA4_EOS_TOKEN!r} (got {tok.eos_token!r}/{tok.eos_token_id})"
        assert tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id, "gemma4 tokenizer: pad must differ from eos"
        return tok
    return AutoTokenizer.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode])

def load_model(model_shortcode: str, device_map: str = "cuda:0"):
    """
    Load a model basmodel_shortcodee provided model_id.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."

    if "granite" in model_shortcode:
        from .granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
        return GraniteMoeForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], device_map=device_map)
    elif model_shortcode == "qwen36":
        # Exact match BEFORE the `"qwen" in` substring branch (which would route
        # to the vendored Qwen1.5-MoE class). Text-only class: transformers maps
        # the multimodal checkpoint's `model.language_model.*` keys onto
        # `model.layers.*` and ignores `model.visual.*` / `mtp.*`, so the decoder
        # stack sits at model.model.layers (same depth as Granite under PEFT).
        import torch
        from transformers import Qwen3_5MoeForCausalLM   # transformers>=5 only (lazy: moe_env never imports it)
        kwargs = dict(dtype=torch.bfloat16, device_map=device_map, attn_implementation="sdpa")
        experts_impl = os.environ.get("QWEN_EXPERTS_IMPL")          # grouped_mm | batched_mm | eager; unset = HF default
        if experts_impl:
            kwargs["experts_implementation"] = experts_impl
        if os.environ.get("QWEN_USE_HUB_KERNELS", "0") == "1":      # optional Gated-DeltaNet Hub kernel
            kwargs.update(use_kernels=True, trust_remote_code=True)
        m = Qwen3_5MoeForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], **kwargs)
        # Never used (Granite's aux-loss coef is 0 too); after the router swap the
        # native OutputRecorder would see no Qwen3_5MoeTopKRouter anyway.
        m.config.output_router_logits = False
        return m
    elif model_shortcode == "gemma4":
        # Text-only class on the multimodal checkpoint. Unlike qwen3_5 (whose
        # `qwen3_5_text` conversion strips the `language_model` prefix),
        # transformers has NO conversion entry for `gemma4_text`, so without the
        # explicit key_mapping every decoder weight would be reported missing and
        # silently randomly initialised. The mapping renames
        # model.language_model.<x> -> model.<x>; vision/audio tower keys are
        # unexpected and dropped; lm_head is tied to embed_tokens (no checkpoint
        # key). Loading info is checked so a mapping regression fails loudly.
        import torch
        from transformers import Gemma4ForCausalLM   # transformers>=5.5 only (lazy: moe_env never imports it)
        kwargs = dict(dtype=torch.bfloat16, device_map=device_map, attn_implementation="sdpa",
                      key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        experts_impl = os.environ.get("GEMMA_EXPERTS_IMPL")        # grouped_mm | batched_mm | eager; unset = HF default
        if experts_impl:
            kwargs["experts_implementation"] = experts_impl
        out = Gemma4ForCausalLM.from_pretrained(MODEL_SHORTCODE2ID[model_shortcode], **kwargs)
        m, info = out if isinstance(out, tuple) else (out, {})
        info = dict(info) if info else {}
        missing = [k for k in info.get("missing_keys", []) if not str(k).startswith("lm_head.")]
        non_text_prefixes = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")
        unexpected = [k for k in info.get("unexpected_keys", []) if not str(k).startswith(non_text_prefixes)]
        if missing or unexpected:
            raise RuntimeError(
                f"gemma4 text-only load did not line up with the checkpoint: {len(missing)} missing text keys "
                f"(e.g. {missing[:3]}), {len(unexpected)} unexpected non-vision keys (e.g. {unexpected[:3]}). "
                f"Check the key_mapping against the transformers version.")
        print(f"--- gemma4: {len(info.get('unexpected_keys', []))} vision/audio checkpoint keys ignored; "
              f"text stack fully loaded ({len(m.model.layers)} layers) ---")
        return m
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
        target_modules = list(LORA_QKV_TARGETS.get(model_shortcode, LORA_QKV_TARGETS["default"]))
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

    _enable_gradient_checkpointing(peft_model)

    peft_model.print_trainable_parameters()

    return peft_model

def load_peft_model_and_adapter(model_shortcode: str, adapter_path: str, eval_mode: bool = True, device_map="cuda:0") -> PeftModel:
    """Loads the base model and applies the trained LoRA adapter.

    If the adapter directory also contains expert-LoRA weights (Stage 1 trained
    with finetune_mode='qkv_experts'), they are injected and loaded too, so
    every downstream stage sees the same Stage-1 model.
    """
    assert model_shortcode in MODEL_SHORTCODE2ID, f"Model shortcode '{model_shortcode}' not defined."
    if adapter_path is None:
        # Untuned model (zero-shot rows). PeftModel.from_pretrained(base, None)
        # crashes, so wrap the base model in a FRESH Q/K/V LoRA instead: LoRA B is
        # zero-initialised, so in eval mode the forward pass is exactly the base
        # model, while the object keeps the PeftModel structure
        # (model.base_model.model.model.layers[i].block_sparse_moe.router) that
        # every evaluator relies on.
        print(f"No adapter requested: loading base model {model_shortcode} with a fresh (identity) LoRA wrapper")
        peft_model = load_peft_model(model_shortcode, finetune_mode="qkv", device_map=device_map)
        if eval_mode:
            peft_model.eval()
        return peft_model
    print(f"Loading base model: {model_shortcode}")
    base_model = load_model(model_shortcode, device_map=device_map)

    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    # Same env-gated opt-in as load_peft_model, so the MAP / FCVR stages (which
    # load the Stage-1 adapter through this path, with the default eval_mode and
    # then call .train()) can checkpoint activations too. Inert in eval mode.
    _enable_gradient_checkpointing(peft_model)

    from .expert_lora import EXPERT_LORA_FILENAME, has_expert_lora, expert_lora_path, load_expert_lora
    if has_expert_lora(adapter_path):
        print(f"Found expert-LoRA weights in {adapter_path}; loading them.")
        load_expert_lora(peft_model, expert_lora_path(adapter_path))
    else:
        print(f"No {EXPERT_LORA_FILENAME} in {adapter_path}; experts stay at their pre-trained weights.")

    if eval_mode:
        peft_model.eval()

    return peft_model