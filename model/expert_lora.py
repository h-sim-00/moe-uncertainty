# model/expert_lora.py
"""
LoRA for the MoE *expert* networks (paper Appendix D.2, Stage 1).

The paper's Stage-1 deterministic MAP adaptation applies LoRA adapters to
"the attention modules (Q/K/V projections) and the Expert networks". PEFT can
wrap the Q/K/V `nn.Linear`s directly, but Granite keeps its experts in
`GraniteMoeParallelExperts`: a custom module holding a single 3-D
`nn.Parameter` of shape [num_experts, output_size, input_size] that is applied
per-expert with `F.linear` inside a loop. PEFT has no injector for that module
type, so this file provides one.

`ParallelExpertsLoRA` wraps a `GraniteMoeParallelExperts` layer, keeps the base
weight frozen, and adds an independent low-rank update to every expert:

    y_e = x_e W_e^T + (alpha / r) * ((dropout(x_e) A_e^T) B_e^T)

with A_e in R^{r x input_size} and B_e in R^{output_size x r}. B is zero-init,
so the wrapped layer is exactly the pre-trained layer at step 0.

Naming note: the factors are deliberately NOT called `lora_A` / `lora_B`.
PEFT's `get_peft_model_state_dict` selects adapter tensors with `"lora_" in
key`, so those names would be swept into the PEFT adapter checkpoint (and then
rejected as unexpected keys on load). Expert-LoRA weights are saved separately
via `save_expert_lora` and restored via `load_expert_lora`.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Written next to the PEFT adapter inside the Stage-1 adapter directory.
EXPERT_LORA_FILENAME = "expert_lora.pt"

# The two GraniteMoeParallelExperts instances inside a GraniteMoeMoE block.
EXPERT_MODULE_NAMES = ("input_linear", "output_linear")


class ParallelExpertsLoRA(nn.Module):
    """LoRA around a `GraniteMoeParallelExperts` layer (one adapter per expert)."""

    def __init__(self, base_layer: nn.Module, r: int = 64, lora_alpha: int = 16,
                 lora_dropout: float = 0.01):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = base_layer.num_experts
        self.input_size = base_layer.input_size
        self.output_size = base_layer.output_size
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout_p = lora_dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        weight = base_layer.weight
        factory = {"dtype": weight.dtype, "device": weight.device}

        # A: [E, r, input_size], Kaiming-uniform per expert (PEFT's LoRA init).
        a_init = torch.empty(self.num_experts, r, self.input_size, **factory)
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(a_init[e], a=math.sqrt(5))
        self.expert_a = nn.Parameter(a_init)

        # B: [E, output_size, r], zero-init so the update starts at exactly 0.
        self.expert_b = nn.Parameter(
            torch.zeros(self.num_experts, self.output_size, r, **factory)
        )

    def forward(self, inputs, expert_size):
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            x = input_list[i]
            base_out = F.linear(x, self.base_layer.weight[i])
            delta = F.linear(F.linear(self.dropout(x), self.expert_a[i]), self.expert_b[i])
            output_list.append(base_out + delta * self.scaling)
        return torch.cat(output_list, dim=0)

    def extra_repr(self) -> str:
        return (f"num_experts={self.num_experts}, r={self.r}, "
                f"lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout_p}")


def _get_decoder(model):
    """Return the module that owns `.layers`, for a raw or PEFT-wrapped model."""
    for path in (("base_model", "model", "model"), ("model",), ()):
        obj = model
        for attr in path:
            if not hasattr(obj, attr):
                obj = None
                break
            obj = getattr(obj, attr)
        if obj is not None and hasattr(obj, "layers"):
            return obj
    raise AttributeError("Could not locate the decoder stack (.layers) on the given model.")


def _moe_block(layer):
    """Return the MoE block of a decoder layer, or None if it has none."""
    return getattr(layer, "block_sparse_moe", None)


def inject_expert_lora(model, r: int = 64, lora_alpha: int = 16,
                       lora_dropout: float = 0.01, layers=None, verbose: bool = True):
    """Wrap every expert matrix with `ParallelExpertsLoRA`, in place.

    Args:
        layers: optional iterable of layer indices; defaults to all MoE layers.
    Returns:
        The number of trainable expert-LoRA parameters that were added.
    """
    decoder = _get_decoder(model)
    target_layers = range(len(decoder.layers)) if layers is None else sorted(set(layers))

    added, wrapped = 0, 0
    for idx in target_layers:
        block = _moe_block(decoder.layers[idx])
        if block is None:
            continue
        for name in EXPERT_MODULE_NAMES:
            module = getattr(block, name, None)
            if module is None or isinstance(module, ParallelExpertsLoRA):
                continue
            lora_module = ParallelExpertsLoRA(
                module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            setattr(block, name, lora_module.to(module.weight.device))
            added += lora_module.expert_a.numel() + lora_module.expert_b.numel()
            wrapped += 1

    if verbose:
        print(f"--- Expert LoRA: wrapped {wrapped} expert matrices "
              f"(r={r}, alpha={lora_alpha}, dropout={lora_dropout}) "
              f"| {added:,} trainable params added ---")
    return added


def iter_expert_lora_modules(model):
    """Yield (key, module) for every injected ParallelExpertsLoRA."""
    decoder = _get_decoder(model)
    for idx, layer in enumerate(decoder.layers):
        block = _moe_block(layer)
        if block is None:
            continue
        for name in EXPERT_MODULE_NAMES:
            module = getattr(block, name, None)
            if isinstance(module, ParallelExpertsLoRA):
                yield f"layers.{idx}.block_sparse_moe.{name}", module


def expert_lora_parameters(model):
    """All trainable expert-LoRA tensors (useful for building an optimizer)."""
    params = []
    for _, module in iter_expert_lora_modules(model):
        params.extend([module.expert_a, module.expert_b])
    return params


def save_expert_lora(model, path: str):
    """Save the expert-LoRA factors plus the config needed to rebuild them."""
    modules = dict(iter_expert_lora_modules(model))
    if not modules:
        print("--- Expert LoRA: nothing to save (no wrapped expert matrices) ---")
        return

    first = next(iter(modules.values()))
    payload = {
        "config": {
            "r": first.r,
            "lora_alpha": first.lora_alpha,
            "lora_dropout": first.lora_dropout_p,
        },
        "state_dict": {},
    }
    for key, module in modules.items():
        payload["state_dict"][f"{key}.expert_a"] = module.expert_a.detach().cpu()
        payload["state_dict"][f"{key}.expert_b"] = module.expert_b.detach().cpu()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"Saved expert-LoRA weights for {len(modules)} expert matrices to {path}")


def load_expert_lora(model, path: str, device=None):
    """Inject expert LoRA with the saved config, then load the saved factors."""
    payload = torch.load(path, map_location="cpu")
    config = payload["config"]
    state_dict = payload["state_dict"]

    # Only inject into the layers that actually have saved weights.
    saved_layers = sorted({int(k.split(".")[1]) for k in state_dict})
    inject_expert_lora(
        model,
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        layers=saved_layers,
        verbose=False,
    )

    modules = dict(iter_expert_lora_modules(model))
    missing = [k for k in modules if f"{k}.expert_a" not in state_dict]
    if missing:
        raise KeyError(f"Expert-LoRA checkpoint {path} is missing weights for: {missing}")

    for key, module in modules.items():
        target = module.expert_a.device
        with torch.no_grad():
            module.expert_a.copy_(state_dict[f"{key}.expert_a"].to(target, module.expert_a.dtype))
            module.expert_b.copy_(state_dict[f"{key}.expert_b"].to(target, module.expert_b.dtype))

    print(f"Loaded expert-LoRA weights for {len(modules)} expert matrices from {path} "
          f"(r={config['r']}, alpha={config['lora_alpha']})")
    return model


def expert_lora_path(adapter_path: str) -> str:
    return os.path.join(adapter_path, EXPERT_LORA_FILENAME)


def has_expert_lora(adapter_path: str) -> bool:
    return os.path.isfile(expert_lora_path(adapter_path))
