"""Per-architecture adapters + the two accessors every script should use.

Granite keeps its MoE block at `layer.block_sparse_moe` (router = `.router`);
the Qwen containers (`qwen_adapter`, `qwen36_adapter`) replace `layer.mlp` and
expose the same `.router`; Gemma 4 (`gemma4_adapter`) keeps its router directly
on the decoder layer, so its container replaces `layer.router` and exposes the
inner router as `layer.router.router`. `moe_router(layer)` hides those
differences so the evaluators / analysis scripts do not hardcode any of them.

`get_adapter(model_shortcode)` returns the six swap/save/load functions of one
architecture under uniform names (lazy imports: importing this package never
pulls in transformers>=5-only code).
"""
from types import SimpleNamespace


def _moe_block(layer):
    """The module that OWNS the (possibly Bayesian) router: Granite
    layer.block_sparse_moe, Qwen layer.mlp (container), Gemma 4 layer.router
    (container; Gemma's layer.mlp is its dense MLP and has no `.router`).
    A native, un-wrapped Qwen/Gemma router has no `.router` -> None."""
    for name in ("block_sparse_moe", "mlp", "router"):
        block = getattr(layer, name, None)
        if block is not None and hasattr(block, "router"):
            return block
    return None


def moe_router(layer):
    """The (possibly Bayesian) router module of a decoder layer."""
    block = _moe_block(layer)
    if block is None:
        raise AttributeError(
            "no MoE router on this decoder layer (Granite: layer.block_sparse_moe.router; "
            "Qwen: layer.mlp.router; Gemma 4: layer.router.router -- run the adapter's container swap first)")
    return block.router


def set_moe_router(layer, router):
    block = _moe_block(layer)
    if block is None:
        raise AttributeError("decoder layer has no MoE block with a .router to replace")
    block.router = router


def get_adapter(model_shortcode):
    """-> SimpleNamespace(swap, save_map, load_map, prepare, save_bayes, load_bayes,
    router_config[, ensure_containers]). `router_config(config)` maps the model
    config onto what the router classes read (identity for Granite)."""
    if model_shortcode == "granite":
        from . import granite_adapter as a
        return SimpleNamespace(
            swap=a.swap_granite_moe_blocks, save_map=a.save_granite_map_routers,
            load_map=a.load_granite_map_routers, prepare=a.prepare_granite_bayesian_routers,
            save_bayes=a.save_granite_bayesian_routers, load_bayes=a.load_granite_bayesian_routers,
            router_config=lambda config: config,
        )
    if model_shortcode == "qwen36":
        from . import qwen36_adapter as a
        return SimpleNamespace(
            swap=a.swap_qwen36_moe_blocks, save_map=a.save_qwen36_map_routers,
            load_map=a.load_qwen36_map_routers, prepare=a.prepare_qwen36_bayesian_routers,
            save_bayes=a.save_qwen36_bayesian_routers, load_bayes=a.load_qwen36_bayesian_routers,
            router_config=a.router_config_for, ensure_containers=a.ensure_containers,
        )
    if model_shortcode == "gemma4":
        from . import gemma4_adapter as a
        return SimpleNamespace(
            swap=a.swap_gemma4_moe_blocks, save_map=a.save_gemma4_map_routers,
            load_map=a.load_gemma4_map_routers, prepare=a.prepare_gemma4_bayesian_routers,
            save_bayes=a.save_gemma4_bayesian_routers, load_bayes=a.load_gemma4_bayesian_routers,
            router_config=a.router_config_for, ensure_containers=a.ensure_containers,
        )
    raise KeyError(f"no router adapter registered for model_shortcode={model_shortcode!r} "
                   f"(granite | qwen36 | gemma4)")
