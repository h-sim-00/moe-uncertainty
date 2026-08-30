"""Per-architecture adapters + the two accessors every script should use.

Granite keeps its MoE block at `layer.block_sparse_moe` (router = `.router`);
the Qwen containers (`qwen_adapter`, `qwen36_adapter`) replace `layer.mlp` and
expose the same `.router`. `moe_router(layer)` hides that difference so the
evaluators / analysis scripts do not hardcode `block_sparse_moe.router`.

`get_adapter(model_shortcode)` returns the six swap/save/load functions of one
architecture under uniform names (lazy imports: importing this package never
pulls in transformers>=5-only code).
"""
from types import SimpleNamespace


def moe_router(layer):
    """The (possibly Bayesian) router module of a decoder layer."""
    block = getattr(layer, "block_sparse_moe", None)
    if block is None:
        block = getattr(layer, "mlp", None)
    router = getattr(block, "router", None)
    if router is None:
        raise AttributeError(
            "no MoE router on this decoder layer (Granite: layer.block_sparse_moe.router; "
            "Qwen: layer.mlp.router -- run the adapter's container swap first)")
    return router


def set_moe_router(layer, router):
    block = getattr(layer, "block_sparse_moe", None)
    if block is None:
        block = getattr(layer, "mlp", None)
    if block is None or not hasattr(block, "router"):
        raise AttributeError("decoder layer has no MoE block with a .router to replace")
    block.router = router


def get_adapter(model_shortcode):
    """-> SimpleNamespace(swap, save_map, load_map, prepare, save_bayes, load_bayes,
    router_config). `router_config(config)` maps the model config onto what the
    router classes read (identity for Granite)."""
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
    raise KeyError(f"no router adapter registered for model_shortcode={model_shortcode!r} "
                   f"(granite | qwen36)")
