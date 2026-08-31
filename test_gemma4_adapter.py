"""CPU unit test for model/adapters/gemma4_adapter.py on a TINY random Gemma 4 text
model (branch OBQA-gemma). No checkpoint, no GPU: runs in seconds wherever
transformers>=5.5 + peft are installed (Isambard qwen_env login node is fine).

    python test_gemma4_adapter.py

Checks
  1. BayesianGemma4Router(MoERouter) reproduces the native Gemma4TextRouter
     (probabilities, top-k experts, routing weights incl. per_expert_scale) in fp32
  2. full-model logits are unchanged by ensure_containers; moe_router(layer) finds
     the inner router on every layer and its output[-1] is [N, E]; set_moe_router works
  3. MAP swap: exactly the inner routers' Linear is trainable; save_map/load_map round-trip
  4. FCVR: prepare on a layer subset trains only the variational heads, the ELBO
     backward reaches them and nothing else, save_bayes/load_bayes round-trip
     reproduces the heads bit-for-bit, deterministic read-out is finite
  5. LoRA qkv wrap: v_proj is absent on the global (attention_k_eq_v) layer and
     PEFT still wraps q/k there
The model-level checks (real checkpoint, bf16, tokenizer) are in smoke-gemma4-model.py.
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from model import LORA_QKV_TARGETS
from model.adapters import get_adapter, moe_router, set_moe_router
from model.adapters.gemma4_adapter import BayesianGemma4Router, router_config_for, ensure_containers
from model.routers.base import MoERouter
from model.routers.fcvr import FullCovarianceVariationalRouter

FAILS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def tiny_config():
    return Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
        head_dim=16, global_head_dim=32, num_experts=8, top_k_experts=2,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"], sliding_window=8,
        max_position_embeddings=64, enable_moe_block=True, hidden_size_per_layer_input=0,
        final_logit_softcapping=30.0, attention_k_eq_v=True, tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )


def dense(idx, w, E):
    return torch.zeros(idx.shape[0], E).scatter(1, idx.long(), w.float())


def build():
    torch.manual_seed(0)
    cfg = tiny_config()
    base = Gemma4ForCausalLM(cfg).eval()
    # make the frozen router pre-processing non-trivial so a missing factor would show
    with torch.no_grad():
        for layer in base.model.layers:
            layer.router.scale.normal_(1.0, 0.3)
            layer.router.per_expert_scale.uniform_(0.5, 1.5)
            layer.router.proj.weight.normal_(0, 0.5)
    peft = get_peft_model(base, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
                                           target_modules=list(LORA_QKV_TARGETS["gemma4"])))
    peft.eval()
    return cfg, peft


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    cfg, peft = build()
    causal = peft.base_model.model.model
    E, k = cfg.num_experts, cfg.top_k_experts
    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    attn = torch.ones_like(ids)

    # ---- 1. router equivalence -------------------------------------------------
    print("\n[1] router equivalence (fp32, layer 0)")
    native = causal.layers[0].router
    x = torch.randn(24, cfg.hidden_size)
    with torch.no_grad():
        ref_p, ref_w, ref_i = native(x)
        cont = BayesianGemma4Router(cfg, native, MoERouter)
        p, w, i = cont(x)
        u = cont.router_input(x)
        _, _, _, _, logits = cont.router(u)
    check(torch.allclose(logits, F.linear(u, native.proj.weight.float()), atol=1e-6), "container logits == F.linear(u, proj.W)")
    check(torch.allclose(p, ref_p, atol=1e-6), "router probabilities equal")
    check((i.sort(-1).values == ref_i.sort(-1).values).all().item(), "identical top-k experts")
    check(torch.allclose(dense(i, w, E), dense(ref_i, ref_w, E), atol=1e-6), "routing weights equal (incl. per_expert_scale)")
    rc = router_config_for(cfg)
    check(rc.num_local_experts == E and rc.num_experts_per_tok == k and rc.hidden_size == cfg.hidden_size, "router_config_for maps names")

    # ---- 2. containers on the full model -----------------------------------------
    print("\n[2] ensure_containers on the PEFT-wrapped model")
    with torch.no_grad():
        before = peft(input_ids=ids, attention_mask=attn).logits
        ensure_containers(peft)
        ensure_containers(peft)                                   # idempotent
        check(all(isinstance(l.router, BayesianGemma4Router) for l in causal.layers), "container on every layer (idempotent)")
        check(all(isinstance(moe_router(l), MoERouter) for l in causal.layers), "moe_router(layer) -> inner MoERouter")
        shapes = {}
        hs = [moe_router(l).register_forward_hook(lambda m, inp, out, li=li: shapes.__setitem__(li, tuple(out[-1].shape)))
              for li, l in enumerate(causal.layers)]
        after = peft(input_ids=ids, attention_mask=attn).logits
        for h in hs:
            h.remove()
    check(torch.allclose(before, after, atol=1e-5), f"full-model logits unchanged (max |diff| {(before-after).abs().max().item():.2e})")
    check(len(shapes) == cfg.num_hidden_layers and all(s == (ids.numel(), E) for s in shapes.values()),
          f"router hook output[-1] is [N={ids.numel()}, {E}] on all layers")
    probe = MoERouter(rc, existing_router=moe_router(causal.layers[1]))
    set_moe_router(causal.layers[1], probe)
    check(moe_router(causal.layers[1]) is probe, "set_moe_router replaces the inner router")

    # ---- 3. MAP swap + save/load ---------------------------------------------------
    print("\n[3] MAP swap / save / load")
    adapter = get_adapter("gemma4")
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            adapter.swap(peft)
            trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
            check(len(trainable) == cfg.num_hidden_layers and all(n.endswith(".router.router.layer.weight") for n in trainable),
                  f"MAP: only the {cfg.num_hidden_layers} inner router Linears are trainable ({sorted(trainable)[0]})")
            args = argparse.Namespace(model_shortcode="gemma4", dataset_shortcode="tiny", map_suffix="t")
            with torch.no_grad():
                moe_router(causal.layers[2]).layer.weight.add_(0.123)
            adapter.save_map(peft, args)
            check(os.path.exists("router_weights/base/gemma4_tiny-t/layer_2_weights.pt"), "MAP weights saved under router_weights/base/gemma4_tiny-t")
            saved = moe_router(causal.layers[2]).layer.weight.clone()
            cfg2, peft2 = build()
            adapter.load_map(peft2, args)
            check(torch.equal(moe_router(peft2.base_model.model.model.layers[2]).layer.weight, saved), "load_map round-trip bit-exact")

            # ---- 4. FCVR ---------------------------------------------------------------
            print("\n[4] FCVR prepare / ELBO backward / save / load")
            layers = [1, 2]
            fargs = argparse.Namespace(model_shortcode="gemma4", dataset_shortcode="tiny", swap_layers=layers, load_layers=[],
                                       train_layers=layers, run_suffix="rs")
            adapter.prepare(peft2, method="fcvr", args=fargs)
            c2 = peft2.base_model.model.model
            check(all(isinstance(moe_router(c2.layers[l]), FullCovarianceVariationalRouter) for l in layers)
                  and isinstance(moe_router(c2.layers[0]), MoERouter) and not isinstance(moe_router(c2.layers[0]), FullCovarianceVariationalRouter),
                  "FCVR routers on the swap layers only")
            tr = {n for n, p in peft2.named_parameters() if p.requires_grad}
            check(tr and all((".router.router.backbone." in n or ".router.router.mean_head." in n or ".router.router.cholesky_head." in n) for n in tr)
                  and all(f".layers.{l}." in n for n in tr for l in [int(n.split('.layers.')[1].split('.')[0])]) and
                  {int(n.split(".layers.")[1].split(".")[0]) for n in tr} == set(layers),
                  f"only backbone/mean_head/cholesky_head of layers {layers} trainable ({len(tr)} tensors)")
            peft2.train()
            labels = ids.clone(); labels[:, :-3] = -100
            out = peft2(input_ids=ids, attention_mask=attn, labels=labels)
            kl = sum(moe_router(c2.layers[l]).kl_divergence(mask=attn.reshape(-1)) for l in layers)
            (out.loss + 0.01 * kl).backward()
            grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in peft2.named_parameters() if p.requires_grad)
            no_leak = all(p.grad is None for n, p in peft2.named_parameters() if not p.requires_grad)
            check(grads_ok and no_leak, f"ELBO backward: finite grads on every trainable head, none elsewhere (loss {out.loss.item():.3f}, kl {kl.item():.3f})")
            adapter.save_bayes(peft2, method="fcvr", args=fargs)
            check(all(os.path.exists(f"router_weights/fcvr/fcvr-gemma4-tiny-rs/layer_{l}_weights.pt") for l in layers),
                  "FCVR weights saved under router_weights/fcvr/fcvr-gemma4-tiny-rs")
            ref_heads = {l: {n: p.detach().clone() for n, p in moe_router(c2.layers[l]).named_parameters()} for l in layers}
            cfg3, peft3 = build()
            largs = argparse.Namespace(model_shortcode="gemma4", dataset_shortcode="tiny", swap_layers=layers, run_suffix="rs")
            adapter.load_bayes(peft3, method="fcvr", args=largs)
            c3 = peft3.base_model.model.model
            same = all(torch.equal(dict(moe_router(c3.layers[l]).named_parameters())[n], t)
                       for l in layers for n, t in ref_heads[l].items() if not n.startswith("mean_base"))
            check(same, "load_bayes round-trip: backbone/mean_head/cholesky_head bit-exact")
            for l in layers:
                moe_router(c3.layers[l]).deterministic_readout = True
            peft3.eval()
            with torch.no_grad():
                lg = peft3(input_ids=ids, attention_mask=attn).logits
                L = moe_router(c3.layers[2]).last_cholesky_factor
            check(torch.isfinite(lg).all().item() and L.shape == (ids.numel(), E, E), f"deterministic FCVR read-out finite, Cholesky {tuple(L.shape)}")
            missing = argparse.Namespace(model_shortcode="gemma4", dataset_shortcode="tiny", swap_layers=[0], run_suffix="rs")
            try:
                adapter.load_bayes(peft3, method="fcvr", args=missing); check(False, "load_bayes raises on a missing layer file")
            except FileNotFoundError:
                check(True, "load_bayes raises FileNotFoundError on a missing layer file")
        finally:
            os.chdir(cwd)

    # ---- 5. LoRA targets ----------------------------------------------------------
    print("\n[5] LoRA qkv targets on the tiny model")
    parents = {n.rsplit(".lora_A", 1)[0] for n, _ in peft.named_parameters() if ".lora_A" in n}
    glob = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    check(all(causal.layers[g].self_attn.v_proj is None for g in glob), f"global layers {glob} have no v_proj (attention_k_eq_v)")
    check({int(n.split(".layers.")[1].split(".")[0]) for n in parents} == set(range(cfg.num_hidden_layers)), "LoRA reaches every layer")
    check(sum(n.endswith("v_proj") for n in parents) == cfg.num_hidden_layers - len(glob), "v_proj LoRA only on sliding layers")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
