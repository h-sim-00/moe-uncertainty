"""Model-layer smoke test for the gemma4 (Gemma 4 26B-A4B-it) port -- branch OBQA-gemma.

Runs on ONE GPU in a few minutes (needs the ~52 GB checkpoint in HF_HOME):
  1. load `gemma4` via model.load_model: text-only Gemma4ForCausalLM on the
     multimodal checkpoint (explicit key_mapping; load_model raises if any text
     weight is missing), Gemma4TextConfig, 30 layers, bf16, MoE on every layer,
     E=128 top-8
  2. tokenizer / prompt: eos is <turn|> (pad differs), A..E single tokens, the
     assistant header ends "<|turn>model\n<|channel>thought\n<channel|>", prompt+"A"
     keeps the bare letter token, "\nExplanation:" marker split, exactly ONE <bos>
     whether or not add_special_tokens is passed
  3. router equivalence on layer 0: native Gemma4TextRouter vs
     BayesianGemma4Router(MoERouter) -- container logits == F.linear(u, proj.W),
     identical top-k / weights on rows without bf16 near-ties; fresh FCVR router
     (deterministic_readout) ~ equal; Cholesky factor [N, 128, 128]
  4. full-model logits before/after ensure_containers within bf16 noise
  5. PEFT qkv targets: 30 layers covered, only under self_attn., v_proj LoRA on
     the 25 sliding-window layers (the 5 global layers have no v_proj)
  6. router hooks: output[-1] of every layer's inner router is [N, 128]
  7. FCVR ELBO grads equal with/without gradient checkpointing (depth layer set)
  8. greedy generation from the MCQ prompt: first new token is a letter and the
     model stops at <turn|> within 8 tokens

    python smoke-gemma4-model.py            # all checks
    python smoke-gemma4-model.py --quick    # skip the full-model checks (4, 7, 8)
"""
import argparse
import os
import sys

# The smoke controls activation checkpointing itself (check 7 compares both
# modes); the sbatch wrapper exports GRADIENT_CHECKPOINTING=1 for training.
os.environ["GRADIENT_CHECKPOINTING"] = "0"

import torch
import torch.nn.functional as F

from utils import setup_environment
from model import MODEL_SHORTCODE2ID, load_model, load_tokenizer, load_peft_model
from model.adapters import get_adapter, moe_router
from model.adapters.gemma4_adapter import BayesianGemma4Router, router_config_for
from model.routers.base import MoERouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from utils.prompt import multiple_choice_prompt_engineer, MCQ_SYSTEM_INSTRUCTION

FCVR_LAYERS = [5, 6, 7, 8, 18, 19, 26, 27, 28, 29]     # the 'depth' set (medmcqa-arms-lib.sh, gemma4)
HEADER_TAIL = "<|turn>model\n<|channel>thought\n<channel|>"
FAILS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def dense_weights(idx, w, E):
    """[N, k] (indices, weights) -> [N, E] dense routing weights (order-independent)."""
    out = torch.zeros(idx.shape[0], E, device=idx.device, dtype=torch.float32)
    return out.scatter(1, idx.long(), w.float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    setup_environment()
    torch.manual_seed(0)

    # ---- 1. load -------------------------------------------------------------
    print("\n[1] loading", MODEL_SHORTCODE2ID["gemma4"])
    model = load_model("gemma4", device_map=args.device)
    cfg = model.config
    check(type(cfg).__name__ == "Gemma4TextConfig", f"config class {type(cfg).__name__}")
    check(len(model.model.layers) == 30, f"{len(model.model.layers)} layers")
    check(next(model.parameters()).dtype == torch.bfloat16, "bf16 weights")
    check(cfg.num_experts == 128 and cfg.top_k_experts == 8, f"E={cfg.num_experts} top_k={cfg.top_k_experts}")
    check(all(getattr(l, "enable_moe_block", False) for l in model.model.layers), "MoE block enabled on every layer")
    check(all(hasattr(l, "router") and hasattr(l, "experts") and hasattr(l, "mlp") for l in model.model.layers),
          "every layer has router / experts / (dense) mlp")
    print("  experts implementation:", getattr(cfg, "_experts_implementation", "?"),
          "| attn:", getattr(cfg, "_attn_implementation", "?"),
          "| lm_head tied:", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
    n_total = sum(p.numel() for p in model.parameters())
    check(20e9 < n_total < 30e9, f"{n_total/1e9:.2f} B params (text-only)")
    # A randomly initialised decoder would have near-uniform routers; the pre-trained proj is not.
    w0 = model.model.layers[0].router.proj.weight.float()
    check(w0.abs().mean().item() > 1e-4 and w0.std().item() > 1e-4, f"layer-0 router proj looks loaded (std {w0.std().item():.3e})")

    # ---- 2. tokenizer / prompt ---------------------------------------------
    print("\n[2] tokenizer / prompt")
    tok = load_tokenizer("gemma4")
    check(tok.eos_token == "<turn|>" and tok.eos_token_id == tok.convert_tokens_to_ids("<turn|>"),
          f"eos={tok.eos_token!r} ({tok.eos_token_id})")
    check(tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id,
          f"pad={tok.pad_token!r} ({tok.pad_token_id}) differs from eos")
    ids = [tok.convert_tokens_to_ids(c) for c in "ABCDE"]
    check(all(i is not None and i != tok.unk_token_id for i in ids), f"A..E single tokens {ids}")
    prompt = multiple_choice_prompt_engineer(
        {"question": "Question: What color is the sky?\nChoices:\nA. blue\nB. green\nC. red\nD. black\nAnswer:",
         "answer": "A", "id": "x"}, tokenizer=tok, system_instruction=MCQ_SYSTEM_INSTRUCTION)["question"]
    check(prompt.endswith(HEADER_TAIL), f"prompt tail {prompt[-60:]!r}")
    sep = tok(prompt, add_special_tokens=False).input_ids
    check(sep[0] == tok.bos_token_id and sep.count(tok.bos_token_id) == 1, "exactly one <bos> (from the chat template)")
    check(tok(prompt).input_ids == sep, "add_special_tokens default adds no second <bos>")
    joint = tok(prompt + "A", add_special_tokens=False).input_ids
    check(joint[:len(sep)] == sep and joint[len(sep):] == [ids[0]], "prompt + 'A' tokenises as prompt ids + [A]")
    marker = tok("\nExplanation:", add_special_tokens=False).input_ids
    joint2 = tok(prompt + "A\nExplanation:", add_special_tokens=False).input_ids
    check(joint2 == sep + [ids[0]] + marker, f"prompt + 'A\\nExplanation:' == prompt + [A] + marker {marker}")
    joint3 = tok(prompt + "A" + tok.eos_token, add_special_tokens=False).input_ids
    check(joint3 == sep + [ids[0], tok.eos_token_id], "prompt + 'A' + eos ends in the single <turn|> id")

    # ---- 3. router equivalence -------------------------------------------------
    print("\n[3] router equivalence (layer 0)")
    layer0 = model.model.layers[0]
    native = layer0.router
    E, k = cfg.num_experts, cfg.top_k_experts
    x = (torch.randn(64, cfg.hidden_size, device=args.device) * 0.5).to(torch.bfloat16)   # flat [N, H] as the layer passes
    with torch.no_grad():
        ref_probs, ref_w, ref_i = native(x)
        cont = BayesianGemma4Router(cfg, native, MoERouter).to(args.device)
        probs, w, i = cont(x)
        u = cont.router_input(x)
        ref_logits = F.linear(u, native.proj.weight.float())
        _, _, _, _, logits = cont.router(u)
        check(torch.allclose(logits, ref_logits, atol=1e-4, rtol=1e-4), "container router logits == F.linear(u, proj.W)")
        # rows where the bf16 (native) and fp32 (ours) projections agree on the top-k set
        u_bf = native.norm(x) * native.scale * native.scalar_root_size
        top_bf = native.proj(u_bf).float().topk(k, dim=-1).indices.sort(-1).values
        top_fp = ref_logits.topk(k, dim=-1).indices.sort(-1).values
        stable = (top_bf == top_fp).all(-1)
        check(stable.float().mean().item() > 0.9, f"{stable.float().mean().item():.3f} of rows have stable top-k")
        check((i.sort(-1).values[stable] == ref_i.sort(-1).values[stable]).all().item(), "identical top-k experts on stable rows")
        d = (dense_weights(i, w, E) - dense_weights(ref_i, ref_w, E)).abs()[stable]
        rel = d.max().item() / (ref_w.float().abs().mean().item() + 1e-6)
        check(rel < 5e-2, f"MAP container vs native routing weights: max|diff|/mean|w| = {rel:.3e} on stable rows")
        check(probs.shape == ref_probs.shape == (64, E), f"router probabilities shape {tuple(probs.shape)}")
        fc = FullCovarianceVariationalRouter(router_config_for(cfg), existing_router=cont.router).to(args.device)
        fc.deterministic_readout = True
        cont.router = fc
        _, w2, i2 = cont(x)
        d2 = (dense_weights(i2, w2, E) - dense_weights(i, w, E)).abs()
        check(d2.max().item() / (w.float().abs().mean().item() + 1e-6) < 5e-2,
              f"fresh FCVR (deterministic) vs MAP container: {d2.max().item()/(w.float().abs().mean().item()+1e-6):.3e}")
        check(fc.last_cholesky_factor.shape == (64, E, E), f"cholesky factor [N, {E}, {E}]")
    del cont, fc

    # ---- 5. PEFT targets ---------------------------------------------------
    print("\n[5] PEFT qkv targets")
    del model
    torch.cuda.empty_cache()
    peft_model = load_peft_model("gemma4", finetune_mode="qkv", device_map=args.device)
    lora_parents = {n.rsplit(".lora_A", 1)[0] for n, _ in peft_model.named_parameters() if ".lora_A" in n}
    layers_hit = {int(n.split(".layers.")[1].split(".")[0]) for n in lora_parents}
    check(len(layers_hit) == 30, f"LoRA on {len(layers_hit)} layers")
    check(all(".self_attn." in n for n in lora_parents), "LoRA only under self_attn.")
    n_v = sum(n.endswith("v_proj") for n in lora_parents)
    n_sliding = sum(t == "sliding_attention" for t in cfg.layer_types)
    check(n_v == n_sliding, f"v_proj LoRA on {n_v} layers (= {n_sliding} sliding-window layers; global layers have V=K)")
    n_tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  trainable {n_tr/1e6:.1f} M")

    # ---- 4/6. containers + hooks on the full model ---------------------------
    print("\n[4/6] containers + router hooks")
    peft_model.eval()
    causal = peft_model.base_model.model.model
    tok.padding_side = "left"
    inp = tok([prompt, prompt + "A\nExplanation: the sky"], return_tensors="pt", padding=True,
              add_special_tokens=False).to(args.device)
    with torch.no_grad():
        before = peft_model(**inp).logits.float() if not args.quick else None
        get_adapter("gemma4").ensure_containers(peft_model)
        check(all(isinstance(l.router, BayesianGemma4Router) for l in causal.layers), "containers on all 30 layers")
        shapes = {}
        hs = [moe_router(l).register_forward_hook(lambda m, i, o, li=li: shapes.__setitem__(li, tuple(o[-1].shape)))
              for li, l in enumerate(causal.layers)]
        after = peft_model(**inp).logits.float()
        for h in hs:
            h.remove()
    N = inp["input_ids"].numel()
    check(len(shapes) == 30 and all(s == (N, E) for s in shapes.values()),
          f"router hook output[-1] is [N={N}, {E}] on all 30 layers")
    if before is not None:
        mask = inp["attention_mask"].bool()
        dl = (after - before).abs()[mask].max().item()
        agree = (after.argmax(-1) == before.argmax(-1))[mask].float().mean().item()
        check(agree > 0.98, f"full-model logits before/after containers: max|diff| {dl:.3f}, argmax agreement {agree:.3f}")

    # ---- 8. generation ---------------------------------------------------------
    if not args.quick:
        print("\n[8] greedy generation (base model through the identity LoRA + MAP containers)")
        one = tok([prompt], return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            gen = peft_model.generate(**one, max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
                                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        new = gen[0, one["input_ids"].shape[1]:].tolist()
        print("  generated:", repr(tok.decode(new)))
        check(len(new) > 0 and new[0] in ids, f"first generated token is a letter ({tok.convert_ids_to_tokens(new[:1])})")
        check(tok.eos_token_id in new, "generation stopped at <turn|> within 8 tokens")

    # ---- 7. FCVR grads with/without gradient checkpointing -------------------
    if not args.quick:
        print("\n[7] FCVR ELBO grads with vs without gradient checkpointing")
        import argparse as _ap
        fargs = _ap.Namespace(model_shortcode="gemma4", dataset_shortcode="smoke", swap_layers=FCVR_LAYERS,
                              load_layers=[], train_layers=FCVR_LAYERS, run_suffix=None)
        get_adapter("gemma4").prepare(peft_model, method="fcvr", args=fargs)
        labels = inp["input_ids"].clone(); labels[:, :-4] = -100
        grads = {}
        for gc in (False, True):
            if gc:
                peft_model.enable_input_require_grads()
                peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            peft_model.train()
            for r in (moe_router(causal.layers[l]) for l in FCVR_LAYERS):
                for p in r.parameters():
                    p.grad = None
            torch.manual_seed(123)
            out = peft_model(**inp, labels=labels)
            kl = sum(moe_router(causal.layers[l]).kl_divergence(mask=inp["attention_mask"].reshape(-1)) for l in FCVR_LAYERS)
            (out.loss + 0.01 * kl).backward()
            grads[gc] = torch.cat([moe_router(causal.layers[l]).cholesky_head.weight.grad.flatten().float().cpu()
                                   for l in FCVR_LAYERS])
            print(f"  gc={gc}: loss {out.loss.item():.4f} kl {kl.item():.4f} |grad| {grads[gc].norm():.4e}")
        rel = (grads[True] - grads[False]).norm().item() / (grads[False].norm().item() + 1e-12)
        check(rel < 1e-2, f"cholesky_head grads agree with/without checkpointing (rel diff {rel:.2e})")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
