"""Model-layer smoke test for the qwen36 (Qwen3.6-35B-A3B) port -- branch OBQA-qwen.

Runs on ONE GPU in a few minutes (needs the 72 GB checkpoint in HF_HOME):
  1. load `qwen36` via model.load_model: text config, 40 layers, bf16, no missing keys
  2. tokenizer / prompt: assistant header ends with the empty <think> block,
     A..E are single tokens, "\n\n" + "A" keeps the bare letter token
  3. block equivalence on layer 0: native Qwen3_5MoeSparseMoeBlock vs
     BayesianQwen35MoeBlock(MoERouter) -- router logits == F.linear(x, gate.W),
     identical top-k, outputs allclose on rows without bf16 near-ties;
     and with a fresh FCVR router (deterministic_readout) ~ equal
  4. full-model logits before/after ensure_containers within bf16 noise
  5. PEFT qkv targets: 40 layers covered, only under self_attn./linear_attn.
  6. router hooks: output[-1] of every layer's router is [N, 256]
  7. FCVR ELBO grads equal with/without GRADIENT_CHECKPOINTING (RNG restored
     by the non-reentrant checkpoint)

    python smoke-qwen36-model.py            # all checks
    python smoke-qwen36-model.py --quick    # skip the two full-model checks (4, 7)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

from utils import setup_environment
from model import MODEL_SHORTCODE2ID, load_model, load_tokenizer, load_peft_model
from model.adapters import get_adapter, moe_router
from model.adapters.qwen36_adapter import BayesianQwen35MoeBlock, router_config_for
from model.routers.base import MoERouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from utils.prompt import multiple_choice_prompt_engineer, MCQ_SYSTEM_INSTRUCTION

FCVR_LAYERS = [5, 6, 7, 8, 19, 20, 28, 29, 30, 31]
FAILS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    setup_environment()
    torch.manual_seed(0)

    # ---- 1. load -------------------------------------------------------------
    print("\n[1] loading", MODEL_SHORTCODE2ID["qwen36"])
    model = load_model("qwen36", device_map=args.device)
    cfg = model.config
    check(type(cfg).__name__ == "Qwen3_5MoeTextConfig", f"config class {type(cfg).__name__}")
    check(len(model.model.layers) == 40, f"{len(model.model.layers)} layers")
    check(next(model.parameters()).dtype == torch.bfloat16, "bf16 weights")
    check(cfg.output_router_logits is False, "output_router_logits False")
    check(cfg.num_experts == 256 and cfg.num_experts_per_tok == 8, f"E={cfg.num_experts} top_k={cfg.num_experts_per_tok}")
    print("  experts implementation:", getattr(cfg, "_experts_implementation", "?"),
          "| attn:", getattr(cfg, "_attn_implementation", "?"))
    n_total = sum(p.numel() for p in model.parameters())
    check(30e9 < n_total < 40e9, f"{n_total/1e9:.2f} B params (text-only)")

    # ---- 2. tokenizer / prompt ---------------------------------------------
    print("\n[2] tokenizer / prompt")
    tok = load_tokenizer("qwen36")
    check(tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id,
          f"pad={tok.pad_token!r} eos={tok.eos_token!r}")
    ids = [tok.convert_tokens_to_ids(c) for c in "ABCDE"]
    check(all(i is not None and i != tok.unk_token_id for i in ids), f"A..E single tokens {ids}")
    prompt = multiple_choice_prompt_engineer(
        {"question": "Question: What color is the sky?\nChoices:\nA. blue\nB. green\nC. red\nD. black\nAnswer:",
         "answer": "A", "id": "x"}, tokenizer=tok, system_instruction=MCQ_SYSTEM_INSTRUCTION)["question"]
    check(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), f"prompt tail {prompt[-60:]!r}")
    joint = tok(prompt + "A", add_special_tokens=False).input_ids
    sep = tok(prompt, add_special_tokens=False).input_ids
    check(joint[:len(sep)] == sep and joint[len(sep):] == [ids[0]], "prompt + 'A' tokenises as prompt ids + [A]")
    marker = tok("\nExplanation:", add_special_tokens=False).input_ids
    joint2 = tok(prompt + "A\nExplanation:", add_special_tokens=False).input_ids
    check(joint2 == sep + [ids[0]] + marker, f"prompt + 'A\\nExplanation:' == prompt + [A] + marker {marker}")

    # ---- 3. block equivalence -------------------------------------------------
    print("\n[3] block equivalence (layer 0)")
    layer0 = model.model.layers[0]
    native = layer0.mlp
    x = (torch.randn(2, 16, cfg.hidden_size, device=args.device) * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        ref = native(x)
        ref = ref[0] if isinstance(ref, tuple) else ref
        cont = BayesianQwen35MoeBlock(cfg, native, MoERouter).to(args.device)
        out = cont(x)
        xf = x.view(-1, cfg.hidden_size).float()
        ref_logits = F.linear(xf, native.gate.weight.float())
        _, _, _, _, logits = cont.router(xf)
        check(torch.allclose(logits, ref_logits, atol=1e-4, rtol=1e-4), "container router logits == F.linear(x, gate.W)")
        top_ref = ref_logits.topk(cfg.num_experts_per_tok, dim=-1).indices.sort(-1).values
        top_bf = F.linear(x.view(-1, cfg.hidden_size), native.gate.weight).float().topk(cfg.num_experts_per_tok, dim=-1).indices.sort(-1).values
        stable = (top_ref == top_bf).all(-1)                      # rows where bf16/fp32 gates agree on top-k
        d = (out.view(-1, cfg.hidden_size).float() - ref.view(-1, cfg.hidden_size).float()).abs()
        scale = ref.float().abs().mean().item() + 1e-6
        rel = (d[stable].max().item() / scale) if stable.any() else float("nan")
        check(stable.float().mean().item() > 0.9, f"{stable.float().mean().item():.3f} of rows have stable top-k")
        check(rel < 5e-2, f"MAP container vs native block: max|diff|/mean|ref| = {rel:.3e} on stable rows")
        fc = FullCovarianceVariationalRouter(router_config_for(cfg), existing_router=cont.router).to(args.device)
        fc.deterministic_readout = True
        cont.router = fc
        out2 = cont(x)
        d2 = (out2.view(-1, cfg.hidden_size).float() - out.view(-1, cfg.hidden_size).float()).abs()
        check(d2.max().item() / scale < 5e-2, f"fresh FCVR (deterministic) vs MAP container: {d2.max().item()/scale:.3e}")
        check(fc.last_cholesky_factor.shape == (32, cfg.num_experts, cfg.num_experts), "cholesky factor [N, 256, 256]")
    del cont, fc

    # ---- 5. PEFT targets ---------------------------------------------------
    print("\n[5] PEFT qkv targets")
    del model
    torch.cuda.empty_cache()
    peft_model = load_peft_model("qwen36", finetune_mode="qkv", device_map=args.device)
    lora_parents = {n.rsplit(".lora_A", 1)[0] for n, _ in peft_model.named_parameters() if ".lora_A" in n}
    layers_hit = {int(n.split(".layers.")[1].split(".")[0]) for n in lora_parents}
    check(len(layers_hit) == 40, f"LoRA on {len(layers_hit)} layers")
    check(all(".self_attn." in n or ".linear_attn." in n for n in lora_parents), "LoRA only under self_attn./linear_attn.")
    n_tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  trainable {n_tr/1e6:.1f} M")

    # ---- 4/6. containers + hooks on the full model ---------------------------
    print("\n[4/6] containers + router hooks")
    peft_model.eval()
    causal = peft_model.base_model.model.model
    inp = tok([prompt, prompt + "A\nExplanation: the sky"], return_tensors="pt", padding=True,
              add_special_tokens=False).to(args.device)
    with torch.no_grad():
        before = peft_model(**inp).logits.float() if not args.quick else None
        get_adapter("qwen36").ensure_containers(peft_model)
        shapes = {}
        hs = [moe_router(l).register_forward_hook(lambda m, i, o, li=li: shapes.__setitem__(li, tuple(o[-1].shape)))
              for li, l in enumerate(causal.layers)]
        after = peft_model(**inp).logits.float()
        for h in hs:
            h.remove()
    N = inp["input_ids"].numel()
    check(len(shapes) == 40 and all(s == (N, cfg.num_experts) for s in shapes.values()),
          f"router hook output[-1] is [N={N}, 256] on all 40 layers")
    if before is not None:
        mask = inp["attention_mask"].bool()
        dl = (after - before).abs()[mask].max().item()
        agree = (after.argmax(-1) == before.argmax(-1))[mask].float().mean().item()
        check(agree > 0.98, f"full-model logits before/after containers: max|diff| {dl:.3f}, argmax agreement {agree:.3f}")

    # ---- 7. FCVR grads with/without gradient checkpointing -------------------
    if not args.quick:
        print("\n[7] FCVR ELBO grads with vs without gradient checkpointing")
        import argparse as _ap
        fargs = _ap.Namespace(model_shortcode="qwen36", dataset_shortcode="smoke", swap_layers=FCVR_LAYERS,
                              load_layers=[], train_layers=FCVR_LAYERS, run_suffix=None)
        get_adapter("qwen36").prepare(peft_model, method="fcvr", args=fargs)
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
