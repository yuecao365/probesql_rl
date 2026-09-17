"""Turn a veRL RL checkpoint into a merged model vLLM can serve.

    python scripts/rl_to_hf.py ckpt/rl/<run>/global_step_N/actor \
        --base models/sft_v2_ep3 --out models/arm2_rl

verl's own model_merger cannot do this one. It expects the checkpoint to carry the whole
model and validates that it wrote a model.safetensors; with `save_lora_only` the shard holds
only the 504 adapter tensors, so the merge produces a directory with a tokenizer and no
weights. What the shard does hold is exactly a PEFT state dict -- `base_model.model...
lora_A.weight` and `lora_B.weight` -- so the adapter can be written out in PEFT's own layout
and applied to the base the run started from.

The base is the SFT checkpoint, already merged. RL trained a second adapter on top of it, so
the result is SFT's behaviour plus whatever GRPO changed, which is what the evaluation has to
measure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor_dir", help="the global_step_N/actor directory")
    ap.add_argument("--base", default="models/sft_v2_ep3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    shards = sorted(f for f in os.listdir(args.actor_dir) if f.startswith("model_world_size"))
    if not shards:
        raise SystemExit(f"no model shard in {args.actor_dir}")
    state: dict = {}
    for s in shards:
        part = torch.load(os.path.join(args.actor_dir, s), map_location="cpu", weights_only=False)
        state.update(part if isinstance(part, dict) else part.state_dict())
    lora = {k: v for k, v in state.items() if "lora_" in k}
    if not lora:
        raise SystemExit(f"{len(state)} tensors and none of them are LoRA")
    print(f"{len(lora)} adapter tensors out of {len(state)}")

    targets = sorted({k.split(".")[-3] for k in lora})
    print(f"target modules: {targets}")

    adapter = os.path.join(args.out + "_adapter")
    os.makedirs(adapter, exist_ok=True)
    from safetensors.torch import save_file

    # Keys go out exactly as verl saved them. They are already PEFT's on-disk format --
    # base_model.model...lora_A.weight -- and an earlier version of this script stripped the
    # .weight suffix, which made PEFT match nothing, warn about missing adapter keys, and
    # merge a no-op. The evaluation would then have reported RL as having changed nothing.
    save_file({k: v.contiguous() for k, v in lora.items()},
              os.path.join(adapter, "adapter_model.safetensors"))
    json.dump({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": args.rank,
        "lora_alpha": args.alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": targets, "inference_mode": True,
        "base_model_name_or_path": os.path.abspath(args.base),
        "fan_in_fan_out": False, "modules_to_save": None,
    }, open(os.path.join(adapter, "adapter_config.json"), "w"), indent=2)
    print(f"wrote {adapter}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("merging onto the base ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    probe_name, probe_before = next(
        (n, p.detach().clone()) for n, p in model.named_parameters() if "layers.0.self_attn.q_proj" in n
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    probe_after = dict(model.named_parameters())[probe_name]
    delta = (probe_after - probe_before).abs().max().item()
    if delta == 0.0:
        raise SystemExit(
            f"merge changed nothing: {probe_name} is bit-identical to the base. The adapter "
            f"keys did not match, and an evaluation of this model would have measured the base."
        )
    print(f"merge verified: {probe_name} moved by up to {delta:.3e}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    for extra in ("chat_template.jinja", "generation_config.json"):
        src = os.path.join(args.base, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, extra))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
