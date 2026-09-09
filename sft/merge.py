"""Merge a LoRA adapter into its base model so vLLM and the RL trainer can load it as a plain checkpoint.

    python sft/merge.py --base models/Qwen2.5-7B-Instruct --adapter ckpt/sft_r32 --out models/qwen7b-sft
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)


if __name__ == "__main__":
    main()
