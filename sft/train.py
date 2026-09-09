"""LoRA SFT on rejection-sampled trajectories.

    python sft/train.py --model models/Qwen2.5-7B-Instruct --data data/sft/teacher.jsonl --out ckpt/sft_r32

Plain transformers + peft rather than a training framework, so the examples
are tokenized by sft.data.encode (the same chat template the rollout server
uses) and the loss mask is the one scripts/dump_mask.py verifies. LoRA on every
linear layer with r=32 is enough capacity for behaviour-level SFT; the trade-off
is a merge step (sft/merge.py) before the adapter can serve as an RL start.
"""

import argparse
import json
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, ".")

from sft.data import IGNORE, encode  # noqa: E402

LINEAR_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16, help="effective batch via gradient accumulation")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.data) as f:
        raw = [json.loads(line) for line in f]
    examples = [ex for ex in (encode(tok, r["messages"], args.max_len) for r in raw) if ex]
    learned = sum(sum(l != IGNORE for l in ex["labels"]) for ex in examples)
    total = sum(len(ex["input_ids"]) for ex in examples)
    print(f"{len(examples)}/{len(raw)} examples within {args.max_len} tokens; {learned}/{total} tokens in the loss ({learned / total:.1%})")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=LINEAR_LAYERS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=args.batch,
            lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=5,
            save_strategy="epoch", report_to=[], remove_unused_columns=False, group_by_length=True,
        ),
        train_dataset=examples,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
