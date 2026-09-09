"""Print one SFT example split into learned and masked segments, for eyeballing the loss mask.

    python scripts/dump_mask.py data/sft/teacher.jsonl --model models/Qwen2.5-7B-Instruct --index 0

The learned segments must contain only what the model itself generated. If a
byte of a tool result shows up there, training would teach the model to recite
database rows.
"""

import argparse
import json
import sys

from transformers import AutoTokenizer

sys.path.insert(0, ".")

from sft.data import IGNORE, encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=12288)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.data) as f:
        record = json.loads(f.readlines()[args.index])
    ex = encode(tok, record["messages"], args.max_len)
    ids, labels = ex["input_ids"], ex["labels"]

    segments, current, learned = [], [], None
    for t, l in zip(ids, labels):
        flag = l != IGNORE
        if flag != learned and current:
            segments.append((learned, current))
            current = []
        learned, current = flag, current + [t]
    segments.append((learned, current))

    for i, (flag, toks) in enumerate(segments):
        print(f"\n===== segment {i}: {'LEARNED' if flag else 'masked'} ({len(toks)} tokens) =====")
        print(tok.decode(toks))
    n_learned = sum(l != IGNORE for l in labels)
    print(f"\n{n_learned}/{len(ids)} tokens learned ({n_learned / len(ids):.1%}); {sum(f for f, _ in segments)} assistant turns")


if __name__ == "__main__":
    main()
