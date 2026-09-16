"""Turn a tau2 run into SFT examples: reject, rank, and write self-contained records.

    /root/autodl-tmp/tau2-bench/.venv/bin/python scripts/build_sft.py \
        outputs/teacher_telecom.json --per-task 2 --out data/sft/teacher.jsonl

Runs under tau2's venv, because only tau2 knows the two things a record needs beyond the
transcript: the domain's tool schemas and the exact system prompt the agent was given. The
record it writes carries both, so `sft/train.py` in the training env never imports tau2 and
the file stays valid even if the benchmark moves underneath it.

Prints the rejection histogram, which is the number to look at before trusting any SFT run:
it says whether the teacher is producing usable behaviour or just occasionally getting lucky.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.filter import agent_calls, agent_messages, length, reject_reason, to_messages  # noqa: E402


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f)


def domain_context(domain: str) -> tuple[list[dict], str]:
    """The tool schemas and system prompt the agent actually ran with."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)(solo_mode=False)
    # Store bare function schemas; sft.data.as_tools adds the envelope the template wants.
    tools = [t.openai_schema["function"] for t in env.get_tools()]
    policy = env.get_policy()
    return tools, SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--per-task", type=int, default=2, help="keep the N shortest clean trajectories per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-tool-errors", action="store_true",
                    help="keep trajectories that hit a tool error and recovered")
    args = ap.parse_args()

    sims = load(args.results)["simulations"]
    tools, system_prompt = domain_context(args.domain)

    # Both strictness levels are reported: the gap is the cost of excluding recovery.
    strict_ok = sum(1 for s in sims if reject_reason(s) is None)
    lenient_ok = sum(1 for s in sims if reject_reason(s, allow_tool_errors=True) is None)

    reasons = collections.Counter()
    accepted = collections.defaultdict(list)
    for s in sims:
        reason = reject_reason(s, allow_tool_errors=args.allow_tool_errors)
        reasons[reason or "accepted"] += 1
        if reason is None:
            accepted[s["task_id"]].append(s)

    kept = []
    for task_id, group in accepted.items():
        for s in sorted(group, key=length)[: args.per_task]:
            kept.append(
                {
                    "task_id": task_id,
                    "trial": s.get("trial"),
                    "agent_turns": len(agent_messages(s)),
                    "agent_calls": len(agent_calls(s)),
                    "messages": to_messages(s, system_prompt),
                    "tools": tools,
                }
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tasks = len({s["task_id"] for s in sims})
    print(f"{len(sims)} rollouts over {n_tasks} tasks")
    for reason, n in reasons.most_common():
        print(f"  {reason:28s} {n:5d}  {n / len(sims):6.1%}")
    print(f"accepted: strict {strict_ok}/{len(sims)} ({strict_ok / len(sims):.1%}) · "
          f"tool errors allowed {lenient_ok}/{len(sims)} ({lenient_ok / len(sims):.1%})")
    turns = [r["agent_turns"] for r in kept]
    print(f"\n{len(kept)} examples from {len(accepted)} tasks ({len(accepted) / n_tasks:.1%} covered) -> {args.out}")
    if turns:
        print(f"agent turns: min {min(turns)} / median {sorted(turns)[len(turns) // 2]} / max {max(turns)}")


if __name__ == "__main__":
    main()
