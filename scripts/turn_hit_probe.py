"""Offline validation for arm 4: can a per-turn hit signal be recovered from a trajectory?

Replays every saved episode prefix-by-prefix through tau2's own `_check_actions`, which is a
pure function of the tool calls in the prefix and therefore monotone by construction. What the
replay has to establish is not monotonicity but *spread*: if every expected action first
matches inside the same assistant turn, turn-level credit has nothing to separate.

A gold action with requestor "user" is executed by the customer, in a UserMessage. The
assistant turn that earned it is the one immediately before -- the turn that told the customer
which switch to flip -- so credit is attributed backwards to that turn.
"""
import json, sys, collections, statistics
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")

from tau2.data_model.tasks import Action
from tau2.data_model.message import ToolCall

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_arm3_s25.json/results.json"
d = json.load(open(path))
tasks = {t["id"]: t for t in d["tasks"]}

n_sim = n_with_actions = 0
spread_turns, n_hit_turns, n_actions, frac_hits_last_turn = [], [], [], []
rel_positions = []
solved_spread = []
degenerate = 0
attribution = collections.Counter()

for sim in d["simulations"]:
    task = tasks.get(sim["task_id"])
    if not task or not (task.get("evaluation_criteria") or {}).get("actions"):
        continue
    golden = [Action(**a) for a in task["evaluation_criteria"]["actions"]]
    msgs = sim["messages"]
    # assistant turn index for every message; user/tool messages map to the turn before them
    assistant_turn = -1
    turn_of_msg = []
    for m in msgs:
        if m["role"] == "assistant":
            assistant_turn += 1
        turn_of_msg.append(assistant_turn)
    n_turns = assistant_turn + 1
    if n_turns <= 0:
        continue

    first_hit = {}          # action_id -> assistant turn that earned it
    seen = []
    for i, m in enumerate(msgs):
        if m["role"] not in ("assistant", "user") or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            seen.append(ToolCall(**{k: tc[k] for k in ("id", "name", "arguments") if k in tc}))
        for g in golden:
            if g.action_id in first_hit:
                continue
            if any(g.compare_with_tool_call(t) for t in seen):
                first_hit[g.action_id] = max(0, turn_of_msg[i])
                attribution[m["role"]] += 1

    if not first_hit:
        continue
    n_sim += 1
    n_with_actions += len(golden)
    hits = sorted(first_hit.values())
    n_actions.append(len(golden))
    n_hit_turns.append(len(set(hits)))
    spread_turns.append(max(hits) - min(hits))
    frac_hits_last_turn.append(sum(1 for h in hits if h == n_turns - 1) / len(hits))
    rel_positions += [h / max(1, n_turns - 1) for h in hits]
    if len(set(hits)) == 1:
        degenerate += 1
    if (sim.get("reward_info") or {}).get("reward", 0) >= 1.0:
        solved_spread.append(max(hits) - min(hits))

def pct(xs, q): return statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 2 else float("nan")

print(f"episodes with >=1 matched action : {n_sim}")
print(f"expected actions per task        : mean {statistics.mean(n_actions):.2f}  max {max(n_actions)}")
print(f"credited to (message role)       : {dict(attribution)}")
print()
print(f"distinct turns carrying a hit    : mean {statistics.mean(n_hit_turns):.2f}  "
      f"median {statistics.median(n_hit_turns):.0f}  max {max(n_hit_turns)}")
print(f"episodes where all hits land in ONE turn : {degenerate} / {n_sim} = {degenerate/n_sim:.1%}")
print(f"turn spread (last hit - first hit): mean {statistics.mean(spread_turns):.2f}  "
      f"median {statistics.median(spread_turns):.0f}  p90 {pct(spread_turns,90):.1f}")
if solved_spread:
    print(f"  ... among SOLVED episodes      : mean {statistics.mean(solved_spread):.2f}  "
          f"median {statistics.median(solved_spread):.0f}")
print()
print(f"hits landing on the FINAL turn   : {statistics.mean(frac_hits_last_turn):.1%} of hits")
print(f"relative position of a hit       : median {statistics.median(rel_positions):.2f}  "
      f"p10 {pct(rel_positions,10):.2f}  p90 {pct(rel_positions,90):.2f}")
h = collections.Counter(min(9, int(p * 10)) for p in rel_positions)
print("  decile histogram (0.0->1.0)    :", [h.get(i, 0) for i in range(10)])
