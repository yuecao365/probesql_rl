"""Follow-up: is the 15.6% degenerate share just the single-action tasks, and what does the
local term L_k actually look like inside an all-pass group -- the case where arm 3's advantage
is identically zero and arm 4 has to carry the whole signal on its own."""
import json, sys, collections, statistics
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from tau2.data_model.tasks import Action
from tau2.data_model.message import ToolCall

LAM, W_HIT = 0.3, 0.2

def hits_of(sim, task):
    golden = [Action(**a) for a in task["evaluation_criteria"]["actions"]]
    msgs = sim["messages"]
    t, turn_of = -1, []
    for m in msgs:
        if m["role"] == "assistant":
            t += 1
        turn_of.append(t)
    n_turns = t + 1
    first, seen = {}, []
    for i, m in enumerate(msgs):
        if m["role"] not in ("assistant", "user") or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            seen.append(ToolCall(**{k: tc[k] for k in ("id", "name", "arguments") if k in tc}))
        for g in golden:
            if g.action_id not in first and any(g.compare_with_tool_call(x) for x in seen):
                first[g.action_id] = max(0, turn_of[i])
    return n_turns, len(golden), sorted(first.values())

def local_returns(n_turns, m, hits):
    r = [0.0] * n_turns
    for h in hits:
        r[h] += W_HIT / m
    L, run = [0.0] * n_turns, 0.0
    for k in range(n_turns - 1, -1, -1):
        run = r[k] + LAM * run
        L[k] = run
    return L

d = json.load(open(sys.argv[1]))
tasks = {t["id"]: t for t in d["tasks"]}
by_m = collections.defaultdict(lambda: [0, 0])
groups = collections.defaultdict(list)
for sim in d["simulations"]:
    task = tasks.get(sim["task_id"])
    if not task or not (task.get("evaluation_criteria") or {}).get("actions"):
        continue
    n_turns, m, hits = hits_of(sim, task)
    if not hits or n_turns <= 0:
        continue
    by_m[m][0] += 1
    by_m[m][1] += (len(set(hits)) == 1)
    solved = (sim.get("reward_info") or {}).get("reward", 0) >= 1.0
    groups[sim["task_id"]].append((n_turns, m, hits, solved))

print("m = expected actions | episodes | all hits in one turn")
for m in sorted(by_m):
    n, deg = by_m[m]
    print(f"  m={m:<3} {n:>5}   {deg/n:>6.1%}")

allpass = [g for g in groups.values() if len(g) >= 2 and all(s for *_, s in g)]
print(f"\nall-pass groups (arm 3 advantage == 0 there): {len(allpass)} / {len(groups)}")
within, per_group_sd, same_len = [], [], 0
for g in allpass:
    Ls = []
    if len({n for n, *_ in g}) == 1:
        same_len += 1
    for n_turns, m, hits, _ in g:
        L = local_returns(n_turns, m, hits)
        Ls += L
        within.append(max(L) - min(L))
    per_group_sd.append(statistics.pstdev(Ls))
print(f"  groups where every rollout also has the SAME turn count "
      f"(eff degenerate too, arm 3 truly zero): {same_len}")
print(f"  within-trajectory L spread : mean {statistics.mean(within):.4f}")
print(f"  pooled L std inside group  : mean {statistics.mean(per_group_sd):.4f}")
print(f"  ratio to arm 3's typical |adv| 0.105 (all-pass) : "
      f"{statistics.mean(per_group_sd)/0.105:.2f}x")
