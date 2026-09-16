#!/usr/bin/env bash
# Sample teacher trajectories for SFT. Run staged: a small N first, read the rejection
# histogram from scripts/build_sft.py, then continue with a larger N if it passes.
#
#   bash scripts/serve_vllm.sh models/Qwen3-14B qwen14b     # teacher on the GPU, alone
#   N=15 bash scripts/tau2_teacher.sh                       # staged probe
#   N=150 bash scripts/tau2_teacher.sh                      # the real run
#
# The teacher is Qwen3-14B: same family as the student, so its trajectories are close to
# the student's own distribution, and it is the strongest Qwen3 scale on telecom (29.9
# zero-shot, above 32B's 24.8). Distribution distance matters more than raw strength here
# -- MUA-RL's cold start halved the 8B's telecom score, and the likeliest reason is that
# their SFT data came from retail/airline, domains with no dual control at all.
#
# Sampling is stratified: the training pool is 89% mms_issue and has 12 transfer-only
# tasks against the benchmark's 17.5%, so a uniform sample would teach the wrong mix.
#
# --task-split-name full is required: the pool lives outside the `base` split that
# the flag defaults to, and the runner refuses ids it cannot find there. Use the split,
# not --task-set-name telecom_full: that set is registered without a split function
# while the runner always passes task_split_name, so it raises a TypeError.
set -euo pipefail

REPO=/root/probesql
TAU2=/root/autodl-tmp/tau2-bench
N=${N:-15}
G=${G:-4}
SEED=${SEED:-0}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
TEACHER=${TEACHER:-qwen14b}
# A local vLLM teacher needs api_base; an API teacher (deepseek/...) must not get one.
case "$TEACHER" in
  */*) TEACHER_MODEL="$TEACHER"
       TEACHER_ARGS='{"temperature":1.0}' ;;
  *)   TEACHER_MODEL="openai/$TEACHER"
       TEACHER_ARGS="{\"temperature\":1.0,\"api_base\":\"$BASE_URL\",\"api_key\":\"dummy\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" ;;
esac
USER_LLM=${USER_LLM:-deepseek/deepseek-chat}
TAG=${TAG:-$(echo "$TEACHER" | tr "/" "_")}
OUT=$REPO/outputs/teacher_${TAG}_n${N}.json

set -a; source "$REPO/.env"; set +a

RESUME=""
if [ -e "$OUT" ]; then
  if [ "${FORCE:-0}" = 1 ]; then mv "$OUT" "$OUT.$(date +%s).bak"
  else RESUME=--auto-resume; fi     # continue the run instead of discarding it
fi

# Stratified draw from full minus base: keep the benchmark's issue mix and its 17.5%
# share of transfer-only tasks, both of which the raw pool badly misrepresents.
mapfile -t TASK_IDS < <("$TAU2/.venv/bin/python" - "$SEED" "$N" <<'PY'
import json, random, sys, collections
seed, n = int(sys.argv[1]), int(sys.argv[2])
D = "/root/autodl-tmp/tau2-bench/data/tau2/domains/telecom/"
splits = json.load(open(D + "split_tasks.json"))
tasks = {t["id"]: t for t in json.load(open(D + "tasks.json"))}
base = set(splits["base"])
pool = [i for i in tasks if i not in base]                      # leak-free training pool

def actions(i):
    return [a["name"] for a in (tasks[i]["evaluation_criteria"].get("actions") or [])]
def issue(i):
    return i.split("]")[0].strip("[")

transfer = [i for i in pool if actions(i) == ["transfer_to_human_agents"]]
rest = [i for i in pool if i not in set(transfer)]
rng = random.Random(seed)

# Benchmark proportions, measured on base: 17.5% transfer, then 43/32/25 across issues.
n_transfer = max(1, round(n * 0.175))
picked = rng.sample(transfer, min(n_transfer, len(transfer)))
by_issue = collections.defaultdict(list)
for i in rest:
    by_issue[issue(i)].append(i)
for name, share in (("mms_issue", 0.43), ("mobile_data_issue", 0.32), ("service_issue", 0.25)):
    want = round((n - n_transfer) * share)
    have = by_issue.get(name, [])
    picked += rng.sample(have, min(want, len(have)))
while len(picked) < n and rest:                                  # top up if a stratum ran dry
    extra = rng.choice(rest)
    if extra not in picked:
        picked.append(extra)
for i in picked[:n]:
    print(i)
PY
)
echo "sampled ${#TASK_IDS[@]} tasks (seed $SEED), teacher=$TEACHER, G=$G"

cd "$TAU2"
env -u HTTPS_PROXY -u HTTP_PROXY DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  .venv/bin/tau2 run \
    --domain telecom \
    --task-split-name full \
    --agent-llm "$TEACHER_MODEL" \
    --agent-llm-args "$TEACHER_ARGS" \
    --user-llm "$USER_LLM" \
    --user-llm-args '{"temperature":0.0}' \
    --task-ids "${TASK_IDS[@]}" \
    --num-trials "$G" ${RESUME:-} \
    --max-concurrency "${CONCURRENCY:-8}" \
    --save-to "$OUT"

echo
"$TAU2/.venv/bin/python" "$REPO/scripts/build_sft.py" "$OUT" \
  --allow-tool-errors --per-task 2 --out "$REPO/data/sft/teacher_n${N}.jsonl"
