#!/usr/bin/env bash
# Wait for the arm 3dr evaluation chain, then recompute everything a reader sees.
#
# Only numbers are generated here. The prose in the README says what the numbers mean, and
# what they mean depends on whether the shaped reward survives without the std normalisation
# it was accidentally confounded with -- which is the question this run exists to answer. A
# script that wrote that sentence in advance would be writing a conclusion before the result.
set -uo pipefail
cd /root/probesql
VENV=/root/autodl-tmp/tau2-bench/.venv/bin/python
LOG=logs/finalize.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the evaluation chain"
while pgrep -f "[e]val_arm3dr.sh" > /dev/null; do sleep 60; done
say "chain ended"

for a in arm3dr_s15 arm3dr_s25; do
  [ -f "outputs/eval_$a.json/results.json" ] && say "$a: evaluated" || say "$a: MISSING"
done

say "recomputing the results table and every paired test"
"$VENV" scripts/rl_stats.py > docs/assets/stats.txt 2>/dev/null && say "wrote docs/assets/stats.txt"

say "regenerating figures"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate rl
python scripts/readme_figs.py >> "$LOG" 2>&1 && say "wrote docs/assets/{ladder,efficiency,coverage}.png"

{
  echo
  echo "## arm 3dr — the shaped reward without the normalisation (auto-generated $(date '+%F %H:%M'))"
  echo
  echo 'Same reward as arm 3, `norm_adv_by_std_in_grpo=False`, everything else identical. This'
  echo 'separates "the shaped reward did this" from "dividing by the group std did this".'
  echo
  echo '```'
  cat docs/assets/stats.txt
  echo '```'
} >> docs/results.md
say "appended the numbers to docs/results.md"
say "done -- README prose still to be written against these numbers"
