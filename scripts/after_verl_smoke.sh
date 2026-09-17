#!/usr/bin/env bash
# Queued chain: when the verl install ends, check the API that rl/tau2_tool.py was written
# against. It was written from the wheel's source before the install finished, so every
# assumption in it is unverified -- this prints the real signatures instead of guessing.
# CPU only; it deliberately does not touch the GPU, which the entropy chain is using.
set -uo pipefail
cd /root/probesql
PY=/root/miniconda3/envs/verl/bin/python
LOG=logs/verl_smoke.log
: > "$LOG"

echo "[$(date '+%F %T')] waiting for pip to finish" | tee -a "$LOG"
while pgrep -f "envs/verl/bin/pip install" >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] pip done" | tee -a "$LOG"
tail -3 logs/verl_pip2.log | tee -a "$LOG"

$PY - 2>&1 <<'PYEOF' | tee -a "$LOG"
import inspect
try:
    import verl; print("verl", verl.__version__)
except Exception as e:
    print("IMPORT FAILED:", e); raise SystemExit(1)

from verl.tools.base_tool import BaseTool
print("\n-- BaseTool methods rl/tau2_tool.py overrides --")
for m in ("create", "execute", "calc_reward", "release"):
    f = getattr(BaseTool, m, None)
    print(f"  {m}: {inspect.signature(f) if f else 'MISSING'}")

print("\n-- ToolResponse fields --")
from verl.tools.schemas import ToolResponse
print(" ", getattr(ToolResponse, "model_fields", ToolResponse.__dict__).keys())

print("\n-- registered advantage estimators --")
from verl.trainer.ppo import core_algos
reg = getattr(core_algos, "ADV_ESTIMATOR_REGISTRY", None)
print(" ", sorted(reg) if reg else "registry name differs; inspect core_algos")
print("\n  register_adv_est:", inspect.signature(core_algos.register_adv_est))
PYEOF
echo "[$(date '+%F %T')] smoke finished" | tee -a "$LOG"
