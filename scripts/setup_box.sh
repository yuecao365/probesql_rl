#!/usr/bin/env bash
# Rebuild this project's environment on a fresh AutoDL box. Idempotent: safe to re-run.
#
#   bash scripts/setup_box.sh            # everything
#   bash scripts/setup_box.sh rl         # just the training env
#   bash scripts/setup_box.sh tau2       # just the benchmark env
#   bash scripts/setup_box.sh model      # just the Qwen3-8B download
#   bash scripts/setup_box.sh verl       # the RL trainer, in its own env
#
# Every non-obvious line here is a bug that cost a session once. They are commented
# with what goes wrong without them; do not "clean up" a line you cannot explain.
set -euo pipefail

DATA=/root/autodl-tmp
REPO=/root/probesql
PYPI=https://mirrors.aliyun.com/pypi/simple/
GH=https://gh-proxy.com          # GitHub is unreachable directly; this mirror works for tarballs
WHAT=${1:-all}

step() { echo; echo "=====  $*  ====="; }

# ---------------------------------------------------------------- training env
setup_rl() {
  step "conda env 'rl' (python 3.10, vLLM + torch)"
  source /root/miniconda3/etc/profile.d/conda.sh
  pip config set global.index-url "$PYPI" >/dev/null

  # NOTE: conda's solver needs >2 GB and is OOM-killed in AutoDL's no-GPU mode.
  # Create this env while the GPU is attached, or the process dies with no message.
  conda env list | grep -q '^rl ' || conda create -y -n rl python=3.10
  conda activate rl

  pip install -q \
    "torch==2.13.0" "vllm==0.28.0" "transformers==5.16.1" "peft==0.20.0" \
    "accelerate==1.14.0" "openai==3.8.0" "numpy==2.2.6" \
    pytest pypdf huggingface_hub

  # hf-xet resolves large files to cas-server and gets a 401 behind the mirror.
  pip uninstall -y -q hf-xet 2>/dev/null || true

  # The GPU image ships an old system libstdc++ (CXXABI <= 1.3.13). pip's torch loads it
  # first, and then `import sqlite3` fails asking for 1.3.15. Preloading conda's own copy
  # fixes it for every process in the env. OMP_NUM_THREADS is otherwise an invalid 0.
  conda env config vars set \
    LD_PRELOAD=/root/miniconda3/envs/rl/lib/libstdc++.so.6 \
    OMP_NUM_THREADS=16 -n rl

  echo "ok: reactivate the env for the vars to take effect"
}

# ---------------------------------------------------------------- tau2-bench
setup_tau2() {
  step "tau2-bench (its own python 3.12 venv, never mixed with the training env)"
  mkdir -p "$DATA"
  if [ ! -d "$DATA/tau2-bench" ]; then
    # git clone over the SSH tunnel dies on sideband packets; a single HTTP GET does not.
    # --speed-limit/--speed-time abort a stalled transfer so --retry can act on it;
    # without them a dead connection hangs the script indefinitely rather than retrying.
    curl -sSL --retry 5 --retry-all-errors --speed-limit 1024 --speed-time 60 \
      -o "$DATA/tau2.tar.gz" \
      "$GH/https://github.com/sierra-research/tau2-bench/archive/refs/heads/main.tar.gz"
    tar tzf "$DATA/tau2.tar.gz" >/dev/null || { echo "tarball incomplete"; exit 1; }
    tar xzf "$DATA/tau2.tar.gz" -C "$DATA"
    mv "$DATA/tau2-bench-main" "$DATA/tau2-bench"
    rm -f "$DATA/tau2.tar.gz"
  fi

  # uv rather than conda: tau2 needs python >=3.12 while the training env is 3.10, and
  # uv's resolver survives the 2 GB cgroup that kills conda's.
  command -v uv >/dev/null || pip install -q uv
  cd "$DATA/tau2-bench"
  [ -d .venv ] || uv venv --python 3.12 .venv
  # --python is required: without it uv installs into whatever conda env is active.
  UV_DEFAULT_INDEX=$PYPI uv pip install --python .venv/bin/python -e ".[gym]"
  # Upstream packaging bug: tau2/__init__ imports the voice module unconditionally,
  # but websockets only ships in the [voice] extra, so the base install cannot import.
  UV_DEFAULT_INDEX=$PYPI uv pip install --python .venv/bin/python -q websockets

  .venv/bin/python -c "import tau2, gymnasium; print('tau2 import ok')"
}

# ---------------------------------------------------------------- base model
setup_model() {
  step "Qwen3-8B (~16 GB)"
  mkdir -p "$DATA/models"
  [ -f "$DATA/models/Qwen3-8B/config.json" ] && { echo "already present"; return; }
  # Unset the proxies: with them set, large files redirect to a host outside no_proxy
  # and crawl through the tunnel until they time out.
  env -u HTTPS_PROXY -u HTTP_PROXY HF_ENDPOINT=https://hf-mirror.com \
    /root/miniconda3/envs/rl/bin/hf download Qwen/Qwen3-8B --local-dir "$DATA/models/Qwen3-8B"
}

# ---------------------------------------------------------------- RL trainer
setup_verl() {
  step "veRL in its own conda env (version conflict with the SFT env)"
  # verl 0.9.0 pins transformers <5.11 while the rl env runs 5.16.1, and the loss mask in
  # sft/data.py was written against 5.16's chat-template behaviour. Installing verl into
  # `rl` would downgrade transformers under the one piece of code that must not move.
  # The two envs hand off through model files on disk, never through imports.
  source /root/miniconda3/etc/profile.d/conda.sh
  conda env list | grep -q '^verl ' || conda create -y -n verl python=3.10
  # Use the env's pip by absolute path rather than `conda activate`: activation does not
  # take in a non-interactive subshell, and the install then silently lands nowhere (or
  # worse, in whatever env was already active).
  /root/miniconda3/envs/verl/bin/pip install -q "verl[vllm]==0.9.0"
  conda env config vars set \
    LD_PRELOAD=/root/miniconda3/envs/verl/lib/libstdc++.so.6 \
    OMP_NUM_THREADS=16 -n verl
  /root/miniconda3/envs/verl/bin/python -c "import verl; print('verl', verl.__version__)"
}

# ---------------------------------------------------------------- repo wiring
setup_repo() {
  step "repo symlinks and secrets"
  mkdir -p "$DATA/models" "$DATA/ckpt" "$REPO/outputs" "$REPO/logs"
  ln -sfn "$DATA/models" "$REPO/models"
  ln -sfn "$DATA/ckpt" "$REPO/ckpt"
  [ -f "$REPO/.env" ] || echo "DEEPSEEK_API_KEY=<paste it here>" > "$REPO/.env"
  echo "ok: .env holds DEEPSEEK_API_KEY and is git-ignored"
}

case "$WHAT" in
  rl) setup_rl ;;
  verl) setup_verl ;;
  tau2) setup_tau2 ;;
  model) setup_model ;;
  repo) setup_repo ;;
  all) setup_rl; setup_tau2; setup_model; setup_repo ;;
  # verl is not in `all`: it is a separate env and only needed once RL starts.
  *) echo "usage: $0 [all|rl|tau2|model|repo]"; exit 1 ;;
esac

step "verify"
cat <<'EOF'
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rl
  cd /root/probesql && pytest -q                 # 6 tests, needs models/Qwen3-8B on disk
  nvidia-smi                                     # confirms the GPU is attached
EOF
