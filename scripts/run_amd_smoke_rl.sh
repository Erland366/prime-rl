#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate primerl

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${WANDB_API_KEY:?WANDB_API_KEY must be set in .env or the shell environment}"

export WANDB_ENTITY="${WANDB_ENTITY:-erlandpg}"
export WANDB_PROJECT="${WANDB_PROJECT:-prime-rl}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export PYTHONPATH="/opt/rocm/share/amd_smi:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

wandb login "$WANDB_API_KEY"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-amd-mi210-rl-smoke-${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/amd_mi210_4gpu_rl_smoke_${TIMESTAMP}}"

python -m prime_rl.entrypoints.rl \
  @ configs/smoke/amd_mi210_4gpu_rl.toml \
  --clean-output-dir \
  --output-dir "$OUTPUT_DIR" \
  --wandb.project "$WANDB_PROJECT" \
  --wandb.name "$RUN_NAME" \
  --model.name "${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}" \
  --max-steps "${MAX_STEPS:-12}" \
  --seq-len "${SEQ_LEN:-256}" \
  --orchestrator.batch_size "${ORCH_BATCH_SIZE:-8}" \
  --orchestrator.rollouts_per_example "${ROLLOUTS_PER_EXAMPLE:-4}" \
  --orchestrator.sampling.max-completion-tokens "${MAX_TOKENS:-32}" \
  --inference.gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --inference.model.max_model_len "${MAX_MODEL_LEN:-256}"
