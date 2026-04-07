#!/usr/bin/env bash
set -euo pipefail

source_conda() {
    if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniforge3/etc/profile.d/conda.sh"
        return
    fi
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        return
    fi
    if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        return
    fi
    eval "$(conda shell.bash hook)"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/smoke/amd_mi210_4gpu_sft.toml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-primerl}"
NUM_GPUS="${NUM_GPUS:-4}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_NAME="${MODEL_NAME:-PrimeIntellect/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/amd_mi210_4gpu_sft_smoke}"
MAX_STEPS="${MAX_STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
WANDB_PROJECT="${WANDB_PROJECT:-prime-rl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-amd-mi210-4gpu-sft-smoke-$(date -u +%Y%m%d-%H%M%S)}"
WANDB_ENTITY="${WANDB_ENTITY:-erlandpg}"
MODEL_DEBUG_NUM_LAYERS="${MODEL_DEBUG_NUM_LAYERS:-}"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export HIP_VISIBLE_DEVICES
export WANDB_PROJECT
export WANDB_ENTITY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

source_conda
conda activate "${CONDA_ENV_NAME}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is not set. Add it to .env or export it before running this script." >&2
    exit 1
fi

python - <<'PY'
import os
import wandb

device_count = len([value for value in os.environ["HIP_VISIBLE_DEVICES"].split(",") if value.strip()])
print(f"HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}")
print(f"WANDB_ENTITY={os.environ['WANDB_ENTITY']}")
print(f"WANDB_PROJECT={os.environ['WANDB_PROJECT']}")
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
if torch.cuda.device_count() != device_count:
    raise SystemExit(
        f"Expected {device_count} visible GPU(s) from HIP_VISIBLE_DEVICES but torch sees {torch.cuda.device_count()}"
    )
for idx in range(torch.cuda.device_count()):
    print(f"device{idx}", torch.cuda.get_device_name(idx))
PY

CMD=(
    python -m prime_rl.entrypoints.sft
    @ "${CONFIG_PATH}"
    --clean-output-dir
    --output-dir "${OUTPUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --deployment.num-gpus "${NUM_GPUS}"
    --model.name "${MODEL_NAME}"
    --model.attn sdpa
    --model.cp 1
    --model.seq-len "${SEQ_LEN}"
    --data.seq-len "${SEQ_LEN}"
    --data.batch-size "${BATCH_SIZE}"
    --data.micro-batch-size "${MICRO_BATCH_SIZE}"
    --wandb.project "${WANDB_PROJECT}"
    --wandb.name "${WANDB_RUN_NAME}"
)

if [ -n "${MODEL_DEBUG_NUM_LAYERS}" ]; then
    CMD+=(--model.debug.num-layers "${MODEL_DEBUG_NUM_LAYERS}")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
