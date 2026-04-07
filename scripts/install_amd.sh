#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

REPO_ID="prime-rl"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-primerl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
SKIP_CLONE="${SKIP_CLONE:-1}"
VALIDATE="${VALIDATE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/amd_debug_sft}"

has_ssh_access() {
    set +e
    timeout 5s git ls-remote --heads "git@github.com:PrimeIntellect-ai/${REPO_ID}.git" >/dev/null 2>&1
    rc=$?
    set -e
    return $rc
}

ensure_known_hosts() {
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
    if command -v ssh-keyscan >/dev/null 2>&1; then
        ssh-keyscan -H github.com 2>/dev/null | sort -u | tee -a "${HOME}/.ssh/known_hosts" >/dev/null
        chmod 600 "${HOME}/.ssh/known_hosts"
    fi
}

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
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return
    fi
    echo "conda was not found. Install Miniforge or Conda first." >&2
    exit 1
}

ensure_repo() {
    if [ "${SKIP_CLONE}" -eq 1 ]; then
        log_info "Skipping clone; assuming the current directory is the repo root."
        return
    fi

    ensure_known_hosts
    if has_ssh_access; then
        log_info "Cloning PRIME-RL via SSH."
        git clone "git@github.com:PrimeIntellect-ai/${REPO_ID}.git"
    else
        log_warn "SSH auth unavailable. Cloning PRIME-RL via HTTPS."
        git clone "https://github.com/PrimeIntellect-ai/${REPO_ID}.git"
    fi
    cd "${REPO_ID}"
}

ensure_conda_env() {
    source_conda
    if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
        log_info "Creating conda env ${CONDA_ENV_NAME} with Python ${PYTHON_VERSION}."
        conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
    fi
    conda activate "${CONDA_ENV_NAME}"
}

install_base_packages() {
    log_info "Upgrading pip/setuptools/wheel in ${CONDA_ENV_NAME}."
    python -m pip install --upgrade pip setuptools wheel

    log_info "Installing ROCm PyTorch wheels from ${ROCM_INDEX_URL}."
    python -m pip install \
        --index-url "${ROCM_INDEX_URL}" \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1
}

install_prime_rl() {
    log_info "Installing PRIME-RL in editable mode."
    python -m pip install -e . --no-deps

    log_info "Installing the AMD-validated dependency set."
    python -m pip install \
        psutil \
        beartype \
        "datasets==4.6.1" \
        jaxtyping \
        loguru \
        pyarrow \
        pydantic \
        tomli \
        tomli-w \
        torchdata \
        wandb \
        pyzmq \
        aiolimiter \
        tenacity \
        openai \
        rich \
        setproctitle \
        uvloop \
        "transformers==5.5.0" \
        "liger-kernel>=0.5.10" \
        "pydantic-config @ git+https://github.com/samsja/pydantic_config.git@main" \
        "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e" \
        "dion @ git+https://github.com/samsja/dion.git@d891eeb" \
        "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@0760204"
}

validate_install() {
    if [ "${VALIDATE}" -ne 1 ]; then
        log_warn "Skipping validation run because VALIDATE=${VALIDATE}."
        return
    fi

    log_info "Checking ROCm visibility from PyTorch."
    python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("device0", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
y = x @ x
print("matmul_ok", tuple(y.shape), float(y.mean().cpu()))
PY

    log_info "Running a 1-step AMD SFT validation."
    python -m prime_rl.entrypoints.sft \
        @ configs/debug/sft/train.toml \
        --output-dir "${OUTPUT_DIR}" \
        --clean-output-dir \
        --max-steps 1 \
        --data.seq-len 128 \
        --data.batch-size 1 \
        --data.micro-batch-size 1 \
        --model.name Qwen/Qwen3-0.6B \
        --model.attn sdpa \
        --model.debug.num-layers 2
}

print_summary() {
    cat <<EOF

AMD installation completed.

Conda environment:
  conda activate ${CONDA_ENV_NAME}

Validated trainer command:
  python -m prime_rl.entrypoints.sft @ configs/debug/sft/train.toml \\
    --output-dir ${OUTPUT_DIR} \\
    --clean-output-dir \\
    --max-steps 1 \\
    --data.seq-len 128 \\
    --data.batch-size 1 \\
    --data.micro-batch-size 1 \\
    --model.name Qwen/Qwen3-0.6B \\
    --model.attn sdpa \\
    --model.debug.num-layers 2

Notes:
  - This path is validated for the SFT trainer on AMD ROCm.
  - The vLLM / inference stack is not installed by this script.
  - Keep context parallelism disabled on AMD for this path: model.cp = 1.
  - Use sdpa on AMD unless you have independently validated a FlashAttention-compatible path.

EOF
}

main() {
    ensure_repo
    ensure_conda_env
    install_base_packages
    install_prime_rl
    validate_install
    print_summary
}

main "$@"
