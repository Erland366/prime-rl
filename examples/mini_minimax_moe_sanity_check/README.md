# Mini MiniMax MoE Sanity Check

This example runs a short single-node RL job with `Erland/mini-minimax-m2`
against the local `sanity-check` environment.

The environment package lives under `src/sanity_check`, so `PYTHONPATH=$PWD/src`
is required when launching locally.

On the local MI210 / ROCm path, the inference server also needs:

- `VLLM_TARGET_DEVICE=rocm`
- `NO_PROXY=127.0.0.1,localhost`
- `no_proxy=127.0.0.1,localhost`

## Training

Run the example from the repository root:

```bash
PYTHONPATH=/opt/rocm/share/amd_smi:$PWD/src \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
VLLM_TARGET_DEVICE=rocm \
uv run rl @ examples/mini_minimax_moe_sanity_check/rl.toml \
  --output-dir outputs/mini-minimax-moe-sanity-check
```

or if you use conda:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh" && \
conda activate primerl && \
set -a && source .env && set +a && \
export PYTHONPATH="/opt/rocm/share/amd_smi:$PWD/src" && \
export NO_PROXY="127.0.0.1,localhost" && \
export no_proxy="127.0.0.1,localhost" && \
export VLLM_TARGET_DEVICE="rocm" && \
OUTPUT_DIR="outputs/mini-minimax-moe-sanity-check-$(date +%Y%m%d-%H%M%S)" && \
python -m prime_rl.entrypoints.rl @ examples/mini_minimax_moe_sanity_check/rl.toml --output-dir "$OUTPUT_DIR"
```

This configuration mirrors the mini-GLM MoE sanity check:

- `3` training GPUs
- `1` inference GPU
- trainer `max_steps = 2000`
- orchestrator `max_steps = 12`
- `openai/gsm8k` subset `main` as the prompt source

The reward encourages the model to stop talking immediately, so this is useful
as a fast integration test for the full RL stack.
