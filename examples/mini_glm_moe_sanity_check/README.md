# Mini GLM MoE Sanity Check

This example runs a short single-node RL job with `Erland/mini-glm-moe` against the local `sanity-check` environment.

The environment package lives under `src/sanity_check`, so `PYTHONPATH=$PWD/src` is required when launching locally.

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
uv run rl @ examples/mini_glm_moe_sanity_check/rl.toml \
  --output-dir outputs/mini-glm-moe-sanity-check
```

or if you use conda

```
source "$HOME/miniforge3/etc/profile.d/conda.sh" && \
conda activate primerl && \
set -a && source .env && set +a && \
export PYTHONPATH="/opt/rocm/share/amd_smi:$PWD/src" && \
export NO_PROXY="127.0.0.1,localhost" && \
export no_proxy="127.0.0.1,localhost" && \
export VLLM_TARGET_DEVICE="rocm" && \
OUTPUT_DIR="outputs/mini-glm-moe-sanity-check-20260420-10" && \
echo "OUTPUT_DIR=$OUTPUT_DIR" && \
python -m prime_rl.entrypoints.rl @ examples/mini_glm_moe_sanity_check/rl.toml --output-dir "$OUTPUT_DIR"
```

This configuration is intentionally small:

- `3` training GPUs
- `1` inference GPU
- trainer `max_steps = 2000`
- orchestrator `max_steps = 12`
- `openai/gsm8k` subset `main` as the prompt source

The reward encourages the model to stop talking immediately, so this is useful as a fast integration test for the full RL stack.

W&B runs in shared mode by default for this example, so trainer and orchestrator metrics land in the same run. That means the reward curves are visible alongside trainer loss metrics in one place.
