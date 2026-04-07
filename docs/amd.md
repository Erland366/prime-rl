# AMD Install

This page documents the AMD ROCm installation path that was validated locally for PRIME-RL on:

- GPU: `AMD Instinct MI210`
- ROCm-visible arch: `gfx90a`
- Python: `3.12`
- PyTorch: `2.9.1+rocm6.3`

This is not a blanket statement that the full PRIME-RL stack is AMD-supported. The validated paths are:

- editable PRIME-RL install in a `conda` environment
- ROCm PyTorch wheels
- 4-GPU SFT trainer smoke run
- single-node RL smoke run with 4 GPUs (`1 infer + 3 train`)
- `sdpa` attention backend
- `model.cp = 1`

The RL path documented here relies on a local repo environment module (`smoke-reverse-text`) instead of the external `reverse-text` package.

## 4-GPU Smoke Run

For a trainer-only smoke test on 4x MI210 GPUs with W&B logging under `erlandpg`, use:

```bash
bash scripts/run_amd_smoke_sft.sh
```

The script:

- activates `conda activate primerl`
- sources `.env`
- logs into W&B with `WANDB_API_KEY`
- exports `WANDB_ENTITY=erlandpg`
- pins `HIP_VISIBLE_DEVICES=0,1,2,3`
- launches a 4-GPU SFT smoke run with `sdpa` and `cp = 1`

Default smoke config:

- model: `PrimeIntellect/Qwen3-1.7B`
- dataset: `PrimeIntellect/Reverse-Text-SFT`
- GPUs: `4`
- steps: `8`
- sequence length: `1024`

Useful overrides:

```bash
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct MAX_STEPS=4 bash scripts/run_amd_smoke_sft.sh
```

```bash
MODEL_DEBUG_NUM_LAYERS=2 MAX_STEPS=2 bash scripts/run_amd_smoke_sft.sh
```

## 4-GPU RL Smoke Run

For a full local RL smoke run on 4x MI210 GPUs with W&B logging under `erlandpg`, use:

```bash
bash scripts/run_amd_smoke_rl.sh
```

The script:

- activates `conda activate primerl`
- sources `.env`
- logs into W&B with `WANDB_API_KEY`
- exports `WANDB_ENTITY=erlandpg`
- exports both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`
- exports `PYTHONPATH=/opt/rocm/share/amd_smi:$REPO_ROOT/src` so vLLM can import ROCm's `amdsmi` package and the local `smoke-reverse-text` environment module
- exports `NO_PROXY` and `no_proxy` for `127.0.0.1,localhost` so orchestrator health checks do not get sent through host HTTP proxies
- launches single-node RL with `3` training GPUs and `1` inference GPU

Default RL smoke config:

- model: `Qwen/Qwen2.5-3B-Instruct`
- GPUs: `1 infer + 3 train`
- steps: `12`
- sequence length: `256`
- orchestrator batch size: `8`
- rollouts per example: `4`
- max generation tokens: `32`

Useful overrides:

```bash
MAX_STEPS=4 RUN_NAME=amd-rl-validate bash scripts/run_amd_smoke_rl.sh
```

```bash
MODEL_NAME=Qwen/Qwen3-1.7B MAX_STEPS=8 bash scripts/run_amd_smoke_rl.sh
```

This validates the local RL launcher, orchestrator, trainer, and vLLM inference path on MI210. It does not validate external Hub environment installation.

## Quick Start

From the repo root:

```bash
conda activate primerl
bash scripts/install_amd.sh
```

If the `primerl` environment does not exist yet, the script creates it automatically.

The script uses these defaults:

- `CONDA_ENV_NAME=primerl`
- `PYTHON_VERSION=3.12`
- `ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3`
- `VALIDATE=1`

You can override them:

```bash
CONDA_ENV_NAME=primerl VALIDATE=0 bash scripts/install_amd.sh
```

## What The Script Installs

1. ROCm PyTorch wheels:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.3 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

2. PRIME-RL editable install:

```bash
python -m pip install -e . --no-deps
```

3. The extra runtime dependencies needed for the AMD-validated SFT path:

- `transformers==5.5.0`
- `datasets==4.6.1`
- `liger-kernel`
- `torchtitan`
- `dion`
- `verifiers`
- `psutil`
- the rest of the core Python runtime packages used by the trainer

## Validation Command

The script finishes by running a 1-step SFT validation:

```bash
python -m prime_rl.entrypoints.sft @ configs/debug/sft/train.toml \
  --output-dir outputs/amd_debug_sft \
  --clean-output-dir \
  --max-steps 1 \
  --data.seq-len 128 \
  --data.batch-size 1 \
  --data.micro-batch-size 1 \
  --model.name Qwen/Qwen3-0.6B \
  --model.attn sdpa \
  --model.debug.num-layers 2
```

This path was validated to complete successfully on MI210.

## Operating Notes

- Keep `model.cp = 1`. The AMD path documented here does not rely on ring-flash-attention.
- Use `--model.attn sdpa` for AMD unless you have separately validated another backend.
- Expect the trainer to log non-fatal `cache_position` documentation warnings from some model classes during import.
- Expect a ROCm allocator warning about `PYTORCH_CUDA_ALLOC_CONF`; it does not block the validated SFT run.
- PRIME platform monitoring is optional. Local trainer startup should not require `prime_cli` unless Prime monitoring is explicitly configured.
- The RL smoke script forces vLLM's ROCm simple-compile backend to `eager` to avoid a Torch Inductor sampler crash on this MI210 setup.
- The locally validated ROCm RL path also requires the editable `vllm` install to keep `vllm/v1/sample/ops/logprobs.py::batched_count_greater_than` eager. On this machine, the compiled version crashed with `KernelMetadata.cluster_dims` errors during chat-completion logprobs.
- The local PRIME-RL tree now falls back cleanly when `transformers.core_model_loading` is absent. This matters for the ROCm RL path because the locally built `vllm` install downgraded `transformers` to `4.57.6`.

## Known Limits

- The documented RL smoke path currently depends on a local repo environment module because the external `reverse-text` package was not reachable from this machine.
- `scripts/install_amd.sh` still installs only the trainer-oriented dependencies. The ROCm `vllm` bring-up was performed separately.
- The validated RL bring-up here used a locally editable ROCm `vllm` checkout rather than a stock wheel, plus the eager logprob helper workaround described above.
- The upstream README still targets NVIDIA as the primary supported path.
- Peak FLOPS accounting falls back to an A100 baseline because MI210 is not explicitly modeled in the current perf table.
