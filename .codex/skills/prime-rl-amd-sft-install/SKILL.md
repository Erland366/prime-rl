---
name: prime-rl-amd-sft-install
description: >
  Validated AMD ROCm install path for PRIME-RL SFT on MI210 / gfx90a.
  Use when: bringing up PRIME-RL on AMD for local single-GPU trainer validation.
metadata:
  short-description: "AMD MI210 SFT bring-up recipe"
  tags:
    - research
    - rocm
    - amd
    - sft
    - install
  domain: research
  created: 2026-04-07
  author: Codex
---

# PRIME-RL AMD SFT Install

## General Description

This skill captures a validated AMD ROCm installation path for PRIME-RL on AMD Instinct MI210 (`gfx90a`). It is intended for local single-GPU SFT trainer bring-up, not the full inference stack.

The key lesson is that portability depends on both dependency selection and optional import behavior. The validated path uses ROCm PyTorch, `sdpa`, and `model.cp = 1`.

## When to Apply

Use this knowledge when:
- You need PRIME-RL running on AMD MI210 / gfx90a
- You only need the trainer path, especially SFT validation
- The default CUDA-oriented install flow does not work

Do NOT use when:
- You need a validated `vllm` inference path on AMD
- You need context parallelism with ring-flash-attention

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| GPU | AMD Instinct MI210 | ROCm-visible arch `gfx90a` |
| PyTorch | `2.9.1+rocm6.3` | Validated |
| Validation run | 1-step SFT | Completed successfully |
| Peak memory | 4.7 GiB | Reduced 2-layer Qwen3-0.6B run |
| Attention backend | `sdpa` | Validated on AMD |
| Context parallelism | `cp=1` | Required for documented path |

## Recommended Practice

Use a conda env with Python 3.12, install ROCm PyTorch wheels first, then install PRIME-RL editable with a curated dependency set instead of the default CUDA-oriented sync path.

### Step 1: Install ROCm PyTorch

```bash
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.3 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

### Step 2: Install PRIME-RL and validated runtime deps

```bash
python -m pip install -e . --no-deps
python -m pip install \
  psutil beartype "datasets==4.6.1" jaxtyping loguru pyarrow pydantic \
  tomli tomli-w torchdata wandb pyzmq aiolimiter tenacity openai rich \
  setproctitle uvloop "transformers==5.5.0" "liger-kernel>=0.5.10" \
  "pydantic-config @ git+https://github.com/samsja/pydantic_config.git@main" \
  "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e" \
  "dion @ git+https://github.com/samsja/dion.git@d891eeb" \
  "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@0760204"
```

### Step 3: Validate with the AMD-safe trainer command

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

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Default CUDA-oriented install flow | Repo dependency sources are CUDA-pinned | Use a curated ROCm install path |
| Eager `ring_flash_attn` import | `flash_attn` unavailable on AMD path | Import ring-flash-attn only when CP is enabled |
| Eager Prime monitor import | `prime_cli` missing for local runs | Keep optional monitor backends lazily imported |
| `random_init` reduced-layer validation | Meta-device load path incompatible for this setup | Use real model weights for Qwen validation |
| Assuming inference stack works too | `vllm` was not validated on AMD | Keep scope explicit: trainer-only |

## Configuration

```yaml
conda_env: primerl
python: "3.12"
torch: "2.9.1+rocm6.3"
attention_backend: sdpa
context_parallelism: 1
validation_model: Qwen/Qwen3-0.6B
validation_num_layers: 2
validation_steps: 1
validation_seq_len: 128
```

## References

- Related reports: `references/experiment-log.md`
- Related docs: `docs/amd.md`
- Related script: `scripts/install_amd.sh`
