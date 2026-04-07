---
name: prime-rl-amd-rl-mi210
description: >
  Validated single-node PRIME-RL ROCm path for AMD MI210 using 4 GPUs.
  Use when: bringing up local RL with 1 inference GPU and 3 training GPUs on gfx90a.
metadata:
  short-description: "Working MI210 RL recipe for PRIME-RL"
  tags:
    - research
    - rocm
    - amd
    - rl
    - vllm
  domain: research
  created: 2026-04-07
  author: Codex
---

# PRIME-RL AMD RL MI210

## General Description

This skill captures the validated end-to-end RL bring-up for PRIME-RL on `4x AMD Instinct MI210` (`gfx90a`). It covers the stable runtime configuration, the required ROCm compatibility constraints, and the concrete failure modes that blocked rollout, training, weight broadcast, and final checkpointing on this machine.

The validated path is local and explicit: editable PRIME-RL, editable ROCm `vllm`, `sdpa`, `model.cp = 1`, `1` inference GPU, `3` training GPUs, and a local smoke environment module instead of the external `reverse-text` dependency.

## When to Apply

Use this knowledge when:
- You need a local single-node PRIME-RL RL smoke run on MI210 / gfx90a
- You are using `1` inference GPU and `3` training GPUs
- You need local W&B-logged validation before launching a long tmux run

Do NOT use when:
- You need a stock-wheel `vllm` recipe without local source edits
- You need a validated AMD context-parallel path beyond `cp = 1`
- You need to assume the external `reverse-text` package works on the machine

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| GPUs | `4x MI210` | `1 infer + 3 train` |
| Model | `Qwen/Qwen2.5-3B-Instruct` | Validated smoke model |
| Attention | `sdpa` | Validated |
| Context parallelism | `cp = 1` | Required for documented path |
| Sequence length | `256` | Validated |
| Orchestrator batch size | `8` | Validated |
| Rollouts per example | `4` | Validated |
| Clean validation | `MAX_STEPS=2` | Rollout + train + broadcast + final checkpoint passed |

## Recommended Practice

Use the repo smoke script with explicit ROCm environment variables, local loopback proxy exclusions, and the local smoke environment module.

### Step 1: Use the validated environment

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate primerl
```

### Step 2: Use the AMD RL smoke launcher

```bash
RUN_NAME=amd-mi210-rl-validate \
OUTPUT_DIR=outputs/amd_mi210_4gpu_rl_validate \
MAX_STEPS=2 \
bash scripts/run_amd_smoke_rl.sh
```

### Step 3: Only launch the long run after the short run is green

```bash
RUN_NAME=amd-mi210-rl-overnight \
OUTPUT_DIR=outputs/amd_mi210_4gpu_rl_overnight \
MAX_STEPS=15000 \
bash scripts/run_amd_smoke_rl.sh
```

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| `pynvml`-dependent launcher assumptions | ROCm environment did not provide NVML | RL launcher must fall back to `torch.cuda` and respect `HIP_VISIBLE_DEVICES` |
| External `reverse-text` env dependency | Package access was unavailable from this machine | Keep a local smoke environment module for AMD bring-up |
| Localhost inference requests failed or hit proxy responses | Host proxy vars captured loopback traffic | Set both `NO_PROXY` and `no_proxy` for `127.0.0.1,localhost` |
| vLLM chat-completion logprobs crashed on ROCm | Triton/Inductor `KernelMetadata.cluster_dims` failure in logprob helper | Keep `vllm/v1/sample/ops/logprobs.py::batched_count_greater_than` eager in the editable ROCm checkout |
| Trainer loss helper compile crash | Same ROCm Triton metadata failure in `selective_log_softmax` / `compute_entropy` | Skip `torch.compile` for these helpers on ROCm |
| `transformers.core_model_loading` import failures | ROCm `vllm` install downgraded `transformers` to `4.57.6` | PRIME-RL must fall back cleanly when that helper module is absent |

## Configuration

```yaml
conda_env: primerl
gpu: AMD Instinct MI210
gcn_arch: gfx90a
model: Qwen/Qwen2.5-3B-Instruct
attention_backend: sdpa
context_parallelism: 1
deployment:
  num_infer_gpus: 1
  num_train_gpus: 3
seq_len: 256
orchestrator_batch_size: 8
rollouts_per_example: 4
max_completion_tokens: 32
```

## References

- Related notes: `docs/amd.md`
- Related launcher: `scripts/run_amd_smoke_rl.sh`
- Related config: `configs/smoke/amd_mi210_4gpu_rl.toml`
- Related log entry: `references/experiment-log.md`
