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
- You have not verified that the active editable `vllm` path resolves from shared storage rather than a node-local `/tmp` tree

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
| Overnight status at `64G` Slurm RAM | Failed | Trainer hit a Slurm cgroup OOM during periodic HF weight export at step `250` |

## Recommended Practice

Use the repo smoke script with explicit ROCm environment variables, local loopback proxy exclusions, and the local smoke environment module.

Before treating a new-node failure as a hardware or cluster regression, verify the active editable `vllm` resolution:

```bash
python -m pip show vllm
python -c "import vllm; print(vllm.__file__, vllm.__version__)"
```

If the editable project location points at `/tmp/...`, the environment is not portable across nodes and must be reinstalled from shared storage first.

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

### Step 4: Size host RAM for overnight runs, or avoid periodic HF weight exports

For the validated MI210 RL path, do not assume a short clean run proves the overnight checkpoint policy is safe. The live RL loop can exceed a small Slurm host-memory reservation when the trainer gathers a full HF checkpoint to CPU while the inference engine is also reloading broadcast weights.

Use at least `96G`, and preferably `128G`, of Slurm job memory for overnight runs if periodic HF `weights/step_*` exports remain enabled. If you want the safer policy, keep periodic trainer resume checkpoints but defer HF `weights/step_*` export until final shutdown.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| `pynvml`-dependent launcher assumptions | ROCm environment did not provide NVML | RL launcher must fall back to `torch.cuda` and respect `HIP_VISIBLE_DEVICES` |
| External `reverse-text` env dependency | Package access was unavailable from this machine | Keep a local smoke environment module for AMD bring-up |
| Localhost inference requests failed or hit proxy responses | Host proxy vars captured loopback traffic | Set both `NO_PROXY` and `no_proxy` for `127.0.0.1,localhost` |
| vLLM chat-completion logprobs crashed on ROCm | Triton/Inductor `KernelMetadata.cluster_dims` failure in logprob helper | Keep `vllm/v1/sample/ops/logprobs.py::batched_count_greater_than` eager in the editable ROCm checkout |
| `primerl` worked on one node and failed on another | Editable `vllm` metadata pointed at a node-local `/tmp` source tree instead of shared storage | Treat editable source paths as part of the runtime contract; verify `pip show vllm` before blaming cluster or GPU differences |
| Shared `vllm` rebuild still regressed on ROCm | The shared checkout was missing the previously validated eager logprob workaround | Portability requires both the right `vllm` version and the ROCm-specific patch state in the active checkout |
| Trainer loss helper compile crash | Same ROCm Triton metadata failure in `selective_log_softmax` / `compute_entropy` | Skip `torch.compile` for these helpers on ROCm |
| `transformers.core_model_loading` import failures | ROCm `vllm` install downgraded `transformers` to `4.57.6` | PRIME-RL must fall back cleanly when that helper module is absent |
| Overnight run killed at periodic checkpoint step `250` | Slurm job had `mem=64G`, and rank `0` was OOM-killed while gathering and writing the HF `weights/step_250` checkpoint during concurrent inference reload | Treat periodic HF weight export as a separate host-memory risk from trainer resume checkpoints; request more RAM or defer HF export until shutdown |
| Retry run hit distributed watchdog timeouts after an earlier failed repro | Stale trainer or inference processes were still alive on the same GPUs | Clean the node before retrying; do not treat distributed timeouts on a dirty node as conclusive evidence |

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
slurm_memory:
  minimum_recommended: 96G
  preferred_for_overnight: 128G
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
