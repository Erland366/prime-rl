# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| (Template) | Describe the error message or behavior | Root cause analysis | Step-by-step fix |

## Editable vLLM install tied to /tmp breaks across nodes

**Added:** 2026-04-08
**Domain:** research

### Symptom

The same `primerl` environment works on one cluster node but fails on another, even though the GPU type is still MI210 and the repository checkout is unchanged.

Typical signs include:

```text
ModuleNotFoundError: No module named 'vllm'
```

or `pip show vllm` reporting an editable project location under `/tmp/...`.

### Cause

The environment is not actually portable if editable package metadata points at a node-local source tree. In the investigated case, `primerl` still contained `vllm` editable metadata pointing to `/tmp/vllm-v0.19.0-rocm`, so moving to a new node changed the effective runtime even though the conda environment name stayed the same.

### Solution

1. Inspect the active package resolution:
   `python -m pip show vllm`
2. Check for stale editable metadata in `site-packages`, especially:
   - `vllm.egg-link`
   - `__editable__*.pth`
   - editable finder files
   - old `direct_url.json`
3. Reinstall the editable package from shared storage, not `/tmp`.
4. Remove stale editable metadata that still points to the old node-local path.
5. Revalidate with:
   `python -c "import vllm; print(vllm.__file__, vllm.__version__)"`

### Prevention

Do not use node-local editable source trees for cluster workflows that may move across nodes. Keep editable checkouts in shared storage and verify the resolved path before diagnosing hardware or runtime regressions.

### Related

- Skill: `prime-rl-amd-rl-mi210`

## Stale PRIME-RL processes invalidate ROCm repros

**Added:** 2026-04-08
**Domain:** research

### Symptom

A later retry shows confusing trainer or NCCL failures after an earlier RL bring-up already failed.

Typical symptoms include:

```text
ProcessGroupNCCL watchdog caught collective operation timeout
```

or inexplicable hangs while GPUs appear busy from an earlier attempt.

### Cause

Failed RL bring-up can leave trainer or inference processes alive on the same GPUs. A later repro then runs on a dirty node, so the resulting timeout or hang is not a clean signal about the current code or configuration.

### Solution

1. Before retrying, inspect the node for leftover PRIME-RL or vLLM processes:
   `ps -ef | rg 'PRIME-RL::|VLLM::EngineCore|torchrun --role=trainer'`
2. Terminate stale trainer, inference, and engine processes from earlier failed runs.
3. Confirm the node is clean before launching the next repro.
4. Only treat the next failure as diagnostic once process cleanup is verified.

### Prevention

After any failed RL validation, clean up stale trainer and inference processes before changing hypotheses. Otherwise a later distributed timeout may be a contamination artifact, not a real regression.

### Related

- Skill: `prime-rl-amd-rl-mi210`

## Slurm cgroup OOM during RL weight checkpoint on MI210

**Added:** 2026-04-07
**Domain:** research

### Symptom

An overnight RL run trains normally, then the trainer dies during checkpointing with:

```text
Signal 9 (SIGKILL) received by PID ...
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
```

The run may still have a complete trainer `.distcp` checkpoint, while the matching `weights/step_*` directory is empty or incomplete.

### Cause

This can happen when the Slurm job memory reservation is too small for the periodic HF weight export. On the validated MI210 RL path, the trainer checkpoint manager writes two different artifacts:

1. the regular trainer resume checkpoint
2. a separate HF-compatible weight checkpoint that gathers the full model to CPU on rank 0 before writing sharded safetensors

If the inference engine is also reloading broadcast weights at the same time, the combined host RAM can exceed the job's Slurm cgroup memory limit. In the investigated failure, job `40407` had `mem=64G`, and the job cgroup recorded `oom_kill 1`.

### Solution

1. Confirm the failure mode with Slurm cgroup evidence:
   `cat /sys/fs/cgroup/system.slice/slurmstepd.scope/job_<JOBID>/memory.events`
2. Check whether the trainer checkpoint exists but `weights/step_*` is missing or empty.
3. Increase the Slurm memory request for overnight RL runs, preferably to `96G` or `128G`.
4. Prefer periodic trainer resume checkpoints without periodic HF `weights/step_*` exports during the live RL run.
5. If HF weights are needed, write them only at the end after inference and orchestrator shutdown.

### Prevention

Do not treat a successful short validation run as proof that the overnight checkpoint policy is safe. The CPU-memory peak from periodic HF weight export can appear only later under the full live RL loop. On this MI210 setup, size the Slurm memory budget for concurrent trainer checkpointing and inference reload, or disable periodic HF weight exports during training.

### Related

- Skill: `prime-rl-amd-rl-mi210`

## ring_flash_attn import failure on AMD

**Added:** 2026-04-07
**Domain:** research

### Symptom

Trainer import fails before startup on AMD systems without FlashAttention installed.

```text
ModuleNotFoundError: No module named 'flash_attn'
```

### Cause

`ring_flash_attn` is only needed for context parallelism, but eager imports can force it to load even when `model.cp = 1` and the run does not use context parallelism.

### Solution

1. Keep context parallelism disabled for the AMD SFT path: `model.cp = 1`.
2. Use `--model.attn sdpa` for the validated AMD trainer path.
3. Use a PRIME-RL version where `ring_flash_attn` is imported lazily for CP-only code paths.

### Prevention

Do not assume FlashAttention is optional if it is imported at module import time. Keep CP-related dependencies behind CP-only execution paths.

### Related

- Skill: `prime-rl-amd-sft-install`

## vLLM ROCm logprob compile crash on MI210

**Added:** 2026-04-07
**Domain:** research

### Symptom

Local ROCm inference starts, but `/v1/chat/completions` with logprobs crashes the engine:

```text
torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'
```

This showed up during PRIME-RL rollout generation, where the orchestrator needs token logprobs from the vLLM server.

### Cause

On this MI210 / ROCm setup, vLLM's compiled logprob rank helper in `vllm/v1/sample/ops/logprobs.py` triggers a Triton/Inductor path that assumes metadata fields not present in the generated ROCm kernel.

### Solution

1. Use the editable ROCm `vllm` checkout instead of assuming a stock wheel is sufficient.
2. Keep the affected helper eager:
   `vllm/v1/sample/ops/logprobs.py::batched_count_greater_than`
3. Keep PRIME-RL's ROCm inference path on `enforce_eager=True`.
4. Re-run a direct `/v1/chat/completions` probe with `logprobs=true` before retrying the full RL loop.

### Prevention

Do not treat "server started" as sufficient validation on ROCm. Always test the actual logprob-bearing completion path that the RL orchestrator uses.

### Related

- Skill: `prime-rl-amd-rl-mi210`

## transformers.core_model_loading missing after ROCm vLLM install

**Added:** 2026-04-07
**Domain:** research

### Symptom

Trainer-side weight broadcast or final checkpoint save fails with:

```text
ModuleNotFoundError: No module named 'transformers.core_model_loading'
```

### Cause

The locally built ROCm `vllm` install downgraded `transformers` to `4.57.6`, where `transformers.core_model_loading` is no longer present. PRIME-RL had assumed that module existed for `revert_weight_conversion`.

### Solution

1. Detect whether `transformers.core_model_loading` exists before importing it.
2. If it does exist, use `revert_weight_conversion` as before.
3. If it does not exist, log a warning and save or broadcast the gathered HF-style weights as-is.
4. Revalidate both filesystem weight broadcast and final weight checkpoint save after the fallback is added.

### Prevention

Do not hard-code imports against optional or version-sensitive internals from `transformers` without a compatibility fallback.

### Related

- Skill: `prime-rl-amd-rl-mi210`

## prime_cli import failure during local trainer startup

**Added:** 2026-04-07
**Domain:** research

### Symptom

Local trainer startup fails even though Prime platform monitoring is not configured.

```text
ModuleNotFoundError: No module named 'prime_cli'
```

### Cause

The Prime monitor backend was imported unconditionally by the monitor factory, so local runs required `prime_cli` even when no Prime monitor was requested.

### Solution

1. Keep Prime monitoring disabled unless you explicitly need it.
2. Use a PRIME-RL version where the Prime monitor backend is imported lazily.
3. Only install the Prime client package if you actually need Prime monitoring features.

### Prevention

Treat external service integrations as optional dependencies and import them only when the corresponding backend is enabled.

### Related

- Skill: `prime-rl-amd-sft-install`

## ROCm allocator config warning

**Added:** 2026-04-07
**Domain:** research

### Symptom

The trainer emits a warning on startup on ROCm:

```text
PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
```

### Cause

The launcher still sets `PYTORCH_CUDA_ALLOC_CONF`, which ROCm accepts with a warning but now considers deprecated.

### Solution

1. Treat this as non-blocking for the validated MI210 SFT path.
2. If you want a clean log, switch the launcher environment variable to `PYTORCH_ALLOC_CONF`.
3. Re-run the AMD validation command to confirm behavior is unchanged.

### Prevention

Prefer backend-neutral PyTorch allocator environment variables when sharing launchers across CUDA and ROCm.

### Related

- Skill: `prime-rl-amd-sft-install`
