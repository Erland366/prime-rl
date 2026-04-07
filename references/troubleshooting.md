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
