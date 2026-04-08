# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

<!-- New entries go above this line -->
## 2026-04-07 — PRIME-RL AMD MI210 Overnight RL Failure Retrospective

**Type:** Retrospective
**General description:** The overnight MI210 RL run trained correctly for hours but was killed by the Slurm memory cgroup during the periodic HF weight checkpoint at step 250.

### Details

Investigated the failed overnight run in `outputs/amd_mi210_4gpu_rl_overnight_20260407/` after the trainer exited with `Signal 9 (SIGKILL)` on rank 0 at `2026-04-07 16:37:26 UTC`. The trainer-side `.distcp` checkpoint for `step_250` completed successfully, but the separate HF-compatible weight export under `weights/step_250/` remained empty.

The decisive host evidence came from the Slurm cgroup for job `40407`. The job was launched with a hard `64G` memory reservation, and `/sys/fs/cgroup/system.slice/slurmstepd.scope/job_40407/memory.events` recorded `oom_kill 1`. This rules out an application-level exception as the primary cause and indicates a host RAM kill by the scheduler.

The timing matches the heaviest checkpoint phase. The trainer entered `Saving weight checkpoint at step 250` immediately after the regular trainer checkpoint completed. That path gathers the full model weights to CPU on rank 0 before writing sharded safetensors. At nearly the same time, the inference engine was still handling the filesystem broadcast for `step_250`, including checkpoint prefetch and weight reload. The combined resident memory of trainer checkpoint materialization plus inference reload appears to have exceeded the `64G` Slurm limit.

### Key Points

- The validated MI210 RL path remains correct for rollout, training, filesystem broadcast, and short clean shutdown.
- The overnight failure was operational, not algorithmic: a Slurm cgroup OOM kill during the periodic HF weight export.
- The regular trainer checkpoint path is not the failing part; the failing part is the additional HF `weights/step_*` export done while the live RL system is still running.
- A short validation run can pass even when the overnight periodic checkpoint policy is unsafe, because the final weight export happens after the rest of the run is winding down.

### Links

- Overnight run: `outputs/amd_mi210_4gpu_rl_overnight_20260407/`
- Validation run: `outputs/amd_mi210_4gpu_rl_validate_20260407h/`
- Trainer checkpoint code: `src/prime_rl/trainer/ckpt.py`
- Weight gathering code: `src/prime_rl/trainer/weights.py`

## 2026-04-07 — PRIME-RL AMD MI210 RL Bring-Up Retrospective

**Type:** Retrospective
**General description:** Brought the single-node PRIME-RL loop to a working end-to-end state on 4x MI210 with local ROCm inference and trainer compatibility fixes.

### Details

Validated a local RL smoke path in the `primerl` conda environment on `4x AMD Instinct MI210` using `Qwen/Qwen2.5-3B-Instruct`, `1` inference GPU, `3` training GPUs, `sdpa`, `model.cp = 1`, `seq_len = 256`, `orchestrator.batch_size = 8`, and `rollouts_per_example = 4`. The clean validation run completed end-to-end with `MAX_STEPS = 2`, including rollout generation, trainer updates, filesystem weight broadcast, final checkpoint writing, and W&B sync under `erlandpg`.

The bring-up required a local `smoke-reverse-text` environment module because the external `reverse-text` package was not available from this machine. It also required several AMD-specific compatibility fixes:
- PRIME-RL launcher fallback when `pynvml` is unavailable on ROCm
- loopback proxy exclusions for local orchestrator-to-inference traffic
- ROCm vLLM logprob helper forced to eager to avoid `KernelMetadata.cluster_dims` Torch Inductor crashes
- trainer loss helpers kept eager on ROCm to avoid the same Triton metadata crash
- fallback behavior when `transformers.core_model_loading` is missing after the ROCm `vllm` install downgraded `transformers`

### Key Points

- The local single-node RL path now works on MI210 for the validated smoke configuration.
- The critical stable configuration is `sdpa` attention, `model.cp = 1`, `1 infer + 3 train`, and a locally editable ROCm `vllm`.
- The main blockers were not algorithmic; they were ROCm inference/runtime compatibility issues and package-version assumptions.
- The validated RL smoke run reached real rollout, training, weight broadcast, and final checkpoint save successfully.

### Links

- Validation run: `outputs/amd_mi210_4gpu_rl_validate_20260407h/`
- Overnight run: `outputs/amd_mi210_4gpu_rl_overnight_20260407/`
- Notes: `docs/amd.md`

## 2026-04-07 — PRIME-RL AMD MI210 Install Retrospective

**Type:** Retrospective
**General description:** Distilled the working ROCm install and debug-run path for PRIME-RL on AMD MI210 / gfx90a.

### Details

Validated a local AMD install flow in the `primerl` conda environment using `torch 2.9.1+rocm6.3` and a single-GPU SFT trainer path. The successful validation run used `Qwen/Qwen3-0.6B`, `--model.attn sdpa`, `--model.debug.num-layers 2`, `--max-steps 1`, `--data.seq-len 128`, `--data.batch-size 1`, and `--data.micro-batch-size 1`. The trainer completed successfully on `AMD Instinct MI210`.

The work also identified several startup blockers that were not essential for the validated AMD path:
- `ring_flash_attn` was imported too early even when `model.cp = 1`
- Prime monitor imports required `prime_cli` even when Prime monitoring was disabled
- local startup depended on undeclared runtime packages such as `psutil`, `dion`, and `verifiers`

### Key Points

- The single-GPU SFT trainer works on AMD ROCm with `sdpa` and `model.cp = 1`.
- The validated dependency baseline is anchored on ROCm PyTorch wheels plus editable PRIME-RL install and a small set of additional runtime packages.
- The full inference stack was not validated; `vllm` remains outside the documented AMD path.
- Import-time optional dependency handling matters for portability more than kernel support alone.

### Links

- Log: `outputs/amd_debug_sft/logs/trainer.log`
- Script: `scripts/install_amd.sh`
- Notes: `docs/amd.md`
