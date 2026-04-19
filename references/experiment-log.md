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
## 2026-04-19 — PRIME-RL AMD MI210 Async-4 Overnight Reverse-Text Retrospective

**Type:** Retrospective
**General description:** The async-4 MI210 overnight run completed end-to-end and increased sequence length plus off-policy depth, but it never achieved nonzero reward and mostly generated gibberish by the end.

### Details

Ran a local single-node RL overnight job on `4x AMD Instinct MI210` using `1` inference GPU and `3` training GPUs with `Qwen/Qwen2.5-3B-Instruct`, `sdpa`, `cp = 1`, shared W&B logging, and explicit async overlap. The final launch used `max_async_level = 4`, `trainer.model.seq_len = 512`, `orchestrator.seq_len = 512`, `orchestrator.batch_size = 16`, `orchestrator.rollouts_per_example = 4`, `orchestrator.max_inflight_rollouts = 128`, `orchestrator.sampling.max_completion_tokens = 256`, and longer local `smoke-reverse-text` examples with `min_length = 64` and `max_length = 128`.

To avoid the previously documented overnight host OOM mode, the run kept periodic resume checkpoints but disabled periodic HF master-weight gathering with `trainer.ckpt.skip_gather_master_weights = true`, while retaining `ckpt.interval = 200` and `keep_last = 1`.

Operationally, the run was stable. The launcher, inference server, orchestrator, and trainer all started cleanly, W&B shared mode synced successfully, and the full `1000` training steps completed. Trainer throughput settled around `320-329 tokens/s`, peak trainer memory reached `28.3 GiB`, and the trainer finished with a final checkpoint and clean shutdown. On the orchestrator side, the async target was reached and maintained: from early in the run onward it stayed at `Async Level: 4`.

The main negative result is that the run did not learn. Reward remained `0.0000` for all `1000` logged orchestrator steps. As the run progressed, sequence length increased and often hit the generation cap, but rollout quality degraded sharply: late in training the orchestrator was repeatedly logging `13-16 / 16` rollouts as gibberish. Off-policy pressure also rose well beyond the intended async gap because stale in-flight requests kept accumulating across updates. `Max. Off-Policy Level` reached `8` by step `12` and stayed there for most of the run, with repeated cancellation bursts of old rollout requests.

### Key Points

- The AMD MI210 local async RL path is stable enough for a real overnight run with W&B, periodic resume checkpoints, and clean final shutdown.
- Increasing sequence length and in-flight rollout depth did increase memory and async pressure, but only modestly increased trainer VRAM: peak trainer memory rose to `28.3 GiB`.
- The overnight configuration optimized for async overlap rather than learning quality. The run saturated the toy reverse-text environment with long, mostly gibberish completions and never achieved positive reward.
- `max_async_level = 4` did not cap the reported off-policy depth in this setup because `max_off_policy_steps` remained at its default and stale requests accumulated across weight updates.
- Skipping periodic HF master-weight gathering appears to be the right operational safeguard for long MI210 runs on this machine.

### Links

- Overnight run: `outputs/amd-mi210-overnight-async4-vram512-20260418_181258/`
- W&B: `https://wandb.ai/erlandpg/prime-rl/runs/3f274068efc043fbacb01402904034ae`
- Status snapshot: `outputs/amd-mi210-overnight-async4-vram512-20260418_181258/STATUS.md`

## 2026-04-08 — PRIME-RL AMD MI210 New-Node Bring-Up Retrospective

**Type:** Retrospective
**General description:** The higher-RAM MI210 node did not reproduce the old Slurm OOM, but it exposed a non-portable editable `vllm` install and a missing ROCm logprob workaround in the active shared checkout.

### Details

Investigated why the same `primerl` environment failed on a new MI210 node even though the cluster and GPU type were unchanged. The first root cause was environment portability: `primerl` still contained editable `vllm` metadata pointing to `/tmp/vllm-v0.19.0-rocm`, a node-local source path that did not exist on the new machine. This meant the effective runtime was not actually the same across nodes.

Rebuilt the previously working ROCm `vllm` version (`v0.19.0`, commit `2a69949bd`) into shared storage under `forge-workspace/vllm-v0.19.0`, cleaned the stale editable metadata, and confirmed `primerl` now resolved `vllm` from shared storage instead of `/tmp`. After that, the next blocker matched the previously documented MI210 ROCm issue: the active shared `vllm` checkout did not yet contain the eager workaround for `vllm/v1/sample/ops/logprobs.py::batched_count_greater_than`, so inference crashed again with the `KernelMetadata.cluster_dims` Torch Inductor error until the workaround was re-applied.

Once the shared `vllm` tree was patched, the inference server became healthy again and served real logprob-bearing `/v1/chat/completions` requests on the new node. The higher-RAM job did not show the old Slurm cgroup kill signature during these runs: `/sys/fs/cgroup/system.slice/slurmstepd.scope/job_40515/memory.events` remained at `oom 0` and `oom_kill 0`.

The shortened `--ckpt.interval 10` checkpoint probe still did not conclusively answer the original checkpoint question on this node. One attempt was contaminated by stale trainer processes from an earlier failed run, leading to a misleading NCCL watchdog timeout. A later clean rerun was blocked by Hugging Face metadata/network timeouts before inference became fully healthy. So the new-node retrospective changes the diagnosis, but it does not yet produce a final answer on the step-10 checkpoint behavior.

### Key Points

- The new-node failure was not caused by MI210 hardware differences; it was caused by a non-portable editable `vllm` install and missing shared-checkout patch state.
- The old `64G` Slurm OOM / `SIGKILL` symptom has not been observed on the higher-RAM node so far.
- For this workflow, "same conda env name" is not enough. The editable source path and local ROCm patch state must also match.
- Distributed timeouts after failed bring-up are not trustworthy until stale trainer and inference processes are removed from the node.

### Links

- Broken new-node repro: `outputs/amd_mi210_4gpu_pipelinerl_128g_ckpt10_fixed_20260408/`
- Shared-checkout patched repro: `outputs/amd_mi210_4gpu_pipelinerl_128g_ckpt10_fixed2_20260408/`
- Clean rerun blocked by network timeouts: `outputs/amd_mi210_4gpu_pipelinerl_128g_ckpt10_clean_20260408/`
- Shared ROCm vLLM checkout: `/vast/users/qirong.ho/forge-workspace/vllm-v0.19.0/`

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

## 2026-04-19 — PRIME-RL AMD MI210 Mini GLM MoE RL Smoke

**Type:** Retrospective
**General description:** Validated the prebuilt mini GLM MoE RL smoke on `4x MI210` after applying the missing ROCm MoE inference and trainer fallbacks.

### Details

Started from the mirrored Hugging Face model `Erland/mini-glm-moe` and ran the local `1 infer + 3 train` MI210 RL smoke path with `seq_len = 256`, `orchestrator.batch_size = 8`, `rollouts_per_example = 4`, `max_completion_tokens = 32`, and W&B shared mode.

The first MoE run failed on the inference side even though the existing ROCm logprob eager workaround was already in place. vLLM crashed during EngineCore initialization in `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py::grouped_topk` with the same Torch Inductor `KernelMetadata.cluster_dims` error that had already been seen in other ROCm-compiled helpers. The working fix was to keep that grouped-topk router eager in the editable ROCm `vllm` checkout, not just inside the local PRIME-RL process.

Once inference was healthy, the next blocker moved to the trainer. PRIME-RL's custom MoE layer attempted to use `torch._grouped_mm`, which failed on MI210 with `RuntimeError: grouped gemm is not supported on ROCM`. The validated trainer-side fix was to disable grouped GEMM explicitly via `--trainer.model.no-moe-use-grouped-mm`.

With both fallbacks in place, the `12`-step `Erland/mini-glm-moe` smoke run completed end to end, including rollout generation, trainer updates, final checkpoint writing, and W&B sync. Peak trainer memory reached `6.2 GiB` per training GPU.

### Key Points

- Dense-model ROCm validation was not sufficient for MoE. The MoE router added a second vLLM compile surface that also needed an eager fallback.
- On MI210, trainer-side custom MoE currently requires `moe_use_grouped_mm = false`.
- The validated `Erland/mini-glm-moe` smoke configuration now works end to end on `4x MI210`.

### Links

- Successful run: `outputs/amd-mi210-m3-erland-mini-glm-moe-rocmfix4-20260419_104947/`
- W&B: `https://wandb.ai/erlandpg/prime-rl/runs/b8483a42613c4c1ba3a86769b55c1dc4`
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
