import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "rocm":
        import torch
        import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as grouped_topk_router
        import vllm.v1.sample.sampler as sampler
        import vllm.v1.sample.ops.logprobs as logprobs_ops
        from vllm.platforms.interface import Platform
        from vllm.platforms.rocm import RocmPlatform

        Platform.simple_compile_backend = "eager"
        RocmPlatform.simple_compile_backend = "eager"

        # vLLM's compiled logprob rank helper still triggers a ROCm/Triton
        # crash on MI210 during chat completions. Keep this path eager.
        def batched_count_greater_than_rocm(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            return (x >= values).sum(-1)

        # vLLM's compiled grouped MoE router also hits the same ROCm/Triton
        # metadata crash on MI210 for GLM-style MoE models. Keep it eager.
        def grouped_topk_rocm(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
            num_expert_group: int = 0,
            topk_group: int = 0,
            scoring_func: str = "softmax",
            routed_scaling_factor: float = 1.0,
            e_score_correction_bias: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

            if scoring_func == "softmax":
                scores = torch.softmax(gating_output, dim=-1)
            elif scoring_func == "sigmoid":
                scores = gating_output.sigmoid()
            else:
                raise ValueError(f"Unsupported scoring function: {scoring_func}")

            num_token = scores.size(0)
            if e_score_correction_bias is not None:
                original_scores = scores
                scores = scores + e_score_correction_bias.unsqueeze(0)
                group_scores = (
                    scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
                )
            else:
                group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values

            use_sorted = grouped_topk_router.envs.VLLM_BATCH_INVARIANT
            group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
                .reshape(num_token, -1)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

            if e_score_correction_bias is not None:
                topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
                topk_weights = original_scores.gather(1, topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)

            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            if routed_scaling_factor != 1.0:
                topk_weights = topk_weights * routed_scaling_factor
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        logprobs_ops.batched_count_greater_than = batched_count_greater_than_rocm
        sampler.batched_count_greater_than = batched_count_greater_than_rocm
        grouped_topk_router.grouped_topk = grouped_topk_rocm


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
