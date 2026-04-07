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

        logprobs_ops.batched_count_greater_than = batched_count_greater_than_rocm
        sampler.batched_count_greater_than = batched_count_greater_than_rocm


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
