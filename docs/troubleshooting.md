# Troubleshooting

> My API keeps timing out.

We already set much larger timeout limits for the API clients that we use for training and evals. If you still encounter API timeout or connection errors, then this may be caused by your OS limiting the number of open file descriptors. Try increasing the maximum number of open files with

```bash
ulimit -n 32000
```

> I'm getting CUDA out of memory errors.

Assuming this is happening on the RL or SFT trainer, you can try the following:
- Use full activation checkpointing (`--model.ac`)
- Reduce the the micro batch size (`--data.micro-batch-size`) and sequence length (`--data.seq-len`)
- (*Experimental*) Use context parallelism with `--model.cp`

> I cannot pass my TOML config file

Check that you *did* leave a whitespace between the `@` and the config file (e.g. `uv run ... @ path/to/config.toml` instead of `uv run ... @path/to/config.toml`). Also, make sure that your TOML config matches the configuration schema. If not, the Pydantic error message (which arguably is quite ugly) will hopefully point you in the right direction.

> Importing the trainer fails because `ring_flash_attn` or `flash_attn` is missing.

`ring_flash_attn` is only required when context parallelism is enabled. If you are running a single-GPU debug or benchmark path, keep `model.cp = 1` and switch the attention backend to `sdpa` so the trainer does not require FlashAttention kernels:

```bash
uv run sft @ configs/debug/sft/train.toml --model.attn sdpa
```

The same applies to the RL trainer:

```bash
uv run trainer @ configs/debug/rl/train.toml --model.attn sdpa
```

> Local trainer startup fails with `ModuleNotFoundError: prime_cli`.

The Prime monitor backend is optional. If you are not configuring Prime monitoring for the run, PRIME-RL should not need the `prime` client package just to start a local trainer. Upgrade to a version that lazily imports the Prime monitor backend, or disable the Prime monitor for the run.
