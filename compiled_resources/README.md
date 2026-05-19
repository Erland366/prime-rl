# Local Compiled Resources

Project-specific curated references for PRIME-RL work live under this directory.
Each subdirectory is a domain with its own `README.md` index and per-entry
freshness timestamps.

## Domains

- `gemm_moe_learning/`: grouped GEMM and MoE implementation references for
  experiments in `testing_chamber/test_grouped_gemm.ipynb`.

## Freshness Protocol

This local folder follows the global compiled resources mechanism documented in
`~/dotfiles/compiled_resources/README.md`:

1. Read the domain `README.md` before using resources from that domain.
2. Check for new files in the domain that are not indexed.
3. Compare each resource's latest git commit date with its `INDEXED` timestamp.
4. Update only stale or missing entries, then update `LAST_INDEXED`.

Local resources take precedence over global resources for the same domain.
