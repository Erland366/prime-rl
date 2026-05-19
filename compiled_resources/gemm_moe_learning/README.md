# gemm_moe_learning Resources

<!--
================================================================================
FRESHNESS TRACKING SYSTEM - LLM MUST READ THIS FIRST
================================================================================

PURPOSE: This README indexes resources for the gemm_moe_learning domain. To save tokens, we track freshness
PER-ENTRY so you only re-process what changed, not the entire file.

GLOBAL TIMESTAMP (detect NEW untracked resources):
-->
<!-- LAST_INDEXED: 2026-05-03 -->
<!--

================================================================================
LLM FRESHNESS CHECK PROTOCOL
================================================================================

STEP 1: DETECT NEW RESOURCES (run once per session)
        Compare folder contents against entries in this README.
        ```bash
        cd /vast/users/qirong.ho/erland/Python_project/prime-rl/compiled_resources/gemm_moe_learning
        for f in */ *.md *.pdf *.py 2>/dev/null; do
          [ -e "$f" ] || continue
          name=$(basename "$f")
          grep -qF "$name" README.md || echo "NEW: $f"
        done
        ```
        -> If "NEW" resources found: add entries for them.

STEP 2: CHECK WHICH ENTRIES ARE STALE (only when needed)
        Each entry has: <!-- INDEXED: YYYY-MM-DD -->
        Compare against git commit date:
        ```bash
        git log -1 --format="%as" -- <resource_name>/
        ```
        -> If git date > INDEXED date: that entry is STALE.

STEP 3: UPDATE ONLY STALE ENTRIES
        - Read ONLY the stale resource (not all resources!).
        - Update ONLY that entry's bullet points.
        - Update ONLY that entry's <!-- INDEXED: YYYY-MM-DD --> to today.
        - Leave fresh entries untouched.

STEP 4: UPDATE GLOBAL TIMESTAMP
        After any changes, update LAST_INDEXED above to today's date.

================================================================================
ENTRY FORMAT (every entry follows this structure)
================================================================================

## N. ResourceName **DD Month YYYY**
<!-- INDEXED: YYYY-MM-DD -->
- Summary bullet with file reference (`file:line-range`)
- More bullets...

### How to use (optional)
```python
# Code example if applicable
```

================================================================================
-->

## Resources

## 1. mixtral_modeling_mixtral_lines_61_135 **03 May 2026**
<!-- INDEXED: 2026-05-03 -->
- Source reference for Hugging Face Mixtral MoE components around the MLP and sparse MoE block, especially router logits, top-k expert selection, expert dispatch, and weighted aggregation.
- Use this as the baseline readable PyTorch implementation when studying MoE routing before grouped GEMM optimization.
- File: `mixtral_modeling_mixtral_lines_61_135.md`
- Original URL: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L61-L135

## 2. transformers_pr_42697_experts_backends **03 May 2026**
<!-- INDEXED: 2026-05-03 -->
- PR adding Transformers experts backend plumbing for MoE, including eager, batched_mm, and grouped_mm implementations.
- Key study files include src/transformers/integrations/moe.py, docs/source/en/experts_interface.md, modeling_utils.py, and the MoE model integration call sites such as Mixtral.
- File: `transformers_pr_42697_experts_backends.md`
- Original URL: https://github.com/huggingface/transformers/pull/42697/changes#diff-26783ca033d92b4ce2e01eb691dbf5b05dd972b0cbfdc69fc726cf77a9dcb011

## 3. transformers_pr_42697_integrations_moe_snapshot **03 May 2026**
<!-- INDEXED: 2026-05-03 -->
- Focused snapshot of the PR's new `src/transformers/integrations/moe.py` implementation at head commit `2ebaba2d481af235a20f012572c41e558cb1e908`.
- Includes the `batched_mm_experts_forward`, `grouped_mm_experts_forward`, `ExpertsInterface`, and `use_experts_implementation` code paths.
- File: `transformers_pr_42697_integrations_moe_snapshot.md`
- Original PR: https://github.com/huggingface/transformers/pull/42697
