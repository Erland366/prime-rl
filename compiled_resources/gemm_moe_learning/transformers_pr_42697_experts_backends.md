# PR #42697: batched and grouped experts implementations

**State:** MERGED | **Author:** @IlyasMoutawwakil | **Created:** 2025-12-08 09:11 UTC | **Merged:** 2026-01-05 09:53 UTC
**Branch:** moe-imp → main

---

## Description

# What does this PR do?

I have started experimenting with pure pytorch MoE implementations following the [HF exporters PR](https://github.com/huggingface/transformers/pull/41992) while trying to find a traceable/exportable variant for onnx/openvino.

In this PR I copy the `attn_implementation` API into a similar `experts_implementation` API, and added two new implementations:
- `batched_mm` (the exportable one) which uses `torch.bmm`, is fastest on single batch size / small inputs.
- `grouped_mm` (the pytorch custom kernel one) inspired from torchtitan's moe imp (using `torch._grouped_mm`), which is generally fastest.

# benchmark
An initial benchmark shows promising results on (A100), I know that the `torch._grouped_mm` uses bfloat16 or something under the hood, so these might not be apples to apples (i'm still looking for more references on this function and how to use it "equivalently")

# MoE Implementations Benchmark

Benchmark script: [bench.py](https://github.com/user-attachments/files/24125816/bench.py)

It uses qwen2_moe ("Qwen/Qwen1.5-MoE-A2.7B", bfloat16) where latency and memory are for the forward pass / prefill

TLDR; for very small inputs batched_mm can be extremely fast and even faster with compilation, for bigger inputs grouped_mm is unbeatable but it doesn't seem to get much faster with torch compilation.

| Batch Size | Seq Length | Torch Compile              | Implementation | Mean Latency (ms) | Median Latency (ms) | P90 Latency (ms) | Peak Mem (MB) |
| ---------- | ---------- | -------------------------- | -------------- | ----------------- | ------------------- | ---------------- | ------------- |
| 1          | 16         | False                      | eager          | 271.80            | 272.94              | 295.34           | 27324.65      |
| 1          | 16         | True                       | eager          | 351.86            | 351.64              | 384.64           | 27329.29      |
| 1          | 16         | max-autotune-no-cudagraphs | eager          | 352.52            | 352.15              | 382.79           | 27329.29      |
| 1          | 16         | False                      | batched_mm     | 52.03             | 52.07               | 52.67            | 28382.50      |
| 1          | 16         | True                       | batched_mm     | 53.04             | 53.04               | 53.11            | 28029.63      |
| 1          | 16         | max-autotune-no-cudagraphs | batched_mm     | 23.87             | 23.86               | 24.02            | 27329.29      |
| 1          | 16         | False                      | grouped_mm     | 64.27             | 64.09               | 65.49            | 27329.29      |
| 1          | 16         | True                       | grouped_mm     | 59.45             | 59.52               | 60.99            | 27329.29      |
| 1          | 16         | max-autotune-no-cudagraphs | grouped_mm     | 59.61             | 59.55               | 60.89            | 27329.29      |
| 1          | 128        | False                      | eager          | 471.73            | 472.65              | 487.97           | 27396.46      |
| 1          | 128        | True                       | eager          | 637.32            | 613.70              | 845.01           | 27429.82      |
| 1          | 128        | max-autotune-no-cudagraphs | eager          | 620.21            | 619.35              | 657.74           | 27429.82      |
| 1          | 128        | False                      | batched_mm     | 316.67            | 316.94              | 317.92           | 35854.56      |
| 1          | 128        | True                       | batched_mm     | 370.29            | 370.29              | 370.57           | 33031.64      |
| 1          | 128        | max-autotune-no-cudagraphs | batched_mm     | 151.87            | 150.38              | 158.01           | 27429.82      |
| 1          | 128        | False                      | grouped_mm     | 78.50             | 78.53               | 80.00            | 27429.82      |
| 1          | 128        | True                       | grouped_mm     | 72.95             | 72.99               | 74.60            | 27429.82      |
| 1          | 128        | max-autotune-no-cudagraphs | grouped_mm     | 72.71             | 72.89               | 73.55            | 27429.82      |
| 4          | 16         | False                      | eager          | 431.87            | 433.38              | 448.01           | 27391.57      |
| 4          | 16         | True                       | eager          | 566.63            | 569.74              | 598.98           | 27372.12      |
| 4          | 16         | max-autotune-no-cudagraphs | eager          | 563.13            | 567.79              | 588.25           | 27372.12      |
| 4          | 16         | False                      | batched_mm     | 163.41            | 163.38              | 164.84           | 31585.54      |
| 4          | 16         | True                       | batched_mm     | 189.18            | 189.08              | 189.79           | 30173.45      |
| 4          | 16         | max-autotune-no-cudagraphs | batched_mm     | 79.15             | 79.10               | 79.74            | 27372.11      |
| 4          | 16         | False                      | grouped_mm     | 75.23             | 75.18               | 76.74            | 27372.11      |
| 4          | 16         | True                       | grouped_mm     | 70.35             | 70.40               | 71.71            | 27372.12      |
| 4          | 16         | max-autotune-no-cudagraphs | grouped_mm     | 70.26             | 70.43               | 71.32            | 27372.12      |
| 4          | 128        | False                      | eager          | 526.88            | 522.75              | 570.01           | 27632.62      |
| 4          | 128        | True                       | eager          | 678.18            | 677.54              | 690.97           | 27762.46      |
| 4          | 128        | max-autotune-no-cudagraphs | eager          | 676.22            | 677.07              | 681.91           | 27762.45      |
| 4          | 128        | False                      | batched_mm     | 1235.25           | 1235.33             | 1237.90          | 61465.85      |
| 4          | 128        | True                       | batched_mm     | 1505.00           | 1503.31             | 1536.10          | 50174.26      |
| 4          | 128        | max-autotune-no-cudagraphs | batched_mm     | 572.37            | 570.81              | 589.74           | 27762.45      |
| 4          | 128        | False                      | grouped_mm     | 80.95             | 81.06               | 81.70            | 27762.45      |
| 4          | 128        | True                       | grouped_mm     | 79.67             | 79.69               | 80.54            | 27762.45      |
| 4          | 128        | max-autotune-no-cudagraphs | grouped_mm     | 83.29             | 79.83               | 111.83           | 27762.46      |



## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

<!-- Your PR will be replied to more quickly if you can figure out the right person to tag with @

 If you know how to use git blame, that is the easiest way, otherwise, here is a rough guide of **who to tag**.
 Please tag fewer than 3 people.

Models:

- text models: @ArthurZucker @Cyrilvallez
- vision models: @yonigozlan @molbap
- audio models: @eustlb @ebezzam @vasqu
- multimodal models: @zucchini-nlp
- graph models: @clefourrier

Library:

- generate: @zucchini-nlp (visual-language models) or @gante (all others)
- continuous batching: @remi-or @ArthurZucker @McPatate
- pipelines: @Rocketknight1
- tokenizers: @ArthurZucker and @itazap
- trainer: @SunMarc
- attention: @vasqu @ArthurZucker @CyrilVallez
- model loading (from pretrained, etc): @CyrilVallez
- distributed: @3outeille @ArthurZucker
- CIs: @ydshieh

Integrations:

- ray/raytune: @richardliaw, @amogkam
- Big Model Inference: @SunMarc
- quantization: @SunMarc @MekkCyber
- kernels: @MekkCyber @drbh
- peft: @BenjaminBossan @githubnemo

Devices/Backends:

- AMD ROCm: @ivarflakstad
- Intel XPU: @IlyasMoutawwakil
- Ascend NPU: @ivarflakstad

Documentation: @stevhliu

Research projects are not maintained and should be taken as is.

 -->

---

## Files Changed (65 files, +1060 -374)

- `docs/source/en/_toctree.yml` (+2 -0)
- `docs/source/en/experts_interface.md` (+166 -0)
- `src/transformers/cache_utils.py` (+17 -15)
- `src/transformers/configuration_utils.py` (+29 -1)
- `src/transformers/generation/utils.py` (+12 -1)
- `src/transformers/integrations/__init__.py` (+12 -0)
- `src/transformers/integrations/moe.py` (+240 -0)
- `src/transformers/modeling_utils.py` (+126 -0)
- `src/transformers/models/afmoe/modeling_afmoe.py` (+1 -2)
- `src/transformers/models/bamba/modeling_bamba.py` (+1 -2)
- `src/transformers/models/bamba/modular_bamba.py` (+1 -1)
- `src/transformers/models/dbrx/configuration_dbrx.py` (+9 -1)
- `src/transformers/models/deepseek_v2/modeling_deepseek_v2.py` (+6 -3)
- `src/transformers/models/deepseek_v2/modular_deepseek_v2.py` (+4 -2)
- `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py` (+6 -3)
- `src/transformers/models/deepseek_v3/modular_deepseek_v3.py` (+4 -2)
- `src/transformers/models/dots1/modeling_dots1.py` (+11 -3)
- `src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py` (+7 -4)
- `src/transformers/models/ernie4_5_moe/modular_ernie4_5_moe.py` (+1 -27)
- `src/transformers/models/ernie4_5_vl_moe/modeling_ernie4_5_vl_moe.py` (+3 -2)
- `src/transformers/models/falcon_mamba/modeling_falcon_mamba.py` (+1 -1)
- `src/transformers/models/flex_olmo/modeling_flex_olmo.py` (+6 -3)
- `src/transformers/models/glm4_moe/modeling_glm4_moe.py` (+6 -3)
- `src/transformers/models/glm4_moe/modular_glm4_moe.py` (+1 -1)
- `src/transformers/models/glm4v_moe/modeling_glm4v_moe.py` (+12 -3)
- `src/transformers/models/gpt_oss/modeling_gpt_oss.py` (+1 -2)
- `src/transformers/models/gpt_oss/modular_gpt_oss.py` (+1 -1)
- `src/transformers/models/granitemoe/modeling_granitemoe.py` (+1 -2)
- `src/transformers/models/granitemoe/modular_granitemoe.py` (+1 -2)
- `src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py` (+7 -4)
- `src/transformers/models/granitemoeshared/modeling_granitemoeshared.py` (+1 -2)
- `src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py` (+11 -3)
- `src/transformers/models/hunyuan_v1_moe/modular_hunyuan_v1_moe.py` (+4 -2)
- `src/transformers/models/jamba/modeling_jamba.py` (+8 -2)
- `src/transformers/models/jamba/modular_jamba.py` (+1 -1)
- `src/transformers/models/jetmoe/modeling_jetmoe.py` (+1 -1)
- `src/transformers/models/jetmoe/modular_jetmoe.py` (+1 -0)
- `src/transformers/models/lfm2_moe/modeling_lfm2_moe.py` (+10 -3)
- `src/transformers/models/lfm2_moe/modular_lfm2_moe.py` (+2 -28)
- `src/transformers/models/mamba/modeling_mamba.py` (+1 -1)
- `src/transformers/models/mamba2/modeling_mamba2.py` (+1 -1)
- `src/transformers/models/minimax/modeling_minimax.py` (+8 -2)
- `src/transformers/models/minimax/modular_minimax.py` (+1 -1)
- `src/transformers/models/mixtral/modeling_mixtral.py` (+11 -3)
- `src/transformers/models/mixtral/modular_mixtral.py` (+6 -2)
- `src/transformers/models/olmoe/modeling_olmoe.py` (+11 -3)
- `src/transformers/models/olmoe/modular_olmoe.py` (+4 -2)
- `src/transformers/models/phimoe/modeling_phimoe.py` (+11 -3)
- `src/transformers/models/qwen2_moe/modeling_qwen2_moe.py` (+11 -3)
- `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py` (+11 -3)
- `src/transformers/models/qwen3_next/modeling_qwen3_next.py` (+2 -1)
- `src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py` (+12 -3)
- `src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py` (+67 -94)
- `src/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py` (+24 -81)
- `src/transformers/utils/__init__.py` (+1 -0)
- `src/transformers/utils/import_utils.py` (+5 -0)
- `tests/causal_lm_tester.py` (+2 -2)
- `tests/generation/test_utils.py` (+6 -28)
- `tests/models/deepseek_v3/test_modeling_deepseek_v3.py` (+3 -3)
- `tests/models/ernie4_5_vl_moe/test_modeling_ernie4_5_vl_moe.py` (+3 -3)
- `tests/models/jamba/test_modeling_jamba.py` (+1 -1)
- `tests/models/lfm2_moe/test_modeling_lfm2_moe.py` (+4 -0)
- `tests/models/olmoe/test_modeling_olmoe.py` (+1 -1)
- `tests/test_modeling_common.py` (+118 -0)
- `tests/utils/test_configuration_utils.py` (+1 -0)

---

## Commits (74)

- `bb6ac59` meo implementation
- `796998f` support more MoEs
- `46677c8` tests
- `ff40d3b` add comments
- `214a99b` add grouped_mm support
- `49726a7` typing act_fn and adding stride 16 note
- `7161590` style
- `7a0282f` Merge branch 'main' into moe-imp
- `22d88bf` fix dbrx config
- `9dcd7c3` fix config test
- `ea2f391` add licence and better stride conditions
- `c9745f3` comment
- `afa79da` no need to pad tesnors to 16 byte strides if we made sure our tiny te…
- `ff8fbd4` use a class decorator with a registration interface
- `366db5f` remove line
- `af9e166` remove unnecessary
- `4241484` register config with the decorator
- `936ea8d` fix redundant
- `e8beb7e` reduce changes some more
- `aca1a9c` fix
- `67c215b` fix
- `949db68` import from integrations
- `61f9964` Merge branch 'main' into moe-imp
- `e45de99` remove empty lines
- `ac16a7c` use histc instead of bincount
- `b7fe877` fix cpu histc not supporting long
- `60f13d0` Merge branch 'main' into moe-imp
- `6a61445` docs
- `b8614c8` added benchmark to docs
- `f232831` Merge branch 'main' into moe-imp
- `62136f5` add to from_pretrained's docstring
- `45a7437` make grouped_mm the deafault when possible
- `aedb599` Update docs/source/en/experts_interface.md
- `feea210` Update docs/source/en/experts_interface.md
- `9635cb1` Update docs/source/en/experts_interface.md
- `28ee4b0` Update docs/source/en/experts_interface.md
- `6967862` Update docs/source/en/experts_interface.md
- `7f46396` Update docs/source/en/experts_interface.md
- `22ecca1` Update docs/source/en/experts_interface.md
- `d024cb2` Update docs/source/en/experts_interface.md
- `2673c8c` Update docs/source/en/experts_interface.md
- `5770a7d` Update docs/source/en/experts_interface.md
- `0d75c7c` Update docs/source/en/experts_interface.md
- `27fae8d` Update docs/source/en/experts_interface.md
- `fdc5e9e` Update docs/source/en/experts_interface.md
- `7f35f86` Apply suggestions from code review
- `b8e3fef` Apply suggestion from @stevhliu
- `2013cf5` Apply suggestion from @stevhliu
- `ecfa0c7` Merge branch 'main' into moe-imp
- `eb42ee4` make qwen3 vl moe inherit its experts and sparse moe blocks from qwen…
- `26ac01f` create _supports_grouped_mm flag and use it for testing
- `a339296` fix copies
- `5df4f6f` better grouped mm checks
- `077cc68` fix model size failure
- `8f58984` better docs
- `bca5046` get rid of class property _supports_grouped_mm
- `106814c` add method calling checks and fix models that didn't have experts
- `d6bfd2d` fix copies
- `628a846` fix
- `69de1dc` fix
- `4085676` more cleanup
- `885a39d` clean
- `a4447fb` document compilation behaviour
- `cb0236b` docs
- `e63e296` Merge branch 'main' into moe-imp
- `203d043` fix new moe after merge
- `880f68e` fix the new ernie 4.5 vl moe testing
- `2d1df7d` support fullgraph automatic compilation for MoEs
- `8ae3556` Merge branch 'main' into moe-imp
- `707adf1` fix lazy initialization
- `cc1b469` Merge branch 'main' into moe-imp
- `8b00521` disable fullgraph for granitemoe and jetmoe because of topk gating
- `4974146` avoid implicit fallback in experts implementation and only do it when…
- `2ebaba2` style

---

## Reviews (28)

### @ArthurZucker — Comment — 2025-12-10 10:05 UTC

### @IlyasMoutawwakil — Comment — 2025-12-11 09:34 UTC

### @IlyasMoutawwakil — Comment — 2025-12-11 10:26 UTC

### @IlyasMoutawwakil — Comment — 2025-12-11 10:35 UTC

### @IlyasMoutawwakil — Comment — 2025-12-11 10:48 UTC

### @IlyasMoutawwakil — Comment — 2025-12-12 11:14 UTC

### @IlyasMoutawwakil — Comment — 2025-12-16 10:33 UTC

### @ArthurZucker — ✓ Approved — 2025-12-17 10:23 UTC

Ok very nice.
- we need a big doc page just for this + the benches on the PR and doc shows how to run them

It seems fine do de-corelate from the `@use_hf_hub _kernel` as this is EXPERT specific so yes let's go.

Let's document what this supports (compile).
Let's find a default for the impolementation to not be eager pls!

Let's make sure we have a big test eager matches gmm test please!

Make sure setting - reseting the implementation properly updates the foward

### @IlyasMoutawwakil — Comment — 2025-12-18 11:53 UTC

### @stevhliu — ✓ Approved — 2025-12-18 19:04 UTC

very nice thanks!

### @ArthurZucker — ✓ Approved — 2025-12-19 14:12 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 14:18 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 14:19 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 15:49 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 16:13 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 16:13 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 16:19 UTC

### @IlyasMoutawwakil — Comment — 2025-12-19 16:27 UTC

### @ArthurZucker — ✓ Approved — 2025-12-19 17:00 UTC

🚀

### @IlyasMoutawwakil — Comment — 2025-12-22 09:48 UTC

### @IlyasMoutawwakil — Comment — 2025-12-22 09:52 UTC

### @IlyasMoutawwakil — Comment — 2025-12-22 10:10 UTC

### @IlyasMoutawwakil — Comment — 2025-12-22 10:10 UTC

### @IlyasMoutawwakil — Comment — 2025-12-22 11:18 UTC

### @ArthurZucker — Comment — 2025-12-22 14:22 UTC

### @IlyasMoutawwakil — Comment — 2026-01-05 09:42 UTC

### @IlyasMoutawwakil — Comment — 2026-01-05 09:43 UTC

### @IlyasMoutawwakil — Comment — 2026-01-05 09:46 UTC


---

## Comments (6)

### @HuggingFaceDocBuilderDev — 2025-12-08 09:20 UTC

The docs for this PR live [here](https://moon-ci-docs.huggingface.co/docs/transformers/pr_42697). All of your documentation changes will be reflected on that endpoint. The docs are available until 30 days after the last update.

### @stevhliu — 2025-12-15 23:45 UTC

subscribing so i can add some docs for this once it's ready :)

### @IlyasMoutawwakil — 2025-12-22 11:15 UTC

I updated the branch and the latest ernie_4_5_vl_moe

### @github-actions — 2026-01-01 13:48 UTC

**[For maintainers]** Suggested jobs to run (before merge)

run-slow: afmoe, bamba, dbrx, deepseek_v2, deepseek_v3, dots1, ernie4_5_moe, ernie4_5_vl_moe, falcon_mamba, flex_olmo, glm4_moe, glm4v_moe, gpt_oss, granitemoe, granitemoehybrid

### @github-actions — 2026-01-01 14:00 UTC

View the CircleCI Test Summary for this PR:

https://huggingface.co/spaces/transformers-community/circle-ci-viz?pr=42697&sha=2ebaba

### @ArthurZucker — 2026-01-05 10:05 UTC

Kudos!

---

*Source: https://github.com/huggingface/transformers/pull/42697*