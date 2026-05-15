# Sub-0.192 Frontier Gate And Pair XRay

date: 2026-05-15
research_only: true
score_claim: false
dispatch_attempted: false

## Why This Landed

The near-`0.192` PR101 FEC6 packet is a legitimate `[contest-CPU]` near-miss,
but its paired `[contest-CUDA]` score is `0.22621002169349796`. The CUDA miss is
not rate dominated:

- CUDA-minus-CPU total score delta: `0.03415870481239236`
- PoseNet contribution delta: `0.02388870481239234`
- SegNet contribution delta: `0.010270000000000001`
- byte-equivalent CUDA gap to `<0.192`: `51378` bytes

Therefore additional PR101/FEC6 byte-only polishing is not frontier work unless
it is preceded by a component-moving CUDA-in-loop selector/repair artifact.

## Code Landed

- `tools/sub_0192_viability_guard.py`
  - Parses result-review/profile JSON or JSONL.
  - Computes byte-equivalent gap to the default `0.192` target.
  - Marks tactics with insufficient remaining byte mass or measured delta as
    `not_frontier_eligible`.
  - Always emits false-authority fields.

- `tools/xray_pair_component_errors.py`
  - Reads an inflated raw output and official upstream scorers.
  - Emits per-pair PoseNet, SegNet, component-score, and frame0/frame1 pixel
    error rows.
  - Gives selector/film-grain/foveation work a concrete hard-pair surface.

- `tools/operator_briefing.py` and `tools/all_lanes_preflight.py`
  - Surface `tools/xray_pair_component_errors.py` as an operator-visible XRay
    diagnostic.
  - Keep it under the existing false-authority diagnostic gate.

## Smoke Artifacts

```bash
.venv/bin/python tools/sub_0192_viability_guard.py \
  --input .omx/research/pr101_fec6_byte_escape_profile_20260515_codex.json \
  --input .omx/research/pr101_fec6_fixed_huffman_k16_cpu_result_review_20260515_codex.json \
  --input .omx/research/pr101_fec6_fixed_huffman_k16_cuda_result_review_20260515_codex.json \
  --input experiments/results/modal_auth_eval/pr106_hdm12_hlm3_fmt0c_t4_20260515T090445Z/contest_auth_eval.json \
  --input experiments/results/modal_auth_eval_cpu/pr106_format0c_exact_radix_paired_20260515T0918Z_cpu/contest_auth_eval.json \
  --format markdown \
  --output experiments/results/sub_0192_viability_guard_20260515_codex/pr101_pr106_guard.md
```

```bash
.venv/bin/python tools/xray_pair_component_errors.py \
  --inflated-dir experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/local_macos_cpu_eval_work/inflated \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --batch-size 8 \
  --max-pairs 8 \
  --top-k 5 \
  --label pr101_fec3_k8_local_cpu_smoke \
  --archive experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip \
  --output-dir experiments/results/pair_component_xray_pr101_fec3_k8_cpu_smoke_20260515_codex
```

## Next Frontier Constraint

Promote only lanes that can plausibly beat `<0.192` on the relevant axis:

1. CUDA-in-loop per-pair/per-mode component rows.
2. Charged selector or repair objective with byte accounting.
3. Byte-closed archive/runtime packet.
4. Paired exact CPU and CUDA auth eval before any frontier language.

PR106-derived transforms are higher priority than more PR101 byte polishing:
format0C shows CUDA PoseNet can improve materially while SegNet barely worsens,
which is the orthogonal signal needed for a component-moving repair.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_sub_0192_viability_guard.py \
  src/tac/tests/test_xray_pair_component_errors.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py::test_operator_briefing_xray_gate_requires_visible_false_authority_tools \
  src/tac/tests/test_operator_briefing.py::test_briefing_json_composite_has_all_three_keys -q

.venv/bin/ruff check \
  tools/xray_pair_component_errors.py \
  tools/sub_0192_viability_guard.py \
  src/tac/tests/test_sub_0192_viability_guard.py \
  src/tac/tests/test_xray_pair_component_errors.py \
  tools/operator_briefing.py \
  tools/all_lanes_preflight.py \
  src/tac/tests/test_operator_briefing.py
```

Result: focused tests and ruff passed.
