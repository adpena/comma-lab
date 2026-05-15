# PR101 FEC6 Paired CPU/CUDA Axis Xray - 2026-05-15

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
evidence_grade: diagnostic_only_paired_exact_axis_artifacts

## Artifact

New xray tool:

```bash
.venv/bin/python tools/xray_paired_cpu_cuda_axis_delta.py \
  --cpu-auth-eval-json experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json \
  --cuda-auth-eval-json experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json \
  --cpu-inflated-outputs-manifest experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/inflated_outputs_manifest.json \
  --cuda-inflated-outputs-manifest experiments/results/modal_auth_eval/archive_6bae0201fb08/inflated_outputs_manifest.json \
  --label pr101_fec6_fixed_huffman_k16 \
  --target-score 0.192 \
  --output-dir experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex
```

Generated outputs:

- `experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/paired_axis_delta.json`
- `experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/paired_axis_delta.md`
- `experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/rebuild_command.txt`

## Finding

The PR101 FEC6 fixed-Huffman K16 packet remains a valid `[contest-CPU]`
near-miss and a non-promotable `[contest-CUDA]` result:

- archive sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- archive bytes: `178517`
- `[contest-CPU]`: `0.1920513168811056`
- `[contest-CUDA/T4]`: `0.22621002169349796`

The paired-axis xray quantifies why byte-only polishing is the wrong next move
for CUDA:

- total CUDA-minus-CPU score delta: `0.03415870481239236`
- byte-equivalent of that gap: `51300.21103151698` bytes
- dominant component: PoseNet contribution
- seg contribution delta: `0.010270000000000001`
- pose contribution delta: `0.02388870481239234`
- rate contribution delta: `0.0`

Target `<0.192` byte gaps if components are unchanged:

- `[contest-CPU]`: `78` bytes
- `[contest-CUDA]`: `51378` bytes

Inflated raw-output aggregate hashes differ across axes:

- CPU aggregate: `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d`
- CUDA aggregate: `6fe2b1941f10e4f984dcda96f84acb60b06c353cc8868fba33b7f3a32508f9ed`

## Entropy Cross-Check

Also regenerated section entropy xray for PR101 FEC6 vs PR106 PacketIR format
`0x0A`:

```bash
.venv/bin/python tools/xray_archive_section_entropy_heatmap.py \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --archive experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_sidecar_format_0x0a.archive.zip \
  --label pr101_fec6_k16 \
  --label pr106_packetir_format0a \
  --output-dir experiments/results/xray_entropy_pr101_fec6_vs_pr106_packetir_20260515_codex
```

Result:

- PR101 FEC6: one `x` member, payload entropy `7.9987` bpb, estimated floor
  headroom `30` bytes.
- PR106 PacketIR format `0x0A`: one `x` member, payload entropy `7.9986` bpb,
  estimated floor headroom `32` bytes.

The entropy result independently supports the same conclusion: the current
single-member packets are saturated for generic entropy recoding. Future score
movement must come from a different transform, component-moving selector, or
trained substrate, not another generic byte recode.

## Decision

Do not spend another exact eval on rate-only polishing for the current PR101
FEC6 selector packet. Reopen this family only through:

1. CUDA-in-loop per-pair/per-mode component rows;
2. a charged selector objective that prices bytes and scorer components;
3. exact CPU and exact CUDA eval on a new byte-closed packet;
4. or a larger substrate change that moves PoseNet/SegNet rather than archive
   bytes alone.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_xray_paired_cpu_cuda_axis_delta.py
.venv/bin/ruff check tools/xray_paired_cpu_cuda_axis_delta.py tests/test_xray_paired_cpu_cuda_axis_delta.py
.venv/bin/python -m py_compile tools/xray_paired_cpu_cuda_axis_delta.py
```

Results:

- `3 passed`
- `All checks passed!`
- py_compile passed
