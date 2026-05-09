# A5 Binary Q-Bits Side-Info Improvement - 2026-05-09

## Verdict

Small local A5 improvement landed. The compact binary q-bit side-info encoding
preserves the same q7/q8 SegNet-protected q-bit schedule while reducing packet
side-info bytes from `225` to `77`.

This is not a global frontier candidate. It improves the best A5 post-hoc
scalar schedule but remains far worse than the A1/PR101 public-axis anchors, so
no exact CUDA or contest-CPU spend is warranted without a stronger A5 allocation
change.

## Code Path

New reusable encoding:

- `tac.codec.frame_conditional_bit_budget.pack_frame_conditional_binary_q_bits`
- `tac.codec.frame_conditional_bit_budget.unpack_frame_conditional_binary_q_bits`
- `tools/build_pr101_frame_conditional_runtime_packet.py --q-bits-sideinfo-encoding binary_low_high_mask`

Default packet behavior remains `raw3`.

Encoding semantics:

- byte 0: `low_q_bits - 1`
- byte 1: `high_q_bits - 1`
- remaining bytes: 1-bit MSB-first selector mask, `0=low`, `1=high`
- fail-closed for schedules with more than two q-bit values

## Candidate

- Schedule:
  `reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json`
- Packet:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_binaryq_20260509_codex/packet/archive.zip`
- Archive bytes: `178,095`
- Archive SHA-256:
  `dd5da6b636266f89768661c758c96355531d8305b2bbd1bbb7063c48b7297876`
- Runtime tree SHA-256:
  `29f5cedbb471a71b84cdd5a9bcac6dcce08e14decbbbe23b188e18fcebfad2e3`
- q-bit summary: `510` pairs at q7, `90` pairs at q8
- q-bit side-info encoding: `binary_low_high_mask`
- q-bit side-info bytes: `77`
- q-bit side-info SHA-256:
  `c79500df5b91ef7e4e0d9a0a5496d3e0ec50a3f663f3cc98fa7e4fc9073ba4a9`

## macOS CPU Advisory

- Evidence grade: `[macOS-CPU advisory]`
- Score claim: `false`
- Score: `0.20101191630821824`
- PoseNet: `0.00003517`
- SegNet: `0.00063672`
- Rate term: `0.11858625`
- Samples: `600`
- Eval JSON:
  `experiments/results/pr101_a5_q7_seg_binaryq_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json`
- Dispatch claim terminal status:
  `completed_macos_cpu_advisory_improved_a5_not_promotable`

Comparison:

- Prior best A5 scalar SegNet-protected q7/q8: `0.201110` at `178,243 B`
- New binary q-bit side-info q7/q8: `0.20101191630821824` at `178,095 B`
- Improvement: about `9.8e-5` score points and `148 B`

## Commands

```bash
.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json \
  --q-bits-sideinfo-encoding binary_low_high_mask \
  --recompute-wire-contract-for-q-bits \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_binaryq_20260509_codex \
  --candidate-id pr101_a5_trust_q7_low0p85_seg_binaryq_20260509_codex \
  --force

.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_binaryq_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_binaryq_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_a5_q7_seg_binaryq_runtime_packet_20260509_codex/work \
  --json-out experiments/results/pr101_a5_q7_seg_binaryq_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

## Classification

Measured improvement inside the A5 scalar trust-region family. Not promotable
because it is non-CUDA advisory evidence and the score remains about `0.00816`
worse than the current A1 public-axis anchor.

Reactivation criteria:

- exact CUDA/contest-CPU spend only if A5 allocation improves to within `0.001`
  of A1/PR101, or if it stacks with a new A5 allocation that materially lowers
  SegNet distortion;
- next A5 work should be local/channel-aware allocation or q-bit-noise training,
  not another global scalar q7/q8 ordering;
- compact side-info should be retained as a byte compiler pass for any future
  one- or two-level q-bit packet.
