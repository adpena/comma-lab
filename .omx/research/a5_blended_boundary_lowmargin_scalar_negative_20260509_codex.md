# A5 Blended Boundary/Low-Margin Scalar Negative - 2026-05-09

## Verdict

Measured configuration retired. The conservative blended A5 q-bit scalar
schedule does not beat the best prior scalar A5 q7/q8 advisory point.

This is not a family kill. It retires only the tested global per-pair
`boundary + low_margin` max-risk scalar schedules.

## Tooling Landed

`tools/build_a5_score_marginal_qbits_schedule.py` now supports blended risk
schedules:

```bash
--blend-sources boundary,low_margin
--blend-mode max
```

Semantics: each source is converted into normalized risk ranks where `1` means
"protect at base q bits"; `max` is the conservative union of risk signals. Low
q bits are assigned only to the lowest blended-risk pairs.

Focused test:

```bash
.venv/bin/python -m pytest tests/test_build_a5_score_marginal_qbits_schedule.py -q
```

Result: `6 passed`.

## Byte-Gate Bracket

### q7/q8 low0p75 boundary+low-margin max

- Schedule:
  `reports/a5_score_marginal_trust_region_q7_low0p75_boundary_lowmargin_max_20260509.json`
- Packet:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p75_boundary_lowmargin_max_20260509_codex/packet/archive.zip`
- Archive bytes: `178453`
- Archive SHA-256:
  `d0fa355e79daad4f44b94a278409cffe4a7b8b90f1956b4c6c2b8b69d341f008`
- Classification: byte-gate fail. It is larger than the PR101 source archive
  while still being a lossy transform, so it was not eval-spent.

### q7/q8 low0p85 boundary+low-margin max

- Schedule:
  `reports/a5_score_marginal_trust_region_q7_low0p85_boundary_lowmargin_max_20260509.json`
- Packet:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex/packet/archive.zip`
- Archive bytes: `178243`
- Archive SHA-256:
  `6555806c0ac47833f2dae4dcd0a6ce342deb30446dd15412f846b9c164519023`
- Runtime tree SHA-256:
  `aed6a709b7f3cc657d6b3485973c240bd9256f5926b4b834a7f9e2061200c782`
- q-bit summary: `510` pairs at q7, `90` pairs at q8.

macOS CPU advisory:

- Score: `0.2011544130392937`
- PoseNet: `0.00003520`
- SegNet: `0.00063708`
- Rate term: `0.11868475`
- Eval JSON:
  `experiments/results/pr101_a5_q7_boundary_lowmargin_max_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json`

Comparison:

- Prior best scalar SegNet-protected q7/q8: `0.201110`
- Prior low-margin q7/q8: `0.201122`
- Prior boundary q7/q8: `0.201158`
- New blended boundary+low-margin max q7/q8: `0.201154`

The blended schedule barely improves boundary-only, but it does not beat
low-margin or SegNet-protected scalar schedules. No exact CUDA/contest-CPU
spend is warranted.

## Commands

```bash
.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest reports/a5_segnet_boundary_marginals_pr101_20260509.json \
  --json-out reports/a5_score_marginal_trust_region_q7_low0p85_boundary_lowmargin_max_20260509.json \
  --candidate-id a5_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 7 \
  --low-fraction 0.85 \
  --blend-sources boundary,low_margin \
  --blend-mode max

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q7_low0p85_boundary_lowmargin_max_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex \
  --candidate-id pr101_a5_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex \
  --force

.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_boundary_lowmargin_max_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_a5_q7_boundary_lowmargin_max_runtime_packet_20260509_codex/work \
  --json-out experiments/results/pr101_a5_q7_boundary_lowmargin_max_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

## Reactivation Criteria

- Replace global per-pair scalar ranking with local within-pair/channel
  boundary-aware allocation.
- Train or fine-tune with q-bit noise in the loop so the renderer adapts to
  the packet wire contract.
- Reopen exact eval only if macOS advisory beats the current best A5 scalar
  point by at least `0.001`, or if byte savings become large enough to pay the
  observed SegNet distortion.
