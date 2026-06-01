# Codex Findings: SNeRV Score-Weighted HF Decoder Fit Smoke

UTC: 2026-06-01T22:25:41Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Lane: `lane_snerv_score_weighted_hf_decoder_fit_smoke_20260601`

## What Landed

Implemented a score-aware HF decoder fit hook for the existing numpy-portable
SNeRV linear decoder:

- `fit_hf_decoder_weighted_least_squares(...)` in
  `src/tac/substrates/snerv_inverse_steg_carrier/carrier.py`
- package export in `src/tac/substrates/snerv_inverse_steg_carrier/__init__.py`
- advisory knobs:
  - `--hf-decoder-fit-mode least_squares|score_weighted`
  - `--hf-decoder-saliency-gain`

This keeps the receiver decoder unchanged: it is still a small deterministic
linear HF predictor. The new fit path changes only how its shared weights are
fit, using DWT-domain saliency weights from the scorer-derived pixel cotangent.

## Verification

```text
.venv/bin/ruff check src/tac/substrates/snerv_inverse_steg_carrier/carrier.py src/tac/substrates/snerv_inverse_steg_carrier/__init__.py src/tac/substrates/snerv_inverse_steg_carrier/advisory.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_carrier.py tools/run_snerv_inverse_steg_advisory.py
All checks passed!

.venv/bin/python -m pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_carrier.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_archive_replay.py -q
15 passed in 0.60s
```

The new carrier test is not a stub: it verifies the weighted solver changes
decoder kernels and reduces the weighted HF residual objective versus the
unweighted fit on synthetic pyramids.

## Smoke Result

Command shape:

```text
.venv/bin/python tools/run_snerv_inverse_steg_advisory.py --n-pairs 1 --levels 4 --bits-per-coeff 5.0 --step-map-coder-mode waterfill --step-map-waterfill-bits-per-coeff 6.0 --hf-decoder-fit-mode score_weighted --hf-decoder-saliency-gain 8.0
```

Advisory artifact:

- Path: `.omx/research/snerv_score_weighted_decoder_fit_waterfill_advisory_20260601T222541Z.json`
- SHA-256: `b1e0bed066f20a5aa923dabccfe0bdbbcedc5cd4c71060b9bed9078d31494c46`
- Receiver archive SHA-256: `9a4a9d6e326ee21b5685b0f25ca01f16b6798d22f92dff2249b951048217d4f4`
- Receiver replay verified: `true`
- Archive bytes: `33824`
- Step-map mode: `waterfill`
- HF decoder fit mode: `score_weighted`
- HF decoder saliency gain: `8.0`
- `d_seg_linf`: `0.02230326272547245`
- `d_pose_linf`: `5.746245861053467`
- `score_linf`: `9.833247919737238`
- `score_l2`: `9.714336066307615`

Adjudication artifact:

- Path: `.omx/research/snerv_score_weighted_decoder_fit_waterfill_adjudication_20260601T222541Z.json`
- SHA-256: `517f3f28a7d0b0912c71daf339ef8108340b6bbc1d7c885e64fb10065e8e0f4e`
- Classification: `rate_below_frontier_pose_or_seg_destroyed`
- Ready for exact eval dispatch: `false`

## Comparison To Prior Waterfill Baseline

Prior least-squares waterfill one-pair baseline:

- Path: `.omx/research/snerv_inverse_steg_advisory_waterfill_20260601T221200Z.json`
- Archive bytes: `33754`
- `d_seg_linf`: `0.02264404296875`
- `d_pose_linf`: `2.1390697956085205`
- `score_linf`: `6.911887587116307`

The score-weighted linear HF fit slightly improved `d_seg` but worsened
`d_pose` by about `2.69x` and worsened advisory score. This measured
configuration is a NO-GO.

## Verdict

NO-GO for this specific simple score-weighted linear HF decoder fit
configuration.

Do not exact-dispatch it. Do not promote it. Do not treat it as a score claim.

This does not retire score-aware decoder fitting as a family. It says that the
naive DWT-detail weighted closed-form linear predictor is the wrong place or
wrong weighting for pose at this operating point. The next decoder-fit attempt
should move the objective into the full reconstructed-frame/scorer loop or into
a learned/nonlinear HF decoder/QAT path, with the waterfill packet still in-loop.

## Next Step

Use the work order as a contract, but replace this simple linear weighted fit
with an in-loop learned decoder/QAT smoke. The success criterion is not lower
weighted HF residual; it is lower `d_pose` and `d_seg` after receiver replay at
roughly the same archive bytes.
