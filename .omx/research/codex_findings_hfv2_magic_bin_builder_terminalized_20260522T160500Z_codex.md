# Codex Findings - HFV2 Magic-Bin Builder Terminalized

Date: 2026-05-22T16:05:00Z

## Summary

Terminalized the HFV2 sparse sidecar builder WIP into the canonical magic-byte
`.bin` archive contract.

Changes:

- `tools/build_hfv1_sparse_sidecar_candidate.py` now emits the HFV2 payload as
  `foveation_params.bin`, not `foveation_params.hfv2`, so the archive passes
  the existing `_KNOWN_ARCHIVE_SUFFIXES` validator without widening evaluator
  policy.
- The generated runtime dispatches on magic bytes: `HFV1` loads dense legacy
  params; `HFV2` loads pair-sparse params.
- The patcher migrates older generated HFV2 runtimes that still referenced
  `foveation_params.hfv2`, rather than returning early on `HFV2_MAGIC`.
- HFV2 missing-pair lookup now returns the encoded `default_row`, not `None`.
- `tools/prove_hfv2_sparse_inflate_parity.py` now recognizes HFV2 payloads
  carried under `foveation_params.bin`, matching the runtime and builder.
- The builder enforces the exact member order
  `["foveation_params.bin", "x"]` and fails closed if the canonical archive
  validator cannot be imported.

## Artifact

Generated local candidate:

- Output directory:
  `experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z`
- Archive SHA-256:
  `45b85d01b72ff06efecb418c5f2777ea1c0a5a0adc33fe245b19851f1eabc5ca`
- Archive bytes: `179023`
- Members: `foveation_params.bin` and `x`
- HFV2 payload bytes: `390`
- Runtime `inflate.py` SHA-256:
  `2766ea9a81f11d823d142b599c3472d47f62483a66215afc768ac25e30c5e56d`
- Output archive manifest SHA-256:
  `d2c06389033f61c261b3c2634002f60ae38eb59e5fd328458820358936e15bee`
- Sparse pair count: `16`
- Row parity exact: `true`
- Score claim: `false`

The artifact is ignored rebuildable state under `experiments/results/*`.

## Existing exact-eval terminal anchors

The same archive SHA already has recovered Modal auth-eval anchors:

- CPU: `0.3209922342821203 [contest-CPU]`
- CUDA: `0.3374449722973937 [contest-CUDA]`

Those scores are not frontier-moving; they terminalize HFV2 as an off-frontier
apparatus/procedure lane, not as a candidate for PR110 replacement.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv1_sparse_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_hfv2_sparse_sidecar_tools.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv1_sparse_sidecar_candidate.py \
  --output-dir experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --dense-archive experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip \
  --sparse-archive experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z/submission_dir_hfv2 \
  --pair-indices 0,64,79,126,162,163,293,502,507,508,515,518,526,531,537,545,546,599 \
  --batch-pairs 3 \
  --device cpu \
  --output-dir experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z/parity_sparse_pairs_plus_defaults
```

Results:

- Builder emitted archive SHA
  `45b85d01b72ff06efecb418c5f2777ea1c0a5a0adc33fe245b19851f1eabc5ca`.
- Archive whitelist validation passed inside the builder.
- Focused patcher/member-shape tests passed: `3 passed`.
- Parity proof checked 36 frames covering all 16 sparse pairs plus default
  pairs `0` and `599`.
- `output_sha256_match=true`
- `tensor_equal=true`
- `max_abs_diff=0.0`
- Frontier scanner reported no drift.

## Next

No further HFV2 exact-eval spend is recommended for this archive. Preserve the
magic-byte `.bin` pattern as a cross-substrate template, and route current
frontier effort back to MLX/DQS candidate-generation and scorer-response
calibration.
