# SegMap HM-S/SA Lossy Pack Contract - 2026-04-30

## Scope

Worker ownership: SegMap HM-S/SA pack and roundtrip contract only. No score
promotion or method retirement is claimed here.

## Diagnosis

Harvested failures:

- `experiments/results/live_harvest_35885106_lane_hm_s_20260430`
- `experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430`

Both runs trained successfully and stopped at Stage 3 pack. The remote scripts
used `verify_roundtrip(..., tol=1e-6)` after Selfcomp-style block-FP packing.
That tolerance is a lossless-style contract and is wrong for this intentionally
lossy renderer-weight pack.

Measured against harvested `segmap_inference.pt` and `segmap_weights.tar.xz`:

- HM-S max per-key MSE: `0.00029666198` on `layer_in.weight`
- SA max per-key MSE: `0.000322065316` on `layer_in.weight`

Shapes, keys, HWOI layout, qint/exponent pairing, and finite decode were
valid. This is not evidence that the packer is broken; it is evidence that the
post-training pack gate used the wrong contract.

## Repair

Implemented explicit `segmap_block_fp_per_channel_lossy_v1` metadata:

- per-tensor MSE gate: `SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL = 1e-3`
- metadata embedded in `segmap_weights.tar.xz` `meta.json`
- lane provenance records `segmap_pack_roundtrip.json` and marks
  `archive_level_exact_eval_required=true`
- HM-S/SA scripts continue only to canonical CUDA `contest_auth_eval.py`
  before any score claim
- preflight rejects HM-S/SA scripts that use `tol=1e-6` or omit lossy contract
  metadata / CUDA archive eval gating

## Verification

- `.venv/bin/python -m py_compile src/tac/block_fp_codec.py src/tac/preflight.py src/tac/tests/test_block_fp_codec.py src/tac/tests/test_segmap_lossy_pack_contract.py`
- `bash -n scripts/remote_lane_hm_s_segmap_homography.sh`
- `bash -n scripts/remote_lane_sa_segmap_clone.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_block_fp_codec.py src/tac/tests/test_segmap_lossy_pack_contract.py -q`
- `.venv/bin/python - <<'PY' ... check_segmap_hm_sa_lossy_pack_contract(strict=True) ... PY`
- `git diff --check -- <touched files>`

Repository-wide `git diff --check` still fails on unrelated trailing
whitespace in `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`.
