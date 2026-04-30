# Lane 12 / Alpha-Geo Dispatch Readiness Guardrail - 2026-04-30

Scope: Lane 12 NeRV dispatch/readiness and Alpha-Geo-0/1 mask comparison only.
No score claim and no paid compute launched.

## Evidence Boundary

This is code/test hardening and dispatch-readiness analysis. It cannot promote,
rank, kill, or retire a method family. Exact score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Inspection Result

Current canonical local artifacts exist:

```text
Lane G v3 base archive:
  path = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
  bytes = 694074
  sha256 = 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b
  members = renderer.bin, masks.mkv, optimized_poses.pt

Lane 12 jsonfix40 negative archive:
  path = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
  bytes = 296478
  sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
  exact CUDA score = 26.03719330455429
```

Dispatch was not exact-ready from canonical artifacts. The remote script still
allowed a default rerun of the retired fresh-SegNet target path and defaulted
the base archive to `submissions/robust_current/archive.zip`, not the Lane G v3
archive used by the Alpha-Geo comparisons.

## Patch

Changed `scripts/remote_lane_nerv.sh` to fail closed around the known unsafe
paths:

- Defaults `BASE_ARCHIVE` to
  `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`.
- Defaults `GT_MASKS_SOURCE=decoded-baseline`.
- Defaults `RUN_AUTH_EVAL=0`, so an accidental launch stops after deterministic
  archive build and cannot create score evidence.
- Blocks `GT_MASKS_SOURCE=segnet` unless `ALLOW_RETIRED_SEGNET_TARGET=1` is set
  for a documented forensic rerun.
- Requires `POSE_REGEN_PROVENANCE` and `ALPHA_GEO_PROVENANCE` for
  `RUN_AUTH_EVAL=1`; the stale-pose bypass has been removed.
- Requires a valid `.omx/state/lane12_nerv_l2_clearance.json` before any new
  NeRV retraining starts.

Changed `src/tac/tests/test_lane12_nerv_dependency_closure.py` to assert the
new defaults and that the exact CUDA eval path remains present but gated.

## Readiness Verdict

Training/build-only Alpha-Geo-1 candidate generation is scriptable from the
canonical Lane G v3 artifact, but exact dispatch is still blocked by missing
pose-regeneration provenance and candidate-vs-baseline geometry gate results.

Do not queue paid exact auth eval for Lane 12 until a candidate archive records:

1. decoded-baseline target custody,
2. Alpha-Geo geometry diagnostics in band or a reviewed exception,
3. regenerated pose provenance against the candidate mask stream, and
4. deterministic archive manifest and payload closure.

## Next Commands

Local focused verification:

```bash
.venv/bin/python -m py_compile src/tac/tests/test_lane12_nerv_dependency_closure.py
bash -n scripts/remote_lane_nerv.sh
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py -q
git diff --check
```

Exact dispatch command: none is ready yet.

Training/build-only remote command template after explicit approval for paid
candidate generation:

```bash
.venv/bin/python scripts/launch_lane_with_retry.py \
  --lane-script scripts/remote_lane_nerv.sh \
  --label lane_12_nerv_alpha_geo1_decoded_baseline_20260430 \
  --max-dph 0.30 \
  --predicted-band 0.95 1.30 \
  --estimated-cost 1.00 \
  --max-retries 3
```

Exact eval remains a separate reviewed step and must set `RUN_AUTH_EVAL=1` only
after pose-regeneration provenance exists.
