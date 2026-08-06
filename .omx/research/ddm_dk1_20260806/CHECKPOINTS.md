# ddm_dk1 - Checkpoints

## Landed Artifacts

| Artifact | Purpose |
|---|---|
| `src/tac/optimization/lattice_native_pose_null_realizer.py` | Callable DK1 lattice-native 2x2 block realizer using exact #580 D weights. |
| `src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py` | Focused tests for geometry, kernel, solvers, and diagnostics. |
| `tools/measure_ddm_dk1_lattice_realizer.py` | Reproducible small-n measurement tool for real phase-field blocks. |
| `.omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json` | Measurement receipt with input hashes, selected blocks, method diagnostics, and aggregate ladder. |
| `.omx/research/ddm_dk1_20260806/RECEIPT.md` | Human receipt and seam ledger. |
| `.omx/research/ddm_dk1_20260806/NEXT_IF_RESUMED.md` | Resume/follow-on fire orders. |

## Verification Commands

Run from `/Users/adpena/Projects/pact`:

```bash
.venv/bin/python -m pytest src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py -q
```

Expected result: `7 passed`.

```bash
.venv/bin/python tools/measure_ddm_dk1_lattice_realizer.py \
  --n-blocks 4 \
  --threads 6 \
  --out .omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json
```

Expected result: JSON receipt with aggregate ladder:

- naive mean pose leakage sq `0.07217925000000001`
- Dykstra mean pose leakage sq `0.03997194204817428`
- CVP mean pose leakage sq `0.007107145150751825`

## Data Custody

No bulk artifact was created by DK1. The measurement reads SSD-custodied inputs:

- `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw`
  SHA-256 `82de098f5b97e6c61c7a53b4180f425117ea2e3c89e6ab435e7aea423f81291a`
- `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy`
  SHA-256 `71365c74d49c2c0f611b4a3e01cbfe735177398c0510751bdf0f4642fad5af0d`

No transient-path persisted evidence. No files under `upstream/` were touched.

## Completion Boundary

Complete in this checkpoint:

- Exact-D private support extraction with nonuniform weights.
- Naive vs Dykstra vs CVP/Babai kept-set race.
- Real small-n PoseNet leakage measurement against tq1c parent pairs.
- Float-first seam sweep and owner ledger.

Not complete:

- No SegNet population verdict.
- No archive build.
- No exact CPU/CUDA score.
- No n600 scorer use.
- No wiring into Q3X/SQ1/Q31/LR2 successor scripts.
