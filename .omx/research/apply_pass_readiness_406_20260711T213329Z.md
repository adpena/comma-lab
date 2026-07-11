# #406 APPLY-PASS READINESS — one-command rate/pose per-lever ΔS harness (DRY-RUN LANDED)

**Date:** 2026-07-11 · **git:** 9753cda712 · **task:** #406 (operator-approved 2026-07-11)
**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE · **pointer 0.19108282 UNMOVED (MEANS)**
**Subagent:** applypass-406

## What this is

`tools/witness_apply_pass.py` — ONE harness that turns a FROZEN witness checkpoint into a
per-lever rate/pose ΔS-attribution table. It fires the moment the live v9 run
(`v9_cgauge_432_coherent_arm_20260711`, pid 44251, owns the GPU) produces its first good
checkpoint. It is a pure ORCHESTRATOR: every stage delegates to the canonical existing tool
(NO re-derived codec math), every number is a REAL measurement of a REAL byte-closed blob, and
every emitted scalar is a canonical `tac.verdicts.MeasurementRow` (axis tag + provenance +
noise-floor-honesty + n_samples-reason + review_status). Reused, not rebuilt:
`levelset_byte_close_and_eval.py` (baseline + fold), `apply_sensitivity_bitalloc_witness.py`
(#336), `tac.verdicts.measurement_row` (#388 canonical row), `gt_pose_targets.pt` (real 600×6
PoseNet target for #140).

## Stages (ordered rate→pose)

| # | lever | tool | dry-run status |
|---|-------|------|----------------|
| 1 | baseline_byte_close | levelset_byte_close_and_eval | **MEASURED** |
| 2 | bit_alloc (#336) sensitivity KKT reverse-waterfill | apply_sensitivity_bitalloc_witness | **STAGED** (scorer-heavy) |
| 3 | low_rank_pose (#140) rank-2 SVD codec | in-harness numpy SVD + brotli | **MEASURED** |
| 4 | sidecar_fold | levelset_byte_close --fold-pose-sidecar | **STAGED** (fail-closed NO-FAKE) |
| 5 | STUB #311 TropNNC | NOT BUILT | **OWED** (loud NotImplementedError if `--run-stub`) |
| 6 | STUB #401 blind-coordinate | NOT BUILT | **OWED** (loud NotImplementedError if `--run-stub`) |

## Dry-run numbers (frozen ckpt: `levelset_v752_baseline_20260710T185913Z/levelset_witness_ema_BEST.npz`, d_seg=0.0293 @ ep150; sha256 recorded in summary; copied into out-dir, never in-place)

Out-dir: `experiments/results/apply_pass_dryrun_20260711T213329Z/`

- **baseline** — `archive_zip_bytes = 83905 B` `[macOS-CPU advisory]` (byte term exact; max-pairs 6, byte term is pair-count-independent). Rate term at 25·83905/37545489 ≈ 0.0559.
- **low_rank_pose (#140)** — scalar-store baseline (fp16+zlib) `6763 B`; rank-2 q10 codec `1125 B` → **6.01× byte win**; reconstruction-floor `d_pose = 0.000243` `[prediction]` → √(10·d_pose) contribution **0.0493** (BELOW the banked R1-dxi pose contribution 0.127, so rank-2 target storage is not the pose bottleneck). Measured from the DEQUANTIZED factors (truncation + quantization = the real floor); finiteness-guarded.
- **bit_alloc (#336)** — STAGED: emits the exact `apply_sensitivity_bitalloc_witness` argv (probe/eval + mean-bits). Calibration measures d_seg through R → loads SegNet + gt cache, so it is default-off under containment; fire with `--fire-scorer-stages` when the machine is free.
- **sidecar_fold** — STAGED: `--fold-pose-sidecar` is fail-closed (NO-FAKE — refuses to fabricate a `posenet_targets.bin`). Emits the exact fold argv; needs the built pose sidecar bin (`tac.scorer_targets.extract_and_save`). Also flags that a code-pose witness pays DEAD bytes when folded (loud honesty).
- **stubs** — `--run-stub tropnnc_311` / `blind_coord_401` raise `NotImplementedError` (verified). No fake apply.

Emitted artifacts: `apply_pass_summary.json` (full per-stage + staged argv), `apply_pass_rows.jsonl`
(one MeasurementRow per measured scalar; all `is_authority_axis=False`, `is_load_bearing=False`),
`baseline_byte_close.json`. Errors: 0.

## The exact one-command that fires against v9's first good checkpoint

```
.venv/bin/python tools/witness_apply_pass.py \
    --ckpt-dir experiments/results/v9_cgauge_432_coherent_arm_20260711 \
    --npz-name levelset_witness_ema_BEST.npz \
    --pose-target experiments/results/lane_a_landed/gt_pose_targets.pt \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --fire-scorer-stages --probe-pairs 16 --eval-pairs 96 --max-pairs 600
```
(scorer stages fire only when the machine is free; still `[macOS-CPU advisory]`. A PROMOTABLE ΔS
row needs the full n600 byte-close + `upstream/evaluate.py` on contest-CPU Linux x86_64 —
NO paid dispatch without explicit operator GO. Every stage already emits its n600 staged argv.)

## What is MEASURED vs STAGED vs OWED (honesty ledger)

- MEASURED today (real byte-closed blobs, advisory): baseline archive bytes; #140 rank-2 pose
  codec bytes + reconstruction-floor d_pose.
- STAGED (wired, argv emitted, not fired under containment): #336 bit-alloc (needs scorer +
  gt cache); sidecar-fold (needs built posenet_targets.bin); every lever's n600 through-R +
  exact-eval command.
- OWED (NOT built): #311 TropNNC, #401 blind-coordinate — explicit fail-loud stubs.

## Containment honored

CPU-light + serial; each stage a subprocess that exits; scorer stages default-STAGED; NO Modal/paid
dispatch; frozen ckpt copied (never in-place); did NOT touch the two live trainer files nor
`src/tac/witness_run_artifacts.py` (sister codex in-flight).
