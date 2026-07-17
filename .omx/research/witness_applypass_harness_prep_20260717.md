# #406 PREP — apply-pass BATCH harness built (registry-driven per-lever ΔS), DRY-RUN validated

Date: 2026-07-17 · Agent: p0_406_applypass_prep · $0, CPU 1-thread, read-only on checkpoints
Axis: **[macOS-CPU advisory] NON-PROMOTABLE** — pointer UNMOVED (compress-half MEANS). No scorer
n600 fired (live 74–78 GiB P0 trainer owns the machine; ~14.9 GiB truly-free).

## What landed

- `tools/witness_applypass_batch.py` — ONE registry-driven orchestrator. Given a FROZEN witness
  npz it runs the REGISTERED SEQUENCE of levers; each emits real byte-close blob bytes and (in
  fire mode) n600 chunked frozen-CPU-scorer d_seg/d_pose as `tac.verdicts.MeasurementRow`, folded
  into a per-lever ΔS table (advisory, NON-PROMOTABLE).
- `tools/tests/test_witness_applypass_batch.py` — 15 no-scorer tests (registry integrity,
  transform math, ΔS arithmetic, key-presence, dep/skip, memory-guard refusal, dry-run e2e).

## Why a SECOND tool, not a duplicate (anti-duplication honored)

`tools/witness_apply_pass.py` (landed 2026-07-14) is the canonical home of the five pre-built
levers (#336, #140, sidecar-fold, #311, #401). It PREDATES the #519 gauge/palette result
(`.omx/research/null_subspace_rate_measure_20260717.md`) and has no compose-best mode. The batch
tool ADDS exactly those two on a thin lever REGISTRY, and **reuses, never reimplements**:
- #519 param-space levers measured HERE via `null_subspace_rate_measure`'s `project_gauge` /
  `project_palette_gauge` / `build_blob_for` / `decode_frames` / `score_frames` (the #519 path).
- the five pre-built levers are registered as fail-closed DELEGATES to `witness_apply_pass.py`.
- `--compose-best` chains the Δd_seg-favorable atomic transforms and RE-measures (composition ≠
  sum — measured through the real decode, never summed).

## STORES CONSULTED (proactive recall)

- `tools/graph_memory_recall.py`: "sensitivity bit allocation apply" → **FEED-applypass-406** (the
  one-command APPLY-PASS harness DRY-RUN, 2026-07-11) + `tools/apply_sensitivity_bitalloc_witness.py`
  + eq `heterogeneous_per_tensor_bit_allocation_compounding_v1`; "TropNNC" → **FEED-tropnnc311** +
  eq `tropnnc_dense_trunk_exact_dseg_reduction_empty_v1` (dense τ=1 trunk = 0 exact-Δd_seg reduction;
  FORMULATION-scoped) + `tac.boundary_math.tropnnc_witness_reduction`; "low-rank pose codec" →
  **FEED-sp** (SPD-cone pose-section codec) + task #140 (rank-2 SVD 2.7× smaller at MSE≤d_pose);
  "blind coordinate" → **FEED-blindcoord-401** (230,904 px/frame blind, PROVEN 0 Δd_seg/Δd_pose
  n600, 20.55% byte delta on camera-res payload; DIRECT saving 0 on a pure-generator archive) +
  `tac.through_r.blind_coordinate` + eq `blind_coordinate_rate_lever_v1`.
- `.omx/research/null_subspace_rate_measure_20260717.md` (#519) + `tools/null_subspace_rate_measure.py`
  — the gauge/palette canonicalization transforms + the byte-close+decode+score path REUSED verbatim.
  Gauge is a PRECISION lever, not a rate lever (dense int8+brotli prices ELEMENTS not norm; ±11 B).
- `tools/levelset_byte_close_and_eval.py` — `_load_levelset_ckpt` / `build_levelset_blob` /
  `detect_self_orient` / `_dequant_blob` / `numpy_oracle_reference_frames` (the byte-close orbit).
- `tools/witness_apply_pass.py` — the canonical delegate home (its `_bytes_row`/`_scalar_row`
  honesty-envelope discipline is mirrored).
- MEMORY: `admission_gate_naive_counts_reclaimable_as_committed` (drove the memory-guard fix below);
  #205 OOM lesson (verdict-batch spike) → chunked scorer + memory REFUSE guard.

## Per-lever wiring status (DRY-RUN measured on donor mod32cap ep650)

Donor: `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`
Baseline 0.bin = **84,126 B**. Dry-run peak RSS **~72 MB, 1.8 s** (no torch/frames/gt-cache).

| lever | kind | dry-run status | Δbytes | notes |
|---|---|---|---|---|
| gauge_519 | param_transform | **dryrun (WIRED)** | **+11** | reproduces #519 leg-3 exactly; PRECISION lever |
| palette_canon_519 | param_transform | **dryrun (WIRED)** | **−6** | reproduces #519 leg-3 exactly; argmax-preserving |
| both_canon_519 | param_transform | **dryrun (WIRED)** | **+11** | gauge ∘ palette composed then byte-closed |
| bit_alloc_336 | delegate | staged (WIRED) | — | → `apply_sensitivity_bitalloc_witness.py` via witness_apply_pass |
| low_rank_pose_140 | delegate | owed (WIRED) | — | needs `--pose-target (600,6)`; delegates to witness_apply_pass:_low_rank_pose |
| tropnnc_311 | delegate | staged (WIRED) | — | → `tac.boundary_math.tropnnc_witness_reduction` (dep import OK) |
| blind_coord_401 | delegate | staged (WIRED) | — | receiver-side; dep import OK; DIRECT saving 0 on pure-generator |
| compose_best | mode | **dryrun (WIRED)** | +11 | composes atomic gauge+palette (dry-run preview) |

All deps import cleanly → **no lever SKIPPED**. A SKIPPED row is emitted only on a genuine
missing dep / missing param key (fail-closed, never a fake pass) — covered by tests.

## Memory guard — bug found + fixed in-prep (the admission-gate confound)

First fire-mode test did NOT refuse: `psutil.available` reported **58.7 GiB** (counts reclaimable/
inactive as available) and cleared the 24 GiB floor, so the heavy path started before the 2-min
SIGTERM killed it (trainer UNHARMED — pid 13783 verified alive throughout). Fixed `_free_gib()` to
return the **CONSERVATIVE MIN(psutil.available, vm_stat free+speculative)** = 3.8–14.9 GiB truly-free,
so fire mode now REFUSES (rc=4) immediately next to the live trainer. Regression test asserts the
refusal fires BEFORE any `load_scorers` (a tripwire fails the test if scorers load). This is the
`admission_gate_naive_counts_reclaimable` lesson applied to a safety REFUSE guard.

## The EXACT post-run invocation (fire the full per-lever ΔS table when the machine is FREE)

```
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python -u tools/witness_applypass_batch.py \
  --ckpt-dir <v9_c2_run_dir> \
  --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pairs 600 --compose-best \
  --pose-target <path/to/(600,6)_posenet_target.pt> \
  --out-dir experiments/results/applypass_batch_$(date -u +%Y%m%dT%H%M%SZ)
```
(omit `--pose-target` and low_rank_pose_140 stays OWED; delegate levers additionally need a
`witness_apply_pass.py --fire-scorer-stages` pass — the staged argv is emitted per lever in the
summary json. Lower `--min-free-gib` only deliberately, never next to the live trainer.)

## Residuals (honest)

- Fire mode NOT run (memory; correct). The param-transform ΔS numbers exist only as +11/−6/+11
  BYTE deltas until an n600 scorer pass fires; #519 leg-4 (n32 advisory) already showed Δd_seg
  −4.8e-6..−1.3e-5 (favorable, INSTANCE-scope, NOT n600) — do not cite as established.
- Delegate FIRE currently emits the staged `witness_apply_pass.py --fire-scorer-stages` argv and
  (in fire mode) marks `delegated`; harvesting its `apply_pass_rows.jsonl` back into the batch
  table is a follow-up (the numbers are produced by the canonical home, not lost).
- compose-best winner-selection (Δd_seg≤0) is fire-only; dry-run composes ALL atomic as a preview.
- `# FORMALIZATION_PENDING`: no canonical equation registered — this is apparatus (a harness), not
  a measured finding; register only if/when an n600 fire produces a load-bearing per-lever ΔS.
```
```
