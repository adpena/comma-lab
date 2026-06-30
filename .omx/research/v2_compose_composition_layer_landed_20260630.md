# v2_compose composition layer LANDED (PHASE-A plumbing) — 2026-06-30

**Authority / status (NO-FAKE + means≠ends):** this landing is composition PLUMBING (a MEANS).
The frontier pointer is **UNMOVED at contest-CPU 0.19110**. Nothing here is a score, frontier,
promotion, or kill. The pointer moves ONLY when the residual-INR GPU run lands + the 4-section
archive byte-closes + `upstream/evaluate.py` (CPU + CUDA) returns below 0.19110. Every emitted
number is `[advisory] NON-PROMOTABLE`. CPU-only; no GPU; no training launched; live n600 untouched.

Implements the spec `.omx/research/v2_coherent_automated_composition_pipeline_spec_20260630T201213Z.md`
(the ~30% missing CPU composition layer). NEW package `src/tac/v2_compose/` + entry point
`tools/compose_witness_archive.py`.

## What landed (all REAL, tested CPU)

| Module | Seam | What it does |
|---|---|---|
| `store_learn_split.py` | S1 (keystone) | `encode_known_split(warp_through_R, reach_kstar)` — ENCODES the KNOWN deep-math split (not discovery), parameterized by the MEASURED per-class warp-through-R recoverability so it generalizes to any clip. On the real grok+reach JSONs reproduces: GENERATE=(Road=ground_homography, Undriv=rotation_only, MyCar=identity) == BULK_IDX[0,2,4]; LEARN=(Lane, Movable); keyframes=13; break-even d_seg=0.00166. Decision uses ONLY the measured d_seg signature, NEVER a hardcoded class index. |
| `bulk_generator.py` | S2 | deterministic STORE+GENERATE bulk: REUSES (faithful-by-construction import) the proven reach-tool warp/render/R primitives. Cache-agnostic NO-FAKE anchor = SegNet selfcheck (gt_f1→argmax == cached lstars, 0 px). |
| `residual_target.py` | S3 (keystone) | `residual = GT_partition − bulk_through_R` — the bulk is SUBTRACTED, so the residual INR carries ONLY the cells the bulk gets wrong (Lane+Movables). This is the rate-bearing difference vs `--structured-init` (which keeps the bulk inside the INR weights). |
| `pose_sidecar.py` | S5 | dual-use stored pose sidecar from the cache `gt_poses` ($0, no PoseNet re-run; reuses `tac.scorer_targets` PNTG). |
| `archive_grammar.py` | S4 | 4-section byte-close (MAGIC + length-prefix, modeled on the byte-close tool) {store, residual, pose, manifest}; keyframes via `contour_codec` (bit-exact); + the self-contained MLX-free inflate.py (the deterministic bulk generator COMPILED into inflate, rule-118 FREE; loads NO scorers). |
| `launch_command.py` | — | emits the FLAG-VALIDATED residual-INR launch command (statically parses the real trainer argparse; raises on any invented flag; perf-env prefix; HOLD for operator GO — never fires). |
| `tools/compose_witness_archive.py` | — | the single entry point: PHASE-A (encode→bulk→residual→pose→emit launch cmd) + PHASE-B (4-section byte-close of the DETERMINISTIC FLOOR → inflate → realized d_seg → staged dual CPU/CUDA eval cmd). |

## Verified (advisory, n6 + synthetic)
- **NO-FAKE inflate parity (bit-exact):** the inflate.py inlined numpy warp/render == the proven
  reach-tool path (test asserts byte-identical camera frames, fp16-quantization matched).
- **Archive→inflate→SegNet chain:** PHASE-B realized d_seg 0.0324 ≈ PHASE-A compress-side bulk 0.0322.
- **SegNet selfcheck = 0 px** (cache-agnostic faithfulness).
- 33 CPU tests green; flag-validator all_valid=True on the real trainer argparse.

## The residual-INR launch command (HOLD for operator GO — emitted, NOT fired)
```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir <run> --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1500 --seed 0 --mlx-device gpu \
  --hidden-dim 48 --mod-dim 16 --ema-decay 0.997 \
  --stage-checkpoints --curriculum --structured-init --lane-prior-phi1
```

## The two OPEN quantities (honest)
1. **S3 GPU-side gap (NEEDS-WIRING):** the trainer has NO residual-target-subtraction flag. The
   emitted command uses `--structured-init`+`--lane-prior-phi1` (init-into-weights), which does NOT
   by itself shrink the rate. The TRUE residual-only mode (consume `residual_target.npz`, subtract
   the bulk, size the INR for the residual) is the rate-bearing trainer change to wire next. The
   `residual_target.npz` this pipeline emits is its input.
2. **The decisive unknown:** can a SMALL residual INR close d_seg from the ~0.0185 deterministic
   floor to ~6e-4 at a small byte cost (Lane R-survival physics, GAP2)? Settled ONLY by the GPU run
   + byte-closed exact eval.

## 6-hook wire-in (Catalog #125)
1. sensitivity-map: ACTIVE — `break_even_d_seg` at the known-store rate is the d_seg target.
2. Pareto: ACTIVE — the byte-budget decomposition (store+pose+residual) is the rate constraint.
3. bit-allocator: ACTIVE — `byte_accounting` + `rate_term` per section.
4. cathedral autopilot: N/A — composition actuator, not a ranker input yet (residual run feeds it).
5. continual-learning posterior: N/A — plan-time arithmetic, no posterior mutation.
6. probe-disambiguator: N/A — the split is KNOWN deep-math, not a competing-interpretation choice.

`council_predicted_mission_contribution: frontier_breaking` (unblocks the binding sub-0.15 GPU run).

Cross-refs: spec `v2_coherent_automated_composition_pipeline_spec_20260630T201213Z` · DAG FEED-iz/ja/lj/lk/ll ·
`[[gr-unified-action-full-witness-architecture-20260629]]` · `[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]`.
