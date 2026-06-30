# RESIDUAL-ONLY TRAINING MODE + INFLATE RESIDUAL-COMPOSE LANDED (v2 gap #1 + gap #2) — 2026-06-30

**Authority / status (NO-FAKE supreme rule + means != ends).** This is the BUILD that ENABLES the
binding hybrid residual-INR run — it is a MEANS, not the end. The frontier pointer is **UNMOVED at
contest-CPU 0.19110**. Nothing here is a score, frontier, promotion, or kill. The pointer moves ONLY
when the residual-INR GPU run lands + the 4-section archive byte-closes + `upstream/evaluate.py`
(CPU + CUDA, NEVER MPS) returns below 0.19110. Every number below is `[advisory] NON-PROMOTABLE`.
CPU-only build; NO GPU; NO training launched; the live n600 baseline (pid 38641, daemon
`levelset_n600_v2_attrclean_20260630T194549Z`) was UNTOUCHED (verified epoch 51, 1h35m, healthy).

Closes the two OPEN seams the PHASE-A composition landing
(`v2_compose_composition_layer_landed_20260630.md`) flagged: **S3 (gap #1)** — the trainer had no
residual-target-subtraction mode (the emitted command used `--structured-init` = init-into-weights =
NO rate shrink); and **gap #2** — the inflate residual-compose hook was a `raise SystemExit` stub.

## The rate-bearing mechanism (why this shrinks the archive)

`--structured-init` (the prior proxy) bakes the bulk SDFs INTO the INR weights as a train-time init:
the trained INR still encodes the WHOLE partition -> the ~90 KB full-partition INR ships in
`archive.zip` -> **no rate shrink**. The residual-only mode moves the bulk OUTSIDE the counted
weights:

    composed_rgb = where(composition_mask, INR_residual_rgb, deterministic_bulk_rgb)

- the **deterministic bulk** is GENERATED at decode (stored keyframes + per-class stratified warp +
  palette/R1 ramp) — rule-118 FREE, 0 counted bytes (only keyframes/pose/calib are stored);
- the **composition mask** is DERIVED from the bulk's OWN warped label map
  (`isin(label, {Lane=1, Movable=3})` + optional dilate) — also regenerated FREE at inflate, **0
  shipped bytes** (the mask is NOT GT-derived, so no GT leak and no large mask payload — the design
  flaw of shipping a GT-residual mask is avoided);
- the **INR** trains on (and ships) ONLY the Lane+Movable residual annulus -> sized small
  (`--hidden-dim 48 --mod-dim 16` vs the full-partition 96/32) -> the **rate win**.

The composition is ONE math (`tac.v2_compose.residual_compose.compose_residual_rgb` +
`derive_composition_mask`) shared bit-identically by the trainer (compress), the realized verdict,
and the self-contained `inflate.py` decoder — so train == inflate (NO-FAKE).

## What changed in the trainer (+ proof the baseline path is BYTE-UNCHANGED)

1. **Sibling render seam** (`experiments/train_witness_realized_through_R_mlx.py`, surgical +
   default-preserving): `render_through_R_mlx(..., compose_fn=None)` composes the bulk before R when
   given; `make_loss_fn(..., render_fn=None)` uses `render_fn` for every realized render when given.
   **Both default to the exact pre-residual behavior** (compose_fn None => no compose; render_fn None
   => the bare `render_through_R_mlx`).
2. **Trainer residual mode** (`experiments/train_levelset_witness_realized_through_R_mlx.py`): new
   `--residual-mode` + `--residual-target-npz` flags (default OFF). In `run_train`, when ON: load
   the residual bundle, build `_compose_mx` (MLX, for the loss + levers via `_render_R`) and
   `_compose_np` (numpy, for the verdict + LEVER-5 hardness). The bulk/mask arrays live in CLOSURE
   scope, **NOT model attributes** -> they are NEVER in `model.parameters()` => the EMA / optimizer /
   quantized blob / per-stage checkpoints see ONLY the INR (**the bulk does NOT ship** — that IS the
   rate win). Resumability + per-stage checkpoints + EMA-shadow preserved (the bulk is reloaded from
   the bundle on resume, never checkpointed).
3. **Surgical-lever coherence (coordinator requirement, addressed):** the surgical levers
   (`--lane-thin-*` / `--margin-saliency-*` / `--lane-edge-*` / LEVER-5 `--hardness-*`) route their
   realized seg forward through `_render_R` (the lever line + the hardness baseline) -> they weight
   the **COMPOSED-render** d_seg. Since the residual IS the Lane+Movable annulus, those levers are
   maximally relevant. A dedicated test proves a lane-thin lever measurably reweights the realized
   seg loss on the composed render (not a silent no-op), so the theta* sweep runs on the hybrid.
4. **Fail-closed guards:** `--residual-mode` requires `--residual-target-npz`; incompatible with
   `--structured-init` / `--lane-prior-phi1` / `--freeze-decoder-fit-codes` (the contradictory
   bake-into-weights mechanisms); `--residual-target-npz` without `--residual-mode` fails.

**Baseline byte-unchanged proof:** with `--residual-mode` OFF (default), `residual_mode=False` =>
`_render_R IS render_through_R_mlx` (same object), `_compose_np is None`, `render_fn=None` passed to
`make_loss_fn`, the verdict compose branches are skipped. Tested:
`render_through_R_mlx(compose_fn=None)` is BYTE-IDENTICAL to the bare render, and
`make_loss_fn(render_fn=None)` produces the SAME loss as `render_fn=render_through_R_mlx`. The live
n600 run was unaffected (separate process; already imported its code; the new path is flag-gated).

## Inflate residual-compose (gap #2) — PARITY: **PASS (bit-exact)**

`src/tac/v2_compose/archive_grammar.py`: added `build_residual_blob` / `parse_residual_blob` (the
COUNTED LEARN-tier INR: int8+brotli weights + per-frame code + cfg; the curvelet bank B is regen'd
FREE, rule-118) and `residual_inflate_reference` (the numpy ORACLE from the BUILT tac primitives).
The self-contained MLX-free `inflate.py` template now decodes the residual blob, regenerates the
curvelet feats, forwards the numpy level-set INR, derives the bulk-label composition mask, composes
`where(mask, INR, bulk)`, and bicubic-ups to camera uint8. The inflate also now adapts the warp
intrinsics/grid to the stored resolution (identical to 384x512 in production; enables any-res test).

**PARITY (the NO-FAKE faithfulness anchor):** the inflate.py run AS A SUBPROCESS (exactly as the
contest) is **bit-identical** to the numpy oracle (`max abs diff = 0`) — WITH a real per-class warp
+ a non-empty residual INR — and the empty-residual archive still produces the deterministic floor
(frame0 == frame1). The 4-section archive (store + residual + pose + manifest) byte-closes.

## The corrected residual-INR launch command (HOLD — emitted, NOT fired; flag-validated)

`build_residual_only_command` (supersedes `build_residual_inr_command`'s `--structured-init` path);
**all_flags_valid=True** against the real argparse (no invented flags):

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_residual_n600_<utc> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1500 --seed 0 --mlx-device gpu \
  --hidden-dim 48 --mod-dim 16 --ema-decay 0.997 \
  --residual-target-npz experiments/results/<run>/residual_target.npz \
  --stage-checkpoints --residual-mode --curriculum
```

This is the CLEAN BASE (best-known priors, optimal-form-first: curriculum + per-stage ckpts + small
INR; perf env per the launch-gate memo — verify `custom_grouped_backward active=true` at launch).
**The theta* lever sweep runs ON this hybrid residual** (re-pointed off the old full-partition
warm-start): each theta* arm = this command + ONE surgical lever varied (e.g.
`--lane-thin-weight 2.0 --lane-thin-start-epoch 300`, validated all_flags_valid=True). HOLD for
operator GO + GPU reallocation; this build did NOT fire it.

The `residual_target.npz` input is produced by `tac.v2_compose.bulk_generator.generate_bulk_render_and_labels`
(light, no-SegNet warp+render) -> `tac.v2_compose.residual_compose.build_residual_training_bundle`
-> `save_residual_training_bundle` (a FREE training artifact carrying the deterministic bulk RGB +
the bulk-derived composition mask; NOT an archive section).

## Byte accounting (the rate budget) + the explicit OPEN quantity

| Section | Bytes | Status |
|---|---|---|
| Partition keyframes (13, contour-coded) + warp mask + calib | ~9.0 KB store | MEASURED (FEED-ll reach k*=47) |
| Pose sidecar (6 scalars/pair, fp16+zlib) | ~0.9 KB | MEASURED |
| **Residual INR weights (Lane+Movable ONLY)** | **OPEN** | **the single unmeasured quantity** |
| FREE-generated (warp / SDF render / R1 ramp / curvelet B / composition mask) | 0 | rule-118 |

**THE OPEN QUANTITY (honest):** can the SMALL residual INR (hidden 48 / mod 16) close d_seg from the
deterministic bulk floor (~0.0185) to the sub-0.15 budget (~6e-4 ... 1.4e-3) at a SMALL byte cost?
The rate axis is GREEN (the bulk is off the counted ledger); the binding wall is the residual INR's
d_seg efficacy + the Lane R-survival physics (GAP2). This is settled ONLY by the (HELD) GPU residual
run + the byte-closed exact eval. The composition is engineered so that one GPU run is the decisive
measurement (the (d_seg, bytes) point that does or does not beat 0.19110 / reach sub-0.15);
everything else is $0/CPU and built + tested now.

## Tests (17, all green, CPU-only)

`src/tac/tests/test_v2_residual_compose.py`: residual_compose mask/compose/dilate/bundle-roundtrip +
/tmp-refusal (6); residual blob build/parse roundtrip + byte accounting (2); **inflate bit-exact
parity (subprocess vs oracle, with warp) + deterministic-floor (2)**; render-seam compose_fn=None
byte-identical + compose-correctness + make_loss_fn render_fn default-identical (3); the surgical
lever reweights the COMPOSED-render seg loss (1); 4 fail-closed config guards (4). All MLX ops run on
the **CPU device** (`temporary_mlx_device("cpu")`) so they never contend with the live GPU baseline.

## 6-hook wire-in (Catalog #125)

1. sensitivity-map: ACTIVE — `bulk_dseg` floor (residual_target) is the d_seg target the INR closes.
2. Pareto: ACTIVE — `byte_accounting` (store + pose + residual) is the rate constraint; residual = OPEN.
3. bit-allocator: ACTIVE — `build_residual_blob` int8+brotli per-tensor (the counted LEARN bytes).
4. cathedral autopilot: N/A — composition actuator; the residual run feeds the ranker once measured.
5. continual-learning posterior: N/A — plan/build-time plumbing, no posterior mutation.
6. probe-disambiguator: N/A — the composition rule is KNOWN deep-math, not a competing-interpretation.

`council_predicted_mission_contribution: frontier_breaking` (unblocks the binding sub-0.15 GPU run).

Cross-refs: spec `v2_coherent_automated_composition_pipeline_spec_20260630T201213Z` ·
`v2_compose_composition_layer_landed_20260630` · DAG FEED-iz/ja/lj/lk/ll ·
`[[gr-unified-action-full-witness-architecture-20260629]]` ·
`[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]` · CLAUDE.md rule-118 + "Pose is SOLVED".
