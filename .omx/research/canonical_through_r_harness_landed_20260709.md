# Canonical through-R harness + scaffold assembler — LANDED (CANONICALIZATION UNIT 1, #388)

**Date:** 2026-07-09 · **Operator GO:** "Go on building all" (task #388) · $0, no GPU, run dirs
read-only. **Pointer contest-CPU 0.19110 UNMOVED** — this is MEANS (measurement apparatus), not a
lever, not an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.

## What landed

`src/tac/through_r/` — the canonical home for the two most-rebuilt patterns in the campaign:

| module | role |
|---|---|
| `resolution_chain.py` | THE authoritative pinned R chain (camera 874×1164, SegNet 384×512, seq_len 2), source-verified against `upstream/{evaluate,modules,frame_utils}.py` + `tac.contest_eval_contract`; R first-half operator `render_grid_to_camera_uint8` (bicubic↑ + uint8); `describe()` provenance; WH-vs-HW transposition guard; `verify_against_upstream()` fail-closed. |
| `harness.py` | `measure_through_r(...)`: candidate frames → R → frozen CPU-torch SegNet argmax → per-class + aggregate d_seg vs cached `lstars`. n600 toy-refusal (`allow_subset_reason`); `backend='cpu-torch'` the ONLY authority (no proxy branch, P9); chunked SegNet forward (n600-verdict-OOM law). |
| `scaffold_assembler.py` | canonical composite-argmax assembler (promoted from `inc1a_harness/composite_assembler.py`): Laguerre/tropical argmax composition, pluggable `b_c` (#386 dispatcher), bounded d_seg-monotone reconciliation. |

**Migration:** `tac.inc1a_harness.composite_assembler` is now a thin delegation shim re-exporting the
canonical `scaffold_assembler`. `Inc1aAssemblerError` is preserved as an alias of the canonical
`ScaffoldAssemblerError` (same class object) so every `except`/`pytest.raises` call site is unaffected.
The inc1a 27-test suite stays green; `analytic_smoke.py` + `mask_dseg_meter.py` + `decoupling_screen.py`
+ `__init__.py` unchanged.

## Why (the repetition receipts + the bug class it extincts)

Subagents kept re-deriving `load gt_n600 → compose/render candidate fields → R → frozen CPU-torch
SegNet argmax → per-class d_seg vs lstars`: `src/tac/inc1a_harness/*`, `experiments/probe_flip_bc_n600_gate.py`,
`experiments/probe_laguerre_logit_offset_sweep.py`, `src/tac/boundary_math/movable_deshare.py`, OT
verdict scripts. Each re-derivation re-risked the **flip-resolution bug class** (operator binding).
The fix is a single pinned chain with an explicit transposition guard: upstream constant NAMES are
`(W,H)` tuples (`camera_size=(1164,874)`, `segnet_model_input_size=(512,384)`); numpy `.shape` and
`torch.interpolate(size=...)` are `(H,W)`. The chain owns exactly ONE resize (bicubic↑ to camera); the
R second half (bilinear↓ to 384×512) is **delegated to the real `SegNet.preprocess_input`**, removing
the second place a transpose could enter. #149 camera-res PLACEMENT is documented as an intended
exception (legal field placement ≠ the compare grid, which is always SEG_HW post-argmax).

## Verification (apparatus validation, MEASURED)

- `ruff check` clean. 57 tests green (30 new + 27 inc1a regression), **0 skips** — the gt cache +
  frozen SegNet weights are present, so the scorer-gated tests genuinely ran through R.
- **Strongest correctness test (ran through the real SegNet):** feeding cached `gt_f1` back through
  `measure_through_r` reproduces `lstars` EXACTLY → agg d_seg == 0.0 by construction (they were
  computed by this same SegNet on `gt_f1`); `total_flips == 0`.
- **Chunk bit-identity:** `verdict_batch=1` vs `verdict_batch=0` (single batch) → BIT-IDENTICAL
  realized argmax + agg d_seg (BatchNorm running stats in `.eval()`; argmax per-pixel).
- Fail-closed drift test: monkeypatched upstream constants → `verify_against_upstream()` RAISES.

## Consumer list for the #389 verdicts sweep

The parallel #389 build owns `src/tac/verdicts/` (`MeasurementRow`). Per the parallel-build fence, this
unit does **NOT import** `tac.verdicts`; instead a `# TODO(#389): emit MeasurementRow` marker sits at
the per-pair row site in `harness.py::measure_through_r`. When #389 wires the sweep, the emit point is:

- `tac.through_r.harness.measure_through_r` → the `per_pair` loop (one row per pair: pair_idx,
  realized-through-R d_seg, per-class flip attribution available from `per_class_flip_stats`, backend,
  input_space, `subset_reason`, `THROUGH_R_LABEL`).
- `ThroughRResult` already carries the aggregate + per-class fields a `MeasurementRow` summary needs.

Downstream consumers that should MIGRATE onto this harness (follow-up, not this unit): the flip-b_c
gate + Laguerre sweep probes (`experiments/probe_*`), `movable_deshare` measurement path, any future
per-class through-R verdict.

## 6-hook wire-in declaration (Catalog #125)

1. **sensitivity-map** — N/A (apparatus; produces measurements, not per-axis byte savings).
2. **Pareto constraint** — N/A (no new archive-grammar constraint).
3. **bit-allocator hook** — N/A (no per-tensor importance change).
4. **cathedral autopilot dispatch** — N/A (not archive-deployable; a local measurement harness).
5. **continual-learning posterior** — DEFERRED to #389 (the `MeasurementRow` emit is the posterior
   wire-in; marked with `# TODO(#389)` to avoid importing the parallel sibling).
6. **probe-disambiguator** — N/A (single authority path, no 2+ defensible interpretations).

## Triality legs

- **DAG:** `FEED-canon-u1` appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** N/A-with-reason — apparatus (measurement harness + assembler), no witness lever / trainer
  flag / curriculum surface for `witness_dsl` to hold.
- **equations:** N/A-with-reason — reuses the ALREADY-registered `d_seg` authority functional + score
  law; no NEW measured relation. The producer/consumer apparatus contract is the invariant.

STORES CONSULTED: the DAG (FEED-08 flip-resolution grid audit), `experiments/probe_flip_bc_n600_gate.py`,
`train_witness_realized_through_R_mlx` (R + `cpu_verdict_*`), `tac.boundary_math.seg_core`,
`tac.contest_eval_contract`, `tac.local_acceleration.metal_fused_r_operator`,
`inc1a_harness/composite_assembler.py`.
