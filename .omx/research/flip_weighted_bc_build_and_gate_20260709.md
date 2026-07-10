# Flip-weighted b_c BUILD + n600 gate (#386, crucible-3 N-1 reformulation) — BOTH arms LOSE; global b_c SATURATED at no_offset

**Date:** 2026-07-09 · **Author:** claude BUILD subagent (#386 "Build all unbuilt", flip-weighted b_c arm)
**Authority:** `[macOS-CPU advisory]` — realized-through-R on the frozen CPU SegNet, fp32 EMA render, ALL 600
pairs. NON-PROMOTABLE until byte-closed exact eval. **Pointer 0.19110 UNMOVED (means). #205 untouched.**

## TL;DR (answer first)

Both crucible-3 A.3 reformulations of the N-1-falsified area-mass b_c objective were BUILT as real
selectable solver modes and MEASURED at full n600 through R on the SAME harness/checkpoint/protocol as the
N-1 verdict (baseline reproduced **bit-identically**: 0.003143556382921007). **BOTH LOSE, decisively — the
global post-hoc decode-time b_c lever is SATURATED at no_offset for this trunk.** v8 increment-1a ships
`b_c = no_offset` (SPEC_v8 §A.3's safe default is now the MEASURED optimum).

## The measured 3-arm gate (realized-through-R d_seg, frozen CPU SegNet, mod32cap ep650 EMA-BEST, n600)

| arm | n600 d_seg | Δ vs no_offset | rel-sig ÷0.0411 | b_Lane |
|---|---|---|---|---|
| **no_offset** | **0.0031436** | — | — | 0 |
| flip_weighted (OT→flip-share) | 0.0196734 | **+0.01653 WORSE** (6.3×) | 40% of gap, as HARM | +48.8 |
| flip_median (S1 Hamming median) | 0.0215612 | **+0.01842 WORSE** (6.9×) | 45% of gap, as HARM | +43.4 |

N-1 reference (same protocol): menon 0.0033119 · ot_newton 0.0048921 — the flip arms are ~4× worse than
even the falsified AREA objective. Per-class Lane d_seg: 0.212 (no_offset) → 0.588 / 0.613 (the arms that
"target" Lane make it ~3× worse). Result JSON sha256 `ad3f863e9de0e0d8bc14df8ee16f8e2564fa01938f2bd7cf8a9511fb2b86fb8f`
(`experiments/results/flip_bc_n600_gate_20260709/flip_bc_n600_result.json`; log `n600.log`, 44.5 min local
M5 Max CPU, verdict-batch 32 chunked per the n600-verdict-OOM law).

## What was BUILT (NO-FAKE, each mode does its claimed work on real inputs)

- **`flip_median_offsets`** (`src/tac/boundary_math/laguerre_logit_offset.py`) — S1's Hamming-optimal
  per-edge flip-margin median. Crucible-3 P3 F3 was right: it is NOT expressible as an OT target-mass
  choice; built as a sibling closed-form solve: per unordered edge {i,j}, `b_i − b_j = −median(φ_i − φ_j
  over the edge's flip pixels)`, reconciled across edges by a flip-count-weighted graph-Laplacian least
  squares with the zero-sum pinv gauge. Zero flips → b=0 (correct no-op).
- **`solve_head_offsets(mode="flip_weighted")`** — the SAME BUILT damped-Newton OT solver (#288), target
  masses = `perclass_verdict.flip_share_by_class` (the canonical sensor, lazy-imported; derived internally
  from phi+gt — a raw `target_masses` that could smuggle area counts back in is structurally rejected).
- **Realized-flip identification (`pred` kwarg):** flips are identified by the REALIZED baseline SegNet
  argmax, not the phi-argmax proxy (which over-counts ~50× on this witness: 370,829 realized flips vs
  ~7.4M phi-proxy flips at n600). Both solvers accept `pred`; `pred_is_realized` is stamped in info.
- **NO-FAKE raises:** every mode raises `LaguerreLogitOffsetError` without its required inputs; no mode
  silently degenerates to another.
- **DSL leg:** `HeadOffsetSolver(mode=...)` extended to the 4-mode vocabulary (`witness_dsl/
  curriculum_dsl.py`, commit dd5057c6c). **Trainer leg:** `--head-offset-solver` choices + `gt` wired at
  the advisory verdict readout (`train_levelset_witness_realized_through_R_mlx.py`).
- **Gate:** `experiments/probe_flip_bc_n600_gate.py` — reuses the N-1 probe's caching + through-R
  machinery verbatim (imports from `probe_laguerre_logit_offset_sweep`); solves on the cached phi stack;
  chunked SegNet forward.
- **Tests:** 13 new (`src/tac/boundary_math/tests/test_flip_offsets.py`) + updated solver-set pin; 58
  boundary-math offset tests pass; ruff F clean.

## Mechanism (why both lose, MEASURED)

The trunk's phi-space Lane erasure is DEEP: the realized-flip pixels carry LARGE phi margins (per-edge
flip-margin median ≈ 43 logits — the witness is confident-wrong at the flips, not near-tie). So any global
5-scalar offset that reaches the flip mass must boost Lane by ~+43…+49 → over-predicts Lane EVERYWHERE
(Lane per-class 0.212→0.61, Road 0.0047→0.022). On a synthetic NEAR-BOUNDARY erasure (small flip margins),
flip_median HELPS (−0.025, b₁=+0.23 — test-pinned direction); the real trunk is simply not in that regime.

## PER-MODE verdict scopes (verdict-scope ladder; one failed formulation ≠ family dead)

- **M-a flip_weighted — verdict_scope: FORMULATION** ("flip-share targets through the mass-OT mechanism").
  P3 F3 empirically CONFIRMED: OT-to-flip-share re-inherits and AMPLIFIES N-1's cell-inflation pathology.
- **M-b flip_median — verdict_scope: FORMULATION with a REGIME qualifier.** The Hamming-median derivation
  is VALID in the small-offset local-boundary-shift regime; the solved b₁≈+43 EXITS that validity domain.
  A **REGIME VIOLATION on THIS eroded trunk, NOT a refutation of the median law itself.**
- **Shared conclusion:** no GLOBAL per-class offset rescues an eroded trunk. With menon / ot_newton /
  flip_weighted / flip_median all measured worse AND N-1's exhaustive ±0.4 sweep best at Δ −3.4e-8, the
  global post-hoc b_c lever is SATURATED at no_offset here. NOT the solvers (exact/closed-form, tested).

## untested formulations / alternatives (reformulation queue)

1. **per-EDGE b_c on the fresh v8 Stage-A decoupled fields** (S1's original per-edge form — the crucible-3
   v8 route; non-eroded by construction). The natural next home for the median law inside its validity domain.
2. **Offsets solved jointly WITH training** (not post-hoc on a frozen eroded phi).
3. **Magnitude-clamped offsets** within the derivation's validity domain (the N-1 sweep's ~0 best bounds
   this at ~nil headroom on THIS trunk).
4. Per-pair / spatially-varying b_c(x) (annulus-gated).

## Provenance (STORES CONSULTED + artifacts)

- N-1 verdict: `.omx/research/ot_offset_n600_verdict_20260709.md` (harness cited + reused) · crucible-3
  `SYNTHESIS_DRAFT_v8_20260709.md` §A.3/D3 + `P3_redteam_verdict_20260709.md` F3 · `docs/
  operating_manual_craft_handoff.md` · sensor `tac.witness_control.perclass_verdict` (flip_share_by_class).
- Checkpoint (READ-ONLY): `levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`
  (ep650) · GT: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`.
- **Triality:** DAG `FEED-flipbc` · DSL `HeadOffsetSolver` 4-mode (dd5057c6c) · equations anchor
  `laguerre_flip_bc_reformulation_gate_n600_mod32cap_ep650_20260709` appended to
  `laguerre_ot_head_offset_v1` (per-mode scopes + validity_domain, VERIFIED_VIA_EMPIRICAL_ANCHOR).

means ≠ ends: this unit BUILDS + MEASURES the mechanism at authoritative advisory scale; it makes NO score
claim. Pointer **0.19110 UNMOVED**.

## Canonical equations (Catalog #344)
Consumes `flip_margin_step_law_v1` + the v8 geometric-rate decomposition equations in `tac.canonical_equations`; the b_c gate's own law # FORMALIZATION_PENDING: registers on the first through-R measured gate row.
