# OT head-offset n600 verdict — #288 subset verdict CONFIRMED at full scale (task #381 item-2)

**Date:** 2026-07-09 · **Author:** claude recovery subagent (`ot_offset_n600_recovery_20260709`)
**Authority:** `[macOS-CPU advisory]` — realized-through-R on the frozen CPU SegNet, fp32 EMA render.
NON-PROMOTABLE until byte-closed exact eval. **Pointer 0.19110 UNMOVED (means). #205 untouched.**

## TL;DR (the verdict)

The #288 damped-Newton semi-discrete OT per-class head-offset **mass-matching objective is a MEASURED
NEGATIVE for realized d_seg, CONFIRMED at the full admissible n600 scale.** The subset ordering
(`no_offset < menon < ot_newton`, lower = better) REPRODUCES with the same order and sign at all 600
pairs. No offset arm improves d_seg; **no-offset wins.** The OT solver itself is EXACT (converged,
mass-matched to 2.8e-11) — the negative is about the *objective* (matching Laguerre cell masses to GT
class frequencies), NOT the solver.

**verdict_scope: FORMULATION** — "cell-mass-matching to GT class frequencies as a d_seg surrogate", at
THIS checkpoint (mod32cap ep650 EMA-BEST) and THIS tau (1.0). NOT a PARADIGM or FAMILY kill: the OT
solve is correct and byte-free; other target-mass definitions are UNTESTED (see reformulation queue).

## The measured 3-arm gate (realized-through-R d_seg, frozen CPU SegNet, lower = better)

| scale | no_offset | menon (Δ vs no-offset) | ot_newton (Δ vs no-offset) | source |
|---|---|---|---|---|
| n2  | 0.0030060 | 0.0032094 (+2.03e-4) | 0.0065155 (+3.51e-3) | smoke (numpy path, MLX blocked) |
| n6  | 0.0028314 | 0.0030755 (+2.44e-4) | 0.0052202 (+2.39e-3) | smoke |
| n24 | 0.0027349 | 0.0029672 (+2.32e-4) | 0.0047860 (+2.05e-3) | #288 subset |
| n48 | 0.0027252 | 0.0029322 (+2.07e-4) | 0.0048664 (+2.14e-3) | #288 subset (equation anchor) |
| **n600** | **0.0031436** | **0.0033119 (+1.68e-4)** | **0.0048921 (+1.75e-3)** | **THIS gate (all 600 pairs)** |

- **Order + sign IDENTICAL at every scale.** The absolute d_seg is slightly higher at n600 (0.00314 vs
  0.00273 at n48 — the n48 subset was marginally easier), but the RELATIVE verdict is unchanged: both
  offset arms HURT; mass-matching (ot_newton) hurts ~10× more than the priors-only Menon heuristic.
- **OT solver correctness (n600):** `converged=True`, `iters=8`, `max_mass_err=2.82e-11` — the
  damped-Newton solve on the concave semi-discrete OT dual reaches the target masses exactly. The
  negative is the objective, not the solver.

## Mechanism (why mass-matching hurts, MEASURED at n600)

Per-class d_seg, no-offset → ot_newton (n600):

| class | prior | no_offset d_seg | ot_newton d_seg | offset b_c (ot) |
|---|---|---|---|---|
| 0 Road | 0.2323 | 0.00468 | 0.00848 | −9.64 |
| 1 Lane | 0.0059 | 0.21182 | 0.26515 | **+26.47** |
| 2 Undrivable | 0.4952 | 0.00068 | 0.00132 | −7.54 |
| 3 Movable | 0.0124 | 0.02816 | 0.04582 | −5.78 |
| 4 MyCar | 0.2543 | 0.00050 | 0.00059 | −3.51 |

The OT solve inflates the rare-Lane cell (offset **+26.5**) to hit its 0.59% GT mass. But this
OVER-predicts Lane relative to the witness's own boundary placement, and the frozen SegNet re-read
PENALISES the over-prediction — Lane per-class d_seg rises 0.212 → 0.265, and EVERY class gets worse.
The Menon offsets are smaller (Lane +2.53) so the harm is milder but still net-positive. The witness's
learned boundary placement (no offset) is already closer to the scorer's argmax than any mass-corrected
reweight. This is the same mechanism the subset gate measured, now confirmed at full scale.

The winner sweep independently agrees: best swept offset is class-1 −0.2 with Δ = **−3.39e-8** (a
numerically negligible improvement, i.e. effectively no-offset is optimal over the ±0.4 grid).

## Modal harvest outcome (HARVEST-OR-LOSE, honest)

- **Modal call `fc-01KX4B0A620FCQR6ZWJKRZ7XRT` (comma-ot-offset-n600, Linux x86_64 CPU) FAILED:**
  dispatched 2026-07-09T21:02:30Z, **CANCELLED at 21:14:28Z (~12 min)**. Modal `cpu=2` cached only
  24/600 pairs in 426 s (~3 h projected for the full caching pass — impractical). The predecessor
  cancelled it and pivoted to a LOCAL macOS-CPU run. `FunctionCall.get()` now returns
  `RemoteError: Function call was cancelled by user or a failure` — NO n600 result from Modal.
- **Cost: ~$0.15 ESTIMATE** (cpu=2 + 16 GiB × ~12 min; Modal exposes no per-call cost via CLI, so this
  is derived from elapsed × rate). Recorded to `.omx/state/modal_call_id_ledger.jsonl`
  (`event_type=manually_terminated`, `cost_actual_usd=0.15`) + terminal lane-claim row
  (`stopped_modal_cancelled_pivot_local`).
- **The authoritative n600 result is the LOCAL run** (the predecessor's detached probe, pid 55972,
  started 2026-07-09 16:16 CDT, survived the predecessor's session death, ran ~79 min on the M5 Max
  and finished cleanly). Recovery subagent harvested its output; a redundant recovery re-run (pid
  45862) was killed to avoid a same-path write race.

## Provenance (STORES CONSULTED + artifacts)

- **Checkpoint (READ-ONLY):** `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/
  levelset_witness_ema_BEST.npz` (mod32cap ep650 EMA-BEST; sha256 `6dd28a6e295d007e…`) — the SAME
  snapshot the #288 subset gate used. Never mutated; offset folds byte-free into a COPY of
  `out_sdf.bias`.
- **GT cache (slim lstars-only):** `experiments/results/ot_offset_n600_modal_20260709/
  gt_n600_lstars_slim.npz` (file sha256 `e41718a047b77b90…`; lstars content sha per dispatcher docstring
  `bf1d0e5c…`) — real frozen-SegNet argmax GT, all 600 pairs.
- **Result JSON (durable local evidence; results/ is gitignored):**
  `experiments/results/ot_offset_n600_modal_20260709/ot_offset_n600_LOCAL_result.json`
  (sha256 `55158cdf613ab105…`). Progress log: `local_n600.log`.
- **Gate probe:** `experiments/probe_laguerre_logit_offset_sweep.py` (3-arm no-offset / menon /
  ot_newton, realized-through-R). **Mechanism:** `tac.boundary_math.laguerre_logit_offset.
  {damped_newton_ot_offsets, solve_head_offsets}`. **Dispatcher (committed by this unit):**
  `experiments/modal_ot_offset_n600_gate.py`.
- **STORES CONSULTED:** DAG `FEED-otoffset` (task #288, line ~11908 — the subset gate n48/n24 verdict) ·
  DAG `FEED-04a` (the damped-Newton OT solver BUILD) · canonical equation
  `laguerre_ot_head_offset_v1` (`src/tac/canonical_equations/laguerre_ot_head_offset_20260709.py`; n48 +
  solver-correctness anchors) · subagent_progress `ot_offset_wire_288` (the DSL+eq+probe+trainer wire-in) ·
  the n2/n6 smokes in the run dir.

## Reformulation queue (the FORMULATION-negative leaves these OPEN — do NOT close the paradigm)

Per CLAUDE.md "Forbidden premature KILL" + verdict-scope ladder — the mass-matching objective failed at
this checkpoint/tau, but the OT solver is exact and byte-free. Still-open reformulations:

1. **Flip-weighted target masses (the primary queued arm).** Match masses weighted by per-pixel FLIP
   probability (margin/saliency) rather than raw GT frequency — target the boundary-annulus mass the
   scorer actually re-reads, not the bulk cell mass. Raw-frequency matching over-inflates the rare-Lane
   BULK; flip-weighting would concentrate on the codim-1 boundary where d_seg lives.
2. **Other tau** (the soft/hard cell-mass temperature; only tau=1.0 measured).
3. **Other converged checkpoints** (only mod32cap ep650 measured; a witness with different boundary
   placement may respond differently).
4. **Per-pair (not global) offsets** — the current solve uses one zero-sum b* over all pairs.

## Triality legs

- **DAG:** `FEED-otoffset-n600` (appended this unit).
- **equations:** n600 anchor appended to `laguerre_ot_head_offset_v1`
  (`laguerre_ot_head_offset_dollar0_gate_n600_mod32cap_ep650_20260709`, VERIFIED_VIA_EMPIRICAL_ANCHOR).
- **DSL:** N/A — this is the MEASUREMENT of a NEGATIVE lever; the `HeadOffsetSolver` DSL lever already
  exists (`--head-offset-solver {off,menon,ot_newton}`, default `off` = byte-identical) and stays
  default-off (the gate confirms it should NOT be turned on for d_seg at this checkpoint).

means ≠ ends: this unit MEASURES the mechanism at authoritative scale; it makes NO score claim. Pointer
**0.19110 UNMOVED**, #205 READ-ONLY/untouched.
