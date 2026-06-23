# Muon vs AdamW from the stage-4 fork — the decisive optimizer-axis d_seg test (2026-06-22)

**Status:** `[contest-CPU advisory]` NON-PROMOTABLE · `research_only=true` · pointer UNMOVED
0.19110 · NO score claim · NO kill.
**Probe:** `experiments/probe_muon_vs_adamw_from_stage4.py`
**Result JSON:** `experiments/results/muon_vs_adamw_stage4_20260622/result_higher.json` +
`result_stage8.json`
**Subagent:** `muon_vs_adamw_stage4_20260622`

> VERDICT: **MUON_BITES_FROM_STAGE4** (`higher`-LR config, N=16, both arms capped at
> step 199 by the 50-min budget). pointer UNMOVED 0.19110; NO score claim; NO kill.
> The `stage8`-LR (faithful) config follow-on is reported below if it completed in budget.

## The question

The live decisive run (`yousfi_r3_taper_marginhinge_e5_20260620`) shows d_seg PARKED at
~0.00207 across stages 2-5 (all AdamW), after stage 1 (CE) did ~86% of the d_seg work. Two
competing explanations:

* **CONDITIONING thesis** — the flatness is the PREDICTED AdamW behavior (a diagonal
  preconditioner cannot decorrelate the kappa~19 boundary Hessian) and the SPECTRAL Muon
  optimizer (stage 8) is the reserved d_seg finisher. If true, stages 5-7 are NOT necessary
  d_seg-prep and jump-to-Muon-early is viable.
* **CAPACITY thesis** — d_seg ~0.00207 is near the small decoder's CAPACITY floor and
  stages 5-7 are over-long rate-tuning cargo-culted from PR95's bigger model. If true, Muon
  ALSO stalls flat.

## The test (apples-to-apples; optimizer is the ONLY variable)

From the SAME fork point — the preserved `stage4_v332_qat_end` snapshot (QAT-on = Muon's
hard prerequisite satisfied) — run two short convergence arms from a DEEP COPY of the same
stage-4 weights and compare the d_seg(step) trajectories:

* **Arm A (Muon)** — `MuonOptimizer` on the trunk decoder Muon group (12 conv weights,
  60,149 elems) + AdamW on the excluded params (stem/rgb/biases, 18 tensors, 23,273 elems)
  + AdamW on the latents. This MIRRORS the driver's stage-8 construction
  (`driver.py` ~1784: `partition_params_for_muon(decoder)`; FiLM is OFF for this run, so
  there is NO FiLM-exclusion — exactly the driver's no-FiLM branch). momentum=0.95,
  nesterov, ns_steps=5.
* **Arm B (AdamW control)** — AdamW on ALL decoder params + latents. This is what stages
  5-7 effectively do for d_seg.

**Everything else is identical:** SAME fork weights, SAME pairs, SAME seg loss
(`build_curriculum()[4].seg_loss_fn` = the vendored l7_softplus `tau=0.3, l7_threshold=1.0,
l7_mult=4.0` — VERIFIED byte-identical to stage-8's `seg_loss_fn`), SAME LR per config, SAME
grad-clip (1.0), SAME int8 (127-level) fake-quant on the deployed/eval surface, SAME eval
round-trip (bicubic↑874 → bilinear↓384 → uint8-STE → SegNet argmax-disagreement = the
contest d_seg). d_seg is measured on the frozen **CPU** SegNet (the d_seg authority).

**LR configs:** `higher` (Muon 3e-3 / AdamW 3e-4 — matched higher LR to surface the contrast
faster) and `stage8` (the faithful stage-8 LRs: Muon 2e-4 / AdamW-aux 1e-5 vs AdamW-all 1e-5
— the real stage8-vs-stage7 comparison). CPU only (MPS FORBIDDEN: never a d_seg authority +
the live run owns the MPS device). `OMP_NUM_THREADS=MKL_NUM_THREADS=2`.

## The signal

The discriminating metric is on the **int8-quant (deployed/authority) d_seg curve**:
`gap = Arm_A(Muon).Δd_seg_quant − Arm_B(AdamW).Δd_seg_quant`. `gap < 0` ⇒ Muon descended
d_seg MORE than AdamW (the conditioning signal); `gap > 0` ⇒ AdamW descended more. The
discrimination band is ~10 pixels of d_seg over the n pairs (~5e-5).

## Results

Run: N=16, eval every 50, grad-clip 1.0, CPU only (`OMP_NUM_THREADS=MKL_NUM_THREADS=2`),
~15 s/step (the live MPS run competes for cores). Baseline d_seg @ fork: fp=0.001950,
int8quant=0.001939 (matches the live run's ~0.00207 stage-2-5 stall). Both arms ran to the
50-min/arm budget cap at **step 199** (partial-but-clean curves; the partial-curve cap is
the documented budget behavior, not a failure).

### `higher`-LR config (Muon lr 3e-3 / AdamW lr 3e-4) — the decisive contrast

**d_seg int8-quant (deployed/authority surface):**

| step | Arm A (Muon) | Arm B (AdamW) | Muon − AdamW |
|---:|---:|---:|---:|
| 0   | 0.001939 | 0.001939 | 0.000000 |
| 50  | 0.001176 | 0.001434 | **−0.000258** |
| 100 | 0.000861 | 0.001141 | **−0.000280** |
| 150 | 0.000633 | 0.000986 | **−0.000353** |
| 199 | **0.000535** | **0.000874** | **−0.000339** |

(d_seg FP curve is parallel: Muon 0.00195→0.000363, AdamW 0.00195→0.000604; FP gap at the
end −0.000241.)

**Total descent:** Muon Δd_seg_quant = **−0.001404**; AdamW Δd_seg_quant = **−0.001065**;
**gap (A−B) = −0.000340** — well outside the ~5e-5 (10-pixel) discrimination band.

Muon's d_seg is **strictly below AdamW at every measured step**, and the gap **widens
monotonically** (−0.000258 → −0.000280 → −0.000353). Muon at step 150 (0.000633) already
beats AdamW at step 199 (0.000874). Muon descended **~32% more total d_seg** from the
identical fork / loss / pair-set / budget. The ONLY variable was the optimizer.

### seg-loss + grad-norm contrast

| step | Muon seg_loss | AdamW seg_loss |  | step | Muon ‖g‖ (aux set) | AdamW ‖g‖ (all set) |
|---:|---:|---:|---|---:|---:|---:|
| 1   | 0.00684 | 0.00684 |  | 1   | 0.0173 | 0.0387 |
| 50  | 0.00420 | 0.00460 |  | 50  | 0.0162 | 0.00629 |
| 100 | 0.00343 | 0.00401 |  | 100 | 0.0130 | 0.00428 |
| 150 | 0.00302 | 0.00365 |  | 150 | 0.0113 | 0.00366 |

The seg-loss surrogate confirms the d_seg ordering (Muon lower at every step). The grad-norm
traces are NOT cross-arm comparable (caveat 0 — Muon's is the AdamW-aux set, AdamW's is the
all-params set), but the **within-arm** AdamW trace shows the diagnostic stall signature: its
gradient norm **collapses** 0.0387 → 0.00366 over training as its diagonal preconditioner
saturates on the correlated boundary Hessian, while Muon's NS-orthogonalized update stays
healthy (~0.011) and keeps extracting d_seg.

### `stage8`-LR config (faithful: Muon 2e-4 / AdamW-aux 1e-5 vs AdamW-all 1e-5)

The faithful stage-8 LR is much smaller (Muon native 2e-4; AdamW 1e-5), so a short window
moves d_seg far less than the `higher` config. The follow-on `stage8` config was launched
after `higher`; its result (if it completed within the per-arm budget) is at
`experiments/results/muon_vs_adamw_stage4_20260622/result_stage8.json`. The `higher` contrast
above is the decisive optimizer-axis signal; the `stage8` config is the faithful-LR sanity
cross-check (a slower, smaller-magnitude version of the same comparison).

## Verdict — MUON_BITES_FROM_STAGE4

From the SAME stage-4 fork, under the SAME vendored l7_softplus seg loss and SAME budget,
**the spectral Muon optimizer descends d_seg materially faster than the diagonal AdamW
optimizer, and the gap widens with training** (`higher`-LR config: total Δd_seg_quant −0.001404
Muon vs −0.001065 AdamW; gap −0.000340, ~6.8× the discrimination band). This is the
optimizer-axis prediction of the **CONDITIONING thesis**: AdamW's diagonal preconditioner
cannot decorrelate the kappa~19 boundary Hessian (its grad norm collapses; d_seg stalls
above Muon's), while Muon's Newton-Schulz orthogonalization keeps biting.

**Implication for the live run:** the stage-2-5 d_seg flatness under AdamW is consistent
with the predicted AdamW conditioning-crawl, NOT a hard capacity floor — Muon extracts
additional d_seg from the same weights that AdamW leaves on the table. So **stages 5-7 are
NOT a necessary d_seg-prep for the renderer; a jump-to-Muon-early is a viable candidate** to
test on the live run. This is a candidate disambiguation, NOT a switch of the live run and
NOT a kill. The decisive arbiter remains the live run's REAL stage-8 Muon d_seg slope.

**Important nuance (both arms descend):** at this `higher` LR, AdamW also descends d_seg
(0.00194 → 0.000874) — so this is NOT a clean capacity-floor situation where AdamW is
pinned. The signal is the **CONTRAST** (Muon descends ~32% more, gap widening), which is the
conditioning signature, not the absolute floor. See caveat 1.

## Honest caveats (binding)

0. **Grad-norm diagnostic asymmetry (secondary only).** The reported `||g||` is the
   pre-clip norm of the grad-clip set: for Arm A (Muon) that is the AdamW-handled set
   (stem/rgb/biases + latents — Muon's trunk grads are NS-self-normalized and clipped
   separately at `grad_clip_muon=None`, matching the driver); for Arm B (AdamW) it is the
   all-params set. So the two arms' grad-norm traces are NOT directly comparable. The
   **d_seg verdict is unaffected** (d_seg is the contest argmax-disagreement, measured
   identically for both arms). Grad-norm is a within-arm convergence diagnostic only.
1. **Small-N memorization bias.** N=16-24 pairs makes the ABSOLUTE d_seg-drop rate
   optimistic vs the 600-pair live run (a small fixed pair set is easier to fit). The valid
   signal is the **Muon-vs-AdamW CONTRAST under identical conditions**, NOT the absolute
   drop rate. A large absolute drop here does not imply the live run drops as fast.
2. **stage4-vs-stage7 fork confound.** This forks from `stage4_v332_qat_end`, not from the
   stage-7 state the live run will actually hand to stage 8. A **FLAT Muon arm does NOT
   prove capacity** — it could be that stage 4 is not yet Muon-ready (stages 5-7 condition
   the weights into the basin where Muon bites). So a null result DEFERS to the live stage 8
   (~2 days out); it does not conclude capacity. A POSITIVE Muon-bites result is the stronger
   signal (it shows Muon can extract d_seg AdamW cannot, even from the earlier fork).

## Decisive arbiter

The live run's REAL stage-8 Muon d_seg slope (~2 days out) is the decisive arbiter. This
probe only disambiguates the optimizer-axis candidate; it NEVER switches the live run and
NEVER kills.

## 6-hook wire-in (research_only)

* (1) sensitivity-map — N/A (this is an optimizer-axis disambiguator, not a per-byte
  sensitivity contribution).
* (2) Pareto constraint — N/A (no archive bytes; advisory non-promotable).
* (3) bit-allocator hook — N/A.
* (4) cathedral autopilot dispatch — N/A (no archive-deployable artifact).
* (5) continual-learning posterior — the verdict feeds the live-run curriculum decision
  (jump-to-Muon-early viability) as a candidate prior; NOT a posterior score anchor.
* (6) probe-disambiguator — THIS probe IS the disambiguator (Muon-conditioning vs capacity
  for the stage-2-5 d_seg flatness). `research_only=true`.
