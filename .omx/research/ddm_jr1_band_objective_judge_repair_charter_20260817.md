# ddm_jr1 charter — REPAIR THE JUDGE on the only seg rung whose ceiling exceeds the gap

Operator authority: standing autonomous frontier-lowering GO + 2026-08-16 "full authorization to
pursue all pattern follow ons work and regimes level work and everything". Owner of any heavy
FIRE: **MAIN** (governed Metal slot). This arm runs the $0 legs, repairs the instrument, and seals
a fire-ready ticket. It does **not** launch training.

## Why this rung and not another (measured, do not re-derive)

`sx1` priced the entire seg axis and the result is a corner solution.

| rung | bytes | ceiling | status |
|---|---:|---|---|
| byte-carrying correction channel | 33,235 | needs **η 1.0069** to close the gap | **DEAD** — η ≤ 1 by construction; measured η 0.6235, 0-of-9 above bar |
| R4 flat-anchor band repaint α-ladder | 15 | — | **REFUTED 5/5** (`ddm_rt1_r4_alpha_ladder_verdict_20260816.md`) |
| R7 sub-pixel / AA pre-R placement | 0 | — | **CLOSED** — rt1 measured R supplies **exactly 0** flips |
| **R6 band objective in training** | **0** | **−0.028604 S = 298% of the gap**, needs only **33.55%** realized | **BUILT, judge defective** |

R6 is the only rung on the axis whose ceiling exceeds the gap, and it costs zero bytes, so any
realized recovery is pure gain. Its blocker is **instrumental**, and the repair is mostly free.

**The mechanism R6 attacks (rg1b, MEASURED on real transmitted-label fields):** the stock
`curriculum_loss` allocates **2.161%** of its gradient mass to the 1-px label band, which is
**2.157%** of pixels — ratio **1.0016**, i.e. strictly by AREA — while **99.22%** of the seg debt
sits there. That is a **45.9× misallocation** and an **83.3°** angle between the stock gradient and
the debt-weighted gradient, in all three curriculum phases.

**Why the one run that exists proved nothing:** rg1b measured that in this regime
`peak_flips = 118,563.2·‖Δw‖₁₀₀^0.457640` (r² 0.9969, σ_log 0.0728). Flips are a function of weight
displacement alone, so a fixed-step flip-trajectory probe **cannot distinguish objectives at all**.
The band arm's `improved_over_init = NO` invalidates the probe design, not the objective — and the
objective demonstrably did rotate the realized update (`cos` < 0.90 on the Adam-relevant metric in
**all nine** cells, < 0.95 everywhere).

## THE DEFECT I AM HANDING YOU, WHICH IS NOT IN ANY MEMO

`sx1` §5.3 cites the band arm at **−0.871σ** off the law as encouraging. Read `rg1b` §6.3 at source:
the band arm is **the fifth point the law is fit on**. An in-sample residual against a curve that
was fit including the test point is not a direction signal. This is the pk3/pk4 defect
(23/23 in-sample → 0/23 leave-one-out, `ddm_pk4_optimal_form_frame0_pose_verdict_20260813`) one
level up — in the judge instead of the model. **Do not inherit the −0.871σ.** Re-derive it
out-of-sample or report that it cannot be.

## Legs

**LEG A — the out-of-sample residual judge ($0, no training run).**
Refit the displacement law on the **stock arms only**, with the band arm held out, then score the
band arm against that out-of-sample curve. Report the honest σ and the honest n. Establish at
source how many arms exist and what each is: `/Volumes/APDataStore/pact/ddm_lr1/{C0,A1,A2,A3,W1}`
has five directories while rg1b calls the band arm "the fifth point" — resolve that discrepancy from
the receipts, do not assume. If leave-one-out is not computable at n this small, say so plainly and
state the n it would need; a refusal with a number is worth more than a fitted story.

**LEG B — the realized-AdamW cosine ($0, desk computation).**
rg1b's item 3. It measured the *raw* and *sign-limit* cosines; the true update uses accumulated
`m`/`v`, which are retained. Reconstruct the realized AdamW step and close the gap between
"gradient rotated 83°" and "**step** rotated". This is the difference between a claim about the
loss and a claim about what the optimizer actually did.

**LEG C — the matched-‖Δw‖ ticket (SEAL ONLY; MAIN fires).**
rg1b's item 1(a): compare arms at **matched ‖Δw‖₁₀₀** rather than matched steps. Derive the step
budget each arm needs to reach a common displacement target from the law itself. Per-stage
checkpoints P0, resumable-from-disk, memory preflight at the real config, governed launcher.
Emit the exact command; do not run it.

## PRE-REGISTERED FORK — write this before you measure

- **DIRECTION REAL:** band arm's out-of-sample residual is negative beyond the honest significance
  bar you state in Leg A, **and** Leg B shows the realized step (not just the gradient) rotated.
  Then R6 is a live supplier on the only rung whose ceiling exceeds the gap → Leg C's ticket is the
  campaign's next heavy fire, and you say so.
- **DIRECTION NULL:** residual within noise out-of-sample. Then the pixel-reweighting family closes
  at **FORMULATION** scope with a measured law — a real result — and the seg axis routes to R5
  (solved-prototype ordered camera paint, ancestor-scoped: the *mechanism* transfers, the numbers do
  not, per [[m18]]) or to the vehicle question. Say which, and why.
- **UNDERPOWERED:** n is too small to separate the two. Then name the exact number of arms and the
  exact displacement spread that would separate them, and price that as Leg C's real cost. This is a
  legitimate outcome; a fabricated significance is not.

## OPTIMAL FORM

- **Reference form.** The family's reference is rg1b's own instrument, at its own scale, on its own
  vehicle: `src/tac/pr130_lift/band_objective.py`
  (sha256 `81e187f67e03757a19b21873ce9abd10f5f7f5e05aa348e93001e751dfa7060e`) wired into
  `src/tac/pr130_lift/train_semantic_quantized_resumable.py`
  (sha256 `b486f416e99efe14361f36603703ea68788ceba59f10fba50e92681438b086fe`) behind
  `--band-objective-weight` (default 0.0, byte-identical when off; argparse at :930, consumers at
  :1179/:1232, provenance at :1496). Instrument control PASSES bit-exact: the band recomputed from
  transmitted labels equals the retained `free_band_mask.npy` frame-for-frame across all 600 frames
  at **2,551,464** px. Same n600 population, same retained payloads
  (`/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/`,
  `/Volumes/APDataStore/pact/ddm_lr1/`), same displacement law.
- **Every delta is a SCOPE reduction, none is a MECHANISM reduction.** Legs A and B are pure
  re-analysis of *already-retained* payloads at full n600 — no reduction of any kind. Leg C is
  design-only. The objective, the weight table, the band geometry, and the trainer are used exactly
  as built; nothing is stubbed, sampled, or stood in for.
- **TOY-BRACKET: none.** If a leg cannot be done at real form inside budget, deliver it as a
  declared unbuilt follow-on naming the exact blocker. Never a stub that pretends.
- **Provenance pins.** rg1b memo sha256
  `e1306299a19901923bcde9c6ec5ec336beadf689b989bdda0f11e9cb89e3643b` · sx1 memo sha256
  `33703277bd988dbc71743e28f471f58087dc37d75e2f351f8e1d2b588884d0ef` · rt1 memo sha256
  `3ad50f7bc77c6a4abb739321d734ed820a86ed754e804fa93e1ed362f6d9b972`.

## Recall you MUST do at source (never from this charter's summary)

`ddm_rg1b_band_objective_build_20260816.md` §6.1–§6.6 (the law, the cosine cells, the adjudication,
and what §6 explicitly did NOT establish) · `ddm_sx1_seg_cure_ladder_20260816.md` §4–§6 (the
joint-support MAX law: every seg lever acts on the same 2,551,464-px support, so recoveries **cannot
be summed**) · `ddm_rt1_seg_roundtrip_decomposition_20260816.md` §6 · the R4 verdict
`ddm_rt1_r4_alpha_ladder_verdict_20260816.md` (do not re-propose R4) ·
`ddm_b2e_edit_replay_admission_verdict_20260816.md` (its NEXT_IF_RESUMED row 2 asks whether this
trainer trains **at all**: 3,000 steps at lr 2e-7 moved ΔS_adv by +0.000336 and weight entropy by
9 bytes — if Leg C's displacement target is unreachable at the working lr, that is a Leg-C blocker
and you must say so) · the wd3 disposition law (judge fresh students at the **seg asymptote**; a
negative needs n120, never n60) · the lr1 F3 instrument warning (read `history` in every lr1
`result.json`, **never** the top-level headline).

## Constraints (binding)

Commit via `tools/subagent_commit_serializer.py` with `--expected-content-sha256` set to the
**POST-EDIT working-tree sha** of each file (not HEAD — this is the single most common arm error).
Two review passes on every `.py` via `tools/review_tracker.py mark-file`; never `REVIEW_GATE_OVERRIDE`
on `.py`. **ALWAYS KEEP THE PAYLOAD** — persist bytes plus sha256 and byte count for anything you
materialize; a scalar-only artifact where a payload existed in memory is forbidden at the typing
moment. `upstream/` is read-only. No Modal spend (cap is 93.1% consumed). Payloads to
`/Volumes/APDataStore/pact/`; **read** from `/Volumes/VertigoDataTier/` but never write there.
Every number carries its axis label — `[macOS-CPU advisory]` is never a score. Persist the final
message with a `NEXT_IF_RESUMED` block.

**Own-vehicle frontier, unmoved by this unit:** hv1 ep0634 **S 0.15959729295498598 @ 182,759 B**
`[contest-CUDA T4 n600]`, archive sha256
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`. Gap to 0.15: **−0.0095973**.
