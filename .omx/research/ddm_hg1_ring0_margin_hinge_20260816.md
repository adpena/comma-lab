---
arm: ddm_hg1
title: "rn1's rho and its 1.53x prize REPRODUCE (re-priced 1.519x on the exact signed margin), but the lever rn1 routed to is ALREADY BUILT -- make_loss_fn's margin_hinge IS relu(target - (logit[GT] - max_{c!=GT} logit)) -- and the ring-0 restriction rn1 implies is a 0.44% NO-OP because the margin field is small only at the boundary; what is actually wrong is the TARGET (the trainer default 1.0 spends 97.65% of the hinge's gradient on already-correct pixels vs the DERIVED m_safe=0.03918), and the hinge cannot go on hv1's trainer at all because that trainer is a LABEL predictor with zero SegNet whose token field is already 34.9x better than the d_seg it feeds"
utc: 2026-08-16
parent: ".omx/research/ddm_rn1_render_boundary_mechanism_20260816.md"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet, batch-1, upstream preprocess verbatim -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle; family verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_hg1 — the signed ring-0 margin hinge

STORES CONSULTED: parent `ddm_rn1_render_boundary_mechanism_20260816.md` (all sections, plus its
five retained receipts) · `ddm_rt1_seg_roundtrip_decomposition_20260816.md` ·
`ddm_bo1_base_objective_menu_order_20260802.md` (§1.1, §3.2–3.4, §5 F2, addendum L528) ·
`ddm_control_surface_exact_quartering_20260731.md` (§3, §3b) ·
`ddm_q31_20260804/Q31_Q3_CONSTRAINED_SOLVE_RECEIPT_20260804.md` ·
`ddm_uv1_ep854_pose_illegibility_reject_20260802.md` + `ddm_cr2r_..._matched_control_20260802.md`
(the real #889 receipts) · `charters/ddm_js4_pose_null_projected_conditioning_20260812.md` ·
`reports/delta_R_noise_floor.json` · `experiments/train_tr1_partition_renderer_mlx.py` ·
`experiments/train_witness_realized_through_R_mlx.py::make_loss_fn` ·
`src/tac/boundary_math/levelset_micro_batch_loss.py` · `tools/train_ddm_cl1_hpac_capacity_mps.py` ·
`experiments/ddm_rx2_mc36_label_hpac.py` · memories [[m96]] [[m88]], CLAUDE.md
"SegNet vs PoseNet importance — operating-point dependent" + "off is a tracked queue".

## ANSWER FIRST

**rn1's finding reproduces. The lever it routes to already exists. The ring-0 framing that names
the prize does not need a ring-0 mask to collect it. And it cannot be built where the charter sent
me.**

1. **The re-derivation REPRODUCES, exactly.** Independent tool, independent quantity: flips
   **5,448**, ring-0 pixels **408,678**, correct-on-ring-0 **403,254** — all three identical to
   rn1. Median headroom at correct ring-0 **0.9703587** against rn1's **0.9703590**. rho(0.1)
   **2.158** against rn1's **2.138** (0.93%).

2. **My attack on rn1's proxy lands, and it is small.** rn1's ladder is built on `top1 − top2`,
   which equals the deficit a hinge must close only where GT is the runner-up. I computed the exact
   signed margin `m = logit[GT] − max_{c≠GT} logit[c]` directly. GT is the runner-up on **98.018%**
   of flips. The prize at +0.1 logits re-prices from rn1's **1.53× → 1.519×** of the remaining gap.
   The proxy was sound; the upper bound survives on the exact quantity.

3. **The hinge is ALREADY BUILT and I did not rebuild it.** `make_loss_fn`'s `margin_hinge` form is
   `relu(margin_target − signed)` with `signed = logit[GT] − max_{c≠GT} logit[c]` — the same object,
   term for term. Building a second one would have been a duplicate wearing a new name.

4. **The ring-0 restriction is a 0.44% NO-OP — MEASURED, and it falsifies my own build plan.** At
   the operative target the existing *global* hinge is already **99.56%** ring-0-concentrated. A
   support mask would remove **50 pixels of 11,281** active. The reason is structural: the margin
   field is small **only** near the decision boundary, so `relu(target − m)` at a small target is
   *already* an implicit ring-0 selector. rn1's ring-0 framing correctly says WHERE the prize is; it
   does not imply a mask is needed to reach it.

5. **What is actually wrong is the TARGET, and that is a new number.** The trainer's hand-typed
   `--margin-target 1.0` activates on **1.2295%** of the frame, of which only 5,448 pixels are
   flips — **97.65% of the hinge's gradient lands on pixels that are already correct.** The derived
   target is `m_safe = 2 × delta_R = 0.03918`, where `delta_R = 0.019590163` is the MEASURED p95
   uint8-induced margin perturbation. The default sits **25.5× above** the floor a corrected pixel
   needs to survive the round trip.

6. **THE VEHICLE FORK — the finding that changes the routing.** hv1's live trainer
   (`tools/train_ddm_cl1_hpac_capacity_mps.py` → the RX2 HPAC reference) contains **zero**
   references to SegNet, PoseNet, render, or rgb. It is a label-field predictor. Its shipped token
   field reproduces the GT SegNet argmax to **8.48e-06** while d_seg is **2.9611e-04** — the axis is
   **34.9× larger than the label error that feeds it**. So the entire seg axis is RENDERER
   realization error, the label half is already 35× past the point of mattering, and **the hinge
   cannot be added to hv1's trainer at all** — there is no SegNet forward there to hinge on. It
   belongs in `train_tr1_partition_renderer_mlx.py`, the partition renderer, which descends through
   R against the frozen SegNet and where the hinge already lives.

7. **Q3 is also already built, and the charter overstated it.** `--seg-grad-q3-project` exists,
   args-only, default off. But "a Q3-constrained variant CANNOT create pose damage (exact kernel)"
   is bo1's *sharpened, placement-conditional restatement*, which bo1 itself labels
   **UNBUILT/UNMEASURED** (L528). The exact kernel holds **pre-quantization only** — #532 measured
   uint8 rounding breaking it at **62.74 against 1.7e-13** — and the one Q3-constrained solve that
   exists (`ddm_q31`, n32) returned `Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE` with a d_pose
   mean ratio of **1.0424**, not flat. I therefore did **not** make Q3 a default of the hinge.

8. **What I built is the missing custody, not a missing term.** All four steering flags
   (`--seg-form-start`, `--margin-target`, `--margin-weighted-loss`, `--seg-grad-q3-project`) were
   UNMAPPED by the DSL on the live vehicle (coverage 19/116 = 16.4%). Two `Lever` factories now hold
   them, with the target resolved LIVE from the measured artifact through the registered
   `margin_band_satisficing_threshold_v1` law — never a literal. 15 tests.

**Pointer UNMOVED.** hv1 ep0634 remains S 0.15959729295498598 @ 182,759 B [contest-CUDA T4]. This
unit reproduced a coefficient, killed its own build plan with a measurement, corrected a routing,
and sealed an A/B. It did not lower the score.

## §0 Prior-law prediction lines — stated BEFORE the measurements

1. **Charter premise: rn1's ladder is optimistic because it uses `top1 − top2` instead of the exact
   signed margin.** PREDICTION: the exact ladder will be materially worse and will re-price the
   prize down. **HELD IN SIGN, WRONG IN SIZE** — 98.018% runner-up, prize 1.53× → 1.519×, a 0.9%
   haircut. I expected the 1.7% non-runner-up flips to sit deeper in the deficit tail. They do not.
2. **Charter task: build a ring-0 support restriction on the hinge.** PREDICTION: restricting the
   hinge to ring-0 will concentrate its gradient and is the buildable delta. **FALSIFIED** (§2) —
   99.56% already concentrated; the restriction removes 0.44% of the support.
3. **My own premise on opening the trainer: the hinge does not exist and must be written.**
   **FALSIFIED** (§3) — it exists, with the exact signed-margin definition.
4. **Charter constraint: build it Q3-constrained by default, because a Q3 burn cannot damage pose.**
   PREDICTION: I will compose Q3 into the treatment arm. **REFUSED after reading the receipts**
   (§5) — the claim is pre-quantization-only and its one build measured 1.0424, so composing it by
   default would both confound the A/B and rest on an overstated law.
5. **Suspected bug: the micro-batch loss twin ignores the margin weight for `margin_hinge` while
   the trainer's guard says the form honors it.** PREDICTION: a live silent-inert confound.
   **WITHDRAWN** (§6) — tr1 imports the *serial* `make_loss_fn`, which does honor it. The
   divergence is latent in the levelset twin only. I record it because I nearly reported it.

## §1 The re-derivation — independent quantity, same answer

Tool `experiments/ddm_hg1_ring0_margin_hinge.py`, stage `rederive`. n=96 seeded-random pairs
(seed 20260816, the same seeded selection rn1 used — never a prefix, per [[m88]]/[[m96]]). Same
instrument pins as rt1/rn1, so the rows are leg-to-leg comparable.

The quantity is **not** rn1's. rn1 measured `gap = top1 − top2`. I measure the exact hinge
argument, `m = logit[GT] − max_{c≠GT} logit[c]`: negative exactly at a flip (deficit = −m),
positive at a correct pixel (headroom = m).

| quantity | ddm_hg1 (signed margin) | ddm_rn1 (top1−top2) | agreement |
|---|---:|---:|---|
| flips | **5,448** | 5,448 | exact |
| ring-0 pixels (our tokens) | **408,678** | 408,678 | exact |
| correct on ring-0 | **403,254** | 403,254 | exact |
| median at flips | 0.09976 | 0.09900 | 0.8% (mine is stricter, as it must be) |
| median headroom at correct ring-0 | 0.9703587 | 0.9703590 | 7 digits |
| rho(0.01) | 0.99500 | 0.98515 | 1.0% |
| rho(0.10) | 2.15778 | 2.13784 | 0.93% |
| rho(0.30) | 7.12405 | 7.05834 | 0.93% |

**rho reproduces.** The undirected exchange rate is a fair coin at delta→0 on the exact quantity
too, so rn1's closure of the decode-side family stands on my instrument as well as its own.

### §1.1 The prize, re-priced on the exact quantity

| achieved shift | share of flips recovered | S units (n600 equiv) | × the gap | rn1's figure |
|---|---:|---:|---:|---:|
| +0.01 | 7.342% | 0.002175 | 0.227× | 0.23× |
| +0.03 | 19.438% | 0.005757 | 0.600× | 0.61× |
| **+0.1** | **49.211%** | **0.014575** | **1.519×** | **1.53×** |
| +0.3 | 84.783% | 0.025111 | 2.616× | 2.64× |
| +1.0 | 99.413% | 0.029443 | 3.068× | 3.08× |

GT is the runner-up on **5,340 of 5,448** flips (98.018%). rn1's proxy therefore over-counts by
0.46–0.79 percentage points across the ladder. **The 1.53× headline becomes 1.519×.**

⚠ This is still **DERIVED and an UPPER BOUND**, exactly as rn1 labelled it. It is what the axis
pays *if* the model achieves the shift. Nothing here promises the model can.

⚠ **A distinction rn1's phrasing blurs, and it matters for the launch.** rn1 writes "hinge +0.1
logits". The ladder is indexed by the **achieved margin shift**; the trainer's `--margin-target` is
the hinge's **activation threshold**. They are different objects. Every flip has `m < 0`, so every
flip is active for any positive target — the target does not select which flips get gradient, it
selects how many already-correct pixels get pulled along, and it caps the final margin. How far
down the ladder a run actually gets is a capacity question, and only a launch answers it.

## §2 The ring-0 restriction is a no-op — the measurement that killed my build plan

I was chartered to build a ring-0 support restriction. Before building it I measured what it would
remove. n=96, over all 18,874,368 scored pixels:

| hinge target | frame active | of which on ring-0 | **already ring-0-concentrated** | pixels a mask would remove |
|---|---:|---:|---:|---:|
| 0.05 | 0.0416% | 0.0414% | **99.52%** | 38 |
| **0.10** | **0.0597%** | **0.0594%** | **99.56%** | **50** |
| 0.20 | 0.1186% | 0.1180% | 99.47% | 119 |
| 0.30 | 0.2045% | 0.2031% | 99.30% | 269 |
| 0.50 | 0.4222% | 0.4163% | 98.61% | 1,107 |
| 1.00 | 1.2295% | 1.1528% | 93.77% | 14,462 |

At the operative target a ring-0 mask removes **50 pixels of 11,281 active — 0.44%**.

**Why, and this is the general statement.** The margin field is small only where the decision is
close, and the decision is close only at the boundary. `relu(target − m)` is therefore *already* a
soft ring-0 selector, and it becomes a sharper one as the target falls. A support mask is redundant
with the loss form. rn1's ring-0 language is a correct description of where the error lives; it is
not an instruction to add a mask.

**A mask is not free, either.** It would need the GT boundary at train time and would add a
second definition of "the ring" beside the one the loss already implies. Two definitions of the
same support is exactly the drift the repo's config-orphan discipline exists to prevent.

## §3 The hinge already exists — read at source

`experiments/train_witness_realized_through_R_mlx.py::make_loss_fn`:

```
elif form == "margin_hinge":
    signed = _live_signed()                                   # logit[GT] - max_{c!=GT} logit
    hinge_map = mx.maximum(margin_target - signed, 0.0)
    if apply_mw:
        hinge_map = hinge_map * _live_margin_weight(...)
    seg_l = mx.mean(hinge_map if seg_pixel_w is None else hinge_map * seg_pixel_w)
```

`_live_signed()` is `sum(logits * onehot) − max(logits + onehot·(−1e9))` — the exact signed margin
I re-derived in §1. So the trainer already computes rn1's quantity and already hinges on it. It is
reachable as `--seg-form-start margin_hinge`, and `margin_hinge` is in the trainer's
`MARGIN_WEIGHTED_HONORING_SEG_FORMS`, so the annulus reweight composes without tripping
`assert_margin_weighted_loss_is_honored`.

**STRUCTURAL LIMIT, and a launch must plan around it.** `reachable_seg_forms` gives an outgoing
transition to `ce` alone (ce → tau_softplus at the knee). Every other start is terminal. So
`margin_hinge` **cannot be scheduled as a finishing stage** — a run occupies it from epoch 0 and
has no CE trunk. That forces the A/B to be fresh-start on both arms and judged at the seg
asymptote, never against the warm incumbent (wd3: the fresh-vs-warm floor is 2.5×).

## §4 The target is the real defect — 97.65% of the gradient is wasted

At `--margin-target 1.0`, 232,000 of the frame's pixels are active and 5,448 of them are flips.
**97.65% of the hinge's gradient pulls pixels that are already correct.** At 0.1 that falls to
51.7%; at 0.05 to 30.6%.

The principled target is not "smaller" — it is the floor below which a corrected pixel cannot
survive the round trip the scorer applies. That floor is measured and registered:

| constant | value | provenance |
|---|---:|---|
| `delta_R` | **0.019590163230895963** | MEASURED p95 of \|uint8-induced margin perturbation\| over the annulus, `reports/delta_R_noise_floor.json` |
| headroom | 2.0 | the multiplier the sister `MarginBandSatisficing` lever already uses on the same annulus |
| **`m_safe`** | **0.039180326461791926** | `margin_band_satisficing_threshold_v1`, resolved LIVE |

The trainer default is **25.5×** that. Parking a corrected pixel below `delta_R` leaves it inside
the noise the uint8 round trip injects, so it can flip back; parking it at 1.0 spends capacity
buying safety no pixel needs. `m_safe` is the smallest target at which a fix is robust.

**Two forces on one annulus must not disagree about "safe".** The lever reuses the sister lever's
law and headroom rather than introducing a second threshold. A test enforces the agreement.

### §4.1 What this axis is worth, relative to the gap (MAIN, appended 2026-08-16)

The §4 argument above is a two-sided THRESHOLD DERIVATION, not a magnitude dismissal: below
`delta_R` a correction sits inside a MEASURED perturbation (p95 over the annulus, receipt
`reports/delta_R_noise_floor.json`) and can flip back; at 1.0 the target buys safety no pixel
needs. Both sides are bounded by measurement, not by eyeball. The `#404` detector fired on it
anyway, and the demand it makes is worth paying, because this memo never stated the one number
that justifies spending an 8-hour Metal slot on the arm:

| quantity | value |
|---|---:|
| S at the current operating point | 0.15959729295498598 |
| remaining gap to 0.15 | **0.0095973** |
| seg term (100·d_seg) | **0.029611** |
| seg term ÷ remaining gap | **3.085×** |
| seg recovery that closes sub-0.15 on this axis ALONE | **32.41%** |

So the seg axis is worth three times the entire remaining gap, and roughly a third of it would
finish the job unaided. That is why the arm gets the slot. Note the pre-registered primary
falsifier (25% of the re-derived ladder) sits deliberately BELOW the 32.41% gap-closing
threshold — the arm can fail its own bar and the measurement still buys a real bound on the
family. Relative significance is the operative test near a goal; absolute smallness is not
(memory: relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS).

verdict_scope: INSTANCE — this is one arm's justification arithmetic at one operating point,
not a family claim. It moves when the pointer moves.

## §5 Q3 — built, composable, and NOT a default

`--seg-grad-q3-project` already projects the seg gradient entering rendered frame_1 blockwise onto
the frame_1 yuv6-null subspace (sq1's exact `P = I − pinv(A)·A`, rank 6 of 12 per 2×2 block, Q3
dim 6 × 49,152 = 294,912). Forward pixels unchanged; the JD1 pose path uses the unwrapped loss.

I did not make it a hinge default, for three measured reasons:

1. **The exact-kernel claim is pre-quantization only.** `A·δ = 0` ⟹ the pose input is bit-identical
   (measured 5.684e-14 at the scorer input against 4.855 for a same-norm generic control). But #532
   measured uint8 rounding breaking it at **62.74 vs 1.7e-13**.
2. **The one build that exists did not clear.** `ddm_q31`, n32: verdict
   `Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE`, target survival 0.2304 against a 0.6964 bar,
   d_pose mean ratio **1.0424 — not flat**, 32/32 rows cap-bound.
3. **Composing it into the treatment changes two things at once.** The A/B would no longer isolate
   the hinge.

The charter's framing — "a Q3-CONSTRAINED variant CANNOT create pose damage (exact kernel)" — is
bo1's sharpened restatement, which **bo1 itself labels UNBUILT/UNMEASURED** at L528. The measured
#889 receipts (uv1 n=4, cr2r n=74) are about *unconstrained* placement. Q3 is sealed as a separate
arm with a declared fire condition.

## §6 The withdrawn bug — recorded because I nearly filed it

`src/tac/boundary_math/levelset_micro_batch_loss.py:297` applies only the focal weight in the
`margin_hinge` branch and never the margin weight, while the trainer's guard lists `margin_hinge`
as honoring `--margin-weighted-loss`. That reads as a silent-inert confound. **It is not live
here:** tr1 imports the *serial* `make_loss_fn`, whose `margin_hinge` branch does apply
`apply_mw`. The divergence is latent in the levelset micro-batch twin only. Anyone routing
`margin_hinge` + `--margin-weighted-loss on` through that twin should check it first.

## §7 What landed

- `experiments/ddm_hg1_ring0_margin_hinge.py` — stages `rederive` (signed-margin ladder, rho,
  per-class shares, the runner-up discrepancy) and `binding` (the inert / global-push falsifier).
- `src/tac/witness_dsl/hg1_ring0_margin_hinge_levers_20260816.py` — `lever_hg1_ring0_margin_hinge`
  (target resolved LIVE from the measured artifact, with a LawRef) and
  `lever_hg1_q3_constrained_seg_grad`. Per-arm module, matching the tk1/bi1/pt2 precedent;
  `curriculum_dsl.py` is held by another agent and I did not touch it.
- `src/tac/witness_dsl/tests/test_hg1_ring0_margin_hinge_levers.py` — 15 tests. The two
  load-bearing ones check no flag is invented (against the trainer's own argparse) and that the
  target **moves when the artifact moves** (so it is derived, not a literal wearing a law's name).
- `.omx/research/configs/ddm_hg1_ring0_hinge_sealed_ab_20260816.json` — the 3-arm design record,
  `sealed_sha256 f7bbdf0f6b4bba64…`. **SUPERSEDED for firing** (see below). **NOT LAUNCHED.**
- `experiments/ddm_hg1_seal_tr1_ab_tickets.py` + the two launcher-loadable tickets
  `.omx/research/configs/ddm_hg1_tr1_ticket_arm_{a_control_ce,b_hinge}_20260816.json` — the
  fireable seal. **Both DRY-RUN OK on every gate.** **NOT LAUNCHED.**

### §7.1 The re-seal (MAIN blocker, closed)

The first seal used a private `ddm_hg1_tr1_sealed_ab.v1` schema. It is correct in content and
**unfireable**: `tools/launch_tr1_run.py` refuses any schema but `ddm_tb1_tr1_sealed_ticket.v1`.
Re-emitted as one ticket per arm. Two things the re-seal forced, both improvements:

1. **The argv is COMPILED, not hand-assembled.** The launcher's G1 gate recompiles argv from the
   ticket's own levers through `TR1RendererProgramV1.compile_trainer_argv()` and refuses on drift.
   A hand-written argv that merely looks right is exactly what G1 catches.
2. **The lv1 base bundled the seg FORM with the seg TRUNK WEIGHTS in one `tr1_seg_ce` lever.**
   Swapping that bundle wholesale would have silently dropped `--class-weight-lane` and `--w-seg`
   from arm B while the arms still *looked* matched. The bundle is split into
   `hg1_seg_trunk_weights` (identical on both arms) and a per-arm form lever.

Measured arm delta — the ONLY four flags that differ:

| flag | arm A | arm B |
|---|---|---|
| `--seg-form-start` | `ce` | `margin_hinge` |
| `--margin-target` | `1.0` | `0.039180326461791926` |
| `--margin-weighted-loss` | *(absent)* | `on` |
| `--out-dir` | `…/arm_a_control_ce` | `…/arm_b_hinge` |

Everything else is byte-identical, including the resumability P0 (`--basin-handoff on`,
`--max-wall-minutes 480.0`, `--epochs 400`, `--seed 0`). `scope_laws` is empty: every registered
scope law is a jd1/jd3 pose-retreat policy and none governs a seg-form A/B, so declaring one to
make a gate run would be inventing scope. `ticket_hash` is still emitted truthfully and
recomputes under the canonical `ticket_payload_hash` on both tickets.

**SERIAL, not concurrent.** The launcher's G4 gate admits ONE n600 job at a time. Memory is not
the binding constraint (87.7 GiB free against a 25.6 GiB floor) — the scorer slot is.

## §8 The seal — 3 arms, declared fire order

Base argv inherited from the sealed lv1 tr1 ticket, so the arms differ from a proven config by the
seg form alone. Arms write to distinct out-dirs on APDataStore.

| arm | delta from control | fires |
|---|---|---|
| **A control** | — (`--seg-form-start ce`, `--margin-target 1.0`) | with B |
| **B hinge** | `--seg-form-start margin_hinge`, `--margin-target 0.03918`, `--margin-weighted-loss on` | with A |
| **C hinge+Q3** | B plus `--seg-grad-q3-project on` | ONLY if B shows a seg gain AND a measured pose cost |

**Pre-registered falsifier.** Realized seg recovery below **25%** of the re-derived ladder (half of
the 49.2% at +0.1) at the seg asymptote ⟹ the DERIVED upper bound is not reachable by this term;
report the routing honestly and do **not** re-tune into a result. Plus: hinge active fraction ≈ 0 ⟹
INERT and the run is confounded (expected support at the sealed target is 1.91–2.74% of ring-0);
active fraction approaching the whole frame ⟹ a global margin push that will fight rate.

**Pose is the binding constraint, not seg.** rn1 measured the pose marginal at **6.03×** seg's at
hv1's d_pose 6.88e-06. A hinge that recovers 1.519× of the gap on seg and spends a little pose
legibility can still lose. Arm B owes a measured pose leg — and it may **not** be quoted from the
advisory instrument, which rn1 measured **18.2× optimistic on pose** (the seg half is sound at
2.5%). Coordinate with `pi2` before quoting any pose coefficient.

## §9 What this unit did NOT establish

- **No score.** Every number is `[macOS-CPU advisory]`. The pointer is unmoved.
- **The hinge was not run.** Binding is proven *analytically* on the retained margin field (the
  support is 1.91–2.74% of ring-0 at the sealed target, neither 0 nor 1). Whether the term is
  binding *in a live optimizer* — its gradient share at a stage boundary, and whether it moves
  ring-0 margins in the signed direction on real training frames — is exactly what arm B measures
  and I did not launch it. **The charter's step 3 is therefore PARTIALLY discharged**, and I am
  labelling it rather than claiming it.
- **The 34.9× label-vs-axis ratio is measured on 6 seeded pairs plus the shipped selection
  telemetry**, not on all 600. The sign is not a close call (1–3 mismatched pixels per frame
  against ~57 flips) but the coefficient is a small-n estimate.
- **I did not measure whether the renderer has the capacity** to move ring-0 margins by +0.1. That
  is the whole capacity question and the reason the ladder is an upper bound.
- **rho and the ladder are n=96 seeded-random, not n600.** Per [[m96]] a seeded subset may refute a
  bar but may not license a LIVE verdict.
- **The Q3 exact-kernel arithmetic is inherited, not re-derived here.** I read the receipts and
  scoped the claim; I did not rebuild the projector or re-measure the 5.684e-14.
- **DSL discovery gap, reported not fixed.** `lever_factories()` scans `curriculum_dsl.py` only, so
  no per-arm module (tk1, bi1, pt2, mine) is discovered and `completeness()` still reports these
  flags UNMAPPED. That is pre-existing and affects every per-arm module. Fixing it means editing
  `lever_registry.py` or the hot `curriculum_dsl.py` — owner needed, not mine to take unilaterally.
- **A pre-existing failure I did not cause:** `test_lever_registry.py::test_332_coverage_rose_from_
  deorphaning` fails identically with my module removed (verified by removal). Not mine, not fixed.

## §10 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_hg1_ring0_margin_hinge_20260816/` (APDataStore 240 GiB free;
VertigoDataTier is at 893 MiB and was not targeted).

| artifact | bytes | sha256 | what it is |
|---|---:|---|---|
| `HG1_MARGIN_FIELD_n96.npy` | 75,497,600 | `f99100e1dd15773ad20e6a9b2c5093df5fe82fa3ba0a25ff1ed463fd988d26f5` | the per-pair SIGNED margin field, float32 (96,384,512) — the payload, not its length |
| `HG1_REDERIVE_n96.json` | — | — | the ladder, rho, per-class shares, runner-up census |
| `HG1_BINDING_n96.json` | — | — | active-fraction rows per target (the inert / global-push falsifier) |
| `HG1_REDERIVE_smoke4.json` + `HG1_MARGIN_FIELD_smoke4.npy` | 3,145,856 | `d5624d4d…` | the n=4 smoke, retained |
| `rederive_n96.log` | — | — | per-pair run log |

The binding stage consumes the margin field **by sha256**, never by a remembered number. Consumed
unmodified: the wc1 retained decode `0.raw` (3,662,409,600 B), the hv1 ep0634
`decoded_spatial_tokens.rc64.bin` (117,964,800 B), the qs3 `gt_argmax_n600.npy`.

## NEXT_IF_RESUMED

| # | work | owner | fire condition |
|---|---|---|---|
| 1 | Fire arm A (`ddm_hg1_tr1_ticket_arm_a_control_ce_20260816.json`), then arm B (`…arm_b_hinge…`) — SERIAL, G4 admits one n600 job. Both DRY-RUN OK. Judge at the seg asymptote against the pre-registered 25% falsifier | **MAIN** (Metal slot) | NOW — re-seal complete, both dry-runs pass, Modal untouched |
| 2 | Arm C (hinge + Q3) | MAIN | ONLY if arm B shows a seg gain AND a measured pose cost |
| 3 | Measured pose leg for arm B, on a non-advisory instrument | `pi2` + hg1 | arm B completes; do not quote advisory pose magnitudes (18.2× optimistic) |
| 4 | Own the DSL discovery gap: per-arm lever modules are invisible to `lever_factories()` | unowned — needs a registry owner | after `curriculum_dsl.py` is released by its current holder |
| 5 | Latent twin divergence: `levelset_micro_batch_loss.py:297` drops the margin weight for `margin_hinge` | unowned | before anyone routes `margin_hinge` through the micro-batch twin |
| 6 | The renderer-capacity question the ladder cannot answer: can the render move ring-0 margins by +0.1 at all? | successor of arm B | arm B lands below the falsifier |

**Own-vehicle frontier: S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] — UNMOVED.**
