---
arm: ddm_ps2
title: "The n-scaling my charter ordered was already on disk, unconsumed, and it inverts the charter's premise. fo2h measured the projected channel at n=48 seven hours before my charter was written: eta CLEARS fo1's bar (0.5804 > 0.5196, as the charter predicted) but the pose leg INVERTS -- pn2's x0.7935 was a small-sample artifact, and the ratio stabilises at ~1.37 for n>=20. Composing the three terms fo2h measured but never added: seg+rate -0.000336, pose +0.001424, NET +0.001088 -- the channel is a LOSS that moves 13.3% of the gap the wrong way. na9's 0.010423 S 'pose cost removed' is CONFIRMED as a removal (I measure 0.011501) and WRONG as a credit: it is priced against the unprojected arm, and the shipping baseline is no-edit. The family's ceiling before pose is +0.00037 S = 4.5% of the gap, so F2's headline overstates the channel's ceiling ~24x."
utc: 2026-08-18
axis: "[macOS-CPU advisory] -- arithmetic over fo2h retained per-pair rows; NO scorer was run; NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "fx1 S 0.15816036933414834 @ 180,601 B [contest-CUDA T4 n600], archive sha 65c75d7f... -- UNMOVED by this unit"
verdict_scope_default: "FORMULATION on post-hoc pose-null-projected seg-correction overlays on the frozen artifact; stated inline per row"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ps2 — the pose axis: F2's projection lever, adjudicated on the axis that scores it

**Operator directive:** *"Attack all and pose and all 1111."* Pose contributes 0.008295 S = 102% of
the remaining gap on the fx1 frontier. na9's F2 ranked the projection lever as the head of the pose
queue. This arm was chartered to buy the n-scaling that decides it.

## RESULT FIRST — the n-scaling was already bought, and it inverts the charter

**My charter's premise was one arm stale.** It ordered *"extend pn2's matched A/B from n=12 to
n∈{24,48}"*. `fo2h` ran exactly that experiment and wrote
`FO2H_ETA_ADJUDICATION.json` at **2026-08-17T17:39:43Z** — n=48, seeded-random, disjoint from pn2's
12 by construction. My charter was written at 00:50Z the next day, **7h11m later**, and cites pn2's
n=12 figures as live. The RECALL FIRST clause is what caught it; nothing was rebuilt.

**Zero documents had consumed that result.** A repo-wide sweep for `0.5804`, `1.3725`,
`CONFIRMED-HARDENED`, `0.0014235` returns hits only in fo2h's own producing script. Meanwhile the
superseded n=12 figure propagated into **four documents written after the result existed**:
`ddm_tc1_tr1_lifecycle_spec` (19:11Z), `ddm_na9` (00:28Z), the **DAG** FEED-2026-08-17d (00:45Z),
and **my own charter** (00:50Z). fo2h's own memo still reads `status: IN FLIGHT … LEG 1's n=48
solve is still filling` and its LEG 3 table still carries the n=12 `×0.793`.

**Three findings, in the order that matters.**

**1. The charter's falsifier did not fire, and it was aimed at the wrong term.** Pre-registered:
*"FALSIFIER: η(n=48) < 0.52."* Measured η(n=48) = **0.5804** against the bar **0.5196** — it
clears, and the central estimate the charter predicted (η ~0.55) was close. **The channel fails
anyway**, on a term the falsifier does not contain.

**2. The pose leg inverts with n, and pn2's ×0.7935 is a small-sample artifact.** Cumulative pose
ratio on a seeded shuffle of fo2h's 48 pairs:

| n | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 36 | 44 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled η | 0.5217 | 0.5583 | 0.5657 | 0.5679 | 0.5671 | 0.5773 | 0.5800 | 0.5711 | 0.5817 | **0.5804** |
| pose ratio | 0.7078 | 1.0329 | 1.0136 | 1.1042 | 1.3893 | 1.4188 | 1.3915 | 1.3559 | 1.3188 | **1.3725** |
| ΔS_pose | −0.001317 | +0.000135 | +0.000056 | +0.000422 | **+0.001483** | +0.001586 | +0.001490 | +0.001364 | +0.001231 | **+0.001424** |

η is flat in n from n≈8. **The pose ratio starts below 1 (a credit), crosses 1.0 by n=8, and
stabilises at 1.32–1.43 from n≥20.** pn2 read x0.7935 at n=12 on its own 12 pairs; this is the
same regime, and it does not survive. Only **13 of 48** pairs improve on pose.

**3. The three terms, composed — which no document has done.** fo2h's verdict arithmetic is
`net_dS(eta, flips, total_B) = -eta*flips*SEG_DS_PER_FLIP + total_B*RATE_DS_PER_BYTE` — seg gain
plus rate cost, **and nothing else**. It measured the pose leg and reported it *beside* the verdict.
The channel edits pixels PoseNet reads, so the total is `ΔS_seg + ΔS_rate + ΔS_pose`:

| term | ΔS at n=48 |
|---|---:|
| seg gain at η=0.5804 over 6,512 described flips | **−0.003204** |
| rate, fo1's measured 4,308 B | **+0.002869** |
| seg+rate (fo2h's published verdict number, re-derived) | **−0.000336** |
| **pose** | **+0.001424** |
| **JOINT** | **+0.001088** |

**The channel is a net LOSS of +0.001088 S — it moves 13.3% of the fx1 gap the wrong way.** The
pose cost is **4.24×** the seg+rate gain.

## The framing error, named precisely: the baseline

na9 F2 states: *"unprojected d_pose ×4.6089 vs projected ×0.7935 … 0.010423 S of pose cost removed,
1.09× the entire remaining gap."* **The removal is real and I confirm it at larger n.** fo2h's
matched A/B (n=16, same pairs, same solver, only the projection differs), with both arms priced
against the **shipping** baseline — no edit, pose cost exactly zero:

| arm | pose ratio | ΔS_pose vs shipping |
|---|---:|---:|
| unprojected | 6.9563 | **+0.013588** |
| projected | 1.5663 | **+0.002087** |
| **removed by the projection** | | **0.011501** |

**0.011501 S removed — na9's 0.010423 was right, and the mechanism is confirmed at n=16, not
n=12.** What is wrong is the sign of what remains. The projection is measured against the
*unprojected edit*; the shipping baseline is *no edit*. Against the baseline that actually ships,
the projected arm is a pose **cost** of +0.002087, not a credit. This is
`a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803` on the pose axis: a large,
real, correctly-measured removal of a cost the treatment itself created.

**The mechanism claim survives; the magnitude claim inverts.** A repair that says only *"the
projection still helps"* reproduces the same error one level up.

## The spread fo2h applied to η and never to pose

fo2h hardened η with 20,000 bootstrap resamples, a two-shard σ, and a refuse-on-straddle rule, then
declared SUPPLIER CONFIRMED-HARDENED. It applied **no spread estimator to the pose leg**, which is
the binding term once composed. Same protocol, same estimator, applied to the term that binds
(seed 20260818, n=20,000, pair-level resample of the ratio-of-means):

| | pose ratio |
|---|---:|
| p2.5 / p16 / **p50** / p84 / p97.5 | 1.0523 / 1.1888 / **1.3701** / 1.6122 / 1.9139 |
| sd | 0.2213 |
| **break-even ratio** (where the joint hits zero) | **1.0825** |
| measured / break-even | **1.268×** |
| **fraction of draws that net-supply** | **4.23%** |

The joint ΔS distribution is `p2.5 = −0.000121, p50 = +0.001079, p97.5 = +0.002846`. **95.8% of
the bootstrap mass is a loss.**

## Is there a survivor? Three tests, and an honest non-closure

Closing here would be a premature KILL: the channel is selected by a **seg-only** objective (sr1's
waterfill ranks cells by seg value per byte) and nothing had asked what it does under the objective
that scores.

**(a) Oracle gate — an upper bound, not a candidate.** The encoder owns a scorer, so it may measure
each pair's joint effect and ship the winning subset, paying the subset index as an exact
combinatorial rank over the clip's 600 pairs. Best over all 49 prefixes: **k=38, ΔS_joint =
+0.000221 — still a loss.** Gating works on pose (ratio → 0.9833, a small credit) but η collapses
to **0.4737, below fo1's own break-even 0.5196**, while the 4,308 B payload is still paid. The
index costs 54.8 B and is not the binding term — the verdict is unchanged if the index were free.

**(b) Feature gate — the only shippable kind — does not generalise.** LOO over nine edit-side
features computable without evaluating PoseNet (`support_px`, `snap_tax`, `n_described_ring0`,
`flips_before`, the four `yuv6_shift` magnitudes, `d_pose_before`). **Every feature scores LOO
0.7708 against a majority baseline of 0.7917 — all nine are worse than always predicting "help".**
Best |correlation| with pose excess is 0.246 (`d_pose_before`). This is the `pk3` genus exactly
(23/23 in-sample → 0/23 LOO), which is why no in-sample number is computed anywhere in the module.

**(c) The term-wise lower bound does NOT close the family, and I will not claim it does.** Bounding
each term by its own unconstrained optimum over all 2^48 subsets (seg: every flip-reducing pair;
pose: every pose-reducing pair; rate: floor at 4,308 B) gives **−0.001109**. It is negative, so it
proves nothing — seg and pose disagree about which pairs to keep, and the bound is attained by no
single subset. **Honest status: the sweep found no survivor; the bound does not forbid one.**

## Where a survivor could still live — the missing measurement, named

The refutation is at **one operating point**: fo1's 41-cell / 4,308 B selection, whose seg gain
exceeds its rate cost by only ~10%. fo2h LEG 2 measured real coder bytes for all 74 live prefixes.
Pricing each at the hardened η gives the pose ratio each point could absorb and still supply:

| cells | flips | bytes | seg+rate margin | **tolerable pose ratio** |
|---:|---:|---:|---:|---:|
| 9 | 175 | 86.5 | +0.000029 | 1.0069 |
| 28 | 2,120 | 1,163.3 | +0.000269 | 1.0658 |
| **34** | **4,946** | **3,099.7** | **+0.000370** | **1.0911** |
| 41 (incumbent) | 6,512 | 4,317.6 | +0.000329 | 1.0809 |
| 50 | 12,449 | 8,952.9 | +0.000164 | 1.0399 |
| 74 | 34,666 | 33,496.9 | −0.005247 | 0.1352 |

**The most tolerant point in the entire measured family absorbs a 1.0911 pose ratio. The one pose
measurement we have is 1.3725 — 25.8% over the family's most tolerant point.** Pose is measured
only at the **full ring-0 edit**; m=34 edits fewer cells and should cost less pose. That is the
missing number, and it is now specified rather than assumed.

**The prioritisation fact this yields, which is bigger than the verdict.** The family's seg+rate
margin never exceeds **+0.00037 S anywhere** = **4.5% of the fx1 gap**. na9 ranked F2 as worth
*"1.09× the entire remaining gap."* **The ceiling is ~24× smaller than the headline, and the
measured value is negative.** F2 is not the head of the pose queue.

## PRIOR-LAW PREDICTION — VERDICT

| charter prediction | verdict | measured |
|---|---|---|
| advantage shrinks but η stays above break-even at n=48 (η ~0.55 vs 0.5196) | **HIT** | η = 0.5804, clears; the matched advantage plateaus at +4.87% rather than decaying to zero |
| a bounded GO worth −1e-3..−4e-3 S | **REFUTED, sign inverted** | joint **+0.001088**, a loss |
| FALSIFIER: η(n=48) < 0.52 → close at FORMULATION | **did not fire — and was mis-specified** | the binding term is pose, which the falsifier does not contain |

**What the miss teaches.** I pre-registered a falsifier on the term the *prior arm* had measured,
not on the term that *governs the score*. fo1's break-even η is a two-term bar (seg vs rate) and I
inherited it as if it were the verdict. A falsifier that cannot see the term carrying 424% of the
margin is a gauge that cannot fail in the direction the truth lies — the VACUITY==PASS genus at the
falsifier surface. **My own control reproduced the same disease inside this arm**: the first draft
listed `net_dS_n48` as re-derived while asserting nothing, and that vacuous control hid a wrong
flip-count constant (I used the 48-pair sample's flips instead of the channel's clip-wide 6,512,
inflating the seg gain 4.5×). The cure is landed in the module: every published figure is bound to
a re-derivation in one dict, and a coverage gap between the two sets is fail-closed.

## VERDICT

**F2 REFUTED-ON-THE-JOINT-AXIS. Scope: FORMULATION** — post-hoc pose-null-projected seg-correction
overlays on the frozen hv1/rr4-lineage artifact, ring-0 described set, r=1 support, this solver
budget, n=48 out-of-sample, at the 41-cell operating point.

**NOT FAMILY, and the distinction is load-bearing:** the projection **mechanism** is confirmed and
large (removes 0.011501 S of pose cost, 4.44× reduction in pose ratio); pose at the m=34 operating
point is **unmeasured**; and the term-wise bound does not forbid a survivor. Per the
forbidden-premature-KILL rule this is **DEFERRED-pending-the-m=34-pose-measurement**, not killed.

**No T4 fire-order is sealed. There is nothing to fire** — the measured candidate is a loss, and
sealing an order for it would be the fake this arm exists to prevent. Per m96, a seeded-random
sample may REFUTE a bar; clearing one does not license a LIVE n600 verdict.

## Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_ps2/`, sha256 manifest in `RECEIPT.md`. Both adjudication JSONs
retain the **full per-point sweeps** (49 oracle-gate prefixes, 74 waterfill operating points, 48
cumulative-curve points, all nine LOO feature rows), not only the winners. No new payload was
materialised beyond these — this arm ran no solver and no scorer, and I say so rather than leaving
the law satisfied silently.

## STORES CONSULTED

* **The result that supersedes my charter:** `/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/`
  — `FO2H_ETA_ADJUDICATION.json` (read in full: eta, spread, pose_leg, matched_AB_new_sample,
  sampling, verdict), `FO2H_WATERFILL_MEASURED.json` (74 rows), `RECEIPT.md`, and the retained
  `null_shardA` / `null_shardB` / `free_matched16` `ETA_GATE_ROWS.jsonl` (the 48+16 per-pair rows
  every number here is re-derived from).
* **The arithmetic, read at source not inferred:** `experiments/ddm_fo2h_eta_adjudicate.py` — the
  module docstring's pre-registered adjudication, `net_dS` (:199, confirming pose is absent from
  the verdict), `pose_agg_ratio` (:109), `pose_concentration` (:118), and the frozen pins block
  (:62-72, source of `FO1_BREAKEVEN_ETA`, `FO1_TOTAL_B`, `D_POSE_N600`).
* **The framing this corrects:** `ddm_na9_gestalt_negative_audit_20260818.md` §F2 (the 0.010423
  claim, the "1.09× the entire remaining gap" ranking, and its own honest limit #4 *"F2 rests on
  n=12 seeded-random η whose advantage regressed monotonically with n"*) ·
  `ddm_pn2_posenull_seg_channel_20260817.md` (the matched A/B, the regression series 15.6 → 8.1,
  the refusal to quote a level) · `ddm_fo1_waterfill_real_coder_20260817.md` (the 4,308 B and the
  0.5196 break-even, both frozen) · `ddm_fo2h_eta_hardening_20260817.md` (LEG 2 and LEG 3; its
  frontmatter still declares LEG 1 in flight).
* **Propagation sweep (this unit):** `.omx/research/*.md`, `.omx/research/charters/*.md`, the
  `sub015_DAG` (grep `fo2h` → 0 hits; last pose figure at FEED-2026-08-17d is ×0.7935),
  `.omx/state/main_hot_state.md` (no mention of the channel in either direction; it does carry
  *"pose term ≈ 0.00829 ≈ 102% of the remaining gap — pose is THE axis"*), and `git log` since
  08-17 (26 commits after the result, none referencing fo2h).
* **Graveyard, honoured not re-run:** `ddm_pk3_frame0_pose_representation_20260813.md` +
  `ddm_pk4_optimal_form_frame0_pose_20260813.md` (the LOPO trap — the reason §(b) reports LOO only)
  · `ddm_qs5_resolve_compensation_20260813.md` (in-compile compensation) ·
  `ddm_t1h_pose_coeff_resolve_headroom_20260817.md` §11 (the axis-oracle law — why no CPU-axis
  acceptance is claimed here) · `ddm_rt2_manufactured_seg_mechanism_20260817.md` (the 6/12-DOF null
  space the projection uses).
* **Governing:** `CLAUDE.md` (NO-FAKE; THE GOAL; ALWAYS KEEP THE PAYLOAD; verdict-scope ladder;
  forbidden-premature-KILL; own-vehicle end-of-turn line) ·
  `docs/operating_manual_craft_handoff.md` · `.omx/research/charters/ddm_ps2_…_20260818.md` ·
  memories [[m44]] (never recall from working memory alone — the clause that caught the stale
  premise), [[m88]]/[[m96]] (prefix genus; fo2h's sampling law inherited), [[m90]] (the floor you
  divide by), `a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803`,
  `measured_object_vs_named_object_20260816`,
  `charter_recall_validation_is_apparatus_not_volition_20260816`.

## NEXT_IF_RESUMED — every row exits owned

This arm may not fire, dispatch, or run a scorer, so no row exits FIRED.

1. **QUEUED-WITH-A-FIRE-ORDER — owner MAIN, $0, `.md` only.** Correct the four documents carrying
   the superseded n=12 pose figure, **headline and body** (`corrections land in bodies, headlines
   keep the stale number`): `ddm_na9` §F2, the **DAG** FEED-2026-08-17d, `ddm_tc1_tr1_lifecycle_spec`
   :210 + :353, and `charters/ddm_ps2_…` :5-7. The correct sentence is *"the projection removes
   0.011501 S of pose cost it created; the projected arm remains a +0.001424 S pose cost against
   the shipping baseline, and the channel nets +0.001088 S."* **Fire trigger: before any arm
   inherits F2 as the ranked pose head** — my own charter is the proof that this already happened
   once.
2. **QUEUED-WITH-A-FIRE-ORDER — owner MAIN or an fo2h successor, $0, `.md`.** fo2h's memo
   frontmatter still says `status: IN FLIGHT … LEG 1's n=48 solve is still filling`. LEG 1
   completed at 17:39:43Z with a verdict. **Fire trigger: next edit to that memo.** A finished
   result behind an in-flight banner is invisible in exactly the way this arm just measured.
3. **QUEUED — owner: a pose-axis successor. The one measurement that could reopen F2.** Pose ratio
   at the **m=34** operating point (3,099.7 B, 4,946 flips), whose tolerable ratio is 1.0911 — the
   family maximum. Bounded CPU-torch, n≤48, on fo2h's existing 48-pair seeded-random sample so the
   comparison is matched. **Fire trigger: only if the pose queue has nothing above 4.5% of the gap**
   — because 4.5% is this family's ceiling even on a total success, and that is the number that
   should decide whether anyone spends the compute.
4. **FOLDED — the feature-gated channel.** Nine edit-side features, all LOO-worse than the majority
   baseline. No successor owed unless a genuinely new feature class appears.
5. **FOLDED — the seg-only selection objective.** The waterfill ranks cells by seg value per byte
   on a channel that also moves pose. Any re-selection must optimise the joint objective; the
   pose-budget table above is the input it needs. Superseded by row 3, not a separate queue.

**Own-vehicle frontier: fx1 `S 0.15816036933414834 @ 180,601 B [contest-CUDA T4 n600]`, archive sha
`65c75d7f…` — UNMOVED by this unit.** This arm fired nothing, spent $0, ran no scorer, and did not
lower the score. It found the experiment its own charter ordered already complete and unconsumed on
disk, composed the third term that decides it, and converted a channel the corpus ranked at 1.09× the
gap into a measured net loss with a 4.5% ceiling.
