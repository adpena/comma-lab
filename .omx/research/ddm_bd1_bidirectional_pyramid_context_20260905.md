# ddm_bd1 — bidirectional (B-pyramid) temporal context for the label-field coder: CLOSED at family scope

Tokens: `[no-triality] [p0-ledger-ok]` · Arm: ddm_bd1 (Opus) · 2026-09-05 · lane
`lane_ddm_bd1_bidirectional_pyramid_context_20260905` · axis of every number below:
`[exact local bit/byte arithmetic, scorer-free]` · `score_claim=false` · no scorer, no Modal, no Metal, no training.

## VERDICT

**Falsifier F1 fired. The door is CLOSED at family scope.** The counting-model screen prices every realizable
B-pyramid over the frontier archive's label field at a **2.84–5.68 % net saving** on the 113,419 B RC64 token
stream. F1's bar was 8 %. No training was launched; Metal was never requested.

The closure is stronger than "the charter's GOP=8 layout misses." I priced the **whole family**:

| reading | net saving on the 113,419 B stream |
|---|---|
| charter's GOP = 8 | 4.06 % / 4.47 % / 5.30 % = 4,608 / 5,069 / 6,015 B |
| best of GOP ∈ {2,4,8,16,32} (all = GOP 32) | 4.46 % / 4.85 % / 5.68 % = 5,053 / 5,504 / 6,442 B |
| conservative bracket (per-level fit, KT) at GOP 8 | 0.48 % / 2.97 % / 2.77 % = 542 / 3,371 / 3,139 B |
| **UNATTAINABLE ceiling** — every pair coded bidirectionally at d = 1 | **7.78 % / 7.46 % / 9.03 % = 8,828 / 8,463 / 10,242 B** |

Three numbers per row = the three independent context ladders (α, β, γ). The ceiling is unattainable by
construction — no decode order can give every pair both neighbours, someone must be coded first — and it is
the *supremum* of the family. **F1's 8 % bar bites the supremum itself**, not merely the charter's layout.
GOP 16 → 32 gains only +0.12 pp per doubling and the increments are shrinking geometrically, so the family is
**saturated**, not merely unswept.

Byte accounting against F2's bar (≥ 5,000 B), for the record even though F1 already closes:
`conv_future` costs a DERIVED **≈ 1,034 B** (2,880 weights + 64 bias at cl2's MEASURED 2.810 bits/param
average over 38,341 params → 13,466 B), plus ≈ 11 B for the 4-way level embedding. Net = gross − ≈ 1,045 B:
best case 5,397 B (marginal), central case 4,008 / 4,459 B (**fails F2**), conservative bracket −503 … 2,326 B.
Two independent falsifiers agree.

## THE MECHANISM (the finding worth keeping)

**The past plane and the future plane are near-redundant on this field, because the label field's temporal
autocorrelation decays far more slowly than natural video's.** One MEASURED number carries it: the P-only cost
at distance 32 is only **1.069–1.129×** the P-only cost at distance 1. Going *thirty-two pairs back* costs
7–13 % more than going one pair back.

That single fact kills the lever from both ends at once:

* it makes keyframes **cheap** (the pyramid's one favourable surprise — see residual 3 below), but
* it makes the future plane **redundant**: when the past at distance 32 is nearly as informative as the past
  at distance 1, the future at distance 1 has little left to add that the past at distance 1 did not already
  give. The d = 1 bidirectional gain is only 7.3–9.0 %, and that *is* the family ceiling.

This is a genuine **anti-transfer** result against the video-coding prior the charter reasoned from. B-pyramids
buy 20–35 % on natural video because pixel *intensity* decorrelates fast with temporal distance, so a second
reference at half the distance carries real new information. A SegNet **argmax label field** does not
decorrelate fast — it is piecewise-constant over large regions that persist for tens of pairs — so the B-frame
lever collapses to a few percent. The prior transferred the *conclusion*; it did not transfer the *premise*.

hc1's decomposition transfers exactly and localises where the small gain lives. My instrument reproduces hc1's
shape (no-branch share 72.9–74.9 % vs hc1's MEASURED 67.5 % on the shipped stream — the weaker model pays
proportionally more on the hard branch, as expected). At d = 1 the future plane cuts the **"no" (wrong-prediction)
branch by 9.3–11.6 %** and the **"yes" (confirmation) branch by only 1.3–1.4 %**. So the door's *premise* was
right — the future plane does speak to the boundary-jitter binary, and essentially nothing else — but the
*magnitude* is a third of what was predicted, and the pyramid then gives most of it back.

## SCREEN TABLE (MEASURED)

Instrument: adaptive-count model, ONE model class per comparison, arms differing only by the next-plane taps.
Context = causal raster spatial taps × previous-plane reduction [× next-plane reduction]. Three ladders:
α = 4 spatial taps (W, NW, N, NE) × 25-state plane code; β = 6 taps (+ WW, NN) × 5-state (centre only);
γ = 3 taps (W, NW, N) × 25-state. Plane 25-state code = (centre class, encroaching class in the 3×3) — the
boundary-jitter signal. Two cost readings bracket a trained mixer: **KT** (exact sequential Krichevsky–Trofimov
code length, charges the full per-context learning cost the bidirectional arm pays 5–25× more of → LOWER bound
on gain) and **plug-in** (final smoothed distribution, ignores learning cost → UPPER bound, and the only
reading that attributes cost per pair). Common window = the 536 pairs for which every measured distance has
both neighbours.

Per-distance ratios, PB (past+future) over P (past only), plug-in:

| d | α total | β total | γ total | α "no" | β "no" | γ "no" | α "yes" | β "yes" | γ "yes" |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.9229 | 0.9270 | 0.9101 | 0.8996 | 0.9068 | 0.8842 | 0.9856 | 0.9872 | 0.9866 |
| 2 | 0.9343 | 0.9375 | 0.9243 | 0.9074 | 0.9173 | 0.8978 | 1.0073 | 0.9976 | 1.0025 |
| 4 | 0.9362 | 0.9421 | 0.9272 | 0.9094 | 0.9231 | 0.9017 | 1.0113 | 0.9998 | 1.0053 |
| 8 | 0.9568 | 0.9595 | 0.9530 | 0.9357 | 0.9444 | 0.9348 | 1.0182 | 1.0061 | 1.0120 |
| 16 | 0.9534 | 0.9585 | 0.9500 | 0.9312 | 0.9432 | 0.9311 | 1.0187 | 1.0062 | 1.0120 |
| 32 | 0.9544 | 0.9612 | 0.9535 | 0.9327 | 0.9467 | 0.9358 | 1.0196 | 1.0068 | 1.0129 |

The temporal-decay row — P-only at distance d over P-only at distance 1, plug-in (this is the number that
closes the door):

| d | 1 | 2 | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|--:|--:|
| α | 1.0000 | 1.0214 | 1.0528 | 1.0889 | 1.0993 | 1.1146 |
| β | 1.0000 | 1.0170 | 1.0351 | 1.0531 | 1.0611 | 1.0688 |
| γ | 1.0000 | 1.0136 | 1.0460 | 1.0942 | 1.1077 | 1.1292 |

Charter GOP = 8 pyramid, per level, exact per-pair attribution (α; the other ladders agree within 1 pp):

| level | pairs | d | bidi | cost ÷ shipped causal-d1 on the same pairs |
|---|--:|--:|---|--:|
| level8 keyframe | 74 | 8 | no | **1.0986** (the keyframe tax) |
| level4 | 74 | 4 | yes | 0.9773 |
| level2 | 149 | 2 | yes | 0.9576 |
| level1 | 299 | 1 | yes | 0.9213 |
| fallback (edge) | 3 | 1 | no | 1.0000 |
| **net** | 599 | | | **0.95937 → −4.06 %** |

Instrument validation, before any verdict was read: the KT closed form was checked against an independent
sequential simulation (agreement 2.3e-13); both context builders were checked against slow per-pixel reference
implementations (400 / 200 random probes, 0 mismatches); the GOP-8 ratio-composition estimate (4.13 / 4.57 /
5.32 %) reproduces the exact per-level attribution (4.06 / 4.47 / 5.30 %). Count dilution is not a confound:
only 1,015–9,598 contexts are ever occupied (of up to 390,625 nominal), i.e. ≥ 13,000 symbols per live context,
and KT and plug-in agree to within ~2 %.

## PRIOR-LAW PREDICTION vs MEASURED (residuals; m38)

| # | charter PREDICTED | MEASURED | residual |
|---|---|---|---|
| 1a | bidirectional at d = 1 cuts the **"no" branch 30–45 %** | **9.3–11.6 %** | predicted **3.2–3.9× too high** |
| 1b | …and the **"yes" branch 15–25 %** | **1.3–1.4 %** | predicted **~15× too high** |
| 2a | d = 2: **20–30 %** on the "no" branch | **8.3–10.2 %** | predicted **2.4–3.0× too high** |
| 2b | d = 4: **10–20 %** on the "no" branch | **7.7–9.1 %** | at/just below the band's floor |
| 3 | keyframes at d = 8 P-only **+40…+80 % worse** than causal d = 1 | **+5.3…+9.4 %** | predicted **5–9× too pessimistic** |
| 4 | net pyramid **−15…−30 % = −17…−34 KB**; model **+ ≤ 1,500 B** | net **−2.84…−5.68 % = −3,219…−6,442 B**; model **≈ +1,045 B** (DERIVED) | saving predicted **3–9× too high**; model-byte budget **held** |

Residuals 3 and 4 have opposite signs and the same root: the field's slow temporal decay. It made keyframes
6–9× cheaper than predicted (helping the pyramid) and the future plane 3–4× less informative than predicted
(hurting it more). Net, the door does not pay.

## Attack on my own conclusion (§6 of the operating manual)

The load-bearing worry is **direction of bias**: could the screen UNDER-state what the trained mixer would get?

1. **The counting model is weaker than the mixer, and that biases my estimate UP, not down.** Scaled to 600
   pairs the α causal-d1 arm codes this field at ≈ 195 KB against the mixer's MEASURED 113,419 B — the mixer is
   **1.72× stronger**. A stronger base model extracts *less* marginal value from a partially redundant extra
   source. Same direction: my spatial context is 6 raster taps while the mixer has a 7×7 masked conv, two
   dilated convs and the SPM patch pooling; that weakness inflates exactly the residual entropy a temporal
   source can claim. So 4–5.7 % is an over-estimate of the mixer's gain.
2. **The joint is represented, not just the marginals.** Arm B's context carries the full prev × next joint
   (625 joint plane states in α/γ), so "past says A, future says B → mid-transition" is expressible. The gain
   is not being lost to a factorised context.
3. **Count dilution is measured away** (≥ 13,000 symbols per live context; KT ≈ plug-in within ~2 %), and the
   verdict is identical across three ladders that differ 5× in nominal cardinality.
4. **Model-class mismatch is real and named**: the shipped coder is a wavefront (patch-group, delta = 2) model
   with a corrector stage, not raster autoregression. The screen is a prior-transfer instrument, and the ratio
   is what transfers — never the absolute. Argument 1 bounds which way the mismatch pushes.
5. **The layout is not the constraint** — I priced GOP ∈ {2,4,8,16,32} plus the unattainable ceiling, so the
   closure does not rest on the charter's particular pyramid.

What would overturn this: a mechanism that makes the future plane carry information the past plane does not,
rather than a better way to *schedule* the same two planes. Named as ITEM 2 below.

## verdict_scope

**FAMILY** — *bidirectional / B-pyramid temporal context (a second reference plane at +d, any GOP) supplied to
the HPAC integer context-mixing model over this GT SegNet argmax label field.* The closure covers every
decode-order-realizable member and its unattainable supremum, on the pinned field
`tokens_null.u8` sha256 `cc10a7b0…636efb`, against the cl2 λ=1.0 frontier stream of 113,419 B.

What is **NOT** closed, and must not be read as closed: (a) *more causal context* — additional PAST planes at
several distances, a different question with a different redundancy structure; (b) the corrector/model axis
(fx1/fx2), untouched here; (c) motion-compensated references — mc1 closed the MC *previous* plane at +160 B,
and nothing here re-opens or extends that; (d) any claim about natural-video B-frames, whose premise (fast
intensity decorrelation) this field does not satisfy.

Re-open criteria: a MEASURED demonstration that the future plane carries label-field information the past plane
does not (e.g. conditional mutual information I(X_t ; X_{t+1} | X_{t−1}, spatial) materially above the
9.3–11.6 % no-branch figure measured here), or a change of coded object that restores fast temporal decorrelation.

## Ladder row

| rung | object | net stream saving | model cost | net | verdict |
|---|---|--:|--:|--:|---|
| GOP 2 | keyframes d=2 + level-1 bidi | 3,219 / 3,346 / 4,410 B | ≈ 1,045 B | 2,174 / 2,301 / 3,365 B | below F1 |
| GOP 4 | + level-2 | 4,159 / 4,424 / 5,623 B | ≈ 1,045 B | 3,114 / 3,379 / 4,578 B | below F1 |
| **GOP 8 (charter)** | + level-4 | 4,608 / 5,069 / 6,015 B | ≈ 1,045 B | 3,563 / 4,024 / 4,970 B | **below F1** |
| GOP 16 | + level-8 | 4,908 / 5,370 / 6,307 B | ≈ 1,045 B | 3,863 / 4,325 / 5,262 B | below F1 |
| GOP 32 (family best) | + level-16 | 5,053 / 5,504 / 6,442 B | ≈ 1,045 B | 4,008 / 4,459 / 5,397 B | below F1 |
| ceiling (unattainable) | every pair bidi d=1 | 8,828 / 8,463 / 10,242 B | ≈ 1,045 B | 7,783 / 7,418 / 9,197 B | supremum, still ≤ F1 for 2 of 3 ladders |

## Artifacts (KEEP THE PAYLOAD)

Store `/Volumes/VertigoDataTier/pact/ddm_bd1_bidirectional_pyramid_context/` (APFS).
`PAYLOAD_MANIFEST.json` (3,989 B, sha256 `27af5080…`) carries sha256 + bytes for all 13 artifacts:

* input field `tokens_null.u8` — 117,964,800 B, sha256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
* `screen_report_v3.json` — 91,143 B, sha256 `ed8062cefcb8d551…` (the authoritative report; v1/v2 retained)
* three launch manifests + run logs + `safe_run` status receipts

Producer: `experiments/ddm_bd1_bidirectional_context_screen.py` (committed). The only intermediate not
persisted is the per-(ladder, distance, arm) count histogram set; it is a deterministic function of the
sha-pinned field plus the committed producer and rebuilds in **134 s** of single-process CPU, and every derived
statistic it carries is already in the report — certified rebuildable, no signal lost.

## Owed items

## ITEM 1 — Register the closure so the next arm cannot re-open the door blind
Register `bidirectional_pyramid_context_gain_v1` in the canonical-equations registry with the measured
temporal-decay law and the family ceiling, so a future charter proposing a second reference plane on this field
gets the prior-law prediction line for free. Owner: ddm_bd1 (this unit).

## ITEM 2 — The residue this screen did NOT price: multiple PAST planes
The measured slow decay says the past plane at distance 32 still carries 89–94 % of the distance-1 plane's
predictive value. That is a statement about *redundancy between references*, and it cuts against a second
FUTURE plane — but it says nothing about whether two PAST planes at different distances (d = 1 and d = 2, say)
carry complementary information about *motion* that either alone lacks, which is a different estimand. The same
$0 instrument prices it in ~2 min per configuration by swapping the next-plane taps for a second previous-plane
tap set. Not a bd1 deliverable (bd1's object is bidirectional); named so it is not lost. Owner: unassigned.

## ITEM 3 — hc1's decomposition reproduced by an independent instrument
This screen independently reproduces hc1's yes/no branch shape (no-branch share 72.9–74.9 % under a
counting model vs hc1's 67.5 % under the shipped mixer). That is a free cross-check of hc1 by a different
model class and is worth recording beside hc1's own number. Owner: unassigned.

---

`cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]` — UNMOVED by this unit, by design: bd1's
falsifier fired at the $0 screen and the correct action was to spend nothing further.
