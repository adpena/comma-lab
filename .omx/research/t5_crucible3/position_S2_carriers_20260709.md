# P1 SEAT S2 — CARRIER COMPOSITION / RATE-EV (v8 optimal final form)

**Seat:** S2 (Fridrich/Yousfi/Ballé lens). **Charter:** OPEN Q1 (increment-1 carrier set; independent-
compose vs edge-centric-must-not-double-pay) PRIMARY + OPEN Q2 (THE residual-coder decision — the 0.079 S
named enemy) PRIMARY; the measured negatives N-1..N-4 (do NOT re-open blind); the scaffold byte-cost scope
bug (DELTA §D). INDEPENDENT position — no cross-read of crucible-3 sibling seats. Cites
`docs/operating_manual_craft_handoff.md` (re-derive from the primary artifact; label every claim by how it
was obtained; spend depth on the number that drives a decision + any word "complete/lossless/free").
Pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS. `[no-triality]` (position doc; P2 owns
leg propagation).

STORES CONSULTED: `t5_crucible3/{CONVENING,DELTA_GROUNDING,ORCHESTRATION_LEDGER}_20260709.md` · the rate
ledger RE-DERIVED from the registered equation `src/tac/canonical_equations/v8_geometric_rate_decomposition_
20260709.py` (NOT memo-trusted) · `v8_movable_residual_rollup_20260709.md` §B/§C (the residual sidecars) ·
`v8_roadlane_geometric_rate_20260709.md` (the 0.042 Road/Lane residual + the median-smooth coverage cost) ·
`src/tac/boundary_math/road_undriv_bulk_field.py` (RE-READ on-disk: `bulk_boundary_byte_cost` L385 vs
`horizon_poly_xi_byte_cost` L470 — the byte-cost bug re-derived from source, not the memo) · crucible-2 S2
position (the seat PATTERN only, explicitly allowed) · DELTA §H measured negatives N-1..N-4. **NOT
consulted (independence):** crucible-3 sibling positions S1/S3/S4/S5/S6 (none exist yet / would-be-read
blind); SPEC_v8 §1/§6 verbatim (worked from DELTA_GROUNDING, the P1 pack).

---

## 0. THE ONE-SENTENCE POSITION

The residual-coder decision is NOT a 3-way pick {close-0.079 / dominant-only / lossy} — it is **ONE
rate-distortion waterfill**: `min_r [ 100·d_seg(r) + 25·resid_bytes(r)/N ]` over residual coverage `r`,
whose cost-per-residual-byte is LOWERED by two build levers (de-share attribution + curve-relative
offset coding) and whose px are selected FLIP-WEIGHTED (code the uncovered px near flips, drop the ones in
stable regions); increment-1 ships the ONE new edge-centric Road+Undriv field (calling
`horizon_poly_xi_byte_cost`, NEVER the scope-buggy `bulk_boundary_byte_cost`) + REUSE 4 carriers + this
shared SOLVED residual coder as the ONE cheap co-build, and its operating point is MEASURED — a point on
the 0.061→0.140 range, not pre-committed.

---

## 1. THE DECISIVE REFRAME — THE RESIDUAL DECISION IS AN R-D WATERFILL, NOT A 3-WAY CHOICE (DERIVED)

This is the load-bearing move of my seat, and I re-derived it from the scoring function, not a memo.

DELTA §E frames Q2 as three options: (a) close 0.079 via {de-share + curve-relative} to land COMPLETE
below frontier; (b) ship DOMINANT-ONLY (0.061) with the residual as a measured-owed sidecar; (c) accept a
LOSSY operating point (N-4 says dominated on the lane edge). **All three are points on ONE axis and the
framing hides the real objective.** The uncovered boundary px you DON'T code do not vanish — in the frozen
scorer they FLIP the SegNet argmax (they sit in the ~4.7%-area boundary annulus that carries ~97% of d_seg,
DELTA §F). So NOT coding a residual px is not a free rate saving; it is a rate-for-d_seg TRADE. The correct
object is the score itself:

    S_partial(r) = 100 · d_seg(r)  +  25 · resid_bytes(r) / 37_545_489        (+ dominant 0.061 rate fixed)

where `r ∈ [0,1]` is residual coverage. `r=0` = dominant-only (rate 0.061, MAX d_seg from uncovered
flips); `r=1` = complete-lossless (rate 0.140, MIN uncovered-flip d_seg). **The optimum is INTERIOR and is
a classic KKT waterfill** (Ballé's R-D, SOLVE-WHERE-SOLVABLE per Synthesis Requirement A): code residual px
in DECREASING order of `Δd_seg_saved / Δbytes`, stop when the marginal d_seg-saved-per-byte falls to the
rate cost per byte (`25/N` per counted byte). Options (a)/(b)/(c) are the two endpoints + a strawman; the
answer is the solved interior point.

**Why this reframe is the whole seat:** it converts "which coder" (a rate question) into "which px are
worth their bytes in d_seg" (a score question), which is exactly the FLIP-WEIGHTED discipline N-1's open
reformulation demands and the annulus-precision allocation risk-4 wants. It also dissolves the false
tension between "beat frontier on complete" and "N-4 lossy is dominated": see §4.

---

## 2. Q1 — CARRIER COMPOSITION: THE INDEPENDENCE STRUCTURE + THE DE-SHARE HUB (the must-not-double-pay map)

**The carriers are NOT all independent-compose.** I re-derived the sharing structure from the region-
adjacency graph (Road = hub) + the residual rollup §A/§B:

| carrier | edge(s) owned | dominant S | residual S | shares boundary px with |
|---|---|---|---|---|
| Lane analytic band (`analytic_lane_render_band`) | Road↔Lane | 0.0275 | **0.04203** | Movable (lane fragments near cars) |
| Road+Undriv bulk field (**NEW**, `road_undriv_bulk_field`) | Road↔Undriv horizon | 0.0032 | **0.01892** | Movable (secondary arcs = car tops breaking the horizon) |
| Movable sparse sites (v7.5 Hungarian ξ-track) | Road↔Mov + Undriv↔Mov | 0.00344 | **0.01741** | **BOTH above** (it is the de-share hub) |
| MyCar hood static (`hood_static_component`) | Road↔MyCar | 0.0202 | 0 (complete) | — (rigid static, disjoint) |
| b_c tie bias (`laguerre_logit_offset`) | Road↔Lane calibration | ~0 | — | — |

**The decisive structural fact: Movable is the DE-SHARE HUB.** Its sites intersect BOTH the horizon
residual (secondary arcs, MEASURED 1.6–2.0 crossings/row — car tops crossing the horizon line) AND the
Road/Lane residual (lane fragments near cars). Movable already de-shares its own two edges internally (ONE
carrier for Road↔Mov + Undriv↔Mov; the two sides of the same blobs, DELTA §D). But at the WHOLE-SCENE
level, its footprint ⊂ two OTHER edges' residual-owed sets. **This is risk-1 (EDGE-DUPLICATION) surviving
into the residual layer.** SPEC_v8's edge-centric cure de-duplicates the DOMINANT streams (one field per
edge); it does NOT automatically de-duplicate the RESIDUALS. The DELTA's "de-share double-count" headroom
lever is exactly this un-fixed residual overlap.

**⇒ The composition rule (a VALUE, the accounting order):** assign every uncovered boundary px to EXACTLY
ONE carrier — the carrier whose region it borders — via a **Movable-first attribution pass**: (1) compute
the Movable site footprint; (2) SUBTRACT it from the horizon residual owed-set AND the Road/Lane residual
owed-set BEFORE either is coded. This is pure de-duplication — **no coder change, just correct attribution
= genuinely free** (removes double-paid px). The horizon secondary arcs and the near-car lane fragments are
then paid ONCE, in the Movable carrier (where they already largely sit — 70% bbox cover), never twice.

**Which carriers are truly independent-compose:** MyCar hood (rigid static, spatially disjoint from every
other edge — no shared px, composes freely) and the b_c tie bias (~0 bytes, a logit offset not a boundary).
Everything else touches the Road hub and MUST route through the Movable-first attribution pass.

**THE SCAFFOLD BYTE-COST BUG (re-derived from source, NOT the memo — DELTA §D standing catch):** I RE-READ
`road_undriv_bulk_field.py`. It already has TWO functions: `bulk_boundary_byte_cost` (L385) measures the
FULL Road MASK (packbits + row-span RLE of the whole Road region → the full Road perimeter, all 4 of Road's
edges) — this is the naive 707 B/frame path, self-labelled "conservative FULL-boundary number"; and
`horizon_poly_xi_byte_cost` (L470) measures the DOMINANT horizon arc only (deg-3 poly + ξ, with
`residual_sidecar_owed=True` and an honest `scope_note`). **The bug is not "missing a mode" — it is that
the interface exposes BOTH and the naive one is the default-shaped call.** The edge-centric Road↔Undriv
field's zero-set IS the horizon (~426 px = 19% of the 2228-px Road perimeter, MEASURED); the other 81%
(Road↔Lane 47% / Road↔MyCar 23% / Road↔Movable 5%) is reconstructed by the OTHER carriers in the tropical
argmax composition, so paying for it here IS the risk-1 double-count. **VALUE for the increment-1 config:
call `horizon_poly_xi_byte_cost`; `bulk_boundary_byte_cost` is a diagnostic-only ceiling and MUST NOT
appear in any rate claim. Standing seal check: grep the config for `bulk_boundary_byte_cost` in a
rate-claim path = FAIL.** (Second sub-bug, DELTA §D: the field must be multi-component-Road-aware — Road is
multi-blob in 37.2% of frames — but that is a d_seg-correctness bug, S3's field-representation territory,
not a byte-cost claim; I flag it, S3 owns it.)

---

## 3. Q2 — THE RESIDUAL-CODER DECISION: THE VALUE (the named enemy attacked)

**The named enemy is the 0.079 S residual sidecar** (RE-DERIVED from the registered equation: complete
0.140 − dominant 0.061 = 0.079; decomposition Road/Lane 0.04203 + horizon 0.01892 + Movable 0.01741 =
0.07836 ✓, hood 0 + Lane/* 0). It is genuinely near coordinate entropy on today's generic coder — a
chain-code residual buys only **−13%** (MEASURED on Movable, §C). My decision is a VALUE, three parts:

**(V-1) BUILD the shared residual coder as CURVE-RELATIVE + DE-SHARE-AWARE (the cost-per-byte lever).**
The two DELTA headroom levers are NOT alternatives — they PARTITION the residual by distance-to-generator:
- **De-share** handles the FAR-from-generator px (secondary arcs, near-car fragments) by ATTRIBUTING them
  to their true carrier (Movable) — §2's Movable-first pass. These px are far from the horizon/lane
  generator, so curve-relative offset coding would NOT help them; correct attribution does.
- **Curve-relative** handles the NEAR-generator px (the poly-fit residual — horizon 1.46 px median, lane
  1.00 px median, MEASURED): code the signed NORMAL OFFSET from the generator curve (a small int in a thin
  band) + the ALONG-curve arc-length position, instead of the absolute (row, col) flat-index the generic
  coder uses. For a band of half-width W px the normal offset needs ~log2(2W+1) bits vs the ~0.4–0.6 B/px
  absolute-coordinate cost. This is Ballé/UNIWARD in the coordinate domain: the residual lives on a
  1-D manifold (the generator curve), so code it in the manifold's chart.
These are cheap because they are SOLVE-not-train (a coordinate transform + a coder swap; no training loop).

**(V-2) SELECT the residual px FLIP-WEIGHTED via a KKT waterfill (the operating point — §1 reframe).**
After (V-1) lowers the cost-per-byte, do NOT reflexively code ALL px to "complete." Code residual px in
decreasing `Δd_seg_saved / Δbytes`; a px whose omission does not flip a SegNet argmax cell (a faint
occluded fragment in a stable region) has ~0 d_seg-value and is DROPPED (pure rate saving — net-NEGATIVE
S). Px near flips (in the annulus) are kept. **This is the flip-weighted discipline N-1 opened + the
annulus-precision allocation risk-4 wants, applied to residual-px selection.** The waterfill is SOLVED in
structure; its DATA (per-px d_seg-value) is the MEASURED d_seg-through-R that increment-1 already runs — so
the operating point is a by-product of the d_seg measurement, not a separate governed cost.

**(V-3) The OPERATING POINT is MEASURED, pre-registered as a decision rule (a VALUE, not a TBD):**
> Ship the waterfill-optimal `r*` = the coverage where marginal `Δd_seg_saved·100 = Δbytes·25/N`.
> Fallback if `d_seg(r)` cannot be measured through R in the increment-1 budget: ship **complete-lossless
> with (V-1)** (the conservative correct default — lossless never raises d_seg vs the dominant-only
> alternative), and record the waterfill as owed. NEVER ship dominant-only as a rate CLAIM without stating
> the uncovered-flip d_seg penalty (that would be cherry-picking 0.061 as if it were lossless — NO-FAKE).

**Is the residual coder a REQUIRED increment-1 co-build (the Q1 "smallest decisive" sub-question)?** YES,
but CHEAP and OFF the d_seg critical path. Rationale: (i) it is SHARED — built once, all three residual
edges use it; (ii) it is SOLVE-not-train — a coordinate transform + attribution pass + waterfill, no
training loop, days not weeks; (iii) without it, increment-1's rate story stays a RANGE (0.061→0.140) and
v8's rate advantage over v7.5.2 is only "wash on dominant" — the coder converts the range to a POINT the
P8 comparison brief needs; (iv) the d_seg-through-R measurement (the actual BET) does NOT wait on it. So:
**increment-1 = {the ONE Road+Undriv field build} + {the shared solved residual coder co-build}**, two
builds, the second small.

---

## 4. THE MEASURED NEGATIVES — do NOT re-open blind + the N-4 boundary I must NOT cross (verdict_scope)

- **N-1 (OT/Laguerre area-mass head offsets HURT, both arms; eq `laguerre_ot_head_offset`).** verdict_scope:
  **FORMULATION** (mass-matching to raw GT area frequencies as a d_seg surrogate). Consequence for my seat:
  the b_c tie bias in the carrier table is calibrated **FLIP-WEIGHTED** (match argmax to where flips are,
  not to raw area), OUT of the scorer-gradient loop (risk-2 guard). Do NOT propose an area-mass OT offset.
- **N-2 (lane-ξ ego-transport NO-GO; eq `lane_groundframe_xi_transport_no_collapse_v1`).** verdict_scope:
  **FORMULATION** (the lane RATE axis on a ground-canonicalized chart). Consequence: use **0.0275 S** for
  Road/Lane dominant; do NOT project a horizon-class ξ transfer onto the ground-frame lane generator OR its
  residual. The horizon DOES use ξ (image-frame, removable ego-pitch intercept) — the asymmetry is real.
- **N-3 (dense medial ≈ bitmap, 1.09×).** verdict_scope: **FORMULATION** (the *dense* generator). The
  few-coefficient parametric generator is the lever, not the dual choice. Do NOT build a dense medial axis.
- **N-4 (waterfill on Road/Lane net-negative; the knee at lossless).** verdict_scope: **FORMULATION** —
  and this is the one I must NOT over-generalize. N-4 was MEASURED on **GENERATOR-BAND quantization** (the
  median-smooth of the lane band: rate 0.0275→0.0176 but coverage 72.5%→66.8%, lane-recall 63.3%→**48.0%**
  — RE-DERIVED from `v8_roadlane_geometric_rate` §"Temporal-denoise headroom"). It says: **do not lossily
  quantize the GENERATOR.** My (V-2) residual-px waterfill is a DIFFERENT lever — it selects which UNCOVERED
  px to code losslessly, guided by their d_seg-value; it never touches the generator band. **N-4 does NOT
  bind the residual-px selection** (different surface, different distortion mechanism). BUT I inherit N-4's
  CAUTION: whether the residual-px waterfill is net-positive is UNMEASURED — increment-1 MEASURES it, and
  the DEFAULT when it is a wash is lossless-complete (V-3 fallback), never a lossy generator trade.

---

## 5. THE SUB-0.15 ARITHMETIC — WHY THIS DECISION IS THE HIGHEST-EV RATE CALL (relative-significance)

Remaining gap to the sub-0.15 target = pointer 0.19110 − 0.15 = **0.0411**. Magnitudes below are normalized
by it (per the anti-magnitude-dismissal discipline).

RE-DERIVED from the pointer decomposition (rate term 0.118, so d_seg+pose = 0.19110 − 0.118 = 0.073):
- **AT FIXED pointer-level d_seg+pose (0.073):** v8 dominant-only rate 0.061 → S = 0.073 + 0.061 = **0.134
  < 0.15 (SUB-0.15, advisory)**. v8 complete rate 0.140 → S = 0.073 + 0.140 = **0.213 > 0.19110 (WORSE
  than pointer)**. **The 0.079 residual gap = 192% of the entire remaining budget-to-target** — it is
  literally the difference between sub-0.15 and worse-than-pointer, at fixed d_seg. This is why it is THE
  named enemy, not a rounding detail; and it is why (V-2)'s waterfill (which lands the interior point) is
  the highest-EV rate call in the whole v8 stack.
- **THE HONEST CONDITIONAL (the caveat that travels with the number):** this arithmetic assumes v8's
  TRAINED d_seg reaches pointer-level (0.073). It does NOT — v8's d_seg-through-R is the UNPROVEN
  edge-centric-decoupling BET (DELTA §J; v7.5's advisory witness d_seg is 0.455, far above pointer). So the
  residual-coder EV is **CONDITIONAL on the d_seg BET landing**. The rate half already banks 0.057 S
  headroom on dominant (139% of the gap) IF d_seg holds; the residual decision protects that headroom from
  being eaten by uncovered-flip d_seg. **d_seg is the blocker; the residual coder is how the rate win
  survives contact with the annulus.** I flag this so the synthesis does not quote 0.134 as a v8 result —
  it is a fixed-d_seg counterfactual, MEASURED only by increment-1's byte-close + d_seg-through-R.

---

## 6. CONFIG-SHAPED BLOCK — the increment-1 carrier set + residual coder (S2 position)

```
# ============ v8 INCREMENT-1 CARRIER SET (S2 position) ============
# THE ONE NEW BUILD (d_seg BET; edge-centric):
road_undriv_bulk_field    = BUILD  # the Road<->Undriv edge-centric bulk-boundary field.
                                   #   byte-cost CALL = horizon_poly_xi_byte_cost (L470)  <-- NOT bulk_boundary_byte_cost
                                   #   dominant 0.0032 S (deg-3 poly + xi, 14.6x MEASURED). residual_sidecar_owed=True.
                                   #   MUST be multi-component-Road-aware (Road multi-blob 37.2% frames) -- S3 owns correctness.

# REUSE (4 carriers, exist on-disk, verified DELTA §K):
lane_analytic_band        = REUSE  # analytic_lane_render_band (LBND2). dominant 0.0275 (N-2: NO xi transfer, ground-frame).
movable_sparse_sites      = REUSE  # v7.5 Hungarian xi-track bbox. dominant 0.00344 (de-shares Road<->Mov + Undriv<->Mov).
                                   #   = THE DE-SHARE HUB: its footprint is subtracted from horizon + lane residuals FIRST.
mycar_hood_static         = REUSE  # hood_static_component. 0.0202 COMPLETE (no residual; rigid, disjoint, independent-compose).
bc_tie_bias               = REUSE  # laguerre_logit_offset. FLIP-WEIGHTED (N-1), calibrated OUT of scorer-gradient loop (risk-2).

# THE ONE CHEAP CO-BUILD (shared, SOLVE-not-train; the named-enemy attack):
residual_coder            = BUILD  # shared across all 3 residual edges. TWO cost-per-byte levers + 1 selection rule:
  de_share_attribution    = ON     #   (V-1a) Movable-first pass: assign each uncovered px to EXACTLY ONE carrier;
                                   #          subtract Movable footprint from horizon + Road/Lane residual owed-sets. FREE.
  curve_relative_offset   = ON     #   (V-1b) code signed NORMAL offset from generator + along-curve arc-length,
                                   #          NOT absolute (row,col) flat-index. For the NEAR-generator poly-fit residual.
  flip_weighted_waterfill = ON     #   (V-2) KKT: code residual px by decreasing d_seg_saved/byte; stop at 25/N marginal.
                                   #          FALLBACK (V-3) if d_seg(r)-through-R unmeasured in budget = complete-lossless.

# OPERATING POINT (V-3, a VALUE not a TBD):
residual_operating_point  = MEASURED   # r* = waterfill knee (d_seg_saved*100 = bytes*25/N marginal).
                                       # pre-registered range 0.061 (dominant) -> 0.140 (complete); r* interior.
                                       # NEVER ship dominant-only as a rate CLAIM without the uncovered-flip d_seg penalty (NO-FAKE).

# ============ STANDING SEAL CHECKS (S2-owned) ============
#  1. grep config for `bulk_boundary_byte_cost` in any RATE-CLAIM path        -> FAIL (scope bug; use horizon_poly_xi).
#  2. any residual px paid by TWO carriers (Movable footprint not subtracted) -> FAIL (risk-1 in residual layer).
#  3. any GENERATOR-band lossy quantization proposed as a residual-gap closer -> FAIL (N-4 dominated; FORMULATION-scope).
#  4. b_c calibrated area-matched (not flip-weighted) OR inside scorer-grad   -> FAIL (N-1 + risk-2).
#  5. lane generator or its residual projected with an xi ego-transport       -> FAIL (N-2 LAW; ground-frame chart).
#  DO-NOT-REOPEN-BLIND: N-1 area-mass OT / N-2 lane-xi / N-3 dense-medial / N-4 GENERATOR-band waterfill.
```

---

## 7. EPISTEMIC LABELS + FLAGS FOR SYNTHESIS / RED-TEAM

- **DERIVED (mine, load-bearing, ATTACK IT):** the R-D-waterfill reframe (§1) — that Q2's three options are
  two endpoints of one `min_r [100·d_seg(r) + 25·bytes(r)/N]` axis, and the answer is the SOLVED interior
  flip-weighted point. This is MY interpretation, re-derived from the scoring function, NOT an established
  SPEC clause. If red-team rejects it, the fallback is the DELTA's discrete choice — and my discrete
  recommendation is then **complete-lossless with (V-1)** (curve-relative + de-share), because lossless
  never trades d_seg (V-3). I believe the waterfill holds because the uncovered px are IN the annulus (so
  they carry d_seg-value that must enter the objective) — but it is the single most-attackable claim here.
- **MEASURED (re-derived from primary artifacts):** the rate ledger 0.339/0.061/0.140 + the residual
  decomposition 0.04203/0.01892/0.01741 (registered equation, RE-DERIVED not memo-trusted); the two
  byte-cost functions in `road_undriv_bulk_field.py` (RE-READ on-disk L385/L470); the horizon-scope 426/2228
  px = 19% (DELTA §D); Movable's 1.6–2.0 crossings/row (§C); the median-smooth coverage cost 72.5→66.8%,
  recall 63.3→48.0% (N-4 surface, RE-DERIVED from the Road/Lane memo); chain-code −13% (§C); N-1/N-2 signs.
- **ESTIMATED (flag, not load-bearing):** de-share headroom ~0.006–0.020 S (removes the secondary-arc +
  near-car fraction of the horizon 0.019 + lane 0.042 residuals — the exact split is MEASURED by the
  Movable-first pass, not yet run); curve-relative headroom ~20–40% on the near-generator poly-fit portion
  only (the faint/occluded fragments that dominate the Road/Lane 0.042 residual are near coordinate entropy
  and do NOT compress — so curve-relative does NOT by itself reach complete-below-frontier). I deliberately
  do NOT assert either lever closes the full 0.079 — that is the RESIDUAL-CODER-OPTIMISM trap (S5's target).
- **ASSUMED_AWAITING_VERIFICATION:** that the residual-px waterfill is net-positive (inherits N-4 caution;
  increment-1 measures it); that v8's trained d_seg reaches pointer-level (the §5 conditional — the BET).
- **Honest tension I did NOT resolve:** the Road/Lane residual (0.042, 53% of the enemy) is dominated by
  faint/occluded lane fragments, which may be BOTH far from the generator (curve-relative no help) AND
  d_seg-VALUABLE (lanes are the flip-prone class — dropping them via waterfill could COST d_seg). If so,
  the biggest residual chunk resists all three of my levers and the honest operating point is closer to
  complete (0.140) than to dominant — which pushes v8's rate toward a WASH-with-frontier, not a win. This
  is the load-bearing empirical question increment-1 must answer; I bias to building the coder + measuring
  rather than pre-committing, precisely because this chunk's disposition is unknown.
- **Cross-seat couplings I depend on (flag for P2):** (a) the flip-weighted waterfill's DATA is S4's
  d_seg-through-R measurement + S3's P-C paint floor — my operating point is a by-product of their heavy
  measurement, not a separate cost; (b) the multi-component-Road byte-cost sub-bug is S3's field-correctness
  territory; (c) whether the residual coder is on or off the increment-1 critical path depends on S5's
  apparatus×5 opportunity-cost read vs the sealed v7.5.2 — I hold it is OFF the d_seg critical path and
  cheap, so it composes without gating the BET.

## OPEN QUESTIONS FOR P2 (the CHIEF-DESIGNER)

1. **Adopt the R-D-waterfill reframe (§1) or the DELTA's discrete 3-option choice?** If discrete, my pick is
   complete-lossless-with-(V-1); if waterfill, the operating point is MEASURED by increment-1. P2 must pin
   ONE (the contract forbids TBD).
2. **Is the residual coder ON the increment-1 build (my §3 YES-but-cheap) or deferred to increment-2?** The
   d_seg BET does not need it; the rate POINT (vs range) does. P2 weighs against apparatus cost (S5).
3. **The Road/Lane 0.042 residual disposition (my §7 honest tension):** if it is far-from-generator AND
   d_seg-valuable, no lever touches it and v8 rate is a wash-with-frontier. Does the comparison brief carry
   the RANGE 0.061→0.140 with this as the binding uncertainty, or does P-C/increment-1 resolve it first?
