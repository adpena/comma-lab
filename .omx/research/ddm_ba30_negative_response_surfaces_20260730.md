# ddm_ba30 — THE 2026-07-30 NEGATIVES AS RESPONSE SURFACES (unification + completeness audit)

**Window: 2026-07-30 only** (sisters hold 07-29 and 07-31). Operator directive: *"they're probably not
optimal. Don't make binary judgment prematurely. In fact, never make binary judgment. Always seek to
understand."* + *"I've told you numerous times no binary results ever. We are proceeding Einsteinian and
according to our design and guiding philosophies and principles, which include unification, completeness."*

**POINTER HONESTY FIRST: submittable `0.1910828242 [contest-CPU]` UNMOVED.** This is a read-only audit.
No scorer slot taken, no training, no dispatch, no pointer mutation. Every number is quoted from a
committed memo or recomputed at $0 from an existing committed cache; each carries
MEASURED / DERIVED / LABELED. `[no-triality] [p0-ledger-ok]`.

**STORES CONSULTED:** all 30 `.omx/research/*20260730*` memos (full read: gc10, gc11, gc9, ea1, dw1,
pj1, ax1, gr1, nv1, pa1r, pa1b, ps1, zb1, kl1, qp1, v4b, v4c, pm1, pi2, co9, fu1, su2, qa45); the 07-30
commit range (`git log --since 07-30 --until 07-31`, 50 commits); `ddm_deferral_queue_ledger_20260729.md`
(95 QA rows); `ddm_fp1_class_field_projection_20260731.md` and `ddm_gc12_wall_branch_convocation_20260731.md`
(OUT of window — consulted only to place the successor coordinate BR-D fired on); memories
`boolean_flags_are_a_ui_over_a_continuum…`, `pose_is_the_largest_axis…`, `negative_existence_claims…`,
`opportunity_pools_non_additive…`, CLAUDE.md class-index law. **Deliberately NOT loaded:** the 07-29 and
07-31 memo bodies beyond the two named (sister windows), the burn_out run dir (hands-off), the named
parallel-session files.

**DENOMINATOR: 44 negative-shaped verdicts enumerated in-window · 44 placed on 6 surfaces · 0 unreachable.**
Of the 44, **6 are ledger/apparatus housekeeping** (QA18 closed-mooted, QE03 re-gate, the stale-digest
correction, QA43 orphaned-behind-dead-gate, joint_tail deferred-sealed, rowband blocked-on-grid) and carry
no measurement; they are placed as *coordinates that were unreachable*, not as points. 8 further rows sit
on a surface at a coordinate that is itself unmeasured — each is named at its surface.

---

## §0 THE ANSWER, LEAD

Six response surfaces carry all 44. Two results dominate:

1. **SURFACE B (reachability).** `pj1`'s capacity floor **f = 0.504824** and `fp1`'s trained-head point
   estimate **f′ = 0.499366** are not two capacity measurements. Recomputed at $0 from
   `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`: the **constant-Undrivable predictor scores
   d_seg = 0.50482448154025605**, per-class `[.23233240763346355, .00585461934407552, 0.0,
   .01237932840983073, .25425812615288629]`. Checked against pj1's own receipt
   `ddm_pj1_20260730/warm_l2/gate/p1_receiver_realized_verdict.json` (`d_seg_mean`, `per_class_d_seg`,
   n_pairs 600, gt_cache gt_n600): **abs diff 0.00e+00 on all five classes and on the total — bit-exact in
   float64.** fp1's f′ matches to 2–3 significant figures on every class and is 1.08 % below the total.
   **Both probes measured the degenerate all-majority-class collapse, whose d_seg is a dataset constant
   (1 − 0.495176), independent of tokens, head size, trunk decodability, objective, or photometric range.**
   Neither number is a capacity.
   (fp1's LOAD-BEARING verdict — the **receiver floor F1 = 0.008305**, computed on the *GT* argmax and
   therefore collapse-immune — is untouched, airtight, and BR-D's firing on it is sound. §2.)

2. **SURFACE A (seg↔rate exchange).** The score fixes one level set: **15,018 B per 1e-4 d_seg**
   (= 1.27311 B/flip). At the same parent B on the same day, the **reclaim** direction sits at 10,805
   (0.72× water, "bank EMPTY", 22:49) and the **spend** direction at 5,542 (2.71× bargain, reported as an
   aside in the control arm of a negative, 23:09). They are not reciprocal — ratio **1.95×** — which means
   the two moves travel different directions in token space, and **alternating them is arithmetically
   net-positive** (≈ 30.4 KB at constant d_seg ⇒ ≈ 0.0203 S). Neither memo cites the other's rate.
   Separately, the day's **winning** encode-side reclaim (`gr1` cell_drop50, **50,185 B per 1e-4 = 3.34×
   water**, byte-closed n600) landed at **01:44** and is cited by neither 22:49 nor 23:09.

---

## §1 SURFACE A — THE MARGINAL SEG↔RATE EXCHANGE `m`

**The object.** `m = |ΔB| / |Δd_seg|`, reported here in **bytes per 1e-4 d_seg**. Every "dominated",
"worse_s", "bank EMPTY", "NO-GO", "break-even" on the rate axis is one point on this one surface.

**The level set is a constant of the score, not a knob.** `100·Δd_seg = 25·ΔB/37,545,489` ⇒
**w = 100·37,545,489/25 = 150,181,956 B per unit d_seg = 15,018.2 B per 1e-4 = 1.27311 B/flip**
(MEASURED-by-definition; reproduces gr1's stated water 1.273). Reclaiming pays iff `m > w`; spending pays
iff `m < w`. **Because w is fixed, a verdict on this surface reports only WHERE WE WERE STANDING.**

### §1.1 Every in-window point, placed (coordinates: direction · admission unit · depth · parent · loop)

| # | source | dir | unit | depth/rule | parent | ΔB | Δd_seg | **m** | m/w |
|---|---|---|---|---|---|---:|---:|---:|---:|
| A1 | gr1 cell_drop35 (n48) | reclaim | cell col | 35 % low-\|g\|-sum | pfs1 D1 | 130,160 | 1.9e-5 | 685,053 | 45.6× |
| A2 | **gr1 cell_drop50 (n600, byte-closed)** | reclaim | cell col | 50 % | pfs1 D1 | 210,775 | 4.20e-4 | **50,185** | **3.34×** |
| A3 | gr1 cell_drop63 (n48) | reclaim | cell col | 63 % | pfs1 D1 | 292,181 | 1.150e-3 | 25,407 | 1.69× |
| A4 | **nv1 thr1** | reclaim | token \|signed\| | ≤1 lvl (35.2 % nz) | B ep440 | 62,452 | 5.78e-4 | **10,805** | **0.72×** |
| A5 | gr1 cell_drop75 (n48) | reclaim | cell col | 75 % | pfs1 D1 | 372,205 | 5.630e-3 | 6,611 | 0.44× |
| A6 | gr1 tok_rung_b (n48) | reclaim | token, graded | {L8,L4} | pfs1 D1 | 156,523 | 2.587e-3 | 6,051 | 0.40× |
| A7 | gr1 tok_rung_a (n48) | reclaim | token, graded | {L8,L4} | pfs1 D1 | 99,238 | 1.861e-3 | 5,332 | 0.36× |
| A8 | nv1 thr2 (the "117 KB knee") | reclaim | token \|signed\| | ≤2 (66.1 %) | B | 139,296 | 3.334e-3 | 4,178 | 0.28× |
| A9 | gr1 tok_drop50 (n48) | reclaim | token \|g\| | 50 % | pfs1 D1 | 107,726 | 2.789e-3 | 3,862 | 0.26× |
| A10 | nv1 seg-aware p75 (QA80-gated) | reclaim | cell safety, **∞ amp** | safest 25 % | B | 48,951 | 1.740e-3 | 2,814 | 0.19× |
| A11 | gr1 tok_drop35 (n48) | reclaim | token \|g\| | 35 % | pfs1 D1 | 33,580 | 1.251e-3 | 2,684 | 0.18× |
| A12 | gr1 tok_drop65 (n48) | reclaim | token \|g\| | 65 % | pfs1 D1 | 211,500 | 8.152e-3 | 2,594 | 0.17× |
| A13 | **pa1r delta-sparsity, matched-epoch** | reclaim | in-loop group-L2 | w=0.03 | B ep464 | 868 | 3.70e-5 | 2,346 | 0.156× |
| A14 | nv1 thr3 | reclaim | token \|signed\| | ≤3 (82 %) | B | 188,009 | 8.825e-3 | 2,130 | 0.14× |
| A15 | **pa1r delta-sparsity, endpoint** | reclaim | in-loop group-L2 | w=0.03, 27/58 ep | B | 5,688 | 2.73e-4 | 2,084 | 0.139× |
| A16 | nv1 seg-aware p50 | reclaim | cell safety, **∞ amp** | safest 50 % | B | 107,570 | 7.285e-3 | 1,477 | 0.10× |
| A17 | nv1 thr4 | reclaim | token \|signed\| | ≤4 (91.9 %) | B | 221,288 | 1.6078e-2 | 1,376 | 0.09× |
| A18 | gr1 tok_drop27 (n48) | reclaim | token \|g\| | 27 % (all zero-grad) | pfs1 D1 | 9,212 | 9.17e-4 | 1,005 | 0.067× |
| A19 | **pa1r margin_quant** | reclaim | per-cell quant map | 13-tier | B | 169 | 6.46e-4 | 26 | 0.0017× |
| A20 | co9 QA03 GN/CG seg correction | reclaim | per-flip sidecar | full pop. | in-band | — | — | 1.45 B/flip | 0.878× |
| — | — | — | — | — | — | — | — | — | — |
| A21 | zb1 B-window start | **spend** | training | ep~404 | B | — | — | 444 | 0.030× (33× bargain) |
| A22 | zb1 burn endpoint | **spend** | training | ep399 | bc1 | — | — | 478 | 0.032× (31×) |
| A23 | zb1 B-window end | **spend** | training | ep~440 | B | — | — | 1,332 | 0.089× (11×) |
| A24 | dw1 B control (E2→B) | **spend** | training | 40 ep | E2 | ~5,200 | 1.619e-4 | ~3,210 | 0.21× (4.7×) |
| A25 | **pa1r control_tail** | **spend** | training | 58 ep | B ep440 | 9,621 | 1.736e-4 | **5,542** | **0.369× (2.71×)** |

All values MEASURED at the cited receipt except A24 (ΔB DERIVED from zb1 §3's "250.7→255.9 KB" token
read, ±). A1/A3/A5–A12/A18 are n48 (gr1's own caveat); A2 is the n600 byte-closed confirm.

### §1.2 The shape of the surface, and what moves position on it

- **Coordinate 1 — ADMISSION UNIT, the dominant one.** Near the origin the cell column beats the token by
  **41–113×** (A1 685,053 / A2 50,185 vs best token A6 6,051). By 65–75 % depth the two families converge
  (A5 6,611 vs A6 6,051 = 1.09×) and both are under water. **Take the cell unit AND stay shallow.**
  gr1 names the mechanism: SMEVR conditions on the per-cell temporal mode, so scattered token drops fight
  the coder. **The reclaim unit must equal the coder's conditioning unit.** This is exactly MAIN's worked
  example one surface over: coder conditioning and the paying level set are the same degree of freedom.
- **Coordinate 2 — AMPLITUDE, and one empty 2×2 cell.** nv1 sampled global-magnitude-bounded at ALL cells
  (A4) and cell-safety-gated at UNBOUNDED amplitude (A10, A16). nv1's own diagnosis is exact — *"the band
  lemma's amplitude ≤ flip-distance guard is violated by mode-snap amplitude"* — and it points straight at
  the conjunction **(|signed| ≤ 1) ∧ (safest p%)**, the band-lemma-correct form, which is the **unsampled
  cell of the 2×2**. The sketch-1 falsifier ("field-gated ≤ global-q at matched Δd_seg") therefore fired
  against a form that violates the lemma the sketch was built on. Scope supported: the two sampled cells.
- **Coordinate 3 — DIRECTION, and it is not reciprocal.** At parent B: reclaim m = 10,805 (A4), spend
  m = 5,542 (A25). A smooth reversible frontier would give one number. Ratio **1.95×**.
- **Coordinate 4 — EPOCH.** The spend price rises monotonically along training: 444 → 1,332 → 3,210 →
  5,542 (A21→A23→A24→A25) and has **not reached water** (0.369× at the last point; pa1r §6.2 records
  control_tail's final gates still `COUPLED_DESCENT`, un-exhausted).
- **Coordinate 5 — PARENT RATE.** nv1's B carries 259,407 B (rate 0.173); gr1's pfs1 D1 carries 569,996 B
  (rate 0.380). Fatter parent ⇒ cheaper to reclaim. A2 and A4 differ in unit *and* parent; the two effects
  are not separated by any in-window measurement (**named unmeasured coordinate**).

### §1.3 SURFACE A absorbs the whole "aiming vs noise" cluster

`qp1` QA05, `co9`'s white-jitter laws, `gr1`'s |g| failure and `nv1`'s cell-safety failure were four
separate negatives about whether structure-informed aiming beats unaimed action. Placed on Surface A they
are one statement along coordinate 1: **aiming pays iff the aiming unit is at-or-above the coder's
conditioning unit.**

| aim | unit | result |
|---|---|---|
| qp1 QA05 atlas flip-mass × colour, rank-1 | **pixel** (below any coder unit) | best structured +314 vs random max +446, mean+2σ +392.8 → falsifier fired |
| gr1 token \|g\| | **token** (below cell) | A18 m = 1,005 = 0.067× water |
| nv1 QA80 flip-distance | **cell**, but amplitude unbounded | A10/A16, 0.19×/0.10× |
| **gr1 cell \|g\|-sum** | **cell = the SMEVR conditioning unit** | **A2 3.34× water — pays** |

Two things Surface A keeps that the four verdicts discarded:

- **qp1's family MEAN, not its max.** structured mean −39.7 vs random mean −90.3 ⇒ the aimed family is
  **+50.6 flips better on average**. With n=36 vs n=72 and σ=241.6, SE(Δmean) ≈ 49 ⇒ **≈ 1.03 σ** — not
  significant, but it is the unbiased statistic and it points opposite to the headline, which compared
  best-of-36 against best-of-72 (an order statistic over 2× the draws). Both facts are in qp1's own table.
- **qp1's amplitude ladder is truncated at its own optimum.** Ladder ±{1,2,4}; the best structured
  candidate is `atlas_lum@+4` and the best random is `rand_08@+4` — both at the **maximum tested
  amplitude**, with rand_08@+2 = +238 < rand_08@+4 = +446 (monotone in amplitude over the sampled range).
  Amplitudes 8 and 16 are unsampled. MEASURED from qp1 §1.1.

### §1.4 What Surface A says about where to stand (DERIVED, from two MEASURED rates)

Reclaim at A4 (10,805 B per 1e-4), then re-spend at ≤ A25 (5,542 B per 1e-4): reclaim 62,452 B for
+5.78e-4, then buy back 5.78e-4 for ≤ 32,033 B ⇒ **net ≈ 30,419 B at constant d_seg ≈ 0.0203 S**.
The buy-back price is an *upper* bound: the snapped parent sits at d_seg 0.005693, i.e. **earlier** on the
descent than control_tail's range, and zb1 §3's price law is monotone-rising in progress. The untested
risk is that retraining refills exactly the snapped deltas — that is the one thing to measure, not the
sign of the arithmetic. nv1 queued this as reformulation row 1 and labeled it *"doubtful"*; the two
measured rates make it **positive-by-arithmetic and untested**, which is a different state.
**Also live and cheap:** A2's n600 confirm was run at ONE grid point and its *average* m is 3.34× water,
so the n600 crossing sits deeper than drop50; the n600 marginal past drop50 is unmeasured (drop55/drop60).
(At n48 the crossing is correctly bracketed between drop50 and drop63 — gr1's knee choice is sound there.)

---

## §2 SURFACE B — REACHABILITY / PROJECTION RESIDUAL

**The object.** How much of a target field lies in the range of the generator's currently-free directions.
Coordinates: (what is free · target manifold · space the objective lives in · output chart).

### §2.1 THE COLLAPSE — two headline "capacity" numbers are one dataset constant

$0 recomputation, `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (`lstars`, 600×384×512 int64),
checked digit-for-digit against pj1's committed receipt
`/Volumes/VertigoDataTier/pact/ddm_pj1_20260730/warm_l2/gate/p1_receiver_realized_verdict.json`
(schema `ddm_pb1_p1_receiver_realized_verdict.v1`, n_pairs 600, archive sha `f2c2c377…`,
receiver sha `bb8fa481…`, `evidence_axis "[macOS-CPU advisory]"`):

| class | GT area frac (MEASURED) | const-Undriv d_seg (DERIVED) | **pj1 receipt `per_class_d_seg`** | abs diff | fp1 f′ |
|---|---:|---:|---:|---:|---:|
| Road | 0.23233240763346355 | 0.23233240763346355 | 0.23233240763346355 | **0.00e+00** | 0.230 |
| Lane | 0.00585461934407552 | 0.00585461934407552 | 0.00585461934407552 | **0.00e+00** | 0.006 |
| Undriv | **0.49517551845974395** | 0.0 | 0.0 | **0.00e+00** | 0.0001 |
| Movable | 0.01237932840983073 | 0.01237932840983073 | 0.01237932840983073 | **0.00e+00** | 0.012 |
| MyCar | 0.25425812615288629 | 0.25425812615288629 | 0.25425812615288629 | **0.00e+00** | 0.251 |
| **total** | | **0.50482448154025605** | **0.50482448154025605** | **0.00e+00** | **0.499366** |

**Bit-exact in float64 on all five classes and on the total.** The receipt's own stated convention —
`per_class[c] = sum over pairs of |{px: gt==c and realized!=gt}| / total_px` — makes the identity explicit:
pj1's fitted state is wrong on *every* pixel whose GT is not Undrivable and right on *every* pixel whose GT
is Undrivable, across all 117,964,800 pixels. That is the constant-Undrivable predictor, exactly.

fp1's f′ read at the same precision from `ddm_fp1_20260731/fprime_solved/realized_verdict.json`
(`d_seg_mean` 0.4993662346733941, n_pairs 600):

| class | const-Undriv | fp1 f′ receipt | f′ as % of const |
|---|---:|---:|---:|
| Road | 0.232332407633 | 0.229923960368 | 98.96 % |
| Lane | 0.005854619344 | 0.005849821303 | 99.92 % |
| Undriv | 0.000000000000 | 0.000101123386 | — (was 0) |
| Movable | 0.012379328410 | 0.012305230035 | 99.40 % |
| MyCar | 0.254258126153 | 0.251186099582 | 98.79 % |
| **total** | **0.504824481540** | **0.499366234673** | **98.92 %** |

**After 50 epochs of converged/plateaued training (CE 0.689→0.550), the head sits 98.92 % of the way to
being a constant-majority predictor** — it recovers **1.08 %** of the distance from that corner to zero,
and its one genuinely non-degenerate behaviour is that it now *also* misses 1.01e-4 of Undrivable (which
a pure constant predictor gets free). This is collapse, quantified, not a capacity reading.

**Consequence (DERIVED).** Both probes ended at the same degenerate corner: predict everything Undrivable.
That corner's d_seg is `1 − p(majority) = 1 − 0.495176`, a property of the *dataset*, carrying zero
information about token grammar, head size, trunk decodability, objective form, or photometric range.
pj1 already declared its number confounded and closed the formulation — correct, and its stated mechanism
("argmax destruction from an unreachable-target objective") is directionally right; the sharper statement
is that the output is *exactly* the constant predictor, which is a **collapse signature**, not a range
gradient. fp1 reported f′ = 0.4994 as evidence of a "SECOND wall … the frozen trunk features are NOT
small-conv-decodable"; a value at the majority-class baseline is the classical imbalanced-CE collapse.
fp1's own apparatus caught the *init*-time version of this (dead Lane channel, rc=3 abort, cured by
frequency calibration) — **the cure was applied at init only; the 50-epoch loss carries margin weighting,
not class weighting** (fp1 §1/§3). So the trunk-decodability claim rests on a run whose observable is
collapse; scope supported is INSTANCE(this head, this loss), and the *reformulation* it routes away from
(head work) is not closed by it.

**What is NOT touched.** fp1's F1 receiver floor **0.008305** paints the *GT* argmax with solved
prototypes, so no head and no collapse enters it; `f′ ≥ F1` by construction. **BR-D's firing condition
(`f′ > 2e-3`) is satisfied by F1 alone and is sound.** The 07-31 branch decision does not depend on the
collapse. Per-class F1 `[Road .005169, Lane .001014, Undriv .0000222, Movable .001979, MyCar .000121]`
(Road 62 % + Movable 24 % of the floor) is a clean boundary-bleed signature and a genuine measurement.

### §2.2 Every reachability point, placed

| point | free params | target | space | residual | note |
|---|---|---|---|---|---|
| pj1 f | tokens (head frozen) | v10 dark solve frames | photometric L2 | **0.504824 = collapse corner** | render dark-floor **67.95** vs target mean **20.7** ⇒ manifold distance **3.28×** (MEASURED, pj1 R1/R4) |
| ax1 §0 pre-registration | — | — | — | predicted f ∈ [7e-4, 2e-3] | missed ~250×; the falsified premise ("all residual is capacity by construction") is precisely the assumption distance = 1.0 |
| dw1 A | tokens + trunk | same field, SegNet logits | scorer space | KD loss 19.8→8.1 **fell**; realized d_seg **rose** +1.37e-5/ep | reachable in soft space, unreachable through uint8-STE/R |
| dw1 C − dw1 A | + per-channel output residual | same | scorer space | **C − A = −5.73e-5 = 1.92× the 2.99e-5 noise, FAVOURABLE** | the only available contrast on the chart DOF is positive; unreported |
| fp1 f′ | 3×3 head on frozen trunk | QA75 field | scorer CE | **0.499366 = collapse corner** | see §2.1 |
| **fp1 F1** | **none (GT argmax)** | — | realized | **0.008305** | airtight, collapse-immune, load-bearing |
| pi2 fixed k≤8 | shared basis | ∂p0/∂pixel | image space | captures **36 %**; k90 = 43/53 | isotropic expectation = 8/53 = 15 % ⇒ **2.4× above chance**: under-dimensioned, not structureless |
| pm1 QA53 pitch | 1 global const | — | pose | grid {−0.02, 0, +0.02}: 22.734 / 25.49 / 84.46 | see §2.3 |

### §2.3 The level set, and what moves it

The verdict word on this surface flips when the target enters the range of the free directions.

- **The chart DOF is ONE degree of freedom appearing on three surfaces and is never varied alone.**
  pj1's wall is an additive output-range floor (67.95 vs 20.7). dw1 built `head_relax_gain` — *"a trainable
  per-channel output residual (init 0 ⇒ warm-start-EQUIVALENT to sigmoid×255)"*, unit-tested
  render-identical-at-init — which is **exactly the parameter that moves an additive range floor**. It was
  run only as C = A + relax. The 2×2 `{relax} × {distill}` has **(relax, no-distill) empty**, and the one
  available main-effect contrast (C − A) is **favourable at 1.92× noise**. dw1's conclusion *"the rgb
  output chart is NOT the binding constraint"* is drawn from a contrast taken only inside distill; it
  requires the interaction to be absent, which is unmeasured. Cost to fill: one 40-epoch warm window,
  already-built lever, already-sealed harness.
- **pm1 QA53's grid brackets an interior minimum that is not at the grid point.** f(0)=22.734,
  f(+0.02)=25.49, f(−0.02)=84.46 — a **22× asymmetric** neighbourhood, and 57/112 pairs prefer +0.02/+0.04.
  The quadratic through the three points has its vertex at **p* ≈ +0.0046 rad**, between grid points and
  unsampled (DERIVED; the quadratic is a poor fit given the asymmetry, so treat p* as "the grid does not
  resolve the minimum", not as a predicted optimum). The verdict *"pitch = 0 IS the aggregate global
  optimum"* is a grid-minimum claim. The probe is a $0 transfer sweep on an existing harness.
- **pi2's re-aim is the model citizen of this surface.** A fixed basis fails at k=8 (36 %), and the memo
  replaces it with a receiver-DERIVED per-pair basis (0 shipped bytes, ~1–2 coefficients) instead of
  killing the family. It changed the coordinate rather than reporting the verdict. No debt.

---

## §3 SURFACE C — STRUCTURE vs WHITE IN PER-PAIR SOLVER-OUTPUT FIELDS

**The object.** The temporal / spectral structure of fields produced by independent per-pair GN solves.
Coordinate for the temporal axis: `r = std(diff)/std(value)`; equivalently lag-1 autocorrelation
**ρ = 1 − r²/2** (DERIVED from `Var(xₜ−xₜ₋₁) = 2σ²(1−ρ)`). **The level set is r = √2 = 1.41421 (ρ = 0).**

| field | source | r (MEASURED) | **ρ (DERIVED)** | verdict as written |
|---|---|---:|---:|---|
| pose p_star **dim 0** | kl1 §0 | **1.14** | **+0.350** | "temporally WHITE" |
| pose p_star dims 1–5 | kl1 §0 | 1.40 | +0.020 | "temporally WHITE" |
| exposure `a` | v4c §9 | 1.42 | −0.008 | "NO-LAW" |
| exposure `b` | v4c §9 | 1.37 | +0.062 | "NO-LAW" |

**Every measured field is at or below the white level set except dim 0, which sits at ρ = +0.35.** The
family label is exact for 5 of 6 pose dims and for both exposure channels, and **false for dim 0** — the
one dim `pi2` measured as the only f16-marginal one (**0.040 S per ULP**, kl1 §9).

**The rate verdict on dim 0 is nonetheless SOUND, and here is why (DERIVED):** a first-order predictor at
ρ = 0.35 leaves residual variance `1 − ρ² = 0.877`, i.e. **½·log₂(1/0.877) = 0.0944 bits/sample** ⇒ ≈ 7 B
over 600 samples. kl1's byte-plane conclusion is not in danger. **The verdict that is NOT supported is the
distortion one.** kl1 §9 already found the order-0 (shared DC mean) form of exactly this and elevated it —
*"the shared-mean (DC) component that LOSES on rate WINS on distortion"*, +4 effective bits on dim 0 ⇒
~0.03 S potential. The order-1 (ρ = 0.35) form is the same lever one rung up and is unexamined. **The same
standardization correction kl1 applied to the SVD axis (B3, "scale artifact") was never applied to the
temporal axis** — the temporal ratio was read per-dim but the family verdict was taken over the pooled set.

**The SVD half of this surface, standardized (the coordinate kl1 itself introduced):**

| field | standardized top-1 energy | isotropic (1/6) | ratio |
|---|---:|---:|---:|
| kl1 two-plane tail | 0.714 | 0.167 | **4.3×** |
| pi2 image-space per-dim Jacobian | 0.353 | 0.167 | 2.1× |
| v4c QA50 re-solve correction | 0.221 | 0.167 | 1.32× |
| v4b QA50 (raw only; standardized not reported) | 0.930 raw | — | **unmeasured coordinate** |

v4c §7 explicitly re-applied kl1's standardization to the new field and got **[0.221 … 0.130] = flat**, and
closed QA61 on it. That is the correct operation and it is the reason QA61's closure is sound.
v4b §7's `σ = [30.55, 7.47, …]`/"93.0 % energy" is the **raw** spectrum — the exact quantity kl1 B3 proved
is a scale artifact — and it was used to name "the next systematic axis is rank-1 forward-speed". v4c
retired that premise 3 h later on a different field; the v4b field's standardized spectrum is unmeasured.

**What moves this level set:** the axis the field is read on (rate vs distortion — the same field is white
for one and structured for the other), and standardization (raw spectra manufacture rank-1 laws).

---

## §4 SURFACE D — POSE-BASE PHOTOMETRIC LEGIBILITY

**The object.** d_pose achievable by one solver family, as a function of the base it solves on. Every
in-window pose negative is one base coordinate on this surface.

| base | f1 mean | d_pose (MEASURED) | pose term √(10·d) | source |
|---|---:|---:|---:|---|
| v4d line (07-31, out of window; cited) | — | 0.0086 | 0.293 | memory `box_retired…` |
| **v4c cell_drop50, re-solved + rung-B** | — | **0.010384** | **0.322248** | v4c §1/§4 |
| v4c pre-photometric | — | 0.016926 | 0.41141 | v4c §3 |
| v4b Knee-A static two-plane | — | 0.0636508 | 0.797815 | v4b §1 |
| pfs1 D1 | — | 0.22144216 | 1.488093 | qp1 §0 |
| pb1 control (solver control) | **51** | 0.1720 | 1.311 | ps1 control |
| **ps1 B-control, S2 terminal solve** | **107** | **20.4075** | **14.285** | ps1 ladder |
| ps1 B-control, S1 warp | 107 | 27.8185 | 16.679 | ps1 ladder |
| ps1 B-control, S0 zeros stub | 107 | 160.0998 | 40.012 | ps1 ladder |

**Range across one solver family: 0.0086 → 20.41 = 2373×.** The solver's *relative* gain is nearly
identical at both ends — pb1 0.2474→0.1720 = **1.44×**, B-control 27.82→20.41 = **1.36×** — so essentially
all 2373× is the base coordinate, exactly as ps1 states. **This is a base-selection surface, not a
solver-capability surface.**

**Two coordinates the aggregate verdicts hide:**

- **ps1's distribution.** n600 mean 20.4075 but **median 7.55**, max 156.55, 502/600 improve (ps1 ladder).
  A mean 2.7× its median is a mixture; "structurally walled" is a population statement over a distribution
  whose per-pair split is not reported. Whether a legible subpopulation exists on B is unmeasured and $0 to
  read from `ps1_ladder.partial.jsonl` (600 rows, already on SSD).
- **Nobody chose a base for pose, the largest-weight axis.** ps1's parent B was selected by the seg arm on
  seg; v4c's base was adjudicated on seg+rate with pose measured afterwards as damage (+0.061 S, found
  recoverable). Standing axis-weight read (memory `pose_is_the_largest_axis_on_the_own_vehicle_1_24_S`):
  **pose ≈ 1.24 S > seg 0.431 + rate 0.239 combined.** ps1's own routing directive — pose must be born
  legible in the burn (QA80 + JOINT descent) — is a *burn-3 config* consequence; the *base-selection*
  consequence for the live composed line is not drawn anywhere in-window.

**What moves this level set:** the base, and only the base. And the base is currently chosen by two other
surfaces' criteria.

---

## §5 SURFACE E — THE READING HORIZON (when we looked)

Not a physical surface; the coordinate every other surface is read at. Every in-window verdict is a
function of it, and the underlying observables are **non-monotone** in it.

| observable | short horizon | long horizon | consequence |
|---|---|---|---|
| dw1 loss-form race | 12 ep: kd_logits+attack **0.0050507** wins | 40 ep: **that same winner reverses** (+1.37e-5/ep, A1 refuse @ep430) | dw1 §4 states it: *"The 12-ep mini-race horizon ended exactly at A's transient dip"* |
| dw1 A trajectory | ep409 **0.004995, ahead of B** | ep430 0.005463, refused | the transient is a schedule object; anneal-to-CE named in §5, never priced |
| dw1 race top-2 | winner 0.0050507 vs argmax_ce uniform **0.0050600** — Δ 9.3e-6, **inside** the 2.99e-5 noise floor | argmax_ce at 40 ep **never run** | the race did not resolve its top two, and the un-run runner-up is the one form with **no soft dark-knowledge** — the exact mechanism dw1 blames for the reversal |
| pa1r delta arm | refused ep469 (27 of 58 force-epochs) | control did 0.00527→0.00494 in its **last 35 ep** | the headline "full budget" favourability 0.139 compares 27 ep to 58 ep; pa1r flags it |
| gr1 sample size | n48 d_seg 0.003947 | n600 0.004310 (**+9 %**) | gr1 states it; the n600 confirm covers one grid point |
| zb1/pa1r spend price | 444 B/1e-4 | 5,542 B/1e-4 | Surface A coordinate 4 |

**Scope note supported by this surface.** dw1's verdict is written FORMULATION on the strength of *"all 3
loss forms × attack raced at their own optima, 6 configs"* — but the race ran at 12 ep, dw1's own A-reversal
proves the 12-ep ranking is unreliable at this operating point, and the top two were separated by less than
the noise floor. The evidence supports **INSTANCE(kd_logits+attack, 40 ep, from E2)**; the 5 unraced-at-40-ep
forms — argmax_ce foremost — are the untested reformulations that belong with the negative.

---

## §6 UNIFICATION — SHARED COORDINATES ARE SHARED DEGREES OF FREEDOM

| DOF | Surface A | Surface B | Surface C | Surface D | Surface E |
|---|---|---|---|---|---|
| **granularity / unit** | coord 1: token vs cell (41–113×) | — | per-pair vs shared-DC (kl1 §9) | — | n48 vs n600 |
| **coder conditioning unit** | sets where reclaim pays | — | sets where a "law" pays | — | — |
| **output chart / head range** | — | pj1 wall 3.28×; dw1 `head_relax_gain` | — | f1 mean 51 vs 107 | — |
| **standardization** | — | — | SVD **and** temporal | — | — |
| **amplitude** | coord 2 (band lemma) | — | — | — | — |
| **epoch / horizon** | coord 4 (price rises 444→5,542) | dw1 A reversal | — | — | the coordinate itself |
| **base / parent** | coord 5 | pj1, fp1 parents | — | the ONLY coordinate | — |

Three couplings are load-bearing:

1. **Coder conditioning unit ≡ the paying aiming unit ≡ the paying reclaim unit.** One knob sets Surface
   A's position and the entire aiming/no-aiming cluster (§1.3). This is MAIN's worked example (coder
   quality ≡ band edge) instantiated on a second surface, which is evidence the coupling is general.
2. **Output chart is one knob on Surfaces B and D.** pj1's range wall (67.95 vs 20.7), dw1's built-and-
   unfired `head_relax_gain`, and ps1's legibility gap (f1 mean 107 vs 51) are the same additive output-
   range degree of freedom read in pixels, in code, and in pose. It was measured on none of them alone.
3. **Standardization is one operation with two applications.** kl1 introduced it for the SVD axis and
   proved a rank-1 "law" was a scale artifact; v4c re-applied it correctly to a new field (QA61 sound);
   nobody applied it to the temporal axis, where per-dim it separates ρ = 0.35 from ρ ≈ 0. v4b's raw-SVD
   "93 % rank-1" is the un-standardized reading that v4c later retired on a sibling field.

**One organisational fact, MEASURED, that the surfaces make visible.** On Surface A the day's *winning*
encode-side reclaim (A2, 3.34× water, byte-closed n600) landed at **07-30 01:44** (`7f54c339ee`). The
day's "encode-side reclaim is dead / bank EMPTY / co-location is structural" verdict landed at **22:49**
(`838273e222`) and the in-loop twin at **23:09** (`083726730b`). `grep -c gr1` in nv1 = **0**, in pa1r =
**0**; `grep -c nv1` in gr1 = **0** (gr1 predates them). Both later memos are otherwise recall-disciplined
(long STORES-CONSULTED blocks). The three points are on one surface at three coordinates; read together
they say the reclaim direction pays at the cell unit and not at the token unit — which is a coordinate,
not a family verdict.

---

## §7 WHERE WE STAND, AND WHICH WAY EACH SURFACE FALLS

Standing state (MEASURED / cited): pointer 0.1910828242 [contest-CPU] UNMOVED; official bar 0.172141.
The live own-vehicle exact-protocol line moved **20.27 → 1.5343 (v4b, gate-measured) → 0.992972 (v4c) →
0.9639878 (v4d, 07-31)** — i.e. **the B-control lineage ps1/nv1/pa1r/dw1 measured on is not the live
composed line**; the live line is the pfs1→v4x composition whose base is gr1's cell_drop50. Axes on that
line: seg 0.431, pose 0.322 (v4c) / ~0.29 (v4d), rate 0.239.

| surface | where we stand | direction it falls |
|---|---|---|
| **A** exchange | B: spend 0.369× water, reclaim 0.72×. Live line: cell_drop50 at 3.34× water, n600-confirmed at one grid point | **spend more on the training trajectory (still 2.71× bargain, un-exhausted); reclaim at the cell unit and only shallow; alternate the two** |
| **B** reachability | two headline probes at the collapse corner; one airtight receiver floor 0.008305 | **vary the chart DOF alone (relax without distill); re-read any head probe against the majority-class baseline before calling it capacity** |
| **C** law-vs-white | rate side closed and sound; distortion side open on dim 0 (ρ = 0.35, 0.040 S/ULP) | **read the distortion axis; standardize the temporal axis per-dim; v4b's raw SVD needs standardizing** |
| **D** pose base | 2373× across bases; base chosen by seg and rate criteria only | **choose the base for the largest-weight axis; read ps1's per-pair distribution ($0, on SSD)** |
| **E** horizon | verdicts read at 12 ep, 27/58 ep, n48 | **re-read the un-run long-horizon arms before promoting a formulation scope** |

---

## §8 SOUND NEGATIVES, CALLED SOUND (with the reason)

These carry no discarded signal I can find, and their reformulations are already correctly aimed:

- **fp1 F1 receiver floor 0.008305 and the BR-D firing.** Uses GT argmax ⇒ `f′ ≥ F1` by construction;
  collapse-immune; n600; per-class decomposition given; verdict_scope FORMULATION and the reformulation
  queue is re-routed to the *receiver* rather than the charter default. Airtight.
- **v4c rung-A global g\* = 0.0.** Grid {0, 0.5, 1.0} → 0.010384 / 0.011843 / 0.016223. Quadratic vertex at
  g ≈ +0.0003 (DERIVED) ⇒ the boundary IS the optimum; and the sign is derived free from yaw, so a global
  constant *must* be ≈0 by symmetry. The per-pair form is measured **positive** (0.009533, −0.0135 S at
  ~150 B) and routed to v4d rather than discarded. Exemplary.
- **v4c QA58 and QA61.** QA58: ρ_a = −0.008, ρ_b = +0.062 — genuinely white; byte-plane 1,804 vs AR(1)
  1,935 (7 %). QA61: standardization re-applied to the *new* field, [0.221 … 0.130] flat. Both correct
  operations on the correct coordinate.
- **pi2's re-aim.** Changed the coordinate (fixed → receiver-derived per-pair) instead of killing QA47,
  and the replacement is strictly cheaper (0 shipped bytes).
- **gr1's cell knee at n48.** The n48 marginal crossing is correctly bracketed between drop50 and drop63
  (m falls 124,023 → 7,637 across water 15,018); drop50 is the right grid choice at n48. Only the n600
  extension is open.
- **kl1's rate verdicts.** Byte-plane is at the alphabet floor; the ρ = 0.35 residual is worth ≈ 7 B. The
  memo also raced the laws rather than assuming them, proved losslessness before reporting sizes, and
  flagged its own distortion column in §9.
- **zb1 §5's own Assumption-Adversary catch** (via gc10): *"Two points define a line, not a hull"* — the
  hull claim was scoped down by the apparatus, in-window, without prompting. The instrument worked.
- **nv1's decisive reframe c ≠ S** (product invariant vs additive score) is correct and important: it
  retracted a ×1.9 "hull move" that was a multiplicative artifact, and pa1r immediately re-instrumented its
  analyzer to the additive verdict with a test encoding the finding. That is the right direction of travel.

---

## §9 ROUND-1 ADVERSARIAL REVIEW OF THIS MEMO

- **The collapse identity could be coincidence — closed.** I did not stop at the memo text: I read pj1's
  source receipt and compared at full float64 precision. Six independent agreements (five classes + total)
  at **abs diff 0.00e+00** is not coincidence. The receipt is a real gate artifact (n_pairs 600, archive
  and receiver sha256, wall 143.4 s, `generated_by tools/pb1_receiver_realized_verdict.py`), so the vector
  was produced by the verdict path, not derived from the prior. **Labeled: VERIFIED AT SOURCE.** fp1's f′
  receipt was opened too (`fprime_solved/realized_verdict.json`, n_pairs 600): 98.92 % of the constant
  corner, per-class 98.79–99.92 %. Both identities are now source-verified, not memo-quoted.
- **The Surface-A arbitrage assumes both marginal rates survive composition.** They are measured at one
  point each, in opposite directions, on the same parent. Composition is untested and could fail by exactly
  the co-location mechanism nv1 measured (corr −0.51). I claim the *sign of the arithmetic*, not the
  outcome; nv1's own queue row 1 is the same object and I am changing its label from "doubtful" to
  "positive-by-arithmetic, untested", which is a scope claim about evidence, not a prediction.
- **A2 vs A4 confounds unit with parent.** Stated as a named unmeasured coordinate in §1.2; I did not
  separate them and no in-window measurement does.
- **ρ = 1 − r²/2 assumes stationarity.** For a non-stationary field the identity is approximate. The
  conclusion (dim 0 is materially more predictable than dims 1–5) survives any reasonable violation because
  it is a *comparison* at fixed method, but the absolute 0.350 is model-dependent. **Labeled DERIVED.**
- **Am I replacing one binary with another?** The risk is real for §2.1: "not a capacity measurement" reads
  as a verdict. It is a coordinate statement — both probes landed at one specific point of the output
  simplex whose value is a dataset constant — and the constructive consequence (re-read head probes against
  the majority-class baseline; vary the chart DOF alone) is named rather than a kill.
- **My "nobody cites gr1" claim is an existence claim.** Scope stated exactly: `grep -c 'gr1'` over the two
  memo files = 0. I did **not** exhaustively search commit messages, the DAG, telemetry, or the ledger for a
  gr1 citation in the nv1/pa1r work, so the honest form is: *the two memo bodies do not cite gr1*, not
  *nobody knew*.
- **Would my analysis pass if the memos were fine?** §8 exists to answer this: 8 of the 44 verdicts are
  called sound with mechanism, and two of the surfaces (C rate side, and the v4c cluster) are closed
  correctly. The audit does not find a defect everywhere it looks.

---

## §10 CUSTODY, LABELS, SCOPE

- **Recomputed at $0 here:** class prior + constant-predictor d_seg from
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (`lstars`, 600×384×512, MEASURED); water level
  15,018.2 B per 1e-4 d_seg = 1.27311 B/flip (MEASURED-by-definition from `evaluate.py` coefficients);
  every `m` in §1.1 (DERIVED by division from the cited memos' ΔB and Δd_seg); ρ from r (DERIVED);
  the QA53 quadratic vertex and the qa05 mean-difference SE (DERIVED).
- **Axis:** every quoted result is `[macOS-CPU advisory]` or `[macOS-CPU/MLX advisory]` per its source
  memo; nothing here is a score claim; `score_claim=false`; `promotion_eligible=false`.
- **verdict_scope of this audit:** it re-scopes **no** prior verdict by fiat. It places each on a surface,
  names the coordinate the evidence reaches, and names the unmeasured coordinates. Where I state a scope is
  narrower than written (dw1 FORMULATION → INSTANCE(kd_logits+attack, 40 ep); nv1 two-of-four 2×2 cells;
  fp1's second wall INSTANCE(this head, this loss)), the reason is given inline and the original memo's
  own numbers are the evidence.
- **No file outside this memo was modified.** No scorer slot, no training, no dispatch, no `/tmp` in any
  evidence path.

pointer 0.1910828242 [contest-CPU] UNMOVED · `[no-triality]` `[p0-ledger-ok]`
