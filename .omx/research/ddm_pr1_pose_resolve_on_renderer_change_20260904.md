# ddm_pr1 — the terminal pose re-solve recovers 16.42x, not the 8.0x that was transferred; the door narrows 81x -> 41x -> 26x and stays shut; and the assumption underneath the whole arithmetic is measured FALSE

Arm: `ddm_pr1_pose_resolve_on_renderer_change`. Tokens: `[no-triality] [p0-ledger-ok]`.
Craft contract: `docs/operating_manual_craft_handoff.md`.
Axis of every measured row below: **`[macOS-CPU advisory]`, frozen CPU-torch PoseNet, DALI-lineage GT**,
unless it cites the contest-CUDA T4 receipt. `score_claim=false`, `promotable=false`.

## ANSWER FIRST

1. **The re-solve was never run, and it recovers 2.05x more than the number that stood in for it.**
   `ddm_ft1`'s closure substituted `ddm_jg5`'s **8.0x** TOKEN-edit carrier recovery for a renderer
   change. MEASURED at n600 on the shipped carrier's own solver: **16.42x** mean-based, **3,424x**
   median per pair, 598 of 600 pairs improved. `d_pose` falls **0.015482008638444032 -> 0.000942729261043249**.

2. **The operator's pre-registered intuition HOLDS. The door stays shut anyway, and those are different
   events.** Post-re-solve coupling **k_post = 13.82** against the charter's success band of `< 20`, and
   the charter's falsifier (recovery `< 3x`) did NOT fire. But the payable bar at a 25% seg cut is
   **k_post <= 0.20997**, so the move is still **41.5x over the ceiling** — down from ft1's **80.7x**,
   by a factor of 1.95, and down to **26.5x** once §12's selector is added on top. I pre-registered that
   these two verdicts are separate and would be reported apart; they are.

3. **Four of my own seven numeric predictions were wrong, every one of them in the optimistic
   direction; three held.** WRONG: `d_pose_after` in [1e-05, 1e-04] (landed 9.43e-04, an order of
   magnitude high); mean-based recovery in [30x, 1000x] (landed 16.42x, below the band); payability
   "a coin-flip, under even odds" (it is 41.5x over); a base control that would also improve (it
   recovers 1.0045x — nothing). RIGHT: median per-pair recovery > 100x and at least 10x above the
   mean-based one (3,424x and 208x); `k_post` in [0.1, 20] (13.82); the charter's falsifier does not
   fire. **The charter's prediction, which was the operator's, is the one that held cleanly.**

4. **The residue that survives is a REPRESENTATION limit, not a search limit.** Every one of the 600
   pairs demands a Gauss–Newton step larger than `ddm_up2`'s ±2 search radius; the median demand is
   **444 int12 code units** and **9.67% of pairs demand more than the entire 4,095-unit lattice span**.
   542 of 600 stop at `no_improving_step` — the receiver evaluating a real proposal and refusing it.
   After the re-solve, **ten pairs own 69.0% of the n600 pose mean**.

5. **The terminal re-solve is not free: it costs +125 archive bytes.** ft1's FIRE_ORDER calls FO-2
   "0 archive bytes". Measured through the CAP1 Rice pricer whose control reproduces the shipped
   payload exactly (78,628 bits): 6,847 changed coordinates over 598 pairs re-code to
   **9,829 -> 9,954 B, ΔB = +125, ΔS_rate = +8.323e-05**.

6. **THE FINDING THAT OUTRANKS THE REST: the assumption under the whole closing arithmetic is measured
   FALSE.** The registered law names local linearity of the realized map as its one stated-not-measured
   premise and calls it "the cheapest falsification available". I built it: the point-reflection of the
   candidate step through the shipped weights (realized, cosine **−0.941**, norm ratio **1.014**). The
   reflected object does not lower d_seg — it raises it **21.55x more** than the forward step did
   (Δd_seg **+0.0014595** vs **+6.774e-05**) — and its coupling is **22.19**, **10.3x** away from the
   forward 228.45. Along the one direction anyone can realize and export, **d_seg rises both ways**. So
   the seg-only renderer axis is closed less by the pose price than by the seg gain being unreachable
   from these weights along this direction at all.

7. **A live, cheap, unclaimed lever fell out of the mechanism.** The residue lives where the 12-dim
   carrier cannot reach; the receiver already ships a second per-pair actuator on the same frame. Swept
   over all 600 pairs of the **shipped afr1 object**: 39 pairs beat their current selector mode by >1%,
   one shipped-active pair is actively harmful, and the whole adoption prices at **+36 B for a net
   −1.032e-04 S** `[macOS-CPU advisory projection]`. It needs an encoder, a batch-8 re-measure and a T4
   row before it is anything — and it is the one item in this memo with a direct path to a lower exact
   score. On the damaged candidate the same sweep buys a further **1.57x** on top of the re-solve,
   **50.3%** of it from the ten pairs the carrier provably cannot reach.

**Pointer: UNMOVED.** No exact row was bought by this arm.

## 1. THE PRE-REGISTRATION, READ OUT BEFORE THE NUMBERS

Two documents were committed before any aggregate existed, and both are quoted here in full before a
single result appears.

**The charter** (`.omx/research/charters/ddm_pr1_pose_resolve_on_renderer_change_20260904.md`, commit
`77ad212ad`, written before any measurement):

> Pre-registered prediction (operator's intuition): the re-solve recovers most of the renderer-induced
> pose damage (post-re-solve coupling < 20, i.e. > 10× recovery). Falsifier: recovery < 3× (coupling >
> 70) — then the closure stands with the re-solve measured, not assumed.

**This arm's own pre-registration** (`.omx/research/ddm_pr1_prereg_20260904.md`, commit `a4b57f99c`,
written with ~13 of 600 solver rows visible and no aggregate) carried seven numeric predictions and one
structural warning. The warning is the one that matters:

> At a 25% seg cut the payable ceiling is `d_pose ≤ 1.694e-05`. Writing the post-re-solve coupling as
> `k_post = (d_pose_after − d_pose_base)/|Δd_seg|`, a 25% cut is payable **iff** `k_post ≤ 0.2098`.
> **That bar is 95× stricter than the charter's own success band of `k_post < 20`.** The two events are
> NOT the same event, and I pre-commit to reporting them separately.

I also recorded, before the aggregate, what would make me wrong in a way that matters: a `d_pose_after`
below the AFR1 base would make `k_post` negative and the coupling language meaningless for this
candidate; and a base control that recovers as much as the candidate would mean the "recovery" is the
shipping chain's own unclaimed slack rather than a repair of renderer damage.

**What I had already seen when the pre-registration was written, stated rather than hidden:** about 13
per-pair solver rows and a 7-pair timing smoke. The per-pair recoveries were already visibly
heavy-tailed (1.2×, 4.4×, 9.8×, 19×, 2.8e3×, 3.7e3×, 3.0e4×, 1.5e6×, 7.6e6×). So the pre-registration is
clean for every AGGREGATE below and is NOT clean for the shape of the per-pair distribution. It says so
in its own text.

**The solver was fixed before the result, for the reason that matters** ([[m116]]: never let the
derivation control the falsifier it is tested by). The charter names
`experiments/ddm_up2_shipping_pose_solve.py` as the reference form. I did not use its solver, and the
reason is in `ddm_jg5_pose_resolve_on_edited_renders_20260819.md` §4, verbatim:

> This is the same shape as `ddm_up2`'s ±2 search radius — the very defect br1 was built to escape.

Running up2's `solve_pair_realized` here would have measured the SOLVER's ceiling and reported it as the
CARRIER's, in an arm whose whole output is "how much does the re-solve recover". The solver used is
`ddm_jg5.refine_pair` — br1's damped Gauss–Newton on the shipped 12-dim basis and signed-int12 lattice
with the ±2 polish, under jg5's DERIVED materiality stop — which is the OPTIMAL FORM of this family and
is the same tool the shipping chain's own last carrier solve used. `--solver up2` remains available as a
labelled control. **MEASURED** justification for the choice, not stylistic: §5 reports the Gauss–Newton
step each pair's residual demands, and the great majority demand more than 2 code units.

## 2. THE INSTRUMENT, AND THE CONTROL THAT MAKES IT MEAN SOMETHING

The instrument composes the pair exactly as the shipped receiver does, **verified at source**:

| frame | built by | source |
|---|---|---|
| `2p+1` (semantic master) | `interpolate(SemanticTokenRenderer(tokens, idx), (874,1164), bilinear, align_corners=False).clamp(0,255).round()` | `submissions/semantic_joint_ctxmix/cpr1/inflate.py:315-326` |
| `2p` (pose carrier) | `interpolate((127.5 + 64·basis@coeff/√12).clamp.round, (874,1164), bicubic, align_corners=False).clamp(0,255).round()`, then the frame-0 selector | `cpr1/inflate.py:335-352` + `runtime/f26_inflate.py:661-671` |

Three properties of the afr1 body were **MEASURED, not assumed**, before anything else ran:

* the archive is the frontier body — sha256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
  180,002 B, gated in `build_instrument` and refused otherwise;
* it carries **no compensation overlay** (`compensation_blob is None`), so the base coefficients are the
  shipped ones;
* its frame-0 selector is active on exactly **5 of 600 pairs — 60, 85, 116, 241, 373**. `ddm_ft1`'s
  verdict instrument does not apply the selector; this one does, so those five pairs are composed here
  the way the receiver composes them and were not in ft1's n200 base.

The renderer state is **always** the one the receiver parses back out of a 36,130 B SM3R section, never a
trainer checkpoint's own weights: ft1 §6 measured the deployed encoder discarding 190 of 192 FiLM rows
per pruned tensor, so a checkpoint's state is a model that never ships.

**The calibration control.** With the shipped renderer and the shipped carrier codes, this instrument
measures n600

> `d_pose = 6.3656845167356244e-06`

against the AFR1 contest-CUDA T4 receipt's **6.37e-06** — **0.068% apart** — and against `ddm_jg5`'s own
published DALI batch-8 figure **6.365684e-06** — the same number to **seven significant figures**. Two
things follow, and neither was assumed:

1. the composition, the GT lineage, the batch shape and the frame-1 path are all right, because a wrong
   one of any of them moves this number by orders of magnitude (ft1 measured the PyAV lineage at 30.7×
   off on exactly this quantity);
2. the afr1 carrier **is** jg5's re-solved carrier, sitting at its Gauss–Newton fixed point. That is the
   prior for the base control in §8, and it was measured rather than inferred from lineage prose.

Everything below is a DIFFERENCE taken on this one instrument at one declared batch shape (8), with the
same pair set, the same GT cache and the same body — enforced in code by the report's `_same_instrument`
gate, because jg5 §4b measured this forward moving with the batch shape.

## 3. THE MEASUREMENT

### 3.1 The seg leg, and the FIRE_ORDER step that had never been run

`ddm_ft1`'s fire order defines FO-1 as the n600 realized verdict of the candidate, and FO-2 (the pose
re-solve) as gated on it. Neither had run. FO-1 is run here, unmodified, on the step-600 checkpoint.
**MEASURED**, n600, ft1's own instrument:

| object | d_seg | d_pose | S @ 180,002 B |
|---|---|---|---|
| base (shipped renderer) | 0.00020132276746961806 | 6.615105594618614e-06 | 0.14812154996890514 |
| **candidate, REALIZED** | 0.0002690633138020833 | 0.01548301631695627 | 0.5402469162530941 |
| trained weights (diagnostic, never ships) | 0.00044419182671440974 | 0.0774879410559665 | 1.0445474761701012 |

* **Δd_seg = +6.774054633246527e-05** — the coupling's denominator, a DIFFERENCE taken with base and
  candidate on ONE instrument, so a common-mode instrument offset cancels.
* The base row's d_seg is **0.033%** from the contest-CUDA T4 receipt's 0.00020139. The instrument is
  aimed correctly.
* The exported section is byte-identical to the one this arm measures pose on — sha
  `819c28e8971020fb…`, 36,130 B, `size_preserved true`, `parse_back_max_abs_delta 0.0`. The two
  instruments are looking at the same object, checked rather than assumed.
* B/H/W (fcd1 law): **B = 3,752 fixed, H = 11,743 broken, W = 77**, selectivity **0.3195** — the
  candidate breaks 3.13 correct pixels for every one it fixes, on 117,949,228 untouched positions.
* The pre-re-solve coupling at n600 is **228.47**, against ft1's n200 **217.30** (5.1% apart) and rf1's
  **166.81**. The law's band survives the move to the full field.

### 3.2 The pose legs, all on ONE instrument at ONE batch shape

**MEASURED**, n600, batch 8, DALI GT, this arm's receiver-faithful composition:

| object | carrier codes | d_pose | × the AFR1 base |
|---|---|---:|---:|
| base, shipped renderer | shipped (`1a5b7a46…`) | **6.3656845167356244e-06** | 1.000× |
| candidate, STALE carrier | shipped (`1a5b7a46…`) | **0.015482008638444032** | 2,432.1× |
| **candidate, RE-SOLVED carrier** | re-solved (`098085f9…`) | **0.000942729261043249** | **148.1×** |

* The base row is **0.068%** from the contest-CUDA T4 receipt's 6.37e-06 and matches `ddm_jg5`'s own
  published DALI batch-8 figure to seven significant figures (§2).
* The candidate's stale-carrier row is **0.0065%** from ft1's independent n600 measurement of the same
  object on a different composition (0.01548301631695627). Two instruments, one number.
* The re-solve changes **6,847 coordinates across 598 of 600 pairs**; two pairs keep the shipped codes.

Composed, with the re-solve's measured **+125 B** carried:

| object | seg leg | pose leg | rate leg | S |
|---|---:|---:|---:|---:|
| base (this instrument) | 0.02013228 | 0.00797852 | 0.11985594 | 0.14796674 |
| candidate, stale carrier | 0.02690633 | 0.39347184 | 0.11985594 | 0.54023411 |
| candidate, re-solved | 0.02690633 | **0.09709425** | 0.11993918 | **0.24393975** |

The re-solve takes **0.29638 S** off the pose leg — a very large move that still leaves the candidate
**+0.09597 S** worse than the object it was trying to improve, because it was worse on seg to begin
with.

## 4. THE RECOVERY DISTRIBUTION

**MEASURED**, n600, per pair, from the re-measured batch-8 code sets:

| statistic | value |
|---|---:|
| mean-based recovery `mean_before / mean_after` | **16.42** |
| median per-pair recovery | **3,424.5** |
| geometric mean per-pair | 3,697.6 |
| pairs improved | **598 / 600** |
| pairs unchanged or worse | 2 |
| share of the post-solve mean from the worst 10 pairs | **69.0%** |
| the factor ft1 transferred (jg5, token edits) | 8.0 |
| the factor fcd2 measured (token edits) | 5.87 |

Per-pair quantiles of the recovery: `p0 1.0 · p5 1.65 · p25 105.9 · p50 3,424 · p75 156,667 ·
p95 5.43e6 · p100 2.86e8`.

**The mean and the median are 208× apart, and only one of them is the score.** d_pose enters S as a mean
over pairs, so the number that matters is the mean-based 16.42× — and it is set by the small number of
pairs that barely recover, not by the typical pair that recovers by three orders of magnitude. Anyone
quoting "the re-solve recovers thousands of times" would be quoting the median of a distribution whose
mean is owned by its tail.

**A methodological note worth carrying.** Partial estimates of this quantity are not merely noisy, they
are BIASED LOW and drift monotonically: over the run the mean-based recovery read 52.5× at 46 rows,
48.6× at 56, 30.5× at 182, 14.1× at 396, and 16.42× at 600. Each new heavy-tail pair resets it. A
subset verdict on a tail-dominated mean is not a smaller version of the n600 verdict; it is a different
number ([[m88]] on a new axis).

## 5. WHY THE RESIDUE SURVIVES — a REPRESENTATION limit, not a search limit

This is the mechanism finding, and it is the one that decides what to build next.

Every solver row carries `demanded_code_units_max`: the size, in signed-int12 code units, of the
Gauss–Newton step that pair's pose residual asks for on its worst coordinate. The shipped lattice spans
**4,095** code units end to end. Measured over the solved rows:

| statistic over the 600 solved pairs | value |
|---|---:|
| median demanded step | **444.1 code units** |
| p90 | 3,852.4 |
| max | 54,225.3 |
| **fraction demanding more than up2's ±2 radius** | **100.0%** |
| **fraction demanding more than the whole 4,095-unit lattice span** | **9.67%** |
| mean coordinates actually changed, of 12 | 11.41 |
| stop reason `no_improving_step` | **542 / 600** |
| stop reason `converged_below_materiality_floor` | 58 / 600 |

The ten pairs that own 69.0% of the post-solve mean, with their demands: pair 396 (d_pose 5.955e-02,
demand 2,012), 395 (5.882e-02, 2,004), 386 (5.618e-02, 10,639), 324 (5.366e-02, 6,346), 387 (4.698e-02,
1,491), 319 (3.613e-02, 3,187), 59 (2.848e-02, 1,865), 368 (1.976e-02, 2,477), 385 (1.571e-02, 2,498),
400 (1.509e-02, 1,846). **All ten stop at `no_improving_step`.**

Two readings, both **MEASURED**:

1. **The search radius was never the binding limit — but it was never survivable either.** Every solved
   pair demands more than the 2 code units `ddm_up2`'s greedy neighbourhood can move in one step. That
   is the arm's own justification for departing from the charter's named solver, expressed as a number
   rather than a citation.
2. **A large minority of pairs demand a correction the representation cannot express.** Their stop
   reason is `no_improving_step` — the receiver evaluated a real proposal and refused it — so this is a
   physical stop at the basis and the lattice, not a tolerance. On those pairs "solve harder" is closed.
   The only lever left is "represent more", and that has a byte price this arm can already quote
   (§8.3).

The corollary matters more than either reading: **the surviving pose leg is a SPARSE object.** A
handful of pairs own almost all of it (§4), and they are the pairs whose residual points out of the
carrier's 12-dimensional span. That is the same shape as `[[m121]]` (sparse × learned prior) and
`[[m145]]` (dD/dB bimodal), arriving on the pose axis.

## 6. THE COUPLING, PRE AND POST

**MEASURED**, all on the n600 field, with the denominator
`Δd_seg = +6.774054633246527e-05` taken from §3.1's single-instrument difference:

| coupling | value | what it is |
|---|---:|---|
| `k_pre` — this arm, n600 | **228.4546522251837** | the stale-carrier coupling, the quantity both registered anchors are |
| `k_pre` — ft1, n200 | 217.30366224024704 | 5.1% away; the law's second anchor |
| `k_pre` — rf1, n600 | 166.80837961844966 | an un-retrained structural change; the law's first anchor |
| **`k_post` — this arm, n600** | **13.822793396600533** | the SAME move after the shipping chain's terminal re-solve |
| `k_reflected` — §11 | 22.188200203315695 | the same step, opposite sense, before any re-solve |

Two things follow.

1. **`k_pre` and `k_post` are different quantities and the registered law only carries the first.**
   `k_pre / k_post = 16.53` — which is the recovery, as it must be. Any verdict that prices a renderer
   change with `k_pre` prices an object the shipping chain does not ship, because the chain re-solves
   the carrier after every seg change. That is now a domain exclusion on the law (§9).
2. **`k` is not a property of the direction.** Forward 228.45, reflected 22.19 — 10.3× apart on two
   steps of equal length and opposite sense (§11).

## 7. THE CORRECTED CLOSING ARITHMETIC

### 7.1 The frame, re-derived from the receipt rather than copied

The AFR1 row (`experiments/results/modal_auth_eval_mirror/contest_auth_eval_modal-ddm_afr1_tile48_groupbin8_cuda_n600_20260831.json`,
`[contest-CUDA T4 n600]`) decomposes exactly:

| leg | value | from |
|---|---:|---|
| seg | 0.02013900 | `100 · 0.00020139` |
| pose | 0.00798123 | `√(10 · 6.37e-06)` |
| rate | 0.11985594 | `25 · 180002 / 37545489` |
| **S** | **0.14797617125559104** | sum — reproduces the receipt to all 17 digits |

This arm buys `ΔB = 0` on the renderer leg, so the whole move is paid on distortion and a candidate
promotes iff

    √(10·d_pose_new) < 0.00798123 + 100·|Δd_seg|      with Δd_seg NEGATIVE (seg improved)

**DERIVED** — writing `k_post = (d_pose_new − d_pose_base)/|Δd_seg|` and solving:

| seg cut | Δd_seg | payable pose ceiling | × base | **k_post must be ≤** |
|---|---:|---:|---:|---:|
| 10% | −2.013900e-05 | 9.990258e-06 | 1.568× | **0.179764** |
| 25% | −5.034750e-05 | 1.694157e-05 | 2.660× | **0.209972** |
| 50% | −1.006950e-04 | 3.258288e-05 | 5.115× | **0.260320** |

The bar barely moves with the size of the cut: buying more seg buys more pose budget, but the coupling
buys more pose damage at the same rate. **k_post ≲ 0.2 is the whole gate**, at any useful cut.

### 7.2 A sign error in the charter's own inequality, stated because it changes the answer

The charter writes the promote test as `√(10·d_pose_after) − 0.00798123 < 100·|Δd_seg|`, with absolute
value. Applied literally at this candidate's Δd_seg, which is **POSITIVE** (seg got WORSE), that form
grants a pose ceiling of **2.223448e-05 = 3.49× the AFR1 base** — i.e. **+0.006930 S of pose budget for an
object that made segmentation worse**. The signed form grants nothing: the ceiling stays at the base's
own 6.370003e-06. `payable_pose_ceiling` in this arm's code implements the signed form and has a
regression test (`test_a_seg_regression_buys_no_pose_budget`) that a `+Δd_seg` buys no budget.

Consequence, stated plainly: **this candidate cannot promote at any pose value whatsoever.** It is worse
on seg AND worse on pose. Every closing number below is therefore about the LAW — what a hypothetical
seg-IMPROVING renderer move would cost after the re-solve — and not about promoting this object.

### 7.3 The corrected table

At `Δd_seg = −5.03475e-05` (a 25% cut of the AFR1 T4 d_seg), ceiling **1.694157e-05**:

| what is being priced | k used | recovery | predicted d_pose | × base | **× the ceiling** |
|---|---:|---:|---:|---:|---:|
| ft1's published closure (its own k, transferred recovery) | 217.30 | 8.0 (transferred) | 1.3676e-03 | 215× | **80.7×** |
| the same, with this arm's n600 `k_pre` | 228.45 | 8.0 (transferred) | 1.4441e-03 | 227× | **85.2×** |
| **this arm, MEASURED end to end** | **13.82 (`k_post`)** | **16.42 (measured)** | **7.0231e-04** | **110×** | **41.5×** |
| no re-solve at all | 228.45 | 1.0 | 1.1508e-02 | 1,807× | 679.3× |

**The correction is real and it is a factor of 1.95 — and the answer does not change.** The measured
recovery is 2.05× the transferred one, the overshoot falls from 80.7× to 41.5×, and the payable bar
`k_post ≤ 0.20997` is still missed by a factor of **65.8**. The seg-only renderer formulation stays
CLOSED, and it is now closed on a measured re-solve rather than an assumed one.

**Where my `k_post` is likely an UPPER bound, stated because it cuts against my own conclusion.** The
surviving residue is concentrated on pairs whose demanded step exceeds the lattice (§5). Halving the
seg damage would halve the pose damage before the solve and could bring some of those pairs back inside
reach, so `k_post` plausibly FALLS at smaller cuts and the true overshoot at a 25% cut may be less than
41.5×. It would have to fall by 65.8× to matter. §11 is the reason I do not pursue that: there is no
seg-improving move along this direction to price.

## 8. CONTROLS, AND WHAT EACH ONE RULES OUT

### 8.1 The base is already at the solver's fixed point (the slack-vs-repair control)

The recovery in §4 only means "repair of renderer-induced damage" if the solver would find nothing on the
SHIPPED renders. If it found a comparable amount there, the "recovery" would be the shipping chain's own
unclaimed slack and the coupling would have to be re-derived against a re-solved base. I pre-registered
this as one of the two ways I could be wrong in a way that matters, so I ran it.

**The identical solver, identical stopping rule, identical instrument, on the shipped renderer, seeded
random n=200 (seed 20260904, never a prefix), 1,673 s:**

| | |
|---|---:|
| pairs where any code changed | **2 / 200** |
| coordinates changed, total | **12 of 2,400** |
| mean d_pose before → after | 9.014271047465148e-06 → 8.973782731631621e-06 |
| **mean-based recovery** | **1.00451×** |
| stop reasons | **200 / 200 `no_improving_step`** |

The base carrier is at the Gauss–Newton fixed point of the shipped basis and lattice, to within 0.45%.
This is the direct form of the inference §2 already made indirectly from the seven-figure agreement with
jg5's own converged number. **All of the candidate's recovery is repair, not slack**, and the coupling's
baseline is correctly the shipped carrier rather than a re-solved one.

(The `demanded_code_units_max` on the base is a median of 412.7 with 92.7% of pairs above 2 — so even at
the fixed point the residual points mostly out of the carrier's reach. The carrier is not "solved
because there is nothing left"; it is solved because nothing left is *representable*.)

### 8.2 d_seg cannot move, and this is structural rather than measured

`upstream/modules.py:107-108`: `SegNet.preprocess_input` is `x = x[:, -1, ...]` followed by the resize —
**SegNet sees frame 1 only.** The re-solve writes frame 0 (the carrier) and nothing else. So the seg leg
is invariant to every code this arm changed, by construction. I did not spend an n600 SegNet pass
re-measuring an invariant; `ddm_jg5` §6.3 proved the same invariance the byte way, by diffing the hpac,
semantic and token-tail sections of every candidate it built.

### 8.3 The re-solve is NOT free in bytes

`ddm_ft1`'s FIRE_ORDER describes FO-2 as *"12 int12 coefficients × 600 pairs; 0 archive bytes"*. That is
wrong in general: the carrier codes are Rice-coded, so changing them changes the payload length.
`ddm_jg5` measured its own MIXED splice (455 re-solved pairs, 145 original) at **+45 B**, by building.

This arm prices a FULL 600-pair re-solve with the same pricer (`ddm_t1h_carrier_byte_pricer`, reached
through `up2.price_full_resolve_bytes`), and the pricer's anchor control passes on the afr1 body: from
the shipped codes it reproduces the shipped Rice payload exactly at **78,628 bits = 9,829 B**, so the
delta it reports is anchored rather than modelled.

**MEASURED**, on the full 600-pair re-solve
(`retained/codes_resolved_step600_n600.json`):

| | shipped | re-solved | delta |
|---|---:|---:|---:|
| Rice payload bits | 78,628 | 79,627 | +999 |
| Rice payload bytes | 9,829 | 9,954 | **+125** |
| changed coordinates | — | 6,847 | over 598 of 600 pairs |

`ΔS_rate = +8.323236914027142e-05`. Carried in every composed S in this memo. For scale, that byte cost
alone is 0.4% of the pose-leg saving the re-solve buys — immaterial here, and NOT immaterial for a
candidate whose pose gain is small, which is exactly the regime §12's selector lever lives in.

### 8.4 The comparison is controlled to one variable

The base and stale-carrier rows differ in exactly one input. Measured, from the receipts:

| field | base row | candidate row |
|---|---|---|
| carrier codes sha256 | `1a5b7a46930f653b…` | `1a5b7a46930f653b…` (identical) |
| batch size | 8 | 8 |
| pair set | full n600 | full n600 (identical index list) |
| GT cache | `gt_cache_dali.pt` | `gt_cache_dali.pt` |
| body sha256 | `cbb8d928…` | `cbb8d928…` |
| **renderer section sha256** | **`17e0fd0b…`** | **`819c28e8…`** |

One variable changed. That is what makes the difference a coupling and not a comparison.

## 9. THE EQUATIONS LEG

**Law refined:** `renderer_seg_pose_coupling_shipped_object_v1` —
`tac.canonical_equations.renderer_seg_pose_coupling_20260903` (`tac.canonical_equations`), registered by
`ddm_eq1` on 2026-09-04 with two anchors (rf1 166.81 structural, ft1 217.30 trained).

**Relation:** DOMAIN REFINEMENT + IN-DOMAIN ANCHOR. The refinement is not cosmetic — the registered
payload carried

```
domain_of_validity["carrier_recovery_measured"] = [5.87, 8.0]
```

and neither number was measured for a renderer change. 8.0× is `ddm_jg5`'s n600 carrier recovery on
TOKEN edits and 5.87× is `ddm_fcd2`'s. The law's closing table multiplied a renderer-change Δd_pose by
that factor and concluded "81× over the ceiling"; that conclusion is the reason a renderer charter is
refused before it launches. Applied through
`tac.canonical_equations.registry.update_equation_with_domain_refinement`, this arm's refinement:

* **includes** `k` read as a PRE-re-solve coupling — the quantity both existing anchors actually are;
* **excludes** using a PRE-re-solve `k` inside a closure, promotion or refusal verdict WITHOUT running
  the terminal re-solve, because the shipping chain re-solves the carrier after every seg change, so a
  pre-re-solve `k` prices an object that never ships;
* **excludes** transferring a TOKEN-edit carrier recovery factor onto a RENDERER change;
* **excludes** reading the mean-based recovery as the typical pair's recovery;
* adds `carrier_recovery_measured_renderer_change` and `post_re_solve_payability_bar` with its
  derivation.

The anchor is then appended through `update_equation_with_empirical_anchor`. Its residual is scored
against the band's own quantity — the PRE-re-solve `k` — never against the POST-re-solve one, because
the registered band was fitted to the first and scoring a residual against a centre fitted to a
different quantity is the arithmetic version of a cross-instrument comparison. Both calls are made by
`tools/register_ddm_pr1_coupling_refinement_20260904.py`, which READS every value out of the arm's own
report JSON so a transcription slip cannot enter the registry (regression-tested).

**LANDED.** The registry now carries three anchors on this law —
`rf1_film_amortized_flat_w96_structural_coupling_20260824`,
`ft1_shipped_renderer_seg_only_finetune_coupling_step600_20260903` and
`pr1_terminal_pose_resolve_on_ft1_step600_renderer_change_20260904` — with
`domain_of_validity.carrier_recovery_measured_renderer_change = {mean_based 16.42, median_per_pair
3424.49, n_pairs 600}`, `coupling_post_re_solve_measured = 13.8228`, and three explicit exclusions.
Verified by reading the registry back after the write.

## 10. HONEST LIMITS AND verdict_scope

**`verdict_scope: FORMULATION`** for the mechanism — the terminal pose re-solve, applied to a seg-only
renderer change on the shipped 36,130 B SM3R renderer at its own size, on the afr1 body. **`INSTANCE`**
for every specific number: they are this candidate, this checkpoint, this damage magnitude.

What this arm does NOT establish, listed rather than left implicit:

1. **Direction symmetry is still an assumption.** Every anchor of the law, and this candidate, moved
   d_seg the WRONG way. The corrected closing arithmetic applies `k_post` to a seg DECREASE that has
   never been realized. §11 reports the first artifact built to attack this, and its result.
2. **One damage magnitude.** `k_post` is measured at one Δd_seg. The registered law is linear in
   Δd_seg by assumption, not by measurement, and this arm adds one point rather than a slope.
3. **Advisory axis.** Every d_pose here is `[macOS-CPU advisory]`. The base row's 0.068% agreement with
   the contest-CUDA T4 receipt is a strong calibration, not authority. Only `upstream/evaluate.py` on
   contest hardware is a score, and no exact row was bought.
4. **The seg leg is a cross-instrument input.** Δd_seg comes from `ddm_ft1`'s verdict instrument, which
   renders frame 1 in batches of 4; this arm renders at batch 1, which is what the CPU receiver does
   (`cpr1/inflate.py:315` — `semantic_batch = 8 if device.type == "cuda" else 1`). Because the coupling's
   denominator is a DIFFERENCE taken with base and candidate on that ONE instrument, a common-mode
   offset cancels; a shape-dependent DIFFERENTIAL would not, and I did not measure that.
5. **Pairs that stopped at the derived materiality floor have more available.** On those the reported
   recovery is a LOWER bound — bounded, by construction, by less than one instrument band of score.
6. **The selector sweep is a screening instrument**, not a verdict: single-pair evaluation is batch 1 by
   construction, and any adopted selector set must be re-measured at the declared batch shape and
   byte-closed before it is a candidate.

## 11. THE REFLECTED STEP — attacking the law's own named assumption

The registered law names exactly one thing as stated-not-measured:

> DIRECTION SYMMETRY: both anchors moved d_seg UP; the closing arithmetic applies the same k to a seg
> DECREASE, i.e. it assumes local linearity of the realized map around the shipped weights. Cheapest
> falsification available.

No seg-improving renderer change exists to measure — ft1's run closed with `best_step = 0` and
`improved_over_init = false`, so its four evaluated points are all on the wrong side. The cheapest
available object is therefore the point-reflection of the realized candidate through the shipped
weights, `W_reflected = 2·W_shipped − W_candidate`: the same step along the seg-only direction with the
sign flipped.

**The first question is whether the deployed encoder can even carry the opposite step**, because the
SM3R encoder derives its per-tensor scales from the values. **MEASURED** (`ddm_pr1_reflect_renderer_step`,
retained `pr1_reflected_step600_record.json`), comparing the REALIZED reflected step against the
REALIZED candidate step in weight space:

| quantity | value | perfect reflection |
|---|---:|---:|
| cosine | **−0.9408938316242174** | −1.0 |
| norm ratio | **1.0136714794583306** | 1.0 |
| section bytes | 36,130 (size-preserving) | — |
| parse-back max abs delta | 0.0 | 0.0 |
| intended-vs-realized max abs delta | 0.0006647109985351562 | — |

The encoder carries it: the realized reflection is a genuine opposite step at 94% alignment and matched
length. **The symmetry question is askable of this representation**, which was not obvious in advance.

### 11.1 The assumption is MEASURED FALSE, and not marginally

n600, both instruments, on the realized reflected section (sha `d5b7c2810bad08a8…`, 36,130 B,
`size_preserved true`, intended-vs-realized gap 0.0 because the checkpoint already carries the
parsed-back state):

| object | d_seg | Δd_seg | d_pose | Δd_pose | coupling |
|---|---:|---:|---:|---:|---:|
| base (shipped) | 0.00020132276746961806 | — | 6.615105594618614e-06 | — | — |
| candidate (forward step) | 0.0002690633138020833 | **+6.774054633246527e-05** | 0.01548301631695627 | +1.5476401e-02 | **228.47** |
| **reflected (−1 × the same step)** | **0.001660859849717882** | **+0.0014595370822482639** | 0.03239111609068234 | +3.2384501e-02 | **22.19** |

Read it in the order that matters:

1. **The reflected step does not lower d_seg. It raises it — by 21.55× more than the forward step did.**
   The shipped weights are not sitting on a locally linear slope along this direction; along this
   direction, at this step length, they sit at the bottom of a sharp valley in d_seg and BOTH ways are
   uphill. Its B/H/W is 5,884 fixed against **178,058 broken**, selectivity 0.033.
2. **The coupling is not a property of the direction. It is a property of the SIGNED step**: 228.47
   forward, 22.19 backward — **10.3× apart** — for two steps of the same length (norm ratio 1.014) and
   opposite sense (cosine −0.941).
3. `d_pose` on this arm's own instrument, independently: **3.2397760991e-02**, against ft1's
   3.239111609068234e-02 — 0.02% apart, so the two instruments agree on the reflected object as they do
   on the forward one.

**What this closes and what it does not.** It closes the law's stated assumption: local linearity of
the realized map around the shipped weights is **FALSIFIED**, in the only direction anyone can currently
test, by a factor of 21.6 on the seg response and 10.3 on the coupling. It does NOT establish that the
shipped weights are a local minimum in every direction — this is a two-point probe along ONE direction
at ONE (large, export-quantised) step length, and I will not claim more than the two points support.

**The consequence for the closure is bigger than the correction to the recovery factor.** ft1's
arithmetic — and mine in §7 — prices "a 25% seg cut along the seg-only direction". The measured local
geometry does not offer such a move: along the one direction that has been realized and exported, seg
rises in both senses. So the closure of the seg-only renderer axis rests, after this arm, not on the
pose price being unpayable but on **the seg gain being unreachable from these weights along this
direction at all**. That is a different and stronger statement, and it is measured rather than
extrapolated.

## 12. THE PER-PAIR SELECTOR — a different actuator on the same frame

§5 measures the surviving residue as a REPRESENTATION limit of the 12-dim carrier. The shipped receiver
carries a second, independent actuator on the very same frame: the frame-0 selector
(`runtime/frame0_selector.py`), a per-pair integer pixel op chosen from a fixed 8-entry table
(identity, ±1 luma, a channel tilt, two rolls, two tile dithers), shipped as a sparse
combinatorial-rank blob of **14 B** with exactly **5 of 600** pairs currently non-identity.

That is precisely the shape of actuator the residue calls for: **per-pair** (so it has the admission
lever a renderer change lacks), **coarse and large-amplitude** (so it can move where a 12-dim step
demanding thousands of code units cannot), and **nearly free** (the blob's own format prices going from
5 active pairs to 15 at about +12 B).

### 12.1 On the LIVE frontier the selector carries unclaimed pose

I swept all 8 modes on all 600 pairs of the **shipped** object with the **shipped** carrier codes — the
live afr1 body, not the candidate. 1,363 s of CPU. **MEASURED**, batch 1 (a single-pair evaluation is
batch 1 by construction):

| | |
|---|---:|
| pairs whose best mode beats the shipped mode | **49 / 600** |
| … with a >1% margin (≫ the measured batch-shape spread) | **39** |
| n600 mean d_pose gain, margin-gated | **2.013298e-07** |
| pairs newly non-identity | 38 |
| **shipped-active pairs whose best mode is IDENTITY** | **1 — pair 85, mode 3 → 0, 1.93×** |

Pair 85 is the sharp detail: a selector op the archive currently ships is **actively harmful** on the
current renders. It was chosen for an earlier body and never re-derived — the same expiry class as
`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`, in shipped bytes.

**The byte price is exact, not modelled.** The blob length is a closed function of the active count:
`7 + ceil(bit_length(C(600,k)−1)/8) + ceil(3k/8)` (`runtime/frame0_selector.py:96-107`). At k=5 it
returns **14 B** — exactly what the archive carries, so the formula is the receiver's own.

| adoption | active pairs | blob | ΔB | ΔS_pose | ΔS_rate | **net ΔS** |
|---|---:|---:|---:|---:|---:|---:|
| margin-gated (ratio > 1.01) | 42 | 50 B | **+36 B** | −1.271836e-04 | +2.397092e-05 | **−1.032126e-04** |
| ungated (every positive gain) | 52 | 59 B | +45 B | −1.278632e-04 | +2.996365e-05 | −9.789950e-05 |

The margin gate is also the byte-optimal choice: the extra 10 pairs cost more rate than they buy pose.

> **`[macOS-CPU advisory projection]` — S 0.14797617125559104 → 0.14787295862740366, net −1.032e-04.**
> **NOT a score.** It is a screening measurement at batch 1, composed against a batch-8 population mean
> (a cross-shape step, bounded by jg5's measured spread at ≲0.3% of the gain), with the carrier held
> fixed per pair. It needs a byte-closed build and an `upstream/evaluate.py` row before it is anything
> else. It is also almost certainly a LOWER bound on this axis: re-solving each changed pair's carrier
> against its new frame 0 can only help, and this sweep did not.

### 12.2 On the candidate, after the re-solve

The same sweep on the re-solved candidate, all 600 pairs, 1,828 s. **MEASURED**:

| | |
|---|---:|
| pairs improved / with >1% margin | 152 / **137** |
| n600 mean d_pose gain | **3.406045e-04** |
| d_pose after the re-solve | 0.000942729261043249 |
| **d_pose after re-solve + selector** | **6.021247e-04** (a further **1.57×**) |
| **share of that gain from the ten pairs that own the pose leg** | **50.3%** |
| active selector pairs / blob / ΔB | 140 / 119 B / **+105 B** |
| **`k_post` with the selector** | **8.79**, down from 13.82 |

**The actuator lands where the mechanism said it would.** Half the gain comes from the ten pairs whose
Gauss–Newton demand exceeds the lattice — the pairs the carrier provably cannot reach. Pair 61 improves
**16.8×**, pair 368 **7.2×**, pair 309 **10.8×**, all by a single ROLL or CHANNEL op.

And it still does not open the door: at a 25% seg cut, `k_post = 8.79` predicts `d_pose = 4.489e-04`,
which is **26.5× the payable ceiling** (down from 41.5× with the re-solve alone, and from ft1's 80.7×).
Two successive corrections — the measured recovery, then the selector — move the overshoot 80.7 → 41.5 →
26.5, a cumulative factor of 3.05, against the factor of 65.8 that would be needed.

## GESTALT-DELTA

1. **A transferred constant was doing load-bearing work inside a CLOSURE, and it was wrong by a large
   factor.** `ddm_ft1` refused the renderer axis with "81× over the ceiling"; that 81× is
   `k · cut / 8.0 / ceiling`, and the `8.0` is `ddm_jg5`'s TOKEN-edit carrier recovery imported onto a
   renderer change. This arm measured the real one. The class is
   [[cross-regime-constant-transfer-genus-finishing-stage]] — but with a new and worse costume: the
   previous instances transferred a constant into a LAUNCH decision, where a bad launch is visible.
   This one transferred a constant into a REFUSAL, where being wrong is silent forever. **A refusal
   built on a borrowed number is the most expensive kind, because nothing ever comes back to check it.**
   **MEASURED here:** the correction is 2.05× on the recovery and 1.95× on the overshoot, and the
   verdict survives it. The lesson is not that the closure was wrong; it is that nobody knew whether it
   was, for the whole time it was standing.
2. **The surviving pose residue is a REPRESENTATION limit, and that reframes the whole carrier axis.**
   Every solved pair's Gauss–Newton step demands more than up2's ±2 radius, and a large minority demand
   more than the entire int12 lattice span. The pairs that survive the re-solve are the ones whose
   residual points OUT of the carrier's 12-dimensional span. "Solve harder" is closed on this object.
   "Represent more", or "actuate differently", are the only two live moves — and the second one is
   nearly free.
3. **The pose leg is a SPARSE object.** After the re-solve, a handful of pairs own almost all of the
   n600 mean. That is `[[m121]]` (sparse × learned prior) and `[[m145]]` (dD/dB bimodal) arriving on the
   pose axis, and it has a direct operational consequence: a per-pair actuator aimed at ten pairs is
   worth more than any uniform improvement.
4. **The frame-0 selector is that per-pair actuator, it is already in the receiver, and the LIVE frontier
   has not claimed it.** 39 of 600 pairs beat their shipped selector mode by >1%, one shipped-active
   pair is actively harmful, and the whole adoption prices at +36 B for a net **−1.032e-04 S** advisory
   projection. `ddm_ft1` wrote "a renderer change has no per-pair admission lever" — true of the RENDER,
   and false of the frame the render is scored against.
5. **An advisory CPU pose instrument, composed correctly, reproduces the contest-CUDA pose leg to
   0.068%.** That is worth carrying forward as a calibration fact: DALI GT + the receiver's own frame-1
   path at batch 1 + carrier-and-selector frame 0 + batch-8 scoring. Get any one of them wrong and the
   number moves by orders of magnitude (ft1 measured 30.7× on the lineage alone).
   6. **My own predictions were worse than the operator's, in a specific and instructive way.** I
   over-estimated the recovery (predicted 30–1000×, measured 16.42×) because I extrapolated from the
   MEDIAN pair I could see in the early rows, and under-estimated how completely a mean over per-pair
   MSEs is owned by its tail — the very effect I had written into the pre-registration one paragraph
   earlier. Knowing a bias by name is not the same as pricing it.

## NEXT_IF_RESUMED

Everything below is a typed order against artifacts that exist, not a list of intentions.

0. **State of the runs.** All complete; receipts under
   `/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/retained/`. Every solve is `--resume-from`-able by
   pair through its `rows.jsonl`; re-running a solve command continues it rather than restarting.

1. **FIRE FIRST — byte-close and buy the selector ratchet on the LIVE frontier.** §12.1 measures
   `−1.032e-04 S` (advisory) for +36 B on the shipped afr1 object, from a sweep that already exists
   (`retained/selector_sweep_base_shipped_n600.json`). It needs three things this arm did not build:
   (a) a selector ENCODER — the shipped runtime is decode-only, and the blob is a combinatorial-rank
   format whose length is already exactly computable (`7 + ceil(bit_length(C(600,k)−1)/8) + ceil(3k/8)`,
   verified against the shipped 14 B at k=5); (b) a re-measure of the adopted set at the declared batch
   shape 8; (c) a carrier re-solve on the changed pairs, which can only improve it. Then splice, and
   `tools/fire_modal_auth_eval.py --seal`. **PROMOTE IFF exact S < 0.14797617125559104 on
   `[contest-CUDA T4 n600]`.** This is the only item here with a direct path to a lower exact score.

2. **Sweep the selector on the pairs the carrier cannot reach, jointly.** §12.2 measures the selector on
   the re-solved candidate; the general order is to alternate `selector` and `solve` on the ~10 pairs
   that own the pose leg until neither moves. The instrument for both halves is committed
   (`experiments/ddm_pr1_pose_resolve_on_renderer_change.py`, modes `selector` and `solve`), and the
   selector sweep costs about 2.3 s per pair.

3. **The representation question is now byte-closed, so ask it.** A quarter of pairs demand a
   Gauss–Newton step larger than the whole int12 lattice span. The two moves are more carrier
   dimensions or finer quantisation, and both have an exactly computable rate cost through the same
   Rice pricer this arm anchored (`up2.price_full_resolve_bytes`, control reproduces the shipped 78,628
   bits). Price `d_pose(K) vs bytes(K)` for K = 12, 16, 24 before building anything.

4. **Re-derive the coupling law's OTHER assumption: linearity in |Δd_seg|.** The law is linear in the
   seg move by assumption; this arm adds one point at `Δd_seg = 6.774e-05`, and §5 gives a mechanism
   (lattice saturation) that predicts `k_post` FALLS at smaller cuts. A second point at roughly half the
   damage — the step-1,200 checkpoint is retained and its section exports in seconds — would turn a
   line through one point into a measured slope.

5. **Do not spend the window on the step-1,800 checkpoint for its own sake.** It is +6.54% on seg (still
   worse than shipped) and its value is only as the second point in item 4.

98. **Do NOT re-run a seg-only renderer fine-tune** at any learning rate. The corrected arithmetic still
    refuses it, and the refusal no longer rests on a transferred recovery factor.
99. **The JOINT (pose-priced) formulation remains OUT of the law's domain and OPEN.** Nothing here
    touches it. ft1's prior stands: w96b ran pose in-loop from step zero and still landed 204× over.

## RECEIPTS

All under `/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/`. Every measurement persisted its payload,
not only its length: per-pair `d_pose` vectors, pose vectors and code tables are `.npy` beside each
report with their sha256 in the JSON.

| artifact | what it is |
|---|---|
| `retained/measure_base_shipped_codes_n600.json` | base d_pose n600 + `*_payload/` (per-pair, poses, codes) |
| `retained/measure_cand_step600_stale_carrier_n600.json` | the stale-carrier row |
| `retained/measure_cand_step600_re_solved_n600.json` | the re-solved row |
| `retained/measure_reflected_step600_stale_carrier_n600.json` | the reflected step's pose row |
| `retained/codes_resolved_step600_n600.npy` + `.json` | the re-solved 600×12 code table, sha `098085f9…`, priced |
| `solve_cand_s0,s1/rows.jsonl` + `SUMMARY.json` | 600 per-pair solver rows, resumable |
| `solve_base_control/rows.jsonl` + `SUMMARY.json` | the n200 base control |
| `retained/verdict_ft1_step600_n600.json` | ft1 FO-1 at n600 (d_seg, B/H/W) |
| `retained/verdict_reflected_step600_n600.json` | the reflected step's realized d_seg |
| `retained/pr1_reflected_step600_{semantic_section.bin,checkpoint.pt,record.json}` | the reflected object |
| `retained/selector_sweep_base_shipped_n600.json` | the LIVE-frontier selector sweep |
| `retained/selector_sweep_cand_re_solved_n600.json` | the candidate selector sweep |
| `retained/report_step600_n600.json` | the coupling report the equations leg reads |

The 1.83 GB odd-frame render field is NOT persisted: it is deterministically rebuildable from the
retained 36,130 B section plus `tokens.u8` through the receiver's own frame-1 path, and `measure
--render-digest` will emit its sha256 for a successor who wants to prove byte-identity.

## Own-vehicle frontier

**afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED by this arm.**
