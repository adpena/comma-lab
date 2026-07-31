# ddm_ba31 — the 2026-07-31 negatives, placed as points on response surfaces

**Date:** 2026-07-31 · **Actor:** ddm_ba31 (Opus arm) · **Window:** 2026-07-31 only (sisters cover 07-29 / 07-30).
**Authority:** `[macOS-CPU advisory]` / derived-from-committed-receipts. `research_only=true`, `score_claim=false`.
**Pointer `0.1910828242 [contest-CPU]` UNMOVED** — this unit is $0, read-only: no scorer slot taken, no training
launched, no archive built.

**Operator directive this executes (verbatim, 2026-07-31):** *"I've told you numerous times no binary results ever.
We are proceeding Einsteinian and according to our design and guiding philosophies and principles, which include
unification, completeness, and more."* Preceded by *"they're probably not optimal. Don't make binary judgment
prematurely. In fact, never make binary judgment. Always seek to understand."*

**What this document is NOT:** it is not a ranked list of negatives, and it does not say which verdicts were
"wrong." Replacing `DEAD` with `DEAD, 1.14× from the line` is still a binary with a footnote. Every negative
recorded on 2026-07-31 is here re-entered as **a measurement of one point on a continuous response surface**,
with all of its coordinates. Where a verdict is sound, it is called sound and the reason is given.

---

## §0 The six surfaces

Every negative in the window lands on one of six continuous objects. Same surface at different coordinates is
ONE finding, not several.

| # | Surface | The real-line quantity | What moves its level sets |
|---|---|---|---|
| **A** | **Base-error axis** ρ = d_seg | fraction of sites in error | burn capacity (moves our position); coder coherence + formulation expressiveness (move the *edges*) |
| **B** | **Correction price** b(ρ, coder) | bytes per corrected flip | coder coherence (measured 11.4× span); position-vs-label split |
| **C** | **Instrument distortion** | signed relative bias of a reading | sampling design, estimator, basis matching, granularity |
| **D** | **Build completeness** | distance from designed → fired | wiring (this surface *generates* points on C and on A/B verdicts) |
| **E** | **Joint exchange rate** | ΔS per byte, per axis | where each axis sits vs its own knee |

**Surfaces A and B share the coordinate ρ. Surfaces B and E share the water level 1.2731 B/flip — which §5b.3
shows is not an empirical constant at all but the score function's own exact seg↔rate exchange rate.**
| **F** | **Governance / wall-clock** | hours against a chosen cap | the cap constant itself (6.0 h) |

Surfaces **A** and **B** share the coordinate ρ — they are two readings of one object. Surface **D** is the
upstream generator of much of **C**. Surface **F** decided two of the day's three largest calls, and no
measurement surface was consulted for them (correctly, and MAIN said so).

---

## §1 SURFACE A — the base-error axis ρ = d_seg

Every seg number produced or carried on 07-31 is a coordinate on **one real line**. Two independent families of
level sets are overlaid on it, from two artifacts that never referenced each other.

### A.1 The level sets

**Family 1 — pp1 correction-rationality band edges** (`ddm_pp1_correction_stream_position_band_v1`,
`src/tac/canonical_equations/ddm_pp1_correction_stream_position_band_20260728.py`; receipt
`.omx/research/ddm_pp1_band_lemma_receipt_20260728.json`):

| edge | ρ | source |
|---|---:|---|
| coherent lower edge `rho_c` | **5.015010841012719e-4** | MEASURED, 9-density coherent sweep |
| uniform-bound lower edge `rho_u` | **8.591513961238275e-4** | DERIVED, `2^(-8·1.2731)` |
| incoherent lower edge | **≈1.5e-3** | MEASURED, random-support + generic LZMA |
| band upper edge | **1.0e-2** | sp1's 421,366 B support wall at ρ=0.864% |

**The lower edge is not a constant. Its measured span across coder families is 1.5e-3 / 5.015e-4 = 2.991×**
(uniform→coherent alone = **1.7132×**). The equation's own `domain_of_validity["context_shift"]` says it:
*"boundary clustering lowers position entropy below the uniform combinatorial rate."* **Coder quality and band
edge are the same degree of freedom.** r7's 14-coder race is a measured axis that moves this level set.

**Family 2 — fl1 smooth-label GT-flicker floors** (`.omx/research/ddm_fl1_perclass_flicker_floors_20260731.md` §2,
MEASURED from `gt_n600.npz`, sha `cf8d83605d…`, N=598 interior pairs, reproduced to +4.4e-7):

| class | floor d_seg | floor S |
|---|---:|---:|
| Movable | 2.847e-4 | 0.02847 |
| Undrivable | 3.939e-4 | 0.03939 |
| MyCar | 4.343e-4 | 0.04343 |
| Road | 1.8894e-3 | 0.18894 |
| Lane | 2.3162e-3 | 0.23162 |
| **aggregate** | **5.3184e-3** | **0.53184** |

**This floor is also not a constant.** It is the level set at *formulation = smooth-label / temporal-majority*.
fl1 states this correctly and refuses the hard-floor reading. Its measured formulation span on the same axis:
aggregate floor 5.3185e-3 vs PR130 phase-faithful 2.966e-4 = **17.93×**.

### A.2 Every 07-31 coordinate on the axis, in order

| ρ = d_seg | what it is | pp1 regime | ρ/rho_c |
|---:|---|---|---:|
| 2.847e-4 | Movable smooth-label floor | concede | 0.568 |
| 2.966e-4 | PR130 native rail (ships NO correction stream) | concede | 0.591 |
| 3.939e-4 | Undrivable smooth-label floor | concede | 0.785 |
| 4.343e-4 | MyCar smooth-label floor | concede | 0.866 |
| **5.015e-4** | **`rho_c` — coherent band lower edge** | — | 1.000 |
| 6.0e-4 | corner-C seg target (gc13 §4, = 0.06 S) | correct | 1.196 |
| **8.592e-4** | **`rho_u` — uniform band lower edge** | — | 1.713 |
| **≈1.5e-3** | **incoherent band lower edge** | — | 2.991 |
| 1.8894e-3 | Road smooth-label floor | correct | 3.768 |
| 2.3162e-3 | Lane smooth-label floor | correct | 4.618 |
| 3.89011e-3 | tb1 burn endpoint (pfs1 D1 evaluate.py receipt) | correct | 7.757 |
| **3.943024e-3** | **burn-4 ep854 endpoint — WHERE WE STAND** | **correct** | **7.862** |
| 4.264052e-3 | ep641 (xp1 r1c) | correct | 8.503 |
| 4.310379e-3 | gr1_cell_drop50 seg base (v4d archive) | correct | 8.594 |
| 5.3184e-3 | aggregate smooth-label flicker floor | correct | 10.605 |
| **1.0e-2** | **band upper edge** | — | 19.94 |
| 1.38e-2 / 1.41e-2 | tr1 T2 plain / lotto | explode | 27.5 / 28.1 |
| 7.02157e-2 | fd1 | explode | 140.0 |
| **7.05192e-2** | **W_joint pre-arc — the base the "corrections DEAD" line is parented on** | explode | 140.6 |

### A.3 What this placement says that no single artifact said

**(i) The `regime=EXPLODE → corrections DEAD` line and the `IN-BAND → re-grade DUE` line are the same law read at
two coordinates 17.9× apart.** `src/tac/ddm_costate_organ.py:3116-3125` emits the first from `band_pos['base_d_seg']`
= the **pre-arc** W_joint base 7.05192e-2 (`:3117`); `:3191-3204` emits the second from the **live** burn endpoint
3.89011e-3 (`:3197` DEAD branch / `:3200` IN-BAND branch — one `if/else`, one law, two coordinates).
The organ already labels the first `DDM-band[pre-arc describe-line base]` and says *"the LIVE burn endpoint is in
DDM-parents, not here"* — so the source is honest. But both lines are in the same digest, and the digest drops
`DDM-band` and `DDM-parents` into `_HOOK_DETAIL_PREFIXES` (`tools/costate_digest.py:78, 83`), so at session start a
reader sees neither. That drop is **deliberate and documented** (a measured 52% session-start token cut,
fail-toward-visible drop-list, every byte one `--full` away) — a budget decision, not a defect. The consequence is
still real: the surviving impression at t=0 is a tag. **The campaign crossed a level set on this axis — that
crossing is the finding, and a 3-valued tag cannot express it at any verbosity.**

**(ii) Three of the five per-class flicker floors lie BELOW the correction band's lower edge — and all three lie
within the measured coherence shift of it.**

| class | floor ρ | `rho_c`/floor | vs measured edge span |
|---|---:|---:|---|
| Movable | 2.847e-4 | **1.7616×** below | inside 1.7132× (uniform→coherent), inside 2.991× (incoherent→coherent) |
| Undrivable | 3.939e-4 | **1.2732×** below | inside both |
| MyCar | 4.343e-4 | **1.1547×** below | inside both |

Road (3.768×) and Lane (4.618×) sit **inside** the band. So: **whether a per-class correction stream is priced in
at all is decided, for three of five classes, entirely by coder coherence — at exactly the magnitude the coherence
effect was measured to have.** No regime tag can answer that; a coder race can. (Caveat, labeled: restricting a
correction's support to one class's region raises the *effective* density and moves each class rightward — that is
the same coherence mechanism, which is precisely the point.)

**(iii) Corner-C sits 1.196× above `rho_c`.** Our stated seg target lands 20% inside the band's coherent edge and
**below** its uniform edge (0.699× of `rho_u`). At the target we are aiming for, the correction question is not
settled by the base at all.

---

## §2 SURFACE B — the correction price b(ρ, coder)

The same receipt carries a **14-point measured curve**, which the consuming organ reduces to one of three words.

### B.1 The measured surface (receipt `ddm_pp1_band_lemma_receipt_20260728.json`)

Water level = **1.2731 B/flip** (registered region-merge concede price). `eff` = measured / uniform bound: lower
is a better coder.

**Coherent family (margin-thresholded boundary-clustered support, n600 GT):**

| τ | ρ | b measured | uniform bound | **eff** | b/water |
|---:|---:|---:|---:|---:|---:|
| 0.008 | 2.24745e-4 | 1.468656 | 1.514928 | 0.96946 | 1.1536 |
| 0.02 | 5.63202e-4 | 1.244830 | 1.349258 | 0.92260 | 0.9778 |
| 0.05 | 1.41313e-3 | 1.005441 | 1.183361 | 0.84965 | 0.7898 |
| 0.1 | 2.82354e-3 | 0.802635 | 1.058535 | 0.75825 | 0.6305 |
| 0.2 | 5.62195e-3 | 0.587224 | 0.934339 | 0.62849 | 0.4613 |
| 0.4 | 1.11236e-2 | 0.359315 | 0.811278 | 0.44290 | 0.2822 |
| 0.8 | 2.17047e-2 | 0.199917 | 0.690731 | 0.28943 | 0.1570 |
| 1.5 | 3.79993e-2 | 0.118064 | 0.589735 | 0.20020 | 0.0927 |
| 3.0 | 7.00618e-2 | 0.067909 | 0.479404 | 0.14165 | 0.0533 |

**Incoherent family (random subsampled support + generic LZMA):**

| ρ | b measured | uniform bound | **eff** |
|---:|---:|---:|---:|
| 9.99959e-5 | 2.692523 | 1.660971 | 1.62105 |
| 2.99996e-4 | 1.972675 | 1.462846 | 1.34852 |
| 1.0e-3 | 1.403899 | 1.245723 | 1.12698 |
| 3.0e-3 | 0.962297 | 1.047603 | 0.91857 |
| 1.0e-2 | 0.559067 | 0.830482 | 0.67318 |

**Coder efficiency spans 0.14165 → 1.62105 = 11.44×** across the measured surface, and is **monotone in ρ within
each family** — clustering pays more as errors get denser. At matched density the pure coherence gain is
**1.28× at ρ=1e-3** and **1.23× at ρ=3e-3**. None of this is visible in `regime ∈ {concede, correct, explode}`.

### B.2 QA03/QA04 "MEASURED BREAK-EVEN" — the missing decomposition

`src/tac/ddm_costate_organ.py:2480-2493` records, as the prior attached to the in-band re-grade:

> *"QA03 full-population GN/CG seg solve: +1,866 flips, ΔS_seg −0.001582 = 1.15% of the −0.138 ceiling,
> **1.45 B/flip ≈ water 1.27** (~break-even) = 3rd white-jitter confirmation; QA04 attack-search round-2:
> +773 flips, ΔS_seg −0.000655 @ 800 evals = 4th confirmation"*
> → *"corrections-class duties stay low-priority — the seg descent lever is the burn (lower the base), not any
> correction stream at this base"*

**The break-even is a ratio, and it is 1.13895** (1.45 / 1.2731). **A break-even is ratio 1.0 — a point exactly ON
a level set, which is the most informative coordinate on the whole surface.** It is being consumed as a stopping
reason.

**The decomposition nobody did.** The band law's `domain_of_validity["excluded"]` explicitly excludes *"the LABEL
(which-class) cost of a correction stream (this law is POSITION only)."* At the burn base ρ = 3.89011e-3:

| term | B/flip | note |
|---|---:|---|
| QA03 measured **total** | **1.4500** | MEASURED |
| water level | 1.2731 | registered |
| law's **uniform** position bound at this ρ | 1.0007 | DERIVED, `log2(1/ρ)/8` |
| law's **measured-coherent** position at this ρ | **0.7024** | DERIVED by log-density interpolation of the 9-point curve |
| ⇒ **non-position residue** | **≥ 0.7476** | **51.6% of QA03's total** |
| position headroom under water | 0.5707 | water/position = **1.812×** |

**QA03's total exceeds even the *uniform upper bound* on position cost by 1.449×.** So the 1.45 B/flip is not a
position-cost measurement at all: **at least 51.6% of it is label + solver overhead, a term the governing law
declares out of domain.** The position budget at our live base has 1.812× of slack under the water level.
*(Labeled assumption: the interpolation assumes QA03's flip support is comparably coherent to the receipt's
margin-thresholded synthetic support. If QA03's support is* less *coherent, the residue is smaller; if more, larger.
Either way the decomposition — not the composite — is the decidable quantity, and it is one measurement away.)*

**What the composite 1.45 CAN support:** that *this particular solver, at this base, with label cost included,* did
not beat conceding. **What it cannot support:** that corrections are priced out at this base — because the law it
is being compared against prices only the half of the cost that has 1.8× of slack.

### B.3 The decomposition flips the SIGN of the whole correction strategy

Price the idealized full-residual correction at each of the three prices. `N = 600·384·512 = 117,964,800`;
rate cost `= 25·B/37,545,489`; seg gain `= 100·d_seg`.

**ja1/v4c base, d_seg = 0.00431179 ⇒ 508,639 flips, seg = 0.431179 S:**

| price basis | B/flip | bytes | rate cost | seg gain | **NET S** |
|---|---:|---:|---:|---:|---:|
| law **coherent position** (interp) | 0.6702 | 340,897 | +0.226989 | −0.431179 | **−0.204190 (win)** |
| law **uniform** position bound (no interp) | 0.9822 | 499,579 | +0.332649 | −0.431179 | **−0.098530 (win)** |
| **QA03 composite** | 1.4500 | 737,527 | +0.491089 | −0.431179 | **+0.059910 (lose)** |

**burn-4 ep854 base, d_seg = 0.003943024 ⇒ 465,138 flips, seg = 0.394302 S:** −0.178065 / −0.085109 / **+0.054786**.

**The strategy's sign is not a property of the world — it is a property of which half of the cost you price.**
The swing on the ja1 base is **0.264 S**, larger than the entire rate axis (0.239543). Even the *conservative*
uniform bound, which needs **no interpolation at all**, puts it at **−0.0985 S — a win.** The composite is the
only one of the three that loses.

**Caveats, stated:** (a) this is the idealized "correct every flip to zero residual" calculation — QA03 actually
addressed 1,866 flips = **0.367%** of the residual, so this prices the *slope*, not a realized move; (b) the
coherent row is interpolated on synthetic supports (the uniform row is not, and it already wins); (c) **the label
cost is real and must be paid somewhere** — the claim here is only that it has never been separated from position,
and that the decision inverts across the separation.

**Named cheap test (the one measurement that decides it):** re-read QA03's own 1,866-flip support and split its
coded bytes into (position stream) + (label stream), using the same `packbits→LZMA1-x9e` / #307 contour-chain
coders the pp1 receipt already used. That is $0 and scorer-free. **Hypothesis to test, not assume:** the label
(which-class) stream may be largely predictable from *neighbouring* labels — the same boundary coherence the
position coder already exploits for 1.28× — in which case the composite falls toward the position price. It may
also not; the point is that the pool is currently marked `do_not_spend` on an undecomposed number.

---

## §3 SURFACE C — instrument distortion

Every "the reading was wrong" finding on 07-31, placed as a signed relative bias. This is a continuum, and today's
readings populate five orders of magnitude of it.

| bias | magnitude | direction | source |
|---|---:|---|---|
| gate **level** vs n600 | **0.374%** | ~unbiased | ep805 gate 0.0040519 vs full_confirm 0.004067128 |
| gate **delta** overstates descent | **7.6%** | optimistic | independently matching gc14's 7.2% |
| A1 gate block over-weighting | **16.67×** | amplifies block drift | non-probability sample reduced with an unweighted mean (gd1 #817) |
| A1 block hardness vs population | **+6.3 – 8.3%** | harder | gd1 |
| A1 **block** Lane composition | **−16.2%** | Lane-poor | gd1 |
| A1 **36-pair gate** Lane composition | **+3.3%** | Lane-**rich** (`lane_frac` 0.006050 vs pop 0.005855) | round-2 review |
| gate aliasing | ~30-gate oscillation sampled at 5×5 | sign flips | `n_points=5` × `--gate-every 5`; a 7-gate monotone run ≈ 1 alias period |
| byte-ledger error spread | **2,018 B = 0.001344 S** | non-monotone | **45% of the 4,497 B spread it was arbitrating**; w02 vs ep809 INVERTS (ledger −44 B, real +20 B) |
| restart p, per-epoch vs raw | **2×** | optimistic | 0.0056 vs 22/1953×2 = 0.0225 |
| tr1 rehearsal camera agreement | **0.1987 → 0.99997346** | false BLOCKED | `_mlx_reference` never sets `_quant_engaged=True` (#828); float max_abs 55.14 → 2.14e-4 |
| detection coverage — lever_registry | **1/171 modules** = 0.585% | blind | sb2 |
| detection coverage — findings gate | **0/1,260** = 0% | blind | sb2 |
| detection coverage — duty-queue | **116/177** = 65.5% | partial | sb2 |

**The "silently-wrong instrument" count is itself a running census, not a verdict: 4 → 3 → 4 within one day.**
Round 1 claimed four; round 2 **refuted one of them in direction** and corrected to three; #828 later found a
genuinely new fourth. The count is a coordinate that moves as the census proceeds — it was never a fact about the
world, and reading it as one is what made the correction feel like a reversal.

**MAIN's own round-2 self-refutation is the cleanest point on this surface, and it is SOUND.** MAIN's round-1 root
cause (constraint compares an n600 budget against a Lane-**poor** 36-pair subset ⇒ g biased negative by
construction) was refuted on two independent legs: (a) the −16.2% is the **4-pair block**, while the 36-pair
**gate** — what `realized_lane_s` is actually computed on — is **+3.3% MORE** Lane, so any mismatch pushes g
**positive**; (b) a construction bias would be ~constant, but g grows **14.02×** (−0.003452 → −0.048409) tracking
the real Lane descent. MAIN then flagged that this was *"the exact block↔gate conflation I had RETRACTED ~200 lines
earlier in the same document."* That is the audit working. Call it sound.

---

## §4 SURFACE D — build completeness (the generator of false negatives)

`ddm_sb2` (#819) measured the distance from *designed* to *fired* as **five grades**, not a bit:

| grade | count | detectable by |
|---|---:|---|
| 1. built-and-fired | **2** | everything |
| 2. built-never-fired | **165** | registry |
| 3. **BUILT-ELSEWHERE-UNWIRED-HERE** | **8** | **nothing automated** |
| 4. designed-stub | **10** (2 silent) | partial |
| 5. not-even-designed | **12** | nothing |
| **total** | **197** | — |

Repairs measured in the same unit: registry **1 → 171** modules, **116 → 177** factories, stubs **0 → 10**, cached
**1096×** *(because "a slow gate is a disabled gate — which is how this survived")*.

**Why this surface generates negatives on the other surfaces.** Three measured instances from 07-31:

1. **4 of gc15's 5 reset arms had no mechanism ANYWHERE.** `tac.optimization.reset_operator` was verified wider
   than predicted: tr1's optimizer is one line, all 6 `save_checkpoint` sites pass `opt_state_flat={}`, and a
   64-flag census returns zero for `adam|beta|bias|moment|restart|precond|warmup`. A comparison across those arms
   was reading **build asymmetry**, not physics.
2. **"from-birth-KD vs warm continuation resolved by BUILD ASYMMETRY, not measurement."** The charter marks
   from-birth-KD `DEFERRED`; `TR1KDWarmStart` was **never buildable on tr1**. *A `DEFERRED` label on an unbuilt
   branch reads downstream as a resolved decision.* MAIN named this itself — sound, and it is the single most
   transferable statement of the day.
3. **rowband D8 never executed one step** ⇒ any D16 verdict is a build artifact. **lane-guard component form is a
   stub while the pixel form fired in all three windows.**

Independently, sb2 reproduced gc15's per-boundary impulse as **1,212.57** sign-steps against the real MLX Adam,
with the refinement that truncating at a few thousand steps **under-prices** it. gc16 then bounded restart
enthusiasm correctly: **I = 1,212.6 is FIXED and cadence-independent** (99.7% delivered by epoch 67 regardless of
window length), and gc14's geometric envelope caps 2× cadence at **~0.019 S ≈ 2.4% of the 0.802 S gap**. That is a
level-set statement with an explicit ratio, not a kill — sound.

---

## §5 SURFACE F — governance / wall-clock (the surface that actually decided the day)

| decision | quantity | level set | ratio |
|---|---:|---:|---:|
| no window_04 | projected end ~6.5 h | 6.0 h cap / 20:17:42Z hard stop | **1.083×** |
| ALARM #3 cure refused | projected 6.602 h | 6.0 h cap | **1.1003×** |
| b4r earlier refusal | 50.5 min dead time | same cap | — |

**Two of the day's three terminal calls were decided here, and MAIN said so explicitly** — *"NO WINDOW_04 — on
GOVERNANCE, not measurement"* — while noting the endpoint **measurement** stage is explicitly not truncated by the
cap. That is exactly the right shape: the constraint is named, its level set (6.0 h) is a chosen constant, and the
distance to it is reported. **Sound.**

The refusal of the prescribed ALARM #3 cure rests on **four** grounds, only one of which is the cap, and the
strongest is a *prediction from telemetry*: with `g < 0` and `η_λ = 66.225`, `λ_init = 0.1` **decays back toward 0**
⇒ the cure is predicted INERT, so 2.17 h and a cap breach would buy a null. That is a derived, falsifiable
prediction, not a preference — sound, and it names its own test.

### The lane guard as a level-set problem (the reformulation is already correct in-tree)

The guard never actuated: `lambda_lane == 0.0`, `uniq == 1`, **all 58 gates, both windows**; `complementarity == −0.0`.
Constraint slack fraction −g/budget against fixed `budget_s_units` **0.12589**:

| point | g | slack fraction |
|---|---:|---:|
| first gate | −0.003452 | **2.74%** |
| w02 span | −0.009189 → −0.031102 | 7.30% → 24.7% |
| w03 span | −0.033588 → −0.048409 | 26.7% → **38.5%** |

realized Lane fell 0.116701→0.094788 (w02) and 0.092302→0.077481 (w03). **λ = 0 under g < 0 is correct KKT for a
genuinely slack constraint** — we simply never reached the level set. And the level set's *location* is
`LANE_BUDGET_S_UNITS = 0.12589` (`src/tac/optimization/lane_guard.py:59`; sister
`LANE_BUDGET_DSEG = 0.0012589` at `:60` — **exactly** fl1's ep641 Lane residual d_seg, confirming the placement in
§7), **pinned at the starting Lane S and never tightened**. Round
2 derived the fix: *"A guard with a fixed starting budget can only protect against becoming worse than the start;
it can NEVER HOLD A GAIN"* → **ratchet** the budget (`budget ← min(budget, realized + derived margin)` at accepted
boundaries). **That is precisely "the level set is a function of our own knob."** Sound, and already the right
shape.

Note the sequence honestly: the ALARM's Lane premise was **refuted a third time** by direct per-class measurement —
through the entire window_03 rise **Lane keeps IMPROVING** (0.0959 → 0.0820 S, the best Lane of any candidate);
the rise is carried by **Undrivable (+0.026111 S)** and Road. The guard was inactive *because Lane was fine*.

---

## §5b SURFACE E — the joint exchange rate (ja1 / QA73), and the identity that ties it to Surface B

**Base (own-vehicle advisory row, archive `b6365270`, 359,750 B; `.omx/research/ddm_ja1_joint_sensitivity_atlas_20260731.json`):**

`S = 0.992972 = seg 0.431179 + pose 0.322250 + rate 0.239543` (sums EXACTLY), with
`d_seg = 0.00431179`, `d_pose = 0.0103845`.

**⚠ Coordinate correction, stated plainly:** the standing "pose ≈ 1.24 S is the largest axis" row is a
**different, earlier coordinate on the same axis**. At the ja1 measured base pose is **0.322250** and **seg
0.431179 is the largest axis**. The MEMORY row itself records the move (*"our own solve already did 5.2×
(1.53→0.29)"*). Carrying 1.24 forward as current is the staleness confound in miniature — one axis, two
coordinates, and the ordering between axes **inverts** between them.

### E.1 The measured ranking (`ddm_ja1_joint_waterfill_table_20260731.json`)

| rank | id | pool | ΔS | ΔB | ΔS/kB | label |
|---:|---|---|---:|---:|---:|---|
| 1 | QA66 | photometric | **−0.013385391583283423** | **150 B** | **−0.08924** | REALIZED-live-base |
| 2 | QA72a+QA54 | information / pose-precision | — | **0 B** | — | DERIVED → REALIZED (owed); **DECISIVE $0** |
| 3 | QA68 | pose-content (expert menu) | — | ~1–3 KB | — | DERIVED headroom, **UNBUILT** |
| 4 | QA65 | pose-precision (storage) | ≤ ~one p0-ULP | few hundred B | — | DERIVED-**bounded-small** (demoted from "~0.03 S optimistic") |
| 5 | QA24 | seg-**CAPACITY** (re-burn) | ≤ **−0.098** seg+rate (lower bound) | re-allocates the token base | — | DERIVED lower bound, HEAVY, operator-GO, **PARALLEL** |

**Standing law on this surface:** *"no rung fires on axis identity; every rung fires on its JOINT realized
exchange rate read from THIS table. Pools are NON-ADDITIVE (same-pool levers COMPETE, never sum)."*

### E.2 The three `saturated_do_not_spend` pools — all three are Surface-A/B coordinates

| pool | why saturated | measured |
|---|---|---|
| token-cell bytes (gr1) | `cell_drop50` **is** the seg+rate **knee** | restore **+0.047 S @ +80,615 B** · drop-more **+0.052 S @ −81,406 B** — both DOMINATED |
| rate / lossless container | kl1 pose reformat + manifest/selector deflate already CONSUMED in v4c | token stream at SMEVR entropy floor (QA08 ceiling-closed ≤1.6 KB); ~0 cheap slack |
| **seg corrections** | co9 white-jitter | **1.45 B/flip ≈ water 1.27** ⇒ *"corrections do not descend seg here"* |

**The third row is §2.2's composite, consumed as an allocator exclusion.** So the missing position/label
decomposition is not an academic point: it is the sole basis on which an entire pool is marked *do-not-spend*.

`allocation_surprise` (verbatim, and it is exactly the right shape): *"The axis-reflex reads v4c (seg 0.431 =
LARGEST axis) and says 'attack seg with bytes'. MEASURED: the seg BYTE pool is SATURATED at its knee … Every
cheap LIVE byte lever is on the POSE axis … Biggest-axis-first is exactly wrong here."*

### E.3 **THE IDENTITY — Surface B's water level IS Surface E's seg↔rate exchange rate**

The ja1 table carries `rate_break_even.tokens_B_per_flip = 1.273`, noted as *"25·ΔB/N = 100·Δd_seg at the seg
water"*. Solving that identity exactly:

```
water = 100 · (37,545,489 / 25) / (600 · 384 · 512)
      = 100 · 1,501,819.56 / 117,964,800
      = 1.2731082153320312  B/flip
```

The pp1 law's registered `WATER_B_PER_FLIP = 1.2731` differs from this by **8.215e-6** — i.e. it is the same
number, rounded.

**Consequence: the 1.2731 B/flip "water level" is not an empirical constant, a coder property, or a measured
threshold. It is the contest score function's own seg↔rate exchange rate, exact.** Two artifacts derived it
independently — pp1 as a coding water level, ja1 as an allocator break-even — and never noticed they were the
same object.

**Reach of the identity.** `1.2731` is not confined to pp1: it is also a live GO gate elsewhere in the organ —
`src/tac/ddm_costate_organ.py:1325` (*"GO iff measured coded B/err < 1.2731 AND composed S improves"*) and
`:1293` (*"B/flip of a token-coordinate entry vs the 1.2731 water"*). Every one of those gates is comparing a
candidate's **composite** price against the scorer's exchange rate, with no position/label split.

That fixes the meaning of every "break-even" on Surface B. `1.45 / 1.2731 = 1.13895` says: *this correction
stream bought seg at 1.139× the price the score itself charges for rate.* **That price is fixed by the scorer
and is not ours.** What IS ours on that ratio is (a) coder efficiency — measured 11.44× span — and (b) the
position/label split — measured to be **≥51.6% non-position** at our base. Both of our knobs are larger than
the 13.9% gap.

---

## §6 Where we stand, and which way each surface falls

| surface | our coordinate | direction it falls | next thing that moves it |
|---|---|---|---|
| **A** ρ = d_seg | **3.943024e-3** (ep854), in-band, 7.86× above `rho_c` | left, by burn capacity | continuation; the ratchet guard to *hold* gains |
| **A** (edges) | `rho_c` 5.015e-4 out of a measured 2.99× span | edge moves left with better coders | r7's 14-coder race, priced against 3 of 5 per-class floors |
| **B** price | QA03 composite 1.13895× water; position only **0.5514×** water | falls with coherence and with label-cost separation | decompose QA03's 1.45 into position + label |
| **C** instrument | Lane verdicts fail L3 apparatus-validity | falls with probability sampling + HT estimation | gd1's HT estimator on the same 58 points, 36 renders, $0 |
| **D** build | 197 rows, 8 undetectable-by-anything | falls with wiring; **grade 5 is invisible by construction** | #820 TR1ResetOperatorWiring; per-vehicle closure audits |
| **E** exchange | seg 0.431179 > pose 0.322250 > rate 0.239543 (S=0.992972 @ 359,750 B); **all 3 cheap-byte pools saturated except pose** | fire on JOINT rate, never axis identity | QA72a+QA54 ($0, rank 2, DECISIVE) before building QA65/QA68 |
| **F** governance | 6.0 h cap, breached by 1.08–1.10× | it is a chosen constant | a sealed charter for a new experiment, not a cap-busting tail |

---

## §7 Unification — the shared degrees of freedom

1. **Coder quality ≡ band edge (Surface A ⊗ B).** They are one DOF. The lower edge moved 1.7132× from uniform to
   coherent and spans 2.991× to incoherent. Three of five per-class flicker floors sit within 1.76× of the edge.
   ⇒ *the per-class correction question is a coder question, and the coder axis is exactly as large as the gap.*
2. **Formulation expressiveness ≡ distortion floor (Surface A).** The 5.3184e-3 floor is the smooth-label level
   set; phase-faithful renderers sit 17.93× below it. fl1 states this correctly and refuses the hard-floor read.
3. **Build completeness ≡ verdict validity (Surface D → C → A/B).** An unbuilt branch produces a `DEFERRED` label
   that reads as a decision; 4 of 5 reset arms had no mechanism; D8 never ran a step. ⇒ *before any A/B verdict,
   the arms' mechanisms must be confirmed present — sb2's grade-3 (8 rows) is detectable by nothing automated.*
4. **Sampling design ≡ verdict scope (Surface C).** 16.67× block over-weighting and a ~30-gate alias are why every
   Lane verdict today is INSTANCE-scoped. The same 36 renders re-read with an HT estimator is $0.
5. **Water level ≡ seg↔rate exchange rate (Surface B ⊗ E).** `1.2731 B/flip` is `100·(37,545,489/25)/(600·384·512)`
   **exactly** (§5b.3). pp1 registered it as a coding water level, ja1 as an allocator break-even; they are one
   number. ⇒ *every break-even ratio on Surface B is a statement about the score function, and the only movable
   terms in it are ours: coder efficiency (11.44× measured) and the position/label split (≥51.6% non-position).*
6. **The chosen constant is the recurring shape — with ONE exception that is genuinely not ours.**
   `budget = 0.12589` (pinned at start), `corner-C = 0.06 S`, `cap = 6.0 h`, `band upper = 1e-2`, `β₂ = 0.999`,
   `falsifier threshold = 0.14071` — **all ours**. `water = 1.2731` is **not**: it is the scorer's arithmetic.
   **Every "DEAD / break-even / infeasible / fires-for-all-5" on 07-31 is a statement about where one of the
   first six was placed.** Distinguishing which constants are ours from which are the scorer's is the whole
   difference between a wall and a knob.

### The falsifier with no power (fl1, worked in full)

fl1's pre-registered gc13 R3 falsifier fired for **all 5 classes** (Road 7.13×, Lane 13.08×, Undrivable 5.02×,
Movable 5.33×, MyCar 16.77×). Those five numbers are **not five measurements**. corner-C is a
proportional-to-current-control split, so `cornerC_c / resid_c` is a **constant 0.14065 – 0.14082** across all five
classes. Therefore:

> `floor/cornerC  ==  (floor/resid) / 0.14071`

The falsifier's column is the **piercing ratio rescaled by one constant**. Its level set sits at
piercing ratio = **0.14071**, and the measured piercing ratios are **0.7067 – 2.3603**. **The nearest class
(Undrivable) is 5.02× above the trigger; the test could not have failed to fire at any plausible operating point.**

The quantity it *rescaled away* is the informative one:

| class | floor ρ | resid ρ (ep641) | **piercing ratio** |
|---|---:|---:|---:|
| Undrivable | 3.939e-4 | 5.574e-4 | **0.7067** (above floor) |
| Movable | 2.847e-4 | 3.792e-4 | **0.7508** (above floor) |
| Road | 1.8894e-3 | 1.8845e-3 | **1.0026** (exactly ON the floor) |
| Lane | 2.3162e-3 | 1.2589e-3 | **1.8399** (pierced) |
| MyCar | 4.343e-4 | 1.840e-4 | **2.3603** (pierced) |
| aggregate | 5.3185e-3 | 4.2640e-3 | **1.2473** |

Spread **3.340×**, and Road sits at **1.0026 — a second exact on-the-level-set coordinate**, alongside QA03's
1.13895. fl1 *did* publish both columns and *did* reach the correct scoped verdict (**NO re-waterfill**, binding
leg = ru1's ~6e-4 GT-jitter-typed reachable, binding phase-debt = Lane). **The artifact is sound.** The loss is
only that the verdict was drawn from the rescaled column, where the threshold's 5–17× distance is invisible.

---

## §8 Denominator + honesty

- **Window:** 2026-07-31 only. 50 commits, 43 `.omx/research/*20260731*` artifacts.
- **Negatives enumerated:** **34 of the 43** artifacts match the negative-verdict lexicon
  (`REFUTED|NO-GO|DEAD|DOMINATED|FAILED|EXHAUSTED|SEALED|INFEASIBLE|EMPTY|KILL|FALSIFIED`, case-insensitive) —
  that is a **lexicon match, not an adjudicated count**, and is stated as such. Of the 35 `.md` artifacts, 5
  carry no match at all (`ddm_us1`, `ddm_tt1`, `ddm_dg1`, `ddm_b2p`, `ddm_b2p_DAG_FEED`). Every negative reached
  in this unit is placed on one of the six surfaces; none is discarded.
- **Placed with full arithmetic here:** the pp1 band (14 measured points + 4 edges), fl1 (5 floors + 5 piercing
  ratios + the falsifier level set), QA03/QA04, the lane-guard slack trace (58 gates), the instrument table
  (12 rows), sb2 (5 grades / 197 rows), the governance ratios.
- **Placed in the two folds (§10, §11):** the 25 remaining artifacts, incl. the five convocations. Denominator
  for the whole unit: **43 artifacts found · 43 read · 0 unreachable**; every negative reached is on one of the
  six surfaces, none discarded, no seventh surface required.
- **Unreachable in this unit:** the `full_curve_ssd` at
  `/Volumes/VertigoDataTier/pact/ddm_pp1_20260728/r2_band_lemma_curve_n600.json` was not opened (the committed
  receipt carries the 14 points used here); QA03/QA04's raw support fields were not re-read, so the
  position/label split is DERIVED from the law's curve, not measured on QA03's own support.
- **Negative-existence discipline:** no claim here is of the form "X does not exist." Where something was not
  found, the scope is stated: *"the position/label decomposition of QA03's 1.45 B/flip does not appear in
  `ddm_costate_organ.py:2480-2493`, in `ddm_pp1_...20260728.py`, or in the pp1 receipt"* — those three surfaces,
  and no wider claim.
- **Sound verdicts, called sound, with reasons:** fl1's scoped NO-re-waterfill · MAIN's round-2 self-refutation of
  its own lane-guard root cause · MAIN's KKT reading of λ=0 · MAIN's governance-not-measurement framing for
  window_04 · MAIN's four-ground ALARM #3 refusal (the inertness prediction is derived and falsifiable) · MAIN's
  byte-ledger law (*select on d_seg-dominated margins; byte-close any d_seg-TIED selection*) · gc16's cadence bound
  (~0.019 S = 2.4% of the 0.802 S gap) · sb2's build-asymmetry finding.
- **Authority:** everything is `[macOS-CPU advisory]` / derived. Pointer **0.1910828242 [contest-CPU] UNMOVED**.

## §10 Harvest fold — the remaining 07-31 negatives, placed

Read-only harvest of 25 further artifacts. Every negative below is a coordinate on one of the six surfaces.
Nothing is a new surface.

### §10.A — Surface A: more coordinates on ρ, and a DECOMPOSITION of our position

**New coordinates (all `[macOS-CPU advisory]`, n600 unless noted):**

| ρ = d_seg | what | source |
|---:|---|---|
| **1.52e-4** | **ms2r_r3 compress-time EXACT SOLVE** (17,927 errors) | sg1 §2 |
| 2.966e-4 | PR130 rail | (already placed) |
| 3.88778e-3 | sg1 re-measured tr1 endpoint (vs gr1's 3.89011e-3, Δ **2.3e-6**) | sg1 |
| 3.882e-3 | gr1 `drop35` @ **439,836 B** | ja1 atlas surface 9 |
| 3.943024e-3 / 3.947e-3(n48) / 4.310379e-3(n600) | **gr1 `cell_drop50` @ 359,221 B — the SAME archive, n48 vs n600 differ by 9.2%** | ja1 atlas 9 |
| 4.264052e-3 | r1c ep641 (xp1 base) | xp1 |
| 4.9411e-3 | pa1r `control_tail` ep499/500 (fp1 + qa92 base) | fp1/qa92 |
| 5.013e-3 | gr1 `drop63` @ **277,815 B** | ja1 atlas 9 |
| 5.169e-3 | bc1 from-birth ep399 | cn3 §5 |
| **8.305e-3** | **fp1 BR-B flat-paint RECEIVER FLOOR** (lower bound, GT-perfect input) | fp1 §1 |
| **0.499366** | **fp1 TRAINED-HEAD f′ — a second, independent wall** | fp1 §3 |

**A.4 The amortization decomposition — our position on ρ splits into a floor and a gap.** sg1 measured, per class,
renderer errors vs the compress-time exact-solve concede:

| class | renderer err | exact-solve concede | gap | gap % | err-rate in class |
|---|---:|---:|---:|---:|---:|
| Lane | 177,631 | 2,556 | 175,075 | **98.6%** | 25.72% |
| Road | 141,985 | 7,833 | 134,152 | 94.5% | 0.52% |
| Movable | 69,158 | 1,346 | 67,812 | 98.1% | 4.74% |
| Undrivable | 58,907 | 3,622 | 55,285 | 93.9% | 0.10% |
| MyCar | 10,940 | 2,570 | 8,370 | 76.5% | 0.037% |
| **TOTAL** | **458,621** | **17,927** | **≥440,694** | **≥96.1%** | — |

Amortization ratio **0.00389 / 1.52e-4 = 25.58×** (Lane alone **69.5×**). **This is not a wall — it is a distance
to a level set we have already measured a witness at.** sg1 states the containment caveat (the per-class
subtraction assumes exact-solve errors ⊆ renderer errors), so 96.1% is a **lower bound**.

**⭐ CROSS-SURFACE: the exact-solve floor 1.52e-4 lies BELOW `rho_c` = 5.015e-4, by 3.30×.** So a witness at the
exact-solve floor would be in the **concede** regime — a correction stream would be priced out by position cost
alone. **The correction-rationality band and the achievable-distortion floor bracket our position from opposite
sides**, and the whole live campaign (3.88e-3 – 4.94e-3) sits between them, 7.7–9.9× above `rho_c` and 25.6×
above the solve floor.

**A.5 What moves the floors (all measured this day, none constant):**
- **Formulation** — smooth-label 5.3184e-3 → phase-faithful 2.966e-4 (**17.93×**); flat-paint receiver 8.305e-3 →
  textured render 4.9411e-3 (**1.68× the OTHER way** — fp1's own point: the flat chart is *worse*, not better).
- **Amortization** — learned renderer 3.89e-3 → compress-time solve 1.52e-4 (**25.58×**).
- **Head decodability** — fp1's 3×3-conv head on the frozen trunk plateaus at CE 0.550, f′ **0.499366**, ~**60×**
  above its own receiver floor. Two walls with **orthogonal class signatures**: the receiver floor is
  Road 62% + Movable 24%; the head wall is Road 46% + MyCar 50%. **Neither is "the" wall.**

### §10.B — Surface B: the correction price, extended by an exchange rate nobody named as one

**B.4 QA92's ERF collateral is a PRICE, and it is the day's largest measured negative.**

| quantity | value |
|---|---:|
| target pool **P** (erased super-nucleus Lane) | **0.04189 S** |
| **O** oracle recovery fraction (pixel flip-mass) | **0.40732** (comp-level 0.421) |
| **F** flat-prototype recovery fraction | **0.19394** (comp-level 0.233) |
| P·O recovered | **0.01706 S** |
| P·F recovered | **0.00812 S** |
| **collateral, oracle** (off-target flips created) | **+0.31698 S** |
| **collateral, flat** | **+0.23300 S** |
| **JOINT ΔS oracle / flat** | **+0.29992 / +0.22487 S (both WORSE)** |
| **collateral / recovery** | **18.58×** |

Base = pa1r `control_tail` d_seg 0.0049411, n600, support = GT Lane 8-conn components >5 px, erased iff <50%
Lane-classified; oracle fill = real GT camera RGB; flat fill = fp1's solved prototype `[77.43, 86.71, 118.53]`.

**The continuum this prices: S created off-target per S recovered on-target — an exchange rate, 18.58 : 1.** That
is the same *kind* of quantity as the 1.2731 B/flip water level, on a different pair of axes.

**⭐ The level set here is a knob nobody swept.** Per-class collateral, oracle vs flat:
`Road +0.125 / +0.130` · `Lane +0.034 / +0.026` · `Undriv +0.055 / +0.024` · `Movable +0.065 / +0.028` ·
`MyCar +0.020 / +0.018`. **Road's collateral is essentially INVARIANT to fill content (+0.125 vs +0.130, 4%)** —
so the dominant damage term is caused by the **stroke GEOMETRY**, not what is painted. Stroke geometry is a
`+1 px binary dilation`, **fixed at one value and never swept**. The identity-fill control proves the artifact
term is exactly zero (`max|Δd_seg| = 0.0`), so the collateral is real receiver physics — but it is physics
measured at a single point of a stroke-width continuum whose per-class signature says that continuum is binding.
qa92 correctly does **not** close the family (O = 0.407 ≥ 0.25 ⇒ `verdict_scope = FORMULATION`). **Sound.**

**B.5 Other prices, placed against the 1.2731 water:**

| candidate | measured B/flip (or B/err) | vs water 1.2731 |
|---|---:|---:|
| gr1 token-granular corrections | **0.04 – 0.51** | **2.5× – 31.8× BELOW** ⇒ DOMINATED |
| W1-COH phase carrier | **0.075 – 0.141** | **9.0× – 17.0× BELOW** |
| QA03 full-population GN/CG (composite) | **1.45** | **1.139× ABOVE** |
| law's coherent POSITION at our base | **0.7024** | **1.812× BELOW** |

**Note the shape:** the two carrier families that were called dominated are dominated by **an order of magnitude**,
while the correction that was called break-even is **13.9% over** — and its position half is **81% under**. Those
are not the same verdict at different confidence; they are opposite ends of the same real line.

**B.6 The renderer-weight rate pool, priced exactly (fh1 R5).** renderer 3,284 B of 253,858 B counted ⇒ the entire
lever ceiling is `25·3284/37,545,489` = **0.00219 S = 1.3%** of counted bytes. Demoted to rank 9/10. **Sound** —
this is a ceiling, not an estimate, and it is computed rather than eyeballed.

### §10.C — Surface C: the noise floor that bounds every verdict of the day

**⭐ C.1 The vehicle is RERUN-NONDETERMINISTIC, and the scatter was measured (lg1).** Identical code + argv,
both devices:

| device | run A | run B | Δ |
|---|---:|---:|---:|
| Metal, ep1 | 0.4476 | 0.4413 | **0.0063** |
| forced-CPU, post | 0.4544 | 0.4813 | **0.0269** |
| forced-CPU, pre | 0.5061 | 0.5217 | **0.0156** |
| counted bytes | — | — | **±2 B even at ep0** |
| ep0 realized-gate d_seg | 0.5078303019205729 | identical | **bit-equal across all 8 OFF/ON runs** |

Coordinates: **ep1/ep2, n=4 pairs** — early training, tiny batch; whether it shrinks by ep400/n600 is
**unmeasured**. The ep0 bit-equality shows uint8-R + argmax absorbs sub-LSB float noise at ep0 only.
**This is the single most consequential number in the harvest**: several of the day's verdict deltas
(QA79 −8.4e-5; sg1's 2.3e-6 cross-instrument agreement; xp1's ±0.002) are **1–3 orders of magnitude below**
this scatter. They were taken on frozen bases where scatter should be far lower — **but no arm established the
frozen-base noise floor**, so the comparison is open, in both directions.

**C.2 The instrument table, extended:**

| bias | magnitude | source |
|---|---:|---|
| A1 gate total design error (τ0.25 / τ1.0 / lane_frac) | **1.914% / 1.769% / 3.337%** | gd1 JSON |
| **HT fix DEGRADES the frozen-seed error on 2 of 7 proxies** | **lane_frac +68.90% worse · boundary_frac +30.53% worse** (block and SRS terms have opposite signs and partially cancel in the unweighted estimator) | derived from gd1 JSON — **not in the memo** |
| Neyman re-design prediction, same 32-pair budget | var ratio **0.12766**, **−64.27% SE**; allocation `[8,3,4,17]` — **stratum 4 absorbs 53% of the sample**; stratum SDs span **5.33×** | gd1 |
| provenance bijection gate | **3,862 live violations, WARN-ONLY, denominator not stated** | gd1 |
| lever_registry scan set | **1 of 171 modules (0.585%)** — `_module_source()` reads one file | cn3, self-verified |
| codex-findings gate scan set | **0 of 1,260 files** (mtime < 3 days) — structurally vacuous | cn3 |
| Catalog #396 | correct scan set, **433 live / 108 in-window, never strict-flipped** | cn3 |
| factory discovery, repaired | 116 → **177** (+52.6%); stubs 0 → **10** (2 SILENT); cached **1096×** | sb2 |
| fire rate | **2 fired of 177 visible factories = 1.13%** | sb2 |
| open task ledger | **51 non-terminal: 29 STALE · 18 DUPLICATED · 2 free-unblocks · 1 ORPHANED · 1 LIVE-AND-CORRECT (1.96%)** | cn3 |
| rv1 `reactivated` field | **hardcoded `False` literal, no writer; 5 of 8 rows already discharged 2–3 days earlier; the test asserted the constant** | rx1 |
| `epochs_per_gate` 5 vs 10 | **flips the birth-completion verdict**: r1c `fired=True, slope 1.60, ε 1.88` vs lp2 `fired=False, slope 7.60, ε 3.34` — same producer, same nominal object | r1c / lp2 |
| Hilbert "+452 B" | welds **425 B of 2D-vs-1D context** + **27 B of order**; context-matched, **serpentine BEATS raster by −37 B**, and by **14 B (23%)** on the keep-mask under brotli11 | gd1 receipt |
| byte-ledger coder surrogate | zlib overstated SMEVR by **16%** on a synthetic field (1136 vs 951 B); real-field bias unmeasured | b2b T5 |
| EMA clamp | derived 0.99986667 clamped to 0.9995 ⇒ warmup **15,038 → 4,000 = 3.76× wrong horizon, live in the burn** | b2b T6 |

**⭐ C.3 The law cn3 proposed, and it is the correct generalization of everything in this table:**
> *"A gate's LIVE-COUNT-0 is meaningless until its DENOMINATOR is asserted. Every gate should report
> `checked N of M`, and a gate whose N is 0 should fail loud, not pass quiet."*

**"0 violations" welds "clean" with "not looked at."** Measured N/M today: **0.585%** · **0%** · **1.13%** ·
**84.6%** (us1's "0 CRITICAL" = 11 of 13 re-derivable in-mode, which us1 states itself). **Sound, and it is the
single most transferable finding of the day.**

### §10.D — Surface D: build completeness, confirmed and extended

- sb2's five grades stand: **2 / 165 / 8 / 10 / 12 = 197**; grade 3 (BUILT-ELSEWHERE-UNWIRED-HERE, 8 rows)
  detectable by **nothing automated**. Of gc15's five pre-registered reset arms, **exactly 1 of 5 (20%) was
  reachable** — the race would have measured the same arm five times. The gate `check_no_stub_lever_factories`
  is **WARN-ONLY at live count 10, deliberately not flipped** (all ten are other arms' chartered builds) —
  **sound**, and it names its flip condition (live count 0).
- cn3: `ddm_b2b`'s burn-2 stack is **BUILT, TESTED, 4 commits landed, NEVER FIRED**, while the slot its own memo
  names as its consumer went to continuation windows measured at **r = 0.310, 2.3% of the gap**. cn3's corollary
  is right and new: **a BUILD with a named consumer and no fire receipt is a costlier orphan than an orphaned
  finding, because its cost is already sunk.**
- **Two arms, opposite dispositions on ONE continuum, same day:** fh1 proposes racing `class_weight_lane`
  ∈ {2, 4, 8} (never-fired at default 1.0); sg1 records lv1 §D.4 **REJECTED 2.0** on bulk price and rules
  "keep 1.0". Neither cites the other. **The knob is a real line sampled at 1.0 and 2.0 with one rejection —
  that is the finding, not either disposition.**

### §10.E — Surface E: the exclusion arithmetic, and a FOURTH appearance of the exchange rate

**E.4 cn3's joint-exclusion arithmetic — the day's cleanest surface statement:**

```
banked pose fallback (gc14 §13)                 0.12689
best byte-closed rate ever built (Knee-B,
  174,578 B = 25·174578/37,545,489)           + 0.11624
                                              ---------
                                                0.24313   with ZERO seg budget
bar  min(0.15, official 0.172141)               0.17214
                                              ---------
EXCESS                                        + 0.07099   ⇒ 41.2% over
```
Required multiples against corner C: **seg 7.19× · pose 19.19× · rate 2.77×**. Neither banked component is
individually excluded (pose alone = 74% of the bar; Knee-B alone = 68%); **the exclusion is JOINT** — and cn3
explicitly guards the subtraction against the non-additive-pools law by noting the two are in *different* pools.
**Sound, and it is the correct shape**: a statement about where a point sits relative to a level set, not a kill.

**⭐ E.5 THE SAME EXCHANGE RATE, FOUND FOUR TIMES — and the one that steers the live burn is 35% off.**

| appearance | form | value | agrees with the score? |
|---|---|---:|---|
| pp1 band lemma | `WATER_B_PER_FLIP` | **1.2731 B/flip** | ✔ (8.2e-6 rounding) |
| ja1 waterfill | `rate_break_even.tokens_B_per_flip` | **1.273 B/flip** | ✔ |
| exact score arithmetic | `100·(37,545,489/25)/(600·384·512)` | **1.2731082153320312** | — (it IS the score) |
| **b2b derived `--w-rate`** | `(25/37,545,489)·n/8`, n = 923,136 counted tokens | **0.0768** | ✔ (same constant, loss-weight form) |
| **the LIVE burn's `--w-rate`** | inherited constant | **0.05** | ✘ — **65.1% of S-commensurate** |

`25/37,545,489 = 6.6586e-7`; `× 923,136/8 = 0.076834`; `0.05 / 0.076834 = 0.6508`.

**The score's seg↔rate exchange rate is one number wearing four costumes.** Three of the four agree. The fourth
is the only one **actually steering a live training run**, and it under-weights the rate gradient by **34.9%**,
biasing the burn toward spending bytes. r1c **refused** to change it mid-lineage (*"a mid-lineage change would be
an un-raced lever"*) — **which is the right call on comparability grounds and does not make the number right.**
The size of the resulting byte over-spend is **not measured anywhere.**

**E.6 The token RD curve has exactly three measured points** (ja1 atlas surface 9): `drop35` 439,836 B @ 3.882e-3
· `drop50` 359,221 B @ 3.947e-3 (n48) / 4.310e-3 (n600) · `drop63` 277,815 B @ 5.013e-3. The `saturated` verdict
comes from a symmetric ~+0.05 S penalty for an ~80 kB move in **either** direction — a sharply-curved,
flat-bottomed minimum sampled at three points. **Note the n48↔n600 gap on the SAME archive: 9.2%** — larger than
several deltas adjudicated elsewhere today.

**E.7 QA65 demoted by stage attribution, correctly.** pi2's one-ULP ceiling **0.040 S** assumed |p0| = 35; the
shipped p0 median is **31.2** ⇒ f16-ULP **0.015625**, and per-pair d_pose is **784×** content-spread with
**90% of pose mass in 88 pairs (14.7%)**. Measured storage tax: **10.5%** of recoverable at plain f16, **1.67%**
at offset f16 (the offset lattice removes **~84%** of it). **Pose is content-limited, not storage-limited** —
a re-placement of a rung on a continuum, from a number that was "optimistic," to a bound. **Sound.**

### §10.F — where the harvest changes the picture

1. **Our position on ρ is bracketed on both sides by measured level sets** (exact-solve floor 1.52e-4 below;
   smooth-label floor 5.3184e-3 above; band edges 5.015e-4 / 1e-2 across), and **96.1% of the distance to the
   lower one is amortization gap** — the largest single characterized headroom of the day.
2. **The day's two biggest negatives are both PRICES, not walls:** QA92's 18.58:1 collateral-to-recovery and
   QA03's 1.139× composite. Both were measured at a single point of a knob (stroke width; position/label split)
   that their own per-class data identifies as binding.
3. **Nearly every "count" reported today has an unstated denominator**, and cn3 named the law. The lg1 rerun
   scatter is the same defect in the continuous domain: a verdict delta reported without its noise floor.

---

## §11 Convocation fold (gc12 / gc13 / gc14 / gc15 / ph3) — the surfaces, sharpened

### ⭐ §11.1 THE EXCHANGE-RATE SURFACE HAS A MEASURED SIGN CONTRADICTION

Two 07-31 convocations measured the seg↔rate exchange on the same vehicle and reported **opposite signs**,
neither citing the other:

| source | claim | number |
|---|---|---|
| gc12:79-81 | *"S-progress = SPEND bytes to BUY d_seg"* | favorability **~2.7× FOR spending**; reclaim pays a **6–7× co-location tax** |
| gc14:81 | *"it is **not an exchange** … bytes and d_seg **drift together (both worse), not traded**"* | ratio 0.021003/0.002700 = **7.8:1**; gate-to-gate correlation **r = +0.212, t = +1.30** |

gc14's corollary is the correct generalization and belongs on Surface E as a law:
> **"An exchange rate is only meaningful between quantities shown to be causally coupled."**

An exchange requires **r < 0**. The measured r is **+0.212** — same-direction drift. So the 7.8:1 (and gc12's
2.7×) are **ratios of two independently-drifting quantities**, not prices. Supporting decomposition (gc14:78,82):
`renderer_bytes = 3,284` and `selector_ledger_bytes = 216` are **constant at every gate** — all byte variation is
in `tokens_bytes_smevr`, and **raw entropy is monotone (`tokens_bytes_zlib` 439,843 → 454,511, +3.3%) while
counted bytes are NOT** (272,883 → 271,508 → 276,224 → 275,872), because the coder ratio drifted
**0.6125 → 0.5993**. **The counted-byte headline welds monotone entropy growth with a compensating coder.**

**This is the one place where §5b.3's identity does NOT rescue the reading.** The score's 1.2731 B/flip is a
*true* exchange rate — it is what the scorer charges. gc12/gc14's numbers are *observed co-movements* in a
training run, which is a different object entirely. **Two things called "exchange rate" today: one is exact
arithmetic, one is an uncontrolled correlation of +0.212.**

### §11.2 Surface A — the boundary step, and a competing hypothesis nothing distinguishes

**The descent is BOUNDARY-LOCALIZED, and the boundary is a measured optimizer artifact.**
- Within window_02's 29 gates the realized d_seg OLS slope is **−1.46e-7/gate, t = −0.09 — statistically ZERO** —
  while **training loss fell 13.4% (0.54275 → 0.46984)**. *139 epochs of genuine loss descent bought no realized
  d_seg.* window_03 reproduces it: a **−1.118e-4 step** at ep805→809 then flat; |step|/|Δwindow| = **1.36**.
- gc15 derived the mechanism from source: **MLX `Adam` defaults `bias_correction=False`**, so a reset makes
  **η(t) = (1−β₁ᵗ)/√(1−β₂ᵗ)** — **η(1) = 3.1623, peak η(12) = 6.5685**, decaying with τ = 1/(1−β₂) = 1000 steps.
  Integral **I = 1,212.6 sign-steps = 16.17 epochs = 11.5% of a 140-ep window, 81.7% inside 13 epochs**, and
  **99.7% delivered by epoch 67 regardless of window length**. It predicts **four** gc14 observations without
  being fitted to any. Independently reproduced by sb2 as **1,212.57** against the real MLX Adam.
- **The zero-reset is a standing-law violation nobody derived:** with `bias_correction=False` and `v = 0`, the
  first step is `lr·(0.1g)/(0.0316|g|) = **3.16·lr·sign(g)`** — a uniform-magnitude, metric-free sign step, i.e.
  the **maximum-entropy step**, forbidden by our own `generic_basis_metric_never_optimal`. A 64-flag census
  returns **zero** matches for `adam|beta|bias|moment|restart|precond|warmup`: **the entire reset operator is
  unexposed, unconfigurable, and undeclared.**

**⭐ And the competing hypothesis is undistinguished (gc15:505, H5): "displacement is not learning."** A 6.57×
step in a well-conditioned basin may simply arrive where the run would have arrived later — **time-travel, not
gain** — in which case r = 0.310 is the ordinary decay of ordinary training sampled at boundaries, and the whole
"restarts are a lever" framing collapses. **Nothing in the record distinguishes H5 from H1.** The discriminator
is one line: run arm A **16 epochs longer with no reset** and see if the step reproduces. H5 was not in the
hypothesis table when its priors were set (H1 0.6 / H2 0.2 / H3 0.1 / H4 0.1).

**Honest bound on all of it (gc15:149, Contrarian gc15:504):** the constant-g idealization makes η an **upper
envelope**; the true multiplier lies in **[1, 6.5685]** and the displacement in **[0, 16.17] epochs**. *"16.17
epochs is an upper bound presented as a quantity."* Marked `PROVISIONAL-PENDING-VERIFICATION` in gc15 itself.

**A.6 The seg-continuation decay r is a 42×-wide sensitivity, from n = 2 across an instrument change.**
Measured **r = 0.310**; closing the 0.36640 S seg debt needs **r ≥ 0.9458**. Sensitivity: r=0.50 → 0.02100 S ·
r=0.70 → 0.04901 · r=0.85 → 0.11902 · r=0.95 → **0.39906** — the answer spans **42×** over the plausible range.
Schmidhuber's dissent is the correct scoping: *"r = 0.310 is n=2 and both points come from **different
apparatus** … A decay ratio computed across an instrument change is not a decay ratio."* And the ep665 point
inside that fit is **positive (+0.00131)** — an apparatus cost embedded in the trajectory used to fit r, equal to
**7.2% of the descent being claimed**.

**A.7 The exact-solve floor is a CURVE, not a point (ph3 §8b, operator-caught).** Custody holds **two** solved
objects, and §8 conflated them:

| object | errors | d_seg | materialized | S receipt |
|---|---:|---:|---:|---:|
| BOX-tolerance solve | 136,839 | 0.00116 | 291.2 MB | 194.4 |
| **EXACT C1 solve** | **17,927** | **≈1.52e-4** | 409.5 MB | 272.7 |

**7.64× error reduction for a 1.41× byte increase.** The amortization gap was quoted as **3.35×** from the weaker
point; against the exact teacher it is **25.6×**. My §10.A.4 used the exact column — confirmed, and the
correction is the *shape*: the solve family is a tradeoff curve sampled at two points, and which point you call
"the floor" changes the headline by 7.6×.

### ⭐ §11.3 Surface B — the realization wall is ~1/k LSB, not 1 LSB

ph3 §5.2 re-derives the level set that killed the steering-atom family:

> *"**The resize divides the quantum.** R averages ~k camera pixels per scorer pixel (**k ≈ 30 area**). A COHERENT
> pattern of ±1-LSB roundings across a support realizes EFFECTIVE sub-LSB moves … **The realization wall is not
> 1 LSB; it is ~1/k LSB for coherent dithered actuators.** Named reopening: the steering-atom family (pi2, killed
> at naive rounding) **re-enters at DITHERED-REALIZATION scope — the atoms died because rounding was treated as
> given, not designed.**"

**The kill measured AMPLITUDE (sub-LSB per pixel) and never measured SUPPORT × COHERENCE.** The proposed
ranking replaces gradient norm with **(LSBs moved × support area × coherence)**. The two-plane warp that *works*
is 6 params moving ~half the frame coherently; the atoms that *died* had the same amplitude and no support.
**Same continuum, two ends, one measured.** ph3 also notes part of the measured **3–10× tangent overshoot** may
be a **rounding artifact** (independent per-param rounding is **Babai rounding — provably suboptimal on
anisotropic lattices**; head cond 24.8, per-dim sensitivity spread ~600×), not intrinsic realization.

**B.7 fp1's flat-paint floor hides a 235× per-class spread.** Aggregate 0.008305 =
`Road 0.00517 · Lane 0.00101 · Movable 0.00198 · MyCar 1.2e-4 · Undrivable 2.2e-5`. **Road/Undrivable = 235×.**
The "DEAD by construction" verdict is a single scalar over that spread. And fp1's **second** wall — the trained
3×3 head at f′ **0.499366**, ~60× above its own receiver floor — has an **orthogonal class signature**
(Road 46% + MyCar 50%, vs the receiver floor's Road 62% + Movable 24%). **Neither is "the" wall.**

### §11.4 Surface C — three more instrument coordinates, one below its own resolution

| finding | numbers |
|---|---|
| **A "FLAT" verdict below the instrument's self-disagreement** (gc13 B6) | `Δ_gate_dseg_vs_parent = +4.24e-7` vs gate-vs-full_confirm disagreement **on identical state** ~**1.1e-6** ⇒ **signal is 2.6× BELOW the instrument's own resolution**. A trigger reading it *"would be controlling below the noise floor."* |
| **term_domination fired FALSE POSITIVE on the scored term** (gc13 B10) | seg = **68–70%** of a **two-term seg-only** loss vs a **40%** threshold **ported unchanged from v9**. Dominance by the scored term is guaranteed **by design** on a lean vehicle. The predicate has **no scored/non-scored distinction** — it welds "term share" with "term role." Law extracted: **alarm predicates and thresholds are per-vehicle calibration objects.** In the same window the *derived-ε* lane key did **not** false-fire — a within-receipt control. |
| **The 5-gate estimator aliases a ~30-gate oscillation** (gc14:29) | Lane slope **+11.66/gate, t = +5.54 (n=5)** · **−0.497/gate, t = −1.50 (n=29)** · **−0.158/gate, t = −0.69 (n=38)**. The 5-point window **begins at Lane's trough (485 @ ep784)**. **The same `n_points = 5` produced gc12's "+8.75 births/gate" premise.** New law: *constants-are-poison applies to the **estimator window**, not only to the threshold.* |
| **UNDRIV_EROSION read convergence-to-GT as erosion** (gc14:77) | GT betti0 **38**; realized **42 → 41**; **\|realized − GT\| 4 → 3 = IMPROVED**; `gt_components_erased` **2 → 2 = UNCHANGED**. The observable lacked a **GT reference term** — it welds "losing true components" with "shedding spurious ones," which have **opposite sign in S**. The ε derivation was sound; the observable was wrong. |
| **MyCar's dual has zero pool** (gc14:426) | `betti0_realized = 36 = GT 36` at **every one of 38 gates, zero variance**; MyCar's whole level is **0.0184 S**. |

### §11.5 Surface E — near-parity, and what that does to "the largest axis"

ph3 §7 states it exactly: at S **0.992972** the three axes are **seg 0.431179 · pose 0.322250 · rate 0.239543** —
**all inside a 1.8× band**. Therefore *"marginal prices have converged, so the next −0.1 lives wherever the JOINT
exchange rate says, not where the last win was. **Axis-reflex (pose-first inertia, seg-first reflex) becomes the
dominant allocation error exactly here.**"* **This retires "which axis is largest" as a decision input**, and
retroactively explains my own §6 error (carrying "pose ≈ 1.24 S" — see §9.6): axis ordering is not just stale,
it is **uninformative at this operating point**.

**E.8 QA24's "MEASURED-DOMINATED" reverses under matched compute (gc15 §9).** The standing verdict (fresh 0.686
vs warm 0.608, *"closed at INSTANCE"*) is confounded three ways, **all moving together**: (1) **compute deficit
+60.7%** (ep399 vs ep641 — *"the binding one, and it was never stated anywhere"*); (2) **byte mismatch 19,146 B =
0.01275 S in fresh's favour**, while the pre-registered falsifier said *"at matched bytes"* and **the comparison
was never matched**; (3) the D16 grid cap. On the matched-epoch projection the sign flips: **fresh ahead ~0.035 S**,
reversing a 0.078 S "domination." A fourth confound is named: the warm arm accumulated **more restart impulses**
(16.17 free epochs each), so part of the residual warm advantage is **the reset bonus, not the warmness**.

**E.9 The census is entirely below its own routing threshold.** Every seg-side pool at gc15 §13 sits under the
0.05 S Contrarian bound: continuation **0.00946** · cadence 2× **≤0.019** · reset-metric **0.011–0.047** ·
post-hoc injection **0.0171**. gc15 **pursues the reset-metric anyway** and says why — standing-law violation,
$0 build, and it **re-prices the campaign's only measured descent** — *"Contrarian's bound is respected and
explicitly overridden with reasons."* **Sound**: an explicit override with named grounds on an axis (information
value) the census does not price is exactly right, and is the opposite of a threshold applied blind.

### §11.6 What the convocations do to the six surfaces

1. **Surface E gains a contradiction and loses a metric.** "Exchange rate" was being used for two different
   objects; only the score's own 1.2731 is one. gc14's coupling test (**r = +0.212**) is the missing precondition,
   and it is cheap.
2. **Surface A's position is partly an optimizer artifact of unknown size.** The boundary step is
   MEASURED-mechanistic (η, I = 1,212.6) but its *interpretation* (gain vs time-travel) is undecided by one
   16-epoch control run that has never been made.
3. **Surface B's level set moved by ~30×** the moment coherence entered the ranking — and the family it killed
   was killed on amplitude alone.
4. **Surface C now has a case where the signal was 2.6× below the instrument's self-disagreement**, and two
   predicates whose *observables* (not thresholds) were wrong.
5. **Surface D's reach is larger than sb2 measured:** gc15 §4 shows the "the stack has never run from ep0"
   argument is carried by **five of six forces being unbuilt stubs** — *"the binding blocker is wiring, not
   birth"* — and gc14 proves λ_Lane was **0.0 at all 38 gates**, so "guarded" and "unguarded" windows are the
   **same state**.
6. **Sound, called sound:** gc14's self-corrections (S3/S4/S7/S8 all overturning its own or MAIN's prior
   readings), gc14's own V17 (*"Pointer moved — **NO**"*), gc15's recorded sign error (*"A memo built on the
   wrong sign would have recommended the exact opposite intervention"*), ph3's operator-caught two-solve
   conflation, gc13's B10 FALSE-POSITIVE adjudication, and gc12's plain §7 statement (*"the wall branch does NOT
   reach 0.172141 — stated plainly"*). **Every one of these is an arm overturning its own headline. That is the
   apparatus working, and it is the single most encouraging pattern in the window.**

---

## §9 Round-1 adversarial self-review of this document

1. **"Coder quality moves the edge past the three low floors" — is it established or plausible?** *Plausible,
   bounded.* MEASURED: the edge spans 2.991× across two coder families. DERIVED: the three floors sit 1.15–1.76×
   below the coherent edge. NOT measured: that a specific coder achieves the needed shift *on those classes'
   supports*. Stated as "decided by coder coherence," never as "corrections are in."
2. **The QA03 decomposition uses interpolation on a synthetic support.** Named in §2.2. The load-bearing part is
   weaker and safe: QA03's 1.45 exceeds even the *uniform upper bound* 1.0007 by 1.449×, which needs no
   interpolation. The 51.6% figure is the interpolated version and is labeled DERIVED.
3. **Am I replacing binaries with new binaries?** Checked each section. §7's falsifier analysis risks reading as
   "the falsifier was worthless" — it is not: it correctly re-confirmed a scope law per class, and fl1 published
   the un-rescaled column. Wording adjusted to "no power at this operating point," which is a coordinate claim.
4. **Is Surface F real or an excuse?** Real: the cap is a chosen constant with measured distances (1.083×,
   1.1003×), and MAIN labeled the mechanism correctly. Recorded as sound.
5. **Did I audit MAIN as hard as the arms?** MAIN holds 6 of the placements (ALARM #3, window_04, cn3 withdrawal,
   byte-ledger, both review rounds, instrument count 4→3→4). Three are called sound; the instrument-count churn is
   re-framed as a census coordinate rather than an error.
6. **Did I check my own carried constants?** No, at first — I wrote §6 with "pose ≈ 1.24 S" straight from the
   standing MEMORY row, then found the ja1 measured base at **pose 0.322250 / seg 0.431179**, which **inverts the
   axis ordering**. Corrected in §5b. This is the same staleness confound the document is about, committed by the
   document, and caught only because Surface E was folded in. Recorded rather than silently fixed.
7. **Round 2, after the full harvest returned:** the convocation fold (§11) **contradicted my own framing once**
   — I built the whole document around exchange rates, and gc14 measured that the campaign's most-cited
   "exchange rate" is a **correlation of +0.212**, i.e. not an exchange at all. §11.1 records it rather than
   absorbing it. The identity in §5b.3 survives (the scorer's 1.2731 is exact arithmetic), but the *word* was
   doing work across two incompatible objects and I did not notice until the harvest.
8. **Coverage is now complete for the window:** all 43 artifacts + 50 commits + MAIN's `current_focus` entries
   were read (by me or by three read-only harvesters). Every negative reached is placed. §11.6 states what the
   last fold changed.
9. **Completeness gaps I can still name:** (a) the ja1 atlas's remaining `surfaces` entries beyond the six read
   here (`segnet_head_rank4_hyperplanes`, `segnet_stride2_skip_lane_bottleneck`, `posenet_head_rank6_p0_dominance`,
   `pose_per_pair_occupancy_v4c`, `resize_null_ker_A_shared`, `photometric_response_rungB`) are structural
   transfer surfaces, not negative-verdict coordinates, so they are cited but not re-tabulated; (b) the three
   sister-arm harvests (07-31 convocations, audits, and remaining arms) were dispatched read-only and any negative
   they surface that is not already on one of the six surfaces should be appended here rather than filed
   separately — the surfaces, not the artifact list, are the unit.
