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
= the **pre-arc** W_joint base 7.05192e-2; `:3197-3204` emits the second from the **live** burn endpoint 3.89011e-3.
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
- **Negatives enumerated:** the full artifact set was scanned for negative-shaped verdicts; 34 of 43 artifacts
  carry at least one. Every negative reached is placed on one of the six surfaces above; none is discarded.
- **Placed with full arithmetic here:** the pp1 band (14 measured points + 4 edges), fl1 (5 floors + 5 piercing
  ratios + the falsifier level set), QA03/QA04, the lane-guard slack trace (58 gates), the instrument table
  (12 rows), sb2 (5 grades / 197 rows), the governance ratios.
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
7. **Completeness gaps I can still name:** (a) the ja1 atlas's remaining `surfaces` entries beyond the six read
   here (`segnet_head_rank4_hyperplanes`, `segnet_stride2_skip_lane_bottleneck`, `posenet_head_rank6_p0_dominance`,
   `pose_per_pair_occupancy_v4c`, `resize_null_ker_A_shared`, `photometric_response_rungB`) are structural
   transfer surfaces, not negative-verdict coordinates, so they are cited but not re-tabulated; (b) the three
   sister-arm harvests (07-31 convocations, audits, and remaining arms) were dispatched read-only and any negative
   they surface that is not already on one of the six surfaces should be appended here rather than filed
   separately — the surfaces, not the artifact list, are the unit.
