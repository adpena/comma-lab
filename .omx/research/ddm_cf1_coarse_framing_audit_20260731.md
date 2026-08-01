# ddm_cf1 — the COARSE-FRAMING audit of the 2026-07-24 → 07-31 corpus

**Date:** 2026-07-31 · **Arm:** ddm_cf1 · **Axis:** `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false` · **Pointer:** 0.1910828242 `[contest-CPU]` **UNMOVED** —
this document audits instruments, not score. No number here is a score claim. `upstream/` untouched.

**Why this exists.** Operator, verbatim 2026-07-31: *"Much of your framing has been too coarse."*
+ *"do a subagent to do it against the last week's stuff looking for framing that was too coarse."*

---

## 0. THE BUG CLASS, and why it is not "a wrong number"

A **COARSE INSTRUMENT** answers a strictly weaker question than the one the verdict rests on,
while reading as if it answered the real one. The number is *arithmetically right, of a cruder
object*. No assertion fires, no gate trips, no correctness check can see it — which is why it
survives review and why the operator, not the apparatus, caught it.

Sister precedent already in custody: **#580 replaced a 22.6969% axis-aligned dimension count with
80.6742% real-linear nullity — the coarse instrument under-counted by 3.55×**
(`.omx/research/collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md:327`).

**Output-form rule applied to this document (per
`boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment_20260731`):** every row below is a
**placed point on an instrument ladder**, not a pass/fail. The ladder used throughout is the one
the corpus itself named today:

> **rung 1 dimension/count → rung 2 L2 spectrum/energy → rung 3 margin-or-Fisher-weighted measure**
> (`.omx/research/ddm_control_surface_exact_quartering_20260731.md:124`)

A row's "coarseness" is *which rung it stands on* versus *which rung its consumer needs*. Rung 1
is not wrong; it is a lower rung. Producing a ranked table of binaries here would commit the very
error being audited.

---

## 1. THE TAXONOMY — 8 seeded, 5 added (13 + 2 variants)

| # | class | one-line signature |
|---|---|---|
| 1 | DIMENSION COUNT where a SPECTRUM/MEASURE was needed | a nullity/rank/DOF-count priced as if it were sensitivity mass |
| 2 | BINARY PREDICATE ON A CONTINUUM | reaches/doesn't · DEAD/ALIVE · zero-flip/nonzero · a regime tag over a measured curve |
| 3 | L2 ENERGY where SCORER-SENSITIVITY was needed | our own g3 law: *flip/margin-weighted, never L2 energy* |
| 4 | AVERAGE where MARGINAL was needed | a mean cost crossing a threshold, consumed by a waterfill that spends at the margin |
| 5 | SCALAR where a FIELD was needed | "a margin floor", "the noise floor", "the resize" |
| 6 | GLOBAL where PER-ELEMENT was needed | one format/quantizer for all elements |
| 7 | SCOPE-DROPPING COMPRESSION | the qualifier lost in transit — *which scorer · which vehicle · which measure · which rung* |
| 8 | HEADLINE COMPOSITE where DECOMPOSITION was needed | a bare composite quoted without its terms; **or a computed bracket reported as one edge** |
| **9** | **NO-CONTRAST INSTRUMENT (degenerate-baseline collapse)** | the number equals what a *trivial predictor* scores ⇒ carries zero bits about the system |
| **10** | **ONE WORD, THREE MEASURES (homonym drift)** | one term (`invisible`, `free`, `null`) silently naming a dimension, an energy, AND a counted-byte quantity |
| **11** | **MECHANISM-PRESENCE STANDING IN FOR PERFORMANCE** | build asymmetry between arms read as a physics verdict; `DEFERRED` on an *unbuildable* branch read as a decision |
| **12** | **REPRODUCIBILITY RESIDUAL AS THE NOISE FLOOR** | a *within*-run variance spent as the denominator of a *between*-condition comparison |
| **13** | **FALSE DECOMPOSITION** | per-element attribution of an irreducibly JOINT quantity — the *inverse* of class 6 |

Variants worth naming: **8a SIGN-WELDING** (a composite whose terms have opposite sign in S ⇒ no
directional information) · **1a DEGENERATE MULTIPLICITY** (N "independent" corroborations that are
one column times a constant).

**Provenance of the classes.** 1–8 seeded by the operator's brief. **9 was named by the corpus**
(ddm_ba30). **10 is this sweep's** (§3). **11–13 and the two variants came from the three sub-sweeps**
(§9c) — 11 and 13 from `ba31`/`sb2`/`gc13`'s own language, 12 from a cross-file pattern none of the
memos had named. **So 5 of the 13 classes were already in the corpus's vocabulary before this audit
ran** — which is itself the calibration result: the campaign can name these; it does not yet gate
on them.

**Class 9 was found and named by the corpus itself, not by me** — credit
`measured_number_equal_to_a_degenerate_baseline_is_not_a_measurement_20260731` (ddm_ba30,
MAIN-verified float64): pj1's renderer "capacity floor" `f = 0.504824` **is** the
constant-Undrivable predictor `0.50482448154026` (Undrivable = 49.52% of 117,964,800 px), matching
per-class at abs diff `0.00e+00` on all five classes; fp1's trained head `f' = 0.499366` sits at
98.92% of the same corner after 50 converged epochs. It is a distinct class because it is
*invisible to correctness checking by construction*. **Cure (theirs): a probe must report its
degenerate-baseline comparison or it has not shown it measured its system.**

**Class 10 is the finding of this sweep** and is developed in §3.

---

## 2. WHAT WAS ALREADY CAUGHT — the recall leg (do not re-credit these to me)

The corpus self-corrects heavily. Three of the four seeded worked examples were **already closed
by the corpus on 2026-07-31, before this arm ran**. Recording them so no future reader mistakes
this audit for their discovery:

| seeded example | already caught by | where |
|---|---|---|
| DOF quartering = "exactly 25.0% each" | **the same memo, §3b** | `ddm_control_surface_exact_quartering_20260731.md:101-126` — *"§3 counts DIMENSIONS. That is the crudest measure of a subspace — a 25%-dimensional subspace can carry 1% or 99% of the sensitivity mass"*; supplies cond **2.6045**, `tr(MᵀM)/12 = 0.166074 = 0.3321×` naive, the three-tier luma/chroma-mean/chroma-zero-mean ladder, **and** the self-attack that `tr(MᵀM)` is *itself* rung 2 |
| 3-valued regime tag over a 9-point curve | **`ddm_surface_correction_economics_20260731.md`** — its title is literally *"(not a regime tag)"* | `:1-11`, `:26-38` |
| AVERAGE-vs-MARGINAL in the band lemma | **same memo, §3** | `:60-82` — *"The law's sentence 'sub-ρ_c correction machinery is permanently pointless' is a statement about the average, applied as if it governed the margin. 'Permanently' is the binary."* Every rung from τ=0.02 up has marginal cost **below** W |
| axis weights "pose ~1.24 S" carried as a routing weight | **`ddm_gc16_…_20260731.md:59-77`** | re-derived from primary artifacts; ranked instead by **gap to a demonstrated floor** (seg 0.4015 > pose 0.2776 > rate 0.1126, summing exactly to the PR130 row 0.172141 ✓) |

**Consequence for this audit:** the live coarse framing is *not* in the memos that named the
problem. It is in (a) the **compression surfaces** those memos feed — MEMORY files, index lines,
priced-spec tables — and (b) **one word** used across three different measures. Both are below.

---

## 3. THE HEADLINE FINDING — class 10, `scorer-invisible` names three different measures

**verdict_scope: FAMILY** (six memos, three authors, spanning 07-17 → 07-31). This is the one row
I would act on.

Three *correct, separately measured* quantities are carried under one word:

| quantity | value | what it actually measures | primary artifact |
|---|---:|---|---|
| **ker(A) DIMENSION** | **80.6742315223%** | `1 − (384·512)/(874·1164)`; definitional nullity of the resize | `null_subspace_rate_measure_20260717.md:38,98` |
| **ker(A) ENERGY of a render** | **52.42%** `[52.35, 52.58]` (mean-removed 52.88%) | energy fraction of byte-closed decoded uint8 camera frames lying in ker(A) | `null_subspace_rate_measure_20260717.md:101-104` |
| **ker(A) ENERGY of the frame_0 crush error** | **56.5–57.3%** | a *different* signal's energy in the same subspace | `ddm_da1_telemetry_decomposition_20260728.md:151` |

They differ because natural renders concentrate energy at low frequency: **80.67% of the
dimensions die but only ~52% of the energy does.** The 07-17 memo is exemplary — it states both,
side by side, distinctly labelled. **That memo is the model; the drift is downstream of it.**

And there is a **fourth** quantity the word also gets attached to, whose measured value is **zero**:

> `codex_findings_ddm_m4_rate_floor_einstein_avenue_20260723_codex.md:87` — **"`80.67% * archive_bytes`
> is forbidden arithmetic"**
> `codex_findings_ddm_m6_close_22645_byte_gap_20260723_codex.md:28` — ker(A) payload hiding,
> admitted bytes **0**: *"MEASURED ZERO. The 80.6742315223% nullity is geometric freedom; no
> parser-consumed counted payload was removed."*

### 3a. Where the drift is live

| row | claim (verbatim) | file:line | rung used | rung the consumer needs |
|---|---|---|---|---|
| **A** | *"store range(A) only, synthesize ker(A) (80.67% invisible) free at decode (#580)"* — in the **`bit-depth × invariance lever`** column of a **priced spec** | `ddm_ar1_archetype_codec_priced_spec_20260728.md:104` (repeated `:158`, `:202`) | **1** (dimension) | **counted bytes** — where m6 MEASURED **0 B**, 5 days earlier |
| **B** | *"**ker(A) = 80.67% scorer-invisible; 56.5–57.3% of frame_0 crush error ker(A)-invisible**"* — a dimension and an energy under one word, one line, no distinction | `council_gc5_schmidhuber_micro_macro_bridge_20260728.md:157` | 1 **and** 2, unlabelled | labelled separation |
| **C** | *"ker(A) for the 874→384 bilinear resize is **~52%** (#580: range(A) = 19.33%); yuv6 kills a further exact 50%. ⇒ **frame_0's scored DOF ≤ 0.4833 × 0.50 = 24.2%; ≥75.8% of frame_0's payload DOF are provably unscored**"* | `ddm_gc16_upstream_score_lowering_convocation_20260731.md:306-308` | **2** (energy) substituted into a **DOF** product | **1**, consistently |

**Row C is internally contradictory in its own parenthetical**: it asserts ker(A) ≈ 52% while
citing range(A) = 19.33%, whose complement is 80.67%. Re-derived (exact, this session):

```
range(A) dim frac = (384·512)/(874·1164) = 0.193258   ⇒ ker(A) = 0.8067423  (#580 ✓)
gc16 as written  : 0.4833   × 0.50 = 0.24165  ⇒ ≥75.8% unscored
DOF-consistent   : 0.193258 × 0.50 = 0.096629 ⇒ ≥90.3% unscored
```

**The scored-DOF figure is overstated 2.50×.** The direction is *conservative* (the true DOF claim
is stronger), and the paragraph's conclusion — *"the live vehicle already exploits this: v4d stores
no frame_0 at all … Credit, not a proposal"* — is unchanged. So **row C is PRECISION-ONLY.**
**Row A is the one that bites**, because it re-enters a *priced spec* as a rate lever after the
counted-byte answer was measured at 0 B.

### 3b. Already-fine control — the corpus contains the correct form

- `ddm_ax1_all_axes_derivation_20260730.md:70` — *"ker(A) 80.67% resize nullity; **52% scorer-invisible
  render energy**"* — **both, labelled, in one line. This is the model sentence.**
- `ddm_ee1_einstein_fresh_eyes_capstone_20260728.md:56` — *"~80.7% of camera-resolution pixel
  **dimensions** are scorer-invisible"* — the unit is stated.
- `ddm_iv3_codec_artist_synergy_bridges_20260728.md:40` — *"ker(A) ≈ **52%** scorer-invisible render
  **energy**"* — the unit is stated.
- `council_gc5_…:157` correctly types row B6 as **GAPPED** on the *uint8/linearization* axis
  (#532, Δ=62.74) — it caught a different gap in the same claim, just not this one.

### 3c. The cheap resolution + its consumer

**$0.** One three-column table appended to
`null_subspace_rate_measure_20260717.md` (append-only, per Catalog #110/#113) fixing the
vocabulary: `ker_dim_frac = 0.8067423` · `ker_render_energy_frac = 0.5242` ·
`ker_counted_bytes = 0 B (m6, MEASURED)`. **Consumer:** the `ddm_ar1` priced spec's
`bit-depth × invariance lever` column, and any future rate claim citing #580. **Falsifier for the
naming rule:** none needed — it is a decomposition, not a hypothesis.

---

## 4. THE ROWS — placed points

Legend: **VC** = the finer read plausibly flips or rescopes a still-load-bearing decision ·
**PO** = precision-only · **SC** = self-caught in the same memo or an explicitly cited sister ·
**AF** = already-fine (control group).

| # | claim + file:line | class | finer instrument available AT THE TIME | placed | $0 resolver → consumer |
|---|---|---|---|---|---|
| **1** | `ddm_ar1_…20260728.md:104` *"synthesize ker(A) (80.67% invisible) free at decode"* in a priced-spec rate column | 1, 10 | m4/m6 (07-23): counted-byte answer **0 B**; "80.67%·archive_bytes is forbidden arithmetic" | **VC** | §3c naming table → ar1 spec column |
| **2** | `ddm_gc16_…:306-308` `0.4833 × 0.50 = 24.2%` scored **DOF** | 1, 10 | `(384·512)/(874·1164) = 0.193258` — cited in the same sentence | **PO** (2.50× overstated, conservative direction) | recompute inline → gc16 §5a |
| **3** | `council_gc5_…:157` dimension + energy under one word, one line | 10 | ax1:70's labelled form, landed 2 days later | **PO** | §3c |
| **4** | **MEMORY file drops its own memo's §3b.** `control_surface_exact_dof_quartering_q3_seg_only_pose_null_20260731.md` carries *"Exactly 25.0% each"* and the 4-row DOF table; it does **not** carry cond 2.6045, `tr(MᵀM)/12 = 0.3321×`, the three-tier ladder, or the rung-1→2→3 ladder that the source memo made its §3b | **7** | the source memo, same session, same author: `ddm_control_surface_…:101-126` | **VC** — see §5 | append §3b's ladder to the memory file → every future recall of Q3 |
| **5** | `pose_is_the_largest_axis_on_the_own_vehicle_1_24_S_20260731.md` frontmatter `description:` — *"the POSE payload is worth ~1.24 S … **Rank levers by axis weight**"* | 7, 8 | `gc16:59-77` — 1.24 is a *realized-improvement delta across two vehicles*; the **prospective** routing weight is gap-to-floor: seg 0.4015 > pose 0.2776 > rate 0.1126 | **VC** — and it **already misfired once**: MEMORY.md records *"MAIN mis-ranked work with it despite this row's own guard clause"* | append-only `superseded_by`/correction line; MEMORY.md is already fixed, the file and its `description:` are not |
| **6** | `ddm_wr1_reverse_waterfill_20260729.md:54` defines **`S_ref_ceiling`**; the descent table `:58-72` tabulates only `S_ref (flipfree)` and `S_if_solved` | **8** | the receipt already stores it per row (`S_vs_ref_flipceiling`, `dseg_pred_ceiling`) | **VC** — **RESOLVED BELOW, §6** | done at $0 this session → #766 / Knee-selection |
| **7** | `ddm_wr1_…:37,78` *"486 zero-flip cells"* → *"Drops ALL 486 zero-flip cells … at PREDICTED ~zero d_seg cost"* — cell selection by a **zero/nonzero flip count**, on a population whose sensitivity is a continuous margin field | **2**, 5 | the memo's *own* κ = 0.0753 logits/quantum flip-**creation** model (`:44`), and `margin_budget_field.py` / the zb1 n600 exact flip-distance field (`d = \|m\|/‖Δw‖`, q05 median 0.4302) | **PO** — §6 shows the ceiling vindicates Knee A exactly | rank the 486 by flip-**distance** headroom, not by flip **count** → Knee-A safety margin |
| 8 | `ddm_wr1_…:32` *"The sensitivity map (analysis-only; **Fisher/margin currency, never L2**)"* | — | — | **AF** — the g3 law honored explicitly at the section header | — |
| 9 | `ddm_gc16_…:105-119` degenerate-baseline control: 200 iid draws at matched per-matrix Frobenius norm; trained cond **547.0** vs control max **3.8535** (142×); control argmax dim uniform vs trained dim 0 | — | — | **AF** — this is class 9's cure, executed | — |
| 10 | `ddm_gc16_…:145-160` the §2 bound is **stated, then attacked by its own author** and found conditional: `pfs1` receipt MEASURES dims-1–2 residual std **3.3× / 8.0×** their GT std, **20× the assumed ceiling** | — | — | **AF** — §6 of the operating manual, executed | — |
| 11 | `ddm_gc16_…:326-332` stem Frobenius energy share (an L2 read) explicitly demoted: *"consistent with but **weaker than** pi2's end-to-end 99.3% luma … **pi2 is the authority**"* | — | — | **AF** — a rung-2 instrument that names the rung-3 authority above it | — |
| 12 | `ddm_control_surface_…:101-126` §3b in full | 1 | — | **SC** — the exemplar | — |
| 13 | `ddm_surface_correction_economics_…:26-38, 60-82` | 2, 4 | — | **SC** — the exemplar | — |
| 14 | `ddm_ba31_negative_surfaces_20260731.md:619` *"measured at a single point of a stroke-width continuum whose per-class signature says that continuum is binding"* | 2 | — | **SC** | — |
| 15 | pj1/fp1 "capacity floor" ≡ constant-Undrivable predictor | **9** | — | **SC** — ba30, MAIN-verified float64 | — |

---

## 5. THE SINGLE HIGHEST VERDICT-CHANGING ROW — #4, and why

**The memory file for the DOF quartering re-commits the coarse read that its own source memo
retired, in the same session, and it is the surface future sessions actually read.**

The source memo `ddm_control_surface_exact_quartering_20260731.md` does the right thing twice:
§3 gives the dimension count, §3b demolishes it as rung 1 and names rung 3 (margin/Fisher-weighted)
as **THE object, UNMEASURED**. The MEMORY file carries §3's table and *"Exactly 25.0% each"* and
**not** §3b. MEMORY.md's index line likewise compresses to `Q3 frame_1-yuv6-null=SEG-ONLY d_pose
EXACTLY 0`.

**This is the exact disease the source memo diagnoses about a predecessor.** From `:176-177`:

> *"MEMORY.md compresses it to `\"chroma <2px INVISIBLE\"`, which reads as invisible to both. The
> signal was mis-homed, not missing."*

The memo names the failure mode at the compression surface — and then its own compression surface
commits it.

**Why it changes a verdict, concretely.** The memo's §8 names **THE ONE MEASUREMENT**: *"Perturb
frame_1 within Q3 ∩ range(R) ∩ uint8-realizable, at a margin floor … and measure Δd_seg."* A reader
holding only the MEMORY file knows Q3 is 294,912 DOF, uniform, 25%. That reader will perturb Q3
**isotropically** — which is precisely the generic control the campaign's own standing law forbids
(`generic_basis_metric_never_optimal_cosine_fourier_euclid_20260729`: *"GENERIC TRIPLE = CONTROL
NEVER OPTIMAL"*). §3b says the surviving directions are **graded three-tier** (luma gain ~0.67 ·
chroma-mean 0.27–0.32 · chroma-zero-mean exactly 0) and that the correct allocation instrument is
rung 3, unmeasured. **Isotropic Q3 is a designed null result:** if Q3's Δd_seg comes back ~0, the
memo's own §8 reads that as *"Q3 is shared blind space,"* closing "chroma as a free seg actuator" —
a **FAMILY-scope closure drawn from a rung-1 experiment design**. That is the costly outcome.

**The fix, $0, same session:** append §3b's spectrum + the rung-1→2→3 ladder to the memory file and
to the MEMORY.md line, and re-state §8's measurement as *"perturb Q3 along its rung-3-ranked
directions, with the isotropic draw as the CONTROL arm, not as the treatment."*
**Consumer:** the Q3 scorer slot. **Falsifier for this audit row:** if the rung-3 ranking is
measured and turns out flat across Q3's directions, the isotropic design was adequate and this row
demotes to PRECISION-ONLY.

---

## 6. ROW #6 RESOLVED AT $0 — the wr1 bracket, recovered from custody

`ddm_wr1_reverse_waterfill_20260729.md:54` defines three S readings; its table prints two.
The third (`S_vs_ref_flipceiling`) **was computed and stored** —
`/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_descent_receipt.json`, per row. Recovered here
(no new compute, no forward pass, `[macOS-CPU advisory]`, `score_claim=false`):

| k cells | archive B | dropped flip mass | `S_ref` flipfree | **`S_ref` CEILING** | bracket width |
|---:|---:|---:|---:|---:|---:|
| 0 (ref) | 569,996 | 0 | 2.2566 | 2.2566 | — |
| 100 | 482,742 | 0 | 2.1985 | 2.1985 | **0.0000** |
| 200 | 409,534 | 0 | 2.1498 | 2.1498 | **0.0000** |
| 300 | 346,671 | 0 | 2.1079 | 2.1079 | **0.0000** |
| 400 | 297,368 | 0 | 2.0751 | 2.0751 | **0.0000** |
| **486 (Knee A)** | **274,333** | **0** | **2.0598** | **2.0598** | **0.0000** |
| 540 | 227,327 | 2,313 | 2.0285 | 2.0304 | 0.0020 |
| **600 (Knee B)** | **174,578** | **14,659** | **1.9933** | **2.0058** | **0.0124** |
| 660 | 118,245 | 64,823 | 1.9558 | 2.0108 | 0.0550 |
| 730 | 51,128 | 244,705 | 1.9111 | **2.1186** | 0.2074 |
| 768 (all) | 14,303 | 458,738 | 1.8866 | **2.2755** | **0.3889** |

**Two things the omitted column says that the printed one cannot.**

1. **It VINDICATES Knee A, exactly.** For every `k ≤ 486` the bracket width is **0.0000** — not
   approximately, structurally: the 486 cells carry zero current flip mass, so the worst case *is*
   the best case. The memo's *"FREE safe floor"* language is **stronger** than it argued, and this
   is the good news the dropped column was carrying. (Residual risk is flip *creation*, row #7 —
   a different axis, priced by κ, not by this bracket.)
2. **It REVERSES the tail.** The printed column descends monotonically 2.2566 → 1.8866 across all
   11 rows. Under the ceiling the curve **bottoms near k ≈ 540–600 (2.0304 / 2.0058) and then rises
   back to 2.2755 at k = 768 — above the reference 2.2566.** The full-drop row is net-negative in
   the worst case. A consumer reading only the printed table sees "more dropping is always better";
   the bracket says the free-ness is spent by k ≈ 600.

**Consumer:** task #766 (wr1 reverse-waterfill) knee selection, and `ddm_ck1_composed_kneeA`.
**Class:** 8 — a *computed* bracket reported as one edge. This is a distinct and more insidious
shape than an uncomputed bound: the rigor was done and lost at the presentation surface.
**verdict_scope: INSTANCE** (this table). I did not sweep other memos for defined-but-untabulated
columns; see §8.

---

## 7. WHAT THE PATTERN SAYS (across rows, not within them)

- **The coarseness is not in the analysis — it is in the TRANSIT.** Rows 1–6 are all
  *compression-surface* failures: memo → MEMORY file (#4), memo → frontmatter `description:` (#5),
  receipt → markdown table (#6), measured guard → later priced spec (#1). The corpus's *analysis*
  rung is high (rows 8–15). Its *restatement* rung drops one level per hop. **The apparatus has no
  gate on restatement fidelity**, which is why the operator is the detector.
- **The corpus is already better at this than its reputation.** Of the four seeded worked examples,
  **three were closed by the corpus before this arm ran** (§2). An audit that reported only hits
  would have mis-stated the base rate.
- **Rung 3 is systematically absent.** `ddm_control_surface_…:124` names it explicitly: *"margin/
  Fisher-weighted measure (**THE object, UNMEASURED**)"*. The `ms3/ms4` metric-custody bundle is
  named as the existing producer. Every row in §3 and §5 bottoms out at the same missing rung.
  **This is one build, not six fixes.**
- **Direction of error is not random: 5 of 6 live rows are CONSERVATIVE** (row 2 understates our
  freedom 2.5×; row 6 vindicates Knee A; row 7's zero-flip rule is the safe side). Only rows 1 and
  5 err toward optimism, and both are *rate/routing* claims. Per the operating manual §3
  (`probability × blast-radius × silence`), the optimistic ones are where depth belongs.

---

## 8. WHAT THIS SWEEP DID **NOT** COVER — honest coverage statement

Stated as scope, never as a negative-existence claim (per
`negative_existence_claims_are_the_days_dominant_error_class_20260731`): everything below is
**"did not search"**, not **"there is none."**

1. **Denominator.** The window holds **439** `.omx/research/*.md` files dated 2026-07-24 → 07-31.
   *My own leg:* **3 read in full** + 3 MEMORY files, **~68% of gc16** (651 lines), and targeted
   sections of ar1, gc5, wr1, ax1, ee1, iv3, da1, ba31, m4, m6, null_subspace — **~11 files**,
   **~85 claims at depth**, plus six signature greps across all 439. *Sub-sweeps:* **29 further
   files read in full**, **~911 claims**. **Union ≈ 40 of 439 files ≈ 9.1% of the window at depth.**
   **The remaining ~91% is unswept** — including the entire `codex_findings_*` and
   `codex_session_summary_*` population (~60 files in-window), which no leg opened.
2. **Second-hand rows.** §9b's six rows are **not verified by me**. I re-derived five of the
   sweeps' rows (§9a) and **two changed grade** — a 40% grade-revision rate on a 5-row sample. By
   that base rate, roughly two of §9b's six are likely mis-graded, and I do not know which.
3. **Classes 5 and 6 are under-sampled.** I grepped for them and read no memo *specifically* for
   scalar-where-field or global-where-per-element. The one instance I can name (#7) surfaced
   incidentally. `ddm_control_surface_…:46-61` treats precision as explicitly per-element, which is
   the AF form — but I did not check whether its *consumers* preserved that.
4. **Class 8 was probed at exactly one surface** (wr1's table). I did **not** sweep the corpus for
   other *defined-but-untabulated* columns or *stored-but-unreported* receipt fields, which §6
   shows is a productive and cheap search. **This is the single highest-yield unrun extension of
   this audit** and it is a mechanical grep (`receipt keys ∖ markdown table columns`).
5. **`.omx/state/` and the canonical-equations registry are unswept.** A coarse instrument encoded
   as a *registered law* outranks one in a memo, and I did not look. `ddm_pp1_correction_stream_
   position_band_v1` is the known example — caught by the corpus, not by me.
6. **Code was not audited.** Every row is documentation-surface. Whether the *producers*
   (`lever_registry`, the costate organ, `margin_budget_field.py`) compute rung-1 quantities where
   their consumers need rung 3 is unexamined. `ddm_cn3_…:136` reports one such case
   (`completeness()` ASTs one file of ~180, so `stale == []` is *vacuous*) — a class-9 shape in
   code, found by cn3.
7. **No row here is byte-closed.** §6's table is recovered from an existing receipt; §3's
   arithmetic is exact but definitional. **Nothing in this document is a score claim and the
   pointer did not move.**

---

## 9. SUB-SWEEP FOLD-IN — 29 further files, 3 parallel read-only sweeps

Groups: **(1)** convocations gc9–gc15 · **(2)** probes pj1/fp1/dw1/ps1/nv1/r1c/xp1/zb1/pa1b/gr1/
v4b/v4d/bc1 · **(3)** audits cn3/gd1×2/us1/sb2/lg1/fl1/ba31/b4s.

**Combined denominator: ~911 distinct load-bearing claims** (293 + 148 + ~470). Live rows **~34**;
self-caught **~40**; already-fine control rows **~45**. **Live-hit rate ≈ 3.7%.** The corpus's
*analysis* rung is high; §7's transit thesis is reproduced independently by all three sweeps.

### 9a. Rows I RE-DERIVED myself before promoting (per operating manual §4)

Sweep rows are second-hand. I re-derived five; all five quoted accurately. Two changed grade.

| row | re-derivation | grade after re-derivation |
|---|---|---|
| **`ddm_v4b_composed_gate_20260730.md:147-151` → `:158-159`** — *"σ = [30.55, 7.47, 3.75, 0.032, 0.015, 0.013], energy [93.0 %, 5.6 %, 1.4 %, 1e-6, …] … Rotation dims (3–5) carry ~1e-6 energy — **near-INERT**"* → used to design *"**v4c rate lever:** store the tail two-plane poses as (single-plane pose + ONE scalar dz correction) instead of full 6-DOF"* | **CONFIRMED verbatim.** This is **class 3** against our own named law, on **raw-unit** SVD energy — and `gc16:236-260` independently unified exactly this: in raw contest coordinates dim0's scale is ~10³× the others, so the correct conclusion is *asymmetric storage **precision***, **not** that the small dims are inert **actuators**. `ddm_v4d_adaptive_hybrid_20260731.md:61` then measures the **shear/yaw-derived per-pair beta as the LARGEST single rung (−0.013485 S)** — larger than the dim0 offset (−0.009196, `:62`). **"Near-inert by energy" was not near-inert by S.** | **VERDICT-CHANGING** (confirmed; the strongest class-3 row in the corpus) |
| **`ddm_pa1b_pool_a_harness_20260730.md:125-128`** — the whole Pool-A race admit/exit rule keyed on *"if ANY matched-bytes arm has **c < 0.08815** … ⇒ the hull MOVED … each such lever EXITS the burn-3 stack"*, while `ddm_nv1_hull_move_20260730.md:83-87` proves **the same day**: *"`product_c` … **This is a MULTIPLICATIVE ARTIFACT** … on S the snap **LOSES at every depth**"* | **QUOTES CONFIRMED — but the code was already fixed.** `git log 083726730b`: *"pa1r: extend Pool-A hull analyzer to **ADDITIVE-S verdict** (nv1 reframe) — c demoted to telemetry … 8 new tests incl the nv1 c-artifact"*. (I did **not** locate `iso_c` / `HullCurvatureAnalyzer` / `pool_a_race_programs` under `src`/`tools`/`experiments` in a 60 s grep — *did not find in scope*, not *does not exist*.) | **DOWNGRADED VERDICT-CHANGING → PRECISION-ONLY.** The residue is a **doc↔code divergence**: the memo is the pre-registration surface a reader consults and still carries the retracted currency. Cure is already named by `ddm_cn3_…:573-575` (a `code_state` column) |
| **`ddm_pa1b_…:17-18`** STORES-CONSULTED — *"dw1 re-price (distill term DROPPED; **plain-continuation dividend RETIRED — NOT carried in the stack arithmetic**)"* vs `ddm_dw1_qa75_distill_window_20260730.md:158-161` — *"What dw1 **ADDS** to the stack instead (MEASURED): the **plain-continuation dividend** — E2 is not converged; 40 ep of continued training bought Δd_seg −1.6e-4 (−0.016 S·seg) at zero design cost"* | **CONFIRMED verbatim, both sides.** Not a coarsening — a **flat inversion in transit**. dw1 is vindicated downstream: `ddm_xp1_exact_p_20260731.md:29` measures 140 further epochs buying Δd_seg −0.000677 | **VERDICT-CHANGING** (confirmed). Purest **class 7** instance found |
| **`ddm_ps1_pose_stage_20260730.md:88-89`** — *"on a photometric-walled base the residual is not geometric (**the whole field is ~5–100, not a tail**). su2's >600 B/admitted-pair falsifier is moot — **no pair is admissibly close**"* (kills the QA43 tail solver, S3) | **CONFIRMED — and refuted by the same memo's own table.** `ps1:56` reports S2 terminal solve *"MEASURED n600 (**med 7.55, max 156.55**)"*. A field with median 7.55 and max 156.55 is **heavy-tailed by a factor of 20**; "the whole field is ~5–100" is a **range**, and a range is the coarsest summary of a distribution (**class 5**). A *tail* solver operates on the good pairs, which a range cannot see | **VERDICT-CHANGING** (confirmed). Resolver is **$0 and seconds**: the 600-row per-pair array is already on disk (`ps1:113` `ps1_ladder.partial.jsonl`) — print the quantile curve and count pairs below the admission bar |
| **`ddm_b4s_burn4_charter_20260731.md:236`** — the burn's STOP gate: *"final_gate_dseg ≥ parent 0.004278 + **noise 3e-5** → **burn4.ALARM (RESMOKE_REGRESSED)** and STOP"*, a **hand constant**, while `ddm_lg1_lane_guard_20260731.md:97-99` forbids exactly this three files over: *"SE from the rows themselves … a **window-SE-derived slack, never a hand constant**"* | **CONFIRMED verbatim, both sides.** And the margin is thin: `b4s:176-181` records the pre-amendment kill at *"delta **+2.3e-5** within noise"* — **1.30×** against the hand constant. Flagged by the sweep as its one VERDICT-CHANGING row **with no sister catch anywhere**; I found none either (*did not find in scope*) | **VERDICT-CHANGING** (confirmed). This gate decides whether a multi-hour burn fires. Resolver: derive `t_crit·SE` from the 38 logged r1c parent gates already in `telemetry.jsonl` |

### 9b. Rows I did NOT re-derive — carried as SECOND-HAND

Reported for routing; **not promoted**, and no downstream consumer should treat them as verified.
Highest-ranked by their sweeps:

- `ddm_gc15_…:275` — the matched-compute sign-flip projected at a **per-epoch** rate (**class 4**)
  when gc15's own §5.1/§6 make the economics **impulse**-keyed. *My partial read:* gc15 **does**
  self-downgrade this to `PROVISIONAL-PENDING-VERIFICATION` / `DERIVED-BY-EXTRAPOLATION` at `:288`
  and names re-measurement as *"the only honest closure"* — so I grade it **SELF-CAUGHT-PARTIAL**,
  softer than the sweep did. The un-caught residue is real: §5.1 says per-epoch economics are
  mis-specified and §8 then uses them.
- `ddm_gc14_…:93` — *"consumed **81%** of that entire pool"*: an all-5-class Δ attributed to a
  2-class pool; the memo's own TEST 3 (`:141-148`) is the decomposition (**class 1/6**).
- `ddm_lg1_…:30` — the Lane guard budget **pinned at t=0 and never ratcheted**; `ba31:343` reports
  **λ = 0 across all 58 gates** — a guard that never actuated (**class 5**).
- `ddm_lg1_…:18` vs `ddm_gd1_undecided_…:196` — `g` reduced as an unweighted 36-pair mean on a
  block measured Lane-poor **−16.2%**, while `horvitz_thompson_mean` landed the **same day**.
- `ddm_bc1_…:81-83, :73, :116` — four pose-solver rows run at `s_t = 1.0` when `ST_GRID` is
  **0.005–0.24** (a **4–200×** scale error), then closed `LITE-absolute = INSTANCE-DEAD` and
  headlined *"FUNDAMENTAL"* while the `n = 2–3 pairs` qualifier sits at `:188` (**class 2 + 7**).
- `ddm_gd1_undecided_…:105` — *"removes 29-39% of the design error"* is an average over 7 proxies of
  which **2 have the opposite sign**, one being `lane_frac +68.90% worse` — the Lane guard's own
  estimand (**class 4**; `ba31:664`).

### 9c. NEW sub-classes the sweeps found — folded into §1 as 11–13

| # | class | evidence |
|---|---|---|
| **11** | **MECHANISM-PRESENCE STANDING IN FOR PERFORMANCE** (build-asymmetry read as physics) | `ddm_ba31_…:308` *"A comparison across those arms was reading **build asymmetry**, not physics."* · `:311` *"A `DEFERRED` label on an unbuilt branch reads downstream as a **resolved decision**."* · live: `b4s:50` DEFERs from-birth-KD, which `sb2:102` shows was **never buildable on tr1**. **Nothing automated detects it** (`sb2:68`) |
| **12** | **REPRODUCIBILITY RESIDUAL QUOTED AS THE NOISE FLOOR** (variance-source substitution) | Every noise floor in the probe group is a *within*-axis quantity (gate-to-gate residual, determinism check) spent as the denominator of a *between*-condition comparison. `dw1:119` *"noise floor = 2.99e-5 (B's gate residual std about its own trend)"* → used for a between-**arm** split at `:122`. `bc1:123` promotes a determinism check (*"noise floor EXACTLY 0.0"*) to *"always informative"*. **No seed replicate anywhere in the group.** One replicate of the dw1 B window re-prices three rows at once |
| **13** | **FALSE DECOMPOSITION** (per-element where the object is irreducibly joint — the *inverse* of class 6) | `gc13:26-28` — *"per-class rate attribution is **ill-posed**"* because the tr1 partition stream is **class-SHARED**; cure at `gc13:174`: rate rows are **per-STREAM**, exchanged to classes *"**through the waterfill, never by attribution**"* |

Two sharper **variants** of existing classes, also worth naming:
**8a SIGN-WELDING** — a composite whose terms have opposite sign in S, so it carries no directional
information (`ba31:871`: *"it welds 'losing true components' with 'shedding spurious ones,' which
have **opposite sign in S**"*). **1a DEGENERATE MULTIPLICITY** — N apparent corroborations that are
one column times a constant (`ba31:486-489`: fl1's *"fires for all 5 classes"*, where
`floor/cornerC ≡ (floor/resid)/0.14071` exactly, so *"the test **could not have failed to fire**"*).

### 9d. The counter-observation — coarseness is not monotonically wrong

Recorded because an audit that only found hits would be mis-calibrated. **Twice in the corpus the
COARSE instrument BEAT the fine one, measured, and the memos said so:** `v4b:35-36` (a 2-region
static horizon beat the per-pixel GT mask) and `nv1:111-114` (magnitude-ranking beat the QA80-field
cell-safety ranking). **The failure mode in this corpus is almost always *unstated instrument
choice*, not instrument crudeness.** That reframes the cure: not "always use the finer instrument"
but **"name which rung you are on, and say why that rung."**

### 9e. Best control rows (the corpus at its finest — these are the model)

- `ddm_fl1_…:88` — *"shifts every row down 0.33% and **changes no verdict**"*: an undecided default
  answered with a **measured sensitivity of the verdict**, not a note of its existence. **The single
  best control row found.**
- `ddm_xp1_…:73` — *"**the composite headline (−0.000677) would have hidden it**"* — class 8 refused
  by name, with the per-class ΔS ledger supplied.
- `ddm_fp1_…:109-110` — *"Sub-nucleus: **37% by COUNT but only 2.33% by AREA**"* — the count↔measure
  distinction made load-bearing; it is what defeats the #315 nucleus law.
- `ddm_cn3_…:150` — *"**A gate's LIVE-COUNT-0 is meaningless until its DENOMINATOR is asserted.**"*
  The corpus's own statement of class 1, derived from three measured instances.
- `ddm_gc15_…:179` — *"Precision on the null subspace (**do not conflate — I nearly did**)"* —
  and `:177`: *"The **shape within** the visible subspace … must be **RACED, not presumed**."*
- `ddm_gc16_…:105-119` — the 200-draw degenerate-baseline control (class 9's cure, executed).
- `ddm_v4d_…:87-89` — mean **and** median **and** mass-concentration reported together.

### 9f. Sweep coverage gaps (their words, preserved)

None of the three sweeps opened receipts, SSD custody, or source; all "the finer instrument was
available" claims are memo-internal or rest on a sister memo's *quotation* of a third file. Group 1
under-swept gc13/gc14 §7 crosswalks; group 3 used `ba31` as an oracle rather than as a subject and
owes it a dedicated pass; group 2 could not disambiguate `v4d`'s `~1e-3` fidelity band (attached to
an **S residual** at `:31` and to a **d_pose** value at `:132` — if d_pose, ±1e-3 moves √(10·d̄) by
≈ +0.0166 S, **57% of the claimed −0.028986 win**, and "first sub-0.97" is not assured). **That
disambiguation is one line of text and is the cheapest unclosed item in this document.**

---

## 10. SISTERS

`ddm_control_surface_exact_quartering_20260731.md` (§3b — the instrument ladder this audit uses) ·
`ddm_surface_correction_economics_20260731.md` (the average↔marginal exemplar) ·
`ddm_gc16_upstream_score_lowering_convocation_20260731.md` (§0 axis-weight correction; §1 the
degenerate-baseline control) · `null_subspace_rate_measure_20260717.md` (the model sentence: both
measures, labelled) · `codex_findings_ddm_m4_…:87` + `_m6_…:28` (the counted-byte guard) ·
`ddm_wr1_reverse_waterfill_20260729.md` (§6) · memories
`boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment_20260731` ·
`measured_number_equal_to_a_degenerate_baseline_is_not_a_measurement_20260731` (class 9) ·
`decompose_every_headline_number_disaggregation_is_the_signal_20260720` ·
`generic_basis_metric_never_optimal_cosine_fourier_euclid_20260729` ·
`negative_existence_claims_are_the_days_dominant_error_class_20260731` ·
`pose_is_the_largest_axis_on_the_own_vehicle_1_24_S_20260731` (row #5's subject).
