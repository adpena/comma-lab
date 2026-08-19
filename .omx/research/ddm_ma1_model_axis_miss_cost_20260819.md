# ddm_ma1 — the within-miss relative law: the last un-adapted sector, built and priced

Date: 2026-08-19 · Arm: `ddm_ma1_model_axis_miss_cost_20260819` · Authority: exact
decode-identical code-length measurement, full n600 · **Score claim: false** ·
**Pointer moved: false** · **Byte-closed: NO** (blocker in §7)

`verdict_scope`: **INSTANCE** — the live ck1/ck2 tail (fx2's D1 build over the rr4 token
field). The byte credits are properties of *this* stream. No Modal dispatch, no seal
executed, $0 spent.

## Conclusion first

**I built the thing fx1 priced and never built, and it measures −104.584 B on the full
n600 field** — from 109,800.4393 B to 109,695.8553 B against the live D1 law, at a
decode-identical token field. Projected archive **−105 B**, **ΔS −6.9915e-05**, **20.0×**
the −3.5e-6 admit bar, on either live body.

**But the headline of this arm is a correction, not the byte count, and it changes what
the next arm should do.** My charter (and ck2 §4, and fx1 §5) described the miss sector as
a **77,241 B reservoir** — "43.7% of the whole member, 7.2× the entire gap to 0.15" — with
"about 1.9% extracted". **That framing is a vacuous denominator and I am withdrawing it.**
Measured here:

| quantity | bytes | what it actually is |
|---|---:|---|
| "cost of being a miss" | 77,241.46 | an **entropy**, overwhelmingly irreducible. fx2 §6 already measured the hit-event floor at 108,151 B against D1's 108,553 — so ~400 B, not 77,241 B, is in play there |
| within-miss **ceiling** (fx1 §5) | 1,247.19 | a **perfect-oracle** bound: it assumes a model that names the miss class for free |
| within-miss **hindsight-reachable** (this arm, §5) | **~180** | the best per-cell multiplicative correction fitted *in hindsight on the scored clip* — an upper bound no online model can reach |
| **realized here, online** | **104.584** | **58% of the best hindsight bound**, 68% of its own context's |

So the token stream's honest remaining free-rate reservoir is **~400 B (hit event, and
fx2 already took 797 of it) + ~180 B (this sector, of which I took 105)** — call it a few
hundred bytes, **not 77 KB**. Nobody should aim an arm at "11% of 77,241 B" again; that
number cannot be spent.

**Own-vehicle frontier: `S = 0.15710198138050818 @ 177,182 B [contest-CUDA T4, n600]`
(the ck1 pointer). This arm did NOT move it** — only MAIN's fire can, and this candidate
is not byte-closed.

## 1. Controls — the instrument before any verdict

All full n600, all re-run in my own hands. No row below is admissible unless these hold.

| control | target | measured | verdict |
|---|---:|---:|---|
| uncorrected HPAC cross-entropy | 112,109.57757858819 B | 112,109.5775785882 B | **+0.0000000 B** |
| within-miss sector (fx1 §5) | 1,247.19 B | 1,247.18782 B | −0.00218 (fx1 reported 2dp) |
| hit-event (fx2 §6) | 110,862.39 B | 110,862.38975 B | −0.00025 |
| being-a-miss (fx1 §5) | 77,241.46 B | 77,241.46310 B | +0.00310 |
| miss positions | 223,694 | **223,694** | exact |
| neighbour causality, all 4 offsets (fx2 R1) | 98.6945 / 98.6945 / 98.6301 / 97.3425 % | **identical to 4dp** | free positive control |
| **C1**: fx2 D1 through MY harness | −710.84 B vs rr4 | **−710.838 B** | +0.0017 (fx2 reported 2dp) |
| **C2**: my class, sector OFF ≡ C1 | bit-identical | **0.0 bits, same payload sha `d5301f4d…`** | **pass** |

**C2 is the load-bearing one.** My class collapses onto the law it extends *bit-for-bit*
at full scale, so every delta below is the new sector and never the plumbing.

**An independent check that I priced the LIVE body and not rr4's.** My D1 token projection
is `ceil(109,800.4393) = 109,801 B`; plus the 96 B residual table that gives **109,897 B**,
which is exactly the ck1 tail ck2 §4 reports. The lineage's own census closes on my
instrument without being told to.

**The test suite is not vacuous.** 19 tests; three defects injected and all three caught —
an inert stage, a reweight that leaks into the argmax column, and a non-causal neighbour
read. 124 inherited fx1/fx2 tests still pass.

## 2. The mechanism, and why it is the archive's own estimator

Verified at source, `ddm_rr4_free_corrector_v2.py:271-293`: `coding_row` scales every
non-argmax column by the **single scalar** `(1 - q) / one_minus`, and `observe` (L295-303)
folds only `hit = decoded == arg`. So the relative law inside the miss sector is exactly
the neural prior's, and the hit-event statistics never see which class a miss landed on.

That gives the **separability theorem** this arm rests on: a within-miss model perturbs
neither the hit-event code length nor the hit-event model's trajectory, so
`delta(total) == delta(within-miss)` **exactly**. It let me extract the 223,694-record
sector once and race architectures in **1.7 s** each instead of ~4 minutes — and the
full-field run then measured **−104.584 B** against a sector race that predicted
**−104.584 B**, so the theorem is confirmed rather than assumed.

The estimator is ddm_rr4's own KT count ratio, pointed one axis over:

```
M[c, k] = (n[c, k] + KT) / (e[c, k] + KT)        observed / prior-expected
w_k = row64[k] * M[c, k]                          (k != argmax)
coded_k = w_k * (S / W) * (1 - q) / one_minus     S = sum row64, W = sum w
```

`S / W` preserves the sector's total mass, which is what makes the identity control exact:
with `M == 1`, `W == S` bitwise, `S / W == 1.0` exactly, and the shipped row comes back
unchanged. Cold cells emit `M = 1`, i.e. exactly the prior — ddm_rr4's own cold-context
contract. No `log`/`exp`/`pow`/`**` anywhere on the decision path (AST-gated), so the
`ddm_rr2` desynchronisation class (S = 27.83) cannot recur.

## 3. The race — 60+ rows, full n600 sector, decode-identical, $0

Cell contexts, all causal and free at the receiver. Best of each family:

| Δ bytes | cell | cells | what it adds |
|---:|---|---:|---|
| **−104.584** | **`nb3_prev1`** | **1,296** | **3 causal neighbours + previous frame — SHIPPED** |
| −104.417 | `arg_nb3_prev1` | 6,480 | + argmax (5× the cells, 0.17 B worse) |
| −99.624 | `arg_nb3` | 1,080 | drops the temporal leg |
| −86.868 | `arg_nbup_nbleft_prev1` | 1,080 | 2 neighbours |
| −57.975 | `arg_nbmode_prev1` | 180 | neighbour *mode* instead of the joint |
| −30.232 | `arg_ubin8` | 40 | prior confidence only |
| −8.494 | `arg` | 5 | class only |
| −3.173 | `none` | 1 | a single global multiplier |

**The mechanism is legible and it is geometric**: a miss is the decoder discovering it is
in the *neighbouring* region, and the already-decoded neighbours name that region. The
richest contexts then *lose* (`arg_nb4_prev1` −99.5, `arg_nb3_prev1_bnd` −100.0,
`arg_nb3_prev1_prev2` −100.4) — real saturation, because an online model pays its own
learning cost and the code length is therefore its own held-out check (fx2 §6's argument).

**Relational tables were built and refused.** A second table keyed by "is class k present
in my decoded neighbourhood" adds **0.5 B** on top of the best cell (−104.584 → −104.944)
for a whole extra structure and extra decode work. Not shipped. The reason is fx2 §7's
context dilution: the cell already carries the neighbours, so the relational feature
re-estimates what is already estimated.

### The two constants I refused to inherit, and what refusing them was worth

`ddm_rr4`'s `MIN_COUNT = 32` and `[2^-4, 2^4]` clamp were derived for a 51,200-cell context
over 117.9M positions. This sector is 1,296 cells over 223,694 records — a different
regime, so both were swept (`cross_regime_constant_transfer` genus).

* **`min_count` is monotone; 1 wins.** Carrying 32 costs **12.9 B**, 13% of the whole win.
  KT already handles a cold cell safely; a hard gate on top of it only discards evidence.
* **And carrying it would have produced a FALSE SATURATION VERDICT.** At `mc=32` the rich
  contexts *lose* (`arg_nb3` −86.7 beats `arg_nb3_prev1` −78.6) and I wrote down "context
  saturates". At `mc=1` the ordering **inverts** and `nb3_prev1` wins by 18 B. A carried
  constant did not merely cost bytes, it manufactured a wrong conclusion about the family.
* **The clamp is a broad plateau** (8 → 1024 all within 0.7 B) whose peak sits at
  ddm_rr4's own 2^4. Same value, now measured rather than carried.

## 4. Decode wall-clock — the constraint that actually decides this

On this lineage the decode budget, not the byte count, picks the candidate: fx1 measured
the real parse-back at **1,639.78 s against 1,800 s**, and fx2 projected D1 at 1,682 s with
**118 s of margin** — the thinnest number in the lineage. Measured serially, two
independent pairings:

| pairing | baseline | candidate | marginal |
|---|---:|---:|---:|
| in-run, C2 (my class, sector OFF) → candidate | 176.9 s | 196.2 s | **+19.3 s** |
| isolated foreground, C1 (live D1 law) → candidate | 177.8 s | 190.3 s | **+12.5 s** |

The two baselines agree to 0.9 s, which cross-validates them. (C1 measured 203.6 s as the
*first* run of the in-run triple — cold page cache on the 1.18 GB memmap — so within-run
ordering carries ~27 s of noise and the isolated pair is the cleaner instrument.)

Applying fx1's one-point parse-back calibration (×1.260):

| | harness | parse-back | D1's 118 s margin becomes | exchange |
|---|---:|---:|---:|---:|
| optimistic | +12.5 s | +15.8 s | 102.2 s | 6.64 B/s |
| **best estimate** | **+15.9 s** | **+20.0 s** | **98.0 s** | **5.22 B/s** |
| conservative | +19.3 s | +24.3 s | 93.7 s | 4.30 B/s |

**The finding worth banking: this sector is the cheapest remaining byte on the token
stream per unit decode time, by a wide margin.** fx2's only remaining hit-event upgrade
(E1, 19 members) buys 87 B for +89 s = **0.98 B/s**; this buys 104.6 B for ~20 s =
**5.2 B/s**, **5.3× better** (4.4× on the conservative timing). The reason is structural:
each extra mixer member adds a table update per group, while this adds one gather over
neighbours the corrector **already computed** — I cache fx2's own `_causal_neighbours`
rather than recomputing it, so the marginal is the index arithmetic and the reweight alone.

## 5. Where the sector's floor actually is (the pricing that supersedes the ceiling)

fx1's 1,247.19 B is a perfect-oracle bound. Two measurements say what is really reachable.

**First, every model that REPLACES the prior is far worse than the prior.** KT-smoothed
static per-cell laws, in-sample (hindsight, so a genuine bound):

| context | cells used | in-sample floor | vs the prior's 1,247.19 B |
|---|---:|---:|---:|
| `arg_nb4_prev1` | 849 | 5,625.0 B | **4.5× WORSE** |
| `arg_nb3_prev1` | 621 | 5,650.4 B | 4.5× worse |
| `arg` | 5 | 27,785.1 B | 22× worse |

The prior's relative law is **per-position and continuous**; a categorical cell table is
neither, and throws away almost everything. That is precisely why the shipped design must
*multiply* the prior rather than replace it — and it retires "model the within-miss
distribution" as a standalone idea.

**Second, the bound that governs THIS model class** — the best per-cell *multiplier* fitted
in hindsight on the whole scored clip:

| context | hindsight multiplicative | gain vs prior |
|---|---:|---:|
| `arg_nb4_prev1` | 1,066.9 B | **−180.3 B** |
| `arg_nb3_prev1` | 1,073.1 B | −174.1 B |
| **`nb3_prev1` (shipped)** | **1,093.8 B** | **−153.4 B** |
| `arg` | 1,240.1 B | −7.1 B |

So the realistic reservoir here is **~180 B**, and the online model banks **104.584 B** —
**58% of the best hindsight bound across all contexts, 68% of its own context's**. The
residual is ~50-75 B, hindsight-optimal and therefore not fully reachable. **This sector is
now close to done, and it should not absorb another arm.**

## 6. Composition with the live bodies

`non_semantic_sections_unchanged = True` in ck1's own generation receipt, and ck2 §4 reports
the tail unchanged at 109,897 B — so the token stream I priced is the one both live bodies
ship. Projected, on the calibrated `archive = base − (token_D1 − token_ma1)` formula whose
own control landed at `token_delta_vs_target: 0`:

| body | archive sha | bytes | → bytes | S | → S | gap to 0.15 closed |
|---|---|---:|---:|---|---|---:|
| ck1 pointer | `35c318d5…` | 177,182 | **177,077** | 0.15710198138 | **0.15703206619** | 0.98% |
| ck2 plane2 | `0aa1cada…` | 176,525 | **176,420** | 0.15666451205 | **0.15659459686** | 1.05% |

ΔS = **−6.9915e-05** on either, because the rate term is axis-independent and the two
distortion legs cannot move — the decoded token field is bit-identical by construction, and
C2 proves the construction. **The candidate survives either pointer outcome**, which is why
it is priced on both.

ck2 transforms the layout of the semantic and carrier bodies; this transforms the tail's
probability law. fx1 §9.4 established those byte regions are disjoint in both directions.
**I did not measure the composition** — that is a build step, not an inference, and it is
owed alongside the byte-close.

## 7. Honest limits

* **NOT BYTE-CLOSED, and the blocker is the one fx2 hit.** `ddm_pq2_compress_e2e.py:116`
  pins `rc64_source_sha256 = 5c75e2c7…`. I hashed **every** `rc64_backend.c` on both trees:
  158 copies, **2 distinct contents** (`05839d14…` ×157, `b249b77b…` ×1), and **neither
  matches**. I did **not** bypass the fail-closed check — that check is what stands between
  a rebuild and silently different bytes. Every archive figure above is a projection on a
  calibrated formula, not a measurement. Clearing this pin is the first owed step.
* **Selection on the scored clip.** The cell, `min_count` and the clamp were chosen by
  racing on the scored video, exactly as fx1 §7 and fx2 §9 disclosed for their own. Stated
  as a bound rather than a defence: the *family* is robustly negative — **all 12** first-pass
  cell contexts measured negative, and every one of the 60+ rows raced. The shipped cell
  carries no fitted scalar, only a partition of already-decoded classes, and its hindsight
  bound (§5) says at most ~75 B more is available to any amount of further selection.
* **The ~180 B hindsight bound is itself in-sample** and therefore optimistic; the true
  online-reachable figure is below it. Held-out static fits are worse than the prior in
  every row, which is fx2 §6's non-stationarity finding reproduced on this sector.
* **Cross-platform exactness is argued, not demonstrated.** Every operation is IEEE
  correctly rounded and the AST gate refuses transcendentals, but `ddm_rr2` is the receipt
  that a correct local proof is not a cross-platform one. Parse-back is a hard gate.
* **The decode margin is genuinely thin and this makes it thinner.** ~98 s of 1,800 s on a
  projection built from a one-point calibration over an advisory-local base. If MAIN's T4
  headroom is tighter than projected, this row and fx2's D1 compete for the same margin —
  and on the measured exchange rate this one is 5.3× the better buy.
* **Relational tables and the four-neighbour template are measured negatives**, not
  untried. Recorded so a later arm does not re-derive them.

## 8. Custody

All under `/Volumes/APDataStore/pact/ddm_ma1/`, 31 files / 17.7 MB, every one hashed in
`MA1_RETENTION_MANIFEST.json`. **Per-architecture payloads retained, not only the winner.**

| path | sha256 (16) | contents |
|---|---|---|
| `retained/miss_sector_n600.npz` | `d23db2bd6c4b1a00` | the 223,694-record sector itself — features, prior rows, decode order |
| `retained/bits_C1_fx2_D1_live_law_n600.npy` | `d5301f4dfe840626` | control C1 |
| `retained/bits_C2_ma1_within_miss_OFF_n600.npy` | `d5301f4dfe840626` | control C2 — **same sha as C1, the nesting proof as a byte fact** |
| `retained/bits_ma1_nb3_prev1_n600.npy` | `23005a8b0994b058` | the candidate |
| `race/r1…r9_*.json` | — | all 60+ raced rows with their deltas |
| `probe/sector_floor.json` | — | §5's floors + the hindsight multiplicative bound |
| `race/confirm_full_field.json` | `4c56d4cca37d8dc4` | the three-row serial confirm |

Code (commit `3296b557c3`): `experiments/ddm_ma1_{extract_miss_sector,race_within_miss,
within_miss_corrector,confirm_full_field,sector_floor}.py` ·
`src/tac/micro_edit/tests/test_ma1_within_miss.py` (19 tests).

## STORES CONSULTED

`.omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md` §5 + §9 (the decomposition,
the 1,247 B ceiling, the parse-back calibration) · `.omx/research/ddm_fx2_model_axis_all_
sections_20260818.md` §3-§8 (D1, the SSE negative, the R6 floor argument, the byte-close
blocker) · `.omx/research/ddm_ck2_container_plane2_eleventh_move_20260819.md` §3-§4 (the
live bodies, the section census, the reservoir framing this memo corrects) ·
`.omx/research/ddm_ck2_fire_order_draft_20260819.md` (both bodies' S and bytes) ·
`/Volumes/APDataStore/pact/ddm_ck1/GENERATION_RECEIPT.json` (**verified at source**:
`non_semantic_sections_unchanged = True`) · `experiments/ddm_rr4_free_corrector_v2.py:240-303`
(**read at source**, not relayed: the rank-one transport, the untouched miss sector) ·
`experiments/ddm_fx2_model_axis_corrector.py:393-500` (the causal template I cache) ·
`src/tac/micro_edit/coder_replay.py` (the instrument) · memory
`[[probability-model-axis-live-fx1-sweep-prior]]` (dial ordering inherited, never
re-measured) · `[[cross_regime_constant_transfer_genus_finishing_stage]]` (§3) ·
`[[the_denominator_and_the_falsifier_can_both_be_vacuous_20260816]]` (the §Conclusion
correction) · `[[read-closed-negatives-as-actuator-datasheets]]`.

## NEXT_IF_RESUMED, ranked

1. **Clear the `rc64_source_sha256` pin, then byte-close both this and fx2's D1.** It blocks
   two arms now, not one. The fix is a `--recipe-json` carrying the correct pin or restoring
   the pinned file; do **not** relax the check. Until it clears, no token-stream work in this
   lineage can produce an archive byte.
2. **Compose ma1 + ck2 plane2 and parse-back once.** Disjoint regions by fx1 §9.4, but
   measure it. Expected 176,420 B / S 0.15659460.
3. **Do NOT aim another arm at the miss sector.** §5 prices the remaining hindsight-optimal
   headroom at ~75 B and the online-reachable figure below that.
4. **Do NOT re-run the reservoir framing.** The 77,241 B figure is an entropy; §Conclusion
   replaces it with ~400 B (hit event, mostly spent) + ~180 B (this sector, mostly spent).
   The rate axis on this lineage is close to exhausted and ck2 §4 reached the same wall from
   the container side — the remaining ~82% of the gap is a representation question.
5. **Re-derive `min_count` whenever a model in this family moves regime.** §3 is the second
   receipt in two arms that an inherited constant flipped a family verdict, not just a byte
   count (fx2 §5's learning-rate plateau was the first).

---

## SUPERSESSION (append-only, added 2026-08-19 by `ddm_rv14f` — original text above UNTOUCHED)

Per Catalog #110/#113 the claims above are preserved verbatim as
HISTORICAL_PROVENANCE. This section supersedes one of them.

**Superseded claim — §7 "Honest limits", first bullet (lines 226-229):**

> *"I hashed **every** `rc64_backend.c` on both trees: 158 copies, **2 distinct contents**
> (`05839d14…` ×157, `b249b77b…` ×1), and **neither matches**."*

**and §NEXT_IF_RESUMED item 1 (lines 290-292):**

> *"Clear the `rc64_source_sha256` pin … Until it clears, no token-stream work in this
> lineage can produce an archive byte."*

### The correction

**The pinned file exists and has since 2026-08-10.** Measured by direct hash:

    /Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c
    sha256 5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6   12,222 B

Three arms reached this independently: `ddm_rc1x` (a6e07d42df), `ddm_rv13` from the
filesystem, and this registry pass. **The pin blocker was never real.** ma1's byte-close was
not blocked by a missing file; it was blocked by a sweep whose scope did not reach the tree
the file was on.

### The denominator, with its scope stated (`m53`)

The "158 copies, 2 distinct contents" count is not wrong — it is **scoped**, and the scope
was not stated beside the number. Re-measured with the scope declared:

    scope:  /Volumes/VertigoDataTier/pact + /Volumes/APDataStore/pact + /Users/adpena/Projects/pact
    method: find -name rc64_backend.c -not -name '._*' -type f, then sha256 every hit
    result: 241 files, 4 distinct contents

Three published counts for what read as one sweep — ma1 *158 copies*, fx2 *252 `.c` files*,
rc1x *232 copies* — are three different scopes over two different populations (`rc64_backend.c`
files vs all `.c` files). None stated its scope. That is the `m53` negative-existence law
half-applied: rc1x named the genus, and the campaign kept publishing bare counts.

**The four bodies are now enumerated once, with roles**, in
`reverse_engineering/rc64_backend_role_registry.{md,json}`. Read that before searching for
this filename again — a name-keyed search has 241 hits and can land on any of four bodies,
two of them encoder-class.

### Corrected status

**UNBLOCKED-BUT-DOMINATED, not blocked.** ma1 byte-closed on the D1 lineage 2026-08-19 at
180,345 B via the rc1x two-role recipe. That is dominated by the live pointer (to1 176,420 B,
and since then up3 at S 0.15652626). The owed step in NEXT_IF_RESUMED item 1 is **discharged**;
items 2-5 are unaffected by this correction.

Sources: `.omx/research/ddm_rv13_landing_wave_review_20260819.md` §F1/§F8/§F13 ·
`.omx/research/ddm_rc1x_rc64_recipe_fix_20260819.md` ·
`.omx/research/ddm_fx2_t4_sealed_fire_order_SUPERSESSION_20260819.json` ·
`.omx/research/ddm_rv14f_rv13_fix_batch_20260819.md`
