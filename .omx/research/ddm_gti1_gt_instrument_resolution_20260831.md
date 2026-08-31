# ddm_gti1 — the instrument join RESOLVED: both numbers are real, DALI is axis-correct, and 95.65% of the PyAV figure is a property of the GROUND TRUTHS

`axis: [macOS-CPU scorer-free exact count]` · `score_claim: false` · `promotable: false`
`verdict_scope: the lb1 token field (sha 9ba2e52b…) against the two legitimate GT decode lineages,`
`n600, all 117,964,800 positions. Settles which reading is axis-correct and re-prices the slack.`
`Opens nothing. Pointer UNMOVED.`
Date: 2026-08-31 · Owner: ddm_gti1 (Opus arm) · Consumers: `ddm_gestalt_the_chasm_not_the_cross`
§4a · #1260 · #1142 · #1255/#1257 · [[m99]] · [[m88]]

## STORES CONSULTED

`upstream/evaluate.py` + `upstream/frame_utils.py` (READ AT SOURCE, read-only) ·
`src/tac/gt_lineage.py` + `src/tac/gt_lineage_registry.json` (59 artifacts) ·
`GF1_FAMILY_CAPACITY_CROSSCHECK.json` (gf1's own declared target paths) ·
`ddm_gl2_u8_lineage_coverage_20260816.md` · `ddm_gl1` · `ddm_pi2` · `ddm_td1` · `ddm_rt1` ·
`ddm_ms9` · `ddm_tv2_evaluator_tolerance_curve_20260824.md` ·
`experiments/ddm_rx1_rate_representation_attack.py` · hot-state POINTER_LINE.

---

## 1. THE ANSWER

I measured all three pairwise counts over the same n600 field. Both positive controls PASS and the
five-cell partition sums to the field exactly.

| pair | mismatches / 117,964,800 | % of field |
|---|---:|---:|
| lb1 vs **DALI** GT | **1,717** | **0.0014555%** |
| lb1 vs **PyAV** GT | **20,764** | **0.0176019%** |
| DALI GT vs PyAV GT | 20,673 | 0.0175247% |

**Both carried numbers were real and correctly transcribed.** The hot-state 0.0176% reproduces to
three significant figures; MAIN's back-solved 20,762 was within 2 of the true 20,764. Nothing was
fabricated. What was missing was the *reading*.

**The axis-correct number is 1,717 = 0.00146%.** `upstream/evaluate.py:31-42` selects the GT decoder
by device — `device.type == "cuda"` → `DefaultDatasetClass = DaliVideoDataset`, `else` →
`AVVideoDataset` — and line 58 builds `ds_gt` from it. Our frontier is `[contest-CUDA T4 n600]`, so
the authority GT is DALI. This confirms `tac.gt_lineage.AUTHORITY_LINEAGE = DALI_NVDEC` from the
upstream source rather than from the registry's own assertion.

**No CLAUDE.md conflict.** `frame_utils.py:201` shows `AVVideoDataset` calls the canonical
`yuv420_to_rgb`, not the forbidden PyAV `rgb24` path. Both lineages are legitimate; PyAV is simply
the **contest-CPU** axis. The rule and this finding are not in tension, exactly as `gt_lineage.py`
already documented.

## 2. THE STRUCTURE — and this is the part worth carrying

Three Hamming counts cannot tell "lb1 has its own errors on top of the split" from "lb1's apparent
PyAV error IS the split." The partition can, so I measured it:

| agreement pattern | sites |
|---|---:|
| all three equal | 117,943,223 |
| **lb1 == DALI, PyAV differs** | **19,860** |
| lb1 == PyAV, DALI differs | 813 |
| **DALI == PyAV, lb1 differs** | **904** |
| all three distinct | **0** |

It closes exactly: `813 + 904 = 1,717` · `19,860 + 904 = 20,764` · `19,860 + 813 = 20,673`.

> **95.65% of lb1's PyAV-instrument error (19,860 of 20,764) is a property of the GT PAIR, not of
> lb1.** And `all_three_distinct = 0` — lb1's token equals at least one ground truth at *every one*
> of the 117,964,800 positions.

So there is a third and better number than either of the two in dispute. lb1's **lineage-invariant**
error — both ground truths agree and lb1 is still wrong — is **904 sites = 0.000766%**. Even the
DALI figure of 1,717 is 47.4% lineage artifact (the 813 sites where lb1 matches PyAV and DALI is the
outlier).

## 3. THE TRACE — untraced but REPRODUCED, which is the #878 genus, not fabrication

I could not find a producer for 0.0176%. `0.0176` / `99.9824` / `20762` appear nowhere in
`.omx/research/**`, `experiments/**`, `src/**` or `tools/**` as a token-error figure; every other
corpus hit is a different quantity in different units (a 2026-05 baseline `d_seg` of 0.0176, a
renderer-pullback cosine `0.0176974146`, a `0.0176 S` roadlane rate). And
**`.omx/state/main_hot_state.md` is UNTRACKED — `git log` on it returns zero commits** — so there is
no git provenance either.

The honest disposition is therefore **not** "no producer, discard it." I re-measured the quantity and
it is right. This is a real measurement whose producing receipt was never written down — the
record-censored genus (#878), where the number survived and its reading semantics did not. That is
precisely [[m99]]: *a retained number does not carry its own reading semantics.*

**Provenance of the field itself is solid**, and I checked it rather than assuming it. The measured
field is `ddm_dc1_20260816/retained/redecoded_tokens_n600.u8`, sha `9ba2e52b3096…`, which is
literally `EXPECTED_SPATIAL_SHA256` at `experiments/ddm_rx1_rate_representation_attack.py:65` and is
asserted against the receiver's `decoded_token_sha256` at `:1206`. The shipped token stream is
lossless and digest-pinned, which is *why* hv1 (2026-08-16, `ddm_td1`/`ddm_rt1`) and lb1 (2026-08-29)
report the identical 1,717: same field, by construction. The number is not stale-inherited.

## 4. CONSUMERS — re-derived, with per-row disposition

| consumer | disposition | why |
|---|---|---|
| chasm §1 box, lb1 token-error cell | **RE-DERIVED** | 0.0176% → **0.00146%** (DALI). Byte half still OUT at 1.305×; headline 45–53× untouched, as §4a already said. |
| lb1 accuracy slack | **RE-DERIVED** | 17.0% → **93.1%** under the carried 0.0212% bar (96.4% on the lineage-invariant 904). MAIN's 5.5× arithmetic was correct; only the attribution needed fixing. |
| tolerance ladder #1255/#1257 (tv2, 33.7×) | **UNAFFECTED — routing was misdirected** | tv2's own header declares `verdict_scope: the dx2 object` (archive sha `976f706d…`, 180,368 B — *not* lb1) and `GT lineage PYAV_YUV420_TO_RGB (NOT the authority DALI_NVDEC)`. It never consumed lb1's slack; it measured dx2's τ curve. tv2 labelled its own axis honestly. §4a named it as "the first consumer to re-check"; on inspection it is not a consumer at all. |
| gf1 frontier `packet_B + 0.2909×mismatches < 85,020` | **UNAFFECTED** | gf1's receipt names `dali_gt` explicitly. It was already on the axis-correct instrument; the 5.09× refusal is untouched. |
| the 0.0212% / 0.0178% bars | **CANNOT BE RE-DERIVED — see §5** | equally untraced, and now the only untraced number in the box. |

**Does the 5.5× open a byte route? No — and the reason is measured, not assumed.** More slack means
lb1 could trade accuracy for bytes. `ddm_tv2` already priced that trade on this token stream:
**moving 100,000 uniformly-chosen tokens releases 87.8 bytes.** The whole slack under the carried bar
(~23,291 extra errors) is worth roughly **20 B** against a **42,097 B** demand. The correction matters
for correctness and for any future object; it does not move lb1's rate problem. THE CROSS stands.

## 5. THE BAR IS NOW THE ONLY UNTRACED NUMBER, AND IT MAY BE A UNITS ERROR — FLAGGED, NOT CLOSED

`0.0212%` and `0.0178%` appear **only** in hot-state and the chasm memo. Zero producers in the
corpus. The chasm memo's own §4 says so of the first one.

There is a live risk I can state but not settle. `ddm_td1` measured on this exact vehicle that
**1,717 token errors coexist with 34,930.6 scored flips**, and concluded ~95% of the seg term is
render→SegNet round-trip loss *"that our labels did not cause."* Token error and `d_seg` are
therefore **not proportional** — the map is not a multiplier. Two consequences:

1. A "token-error bar" obtained by converting a `d_seg` budget through a single ratio is unsound in
   both directions.
2. `0.0212%` is suspiciously close to a `d_seg` budget expressed as a percentage of sites: holding
   lb1's distortion fixed, the sub-0.12 seg allowance is `0.0201399 S` → `d_seg = 2.01399e-4` →
   **0.0201399% of sites**. If the bar is a `d_seg` bar being compared against *token* errors, the
   whole accuracy column of the box is cross-unit.

I am **naming** this, not resolving it — the same discipline §4a applied to me. **OWED to MAIN:**
derive the accuracy bar from the S-arithmetic at the live operating point, state its units
explicitly, and re-read the box's accuracy column against it.

## 6. THE FORK — measurement CLEAN, registry coverage NOT

**The fork is not silently live in the numbers.** The DALI target the whole family cross-check uses,
`dali_gt_full_n600.u8` (sha `a98b9067…`), is **bit-identical — 0 differing sites — to the registered
DALI artifact `91d3ff11a904c476`** (qs3's `gt_argmax_n600.npy` payload). That is
`EMPIRICAL_EXACT_MATCH`, the strongest rung on gl1's ladder. gf1, bz2, bz2d and bs3 all read the
authority lineage. Correct — but by construction, not by record.

**The registry coverage gap is live and structural.** Denominator, measured:

- `dali_gt_full_n600.u8` is **UNREGISTERED**. It is the only ground truth among the fields this
  window's cross-check reads, so on the GT axis that is **1 of 1 unregistered**. (All 9 comparison
  fields are unregistered; the other 8 are candidate/generated fields the GT registry does not claim
  to cover, and I am not inflating them into the GT count.)
- `tac.gt_lineage.GT_ARTIFACT_SEARCH_ROOTS` holds **three VertigoDataTier paths and zero APDataStore
  paths**, so `assert_gt_population_registered()` is **structurally blind** to the entire
  APDataStore-resident GT population — every arm since ~2026-08-28.

This is the `ddm_gl2` genus one axis over: gl2 cured an **extension** allow-list that failed open on
`.u8`; the surviving hole is a **root** allow-list that fails open on a whole volume. gl2's own
argument applies unchanged — an allow-list fails OPEN on everything nobody anticipated.

Registry-ready row for MAIN (measured here, not hand-typed):
`sha256 a98b90678ca5d4e12b385d2c8596839b368af8d52277eea3c1d3666f7a4c9b3d` · 117,964,800 B ·
`dali_gt_full_n600.u8` · lineage `DALI_NVDEC` · evidence `EMPIRICAL_EXACT_MATCH` (0/117,964,800 vs
`91d3ff11a904c476`).

## 7. DOES ANYTHING FLIP?

**No campaign verdict flips.** THE CROSS stands; the 45–53× class bar stands; gf1's 5.09× refusal
stands; the 42,097 B demand stands; the route table stays empty. Three corrections land:

1. lb1's accuracy slack is **93.1%**, not 17.0% — and **96.4%** on the lineage-invariant reading.
2. §4a's routing to the tolerance ladder was misdirected; tv2 is a different object on a
   self-declared non-authority axis.
3. The accuracy **bar** is now the sole untraced number in the box, with a named units risk.

## 8. HONEST LIMITS

- I did not run a scorer. Token-field mismatch is **not** `d_seg` and must never be read as one.
- I did not verify that lb1's shipped archive decodes to `9ba2e52b…`; I verified that the shipped
  receiver **asserts** it does (`ddm_rx1:1206`). That is a code-level pin, not an archive replay.
- The bar remains untraced. §5 is a flag with arithmetic, not a verdict.
- The `~20 B` tolerance figure in §4 transfers tv2's dx2 price to lb1. Same pinned token field, but
  a different archive — treat it as an order-of-magnitude, not a measured lb1 row ([[m143]]).

## CUSTODY

Instrument `experiments/ddm_gti1_gt_instrument_triple.py`. Receipt
`/Volumes/APDataStore/pact/ddm_gti1_gt_instrument/GTI1_GT_INSTRUMENT_TRIPLE.json`
(sha256 `f5f5be3f24ac1e09935bd35c730b9b4b2672bcd3c0c9f678b0e232a9118d4209`), carrying sha256+bytes for
all three input fields, both positive controls with their expected values, and the partition.
Elapsed 11.4 s. Inputs: `9ba2e52b…` (lb1 field) · `a98b9067…` (DALI GT) · `36c6be71…` (PyAV GT).

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B,`
`d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
