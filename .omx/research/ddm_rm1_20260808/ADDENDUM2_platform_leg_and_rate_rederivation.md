# ADDENDUM 2 — the platform leg is CLOSED, and the decoder confound is DISTORTION-only

Closes two legs the roadmap ADDENDUM explicitly listed as STILL UNMEASURED, using the durable
DALI asset the one qualifying Modal job already bought. Fully LOCAL, no dispatch.
`[macOS-CPU advisory]`, `score_claim=false`.

Sources: `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_{dali,av}.pt`
and `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (`lstars`). N = 117,964,800 labels.

## 1. THE PLATFORM LEG — CLOSED (was: "a separate free comparison against gt_n600.npz")

| comparison | disagreeing labels | rate | S units |
|---|---:|---:|---:|
| Modal-DALI vs Modal-AV | 20,671 | 1.7523023e-04 | 0.017523 |
| **Modal-AV vs LOCAL-macOS-AV** | **2** | **1.6954210e-08** | **0.000002** |
| Modal-DALI vs LOCAL-macOS-AV | 20,671 | 1.7523023e-04 | 0.017523 |

**Our local macOS AV decode is effectively identical to Modal's AV decode — 2 labels in
118 million.** The platform is NOT a confound. The entire measured delta is DECODER
(PyAV vs DALI/nvdec), exactly as the ADDENDUM's mechanism section derived.

Corollary worth stating: DALI-vs-LOCAL equals DALI-vs-Modal-AV to the label. So the
authority delta transfers to our local pipeline exactly; no platform correction term exists.

## 2. RATE RE-DERIVED ON THE AUTHORITY CACHE — the confound does NOT touch rate

Same generic coder (lzma preset 9|EXTREME) on each label field:

| label field | bytes | bits/symbol | vs PR130 HPAC 116,980 |
|---|---:|---:|---:|
| **Modal-DALI [AUTHORITY]** | **410,392** | 0.027831 | **3.508×** |
| Modal-AV | 410,548 | 0.027842 | 3.510× |
| LOCAL-macOS-AV | 410,584 | 0.027845 | 3.510× |

Spread across all three = **192 B on ~410,500 = 0.047%.**

**The decoder confound is a DISTORTION confound, not a RATE confound.** 20,671 differing
labels move d_seg by 61.25% of PR130's entire seg term while moving coded size by
essentially nothing. The two axes must therefore be gated separately.

## 3. What this updates in the roadmap

- **ADDENDUM §2c "the 135,732 vs 137,159 B comparison remains UNSAFE until re-derived on DALI
  labels" SPLITS.** The RATE half is now safe: provenance moves ~65 B on a 137 KB stream
  against a claimed 1,427 B gap — 4.6% of the delta, same sign-insensitive magnitude on all
  three fields. The DISTORTION half stays UNSAFE and is where the 61.25% lands.
- **CODER_LINEAGE_VS_HPAC.md's cache caveat is DISCHARGED.** That race ran on
  LOCAL-macOS-AV; on the AUTHORITY DALI cache the same coder gives 3.508× vs the
  reported 3.510×. The 2.2–3.6× verdict is unchanged by cache provenance.
- **A correction to that memo's §5 stands separately:** it said PR130's tokens are a *fitted*
  partition vs our *oracle*. That is WRONG per the ADDENDUM's source read
  (`build_gt_cache_official.py:50-58`) — both are the oracle argmax; they differ by DECODER.
  Same object, different decode. The comparison was tighter than claimed, not looser.

## 4. Also corrected: PR130_BASE_ROADMAP.md per-object share arithmetic

The roadmap's anatomy table lists tokens 116,980 + semantic 40,252 + carrier 23,054 +
hpac 20,179 + ZIP 100 = **200,565**, but the archive is **191,052**. The model sub-blobs are
listed RAW (83,485) while they ship LZMA'd jointly to **73,968**. Those are CONTENT shares,
not ARCHIVE shares. Archive shares are tokens 61.26% / models-compressed 38.74%. This matters
for waterfilling: a lever on `semantic_blob` buys COMPRESSED bytes at the joint-stream margin,
not raw bytes.

## 5. NOT checked

- Which convention nvdec actually emits — still unnamed direction, as the ADDENDUM states.
- The HPAC-coded (not generic-coded) rate on DALI labels: this measured the generic-coder
  proxy across the three fields. The 137 KB figures are HPAC numbers and their DALI
  re-derivation needs the prior retrained on DALI labels (that is hb3's job, on local Metal).
- No score claim. `score_claim=false`.
