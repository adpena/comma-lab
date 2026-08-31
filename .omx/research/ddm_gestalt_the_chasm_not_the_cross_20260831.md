# THE CHASM, NOT THE CROSS — every object is byte-feasible-and-45×-inaccurate, except one that is accurate and 1.305× over

`axis: [byte-exact real coders + macOS-CPU scorer-free exact count]` · `score_claim: false` · `promotable: false`
`verdict_scope: RE-READS existing measured artifacts (rd2's phase-A byte curve + this window's`
`cross-family counts) onto ONE plane. Opens nothing, closes nothing, escalates nothing. The`
`contribution is that the campaign's remaining problem is now stated as a single ratio.`
Date: 2026-08-31 · Owner: MAIN · Consumers: THE CROSS · #1267 · #1332 (xo1) · #1247/#1262 · [[m144]]

## STORES CONSULTED

`ddm_rd2_hg1_rate_distortion_curve/retained/rd2_phaseA_byte_curve.json` (READ AT SOURCE — 33 rows,
full-set to 100% corrections, real coders, its own 4 controls green incl. `coder_race_reproduces_
shipped_359280` and `base_flip_count_matches_bo2_40981`) · this window's
`GF1_FAMILY_CAPACITY_CROSSCHECK.json` · hot-state POINTER_LINE (the 0.0212% bar, provenance noted
in §4) · the pincer memo (same day).

## 1. ONE PLANE, FIVE OBJECTS, ONE BOX

Sub-0.12 requires **bytes ≤ 137,986** AND token error ≤ **~0.0212%**. Every object we have measured:

| object | bytes | byte half | token error | accuracy half |
|---|---:|:---:|---:|:---:|
| HG1 analytic packet (gf1) | 47,603 | IN | 1.1232% | **OUT (53×)** |
| born-small @ the cap (rd2 best-ordering) | 122,336 | IN | 0.9619% | **OUT (45×)** |
| *the sub-0.12 box corner* | *137,986* | — | *0.0212%* | — |
| **lb1 SHIPPED POINTER** | **180,083** | **OUT (1.305×)** | **0.0176%** | **IN** |
| born-small @ zero error (rd2, measured) | 460,652 | OUT (3.34×) | 0.0000% | IN |

**Zero of rd2's 11 frontier points are inside both halves.** Seven are inside the byte half — best
accuracy there 0.9619%, **45.4× the bar**. Exactly one is inside the accuracy half — cheapest there
460,652 B, **3.34× the cap**.

> **lb1 is the ONLY object inside the accuracy half, and it misses the byte half by 42,097 B — 1.305×.**
> Everything else is comfortably inside the byte half and 45–53× outside the accuracy half.

## 2. WHY "THE CROSS" IS THE WRONG PICTURE

The standing read is *born holds RATE, lb1 holds DISTORTION; cross them and you land sub-0.12*, with
`{byte-feasible} ∩ {distortion-feasible}` measured EMPTY at n=4–5 sampled points. The rd2 curve
replaces those points with a **continuum**, and the emptiness turns out not to be a sampling accident:

```
marginal bits/correction along the measured curve
  0.220  ->  0.422  ->  0.626  ->  0.830  ->  0.831  ->  0.984
        ->  1.187  ->  1.756  ->  2.240  ->  3.639            = 16.5x rise
```

The cheap corrections are the ones that buy almost no accuracy; the corrections that reach the bar
cost **3.639 bits each**. Born-small's rate advantage is therefore **not a half of the answer that
composition can pair with lb1's other half** — it exists *only* at accuracy levels 45× too coarse,
and buying the accuracy back costs 3.34× the cap. Two populations separated by a convex wall is a
**chasm**, not a cross: there is no meeting point to compose toward.

Same-object cross-check: born-small reaching **lb1's own accuracy** costs ~451,000 B (interpolated
between rd2's measured 308,840 B @ 0.2829% and 460,652 B @ 0.0000%) — **2.5× more than lb1 pays for
the same accuracy.** Where the two families can be compared at all, lb1 wins by 2.5×.

## 3. WHAT THAT MAKES THE REMAINING PROBLEM

**1.305× on one object.** Not "find a different object" — every different object measured is *worse
where it counts*, and the pincer memo (same day) showed the analytic-generator family converges at
1.12% across two unrelated constructions while the budget demands 0.018%.

This sharpens what a successor must be. Not *cheaper* — born-small is already 1.47× cheaper than lb1
and it does not help. It must be **cheaper at lb1's accuracy**, and rd2 prices that requirement for
the one alternative family we can price: 460,652 B, i.e. 2.56× worse than lb1 at better accuracy.

The 42,097 B demand on lb1 stands exactly where it was; what changed is that the *alternative-object*
escape route now has a measured shape instead of a hope, and the shape is a convex wall.

## 4. HONEST LIMITS

- The **0.0212% accuracy bar** is carried from hot-state, not re-derived here. The conclusion is
  insensitive to it: at 2× either way the byte-half objects are still 22–91× outside.
- rd2's curve is the **born-small + correction-residual composite**, one family. It does not bound
  representations that are not "small body + coded corrections."
- **§4a below: I checked that join immediately and it FIRED.** The two instruments disagree 12.1×.
- Nothing here is a score. No pointer movement was attempted.

## 4a. THE INSTRUMENT JOIN FIRED — 12.1×, two instruments, one label

I flagged the lb1-error instrument join as "the one thing worth checking." I checked it in the same
window and it disagrees:

| "lb1's token error" | mismatches | % of field | slack under the 0.0212% bar |
|---|---:|---:|---:|
| vs DALI GT (MEASURED here, `count_nonzero`) | 1,717 | **0.00146%** | **93.1%** |
| PYAV instrument (carried in hot-state) | 20,762 | **0.01760%** | **17.0%** |

**12.1× apart, both labelled the same thing.** This is the [[m99]] / #1260 genus — a retained number
does not carry its own reading semantics — and the known cause is the GT-lineage fork (#1142: DALI on
the CUDA axis, PyAV/`frame_utils` on CPU).

**What it does NOT change:** the families sit at 1.123–1.299%, i.e. **53–61× the bar**. A 12.1×
instrument shift cannot close a 53× gap in either direction, so §1's box arithmetic and the 45–53×
headline survive whichever instrument is authoritative.

**What it DOES change, and it is not small:** lb1's *accuracy slack* is **93.1% or 17.0% depending on
the instrument — a 5.5× swing.** Every calculation that trades lb1 accuracy for bytes inherits that
factor. The frontier is scored on **contest-CUDA**, whose GT decode is **DALI**, which suggests the
0.00146% row is the axis-matched one and lb1 has 5.5× more slack than the campaign has assumed — but
I have not traced the 0.0176% figure to its producing instrument, so I am NAMING this, not resolving
it. **Owed:** identify what generated 0.0176%, then re-derive any slack-dependent quantity on the
axis-matched GT. The tolerance ladder (#1255/#1257, best 33.7× over) is the first consumer to re-check.

## 4b. RESOLUTION OF §4a — APPEND-ONLY, 2026-08-31, by `ddm_gti1`

§4a above is preserved verbatim; nothing in it is rewritten. This block records what the owed
measurement found. Full write-up: `.omx/research/ddm_gti1_gt_instrument_resolution_20260831.md`.
Receipt: `/Volumes/APDataStore/pact/ddm_gti1_gt_instrument/GTI1_GT_INSTRUMENT_TRIPLE.json`.

**Both figures were real; the reading was missing.** All three pairwise counts, same n600 field,
both positive controls PASS, five-cell partition sums exactly:

| pair | mismatches | % of field |
|---|---:|---:|
| lb1 vs **DALI** GT | 1,717 | 0.0014555% |
| lb1 vs **PyAV** GT | **20,764** | **0.0176019%** |
| DALI GT vs PyAV GT | 20,673 | 0.0175247% |

The carried 0.0176% reproduces to three significant figures — the back-solved 20,762 was within 2 of
the true 20,764. It has **no producing receipt anywhere in the corpus**, and `main_hot_state.md` is
untracked, so it is the record-censored genus (#878), not a fabrication.

**DALI is axis-correct, confirmed at source.** `upstream/evaluate.py:31-42`: `device.type == "cuda"`
→ `DaliVideoDataset`, `else` → `AVVideoDataset`; line 58 builds `ds_gt` from it. The frontier is
contest-CUDA. No CLAUDE.md conflict — `frame_utils.py:201` shows `AVVideoDataset` uses the canonical
`yuv420_to_rgb`, not the forbidden `rgb24`.

**The structural finding §4a could not see.** Agreement partition: all-equal 117,943,223 ·
`lb1==DALI, PyAV differs` **19,860** · `lb1==PyAV, DALI differs` 813 · `DALI==PyAV, lb1 differs`
**904** · all-three-distinct **0**. So **95.65% of lb1's PyAV error is a property of the GT PAIR**,
and lb1's lineage-invariant error — both GTs agree, lb1 still wrong — is **904 = 0.000766%**.

**Corrections to §4a.** (1) lb1's slack is **93.1%** (96.4% lineage-invariant), not 17.0%; §4a's
5.5× arithmetic was correct, only the attribution needed fixing. (2) §4a routed the re-check to the
tolerance ladder (#1255/#1257); **that routing was misdirected** — `ddm_tv2` declares
`verdict_scope: the dx2 object` (archive `976f706d…`, 180,368 B, not lb1) and self-declares PyAV
lineage. It is not a consumer of lb1's slack. (3) The **0.0212% bar is now the only untraced number
in §1's box**, with a named units risk: `ddm_td1` measured 1,717 token errors coexisting with
34,930.6 scored flips, so token error and `d_seg` are not proportional, and 0.0212% is close to the
sub-0.12 `d_seg` allowance expressed as a percentage of sites (0.0201399%). Flagged, not closed.

**§1, §2 and §3 are UNCHANGED, and now for a measured reason.** `ddm_tv2` priced this trade on this
token stream: moving 100,000 tokens releases **87.8 B**. The whole slack is worth ~20 B against the
42,097 B demand. The 5.5× is a correctness fix, not a byte route. THE CROSS stands.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B,`
`d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
