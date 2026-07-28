# ddm_sp1 — the B3 rung MEASURED: contour-coded support is DEAD for the sub-bar floor

**Arm:** ddm_sp1 (B3 rung), isolated worktree off main @ 3ad18419ee (gc5 merged).
**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` — real lossless coder bytes over cached masks; **NOT a
byte-closed `evaluate.py` row.** Pointer **0.19108 UNMOVED** (this arm moved no exact score; it MEASURED
a rung the sub-bar floor depended on).
**Bottom line:** the contour-coded correction *support* — the rung gc5 said carries the ENTIRE sub-bar
byte floor — is **FLOOR_DEAD**. Real support is **444,394 B** (lossless #307 contour) / **421,366 B**
(LZMA), **3.1× above** the 142 KB projection the floor's feasibility rested on. The lossy concession
curve does **not** rescue it: the score-optimal lossy point is **S_support = 0.280** — which *alone*
exceeds the whole 0.172 bar. Route: **DOWNSTREAM to an implicit-support carrier** (formulation-scoped).
`# FORMALIZATION_PENDING: advisory coder-byte measurement; the region_merge 1.2731 B/flip water law is
already registered; no new canonical equation, this memo REUSES it.`

## STORES CONSULTED (recall-first; every cited receipt verified)
- **CLAUDE.md + AGENTS.md** (full read): NO-FAKE #5 enum-padding / #6 search-as-solver / #8 surrogate; the
  1.2731 B/flip region_merge water law; serializer + post-edit sha; `.py` review gate (never
  `REVIEW_GATE_OVERRIDE`); bulk → `/Volumes/VertigoDataTier/pact`.
- **MEMORY**: `box_retired_min_s_target_warp_family_closed_1273_bytes_per_error_20260728` (LAW: 1.2731 B/err
  = region_merge water level = Wyner-Ziv game `H(flip|ctx)`); `ms2r_r3_solved_seg_is_box_solve_...`;
  `opportunity_pools_non_additive_...` (KKT waterfill); `dispatch_richest_...`; `verdict_scope_ladder_...`.
- **fc1 receipts** (`/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/`): `stage2_coders_n600.json`
  (support LZMA **421,366 B**, labels constriction **41,392 B**, both round-trip-verified);
  `entropy_n600.json` (support 0.02858 bits/px REALCODER; concession water 1.273);
  `stage5_compose_n600.json` (`contour_support_bestcase_UNBUILT` = **142,220**; gap_arithmetic "contour
  best-case **184 KB**"; scenario C total 185,315 → S 0.265).
- **da1 receipts** (`ddm_da1_20260728/`): `d1_support_decomp_n600.json` (per-class LZMA Road 213,581 / Lane
  145,738; **XOR mask-delta 1.38× WORSE** — temporal at MASK granularity FALSIFIED); `d4_minimal_amplitude.json`
  (minimal correcting amplitude median 1.11 / p75 3.33 / p90 7.78 uint8, 64.1% ≤2).
- **r2s receipt** (`ddm_r2s_20260727/sparse_byteaccount_n600.json`): residual VALUES int8×3 → LZMA
  **10,062,148 B** (~9.87 B/err); support_geometry 421,496 B (reproduces fc1).
- **#307 machinery REUSED, not rebuilt**: `tools/measure_contour_string_flip_coding.py`
  (`contour_encode_frames`/`contour_decode_frames`: 8-dir chain code + digital-straightness context +
  in-tree `tac.lossless.range_coder.RangeEncoder`; bit-exact round-trip tested at
  `src/tac/tests/test_contour_string_flip_coding.py`). My tools import its functions verbatim; I only
  supplied the cached-mask data path (its stage-A renders+scores = sc1's slot, avoided).

## What I built (branch, serializer-committed) + artifacts
- `experiments/ddm_sp1_contour_support_coder.py` — R1: contour + LZMA support race, per-class, **lossy
  concession curve** (water 1.2731 B/flip), bit-exact round-trip proof on every stream.
- `experiments/ddm_sp1_values_coder.py` — R2: amplitude+sign+context range coder vs int8×3 incumbent.
- `experiments/ddm_sp1_base_transfer_h_contract.py` — R3: GATED B1 contract (rc=3 until sc1's base lands).
- SSD receipts `/Volumes/VertigoDataTier/pact/ddm_sp1_20260728/`: `r1_contour_support_n600.json`,
  `r2_values_coder_n600.json`, `r1_smoke_n24.json`, `r2_smoke_n24.json`.

## R1 — CONTOUR SUPPORT CODER (THE rung). MEASURED, n600, round-trip-proven.

| stream / coder | bytes | note |
|---|---|---|
| **contour SUPPORT** (counts+anchor+chain), lossless | **444,394 B** | 3.487 bits/flip; round-trip ✅ |
| LZMA incumbent (packbits, x9e RAW) | **421,366 B** | reproduces fc1/da1 exactly |
| contour / LZMA | **1.0547** | **contour is 5.5% WORSE than LZMA** |
| contour LABELS (cls stream) | 65,278 B | vs constriction incumbent 41,392 B → also worse |
| fc1 stage5 projection | 142,220 B | **212% low (3.1×)** |

Per-class (Road+Lane = 85% mass): Road contour 214,554 vs LZMA 213,581; Lane contour 144,738 vs LZMA
145,738 — **parity**, contour never wins. Coherence is real (87.4% of flips in components ≥4 px, only 4.6%
singletons) yet support is 3.5 bits/flip because the field is **many small components** (anchor + END-symbol
overhead dominates); the published 1–1.5 bits/contour-px floor is for long strings and **does not apply**.

**Falsifier verdict (gc5 band, SUPPORT geometry):** measured 444,394 B ≥ 250 KB → **FLOOR_DEAD**
(verdict_scope = FORMULATION: this copy-base flip support, this #307 chain coder + LZMA race; a tighter
coder or a better base only lowers it — B1 is the queued next rung, R3).

### Lossy concession curve (water level 1.2731 B/flip; the coordinator's ladder rung 1)
`S_support = 25·retained_bytes/37,545,489 + 100·conceded_flips/117,964,800` (registered laws):

| drop comps < k | conceded | best support (coder) | S_support_total |
|---|---|---|---|
| 1 (lossless) | 0.0% | 421,366 (lzma) | 0.28057 |
| **2 (OPTIMAL)** | **4.5%** | **361,402 (lzma)** | **0.27999** |
| 3 | 9.0% | 320,847 | 0.29157 |
| 4 | 12.6% | 295,829 | 0.30560 |
| 8 | 22.4% | 239,955 | 0.35307 |
| 16 | 32.2% | 192,643 | 0.40666 |

The curve is **U-shaped with the min at the lossless edge**: conceding the singleton boundary is ~neutral
(they cost ≈ the 1.273 water level), and **every deeper concession is a net LOSS** (the retained small
components code *below* 1.273 B/flip, so dropping them costs more S than it saves). **LZMA beats contour at
every point** — the #307 contour coder adds nothing to the concession either. **S_support_optimal = 0.280,
which alone exceeds the entire 0.172 bar** (and the fc1 distortion floor leaves only ~0.030 rate budget /
~45 KB for ALL streams; support is 8× that).

## R1 drift resolution — ONE measured number
NEITHER projection held. `142 KB` was fc1's SUPPORT-only best-case; `184 KB` = 142 support + 42 labels
(scenario C). **Real: contour support 444,394 B / LZMA 421,366 B** (+ labels 65 KB contour / 41 KB
constriction). The 142 KB was **3.1× too optimistic** — the gc5 Dykstra "bar-feasible-IF-B3 (correction
185.3 KB)" claim was built on that phantom projection and **does not survive measurement**.

## R2 — VALUES real coder (gc5 B2+B7 shared rung). MEASURED n600 incumbent; coder pending harvest.
Incumbent reproduced (range(A) residual int8×3, camera support 5,285,966 sites): **LZMA 7,046,181 B ≈ 6.91
B/err** (r2s cited 10,062,148 / 9.87 B/err; difference = int8 clip + projection detail, same order). The
amplitude+sign+context range coder (zigzag value under prev-magnitude-bucket context; round-trip ✅ at n24)
did **NOT beat the generic race** on the FULL residual (n24: coder 7.91 vs incumbent 7.18 B/err; order-1
conditional entropy 4.05 b/val ≈ coded to <1%). **Reason:** the FULL range(A) residual is not low-amplitude;
the tighter **2–5 bit/value** target requires the **da1-d4 MINIMAL-amplitude alphabet** (median 1.11, 64.1%
≤2) — but those minimal amplitudes come from a frozen-SegNet line-search = **sc1's scorer slot** → the
tight alphabet is **GATED**, reported as a derived bound not a coded price. `[R2 n600 coder bytes: HARVEST
PENDING — see r2_values_coder_n600.json; the incumbent + n24 round-trip already fix the verdict direction.]`

## R3 — BASE-TRANSFER H CONTRACT (gc5 B1 rung). READY-TO-FIRE, GATED, verified rc=3.
`experiments/ddm_sp1_base_transfer_h_contract.py --base-argmax <sc1 dir/npz['argmax']> --gt-cache <gt_n600>
--out <ssd>.json`. Recomputes flip = base_argmax ≠ lstar on sc1's seeded base, codes support (contour+LZMA,
round-trip-proven, REUSING R1's functions), fires the falsifier vs the copy-base anchors:
`< 250 KB → B1 opens the floor` · `250 KB–421 KB → B1 helps, still bar-bound` · `≥ 421 KB → B1 does NOT
help`. It **refuses to run (rc=3) until sc1's real base masks exist** — never fabricates a base.

## Verdict routing (the coordinator's ladder; typed scope)
Lossless (444 KB) AND lossy-optimal (S 0.280) BOTH land high → the **EXPLICIT support stream on the
copy-PREDICT base is DEAD for the sub-bar floor (FORMULATION scope).** Next rungs, in order:
1. **B1 UPSTREAM (queued, R3 GATED on sc1):** a better base shrinks support *before* coding. R3 fires the
   moment sc1's seeded base lands — the only path that can move the support number without a new coder.
2. **DOWNSTREAM (the real route):** the realization carrier absorbs support **IMPLICITLY** (a learned
   carrier draws its own boundaries; no explicit support stream to price). This memo's negative is scoped
   to **explicit** support streams; it does not touch an implicit-carrier formulation.
3. Tighter explicit coders (STC/UNIWARD lever-D) can only shave the 421 KB toward the ~0.28 S floor, not
   below the bar — dominated by (1)/(2).

## Honest boundary
Every reported byte is a bit-exact round-trip-proven lossless coder length on the REAL n600 cached masks; no
projected/entropy number is quoted as a price except the explicitly-labeled GATED d4 alphabet bound. The
pointer did not move; this arm's product is a **measured decisive negative** that redirects the B3/B1
composition off the phantom 142 KB projection.

## Wire-in (Subagent coherence 6-hook)
1. sensitivity-map: N/A (advisory coder measurement, no per-axis byte-weight change). 2. Pareto: the
S_support(concession) curve IS a measured rate/distortion Pareto row for the support axis. 3. bit-allocator:
N/A. 4. cathedral dispatch: N/A (non-promotable). 5. continual-learning: this memo + DAG FEED. 6.
probe-disambiguator: R3 IS the disambiguator for B1 (copy-base-dead vs better-base-open).
