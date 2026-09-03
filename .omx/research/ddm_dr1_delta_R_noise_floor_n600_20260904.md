---
title: "delta_R is 0.021881818771362305 at n600, not 0.019590163230895963 — the pre-registered falsifier FIRED at +11.70%, the n96 prefix understated the R noise floor, every m_safe derived from it is 11.7% too low, and the per-class spread is 2.024x (Lane 0.012856 vs Undrivable 0.026026) so one scalar cap mis-scales by up to 1.70x"
arm: ddm_dr1
charter: .omx/research/charters/ddm_dr1_delta_R_noise_floor_n600_20260903.md
charter_commit: 4870d475c
utc: 2026-09-03T22:11:45Z
verdict_scope: "[PyAV frames . macOS-CPU advisory . frozen CPU-torch SegNet . NON-PROMOTABLE]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_dr1 — the R-chain margin noise floor at n600

## The finding, first

**δ_R = 0.021881818771362305** (MEASURED, n600, all 600 pairs, band 1.0, frozen CPU-torch SegNet).

The pre-registered falsifier **FIRED**. The charter predicted the n600 value would land within ±10% of
the n96 value 0.019590163230895963, i.e. inside [0.017631147, 0.021549180]. It landed at **+11.698%**,
ratio **1.1169799104507785** — above the band. The 96-frame contiguous prefix **understated** the noise
floor. This is [[m88]] firing again: a prefix of a skewed population is a different population.

The consequence the charter named is now live: **every `m_safe` derived from the n96 constant is 11.7%
too low.** The DERIVED headroom is unchanged at 2 (`ceil(0.03887428045272823 / 0.021881818771362305)
= ceil(1.7765) = 2`), so:

| quantity | n96 (prior) | n600 (this arm) | ratio |
|---|---:|---:|---:|
| **δ_R** | 0.019590163230895963 | **0.021881818771362305** | **1.116980** |
| DERIVED headroom | 2.0 | 2.0 | 1.000 |
| **DERIVED m_safe** | 0.039180326461791926 | **0.04376363754272461** | **1.116980** |
| full-R annulus p95 | 0.03712034225463867 | 0.03887428045272823 | 1.047 |
| annulus area frac | 0.025631957583957247 | 0.02670194837782118 | 1.042 |
| all-pixel p95 | 0.038173675537109375 | 0.0383458137512207 | **1.0045** |
| all-pixel mean | 0.01356075331568718 | 0.013560148887336254 | **0.99996** |

Direction matters. `m_safe` is a satisficing TARGET: push a boundary pixel's margin up to it, then
stop. A target that is too low stops the gradient too early, so pixels declared R-safe at 0.039180 are
still flippable by the real uint8 noise. **The n96 constant was anti-conservative, not merely stale.**

## The positive control — the +11.70% is 100% cohort, 0% instrument

I recomputed δ_R over frames 0–95 **from this run's own retained payload**:

```
prefix[0:96] of THIS run   delta_R = 0.019590163230896
independent n96 artifact   delta_R = 0.019590163230896   (reports/delta_R_noise_floor.json, 2026-07-12)
reproduction rel-diff              = +0.000e+00
```

Bit-identical, across two months, two cohorts, two thread counts and two runs. So the measurement is
deterministic, `gt_n96.npz` is exactly the first 96 frames of `gt_n600.npz`, and **the entire deviation
is attributable to which frames were measured** — there is no code, environment, or instrument
contribution to explain away.

The prefix is unrepresentative in the expected direction, and the hot region sits just past it:

| frame range | mean per-frame annulus p95 | ratio vs all-600 |
|---|---:|---:|
| [0:96] (the n96 prefix) | 0.019845 | **0.9045** |
| [96:300] | 0.023601 | 1.0757 |
| [300:600] | 0.021482 | 0.9791 |
| [0:600] | 0.021941 | 1.0000 |

Pooled δ_R on the complement [96:600] = **0.022261619567871**. Per-frame p95 ranges 0.013300 to
0.091214 — a 6.86× spread across frames, which is why 96 frames cannot pin a population p95.

## The sharpest structural finding — the bias is ANNULUS-specific, not a frame-statistics effect

Over **all** pixels the n96 prefix is essentially unbiased: p95 moves +0.45%, mean moves −0.004%. Over
the **annulus** it is biased +11.70%. The prefix does not have different global image statistics; it
has a different **boundary** population. Any constant measured on a boundary-restricted set from a
contiguous prefix inherits this, and a global-pixel sanity check will not detect it. That is a reusable
detector for the [[m88]] genus, and it is why `annulus_area_frac` also moved (+4.17%) while the global
statistics did not.

## δ_R is robust to the annulus definition — so the band choice is not the confound

Computed free from the same pass (nested subsets of the same annulus):

| annulus band | pixels | δ_R |
|---|---:|---:|
| \|GT margin\| < 1.00 (the fh1 form) | 3,149,890 | 0.021881819 |
| \|GT margin\| < 0.50 | 1,630,790 | 0.020604742 |
| \|GT margin\| < 0.25 | 827,002 | 0.020070553 |

Narrowing the band 4× moves δ_R only **−8.28%**. Even the narrowest n600 band (0.020071) sits **+2.45%
above** the n96 band-1.0 value. The falsifier verdict therefore survives every annulus definition
measured — it cannot be explained by the band. This also retires the charter's optional "second run at
band 0.5": it cost 0 s inside the same pass, and it is answered.

## Per-class δ_R — a 2.024× spread, in the units the cap actually uses

MEASURED, n600, pooled over the annulus, `lstars` in the canonical comma10k order:

| class | annulus px | p50 | **δ_R_c (p95)** | DERIVED m_safe_c (headroom 2) | δ_R_c / δ_R_global |
|---|---:|---:|---:|---:|---:|
| Road | 1,542,822 | 0.005219 | 0.022577 | 0.045154 | 1.032 |
| **Lane** | 513,240 | 0.003103 | **0.012856** | **0.025712** | **0.588** |
| **Undrivable** | 494,674 | 0.006790 | **0.026026** | **0.052052** | **1.189** |
| Movable | 258,344 | 0.004950 | 0.018433 | 0.036866 | 0.842 |
| MyCar | 340,810 | 0.006953 | 0.023853 | 0.047706 | 1.090 |

Spread **0.026026 / 0.012856 = 2.024×**. Two consequences, both DERIVED from the table:

1. **Lane is the cheapest class to make R-safe, by 1.702×.** A single global cap at 0.043764 keeps
   pushing Lane margins to 1.70× past the level Lane actually needs. Lane is 0.59% of area but 33.56%
   of the model bits and ~90.1% of the rate demand ([[m131]]) — so the waste lands exactly on the
   class the rate corner cannot afford.
2. **Undrivable is under-protected by the global cap by 1.189×.** Sites the global cap calls safe are
   still inside Undrivable's own R-noise band.

This is the same shape as vr1 row 4 (per-class-pair flipdist spread **2.185×** from the exact rank-4
head), arrived at by a fully independent instrument — and, unlike row 4's feature-space quantity, δ_R_c
is already in **logit-margin units**, the units `m_safe` and `τ` live in. The two rows now agree that a
single scalar mis-scales this axis by roughly 2×.

## Verdict scope — stated, not laundered

`[PyAV frames · macOS-CPU advisory · frozen CPU-torch SegNet · NON-PROMOTABLE]`. This sets a **lever
constant**. It is not a score, it is not an axis, and it moves no pointer.

**The GT lineage caveat, and its measured size.** VERIFIED AT SOURCE: `gt_cache_dali.pt` holds only
`pose (600,6)` and `seg (600,384,512)` — **no frames**. A DALI-lineage δ_R is therefore not measurable
from any cached artifact; PyAV frames are the only option, and δ_R uses the frames plus SegNet's own
margins, never the `lstars` table. I bounded the exposure rather than asserting it away (MEASURED):

- 20,671 PyAV↔DALI argmax disagreements over 117,964,800 pixels = **0.017523%**;
- **100.00%** of those 20,671 sites lie **inside** the annulus (they are exactly the low-margin sites,
  as the mechanism predicts);
- they are **0.6562%** of the 3,149,890 annulus pixels.

So the annulus *definition* can differ between lineages by at most 0.66% of its pixels — **15× below**
the ±10% falsifier band that just fired at +11.70%. The lineage cannot account for this result.

**Instrument caveat.** δ_R is measured on `top1 − top2`; the consumer hinge uses the signed margin
`logit[GT] − max_{c≠GT} logit[c]`. hg1 MEASURED these two agreeing to ~1% on the same statistics (GT is
the runner-up on 98.018% of flips), so the transfer is TRANSFERRED-with-a-1%-caveat, well inside the
band. It is not re-derived here.

## The wiring gap — say it plainly

The canonical law reads the **n96** file. `margin_band_satisficing_threshold_20260712.DELTA_R_ARTIFACT
= "reports/delta_R_noise_floor.json"`, and its class-4 WAIVER fallbacks
(`FALLBACK_DELTA_R`, `FALLBACK_FULL_R_ANNULUS_P95`, `FALLBACK_N_FRAMES = 96`) are the n96 values. My
n600 number lands in `reports/delta_R_noise_floor_n600.json`. **Until someone repoints it, every live
consumer still resolves `m_safe = 0.039180326461791926`** — the value this arm just measured to be
11.7% too low. That is [[m107]] (split banks) and [[m56]] (unwired-but-built) waiting to happen.

I did **not** repoint it. My charter does not authorize changing a live lever default, the QBR1 burn is
resident, and the fallbacks would go inconsistent with the pointer in the same edit. The repoint is
named below as an owed step, with its exact shape. I verified it is mechanically trivial:
`resolve_margin_band_threshold(artifact_path=...)` already accepts an explicit path and already returns
`n_frames`, so a consumer can assert its own cohort size today.

## MEASURED / DERIVED / TRANSFERRED ledger

- MEASURED: δ_R and every quantile in `reports/delta_R_noise_floor_n600.json`; the per-class and
  sub-band tables; the per-frame trajectory; the prefix positive control; the 20,671/100.00%/0.6562%
  lineage numbers; wall 1003.18 s; peak RSS 6466.36 MiB; exit 0.
- DERIVED: headroom 2 (`minimum_integer_headroom`); `m_safe` global and per class (`headroom · δ_R`,
  the registered law); all ratios in the tables above.
- TRANSFERRED: hg1's ~1% signed-margin↔top1−top2 agreement; [[m131]] Lane bit/rate shares; vr1 row 4's
  2.185× flipdist spread.
- ASSUMED: nothing load-bearing. The per-class `m_safe_c` applies the **global** DERIVED headroom to
  each class's own δ_R_c — headroom is a policy factor, δ_R is the measurement. A per-class headroom is
  not derived here and is not claimed.

## Payload (ALWAYS KEEP THE PAYLOAD)

`/Volumes/APDataStore/pact/ddm_dr1_delta_R_n600/`

| file | bytes | sha256 |
|---|---:|---|
| `retain/m0_no_uint8.npy` (600,384,512) fp32 | 471,859,328 | `4cbf904b8d762579ecf0bde0f6536bc4ad51f880ae396aa0bd0bad3d7ce2727f` |
| `retain/m1_with_uint8.npy` (600,384,512) fp32 | 471,859,328 | `2e5b1011b13a0c9bdfdf3e9a4204a633ab7fc1cab38c6a91d3242187267447cb` |
| `delta_R_receipts_n600.json` (600 per-frame rows + pooled tables) | 485,548 | — |
| `run/` launch manifest, run.log, safe_run status receipt | — | — |

Input `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` sha256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.

The payload is COMPLETE, not decorative: δ_R replays from `m0`/`m1` plus the cache margins to
**0.021881818771362305** — the headline, bit-for-bit. Any band, any class, any frame subset is
recomputable without re-running the 16.7-minute pass. That is how the prefix control above was
produced.

## Apparatus

- Tool **re-run, never rebuilt** (SPEC_v75 §8B): `tools/measure_delta_R_noise_floor.py --n 600
  --gt-npz .../gt_n600.npz --band 1.0 --threads 4`. Added flags are default-off and RECEIPT-ONLY:
  `--receipts-out`, `--retain-dir`, `--threads`. Proof they do not touch the measurement: the `--out`
  JSON is byte-identical with and without them (verified at n=8, and pinned by a test).
- Registered anchor `margin_band_delta_r_noise_floor_n600_20260904` on
  `margin_band_satisficing_threshold_v1` through
  `append_empirical_anchor_to_equation_with_posterior_update`. No hand-edit; the helper did not refuse.
  `EmpiricalAnchor.residual = 0.0` **deliberately**: the law `m_safe = headroom · δ_R` is exact
  arithmetic, so its residual is zero at every anchor. The n96→n600 change is a change in the measured
  INPUT, not a law error; putting it in `residual` would teach the posterior that a multiplication is
  inaccurate. It is recorded inside `empirical_output.prefix_bias_check` instead.
- Governed launch: `tools/launch_detached_process.py`, nice 10, 4 threads, wall cap 5400 s.
  MEASURED wall **1003.18 s**, peak RSS **6466.36 MiB**, exit 0.

## GESTALT-DELTA — what row 6 may now race with

Row 6 (`MarginBandSatisficing`, FOLD-NOW) may now race a **measured-at-n600** cap:

- **Global:** `m_safe = 0.04376363754272461` (headroom 2, DERIVED). Not 0.039180326461791926.
- **Per class:** Lane 0.025712 · Movable 0.036866 · Road 0.045154 · MyCar 0.047706 · Undrivable
  0.052052. The 2.024× spread makes a per-class cap a *derivable* variant rather than a guess, and it
  composes with vr1 row 4's per-class-pair τ — both are the same "one scalar mis-scales by ~2×" defect
  seen from two instruments.
- **Robustness banked:** δ_R moves only −8.28% across a 4× band narrowing, so the cap is not sensitive
  to the annulus definition a racer picks.

Two downstream rows are **re-graded, not refuted** (their qualitative conclusions survive; their exact
numbers were computed at the n96 m_safe and are owed a cheap re-measure on already-retained fields):

- **hg1 §5**: "the trainer default `--margin-target 1.0` sits 25.5× above the floor" → the multiple is
  **22.85×** at the n600 floor. The finding stands; the number moves.
- **nx1 Finding 1** table row "hard hinge at m_safe = 0.03918 → 0.0384% active, 75.114% grad mass on
  flips, 24.89% waste": raising the target 11.7% raises the active fraction and the already-correct
  share, so the waste is somewhat **higher** than 24.89% (DERIVED direction, magnitude UNMEASURED). The
  claim that the derived scale beats the shipped τ by ~3× on flip-gradient share is not threatened.

Honest limit: this arm measured a constant. It did not train, byte-close, or score anything.
`treatment_delta_s` for row 6 remains **UNMEASURED**, and `headroom_3_status` remains
**OPEN_UNMEASURED** — the headroom-2-vs-3 A/B is still owed and is now cheaper to state, since headroom
2 covers the n600 full-R annulus p95 with 12.6% slack (0.043764 vs 0.038874) where it covered the n96
one with only 5.5%.

## NEXT_IF_RESUMED — every row carries a disposition, an owner and a fire condition ([[m113]])

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **Repoint the law consistently** | **QUEUED-WITH-FIRE-ORDER, fires FIRST** | MAIN (operator call) | fires at the next burn boundary — it changes a live lever default while QBR1 is resident |
| 2 | **Re-measure hg1/nx1 gradient mass at the n600 m_safe** | **QUEUED-WITH-FIRE-ORDER, fires after #1** | MAIN to assign; $0 local CPU | fires once #1 lands, so the re-measure reads the same constant the trainer will |
| 3 | **Race per-class `m_safe_c` vs the global cap** | **FOLDED into row 6's race** | row 6 racer, jointly with vr1 row 4 | fires when row 6 is raced; do not fire standalone |
| 4 | **Re-derive other prefix-measured annulus constants** | **QUEUED, no fire order — needs a census first** | unowned; MAIN to assign or close | fires only if a census finds a second annulus-restricted n96 constant in a live consumer |

1. **Repoint the law, in one commit, consistently.** Set `DELTA_R_ARTIFACT =
   "reports/delta_R_noise_floor_n600.json"` **and** update `FALLBACK_DELTA_R`,
   `FALLBACK_FULL_R_ANNULUS_P95`, `FALLBACK_N_FRAMES` to the n600 values in the same edit — a repoint
   that leaves n96 fallbacks behind is a silent n96 revival whenever the artifact is missing. Not done
   here: it changes a live lever default while the QBR1 burn is resident, which is an operator call.
   Until it lands, every consumer resolves the value this arm measured to be 11.7% low.
2. **Re-measure hg1/nx1's gradient-mass table at m_safe = 0.043764.** $0 — nx1 already retained the
   margin field at `/Volumes/APDataStore/pact/ddm_hg1_ring0_margin_hinge_20260816/HG1_MARGIN_FIELD_n96.npy`.
   Note that field is itself n96; the honest version re-runs it on an n600 field.
3. **Race per-class `m_safe_c` against the global cap** as row 6's variant, jointly with vr1 row 4's
   per-class-pair τ. The prediction to pre-register: the gain concentrates on Lane (1.702× of wasted
   push freed) and shows up as rate, not as d_seg.
4. **Re-derive any other constant measured on a boundary-restricted n96 prefix.** This arm's sharpest
   transferable result is that a global-pixel sanity check does NOT detect annulus-restricted prefix
   bias (+0.45% global vs +11.70% annulus). Every prefix-measured annulus constant in the corpus is
   suspect by the same mechanism. I did not run that census; naming it without owning it would be the
   deferral-scatter this repo already extincted ([[m36]]), so it is explicitly UNOWNED above.

---

Pointer honesty: this arm measured a lever constant on an advisory axis. It trained nothing, closed no
bytes, and could not move the frontier.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — UNMOVED.

## ADDENDUM (MAIN, 2026-09-04) — NEXT #1 FIRED: the law is repointed, consistently, in one commit

Safe now, not "at the next burn boundary": the QBR1 burn executes from the sealed source snapshot on Vertigo
(`sealed_source_106d0dd0_v2`), never from the working tree, and its trainer consumes no satisficing cap
(0 references in the sealed `ddm_qbt1_qbflow_trainer.py`). So the repoint cannot touch the running discriminator.
Changed together: `DELTA_R_ARTIFACT` → `reports/delta_R_noise_floor_n600.json` with `FALLBACK_DELTA_R`
0.021881818771362305 / `FALLBACK_FULL_R_ANNULUS_P95` 0.03887428045272823 / `FALLBACK_N_FRAMES` 600 (headroom stays
DERIVED 2 → m_safe 0.04376363754272461); the n96 path kept as `DELTA_R_ARTIFACT_N96_HISTORICAL`; the builder's anchor
is now `margin_band_delta_r_noise_floor_n600_20260904` and names what it supersedes; `hg1_ring0_margin_hinge_levers`
repointed; `tools/measure_joint_seg_pose_rate.py` and `tools/dual_metric_readback.py` now RESOLVE m_safe through the
law instead of carrying 0.039180326461791926 as a literal ([[m107]] split-banks cure); three pinned tests and two
docstrings updated. Live resolution verified: delta_r 0.021881818771362305 · headroom 2 · m_safe 0.04376363754272461 ·
n 600 · fallback False. NEXT #2 (re-measure hg1/nx1 gradient mass at the n600 m_safe) is now unblocked.
