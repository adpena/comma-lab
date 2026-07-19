# ARM G — #425 phase-carrier STORE leg BUILT + n600 MEASURED (curve-domain per-dash δ(s) codec)

**UTC** 2026-07-17 · **branch** `p0_build_phase_carrier_425_20260717` · **authority**
`[macOS-CPU advisory]` label-space · `score_claim=false; promotable=false` · **pointer 0.19108 UNMOVED — MEANS.**
Charter: `.omx/tmp/build_wave_20260717/ARM_G_425.md` (lane-crux-3's unbuilt rate half; SPEC_v10 §13.9 "the
#425 STORE leg remains the unbuilt half" — now BUILT + measured end-to-end).

## STORES CONSULTED (proactive recall)
CLAUDE.md · SPEC_v10 §13 (via the SPEC branch) · `temporal_advection_stratified_20260715.md` (the δ(s)
jitter prior + the named reactivation) · `lane_channel_deep_refactorization_20260716.md` (§3 dash
geometry, §4 churn 9.4/step upper bound + blink-back OPEN, §5 anchor budget 0.9–1.8 KB) ·
`necessity_solver_inverse_factorization_20260715.md` (K-ladder, DP-vertices brotli precedent) ·
`p0_425_phase_carrier_byte_close_row_20260716.md` (#359 raster row 10,682 B) · Arm B's
`phase_primitives.py` (read via `git show p0_build_forces_triggers_20260717:src/tac/boundary_math/phase_primitives.py`
— `event_fallback_ref_and_weight_numpy` semantics mirrored at island level; `cross_scored_frame_xi_interp`
IMPORTED, not duplicated) · `curve_relative_offset_coder.py` (#386 — REAL, CONTINUED not superseded: it is
the WITHIN-FRAME spatial chart; this build is its TEMPORAL complement) · `phase_residual_carrier.py` (#359
raster sister) · `spec_c2_surgical_20260716.py` (`phase_residual_carrier_425` slot precedent).

## What was BUILT (commits on this branch)
1. `src/tac/boundary_math/dash_phase_carrier.py` (+18 tests) — commit `1b09f7ac81`, telemetry+calibration
   in `30e262538d`. Encoder: 8-conn interior Lane islands → ξ-advected world TRACKS (greedy deterministic
   matching, radius 6 px) → per matched dash a curve-relative residual **(δs, δn)** (rotated into the
   dash's anchor tilt frame; s = the along-dash PHASE coordinate), quantized q=1 px, coded with a
   **prior-derived canonical Huffman** over {0,±1,±2,ESC} whose probabilities ARE the measured jitter
   prior (code lengths ship IN the header — no table in decoder code). **Explicit events**: 1 alive-bit
   per live track (death), full anchors (u16² + tilt4b + varint area) for births, and **REBIRTH codes**
   (varint index into the ξ-advected dormant pool, horizon 30) — the world-frame amortization. Closed
   loop: encoder predicts from fp16-roundtripped ξ and decoded states (no drift). NO-FAKE: every encode
   runs the FULL decoder and refuses on any mismatch; the bit-reader refuses ≥1 unread byte AND nonzero
   padding; decode refuses trailing section bytes.
2. `tools/levelset_byte_close_and_eval.py` `--dash-phase-carrier` (+6 tests) — commit `47d3f3a2b8`:
   selectable section mirroring the #359 staging (byte-identical when off; bytes measured; d_seg
   OWED-labelled). `--dash-phase-no-xi` = the L68 composition mode (ξ omitted; decoder takes the banked dxi).
3. `tools/measure_dash_phase_carrier_n600.py` + the n600 row — commit `30e262538d` (results JSON local at
   `experiments/results/dash_phase_carrier_n600_20260717/results.json`, ignored dir; all numbers below).
4. Canonical equation `dash_phase_carrier_rate_blinkback_prior_divergence_v1` REGISTERED (+7 tests) —
   commit (equation): `src/tac/canonical_equations/dash_phase_carrier_rate_law_20260717.py`.

**Pre-registered BEFORE measuring** (per charter): canonical Huffman on the site prior
{p0=.404, p±1=.1595, p±2=.0375, pESC=.202} ⇒ E[len]=2.267 bits/component ⇒ **4.53 bits/dash** expected.

## MEASURED — the rate table (n600, frozen SegNet argmax `gt_n600.npz`, deterministic)

| quantity | value | label |
|---|---|---|
| section total / excl-ξ | 37,158 B / **29,958 B** (49.9 B/frame) | MEASURED |
| ξ block (fp16, P×6) | 7,200 B — **0 marginal** in composition (L68 dxi banked; `include_xi=False` mode built) | MEASURED/cited |
| vs lane-only per-frame naive (packbits+zlib9, measured here) | 338,523 B → **11.3× UNDER** | MEASURED |
| vs naive per-frame anchor stream (11,446 obs × ~5.5 B) | 62,953 B → 2.1× under | MEASURED |
| vs all-class partition naive (advection memo) | 601,800 B (cited) | cited |
| vs world-anchor budget band 0.9–1.8 KB (lane memo §5, DERIVED there) | **16.6× OVER** — the gap IS the event stream | MEASURED vs DERIVED |
| bit breakdown | alive 11,427 · δ 94,904 · birth 72,916 · rebirth 53,368 bits | MEASURED |
| δ-stream: prior code vs zlib9 on same ints | 11,863 vs 9,498 B — **iid prior code LOSES by 20%** | MEASURED |
| bits/matched-dash: pre-registered vs realized | 4.53 vs **9.58** (ESC 26.5%) | MEASURED |
| tracking (P=600) | 19.08 obs/frame · 1,544 tracks · 4,270 matches · 5,632 rebirths · 1,522 post-f0 births · 7,157 deaths | MEASURED |
| **blink-back fraction** (lane memo §4 OPEN item — first measurement) | **0.787** | **MEASURED** |
| transport coverage after frame 0 (live+dormant) | 86.7% (13.3% genuinely new anchors) | MEASURED |
| pose→ξ calibration | s_t=−0.00322, s_r=0, pitch=−0.01 (advection-memo fit). Raw s_t=1 (the #359 convention) mis-advects: n20 coverage 52%→82%, section −24% | MEASURED (fit cited) |

## MEASURED — recovery (LABEL-SPACE; honest scope)
Through-R d_seg on the c2 ep725 EMA is **NOT run** — it needs trainer render plumbing against the SACRED
live-run dir under the 60 GiB trainer + 8 GB arm budget. Per charter fallback, recovery is measured at the
**partition/argmax level** and labelled so:

| arm | centroid offset d0 / ≤1 / ≤2 / >2 px | mean px | lane-layer raster XOR rate* |
|---|---|---|---|
| transport-only (ξ-advect, no δ) | 4.5% / 13.7% / 35.7% / **64.3%** | 2.81 | 1.173 |
| persist (no advect) | — | — | 1.129 |
| **phase-correct (decoded)** | **79.1% / 100% / 100% / 0%** | **0.38** | **0.749** |

\* XOR px / GT-scope px on matched tracks, shape-persistence approximation (prev observed island pixels
placed at the target centroid; GT reference = the matched observations' own pixels). Rates >1 for
persist/transport = a 3 px error on a median-12 px island near-doubles the XOR. The δ correction buys back
**~34% of the lane-layer XOR vs persistence**; transport ALONE is WORSE than persist (consistent with the
advection memo's ξ-marginal ≈ 0 — ξ's value here is the WORLD-FRAME IDENTITY for events, not the warp).
Coverage: phase-correct = 100% by construction (births anchor-coded); transport-only leaves 13.3% uncovered
— the store-side twin of Arm B's 26.3% straddle-coverage gap, same event concept at island granularity.

## The two measured DIVERGENCES (the equation's payload)
1. **Site prior ↛ dash centroids.** The separatrix-site jitter prior (d0 40.4 / ≤1 72.3 / ≤2 79.8 / >2
   20.2%) does NOT describe per-dash centroid offsets (>2 px 64.3%): centroids of small churning islands
   move more than boundary sites (partial-area change shifts the mean; the global 3-scalar ξ fit adds
   residual). Consequence: 9.58 realized vs 4.53 pre-registered bits/dash, and the iid prior code loses
   to zlib9. The code table is a HEADER (recalibration = a header change, not a decoder change).
2. **The 0.9–1.8 KB budget assumed FREE visibility.** The lane memo's world-anchor budget prices anchors
   only, with the persistence-class visibility generator FREE (rule-118 code). Measured: the event stream
   (alive+birth+rebirth ≈ 137.7k bits ≈ 17.2 KB) + δ (11.9 KB) is the cost. Blink-back 0.787 says the
   world-anchor READING is right (few genuinely new anchors); the missing piece is the deterministic
   visibility/persistence generator that would delete the alive-bit + rebirth streams.

## Rule-118 / NO-FAKE boundary (adversarially checked)
- **COUNTED**: header (incl. the 6 code lengths — the prior-derived table ships as seed, NOT decoder
  code), event stream, anchors, δ symbols + ESC varints, (optional) fp16 ξ.
- **FREE**: ξ point-advection homography (generic geometry, same `GroundHomographyGeom.eon` status as the
  shipped pose carrier), canonical Huffman decoder (generic given header lengths), the downstream dash
  rasterizer. Encoder-side extraction/matching/calibration never run at decode; the calibration scalars
  (s_t/s_r/pitch) touch only the encoder — the STORED ξ is post-calibration, so **no video-derived
  constant hides in decoder code**.
- **Every-seed-byte consumption**: `_BitReader.assert_fully_consumed()` refuses ≥8 unread bits and any
  nonzero padding bit; decode refuses trailing section bytes; the residual cursor must land exactly.
  This IS the no-op-detector logic at the codec level, run on EVERY encode via the mandatory full-decode
  bit-identity self-check (`DashPhaseError` on any mismatch). Tests cover corrupt magic, trailing bytes,
  partial consumption.

## Composition / antagonism vs existing levers
- **#359 raster phase carrier (10,682 B)**: COMPLEMENTARY, different objects — #359 polishes sub-pixel tie
  phase GIVEN decoder-derivable geometry; #425 carries the dash GEOMETRY + EVENTS themselves (the birth
  structure the witness cannot render). **Sister-audit flag (cure adjacency): #359 uses s_t=1.0/s_r=1.0
  pose→ξ and measured ξ_amort 1.041 (transport NOT helping) — the same calibration mis-scale measured
  here (coverage 52%→87% on fixing it) plausibly explains it; re-measure #359 with the fitted calibration.**
- **Arm B event-fallback (train side)**: same event definition (no valid transported reference ⇒ birth) at
  island vs straddle granularity; the carrier's birth anchors are the STORE twin of the training force —
  lane-crux-3 now has BOTH legs built. No antagonism (different phases: train vs archive).
- **L68 dxi (7.2 KB)**: compose via `include_xi=False` / `--dash-phase-no-xi` → ξ 0 marginal.
- **#386 curve_relative_offset_coder**: CONTINUED — it codes within-frame shape residual n(s); #425 codes
  cross-frame phase δ(s). A composed carrier (anchor + phase + shape) is the natural v8 P7 shape.
- **Lane render band / #287 comb**: comb REFUTED (lane memo §5); the per-dash anchors here are the
  anchor-field replacement that memo called for.

## ROUND-1 ADVERSARIAL SELF-REVIEW (own attack)
- **"Is the entropy model overfit to the prior?"** — measured the OPPOSITE failure: the prior is
  mis-LEVELED (site vs dash), not overfit; realized 2.1× the pre-registered bits and iid loses to zlib9.
  Registered as the equation's headline residual, with 3 named reactivation levers (dash-level measured
  prior · per-track context coding · free visibility generator).
- **"Does the receiver actually consume every seed byte?"** — yes, structurally refused otherwise (above).
  NOT yet proven: consumption inside a composed `archive.zip` through `inflate.py` — the section is
  byte-close-SELECTABLE (same NO-FAKE staging as #359); the `_io_pack` grammar fold + through-R A/B is the
  named next step, d_seg stays OWED until then.
- **"Any video-derived table in generator code?"** — no (boundary above); the one borderline surface is
  the encoder-side calibration defaults in `DashPhaseConfig` (video-fitted scalars) — encoder-only, but
  flagged for the provenance ladder (they should become a LawRef citing the advection-memo fit).
- **Weakest measured link**: the lane-layer XOR substitution is a SHAPE-PERSISTENCE approximation on
  matched tracks only (births excluded from the raster metric; their coverage is reported separately);
  XOR>1 baselines show the metric is harsh on 12 px islands. It ranks arms; it is NOT a d_seg.
- **Not measured**: per-component (δs vs δn) histograms; Hungarian vs greedy matching; tilt/area drift
  over track life; dormant-horizon sweep; the 1-frame gap; through-R d_seg (OWED).
- **verdict_scope: FORMULATION** on every negative here (iid-code loss, budget overshoot) — the family
  (curve-domain phase store) is ALIVE and measured 11.3× better than lane-naive at first build.

## Triality
- **equations**: `dash_phase_carrier_rate_blinkback_prior_divergence_v1` REGISTERED (live registry row,
  agent=claude subagent=arm_g_425).
- **DSL**: the carrier is an ARCHIVE-SHAPE section, byte-close-selectable (`--dash-phase-carrier`),
  mirroring the #359 precedent (`spec_c2_surgical.C2_SOLVE_SEED_DROP_SLOTS['phase_residual_carrier_425']`);
  never-invent-flags forbids a trainer Lever with no trainer flag — no trainer surface exists or is needed.
- **DAG** (ready-to-paste FEED row for the post-v9c2 boundary merge, to avoid cross-branch DAG conflicts):

> **FEED-425-store (2026-07-17, Arm G):** #425 STORE leg BUILT+MEASURED n600 — curve-domain per-dash δ(s)
> codec (`dash_phase_carrier.py`, byte-close `--dash-phase-carrier`): 29,958 B excl-ξ = 11.3× under lane
> naive (raster-transport 0.71<1 REACTIVATION CONFIRMED at object level) but 16.6× over the world-anchor
> budget (gap = event stream; free visibility generator owed). Blink-back **0.787 MEASURED** (lane memo §4
> OPEN closed). Site jitter prior does NOT transfer to dash centroids (gt2 64% vs 20%; 9.58 vs 4.53
> bits/dash; iid prior code loses to zlib9 by 20% → dash-level prior + context owed). Label-space recovery:
> phase-correct XOR 0.749 vs persist 1.129 (−34%); transport-only WORSE than persist (ξ value = world
> identity, not warp). Sister flag: #359's ξ_amort 1.041 plausibly the s_t=1 calibration mis-scale
> (measured cure: fitted s_t=−0.00322 → coverage 52→87%). Equation
> `dash_phase_carrier_rate_blinkback_prior_divergence_v1`. Through-R d_seg OWED. Pointer 0.19108 UNMOVED.

**Pointer 0.19108 UNMOVED — this is MEANS (a measured carrier row + two measured divergences), not a score.**
