# T5 CRUCIBLE — V6 GATE PROBES: P-TAU2 (f_target / fixed-point τ* knee) + P-DITHER (B19 decode-side seeded dither $0 A/B) — VERDICTS: 0.31 STANDS (knee f_target ≈ 0.862 ≈ q̂) · B19 KILL-THIS-FORM (Δd_seg = +2.1277533637e-6 ≥ 0; kill ~10σ robust to seed)

`[no-triality]`
review_status: fresh-eyes-measured(1)
axis: ALL numbers [macOS-CPU advisory] / [macOS-MLX research-signal] — NO score claims;
pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS.
Run dirs READ-ONLY (mod32cap run + the R6 packet artifacts are consumed, never mutated).

STORES CONSULTED: `ORCHESTRATION_LEDGER.md` (requirements H/J/L/P/Q/R + the NO-OPEN-GATES
sequencing rule + the v6 landing fold) · `DRAFT_OPTIMAL_STACK_v6_20260708.md` §0.3 (B19 spec +
census + M1 term) · §1.4a (fixed-point τ* convention + launch constant 0.31 + the q̂=0.85
bracket) · §5 (B19 lever spec) · §7c (pre-registered bands/kills for P-TAU2 + P-DITHER — honored
EXACTLY, no post-hoc adjustment) · `probe_tau_confirm_ep1000_20260708.md` (the m_q instrument +
the margin-field trap + the measured quantile tables this probe extends) ·
`tools/witness_tau_mq_confirm.py` + `src/tac/witness_annulus_metrics.py` (the reused instrument) ·
`recess_wave1_R1_R3_R6_20260707.md` §R6 (the parity-row byte-close artifacts: packet
`experiments/results/levelset_packet_20260708T013253Z/` + inflated/0.raw + r6_verdict_pairs.jsonl
— the undithered arm + the chunked-driver pattern reused) · the packet's own `inflate.py`
(read line-by-line; `_R` at L349 = the uint8 quantization site; pose_carrier=None verified →
`_R` fires exactly twice per `_render_pair`, (pair,frame) keying exact) ·
`experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json` (P-DZ locked-mass
geometry: far-range lane rows 176–224, horizon shadows, hood boundary) ·
`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (GT argmax + the TRUE GT-margin axis) ·
`.omx/research/t5_crucible/artifacts/tau_mq_maps/maps_{BEST_ep650,END_ep1000}.npz` (the rendered
witness argmax maps reused for P-TAU2).

## STATUS CHECKPOINT
- [x] specs + bands read; artifacts located; keying verified (pose_carrier None)
- [x] P-TAU2 instrument (knee criterion PRE-STATED below, before computing) + run — §A
- [x] P-DITHER instrument + full 600-pair dithered decode + both-arm n600 verdicts + compare — §B
- [x] verdicts + tables; tests green (27 + 7); ruff F clean

## DURABLE INSTRUMENTS (req Q — probes become toolbelt)

| surface | path |
|---|---|
| knee math (pure numpy, tested) | `src/tac/witness_annulus_metrics.py::{flip_margin_values, flip_margin_cdf, kneedle_knee, max_curvature_knee, flip_mass_knee_analysis}` |
| P-TAU2 CLI (any maps npz + GT cache) | `tools/witness_tau_knee.py` |
| B19 dither transform (the ~25-LOC core; rule-118 generic; OFF-identical at amp=0) | `src/tac/witness_control/decode_dither.py` |
| P-DITHER A/B driver (chunked+resumable decode/verdict/compare on ANY byte-close packet) | `tools/witness_dither_decode_ab.py` |
| tests | `src/tac/tests/test_witness_annulus_convergence.py` (27/27, 5 new) + `src/tac/tests/test_decode_dither.py` (7/7 new) |
| artifacts (full precision) | `.omx/research/t5_crucible/artifacts/tau_knee_ptau2_20260708.json` · COMMITTED copies `.omx/research/t5_crucible/artifacts/pdither_compare_20260708.json` + `pdither_verdict_dithered_20260708.jsonl` · local (gitignored, deterministically REBUILDABLE from the committed packet + instrument + decode_state.jsonl — no signal loss) `experiments/results/t5_pdither_ab_20260708/{pdither_compare.json, verdict_*.jsonl, argmax_*.i8, dithered.raw}` |

Named consumers (req P): P-TAU2 → SC-3 (quantile-convention owner; the knee f_target ≈ 0.862 is
the static prior its live law starts from) + §1.4a τ_end promotion path. P-DITHER → B19
build-item disposition (§10) + the §0.3 locked-mass coverage computation (the churn-ratio row
below is a NEW measured input to the run-2 asymptote row) + the #149 reformulation queue.

## §A — P-TAU2 MEASURED (full precision; artifact `.omx/research/t5_crucible/artifacts/tau_knee_ptau2_20260708.json`)

Instrument: `tools/witness_tau_knee.py` + `tac.witness_annulus_metrics.{flip_margin_values,
flip_margin_cdf,kneedle_knee,max_curvature_knee,flip_mass_knee_analysis}` (pure numpy). Data =
the SAME 16-pair rendered witness-argmax maps the τ-CONFIRM instrument produced (real
render-through-R + frozen CPU-torch SegNet), margin axis from gt_n600 (never the maps npz).
`[macOS-numpy advisory . NON-PROMOTABLE; 16-pair strided subset — this probe re-derives a
CONSTANT, it kills nothing]`

| leg | criterion | endpoint | m_knee | implied f_target | τ* = m_knee/ln5 | τ*(f±0.05) |
|---|---|---|---|---|---|---|
| BEST_ep650 (n_flips 10,872) | kneedle | q95 | 0.306959 | 0.684787 | 0.190724 | [0.1655, 0.2218] |
| | max_curv | q95 | 0.492644 | 0.831678 | 0.306097 | [0.2593, 0.3747] |
| | **kneedle (PRIMARY)** | **q99** | **0.553328** | **0.861663** | **0.343802** | **[0.288437, 0.431976]** |
| | max_curv | q99 | 0.781373 | 0.932671 | 0.485494 | [0.3764, 0.9606] |
| | kneedle | max | 0.767376 | 0.930556 | 0.476797 | [0.3732, 0.8761] |
| | max_curv | max | 0.799944 | 0.936258 | 0.497033 | [0.3818, 1.1496] |
| END_ep1000 (n_flips 12,212) | kneedle | q95 | 0.375609 | 0.716345 | 0.233379 | [0.2015, 0.2758] |
| | max_curv | q95 | 0.626195 | 0.864559 | 0.389077 | [0.3255, 0.5010] |
| | **kneedle (PRIMARY)** | **q99** | **0.619822** | **0.862512** | **0.385117** | **[0.322451, 0.493240]** |
| | max_curv | q99 | 0.637454 | 0.868654 | 0.396072 | [0.3308, 0.5139] |
| | kneedle | max | 0.850207 | 0.923600 | 0.528263 | [0.4068, 0.9958] |
| | max_curv | max | 0.873824 | 0.926630 | 0.542937 | [0.4124, 1.0973] |

τ*(f) sensitivity table (the f_target → τ* map; sensitivity ≈ **±0.07–0.09 τ\* per ±0.05 of
f_target** at the knee = the primary rows' half-spreads 0.071770 / 0.085320):

| f | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|
| τ*(ep650) | 0.0167 | 0.0355 | 0.0574 | 0.0820 | 0.1115 | 0.1488 | 0.2000 | 0.2774 | 0.3372† | 0.4076 |
| τ*(ep1000) | 0.0186 | 0.0383 | 0.0625 | 0.0906 | 0.1230 | 0.1666 | 0.2228 | 0.3075 | 0.3706† | 0.4619 |

† f=0.85 read from the artifact's f-table (q̂ = 0.85 = v6's SC-3 convention).

**Findings (full precision):**
1. **Implied f_target ≈ 0.862 on BOTH legs** — kneedle@q99 gives f_target = 0.8616626 (ep650)
   / 0.8625123 (ep1000): leg-stable to 3 decimal places, and it lands ON v6's q̂ = 0.85
   bracket-midpoint convention (independent corroboration of SC-3's choice from a DERIVED
   criterion — the marginal-return-collapse point is at f ≈ 0.86).
2. **Primary knee τ\***: 0.343802 (ep650-best) / 0.385117 (ep1000). Launch constant 0.31 sits
   0.034 / 0.075 BELOW the primary knees — inside ep650's f±0.05 sensitivity interval
   [0.2884, 0.4320], marginally below ep1000's [0.3225, 0.4932].
3. **Inversion**: τ_end = 0.31 ⟺ implied f_target = **0.8347130243** (ep650) /
   **0.8019980347** (ep1000) — i.e. the launch constant is the knee-f minus 0.03–0.06.
4. **The knee is NOT sharply localized** (the criterion/endpoint robustness spread is wide:
   full band [0.190724, 0.542937]): the static CDF bends GRADUALLY (heavy tail, no crisp
   elbow) — which independently CONFIRMS v6 §1.4a's decision that f_target is not derivable
   from a static snapshot and must be the run-1 LIVE conversion measurement. This probe's
   knee is the honest $0 PRIOR, not the law.

**VERDICT (per the charter rule, applied to the pre-registered reporting scope):**
**0.31 STANDS.** It lies inside the knee-derived band [0.190724, 0.542937] (leg × criterion ×
endpoint, as pre-registered), inside the ep650 primary sensitivity interval, and equals the
only measured optimum anchor (ep650-best τ = 0.3098). The static knee prior mildly favors a
slightly HIGHER endpoint (primary knees 0.344–0.385 > 0.31, direction consistent with §1.4a's
own over-descent reading) — recorded as a directional note for SC-3's live law, NOT a
correction (per §7c P-TAU2 is a reporting probe; the fail-safe constant stands regardless;
promotion waits for the live f_target). verdict_scope: INSTANCE (a constant's prior on this
vehicle's static end-state fields; 16-pair advisory subset; nothing dies).

## PRE-REGISTRATION (stated BEFORE computing — no post-hoc adjustment)

### P-TAU2 knee criterion (derived, not eyeballed)
μ(m) = flip-mass CDF on the TRUE GT-cache margin axis (each flip pixel = one unit of mass;
margin axis NEVER from the maps npz per the tau-confirm trap). The anneal endpoint should sit
where further τ-descent stops converting flip mass, i.e. where the marginal conversion rate
dμ/dm collapses. PRIMARY criterion: **Kneedle elbow-of-marginal-return** (Satopää, Albrecht,
Irwin, Raghavan 2011, "Finding a 'Kneedle' in a Haystack", arXiv:1608.04355-adjacent canonical;
IEEE ICDCS-W 2011): normalize both axes to [0,1] over [0, m_hi], knee = argmax_m
[μ_norm(m) − m_norm]. DERIVATION of why this is the collapse point: the argmax condition is
μ'(m_knee) = μ(m_hi)/m_hi — the knee is exactly where the MARGINAL conversion rate falls to the
SWEEP-AVERAGE conversion rate; below it every unit of τ·ln5 converts more than average, above it
less (elbow of marginal return, closed-form, no eyeballing). SECONDARY (robustness): max
curvature of the Gaussian-smoothed normalized CDF (interior-restricted). Endpoint convention
m_hi = m_q99 of the flip mass (tail beyond q99 is structural under any reading); robustness
endpoints {q95, max}. Reported: m_knee, implied f_target = μ(m_knee) (absolute flip-mass
fraction), τ* = m_knee/ln5, per leg (ep650-best, ep1000) × criterion × endpoint; sensitivity =
Δτ* for f_target ± 0.05 + a τ*(f) table at 0.05 steps.
VERDICT RULE (from the charter): 0.31 inside the knee-derived band → launch value STANDS;
outside → corrected value at full precision. Scope: this probe re-derives a CONSTANT
(scope=INSTANCE-of-the-constant); per §7c P-TAU2 is a REPORTING probe — the fail-safe constant
0.31 stands regardless, and the LIVE conversion-rate f_target (v6 §1.4a: "NOT derivable from a
static snapshot") remains run-1/SC-3 work. This static knee = the best $0 PRIOR for f_target
(the marginal-return collapse point of the static field), honestly scoped as such.

### P-DITHER pre-registered config + band/kill (v6 §7c row, verbatim-honored)
- Transform (B19 minimal form, ~15–25 LOC core): seeded ordered dither at the uint8
  quantization of the packet's own decode path (`_R`: bicubic-up → **+ amp·(u − 0.5)** → round
  → clamp → uint8). PRIMARY config: Bayer-8 ordered pattern (recursive, generic algorithm),
  amplitude **amp = 1.0 quantum** (u ∈ [0,1) centered → ±0.5 = exactly fills the rounding
  deadzone), seeded per-(pair, frame, channel) integer roll offsets from PCG64(seed=0xB19) —
  deterministic-in-seed, rule-118 clean (NO video-derived table; seed + amp are config scalars).
  Both frames dithered (B19-faithful); SegNet reads frame1 only, so the d_seg gate rides frame1.
- A/B: dithered decode of the EXISTING mod32cap ep650 byte-close packet vs the EXISTING
  undithered R6 raw; n600 chunked verdict both arms via the frozen CPU-torch
  `cpu_verdict_d_seg_argmax_batch` (argmax retained for the per-class-pair split).
  Archive bytes UNCHANGED by construction (decode-side; rate term sizes archive.zip only).
- **fire: Δd_seg ≤ −1e-5** (≥ 0.56× crossing margin in S, req-J-denominated) at unchanged
  bytes; **kill THIS FORM: Δd_seg ≥ 0** (decode-side zeroth-order dither stays unadmitted;
  trained-with dither = the named reformulation, run-2). In-between (−1e-5 < Δ < 0): stays
  gated (no fire, no kill). Scope of any kill: FORMULATION (this zeroth-order decode-side
  form); reformulation queue pre-enumerated: amplitude sweep · blue-noise vs white ·
  band-restricted (locked-geometry-masked) dither · trained-with dither.
- Instrument-validation gate: the recomputed undithered per-pair d_seg must reproduce
  `r6_verdict_pairs.jsonl` bit-for-bit (same helper, same raw) before the A/B is admissible.

(Sections below filled as measured.)

## §B — P-DITHER MEASURED (full precision; artifact `experiments/results/t5_pdither_ab_20260708/pdither_compare.json`)

Config as pre-registered: mode=bayer8, amp=1.0 quantum, seed=0xB19; packet = the R6 mod32cap
ep650 byte-close (`levelset_packet_20260708T013253Z`, archive.zip 83,427 B — bytes UNCHANGED by
construction, decode-side only). Dithered arm = the packet's OWN inflate body with exactly the
one added pre-round term; decode 600/600 pairs chunked (4×150, ~86 s each, 12 workers); verdict
= frozen CPU-torch `cpu_verdict_d_seg_argmax_batch`, n600 BOTH arms, chunked+resumable.
`[macOS-CPU advisory . NON-PROMOTABLE]`

**Instrument validation (both passed BEFORE the A/B was read):**
- amp=0 arm reproduces the undithered R6 raw **bit-for-bit** (first 3 pairs byte-compare) —
  OFF-identical proven, byte-close-selectable semantics hold.
- Recomputed undithered per-pair d_seg matches `r6_verdict_pairs.jsonl` **bit-for-bit on all
  600 pairs** (`undithered_matches_r6_bitforbit: true`).
- Transform sanity: 25.02% of camera bytes change, max |Δ| = 1 quantum exactly, mean Δ =
  −8.74e-5 (~zero-mean) — the intended deadzone-filling perturbation, nothing larger.

**THE GATE ROW (n600, full precision):**

| quantity | value |
|---|---|
| d_seg undithered (R6-validated) | 0.0036145697699652775 |
| d_seg dithered | 0.0036166975233289924 |
| **Δd_seg** | **+2.1277533637149328e-06** (= +0.00021277533637 S) |
| fire bar (pre-registered) | Δ ≤ −1e-5 → NOT reached |
| kill bar (pre-registered) | Δ ≥ 0 → **FIRES** |

**Mechanism decomposition (the census-facing split the charter asked for):**

| split | fixed (und-flip→dit-correct) | created (und-correct→dit-flip) | net (created−fixed) |
|---|---:|---:|---:|
| TOTAL | 10,085 | 10,336 | **+251** |
| far-range-lane rows 176–224 (P-DZ census band; 59.0% of ALL churn) | 5,960 | 6,089 | +129 |
| GT class Lane | 2,497 | 2,650 | +153 |
| GT margin [0.0, 0.1) | 2,420 | 2,448 | +28 (ratio 0.9886) |
| GT margin [0.1, 0.5) | 5,253 | 5,373 | +120 (ratio 0.9777) |
| GT margin [0.5, 2.0) | 2,170 | 2,288 | +118 (ratio 0.9484) |
| GT margin [2.0, ∞) | 242 | 227 | −15 (ratio 1.0661) |

Top class-pairs churned (fixed / created): Lane→Road 2,488/2,643 · Road→Lane 2,164/2,109 ·
Undrivable→Road 980/960 · Road→Undrivable 943/1,003 · Movable→Undrivable 726/737 — the SAME
pairs as the standing flip-mass ranking: the dither reaches exactly the census geometry.

**What the numbers say (the physics row, req P — feeds SC-16/locked-mass coverage):**
1. The dither is NOT inert: it churns 20,421 boundary pixels (≈4.8% of the standing 426k
   flips), with 59% of churn concentrated in the far-range-lane census band — the lever
   REACHES the locked geometry.
2. But the churn is DIRECTION-BLIND: fixed/created ratio ≈ 0.95–0.99 in EVERY low/mid margin
   band. An unbiased zeroth-order decode-side perturbation converts locked mass in BOTH
   directions — it randomizes sub-quantum boundary placement instead of recovering it. The
   measured ratio ≈ 0.98 IS the direct measurement of "how much net GT information the
   witness's sub-quantum residual carries at this checkpoint": ≈ none (slightly negative).
3. **Seed-robustness of the kill (derived from the churn statistics, not asserted):** under a
   direction-blind churn null, sd(net) ≈ √20,421 ≈ 143 flips (independence approx —
   spatially correlated flips make the true sd larger, direction unchanged). The fire bar
   −1e-5 = −1,180 net flips sits **~10σ** from the measured +251. No seed choice of THIS FORM
   plausibly fires; the kill is not a single-seed accident.
4. Reformulation queue read from the SAME artifact ($0, measured priors, no extra runs):
   **amplitude sweep** — scales churn volume, not the ≈0.98 ratio → expected Δ ≥ 0;
   **blue-noise vs white vs bayer** — changes churn spatial correlation, not per-pixel
   direction-blindness → expected Δ ≥ 0; **band-restricted (locked-geometry-masked)** — the
   per-band ratios are ≤ 1 in every band the mask would select → expected Δ ≥ 0 from the
   measured splits. The only queue member that changes the RATIO is one where the RENDER
   knows the dither: **trained-with dither** (v6's named run-2 reformulation) or render-side
   informed placement (#149's actual mechanism, 12× boundary-flip win in its 2026-06-19
   probe). These stay OPEN.

**VERDICT: B19 dies AS-FORMULATED — KILL, verdict_scope=FORMULATION** (zeroth-order
decode-side seeded dither at the uint8 quantization; the pre-registered kill Δd_seg ≥ 0 fires
at +2.1277533637149328e-06). Per v6 §7c executed EXACTLY: decode-side zeroth-order dither
stays UNADMITTED; B19 does NOT ship in run-1's byte-close-selectable set; no launch-blocking
dependency existed either way (§0.3). NOT killed: the #149 sub-pixel-placement MECHANISM
(probe-proven real), the deadzone census + M1 term (untouched — the locked mass is real; this
probe shows it cannot be harvested by unbiased decode-side noise), trained-with dither and the
enumerated reformulations above (run-2 queue, now with measured priors AGAINST the unbiased
decode-side variants and FOR the render-informed ones). Honest note for §0.3: the crossing
case's "dither-class decode repair" leg should be re-pointed at render-informed forms; the
composed-family asymptote's locked-mass coverage now has a measured NEGATIVE prior for its
cheapest lever — the band/clamp/island large-amplitude levers carry the locked-mass burden.

## HONEST LIMITS
- All rows [macOS-CPU advisory]; d_pose not an axis here (pose-BLIND run, w_pose=0); NO score
  claims; pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS.
- P-TAU2: 16-pair strided advisory subset (the τ-CONFIRM maps); the knee is a STATIC prior —
  the live conversion-rate f_target (v6 §1.4a) remains run-1/SC-3 work; the wide
  criterion/endpoint spread (τ* 0.19–0.54) is itself the finding that no crisp static elbow
  exists.
- P-DITHER: n600 full-scale on the REAL byte-close decode path (not a toy); ONE seed measured
  + a derived (independence-approximate) seed-robustness bound; one amplitude/pattern point
  measured with in-artifact reads (not runs) for the other unbiased variants — their Δ ≥ 0
  expectations are measured-prior extrapolations, labelled as such, NOT kills (req R: each
  reformulation keeps its own probe right).
- Peak RSS: verdict chunks ~4–5 GiB (mmap raws + GT cache slices); decode workers ~1 GiB each
  ×12; within the ~8 GiB advisory budget per process.

## SEAL HAND-OFF (NO-OPEN-GATES): both v6 $0 gates now RESOLVED
- P-TAU2 → RESOLVED (reporting probe delivered: knee f_target ≈ 0.862 both legs; launch
  constant 0.31 STANDS inside the knee band; directional note for SC-3 recorded).
- P-DITHER → RESOLVED (KILL-this-form; B19 leaves the run-1 build list §10; §12 gains the row
  "decode-side zeroth-order seeded dither = formulation-dead (P-DITHER)"; reformulation queue
  enumerated with measured priors).
- The final seal round may proceed with zero open $0 gates per the sequencing rule.
