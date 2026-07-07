# Dash-comb probe verdict (#287) — the two owed anchors of `dash_erasure_homogenization_v1` MEASURED — 2026-07-07

**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE. Pointer **0.19110 UNMOVED** — this is a means
(mechanism measurement), not an exact row. All numbers n600 (ALL 600 pairs), frozen mod32cap
EMA-best **ep650**, through the EXACT contest R (torch bicubic↑874×1164 → round/clamp/uint8 →
SegNet contest bilinear → argmax) vs GT `lstars`, frozen CPU-torch SegNet, never MPS.

**Apparatus:** `tools/dash_comb_probe_n600.py` (committed 9556d38f2; chunked resumable foreground,
atomic tmp+replace state, verdict batches 6, free-RAM ≥ 20 GiB gate, peak RSS **5.7 GiB**).
Two harness kills (exit 144, the known ~5-min limit) became progress via the checkpoint —
resume spine worked exactly as designed (204→300→372→450→510→570→600).
Results JSON (gitignored, rebuildable): `experiments/results/dash_comb_probe_20260707/dash_comb_probe_n600_20260707.json`.

## Probe 1 — corrector A/B (owed anchor i)

| condition | d_seg (mean) | Δ vs witness | lane recall | dash-gap FP |
|---|---|---|---|---|
| c1 witness alone | 0.00315 | — | 0.7795 | 0.00013 |
| c2 + SOLID band (homogenized) | 0.01356 | +0.01042 | 0.7326 | 0.00174 |
| c3 + band × EGO-PHASE COMB (#287) | 0.00695 | +0.00380 | 0.7291 | 0.00036 |
| c4 + band × per-pair FITTED gate (#215) | 0.00698 | +0.00383 | 0.7297 | 0.00062 |

(c1 baseline 0.00315 vs the run's live EMA-best 0.00337 — reproduces the trained level; the
witness-alone leg is the internal positive control and it registered.)

**VERDICT — MECHANISM CONFIRMED, CORRECTOR-AS-COMPOSITE NO-GO:**

1. **The comb is a real dash corrector.** It removes **86.0%** of the ADDED dash-gap FP the
   homogenized solid band introduces (added gap-FP: solid +0.00161 → comb +0.00023) and **63.5%**
   of the solid band's added d_seg (+0.01042 → +0.00380). This is the homogenization law's
   corrector prediction (`u ≈ ū + δ·v(x/δ)` with phase from ξ) registering on the R+SegNet leg.
2. **Global comb ≡ per-pair fitted gate at a fraction of the bytes.** The ego-phase comb
   (global period/duty/ego-scale + per-slot phase + sparse strided anchors ≈ **186 floats**)
   MATCHES the byte-expensive per-pair fitted gate on d_seg (0.00695 vs 0.00698) and BEATS it
   on dash-gap FP (0.00036 vs 0.00062, −42%). The dash phase really is transported by ego-distance
   — the per-pair phase storage is redundant with ξ. This is the byte-accounting half of the owed anchor.
3. **BUT every band composite is NET-NEGATIVE on this checkpoint** (+0.0038 d_seg for the best
   gate; recall DROPS 0.780→0.729). The ep650 witness already renders lanes near its floor;
   post-hoc compositing the analytic band (u-mask OFF by design, isolating the gate contrast)
   adds FP without adding recall. **Do NOT upgrade to `_corrector_v1` as a render-time composite.**
   The corrector's remaining live path is **IN-TRAINING**: fire the DSL lever `n287_dash_comb`
   (`--lane-band-dash-comb`, wired in the levelset trainer) so the witness learns against the
   combed prior instead of being overwritten by it.

Per-band FP (forward-range bands 4–10/10–20/20–35/35–55/55+ m): comb/solid FP ratio 0.48/0.57/
0.46/0.43/0.80 — the comb's advantage is largest exactly where δ_along ≫ smoothing scale
(δ_along per band: 110/19.8/5.4/1.9/0.39 px; dominant slot −1 ego-lane-left, period 7.54 m,
global fit period 12.08 m, duty 0.365, ego scale 0.1036); the far band is below-crossover and
gate-off (>55 m) by design.

## Probe 2 — τ/smoothing crossover (owed anchor ii)

Render-side analog: Gaussian-blur the c3 comb-gated alpha at σ ∈ {1,2,4,8} px before compositing.
**PREDICTION** (homogenization): dash-gap FP rises from combed toward solid level as σ/δ_along
crosses O(1). **MEASURED:** d_seg falls monotonically 0.00695→0.00340 and gap-FP falls
0.00036→0.00017 — toward the **no-op (c1)**, not toward the solid (c2) level.

**VERDICT — REFUTED-AS-IMPLEMENTED (amplitude confound).** The band is thin (softness 1 px);
blurring collapses peak α below compositing relevance, so the sweep measures corrector-amplitude
decay, not dash homogenization. The crossover SHAPE on the render leg is NOT observed with this
design. Honest state: the training-flow τ-crossover (dash-gap FP vs τ/δ_along across a τ-anneal
on a fixed ckpt family) remains OWED; a render-side re-design would need amplitude-normalized
blur (renormalize peak α post-blur). The law's crossover half keeps its five held anchors; this
probe neither confirms nor damages it.

## Equations leg (appended, APPEND-ONLY)

`EmpiricalAnchor` **`dash_comb_corrector_ab_and_tau_crossover_n600_20260707`** appended to
**`dash_erasure_homogenization_v1`** via `tac.canonical_equations.update_equation_with_empirical_anchor`
(registry event `anchor_appended`; equation now holds 2 anchors). Headline residual **0.1403**
(= 1 − 0.8597 measured share of added gap-FP removed vs the ideal-corrector limit 1.0).
`_corrector_v1` upgrade **NOT taken** — reactivation criteria updated to the in-training form.

## Costate / duty-to-measure (FIRST FIRING)

Digest line before this work (`tools/costate_digest.py`):
`duty-to-measure (36 owed; *=never-fired): AACoverageRender*, AdamBeta2*, AmplifyIsland*, AnalyticLaneRenderBand*, BoundaryDistance*, CacheGtSkeleton* (+30 more)` — **DashComb was in the
never-fired queue.** This probe is the lever's FIRST FIRING at the mechanism level: recorded
`fired` + `measured` events in `tac.witness_dsl.activation_ledger` (agent
`build_wave_a_resume_dash_comb_20260707`), with the honest scope note that the **in-training**
`--lane-band-dash-comb` arm has still never run. Post-recording: **35 owed**, DashComb off the
never-fired list. Witness run health at measurement time: ALIVE pid 97677, best d_seg 0.0033662
@ep650, telemetry ep875, free 51.9 GiB (probe stayed read-only on the live run's artifacts).

## Next actions (ranked)

1. **In-training comb A/B** (the reactivation path): one bounded n600 arm with `n287_dash_comb`
   active vs the sealed base — the only remaining way the comb can LOWER d_seg (operator-GO for
   the launch per CONTAINMENT).
2. **Training-flow τ-crossover** (still owed): measure dash-gap FP across the τ-anneal on saved
   per-stage checkpoints of an existing run — $0, no new training.
3. The τ_end coupling rule (`smoothing_crossover_ok`) stays binding: do not anneal τ below the
   dash period without a corrector-class lever active.

means ≠ ends: pointer 0.19110 moves only via `upstream/evaluate.py` on exact archive bytes.
