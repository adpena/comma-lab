# bs1 PREFLIGHT — the flip-budget density law, and what it does to gc14's restart bound

**Date:** 2026-08-01 · **Axis:** `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false`
**Cost:** $0, scorer-free (read-only over a banked atlas; no scorer slot consumed)
**Task:** #815 (bs1 / gc14-R1 preflight) · **Pointer:** v4d 0.9639878 UNMOVED — this is MEANS.

---

## §0 HEADLINE

gc14 derived a vehicle-native falsifier rather than transplanting one: the SegNet head is exact
rank-4 with flip distance `d = |m|/‖Δw_e‖`, and realized `d_seg` is a *count of margin sign-flips*,
so weight-space averaging over restart excursions reduces the margin field's **variance** near zero
without moving its mean — therefore **restart gain ∝ margin density at zero**, measurable before any
A/B runs. This is that measurement.

**Three results:**

1. **The mechanism SURVIVES.** Implied excursion scale ε\* = **0.001007 budget units = 0.065% of the
   median budget**. A perturbation that small is entirely plausible for an Adam warm restart with
   zeroed moments. The preflight does **not** refute V5.

2. **NEW MEASURED LAW — the flip-budget density is CONSTANT near zero.** `CDF(ε)/ε` is flat to four
   digits across a **30× range** of ε (0.11108 → 0.11080 over ε ∈ [1e-3, 3e-2], a 0.25% spread).
   **ρ₀ = 0.11091 per budget unit.** Nothing in the corpus recorded this; every prior treatment of
   the margin field assumed nothing about its shape near zero.

3. **gc14's own bound is on the WRONG VARIABLE — and this is the decision-relevant consequence.**
   It bounded the restart branch at `2 × 0.00946 = 0.019 S` by varying the restart **COUNT** with a
   fixed per-restart step. But under a constant density, the rescued mass is `ρ₀·ε` — **linear in the
   excursion MAGNITUDE ε**, which is a controllable hyperparameter (restart LR, moment-reset depth),
   not a constant. Count and magnitude are different levers and only one was bounded.

---

## §1 PROVENANCE

| item | value |
|---|---|
| atlas | `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa80_margin_budget/qa80_conservative_budget_n600.npy` |
| sha256 | `ba57fd20db5fa98d6da31e77406ed007896a3ae4a8b3ca2b430538da8e9849e5` |
| schema | `qa80_margin_budget_field.v1`, mode `conservative` |
| geometry | `[600, 384, 512]` = 117,964,800 finite values (full n600, scorer grid) |
| atlas source | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` |
| gc14 receipt | `.omx/research/ddm_gc14_first_descent_20260731.md` |

---

## §2 THE MEASUREMENT

Near-zero CDF of the conservative flip-budget field:

| ε | CDF(ε) | pixels | CDF/ε |
|---:|---:|---:|---:|
| 1e-3 | 1.110755e-04 | 13,103 | 0.11108 |
| 3e-3 | 3.329213e-04 | 39,273 | 0.11097 |
| 1e-2 | 1.109136e-03 | 130,839 | 0.11091 |
| 3e-2 | 3.324110e-03 | 392,128 | 0.11080 |
| 0.1 | 1.091880e-02 | 1,288,034 | 0.10919 |
| 0.3 | 3.081652e-02 | 3,635,265 | 0.10272 |

Linear to 0.25% over ε ∈ [1e-3, 3e-2]; departs only past ε ≈ 0.1. Field median 1.5530, mean 1.4677.

**Inverting for the excursion scale** (rescue fraction = fraction of near-zero mass the averaging
recovers), against gc14's two measured targets:

| rescue_frac | ε\* from window_03 boundary step (1.118e-4) | ε\* from window_02 (2.1003e-4) |
|---:|---:|---:|
| 1.00 | 0.001007 (0.065% of median) | 0.001905 (0.123%) |
| 0.50 | 0.002028 (0.131%) | 0.003798 (0.245%) |
| 0.25 | 0.004051 (0.261%) | 0.007605 (0.490%) |

---

## §3 THE CAMPAIGN IN FLIP-BUDGET COORDINATES (new, compact)

Using ρ₀, any `d_seg` maps to an **effective flip threshold** `ε_eff = d_seg / ρ₀`:

| state | d_seg | ε_eff | % of median budget |
|---|---:|---:|---:|
| burn-4 best | 0.0038892 | 0.03507 | 2.258% |
| bar-required (from S ≥ 100·d_seg) | 0.00172141 | 0.01552 | 0.999% |

**The seg problem restated: shrink the effective flip threshold by 2.26×** — from ~2.3% of the median
margin budget to ~1.0%. That is a far more tractable-sounding statement of the same debt than
"0.36640 S", and it is the same fact.

**Reachable mass vs excursion (HEADROOM, explicitly NOT a gain claim):**

| ε | mass | S-equivalent | % of current d_seg |
|---:|---:|---:|---:|
| 0.001 | 1.1108e-04 | 0.01111 | 2.9% |
| 0.003 | 3.3292e-04 | 0.03329 | 8.6% |
| 0.010 | 1.1091e-03 | 0.11091 | 28.5% |
| 0.030 | 3.3241e-03 | 0.33241 | 85.5% |

---

## §4 HONEST SCOPE — what this does NOT establish

1. **HEADROOM ≠ GAIN.** §3's reachable-mass table says how much flip mass *sits* in each band. It
   does **not** say averaging captures it. gc14's derivation is a **small-ε linearization** (variance
   reduction with the mean fixed); at larger excursions the same mechanism can *create* flips as
   readily as rescue them. The A/B is still the only thing that measures capture.
2. **The atlas is GT-referenced.** It is the budget field of the reference (`gt_n600.npz`), i.e. the
   *flip-prone population*, not our realized error set. That `d_seg = 0.0038892` lands near
   `CDF(0.035)` is **suggestive, INFERRED, not measured** — "our errors are the low-budget pixels" is
   a hypothesis this file does not test. `ε_eff` is a coordinate change, not a claim about which
   pixels are wrong.
3. **`rescue_frac` is unmeasured.** ε\* scales inversely with it; the ladder is reported rather than
   a single number precisely because the fraction is unknown.
4. **Single source, single mode.** One atlas, `conservative` mode, one axis, advisory. No noise floor
   on ρ₀ (it is a population statistic of a fixed field, so sampling noise is nil, but *mode*
   sensitivity is untested).
5. **V5 remains INFERRED.** This preflight failed to refute the mechanism; that is weaker than
   confirming it. The half-window vs unbroken-control A/B (#815) is unchanged in necessity.

---

## §5 CONSEQUENCES

- **#815 stays live and its outcome bound is REOPENED.** The `2×` framing bounded restart *count*.
  Under ρ₀-constant the lever is `ρ₀·ε` — the A/B should vary **excursion magnitude**, not only
  cadence, or it measures the less interesting of the two axes.
- **A `--restart-cadence` DSL Lever is insufficient by itself**; the magnitude knob (restart LR /
  moment-reset depth) belongs in the same lever, or the count-only lever inherits gc14's bound by
  construction. Per the "off is a tracked queue" rule it lands as a `Lever` factory, not a trainer flag.
- **ρ₀ is a registrable law** (`flip_budget_density_at_zero_v1`): it converts any d_seg into an
  effective threshold and any excursion into a mass, which is exactly the currency the guard ledger
  (#809/cg1) and the per-class targets were missing.
- **Relative significance** (magnitude-dismissal discipline): gap = 0.9639878 − 0.172141 = 0.7918468.
  gc14's restart ceiling 0.019 S = **2.40% of gap**; the month's only measured descent (window_02,
  0.018303 S) = **2.31%**. Ratio 1.038×. This is not a small lever.
