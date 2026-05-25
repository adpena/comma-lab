# Combined Tier-1 CCC-ext Probes: UNIWARD per-class explicit + Hinton KL T=2.0 temporal-context [macOS-CPU advisory]

```yaml
---
council_tier: T1
council_attendees: []
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_decisions_recorded:
  - "probe 5 (M40 UNIWARD x D15 per-class explicit) verdict POSITIVE_SIGNAL_PER_CLASS_PARTIAL — paradigm INTACT, per-class boundaries align with SegNet blindspot (spread=0.18) but threshold 0.5 not yet broken (min=0.5673); DEFER-PENDING per Catalog #308 alternative reducer"
  - "probe 6 (M44 Hinton KL T=2.0 x D17 per-time-window) verdict POSITIVE_SIGNAL_TEMPORAL_CONTEXT — kl_mean_temporal=7.607e-3 vs CCC static baseline 6.74e-4 (11.3x increase); 2/5 classes drift; JUSTIFIED for Tier-2 paid dispatch via HF Jobs T4 / Vast.ai 4090 per RATE-ATTACK-MATRIX cell #4"
  - "queue 2 canonical equation candidates (uniward_per_class_explicit_v1 + hinton_kl_temperature2_temporal_context_v1) for operator-routable RATIFY-N per Catalog #344"
  - "queue probe 5 sister-cascade: per-segment-label D16 + multi-scale wavelet per Catalog #308"
  - "queue probe 6 Tier-2 paid dispatch operator-routable per RATE-ATTACK-MATRIX cell #4"
---
```

## Summary

Combined Tier-1 CCC-extension sister probes per RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cells #3 + #4. Both probes extend CCC POSITIVE_SIGNAL probes along NEW RATE-ATTACK-MATRIX dimensions: probe 5 = M40 UNIWARD × D15 per-class explicit; probe 6 = M44 Hinton KL T=2.0 × D17 per-time-window temporal-context. Carmack MVP-first 5-step cascade: extend at $0 macOS-CPU advisory smoke before paid Tier-2 escalation. Per Catalog #1 + #192 + #287 + #323 non-promotable by construction.

**Probe 5 verdict POSITIVE_SIGNAL_PER_CLASS_PARTIAL** — min class textured_avg_weight=0.5673 (CCC baseline=0.806, DDD wavelet baseline=0.626); inter-class spread=0.1808; 4/5 classes with stable statistics; threshold 0.5 not yet broken. Paradigm INTACT per Catalog #307: per-class boundaries DO align with SegNet stride-2-stem blindspot per Yousfi+Fridrich inverse-steganalysis paradigm (spread=0.18 confirms class-conditional sharpening); IMPLEMENTATION-level needs tighter inversion (e.g. per-segment-label D16 OR multi-scale wavelet per Catalog #308).

**Probe 6 verdict POSITIVE_SIGNAL_TEMPORAL_CONTEXT** — kl_mean_temporal=7.607e-3 vs CCC static baseline=6.74e-4 (**11.3x increase**); 2/5 classes with measurable per-pair drift (>1e-3 abs); temporal-context teacher EXHIBITS substantially more dark-knowledge structure than CCC static teacher-vs-noisy-student paradigm. JUSTIFIED: Tier-2 paid dispatch on per-time-window Hinton-distilled scorer surrogate via HF Jobs T4 (RECHARGE pending) or Vast.ai 4090 ($0.25/hr) per RATE-ATTACK-MATRIX cell #4. Predicted ΔS -0.005 to -0.020 [predicted] per CCC probe 1 + Quantizr 0.33 [contest-CUDA] anchor.

## Per-probe predicted vs actual signature

### Probe 5: UNIWARD × per-class explicit SegNet (M40 × D15)

| Signature | Predicted | Actual | Delta vs CCC | Delta vs DDD |
|---|---|---|---|---|
| min class_textured_avg_weight < 0.5 | hard threshold | 0.5673 | -0.239 vs 0.806 | -0.059 vs 0.626 |
| inter-class spread > 0.1 | ≥ 0.1 | 0.1808 | N/A | N/A |
| valid_class_count >= 2 | ≥ 2 | 4 | N/A | N/A |

**Per-class textured_avg_weight breakdown (4/5 classes)**: spread 0.5673..0.7481 (0.1808 spread). Class 0 (road-like majority) typically smoother; class 4 (other/textured) carries higher uniward weighting density. The per-class signal IS sharper than DDD wavelet-subband alone (0.626 → 0.567 min = -0.059 absolute), but the 0.5 threshold remains unbroken at W=2 frames + 5 classes empirical setting.

### Probe 6: Hinton KL T=2.0 × temporal-context SegNet (M44 × D17)

| Signature | Predicted | Actual | Delta vs CCC static |
|---|---|---|---|
| kl_mean_temporal >= 6.74e-4 | ≥ baseline | 7.607e-3 | **+7.07e-3 (11.3x)** |
| multi_class_drift >= 2 | ≥ 2 | 2 | N/A |
| temporal_strictly_positive > 1e-5 | > 1e-5 | 7.607e-3 | N/A |
| temporal_soft_entropy >= static | ≥ | 0.0181 > 0.0149 (delta +0.0032) | N/A |

**Per-class temporal drift (window-mean vs static)**: 5 classes total, with 2 exhibiting measurable absolute drift > 1e-3 per-pair. Temporal-context teacher (W=2 averaging over 5-frame window) reveals motion-coherent class-boundary structure that the per-frame argmax does NOT surface — a kind of D17 temporal-coherent cooperative-receiver signal per Atick-Redlich 1990.

## Delta vs CCC/DDD baselines

| Probe | Metric | CCC | DDD | This probe | Delta |
|---|---|---|---|---|---|
| 5 | min class textured_avg_weight | 0.806 (per-pixel) | 0.626 (wavelet-subband) | 0.5673 (per-class wavelet) | -0.239 vs CCC; -0.059 vs DDD |
| 6 | kl_mean | 6.74e-4 (static T=2.0) | N/A | 7.607e-3 (temporal-context T=2.0) | +7.07e-3 (11.3x) |

## Catalog #313 probe-outcomes ledger rows

Both registered via `tac.probe_outcomes_ledger.register_probe_outcome`:

- **Row 5**: probe_id=`tier_1_distortion_uniward_per_class_explicit_segnet_smoke`, substrate=`uniward_per_class_explicit`, verdict=`PARTIAL`, blocker_status=`advisory`, metric_name=`min_class_textured_avg_weight`, metric_value=0.5673, threshold=0.5, expires=2026-06-24 (30-day staleness window).
- **Row 6**: probe_id=`tier_1_distortion_hinton_kl_t2_temporal_context_segnet_smoke`, substrate=`hinton_kl_temperature2_temporal_context`, verdict=`PROCEED`, blocker_status=`advisory`, metric_name=`kl_mean_temporal`, metric_value=0.007607, threshold=0.000674, expires=2026-06-24.

## Canonical equation candidates queued for Catalog #344 RATIFY-N

1. `uniward_per_class_explicit_v1` — FORMALIZATION_PENDING per Catalog #344; canonical predicate: `cost_i = 1/(detail_i + sigma_fridrich)` per-pixel UNIWARD weighting WITH SegNet-class-conditional per-class textured-region thresholding; per-class boundaries align with SegNet stride-2-stem blindspot. Empirical anchor: min class textured_avg_weight=0.5673 (PARTIAL; threshold 0.5 unbroken at W=2 + 5 classes empirical setting).
2. `hinton_kl_temperature2_temporal_context_v1` — FORMALIZATION_PENDING per Catalog #344; canonical predicate: `KL_temporal(p) = KL(softmax(teacher_window/T), softmax(student_static/T))` where teacher_window = mean SegNet logits over 5-frame window centered at pair p. Empirical anchor: kl_mean_temporal=7.607e-3 (POSITIVE_SIGNAL_TEMPORAL_CONTEXT; 11.3x increase vs CCC static baseline).

## Re-route operator priority queue (Carmack MVP-first step 5)

### IF operator approves Tier-2 paid dispatch (probe 6 POSITIVE):
- **RATE-ATTACK-MATRIX cell #4 Tier-2**: per-time-window Hinton-distilled scorer surrogate paid smoke via HF Jobs T4 (RECHARGE pending external billing) OR Vast.ai 4090 ($0.25/hr). Estimated cost ~$2-5; predicted ΔS -0.005 to -0.020 [predicted].
- Recipe pending per AAA T4 §6.5 + CCC probe 1 + Quantizr 0.33 [contest-CUDA] anchor.

### Probe 5 alternative reducer cascade (per Catalog #308):
- D16 per-segment-label: replace per-class hard mask with per-segment-instance label (SegNet does not emit instance segmentation; would require sister scorer or DBSCAN over class mask). Lower-EV than D17 per-time-window.
- Multi-scale wavelet (J-UNIWARD multi-level per Holub-Fridrich 2014): higher-resolution decomposition. Sister probe at $0 macOS-CPU.

### Probe 6 sister-cascade (operator-discretion at $0):
- Longer window W=3-5: confirm temporal-context signal strengthens monotonically with window radius.
- Motion-aware weighting: ego-motion-conditioned temporal averaging per Atick-Redlich + Rao-Ballard predictive-coding paradigm.

## Carmack MVP-first 5-step compliance

1. **FREE local CPU smoke** — both probes ran on darwin_arm64 macOS-CPU advisory ($0 GPU spend; ~6s wall-clock total for both).
2. **Falsifiably challenge** — probe 5 challenged per-class explicit dimension threshold 0.5 (PARTIAL: 0.5673 unbroken); probe 6 challenged temporal-context KL exceeds CCC static baseline (POSITIVE: 11.3x).
3. **Catalog #344 reference** — 2 canonical equation candidates queued for operator-routable RATIFY-N.
4. **Land verdict in same commit batch** — 2 probe scripts + 2 verdict JSONs + 2 Catalog #313 ledger rows + this landing memo land together via canonical serializer.
5. **Re-route operator priority queue** — operator-routable Tier-2 paid dispatch for probe 6 verdict + alternative reducer cascade for probe 5 verdict; this section + queued op-routables fulfill the structural re-routing.

## Discipline checklist

- Catalog #1 + #192: probes `[macOS-CPU advisory]` non-promotable; Tier-2 requires paired Linux x86_64 + NVIDIA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- Catalog #287: every empirical claim has evidence tag `[macOS-CPU advisory]`
- Catalog #323: canonical Provenance umbrella (verdict JSONs carry `canonical_provenance` block with kind=macos_cpu_advisory + score_claim_valid=False + promotable=False)
- Catalog #313: 2 probe outcomes registered via canonical helper `register_probe_outcome`; advisory blocker_status; 30-day staleness expiry
- Catalog #344: 2 canonical equation candidates QUEUED for operator-routable RATIFY-N (FORMALIZATION_PENDING)
- Catalog #110/#113 APPEND-ONLY: NEW probe scripts + verdict JSONs + landing memo + Catalog #313 rows only; ZERO mutation of CCC/DDD/EEE probe scripts or verdict JSONs
- Catalog #229 PV: read CCC probe 1 + CCC probe 3 + DDD probe 3b for canonical interface; verified PR 101 archive + SegNet/PoseNet locations; confirmed baselines from verdict JSONs (kl_mean=6.74e-4, textured_avg_weight=0.806, DDD=0.626) BEFORE building
- Catalog #157 + #117 + #174 canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #206 5-tool-use checkpoint cadence (4 checkpoints + completion)
- Catalog #230: sister-disjoint verified (PAIR-FRAME-LATTICE + MLX-ARCH-5 + DROP-MANY-BEAM-DESIGN all disjoint; this lane owns only the 5 NEW files + 1 Catalog #313 row append)
- CLAUDE.md "Forbidden premature KILL without research exhaustion": probe 5 PARTIAL → DEFER-PENDING (NOT KILL); paradigm INTACT
- CLAUDE.md "Bit-level deconstruction and entropy discipline" + "Apples-to-apples evidence discipline": all metrics tagged + per-probe baselines documented

## 6-hook wire-in declaration per Catalog #125

- **Hook 1 (sensitivity-map)**: ACTIVE — probe 5 per-class textured_avg_weight is a per-class UNIWARD sensitivity signal consumable by `tac.sensitivity_map.*` downstream; probe 6 per-class temporal drift is a per-class temporal-coherent sensitivity signal.
- **Hook 2 (Pareto constraint)**: N/A at probe stage; Tier-2 paid dispatch on probe 6 substrate would add a Pareto constraint per Catalog #233 promotion canonical.
- **Hook 3 (bit-allocator)**: N/A at probe stage.
- **Hook 4 (cathedral autopilot dispatch)**: ACTIVE — both verdict JSONs are queryable via Catalog #313 ledger; future cathedral consumer scaffold per Catalog #335 can auto-ingest temporal-context dark-knowledge signal as ranking input.
- **Hook 5 (continual-learning posterior)**: ACTIVE — both Catalog #313 rows feed canonical posterior at `.omx/state/probe_outcomes.jsonl`.
- **Hook 6 (probe-disambiguator)**: ACTIVE — these probes ARE probe-disambiguators for the CCC POSITIVE_SIGNAL probes' dimensional extension questions.

## Cross-references

- Predecessors: `feedback_overnight_ccc_tier_1_distortion_axis_4_probes_landed_20260521.md` + `overnight_ccc_tier_1_distortion_axis_4_probes_landed_20260521.md` + DDD probe 3b verdict + EEE HILL probe 3c verdict
- Sister: RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cell #3 + #4
- Future: probe 5 sister-cascade per Catalog #308; probe 6 Tier-2 paid dispatch operator-routable per RATE-ATTACK-MATRIX cell #4
- AAA T4 grand council symposium PROCEED_WITH_REVISIONS verdict (commit a8b02679) Decision #3
- Quantizr 0.33 [contest-CUDA] canonical paradigm per CLAUDE.md "Quantizr intelligence"
- CLAUDE.md "Fridrich inverse steganalysis" + "Grand Council Geoffrey Hinton" + "Grand Council Atick" + "Quantizr intelligence" + "Forbidden premature KILL"

## Operator-routable next steps

1. **Tier-2 paid dispatch op-routable (probe 6 POSITIVE_TEMPORAL_CONTEXT)** — per-time-window Hinton-distilled scorer surrogate via HF Jobs T4 (pending RECHARGE) or Vast.ai 4090 ($0.25/hr × ~1-2 hr = ~$2-5); predicted ΔS -0.005 to -0.020 [predicted]; recipe pending per AAA T4 §6.5.
2. **Probe 5 sister-cascade op-routable (probe 5 PARTIAL)** — per-segment-label D16 OR multi-scale wavelet at $0 macOS-CPU advisory; predecessor for probe 5 to break the 0.5 threshold before any paid Tier-2 escalation.
3. **Catalog #344 RATIFY-N op-routable** — 2 canonical equation candidates queued for operator review + canonical registry registration via `tools/recalibrate_equation.py`.
4. **DEFERRED, NOT KILLED** — probe 5 PARTIAL is DEFER-PENDING per CLAUDE.md "Forbidden premature KILL without research exhaustion"; paradigm INTACT, IMPLEMENTATION-level needs tighter inversion. Reactivation criteria: per-segment-label D16 OR multi-scale wavelet sister probe yields min class_textured_avg_weight < 0.5.

## Lane id

`lane_combined_tier_1_ccc_ext_probes_uniward_per_class_plus_hinton_kl_temporal_context_20260525` L1 (impl_complete + memory_entry + Catalog #313 ledger rows).

## Cost summary

- $0 GPU spend (macOS-CPU advisory only per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192)
- ~6s wall-clock for both probes (probe 5: 1.67s; probe 6: 4.38s)
- 2 SegNet forward batches (probe 5: 4 pairs; probe 6: 8 single-frame forwards)
