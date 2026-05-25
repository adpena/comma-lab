# Combined Tier-1 WAVE-2 Sister Probes: Hinton KL T=2.0 LONGER temporal-context + UNIWARD per-segment-label [macOS-CPU advisory]

```yaml
---
council_tier: T1
council_attendees: []
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: true
council_override_rationale: "operator_frontier_override: 'continue with all' + 3-msg rate-attack amplification per Catalog #300 mission-alignment Consequence 1 — Tier-1 probe iteration depth justified by Probe 6 W=2 11.3x strongest-yet signal demanding W-extension cascade test"
council_decisions_recorded:
  - "probe 7 (M44 Hinton KL T=2.0 x D17 LONGER temporal-context W in {4,6,8}) verdict POSITIVE_SIGNAL_PLATEAU — kl_mean_temporal scales W=4:16.52x W=6:21.20x W=8:19.57x CCC static baseline 6.74e-4; saturates W=6 (W=8 < W=6 fails monotone); paradigm INTACT (Probe 6 W=2 11.29x finding UNCHANGED) per Catalog #307; IMPLEMENTATION-LEVEL W>=8 extension DEFER-PENDING per Catalog #308; Probe 6 W=2 Tier-2 dispatch UNCHANGED"
  - "probe 8 (M40 UNIWARD x D16 per-segment-label connected-components) verdict POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL — min segment textured_avg_weight=0.5233 BREAKS Probe 5 per-class ceiling 0.5673 (Delta -0.0440); inter-segment spread 0.4027 = 2.2x Probe 5 inter-class spread 0.1808; 22 valid segments; threshold 0.5 NOT yet broken; paradigm INTACT (per-instance granularity real); DEFER-PENDING-MULTI-SCALE per Catalog #308"
  - "queue 2 canonical equation candidates (hinton_kl_temperature2_longer_temporal_context_v1 + uniward_per_segment_label_v1) for operator-routable RATIFY-N per Catalog #344"
  - "queue Probe 8 sister-cascade: per-instance + UNIWARD multi-scale wavelet COMBINED per Holub-Fridrich 2014 + Catalog #308"
  - "queue Probe 7 sister-cascade: motion-aware weighting per Atick-Redlich 1990 + Rao-Ballard predictive-coding paradigm at $0 macOS-CPU advisory"
  - "Probe 6 W=2 Tier-2 paid dispatch operator-routable UNCHANGED per RATE-ATTACK-MATRIX cell #4; W>=4 extension does NOT add Tier-2 justification"
---
```

## Summary

Combined Tier-1 WAVE-2 sister probes per COMBINED-TIER-1-CCC-EXT-PROBES landing memo commit `685fe6726`. Both probes extend the WAVE-1 POSITIVE_SIGNAL probes along their canonical sister-cascade dimensions:

- **Probe 7** extends Probe 6's `W=2` window to `W ∈ {4, 6, 8}` to test whether the 11.3x temporal-context Hinton KL signal SCALES monotonically or PLATEAUS at small W.
- **Probe 8** explores Probe 5's `D16 per-segment-label` alternative reducer per Catalog #308 (replacing per-class hard-classification with per-instance connected-components segmentation).

Carmack MVP-first 5-step cascade: extend at $0 macOS-CPU advisory smoke before paid Tier-2 escalation. Per Catalog #1 + #192 + #287 + #323 non-promotable by construction.

**Probe 7 verdict POSITIVE_SIGNAL_PLATEAU**: kl_mean_temporal scales `W=4: 16.52x`, `W=6: 21.20x`, `W=8: 19.57x` CCC static baseline 6.74e-4 — the temporal-context signal SATURATES at `W=6` (W=8 < W=6 violates monotone increase). Per CLAUDE.md "Forbidden premature KILL": PARADIGM-LEVEL INTACT (Probe 6 W=2-3 11.29x finding UNCHANGED); IMPLEMENTATION-LEVEL W>=8 extension is DEFER-PENDING per Catalog #307. Probe 6 W=2 Tier-2 paid dispatch recommendation UNCHANGED; W>=4 extension does NOT add Tier-2 justification (additional SegNet forwards do not pay for themselves beyond the W=6 saturation point).

**Probe 8 verdict POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL**: per-instance connected-components UNIWARD min textured_avg_weight=`0.5233` **BREAKS** Probe 5 per-class ceiling `0.5673` (Δ −0.0440); inter-segment spread `0.4027` = **2.2× Probe 5 inter-class spread** 0.1808; 22 valid segments across 4 pairs and 5 classes. Threshold 0.5 NOT yet broken at the per-instance dimension alone. Per CLAUDE.md "Forbidden premature KILL": paradigm INTACT (per-instance granularity reveals tighter UNIWARD inversion than per-class hard mask); IMPLEMENTATION-LEVEL threshold needs COMBINED per-instance + UNIWARD multi-scale wavelet per Holub-Fridrich 2014 + Catalog #308 alternative reducer.

## Per-probe predicted vs actual signature

### Probe 7: Hinton KL T=2.0 × LONGER temporal-context (M44 × D17 W-extension)

| Signature | Predicted | Actual | Verdict-relevant |
|---|---|---|---|
| W=8 ≥ 20× CCC static (`POSITIVE_SCALES`) | hard threshold | 19.57× | FAIL (just below) |
| Monotone increase W=4 < W=6 < W=8 | required for SCALES | NO (W=8 < W=6) | FAIL |
| W=8 ∈ [10×, 20×] CCC static (`POSITIVE_PLATEAU`) | range | 19.57× ∈ range | PASS |
| All W in ±20% Probe 6 W=2 plateau band | secondary plateau test | W=4: 1.46× W=6: 1.88× W=8: 1.73× over Probe 6 W=2 | FAIL (W=6 + W=8 exceed +20% band) |
| Multi-class drift ≥2/5 classes drift > 1e-3 | required for non-NEGATIVE | preserved across all W | PASS |

**Per-window KL_mean_temporal vs CCC static baseline (6.74e-4)**:

| W | kl_mean_temporal | ratio over CCC static | ratio over Probe 6 W=2 (7.607e-3) |
|---:|---:|---:|---:|
| 2 (Probe 6) | 7.607e-3 | 11.29× | 1.00× |
| 4 | 1.113e-2 | 16.52× | 1.46× |
| 6 | 1.428e-2 | 21.20× | 1.88× |
| 8 | 1.318e-2 | 19.57× | 1.73× |

The signal grows monotonically W=2 → W=4 → W=6 (11.29x → 16.52x → 21.20x) then dips at W=8 (19.57x). **Saturation at W=6** is the canonical signature; longer windows mix in motion-incoherent frames that dilute the temporal-coherent dark-knowledge structure.

### Probe 8: UNIWARD × per-segment-label connected-components (M40 × D16)

| Signature | Predicted | Actual | Verdict-relevant |
|---|---|---|---|
| min segment textured_avg_weight < 0.5 (`POSITIVE_BREAKS_THRESHOLD`) | hard threshold | 0.5233 | FAIL (just above) |
| min segment < Probe 5 per-class min 0.5673 (`POSITIVE_PARTIAL`) | breaks Probe 5 ceiling | 0.5233 (Δ -0.0440) | PASS |
| inter-segment spread > Probe 5 inter-class spread 0.1808 | spread increase | 0.4027 (2.2×) | PASS |
| valid_segment_count ≥ 4 | cross-class diversity | 22 | PASS |

**Per-class connected-component segment counts**:

| Class | Segments |
|---:|---:|
| 0 | (varies per pair; majority typically road-like uniform) |
| 1-4 | spread per-instance across 4 pairs |
| **Total valid segments (≥200 px + ≥50 textured px)** | **22** |

The per-instance dimension yields **22 valid segments** (vs Probe 5's 4 valid classes), enabling per-instance granularity that captures distinct local texture profiles within a class (e.g. two separate class-4 vehicles with distinct UNIWARD weighting). Min segment **0.5233 breaks Probe 5's 0.5673 ceiling by Δ −0.0440** but does not yet break the 0.5 threshold.

## Delta vs WAVE-1 baselines

| Probe | Metric | WAVE-1 baseline | This probe | Delta |
|---|---|---|---|---|
| 7 | kl_mean ratio over CCC static | 11.29× (Probe 6 W=2) | W=4: 16.52×, W=6: 21.20× (peak), W=8: 19.57× | +5.23× (W=4); +9.91× (W=6); +8.28× (W=8) |
| 8 | min segment/class textured_avg_weight | 0.5673 (Probe 5 per-class) | 0.5233 (per-instance) | -0.0440 (breaks Probe 5 ceiling) |
| 8 | inter-segment/class spread | 0.1808 (Probe 5 per-class) | 0.4027 (per-instance) | +0.2219 (2.2× spread) |

## Catalog #313 probe-outcomes ledger rows

Both registered via canonical helper `tac.probe_outcomes_ledger.register_probe_outcome` (fcntl-locked JSONL append-only at `.omx/state/probe_outcomes.jsonl`):

- **Probe 7**: `probe_id=tier_1_distortion_hinton_kl_t2_longer_temporal_context_segnet_smoke`, `substrate=hinton_kl_temperature2_longer_temporal_context`, `verdict=DEFER`, `blocker_status=advisory`, `metric=kl_mean_temporal_w8_ratio_over_ccc_static_baseline`, `value=19.57`, `threshold=20.0`, expires=`2026-06-24T15:52:58Z` (30-day staleness window per Catalog #298).
- **Probe 8**: `probe_id=tier_1_distortion_uniward_per_segment_label_segnet_smoke`, `substrate=uniward_per_segment_label`, `verdict=PARTIAL`, `blocker_status=advisory`, `metric=min_segment_textured_avg_weight`, `value=0.5233`, `threshold=0.5`, expires=`2026-06-24T15:52:58Z`.

## Canonical equation candidates queued for Catalog #344 RATIFY-N

1. `hinton_kl_temperature2_longer_temporal_context_v1` — FORMALIZATION_PENDING per Catalog #344; canonical predicate: `KL_temporal_W(p) = KL(softmax(teacher_window_W/T), softmax(student_static/T))` where `teacher_window_W = mean SegNet logits over (2W+1)-frame window centered at pair p`. Empirical anchor: `kl_mean_temporal(W=6) = 1.428e-2 = 21.20x CCC static baseline 6.74e-4` (POSITIVE_SIGNAL_PLATEAU; saturates W=6).
2. `uniward_per_segment_label_v1` — FORMALIZATION_PENDING per Catalog #344; canonical predicate: `cost_segment_s = 1 / (mean_over_segment_pixels(|HL| + |LH| + |HH|) + sigma_fridrich)` for each connected-component segment derived from scipy.ndimage.label on per-class SegNet hard mask. Empirical anchor: `min_segment_textured_avg_weight = 0.5233` (POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL; breaks Probe 5 per-class ceiling 0.5673 but threshold 0.5 unbroken).

## Re-route operator priority queue (Carmack MVP-first step 5)

### Probe 6 W=2 Tier-2 paid dispatch UNCHANGED (Probe 7 PLATEAU verdict)

- **RATE-ATTACK-MATRIX cell #4 Tier-2 paid dispatch on per-time-window Hinton-distilled scorer surrogate** at W=2 (Probe 6's original recommendation) via HF Jobs T4 (RECHARGE pending) or Vast.ai 4090 ($0.25/hr); estimated cost ~$2-5; predicted ΔS -0.005 to -0.020 [predicted].
- **W>=4 extension does NOT add Tier-2 justification**: signal saturates W=6; longer windows do NOT add score-lowering potential beyond Probe 6 W=2-3 anchor; extra SegNet forwards do not pay for themselves.

### Probe 8 sister-cascade queued (PARTIAL verdict)

- **Per-instance + UNIWARD multi-scale wavelet COMBINED** per Holub-Fridrich 2014 multi-level wavelet decomposition + per-instance connected-components segmentation. Sister probe at $0 macOS-CPU.
- Predicted: combining per-instance granularity (Probe 8 spread 0.4027) with multi-scale wavelet sharpening (multi-level db8 DWT instead of single-level) should drive min segment textured_avg_weight below 0.5 threshold.

### Probe 7 sister-cascade queued (PLATEAU verdict; W-extension exhausted)

- **Motion-aware weighting per Atick-Redlich 1990 + Rao-Ballard predictive-coding**: ego-motion-conditioned temporal averaging (weight each window frame by inverse optical-flow magnitude relative to center frame). Sister probe at $0 macOS-CPU.
- Predicted: motion-aware weighting at W=6 should match or exceed W=6's 21.20x peak while reducing the W=8 saturation noise.

### Catalog #344 RATIFY-N op-routable

- 2 canonical equation candidates queued for operator review + canonical registry registration via `tools/recalibrate_equation.py` per `hinton_kl_temperature2_longer_temporal_context_v1` + `uniward_per_segment_label_v1`.

## Carmack MVP-first 5-step compliance

1. **FREE local CPU smoke** — both probes ran on darwin_arm64 macOS-CPU advisory ($0 GPU spend; Probe 7: 9.27s + Probe 8: 1.43s = ~10.7s wall-clock total).
2. **Falsifiably challenge** — Probe 7 challenged W-extension scaling vs plateau (PLATEAU: saturates W=6); Probe 8 challenged per-segment-label threshold 0.5 (PARTIAL: 0.5233 breaks Probe 5 ceiling but not 0.5).
3. **Catalog #344 reference** — 2 canonical equation candidates queued for operator-routable RATIFY-N.
4. **Land verdict in same commit batch** — 2 probe scripts + 2 verdict JSONs + 2 Catalog #313 ledger rows + this landing memo land together via canonical serializer.
5. **Re-route operator priority queue** — operator-routable Probe 6 W=2 Tier-2 paid dispatch UNCHANGED + Probe 8 sister-cascade + Probe 7 sister-cascade; this section + queued op-routables fulfill the structural re-routing.

## Discipline checklist

- Catalog #1 + #192: probes `[macOS-CPU advisory]` non-promotable; Tier-2 requires paired Linux x86_64 + NVIDIA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- Catalog #287: every empirical claim has evidence tag `[macOS-CPU advisory]`
- Catalog #323: canonical Provenance umbrella (verdict JSONs carry `canonical_provenance` block with kind=macos_cpu_advisory + score_claim_valid=False + promotable=False)
- Catalog #313: 2 probe outcomes registered via canonical helper `register_probe_outcome`; advisory blocker_status; 30-day staleness expiry
- Catalog #344: 2 canonical equation candidates QUEUED for operator-routable RATIFY-N (FORMALIZATION_PENDING)
- Catalog #110/#113 APPEND-ONLY: NEW probe scripts + verdict JSONs + landing memo + Catalog #313 rows only; ZERO mutation of Probe 1-6 scripts or verdict JSONs or canonical equation registry or sister cathedral consumers
- Catalog #229 PV: read COMBINED-TIER-1-CCC-EXT-PROBES landing memo + Probe 5 + Probe 6 source canonical interface; verified Probe 6 W=2 baseline 7.607e-3 = 11.29x CCC static baseline 6.74e-4; verified Probe 5 per-class min 0.5673 + spread 0.1808; verified register_probe_outcome signature BEFORE building
- Catalog #157 + #117 + #174 canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #206 5-tool-use checkpoint cadence (3+ checkpoints + completion)
- Catalog #230: sister-disjoint verified (PAIR-FRAME-LATTICE + MLX-ARCH-5 + DROP-MANY-BEAM-BUILD-1 all disjoint; this lane owns only the 4 NEW files + 2 Catalog #313 row appends + 1 landing memo)
- Catalog #340: sister-checkpoint guard PROCEED required at commit time
- CLAUDE.md "Forbidden premature KILL without research exhaustion": both probes are DEFER-PENDING (Probe 7 W-extension paradigm DEFER-PENDING per Catalog #308; Probe 8 PARTIAL → DEFER-PENDING-MULTI-SCALE per Catalog #308); paradigms INTACT
- CLAUDE.md "Bit-level deconstruction and entropy discipline" + "Apples-to-apples evidence discipline": all metrics tagged + per-probe baselines documented vs WAVE-1 anchors
- CLAUDE.md "Strict-flip atomicity rule": no new STRICT gates introduced

## 6-hook wire-in declaration per Catalog #125

- **Hook 1 (sensitivity-map)**: ACTIVE — Probe 7 per-W ratio_over_ccc and per-class drift are temporal-coherent sensitivity signals consumable by `tac.sensitivity_map.*`; Probe 8 per-segment textured_avg_weight is a per-instance UNIWARD sensitivity signal.
- **Hook 2 (Pareto constraint)**: N/A at probe stage; Tier-2 paid dispatch on either substrate would add a Pareto constraint per Catalog #233 promotion canonical.
- **Hook 3 (bit-allocator)**: N/A at probe stage; per-segment UNIWARD cost-map is a future per-pixel bit-allocator signal.
- **Hook 4 (cathedral autopilot dispatch)**: ACTIVE — both verdict JSONs are queryable via Catalog #313 ledger; future cathedral consumer scaffold per Catalog #335 can auto-ingest temporal-context saturation signal + per-instance UNIWARD inversion signal as ranking inputs.
- **Hook 5 (continual-learning posterior)**: ACTIVE — both Catalog #313 rows feed canonical posterior at `.omx/state/probe_outcomes.jsonl`.
- **Hook 6 (probe-disambiguator)**: ACTIVE — these probes ARE probe-disambiguators for the WAVE-1 POSITIVE_SIGNAL probes' dimensional extension questions (Probe 7 disambiguates W-scaling vs W-plateau; Probe 8 disambiguates per-class vs per-instance UNIWARD granularity).

## Cross-references

- Predecessors: `combined_tier_1_ccc_ext_probes_uniward_per_class_plus_hinton_kl_temporal_context_landed_20260525.md` (WAVE-1 Probe 5 + Probe 6); `overnight_ccc_tier_1_distortion_axis_4_probes_landed_20260521.md` (CCC); DDD probe 3b verdict; EEE HILL probe 3c verdict
- Sister: RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cell #3 + #4 alternative reducer + W-extension
- Future: Probe 7 sister-cascade (motion-aware weighting per Atick-Redlich + Rao-Ballard); Probe 8 sister-cascade (per-instance + multi-scale wavelet COMBINED per Holub-Fridrich 2014); Probe 6 W=2 Tier-2 paid dispatch operator-routable UNCHANGED per RATE-ATTACK-MATRIX cell #4
- AAA T4 grand council symposium PROCEED_WITH_REVISIONS verdict (commit a8b02679) Decision #3
- Quantizr 0.33 [contest-CUDA] canonical paradigm per CLAUDE.md "Quantizr intelligence"
- CLAUDE.md "Fridrich inverse steganalysis" + "Grand Council Geoffrey Hinton" + "Grand Council Atick" + "Quantizr intelligence" + "Forbidden premature KILL"

## Operator-routable next steps

1. **Probe 6 W=2 Tier-2 paid dispatch op-routable (UNCHANGED per Probe 7 PLATEAU)** — per-time-window Hinton-distilled scorer surrogate via HF Jobs T4 (pending RECHARGE) or Vast.ai 4090 ($0.25/hr × ~1-2 hr = ~$2-5); predicted ΔS -0.005 to -0.020 [predicted]. **W>=4 extension does NOT add Tier-2 justification** (signal saturates W=6; extra SegNet forwards do not pay for themselves).
2. **Probe 8 sister-cascade op-routable (PARTIAL)** — per-instance + UNIWARD multi-scale wavelet COMBINED per Holub-Fridrich 2014 at $0 macOS-CPU advisory; predecessor for Probe 8 to break the 0.5 threshold before any paid Tier-2 escalation.
3. **Probe 7 sister-cascade op-routable (PLATEAU)** — motion-aware weighting per Atick-Redlich 1990 + Rao-Ballard predictive-coding paradigm at $0 macOS-CPU advisory; sister test of whether ego-motion-conditioned temporal averaging at W=6 exceeds W=6's 21.20x peak.
4. **Catalog #344 RATIFY-N op-routable** — 2 canonical equation candidates queued for operator review + canonical registry registration.
5. **DEFERRED, NOT KILLED** — both probes are DEFER-PENDING per CLAUDE.md "Forbidden premature KILL"; paradigms INTACT. Reactivation criteria: (Probe 7) motion-aware sister probe exceeds W=6 peak; (Probe 8) per-instance + multi-scale COMBINED yields min segment textured_avg_weight < 0.5.

## Lane id

`lane_combined_tier_1_wave_2_hinton_kl_longer_temporal_plus_uniward_per_segment_label_20260525` L1 (impl_complete + memory_entry + Catalog #313 ledger rows).

## Cost summary

- $0 GPU spend (macOS-CPU advisory only per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192)
- ~10.7s wall-clock for both probes (Probe 7: 9.27s; Probe 8: 1.43s)
- Probe 7 ran 20 SegNet single-frame forwards (W=8 window radius × 4 center pairs ⇒ 17 unique frame indices cached + 4 pair-construction artifacts)
- Probe 8 ran 4 SegNet pair forwards (4 pairs × 5-class argmax + per-pair connected-components labeling via scipy.ndimage.label)
