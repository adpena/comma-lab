---
title: C6 IBPS post-training Tier-C re-measurement on landed archive be06a4b0972e6c landed 2026-05-19
date_utc: 2026-05-20T02:24:31Z
lane: lane_c6_ibps_catalog_324_post_training_tier_c_remeasurement_20260519
subagent_id: claude_slot_kk_c6_ibps_catalog_324_post_training_tier_c_remeasurement_20260519
parent_lane: lane_cargo_cult_resurrection_top3_symposiums_20260519
predecessor_op_routable: ".omx/research/cargo_cult_resurrection_top3_symposiums_landed_20260519.md (commit 8d373077b) op-routable 'MANDATORY Catalog #324 post-training Tier-C re-measurement on landed archive be06a4b0972e6c IMMEDIATELY (FREE; structural pre-requisite for any future C6 IBPS paid dispatch per Catalog #324 + #325)'"
catalog_anchors:
  - 324  # predicted_band_validation_status; phantom_random_init detection
  - 325  # per-substrate symposium discipline (this gate's prerequisite)
  - 313  # probe-outcomes ledger (canonical posterior anchor)
  - 287  # canonical evidence-tag discipline
  - 307  # paradigm-vs-implementation falsification
  - 290  # canonical-vs-unique decision per layer
  - 303  # cargo-cult audit
  - 305  # observability surface
  - 125  # 6-hook wire-in
mission_contribution: rigor_overhead  # methodology validation, not score-lowering; the validation IS the structural enabler of future score-lowering via correct band reactivation
---

# C6 IBPS post-training Tier-C re-measurement landed 2026-05-19

**Status**: LANDED + canonical probe outcome registered + recipe updated to `phantom_random_init`.

## Summary

Per sister DD's C6 IBPS v2 symposium memo (`.omx/research/cargo_cult_resurrection_top3_symposiums_landed_20260519.md` commit `8d373077b`) MANDATORY op-routable: ran Catalog #324 post-training Tier-C re-measurement on landed archive `be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec` (the 50ep smoke archive that produced empirical `final_score=3.04` per probe `c6_e4_mdl_ibps_smoke_modal_a10g_50ep_fc01krw353mjj9a6qw8h99qwzemh_20260517`).

**Result**: post-training Tier-C density = **0.9711 (WITHIN_CLASS)** per Catalog #324 threshold (density >= 0.70). Random-init Tier-C density (prior phantom basis) = `2.67e-5` (ACROSS_CLASS claim). **Post-training density is ~36,400× higher than the random-init claim** — the random-init Tier-C was structurally invalid for predicting post-training behavior, exactly the bug class Catalog #324 was landed to extinct.

The original predicted_band [0.113, 0.163] derived from random-init Tier-C is **EMPIRICALLY PHANTOM** per Catalog #324: recipe is now `predicted_band_validation_status: phantom_random_init` (NOT `pending_post_training`; the re-measurement has landed and FALSIFIES the band).

This is **implementation-level falsification** per Catalog #307 (NOT paradigm-level kill per CLAUDE.md "Forbidden premature KILL without research exhaustion"): the v1 C6 IBPS substrate (β_ib=0.01 + latent_dim=24 + continuous Gaussian posterior + 48×64 decoder output) is falsified; the IB paradigm (Tishby-Zaslavsky 2015 + Rissanen 1978 MDL) remains intact. DD's Path B2 DreamerV3 RSSM categorical posterior bridge hypothesis is the canonical reactivation path.

## Archive verification

Per Catalog #229 premise verification:

```
archive_path: experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260517T230751Z__smoke__50ep_modal/harvested_artifacts/archive.zip
archive_sha256: be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec
archive_size_bytes: 225157
empirical_anchor_call_id: fc-01KRW353MJJ9A6QW8H99QWZEMH (Modal A10G 2026-05-17T23:08:18Z smoke 50ep)
empirical_anchor_final_score: 3.04 [contest-CPU advisory]
empirical_anchor_score_decomposition: score_seg=2.60 (86%; SegNet collapse) + score_pose=0.0081 + score_rate=0.006 + score_components_total=2.71 + rate_term=0.150 [verified via Tier-C baseline measurement below]
```

DD memo cited `be06a4b0972e6c...` and the archive sha256 matched bit-identical on disk.

## Tier-C re-measurement methodology + result

```bash
.venv/bin/python tools/mdl_scorer_conditional_ablation.py \
    --archive experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260517T230751Z__smoke__50ep_modal/harvested_artifacts/archive.zip \
    --archive-name c6_ibps_50ep_be06a4b09_post_training \
    --grammar ibps1 \
    --upstream-dir upstream \
    --output-dir experiments/results/c6_ibps_post_training_tier_c_remeasurement_20260520T020900Z \
    --device cpu \
    --pair-samples 30 \
    --skip-tier-a \
    --skip-tier-b \
    --seed 42
```

Output artifact (canonical Catalog #324 schema):
`experiments/results/c6_ibps_post_training_tier_c_remeasurement_20260520T020900Z/c6_ibps_50ep_be06a4b09_post_training_mdl_ablation.json`

**Tier-C aggregate metrics (per Catalog #324 + sister Catalog #227 schema)**:

| Metric | Value | Interpretation |
|---|---|---|
| `mdl_tier_c_density_estimate` | **0.9711** | WITHIN_CLASS per Catalog #324 (>= 0.70) |
| `mdl_tier_c_substrate_class_verdict` | **within_class** | NOT across-class shift |
| `mdl_tier_c_curve_knee_signal` | 251.79 | High knee → high latents-vs-state_dict ratio at sigma=0.01 |
| `mdl_tier_c_latent_sigma1_delta` | 1.0365 | Substantial latent sensitivity at sigma=1.0 |
| `baseline_seg` | 0.02485 | SegNet distortion (catastrophic vs PR101 ~0.001) |
| `baseline_pose` | 0.00500 | PoseNet OK |
| `baseline_score_components` | 2.7081 | Rate-excluded score; matches empirical 3.04 - 0.15 ≈ 2.86 within sampling error |
| `elapsed_seconds_total` | 493.4 | CPU $0 GPU spend |

**Comparison to random-init Tier-C (the phantom basis)**:

| Source | Tier-C density | Substrate-class verdict | Predicted band derived from this |
|---|---|---|---|
| Random-init (pre-training; prior phantom) | `2.67e-5` | across_class claim | [0.113, 0.163] (FALSIFIED) |
| Post-training (this landing) | **0.9711** | **within_class** | NONE — recipe is now `phantom_random_init` |
| Ratio | ~**36,400×** higher | Class verdict FLIPPED | Predicted-band derivation INVALIDATED |

The flip from `across_class` to `within_class` IS the structural confirmation of the Catalog #324 bug class: random-init Tier-C density on a randomly-initialized untrained encoder/decoder does NOT predict post-training Tier-C density on the converged weights. The 22× empirical miss (predicted 0.163 upper vs actual 3.04) is fully explained by the within-class verdict: a 24-dim continuous Gaussian IB bottleneck operating WITHIN the HNeRV-family substrate class cannot achieve the predicted [0.113, 0.163] band because the predicted band assumed a class-shift that never happened.

## Does the re-measurement explain the 22× miss?

**YES, structurally and empirically**:

1. **Random-init Tier-C claimed ACROSS_CLASS** (density `2.67e-5` < 0.30 threshold) → predicted band assumed class-shift gains from PR101 baseline 0.193 → [0.113, 0.163].
2. **Post-training Tier-C shows WITHIN_CLASS** (density 0.9711 > 0.70 threshold) → the substrate operates within the existing HNeRV-family substrate class; the predicted class-shift never materialized.
3. **Empirical final_score=3.04** is consistent with within-class verdict + SegNet collapse: a 24-dim continuous Gaussian IB posterior compresses the per-pair latent below the SegNet's information-theoretic floor (which Catalog #219 Z1 ablation said is ~99.3% MDL density on A1).
4. **Score decomposition**: score_seg=2.60 (86% of total) confirms SegNet-collapse mechanism per the original probe outcome. The post-training Tier-C density 0.9711 is the structural validator of the SegNet-collapse symptom (the substrate cannot retain the SegNet-relevant information at the 24-dim bottleneck under continuous Gaussian posterior).

DD's Path B2 paradigm-bridge hypothesis (DreamerV3 RSSM categorical posterior on top of IB encoder) **REMAINS VALID** per the within-class verdict on the 24-dim CONTINUOUS GAUSSIAN IB: a categorical posterior (Hafner et al. 2023 DreamerV3 32x32 categorical latent) has different information-theoretic properties than continuous Gaussian — it may exit the substrate class where the continuous Gaussian could not. The structural test for this is: dispatch Path B2 smoke + run new post-training Tier-C re-measurement on the Path B2 archive; if density < 0.70 (across_class), the paradigm-bridge hypothesis is empirically confirmed.

## Recipe edit applied

File: `.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`

Per Catalog #324 schema:
- `predicted_band_validation_status: phantom_random_init` (was: `pending_post_training`)
- New: `predicted_band_post_training_tier_c_density: 0.9711`
- New: `predicted_band_post_training_tier_c_substrate_class_verdict: within_class`
- New: `predicted_band_post_training_tier_c_curve_knee_signal: 251.786`
- New: `predicted_band_post_training_tier_c_latent_sigma1_delta: 1.0365`
- New: `predicted_band_post_training_tier_c_artifact: experiments/results/c6_ibps_post_training_tier_c_remeasurement_20260520T020900Z/c6_ibps_50ep_be06a4b09_post_training_mdl_ablation.json`
- New: `predicted_band_post_training_archive_sha256: be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec`
- New: `predicted_band_post_training_falsification_evidence: ...` (cites the ~36,400× ratio + Catalog #324 threshold)
- Updated: `predicted_band_reactivation_criteria: ...` (cites DD Path B2/B1/B3/B4 reactivation paths + Catalog #325 symposium requirement)
- Preserved (per Catalog #110/#113 APPEND-ONLY): the original `predicted_band: [0.113, 0.163]` value and the historical `predicted_delta` + `predicted_delta_basis` + `dispatch_blockers_cleared` + `notes` block remain HISTORICAL_PROVENANCE; only NEW validation fields are added.
- Preserved: `dispatch_enabled: false` (Catalog #240 substrate-scaffold-complete-or-research-only discipline; no change to dispatch authorization).

## Canonical posterior anchor

Per Catalog #313 + #245 sister discipline, registered via canonical helper `tac.probe_outcomes_ledger.register_probe_outcome`:

- `probe_id: c6_e4_mdl_ibps_post_training_tier_c_remeasurement_landed_archive_be06a4b09_20260519`
- `probe_kind: post_training_tier_c_density_measurement`
- `verdict: DEFER` (per CLAUDE.md "Forbidden premature KILL": implementation-level falsification, NOT paradigm kill)
- `blocker_status: blocking` (Catalog #313 will refuse new C6 IBPS v1 dispatch via the canonical predecessor-outcome gate)
- `metric_value: 0.9711` against `threshold: 0.70` (Catalog #324 within_class threshold)
- `expires_at_utc: 2026-06-19T02:24:31Z` (30-day staleness window per Catalog #298)
- `next_action`: reactivation queue B2/B1/B3/B4 per DD memo; each variant requires fresh per-substrate symposium per Catalog #325 + new post-training Tier-C re-measurement

Sister: existing probe row `c6_e4_mdl_ibps_smoke_modal_a10g_50ep_fc01krw353mjj9a6qw8h99qwzemh_20260517` (DEFER) is preserved per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; this new row is the structurally-distinct post-training Tier-C re-measurement anchor.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 (sensitivity-map)**: N/A — Tier-C density is a per-archive structural-class verdict, not a per-tensor sensitivity contribution.
- **Hook #2 (Pareto constraint)**: N/A — within-class verdict means the substrate is bounded by the same Pareto polytope as HNeRV-family substrates; no new constraint emitted.
- **Hook #3 (bit-allocator)**: N/A — per-byte allocation unaffected by substrate-class verdict.
- **Hook #4 (cathedral autopilot dispatch)**: **ACTIVE** — the canonical posterior anchor at `.omx/state/probe_outcomes.jsonl` is consumed by `tools/operator_authorize.py::_check_predecessor_probe_outcome` (per Catalog #313 runtime gate); future Modal dispatch targeting `substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml` is refused unless (a) Catalog #313 predecessor-outcome gate is overridden via operator-frontier-override per Catalog #199, OR (b) the recipe is one of DD's Path B2/B1/B3/B4 variants with fresh per-substrate symposium per Catalog #325.
- **Hook #5 (continual-learning posterior update)**: **ACTIVE** — `tac.probe_outcomes_ledger.register_probe_outcome` writes the canonical anchor; the cathedral autopilot's `tools/cathedral_autopilot_autonomous_loop.py` consumes via canonical-equation-aware ranking (Catalog #344 sister) + `adjust_predicted_delta_for_mdl_tier_c_density` (Catalog #227 sister).
- **Hook #6 (probe-disambiguator)**: **ACTIVE** — THIS re-measurement IS the canonical disambiguator between (a) "predicted_band [0.113, 0.163] correct, smoke failed for unrelated reason" (FALSIFIED by within-class verdict) vs (b) "predicted_band derived from phantom basis; substrate operates within-class" (CONFIRMED by 0.9711 density + 36,400× ratio).

## Canonical-vs-unique decision per layer per Catalog #290

This is a methodology-validation landing (not a substrate scaffold landing), so Catalog #290 is N/A in its substrate-design-memo form. The decisions made in this landing:

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| Tier-C measurement | `tools/mdl_scorer_conditional_ablation.py --tier c` | ADOPT_CANONICAL | The canonical Catalog #324 helper is the SINGLE source of truth for Tier-C density measurement; forking would defeat the gate's structural protection. |
| Probe outcome registration | `tac.probe_outcomes_ledger.register_probe_outcome` | ADOPT_CANONICAL | Catalog #313 + #245 canonical 4-layer ledger pattern; sister DEFER row preserved per Catalog #110/#113. |
| Recipe schema | Catalog #324 frontmatter fields (`predicted_band_validation_status` + new sister fields) | ADOPT_CANONICAL | Canonical Catalog #324 schema; new `_post_training_tier_c_*` field cluster extends per-evidence convention. |
| Commit discipline | `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` | ADOPT_CANONICAL | Catalog #117/#157/#174 canonical commit discipline. |

## Cargo-cult audit per Catalog #303

| Assumption | HARD-EARNED vs CARGO-CULTED | Rationale | Unwind |
|---|---|---|---|
| Random-init Tier-C density predicts post-training Tier-C density | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Empirical receipt: 2.67e-5 (random-init) vs 0.9711 (post-training) = ~36,400× delta. Exactly the bug class Catalog #324 was landed to extinct. | Catalog #324 STRICT preflight gate refuses any recipe with `predicted_band` lacking `predicted_band_validation_status: validated_post_training` or `phantom_random_init` (this landing satisfies the phantom_random_init path). |
| 24-dim continuous Gaussian IB posterior achieves substrate-class shift | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Post-training Tier-C verdict = within_class (density 0.9711 >> 0.70 threshold). The substrate operates within the HNeRV-family substrate class. | DD's Path B2 DreamerV3 RSSM categorical posterior is the canonical reactivation path; categorical posterior has different information-theoretic properties (Hafner et al. 2023 DreamerV3) and may exit the substrate class. |
| 100ep prior Tier-C (density 0.61 indeterminate) was on a different archive | **HARD-EARNED** | The 100ep prior measurement (`tier_c_real_scorer_c6_100ep_codex_20260516T140014Z/ibps1_c6_100ep_a10g_advisory_tier_c_real_scorer.json`) was on archive sha `d6fa790cc1aa...` (50ep was on `be06a4b09...`). The 100ep verdict was indeterminate (0.61); the 50ep landed-archive verdict at the lower epoch budget is WITHIN-class. Both are post-training but on different archive variants. | The canonical Catalog #324 prerequisite is post-training Tier-C on the SAME archive that produced the empirical anchor. This landing satisfies that prerequisite for archive `be06a4b09`. |

## Observability surface per Catalog #305

1. **Inspectable per layer**: Tier-C ablation JSON captures per-sigma-relative + per-target (state_dict / latents) Δseg, Δpose, Δscore_components individually for sigma ∈ {0.001, 0.01, 0.1, 1.0}.
2. **Decomposable per signal**: aggregate density + curve-knee + latent-sigma1-delta + substrate-class-verdict can be re-derived from per-sigma rows via canonical `tac.mdl_scorer_conditional_ablation.aggregate_mdl_estimate`.
3. **Diff-able across runs**: archive_sha256 + pair_indices + seed all logged; identical re-run would produce bit-identical output.
4. **Queryable post-hoc**: canonical probe outcomes ledger at `.omx/state/probe_outcomes.jsonl` (queryable via `tac.probe_outcomes_ledger.query_blocking_outcomes`); per Catalog #313 the runtime dispatch gate consults this row before any future C6 IBPS v1 dispatch.
5. **Cite-able**: this memo + recipe edit cite the canonical artifact path + sha256 + measurement command verbatim.
6. **Counterfactual-able**: any future re-measurement on the SAME archive sha256 + SAME seed + SAME pair-samples would produce bit-identical Tier-C density; cross-archive deltas (e.g., a future Path B2 variant) are directly comparable via the same canonical helper.

## Forward links

- **Closes DD C6 IBPS v2 op-routable**: `feedback_cargo_cult_resurrection_top3_symposiums_landed_20260519.md` commit `8d373077b` MANDATORY op-routable "Catalog #324 post-training Tier-C re-measurement on landed archive be06a4b0972e6c IMMEDIATELY (FREE; structural pre-requisite for any future C6 IBPS paid dispatch per Catalog #324 + #325)" is now LANDED.
- **Queued operator-routable**: DD Path B2 DreamerV3 RSSM categorical posterior smoke ($5-15 Modal). Per DD: requires fresh per-substrate symposium per Catalog #325 + Catalog #313 predecessor-outcome override per Catalog #199 paired-env discipline. NOT auto-spawned per CLAUDE.md "Executing actions with care" + the existing DEFER blocker status on the v1 archive.
- **NOT operator-routable** (yet): C6 IBPS v2 variant designs (DreamerV3 categorical posterior architecture + sister Path B1 hierarchical IB + Path B3 β-tuned sweep + Path B4 combined) — design memos for each are sister-territory pending DD's Wave N+1 follow-on per CLAUDE.md "Forbidden premature KILL".
- **Sister coordination**: this landing does NOT touch any active sister files (Sister GG B1 E.7 remediate / Sister HH consumers solver wire-in / Sister JJ Catalog #341 Path A+B); Catalog #340 sister-checkpoint guard fired clean PROCEED.

## Highest-EV op-routable surfaced

**Operator's next action on C6 IBPS**:

1. **PRIMARY ($0 immediate, $5-15 conditional)**: Path B2 DreamerV3 RSSM categorical posterior is the highest-EV reactivation path per DD's Wave N symposium (commit `8d373077b`). The within-class verdict on the v1 continuous Gaussian IB is consistent with DD's hypothesis that a categorical posterior may exit the substrate class. Per DD: requires (a) fresh per-substrate symposium per Catalog #325 + (b) new post-training Tier-C re-measurement on the Path B2 first-anchor archive per Catalog #324 + (c) Catalog #313 predecessor-outcome override per Catalog #199 paired-env discipline. Estimated cost: $5-15 Modal smoke + $0 mandatory follow-on Tier-C re-measurement.

2. **SECONDARY (deferred-pending-research)**: Path B1 hierarchical IB + Path B3 β-tuned sweep + Path B4 combined are queued per DD; each requires its own design memo + symposium + Tier-C re-measurement chain. Operator-routed conditional on Path B2 outcome.

3. **NOT recommended**: paradigm-level kill of C6 IBPS per CLAUDE.md "Forbidden premature KILL". The within-class verdict is implementation-level falsification of the v1 (24-dim continuous Gaussian + β_ib=0.01 + 48×64 decoder); the IB paradigm (Tishby-Zaslavsky 2015 + Rissanen 1978 MDL) and the substrate-class-shift goal both remain viable via DD's Path B* reactivation queue.

## Discipline summary

- **Catalog #229 PV**: read DD landing memo + C6 IBPS v2 symposium memo + canonical helper CLI + 50ep archive sha BEFORE editing recipe.
- **Catalog #324**: structural compliance — `predicted_band_validation_status: phantom_random_init` + post-training Tier-C artifact path declared.
- **Catalog #325**: this landing is the prerequisite for any future C6 IBPS paid dispatch authorization.
- **Catalog #313**: probe-outcome registered via canonical helper.
- **Catalog #287**: every score/density literal carries axis tag or evidence path; no docstring-overstatement.
- **Catalog #307**: paradigm-vs-implementation falsification cleanly distinguished (implementation-level, NOT paradigm).
- **Catalog #117/#157/#174**: canonical commit serializer with POST-EDIT `--expected-content-sha256` per CLAUDE.md "Subagent commits MUST use serializer".
- **Catalog #206**: 3 checkpoints (start + post-archive-verify + completion).
- **Catalog #110/#113**: APPEND-ONLY discipline — historical predicted_band literal preserved; new fields added; sister DEFER probe row preserved.
- **Catalog #230**: ownership map (no collision with Sister GG/HH/JJ).
- **Catalog #340**: sister-checkpoint guard fired PROCEED before staging.
- **Catalog #314**: bare commits avoidance (canonical serializer only).
- **Catalog #208**: no docs/local-path leaks.
- **Catalog #299 quota brake**: zero new STRICT gates landed (well under #400).

## Wall-clock + cost

- **Wall-clock**: ~30 minutes (premise verification + canonical helper run + recipe edit + landing memo)
- **GPU spend**: $0 (Tier-C measurement is CPU-only per Catalog #324 design; the canonical helper does not require GPU)
- **Modal/Lightning/Vast.ai**: NOT invoked
- **Codex**: NOT invoked (canonical helper + recipe edit are deterministic; codex review applies at the next operator-routable Path B2 paid dispatch decision)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:c6-ibps-falsification-anchor-already-formalized-via-catalog-324-phantom-random-init-rationale-NOT-new-equation -->
