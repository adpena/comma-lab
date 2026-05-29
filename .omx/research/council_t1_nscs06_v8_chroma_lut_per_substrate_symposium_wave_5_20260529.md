---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, AssumptionAdversary, NSCS06-v6-v7-author-cite, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Wave 5 lands 2 canonical helpers + extincts 1 prior cargo-cult bypass + identifies 1 NEW cargo-cult with opt-in unwind. But the dispatch that would settle cargo-cult #6 empirical question (does MODE help or hurt contest scorer?) is DEFERRED. The frontier-breaking value of Wave 5 is conditional on operator-routable next-step #3 (paired-CUDA RATIFICATION ~$0.06)."
  - member: AssumptionAdversary
    verbatim: "Sister cargo-cult #4 (per-(level,class) median aggregation) remains UNFIXED at helper surface despite being flagged in 2026-05-26 audit. RECOMMENDED-WAVE-6 op-routable but the META-class is: 'cargo-cult unwind helper landed but never wired into the canonical compress-side function' recurs across BOTH cargo-cult #3 (just fixed) AND cargo-cult #4 (still pending). The structural protection per Catalog #335 sister-extinction is NOT YET universal across the v8 substrate's compress side."
council_assumption_adversary_verdict:
  - assumption: "Wave 5 audit found all material cargo-cults"
    classification: HARD-EARNED
    rationale: "Cross-checked against 2026-05-26 audit's 12 assumptions + sister CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS declarations in revisions.py. The 1 new cargo-cult #6 found IS the sister-pattern of #4 at a different helper surface."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "byte-parity preservation on NEAREST default is structurally necessary per CLAUDE.md frontier-pointer non-negotiable"
    classification: HARD-EARNED
    rationale: "All 6 prior NSCS06 v8 Modal dispatches keyed empirical anchors to NEAREST byte hashes; switching default would invalidate them per Catalog #110/#113 HISTORICAL_PROVENANCE."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "MODE opt-in is the canonical UNWIND PATH form for archive-bytes-changing helper additions"
    classification: HARD-EARNED
    rationale: "Sister of Catalog #335 cathedral-consumer canonical contract + the operator's general 'opt-in for archive-bytes change' discipline per CLAUDE.md."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
council_decisions_recorded:
  - "op-routable #1 HIGH: Wave 5 cargo-cult #3 wire-in + #6 NEW unwind helper landed; 226 tests pass with byte parity"
  - "op-routable #2 MEDIUM: Wave 6 extends helper pattern to build_chroma_lut_from_ground_truth (cargo-cult #4 unwind via aggregation policy)"
  - "op-routable #3 operator-routable: paired-CUDA RATIFICATION on NEAREST default vs MODE-per-cell would settle cargo-cult #6 empirical question (~$0.06 per Catalog #246)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: plateau_adjacent
---

# Per-substrate symposium NSCS06 v8 chroma_lut — Wave 5 cargo-cult re-audit + integrated fix

**Lane**: `lane_wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_20260529`
**Cadence**: Wave 5 falls WITHIN the 14-day per-substrate symposium window per Catalog #325 (window: 2026-05-21 → 2026-06-04). Per Catalog #325 the 14-day window is operator-routed; Wave 5 is the canonical iteration of the prior 2026-05-21 + 2026-05-28 symposium memos.

## 6-step contract per Catalog #325

### Step 1 — Cargo-cult audit per Catalog #303

Re-verified against 2026-05-26 12-assumption audit + Wave 5 NEW findings:

| # | Assumption | Status | Wave 5 action |
|---|---|---|---|
| 1 | SegNet stride-2 stem destroys chroma below (256,192) | HARD-EARNED | PRESERVE |
| 2 | 16-level grayscale quantization is empirically optimal | CARGO-CULTED (predecessor) | UNWIND-TEST-DEFERRED to MLX iteration arm (predecessor `enumerate_cargo_cult_unwind_arms`) |
| 3 | Trainer routes through canonical sha256_truncated helper | **CARGO-CULTED-FIXED (Wave 5 catch)** | **FIXED via canonical helper wire-in** |
| 4 | Per-(level,class) MEDIAN aggregation is optimal | CARGO-CULTED (predecessor + Wave 5) | RECOMMENDED-WAVE-6 op-routable |
| 5 | L0 SCAFFOLD cls=0 uniform inflate | HARD-EARNED-FIXED (v3 CH08 landing 2026-05-26) | PRESERVE |
| 6 | **Strided NEAREST top-left point-sample is the only canonical cls_lowres downsample policy** | **CARGO-CULTED (Wave 5 NEW)** | **FIXED via canonical helper + MODE opt-in arm** |
| 7 | 6-DOF affine warp preserves v8 distinguishing feature | HARD-EARNED (v7 unwind) | PRESERVE |
| 8 | Cross-substrate sharing of derive_codebook_from_seed | HARD-EARNED for shape-agnostic; CARGO-CULTED for distribution-agnostic | UNWIND-ARM landed (`gt_distribution_matched_seed`) |
| 9 | SegNet stride-2 + per-(level,class) chroma → SegNet argmax sensitive | CARGO-CULTED-CRITICAL (predecessor) | MLX-LOCAL probe extends to validate |
| 10 | 4096-byte LUT footprint with zero-padding is byte-stable for #26 | HARD-EARNED | PRESERVE |
| 11 | Compress-side LUT derivation acceptable wall-clock cost | HARD-EARNED | PRESERVE |
| 12 | CH08 archive grammar byte-stable across runs | HARD-EARNED | PRESERVE |
| 13 | Predicted-band rate-axis-only is EMPIRICALLY-RELEVANT | CARGO-CULTED-CRITICAL (predecessor) | DEFER-PENDING-paired-smoke per Catalog #324 |

### Step 2 — 9-dim checklist evidence

| Dim | Status | Evidence |
|---|---|---|
| UNIQUENESS | PASS | The (levels, classes, 3) LUT shape + MODE-per-cell downsample policy are NSCS06 v8 substrate-unique |
| BEAUTY+ELEGANCE | PASS | Canonical helper API ~370 LOC reviewable in 30 sec; argparse flag is one line |
| DISTINCTNESS | PASS | Sister of cargo-cult #4 but at a DIFFERENT helper surface (cls_lowres vs chroma_lut aggregation) |
| RIGOR | PASS | Catalog #229 PV + Wave 5 re-audit + 226 tests pass + Catalog #287 + #323 Provenance |
| OPTIMIZATION-PER-TECHNIQUE | PASS | MODE policy preserves boundary dominant class; canonical helper sister-extincts cargo-cult #3 wire-in bypass |
| STACK-OF-STACKS-COMPOSABILITY | PASS | NEAREST default preserves prior dispatch byte parity (composes with Wave N+42 PR-95-parity packet + Wave N+49 PR111-candidate composite) |
| DETERMINISTIC-REPRODUCIBILITY | PASS | Both helpers deterministic + byte-stable; new tests pin invariants |
| EXTREME-OPTIMIZATION-PERFORMANCE | N/A at this layer | Helper is O(n_pairs * gh * gw * num_classes) bincount; bounded |
| OPTIMAL-MINIMAL-CONTEST-SCORE | PENDING-PAIRED-SMOKE per Catalog #324 | MODE arm empirical question deferred to operator-routable paired-CUDA dispatch |

### Step 3 — Observability surface per Catalog #305

- **Per-call telemetry**: `mode_vs_nearest_agreement_fraction` ∈ [0, 1] surfaces empirical relevance of cargo-cult #6 unwind for actual cls_full distribution at every call site.
- **Verdict dataclass `ClsLowresDownsampleVerdict`** carries policy + shape + sha256 + agreement_fraction + canonical non-promotable Provenance per Catalog #287 + #323.
- **Per-cell counterfactual**: enabling MODE policy and comparing archive sha256 against NEAREST archive sha256 directly answers "did MODE materially change archive bytes?" at $0 cost.
- **Cite-able**: every output verdict carries `policy` + `axis_tag` + `evidence_grade='research-signal'` so downstream consumers cannot promote without paired empirical anchor.

### Step 4 — T1 sextet pact deliberation + 4 grand-council additions

Inner sextet: Shannon + Dykstra + Yousfi + Fridrich + Contrarian + AssumptionAdversary.

Grand-council additions per Wave 5 topic relevance:
- **Rudin** (interpretable ML co-lead): canonical disambiguator MODE vs NEAREST is a Rashomon-class research-signal surfaceable via the per-call agreement_fraction metric.
- **Daubechies** (wavelet multi-scale co-lead): cls_lowres downsample IS a multi-scale partition discovery surface; per-cell MODE preserves the canonical Daubechies "coarsest-dominates-on-disagreement" rule per `[[wavelet-multi-scale-ranker-canonical-routing]]`.
- **NSCS06-v6-v7-author-cite**: the 44% v6→v7 cargo-cult-unwind methodology applied to canonical helper bypass at the meta-layer.
- **PR95Author**: validates that canonical-helper sister-extinction discipline matches PR95-parity discipline per `[[hnerv-leaderboard-implementation-parity-discipline]]` L7 (bolt-on vs substrate-engineering split).

Quorum: 10 voices (6 sextet + 4 grand council) >> 6-quorum required for T1.

### Step 5 — Per-substrate reactivation criteria pinned

Per CLAUDE.md "Forbidden premature KILL":
1. **Wave 6** ($0 MLX-LOCAL): extend canonical helper pattern to `build_chroma_lut_from_ground_truth` (cargo-cult #4 unwind)
2. **OPERATOR-ROUTABLE** (~$0.06 paid Modal): paired-CUDA RATIFICATION on NEAREST default vs MODE-per-cell would empirically settle cargo-cult #6
3. **AUTOMATIC**: re-symposium re-deliberation per Catalog #325 14-day window or Wave 6 cargo-cult #4 unwind landing
4. **CONDITIONAL**: if Wave 7+ identifies additional NEW cargo-culted assumptions in v8, sister Wave-style re-audit

### Step 6 — Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status`: `pending_post_training` (preserved from 2026-05-21 recipe).
`predicted_band_validation_reactivation_criterion`: post-training Tier-C re-measurement on landed paired smoke archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`.

Wave 5 does NOT change the predicted band. The new canonical equation `cls_lowres_downsample_policy_boundary_preservation_v1` has its OWN predicted band `[-0.001, +0.001]` for the MODE arm pending paired-smoke per Catalog #324.

## CLAUDE.md compliance per Catalog #325 sister catalogs

- **Catalog #290 canonical-vs-unique decision per layer**: documented for both Wave 5 helpers in their module docstrings
- **Catalog #294 9-dim checklist evidence**: Step 2 above
- **Catalog #303 cargo-cult audit per assumption**: Step 1 above
- **Catalog #305 observability surface**: Step 3 above
- **Catalog #296 Dykstra-feasibility predicted band**: NEAREST baseline + MODE alternative = Dykstra-feasibility intersection check (both feasible at same byte cost)
- **Catalog #292 per-deliberation assumption surfacing**: AssumptionAdversary verdicts above
- **Catalog #300 council deliberation v2 frontmatter**: this memo's frontmatter
- **Catalog #346 canonical roster validation**: 10-voice sextet + 4-grand-council quorum satisfies T1 + topic-relevant specialists
- **Catalog #363 empirical-verification-status taxonomy**: all 3 AssumptionAdversary verdicts carry `VERIFIED_VIA_SOURCE_INSPECTION`

## Cross-references

- Wave 5 landing memo: `.omx/research/wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_landed_20260529.md`
- Wave 5 retroactive sweep: `.omx/research/retroactive_sweep_for_wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_20260529T213000Z.md`
- Prior symposium memos: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` + `.omx/research/council_t2_composite_nscs06_v8_plus_compound_c_pr111_candidate_per_substrate_symposium_20260528.md`
- Prior cargo-cult audit: `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Canonical equations registry: `cls_lowres_downsample_policy_boundary_preservation_v1`
- Canonical anti-patterns registry: `cls_lowres_nearest_strided_without_empirical_vs_mode_v1`
- Council deliberation anchor: `wave_5_nscs06_v8_chroma_lut_cargo_cult_re_audit_plus_fix_20260529`
- Probe outcome ledger: `wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_20260529`
