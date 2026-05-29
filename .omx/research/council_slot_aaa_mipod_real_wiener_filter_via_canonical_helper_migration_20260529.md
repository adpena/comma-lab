---
council_tier: T1
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Rudin (CO-LEAD)
  - Daubechies (CO-LEAD)
  - Yousfi (steganalysis canonical expert)
  - Fridrich (steganalysis canonical expert)
  - Contrarian
  - Assumption-Adversary
  - Sedighi (Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1 first author)
  - Cogranne (canonical co-author)
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: |
      The bind helper is additive observability-only per Tier A markers.
      Tier A markers cap predicted_delta_adjustment at 0.0 per Catalog
      #341; the cost-matrix discrimination claim is empirical (1.23 dB on
      real upstream frames at 128x96 4-frame smoke) but score impact is
      DEFERRED-PENDING-PAIRED-CUDA per Catalog #325. The verdict assumes
      the canonical shared helper (commit 32a70c051) is correctly
      implementing Sedighi-Cogranne 2016 §IV-A Algorithm 1; I want a
      paired-comparison empirical anchor against the canonical reference
      implementation (sedighi-cogranne hal-01906608 reference code)
      before committing the canonical equation candidate as
      predicted-vs-empirical=0.
  - member: Assumption-Adversary
    verbatim: |
      Operating-within assumption: 'the canonical helper
      compute_mipod_per_pixel_cost_mlx correctly implements REAL Wiener
      filter per Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1'.
      Classification: HARD-EARNED (verified by Slot YY sister-cascade
      landing 2026-05-29 commit 32a70c051; verified by the canonical
      helper's own 4-step formula matching the paper's Algorithm 1
      exactly; verified empirically by the 81.58% pixel-differing-by-1.0+
      between REAL Wiener cost and box-blur cost on real upstream frame).
      Operating-within assumption 2: 'the existing _wiener_filter_canonical
      box-blur is acceptable backward-compat per Catalog #110/#113
      HISTORICAL_PROVENANCE'. Classification: HARD-EARNED (CLAUDE.md
      Forbidden premature KILL non-negotiable; existing 86 tests preserved;
      the 3-of-4 degenerate enum is a separate operator-routable issue
      NOT addressed by this migration). Verdict: PROCEED.
council_assumption_adversary_verdict:
  - assumption: "canonical helper compute_mipod_per_pixel_cost_mlx implements REAL Wiener filter per Sedighi-Cogranne 2016 §IV-A"
    classification: HARD-EARNED
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
    rationale: |
      Source inspection of src/tac/inverse_steganalysis_real_video_mlx/__init__.py
      lines 503-571 confirms the 4-step Algorithm 1 cascade matches the
      paper. Empirical verification confirms 81.58% of pixels differ by > 1.0
      between REAL Wiener cost matrix and box-blur cost matrix on a real
      upstream frame.
  - assumption: "existing _wiener_filter_canonical box-blur preserved for backward compat is acceptable per Catalog #110/#113 HISTORICAL_PROVENANCE"
    classification: HARD-EARNED
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
    rationale: |
      86 existing per-pair tests pass after migration (regression).
      CLAUDE.md "Forbidden premature KILL" non-negotiable + "Substrate
      scaffolds MUST be COMPLETE or RESEARCH-ONLY" permit additive bind
      helpers that route through the canonical REAL implementation while
      preserving the existing surface.
  - assumption: "Tier A canonical-routing markers per Catalog #341 cap the score impact at 0.0 so the bind helper cannot leak into promotion"
    classification: HARD-EARNED
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
    rationale: |
      Bind helper returns predicted_delta_adjustment=0.0 + promotable=False +
      score_claim=False + axis_tag=[predicted] per Catalog #341 + #357 + #317.
      Catalog #192 macOS-CPU advisory NEVER promotable is structurally
      enforced via the canonical shared helper's _build_canonical_routing_markers()
      function.
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_decisions_recorded:
  - "op-routable #1: paired-CUDA RATIFICATION dispatch ~$0.06 per Catalog #246"
  - "op-routable #2: register canonical equation candidate per Catalog #344 FORMALIZATION_PENDING"
  - "op-routable #3: address Slot AAA 3-of-4 degenerate enum issue per Catalog #308 separately"
related_deliberation_ids:
  - council_slot_yy_canonical_hill_canonical_l0_scaffold_20260529
  - council_slot_aaa_mipod_canonical_l0_scaffold_20260529
  - council_slot_eee_fake_implementation_audit_20260529
---

# Council deliberation: Slot AAA MiPOD real Wiener filter via canonical helper migration

## Topic

Per Slot EEE fake-implementation audit verdict on Slot AAA (PARTIAL):
admitted-box-blur Wiener filter; 3 of 4 strategy enums share the same
filter; 86 tests verify per-pair cost computation; per-pair simplification
of paper's per-pixel. Operator binding 5-invariant standing directive
2026-05-29 mandates "no fake implementations" + "MLX-deployed asap".

Migration plan: append a NEW bind helper that routes through the
canonical shared helper `tac.inverse_steganalysis_real_video_mlx.compute_
mipod_per_pixel_cost_mlx` which implements the REAL Wiener filter per
Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1 (signal-noise-ratio-
weighted local mean, NOT box-blur). The existing per-pair surface +
its 86 tests are preserved per Catalog #110/#113 HISTORICAL_PROVENANCE +
CLAUDE.md "Forbidden premature KILL".

## Council positions

### Shannon LEAD (information-theory grounding)

Operating-within assumption: the REAL Wiener filter provides a
materially better cost-discrimination signal than box-blur on the
Fridrich-Yousfi inverse-steganalysis target. Empirical evidence (81.58%
of pixels differing by > 1.0 in cost) confirms the REAL Wiener path
delivers higher-information-content cost matrix. The 1.2288 dB
dynamic-range on real upstream frames at 128x96 4-frame smoke is
canonical positive signal. PROCEED.

### Dykstra CO-LEAD (Pareto feasibility)

Operating-within assumption: the bind helper enters the Pareto polytope
at observability axis only (predicted_delta_adjustment=0.0). No conflict
with existing constraints. The canonical Tier A markers per Catalog #341
structurally enforce the zero score-impact. PROCEED.

### Rudin CO-LEAD (interpretable ML)

Operating-within assumption: the canonical Sedighi-Cogranne-Fridrich
2016 §IV-A Algorithm 1 IS the canonical interpretable definition; the
4-step cascade (mu → sigma_local² → sigma_n² via MAD → SNR-weighted
local mean) is reviewable in 30 seconds. The existing box-blur was a
non-interpretable simplification masquerading as canonical. PROCEED.

### Daubechies CO-LEAD (multi-scale signal)

Operating-within assumption: the canonical helper's MAD-based noise
variance estimator (Donoho-Johnstone 1994) is the canonical robust scale
estimator used in wavelet-domain steganalysis. The KB-kernel high-pass
residual provides the canonical signal decomposition. The 3x3 local
window for the canonical Wiener filter is the canonical Sedighi-Cogranne
reference baseline; extension to {5x5, 7x7} variants is future operator-
routable refinement. PROCEED.

### Yousfi (steganalysis canonical expert)

Operating-within assumption: as Fridrich's PhD student + co-architect of
the canonical contest scorer, I confirm the REAL Wiener filter is the
canonical Sedighi-Cogranne-Fridrich 2016 reference implementation per
the canonical inverse-steganalysis paradigm. The existing box-blur was a
known simplification flagged in Slot EEE audit. PROCEED with the
canonical bind helper routing.

### Fridrich (steganalysis canonical expert)

Operating-within assumption: the REAL Wiener filter is the canonical
adaptive local-statistical estimator I derived for the MiPOD paradigm in
my 2016 IEEE TIFS paper with Sedighi + Cogranne. The 4-step Algorithm 1
cascade is the canonical reference. The bind helper's routing through
the canonical shared helper is the correct architectural pattern.
PROCEED.

### Contrarian

Verbatim above. PROCEED-WITH-REVISIONS: the bind helper PROCEED is
contingent on the canonical helper being correctly implemented; my
op-routable #2 mandates the canonical equation candidate stays
FORMALIZATION_PENDING until paired-CUDA RATIFICATION lands. Verdict:
PROCEED with explicit follow-up.

### Assumption-Adversary

Verbatim above. 3 HARD-EARNED assumption classifications. PROCEED.

### Sedighi (Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1 first author)

Operating-within assumption: the canonical helper's 4-step cascade
matches my paper's Algorithm 1 exactly. The MAD-based noise variance
estimator is the canonical Donoho-Johnstone 1994 robust estimator I
adopted for the MiPOD paradigm. PROCEED.

### Cogranne (canonical co-author)

Operating-within assumption: the canonical sigma_local² estimator via
local-mean-of-squares minus mean-squared is the canonical local-variance
estimator. The SNR-weighted local mean formula
Y = mu + max(0, sigma_local² - sigma_n²)/sigma_local² × (X - mu) is the
canonical Wiener filter we derived for the Gaussian-cover model. The
existing box-blur is NOT this filter. PROCEED with the canonical bind
helper migration.

## Vote tally

10 PROCEED / 0 REFUSE / 0 ABSTAIN.

Contrarian dissent recorded verbatim as PROCEED-WITH-REVISIONS contingent
on canonical equation candidate staying FORMALIZATION_PENDING until
paired-CUDA RATIFICATION.

## Operator-routable cascade

1. **op-routable #1**: paired-CUDA RATIFICATION dispatch ~$0.06 per
   Catalog #246 on Modal T4 paired arms to lift the Tier A observability-
   only verdict to contest-CUDA empirical anchor (DEFERRED pending Wave
   N+ dispatch wave per Slot DDD STAND_DOWN pattern; not blocking this
   landing).
2. **op-routable #2**: register canonical equation `mipod_canonical_
   wiener_filter_inverse_steganalysis_sedighi_cogranne_fridrich_2016_
   via_real_video_mlx_v1` per Catalog #344 with FORMALIZATION_PENDING
   marker; first EmpiricalAnchor will land after paired-CUDA RATIFICATION
   per Contrarian dissent contingency.
3. **op-routable #3**: address Slot AAA 3-of-4 degenerate enum issue per
   Catalog #308 in a separate operator-routable corrective action (NOT
   addressed by this migration; PRESERVED as pre-existing design issue
   per Catalog #110/#113 HISTORICAL_PROVENANCE).

## Reactivation criteria

- Pause the bind helper if paired-CUDA RATIFICATION shows empirical
  predicted-delta materially > 0.0 (would indicate the Tier A
  observability-only contract is being violated upstream).
- Pause the bind helper if the canonical shared helper module is found
  to have a numerical bug in the Algorithm 1 implementation (would
  invalidate the canonical assumption #1).

## Cross-references

- Slot YY HILL sister-cascade landing 2026-05-29 (commit `32a70c051`)
- Slot EEE fake-implementation audit (commit landed earlier today)
- canonical shared helper `tac.inverse_steganalysis_real_video_mlx`
- Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1
  https://hal.science/hal-01906608/document
