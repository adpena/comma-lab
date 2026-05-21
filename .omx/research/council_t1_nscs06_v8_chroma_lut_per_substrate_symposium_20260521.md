---
council_tier: T1
council_attendees:
  - Shannon
  - Carmack
  - Hotz
  - Daubechies
  - Mallat
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Assumption-Adversary
    verbatim: |
      The cargo-cult audit per Catalog #303 enumerates 7 assumptions but only
      3 are flagged CARGO-CULTED and queued for UNWIND-TEST. The remaining 4
      HARD-EARNED classifications inherit empirical confidence from sister
      substrates (v7 cargo-cult-unwind methodology + sister PROCEDURAL VARIANT
      pattern). The risk: if the v8 first paired smoke drifts >2x from
      predicted ΔS = -0.002706, we will not know which assumption layer
      drifted because the test plan does not separately probe assumptions
      4-7. Recommend: REVISION #4 below adds a per-assumption ablation
      ladder for the first paired smoke harvest.
council_assumption_adversary_verdict:
  - assumption: "16-level luma quantization captures chroma-relevant variation"
    classification: CARGO-CULTED
    rationale: "Inherited from canonical AV1 codecs without empirical
      validation that NSCS06 v8's specific compress-time signal axis
      benefits from 16 levels vs other choices"
  - assumption: "Per-(level, class) LUT median is the optimal aggregation"
    classification: CARGO-CULTED
    rationale: "Inherited from v7 per-class median pattern without testing
      against per-class mode / trimmed mean / k-medoids cluster center"
  - assumption: "PCG64 seed -> uniform-distributed LUT bytes matches GT chroma distribution"
    classification: CARGO-CULTED
    rationale: "Inherited from canonical PROCEDURAL VARIANT pattern; the
      canonical equation #26 IN-DOMAIN context membership IS the empirical
      anchor that grounds this assumption but requires the first paired
      smoke to confirm operationally"
  - assumption: "Catalog #205 inflate device-fork produces byte-identical raw frames"
    classification: HARD-EARNED
    rationale: "Sister v7 inflate produces byte-stable raw bytes; v8 inflate
      copies the canonical Catalog #205 pattern (numpy + Pillow only; no
      CUDA-specific operations)"
  - assumption: "6-DOF affine warp (cargo-cult #4 unwound by v7) preserves v8 distinguishing feature"
    classification: HARD-EARNED
    rationale: "Empirically validated by v7 cargo-cult-unwind 44% improvement
      (105.15 -> 58.89 contest-CPU; commit 4292c8ce2)"
  - assumption: "SCAFFOLD class=0 uniform per-cell mask is acceptable for L0"
    classification: HARD-EARNED
    rationale: "Matches v7 L1 SCAFFOLD pattern; CLS_STREAM consumption
      deferred to L1; explicitly acknowledged as SCAFFOLD-DEFERRED-INTEGRATION"
  - assumption: "Cross-substrate sharing of derive_codebook_from_seed does NOT suppress v8 distinguishing feature"
    classification: HARD-EARNED
    rationale: "The canonical helper is shape-and-dtype-agnostic; v8 derives
      (16, 5, 3) uint8 which is a different shape from grayscale_lut's
      (256,), DP1's basis tensor, or VQ-VAE's (K, D) codebook. The PCG64
      seed -> LUT bytes derivation does not constrain the substrate's
      distinguishing chroma-LUT shape"
council_decisions_recorded:
  - "op-routable #1: per-substrate symposium 14-day window opens 2026-05-21; reactivation criterion = PROCEED-unconditional verdict from a T1 working group on or before 2026-06-04"
  - "op-routable #2: pre-symposium dispatch BLOCKED via recipe `research_only: true` + `dispatch_enabled: false` per Catalog #240"
  - "op-routable #3: first paired smoke (post-PROCEED-unconditional) costs ~$0.50 Modal T4 per recipe cost_band; queues post-symposium"
  - "op-routable #4 (REVISION per Assumption-Adversary VETO): first paired smoke harvest MUST include per-assumption ablation ladder for CARGO-CULTED assumptions 1-3 (luma quantization levels: 8 / 16 / 32; aggregation: median / mode / k-medoids; PCG64 vs xorshift vs lcg generator kind)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3
  - council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2
---

# T1 per-substrate symposium: nscs06_v8 chroma-LUT

**Date:** 2026-05-21
**Tier:** T1 Working Group (per CLAUDE.md "Council hierarchy: 4-tier protocol")
**Attendees:** Shannon LEAD + Carmack + Hotz + Daubechies + Mallat + Assumption-Adversary
**Verdict:** PROCEED_WITH_REVISIONS (4 binding revisions; see below)
**Substrate:** `nscs06_v8_chroma_lut` (per `src/tac/canonical_equations/procedural_codebook_savings.py:102` `_INCLUDED_CONTEXTS`)
**Window:** 2026-05-21 -> 2026-06-04 (14 days; per Catalog #325 per-substrate symposium gating)
**Sister design memo:** `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`

This memo satisfies Catalog #325 6-step canonical contract:

1. **Cargo-cult audit per Catalog #303** — see frontmatter `council_assumption_adversary_verdict` + sister design memo §"Cargo-cult audit per assumption"
2. **9-dimension success checklist evidence per Catalog #294** — see sister design memo §"9-dimension success checklist evidence"
3. **Observability surface declaration per Catalog #305** — see sister design memo §"Observability surface"
4. **T1 working-group deliberation** with Shannon LEAD + Carmack + Hotz + Daubechies + Mallat + Assumption-Adversary (see attendees + dissent above)
5. **Per-substrate reactivation criteria** — see frontmatter `council_decisions_recorded`
6. **Catalog #324 post-training Tier-C validation discipline** — `predicted_band_validation_status: pending_post_training` declared in recipe; reactivation criterion = "post-training Tier-C re-measurement on landed paired smoke archive sha via tools/mdl_scorer_conditional_ablation.py --tier c"

---

## Section 1. Shannon LEAD position

**Operating-within assumption:** the canonical equation #26 closed form `ΔS = -25 * (N - K) / 37_545_489` IS the rate-axis Shannon-bound prediction for procedural codebook substitution in IN-DOMAIN contexts.

**Position:** PROCEED. The v8 substrate has the cleanest canonical-equation-grounded prediction of any procedural-variant substrate landed so far. The bytes-saved match the prediction byte-for-byte at L0 smoke (4064 bytes). The remaining empirical question is whether the seg + pose axes drift; the per-substrate symposium 14-day window is structurally adequate for the first paired smoke to land within budget.

**Vote:** PROCEED.

## Section 2. Daubechies + Mallat CO-LEAD position

**Operating-within assumption:** the multi-scale partition discovery framing per CASCADE COMPRESSION symposium PRIORITY 3 + Mallat wavelet detail-subband analog framing positions chroma channels as the wavelet detail-subband analog where the contest scorer's perceptual sensitivity is LOWEST.

**Position:** PROCEED. The v8 LUT shape `(16 levels x 5 classes x 3 RGB)` IS the canonical hierarchical-coarse-gates-fine wavelet structure: coarse-scale class index gates the fine-scale (level, channel) lookup. The structural elegance + canonical equation #26 IN-DOMAIN membership + Atick-Redlich retinal redundancy reduction (chroma is least-perceptually-relevant) jointly justify the substrate as second-priority IN-DOMAIN per the symposium ratification.

**Vote:** PROCEED (with REVISION #2 below to add Daubechies-style multi-scale-feasibility check on the first paired smoke).

## Section 3. Carmack + Hotz position

**Operating-within assumption:** every layer should be reviewable in 30 seconds (Carmack) + canonical helpers shared where they serve, forked where they suppress (Hotz / UNIQUE-AND-COMPLETE-PER-METHOD operating mode).

**Position:** PROCEED. The v8 substrate package is ~700 LOC across 4 files (architecture + archive + inflate + procedural_variant + substrate_contract). Inflate is ~120 LOC with the chroma LUT lookup + 6-DOF affine warp. Every layer reviewable in 30s; canonical helpers shared via `tac.procedural_codebook_generator.derive_codebook_from_seed` + `tac.substrates._shared.trainer_skeleton`; substrate-specific (chroma LUT shape + per-pixel lookup) UNIQUE per Catalog #290.

**Vote:** PROCEED (with REVISION #3 below to add Carmack MVP-first 5-step recipe verification before first paired smoke).

## Section 4. Assumption-Adversary VETO + REVISIONS

**Operating-within assumption:** every prediction is CARGO-CULTED until empirically validated; the canonical equation #26 prediction is HARD-EARNED on the rate-axis but CARGO-CULTED on the seg + pose axes for v8 specifically (the equation does not predict seg/pose impact of the LUT-shape choice).

**Position:** PROCEED_WITH_REVISIONS. The 4 binding revisions extend the assumption-coverage of the first paired smoke harvest so a >2x drift can be attributed to the right assumption layer rather than triggering a generic falsification.

**REVISION #1:** The first paired smoke harvest MUST include per-assumption ablation ladder for CARGO-CULTED assumptions 1-3 (luma quantization levels: 8 / 16 / 32; aggregation: median / mode / k-medoids; PCG64 vs xorshift vs lcg generator kind). Cost estimate: 3 ablation arms x $0.50 each = $1.50 incremental over base $0.50 smoke = $2.00 total.

**REVISION #2:** Add Daubechies-style multi-scale-feasibility check on the first paired smoke per Catalog #296: verify that the wavelet-style hierarchical-coarse-gates-fine LUT structure produces additive seg+pose contributions consistent with the Dykstra-feasibility intersection of (rate <= R) AND (seg <= S) AND (pose <= P) constraints. If non-additive: pivot to UNWIND-TEST per cargo-cult-audit assumptions 1-3.

**REVISION #3:** Add Carmack MVP-first 5-step recipe verification before first paired smoke: (a) verify CH08 v2 archive parses cleanly on Modal worker; (b) verify inflate roundtrip produces canonical raw bytes count; (c) verify chroma LUT lookup correctness against a known synthetic seed; (d) verify byte-mutation distinguishing-feature smoke per Catalog #272 passes; (e) verify Catalog #205 inflate-device-fork passes for CPU + CUDA paths.

**REVISION #4:** Per-assumption ablation table format MUST be machine-readable JSON output to `.omx/state/nscs06_v8_per_assumption_ablation_<utc>.json` so the cathedral autopilot ranker can consume the verdicts via canonical Provenance per Catalog #287 + #323.

**Vote:** PROCEED_WITH_REVISIONS (the 4 binding revisions are operator-routable conditions for L1 promotion; if revisions are not applied, the symposium falls back to DEFER_PENDING_EVIDENCE).

## Section 5. Composite verdict

**PROCEED_WITH_REVISIONS** with the 4 binding revisions above. Quorum: 6/6 attendees present; PROCEED-with-revisions majority vote 5/6 (Shannon + Daubechies + Mallat + Carmack + Hotz) + 1 PROCEED-with-revisions-binding (Assumption-Adversary).

The 14-day symposium window opens 2026-05-21; reactivation criterion for L1 promotion = PROCEED-unconditional verdict from a follow-on T1 working group on or before 2026-06-04 after the 4 binding revisions are applied to the first paired smoke harvest plan.

Pre-symposium dispatch BLOCKED via:
- recipe `research_only: true` + `dispatch_enabled: false`
- trainer `_full_main` raises NotImplementedError
- Catalog #240 + Catalog #325 STRICT preflight gate refusal

---

## Section 6. Cross-references

- CASCADE COMPRESSION symposium commit `d125af6c3` PRIORITY 3 + Revision #5 (NSCS06 v8 chroma_lut BUILD elevated as second-priority IN-DOMAIN substrate per Daubechies + Mallat multi-scale partition discovery framing)
- HONEST CASCADE-MORTALITY ASSESSMENT commit `d884dd6aa` Rank 2 (HIGH P(actual score reduction) per probabilistic ranking)
- NSCS06 v6 -> v7 cargo-cult-unwind methodology commit `4292c8ce2` (44% improvement empirically validated rescue path; sister inheritance)
- Sister procedural-variant substrates: grayscale_lut (commit `f037d1144`) + DP1 (commit `9cbfa471c`) + VQ-VAE (commit `6fea30f22`)
- Canonical equation #26: `procedural_codebook_from_seed_compression_savings_v1` at `src/tac/canonical_equations/procedural_codebook_savings.py`; `nscs06_v8_chroma_lut` IS in `_INCLUDED_CONTEXTS`
- Sister design memo: `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`

---

## Section 7. Canonical posterior anchor wire-in (Catalog #300 + #346)

This memo's verdict + dissent + assumption-adversary classifications + decisions are appended to the canonical council deliberation posterior via:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521",
    topic="nscs06 v8 chroma-LUT per-substrate symposium per Catalog #325",
    council_tier=CouncilTier.T1,
    council_attendees=("Shannon", "Carmack", "Hotz", "Daubechies", "Mallat", "Assumption-Adversary"),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    deferred_substrate_id="nscs06_v8_chroma_lut",
    # ... dissent + assumption-adversary verdicts + decisions per frontmatter
)
append_council_anchor(record)
```

Per Catalog #300 + #346 the canonical roster validator `tac.canonical_council_roster.validate_council_dispatch_roster(council_attendees, topic_tokens=['substrate', 'chroma_lut', 'nscs06_v8'], tier=CouncilTier.T1)` returns `complete=True` because the T1 working group has 6 attendees (above T1 minimum); the per-substrate symposium discipline gates promotion to L1 not the council quorum.
