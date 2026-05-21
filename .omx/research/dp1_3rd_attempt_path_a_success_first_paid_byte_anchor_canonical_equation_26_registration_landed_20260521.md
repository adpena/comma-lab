---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "registered canonical equation #26 in-domain dp1_codebook_bytes byte-anchor: residual 6.3e-5 HARD-EARNED"
  - "updated Modal call_id ledger: both arms harvested rc=0 evidence_grade=scaffold-only-no-score-claim"
  - "registered Catalog #313 probe-outcome PROCEED non-blocking advisory"
  - "operator-routable: follow-up paired dispatch with DPP_RUN_AUTH_EVAL=1 to promote byte-anchor to contest-axis score-anchor"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - dp1_3rd_attempt_path_a_first_paid_contest_axis_empirical_anchor_registration_20260521
---

# DP1 3rd-attempt PATH A success: first paid byte-anchor for canonical equation #26 registration LANDED 2026-05-21

## Summary

OVERNIGHT-Z-RESUME successfully resumed predecessor crashed at 309s rate-limit. Harvested both arms of DP1 3rd-attempt paired Modal T4 dispatch (PATH A verdict: BOTH arms rc=0 at 2026-05-21T14:37Z). Registered canonical equation #26 IN-DOMAIN `dp1_codebook_bytes` empirical anchor with HARD-EARNED 2σ residual.

## Empirical observations

**Baseline arm** (`fc-01KS5CTJEM7V90152QTWPDX7D2`):
- rc=0, elapsed=2333.03s, scaffold-only-no-score-claim
- `archive.zip` sha `b5ac83d17d4a935564b7836c7534db40329f9de3dbd237fcaeeca2b2a8b10901` 25733 B
- `0.bin` sha `a99054c65402b03a556783d5dc7ebbc428e89c08c5ee9aa8de9d59e3d47a8fbb` 26050 B
- 23 saved artifacts

**Procedural arm** (`fc-01KS5CXQXNSKYACT32WAAMXHKC`):
- rc=0, elapsed=2311.25s, scaffold-only-no-score-claim
- `archive.zip` sha `a2e52986d288f6e388b0a6708148395910670e9973ea0980551d0d5b2c41e6e6` 18269 B
- `0.bin` sha `52b80f250326a8378997caf07a4ce98066a95885ee82e1fde0a779263f657f41` 18927 B
- 24 saved artifacts
- procedural variant provenance: seed sha `35d4ae6c3dd8f7c164a47ee4cf841651148fe95a41982f81ef1244f4bfd96e2b` (32 B); predicted ΔS = -0.005033

## Canonical equation #26 verdict

In-domain context: `dp1_codebook_bytes`

| Metric | Predicted | Empirical (archive_zip) | Empirical (0.bin) | Residual |
|---|---:|---:|---:|---:|
| Bytes saved | 7558 | 7464 | 7123 | -94 / -435 |
| ΔS_rate | -5.033e-3 | -4.970e-3 | -4.743e-3 | -6.30e-5 / -2.90e-4 |

**Verdict**: `HARD_EARNED_within_byte_accounting_residual_under_1e_4_for_codebook_replacement_REPLACEMENT_paradigm_per_canonical_equation_26_in_domain_dp1_codebook_bytes`

Residual archive_zip = 6.30e-5 (well under 1e-3 2σ threshold); residual 0.bin = 2.90e-4 (also under 1e-3). The procedural codebook approach correctly applies canonical equation #26's REPLACEMENT paradigm (not residual-correction hybrid per Catalog #359).

## Critical scope clarification

These arms did **NOT** run `contest_auth_eval.py`. Per harvest_summary.json: `auth_eval_device: "cpu"`, `auth_eval_advisory_only: true`, `score_claim: false`, `promotion_eligible: false`. Per stdout tail: both dispatches ended at "Phase 2 full training complete / Exiting cleanly (rc=0)" with NO auth_eval invocation.

This is therefore a **byte-axis empirical anchor**, NOT a contest-axis score anchor. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable, promotion to contest-axis evidence requires paired CPU+CUDA auth_eval on the EXACT archive bytes (`b5ac83d17d4a93` baseline + `a2e52986d288f6` procedural).

## 4-path verdict

**Verdict: D** = above-frontier-but-canonical-equation-validated. Specifically:

- **NOT-frontier-beating** on either axis: byte-axis comparison alone is not informative for cross-substrate frontier-relevance because DP1 archives (18-26KB) target a fundamentally different rate/distortion regime than FEC6 frontier (178KB) / PR106 latent (186KB). Pose/SegNet distortion is unknown without auth_eval.
- **Canonical equation #26 validated**: the IN-DOMAIN `dp1_codebook_bytes` context produced HARD-EARNED byte-accounting at residual 6.3e-5 — first paid full-training empirical anchor for this context (prior 5 anchors were predicted/advisory or sister contexts).

## Frontier-relevance check per Catalog #316

Current frontier (read-only verification per Catalog #343):
- contest-CPU: 0.192051 on archive sha `6bae0201fb08...` 178517 B (FEC6)
- contest-CUDA: 0.20533 on archive sha `9cb989cef519...` 186876 B (PR106 format0d)

DP1 3rd-attempt PATH A produced NO contest-axis scores; no frontier comparison possible. `reports/latest.md` NOT updated.

## Sister coordination

NO active sister subagents during my work (operator stagger discipline). Predecessor crashed before producing collision-risk edits.

## Operator-routable follow-up

To convert this byte-axis anchor into a contest-axis score anchor:

```bash
# Follow-up paired dispatch with DPP_RUN_AUTH_EVAL=1 enabled
# (current recipe defaults to advisory_only=true; needs env override + paired-eval contract)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_modal_t4_dispatch \
    --paired \
    --env DPP_RUN_AUTH_EVAL=1
```

Expected outcome: paired CPU+CUDA auth_eval JSONs land in harvested artifacts; full Catalog #324 post-training Tier-C validation path opens; contest-axis score-anchor promotion possible.

## Discipline declarations

- Catalog #206: predecessor checkpoint read + own checkpoint emitted (2 total in resume)
- Catalog #117/#157/#174/#235: commit via canonical serializer with POST-EDIT `--expected-content-sha256`
- Catalog #110/#113: APPEND-ONLY HISTORICAL_PROVENANCE — only NEW rows on `modal_call_id_ledger.jsonl` + `probe_outcomes.jsonl` + `canonical_equations_registry.jsonl`
- Catalog #245: canonical Modal call_id ledger updated via `update_call_id_outcome`
- Catalog #313: probe-outcomes ledger PROCEED registered
- Catalog #316: frontier-pointer comparison performed (read-only); no `reports/latest.md` update needed
- Catalog #323: canonical Provenance attached to empirical anchor via `build_provenance_for_research_sidecar` with explicit `axis_tag=[byte-budget rate-axis only]`
- Catalog #344: canonical equation #26 anchor registered via `update_equation_with_empirical_anchor`
- Catalog #287: NO docstring overstatement — all numbers tagged `[byte-budget rate-axis only]` / `[predicted]`
- Catalog #300 v2 frontmatter: T1 worker, PROCEED verdict, apparatus_maintenance contribution

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A (no per-element sensitivity update; byte-anchor only)
2. Pareto constraint: N/A (no scorer distortion measured)
3. Bit-allocator: ACTIVE via canonical equation #26 anchor (registered ΔS-per-byte predictor strengthens bit-allocator prior for `dp1_codebook_bytes` IN-DOMAIN)
4. Cathedral autopilot dispatch hook: ACTIVE via auto-discovered consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` (canonical equation #26 has registered consumer per Catalog #335)
5. Continual-learning posterior: ACTIVE via fcntl-locked append to `.omx/state/canonical_equations_registry.jsonl` + `.omx/state/probe_outcomes.jsonl` + `.omx/state/modal_call_id_ledger.jsonl`
6. Probe-disambiguator: ACTIVE — the canonical equation #26 IN-DOMAIN `dp1_codebook_bytes` context's HARD-EARNED verdict IS the disambiguator between (a) byte-anchor-validates-equation-but-no-score-claim vs (b) needs-contest-axis-promotion

## Files touched

- `.omx/state/modal_call_id_ledger.jsonl` (2 outcome rows appended)
- `.omx/state/probe_outcomes.jsonl` (1 PROCEED row appended)
- `.omx/state/canonical_equations_registry.jsonl` (1 anchor_appended event appended for equation `procedural_codebook_from_seed_compression_savings_v1`)
- `.omx/state/subagent_progress.jsonl` (own checkpoint rows)
- `.omx/research/dp1_3rd_attempt_path_a_success_first_paid_byte_anchor_canonical_equation_26_registration_landed_20260521.md` (THIS memo)

## Lane

`lane_overnight_z_dp1_3rd_attempt_path_a_first_paid_contest_axis_empirical_anchor_registration_20260521` L1 (impl_complete + memory_entry).

Cost: $0 incremental (paid dispatches already complete at 2026-05-21T13:50-14:32Z). Wall-clock for resume: ~15 minutes.
