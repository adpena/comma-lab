---
name: z7-mamba-2-multi-week-path-forward-20260518
council_tier: T1
council_attendees: [Hafner, Schmidhuber, Tao, Hotz, Hassabis, Contrarian, Assumption-Adversary]
council_quorum_met: false
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Hotz
    verbatim: "Per-blocker resolution plan should not pretend Wave N+1 council convocation is free. Operator-attention budget is finite per Catalog #300; sequencing must respect T3 budget ≤3/week."
council_assumption_adversary_verdict:
  - assumption: "Wave N+1/N+2/N+3/N+4 sequential cadence is feasible within ~30-day staleness window per Catalog #298"
    classification: HARD-EARNED-PARTIAL
    rationale: "Wave N+1 + N+2 can land within 7-14 days IF council convocation prompt + grad-clip CLI flag + Modal smoke fire in parallel. Wave N+3 + N+4 require empirical anchors so depend on N+2 outcome; can extend 14-21 days."
council_decisions_recorded:
  - "8 dispatch_blocker resolutions enumerated"
  - "4-wave sequencing finalized with cost-per-wave + operator-routable decisions"
  - "Recommended Wave N+1 immediate action: convocation prompt + MPS proxy + design-space ratification"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: z7_mamba2
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
horizon_class: asymptotic_pursuit
---

# Z7-Mamba-2 multi-week path forward — 8 blockers + 4-wave sequencing

**Lane**: `lane_z7_mamba_2_stability_multi_week_path_forward_20260518` L1
**Sibling design-space memo**: `.omx/research/z7_mamba_2_stability_design_space_20260518.md` (5 candidates ranked)
**Sibling symposium DRAFT**: `.omx/research/council_t3_z7_mamba_2_stability_path_forward_symposium_DRAFT_20260518.md`

## Per-blocker resolution plan (8 dispatch_blockers per recipe)

| # | Recipe blocker (verbatim) | Classification | Resolution path | Editor effort | GPU cost |
|---|---|---|---|---|---|
| 1 | `z7_mamba2_trainer_full_main_raises_NotImplementedError_per_catalog_240` | INFRASTRUCTURE-STALE-CLAIM | **STALE — already resolved**. `experiments/train_substrate_time_traveler_l5_z7_mamba2.py:856` `_full_main` IS implemented (full PR95-paradigm: real pair decode + Mamba-2 autoregression + Z6 PixelShuffle decoder + score-aware loss + Z7MCM2 archive + deterministic ZIP + inflate runtime + optional verify + static-control). Predecessor audit + design memo §17 stale. Action: remove blocker #1 from recipe `dispatch_blockers` list per Catalog #240. Sister Slot 5 META-bug audit will document the stale-trainer-state cargo-cult. | 5 min recipe edit | $0 |
| 2 | `z7_mamba2_substrate_module_absent_pre_build_per_z7_symposium_revision_6` | INFRASTRUCTURE-STALE-CLAIM | **STALE — already resolved**. `src/tac/substrates/time_traveler_l5_z7_mamba2/` exists with architecture.py + archive.py + __init__.py. Per `ls`: 3 modules present. Action: remove blocker #2 from recipe. | 5 min recipe edit | $0 |
| 3 | `z7_mamba2_dispatch_requires_z7_gru_wave_2_disambiguator_outcome_per_revision_1` | DESIGN-DEPENDENCY | Z7-GRU Wave 2 paired identity-disambiguator is cross-substrate prerequisite per Quantizr Revision #6. Action: register cross-substrate probe outcome `z7_gru_wave_2_disambiguator_pending` per Catalog #313. Per Catalog #313 staleness window: 30-day default. **Operator-routable**: convene Z7-GRU Wave 2 council session in parallel with Z7-Mamba-2 stability sweep (independent operator-attention budget). | 30 min probe-outcome registration + cross-reference cite | $0 |
| 4 | `z7_mamba2_dispatch_requires_wave_n_plus_1_council_after_z7_gru_outcome` | OPERATOR-ROUTABLE | Wave N+1 T3 grand council deliberation required after Z7-GRU outcome lands. Action: DRAFT memo at `.omx/research/council_t3_z7_mamba_2_stability_path_forward_symposium_DRAFT_<utc>.md` per Catalog #325 6-step contract + Catalog #300 v2 frontmatter. **Operator-routable**: ratify DRAFT into full council convocation when T3 cadence budget allows (≤3/week per Catalog #300; current week's T3s logged per `tools/audit_council_tier_cadence.py`). | 1.5h DRAFT + operator convocation 90 min | $0 |
| 5 | `z7_mamba2_beta_ib_parameter_requires_c6_ibps_phase_2_empirical_beta_anchor_per_revision_5` | EMPIRICAL-CROSS-SUBSTRATE-DEPENDENCY | C6 IBPS Phase 2 β-IB-Lagrangian empirical anchor per Hafner Revision #5 inheritance. Per `tac.probe_outcomes_ledger`: register cross-substrate dependency probe outcome. Per CLAUDE.md "Forbidden premature KILL": Z7-Mamba-2 β-parameter can use cold-start guess for Wave N+1 stability sweep (no β-IB inheritance needed for stability-fix smoke; β-tuning is Wave N+3 promotion concern). Action: register probe outcome with reactivation criterion = "C6 IBPS Phase 2 empirical β-optimal lands OR Wave N+3 ratifies cold-start β for stability-fix scope". | 30 min probe-outcome registration | $0 |
| 6 | `z7_mamba2_wave_2_probe_requires_paired_exact_eval_json_from_probe_z7_mamba2_temporal_coherence_vs_static_capacity_disambiguator` | DESIGN-PROBE-DISAMBIGUATOR | Per design memo §15 codex hardening: `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` already exists + hardened against false-authority from `score_claim_valid: false` source JSONs. Action: Wave N+2 produces paired exact-eval JSON via canonical probe-disambiguator tool. Cost INCLUDED in Wave N+2 grad-clip sweep envelope ($5). | 0 (existing tool) | INCLUDED in Wave N+2 |
| 7 | `z7_mamba2_requires_same_archive_bytes_identity_disambiguator_before_full_dispatch` | DESIGN-IDENTITY-DISAMBIGUATOR | Per design memo §7 + §11: identity-predictor ablation flag `--identity-predictor` already exists in trainer argparse (per TIER_1_OPERATOR_REQUIRED_FLAGS). Same-archive-bytes paired comparison fires when both Mamba-2 + identity runs share same encoder/decoder/latent weights but only predictor differs. Action: Wave N+2 sweep includes identity-control paired with each grad-clip+LR-warmup config. | 0 (existing flag) | INCLUDED in Wave N+2 |
| 8 | `z7_mamba2_mamba_ssm_pypi_install_must_succeed_in_modal_a100_image_pre_dispatch` | INFRASTRUCTURE-EMPIRICAL | `mamba_ssm` PyPI install on Modal A100 needs validation. Per design memo §13: Mamba-2 reference_torch backend works WITHOUT mamba_ssm on MPS; CUDA path needs mamba_ssm for kernel acceleration. Action: pre-flight smoke ($0.10 sanity-check Modal CPU dispatch testing `pip install mamba-ssm` → `import mamba_ssm` → `print(mamba_ssm.__version__)`). If FAILS: fallback to reference_torch backend (slower but architecturally identical per `--mamba2-backend` flag). | 30 min pre-flight smoke dispatcher | $0.10 |

**Aggregate blocker resolution editor effort**: ~3h (mostly recipe edits + probe-outcome registrations + 1 pre-flight smoke).
**Aggregate blocker resolution GPU cost**: $0.10 (mamba_ssm pre-flight only).

## 4-wave sequencing

```
Wave N+1 (THIS subagent + Wave N+1 council convocation; $0 editor + operator-90-min)
  ├── This memo (multi-week path forward) — DONE
  ├── Sister design-space memo (5 candidates ranked) — DONE
  ├── Sister symposium DRAFT (T3 v2-frontmatter, 6-step contract) — DONE
  ├── Operator-routable: convene T3 grand council on Z7-Mamba-2 stability per DRAFT
  ├── Operator-routable: pre-flight smoke for mamba_ssm Modal A100 import ($0.10)
  ├── Operator-routable: MPS proxy test on M5 Max — verify Mamba-2 reference_torch
  │   forward doesn't NaN at canonical 64-pair scale ($0 LOCAL)
  └── Update recipe: remove stale blockers #1 + #2 (5 min edit)

Wave N+2 (gated on Wave N+1 council PROCEED-unconditional + operator-approved $5 envelope)
  ├── Implement --grad-clip-norm + --lr-warmup-steps CLI flags in Z7-Mamba-2 trainer
  │   (~30 min editor; 2 torch one-liners in _full_main)
  ├── Modal T4 sweep: 9 configs (grad_clip ∈ {0.5, 1.0, 5.0} × LR_warmup ∈ {500, 2000, 5000})
  │   at canonical 64-pair scale + 100ep smoke + identity-control paired
  ├── Harvest via tac.deploy.modal.call_id_ledger.update_call_id_outcome
  └── Verdict gate:
      ├── IF ANY converges → reactivate Z7-Mamba-2 per Catalog #315 OPTIMAL-FORM
      │   → Wave N+3 paths (A) escalate to full dispatch
      └── IF ALL 9 NaN → pivot per Catalog #298 + #308 → Wave N+3 path (B) pivot

Wave N+3 path (A): Z7-Mamba-2 stability-fix-validated full dispatch ($20-30 Modal A100)
  ├── Wave N+1 council PROCEED-unconditional (sister DRAFT memo ratification)
  ├── Full Z7-Mamba-2 100ep training with best grad-clip + LR-warmup config
  ├── Z7MCM2 archive emission + paired CPU/CUDA exact-eval per CLAUDE.md "Submission
  │   auth eval BOTH CPU AND CUDA"
  ├── Post-training Tier-C validation per Catalog #324 (predicted_band_validation_status:
  │   validated_post_training)
  └── Cross-substrate composition Wave N+4 paths

Wave N+3 path (B): pivot to FiLM-LSTM (lowest cost / highest P(success); $5-15)
  ├── Wave N+2 outcome: 9-of-9 NaN OR best converges at score WORSE than Z6-v1 baseline
  ├── Implement FiLM-LSTM substrate scaffold (~3 days editor; sister to Z6-v2 patterns)
  ├── Per-substrate symposium per Catalog #325 (NEW substrate; full 6-step contract)
  ├── 100ep Modal T4 smoke + paired identity-control
  └── Verdict gate:
      ├── IF FiLM-LSTM produces empirical anchor → Wave N+4 full dispatch
      └── IF FiLM-LSTM ALSO fails → all 3 of {Mamba-2, S4, FiLM-LSTM} or {Mamba-2,
          DreamerV3-RSSM, FiLM-LSTM} alternative-probe-methodologies per Catalog #308
          → predictive-coding-recurrent paradigm DEFER per Catalog #298 (NOT KILL)
          → reactivation criterion = NeRV-family predictive-coding-without-recurrent-state OR
            foveation IDEAS without recurrent state

Wave N+4 (gated on Wave N+3 success, $20-50 paid GPU + 2-day editor)
  ├── Per-substrate symposium for whichever stability-fix-class won (Mamba-2 vs FiLM-LSTM
  │   vs DreamerV3-RSSM)
  ├── Full Modal A100 dispatch with canonical 6-step contract evidence
  ├── Composition Wave: Z7-stability-winner + NSCS06v8 chroma + DP1 pretraining + D1
  │   SegNet overlay per design memo §3.6 + Catalog #319 DeliverabilityProof
  └── Promotion path per CLAUDE.md "Apples-to-apples evidence discipline": paired CPU
      Linux x86_64 + CUDA T4 anchors per CLAUDE.md "Submission auth eval BOTH"
```

## Cost summary

| Wave | Editor effort | GPU cost | Operator attention |
|---|---|---|---|
| N+1 | 3h (DONE this subagent) | $0 + $0.10 pre-flight | 90 min council + ratification |
| N+2 | 30 min | $5 | review smoke harvest |
| N+3 path A | ~0 (sister-subagent commits if approved) | $20-30 | review full dispatch outcome |
| N+3 path B | 3 days editor | $5-15 | per-substrate symposium for FiLM-LSTM |
| N+4 (best case) | 2 days editor | $20-50 | promotion review |
| **Total best-case** | **~7 days editor** | **$45-95** | **~5h operator attention spread over 2-3 weeks** |

## Operator-routable decisions (immediate)

1. **Convene Wave N+1 T3 grand council** on Z7-Mamba-2 stability per sister DRAFT memo. Operator-attention budget per Catalog #300: this is 1-of-≤3 T3s/week.
2. **Approve $5.10 Wave N+2 envelope** (9-config Modal T4 smoke + $0.10 mamba_ssm pre-flight) per CLAUDE.md "Race-mode rigor inversion" + "Modal `.spawn()` HARVEST OR LOSE".
3. **Authorize recipe edit** removing stale blockers #1 + #2 (5 min; reversible).
4. **Decide MPS proxy posture**: run pre-flight reference_torch forward-pass NaN check on M5 Max (sister Slot 1 MPS-gap-experiment subagent may consume); $0 LOCAL only.
5. **Cross-substrate sequencing**: convene Z7-GRU Wave 2 council in parallel (independent T3 slot) so Z7-Mamba-2 stability sweep doesn't gate on Z7-GRU outcome unnecessarily.

## Per CLAUDE.md "Forbidden premature KILL without research exhaustion"

This memo enumerates 5 candidates × 4 waves = 20 reactivation paths. NO KILL VERDICT. Substrate-class predictive-coding-recurrent stays in `asymptotic_pursuit` horizon-class per Catalog #309 + design memo §3 Dimension 10. Per Catalog #308 N>=3 alternative-probe-methodologies: 3 stability-fix-class (a/b/c) + 2 pivot-class (d/e) = 5 candidates. Per Catalog #298 30-day staleness window: each Wave's outcome resets staleness clock; multi-week timeline within window.

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A (path-forward memo)
2. Pareto constraint: ACTIVE (5 candidates × 4 waves emit per-wave predicted bands)
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE (Wave N+1/N+2/N+3/N+4 sequencing IS dispatch routing)
5. Continual-learning posterior: ACTIVE — `tac.council_continual_learning.append_council_anchor` for this T1 working-group memo
6. Probe-disambiguator: ACTIVE — Wave N+2 grad-clip sweep + identity-control IS the canonical Mamba-2 stability disambiguator

## Catalog #229 PV

Inherited from sister design-space memo PV-0 through PV-4 (same canonical helpers, same predecessor audit, same trainer-state verification, same probe-ledger query, same sister-subagent coordination).

## Atom emission per Catalog #245/#323

Atom: `build_council_deliberation_atom(atom_id="z7_mamba_2_multi_week_path_forward_20260518", deliberation_id="z7_mamba_2_multi_week_path_forward", council_tier="T1", council_verdict="PROCEED_WITH_REVISIONS", predicted_impact_lower=-0.030, predicted_impact_upper=-0.003, cost_envelope_usd=95.00, memory_path=".omx/research/z7_mamba_2_multi_week_path_forward_20260518.md")` — path-forward enumeration; cost envelope = best-case Wave N+1 → N+4 total per cost summary table; NO score claim per CLAUDE.md "Apples-to-apples evidence discipline"

## Cross-references

- Parent T3 council: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
- Sibling design-space memo: `.omx/research/z7_mamba_2_stability_design_space_20260518.md`
- Sibling symposium DRAFT: `.omx/research/council_t3_z7_mamba_2_stability_path_forward_symposium_DRAFT_20260518.md`
- Predecessor audit: `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Recipe: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
- Trainer: `experiments/train_substrate_time_traveler_l5_z7_mamba2.py`
- Mamba-2 predictor canonical helper: `src/tac/optimization/mamba2_predictor.py`
- Z7-Mamba-2 substrate module: `src/tac/substrates/time_traveler_l5_z7_mamba2/`
- Z7-Mamba-2 design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Probe ledger entry: `z7_mamba2_canonical_scale_stability_20260518` (verdict=DEFER; blocker_status=blocking)
- CLAUDE.md non-negotiables: "Forbidden premature KILL", "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY", "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium", "Race-mode rigor inversion", "Mission alignment", "Apples-to-apples evidence discipline", "Submission auth eval — BOTH CPU AND CUDA"
