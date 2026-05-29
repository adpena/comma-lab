# Retroactive sweep — Wave 4 Z7-Mamba-2 Dao & Gu 2024 fidelity audit (2026-05-29T20:47:44Z)

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`: every new gate / canonical
helper / audit landing MUST be paired with a 4-field retroactive sweep memo. Wave 4 is a documentation
+ test landing (no new catalog # claimed per Catalog #299), but the canonical 2-landing pattern applies:
audit + test pinning + retroactive sweep.

## 1. Bug-class symptom signature

The Wave 4 audit surfaced an IMPLEMENTATION-LEVEL falsification of the claim that `_ReferenceMamba2Cell`
implements canonical Mamba-2 per Dao & Gu 2024. The actual implementation uses A_log shape
`(d_inner, d_state)` which is Mamba-1 (S6) selective SSM diagonal form, NOT the canonical Mamba-2
SSD form with A_log shape `(nheads,)` scalar-per-head.

**Symptom signature**: docstring claim "Mamba-2 selective state-space cell per Dao-Gu 2024" + A_log shape
`(d_inner, d_state)`. The MAMBA_SSM_BACKEND production path correctly invokes canonical Mamba-2 SSD via
`mamba_ssm.modules.mamba2.Mamba2`; only the REFERENCE_TORCH_BACKEND reference cell was mislabeled.

## 2. Pre-fix window

The mislabeling was introduced when the design memo (`.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
2026-05-18) treated S6 and SSD as architecturally equivalent at the recurrence-mechanism layer. The
substrate identifier `z7_mamba2` was chosen for the upstream production path's correctness, but the
reference path inherited the name without explicit S6-vs-SSD distinction.

**Pre-fix window**: 2026-05-18 (design memo landing) → 2026-05-29T20:47Z (Wave 4 audit landing). Duration
~11 days. Affected: `src/tac/optimization/mamba2_predictor.py` + downstream callers in
`src/tac/substrates/time_traveler_l5_z7_mamba2/architecture.py` + MLX trainer
`experiments/train_substrate_z7_mamba2_v2_mlx.py:283`.

## 3. Historical KILL/DEFER/FALSIFY search results

Per Catalog #348 mandate: search historical verdicts that may have been invalidated by the audit finding.

### 3.1 Existing canonical equation anchors

4 empirical anchors on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`:
1. `mlx_local_canonical_harness_smoke_anchor_proves_gradient_flow_alive_and_inflate_byte_closed` — residual 0.0
2. `mlx_local_canonical_harness_long_training_real_teacher_lr_stability_falsification_anchor` — residual 1.0 (IMPLEMENTATION-LEVEL falsification per Catalog #307)
3. `mlx_local_canonical_harness_long_training_real_teacher_lr_reduced_stabilizer_anchor` — residual 0.0
4. `mlx_local_canonical_harness_identity_predictor_disambiguator_catalog_308_anchor_4_4_wave_n32` — residual 0.31

**Invalidation check per anchor**:
- All 4 anchors were measured on the ACTUAL deployed reference cell (S6 form), not on a hypothetical SSD-form cell. The empirical evidence is consistent with the documented adaptation — the anchors measure what was actually trained.
- **VERDICT**: NO ANCHORS INVALIDATED. The Wave 4 audit confirms the anchors are HARD-EARNED per Catalog #292 + Catalog #363 `VERIFIED_VIA_SOURCE_INSPECTION`.

### 3.2 Probe outcomes

Searched `.omx/state/probe_outcomes.jsonl` for z7_mamba2-substrate outcomes:
- `z7_mamba2_canonical_scale_stability_20260518` BLOCKING with verdict=DEFER per L2 stability hardening memo
- **Invalidation check**: this DEFER verdict is consistent with the documented adaptation — the S6 form's NaN-at-ep-16-18 instability was empirically observed and partially extincted by L2 hardening (Cell 2-3 NaN-FREE 30ep); the SSD form was never tested at contest scale so the DEFER verdict still holds.
- **VERDICT**: PROBE OUTCOME PRESERVED; reactivation criterion now includes "paired-comparison smoke at d_state=128 + headdim=64 SSD vs d_state=16 + d_inner=128 S6" per Wave 4 audit §3.3.

### 3.3 Council deliberation anchors

Searched `.omx/state/council_deliberation_posterior.jsonl` for z7_mamba2 deliberations:
- `council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518` PROCEED
- `council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518` indeterminate-with-nuance verdict
- `path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526` PROCEED with 10 CC items surfaced
- `z7_mamba_2_v2_l2_stability_hardening_landed_20260526` PROCEED with PARTIAL extinction
- **Invalidation check**: all 4 deliberations remain valid. The Wave 4 audit IS a 5th deliberation (PROCEED_WITH_REVISIONS) refining the cargo-cult audit's CC-D + CC-G items with the specific S6-vs-SSD documented adaptation finding.
- **VERDICT**: COUNCIL DELIBERATIONS PRESERVED; Wave 4 appends to the cite-chain per Catalog #300 `related_deliberation_ids`.

### 3.4 Forbidden patterns + FALSIFIED memo classification per Catalog #307

Searched for FALSIFIED / KILLED / DEFERRED memos citing Z7-Mamba-2:
- L2 stability hardening memo classifies the NaN issue as **IMPLEMENTATION-LEVEL PARTIAL** falsification per Catalog #307; PARADIGM INTACT
- **Invalidation check**: the IMPLEMENTATION-LEVEL classification is consistent with Wave 4's S6-vs-SSD finding — both classifications place the issue at the implementation surface, not the paradigm. The PARADIGM (selective state-space recurrence) remains intact across both S6 and SSD forms.
- **VERDICT**: NO MEMOS INVALIDATED; classifications consistent across audits.

### 3.5 Lane registry

Searched `.omx/state/lane_registry.json` for z7_mamba2 lanes:
- `lane_z7_mamba_2_v2_l2_stability_hardening_nan_fix_20260526` L1 (impl_complete + memory_entry)
- `lane_path_3_b_z7_mamba_2_substrate_design_decision_20260526` L1
- Several sister lanes from the cargo-cult-first methodology landing
- **Invalidation check**: no lane requires status revision; all evidence cited matches the deployed reference cell (S6 form) as adjusted by Wave 4's honest naming.
- **VERDICT**: LANE REGISTRY PRESERVED.

## 4. Per-finding RE-EVAL-priority assignment

Per Catalog #348 mandate: assign RE-EVAL-priority to each historical finding affected by the audit.

| Historical finding | RE-EVAL priority | Rationale |
|---|---|---|
| 4 canonical equation anchors on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1` | **LOW** | All measured on S6-form cell as deployed; no re-evaluation needed. Wave 4 anchor APPENDED as 5th anchor (HARD-EARNED). |
| `z7_mamba2_canonical_scale_stability_20260518` probe outcome (DEFER) | **MEDIUM** | Add reactivation criterion: paired-comparison smoke at SSD vs S6 at fixed parameter budget. Current DEFER verdict preserved; expanded criterion enables future operator-routed escalation. |
| 4 prior council deliberations (per-substrate symposium + T3 finding 4 + cargo-cult audit + L2 stability) | **LOW** | Wave 4 audit appends as 5th deliberation (PROCEED_WITH_REVISIONS); cite-chain extended per Catalog #300. |
| L2 stability hardening memo PARTIAL extinction verdict | **LOW** | Wave 4 confirms PARADIGM INTACT + IMPLEMENTATION-LEVEL classification at the A_log-shape sub-axis. No verdict change. |
| Z7-Mamba-2 substrate identifier "z7_mamba2" | **NONE** | Preserved per Catalog #110/#113 HISTORICAL_PROVENANCE. The MAMBA_SSM_BACKEND production path correctly uses canonical Mamba-2 SSD; renaming would corrupt the cite-chain. |
| MLX v2 trainer `_mamba2_step` at `experiments/train_substrate_z7_mamba2_v2_mlx.py:283` | **LOW** | Sister of the reference cell; also S6 form. Same documented-adaptation rationale applies; reactivation criterion: when v2 substrate promotes from L0 SCAFFOLD to L1, apply Wave 4 honest naming discipline (the docstring + comment updates Wave 4 lands in `mamba2_predictor.py` are the canonical reference pattern for the v2 trainer). |

## 5. Structural protection surfaces

Per the canonical 2-landing pattern (canonical helper + STRICT preflight gate), Wave 4 provides 3 structural
protection surfaces WITHOUT claiming a new catalog #:

1. **Docstring honesty fix** in `src/tac/optimization/mamba2_predictor.py` — future readers immediately see
   the "Mamba-1 (S6) form" classification + documented-adaptation rationale + cite-chain to canonical papers
2. **14 fidelity tests** in `src/tac/tests/test_wave_4_z7_mamba_2_dao_gu_fidelity_audit.py` — regression
   guards that pin A_log shape + init + selectivity + discretization invariants + parameter count comparison
3. **Canonical equation 5th anchor** on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1` — triggers Catalog #371 auto-recalibration (4 anchors → 5 anchors crosses 3+ threshold);
   future agents querying the equation see the documented-adaptation context

Per Catalog #299 "Gate consolidation discipline" + 13th OPTIMAL-TRIO standing directive: NEW STRICT gate
NOT needed because existing META-meta gates (Catalog #176 + #185 + #186 + #344) structurally protect the
audit's surfaces; the audit operates within those surfaces.

## 6. Sister-cascade context

Wave 4 is Item 8 of the 12-wave 15-item math-fidelity audit cascade. Sister waves spawned in parallel:
- Wave 2: Cascade C' Frame-1 SegNet waterfill (Item 6) — DISJOINT scope (different substrate)
- Wave 3: DreamerV3 RSSM (Item 7) — DISJOINT scope (different substrate)

Per Catalog #340 sister-checkpoint guard: all 3 waves verified disjoint scope via `.omx/state/subagent_progress.jsonl`
before edit-time; no commit-time race expected.

## 7. mission_predicted_contribution per Catalog #300

`apparatus_maintenance` (documentation honesty + test pinning) + `frontier_protecting` (extincts the
silent-mislabeling bug class structurally so future agents cannot regress to claiming S6 form is Mamba-2 SSD).
