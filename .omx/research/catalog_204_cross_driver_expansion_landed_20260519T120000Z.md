---
council_tier: T1
council_attendees: [Claude-Catalog204-CrossDriverExpansion-Subagent]
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "Apply Catalog #204 canonical 3-branch OUTPUT_DIR pattern to 47 NON_COMPLIANT substrate drivers"
  - "Extend Catalog #204 STRICT preflight gate scope to scan scripts/remote_lane_substrate_*.sh glob via _CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE"
  - "Add 20 dedicated tests covering helper regex, end-to-end gate behavior, waiver semantics, and live-repo regression guards"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
---

# CATALOG #204 CROSS-DRIVER EXPANSION (47 substrate drivers + gate scope-extension)

## Summary

Extincted the durable-output bug class STRUCTURALLY across all 50 substrate drivers via two surfaces:

**Surface 1 — Driver fixes (commit `b43c8f2fd`)**: 47 NON_COMPLIANT substrate drivers refactored to the canonical 3-branch Modal-aware OUTPUT_DIR resolution pattern from `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh` lines 50-59.

**Surface 2 — Gate scope-extension**: `check_pr95plus_modal_smoke_uses_durable_provider_output` (Catalog #204) extended to scan `scripts/remote_lane_substrate_*.sh` glob via canonical 3-branch regex; STRICT-from-byte-one with live count 0 at landing.

## Pre-fix audit

50 substrate drivers under `scripts/`:
- **3 COMPLIANT** (already had canonical 3-branch): PR95++ (PR101 LC v2), stack_of_stacks (sister `956ad2e76` 2026-05-19), d1_segnet_margin_polytope
- **47 NON_COMPLIANT**: every other substrate driver defaulted OUTPUT_DIR to `$LOG_DIR/output` (which resolves to `/tmp/pact` on Modal workers, refused by `contest_auth_eval.py` per CLAUDE.md "Forbidden /tmp paths in any persisted artifact")
  - **39 FIX_A bucket** ("CANONICAL_DEFAULT"): had bare `OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"`
  - **8 FIX_B bucket** ("DEFAULT_OUTPUT_DIR refactor"): Z3/Z4/Z5/Z6/Z7 family had ad-hoc 2-branch DEFAULT_OUTPUT_DIR logic that didn't match the canonical 3-branch shape

## Canonical fix pattern

```bash
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_<id>_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${<NS>_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$<NS>_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
```

Each driver was parameterized with its per-namespace operator-override env var (e.g. `BALLE_RENDERER_OUTPUT_DIR`, `Z6_OUTPUT_DIR`, `SANE_HNERV_OUTPUT_DIR`). 47 drivers verified `bash -n` syntax clean post-edit.

## Per-driver audit table

All 47 NON_COMPLIANT drivers fixed in commit `b43c8f2fd`:

a1_plus_lapose / a1_plus_wavelet_residual / atw_codec_v2 / balle_renderer / block_nerv / c1_world_model_foveation / c6_e4_mdl_ibps / cnerv / cool_chic / d4_wyner_ziv_frame_0 / ds_nerv / e_nerv / ego_nerv / ff_nerv / grayscale_lut / hi_nerv / hybrid_renderer_residual / lane_12_v2_nerv / nervdc / nscs01_nullspace_split_renderer / nscs02_downsampled_renderer / nscs03_end_to_end_balle_joint_codec / nscs06_carmack_hotz_strip_everything / nscs06_v8_path_b_wavelet / pretrained_driving_prior / rudin_floor_interpretable_ml / s2sbs_byte_stuffing / sabor_boundary_only_renderer / sane_hnerv / sar_coherent_pose_pairs / self_compress_nn / siren / stc_v2 / tc_nerv / time_traveler_l5_autonomy / time_traveler_l5_tt5l_v2 / time_traveler_l5_z6 / time_traveler_l5_z7_lstm_predictive_coding / time_traveler_l5_z7_mamba_2 / tishby_ib_pure / vq_vae / wavelet / wyner_ziv_cooperative_receiver / z3_balle_hyperprior_bolton / z3_g1_scorer_softmax_hyperprior_gating / z4_cooperative_receiver_loss / z5_predictive_coding_world_model

## Catalog #204 scope-extension

Per sister `956ad2e76` recommendation + Catalog #244 NVML wave precedent: instead of landing a new sister gate (#338 or similar), the existing Catalog #204 gate was SCOPE-EXTENDED per CLAUDE.md "Gate consolidation discipline" (Catalog #299 quota brake currently well under #400 ceiling).

**New gate behavior** at `src/tac/preflight.py::check_pr95plus_modal_smoke_uses_durable_provider_output`:

1. PR95++-specific anchor checks (existing): driver / trainer / Modal dispatcher contract snippets
2. **NEW** cross-driver scan via `_CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE`: every `scripts/remote_lane_substrate_*.sh` must EITHER have the canonical 3-branch shape OR no `OUTPUT_DIR=` assignment OR a same-line `# CATALOG_204_CROSS_DRIVER_WAIVED:<rationale>` waiver (placeholder `<rationale>` / `<reason>` literals rejected)

**Live count at landing: 0** (all 50 drivers compliant post-Phase-2).

## Tests

`src/tac/tests/test_check_204_cross_driver_expansion.py`: 20 tests covering helper regex unit cases (canonical shape match / non-compliant rejection / optional `-d /modal_results` clause / waiver extraction / bare-marker rejection), end-to-end gate (empty canvas / compliant driver / non-compliant flagged / scaffold-without-OUTPUT_DIR skipped / waiver accepted / placeholder rejected / multi-violation aggregation / strict-mode raise / clean strict silent / mixed compliant+non-compliant / DEFAULT_OUTPUT_DIR variant rejected), and live-repo regression guards (cross-driver clean / total clean / orchestrator strict=True preserved).

Sister tests (`src/tac/tests/test_check_204_pr95plus_modal_durable_output.py`, 5 tests) still pass — no regression of the original PR95++ anchor behavior.

## Bug-class extinction matrix

The cross-driver gate expansion follows the **Catalog #244 NVML wave precedent**: when same bug class hits 3+ surfaces, a single gate scope-extension extincts the class at the cross-cutting layer instead of landing per-instance gates.

Sister anchors closed by this expansion:
- **PR95++ (2026-05-14)** — original Catalog #204 anchor; PR101 LC v2 driver
- **stc_v2 (2026-05-14)** — sister regression with same bug class
- **stack_of_stacks (2026-05-19)** — sister `956ad2e76` driver fix anchor
- **47 dormant drivers** — would have hit the same bug class on first Modal dispatch (anchored prospectively, no GPU spend wasted)

## Sister coordination per Catalog #230

DISJOINT scope from 4 in-flight sister subagents at start (MPS Phase B / R11 H1-1+H1-6 / R11 remaining / Cable C6 RE-EVAL-HIGH). Two preflight.py-editing sister subagents (R11 H1-1+H1-6 and R11 remaining) committed BEFORE my preflight.py edit (commits `c8d51ebb5` + `635c41972`); waited for both to land before extending Catalog #204 to avoid Catalog #157 commit-swap. My preflight.py edit ONLY touched the Catalog #204 block (lines 46304-46420 in pre-edit state) — disjoint from R11/MPS sister territories.

Files touched:
- 47 `scripts/remote_lane_substrate_*.sh` drivers (Phase 2, commit `b43c8f2fd`)
- `src/tac/preflight.py` (Catalog #204 block scope-extension — Phase 3)
- `src/tac/tests/test_check_204_cross_driver_expansion.py` (NEW — Phase 3)
- `CLAUDE.md` Catalog #204 row (Phase 3)
- `.omx/state/lane_registry.json` (lane registration via `tools/lane_maturity.py`)
- This memo + memory entry (Phase 4)

## Premise verification per Catalog #229

PV-1: sister `956ad2e76` memo + commit confirms bug class + canonical fix template → PASSED
PV-2: existing Catalog #204 gate at `src/tac/preflight.py:46308-46398` is PR95++-only-scoped → CONFIRMED before edit
PV-3: 50 substrate drivers + 3 baseline COMPLIANT before fix → empirically audited via Python regex sweep
PV-4: 47 driver fixes pass `bash -n` syntax check → ALL CLEAN post-edit
PV-5: Catalog #244 NVML wave precedent is canonical cross-driver gate scope-extension pattern → CONFIRMED
PV-6: Sister subagents (slot 2 + slot 5) committed preflight.py BEFORE my Phase 3 edit → CONFIRMED via `git log`
PV-7: Catalog #299 quota brake under #400 ceiling so scope-extension preferred over new gate → CONFIRMED (current max ~337)

## 6-hook wire-in declaration per Catalog #125

- Hook 1 (sensitivity-map): N/A — defensive validator gate, no signal contribution
- Hook 2 (Pareto constraint): N/A
- Hook 3 (bit-allocator): N/A
- Hook 4 (cathedral autopilot dispatch): ACTIVE — autopilot dispatch ranker now consumes 47 newly-protected substrate drivers without auth_eval temp-output refusal risk
- Hook 5 (continual-learning posterior): N/A — driver-source fix, no posterior signal
- Hook 6 (probe-disambiguator): ACTIVE — the canonical 3-branch shape IS the structural disambiguator between Modal-aware and Modal-vulnerable drivers

## Catalog #314 absorption-pattern avoidance

Phase 2 commit `b43c8f2fd` bounded scope: 47 substrate driver edits + 1 commit body. No preflight.py / test / CLAUDE.md edits — disjoint from any in-flight sister subagent's working tree.

Phase 3 commit (this commit batch) bounded scope: `src/tac/preflight.py` (Catalog #204 block only) + new test file + CLAUDE.md (Catalog #204 row only) + lane registry entry + this memo + memory entry. Sequenced AFTER sister R11 + R11-remaining commits to avoid Catalog #157 commit-swap race on preflight.py.

## Catalog #117/#157/#174 commit discipline

Phase 2: canonical serializer + 94 `--expected-content-sha256` flags (47 drivers × 2 flag tokens each).

Phase 3: canonical serializer + per-file `--expected-content-sha256` post-edit shas (CRITICAL for preflight.py concurrency with sister subagent edits).

## Cross-references

- Sister anchor: `.omx/research/catalog_204_driver_fix_plus_free_cuda_recovery_landed_20260519T062024Z.md` (commit `956ad2e76`)
- Canonical fix template: `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh` lines 50-59
- Sister cross-driver gate precedent: Catalog #244 NVML wave (`check_remote_lane_scripts_carry_canonical_nvml_block`)
- CLAUDE.md "Forbidden /tmp paths in any persisted artifact (the transient-evidence trap)"
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
- CLAUDE.md "Gate consolidation discipline" (Catalog #299 quota brake)
- Catalog #167 (smoke-before-full); #166 (Modal source/head parity); #203 (Modal runtime dep closure); #244 (NVML env block)

## META-pattern

**Cross-driver bug classes need cross-driver gates.** When the same bug class hits 3+ surfaces (PR95++ + stc_v2 + stack_of_stacks), per-instance fixes leave the bug class active at 6-7× other surfaces per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. The structural protection is gate scope-extension (preferred per Catalog #299 quota consolidation) OR new sister gate (when the bug class is genuinely orthogonal). This expansion follows the canonical Catalog #244 NVML wave: scope-extend the existing gate to scan the cross-cutting glob; refuse drivers via canonical 3-branch regex; require explicit waivers.

## Provenance

- `subagent_id`: catalog_204_cross_driver_expansion_20260519
- `lane_id`: `lane_catalog_204_cross_driver_expansion_20260519`
- `evidence_grade`: driver_source_fix + strict_preflight_extension (not a score claim; this is structural extinction of a bug class across the substrate canvas)
