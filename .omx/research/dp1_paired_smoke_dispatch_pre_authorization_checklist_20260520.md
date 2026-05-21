---
council_tier: T1
council_attendees: [Operator-Pre-Authorization-Verifier]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_decisions_recorded:
  - "OP-ROUTABLE #1: author the 3 paired-smoke recipe YAMLs (DEFER prerequisite)"
  - "OP-ROUTABLE #2: extend DP1 trainer _full_main with DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1 branch"
  - "OP-ROUTABLE #3: extend DP1 inflate runtime with derive_codebook_from_seed re-derivation"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: pretrained_driving_prior_procedural_codebook_replacement
deferred_substrate_retrospective_due_utc: "2026-06-19T23:59:00Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  contest_cpu_with_frame_response: [-0.005000, +0.000500]
  contest_cpu_rate_axis_only: [-0.002706, -0.002706]
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
parent_design_memo: .omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md
parent_landing_memo: feedback_dp1_procedural_trainer_build_landed_20260520.md
build_landing_commit: 9cbfa471c
build_landing_utc: 2026-05-20T23:53:41Z
sister_lanes:
  - tools/run_magic_codec_dense_streams_dwt_residual_smoke.py (MAGIC CODEC PAIR #1; DISJOINT)
  - src/tac/canonical_equations/procedural_codebook_savings.py (EQUATION #26 DOMAIN REFINEMENT; DISJOINT)
---

<!--
Catalog #344 canonical-equations-registry cross-reference: this checklist
verifies the dispatch chain for the $0.30 Modal T4 paired CPU+CUDA smoke
that will land the FIRST EMPIRICAL ANCHOR for canonical equation
`procedural_codebook_from_seed_compression_savings_v1` at
`src/tac/canonical_equations/procedural_codebook_savings.py`. The equation
is currently registered with 0 empirical anchors (3 registry events:
2 anchor_appended preserving 0 anchors + 1 domain_refined per Slot 3 DWT
detail-subband exclusion landed commit 8d8a7c6c5).
-->
<!-- HISTORICAL_SCORE_LITERAL_OK:checklist_describes_predicted_band_no_score_authority_claim_2026-05-20 -->

# DP1 PAIRED-SMOKE DISPATCH PRE-AUTHORIZATION CHECKLIST (2026-05-20)

## Section 1. Summary

**Verdict**: **DEFER** pending 3 prerequisite landings (recipe YAMLs + trainer
flag extension + inflate runtime extension). The substrate scaffold IS
production-ready per parent BUILD commit `9cbfa471c`, but the 3 paired-smoke
recipes designed in parent §4 are NOT committed to
`.omx/operator_authorize_recipes/` and the trainer's `_full_main` does NOT yet
honor the new `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1` flag described in
recipe outline #2. The operator can NOT fire $0.30 paired smoke until these
prerequisites land per the 3-stage DP1 design memo OP-ROUTABLE #2 + #3 sequencing.

**Scope of THIS memo**: per-gate verification of the 18 canonical
Catalog discipline gates that govern any DP1 paired-smoke dispatch. Surfaces
each blocker explicitly so the operator can DEFER with confidence rather than
fire spend and crash mid-dispatch.

**Sister-DISJOINT** from MAGIC CODEC PAIR #1
(`tools/run_magic_codec_dense_streams_dwt_residual_smoke.py`; commit
`debbc5833`) + CANONICAL EQUATION #26 DOMAIN REFINEMENT
(`src/tac/canonical_equations/procedural_codebook_savings.py`; commit
`8d8a7c6c5`). Both sister lanes touch different file paths.

## Section 2. PV / premise verification table

Per Catalog #229.

| Premise | Source | Verification | Verdict |
|---|---|---|---|
| Parent BUILD landing commit `9cbfa471c` | `git log 9cbfa471c -1` | wave-3-dp1-procedural-trainer-build-l0-scaffold; Wed May 20 18:53:41 2026 | ✓ |
| Parent design memo exists | `.omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md` | 546 lines; 14 sections; commit `498805f8d`; T2 council PROCEED_WITH_REVISIONS | ✓ |
| Sister-checkpoint guard PROCEED | `tools/check_sister_files_recently_landed.py` | "no sister commits touched any of 1 target file(s) within the 12-hour lookback window" | ✓ |
| DP1 procedural variant module exists | `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` | 20.6 KB; ~400 LOC | ✓ |
| `PROCEDURAL_VARIANT_AVAILABLE=True` flag in `__init__.py` | grep | exported + True | ✓ |
| 15/15 procedural variant tests PASS | `.venv/bin/python -m pytest src/tac/substrates/pretrained_driving_prior/tests/test_procedural_variant.py` | 15 passed in 1.04s | ✓ |
| Canonical equation registered | `.omx/state/canonical_equations_registry.jsonl` | `procedural_codebook_from_seed_compression_savings_v1` 4 events; 0 empirical anchors (predicted-only) | ✓ |
| DP1 symposium memo ≤14 days | `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` | 3 days old; PROCEED_WITH_REVISIONS; Catalog #325 OK | ✓ |
| Catalog #313 probe-outcomes ledger clean | `tools/check_predecessor_probe_outcome.py` | "no blocking predecessor outcome for 'pretrained_driving_prior'" + "no blocking predecessor outcome for 'pretrained_driving_prior_procedural_codebook_replacement'" | ✓ |
| Parent recipe exists | `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` | 6.9 KB; smoke_only=true | ✓ |
| 3 paired-smoke recipes ABSENT | `ls substrate_pretrained_driving_prior_{procedural,original,null_exploit}*.yaml` | NO MATCH (none exist) | **✗ BLOCKER** |
| DP1 driver Catalog #204 Modal-aware OUTPUT path | grep `MODAL_RUNTIME` in `scripts/remote_lane_substrate_pretrained_driving_prior.sh` | 3-branch canonical override present | ✓ |
| Parent recipe DPP_OUTPUT_DIR risk | `DPP_OUTPUT_DIR: /workspace/pact/lane_pretrained_driving_prior_results/output` | env-override takes precedence over driver's Modal-aware branch (`${DPP_OUTPUT_DIR:-}` first) | **⚠️ ATTENTION** |
| `modal_train_lane.py` Catalog #339 fail-closed | grep `register_dispatched_call_id_fail_closed\|LedgerRegistrationFailedError` | 5 references; canonical fix landed | ✓ |
| Catalog #244 NVML env block in driver | grep `DALI_DISABLE_NVML\|CUBLAS_WORKSPACE_CONFIG\|PYTORCH_CUDA_ALLOC_CONF` | 4 references; canonical 3-export present | ✓ |
| `mount_manifest.py` Catalog #165 mtime stability | grep `verify_mount_set_mtime_stability\|MountUploadRaceError` | 16 references; canonical present | ✓ |
| Operator-authorize Catalog #199 paired-env constants | grep in `tools/operator_authorize.py` | `_SESSION_DIRECTIVE_ENV_VAR` + `_SESSION_BUDGET_ENV_VAR` present | ✓ |
| DP1 lane registry L1 impl_complete | `.omx/state/lane_registry.json` | lane `lane_dp1_procedural_codebook_replacement_variant_20260520` L1; impl_complete=true; OPERATIONAL mechanism | ✓ |
| 1094 lanes validate cleanly | `tools/lane_maturity.py validate` | "OK — 1094 lane(s) validated cleanly" | ✓ |

**PV verdict**: 17 of 19 premises verified ✓; 1 BLOCKER ✗ (3 paired-smoke
recipes ABSENT); 1 ATTENTION ⚠️ (parent recipe DPP_OUTPUT_DIR routes around
Catalog #204 Modal-aware branch). The substrate scaffold + canonical
infrastructure ARE production-ready; the missing surface is the operator-routable
recipe + trainer flag wire-in for the procedural variant.

## Section 3. The 18-gate verification table

Per CLAUDE.md Catalog discipline + the parent design memo's enumeration.

| # | Catalog | Gate | Verdict | Evidence / Blocker |
|---|---|---|---|---|
| 1 | #270 | Dispatch optimization protocol Tier 1/2/3 umbrella | **PASS** (parent recipe) | tier1.signals=5/5 + tier2.signals=8/8 + tier3.signals=5/5 on parent recipe per `tools/local_pre_deploy_check.py`; ⚠️ procedural-variant recipe MUST inherit |
| 2 | #240 | Recipe-vs-trainer-state consistency | **DEFER** | Parent recipe is `smoke_only=true`; the 3 paired-smoke recipes designed in §4 do NOT exist; recipe outline #2 specifies `smoke_only=false` + `dispatch_enabled=false` (operator-gated flip) per design memo §4 |
| 3 | #220 | Substrate L1+ scaffold operational mechanism | **PASS** | lane registry `score_improvement_mechanism_status=OPERATIONAL` + `runtime_overlay_consumed=true` + `byte_mutation_smoke_passes=true` + `archive_bytes_added=-4064` (NEGATIVE; variant REDUCES bytes) |
| 4 | #272 | Distinguishing-feature integration contract | **PASS** | `test_compose_with_procedural_codebook_byte_mutation_smoke_catalog_272` flip-1-seed-byte PASSED at BUILD landing; codebook section bytes differ + sister sections byte-identical |
| 5 | #244 | Remote lane canonical NVML env block | **PASS** | `scripts/remote_lane_substrate_pretrained_driving_prior.sh` carries all 3 canonical exports (`DALI_DISABLE_NVML=1` + `CUBLAS_WORKSPACE_CONFIG` + `PYTORCH_CUDA_ALLOC_CONF`) |
| 6 | #166 | Modal dispatch HEAD parity ledger | **PASS** | `experiments/modal_train_lane.py` `_git_dirty_tree_summary` + `metadata schema v2_catalog166` + sentinel SHA capture + worker-side `modal_worker_head_ledger.json` write |
| 7 | #245 | Modal call_id canonical ledger | **PASS** | `tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed` fcntl-locked JSONL APPEND-ONLY |
| 8 | #339 | Modal silent-no-spawn structural extinction | **PASS** | 5 references to `register_dispatched_call_id_fail_closed` + `LedgerRegistrationFailedError` in `experiments/modal_train_lane.py`; canonical fix landed |
| 9 | #143 | Paid job register-before-submit | **N/A** | Catalog #143 scope is Lightning dispatcher (`*lightning*.py`); Modal dispatcher #339 sister surface satisfies the equivalent pattern |
| 10 | #167 | Smoke-before-full pattern | **DEFER** | `tools/run_modal_smoke_before_full.py` is canonical wrapper; the 3 paired-smoke recipes will route through it once authored |
| 11 | #199 | Operator-authorize paired-env bypass discipline | **PASS** | `tools/operator_authorize.py` `_SESSION_DIRECTIVE_ENV_VAR` + `_SESSION_BUDGET_ENV_VAR` paired-env discipline present; NO bare bypass allowed |
| 12 | #205 | Submission inflate.py canonical select_inflate_device | **N/A** | DP1 submission has its own inflate runtime; procedural variant re-uses existing DP1 inflate (no new submissions/ inflate.py landed) |
| 13 | #325 | Per-substrate symposium ≤14 days | **PASS** | DP1 deep-dive symposium 2026-05-17 (3 days old) PROCEED_WITH_REVISIONS T3 grand council; well within 14-day window; council_deliberation_posterior anchor present |
| 14 | #324 | Post-training Tier-C validation discipline | **PASS** | `predicted_band_validation_status: pending_post_training` declared in parent design memo frontmatter; first paired smoke IS the post-training validation |
| 15 | #313 | Predecessor probe-outcomes ledger | **PASS** | `tools/check_predecessor_probe_outcome.py --substrate pretrained_driving_prior` + `--substrate pretrained_driving_prior_procedural_codebook_replacement` both "no blocking predecessor outcome" |
| 16 | #152 | Operator wrapper validates required input files | **PASS** | parent recipe declares `required_input_files: --video-path upstream/videos/0.mkv`; sister extra-mount staging satisfied via STRUCTURAL_MINIMUM_DIRS |
| 17 | #131 | Fcntl-locked JSONL state writes | **PASS** | canonical equations registry + lane registry + commit-serializer log + modal_call_id_ledger + probe_outcomes_ledger all fcntl-locked |
| 18 | #344 | Canonical equation cross-reference | **PASS** | Parent design memo + this checklist both cite `procedural_codebook_from_seed_compression_savings_v1` registry path; canonical equation 26 has 3 registry events including 1 domain_refined per sister Slot 3 |

**Gate summary**: **14 PASS** + **3 DEFER** (Catalog #240 + #167 + N/A for #143 + #205) + **0 FAIL** + **1 PASS-WITH-ATTENTION** (Catalog #270 — verdict transfers to the new recipes when authored).

The DEFER-3 set is structurally bound to the SAME prerequisite: **the 3
paired-smoke recipes are not yet authored**. The substrate scaffold + canonical
infrastructure pass all other gates.

## Section 4. Per-recipe verification

Per parent design memo §4 (3 recipe outlines specified; 0 committed to
`.omx/operator_authorize_recipes/`).

### Recipe #1 — DP1 ORIGINAL baseline (paired-smoke)

| Field | Specified | Actual | Verdict |
|---|---|---|---|
| Filename | `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch.yaml` | **NOT EXIST** | ✗ |
| `smoke_only` | `false` | N/A (file absent) | ✗ |
| `dispatch_enabled` | `false` (operator-gated flip) | N/A | ✗ |
| `cost_band.hand_calibrated_fallback_p50_usd` | 0.30 | N/A | ✗ |
| `predicted_band` | parent's `[0.175, 0.190]` holistic DP1 band | N/A | ✗ |
| Inheritance from parent recipe | Required | N/A | ✗ |

**Verdict**: **DEFER — file ABSENT**. Per operator OP-ROUTABLE #1 (DEFER
prerequisite), this recipe must be authored before any paired smoke.

### Recipe #2 — DP1 PROCEDURAL replacement variant (canonical first anchor)

| Field | Specified | Actual | Verdict |
|---|---|---|---|
| Filename | `substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml` | **NOT EXIST** | ✗ |
| `smoke_only` | `false` | N/A | ✗ |
| `dispatch_enabled` | `false` (operator-gated flip) | N/A | ✗ |
| `predicted_band` | `[-0.005000, +0.000500]` | N/A | ✗ |
| `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1` env var | Required | trainer does NOT recognize this flag yet (no argparse extension landed) | ✗ |
| `DPP_PROCEDURAL_SEED_BYTES=32` | Required | N/A | ✗ |
| `DPP_PROCEDURAL_GENERATOR_KIND=pcg64` | Required | N/A | ✗ |
| `DPP_PROCEDURAL_OUTPUT_SHAPE=1024,4` | Required | N/A | ✗ |
| `DPP_PROCEDURAL_DTYPE=uint8` | Required | N/A | ✗ |
| `DPP_PROVENANCE_PRESERVATION_REQUIRED=1` | Required | N/A | ✗ |

**Verdict**: **DEFER — file ABSENT + trainer wire-in PENDING**. This is the
CANONICAL FIRST EMPIRICAL ANCHOR recipe; both surfaces must land before paired
smoke is admissible.

### Recipe #3 — DP1 NULL-EXPLOIT variant (optional control)

| Field | Specified | Actual | Verdict |
|---|---|---|---|
| Filename | `substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch.yaml` | **NOT EXIST** | ✗ |
| `DPP_NULL_CODEBOOK_BYTES=1` flag | Required | trainer does NOT recognize | ✗ |
| `predicted_band` | `[-0.002706, +0.100000]` (WIDE band) | N/A | ✗ |
| `research_only` | true (explicitly per §4 #3 notes) | N/A | ✗ |

**Verdict**: **DEFER + OPTIONAL — file ABSENT**. This is the OPTIONAL
disambiguator recipe; only required if the procedural variant's empirical
residual is ambiguous (post-Recipe-#2 dispatch).

### Recipe-set verdict

**All 3 recipes ABSENT**. The operator-routable next-action is to author Recipe
#1 + Recipe #2 first (Recipe #3 is OPTIONAL pending Recipe #2 outcome).

## Section 5. Anticipated post-dispatch flow

Per parent design memo §5 (5-step canonical-equation update flow).

### Step 1. Land the paired smoke + harvest results

After OP-ROUTABLE #1 + #2 + #3 prerequisites land, the operator routes via:

```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch \
    --paired-axis cuda+cpu \
    --max-spend-usd 0.30
```

Smoke-before-full per Catalog #167 (auto-gated inside the canonical
wrapper). Modal `.spawn()` per Catalog #245 → `register_dispatched_call_id_fail_closed`
per Catalog #339. Harvest within 24h per Catalog #330 via
`tools/parallel_harvest_actuator.py` or `tools/harvest_modal_calls.py`.

### Step 2. Verify byte-mutation smoke per Catalog #272

```bash
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
    --lane-id lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520 \
    --archive-zip experiments/results/<lane>_modal/contest_archive.zip \
    --inflate-sh experiments/results/<lane>_modal/inflate.sh \
    --distinguishing-bytes-path procedural_seed_section
```

Expected: PASSED verdict (mutating 1 byte of the 32-byte seed → re-derives
different codebook → renders different frames → different scored output).
FAIL = research-substrate trap (8th forbidden pattern) → DEFER-pending-architecture-redesign.

### Step 3. Update canonical equation with empirical anchor

Per Catalog #344 + `tac.canonical_equations.update_equation_with_empirical_anchor`:

```python
from tac.canonical_equations import (
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)

first_empirical_anchor = EmpiricalAnchor(
    anchor_id="dp1_procedural_codebook_first_paired_smoke_anchor_20260520",
    measurement_utc="<paired-smoke-completion-utc>",
    inputs={
        "substrate_id": "pretrained_driving_prior_procedural_replacement",
        "n_codebook_bytes": 4096,
        "k_seed_bytes": 32,
        "axis_tag_cpu": "[contest-CPU]",
        "axis_tag_cuda": "[contest-CUDA T4]",
        "scored_archive_sha256_baseline": "<sha256>",
        "scored_archive_sha256_procedural": "<sha256>",
        "byte_mutation_smoke_verdict": "PASSED",
        "canonical_helper_invocation": "tac.procedural_codebook_generator.derive_codebook_from_seed",
    },
    predicted_output={
        "predicted_delta_s_contest_cpu": -0.002706,
        "predicted_delta_s_contest_cuda": -0.002706,
        "predicted_bytes_saved": 4064,
    },
    empirical_output={
        "empirical_delta_s_contest_cpu": "<measured>",
        "empirical_delta_s_contest_cuda": "<measured>",
        "empirical_bytes_saved": "<measured>",
    },
    residual=abs(predicted_delta_s_cpu - empirical_delta_s_cpu),
    # ... canonical Provenance per Catalog #323
)

update_equation_with_empirical_anchor(
    equation_id="procedural_codebook_from_seed_compression_savings_v1",
    new_anchor=first_empirical_anchor,
)
```

### Step 4. Sister-update Catalog #322 + Catalog #324

Catalog #322: **N/A** for single-substrate anchor (composition_alpha requires
pair-wise measurement; only 2+ substrate anchors produce composition_alpha).
Catalog #324: `predicted_band_validation_status` flips from `pending_post_training`
to `validated_post_training`.

### Step 5. Wire-in to autopilot via Catalog #335

The `procedural_codebook_savings_consumer` (auto-discovered cathedral consumer
per Catalog #335 paradigm) consumes the canonical equation posterior; the next
autopilot loop tick re-ranks DP1 procedural-replacement candidates based on the
empirical anchor instead of predicted-only.

## Section 6. Overall green-light verdict

### Verdict: **DEFER**

**Rationale**: 3 structurally-bound prerequisites must land before the
$0.30 paired smoke can fire:

1. **Author 3 paired-smoke recipe YAMLs** (DEFER prerequisite for Catalog #240).
   The parent recipe is `smoke_only=true`; the 3 paired-smoke variants
   designed in parent §4 are not yet committed to `.omx/operator_authorize_recipes/`.
2. **Extend DP1 trainer `_full_main` with `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1`
   branch** (DEFER prerequisite for Recipe #2). The trainer must invoke
   `tac.procedural_codebook_generator.derive_codebook_from_seed(...)` at
   archive-emit time and emit the 32-byte seed in lieu of the trained
   4096-byte codebook. The new env-var → CLI flags (6 new) must be wired per
   Catalog #151 manifest discipline.
3. **Extend DP1 inflate runtime** with the canonical helper re-derivation
   call (preserves Catalog #328 LOC budget per HNeRV parity L4). The inflate.py
   must call `derive_codebook_from_seed(seed_bytes=archive[seed_section], ...)`
   and substitute back into the codebook section at decode time.

**14 of 18 canonical gates PASS** for the substrate scaffold + canonical
infrastructure. The remaining 3 DEFER gates (Catalog #240 + #167 + per-recipe
DPP_OUTPUT_DIR attention) trace to the SAME structural blocker (the 3 paired-smoke
recipes are absent). The 1 ATTENTION-level concern (parent recipe's
`DPP_OUTPUT_DIR: /workspace/pact/...` would route around Catalog #204's
Modal-aware branch) is automatically resolved when Recipe #2 specifies
`DPP_OUTPUT_DIR: /modal_results/${INSTANCE_JOB_ID}/output` per Catalog #204
canonical pattern in parent design memo §4.

### Operator-routable canonical command (BLOCKED PENDING PREREQUISITES)

Per Catalog #199 paired-env discipline. **DO NOT FIRE** until the 3
prerequisites above land. Once they do, the canonical operator-routable
command is:

```bash
# Step 1: ensure paired-env attestation per Catalog #199
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.30

# Step 2: route via canonical operator-authorize chain
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch \
    --paired-axis cuda+cpu \
    --max-spend-usd 0.30
```

The canonical entry point fires the chain:
local_pre_deploy_check (Catalog #243) → codex pre-dispatch review (Catalog #271)
→ probe-predecessor (Catalog #313) → sentinel files check (Catalog #166)
→ NVML env block (Catalog #244) → Modal HEAD parity → `.spawn()` →
`register_dispatched_call_id_fail_closed` (Catalog #339) → 24h harvest window
(Catalog #330) → byte-mutation smoke (Catalog #272) → canonical equation update
(Catalog #344) → autopilot re-rank (Catalog #335).

## Section 7. Top-3 operator-routable next actions

### Top-3 op-routables for unblock (ordered)

1. **OP-ROUTABLE #1 — Author 3 paired-smoke recipe YAMLs** (~5-10 min wall-clock; $0).
   Spawn a sister subagent to write
   `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_{original_baseline,procedural_codebook,null_exploit_codebook}_modal_t4_paired_dispatch.yaml`
   per parent design memo §4 exact YAML specs. Each recipe inherits from
   parent `substrate_pretrained_driving_prior_modal_t4_dispatch.yaml`.
   Key fields: `smoke_only: false` + `dispatch_enabled: false` +
   `cost_band.hand_calibrated_fallback_p50_usd: 0.30` +
   `cost_band.epochs: 100` + 3-export NVML block + `DPP_OUTPUT_DIR:
   /modal_results/${INSTANCE_JOB_ID}/output` (Catalog #204 canonical) +
   the 6 new procedural env-vars on Recipe #2.

2. **OP-ROUTABLE #2 — Extend DP1 trainer `_full_main`** (~30-60 min wall-clock; $0).
   Spawn sister subagent to extend
   `experiments/train_substrate_pretrained_driving_prior.py::_full_main` with
   `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1` branch that invokes
   `tac.substrates.pretrained_driving_prior.distillation_procedural_variant.compose_with_procedural_codebook(...)`
   at archive-emit time. Add 6 new argparse flags per Catalog #151 manifest
   discipline. Preserve Catalog #210 DP1 codebook provenance metadata in the
   seed-derivation path. Tests must pass (15 procedural variant + 191 existing
   DP1 = 206 total).

3. **OP-ROUTABLE #3 — Extend DP1 inflate runtime** (~15-30 min wall-clock; $0).
   Spawn sister subagent to extend the DP1 inflate runtime (inside the
   substrate package's `inflate.py` if it exists, or via the trainer's
   `_write_runtime` emission). The inflate path must detect the
   procedural-replacement variant (via a flag/section presence in the archive),
   invoke `derive_codebook_from_seed(seed_bytes=32B, output_shape=(1024,4),
   dtype=np.uint8, generator_kind="pcg64")` and substitute back into the
   codebook section. LOC budget per Catalog #328 must remain ≤200.

After OP-ROUTABLE #1 + #2 + #3 land, this checklist's verdict transitions
**DEFER → GREEN-LIGHT** and the operator can fire the canonical command in §6.

### Deferred routables

- **DP1 NULL-EXPLOIT control smoke** (Recipe #3): fire only if procedural
  variant's empirical ΔS lands in `[-0.001, +0.001]` ambiguous band.
- **5-substrate aggregate paired smoke**: spawn after 4+ per-substrate
  anchors land (DP1 first + 3 sisters per matrix memo §10).

## Section 8. Sister coordination

| Sister | Lane / files | Verdict |
|---|---|---|
| MAGIC CODEC PAIR #1 | `tools/run_magic_codec_dense_streams_dwt_residual_smoke.py` (commit `debbc5833`) | **DISJOINT** — different file path; this checklist does not touch `tools/run_magic_codec_*.py` |
| CANONICAL EQUATION #26 DOMAIN REFINEMENT | `src/tac/canonical_equations/procedural_codebook_savings.py` (commit `8d8a7c6c5`) | **DISJOINT** — this checklist does not modify the canonical equation; only references it for provenance |
| Catalog #340 sister-checkpoint guard | `tools/check_sister_files_recently_landed.py --lookback-hours 12` | **PROCEED** at step 0 |

## Section 9. Mission contribution per Catalog #300

**council_predicted_mission_contribution**: `frontier_breaking_enabler`

This checklist enables the FIRST EMPIRICAL ANCHOR for canonical equation
`procedural_codebook_from_seed_compression_savings_v1` once the 3 prerequisites
land. The anchor unblocks the autopilot's ability to trust the predicted
ΔS=-0.002706 (or recalibrate based on empirical residual). Per parent design
memo §12, DP1 procedural-replacement alone is `frontier_protecting` (does NOT
break 0.18 alone; ΔS ≈ -0.003); the AGGREGATE 5-substrate path is the
`frontier_breaking` contribution, and DP1 is the canonical FIRST anchor
on that path. THIS checklist is the `frontier_breaking_enabler` because
it surfaces the precise blockers preventing OP-ROUTABLE #3 from firing.

## Section 10. 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: **N/A** (defensive verification checklist; no
  signal contribution — surfaces blockers, does not emit ranking signal)
- **hook #2 Pareto constraint**: **N/A** (no Pareto signal)
- **hook #3 bit-allocator**: **N/A** (no bit-allocator signal)
- **hook #4 cathedral autopilot dispatch**: **ACTIVE** (prevents premature
  paid dispatch + protects the canonical 4-layer probe-outcomes ledger by
  surfacing DEFER verdict; the autopilot's procedural_codebook_savings_consumer
  cathedral consumer remains observability-only until the first empirical
  anchor lands)
- **hook #5 continual-learning posterior**: **N/A at checklist landing**;
  becomes ACTIVE once OP-ROUTABLE #3 fires the first empirical anchor (per
  Step 3 above's `update_equation_with_empirical_anchor` call which feeds
  the canonical posterior)
- **hook #6 probe-disambiguator**: **ACTIVE** (the per-recipe verification
  table IS the canonical disambiguator between ready-to-fire vs DEFER state;
  the 3-recipe design [#1 baseline / #2 procedural / #3 null-exploit] is the
  canonical 3-way design-space disambiguator)

## Section 11. Cross-references

- **Parent design memo**: `.omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md` (commit `498805f8d`)
- **Parent BUILD landing memo**: `feedback_dp1_procedural_trainer_build_landed_20260520.md` (commit `9cbfa471c`)
- **Parent BUILD landing design memo**: `.omx/research/dp1_procedural_variant_build_landed_20260520.md`
- **DP1 deep-dive symposium**: `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` (3 days old; PROCEED_WITH_REVISIONS; T3 grand council; 12 attendees)
- **Canonical equation builder**: `src/tac/canonical_equations/procedural_codebook_savings.py`
- **Canonical equation registry**: `.omx/state/canonical_equations_registry.jsonl` (4 events; 0 empirical anchors)
- **Canonical procedural seed helper**: `src/tac/procedural_codebook_generator/seed_derived_codebook.py`
- **Canonical procedural codebook savings consumer**: `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/__init__.py`
- **DP1 substrate package**: `src/tac/substrates/pretrained_driving_prior/` (13 .py files + tests)
- **DP1 procedural variant**: `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` (~400 LOC; 15/15 tests PASS)
- **DP1 trainer**: `experiments/train_substrate_pretrained_driving_prior.py` (82.8K)
- **DP1 parent recipe**: `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` (6.9K; smoke_only=true)
- **DP1 remote driver**: `scripts/remote_lane_substrate_pretrained_driving_prior.sh` (Catalog #204 + #244 canonical)
- **5-substrate matrix memo**: `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md`
- **CLAUDE.md non-negotiables cited**:
  - "Executing actions with care" (pre-authorization verification before paid spend)
  - "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  - "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch"
  - "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
  - "Canonical equations + models registry"
  - "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
  - "Modal `.spawn()` HARVEST OR LOSE"
  - "Apples-to-apples evidence discipline"
- **Catalog # gates cited (18 + sister meta)**: #131/#138 fcntl-locked JSONL; #143 paid job register-before-submit (N/A Modal scope); #151 trainer flag manifest; #152 required input file validation; #165 Modal mount mtime stability; #166 Modal HEAD parity ledger; #167 smoke-before-full pattern; #199 paired-env bypass discipline; #205 inflate device selector (N/A); #220 substrate L1+ operational mechanism; #229 PV; #240 recipe-vs-trainer-state consistency; #244 Modal NVML env block; #245 Modal call_id ledger; #270 dispatch optimization protocol umbrella; #272 distinguishing-feature integration contract; #313 probe-outcomes ledger; #322/#323 canonical Provenance; #324 post-training Tier-C validation; #325 per-substrate symposium recency; #328 inflate.py LOC budget; #335 cathedral consumer Protocol; #339 Modal silent-no-spawn extinction; #344 canonical equations registry

---

## Sign-off

- **Subagent ID**: `wave-3-dp1-paired-smoke-dispatch-pre-authorization-checklist-20260520`
- **HEAD at start**: `9cbfa471c` (DP1 procedural variant BUILD landing)
- **UTC**: 2026-05-20T23:58:18Z (memo creation)
- **Verdict**: **DEFER** (3 prerequisites — recipe YAMLs + trainer flag wire-in + inflate runtime extension); 14 of 18 canonical gates PASS for the substrate scaffold + canonical infrastructure
- **Cost**: $0 paid GPU; ~45 min wall-clock
- **Files touched**: 1 (this memo)
- **Lane**: `lane_wave_3_dp1_paired_smoke_dispatch_pre_authorization_checklist_20260520`
- **Sister-DISJOINT**: verified disjoint from MAGIC CODEC PAIR #1 + CANONICAL EQUATION #26 DOMAIN REFINEMENT (different file paths)
- **6-hook wire-in declaration**: §10 hooks #4 + #6 ACTIVE; #5 ACTIVE after Step 3 first-anchor landing; #1 + #2 + #3 N/A (defensive verification gate)
- **mission_predicted_contribution**: `frontier_breaking_enabler`
- **Discipline**: Catalog #110+#113 APPEND-ONLY + canonical serializer + 6-hook wire-in + per-recipe verification + per-gate verification + anticipated post-dispatch flow + sister coordination

---

<!-- END PRE-AUTHORIZATION CHECKLIST MEMO -->
