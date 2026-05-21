# DP1 Paired-Smoke Recipes Parity Audit

timestamp_utc: 2026-05-21T04:53:43Z
agent: codex
lane_id: lane_codex_dp1_paired_smoke_parity_audit_20260521
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[empirical:repo-state-audit]"

## Summary verdict

Verdict: NEEDS_RECIPE_METADATA_REVISION_BEFORE_NEXT_PAID_DP1_DISPATCH

The three DP1 paired-smoke recipes are parity-clean on the execution axes that
matter for an apples-to-apples baseline/procedural comparison: GPU, VRAM,
decode strategy, target modes, canary status, lane script, cost band, paired
axis, full-run source mode, auth-eval skip, and Modal/CUDA hygiene.

One drift class remains: all DP1 recipes that declare `predicted_band` omit an
explicit `predicted_band_validation_status`. The current Catalog #324 audit
passes these rows because they are `research_only: true`, but the routing
directive explicitly requires `validated_post_training` or
`pending_post_training`. The safe 1-line recipe-side fix is to add:

```yaml
predicted_band_validation_status: pending_post_training
```

to the standalone DP1 recipe and the three paired-smoke recipes, plus a short
`predicted_band_reactivation_criteria` string if we want parity with newer
substrate recipes.

Live DP1 training status also blocks paired harvest: the latest baseline and
procedural Modal calls both exceeded their 5400s budget while still polling as
`running_or_pending`.

## Inputs read

- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch.yaml`
- `scripts/remote_lane_substrate_pretrained_driving_prior.sh`
- `experiments/train_substrate_pretrained_driving_prior.py`
- `.omx/research/dp1_paired_smoke_recipes_landed_20260521_codex.md`
- `.omx/research/dp1_procedural_paired_harvest_planner_landed_20260521_codex.md`
- `.omx/research/dp1_streamer_no_chunk_ids_dispatch_failure_20260521T031333Z_codex.md`
- `.omx/state/modal_call_id_ledger.jsonl` filtered to `lane_dp1_`

## Per-recipe verdict table

| recipe_name | status | drift_class | notes |
|---|---|---|---|
| substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch | DRIFT_DETECTED | predicted_band_validation_status | Execution axes match procedural arm. Add `predicted_band_validation_status: pending_post_training`. |
| substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch | DRIFT_DETECTED | predicted_band_validation_status | Deliberate knobs are `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1` and provenance path. Add `predicted_band_validation_status: pending_post_training`. |
| substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch | DRIFT_DETECTED | dispatch_enabled, predicted_band_validation_status | `dispatch_enabled=false` is intentional control gating. Add `predicted_band_validation_status: pending_post_training`. |
| substrate_pretrained_driving_prior_modal_t4_dispatch | DRIFT_DETECTED | sibling-scope, predicted_band_validation_status | Historical standalone scaffold, not a paired-smoke sibling. Add validation status if retained as a dispatchable recipe. |

## Parity matrix

| axis | baseline | procedural | null_control | verdict |
|---|---|---|---|---|
| gpu | T4 | T4 | T4 | PARITY |
| min_vram_gb | 16 | 16 | 16 | PARITY |
| pyav_decode_strategy | cpu_thread_async_upload | cpu_thread_async_upload | cpu_thread_async_upload | PARITY |
| target_modes | contest_one_video_replay, research_substrate | contest_one_video_replay, research_substrate | contest_one_video_replay, research_substrate | PARITY |
| canary_status | independent_substrate | independent_substrate | independent_substrate | PARITY |
| lane_script | scripts/remote_lane_substrate_pretrained_driving_prior.sh | scripts/remote_lane_substrate_pretrained_driving_prior.sh | scripts/remote_lane_substrate_pretrained_driving_prior.sh | PARITY |
| cost_band | 100 epochs, T4, p50 0.30 | 100 epochs, T4, p50 0.30 | 100 epochs, T4, p50 0.30 | PARITY |
| paired_axis | enabled, CPU fallback p50 0.10 | enabled, CPU fallback p50 0.10 | enabled, CPU fallback p50 0.10 | PARITY |
| smoke_only | false | false | false | PARITY |
| research_only | true | true | true | PARITY |
| dispatch_enabled | true | true | false | INTENTIONAL_DIFFERENCE |
| DPP_RUN_FULL | 1 | 1 | 1 | PARITY |
| DPP_SKIP_AUTH_EVAL | 1 | 1 | 1 | PARITY |
| DPP_USE_STREAMER | 0 | 0 | 0 | PARITY |
| DPP_CACHE_DIR | /root/.cache/tac/comma2k19_chunks | /root/.cache/tac/comma2k19_chunks | /root/.cache/tac/comma2k19_chunks | PARITY |
| DPP_PROCEDURAL_CODEBOOK_REPLACEMENT | 0 | 1 | 1 | INTENTIONAL_DIFFERENCE |
| DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL | 0 | 0 | 1 | INTENTIONAL_DIFFERENCE |
| DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH | absent | present | present | INTENTIONAL_DIFFERENCE |
| predicted_band | [0.175, 0.190] | [-0.005, 0.0005] | [-0.002706, 0.100000] | INTENTIONAL_DIFFERENCE |
| predicted_band_validation_status | missing | missing | missing | DRIFT_DETECTED |

## DRIFT_DETECTED details

### predicted_band_validation_status missing

Evidence:

- `tools/audit_predicted_band_provenance.py` reports the four DP1 recipes as
  `PASS | research_only`, so current strict preflight does not fail them.
- The routing directive's Catalog #324 audit requirement is stricter: every
  recipe with `predicted_band` must carry `validated_post_training` or
  `pending_post_training`.

Root cause: the paired-smoke recipe landing relied on `research_only: true`
for Catalog #324 false-authority safety, but did not also add the explicit
status field used by newer recipe patterns.

Recommended 1-line fix per recipe:

```yaml
predicted_band_validation_status: pending_post_training
```

Recommended fuller fix:

```yaml
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criteria: "Replace this planning band only after paired DP1 CPU/CUDA training-output harvest and post-training Tier-C density / exact score review."
```

### null-control dispatch_enabled=false

Evidence: `operator_authorize.py --dry-run` exits `0` but reports the null
control would refuse because `dispatch_enabled=false`.

Verdict: INTENTIONAL_DIFFERENCE, not a parity bug. The recipe itself says the
null-control arm is optional and should only flip after an ambiguous
baseline/procedural residual.

### standalone DP1 scaffold recipe

Evidence: `substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` is
`smoke_only: true`, uses the historical scaffold lane, and targets production
generalization modes not present in the paired-smoke recipes.

Verdict: not a sibling for paired-smoke parity. It still has a `predicted_band`
without an explicit validation status, so it should receive the same metadata
fix if retained as a dispatchable operator recipe.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_predecessor_probe_outcome.py --substrate pretrained_driving_prior
```

Result: `OK: no blocking predecessor outcome for 'pretrained_driving_prior'`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch
```

Result: `ALL 9 CHECKS PASSED. Safe to dispatch.`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch
```

Result: `ALL 9 CHECKS PASSED. Safe to dispatch.`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch
```

Result: failed only because `_full_main` is implemented while
`dispatch_enabled=false`. This matches the intentional control gating.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --dry-run --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --dry-run --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --dry-run --recipe substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch
```

Result: all exit `0`; baseline/procedural print dispatch plans, null-control
prints the expected `dispatch_enabled=false` refusal.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_predicted_band_provenance.py
```

Result: `FAIL: 0`; DP1 rows pass under the current helper as `research_only`.
Directive-level stricter parity still recommends explicit status fields.

## Catalog checks

### Catalog #357 Tier B

Verdict: N/A. These are operator-authorize recipes, not cathedral consumer
Tier B surfaces.

### Catalog #325 per-substrate symposium

Verdict: present. Relevant memos within the active window:

- `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md`
- `.omx/research/council_t3_dp1_deep_dive_per_substrate_symposium_DRAFT_20260519T053356Z.md`
- `.omx/research/council_per_substrate_symposium_unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision_20260520T175410Z.md`

### Catalog #324 predicted_band validation status

Verdict: drift detected at directive level, pass at current helper level. The
helper accepts `research_only: true`; the directive asks for an explicit
status token. Add `pending_post_training`.

### Catalog #244 NVML env block

Verdict: pass. The remote lane driver exports:

```bash
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
```

The paired recipes also set all three env vars explicitly.

### Catalog #240 recipe-vs-trainer-state

Verdict: pass for baseline/procedural; intentional non-dispatchable status for
null control. The trainer `_full_main` is implemented and `local_pre_deploy`
passes for the two active paired-smoke recipes.

## Live DP1 call-status blocker

Latest local plan status:

- baseline call `fc-01KS48HTG7BNTYB9AXTHZH4W4E`: exceeded 5400s cap and still
  polls `running_or_pending`.
- procedural call `fc-01KS48PF17Z6WYAHTMJKJR1GZD`: exceeded 5400s cap and still
  polls `running_or_pending`.

Therefore paired auth eval is blocked. Do not run the emitted paired auth-eval
commands until both training arms produce harvested `archive.zip` plus
`submission/` custody.

## 6-hook wire-in declaration

- Hook #1 sensitivity-map: N/A for audit memo.
- Hook #2 Pareto constraint: N/A for audit memo.
- Hook #3 bit-allocator: N/A for audit memo.
- Hook #4 cathedral autopilot dispatch: ACTIVE. This memo routes a metadata
  revision before more DP1 paid dispatch.
- Hook #5 continual-learning posterior: ACTIVE if the metadata drift is fixed
  and future dispatch attempts proceed.
- Hook #6 probe-disambiguator: N/A for audit memo; the DP1 paired baseline vs
  procedural setup remains the disambiguator once training outputs exist.

## Recommended next action

1. Add explicit `predicted_band_validation_status: pending_post_training` to
   all four DP1 recipes with a short reactivation criterion.
2. Treat the current baseline/procedural training calls as over-cap and not
   harvest-ready unless Modal later returns artifacts.
3. Before any re-dispatch, shorten the DP1 first-anchor recipe runtime or add
   a cheaper trainer knob so the 100-epoch cache-source path cannot occupy the
   full 5400s budget without producing artifacts.
