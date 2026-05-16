# L5 scaffold signal capture and adversarial greenup

Date: 2026-05-16

## Objective

Preserve the partner WIP `registered_substrate.py` signal for the Tishby
IB-pure L5/asymptotic-pursuit scaffold without silently leaving it orphaned.
While validating that capture, additional partner WIP landed in Z6, Rudin
remote-driver, L5 v2, operator-briefing, and Catalog #311 surfaces; this
ledger records the coherent no-signal-loss greenup boundary.

## Captured artifact

- Source file: `src/tac/substrates/tishby_ib_pure/registered_substrate.py`
- Contract id: `tishby_ib_pure`
- Lane id: `lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516`
- Research status: `recipe_research_only=True`, `score_claim=False`
- Probe hook: `tools/check_variational_ib_tractability.py`
- Design provenance:
  `.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md`

Additional captured partner surfaces:

- `experiments/train_substrate_tishby_ib_pure.py`
- `experiments/train_substrate_time_traveler_l5_z6.py`
- `scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
- `scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh`
- `src/tac/autopilot_rudin_daubechies/compressive_sensing_lattice_recovery.py`
- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/preflight.py` Catalog #311 tightening
- `tools/operator_briefing.py`

## No-signal-loss integration

The untracked partner file was not rewritten or discarded. It was wired into
the package import surface through `TISHBY_IB_PURE_CONTRACT`, so normal
`tac.substrates.tishby_ib_pure` imports trigger the same `@register_substrate`
validation as direct `registered_substrate.py` imports.

The Tishby smoke trainer was adjusted to match the actual L1 facade contract:
it no longer assumes unlanded neural `encoder` / `decoder` attributes, writes a
small TIBP1 metadata archive, tags `score_claim=false`, and records
`[diagnostic-CPU; tishby_ib_pure_smoke]` rather than a contest axis.

The Z6 smoke path now caps smoke epochs at 3 even when a full-run default such
as 300 is supplied through wrapper defaults. This prevents the smoke wrapper
from accidentally becoming a full-training run before Phase-2 lift.

Two adversarial-review failures found by the focused suite were fixed:

- pairwise coherence uses canonical unordered keys, so the parent/child test now
  checks `("child", "parent")` instead of the stale ordered key.
- Catalog #311 now accepts the valid phrase `pose-conditioned autoregressive
  predictor` as an ego-motion-conditioned predictive-coding declaration.

## Verification

Commands run before commit:

```bash
.venv/bin/python -m ruff check src/tac/substrates/tishby_ib_pure
.venv/bin/python -m pytest src/tac/substrates/tishby_ib_pure/tests -q
.venv/bin/python - <<'PY'
import importlib
mod = importlib.import_module('tac.substrates.tishby_ib_pure.registered_substrate')
print(mod.TISHBY_IB_PURE_CONTRACT.id)
PY
.venv/bin/python -m pytest \
  src/tac/autopilot_rudin_daubechies/tests/test_compressive_sensing_lattice_recovery.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_check_311_predictive_coding_ego_motion_conditioning.py \
  src/tac/substrates/tishby_ib_pure/tests \
  src/tac/substrates/time_traveler_l5_z6/tests -q
.venv/bin/python experiments/train_substrate_tishby_ib_pure.py \
  --output-dir /tmp/pact_tishby_smoke2 --epochs 3 --device cpu --smoke
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --output-dir /tmp/pact_z6_cap_smoke2 --smoke --epochs 300 --batch-size 1 --device cpu
bash -n scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh \
  scripts/remote_lane_substrate_time_traveler_l5_z6.sh
```

Observed evidence:

- Ruff passed on the Tishby package.
- Targeted Ruff passed on all changed non-preflight Python surfaces. Full-file
  `src/tac/preflight.py` Ruff remains noisy from pre-existing repository-wide
  style debt, so this pass used py_compile plus focused Catalog #311 tests for
  the changed preflight section.
- Tishby package tests passed, including contract registration from package
  import: `4 passed`.
- Direct `registered_substrate.py` import returned `tishby_ib_pure`.
- Expanded focused suite passed: `230 passed in 61.33s`.
- Tishby smoke produced `archive_bytes=275`, `score_claim=false`,
  `score_axis=diagnostic_cpu`, `research_only=true`.
- Z6 smoke with `--epochs 300` produced `epochs=3`, `smoke_epoch_cap=3`,
  `evidence_grade=smoke-no-scorer`, and no score claim.
- Both touched remote lane scripts passed `bash -n`.

Additional final greenup after the L5 v2/lattice/Rudin/Z6 hardening pass
on this worktree:

```bash
.venv/bin/python -m ruff check \
  experiments/train_substrate_tishby_ib_pure.py \
  experiments/train_substrate_time_traveler_l5_z6.py \
  src/tac/autopilot_rudin_daubechies/compressive_sensing_lattice_recovery.py \
  src/tac/autopilot_rudin_daubechies/tests/test_compressive_sensing_lattice_recovery.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/substrates/rudin_floor_interpretable_ml/tests/test_rudin_floor_l1_scaffold.py \
  src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py \
  src/tac/substrates/tishby_ib_pure \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_check_311_predictive_coding_ego_motion_conditioning.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  tools/operator_briefing.py
.venv/bin/python -m py_compile \
  experiments/train_substrate_tishby_ib_pure.py \
  experiments/train_substrate_time_traveler_l5_z6.py \
  src/tac/preflight.py \
  src/tac/autopilot_rudin_daubechies/compressive_sensing_lattice_recovery.py \
  src/tac/optimization/l5_staircase_v2.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  tools/operator_briefing.py \
  src/tac/substrates/tishby_ib_pure/registered_substrate.py
bash -n \
  scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh \
  scripts/remote_lane_substrate_time_traveler_l5_z6.sh \
  scripts/remote_lane_substrate_tishby_ib_pure.sh
.venv/bin/python -c "import pathlib, yaml; p=pathlib.Path('.omx/operator_authorize_recipes/substrate_tishby_ib_pure_modal_a100_dispatch.yaml'); data=yaml.safe_load(p.read_text()); assert data['dispatch_enabled'] is False; assert data['research_only'] is True"
.venv/bin/python -m pytest \
  src/tac/autopilot_rudin_daubechies/tests/test_compressive_sensing_lattice_recovery.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_check_311_predictive_coding_ego_motion_conditioning.py \
  src/tac/substrates/tishby_ib_pure/tests \
  src/tac/substrates/time_traveler_l5_z6/tests -q
.venv/bin/python experiments/train_substrate_tishby_ib_pure.py \
  --output-dir /tmp/pact_tishby_smoke_final --epochs 3 --device cpu --smoke
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --output-dir /tmp/pact_z6_cap_smoke_final --smoke --epochs 300 --batch-size 1 --device cpu
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_tishby_ib_pure_modal_a100_dispatch --dry-run
.venv/bin/python tools/lane_maturity.py validate
.venv/bin/python -c "from tac.preflight import check_register_substrate_contract_fields_canonical, check_predictive_coding_substrate_design_has_ego_motion_conditioning; print(len(check_register_substrate_contract_fields_canonical(repo_root='.', strict=False, verbose=False))); print(len(check_predictive_coding_substrate_design_has_ego_motion_conditioning(repo_root='.', strict=False, verbose=False)))"
git diff --check
```

Observed final evidence:

- Targeted Ruff passed on all changed non-preflight Python surfaces.
- py_compile passed on the changed Python surfaces including `src/tac/preflight.py`.
- `bash -n` passed on all three touched remote-lane scripts.
- Tishby operator-authorize dry-run refused dispatch because
  `dispatch_enabled=false` and listed the D4/VIB/Dykstra/path-council blockers.
- Focused L5/L5 v2/autopilot/Tishby/Z6 tests passed: `240 passed in 63.90s`.
- `tools/lane_maturity.py validate` passed: `773 lane(s) validated cleanly`.
- Tishby smoke produced `archive_bytes=275`, `score_claim=false`,
  `score_axis=diagnostic_cpu`, `research_only=true`.
- Z6 smoke with full-run epoch value capped correctly:
  `requested_epochs=300`, `epochs=3`, `smoke_epoch_cap=3`.
- Substrate contract-field preflight returned 0 findings.
- Catalog #311 stricter check is still warn-only and returned 11 existing
  design-memo findings after the ego-motion-plus-predictive-token tightening.
- `git diff --check` passed.

Adversarial reviewer follow-up (same turn) found four issues before commit:

- Z6/Rudin wrappers required a job id but did not verify a live active claim.
  Fixed by calling `tools/claim_lane_dispatch.py summary --live-only --format
  json` and refusing startup unless `(lane_id, instance_job_id)` is active.
- Z6 wrapper defaulted `Z6_EPOCHS` to 3, which protected smoke but would
  under-train a future full run. Fixed by restoring wrapper default 300 and
  relying on trainer-side `_smoke_effective_epochs()` to cap `--smoke`.
- Cathedral lattice diagnostic was output-only. Fixed by adding a candidate
  blocker whenever `recovery_regime != EXACT`, preventing operator-authorized
  self-dispatch against an unreliable lattice posterior.
- L5 v2 readiness still displayed stale L1 scaffold next-action IDs after L1
  scaffolds landed. Fixed by marking effective next actions
  `completed_or_superseded:*` and setting
  `ready_for_recommended_next_action=false` when the L1 artifact set exists.

Post-fix focused suite passed: `248 passed in 18.58s`, targeted Ruff passed,
`bash -n` passed on all three touched remote scripts, and py_compile passed on
the changed executable Python surfaces.

Post-fix repo gates also passed: `tac.preflight --no-codebase`, lane maturity
validation (`773 lane(s) validated cleanly`), Z6 smoke (`requested_epochs=300`,
`epochs=3`, `smoke_epoch_cap=3`), Tishby diagnostic smoke (`score_claim=false`,
`research_only=true`, `score_axis=diagnostic_cpu`, `roundtrip_ok=true`),
Cathedral lattice smoke with the FAILED-regime dispatch blocker surfaced, and
operator briefing JSON with all three L5 v2 sample next actions marked
`completed_or_superseded:*`.

## Frontier relevance

This does not create a score claim. It prevents the Tishby IB-pure L5 scaffold
from becoming an orphaned META-layer signal and keeps its Phase-2 lift path
queryable by the substrate registry, autopilot hooks, and future paired-axis
dispatch planning.
