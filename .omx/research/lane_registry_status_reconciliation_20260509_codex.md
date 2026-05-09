# Lane Registry Status Reconciliation - 2026-05-09

<!-- generated_at: 2026-05-09T23:22:41Z -->
<!-- evidence_grade: registry_reconciliation; no dispatch; no lane claim -->

## Scope

Reconciled the stale Track 4, T9, and Lane 12-v1/v2 registry/status rows
flagged in `.omx/research/roadmap_outstanding_work_audit_20260509_agent.md`.

No remote job, GPU job, exact eval, or lane claim was launched. No sidecar
builder or test file was modified.

## Preflight Read

- Read `AGENTS.md`, `CLAUDE.md`, and `PROGRAM.md` before mutation.
- Read `tools/lane_maturity.py --help` before invoking mutation commands.
- Read current directive files under `.omx/research/*_directive_*` dated
  2026-05-09.
- Repo-root `MEMORY.md` is absent, matching the process-surface inconsistency
  already noted by the outstanding-work audit.

## Registry Mutations Applied Via `tools/lane_maturity.py`

### Track 4 - `track1_paradigm_delta_track4_uniward_stc_hessian_a1`

Updated stale evidence from "pending GHA dispatch" to the harvested May 9
`[contest-CPU]` negative anchor:

- `real_archive_empirical=true` now cites
  `.omx/research/track4_uniward_stc_hessian_a1_contest_cpu_anchor_20260509.md`.
- `strict_preflight=true` now cites Catalog #123 in `src/tac/preflight.py`.
- `three_clean_review=true` now cites
  `.omx/research/track4_bug_class_fix_3_clean_pass_review_20260509.md`.
- `reactivation_criteria` now records score-gradient saliency, full 600-pair
  saliency, sub-cliff savings, latent-blob STC, and non-score-gradient
  substrate checks.

Status interpretation: Track 4 remains L2 because implementation and real
archive empirical evidence are both historical facts, but the measured v1
configuration is negative and must not receive new CUDA spend without meeting
the reactivation criteria.

### T9 - `lane_t9_cross_archive_substrate_composition`

Applied the explicit operator DEFER directive from
`.omx/research/defer_t9_directive_for_a0be36e_20260509.md`:

- `impl_complete=false`.
- `three_clean_review=false`.
- Level recomputed to L0.
- `reactivation_criteria` now requires an A1 contest-CUDA substrate, at least
  one second composable contest-CUDA substrate, council approval for
  single-axis branching, or an explicit rescope to one A1 axis.

Status interpretation: T9 is a deferred sketch only. Cross-archive substrate
composition should not route implementation or dispatch work from this row.

### Lane 12-v1 - `lane_12_nerv_mask_codec`

No gate mutation was applied. The prior
`.omx/research/representation_lane_8_field_backfill_20260509.md` explicitly
states that this row should retain its historical L2 gates as forensic anchor
evidence while `research_only=true` and `reactivation_criteria` mark it
superseded by `lane_12_v2_nerv_as_renderer`.

Status interpretation: Lane 12-v1 is forensic and superseded, not active
score work and not killed.

### Lane 12-v2 - `lane_12_v2_nerv_as_renderer`

Added explicit `reactivation_criteria` for Phase B:

- self-contained contest runtime;
- differentiable scorer-preprocess gradcheck;
- PR95/PR100 training-stack parity or documented deviation;
- exact packet builder plus no-op proof;
- paired `[contest-CUDA]` and `[contest-CPU]` once shippable.

Status interpretation: Lane 12-v2 remains L1 Phase A scaffold/design. It has
no empirical score claim and must stay local until the Phase B blockers clear.

## Remaining Blocker

`tools/lane_maturity.py set-field` intentionally supports only
`lane_class`, `research_only`, `reactivation_criteria`, and
`design_evidence.*`. It does not expose a notes/status mutation surface.
Because `CLAUDE.md` says registry mutation must go through the lane-maturity
CLI, I did not hand-edit stale `notes` strings in
`.omx/state/lane_registry.json`.

Known residual note staleness:

- Track 4 `notes` still mention "pending GHA dispatch" even though gate
  evidence and reactivation criteria now point to the harvested negative.
- T9 `notes` still describe a Phase 2 building block even though the gates now
  correctly mark L0 deferred.
- Lane 12-v1 `notes` still say "Audit: Level 1" while the intended current
  interpretation is "L2 forensic anchor, research_only superseded by v2".

Any future cleanup should either extend `tools/lane_maturity.py set-field` with
a reviewed `notes`/`status` mutation surface or land an operator-approved
manual registry rescue. Until then, the gate state, `research_only`, and
`reactivation_criteria` fields are the authoritative machine-readable routing
surface.

## Verification Plan

- Run `tools/lane_maturity.py validate`.
- Regenerate `reports/lane_maturity.md`.
- Run a focused audit filter for Track 4, T9, and Lane 12 rows.

## Verification Results

Codex parent re-ran the machine gates after worker handoff:

```bash
.venv/bin/python tools/lane_maturity.py validate
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.preflight import check_lane_registry_consistent
check_lane_registry_consistent(strict=True, verbose=False)
print('lane registry consistent')
PY
git diff --check -- .omx/research/a1_sidecar_resumable_search_state_20260509_codex.md reports/lane_maturity.md .omx/research/lane_registry_status_reconciliation_20260509_codex.md
```

Observed:

- `OK - 166 lane(s) validated cleanly.`
- `lane registry consistent`
- `git diff --check` emitted no findings.
