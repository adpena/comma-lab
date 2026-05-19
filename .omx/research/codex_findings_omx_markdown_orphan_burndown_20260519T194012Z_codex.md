# Codex Findings - OMX Markdown Orphan Burndown

**UTC:** 2026-05-19T19:40:12Z  
**Actor:** Codex  
**Source sweep:** `.omx/research/codex_findings_omx_markdown_directive_orphan_sweep_20260519T190927Z_codex.md`  
**Score claim:** none.

## Verdict

The `.omx` sweep produced concrete closures, not only inventory. This pass
converted five orphan/design-memo findings into reusable code, tests, or
operator-facing control surfaces.

| Finding | Closure | Promotion stance |
|---|---|---|
| SIREN placeholder still in canonical default sweeps | Removed `siren_renderer` from pre-entropy and Q6 default authority maps until a real trained payload exists; Q6 keeps C(10,2) geometry via `distilled_segnet`. | no score claim |
| PR106 PacketIR runtime-consumption rows stranded | Regenerated runtime-consumption evidence and candidate matrix; 15/16 candidates now have next exact-eval targets, while `format_0x04_rank_elided` fails closed on current-runtime decode exception. | no promotion; paired exact eval still required |
| Catalog #309 horizon-class parser false-positive | `**horizon_class:** plateau_adjacent` style fields now satisfy the gate; focused regression added. | documentation-gate only |
| `pyppmd` import compliance drift | Added strict package-code scanner for unwaived `pyppmd` imports; legacy PPMd replay/HPAC paths now carry explicit narrow `PYPPMD_LGPL_OK:` waivers. | OSS/compliance guard |
| Master-gradient alternative-reducer consumer marked UNWIRED | Alternative-reducer manifests can request diagnostic Wyner-Ziv covariance via `tac.master_gradient_consumers.wyner_ziv_side_info_covariance`; driver exposes CLI flags; audit now reports zero unwired surfaces. | diagnostic only |
| Trainer optimization-helper directive unmirrored | Added reusable AST audit module + operator CLI + warn-only preflight hook. Live surface: 49 trainers scanned, 24 accepted, 25 missing, 0 waived. | backfill queue, not strict yet |

## PacketIR Runtime Consumption

The PacketIR row changed from "stranded" to an evidence-backed queue:

- `runtime_consumed_needs_paired_exact_eval`: 2
- `single_axis_exact_measured_needs_pair`: 13
- `runtime_consumption_blocked`: 1 (`format_0x04_rank_elided`)
- `next_exact_eval_target_count`: 15

The format `0x04` candidate now proves the important negative correctly:
packet accounting can pass while the current runtime decoder still rejects the
rank-elided format. The manifest keeps `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Trainer Optimization Helper Audit

New reusable surfaces:

- `src/tac/trainer_optimization_helper_audit.py`
- `tools/check_trainer_optimization_helpers.py`
- `check_substrate_trainers_use_canonical_optimization_helpers(...)` in
  `tac.preflight`

The check uses AST import-plus-call evidence and tokenized comment waivers.
Docstrings, comments, and copied examples do not satisfy the contract. It is
warn-only in `preflight_all()` because the live count is intentionally nonzero.

Current live counts:

- scanned trainers: 49
- canonical/direct helper accepted: 24
- missing helper or waiver: 25
- explicit waivers: 0

## Verification

Focused green tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_extract_canonical_tasks_from_directive.py \
  src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py \
  src/tac/tests/test_q6_preprobe_extended.py \
  src/tac/tests/test_pr106_packetir_candidate_matrix.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py::test_pr106_runtime_consumption_fails_closed_for_format04_runtime_gap \
  src/tac/tests/test_check_309_horizon_class_declaration.py \
  src/tac/tests/test_check_no_unwaived_pyppmd_imports.py \
  src/tac/tests/test_probe_alternative_reducers.py \
  src/tac/tests/test_trainer_optimization_helper_audit.py
```

Ruff was run on touched Python surfaces. `tools/check_trainer_optimization_helpers.py`
reports `scanned=49 accepted=24 missing=25 waived=0`.

## Remaining Work

1. Backfill or waive the 25 trainer optimization-helper rows, starting with the
   six scorer-hot-loop trainers surfaced by the audit.
2. Dispatch the 15 PR106 PacketIR next exact-eval targets only through normal
   lane-claim and paired-axis custody.
3. Continue reconciling the 43 newly extractable `OP-N` rows; do not bulk
   register them without sister-work absorption checks.
