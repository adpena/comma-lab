# Master-gradient wire-in audit v2 - Codex

**Date:** 2026-05-19T12:25Z  
**Task:** `codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::WIRE_IN_4`  
**Directive:** `.omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md`  
**Artifact JSON:** `.omx/research/master_gradient_wire_in_audit_v2_20260519_codex.json`  
**Score claim:** false  
**Promotion eligible:** false  
**Mutation scope:** new audit ledger only plus canonical task-status rows; no shared code surfaces edited.

## Sister-WIP guard

Preflight found active sister churn on `src/tac/preflight.py`,
`experiments/modal_train_lane.py`, `src/tac/deploy/modal/call_id_ledger.py`,
`tools/operator_authorize.py`, `tools/subagent_commit_serializer.py`,
`src/tac/commit_safety/`, MPS diagnostics, and cathedral consumer WIP. This
audit intentionally avoided those write sets. It is a structural coverage
snapshot, not a code landing.

## Commands

```bash
.venv/bin/python tools/extract_canonical_tasks_from_directive.py --register-all --owner codex --actor codex_session_019de465 --session-id 019de465 --json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from pathlib import Path; from tac.canonical_task_status import register_task; row=register_task("codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::WIRE_IN_4", ".omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md", "Wire-in #4 master-gradient wire-in audit across analytical surfaces", "codex", predicted_cost_usd=0.0, actor="codex_session_019de465", session_id="019de465", repo_root=Path("."), notes="registered_from_cluster_directive_after_extractor_noop; read-only audit/write-new-ledger scope chosen to avoid active sister WIP")'
.venv/bin/python tools/canonical_task_status.py update --task-id 'codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::WIRE_IN_4' --status in_progress --actor codex_session_019de465 --session-id 019de465 --notes 'Claiming safe read-only master-gradient wire-in audit; code/preflight/Modal/cathedral write sets avoided due active sister WIP.'
.venv/bin/python tools/audit_master_gradient_wire_in_coverage.py --output .omx/research/master_gradient_wire_in_audit_v2_20260519_codex.json
.venv/bin/python tools/audit_master_gradient_wire_in_coverage.py --summary
rg -l "load_master_gradient_for_archive|MasterGradientAnchor|master_gradient" src/tac tools experiments --glob '*.py'
```

## Result

The canonical audit helper reports:

- Frontier archive coverage: 2 of 8 have a materialized master-gradient anchor.
- Authoritative-axis archive coverage: 0 of 8.
- Surface coverage: 12 of 13 canonical surfaces are active or helper-wired.
- Surface coverage moved from the inventory memo's 47.0% baseline to 92.3% in
  the current structural snapshot.
- Remaining canonical unwired surface:
  `tools.probe_alternative_reducers_latent_class_conditioning`.

The file-level scan found 169 Python files under `src/tac`, `tools`, and
`experiments` with `master_gradient` references. Most are tests, tooling,
comments, or lower-level producer/consumer internals; the committed JSON keeps
the canonical 13-surface inventory separate from that broad grep count so the
coverage percentage is not inflated by test files or incidental mentions.

## Authority interpretation

This is the important distinction:

- Surface wiring is now mostly present structurally.
- Archive authority is still not contest-promotable because the two live anchors
  are `[macOS-CPU advisory]` and `is_authoritative_axis=false`.

Any consumer that needs a leaderboard or dispatch authority signal must continue
to fail closed until exact contest-axis anchor custody exists for the target
archive. The v2 audit therefore supports planner observability and queue
prioritization only; it does not authorize score promotion, rank/kill decisions,
or exact-dispatch readiness.

## Follow-ups

1. Resolve the sole remaining canonical unwired surface:
   `tools.probe_alternative_reducers_latent_class_conditioning`. If the tool is
   still planned but absent, create the Atom against the planned reducer probe
   rather than citing it as landed implementation.
2. Materialize authoritative master-gradient anchors on the contest axis for
   the frontier archives that are currently placeholder-only. This is the real
   authority bottleneck; structural consumers cannot substitute for custody.
3. Re-run this audit after the active per-byte sensitivity consumer and HF Jobs
   dispatcher land, because the current worktree contains visible sister WIP in
   both of those areas.

## Verification status

The JSON report validates through `python -m json.tool`. Canonical task status
validated before task closure and should be validated again after the completion
row is appended.
