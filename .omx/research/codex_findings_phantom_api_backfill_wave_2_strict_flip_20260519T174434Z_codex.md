# Codex Findings - Catalog #287 Phantom-API Backfill Wave 2 Strict Flip

**UTC:** 2026-05-19T17:44:34Z
**Task:** `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_C`
**Lane:** `lane_phantom_api_backfill_wave_2_20260519`
**Verdict:** PROCEED, implemented
**Score claim:** none

## Finding

Catalog #287 was still warn-only because two sub-scopes had residual live
violations:

- Sub-scope A: legacy source comments/docstrings with percentage or multiplier
  claims but no adjacent evidence-grade tag.
- Sub-scope B: `.omx/research` memos citing non-importable `tac.*` helper names
  as if they were executable APIs.

This was a false-authority risk, not a score signal. The same prose could be
read by future agents as implemented infrastructure even when the importable
surface did not exist.

## Work Landed

- Recomputed the live Catalog #287 baseline: 229 total violations, with 207
  from `.omx/research` phantom-API citations.
- Preserved old `.omx/research` dated memos as historical provenance: the
  initial in-place waiver sweep was reverted after sister review identified the
  append-only contract conflict.
- Added tracked append-only waiver authority in
  `.omx/state/catalog_287_phantom_api_waivers.jsonl`, with exact
  `(relpath, line, dotted)` plus line-SHA entries for historical/proposal
  citations that are not executable API authority.
- Tightened the waiver semantics after sister adversarial review: file-level
  `PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE` suppresses proposal-only prose, but it
  no longer suppresses active authority lines. Lines that still use active
  language such as `ACTIVE`, `wire`, `hook`, `consume`, `register`, or `dispatch`
  must now cite an importable module or a real attribute on an importable parent
  module, or be waived by exact sidecar entry.
- Closed the Rawls false-authority gap: a parent import plus callable-looking
  terminal token no longer passes unless `hasattr(parent_module, terminal)` is
  true. Examples like `tac.unified_action.S_total` and
  `tac.master_gradient_consumers.adjust_predicted_delta_for_venn_classification_v2`
  now require explicit historical/proposal waiver authority.
- Added explicit `[prediction]`, `[MPS-research-signal]`, or `[advisory only]`
  tags next to residual source multiplier/percentage claims.
- Flipped the preflight orchestrator callsite to
  `check_no_docstring_overstatement_without_evidence_tag(strict=True)`.
- Fixed a `python -m tac.preflight` authority-ordering bug exposed by the
  slow all-scope sweep: the CLI entrypoint now defers through
  `_preflight_cli_main()` at EOF, so later Catalog #324/#325/#326 gate
  definitions exist before Catalog #286 verifies catalog-row callables.
- Appended the CLAUDE.md Catalog #287-v2 strict-flip note.

## Verification

Direct strict invocation passes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("preflight", "src/tac/preflight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.check_no_docstring_overstatement_without_evidence_tag(
    repo_root=Path("."),
    strict=True,
    scan_research_memos=True,
    scan_memory_files=False,
)
print("strict-ok")
PY
```

Observed result: `strict-ok`.

Durable count summary:

- Initial Catalog #287 direct strict baseline: 229 violations.
- Research memo phantom-helper sub-scope: 207 violations.
- Source evidence-tag sub-scope: 22 violations.
- Rawls-hardening pass after exact `hasattr` authority semantics: 325 raw
  findings across restored historical memos, collapsed to 323 exact
  sidecar-waiver entries after duplicate `(relpath, line, dotted)` references
  were deduplicated.
- Final direct Catalog #287 strict scan: 0 violations.

## Residual Blocker Left Intact

Full `preflight_all()` now reaches a different global dispatch blocker:
`check_substrate_at_optimal_form_before_paid_dispatch` reports 17 substrate
lanes at LIFTED-TRAINER form with outstanding sextet/grand-council
`PROCEED_WITH_REVISIONS` verdicts. That is a real substrate/council authority
blocker and was intentionally not weakened or papered over in this Codex
landing. The prior CLI-only Catalog #324/#325/#326 phantom-callable report was
fixed by moving the `python -m tac.preflight` execution point to EOF.

## Authority Boundary

This landing does not validate any score, substrate, rate attack, or archive.
It only prevents source and research prose from presenting untagged empirical
claims or phantom helper names as executable authority.

## 6-Hook Wire-In

1. Sensitivity map: N/A; defensive text/prose authority gate.
2. Pareto constraint: N/A.
3. Bit allocator: N/A.
4. Cathedral autopilot dispatch: ACTIVE; autopilot-facing research prose can no
   longer cite phantom helper names without import authority or waiver.
5. Continual-learning posterior: ACTIVE indirectly through cleaner research
   memo authority and the canonical task-status completion row.
6. Probe disambiguator: ACTIVE; strict gate distinguishes importable source
   authority from proposal-only prose.
