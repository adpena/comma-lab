# Codex Findings: TAC Docs Authority

Timestamp: 2026-05-19T07:19:26Z
Actor: codex-019de465
Scope: public documentation and terminology authority; no score claim.

## Findings

1. Root README title drift
   - `README.md` still presented the public repo as `comma-lab / pact`.
   - `pact` is a valid historical/local checkout alias, but public surfaces
     should lead with `comma-lab`.
   - Fix: title now uses `# comma-lab`, with explicit alias-containment text.

2. Ambiguous TAC acronym in compliance authority
   - `docs/contest_compliance_authority.md` used "first-class TAC design path"
     without expanding the acronym in that local context.
   - Fix: expanded to "Task-Aware Compression (`tac`) design path".

3. Historical lossless plans looked live
   - `docs/superpowers/specs/2026-04-10-tac-lossless-design.md` and
     `docs/superpowers/plans/2026-04-10-tac-lossless-implementation.md` used
     bare `TAC` headings and unchecked plan syntax.
   - Fix: both now carry historical/provenance banners and expanded headings.

4. Docs tree lacked an index
   - Public readers could land directly on stale planning docs without a
     current-vs-historical map.
   - Fix: added `docs/README.md` with current public docs, historical/internal
     plan routing, and terminology guard command.

5. `comma_lab` typed-package status was inconsistent
   - Docs describe `comma_lab` as a real operations package, but package data
     only marked `tac` typed.
   - Fix: added `src/comma_lab/py.typed`, package-data entry, and contributing
     quality commands that include both `src/tac/` and `src/comma_lab/`.

## Guardrail

`tools/check_tac_terminology.py` now also verifies:

- canonical public README title;
- `pact` alias containment;
- active-repo status in root README;
- docs index presence;
- `comma_lab` typed marker;
- expanded procedural-generation TAC phrasing;
- no bare `TAC` headings in `docs/**/*.md` outside the terminology authority.

## Verification

- [empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_tac_terminology_guard.py] passed, 6 tests.
- [empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_tac_terminology.py --strict --json] passed, finding_count=0.
- [empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/check_tac_terminology.py src/tac/tests/test_tac_terminology_guard.py] passed.

