# Codex Findings: TAC Naming And Public Docs Guard (2026-05-19)

## Verdict

`tac` should remain canonically expanded as **Task-Aware Compression**, not
"Task-Aware Codec."

Reason: "compression" is the field-level objective: optimize rate/distortion
against downstream machine-perception tasks or scorers. A "codec" is a concrete
encoder/decoder, entropy coder, packet grammar, or archive/inflate runtime
inside that broader stack.

## External Terminology Grounding

- MPEG uses "Video coding for machines" for standards work around video
  bitstreams/descriptors optimized for machine-task performance:
  <https://www.mpeg.org/standards/MPEG-AI/2/>
- MPEG uses "Feature coding for machines" when the compressed representation is
  a feature/tensor/neural object rather than reconstructed RGB:
  <https://www.mpeg.org/standards/MPEG-AI/4/>
- Research literature uses "task-aware compression" / "task-aware source
  coding" / "task-oriented compression" for downstream-task-conditioned
  rate-distortion objectives.

## Changes Landed

- Extended `tools/check_tac_terminology.py` beyond canonical files so public
  docs under `docs/` cannot reintroduce "Task-Aware Codec (TAC)" or related
  stale TAC expansions.
- Added public-doc stale-claim phrases for live-leader, current-result,
  Apogee-current, direct-production-map, and venue-commitment wording.
- Added PyPI/search keywords for `task-oriented-compression`,
  `coding-for-machines`, and `feature-coding-for-machines`.
- Corrected `HANDOFF.md` to point at the actual challenge repository instead
  of the comma2k19 data repository.
- Reframed stale production/Ara/paper surfaces as historical or draft-only
  until evidence-grade exact-eval/public-disclosure authority exists.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_tac_terminology.py --strict --json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_tac_terminology_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/check_tac_terminology.py src/tac/tests/test_tac_terminology_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/check_tac_terminology.py
git diff --check -- HANDOFF.md SYSTEM_MAP.md docs/site/production_deployment.md docs/production_and_applications.md docs/paper/05_production.md docs/paper/ara/PAPER.md docs/terminology_and_boundaries.md pyproject.toml src/tac/tests/test_tac_terminology_guard.py tools/check_tac_terminology.py
```

All passed after fixing one test-fixture parent-directory error.
