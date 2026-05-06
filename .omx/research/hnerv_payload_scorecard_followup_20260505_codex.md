# HNeRV Payload Scorecard Follow-Up - 2026-05-05

## Scope

Implemented a local-only scorecard readiness slice for public HNeRV repack and
payload-anatomy follow-ups. This does not launch GPU work, build a submission,
or make a new score claim.

## Patch

- `experiments/build_hnerv_frontier_scorecard.py` now inspects a sibling
  `archive.zip` when present and fails closed if its bytes or SHA disagree with
  the exact-eval JSON provenance.
- Scorecard rows now keep both actual archive member identity and matched
  profile identity, allowing byte-preserving repacks to inherit payload anatomy
  by payload SHA even when archive SHA changes.
- The JSON/Markdown scorecard now emits byte-identical payload groups and
  section-ranked follow-up targets, explicitly marking repacks as custody
  controls until a byte-different archive and exact CUDA replay exist.

## Safety

- Standard-library ZIP/JSON inspection only; no scorer import and no GPU
  dispatch.
- Exact eval JSON remains the score-truth pointer; payload sections are
  forensic optimization routing, not promotion evidence.
- The mismatch guard prevents stale or mismatched `archive.zip` siblings from
  silently seeding readiness decisions.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_build_hnerv_frontier_scorecard.py`
  passed.
- `.venv/bin/ruff check experiments/build_hnerv_frontier_scorecard.py
  src/tac/tests/test_build_hnerv_frontier_scorecard.py` passed.
- Temp CLI smoke over the local PR105/PR106 adapter and x-repack exact-eval
  artifacts produced four rows, two byte-identical payload groups, two
  member-SHA fallback rows, and section follow-up targets.
