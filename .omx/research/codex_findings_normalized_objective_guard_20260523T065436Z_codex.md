# Codex Findings - Normalized Objective Boundary Guard

- **UTC:** 2026-05-23T06:54:36Z
- **Lane:** `lane_codex_normalized_objective_guard_20260523`
- **Scope:** MLX/window response objective normalization across planner boundaries
- **Authority:** non-score planning hardening; no score claim

## Finding

Most MLX/DQS1 consumers already use normalized full-video fields rather than
raw singleton-window gains. The remaining bug class is boundary trust: downstream
selection and portfolio planners accepted normalized-looking JSON fields without
recomputing the invariant from raw gain, source sample count, denominator,
archive-byte delta, and the contest rate term.

That matters because a hand-built or stale selection artifact could carry raw
singleton gain in `normalized_full_video_scorer_gain_vs_baseline` and look
plausible to later queue/portfolio code.

## Fix Landed

- Added `tac.optimization.normalized_objective`:
  - `compute_normalized_full_video_gain(...)`;
  - `normalized_full_video_objective_metrics(...)`;
  - `require_normalized_full_video_objective(...)`.
- Wired MLX effective spend-triage selection to recompute:
  - normalized gain = `observed_gain * source_n_samples / 600`;
  - projected full-video delta = `rate_delta - normalized_gain`;
  - normalized byte margin = `normalized_gain / RATE_SCORE_PER_BYTE - added_bytes`.
- Wired cross-family MLX portfolio ingestion to require:
  - normalized selection basis;
  - explicit `requires_exact_auth_eval_before_score_claim=true`;
  - recomputed normalized-objective consistency.
- Updated selection Markdown/report labels so normalized gain/margin are primary
  and raw window gain is explicitly labeled.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_normalized_objective.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py
```

Result: `26 passed`.

```bash
.venv/bin/python -m ruff check src/tac/optimization/normalized_objective.py src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/optimization/cross_family_candidate_portfolio.py src/tac/tests/test_normalized_objective.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py
```

Result: `All checks passed!`.

```bash
git diff --check
```

Result: passed.

## Remaining Work

- Reuse the guard in scorer-response summary/planning, DQS selector/packet/
  feedback boundaries, and learned-sweep candidate ingestion.
- Add a small preflight/report check that selected-row renderers do not display
  raw singleton gain as the primary score-lowering objective.
