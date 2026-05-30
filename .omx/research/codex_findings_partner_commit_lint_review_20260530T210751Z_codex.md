# Codex Findings: Partner Commit Lint Review

UTC: 2026-05-30T21:07:51Z

## Scope

Adversarial review of already-committed partner deltas on `main` relative to
`origin/main`, focused on score-lowering infrastructure risk rather than
absorbing unrelated worktree changes.

## Evidence

- Targeted partner regression suites passed:
  - `src/tac/substrates/_shared/mamba2_ssd/tests/test_mamba2_ssd.py`
  - `src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py`
  - `src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_substrate_wire_in.py`
  - `src/tac/master_gradient_pose_vulnerability/tests/test_pose_vulnerability_map.py`
  - `src/tac/scorer_surrogate/posenet_mae_v/tests/test_surrogate.py`
  - `src/tac/composition/alaska_inverse_steganalysis_patterns/tests`
  - `src/tac/composition/fridrich_school_inverse_steganalysis_patterns/tests`
  - `src/tac/composition/yuv6_chroma_subsampled_perturbation_operator/tests`
- Result: 336 focused tests passed across those surfaces.
- No reviewed committed delta granted proxy, MLX-only, or prose-only rows new
  score authority.

## Finding

Full lint over committed partner Python files in `origin/main..HEAD` failed with
137 Ruff violations. The failures are concentrated in the ALASKA/Fridrich
pattern packages, Mamba2 SSD shared substrate, PR95-faithful tests, probe
outcome helper tests/tools, and PoseNet MAE-V surrogate import surfaces.

Representative classes:

- unsorted imports and unsorted `__all__` exports;
- obsolete `typing.Tuple`/`typing.Mapping`/`typing.Sequence` usage;
- unused imports and unused `noqa` directives;
- invalid escape sequences in docs;
- avoidable boolean-return branches and missing `datetime.UTC`;
- test regex strings that should be raw strings.

## Verdict

This is not a score-authority breach, but it is a real maintainability and
automation blocker. Do not silently absorb the partner lint debt into unrelated
contract-first score-lowering commits. Fix it in a dedicated cleanup slice or
require the owning partner lane to land a lint-clean follow-up before extending
those packages.

## Next Action

Keep the current contract-first archive-bound candidate migration scoped. Any
future acquisition or materializer wiring that touches the affected partner
packages should first make the local package lint-clean so score-lowering work
does not inherit preventable automation friction.
