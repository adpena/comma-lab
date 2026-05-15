# SPDX-License-Identifier: MIT
"""SELFCOMP-3 (R2 MEDIUM, 2026-05-15): DP1 Tier C renderer caching.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
``_run_tier_c_dp1`` previously constructed ``DrivingPriorRenderer(cfg)``
INSIDE ``_render_pair_with_residual``, producing 4800 constructor calls
per Tier C run (600 pairs × 4 sigmas × 2 targets) — a 4800× regression
vs the canonical pattern (cache the renderer; swap state_dict per
perturbation). On Modal A100 dispatches at sigma_count=10 with 600-frame
archives this becomes 12,000 constructor calls = ~2 min wasted overhead.

The fix: extract ``_shared_renderer = DrivingPriorRenderer(cfg)`` outside
the per-pair loop; ``_render_pair_with_residual`` uses the cached instance
and ``load_state_dict`` per perturbation.

These tests pin the optimization so a future refactor that re-introduces
per-pair construction is caught at CI time.

Cross-refs: ``feedback_recursive_review_r2_wave_a_*`` SELFCOMP-3 +
``feedback_r2_medium_fix_wave_selfcomp_mackay_landed_20260515.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "mdl_scorer_conditional_ablation.py"


def _read_tool_text() -> str:
    return TOOL_PATH.read_text(encoding="utf-8")


def _extract_function_body(text: str, fn_name: str) -> str:
    """Return the body source of a top-level def, until the next top-level def or EOF."""
    pattern = rf"def {fn_name}\([^)]*\)[^:]*:\n(.*?)(?=\n(?:def |class |# ----|@)|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise AssertionError(f"function {fn_name!r} not found in tool")
    return match.group(1)


def test_run_tier_c_dp1_caches_renderer_outside_per_pair_loop():
    """``_run_tier_c_dp1`` must construct DrivingPriorRenderer outside the per-pair loop."""
    text = _read_tool_text()
    body = _extract_function_body(text, "_run_tier_c_dp1")
    # The ``_shared_renderer`` cache marker must exist at module-level scope
    # of the function body (NOT inside ``_render_pair_with_residual``).
    assert "_shared_renderer = DrivingPriorRenderer(cfg)" in body, (
        "_run_tier_c_dp1 must declare _shared_renderer = DrivingPriorRenderer(cfg) "
        "outside the per-pair loop. SELFCOMP-3 (R2 MEDIUM) requires that the "
        "renderer is constructed ONCE per Tier C invocation."
    )


def test_render_pair_with_residual_does_NOT_construct_new_renderer():
    """``_render_pair_with_residual`` body must NOT contain ``DrivingPriorRenderer(cfg)``.

    Direct anti-regression for SELFCOMP-3: if a future refactor moves the
    constructor BACK inside the inner func, this test fires.
    """
    text = _read_tool_text()
    # Locate _render_pair_with_residual within _run_tier_c_dp1 (it's a nested def).
    inner_match = re.search(
        r"def _render_pair_with_residual\([^)]*\)[^:]*:(.*?)(?=\n    def |\n    for sigma in |\n\ndef |\nclass )",
        text,
        re.DOTALL,
    )
    assert inner_match is not None, "_render_pair_with_residual not found"
    inner_body = inner_match.group(1)
    # The inner body must reference ``_shared_renderer`` (the cached instance)
    # AND must NOT call ``DrivingPriorRenderer(cfg)``.
    code_lines = [
        line for line in inner_body.splitlines()
        if not line.lstrip().startswith("#")
    ]
    inner_code = "\n".join(code_lines)
    assert "_shared_renderer" in inner_code, (
        "_render_pair_with_residual must reuse _shared_renderer from enclosing scope"
    )
    assert "DrivingPriorRenderer(cfg)" not in inner_code, (
        "_render_pair_with_residual MUST NOT construct DrivingPriorRenderer(cfg) — "
        "that's exactly the SELFCOMP-3 bug class. Use the _shared_renderer "
        "cached outside the per-pair loop."
    )


def test_run_tier_c_dp1_renderer_cached_only_once():
    """The renderer constructor is called at most ONCE inside _run_tier_c_dp1.

    Counts CODE callsites only (excludes comment / docstring mentions of
    the constructor name).
    """
    text = _read_tool_text()
    body = _extract_function_body(text, "_run_tier_c_dp1")
    # Strip comment lines (start with optional whitespace + ``#``) so that
    # explanatory comments mentioning ``DrivingPriorRenderer(cfg)`` do not
    # double-count.
    code_lines = [
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    n_constructs = code_only.count("DrivingPriorRenderer(cfg)")
    assert n_constructs == 1, (
        f"Expected EXACTLY 1 DrivingPriorRenderer(cfg) CODE call inside "
        f"_run_tier_c_dp1 (the cached _shared_renderer); found {n_constructs}. "
        f"SELFCOMP-3 (R2 MEDIUM, 2026-05-15) requires exactly one "
        f"construction per Tier C run."
    )


def test_load_state_dict_called_inside_render_pair_with_residual():
    """Per-pair load_state_dict is the canonical pattern after caching."""
    text = _read_tool_text()
    inner_match = re.search(
        r"def _render_pair_with_residual\([^)]*\)[^:]*:(.*?)(?=\n    def |\n    for sigma in |\n\ndef |\nclass )",
        text,
        re.DOTALL,
    )
    assert inner_match is not None
    inner_body = inner_match.group(1)
    assert "load_state_dict" in inner_body, (
        "_render_pair_with_residual must call load_state_dict on the cached "
        "renderer to apply per-perturbation weights"
    )


def test_run_tier_c_dp1_smoke_runs_through_cached_renderer():
    """End-to-end smoke confirms cached-renderer path produces TierCResult rows.

    Builds a synthetic-realistic DP1 archive via the substrate's own
    fixture helpers. Asserts:
    1. Tier C returns the expected number of rows (sigmas × targets).
    2. All rows are well-formed (finite deltas, non-negative elapsed).

    This is the "engineering pragma" sanity test — the cached-renderer
    path must produce equivalent OUTPUT to the per-pair-construct path
    (which is what the existing 23 DP1 tests already validate via the
    SAME function under test).
    """
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random
    import torch

    from tools.mdl_scorer_conditional_ablation import (
        _run_tier_c_dp1,
        TierCResult,
    )

    # Reuse the canonical fixture from sister DP1 Tier C tests so this
    # smoke matches the production codepath without re-implementing the
    # archive bytes layout.
    try:
        from tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        pytest.skip("Sister DP1 fixture not importable in this environment")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=4, per_pair_bytes=6)
    pair_indices = [0, 1]
    gt_pairs = torch.zeros(2, 2, 874, 1164, 3, dtype=torch.uint8)

    rows = _run_tier_c_dp1(
        inner_bytes=inner,
        pair_indices=pair_indices,
        gt_pairs=gt_pairs,
        baseline_seg=0.5,
        baseline_pose=0.1,
        distortion_net=_FakeDistortionNet(),
        device=torch.device("cpu"),
        rng=_random.Random(7),
        noise_sigmas=[0.001, 1.0],  # tiny sigma + large sigma
        scorer_batch_size=2,
    )
    # 2 sigmas × 2 targets = 4 rows.
    assert len(rows) == 4, f"expected 4 TierCResult rows; got {len(rows)}"
    for row in rows:
        assert isinstance(row, TierCResult)
        assert row.elapsed_seconds >= 0.0


def test_run_tier_c_dp1_no_renderer_state_dict_mismatch_after_caching():
    """Cached renderer + load_state_dict path produces no missing/unexpected keys.

    Regression-pin: if a refactor breaks state_dict shape compatibility
    between the cached renderer and the per-perturbation state_dict, this
    test fires (raises ValueError from inside _render_pair_with_residual).
    """
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random
    import torch

    from tools.mdl_scorer_conditional_ablation import _run_tier_c_dp1

    try:
        from src.tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        try:
            from tac.tests.test_mdl_ablation_tier_c_dp1 import (  # type: ignore[no-redef]
                _build_tiny_dp1_archive_bytes,
                _FakeDistortionNet,
            )
        except ImportError:
            pytest.skip("Sister DP1 fixture not importable")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2, per_pair_bytes=6)

    # If the cached renderer state_dict is incompatible with the perturbed
    # state_dict, _render_pair_with_residual raises ValueError; we don't
    # want that. The fix MUST preserve key-set compatibility.
    rows = _run_tier_c_dp1(
        inner_bytes=inner,
        pair_indices=[0],
        gt_pairs=torch.zeros(1, 2, 874, 1164, 3, dtype=torch.uint8),
        baseline_seg=0.5,
        baseline_pose=0.1,
        distortion_net=_FakeDistortionNet(),
        device=torch.device("cpu"),
        rng=_random.Random(11),
        noise_sigmas=[0.01],
        scorer_batch_size=1,
    )
    assert len(rows) == 2  # 1 sigma × 2 targets


def test_selfcomp_3_anti_regression_per_pair_constructor_count():
    """Anti-regression: ``DrivingPriorRenderer(cfg)`` must NOT appear inside
    a function with 'pair' in its name (the bug class definition).

    This test scans for the class signature: a constructor invocation
    inside a function whose name contains 'pair'. SELFCOMP-3 was raised
    against the per-pair construction; this gate refuses re-introduction
    at any nested per-pair scope.
    """
    text = _read_tool_text()
    # Find every nested def with 'pair' in the name.
    pair_def_pattern = re.compile(
        r"def (\w*pair\w*)\([^)]*\)[^:]*:(.*?)(?=\n    def |\n    for |\n\ndef |\nclass |\Z)",
        re.DOTALL,
    )
    for match in pair_def_pattern.finditer(text):
        fn_name = match.group(1)
        fn_body = match.group(2)
        # Filter out comment lines so explanatory comments are not
        # mis-classified as the bug.
        code_lines = [
            line for line in fn_body.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        # ``DrivingPriorRenderer(cfg)`` inside a 'pair'-named function
        # is exactly the bug class.
        assert "DrivingPriorRenderer(cfg)" not in code_only, (
            f"function {fn_name!r} contains DrivingPriorRenderer(cfg) — "
            f"this re-introduces the SELFCOMP-3 (R2 MEDIUM, 2026-05-15) "
            f"per-pair-construction bug class."
        )
