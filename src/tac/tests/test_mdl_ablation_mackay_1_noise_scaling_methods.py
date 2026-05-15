# SPDX-License-Identifier: MIT
"""MACKAY-1 (R2 MEDIUM, 2026-05-15): Tier C noise-scaling-method parameter.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
``_run_tier_c_*`` previously used per-tensor std-based noise scaling
(``rel_std = v.std().clamp(min=1e-8); noise = randn * (rel_std * sigma)``)
which assumes the per-tensor weight distribution is approximately
Gaussian. For substrates with sparse / long-tail weight distributions
(DP1's coordinate-MLP at hidden=64 has very different shape than A1's
HNeRV decoder at 162KB), per-tensor std UNDERESTIMATES the perturbation
magnitude needed to reach a fixed fractional information loss.

The fix adds a ``noise_scaling_method`` kwarg to all 4 ``_run_tier_c_*``
funcs (default ``"gaussian_std"`` for back-compat) routing through a
canonical ``_compute_noise_scale`` helper supporting:

* ``"gaussian_std"`` — std(W).clamp(min=1e-8); legacy default.
* ``"mad_robust"``  — MAD(W) * 1.4826; robust to long-tail W.
* ``"iqr_robust"``  — IQR(W) / 1.349; robust to long-tail W.

These tests pin the API + math + back-compat invariants.

Cross-refs: ``feedback_recursive_review_r2_wave_a_*`` MACKAY-1 +
``feedback_r2_medium_fix_wave_selfcomp_mackay_landed_20260515.md``.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "mdl_scorer_conditional_ablation.py"


def _import_helper():
    from tools.mdl_scorer_conditional_ablation import (
        DEFAULT_TIER_C_NOISE_SCALING_METHOD,
        TIER_C_NOISE_SCALING_METHODS,
        _compute_noise_scale,
        _run_tier_c_a1,
        _run_tier_c_dp1,
        _run_tier_c_ibps1,
        _run_tier_c_pr106,
    )
    return {
        "DEFAULT": DEFAULT_TIER_C_NOISE_SCALING_METHOD,
        "METHODS": TIER_C_NOISE_SCALING_METHODS,
        "_compute_noise_scale": _compute_noise_scale,
        "_run_tier_c_a1": _run_tier_c_a1,
        "_run_tier_c_pr106": _run_tier_c_pr106,
        "_run_tier_c_ibps1": _run_tier_c_ibps1,
        "_run_tier_c_dp1": _run_tier_c_dp1,
    }


# ----------------------------------------------------------------------
# Module-level constant + helper API tests
# ----------------------------------------------------------------------


def test_canonical_methods_constant_exposes_three_methods():
    """``TIER_C_NOISE_SCALING_METHODS`` exposes the canonical 3 methods."""
    api = _import_helper()
    methods = api["METHODS"]
    assert isinstance(methods, tuple)
    assert set(methods) == {"gaussian_std", "mad_robust", "iqr_robust"}, (
        f"canonical methods must be exactly {{gaussian_std, mad_robust, "
        f"iqr_robust}}; got {methods}"
    )


def test_default_method_is_gaussian_std_for_back_compat():
    """``DEFAULT_TIER_C_NOISE_SCALING_METHOD`` == 'gaussian_std' (back-compat)."""
    api = _import_helper()
    assert api["DEFAULT"] == "gaussian_std", (
        "default MUST be 'gaussian_std' so all pre-MACKAY-1 Tier C "
        "results remain bit-faithfully reproducible. Changing the "
        "default to a robust method silently invalidates every prior "
        "Tier C density measurement."
    )


def test_compute_noise_scale_unknown_method_raises():
    """Unknown method names raise ``ValueError`` (no silent fallback)."""
    api = _import_helper()
    with pytest.raises(ValueError, match="unknown noise_scaling_method"):
        api["_compute_noise_scale"](torch.randn(64), method="not_a_real_method")


def test_compute_noise_scale_gaussian_std_matches_legacy_formula():
    """``method='gaussian_std'`` returns the same value as ``v.std().clamp(min=1e-8)``."""
    api = _import_helper()
    torch.manual_seed(7)
    v = torch.randn(1024)
    expected = v.std().clamp(min=1e-8).item()
    actual = api["_compute_noise_scale"](v, method="gaussian_std").item()
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8), (
        f"gaussian_std must equal v.std().clamp(min=1e-8); "
        f"expected={expected:.6e} actual={actual:.6e}"
    )


def test_compute_noise_scale_mad_robust_uses_gaussian_equivalent_constant():
    """``method='mad_robust'`` returns ``MAD(v) * 1.4826``."""
    api = _import_helper()
    torch.manual_seed(11)
    v = torch.randn(1024)
    flat = v.detach().reshape(-1).float()
    median = flat.median()
    mad = (flat - median).abs().median()
    expected = (mad * 1.4826).clamp(min=1e-8).item()
    actual = api["_compute_noise_scale"](v, method="mad_robust").item()
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8), (
        f"mad_robust must equal MAD(v) * 1.4826; "
        f"expected={expected:.6e} actual={actual:.6e}"
    )


def test_compute_noise_scale_iqr_robust_uses_gaussian_equivalent_constant():
    """``method='iqr_robust'`` returns ``IQR(v) / 1.349``."""
    api = _import_helper()
    torch.manual_seed(13)
    v = torch.randn(1024)
    flat = v.detach().reshape(-1).float()
    q = torch.quantile(flat, torch.tensor([0.25, 0.75]))
    iqr = q[1] - q[0]
    expected = (iqr / 1.349).clamp(min=1e-8).item()
    actual = api["_compute_noise_scale"](v, method="iqr_robust").item()
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8), (
        f"iqr_robust must equal IQR(v) / 1.349; "
        f"expected={expected:.6e} actual={actual:.6e}"
    )


def test_compute_noise_scale_clamps_zero_spread_tensor():
    """A constant tensor (zero spread) gets clamped to ``floor=1e-8``."""
    api = _import_helper()
    v = torch.zeros(64)
    for method in ("gaussian_std", "mad_robust", "iqr_robust"):
        scale = api["_compute_noise_scale"](v, method=method).item()
        assert scale >= 1e-8 - 1e-12, (
            f"method={method} on zero-spread tensor must return >= floor=1e-8; "
            f"got {scale}"
        )


def test_compute_noise_scale_handles_empty_tensor():
    """An empty tensor returns the floor (no crash on edge case)."""
    api = _import_helper()
    v = torch.empty(0)
    scale = api["_compute_noise_scale"](v).item()
    assert scale >= 1e-8 - 1e-12, (
        f"empty tensor must return >= floor=1e-8; got {scale}"
    )


def test_compute_noise_scale_robust_methods_are_more_robust_to_outliers():
    """MAD/IQR scales are SMALLER than std on a long-tail (outlier) tensor.

    Sanity check the math: if we add a single huge outlier to a Gaussian
    sample, std() balloons but MAD/IQR (which use medians) do not.
    """
    api = _import_helper()
    torch.manual_seed(17)
    base = torch.randn(1024)
    base[0] = 1000.0  # one big outlier
    s_gaussian = api["_compute_noise_scale"](base, method="gaussian_std").item()
    s_mad = api["_compute_noise_scale"](base, method="mad_robust").item()
    s_iqr = api["_compute_noise_scale"](base, method="iqr_robust").item()
    assert s_gaussian > s_mad, (
        f"gaussian_std {s_gaussian:.4e} should be > mad_robust {s_mad:.4e} "
        "on outlier-laden data (the MAD-vs-std difference IS the MACKAY-1 "
        "motivation)"
    )
    assert s_gaussian > s_iqr, (
        f"gaussian_std {s_gaussian:.4e} should be > iqr_robust {s_iqr:.4e} "
        "on outlier-laden data"
    )


# ----------------------------------------------------------------------
# Per-_run_tier_c_* signature tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", [
    "_run_tier_c_a1",
    "_run_tier_c_pr106",
    "_run_tier_c_ibps1",
    "_run_tier_c_dp1",
])
def test_tier_c_func_accepts_noise_scaling_method_kwarg(fn_name):
    """All 4 ``_run_tier_c_*`` funcs accept the ``noise_scaling_method`` kwarg."""
    api = _import_helper()
    sig = inspect.signature(api[fn_name])
    assert "noise_scaling_method" in sig.parameters, (
        f"{fn_name} must accept noise_scaling_method= per MACKAY-1 (R2 MEDIUM, "
        f"2026-05-15); current signature parameters: {list(sig.parameters)}"
    )


@pytest.mark.parametrize("fn_name", [
    "_run_tier_c_a1",
    "_run_tier_c_pr106",
    "_run_tier_c_ibps1",
    "_run_tier_c_dp1",
])
def test_tier_c_func_default_is_gaussian_std(fn_name):
    """All 4 funcs default to ``"gaussian_std"`` for back-compat."""
    api = _import_helper()
    sig = inspect.signature(api[fn_name])
    default = sig.parameters["noise_scaling_method"].default
    assert default == "gaussian_std", (
        f"{fn_name}.noise_scaling_method default MUST be 'gaussian_std' for "
        f"back-compat (pre-MACKAY-1 Tier C result reproducibility); got {default!r}"
    )


# ----------------------------------------------------------------------
# Static gate: callsites in tier_c funcs use _compute_noise_scale
# ----------------------------------------------------------------------


def test_tier_c_funcs_use_compute_noise_scale_helper():
    """All 4 ``_run_tier_c_*`` bodies route noise scaling through the helper.

    Static anti-regression: the helper centralization is the structural
    fix; if a future refactor inlines ``v.std().clamp(min=1e-8)`` again
    inside any tier_c body, this gate fires.
    """
    text = TOOL_PATH.read_text(encoding="utf-8")
    for fn_name in ("_run_tier_c_a1", "_run_tier_c_pr106", "_run_tier_c_ibps1", "_run_tier_c_dp1"):
        body_match = re.search(
            rf"def {fn_name}\([^)]*\)[^:]*:\n(.*?)(?=\n(?:def |class |# ----|@)|\Z)",
            text,
            re.DOTALL,
        )
        assert body_match is not None, f"{fn_name} not found"
        body = body_match.group(1)
        # Count code-only lines that match the legacy pattern.
        code_lines = [
            line for line in body.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        legacy_callsites = code_only.count(".std().clamp(min=1e-8)")
        assert legacy_callsites == 0, (
            f"{fn_name} contains {legacy_callsites} legacy '.std().clamp(min=1e-8)' "
            f"callsites — these MUST route through _compute_noise_scale per "
            f"MACKAY-1 (R2 MEDIUM, 2026-05-15)"
        )
        helper_callsites = code_only.count("_compute_noise_scale(")
        assert helper_callsites >= 2, (
            f"{fn_name} should route ≥2 noise-scaling callsites through "
            f"_compute_noise_scale (state_dict + latents perturbations); "
            f"found {helper_callsites}"
        )


# ----------------------------------------------------------------------
# End-to-end: passing a non-default method produces DIFFERENT noise
# ----------------------------------------------------------------------


def test_tier_c_dp1_method_default_matches_explicit_gaussian_std():
    """Calling without ``noise_scaling_method`` == calling with explicit 'gaussian_std'."""
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random

    from tools.mdl_scorer_conditional_ablation import _run_tier_c_dp1

    try:
        from tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        pytest.skip("Sister DP1 fixture not importable")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2, per_pair_bytes=6)
    gt = torch.zeros(1, 2, 874, 1164, 3, dtype=torch.uint8)

    def _run(method=None):
        kwargs = {} if method is None else {"noise_scaling_method": method}
        return _run_tier_c_dp1(
            inner_bytes=inner,
            pair_indices=[0],
            gt_pairs=gt,
            baseline_seg=0.5,
            baseline_pose=0.1,
            distortion_net=_FakeDistortionNet(),
            device=torch.device("cpu"),
            rng=_random.Random(7),  # same seed → identical sequence
            noise_sigmas=[0.001],
            scorer_batch_size=1,
            **kwargs,
        )

    # Use the same seeded torch RNG so the perturbation Gaussian is identical.
    torch.manual_seed(101)
    rows_default = _run()
    torch.manual_seed(101)
    rows_explicit = _run(method="gaussian_std")
    assert len(rows_default) == len(rows_explicit) == 2
    for a, b in zip(rows_default, rows_explicit):
        assert math.isclose(a.delta_seg, b.delta_seg, abs_tol=1e-6), (
            f"default delta_seg={a.delta_seg} vs explicit gaussian_std "
            f"delta_seg={b.delta_seg} — back-compat broken"
        )
        assert math.isclose(a.delta_pose, b.delta_pose, abs_tol=1e-6), (
            f"default delta_pose={a.delta_pose} vs explicit gaussian_std "
            f"delta_pose={b.delta_pose} — back-compat broken"
        )


def test_tier_c_dp1_mad_robust_method_runs_without_error():
    """End-to-end smoke: passing 'mad_robust' produces well-formed rows."""
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random

    from tools.mdl_scorer_conditional_ablation import (
        _run_tier_c_dp1,
        TierCResult,
    )

    try:
        from tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        pytest.skip("Sister DP1 fixture not importable")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2, per_pair_bytes=6)
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
        noise_scaling_method="mad_robust",
    )
    assert len(rows) == 2  # 1 sigma × 2 targets
    for row in rows:
        assert isinstance(row, TierCResult)
        assert math.isfinite(row.delta_seg)
        assert math.isfinite(row.delta_pose)


def test_tier_c_dp1_iqr_robust_method_runs_without_error():
    """End-to-end smoke: passing 'iqr_robust' produces well-formed rows."""
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random

    from tools.mdl_scorer_conditional_ablation import (
        _run_tier_c_dp1,
        TierCResult,
    )

    try:
        from tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        pytest.skip("Sister DP1 fixture not importable")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2, per_pair_bytes=6)
    rows = _run_tier_c_dp1(
        inner_bytes=inner,
        pair_indices=[0],
        gt_pairs=torch.zeros(1, 2, 874, 1164, 3, dtype=torch.uint8),
        baseline_seg=0.5,
        baseline_pose=0.1,
        distortion_net=_FakeDistortionNet(),
        device=torch.device("cpu"),
        rng=_random.Random(13),
        noise_sigmas=[0.01],
        scorer_batch_size=1,
        noise_scaling_method="iqr_robust",
    )
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, TierCResult)
        assert math.isfinite(row.delta_seg)
        assert math.isfinite(row.delta_pose)


def test_tier_c_dp1_unknown_method_raises():
    """Passing an unknown ``noise_scaling_method`` raises ``ValueError``."""
    pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import random as _random

    from tools.mdl_scorer_conditional_ablation import _run_tier_c_dp1

    try:
        from tac.tests.test_mdl_ablation_tier_c_dp1 import (
            _build_tiny_dp1_archive_bytes,
            _FakeDistortionNet,
        )
    except ImportError:
        pytest.skip("Sister DP1 fixture not importable")

    inner, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2, per_pair_bytes=6)
    with pytest.raises(ValueError, match="unknown noise_scaling_method"):
        _run_tier_c_dp1(
            inner_bytes=inner,
            pair_indices=[0],
            gt_pairs=torch.zeros(1, 2, 874, 1164, 3, dtype=torch.uint8),
            baseline_seg=0.5,
            baseline_pose=0.1,
            distortion_net=_FakeDistortionNet(),
            device=torch.device("cpu"),
            rng=_random.Random(17),
            noise_sigmas=[0.01],
            scorer_batch_size=1,
            noise_scaling_method="not_a_real_method",
        )
