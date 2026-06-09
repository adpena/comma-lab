# SPDX-License-Identifier: MIT
"""Tests for Catalog #384 — the OBJECTIVE-STARVATION gate.

NO FAKE Class-2 discipline: these tests verify BEHAVIOR (the gate flags a
score-aware claimant with 0.0-default weights; passes a recon-only opt-out;
passes explicit-nonzero weights), not just constants. Replacing the gate body
with `return []` would make ``test_claimant_with_zero_default_is_flagged`` and
``test_live_repo_has_three_known_violations`` FAIL.
"""

from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    check_score_aware_run_has_nonzero_scorer_objective_weights,
)
from tac.substrates._shared.score_aware_objective_weight_audit import (
    audit_score_aware_objective_weights,
    file_has_objective_starvation,
)

# --------------------------------------------------------------------------
# Canonical helper unit (file_has_objective_starvation)
# --------------------------------------------------------------------------


def test_not_a_claimant_is_skipped() -> None:
    # No score-aware token -> no obligation, even with a 0.0-default weight.
    src = (
        '"""A plain recon trainer."""\n'
        "def main(segnet_distillation_weight: float = 0.0): ...\n"
    )
    assert file_has_objective_starvation(src, "x.py") is None


def test_claimant_with_zero_default_is_flagged() -> None:
    # The canonical bug: claims score-aware, SegNet+Pose weights default 0.0.
    src = (
        '"""A score-aware MLX trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 0.0,\n"
        "    pose_distillation_weight: float = 0.0,\n"
        "): ...\n"
    )
    finding = file_has_objective_starvation(src, "train_substrate_x.py")
    assert finding is not None
    assert finding.path == "train_substrate_x.py"
    assert "segnet" in finding.missing_axes
    assert "pose" in finding.missing_axes
    assert "OBJECTIVE-STARVATION" in finding.message()


def test_explicit_nonzero_weights_pass() -> None:
    # Both axes set to explicit nonzero -> OK (genuine score-aware run).
    src = (
        '"""score-aware trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 100.0,\n"
        "    pose_distillation_weight: float = 3.1623,\n"
        "): ...\n"
    )
    assert file_has_objective_starvation(src, "x.py") is None


def test_partial_nonzero_only_one_axis_is_flagged() -> None:
    # SegNet explicit-nonzero but Pose still 0.0 -> pose axis starved.
    src = (
        '"""score_aware trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 100.0,\n"
        "    pose_distillation_weight: float = 0.0,\n"
        "): ...\n"
    )
    finding = file_has_objective_starvation(src, "x.py")
    assert finding is not None
    assert finding.missing_axes == ("pose",)


def test_score_aware_false_opt_out_passes() -> None:
    src = (
        '"""score-aware-capable trainer (running recon-only here)."""\n'
        "score_aware = False\n"
        "def main(segnet_distillation_weight: float = 0.0): ...\n"
    )
    assert file_has_objective_starvation(src, "x.py") is None


def test_scoreaware_false_opt_out_passes() -> None:
    src = (
        '"""scoreaware run."""\n'
        "scoreaware = false\n"  # lowercase false also accepted
        "def main(pose_distillation_weight: float = 0.0): ...\n"
    )
    # 'false' lowercase is not Python but appears in YAML/config; the opt-out
    # regex matches [Ff]alse, so this opts out.
    assert file_has_objective_starvation(src, "x.yaml") is None


def test_research_only_opt_out_passes() -> None:
    src = (
        '"""score-aware research scaffold."""\n'
        "research_only = True\n"
        "def main(segnet_distillation_weight: float = 0.0): ...\n"
    )
    assert file_has_objective_starvation(src, "x.py") is None


def test_dispatch_enabled_false_opt_out_passes() -> None:
    src = (
        "# score-aware recipe\n"
        "dispatch_enabled: false\n"
        "segnet_distillation_weight: 0.0\n"
    )
    assert file_has_objective_starvation(src, "recipe.yaml") is None


def test_same_line_waiver_passes() -> None:
    src = (
        '"""score-aware trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 0.0,  "
        "# SCORE_AWARE_OBJECTIVE_WEIGHTS_OK:intentional recon warmup stage 0\n"
        "    pose_distillation_weight: float = 0.0,\n"
        "): ...\n"
    )
    assert file_has_objective_starvation(src, "x.py") is None


def test_placeholder_waiver_is_rejected() -> None:
    src = (
        '"""score-aware trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 0.0,  "
        "# SCORE_AWARE_OBJECTIVE_WEIGHTS_OK:<rationale>\n"
        "    pose_distillation_weight: float = 0.0,\n"
        "): ...\n"
    )
    # Placeholder rationale cannot self-waive (Catalog #287) -> still flagged.
    finding = file_has_objective_starvation(src, "x.py")
    assert finding is not None


def test_none_default_is_flagged() -> None:
    # A weight defaulting to None (the SNeRV observed_segnet_distillation_weight
    # =None anchor) is also starvation.
    src = (
        '"""score-aware trainer."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float | None = None,\n"
        "    pose_distillation_weight: float | None = None,\n"
        "): ...\n"
    )
    finding = file_has_objective_starvation(src, "x.py")
    assert finding is not None


def test_no_zero_default_weight_means_no_finding() -> None:
    # Claims score-aware but never declares an objective weight at all -> the
    # gate has nothing to attribute a starvation to (no 0.0/None default).
    src = '"""score-aware trainer."""\ndef main(lr: float = 1e-3): ...\n'
    assert file_has_objective_starvation(src, "x.py") is None


# --------------------------------------------------------------------------
# audit_score_aware_objective_weights over a synthetic scan surface
# --------------------------------------------------------------------------


def test_audit_over_explicit_scan_paths(tmp_path) -> None:
    good = tmp_path / "train_substrate_good.py"
    good.write_text(
        '"""score-aware."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 100.0,\n"
        "    pose_distillation_weight: float = 1.0,\n"
        "): ...\n"
    )
    bad = tmp_path / "train_substrate_bad.py"
    bad.write_text(
        '"""score-aware."""\n'
        "def main(\n"
        "    segnet_distillation_weight: float = 0.0,\n"
        "    pose_distillation_weight: float = 0.0,\n"
        "): ...\n"
    )
    findings = audit_score_aware_objective_weights(
        tmp_path, scan_paths=[good, bad]
    )
    assert len(findings) == 1
    assert findings[0].path.endswith("train_substrate_bad.py")


# --------------------------------------------------------------------------
# The preflight gate wrapper
# --------------------------------------------------------------------------


def test_gate_warn_only_returns_list_does_not_raise() -> None:
    v = check_score_aware_run_has_nonzero_scorer_objective_weights(
        strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_gate_strict_raises_on_live_violations() -> None:
    # Live repo currently has the 3 known objective-starvation carriers.
    with pytest.raises(PreflightError, match="OBJECTIVE-STARVATION"):
        check_score_aware_run_has_nonzero_scorer_objective_weights(strict=True)


def test_live_repo_has_three_known_violations() -> None:
    """Catalog #185 sister-callable: the live count is exactly 3 (the SNeRV
    harness + HiNeRV MLX-local + Z7-Mamba-2 MLX-local backfill list)."""
    v = check_score_aware_run_has_nonzero_scorer_objective_weights(
        strict=False, verbose=False
    )
    assert len(v) == 3, f"expected 3 known carriers, got {len(v)}: {v}"
    joined = "\n".join(v)
    assert "snerv_inverse_steg_carrier/mlx_native_train_export.py" in joined
    assert "train_substrate_hi_nerv_mlx_local.py" in joined
    assert "train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py" in joined


def test_gate_verbose_does_not_raise(capsys) -> None:
    check_score_aware_run_has_nonzero_scorer_objective_weights(
        strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "catalog-384" in out


def test_gate_accepts_string_repo_root() -> None:
    # repo_root may be a str; should not raise in WARN-ONLY.
    v = check_score_aware_run_has_nonzero_scorer_objective_weights(
        strict=False, repo_root="."
    )
    assert isinstance(v, list)


def test_gate_registered_in_preflight_all_warn_only() -> None:
    """The gate is registered in preflight_all WARN-ONLY (does not raise)."""
    import inspect

    from tac import preflight

    src = inspect.getsource(preflight.preflight_all)
    assert "check_score_aware_run_has_nonzero_scorer_objective_weights(" in src
    # Confirm the registered callsite is strict=False (WARN-ONLY).
    idx = src.index("check_score_aware_run_has_nonzero_scorer_objective_weights(")
    window = src[idx : idx + 120]
    assert "strict=False" in window
