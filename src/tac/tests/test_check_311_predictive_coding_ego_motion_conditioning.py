# SPDX-License-Identifier: MIT
"""Tests for Catalog #311 ``check_predictive_coding_substrate_design_has_ego_motion_conditioning``.

Per Z6/Z7/Z8 design memo Pattern H (commit ``aa412d2db``).

The gate refuses substrate design memos invoking Atick-Redlich cooperative-
receiver framing WITHOUT explicit ego-motion-conditioned next-frame
prediction.

Empirical anchor: Z4 cooperative-receiver loss as scaffolded was generic
I(T;Y) maximization without ego-motion conditioning; Z6/Z7/Z8 explicitly
bind cooperative-receiver to ego-motion-conditioned next-frame prediction
via the FOE prior per Ballard's embodied-vision lens.

Sister of Catalog #290 / #294 / #310 (paired G/H/I pattern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_predictive_coding_substrate_design_has_ego_motion_conditioning,
)


def _research_dir(repo_root: Path) -> Path:
    d = repo_root / ".omx" / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(d: Path, name: str, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_311_live_repo_count_bounded() -> None:
    violations = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        strict=False, verbose=False,
    )
    assert len(violations) <= 20, (
        f"Live count {len(violations)} > 20; backfill regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_311_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_memo_without_cooperative_receiver_token_out_of_scope(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "no atick.\n")
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_pre_cutoff_exempt(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "Atick-Redlich cooperative-receiver applied.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_non_design_memo_not_scanned(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "report_20260516.md", "Atick-Redlich cooperative-receiver.\n")
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: cooperative-receiver without ego-motion = flagged
# ---------------------------------------------------------------------------


def test_311_atick_redlich_without_ego_motion_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nAtick-Redlich cooperative-receiver applied to the scorer; generic I(T;Y).\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_311_cooperative_receiver_alone_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "y_design_20260516.md",
        "# Y\n\nCooperative-receiver framework with generic I(T;Y) objective.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_311_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "Atick-Redlich applied.\n")
    _write(research, "b_design_20260516.md", "cooperative-receiver bolt-on.\n")
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


# ---------------------------------------------------------------------------
# Negative: ego-motion-conditioning accepts
# ---------------------------------------------------------------------------


def test_311_ego_motion_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nAtick-Redlich cooperative-receiver via ego-motion-conditioned next-frame predictor.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_foe_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nAtick-Redlich cooperative-receiver bound to FOE (focus-of-expansion) prior.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_next_frame_predictor_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\ncooperative-receiver with next-frame predictor architecture.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_autoregressive_predictor_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\ncooperative-receiver; autoregressive predictor unrolled across pairs.\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_311_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nAtick-Redlich cooperative-receiver applied.\n\n# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:research-only-scoping-memo\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_311_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nAtick-Redlich cooperative-receiver applied.\n\n# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:<rationale>\n",
    )
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_311_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "Atick-Redlich cooperative-receiver applied.\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_predictive_coding_substrate_design_has_ego_motion_conditioning(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #311" in str(exc_info.value)
    assert "Pattern H" in str(exc_info.value)


def test_311_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_predictive_coding_substrate_design_has_ego_motion_conditioning(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
