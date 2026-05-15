# SPDX-License-Identifier: MIT
"""Tests for Catalog #273-#278 STRICT preflight self-protection gates."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_preflight_compressive_landscape_canonical_use,
    check_preflight_falling_rule_list_canonical_use,
    check_preflight_gosdt_dispatcher_whiteboard_discipline,
    check_preflight_rashomon_ensemble_continual_update_locked,
    check_preflight_slim_risk_scorer_canonical_use,
    check_preflight_wavelet_multi_scale_contract,
)

# ── Live-repo regression guards ──────────────────────────────────────────


def test_check_273_live_count_zero():
    assert check_preflight_slim_risk_scorer_canonical_use() == []


def test_check_274_live_count_zero():
    assert check_preflight_falling_rule_list_canonical_use() == []


def test_check_275_live_count_zero():
    assert check_preflight_rashomon_ensemble_continual_update_locked() == []


def test_check_276_live_count_zero():
    assert check_preflight_compressive_landscape_canonical_use() == []


def test_check_277_live_count_zero():
    assert check_preflight_wavelet_multi_scale_contract() == []


def test_check_278_live_count_zero():
    assert check_preflight_gosdt_dispatcher_whiteboard_discipline() == []


# ── Synthetic positive tests (forbidden token in tmp file under src/tac/) ──


def _make_tmp_repo_with_violation(tmp_path: Path, token: str) -> Path:
    """Build a fake repo skeleton with a violation file under src/tac/."""
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "violator.py").write_text(
        f"# SPDX-License-Identifier: MIT\nx = '{token}'\n", encoding="utf-8"
    )
    return tmp_path


def test_check_273_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_float_coef")
    violations = check_preflight_slim_risk_scorer_canonical_use(repo_root=repo)
    assert any("preflight_float_coef" in v for v in violations)


def test_check_273_strict_raises_synthetic(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_float_coef")
    with pytest.raises(PreflightError):
        check_preflight_slim_risk_scorer_canonical_use(
            repo_root=repo, strict=True
        )


def test_check_273_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_float_coef'  # PREFLIGHT_SLIM_FLOAT_COEF_OK:operator-reviewed\n",
        encoding="utf-8",
    )
    violations = check_preflight_slim_risk_scorer_canonical_use(repo_root=tmp_path)
    assert violations == []


def test_check_273_placeholder_waiver_rejected(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_float_coef'  # PREFLIGHT_SLIM_FLOAT_COEF_OK:<rationale>\n",
        encoding="utf-8",
    )
    violations = check_preflight_slim_risk_scorer_canonical_use(repo_root=tmp_path)
    assert any("preflight_float_coef" in v for v in violations)


def test_check_274_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_ascending_rule_list")
    violations = check_preflight_falling_rule_list_canonical_use(repo_root=repo)
    assert any("preflight_ascending_rule_list" in v for v in violations)


def test_check_274_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_ascending_rule_list'  # PREFLIGHT_NON_FALLING_RULE_OK:legacy-test-fixture\n",
        encoding="utf-8",
    )
    assert check_preflight_falling_rule_list_canonical_use(repo_root=tmp_path) == []


def test_check_275_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "PreflightRashomonEnsemble_NON_PERSISTED")
    violations = check_preflight_rashomon_ensemble_continual_update_locked(repo_root=repo)
    assert any("PreflightRashomonEnsemble_NON_PERSISTED" in v for v in violations)


def test_check_275_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'PreflightRashomonEnsemble_NON_PERSISTED'  # PREFLIGHT_RASHOMON_NON_PERSISTED_OK:in-memory-only\n",
        encoding="utf-8",
    )
    assert check_preflight_rashomon_ensemble_continual_update_locked(repo_root=tmp_path) == []


def test_check_276_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_dense_anchor_reconstruction")
    violations = check_preflight_compressive_landscape_canonical_use(repo_root=repo)
    assert any("preflight_dense_anchor_reconstruction" in v for v in violations)


def test_check_276_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_dense_anchor_reconstruction'  # PREFLIGHT_DENSE_ANCHOR_OK:diagnostic\n",
        encoding="utf-8",
    )
    assert check_preflight_compressive_landscape_canonical_use(repo_root=tmp_path) == []


def test_check_277_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_single_scale_for_multi_scale_problem")
    violations = check_preflight_wavelet_multi_scale_contract(repo_root=repo)
    assert any("preflight_single_scale_for_multi_scale_problem" in v for v in violations)


def test_check_277_strict_raises_synthetic(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_fine_overrides_coarse_gate")
    with pytest.raises(PreflightError):
        check_preflight_wavelet_multi_scale_contract(repo_root=repo, strict=True)


def test_check_277_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_skip_coarsest_scale_evaluation'  # PREFLIGHT_SINGLE_SCALE_OK:diagnostic-mode\n",
        encoding="utf-8",
    )
    assert check_preflight_wavelet_multi_scale_contract(repo_root=tmp_path) == []


def test_check_278_flags_synthetic_violation(tmp_path):
    repo = _make_tmp_repo_with_violation(tmp_path, "preflight_auto_promote_whiteboard_rule")
    violations = check_preflight_gosdt_dispatcher_whiteboard_discipline(repo_root=repo)
    assert any("preflight_auto_promote_whiteboard_rule" in v for v in violations)


def test_check_278_strict_raises_synthetic(tmp_path):
    repo = _make_tmp_repo_with_violation(
        tmp_path, "preflight_promote_whiteboard_rule_without_operator_review"
    )
    with pytest.raises(PreflightError):
        check_preflight_gosdt_dispatcher_whiteboard_discipline(
            repo_root=repo, strict=True
        )


def test_check_278_same_line_waiver_accepted(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_auto_promote_whiteboard_rule'  # PREFLIGHT_AUTO_PROMOTE_OK:operator-pre-approved-config\n",
        encoding="utf-8",
    )
    assert check_preflight_gosdt_dispatcher_whiteboard_discipline(repo_root=tmp_path) == []


def test_check_278_placeholder_waiver_rejected(tmp_path):
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "ok.py").write_text(
        "x = 'preflight_auto_promote_whiteboard_rule'  # PREFLIGHT_AUTO_PROMOTE_OK:<rationale>\n",
        encoding="utf-8",
    )
    violations = check_preflight_gosdt_dispatcher_whiteboard_discipline(repo_root=tmp_path)
    assert any("preflight_auto_promote_whiteboard_rule" in v for v in violations)


# ── Self-exempt path tests ─────────────────────────────────────────────────


def test_self_exempt_paths_skipped(tmp_path):
    """The canonical helper package itself is skipped from the scan."""
    src = tmp_path / "src" / "tac" / "preflight_rudin_daubechies"
    src.mkdir(parents=True)
    (src / "self_exempt.py").write_text(
        "x = 'preflight_float_coef'\n", encoding="utf-8"
    )
    # The exempt-path marker matches "src/tac/preflight_rudin_daubechies/"
    violations = check_preflight_slim_risk_scorer_canonical_use(repo_root=tmp_path)
    assert violations == []


def test_test_files_self_exempt(tmp_path):
    """test_*.py files are self-exempt (they reference forbidden tokens in fixtures)."""
    src = tmp_path / "src" / "tac"
    src.mkdir(parents=True)
    (src / "test_my_thing.py").write_text(
        "x = 'preflight_float_coef'\n", encoding="utf-8"
    )
    assert check_preflight_slim_risk_scorer_canonical_use(repo_root=tmp_path) == []


# ── Wire-in regression guards ──────────────────────────────────────────────


def test_orchestrator_wires_273_strict_true():
    """Catalog #273 wired into preflight_all() with strict=True (Catalog #176)."""
    pf_path = Path(__file__).resolve().parents[2] / "preflight.py"
    text = pf_path.read_text(encoding="utf-8")
    assert "check_preflight_slim_risk_scorer_canonical_use(\n            strict=True" in text


def test_orchestrator_wires_278_strict_true():
    """Catalog #278 wired into preflight_all() with strict=True (Catalog #176)."""
    pf_path = Path(__file__).resolve().parents[2] / "preflight.py"
    text = pf_path.read_text(encoding="utf-8")
    assert "check_preflight_gosdt_dispatcher_whiteboard_discipline(\n            strict=True" in text


def test_all_six_gates_wired_into_preflight_all():
    """All 6 Catalog #273-#278 gates wired with strict=True (Catalog #176)."""
    pf_path = Path(__file__).resolve().parents[2] / "preflight.py"
    text = pf_path.read_text(encoding="utf-8")
    for name in (
        "check_preflight_slim_risk_scorer_canonical_use",
        "check_preflight_falling_rule_list_canonical_use",
        "check_preflight_rashomon_ensemble_continual_update_locked",
        "check_preflight_compressive_landscape_canonical_use",
        "check_preflight_wavelet_multi_scale_contract",
        "check_preflight_gosdt_dispatcher_whiteboard_discipline",
    ):
        assert f"{name}(\n            strict=True" in text, f"{name} not wired strict=True"
