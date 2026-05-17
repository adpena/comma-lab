# SPDX-License-Identifier: MIT
"""Catalog #319 preflight coverage for proof-gated Wyner-Ziv rewards."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import (
    PreflightError,
    check_substrate_wyner_ziv_reweight_has_deliverability_proof,
)


def _write_autopilot(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    path = tools / "cathedral_autopilot_autonomous_loop.py"
    path.write_text(source, encoding="utf-8")
    return root


def test_check_319_refuses_blanket_high_pair_invariant_factor(tmp_path: Path) -> None:
    root = _write_autopilot(
        tmp_path,
        """
def adjust_predicted_delta_for_venn_classification(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR
    return predicted_delta
""",
    )
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=root
    )
    assert violations
    assert any("DeliverabilityProof" in v for v in violations)
    assert any("blanket HIGH_PAIR_INVARIANT" in v for v in violations)


def test_check_319_strict_raises_on_missing_proof_consultation(tmp_path: Path) -> None:
    root = _write_autopilot(
        tmp_path,
        """
def adjust_predicted_delta_for_venn_classification(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR
    return predicted_delta
""",
    )
    with pytest.raises(PreflightError, match="Catalog #319"):
        check_substrate_wyner_ziv_reweight_has_deliverability_proof(
            repo_root=root, strict=True
        )


def test_check_319_accepts_proof_weighted_reward_path(tmp_path: Path) -> None:
    root = _write_autopilot(
        tmp_path,
        """
def _venn_deliverability_reward_factor_for_archive(archive_sha256):
    from tac.wyner_ziv_deliverability.proof_builder import (
        load_deliverability_proof_for_archive,
        verify_deliverability_proof_contest_compliance,
    )
    proof = load_deliverability_proof_for_archive(archive_sha256)
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    return 1.20 if ok else 1.0

def adjust_predicted_delta_for_venn_classification(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _venn_deliverability_reward_factor_for_archive(archive_sha256)
    return predicted_delta
""",
    )
    assert check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=root
    ) == []


def test_check_319_rejects_placeholder_waiver(tmp_path: Path) -> None:
    root = _write_autopilot(
        tmp_path,
        """
def adjust_predicted_delta_for_venn_classification(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta  # VENN_REWEIGHT_DELIVERABILITY_OK:<rationale>
    return predicted_delta
""",
    )
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=root
    )
    assert violations
    assert any("placeholder" in v for v in violations)


def test_check_319_accepts_specific_waiver(tmp_path: Path) -> None:
    root = _write_autopilot(
        tmp_path,
        """
def adjust_predicted_delta_for_venn_classification(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta  # VENN_REWEIGHT_DELIVERABILITY_OK:disabled reward while proof migrates
    return predicted_delta
""",
    )
    assert check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=root
    ) == []


def test_check_319_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight.preflight_all)
    idx = source.find("check_substrate_wyner_ziv_reweight_has_deliverability_proof")
    assert idx >= 0, "Catalog #319 must be wired into preflight_all"
    snippet = source[idx : idx + 220]
    assert "strict=True" in snippet
