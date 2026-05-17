# SPDX-License-Identifier: MIT
"""Regression coverage for Venn reweighting + predicted dispatch risk composition."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tac.wyner_ziv_deliverability import DeliverabilityProof


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_autopilot_module():
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop_venn_risk",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop_venn_risk", mod)
    spec.loader.exec_module(mod)
    return mod


def _write_venn_sidecar(root: Path, sha: str, *, pair_invariant: int = 9000) -> None:
    root.mkdir(parents=True)
    sidecar = root / f"venn_classification_{sha[:12]}_20260517T210000.json"
    sidecar.write_text(
        json.dumps(
            {
                "class_counts": {
                    "PAIR_SPECIFIC": 100,
                    "PAIR_INVARIANT": pair_invariant,
                    "PAIR_NEUTRAL": 900,
                    "DEAD": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_deliverability_proof(root: Path, proof: DeliverabilityProof) -> None:
    root.mkdir(parents=True)
    sidecar = root / f"proof_{proof.archive_sha256[:12]}_20260517T210000Z.json"
    sidecar.write_text(json.dumps(proof.as_dict()), encoding="utf-8")


def test_venn_reweight_does_not_replace_predicted_dispatch_risk_refusal(
    tmp_path, monkeypatch
) -> None:
    """HIGH PAIR_INVARIANT Venn reward must not resurrect a risk-refused row."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "66" * 32
    _write_venn_sidecar(fake_root, sha)
    candidate = mod.CandidateRow(
        candidate_id="risk_refused_venn_rewarded",
        family="fec6_like",
        predicted_score_delta=-0.012,
        expected_information_gain=1.0,
        estimated_dispatch_cost_usd=1.0,
        archive_sha256=sha,
        predicted_dispatch_risk=75.0,
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(candidate)
    assert rank_key == mod.PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR
    assert rank_key == 0.0


def test_high_pair_invariant_without_deliverability_proof_gets_no_reward(
    tmp_path, monkeypatch
) -> None:
    """Venn HIGH_PAIR_INVARIANT alone is not enough to improve the rank key."""
    mod = _load_autopilot_module()
    fake_venn_root = tmp_path / "master_gradient_consumers"
    fake_proof_root = tmp_path / "wyner_ziv_deliverability"
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_venn_root)
    monkeypatch.setattr(mod, "_WYNER_ZIV_DELIVERABILITY_PROOF_ROOT", fake_proof_root)
    sha = "77" * 32
    _write_venn_sidecar(fake_venn_root, sha)

    assert mod.adjust_predicted_delta_for_venn_classification(-0.012, sha) == -0.012


def test_high_pair_invariant_uses_deliverability_proof_weighted_reward(
    tmp_path, monkeypatch
) -> None:
    """Compliant proof applies byte-weighted Tier 1/Tier 2 reward factors."""
    mod = _load_autopilot_module()
    fake_venn_root = tmp_path / "master_gradient_consumers"
    fake_proof_root = tmp_path / "wyner_ziv_deliverability"
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_venn_root)
    monkeypatch.setattr(mod, "_WYNER_ZIV_DELIVERABILITY_PROOF_ROOT", fake_proof_root)
    sha = "88" * 32
    _write_venn_sidecar(fake_venn_root, sha)
    _write_deliverability_proof(
        fake_proof_root,
        DeliverabilityProof(
            archive_sha256=sha,
            candidate_shared_prior_byte_count=10,
            tier_1_byte_count=5,
            tier_2_byte_count=5,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            tier_1_byte_indices=(0, 1, 2, 3, 4),
            tier_2_byte_indices=(5, 6, 7, 8, 9),
            contest_compliance_verdict="compliant",
            canonical_helper_invocation=(
                "tac.wyner_ziv_deliverability (zero-cost only; "
                "Comma2k19LocalCache.fetch_chunk unavailable)"
            ),
            inflate_py_loc_estimate=40,
        ),
    )
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    expected_factor = (5 * 1.20 + 5 * 1.10) / 10
    assert adjusted == pytest.approx(raw * expected_factor)


def test_high_pair_invariant_noncompliant_proof_gets_no_reward(
    tmp_path, monkeypatch
) -> None:
    """Tier 4/scorer-dependent proof blocks the positive Venn reward."""
    mod = _load_autopilot_module()
    fake_venn_root = tmp_path / "master_gradient_consumers"
    fake_proof_root = tmp_path / "wyner_ziv_deliverability"
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_venn_root)
    monkeypatch.setattr(mod, "_WYNER_ZIV_DELIVERABILITY_PROOF_ROOT", fake_proof_root)
    sha = "99" * 32
    _write_venn_sidecar(fake_venn_root, sha)
    _write_deliverability_proof(
        fake_proof_root,
        DeliverabilityProof(
            archive_sha256=sha,
            candidate_shared_prior_byte_count=4,
            tier_1_byte_count=0,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=4,
            tier_4_byte_indices=(0, 1, 2, 3),
            contest_compliance_verdict="non_compliant",
            canonical_helper_invocation="unknown",
        ),
    )
    raw = -0.012
    assert mod.adjust_predicted_delta_for_venn_classification(raw, sha) == raw
