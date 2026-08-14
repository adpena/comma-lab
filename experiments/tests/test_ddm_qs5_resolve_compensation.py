from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs5_resolve_compensation as qs5
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def test_changed_compile_object_refuses_stale_compensation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = tmp_path / "bank"
    monkeypatch.setattr(qs1, "JS6_BANK", bank)
    proposal_id = "proposal"
    canonical = bank / "proposals" / proposal_id / "candidate_tokens.uint8.npy"
    changed = tmp_path / "changed.uint8.npy"
    _save(canonical, np.zeros((2, 2), dtype=np.uint8))
    _save(changed, np.ones((2, 2), dtype=np.uint8))
    row = {
        "proposal_id": proposal_id,
        "pair": 7,
        "candidate_tokens_path": str(changed),
        "solve": {"final_codes": [0] * 12},
    }
    with pytest.raises(qs1.QS1Error, match="lacks an exact-object compensation solve"):
        qs1.assert_compensation_matches_compile_object(row)


def test_changed_compile_object_accepts_matching_fresh_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = tmp_path / "bank"
    monkeypatch.setattr(qs1, "JS6_BANK", bank)
    proposal_id = "proposal"
    canonical = bank / "proposals" / proposal_id / "candidate_tokens.uint8.npy"
    changed = tmp_path / "changed.uint8.npy"
    master = tmp_path / "master.uint8.npy"
    _save(canonical, np.zeros((2, 2), dtype=np.uint8))
    _save(changed, np.ones((2, 2), dtype=np.uint8))
    _save(master, np.full((2, 2, 3), 127, dtype=np.uint8))
    token_record = qs1.file_record(changed)
    master_record = qs1.file_record(master)
    fingerprint = qs1.compensation_object_fingerprint(
        pair=7,
        semantic_tokens=token_record,
        master_camera=master_record,
    )
    row = {
        "proposal_id": proposal_id,
        "pair": 7,
        "candidate_tokens_path": str(changed),
        "compensation_object": {
            "schema": "ddm_qs1_compensation_object_binding.v1",
            "pair": 7,
            "semantic_tokens": token_record,
            "master_camera": master_record,
            "fingerprint_sha256": fingerprint,
            "exact_master_rendered_from_semantic_tokens": True,
        },
        "solve": {
            "final_codes": [0] * 12,
            "compensation_object_fingerprint_sha256": fingerprint,
        },
    }
    result = qs1.assert_compensation_matches_compile_object(row)
    assert result["passed"] is True
    assert result["object_changed_from_canonical_proposal"] is True
    assert result["mode"] == "EXACT_OBJECT_BOUND_FRESH_SOLVE"


def test_neutral_detrim_restores_connectors_but_excludes_negative_sites() -> None:
    sites, counts = qs5.restore_neutral_connective_support(
        [
            {
                "site_flat": 8,
                "B": 3,
                "H": 0,
                "W": 0,
                "strict_support_keep": True,
            },
            {
                "site_flat": 2,
                "B": 0,
                "H": 0,
                "W": 0,
                "strict_support_keep": False,
            },
            {
                "site_flat": 5,
                "B": 1,
                "H": 1,
                "W": 0,
                "strict_support_keep": False,
            },
            {
                "site_flat": 9,
                "B": 1,
                "H": 2,
                "W": 0,
                "strict_support_keep": False,
            },
        ]
    )
    assert sites.tolist() == [2, 5, 8]
    assert counts == {
        "strict_sites": 1,
        "neutral_restored_sites": 2,
        "negative_sites_excluded": 1,
        "model_B": 4,
        "model_H": 1,
        "model_W": 0,
    }


def test_qs5_runner_does_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=qs5.REPO,
        roots=[
            Path("experiments/ddm_qs1_frame0_schur_coupled_solve.py"),
            Path("experiments/ddm_qs5_resolve_compensation.py"),
        ],
    )
    assert findings == []
