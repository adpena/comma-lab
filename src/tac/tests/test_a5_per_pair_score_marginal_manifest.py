from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from tac.codec.frame_conditional_bit_budget import pack_frame_conditional_q_bits
from tac.repo_io import json_text, sha256_bytes

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_a5_per_pair_score_marginal_manifest.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_a5_per_pair_score_marginal_manifest", TOOL
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")
    return path


def test_a5_score_marginal_manifest_binds_q_bits_and_pair_scores(tmp_path: Path) -> None:
    tool = _load_tool()
    q_bits = [2, 4, 8]
    sideinfo = pack_frame_conditional_q_bits(q_bits)
    member_payload = b"HEAD" + sideinfo + b"TAIL"
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", member_payload)

    a5 = _write_json(
        tmp_path / "a5.json",
        {
            "schema": "pr101_frame_conditional_bit_anchor.v1",
            "score_claim": False,
            "n_pairs": 3,
        },
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        {
            "score_claim": False,
            "candidate_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": "a" * 64,
            },
            "archive_member_manifest": {
                "member_name": "x",
                "q_bits_sideinfo_offset": 4,
                "q_bits_sideinfo_bytes": len(sideinfo),
            },
        },
    )
    pair_map = _write_json(
        tmp_path / "pair_difficulty.json",
        {
            "pairs_by_difficulty": [
                {
                    "pair_idx": 0,
                    "score": 0.1,
                    "pose_contribution": 0.01,
                    "seg_contribution": 0.09,
                },
                {
                    "pair_idx": 1,
                    "score": 0.3,
                    "pose_contribution": 0.10,
                    "seg_contribution": 0.20,
                },
                {
                    "pair_idx": 2,
                    "score": 0.9,
                    "pose_contribution": 0.30,
                    "seg_contribution": 0.60,
                },
            ]
        },
    )

    payload = tool.build_manifest(
        a5_manifest_path=a5,
        candidate_archive_manifest_path=candidate,
        pair_difficulty_map_path=pair_map,
        repo_root=tmp_path,
    )

    assert payload["score_claim"] is False
    assert payload["marginal_evidence_available"] is True
    assert payload["n_pairs"] == 3
    assert payload["per_pair_q_bits"] == q_bits
    assert payload["q_bits_sideinfo"]["sha256"] == sha256_bytes(sideinfo)
    assert payload["alignment"]["q_bits_vs_score_pearson"] > 0.9
    assert len(payload["per_pair_score_marginals"]) == 3


def test_a5_score_marginal_manifest_rejects_missing_pair_coverage(tmp_path: Path) -> None:
    tool = _load_tool()
    sideinfo = pack_frame_conditional_q_bits([2, 4])
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"X" + sideinfo)
    a5 = _write_json(
        tmp_path / "a5.json",
        {"schema": "pr101_frame_conditional_bit_anchor.v1", "score_claim": False, "n_pairs": 2},
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        {
            "candidate_archive": {"path": str(archive), "bytes": 1, "sha256": "b" * 64},
            "archive_member_manifest": {
                "member_name": "x",
                "q_bits_sideinfo_offset": 1,
                "q_bits_sideinfo_bytes": len(sideinfo),
            },
        },
    )
    pair_map = _write_json(
        tmp_path / "pair_difficulty.json",
        {
            "pairs_by_difficulty": [
                {
                    "pair_idx": 0,
                    "score": 0.1,
                    "pose_contribution": 0.01,
                    "seg_contribution": 0.09,
                }
            ]
        },
    )

    try:
        tool.build_manifest(
            a5_manifest_path=a5,
            candidate_archive_manifest_path=candidate,
            pair_difficulty_map_path=pair_map,
            repo_root=tmp_path,
        )
    except tool.A5ScoreMarginalManifestError as exc:
        assert "cover every pair" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("expected coverage validation failure")
