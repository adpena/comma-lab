from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.repo_io import json_text, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "src" / "tac" / "tests" / "test_pr101_frame_conditional_runtime_packet.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "test_pr101_frame_conditional_runtime_packet_helpers",
        HELPERS_PATH,
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


def test_cli_consumes_score_marginal_qbits_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers = _load_helpers()
    tool = helpers._load_tool()
    helpers._install_tiny_pr101_contract(monkeypatch, tool)
    q_ordered = np.array(
        [
            [255, 128, 17, 1],
            [64, 32, 16, 8],
        ],
        dtype=np.uint8,
    )
    q_bits = np.array([4, 8], dtype=np.uint8)
    latent_blob = helpers._encode_tiny_latent_blob(tool, q_ordered)
    monkeypatch.setattr(tool, "PR101_LATENT_BLOB_LEN", len(latent_blob))
    source_payload = b"DECO" + latent_blob + b"SIDE"
    source_archive = tmp_path / "source" / "archive.zip"
    helpers._write_zip(source_archive, source_payload)
    runtime = helpers._write_runtime(tmp_path / "runtime", latent_blob_len=len(latent_blob))
    a5_manifest = helpers._a5_manifest(
        tmp_path / "a5.json",
        q_bits=q_bits,
        q_ordered=q_ordered,
        source_payload=source_payload,
    )
    q_bits_json = _write_json(
        tmp_path / "score_marginals.json",
        {
            "schema": tool.A5_SCORE_MARGINAL_SCHEMA,
            "score_claim": False,
            "per_pair_q_bits": [4, 8],
        },
    )
    output_dir = tmp_path / "out"

    assert (
        tool.main(
            [
                "--a5-manifest",
                str(a5_manifest),
                "--source-archive",
                str(source_archive),
                "--source-runtime-dir",
                str(runtime),
                "--q-bits-json",
                str(q_bits_json),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    manifest = json.loads(
        (output_dir / "candidate_archive_manifest.json").read_text(encoding="utf-8")
    )
    schedule = manifest["q_bits_schedule"]
    assert schedule["schedule_source"] == "json_override"
    assert schedule["source_schema"] == tool.A5_SCORE_MARGINAL_SCHEMA
    assert schedule["source_key"] == "per_pair_q_bits"
    assert schedule["source_artifact"]["sha256"] == sha256_file(q_bits_json)
    assert schedule["q_bits_count"] == 2
    assert schedule["q_bits_unique_counts"] == {"4": 1, "8": 1}
    assert schedule["score_marginal_manifest_consumed"] is True
    assert "per_pair_score_marginal_manifest_missing" not in manifest["dispatch_blockers"]
    assert "per_pair_score_marginal_manifest_consumed" in manifest[
        "cleared_local_readiness_artifacts"
    ]
    with zipfile.ZipFile(REPO_ROOT / manifest["candidate_archive_relpath"]) as zf:
        assert zf.namelist() == ["x"]


def test_qbits_override_json_rejects_ambiguous_keys(tmp_path: Path) -> None:
    helpers = _load_helpers()
    tool = helpers._load_tool()
    q_bits_json = _write_json(
        tmp_path / "ambiguous.json",
        {"per_pair_q_bits": [4, 8], "q_bits": [4, 8]},
    )

    with pytest.raises(tool.FrameConditionalRuntimePacketError, match="exactly one"):
        tool._load_q_bits_override_json(q_bits_json)


def test_changed_qbits_require_explicit_wire_contract_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers = _load_helpers()
    tool = helpers._load_tool()
    helpers._install_tiny_pr101_contract(monkeypatch, tool)
    q_ordered = np.array(
        [
            [255, 128, 17, 1],
            [64, 32, 16, 8],
        ],
        dtype=np.uint8,
    )
    original_q_bits = np.array([4, 8], dtype=np.uint8)
    changed_q_bits = [8, 8]
    latent_blob = helpers._encode_tiny_latent_blob(tool, q_ordered)
    monkeypatch.setattr(tool, "PR101_LATENT_BLOB_LEN", len(latent_blob))
    source_payload = b"DECO" + latent_blob + b"SIDE"
    source_archive = tmp_path / "source" / "archive.zip"
    helpers._write_zip(source_archive, source_payload)
    runtime = helpers._write_runtime(tmp_path / "runtime", latent_blob_len=len(latent_blob))
    a5_manifest = helpers._a5_manifest(
        tmp_path / "a5.json",
        q_bits=original_q_bits,
        q_ordered=q_ordered,
        source_payload=source_payload,
    )
    q_bits_json = _write_json(
        tmp_path / "changed_score_marginals.json",
        {
            "schema": tool.A5_SCORE_MARGINAL_SCHEMA,
            "score_claim": False,
            "per_pair_q_bits": changed_q_bits,
        },
    )

    with pytest.raises(SystemExit, match="latent_wire_payload"):
        tool.main(
            [
                "--a5-manifest",
                str(a5_manifest),
                "--source-archive",
                str(source_archive),
                "--source-runtime-dir",
                str(runtime),
                "--q-bits-json",
                str(q_bits_json),
                "--output-dir",
                str(tmp_path / "blocked"),
            ]
        )

    assert (
        tool.main(
            [
                "--a5-manifest",
                str(a5_manifest),
                "--source-archive",
                str(source_archive),
                "--source-runtime-dir",
                str(runtime),
                "--q-bits-json",
                str(q_bits_json),
                "--recompute-wire-contract-for-q-bits",
                "--output-dir",
                str(tmp_path / "allowed"),
            ]
        )
        == 0
    )
    manifest = json.loads(
        (tmp_path / "allowed" / "candidate_archive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reconciliation = manifest["wire_contract_reconciliation"]
    assert reconciliation["selected_source"] == "materialized_q_bits_override"
    assert reconciliation["q_bits_wire_contract_override_allowed"] is True
    assert reconciliation["source_wire_contract_matches_materialized"] is False
    assert reconciliation["materialized_wire_contract_matches_materialized"] is True
    assert manifest["q_bits_schedule"]["q_bits_unique_counts"] == {"8": 2}
    assert manifest["q_bits_schedule"]["score_marginal_manifest_consumed"] is True
