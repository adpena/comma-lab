# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tac.repo_io import sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "attach_hnerv_rate_recode_packet_links.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("attach_hnerv_rate_recode_packet_links", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_attach_packet_links_updates_only_matching_variant(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    candidate_result_path = tmp_path / "candidate_result.json"
    packet_path = tmp_path / "packet.json"
    archive_manifest_path = tmp_path / "release_surface" / "archive_manifest.json"
    write_json(archive_manifest_path, {"archive": {"sha256": "b" * 64, "bytes": 120}})
    write_json(candidate_result_path, _candidate_result())
    write_json(
        packet_path,
        _packet(
            candidate_result_path=candidate_result_path,
            archive_manifest_path=archive_manifest_path,
        ),
    )

    linked = module.attach_packet_links(profile, packet_path=packet_path)

    assert linked["byte_closed_candidate_packet_count"] == 1
    current = linked["variants"][0]
    matched = linked["variants"][1]
    assert current["variant"] == "brotli_q11_current_raw"
    assert "archive_manifest_path" not in current
    assert matched["variant"] == "brotli_q10_current_raw"
    assert matched["byte_closed_candidate_packet_attached"] is True
    assert matched["archive_manifest_path"] == archive_manifest_path.as_posix()
    assert matched["archive_manifest_sha256"] == sha256_file(archive_manifest_path)
    assert matched["candidate_archive_sha256"] == "b" * 64
    assert matched["candidate_archive_bytes"] == 120
    assert matched["ready_for_exact_eval_dispatch"] is False
    packet_row = linked["byte_closed_candidate_packets"][0]
    assert packet_row["variant"] == "brotli_q10_current_raw"
    assert packet_row["packet_sha256"] == sha256_file(packet_path)
    assert packet_row["candidate_result_sha256"] == sha256_file(candidate_result_path)


def test_attach_packet_links_refuses_source_mismatch(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["source_archive_sha256"] = "0" * 64
    candidate_result_path = tmp_path / "candidate_result.json"
    packet_path = tmp_path / "packet.json"
    archive_manifest_path = tmp_path / "release_surface" / "archive_manifest.json"
    write_json(archive_manifest_path, {"archive": {"sha256": "b" * 64, "bytes": 120}})
    write_json(candidate_result_path, _candidate_result())
    write_json(
        packet_path,
        _packet(
            candidate_result_path=candidate_result_path,
            archive_manifest_path=archive_manifest_path,
        ),
    )

    with pytest.raises(
        module.HnervRateRecodePacketLinkError,
        match="source archive sha256 mismatch",
    ):
        module.attach_packet_links(profile, packet_path=packet_path)


def _profile(root: Path) -> dict:
    return {
        "schema_version": 1,
        "source_label": "fixture-pr106",
        "source_archive_sha256": "a" * 64,
        "variants": [
            {
                "variant": "brotli_q11_current_raw",
                "codec": "brotli",
                "sha256": "1" * 64,
                "byte_delta_vs_source_section": 0,
                "raw_equal": True,
            },
            {
                "variant": "brotli_q10_current_raw",
                "codec": "brotli",
                "sha256": "2" * 64,
                "byte_delta_vs_source_section": -151,
                "raw_equal": True,
            },
        ],
        "fixture_root": root.as_posix(),
    }


def _candidate_result() -> dict:
    return {
        "source_archive_sha256": "a" * 64,
        "candidate_archive_path": "candidate.zip",
        "candidate_archive_sha256": "b" * 64,
        "candidate_archive_bytes": 120,
        "candidate_member_name": "0.bin",
        "attempts": [
            {
                "section_name": "decoder_packed_brotli",
                "accepted_for_candidate": True,
                "candidate_section_sha256": "2" * 64,
                "byte_delta": -151,
            }
        ],
    }


def _packet(*, candidate_result_path: Path, archive_manifest_path: Path) -> dict:
    return {
        "schema": "hnerv_lowlevel_exact_eval_operator_packet_v1",
        "source_archive_sha256": "a" * 64,
        "archive_sha256": "b" * 64,
        "archive_bytes": 120,
        "byte_delta": -151,
        "static_packet_ready": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "submit_blockers": ["missing_operator_exact_cuda_approval"],
        "score_blockers": ["exact_cuda_auth_eval_not_run_for_candidate"],
        "artifacts": {
            "candidate_result": candidate_result_path.as_posix(),
        },
        "release_surface": {
            "files": {
                "archive_manifest.json": {
                    "path": archive_manifest_path.as_posix(),
                    "exists": True,
                    "sha256": sha256_file(archive_manifest_path),
                    "bytes": archive_manifest_path.stat().st_size,
                }
            }
        },
    }
