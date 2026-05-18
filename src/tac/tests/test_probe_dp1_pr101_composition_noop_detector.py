# SPDX-License-Identifier: MIT
"""Tests for the DP1+PR101 composition no-op detector."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from probe_dp1_pr101_composition_noop_detector import (  # noqa: E402
    CONTEST_ARCHIVE_NORMALIZER_BYTES,
    build_probe_payload,
)
from tac.substrates.pretrained_driving_prior.composition import (  # noqa: E402
    DPCOMP_HEADER_SIZE,
    DPCOMP_MAGIC,
    DPCOMP_SCHEMA_VERSION,
)


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _write_packet(tmp_path: Path, *, guard: bool = True) -> Path:
    dp1_bytes = b"dp1-prior-bytes"
    base_bytes = b"fec6-base-archive-bytes"
    archive_bytes = (
        struct.pack(
            "<4sBI4s",
            DPCOMP_MAGIC,
            DPCOMP_SCHEMA_VERSION,
            len(dp1_bytes),
            b"PR01",
        )
        + dp1_bytes
        + base_bytes
    )

    packet_dir = tmp_path / "experiments/results/dp1_plus_fec6_composition_test"
    packet_dir.mkdir(parents=True)
    dp1_path = tmp_path / "experiments/results/dp1/0.bin"
    fec6_path = tmp_path / "experiments/results/fec6/archive.zip"
    dp1_path.parent.mkdir(parents=True)
    fec6_path.parent.mkdir(parents=True)
    dp1_path.write_bytes(dp1_bytes)
    fec6_path.write_bytes(base_bytes)
    (packet_dir / "archive.zip").write_bytes(archive_bytes)
    (packet_dir / "archive_manifest.json").write_text(
        json.dumps(
            {
                "archive_relpath": "archive.zip",
                "archive_sha256": _sha(archive_bytes),
                "archive_size_bytes": len(archive_bytes),
                "base_substrate": "pr101",
                "composition_schema_version": DPCOMP_SCHEMA_VERSION,
                "dp1_source_sha256": _sha(dp1_bytes),
                "dp1_source_size_bytes": len(dp1_bytes),
                "fec6_source_sha256": _sha(base_bytes),
                "fec6_source_size_bytes": len(base_bytes),
                "header_size_bytes": DPCOMP_HEADER_SIZE,
            }
        ),
        encoding="utf-8",
    )
    (packet_dir / "build_manifest.json").write_text(
        json.dumps(
            {
                "archive_relpath": "archive.zip",
                "archive_sha256": _sha(archive_bytes),
                "archive_size_bytes": len(archive_bytes),
                "lane_id": "lane_dp1_plus_fec6_dual_stacking_build_20260517",
                "operational_mechanism_status": "OPERATIONAL_DEFERRED_TO_L2",
                "dp1_source": {
                    "path": str(dp1_path.relative_to(tmp_path)),
                    "sha256": _sha(dp1_bytes),
                    "size_bytes": len(dp1_bytes),
                },
                "fec6_source": {
                    "path": str(fec6_path.relative_to(tmp_path)),
                    "sha256": _sha(base_bytes),
                    "size_bytes": len(base_bytes),
                },
            }
        ),
        encoding="utf-8",
    )
    if guard:
        inflate_text = (
            "strength = float(os.environ.get('PACT_DP1_PRIOR_STRENGTH', '0.0'))\n"
            "if strength > 0.0:\n"
            "    raise RuntimeError('requires L2 INTEGRATION')\n"
        )
    else:
        inflate_text = "def main():\n    return 0\n"
    (packet_dir / "inflate.py").write_text(inflate_text, encoding="utf-8")
    return packet_dir


def test_build_probe_payload_verifies_l1_rate_only_noop_packet(tmp_path: Path) -> None:
    packet_dir = _write_packet(tmp_path)

    payload = build_probe_payload(
        packet_dir,
        repo_root=tmp_path,
        created_utc="2026-05-18T06:30:00Z",
    )

    assert payload["schema"] == "dp1_pr101_composition_noop_detector_v1"
    assert payload["verdict"] == "l1_rate_only_noop_verified"
    assert payload["structural_pass"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["structural_blockers"] == []
    assert "not_score_authority" in payload["blockers"]
    assert payload["build_manifest_lane_id"] == (
        "lane_dp1_plus_fec6_dual_stacking_build_20260517"
    )
    assert payload["archive_components"]["base_substrate"] == "pr101"
    assert payload["archive_components"]["header_size_bytes"] == DPCOMP_HEADER_SIZE

    expected_delta = (
        25.0
        * (
            payload["archive_components"]["archive_size_bytes"]
            - payload["archive_components"]["base_archive_size_bytes"]
        )
        / CONTEST_ARCHIVE_NORMALIZER_BYTES
    )
    assert payload["rate_axis_delta_if_frames_identical"] == pytest.approx(
        expected_delta
    )


def test_build_probe_payload_blocks_manifest_source_mismatch(tmp_path: Path) -> None:
    packet_dir = _write_packet(tmp_path)
    manifest_path = packet_dir / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fec6_source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = build_probe_payload(packet_dir, repo_root=tmp_path)

    assert payload["verdict"] == "blocked_structural_mismatch"
    assert payload["structural_pass"] is False
    assert "archive_manifest_fec6_source_sha256_mismatch" in payload[
        "structural_blockers"
    ]
    assert payload["score_claim"] is False


def test_build_probe_payload_blocks_wrong_build_manifest_lane(tmp_path: Path) -> None:
    packet_dir = _write_packet(tmp_path)
    manifest_path = packet_dir / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lane_id"] = "lane_unrelated_pr101_packet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = build_probe_payload(packet_dir, repo_root=tmp_path)

    assert payload["verdict"] == "blocked_structural_mismatch"
    assert payload["structural_pass"] is False
    assert payload["build_manifest_lane_id"] == "lane_unrelated_pr101_packet"
    assert "build_manifest_lane_id_mismatch" in payload["structural_blockers"]
    assert payload["score_claim"] is False


def test_build_probe_payload_blocks_missing_l2_strength_guard(tmp_path: Path) -> None:
    packet_dir = _write_packet(tmp_path, guard=False)

    payload = build_probe_payload(packet_dir, repo_root=tmp_path)

    assert payload["verdict"] == "blocked_structural_mismatch"
    assert payload["structural_pass"] is False
    assert "inflate_py_dp1_strength_env_missing" in payload["structural_blockers"]
    assert "inflate_py_l2_strength_guard_missing" in payload["structural_blockers"]


def test_dp1_paired_recipe_uses_correct_rate_axis_delta() -> None:
    recipe = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_paired_dispatch.yaml"
    )
    text = recipe.read_text(encoding="utf-8")

    assert "+0.0000172 contest rate term" not in text
    assert "earlier +0.0000172 arithmetic was off by 1000x" in text
    assert "0.19207 CPU" not in text
    assert "+0.017197139182" in text
    assert "dispatch_enabled: false" in text
    assert "score_claim: false" in text
    assert "dp1_pr101_composition_noop_probe_20260518_codex.json" in text
