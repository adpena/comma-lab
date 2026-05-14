# SPDX-License-Identifier: MIT
"""Tests for the Alpha INR/TinyNeRV readiness scaffold."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_inr_readiness.py"
SPEC = importlib.util.spec_from_file_location("alpha_inr_readiness", MODULE_PATH)
assert SPEC is not None
readiness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = readiness
SPEC.loader.exec_module(readiness)


def _tiny_masks() -> np.ndarray:
    return np.array(
        [
            [
                [0, 0, 1, 1, 4],
                [0, 2, 2, 1, 4],
                [3, 3, 2, 4, 4],
            ],
            [
                [0, 1, 1, 1, 4],
                [0, 2, 2, 2, 4],
                [3, 3, 2, 4, 0],
            ],
        ],
        dtype=np.uint8,
    )


def _write_archive(
    path: Path,
    *,
    mask_bytes: bytes = b"mask-member-payload",
    unsafe_member: str | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer-bytes")
        zf.writestr("masks.mkv", mask_bytes)
        zf.writestr("optimized_poses.bin", b"pose-bytes")
        if unsafe_member is not None:
            zf.writestr(unsafe_member, b"unsafe")
    return path


def _write_decoded_masks(path: Path, masks: np.ndarray) -> Path:
    np.save(path, masks)
    return path


def _manifest_for(
    path: Path,
    *,
    archive: Path,
    masks: np.ndarray,
    bad_decoded_sha: bool = False,
) -> Path:
    archive_meta = readiness._audit_archive(
        archive,
        required_members=readiness.REQUIRED_BASELINE_MEMBERS,
    )
    decoded_sha = readiness._sha256_bytes(np.ascontiguousarray(masks).tobytes())
    payload: dict[str, Any] = {
        "schema": "alpha_mask_candidate_builder_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "local_builder_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": readiness.CUDA_AUTH_EVAL_PATH,
        "source": {
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": readiness._sha256_file(archive),
            "mask_member": dict(archive_meta["members"]["masks.mkv"]),
            "decoded_masks": {
                "shape": [int(v) for v in masks.shape],
                "class_id_u8_sha256": "0" * 64 if bad_decoded_sha else decoded_sha,
            },
        },
        "candidate": {
            "score_claim": False,
            "promotion_eligible": False,
            "candidate_archive_readiness": {
                "artifacts_complete_for_selected_frames": True,
                "full_sequence_candidate": True,
                "residual_repair_full_coverage": True,
                "exact_eval_archive_builder_required": True,
                "exact_cuda_auth_eval_required": readiness.CUDA_AUTH_EVAL_PATH,
                "ready_for_exact_eval_finalist_archive_assembly": True,
            },
            "artifacts": [],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_alpha_inr_readiness_validates_custody_identity_and_byte_shape_accounting(tmp_path: Path) -> None:
    masks = _tiny_masks()
    archive = _write_archive(tmp_path / "archive.zip")
    decoded = _write_decoded_masks(tmp_path / "decoded_masks.npy", masks)
    manifest = _manifest_for(tmp_path / "alpha_mask_candidate_manifest.json", archive=archive, masks=masks)
    output = tmp_path / "alpha_inr_readiness.json"
    command = ["alpha_inr_readiness.py", "--unit-test"]

    first = readiness.audit_alpha_inr_readiness(
        baseline_archive=archive,
        decoded_masks_source=decoded,
        candidate_manifest=manifest,
        output_json=output,
        expected_frames=2,
        expected_height=3,
        expected_width=5,
        num_freqs=2,
        hidden_dim=8,
        depth=3,
        command=command,
    )
    first_text = output.read_text()
    second = readiness.audit_alpha_inr_readiness(
        baseline_archive=archive,
        decoded_masks_source=decoded,
        candidate_manifest=manifest,
        output_json=output,
        expected_frames=2,
        expected_height=3,
        expected_width=5,
        num_freqs=2,
        hidden_dim=8,
        depth=3,
        command=command,
        force=True,
    )

    assert json.loads(first_text) == first
    assert output.read_text() == first_text
    assert second == first
    assert first["schema"] == "alpha_inr_readiness_v1"
    assert first["score_claim"] is False
    assert first["score_claim_eligible"] is False
    assert first["promotion_eligible"] is False
    assert first["evidence_grade"] == "empirical"
    assert first["training_performed"] is False
    assert first["remote_job_launched"] is False
    assert first["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in first["canonical_score_source_required"]

    archive_record = first["inputs"]["baseline_archive"]
    assert archive_record["sha256"] == readiness._sha256_file(archive)
    assert archive_record["size_bytes"] == archive.stat().st_size
    assert archive_record["validated_zip_safety"] is True
    assert archive_record["members"]["masks.mkv"]["sha256"] == readiness._sha256_bytes(b"mask-member-payload")

    identity = first["decoded_mask_identity"]
    assert identity["shape"] == [2, 3, 5]
    assert identity["class_id_u8_sha256"] == readiness._sha256_bytes(masks.tobytes())
    assert identity["source_identity_validated_against_candidate_manifest"] is True

    accounting = first["byte_accounting"]
    assert accounting["decoded_mask_raw_u8_bytes"] == 2 * 3 * 5
    assert accounting["raw_one_hot_fp16_logit_bytes"] == 2 * 3 * 5 * 5 * 2
    payloads = {item["weight_dtype"]: item for item in accounting["prototype_payloads"]}
    assert payloads["fp16"]["encoded_payload_bytes"] > payloads["int8"]["encoded_payload_bytes"] > 0
    assert all(item["untrained_prototype_only"] is True for item in accounting["prototype_payloads"])

    gates = first["readiness_gates"]
    assert gates["baseline_archive_custody_validated"]["passed"] is True
    assert gates["decoded_mask_u8_sha256_match_manifest"]["passed"] is True
    assert gates["decoded_mask_expected_shape"]["passed"] is True
    assert gates["runtime_integration_required"]["passed"] is False
    assert gates["exact_cuda_auth_eval_required_before_score_claim"]["passed"] is False
    assert "contest_auth_eval.py --device cuda" in gates["exact_cuda_auth_eval_required_before_score_claim"]["evidence"]
    assert first["readiness_summary"]["ready_for_local_tiny_nerv_training_prototype"] is True
    assert first["readiness_summary"]["ready_for_archive_promotion"] is False


def test_alpha_inr_readiness_rejects_hidden_archive_sidecar(tmp_path: Path) -> None:
    masks = _tiny_masks()
    archive = _write_archive(tmp_path / "archive.zip", unsafe_member="._masks.mkv")
    decoded = _write_decoded_masks(tmp_path / "decoded_masks.npy", masks)

    with pytest.raises(readiness.AlphaINRReadinessError, match="hidden/system archive member"):
        readiness.audit_alpha_inr_readiness(
            baseline_archive=archive,
            decoded_masks_source=decoded,
            output_json=None,
            num_freqs=2,
            hidden_dim=8,
            depth=3,
        )


def test_alpha_inr_readiness_fails_closed_on_decoded_source_identity_mismatch(tmp_path: Path) -> None:
    masks = _tiny_masks()
    archive = _write_archive(tmp_path / "archive.zip")
    decoded = _write_decoded_masks(tmp_path / "decoded_masks.npy", masks)
    manifest = _manifest_for(
        tmp_path / "alpha_mask_candidate_manifest.json",
        archive=archive,
        masks=masks,
        bad_decoded_sha=True,
    )

    with pytest.raises(readiness.AlphaINRReadinessError, match="decoded mask source sha256 mismatch"):
        readiness.audit_alpha_inr_readiness(
            baseline_archive=archive,
            decoded_masks_source=decoded,
            candidate_manifest=manifest,
            output_json=None,
            expected_frames=2,
            expected_height=3,
            expected_width=5,
            num_freqs=2,
            hidden_dim=8,
            depth=3,
        )


def test_alpha_inr_readiness_rejects_non_cuda_auth_eval_evidence(tmp_path: Path) -> None:
    masks = _tiny_masks()
    archive = _write_archive(tmp_path / "archive.zip")
    decoded = _write_decoded_masks(tmp_path / "decoded_masks.npy", masks)
    auth_eval = tmp_path / "contest_auth_eval.json"
    auth_eval.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive.stat().st_size,
                "provenance": {
                    "device": "cpu",
                    "cuda_available": False,
                    "archive_sha256": readiness._sha256_file(archive),
                },
            }
        )
    )

    with pytest.raises(readiness.AlphaINRReadinessError, match="expected 'cuda'"):
        readiness.audit_alpha_inr_readiness(
            baseline_archive=archive,
            decoded_masks_source=decoded,
            candidate_archive=archive,
            contest_auth_eval_json=auth_eval,
            output_json=None,
            num_freqs=2,
            hidden_dim=8,
            depth=3,
        )
