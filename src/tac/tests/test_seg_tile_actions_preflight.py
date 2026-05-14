# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.submission_archive import (
    detect_pose_manifest,
    validate_archive,
    validate_seg_tile_actions_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LIGHTNING_CLI = REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"


def _load_lightning_cli_module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "launch_lightning_batch_job_seg_tile_actions_test",
        LIGHTNING_CLI,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPO_ROOT = tmp_path
    return module


def _semantic_raw4_collision() -> bytes:
    raw = b"".join(
        pair.to_bytes(2, "little") + bytes([109, 92])
        for pair in range(40)
    )
    assert len(raw) == 160
    return raw


def _ambiguous_raw4_raw5_collision() -> bytes:
    raw = b"\x00" * 20
    assert len(raw) % 4 == 0 and len(raw) % 5 == 0
    return raw


def _uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _stored_zip_member(path: Path, name: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def _p3_payload(actions_raw: bytes) -> bytes:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 96, quality=0)
    model_br = brotli.compress(b"QZS3" + b"r" * 96, quality=0)
    actions_br = brotli.compress(actions_raw, quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8, quality=0)
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(model_br), len(actions_br))
        + mask_br
        + model_br
        + actions_br
        + pose_br
    )


def _packed_archive(path: Path, actions_raw: bytes) -> None:
    _stored_zip_member(path, "p", _p3_payload(actions_raw))


def _exact_eval_args(tmp_path: Path, archive: Path) -> argparse.Namespace:
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "archive.zip"},
                    {"path": "submissions/robust_current/inflate.sh"},
                    {"path": "submissions/robust_current/config.env"},
                ]
            }
        )
        + "\n"
    )
    return argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive=str(archive),
        repo_dir=str(tmp_path),
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
        inflate_sh="submissions/robust_current/inflate.sh",
        cloud_account=None,
    )


def test_seg_tile_action_helper_accepts_semantic_raw4_collision() -> None:
    info = validate_seg_tile_actions_payload(_semantic_raw4_collision())

    assert info["record_size"] == 4
    assert info["record_count"] == 40


def test_seg_tile_action_helper_rejects_unresolvable_raw4_raw5_collision() -> None:
    with pytest.raises(ValueError, match="ambiguous seg tile action payload length"):
        validate_seg_tile_actions_payload(_ambiguous_raw4_raw5_collision())


def test_seg_tile_action_helper_accepts_charged_grid_header() -> None:
    raw = (
        b"TG1"
        + (16).to_bytes(2, "little")
        + b"SG2"
        + _uvarint(767)
        + _uvarint(1)
        + _uvarint(599)
        + bytes([92])
    )

    info = validate_seg_tile_actions_payload(raw)

    assert info["encoding"] == "TG1+SG2"
    assert info["tile_size"] == 16
    assert info["record_size"] == 5
    assert info["record_count"] == 1


def test_archive_validator_rejects_packed_ambiguous_seg_tile_actions(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    _packed_archive(archive, _ambiguous_raw4_raw5_collision())

    result = validate_archive(archive, detect_pose_manifest(archive), strict=True)

    assert not result.valid
    assert any("ambiguous seg tile action payload length" in err for err in result.errors)


def test_archive_validator_accepts_packed_semantic_raw4_collision(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    _packed_archive(archive, _semantic_raw4_collision())

    result = validate_archive(archive, detect_pose_manifest(archive), strict=True)

    assert result.valid, result.summary()


def test_exact_eval_submit_blocks_ambiguous_seg_tile_actions_before_dispatch(
    tmp_path: Path,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    archive = tmp_path / "archive.zip"
    _packed_archive(archive, _ambiguous_raw4_raw5_collision())

    with pytest.raises(SystemExit, match="before GPU dispatch"):
        module._validate_exact_eval_submit_inputs(_exact_eval_args(tmp_path, archive))

    _packed_archive(archive, _semantic_raw4_collision())
    module._validate_exact_eval_submit_inputs(_exact_eval_args(tmp_path, archive))


def test_exact_eval_submit_skips_seg_tile_guard_for_external_inflate_runtime(
    tmp_path: Path,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    archive = tmp_path / "archive.zip"
    _stored_zip_member(archive, "p", b"QMA9" + b"\x00" * 32)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "archive.zip"},
                    {"path": "experiments/results/public_replay/inflate.sh"},
                    {"path": "experiments/results/public_replay/config.env"},
                ]
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive=str(archive),
        repo_dir=str(tmp_path),
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
        inflate_sh="experiments/results/public_replay/inflate.sh",
        cloud_account=None,
    )

    module._validate_exact_eval_submit_inputs(args)
