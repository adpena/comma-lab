from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPO_ROOT / "experiments/preflight_qfaithful_successor_geometry_contract.py"


def _load_preflight(name: str = "_qfaithful_successor_preflight_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PREFLIGHT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _pose_bytes(rows: int = 4, *, all_zero: bool = False) -> bytes:
    values: list[float] = []
    for row in range(rows):
        if all_zero:
            values.extend([0.0] * 6)
        else:
            values.extend([20.0 + row, 0.1, -0.2, 0.3, -0.4, 0.5])
    return struct.pack("<" + "e" * len(values), *values)


def _write_member(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    info.extra = b""
    info.comment = b""
    zf.writestr(info, data)


def _write_archive(
    path: Path,
    *,
    poses: bytes,
    renderer: bytes | None = None,
    zoom: bytes | None = b"\x00<\x00=",
    foveation: bytes | None = None,
) -> Path:
    renderer = renderer or (b"QZS3" + (32).to_bytes(2, "little") + b"weights")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        _write_member(zf, "renderer.bin", renderer)
        _write_member(zf, "masks.mkv", b"mask-bytes")
        _write_member(zf, "optimized_poses.bin", poses)
        if zoom is not None:
            _write_member(zf, "zoom_scalars.bin", zoom)
        if foveation is not None:
            _write_member(zf, "foveation_params.bin", foveation)
    return path


def _provenance(poses: bytes, *, pair_count: int = 4) -> dict[str, Any]:
    return {
        "schema": "tiny_qfaithful_successor_fixture",
        "score_claim": False,
        "promotion_eligible": False,
        "eval_roundtrip": True,
        "mask_frame_contract": "half",
        "training_pose_contract": {
            "pose_dim": 6,
            "pair_count": pair_count,
            "pose_sha256": _sha256(poses),
            "training_uses_nonzero_pose_stream": True,
            "zero_pose_fallback_allowed": False,
        },
        "renderer_zoom_contract": {
            "renderer_consumes_ego_flow": False,
            "runtime_consumes_zoom_warp_for_mask_expansion": True,
        },
        "geometry_contract": {
            "member_name": "zoom_scalars.bin",
            "geometry_consumed_by_runtime": True,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_successor_contract_passes_only_with_nonzero_pose_geometry_and_no_score_claim(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight()
    poses = _pose_bytes()
    archive = _write_archive(tmp_path / "archive.zip", poses=poses)
    provenance = _write_json(tmp_path / "provenance.json", _provenance(poses))

    report = preflight.build_report(
        provenance_paths=(provenance,),
        archive=archive,
        label="tiny_pass",
    )

    assert report["training_dispatch_allowed"] is True
    assert report["score_claim"] is False
    assert report["remote_gpu_dispatch_performed"] is False
    assert report["checks"]["training_pose_contract"]["passed"] is True
    assert report["checks"]["geometry_contract"]["passed"] is True
    assert report["archive"]["renderer_wire"]["wire_format"] == "QZS3"


def test_zero_pose_fallback_and_all_zero_archive_pose_fail_closed(tmp_path: Path) -> None:
    preflight = _load_preflight("_qfaithful_successor_preflight_zero_test")
    poses = _pose_bytes(all_zero=True)
    archive = _write_archive(tmp_path / "archive.zip", poses=poses)
    payload = _provenance(poses)
    payload["training_pose_contract"].update(
        {
            "training_uses_nonzero_pose_stream": False,
            "zero_pose_fallback_allowed": True,
        }
    )
    provenance = _write_json(tmp_path / "provenance.json", payload)

    report = preflight.build_report(
        provenance_paths=(provenance,),
        archive=archive,
        label="zero_pose",
    )

    assert report["training_dispatch_allowed"] is False
    blockers = set(report["blockers"])
    assert "training_nonzero_pose_stream_not_proven" in blockers
    assert "zero_pose_fallback_not_forbidden" in blockers
    assert "archive_pose_stream_all_zero" in blockers


def test_configured_zoom_or_foveation_without_runtime_consumption_fails_closed(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight("_qfaithful_successor_preflight_geometry_test")
    poses = _pose_bytes()
    archive = _write_archive(
        tmp_path / "archive.zip",
        poses=poses,
        zoom=b"\x00<\x00=",
        foveation=b"fov",
    )
    payload = _provenance(poses)
    payload.pop("renderer_zoom_contract")
    payload["geometry_contract"] = {
        "zoom_member": "zoom_scalars.bin",
        "foveation_member": "foveation_params.bin",
    }
    provenance = _write_json(tmp_path / "provenance.json", payload)

    report = preflight.build_report(
        provenance_paths=(provenance,),
        archive=archive,
        label="unconsumed_geometry",
    )

    assert report["training_dispatch_allowed"] is False
    blockers = set(report["checks"]["geometry_contract"]["blockers"])
    assert "zoom_warp_geometry_not_consumed_by_runtime" in blockers
    assert "foveation_geometry_not_consumed_by_runtime" in blockers


def test_archive_renderer_must_be_qfai_or_qzs3_runtime_contract(tmp_path: Path) -> None:
    preflight = _load_preflight("_qfaithful_successor_preflight_renderer_test")
    poses = _pose_bytes()
    archive = _write_archive(tmp_path / "archive.zip", poses=poses, renderer=b"BAD!renderer")
    provenance = _write_json(tmp_path / "provenance.json", _provenance(poses))

    report = preflight.build_report(
        provenance_paths=(provenance,),
        archive=archive,
        label="bad_renderer",
    )

    assert report["training_dispatch_allowed"] is False
    assert "renderer_wire_format_not_qfai_or_qzs3" in report["blockers"]
    assert report["checks"]["archive_output_contract"]["passed"] is False


def test_write_report_is_deterministic(tmp_path: Path) -> None:
    preflight = _load_preflight("_qfaithful_successor_preflight_determinism_test")
    poses = _pose_bytes()
    archive = _write_archive(tmp_path / "archive.zip", poses=poses)
    provenance = _write_json(tmp_path / "provenance.json", _provenance(poses))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    report_a = preflight.write_report(
        first,
        provenance_paths=(provenance,),
        archive=archive,
        label="deterministic",
    )
    report_b = preflight.write_report(
        second,
        provenance_paths=(provenance,),
        archive=archive,
        label="deterministic",
    )

    assert report_a == report_b
    assert first.read_bytes() == second.read_bytes()
