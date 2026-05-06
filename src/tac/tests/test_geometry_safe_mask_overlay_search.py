from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "experiments" / "geometry_safe_mask_overlay_search.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "geometry_safe_mask_overlay_search_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_single_p_archive_is_deterministic_and_safe(tmp_path: Path) -> None:
    mod = _load_module()
    payload = b"P6payload"
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"

    mod.write_single_p_archive(a, payload)
    mod.write_single_p_archive(b, payload)

    assert a.read_bytes() == b.read_bytes()
    assert mod.read_single_p_archive(a) == payload
    with zipfile.ZipFile(a, "r") as zf:
        info = zf.infolist()[0]
        assert info.filename == "p"
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_p6_parse_and_rebuild_roundtrip() -> None:
    mod = _load_module()
    payload = (
        b"P6"
        + struct.pack("<IHHH", 4, 3, 2, 7)
        + b"mask"
        + b"mdl"
        + b"ac"
        + b"pose"
    )

    slices = mod.parse_p6_payload(payload)

    assert slices.mask_br == b"mask"
    assert slices.model_br == b"mdl"
    assert slices.actions_br == b"ac"
    assert slices.pose_br == b"pose"
    assert slices.record_count == 7
    assert mod.build_p6_payload(slices) == payload


def test_compare_classes_reports_bounded_disagreement() -> None:
    mod = _load_module()
    source = np.zeros((2, 2, 4), dtype=np.uint8)
    candidate = source.copy()
    candidate[0, 0, 0] = 1
    candidate[1, 1, 2:4] = 2

    metrics = mod.compare_classes(source, candidate)

    assert metrics["zero_disagreement"] is False
    assert metrics["changed_pixel_count"] == 3
    assert metrics["changed_frame_count"] == 2
    assert metrics["max_changed_pixels_per_frame"] == 2
    assert metrics["source_to_candidate_confusion_5x5"][0][0] == 13
    assert metrics["source_to_candidate_confusion_5x5"][0][1] == 1
    assert metrics["source_to_candidate_confusion_5x5"][0][2] == 2


def test_cdo1_overlay_payload_has_sorted_runs_and_sha_contract() -> None:
    mod = _load_module()
    base = np.zeros((2, 2, 6), dtype=np.uint8)
    target = base.copy()
    target[0, 0, 1:4] = 3
    target[0, 0, 5] = 3
    target[1, 1, 0:2] = 4

    runs = mod.cdo1_runs(base, target)
    payload = mod.encode_cdo1_payload(
        base=base,
        target=target,
        runs=runs,
        producer="unit-test",
        policy_id="synthetic",
    )

    assert runs == [
        (0, 0, 1, 3, 3),
        (0, 0, 5, 1, 3),
        (1, 1, 0, 2, 4),
    ]
    magic, version, header_len = mod.CDO1_HEADER_STRUCT.unpack(
        payload[: mod.CDO1_HEADER_STRUCT.size]
    )
    assert magic == b"CDO1"
    assert version == 1
    header_start = mod.CDO1_HEADER_STRUCT.size
    header = json.loads(payload[header_start : header_start + header_len].decode("utf-8"))
    assert header["base_mask_tensor_sha256"] == mod.mask_tensor_sha256(base)
    assert header["reconstructed_mask_u8_sha256"] == mod.mask_tensor_sha256(target)
    assert header["pair_index_basis"] == "half_frame_pair_index"
    assert header["run_count"] == 3
    assert header["selected_pixel_count"] == 6
