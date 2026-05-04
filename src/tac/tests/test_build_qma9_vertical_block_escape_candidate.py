from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_qma9_vertical_block_escape_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_qma9_vertical_block_escape_candidate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_vertical_block_escape_screen_emits_planning_manifest(tmp_path: Path) -> None:
    script = _load_script()
    frame = bytes([0, 0, 1, 1, 2, 2] * 4)
    raw = frame + frame
    qma9 = encode_qma9_mask(raw, frame_count=2, width=4, height=6)
    payload = qma9 + b"model" + b"pose" + b"router"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, payload)
    constants = tmp_path / "inflate.py"
    constants.write_text(
        "\n".join(
            [
                f"RANGE_MASK_BYTES = {len(qma9)}",
                "SPLIT_MODEL_REORDERED_BYTES = 5",
                "POSE_STREAM_BYTES = 4",
                "ROUTER_ACTION_BYTES = 6",
                "PACKED_PAYLOAD_BYTES = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES + POSE_STREAM_BYTES + ROUTER_ACTION_BYTES",
            ]
        )
    )

    manifest = script.build_vertical_block_escape_screen(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        candidate_id="qmb1_tiny",
        frames=2,
        block_width=2,
    )

    candidate_dir = tmp_path / "out" / "qmb1_tiny"
    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["gpu_required"] is False
    assert manifest["subset"]["decode_parity"] is True
    assert manifest["subset"]["raw_sha256"] == sha256_bytes(raw)
    assert manifest["block_copy_opportunities"]["copied_blocks"] == manifest["block_copy_opportunities"]["eligible_blocks"]
    assert (candidate_dir / "manifest.json").exists()
    assert (candidate_dir / "candidate_subset.qmb1").stat().st_size == manifest["subset"]["candidate_qmb1_bytes"]
