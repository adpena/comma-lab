from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_qma9_range_mask_bitstream.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_qma9_range_mask_bitstream_test", SCRIPT)
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


def test_build_profile_emits_pure_python_state_trace_without_gpu_or_cpp(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 0, 1, 1, 2, 2])
    qma9 = encode_qma9_mask(raw, frame_count=1, width=2, height=3)
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

    profile = script.build_profile(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        cpp_profiler=SCRIPT,
        pure_python_max_pixels=len(raw),
        checkpoint_pixels=(0, 5),
        skip_cpp_full=True,
        cpp_timeout_seconds=1,
    )

    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["gpu_required"] is False
    assert profile["qma9_header"]["payload_sha256"] == sha256_bytes(qma9)
    assert profile["pure_python_prefix_trace"]["decoded_prefix_sha256"] == sha256_bytes(raw)
    assert profile["pure_python_prefix_trace"]["checkpoints"][1]["pixel_index"] == 5
    assert profile["cpp_full_profile"] == {"status": "skipped"}
    assert (tmp_path / "out" / "range_mask_pr81.qma9").read_bytes() == qma9


def test_parse_split_constants_accepts_pr84_fixed_slice_source_shape(tmp_path: Path) -> None:
    script = _load_script()
    source = tmp_path / "inflate.py"
    source.write_text(
        "\n".join(
            [
                "RANGE_MASK_BYTES = 159011",
                "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES = 37086",
                "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES = 3035",
                "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES = 15604",
                "SPLIT_MODEL_REORDERED_BYTES = (",
                "    SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
                ")",
                "ROUTER_ACTION_BYTES = 225",
            ]
        ),
        encoding="utf-8",
    )

    constants = script.parse_split_constants(source)

    assert constants["RANGE_MASK_BYTES"] == 159_011
    assert constants["SPLIT_MODEL_REORDERED_BYTES"] == 55_725
    assert constants["POSE_STREAM_BYTES"] == 899
    assert constants["ROUTER_ACTION_BYTES"] == 225
    assert constants["PACKED_PAYLOAD_BYTES"] == 215_860
