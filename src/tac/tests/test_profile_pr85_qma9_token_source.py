from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_qma9_token_source.py"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_qma9_token_source_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _write_pr85_like_archive(path: Path, raw_tokens: bytes, *, frames: int = 2, width: int = 4, height: int = 3) -> tuple[bytes, dict[str, bytes]]:
    qma9 = encode_qma9_mask(raw_tokens, frame_count=frames, width=width, height=height)
    segments = {
        "mask": qma9,
        "model": b"model-payload",
        "pose": b"pose-payload",
        "post": b"post-payload",
        "shift": b"shift-payload",
        "frac": b"frac-payload",
        "frac2": b"frac2-payload",
        "frac3": b"frac3-payload",
        "bias": b"b" * 223,
        "region": b"r" * 273,
        "randmulti": b"randmulti-tail",
    }
    header = b"".join(
        _u24(len(segments[name]))
        for name in ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    )
    bundle = header + b"".join(segments[name] for name in _load_script().SEGMENT_ORDER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as zf:
        zf.writestr("x", bundle)
    return qma9, segments


def test_header_only_profile_is_non_dispatchable(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    raw_tokens = bytes([0, 1, 2, 3, 4, 0] * 4)
    qma9, _segments = _write_pr85_like_archive(archive, raw_tokens)
    out = tmp_path / "profile.json"

    report = script.build_profile(
        archive=archive,
        member="x",
        profile_json=None,
        output_dir=tmp_path / "out",
        output_json=out,
        extract_raw_tokens=False,
        decode_implementation="python",
        cpp_decoder=tmp_path / "missing.cpp",
        cpp_timeout_seconds=5,
        verify_reencode=False,
    )

    assert out.is_file()
    assert report["planning_only"] is True
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["dispatch_unlocked"] is False
    assert report["mask_segment_identity"]["sha256"] == sha256_bytes(qma9)
    assert report["exactness"]["mask_segment_byte_exact"] is True
    assert report["exactness"]["raw_tensor_extracted"] is False
    assert report["token_source"]["shape"] == [2, 4, 3]
    assert report["token_source"]["range_contract"] == {"min": 0, "max": 4, "sentinel_internal_only": 5}


def test_extracts_raw_tokens_and_records_shape_range_sha(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    raw_tokens = bytes([0, 1, 1, 2, 3, 4, 4, 0, 2, 2, 3, 1] * 2)
    qma9, _segments = _write_pr85_like_archive(archive, raw_tokens)
    out = tmp_path / "profile.json"

    report = script.build_profile(
        archive=archive,
        member="x",
        profile_json=None,
        output_dir=tmp_path / "out",
        output_json=out,
        extract_raw_tokens=True,
        decode_implementation="python",
        cpp_decoder=tmp_path / "missing.cpp",
        cpp_timeout_seconds=5,
        verify_reencode=True,
    )

    token_source = report["token_source"]
    assert token_source["extracted"] is True
    assert token_source["shape"] == [2, 4, 3]
    assert token_source["bytes"] == len(raw_tokens)
    assert token_source["sha256"] == sha256_bytes(raw_tokens)
    assert token_source["observed_range"] == {"min": 0, "max": 4}
    assert token_source["invalid_symbol_values"] == []
    assert token_source["class_counts"] == {"0": 4, "1": 6, "2": 6, "3": 4, "4": 4}
    assert report["exactness"]["raw_tensor_exact"] is True
    assert report["reencode_check"]["performed"] is True
    assert report["reencode_check"]["byte_exact"] is True
    assert report["reencode_check"]["source_mask_sha256"] == sha256_bytes(qma9)
    assert json.loads(out.read_text())["dispatch_unlocked"] is False


def test_cli_writes_requested_json_with_explicit_flags(tmp_path: Path, capsys) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    raw_tokens = bytes([0, 1, 2, 3, 4, 0] * 4)
    _write_pr85_like_archive(archive, raw_tokens)
    out = tmp_path / "profile.json"

    rc = script.main(
        [
            "--archive",
            str(archive),
            "--member",
            "x",
            "--output-dir",
            str(tmp_path / "out"),
            "--output-json",
            str(out),
            "--extract-raw-tokens",
            "--decode-implementation",
            "python",
            "--verify-reencode",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["tool"] == "experiments/profile_pr85_qma9_token_source.py"
    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert "dispatch_unlocked=false" in capsys.readouterr().out
