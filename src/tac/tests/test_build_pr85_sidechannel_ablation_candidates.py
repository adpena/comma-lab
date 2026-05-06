from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_sidechannel_ablation_candidates.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("build_pr85_ablation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _fixture_archive(path: Path) -> None:
    decoded = {
        "mask": b"QMA9" + (1).to_bytes(4, "little") + (2).to_bytes(4, "little") + (3).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"x",
        "model": b"QH0model",
        "pose": b"P1D1" + bytes([0]),
        "post": bytes([1]) * 2400,
        "shift": b"SD4" + bytes([1]) * 600,
        "frac": b"FV1" + (1).to_bytes(2, "little") + b"\x00\x05",
        "frac2": b"FH2" + bytes([8]) * 600,
        "frac3": b"FD3" + bytes([1]) * 600,
        "bias": b"BD1" + bytes([1]) * 600,
        "region": b"RH1" + bytes([1]) * 600,
        "randmulti": bytes(sum(spec[3] for spec in module.HEADERLESS_RANDMULTI_SPECS)),
    }
    segments = {
        name: payload if name == "mask" else brotli.compress(payload, quality=5)
        for name, payload in decoded.items()
    }
    segments["bias"] = b"B" * 223
    segments["region"] = b"R" * 273
    header = b"".join(_u24(len(segments[name])) for name in module.SEGMENT_ORDER[:8])
    _zip(path, header + b"".join(segments[name] for name in module.SEGMENT_ORDER))


def test_builds_neutral_motion_stack_archive_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    payload = module.build_candidates(source, out_dir, policy_ids=["minus_motion_stack"])

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["policy_id"] == "minus_motion_stack"
    assert set(candidate["neutralized_segments"]) == {"shift", "frac", "frac2", "frac3"}
    assert candidate["candidate"]["member_name"] == "x"
    assert candidate["candidate"]["zip_stored"] is True
    assert (out_dir / "minus_motion_stack" / "archive.zip").is_file()
    assert (out_dir / "minus_motion_stack" / "manifest.json").is_file()


def test_unknown_policy_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _fixture_archive(source)

    try:
        module.build_candidates(source, tmp_path / "out", policy_ids=["bogus"])
    except ValueError as exc:
        assert "unknown policy" in str(exc)
    else:
        raise AssertionError("unknown policy should fail")


def test_fixed_length_runtime_segments_fail_closed_on_size_change(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _fixture_archive(source)

    try:
        module._validate_replacement_segment("bias", b"B" * 223, brotli.compress(b"BD1" + bytes(600), quality=11))
    except ValueError as exc:
        assert "fixed-length PR85 v5 segment" in str(exc)
    else:
        raise AssertionError("fixed-length segment replacement should fail when size changes")


def test_cli_writes_summary(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    assert module.main(["--archive", str(source), "--out-dir", str(out_dir), "--policy", "minus_post"]) == 0

    assert (out_dir / "candidate_summary.json").is_file()
    assert (out_dir / "minus_post" / "archive.zip").is_file()
    assert '"policy_id": "minus_post"' in capsys.readouterr().out
