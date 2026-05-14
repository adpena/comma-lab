# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments/profile_pr96_rem2_hnerv_packing.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr96_rem2_hnerv_packing", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _noise(size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out.extend(hashlib.sha256(f"block-{counter}".encode("ascii")).digest())
        counter += 1
    return bytes(out[:size])


def _decoder_payload() -> bytes:
    record = io.BytesIO()
    name = b"rgb_0.bias"
    record.write(struct.pack("<I", len(name)))
    record.write(name)
    record.write(struct.pack("<I", 1))
    record.write(struct.pack("<I", 3))
    record.write(struct.pack("<f", 0.25))
    record.write(b"\xff\x00\x01")

    br_raw = struct.pack("<I", 1) + record.getvalue()
    br = brotli.compress(br_raw, quality=5)
    header = struct.pack("<IIIIB", len(br), 0, 0, 0, 2)
    return header + br + _noise(4096)


def _latents_payload() -> bytes:
    return (
        struct.pack("<II", 4, 2)
        + b"\x00\x00\x01\x00"
        + b"\x00<\x00<"
        + bytes([0, 1, 1, 1, 2, 1, 3, 1])
    )


def _write_archive(path: Path) -> dict[str, bytes]:
    payloads = {
        "decoder.bin": _decoder_payload(),
        "latents.bin": _latents_payload(),
        "p": b"\x00" * 930,
    }
    with zipfile.ZipFile(path, "w", allowZip64=False) as zf:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 5, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED if name != "p" else zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return payloads


def _write_runtime(path: Path, *, reads_p: bool = False) -> None:
    p_read = "\n    (archive_dir / \"p\").read_bytes()" if reads_p else ""
    path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "def main():",
                "    archive_dir = Path('archive')",
                "    (archive_dir / \"decoder.bin\").read_bytes()",
                "    (archive_dir / \"latents.bin\").read_bytes()",
                p_read,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_pr96_profiler_builds_member_preserving_and_drop_unused_candidates(tmp_path: Path) -> None:
    module = load_module()
    archive = tmp_path / "archive.zip"
    runtime_py = tmp_path / "inflate.py"
    payloads = _write_archive(archive)
    _write_runtime(runtime_py)

    profile = module.build_profile(
        archive,
        runtime_py,
        tmp_path / "out",
        deflate_levels=[1, 6, 9],
    )

    assert profile["schema"] == "pr96_rem2_hnerv_packing_profile_v1"
    assert profile["score_claim"] is False
    assert profile["safety"]["does_not_launch_remote_jobs"] is True
    assert profile["runtime_static_read_set"]["read_members"] == ["decoder.bin", "latents.bin"]
    assert profile["byte_opportunities"]["unused_members"] == ["p"]
    assert profile["pr96_payloads"]["decoder"]["brotli_records"]["count"] == 1
    assert profile["pr96_payloads"]["latents"]["n_rows"] == 4

    candidates = {candidate["label"]: candidate for candidate in profile["candidates"]}
    assert set(candidates) == {
        "member_preserving_repack",
        "drop_statically_unused_members_repack",
    }
    member_preserving = candidates["member_preserving_repack"]
    drop_unused = candidates["drop_statically_unused_members_repack"]
    assert member_preserving["archive_byte_delta"] < 0
    assert drop_unused["archive_byte_delta"] < member_preserving["archive_byte_delta"]
    assert drop_unused["removed_members"] == ["p"]
    assert profile["recommended_candidate"] == "drop_statically_unused_members_repack"

    with zipfile.ZipFile(drop_unused["archive"], "r") as zf:
        assert zf.namelist() == ["decoder.bin", "latents.bin"]
        assert zf.read("decoder.bin") == payloads["decoder.bin"]
        assert zf.read("latents.bin") == payloads["latents.bin"]
        assert zf.getinfo("decoder.bin").compress_type == zipfile.ZIP_STORED

    with zipfile.ZipFile(member_preserving["archive"], "r") as zf:
        assert zf.namelist() == ["decoder.bin", "latents.bin", "p"]
        assert zf.read("p") == payloads["p"]
        assert zf.getinfo("p").compress_type == zipfile.ZIP_DEFLATED


def test_pr96_profiler_preserves_p_when_runtime_reads_it(tmp_path: Path) -> None:
    module = load_module()
    archive = tmp_path / "archive.zip"
    runtime_py = tmp_path / "inflate.py"
    _write_archive(archive)
    _write_runtime(runtime_py, reads_p=True)

    profile = module.build_profile(
        archive,
        runtime_py,
        tmp_path / "out",
        deflate_levels=[1, 6, 9],
    )

    assert profile["byte_opportunities"]["unused_members"] == []
    assert [candidate["label"] for candidate in profile["candidates"]] == [
        "member_preserving_repack"
    ]
    candidate = profile["candidates"][0]
    with zipfile.ZipFile(candidate["archive"], "r") as zf:
        assert zf.namelist() == ["decoder.bin", "latents.bin", "p"]


def test_pr96_profiler_cli_writes_manifest(tmp_path: Path) -> None:
    module = load_module()
    archive = tmp_path / "archive.zip"
    runtime_py = tmp_path / "inflate.py"
    output_dir = tmp_path / "out"
    _write_archive(archive)
    _write_runtime(runtime_py)

    class Args:
        def __init__(self) -> None:
            self.archive = str(archive)
            self.runtime_py = str(runtime_py)
            self.output_dir = str(output_dir)
            self.deflate_levels = "1,6,9"
            self.keep_unused = False

    assert module.run(Args()) == 0
    manifest = json.loads((output_dir / "profile_pr96_rem2_hnerv_packing.json").read_text())
    assert manifest["recommended_candidate_archive"].endswith("archive.pr96_drop_unused_repack.zip")
    assert Path(manifest["recommended_candidate_archive"]).is_file()
