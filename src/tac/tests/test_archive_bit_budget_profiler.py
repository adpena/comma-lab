# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import random
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "experiments" / "archive_bit_budget_profiler.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("archive_bit_budget_profiler", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _rpk1_payload() -> bytes:
    members = [
        ("renderer.bin", b"QZS3" + b"renderer" * 20),
        ("masks.mkv", b"\x12\x00\x0a\x0a" + b"mask" * 31),
        ("optimized_poses.bin", b"QP1" + b"pose" * 11),
    ]
    header = {
        "schema": "renderer_payload_v1",
        "members": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256(data),
                "codec": "raw",
                "decoded_bytes": len(data),
            }
            for name, data in members
        ],
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return b"RPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(
        data for _, data in members
    )


def _deterministic_bytes(size: int, *, seed: int, prefix: bytes, vlq_safe: bool = False) -> bytes:
    rng = random.Random(seed)
    limit = 128 if vlq_safe else 256
    return prefix + bytes(rng.randrange(0, limit) for _ in range(size - len(prefix)))


def _brotli_stream_with_exact_len(
    target_len: int,
    *,
    seed: int,
    prefix: bytes,
    vlq_safe: bool = False,
) -> tuple[bytes, bytes]:
    brotli = pytest.importorskip("brotli")
    for raw_len in range(max(len(prefix), target_len - 32), target_len + 1):
        raw = _deterministic_bytes(raw_len, seed=seed, prefix=prefix, vlq_safe=vlq_safe)
        compressed = brotli.compress(raw, quality=0, mode=brotli.MODE_GENERIC, lgwin=22)
        if len(compressed) == target_len:
            return raw, compressed
    raise AssertionError(f"could not synthesize Brotli stream of length {target_len}")


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


def _sg2_action_stream_with_exact_len(target_len: int) -> bytes:
    brotli = pytest.importorskip("brotli")
    for group_count in range(1, 200):
        raw = bytearray(b"SG2")
        for group in range(group_count):
            tile = (83 + group) % 141
            count = 1 + (group % 3 == 0)
            raw += _uvarint(tile) + _uvarint(count)
            frame = 30 + group * 5
            for idx in range(count):
                raw += _uvarint(frame if idx == 0 else 1)
                raw.append((2 + group + idx) % 108)
        for quality in range(12):
            compressed = brotli.compress(
                bytes(raw),
                quality=quality,
                mode=brotli.MODE_GENERIC,
                lgwin=10,
            )
            if len(compressed) == target_len:
                return compressed
    raise AssertionError(f"could not synthesize SG2 Brotli stream of length {target_len}")


def test_argparse_surface_exposes_report_outputs() -> None:
    module = _load_module()
    parser = module.build_arg_parser()
    args = parser.parse_args(
        [
            "a.zip",
            "b.zip",
            "--output-json",
            "profile.json",
            "--output-csv",
            "profile.csv",
            "--output-md",
            "profile.md",
        ]
    )

    assert [str(path) for path in args.archives] == ["a.zip", "b.zip"]
    assert str(args.output_json) == "profile.json"
    assert str(args.output_csv) == "profile.csv"
    assert str(args.output_md) == "profile.md"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_profile_rpk1_archive_reports_segments_and_no_score_claim(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    _stored_zip(archive, {"p": _rpk1_payload()})

    report = module.build_report([archive])

    assert report["schema"] == "archive_bit_budget_profile_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert "upstream/evaluate.py on CUDA" in report["canonical_score_source_required"]
    profiled = report["archives"][0]
    assert profiled["member_count"] == 1
    assert profiled["members"][0]["name"] == "p"
    anatomy = profiled["members"][0]["fixed_slice_or_payload_anatomy"]
    assert anatomy["payload_format"] == "rpk1_renderer_payload"
    assert [segment["name"] for segment in anatomy["segments"]] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.bin",
    ]
    assert all(segment["rate_score_contribution"] > 0 for segment in anatomy["segments"])
    assert all(
        segment["compression_probe"]["best_probe"]["bytes"] > 0
        for segment in anatomy["segments"]
    )


def test_profile_outer_brotli_pr64_length_table_p_segment(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    module = _load_module()
    archive = tmp_path / "archive.zip"
    masks = b"\x12\x00\x0a\x0a" + b"mask-obu" * 9
    renderer = b"QZS3" + b"renderer" * 13
    pose = b"QP1" + b"pose" * 4
    raw = struct.pack("<III", len(masks), len(renderer), len(pose)) + masks + renderer + pose
    _stored_zip(archive, {"p": brotli.compress(raw, quality=5)})

    report = module.build_report([archive])

    anatomy = report["archives"][0]["members"][0]["fixed_slice_or_payload_anatomy"]
    assert anatomy["payload_format"] == "public_pr64_mask_first_len_table_outer_brotli"
    assert anatomy["source_payload_bytes"] < anatomy["decoded_payload_bytes"]
    assert [(segment["name"], segment["codec"]) for segment in anatomy["segments"]] == [
        ("masks.mkv", "raw"),
        ("renderer.bin", "raw"),
        ("optimized_poses.bin", "pose_qp1_v1"),
    ]


def test_profile_public_pr75_p6_payload_reports_semantic_segments(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    module = _load_module()
    archive = tmp_path / "archive.zip"
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask" * 31
    renderer_raw = b"QZS3" + b"renderer" * 21
    pose_raw = b"QP1" + b"pose" * 17
    action_delta_varints = bytes(range(20))
    record_count = 5
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(action_delta_varints), record_count)
        + mask_br
        + renderer_br
        + action_delta_varints
        + pose_br
    )
    _stored_zip(archive, {"p": payload})

    report = module.build_report([archive])

    member = report["archives"][0]["members"][0]
    assert member["payload_type_guess"]["guess"] == "public_pr75_p6_segactions_payload"
    anatomy = member["fixed_slice_or_payload_anatomy"]
    assert anatomy["payload_format"] == "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    assert [(segment["name"], segment["codec"]) for segment in anatomy["segments"]] == [
        ("masks.mkv", "brotli_av1_obu"),
        ("renderer.bin", "brotli_qzs3"),
        ("seg_tile_actions.bin", "seg_tile_actions_delta_varint_v1"),
        ("optimized_poses.qp1", "public_qp1_brotli"),
    ]
    assert [segment["encoded_bytes"] for segment in anatomy["segments"]] == [
        len(mask_br),
        len(renderer_br),
        len(action_delta_varints),
        len(pose_br),
    ]
    assert anatomy["segments"][2]["decoded_bytes_estimate"] == record_count * 4


def test_profile_public_pr75_minp_fixedslice_reports_observed_slices(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 55_756
    actions_br = b"A" * 255
    pose_br = b"P" * 898

    def fake_decompress(data: bytes) -> bytes | None:
        if data == mask_br:
            return b"\x12\x00\x0a\x0a" + b"mask"
        if data == renderer_br:
            return b"QZS3" + b"renderer"
        if data == actions_br:
            return b"SG2" + _uvarint(109) + _uvarint(1) + _uvarint(33) + bytes([2])
        if data == pose_br:
            return b"QP1" + (5120).to_bytes(2, "little") + b"pose"
        return None

    module._try_brotli_decompress = fake_decompress
    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 276_381
    _stored_zip(archive, {"p": payload})

    report = module.build_report([archive])

    anatomy = report["archives"][0]["members"][0]["fixed_slice_or_payload_anatomy"]
    assert anatomy["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert [(segment["name"], segment["encoded_bytes"]) for segment in anatomy["segments"]] == [
        ("masks.mkv", 219_472),
        ("renderer.bin", 55_756),
        ("seg_tile_actions.bin", 255),
        ("optimized_poses.qp1", 898),
    ]


def test_profile_public_pr77_fixedslice_reports_tile_delta_slices(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 55_756
    actions_br = b"A" * 325
    pose_br = b"P" * 898

    def fake_decompress(data: bytes) -> bytes | None:
        if data == mask_br:
            return b"\x12\x00\x0a\x0a" + b"mask"
        if data == renderer_br:
            return b"QZS3" + b"renderer"
        if data == actions_br:
            return (
                _uvarint(109) + _uvarint(2)
                + _uvarint(33) + bytes([2])
                + _uvarint(3) + bytes([3])
            )
        if data == pose_br:
            return b"QP1" + (5120).to_bytes(2, "little") + b"pose"
        return None

    module._try_brotli_decompress = fake_decompress
    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 276_451
    _stored_zip(archive, {"p": payload})

    report = module.build_report([archive])

    anatomy = report["archives"][0]["members"][0]["fixed_slice_or_payload_anatomy"]
    assert anatomy["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert [(segment["name"], segment["encoded_bytes"]) for segment in anatomy["segments"]] == [
        ("masks.mkv", 219_472),
        ("renderer.bin", 55_756),
        ("seg_tile_actions.bin", 325),
        ("optimized_poses.qp1", 898),
    ]
    assert anatomy["segments"][2]["decoded_bytes_estimate"] == 8


def test_public_fixed_slice_parser_requires_all_brotli_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    mask = b"M" * 219_472
    bad_prefix_len = 5
    renderer = b"R" * 7
    pose = b"P" * 3
    payload = mask + renderer + pose

    monkeypatch.setattr(module, "_public_pr67_model_lens", lambda _payload_len: [bad_prefix_len, len(renderer)])

    def fake_decompress(data: bytes) -> bytes | None:
        if data == mask:
            return b"\x12\x00\x0a\x0a" + b"obu"
        if data == renderer:
            return b"QZS3" + b"renderer"
        if data == pose:
            return b"QP1" + b"pose"
        return None

    monkeypatch.setattr(module, "_try_brotli_decompress", fake_decompress)

    parsed = module._parse_public_pr63_or_pr67(payload)

    assert parsed is not None
    payload_format, segments = parsed
    assert payload_format == "public_pr67_qzs3_qp1_fixed_slices"
    assert [(segment.name, len(segment.encoded_bytes)) for segment in segments] == [
        ("masks.mkv", len(mask)),
        ("renderer.bin", len(renderer)),
        ("optimized_poses.bin", len(pose)),
    ]
    assert [segment.decoded_bytes_estimate for segment in segments] == [7, 12, 7]


def test_public_pr67_model_lens_are_not_brittle_to_mixed_pose_length() -> None:
    module = _load_module()
    # PR67 renderer (56093B) plus C067 pose (677B) is a valid fixed-slice mix,
    # but its total length does not match the original PR67 archive band.
    payload_len = 219_472 + 56_093 + 677

    candidates = module._public_pr67_model_lens(payload_len)

    assert 56_093 in candidates
    assert 55_965 in candidates


def test_profile_rejects_zip_slip_and_duplicate_members(tmp_path: Path) -> None:
    module = _load_module()
    unsafe = tmp_path / "unsafe.zip"
    _stored_zip(unsafe, {"../p": b"payload"})
    with pytest.raises(ValueError, match="unsafe ZIP member path"):
        module.profile_archive(unsafe)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as zf:
        zf.writestr("p", b"one")
        zf.writestr("p", b"two")
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        module.profile_archive(duplicate)


def test_main_writes_json_csv_and_markdown_without_runtime_imports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output_json = tmp_path / "profile.json"
    output_csv = tmp_path / "profile.csv"
    output_md = tmp_path / "profile.md"
    _stored_zip(archive, {"p": _rpk1_payload()})

    before = set(sys.modules)
    rc = module.main(
        [
            str(archive),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    after = set(sys.modules)

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["score_claim"] is False
    assert json.loads(output_json.read_text())["archives"][0]["path"] == str(archive.resolve())
    assert output_csv.read_text().splitlines()[0].startswith("row_type,archive,member")
    markdown = output_md.read_text()
    assert "Archive Bit Budget Profile" in markdown
    assert "not score evidence" in markdown
    assert "promotion eligible: `False`" in markdown
    imported = after - before
    assert "torch" not in imported
    assert not any(name == "tac" or name.startswith("tac.") for name in imported)
    assert not any(name == "upstream" or name.startswith("upstream.") for name in imported)
