# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr75_minp_p6_stream_mix_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr75_minp_p6_stream_mix_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
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


def _action_records(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair_index, tile_id, action_id in records:
        out += pair_index.to_bytes(2, "little") + bytes([tile_id, action_id])
    return bytes(out)


def _p6_payload(
    builder: Any,
    *,
    renderer_tag: bytes,
    pose_tag: bytes,
    actions_raw: bytes,
) -> tuple[bytes, dict[str, bytes]]:
    decoded = {
        "masks.mkv": b"\x12\x00\x0a\x0a" + b"mask" * 24,
        "renderer.bin": b"QZS3" + renderer_tag + b"renderer" * 12,
        "seg_tile_actions.bin": actions_raw,
        "optimized_poses.qp1": b"QP1" + (5120).to_bytes(2, "little") + pose_tag,
    }
    mask_br = brotli.compress(decoded["masks.mkv"], quality=0, lgwin=10)
    renderer_br = brotli.compress(decoded["renderer.bin"], quality=0, lgwin=10)
    delta = builder.encode_delta_varint_actions(actions_raw)
    actions_br = brotli.compress(delta, quality=0, lgwin=10)
    pose_br = brotli.compress(decoded["optimized_poses.qp1"], quality=0, lgwin=10)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), len(actions_raw) // 4)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    return payload, decoded


def test_p6_stream_mix_is_deterministic_and_records_non_noop(tmp_path: Path) -> None:
    builder = _load_builder()
    c089_payload, _c089_decoded = _p6_payload(
        builder,
        renderer_tag=b"c089",
        pose_tag=b"pose-c089",
        actions_raw=_action_records([(3, 7, 1), (9, 8, 2)]),
    )
    public_payload, public_decoded = _p6_payload(
        builder,
        renderer_tag=b"pub!",
        pose_tag=b"pose-public",
        actions_raw=_action_records([(3, 7, 1), (12, 9, 4)]),
    )
    c089_archive = tmp_path / "c089.zip"
    public_archive = tmp_path / "public.zip"
    _stored_zip(c089_archive, {"p": c089_payload})
    _stored_zip(public_archive, {"p": public_payload})

    out = tmp_path / "out"
    summary = builder.build_candidates(
        c089_archive=c089_archive,
        public_archive=public_archive,
        output_dir=out,
        params=[(0, 0, 10, 0)],
    )
    first_archive = out / "p6_public_renderer_only" / "archive.zip"
    first_bytes = first_archive.read_bytes()
    summary_again = builder.build_candidates(
        c089_archive=c089_archive,
        public_archive=public_archive,
        output_dir=out,
        force=True,
        params=[(0, 0, 10, 0)],
    )

    assert first_archive.read_bytes() == first_bytes
    assert summary["candidates"][0]["score_claim"] is False
    assert summary_again["candidates"][0]["archive_sha256"] == summary["candidates"][0]["archive_sha256"]
    manifest = json.loads((out / "p6_public_renderer_only" / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["decoded_stream_closure"]["status"] == "passed"
    assert manifest["noop"] is False
    assert manifest["noop_status"] == "decoded_stream_changed_vs_c089"
    assert manifest["non_noop_proof"]["changed_decoded_streams_vs_c089"] == ["renderer"]
    assert manifest["decoded_stream_closure"]["candidate_decoded_members"]["renderer.bin"]["sha256"]
    assert manifest["decoded_stream_closure"]["candidate_decoded_members"]["renderer.bin"]["bytes"] == len(
        public_decoded["renderer.bin"]
    )
    assert set(manifest["selected_encoded_stream_sha256s"]) == {
        "actions",
        "mask",
        "pose",
        "renderer",
    }
    with zipfile.ZipFile(first_archive) as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["p"]
    assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
    assert infos[0].compress_type == zipfile.ZIP_STORED


def test_fixed_public_minp_split_validation_uses_observed_slices() -> None:
    builder = _load_builder()
    payload = b"M" * 219_472 + b"R" * 55_756 + b"A" * 255 + b"P" * 898

    streams = builder._parse_encoded_streams(payload)  # noqa: SLF001

    assert len(streams.mask_br) == 219_472
    assert len(streams.renderer_br) == 55_756
    assert len(streams.actions_br) == 255
    assert len(streams.pose_br) == 898
    assert streams.action_record_count is None


@pytest.mark.parametrize("bad_member", [".DS_Store", "__MACOSX/._p", "../p"])
def test_source_archives_reject_hidden_and_zip_slip_members(
    tmp_path: Path,
    bad_member: str,
) -> None:
    builder = _load_builder()
    good_payload, _decoded = _p6_payload(
        builder,
        renderer_tag=b"c089",
        pose_tag=b"pose",
        actions_raw=_action_records([(1, 2, 3)]),
    )
    bad_archive = tmp_path / "bad.zip"
    _stored_zip(bad_archive, {"p": good_payload, bad_member: b"junk"})
    good_archive = tmp_path / "good.zip"
    _stored_zip(good_archive, {"p": good_payload})

    with pytest.raises(ValueError, match="hidden/system archive member|unsafe ZIP member path"):
        builder.build_candidates(
            c089_archive=bad_archive,
            public_archive=good_archive,
            output_dir=tmp_path / "out",
            params=[(0, 0, 10, 0)],
        )
