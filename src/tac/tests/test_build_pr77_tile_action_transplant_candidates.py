from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr77_tile_action_transplant_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr77_tile_action_transplant_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _action_records(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += pair.to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


class FakeFixedSliceUnpacker:
    payload_format = "public_pr75_qzs3_qp1_segactions_fixed_slices"

    def __init__(
        self,
        *,
        pr77_payload: bytes,
        pr77_decoded: dict[str, bytes],
        target_payload: bytes,
        target_decoded: dict[str, bytes],
        candidate_payload: bytes,
        candidate_decoded: dict[str, bytes],
        segment_sizes: dict[bytes, dict[str, int]],
    ) -> None:
        self._decoded_by_payload = {
            pr77_payload: pr77_decoded,
            target_payload: target_decoded,
            candidate_payload: candidate_decoded,
        }
        self._segment_sizes = segment_sizes

    def _parse_payload(self, payload: bytes):
        decoded = self._decoded_by_payload.get(payload)
        if decoded is None:
            if payload.startswith(b"P6"):
                return {
                    "payload_format": "public_pr75_qzs3_qp1_segactions_p6_delta_varint",
                    "members": [],
                }, {}
            raise ValueError("unexpected payload")
        offset = 0
        members = []
        sizes = self._segment_sizes[payload]
        for name in ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1"):
            size = sizes[name]
            raw = payload[offset : offset + size]
            offset += size
            members.append(
                {
                    "name": name,
                    "bytes": len(raw),
                    "sha256": _sha256(raw),
                    "codec": "test_codec",
                    "decoded_bytes": len(decoded[name]),
                    "decoded_sha256": _sha256(decoded[name]),
                }
            )
        assert offset == len(payload)
        return {"payload_format": self.payload_format, "members": members}, decoded


def test_pr77_action_transplant_builder_is_deterministic_and_records_guards(tmp_path: Path) -> None:
    builder = _load_builder()
    mask = b"mask-segment"
    pr77_renderer = b"renderer-pr77"
    target_renderer = b"renderer-target"
    pr77_actions_raw = b"A" * 325
    target_actions_raw = b"B" * 236
    pose = b"pose-segment"
    pr77_actions_decoded = _action_records([(11, 82, 2), (599, 140, 107)])
    target_actions_decoded = _action_records([(22, 83, 3)])
    pr77_payload = mask + pr77_renderer + pr77_actions_raw + pose
    target_payload = mask + target_renderer + target_actions_raw + pose
    candidate_payload = mask + target_renderer + pr77_actions_raw + pose
    pr77_decoded = {
        "masks.mkv": b"decoded-mask",
        "renderer.bin": b"decoded-renderer-pr77",
        "seg_tile_actions.bin": pr77_actions_decoded,
        "optimized_poses.qp1": b"decoded-pose",
    }
    target_decoded = {
        "masks.mkv": b"decoded-mask",
        "renderer.bin": b"decoded-renderer-target",
        "seg_tile_actions.bin": target_actions_decoded,
        "optimized_poses.qp1": b"decoded-pose",
    }
    candidate_decoded = {
        **target_decoded,
        "seg_tile_actions.bin": pr77_actions_decoded,
    }
    segment_sizes = {
        pr77_payload: {
            "masks.mkv": len(mask),
            "renderer.bin": len(pr77_renderer),
            "seg_tile_actions.bin": len(pr77_actions_raw),
            "optimized_poses.qp1": len(pose),
        },
        target_payload: {
            "masks.mkv": len(mask),
            "renderer.bin": len(target_renderer),
            "seg_tile_actions.bin": len(target_actions_raw),
            "optimized_poses.qp1": len(pose),
        },
        candidate_payload: {
            "masks.mkv": len(mask),
            "renderer.bin": len(target_renderer),
            "seg_tile_actions.bin": len(pr77_actions_raw),
            "optimized_poses.qp1": len(pose),
        },
    }
    unpacker = FakeFixedSliceUnpacker(
        pr77_payload=pr77_payload,
        pr77_decoded=pr77_decoded,
        target_payload=target_payload,
        target_decoded=target_decoded,
        candidate_payload=candidate_payload,
        candidate_decoded=candidate_decoded,
        segment_sizes=segment_sizes,
    )
    pr77_archive = tmp_path / "pr77.zip"
    target_archive = tmp_path / "target.zip"
    _stored_zip(pr77_archive, pr77_payload)
    _stored_zip(target_archive, target_payload)

    output_dir = tmp_path / "out"
    summary = builder.build_candidates(
        pr77_archive=pr77_archive,
        targets=[("target", target_archive)],
        output_dir=output_dir,
        unpacker=unpacker,
    )
    first_archive = output_dir / "pr77_actions_on_target" / "archive.zip"
    first_bytes = first_archive.read_bytes()
    summary_again = builder.build_candidates(
        pr77_archive=pr77_archive,
        targets=[("target", target_archive)],
        output_dir=output_dir,
        force=True,
        unpacker=unpacker,
    )

    assert first_archive.read_bytes() == first_bytes
    assert summary["candidates"][0]["archive_sha256"] == summary_again["candidates"][0]["archive_sha256"]
    assert summary["score_claim"] is False
    assert summary["pr77_action_stream"]["encoded_bytes"] == 325
    assert summary["pr77_action_stream"]["decoded"]["record_count"] == 2
    manifest = json.loads((output_dir / "pr77_actions_on_target" / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_parse_validation"]["segments"]["seg_tile_actions.bin"]["source_label"] == "pr77"
    assert manifest["compatibility"]["semantic_status"] == "runtime_parse_only_non_action_mismatch"
    assert manifest["archive_byte_profile"]["schema"] == "archive_byte_profile_v1"
    with zipfile.ZipFile(first_archive) as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["p"]
    assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
    assert infos[0].compress_type == zipfile.ZIP_STORED


def test_pr77_action_transplant_skips_non_fixedslice_targets(tmp_path: Path) -> None:
    builder = _load_builder()
    mask = b"mask"
    renderer = b"renderer"
    actions = b"A" * 325
    pose = b"pose"
    pr77_payload = mask + renderer + actions + pose
    pr77_decoded = {
        "masks.mkv": b"decoded-mask",
        "renderer.bin": b"decoded-renderer",
        "seg_tile_actions.bin": _action_records([(11, 82, 2)]),
        "optimized_poses.qp1": b"decoded-pose",
    }
    segment_sizes = {
        pr77_payload: {
            "masks.mkv": len(mask),
            "renderer.bin": len(renderer),
            "seg_tile_actions.bin": len(actions),
            "optimized_poses.qp1": len(pose),
        }
    }
    unpacker = FakeFixedSliceUnpacker(
        pr77_payload=pr77_payload,
        pr77_decoded=pr77_decoded,
        target_payload=pr77_payload,
        target_decoded=pr77_decoded,
        candidate_payload=pr77_payload,
        candidate_decoded=pr77_decoded,
        segment_sizes=segment_sizes,
    )
    pr77_archive = tmp_path / "pr77.zip"
    p6_archive = tmp_path / "p6.zip"
    _stored_zip(pr77_archive, pr77_payload)
    _stored_zip(p6_archive, b"P6-not-fixed")

    summary = builder.build_candidates(
        pr77_archive=pr77_archive,
        targets=[("c089_p6", p6_archive)],
        output_dir=tmp_path / "out",
        unpacker=unpacker,
    )

    assert summary["candidates"] == []
    assert summary["skipped_targets"][0]["label"] == "c089_p6"
    assert summary["skipped_targets"][0]["status"] == "skipped_not_runtime_compatible_fixedslice_pr75"
