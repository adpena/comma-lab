from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path


def _load_module():
    path = Path("experiments/build_fixedslice_segment_mix_candidates.py").resolve()
    spec = importlib.util.spec_from_file_location("build_fixedslice_segment_mix_candidates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("p", payload)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_parse_mix_spec_requires_all_segments() -> None:
    module = _load_module()
    candidate_id, mapping = module._parse_mix_spec("combo:mask=a,renderer=b,pose=c")
    assert candidate_id == "combo"
    assert mapping == {
        "masks.mkv": "a",
        "renderer.bin": "b",
        "optimized_poses.bin": "c",
    }


def test_read_single_member_payload_rejects_multi_member(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p", b"abc")
        zf.writestr("q", b"def")

    try:
        module._read_single_member_payload(archive)
    except ValueError as exc:
        assert "single member" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsafe multi-member archive to fail")


def test_write_archive_is_deterministic(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    module._write_archive(first, b"payload")
    module._write_archive(second, b"payload")
    assert first.read_bytes() == second.read_bytes()


def test_load_source_slices_public_fixed_slices_in_wire_order(tmp_path: Path) -> None:
    module = _load_module()
    mask = b"mask-segment"
    renderer = b"renderer-segment"
    pose = b"pose-segment"
    archive = tmp_path / "archive.zip"
    _write_archive(archive, mask + renderer + pose)

    class FakeUnpacker:
        @staticmethod
        def _parse_payload(payload: bytes):
            assert payload == mask + renderer + pose
            header = {
                "payload_format": "public_pr67_qzs3_qp1_fixed_slices",
                # Runtime metadata may be logical order; the raw fixed-slice wire
                # order remains mask, renderer, pose.
                "members": [
                    {
                        "name": "renderer.bin",
                        "bytes": len(renderer),
                        "sha256": _sha256(renderer),
                        "decoded_sha256": _sha256(b"decoded-renderer"),
                    },
                    {
                        "name": "masks.mkv",
                        "bytes": len(mask),
                        "sha256": _sha256(mask),
                        "decoded_sha256": _sha256(b"decoded-mask"),
                    },
                    {
                        "name": "optimized_poses.bin",
                        "bytes": len(pose),
                        "sha256": _sha256(pose),
                        "decoded_sha256": _sha256(b"decoded-pose"),
                    },
                ],
            }
            decoded = {
                "masks.mkv": b"decoded-mask",
                "renderer.bin": b"decoded-renderer",
                "optimized_poses.bin": b"decoded-pose",
            }
            return header, decoded

    source = module._load_source("fake", archive, FakeUnpacker())

    assert source.segments == {
        "masks.mkv": mask,
        "renderer.bin": renderer,
        "optimized_poses.bin": pose,
    }


def test_build_candidates_validates_runtime_parse_contract(tmp_path: Path) -> None:
    module = _load_module()
    mask = b"mask-segment"
    renderer = b"renderer-segment"
    pose = b"pose-segment"
    sources = {
        "a": module.SourceArchive(
            label="a",
            archive_path=(tmp_path / "source.zip"),
            archive_bytes=123,
            archive_sha256="source-sha",
            payload_bytes=mask + renderer + pose,
            payload_sha256=_sha256(mask + renderer + pose),
            payload_format="public_pr67_qzs3_qp1_fixed_slices",
            segments={
                "masks.mkv": mask,
                "renderer.bin": renderer,
                "optimized_poses.bin": pose,
            },
        )
    }

    class BadRuntimeUnpacker:
        @staticmethod
        def _parse_payload(payload: bytes):
            assert payload == mask + renderer + pose
            return {
                "payload_format": "public_pr67_qzs3_qp1_fixed_slices",
                "members": [
                    {"name": "masks.mkv", "bytes": len(mask), "sha256": _sha256(mask)},
                    {
                        "name": "renderer.bin",
                        "bytes": len(renderer) - 1,
                        "sha256": _sha256(renderer[:-1]),
                    },
                    {"name": "optimized_poses.bin", "bytes": len(pose), "sha256": _sha256(pose)},
                ],
            }, {
                "masks.mkv": b"decoded-mask",
                "renderer.bin": b"decoded-renderer",
                "optimized_poses.bin": b"decoded-pose",
            }

    try:
        module.build_candidates(
            sources=sources,
            mixes=[
                (
                    "bad_runtime_boundary",
                    {"masks.mkv": "a", "renderer.bin": "a", "optimized_poses.bin": "a"},
                )
            ],
            output_dir=tmp_path / "out",
            force=True,
            unpacker=BadRuntimeUnpacker(),
        )
    except ValueError as exc:
        assert "runtime parse byte mismatch for renderer.bin" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("runtime parser mismatch was accepted")


def test_build_candidates_records_runtime_parse_validation(tmp_path: Path) -> None:
    module = _load_module()
    mask = b"mask-segment"
    renderer = b"renderer-segment"
    pose = b"pose-segment"
    sources = {
        "a": module.SourceArchive(
            label="a",
            archive_path=(tmp_path / "source.zip"),
            archive_bytes=123,
            archive_sha256="source-sha",
            payload_bytes=mask + renderer + pose,
            payload_sha256=_sha256(mask + renderer + pose),
            payload_format="public_pr67_qzs3_qp1_fixed_slices",
            segments={
                "masks.mkv": mask,
                "renderer.bin": renderer,
                "optimized_poses.bin": pose,
            },
        )
    }

    class GoodRuntimeUnpacker:
        @staticmethod
        def _parse_payload(payload: bytes):
            assert payload == mask + renderer + pose
            return {
                "payload_format": "public_pr67_qzs3_qp1_fixed_slices",
                "members": [
                    {
                        "name": "masks.mkv",
                        "bytes": len(mask),
                        "sha256": _sha256(mask),
                        "decoded_sha256": _sha256(b"decoded-mask"),
                    },
                    {
                        "name": "renderer.bin",
                        "bytes": len(renderer),
                        "sha256": _sha256(renderer),
                        "decoded_sha256": _sha256(b"decoded-renderer"),
                    },
                    {
                        "name": "optimized_poses.bin",
                        "bytes": len(pose),
                        "sha256": _sha256(pose),
                        "decoded_sha256": _sha256(b"decoded-pose"),
                    },
                ],
            }, {
                "masks.mkv": b"decoded-mask",
                "renderer.bin": b"decoded-renderer",
                "optimized_poses.bin": b"decoded-pose",
            }

    module.build_candidates(
        sources=sources,
        mixes=[
            (
                "good_runtime_boundary",
                {"masks.mkv": "a", "renderer.bin": "a", "optimized_poses.bin": "a"},
            )
        ],
        output_dir=tmp_path / "out",
        force=True,
        unpacker=GoodRuntimeUnpacker(),
    )

    manifest = (tmp_path / "out" / "good_runtime_boundary" / "build_manifest.json").read_text()
    assert '"runtime_parse_validation"' in manifest
    assert '"renderer.bin"' in manifest
