from __future__ import annotations

import hashlib
import importlib.util
import lzma
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_cmg3_rowspan_candidate.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cmg3_payload_roundtrips_through_runtime_loader(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_builder_payload_test")
    inflate = _load(INFLATE_PATH, "_cmg3_inflate_payload_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[:, 80:180, 120:260] = 2
    source[:, 160:240, 220:420] = 4
    spans = builder.row_spans(source, row_stride=4)
    policy = builder.choose_rowspan_policy(spans, source, row_stride=4)
    payload, header = builder.encode_cmg3_payload(
        spans,
        row_stride=4,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        policy=policy,
        compressor="lzma_xz",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    masks = inflate._load_masks_from_cmg3(path, expected_frames=1)
    expected = builder.reconstruct_row_spans(
        spans,
        row_stride=4,
        default_class=policy["default_class"],
        row_fill=policy["row_fill"],
        draw_order=tuple(policy["draw_order"]),
    )

    np.testing.assert_array_equal(masks.numpy(), expected)
    assert header["body_sha256"] == hashlib.sha256(lzma.compress(spans.astype("<i2").tobytes(), preset=6, format=lzma.FORMAT_XZ)).hexdigest()


def test_cmg3_runtime_rejects_tampered_body(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_builder_tamper_test")
    inflate = _load(INFLATE_PATH, "_cmg3_inflate_tamper_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    spans = builder.row_spans(source, row_stride=4)
    policy = builder.choose_rowspan_policy(spans, source, row_stride=4)
    payload, _header = builder.encode_cmg3_payload(
        spans,
        row_stride=4,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        policy=policy,
        compressor="lzma_xz",
    )
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    path = tmp_path / "masks.cmg3"
    path.write_bytes(bytes(tampered))

    try:
        inflate._load_masks_from_cmg3(path, expected_frames=1)
    except ValueError as exc:
        assert "body SHA mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered CMG3 payload was accepted")


def test_cmg3_policy_search_prefers_lower_disagreement() -> None:
    builder = _load(BUILDER_PATH, "_cmg3_builder_policy_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[:, 12:96, 10:500] = 1
    source[:, 52:180, 80:240] = 4
    spans = builder.row_spans(source, row_stride=4)
    policy = builder.choose_rowspan_policy(spans, source, row_stride=4)

    all_scores = [item["pixel_disagreement"] for item in policy["searched_policies"]]
    assert policy["pixel_disagreement"] == min(all_scores)
    assert policy["row_fill"] in {"nearest", "forward", "linear"}
    assert policy["searched_policy_count"] == 3 * 5 * 120
    assert policy["policy_search"]["type"] == "complete_finite_rowspan_policy_space"


def test_cmg3_linear_row_fill_interpolates_sampled_spans() -> None:
    builder = _load(BUILDER_PATH, "_cmg3_builder_linear_fill_test")

    spans = np.full((1, 5, 2, 2), -1, dtype=np.int16)
    spans[0, 2, 0] = (10, 20)
    spans[0, 2, 1] = (30, 50)

    expanded = builder.expanded_row_spans(spans, height=4, row_stride=2, row_fill="linear")

    assert expanded.shape == (1, 5, 4, 2)
    assert tuple(int(v) for v in expanded[0, 2, 0]) == (10, 20)
    assert tuple(int(v) for v in expanded[0, 2, 1]) == (20, 35)
    assert tuple(int(v) for v in expanded[0, 2, 2]) == (30, 50)
    assert tuple(int(v) for v in expanded[0, 2, 3]) == (30, 50)


def test_cmg3_runtime_accepts_linear_row_fill(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_builder_linear_runtime_test")
    inflate = _load(INFLATE_PATH, "_cmg3_inflate_linear_runtime_test")

    source = np.zeros((2, 384, 512), dtype=np.uint8)
    for y in range(384):
        source[:, y, 50 + y // 4 : 180 + y // 3] = 2
    spans = builder.row_spans(source, row_stride=4)
    policy = {
        "default_class": 0,
        "row_fill": "linear",
        "draw_order": [0, 1, 2, 3, 4],
    }
    payload, _header = builder.encode_cmg3_payload(
        spans,
        row_stride=4,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        policy=policy,
        compressor="lzma_xz",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    expected = builder.reconstruct_row_spans(
        spans,
        row_stride=4,
        default_class=0,
        row_fill="linear",
        draw_order=(0, 1, 2, 3, 4),
    )
    masks = inflate._load_masks_from_cmg3(path, expected_frames=2)

    np.testing.assert_array_equal(masks.numpy(), expected)


def test_cmg3_member_is_allowed_in_packed_payload(tmp_path: Path) -> None:
    packer = _load(REPO / "experiments" / "build_renderer_packed_payload_archive.py", "_cmg3_packer_test")
    unpacker = _load(REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py", "_cmg3_unpacker_test")

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"QZS3fake")
        zf.writestr("masks.cmg3", b"CMG3fake")
        zf.writestr("optimized_poses.bin", struct.pack("<" + "e" * 12, *([0.0] * 12)))

    archive = tmp_path / "archive.zip"
    result = packer.build_packed_archive(
        source,
        archive,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    assert result["score_claim"] is False
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["p"]
        payload = zf.read("p")
    archive_dir = tmp_path / "extracted"
    archive_dir.mkdir()
    (archive_dir / "p").write_bytes(payload)
    summary = unpacker.unpack_renderer_payload(archive_dir)
    assert sorted(member["name"] for member in summary["members"]) == [
        "masks.cmg3",
        "optimized_poses.bin",
        "renderer.bin",
    ]
