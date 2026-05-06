"""Tests for the prototype QBF1 JointFrameGenerator block-FP container."""

from __future__ import annotations

import importlib.util
import json
import pickle
import struct
import sys
from pathlib import Path

import pytest
import torch

from tac.qbf1_renderer_codec import (
    QBF1_MAGIC,
    QBF1_V2_BYTE_PROFILE_SCHEMA,
    QBF1_V2_REFERENCE_QZS3_NBYTES,
    QBF1ByteAccounting,
    QBF1CodecError,
    decode_qbf1_state_dict,
    load_qbf1,
    pack_qbf1_state_dict,
    profile_qbf1_v2_renderer_bytes,
    qbf1_byte_accounting,
    unpack_qbf1_container,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer


REPO_ROOT = Path(__file__).resolve().parents[3]
UNPACK_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
PROFILE_PATH = REPO_ROOT / "experiments" / "profile_qbf1_v2_renderer_bytes.py"


def _load_unpacker():
    spec = importlib.util.spec_from_file_location("qbf1_unpack_renderer_payload_test", UNPACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_profiler():
    spec = importlib.util.spec_from_file_location("qbf1_v2_renderer_byte_profile_test", PROFILE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    return {
        "frame1_head.block1.film_proj.bias": torch.linspace(-0.2, 0.2, 9),
        "mask_decoder.embedding.weight": torch.tensor(
            [
                [0.0, 0.25, -0.5],
                [1.0, -1.5, 2.0],
            ],
            dtype=torch.float32,
        ),
        "pose_mlp.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0,
        "zero_block.weight": torch.zeros(5, dtype=torch.float32),
    }


def test_qbf1_joint_frame_generator_roundtrip_is_deterministic() -> None:
    """The same JointFrameGenerator state_dict always emits identical bytes."""

    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer()
    state_dict = model.state_dict()

    blob_a = pack_qbf1_state_dict(state_dict, block_size=32)
    blob_b = pack_qbf1_state_dict(state_dict, block_size=32)
    reversed_state_dict = dict(reversed(list(state_dict.items())))
    blob_c = pack_qbf1_state_dict(reversed_state_dict, block_size=32)

    assert blob_a == blob_b
    assert blob_a == blob_c
    assert blob_a.startswith(QBF1_MAGIC)

    container = unpack_qbf1_container(blob_a)
    decoded = decode_qbf1_state_dict(blob_a)
    assert container.tensor_names() == tuple(sorted(state_dict))
    assert set(decoded) == set(state_dict)

    metadata = container.metadata_by_name()
    for name, original in state_dict.items():
        rec = metadata[name]
        restored = decoded[name]
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert rec.shape == tuple(original.shape)
        assert rec.dtype == "float32"
        assert rec.block_size == 32
        assert rec.num_blocks == len(rec.scales)
        max_scale = max(rec.scales) if rec.scales else 0.0
        max_error = (restored.float() - original.float()).abs().max().item()
        assert max_error <= max_scale + 1e-6


def test_qbf1_no_pickle_or_torch_load_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pack/unpack/decode must not use pickle or torch.load."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pickle/torch.load must not be used by QBF1")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)

    blob = pack_qbf1_state_dict(_tiny_state_dict(), block_size=4)
    container = unpack_qbf1_container(blob)
    decoded = decode_qbf1_state_dict(blob)

    assert container.tensor_names() == tuple(sorted(_tiny_state_dict()))
    assert decoded["pose_mlp.0.weight"].shape == (3, 4)


def test_qbf1_loads_joint_frame_generator_without_torch_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QBF1 can construct a JointFrameGenerator without pickle fallback."""

    model = build_quantizr_faithful_renderer()
    blob = pack_qbf1_state_dict(model.state_dict(), block_size=32)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("QBF1 loader must not use torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    loaded = load_qbf1(blob, device="cpu")

    assert loaded.pose_dim == model.pose_dim
    assert set(loaded.state_dict()) == set(model.state_dict())
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    pose = torch.zeros((1, loaded.pose_dim), dtype=torch.float32)
    frame1, frame2 = loaded(mask, pose)
    assert frame1.shape == (1, 3, 384, 512)
    assert frame2.shape == (1, 3, 384, 512)
    assert torch.isfinite(frame1).all()
    assert torch.isfinite(frame2).all()


def test_qbf1_inflate_renderer_loader_dispatches_before_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contest inflate renderer recognizes QBF1 magic explicitly."""

    from submissions.robust_current import inflate_renderer

    model = build_quantizr_faithful_renderer()
    renderer_path = tmp_path / "renderer.bin"
    renderer_path.write_bytes(pack_qbf1_state_dict(model.state_dict(), block_size=32))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("QBF1 runtime path fell through to torch.load")

    monkeypatch.setattr(inflate_renderer.torch, "load", forbidden)
    renderer = inflate_renderer._load_renderer(str(renderer_path), "cpu")

    assert getattr(renderer, "q_faithful", False) is True
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    pose = torch.zeros((1, renderer.pose_dim), dtype=torch.float32)
    pair = renderer(mask, mask, pose=pose)
    assert pair.shape == (1, 2, 384, 512, 3)
    assert torch.isfinite(pair).all()


def test_qbf1_pr64_mask_first_payload_is_detected_as_renderer() -> None:
    """Packed mask-first archives must not swap QBF1 renderer and masks."""

    unpacker = _load_unpacker()
    masks = b"\x12\x00mask-obu"
    renderer = QBF1_MAGIC + b"renderer-readiness-bytes"
    poses = struct.pack("<" + "e" * 12, *([0.0] * 12))
    payload = struct.pack("<III", len(masks), len(renderer), len(poses)) + masks + renderer + poses

    parsed = unpacker._try_parse_pr64_len_table_payload(payload)

    assert parsed is not None
    header, members = parsed
    assert header["payload_format"] == "public_pr64_mask_first_len_table"
    assert members["renderer.bin"] == renderer
    assert members["masks.mkv"] == masks


def test_qbf1_bad_magic_fails_closed() -> None:
    blob = bytearray(pack_qbf1_state_dict(_tiny_state_dict(), block_size=4))
    blob[:4] = b"BAD!"

    with pytest.raises(QBF1CodecError, match="bad QBF1 magic"):
        unpack_qbf1_container(blob)


def test_qbf1_truncated_payload_fails_closed() -> None:
    blob = pack_qbf1_state_dict(_tiny_state_dict(), block_size=4)

    with pytest.raises(QBF1CodecError, match="length mismatch"):
        unpack_qbf1_container(blob[:-1])


def test_qbf1_byte_size_accounting_and_tensor_metadata() -> None:
    blob = pack_qbf1_state_dict(_tiny_state_dict(), block_size=4)
    container = unpack_qbf1_container(blob)
    accounting = qbf1_byte_accounting(blob)

    assert isinstance(accounting, QBF1ByteAccounting)
    assert accounting == container.byte_accounting
    assert accounting.total_nbytes == len(blob)
    assert accounting.header_nbytes == 20
    assert accounting.payload_nbytes == sum(
        record.metadata.encoded_nbytes for record in container.tensors
    )
    assert accounting.tensor_payload_nbytes == accounting.payload_nbytes
    assert accounting.metadata_nbytes == len(blob) - accounting.header_nbytes - accounting.payload_nbytes

    for record in container.tensors:
        meta = record.metadata
        assert meta.quant_dtype == "int8"
        assert meta.scale_dtype == "float32"
        assert meta.scale_nbytes == len(meta.scales) * 4
        assert meta.payload_nbytes == len(record.quantized)
        assert meta.payload_nbytes == meta.value_count
        assert meta.encoded_nbytes == meta.scale_nbytes + meta.payload_nbytes
        assert meta.original_nbytes == meta.value_count * 4
        assert meta.block_size == 4
        assert all(scale >= 0.0 for scale in meta.scales)

    zero_meta = container.metadata_by_name()["zero_block.weight"]
    assert zero_meta.scales == (0.0, 0.0)


def test_qbf1_v2_byte_profile_marks_self_describing_no_go_vs_qzs3() -> None:
    """QBF1-v2 must not advance unless it locally beats the QZS3 renderer slice."""

    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    profile = profile_qbf1_v2_renderer_bytes(
        model,
        block_sizes=(32, 64, 128),
        reference_qzs3_nbytes=QBF1_V2_REFERENCE_QZS3_NBYTES,
    )

    assert profile["schema"] == QBF1_V2_BYTE_PROFILE_SCHEMA
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["reference"]["actual_local_raw_nbytes"] == QBF1_V2_REFERENCE_QZS3_NBYTES

    best = profile["best"]
    assert best["qbf1_v1"]["raw_nbytes"] > QBF1_V2_REFERENCE_QZS3_NBYTES
    assert best["qbf1_v2_self_describing"]["raw_nbytes"] > QBF1_V2_REFERENCE_QZS3_NBYTES
    assert best["qbf1_v2_self_describing"]["strict_pickle_free_loader_ready"] is False
    assert profile["readiness"]["qbf1_v2_go"] is False
    assert profile["readiness"]["qbf1_v2_dispatchable"] is False
    assert any("existing QZS3" in reason for reason in profile["readiness"]["no_go_reasons"])

    qzs3_policy = best["existing_qzs3_block_policy"]
    assert qzs3_policy["wire_format"] == "QZS3"
    assert qzs3_policy["is_qbf1_v2"] is False
    assert qzs3_policy["raw_nbytes"] < QBF1_V2_REFERENCE_QZS3_NBYTES


def test_qbf1_v2_profile_script_writes_empirical_no_score_json(tmp_path: Path) -> None:
    profiler = _load_profiler()
    out = tmp_path / "qbf1_v2_profile.json"

    profile = profiler.write_profile(
        out,
        block_sizes=(32, 128),
        reference_qzs3_nbytes=QBF1_V2_REFERENCE_QZS3_NBYTES,
    )

    assert json.loads(out.read_text()) == profile
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["readiness"]["qbf1_v2_go"] is False
