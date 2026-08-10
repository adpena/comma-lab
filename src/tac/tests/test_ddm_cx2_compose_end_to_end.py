"""Receiver and wire-identity tests for the PR130 CX2 composition."""

from __future__ import annotations

import importlib
import importlib.util
import lzma
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import brotli
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "src" / "tac" / "pr130_runtime" / "dv1_cpu_runtime"
def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prior_receiver = sys.modules.get("receiver")
sys.path.insert(0, str(TREE))
try:
    receiver = _load_module("ddm_cx2_receiver", TREE / "receiver.py")
    sys.modules["receiver"] = receiver
    inflate = _load_module("ddm_cx2_inflate", TREE / "inflate.py")
finally:
    sys.path.remove(str(TREE))
    if prior_receiver is None:
        sys.modules.pop("receiver", None)
    else:
        sys.modules["receiver"] = prior_receiver

BASE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/"
    "reproduction/archive.zip"
)
SD1_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/"
    "cpu_screen/archives/selected_mixed_n600.zip"
)


def _sections(path: Path) -> tuple[bytes, bytes, bytes, bytes]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("p")
    model_bytes = struct.unpack_from("<I", payload)[0]
    raw = lzma.decompress(payload[4:4 + model_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    return (
        raw[8:semantic_end],
        raw[semantic_end:carrier_end],
        raw[carrier_end:],
        payload[4 + model_bytes:],
    )


def _split_pack(streams: tuple[bytes, bytes, bytes]) -> bytes:
    return struct.pack("<III", *(len(value) for value in streams)) + b"".join(
        streams
    )


def _legacy_int4_reference(
    blob: bytes,
    template: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    remaining = memoryview(blob)
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        shape = tuple(value.shape)
        count = value.numel()
        if value.ndim < 2:
            byte_count = count * 2
            array = inflate.np.frombuffer(
                remaining[:byte_count], dtype="<f2"
            ).copy()
            restored[name] = torch.from_numpy(array).reshape(shape).float()
            remaining = remaining[byte_count:]
            continue
        embedding = name.endswith("embed.weight")
        scale_count = shape[-1] if embedding else shape[0]
        scale_bytes = scale_count * 2
        scales = torch.from_numpy(
            inflate.np.frombuffer(remaining[:scale_bytes], dtype="<f2").copy()
        ).float()
        remaining = remaining[scale_bytes:]
        codes, remaining = inflate.unpack_signed_int4(remaining, count)
        scale_shape = [1] * len(shape)
        scale_shape[-1 if embedding else 0] = scale_count
        restored[name] = codes.reshape(shape).float() * scales.reshape(scale_shape)
    assert not remaining
    return restored


@pytest.mark.parametrize("length", [1, 2, 3, 4095, 4096, 4097, 8193])
def test_cx2_model_section_transform_is_bijective(length: int) -> None:
    semantic = bytes((index * 73 + 19) & 0xFF for index in range(length))
    carrier = bytes((index * 11 + 7) & 0xFF for index in range(length + 1))
    hpac = bytes((index * 29 + 3) & 0xFF for index in range(length + 2))
    encoded = receiver.encode_cx2_model_sections(semantic, carrier, hpac)
    assert receiver.decode_cx2_model_sections(*encoded) == (
        semantic,
        carrier,
        hpac,
    )


def test_cx2_outer_selector_and_real_model_loader_bytes_round_trip() -> None:
    semantic = b"SD1M" + bytes(range(251)) * 7
    carrier = bytes(reversed(range(251))) * 5
    hpac = b"integer-hpac" * 137
    transformed = receiver.encode_cx2_model_sections(semantic, carrier, hpac)
    streams = tuple(
        brotli.compress(value, quality=quality)
        for value, quality in zip(transformed, (10, 11, 10), strict=True)
    )
    models = _split_pack(streams)
    tokens = b"\x01\x02\x03\x04"
    payload = receiver.pack_payload(
        models,
        tokens,
        token_codec="ans",
        model_codec="split_brotli_cx2",
    )
    parts = receiver.split_payload(payload)
    assert parts.model_codec == "split_brotli_cx2"
    assert parts.token_codec == "ans"
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    expected = struct.pack("<II", len(semantic), len(carrier))
    expected += semantic + carrier + hpac
    assert decoded.raw == expected
    assert decoded.codec == "split_brotli_cx2"


@pytest.mark.skipif(not BASE_ARCHIVE.is_file(), reason="PR130 custody is not mounted")
def test_legacy_q4_semantic_loader_is_tensor_identical() -> None:
    semantic, _, _, _ = _sections(BASE_ARCHIVE)
    template = inflate.SemanticTokenRenderer(96).state_dict()
    expected = _legacy_int4_reference(semantic, template)
    allocation, _, format_name = inflate.semantic_allocation(semantic, template)
    actual = inflate.unpack_semantic(semantic, template)
    assert format_name == "legacy_int4"
    assert set(allocation.values()) == {4}
    assert list(actual) == list(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.skipif(not SD1_ARCHIVE.is_file(), reason="SD1 custody is not mounted")
def test_real_sd1m_state_matches_the_research_parser_and_full_loader() -> None:
    semantic, carrier, _, _ = _sections(SD1_ARCHIVE)
    template = inflate.SemanticTokenRenderer(96).state_dict()
    reference = _load_module(
        "ddm_cx2_sd1_reference",
        REPO / "experiments" / "ddm_sd1_semantic_rd_curve.py",
    )
    expected, expected_allocation, expected_format = reference.unpack_semantic_state(
        semantic,
        template,
    )
    allocation, _, format_name = inflate.semantic_allocation(semantic, template)
    actual = inflate.unpack_semantic(semantic, template)
    assert format_name == expected_format == "sd1_mixed_v1"
    assert allocation == dict(expected_allocation)
    assert {
        name for name, bits in allocation.items() if bits == 3
    } == {
        "frame_embed.weight",
        "blocks.1.film.weight",
        "blocks.2.film.weight",
        "blocks.3.film.weight",
    }
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name

    raw = struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier
    semantic_model, basis, coefficients = inflate.unpack_semantic_pose(raw)
    for name in expected:
        assert torch.equal(semantic_model.state_dict()[name], expected[name]), name
    assert basis.shape == (12, 3, 24, 32)
    assert coefficients.shape == (600, 12)


@pytest.mark.skipif(not SD1_ARCHIVE.is_file(), reason="SD1 custody is not mounted")
def test_real_cx2_model_coordinates_have_the_measured_stream_lengths() -> None:
    semantic, carrier, hpac, _ = _sections(SD1_ARCHIVE)
    transformed = receiver.encode_cx2_model_sections(semantic, carrier, hpac)
    streams = tuple(
        brotli.compress(value, quality=quality)
        for value, quality in zip(transformed, (10, 11, 10), strict=True)
    )
    assert tuple(map(len, streams)) == (33_714, 23_058, 14_950)
    decoded = receiver.decode_models(
        _split_pack(streams),
        model_codec="split_brotli_cx2",
    )
    expected = struct.pack("<II", len(semantic), len(carrier))
    expected += semantic + carrier + hpac
    assert decoded.raw == expected


def test_mixed_semantic_headers_fail_closed() -> None:
    template = inflate.SemanticTokenRenderer(96).state_dict()
    with pytest.raises(ValueError, match="truncated mixed semantic header"):
        inflate.semantic_allocation(b"SD1M\x01", template)
    with pytest.raises(ValueError, match="unsupported mixed semantic header"):
        inflate.semantic_allocation(b"SD1M\x02\x10" + b"\x44" * 8, template)
    with pytest.raises(ValueError, match="invalid bit depth"):
        inflate.semantic_allocation(b"SD1M\x01\x10" + b"\x11" * 8, template)


def test_entrypoint_selects_brotli_for_cx2_wire_tag(tmp_path: Path) -> None:
    data_dir = tmp_path / "archive"
    data_dir.mkdir()
    model_word = receiver.MODEL_CODEC_MASK | 1
    (data_dir / "p").write_bytes(struct.pack("<I", model_word) + b"m")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "PR130_DEPENDENCY_SELECTION_ONLY": "1",
        }
    )
    result = subprocess.run(
        [
            str(TREE / "inflate.sh"),
            str(data_dir),
            str(tmp_path / "inflated"),
            str(tmp_path / "names.txt"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == (
        "PR130_DEPENDENCY_SELECTION "
        "model_codec=split_brotli_cx2 needs_brotli=1"
    )


def test_explicit_brotli_cli_provider_is_real_and_bijective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = shutil.which("brotli")
    if cli is None:
        pytest.skip("Brotli CLI is unavailable on this host")
    original = bytes((index * 97 + 31) & 0xFF for index in range(8193))
    encoded = brotli.compress(original, quality=10)
    monkeypatch.setattr(receiver, "brotli", None)
    monkeypatch.setenv("PR130_BROTLI_CLI", cli)
    assert receiver._decompress_brotli(encoded) == original


def test_token_checkpoint_is_atomic_and_bound_to_all_payload_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inflate, "N", 2)
    monkeypatch.setattr(inflate, "EVAL_H", 3)
    monkeypatch.setattr(inflate, "EVAL_W", 4)
    tokens = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
    cache = tmp_path / "tokens.npz"
    receipt = tmp_path / "tokens.json"

    class EmptyAnsDecoder:
        @staticmethod
        def is_empty() -> bool:
            return True

    finish_proof = inflate._finish_token_stage(EmptyAnsDecoder(), "ans")
    written = inflate.write_token_checkpoint(
        tokens,
        finish_proof=finish_proof,
        payload=b"archive-member",
        models_raw=b"model-sections",
        token_payload=b"ans-payload",
        token_codec="ans",
        cache_path=cache,
        receipt_path=receipt,
    )
    restored, resumed = inflate.load_token_checkpoint(
        payload=b"archive-member",
        models_raw=b"model-sections",
        token_payload=b"ans-payload",
        token_codec="ans",
        cache_path=cache,
        receipt_path=receipt,
    )
    assert torch.equal(restored, tokens)
    assert resumed["decoded_token_sha256"] == written["decoded_token_sha256"]
    assert resumed["resumed_from_cache"] is True
    with pytest.raises(ValueError, match="archive_member_sha256"):
        inflate.load_token_checkpoint(
            payload=b"different-archive-member",
            models_raw=b"model-sections",
            token_payload=b"ans-payload",
            token_codec="ans",
            cache_path=cache,
            receipt_path=receipt,
        )
