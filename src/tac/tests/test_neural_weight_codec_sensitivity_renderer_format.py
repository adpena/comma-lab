# SPDX-License-Identifier: MIT
"""Focused tests for the NWCS renderer container format."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from tac.neural_weight_codec_sensitivity import (
    NWCS_RENDERER_MAGIC,
    NWCSRendererTensorEntry,
    SensitivityAwareCodecConfig,
    SensitivityAwareWeightCodec,
    encode_with_variable_codebook,
    export_nwcs_renderer_container,
    is_nwcs_renderer_container,
    load_nwcs_renderer_container,
)

_LEN = struct.Struct("<q")


def _sample_container_bytes(tmp_path: Path | None = None) -> bytes:
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    entries = [
        NWCSRendererTensorEntry.from_tensor_blob(
            "z.weight",
            tensor,
            b"encoded-z",
            block_size=4,
            codebook_sizes=[4, 16],
            original_dtype="float16",
        ),
        {
            "name": "a.bias",
            "shape": [3],
            "dtype": "float32",
            "original_dtype": "float32",
            "block_metadata": {
                "block_size": 4,
                "num_blocks": 0,
                "tail_elements": 3,
            },
            "blob": b"abc",
        },
    ]
    output_path = tmp_path / "renderer.bin" if tmp_path is not None else None
    return export_nwcs_renderer_container(
        entries,
        codec_checkpoint_blob=b"codec-checkpoint",
        output_path=output_path,
        metadata={"source": "unit-test"},
    )


def _replace_header(raw: bytes, mutate) -> bytes:
    offset = len(NWCS_RENDERER_MAGIC)
    header_len = _LEN.unpack_from(raw, offset)[0]
    header_start = offset + _LEN.size
    header_end = header_start + header_len
    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    mutate(header)
    new_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        raw[:offset]
        + _LEN.pack(len(new_header))
        + new_header
        + raw[header_end:]
    )


def test_export_load_roundtrip_on_tiny_synthetic_tensors_and_blobs(tmp_path: Path):
    raw = _sample_container_bytes(tmp_path)
    out_path = tmp_path / "renderer.bin"

    assert raw == out_path.read_bytes()
    assert raw.startswith(NWCS_RENDERER_MAGIC)

    loaded = load_nwcs_renderer_container(raw)
    assert loaded.codec_checkpoint_blob == b"codec-checkpoint"
    assert loaded.header["metadata"] == {"source": "unit-test"}
    assert loaded.header["tensor_count"] == 2
    assert [entry.name for entry in loaded.tensors] == ["a.bias", "z.weight"]
    assert loaded.tensor_blobs() == {
        "a.bias": b"abc",
        "z.weight": b"encoded-z",
    }
    z_entry = loaded.tensors[1]
    assert z_entry.shape == (2, 3)
    assert z_entry.dtype == "float16"
    assert z_entry.original_dtype == "float16"
    assert z_entry.block_metadata["block_size"] == 4
    assert z_entry.block_metadata["num_blocks"] == 1
    assert z_entry.block_metadata["tail_elements"] == 2


def test_load_rejects_bad_magic():
    raw = _sample_container_bytes()
    bad = b"BADMAGIC" + raw[len(NWCS_RENDERER_MAGIC):]

    with pytest.raises(ValueError, match="bad magic"):
        load_nwcs_renderer_container(bad)


def test_load_rejects_duplicate_tensor_names():
    raw = _sample_container_bytes()

    def duplicate_first_name(header: dict) -> None:
        header["tensors"][1]["name"] = header["tensors"][0]["name"]

    bad = _replace_header(raw, duplicate_first_name)

    with pytest.raises(ValueError, match="duplicate tensor name"):
        load_nwcs_renderer_container(bad)


def test_load_rejects_truncated_tensor_blob():
    raw = _sample_container_bytes()
    bad = raw[:-1]

    with pytest.raises(ValueError, match="truncated"):
        load_nwcs_renderer_container(bad)


def test_load_rejects_negative_stream_length():
    raw = _sample_container_bytes()
    offset = len(NWCS_RENDERER_MAGIC)
    header_len = _LEN.unpack_from(raw, offset)[0]
    codec_len_offset = offset + _LEN.size + header_len
    bad = raw[:codec_len_offset] + _LEN.pack(-1) + raw[codec_len_offset + _LEN.size:]

    with pytest.raises(ValueError, match="negative codec checkpoint length"):
        load_nwcs_renderer_container(bad)


def test_magic_detection_accepts_bytes_and_paths(tmp_path: Path):
    raw = _sample_container_bytes(tmp_path)
    path = tmp_path / "renderer.bin"

    assert is_nwcs_renderer_container(raw)
    assert is_nwcs_renderer_container(path)
    assert not is_nwcs_renderer_container(b"NWCS1")
    assert not is_nwcs_renderer_container(b"not-nwcs")
    assert not is_nwcs_renderer_container(tmp_path / "missing.bin")


def test_renderer_export_dispatch_loads_nwcs_tensor_only_container(tmp_path: Path):
    from tac.renderer_export import (
        detect_checkpoint_type,
        load_nwcs_sensitivity_compressed_checkpoint,
    )

    cfg = SensitivityAwareCodecConfig(block_size=4, latent_dim=4, hidden=8, codebook_sizes=[4])
    torch.manual_seed(123)
    codec = SensitivityAwareWeightCodec(cfg)
    tensor = torch.randn(2, 4)
    sensitivity = torch.ones(tensor.numel() // cfg.block_size)
    blob = encode_with_variable_codebook(codec, tensor, sensitivity)

    codec_path = tmp_path / "nwcs_codec.pt"
    torch.save(
        {
            "codec_state_dict": codec.state_dict(),
            "config": cfg.__dict__,
        },
        codec_path,
    )
    bin_path = tmp_path / "renderer_nwcs.bin"
    export_nwcs_renderer_container(
        [
            NWCSRendererTensorEntry.from_tensor_blob(
                "layer.weight",
                tensor,
                blob,
                block_size=cfg.block_size,
                codebook_sizes=cfg.codebook_sizes,
            )
        ],
        codec_checkpoint_blob=codec_path,
        output_path=bin_path,
        metadata={"config": {"tensor_only": True}},
    )

    assert detect_checkpoint_type(bin_path) == "neural_weight_compression_sensitivity_v1"
    model = load_nwcs_sensitivity_compressed_checkpoint(bin_path, device="cpu")
    state = model._nwcs_state_dict
    assert state["layer.weight"].shape == tensor.shape
    assert torch.isfinite(state["layer.weight"]).all()
