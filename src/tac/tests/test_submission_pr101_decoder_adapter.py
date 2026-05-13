from __future__ import annotations

import importlib.util
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    encode_hdm3_q_brotli_split_fixture,
    encode_hdm4_q_brotli_split_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_pr101_schema_packer import encode_pr101_schema_split_fixture

REPO = Path(__file__).resolve().parents[3]


def _load_codec(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_decoder_raw() -> bytes:
    q_bytes = sum(int(np.prod(shape)) for _, shape in PACKED_STATE_SCHEMA)
    q_stream = (np.arange(q_bytes, dtype=np.uint32) % 255).astype(np.uint8)
    scales = np.asarray(
        [1.0 + index / 1000.0 for index, _ in enumerate(PACKED_STATE_SCHEMA)],
        dtype="<f4",
    )
    return q_stream.tobytes() + scales.tobytes()


@pytest.mark.parametrize(
    ("module_name", "codec_path"),
    [
        ("pr106_latent_sidecar_codec_pr101_adapter", "submissions/pr106_latent_sidecar/src/codec.py"),
        ("apogee_v2_codec_pr101_adapter", "submissions/apogee_v2/src/codec.py"),
    ],
)
def test_submission_decoder_adapter_accepts_legacy_and_pr101_schema(
    module_name: str,
    codec_path: str,
) -> None:
    codec = _load_codec(REPO / codec_path, module_name)
    raw = _synthetic_decoder_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    schema_decoder, _stats = encode_pr101_schema_split_fixture(
        parse_packed_decoder_brotli(legacy_decoder)
    )

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    schema_sd = codec.decode_packed_decoder(schema_decoder)

    assert set(schema_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(schema_sd[name], legacy_sd[name]), name

    with pytest.raises(ValueError, match="neither legacy PR106 Brotli nor PR101 schema-split"):
        codec.decode_packed_decoder(b"not-a-decoder")
    with pytest.raises(ValueError, match="bad packed decoder payload"):
        codec.decode_packed_decoder(brotli.compress(b"bad raw", quality=5))


def test_pr106_r2_pr101_runtime_accepts_hdm3_decoder_section() -> None:
    codec = _load_codec(
        REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py",
        "pr106_r2_pr101_codec_hdm3_adapter",
    )
    raw = _synthetic_decoder_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(legacy_decoder)
    hdm3_decoder, _stats = encode_hdm3_q_brotli_split_fixture(parsed)

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    hdm3_sd = codec.decode_packed_decoder(hdm3_decoder)

    assert set(hdm3_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(hdm3_sd[name], legacy_sd[name]), name

    bad_hdm3 = (
        b"HDM3"
        + len(brotli.compress(b"x")).to_bytes(3, "little")
        + brotli.compress(b"x")
        + bytes(4 * len(PACKED_STATE_SCHEMA))
    )
    with pytest.raises(ValueError, match="HDM3 q stream length mismatch"):
        codec.decode_packed_decoder(bad_hdm3)


def test_pr106_r2_pr101_runtime_accepts_hdm4_decoder_section() -> None:
    codec = _load_codec(
        REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py",
        "pr106_r2_pr101_codec_hdm4_adapter",
    )
    raw = _synthetic_decoder_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(legacy_decoder)
    hdm4_decoder, _stats = encode_hdm4_q_brotli_split_fixture(parsed)

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    hdm4_sd = codec.decode_packed_decoder(hdm4_decoder)

    assert set(hdm4_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(hdm4_sd[name], legacy_sd[name]), name

    bad_hdm4 = b"HDM4" + b"\x02" + b"\x00" * (3 * 4) + bytes(4 * len(PACKED_STATE_SCHEMA))
    with pytest.raises(ValueError, match="unsupported HDM4 recipe id"):
        codec.decode_packed_decoder(bad_hdm4)
