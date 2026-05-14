# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    decode_hdm7_q_brotli_len_elided_fixture,
    encode_hdm3_q_brotli_split_fixture,
    encode_hdm4_q_brotli_split_fixture,
    encode_hdm6_q_brotli_tuned_fixture,
    encode_hdm7_q_brotli_len_elided_fixture,
    encode_hdm8_q_brotli_recipe_elided_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_pr101_schema_packer import encode_pr101_schema_split_fixture
from tac.hnerv_lowlevel_packer import read_packed_archive_view

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


def test_pr106_r2_pr101_runtime_accepts_hdm6_decoder_section() -> None:
    codec = _load_codec(
        REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py",
        "pr106_r2_pr101_codec_hdm6_adapter",
    )
    raw = _synthetic_decoder_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(legacy_decoder)
    hdm6_decoder, _stats = encode_hdm6_q_brotli_tuned_fixture(parsed)

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    hdm6_sd = codec.decode_packed_decoder(hdm6_decoder)

    assert set(hdm6_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(hdm6_sd[name], legacy_sd[name]), name

    bad_hdm6 = b"HDM6" + b"\x02" + b"\x00" * (3 * 4) + bytes(4 * len(PACKED_STATE_SCHEMA))
    with pytest.raises(ValueError, match="unsupported HDM6 recipe id"):
        codec.decode_packed_decoder(bad_hdm6)


def test_pr106_r2_pr101_runtime_accepts_hdm7_decoder_section() -> None:
    codec = _load_codec(
        REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py",
        "pr106_r2_pr101_codec_hdm7_adapter",
    )
    raw = _synthetic_decoder_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(legacy_decoder)
    hdm7_decoder, _stats = encode_hdm7_q_brotli_len_elided_fixture(parsed)

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    hdm7_sd = codec.decode_packed_decoder(hdm7_decoder)

    assert set(hdm7_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(hdm7_sd[name], legacy_sd[name]), name

    bad_hdm7 = b"HDM7" + b"\x02" + b"\x00" * (3 * 3) + bytes(4 * len(PACKED_STATE_SCHEMA))
    with pytest.raises(ValueError, match="unsupported HDM7 recipe id"):
        codec.decode_packed_decoder(bad_hdm7)

    _payload, stats = encode_hdm7_q_brotli_len_elided_fixture(parsed)
    scale_len = stats["raw_scale_bytes"]
    final_len = stats["derived_final_chunk_bytes"]
    no_final_chunk = _payload[: len(_payload) - scale_len - final_len] + _payload[-scale_len:]
    with pytest.raises(
        ValueError,
        match="HDM7 derived final q stream chunk length must be positive",
    ):
        codec.decode_packed_decoder(no_final_chunk)


def test_pr106_r2_pr101_runtime_accepts_hdm8_decoder_section() -> None:
    codec = _load_codec(
        REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py",
        "pr106_r2_pr101_codec_hdm8_adapter",
    )
    hdm7_archive = (
        REPO
        / "experiments/results/pr106_r2_hdm6_hlm2_hdm7_candidate_20260514_codex/"
        "pr106_r2_hdm6_hlm2_xmember_hdm7_archive_candidate.zip"
    )
    if not hdm7_archive.exists():
        pytest.skip("HDM7 exact-CUDA candidate artifact is not present in this checkout")
    hdm7_view = read_packed_archive_view(hdm7_archive)
    parsed = decode_hdm7_q_brotli_len_elided_fixture(hdm7_view.packed.decoder_packed_brotli)
    raw = parsed.to_raw()
    legacy_decoder = brotli.compress(raw, quality=5)
    hdm8_decoder, stats = encode_hdm8_q_brotli_recipe_elided_fixture(parsed)

    legacy_sd = codec.decode_packed_decoder(legacy_decoder)
    hdm8_sd = codec.decode_packed_decoder(hdm8_decoder)

    assert stats["header_bytes"] == 4
    assert stats["elided_len24_bytes"] == 9
    assert stats["elided_recipe_id_bytes"] == 1
    assert stats["runtime_fixed_chunk_lengths"] == [130887, 2769, 4397, 31805]
    assert set(hdm8_sd) == set(legacy_sd)
    for name in legacy_sd:
        assert torch.equal(hdm8_sd[name], legacy_sd[name]), name

    no_final_chunk = hdm8_decoder[: -stats["raw_scale_bytes"] - stats["derived_final_chunk_bytes"]] + hdm8_decoder[
        -stats["raw_scale_bytes"] :
    ]
    with pytest.raises(
        ValueError,
        match="HDM8 derived final q stream chunk length must be positive",
    ):
        codec.decode_packed_decoder(no_final_chunk)
