# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_r7_token_coder import factor_mode_delta
from experiments.ddm_vae1_ar_prior_probe import (
    HEADER,
    MAX_CHANNELS,
    DDMVAE1PriorError,
    _decode_counted_ar_prior,
    _formulation_config,
    _formulation_config_sha256,
    _normalize_count_rows,
    _repo_evidence_reference,
    _resolve_preserved_stage,
    _resolve_repo_evidence_reference,
    decode_counted_ar_prior,
    encode_counted_ar_prior,
    fit_counted_ar_prior,
    learned_prior_accounting,
)


def _fixture() -> np.ndarray:
    pairs, height, width, channels = 9, 3, 4, 2
    value = np.zeros((pairs, height, width, channels), dtype=np.uint8)
    for pair in range(pairs):
        value[pair, ..., 0] = (pair + np.arange(width, dtype=np.uint8)[None, :]) % 7
        value[pair, ..., 1] = (2 * pair + np.arange(height, dtype=np.uint8)[:, None]) % 7
    return value


def test_counted_ar_prior_is_deterministic_exact_and_fully_accounted() -> None:
    source = _fixture()
    first = encode_counted_ar_prior(source, levels=7)
    second = encode_counted_ar_prior(source, levels=7)
    assert first == second
    assert np.array_equal(decode_counted_ar_prior(first), source)
    accounting = learned_prior_accounting(first)
    assert accounting.framed_bytes == len(first)
    assert (
        accounting.header_bytes
        + accounting.base_bytes
        + accounting.model_bytes
        + accounting.residual_bytes
        == len(first)
    )
    assert accounting.raw_model_bytes == 2 * 7 * 7 * 7 * 2
    assert accounting.sha256 == hashlib.sha256(first).hexdigest()


def test_fitted_prior_rows_are_positive_normalized_and_context_conditioned() -> None:
    source = _fixture()
    frequencies = fit_counted_ar_prior(source, levels=7)
    assert frequencies.dtype == np.dtype("<u2")
    assert frequencies.shape == (2, 7, 7, 7)
    assert np.all(frequencies > 0)
    assert np.all(frequencies.astype(np.uint32).sum(axis=-1) == 1 << 15)
    assert not np.array_equal(frequencies[0, 0, 0], frequencies[0, 0, 1])


def test_frequency_normalization_keeps_rare_symbols_positive_for_large_rows() -> None:
    counts = np.array([[[[100_000, 0], [70_000, 30_000]]]], dtype=np.uint32)
    frequencies = _normalize_count_rows(counts)
    assert np.all(frequencies > 0)
    assert np.all(frequencies.astype(np.uint32).sum(axis=-1) == 1 << 15)


@pytest.mark.parametrize("mutation", ("truncate", "trailer", "bitflip"))
def test_corruption_and_inert_bytes_are_refused(mutation: str) -> None:
    frame = bytearray(encode_counted_ar_prior(_fixture(), levels=7))
    if mutation == "truncate":
        changed = bytes(frame[:-1])
    elif mutation == "trailer":
        changed = bytes(frame) + b"\0"
    else:
        frame[-1] ^= 1
        changed = bytes(frame)
    with pytest.raises(DDMVAE1PriorError):
        decode_counted_ar_prior(changed)


def test_valid_but_changed_counted_model_is_not_inert() -> None:
    source = _fixture()
    frame = encode_counted_ar_prior(source, levels=7)
    fields = list(HEADER.unpack_from(frame))
    base_length, model_length = int(fields[9]), int(fields[10])
    base_end = HEADER.size + base_length
    model_end = base_end + model_length
    model = bytearray(zlib.decompress(frame[base_end:model_end]))
    base, delta = factor_mode_delta(source, 7)
    used: set[tuple[int, int, int]] = set()
    for channel in range(source.shape[-1]):
        previous = np.zeros(source.shape[1:3], dtype=np.uint8)
        for pair_index in range(source.shape[0]):
            for row, column in np.ndindex(source.shape[1:3]):
                used.add(
                    (
                        channel,
                        int(base[row, column, channel]),
                        int(previous[row, column]),
                    )
                )
            previous = delta[pair_index, ..., channel]
    unused = next(
        context
        for context in np.ndindex(source.shape[-1], 7, 7)
        if context not in used
    )
    row_index = ((unused[0] * 7 + unused[1]) * 7 + unused[2]) * 7
    offset = row_index * 2
    first = int.from_bytes(model[offset : offset + 2], "little")
    second = int.from_bytes(model[offset + 2 : offset + 4], "little")
    assert first > 1
    model[offset : offset + 2] = (first - 1).to_bytes(2, "little")
    model[offset + 2 : offset + 4] = (second + 1).to_bytes(2, "little")
    changed_model = zlib.compress(bytes(model), level=9)
    fields[10] = len(changed_model)
    changed = (
        HEADER.pack(*fields)
        + frame[HEADER.size:base_end]
        + changed_model
        + frame[model_end:]
    )
    assert np.array_equal(
        _decode_counted_ar_prior(changed, canonical=False),
        source,
    )
    with pytest.raises(DDMVAE1PriorError):
        decode_counted_ar_prior(changed)


def test_header_layout_and_input_contract_fail_closed() -> None:
    assert HEADER.size == struct.calcsize("<4sBBBB4HIII32s")
    source = _fixture()
    with pytest.raises(DDMVAE1PriorError):
        encode_counted_ar_prior(source.astype(np.int16), levels=7)
    with pytest.raises(DDMVAE1PriorError):
        encode_counted_ar_prior(source[0], levels=7)
    with pytest.raises(DDMVAE1PriorError):
        encode_counted_ar_prior(source, levels=17)
    malicious_fields = list(HEADER.unpack_from(encode_counted_ar_prior(source, levels=7)))
    malicious_fields[5:9] = [1, 1, 1, MAX_CHANNELS + 1]
    malicious_fields[9:12] = [1, 1, 1]
    malicious_fields[12] = b"\0" * 32
    malicious = HEADER.pack(*malicious_fields) + b"\0\0\0"
    with pytest.raises(DDMVAE1PriorError, match="levels/shape"):
        decode_counted_ar_prior(malicious)


def test_formulation_config_closes_the_measured_negative_scope() -> None:
    config = _formulation_config()
    assert config["context"]["t0_previous_delta"] == 0
    assert config["base"]["stored_global_mode"].startswith("lowest-symbol")
    assert config["model"]["compression_level"] == 9
    assert config["range_coder"]["state_bits"] == 32
    assert len(_formulation_config_sha256(config)) == 64


def test_stage_evidence_references_are_repo_relative_and_reject_escape(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    evidence = repo / ".omx/research"
    stage = evidence / "probe/stage.bin"
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"stage")
    reference = _repo_evidence_reference(
        stage,
        repo=repo,
        evidence_root=evidence,
    )
    assert reference == ".omx/research/probe/stage.bin"
    assert (
        _resolve_repo_evidence_reference(
            reference,
            repo=repo,
            evidence_root=evidence,
        )
        == stage
    )
    with pytest.raises(DDMVAE1PriorError):
        _resolve_repo_evidence_reference(
            str(stage),
            repo=repo,
            evidence_root=evidence,
        )
    with pytest.raises(DDMVAE1PriorError):
        _resolve_repo_evidence_reference(
            "../escaped.bin",
            repo=repo,
            evidence_root=evidence,
        )


def test_missing_local_stage_uses_only_approved_ssd_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence = repo / ".omx/research"
    evidence.mkdir(parents=True)
    cold_root = tmp_path / "ssd/pact"
    cold_stage = cold_root / "probe/stage.bin"
    cold_stage.parent.mkdir(parents=True)
    cold_stage.write_bytes(b"cold-stage")
    stage = {
        "path": ".omx/research/probe/stage.bin",
        "cold_store_path": str(cold_stage),
    }
    assert (
        _resolve_preserved_stage(
            stage,
            repo=repo,
            evidence_root=evidence,
            cold_store_roots=(cold_root,),
        )
        == cold_stage
    )

    escaped_stage = tmp_path / "outside/stage.bin"
    escaped_stage.parent.mkdir(parents=True)
    escaped_stage.write_bytes(b"escaped")
    stage["cold_store_path"] = str(escaped_stage)
    with pytest.raises(DDMVAE1PriorError, match="escaped approved SSD roots"):
        _resolve_preserved_stage(
            stage,
            repo=repo,
            evidence_root=evidence,
            cold_store_roots=(cold_root,),
        )
