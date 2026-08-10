from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import ddm_tm1_token_model_lever as tm1


def test_stratified_frames_are_deterministic_and_not_prefix() -> None:
    first = tm1.stratified_frames()
    second = tm1.stratified_frames()
    assert np.array_equal(first, second)
    assert len(first) == 120
    assert len(np.unique(first)) == 120
    assert all(5 * index <= frame < 5 * (index + 1) for index, frame in enumerate(first))
    assert not np.array_equal(first, np.arange(120))


def test_scan_permutation_and_tiles_are_exact_partition() -> None:
    permutation = tm1.scan_permutation()
    assert len(permutation) == tm1.H * tm1.W
    assert np.array_equal(np.sort(permutation), np.arange(tm1.H * tm1.W))
    tiles = tm1.global_tile_contexts(permutation)
    assert set(np.unique(tiles)) == set(range(16))
    assert np.all(np.bincount(tiles, minlength=16) == (tm1.H // 4) * (tm1.W // 4))


def test_probability_table_matches_float64_then_float32_contract() -> None:
    codes = np.asarray([[0, 8, -8, 16, -16], [3, 3, 3, 3, 3]], dtype=np.int16)
    observed = tm1.probability_tables(codes)
    logits = codes.astype(np.float64) / 8
    logits -= logits.max(axis=1, keepdims=True)
    expected = np.exp(logits)
    expected /= expected.sum(axis=1, keepdims=True)
    expected = expected.astype(np.float32)
    assert observed.dtype == np.float32
    assert np.array_equal(observed, expected)


def test_temperature_fit_and_integer_application_reduce_constructed_nll() -> None:
    # Positive gradient means the base is overconfident; fitted scale must shrink.
    model = tm1.fit_temperature(np.asarray([20.0, 100.0], dtype=np.float64))
    assert model.scale_q8 < 256
    codes = np.asarray([[0, 32, -8, -8, -8]], dtype=np.int16)
    symbols = np.asarray([0], dtype=np.uint8)
    tiles = np.zeros(1, dtype=np.int16)
    before = tm1.ideal_bits(tm1.probability_tables(codes), symbols)
    after_codes = tm1.apply_candidate(model, codes, 0, tiles)
    after = tm1.ideal_bits(tm1.probability_tables(after_codes), symbols)
    assert after < before


def test_temperature_statistics_use_numpy_array_square() -> None:
    codes = np.asarray([[0, 8, -8, 16, -16]], dtype=np.int16)
    symbols = np.asarray([3], dtype=np.uint8)
    tables = tm1.probability_tables(codes)
    gradient, hessian = tm1._temperature_stats(codes, symbols, tables)
    assert np.isfinite(gradient)
    assert hessian > 0


def test_additive_fit_corrects_systematic_class_bias() -> None:
    counts = np.asarray([[90, 10, 0, 0, 0]], dtype=np.float64)
    predicted = np.asarray([[50, 50, 0, 0, 0]], dtype=np.float64)
    model = tm1.fit_additive("class_bias", counts, predicted)
    assert model.corrections.shape == (1, 5)
    assert model.corrections[0, 0] == 0
    assert model.corrections[0, 1] < 0


@pytest.mark.parametrize("name", tm1.CANDIDATES)
def test_sidecar_pack_roundtrip_counts_real_bytes(name: str) -> None:
    if name == "temperature":
        model = tm1.CandidateModel(name, scale_q8=241)
    elif name == "temporal_reversion":
        table = np.zeros((tm1.K, tm1.K), dtype=np.int8)
        table[1, 0] = 2
        model = tm1.CandidateModel(name, corrections=table)
    else:
        table = np.zeros((tm1.ADDITIVE_CONTEXTS[name], tm1.K), dtype=np.int8)
        table[:, 1] = -1
        model = tm1.CandidateModel(name, corrections=table)
    packed, report = tm1.pack_model(model)
    restored = tm1.unpack_model(packed, report["codec_order"])
    assert tm1.model_to_raw(restored) == tm1.model_to_raw(model)
    assert report["packed_bytes"] == len(packed)
    assert report["verified_exact"] is True


def test_temporal_reversion_uses_only_two_causal_prior_frames() -> None:
    table = np.zeros((tm1.K, tm1.K), dtype=np.int8)
    table[1, 0] = 3
    model = tm1.CandidateModel("temporal_reversion", corrections=table)
    codes = np.zeros((3, tm1.K), dtype=np.int16)
    previous_one = np.asarray([1, 0, 1], dtype=np.uint8)
    previous_two = np.asarray([0, 0, 1], dtype=np.uint8)
    corrected = tm1.apply_candidate(
        model,
        codes,
        2,
        np.zeros(3, dtype=np.int16),
        previous_one=previous_one,
        previous_two=previous_two,
    )
    assert corrected[0, 0] == 3
    assert not corrected[1].any()
    assert not corrected[2].any()


def test_append_and_split_sidecar_preserve_base_bytes() -> None:
    base = b"unchanged-pr130-models"
    model = tm1.CandidateModel("temperature", scale_q8=256)
    packed, _ = tm1.pack_model(model)
    candidate = tm1.append_sidecar(base, packed)
    restored_base, restored_sidecar = tm1.split_sidecar(candidate)
    assert restored_base == base
    assert restored_sidecar == packed


def test_synthetic_ans_candidate_roundtrip() -> None:
    codes = np.asarray(
        [[16, 0, -8, -16, -24], [0, 16, -8, -16, -24], [8, 0, 16, -8, -16]],
        dtype=np.int16,
    )
    symbols = np.asarray([0, 1, 2], dtype=np.int32)
    model = tm1.CandidateModel("class_bias", corrections=np.zeros((1, 5), dtype=np.int8))
    tables = tm1.probability_tables(tm1.apply_candidate(model, codes, 0, np.zeros(len(codes), dtype=np.int16)))
    family = tm1.constriction.stream.model.Categorical(perfect=False)
    encoder = tm1.constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(symbols, family, tables)
    blob = encoder.get_compressed().astype("<u4", copy=False).tobytes(order="C")
    decoder = tm1.constriction.stream.stack.AnsCoder(np.frombuffer(blob, dtype="<u4").astype(np.uint32, copy=False))
    observed = decoder.decode(family, tables)
    assert np.array_equal(observed, symbols)
    assert decoder.is_empty()


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    identity = {"fixed": True}
    state_path, _ = tm1.initialize_state(tmp_path, None, identity)
    assert state_path.exists()
    with pytest.raises(RuntimeError, match="requires --resume-from"):
        tm1.initialize_state(tmp_path, None, identity)
    resumed_path, state = tm1.initialize_state(tmp_path, state_path, identity)
    assert resumed_path == state_path
    assert state["input_identity"] == identity


def test_fresh_output_refuses_untracked_residue(tmp_path: Path) -> None:
    (tmp_path / "orphan.bin").write_bytes(b"not-a-checkpoint")
    with pytest.raises(RuntimeError, match="fresh TM1 output root must be empty"):
        tm1.initialize_state(tmp_path, None, {"fixed": True})
