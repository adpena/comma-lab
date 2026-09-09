from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments import ddm_gf3_time_conditioned_factorized_field_pricing as gf3


def exact_global_translation_optimum(frames: np.ndarray, radius: int) -> int:
    """Brute-force the declared family on the common interior for tiny tests."""

    height, width = frames.shape[1:]
    translations = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    interior_y = slice(radius, height - radius)
    interior_x = slice(radius, width - radius)
    best = frames.size
    for offsets in itertools.product(translations, repeat=frames.shape[0]):
        aligned = []
        for frame, (dy, dx) in zip(frames, offsets, strict=True):
            aligned.append(
                frame[
                    radius + dy : height - radius + dy,
                    radius + dx : width - radius + dx,
                ]
            )
        stack = np.stack(aligned)
        template = gf3.plurality_field(stack)
        error = int(np.count_nonzero(stack != template[None, :, :]))
        best = min(best, error)
    assert interior_y.start == radius and interior_x.start == radius
    return best


@pytest.mark.parametrize("seed", range(5))
def test_relaxed_bound_never_exceeds_exact_global_translation_optimum(seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, gf3.NUM_CLASSES, size=(3, 5, 6), dtype=np.uint8)
    presence = gf3.build_presence_cache(frames, radius=1)
    _, _, lower_bound = gf3.relaxed_translation_plurality_lower_bound(presence, gop_length=3)
    exact = exact_global_translation_optimum(frames, radius=1)
    assert lower_bound <= exact


def test_radius_zero_bound_equals_exact_plurality_error() -> None:
    frames = np.array(
        [
            [[0, 1], [2, 3]],
            [[0, 2], [2, 4]],
            [[1, 2], [2, 4]],
        ],
        dtype=np.uint8,
    )
    presence = gf3.build_presence_cache(frames, radius=0)
    maps, totals, lower_bound = gf3.relaxed_translation_plurality_lower_bound(presence, gop_length=3)
    template = gf3.plurality_field(frames)
    expected = int(np.count_nonzero(frames != template[None, :, :]))
    assert maps.shape == (1, 2, 2)
    assert totals.tolist() == [expected]
    assert lower_bound == expected


def test_window_presence_excludes_border_and_records_reachable_classes() -> None:
    frame = np.zeros((5, 5), dtype=np.uint8)
    frame[0, 0] = 4
    frame[2, 2] = 3
    presence = gf3.window_class_presence(frame, radius=1)
    assert presence.shape == (gf3.NUM_CLASSES, 3, 3)
    assert presence[3, 1, 1]
    assert not presence[4, 1, 1]
    assert presence[4, 0, 0]


def test_gop_partitioning_and_lowest_class_tie() -> None:
    frames = np.array(
        [
            [[0, 1]],
            [[1, 0]],
            [[2, 3]],
            [[3, 2]],
        ],
        dtype=np.uint8,
    )
    presence = gf3.build_presence_cache(frames, radius=0)
    maps, totals, lower_bound = gf3.relaxed_translation_plurality_lower_bound(presence, gop_length=2)
    assert maps.shape == (2, 1, 2)
    assert totals.tolist() == [2, 2]
    assert lower_bound == 4
    assert gf3.plurality_field(frames[:2]).tolist() == [[0, 0]]


def test_observed_member_is_achievable_and_nonincreasing() -> None:
    rng = np.random.default_rng(31)
    frames = rng.integers(0, gf3.NUM_CLASSES, size=(4, 5, 6), dtype=np.uint8)
    fields, offsets, decoded, zero_mismatches, observed_mismatches = gf3.observed_member(frames, gop_length=2, radius=1)
    assert np.array_equal(decoded, gf3.render_gop_fields(fields, offsets, 2))
    assert observed_mismatches == np.count_nonzero(decoded != frames)
    assert observed_mismatches <= zero_mismatches


def test_pack_unpack_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gf3, "N_PAIRS", 4)
    monkeypatch.setattr(gf3, "HEIGHT", 2)
    monkeypatch.setattr(gf3, "WIDTH", 3)
    monkeypatch.setattr(gf3, "GOP_LENGTHS", (2,))
    fields = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) % gf3.NUM_CLASSES
    offsets = np.array([[0, 0], [1, -1], [0, 1], [-1, 0]], dtype=np.int8)
    payload = gf3.pack_packet(fields, offsets, 2)
    length, decoded_fields, decoded_offsets = gf3.unpack_packet(payload)
    assert length == 2
    assert np.array_equal(decoded_fields, fields)
    assert np.array_equal(decoded_offsets, offsets)


def test_unpack_rejects_trailing_packet_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gf3, "N_PAIRS", 2)
    monkeypatch.setattr(gf3, "HEIGHT", 2)
    monkeypatch.setattr(gf3, "WIDTH", 2)
    monkeypatch.setattr(gf3, "GOP_LENGTHS", (2,))
    fields = np.zeros((1, 2, 2), dtype=np.uint8)
    offsets = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(gf3.GF3Error, match="length"):
        gf3.unpack_packet(gf3.pack_packet(fields, offsets, 2) + b"unexpected")


def test_canonical_npz_is_byte_deterministic() -> None:
    arrays = {
        "z": np.arange(8, dtype=np.uint8),
        "a": np.arange(6, dtype=np.int16).reshape(2, 3),
    }
    assert gf3.canonical_npz_bytes(arrays) == gf3.canonical_npz_bytes(arrays)


def test_coder_race_retains_every_payload_and_repeat(tmp_path) -> None:
    raw = b"time-conditioned-field" * 64
    result = gf3.coder_race("fixture", raw, tmp_path)
    assert gf3.selected_coder_parseback(result) == raw
    assert set(result["coders"]) == {"brotli_q11", "zlib_9", "lzma2_extreme"}
    for row in result["coders"].values():
        assert row["byte_identical"]
        assert row["parseback_exact"]
        assert gf3.fact_matches(row["primary"])
        assert gf3.fact_matches(row["repeat"])


def test_atomic_retention_refuses_changed_payload(tmp_path) -> None:
    path = tmp_path / "retained.bin"
    gf3.atomic_bytes_once(path, b"first")
    with pytest.raises(gf3.GF3Error, match="refusing to overwrite"):
        gf3.atomic_bytes_once(path, b"second")


def test_domain_residual_hits_exact_remaining_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gf3, "N_PAIRS", 2)
    monkeypatch.setattr(gf3, "HEIGHT", 2)
    monkeypatch.setattr(gf3, "WIDTH", 3)
    target = np.array([[[0, 1, 2], [3, 4, 0]], [[1, 2, 3], [4, 0, 1]]], dtype=np.uint8)
    predicted = np.zeros_like(target)
    observed = int(np.count_nonzero(target != predicted))
    residual, decoded, corrections, remaining = gf3.domain_residual_to_target(target, predicted, remaining_target=2)
    assert residual.shape == (2, 3, 2)
    assert corrections == observed - 2
    assert remaining == 2
    assert np.count_nonzero(decoded != target) == 2


def test_domain_residual_is_noop_when_observed_already_meets_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gf3, "N_PAIRS", 1)
    monkeypatch.setattr(gf3, "HEIGHT", 2)
    monkeypatch.setattr(gf3, "WIDTH", 2)
    target = np.zeros((1, 2, 2), dtype=np.uint8)
    residual, decoded, corrections, remaining = gf3.domain_residual_to_target(target, target.copy(), remaining_target=2)
    assert not residual.any()
    assert np.array_equal(decoded, target)
    assert corrections == 0
    assert remaining == 0


def test_invalid_gop_is_refused() -> None:
    presence = np.ones((5, gf3.NUM_CLASSES, 2, 2), dtype=np.bool_)
    with pytest.raises(gf3.GF3Error, match="divide"):
        gf3.relaxed_translation_plurality_lower_bound(presence, gop_length=2)
