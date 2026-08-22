from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from experiments import ddm_ws0_worldsheet_grammar_price as ws0


def synthetic_labels() -> np.ndarray:
    labels = np.zeros((3, 4, ws0.WIDTH), dtype=np.uint8)
    labels[:, :, 20:80] = 2
    labels[:, :, 80:120] = 0
    labels[:, :, 120:180] = 1
    labels[:, :, 180:260] = 0
    labels[:, :, 260:] = 4
    labels[1, :, 21:81] = 2
    labels[1, :, 121:181] = 1
    labels[2, :, 22:82] = 2
    labels[2, :, 122:182] = 1
    labels[:, 1, 300:320] = 3
    labels[:, 2, 302:322] = 3
    return labels


def expand_to_registered_shape(labels: np.ndarray) -> tuple[ws0.FrameGrammar, ...]:
    frames = []
    for source in labels:
        tiled = np.repeat(source[-1:, :], ws0.HEIGHT, axis=0)
        tiled[: source.shape[0]] = source
        frames.append(ws0.frame_from_labels(tiled))
    return tuple(frames)


def local_races(semantics: ws0.CandidateSemantics) -> dict[str, ws0.StreamRace]:
    races = {}
    for name, records in semantics.stream_records.items():
        canonical = ws0.pack_records(records)
        coded = ws0.brotli.compress(canonical, quality=11)
        races[name] = ws0.StreamRace(
            name=name,
            records_sha256=ws0.sha256_bytes(canonical),
            canonical_raw_bytes=len(canonical),
            payloads={"brotli-q11": coded},
            winner="brotli-q11",
        )
    return races


@pytest.mark.parametrize("mode", ws0.SELECTION_MODES)
def test_receiver_roundtrips_every_selection_mode(mode: str) -> None:
    frames = expand_to_registered_shape(synthetic_labels())
    frames = tuple(frames[index % len(frames)] for index in range(ws0.N_PAIRS))
    table = ws0.induced_rank_table(frames)
    semantics = ws0.build_candidate_semantics(frames, table, mode=mode)
    payload = ws0.build_envelope(table=table, races=local_races(semantics))
    decoded = list(ws0.iter_decode_frames(payload))
    assert len(decoded) == ws0.N_PAIRS
    assert all(
        np.array_equal(got, ws0.render_frame(expected))
        for got, expected in zip(decoded, frames, strict=True)
    )


def test_induced_rank_table_is_counted_and_invertible() -> None:
    frames = expand_to_registered_shape(synthetic_labels())
    table = ws0.induced_rank_table(frames)
    payload = ws0.rank_table_bytes(table)
    assert len(payload) == ws0.N_CLASSES * (ws0.N_CLASSES - 1)
    assert ws0.rank_table_from_bytes(payload) == table


def test_quantization_respects_shift_and_actual_error_caps() -> None:
    frames = expand_to_registered_shape(synthetic_labels())
    result = ws0.quantize_frames(frames, q_step=8, error_cap=100)
    actual = sum(
        int(np.count_nonzero(ws0.render_frame(original) != ws0.render_frame(quantized)))
        for original, quantized in zip(frames, result.frames, strict=True)
    )
    assert result.selected_shift_upper_bound <= 100
    assert actual <= result.selected_shift_upper_bound
    assert result.selected_boundaries > 0


def test_two_bit_padding_fails_closed() -> None:
    payload = bytearray(ws0.pack_two_bit([1]))
    payload[-1] |= 0b100
    with pytest.raises(ws0.WorldsheetError, match="padding"):
        ws0.unpack_two_bit(bytes(payload), 1)


def test_corrupt_stream_roster_fails_closed() -> None:
    frames = expand_to_registered_shape(synthetic_labels())
    frames = tuple(frames[index % len(frames)] for index in range(ws0.N_PAIRS))
    table = ws0.induced_rank_table(frames)
    semantics = ws0.build_candidate_semantics(frames, table, mode="minabs")
    races = local_races(semantics)
    races.pop(ws0.coord_stream_name(0))
    payload = ws0.build_envelope(table=table, races=races)
    with pytest.raises(ws0.WorldsheetError, match="roster"):
        ws0.parse_envelope(payload)


def test_frame_checkpoint_roundtrip_without_pickle(tmp_path) -> None:
    frame = expand_to_registered_shape(synthetic_labels())[0]
    path = tmp_path / "frame.npz"
    path.write_bytes(ws0.frame_to_npz_bytes(frame))
    restored = ws0.frame_from_npz(path)
    for field in dataclasses.fields(ws0.FrameGrammar):
        assert np.array_equal(getattr(frame, field.name), getattr(restored, field.name))


def test_real_three_coder_race_retains_semantic_identity() -> None:
    records = (b"abc", b"abd", b"", b"abc")
    race = ws0._race_records_task(("fixture", records))
    assert set(race.payloads) == set(ws0.CODEC_IDS)
    assert race.winner in race.payloads
    assert race.canonical_raw_bytes == len(ws0.pack_records(records))
