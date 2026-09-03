from __future__ import annotations

import hashlib

import numpy as np

from experiments import ddm_gf2_static_dynamic_generator_form as gf2


def test_translation_roundtrip_slices_cover_expected_region() -> None:
    template = np.arange(gf2.HEIGHT * gf2.WIDTH, dtype=np.uint32).reshape(
        gf2.HEIGHT, gf2.WIDTH
    ) % gf2.NUM_CLASSES
    template = template.astype(np.uint8)
    shifted = gf2.render_translation(template, 3, -4)
    sy, sx, ty, tx = gf2.translation_slices(3, -4)
    assert np.array_equal(shifted[ty, tx], template[sy, sx])
    assert np.all(shifted[:3] == gf2.FILL_CLASS)
    assert np.all(shifted[:, -4:] == gf2.FILL_CLASS)


def test_ordered_translations_prefers_zero_and_is_complete() -> None:
    rows = gf2.ordered_translations(2)
    assert rows[0] == (0, 0)
    assert len(rows) == 25
    assert len(set(rows)) == 25


def test_packed_static_and_offsets_count_real_integer_bytes() -> None:
    static = np.zeros((gf2.HEIGHT, gf2.WIDTH), dtype=np.uint8)
    offsets = np.zeros((gf2.N_PAIRS, 2), dtype=np.int16)
    static_payload = gf2.packed_static(static)
    offsets_payload = gf2.packed_offsets(offsets)
    assert len(static_payload) == gf2.STATIC_HEADER.size + gf2.HEIGHT * gf2.WIDTH
    assert len(offsets_payload) == gf2.OFFSETS_HEADER.size + gf2.N_PAIRS * 2 * 2
    assert static_payload[:4] == gf2.STATIC_MAGIC
    assert offsets_payload[:4] == gf2.OFFSETS_MAGIC
    assert np.array_equal(gf2.unpack_static(static_payload), static)
    assert np.array_equal(gf2.unpack_offsets(offsets_payload), offsets)


def test_charter_ceiling_arithmetic_is_optimistic() -> None:
    repair_sites = (
        gf2.PACKET_CAP_HALF_BYTE_NUMERATOR * gf2.GENERIC_BYTES_PER_SITE_DENOMINATOR
    ) // (2 * gf2.GENERIC_BYTES_PER_SITE_NUMERATOR)
    assert repair_sites == 245_460
    assert gf2.MISMATCH_TARGET + repair_sites == 292_264


def test_fit_offsets_recovers_known_small_translations(monkeypatch) -> None:
    monkeypatch.setattr(gf2, "N_PAIRS", 2)
    monkeypatch.setattr(gf2, "HEIGHT", 4)
    monkeypatch.setattr(gf2, "WIDTH", 5)
    template = np.asarray(
        [
            [2, 2, 2, 2, 2],
            [2, 0, 1, 3, 2],
            [2, 4, 3, 1, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.uint8,
    )
    target = np.stack(
        [
            gf2.render_translation(template, 0, 0),
            gf2.render_translation(template, 1, -1),
        ]
    )
    offsets, scores = gf2.fit_offsets(target, template, radius=1, pair_chunk=1)
    assert offsets.tolist() == [[0, 0], [1, -1]]
    assert scores.tolist() == [20, 20]


def test_tiny_end_to_end_retains_exact_reconstruction_and_resumes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(gf2, "N_PAIRS", 2)
    monkeypatch.setattr(gf2, "HEIGHT", 4)
    monkeypatch.setattr(gf2, "WIDTH", 5)
    monkeypatch.setattr(gf2, "FIELD_SHAPE", (2, 4, 5))
    monkeypatch.setattr(gf2, "FIELD_BYTES", 40)
    monkeypatch.setattr(gf2, "SEARCH_RADIUS", 1)
    monkeypatch.setattr(gf2, "MAX_ALIGNMENT_ITERATIONS", 2)
    monkeypatch.setattr(gf2, "MINIMUM_FREE_BYTES", 1)
    output = tmp_path / "retained"
    monkeypatch.setattr(gf2, "OUTPUT", output)
    field = np.asarray(
        [
            [[2, 2, 2, 2, 2], [2, 0, 1, 3, 2], [2, 4, 3, 1, 2], [2, 2, 2, 2, 2]],
            [[2, 2, 2, 2, 2], [2, 0, 1, 3, 2], [2, 4, 3, 1, 2], [2, 2, 2, 2, 2]],
        ],
        dtype=np.uint8,
    )
    field_path = tmp_path / "field.u8"
    field.tofile(field_path)
    field_sha = hashlib.sha256(field_path.read_bytes()).hexdigest()
    monkeypatch.setattr(gf2, "FIELD_SHA256", field_sha)

    result = gf2.run(output, field_path)
    assert result["static_ceiling"]["aligned"]["total"] == 0
    assert result["residual"]["receiver_reconstruction"]["field"]["sha256"] == field_sha
    assert result["static_ceiling"]["integer_packet_parseback_exact"] is True
    assert (output / "MANIFEST.json").is_file()
    assert gf2.run(output, field_path) == result
