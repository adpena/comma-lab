# SPDX-License-Identifier: MIT
"""End-to-end wire-in tests for the 5 P1 class-shift substrate parsers.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G).

These tests exercise the FULL ``tac.analysis.hnerv_packet_sections`` pipeline
(magic-prefix auto-detection -> parser dispatch -> ScorerConditionalMDLEstimator
role taxonomy -> coverage validation) for the 5 P1 substrates:

- ``C1WMFV1`` (world-model + foveation, magic ``b"WMF\\x01"``)
- ``WZ1``    (Wyner-Ziv cooperative-receiver,           magic ``b"WZ1\\x00"``)
- ``Z4CR1``  (cooperative-receiver loss,                magic ``b"Z4CR"``)
- ``Z5PCWM1``(predictive-coding world-model,            magic ``b"Z5WM"``)
- ``TT5L``   (Time-Traveler L5 autonomy,                magic ``b"TT5L"``)

Sister of ``src/tac/tests/test_hnerv_packet_sections.py`` (covers PR101/PR103/
PR106/A2K1/A5FC/CPLX1 frontier-replay parsers) and the per-substrate canonical
tests under ``src/tac/substrates/<name>/tests/``.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from tac.analysis.hnerv_packet_sections import (
    MANIFEST_SCHEMA,
    PARSER_AUTO,
    PARSER_C1WMFV1,
    PARSER_TT5L,
    PARSER_WZ1,
    PARSER_Z4CR1,
    PARSER_Z5PCWM1,
    build_packet_section_manifest,
    validate_packet_section_manifest,
)
from tac.substrates.c1_world_model_foveation.archive import (
    C1WMFV1_HEADER_FMT,
    C1WMFV1_MAGIC,
    C1WMFV1_SCHEMA_VERSION,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_FMT,
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
)
from tac.substrates.wyner_ziv_cooperative_receiver.archive import (
    WZ1_HEADER_FMT,
    WZ1_MAGIC,
    WZ1_SCHEMA_VERSION,
)
from tac.substrates.z4_cooperative_receiver_loss.archive import (
    Z4CR1_HEADER_FMT,
    Z4CR1_MAGIC,
    Z4CR1_SCHEMA_VERSION,
)
from tac.substrates.z5_predictive_coding_world_model.archive import (
    Z5PCWM1_HEADER_FMT,
    Z5PCWM1_MAGIC,
    Z5PCWM1_SCHEMA_VERSION,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _stored_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _c1wmfv1_payload() -> bytes:
    wm, dec, zi, fov, res, meta = (
        b"\xaa" * 100,
        b"\xbb" * 200,
        b"\xcc" * 50,
        b"\xdd" * 30,
        b"\xee" * 120,
        b'{"k":"v"}',
    )
    header = struct.pack(
        C1WMFV1_HEADER_FMT,
        C1WMFV1_MAGIC,
        C1WMFV1_SCHEMA_VERSION,
        600,
        0,
        1,
        64,
        384,
        512,
        len(wm),
        len(dec),
        len(zi),
        len(fov),
        len(res),
        len(meta),
    )
    return header + wm + dec + zi + fov + res + meta


def _wz1_payload() -> bytes:
    ren, sip, coset, meta = (
        b"\xaa" * 100,
        b"\xbb" * 100,
        b"\xcc" * 50,
        b'{"k":"v"}',
    )
    header = struct.pack(
        WZ1_HEADER_FMT,
        WZ1_MAGIC,
        WZ1_SCHEMA_VERSION,
        600,
        64,
        2,
        32,
        2,
        384,
        512,
        6,
        4,
        len(ren),
        len(sip),
        len(coset),
        len(meta),
    )
    return header + ren + sip + coset + meta


def _z4cr1_payload() -> bytes:
    enc, dec, meta = b"\xaa" * 100, b"\xbb" * 200, b'{"k":"v"}'
    latent = b"\xcc" * (600 * 24)
    header = struct.pack(
        Z4CR1_HEADER_FMT,
        Z4CR1_MAGIC,
        Z4CR1_SCHEMA_VERSION,
        24,
        600,
        len(enc),
        len(dec),
        len(latent),
        len(meta),
    )
    return header + enc + dec + latent + meta


def _z5pcwm1_payload() -> bytes:
    enc, dec, pred, meta = b"\xaa" * 100, b"\xbb" * 200, b"\xcc" * 150, b'{"k":"v"}'
    latent_init = b"\xdd" * 24
    residuals = b"\xee" * (600 * 24)
    ego = b"\xff" * (600 * 8)
    header = struct.pack(
        Z5PCWM1_HEADER_FMT,
        Z5PCWM1_MAGIC,
        Z5PCWM1_SCHEMA_VERSION,
        24,
        8,
        600,
        len(enc),
        len(dec),
        len(pred),
        len(latent_init),
        len(residuals),
        len(ego),
        len(meta),
    )
    return header + enc + dec + pred + latent_init + residuals + ego + meta


def _tt5l_payload() -> bytes:
    wm, sip, ac, meta = b"\xaa" * 100, b"\xbb" * 200, b"\xcc" * 40, b'{"k":"v"}'
    header = struct.pack(
        TT5L_HEADER_FMT,
        TT5L_MAGIC,
        TT5L_SCHEMA_VERSION,
        600,
        64,
        2,
        384,
        512,
        8,
        8,
        6,
        45,
        len(wm),
        len(sip),
        len(ac),
        len(meta),
    )
    return header + wm + sip + ac + meta


# --------------------------------------------------------------------------
# C1WMFV1 wire-in (world-model + foveation)
# --------------------------------------------------------------------------


def test_c1wmfv1_manifest_records_world_model_decoder_foveation_sections(tmp_path: Path) -> None:
    payload = _c1wmfv1_payload()
    archive = tmp_path / "c1wmfv1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="C1WMFV1", parser=PARSER_C1WMFV1)

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_C1WMFV1
    section_names = [section["name"] for section in manifest["sections"]]
    assert section_names == [
        "c1wmfv1_header",
        "world_model_blob",
        "decoder_blob",
        "z_init_blob",
        "foveation_meta_blob",
        "residual_blob",
        "meta_blob",
    ]
    assert validate_packet_section_manifest(manifest) == []


def test_c1wmfv1_manifest_auto_detected_by_magic_prefix(tmp_path: Path) -> None:
    payload = _c1wmfv1_payload()
    archive = tmp_path / "auto_c1wmfv1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="auto", parser=PARSER_AUTO)
    assert manifest["parser"]["name"] == PARSER_C1WMFV1


# --------------------------------------------------------------------------
# WZ1 wire-in (Wyner-Ziv cooperative-receiver)
# --------------------------------------------------------------------------


def test_wz1_manifest_records_renderer_side_info_coset_sections(tmp_path: Path) -> None:
    payload = _wz1_payload()
    archive = tmp_path / "wz1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="WZ1", parser=PARSER_WZ1)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_WZ1
    section_names = [section["name"] for section in manifest["sections"]]
    assert section_names == [
        "wz1_header",
        "renderer_blob",
        "side_info_predictor_blob",
        "coset_indices_blob",
        "meta_blob",
    ]
    assert validate_packet_section_manifest(manifest) == []


def test_wz1_manifest_auto_detected_by_magic_prefix(tmp_path: Path) -> None:
    payload = _wz1_payload()
    archive = tmp_path / "auto_wz1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="auto", parser=PARSER_AUTO)
    assert manifest["parser"]["name"] == PARSER_WZ1


# --------------------------------------------------------------------------
# Z4CR1 wire-in (cooperative-receiver loss)
# --------------------------------------------------------------------------


def test_z4cr1_manifest_records_encoder_decoder_latent_sections(tmp_path: Path) -> None:
    payload = _z4cr1_payload()
    archive = tmp_path / "z4cr1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="Z4CR1", parser=PARSER_Z4CR1)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_Z4CR1
    section_names = [section["name"] for section in manifest["sections"]]
    assert section_names == [
        "z4cr1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    ]
    # encoder_blob role must be training_provenance_only (inherited from IBPS1 sister)
    encoder = next(s for s in manifest["sections"] if s["name"] == "encoder_blob")
    assert encoder["optimization_role"] == "training_provenance_only"
    assert validate_packet_section_manifest(manifest) == []


def test_z4cr1_manifest_auto_detected_by_magic_prefix(tmp_path: Path) -> None:
    payload = _z4cr1_payload()
    archive = tmp_path / "auto_z4cr1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="auto", parser=PARSER_AUTO)
    assert manifest["parser"]["name"] == PARSER_Z4CR1


# --------------------------------------------------------------------------
# Z5PCWM1 wire-in (predictive-coding world-model)
# --------------------------------------------------------------------------


def test_z5pcwm1_manifest_records_predictor_residuals_ego_motion_sections(tmp_path: Path) -> None:
    payload = _z5pcwm1_payload()
    archive = tmp_path / "z5pcwm1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="Z5PCWM1", parser=PARSER_Z5PCWM1)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_Z5PCWM1
    section_names = [section["name"] for section in manifest["sections"]]
    assert section_names == [
        "z5pcwm1_header",
        "encoder_blob",
        "decoder_blob",
        "predictor_blob",
        "latent_init_blob",
        "residuals_blob",
        "ego_motion_blob",
        "meta_blob",
    ]
    # Rao-Ballard predictor is decoder_weight_stream
    predictor = next(s for s in manifest["sections"] if s["name"] == "predictor_blob")
    assert predictor["optimization_role"] == "decoder_weight_stream"
    assert validate_packet_section_manifest(manifest) == []


def test_z5pcwm1_manifest_auto_detected_by_magic_prefix(tmp_path: Path) -> None:
    payload = _z5pcwm1_payload()
    archive = tmp_path / "auto_z5pcwm1.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="auto", parser=PARSER_AUTO)
    assert manifest["parser"]["name"] == PARSER_Z5PCWM1


# --------------------------------------------------------------------------
# TT5L wire-in (Time-Traveler L5 autonomy)
# --------------------------------------------------------------------------


def test_tt5l_manifest_records_world_model_side_info_ac_state_sections(tmp_path: Path) -> None:
    payload = _tt5l_payload()
    archive = tmp_path / "tt5l.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="TT5L", parser=PARSER_TT5L)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_TT5L
    section_names = [section["name"] for section in manifest["sections"]]
    assert section_names == [
        "tt5l_header",
        "world_model_blob",
        "per_pair_side_info_blob",
        "ac_state_blob",
        "meta_blob",
    ]
    # AC state is consumed in TT5L v1 as residual calibration, not range/ANS.
    ac = next(s for s in manifest["sections"] if s["name"] == "ac_state_blob")
    assert ac["optimization_role"] == "sidecar_or_correction_stream"
    assert validate_packet_section_manifest(manifest) == []


def test_tt5l_manifest_auto_detected_by_magic_prefix(tmp_path: Path) -> None:
    payload = _tt5l_payload()
    archive = tmp_path / "auto_tt5l.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="auto", parser=PARSER_AUTO)
    assert manifest["parser"]["name"] == PARSER_TT5L


# --------------------------------------------------------------------------
# Batch manifest covering all 5 P1 substrates
# --------------------------------------------------------------------------


def test_batch_manifest_covers_all_five_p1_class_shift_substrates(tmp_path: Path) -> None:
    """The batch manifest builder must accept all 5 P1 parsers in a single call."""
    from tac.analysis.hnerv_packet_sections import (
        BATCH_SCHEMA,
        build_packet_section_manifest_batch,
        validate_packet_section_manifest_batch,
    )

    archives: list[tuple[str, Path, str]] = []
    for name, payload, parser in [
        ("c1wmfv1", _c1wmfv1_payload(), PARSER_C1WMFV1),
        ("wz1", _wz1_payload(), PARSER_WZ1),
        ("z4cr1", _z4cr1_payload(), PARSER_Z4CR1),
        ("z5pcwm1", _z5pcwm1_payload(), PARSER_Z5PCWM1),
        ("tt5l", _tt5l_payload(), PARSER_TT5L),
    ]:
        archive = tmp_path / f"{name}.zip"
        _stored_zip(archive, "0.bin", payload)
        archives.append((name, archive, parser))

    batch = build_packet_section_manifest_batch(archives)
    assert batch["schema"] == BATCH_SCHEMA
    assert batch["score_claim"] is False
    assert batch["dispatch_attempted"] is False
    assert batch["parser_section_gate"]["ready"] is True
    assert len(batch["records"]) == 5
    assert {record["parser"]["name"] for record in batch["records"]} == {
        PARSER_C1WMFV1,
        PARSER_WZ1,
        PARSER_Z4CR1,
        PARSER_Z5PCWM1,
        PARSER_TT5L,
    }
    assert validate_packet_section_manifest_batch(batch) == []
