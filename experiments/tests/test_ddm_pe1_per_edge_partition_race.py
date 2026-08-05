# SPDX-License-Identifier: MIT
from __future__ import annotations

import brotli
import numpy as np
import pytest

from experiments import ddm_pe1_per_edge_partition_race as pe1


def test_curve_record_roundtrips_and_rasterizes_component() -> None:
    flat = np.asarray([10 * pe1.SEG_W + x for x in range(20, 30)], dtype=np.int32)
    comp = pe1.Component(
        uid=0,
        pair=0,
        edge=(pe1.ROAD, pe1.LANE),
        flat=flat,
        bbox=(10, 20, 11, 30),
        centroid_yx=(10.0, 24.5),
        flip_mass=7,
    )
    params = pe1.fit_curve_params(comp, knot_stride=3)
    decoded = pe1.decode_curve_params(pe1.encode_curve_params(params))
    mask = pe1.rasterize_curve(decoded)
    assert decoded.edge == (pe1.ROAD, pe1.LANE)
    assert int(mask.reshape(-1)[flat].sum()) == flat.size


def test_generator_record_roundtrips_and_decodes_nonempty_bisector() -> None:
    labels = np.full((pe1.SEG_H, pe1.SEG_W), 2, dtype=np.uint8)
    labels[8:18, :25] = pe1.ROAD
    labels[8:18, 25:50] = pe1.LANE
    flat = np.asarray([y * pe1.SEG_W + x for y in range(8, 18) for x in (24, 25)], dtype=np.int32)
    comp = pe1.Component(
        uid=0,
        pair=0,
        edge=(pe1.ROAD, pe1.LANE),
        flat=flat,
        bbox=(8, 24, 18, 26),
        centroid_yx=(12.5, 24.5),
        flip_mass=11,
    )
    params = pe1.fit_generator_params(comp, labels)
    decoded = pe1.decode_generator_params(pe1.encode_generator_params(params))
    mask = pe1.rasterize_generator(decoded)
    assert decoded.edge == (pe1.ROAD, pe1.LANE)
    assert int(mask.sum()) > 0
    assert int(mask.reshape(-1)[flat].sum()) > 0


def test_pe1_section_refuses_corrupt_brotli_payload() -> None:
    frame_records = tuple([b"\x00"] * pe1.N_PAIRS)
    raw = b"".join(frame_records)
    section = (
        pe1.PE1_HEADER.pack(
            pe1.PE1_MAGIC,
            pe1.PE1_VERSION,
            pe1.SEG_H,
            pe1.SEG_W,
            pe1.N_PAIRS,
            pe1.PE1_CURVE,
            pe1.CODEC_IDS["brotli-q11"],
            len(raw),
            pe1.N_PAIRS,
            bytes.fromhex(pe1.sha256_bytes(raw)),
        )
        + brotli.compress(raw, quality=11)
    )
    parsed = pe1.parse_pe1_section(section)
    assert parsed["component_records"] == 0
    broken = bytearray(section)
    broken[-1] ^= 1
    with pytest.raises(pe1.PE1Error):
        pe1.parse_pe1_section(bytes(broken))


def test_xi_transport_uses_tac_lie_translation_first_roundtrip() -> None:
    xi = np.asarray([1.25, -2.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rt = pe1.se3_np.log_se3(pe1.se3_np.exp_se3(xi))
    assert pe1.se3_np.CONVENTION == "translation_first_(rho,omega)"
    assert np.max(np.abs(rt - xi)) < 1e-9
