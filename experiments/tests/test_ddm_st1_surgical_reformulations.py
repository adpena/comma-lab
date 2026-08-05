# SPDX-License-Identifier: MIT
from __future__ import annotations

import brotli
import numpy as np

from experiments import ddm_pe1_per_edge_partition_race as pe1
from experiments import ddm_st1_surgical_reformulations as st1


def test_bucket_payload_roundtrips() -> None:
    qlogits = np.asarray([-512, 0, 512, 1024], dtype="<i2")
    raw = st1.bucket_payload(qlogits, bucket_count=4, radius=2, qscale=512.0)
    header, decoded = st1.decode_bucket_payload(raw)
    assert header["schema"] == "ddm_st1_bucket_student_payload.v1"
    assert header["bucket_count"] == 4
    assert np.array_equal(decoded, qlogits)


def test_context_hash_uses_local_argmax_and_frequency() -> None:
    current = np.zeros((2, st1.SEG_H, st1.SEG_W), dtype=np.uint8)
    freq = np.zeros((st1.SEG_H, st1.SEG_W), dtype=np.uint16)
    pairs = np.asarray([0, 0], dtype=np.int16)
    y = np.asarray([20, 20], dtype=np.int16)
    x = np.asarray([20, 21], dtype=np.int16)
    h1 = st1.context_hashes(
        current=current,
        pairs=pairs,
        y=y,
        x=x,
        frequency_map=freq,
        bucket_count=1024,
        radius=1,
    )
    current[0, 20, 21] = st1.LANE
    freq[20, 21] = 7
    h2 = st1.context_hashes(
        current=current,
        pairs=pairs,
        y=y,
        x=x,
        frequency_map=freq,
        bucket_count=1024,
        radius=1,
    )
    assert h1.shape == (2,)
    assert not np.array_equal(h1, h2)


def test_compress_payload_roundtrips_brotli_candidate(tmp_path) -> None:
    raw = b"ST1 test payload" * 32
    best, rows = st1.compress_payload("fixture", raw, tmp_path)
    assert {row.codec for row in rows} == {"brotli-q11", "lzma1-raw", "zlib-9"}
    if best.codec == "brotli-q11":
        decoded = brotli.decompress((tmp_path / "fixture.brotli-q11.bin").read_bytes())
    elif best.codec == "zlib-9":
        import zlib

        decoded = zlib.decompress((tmp_path / "fixture.zlib-9.bin").read_bytes())
    else:
        decoded = pe1.unlzma1_raw((tmp_path / "fixture.lzma1-raw.bin").read_bytes(), len(raw))
    assert decoded == raw


def test_context_conditioned_record_delta_is_smaller_after_first_record() -> None:
    params0 = pe1.GeneratorParams(
        edge=(st1.ROAD, st1.LANE),
        bbox=(10, 20, 14, 26),
        gen_a_q4=(40, 80),
        gen_b_q4=(44, 84),
    )
    params1 = pe1.GeneratorParams(
        edge=(st1.ROAD, st1.LANE),
        bbox=(11, 21, 15, 27),
        gen_a_q4=(41, 81),
        gen_b_q4=(45, 85),
    )
    rec0 = st1.context_conditioned_record(
        params=params0,
        track_id=3,
        previous=None,
        static_track=True,
    )
    rec1 = st1.context_conditioned_record(
        params=params1,
        track_id=3,
        previous=st1.generator_fields(params0),
        static_track=True,
    )
    assert len(rec1) <= len(rec0)
    assert rec0[2] & 0b100 == 0
    assert rec1[2] & 0b100
