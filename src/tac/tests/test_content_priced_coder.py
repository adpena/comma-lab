# SPDX-License-Identifier: MIT
"""Tests for tac.codec.content_priced_coder (SPEC_v10 P4).

Behavior-verifying: EXACT round-trip of the int8-dequantized counted params, strictly
better-or-equal content-stream bytes vs quantize_levelset_blob, honest skip of #336,
and correct gauge/palette canonicalization round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.codec import content_priced_coder as cpc


def _toy_checkpoint(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "in_proj.weight": (rng.standard_normal((96, 96)) * 0.5).astype(np.float32),
        "in_proj.bias": (rng.standard_normal(96) * 0.3).astype(np.float32),
        "hidden.0.weight": (rng.standard_normal((96, 96)) * 0.4).astype(np.float32),
        "out_sdf.weight": (rng.standard_normal((5, 96)) * 1.2 + 0.7).astype(np.float32),
        "out_sdf.bias": (rng.standard_normal(5) * 0.5 - 0.9).astype(np.float32),
        "out_tex.weight": (rng.standard_normal((3, 96)) * 0.1).astype(np.float32),
        "out_tex.bias": (rng.standard_normal(3) * 0.05).astype(np.float32),
        "palette": (rng.standard_normal((5, 3)) * 0.6 - 1.9).astype(np.float32),
        "code": (rng.standard_normal((40, 32)) * 0.5).astype(np.float32),  # (2P, D)
        "B": rng.standard_normal((32, 8)).astype(np.float32),  # free bank — excluded
        "__cfg_softmax_temp": np.asarray(0.31),  # cfg scalar — excluded
    }


def test_extract_counted_excludes_bank_and_cfg():
    ck = _toy_checkpoint()
    counted = cpc.extract_counted_params(ck)
    assert "B" not in counted
    assert "__cfg_softmax_temp" not in counted
    assert "code" in counted
    assert "out_sdf.weight" in counted


def test_round_trip_exact_default():
    ck = _toy_checkpoint(1)
    verdict = cpc.verify_round_trip(ck)
    assert verdict["exact"] is True
    assert verdict["n_tensors"] >= 8


def test_decode_reproduces_int8_dequant_exactly():
    ck = _toy_checkpoint(2)
    blob = cpc.encode(ck)
    got = cpc.decode(blob)
    want = cpc._expected_dequant(ck, ())
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name], want[name]), name


def test_better_or_equal_bytes_vs_baseline():
    ck = _toy_checkpoint(3)
    rep = cpc.compare_bytes(ck)
    # content-stream (accounting-matched) MUST be <= the baseline total by construction.
    assert rep["delta"]["content_stream_better_or_equal"] is True
    assert rep["content_priced"]["content_stream_bytes"] <= \
        rep["baseline_quantize_levelset_blob"]["total_quantized_blob_bytes"]
    # round-trip is exact
    assert rep["round_trip"]["exact"] is True


def test_donor_checkpoint_wins_bytes():
    donor = ("/Users/adpena/Projects/pact/experiments/results/"
             "levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz")
    import os
    if not os.path.exists(donor):
        pytest.skip("donor checkpoint not present on this host")
    d = dict(np.load(donor))
    rep = cpc.compare_bytes(d)
    assert rep["round_trip"]["exact"] is True
    assert rep["delta"]["content_stream_better_or_equal"] is True
    # On the REAL donor the #461 code frame-delta + permutation genuinely win (MEASURED).
    assert rep["content_priced"]["content_stream_bytes"] < \
        rep["baseline_quantize_levelset_blob"]["total_quantized_blob_bytes"]


def test_gauge_canonicalization_round_trips():
    ck = _toy_checkpoint(4)
    for canon in [("palette",), ("head",), ("palette", "head")]:
        verdict = cpc.verify_round_trip(ck, canonicalize=canon)
        assert verdict["exact"] is True


def test_palette_canonicalization_folds_channel_mean():
    ck = _toy_checkpoint(5)
    base = {k: v for k, v in cpc.extract_counted_params(ck).items() if not cpc._is_code(k)}
    out = cpc.project_palette_gauge(base)
    # palette channel-mean is (near) zero after canonicalization
    assert float(np.abs(out["palette"].mean(axis=0)).max()) < 1e-5
    # out_tex.bias absorbed the mean
    assert not np.allclose(out["out_tex.bias"], base["out_tex.bias"])


def test_head_gauge_removes_class_mean():
    ck = _toy_checkpoint(6)
    base = {k: v for k, v in cpc.extract_counted_params(ck).items() if not cpc._is_code(k)}
    out = cpc.project_head_gauge(base)
    assert float(np.abs(out["out_sdf.weight"].mean(axis=0)).max()) < 1e-5
    assert abs(float(out["out_sdf.bias"].mean())) < 1e-5


def test_bit_alloc_336_is_skipped_honestly():
    ck = _toy_checkpoint(7)
    # passing a bit_alloc value must fail-closed with the recorded skip reason.
    with pytest.raises(cpc.ContentPricedCoderError) as ei:
        cpc.encode(ck, bit_alloc={"anything": 1})
    assert "#336" in str(ei.value)
    # and compare_bytes records the skip
    rep = cpc.compare_bytes(ck)
    assert "bit_alloc_336" in rep["skipped"]


def test_unknown_canonicalization_raises():
    ck = _toy_checkpoint(8)
    with pytest.raises(cpc.ContentPricedCoderError):
        cpc.encode(ck, canonicalize=("nonexistent",))


def test_bad_blob_magic_raises():
    with pytest.raises(cpc.ContentPricedCoderError):
        cpc.decode(b"XXXXnot a blob")


def test_entropy_backend_recorded_in_manifest():
    ck = _toy_checkpoint(9)
    blob, plan = cpc._encode_with_plan(ck, canonicalize=(), bit_alloc=None)
    assert plan.base_backend in (cpc._BK_BROTLI, cpc._BK_LZMA, cpc._BK_ZLIB)
    assert plan.code_backend in (cpc._BK_BROTLI, cpc._BK_LZMA, cpc._BK_ZLIB)
    # full blob decodes back
    assert cpc.decode(blob)


def test_no_code_tensor_still_round_trips():
    ck = _toy_checkpoint(10)
    del ck["code"]
    verdict = cpc.verify_round_trip(ck)
    assert verdict["exact"] is True


def test_multiple_code_tensors_rejected():
    ck = _toy_checkpoint(11)
    ck["extra_code"] = np.zeros((4, 32), dtype=np.float32)
    with pytest.raises(cpc.ContentPricedCoderError):
        cpc.encode(ck)
