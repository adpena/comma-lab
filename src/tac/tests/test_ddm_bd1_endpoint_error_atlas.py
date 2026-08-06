from __future__ import annotations

import inspect

import numpy as np

import experiments.ddm_jd4_endpoint_n600_both_bases as probe


def test_emit_error_atlas_defaults_off_for_receipt_schema_stability():
    args = probe.build_argparser().parse_args([])
    assert args.emit_error_atlas is False

    src = inspect.getsource(probe.main)
    attach = 'receipt["error_atlas_manifest"] = str(error_atlas_manifest_path)'
    assert attach in src
    assert "if args.emit_error_atlas:\n            " + attach in src


def test_pack_error_atlas_map_roundtrips_raster_order():
    realized = np.array([[0, 1, 2, 3], [4, 4, 0, 1]], dtype=np.int64)
    lstar = np.array([[0, 2, 2, 3], [3, 4, 0, 0]], dtype=np.int64)

    packed = probe.pack_error_atlas_map(realized, lstar)
    unpacked = np.unpackbits(packed, bitorder="big")[: realized.size]

    assert unpacked.reshape(realized.shape).astype(bool).tolist() == [
        [False, True, False, False],
        [True, False, False, True],
    ]


def test_write_error_atlas_npz_is_deterministic_and_manifested(tmp_path):
    packed = np.array([[0b10100000], [0b01000000]], dtype=np.uint8)
    pair_ids = [7, 9]
    out = tmp_path / "probe.error_atlas.ema.npz"

    row1 = probe.write_error_atlas_npz(
        out, packed=packed, pair_ids=pair_ids, field_shape=(2, 2, 4), basis="ema")
    sha1 = row1["sha256"]
    bytes1 = out.read_bytes()
    row2 = probe.write_error_atlas_npz(
        out, packed=packed, pair_ids=pair_ids, field_shape=(2, 2, 4), basis="ema")

    assert row2["sha256"] == sha1
    assert out.read_bytes() == bytes1
    assert row1["bytes"] == out.stat().st_size
    assert row1["bitorder"] == "big"
    with np.load(out) as z:
        assert np.array_equal(z["error_atlas_packbits"], packed)
        assert np.array_equal(z["pair_ids"], np.array(pair_ids, dtype=np.int32))
        assert np.array_equal(z["field_shape"], np.array([2, 2, 4], dtype=np.int32))
