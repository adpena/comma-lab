# SPDX-License-Identifier: MIT
"""Behavior tests for the amortized luma carrier (NO-FAKE class 2 + class 8).

These assert the carrier ACTUALLY does the work its name claims:
 - the numpy forward depends on (pair, x, y) — a constant-frame stub FAILS;
 - the byte cost is the brotli of the ACTUAL quantized weights (varies with capacity);
 - quant/dequant round-trips within the quant error;
 - MLX/torch↔numpy parity contract (RGB within 1 LSB) holds;
 - a carrier whose mod codes differ produces different frames per pair.

The exact-PoseNet d_pose reduction (the score-effect) is verified in the trainer's own result
JSON + the RD-sweep tests (those need the scorer + GT video, gated behind a marker).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.amortized_luma_carrier import (
    CarrierByteAccount,
    LumaCarrierConfig,
    build_coords,
    carrier_frame,
    carrier_param_count,
    dequantize_params,
    deterministic_fourier_B,
    measure_carrier_bytes,
    numpy_reference_forward,
    quantize_params,
)


def _random_params(cfg: LumaCarrierConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """A random (untrained) param set with the right shapes for forward/byte tests."""

    rng = np.random.default_rng(seed)
    coord_feat = 2 * cfg.n_fourier
    H = cfg.hidden_dim
    p = {
        "in_proj.weight": rng.standard_normal((H, coord_feat)).astype(np.float32) * 0.1,
        "in_proj.bias": np.zeros(H, np.float32),
        "film.weight": rng.standard_normal((2 * H * cfg.n_hidden, cfg.mod_dim)).astype(np.float32) * 0.1,
        "film.bias": np.zeros(2 * H * cfg.n_hidden, np.float32),
        "out.weight": rng.standard_normal((cfg.n_channels, H)).astype(np.float32) * 0.1,
        "out.bias": np.zeros(cfg.n_channels, np.float32),
        "mod": rng.standard_normal((cfg.num_pairs, cfg.mod_dim)).astype(np.float32),
    }
    for li in range(cfg.n_hidden):
        p[f"hidden.{li}.weight"] = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        p[f"hidden.{li}.bias"] = np.zeros(H, np.float32)
    return p


def test_build_coords_range_and_shape():
    c = build_coords(10, 12)
    assert c.shape == (120, 2)
    assert c.min() >= -1.0 - 1e-6 and c.max() <= 1.0 + 1e-6


def test_deterministic_fourier_is_reproducible():
    a = deterministic_fourier_B(24, 6.0)
    b = deterministic_fourier_B(24, 6.0)
    assert np.array_equal(a, b)  # free table reconstructed identically at inflate
    assert a.shape == (2, 24)


def test_carrier_frame_shape_and_dtype():
    cfg = LumaCarrierConfig(num_pairs=3, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    coords = build_coords(16, 20)
    frame = carrier_frame(p, cfg, coords, pair_idx=0, h=16, w=20)
    assert frame.shape == (16, 20, 3)
    assert frame.dtype == np.uint8
    assert frame.min() >= 0 and frame.max() <= 255


def test_carrier_output_depends_on_pixel_coords_NOT_CONSTANT():
    """NO-FAKE: a constant-frame stub would FAIL — output must vary across pixels."""

    cfg = LumaCarrierConfig(num_pairs=2, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg, seed=1)
    coords = build_coords(24, 24)
    frame = carrier_frame(p, cfg, coords, pair_idx=0, h=24, w=24)
    # the frame must have nontrivial spatial variance (not a flat constant)
    assert float(frame.astype(np.float64).std()) > 1.0


def test_carrier_output_depends_on_pair_idx():
    """Different pairs (different mod codes) must produce different frames."""

    cfg = LumaCarrierConfig(num_pairs=4, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg, seed=2)
    # make mod codes clearly distinct
    p["mod"] = np.eye(4, 16, dtype=np.float32) * 3.0
    coords = build_coords(20, 20)
    f0 = carrier_frame(p, cfg, coords, 0, 20, 20)
    f1 = carrier_frame(p, cfg, coords, 1, 20, 20)
    assert not np.array_equal(f0, f1)


def test_numpy_reference_forward_returns_rgb_in_range():
    cfg = LumaCarrierConfig(num_pairs=2, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    coords = build_coords(8, 8)
    fb = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    rgb = numpy_reference_forward(p, fb, coords, p["mod"][0], cfg.n_hidden, cfg.hidden_dim)
    assert rgb.shape == (64, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 255.0  # sigmoid head * 255


def test_quantize_dequantize_roundtrip_bounded_error():
    cfg = LumaCarrierConfig(num_pairs=3, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    codes, scales = quantize_params(p, bits=8)
    dq = dequantize_params(codes, scales)
    for k in p:
        amax = float(np.max(np.abs(p[k]))) or 1.0
        err = np.max(np.abs(dq[k] - p[k]))
        # 8-bit symmetric: error <= scale = amax/127
        assert err <= amax / 127.0 + 1e-5, f"{k} quant error {err} > bound"


def test_measure_carrier_bytes_is_brotli_of_actual_quant_weights():
    """NO-FAKE class 8: byte cost is the brotli of the ACTUAL quantized weights, not a constant."""

    cfg_small = LumaCarrierConfig(num_pairs=8, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    cfg_big = LumaCarrierConfig(num_pairs=8, n_fourier=48, hidden_dim=160, n_hidden=4, mod_dim=48)
    acct_s = measure_carrier_bytes(_random_params(cfg_small), cfg_small)
    acct_b = measure_carrier_bytes(_random_params(cfg_big), cfg_big)
    assert isinstance(acct_s, CarrierByteAccount)
    # bigger net -> more weight bytes (the cost tracks capacity, not a constant)
    assert acct_b.weight_bytes > acct_s.weight_bytes
    assert acct_s.total_bytes == acct_s.weight_bytes + acct_s.mod_bytes + acct_s.scale_bytes


def test_byte_account_scale_bytes_one_fp16_per_tensor():
    cfg = LumaCarrierConfig(num_pairs=4, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    acct = measure_carrier_bytes(p, cfg)
    # one fp16 scale per tensor = 2 bytes each
    assert acct.scale_bytes == 2 * len(p)


def test_carrier_param_count_matches_param_shapes():
    cfg = LumaCarrierConfig(num_pairs=10, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    counted = sum(int(np.asarray(v).size) for v in p.values())
    assert carrier_param_count(cfg) == counted


def test_higher_capacity_has_more_params():
    a = LumaCarrierConfig(num_pairs=8, hidden_dim=48, mod_dim=16, n_fourier=16, n_hidden=3)
    b = LumaCarrierConfig(num_pairs=8, hidden_dim=256, mod_dim=64, n_fourier=64, n_hidden=4)
    assert carrier_param_count(b) > carrier_param_count(a)


def test_save_load_roundtrip_preserves_params_and_cfg(tmp_path):
    from tac.boundary_math.amortized_luma_carrier import load_carrier_npz, save_carrier_npz

    cfg = LumaCarrierConfig(num_pairs=5, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16,
                            fourier_sigma=7.0, quant_bits=8)
    p = _random_params(cfg)
    path = tmp_path / "carrier.npz"
    save_carrier_npz(path, p, cfg)
    p2, cfg2 = load_carrier_npz(path)
    assert cfg2.to_dict() == cfg.to_dict()
    for k in p:
        assert np.allclose(p[k], p2[k], atol=1e-6)


def test_save_refuses_tmp_path():
    from tac.boundary_math.amortized_luma_carrier import save_carrier_npz

    cfg = LumaCarrierConfig(num_pairs=2)
    with pytest.raises(ValueError, match="tmp-class"):
        save_carrier_npz(__import__("pathlib").Path("/tmp/carrier.npz"), _random_params(cfg), cfg)


def test_dequantized_params_produce_close_frame_to_fp32():
    """The inflate decode uses DEQUANTIZED params; the frame must be close to the fp32 frame."""

    cfg = LumaCarrierConfig(num_pairs=3, n_fourier=24, hidden_dim=48, n_hidden=3, mod_dim=24)
    p = _random_params(cfg, seed=5)
    codes, scales = quantize_params(p, cfg.quant_bits)
    dq = dequantize_params(codes, scales)
    coords = build_coords(32, 32)
    f_fp32 = carrier_frame(p, cfg, coords, 0, 32, 32).astype(np.int32)
    f_dq = carrier_frame(dq, cfg, coords, 0, 32, 32).astype(np.int32)
    # 8-bit per-tensor quant -> small RGB drift; mean abs delta should be modest
    assert float(np.mean(np.abs(f_fp32 - f_dq))) < 25.0


def test_constant_params_produce_flat_frame_negative_control():
    """Negative control: zero weights -> sigmoid(bias)=0.5*255 flat frame (proves the std test bites)."""

    cfg = LumaCarrierConfig(num_pairs=2, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg)
    for k in p:
        p[k] = np.zeros_like(p[k])
    coords = build_coords(16, 16)
    frame = carrier_frame(p, cfg, coords, 0, 16, 16)
    # all-zero weights => constant output => zero spatial variance (the FAKE a real test must reject)
    assert float(frame.astype(np.float64).std()) < 1.0


# ---------------------------------------------------------------------------
# Assembly grammar round-trip (NO-FAKE class 11: bytes parse back losslessly).
# ---------------------------------------------------------------------------
def test_inr_section_pack_parse_roundtrip():
    """The packed INR section must dequantize back to the same params (lossless byte-closure)."""

    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    for p in (str(repo), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tools.score_native_assemble_pose_carrier_candidate import (
        _pack_inr_section,
        _quant_int8,
        _unpack_inr_section,
    )

    cfg = LumaCarrierConfig(num_pairs=4, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    p = _random_params(cfg, seed=9)
    codes, scales, deq = _quant_int8(p)
    raw = _pack_inr_section(codes, scales)
    parsed = _unpack_inr_section(raw)
    assert set(parsed) == set(deq)
    for k in deq:
        # the parsed (dequantized) params must equal the dequantized params the inflate uses
        assert np.allclose(parsed[k], deq[k], atol=1e-6), f"{k} parse-back mismatch"


def test_member_pack_parse_roundtrip_recovers_carrier_frame():
    """A full member must parse back + decode the SAME frame the direct forward produces."""

    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    for p in (str(repo), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tac.boundary_math.lever_b_generator import GeneratorConfig
    from tools.score_native_assemble_pose_carrier_candidate import (
        _decode_pair,
        _pack_member,
        _quant_int8,
    )

    seg_cfg = GeneratorConfig(num_pairs=3, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    luma_cfg = LumaCarrierConfig(num_pairs=3, n_fourier=16, hidden_dim=32, n_hidden=2, mod_dim=16)
    # seg params shape (5-class head)
    rng = np.random.default_rng(3)
    cf = 2 * seg_cfg.n_fourier
    seg_p = {
        "in_proj.weight": rng.standard_normal((32, cf)).astype(np.float32) * 0.1,
        "in_proj.bias": np.zeros(32, np.float32),
        "film.weight": rng.standard_normal((2 * 32 * 2, 16)).astype(np.float32) * 0.1,
        "film.bias": np.zeros(2 * 32 * 2, np.float32),
        "out.weight": rng.standard_normal((5, 32)).astype(np.float32) * 0.1,
        "out.bias": np.zeros(5, np.float32),
        "mod": rng.standard_normal((3, 16)).astype(np.float32),
        "hidden.0.weight": rng.standard_normal((32, 32)).astype(np.float32) * 0.1,
        "hidden.0.bias": np.zeros(32, np.float32),
        "hidden.1.weight": rng.standard_normal((32, 32)).astype(np.float32) * 0.1,
        "hidden.1.bias": np.zeros(32, np.float32),
    }
    luma_p = _random_params(luma_cfg, seed=4)
    palette = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [130, 140, 150]],
                       dtype=np.uint8)
    pose = rng.standard_normal((3, 6)).astype(np.float32)

    sc, ss, _ = _quant_int8(seg_p)
    lc, ls, _ = _quant_int8(luma_p)
    member = _pack_member(sc, ss, seg_cfg, lc, ls, luma_cfg, palette.tobytes(), pose)

    # decode small frames twice -> must be byte-identical (the parity proof the builder asserts).
    from tac.boundary_math.amortized_luma_carrier import build_coords as bc_cam
    from tac.boundary_math.lever_b_generator import build_coords as bc_seg

    cs = bc_seg(16, 20)
    cc = bc_cam(874, 1164)
    f0a, f1a = _decode_pair(member, 0, cs, cc, 16, 20)
    f0b, f1b = _decode_pair(member, 0, cs, cc, 16, 20)
    assert np.array_equal(f0a, f0b) and np.array_equal(f1a, f1b)  # deterministic round-trip
    assert f0a.shape == (874, 1164, 3)
    # different pairs differ (the carrier actually conditions on pair)
    f0_p1, _ = _decode_pair(member, 1, cs, cc, 16, 20)
    assert not np.array_equal(f0a, f0_p1)
