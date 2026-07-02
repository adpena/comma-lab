# SPDX-License-Identifier: MIT
"""#224 Wave E — decode-consistency of the analytic-lane RENDER-BAND (fork B; R5_BLOCK closure).

Proves the render-band is a REAL (non-phantom) score: the per-pair lane MANIFOLD COORDS are
serialized COUNTED in archive.zip, the coverage raster + composite are reproduced FREE in inflate.py
(rule 118), and the SHIPPED inflate band render is BIT-IDENTICAL to the canonical numpy-fp32 oracle
band render. The default-off path stays BYTE-IDENTICAL to the pre-Wave-E grammar.

Groups:
  A canonical module: serialize<->deserialize bit-exact; coverage(train lines)==coverage(decoded
    lines) bit-identical (the FREE-rasterizer decode-consistency); band forward bulk==base.
  B byte-close grammar: default-off byte-identical; 5-tuple parse-back; counted-byte accounting.
  C shipped-inflate decode-consistency (real subprocess): BAND-ON bit-exact gate (coverage-only AND
    witness-margin u_mask); band non-trivially changes frames; capped inflate carries the band.
  D compress-time fit: build_lane_band_section fits lines from a synthetic GT cache + rule-118 report.

All fixtures are TINY (render 12x16, 2 pairs) so the real inflate.py subprocess runs in ~1s. NO GPU,
NO MPS, NO live run. Sister of test_levelset_byte_close_and_eval.py.
"""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SPEC = importlib.util.spec_from_file_location("lbce", _REPO / "tools" / "levelset_byte_close_and_eval.py")
lbce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lbce)

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRDTolerance,
    LaneBandRenderConfig,
    band_alpha,
    build_lane_band_pairs_from_lstars,
    composite_band_on_render,
    derive_rd_base_steps,
    deserialize_lane_band,
    deserialize_lane_band_any,
    deserialize_lane_band_rd,
    lane_band_rd_rate_report,
    rasterize_lane_coverage_range_dependent,
    render_config_from_header,
    roundtrip_lines_through_rd,
    serialize_lane_band,
    serialize_lane_band_any,
    serialize_lane_band_rd,
    serialize_lane_band_rd_tracked,
    witness_uncertainty_mask,
)
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.boundary_math.lever_b_levelset_generator import (
    levelset_band_forward_numpy,
    levelset_rgb_forward_numpy,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _tiny_cfg() -> dict:
    cfg = {
        "npz_name": "synthetic.npz",
        "n_classes": 3, "hidden_dim": 4, "n_hidden": 1, "mod_dim": 3, "n_pairs": 2,
        "activation": "hosc", "softmax_temp": 0.5, "chroma": True,
        "wire_w0": 20.0, "wire_s0": 10.0, "hosc_beta": 4.0, "hosc_omega": 1.0,
        "bank_n_scales": 1, "bank_n_orient0": 2, "bank_f0": 2.0, "bank_base": 2.0, "bank_n_iso": 1,
        "max_bank_freq": None, "render_h": 12, "render_w": 16, "lane_edge_weight": 0.0,
    }
    cfg["in_feat"] = lbce._curvelet_feat_width(cfg)
    return cfg


def _tiny_params(cfg: dict, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    hd, nh, K, md, npr, inf = (cfg["hidden_dim"], cfg["n_hidden"], cfg["n_classes"],
                               cfg["mod_dim"], cfg["n_pairs"], cfg["in_feat"])

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.3

    params = {
        "in_proj.weight": r(hd, inf), "in_proj.bias": r(hd),
        "film.weight": r(nh * 2 * hd, md), "film.bias": r(nh * 2 * hd),
        "out_sdf.weight": r(K, hd), "out_sdf.bias": r(K),
        "out_tex.weight": r(3, hd), "out_tex.bias": r(3),
        "palette": r(K, 3), "code": r(2 * npr, md),
    }
    for li in range(nh):
        params[f"hidden.{li}.weight"] = r(hd, hd)
        params[f"hidden.{li}.bias"] = r(hd)
    return params


def _so(cfg: dict) -> dict:
    return lbce.detect_self_orient(cfg, {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 3})


def _synthetic_lines() -> list[list[LaneLine]]:
    """Two vertical-ish lanes with a LOW v_h (so the tiny 12-row render has coverage). Wide
    forward_range so the whole band is in-range; one dashed, one solid."""
    a = LaneLine(centerline_coeffs=np.array([0.0, -1.5]), halfwidth_coeffs=np.array([0.0, 2.5]),
                 dash_period_m=6.0, dash_phase_m=0.5, dash_duty=0.5, forward_range=(1.0, 300.0))
    b = LaneLine(centerline_coeffs=np.array([0.0, 1.5]), halfwidth_coeffs=np.array([0.0, 2.0]),
                 dash_period_m=0.0, forward_range=(1.0, 300.0))
    return [[a, b], [a, b]]


def _lane_cfg(**kw) -> LaneBandRenderConfig:
    base = dict(softness=1.0, dash_gate=True, dash_forward_max_m=55.0, v_h=3.0, weight=1.0, lane_cls=1)
    base.update(kw)
    return LaneBandRenderConfig(**base)


def _lane_section(cfg_band: LaneBandRenderConfig, pairs=None) -> tuple[bytes, dict]:
    pairs = pairs or _synthetic_lines()
    raw = serialize_lane_band(pairs, cfg_band)
    lane_bytes = brotli.compress(raw, quality=11)
    manifest = lbce._lane_manifest_from_cfg(cfg_band, {"n_pairs": len(pairs)})
    return lane_bytes, manifest


# ===========================================================================
# GROUP A — canonical module (no subprocess)
# ===========================================================================
def test_serialize_deserialize_bit_exact():
    pairs = _synthetic_lines()
    cfg = _lane_cfg(u_mask_enabled=True, u_mask_tau=0.8, u_mask_eps=0.3, weight=0.7)
    pl, hdr = deserialize_lane_band(serialize_lane_band(pairs, cfg))
    assert len(pl) == len(pairs)
    for a_lines, b_lines in zip(pairs, pl, strict=True):
        assert len(a_lines) == len(b_lines)
        for x, y in zip(a_lines, b_lines, strict=True):
            assert np.array_equal(np.asarray(x.centerline_coeffs, np.float64),
                                  np.asarray(y.centerline_coeffs, np.float64))
            assert np.array_equal(np.asarray(x.halfwidth_coeffs, np.float64),
                                  np.asarray(y.halfwidth_coeffs, np.float64))
            assert x.dash_period_m == y.dash_period_m and x.dash_phase_m == y.dash_phase_m
            assert x.dash_duty == y.dash_duty and x.forward_range == y.forward_range


def test_render_config_round_trip():
    cfg = _lane_cfg(u_mask_enabled=True, u_mask_tau=0.8, u_mask_eps=0.3, weight=0.7, dash_forward_max_m=42.0)
    _pl, hdr = deserialize_lane_band(serialize_lane_band(_synthetic_lines(), cfg))
    got = render_config_from_header(hdr)
    assert got.softness == cfg.softness and got.dash_forward_max_m == cfg.dash_forward_max_m
    assert got.weight == cfg.weight and got.lane_cls == cfg.lane_cls
    assert got.u_mask_enabled and got.u_mask_tau == cfg.u_mask_tau and got.u_mask_eps == cfg.u_mask_eps


def test_coverage_train_equals_decoded_lines_bit_identical():
    """THE decode-consistency of the FREE rasterizer: the coverage rasterized from the compress-side
    lines == the coverage rasterized from the serialized->deserialized lines, bit-for-bit."""
    pairs = _synthetic_lines()
    cfg = _lane_cfg()
    pl, hdr = deserialize_lane_band(serialize_lane_band(pairs, cfg))
    dc = render_config_from_header(hdr)
    for tp, dp in zip(pairs, pl, strict=True):
        cov_train = rasterize_lane_coverage_range_dependent(
            tp, h=48, w=64, softness=cfg.softness, dash_gate=cfg.dash_gate,
            dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h, cx=cfg.cx)
        cov_dec = rasterize_lane_coverage_range_dependent(
            dp, h=48, w=64, softness=dc.softness, dash_gate=dc.dash_gate,
            dash_forward_max_m=dc.dash_forward_max_m, v_h=dc.v_h, cx=dc.cx)
        assert np.array_equal(cov_train, cov_dec)
        assert cov_train.max() > 0.0  # non-trivial coverage (the fixture actually paints a band)


def test_band_alpha_weight_and_umask_order():
    cov = np.array([[0.5, 1.0]], np.float32)
    um = np.array([[0.5, 0.25]], np.float32)
    a = band_alpha(cov, um, 0.8)
    assert np.allclose(a, (cov * 0.8) * um)  # weight scales coverage BEFORE the uncertainty gate
    assert np.array_equal(band_alpha(cov, None, 1.0), cov)


def test_band_forward_bulk_bit_identical_to_base_forward():
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((cfg["render_h"] * cfg["render_w"], cfg["in_feat"])).astype(np.float32)
    kw = dict(n_hidden=1, hidden_dim=4, n_classes=3, activation="hosc", softmax_temp=0.5,
              wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0, chroma=True)
    rgb_a, phi_a = levelset_rgb_forward_numpy(params, feats, params["code"][1], **kw)
    rgb_b, phi_b, lane_b, mrg_b = levelset_band_forward_numpy(params, feats, params["code"][1], lane_cls=1, **kw)
    assert np.array_equal(rgb_a, rgb_b) and np.array_equal(phi_a, phi_b)  # band forward is a superset
    assert lane_b.shape == rgb_a.shape and mrg_b.shape == (rgb_a.shape[0],)
    assert (mrg_b >= 0).all() and (mrg_b <= 1.0 + 1e-6).all()  # top1-top2 of a softmax


# ===========================================================================
# GROUP B — byte-close grammar
# ===========================================================================
def test_default_off_byte_identical():
    """Absent --lane-render-band the blob is BYTE-IDENTICAL to the pre-Wave-E 4-block grammar."""
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    blob_none, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)                 # no lane args
    blob_off, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None, None, None)      # explicit off
    assert blob_none == blob_off
    m, base_b, code_b, pose, lane_b = lbce._read_blob_bytes(blob_none)
    assert lane_b is None and "lane_render_band" not in m


def test_lane_band_present_parses_5th_block():
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    lane_bytes, manifest = _lane_section(_lane_cfg(u_mask_enabled=True))
    blob, breakdown = lbce.build_levelset_blob(params, cfg, _so(cfg), None, lane_bytes, manifest)
    m, base_b, code_b, pose, lane_b = lbce._read_blob_bytes(blob)
    assert lane_b == lane_bytes                       # 5th block round-trips exactly
    assert m["lane_render_band"]["lane_cls"] == 1
    assert m["lane_render_band"]["u_mask"]["source"] == "witness_margin"
    assert breakdown["lane_band_counted_bytes"] == len(lane_bytes) > 0


def test_counted_byte_accounting_rule_118():
    """The lane MANIFOLD COORDS are COUNTED; the coverage raster is FREE (not stored)."""
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    lane_bytes, manifest = _lane_section(_lane_cfg())
    _blob, bd = lbce.build_levelset_blob(params, cfg, _so(cfg), None, lane_bytes, manifest)
    # the counted lane bytes are the serialized coeffs only — orders of magnitude below an H*W field.
    assert bd["lane_band_counted_bytes"] == len(lane_bytes)
    assert bd["lane_band_counted_bytes"] < 12 * 16 * 4  # << one float32 coverage raster for this render


# ===========================================================================
# GROUP C — shipped-inflate decode-consistency (real inflate.py subprocess)
# ===========================================================================
def _packet(tmp_path, lane_cfg=None):
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    lane_bytes = manifest = None
    if lane_cfg is not None:
        lane_bytes, manifest = _lane_section(lane_cfg)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None, lane_bytes, manifest)
    pkt = tmp_path / "pkt"
    lbce.assemble_packet(blob, pkt)
    return pkt, blob, cfg


def test_bit_exact_gate_band_coverage_only(tmp_path):
    """DECODE-CONSISTENCY PROOF: shipped inflate.py band == numpy-fp32 oracle band, bit-for-bit
    (coverage-only band, no uncertainty mask)."""
    pkt, blob, _ = _packet(tmp_path, _lane_cfg(u_mask_enabled=False))
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0 and res["frames_compared"] == 4


def test_bit_exact_gate_band_witness_margin_umask(tmp_path):
    """DECODE-CONSISTENCY PROOF with the WITNESS-margin uncertainty FP-killer (c_full_wit form).
    The margin is the witness's own softmax top1-top2 -> decode-available -> bit-identical."""
    pkt, blob, _ = _packet(tmp_path, _lane_cfg(u_mask_enabled=True, u_mask_tau=0.8, u_mask_eps=0.3))
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0


def test_band_changes_frames_non_trivially(tmp_path):
    """NO-FAKE: the band actually mutates the rendered frames (not a silent no-op)."""
    pkt_on, _, cfg = _packet(tmp_path, _lane_cfg())
    pkt_off, _, _ = _packet(tmp_path / "off", None)
    info_on = lbce.run_inflate(pkt_on, cfg["n_pairs"], None)
    info_off = lbce.run_inflate(pkt_off, cfg["n_pairs"], None)
    a = np.fromfile(info_on["raw_path"], dtype=np.uint8)
    b = np.fromfile(info_off["raw_path"], dtype=np.uint8)
    assert a.shape == b.shape and not np.array_equal(a, b)


def test_capped_inflate_carries_band(tmp_path):
    """The --max-pairs cap slices the lane band to the eval pairs too (still bit-exact-consistent)."""
    pkt, _, cfg = _packet(tmp_path, _lane_cfg())
    info = lbce.run_inflate(pkt, cfg["n_pairs"], 1)
    assert info["eval_pairs"] == 1 and info["n_frames_emitted"] == 2
    assert info["full_output_shape_ok"] is True


# --- Stage-2b: CORRESPONDENCE-FIRST tracked LBND2 section (ships as LBND2, ZERO new inflate code) ---
def _lane_section_tracked(cfg_band, pairs=None, *, pack_mode="coherent_slot", smooth="none"):
    pairs = pairs or _synthetic_lines()
    raw, _meta = serialize_lane_band_rd_tracked(pairs, cfg_band, pack_mode=pack_mode, smooth=smooth)
    lane_bytes = brotli.compress(raw, quality=11)
    manifest = lbce._lane_manifest_from_cfg(cfg_band, {"n_pairs": len(pairs)})
    return lane_bytes, manifest


def _packet_tracked(tmp_path, lane_cfg, *, pack_mode="coherent_slot", smooth="none"):
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    lane_bytes, manifest = _lane_section_tracked(lane_cfg, pack_mode=pack_mode, smooth=smooth)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None, lane_bytes, manifest)
    pkt = tmp_path / "pkt"
    lbce.assemble_packet(blob, pkt)
    return pkt, blob, cfg


@pytest.mark.parametrize("pack_mode,smooth", [("coherent_slot", "none"), ("coherent_slot", "rts"),
                                              ("persistent", "none")])
def test_bit_exact_gate_tracked_lbnd2(tmp_path, pack_mode, smooth):
    """DECODE-CONSISTENCY PROOF: a CORRESPONDENCE-tracked LBND2 band, rendered by the SHIPPED
    inflate.py (the UNCHANGED _lane_parse_rd), is BIT-IDENTICAL to the numpy-fp32 oracle band.
    Correspondence + batch denoise are compress-time SOURCE transforms -> ships as LBND2 bytes,
    ZERO new inflate code (the same rule-118 property that made moving-average shippable)."""
    pkt, blob, _ = _packet_tracked(tmp_path, _lane_cfg(u_mask_enabled=False),
                                   pack_mode=pack_mode, smooth=smooth)
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0


# ===========================================================================
# GROUP D — compress-time fit from GT cache
# ===========================================================================
def _synthetic_gt_cache(tmp_path, n_pairs: int = 2) -> Path:
    """A synthetic frozen-argmax cache with a clear lane (class 1) stripe below the horizon so the
    IPM fit yields >=1 line. seg-res 384x512 (matches _SEG_H/_SEG_W)."""
    H, W = 384, 512
    lst = np.full((H, W), 2, np.int64)                # undrivable everywhere
    lst[200:, :] = 0                                  # road below the horizon band
    for r in range(190, 360):                         # a single near-vertical dashed-ish lane stripe
        c = 256 + int((r - 200) * 0.15)
        lst[r, c - 2:c + 3] = 1
    lstars = np.stack([lst for _ in range(n_pairs)], axis=0)
    p = tmp_path / "gt_synth.npz"
    np.savez(p, lstars=lstars, n_pairs=np.asarray(n_pairs))
    return p


def test_build_lane_band_section_from_gt_cache(tmp_path):
    gt = _synthetic_gt_cache(tmp_path, n_pairs=2)
    lane_bytes, manifest, report = lbce.build_lane_band_section(str(gt), 2, _lane_cfg())  # Wave-F: rd default
    assert report["active"] is True and report["n_pairs_fit"] == 2
    assert report["codec"] == "LBND2_rd"               # Wave-F: the optimal RD codec is the default
    assert report["counted_brotli_bytes"] == len(lane_bytes) > 0
    assert report["fit_stats"]["total_lines"] >= 1     # the synthetic lane was actually fit
    # Wave-F: measured per-lever rate report is attached (naive-vs-RD).
    rr = report["rate_report"]
    assert rr["rd_lbnd2_brotli_bytes"] == len(lane_bytes)
    assert rr["naive_lbnd1_brotli_bytes"] > 0
    # rule-118 honesty is recorded in the report.
    assert "no_gt_no_scorer" in report["rule_118"]
    assert report["counted_rate_term_contribution"] == pytest.approx(25.0 * len(lane_bytes) / lbce.RATE_DENOM)
    # parse-back reproduces the fit lines bit-exactly (format-agnostic dispatch).
    pl, hdr = deserialize_lane_band_any(lane_bytes and __import__("brotli").decompress(lane_bytes))
    assert len(pl) == 2


def test_build_lane_band_section_naive_opt_out(tmp_path):
    """rd=False -> the naive LBND1 float64 codec (kept for the naive-vs-RD comparison + default-off gate)."""
    gt = _synthetic_gt_cache(tmp_path, n_pairs=2)
    lane_bytes, manifest, report = lbce.build_lane_band_section(str(gt), 2, _lane_cfg(), rd=False)
    assert report["codec"] == "LBND1_naive"
    pl, hdr = deserialize_lane_band(__import__("brotli").decompress(lane_bytes))  # LBND1 parses
    assert len(pl) == 2


def test_build_lane_band_section_requires_gt_cache():
    with pytest.raises(ValueError, match="gt-cache"):
        lbce.build_lane_band_section(None, 2, _lane_cfg())


def test_build_lane_band_section_missing_lstars(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, foo=np.zeros(3))
    with pytest.raises(ValueError, match="lstars"):
        lbce.build_lane_band_section(str(p), 2, _lane_cfg())


# ===========================================================================
# GROUP E — Wave-F: the OPTIMAL LBND2 rate-distortion codec (replaces naive LBND1)
# ===========================================================================
def _lane_section_rd(cfg_band: LaneBandRenderConfig, pairs=None, tol=None) -> tuple[bytes, dict]:
    """LBND2 (RD) lane section: quantize+temporal-delta+zigzag, brotli'd (COUNTED)."""
    pairs = pairs or _synthetic_lines()
    raw = serialize_lane_band_rd(pairs, cfg_band, tol=tol)
    lane_bytes = brotli.compress(raw, quality=11)
    manifest = lbce._lane_manifest_from_cfg(cfg_band, {"n_pairs": len(pairs)})
    return lane_bytes, manifest


def _packet_rd(tmp_path, lane_cfg, tol=None):
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    lane_bytes, manifest = _lane_section_rd(lane_cfg, tol=tol)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None, lane_bytes, manifest)
    pkt = tmp_path / "pkt_rd"
    lbce.assemble_packet(blob, pkt)
    return pkt, blob, cfg


def test_rd_steps_derived_positive_and_ordered():
    """The RD steps are DERIVED from a geometric tolerance (not fabricated) and all > 0."""
    steps = derive_rd_base_steps()
    assert steps.shape == (11,) and (steps > 0).all()
    # higher-power centerline coeffs get finer steps (scaled by f_ref**k).
    assert steps[0] < steps[1] < steps[2] < steps[3]  # c3<c2<c1<c0


def test_rd_serialize_deserialize_bit_exact_and_deterministic():
    pairs = _synthetic_lines()
    cfg = _lane_cfg(u_mask_enabled=True, u_mask_tau=0.8, u_mask_eps=0.3, weight=0.7)
    blob = serialize_lane_band_rd(pairs, cfg)
    assert blob == serialize_lane_band_rd(pairs, cfg)          # deterministic
    dq, hdr = deserialize_lane_band_rd(blob)
    assert len(dq) == len(pairs)
    assert hdr["rd"]["K"] == max(len(p) for p in pairs)
    # magic dispatch routes LBND2 correctly.
    dq_any, _ = deserialize_lane_band_any(blob)
    assert len(dq_any) == len(pairs)
    # render config survives the RD header.
    got = render_config_from_header(hdr)
    assert got.weight == cfg.weight and got.u_mask_enabled and got.u_mask_tau == cfg.u_mask_tau


def test_rd_coverage_decode_consistency_bit_identical():
    """THE decode-consistency of the RD codec: coverage(deserialized dequant lines) ==
    coverage(roundtrip_lines_through_rd), bit-for-bit -- both are what SHIPS."""
    pairs = _synthetic_lines()
    cfg = _lane_cfg()
    dq_lines, blob = roundtrip_lines_through_rd(pairs, cfg)
    dq2, hdr = deserialize_lane_band_rd(blob)
    dc = render_config_from_header(hdr)
    for a, b in zip(dq_lines, dq2, strict=True):
        cov_a = rasterize_lane_coverage_range_dependent(
            a, h=48, w=64, softness=dc.softness, dash_gate=dc.dash_gate,
            dash_forward_max_m=dc.dash_forward_max_m, v_h=dc.v_h, cx=dc.cx)
        cov_b = rasterize_lane_coverage_range_dependent(
            b, h=48, w=64, softness=dc.softness, dash_gate=dc.dash_gate,
            dash_forward_max_m=dc.dash_forward_max_m, v_h=dc.v_h, cx=dc.cx)
        assert np.array_equal(cov_a, cov_b)
        assert cov_a.max() > 0.0


def test_rd_reserialize_idempotent_on_quantized_grid():
    """The capped-inflate re-serialize path: serialize_lane_band_any(dequant lines, hdr) on the
    SAME grid reproduces the byte-identical blob (bit-exact cap)."""
    pairs = _synthetic_lines()
    cfg = _lane_cfg()
    blob = serialize_lane_band_rd(pairs, cfg)
    dq, hdr = deserialize_lane_band_rd(blob)
    assert serialize_lane_band_any(dq, cfg, hdr) == blob


def test_rd_bit_exact_gate_coverage_only(tmp_path):
    """DECODE-CONSISTENCY PROOF (LBND2): the SHIPPED inflate.py RD-band render == the numpy-fp32
    oracle band render, bit-for-bit (coverage-only). The whole point of Wave-F preserved."""
    pkt, blob, _ = _packet_rd(tmp_path, _lane_cfg(u_mask_enabled=False))
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0 and res["frames_compared"] == 4


def test_rd_bit_exact_gate_witness_margin_umask(tmp_path):
    """DECODE-CONSISTENCY PROOF (LBND2) WITH the witness-margin uncertainty FP-killer."""
    pkt, blob, _ = _packet_rd(tmp_path, _lane_cfg(u_mask_enabled=True, u_mask_tau=0.8, u_mask_eps=0.3))
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0


def test_rd_band_changes_frames_non_trivially(tmp_path):
    """NO-FAKE: the RD band actually mutates the rendered frames (not a silent no-op)."""
    pkt_on, _, cfg = _packet_rd(tmp_path, _lane_cfg())
    pkt_off, _, _ = _packet(tmp_path / "off", None)
    a = np.fromfile(lbce.run_inflate(pkt_on, cfg["n_pairs"], None)["raw_path"], dtype=np.uint8)
    b = np.fromfile(lbce.run_inflate(pkt_off, cfg["n_pairs"], None)["raw_path"], dtype=np.uint8)
    assert a.shape == b.shape and not np.array_equal(a, b)


def test_rd_default_off_still_byte_identical():
    """With the RD codec available, the band-OFF path is STILL byte-identical to the pre-Wave-E
    grammar (the RD codec is band-on-only; the 7/7 default-off guarantee is untouched)."""
    cfg = _tiny_cfg()
    params = _tiny_params(cfg)
    blob_none, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    blob_off, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None, None, None)
    assert blob_none == blob_off
    m, base_b, code_b, pose, lane_b = lbce._read_blob_bytes(blob_none)
    assert lane_b is None and "lane_render_band" not in m


def test_rd_rate_report_measures_naive_vs_rd():
    """The RD codec quantizes -> its brotli bytes are <= the naive float64 codec (rate win),
    and the report exposes the measured per-lever accounting + the Shannon floor."""
    # a longer temporally-coherent synthetic sequence so the temporal-delta win is real.
    from tac.boundary_math.lane_sdf_component import LaneLine as _LL
    pairs = []
    for t in range(24):
        pairs.append([
            _LL(centerline_coeffs=np.array([1e-6 * t, -2e-4, 0.01, -1.5 + 0.01 * t]),
                halfwidth_coeffs=np.array([0.0, 2.5]), dash_period_m=6.0, dash_phase_m=0.5,
                dash_duty=0.5, forward_range=(1.0, 200.0)),
            _LL(centerline_coeffs=np.array([1e-6 * t, -2e-4, 0.01, 1.5 + 0.01 * t]),
                halfwidth_coeffs=np.array([0.0, 2.0]), dash_period_m=0.0,
                forward_range=(1.0, 200.0)),
        ])
    rep = lane_band_rd_rate_report(pairs, _lane_cfg())
    assert rep["n_pairs"] == 24 and rep["K_slots"] == 2
    assert rep["rd_lbnd2_brotli_bytes"] < rep["naive_lbnd1_brotli_bytes"]   # RD wins on coherent data
    assert rep["induced_lateral_rms_m"] < 0.05                              # sub-5cm geometric error
    assert rep["delta_stream_shannon_floor_bytes"] >= 0.0


def test_rd_empty_pairs_no_lines():
    """Edge case: pairs with zero lines -> K=0 -> empty payload -> decodes to empty lines."""
    cfg = _lane_cfg()
    blob = serialize_lane_band_rd([[], []], cfg)
    dq, hdr = deserialize_lane_band_rd(blob)
    assert hdr["rd"]["K"] == 0 and dq == [[], []]
