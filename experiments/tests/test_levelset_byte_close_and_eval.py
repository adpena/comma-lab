# SPDX-License-Identifier: MIT
"""Tests for tools/levelset_byte_close_and_eval.py — the LEVEL-SET witness exact-eval path.

Covers the three deliverables + gates the task requires:
  * byte-close grammar (io_pack/unpack, manifest parse-back, int8 accounting == canonical);
  * the BIT-EXACT round-trip gate (shipped inflate.py == numpy-fp32 oracle) on tiny synthetic
    checkpoints, BOTH non-self-orient AND self-orient;
  * the real upstream/evaluate.py wrapper's report parser + compute_contest_score cross-check +
    the MPS refusal (never run the heavy 600-pair eval in a unit test).
  * checkpoint load (NO-FAKE on missing), self-orient detection, packet assembly, /tmp refusal,
    capped inflate, full-output shape.

All synthetic checkpoints are TINY (render 6x8, 2 pairs, hidden 4) so the real inflate.py subprocess
runs in ~1s. NO GPU, NO MPS, NO touch of any live run.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SPEC = importlib.util.spec_from_file_location("lbce", _REPO / "tools" / "levelset_byte_close_and_eval.py")
lbce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lbce)

from tac.contest_score import compute_contest_score


# ---------------------------------------------------------------------------
# tiny synthetic checkpoint builders
# ---------------------------------------------------------------------------
def _tiny_cfg(self_orient: bool, *, n_dir_freqs: int = 2) -> dict:
    cfg = {
        "npz_name": "synthetic.npz",
        "n_classes": 3, "hidden_dim": 4, "n_hidden": 1, "mod_dim": 3, "n_pairs": 2,
        "activation": "hosc", "softmax_temp": 0.5, "chroma": True,
        "wire_w0": 20.0, "wire_s0": 10.0, "hosc_beta": 4.0, "hosc_omega": 1.0,
        "bank_n_scales": 1, "bank_n_orient0": 2, "bank_f0": 2.0, "bank_base": 2.0, "bank_n_iso": 1,
        "max_bank_freq": None, "render_h": 6, "render_w": 8, "lane_edge_weight": 0.0,
    }
    curv_w = lbce._curvelet_feat_width(cfg)
    cfg["in_feat"] = curv_w + (4 * n_dir_freqs if self_orient else 0)
    return cfg


def _tiny_params(cfg: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    hd, nh, K, md, npr = (cfg["hidden_dim"], cfg["n_hidden"], cfg["n_classes"], cfg["mod_dim"], cfg["n_pairs"])
    inf = cfg["in_feat"]

    def r(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.3)

    params = {
        "in_proj.weight": r(hd, inf), "in_proj.bias": r(hd),
        "film.weight": r(nh * 2 * hd, md), "film.bias": r(nh * 2 * hd),
        "out_sdf.weight": r(K, hd), "out_sdf.bias": r(K),
        "out_tex.weight": r(3, hd), "out_tex.bias": r(3),
        "palette": r(K, 3),
        "code": r(2 * npr, md),
    }
    for li in range(nh):
        params[f"hidden.{li}.weight"] = r(hd, hd)
        params[f"hidden.{li}.bias"] = r(hd)
    return params


def _so(cfg: dict) -> dict:
    return lbce.detect_self_orient(
        cfg, {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 3}
    )


def _write_tiny_npz(ckpt_dir: Path, cfg: dict, params: dict, name: str = "levelset_witness_ema_mlx.npz") -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save = dict(params)
    save["__cfg_hidden_dim"] = cfg["hidden_dim"]
    save["__cfg_n_hidden"] = cfg["n_hidden"]
    save["__cfg_activation"] = cfg["activation"]
    save["__cfg_softmax_temp"] = cfg["softmax_temp"]
    save["__cfg_chroma"] = int(cfg["chroma"])
    save["__cfg_wire_w0"] = cfg["wire_w0"]
    save["__cfg_wire_s0"] = cfg["wire_s0"]
    save["__cfg_hosc_beta"] = cfg["hosc_beta"]
    save["__cfg_hosc_omega"] = cfg["hosc_omega"]
    save["__bank_n_scales"] = cfg["bank_n_scales"]
    save["__bank_n_orient0"] = cfg["bank_n_orient0"]
    save["__bank_f0"] = cfg["bank_f0"]
    save["__bank_base"] = cfg["bank_base"]
    save["__bank_n_iso"] = cfg["bank_n_iso"]
    save["__cfg_max_bank_freq"] = -1.0 if cfg["max_bank_freq"] is None else cfg["max_bank_freq"]
    save["__render_hw"] = np.asarray([cfg["render_h"], cfg["render_w"]], dtype=np.int64)
    p = ckpt_dir / name
    np.savez(p, **save)
    return p


# ---------------------------------------------------------------------------
# byte-close grammar
# ---------------------------------------------------------------------------
def test_io_pack_unpack_roundtrip():
    a, b, c, d = b"manifest", b"basebytes", b"codebytes", b"posebytes"
    blob = lbce._io_pack(a, b, c, d)
    assert blob[: len(lbce._MAGIC)] == lbce._MAGIC
    off = len(lbce._MAGIC)
    got = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        got.append(blob[off : off + n])
        off += n
    assert got == [a, b, c, d]


def test_read_blob_bytes_matches_pack():
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    manifest, base_b, code_b, pose, lane_b = lbce._read_blob_bytes(blob)
    assert manifest["n_pairs"] == cfg["n_pairs"]
    assert manifest["n_classes"] == cfg["n_classes"]
    assert manifest["render_h"] == cfg["render_h"] and manifest["render_w"] == cfg["render_w"]
    assert manifest["base_param_order"] and len(base_b) > 0 and len(code_b) > 0
    assert pose == b""
    assert lane_b is None  # default-off: no 5th block, grammar identical to pre-Wave-E
    assert "lane_render_band" not in manifest


def test_int8_accounting_matches_canonical():
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg)
    _, breakdown = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    assert breakdown["accounting_matches_canonical"] is True
    assert breakdown["base_int8_brotli_bytes"] > 0
    assert breakdown["code_int8_brotli_bytes"] > 0


def test_manifest_records_self_orient_params():
    cfg = _tiny_cfg(True, n_dir_freqs=2)
    params = _tiny_params(cfg)
    so = _so(cfg)
    blob, _ = lbce.build_levelset_blob(params, cfg, so, None)
    m, *_ = lbce._read_blob_bytes(blob)
    assert m["self_orient"] is True
    assert m["n_dir_freqs"] == 2
    assert m["so_iters"] == 3 and m["so_freq_across"] == 32.0 and m["so_freq_along"] == 4.0


# ---------------------------------------------------------------------------
# self-orient detection
# ---------------------------------------------------------------------------
def test_detect_self_orient_none():
    cfg = _tiny_cfg(False)
    so = _so(cfg)
    assert so["self_orient"] is False and so["dir_w"] == 0


def test_detect_self_orient_positive():
    cfg = _tiny_cfg(True, n_dir_freqs=2)
    so = _so(cfg)
    assert so["self_orient"] is True
    assert so["n_dir_freqs"] == 2 and so["dir_w"] == 8


def test_detect_self_orient_refuses_bad_in_feat():
    cfg = _tiny_cfg(False)
    cfg["in_feat"] = cfg["in_feat"] - 1  # < curvelet width => bank cfg cannot reproduce weights
    with pytest.raises(ValueError):
        _so(cfg)


# ---------------------------------------------------------------------------
# checkpoint load (NO-FAKE)
# ---------------------------------------------------------------------------
def test_load_ckpt_roundtrip(tmp_path):
    cfg = _tiny_cfg(True, n_dir_freqs=2)
    params = _tiny_params(cfg)
    _write_tiny_npz(tmp_path, cfg, params)
    loaded_params, loaded_cfg = lbce._load_levelset_ckpt(tmp_path, None)
    assert loaded_cfg["n_pairs"] == cfg["n_pairs"]
    assert loaded_cfg["in_feat"] == cfg["in_feat"]
    assert loaded_cfg["render_h"] == 6 and loaded_cfg["render_w"] == 8
    assert "code" in loaded_params and "out_sdf.weight" in loaded_params


def test_load_ckpt_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lbce._load_levelset_ckpt(tmp_path, None)


def test_refuse_tmp_path():
    with pytest.raises(ValueError):
        lbce._refuse_tmp(Path("/tmp/levelset_packet"), "packet_dir")
    lbce._refuse_tmp(Path("/Users/x/pact/experiments/results/pkt"), "packet_dir")  # ok, no raise


# ---------------------------------------------------------------------------
# packet assembly
# ---------------------------------------------------------------------------
def test_assemble_packet_contents(tmp_path):
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    pkt = tmp_path / "pkt"
    zip_path, zip_bytes = lbce.assemble_packet(blob, pkt)
    assert zip_path.exists() and zip_bytes > 0
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["0.bin"]
        assert zf.read("0.bin") == blob
    assert (pkt / "inflate.py").exists()
    sh = pkt / "inflate.sh"
    assert sh.exists() and (sh.stat().st_mode & 0o111)  # executable


# ---------------------------------------------------------------------------
# BIT-EXACT round-trip gate (the correctness proof): shipped inflate == numpy-fp32 oracle
# ---------------------------------------------------------------------------
def _build_packet(tmp_path, cfg, params):
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    pkt = tmp_path / "pkt"
    lbce.assemble_packet(blob, pkt)
    return pkt, blob


def test_bit_exact_gate_non_self_orient(tmp_path):
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg, seed=1)
    pkt, blob = _build_packet(tmp_path, cfg, params)
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True
    assert res["n_frames_differing"] == 0 and res["max_abs_uint8_diff"] == 0
    assert res["frames_compared"] == 4


def test_bit_exact_gate_self_orient(tmp_path):
    cfg = _tiny_cfg(True, n_dir_freqs=2)
    params = _tiny_params(cfg, seed=2)
    pkt, blob = _build_packet(tmp_path, cfg, params)
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=2, strict=True)
    assert res["bit_exact"] is True and res["max_abs_uint8_diff"] == 0


def test_bit_exact_gate_detects_corruption(tmp_path):
    """NO-FAKE: if the shipped inflate.py is corrupted, the gate MUST catch it (strict raises)."""
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg, seed=3)
    pkt, blob = _build_packet(tmp_path, cfg, params)
    # corrupt the shipped decoder so its output diverges from the oracle.
    ip = pkt / "inflate.py"
    ip.write_text(ip.read_text().replace("* 255.0", "* 254.0"))
    with pytest.raises(RuntimeError):
        lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=1, strict=True)
    # non-strict returns bit_exact False rather than raising.
    res = lbce.bit_exact_roundtrip_gate(pkt, blob, gate_pairs=1, strict=False)
    assert res["bit_exact"] is False and res["n_frames_differing"] > 0


def test_numpy_oracle_reference_frames_shape(tmp_path):
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg, seed=4)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    m, base_b, code_b, _pose, _lane = lbce._read_blob_bytes(blob)
    import brotli

    flat = np.frombuffer(brotli.decompress(base_b), dtype=np.int8)
    dq = {}
    off = 0
    for name in m["base_param_order"]:
        shp = tuple(m["base_shapes"][name])
        n = int(np.prod(shp))
        dq[name] = (flat[off : off + n].astype(np.float32) * float(m["base_scales"][name])).reshape(shp)
        off += n
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32) * m["code_scale"]).reshape(m["code_shape"])
    frames, argmax = lbce.numpy_oracle_reference_frames(dq, code, m, 2)
    assert len(frames) == 4 and len(argmax) == 2
    assert frames[0].shape == (lbce.CAMERA_H, lbce.CAMERA_W, 3)
    assert frames[0].dtype == np.uint8


# ---------------------------------------------------------------------------
# run_inflate: full-output shape + capping
# ---------------------------------------------------------------------------
def test_run_inflate_full_output_shape(tmp_path):
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg, seed=5)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    pkt = tmp_path / "pkt"
    lbce.assemble_packet(blob, pkt)
    info = lbce.run_inflate(pkt, cfg["n_pairs"], None)
    assert info["full_output_shape_ok"] is True
    assert info["n_frames_emitted"] == 2 * cfg["n_pairs"]
    assert info["raw_bytes"] == info["expected_bytes"]


def test_run_inflate_capped(tmp_path):
    cfg = _tiny_cfg(False)
    params = _tiny_params(cfg, seed=6)
    blob, _ = lbce.build_levelset_blob(params, cfg, _so(cfg), None)
    pkt = tmp_path / "pkt"
    lbce.assemble_packet(blob, pkt)
    info = lbce.run_inflate(pkt, cfg["n_pairs"], 1)
    assert info["eval_pairs"] == 1 and info["n_frames_emitted"] == 2
    assert info["full_output_shape_ok"] is True


# ---------------------------------------------------------------------------
# real evaluate.py wrapper: report parse + compute_contest_score cross-check + MPS refusal
# ---------------------------------------------------------------------------
_SYNTH_REPORT = """=== Evaluation config ===
  device: cpu
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 189.59420061
  Average SegNet Distortion: 0.00650273
  Submission file size: 83,062 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00221226
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 44.25
"""


def test_parse_evaluate_report():
    parsed = lbce._parse_evaluate_report(_SYNTH_REPORT)
    assert parsed["d_seg"] == pytest.approx(0.00650273)
    assert parsed["d_pose"] == pytest.approx(189.59420061)
    assert parsed["rate"] == pytest.approx(0.00221226)
    assert parsed["final_score"] == pytest.approx(44.25)
    assert parsed["n_samples"] == 600


def test_parse_evaluate_report_missing_field_raises():
    with pytest.raises(ValueError):
        lbce._parse_evaluate_report("no scores here")


def test_compute_contest_score_matches_report():
    """The wrapper reports compute_contest_score; verify it reproduces evaluate.py's arithmetic
    (byte-identical to upstream/evaluate.py:92) for the parsed components."""
    parsed = lbce._parse_evaluate_report(_SYNTH_REPORT)
    s = compute_contest_score(parsed["d_seg"], parsed["d_pose"], 83_062)
    # evaluate.py rounds Final score to 2 decimals; the recomputed S must agree to that precision.
    assert s == pytest.approx(parsed["final_score"], abs=0.01)


def test_run_upstream_evaluate_refuses_mps(tmp_path):
    (tmp_path / "archive.zip").write_bytes(b"x")
    (tmp_path / "inflated").mkdir()
    (tmp_path / "inflated" / "0.raw").write_bytes(b"y")
    with pytest.raises(ValueError, match="MPS"):
        lbce.run_upstream_evaluate(
            tmp_path, device="mps", uncompressed_dir=_REPO / "upstream" / "videos",
            video_names_file=_REPO / "upstream" / "public_test_video_names.txt",
            archive_bytes=1, timeout=10,
        )


def test_run_upstream_evaluate_missing_inflated_raises(tmp_path):
    (tmp_path / "archive.zip").write_bytes(b"x")
    (tmp_path / "inflated").mkdir()
    with pytest.raises(FileNotFoundError):
        lbce.run_upstream_evaluate(
            tmp_path, device="cpu", uncompressed_dir=_REPO / "upstream" / "videos",
            video_names_file=_REPO / "upstream" / "public_test_video_names.txt",
            archive_bytes=1, timeout=10,
        )
