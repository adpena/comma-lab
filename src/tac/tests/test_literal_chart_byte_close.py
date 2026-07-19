"""(#497 gap-a) literal ground-chart counted receiver program — end-to-end closure tests.

Sealed semantics under test (SPEC ``Phi_p(x) = Q_p(x) Psi(H_p x)`` with the chart as a COUNTED
receiver program):
  * the chart derives ONLY from the counted chart payload (7th blob section: quantized int16
    (P,6) startup pose table + fp32 (6,) scales — ``counted_chart_payload``, NOT the carrier's
    xi_eff which does not exist at ep0);
  * trainer, tool oracle, and shipped receiver all rebuild the chart from the SAME dequantized
    values via the SAME fp64 homography composition (receiver-inlined builder pinned BIT-EXACT
    vs ``GroundFrameChart.build_from_xi`` here, constants read from tac at test time);
  * per-pair features evaluate through the SEALED ``charted_grid_bilinear_v1`` program
    (identity ref pair = exact uncharted grid) — the fast receiver path that replaces the
    n600-prohibitive direct sparse transform;
  * proof authority = ``bit_exact_roundtrip_gate`` with a NONTRIVIAL chart (real motion, at
    least one non-identity homography) — the identity chart alone is NOT closure (NO-FAKE).

Still fail-closed (pinned so the narrowed gates cannot silently widen):
  * chart_pose_dependency='counted_pose_carrier_xi' (structurally unsatisfiable at ep0);
  * chart x post-render supersample (trainer-refused; no checkpoint can exist);
  * chart_enabled checkpoint missing __chart_pose_q/__chart_pose_scales custody.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.boundary_math.ground_frame_chart import ChartCalibration, GroundFrameChart
from tac.boundary_math.localized_basis_frames import (
    CHART_EVAL_SEMANTICS_BILINEAR,
    literal_basis_program_config,
)
from tac.boundary_math.xi_pose_coder import dequantize_xi, quantize_xi

_REPO = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO / "tools" / "levelset_byte_close_and_eval.py"

# tiny-but-REAL forward dims (n-small per the live-c2 memory constraint; NOT evidence rows)
_RH, _RW = 6, 8
_N_PAIRS = 3
_HIDDEN = 4
_N_HIDDEN = 1
_MOD_DIM = 2
_K = 5


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("levelset_byte_close_chart497", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _motion_pose_table(n_pairs: int, seed: int = 0) -> np.ndarray:
    """A (P,6) pose table with REAL forward+lateral+yaw motion (nontrivial homographies)."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((n_pairs, 6), dtype=np.float64)
    poses[:, 0] = 1.0 + 0.2 * rng.standard_normal(n_pairs)      # forward
    poses[:, 1] = 0.05 * rng.standard_normal(n_pairs)            # lateral
    poses[:, 2] = 0.02 * rng.standard_normal(n_pairs)            # vertical
    poses[:, 3:] = 0.01 * rng.standard_normal((n_pairs, 3))      # rotation
    return poses


def _chart_program(**overrides):
    kw = dict(
        chart_enabled=True,
        chart_ref_pair=0,
        chart_pose_dependency="counted_chart_payload",
        chart_eval_semantics=CHART_EVAL_SEMANTICS_BILINEAR,
        chart_fine_factor=2,
    )
    kw.update(overrides)
    return literal_basis_program_config(**kw)


def _full_literal_chart_checkpoint(
    tmp_path: Path,
    program,
    *,
    seed: int = 0,
    pose_table: np.ndarray | None = None,
    omit_chart_custody: bool = False,
    tag: str = "",
) -> Path:
    """A complete shared-head level-set checkpoint with chart custody (NO-FAKE: every param the
    receiver forward reads is present and consumed; the chart custody is the trainer-persisted
    quantized table the counted payload ships)."""
    rng = np.random.default_rng(seed)

    def _w(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * 0.25

    payload: dict[str, np.ndarray] = {
        "code": _w(2 * _N_PAIRS, _MOD_DIM),
        "in_proj.weight": _w(_HIDDEN, 80),
        "in_proj.bias": _w(_HIDDEN),
        "film.weight": _w(_N_HIDDEN * 2 * _HIDDEN, _MOD_DIM),
        "film.bias": _w(_N_HIDDEN * 2 * _HIDDEN),
        "hidden.0.weight": _w(_HIDDEN, _HIDDEN),
        "hidden.0.bias": _w(_HIDDEN),
        "out_sdf.weight": _w(_K, _HIDDEN),
        "out_sdf.bias": _w(_K),
        "out_tex.weight": _w(3, _HIDDEN),
        "out_tex.bias": _w(3),
        "palette": _w(_K, 3),
        "__cfg_basis": np.asarray("literal_polar_curvelet"),
        "__cfg_basis_program_json": np.asarray(
            json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))
        ),
        "__cfg_basis_program_sha256": np.asarray(program.canonical_sha256()),
        "__cfg_basis_taper_folded": np.asarray(0),
        "__cfg_hidden_dim": np.asarray(_HIDDEN),
        "__cfg_n_hidden": np.asarray(_N_HIDDEN),
        "__cfg_activation": np.asarray("hosc"),
        "__render_hw": np.asarray([_RH, _RW]),
        "__bank_n_scales": np.asarray(4),
    }
    if program.chart_enabled and not omit_chart_custody:
        table = pose_table if pose_table is not None else _motion_pose_table(_N_PAIRS, seed)
        q, scales = quantize_xi(table)
        payload["__chart_pose_q"] = np.asarray(q, np.int16)
        payload["__chart_pose_scales"] = np.asarray(scales, np.float32)
    ckpt_dir = tmp_path / f"ckpt_chart_{tag}_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_dir / "levelset_witness_ema.npz", **payload)
    return ckpt_dir


def _dequantized_oracle_inputs(tool: ModuleType, ckpt_dir: Path):
    params, cfg = tool._load_levelset_ckpt(ckpt_dir, "levelset_witness_ema.npz")
    cfg["basis_program_deploy"] = cfg["basis_program"]  # taper off -> deploy == train program
    so = tool.detect_self_orient(cfg, {})
    blob, _manifest = tool.build_levelset_blob(params, cfg, so, None)
    m, base_b, code_b, _pose, _lane, _pcar, chart_b = tool._read_blob_bytes(blob)
    params_d = tool._decode_base_params(m, base_b)
    code_d = tool._decode_code(m, code_b)
    chart_d = (
        tool._parse_chart_payload(chart_b, int(m["chart_payload"]["n_pairs"]))
        if chart_b is not None else None
    )
    return blob, m, params_d, code_d, chart_d


def _receiver_ns(tool: ModuleType) -> dict[str, object]:
    """Exec the FULL generated receiver (embedded module + placement + inflate) — the shipped
    program surface, not a re-typed copy. Registered as a real module in sys.modules so the
    embedded dataclasses can resolve their annotations (exactly as the real inflate.py file
    does when run as __main__)."""
    import sys
    import types

    src = tool._inflate_source_for_manifest({"basis_family": "literal_polar_curvelet"})
    mod = types.ModuleType("shipped_inflate_chart497")
    sys.modules["shipped_inflate_chart497"] = mod
    try:
        exec(compile(src, "<generated inflate.py>", "exec"), mod.__dict__)  # noqa: S102
    except Exception:
        sys.modules.pop("shipped_inflate_chart497", None)
        raise
    return mod.__dict__


# ---------------------------------------------------------------------------
# (i) inlined receiver chart builder == GroundFrameChart.build_from_xi, BIT-EXACT
# ---------------------------------------------------------------------------
def test_receiver_chart_constants_match_tac(tool: ModuleType) -> None:
    from tac.camera import CAMERA_H, CAMERA_W
    from tac.clip_profile import OPENPILOT_DEVICE_HEIGHT_M, camera_for_resolution

    cam = camera_for_resolution(CAMERA_W, CAMERA_H)
    ns = _receiver_ns(tool)
    assert ns["_CH_FX"] == float(cam.fx_native)
    assert ns["_CH_FY"] == float(cam.fy_native)
    assert ns["_CH_CX"] == float(cam.cx_native)
    assert ns["_CH_CY"] == float(cam.cy_native)
    assert ns["_CH_NW"] == float(cam.native_w)
    assert ns["_CH_NH"] == float(cam.native_h)
    assert ns["_CH_D"] == float(OPENPILOT_DEVICE_HEIGHT_M)


@pytest.mark.parametrize("ref_pair", [0, 2])
@pytest.mark.parametrize("regime", ["ground", "rotonly"])
def test_receiver_chart_builder_bit_exact_vs_build_from_xi(
    tool: ModuleType, ref_pair: int, regime: str
) -> None:
    ns = _receiver_ns(tool)
    q, scales = quantize_xi(_motion_pose_table(5, seed=13))
    xi_dq = dequantize_xi(q, scales)
    calib = ChartCalibration()  # canonical defaults (s_t/s_r/pitch) — same values the program carries
    want = GroundFrameChart.build_from_xi(
        xi_dq, ref_pair=ref_pair, calib=calib, grid_hw=(_RH, _RW), regime=regime
    ).H_chart_norm
    got = ns["_chart_H_norm"](
        xi_dq, ref_pair, calib.s_t, calib.s_r, calib.pitch, regime, _RH, _RW
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # nontriviality of the fixture itself: real motion => at least one non-identity homography
    assert any(not np.array_equal(want[t], np.eye(3)) for t in range(want.shape[0]))


# ---------------------------------------------------------------------------
# (ii) counted chart section: byte roundtrip through _io_pack / _read_blob_bytes
# ---------------------------------------------------------------------------
def test_chart_payload_bytes_roundtrip(tool: ModuleType) -> None:
    q, scales = quantize_xi(_motion_pose_table(4, seed=3))
    raw = tool._chart_payload_bytes(np.asarray(q, np.int16), np.asarray(scales, np.float32))
    assert len(raw) == 4 * 12 + 24
    parsed = tool._parse_chart_payload(raw, 4)
    np.testing.assert_array_equal(parsed["q"], q)
    np.testing.assert_array_equal(parsed["scales"], scales)
    with pytest.raises(ValueError, match="chart payload"):
        tool._parse_chart_payload(raw + b"\x00", 4)  # length mismatch fails closed


def test_blob_grammar_chart_block_roundtrip(tool: ModuleType) -> None:
    q, scales = quantize_xi(_motion_pose_table(2, seed=5))
    chart_raw = tool._chart_payload_bytes(np.asarray(q, np.int16), np.asarray(scales, np.float32))
    manifest = {"chart_payload": {"n_pairs": 2}}
    mj = json.dumps(manifest, separators=(",", ":")).encode()
    blob = tool._io_pack(mj, b"B", b"C", None, None, None, chart_raw)
    m, base_b, code_b, pose_b, lane_b, pcar_b, chart_b = tool._read_blob_bytes(blob)
    assert (base_b, code_b, pose_b, lane_b, pcar_b) == (b"B", b"C", b"", None, None)
    assert chart_b == chart_raw
    # exact consumption still fails closed on trailing bytes
    with pytest.raises(ValueError, match="unconsumed"):
        tool._read_blob_bytes(blob + b"\x00")


# ---------------------------------------------------------------------------
# (iii) THE closure proof: bit_exact_roundtrip_gate STRICT with a NONTRIVIAL chart
# ---------------------------------------------------------------------------
def test_chart_bit_exact_roundtrip_gate_strict(tool: ModuleType, tmp_path: Path) -> None:
    prog = _chart_program()
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, seed=11, tag="gate")
    blob, m, params_d, code_d, chart_d = _dequantized_oracle_inputs(tool, ckpt)
    assert m["basis_program"]["chart_enabled"] is True
    assert m["chart_payload"]["n_pairs"] == _N_PAIRS
    assert chart_d is not None
    # the chart is NONTRIVIAL on this fixture (real motion): identity-only would be a vacuous proof
    chart = GroundFrameChart.build_from_xi(
        dequantize_xi(chart_d["q"], chart_d["scales"]),
        ref_pair=0, calib=ChartCalibration(
            s_t=prog.chart_s_t, s_r=prog.chart_s_r, pitch=prog.chart_pitch),
        grid_hw=(_RH, _RW),
    )
    assert any(
        not np.array_equal(chart.H_chart_norm[t], np.eye(3)) for t in range(1, _N_PAIRS)
    ), "fixture produced an identity-only chart — nontrivial closure proof impossible"
    packet_dir = tmp_path / "packet_chart"
    tool.assemble_packet(blob, packet_dir)
    report = tool.bit_exact_roundtrip_gate(packet_dir, blob, gate_pairs=_N_PAIRS, strict=True)
    assert report["bit_exact"] is True
    assert report["frames_compared"] == 2 * _N_PAIRS


def test_chart_native_bit_exact_roundtrip_gate_strict(tool: ModuleType, tmp_path: Path) -> None:
    """chart x native orientation (trainer-producible: charted base + J^-T normal covectors)."""
    prog = _chart_program(native_orientation_enabled=True, fixed_point_iteration_cap=2)
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, seed=17, tag="native")
    blob, m, _params_d, _code_d, _chart_d = _dequantized_oracle_inputs(tool, ckpt)
    assert m["basis_program"]["native_orientation_enabled"] is True
    packet_dir = tmp_path / "packet_chart_native"
    tool.assemble_packet(blob, packet_dir)
    report = tool.bit_exact_roundtrip_gate(packet_dir, blob, gate_pairs=_N_PAIRS, strict=True)
    assert report["bit_exact"] is True
    assert report["frames_compared"] == 2 * _N_PAIRS


# ---------------------------------------------------------------------------
# (iv) nontriviality: chart-on frames differ from chart-off (guards inert wire-in)
# ---------------------------------------------------------------------------
def test_chart_on_frames_differ_from_chart_off(tool: ModuleType, tmp_path: Path) -> None:
    prog_off = literal_basis_program_config()
    prog_on = _chart_program()
    dir_off = _full_literal_chart_checkpoint(tmp_path, prog_off, seed=7, tag="off")
    dir_on = _full_literal_chart_checkpoint(tmp_path, prog_on, seed=7, tag="on")
    _, m_off, p_off, c_off, _ = _dequantized_oracle_inputs(tool, dir_off)
    _, m_on, p_on, c_on, chart_on = _dequantized_oracle_inputs(tool, dir_on)
    for k in p_off:  # identical weights by seed — the ONLY delta is the chart
        np.testing.assert_array_equal(p_off[k], p_on[k])
    frames_off, _ = tool.numpy_oracle_reference_frames(p_off, c_off, m_off, _N_PAIRS)
    frames_on, _ = tool.numpy_oracle_reference_frames(
        p_on, c_on, m_on, _N_PAIRS, chart_payload=chart_on)
    assert any(
        not np.array_equal(a, b) for a, b in zip(frames_off, frames_on, strict=True)
    ), "chart-on rendered bit-identical frames to chart-off — chart wire-in is inert (FAKE closure)"
    # the reference pair (identity chart) itself is the EXACT uncharted program (frame parity there)
    np.testing.assert_array_equal(frames_off[0], frames_on[0])
    np.testing.assert_array_equal(frames_off[1], frames_on[1])


# ---------------------------------------------------------------------------
# (v) gate-pinning: the still-closed combinations refuse (cannot silently widen)
# ---------------------------------------------------------------------------
def _run_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "npz_name": "levelset_witness_ema.npz",
        "max_pairs": None,
        "fold_pose_sidecar": False,
        "pose_sidecar_path": None,
        "gt_cache": None,
        "keep_packet": False,
        "packet_dir": tmp_path / "packet_refuse",
        "skip_parity": True,
        "so_overrides": {},
    }


def test_xi_dependency_chart_refuses(tool: ModuleType, tmp_path: Path) -> None:
    prog = _chart_program(chart_pose_dependency="counted_pose_carrier_xi")
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, tag="xidep")
    with pytest.raises(ValueError, match="counted_chart_payload"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_chart_missing_custody_refuses(tool: ModuleType, tmp_path: Path) -> None:
    prog = _chart_program()
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, omit_chart_custody=True, tag="nocust")
    with pytest.raises(ValueError, match="__chart_pose_q"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_chart_plus_supersample_refuses(tool: ModuleType, tmp_path: Path) -> None:
    prog = _chart_program(aa_mode="supersample", aa_factor=2)
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, tag="chartss")
    with pytest.raises(ValueError, match="chart \\+ post-render supersample"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_capped_inflate_preserves_chart(tool: ModuleType, tmp_path: Path) -> None:
    """The gp-capped repack slices the chart rows; kept pairs' homographies are EXACT (the
    incremental composition only reads poses[ref+1..t]) — proven by gating at gate_pairs=2 < P."""
    prog = _chart_program()
    ckpt = _full_literal_chart_checkpoint(tmp_path, prog, seed=23, tag="cap")
    blob, _m, _p, _c, _cd = _dequantized_oracle_inputs(tool, ckpt)
    packet_dir = tmp_path / "packet_chart_cap"
    tool.assemble_packet(blob, packet_dir)
    report = tool.bit_exact_roundtrip_gate(packet_dir, blob, gate_pairs=2, strict=True)
    assert report["bit_exact"] is True
    assert report["frames_compared"] == 4
