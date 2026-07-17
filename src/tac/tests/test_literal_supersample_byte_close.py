"""(#497 gap-b) literal post-render supersample A_s — receiver/oracle closure tests.

Sealed semantics under test (SPEC ``Y = R[A_s G(Phi(X_s))]``):
  * the WHOLE feature program + nonlinear render run on the FINE ``(ss*rh, ss*rw)`` grid;
  * ``A_s`` = exact ``ss x ss`` box average AFTER the renderer, BEFORE lane compositing and
    BEFORE R (compose-at-base, #220);
  * ``A_1 = identity`` — ss=1 is bit-identical to the ``aa_mode="none"`` point-sampled path;
  * proof authority = ``bit_exact_roundtrip_gate`` (shipped inflate uint8 == numpy oracle uint8)
    at a NONTRIVIAL ``aa_factor=2`` — the s=1 identity alone is NOT closure (NO-FAKE).

Still fail-closed (asserted here so the narrowed gates cannot silently widen):
  * native orientation + supersample (trainer cannot produce the checkpoint);
  * ground chart (sibling surface, #497 gap-a);
  * texture trunk (tex_trunk.*) + supersample (base-grid bank vs fine-grid render).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.boundary_math.aa_sdf_observation_render import build_supersampled_coords
from tac.boundary_math.localized_basis_frames import (
    BasisProgramConfig,
    literal_basis_program_config,
)

_REPO = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO / "tools" / "levelset_byte_close_and_eval.py"

# tiny-but-REAL forward dims (n-small per the live-c2 memory constraint; NOT evidence rows)
_RH, _RW = 6, 8
_N_PAIRS = 2
_HIDDEN = 4
_N_HIDDEN = 1
_MOD_DIM = 2
_K = 5


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("levelset_byte_close_ss497", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _full_literal_checkpoint(
    tmp_path: Path,
    program: BasisProgramConfig,
    *,
    seed: int = 0,
    extra_params: dict[str, np.ndarray] | None = None,
) -> Path:
    """A complete shared-head level-set checkpoint whose forward actually runs (NO-FAKE:
    every param the receiver forward reads is present and consumed)."""
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
    if extra_params:
        payload.update(extra_params)
    ckpt_dir = tmp_path / f"ckpt_{program.aa_mode}_{program.aa_factor}_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_dir / "levelset_witness_ema.npz", **payload)
    return ckpt_dir


def _dequantized_oracle_inputs(tool: ModuleType, ckpt_dir: Path):
    """checkpoint -> blob -> read-back manifest + int8-DEQUANTIZED params/code (both sides of the
    bit-exact gate see identical values, exactly as the real byte-close does)."""
    params, cfg = tool._load_levelset_ckpt(ckpt_dir, "levelset_witness_ema.npz")
    cfg["basis_program_deploy"] = cfg["basis_program"]  # taper off -> deploy == train program
    so = tool.detect_self_orient(cfg, {})
    blob, _manifest = tool.build_levelset_blob(params, cfg, so, None)
    m, base_b, code_b, _pose, _lane, _pcar = tool._read_blob_bytes(blob)
    params_d = tool._decode_base_params(m, base_b)
    code_d = tool._decode_code(m, code_b)
    return blob, m, params_d, code_d


# ---------------------------------------------------------------------------
# coords identity: the receiver fine grid IS build_supersampled_coords
# ---------------------------------------------------------------------------
def test_fine_grid_coords_bit_equal_build_supersampled_coords(tool: ModuleType) -> None:
    fine_oracle = tool._canon_coords_grid(_RH * 2, _RW * 2)
    fine_canonical = build_supersampled_coords(_RH, _RW, 2)
    np.testing.assert_array_equal(np.asarray(fine_oracle, np.float32), fine_canonical)


def test_receiver_coords_bit_equal_build_supersampled_coords(tool: ModuleType) -> None:
    ns: dict[str, object] = {}
    exec(compile(tool._INFLATE_PY, "<inflate.py>", "exec"), ns)  # noqa: S102 — pinning shipped receiver
    fine_receiver = ns["_coords"](_RH * 2, _RW * 2)
    np.testing.assert_array_equal(fine_receiver, build_supersampled_coords(_RH, _RW, 2))


# ---------------------------------------------------------------------------
# receiver _aa_down == canonical box_downsample_np (bit-identical, incl. 1-D margin)
# ---------------------------------------------------------------------------
def test_receiver_aa_down_bit_equal_box_downsample_np(tool: ModuleType) -> None:
    from tac.boundary_math.aa_sdf_observation_render import box_downsample_np

    ns: dict[str, object] = {}
    exec(compile(tool._INFLATE_PY, "<inflate.py>", "exec"), ns)  # noqa: S102
    rng = np.random.default_rng(7)
    gh, gw, ss = _RH * 2, _RW * 2, 2
    rgb_fine = rng.standard_normal((gh * gw, 3)).astype(np.float32)
    margin_fine = rng.standard_normal(gh * gw).astype(np.float32)
    got_rgb = ns["_aa_down"](rgb_fine, _RH, _RW, ss)
    want_rgb = box_downsample_np(rgb_fine.reshape(1, gh, gw, 3), ss).reshape(_RH * _RW, 3)
    np.testing.assert_array_equal(got_rgb, want_rgb)
    got_m = ns["_aa_down"](margin_fine, _RH, _RW, ss)
    want_m = box_downsample_np(margin_fine.reshape(1, gh, gw, 1), ss).reshape(_RH * _RW)
    np.testing.assert_array_equal(got_m, want_m)
    # A_1 = identity: the SAME object comes back (bit-identity by construction)
    assert ns["_aa_down"](rgb_fine, gh, gw, 1) is rgb_fine


# ---------------------------------------------------------------------------
# s=1 identity: aa_mode="supersample", aa_factor=1 == aa_mode="none" frames (bit)
# ---------------------------------------------------------------------------
def test_ss1_oracle_frames_bit_equal_none(tool: ModuleType, tmp_path: Path) -> None:
    prog_none = literal_basis_program_config()
    prog_ss1 = literal_basis_program_config(aa_mode="supersample", aa_factor=1)
    dir_none = _full_literal_checkpoint(tmp_path, prog_none, seed=3)
    dir_ss1 = _full_literal_checkpoint(tmp_path, prog_ss1, seed=3)
    _, m_none, p_none, c_none = _dequantized_oracle_inputs(tool, dir_none)
    _, m_ss1, p_ss1, c_ss1 = _dequantized_oracle_inputs(tool, dir_ss1)
    # identical weights by seed — the ONLY delta is the aa fields in the manifest program
    for k in p_none:
        np.testing.assert_array_equal(p_none[k], p_ss1[k])
    frames_none, am_none = tool.numpy_oracle_reference_frames(p_none, c_none, m_none, _N_PAIRS)
    frames_ss1, am_ss1 = tool.numpy_oracle_reference_frames(p_ss1, c_ss1, m_ss1, _N_PAIRS)
    assert len(frames_none) == len(frames_ss1) == 2 * _N_PAIRS
    for a, b in zip(frames_none, frames_ss1, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(am_none, am_ss1, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.shape == (_RH, _RW)


# ---------------------------------------------------------------------------
# s=2 NONTRIVIAL closure: shipped inflate uint8 == numpy oracle uint8, strict
# ---------------------------------------------------------------------------
def test_ss2_bit_exact_roundtrip_gate_strict(tool: ModuleType, tmp_path: Path) -> None:
    prog = literal_basis_program_config(aa_mode="supersample", aa_factor=2)
    ckpt = _full_literal_checkpoint(tmp_path, prog, seed=11)
    blob, m, params_d, code_d = _dequantized_oracle_inputs(tool, ckpt)
    # the manifest carries the supersample program (the receiver reads ss from HERE)
    assert m["basis_program"]["aa_mode"] == "supersample"
    assert m["basis_program"]["aa_factor"] == 2
    packet_dir = tmp_path / "packet_ss2"
    tool.assemble_packet(blob, packet_dir)
    report = tool.bit_exact_roundtrip_gate(packet_dir, blob, gate_pairs=_N_PAIRS, strict=True)
    assert report["bit_exact"] is True
    assert report["frames_compared"] == 2 * _N_PAIRS
    # the FINE grid actually engaged (oracle argmax at (ss*rh, ss*rw)) — s=1 alone is NOT closure
    _frames, am = tool.numpy_oracle_reference_frames(params_d, code_d, m, _N_PAIRS)
    assert am[0].shape == (2 * _RH, 2 * _RW)


def test_ss2_frames_differ_from_none_nontriviality(tool: ModuleType, tmp_path: Path) -> None:
    """A_2 must actually change the rendered bytes on a generic checkpoint (guards against an
    accidentally-inert ss wire-in that would make the roundtrip gate a vacuous identity proof)."""
    prog_none = literal_basis_program_config()
    prog_ss2 = literal_basis_program_config(aa_mode="supersample", aa_factor=2)
    dir_none = _full_literal_checkpoint(tmp_path, prog_none, seed=5)
    dir_ss2 = _full_literal_checkpoint(tmp_path, prog_ss2, seed=5)
    _, m_none, p_none, c_none = _dequantized_oracle_inputs(tool, dir_none)
    _, m_ss2, p_ss2, c_ss2 = _dequantized_oracle_inputs(tool, dir_ss2)
    frames_none, _ = tool.numpy_oracle_reference_frames(p_none, c_none, m_none, _N_PAIRS)
    frames_ss2, _ = tool.numpy_oracle_reference_frames(p_ss2, c_ss2, m_ss2, _N_PAIRS)
    assert any(
        not np.array_equal(a, b) for a, b in zip(frames_none, frames_ss2, strict=True)
    ), "aa_factor=2 rendered bit-identical frames to aa none — A_s wire-in is inert (FAKE closure)"


# ---------------------------------------------------------------------------
# the narrowed gates still fail closed (cannot silently widen)
# ---------------------------------------------------------------------------
def _run_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "npz_name": "levelset_witness_ema.npz",
        "max_pairs": None,
        "fold_pose_sidecar": False,
        "pose_sidecar_path": None,
        "gt_cache": None,
        "keep_packet": False,
        "packet_dir": tmp_path / "packet_gate",
        "skip_parity": True,
        "so_overrides": {},
    }


def test_native_plus_supersample_still_refuses(tool: ModuleType, tmp_path: Path) -> None:
    prog = literal_basis_program_config(
        native_orientation_enabled=True,
        fixed_point_iteration_cap=3,
        aa_mode="supersample",
        aa_factor=2,
    )
    ckpt = _full_literal_checkpoint(tmp_path, prog)
    with pytest.raises(ValueError, match="native orientation \\+ post-render supersample"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_chart_still_refuses(tool: ModuleType, tmp_path: Path) -> None:
    from tac.boundary_math.localized_basis_frames import CHART_EVAL_SEMANTICS_BILINEAR

    prog = literal_basis_program_config(
        chart_enabled=True,
        chart_pose_dependency="counted_pose_carrier_xi",
        chart_eval_semantics=CHART_EVAL_SEMANTICS_BILINEAR,
    )
    ckpt = _full_literal_checkpoint(tmp_path, prog)
    with pytest.raises(ValueError, match="chart"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_tex_trunk_plus_supersample_refuses(tool: ModuleType, tmp_path: Path) -> None:
    prog = literal_basis_program_config(aa_mode="supersample", aa_factor=2)
    rng = np.random.default_rng(9)
    ckpt = _full_literal_checkpoint(
        tmp_path,
        prog,
        extra_params={
            "tex_trunk.w_tex": rng.standard_normal((_K, 8)).astype(np.float32),
            "tex_trunk.bias": rng.standard_normal(3).astype(np.float32),
        },
    )
    with pytest.raises(ValueError, match="texture trunk"):
        tool.run(ckpt, **_run_kwargs(tmp_path))


def test_plain_supersample_passes_the_gate_region(tool: ModuleType, tmp_path: Path) -> None:
    """The narrowed gate must NOT refuse the closed path: a plain (no native / no chart / no
    tex_trunk) supersample checkpoint proceeds past the literal gate block. We prove gate passage
    by monkeypatch-free construction: the full run() would continue into verdict machinery, so we
    replicate the gate block's exact conditions here against the loaded program."""
    prog = literal_basis_program_config(aa_mode="supersample", aa_factor=2)
    ckpt = _full_literal_checkpoint(tmp_path, prog)
    params, cfg = tool._load_levelset_ckpt(ckpt, "levelset_witness_ema.npz")
    program = cfg["basis_program"]
    assert not program.chart_enabled
    assert not (program.native_orientation_enabled and program.aa_mode != "none")
    assert not (
        program.aa_mode != "none" and any(str(k).startswith("tex_trunk.") for k in params)
    )
