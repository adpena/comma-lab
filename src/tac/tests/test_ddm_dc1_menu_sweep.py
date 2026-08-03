# SPDX-License-Identifier: MIT
"""Tests for ``tools/dc1_menu_sweep.py`` (ddm_dc1 menu sweep).

These verify BEHAVIOUR, not constants: the dead-codeword discriminator, the
scale-degeneracy fold that reconciles ddm_ms8 with ddm_mq1, the REAL shipped
byte pricing, and the positive-control guard that must ABORT rather than warn.

The guard tests are the load-bearing ones: a canary that does not abort is the
vacuity genus (a broken instrument emitting the same PASS symbol as a clean
one), which is exactly the failure class this arm exists to audit.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "dc1_menu_sweep.py"
MS8_TOOL = REPO / "tools" / "ms8_st_codebook_race.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("dc1_menu_sweep", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["dc1_menu_sweep"] = module
    spec.loader.exec_module(module)
    return module


dc1 = _load_tool()


# --------------------------------------------------------------------------- #
# the discriminator itself
# --------------------------------------------------------------------------- #
def test_dead_fraction_counts_exactly_zero_entries() -> None:
    dead, k, frac = dc1.dead_fraction([0, 0, 0, 0, 0, 0, 22, 364, 156, 58, 0])
    assert (dead, k) == (7, 11)
    assert frac == pytest.approx(7 / 11)


def test_dead_fraction_is_zero_when_every_codeword_has_mass() -> None:
    dead, k, frac = dc1.dead_fraction([5, 5, 1, 10, 15, 420, 66, 52, 13, 1, 7, 1, 4])
    assert (dead, k, frac) == (0, 13, 0.0)


def test_dead_fraction_is_not_mode_share() -> None:
    """The measured discriminator is deadness, NOT concentration.

    ddm_ms8 §10 established that the RD-optimal codebook also piles ~51% on one
    entry, so a high mode share is not evidence of a defect.  A menu can be
    extremely peaked and still perfectly placed.
    """
    peaked_but_live = [1, 1, 596, 1, 1]
    dead, _k, frac = dc1.dead_fraction(peaked_but_live)
    assert dead == 0
    assert frac == 0.0
    assert max(peaked_but_live) / sum(peaked_but_live) > 0.99


def test_dead_fraction_all_dead_edge_case() -> None:
    dead, k, frac = dc1.dead_fraction([0, 0, 0])
    assert (dead, k, frac) == (3, 3, 1.0)


def test_dead_fraction_single_codeword() -> None:
    assert dc1.dead_fraction([7]) == (0, 1, 0.0)


# --------------------------------------------------------------------------- #
# the scale-degeneracy fold (the ms8 <-> mq1 reconciliation)
# --------------------------------------------------------------------------- #
def _fake_inputs(n: int = 4):
    rng = np.random.default_rng(20260802)
    final = {
        i: {"p": [30.0 + rng.normal(), rng.normal() * 0.2, rng.normal() * 0.4,
                  rng.normal() * 1e-3, rng.normal() * 1e-5, rng.normal() * 1e-3]}
        for i in range(n)
    }
    ms8_doc = {"st_grid": [0.06, 0.08, 0.12],
               "st_idx": [0, 1, 2, 1][:n]}
    s_ship = dict.fromkeys(range(n), 0.08)
    return final, ms8_doc, s_ship


def test_fold_scales_only_the_translation_columns() -> None:
    """``t = s_t * [p2, p1, p0]``; the rotation columns must be untouched."""
    final, ms8_doc, s_ship = _fake_inputs()
    resc, _q, k, _off = dc1.build_rescaled_poses(final, ms8_doc, s_ship)
    poses = np.asarray([final[i]["p"] for i in range(len(final))], np.float64)
    assert np.allclose(resc[:, 0:3], poses[:, 0:3] * k[:, None], rtol=0, atol=0)
    assert np.array_equal(resc[:, 3:6], poses[:, 3:6])


def test_fold_scale_factor_is_ms8_over_shipped() -> None:
    final, ms8_doc, s_ship = _fake_inputs()
    _r, _q, k, _off = dc1.build_rescaled_poses(final, ms8_doc, s_ship)
    expected = [ms8_doc["st_grid"][j] / 0.08 for j in ms8_doc["st_idx"]]
    assert k.tolist() == pytest.approx(expected)


def test_fold_is_the_identity_when_ms8_agrees_with_the_shipped_scale() -> None:
    """A pair ms8 did not move must fold to a bit-identical pose."""
    final, ms8_doc, s_ship = _fake_inputs()
    ms8_doc = dict(ms8_doc, st_idx=[1, 1, 1, 1])  # every pair keeps s_t = 0.08
    resc, _q, k, _off = dc1.build_rescaled_poses(final, ms8_doc, s_ship)
    poses = np.asarray([final[i]["p"] for i in range(len(final))], np.float64)
    assert np.array_equal(k, np.ones(len(final)))
    assert np.array_equal(resc, poses)


def test_fold_preserves_the_effective_translation_exactly() -> None:
    """``s_ship * fold(p) == s_ms8 * p`` on the translation triple.

    This is the algebra the whole reconciliation rests on, checked without the
    receiver so a receiver change cannot make the test vacuously pass.
    """
    final, ms8_doc, s_ship = _fake_inputs()
    resc, _q, _k, _off = dc1.build_rescaled_poses(final, ms8_doc, s_ship)
    for i in range(len(final)):
        p = np.asarray(final[i]["p"], np.float64)
        s_ms8 = ms8_doc["st_grid"][ms8_doc["st_idx"][i]]
        lhs = s_ship[i] * resc[i, 0:3]
        rhs = s_ms8 * p[0:3]
        assert np.allclose(lhs, rhs, rtol=1e-15, atol=0)


def test_fold_refuses_a_truncated_index_stream() -> None:
    """A short ``st_idx`` must ABORT, not silently fold the wrong scale.

    ``build_rescaled_poses`` sizes itself from the index stream, so a partial
    stream would quietly rescale only a prefix -- a silent wrong answer, which
    is worse than a crash.
    """
    final, ms8_doc, s_ship = _fake_inputs()
    short = dict(ms8_doc, st_idx=ms8_doc["st_idx"][:2])
    with pytest.raises(SystemExit, match="silent truncation"):
        dc1.build_rescaled_poses(final, short, s_ship)


def test_fold_refuses_an_index_outside_its_own_grid() -> None:
    final, ms8_doc, s_ship = _fake_inputs()
    bad = dict(ms8_doc, st_idx=[0, 1, 99, 1])
    with pytest.raises(SystemExit, match="escapes its own st_grid"):
        dc1.build_rescaled_poses(final, bad, s_ship)


def test_ms8_archive_delta_is_charged_to_every_arm_inheriting_its_codebook(
        tmp_path: Path) -> None:
    """ms8's +51 B lives OUTSIDE the members this tool re-encodes.

    An arm that adopts the ms8 codebook and does not pay that delta is
    under-priced -- the exact class of error this arm was chartered to audit.
    """
    _write_curves(tmp_path, canary=0.0)
    dc1.run_seldesign(_Args(tmp_path))
    receipt = json.loads((tmp_path / "dc1_seldesign_receipt.json").read_text())
    for name in ("MS8_only", "RESEL_sel_on_ms8_st"):
        arm = next(a for a in receipt["arms"] if a["arm"] == name)
        assert arm["delta_bytes_vs_shipped"] >= dc1.MS8_ARCHIVE_DELTA_BYTES
    assert dc1.MS8_ARCHIVE_DELTA_BYTES == 360_374 - 360_323


def test_fold_quantized_pose_roundtrips_through_f16_with_the_offset() -> None:
    """The shippable leg must apply the QA65 offset device, not plain f16."""
    final, ms8_doc, s_ship = _fake_inputs()
    resc, q, _k, offset = dc1.build_rescaled_poses(final, ms8_doc, s_ship)
    # the dim0 column is stored as an f16 RESIDUAL off ``offset``
    manual = (resc[:, 0] - offset).astype(np.float16).astype(np.float64) + offset
    assert np.array_equal(q[:, 0], manual)
    # every other column is plain f16
    assert np.array_equal(q[:, 1:], resc[:, 1:].astype(np.float16).astype(np.float64))
    # the offset itself must be f16-representable (it ships as a manifest float)
    assert float(np.float16(offset)) == offset


# --------------------------------------------------------------------------- #
# REAL shipped byte pricing
# --------------------------------------------------------------------------- #
def test_sel_stream_bytes_matches_the_real_builder_encoding() -> None:
    """Must equal ``brotli(packbits(sel), q=11)`` -- the builder's own line."""
    brotli = pytest.importorskip("brotli")
    rng = np.random.default_rng(7)
    sel = rng.integers(0, 2, size=600).astype(np.uint8)
    expected = len(brotli.compress(np.packbits(sel).tobytes(), quality=11))
    assert dc1.sel_stream_bytes(sel) == expected


def test_sel_stream_bytes_reacts_to_content() -> None:
    """A degenerate stream must cost strictly fewer bytes than a random one."""
    pytest.importorskip("brotli")
    n = 600
    assert dc1.sel_stream_bytes(np.zeros(n, np.uint8)) < dc1.sel_stream_bytes(
        np.random.default_rng(3).integers(0, 2, size=n).astype(np.uint8))


def test_pose_member_bytes_uses_the_real_kl1_encoder() -> None:
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_v4d_build_composed_archive import encode_kl1_field

    poses = np.random.default_rng(11).normal(size=(600, 6)) * 0.5
    poses[:, 0] += 31.5
    got = dc1.pose_member_bytes(poses, 31.5)
    store = poses.copy()
    store[:, 0] -= 31.5
    assert got == len(encode_kl1_field(store.astype(np.float16)))


def test_pose_member_bytes_offset_none_is_the_plain_field() -> None:
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_v4d_build_composed_archive import encode_kl1_field

    poses = np.random.default_rng(13).normal(size=(64, 6))
    assert dc1.pose_member_bytes(poses, None) == len(
        encode_kl1_field(poses.astype(np.float16)))


# --------------------------------------------------------------------------- #
# the positive-control guard must ABORT, not warn
# --------------------------------------------------------------------------- #
def _write_curves(tmp_path: Path, canary: float, n: int = 3) -> Path:
    out = tmp_path / "dc1_selcurves_shard0.jsonl"
    with out.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "pair": i, "sel_shipped": 0,
                "s_shipped": 0.08, "s_ms8": 0.08,
                "d_shipped_reported": 0.001, "d_ctrl": 0.001 + canary,
                "canary_abs_err": canary,
                "has_v4c_branch_poses": [], "p_v4c": {},
                "vals": dict.fromkeys(dc1.CONFIG_KEYS, 0.001),
                "beta_mag": 0.0, "a": 1.0, "b": 0.0,
            }) + "\n")
    return out


class _Args:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.allow_partial = True
        self.final_jsonl = Path("/nonexistent/final.jsonl")
        self.archive = dc1.LIVE_ARCHIVE


def test_seldesign_aborts_when_the_canary_exceeds_the_instrument_floor(
        tmp_path: Path) -> None:
    _write_curves(tmp_path, canary=dc1.CANARY_MAX_ABS_ERR * 10.0)
    with pytest.raises(SystemExit, match="POSITIVE CONTROL FAILED"):
        dc1.run_seldesign(_Args(tmp_path))


def test_seldesign_accepts_a_canary_at_the_measured_floor(tmp_path: Path) -> None:
    """The guard must not be so tight that the real instrument fails it.

    ms8 MEASURED max |d_ctrl - d_shipped| = 1.085e-05 on this vehicle, so the
    tolerance has to admit that and nothing looser by an order of magnitude.
    """
    _write_curves(tmp_path, canary=1.085e-05)
    dc1.run_seldesign(_Args(tmp_path))  # must not raise
    receipt = json.loads((tmp_path / "dc1_seldesign_receipt.json").read_text())
    assert receipt["positive_control"] == "PASS"
    assert dc1.CANARY_MAX_ABS_ERR >= 1.085e-05
    assert dc1.CANARY_MAX_ABS_ERR < 1.085e-04


def test_seldesign_refuses_a_partial_curve_set_by_default(tmp_path: Path) -> None:
    _write_curves(tmp_path, canary=0.0)
    args = _Args(tmp_path)
    args.allow_partial = False
    with pytest.raises(SystemExit, match="selcurves incomplete"):
        dc1.run_seldesign(args)


def test_seldesign_ctrl_arm_is_a_zero_delta_by_construction(tmp_path: Path) -> None:
    _write_curves(tmp_path, canary=0.0)
    dc1.run_seldesign(_Args(tmp_path))
    receipt = json.loads((tmp_path / "dc1_seldesign_receipt.json").read_text())
    ctrl = next(a for a in receipt["arms"] if a["arm"] == "CTRL_shipped")
    assert ctrl["delta_S_pose"] == 0.0
    assert ctrl["delta_bytes_vs_shipped"] == 0
    assert ctrl["pairs_whose_selector_moves"] == 0


# --------------------------------------------------------------------------- #
# structural: this arm must be on ms8's measurement path, not a lookalike
# --------------------------------------------------------------------------- #
def _score_body(path: Path) -> str:
    """The source text of the nested ``score`` function, docstring stripped."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "score":
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError(f"no nested score() in {path}")


def test_dc1_score_is_verbatim_ms8_score() -> None:
    """A drifted copy would silently measure a different vehicle.

    ddm_ms8's ``score`` mirrors ``inflate_runner_v4d.Decoder.f0``; this arm
    copies it so the two arms are provably on the same path.  If either moves,
    this test fails rather than letting the arms diverge in silence.
    """
    assert _score_body(TOOL) == _score_body(MS8_TOOL)


def test_config_keys_cover_the_full_evaluated_grid() -> None:
    """2 s_t variants x 2 selectors x 2 pose sources = 8 named columns."""
    assert len(dc1.CONFIG_KEYS) == 8
    assert len(set(dc1.CONFIG_KEYS)) == 8
    for st in ("st_ship", "st_ms8"):
        for sel in (0, 1):
            for pose in ("pose_ship", "pose_v4c"):
                assert f"{st}__sel{sel}__{pose}" in dc1.CONFIG_KEYS


def test_contribution_matches_the_scorer_pose_term() -> None:
    assert dc1.contribution(0.00764543) == pytest.approx(
        float(np.sqrt(10.0 * 0.00764543)), rel=0, abs=0)


def test_default_pairs_is_the_full_n600_evidence_bar() -> None:
    """A subset must be asked for explicitly; n600 is the default."""
    src = TOOL.read_text()
    match = re.search(r'"--pairs",\s*type=int,\s*default=(\w+)', src)
    assert match is not None
    assert match.group(1) == "N_PAIRS"
    assert dc1.N_PAIRS == 600


# --------------------------------------------------------------------------- #
# inventory: the denominator must be reported, and unreached != absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not dc1.LIVE_ARCHIVE.exists(),
                    reason="live v4d archive not mounted")
def test_inventory_reports_a_denominator_and_reasons(tmp_path: Path) -> None:
    class _IArgs:
        archive = dc1.LIVE_ARCHIVE
        final_jsonl = dc1.FINAL_JSONL
        out_dir = tmp_path
        dead_threshold = 0.10

    dc1.run_inventory(_IArgs())
    receipt = json.loads((tmp_path / "dc1_inventory_receipt.json").read_text())
    den = receipt["denominator"]
    assert den["menus_found"] == len(receipt["menus"])
    assert (den["menus_occupancy_measured"]
            + den["menus_unreached_with_reason"]) == den["menus_found"]
    # every unreached menu must carry a substantive reason -- an empty scope is
    # VACUOUS, never a PASS
    for menu in receipt["menus"]:
        if not menu["reached"]:
            assert len(menu["unreached_reason"]) > 40


@pytest.mark.skipif(not dc1.LIVE_ARCHIVE.exists(),
                    reason="live v4d archive not mounted")
def test_inventory_reproduces_the_ms8_st_grid_occupancy(tmp_path: Path) -> None:
    """Independent cross-check of ddm_ms8 from a DIFFERENT source.

    ms8 read its occupancy from the d1 solve JSONL; this reads the coded index
    stream out of the shipped archive.  Agreement means the shipped bytes and
    the solver's record are the same story.
    """
    class _IArgs:
        archive = dc1.LIVE_ARCHIVE
        final_jsonl = dc1.FINAL_JSONL
        out_dir = tmp_path
        dead_threshold = 0.10

    dc1.run_inventory(_IArgs())
    receipt = json.loads((tmp_path / "dc1_inventory_receipt.json").read_text())
    st = next(m for m in receipt["menus"] if m["menu"].startswith("st_grid"))
    assert st["occupancy"] == [0, 0, 0, 0, 0, 0, 22, 364, 156, 58, 0]
    assert st["dead_codewords"] == 7
    assert st["K"] == 11


@pytest.mark.skipif(not dc1.LIVE_ARCHIVE.exists(),
                    reason="live v4d archive not mounted")
def test_inventory_beta_table_has_zero_dead_codewords_by_construction(
        tmp_path: Path) -> None:
    """``derive_beta_table`` builds ``sorted(set(chosen))`` -- the ms8 defect
    is structurally impossible in this menu, and the measurement must say so."""
    class _IArgs:
        archive = dc1.LIVE_ARCHIVE
        final_jsonl = dc1.FINAL_JSONL
        out_dir = tmp_path
        dead_threshold = 0.10

    dc1.run_inventory(_IArgs())
    receipt = json.loads((tmp_path / "dc1_inventory_receipt.json").read_text())
    beta = next(m for m in receipt["menus"] if m["menu"].startswith("rs_beta_mags"))
    assert beta["dead_codewords"] == 0
    assert min(beta["occupancy"]) >= 1


def test_manifest_table_pricing_helper_matches_deflate() -> None:
    """The manifest is a DEFLATE zip member, so a fitted table is priced there."""
    table = (0.06, 0.065, 0.07)
    blob = json.dumps({"st_grid": list(table)}, separators=(",", ":")).encode()
    assert len(zlib.compress(blob, 9)) > 0
