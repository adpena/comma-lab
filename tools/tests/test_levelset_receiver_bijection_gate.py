# SPDX-License-Identifier: MIT
"""Tests for the receiver-consumption bijection gate (#417, NO-FAKE #8 class-fix).

Verifies the DRIFT-PROOF static extraction (reads the receiver source), the orphan
logic (counted-but-inert groups), the explicit allow-list waiver, the dynamic-access
degrade-to-advisory integrity guard, and — the load-bearing proof — that the gate
REFUSES a synthetic tex_trunk/decoupled_head archive while PASSING a valid witness.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# import the gate module directly (tools/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "levelset_receiver_bijection_gate", _TOOLS / "levelset_receiver_bijection_gate.py")
gate = importlib.util.module_from_spec(_spec)
sys.modules["levelset_receiver_bijection_gate"] = gate
_spec.loader.exec_module(gate)


# a tiny synthetic receiver source mirroring the real _INFLATE_PY P-access shapes
_SRC_OK = '''
def _in_proj_h0(P, feats, m):
    return feats @ P["in_proj.weight"].T + P["in_proj.bias"]
def _outputs_from_h0(P, h0, code_row, m):
    film = code_row @ P["film.weight"].T + P["film.bias"]
    for li in range(m["n_hidden"]):
        if has_pl:
            pl = code_row @ P["film_pl.%d.weight" % li].T + P["film_pl.%d.bias" % li]
        h = h0 @ P["hidden.%d.weight" % li].T + P["hidden.%d.bias" % li]
        if has_concat:
            h = h + code_row @ P["concat_pl.%d.weight" % li].T + P["concat_pl.%d.bias" % li]
    phi = h @ P["out_sdf.weight"].T + P["out_sdf.bias"]
    tex = h @ P["out_tex.weight"].T + P["out_tex.bias"]
    rgb = soft @ P["palette"] + tex
    return phi, rgb
'''

_SRC_DYNAMIC = '''
def f(P, names):
    for name in names:
        x = P[name]           # dynamic key -> unresolvable
    return P["in_proj.weight"]
'''


# ------------------------------------------------------- extraction ----------
def test_extract_exact_and_pattern_keys():
    v = gate.extract_consumed_vocabulary(_SRC_OK)
    assert not v.dynamic_access
    assert "in_proj.weight" in v.exact and "palette" in v.exact and "out_tex.bias" in v.exact
    # indexed layer params come through as (prefix, suffix) patterns
    assert ("hidden.", ".weight") in v.patterns
    assert ("film_pl.", ".bias") in v.patterns
    assert ("concat_pl.", ".weight") in v.patterns


def test_consumes_exact_and_indexed():
    v = gate.extract_consumed_vocabulary(_SRC_OK)
    assert v.consumes("in_proj.weight")
    assert v.consumes("hidden.0.weight") and v.consumes("hidden.11.bias")
    assert v.consumes("film_pl.3.weight")
    # NOT consumed: the trainer's optional deep texture head + the #417 offenders
    assert not v.consumes("out_tex_hidden.weight")
    assert not v.consumes("tex_trunk.w_tex")
    assert not v.consumes("decoupled_head.w_in")
    # a non-integer "index" must NOT match the %d pattern
    assert not v.consumes("hidden.foo.weight")


def test_dynamic_access_flagged():
    v = gate.extract_consumed_vocabulary(_SRC_DYNAMIC)
    assert v.dynamic_access is True


# --------------------------------------------------------- orphans -----------
def test_orphans_flags_unconsumed_only():
    v = gate.extract_consumed_vocabulary(_SRC_OK)
    counted = ["in_proj.weight", "hidden.0.weight", "out_sdf.bias",
               "tex_trunk.w_tex", "decoupled_head.w_in"]
    orphans = gate.orphaned_counted_params(counted, v)
    assert orphans == ["decoupled_head.w_in", "tex_trunk.w_tex"]


def test_allow_unconsumed_prefix_waiver():
    v = gate.extract_consumed_vocabulary(_SRC_OK)
    counted = ["in_proj.weight", "tex_trunk.w_tex", "tex_trunk.bias", "decoupled_head.w_in"]
    # waiving the "tex_trunk" prefix clears both tex_trunk.* orphans, not decoupled_head
    orphans = gate.orphaned_counted_params(counted, v, allow_unconsumed=["tex_trunk"])
    assert orphans == ["decoupled_head.w_in"]


# --------------------------------------------------- assert (fail-closed) ----
def test_assert_raises_on_orphan():
    with pytest.raises(gate.ReceiverBijectionError) as ei:
        gate.assert_receiver_bijection(
            ["in_proj.weight", "tex_trunk.w_tex"], _SRC_OK, context="unit")
    msg = str(ei.value)
    assert "tex_trunk.w_tex" in msg and "NO-FAKE #8" in msg and "INERT" in msg


def test_assert_passes_clean():
    # every counted group is consumed -> no raise, returns the vocabulary
    v = gate.assert_receiver_bijection(
        ["in_proj.weight", "in_proj.bias", "film.weight", "hidden.0.weight",
         "hidden.0.bias", "out_sdf.weight", "out_sdf.bias", "out_tex.weight",
         "out_tex.bias", "palette"], _SRC_OK, context="unit")
    assert "palette" in v.exact


def test_assert_waiver_allows_build():
    # with the waiver, the orphan is permitted (build proceeds)
    v = gate.assert_receiver_bijection(
        ["in_proj.weight", "tex_trunk.w_tex"], _SRC_OK,
        context="unit", allow_unconsumed=["tex_trunk"])
    assert not v.dynamic_access


def test_assert_dynamic_degrades_to_advisory():
    # dynamic P-access -> cannot prove orphans -> DO NOT raise, warn instead
    warnings: list[str] = []
    v = gate.assert_receiver_bijection(
        ["in_proj.weight", "tex_trunk.w_tex"], _SRC_DYNAMIC,
        context="unit", warn=warnings.append)
    assert v.dynamic_access is True
    assert warnings and "ADVISORY" in warnings[0]


# ------------------------------------------- real-source integration ---------
def test_real_receiver_source_consumed_set():
    """The ACTUAL shipped _INFLATE_PY consumes exactly the known witness param set.

    #417 FIX LANDED: the receiver now ALSO consumes tex_trunk / out_tex_h / decoupled_head (the
    forward mirrors the trainer MLX submodules, parity-gated in
    tools/tests/test_receiver_bijection_v753_v8_parity.py). Pre-fix this test asserted those groups
    were NOT consumed (the bug); it now asserts they ARE (the fix)."""
    bc = pytest.importorskip("levelset_byte_close_and_eval")
    v = gate.extract_consumed_vocabulary(bc._INFLATE_PY)
    assert not v.dynamic_access, "real receiver must be statically analyzable (full-strength gate)"
    for k in ("in_proj.weight", "film.weight", "out_sdf.weight", "out_tex.weight", "palette"):
        assert k in v.exact
    assert ("hidden.", ".weight") in v.patterns
    # #417: the formerly-orphaned groups are NOW consumed (fix half landed).
    for k in ("tex_trunk.w_tex", "tex_trunk.bias", "out_tex_h.weight", "out_tex_h.bias",
              "decoupled_head.w_in", "decoupled_head.w_film", "decoupled_head.w_out", "decoupled_head.b_out"):
        assert v.consumes(k), f"receiver must now consume {k} (#417 fix)"


def test_real_source_consumes_tex_trunk_archive_after_417_fix():
    """#417 fix half: a base_order carrying tex_trunk / decoupled_head now PASSES (consumed, not
    orphaned). The gate still REFUSES a genuinely-unknown group (drift protection preserved)."""
    bc = pytest.importorskip("levelset_byte_close_and_eval")
    valid = ["in_proj.weight", "in_proj.bias", "film.weight", "film.bias",
             "hidden.0.weight", "hidden.0.bias", "out_sdf.weight", "out_sdf.bias",
             "out_tex.weight", "out_tex.bias", "palette"]
    # valid witness passes
    gate.assert_receiver_bijection(valid, bc._INFLATE_PY, context="real-clean")
    # #417: the tex_trunk + decoupled_head groups are now CONSUMED -> no longer refused.
    gate.assert_receiver_bijection(
        [*valid, "tex_trunk.w_tex", "tex_trunk.bias", "out_tex_h.weight", "out_tex_h.bias",
         "decoupled_head.w_in", "decoupled_head.b_in", "decoupled_head.w_film", "decoupled_head.w_hid",
         "decoupled_head.b_hid", "decoupled_head.w_out", "decoupled_head.b_out"],
        bc._INFLATE_PY, context="real-v753-v8-consumed")
    # drift protection intact: a genuinely-unknown counted group is STILL refused.
    with pytest.raises(gate.ReceiverBijectionError):
        gate.assert_receiver_bijection([*valid, "some_future_unmirror.weight"], bc._INFLATE_PY,
                                       context="real-orphan")
