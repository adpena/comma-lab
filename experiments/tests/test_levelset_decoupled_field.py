# SPDX-License-Identifier: MIT
"""v8 B1 DECOUPLED-FIELD mode — trainer-integration + DSL-lever tests.

Covers the wiring of the per-class decoupled partition head into the LEVELSET trainer
(``experiments/train_levelset_witness_realized_through_R_mlx.py``) + its DSL lever + resume config:

  * THE #1 GATE (safety-critical): ``--decoupled-field`` OFF (default) is BYTE-IDENTICAL to the
    shared-head witness — the same param KEYS, no ``decoupled_head`` attr, byte-identical compose path
    (so a crash-resume of the LIVE run onto merged code is safe).
  * ON: the decoupled head params appear; every forward path (sdf / __call__ / call_batch /
    call_margin) routes through the decoupled fields and runs.
  * composition forward equals shared-mode when classes tied (the increment-1 equivalence anchor,
    realized on the actual witness module).
  * resume round-trip: decoupled params are byte-close-loadable (tree_flatten -> numpy -> model.update).
  * resume config: the ``__cfg_decoupled_field*`` provenance scalars round-trip.
  * DSL: the ``DecoupledField`` lever compiles to the real trainer flags, validates, and the
    lever_registry maps them (no orphaned flag).

CPU-ONLY: never sets mx.gpu (the live run owns it).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_levelset_witness_realized_through_R_mlx as T  # noqa: E402

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)  # NEVER gpu: the live run owns it.
from mlx.utils import tree_flatten  # noqa: E402


def _common(**over):
    base = dict(
        num_pairs=3, in_feat=6, hidden_dim=16, n_hidden=2, mod_dim=8, n_classes=5,
        activation="relu", softmax_temp=0.1, wire_w0=30.0, wire_s0=1.0, hosc_beta=1.0,
        hosc_omega=1.0, chroma=True, render_h=8, render_w=8,
    )
    base.update(over)
    return base


def _param_keys(model):
    return {k for k, _ in tree_flatten(model.parameters())}


# =========================================================================== #1 GATE: OFF byte-identity
def test_off_is_byte_identical_param_keys_and_no_attr():
    off = T.build_levelset_rgb_witness(**_common())
    base = T.build_levelset_rgb_witness(**_common())
    assert _param_keys(off) == _param_keys(base)
    assert getattr(off, "decoupled_head", None) is None
    # no 'decoupled_head' key leaked into the param tree
    assert not any(k.startswith("decoupled_head") for k in _param_keys(off))


def test_off_compose_path_is_the_shared_head():
    """OFF: __call__ must produce the SAME output as the pre-B1 shared out_sdf compose (phi=None)."""
    m = T.build_levelset_rgb_witness(**_common())
    coord = mx.array(np.random.default_rng(0).standard_normal((16, 6)).astype(np.float32))
    # the shared reference: compose over out_sdf(trunk(...)) directly
    h = m._trunk(coord, 0)
    ref = m._compose_rgb(h)               # phi=None -> out_sdf(h)
    got = m(coord, 0)
    assert np.array_equal(np.asarray(ref), np.asarray(got))


# =========================================================================== ON: params + forwards
def test_on_adds_decoupled_head_params():
    on = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                      decoupled_field_layers=2, **_common())
    base = T.build_levelset_rgb_witness(**_common())
    extra = _param_keys(on) - _param_keys(base)
    assert extra == {
        "decoupled_head.w_in", "decoupled_head.b_in", "decoupled_head.w_film",
        "decoupled_head.w_hid", "decoupled_head.b_hid", "decoupled_head.w_out", "decoupled_head.b_out",
    }
    assert getattr(on, "decoupled_head", None) is not None


def test_on_all_forward_paths_route_through_decoupled_head():
    on = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                      decoupled_field_layers=2, **_common())
    coord = mx.array(np.random.default_rng(0).standard_normal((64, 6)).astype(np.float32))
    phi = on.sdf(coord, 0)
    assert phi.shape == (64, 5)
    rgb = on(coord, 0)
    assert rgb.shape == (64, 3)
    rb = on.call_batch(coord, mx.array([0, 2, 4]))
    assert rb.shape == (3, 64, 3)
    mg = on.call_margin(coord, 0)
    assert mg.shape == (64,)
    mx.eval(phi, rgb, rb, mg)
    # sdf must equal the decoupled head's own phi_single (routed, not the shared out_sdf)
    ref = on.decoupled_head.phi_single(coord, on.code[0])
    assert np.allclose(np.asarray(phi), np.asarray(ref), atol=1e-6)


def test_call_batch_row_equals_single_call():
    on = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                      decoupled_field_layers=2, **_common())
    coord = mx.array(np.random.default_rng(0).standard_normal((20, 6)).astype(np.float32))
    idx = [0, 2, 4]
    batch = np.asarray(on.call_batch(coord, mx.array(idx)))
    for row, ci in enumerate(idx):
        single = np.asarray(on(coord, ci))
        assert np.allclose(batch[row], single, atol=1e-5)


# =========================================================================== tied equivalence anchor
def test_tied_fields_give_constant_class0_partition_on_the_witness():
    """With the witness's decoupled fields tied, argmax_c phi_c == class 0 everywhere (the
    composition reduces to shared-mode) — realized on the actual module."""
    on = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                      decoupled_field_layers=2, **_common())
    # tie the fields: copy class-0 slice onto every class in every decoupled param
    dh = on.decoupled_head
    for name in ("w_in", "b_in", "w_film", "w_hid", "b_hid", "w_out", "b_out"):
        arr = np.asarray(getattr(dh, name))
        tied = np.repeat(arr[:1], arr.shape[0], axis=0)
        setattr(dh, name, mx.array(tied))
    mx.eval(dh.parameters())
    coord = mx.array(np.random.default_rng(0).standard_normal((30, 6)).astype(np.float32))
    phi = np.asarray(on.sdf(coord, 0))
    for kc in range(1, 5):
        assert np.allclose(phi[:, 0], phi[:, kc], atol=1e-5)
    assert np.array_equal(np.argmax(phi, axis=-1), np.zeros(30, dtype=int))


# =========================================================================== resume round-trip
def test_decoupled_params_are_byte_close_loadable():
    """Resume P0: serialize the decoupled head (tree_flatten -> numpy) and reload into a fresh ON
    model via model.update -> forward is identical (the checkpoint-resume contract)."""
    src = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                       decoupled_field_layers=2, **_common())
    coord = mx.array(np.random.default_rng(0).standard_normal((16, 6)).astype(np.float32))
    ref = np.asarray(src(coord, 1))
    flat = {k: np.asarray(v) for k, v in tree_flatten(src.parameters())}
    # fresh model with the SAME arch, DIFFERENT init -> load the saved params
    dst = T.build_levelset_rgb_witness(decoupled_field=True, decoupled_field_hidden=8,
                                       decoupled_field_layers=2, **_common())
    from mlx.utils import tree_unflatten
    dst.update(tree_unflatten([(k, mx.array(v)) for k, v in flat.items()]))
    mx.eval(dst.parameters())
    got = np.asarray(dst(coord, 1))
    assert np.array_equal(ref, got)


def test_resume_cfg_scalars_round_trip():
    class Perm(SimpleNamespace):
        def __getattr__(self, k):
            return dict(decoupled_field=True, decoupled_field_hidden=48,
                        decoupled_field_layers=3, mod_dim=32).get(k, 0)
    out = T._build_resume_state_arrays({}, {}, {}, args=Perm(), epoch=5, in_feat=6,
                                       recent_losses=[], provenance={}, resume_registry=None)
    assert int(out["__cfg_decoupled_field"]) == 1
    assert int(out["__cfg_decoupled_field_hidden"]) == 48
    assert int(out["__cfg_decoupled_field_layers"]) == 3


def test_resume_cfg_off_defaults():
    class Perm(SimpleNamespace):
        def __getattr__(self, k):
            return dict(mod_dim=32).get(k, 0)
    out = T._build_resume_state_arrays({}, {}, {}, args=Perm(), epoch=1, in_feat=6,
                                       recent_losses=[], provenance={}, resume_registry=None)
    assert int(out["__cfg_decoupled_field"]) == 0  # OFF by default


# =========================================================================== fail-closed incompat guard
def test_decoupled_field_off_has_no_incompatibilities():
    args = SimpleNamespace(decoupled_field=False, head="etf", margin_field_head_weight=1.0,
                           head_offset_solver="menon")
    assert T._decoupled_field_incompatibilities(args) == []  # OFF => never blocks


def test_decoupled_field_softmax_head_is_compatible():
    args = SimpleNamespace(decoupled_field=True, head="softmax", margin_field_head_weight=0.0,
                           head_offset_solver="off")
    assert T._decoupled_field_incompatibilities(args) == []


@pytest.mark.parametrize("over,needle", [
    (dict(head="etf"), "--head etf"),
    (dict(head="additive-margin"), "--head additive-margin"),
    (dict(margin_field_head_weight=0.5), "--margin-field-head-weight"),
    (dict(head_offset_solver="ot_newton"), "--head-offset-solver"),
])
def test_decoupled_field_refuses_shared_head_levers(over, needle):
    base = dict(decoupled_field=True, head="softmax", margin_field_head_weight=0.0,
                head_offset_solver="off")
    base.update(over)
    incompat = T._decoupled_field_incompatibilities(SimpleNamespace(**base))
    assert any(needle in s for s in incompat), incompat


# =========================================================================== DSL lever leg
def test_dsl_lever_compiles_to_real_flags():
    from tac.witness_dsl import DecoupledField, Lever
    lv = DecoupledField(field_hidden=48, field_layers=3, window=100)
    assert isinstance(lv, Lever)
    assert lv.name == "decoupled_field"
    assert lv.overrides["--decoupled-field"] is True
    assert lv.overrides["--decoupled-field-hidden"] == 48
    assert lv.overrides["--decoupled-field-layers"] == 3
    assert lv.epochs_delta == 100


def test_dsl_lever_flags_are_real_trainer_flags_no_orphan():
    from tac.witness_dsl.curriculum_dsl import real_trainer_flags
    from tac.witness_dsl import DecoupledField
    real = real_trainer_flags(Path(T.__file__))
    for flag in DecoupledField().overrides:
        assert flag in real, f"{flag} is not a real trainer argparse flag (never-invent-flags)"


def test_lever_registry_maps_decoupled_field_no_unmapped():
    from tac.witness_dsl.lever_registry import lever_factories, completeness
    facs = lever_factories()
    assert "DecoupledField" in facs
    assert set(facs["DecoupledField"]) == {
        "--decoupled-field", "--decoupled-field-hidden", "--decoupled-field-layers"}
    c = completeness(Path(T.__file__))
    assert not [f for f in c.unmapped if "decoupled" in f]
    assert not [f for f in c.stale if "decoupled" in f]
