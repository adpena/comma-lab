"""ddm_op2 — tests for the two MEASURED defects `ddm_gd5` left owned-and-deferred.

OP2-1 (`--persist-optimizer-state`): all six ``save_checkpoint`` callsites passed
``opt_state_flat={}``, so NO checkpoint on disk carried optimizer state and every resume built a
fresh ``optim.Adam`` with both moments zeroed (#824 reset arm B). MEASURED price, from the
trainer's own ``optimizer_arm`` row: ``boundary_impulse_epochs_per_reset = 16.167`` — ~218 of a
666-epoch budget at ~13.5 thirty-minute boundaries.

OP2-2 (``ema_basis_held``): the flag was computed from the ``ema_decay`` match ALONE while the row
separately recorded ``parent_gate_basis`` and ``first_gate_basis``. It therefore stamped ``true``
on the window_02 boundary, where the parent reading came off LIVE weights and the child's off the
EMA SHADOW (MEASURED: 0.0157596 vs 0.5118894, a 32x apparent collapse) — certifying two different
objects as commensurable next to a ``boundary_dseg_delta`` of +0.496.

Discipline (CLAUDE.md NO-FAKE #2, "tests-verify-constants-not-behavior"): the load-bearing test
here is BEHAVIOURAL against the real ``mlx.optimizers.Adam`` — ``test_restore_reproduces_an_
uninterrupted_run_bit_identically`` compares actual trained parameters and would fail against any
``return canonical_markers`` stub. Its paired NEGATIVE control asserts the reset path really does
diverge, so a restore that silently did nothing cannot pass both.

No scorer claim is made anywhere here. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    OPT_STATE_SCALAR_KEYS,
    OptimizerStateRestoreError,
    ResumeGeometryMismatch,
    TR1Config,
    assert_resume_geometry_compatible,
    boundary_jump_row,
    build_argparser,
    load_checkpoint,
    no_opt_state,
    opt_state_param_path,
    optimizer_state_to_flat,
    restore_optimizer_state,
    save_checkpoint,
)


def _cfg(**kw) -> TR1Config:
    base = {
        "variant": "plain", "num_pairs": 4, "grid_downsample": 16, "code_width": 2,
        "renderer_width": 8, "token_quant_levels": 16, "seed": 3, "lotto_seed": 118,
        "lotto_mask_density_init": 0.5, "seg_form_start": "ce", "w_seg": 100.0, "lr": 1e-3,
        "batch_pairs": 2, "epochs": 2, "gate_every": 1, "ema_decay": 0.95,
        "ema_decay_provenance": "test", "token_temporal_mode": "shared_base",
        "token_ste": "round", "class_weight_lane": 1.0, "margin_target": 1.0}
    base.update(kw)
    return TR1Config(**base)


# ============================ OP2-1: the optimizer-state BUILD ============================

def _tiny():
    mx.random.seed(0)
    return nn.Linear(3, 2)


def _loss(m):
    return mx.sum(mx.square(m(mx.ones((1, 3)))))


def _steps(model, opt, n):
    vg = nn.value_and_grad(model, _loss)
    for _ in range(n):
        loss, grads = vg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)


def _params(model) -> np.ndarray:
    return np.concatenate([np.asarray(v).ravel()
                           for _, v in sorted(tree_flatten(model.trainable_parameters()))])


def test_restore_reproduces_an_uninterrupted_run_bit_identically():
    """THE positive control for OP2-1, against the real MLX Adam.

    A boundary WITH persisted moments must be a NO-OP on trained bytes; a boundary WITHOUT them
    must not be. Both legs are asserted, so a restore that silently did nothing fails the first
    and a broken reference fails the second.
    """
    straight = _tiny()
    _steps(straight, optim.Adam(learning_rate=0.01, bias_correction=False), 6)

    parent = _tiny()
    parent_opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    _steps(parent, parent_opt, 3)
    snap_params = {k: np.asarray(v) for k, v in tree_flatten(parent.trainable_parameters())}
    snap_opt = optimizer_state_to_flat(parent_opt)
    assert snap_opt, "a stepped optimizer must yield a non-empty flat state"

    def _child():
        m = _tiny()
        m.update(tree_unflatten([(k, mx.array(v)) for k, v in snap_params.items()]))
        return m

    restored = _child()
    restored_opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    row = restore_optimizer_state(restored_opt, restored, snap_opt)
    _steps(restored, restored_opt, 3)

    reset = _child()
    _steps(reset, optim.Adam(learning_rate=0.01, bias_correction=False), 3)

    # ARM C: the boundary is invisible in the trained bytes.
    assert float(np.max(np.abs(_params(straight) - _params(restored)))) == 0.0
    # ARM B (today's behaviour): the boundary is NOT invisible. Without this leg the test above
    # would also pass against a run that never diverged in the first place.
    assert float(np.max(np.abs(_params(straight) - _params(reset)))) > 1e-6

    assert row["moments_restored"] == 4          # weight.m/.v + bias.m/.v
    assert row["step_restored"] == 3
    assert row["learning_rate_restored"] is False
    assert row["score_claim"] is False


def test_restore_keeps_the_live_learning_rate_not_the_parents():
    """A window that changed --lr must NOT silently inherit the parent's (a silent-wrong of
    exactly the kind the resume-geometry guard exists to refuse)."""
    parent = _tiny()
    parent_opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    _steps(parent, parent_opt, 2)
    snap = optimizer_state_to_flat(parent_opt)

    child = _tiny()
    child_opt = optim.Adam(learning_rate=0.005, bias_correction=False)   # different lr
    row = restore_optimizer_state(child_opt, child, snap)
    assert float(child_opt.state["learning_rate"]) == pytest.approx(0.005)
    assert row["parent_learning_rate"] == pytest.approx(0.01)
    assert row["live_learning_rate"] == pytest.approx(0.005)
    assert row["learning_rate_differs"] is True


def test_restore_refuses_a_moment_free_payload():
    """A payload with only scalars would restore NOTHING while claiming arm C — fail closed."""
    m = _tiny()
    opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    with pytest.raises(OptimizerStateRestoreError, match="no per-parameter moments"):
        restore_optimizer_state(opt, m, {"step": np.array(3, dtype=np.uint64)})


def test_restore_refuses_moments_the_live_optimizer_does_not_have():
    m = _tiny()
    opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    with pytest.raises(OptimizerStateRestoreError, match="absent from the live optimizer"):
        restore_optimizer_state(opt, m, {"ghost.m": np.zeros((2, 3), dtype=np.float32),
                                         "ghost.v": np.zeros((2, 3), dtype=np.float32)})


def test_restore_names_params_with_no_checkpointed_moment():
    """A lever introduced since the parent starts at zero moment BY DESIGN — but it is NAMED,
    never silent (the default-off-is-orphan discipline applied to resume state)."""
    parent = _tiny()
    popt = optim.Adam(learning_rate=0.01, bias_correction=False)
    _steps(parent, popt, 2)
    snap = {k: v for k, v in optimizer_state_to_flat(popt).items() if not k.startswith("bias.")}
    child = _tiny()
    copt = optim.Adam(learning_rate=0.01, bias_correction=False)
    row = restore_optimizer_state(copt, child, snap)
    assert row["moments_missing_start_at_zero"] == ["bias.m", "bias.v"]


def test_optimizer_state_to_flat_is_empty_before_any_moment_exists():
    """A freshly constructed optimizer carries only scalars; persisting that would be a
    truthful-looking checkpoint that restores nothing."""
    opt = optim.Adam(learning_rate=0.01, bias_correction=False)
    assert optimizer_state_to_flat(opt) == {}


def test_opt_state_param_path_knows_the_mlx_convention():
    assert opt_state_param_path("tokens_base.m") == "tokens_base"
    assert opt_state_param_path("layers.0.weight.v") == "layers.0.weight"
    for scalar in OPT_STATE_SCALAR_KEYS:
        assert opt_state_param_path(scalar) is None


def test_no_opt_state_rejects_a_placeholder_reason():
    assert no_opt_state("a substantive stated reason") == {}
    for bad in ("", "   ", "n/a"):
        with pytest.raises(ValueError):
            no_opt_state(bad)


# ------- the payload goes THROUGH the resume guard, never around it -------

_MODEL_SHAPES = {"tokens_base": (12, 16, 4), "head": (8,)}
_CKPT_SHAPES = dict(_MODEL_SHAPES)


def test_guard_is_byte_identical_when_no_opt_shapes_are_passed():
    """Default None ⇒ pre-OP2-1 behaviour exactly, so no existing caller changes."""
    assert assert_resume_geometry_compatible(_CKPT_SHAPES, _MODEL_SHAPES) == []
    assert assert_resume_geometry_compatible(_CKPT_SHAPES, _MODEL_SHAPES, None) == []


def test_guard_refuses_a_wrong_shaped_optimizer_moment():
    """The ds16-moments-into-ds32-model case: params could agree and the MOMENTS still not.
    ``optimizer.state`` assignment is the same silent-reshape surface ``Module.update`` is."""
    with pytest.raises(ResumeGeometryMismatch, match=r"opt::tokens_base\.m"):
        assert_resume_geometry_compatible(
            _CKPT_SHAPES, _MODEL_SHAPES, {"tokens_base.m": (24, 32, 4)})


def test_guard_refuses_an_orphaned_optimizer_moment():
    with pytest.raises(ResumeGeometryMismatch, match="no model param"):
        assert_resume_geometry_compatible(
            _CKPT_SHAPES, _MODEL_SHAPES, {"removed_lever.v": (4,)})


def test_guard_exempts_geometry_free_scalars():
    assert assert_resume_geometry_compatible(
        _CKPT_SHAPES, _MODEL_SHAPES,
        {"step": (), "learning_rate": (), "tokens_base.m": (12, 16, 4)}) == []


def test_save_load_roundtrip_carries_opt_keys_and_omits_them_when_absent(tmp_path):
    """Against the REAL producer/consumer pair, not a hand-built npz."""
    from experiments.train_tr1_partition_renderer_mlx import build_module

    cfg = _cfg()
    model = build_module(cfg)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=cfg.lr, bias_correction=False)
    opt.init(model.trainable_parameters())
    mx.eval(opt.state)
    flat = optimizer_state_to_flat(opt)
    assert flat, "init() must materialize per-parameter moments"

    with_opt = tmp_path / "with_opt.npz"
    save_checkpoint(with_opt, model=model, ema={}, opt_state_flat=flat, epoch=5,
                    stage="seg_trunk_tau", cfg=cfg, telemetry_tail=[])
    files = list(np.load(with_opt, allow_pickle=False).files)
    assert sum(k.startswith("opt::") for k in files) == len(flat)
    st = load_checkpoint(with_opt, build_module(cfg))
    assert set(st["opt_flat"]) == set(flat)

    # FLAG-OFF byte-shape invariant: zero opt:: keys, i.e. the pre-OP2-1 checkpoint exactly.
    without = tmp_path / "without_opt.npz"
    save_checkpoint(without, model=model, ema={}, opt_state_flat=no_opt_state(
        "flag off: checkpoint bytes stay identical to every pre-OP2-1 run"),
        epoch=5, stage="seg_trunk_tau", cfg=cfg, telemetry_tail=[])
    assert not any(k.startswith("opt::")
                   for k in np.load(without, allow_pickle=False).files)


def test_persist_flag_is_args_only_and_defaults_off():
    """The sealed-ticket invariant: the flag must NOT be a TR1Config field, or config_hash and
    ema_decay would move underneath a LIVE chain. Compiled against the ACTUAL argparse
    (never-invent-flags), not a hand-written flag list."""
    required = ["--variant", "plain", "--out-dir", "/tmp/ddm_op2_argparse_probe"]
    assert build_argparser().parse_args(required).persist_optimizer_state == "off"
    assert build_argparser().parse_args(
        [*required, "--persist-optimizer-state", "on"]).persist_optimizer_state == "on"
    assert not any("persist" in f for f in TR1Config.__dataclass_fields__)


def test_config_hash_is_invariant_to_the_persist_flag():
    """Directly: two identical configs hash the same and no persist field exists to perturb it."""
    assert _cfg().config_hash() == _cfg().config_hash()
    assert "persist_optimizer_state" not in TR1Config.__dataclass_fields__


# ============================ OP2-2: ema_basis_held ============================

_SHADOW_TAIL = [
    {"event": "a1_gate", "epoch": 940, "realized_gate_dseg_mean": 0.00430,
     "gate_params": "ema_shadow"},
    {"event": "a1_gate", "epoch": 945, "realized_gate_dseg_mean": 0.004201253255208332,
     "gate_params": "ema_shadow"},
]
#: The MEASURED window_01 tail: a FRESH window gates on LIVE weights (global_step below the EMA
#: warmup), so its last reading is ``live_ema_warmup``.
_LIVE_TAIL = [
    {"event": "a1_gate", "epoch": 44, "realized_gate_dseg_mean": 0.01575964,
     "gate_params": "live_ema_warmup"},
]
_DECAY = 0.9999199199199199


def _shadow_gate(epoch=949, dseg=0.00405):
    return {"event": "a1_gate", "epoch": epoch, "realized_gate_dseg_mean": dseg,
            "gate_params": "ema_shadow"}


def test_ema_basis_held_is_false_on_the_measured_window_02_boundary():
    """THE regression test for OP2-2, on the exact MEASURED readings.

    window_01 ep44 read 0.0157596 off LIVE weights; window_02 ep49 read 0.5118894 off the EMA
    SHADOW. Pre-fix this row stamped ``ema_basis_held: true`` beside a +0.496 delta.
    """
    row = boundary_jump_row(_LIVE_TAIL, _DECAY, _DECAY, 47,
                            _shadow_gate(49, 0.51188942), "B")
    assert row["parent_gate_basis"] == "live_ema_warmup"
    assert row["first_gate_basis"] == "ema_shadow"
    assert row["boundary_dseg_delta"] == pytest.approx(0.49612978, abs=1e-8)
    assert row["ema_basis_held"] is False          # <- the fix
    assert row["ema_decay_held"] is True           # the decay leg alone still holds ...
    assert row["gate_basis_held"] is False         # ... and the basis leg is what failed


def test_ema_basis_held_stays_true_when_both_legs_hold():
    row = boundary_jump_row(_SHADOW_TAIL, _DECAY, _DECAY, 947, _shadow_gate(), "Bprime")
    assert row["ema_basis_held"] is True
    assert row["ema_decay_held"] is True and row["gate_basis_held"] is True


def test_ema_basis_held_is_false_when_only_the_decay_drifts():
    """The original condition must not be weakened by the new one."""
    row = boundary_jump_row(_SHADOW_TAIL, 0.99993383, _DECAY, 947, _shadow_gate(), "B")
    assert row["ema_basis_held"] is False
    assert row["ema_decay_held"] is False and row["gate_basis_held"] is True


def test_ema_basis_held_fails_closed_on_an_unverifiable_basis():
    """An absent basis field cannot CERTIFY commensurability — it can only fail to."""
    tail = [{"event": "a1_gate", "epoch": 945, "realized_gate_dseg_mean": 0.0042}]
    row = boundary_jump_row(tail, _DECAY, _DECAY, 947, _shadow_gate(), "B")
    assert row["gate_basis_held"] is False and row["ema_basis_held"] is False


def test_caveat_names_both_legs():
    """A reader of the row must not have to re-derive which conditions 'commensurable' means."""
    row = boundary_jump_row(_SHADOW_TAIL, _DECAY, _DECAY, 947, _shadow_gate(), "B")
    assert "ema_decay_held" in row["caveat"] and "gate_basis_held" in row["caveat"]


# ==================== the SECOND LANDING: the gate that refuses re-introduction ====================

def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


_BARE = ("def save_checkpoint(path, *, model, opt_state_flat, epoch):\n"
         "    return None\n"
         "def train(model):\n"
         "    save_checkpoint('c.npz', model=model, opt_state_flat={}, epoch=7)\n")


def test_gate_refuses_the_bare_literal(tmp_path):
    from tac.confound_gates import (
        check_checkpoint_saves_do_not_silently_drop_optimizer_state as gate,
    )
    _write(tmp_path, "experiments/planted.py", _BARE)
    v = gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1 and "planted.py" in v[0]
    with pytest.raises(Exception, match="opt_state_flat"):
        gate(repo_root=tmp_path, strict=True, verbose=False)


def test_gate_allows_a_resolver_or_a_stated_none(tmp_path):
    """The cure is not 'always persist' — the default must stay OFF for byte-identity. It is
    that the callsite SAYS which it is."""
    from tac.confound_gates import (
        check_checkpoint_saves_do_not_silently_drop_optimizer_state as gate,
    )
    _write(tmp_path, "experiments/resolver.py",
           "def save_checkpoint(path, *, opt_state_flat):\n    return None\n"
           "def train(_opt_state):\n"
           "    save_checkpoint('c.npz', opt_state_flat=_opt_state())\n")
    _write(tmp_path, "experiments/stated.py",
           "def save_checkpoint(path, *, opt_state_flat):\n    return None\n"
           "def no_opt_state(reason):\n    return {}\n"
           "def train():\n"
           "    save_checkpoint('c.npz', opt_state_flat=no_opt_state('a stated reason here'))\n")
    assert gate(repo_root=tmp_path, strict=False, verbose=False) == []


def test_gate_respects_a_substantive_waiver_and_rejects_a_placeholder(tmp_path):
    from tac.confound_gates import (
        check_checkpoint_saves_do_not_silently_drop_optimizer_state as gate,
    )
    _write(tmp_path, "experiments/waived.py",
           "# OPT_STATE_DROP_OK: fixture checkpoint, never resumed by a live run\n" + _BARE)
    assert gate(repo_root=tmp_path, strict=False, verbose=False) == []
    _write(tmp_path, "experiments/placeholder.py",
           "# OPT_STATE_DROP_OK: <rationale>\n" + _BARE)
    assert len(gate(repo_root=tmp_path, strict=False, verbose=False)) == 1


def test_gate_is_live_count_zero_against_the_real_repo():
    """STRICT from byte one — the second landing CLAUDE.md's two-landing rule asks for."""
    from tac.confound_gates import (
        check_checkpoint_saves_do_not_silently_drop_optimizer_state as gate,
    )
    assert gate(strict=False, verbose=False) == []


def test_gate_is_registered_and_carries_a_positive_control():
    """A REFUSE-capable gate landing without a control raises the uncovered DENOMINATOR while the
    coverage floor stays satisfied — the exact arithmetic ddm_gh1's ceiling exists to close."""
    from tac.confound_gates import (
        CONFOUND_GATES,
        MAX_UNCOVERED_REFUSE_GATES,
        POSITIVE_CONTROLS,
        check_refusal_gates_have_live_positive_control,
        positive_control_coverage,
    )
    name = "check_checkpoint_saves_do_not_silently_drop_optimizer_state"
    assert name in {fn.__name__ for fn in CONFOUND_GATES}
    assert name in {c.gate for c in POSITIVE_CONTROLS}
    cov = positive_control_coverage()
    assert name not in cov["uncovered_gates"]
    assert len(cov["uncovered_gates"]) <= MAX_UNCOVERED_REFUSE_GATES
    # The class guard EXECUTES every control, including the new one.
    assert check_refusal_gates_have_live_positive_control(strict=False, verbose=False) == []
