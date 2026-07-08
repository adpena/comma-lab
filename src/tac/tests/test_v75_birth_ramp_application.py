"""v7.5 Lever-2 birth-completion RAMP APPLICATION — the OWED integration landed.

Covers the loss-surface ramp (memo ``v75_birth_counterforce_20260708.md`` §RAMP-LANDED): the
per-class birth-completion multiplier applied to the three birth surfaces (island-amplify /
persistence-recall / logit-adjust offset), each PER-CLASS INDEPENDENTLY, resume-safe, byte-identical
off-path. means != ends: advisory control; only a byte-closed n600 exact row < 0.19110 moves the
pointer 0.19110.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.birth_completion import (
    BirthCompletionController,
    birth_completion_apply_restore,
    birth_completion_state_arrays,
    birth_ramp_multiplier_vector,
    derive_post_level_from_persistence,
)

# ── engine: DERIVED post_level provenance ────────────────────────────────────────────────────────


def test_derive_post_level_is_unformed_fraction():
    """post_level = 1 - tau_persist (retain the unformed-tail fraction of birth force)."""
    assert derive_post_level_from_persistence(0.8) == pytest.approx(0.2)
    assert derive_post_level_from_persistence(0.9) == pytest.approx(0.1)
    assert derive_post_level_from_persistence(1.0) == pytest.approx(0.0)  # fully formed => full hand-off
    # clamped to [0, 1]
    assert derive_post_level_from_persistence(1.5) == 0.0
    assert derive_post_level_from_persistence(-0.5) == 1.0


# ── engine: per-class ramp multiplier vector (drives the logit-adjust offset) ─────────────────────


def test_ramp_vector_identity_before_fire():
    """Pre-fire => every entry 1.0 (byte-identical offset scaling)."""
    ctrl = BirthCompletionController(classes=(1, 3))
    assert birth_ramp_multiplier_vector(ctrl, 50) == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert birth_ramp_multiplier_vector(None, 50) == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_ramp_vector_per_class_independent():
    """Only the FIRED class ramps; the still-growing class stays at 1.0."""
    ctrl = BirthCompletionController(classes=(1, 3), ramp_epochs=50, post_level=0.2)
    ctrl.fired[1] = 100  # lane fired; movable (3) has NOT
    v_fire = birth_ramp_multiplier_vector(ctrl, 100)
    assert v_fire[1] == 1.0 and v_fire[3] == 1.0  # AT the fire epoch, still 1.0 (ramp starts after)
    v_mid = birth_ramp_multiplier_vector(ctrl, 125)  # 25/50 through the ramp
    assert v_mid[1] == pytest.approx(0.6)  # 1 + (0.2-1)*0.5 = 0.6
    assert v_mid[3] == 1.0                 # movable untouched (independent)
    v_done = birth_ramp_multiplier_vector(ctrl, 200)
    assert v_done[1] == pytest.approx(0.2) and v_done[3] == 1.0
    # unwatched classes are never ramped
    assert v_mid[0] == 1.0 and v_mid[2] == 1.0 and v_mid[4] == 1.0


def test_ramp_vector_out_of_range_class_ignored():
    ctrl = BirthCompletionController(classes=(7,))  # class index >= n_classes
    ctrl.fired[7] = 10
    assert birth_ramp_multiplier_vector(ctrl, 100, n_classes=5) == [1.0] * 5


# ── engine: resume — restore latched fire epochs INTO an existing controller ──────────────────────


def test_apply_restore_roundtrip_into_existing_controller():
    """The multiplier trajectory is reconstructed exactly from restored fire epochs."""
    ctrl = BirthCompletionController(classes=(1, 3), ramp_epochs=40, post_level=0.2)
    ctrl.fired[1] = 100
    ctrl.fired[3] = 220
    arrays = birth_completion_state_arrays(ctrl)

    fresh = BirthCompletionController(classes=(1, 3), ramp_epochs=40, post_level=0.2)  # argv-derived
    assert birth_completion_apply_restore(fresh, arrays) is True
    assert fresh.fired == {1: 100, 3: 220}
    # identical subsequent trajectory
    for ep in (100, 130, 250, 400):
        assert birth_ramp_multiplier_vector(fresh, ep) == birth_ramp_multiplier_vector(ctrl, ep)


def test_apply_restore_legacy_sidecar_restores_unfired():
    """A legacy sidecar (no __bc_ keys / empty fired) => un-fired => byte-identical pre-fire."""
    ctrl = BirthCompletionController(classes=(1, 3))
    assert birth_completion_apply_restore(ctrl, {}) is False
    assert ctrl.fired == {}
    # an event-ON-but-unfired sidecar (keys present, empty fired arrays) also restores nothing
    unfired = birth_completion_state_arrays(BirthCompletionController(classes=(1, 3)))
    ctrl2 = BirthCompletionController(classes=(1, 3))
    assert birth_completion_apply_restore(ctrl2, unfired) is False
    assert ctrl2.fired == {}


def test_apply_restore_ignores_unwatched_stale_class():
    """A stale fire key for a no-longer-watched class is not restored (fail-closed)."""
    stale = BirthCompletionController(classes=(1, 3))
    stale.fired[3] = 50
    arrays = birth_completion_state_arrays(stale)
    now_watch_lane_only = BirthCompletionController(classes=(1,))  # movable dropped from watch
    assert birth_completion_apply_restore(now_watch_lane_only, arrays) is False
    assert now_watch_lane_only.fired == {}


def test_apply_restore_none_controller():
    assert birth_completion_apply_restore(None, {"__bc_fired_class": np.asarray([1])}) is False


# ── DSL: the ramp_apply param ─────────────────────────────────────────────────────────────────────


def test_dsl_ramp_apply_emits_flag():
    from tac.witness_dsl.curriculum_dsl import BirthCompletionEvent

    on = BirthCompletionEvent(ramp_apply=True, post_level=0.2)
    assert on.overrides.get("--birth-completion-ramp") is True
    assert on.overrides["--birth-completion-post-level"] == pytest.approx(0.2)
    assert "RAMP APPLIED" in on.notes

    off = BirthCompletionEvent()  # default = detector only
    assert "--birth-completion-ramp" not in off.overrides
    assert "DETECTOR-ONLY" in off.notes


def test_dsl_post_level_validated():
    from tac.witness_dsl.curriculum_dsl import BirthCompletionEvent

    with pytest.raises(ValueError):
        BirthCompletionEvent(post_level=1.5)
    with pytest.raises(ValueError):
        BirthCompletionEvent(post_level=-0.1)


# ── crucible v7.5 composes the ramp ON with the DERIVED post_level ────────────────────────────────


def test_crucible_v75_argv_carries_ramp_and_derived_post_level():
    from tac.witness_autoconfig import (
        _CRUCIBLE_V7_BIRTH_COMPLETION_TAU,
        compile_crucible_v7_config,
    )

    class _GT:
        lstars = [np.zeros((8, 8), np.int64) for _ in range(8)]

    argv = compile_crucible_v7_config(_GT(), num_pairs=8, epochs=3000).argv
    assert "--birth-completion-ramp" in argv
    assert "--birth-completion-event" in argv
    pl = float(argv[argv.index("--birth-completion-post-level") + 1])
    assert pl == pytest.approx(1.0 - _CRUCIBLE_V7_BIRTH_COMPLETION_TAU)  # DERIVED, not a magic literal


# ── island-amplify: per-class split identity + independence (MLX) ─────────────────────────────────


def _island_fixture(seed: int = 0):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    h, w = 16, 24
    signed = mx.array(rng.standard_normal((1, h, w)).astype(np.float32))
    lane = np.zeros((h, w), bool)
    lane[2:5, 3:10] = True
    mov = np.zeros((h, w), bool)
    mov[9:13, 5:15] = True
    union = lane | mov
    wt = np.zeros((h, w), np.float32)
    wt[union] = rng.uniform(0.5, 2.0, int(union.sum())).astype(np.float32)
    wt[union] /= wt[union].mean()  # mean-1 over the union (== island_persistence_weight contract)
    return (mx.array(wt[None]), mx.array(lane.astype(np.float32)[None]),
            mx.array((mov & ~lane).astype(np.float32)[None]), signed)


def test_island_perclass_identity_at_unit_multiplier():
    """mult_a == mult_b == 1.0 => the per-class split EQUALS the single combined term."""
    from tac.boundary_math.island_protection import (
        island_birth_from_signed_mx,
        island_birth_perclass_from_signed_mx,
    )

    wt, lane_m, mov_m, signed = _island_fixture()
    comb = float(island_birth_from_signed_mx(signed, wt, 1.0, form="hinge"))
    pc = float(island_birth_perclass_from_signed_mx(signed, wt, lane_m, mov_m, 1.0, 1.0, 1.0,
                                                    form="hinge"))
    assert pc == pytest.approx(comb, abs=1e-5)


def test_island_perclass_partitions_and_is_independent():
    """lane->0 keeps movable's share; movable->0 keeps lane's; the two shares sum to the combined."""
    from tac.boundary_math.island_protection import (
        island_birth_from_signed_mx,
        island_birth_perclass_from_signed_mx,
    )

    wt, lane_m, mov_m, signed = _island_fixture(3)
    comb = float(island_birth_from_signed_mx(signed, wt, 1.0, form="hinge"))
    only_mov = float(island_birth_perclass_from_signed_mx(signed, wt, lane_m, mov_m, 1.0, 0.0, 1.0))
    only_lane = float(island_birth_perclass_from_signed_mx(signed, wt, lane_m, mov_m, 1.0, 1.0, 0.0))
    assert only_mov > 0.0 and only_lane > 0.0
    assert only_mov + only_lane == pytest.approx(comb, abs=1e-5)  # disjoint partition of the support


def test_island_perclass_softplus_form_identity():
    from tac.boundary_math.island_protection import (
        island_birth_from_signed_mx,
        island_birth_perclass_from_signed_mx,
    )

    wt, lane_m, mov_m, signed = _island_fixture(5)
    comb = float(island_birth_from_signed_mx(signed, wt, 0.5, form="softplus"))
    pc = float(island_birth_perclass_from_signed_mx(signed, wt, lane_m, mov_m, 0.5, 1.0, 1.0,
                                                    form="softplus"))
    assert pc == pytest.approx(comb, abs=1e-5)


# ── persistence-recall: per-class scale byte-identity + effect ────────────────────────────────────


def _persist_fixture(seed: int = 1):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    h, w = 16, 20
    logits = mx.array(rng.standard_normal((1, h, w, 5)).astype(np.float32))
    lab = rng.integers(0, 5, (1, h, w))
    oh = np.zeros((1, h, w, 5), np.float32)
    for c in range(5):
        oh[..., c] = (lab == c)
    return logits, mx.array(oh)


def test_persistence_recall_scale_none_and_ones_are_byte_identical():
    from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_mlx

    logits, oh = _persist_fixture()
    base = float(persistence_topology_loss_mlx(logits, oh, [1, 3]))
    none = float(persistence_topology_loss_mlx(logits, oh, [1, 3], recall_class_scale=None))
    ones = float(persistence_topology_loss_mlx(logits, oh, [1, 3], recall_class_scale=[1.0, 1.0]))
    assert none == base
    assert ones == base  # exact: recall * 1.0 is bit-exact fp32


def test_persistence_recall_scale_ramps_per_class():
    from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_mlx

    logits, oh = _persist_fixture(2)
    base = float(persistence_topology_loss_mlx(logits, oh, [1, 3]))
    zero_mov = float(persistence_topology_loss_mlx(logits, oh, [1, 3], recall_class_scale=[1.0, 0.0]))
    assert zero_mov != base  # movable recall handed off => term changes


def test_persistence_recall_scale_length_mismatch_fails_closed():
    from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_mlx

    logits, oh = _persist_fixture()
    with pytest.raises(ValueError):
        persistence_topology_loss_mlx(logits, oh, [1, 3], recall_class_scale=[1.0])


# ── trainer argparse exposes the apply switch (never-invent-flags: it EXISTS) ─────────────────────


def test_trainer_argparse_has_birth_completion_ramp_flag():
    from tac.witness_dsl.curriculum_dsl import real_boolean_flags

    flags = real_boolean_flags()  # parses the live-launch (levelset) trainer argparse
    # BooleanOptionalAction => the DSL may emit it bare (True) or --no- (False) without crashing.
    assert "--birth-completion-ramp" in flags
    assert "--birth-completion-event" in flags  # sibling detector switch (sanity)


# ── resume registry: the trainer's __bc_ FunctionResumable wiring round-trips (registration path) ──


def test_resume_registry_birth_completion_wiring_roundtrip():
    """Mirror the trainer's ``register('birth_completion', '__bc_', FunctionResumable(...))``: the
    __bc_ prefix is accepted, write delegates to state_arrays, restore updates the live controller,
    and an OFF controller emits NO keys + NO manifest (byte-identical)."""
    from tac.witness_control.resume_registry import FunctionResumable, ResumeRegistry

    # ON: controller with a latched fire round-trips through the registry write/restore.
    src = BirthCompletionController(classes=(1, 3), ramp_epochs=40, post_level=0.2)
    src.fired[3] = 220
    reg = ResumeRegistry()
    reg.register("birth_completion", "__bc_", FunctionResumable(
        write=lambda _p: birth_completion_state_arrays(src),
        restore=lambda _p, cfg: birth_completion_apply_restore(src, cfg)))
    arrays = reg.state_arrays()
    assert any(k.startswith("__bc_") for k in arrays)          # keys written under the prefix
    assert all(k.startswith(("__bc_", "__resume")) for k in arrays)  # nothing leaks another prefix

    live = BirthCompletionController(classes=(1, 3), ramp_epochs=40, post_level=0.2)
    reg2 = ResumeRegistry()
    reg2.register("birth_completion", "__bc_", FunctionResumable(
        write=lambda _p: birth_completion_state_arrays(live),
        restore=lambda _p, cfg: birth_completion_apply_restore(live, cfg)))
    reg2.restore(arrays)
    assert live.fired == {3: 220}

    # OFF (controller None) => write returns {} => no keys => no manifest (byte-identical contract).
    reg_off = ResumeRegistry()
    reg_off.register("birth_completion", "__bc_", FunctionResumable(
        write=lambda _p: birth_completion_state_arrays(None),
        restore=lambda _p, cfg: birth_completion_apply_restore(None, cfg)))
    assert reg_off.state_arrays() == {}
