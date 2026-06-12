# SPDX-License-Identifier: MIT
"""NO-FAKE checkpoint/resume equivalence tests for the capstone trainer.

The load-bearing claim: a run that DIES mid-stage and RESUMES from the latest
checkpoint continues the EXACT same descent trajectory as an uninterrupted run.
The decisive test trains a reference trainer continuously, then trains a second
trainer to the checkpoint point, SAVES, builds a FRESH trainer, LOADS, and runs
the remaining steps — the two final states must be bit-identical (params, EMA
shadow, optimizer momentum, VQ codebook, d_seg).

A stub that saved only the params (and silently dropped the optimizer momentum
or the EMA shadow or the VQ codebook) would FAIL this test — the resumed
trajectory would diverge after the first post-resume step.

These reuse the synthetic frozen-scorer fixture (``_build_capstone_setup`` in
``test_capstone_vq_nerv``) because the checkpoint round-trip is
architecture-AGNOSTIC: it tests that trainer STATE serializes + restores
bit-exactly, which does not require the real EfficientNet scorer (that is the
throughput micro-benchmark's job).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from tac.capstone_vq_nerv.checkpoint import (  # noqa: E402
    CheckpointPosition,
    checkpoint_exists,
    is_done,
    load_checkpoint,
    read_manifest,
    save_checkpoint,
    write_done_marker,
)
from tac.capstone_vq_nerv.tests.test_capstone_vq_nerv import (  # noqa: E402
    _build_capstone_setup,
)


def _run_steps(trainer, n_steps, *, seed=0):
    """Run ``n_steps`` deterministic training steps over the full pair set."""
    rng = np.random.RandomState(seed)
    for _ in range(n_steps):
        perm = rng.permutation(trainer.n_pairs)
        trainer.step(perm, lr_scale=1.0)
        trainer._ema.update(trainer.bundle)


def _flat_params(trainer):
    """Flat dict of the bundle param arrays as numpy (for bit-exact compare)."""
    return {k: np.array(v) for k, v in tree_flatten(trainer.bundle.parameters())}


def _ema_shadow(trainer):
    return {k: np.array(v) for k, v in trainer._ema.shadow.items()}


def _vq_state(trainer):
    q = getattr(trainer.bundle, "quantizer", None)
    if q is None or getattr(q, "_codebook", None) is None:
        return {}
    return {
        "codebook": np.array(q._codebook),
        "cluster": np.array(q._ema_cluster_size),
        "ema_w": np.array(q._ema_w),
    }


def _assert_state_equal(a, b, *, atol=0.0):
    fa, fb = _flat_params(a), _flat_params(b)
    assert set(fa) == set(fb)
    for k in fa:
        np.testing.assert_allclose(fa[k], fb[k], atol=atol, err_msg=f"param {k}")
    sa, sb = _ema_shadow(a), _ema_shadow(b)
    assert set(sa) == set(sb)
    for k in sa:
        np.testing.assert_allclose(sa[k], sb[k], atol=atol, err_msg=f"ema {k}")
    va, vb = _vq_state(a), _vq_state(b)
    assert set(va) == set(vb)
    for k in va:
        np.testing.assert_allclose(va[k], vb[k], atol=atol, err_msg=f"vq {k}")
    # Optimizer state.
    assert a.opt_state.step == b.opt_state.step
    assert set(a.opt_state.muon_buffers) == set(b.opt_state.muon_buffers)
    for k in a.opt_state.muon_buffers:
        np.testing.assert_allclose(
            np.array(a.opt_state.muon_buffers[k]),
            np.array(b.opt_state.muon_buffers[k]),
            atol=atol, err_msg=f"muon {k}",
        )
    assert a._mech_step == b._mech_step
    assert a._ema._num_updates == b._ema._num_updates


def test_save_creates_complete_checkpoint(tmp_path):
    _, _, trainer = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(trainer, 3)
    out = save_checkpoint(trainer, tmp_path / "ckpt", CheckpointPosition(0, 3))
    assert checkpoint_exists(out)
    man = read_manifest(out)
    assert man["stage_index"] == 0
    assert man["epoch_in_stage"] == 3
    assert man["n_pairs"] == 6
    assert man["has_vq"] is True  # default carrier is vq_index
    assert man["opt_step"] >= 1  # at least one optimizer step happened


def test_resume_is_bit_identical_to_uninterrupted_run(tmp_path):
    """THE decisive kill+restart test: resumed == uninterrupted, bit-for-bit."""
    # Reference: 8 steps continuous.
    _, _, ref = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(ref, 8)

    # Interrupted: 5 steps -> checkpoint -> FRESH trainer -> load -> 3 more steps.
    _, _, a = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(a, 5)
    out = save_checkpoint(a, tmp_path / "ckpt", CheckpointPosition(0, 5))
    del a  # simulate process death

    _, _, b = _build_capstone_setup(n_pairs=6, seed=0)
    # b is freshly initialized (DIFFERENT random init from `a` because the bundle
    # is re-seeded, but identical seed -> identical init; the point is the LOAD
    # overwrites whatever b had).
    pos = load_checkpoint(b, out)
    assert pos == CheckpointPosition(0, 5)
    _run_steps(b, 3, seed=0)
    # The RNG for steps 5..7 must match the reference's steps 5..7. The reference
    # used a single RandomState(0) advanced 8 times; the resumed run uses a fresh
    # RandomState(0) advanced 3 times -> DIFFERENT perms. So re-run the reference's
    # tail with the SAME 3-step RNG the resume used, for an apples-to-apples compare.
    _, _, ref2 = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(ref2, 5)
    _run_steps(ref2, 3, seed=0)

    _assert_state_equal(ref2, b)


def test_resume_d_seg_matches(tmp_path):
    """The resumed trainer's exact d_seg matches the uninterrupted reference."""
    _, _, ref = _build_capstone_setup(n_pairs=6, seed=1)
    _run_steps(ref, 4)
    _run_steps(ref, 2, seed=7)
    d_seg_ref = ref.exact_d_seg(use_ema=False)

    _, _, a = _build_capstone_setup(n_pairs=6, seed=1)
    _run_steps(a, 4)
    out = save_checkpoint(a, tmp_path / "ckpt", CheckpointPosition(0, 4))
    _, _, b = _build_capstone_setup(n_pairs=6, seed=1)
    load_checkpoint(b, out)
    _run_steps(b, 2, seed=7)
    d_seg_b = b.exact_d_seg(use_ema=False)

    assert d_seg_b == pytest.approx(d_seg_ref, abs=1e-6)


def test_load_rejects_mismatched_basis(tmp_path):
    _, _, a = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(a, 2)
    out = save_checkpoint(a, tmp_path / "ckpt", CheckpointPosition(0, 2))
    _, _, other = _build_capstone_setup(n_pairs=4, seed=0)  # different n_pairs
    with pytest.raises(ValueError, match="cannot resume a different basis"):
        load_checkpoint(other, out)


def test_done_marker_roundtrip(tmp_path):
    assert not is_done(tmp_path / "run")
    write_done_marker(tmp_path / "run", {"final_d_seg": 0.012, "stages": 8})
    assert is_done(tmp_path / "run")


def test_atomic_write_leaves_no_tmp(tmp_path):
    _, _, trainer = _build_capstone_setup(n_pairs=6, seed=0)
    _run_steps(trainer, 2)
    out = save_checkpoint(trainer, tmp_path / "ckpt", CheckpointPosition(0, 2))
    tmps = [p for p in out.iterdir() if ".tmp" in p.name]
    assert tmps == [], f"left tmp files: {tmps}"


def test_ema_shadow_actually_restored_not_reinitialized(tmp_path):
    """A stub that dropped the EMA shadow would re-init it to live params; this
    proves the SHADOW (the export/inference bytes) round-trips distinctly."""
    _, _, a = _build_capstone_setup(n_pairs=6, seed=3)
    _run_steps(a, 6)  # shadow diverges from live (decay 0.95)
    shadow_before = _ema_shadow(a)
    live_before = _flat_params(a)
    # Shadow must NOT equal live (EMA lag) — otherwise the test is vacuous.
    diffs = [
        np.max(np.abs(shadow_before[k] - live_before[k]))
        for k in shadow_before
        if k in live_before
    ]
    assert max(diffs) > 1e-6, "shadow == live; test cannot distinguish restore"
    out = save_checkpoint(a, tmp_path / "ckpt", CheckpointPosition(0, 6))
    _, _, b = _build_capstone_setup(n_pairs=6, seed=3)
    load_checkpoint(b, out)
    shadow_after = _ema_shadow(b)
    for k in shadow_before:
        np.testing.assert_allclose(shadow_after[k], shadow_before[k], atol=0.0)
