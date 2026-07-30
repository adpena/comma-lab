"""ddm_dw1 — tests for the QA75 solve-frame distill window (fork discriminator).

Unit tests of the MECHANICS ONLY (distill loss forms, byte-identity when OFF, head-relax
warm-start equivalence + guard, resume ema-backfill, DSL compile/validate + the matched
argv diffs). Synthetic tensors verify CODE behavior only — NO scorer-behavior/score claim
is made from these tests (NO-FAKE #3); realized evidence comes from the governed windows.
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

mx = pytest.importorskip("mlx.core")

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    TR1Config,
    build_module,
    load_checkpoint,
    save_checkpoint,
)
from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _solve_frame_distill_loss_mlx,
    make_loss_fn,
)
from tac.witness_dsl.spec_tr1_dw1_distill_window_20260730 import dw1_window_program  # noqa: E402
from tac.witness_dsl.spec_tr1_renderer_20260728 import (  # noqa: E402
    lever_head_range_relax,
    lever_solve_frame_distill,
)


def _cfg(**kw) -> TR1Config:
    base = {
        "variant": "lotto", "num_pairs": 4, "grid_downsample": 16, "code_width": 2,
        "renderer_width": 8, "token_quant_levels": 16, "seed": 3, "lotto_seed": 118,
        "lotto_mask_density_init": 0.5, "seg_form_start": "tau_softplus", "w_seg": 100.0,
        "lr": 1e-3, "batch_pairs": 2, "epochs": 2, "gate_every": 1, "ema_decay": 0.95,
        "ema_decay_provenance": "test", "token_temporal_mode": "shared_base",
        "token_ste": "round", "class_weight_lane": 1.0, "margin_target": 1.0}
    base.update(kw)
    return TR1Config(**base)


# ---------------------------------------------------------------- distill loss forms ----
def _rand_logits(seed, h=6, w=8, k=5):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, h, w, k)).astype(np.float32))


def _oh_from_logits(logits):
    arg = np.argmax(np.asarray(logits), axis=-1)  # (1,h,w)
    k = logits.shape[-1]
    return mx.array((arg[..., None] == np.arange(k)).astype(np.float32))


@pytest.mark.parametrize("form", ["kd_logits", "margin_field"])
def test_distill_form_zero_when_student_matches_teacher(form):
    teacher = _rand_logits(1)
    student = teacher  # perfect student
    lstar_oh = _oh_from_logits(teacher)
    margin = mx.zeros((teacher.shape[1], teacher.shape[2]))
    val = float(_solve_frame_distill_loss_mlx(mx, student, teacher, lstar_oh, margin,
                                              form, 2.0, 0.0))
    # KD -> 0 when identical distributions; margin_field -> 0 (student margin == teacher margin).
    assert val == pytest.approx(0.0, abs=1e-4), f"{form} not ~0 when matched: {val}"


def test_argmax_ce_zero_when_student_onehot_on_teacher_argmax():
    # argmax_ce = -log p_student(teacher_argmax); ~0 only when the student is confident on the
    # teacher's argmax (a soft-but-matched student still pays its own entropy, correctly > 0).
    teacher = _rand_logits(1)
    arg = np.argmax(np.asarray(teacher), axis=-1)  # (1,h,w)
    k = teacher.shape[-1]
    student = mx.array((arg[..., None] == np.arange(k)).astype(np.float32) * 30.0)  # near one-hot
    lstar_oh = _oh_from_logits(teacher)
    margin = mx.zeros((teacher.shape[1], teacher.shape[2]))
    val = float(_solve_frame_distill_loss_mlx(mx, student, teacher, lstar_oh, margin,
                                              "argmax_ce", 2.0, 0.0))
    assert val == pytest.approx(0.0, abs=1e-4), f"argmax_ce not ~0 for one-hot student: {val}"


@pytest.mark.parametrize("form", ["kd_logits", "margin_field", "argmax_ce"])
def test_distill_form_positive_when_mismatched(form):
    teacher = _rand_logits(1) * 3.0  # confident teacher
    student = _rand_logits(2) * 0.1  # near-uniform student
    lstar_oh = _oh_from_logits(teacher)
    margin = mx.ones((teacher.shape[1], teacher.shape[2]))
    val = float(_solve_frame_distill_loss_mlx(mx, student, teacher, lstar_oh, margin,
                                              form, 2.0, 0.0))
    assert np.isfinite(val) and val > 0.0, f"{form} not positive on mismatch: {val}"


def test_distill_attack_weighting_changes_value():
    teacher = _rand_logits(1) * 3.0
    student = _rand_logits(2) * 0.1
    lstar_oh = _oh_from_logits(teacher)
    # non-uniform GT margin -> attack weighting must re-weight (differ from uniform).
    rng = np.random.default_rng(5)
    margin = mx.array(rng.random((teacher.shape[1], teacher.shape[2])).astype(np.float32) * 4.0)
    uni = float(_solve_frame_distill_loss_mlx(mx, student, teacher, lstar_oh, margin,
                                              "kd_logits", 2.0, 0.0))
    att = float(_solve_frame_distill_loss_mlx(mx, student, teacher, lstar_oh, margin,
                                              "kd_logits", 2.0, 1.0))
    assert np.isfinite(att) and abs(att - uni) > 1e-6


def test_distill_unknown_form_raises():
    t = _rand_logits(1)
    with pytest.raises(ValueError, match="unknown distill_form"):
        _solve_frame_distill_loss_mlx(mx, t, t, _oh_from_logits(t),
                                      mx.zeros((t.shape[1], t.shape[2])), "bogus", 2.0, 0.0)


# ---------------------------------------------------------------- loss_fn byte-identity ----
def _fake_loss_pieces(h=6, w=8, k=5):
    teacher = _rand_logits(11, h, w, k)
    student = _rand_logits(22, h, w, k)
    lstar_oh = _oh_from_logits(teacher)
    margin = mx.ones((h, w))

    class _FakeAdapter:
        def segnet(self, f1):
            return student  # fixed student logits (render ignored)

    def _fake_render(model, cf, code, rh, rw):
        return mx.zeros((1, rh, rw, 3))

    loss_fn = make_loss_fn(_FakeAdapter(), h, w, render_fn=_fake_render, seg_loss="tau_softplus")
    return loss_fn, teacher, lstar_oh, margin


def test_loss_fn_distill_off_byte_identical():
    loss_fn, teacher, lstar_oh, margin = _fake_loss_pieces()
    base = float(loss_fn(None, None, 0, 0, lstar_oh, margin, mx.zeros((6,)), 100.0, 0.0, 0.0,
                         1.0, seg_form="tau_softplus", compute_pose=False))
    off_none = float(loss_fn(None, None, 0, 0, lstar_oh, margin, mx.zeros((6,)), 100.0, 0.0, 0.0,
                             1.0, seg_form="tau_softplus", compute_pose=False,
                             distill_logits=None, distill_weight=100.0))
    off_w0 = float(loss_fn(None, None, 0, 0, lstar_oh, margin, mx.zeros((6,)), 100.0, 0.0, 0.0,
                           1.0, seg_form="tau_softplus", compute_pose=False,
                           distill_logits=teacher, distill_weight=0.0))
    assert base == off_none == off_w0  # OFF (logits None OR weight 0) == the pre-distill return


def test_loss_fn_distill_on_adds_positive_term():
    loss_fn, teacher, lstar_oh, margin = _fake_loss_pieces()
    base = float(loss_fn(None, None, 0, 0, lstar_oh, margin, mx.zeros((6,)), 100.0, 0.0, 0.0,
                         1.0, seg_form="tau_softplus", compute_pose=False))
    on = float(loss_fn(None, None, 0, 0, lstar_oh, margin, mx.zeros((6,)), 100.0, 0.0, 0.0,
                       1.0, seg_form="tau_softplus", compute_pose=False,
                       distill_logits=teacher, distill_weight=100.0, distill_form="kd_logits"))
    assert on > base  # distill term is additive and > 0 for a mismatched teacher


# ---------------------------------------------------------------- head-relax Window C ----
def test_head_relax_warm_start_equivalent_at_init():
    from mlx.utils import tree_flatten

    cfg_off = _cfg(renderer_head_mode="rgb", head_range_relax="off")
    cfg_lin = _cfg(renderer_head_mode="rgb", head_range_relax="linear")
    m_off = build_module(cfg_off)
    m_lin = build_module(cfg_lin)
    mx.eval(m_off.parameters(), m_lin.parameters())
    # linear model has the EXTRA head_relax_gain (init 0); all other params share the seed.
    keys_lin = {k for k, _ in tree_flatten(m_lin.trainable_parameters())}
    keys_off = {k for k, _ in tree_flatten(m_off.trainable_parameters())}
    assert keys_lin - keys_off == {"head_relax_gain"}
    assert float(mx.max(mx.abs(m_lin.head_relax_gain))) == 0.0
    r_off = np.asarray(m_off.render_frame(0))
    r_lin = np.asarray(m_lin.render_frame(0))
    assert np.array_equal(r_off, r_lin)  # gain 0 => head == sigmoid(x)*255 EXACTLY


def test_head_relax_gain_de_saturates_after_perturb():
    cfg_lin = _cfg(renderer_head_mode="rgb", head_range_relax="linear")
    m = build_module(cfg_lin)
    r0 = np.asarray(m.render_frame(0)).copy()
    m.head_relax_gain = mx.array(np.full((3,), 0.5, dtype=np.float32))  # non-zero gain
    mx.eval(m.parameters())
    r1 = np.asarray(m.render_frame(0))
    assert not np.array_equal(r0, r1)  # a non-zero gain changes the output (residual active)


def test_head_relax_linear_requires_rgb_head():
    with pytest.raises(ValueError, match="requires renderer_head_mode='rgb'"):
        build_module(_cfg(renderer_head_mode="class_field", head_range_relax="linear"))


# ---------------------------------------------------------------- resume ema-backfill ----
def test_new_param_absent_from_checkpoint_and_backfillable(tmp_path):
    from mlx.utils import tree_flatten

    # An OFF checkpoint (no head_relax_gain) — the E2 analogue.
    m_off = build_module(_cfg(head_range_relax="off"))
    ema_off = {k: mx.array(v) for k, v in tree_flatten(m_off.trainable_parameters())}
    ckpt = tmp_path / "off.npz"
    save_checkpoint(ckpt, model=m_off, ema=ema_off, opt_state_flat={}, epoch=400,
                    stage="seg_trunk_tau", cfg=_cfg(head_range_relax="off"), telemetry_tail=[])
    # A linear model resumes from it: load_checkpoint leaves head_relax_gain at init; ema lacks it.
    m_lin = build_module(_cfg(head_range_relax="linear"))
    st = load_checkpoint(ckpt, m_lin)
    ema = st["ema"]
    model_init = dict(tree_flatten(m_lin.trainable_parameters()))
    backfilled = [k for k in model_init if k not in ema]
    assert backfilled == ["head_relax_gain"]  # exactly the new param needs backfill
    for k in backfilled:
        ema[k] = mx.array(model_init[k])
    assert set(ema) == set(model_init)  # after backfill live and shadow agree on keys
    assert float(mx.max(mx.abs(ema["head_relax_gain"]))) == 0.0  # backfilled at init 0


# ---------------------------------------------------------------- DSL compile + diffs ----
def _mask_stub(tmp_path):
    p = tmp_path / "keep.npy"
    np.save(p, np.ones((24, 32), dtype=bool))
    return str(p)


def test_distill_lever_compiles_and_validates(tmp_path):
    lev = lever_solve_frame_distill("/x/cache.npy", form="margin_field", weight=100.0,
                                    temp=2.0, attack_temp=1.0)
    assert lev.overrides["--distill-form"] == "margin_field"
    assert lev.constant_manifest["--distill-temp"]["rung"].startswith("CANONICAL")
    with pytest.raises(ValueError, match="distill form"):
        lever_solve_frame_distill("/x", form="bogus")
    with pytest.raises(ValueError, match="head_range_relax"):
        lever_head_range_relax("bogus")


def test_dw1_matched_argv_diffs(tmp_path):
    common = dict(mask_path=_mask_stub(tmp_path), gt_cache="/gt.npz",
                  resume_from="/e2.npz", distill_field_cache="/cache.npy",
                  distill_form="kd_logits", window_epochs=60)
    prog_b = dw1_window_program("control", "/out/b", **common)
    prog_a = dw1_window_program("distill", "/out/a", **common)
    prog_c = dw1_window_program("distill_head_relax", "/out/c", **common)
    av_b = prog_b.compile_trainer_argv()
    av_a = prog_a.compile_trainer_argv()
    av_c = prog_c.compile_trainer_argv()

    def flags(argv):
        return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}

    fb, fa, fc = flags(av_b), flags(av_a), flags(av_c)
    # A-vs-B differs by EXACTLY the 5 distill flags (matched-config discipline, guard 3);
    # --out-dir differs by construction and is excluded from the flag comparison.
    a_minus_b = {k for k in fa if fa.get(k) != fb.get(k)} - {"--out-dir"}
    assert a_minus_b == {"--distill-field-cache", "--distill-form", "--distill-weight",
                         "--distill-temp", "--distill-attack-temp"}
    # C-vs-A differs by EXACTLY --head-range-relax.
    c_minus_a = {k for k in fc if fc.get(k) != fa.get(k)} - {"--out-dir"}
    assert c_minus_a == {"--head-range-relax"}
    assert fc["--head-range-relax"] == "linear"
    # epochs = E2 resume epoch (400) + 1 + window (60) so range(401, 461) trains exactly 60.
    assert fb["--epochs"] == "461"
    # control has NO distill/head flags (byte-identical continuation).
    assert "--distill-field-cache" not in fb and "--head-range-relax" not in fb
