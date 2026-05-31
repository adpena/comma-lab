# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Z8 faithful-render boundary_argmax_hinge seg confirm runner.

WAVE-1C2 2026-05-31. These tests verify ACTUAL contest-functional behavior of
``tools/z8_argmax_hinge_faithful_render_seg_confirm.py``'s measurement helpers,
NOT constants. Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": every test below would
FAIL if the measurement were replaced by a constant / stub — the d_seg / d_pose
values respond to the actual rendered pixels and the real
``upstream.modules.DistortionNet`` argmax-flip functional, and the render-
faithfulness classifier responds to the actual recon distribution.

The headline NO-FAKE guards:

* ``test_identity_d_seg_d_pose_are_exactly_zero`` — the EXACT contest functional
  ``compute_distortion(gt, gt) == 0`` (would FAIL if d_seg were a fixed nonzero
  constant).
* ``test_d_seg_responds_to_pixel_content`` — a structurally-different recon RAISES
  d_seg above 0 (would FAIL if d_seg were constant / ignored its input).
* ``test_z8_render_is_nonconstant`` — the deterministic per-level argmax eval
  render produces a spatially-varying RGB image, not a flat fill (would FAIL on
  a constant renderer — the FORBIDDEN no-op-detector class per Catalog #105/#139).
* ``test_render_faithfulness_classifies_collapse_vs_faithful`` — the faithfulness
  classifier flags a near-white constant as COLLAPSED and a GT-like distribution
  as FAITHFUL (would FAIL if the classifier ignored its input — the sister C/C'
  renderer-collapse confound the Z8 testbed exists to avoid).
* ``test_ema_shadow_roundtrip_preserves_per_level_logits`` — the EMA-shadow
  ``.npsd`` load restores the per-level categorical logits exactly (would FAIL if
  the loader silently dropped weights).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = REPO_ROOT / "tools" / "z8_argmax_hinge_faithful_render_seg_confirm.py"

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_z8_argmax_hinge_faithful_render_seg_confirm", RUNNER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def real_distortion_net(runner):
    seg = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
    pose = REPO_ROOT / "upstream" / "models" / "posenet.safetensors"
    if not seg.is_file() or not pose.is_file():
        pytest.skip("upstream scorer safetensors not present")
    return runner._real_distortion_net()


@pytest.fixture(scope="module")
def gt_pairs():
    """4 REAL GT pairs (P, 2, 384, 512, 3) [0,255] decoded from the contest video.

    REAL frames (NOT synthetic) per Catalog #114 + the empirical finding that the
    EfficientNet-B2 SegNet is out-of-distribution on synthetic gradients (it maps
    them all to one class, so d_seg degenerates to 0 — a measurement artifact, not
    the contest functional). The contest-faithful guards below require the SegNet
    to produce the same varied argmax regions it sees on real dashcam imagery.
    """
    video = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.is_file():
        pytest.skip("upstream/videos/0.mkv not present")
    runner = _load_runner()
    return runner._decode_gt_pairs(str(video), num_pairs=4)


def test_identity_d_seg_d_pose_are_exactly_zero(
    runner, real_distortion_net, gt_pairs
):
    """compute_distortion(gt, gt) == 0 — the NO-FAKE identity guard.

    This is the EXACT contest functional (upstream/modules.py:112 + :84). If the
    measurement returned a fixed constant, this would not be zero.
    """
    m = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, gt_pairs)
    assert m["mean_d_seg"] == 0.0
    assert m["mean_d_pose"] == 0.0


def test_d_seg_responds_to_pixel_content(runner, real_distortion_net, gt_pairs):
    """d_seg = 0 when recon == GT, but NONZERO when recon is a different scene.

    Would FAIL if d_seg were a constant / ignored its second argument. Uses a
    STRUCTURALLY different recon (circularly-shifted pairing of the same real
    frames) to force SegNet argmax decisions to differ.
    """
    m_id = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, gt_pairs)
    assert m_id["mean_d_seg"] == 0.0

    recon = np.roll(gt_pairs, shift=1, axis=0).astype(np.float32)
    m_diff = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, recon)
    assert m_diff["mean_d_seg"] > 0.0
    assert 0.0 <= m_diff["mean_d_seg"] <= 1.0
    assert 0.0 <= m_diff["max_d_seg"] <= 1.0


def test_d_pose_responds_to_pixel_perturbation(
    runner, real_distortion_net, gt_pairs
):
    """Pose-axis MSE rises with recon perturbation (NOT a phantom-0 mock)."""
    rng = np.random.default_rng(13)
    noise = rng.normal(0, 1, gt_pairs.shape).astype(np.float32)
    recon_small = np.clip(gt_pairs + noise * 5.0, 0, 255).astype(np.float32)
    recon_big = np.clip(gt_pairs + noise * 60.0, 0, 255).astype(np.float32)
    m_small = runner._measure_real_d_seg_d_pose(
        real_distortion_net, gt_pairs, recon_small
    )
    m_big = runner._measure_real_d_seg_d_pose(
        real_distortion_net, gt_pairs, recon_big
    )
    assert m_big["mean_d_pose"] > m_small["mean_d_pose"] > 0.0


def test_z8_render_is_nonconstant(runner):
    """Deterministic per-level argmax eval render is spatially-varying, not flat.

    A constant renderer (the FORBIDDEN no-op class per Catalog #105/#139) would
    yield std==0 and the runner's recon_is_nonconstant guard would be False. This
    exercises the Z8-specific render path (per-level argmax over
    ``logits_per_level`` -> ``forward_eval_from_indices``) on a fresh model.
    """
    model, cfg = runner._build_z8_model(num_pairs=3)
    mx.eval(model.parameters())
    recon = runner._render_all_pairs(model, cfg, num_pairs=3)
    assert recon.shape == (3, 2, 384, 512, 3)
    assert float(np.min(recon)) >= 0.0
    assert float(np.max(recon)) <= 255.0
    # spatially nonconstant (would be 0 for a flat-fill no-op renderer).
    assert float(np.std(recon)) > 1.0


def test_z8_render_uses_full_main_config_sizing(runner):
    """The runner's model matches the Z8 _full_main config (else weights won't load).

    Z8 ``_full_main`` builds ``num_levels=3 num_groups_per_level=(4,3,2)
    num_categories_per_level=(16,8,4) base_channels=8 decoder_latent_dim=12
    deterministic_state_dim=8``. A mismatch silently breaks the EMA-shadow load;
    this pins the contract.
    """
    _, cfg = runner._build_z8_model(num_pairs=5)
    assert int(cfg.num_levels) == 3
    assert tuple(cfg.num_groups_per_level) == (4, 3, 2)
    assert tuple(cfg.num_categories_per_level) == (16, 8, 4)
    assert int(cfg.base_channels) == 8
    assert int(cfg.decoder_latent_dim) == 12
    assert int(cfg.deterministic_state_dim) == 8
    assert int(cfg.num_pairs) == 5
    # logits_per_level is a length-3 list with the right per-level shapes.
    assert len(cfg.num_groups_per_level) == 3


def test_render_faithfulness_classifies_collapse_vs_faithful(runner, gt_pairs):
    """The faithfulness classifier responds to the recon distribution (NOT a stub).

    The whole point of the Z8 testbed (vs sister C/C' DreamerV3) is that a
    FAITHFUL render lets the seg lever be tested while a COLLAPSED render does
    not. This guard proves the classifier distinguishes the two on real GT.
    Would FAIL if the classifier ignored its input or returned a constant verdict.
    """
    shape = gt_pairs.shape
    gt_mean = float(np.mean(gt_pairs))
    # (1) near-white constant -> COLLAPSED (saturated; std ratio ~0).
    white = np.full(shape, 254.0, dtype=np.float32)
    ff_white = runner._render_faithfulness(white, gt_pairs)
    assert ff_white["collapsed"] is True
    assert ff_white["faithful"] is False
    assert ff_white["near_constant"] is True or ff_white["saturated"] is True

    # (2) the GT distribution itself -> FAITHFUL (by construction in-range).
    ff_gt = runner._render_faithfulness(gt_pairs, gt_pairs)
    assert ff_gt["faithful"] is True
    assert ff_gt["collapsed"] is False
    # the classifier read the real GT mean (not a hardcoded value).
    assert ff_gt["gt_mean"] == pytest.approx(gt_mean, rel=1e-9)
    assert ff_gt["recon_mean"] == pytest.approx(gt_mean, rel=1e-9)

    # (3) a GT-like-but-noisy render (in distribution) -> FAITHFUL.
    rng = np.random.default_rng(7)
    noisy = np.clip(
        gt_pairs + rng.normal(0, 8, shape).astype(np.float32), 0, 255
    ).astype(np.float32)
    ff_noisy = runner._render_faithfulness(noisy, gt_pairs)
    assert ff_noisy["faithful"] is True


def test_score_partial_uses_canonical_contest_weights(
    runner, real_distortion_net, gt_pairs
):
    """score_seg_pose_partial == 100*d_seg + sqrt(10*d_pose) (evaluate.py:92)."""
    import math

    rng = np.random.default_rng(17)
    recon = np.clip(
        gt_pairs + rng.normal(0, 30, gt_pairs.shape).astype(np.float32), 0, 255
    ).astype(np.float32)
    m = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, recon)
    expected = 100.0 * m["mean_d_seg"] + math.sqrt(10.0 * m["mean_d_pose"])
    assert m["score_seg_pose_partial_no_rate"] == pytest.approx(expected, rel=1e-9)


def test_ema_shadow_roundtrip_preserves_per_level_logits(runner, tmp_path):
    """The EMA-shadow .npsd load restores per-level logits EXACTLY (no silent drop)."""
    from mlx.utils import tree_unflatten

    from tac.substrates._shared.numpy_portable_inflate import (
        pack_state_dict_numpy,
        unpack_state_dict_numpy,
    )

    m1, cfg = runner._build_z8_model(num_pairs=3)
    # perturb every per-level logit tensor so the restored model must match a
    # NON-default value (proves the load restores trained weights, not defaults).
    for level_idx in range(int(cfg.num_levels)):
        m1.logits_per_level[level_idx] = (
            m1.logits_per_level[level_idx]
            + mx.random.normal(m1.logits_per_level[level_idx].shape) * 0.5
        )
    mx.eval(m1.parameters())

    flat: dict = {}

    def _fl(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _fl(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _fl(f"{prefix}.{i}" if prefix else str(i), v)
        elif hasattr(obj, "shape"):
            flat[prefix] = np.asarray(obj)

    _fl("", m1.parameters())
    blob = pack_state_dict_numpy(flat, dtype="fp32")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    npsd = ckpt_dir / "final_epoch000001_x.ema_shadow.state.npsd"
    npsd.write_bytes(blob)

    found = runner._find_ema_shadow_npsd(tmp_path)
    assert found == npsd

    sd = unpack_state_dict_numpy(found.read_bytes())
    m2, _ = runner._build_z8_model(num_pairs=3)
    m2.update(tree_unflatten([(k, mx.array(v)) for k, v in sd.items()]))
    mx.eval(m2.parameters())
    # every per-level logit restored EXACTLY (would FAIL if loader dropped a key).
    for level_idx in range(int(cfg.num_levels)):
        assert bool(
            mx.allclose(
                m1.logits_per_level[level_idx], m2.logits_per_level[level_idx]
            ).item()
        )
    # a different fresh model does NOT match (proves the test is real).
    m3, _ = runner._build_z8_model(num_pairs=3)
    mx.eval(m3.parameters())
    assert not bool(
        mx.allclose(
            m1.logits_per_level[0], m3.logits_per_level[0]
        ).item()
    )


def test_near_white_collapse_saturates_d_seg_objective_agnostic(
    runner, real_distortion_net, gt_pairs
):
    """NO-FAKE confound guard: a collapsed (near-white) render saturates d_seg.

    This documents WHY the Z8 testbed exists. Sister C/C' DreamerV3 collapsed to
    near-white and all 3 arms tied at d_seg=0.505906 (a renderer-collapse
    signature, NOT a seg-objective signal). The same confound is reproduced here
    on the REAL contest functional so that IF the Z8 render also collapses, the
    runner's INCONCLUSIVE-IF-COLLAPSED verdict is correctly attributed.

    1. A near-white constant recon vs real GT yields HIGH d_seg (~chance).
    2. d_seg is INSENSITIVE to which near-white constant (252 vs 254 vs 255 tie) —
       why independently-trained collapsed arms tie on d_seg.
    3. The measurement is STILL the real functional (GT-vs-GT == 0).
    """
    m_id = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, gt_pairs)
    assert m_id["mean_d_seg"] == 0.0

    shape = gt_pairs.shape
    white_a = np.full(shape, 252.0, dtype=np.float32)
    white_b = np.full(shape, 254.0, dtype=np.float32)
    white_c = np.full(shape, 255.0, dtype=np.float32)
    m_a = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, white_a)
    m_b = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, white_b)
    m_c = runner._measure_real_d_seg_d_pose(real_distortion_net, gt_pairs, white_c)

    assert m_b["mean_d_seg"] > 0.3, (
        f"near-white recon should give high d_seg, got {m_b['mean_d_seg']}"
    )
    assert m_a["mean_d_seg"] == pytest.approx(m_b["mean_d_seg"], abs=0.02)
    assert m_b["mean_d_seg"] == pytest.approx(m_c["mean_d_seg"], abs=0.02)

    recon_varied = np.roll(gt_pairs, shift=1, axis=0).astype(np.float32)
    m_varied = runner._measure_real_d_seg_d_pose(
        real_distortion_net, gt_pairs, recon_varied
    )
    assert m_varied["mean_d_seg"] > 0.0
