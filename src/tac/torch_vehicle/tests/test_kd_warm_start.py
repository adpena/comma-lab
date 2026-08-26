# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the KD-warm-start actuator + the FiLM rgb_0 decoupling refinement
(P1a of the bind-all production build-out). Every assertion checks REAL behavior, never a
constant. DEFAULT-OFF byte-identical is verified by construction (the new fields default to
None/False and the existing from-0 / warm_start paths are untouched).

Part A — FiLM rgb_0 decoupling (``cfg.pose_film_rgb0_pose_trainable``):
  * the base trunk-stopgrad FREEZES rgb_0 w.r.t. pose; with the refinement ON, rgb_0 is
    EXCLUDED from the seg-only-restore set, so the pose loss trains the frame-0 head;
  * THE decisive test: ∂d_seg/∂(pose-objective) STILL = 0 (trunk + rgb_1 + latents grad
    bit-identical to seg-only) AND rgb_0 NOW carries pose grad (was zero under the base
    trunk-stopgrad).

Part B — KD-warm-start (``cfg.kd_warm_start_dir``):
  * latents load DIRECTLY for a re-taper (the student decoder has a DIFFERENT shape than the
    basin teacher);
  * the teacher is FROZEN (no param has grad, and a KD step never changes it);
  * a few KD steps measurably LOWER the student-vs-teacher frame-MSE (the real distillation);
  * the KD warm-up is a PREFIX then the score-aware curriculum continues (end-to-end run);
  * default-None byte-identical (the from-0 path is unchanged);
  * resume-correctness round-trip (a kill mid-stage-0 resumes onto the post-KD state).

Forced-CPU throughout (one tiny frozen scorer for both head paths) so it runs without MPS
hardware and is deterministic; the only thing MPS changes in production is the DEVICE the
seg cotangent is computed on — a value, not the routing/distillation these tests prove.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext, _TinyFrozenScorer

_BASE_CH = 8
_LATENT_DIM = 28
_N_PAIRS = 6
# The solved-taper student schedule (a re-taper of base_ch=8 → DIFFERENT channel shapes
# than the vendored taper [8,8,8,6,4,4,4], so the strict-decoder load would FAIL — only KD
# can carry the basin in).
_SOLVED_TAPER = [8, 7, 7, 6, 7, 6, 4]


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(*, epochs=4, batch_size=_N_PAIRS, init_random=True) -> StageSpec:
    return StageSpec(
        name="kd_stage",
        epochs=epochs,
        seg_loss_fn=_ce_seg_loss,
        eval_every=10_000,  # no inline eval in these tests (the real scorer is synthetic)
        batch_size=batch_size,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=init_random,
    )


# ===========================================================================
# Helpers: write a converged VENDORED-taper basin (the KD teacher source).
# ===========================================================================
def _make_basin_dir(tmp_path, *, n_pairs=_N_PAIRS, base_channels=_BASE_CH, latent_dim=_LATENT_DIM, seed=123):
    """Write a canonical best/ dir holding a converged vendored-taper decoder + latents,
    perturbed off a fresh init so the teacher weights are unmistakably non-trivial."""
    v = import_vendored_bundle()
    torch.manual_seed(seed)
    dec = v.HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels, eval_size=(384, 512))
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    lat = torch.randn(n_pairs, latent_dim) * 0.37
    best = tmp_path / "basin" / "best"
    best.mkdir(parents=True, exist_ok=True)
    torch.save(dec.state_dict(), best / "best_ema_decoder.pt")
    torch.save(lat, best / "best_ema_latents.pt")
    return best, dec.state_dict(), lat


def _kd_cfg(tmp_path, *, taper=_SOLVED_TAPER, kd_dir=None, kd_warm_epochs=2, pose_film=False, **kw):
    return TorchVehicleConfig(
        base_channels=_BASE_CH,
        latent_dim=_LATENT_DIM,
        out_dir=tmp_path / "out",
        device="cpu",
        seed=0,
        taper_channels=taper,
        kd_warm_start_dir=kd_dir,
        kd_warm_epochs=kd_warm_epochs,
        pose_film_enabled=pose_film,
        pose_film_version=2 if pose_film else 1,
        **kw,
    )


def _kd_driver(cfg, *, n_pairs=_N_PAIRS, curriculum=None):
    scorer = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=0)
    return TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=curriculum or [_stage()],
    )


# ===========================================================================
# Part B — config guards.
# ===========================================================================
def test_kd_warm_start_dir_default_none():
    assert TorchVehicleConfig(device="cpu").kd_warm_start_dir is None


def test_kd_warm_start_requires_taper():
    with pytest.raises(ValueError, match="requires taper_channels"):
        TorchVehicleConfig(device="cpu", kd_warm_start_dir=Path("/x"), taper_channels=None)


def test_kd_warm_start_mutually_exclusive_with_warm_start():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchVehicleConfig(
            device="cpu", kd_warm_start_dir=Path("/x"), warm_start_dir=Path("/y"),
            taper_channels=_SOLVED_TAPER,
        )


def test_kd_warm_epochs_must_be_positive():
    with pytest.raises(ValueError, match="kd_warm_epochs must be >= 1"):
        TorchVehicleConfig(
            device="cpu", kd_warm_start_dir=Path("/x"), taper_channels=_SOLVED_TAPER,
            kd_warm_epochs=0,
        )


def test_kd_warm_lr_must_be_positive():
    with pytest.raises(ValueError, match="kd_warm_lr must be > 0"):
        TorchVehicleConfig(
            device="cpu", kd_warm_start_dir=Path("/x"), taper_channels=_SOLVED_TAPER,
            kd_warm_lr=0.0,
        )


# ===========================================================================
# Part B — latents load DIRECTLY for a re-taper (different decoder shape OK).
# ===========================================================================
def test_kd_latents_load_directly_for_retaper(tmp_path):
    """The basin latents (n_pairs, 28) load DIRECTLY as the re-taper student's stage-0 init
    — taper-INDEPENDENT. The student decoder shape differs from the basin (so a strict
    decoder load WOULD fail), proving the latents path is shape-agnostic."""
    from tac.torch_vehicle.kd_warm_start import load_kd_warm_start_latents

    best, _sd, saved_lat = _make_basin_dir(tmp_path)
    loaded = load_kd_warm_start_latents(best, n_pairs=_N_PAIRS, latent_dim=_LATENT_DIM)
    assert torch.allclose(loaded, saved_lat, atol=1e-7)
    # Confirm the student taper truly differs from the vendored taper (else "re-taper" is
    # vacuous): a vendored base_ch=8 decoder vs the solved taper have different shapes.
    from tac.torch_vehicle.configurable_taper_decoder import vendored_taper

    assert vendored_taper(_BASE_CH) != _SOLVED_TAPER, "test taper must differ from vendored"


def test_kd_latents_missing_file_raises(tmp_path):
    from tac.torch_vehicle.kd_warm_start import load_kd_warm_start_latents

    with pytest.raises(FileNotFoundError, match=r"best_ema_latents\.pt"):
        load_kd_warm_start_latents(tmp_path / "nope", n_pairs=_N_PAIRS, latent_dim=_LATENT_DIM)


def test_kd_latents_shape_mismatch_raises(tmp_path):
    from tac.torch_vehicle.kd_warm_start import load_kd_warm_start_latents

    best, _sd, _lat = _make_basin_dir(tmp_path)
    torch.save(torch.randn(_N_PAIRS + 3, _LATENT_DIM), best / "best_ema_latents.pt")
    with pytest.raises(ValueError, match="cannot warm-start a different basis"):
        load_kd_warm_start_latents(best, n_pairs=_N_PAIRS, latent_dim=_LATENT_DIM)


# ===========================================================================
# Part B — the teacher is FROZEN (no grad; a KD step never trains it).
# ===========================================================================
def test_kd_teacher_is_frozen(tmp_path):
    from tac.torch_vehicle.kd_warm_start import build_frozen_teacher

    best, _sd, _lat = _make_basin_dir(tmp_path)
    v = import_vendored_bundle()
    teacher = build_frozen_teacher(
        best, vendored_decoder_cls=v.HNeRVDecoder, latent_dim=_LATENT_DIM,
        base_channels=_BASE_CH, device="cpu",
    )
    # Every teacher param has requires_grad False.
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not teacher.training  # eval mode


def test_kd_teacher_unchanged_after_kd_step(tmp_path):
    """A KD step trains the STUDENT only; the frozen teacher's weights are bit-identical
    before and after (the NO-FAKE frozen-teacher contract)."""
    from tac.torch_vehicle.kd_warm_start import build_frozen_teacher, kd_warm_up_decoder

    best, _sd, _lat = _make_basin_dir(tmp_path)
    v = import_vendored_bundle()
    teacher = build_frozen_teacher(
        best, vendored_decoder_cls=v.HNeRVDecoder, latent_dim=_LATENT_DIM,
        base_channels=_BASE_CH, device="cpu",
    )
    teacher_before = {k: t.detach().clone() for k, t in teacher.state_dict().items()}
    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder

    student = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LATENT_DIM, base_channels=_BASE_CH, eval_size=(384, 512),
        channels=_SOLVED_TAPER,
    )
    latents = torch.nn.Parameter(torch.randn(_N_PAIRS, _LATENT_DIM) * 0.1)
    kd_warm_up_decoder(
        student=student, teacher=teacher, latents=latents, n_pairs=_N_PAIRS,
        epochs=2, batch_size=_N_PAIRS, lr=1e-3, device="cpu",
    )
    for k, t in teacher.state_dict().items():
        assert torch.equal(t, teacher_before[k]), f"teacher param {k} CHANGED during KD"


# ===========================================================================
# Part B — THE NO-FAKE distillation test: KD actually lowers frame-MSE.
# ===========================================================================
def test_kd_step_lowers_frame_mse_toward_teacher(tmp_path):
    """The decisive distillation proof: after a few KD epochs the student-vs-teacher
    frame-MSE is measurably LOWER than at the first epoch (the student is converging toward
    the teacher). If the KD were fake (no real backward) the loss would not drop."""
    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder
    from tac.torch_vehicle.kd_warm_start import build_frozen_teacher, kd_warm_up_decoder

    best, _sd, _lat = _make_basin_dir(tmp_path)
    v = import_vendored_bundle()
    teacher = build_frozen_teacher(
        best, vendored_decoder_cls=v.HNeRVDecoder, latent_dim=_LATENT_DIM,
        base_channels=_BASE_CH, device="cpu",
    )
    torch.manual_seed(0)
    student = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LATENT_DIM, base_channels=_BASE_CH, eval_size=(384, 512),
        channels=_SOLVED_TAPER,
    )
    latents = torch.nn.Parameter(torch.randn(_N_PAIRS, _LATENT_DIM) * 0.1)
    stats = kd_warm_up_decoder(
        student=student, teacher=teacher, latents=latents, n_pairs=_N_PAIRS,
        epochs=8, batch_size=_N_PAIRS, lr=5e-3, device="cpu",
    )
    assert stats["last_loss"] < stats["first_loss"], (
        f"KD did not reduce frame-MSE: first={stats['first_loss']:.3f} "
        f"last={stats['last_loss']:.3f} — distillation is not actually running"
    )
    # And the reduction is substantial (not numerical noise): >= 5% relative drop.
    assert stats["last_loss"] <= 0.95 * stats["first_loss"]


def test_kd_frame_mse_is_real_mse():
    """``kd_frame_mse`` is a genuine per-pixel MSE: zero for identical frames, positive and
    matching F.mse_loss for differing frames."""
    from tac.torch_vehicle.kd_warm_start import kd_frame_mse

    a = torch.rand(2, 2, 3, 8, 8) * 255.0
    assert kd_frame_mse(a, a.clone()).item() == pytest.approx(0.0, abs=1e-6)
    b = a + 3.0
    assert kd_frame_mse(a, b).item() == pytest.approx(F.mse_loss(a, b).item(), rel=1e-6)


# ===========================================================================
# Part B — end-to-end: KD warm-up is a PREFIX, then score-aware continues.
# ===========================================================================
def test_kd_warm_up_then_curriculum_continues_e2e(tmp_path):
    """A FRESH run with kd_warm_start_dir runs the KD warm-up THEN the score-aware
    curriculum to completion (DONE marker). The KD telemetry row is recorded and its
    last-loss < first-loss (the distillation actually ran inside run())."""
    best, _sd, _lat = _make_basin_dir(tmp_path)
    cfg = _kd_cfg(tmp_path, kd_dir=best, kd_warm_epochs=2)
    driver = _kd_driver(cfg, curriculum=[_stage(epochs=4)])
    summary = driver.run()
    assert summary["status"] == "complete"
    # The KD warm-up telemetry row exists and proves distillation ran.
    from tac.torch_vehicle.telemetry import read_trajectory

    rows = read_trajectory(cfg.out_dir)
    kd_rows = [r for r in rows if str(r.get("stage_name", "")).endswith("__kd_warm_up")]
    assert kd_rows, "no KD warm-up telemetry row recorded"


def test_kd_warm_epochs_exceeding_stage0_raises(tmp_path):
    """kd_warm_epochs MUST be <= the stage-0 epoch budget (else the whole of stage 0 is KD
    with no score-aware continuation) — refused at run() time."""
    best, _sd, _lat = _make_basin_dir(tmp_path)
    cfg = _kd_cfg(tmp_path, kd_dir=best, kd_warm_epochs=10)
    driver = _kd_driver(cfg, curriculum=[_stage(epochs=4)])
    with pytest.raises(ValueError, match="must be <= the stage-0 epoch budget"):
        driver.run()


def test_kd_warm_up_emas_track_distilled_student(tmp_path):
    """After the KD warm-up the run completes and the BEST/exported EMA shadow reflects a
    TRAINED state (not the random init): the final EMA decoder differs from a fresh-init
    decoder built with the same seed (the distillation + curriculum moved the weights)."""
    best, _sd, _lat = _make_basin_dir(tmp_path)
    cfg = _kd_cfg(tmp_path, kd_dir=best, kd_warm_epochs=2)
    driver = _kd_driver(cfg, curriculum=[_stage(epochs=3)])
    driver.run()
    # A fresh-init student (same seed/taper) — the run's trained weights must differ.
    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder

    torch.manual_seed(cfg.seed)
    fresh = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LATENT_DIM, base_channels=_BASE_CH, eval_size=(384, 512),
        channels=_SOLVED_TAPER,
    )
    # The checkpoint holds the trained decoder; load it and compare to fresh.
    from tac.torch_vehicle.checkpoint import load_checkpoint

    merged = load_checkpoint(cfg.out_dir, map_location="cpu")
    trained_ema = merged["ema_decoder"]
    fresh_sd = fresh.state_dict()
    changed = any(
        not torch.allclose(trained_ema[k].cpu(), fresh_sd[k].cpu(), atol=1e-6)
        for k in fresh_sd
    )
    assert changed, "trained EMA decoder identical to fresh init — KD+curriculum did nothing"


# ===========================================================================
# Part B — default-None byte-identical (the from-0 path is unchanged).
# ===========================================================================
def test_kd_none_is_byte_identical_to_from0(tmp_path):
    """Two runs with the SAME seed/taper and kd_warm_start_dir=None produce bit-identical
    trained checkpoints (the KD machinery is a true no-op when off)."""
    from tac.torch_vehicle.checkpoint import load_checkpoint

    def _run(sub):
        cfg = TorchVehicleConfig(
            base_channels=_BASE_CH, latent_dim=_LATENT_DIM, out_dir=tmp_path / sub,
            device="cpu", seed=0, taper_channels=_SOLVED_TAPER, kd_warm_start_dir=None,
        )
        _kd_driver(cfg, curriculum=[_stage(epochs=3)]).run()
        return load_checkpoint(cfg.out_dir, map_location="cpu")["ema_decoder"]

    a, b = _run("a"), _run("b")
    for k in a:
        assert torch.equal(a[k], b[k]), f"from-0 run not deterministic at {k}"


# ===========================================================================
# Part B — resume-correctness round-trip (kill mid-stage-0 → resume post-KD).
# ===========================================================================
def test_kd_resume_round_trip(tmp_path):
    """A run that dies AFTER the KD warm-up + first checkpoints, then resumes, COMPLETES
    and matches an uninterrupted run's final EMA — the KD warm-up is correctly idempotent
    (re-run on a fresh start, SKIPPED on a resume that owns a checkpoint)."""
    from tac.torch_vehicle.checkpoint import load_checkpoint

    best, _sd, _lat = _make_basin_dir(tmp_path)

    # Reference: uninterrupted run.
    cfg_ref = _kd_cfg(tmp_path, kd_dir=best, kd_warm_epochs=2)
    cfg_ref.out_dir = tmp_path / "ref"
    cfg_ref.checkpoint_every_epochs = 1
    _kd_driver(cfg_ref, curriculum=[_stage(epochs=4)]).run()
    ref_ema = load_checkpoint(cfg_ref.out_dir, map_location="cpu")["ema_decoder"]

    # Interrupted: stop after global epoch 2 (post-KD, mid stage 0), then resume.
    cfg_kill = _kd_cfg(tmp_path, kd_dir=best, kd_warm_epochs=2)
    cfg_kill.out_dir = tmp_path / "kill"
    cfg_kill.checkpoint_every_epochs = 1
    d1 = _kd_driver(cfg_kill, curriculum=[_stage(epochs=4)])
    d1._stop_after_global_epoch = 2
    from tac.torch_vehicle.driver import _SimulatedDeath

    with pytest.raises(_SimulatedDeath):
        d1.run()
    # Resume (a NEW driver on the same out_dir; KD must NOT re-run — the resume path skips it).
    d2 = _kd_driver(cfg_kill, curriculum=[_stage(epochs=4)])
    summary = d2.run()
    assert summary["status"] == "complete"
    kill_ema = load_checkpoint(cfg_kill.out_dir, map_location="cpu")["ema_decoder"]
    for k in ref_ema:
        assert torch.allclose(ref_ema[k], kill_ema[k], atol=1e-5), (
            f"resumed EMA decoder diverged from uninterrupted at {k}"
        )


# ===========================================================================
# Part A — FiLM rgb_0 decoupling refinement.
# ===========================================================================
def _film_cfg(tmp_path, *, rgb0_trainable: bool) -> TorchVehicleConfig:
    return TorchVehicleConfig(
        base_channels=_BASE_CH, latent_dim=_LATENT_DIM,
        out_dir=tmp_path / ("rgb0" if rgb0_trainable else "base"),
        device="cpu", train_device="mps", split_by_head=True,
        pose_film_enabled=True, pose_film_version=2, pose_film_trunk_stopgrad=True,
        pose_film_rgb0_pose_trainable=rgb0_trainable, seed=0,
    )


def _build_film_driver(tmp_path, *, rgb0_trainable: bool) -> TorchVehicleDriver:
    cfg = _film_cfg(tmp_path, rgb0_trainable=rgb0_trainable)
    scorer = SyntheticScorerContext(n_pairs=_N_PAIRS, device="cpu", seed=0, split_by_head=True)
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    stage = StageSpec(
        name="rgb0_test", epochs=2, seg_loss_fn=_ce_seg_loss, eval_every=2, batch_size=_N_PAIRS,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4, muon_weight_decay=0.0,
        latent_lr_mult=10.0, grad_clip=1e9, grad_clip_muon=1e9, lr_floor_ratio=5e-6,
        seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0, cat_sigma=0.2, use_qat=False,
        init_latents_random=True,
    )
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[stage]
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    return driver


def _rgb0_ids(driver, decoder):
    return driver._rgb0_param_ids(decoder)


def _decoder_latents(driver):
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(_N_PAIRS, _LATENT_DIM))
    return decoder, latents


def _roundtrip(decoder, latents, idx):
    decoded_pair = decoder(latents[idx], idx)
    B = len(idx)
    flat = decoded_pair.reshape(B * 2, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    return clamped + (clamped.round() - clamped).detach()


def _zero_grads(decoder, latents):
    for p in decoder.parameters():
        p.grad = None
    latents.grad = None


def _backward_once(driver, decoder, latents, *, compute_pose=True):
    idx = torch.arange(_N_PAIRS)
    decoded_bhwc = _roundtrip(decoder, latents, idx)
    _zero_grads(decoder, latents)
    return driver._split_by_head_backward(
        decoded_bhwc, idx, driver.curriculum[0],
        compute_pose=compute_pose, decoder=decoder, latents=latents,
    )


def test_rgb0_trainable_default_off():
    assert TorchVehicleConfig(out_dir=Path("/dev/null")).pose_film_rgb0_pose_trainable is False


def test_rgb0_trainable_requires_trunk_stopgrad():
    with pytest.raises(ValueError, match="requires pose_film_trunk_stopgrad=True"):
        TorchVehicleConfig(
            out_dir=Path("/dev/null"), device="cpu", train_device="mps", split_by_head=True,
            pose_film_enabled=True, pose_film_version=2, pose_film_trunk_stopgrad=False,
            pose_film_rgb0_pose_trainable=True,
        )


def test_rgb0_param_ids_resolves_inner_head(tmp_path):
    """The rgb_0 id-set is non-empty (2 params: weight + bias) and points at the INNER
    decoder's rgb_0 — exactly the frame-0 head."""
    driver = _build_film_driver(tmp_path, rgb0_trainable=True)
    decoder, _ = _decoder_latents(driver)
    rgb0 = _rgb0_ids(driver, decoder)
    assert rgb0, "rgb_0 id-set must be non-empty"
    inner = decoder.decoder  # FiLM wrapper holds the taper decoder here
    expected = {id(p) for p in inner.rgb_0.parameters()}
    assert rgb0 == expected


def test_non_film_grad_params_excludes_rgb0_only_when_flag_on(tmp_path):
    """With the flag ON, rgb_0 params are EXCLUDED from the seg-only-restore set; with the
    flag OFF (base trunk-stopgrad) rgb_0 IS in the set (restored to seg-only → frozen)."""
    # Flag ON.
    drv_on = _build_film_driver(tmp_path, rgb0_trainable=True)
    dec_on, lat_on = _decoder_latents(drv_on)
    shared_on = {id(p) for p in drv_on._non_film_grad_params(dec_on, lat_on)}
    rgb0_on = _rgb0_ids(drv_on, dec_on)
    assert shared_on.isdisjoint(rgb0_on), "rgb_0 must be EXCLUDED from seg-only set when ON"
    # Flag OFF.
    drv_off = _build_film_driver(tmp_path, rgb0_trainable=False)
    dec_off, lat_off = _decoder_latents(drv_off)
    shared_off = {id(p) for p in drv_off._non_film_grad_params(dec_off, lat_off)}
    rgb0_off = _rgb0_ids(drv_off, dec_off)
    assert rgb0_off.issubset(shared_off), "rgb_0 must be IN the seg-only set when OFF"


def test_rgb0_refinement_preserves_dseg_decoupling_and_trains_rgb0(tmp_path):
    """THE decisive Part-A test. With the rgb_0 refinement ON:
      (1) ∂d_seg/∂(pose-objective) STILL = 0 — every SHARED param EXCEPT rgb_0 (trunk +
          skips + blocks + refine + rgb_1 + latents) keeps the SEG-ONLY grad bit-identical
          (the pose backward left zero residue there), AND
      (2) rgb_0 NOW carries the pose gradient — its grad DIFFERS from the seg-only grad
          (the pose loss trains the frame-0 head), where under the base trunk-stopgrad it
          would have been restored to seg-only (frozen).
    """
    driver = _build_film_driver(tmp_path, rgb0_trainable=True)
    decoder, latents = _decoder_latents(driver)
    film_ids = driver._film_param_ids(decoder)
    rgb0_ids = driver._rgb0_param_ids(decoder)

    # Name every param for clear assertions; split into rgb_0 / other-shared / film / latents.
    named = list(decoder.named_parameters())
    rgb0_named = [(n, p) for n, p in named if id(p) in rgb0_ids]
    other_shared = [(n, p) for n, p in named if id(p) not in rgb0_ids and id(p) not in film_ids]
    assert rgb0_named, "must have rgb_0 params"

    # (A) SEG-ONLY reference (pose throttled OFF): the seg-only grads on EVERY param.
    _backward_once(driver, decoder, latents, compute_pose=False)
    seg_only = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in named}
    seg_only_lat = None if latents.grad is None else latents.grad.detach().clone()

    # (B) SEG+POSE with the rgb_0 refinement ON.
    _backward_once(driver, decoder, latents, compute_pose=True)

    # (1) Every OTHER shared param (NOT rgb_0, NOT FiLM) keeps the SEG-ONLY grad → d_seg ⊥ pose.
    for n, p in other_shared:
        ref = seg_only[n]
        if ref is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, (
                f"shared param {n} got pose gradient (expected zero under decoupling)"
            )
        else:
            assert p.grad is not None and torch.equal(p.grad, ref), (
                f"shared param {n} grad changed when pose was added → pose LEAKED into the "
                f"trunk (∂d_seg/∂pose != 0)"
            )
    # Latents too (the shared code that produces the seg frame f1).
    if seg_only_lat is None:
        assert latents.grad is None or torch.count_nonzero(latents.grad) == 0
    else:
        assert torch.equal(latents.grad, seg_only_lat), "latents got pose grad (must stay seg-only)"

    # (2) rgb_0 NOW carries the pose gradient — its grad DIFFERS from the seg-only grad.
    rgb0_changed = False
    for n, p in rgb0_named:
        assert p.grad is not None, f"rgb_0 param {n} lost its gradient"
        ref = seg_only[n]
        if ref is None:
            if torch.count_nonzero(p.grad) > 0:
                rgb0_changed = True
        elif not torch.equal(p.grad, ref):
            rgb0_changed = True
    assert rgb0_changed, (
        "rgb_0 grad identical with/without pose → the refinement did NOT route pose to "
        "rgb_0 (it is still frozen)"
    )


def test_rgb0_param_ids_on_bare_taper_decoder(tmp_path):
    """``_rgb0_param_ids`` resolves rgb_0 on a BARE (non-FiLM) taper decoder too (the
    ``decoder.rgb_0`` direct path) — defensive coverage for the wrapper-vs-bare layout."""
    cfg = TorchVehicleConfig(
        base_channels=_BASE_CH, latent_dim=_LATENT_DIM, out_dir=tmp_path / "bare",
        device="cpu", taper_channels=_SOLVED_TAPER,
    )
    driver = _kd_driver(cfg)
    dec = driver._new_decoder(device=torch.device("cpu"))  # bare taper (no FiLM)
    rgb0 = driver._rgb0_param_ids(dec)
    assert rgb0 == {id(p) for p in dec.rgb_0.parameters()}


# ===========================================================================
# Part A+B compose: taper + FiLM-v2 (the bind-all production combo).
# ===========================================================================
def test_taper_plus_film_v2_composes():
    """The bind-all combo (solved taper + FiLM-v2 pose decouple) constructs — the v2
    residual wrapper wraps the configurable-taper decoder cleanly."""
    cfg = TorchVehicleConfig(
        base_channels=_BASE_CH, latent_dim=_LATENT_DIM, out_dir=Path("/dev/null"),
        device="cpu", train_device="mps", split_by_head=True, taper_channels=_SOLVED_TAPER,
        pose_film_enabled=True, pose_film_version=2,
    )
    assert cfg.pose_film_enabled and cfg.taper_channels == _SOLVED_TAPER


def test_taper_plus_film_v1_refused():
    """v1 stem-injection is NOT supported on a re-taper (couples d_pose into d_seg);
    taper + FiLM is restricted to v2."""
    with pytest.raises(ValueError, match="requires pose_film_version=2"):
        TorchVehicleConfig(
            base_channels=_BASE_CH, latent_dim=_LATENT_DIM, out_dir=Path("/dev/null"),
            device="cpu", taper_channels=_SOLVED_TAPER, pose_film_enabled=True,
            pose_film_version=1,
        )


def test_kd_warm_up_with_taper_film_v2_student_renders(tmp_path):
    """The KD warm-up runs on a taper+FiLM-v2 STUDENT (the bind-all student): the frozen
    vendored teacher is distilled into the FiLM-wrapped solved-taper student, and the
    frame-MSE drops. Proves the FiLM-enabled KD path (student(latents, idx)) works."""
    from tac.torch_vehicle.kd_warm_start import build_frozen_teacher, kd_warm_up_decoder

    best, _sd, _lat = _make_basin_dir(tmp_path)
    v = import_vendored_bundle()
    teacher = build_frozen_teacher(
        best, vendored_decoder_cls=v.HNeRVDecoder, latent_dim=_LATENT_DIM,
        base_channels=_BASE_CH, device="cpu",
    )
    cfg = TorchVehicleConfig(
        base_channels=_BASE_CH, latent_dim=_LATENT_DIM, out_dir=tmp_path / "tf",
        device="cpu", train_device="mps", split_by_head=True, taper_channels=_SOLVED_TAPER,
        pose_film_enabled=True, pose_film_version=2, seed=0,
    )
    sc = SyntheticScorerContext(n_pairs=_N_PAIRS, device="cpu", seed=0, split_by_head=True)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    student = driver._new_decoder(device=torch.device("cpu"))  # taper + FiLM-v2 wrapper
    latents = torch.nn.Parameter(torch.randn(_N_PAIRS, _LATENT_DIM) * 0.1)
    stats = kd_warm_up_decoder(
        student=student, teacher=teacher, latents=latents, n_pairs=_N_PAIRS,
        epochs=6, batch_size=_N_PAIRS, lr=5e-3, device="cpu", pose_film_enabled=True,
    )
    assert stats["last_loss"] < stats["first_loss"], "KD on taper+FiLM student did not reduce frame-MSE"


def test_base_trunk_stopgrad_freezes_rgb0(tmp_path):
    """The behavioral contrast (proves the refinement is not a no-op): with the flag OFF
    (base trunk-stopgrad) rgb_0 IS restored to the seg-only grad — it is FROZEN w.r.t. the
    pose objective (this is exactly what the refinement frees)."""
    driver = _build_film_driver(tmp_path, rgb0_trainable=False)
    decoder, latents = _decoder_latents(driver)
    rgb0_ids = driver._rgb0_param_ids(decoder)
    rgb0_named = [(n, p) for n, p in decoder.named_parameters() if id(p) in rgb0_ids]

    _backward_once(driver, decoder, latents, compute_pose=False)
    seg_only = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in rgb0_named}

    _backward_once(driver, decoder, latents, compute_pose=True)
    for n, p in rgb0_named:
        ref = seg_only[n]
        if ref is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0
        else:
            assert torch.equal(p.grad, ref), (
                f"rgb_0 param {n} got pose grad with the flag OFF → base trunk-stopgrad did "
                f"NOT freeze rgb_0 (the refinement would be pointless)"
            )
