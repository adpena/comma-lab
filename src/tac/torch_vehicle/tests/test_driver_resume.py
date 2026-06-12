# SPDX-License-Identifier: MIT
"""NO-FAKE kill+restart equivalence tests for the torch-vehicle driver.

The load-bearing claim (MLX parity): a run that DIES mid-stage and RESUMES from
the latest checkpoint continues the EXACT same descent trajectory as an
uninterrupted run. A stub that saved only the params (and silently dropped the
AdamW/Muon momentum, the EMA shadow, or the RNG state) would FAIL these tests —
the resumed trajectory would diverge after the first post-resume step.

The decisive test (``test_resume_is_bit_identical``) trains a reference driver
continuously, then trains a second driver to a checkpoint point, SAVES, builds a
FRESH driver, LOADS, runs the remaining epochs — and the two final states must
be bit-identical (decoder, latents, EMA shadow, optimizer momentum).

``test_real_subprocess_kill_restart`` does the REAL thing the prompt demands: a
child process trains N epochs and is HARD-KILLED (SIGKILL) mid-run; a second
child resumes from the durable checkpoint; the resumed final state matches an
uninterrupted reference's final state. A death costs at most one checkpoint.

These use the SyntheticScorerContext (architecture-AGNOSTIC resume round-trip)
so they are fast + deterministic — the real EfficientNet scorer is not required
to prove STATE serializes + restores bit-exactly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.torch_vehicle.checkpoint import checkpoint_exists, is_done, read_manifest
from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce_seg_loss(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _short_curriculum(total_epochs: int, eval_every: int = 1000) -> list[StageSpec]:
    """A 1-stage tiny curriculum (AdamW-only CE) for fast deterministic tests.

    eval_every is set high to skip the (slow, scorer-bound) eval in the bit-exact
    state tests — we only need the TRAINING-STEP state to round-trip. A separate
    test exercises the eval/BEST path."""
    return [
        StageSpec(
            name="test_stage_ce",
            epochs=total_epochs,
            seg_loss_fn=_ce_seg_loss,
            eval_every=eval_every,
            batch_size=4,
            ema_decay=0.999,
            use_muon=False,
            adamw_lr=1e-3,
            muon_lr=2e-4,
            muon_weight_decay=0.0,
            latent_lr_mult=10.0,
            grad_clip=1.0,
            grad_clip_muon=1.0,
            lr_floor_ratio=5e-6,
            seg_weight=100.0,
            pose_weight=1.0,
            cat_lambda=0.0,
            cat_sigma=0.2,
            use_qat=False,
            init_latents_random=True,
        )
    ]


def _muon_curriculum(total_epochs: int) -> list[StageSpec]:
    """A 1-stage Muon curriculum so the Muon momentum-buffer round-trip is tested
    (a stub that saved only AdamW state would diverge here)."""
    return [
        StageSpec(
            name="test_stage_muon",
            epochs=total_epochs,
            seg_loss_fn=_ce_seg_loss,
            eval_every=1000,
            batch_size=4,
            ema_decay=0.999,
            use_muon=True,
            adamw_lr=1e-5,
            muon_lr=2e-4,
            muon_weight_decay=5e-4,
            latent_lr_mult=10.0,
            grad_clip=1.0,
            grad_clip_muon=1.0,
            lr_floor_ratio=5e-6,
            seg_weight=100.0,
            pose_weight=1.0,
            cat_lambda=0.0,
            cat_sigma=0.1,
            use_qat=False,
            init_latents_random=True,
        )
    ]


def _build_driver(out_dir, curriculum, *, base_ch=8, n_pairs=6, seed=0):
    cfg = TorchVehicleConfig(
        base_channels=base_ch,
        latent_dim=28,
        out_dir=Path(out_dir),
        checkpoint_every_epochs=1,
        device="cpu",
        seed=seed,
    )
    scorer = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=seed)
    vendored = import_vendored_bundle()
    return TorchVehicleDriver(cfg, scorer=scorer, vendored=vendored, curriculum=curriculum)


def _final_state(driver):
    """Snapshot the driver's final mutable state as numpy for bit-exact compare.

    Re-loads from the durable checkpoint the driver just wrote (the resume
    surface) so we compare exactly what a resumed run would see."""
    from tac.torch_vehicle.checkpoint import load_checkpoint

    merged = load_checkpoint(driver.cfg.out_dir)
    out = {}
    for k, v in merged["decoder"].items():
        out[f"dec.{k}"] = np.array(v)
    out["latents"] = np.array(merged["latents"])
    for k, v in merged["ema_decoder"].items():
        out[f"ema.{k}"] = np.array(v)
    out["ema_latents"] = np.array(merged["ema_latents"])
    return out, merged["adamw"], merged.get("muon")


def _assert_state_equal(a, b, *, atol=0.0):
    fa, adamw_a, muon_a = a
    fb, adamw_b, muon_b = b
    assert set(fa) == set(fb), f"key mismatch: {set(fa) ^ set(fb)}"
    for k in fa:
        np.testing.assert_allclose(fa[k], fb[k], atol=atol, err_msg=f"array {k}")
    # AdamW per-param exp_avg / exp_avg_sq + step.
    sa, sb = adamw_a["state"], adamw_b["state"]
    assert set(sa) == set(sb)
    for pid in sa:
        for field in ("exp_avg", "exp_avg_sq"):
            if field in sa[pid]:
                np.testing.assert_allclose(
                    np.array(sa[pid][field]), np.array(sb[pid][field]),
                    atol=atol, err_msg=f"adamw {pid}.{field}",
                )
    # Muon momentum buffers.
    if muon_a is not None and muon_b is not None:
        ma, mb = muon_a["state"], muon_b["state"]
        assert set(ma) == set(mb)
        for pid in ma:
            if "momentum_buffer" in ma[pid]:
                np.testing.assert_allclose(
                    np.array(ma[pid]["momentum_buffer"]),
                    np.array(mb[pid]["momentum_buffer"]),
                    atol=atol, err_msg=f"muon {pid}.momentum_buffer",
                )


def _run_with_simulated_death(out_dir, curriculum, *, die_at, **kw):
    """Run the IDENTICAL curriculum but die (raise) after ``die_at`` global epochs,
    AFTER the checkpoint lands. Returns the driver (checkpoint on disk)."""
    from tac.torch_vehicle.driver import _SimulatedDeath

    d = _build_driver(out_dir, curriculum, **kw)
    d._stop_after_global_epoch = die_at
    with pytest.raises(_SimulatedDeath):
        d.run()
    assert not is_done(d.cfg.out_dir), "death must NOT mark the run done"
    return d


def test_resume_is_bit_identical_adamw(tmp_path):
    """THE decisive kill+restart test (AdamW path): resumed == uninterrupted.

    Both runs use the IDENTICAL 8-epoch curriculum (so the cosine LR schedule —
    which depends on spec.epochs — is the same). The interrupted run dies after
    5 epochs (checkpoint landed), then a FRESH driver resumes the SAME curriculum
    and finishes. The final state must match the uninterrupted reference."""
    ref = _build_driver(tmp_path / "ref", _short_curriculum(8), seed=0)
    ref.run()
    ref_state = _final_state(ref)

    # Die after 5 epochs of the SAME 8-epoch curriculum.
    _run_with_simulated_death(tmp_path / "run", _short_curriculum(8), die_at=5, seed=0)
    # Resume the SAME 8-epoch curriculum from the durable checkpoint.
    b = _build_driver(tmp_path / "run", _short_curriculum(8), seed=0)
    b.run()
    b_state = _final_state(b)

    _assert_state_equal(ref_state, b_state, atol=0.0)


def test_resume_is_bit_identical_muon(tmp_path):
    """Muon path: the momentum buffer must round-trip (a stub saving only AdamW
    state would diverge here)."""
    ref = _build_driver(tmp_path / "ref", _muon_curriculum(6), seed=1)
    ref.run()
    ref_state = _final_state(ref)

    _run_with_simulated_death(tmp_path / "run", _muon_curriculum(6), die_at=4, seed=1)
    b = _build_driver(tmp_path / "run", _muon_curriculum(6), seed=1)
    b.run()
    b_state = _final_state(b)
    _assert_state_equal(ref_state, b_state, atol=0.0)


def _multi_stage_curriculum() -> list[StageSpec]:
    """3-stage curriculum (AdamW -> AdamW -> Muon) so a resume that CROSSES a
    stage transition (incl. the AdamW->Muon optimizer switch) is tested."""

    def stage(name, ep, muon=False):
        return StageSpec(
            name=name, epochs=ep, seg_loss_fn=_ce_seg_loss, eval_every=1000, batch_size=4,
            ema_decay=0.999, use_muon=muon, adamw_lr=(1e-5 if muon else 1e-3), muon_lr=2e-4,
            muon_weight_decay=(5e-4 if muon else 0.0), latent_lr_mult=10.0, grad_clip=1.0,
            grad_clip_muon=1.0, lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
            cat_lambda=0.0, cat_sigma=0.2, use_qat=False, init_latents_random=(name == "s1"),
        )

    return [stage("s1", 3), stage("s2", 3), stage("s3", 3, muon=True)]


def _qat_c1a_curriculum(total_epochs: int) -> list[StageSpec]:
    """A 1-stage QAT + C1a-entropy curriculum (the stages 5-8 weight-domain
    mechanism path), using the REAL vendored l7_softplus seg loss, apply_qat /
    restore_qat, and cat_entropy_v2. A resume through this most-complex stage must
    be bit-identical (the QAT apply/restore-around-forward + entropy gradient must
    round-trip)."""
    from tac.torch_vehicle.vendored_imports import import_vendored

    losses = import_vendored("losses")

    def l7(s, t):
        return losses.l7_softplus_seg_loss(s, t, tau=0.3, l7_threshold=1.0, l7_mult=4.0)

    return [
        StageSpec(
            name="test_stage_qat_c1a", epochs=total_epochs, seg_loss_fn=l7, eval_every=1000,
            batch_size=4, ema_decay=0.999, use_muon=False, adamw_lr=3e-5, muon_lr=2e-4,
            muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
            lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.01,
            cat_sigma=0.2, use_qat=True, init_latents_random=True,
        )
    ]


def test_resume_bit_identical_through_qat_c1a_stage(tmp_path):
    """A death + resume THROUGH a QAT + C1a stage (the most complex weight-domain
    mechanism) must be bit-identical (the real l7/apply_qat/cat_entropy_v2 path)."""
    ref = _build_driver(tmp_path / "ref", _qat_c1a_curriculum(4), seed=0)
    ref.run()
    ref_state = _final_state(ref)

    _run_with_simulated_death(tmp_path / "run", _qat_c1a_curriculum(4), die_at=2, seed=0)
    b = _build_driver(tmp_path / "run", _qat_c1a_curriculum(4), seed=0)
    b.run()
    b_state = _final_state(b)
    _assert_state_equal(ref_state, b_state, atol=0.0)


def test_resume_bit_identical_across_stage_transition(tmp_path):
    """A death MID-stage-2 must resume + cross into the Muon stage with the EXACT
    same trajectory as an uninterrupted run — the real-world multi-stage case.

    A stub that dropped the EMA-shadow / decoder / latents carry across the stage
    boundary (or rebuilt the wrong optimizer on resume) would diverge here."""
    ref = _build_driver(tmp_path / "ref", _multi_stage_curriculum(), seed=2)
    ref.run()
    ref_state = _final_state(ref)

    # Die at global epoch 5 (mid stage 2, before the Muon stage 3).
    _run_with_simulated_death(tmp_path / "run", _multi_stage_curriculum(), die_at=5, seed=2)
    b = _build_driver(tmp_path / "run", _multi_stage_curriculum(), seed=2)
    b.run()
    b_state = _final_state(b)
    _assert_state_equal(ref_state, b_state, atol=0.0)


def test_resume_bit_identical_death_at_exact_stage_boundary(tmp_path):
    """A Modal preemption EXACTLY at a stage boundary (the AdamW->Muon transition)
    must resume bit-identically. The end-of-stage checkpoint records
    (stage_index, spec.epochs); the resumed run runs an empty epoch range for the
    completed stage, carries forward, and proceeds into the next (Muon) stage."""
    ref = _build_driver(tmp_path / "ref", _multi_stage_curriculum(), seed=4)
    ref.run()
    ref_state = _final_state(ref)

    # s1+s2+s3 = 3+3+3; die at global epoch 3 = end of stage 1 (the boundary).
    _run_with_simulated_death(tmp_path / "run", _multi_stage_curriculum(), die_at=3, seed=4)
    b = _build_driver(tmp_path / "run", _multi_stage_curriculum(), seed=4)
    b.run()
    b_state = _final_state(b)
    _assert_state_equal(ref_state, b_state, atol=0.0)


def test_checkpoint_written_every_epoch(tmp_path):
    d = _build_driver(tmp_path / "run", _short_curriculum(3), seed=0)
    d.run()
    assert checkpoint_exists(d.cfg.out_dir)
    man = read_manifest(d.cfg.out_dir)
    assert man["base_channels"] == 8
    assert man["n_pairs"] == 6


def test_resume_rejects_mismatched_basis(tmp_path):
    _run_with_simulated_death(tmp_path / "run", _short_curriculum(4), die_at=2, base_ch=8, seed=0)
    # A different base_channels cannot resume.
    other = _build_driver(tmp_path / "run", _short_curriculum(4), base_ch=12, seed=0)
    with pytest.raises(ValueError, match="cannot resume a different basis"):
        other.run()


def test_resume_rejects_mismatched_n_pairs(tmp_path):
    _run_with_simulated_death(tmp_path / "run", _short_curriculum(4), die_at=2, n_pairs=6, seed=0)
    other = _build_driver(tmp_path / "run", _short_curriculum(4), n_pairs=4, seed=0)
    with pytest.raises(ValueError, match="cannot resume a different basis"):
        other.run()


def test_already_done_is_idempotent(tmp_path):
    d = _build_driver(tmp_path / "run", _short_curriculum(2), seed=0)
    d.run()
    # A second run() with the DONE marker present is a no-op.
    d2 = _build_driver(tmp_path / "run", _short_curriculum(2), seed=0)
    out = d2.run()
    assert out["status"] == "already_done"


# ---------------------------------------------------------------------------
# REAL subprocess SIGKILL + restart (the prompt's "real kill+restart" mandate)
# ---------------------------------------------------------------------------
_CHILD_SCRIPT = '''
import sys, time
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, {repo_src!r})
from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import TorchVehicleConfig, TorchVehicleDriver, import_vendored_bundle
from tac.torch_vehicle.scorer_context import SyntheticScorerContext

def ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)

def curric(n):
    return [StageSpec(name="ce", epochs=n, seg_loss_fn=ce, eval_every=10000, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4, muon_weight_decay=0.0,
        latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0, lr_floor_ratio=5e-6,
        seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0, cat_sigma=0.2, use_qat=False,
        init_latents_random=True)]

out_dir = sys.argv[1]
total = int(sys.argv[2])
mode = sys.argv[3]  # "slow" (sleeps so the parent can SIGKILL) or "finish"
cfg = TorchVehicleConfig(base_channels=8, latent_dim=28, out_dir=Path(out_dir),
                         checkpoint_every_epochs=1, device="cpu", seed=0)
scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
drv = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curric(total))
if mode == "slow":
    # Patch the per-epoch hook to sleep so the parent can kill mid-run AFTER a
    # checkpoint has landed.
    orig = drv._train_one_epoch
    def slow_epoch(rt, spec, *args, **kwargs):
        r = orig(rt, spec, *args, **kwargs)
        (Path(out_dir)/"_epoch_done").write_text("1")
        time.sleep(2.0)
        return r
    drv._train_one_epoch = slow_epoch
drv.run()
(Path(out_dir)/"_clean_exit").write_text("1")
'''


@pytest.mark.timeout(120)
def test_real_subprocess_kill_restart(tmp_path):
    """REAL kill+restart: a child trains, is SIGKILLed mid-run after a checkpoint
    lands, a second child resumes, and the resumed final state matches an
    uninterrupted reference."""
    repo_src = str(Path(__file__).resolve().parents[4] / "src")
    script = tmp_path / "child.py"
    script.write_text(_CHILD_SCRIPT.format(repo_src=repo_src))

    # 1) Uninterrupted reference (separate dir).
    ref_dir = tmp_path / "ref"
    subprocess.run(
        [sys.executable, str(script), str(ref_dir), "6", "finish"],
        check=True, capture_output=True, timeout=90,
    )
    assert (ref_dir / "_clean_exit").exists()
    from tac.torch_vehicle.checkpoint import load_checkpoint

    ref_merged = load_checkpoint(ref_dir)

    # 2) Interrupted run: launch "slow", wait until >=1 epoch checkpoint lands,
    # then SIGKILL.
    run_dir = tmp_path / "run"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(run_dir), "6", "slow"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait for at least 2 epochs to have checkpointed (so resume is mid-run).
    deadline = time.time() + 60
    while time.time() < deadline:
        if checkpoint_exists(run_dir):
            man = read_manifest(run_dir)
            if man["epoch_in_stage"] >= 2:
                break
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"child exited early: {err.decode()[:2000]}")
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("child never reached epoch>=2 checkpoint")

    killed_at = read_manifest(run_dir)["epoch_in_stage"]
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert not is_done(run_dir), "run should NOT be marked done — it was killed"
    assert not (run_dir / "_clean_exit").exists()

    # 3) Resume: a fresh child finishes the curriculum from the durable checkpoint.
    r = subprocess.run(
        [sys.executable, str(script), str(run_dir), "6", "finish"],
        capture_output=True, timeout=90,
    )
    assert r.returncode == 0, f"resume child failed: {r.stderr.decode()[:2000]}"
    assert is_done(run_dir)
    resumed_merged = load_checkpoint(run_dir)

    # 4) The resumed final decoder + EMA must match the uninterrupted reference
    # bit-for-bit (the death cost <= 1 epoch and did not perturb the trajectory).
    for k in ref_merged["decoder"]:
        np.testing.assert_allclose(
            np.array(resumed_merged["decoder"][k]),
            np.array(ref_merged["decoder"][k]),
            atol=0.0, err_msg=f"resumed decoder.{k} != reference (killed at ep {killed_at})",
        )
    for k in ref_merged["ema_decoder"]:
        np.testing.assert_allclose(
            np.array(resumed_merged["ema_decoder"][k]),
            np.array(ref_merged["ema_decoder"][k]),
            atol=0.0, err_msg=f"resumed ema.{k} != reference",
        )
    np.testing.assert_allclose(
        np.array(resumed_merged["latents"]), np.array(ref_merged["latents"]), atol=0.0
    )
