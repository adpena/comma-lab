# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the per-stage CHECKPOINT PRESERVATION + disk-hygiene manifest.

The operator's explicit anchor-hardening ask: the decisive prune-SOURCE run must
preserve, beyond the rolling resume checkpoint, (b) PER-STAGE snapshots so the
capacity-RD prune-path (or any fork) can start from the boundary of ANY completed
curriculum stage, and the certify-or-block preservation manifest for a lossless
cold-store/move (the "Local Disk / SSD spill / auto-cleanup / provenance"
non-negotiable).

The load-bearing claims these tests prove (NO-FAKE — each would FAIL if the feature
were a no-op):

* DEFAULT OFF is byte-identical: ``preserve_stage_snapshots=False`` writes NO
  ``stage_snapshots/`` dir + ``preservation_manifest=False`` writes NO manifest, and
  the trained state is bit-identical to a run with the flags absent.
* ON writes ONE preserved snapshot per COMPLETED stage, each a COMPLETE state
  (decoder + latents + EMA shadow + optimizer) loadable via the SAME
  ``load_checkpoint``.
* The snapshot state is BIT-IDENTICAL to the rolling checkpoint state captured at
  the same stage boundary (the snapshot is a true preserved copy, not a re-derived
  approximation).
* A FORK can RESUME from a stage snapshot and finish the curriculum bit-identically
  to an uninterrupted reference — the prune-ready fork-from-any-stage contract.
* Snapshots are IDEMPOTENT across a resume (a re-completed stage rewrites the same
  dir; the snapshot count stays bounded by the stage count, not the resume count).
* The preservation manifest records each durable artifact with bytes + SHA-256 + the
  rebuild command (the certify-or-block proof), and the SHA matches the file bytes.
* The ``best/`` dir holds the EMA shadow (decoder + latents) — the EXACT artifact the
  capacity-RD prune-path's load contract consumes.

Uses the SyntheticScorerContext (architecture-AGNOSTIC) so the multi-stage round-trip
is fast + deterministic — the real scorer is not needed to prove STATE preservation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tac.torch_vehicle.checkpoint import (
    checkpoint_exists,
    list_stage_snapshots,
    load_checkpoint,
    read_manifest,
    stage_snapshot_dir,
)
from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _stage(name: str, epochs: int) -> StageSpec:
    return StageSpec(
        name=name, epochs=epochs, seg_loss_fn=_ce, eval_every=1000, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _two_stage() -> list[StageSpec]:
    # Stage 0 (random latents) + stage 1 (carries from stage 0). Two boundaries =>
    # two preserved snapshots when the lever is on.
    return [
        _stage("stage_a", 3),
        StageSpec(
            name="stage_b", epochs=3, seg_loss_fn=_ce, eval_every=1000, batch_size=4,
            ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
            muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0,
            grad_clip_muon=1.0, lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
            cat_lambda=0.0, cat_sigma=0.2, use_qat=False, init_latents_random=False,
        ),
    ]


def _driver(out_dir, *, preserve=False, manifest=False, rebuild_command=None, seed=0):
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=Path(out_dir), checkpoint_every_epochs=1,
        device="cpu", train_device="cpu", split_by_head=False, seed=seed,
        preserve_stage_snapshots=preserve, preservation_manifest=manifest,
        rebuild_command=rebuild_command,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=seed)
    return TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=_two_stage(),
    )


def _state_hash_of(merged: dict) -> str:
    h = hashlib.sha256()
    for top in ("decoder", "ema_decoder"):
        for k in sorted(merged[top]):
            h.update(k.encode())
            h.update(np.ascontiguousarray(np.array(merged[top][k])).tobytes())
    for top in ("latents", "ema_latents"):
        h.update(np.ascontiguousarray(np.array(merged[top])).tobytes())
    return h.hexdigest()


def _run_hash(out_dir, *, preserve=False, manifest=False) -> str:
    d = _driver(out_dir, preserve=preserve, manifest=manifest)
    d.run()
    return _state_hash_of(load_checkpoint(out_dir))


# ---------------------------------------------------------------------------
# 1. DEFAULT OFF is byte-identical (the live-basin-unaffected contract).
# ---------------------------------------------------------------------------
def test_default_off_writes_no_snapshots_or_manifest(tmp_path):
    d = _driver(tmp_path / "off")
    d.run()
    assert not (tmp_path / "off" / "stage_snapshots").exists()
    assert not (tmp_path / "off" / "preservation_manifest.json").exists()


def test_preservation_is_byte_identical_to_default(tmp_path):
    """ON vs OFF => bit-identical trained state. The preservation IO must NOT touch
    the training math (it only ADDS preserved copies of the SAME captured state)."""
    off = _run_hash(tmp_path / "off", preserve=False, manifest=False)
    on = _run_hash(tmp_path / "on", preserve=True, manifest=True)
    assert on == off, (
        "preserve_stage_snapshots / preservation_manifest changed the trained state — "
        f"it is NOT a pure additive-IO preservation. off={off} on={on}"
    )


def test_config_defaults_are_off():
    cfg = TorchVehicleConfig(base_channels=8, latent_dim=28, device="cpu", train_device="cpu")
    assert cfg.preserve_stage_snapshots is False
    assert cfg.preservation_manifest is False
    assert cfg.rebuild_command is None


# ---------------------------------------------------------------------------
# 2. ON writes ONE complete loadable snapshot per completed stage.
# ---------------------------------------------------------------------------
def test_one_snapshot_per_completed_stage(tmp_path):
    d = _driver(tmp_path / "run", preserve=True)
    d.run()
    snaps = list_stage_snapshots(tmp_path / "run")
    assert len(snaps) == 2, f"expected 2 stage snapshots (2 stages), got {len(snaps)}: {snaps}"
    # Canonical dir names, in curriculum order.
    assert snaps[0] == stage_snapshot_dir(tmp_path / "run", 0, "stage_a")
    assert snaps[1] == stage_snapshot_dir(tmp_path / "run", 1, "stage_b")
    # Each is a COMPLETE loadable checkpoint with the full state.
    for i, snap in enumerate(snaps):
        assert checkpoint_exists(snap)
        merged = load_checkpoint(snap)
        for key in ("decoder", "latents", "ema_decoder", "ema_latents", "adamw"):
            assert key in merged, f"snapshot {snap} missing {key}"
        man = read_manifest(snap)
        # The snapshot manifest marks itself + records the completed-stage position.
        assert man["extra"]["is_stage_snapshot"] is True
        assert man["extra"]["snapshot_stage_index"] == i
        assert man["stage_index"] == i


def test_snapshot_is_bit_identical_to_rolling_at_boundary(tmp_path):
    """The LAST stage snapshot must equal the rolling checkpoint state at run end
    (both captured at the final stage boundary from the SAME _capture_state)."""
    d = _driver(tmp_path / "run", preserve=True)
    d.run()
    rolling = _state_hash_of(load_checkpoint(tmp_path / "run"))
    last_snap = list_stage_snapshots(tmp_path / "run")[-1]
    snap = _state_hash_of(load_checkpoint(last_snap))
    assert snap == rolling, (
        "the final stage snapshot is NOT bit-identical to the rolling checkpoint at "
        f"the same boundary — preservation re-derived state instead of copying it. "
        f"rolling={rolling} snapshot={snap}"
    )


# ---------------------------------------------------------------------------
# 3. A FORK can RESUME from a stage snapshot (the prune-ready contract).
# ---------------------------------------------------------------------------
def test_fork_resumes_from_stage_snapshot_bit_identical(tmp_path):
    """Restore the stage-0 snapshot into a fresh out_dir and finish the curriculum;
    the result must match an uninterrupted reference bit-for-bit. This is the
    fork-from-any-stage / prune-source contract the operator asked for."""
    # Reference: uninterrupted run.
    ref = _driver(tmp_path / "ref", preserve=True)
    ref.run()
    ref_hash = _state_hash_of(load_checkpoint(tmp_path / "ref"))

    # Copy the stage-0 snapshot into a fresh fork dir as that fork's rolling
    # checkpoint (the snapshot layout IS the rolling checkpoint layout), then resume.
    snap0 = stage_snapshot_dir(tmp_path / "ref", 0, "stage_a")
    fork = tmp_path / "fork"
    fork.mkdir()
    import shutil

    for fn in ("torch_vehicle_checkpoint_state.pt", "torch_vehicle_checkpoint_manifest.json"):
        shutil.copy(snap0 / fn, fork / fn)
    # A resume from the stage-0 snapshot starts stage 1 (position (0, epochs) -> next).
    forked = _driver(fork, preserve=False)
    forked.run()
    fork_hash = _state_hash_of(load_checkpoint(fork))
    assert fork_hash == ref_hash, (
        "a fork resumed from the stage-0 snapshot did NOT reproduce the reference — "
        f"the snapshot is not a faithful resume point. ref={ref_hash} fork={fork_hash}"
    )


# ---------------------------------------------------------------------------
# 4. Idempotent across a resume (snapshot count bounded by stage count).
# ---------------------------------------------------------------------------
def test_snapshots_idempotent_across_resume(tmp_path):
    """A run that dies mid-stage-1 and resumes must NOT accumulate duplicate
    stage snapshots — the re-completed stage rewrites the SAME dir."""
    from tac.torch_vehicle.driver import _SimulatedDeath

    import pytest

    d = _driver(tmp_path / "run", preserve=True)
    # Die after stage 0 completes (3 epochs) + 1 epoch into stage 1 => stage-0
    # snapshot already written; stage-1 not yet.
    d._stop_after_global_epoch = 4
    with pytest.raises(_SimulatedDeath):
        d.run()
    assert len(list_stage_snapshots(tmp_path / "run")) == 1
    # Resume + finish.
    d2 = _driver(tmp_path / "run", preserve=True)
    d2.run()
    # Exactly 2 (one per stage) — NOT 3 (no duplicate stage-0 from the resume).
    assert len(list_stage_snapshots(tmp_path / "run")) == 2


# ---------------------------------------------------------------------------
# 5. The preservation manifest records bytes + SHA-256 + rebuild command.
# ---------------------------------------------------------------------------
def test_preservation_manifest_records_artifacts_with_verified_sha(tmp_path):
    cmd = "experiments/launch_split_by_head_basin.py --base-channels 36 --foo bar"
    # eval_every=1 so an eval fires and the best/ EMA shadow is written → recorded.
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", train_device="cpu", split_by_head=False,
        seed=0, preserve_stage_snapshots=True, preservation_manifest=True,
        rebuild_command=cmd,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    evalled = [_stage("stage_a", 2), StageSpec(
        name="stage_b", epochs=2, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=False,
    )]
    evalled[0] = StageSpec(
        name="stage_a", epochs=2, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )
    d = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=evalled,
    )
    d.run()
    mpath = tmp_path / "run" / "preservation_manifest.json"
    assert mpath.exists(), "preservation_manifest=True must write the manifest"
    man = json.loads(mpath.read_text())
    assert man["rebuild_command"] == cmd
    assert man["config"]["base_channels"] == 8
    assert man["artifacts"], "manifest must enumerate the durable artifacts"
    # The SHA-256 of each recorded artifact must match the file bytes (NO-FAKE: the
    # manifest is a real custody record, not a fabricated hash).
    kinds = {a["kind"] for a in man["artifacts"]}
    assert any(k.startswith("rolling_resume") for k in kinds)
    assert any(k.startswith("stage_snapshot_state") for k in kinds)
    assert "best_ema_decoder" in kinds  # the prune-ready EMA shadow is recorded
    total = 0
    for a in man["artifacts"]:
        p = Path(a["path"])
        assert p.exists(), f"manifest cites a non-existent artifact: {p}"
        assert a["bytes"] == p.stat().st_size
        total += a["bytes"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert a["sha256"] == h, f"manifest SHA mismatch for {p}"
    assert man["total_bytes"] == total


# ---------------------------------------------------------------------------
# 6. The best/ dir holds the EMA shadow (the prune-path load contract).
# ---------------------------------------------------------------------------
def test_best_dir_holds_ema_shadow_for_prune_path(tmp_path):
    """The capacity-RD prune-path loads best/best_ema_decoder.pt + best_ema_latents.pt
    (the EMA shadow = the inference/export weights). Prove the run writes exactly that
    pair, loadable as the decoder state_dict + the latent tensor."""
    # eval_every=1 so an eval fires and the best/ dir is written.
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", train_device="cpu",
        split_by_head=False, seed=0,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    stage = StageSpec(
        name="s", epochs=2, seg_loss_fn=_ce, eval_every=1, batch_size=4, ema_decay=0.999,
        use_muon=False, adamw_lr=1e-3, muon_lr=2e-4, muon_weight_decay=0.0,
        latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0, lr_floor_ratio=5e-6,
        seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0, cat_sigma=0.2, use_qat=False,
        init_latents_random=True,
    )
    d = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[stage],
    )
    d.run()
    best = tmp_path / "run" / "best"
    dec = best / "best_ema_decoder.pt"
    lat = best / "best_ema_latents.pt"
    assert dec.exists() and lat.exists(), "best/ must hold the EMA shadow for the prune-path"
    sd = torch.load(dec, map_location="cpu", weights_only=False)
    assert isinstance(sd, dict) and sd, "best_ema_decoder.pt must be a non-empty state_dict"
    latents = torch.load(lat, map_location="cpu", weights_only=False)
    assert tuple(latents.shape) == (6, 28), "best_ema_latents.pt must be (n_pairs, latent_dim)"
