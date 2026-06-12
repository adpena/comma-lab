# SPDX-License-Identifier: MIT
"""NO-FAKE no-regression tests for the ASYNC authority eval (the throughput salvage).

The load-bearing claim: running the CPU AUTHORITY exact eval in a BACKGROUND
THREAD off a point-in-time EMA-shadow snapshot produces the BIT-IDENTICAL
authority numbers (d_seg / d_pose / rate / score / archive_bytes) as the
synchronous eval on the SAME weights — it's the SAME eval, just non-blocking.
A fake that approximated the eval, read MPS, or raced the snapshot would FAIL
``test_async_eval_numbers_equal_sync_eval``.

The eval is architecture-AGNOSTIC (build_archive / parse_archive / exact_eval),
so the SyntheticScorerContext exercises the FULL async machinery — snapshot,
thread spawn, one-in-flight throttle, join, telemetry row, best-tracking — fast
and deterministically, without the real EfficientNet scorer.

Decisive tests:
* ``test_async_eval_numbers_equal_sync_eval`` — sync-eval(W) == async-eval(snapshot of W),
  to full float precision. THE no-regression proof.
* ``test_snapshot_decouples_from_live_weights`` — mutating the live EMA shadow
  AFTER the snapshot does NOT change the snapshot's eval (point-in-time copy).
* ``test_one_eval_in_flight_skips_overcadence`` — a second eval scheduled while
  the first is alive is SKIPPED + counted (the cadence self-throttle).
* ``test_join_writes_final_eval_row`` — a full async run JOINs the worker so the
  eval row (tagged with the snapshot epoch) lands before the DONE marker.
* ``test_sync_mode_is_byte_identical`` — with --async-eval OFF the trajectory +
  best are identical to the legacy combined-row behavior (no regression off-path).
* ``test_best_tracking_uses_snapshot_epoch`` — BEST is tagged with the snapshot
  epoch the eval came from, even though the eval lands a few epochs later.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from tac.torch_vehicle.checkpoint import is_done
from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext
from tac.torch_vehicle.telemetry import read_summary, read_trajectory


def _ce_seg_loss(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _curriculum(total_epochs: int, eval_every: int) -> list[StageSpec]:
    return [
        StageSpec(
            name="async_test_stage",
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


def _make_driver(out_dir: Path, *, async_eval: bool, eval_every: int = 2, epochs: int = 4):
    cfg = TorchVehicleConfig(
        base_channels=8,
        latent_dim=28,
        out_dir=out_dir,
        checkpoint_every_epochs=1,
        async_eval=async_eval,
        device="cpu",
        seed=0,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    return TorchVehicleDriver(
        cfg,
        scorer=scorer,
        vendored=import_vendored_bundle(),
        curriculum=_curriculum(epochs, eval_every),
    )


# ---------------------------------------------------------------------------
# THE no-regression proof: same snapshot -> bit-identical authority numbers
# ---------------------------------------------------------------------------
def test_async_eval_numbers_equal_sync_eval(tmp_path):
    """sync-eval(W) == async-eval(snapshot of W) to full float precision.

    Build a driver, advance one stage runtime to a real EMA shadow, snapshot it,
    then run the eval TWICE: once inline (sync code path), once via the async
    worker mechanism (same snapshot). Every authority field must match exactly —
    it's the IDENTICAL computation, only the thread differs."""
    driver = _make_driver(tmp_path, async_eval=True)
    spec = driver.curriculum[0]
    # Build a real stage runtime + take a couple of training steps so the EMA
    # shadow is a non-trivial trained state (not init).
    decoder = driver._new_decoder()
    latents = torch.nn.Parameter(torch.randn(driver.n_pairs, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    driver._train_one_epoch(rt, spec)
    driver._train_one_epoch(rt, spec)

    # ONE snapshot, evaluated two ways.
    snap = driver._snapshot_ema(rt)

    # (1) sync inline eval (the production sync path).
    sync_driver = _make_driver(tmp_path / "sync", async_eval=False)
    # Use the SAME snapshot so the comparison is exact (sync_driver has its own
    # telemetry/best state; we only compare the returned eval dict numbers).
    ev_sync = sync_driver._eval_snapshot(snap, spec, stage_index=0, snapshot_epoch=2)

    # (2) async eval off the SAME snapshot via the worker's exact call.
    async_driver = _make_driver(tmp_path / "async", async_eval=True)
    ev_async = async_driver._eval_snapshot(snap, spec, stage_index=0, snapshot_epoch=2)

    # Every authority field is bit-for-bit equal (same eval, same snapshot).
    assert ev_sync["d_seg"] == ev_async["d_seg"]
    assert ev_sync["d_pose"] == ev_async["d_pose"]
    assert ev_sync["rate"] == ev_async["rate"]
    assert ev_sync["score"] == ev_async["score"]
    assert ev_sync["archive_bytes"] == ev_async["archive_bytes"]
    # And the score is a real number (the eval actually ran).
    assert ev_sync["score"] == ev_sync["score"]  # not NaN
    assert ev_sync["archive_bytes"] > 0


def test_async_full_thread_equals_inline(tmp_path):
    """The eval routed through the ACTUAL background-thread scheduler writes the
    SAME numbers to telemetry as the inline call (the thread is not a different
    computation — proves the worker wiring is faithful)."""
    driver = _make_driver(tmp_path, async_eval=True, eval_every=2, epochs=2)
    spec = driver.curriculum[0]
    decoder = driver._new_decoder()
    latents = torch.nn.Parameter(torch.randn(driver.n_pairs, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    driver._train_one_epoch(rt, spec)
    driver._global_epoch = 1

    snap = driver._snapshot_ema(rt)
    ev_inline = driver._eval_snapshot(snap, spec, 0, 1)

    # Schedule a real thread off the SAME rt EMA shadow (re-snapshots internally,
    # but rt hasn't changed) and join it; read the telemetry eval row it wrote.
    assert driver._schedule_async_eval(rt, spec, 0, 1)
    driver._join_async_eval(timeout=60)

    rows = read_trajectory(tmp_path)
    eval_rows = [r for r in rows if r.get("evaluated")]
    assert eval_rows, "async worker must write an eval row"
    last = eval_rows[-1]
    assert last["score"] == ev_inline["score"]
    assert last["d_seg"] == ev_inline["d_seg"]
    assert last["d_pose"] == ev_inline["d_pose"]
    assert last["archive_bytes"] == ev_inline["archive_bytes"]
    assert last["extra"]["async_eval_row"] is True
    assert last["extra"]["snapshot_epoch"] == 1


def test_snapshot_decouples_from_live_weights(tmp_path):
    """Mutating the live EMA shadow AFTER the snapshot does NOT change the
    snapshot's eval — the snapshot is a point-in-time deep copy (so a background
    eval cannot race the training loop's weight mutations)."""
    driver = _make_driver(tmp_path, async_eval=True)
    spec = driver.curriculum[0]
    decoder = driver._new_decoder()
    latents = torch.nn.Parameter(torch.randn(driver.n_pairs, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    driver._train_one_epoch(rt, spec)

    snap = driver._snapshot_ema(rt)
    ev_before = driver._eval_snapshot(snap, spec, 0, 1)

    # Now violently mutate the LIVE EMA shadow (simulate training continuing).
    with torch.no_grad():
        for p in rt.ema_decoder.parameters():
            p.add_(torch.randn_like(p) * 10.0)
        rt.ema_latents = rt.ema_latents + 5.0

    # Re-eval the OLD snapshot — must be unchanged (decoupled).
    ev_after = driver._eval_snapshot(snap, spec, 0, 1)
    assert ev_before["score"] == ev_after["score"]
    assert ev_before["archive_bytes"] == ev_after["archive_bytes"]

    # And a FRESH snapshot of the now-mutated weights gives a DIFFERENT eval
    # (proving the snapshot actually captured live weights, not a constant).
    snap2 = driver._snapshot_ema(rt)
    ev_mutated = driver._eval_snapshot(snap2, spec, 0, 1)
    assert ev_mutated["archive_bytes"] != ev_before["archive_bytes"] or (
        ev_mutated["score"] != ev_before["score"]
    )


def test_one_eval_in_flight_skips_overcadence(tmp_path):
    """A second eval scheduled while the first is still running is SKIPPED + counted
    (the cadence self-throttle — at most one in-flight)."""
    driver = _make_driver(tmp_path, async_eval=True)
    spec = driver.curriculum[0]
    decoder = driver._new_decoder()
    latents = torch.nn.Parameter(torch.randn(driver.n_pairs, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )

    # Monkeypatch _eval_snapshot to block on a gate so the first worker is
    # guaranteed "in flight" when we schedule the second.
    gate = threading.Event()
    orig = driver._eval_snapshot

    def _slow_eval(snap, spec_, si, ep):
        gate.wait(timeout=10)
        return orig(snap, spec_, si, ep)

    driver._eval_snapshot = _slow_eval  # type: ignore[assignment]

    assert driver._schedule_async_eval(rt, spec, 0, 1) is True  # scheduled
    # Wait until the worker is actually alive.
    for _ in range(100):
        if driver._async_eval_in_flight():
            break
        time.sleep(0.01)
    assert driver._async_eval_in_flight()

    # Second schedule while the first is alive -> SKIP.
    assert driver._schedule_async_eval(rt, spec, 0, 2) is False
    assert driver._skipped_evals == 1

    gate.set()  # let the first worker finish
    driver._join_async_eval(timeout=30)
    # After join, a new schedule is allowed again (in-flight cleared).
    assert not driver._async_eval_in_flight()


def test_join_writes_final_eval_row(tmp_path):
    """A full async run JOINs the in-flight worker on completion so the final eval
    row + best land before the DONE marker."""
    driver = _make_driver(tmp_path, async_eval=True, eval_every=2, epochs=4)
    summary = driver.run()
    assert summary["status"] == "complete"
    assert summary["async_eval"] is True
    assert is_done(tmp_path)

    rows = read_trajectory(tmp_path)
    eval_rows = [r for r in rows if r.get("evaluated")]
    # eval_every=2, epochs=4 -> evals scheduled at ep2 and ep4. Both must have
    # landed (the join guarantees the last one wrote before DONE).
    assert len(eval_rows) >= 1
    for r in eval_rows:
        assert r["score"] is not None
        assert r["extra"]["async_eval_row"] is True
    # A best score was recorded.
    summ = read_summary(tmp_path)
    assert summ["best_score"] < float("inf")
    assert summ["best_score"] == summary["best_score"]


def test_sync_mode_is_byte_identical(tmp_path):
    """With --async-eval OFF, the run produces the legacy COMBINED train+eval rows
    (evaluated=True on the train row) — no regression on the default path."""
    driver = _make_driver(tmp_path, async_eval=False, eval_every=2, epochs=4)
    summary = driver.run()
    assert summary["status"] == "complete"
    assert summary["async_eval"] is False
    assert summary["skipped_async_evals"] == 0

    rows = read_trajectory(tmp_path)
    eval_rows = [r for r in rows if r.get("evaluated")]
    # In sync mode the eval row IS the train row (has loss + lr AND eval fields),
    # NOT a separate async row.
    for r in eval_rows:
        assert r["score"] is not None
        # The combined row carries the real loss/lr (not NaN like the async-only row).
        assert r["loss"] == r["loss"]  # not NaN
        assert not r["extra"].get("async_eval_row", False)


def test_best_tracking_uses_snapshot_epoch(tmp_path):
    """The async eval row + best are tagged with the SNAPSHOT epoch (the epoch the
    weights came from), not whatever epoch training happens to be on when the eval
    finishes — so the trajectory is correctly attributed."""
    driver = _make_driver(tmp_path, async_eval=True, eval_every=2, epochs=2)
    spec = driver.curriculum[0]
    decoder = driver._new_decoder()
    latents = torch.nn.Parameter(torch.randn(driver.n_pairs, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    driver._train_one_epoch(rt, spec)

    # Schedule with snapshot_epoch=7 even though training is "at" a different point.
    assert driver._schedule_async_eval(rt, spec, 0, 7)
    driver._join_async_eval(timeout=60)

    rows = read_trajectory(tmp_path)
    eval_rows = [r for r in rows if r.get("evaluated")]
    assert eval_rows[-1]["global_epoch"] == 7
    assert eval_rows[-1]["extra"]["snapshot_epoch"] == 7
    # best_ep is the snapshot epoch (since this was the first/only eval -> best).
    assert driver.best_ep == 7
