#!/usr/bin/env python
"""Durable watcher that preserves a training run's checkpoint at EVERY stage boundary.

THE PROBLEM. ``tac.torch_vehicle.driver`` writes only a ROLLING ``torch_vehicle_checkpoint_state.pt``
(overwritten every ``--checkpoint-every-epochs``) plus a best-SCORE ``best/`` dir (also overwritten as the
score improves). Neither preserves the END-OF-STAGE state at each curriculum boundary — the ideal fork points
for ALTERNATIVE downstream experiments (different optimizer / longer polish / loss swap) WITHOUT re-burning the
epochs behind them, and the safety net if a long stage (e.g. stage 5 C1a-L7, 9000 ep) crashes near its end.
Once the next stage starts, the rolling checkpoint is clobbered within ~45s.

THE MECHANISM (per-stage). Poll the run's checkpoint manifest. Keep a ROLLING working copy of the latest
checkpoint (``<out>/.rolling_snapshot/``) refreshed whenever the epoch advances. The instant the manifest's
``stage_name`` CHANGES, the rolling copy still holds the PRIOR stage's last checkpoint — preserve it into
``<out>/stage_snapshots/<prior_stage>_end/`` (full .pt + manifest + the ``best_ema_decoder.pt`` /
``best_ema_latents.pt`` warm-start pair, format per driver.py:3030-3031) + a ``STAGE_END.marker``. The
stage7->8 boundary (``has_muon`` flips True) ALSO writes the named ``<out>/pre_stage8_snapshot/`` alias.
The run's FINAL state is captured into ``<out>/final_snapshot/`` when the run stops (manifest goes stale).

This SUBSUMES the prior pre-stage-8-only watcher: pre_stage8 == stage7_sigma_sweep_end.

CORRECTNESS / HARDENING (adversarial rounds 1-3):
* The driver writes both files via ``os.replace`` (checkpoint.py:179-186) → a copy NEVER sees a partial .pt
  (always complete old-or-new; verified by POSIX open-fd semantics). ``_mtime_stable`` is belt-and-suspenders.
* EXIT GATES ON MANIFEST STALENESS (observe-relative), NOT bare PID death — a run RESTART changes the PID but
  the manifest keeps advancing within minutes, so the watcher is restart-robust. ``--watch-pid`` is advisory.
* Per-stage preserve reads the SNAPSHOT's own manifest and records ``has_muon`` in the marker so a rare
  blob-then-manifest skew is detectable, not silent.
* End-to-end load round-trip verified on a real checkpoint (warm-start pair strict-loads + forwards; full .pt
  exact-resumes). Pure decision/tracking helpers are unit-tested (test_snapshot_stage_boundary.py).

HOW TO LAUNCH FROM A SNAPSHOT:
* PRIMARY — continue from a boundary with a modified downstream: drop a ``<stage>_end/`` (or pre_stage8)
  ``torch_vehicle_checkpoint_state.pt`` + manifest into a fresh ``--out-dir`` → the driver EXACT-RESUMES there.
* SECONDARY — re-prime: ``--warm-start-dir <snapshot>`` with the SAME ``--taper-channels`` strict-loads the EMA.
* NOT ``--kd-warm-start-dir`` for a tapered DECODER (kd builds a vendored teacher → shape mismatch); the
  latents alone are kd-compatible (taper-independent).

NO score claim; this only COPIES bytes.

Usage (durable detached daemon):
    nohup .venv/bin/python tools/snapshot_run_checkpoint_at_stage_boundary.py \
        --out-dir experiments/results/<run> --watch-pid <python_pid> < /dev/null > <log> 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure decision helpers (unit-tested in tests/test_snapshot_stage_boundary.py)
# ---------------------------------------------------------------------------


def stage_changed(prev_stage: str | None, cur_stage: str | None) -> bool:
    """True iff the curriculum stage_name transitioned (and we knew a prior stage). PURE."""
    return prev_stage is not None and cur_stage is not None and cur_stage != prev_stage


def should_roll_copy(manifest: dict, last_copied: tuple[int, int] | None, start_stage_index: int) -> bool:
    """True iff the rolling working-copy should refresh: in/after the start stage AND the epoch advanced. PURE."""
    stage_index = int(manifest.get("stage_index", -1))
    if stage_index < start_stage_index:
        return False
    key = (stage_index, int(manifest.get("epoch_in_stage", -1)))
    return key != last_copied


def next_change_reference(
    prev_mtime: float | None, cur_mtime: float | None, prev_reference: float, now: float
) -> tuple[float | None, float]:
    """Advance the observe-relative staleness reference. PURE.

    If the manifest mtime CHANGED since last poll, reset the reference clock to ``now`` (the run is alive).
    Otherwise keep the prior reference. Immune to a stale-at-startup mtime: only the *observed* lack of
    advance for ``threshold_s`` triggers exit. Returns ``(new_last_mtime, new_reference)``.
    """
    if cur_mtime is not None and cur_mtime != prev_mtime:
        return cur_mtime, now
    return prev_mtime, prev_reference


def is_stale(reference_time: float | None, now: float, threshold_s: float) -> bool:
    """True if ``now`` is more than ``threshold_s`` past ``reference_time`` (the last OBSERVED manifest
    advance — not the absolute mtime, so a stale-at-startup mtime / slow resume never false-triggers). PURE.
    ``None`` reference → never stale.
    """
    if reference_time is None:
        return False
    return (now - reference_time) > threshold_s


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _log(msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _read_manifest(path: Path) -> dict | None:
    """Defensive manifest read — returns None on transient mid-write / missing."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` -> ``dst`` atomically (tmp in dst's dir + os.replace)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _mtime_stable(path: Path, min_age_s: float = 2.0) -> bool:
    """True if ``path`` was last modified > ``min_age_s`` ago (not mid-write)."""
    try:
        return (time.time() - path.stat().st_mtime) > min_age_s
    except OSError:
        return False


def _manifest_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _refresh_snapshot(ckpt: Path, manifest: Path, snap_dir: Path) -> None:
    """Copy the full checkpoint_state.pt + manifest into snap_dir (exact-resume bundle)."""
    _atomic_copy(ckpt, snap_dir / "torch_vehicle_checkpoint_state.pt")
    _atomic_copy(manifest, snap_dir / "torch_vehicle_checkpoint_manifest.json")


def _emit_warm_start_pair(snap_dir: Path, log_path: Path) -> bool:
    """Extract ema_decoder/ema_latents from the snapshotted .pt into the warm-start pair (format per
    driver.py:3030-3031, loadable by --warm-start-dir). One torch.load. Graceful on missing/bad .pt."""
    import torch  # local import — keep the rolling/file-copy path torch-free

    pt = snap_dir / "torch_vehicle_checkpoint_state.pt"
    try:
        st = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as exc:
        _log(f"WARN could not torch.load {pt} for warm-start extraction: {exc!r}", log_path)
        return False
    if "ema_decoder" not in st or "ema_latents" not in st:
        _log(f"WARN snapshot .pt missing ema_decoder/ema_latents (keys={list(st)[:8]})", log_path)
        return False
    torch.save(st["ema_decoder"], snap_dir / "best_ema_decoder.pt")
    torch.save(st["ema_latents"], snap_dir / "best_ema_latents.pt")
    return True


def _preserve(rolling_dir: Path, dest_dir: Path, *, reason: str, log_path: Path) -> bool:
    """Preserve the current rolling copy into ``dest_dir`` (full .pt + manifest + warm-start pair + marker).

    Returns True if a rolling .pt existed to preserve. Records the snapshot manifest's ``has_muon`` in the
    marker (skew detection). The rolling copy is itself produced by ``os.replace`` so it is never partial.
    """
    src_pt = rolling_dir / "torch_vehicle_checkpoint_state.pt"
    src_man = rolling_dir / "torch_vehicle_checkpoint_manifest.json"
    if not src_pt.exists():
        _log(f"WARN _preserve({reason}): no rolling .pt to preserve into {dest_dir}", log_path)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy(src_pt, dest_dir / "torch_vehicle_checkpoint_state.pt")
    if src_man.exists():
        _atomic_copy(src_man, dest_dir / "torch_vehicle_checkpoint_manifest.json")
    ok = _emit_warm_start_pair(dest_dir, log_path)
    snap_man = _read_manifest(dest_dir / "torch_vehicle_checkpoint_manifest.json")
    has_muon = bool(snap_man.get("has_muon", False)) if snap_man else None
    (dest_dir / "STAGE_END.marker").write_text(
        json.dumps(
            {
                "preserved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": reason,
                "warm_start_pair_emitted": ok,
                "snapshot_manifest_stage": (snap_man or {}).get("stage_name"),
                "snapshot_manifest_epoch_in_stage": (snap_man or {}).get("epoch_in_stage"),
                "snapshot_manifest_has_muon": has_muon,
            },
            indent=2,
        )
    )
    _log(f"PRESERVED [{reason}] -> {dest_dir} (warm_start_pair={ok}, has_muon={has_muon})", log_path)
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="the run dir being watched")
    ap.add_argument(
        "--start-stage-index",
        type=int,
        default=0,
        help="begin rolling-copying once stage_index reaches this (0-based; default 0 = every stage, so "
        "every boundary from the current stage onward is preserved)",
    )
    ap.add_argument(
        "--watch-pid",
        type=int,
        default=0,
        help="training PID — ADVISORY ONLY (logged when it dies). Exit gates on manifest staleness so a run "
        "RESTART does not orphan the watcher.",
    )
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument(
        "--stale-exit-seconds",
        type=float,
        default=600.0,
        help="exit (and do the final copy) when the manifest has not advanced for this long (run stopped). "
        "Must exceed a normal restart gap. Default 600s.",
    )
    ap.add_argument(
        "--also-final-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the FINAL checkpoint into <out-dir>/final_snapshot/ when the run stops",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    rolling_dir = out_dir / ".rolling_snapshot"
    stage_root = out_dir / "stage_snapshots"
    pre_stage8_dir = out_dir / "pre_stage8_snapshot"
    final_dir = out_dir / "final_snapshot"
    manifest_path = out_dir / "torch_vehicle_checkpoint_manifest.json"
    ckpt_path = out_dir / "torch_vehicle_checkpoint_state.pt"
    log_path = stage_root / "watcher.log"
    stage_root.mkdir(parents=True, exist_ok=True)

    _log(
        f"watcher START (per-stage) out_dir={out_dir} start_stage_index={args.start_stage_index} "
        f"watch_pid={args.watch_pid} poll={args.poll_seconds}s stale_exit={args.stale_exit_seconds}s",
        log_path,
    )

    # Seed last_stage from the persisted rolling copy (if any) so a watcher RESTART across a stage
    # transition still preserves the prior stage's end from the rolling dir (R4-11).
    last_stage: str | None = (
        _read_manifest(rolling_dir / "torch_vehicle_checkpoint_manifest.json") or {}
    ).get("stage_name")
    last_copied: tuple[int, int] | None = None
    pid_death_logged = False
    last_manifest_mtime: float | None = _manifest_mtime(manifest_path)
    last_change_time = time.time()

    while True:
        try:
            man = _read_manifest(manifest_path)
            if man is not None:
                cur_stage = man.get("stage_name")  # None on a partial/fieldless read → no spurious transition (R4-6)
                # 1) STAGE TRANSITION → preserve the ENDING stage from the rolling copy.
                if stage_changed(last_stage, cur_stage):
                    _preserve(rolling_dir, stage_root / f"{last_stage}_end", reason=f"{last_stage}->{cur_stage}", log_path=log_path)
                    if bool(man.get("has_muon", False)):  # the stage7->8 boundary
                        _preserve(rolling_dir, pre_stage8_dir, reason="pre_stage8 (has_muon)", log_path=log_path)
                if cur_stage is not None:
                    last_stage = cur_stage
                # 2) ROLL-COPY the current checkpoint (latest of the current stage).
                if should_roll_copy(man, last_copied, args.start_stage_index) and ckpt_path.exists() and _mtime_stable(ckpt_path):
                    _refresh_snapshot(ckpt_path, manifest_path, rolling_dir)
                    last_copied = (int(man.get("stage_index", -1)), int(man.get("epoch_in_stage", -1)))
                    _log(f"rolling-copied {cur_stage or '?'} stage_index={last_copied[0]} ep_in_stage={last_copied[1]}", log_path)

            # ADVISORY pid log (does NOT cause exit — restart-robust).
            if args.watch_pid > 0 and not pid_death_logged and not _pid_alive(args.watch_pid):
                _log(f"watched pid {args.watch_pid} no longer alive (advisory; exit gates on manifest staleness)", log_path)
                pid_death_logged = True

            # RESTART-ROBUST EXIT: the run genuinely stopped (manifest not ADVANCING).
            mtime = _manifest_mtime(manifest_path)
            last_manifest_mtime, last_change_time = next_change_reference(
                last_manifest_mtime, mtime, last_change_time, time.time()
            )
            if is_stale(last_change_time, time.time(), args.stale_exit_seconds):
                _log(f"manifest stale > {args.stale_exit_seconds}s → run stopped (last_stage={last_stage})", log_path)
                if args.also_final_on_exit and ckpt_path.exists():
                    # Snapshot the genuine FINAL state straight from the live file (not the rolling copy).
                    final_dir.mkdir(parents=True, exist_ok=True)
                    _refresh_snapshot(ckpt_path, manifest_path, final_dir)
                    ok = _emit_warm_start_pair(final_dir, log_path)
                    _log(f"FINAL snapshot -> {final_dir} (warm_start_pair={ok})", log_path)
                _log("watcher EXIT", log_path)
                return 0
        except Exception as exc:
            _log(f"WARN poll error (continuing): {exc!r}", log_path)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
