#!/usr/bin/env python
"""Durable watcher that preserves a training run's checkpoint at a stage boundary.

THE PROBLEM. ``tac.torch_vehicle.driver`` writes only a ROLLING ``torch_vehicle_checkpoint_state.pt``
(overwritten every ``--checkpoint-every-epochs``) plus a best-SCORE ``best/`` dir (also overwritten as the
score improves). Neither preserves the END-OF-STAGE-7 state — the pre-stage-8 (pre-Muon) basin — which is the
ideal launching point for ALTERNATIVE stage-8 finishers (different optimizer / longer polish / CE-vs-margin-
hinge) WITHOUT re-burning the ~24,650 epochs of stages 1-7. Once stage 8 starts, the rolling checkpoint is
clobbered within ~45s.

THE MECHANISM. Poll the run's checkpoint manifest. While the ending stage is active (``stage_index >=
--start-stage-index`` AND ``has_muon`` is False), keep a ROLLING copy of the latest checkpoint into
``--snapshot-dir`` (pure file-copy — CPU-light next to a live MPS run; only copies when the epoch advances).
The instant ``has_muon`` flips True (stage 8's first checkpoint), FREEZE: stop refreshing, extract the EMA
decoder + latents into the ``best_ema_decoder.pt`` / ``best_ema_latents.pt`` warm-start pair (format per
driver.py:3030-3031), and write a ``FROZEN.marker``. The frozen copy is the true end-of-stage-7 basin.

HOW TO LAUNCH FROM A SNAPSHOT (verified end-to-end on a real checkpoint, 2026-06-22):
* PRIMARY — alternative stage-8 finisher: drop ``torch_vehicle_checkpoint_state.pt`` + manifest into a fresh
  ``--out-dir`` and relaunch; the driver EXACT-RESUMES at end-of-stage-7 and runs stage 8 (a modified stage-8
  config / muon_lr_floor_fix toggle is resume-safe because has_muon is still False at the frozen state).
* SECONDARY — re-prime from the basin: ``--warm-start-dir <snapshot>`` with the SAME ``--taper-channels`` /
  ``--base-channels`` / ``--latent-dim`` strict-loads the EMA decoder (verified missing=[]/unexpected=[]).
* NOT ``--kd-warm-start-dir`` for the DECODER: kd builds a VENDORED (no-taper) teacher and strict-loads it, so
  a tapered EMA shape-mismatches. The LATENTS alone are kd-compatible (taper-independent), but the tapered
  decoder is not a valid kd teacher.

ADVERSARIAL-REVIEW HARDENING (2026-06-22):
* The driver writes both files via ``os.replace`` (checkpoint.py:179-186) → a copy NEVER sees a partial .pt
  (always complete old-or-new). The ``_mtime_stable`` guard is belt-and-suspenders.
* EXIT GATES ON MANIFEST STALENESS, NOT BARE PID DEATH. A run RESTART (e.g. to apply a fix) changes the PID but
  the manifest keeps advancing within ~minutes; a bare-pid-death exit would orphan the watcher and silently
  miss the stage-8 freeze. So the watcher exits (and does the final copy) only when the manifest has not
  advanced for ``--stale-exit-seconds`` (the run genuinely stopped). ``--watch-pid`` is ADVISORY (logged).
* FREEZE SKEW DETECTION. Because the driver replaces blob-then-manifest, a snapshot could (probability ~1e-4)
  pair a just-started-stage-8 blob with a stage-7 manifest. At freeze the watcher reads the SNAPSHOT's copied
  manifest and WARNS loudly if its ``has_muon`` is True (the rare skew) so the condition is detectable, not silent.

Optionally (``--also-final-on-exit``) copy the FINAL checkpoint into ``<out-dir>/final_snapshot/`` when the run
stops, so the post-stage-8 state is also preserved for additional polish.

NO score claim; this only COPIES bytes. Authority is unchanged.

Usage (durable detached daemon — survives the launching shell):
    nohup .venv/bin/python tools/snapshot_run_checkpoint_at_stage_boundary.py \
        --out-dir experiments/results/<run> --watch-pid <python_pid> \
        < /dev/null > <log> 2>&1 & disown
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

FREEZE = "freeze"
COPY = "copy"
NOOP = "noop"


def decide_action(manifest: dict, last_copied: tuple[int, int] | None, start_stage_index: int) -> str:
    """Decide the per-poll action from a checkpoint manifest. PURE (no IO).

    * ``FREEZE`` when ``has_muon`` is True (stage 8 started) — the prior rolling copy is the end-of-stage-7 state.
    * ``COPY`` when the ending stage is active (``stage_index >= start_stage_index``, ``has_muon`` False) AND the
      (stage_index, epoch_in_stage) advanced since the last copy.
    * ``NOOP`` otherwise.
    """
    if bool(manifest.get("has_muon", False)):
        return FREEZE
    stage_index = int(manifest.get("stage_index", -1))
    epoch_in_stage = int(manifest.get("epoch_in_stage", -1))
    if stage_index >= start_stage_index and (stage_index, epoch_in_stage) != last_copied:
        return COPY
    return NOOP


def next_change_reference(
    prev_mtime: float | None, cur_mtime: float | None, prev_reference: float, now: float
) -> tuple[float | None, float]:
    """Advance the observe-relative staleness reference. PURE.

    If the manifest mtime CHANGED since last poll, reset the reference clock to ``now`` (the run is alive
    and advancing). Otherwise keep the prior reference. This is what makes ``is_stale`` immune to a stale-
    at-startup mtime: the reference is seeded at watcher start and only the *observed* lack of advance for
    ``threshold_s`` triggers exit. Returns ``(new_last_mtime, new_reference)``.
    """
    if cur_mtime is not None and cur_mtime != prev_mtime:
        return cur_mtime, now
    return prev_mtime, prev_reference


def is_stale(reference_time: float | None, now: float, threshold_s: float) -> bool:
    """True if ``now`` is more than ``threshold_s`` past ``reference_time``. PURE.

    ``reference_time`` is the wall-clock time the watcher LAST OBSERVED the manifest advance — NOT the
    absolute manifest mtime. Measuring elapsed-since-observed-change (rather than absolute mtime age) is
    what makes the exit robust to (a) a watcher (re)launched when the manifest is already old, and (b) a
    slow cold resume whose first new checkpoint lags — neither false-triggers, because the reference is
    re-seeded at watcher start and on every observed advance. ``None`` reference → never stale.
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
    """Extract ema_decoder/ema_latents from the snapshotted .pt into the warm-start pair.

    Format matches driver.py:3030-3031 (torch.save(ema_state_dict) + torch.save(ema_latents_tensor)),
    so the result is directly loadable by --warm-start-dir / --kd-warm-start-dir. One torch.load total.
    """
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


def _freeze(snap_dir: Path, last_copied: tuple[int, int] | None, out_dir: Path, log_path: Path) -> None:
    """Freeze the rolling snapshot as the pre-stage-8 basin + emit the warm-start pair + marker."""
    ok = _emit_warm_start_pair(snap_dir, log_path)
    # F1b skew detection: the frozen snapshot's OWN manifest must be has_muon=False (a stage-7 state).
    snap_man = _read_manifest(snap_dir / "torch_vehicle_checkpoint_manifest.json")
    snap_has_muon = bool(snap_man.get("has_muon", False)) if snap_man else None
    if snap_has_muon:
        _log(
            "WARN FREEZE SKEW: the frozen snapshot's manifest has_muon=True — the last rolling copy caught a "
            "stage-8 blob (the ~1e-4 blob-then-manifest race). Snapshot is the FIRST stage-8 step, not strict "
            "end-of-stage-7 (still ~pre-finish basin). Detectable, not silent.",
            log_path,
        )
    (snap_dir / "FROZEN.marker").write_text(
        json.dumps(
            {
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": "has_muon flipped True (stage 8 started)",
                "last_copied_stage_epoch": last_copied,
                "warm_start_pair_emitted": ok,
                "snapshot_manifest_has_muon": snap_has_muon,
                "source_out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    _log(
        f"FROZEN pre-stage-8 snapshot (last_copied={last_copied}, warm_start_pair={ok}, "
        f"snap_has_muon={snap_has_muon}). snap_dir={snap_dir}",
        log_path,
    )


def _copy_final(out_dir: Path, ckpt_path: Path, manifest_path: Path, log_path: Path) -> None:
    final_dir = out_dir / "final_snapshot"
    final_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        _log("WARN no checkpoint_state.pt to copy for final snapshot", log_path)
        return
    _atomic_copy(ckpt_path, final_dir / "torch_vehicle_checkpoint_state.pt")
    if manifest_path.exists():
        _atomic_copy(manifest_path, final_dir / "torch_vehicle_checkpoint_manifest.json")
    ok = _emit_warm_start_pair(final_dir, log_path)
    _log(f"FINAL snapshot copied to {final_dir} (warm_start_pair={ok})", log_path)


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
        "--snapshot-dir",
        type=Path,
        default=None,
        help="where to freeze the pre-stage-8 snapshot (default <out-dir>/pre_stage8_snapshot)",
    )
    ap.add_argument(
        "--start-stage-index",
        type=int,
        default=5,
        help="begin rolling-copying once stage_index reaches this (0-based; default 5 = stage6, "
        "so stages 6+7 are covered before the stage-8 freeze)",
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
        help="exit (and do the final copy) when the manifest has not advanced for this long (the run genuinely "
        "stopped). Must exceed a normal restart gap so a restart does not trip it. Default 600s (10 min).",
    )
    ap.add_argument(
        "--also-final-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the FINAL checkpoint into <out-dir>/final_snapshot/ when the run stops",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    snap_dir: Path = args.snapshot_dir or (out_dir / "pre_stage8_snapshot")
    manifest_path = out_dir / "torch_vehicle_checkpoint_manifest.json"
    ckpt_path = out_dir / "torch_vehicle_checkpoint_state.pt"
    log_path = snap_dir / "watcher.log"
    snap_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"watcher START out_dir={out_dir} snap_dir={snap_dir} start_stage_index={args.start_stage_index} "
        f"watch_pid={args.watch_pid} poll={args.poll_seconds}s stale_exit={args.stale_exit_seconds}s",
        log_path,
    )

    frozen = False
    last_copied: tuple[int, int] | None = None  # (stage_index, epoch_in_stage)
    pid_death_logged = False
    # OBSERVE-RELATIVE staleness (R2-1): track when the watcher last SAW the manifest advance, seeded at
    # start, so a pre-existing old mtime / slow cold resume never false-triggers the exit.
    last_manifest_mtime: float | None = _manifest_mtime(manifest_path)
    last_change_time = time.time()

    while True:
        try:
            man = _read_manifest(manifest_path)
            if man is not None and not frozen:
                action = decide_action(man, last_copied, args.start_stage_index)
                if action == FREEZE:
                    _freeze(snap_dir, last_copied, out_dir, log_path)
                    frozen = True
                elif action == COPY and ckpt_path.exists() and _mtime_stable(ckpt_path):
                    _refresh_snapshot(ckpt_path, manifest_path, snap_dir)
                    last_copied = (int(man.get("stage_index", -1)), int(man.get("epoch_in_stage", -1)))
                    _log(
                        f"rolling-copied {man.get('stage_name', '?')} stage_index={last_copied[0]} "
                        f"ep_in_stage={last_copied[1]}",
                        log_path,
                    )

            # ADVISORY pid log (does NOT cause exit — restart-robust).
            if args.watch_pid > 0 and not pid_death_logged and not _pid_alive(args.watch_pid):
                _log(
                    f"watched pid {args.watch_pid} no longer alive (advisory; exit gates on manifest staleness "
                    f">{args.stale_exit_seconds}s — a restart will refresh the manifest)",
                    log_path,
                )
                pid_death_logged = True

            # RESTART-ROBUST EXIT: the run genuinely stopped (manifest not ADVANCING). Observe-relative:
            # update the reference only when the mtime actually changes, so a stale-at-startup mtime or a
            # slow resume gets a fresh threshold window instead of false-exiting (R2-1).
            mtime = _manifest_mtime(manifest_path)
            last_manifest_mtime, last_change_time = next_change_reference(
                last_manifest_mtime, mtime, last_change_time, time.time()
            )
            if is_stale(last_change_time, time.time(), args.stale_exit_seconds):
                _log(
                    f"manifest stale > {args.stale_exit_seconds}s → run stopped (frozen={frozen})",
                    log_path,
                )
                if not frozen:
                    _log(
                        f"WARN run stopped before stage 8 was detected; pre_stage8_snapshot holds the last "
                        f"rolling copy ({last_copied}) if any",
                        log_path,
                    )
                if args.also_final_on_exit:
                    _copy_final(out_dir, ckpt_path, manifest_path, log_path)
                _log("watcher EXIT", log_path)
                return 0
        except Exception as exc:
            _log(f"WARN poll error (continuing): {exc!r}", log_path)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
