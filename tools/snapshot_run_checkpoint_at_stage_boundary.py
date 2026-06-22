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
decoder + latents into the ``best_ema_decoder.pt`` / ``best_ema_latents.pt`` warm-start pair (the format
``--warm-start-dir`` / ``--kd-warm-start-dir`` consume — verified against driver.py:3030-3031), and write a
``FROZEN.marker``. The frozen copy is the true end-of-stage-7 basin.

Optionally (``--also-final-on-exit``) copy the FINAL checkpoint into ``<out-dir>/final_snapshot/`` when the
watched training PID exits, so the post-stage-8 state is also preserved for additional polish.

NO score claim; this only COPIES bytes. Authority is unchanged. Atomic copies (tmp + os.replace) so a reader
never sees a half-written file; mtime-stable guard so the source is never copied mid-write.

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
    ap.add_argument("--watch-pid", type=int, default=0, help="training PID; exit when it dies (0 = ignore)")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument(
        "--also-final-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the FINAL checkpoint into <out-dir>/final_snapshot/ when the watched PID exits",
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
        f"watch_pid={args.watch_pid} poll={args.poll_seconds}s",
        log_path,
    )

    frozen = False
    last_copied: tuple[int, int] | None = None  # (stage_index, epoch_in_stage)

    def pid_alive(pid: int) -> bool:
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    while True:
        try:
            man = _read_manifest(manifest_path)
            if man is not None and not frozen:
                stage_index = int(man.get("stage_index", -1))
                has_muon = bool(man.get("has_muon", False))
                epoch_in_stage = int(man.get("epoch_in_stage", -1))
                stage_name = str(man.get("stage_name", "?"))

                if has_muon:
                    # Stage 8 has started → the rolling copy already holds the last
                    # has_muon=False (end-of-stage-7) checkpoint. Freeze it.
                    ok = _emit_warm_start_pair(snap_dir, log_path)
                    (snap_dir / "FROZEN.marker").write_text(
                        json.dumps(
                            {
                                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "reason": "has_muon flipped True (stage 8 started)",
                                "last_copied_stage_epoch": last_copied,
                                "warm_start_pair_emitted": ok,
                                "source_out_dir": str(out_dir),
                            },
                            indent=2,
                        )
                    )
                    _log(
                        f"FROZEN pre-stage-8 snapshot (last_copied={last_copied}, warm_start_pair={ok}). "
                        f"snap_dir={snap_dir}",
                        log_path,
                    )
                    frozen = True
                elif stage_index >= args.start_stage_index:
                    key = (stage_index, epoch_in_stage)
                    if key != last_copied and ckpt_path.exists() and _mtime_stable(ckpt_path):
                        _refresh_snapshot(ckpt_path, manifest_path, snap_dir)
                        last_copied = key
                        _log(f"rolling-copied {stage_name} stage_index={stage_index} ep_in_stage={epoch_in_stage}", log_path)

            if not pid_alive(args.watch_pid):
                _log(f"watched pid {args.watch_pid} no longer alive", log_path)
                if not frozen:
                    _log(
                        "WARN run exited before stage 8 was detected; pre_stage8_snapshot holds the "
                        f"last rolling copy ({last_copied}) if any",
                        log_path,
                    )
                if args.also_final_on_exit:
                    final_dir = out_dir / "final_snapshot"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    if ckpt_path.exists():
                        # give the run a moment to finish its last write
                        if not _mtime_stable(ckpt_path, min_age_s=5.0):
                            time.sleep(10)
                        _atomic_copy(ckpt_path, final_dir / "torch_vehicle_checkpoint_state.pt")
                        if manifest_path.exists():
                            _atomic_copy(manifest_path, final_dir / "torch_vehicle_checkpoint_manifest.json")
                        ok = _emit_warm_start_pair(final_dir, log_path)
                        _log(f"FINAL snapshot copied to {final_dir} (warm_start_pair={ok})", log_path)
                    else:
                        _log("WARN no checkpoint_state.pt to copy for final snapshot", log_path)
                _log("watcher EXIT", log_path)
                return 0
        except Exception as exc:
            _log(f"WARN poll error (continuing): {exc!r}", log_path)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
