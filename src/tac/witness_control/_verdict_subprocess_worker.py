# SPDX-License-Identifier: MIT
"""Child-process entrypoint for the #330 killpg-reclaimed verdict path.

Runs the CHUNKED CPU-torch scorer forward (SegNet argmax d_seg + PoseNet MSE d_pose) over the
already-rendered frames handed in via an npz, then writes the verdict rows to a result json and
EXITS — so the OS reclaims the entire fp32/activation transient and the PARENT RSS returns to
baseline. BIT-IDENTICAL to the in-process chunked verdict: it re-loads the SAME frozen scorers and
calls the SAME ``cpu_verdict_*`` primitives with the SAME ``vbatch``.

Crash-safety (#167): this process is its own session leader (parent used ``start_new_session=True``);
a background watcher polls the parent pid (``VERDICT_SUBPROC_PPID``) and hard-exits when the parent
dies, so a SIGKILLed parent leaves NO orphan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from tac import process_liveness

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "src", _REPO / "upstream", _REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _start_parent_death_watch() -> None:
    """Hard-exit this child when the launching parent dies (portable macOS/Linux parent-death guard).

    ``VERDICT_SUBPROC_PPID`` is the parent's pid at launch. Because the child is its own session
    leader, POSIX reparents it to init on parent death rather than signalling it, so we POLL: when
    ``os.getppid()`` no longer equals the recorded launcher pid (reparented to 1), or the recorded pid
    is gone, we ``os._exit`` -> NO orphaned child holding multi-GiB scorer activations."""
    want = os.environ.get("VERDICT_SUBPROC_PPID")
    if not want:
        return
    try:
        want_pid = int(want)
    except ValueError:
        return

    def _watch() -> None:
        while True:
            if os.getppid() != want_pid:
                os._exit(3)  # reparented -> launcher gone
            # Canonical liveness, but zombie_is_dead=False DELIBERATELY: this is
            # a 4 Hz loop inside a memory-constrained child, and the zombie leg
            # forks `ps`.  Nothing is lost -- a zombied parent means we were
            # already reparented, which the getppid check above catches first.
            # CHANGED: PermissionError now reads ALIVE instead of exiting; the
            # old bare `except OSError` swallowed EPERM and could kill a LIVE
            # verdict child on a spurious permission error.
            if (
                process_liveness.pid_state(want_pid, zombie_is_dead=False)
                != process_liveness.ALIVE
            ):
                os._exit(3)
            time.sleep(0.25)

    threading.Thread(target=_watch, name="parent-death-watch", daemon=True).start()


def _rss_peak_gib() -> float:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((ru / (1024.0 ** 3)) if ru > 1e9 else (ru / (1024.0 ** 2)), 3)
    except Exception:
        return -1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="killpg-reclaimed verdict child (#330)")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--seg-form", default="argmax")
    args = ap.parse_args(argv)

    _start_parent_death_watch()

    out_path = args.out_path
    try:
        # SAME primitives + SAME vbatch as the in-process chunked verdict => BIT-IDENTICAL values.
        from train_witness_realized_through_R_mlx import (
            cpu_verdict_d_pose_batch,
            cpu_verdict_d_seg_batch,
        )

        from tac.boundary_math.seg_core import load_real_segnet

        z = np.load(args.in_path, allow_pickle=False)
        n = int(z["n"][0])
        vbatch = int(z["vbatch"][0])
        f0s = [np.asarray(z[f"f0_{i}"], np.uint8) for i in range(n)]
        f1s = [np.asarray(z[f"f1_{i}"], np.uint8) for i in range(n)]
        lstars = [np.asarray(z[f"lstar_{i}"], np.int64) for i in range(n)]
        poses = [np.asarray(z[f"pose_{i}"], np.float64) for i in range(n)]
        del z

        seg_cpu = load_real_segnet("cpu")
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        dn = DistortionNet().eval()
        dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        posenet_cpu = dn.posenet
        for p in posenet_cpu.parameters():
            p.requires_grad = False

        vb = n if (vbatch <= 0 or vbatch >= n) else vbatch
        ds_all: list[float] = []
        dp_all: list[float] = []
        for s in range(0, n, vb):
            e = min(s + vb, n)
            ds_all.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
            dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))

        result = {
            "d_seg_mean": float(np.mean(ds_all)),
            "d_pose_mean": float(np.mean(dp_all)),
            "d_seg": [float(x) for x in ds_all],
            "d_pose": [float(x) for x in dp_all],
            "n": int(n),
            "vbatch": int(vbatch),
            "child_rss_peak_gib": _rss_peak_gib(),
        }
    except Exception as exc:  # NO-FAKE: surface the failure to the parent, never a fabricated verdict.
        result = {"error": f"{type(exc).__name__}: {exc}"}

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    os.replace(tmp, out_path)  # atomic: parent never reads a partial file
    return 0 if "error" not in result else 4


if __name__ == "__main__":
    raise SystemExit(main())
