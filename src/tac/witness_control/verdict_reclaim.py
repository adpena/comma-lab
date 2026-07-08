# SPDX-License-Identifier: MIT
"""Verdict working-set RECLAIM — cheap in-process trim + a killpg-reclaimed subprocess path.

#330 (operator: "the periodic CPU-torch verdict's ~5-6 GB transient working set ratchets parent
RSS instead of returning to the OS"). The per-epoch verdict runs SegNet/PoseNet over the rendered
frames; even chunked (#205 ``--verdict-batch``) each forward casts a uint8 pair stack to fp32 and
allocates EfficientNet-B2 / FastViT-T12 activations — a multi-GiB transient. The allocator
(glibc / macOS libmalloc) keeps those freed pages resident as a high-water mark, so the parent RSS
never falls back to baseline. On a long run that ratchet erodes the safe-run headroom.

Two remedies, honest simplest-fix-wins (MEASURE which one the numbers support):

1. **Cheap in-process reclaim** (``reclaim_process_memory``): ``gc.collect()`` + a platform allocator
   trim (Linux ``malloc_trim(0)``, macOS ``malloc_zone_pressure_relief``) + torch CUDA cache release
   (no-op on CPU). Zero new process, zero serialization. If this alone returns the pages the
   subprocess is unnecessary complexity.

2. **Subprocess verdict** (``run_verdict_in_subprocess``): run the chunked CPU-torch scorer forward in
   a CHILD process (its OWN session/process-group via ``start_new_session=True``, the #167 durable-
   daemon convention). The child holds the entire fp32/activation spike; on child EXIT the OS reclaims
   ALL of it, so the parent RSS returns to baseline by construction. Inputs (already-rendered uint8
   frames + GT argmax/pose targets) cross via a tmpfile; the child re-loads the frozen scorers itself
   (seconds) so nothing un-picklable is serialized. Crash-safety (the #167 non-negotiable): the parent
   registers the child pgid and ``killpg``s it on EVERY exit path; the child also self-terminates when
   the parent dies (poll ``os.getppid()``) so a SIGKILLed parent leaves NO orphan.

AUTHORITY: reclaim is score-neutral by construction — it frees pages, it never touches weights / RNG /
the verdict VALUES. The subprocess path is proven BIT-IDENTICAL to the in-process chunked verdict
(same frozen scorers, same preprocess -> forward -> argmax/MSE) by ``run_verdict_in_subprocess`` calling
the SAME ``cpu_verdict_*`` primitives inside the child. NON-PROMOTABLE: this module never produces a
score; the pointer moves only through a byte-closed ``upstream/evaluate.py`` n600 exact row.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_GIB = 1024.0 ** 3

__all__ = [
    "malloc_trim",
    "reclaim_process_memory",
    "rss_gib",
    "run_verdict_in_subprocess",
]


def rss_gib() -> float:
    """Resident-set size of THIS process in GiB (psutil, then ``resource`` fallback).

    Observability-only; returns -1.0 when unavailable (NO-FAKE: never a fabricated number)."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / _GIB
    except Exception:
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS ru_maxrss is BYTES; Linux is KiB. Heuristic: >1e9 => already bytes.
            return float(ru) / _GIB if ru > 1e9 else float(ru) / (1024.0 ** 2)
        except Exception:
            return -1.0


def malloc_trim() -> str:
    """Ask the C allocator to return freed pages to the OS. Returns the method used.

    - Linux (glibc): ``malloc_trim(0)`` — releases free heap top back to the kernel.
    - macOS (libmalloc): ``malloc_zone_pressure_relief(NULL, 0)`` — asks every zone to give back
      unused pages (the closest analogue; glibc ``malloc_trim`` does not exist on Darwin).
    - Otherwise / on any failure: ``"noop"`` — reclaim degrades to ``gc.collect`` only (never raises).

    Score-neutral: only frees pages, never touches Python objects."""
    sysname = platform.system()
    try:
        if sysname == "Linux":
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=False)
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(ctypes.c_size_t(0))
                return "malloc_trim"
            return "noop"
        if sysname == "Darwin":
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libSystem.dylib")
            if hasattr(libc, "malloc_zone_pressure_relief"):
                # malloc_zone_pressure_relief(malloc_zone_t *zone, size_t goal): NULL zone = all zones,
                # 0 goal = relieve as much as possible. Returns bytes released (ignored).
                libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
                libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                libc.malloc_zone_pressure_relief(None, 0)
                return "malloc_zone_pressure_relief"
            return "noop"
    except Exception:
        return "noop"
    return "noop"


def _release_torch_cache() -> bool:
    """Release torch's CUDA caching-allocator blocks (no-op / False on a CPU-only build). Never raises."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        return False
    return False


def reclaim_process_memory(*, measure: bool = True) -> dict[str, Any]:
    """CHEAP in-process reclaim after a verdict: ``gc.collect()`` + allocator trim + torch cache release.

    Returns a fail-open dict ``{gc_collected, trim_method, torch_cache_released, rss_before_gib,
    rss_after_gib, reclaimed_gib}``. ``measure=False`` skips the RSS reads (cheaper, no numbers).
    Score-neutral BY CONSTRUCTION: frees pages only; the caller's weights / RNG / verdict values are
    untouched -> a trainer that calls this after each verdict stays BIT-IDENTICAL."""
    before = rss_gib() if measure else -1.0
    collected = gc.collect()
    torch_released = _release_torch_cache()
    method = malloc_trim()
    after = rss_gib() if measure else -1.0
    return {
        "gc_collected": int(collected),
        "trim_method": method,
        "torch_cache_released": bool(torch_released),
        "rss_before_gib": round(before, 3),
        "rss_after_gib": round(after, 3),
        "reclaimed_gib": (round(before - after, 3) if (measure and before >= 0 and after >= 0) else None),
    }


# ---------------------------------------------------------------------------
# Subprocess verdict path (killpg-reclaimed child; #167 convention).
# ---------------------------------------------------------------------------
_WORKER_ENTRY = "tac.witness_control._verdict_subprocess_worker"


def _kill_group(pgid: int, *, grace_s: float = 3.0) -> None:
    """SIGTERM then SIGKILL the child's WHOLE process group (no orphan). Idempotent; never raises."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return  # group already gone
        if sig is signal.SIGTERM:
            t0 = time.time()
            while time.time() - t0 < grace_s:
                try:
                    os.killpg(pgid, 0)  # probe: raises when the group is gone
                except OSError:
                    return
                time.sleep(0.05)


def run_verdict_in_subprocess(
    frames0_uint8: list,
    frames1_uint8: list,
    gt_argmax_list: list,
    gt_pose_list: list,
    *,
    vbatch: int,
    seg_form: str = "argmax",
    timeout_s: float = 1800.0,
    scratch_dir: str | Path | None = None,
    python_exe: str | None = None,
) -> dict[str, Any]:
    """Run the chunked CPU-torch verdict in a killpg-reclaimed CHILD process; return the verdict rows.

    The child holds the entire fp32/activation transient; on child exit the OS reclaims it, so the
    PARENT RSS returns to baseline (the #330 ratchet fix). BIT-IDENTICAL to the in-process chunked
    verdict: the child runs the SAME ``cpu_verdict_d_seg_batch`` (or the argmax variant) +
    ``cpu_verdict_d_pose_batch`` with the SAME ``vbatch`` on the SAME frozen scorers.

    Returns ``{d_seg_mean, d_pose_mean, d_seg (list), d_pose (list), n, elapsed_s, child_pid,
    child_rss_peak_gib}``. Raises ``RuntimeError`` on child failure / timeout (the caller falls back to
    the in-process path; NO-FAKE — never a fabricated verdict).

    Crash-safety (#167): the child is its own session leader (``start_new_session=True``); the parent
    ``killpg``s the group on EVERY exit path (finally + timeout); the child self-terminates when the
    parent dies. Scratch tmpfiles are always removed."""
    n = len(frames1_uint8)
    if n == 0 or len(frames0_uint8) != n or len(gt_argmax_list) != n or len(gt_pose_list) != n:
        raise ValueError(
            f"ragged verdict inputs: f0={len(frames0_uint8)} f1={n} "
            f"lstars={len(gt_argmax_list)} poses={len(gt_pose_list)}")

    import numpy as np

    sd = Path(scratch_dir) if scratch_dir is not None else None
    if sd is not None:
        sd.mkdir(parents=True, exist_ok=True)
    in_fd, in_path = tempfile.mkstemp(prefix="verdict_in_", suffix=".npz", dir=str(sd) if sd else None)
    os.close(in_fd)
    out_path = in_path.replace("verdict_in_", "verdict_out_").replace(".npz", ".json")

    payload: dict[str, np.ndarray] = {
        "vbatch": np.asarray([int(vbatch)], np.int64),
        "n": np.asarray([int(n)], np.int64),
    }
    for i in range(n):
        payload[f"f0_{i}"] = np.ascontiguousarray(frames0_uint8[i], np.uint8)
        payload[f"f1_{i}"] = np.ascontiguousarray(frames1_uint8[i], np.uint8)
        payload[f"lstar_{i}"] = np.ascontiguousarray(gt_argmax_list[i], np.int64)
        payload[f"pose_{i}"] = np.ascontiguousarray(gt_pose_list[i], np.float64)

    proc: subprocess.Popen | None = None
    t0 = time.time()
    try:
        np.savez(in_path, **payload)
        del payload
        gc.collect()

        env = dict(os.environ)
        # An EMPTY OMP_NUM_THREADS makes torch warn ("Invalid OMP_NUM_THREADS ... stoi"); drop it so the
        # child inherits torch's own default rather than an invalid empty string.
        if not env.get("OMP_NUM_THREADS"):
            env.pop("OMP_NUM_THREADS", None)
        env["VERDICT_SUBPROC_PPID"] = str(os.getpid())  # child watches this for parent-death exit
        cmd = [
            python_exe or sys.executable, "-m", _WORKER_ENTRY,
            "--in", in_path, "--out", out_path, "--seg-form", str(seg_form),
        ]
        proc = subprocess.Popen(
            cmd, start_new_session=True, env=env,
            stdin=subprocess.DEVNULL,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"verdict subprocess timed out after {timeout_s}s") from exc
        if rc != 0:
            raise RuntimeError(f"verdict subprocess exited rc={rc}")
        if not os.path.exists(out_path):
            raise RuntimeError("verdict subprocess produced no result file")
        with open(out_path, encoding="utf-8") as fh:
            result = json.load(fh)
        if "error" in result:
            raise RuntimeError(f"verdict subprocess error: {result['error']}")
        result["child_pid"] = int(proc.pid)
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result
    finally:
        if proc is not None and proc.poll() is None:
            with _suppress():
                _kill_group(os.getpgid(proc.pid))
        for p in (in_path, out_path):
            with _suppress():
                os.remove(p)


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib for one call site."""

    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, *exc: object) -> bool:
        return True
