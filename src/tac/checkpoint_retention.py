"""Whole-run checkpoint RETENTION — keep the trajectory, never overwrite (operator 2026-06-30).

The default trainer pattern writes only a SINGLE best + latest and OVERWRITES them each improvement, so
the descending trajectory of bests (and the inter-best path) is LOST. Operator directive 2026-06-30:
*"keep the whole run ... that would actually unlock some really interesting science"* (per-epoch d_seg
attribution, the α≈4/d power-law tail validated on real data, warm-start any lever from any epoch, the
Morse-Smale trajectory view of the DAG↔DSL↔equations triality) + *"engineer so it's an easy drop-in
decorator but can also be run manually."*

This module is that core, with THREE entry points over ONE policy (``CheckpointArchiver``):
  1. **decorator** ``@keep_checkpoint_history(run_dir=..., archive_dir=...)`` — drop onto a trainer's save
     fn; after it runs, the new best/latest is archived automatically (zero coupling — works with any save
     fn that writes the standard files).
  2. **context manager** ``with checkpoint_history(run_dir) as arch: ... arch.keep(path, kind=..., epoch=...)``.
  3. **direct call** ``CheckpointArchiver(archive_dir).keep(npz, kind="best", epoch=ep, d_seg=ds)`` or
     ``.scan(run_dir)`` — and the CLI ``tools/archive_witness_checkpoints.py`` (external watcher, for a
     RUNNING arm whose in-process save logic can't be changed).

DISK HYGIENE (CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup ... certify or block"): keep-all locally
while small (the level-set witness ckpt is ~351 KB -> a whole 900-ep run is tens of MB). A moving-window
cap (``keep_window>0``) spills the OLDEST non-best ``latest`` snaps to the SSD tier (VertigoDataTier ->
APDataStore) ONLY after a manifest row records (src,dst,bytes,sha256,epoch,location) -- certify-or-block,
no signal loss. BESTS are never spilled-away unless ``spill_bests=True``.

Atomic (tmp+os.replace), idempotent/resumable (manifest dedups by (kind,epoch[,d_seg])), race-safe reads
(the trainer writes ckpts atomically). Refuses /tmp evidence paths.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_SSD_TIERS = ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _refuse_tmp(path: Path) -> None:
    s = str(Path(path).resolve())
    if any(t in s + "/" for t in _FORBIDDEN_TMP):
        raise ValueError(f"refusing transient /tmp path: {path} (CLAUDE.md durable-evidence rule)")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy(src: Path, dst: Path) -> None:
    _refuse_tmp(dst)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _read_epoch(npz: Path) -> int | None:
    try:
        import numpy as np
        z = np.load(npz, allow_pickle=False)
        return int(z["__epoch"]) if "__epoch" in z.files else None
    except Exception:
        return None


def _pick_spill_tier() -> Path | None:
    for t in _SSD_TIERS:
        base = Path(t)
        if base.parent.exists():  # the /Volumes/<vol> mount is present
            return base / "ckpt_archive_spill"
    return None


class CheckpointArchiver:
    """The retention policy. One per run (or archive dir). Idempotent + atomic + certify-or-block."""

    def __init__(self, archive_dir: str | Path, *, keep_window: int = 0, spill: bool = True,
                 spill_bests: bool = False, min_free_gb: float = 10.0) -> None:
        self.archive_dir = Path(archive_dir)
        self.keep_window = int(keep_window)      # 0 = keep ALL locally
        self.spill = bool(spill)
        self.spill_bests = bool(spill_bests)
        self.min_free_gb = float(min_free_gb)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        _refuse_tmp(self.archive_dir)
        self.manifest = self.archive_dir / "manifest.jsonl"

    # ---- manifest ---------------------------------------------------------------------------------
    def _rows(self) -> list[dict]:
        if not self.manifest.exists():
            return []
        return [json.loads(ln) for ln in self.manifest.read_text().splitlines() if ln.strip()]

    def _append(self, row: dict) -> None:
        with open(self.manifest, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    # ---- direct manual API ------------------------------------------------------------------------
    def keep(self, src_npz: str | Path, *, kind: str, epoch: int | None = None,
             d_seg: float | None = None) -> dict | None:
        """Archive ONE ckpt under an epoch[+d_seg]-encoded name; dedups; never overwrites. Returns the
        manifest row (or None if a duplicate / refused)."""
        src = Path(src_npz)
        if not src.exists():
            return None
        if shutil.disk_usage(self.archive_dir).free / 1e9 < self.min_free_gb:
            return {"REFUSE": "low_disk"}
        ep = epoch if epoch is not None else _read_epoch(src)
        rows = self._rows()
        if kind == "best":
            ds = float(d_seg) if d_seg is not None else -1.0
            if any(r["kind"] == "best" and r.get("epoch") == ep and round(float(r.get("d_seg", -1)), 9) == round(ds, 9) for r in rows):
                return None
            name = f"best_ep{ep}_dseg{ds:.6f}.npz" if ds >= 0 else f"best_ep{ep}.npz"
        else:  # 'latest' (or any periodic snap)
            if any(r["kind"] == kind and r.get("epoch") == ep for r in rows):
                return None
            name = f"{kind}_ep{ep}_{_utc()}.npz"
        dst = self.archive_dir / name
        _atomic_copy(src, dst)
        row = {"kind": kind, "epoch": ep, "d_seg": (float(d_seg) if d_seg is not None else None),
               "src": str(src), "dst": str(dst), "location": "local",
               "bytes": dst.stat().st_size, "sha256": _sha256(dst), "ts": _utc()}
        self._append(row)
        self._enforce_window()
        return row

    # ---- once-pass scan of a run dir (CLI + decorator use this) -----------------------------------
    def scan(self, run_dir: str | Path, *, best_json: str | Path | None = None,
             best_npz: str | Path | None = None, latest_npz: str | Path | None = None,
             snapshot_latest: bool = True) -> dict:
        run_dir = Path(run_dir)
        best_json = Path(best_json) if best_json else run_dir / "levelset_best.json"
        best_npz = Path(best_npz) if best_npz else run_dir / "levelset_witness_ema_BEST.npz"
        latest_npz = Path(latest_npz) if latest_npz else run_dir / "levelset_witness_ema_mlx.npz"
        added = {"best": 0, "latest": 0}
        if best_json.exists() and best_npz.exists():
            try:
                bj = json.loads(best_json.read_text())
                if self.keep(best_npz, kind="best", epoch=int(bj["epoch"]), d_seg=float(bj["d_seg"])):
                    added["best"] += 1
            except Exception as e:
                added["best_error"] = str(e)
        if snapshot_latest and latest_npz.exists():
            if self.keep(latest_npz, kind="latest", epoch=_read_epoch(latest_npz)):
                added["latest"] += 1
        return added

    # ---- moving-window spill (certify-or-block) ---------------------------------------------------
    def _enforce_window(self) -> None:
        if not self.keep_window or self.keep_window <= 0:
            return
        kinds = ["latest"] + (["best"] if self.spill_bests else [])
        for kind in kinds:
            local = sorted(self.archive_dir.glob(f"{kind}_ep*.npz"), key=lambda p: p.stat().st_mtime)
            overflow = local[:-self.keep_window] if len(local) > self.keep_window else []
            tier = _pick_spill_tier() if self.spill else None
            for p in overflow:
                if tier is None:
                    return  # no SSD tier -> BLOCK (keep local bytes; never delete without a cold copy)
                tier.mkdir(parents=True, exist_ok=True)
                dst = tier / p.name
                try:
                    _atomic_copy(p, dst)  # certify FIRST ...
                    self._append({"kind": f"{kind}_spilled", "epoch": _read_epoch(dst), "src": str(p),
                                  "dst": str(dst), "location": "ssd", "bytes": dst.stat().st_size,
                                  "sha256": _sha256(dst), "ts": _utc()})
                    p.unlink()  # ... THEN remove local (cold copy + manifest proof exist)
                    (p.with_suffix(p.suffix + ".pointer")).write_text(str(dst))
                except Exception:
                    pass  # BLOCK on any spill failure -> keep local bytes


# ---- module-level convenience: decorator + context manager + once-pass -----------------------------
def keep_checkpoint_history(run_dir: str | Path | None = None, archive_dir: str | Path | None = None,
                            *, snapshot_latest: bool = True, **policy: Any) -> Callable:
    """DROP-IN decorator. Wrap a trainer save fn; after it runs, the new best/latest are archived.

    Zero coupling: works with any save fn that writes the standard ``levelset_*`` files into ``run_dir``.
    ``run_dir`` defaults to (in priority) the fn's ``run_dir``/``out_dir`` kwarg, else the decorator arg.
    ``archive_dir`` defaults to ``<run_dir>/ckpt_archive``. Extra ``**policy`` -> ``CheckpointArchiver``
    (keep_window, spill, min_free_gb, ...). If the wrapped fn RETURNS a dict with {path,kind,epoch,d_seg}
    that exact ckpt is also kept (precise mode); otherwise the post-hoc scan picks up the new files.

        @keep_checkpoint_history(run_dir=out, keep_window=0)
        def _do_checkpoint(...): ...   # writes BEST.npz / mlx.npz / best.json
    """
    def _decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any):
            result = fn(*args, **kwargs)
            rd = run_dir or kwargs.get("run_dir") or kwargs.get("out_dir") or kwargs.get("out")
            if rd is not None:
                ad = archive_dir or (Path(rd) / "ckpt_archive")
                arch = CheckpointArchiver(ad, **policy)
                if isinstance(result, dict) and result.get("path"):
                    arch.keep(result["path"], kind=result.get("kind", "latest"),
                              epoch=result.get("epoch"), d_seg=result.get("d_seg"))
                arch.scan(rd, snapshot_latest=snapshot_latest)
            return result
        return _wrapped
    return _decorate


class checkpoint_history:  # noqa: N801 (context-manager style)
    """``with checkpoint_history(run_dir) as arch: ... arch.keep(p, kind='best', epoch=ep, d_seg=ds)``."""

    def __init__(self, run_dir: str | Path, archive_dir: str | Path | None = None, **policy: Any) -> None:
        self.run_dir = Path(run_dir)
        self.arch = CheckpointArchiver(archive_dir or (self.run_dir / "ckpt_archive"), **policy)

    def __enter__(self) -> CheckpointArchiver:
        return self.arch

    def __exit__(self, *exc: Any) -> None:
        self.arch.scan(self.run_dir)


def archive_run_once(run_dir: str | Path, archive_dir: str | Path | None = None, *,
                     snapshot_latest: bool = True, **policy: Any) -> dict:
    """One-pass archive of whatever best/latest exist in ``run_dir`` right now (CLI delegates here)."""
    arch = CheckpointArchiver(archive_dir or (Path(run_dir) / "ckpt_archive"), **policy)
    return arch.scan(run_dir, snapshot_latest=snapshot_latest)
