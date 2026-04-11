from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .paths import repo_root


def _lock_dir(name: str, upstream_root: Path) -> Path:
    token = hashlib.sha1(f"{name}|{upstream_root.resolve()}".encode("utf-8")).hexdigest()[:12]
    return repo_root() / ".omx" / "locks" / f"{name}-{token}.lock"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    Returns True if the process exists (or we cannot prove it does not).
    EPERM indicates the process is alive but owned by another user, which
    we still treat as 'alive' so we never steal a lock from a live holder.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # EPERM => alive but unprivileged; ESRCH => gone; anything else, be safe.
        if getattr(exc, "errno", None) == errno.EPERM:
            return True
        if getattr(exc, "errno", None) == errno.ESRCH:
            return False
        return True


@contextmanager
def submission_lock(name: str, upstream_root: Path):
    lock_dir = _lock_dir(name, upstream_root)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        # Check if the lock is stale (holder PID no longer running)
        meta_path = lock_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                holder_pid = meta.get("pid")
                if holder_pid and not _is_pid_alive(int(holder_pid)):
                    # Stale lock — holder is dead. Clean up and retry.
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    try:
                        os.mkdir(lock_dir)
                    except FileExistsError as exc2:
                        raise FileExistsError(
                            f"Lock contention for submission '{name}' at '{upstream_root}' "
                            f"(stale lock cleanup failed)."
                        ) from exc2
                else:
                    details = f" Existing lock metadata: {meta_path.read_text().strip()}"
                    raise FileExistsError(
                        f"Another package/smoke/eval operation is already active for "
                        f"submission '{name}' at upstream root '{upstream_root}'.{details}"
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as exc2:
                # Corrupted meta — treat as stale
                shutil.rmtree(lock_dir, ignore_errors=True)
                try:
                    os.mkdir(lock_dir)
                except FileExistsError as race_exc:
                    raise FileExistsError(
                        f"Lock contention for submission '{name}' at '{upstream_root}' "
                        f"(corrupted meta cleanup raced with another holder)."
                    ) from race_exc
        else:
            # No meta file — treat as stale
            shutil.rmtree(lock_dir, ignore_errors=True)
            try:
                os.mkdir(lock_dir)
            except FileExistsError as race_exc:
                raise FileExistsError(
                    f"Lock contention for submission '{name}' at '{upstream_root}' "
                    f"(missing-meta cleanup raced with another holder)."
                ) from race_exc

    meta = {
        "submission": name,
        "upstream_root": str(upstream_root),
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": os.getpid(),
    }
    (lock_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)
