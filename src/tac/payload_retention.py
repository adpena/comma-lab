"""Canonical payload retention — the CURE side of the ALWAYS KEEP THE PAYLOAD rule.

Operator binding 2026-08-09 (P0, DEF CON 1000): a run that materializes a payload MUST
persist it; a scalar-only artifact is forbidden. :mod:`tac.payload_retention_gate` refuses
the defect. This module is what a caller uses to comply, in one line::

    from tac.payload_retention import retain_payload, retention_root

    out = retention_root("ddm_lv1", need_bytes=estimated)
    record = retain_payload(out / "tokens.zlib9.bin", payload)
    rows["zlib9"] = record["bytes"]          # the scalar, AND the bytes survive

``record`` carries ``path`` / ``bytes`` / ``sha256`` so the next consumer can prove
byte-identity instead of re-encoding (the anchor incident cost two full re-encodes of
bytes we had already produced once, and delayed a measured -2,120 B rate win).

STORAGE ROUTING. CLAUDE.md orders the tiers VertigoDataTier, then APDataStore, then local
by explicit opt-in. The order alone is not enough: on 2026-08-16 VertigoDataTier measured
893 MiB free (100% capacity) while APDataStore had 240 GiB, so a fixed first-tier write
would fail mid-run. :func:`retention_root` therefore walks the tiers in the canonical
order and takes the first one that can actually hold ``need_bytes``, and FAILS CLOSED
naming the free space per tier when none can. Running out of disk is a routing question,
never a licence to discard — per the certify-or-block rule, retention is a precondition
for running.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = [
    "LOCAL_FALLBACK_ROOT",
    "SSD_TIERS",
    "PayloadRetentionError",
    "retain_candidates",
    "retain_payload",
    "retention_root",
]

#: Canonical SSD tiers, in CLAUDE.md's priority order.
SSD_TIERS: tuple[Path, ...] = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

#: Local tier. Used only when ``allow_local=True`` is passed explicitly, per the
#: "local disk only by explicit opt-in" clause.
LOCAL_FALLBACK_ROOT = Path("experiments/results")

#: Free space a tier must retain AFTER the write, so retention never fills a volume.
_RESERVE_BYTES = 2 * 1024**3


class PayloadRetentionError(RuntimeError):
    """No tier can hold the payload, or a write could not be completed."""


def _free_bytes(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def retention_root(
    arm: str,
    *,
    need_bytes: int = 0,
    allow_local: bool = False,
    tiers: Sequence[Path] | None = None,
) -> Path:
    """Return ``<tier>/<arm>/retained/``, creating it, on the first tier that fits.

    Fails closed with the measured free space per tier rather than silently writing
    somewhere that cannot hold the run.
    """
    candidates = list(SSD_TIERS if tiers is None else tiers)
    if allow_local:
        candidates.append(LOCAL_FALLBACK_ROOT.resolve() / "_retained")
    required = max(int(need_bytes), 0) + _RESERVE_BYTES
    surveyed: list[str] = []
    for tier in candidates:
        free = _free_bytes(tier)
        if free is None:
            surveyed.append(f"{tier}: unavailable")
            continue
        if free < required:
            surveyed.append(f"{tier}: {free / 1024**3:.2f} GiB free")
            continue
        root = tier / arm / "retained"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            surveyed.append(f"{tier}: unwritable ({exc})")
            continue
        return root
    raise PayloadRetentionError(
        f"ALWAYS KEEP THE PAYLOAD: no tier can hold {need_bytes} B "
        f"(+{_RESERVE_BYTES / 1024**3:.0f} GiB reserve) for arm {arm!r}. "
        f"Surveyed -> {'; '.join(surveyed) or 'no tiers configured'}. "
        f"This is a storage-ROUTING failure: free a tier, attach one, or pass "
        f"allow_local=True. Do NOT proceed without persisting the payload."
    )


def retain_payload(path: Path | str, payload: bytes) -> dict[str, object]:
    """Persist ``payload`` atomically; return its custody row.

    The row is ``{"path", "bytes", "sha256"}`` — record it in the result JSON so the
    next consumer can prove byte-identity instead of re-encoding.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = bytes(payload)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PayloadRetentionError(f"could not retain payload at {target}: {exc}") from exc
    return {
        "path": str(target),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def retain_candidates(
    root: Path | str,
    payloads: Mapping[str, bytes],
    *,
    suffix: str = ".bin",
) -> dict[str, dict[str, object]]:
    """Retain EVERY candidate's payload, not only the winner's.

    The anchor's loser turned out to be the winner: the discarded ANS payload later
    measured -2,120 B against the shipped range coder. Keeping only the best candidate
    is the same defect one step later.
    """
    base = Path(root)
    return {
        name: retain_payload(base / f"{name}{suffix}", payload)
        for name, payload in payloads.items()
    }
