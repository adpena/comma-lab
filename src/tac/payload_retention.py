"""Canonical payload retention — the CURE side of the ALWAYS KEEP THE PAYLOAD rule.

Operator binding 2026-08-09 (P0, DEF CON 1000): a run that materializes a payload MUST
persist it; a scalar-only artifact is forbidden. :mod:`tac.payload_retention_gate` refuses
the defect. This module is what a caller uses to comply, in one line::

    from tac.payload_retention import retain_payload, retention_root

    out = retention_root("ddm_lv1", need_bytes=estimated)
    record = retain_payload(out / "tokens.zlib9.bin", payload)
    rows["zlib9"] = record["bytes"]          # the scalar, AND the bytes survive

``record`` carries runtime ``path``, durable ``portable_path``, ``bytes``, and ``sha256``
so the next consumer can prove byte-identity instead of re-encoding without publishing
one operator's mount layout (the anchor incident cost two full re-encodes of bytes we
had already produced once, and delayed a measured -2,120 B rate win).

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
    "DEFAULT_TIER1_ROOT",
    "DEFAULT_TIER2_ROOT",
    "LOCAL_FALLBACK_ROOT",
    "SSD_TIERS",
    "PayloadRetentionError",
    "portable_path_form",
    "portable_retention_record",
    "resolve_portable_path",
    "retain_candidates",
    "retain_payload",
    "retention_root",
    "storage_tiers",
]

#: Runtime defaults from CLAUDE.md. Public artifacts should use ``$PACT_TIER1`` /
#: ``$PACT_TIER2``; the absolute defaults exist only at the resolver boundary.
DEFAULT_TIER1_ROOT = Path("/Volumes/VertigoDataTier/pact")  # ABSOLUTE_PATH_OK:canonical-runtime-default
DEFAULT_TIER2_ROOT = Path("/Volumes/APDataStore/pact")  # ABSOLUTE_PATH_OK:canonical-runtime-default

#: Local tier. Used only when ``allow_local=True`` is passed explicitly, per the
#: "local disk only by explicit opt-in" clause.
LOCAL_FALLBACK_ROOT = Path("experiments/results")

#: Free space a tier must retain AFTER the write, so retention never fills a volume.
_RESERVE_BYTES = 2 * 1024**3


class PayloadRetentionError(RuntimeError):
    """No tier can hold the payload, or a write could not be completed."""


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def storage_tiers(environ: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    """Resolve the two canonical SSD tiers from environment or runtime defaults."""

    env = _environment(environ)
    return (
        Path(env.get("PACT_TIER1") or DEFAULT_TIER1_ROOT).expanduser(),
        Path(env.get("PACT_TIER2") or DEFAULT_TIER2_ROOT).expanduser(),
    )


#: Compatibility snapshot for callers that import the historical constant. New code
#: should call :func:`storage_tiers` so environment changes made before a run are seen.
SSD_TIERS: tuple[Path, ...] = storage_tiers()


def resolve_portable_path(
    value: str | os.PathLike[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve ``~``/``$HOME``/``$PACT_TIER1``/``$PACT_TIER2`` deterministically.

    This intentionally expands only the four public path tokens. Arbitrary environment
    expansion in provenance records would make a typo silently change custody.
    """

    env = _environment(environ)
    home = Path(env.get("HOME") or Path.home())
    tier1, tier2 = storage_tiers(env)
    text = os.fspath(value)
    replacements = (
        ("${PACT_TIER1}", tier1),
        ("$PACT_TIER1", tier1),
        ("${PACT_TIER2}", tier2),
        ("$PACT_TIER2", tier2),
        ("${HOME}", home),
        ("$HOME", home),
        ("~", home),
    )
    for token, root in replacements:
        if text == token:
            return root
        if text.startswith(token + "/"):
            return root / text[len(token) + 1 :]
    return Path(text)


def portable_path_form(
    value: str | os.PathLike[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return a public placeholder form for a resolved local path when possible."""

    env = _environment(environ)
    raw = os.fspath(value)
    if not Path(raw).is_absolute() and not raw.startswith(
        ("~", "$HOME", "${HOME}", "$PACT_TIER1", "${PACT_TIER1}", "$PACT_TIER2", "${PACT_TIER2}")
    ):
        return raw
    resolved = resolve_portable_path(value, environ=env).expanduser().resolve(strict=False)
    tier1, tier2 = (
        tier.expanduser().resolve(strict=False) for tier in storage_tiers(env)
    )
    home = Path(env.get("HOME") or Path.home()).expanduser().resolve(strict=False)
    candidates = sorted(
        ((tier1, "$PACT_TIER1"), (tier2, "$PACT_TIER2"), (home, "~")),
        key=lambda item: len(item[0].parts),
        reverse=True,
    )
    for root, token in candidates:
        if resolved == root:
            return token
        try:
            suffix = resolved.relative_to(root)
        except ValueError:
            continue
        return f"{token}/{suffix.as_posix()}"
    return raw


def portable_retention_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return the custody row safe to persist in a public durable artifact.

    ``retain_payload`` keeps ``path`` runtime-resolvable for the immediate caller. A
    writer must pass that row through this function before JSON serialization; the
    resulting ``path`` is portable and the redundant runtime-only field is absent.
    """

    durable = dict(record)
    path = durable.get("path")
    portable = durable.pop("portable_path", None)
    if portable is None:
        if not isinstance(path, (str, os.PathLike)):
            raise PayloadRetentionError("retention row has no resolvable path")
        portable = portable_path_form(path)
    durable["path"] = os.fspath(portable)
    return durable


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
    candidates = list(storage_tiers() if tiers is None else tiers)
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

    ``path`` is the resolved runtime path for immediate consumers; ``portable_path`` is
    the public/durable form to write into provenance records. Both identify the same
    retained bytes and are covered by the row's size and SHA-256.
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
        "portable_path": portable_path_form(target),
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
