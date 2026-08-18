"""Candidate seal contract — brick 1: receiver-pin consistency, derived at consumption.

WHY THIS EXISTS.  ``rr4``-lineage receivers pin the archive they decode: the staged
``inflate.py`` carries ``ARCHIVE_SHA256`` / ``ARCHIVE_BYTES`` module constants and refuses
any other bytes.  That pin is the r3 cure and it is correct — **archive and runtime are ONE
sealed object**.  What was missing is the local half: nothing checked the pin against the
archive actually staged beside it, so a tree whose pin named a *different* candidate was
only caught remotely, at decode time, after the paid meter had started.

THE INVARIANT IS A CONSISTENCY CONSTRAINT, NOT A STORED VALUE.  This module never latches
what the pin *should* say.  It measures the staged ``archive.zip`` at consumption and asks
one question: *does the receiver beside it name these exact bytes?*  That question is
answerable for rr4, for fx1, for sa1 and for every future candidate without editing a
literal — which is the whole defect it replaces (a hardcoded expectation refuses correct
candidates by construction).

VACUITY IS REPORTED, NOT PASSED.  A tree whose receiver carries no pin at all is
``PIN_ABSENT``, never ``CONSISTENT``.  A check that silently greens on an absent instrument
is the silent-instrument disease; callers get the distinct verdict and decide.

THE DETECTOR DOES NOT ZERO ON ITS OWN CURE.  ``repin_receiver`` rewrites the two constants
from the staged archive (the step MAIN previously did by hand), then RE-READS and RE-CHECKS
its own output, restoring the original bytes if the rewrite did not actually produce a
consistent tree.  Re-staging a different archive after a repin goes red again.

AXIS.  Byte identity only.  Nothing here computes, claims, or implies a score.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat as stat_module
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ARCHIVE_MISSING",
    "CONSISTENT",
    "MISMATCH",
    "PIN_ABSENT",
    "RECEIVER_MISSING",
    "ArchiveIdentity",
    "PinConsistency",
    "ReceiverPin",
    "RepinResult",
    "SealContractError",
    "check_pin_consistency",
    "measure_archive_identity",
    "read_frontier_archive_identity",
    "read_receiver_pin",
    "repin_receiver",
]

PIN_SHA_NAME = "ARCHIVE_SHA256"
PIN_BYTES_NAME = "ARCHIVE_BYTES"
DEFAULT_ARCHIVE_NAME = "archive.zip"
DEFAULT_RECEIVER_NAME = "inflate.py"

#: Verdicts.  Every non-``CONSISTENT`` value is a distinct, named reason — the caller is
#: never handed a bare boolean that cannot say *why*.
CONSISTENT = "CONSISTENT"
MISMATCH = "MISMATCH"
PIN_ABSENT = "PIN_ABSENT"
RECEIVER_MISSING = "RECEIVER_MISSING"
ARCHIVE_MISSING = "ARCHIVE_MISSING"

_POINTER_AXES = {
    "contest_cuda": "our_local_frontier_contest_cuda",
    "contest_cpu": "our_local_frontier_contest_cpu",
    "effective": "effective_frontier",
}


class SealContractError(RuntimeError):
    """Raised when a seal operation cannot be performed safely."""


def sha256_file(path: Path) -> str:
    """Stream a file through sha256 (archives are ~180 KB today; streaming stays cheap anyway)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArchiveIdentity:
    """The identity of one archive: its bytes, its sha, and where that claim came from."""

    sha256: str
    bytes: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "bytes": self.bytes, "source": self.source}


def measure_archive_identity(archive_path: Path) -> ArchiveIdentity:
    """Derive an archive identity from the actual bytes on disk. The only authority here."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise SealContractError(f"archive not found: {archive_path}")
    return ArchiveIdentity(
        sha256=sha256_file(archive_path),
        bytes=archive_path.stat().st_size,
        source=f"measured:{archive_path}",
    )


def read_frontier_archive_identity(
    pointer_path: Path | None = None,
    axis: str = "contest_cuda",
) -> ArchiveIdentity:
    """Read the shipped candidate's identity from the canonical frontier pointer, at call time.

    This is the dynamic default the hot state's BINDING CURE names: *derive every admission
    bar from ``canonical_frontier_pointer.json`` AT FIRE TIME; never latch a literal.*  A
    caller that wants a different candidate passes it explicitly; a caller that wants "the
    thing we currently ship" gets whatever the pointer says today, not what it said when
    some module was written.
    """
    if axis not in _POINTER_AXES:
        raise SealContractError(f"unknown pointer axis {axis!r}; expected one of {sorted(_POINTER_AXES)}")
    if pointer_path is None:
        pointer_path = Path(__file__).resolve().parents[2] / ".omx" / "state" / "canonical_frontier_pointer.json"
    pointer_path = Path(pointer_path)
    if not pointer_path.is_file():
        raise SealContractError(f"canonical frontier pointer not found: {pointer_path}")
    document = json.loads(pointer_path.read_text())
    node = document.get(_POINTER_AXES[axis])
    if not isinstance(node, dict):
        raise SealContractError(f"pointer {pointer_path} carries no {_POINTER_AXES[axis]!r} section")
    sha = node.get("archive_sha256")
    size = (node.get("extra") or {}).get("archive_bytes")
    if not sha or not isinstance(size, int):
        raise SealContractError(
            f"pointer axis {axis!r} is missing archive_sha256 or extra.archive_bytes "
            f"(sha={sha!r}, bytes={size!r}) — refusing to guess an admission bar"
        )
    return ArchiveIdentity(sha256=str(sha), bytes=int(size), source=f"frontier_pointer:{axis}:{pointer_path}")


@dataclass(frozen=True)
class ReceiverPin:
    """The archive identity a staged receiver claims, plus where in the file it claims it."""

    receiver_path: Path
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    sha_lineno: int | None = None
    bytes_lineno: int | None = None

    @property
    def is_present(self) -> bool:
        return self.archive_sha256 is not None and self.archive_bytes is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "receiver_path": str(self.receiver_path),
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "pin_present": self.is_present,
        }


def read_receiver_pin(receiver_path: Path) -> ReceiverPin:
    """Parse ``ARCHIVE_SHA256`` / ``ARCHIVE_BYTES`` out of a receiver WITHOUT importing it.

    Importing is not an option: a shipped ``inflate.py`` imports torch and its own runtime
    package at module scope, so an import would be a side effect on the reader's process and
    would fail outright off the candidate's own tree.  An AST read of top-level constant
    assignments is exact for the values we pin and cannot execute anything.
    """
    receiver_path = Path(receiver_path)
    if not receiver_path.is_file():
        raise SealContractError(f"receiver not found: {receiver_path}")
    try:
        tree = ast.parse(receiver_path.read_text(encoding="utf-8", errors="strict"))
    except (SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover - defensive
        raise SealContractError(f"receiver {receiver_path} is not parsable Python: {exc}") from exc

    sha: str | None = None
    size: int | None = None
    sha_line: int | None = None
    size_line: int | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Constant):
            continue
        # A multi-line assignment cannot be rewritten by single-line substitution; refuse to
        # read it as pinnable rather than risk corrupting the receiver later.
        if node.end_lineno is not None and node.end_lineno != node.lineno:
            continue
        if target.id == PIN_SHA_NAME and isinstance(node.value.value, str):
            sha, sha_line = node.value.value, node.lineno
        elif target.id == PIN_BYTES_NAME and isinstance(node.value.value, int):
            size, size_line = int(node.value.value), node.lineno
    return ReceiverPin(
        receiver_path=receiver_path,
        archive_sha256=sha,
        archive_bytes=size,
        sha_lineno=sha_line,
        bytes_lineno=size_line,
    )


@dataclass(frozen=True)
class PinConsistency:
    """The verdict of one pin-consistency check, with both sides of the comparison kept."""

    verdict: str
    runtime_dir: Path
    receiver_path: Path
    archive_path: Path
    measured_sha256: str | None = None
    measured_bytes: int | None = None
    pinned_sha256: str | None = None
    pinned_bytes: int | None = None
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.verdict == CONSISTENT

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "runtime_dir": str(self.runtime_dir),
            "receiver_path": str(self.receiver_path),
            "archive_path": str(self.archive_path),
            "measured_sha256": self.measured_sha256,
            "measured_bytes": self.measured_bytes,
            "pinned_sha256": self.pinned_sha256,
            "pinned_bytes": self.pinned_bytes,
            "problems": list(self.problems),
        }

    def summary(self) -> str:
        if self.verdict == CONSISTENT:
            return (
                f"SEAL PIN CONSISTENT: receiver pins the staged archive "
                f"({self.measured_bytes:,} B sha {str(self.measured_sha256)[:16]}…)"
            )
        return f"SEAL PIN {self.verdict}: " + "; ".join(self.problems)


def check_pin_consistency(
    runtime_dir: Path,
    archive_path: Path | None = None,
    receiver_path: Path | None = None,
) -> PinConsistency:
    """Ask the one question: does the staged receiver name the staged archive's exact bytes?

    Nothing is compared against a remembered value.  Both sides are read at call time from
    the tree in front of us, which is what makes this work for any candidate.
    """
    runtime_dir = Path(runtime_dir)
    archive_path = Path(archive_path) if archive_path else runtime_dir / DEFAULT_ARCHIVE_NAME
    receiver_path = Path(receiver_path) if receiver_path else runtime_dir / DEFAULT_RECEIVER_NAME

    if not receiver_path.is_file():
        return PinConsistency(
            verdict=RECEIVER_MISSING,
            runtime_dir=runtime_dir,
            receiver_path=receiver_path,
            archive_path=archive_path,
            problems=(f"no receiver at {receiver_path}",),
        )
    if not archive_path.is_file():
        return PinConsistency(
            verdict=ARCHIVE_MISSING,
            runtime_dir=runtime_dir,
            receiver_path=receiver_path,
            archive_path=archive_path,
            problems=(f"no archive at {archive_path}",),
        )

    pin = read_receiver_pin(receiver_path)
    measured = measure_archive_identity(archive_path)

    if not pin.is_present:
        missing = [
            name
            for name, value in ((PIN_SHA_NAME, pin.archive_sha256), (PIN_BYTES_NAME, pin.archive_bytes))
            if value is None
        ]
        return PinConsistency(
            verdict=PIN_ABSENT,
            runtime_dir=runtime_dir,
            receiver_path=receiver_path,
            archive_path=archive_path,
            measured_sha256=measured.sha256,
            measured_bytes=measured.bytes,
            pinned_sha256=pin.archive_sha256,
            pinned_bytes=pin.archive_bytes,
            problems=(
                f"{receiver_path.name} declares no {' and no '.join(missing)} — this tree is UNPINNED, "
                "so the archive+runtime seal is weaker than the rr4 lineage's and this check is vacuous "
                "for it (reported, never silently passed)",
            ),
        )

    problems: list[str] = []
    if pin.archive_sha256 != measured.sha256:
        problems.append(
            f"{PIN_SHA_NAME} pins {pin.archive_sha256[:16]}… but the staged archive is {measured.sha256[:16]}…"
        )
    if pin.archive_bytes != measured.bytes:
        problems.append(f"{PIN_BYTES_NAME} pins {pin.archive_bytes:,} B but the staged archive is {measured.bytes:,} B")

    if problems:
        problems.append(
            "archive and runtime are ONE sealed object: re-pin the receiver from the staged "
            "archive (tac.candidate_seal.repin_receiver) or stage the archive this receiver names"
        )
        verdict = MISMATCH
    else:
        verdict = CONSISTENT

    return PinConsistency(
        verdict=verdict,
        runtime_dir=runtime_dir,
        receiver_path=receiver_path,
        archive_path=archive_path,
        measured_sha256=measured.sha256,
        measured_bytes=measured.bytes,
        pinned_sha256=pin.archive_sha256,
        pinned_bytes=pin.archive_bytes,
        problems=tuple(problems),
    )


@dataclass(frozen=True)
class RepinResult:
    """What a re-pin did, with the before/after verdicts so the caller can see it worked."""

    receiver_path: Path
    changed: bool
    dry_run: bool
    verdict_before: str
    verdict_after: str
    old_sha256: str | None
    old_bytes: int | None
    new_sha256: str
    new_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "receiver_path": str(self.receiver_path),
            "changed": self.changed,
            "dry_run": self.dry_run,
            "verdict_before": self.verdict_before,
            "verdict_after": self.verdict_after,
            "old_sha256": self.old_sha256,
            "old_bytes": self.old_bytes,
            "new_sha256": self.new_sha256,
            "new_bytes": self.new_bytes,
        }


def repin_receiver(
    runtime_dir: Path,
    archive_path: Path | None = None,
    receiver_path: Path | None = None,
    dry_run: bool = False,
) -> RepinResult:
    """Re-pin the staged receiver's two constants from the staged archive.

    This is the compose-time step that makes staging a non-rr4 candidate legal: the pin is
    DERIVED from the bytes being staged instead of carried in from whichever candidate the
    tree was copied from.  Only the two constant lines change; every other byte of the
    receiver is preserved.

    The write verifies itself.  After the rewrite the receiver is re-parsed and re-checked,
    and the original bytes are restored if the result is not ``CONSISTENT`` — a repair that
    cannot prove it repaired anything must not be left on disk.
    """
    runtime_dir = Path(runtime_dir)
    archive_path = Path(archive_path) if archive_path else runtime_dir / DEFAULT_ARCHIVE_NAME
    receiver_path = Path(receiver_path) if receiver_path else runtime_dir / DEFAULT_RECEIVER_NAME

    before = check_pin_consistency(runtime_dir, archive_path=archive_path, receiver_path=receiver_path)
    if before.verdict in (RECEIVER_MISSING, ARCHIVE_MISSING):
        raise SealContractError(f"cannot re-pin: {before.summary()}")

    pin = read_receiver_pin(receiver_path)
    if not pin.is_present:
        raise SealContractError(
            f"cannot re-pin {receiver_path}: it declares no {PIN_SHA_NAME}/{PIN_BYTES_NAME} constants. "
            "Adding a pin to an unpinned receiver is a receiver design change, not a staging step."
        )

    measured = measure_archive_identity(archive_path)
    if before.verdict == CONSISTENT:
        return RepinResult(
            receiver_path=receiver_path,
            changed=False,
            dry_run=dry_run,
            verdict_before=before.verdict,
            verdict_after=before.verdict,
            old_sha256=pin.archive_sha256,
            old_bytes=pin.archive_bytes,
            new_sha256=measured.sha256,
            new_bytes=measured.bytes,
        )

    original = receiver_path.read_bytes()
    lines = receiver_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert pin.sha_lineno is not None and pin.bytes_lineno is not None  # is_present guarantees both
    eol = "\r\n" if lines[pin.sha_lineno - 1].endswith("\r\n") else "\n"
    lines[pin.sha_lineno - 1] = f'{PIN_SHA_NAME} = "{measured.sha256}"{eol}'
    lines[pin.bytes_lineno - 1] = f"{PIN_BYTES_NAME} = {measured.bytes:_d}{eol}"
    rewritten = "".join(lines)

    if dry_run:
        return RepinResult(
            receiver_path=receiver_path,
            changed=True,
            dry_run=True,
            verdict_before=before.verdict,
            verdict_after="NOT_WRITTEN_DRY_RUN",
            old_sha256=pin.archive_sha256,
            old_bytes=pin.archive_bytes,
            new_sha256=measured.sha256,
            new_bytes=measured.bytes,
        )

    mode = stat_module.S_IMODE(receiver_path.stat().st_mode)
    temporary = receiver_path.with_name(receiver_path.name + ".repin.tmp")
    temporary.write_text(rewritten, encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, receiver_path)

    after = check_pin_consistency(runtime_dir, archive_path=archive_path, receiver_path=receiver_path)
    if not after.ok:
        receiver_path.write_bytes(original)
        os.chmod(receiver_path, mode)
        raise SealContractError(
            f"re-pin did not produce a consistent tree ({after.verdict}); original receiver bytes restored. "
            + "; ".join(after.problems)
        )

    return RepinResult(
        receiver_path=receiver_path,
        changed=True,
        dry_run=False,
        verdict_before=before.verdict,
        verdict_after=after.verdict,
        old_sha256=pin.archive_sha256,
        old_bytes=pin.archive_bytes,
        new_sha256=measured.sha256,
        new_bytes=measured.bytes,
    )
