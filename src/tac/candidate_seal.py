"""Candidate seal contract — what "sealed" MEANS, derived at consumption, never remembered.

Two bricks.  **Brick 1** (below) answers one question about one staged tree: does the
receiver beside this archive name these exact bytes?  **Brick 2** (further down) makes
"sealed" a checkable noun: one typed document that freezes every pin a fire needs, plus the
validator ``tools/fire_modal_auth_eval.py`` consumes before it will spend a paid call.

WHY BRICK 1 EXISTS.  ``rr4``-lineage receivers pin the archive they decode: the staged
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

--------------------------------------------------------------------------------------
BRICK 2 — THE SEAL DOCUMENT (task #1115, operator 2026-08-18: *"Things can be better
frozen and constrained through engineering."*)

Brick 1 answers one question about one tree.  Brick 2 makes **"sealed" a checkable noun**:
a single typed document that FREEZES every pin a fire needs, and a validator the fire path
CONSUMES so the invalid state is unrepresentable rather than patrolled by attention.

Each field exists because a measured failure needed it, and the validator re-derives that
field from disk at CONSUMPTION time — never at seal time, never from memory:

* ``archive`` (sha + bytes) — ``rr2`` fired a hand-assembled tree whose bytes were never the
  proved bytes (S 27.83 vs projected 0.1585).
* ``runtime`` (content-only FILES digest) — ``ps1u`` r1 failed on a receiver pin that drifted
  between seal and fire.  The digest is CONTENT-ONLY on purpose: the r9m deadlock was two
  validators disagreeing over an environment-coupled tree hash, and the standing cure is a
  digest both sides compute from bytes alone.
* ``receiver_pins`` — per-file shas for the load-bearing decode files, so a drift report can
  name the file instead of only the tree.
* ``admit_bar`` with its DERIVATION INPUTS — ``qs4`` carried ``qs2``'s compensation constant
  onto a different object (+2.4e-4 S).  A bar is meaningless without the baseline it was
  derived against, and baselines move: the seal stores the pointer score it was derived
  from, and the validator REFUSES if the pointer has moved beyond the declared tolerance.
* ``axis`` — CPU and CUDA are separate evidence spaces; an axis waiver hand-supplied at fire
  time is exactly the hand-assembly the error-factory law forbids.
* ``retained_payload_paths`` — ALWAYS KEEP THE PAYLOAD, checked, not asserted.
* ``seal_sha256`` — over the canonical serialization of all of the above, so an edited seal
  is a ``SEAL_TAMPERED`` refusal rather than a quietly different fire.

TWO CONSTRAINTS MAKE THE DOCUMENT HONEST.  (1) **No placeholder passes**: an empty string,
a ``"TBD"``, a non-hex sha, or an all-zero digest REFUSES (Catalog #287 lifted from rationale
strings to data pins).  (2) **The digest is invariant under the fire path's own sanitize
stage** by construction — it skips exactly what the transport zip skips
(``runtime_upload_skip_reason``), so removing macOS ``._`` litter provably cannot move the
number the seal froze.  A seal that its own consumer's first stage could invalidate would be
a seal in name only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat as stat_module
from dataclasses import dataclass, field
from pathlib import Path

from tac.artifact_moved import MovedArtifactError
from tac.artifact_moved import resolve as resolve_artifact

__all__ = [
    "ARCHIVE_MISSING",
    "CONSISTENT",
    "MISMATCH",
    "PIN_ABSENT",
    "PUBLIC_ENTRYPOINT_SMOKE_SCHEMA",
    "RECEIVER_MISSING",
    "SEAL_AXES",
    "SEAL_BAR_DRIFT",
    "SEAL_BYTE_DRIFT",
    "SEAL_DECODE_WALL_CLOCK_INVALID",
    "SEAL_DECODE_WALL_CLOCK_MISSING",
    "SEAL_FILE_MISSING",
    "SEAL_PLACEHOLDER_PIN",
    "SEAL_PUBLIC_SMOKE_INVALID",
    "SEAL_PUBLIC_SMOKE_MISSING",
    "SEAL_RECEIVER_PIN_MISMATCH",
    "SEAL_RUNTIME_DRIFT",
    "SEAL_SCHEMA",
    "SEAL_SCHEMA_V1",
    "SEAL_SCHEMA_VIOLATION",
    "SEAL_SHA_DRIFT",
    "SEAL_TAMPERED",
    "SEAL_VALID",
    "AdmitBar",
    "ArchiveIdentity",
    "PinConsistency",
    "ReceiverPin",
    "RepinResult",
    "RuntimeDigest",
    "SealContractError",
    "SealValidation",
    "build_seal",
    "canonical_seal_bytes",
    "check_pin_consistency",
    "compute_seal_sha256",
    "load_seal",
    "measure_archive_identity",
    "measure_runtime_digest",
    "read_archive_member_identity",
    "read_frontier_archive_identity",
    "read_pointer_state",
    "read_receiver_pin",
    "repin_receiver",
    "validate_seal",
    "write_seal",
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
        # ASSIGN_ONLY_OK: the pin reader accepts ONLY the exact single-line `NAME = <const>` shape the rewriter at write_receiver_pin (:484-485) EMITS via whole-line replacement — a read/write round-trip contract on a sealed surface. Any other shape, including an annotated pin, reads as pin-absent and the consistency check refuses FAIL-CLOSED (never silently wrong); all receiver pins are machine-emitted in this shape by our own writers.
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


# ======================================================================================
# BRICK 2 — the seal DOCUMENT: freeze every pin, constrain the fire path to consume it.
# ======================================================================================

SEAL_SCHEMA_V1 = "candidate_seal.v1"
SEAL_SCHEMA = "candidate_seal.v2"
SEAL_DECODE_WALL_CLOCK_MISSING = "SEAL_DECODE_WALL_CLOCK_MISSING"
SEAL_DECODE_WALL_CLOCK_INVALID = "SEAL_DECODE_WALL_CLOCK_INVALID"

#: Verdicts of the seal-document layer.  Deliberately a separate namespace from brick 1's
#: pin verdicts: a caller must never confuse "this tree's receiver names this archive"
#: (a one-question check) with "this sealed object is still exactly what was sealed".
SEAL_VALID = "SEAL_VALID"
SEAL_SCHEMA_VIOLATION = "SEAL_SCHEMA_VIOLATION"
SEAL_PLACEHOLDER_PIN = "SEAL_PLACEHOLDER_PIN"
SEAL_PUBLIC_SMOKE_MISSING = "SEAL_PUBLIC_SMOKE_MISSING"
SEAL_PUBLIC_SMOKE_INVALID = "SEAL_PUBLIC_SMOKE_INVALID"
SEAL_FILE_MISSING = "SEAL_FILE_MISSING"
SEAL_SHA_DRIFT = "SEAL_SHA_DRIFT"
SEAL_BYTE_DRIFT = "SEAL_BYTE_DRIFT"
SEAL_RUNTIME_DRIFT = "SEAL_RUNTIME_DRIFT"
SEAL_BAR_DRIFT = "SEAL_BAR_DRIFT"
SEAL_TAMPERED = "SEAL_TAMPERED"
#: The staged receiver's OWN ``ARCHIVE_SHA256``/``ARCHIVE_BYTES`` constants name a different
#: archive than the one sealed beside it — a tree that cannot decode itself.  Distinct from
#: ``SEAL_RUNTIME_DRIFT`` (the receiver FILE changed) and from the ``RECEIVER_MISSING`` /
#: ``ARCHIVE_MISSING`` pin verdicts (nothing to compare).  Here both sides are present, intact,
#: and DISAGREE.
SEAL_RECEIVER_PIN_MISMATCH = "SEAL_RECEIVER_PIN_MISMATCH"

#: The field names deliberately follow the already-retained rc1/sj1 smoke vocabulary:
#: ``public_path_probes`` is the direct ``f26_inflate.inflate_archive`` leg and
#: ``inflate_sh_smokes`` is the public-shell leg.  This block adds identity and exception
#: pins to those receipts; it does not create a parallel boolean pass flag.
PUBLIC_ENTRYPOINT_SMOKE_SCHEMA = "candidate_public_entrypoint_smoke.v1"
_SMOKE_ROLES = ("candidate", "frontier")
_CUDA_GATE_EXCEPTION = "RuntimeError"
_CUDA_GATE_MESSAGE = "requires CUDA inflation on linux-nvidia-t4"

#: The axes a seal may declare.  ``advisory`` is included so a non-promotable local row can
#: still be sealed with the same rigor — but it is NEVER a contest score, and the fire path
#: refuses to dispatch a paid Modal call on an advisory seal.
SEAL_AXES = ("contest_cuda", "contest_cpu", "advisory")

#: Strings that look like a value but are the absence of one.  Catalog #287's placeholder
#: rejection, lifted from waiver rationales to structured data pins: a seal carrying
#: ``"pending_ratification"`` where a sha belongs is not a weaker seal, it is not a seal.
_PLACEHOLDER_TOKENS = frozenset(
    {
        "",
        "?",
        "fixme",
        "n/a",
        "na",
        "none",
        "null",
        "pending",
        "pending_ratification",
        "placeholder",
        "tbd",
        "todo",
        "unknown",
        "xxx",
        "<value>",
        "<sha>",
        "<path>",
        "<candidate>",
    }
)


def _is_placeholder(value: object) -> bool:
    """True when ``value`` is a string that stands in for a value instead of being one."""
    if not isinstance(value, str):
        return False
    token = value.strip().lower()
    if token in _PLACEHOLDER_TOKENS:
        return True
    # Angle-bracket templates of any wording: "<archive sha here>", "<fill me in>".
    return token.startswith("<") and token.endswith(">")


def _is_sha256(value: object) -> bool:
    """A real sha256: 64 hex digits and not the all-zero digest (a common stand-in)."""
    if not isinstance(value, str):
        return False
    token = value.strip().lower()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        return False
    return set(token) != {"0"}


@dataclass(frozen=True)
class RuntimeDigest:
    """A content-only digest of a runtime tree: what ships, hashed from bytes alone.

    ``sha256`` covers the sorted ``(relative_path, bytes, sha256)`` triples of every file the
    transport zip would actually carry.  Nothing environment-coupled enters it — no absolute
    path, no mtime, no import manifest — because the r9m deadlock was exactly two validators
    disagreeing over an env-coupled tree hash while the bytes were identical.
    """

    sha256: str
    file_count: int
    total_bytes: int
    files: tuple[tuple[str, int, str], ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "digest_definition": "tac.candidate_seal.measure_runtime_digest",
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }

    def file_map(self) -> dict[str, tuple[int, str]]:
        return {rel: (size, sha) for rel, size, sha in self.files}


def runtime_digest_skip_reason(rel: str) -> str | None:
    """Why ``rel`` is outside the digest: because it provably cannot reach the evaluator.

    Two disjoint reasons, and both are properties of the SHIPPING path, not preferences:

    * ``runtime_upload_skip_reason`` — host metadata and bytecode caches the transport zip
      drops on its way out.
    * a hidden path part — ``validate_runtime_upload_file`` REFUSES any dot-prefixed file or
      directory outright, so a hidden file is not a quiet member of the tree; it is a file
      that would abort the upload. The fire path deletes the two macOS kinds (``._*``,
      ``.DS_Store``) in its sanitize stage and refuses the rest.

    Defining the digest over exactly the shippable set is what makes it survive its own
    consumer. macOS re-creates AppleDouble ``._`` litter on ExFAT the instant anything writes
    to a custody volume — if that litter entered the digest, a seal written on the SSD tier
    would refuse itself minutes later for a reason having nothing to do with the candidate.
    Nothing is lost by excluding it: a hidden file cannot ship, so it cannot change a score.
    """
    from tac.deploy.modal.auth_eval import runtime_upload_skip_reason

    reason = runtime_upload_skip_reason(rel)
    if reason:
        return reason
    if any(part.startswith(".") for part in Path(rel).parts):
        return "hidden path — the upload validator refuses it, so it can never ship"
    return None


def measure_runtime_digest(runtime_dir: Path) -> RuntimeDigest:
    """Hash the runtime tree over exactly the files that can reach the evaluator.

    INVARIANT UNDER THE FIRE PATH'S OWN SANITIZE STAGE, by construction — see
    ``runtime_digest_skip_reason``.  Executed as a control in
    ``src/tac/tests/test_candidate_seal.py``, which is where this definition was corrected:
    a first cut skipped only the transport-zip set, and the control caught ``._inflate.py``
    moving the digest.
    """
    runtime_dir = Path(runtime_dir)
    if not runtime_dir.is_dir():
        raise SealContractError(f"runtime dir not found: {runtime_dir}")

    rows: list[tuple[str, int, str]] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(runtime_dir).as_posix()
        if runtime_digest_skip_reason(rel):
            continue
        rows.append((rel, path.stat().st_size, sha256_file(path)))

    rows.sort()
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return RuntimeDigest(
        sha256=hashlib.sha256(payload).hexdigest(),
        file_count=len(rows),
        total_bytes=sum(size for _, size, _ in rows),
        files=tuple(rows),
    )


def read_archive_member_identity(archive_path: Path, member_name: str) -> tuple[str, int]:
    """Return ``(sha256, bytes)`` of one member INSIDE an archive, read from the zip.

    Our archives are one or two members (``0.bin`` carries ~99% of the bytes), so the member
    pin is the finest-grained statement a seal can make about what will actually be decoded.
    """
    import zipfile

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise SealContractError(f"archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        if member_name not in zf.namelist():
            raise SealContractError(
                f"archive {archive_path} has no member {member_name!r}; members: {sorted(zf.namelist())}"
            )
        blob = zf.read(member_name)
    return hashlib.sha256(blob).hexdigest(), len(blob)


def read_pointer_state(pointer_path: Path | None = None, axis: str = "contest_cuda") -> dict[str, object]:
    """Read the live admission baseline: the pointer's score AND the candidate it belongs to.

    A bar derived against a baseline is only meaningful while that baseline holds, and
    baselines move — this is what the seal freezes so the validator can notice.
    """
    if axis not in _POINTER_AXES:
        raise SealContractError(f"unknown pointer axis {axis!r}; expected one of {sorted(_POINTER_AXES)}")
    if pointer_path is None:
        pointer_path = Path(__file__).resolve().parents[2] / ".omx" / "state" / "canonical_frontier_pointer.json"
    pointer_path = Path(pointer_path)
    if not pointer_path.is_file():
        raise SealContractError(f"canonical frontier pointer not found: {pointer_path}")
    node = json.loads(pointer_path.read_text()).get(_POINTER_AXES[axis])
    if not isinstance(node, dict):
        raise SealContractError(f"pointer {pointer_path} carries no {_POINTER_AXES[axis]!r} section")
    score = node.get("score")
    sha = node.get("archive_sha256")
    if not isinstance(score, (int, float)) or not sha:
        raise SealContractError(
            f"pointer axis {axis!r} is missing score or archive_sha256 (score={score!r}, sha={sha!r}) "
            "— refusing to derive an admission bar from an incomplete baseline"
        )
    return {
        "pointer_axis": axis,
        "pointer_path": str(pointer_path),
        "pointer_score": float(score),
        "pointer_archive_sha256": str(sha),
    }


@dataclass(frozen=True)
class AdmitBar:
    """The falsifier a fired row will be judged against, WITH the inputs it was derived from.

    ``net_dS_threshold`` alone is the qs4 disease: a number carried across regimes with no
    record of the object it was fitted to.  Storing the derivation inputs turns the bar into
    a re-derivable claim — the validator re-reads the pointer and refuses when the ground it
    stood on has moved further than ``pointer_tolerance_abs``.
    """

    rule: str
    net_dS_threshold: float
    pointer_axis: str
    pointer_score_at_seal: float
    pointer_archive_sha256_at_seal: str
    pointer_tolerance_abs: float = 0.0
    require_pointer_archive_identity: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "net_dS_threshold": self.net_dS_threshold,
            "derivation": {
                "pointer_axis": self.pointer_axis,
                "pointer_score_at_seal": self.pointer_score_at_seal,
                "pointer_archive_sha256_at_seal": self.pointer_archive_sha256_at_seal,
                "pointer_tolerance_abs": self.pointer_tolerance_abs,
                "require_pointer_archive_identity": self.require_pointer_archive_identity,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> AdmitBar:
        derivation = payload.get("derivation")
        if not isinstance(derivation, dict):
            raise SealContractError("admit_bar carries no derivation block — an undrivable bar is not a bar")
        return cls(
            rule=payload.get("rule", ""),
            net_dS_threshold=payload.get("net_dS_threshold"),
            pointer_axis=derivation.get("pointer_axis", ""),
            pointer_score_at_seal=derivation.get("pointer_score_at_seal"),
            pointer_archive_sha256_at_seal=derivation.get("pointer_archive_sha256_at_seal", ""),
            pointer_tolerance_abs=float(derivation.get("pointer_tolerance_abs") or 0.0),
            require_pointer_archive_identity=bool(derivation.get("require_pointer_archive_identity", True)),
        )


def canonical_seal_bytes(document: dict) -> bytes:
    """The exact bytes ``seal_sha256`` covers: the document minus its own signature field.

    Sorted keys and tight separators, so two producers that emit the same content emit the
    same digest regardless of key order or whitespace.
    """
    body = {key: value for key, value in document.items() if key != "seal_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_seal_sha256(document: dict) -> str:
    return hashlib.sha256(canonical_seal_bytes(document)).hexdigest()


def _public_smoke_problems(
    block: object,
    *,
    candidate_runtime_dir: Path,
    candidate_archive_path: Path,
    pointer_archive_sha256: str,
) -> tuple[list[str], dict[str, object]]:
    """Re-derive the four public-entrypoint receipt identities and outcomes.

    The receipt is evidence about an executed path, so the validator cannot replay it cheaply;
    it can, however, refuse a vacuous boolean and re-measure every byte identity the execution
    claimed.  Both legs must name the same candidate/control objects, and the control archive
    must be the frontier object against which the seal's admission bar was derived.
    """
    if not isinstance(block, dict):
        return (["public_entrypoint_smoke must be an object"], {})

    problems: list[str] = []
    observed: dict[str, object] = {}
    if block.get("schema") != PUBLIC_ENTRYPOINT_SMOKE_SCHEMA:
        problems.append(
            "public_entrypoint_smoke.schema must be "
            f"{PUBLIC_ENTRYPOINT_SMOKE_SCHEMA!r}, got {block.get('schema')!r}"
        )
    bound = block.get("public_path_probe_seconds")
    if not isinstance(bound, (int, float)) or isinstance(bound, bool) or not (0.0 < float(bound) <= 1800.0):
        problems.append(
            "public_entrypoint_smoke.public_path_probe_seconds must be a positive time bound <= 1800"
        )

    groups = {
        "public_path_probes": ("REACHED_TOKEN_DECODE", None),
        "inflate_sh_smokes": ("REACHED_CUDA_GATE", _CUDA_GATE_EXCEPTION),
    }
    role_identities: dict[str, tuple[str, str, str, str]] = {}
    for group_name, (wanted_outcome, wanted_exception) in groups.items():
        group = block.get(group_name)
        if not isinstance(group, dict):
            problems.append(f"public_entrypoint_smoke.{group_name} must be an object")
            continue
        for role in _SMOKE_ROLES:
            receipt = group.get(role)
            label = f"public_entrypoint_smoke.{group_name}.{role}"
            if not isinstance(receipt, dict):
                problems.append(f"{label} must be an object")
                continue
            required_receipt_fields = {
                "outcome",
                "seconds",
                "exception_class",
                "exception_message",
                "runtime_path",
                "tree_sha256",
                "archive_path",
                "archive_sha256",
            }
            if wanted_exception is not None:
                required_receipt_fields.add("returncode")
            for missing in sorted(required_receipt_fields - set(receipt)):
                problems.append(f"{label} is missing required field {missing!r}")
            outcome = receipt.get("outcome")
            if outcome != wanted_outcome:
                problems.append(f"{label}.outcome must be {wanted_outcome!r}, got {outcome!r}")
            seconds = receipt.get("seconds")
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(float(seconds))
                or float(seconds) < 0.0
            ):
                problems.append(f"{label}.seconds must be a non-negative wall-clock measurement")
            elif isinstance(bound, (int, float)) and float(seconds) > float(bound):
                problems.append(
                    f"{label}.seconds {float(seconds):.6g} exceeds the declared bound {float(bound):.6g}"
                )

            exception_class = receipt.get("exception_class")
            exception_message = str(receipt.get("exception_message") or "")
            if wanted_exception is None:
                if exception_class not in (None, ""):
                    problems.append(
                        f"{label}.exception_class must be null because token decode was reached without exception"
                    )
                if exception_message:
                    problems.append(f"{label}.exception_message must be empty on the no-exception leg")
            else:
                if exception_class != wanted_exception:
                    problems.append(
                        f"{label}.exception_class must be the CUDA gate's {wanted_exception!r}, "
                        f"got {exception_class!r}"
                    )
                if _CUDA_GATE_MESSAGE not in exception_message:
                    problems.append(
                        f"{label}.exception_message does not identify the CUDA gate "
                        f"({_CUDA_GATE_MESSAGE!r})"
                    )
                returncode = receipt.get("returncode")
                if not isinstance(returncode, int) or isinstance(returncode, bool) or returncode == 0:
                    problems.append(f"{label}.returncode must be the shell's non-zero CUDA-gate exit status")

            runtime_path = Path(str(receipt.get("runtime_path") or "")).resolve()
            archive_path = Path(str(receipt.get("archive_path") or "")).resolve()
            tree_sha = str(receipt.get("tree_sha256") or "").lower()
            definition = receipt.get("digest_definition")
            if definition is not None and definition != "tac.candidate_seal.measure_runtime_digest":
                problems.append(f"{label}.digest_definition does not identify the seal digest")
            archive_sha = str(receipt.get("archive_sha256") or "").lower()
            if not runtime_path.is_dir():
                problems.append(f"{label}.runtime_path is not a directory: {runtime_path}")
            if not archive_path.is_file():
                problems.append(f"{label}.archive_path is not a file: {archive_path}")
            identity = (str(runtime_path), tree_sha, str(archive_path), archive_sha)
            previous = role_identities.setdefault(role, identity)
            if previous != identity:
                problems.append(
                    f"{label} names a different {role} tree/archive than the other smoke leg"
                )
            if runtime_path.is_dir():
                measured_tree = measure_runtime_digest(runtime_path)
                if tree_sha != measured_tree.sha256:
                    problems.append(
                        f"{label}.tree_sha256 drifted: receipt {tree_sha[:16]}…, "
                        f"disk {measured_tree.sha256[:16]}…"
                    )
                observed[f"{group_name}.{role}.runtime"] = measured_tree.to_dict()
            if archive_path.is_file():
                measured_archive = measure_archive_identity(archive_path)
                if archive_sha != measured_archive.sha256:
                    problems.append(
                        f"{label}.archive_sha256 drifted: receipt {archive_sha[:16]}…, "
                        f"disk {measured_archive.sha256[:16]}…"
                    )
                observed[f"{group_name}.{role}.archive"] = measured_archive.to_dict()

    candidate_identity = role_identities.get("candidate")
    if candidate_identity is not None:
        expected_candidate = (
            str(candidate_runtime_dir.resolve()),
            measure_runtime_digest(candidate_runtime_dir).sha256,
            str(candidate_archive_path.resolve()),
            measure_archive_identity(candidate_archive_path).sha256,
        )
        if candidate_identity != expected_candidate:
            problems.append(
                "public_entrypoint_smoke candidate receipts do not name the runtime/archive being sealed"
            )
    frontier_identity = role_identities.get("frontier")
    if frontier_identity is not None and frontier_identity[3] != str(pointer_archive_sha256).lower():
        problems.append(
            "public_entrypoint_smoke frontier archive does not match "
            "admit_bar.derivation.pointer_archive_sha256_at_seal"
        )
    return problems, observed


def build_seal(
    *,
    candidate_id: str,
    runtime_dir: Path,
    archive_path: Path | None = None,
    axis: str = "contest_cuda",
    admit_bar: AdmitBar,
    public_entrypoint_smoke: dict,
    decode_wall_clock: dict | None = None,
    receiver_relative_paths: tuple[str, ...] = (DEFAULT_RECEIVER_NAME, "inflate.sh"),
    archive_member_name: str = "",
    retained_payload_paths: tuple[str, ...] = (),
    falsifiers: tuple[str, ...] = (),
    sealed_by: str = "",
    sealed_at_utc: str = "",
    notes: str = "",
) -> dict:
    """Build a seal by MEASURING every pin from disk.  Nothing here is hand-typed.

    A producer that accepted hand-typed shas would reproduce the exact failure the seal
    exists to stop, so this function has no parameter for one.  Callers who hold an expected
    value assert it against the returned document.
    """
    runtime_dir = Path(runtime_dir).resolve()
    archive_path = Path(archive_path).resolve() if archive_path else runtime_dir / DEFAULT_ARCHIVE_NAME
    if axis not in SEAL_AXES:
        raise SealContractError(f"unknown seal axis {axis!r}; expected one of {list(SEAL_AXES)}")
    if _is_placeholder(candidate_id):
        raise SealContractError(f"candidate_id {candidate_id!r} is a placeholder, not an identity")

    archive = measure_archive_identity(archive_path)
    runtime = measure_runtime_digest(runtime_dir)

    smoke_problems, _ = _public_smoke_problems(
        public_entrypoint_smoke,
        candidate_runtime_dir=runtime_dir,
        candidate_archive_path=archive_path,
        pointer_archive_sha256=admit_bar.pointer_archive_sha256_at_seal,
    )
    if smoke_problems:
        raise SealContractError("public-entrypoint smoke refused: " + "; ".join(smoke_problems))

    from tac.decode_wall_clock import validate_decode_wall_clock

    timing_problems, _ = validate_decode_wall_clock(
        decode_wall_clock, runtime_dir=runtime_dir, archive_path=archive_path,
        pointer_archive_sha256=admit_bar.pointer_archive_sha256_at_seal,
    )
    if timing_problems:
        raise SealContractError("decode wall-clock refused: " + "; ".join(timing_problems))

    # Older producers predate digest naming. Validate their bytes first, then name the
    # verified algorithm on a private copy; never mutate a retained input receipt.
    public_entrypoint_smoke = json.loads(json.dumps(public_entrypoint_smoke))
    for group in ("public_path_probes", "inflate_sh_smokes"):
        for role in ("candidate", "frontier"):
            receipt = public_entrypoint_smoke[group][role]
            receipt["digest_definition"] = "tac.candidate_seal.measure_runtime_digest"

    receivers: list[dict[str, object]] = []
    file_map = runtime.file_map()
    for rel in receiver_relative_paths:
        entry = file_map.get(rel)
        if entry is None:
            # Silence would let a caller believe a file is pinned when it is not.
            raise SealContractError(
                f"cannot pin receiver {rel!r}: it is not in the shipped file set of {runtime_dir}. "
                f"Shipped: {sorted(file_map)[:12]}"
            )
        receivers.append({"relative_path": rel, "bytes": entry[0], "sha256": entry[1]})

    document: dict[str, object] = {
        "schema": SEAL_SCHEMA,
        "candidate_id": candidate_id,
        "sealed_at_utc": sealed_at_utc or _utc_now(),
        "sealed_by": sealed_by or "unattributed",
        "axis": axis,
        "archive": {"path": str(archive_path), "sha256": archive.sha256, "bytes": archive.bytes},
        "runtime": {"path": str(runtime_dir), **runtime.to_dict()},
        "receiver_pins": receivers,
        "admit_bar": admit_bar.to_dict(),
        "public_entrypoint_smoke": public_entrypoint_smoke,
        "decode_wall_clock": json.loads(json.dumps(decode_wall_clock)),
        "retained_payload_paths": [str(p) for p in retained_payload_paths],
        "falsifiers": list(falsifiers),
        "notes": notes,
    }
    if archive_member_name:
        member_sha, member_bytes = read_archive_member_identity(archive_path, archive_member_name)
        document["archive_member"] = {
            "name": archive_member_name,
            "sha256": member_sha,
            "bytes": member_bytes,
        }
    document["seal_sha256"] = compute_seal_sha256(document)
    return document


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def write_seal(document: dict, path: Path) -> Path:
    """Write a seal atomically.  A half-written seal is a seal that lies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".seal.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_seal(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SealContractError(f"seal not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SealContractError(f"seal {path} is not parsable JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SealContractError(f"seal {path} is not a JSON object")
    return document


@dataclass(frozen=True)
class SealValidation:
    """The verdict of re-verifying one seal against disk at consumption time."""

    verdict: str
    seal_path: Path
    candidate_id: str = ""
    axis: str = ""
    problems: tuple[str, ...] = field(default_factory=tuple)
    observed: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict == SEAL_VALID

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "seal_path": str(self.seal_path),
            "candidate_id": self.candidate_id,
            "axis": self.axis,
            "problems": list(self.problems),
            "observed": self.observed,
        }

    def summary(self) -> str:
        if self.ok:
            return (
                f"SEAL VALID: {self.candidate_id} [{self.axis}] — archive, runtime digest, "
                f"receiver pins and admit bar all re-derived from disk and agree"
            )
        return f"SEAL {self.verdict}: " + "; ".join(self.problems)


_REQUIRED_SEAL_FIELDS = ("schema", "candidate_id", "axis", "archive", "runtime", "receiver_pins", "admit_bar")


def validate_seal(
    seal_path: Path,
    *,
    pointer_path: Path | None = None,
    check_pointer: bool = True,
    allow_missing_public_smoke: bool = False,
    require_decode_wall_clock: bool = False,
) -> SealValidation:
    """Re-verify EVERY pin against disk.  Fail-closed with a typed reason.

    The order is deliberate and cheapest-first, and each stage returns rather than
    accumulating across layers: a tampered seal must not also be reported as a sha drift,
    because the drift would be measured against a value the tamperer chose.
    """
    seal_path = Path(seal_path)
    try:
        document = load_seal(seal_path)
    except SealContractError as exc:
        return SealValidation(verdict=SEAL_SCHEMA_VIOLATION, seal_path=seal_path, problems=(str(exc),))

    if document.get("schema") == PREFIRE_INTENT_SCHEMA:
        return SealValidation(verdict="PREFIRE_INTENT_SCHEMA_REFUSED", seal_path=seal_path,
                              problems=("candidate_prefire_intent.v1 is not a completed seal",))
    if document.get("schema") == SEAL_SCHEMA_V3:
        try:
            _pf_validate_completed_seal(document, pointer_path=pointer_path)
        except (SealContractError, KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
            return SealValidation(verdict=getattr(exc, "code", "FIRST_MEASUREMENT_RESULT_REFUSED"),
                                  seal_path=seal_path, problems=(str(exc),))

    candidate_id = str(document.get("candidate_id") or "")
    axis = str(document.get("axis") or "")

    # ---- 1. schema ------------------------------------------------------------------
    # Types are checked here, not merely key presence.  A seal whose ``archive`` is a string
    # instead of an object would otherwise sail past the placeholder scan (which skips
    # non-objects) and reach the drift comparison, where ``.get`` on a ``str`` raises
    # AttributeError INTO the fire path — a crash where a refusal belongs, and a traceback a
    # reader can mistake for a tooling bug rather than a bad seal.
    problems = []
    for name in _REQUIRED_SEAL_FIELDS:
        if name not in document:
            problems.append(f"missing required field {name!r}")
    for name in ("archive", "runtime", "admit_bar"):
        if name in document and not isinstance(document[name], dict):
            problems.append(f"field {name!r} must be an object, got {type(document[name]).__name__}")
    if "receiver_pins" in document and not isinstance(document["receiver_pins"], list):
        problems.append(f"field 'receiver_pins' must be a list, got {type(document['receiver_pins']).__name__}")
    for name in ("retained_payload_paths", "falsifiers"):
        if name in document and not isinstance(document[name], list):
            problems.append(f"field {name!r} must be a list, got {type(document[name]).__name__}")
    if "schema" in document and document["schema"] not in (SEAL_SCHEMA_V1, SEAL_SCHEMA, SEAL_SCHEMA_V3):
        # No ``None`` escape hatch: an unversioned seal is one this validator cannot claim to
        # understand, and claiming to is how a v2 document gets validated by v1 rules.
        problems.append(f"unknown seal schema {document.get('schema')!r}; this validator speaks {SEAL_SCHEMA}")
    if axis and axis not in SEAL_AXES:
        problems.append(f"unknown axis {axis!r}; expected one of {list(SEAL_AXES)}")
    runtime_block = document.get("runtime")
    if isinstance(runtime_block, dict) and runtime_block.get("digest_definition") not in (
        None, "tac.candidate_seal.measure_runtime_digest",
    ):
        problems.append("runtime.digest_definition does not identify the seal digest")
    if problems:
        return SealValidation(
            verdict=SEAL_SCHEMA_VIOLATION,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=tuple(problems),
        )

    # ---- 2. placeholders --------------------------------------------------------------
    # Run BEFORE the self-signature check: a placeholder seal is honestly signed garbage,
    # and reporting it as tampering would send the reader hunting for an attacker.
    placeholders = _find_placeholder_pins(document)
    if placeholders:
        return SealValidation(
            verdict=SEAL_PLACEHOLDER_PIN,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=tuple(placeholders),
        )

    # ---- 3. self-signature -------------------------------------------------------------
    declared = str(document.get("seal_sha256") or "")
    recomputed = compute_seal_sha256(document)
    if not declared:
        return SealValidation(
            verdict=SEAL_SCHEMA_VIOLATION,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=("seal carries no seal_sha256 — an unsigned seal cannot detect its own edit",),
        )
    if declared.lower() != recomputed:
        return SealValidation(
            verdict=SEAL_TAMPERED,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(
                f"seal_sha256 declares {declared[:16]}… but the document hashes to {recomputed[:16]}… "
                "— this seal was edited after it was signed",
            ),
            observed={"declared_seal_sha256": declared, "recomputed_seal_sha256": recomputed},
        )

    if "public_entrypoint_smoke" not in document and not allow_missing_public_smoke:
        return SealValidation(
            verdict=SEAL_PUBLIC_SMOKE_MISSING,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(
                "public-entrypoint rule chain: candidate + frontier control -> "
                "f26_inflate.inflate_archive REACHED_TOKEN_DECODE without exception -> "
                "bash inflate.sh REACHED_CUDA_GATE with the CUDA RuntimeError; "
                "seal lacks required field 'public_entrypoint_smoke'",
            ),
        )

    observed: dict[str, object] = {}

    # ---- 4. archive bytes ---------------------------------------------------------------
    archive_block = document["archive"]
    archive_path = Path(str(archive_block.get("path")))
    if not archive_path.is_file():
        return SealValidation(
            verdict=SEAL_FILE_MISSING,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(f"sealed archive is gone: {archive_path}",),
        )
    measured = measure_archive_identity(archive_path)
    observed["archive"] = measured.to_dict()
    if int(archive_block.get("bytes", -1)) != measured.bytes:
        return SealValidation(
            verdict=SEAL_BYTE_DRIFT,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(
                f"archive size drifted: sealed {archive_block.get('bytes'):,} B, on disk {measured.bytes:,} B",
            ),
            observed=observed,
        )
    if str(archive_block.get("sha256", "")).lower() != measured.sha256:
        return SealValidation(
            verdict=SEAL_SHA_DRIFT,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(
                f"archive sha drifted: sealed {str(archive_block.get('sha256'))[:16]}…, "
                f"on disk {measured.sha256[:16]}… (same size, different bytes)",
            ),
            observed=observed,
        )

    member_block = document.get("archive_member")
    if isinstance(member_block, dict):
        try:
            member_sha, member_bytes = read_archive_member_identity(archive_path, str(member_block.get("name")))
        except SealContractError as exc:
            return SealValidation(
                verdict=SEAL_SHA_DRIFT,
                seal_path=seal_path,
                candidate_id=candidate_id,
                axis=axis,
                problems=(str(exc),),
                observed=observed,
            )
        observed["archive_member"] = {"sha256": member_sha, "bytes": member_bytes}
        if str(member_block.get("sha256", "")).lower() != member_sha or int(
            member_block.get("bytes", -1)
        ) != member_bytes:
            return SealValidation(
                verdict=SEAL_SHA_DRIFT,
                seal_path=seal_path,
                candidate_id=candidate_id,
                axis=axis,
                problems=(
                    f"archive member {member_block.get('name')!r} drifted: sealed "
                    f"{str(member_block.get('sha256'))[:16]}…/{member_block.get('bytes')} B, on disk "
                    f"{member_sha[:16]}…/{member_bytes} B",
                ),
                observed=observed,
            )

    # ---- 5. runtime tree ----------------------------------------------------------------
    runtime_block = document["runtime"]
    runtime_dir = Path(str(runtime_block.get("path")))
    if not runtime_dir.is_dir():
        return SealValidation(
            verdict=SEAL_FILE_MISSING,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(f"sealed runtime tree is gone: {runtime_dir}",),
            observed=observed,
        )
    runtime_now = measure_runtime_digest(runtime_dir)
    observed["runtime"] = runtime_now.to_dict()

    # Receiver pins first: they can NAME the drifted file, and a tree-level digest mismatch
    # with no per-file detail is a refusal the reader cannot act on.
    file_map = runtime_now.file_map()
    receiver_problems: list[str] = []
    for pin in document["receiver_pins"]:
        rel = str(pin.get("relative_path"))
        entry = file_map.get(rel)
        if entry is None:
            receiver_problems.append(f"pinned receiver file {rel!r} is missing from the runtime tree")
            continue
        size_now, sha_now = entry
        if str(pin.get("sha256", "")).lower() != sha_now:
            receiver_problems.append(
                f"receiver {rel!r} drifted: sealed {str(pin.get('sha256'))[:16]}…/{pin.get('bytes')} B, "
                f"on disk {sha_now[:16]}…/{size_now} B"
            )
    if receiver_problems:
        return SealValidation(
            verdict=SEAL_RUNTIME_DRIFT,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=tuple(receiver_problems),
            observed=observed,
        )

    if str(runtime_block.get("sha256", "")).lower() != runtime_now.sha256:
        sealed_count = runtime_block.get("file_count")
        return SealValidation(
            verdict=SEAL_RUNTIME_DRIFT,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(
                f"runtime FILES digest drifted: sealed {str(runtime_block.get('sha256'))[:16]}… "
                f"({sealed_count} files, {runtime_block.get('total_bytes')} B), on disk "
                f"{runtime_now.sha256[:16]}… ({runtime_now.file_count} files, {runtime_now.total_bytes} B). "
                "Every pinned receiver still matches, so the change is in an unpinned shipped file.",
            ),
            observed=observed,
        )

    # ---- 5b. the receiver must name THIS archive ------------------------------------------
    # The tree is proven intact and the archive proven byte-identical to the seal.  Neither
    # fact answers the question a reader actually delegates to this verdict: can this tree
    # DECODE the archive staged beside it?  ``inflate.py`` carries its own ARCHIVE_SHA256 /
    # ARCHIVE_BYTES and asserts them at decode; a tree copied from another candidate keeps the
    # DONOR's pin and refuses its own archive.  Both sides can be individually pristine while
    # disagreeing — which is exactly the state that produced SEAL_VALID on an undecodable
    # packet on 2026-08-18 (iv1: receiver pinned 35ac2b9b/181,161 B, archive was
    # 49bb833e/181,475 B), caught only downstream at dispatch time.
    #
    # This lives INSIDE validate_seal, not beside it, because a seal is the object a reader
    # trusts INSTEAD of re-checking.  Every consumer that behaves correctly — skipping its own
    # verification because SEAL_VALID was returned — inherits whatever this function fails to
    # look at.  Defense-in-depth downstream is not a substitute: a seal can be read by things
    # that never dispatch.
    pin_state = check_pin_consistency(runtime_dir, archive_path=archive_path)
    if pin_state.verdict == MISMATCH:
        return SealValidation(
            verdict=SEAL_RECEIVER_PIN_MISMATCH,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=(pin_state.summary(),),
            observed=observed,
        )

    smoke_block = document.get("public_entrypoint_smoke")
    if smoke_block is not None:
        smoke_problems, smoke_observed = _public_smoke_problems(
            smoke_block,
            candidate_runtime_dir=runtime_dir,
            candidate_archive_path=archive_path,
            pointer_archive_sha256=AdmitBar.from_dict(document["admit_bar"]).pointer_archive_sha256_at_seal,
        )
        observed["public_entrypoint_smoke"] = smoke_observed
        if smoke_problems:
            return SealValidation(
                verdict=SEAL_PUBLIC_SMOKE_INVALID,
                seal_path=seal_path,
                candidate_id=candidate_id,
                axis=axis,
                problems=tuple(smoke_problems),
                observed=observed,
            )
    elif allow_missing_public_smoke:
        observed["public_entrypoint_smoke"] = {"missing_allowed_by_consumer": True}

    timing_block = document.get("decode_wall_clock")
    if timing_block is None:
        observed["decode_wall_clock"] = "absent"
        if require_decode_wall_clock or document["schema"] in (SEAL_SCHEMA, SEAL_SCHEMA_V3):
            return SealValidation(
                verdict=SEAL_DECODE_WALL_CLOCK_MISSING, seal_path=seal_path,
                candidate_id=candidate_id, axis=axis,
                problems=("seal lacks required decode_wall_clock measurement",), observed=observed,
            )
    else:
        from tac.decode_wall_clock import validate_decode_wall_clock

        timing_problems, timing_observed = validate_decode_wall_clock(
            timing_block, runtime_dir=runtime_dir, archive_path=archive_path,
            pointer_archive_sha256=AdmitBar.from_dict(document["admit_bar"]).pointer_archive_sha256_at_seal,
        )
        observed["decode_wall_clock"] = timing_observed
        if timing_problems:
            return SealValidation(
                verdict=SEAL_DECODE_WALL_CLOCK_INVALID, seal_path=seal_path,
                candidate_id=candidate_id, axis=axis,
                problems=tuple(timing_problems), observed=observed,
            )

    # ---- 6. retained payload custody -----------------------------------------------------
    missing_payload = []
    for p in document.get("retained_payload_paths", []):
        try:
            resolve_artifact(p)
        except MovedArtifactError as exc:
            missing_payload.append(f"{p}: {exc}")
    if missing_payload:
        return SealValidation(
            verdict=SEAL_FILE_MISSING,
            seal_path=seal_path,
            candidate_id=candidate_id,
            axis=axis,
            problems=tuple(f"retained payload custody path is gone: {p}" for p in missing_payload),
            observed=observed,
        )

    # ---- 7. admit-bar re-derivation --------------------------------------------------------
    bar = AdmitBar.from_dict(document["admit_bar"])
    if check_pointer:
        try:
            live = read_pointer_state(pointer_path=pointer_path, axis=bar.pointer_axis)
        except SealContractError as exc:
            return SealValidation(
                verdict=SEAL_BAR_DRIFT,
                seal_path=seal_path,
                candidate_id=candidate_id,
                axis=axis,
                problems=(f"cannot re-derive the admit bar: {exc}",),
                observed=observed,
            )
        observed["pointer"] = live
        moved = abs(float(live["pointer_score"]) - float(bar.pointer_score_at_seal))
        bar_problems: list[str] = []
        if moved > bar.pointer_tolerance_abs:
            bar_problems.append(
                f"the admission baseline moved {moved:.3e} on axis {bar.pointer_axis!r} "
                f"(sealed {bar.pointer_score_at_seal:.8f} -> live {float(live['pointer_score']):.8f}), "
                f"beyond the declared tolerance {bar.pointer_tolerance_abs:.3e}. The bar "
                f"{bar.net_dS_threshold} was derived against the OLD baseline; re-seal it."
            )
        if bar.require_pointer_archive_identity and str(live["pointer_archive_sha256"]).lower() != str(
            bar.pointer_archive_sha256_at_seal
        ).lower():
            bar_problems.append(
                f"the frontier now points at a DIFFERENT candidate "
                f"(sealed {str(bar.pointer_archive_sha256_at_seal)[:16]}…, live "
                f"{str(live['pointer_archive_sha256'])[:16]}…): a delta is unanchored without its baseline"
            )
        if bar_problems:
            return SealValidation(
                verdict=SEAL_BAR_DRIFT,
                seal_path=seal_path,
                candidate_id=candidate_id,
                axis=axis,
                problems=tuple(bar_problems),
                observed=observed,
            )

    return SealValidation(
        verdict=SEAL_VALID,
        seal_path=seal_path,
        candidate_id=candidate_id,
        axis=axis,
        observed=observed,
    )


def _find_placeholder_pins(document: dict) -> list[str]:
    """Report every field that stands in for a value instead of being one."""
    problems: list[str] = []

    def _require_real(value: object, label: str, *, sha: bool = False) -> None:
        if value is None:
            problems.append(f"{label} is absent")
        elif _is_placeholder(value):
            problems.append(f"{label} is the placeholder {value!r}, not a value")
        elif sha and not _is_sha256(value):
            problems.append(f"{label} is not a sha256 digest: {str(value)[:24]!r}")

    _require_real(document.get("candidate_id"), "candidate_id")
    _require_real(document.get("seal_sha256"), "seal_sha256", sha=True)

    for block_name, sha_field in (("archive", "sha256"), ("runtime", "sha256"), ("archive_member", "sha256")):
        block = document.get(block_name)
        if not isinstance(block, dict):
            continue
        _require_real(block.get(sha_field), f"{block_name}.{sha_field}", sha=True)
        if block_name in ("archive", "runtime"):
            _require_real(block.get("path"), f"{block_name}.path")
        size = block.get("bytes") if block_name != "runtime" else block.get("total_bytes")
        if not isinstance(size, int) or size <= 0:
            problems.append(f"{block_name} declares a non-positive byte count {size!r}")

    for index, pin in enumerate(document.get("receiver_pins") or []):
        if not isinstance(pin, dict):
            problems.append(f"receiver_pins[{index}] is not an object")
            continue
        _require_real(pin.get("relative_path"), f"receiver_pins[{index}].relative_path")
        _require_real(pin.get("sha256"), f"receiver_pins[{index}].sha256", sha=True)
    if not document.get("receiver_pins"):
        problems.append("receiver_pins is empty — an unpinned receiver is the ps1u drift class, unguarded")

    bar = document.get("admit_bar")
    if isinstance(bar, dict):
        _require_real(bar.get("rule"), "admit_bar.rule")
        if not isinstance(bar.get("net_dS_threshold"), (int, float)):
            problems.append(f"admit_bar.net_dS_threshold is not a number: {bar.get('net_dS_threshold')!r}")
        derivation = bar.get("derivation")
        if not isinstance(derivation, dict):
            problems.append("admit_bar.derivation is absent — a bar without its inputs cannot be re-derived")
        else:
            _require_real(derivation.get("pointer_axis"), "admit_bar.derivation.pointer_axis")
            _require_real(
                derivation.get("pointer_archive_sha256_at_seal"),
                "admit_bar.derivation.pointer_archive_sha256_at_seal",
                sha=True,
            )
            if not isinstance(derivation.get("pointer_score_at_seal"), (int, float)):
                problems.append("admit_bar.derivation.pointer_score_at_seal is not a number")

    for index, path in enumerate(document.get("retained_payload_paths") or []):
        _require_real(path, f"retained_payload_paths[{index}]")
    for index, item in enumerate(document.get("falsifiers") or []):
        _require_real(item, f"falsifiers[{index}]")

    return problems

# ddm_pr12, with ddm_ffi3 clarifications: a prefire intent is deliberately NOT a seal.
PREFIRE_INTENT_SCHEMA = "candidate_prefire_intent.v1"
PREFIRE_RISK_SCHEMA = "candidate_prefire_timing_risk.v1"
FIRST_MEASUREMENT_AUTHORIZATION_SCHEMA = "candidate_first_measurement_authorization.v1"
SEAL_SCHEMA_V3 = "candidate_seal.v3"
PREFIRE_REFUSAL_CODES = (
    "PREFIRE_INTENT_SCHEMA_REFUSED", "PREFIRE_INTENT_FALSE_AUTHORITY_REFUSED",
    "PREFIRE_CONTRACT_DRIFT_REFUSED", "PREFIRE_IDENTITY_DRIFT_REFUSED",
    "PREFIRE_NON_TIMING_GATE_REFUSED", "PREFIRE_POINTER_DRIFT_REFUSED",
    "PREFIRE_RISK_EVIDENCE_REFUSED", "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED",
    "FIRST_MEASUREMENT_REPLAY_REFUSED", "FIRST_MEASUREMENT_ARGUMENT_REFUSED",
    "FIRST_MEASUREMENT_LANE_REFUSED", "FIRST_MEASUREMENT_TIMEOUT_REFUSED",
    "FIRST_MEASUREMENT_WARM_REFUSED", "FIRST_MEASUREMENT_T4_POLICY_REFUSED",
    "FIRST_MEASUREMENT_RESULT_REFUSED",
)
PREFIRE_DISPATCH_POLICY = {
    "axis": "contest_cuda", "gpu": "T4", "scorer_device": "cuda", "inflate_device": "auto",
    "inflate_timeout_seconds": 1800, "evaluate_timeout_seconds": 1800,
    "modal_function_timeout_seconds": 4800, "poller_deadline_seconds": 5400,
    "n_samples": 600, "cold_required": True, "checkpoint_resume_required": False,
    "token_cache_status_required": "DISABLED", "source_snapshot_required": True,
    "claim_policy": "require_active", "max_paid_dispatches": 1, "currency": "USD",
    "maximum_total_cost_usd_exclusive": 5.0, "cost_preflight_max_age_seconds": 86400,
}
PREFIRE_TOP_FIELDS = {"schema", "state", "candidate_id", "created_at_utc", "created_by", "producer_source_commit", "score_claim", "promotion_eligible", "timing_clearance", "contract", "candidate", "admit_bar", "public_entrypoint_smoke", "evidence", "dispatch_policy", "retained_payload_paths", "falsifiers", "intent_sha256"}
PREFIRE_EVIDENCE_FIELDS = {"candidate_manifest", "manifest_validation", "twin_encode", "archive_parseback", "raw_identity_n600", "literal_census", "retention_manifest", "timing_risk"}
PREFIRE_IMPLEMENTATION_PATHS = (
    "src/tac/candidate_seal.py", "src/tac/decode_wall_clock.py", "src/tac/git_custody_read.py",
    "tools/make_candidate_seal.py", "tools/authorize_candidate_first_measurement.py",
    "tools/fire_modal_auth_eval.py", "experiments/modal_auth_eval.py",
    "tools/modal_harvest_poller.py", "src/tac/deploy/modal/auth_eval.py",
    "src/tac/deploy/modal/single_flight.py", "src/tac/deploy/modal/call_id_ledger.py",
    "src/tac/modal_source_snapshot.py", "tools/launch_detached_process.py",
    "src/tac/checkpoint_maturity.py", "src/tac/deploy/modal/result_json.py",
)
PREFIRE_MEMO_SHA256 = "50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc"
PREFIRE_MEMO = ".omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md"
PREFIRE_FREEZE = ".omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json"
PREFIRE_SINGLE_AXIS_REASON = (
    "candidate receiver declares linux-nvidia-t4 and refuses CPU by design; this authorization is T4-only"
)


class PrefireRefusal(SealContractError):
    """A first-measurement refusal, never timing or score authority."""

    def __init__(self, code: str, detail: str):
        if code not in PREFIRE_REFUSAL_CODES:
            raise ValueError(f"unknown prefire refusal code: {code}")
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict:
        return {"schema": "candidate_first_measurement_refusal.v1", "code": self.code,
                "detail": self.detail, "score_claim": False, "promotion_eligible": False,
                "created_at_utc": _utc_now()}


def _pf_require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise PrefireRefusal(code, detail)


def prefire_digest(document: object, self_field: str = "") -> str:
    """pr12 canonical UTF-8 digest; omit exactly one named self field."""
    body = {k: v for k, v in document.items() if k != self_field} if self_field else document
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def prefire_file_reference(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _pf_read(path: Path, code: str) -> object:
    try:
        def reject_constant(value):
            raise ValueError(f"non-finite JSON constant {value}")
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("non-finite JSON number")
            return number
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant,
                          parse_float=finite_float, object_pairs_hook=unique_object)
    except (OSError, ValueError, TypeError) as exc:
        raise PrefireRefusal(code, f"cannot read JSON {path}: {exc}") from exc


def _pf_ref(ref: object, code: str, *, parse: bool = True) -> object:
    _pf_require(isinstance(ref, dict), code, "reference must be an object")
    _pf_require(isinstance(ref.get("path"), str) and Path(ref["path"]).is_absolute(), code,
                "reference requires an absolute path")
    path = Path(ref["path"])
    _pf_require(type(ref.get("bytes")) is int and ref["bytes"] > 0 and _is_sha256(ref.get("sha256")),
                code, f"invalid byte/hash reference: {path}")
    try:
        _pf_require(path.is_file() and path.stat().st_size == ref["bytes"]
                    and sha256_file(path) == ref["sha256"], code, f"reference bytes/hash differ: {path}")
    except OSError as exc:
        raise PrefireRefusal(code, f"reference unavailable: {path}: {exc}") from exc
    return _pf_read(path, code) if parse else path


def _pf_number(value: object, code: str, label: str, *, positive: bool = False) -> float:
    import math
    _pf_require(type(value) in (int, float) and math.isfinite(value), code, f"{label}: finite number required")
    _pf_require(not positive or value > 0, code, f"{label}: positive number required")
    return float(value)


def _pf_time(value: object, code: str):
    from datetime import datetime
    try:
        _pf_require(isinstance(value, str) and (value.endswith("Z") or value.endswith("+00:00")),
                    code, "timestamp must be RFC3339 UTC")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _pf_require("T" in value and parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
                    code, "timestamp must contain an explicit UTC time")
        return parsed
    except ValueError as exc:
        raise PrefireRefusal(code, "invalid UTC timestamp") from exc


def _pf_git(repo: Path, *args: str) -> bytes:
    """Read committed custody in-process: preflight never spawns a Git subprocess."""
    from tac.git_custody_read import GitCustodyReader
    try:
        reader = GitCustodyReader(repo)
        if args[0] == "rev-parse":
            return reader.resolve(args[1]).encode()
        if args[:3] == ("show", "-s", "--format=%cI"):
            return reader.committed_at(args[3]).encode()
        if args[0] == "show":
            ref, path = args[1].split(":", 1)
            return reader.blob(ref, path)
        if args[:2] == ("merge-base", "--is-ancestor"):
            _pf_require(reader.ancestor(args[2], args[3]), "PREFIRE_CONTRACT_DRIFT_REFUSED", "commit ancestry differs")
            return b""
        raise ValueError("unsupported custody query")
    except Exception as exc:
        raise PrefireRefusal("PREFIRE_CONTRACT_DRIFT_REFUSED", f"Git custody query failed: {args}: {exc}") from exc


def _pf_blob(repo: Path, commit: str, path: Path) -> None:
    code = "PREFIRE_CONTRACT_DRIFT_REFUSED"
    try:
        rel = path.resolve().relative_to(repo.resolve()).as_posix()
        _pf_require(_pf_git(repo, "show", f"{commit}:{rel}") == path.read_bytes(), code,
                    f"live file differs from committed blob: {path}")
    except (OSError, ValueError) as exc:
        raise PrefireRefusal(code, f"file is not in repository custody: {path}") from exc


def _pf_contract(intent: dict, repo: Path, intent_path: Path | None) -> None:
    code = "PREFIRE_CONTRACT_DRIFT_REFUSED"
    contract = intent["contract"]
    _pf_require(isinstance(contract, dict) and set(contract) == {
        "adjudication_memo", "implementation_commit", "implementation_manifest", "implementation_manifest_sha256"},
        code, "contract fields differ")
    commit = contract["implementation_commit"]
    _pf_require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), code, "full implementation commit required")
    _pf_git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    manifest = _pf_ref(contract["implementation_manifest"], code)
    _pf_require(isinstance(manifest, list) and manifest, code, "sorted implementation manifest array required")
    _pf_require(contract["implementation_manifest_sha256"] == contract["implementation_manifest"]["sha256"],
                code, "implementation manifest digest differs")
    paths = [r.get("path") for r in manifest if isinstance(r, dict)]
    _pf_require(len(paths) == len(manifest) and all(isinstance(p, str) for p in paths)
                and paths == sorted(set(paths)) and set(PREFIRE_IMPLEMENTATION_PATHS) <= set(paths),
                code, "implementation manifest omits a consumer or is unsorted")
    for row in manifest:
        path = repo / row["path"]
        _pf_require(not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts,
                    code, "implementation path escapes repository")
        _pf_require(path.is_file() and sha256_file(path) == row.get("sha256"), code, f"implementation drift: {path}")
        _pf_blob(repo, commit, path)
    _pf_ref(contract["adjudication_memo"], code, parse=False)
    _pf_require(contract["adjudication_memo"]["sha256"] == PREFIRE_MEMO_SHA256, code, "normative pr12 memo pin differs")
    _pf_require(Path(contract["adjudication_memo"]["path"]).resolve() == (repo / PREFIRE_MEMO).resolve(),
                code, "wrong adjudication memo")
    _pf_blob(repo, commit, repo / PREFIRE_MEMO)
    committed_at = _pf_git(repo, "show", "-s", "--format=%cI", commit).decode().strip()
    produced = _pf_ref(intent["evidence"]["candidate_manifest"], code)
    _pf_require(_pf_time(committed_at, code) <= _pf_time(produced.get("production_started_at_utc"), code)
                <= _pf_time(intent["created_at_utc"], code), code, "implementation must precede every producer timestamp")
    production = intent["producer_source_commit"]
    _pf_require(isinstance(production, str) and re.fullmatch(r"[0-9a-f]{40}", production), code, "producer source commit malformed")
    _pf_git(repo, "merge-base", "--is-ancestor", commit, production)
    _pf_git(repo, "merge-base", "--is-ancestor", production, "HEAD")
    if intent_path is not None:
        _pf_blob(repo, "HEAD", intent_path)


def _pf_schema(intent: object) -> None:
    code = "PREFIRE_INTENT_SCHEMA_REFUSED"
    _pf_require(isinstance(intent, dict), code, "intent must be an object")
    forbidden = {"decode_wall_clock", "candidate_t4_receipt", "measured_t4_decode_seconds",
                 "projected_t4_decode_seconds", "timing_passed", "candidate_score", "score"}
    def authority_keys(value):
        if isinstance(value, dict):
            return any(k in forbidden or (k in {"score_claim", "promotion_eligible", "timing_clearance"} and v is True)
                       or authority_keys(v) for k, v in value.items())
        if isinstance(value, list):
            return any(authority_keys(v) for v in value)
        return False
    _pf_require(not authority_keys(intent), "PREFIRE_INTENT_FALSE_AUTHORITY_REFUSED",
                "intent carries timing or score authority")
    _pf_require(all(intent.get(k) is False for k in ("score_claim", "promotion_eligible", "timing_clearance")),
                code, "false authority flags must be explicit typed false")
    _pf_require(set(intent) == PREFIRE_TOP_FIELDS and intent["schema"] == PREFIRE_INTENT_SCHEMA
                and intent["state"] == "PREFIRE_FIRST_MEASUREMENT_ONLY", code, "exact v1 schema/state required")
    for key in ("candidate_id", "created_by"):
        _pf_require(isinstance(intent[key], str) and not _is_placeholder(intent[key]), code, f"invalid {key}")
    _pf_time(intent["created_at_utc"], code)
    for key in ("candidate", "admit_bar", "evidence", "contract", "public_entrypoint_smoke", "dispatch_policy"):
        _pf_require(isinstance(intent[key], dict), code, f"{key} must be an object")
    _pf_require(prefire_digest(intent["dispatch_policy"]) == prefire_digest(PREFIRE_DISPATCH_POLICY),
                code, "dispatch policy must equal the fixed v1 policy, including types")
    _pf_require(set(intent["evidence"]) == PREFIRE_EVIDENCE_FIELDS, code, "evidence fields differ")
    _pf_require(set(intent["candidate"]) == {"archive", "runtime", "normalized_receiver", "receiver_pins", "archive_member"},
                code, "candidate fields differ")
    _pf_require(intent.get("intent_sha256") == prefire_digest(intent, "intent_sha256"), code, "canonical intent digest differs")
    for key in ("retained_payload_paths", "falsifiers"):
        _pf_require(isinstance(intent[key], list) and intent[key] and all(isinstance(s, str)
                    and not _is_placeholder(s) for s in intent[key]), code, f"nonempty {key} required")


def _pf_identity(intent: dict) -> tuple[Path, Path]:
    from tac.decode_wall_clock import measure_receiver_digest
    code = "PREFIRE_IDENTITY_DRIFT_REFUSED"
    candidate = intent["candidate"]
    archive = _pf_ref(candidate["archive"], code, parse=False)
    runtime = candidate["runtime"]
    _pf_require(isinstance(runtime, dict) and isinstance(runtime.get("path"), str)
                and Path(runtime["path"]).is_absolute(), code, "absolute runtime directory required")
    root = Path(runtime["path"])
    measured = measure_runtime_digest(root)
    for key, value in {"sha256": measured.sha256, "file_count": measured.file_count,
                       "total_bytes": measured.total_bytes,
                       "digest_definition": "tac.candidate_seal.measure_runtime_digest"}.items():
        _pf_require(type(runtime.get(key)) is type(value) and runtime.get(key) == value, code, f"runtime {key} differs")
    receiver = candidate["normalized_receiver"]
    _pf_require(receiver == {"digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
                            "sha256": measure_receiver_digest(root)}, code, "normalized receiver differs")
    pins = candidate["receiver_pins"]
    _pf_require(isinstance(pins, list) and {p.get("relative_path") for p in pins if isinstance(p, dict)}
                >= {"inflate.py", "inflate.sh"}, code, "both receiver pins required")
    for pin in pins:
        _pf_require(isinstance(pin, dict) and set(pin) == {"relative_path", "bytes", "sha256"}, code, "malformed receiver pin")
        rel = pin["relative_path"]
        _pf_require(rel in measured.file_map() and measured.file_map()[rel] == (pin["bytes"], pin["sha256"]),
                    code, f"receiver pin differs: {rel}")
    _pf_require(check_pin_consistency(root, archive_path=archive).ok, code, "receiver/archive pins disagree")
    import zipfile
    try:
        with zipfile.ZipFile(archive) as zf:
            _pf_require(zf.namelist() and len(zf.namelist()) == len(set(zf.namelist()))
                        and zf.testzip() is None, code, "invalid ZIP/member bytes")
        member = candidate["archive_member"]
        if member is not None:
            _pf_require(isinstance(member, dict) and set(member) == {"name", "sha256", "bytes"}, code, "bad member pin")
            _pf_require(read_archive_member_identity(archive, member["name"]) == (member["sha256"], member["bytes"]),
                        code, "member pin differs")
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise PrefireRefusal(code, f"archive invalid: {exc}") from exc
    return root, archive


def _pf_pointer(intent: dict, pointer_path: Path | None) -> dict:
    code = "PREFIRE_POINTER_DRIFT_REFUSED"
    bar = intent["admit_bar"]
    _pf_require(set(bar) == {"rule", "net_dS_threshold", "pointer_axis", "pointer_score_at_intent",
                "pointer_archive_sha256_at_intent", "pointer_tolerance_abs", "require_pointer_archive_identity",
                "rate_only_precheck"}, code, "admit bar fields differ")
    _pf_require(bar["pointer_axis"] == "contest_cuda" and type(bar["pointer_tolerance_abs"]) in (int, float)
                and bar["pointer_tolerance_abs"] == 0 and bar["require_pointer_archive_identity"] is True,
                code, "zero-tolerance CUDA pointer identity required")
    live = read_pointer_state(pointer_path=pointer_path, axis="contest_cuda")
    _pf_require(_pf_number(bar["pointer_score_at_intent"], code, "pointer score") == live["pointer_score"]
                and bar["pointer_archive_sha256_at_intent"] == live["pointer_archive_sha256"], code, "pointer moved")
    _pf_require(_pf_number(bar["net_dS_threshold"], code, "bar") < 0, code, "negative admit bar required")
    # The pointer archive is re-read through the smoke's already-bound frontier endpoint.
    base_path = Path(intent["public_entrypoint_smoke"]["public_path_probes"]["frontier"]["archive_path"])
    base = measure_archive_identity(base_path)
    _pf_require(base.sha256 == live["pointer_archive_sha256"], code, "frontier archive differs from live pointer")
    ds = 25 * (intent["candidate"]["archive"]["bytes"] - base.bytes) / 37545489
    expected = {"raw_identity_required": True, "normalizer_bytes": 37545489, "derived_net_dS": ds, "passed": True}
    _pf_require(prefire_digest(bar["rate_only_precheck"]) == prefire_digest(expected)
                and ds < bar["net_dS_threshold"], "PREFIRE_NON_TIMING_GATE_REFUSED", "rate-only admission failed")
    return live


def _pf_endpoints(receipt: dict, intent: dict, code: str) -> None:
    candidate = intent["candidate"]
    for key, wanted in {"archive_sha256": candidate["archive"]["sha256"],
                        "runtime_sha256": candidate["runtime"]["sha256"],
                        "receiver_sha256": candidate["normalized_receiver"]["sha256"]}.items():
        _pf_require(receipt.get(key) == wanted, code, f"evidence endpoint differs: {key}")


def _pf_evidence(intent: dict, root: Path, archive: Path) -> None:
    code = "PREFIRE_NON_TIMING_GATE_REFUSED"
    evidence = {k: _pf_ref(v, code) for k, v in intent["evidence"].items() if k != "timing_risk"}
    _pf_require(all(isinstance(v, dict) for v in evidence.values()), code, "evidence receipts must be objects")
    problems, _ = _public_smoke_problems(intent["public_entrypoint_smoke"], candidate_runtime_dir=root,
        candidate_archive_path=archive, pointer_archive_sha256=intent["admit_bar"]["pointer_archive_sha256_at_intent"])
    _pf_require(not problems, code, "; ".join(problems))
    runtime = measure_runtime_digest(root)
    for name in ("candidate_manifest", "manifest_validation", "twin_encode", "archive_parseback",
                 "raw_identity_n600", "literal_census"):
        _pf_endpoints(evidence[name], intent, "PREFIRE_IDENTITY_DRIFT_REFUSED")
    manifest = evidence["candidate_manifest"]
    verification = evidence["manifest_validation"]
    _pf_require(not Path(intent["evidence"]["manifest_validation"]["path"]).is_relative_to(root),
                code, "manifest must be independently verified outside candidate tree")
    rows = manifest.get("files")
    _pf_require(isinstance(rows, list) and rows, code, "manifest files absent")
    actual = runtime.file_map()
    declared = {}
    for row in rows:
        _pf_require(isinstance(row, dict) and set(row) == {"relative_path", "bytes", "sha256"}, code, "bad dependency row")
        rel = row["relative_path"]
        _pf_require(rel not in declared, code, "duplicate dependency row")
        declared[rel] = (row["bytes"], row["sha256"])
    _pf_require(declared == actual and verification.get("all_hashes_passed") is True
                and verification.get("all_runtime_dependencies_listed") is True
                and verification.get("manifest") == intent["evidence"]["candidate_manifest"], code,
                "dependency manifest/verification not complete")
    _pf_require(manifest.get("producer_source_commit") == intent["producer_source_commit"], code,
                "producer source commit differs")
    produced = _pf_time(manifest.get("production_started_at_utc"), code)
    _pf_require(produced <= _pf_time(intent["created_at_utc"], code), code, "production follows intent")
    twin, parseback = evidence["twin_encode"], evidence["archive_parseback"]
    payloads = twin.get("payloads")
    _pf_require(twin.get("n_samples") == 600 and isinstance(payloads, list) and len(payloads) == 2,
                code, "two full-n600 encoded payloads required")
    for ref in payloads:
        _pf_ref(ref, code, parse=False)
    executions = twin.get("executions")
    _pf_require(isinstance(executions, list) and len(executions) == 2, code, "two independent encoder execution receipts required")
    execution_ids = set()
    for ref, payload in zip(executions, payloads, strict=True):
        execution = _pf_ref(ref, code)
        _pf_require(execution.get("n_samples") == 600 and execution.get("completed") is True
                    and execution.get("payload") == payload and isinstance(execution.get("command"), list)
                    and execution["command"] and execution.get("producer_source_commit") == intent["producer_source_commit"],
                    code, "encoder execution is not complete n600 on the frozen source/payload")
        identity = execution.get("execution_id")
        _pf_require(isinstance(identity, str) and not _is_placeholder(identity) and identity not in execution_ids,
                    code, "twin execution identity absent/duplicated")
        execution_ids.add(identity)
    _pf_require(payloads[0]["path"] != payloads[1]["path"]
                and (payloads[0]["bytes"], payloads[0]["sha256"]) == (payloads[1]["bytes"], payloads[1]["sha256"]),
                code, "twin encodes not distinct retained byte-identical payloads")
    member = parseback.get("member")
    _pf_require(isinstance(member, dict), code, "parsed member pin absent")
    measured_member = read_archive_member_identity(archive, member["name"])
    _pf_require(measured_member == (member.get("sha256"), member.get("bytes"))
                == (payloads[0]["sha256"], payloads[0]["bytes"]), code, "selected payload differs from archive member")
    raw = evidence["raw_identity_n600"]
    _pf_require(raw.get("n_samples") == 600 and raw.get("pair_count") == 600
                and raw.get("entrypoint") == "inflate.sh" and raw.get("checkpoint_resume") is False
                and raw.get("token_cache_status") == "DISABLED", code, "full cold public raw identity required")
    from tac.decode_wall_clock import _cold_public_report
    raw_log = _pf_ref(raw.get("candidate_public_stdout"), code, parse=False)
    try:
        _cold_public_report({"artifacts": {"contest_auth_eval.stdout.log": raw_log.read_text()}})
    except SealContractError as exc:
        raise PrefireRefusal(code, f"public raw cold proof refused: {exc}") from exc
    command = raw.get("command")
    _pf_require(isinstance(command, list) and len(command) >= 2 and command[:2] == ["bash", str(root / "inflate.sh")],
                code, "raw identity did not bind literal public inflate.sh argv")
    for key in ("candidate_raw", "pointer_raw"):
        _pf_ref(raw.get(key), code, parse=False)
        _pf_require(raw[key]["bytes"] == 3662409600, code, "n600 raw byte count differs")
    _pf_require(raw["candidate_raw"]["sha256"] == raw["pointer_raw"]["sha256"]
                and raw.get("pointer_archive_sha256") == intent["admit_bar"]["pointer_archive_sha256_at_intent"],
                code, "raw bytes/pointer identity differ")
    census = evidence["literal_census"]
    _pf_require(census.get("verdict") == "CLEAR" and census.get("rule") == 118
                and census.get("complete") is True, code, "complete rule-118 literal census required")
    census_rows = census.get("files")
    _pf_require(isinstance(census_rows, list) and census_rows, code, "literal census coverage absent")
    _pf_require({r.get("relative_path"): (r.get("bytes"), r.get("sha256")) for r in census_rows
                 if isinstance(r, dict)} == actual and len(census_rows) == len(actual), code,
                 "new receiver/dependency not covered by literal census")
    retention = evidence["retention_manifest"]
    retained = retention.get("payloads")
    _pf_require(isinstance(retained, list) and retained, code, "retention manifest payloads absent")
    paths = set()
    for ref in retained:
        _pf_ref(ref, code, parse=False)
        _pf_require(ref["path"] not in paths, code, "duplicate retained payload")
        paths.add(ref["path"])
    required = {str(archive), *(r["path"] for r in payloads), raw["candidate_raw"]["path"],
                intent["evidence"]["archive_parseback"]["path"], *intent["retained_payload_paths"]}
    _pf_require(required <= paths, code, "retention missing archive/twins/raw/parseback/declared payload")


def prefire_receiver_rows(root: Path) -> list[tuple[str, int, str]]:
    """Materialize the exact existing receiver digest's normalized path rows, without writes."""
    import ast

    from tac.decode_wall_clock import measure_receiver_digest
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "archive.zip" or runtime_digest_skip_reason(rel):
            continue
        data = path.read_bytes()
        if rel == "inflate.py":
            tree = ast.parse(data)
            lines = data.splitlines(keepends=True)
            offsets = [sum(map(len, lines[:i])) for i in range(len(lines))]
            edits = []
            for node in tree.body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id in {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}):
                    value = node.value
                    edits.append((offsets[value.lineno - 1] + value.col_offset,
                                  offsets[value.end_lineno - 1] + value.end_col_offset))
            for start, end in sorted(edits, reverse=True):
                data = data[:start] + b"<ARCHIVE_PIN>" + data[end:]
        rows.append((rel, len(data), hashlib.sha256(data).hexdigest()))
    _pf_require(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
                == measure_receiver_digest(root), "PREFIRE_RISK_EVIDENCE_REFUSED", "normalized row parity failed")
    return rows


def validate_prefire_risk(risk_ref: dict, intent: dict, *, repo: Path) -> dict:
    from tac.decode_wall_clock import measure_receiver_digest, validate_decode_wall_clock
    code = "PREFIRE_RISK_EVIDENCE_REFUSED"
    risk = _pf_ref(risk_ref, code)
    _pf_require(isinstance(risk, dict) and set(risk) == {"schema", "mode", "authority", "timing_clearance", "source_t4_leg", "source_receiver", "candidate_receiver", "diagnostic_reference_receiver", "receiver_delta_manifest", "base_local_diagnostic", "candidate_local_diagnostics", "calculation", "score_claim", "risk_sha256"},
        code, "risk shape differs")
    _pf_require(risk["schema"] == PREFIRE_RISK_SCHEMA and risk["mode"] == "completed_t4_receiver_delta"
                and all(risk[k] is False for k in ("authority", "timing_clearance", "score_claim"))
                and risk["risk_sha256"] == prefire_digest(risk, "risk_sha256"), code, "risk type/digest differs")
    leg = _pf_ref(risk["source_t4_leg"], code)
    _pf_require(isinstance(leg, dict) and leg.get("mode") == "t4_direct", code, "completed source t4_direct required")
    problems, _ = validate_decode_wall_clock(leg, runtime_dir=Path(leg["runtime_dir"]), archive_path=Path(leg["archive_path"]))
    _pf_require(not problems, code, "; ".join(problems))
    source = risk["source_receiver"].get("sha256")
    candidate = intent["candidate"]["normalized_receiver"]["sha256"]
    diagnostic_root = Path(risk["diagnostic_reference_receiver"]["path"])
    _pf_require(source == leg.get("receiver_sha256") and risk["candidate_receiver"] == {"sha256": candidate}
                and risk["diagnostic_reference_receiver"].get("sha256") == candidate
                and measure_receiver_digest(diagnostic_root) == candidate, code, "receiver risk endpoints differ")
    delta = _pf_ref(risk["receiver_delta_manifest"], code)
    source_rows = prefire_receiver_rows(Path(leg["runtime_dir"]))
    candidate_rows = prefire_receiver_rows(Path(intent["candidate"]["runtime"]["path"]))
    smap, cmap = {r[0]: list(r[1:]) for r in source_rows}, {r[0]: list(r[1:]) for r in candidate_rows}
    wanted = [{"relative_path": p, "source": smap.get(p), "candidate": cmap.get(p)} for p in sorted(smap.keys() | cmap.keys())]
    _pf_require(delta.get("files") == wanted and delta.get("source_receiver_sha256") == source
                and delta.get("candidate_receiver_sha256") == candidate, code, "complete normalized receiver delta differs")
    base = risk["base_local_diagnostic"]
    candidates = risk["candidate_local_diagnostics"]
    _pf_require(isinstance(candidates, list) and candidates, code, "candidate diagnostics missing")
    is_rlc2 = "rlc2" in intent["candidate_id"].lower()
    for ref in [base, *candidates]:
        doc = _pf_ref(ref, code)
        wall = _pf_number(ref.get("wall_seconds"), code, "diagnostic seconds", positive=True)
        _pf_require(ref.get("authority") is False and ref.get("actual_verdict") == "REFUSED"
                    and doc.get("wall_seconds") == wall and doc.get("score_claim") is False,
                    code, "diagnostic must retain actual refused verdict and measured wall")
        _pf_require(doc.get("receiver_sha256") == (source if ref is base else candidate), code, "diagnostic receiver differs")
        if not is_rlc2:
            _pf_require(doc.get("actual_verdict") == "REFUSED", code, "diagnostic's retained actual verdict is not REFUSED")
        if ref is not base:
            _pf_require(ref.get("cold") is True and ref.get("n_samples") == 600 and doc.get("cold_start") is True
                        and doc.get("checkpoint_resume") is False and doc.get("frames") == list(range(600)),
                        code, "candidate diagnostic must be cold n600")
    ceiling = max(ref["wall_seconds"] for ref in candidates)
    fraction = max(0, ceiling / base["wall_seconds"] - 1)
    seconds = leg["measured_t4_decode_seconds"]
    projection = seconds * (1 + fraction)
    calculation = {"candidate_local_ceiling_seconds": ceiling, "local_cost_fraction_upper": fraction,
                   "source_t4_seconds": seconds, "t4_risk_ceiling_seconds": projection,
                   "policy_limit_seconds": 1260.0, "hard_timeout_seconds": 1800.0, "passed": True}
    for key in calculation:
        if key != "passed":
            _pf_number(risk["calculation"].get(key), code, key)
    _pf_require(risk["calculation"].get("passed") is True and risk["calculation"] == calculation and projection <= 1260.0 < 1800.0,
                code, "risk arithmetic/1260-second policy differs")
    # RLC2 is the specifically reviewed old-rule diagnostic chain, not an unknown-receiver fallback.
    summary_path = repo / ".omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json"
    if is_rlc2:
        _pf_require(risk["source_t4_leg"]["sha256"] == "ed929b24cc876bf8ffabc3004b856decbb5e73d0fee13c1d3c87d91d659f9521"
                    and risk["source_t4_leg"]["bytes"] == 1553
                    and candidate == "b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d"
                    and summary_path.stat().st_size == 3064
                    and sha256_file(summary_path) == "a8d1e1a781a0c2f80962591288dbcc4ed023113e31766baaec368db8ef75c4ed",
                    code, "RLC2 pinned historical chain differs")
        summary = _pf_read(summary_path, code)
        for ref in candidates:
            matches = [row for row in summary["runs"] if row.get("local_receipt") == ref["path"]]
            _pf_require(len(matches) == 1 and matches[0]["wall_seconds"] == ref["wall_seconds"]
                        and matches[0].get("verdict", "").startswith("REFUSED"), code,
                        "RLC2 diagnostic differs from its retained actual old-rule verdict")
        _pf_require(base["wall_seconds"] == 797.1459791249945 and ceiling == 831.502915124991,
                    code, "RLC2 diagnostics differ from reviewed chain")
    return risk


def validate_prefire_intent(path: Path, *, require_committed: bool = True,
                            repo: Path | None = None, pointer_path: Path | None = None) -> dict:
    """Read every non-timing gate from disk; raise one typed refusal on any unknown state."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    code = "PREFIRE_INTENT_SCHEMA_REFUSED"
    try:
        intent = _pf_read(path, code)
        _pf_schema(intent)
        code = "PREFIRE_CONTRACT_DRIFT_REFUSED"
        _pf_contract(intent, repo, Path(path) if require_committed else None)
        code = "PREFIRE_IDENTITY_DRIFT_REFUSED"
        root, archive = _pf_identity(intent)
        code = "PREFIRE_POINTER_DRIFT_REFUSED"
        _pf_pointer(intent, pointer_path)
        code = "PREFIRE_NON_TIMING_GATE_REFUSED"
        _pf_evidence(intent, root, archive)
        code = "PREFIRE_RISK_EVIDENCE_REFUSED"
        validate_prefire_risk(intent["evidence"]["timing_risk"], intent, repo=repo)
        return intent
    except PrefireRefusal:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, SealContractError) as exc:
        raise PrefireRefusal(code, str(exc)) from exc


def write_prefire_refusal(exc: PrefireRefusal, *, paths: tuple[Path, ...] = (), output_dir: Path | None = None) -> None:
    """Keep typed refusals beside both immutable objects and in authorized output custody."""
    import sys
    destinations = {Path(p).parent for p in paths}
    if output_dir is not None:
        destinations.add(Path(output_dir))
    print(str(exc), file=sys.stderr)
    for directory in destinations:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Each refusal remains independently retained; do not overwrite a prior fact.
            import uuid
            path = directory / f"PREFIRE_REFUSAL_{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(exc.to_dict(), indent=2) + "\n", encoding="utf-8")
        except OSError as write_error:
            print(f"REFUSAL_CUSTODY_FAILED: {directory}: {write_error}", file=sys.stderr)


def _pf_write_new(path: Path, document: dict) -> None:
    """Exclusive, fsynced object creation. Never overwrite immutable lifecycle history."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except FileExistsError as exc:
        raise PrefireRefusal("FIRST_MEASUREMENT_REPLAY_REFUSED", f"object already exists: {path}") from exc


def build_prefire_intent(*, candidate_id: str, runtime_dir: Path, evidence_paths: dict,
                         public_entrypoint_smoke: dict, net_ds_threshold: float,
                         retained_paths: list[str], falsifiers: list[str], out_path: Path,
                         repo: Path | None = None, pointer_path: Path | None = None) -> dict:
    from tac.decode_wall_clock import measure_receiver_digest
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    code = "PREFIRE_CONTRACT_DRIFT_REFUSED"
    frozen = _pf_read(repo / PREFIRE_FREEZE, code)
    commit = frozen["implementation_commit"]
    root = runtime_dir.resolve()
    archive = root / "archive.zip"
    runtime = measure_runtime_digest(root)
    live = read_pointer_state(pointer_path=pointer_path, axis="contest_cuda")
    evidence = {k: prefire_file_reference(v) for k, v in evidence_paths.items()}
    manifest = _pf_ref(evidence["candidate_manifest"], "PREFIRE_NON_TIMING_GATE_REFUSED")
    base_archive = Path(public_entrypoint_smoke["public_path_probes"]["frontier"]["archive_path"])
    ds = 25 * (archive.stat().st_size - base_archive.stat().st_size) / 37545489
    document = {
        "schema": PREFIRE_INTENT_SCHEMA, "state": "PREFIRE_FIRST_MEASUREMENT_ONLY",
        "candidate_id": candidate_id, "created_at_utc": _utc_now(), "created_by": "candidate-prefire-producer",
        "producer_source_commit": manifest["producer_source_commit"], "score_claim": False,
        "promotion_eligible": False, "timing_clearance": False,
        "contract": {"adjudication_memo": prefire_file_reference(repo / PREFIRE_MEMO),
                     "implementation_commit": commit, "implementation_manifest": frozen["implementation_manifest"],
                     "implementation_manifest_sha256": frozen["implementation_manifest"]["sha256"]},
        "candidate": {"archive": prefire_file_reference(archive), "runtime": {"path": str(root), **runtime.to_dict()},
                      "normalized_receiver": {"digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
                                              "sha256": measure_receiver_digest(root)},
                      "receiver_pins": [{"relative_path": p, "bytes": runtime.file_map()[p][0],
                                         "sha256": runtime.file_map()[p][1]} for p in ("inflate.py", "inflate.sh")],
                      "archive_member": None},
        "admit_bar": {"rule": "net dS = dS_rate + 100*(d_seg_new - d_seg_base) + (sqrt(10*d_pose_new) - sqrt(10*d_pose_base)) < threshold",
                      "net_dS_threshold": net_ds_threshold, "pointer_axis": "contest_cuda",
                      "pointer_score_at_intent": live["pointer_score"],
                      "pointer_archive_sha256_at_intent": live["pointer_archive_sha256"],
                      "pointer_tolerance_abs": 0.0, "require_pointer_archive_identity": True,
                      "rate_only_precheck": {"raw_identity_required": True, "normalizer_bytes": 37545489,
                                             "derived_net_dS": ds, "passed": True}},
        "public_entrypoint_smoke": public_entrypoint_smoke, "evidence": evidence,
        "dispatch_policy": dict(PREFIRE_DISPATCH_POLICY), "retained_payload_paths": retained_paths,
        "falsifiers": falsifiers,
    }
    document["intent_sha256"] = prefire_digest(document, "intent_sha256")
    _pf_write_new(out_path, document)
    try:
        return validate_prefire_intent(out_path, require_committed=False, repo=repo, pointer_path=pointer_path)
    except Exception:
        Path(out_path).unlink()  # New failed object only; no input payload or historical object is removed.
        raise


def first_measurement_nonce(intent_digest: str, lane_id: str, instance_job_id: str) -> str:
    return hashlib.sha256("\0".join((intent_digest, lane_id, instance_job_id,
                                   "candidate-first-measurement-v1")).encode("utf-8")).hexdigest()


def _pf_cost(ref: dict, *, now=None) -> dict:
    from datetime import UTC, datetime
    code = "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"
    cost = _pf_ref(ref, code)
    _pf_require(isinstance(cost, dict) and cost.get("gpu") == "T4" and cost.get("currency") == "USD"
                and type(cost.get("paid_dispatches")) is int and cost["paid_dispatches"] == 1,
                code, "cost must bind one T4 dispatch in USD")
    seconds = _pf_number(cost.get("remote_seconds"), code, "remote seconds", positive=True)
    _pf_require(seconds == 4800, code, "cost must bound the full 4800-second function cap")
    fetched = _pf_time(cost.get("provider_price_fetched_at_utc"), code)
    _pf_require(0 <= ((now or datetime.now(UTC)) - fetched).total_seconds() <= 86400, code, "provider price stale/future")
    source = cost.get("provider_price_source")
    _pf_require(isinstance(source, dict) and isinstance(source.get("url"), str)
                and source["url"].startswith("https://"), code, "current provider-price source absent")
    source_doc = _pf_ref(source, code)
    _pf_require(isinstance(source_doc, dict), code, "machine-readable provider price source required")
    resources = cost.get("resources")
    _pf_require(isinstance(resources, list) and resources and all(isinstance(r, dict) for r in resources)
                and {r.get("resource") for r in resources}
                >= {"gpu", "cpu", "memory"}, code, "all charged resources must be enumerated")
    resource_map = {r["resource"]: r for r in resources}
    chargeable = source_doc.get("chargeable_resources")
    _pf_require(isinstance(chargeable, list) and chargeable and all(isinstance(name, str) for name in chargeable)
                and len(chargeable) == len(set(chargeable)) and set(resource_map) == set(chargeable), code,
                "every provider-declared charged resource must be priced, including storage/transfer when charged")
    _pf_require(len(resource_map) == len(resources) and resource_map["gpu"].get("quantity") == 1
                and resource_map["cpu"].get("quantity") == 4 and resource_map["memory"].get("quantity") == 16,
                code, "one T4, four CPUs, sixteen GiB hard resource bounds required")
    total = 0.0
    for row in resources:
        quantity = _pf_number(row.get("quantity"), code, "resource quantity", positive=True)
        price = _pf_number(row.get("usd_per_unit_second"), code, "resource unit price")
        _pf_require(price >= 0, code, "negative resource price")
        field_path = row.get("price_field")
        _pf_require(isinstance(field_path, list) and field_path and all(isinstance(k, str) for k in field_path),
                    code, "price must name its field in retained provider source")
        source_price = source_doc
        for key in field_path:
            _pf_require(isinstance(source_price, dict) and key in source_price, code, "provider price field absent")
            source_price = source_price[key]
        _pf_require(type(source_price) in (int, float) and source_price == price, code, "price differs from provider source")
        subtotal = quantity * price * seconds
        _pf_require(row.get("upper_bound_usd") == subtotal, code, "resource cost not re-derived")
        total += subtotal
    _pf_require(cost.get("upper_bound_usd") == total and 0 < total < 5.0, code, "cost bound must be strictly < 5 USD")
    return cost


def _pf_output(path: object) -> Path:
    code = "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"
    _pf_require(isinstance(path, str) and Path(path).is_absolute(), code, "absolute output path required")
    out = Path(path)
    _pf_require(any(out.resolve().is_relative_to(Path(root)) for root in
                ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")), code, "output must be durable SSD custody")
    _pf_require(out == out.resolve(), code, "output path must be canonical without symlink indirection")
    return out


def build_first_measurement_authorization(*, intent_path: Path, lane_id: str, instance_job_id: str,
                                          output_dir: Path, cost_path: Path, repo: Path | None = None) -> dict:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    intent = validate_prefire_intent(intent_path, repo=repo)
    head = _pf_git(repo, "rev-parse", "HEAD").decode().strip()
    _pf_git(repo, "merge-base", "--is-ancestor", head, "main")
    output = _pf_output(str(output_dir))
    _pf_require(not _is_placeholder(lane_id) and not _is_placeholder(instance_job_id),
                "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED", "real lane and job required")
    cost = prefire_file_reference(cost_path)
    _pf_cost(cost)
    ref = prefire_file_reference(intent_path)
    document = {"schema": FIRST_MEASUREMENT_AUTHORIZATION_SCHEMA, "state": "AUTHORIZED_ONCE",
                "authorized_by": "MAIN", "authorized_at_utc": _utc_now(),
                "intent": {"path": ref["path"], "file_sha256": ref["sha256"], "file_bytes": ref["bytes"],
                           "digest": intent["intent_sha256"], "commit": head},
                "candidate_id": intent["candidate_id"], "axis": "contest_cuda", "lane_id": lane_id,
                "instance_job_id": instance_job_id, "claim_agent": "MAIN", "output_dir": str(output),
                "receipt_path": str(output / "MODAL_REMOTE_RESULT.json"),
                "authorization_nonce": first_measurement_nonce(intent["intent_sha256"], lane_id, instance_job_id),
                "dispatch_policy_sha256": prefire_digest(intent["dispatch_policy"]), "cost_preflight": cost,
                "single_axis_waiver_reason": PREFIRE_SINGLE_AXIS_REASON, "score_claim": False,
                "promotion_eligible": False}
    document["authorization_sha256"] = prefire_digest(document, "authorization_sha256")
    return document


def validate_first_measurement_authorization(path: Path, intent_path: Path, intent: dict, *,
                                             repo: Path | None = None, check_cost: bool = True) -> dict:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    code = "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"
    try:
        auth = _pf_read(path, code)
        expected_keys = {"schema", "state", "authorized_by", "authorized_at_utc", "intent", "candidate_id", "axis", "lane_id", "instance_job_id", "claim_agent", "output_dir", "receipt_path", "authorization_nonce", "dispatch_policy_sha256", "cost_preflight", "single_axis_waiver_reason", "score_claim", "promotion_eligible", "authorization_sha256"}
        _pf_require(isinstance(auth, dict) and set(auth) == expected_keys
                    and auth["schema"] == FIRST_MEASUREMENT_AUTHORIZATION_SCHEMA
                    and auth["state"] == "AUTHORIZED_ONCE" and auth["authorized_by"] == auth["claim_agent"] == "MAIN"
                    and auth["score_claim"] is False and auth["promotion_eligible"] is False
                    and auth["authorization_sha256"] == prefire_digest(auth, "authorization_sha256"), code,
                    "authorization schema/workflow/digest refused")
        ref = auth["intent"]
        actual = prefire_file_reference(intent_path)
        _pf_require(set(ref) == {"path", "file_sha256", "file_bytes", "digest", "commit"}
                    and ref["path"] == actual["path"] and type(ref["file_bytes"]) is int
                    and ref["file_bytes"] == actual["bytes"] and ref["file_sha256"] == actual["sha256"]
                    and ref["digest"] == intent["intent_sha256"], code,
                    "authorization must bind both exact intent file bytes/hash AND canonical digest")
        _pf_require(auth["candidate_id"] == intent["candidate_id"] and auth["axis"] == "contest_cuda"
                    and auth["dispatch_policy_sha256"] == prefire_digest(intent["dispatch_policy"])
                    and auth["single_axis_waiver_reason"] == PREFIRE_SINGLE_AXIS_REASON, code, "authorization field mismatch")
        _pf_require(all(isinstance(auth[k], str) and not _is_placeholder(auth[k]) for k in ("lane_id", "instance_job_id")),
                    code, "lane/job malformed")
        _pf_require(auth["authorization_nonce"] == first_measurement_nonce(intent["intent_sha256"], auth["lane_id"],
                    auth["instance_job_id"]), code, "nonce differs from exact NUL derivation")
        output = _pf_output(auth["output_dir"])
        _pf_require(auth["receipt_path"] == str(output / "MODAL_REMOTE_RESULT.json"), code, "receipt destination differs")
        _pf_require(_pf_time(auth["authorized_at_utc"], code) >= _pf_time(intent["created_at_utc"], code),
                    code, "authorization precedes intent")
        _pf_blob(repo, "HEAD", path)
        _pf_blob(repo, "HEAD", intent_path)
        _pf_blob(repo, ref["commit"], intent_path)
        _pf_git(repo, "merge-base", "--is-ancestor", ref["commit"], "HEAD")
        _pf_git(repo, "merge-base", "--is-ancestor", intent["contract"]["implementation_commit"], ref["commit"])
        head = _pf_git(repo, "rev-parse", "HEAD").decode().strip()
        _pf_git(repo, "merge-base", "--is-ancestor", head, "main")
        _pf_require(re.fullmatch(r"[0-9a-f]{40}", ref["commit"]) is not None, code, "intent commit malformed")
        if check_cost:
            _pf_cost(auth["cost_preflight"])
        else:
            _pf_ref(auth["cost_preflight"], code)
        return auth
    except PrefireRefusal as exc:
        if exc.code == code:
            raise
        raise PrefireRefusal(code, exc.detail) from exc
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PrefireRefusal(code, str(exc)) from exc


def first_measurement_consumption_path(auth: dict, repo: Path | None = None) -> Path:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    return repo / ".omx/state/first_measurement_consumptions" / f"{auth['authorization_nonce']}.json"


def check_first_measurement_unconsumed(auth: dict, *, repo: Path | None = None) -> None:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    code = "FIRST_MEASUREMENT_REPLAY_REFUSED"
    path = first_measurement_consumption_path(auth, repo)
    _pf_require(not path.exists(), code, "nonce already consumed, including partial/ambiguous reservations")
    for previous in path.parent.glob("*.json"):
        old = _pf_read(previous, code)
        _pf_require(old.get("instance_job_id") != auth["instance_job_id"] and old.get("output_dir") != auth["output_dir"],
                    code, "job/output already used")
    out = Path(auth["output_dir"])
    _pf_require(not out.exists() or not any(not p.name.startswith("PREFIRE_REFUSAL_") for p in out.iterdir()),
                code, "output contains an existing or ambiguous dispatch")


def reserve_first_measurement(auth: dict, *, repo: Path | None = None) -> dict:
    """One durable reservation under a global lock; all failures leave nonce consumed."""
    import fcntl
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    target = first_measurement_consumption_path(auth, repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ".reservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_first_measurement_unconsumed(auth, repo=repo)
        record = {"schema": "candidate_first_measurement_consumption.v1", "state": "RESERVED",
                  "authorization_sha256": auth["authorization_sha256"], "authorization_nonce": auth["authorization_nonce"],
                  "prefire_intent_sha256": auth["intent"]["digest"], "intent_file_sha256": auth["intent"]["file_sha256"],
                  "intent_file_bytes": auth["intent"]["file_bytes"], "instance_job_id": auth["instance_job_id"],
                  "output_dir": auth["output_dir"], "receipt_path": auth["receipt_path"], "lane_id": auth["lane_id"],
                  "reserved_at_utc": _utc_now(), "score_claim": False, "promotion_eligible": False,
                  "authorization_containing_commit": _pf_git(repo, "rev-parse", "HEAD").decode().strip()}
        _pf_write_new(target, record)
        _pf_write_new(Path(auth["output_dir"]) / "FIRST_MEASUREMENT_CONSUMPTION.json", record)
        return record


def transition_first_measurement(auth: dict, state: str, *, call_id: str,
                                 receipt_path: Path | None = None, repo: Path | None = None) -> dict:
    """Forward-only, locked, atomic nonce state transitions; never retry/reset a reservation."""
    import fcntl
    import os
    target = first_measurement_consumption_path(auth, repo)
    code = "FIRST_MEASUREMENT_REPLAY_REFUSED"
    with (target.parent / ".reservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = _pf_read(target, code)
        _pf_require(record.get("authorization_sha256") == auth["authorization_sha256"], code, "consumption authorization differs")
        _pf_require((record.get("state"), state) in {("RESERVED", "SPAWNED"), ("SPAWNED", "HARVESTED")}
                    and isinstance(call_id, str) and call_id.startswith("fc-"), code, "invalid forward transition/call id")
        _pf_registered_call(auth, call_id, Path(repo) if repo else Path(__file__).resolve().parents[2])
        if state == "HARVESTED":
            _pf_require(record.get("call_id") == call_id and receipt_path is not None
                        and str(Path(receipt_path).resolve()) == auth["receipt_path"], code, "harvest call/receipt differs")
            record["receipt"] = prefire_file_reference(receipt_path)
        record.update(state=state, call_id=call_id, updated_at_utc=_utc_now())
        for destination in (target, Path(auth["output_dir"]) / "FIRST_MEASUREMENT_CONSUMPTION.json"):
            temporary = destination.with_suffix(".next.json")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        return record


def quarantine_first_measurement_result(result: dict, context: dict) -> dict:
    """Preserve components and payloads while withholding promotion at worker and harvest."""
    result = dict(result)
    result.update(context)
    result.update(score_claim=False, promotion_eligible=False, adjudication_required=True,
                  first_measurement=True)
    return result


def _pf_completion_facts(intent: dict, auth: dict, receipt_path: Path, *, repo: Path) -> tuple[dict, dict]:
    from tac.decode_wall_clock import _cold_public_report, _field, build_t4_direct_leg
    code = "FIRST_MEASUREMENT_RESULT_REFUSED"
    consumed = _pf_read(first_measurement_consumption_path(auth, repo), code)
    _pf_require(consumed.get("state") == "HARVESTED" and consumed.get("authorization_sha256") == auth["authorization_sha256"],
                code, "authorization not harvested")
    _pf_require(consumed.get("receipt") == prefire_file_reference(receipt_path)
                and str(receipt_path.resolve()) == auth["receipt_path"], code, "receipt custody differs")
    _pf_registered_call(auth, consumed.get("call_id"), repo)
    result = _pf_ref(consumed["receipt"], code)
    request_path = Path(auth["output_dir"]) / "modal_cuda_auth_eval_local_request.json"
    request = _pf_read(request_path, code)
    for key, value in {"prefire_intent_sha256": intent["intent_sha256"],
                       "first_measurement_authorization_sha256": auth["authorization_sha256"],
                       "lane_id": auth["lane_id"], "instance_job_id": auth["instance_job_id"],
                       "receipt_path": auth["receipt_path"]}.items():
        _pf_require(result.get(key) == value and request.get(key) == value, code, f"result/request {key} differs")
    _pf_require(result.get("call_id") == consumed.get("call_id") and isinstance(result.get("call_id"), str),
                code, "call lineage differs")
    _pf_require(result.get("score_claim") is False and result.get("promotion_eligible") is False
                and result.get("adjudication_required") is True, code, "result escaped quarantine")
    _pf_require(result.get("worker_request") == prefire_file_reference(request_path), code, "retained worker request differs")
    snapshot = request.get("source_snapshot")
    _pf_require(isinstance(snapshot, dict) and snapshot.get("schema") == "modal_source_snapshot.v1"
                and snapshot.get("complete") is True and not snapshot.get("verify_failures")
                and not snapshot.get("missing_in_source") and _is_sha256(snapshot.get("files_digest"))
                and result.get("source_snapshot") == snapshot, code, "immutable source snapshot custody differs")
    argv = request.get("exact_argv")
    _pf_require(isinstance(argv, list) and argv == result.get("exact_argv"), code, "exact dispatch argv absent/different")
    for flag, value in {"--gpu": "T4", "--scorer-device": "cuda", "--inflate-device": "auto",
                        "--inflate-timeout": "1800", "--evaluate-timeout": "1800",
                        "--claim-policy": "require_active", "--lane-id": auth["lane_id"],
                        "--instance-job-id": auth["instance_job_id"], "--output-dir": auth["output_dir"],
                        "--expected-archive-sha256": intent["candidate"]["archive"]["sha256"]}.items():
        _pf_require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                    and argv[argv.index(flag) + 1] == value, code, f"worker argv {flag} differs")
    expected = {"inflate_timeout_seconds": 1800, "evaluate_timeout_seconds": 1800,
                "modal_function_timeout_seconds": 4800, "poller_deadline_seconds": 5400}
    for key, value in expected.items():
        _pf_require(type(request.get(key)) is int and request[key] == value and result.get(key) == value,
                    "FIRST_MEASUREMENT_TIMEOUT_REFUSED", f"timeout binding {key} differs")
    _pf_require(result.get("returncode") != 124 and not result.get("timed_out"),
                "FIRST_MEASUREMENT_TIMEOUT_REFUSED", "worker timed out")
    try:
        _pf_require(_field(result, ["artifacts", "contest_auth_eval.json", "n_samples"], "n_samples") == 600,
                    "FIRST_MEASUREMENT_WARM_REFUSED", "not n600")
        _cold_public_report(result)
    except SealContractError as exc:
        if isinstance(exc, PrefireRefusal):
            raise
        raise PrefireRefusal("FIRST_MEASUREMENT_WARM_REFUSED", str(exc)) from exc
    try:
        seconds = _field(result, ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"], "seconds")
        seconds = _pf_number(seconds, "FIRST_MEASUREMENT_TIMEOUT_REFUSED", "inflate seconds", positive=True)
        _pf_require(seconds <= 1260, "FIRST_MEASUREMENT_T4_POLICY_REFUSED", "completed inflate exceeds 1260 seconds")
        leg = build_t4_direct_leg(t4_receipt_path=receipt_path,
            runtime_dir=Path(intent["candidate"]["runtime"]["path"]), archive_path=Path(intent["candidate"]["archive"]["path"]))
    except PrefireRefusal:
        raise
    except SealContractError as exc:
        raise PrefireRefusal(code, str(exc)) from exc
    return leg, result


def _pf_completed_score(intent: dict, result: dict) -> None:
    from tac.decode_wall_clock import _field
    code = "FIRST_MEASUREMENT_RESULT_REFUSED"
    import math
    seg = _pf_number(_field(result, ["artifacts", "contest_auth_eval.json", "avg_segnet_dist"], "d_seg"), code, "exact d_seg")
    pose = _pf_number(_field(result, ["artifacts", "contest_auth_eval.json", "avg_posenet_dist"], "d_pose"), code, "exact d_pose")
    _pf_require(result.get("avg_segnet_dist") == seg and result.get("avg_posenet_dist") == pose, code,
                "outer components differ from exact evaluator artifact")
    _pf_require(seg >= 0 and pose >= 0 and result.get("archive_size_bytes") == intent["candidate"]["archive"]["bytes"],
                code, "invalid exact components or archive bytes")
    score = 100 * seg + math.sqrt(10 * pose) + 25 * intent["candidate"]["archive"]["bytes"] / 37545489
    _pf_require(score - intent["admit_bar"]["pointer_score_at_intent"] < intent["admit_bar"]["net_dS_threshold"],
                code, "exact recomputed score misses frozen net-dS bar")



def complete_first_fire_intent(*, intent_path: Path, authorization_path: Path, receipt_path: Path,
                               out_path: Path, repo: Path | None = None, pointer_path: Path | None = None) -> dict:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    intent = _pf_read(intent_path, "PREFIRE_INTENT_SCHEMA_REFUSED")
    _pf_schema(intent)
    auth = validate_first_measurement_authorization(authorization_path, intent_path, intent, repo=repo, check_cost=False)
    try:
        leg, result = _pf_completion_facts(intent, auth, receipt_path, repo=repo)
    except PrefireRefusal:
        raise
    except (SealContractError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PrefireRefusal("FIRST_MEASUREMENT_RESULT_REFUSED", str(exc)) from exc
    intent = validate_prefire_intent(intent_path, repo=repo, pointer_path=pointer_path)
    try:
        _pf_completed_score(intent, result)
    except PrefireRefusal:
        raise
    except (SealContractError, OSError, ValueError, KeyError, TypeError) as exc:
        raise PrefireRefusal("FIRST_MEASUREMENT_RESULT_REFUSED", str(exc)) from exc
    bar = intent["admit_bar"]
    document = build_seal(candidate_id=intent["candidate_id"], runtime_dir=Path(intent["candidate"]["runtime"]["path"]),
        archive_path=Path(intent["candidate"]["archive"]["path"]), axis="contest_cuda",
        admit_bar=AdmitBar(rule=bar["rule"], net_dS_threshold=bar["net_dS_threshold"], pointer_axis="contest_cuda",
            pointer_score_at_seal=bar["pointer_score_at_intent"],
            pointer_archive_sha256_at_seal=bar["pointer_archive_sha256_at_intent"]),
        public_entrypoint_smoke=intent["public_entrypoint_smoke"], decode_wall_clock=leg,
        archive_member_name=(intent["candidate"]["archive_member"] or {}).get("name", ""),
        retained_payload_paths=tuple(intent["retained_payload_paths"]), falsifiers=tuple(intent["falsifiers"]), sealed_by="MAIN")
    leg_path = out_path.with_name(out_path.name + ".decode_wall_clock.json")
    _pf_write_new(leg_path, leg)
    document.update(decode_wall_clock_reference=prefire_file_reference(leg_path))
    document.update(schema=SEAL_SCHEMA_V3, prefire_intent=prefire_file_reference(intent_path),
                    first_measurement_authorization=prefire_file_reference(authorization_path),
                    first_measurement_receipt=prefire_file_reference(receipt_path), prefire_intent_sha256=intent["intent_sha256"])
    document["seal_sha256"] = compute_seal_sha256(document)
    _pf_write_new(out_path, document)
    verdict = validate_seal(out_path, pointer_path=pointer_path, require_decode_wall_clock=True)
    if not verdict.ok:
        out_path.unlink()
        raise PrefireRefusal("FIRST_MEASUREMENT_RESULT_REFUSED", verdict.summary())
    return document


def _pf_validate_completed_seal(document: dict, *, pointer_path: Path | None = None) -> None:
    repo = Path(__file__).resolve().parents[2]
    code = "FIRST_MEASUREMENT_RESULT_REFUSED"
    for name in ("prefire_intent", "first_measurement_authorization", "first_measurement_receipt", "decode_wall_clock_reference"):
        _pf_ref(document.get(name), code)
    path = Path(document["prefire_intent"]["path"])
    intent = validate_prefire_intent(path, repo=repo, pointer_path=pointer_path)
    _pf_require(document.get("prefire_intent_sha256") == intent["intent_sha256"]
                and document["archive"] == intent["candidate"]["archive"]
                and document["runtime"] == intent["candidate"]["runtime"]
                and document["candidate_id"] == intent["candidate_id"], code, "completed seal differs from immutable intent")
    bar = intent["admit_bar"]
    expected_bar = AdmitBar(rule=bar["rule"], net_dS_threshold=bar["net_dS_threshold"], pointer_axis="contest_cuda",
        pointer_score_at_seal=bar["pointer_score_at_intent"],
        pointer_archive_sha256_at_seal=bar["pointer_archive_sha256_at_intent"]).to_dict()
    _pf_require(document["axis"] == "contest_cuda" and document["admit_bar"] == expected_bar
                and document["receiver_pins"] == intent["candidate"]["receiver_pins"]
                and document.get("archive_member") == intent["candidate"]["archive_member"]
                and document["falsifiers"] == intent["falsifiers"]
                and document["retained_payload_paths"] == intent["retained_payload_paths"],
                code, "completed seal changed a frozen non-timing binding")
    auth = validate_first_measurement_authorization(Path(document["first_measurement_authorization"]["path"]),
        path, intent, repo=repo, check_cost=False)
    leg, result = _pf_completion_facts(intent, auth, Path(document["first_measurement_receipt"]["path"]), repo=repo)
    _pf_completed_score(intent, result)
    stored_leg = _pf_ref(document["decode_wall_clock_reference"], code)
    _pf_require(document.get("decode_wall_clock") == leg == stored_leg, code,
                "completed seal leg differs from its retained file or unchanged direct builder")


def _pf_registered_call(auth: dict, call_id: str, repo: Path) -> None:
    """Require the real dispatch registration, not merely a plausible call-id string."""
    code = "FIRST_MEASUREMENT_RESULT_REFUSED"
    ledger = repo / ".omx/state/modal_call_id_ledger.jsonl"
    try:
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise PrefireRefusal(code, f"call-id ledger unavailable: {exc}") from exc
    matches = [r for r in rows if r.get("call_id") == call_id and r.get("event_type") == "dispatched"]
    _pf_require(len(matches) == 1 and matches[0].get("first_measurement_authorization_sha256") == auth["authorization_sha256"]
                and matches[0].get("prefire_intent_sha256") == auth["intent"]["digest"]
                and matches[0].get("intent_file_sha256") == auth["intent"]["file_sha256"]
                and matches[0].get("intent_file_bytes") == auth["intent"]["file_bytes"]
                and matches[0].get("instance_job_id") == auth["instance_job_id"] and matches[0].get("lane_id") == auth["lane_id"]
                and matches[0].get("expected_axis") == "contest_cuda" and matches[0].get("gpu") == "T4"
                and matches[0].get("archive_count") == 1, code, "real one-call T4 registration absent or ambiguous")


def retain_first_measurement_worker_tree(root: Path, context: dict) -> dict:
    """Certify persistent-volume custody; block deletion of uncategorized raw/scratch bytes."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "FIRST_MEASUREMENT_RETENTION.json":
            rows.append(prefire_file_reference(path))
    manifest = {"schema": "candidate_first_measurement_worker_retention.v1", "payloads": rows,
                "exact_argv": context["exact_argv"], "authorization_sha256": context["first_measurement_authorization_sha256"],
                "prefire_intent_sha256": context["prefire_intent_sha256"], "retention_destination": str(root),
                "cleanup_disposition": "BLOCKED_KEEP_BYTES", "cleanup_reason": "no certified lossless local harvest for this remote tree yet",
                "score_claim": False, "promotion_eligible": False}
    path = root / "FIRST_MEASUREMENT_RETENTION.json"
    _pf_write_new(path, manifest)
    return prefire_file_reference(path)
