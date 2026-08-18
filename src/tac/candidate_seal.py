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
    "SEAL_AXES",
    "SEAL_BAR_DRIFT",
    "SEAL_BYTE_DRIFT",
    "SEAL_FILE_MISSING",
    "SEAL_PLACEHOLDER_PIN",
    "SEAL_RECEIVER_PIN_MISMATCH",
    "SEAL_RUNTIME_DRIFT",
    "SEAL_SCHEMA",
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

SEAL_SCHEMA = "candidate_seal.v1"

#: Verdicts of the seal-document layer.  Deliberately a separate namespace from brick 1's
#: pin verdicts: a caller must never confuse "this tree's receiver names this archive"
#: (a one-question check) with "this sealed object is still exactly what was sealed".
SEAL_VALID = "SEAL_VALID"
SEAL_SCHEMA_VIOLATION = "SEAL_SCHEMA_VIOLATION"
SEAL_PLACEHOLDER_PIN = "SEAL_PLACEHOLDER_PIN"
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
        return {"sha256": self.sha256, "file_count": self.file_count, "total_bytes": self.total_bytes}

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


def build_seal(
    *,
    candidate_id: str,
    runtime_dir: Path,
    archive_path: Path | None = None,
    axis: str = "contest_cuda",
    admit_bar: AdmitBar,
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
    if "schema" in document and document["schema"] != SEAL_SCHEMA:
        # No ``None`` escape hatch: an unversioned seal is one this validator cannot claim to
        # understand, and claiming to is how a v2 document gets validated by v1 rules.
        problems.append(f"unknown seal schema {document.get('schema')!r}; this validator speaks {SEAL_SCHEMA}")
    if axis and axis not in SEAL_AXES:
        problems.append(f"unknown axis {axis!r}; expected one of {list(SEAL_AXES)}")
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

    # ---- 6. retained payload custody -----------------------------------------------------
    missing_payload = [p for p in document.get("retained_payload_paths", []) if not Path(str(p)).exists()]
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
