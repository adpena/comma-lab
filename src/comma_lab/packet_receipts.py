# SPDX-License-Identifier: MIT
"""Typed, schema-validated-BEFORE-write writer for gen6 `DOC_DIVERGENCE_RECEIPT` rows.

This is the ``ddm_rv17`` wave-2 **F1** carried item.  The finding was structural, not a
number: *there is no shared receipt writer*.  Every ``DOC_DIVERGENCE_RECEIPT*.json`` in the
gen6 chain was emitted by a one-shot, unreviewed script, and the post-write note check had
to be re-remembered each round.  The empirical anchor is R15, recorded verbatim in R16's
own ``known_defect_in_predecessor`` field:

    R15's ``repo_only_docs["verify_citations.py"].note`` was serialized as a single-element
    JSON **LIST** instead of a string -- a trailing-comma slip in the R15 writer whose
    tuple-guard checked the enclosing dict rather than the note value.  Content intact,
    type wrong.  Append-only forbids editing R15.  *"Class note for the next review wave:
    receipt writers are unreviewed one-shot scripts; a typed schema check before write
    would have refused the R15 list at the source."*

That is exactly what this module is.  The trailing-comma class is made **unrepresentable**:
a receipt is constructed as frozen dataclasses whose ``__post_init__`` refuses a non-string
note, and :func:`write_receipt` validates the whole record and re-parses its own serialized
bytes BEFORE anything reaches disk.  An invalid record leaves **no file** behind.

Scope and boundaries, stated plainly:

* **There was no in-repo append path to rewire.**  A repo-wide grep for
  ``DOC_DIVERGENCE_RECEIPT`` finds research memos plus the two post-write verifiers
  (``verify_receipt_chain.py``, ``verify_citations.py``) in the packet prep directory --
  no writer.  This module is therefore the *first* sanctioned append path, not a
  replacement for one; the next append (MAIN's, at the packet-swap boundary) is the
  trigger that consumes it.
* The schema is **DERIVED from the 14 real receipts**, not invented.  Every field name,
  optionality and nested shape below was read off ``DOC_DIVERGENCE_RECEIPT.json`` and
  ``_R4`` .. ``_R16``.  :func:`validate_receipt_mapping` re-reads all 14 as an executed
  control (see the CLI ``--check``).
* This writer **never edits an existing receipt**.  The chain is append-only; the writer
  refuses a target path that already exists and refuses a successor index that is not
  strictly greater than every index already in the directory.
* Writing a receipt is a **custody** act, not a score.  Nothing here measures, promotes,
  or claims anything about a score.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLI_DESCRIPTION = (
    "Validate gen6 DOC_DIVERGENCE_RECEIPT rows against the typed schema "
    "(the pre-write check, runnable over already-written receipts)."
)

#: The literal every real receipt carries in its ``schema`` field.  Read off all 14.
RECEIPT_SCHEMA = "packet_doc_divergence_receipt_v1"

#: ``DOC_DIVERGENCE_RECEIPT.json`` (the chain's oldest link) ranks 3; the suffixed
#: successors rank by their ``_R<n>``.  Same convention as ``verify_receipt_chain.py``.
RECEIPT_NAME_RE = re.compile(r"^DOC_DIVERGENCE_RECEIPT(?:_R(\d+))?\.json$")
UNSUFFIXED_RECEIPT_RANK = 3

#: ``verify_receipt_chain.py`` refuses a diverged pair whose entry declares no
#: ``publish_source``; step 4A publishes the declared copy and must never infer one.
VALID_PUBLISH_SOURCES = ("prep", "frozen")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ReceiptSchemaError(ValueError):
    """A receipt record violated the schema.  Raised BEFORE any bytes reach disk."""


def _require_str(value: Any, label: str) -> str:
    """Refuse anything that is not a non-empty plain string.

    This single guard is the R15 cure: the note that slipped through was a
    ``["..."]`` list, which is truthy, iterates, and serializes -- so every check that
    asked "is it there?" passed.  The question that catches it is "is it a str?".

    ``bool`` is excluded explicitly because it is not a ``str`` but reads like a value; the
    check is written to refuse it with the same message rather than fall through.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise ReceiptSchemaError(
            f"{label}: expected a string, got {type(value).__name__} ({value!r}). "
            "A one-element list here is the R15 trailing-comma slip -- drop the comma."
        )
    if not value.strip():
        raise ReceiptSchemaError(f"{label}: must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_str(value, label)
    if not _SHA256_RE.match(text):
        raise ReceiptSchemaError(
            f"{label}: expected 64 lowercase hex chars (a sha256), got {text!r}"
        )
    return text


@dataclass(frozen=True)
class DivergedFileEntry:
    """A document that exists in BOTH trees, tracked with both shas.

    Shape derived from the 34 real ``diverged_files`` entries: ``repo_final_sha256`` and
    ``frozen_gen6_sha256`` always; ``publish_source``, ``note`` and ``data_rows``
    optionally.

    This class validates SHAPE only.  The rv17 R11-F2 rule -- a diverged pair must declare
    a ``publish_source`` -- is a WRITE-time policy in
    :func:`check_publish_source_declared`, not a construction invariant, because R4-R8
    predate the rule and must still parse.  :attr:`copies_differ` is what that check reads.
    """

    repo_final_sha256: str
    frozen_gen6_sha256: str
    publish_source: str | None = None
    note: str | None = None
    data_rows: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.repo_final_sha256, "diverged_files.repo_final_sha256")
        _require_sha256(self.frozen_gen6_sha256, "diverged_files.frozen_gen6_sha256")
        if self.note is not None:
            _require_str(self.note, "diverged_files.note")
        if self.data_rows is not None:
            _require_str(self.data_rows, "diverged_files.data_rows")
        if self.publish_source is not None:
            _require_str(self.publish_source, "diverged_files.publish_source")
            if self.publish_source not in VALID_PUBLISH_SOURCES:
                raise ReceiptSchemaError(
                    f"diverged_files.publish_source={self.publish_source!r} must be one of "
                    f"{VALID_PUBLISH_SOURCES!r}"
                )

    @property
    def copies_differ(self) -> bool:
        return self.repo_final_sha256 != self.frozen_gen6_sha256

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "repo_final_sha256": self.repo_final_sha256,
            "frozen_gen6_sha256": self.frozen_gen6_sha256,
        }
        if self.publish_source is not None:
            out["publish_source"] = self.publish_source
        if self.data_rows is not None:
            out["data_rows"] = self.data_rows
        if self.note is not None:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class RepoOnlyDocEntry:
    """A citation-universe document tracked on the prep (repo) side only."""

    repo_final_sha256: str
    note: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.repo_final_sha256, "repo_only_docs.repo_final_sha256")
        if self.note is not None:
            _require_str(self.note, "repo_only_docs.note")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"repo_final_sha256": self.repo_final_sha256}
        if self.note is not None:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class RepoOnlyCorrectedDocEntry:
    """LEGACY shape, present only in the chain's first link (round 3).

    Kept because the schema is derived from the real chain and the first link is part of
    it.  ``frozen_counterpart`` is nullable there and is preserved as ``None``, never
    dropped -- an absent key and an explicit null are different facts.
    """

    repo_final_sha256: str
    frozen_counterpart: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(
            self.repo_final_sha256, "repo_only_corrected_docs.repo_final_sha256"
        )
        if self.frozen_counterpart is not None:
            _require_str(
                self.frozen_counterpart, "repo_only_corrected_docs.frozen_counterpart"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_final_sha256": self.repo_final_sha256,
            "frozen_counterpart": self.frozen_counterpart,
        }


@dataclass(frozen=True)
class FrozenOnlyDocEntry:
    """A citation-universe document that exists only in the frozen tree (e.g. README.md).

    The frozen custody is append-only, so a tracked sha here doubles as a freeze-integrity
    check on that file (rv17 R16-F1).
    """

    frozen_gen6_sha256: str
    note: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.frozen_gen6_sha256, "frozen_only_docs.frozen_gen6_sha256")
        if self.note is not None:
            _require_str(self.note, "frozen_only_docs.note")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"frozen_gen6_sha256": self.frozen_gen6_sha256}
        if self.note is not None:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class DocDivergenceReceipt:
    """One append-only link in the gen6 receipt chain.

    Required fields are the four every one of the 14 real receipts carries (``schema``,
    ``date_utc``, ``author``, ``reason``) plus ``diverged_files``, which every one carries
    and which ``verify_receipt_chain.py`` reads first.  The rest are optional because the
    real chain uses them optionally.

    ``supplements`` names the predecessor this receipt appends after; 13 of 14 carry it
    (only the chain's first link does not), so it is optional here but the writer warns
    when a successor omits it.
    """

    date_utc: str
    author: str
    reason: str
    diverged_files: Mapping[str, DivergedFileEntry]
    repo_only_docs: Mapping[str, RepoOnlyDocEntry] = field(default_factory=dict)
    frozen_only_docs: Mapping[str, FrozenOnlyDocEntry] = field(default_factory=dict)
    supplements: str | None = None
    review_lineage: tuple[str, ...] = ()
    known_defect_in_predecessor: str | None = None
    # LEGACY fields, present only in the chain's first link (round 3). Modelled because
    # the schema is DERIVED from the real chain; a new receipt should not use them.
    authoritative_source: str | None = None
    corrections_applied: tuple[str, ...] = ()
    repo_only_corrected_docs: Mapping[str, RepoOnlyCorrectedDocEntry] = field(
        default_factory=dict
    )
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ReceiptSchemaError(
                f"schema={self.schema!r} must be {RECEIPT_SCHEMA!r}"
            )
        _require_str(self.date_utc, "date_utc")
        if not _DATE_RE.match(self.date_utc):
            raise ReceiptSchemaError(
                f"date_utc={self.date_utc!r} must be YYYY-MM-DD (the form all 14 use)"
            )
        _require_str(self.author, "author")
        _require_str(self.reason, "reason")
        if self.supplements is not None:
            _require_str(self.supplements, "supplements")
        if self.known_defect_in_predecessor is not None:
            _require_str(
                self.known_defect_in_predecessor, "known_defect_in_predecessor"
            )
        if self.authoritative_source is not None:
            _require_str(self.authoritative_source, "authoritative_source")
        for label, seq in (
            ("review_lineage", self.review_lineage),
            ("corrections_applied", self.corrections_applied),
        ):
            if not isinstance(seq, tuple):
                raise ReceiptSchemaError(f"{label} must be a tuple (frozen)")
            for i, item in enumerate(seq):
                _require_str(item, f"{label}[{i}]")
        _validate_entry_map(self.diverged_files, "diverged_files", DivergedFileEntry)
        _validate_entry_map(self.repo_only_docs, "repo_only_docs", RepoOnlyDocEntry)
        _validate_entry_map(self.frozen_only_docs, "frozen_only_docs", FrozenOnlyDocEntry)
        _validate_entry_map(
            self.repo_only_corrected_docs,
            "repo_only_corrected_docs",
            RepoOnlyCorrectedDocEntry,
        )
        if not self.diverged_files and not self.repo_only_docs and not self.frozen_only_docs:
            raise ReceiptSchemaError(
                "a receipt that tracks ZERO documents is refused -- "
                "verify_receipt_chain.py fails closed on it too"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the key order the real receipts use."""
        out: dict[str, Any] = {"schema": self.schema}
        if self.supplements is not None:
            out["supplements"] = self.supplements
        out["date_utc"] = self.date_utc
        out["author"] = self.author
        if self.known_defect_in_predecessor is not None:
            out["known_defect_in_predecessor"] = self.known_defect_in_predecessor
        out["reason"] = self.reason
        if self.authoritative_source is not None:
            out["authoritative_source"] = self.authoritative_source
        if self.corrections_applied:
            out["corrections_applied"] = list(self.corrections_applied)
        out["diverged_files"] = {k: v.to_dict() for k, v in self.diverged_files.items()}
        if self.repo_only_corrected_docs:
            out["repo_only_corrected_docs"] = {
                k: v.to_dict() for k, v in self.repo_only_corrected_docs.items()
            }
        if self.repo_only_docs:
            out["repo_only_docs"] = {k: v.to_dict() for k, v in self.repo_only_docs.items()}
        if self.frozen_only_docs:
            out["frozen_only_docs"] = {
                k: v.to_dict() for k, v in self.frozen_only_docs.items()
            }
        if self.review_lineage:
            out["review_lineage"] = list(self.review_lineage)
        return out


def _validate_entry_map(value: Any, label: str, entry_type: type) -> None:
    if not isinstance(value, Mapping):
        raise ReceiptSchemaError(f"{label}: expected a mapping, got {type(value).__name__}")
    for name, entry in value.items():
        _require_str(name, f"{label}: document name")
        if not isinstance(entry, entry_type):
            raise ReceiptSchemaError(
                f"{label}[{name!r}]: expected {entry_type.__name__}, "
                f"got {type(entry).__name__}"
            )


# ---------------------------------------------------------------------------
# Parsing an existing receipt back into the typed form (the executed control).
# ---------------------------------------------------------------------------


def _entry_from_mapping(payload: Any, label: str, entry_type: type[Any]) -> Any:
    if not isinstance(payload, Mapping):
        raise ReceiptSchemaError(
            f"{label}: expected an object, got {type(payload).__name__}"
        )
    known = {f.name for f in dataclasses.fields(entry_type)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ReceiptSchemaError(
            f"{label}: unknown field(s) {unknown!r}; known fields are {sorted(known)!r}. "
            "A new field is a schema change -- add it to the dataclass, do not smuggle it."
        )
    return entry_type(**dict(payload))


def validate_receipt_mapping(payload: Mapping[str, Any]) -> DocDivergenceReceipt:
    """Parse an already-serialized receipt into the typed form, refusing on any violation.

    This is both the post-write check and the control that proves the schema was DERIVED
    from the real chain rather than invented: all 14 shipped receipts must parse.  R15 is
    the deliberate exception -- it parses only because its list-valued note is the defect
    R16 recorded, so this function reports it rather than accepting it.
    """
    if not isinstance(payload, Mapping):
        raise ReceiptSchemaError(f"receipt: expected an object, got {type(payload).__name__}")
    known = {
        "schema",
        "supplements",
        "date_utc",
        "author",
        "known_defect_in_predecessor",
        "reason",
        "diverged_files",
        "repo_only_docs",
        "frozen_only_docs",
        "review_lineage",
        # legacy, first link only
        "authoritative_source",
        "corrections_applied",
        "repo_only_corrected_docs",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ReceiptSchemaError(
            f"receipt: unknown top-level field(s) {unknown!r}. Known: {sorted(known)!r}"
        )
    for required in ("schema", "date_utc", "author", "reason", "diverged_files"):
        if required not in payload:
            raise ReceiptSchemaError(f"receipt: missing required field {required!r}")
    def _section(key: str, entry_type: type[Any]) -> dict[str, Any]:
        # Guard the container type BEFORE mapping over it: a JSON list of pairs would
        # survive dict() and quietly produce a wrong-shaped section.
        raw = payload.get(key, {})
        if not isinstance(raw, Mapping):
            raise ReceiptSchemaError(
                f"{key}: expected an object, got {type(raw).__name__}"
            )
        return {
            name: _entry_from_mapping(entry, f"{key}[{name!r}]", entry_type)
            for name, entry in raw.items()
        }

    diverged = _section("diverged_files", DivergedFileEntry)
    repo_only = _section("repo_only_docs", RepoOnlyDocEntry)
    frozen_only = _section("frozen_only_docs", FrozenOnlyDocEntry)
    corrected = _section("repo_only_corrected_docs", RepoOnlyCorrectedDocEntry)
    lineage = payload.get("review_lineage", ())
    corrections = payload.get("corrections_applied", ())
    for label, seq in (("review_lineage", lineage), ("corrections_applied", corrections)):
        if isinstance(seq, str):
            raise ReceiptSchemaError(f"{label}: expected a list of strings, got a string")
        # Positive type test, not just a str exclusion: a dict would tuple() into its keys
        # and an int would raise TypeError instead of a typed schema error.
        if not isinstance(seq, (list, tuple)):
            raise ReceiptSchemaError(
                f"{label}: expected a list of strings, got {type(seq).__name__}"
            )
    return DocDivergenceReceipt(
        schema=payload["schema"],
        supplements=payload.get("supplements"),
        date_utc=payload["date_utc"],
        author=payload["author"],
        known_defect_in_predecessor=payload.get("known_defect_in_predecessor"),
        reason=payload["reason"],
        diverged_files=diverged,
        repo_only_docs=repo_only,
        frozen_only_docs=frozen_only,
        review_lineage=tuple(lineage),
        authoritative_source=payload.get("authoritative_source"),
        corrections_applied=tuple(corrections),
        repo_only_corrected_docs=corrected,
    )


def check_publish_source_declared(receipt: DocDivergenceReceipt) -> None:
    """Refuse a diverged pair that declares no ``publish_source`` (rv17 R11-F2).

    Deliberately NOT a dataclass invariant: the rule was introduced at R11, so R4-R8
    legitimately predate it and must still PARSE.  ``verify_receipt_chain.py`` applies it
    to the LATEST receipt only, and this is the pre-write mirror of that -- a rule about
    what may be WRITTEN from now on, not about what the chain already contains.
    """
    undeclared = sorted(
        name
        for name, entry in receipt.diverged_files.items()
        if entry.copies_differ and entry.publish_source is None
    )
    if undeclared:
        raise ReceiptSchemaError(
            f"diverged pair(s) {undeclared!r} DIFFER but declare no publish_source "
            f"(rv17 R11-F2). Step 4A publishes the declared copy, never an inferred one -- "
            f"declare publish_source: {VALID_PUBLISH_SOURCES[0]!r} or "
            f"{VALID_PUBLISH_SOURCES[1]!r}."
        )


# ---------------------------------------------------------------------------
# The append path.
# ---------------------------------------------------------------------------


def receipt_rank(name: str) -> int | None:
    """Chain rank of a receipt filename, or None when the name is not a receipt."""
    match = RECEIPT_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)) if match.group(1) else UNSUFFIXED_RECEIPT_RANK


def next_receipt_name(receipts_dir: Path) -> str:
    """The filename the next appended receipt must take."""
    ranks = [
        rank
        for rank in (receipt_rank(p.name) for p in receipts_dir.iterdir() if p.is_file())
        if rank is not None
    ]
    return f"DOC_DIVERGENCE_RECEIPT_R{(max(ranks) if ranks else UNSUFFIXED_RECEIPT_RANK) + 1}.json"


def serialize_receipt(receipt: DocDivergenceReceipt) -> str:
    """Validate, serialize, and re-parse -- returning bytes only if the round trip holds.

    Applies the write-path policy checks (R11-F2) on top of the shape invariants the
    dataclasses already enforce at construction, then proves FULL round-trip fidelity:
    the serialized bytes must re-parse into a record that reconstructs the identical
    payload.  Parsing successfully is the weaker claim; reconstructing identically is the
    one that guarantees the next reader sees what the writer meant.
    """
    check_publish_source_declared(receipt)
    payload = receipt.to_dict()
    text = json.dumps(payload, indent=2) + "\n"
    reparsed = json.loads(text)
    if reparsed != payload:
        raise ReceiptSchemaError("serialize_receipt: JSON round trip changed the payload")
    if validate_receipt_mapping(reparsed).to_dict() != payload:
        raise ReceiptSchemaError(
            "serialize_receipt: the re-parsed record does not reconstruct the payload; "
            "a field is being dropped or reshaped on the way back in"
        )
    return text


def write_receipt(
    receipt: DocDivergenceReceipt,
    receipts_dir: Path,
    *,
    name: str | None = None,
    allow_overwrite: bool = False,
) -> Path:
    """Append one receipt to the chain, validating BEFORE any bytes reach disk.

    Order matters and is the whole point: the record is validated, serialized, and
    re-parsed first; only then is the target path opened.  A refusal therefore leaves the
    directory byte-identical -- there is no partial file to clean up and no invalid link
    for the next round to inherit.

    Refuses an existing path unless ``allow_overwrite`` (the chain is append-only), and
    refuses a name whose rank does not strictly exceed every rank already present.
    """
    text = serialize_receipt(receipt)  # validate FIRST; raises before any disk contact
    if not receipts_dir.is_dir():
        raise ReceiptSchemaError(f"receipts_dir is not a directory: {receipts_dir}")
    target_name = name or next_receipt_name(receipts_dir)
    rank = receipt_rank(target_name)
    if rank is None:
        raise ReceiptSchemaError(
            f"{target_name!r} is not a receipt filename (expected "
            "DOC_DIVERGENCE_RECEIPT[_R<n>].json)"
        )
    existing = [
        r
        for r in (receipt_rank(p.name) for p in receipts_dir.iterdir() if p.is_file())
        if r is not None
    ]
    if existing and rank <= max(existing) and not allow_overwrite:
        raise ReceiptSchemaError(
            f"{target_name!r} ranks {rank}, not above the chain head {max(existing)}. "
            "The chain is append-only: never edit or re-file an existing link."
        )
    target = receipts_dir / target_name
    if target.exists() and not allow_overwrite:
        raise ReceiptSchemaError(
            f"{target} already exists; receipts are append-only and are never edited"
        )
    target.write_text(text)
    return target


def _check_paths(paths: list[Path]) -> int:
    failures = 0
    for path in sorted(paths, key=lambda p: receipt_rank(p.name) or 0):
        try:
            validate_receipt_mapping(json.loads(path.read_text()))
        except (ReceiptSchemaError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)
        else:
            print(f"ok   {path.name}")
    print(f"\n{len(paths) - failures}/{len(paths)} receipts parse against {RECEIPT_SCHEMA}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--check",
        type=Path,
        nargs="+",
        required=True,
        help="receipt JSON file(s), or a directory of them, to validate against the schema",
    )
    args = parser.parse_args(argv)
    paths: list[Path] = []
    for target in args.check:
        if target.is_dir():
            paths.extend(
                p
                for p in sorted(target.iterdir())
                if p.is_file() and receipt_rank(p.name) is not None
            )
        else:
            paths.append(target)
    if not paths:
        print("FAIL: no receipt files to check (vacuity guard)", file=sys.stderr)
        return 1
    return _check_paths(paths)


if __name__ == "__main__":
    raise SystemExit(main())
