# SPDX-License-Identifier: MIT
"""Content-addressed ground-truth DECODE-LINEAGE registry, and the fail-closed guard over it.

THE BUG THIS EXTINCTS
---------------------
``ddm_pi2`` (2026-08-16, commit ``ed153d0203``) measured that our advisory scorer instrument read
TWO DIFFERENT GROUND TRUTHS at once.  The seg half loaded a cached ``gt_argmax_n600.npy`` of
DALI/nvdec lineage; the pose half had no cache, so it decoded GT fresh with PyAV every run.  One
instrument, two lineages, nobody could see it.  That split *was* the entire 21.4x advisory-vs-CUDA
pose offset that the campaign had been attributing to CPU-vs-CUDA hardware drift -- the actual
scorer-forward term is 3.572e-12, nine orders of magnitude down.

The cost of not knowing: a published dither row was 11.4x optimistic (+0.0024 S when the truth is
+0.0276 S, i.e. 2.87x the entire remaining gap rather than 25% of it).

pi2's own §6 item 8 named the structural gap and left it as its successor's charter:

    "I did not verify ``gt_argmax_n600.npy``'s documented provenance.  I established its lineage
    *empirically*.  No producer receipt was located that says so, which means the seg line has
    been relying on an authority-grade cache **by luck, undocumented** -- a standing fragility."

This module is the cure: lineage becomes RECORDED and ASSERTED instead of lucky and implicit.

WHY CONTENT-ADDRESSED AND NOT PATH-ADDRESSED
--------------------------------------------
``ddm_gl1`` measured **six distinct files named ``gt_argmax_n600.npy``** under different retained
run directories.  pi2 verified exactly one of them.  A path-keyed or name-keyed registry would
therefore reproduce the very bug it claims to cure: it would let one verified file's reputation
launder five unverified ones.  Identity here is the sha256 of the bytes.  A name is not identity.

THE TWO LINEAGES
----------------
``DALI_NVDEC``
    NVDEC RGB output via ``DaliVideoDataset``; **never** calls ``frame_utils.yuv420_to_rgb``.
    This is the lineage the contest-CUDA authority row is measured against, so it is
    :data:`AUTHORITY_LINEAGE`.

``PYAV_YUV420_TO_RGB``
    PyAV decodes planar YUV420, then the official ``frame_utils.yuv420_to_rgb`` performs bilinear
    chroma upsampling (``align_corners=False``) and BT.601 limited-range conversion.

Both are LEGITIMATE decodes.  CLAUDE.md's "GT decodes ONLY via ``frame_utils.yuv420_to_rgb``"
rule is about a THIRD, wrong path -- PyAV ``rgb24``, which manufactures ~100x phantom pose -- and
it is not in tension with this module.  The live question this module answers is narrower and was
never previously asked: *which of the two legitimate decodes does this particular instrument
need, and is it actually reading that one?*

MEASURED SEPARATION between the two lineages, on ``0.mkv``, n600 (``ddm_pi2`` §2/§4, re-derived
by ``ddm_gl1``): seg differs at 20,671 / 117,964,800 sites; pose MSE 1.4061324889e-04.  Scored
through the frozen scorer that is a **1.4425x** d_seg penalty and a **+1.4061e-04** additive
d_pose floor -- i.e. reading the wrong lineage silently reproduces the contest-CPU axis while
claiming to be advisory-for-CUDA.

EVIDENCE LADDER (strongest first) -- mirrors the campaign's ``empirical_verification_status``
-------------------------------------------------------------------------------------------
``PRODUCER_DECLARED``
    A producer receipt or argv names the decoder.  Only the two #906 rulers hold this today.
``EMPIRICAL_EXACT_MATCH`` / ``EMPIRICAL_EXACT_FRAME_SUBSET``
    Bit-identical to a ruler, whole-field or per-frame.
``EMPIRICAL_NEAREST_RULER_*``
    Decisively nearer one ruler than the other; the margin travels with the label.
``NONE``
    Lineage is UNKNOWN.  An honest UNKNOWN is worth more than a guessed label, and
    :func:`assert_gt_lineage` REFUSES on it -- unknown never reads as pass.

FAIL-CLOSED CONTRACT
--------------------
:func:`assert_gt_lineage` refuses on a missing file, on an unregistered sha256, and on a lineage
mismatch.  :func:`assert_single_lineage` refuses when one instrument's GT sources span more than
one lineage -- that is the pi2 bug stated as a predicate.

WHAT THE GAUGE WOULD READ IF THE CURE WERE APPLIED AND NOTHING ELSE CHANGED
--------------------------------------------------------------------------
This is the sister-law check that a detector must not zero out on its own cure.

A NAIVE gauge -- "fraction of GT loads that call :func:`assert_gt_lineage`" -- is REJECTED here:
applying the cure drives it to 100% by construction, whether or not a single declared lineage is
correct.  It measures instrumentation, not reality.

The gauge this module actually exposes, :func:`population_split_report`, counts **distinct
lineages present in the reachable GT population** and **instruments whose sources span more than
one**.  Adding declarations moves it by exactly zero.  It moves only when what an instrument
READS changes.  Applied to the population as measured on 2026-08-16 it still reports a live
split, which is the correct reading, because the split has not been repaired yet -- only made
visible.

Axis: ``[macOS-CPU advisory]``.  Lineage classification is never a score.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AUTHORITY_LINEAGE",
    "DALI_NVDEC",
    "PYAV_YUV420_TO_RGB",
    "REGISTRY_PATH",
    "SPLIT_LINEAGE_WITHIN_ARTIFACT",
    "UNKNOWN_AMBIGUOUS",
    "UNKNOWN_UNCOMPARABLE",
    "GtArtifactLineage",
    "GtLineageError",
    "GtLineageMismatch",
    "GtLineageSplit",
    "GtLineageUnknown",
    "GtSource",
    "assert_gt_lineage",
    "assert_single_lineage",
    "basename_lineage_collisions",
    "is_known_lineage",
    "lineage_of_file",
    "lineage_of_source",
    "load_registry",
    "population_split_report",
    "runtime_decode_lineage",
    "sha256_file",
]

# --- lineage vocabulary --------------------------------------------------------------------
DALI_NVDEC = "DALI_NVDEC"
PYAV_YUV420_TO_RGB = "PYAV_YUV420_TO_RGB"
UNKNOWN_AMBIGUOUS = "UNKNOWN_AMBIGUOUS"
UNKNOWN_UNCOMPARABLE = "UNKNOWN_UNCOMPARABLE"
SPLIT_LINEAGE_WITHIN_ARTIFACT = "SPLIT_LINEAGE_WITHIN_ARTIFACT"

#: The decode lineage the contest-CUDA authority row is measured against.
AUTHORITY_LINEAGE = DALI_NVDEC

#: Lineages that name a real decoder.  Everything else is an absence of knowledge.
RESOLVED_LINEAGES = frozenset({DALI_NVDEC, PYAV_YUV420_TO_RGB})

#: Runtime (non-file) GT sources.  pi2's bug lived HERE: the pose half decoded fresh every run,
#: so it had no file and therefore no sha256 to look up.  A registry that only knew about files
#: would be blind to exactly the half that was wrong.
RUNTIME_DECODE_LINEAGE: dict[str, str] = {
    "frame_utils.yuv420_to_rgb": PYAV_YUV420_TO_RGB,
    "AVVideoDataset": PYAV_YUV420_TO_RGB,
    "DaliVideoDataset": DALI_NVDEC,
}

REGISTRY_PATH = Path(__file__).resolve().parent / "gt_lineage_registry.json"


class GtLineageError(RuntimeError):
    """Base class for every fail-closed refusal in this module."""


class GtLineageUnknown(GtLineageError):
    """The artifact's decode lineage is not recorded, so no claim may rest on it."""


class GtLineageMismatch(GtLineageError):
    """The artifact's recorded lineage is not the one the instrument declared it needs."""


class GtLineageSplit(GtLineageError):
    """One instrument's ground-truth sources span more than one decode lineage (the pi2 bug).

    Carries STRUCTURED detail (``lineages``, ``resolved``) so consumers never have to parse the
    message text.  Recovering data by scraping an exception string is how a message reword
    silently breaks a caller.
    """

    def __init__(self, message: str, *, lineages: tuple[str, ...] = (), resolved: dict[str, str] | None = None):
        super().__init__(message)
        self.lineages = lineages
        self.resolved = dict(resolved or {})


@dataclass(frozen=True)
class GtArtifactLineage:
    """One recorded ground-truth artifact, keyed by the sha256 of its bytes."""

    sha256: str
    bytes: int
    basename: str
    lineage: str
    evidence: str
    measurement: str
    claim_boundary: str
    known_paths: tuple[str, ...] = ()
    #: Every filename these bytes are known by.  One content can carry several names, so a single
    #: ``basename`` field is not enough to detect a cross-lineage name collision.
    known_basenames: tuple[str, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.lineage in RESOLVED_LINEAGES


@dataclass(frozen=True)
class GtSource:
    """A ground-truth source: either a file on disk, or a decoder invoked at runtime."""

    kind: str  # "file" | "runtime_decode"
    ref: str  # a filesystem path, or a key of RUNTIME_DECODE_LINEAGE

    @classmethod
    def file(cls, path: str | Path) -> GtSource:
        return cls(kind="file", ref=str(path))

    @classmethod
    def runtime_decode(cls, name: str) -> GtSource:
        return cls(kind="runtime_decode", ref=name)


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """Stream the sha256 of a file.  Streaming keeps multi-GiB GT caches off the heap."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


#: Digest memo keyed by (resolved path, size, mtime_ns).  GT caches reach 4.8 GiB, so re-hashing
#: on every guarded read would make the guard expensive enough that someone disables it -- and a
#: guard people turn off protects nothing.  The key includes size and mtime, so a file edited in
#: place is re-hashed rather than served stale.
_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _cached_digest(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    digest = _DIGEST_CACHE.get(key)
    if digest is None:
        digest = sha256_file(path)
        _DIGEST_CACHE[key] = digest
    return digest


_REGISTRY_CACHE: dict[str, GtArtifactLineage] | None = None


def load_registry(*, refresh: bool = False) -> dict[str, GtArtifactLineage]:
    """Load the content-addressed registry, keyed by sha256.

    The registry is PRODUCED by ``experiments/ddm_gl1_gt_lineage_census.py`` from measurement.
    It is never hand-typed: a hand-typed lineage is exactly the undocumented claim pi2 flagged.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None and not refresh:
        return _REGISTRY_CACHE
    if not REGISTRY_PATH.exists():
        raise GtLineageUnknown(
            f"GT lineage registry missing at {REGISTRY_PATH}. Regenerate it with "
            "experiments/ddm_gl1_gt_lineage_census.py --emit-registry. Refusing to guess."
        )
    raw = json.loads(REGISTRY_PATH.read_text())
    out: dict[str, GtArtifactLineage] = {}
    for row in raw.get("artifacts", []):
        entry = GtArtifactLineage(
            sha256=row["sha256"],
            bytes=int(row.get("bytes", 0)),
            basename=row.get("basename", ""),
            lineage=row.get("lineage", UNKNOWN_UNCOMPARABLE),
            evidence=row.get("lineage_evidence", "NONE"),
            measurement=row.get("measurement", ""),
            claim_boundary=row.get("claim_boundary", ""),
            known_paths=tuple(row.get("known_paths", ())),
            known_basenames=tuple(
                row.get("known_basenames") or ([row["basename"]] if row.get("basename") else [])
            ),
        )
        out[entry.sha256] = entry
    _REGISTRY_CACHE = out
    return out


def lineage_of_file(path: str | Path, *, registry: dict[str, GtArtifactLineage] | None = None) -> GtArtifactLineage:
    """Return the recorded lineage of a file, BY CONTENT.

    Raises :class:`GtLineageUnknown` when the file is missing or its sha256 is unregistered.
    Refusing on an unregistered file is the point: an unrecorded lineage must never read as
    a pass, because that is indistinguishable from the state pi2 found.
    """
    p = Path(path)
    if not p.is_file():
        raise GtLineageUnknown(f"GT artifact does not exist: {p}")
    reg = registry if registry is not None else load_registry()
    digest = _cached_digest(p)
    entry = reg.get(digest)
    if entry is None:
        raise GtLineageUnknown(
            f"GT artifact {p} (sha256 {digest}) is NOT in the lineage registry. "
            "Its decode lineage is unrecorded, so no seg/pose claim may rest on it. "
            "Classify it with experiments/ddm_gl1_gt_lineage_census.py, or read a registered artifact."
        )
    return entry


def runtime_decode_lineage(name: str) -> str:
    """Return the lineage implied by a runtime decoder name.  Refuses on an unknown decoder."""
    lin = RUNTIME_DECODE_LINEAGE.get(name)
    if lin is None:
        raise GtLineageUnknown(
            f"runtime GT decoder {name!r} has no recorded lineage; known: "
            f"{sorted(RUNTIME_DECODE_LINEAGE)}"
        )
    return lin


def lineage_of_source(
    source: GtSource, *, registry: dict[str, GtArtifactLineage] | None = None
) -> str:
    """Resolve any :class:`GtSource` to a lineage string.  Fail-closed on both kinds."""
    if source.kind == "file":
        return lineage_of_file(source.ref, registry=registry).lineage
    if source.kind == "runtime_decode":
        return runtime_decode_lineage(source.ref)
    raise GtLineageUnknown(f"unknown GtSource kind {source.kind!r}")


def is_known_lineage(path: str | Path) -> bool:
    """Non-raising probe.  Use :func:`assert_gt_lineage` for anything load-bearing."""
    try:
        return lineage_of_file(path).is_resolved
    except GtLineageError:
        return False


def assert_gt_lineage(
    path: str | Path,
    *,
    required: str,
    instrument: str = "<unnamed instrument>",
    registry: dict[str, GtArtifactLineage] | None = None,
) -> GtArtifactLineage:
    """Assert that a GT artifact carries the decode lineage this instrument needs.

    This is the per-read guard.  It refuses -- never warns -- on all three failure modes:
    missing file, unregistered sha256 (lineage unknown), and lineage mismatch.

    Args:
        path: the ground-truth artifact about to be read.
        required: :data:`DALI_NVDEC` or :data:`PYAV_YUV420_TO_RGB`.  Use
            :data:`AUTHORITY_LINEAGE` when the measurement is meant to track the contest-CUDA row.
        instrument: caller name, quoted in the refusal so the offending reader is identifiable.

    Returns:
        The registry entry, so callers can record ``sha256``/``evidence`` in their receipts.

    Raises:
        GtLineageUnknown: file missing, or lineage unrecorded/unresolved.
        GtLineageMismatch: recorded lineage differs from ``required``.
    """
    if required not in RESOLVED_LINEAGES:
        raise ValueError(
            f"required lineage must be one of {sorted(RESOLVED_LINEAGES)}, got {required!r}"
        )
    entry = lineage_of_file(path, registry=registry)
    if not entry.is_resolved:
        raise GtLineageUnknown(
            f"{instrument}: GT artifact {path} has lineage {entry.lineage} "
            f"(evidence {entry.evidence}); it is not resolved to a decoder, so it cannot satisfy "
            f"a {required} requirement. {entry.claim_boundary}"
        )
    if entry.lineage != required:
        raise GtLineageMismatch(
            f"{instrument}: GT artifact {path} is {entry.lineage} lineage but this instrument "
            f"requires {required}. Reading the wrong lineage silently costs 1.4425x on d_seg and "
            f"+1.4061e-04 additive on d_pose (ddm_pi2, n600, 0.mkv) -- it reproduces the "
            f"contest-CPU axis while claiming otherwise. sha256={entry.sha256}"
        )
    return entry


def assert_single_lineage(
    sources: Sequence[GtSource],
    *,
    instrument: str,
    registry: dict[str, GtArtifactLineage] | None = None,
) -> str:
    """Refuse when one instrument's GT sources span more than one decode lineage.

    THIS is the pi2 bug written as a predicate.  pi2's instrument passed every per-read check it
    had, because each half was individually sane; only the SPAN was wrong.  A guard that checks
    reads one at a time cannot see that, which is why this predicate exists alongside
    :func:`assert_gt_lineage` rather than instead of it.

    Returns the single lineage on success.
    """
    if not sources:
        raise GtLineageUnknown(f"{instrument}: no GT sources declared; refusing to certify a span")
    resolved: dict[str, str] = {}
    for s in sources:
        resolved[f"{s.kind}:{s.ref}"] = lineage_of_source(s, registry=registry)
    span = set(resolved.values())
    if len(span) > 1:
        detail = "\n".join(f"    {k} -> {v}" for k, v in sorted(resolved.items()))
        raise GtLineageSplit(
            f"{instrument}: ground-truth sources span {len(span)} decode lineages {sorted(span)}. "
            f"This is the ddm_pi2 defect: one instrument, two ground truths.\n{detail}",
            lineages=tuple(sorted(span)),
            resolved=resolved,
        )
    return next(iter(span))


def basename_lineage_collisions(
    *, registry: dict[str, GtArtifactLineage] | None = None
) -> dict[str, dict[str, list[str]]]:
    """Filenames that resolve to MORE THAN ONE decode lineage, with the sha256 of each side.

    This is the measurement that makes a NAME-KEYED lineage rule visibly unsafe.  ``ddm_pi2``'s
    §0.3 says "keep using the cached ``gt_argmax_n600.npy`` -- DALI lineage", which is true of the
    file pi2 measured.  ``ddm_gl1`` measured seven files with that name across three distinct
    sha256, spanning BOTH lineages -- so an instrument can satisfy the rule by name and still read
    the wrong bytes.  Any surviving name-keyed rule must be re-keyed onto the sha256 reported here.
    """
    reg = registry if registry is not None else load_registry()
    by_name: dict[str, dict[str, list[str]]] = {}
    for entry in reg.values():
        if not entry.is_resolved:
            continue
        for name in entry.known_basenames or (entry.basename,):
            if not name:
                continue
            by_name.setdefault(name, {}).setdefault(entry.lineage, []).append(entry.sha256)
    return {
        name: {lin: sorted(shas) for lin, shas in sides.items()}
        for name, sides in sorted(by_name.items())
        if len(sides) > 1
    }


def population_split_report(
    instrument_sources: dict[str, Iterable[GtSource]] | None = None,
    *,
    registry: dict[str, GtArtifactLineage] | None = None,
) -> dict[str, Any]:
    """The gauge that does NOT zero out when this module's own cure is applied.

    It counts properties of REALITY -- how many decode lineages are present in the reachable GT
    population, and how many instruments read across more than one -- not properties of this
    module's instrumentation.  Adding lineage declarations moves both counts by exactly zero;
    only changing what an instrument READS moves them.
    """
    reg = registry if registry is not None else load_registry()
    tally: dict[str, int] = {}
    for entry in reg.values():
        tally[entry.lineage] = tally.get(entry.lineage, 0) + 1
    lineages_present = sorted({e.lineage for e in reg.values() if e.is_resolved})

    split_instruments: dict[str, list[str]] = {}
    for name, srcs in (instrument_sources or {}).items():
        try:
            assert_single_lineage(list(srcs), instrument=name, registry=reg)
        except GtLineageSplit as exc:
            split_instruments[name] = list(exc.lineages)
        except GtLineageError:
            split_instruments[name] = ["UNRESOLVED"]

    collisions = basename_lineage_collisions(registry=reg)
    return {
        "registered_artifacts": len(reg),
        "lineage_tally": tally,
        "basename_lineage_collisions": collisions,
        "basename_collision_count": len(collisions),
        "distinct_resolved_lineages_present": lineages_present,
        "distinct_resolved_lineage_count": len(lineages_present),
        "population_is_single_lineage": len(lineages_present) <= 1,
        "split_instruments": split_instruments,
        "gauge_note": (
            "Both counts are properties of the artifact population and of what instruments read. "
            "Applying this module's cure (adding lineage declarations) changes neither."
        ),
    }
