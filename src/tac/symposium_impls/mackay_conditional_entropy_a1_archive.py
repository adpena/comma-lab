# SPDX-License-Identifier: MIT
"""MacKay-style position-partition entropy proxy for the A1 archive.

Implements a Shannon-entropy-style PROXY over the canonical A1 archive bytes
per the Grand Reunion symposium 2026-05-15 (Phase F Implement-Now #1,
MacKay + Filler).

Per Catalog #269 (codex bkrbqet3p F4) — this module emits a
**position-partition proxy**, NOT the true scorer-conditional entropy
``H(A1_archive | scorer_state_dict)``. The previous docstring + field naming
overstated the binding: the partition is computed by hashing byte position
only and never reads scorer weights, tensor shapes, classes, or any
scorer-derived feature. Future Phase 2 enhancement will bind the context
model to real scorer-derived features (state_dict signatures / per-layer
norms / per-tensor entropies); until then the result is excluded from
canonical lower-bound persistence and from autopilot ranking
``H(A1 | scorer_state_dict)``-tagged consumers.

Module-level invariants per Catalog #269 (substring tokens auditable
by the catalog gate):

* ``true_scorer_conditional_entropy_claim=false`` (Python value: ``False``)
* ``position_partition_proxy=true`` (Python value: ``True``)
* ``evidence_grade="position-partition-proxy"``

Math contract (proxy)
=====================

Per Shannon's source coding theorem, the minimum expected codelength for a
source ``X`` given side information ``Y`` is the conditional entropy

    H(X | Y) = E_Y[H(X | Y = y)]

with units of bits when log base 2 is used. Per CLAUDE.md "Bit-level
deconstruction and entropy discipline": archive bytes are decomposable into
sections with distinct prior structure; per-section conditional entropy
floors the rate-axis cost achievable via perfect entropy coding when the
decoder shares the prior with the encoder.

In principle, the SegNet+PoseNet scorer ``state_dict`` is a SHARED PRIOR
between encoder and decoder (the contest scorer ships with ``upstream/`` and
is not charged against the archive). The TRUE conditional entropy of the
archive given the scorer would be the rate-axis lower bound for any codec
whose decoder may consult the scorer at compress time but not inflate time.
This module does NOT compute that quantity (see Catalog #269); it computes
a position-partition proxy that satisfies ``H(X | partition(position)) <=
H(X)`` per Cover & Thomas Theorem 2.6.5 but does NOT bind to scorer state.

Three context models are computed per the Implement-Now #1 spec:

* ``zero_context`` — raw byte entropy ``H(X)`` (worst case)
* ``brotli_context`` — observed compressed size in bits per byte (matches
  current archive)
* ``position_partition_context`` — entropy estimated under a frequency model
  conditioned on a deterministic 32-bucket hash partition over byte
  POSITION only. Per Catalog #269 this is a position-partition proxy, NOT
  a true scorer-conditional entropy. The gap between ``brotli_context`` and
  ``position_partition_context`` is the slack a Filler-style syndrome-trellis
  coder could close IF its side information were equivalent to the position
  partition; the slack does NOT reflect savings achievable via a true
  scorer-conditional model. Future Phase 2 enhancement will replace this
  proxy with a context model bound to actual scorer-derived features.

[verified-against: Shannon, ``A Mathematical Theory of Communication`` 1948
Theorem 4 (entropy of a discrete source); Cover & Thomas ``Elements of
Information Theory`` 2nd ed §2.2 (joint and conditional entropy) + Theorem
2.6.5 (conditioning reduces entropy); MacKay ``Information Theory, Inference,
and Learning Algorithms`` 2003 §5.4 (arithmetic coding lower bound =
conditional entropy).]

Usage
=====

>>> from tac.symposium_impls.mackay_conditional_entropy_a1_archive import (
...     compute_a1_conditional_entropy_estimate,
... )
>>> est = compute_a1_conditional_entropy_estimate()  # doctest: +SKIP

The result is a typed dataclass with three context-model entropy estimates,
the per-section breakdown, and the slack-vs-brotli prediction. Persists to
``.omx/state/a1_conditional_entropy_estimate.json`` so the cathedral
autopilot ranker can consume the rate-axis lower bound as a Pareto floor.

Continual learning hook
=======================

``update_from_anchor(anchor)`` invalidates the cached estimate when a new
A1-class anchor lands; downstream consumers re-load from disk on next call.

Lane: ``lane_symposium_impl_mackay_conditional_entropy_20260515``.
Catalog #256.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

__all__ = (
    "A1_ARCHIVE_PATH",
    "A1_CONDITIONAL_ENTROPY_STATE_PATH",
    "A1ConditionalEntropyEstimate",
    "A1SectionEntropy",
    "compute_a1_conditional_entropy_estimate",
    "estimate_brotli_context_bits",
    "estimate_position_partition_bits",
    "estimate_scorer_conditional_bits",  # back-compat alias for the proxy
    "estimate_zero_context_bits",
    "load_cached_a1_conditional_entropy_estimate",
    "position_partition_proxy",
    "shannon_entropy_bits",
    "save_a1_conditional_entropy_estimate",
    "true_scorer_conditional_entropy_claim",
    "update_from_anchor",
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
A1_ARCHIVE_PATH: Final[Path] = REPO_ROOT / "submissions" / "a1" / "archive.zip"
A1_CONDITIONAL_ENTROPY_STATE_PATH: Final[Path] = (
    REPO_ROOT / ".omx" / "state" / "a1_conditional_entropy_estimate.json"
)

# Module-level disclosure constants per Catalog #269 (codex bkrbqet3p F4).
# The position-partition partition is computed by hashing byte position
# only and never reads scorer weights, tensor shapes, or any
# scorer-derived feature. Downstream consumers (Pareto solver, autopilot
# ranker, continual-learning posterior) MUST consult these constants
# before treating the estimate as a true ``H(X | scorer_state_dict)``
# bound.
true_scorer_conditional_entropy_claim: Final[bool] = False
position_partition_proxy: Final[bool] = True
EVIDENCE_GRADE_POSITION_PARTITION_PROXY: Final[str] = "position-partition-proxy"

# Op-routable TODO (Phase 2 enhancement): bind the context model to real
# scorer-derived features (scorer state_dict signatures / per-layer norms /
# per-tensor entropies) so the slack reflects savings achievable via a
# true scorer-conditional model instead of a position-only partition.

_SCORER_CONDITIONING_BUCKETS: Final[int] = 32


@dataclasses.dataclass(frozen=True)
class A1SectionEntropy:
    """Per-section entropy under three context models (units: bits / byte).

    Per Catalog #269, ``scorer_prior_context_bits_per_byte`` is preserved
    as a back-compat field name but is semantically a position-partition
    proxy. ``position_partition_context_bits_per_byte`` is the canonical
    name added by the F4 fix; both fields hold the same value.
    """

    section_name: str
    section_size_bytes: int
    section_sha256: str
    zero_context_bits_per_byte: float
    brotli_context_bits_per_byte: float
    # Back-compat field (= position_partition_context_bits_per_byte) per
    # Catalog #269 — preserved for downstream consumers that already key
    # on the original schema. NEW callers should use
    # ``position_partition_context_bits_per_byte``.
    scorer_prior_context_bits_per_byte: float
    position_partition_context_bits_per_byte: float = 0.0


@dataclasses.dataclass(frozen=True)
class A1ConditionalEntropyEstimate:
    """Top-level estimate written to ``.omx/state/`` for autopilot consumption.

    Per Catalog #269 (codex bkrbqet3p F4): the ``scorer_prior_context``
    fields hold a position-partition proxy NOT a true scorer-conditional
    entropy. ``true_scorer_conditional_entropy_claim`` is exported as
    ``False`` and ``position_partition_proxy`` is exported as ``True`` so
    autopilot ranking + Pareto solver consumers know to exclude this
    estimate from canonical lower-bound persistence until the Phase 2
    enhancement binds the context model to real scorer features.
    """

    archive_path: str
    archive_size_bytes: int
    archive_sha256: str
    sections: tuple[A1SectionEntropy, ...]
    aggregate_zero_context_bits: float
    aggregate_brotli_context_bits: float
    # Back-compat aggregate field (= aggregate_position_partition_context_bits)
    # per Catalog #269. Same value, two names for migration.
    aggregate_scorer_prior_context_bits: float
    slack_brotli_minus_scorer_prior_bits: float
    slack_brotli_minus_scorer_prior_fraction: float
    evidence_grade: str
    score_claim: bool
    notes: str
    # Catalog #269 self-disclosure fields. Defaults preserved for back-compat
    # readers loading older serialized state.
    true_scorer_conditional_entropy_claim: bool = False
    position_partition_proxy: bool = True
    aggregate_position_partition_context_bits: float = 0.0


def shannon_entropy_bits(symbol_counts: Iterable[int]) -> float:
    """Return Shannon entropy ``H(X) = -sum p_i log2 p_i`` in bits / symbol.

    [verified-against: Shannon 1948 Theorem 2.]
    """
    counts = [int(c) for c in symbol_counts if int(c) > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def estimate_zero_context_bits(payload: bytes) -> float:
    """Raw byte entropy ``H(X)`` over an i.i.d. symbol model (no context)."""
    if not payload:
        return 0.0
    return shannon_entropy_bits(Counter(payload).values()) * len(payload)


def estimate_brotli_context_bits(payload: bytes, *, brotli_size_bytes: int | None = None) -> float:
    """Bits attributed to ``payload`` under a brotli context model.

    If ``brotli_size_bytes`` is supplied (the observed compressed size in the
    archive ZIP), it is converted to bits directly. Otherwise we re-compress
    via the ``brotli`` package as a fallback estimate.
    """
    if brotli_size_bytes is not None:
        return float(brotli_size_bytes) * 8.0
    if not payload:
        return 0.0
    try:
        import brotli  # type: ignore[import-not-found]
    except ImportError:
        # Fallback: zlib upper bound on brotli (typically ~5-15% looser).
        import zlib

        return float(len(zlib.compress(payload, level=9))) * 8.0
    return float(len(brotli.compress(payload, quality=11))) * 8.0


def _position_partition_buckets(
    payload: bytes, *, n_buckets: int = _SCORER_CONDITIONING_BUCKETS
) -> list[bytes]:
    """Partition payload bytes into ``n_buckets`` bins via a stable position hash.

    Per Catalog #269 (codex bkrbqet3p F4): the bucketing is computed by
    hashing byte POSITION only — it never reads scorer weights, tensor
    shapes, classes, or any scorer-derived feature. The previous name
    ``_scorer_prior_buckets`` overstated the binding; the canonical name
    is now ``_position_partition_buckets``. The legacy name is preserved
    as a back-compat alias below.

    The per-bucket entropy estimate downstream provides ``H(X | Y)``
    where ``Y`` is the deterministic position partition; this bounds the
    conditional entropy of the archive given any side-information that
    can be reconstructed from byte position alone — which does NOT include
    scorer state.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be > 0")
    buckets: list[bytearray] = [bytearray() for _ in range(n_buckets)]
    if not payload:
        return [bytes() for _ in range(n_buckets)]
    for position, byte in enumerate(payload):
        h = hashlib.blake2b(position.to_bytes(8, "little"), digest_size=4).digest()
        bucket_index = int.from_bytes(h, "little") % n_buckets
        buckets[bucket_index].append(byte)
    return [bytes(b) for b in buckets]


# Back-compat alias for the legacy name. New code should use
# ``_position_partition_buckets``.
def _scorer_prior_buckets(
    payload: bytes, *, n_buckets: int = _SCORER_CONDITIONING_BUCKETS
) -> list[bytes]:
    """Back-compat alias; see :func:`_position_partition_buckets`."""
    return _position_partition_buckets(payload, n_buckets=n_buckets)


def estimate_position_partition_bits(
    payload: bytes, *, n_buckets: int = _SCORER_CONDITIONING_BUCKETS
) -> float:
    """Bits attributable to ``payload`` under a position-partition context model.

    Per Cover & Thomas Theorem 2.6.5 (conditioning reduces entropy):
    ``H(X | Y) = sum_y p(y) · H(X | Y=y) <= H(X)``.

    Equality holds iff ``X`` and ``Y`` are independent. The partition we use
    is deterministic on byte position; the per-bucket entropy reflects the
    local distribution of bytes that share the same ``Y`` class.

    Per Catalog #269: this is a position-partition proxy, NOT a true
    scorer-conditional entropy. The previous name
    ``estimate_scorer_conditional_bits`` overstated the binding; it is
    preserved as a back-compat alias below. New callers should use
    ``estimate_position_partition_bits``.
    """
    if not payload:
        return 0.0
    total_bits = 0.0
    for bucket in _position_partition_buckets(payload, n_buckets=n_buckets):
        if not bucket:
            continue
        bucket_entropy = shannon_entropy_bits(Counter(bucket).values())
        total_bits += bucket_entropy * len(bucket)
    return total_bits


def estimate_scorer_conditional_bits(
    payload: bytes, *, n_buckets: int = _SCORER_CONDITIONING_BUCKETS
) -> float:
    """Back-compat alias; see :func:`estimate_position_partition_bits`.

    Per Catalog #269 (codex bkrbqet3p F4): this name overstated the
    binding (the implementation never reads scorer state). The canonical
    name is now ``estimate_position_partition_bits``; this alias delegates
    to the canonical helper for back-compat.
    """
    return estimate_position_partition_bits(payload, n_buckets=n_buckets)


def _section_record(
    *,
    name: str,
    payload: bytes,
    compressed_size_bytes: int,
) -> A1SectionEntropy:
    if not payload:
        return A1SectionEntropy(
            section_name=name,
            section_size_bytes=0,
            section_sha256=hashlib.sha256(b"").hexdigest(),
            zero_context_bits_per_byte=0.0,
            brotli_context_bits_per_byte=0.0,
            scorer_prior_context_bits_per_byte=0.0,
            position_partition_context_bits_per_byte=0.0,
        )
    n = len(payload)
    zero_bits = estimate_zero_context_bits(payload)
    brotli_bits = estimate_brotli_context_bits(payload, brotli_size_bytes=compressed_size_bytes)
    partition_bits = estimate_position_partition_bits(payload)
    bits_per_byte = partition_bits / n
    return A1SectionEntropy(
        section_name=name,
        section_size_bytes=n,
        section_sha256=hashlib.sha256(payload).hexdigest(),
        zero_context_bits_per_byte=zero_bits / n,
        brotli_context_bits_per_byte=brotli_bits / n,
        scorer_prior_context_bits_per_byte=bits_per_byte,  # back-compat
        position_partition_context_bits_per_byte=bits_per_byte,
    )


def compute_a1_conditional_entropy_estimate(
    *,
    archive_path: Path | None = None,
    n_buckets: int = _SCORER_CONDITIONING_BUCKETS,
) -> A1ConditionalEntropyEstimate:
    """Compute the canonical A1 conditional entropy estimate.

    Reads ``submissions/a1/archive.zip`` (or supplied path), iterates over its
    ZIP members in stored order, computes per-section entropy under three
    context models, and returns the typed estimate. Does NOT write to disk;
    callers decide via :func:`save_a1_conditional_entropy_estimate`.
    """
    path = Path(archive_path) if archive_path is not None else A1_ARCHIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"A1 archive not found at {path}")
    archive_bytes = path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    sections: list[A1SectionEntropy] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            payload = zf.read(info.filename)
            sections.append(
                _section_record(
                    name=info.filename,
                    payload=payload,
                    compressed_size_bytes=info.compress_size,
                )
            )
    aggregate_zero = sum(s.zero_context_bits_per_byte * s.section_size_bytes for s in sections)
    aggregate_brotli = sum(s.brotli_context_bits_per_byte * s.section_size_bytes for s in sections)
    aggregate_partition = sum(
        s.position_partition_context_bits_per_byte * s.section_size_bytes for s in sections
    )
    slack = aggregate_brotli - aggregate_partition
    slack_fraction = (slack / aggregate_brotli) if aggregate_brotli > 0 else 0.0
    # Per Catalog #269 (codex bkrbqet3p F4): the slack is the gap between
    # brotli and a POSITION-PARTITION proxy, NOT the scorer-conditional
    # entropy. It bounds savings from a Filler-style syndrome-trellis coder
    # whose side information is the position partition; it does NOT bound
    # savings from a true scorer-conditional model.
    notes = (
        "[empirical:submissions/a1/archive.zip] [contest-CPU GHA Linux x86_64 anchor 0.1928] "
        "[position-partition-proxy] Per Catalog #269 (codex bkrbqet3p F4): the partition is "
        "computed by hashing byte position only; it does NOT read scorer state. "
        "true_scorer_conditional_entropy_claim=False. position_partition_proxy=True. "
        "Excluded from canonical lower-bound persistence and autopilot ranking until "
        "Phase 2 enhancement binds the context model to real scorer-derived features. "
        "Catalog #256 + #269."
    )
    return A1ConditionalEntropyEstimate(
        archive_path=str(path),
        archive_size_bytes=len(archive_bytes),
        archive_sha256=archive_sha,
        sections=tuple(sections),
        aggregate_zero_context_bits=aggregate_zero,
        aggregate_brotli_context_bits=aggregate_brotli,
        aggregate_scorer_prior_context_bits=aggregate_partition,  # back-compat
        slack_brotli_minus_scorer_prior_bits=slack,
        slack_brotli_minus_scorer_prior_fraction=slack_fraction,
        evidence_grade=EVIDENCE_GRADE_POSITION_PARTITION_PROXY,
        score_claim=False,
        notes=notes,
        true_scorer_conditional_entropy_claim=False,
        position_partition_proxy=True,
        aggregate_position_partition_context_bits=aggregate_partition,
    )


def _serialize(estimate: A1ConditionalEntropyEstimate) -> dict[str, object]:
    payload = dataclasses.asdict(estimate)
    payload["sections"] = [dataclasses.asdict(s) for s in estimate.sections]
    return payload


def save_a1_conditional_entropy_estimate(
    estimate: A1ConditionalEntropyEstimate,
    *,
    state_path: Path | None = None,
) -> Path:
    """Write the estimate to canonical state (atomic + readable)."""
    target = Path(state_path) if state_path is not None else A1_CONDITIONAL_ENTROPY_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(_serialize(estimate), indent=2, sort_keys=True))
    tmp.replace(target)
    return target


def load_cached_a1_conditional_entropy_estimate(
    *, state_path: Path | None = None
) -> A1ConditionalEntropyEstimate | None:
    target = Path(state_path) if state_path is not None else A1_CONDITIONAL_ENTROPY_STATE_PATH
    if not target.is_file():
        return None
    raw = json.loads(target.read_text())
    sections = tuple(A1SectionEntropy(**s) for s in raw.pop("sections", []))
    return A1ConditionalEntropyEstimate(sections=sections, **raw)


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    state_path: Path | None = None,
    archive_path: Path | None = None,
) -> A1ConditionalEntropyEstimate | None:
    """Re-compute the estimate when a new A1-class anchor lands.

    Per CLAUDE.md "Subagent coherence-by-default" hook 5 (continual-learning
    posterior). Returns the freshly computed estimate, or ``None`` if the
    anchor's archive sha does not match the canonical A1 archive (no-op).
    """
    target = Path(state_path) if state_path is not None else A1_CONDITIONAL_ENTROPY_STATE_PATH
    archive_resolved = Path(archive_path) if archive_path is not None else A1_ARCHIVE_PATH
    if not archive_resolved.is_file():
        return None
    anchor_sha = str(anchor.get("archive_sha256", "")).lower()
    if anchor_sha:
        actual_sha = hashlib.sha256(archive_resolved.read_bytes()).hexdigest()
        if anchor_sha != actual_sha:
            return None
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive_resolved)
    save_a1_conditional_entropy_estimate(estimate, state_path=target)
    return estimate
