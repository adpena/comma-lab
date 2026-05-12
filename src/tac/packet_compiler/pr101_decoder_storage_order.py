"""PR101 GOLD ``decoder_storage_order`` permutation + split-brotli boundaries.

This module ports the **storage-order permutation primitive** from the PR101
GOLD-medal submission (``submissions/hnerv_ft_microcodec/src/codec.py``,
lines 32-36) into a typed, golden-vector-backed transducer.

Mechanism (PR101 source): ::

    DECODER_STORAGE_ORDER = (
        14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
        18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
    )
    DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)

PR101's HNeRV-LC decoder has 28 tensors in its canonical state_dict()
iteration order. The encoder REORDERS these tensors before quantisation +
brotli compression, putting highest-entropy tensor groups NEXT TO EACH
OTHER in the byte stream. ``DECODER_STREAM_ENDS`` then partitions the
concatenated bytes into MULTIPLE BROTLI STREAMS so low-entropy tensor
groups get their own stream (which compresses much better than one big
stream with mixed-entropy regions).

The reusable primitive here is the **grammar** (permutation list + stream
boundary list), NOT the specific PR101 permutation. The permutation
itself is hand-tuned per architecture via offline gradient-free search
over the post-quantisation byte distribution. Future ports against
non-HNeRV-LC architectures must derive their own ``storage_order`` and
``stream_ends`` via entropy clustering.

The transducer here exposes:

* :class:`DecoderStorageOrderSchema` — validated (permutation, stream_ends)
  pair with built-in invariant checks.
* :func:`reorder_tensors_for_storage` — apply a storage order to a list
  of byte-tensors and emit the concatenated buffer.
* :func:`restore_tensor_order_from_storage` — invert the permutation.
* :func:`partition_buffer_by_stream_ends` — slice a concatenated buffer
  into the per-stream segments according to ``stream_ends``.

Per CLAUDE.md "HNeRV parity discipline" lesson 2 (Export-first design):
this is a GRAMMAR primitive, NOT a method claim. A downstream packet
compiler must consume this primitive end-to-end against a verified
substrate before any score claim is made.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src/codec.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake;
Catalog #109).

PR101 GOLD anchor data
======================

* PR101 archive score: ``0.193 [contest-CUDA]`` (public claim; not replayed
  internally yet)
* PR101 archive bytes: 162,164 (DECODER_BLOB_LEN) + 15,387 (LATENT_BLOB_LEN)
  + sidecar.
* PR101 uses 7 split-brotli streams (len(DECODER_STREAM_ENDS) == 7) over
  the concatenated 28-tensor zigzag-byte buffer.

target_substrate_hint
=====================

``hnerv_lc_family`` — specifically the PR95-derived HNeRV-LC tensor layout
(28-tensor state_dict, single-video memorisation, 28-dim per-pair latent,
PixelShuffle decoder). This primitive does NOT directly compose onto:

* ``sane_hnerv`` (the new α substrate per
  ``.omx/research/grand_council_fields_medal_substrate_design_20260512.md``)
  because sane_hnerv's tensor layout differs from PR101's HNeRV-LC.
* Non-HNeRV renderers (PSD / Quantizr / SegMap), which have different
  state_dict iteration orders. A new ``storage_order`` permutation
  must be derived per architecture.

predicted_ev_per_byte
=====================

* Rank: ``#2 EV/byte`` at PR106 r2 frontier per
  ``.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json``.
* Basis: ``[predicted; PR101 GOLD byte trace; not yet measured on
  internal substrate]`` — derived from offline analysis of PR101's
  0.193 [contest-CUDA] archive bytes; the split-brotli boundaries
  trace specific byte savings vs. a single-stream baseline.
* est_bytes_saved per archive: ``~100-300 bytes`` (rough envelope from
  PR101 anchor; actual savings depend on the tensor entropy distribution
  of the target substrate).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; permutation/stream-ends invariants checked on
  construction.
* OSS-friendly: public surface is the 4 names re-exported from
  ``tac.packet_compiler``.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only. ``score_claim``, ``promotion_eligible``,
  ``ready_for_exact_eval_dispatch`` remain permanently False per
  ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``.

[empirical:src/tac/packet_compiler/golden_vectors/pr101_decoder_storage_order_v1.json]
"""

from __future__ import annotations

from dataclasses import dataclass


# ── PR101 canonical anchor values (for golden vector + reference only) ──────

PR101_DECODER_STORAGE_ORDER: tuple[int, ...] = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
"""The exact 28-tensor permutation PR101 uses.

This is **anchor data**, NOT a default. Downstream consumers must derive
their own permutation per architecture via offline entropy clustering.
Exposed here so the golden vector can verify byte-exact match against
the PR101 source.
"""

PR101_DECODER_STREAM_ENDS: tuple[int, ...] = (1, 2, 22, 23, 26, 27, 28)
"""The exact 7 split-brotli stream boundaries PR101 uses.

Each value is a position into the storage-order sequence at which a
brotli stream ends (exclusive). Together with PR101's storage order,
this produces 7 separate brotli streams over the concatenated zigzag-
byte buffer:

* stream 0: positions [0, 1)   — 1 tensor
* stream 1: positions [1, 2)   — 1 tensor
* stream 2: positions [2, 22)  — 20 tensors
* stream 3: positions [22, 23) — 1 tensor
* stream 4: positions [23, 26) — 3 tensors
* stream 5: positions [26, 27) — 1 tensor
* stream 6: positions [27, 28) — 1 tensor
"""


# ── Public schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecoderStorageOrderSchema:
    """A (permutation, stream_ends) pair with built-in invariants.

    Attributes
    ----------
    storage_order:
        A permutation of ``range(n_tensors)`` as a tuple of ints. The
        encoder writes tensors in this order; the decoder reads in this
        order. Must be a valid permutation (no repeats, all values in
        [0, n_tensors)).
    stream_ends:
        Brotli stream boundaries as positions into the storage-order
        sequence. The last value must equal ``n_tensors``. Values must
        be strictly increasing in [1, n_tensors].
    n_tensors:
        Number of tensors (derived from storage_order length).

    Notes
    -----
    The validation invariants (strict permutation + strictly-increasing
    boundaries with final == n_tensors) match PR101's wire-format
    contract exactly. Violations raise ``ValueError`` at construction
    time so downstream encoders see structural bugs at table-definition
    time, not at runtime.
    """

    storage_order: tuple[int, ...]
    stream_ends: tuple[int, ...]
    n_tensors: int

    def __post_init__(self) -> None:
        # Validate storage_order is a permutation of range(n_tensors).
        n = self.n_tensors
        if not isinstance(self.storage_order, tuple):
            raise ValueError(
                f"storage_order must be tuple; got {type(self.storage_order)!r}"
            )
        if len(self.storage_order) != n:
            raise ValueError(
                f"storage_order length {len(self.storage_order)} != "
                f"n_tensors {n}"
            )
        if sorted(self.storage_order) != list(range(n)):
            raise ValueError(
                f"storage_order must be a permutation of range({n}); "
                f"got {self.storage_order}"
            )

        # Validate stream_ends.
        if not isinstance(self.stream_ends, tuple):
            raise ValueError(
                f"stream_ends must be tuple; got {type(self.stream_ends)!r}"
            )
        if len(self.stream_ends) == 0:
            raise ValueError("stream_ends must be non-empty")
        prev = 0
        for i, end in enumerate(self.stream_ends):
            if not isinstance(end, int) or isinstance(end, bool):
                raise ValueError(
                    f"stream_ends[{i}] must be int; got {type(end)!r}"
                )
            if end <= prev:
                raise ValueError(
                    f"stream_ends must be strictly increasing; "
                    f"stream_ends[{i}]={end} <= prev={prev}"
                )
            if end > n:
                raise ValueError(
                    f"stream_ends[{i}]={end} > n_tensors={n}"
                )
            prev = end
        if self.stream_ends[-1] != n:
            raise ValueError(
                f"stream_ends[-1] must equal n_tensors; "
                f"got {self.stream_ends[-1]} vs n_tensors={n}"
            )


# ── Transducers ────────────────────────────────────────────────────────────


def reorder_tensors_for_storage(
    tensors: list[bytes] | tuple[bytes, ...],
    schema: DecoderStorageOrderSchema,
) -> bytes:
    """Concatenate ``tensors`` according to ``schema.storage_order``.

    Parameters
    ----------
    tensors:
        A list/tuple of byte-tensors (each already quantised + per-tensor
        sign-mapped to uint8). Length must equal ``schema.n_tensors``.
    schema:
        The storage-order schema.

    Returns
    -------
    bytes
        Concatenation of ``tensors[schema.storage_order[0]] || tensors[
        schema.storage_order[1]] || ...``.

    Raises
    ------
    ValueError
        On length mismatch or non-bytes entries.
    """
    if len(tensors) != schema.n_tensors:
        raise ValueError(
            f"tensors length {len(tensors)} != n_tensors {schema.n_tensors}"
        )
    parts: list[bytes] = []
    for pos, idx in enumerate(schema.storage_order):
        entry = tensors[idx]
        if not isinstance(entry, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"tensors[{idx}] (position {pos}) must be bytes-like; "
                f"got {type(entry)!r}"
            )
        parts.append(bytes(entry))
    return b"".join(parts)


def restore_tensor_order_from_storage(
    reordered: list[bytes] | tuple[bytes, ...],
    schema: DecoderStorageOrderSchema,
) -> tuple[bytes, ...]:
    """Invert :func:`reorder_tensors_for_storage` on a list of per-tensor blobs.

    Parameters
    ----------
    reordered:
        A list/tuple of byte-tensors in STORAGE order (i.e. ``reordered[k]``
        is the tensor that was at position ``k`` in storage). Length must
        equal ``schema.n_tensors``.
    schema:
        The storage-order schema.

    Returns
    -------
    tuple[bytes, ...]
        Tensors in their original (state_dict-iteration-order) positions.

    Raises
    ------
    ValueError
        On length mismatch or non-bytes entries.
    """
    if len(reordered) != schema.n_tensors:
        raise ValueError(
            f"reordered length {len(reordered)} != n_tensors {schema.n_tensors}"
        )
    out: list[bytes | None] = [None] * schema.n_tensors
    for pos, idx in enumerate(schema.storage_order):
        entry = reordered[pos]
        if not isinstance(entry, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"reordered[{pos}] must be bytes-like; got {type(entry)!r}"
            )
        out[idx] = bytes(entry)
    # All slots filled because storage_order is a permutation.
    return tuple(out)  # type: ignore[arg-type]


def partition_buffer_by_stream_ends(
    buffer: bytes,
    schema: DecoderStorageOrderSchema,
    per_tensor_sizes: list[int] | tuple[int, ...],
) -> tuple[bytes, ...]:
    """Slice a concatenated buffer into per-brotli-stream segments.

    Parameters
    ----------
    buffer:
        The concatenated byte buffer (output of
        :func:`reorder_tensors_for_storage` or compatible).
    schema:
        The storage-order schema.
    per_tensor_sizes:
        Sizes (in bytes) of each tensor IN STORAGE ORDER. Length must
        equal ``schema.n_tensors``. ``sum(per_tensor_sizes)`` must equal
        ``len(buffer)``.

    Returns
    -------
    tuple[bytes, ...]
        Per-stream byte segments, one per element of ``schema.stream_ends``.
        ``len(result) == len(schema.stream_ends)``.

    Raises
    ------
    ValueError
        On size mismatch.
    """
    if len(per_tensor_sizes) != schema.n_tensors:
        raise ValueError(
            f"per_tensor_sizes length {len(per_tensor_sizes)} != "
            f"n_tensors {schema.n_tensors}"
        )
    total = sum(int(s) for s in per_tensor_sizes)
    if total != len(buffer):
        raise ValueError(
            f"sum(per_tensor_sizes)={total} != len(buffer)={len(buffer)}"
        )
    for i, s in enumerate(per_tensor_sizes):
        if not isinstance(s, int) or isinstance(s, bool) or s < 0:
            raise ValueError(
                f"per_tensor_sizes[{i}] must be non-negative int; got {s!r}"
            )

    # cumulative byte offsets at each position-in-storage-order boundary.
    cumulative_bytes: list[int] = [0]
    running = 0
    for s in per_tensor_sizes:
        running += int(s)
        cumulative_bytes.append(running)

    segments: list[bytes] = []
    prev_pos = 0
    for end_pos in schema.stream_ends:
        start_byte = cumulative_bytes[prev_pos]
        end_byte = cumulative_bytes[end_pos]
        segments.append(buffer[start_byte:end_byte])
        prev_pos = end_pos
    return tuple(segments)


__all__ = [
    "PR101_DECODER_STORAGE_ORDER",
    "PR101_DECODER_STREAM_ENDS",
    "DecoderStorageOrderSchema",
    "partition_buffer_by_stream_ends",
    "reorder_tensors_for_storage",
    "restore_tensor_order_from_storage",
]
