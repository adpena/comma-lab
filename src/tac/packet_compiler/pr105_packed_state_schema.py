"""PR105 ``kitchen_sink`` packed-state-schema size-sorted helper.

This module extracts the REUSABLE size-sort helper from the PR105 public
submission (``submissions/kitchen_sink/src/codec.py``, line 58) into a typed
transducer.

The mechanism (PR105 source): ::

    PACKED_STATE_SCHEMA = sorted(
        FIXED_STATE_SCHEMA,
        key=lambda item: -int(np.prod(item[1])),
    )

PR105's HNeRV-family archive encodes each tensor in the state-dict as
``(uint8 zigzag-quantised body, fp32 scale)`` pairs. The naive layout
interleaves the per-tensor bodies with their per-tensor scales — bytes
that compress poorly together. PR105's PACKED_STATE_SCHEMA reorders the
tensors so all bodies (which share statistical properties) are emitted
adjacent in the brotli stream, followed by all scales (which are also
adjacent fp32). The size-sort key (largest tensor first) is empirically
defensible: brotli's long-range matches benefit from the highest-entropy
streams appearing first while the entropy model is still building.

The reusable primitive here is the **size-sort key function** — given a
list of (name, shape) tuples, return the PACKED_STATE_SCHEMA ordering
PR105 uses. The downstream consumer is responsible for actually encoding
the body+scale pairs against the sorted schema (PR105 emits all bodies
then all scales — see ``decode_packed_decoder`` lines 78-95).

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr105_intake_20260505_auto/source/submissions/kitchen_sink/src/codec.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; the size-sort is bit-exact on the
  ``pr105_packed_state_schema_v1`` golden vector.
* Pure functional helper — no global mutable state.

[empirical:src/tac/packet_compiler/golden_vectors/pr105_packed_state_schema_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedStateSchemaEntry:
    """One entry in the PR105 packed-state schema.

    Attributes
    ----------
    name:
        Tensor name (e.g. ``"blocks.5.weight"``).
    shape:
        Tensor shape as a tuple of ints.
    n_elements:
        ``int(np.prod(shape))`` — number of elements (used as sort key).
    """

    name: str
    shape: tuple[int, ...]
    n_elements: int


def pack_state_schema_size_sorted(
    schema: list[tuple[str, tuple[int, ...]]]
    | tuple[tuple[str, tuple[int, ...]], ...],
) -> tuple[PackedStateSchemaEntry, ...]:
    """Sort a ``(name, shape)`` schema by descending ``np.prod(shape)``.

    Mirrors PR105's ``PACKED_STATE_SCHEMA = sorted(..., key=lambda item:
    -int(np.prod(item[1])))`` exactly: largest tensors first, ties broken
    by Python's stable sort (preserving the input order for equal sizes).

    Parameters
    ----------
    schema:
        Iterable of ``(name, shape)`` tuples. ``shape`` must be a tuple
        of non-negative ints. Empty shape ``()`` is permitted (scalar
        tensors have ``n_elements = 1`` per numpy convention).

    Returns
    -------
    tuple[PackedStateSchemaEntry, ...]
        Schema entries sorted by descending ``n_elements``; ties broken
        by input order (Python stable sort).

    Raises
    ------
    ValueError
        On non-tuple shape, non-int shape element, or negative shape
        element.
    TypeError
        On non-string name.
    """
    entries: list[PackedStateSchemaEntry] = []
    for i, item in enumerate(schema):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(
                f"schema entry {i} must be a (name, shape) tuple; got {item!r}"
            )
        name, shape = item
        if not isinstance(name, str):
            raise TypeError(
                f"schema entry {i} name must be str; got {type(name)!r}"
            )
        if not isinstance(shape, tuple):
            raise ValueError(
                f"schema entry {i} shape must be tuple; got {type(shape)!r}"
            )
        for j, dim in enumerate(shape):
            if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool):
                raise ValueError(
                    f"schema entry {i} shape[{j}] must be int; got {type(dim)!r}"
                )
            if int(dim) < 0:
                raise ValueError(
                    f"schema entry {i} shape[{j}] must be >= 0; got {dim}"
                )
        # numpy.prod of empty shape returns 1.0 (scalar tensor convention).
        n_elements = int(np.prod(shape)) if len(shape) > 0 else 1
        entries.append(
            PackedStateSchemaEntry(
                name=name,
                shape=tuple(int(d) for d in shape),
                n_elements=n_elements,
            )
        )
    # PR105 uses Python's stable sort with key=-n_elements — ties preserve
    # input order, which is the contract we want.
    entries.sort(key=lambda e: -e.n_elements)
    return tuple(entries)


__all__ = [
    "PackedStateSchemaEntry",
    "pack_state_schema_size_sorted",
]
