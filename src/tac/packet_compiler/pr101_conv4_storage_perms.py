"""PR101 GOLD ``conv4_storage_perms`` per-tensor 4D-axis permutation primitive.

This module ports the **4D-axis permutation primitive** from the PR101
GOLD-medal submission (``submissions/hnerv_ft_microcodec/src/codec.py``,
lines 38-55) into a typed, golden-vector-backed transducer.

Mechanism (PR101 source): ::

    CONV4_STORAGE_PERMS = {
        2:  (3, 0, 2, 1),
        4:  (3, 0, 2, 1),
        6:  (0, 1, 2, 3),
        8:  (3, 0, 1, 2),
        10: (3, 0, 2, 1),
        12: (3, 0, 1, 2),
        14: (1, 0, 2, 3),
        16: (3, 0, 2, 1),
        18: (1, 0, 2, 3),
        20: (0, 3, 2, 1),
        22: (0, 3, 2, 1),
        24: (0, 2, 3, 1),
        26: (0, 1, 3, 2),
    }
    CONV4_INVERSE_PERMS = {
        idx: tuple(np.argsort(perm)) for idx, perm in CONV4_STORAGE_PERMS.items()
    }

For each 4D conv weight tensor at index ``idx``, the encoder transposes
the (out, in, H, W) shape via a hand-tuned 4-tuple axis permutation
BEFORE flattening to bytes for INT8 zigzag encoding. The permutation
makes spatially-adjacent weights MORE temporally adjacent in the byte
stream, improving brotli's LZ77 dictionary hits.

The inverse permutation (``CONV4_INVERSE_PERMS[idx] = argsort(perm)``)
is applied at decode time to restore the original tensor shape.

The reusable primitive here is the **mechanism** (per-tensor 4D-axis
permutation with auto-computed inverse), NOT the specific PR101 table.
The optimal permutation per tensor must be derived offline via 24-perm
exhaustive search over the post-quantisation entropy.

The transducer here exposes:

* :class:`Conv4StoragePermSchema` — validated per-tensor (idx ->
  permutation) table with built-in invariant checks.
* :func:`compute_inverse_perms` — pure inverse-permutation computation
  (argsort) without numpy dependency on input.
* :func:`apply_storage_perm` — apply a 4-tuple axis permutation to a
  bytes blob representing a 4D tensor's flattened weights.
* :func:`apply_inverse_perm` — inverse operation.

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
* CONV4_STORAGE_PERMS covers 13 of PR101's 28 tensors (the 4D conv
  weights). The 13 entries are at indices [2, 4, 6, 8, 10, 12, 14, 16,
  18, 20, 22, 24, 26].

target_substrate_hint
=====================

``hnerv_lc_family`` — specifically the PR95-derived HNeRV-LC tensor
layout (28-tensor state_dict where indices [2, 4, 6, 8, 10, 12, 14, 16,
18, 20, 22, 24, 26] correspond to the 4D conv weight tensors).

This primitive does NOT directly compose onto ``sane_hnerv`` (the
α substrate) or other renderers without per-architecture derivation of
which tensor indices are 4D conv weights AND what their optimal
permutations are. The MECHANISM is reusable; the SPECIFIC TABLE is not.

predicted_ev_per_byte
=====================

* Rank: ``#5 EV/byte`` at PR106 r2 frontier per
  ``.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json``.
* Basis: ``[predicted; PR101 GOLD byte trace; not yet measured on
  internal substrate]`` — per-tensor axis perm traces to PR101 GOLD
  medal archive bytes; estimated ~100-300 bytes per archive from
  better brotli dictionary hits.

CLAUDE.md compliance
====================

* No scorer load — pure stdlib + numpy (optional, only for shape ops).
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; permutation table invariants checked on construction.
* OSS-friendly: public surface is the 5 names re-exported from
  ``tac.packet_compiler``.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only.

[empirical:src/tac/packet_compiler/golden_vectors/pr101_conv4_storage_perms_v1.json]
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


# ── PR101 canonical anchor table (for golden vector + reference only) ──────


PR101_CONV4_STORAGE_PERMS: Mapping[int, tuple[int, int, int, int]] = MappingProxyType({
    2:  (3, 0, 2, 1),
    4:  (3, 0, 2, 1),
    6:  (0, 1, 2, 3),
    8:  (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
})
"""The exact 13-entry per-tensor permutation table PR101 uses.

This is **anchor data**, NOT a default. Downstream consumers must derive
their own table per architecture via offline 24-perm exhaustive search
over the post-quantisation byte distribution per conv tensor.
"""


# ── Public schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Conv4StoragePermSchema:
    """A per-tensor 4D-axis permutation table with built-in invariants.

    Attributes
    ----------
    perms:
        Mapping ``tensor_index -> (4-tuple axis permutation)``. Each
        permutation must be a permutation of ``(0, 1, 2, 3)``.
    inverse_perms:
        Auto-computed mapping ``tensor_index -> argsort(perm)`` at
        construction time. Stored as a frozen view.

    Notes
    -----
    The validation invariants (strict 4-tuple permutation per entry)
    match PR101's wire-format contract exactly. Violations raise
    ``ValueError`` at construction time so downstream encoders see
    structural bugs at table-definition time, not at runtime.
    """

    perms: Mapping[int, tuple[int, int, int, int]]
    inverse_perms: Mapping[int, tuple[int, int, int, int]]

    @classmethod
    def from_perms(
        cls,
        perms: Mapping[int, tuple[int, int, int, int]],
    ) -> "Conv4StoragePermSchema":
        """Build a schema, validating each entry and auto-computing inverses.

        Parameters
        ----------
        perms:
            Mapping ``tensor_index -> (4-tuple)``. Each tuple must be a
            permutation of ``(0, 1, 2, 3)``.

        Raises
        ------
        ValueError
            On any invalid entry (non-int key, non-4-tuple, non-permutation).
        TypeError
            On non-mapping input.
        """
        if not isinstance(perms, Mapping):
            raise TypeError(
                f"perms must be a Mapping; got {type(perms)!r}"
            )
        validated: dict[int, tuple[int, int, int, int]] = {}
        inverses: dict[int, tuple[int, int, int, int]] = {}
        for k, v in perms.items():
            if not isinstance(k, int) or isinstance(k, bool):
                raise ValueError(
                    f"perms key must be int; got {type(k)!r}"
                )
            if k < 0:
                raise ValueError(
                    f"perms key must be >= 0; got {k}"
                )
            if not isinstance(v, tuple):
                raise ValueError(
                    f"perms[{k}] must be tuple; got {type(v)!r}"
                )
            if len(v) != 4:
                raise ValueError(
                    f"perms[{k}] must be 4-tuple; got length {len(v)}"
                )
            for j, axis in enumerate(v):
                if not isinstance(axis, int) or isinstance(axis, bool):
                    raise ValueError(
                        f"perms[{k}][{j}] must be int; got {type(axis)!r}"
                    )
            if sorted(v) != [0, 1, 2, 3]:
                raise ValueError(
                    f"perms[{k}] must be a permutation of (0, 1, 2, 3); "
                    f"got {v}"
                )
            normalised: tuple[int, int, int, int] = (
                int(v[0]), int(v[1]), int(v[2]), int(v[3])
            )
            validated[k] = normalised
            inverses[k] = compute_inverse_perm(normalised)

        return cls(
            perms=MappingProxyType(validated),
            inverse_perms=MappingProxyType(inverses),
        )


# ── Transducers ────────────────────────────────────────────────────────────


def compute_inverse_perm(
    perm: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Compute the inverse permutation of a 4-tuple via argsort semantics.

    Parameters
    ----------
    perm:
        A permutation of ``(0, 1, 2, 3)`` as a 4-tuple of ints.

    Returns
    -------
    tuple[int, int, int, int]
        The inverse permutation. Satisfies ``compute_inverse_perm(perm) ==
        tuple(np.argsort(perm))``.

    Raises
    ------
    ValueError
        On non-permutation input.
    """
    if not isinstance(perm, tuple) or len(perm) != 4:
        raise ValueError(
            f"perm must be 4-tuple; got {perm!r}"
        )
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"perm must be a permutation of (0, 1, 2, 3); got {perm}"
        )
    inv = [0, 0, 0, 0]
    for i, axis in enumerate(perm):
        inv[axis] = i
    return (inv[0], inv[1], inv[2], inv[3])


def compute_inverse_perms(
    perms: Mapping[int, tuple[int, int, int, int]],
) -> Mapping[int, tuple[int, int, int, int]]:
    """Apply :func:`compute_inverse_perm` to every entry of a table.

    Parameters
    ----------
    perms:
        Mapping ``tensor_index -> (4-tuple permutation)``.

    Returns
    -------
    Mapping[int, tuple[int, int, int, int]]
        Frozen view of the inverse table.

    Raises
    ------
    ValueError
        On invalid input.
    """
    return MappingProxyType(
        {k: compute_inverse_perm(v) for k, v in perms.items()}
    )


def apply_storage_perm(
    tensor_bytes: bytes,
    shape: tuple[int, int, int, int],
    perm: tuple[int, int, int, int],
) -> bytes:
    """Apply a 4D-axis permutation to a flattened tensor's bytes.

    Parameters
    ----------
    tensor_bytes:
        The flattened tensor as bytes. Must have length ``prod(shape) * 1``
        (we assume INT8 / UINT8 quantised tensors per PR101 convention).
    shape:
        The 4D shape ``(out, in, H, W)`` of the tensor.
    perm:
        The 4-tuple axis permutation.

    Returns
    -------
    bytes
        The permuted tensor bytes (``np.transpose(reshape(shape), perm)``
        flattened back to bytes).

    Raises
    ------
    ValueError
        On length / shape mismatch.
    """
    if not isinstance(tensor_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"tensor_bytes must be bytes-like; got {type(tensor_bytes)!r}"
        )
    if not isinstance(shape, tuple) or len(shape) != 4:
        raise ValueError(f"shape must be 4-tuple; got {shape!r}")
    for j, d in enumerate(shape):
        if not isinstance(d, int) or isinstance(d, bool) or d < 0:
            raise ValueError(
                f"shape[{j}] must be non-negative int; got {d!r}"
            )
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"perm must be a permutation of (0, 1, 2, 3); got {perm}"
        )
    expected_len = shape[0] * shape[1] * shape[2] * shape[3]
    if len(tensor_bytes) != expected_len:
        raise ValueError(
            f"tensor_bytes length {len(tensor_bytes)} != "
            f"prod(shape)={expected_len} for shape={shape}"
        )
    arr = np.frombuffer(bytes(tensor_bytes), dtype=np.uint8).reshape(shape)
    permuted = np.transpose(arr, perm).copy()
    return permuted.tobytes()


def apply_inverse_perm(
    permuted_bytes: bytes,
    stored_shape: tuple[int, int, int, int],
    inverse_perm: tuple[int, int, int, int],
) -> bytes:
    """Invert :func:`apply_storage_perm` to restore original tensor bytes.

    Parameters
    ----------
    permuted_bytes:
        The permuted tensor as bytes (output of
        :func:`apply_storage_perm`).
    stored_shape:
        The 4D shape AS STORED (i.e. after permutation:
        ``tuple(shape[i] for i in perm)``).
    inverse_perm:
        The inverse permutation (from :func:`compute_inverse_perm`).

    Returns
    -------
    bytes
        The original tensor bytes.

    Raises
    ------
    ValueError
        On length / shape mismatch.
    """
    if not isinstance(permuted_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"permuted_bytes must be bytes-like; got {type(permuted_bytes)!r}"
        )
    if not isinstance(stored_shape, tuple) or len(stored_shape) != 4:
        raise ValueError(
            f"stored_shape must be 4-tuple; got {stored_shape!r}"
        )
    if sorted(inverse_perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"inverse_perm must be a permutation of (0, 1, 2, 3); "
            f"got {inverse_perm}"
        )
    expected_len = (
        stored_shape[0] * stored_shape[1] * stored_shape[2] * stored_shape[3]
    )
    if len(permuted_bytes) != expected_len:
        raise ValueError(
            f"permuted_bytes length {len(permuted_bytes)} != "
            f"prod(stored_shape)={expected_len}"
        )
    arr = np.frombuffer(bytes(permuted_bytes), dtype=np.uint8).reshape(
        stored_shape
    )
    restored = np.transpose(arr, inverse_perm).copy()
    return restored.tobytes()


__all__ = [
    "Conv4StoragePermSchema",
    "PR101_CONV4_STORAGE_PERMS",
    "apply_inverse_perm",
    "apply_storage_perm",
    "compute_inverse_perm",
    "compute_inverse_perms",
]
