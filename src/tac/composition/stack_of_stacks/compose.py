# SPDX-License-Identifier: MIT
"""Stack-of-stacks composition byte format + composer API.

This module implements Idea 3 from the beat-PR95 design memo. The composer
operates as a deterministic byte-level builder that wraps existing
primitives — it never re-implements substrate codecs.

Wire format (single ZIP member ``x`` in archive.zip)
====================================================

The composition archive concatenates:

1. Base substrate bytes (e.g. A1 178,162 B with sidecars, or sane_hnerv) —
   immutable, passed through verbatim per "Apples-to-apples evidence
   discipline".
2. Inner-stack inner sidecars (φ1 SABOR + φ3 S2SBS + score-grad residual),
   each with its own magic-byte trailer (``SAB1``, ``SHF1``, ``SGR1``).
3. Middle-stack cross-substrate selector + alt-substrate base bytes
   (concatenated; D4.B style); selector tells inflate which arm rendered.
4. Outer-stack ensemble selector (``SOS1`` magic at the END of the archive
   bytes) — 1 byte per pair telling inflate which checkpoint to use.

Layout of the ``SOS1`` trailer::

    magic       : 4 bytes ASCII ``"SOS1"``
    version     : uint8 (== 1)
    layer_mask  : uint8 (bit0=inner, bit1=middle, bit2=outer)
    n_pairs     : uint16 LE (== 600 for full contest, smaller for smoke)
    n_arms      : uint8 (K outer-stack arms, in {1, 2, 3})
    arm_meta_len: uint16 LE (length of per-arm metadata JSON, brotli)
    reserved    : 2 bytes (= 0)
    per_pair_arm: n_pairs bytes (each ∈ [0, n_arms-1])
    arm_meta    : brotli-compressed JSON describing each arm's grammar

Per CLAUDE.md "HNeRV parity discipline" lesson 4 (≤ 100 LOC inflate,
≤ 200 with explicit substrate-engineering exemption), the inflate-time
runtime that decodes this format lives at
:mod:`tac.composition.stack_of_stacks.inflate`. The composer itself
(this file) is training/build-time only.

Per CLAUDE.md Catalog #146 (Phase 1 trainer runtime contract):
the inflate runtime sister emits the contest-compliant 3-arg
``inflate.sh archive_dir output_dir file_list`` interface.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import brotli

SCHEMA_VERSION = 1
SOS_SIDECAR_MAGIC = b"SOS1"
SOS_SIDECAR_VERSION = 1
# magic(4s) + version(B) + layer_mask(B) + n_pairs(H) + n_arms(B)
# + arm_meta_len(H) + reserved(2s)
SOS_HEADER_STRUCT = struct.Struct("<4sBBHBH2s")

# Inner-stack sub-sidecar magics (sub-sections within the SOS1 envelope).
SABOR_BOUNDARY_MAGIC = b"SAB1"
S2SBS_HF_MAGIC = b"SHF1"
SCORE_GRAD_RESIDUAL_MAGIC = b"SGR1"

# Layer mask bits (see Idea 3 docstring).
LAYER_BIT_INNER = 1 << 0
LAYER_BIT_MIDDLE = 1 << 1
LAYER_BIT_OUTER = 1 << 2

# Closed-form contest-CPU sensitivity weights at the PR106 r2 operating
# point per CLAUDE.md "SegNet vs PoseNet importance — operating-point
# dependent" + design memo §6. These drive the score-aware mixing rule.
CONTEST_SEG_MARGINAL = 100.0
CONTEST_POSE_MARGINAL_PR106 = 271.0  # at pose_avg ~ 3.4e-5
CONTEST_RATE_MARGINAL = 25.0 / 37_545_489.0  # 25 / video bytes per CLAUDE.md

# Hard caps from CLAUDE.md HNeRV parity discipline lesson 7 (bolt-on ≤ 350
# LOC) + design memo §3 byte-budget envelope.
MAX_OUTER_ARMS = 3
MAX_N_PAIRS = 65_535  # uint16 LE
DEFAULT_N_PAIRS_CONTEST = 600


class StackOfStacksError(ValueError):
    """Raised when a stack-of-stacks compose spec violates byte-budget or
    grammar constraints."""


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryAtomSpec:
    """φ1 SABOR boundary-atom layer spec.

    The SABOR audit (2026-05-13) measured per-frame argmax-stable interior
    capacity in the 14.6 - 263 MB range; we typically reserve a small
    fraction (≤ 8 KB) for boundary-mask atoms that the renderer can use as
    a free-byte channel.

    Args:
        capacity_bytes: bytes reserved from the SABOR free-byte channel.
            Must be ≤ ``audit_capacity_bytes``.
        audit_capacity_bytes: measured capacity from the SABOR audit JSON
            (per-frame stable-interior bytes summed over frames).
        atom_payload: optional pre-encoded payload bytes to stuff into the
            boundary channel. None → zero-init (no-op).
    """

    capacity_bytes: int = 0
    audit_capacity_bytes: int = 0
    atom_payload: bytes | None = None

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise StackOfStacksError(
                f"BoundaryAtomSpec.capacity_bytes must be >= 0; got {self.capacity_bytes}"
            )
        if self.capacity_bytes > self.audit_capacity_bytes:
            raise StackOfStacksError(
                f"BoundaryAtomSpec.capacity_bytes ({self.capacity_bytes}) exceeds "
                f"measured audit_capacity_bytes ({self.audit_capacity_bytes}); "
                "refusing to compose beyond the SABOR audit's empirical capacity "
                "per CLAUDE.md 'Apples-to-apples evidence discipline'"
            )
        if self.atom_payload is not None and len(self.atom_payload) > self.capacity_bytes:
            raise StackOfStacksError(
                f"BoundaryAtomSpec.atom_payload length ({len(self.atom_payload)}) "
                f"exceeds capacity_bytes ({self.capacity_bytes})"
            )


@dataclass(frozen=True)
class HFSidecarSpec:
    """φ3 S2SBS HF byte-stuffing sidecar spec.

    The S2SBS audit (2026-05-13) measured ~38 MB post-ECC capacity in the
    high-frequency blindspot of the scorer pre-processing (the bilinear
    192×256 → 384×512 stretch). We typically reserve a small slice (≤ 8 KB)
    for substrate auxiliary state.

    Args:
        capacity_bytes: bytes to stuff. Must be ≤ ``audit_capacity_bytes``.
        audit_capacity_bytes: measured capacity from the S2SBS audit JSON.
        hf_payload: optional pre-encoded payload to stuff.
    """

    capacity_bytes: int = 0
    audit_capacity_bytes: int = 0
    hf_payload: bytes | None = None

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise StackOfStacksError(
                f"HFSidecarSpec.capacity_bytes must be >= 0; got {self.capacity_bytes}"
            )
        if self.capacity_bytes > self.audit_capacity_bytes:
            raise StackOfStacksError(
                f"HFSidecarSpec.capacity_bytes ({self.capacity_bytes}) exceeds "
                f"measured audit_capacity_bytes ({self.audit_capacity_bytes})"
            )
        if self.hf_payload is not None and len(self.hf_payload) > self.capacity_bytes:
            raise StackOfStacksError(
                f"HFSidecarSpec.hf_payload length ({len(self.hf_payload)}) "
                f"exceeds capacity_bytes ({self.capacity_bytes})"
            )


@dataclass(frozen=True)
class ResidualSpec:
    """Score-gradient-aware residual layer spec.

    The residual is a tiny int8-quantized correction tensor stored in a
    sub-sidecar (magic ``SGR1``). Per Catalog #123, the SALIENCY for which
    parameters to coarsen is determined by the SCORE GRADIENT, NOT the
    weight magnitude.

    Args:
        residual_int8_bytes: pre-quantized int8 residual payload bytes (the
            caller is responsible for the actual quantization).
        scale: float32 inverse-scale to recover floats at inflate time
            (q / scale ≈ real).
    """

    residual_int8_bytes: bytes = b""
    scale: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.residual_int8_bytes, (bytes, bytearray)):
            raise StackOfStacksError(
                "ResidualSpec.residual_int8_bytes must be bytes/bytearray"
            )
        if self.scale <= 0.0 or not math.isfinite(self.scale):
            raise StackOfStacksError(
                f"ResidualSpec.scale must be positive finite; got {self.scale}"
            )


@dataclass(frozen=True)
class InnerStackSpec:
    """Inner-stack spec: substrate base + per-substrate sidecars.

    Args:
        substrate_id: short string id (e.g. "a1", "sane_hnerv"). Used for
            arm-meta JSON only; the composer does NOT validate against a
            registry of substrate ids (caller knows what they pass).
        base_bytes: the base-substrate archive bytes (verbatim immutable).
        boundary_atom_spec: optional φ1 SABOR boundary atom layer.
        hf_sidecar_spec: optional φ3 S2SBS HF byte-stuffing layer.
        residual_spec: optional score-gradient-aware residual layer.
    """

    substrate_id: str
    base_bytes: bytes
    boundary_atom_spec: BoundaryAtomSpec | None = None
    hf_sidecar_spec: HFSidecarSpec | None = None
    residual_spec: ResidualSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.substrate_id, str) or not self.substrate_id:
            raise StackOfStacksError(
                f"InnerStackSpec.substrate_id must be non-empty str; got {self.substrate_id!r}"
            )
        if not isinstance(self.base_bytes, (bytes, bytearray)) or not self.base_bytes:
            raise StackOfStacksError(
                "InnerStackSpec.base_bytes must be non-empty bytes/bytearray"
            )

    def total_added_bytes(self) -> int:
        """Bytes added by this inner stack on top of base_bytes.

        Per-layer header is 6 B (4 magic + 1 version + 1 reserved) plus a
        uint32-LE length prefix (4 B), so 10 B overhead per active layer.
        """
        total = 0
        if self.boundary_atom_spec is not None and self.boundary_atom_spec.capacity_bytes > 0:
            total += _INNER_SUB_HEADER_BYTES + self.boundary_atom_spec.capacity_bytes
        if self.hf_sidecar_spec is not None and self.hf_sidecar_spec.capacity_bytes > 0:
            total += _INNER_SUB_HEADER_BYTES + self.hf_sidecar_spec.capacity_bytes
        if self.residual_spec is not None and len(self.residual_spec.residual_int8_bytes) > 0:
            total += _INNER_SUB_HEADER_BYTES + 4 + len(self.residual_spec.residual_int8_bytes)
            # 4 extra bytes for the float32 scale prefix
        return total


@dataclass(frozen=True)
class MiddleStackSpec:
    """Middle-stack spec: cross-substrate composition over a list of inners.

    The middle stack runs each inner_specs[k] arm through its own renderer
    at inflate time and selects per-pair which arm to use. The selector is
    a uint8 per pair stored in the outer-stack sidecar (or, if the outer
    stack is disabled, in this stack's selector).

    Args:
        inner_specs: ordered tuple of InnerStackSpec (length 1..K). Each
            inner is rendered independently; selection happens at the
            outer or middle layer.
        rate_budget_partition: optional dict mapping ``substrate_id`` to a
            byte budget; the composer asserts each inner's total_added_bytes
            stays within its share. None disables partition enforcement.
        score_aware_mixing: if True, the middle stack stores per-pair-best
            selector based on the caller's `per_pair_score_arm` argument.
            If False, the middle stack picks arm 0 unconditionally.
    """

    inner_specs: tuple[InnerStackSpec, ...]
    rate_budget_partition: dict[str, int] | None = None
    score_aware_mixing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.inner_specs, tuple) or len(self.inner_specs) == 0:
            raise StackOfStacksError(
                "MiddleStackSpec.inner_specs must be a non-empty tuple"
            )
        if len(self.inner_specs) > MAX_OUTER_ARMS:
            raise StackOfStacksError(
                f"MiddleStackSpec.inner_specs length must be ≤ {MAX_OUTER_ARMS}; "
                f"got {len(self.inner_specs)}"
            )
        if self.rate_budget_partition is not None:
            for inner in self.inner_specs:
                if inner.substrate_id in self.rate_budget_partition:
                    budget = self.rate_budget_partition[inner.substrate_id]
                    actual = inner.total_added_bytes()
                    if actual > budget:
                        raise StackOfStacksError(
                            f"MiddleStackSpec: inner substrate {inner.substrate_id!r} "
                            f"added {actual} B but budget is {budget} B"
                        )


@dataclass(frozen=True)
class OuterStackSpec:
    """Outer-stack spec: K-checkpoint ensemble selector.

    Args:
        k: number of checkpoint arms (1..MAX_OUTER_ARMS). k=1 disables
            ensemble (single arm), k>=2 enables per-pair selection.
        per_pair_arm: tuple of length n_pairs giving the arm index for
            each pair (0 ≤ arm < k). Empty tuple → arm 0 for all pairs.
        temperatures: optional tuple of Langevin temperatures used to
            train each arm (informational only; stored in arm_meta).
    """

    k: int = 1
    per_pair_arm: tuple[int, ...] = field(default_factory=tuple)
    temperatures: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.k < 1 or self.k > MAX_OUTER_ARMS:
            raise StackOfStacksError(
                f"OuterStackSpec.k must be in [1, {MAX_OUTER_ARMS}]; got {self.k}"
            )
        if self.per_pair_arm:
            if any(a < 0 or a >= self.k for a in self.per_pair_arm):
                raise StackOfStacksError(
                    f"OuterStackSpec.per_pair_arm entries must be in [0, {self.k}); "
                    f"got {self.per_pair_arm[:5]}..."
                )
            if len(self.per_pair_arm) > MAX_N_PAIRS:
                raise StackOfStacksError(
                    f"OuterStackSpec.per_pair_arm length must be ≤ {MAX_N_PAIRS}; "
                    f"got {len(self.per_pair_arm)}"
                )


# Inner-stack sub-sidecar header: 4 B magic + 1 B version + 1 B reserved
# + 4 B uint32-LE length prefix.
_INNER_SUB_HEADER_STRUCT = struct.Struct("<4sBBI")
_INNER_SUB_HEADER_BYTES = _INNER_SUB_HEADER_STRUCT.size


# ---------------------------------------------------------------------------
# Inner stack
# ---------------------------------------------------------------------------


class InnerStack:
    """Composes one substrate base + its sidecar atoms into bytes.

    The inner stack does NOT touch the base bytes — they pass through
    verbatim — and only APPENDS sidecars in a deterministic order:
    boundary atom (SABOR), HF sidecar (S2SBS), residual (score-grad). Any
    of the three may be absent; the layer_mask bits track which were
    included.
    """

    def __init__(self, spec: InnerStackSpec) -> None:
        self.spec = spec

    def compose(self) -> tuple[bytes, dict[str, Any]]:
        """Return (composed_bytes, meta) for this inner stack.

        ``composed_bytes`` is the base bytes followed by sub-sidecars in
        a deterministic order. ``meta`` is a JSON-serializable dict that
        describes the layer offsets + sizes for inflate-time consumption.
        """
        chunks: list[bytes] = [bytes(self.spec.base_bytes)]
        layers: list[dict[str, Any]] = []
        base_len = len(self.spec.base_bytes)
        cursor = base_len

        if self.spec.boundary_atom_spec is not None:
            sub = self._pack_boundary_atom(self.spec.boundary_atom_spec)
            if sub:
                chunks.append(sub)
                layers.append(
                    {
                        "kind": "sabor_boundary",
                        "magic": SABOR_BOUNDARY_MAGIC.decode("ascii"),
                        "offset": cursor,
                        "length": len(sub),
                    }
                )
                cursor += len(sub)

        if self.spec.hf_sidecar_spec is not None:
            sub = self._pack_hf_sidecar(self.spec.hf_sidecar_spec)
            if sub:
                chunks.append(sub)
                layers.append(
                    {
                        "kind": "s2sbs_hf",
                        "magic": S2SBS_HF_MAGIC.decode("ascii"),
                        "offset": cursor,
                        "length": len(sub),
                    }
                )
                cursor += len(sub)

        if self.spec.residual_spec is not None:
            sub = self._pack_residual(self.spec.residual_spec)
            if sub:
                chunks.append(sub)
                layers.append(
                    {
                        "kind": "score_grad_residual",
                        "magic": SCORE_GRAD_RESIDUAL_MAGIC.decode("ascii"),
                        "offset": cursor,
                        "length": len(sub),
                    }
                )
                cursor += len(sub)

        composed = b"".join(chunks)
        meta = {
            "substrate_id": self.spec.substrate_id,
            "base_length": base_len,
            "total_length": len(composed),
            "added_bytes": len(composed) - base_len,
            "sub_layers": layers,
        }
        return composed, meta

    @staticmethod
    def _pack_sub(magic: bytes, payload: bytes) -> bytes:
        """Pack a sub-sidecar with the canonical header."""
        if len(payload) == 0:
            return b""
        if len(magic) != 4:
            raise StackOfStacksError(f"sub-sidecar magic must be 4 bytes; got {magic!r}")
        header = _INNER_SUB_HEADER_STRUCT.pack(
            magic, 1, 0, len(payload)
        )
        return header + payload

    @classmethod
    def _pack_boundary_atom(cls, spec: BoundaryAtomSpec) -> bytes:
        if spec.capacity_bytes == 0:
            return b""
        payload = spec.atom_payload if spec.atom_payload is not None else b"\x00" * spec.capacity_bytes
        # Pad/truncate to the declared capacity_bytes so the inflate-time
        # reader can locate the bytes deterministically.
        if len(payload) < spec.capacity_bytes:
            payload = payload + b"\x00" * (spec.capacity_bytes - len(payload))
        elif len(payload) > spec.capacity_bytes:
            payload = payload[: spec.capacity_bytes]
        return cls._pack_sub(SABOR_BOUNDARY_MAGIC, payload)

    @classmethod
    def _pack_hf_sidecar(cls, spec: HFSidecarSpec) -> bytes:
        if spec.capacity_bytes == 0:
            return b""
        payload = spec.hf_payload if spec.hf_payload is not None else b"\x00" * spec.capacity_bytes
        if len(payload) < spec.capacity_bytes:
            payload = payload + b"\x00" * (spec.capacity_bytes - len(payload))
        elif len(payload) > spec.capacity_bytes:
            payload = payload[: spec.capacity_bytes]
        return cls._pack_sub(S2SBS_HF_MAGIC, payload)

    @classmethod
    def _pack_residual(cls, spec: ResidualSpec) -> bytes:
        if len(spec.residual_int8_bytes) == 0:
            return b""
        # 4 bytes float32 LE scale + the int8 payload.
        scale_bytes = struct.pack("<f", float(spec.scale))
        return cls._pack_sub(
            SCORE_GRAD_RESIDUAL_MAGIC, scale_bytes + bytes(spec.residual_int8_bytes)
        )


# ---------------------------------------------------------------------------
# Middle stack
# ---------------------------------------------------------------------------


class MiddleStack:
    """Composes multiple inner stacks into one cross-substrate bundle.

    The middle stack runs each inner_specs[k] arm and concatenates their
    composed bytes. The selector (per-pair-best) lives in the outer-stack
    sidecar (or, if no outer stack is requested, is recorded here).
    """

    def __init__(self, spec: MiddleStackSpec) -> None:
        self.spec = spec
        self._inner_stacks: tuple[InnerStack, ...] = tuple(
            InnerStack(inner) for inner in spec.inner_specs
        )

    def compose_arms(self) -> tuple[list[bytes], list[dict[str, Any]]]:
        """Compose each arm into bytes + meta.

        Returns ``(arm_bytes_list, arm_meta_list)`` where each list has
        length ``len(self.spec.inner_specs)``.
        """
        arm_bytes: list[bytes] = []
        arm_meta: list[dict[str, Any]] = []
        for inner_stack in self._inner_stacks:
            b, m = inner_stack.compose()
            arm_bytes.append(b)
            arm_meta.append(m)
        return arm_bytes, arm_meta

    def total_bytes(self) -> int:
        """Sum of all arms' total composed bytes."""
        arm_bytes_list, _ = self.compose_arms()
        return sum(len(b) for b in arm_bytes_list)


# ---------------------------------------------------------------------------
# Outer stack
# ---------------------------------------------------------------------------


class OuterStack:
    """Composes the K-checkpoint ensemble selector + alt-arm payloads.

    For K=1 the outer stack adds no payload bytes (just a trivial header
    indicating single-arm so the inflate runtime knows not to look for a
    selector). For K>=2 the outer stack adds n_pairs bytes (the selector)
    plus the brotli-compressed arm_meta JSON.
    """

    def __init__(self, spec: OuterStackSpec, n_pairs: int) -> None:
        if n_pairs <= 0 or n_pairs > MAX_N_PAIRS:
            raise StackOfStacksError(
                f"OuterStack.n_pairs must be in [1, {MAX_N_PAIRS}]; got {n_pairs}"
            )
        if spec.per_pair_arm and len(spec.per_pair_arm) != n_pairs:
            raise StackOfStacksError(
                f"OuterStackSpec.per_pair_arm length ({len(spec.per_pair_arm)}) "
                f"!= n_pairs ({n_pairs})"
            )
        self.spec = spec
        self.n_pairs = n_pairs

    def pack(self, arm_meta: list[dict[str, Any]], layer_mask: int) -> bytes:
        """Pack the outer-stack SOS1 trailer.

        ``arm_meta`` is the list returned by :meth:`MiddleStack.compose_arms`.
        ``layer_mask`` is the bit-set declaring which layers are active.
        """
        # Per-pair selector (n_pairs bytes; default arm 0 for all if empty).
        selector = bytes(self.spec.per_pair_arm) if self.spec.per_pair_arm else b"\x00" * self.n_pairs

        # Brotli-compressed JSON describing each arm's grammar.
        sos_meta = {
            "schema_version": SCHEMA_VERSION,
            "k": self.spec.k,
            "n_pairs": self.n_pairs,
            "layer_mask": layer_mask,
            "temperatures": list(self.spec.temperatures),
            "arms": arm_meta,
        }
        meta_json = json.dumps(sos_meta, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        meta_brotli = brotli.compress(meta_json, quality=11)
        if len(meta_brotli) > 0xFFFF:
            raise StackOfStacksError(
                f"arm_meta brotli length ({len(meta_brotli)}) > 65535; reduce arm_meta payload"
            )

        header = SOS_HEADER_STRUCT.pack(
            SOS_SIDECAR_MAGIC,
            SOS_SIDECAR_VERSION,
            layer_mask & 0xFF,
            self.n_pairs,
            self.spec.k,
            len(meta_brotli),
            b"\x00\x00",
        )
        return header + selector + meta_brotli


# ---------------------------------------------------------------------------
# Top-level compose + decompose APIs
# ---------------------------------------------------------------------------


def compose_stack_of_stacks(
    *,
    middle_stack_spec: MiddleStackSpec,
    outer_stack_spec: OuterStackSpec | None = None,
    n_pairs: int = DEFAULT_N_PAIRS_CONTEST,
    max_total_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Top-level stack-of-stacks composer.

    Args:
        middle_stack_spec: cross-substrate composition (1..3 arms).
        outer_stack_spec: K-checkpoint ensemble selector. If None, K=1
            single-arm; the arm-meta JSON will indicate this.
        n_pairs: number of pairs in the contest video (600 for full).
        max_total_bytes: optional hard cap on the composed archive bytes.
            Raises StackOfStacksError if the result exceeds this cap.

    Returns:
        ``(composed_bytes, meta)`` where composed_bytes is the SINGLE-ZIP-
        member ``x`` blob to be zipped into the contest archive, and meta
        is a JSON-serializable dict capturing every layer's offset/length.
    """
    middle_stack = MiddleStack(middle_stack_spec)
    arm_bytes_list, arm_meta_list = middle_stack.compose_arms()

    # Concatenate the arms (D4.B style); arm 0 is the "primary" arm whose
    # base substrate carries the archive grammar; the alternative arms'
    # base bytes are appended verbatim and selected per-pair at inflate.
    arm_concat = b"".join(arm_bytes_list)

    # Build layer mask.
    layer_mask = 0
    has_inner_sub_layers = any(
        len(meta.get("sub_layers", [])) > 0 for meta in arm_meta_list
    )
    if has_inner_sub_layers:
        layer_mask |= LAYER_BIT_INNER
    if len(arm_bytes_list) > 1:
        layer_mask |= LAYER_BIT_MIDDLE
    if outer_stack_spec is not None and outer_stack_spec.k > 1:
        layer_mask |= LAYER_BIT_OUTER

    # Outer-stack SOS1 trailer.
    effective_outer = outer_stack_spec if outer_stack_spec is not None else OuterStackSpec(k=1)
    outer_stack = OuterStack(effective_outer, n_pairs=n_pairs)

    # Annotate each arm meta with its offset within the arm_concat blob.
    arm_offsets: list[int] = []
    cursor = 0
    for arm_bytes, meta in zip(arm_bytes_list, arm_meta_list, strict=True):
        meta["arm_offset"] = cursor
        meta["arm_length"] = len(arm_bytes)
        arm_offsets.append(cursor)
        cursor += len(arm_bytes)

    sos_trailer = outer_stack.pack(arm_meta_list, layer_mask)
    composed = arm_concat + sos_trailer

    if max_total_bytes is not None and len(composed) > max_total_bytes:
        raise StackOfStacksError(
            f"composed stack-of-stacks total bytes ({len(composed)}) exceeds "
            f"max_total_bytes ({max_total_bytes})"
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "n_pairs": n_pairs,
        "k": effective_outer.k,
        "layer_mask": layer_mask,
        "total_bytes": len(composed),
        "arm_concat_bytes": len(arm_concat),
        "sos_trailer_bytes": len(sos_trailer),
        "arms": arm_meta_list,
        "arm_offsets": arm_offsets,
    }
    return composed, meta


def decompose_stack_of_stacks(composed_bytes: bytes) -> dict[str, Any]:
    """Inverse of :func:`compose_stack_of_stacks` (decode-only).

    Locates the SOS1 trailer at the end of ``composed_bytes`` and returns
    a structured dict describing the layer offsets/lengths. Does NOT
    decode the underlying substrate bases (those are caller-specific).

    Raises:
        StackOfStacksError: composed_bytes does not contain a valid SOS1
            trailer at the end.
    """
    if len(composed_bytes) < SOS_HEADER_STRUCT.size:
        raise StackOfStacksError(
            f"composed bytes too short ({len(composed_bytes)} B) to contain SOS1 trailer"
        )

    # The SOS1 trailer is at the END; scan backwards for the magic.
    idx = composed_bytes.rfind(SOS_SIDECAR_MAGIC)
    if idx < 0:
        raise StackOfStacksError("no SOS1 magic found in composed bytes")
    if idx + SOS_HEADER_STRUCT.size > len(composed_bytes):
        raise StackOfStacksError("SOS1 magic found but header is truncated")
    try:
        magic, version, layer_mask, n_pairs, k, meta_len, _reserved = (
            SOS_HEADER_STRUCT.unpack_from(composed_bytes, idx)
        )
    except struct.error as exc:
        raise StackOfStacksError(f"failed to unpack SOS1 header: {exc}") from exc
    if magic != SOS_SIDECAR_MAGIC:
        raise StackOfStacksError(f"unexpected magic: {magic!r}")
    if version != SOS_SIDECAR_VERSION:
        raise StackOfStacksError(f"unsupported SOS1 version: {version}")

    selector_start = idx + SOS_HEADER_STRUCT.size
    selector_end = selector_start + n_pairs
    meta_end = selector_end + meta_len
    if meta_end != len(composed_bytes):
        raise StackOfStacksError(
            f"SOS1 trailer length mismatch: expected_end={meta_end} "
            f"composed_len={len(composed_bytes)}"
        )
    selector = composed_bytes[selector_start:selector_end]
    meta_brotli = composed_bytes[selector_end:meta_end]
    try:
        meta_json = brotli.decompress(meta_brotli)
    except Exception as exc:
        raise StackOfStacksError(f"failed to brotli-decompress arm_meta: {exc}") from exc
    try:
        sos_meta = json.loads(meta_json.decode("utf-8"))
    except Exception as exc:
        raise StackOfStacksError(f"failed to parse arm_meta JSON: {exc}") from exc

    arm_concat_bytes = composed_bytes[:idx]
    return {
        "schema_version": sos_meta.get("schema_version", SCHEMA_VERSION),
        "k": k,
        "n_pairs": n_pairs,
        "layer_mask": layer_mask,
        "arm_concat_bytes": arm_concat_bytes,
        "selector": selector,
        "arm_meta": sos_meta,
        "trailer_offset": idx,
        "trailer_length": len(composed_bytes) - idx,
    }


# ---------------------------------------------------------------------------
# Composition rules
# ---------------------------------------------------------------------------


def score_aware_mixing_weights(
    seg_per_pair_arms: list[list[float]],
    pose_per_pair_arms: list[list[float]],
    rate_per_arm_bytes: list[int],
) -> list[int]:
    """Return per-pair best-of-K selector ∈ {0, ..., k-1} given component deltas.

    Operating-point-aware per CLAUDE.md "SegNet vs PoseNet importance —
    operating-point dependent":

    * At PR106 r2 frontier (pose_avg ~ 3.4e-5), pose marginal is 2.71× SegNet.
    * Per-pair contest contribution for arm k =
        100 * seg_pair_arm + sqrt(10 * pose_pair_arm) + (25 * rate_arm / video_bytes)

    Args:
        seg_per_pair_arms: ``[n_pairs][k]`` — per-pair, per-arm SegNet
            distortion estimates.
        pose_per_pair_arms: ``[n_pairs][k]`` — per-pair, per-arm PoseNet
            distortion estimates.
        rate_per_arm_bytes: ``[k]`` — bytes added by each arm (rate term
            is global to the archive, so it's per-arm not per-pair).

    Returns:
        Length-n_pairs list of arm indices (each ∈ ``[0, k)``).
    """
    if not seg_per_pair_arms or not pose_per_pair_arms:
        raise StackOfStacksError(
            "score_aware_mixing_weights requires non-empty per-pair arrays"
        )
    if len(seg_per_pair_arms) != len(pose_per_pair_arms):
        raise StackOfStacksError(
            "seg_per_pair_arms and pose_per_pair_arms must have equal n_pairs"
        )
    k = len(rate_per_arm_bytes)
    if k == 0:
        raise StackOfStacksError("rate_per_arm_bytes must be non-empty")

    selector: list[int] = []
    for pair_idx, (seg_arms, pose_arms) in enumerate(
        zip(seg_per_pair_arms, pose_per_pair_arms, strict=True)
    ):
        if len(seg_arms) != k or len(pose_arms) != k:
            raise StackOfStacksError(
                f"pair {pair_idx}: seg/pose arm counts ({len(seg_arms)}, "
                f"{len(pose_arms)}) must equal k ({k})"
            )
        best_arm = 0
        best_contrib = math.inf
        for k_idx in range(k):
            pose_contrib = math.sqrt(max(0.0, 10.0 * float(pose_arms[k_idx])))
            contrib = (
                CONTEST_SEG_MARGINAL * float(seg_arms[k_idx])
                + pose_contrib
                + CONTEST_RATE_MARGINAL * float(rate_per_arm_bytes[k_idx])
            )
            if contrib < best_contrib:
                best_contrib = contrib
                best_arm = k_idx
        selector.append(best_arm)
    return selector


def validate_byte_budget(
    middle_stack_spec: MiddleStackSpec,
    *,
    base_substrate_bytes: int,
    max_total_bytes: int,
    n_pairs: int = DEFAULT_N_PAIRS_CONTEST,
    outer_stack_spec: OuterStackSpec | None = None,
) -> dict[str, Any]:
    """Predict the composed archive size without actually composing.

    Useful for ranking compose specs before paying the cost of brotli
    compressing arm_meta. Returns a dict with ``predicted_total_bytes``
    and per-layer byte breakdowns.

    Raises:
        StackOfStacksError: predicted_total_bytes exceeds max_total_bytes.
    """
    arm_total = 0
    arm_breakdown: list[dict[str, Any]] = []
    for inner_spec in middle_stack_spec.inner_specs:
        added = inner_spec.total_added_bytes()
        per_arm = len(inner_spec.base_bytes) + added
        arm_total += per_arm
        arm_breakdown.append(
            {
                "substrate_id": inner_spec.substrate_id,
                "base_bytes": len(inner_spec.base_bytes),
                "added_bytes": added,
                "arm_total": per_arm,
            }
        )

    # SOS1 trailer: 12 B header + n_pairs B selector + arm_meta brotli
    # (approx 200-400 B for typical specs).
    effective_outer = outer_stack_spec if outer_stack_spec is not None else OuterStackSpec(k=1)
    sos_trailer_bytes = (
        SOS_HEADER_STRUCT.size + n_pairs + 400  # 400 B is a conservative arm_meta estimate
    )

    predicted_total = arm_total + sos_trailer_bytes
    result = {
        "base_substrate_bytes": base_substrate_bytes,
        "arm_total_bytes": arm_total,
        "sos_trailer_bytes_estimate": sos_trailer_bytes,
        "predicted_total_bytes": predicted_total,
        "max_total_bytes": max_total_bytes,
        "arms": arm_breakdown,
        "k": effective_outer.k,
        "n_pairs": n_pairs,
    }
    if predicted_total > max_total_bytes:
        raise StackOfStacksError(
            f"predicted_total_bytes ({predicted_total}) exceeds "
            f"max_total_bytes ({max_total_bytes}); reduce inner-stack sidecars "
            f"or shrink K"
        )
    return result


__all__ = [
    "CONTEST_POSE_MARGINAL_PR106",
    "CONTEST_RATE_MARGINAL",
    "CONTEST_SEG_MARGINAL",
    "DEFAULT_N_PAIRS_CONTEST",
    "LAYER_BIT_INNER",
    "LAYER_BIT_MIDDLE",
    "LAYER_BIT_OUTER",
    "MAX_N_PAIRS",
    "MAX_OUTER_ARMS",
    "S2SBS_HF_MAGIC",
    "SABOR_BOUNDARY_MAGIC",
    "SCHEMA_VERSION",
    "SCORE_GRAD_RESIDUAL_MAGIC",
    "SOS_HEADER_STRUCT",
    "SOS_SIDECAR_MAGIC",
    "SOS_SIDECAR_VERSION",
    "BoundaryAtomSpec",
    "HFSidecarSpec",
    "InnerStack",
    "InnerStackSpec",
    "MiddleStack",
    "MiddleStackSpec",
    "OuterStack",
    "OuterStackSpec",
    "ResidualSpec",
    "StackOfStacksError",
    "compose_stack_of_stacks",
    "decompose_stack_of_stacks",
    "score_aware_mixing_weights",
    "validate_byte_budget",
]
