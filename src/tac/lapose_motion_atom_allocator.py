# SPDX-License-Identifier: MIT
"""LAPose inverse-dynamics motion-atom allocator — pose-axis residual sidecar.

Lane: ``lane_pose_axis_lapose_motion_atom_allocator_full_scaffold`` (Phase 3
substrate-engineering; pose-axis target). Per the handoff P3 ledger
``~/Downloads/pact_score_lowering_handoff_2026-05-11.md`` ("LAPose-inspired
until a paper-faithful inverse-dynamics encoder and pose head exist; add
class/openpilot manifests, calibrate confidence, and require a charged
archive consumer before dispatch") and Grand Council Insight 3 (pose-axis
lanes carry **2.79× higher EV per byte at PR106 r2 operating point**).

Architecture
============

LAPose-inspired (see Quaternion Renderer / openpilot-side inverse-dynamics
literature). Per-frame **motion atoms** are typed parameter sets that encode
a small, score-relevant local-dynamics correction. The allocator picks bytes
across atom classes to maximise predicted score-Δ subject to a hard byte
budget. Atom classes:

- ``yaw_rate`` — single-scalar pose dim 0 correction (most score-sensitive
  per PR94/PR101 retrospective)
- ``pitch_rate`` — pose dim 1 correction
- ``foveation_pull`` — single Gaussian-pull foveation atom (sister to
  ``tac.foveation_field``; full field is a separate lane)
- ``class_token`` — per-frame openpilot road-region class token (categorical;
  consumed by the renderer to choose a class-specific decode head)

This module ships **research_only=true** semantics until a paper-faithful
inverse-dynamics encoder + pose head exist and openpilot class/confidence
manifests are calibrated. ``score_claim=False`` permanently per CLAUDE.md.

Wire format (LAPose motion atom sidecar)
----------------------------------------
::

    magic          : u8  = 0xFD  (PR106 residual family magic, reused)
    format_id      : u8  = 0x32  (LAPOSE_MOTION_ATOM_FORMAT_ID)
    n_frames       : u32_LE
    n_atoms        : u16_LE
    atom_table     : n_atoms * (
                        atom_class : u8     [0=yaw_rate,1=pitch_rate,
                                             2=foveation_pull,3=class_token]
                        frame_idx  : u32_LE
                        param_bytes: u8     length of params
                        params     : variable
                    )

The decoder reads the atom stream linearly and applies each atom to the
PR106 base output at its frame_idx.

CLAUDE.md compliance
====================
- ``score_claim = False`` permanently until charged archive consumer + exact T4
- ``promotion_eligible = False`` permanently
- ``ready_for_exact_eval_dispatch = False`` until council deliberation
- ``research_only = True`` at this scaffold level per handoff's
  "LAPose-inspired until a paper-faithful inverse-dynamics encoder and pose
  head exist" caveat
- 8 archive-grammar fields declared (see lane registry)
- ``lane_class = substrate_engineering``
- NO scorer load at inflate
- NO ``/tmp`` paths
- NO MPS authoritative
- Sister of ``tac.foveation_field`` (foveation_pull atom is a single-Gaussian
  scalar-budget variant; the full field is the dedicated lane)

8 archive-grammar fields (Catalog #124)
=======================================
- ``archive_grammar``: monolithic ``0.bin`` (0xFD + 0x32 wrapper around PR106)
- ``parser_section_manifest``: linear atom stream; see
  ``LAPOSE_ATOM_STREAM_FORMAT`` constant
- ``inflate_runtime_loc_budget``: 200 LOC waiver under substrate_engineering
- ``runtime_dep_closure``: torch + brotli + numpy (contest runtime)
- ``export_format``: ``lapose_motion_atom_stream_v1``
- ``score_aware_loss``: deferred to allocator (research_only initially)
- ``bolt_on_loc_budget``: substrate_engineering
- ``no_op_detector_planned``: stream must contain at least one non-zero atom;
  tested. Plus per-atom-class predicted Δscore must exceed 1e-6 for
  inclusion.

6-hook wire-in declarations
===========================
- Sensitivity-map: per-atom-class predicted Δpose informs the allocator
- Pareto: candidate when L2 encoder + dispatch + class manifest calibration
  land
- Bit-allocator: this module IS the allocator (informs by allocating)
- Cathedral autopilot: register as ``optimize_mode_transforms`` candidate
- Continual-learning posterior update: triggered on exact T4 result
- Probe-disambiguator: foveation-vs-RAFT-vs-LAPose head-to-head
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field as dataclass_field
from enum import IntEnum


PR106_RESIDUAL_MAGIC = 0xFD
LAPOSE_MOTION_ATOM_FORMAT_ID = 0x32
LAPOSE_MOTION_ATOM_VERSION = 1

# Hard byte budget per CLAUDE.md / handoff operator constraint.
MAX_ENCODED_BYTES = 1024


class AtomClass(IntEnum):
    """Motion atom class codes (wire format)."""

    YAW_RATE = 0
    PITCH_RATE = 1
    FOVEATION_PULL = 2
    CLASS_TOKEN = 3


# Per-atom-class expected params footprint (bytes) for validation.
ATOM_CLASS_PARAM_BYTES = {
    AtomClass.YAW_RATE: 2,  # int16 scaled correction
    AtomClass.PITCH_RATE: 2,
    AtomClass.FOVEATION_PULL: 6,  # (cx, cy, amp) each int16
    AtomClass.CLASS_TOKEN: 1,  # u8 token id
}


@dataclass
class MotionAtom:
    """A single typed atom in the LAPose motion stream."""

    atom_class: AtomClass
    frame_idx: int
    params: bytes  # length must match ATOM_CLASS_PARAM_BYTES[atom_class]

    def validate(self) -> None:
        if not isinstance(self.atom_class, AtomClass):
            raise ValueError(f"atom_class must be AtomClass; got {type(self.atom_class)}")
        if self.frame_idx < 0 or self.frame_idx > 2**31 - 1:
            raise ValueError(f"frame_idx out of range: {self.frame_idx}")
        expected_len = ATOM_CLASS_PARAM_BYTES[self.atom_class]
        if len(self.params) != expected_len:
            raise ValueError(
                f"atom_class {self.atom_class.name} requires {expected_len} param bytes; "
                f"got {len(self.params)}"
            )

    @property
    def byte_cost(self) -> int:
        """Bytes this atom contributes to the encoded stream (header + params)."""
        # Header per atom = 1 (class) + 4 (frame_idx) + 1 (param_bytes_len) + params.
        return 6 + len(self.params)


@dataclass
class LaposeAtomStream:
    """Stream of typed motion atoms applied per-frame to PR106 base output."""

    n_frames: int
    atoms: list[MotionAtom] = dataclass_field(default_factory=list)

    def validate(self) -> None:
        if self.n_frames < 1 or self.n_frames > 2**31 - 1:
            raise ValueError(f"n_frames out of range: {self.n_frames}")
        if len(self.atoms) > 65535:
            raise ValueError(f"too many atoms: {len(self.atoms)} > 65535")
        for atom in self.atoms:
            atom.validate()
            if atom.frame_idx >= self.n_frames:
                raise ValueError(
                    f"atom frame_idx {atom.frame_idx} >= n_frames {self.n_frames}"
                )


@dataclass
class AtomCandidate:
    """Predicted score-Δ + byte cost for the allocator."""

    atom: MotionAtom
    predicted_delta_score: float  # negative is better (lower contest score)
    confidence: float  # in [0, 1]; 1 = fully calibrated

    def value_per_byte(self) -> float:
        """Allocator key: predicted ``|Δscore| * confidence / bytes``.

        Higher is better. Atoms with confidence < 0.1 should be excluded by
        the allocator caller (per CLAUDE.md "calibrate confidence" caveat).
        """
        byte_cost = max(self.atom.byte_cost, 1)
        return abs(self.predicted_delta_score) * self.confidence / byte_cost


class LaposeAtomAllocator:
    """Allocate bytes across motion atoms to maximize predicted score-Δ.

    Per the handoff "calibrate confidence" caveat, the allocator rejects
    candidates with ``confidence < min_confidence`` (default 0.1) regardless
    of predicted-Δ magnitude. This prevents speculative atoms from consuming
    bytes that would be better spent on higher-confidence atoms.
    """

    def __init__(
        self,
        *,
        max_total_bytes: int = MAX_ENCODED_BYTES,
        min_confidence: float = 0.1,
    ) -> None:
        if max_total_bytes < 12:
            # Minimum is header (2) + n_frames (4) + n_atoms (2) + at least
            # one atom (6 + 1 minimum params).
            raise ValueError(f"max_total_bytes must be >= 12; got {max_total_bytes}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1]; got {min_confidence}")
        self.max_total_bytes = max_total_bytes
        self.min_confidence = min_confidence

    def allocate(
        self,
        candidates: list[AtomCandidate],
        *,
        n_frames: int,
    ) -> LaposeAtomStream:
        """Greedy allocator: sort by value_per_byte descending; pack until budget.

        Returns the resulting stream. Refuses to include candidates below
        ``min_confidence``; refuses to include candidates with non-negative
        predicted-Δscore (since negative is better for the contest scorer).
        """
        # Header bytes per encode_lapose_atom_stream:
        # magic(1) + format_id(1) + n_frames(4) + n_atoms(2) = 8 bytes.
        header_bytes = 8
        bytes_available = self.max_total_bytes - header_bytes

        # Filter candidates by confidence + sign-of-delta-score.
        viable = [
            c
            for c in candidates
            if c.confidence >= self.min_confidence and c.predicted_delta_score < 0.0
        ]
        # Sort by value_per_byte descending.
        viable.sort(key=lambda c: c.value_per_byte(), reverse=True)

        selected: list[MotionAtom] = []
        consumed_bytes = 0
        for cand in viable:
            cost = cand.atom.byte_cost
            if consumed_bytes + cost > bytes_available:
                continue
            selected.append(cand.atom)
            consumed_bytes += cost

        stream = LaposeAtomStream(n_frames=n_frames, atoms=selected)
        stream.validate()
        return stream


def encode_motion_atom_stream(
    stream: LaposeAtomStream,
    *,
    enforce_budget: bool = True,
) -> bytes:
    """Encode a ``LaposeAtomStream`` to the wire format."""
    stream.validate()
    out = bytearray()
    out.append(PR106_RESIDUAL_MAGIC)
    out.append(LAPOSE_MOTION_ATOM_FORMAT_ID)
    out += struct.pack("<I", stream.n_frames)
    out += struct.pack("<H", len(stream.atoms))
    for atom in stream.atoms:
        out.append(int(atom.atom_class))
        out += struct.pack("<I", atom.frame_idx)
        out.append(len(atom.params))
        out += atom.params

    encoded = bytes(out)
    if enforce_budget and len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError(
            f"lapose motion atom stream encoded size {len(encoded)} > budget {MAX_ENCODED_BYTES}; "
            "drop low-value atoms"
        )
    return encoded


def decode_motion_atom_stream(blob: bytes) -> LaposeAtomStream:
    """Inverse of :func:`encode_motion_atom_stream`."""
    if len(blob) < 8:
        raise ValueError(f"lapose motion atom blob too short: {len(blob)}")
    if blob[0] != PR106_RESIDUAL_MAGIC:
        raise ValueError(
            f"lapose magic mismatch: 0x{blob[0]:02X} != 0x{PR106_RESIDUAL_MAGIC:02X}"
        )
    if blob[1] != LAPOSE_MOTION_ATOM_FORMAT_ID:
        raise ValueError(
            f"lapose format_id mismatch: 0x{blob[1]:02X} != 0x{LAPOSE_MOTION_ATOM_FORMAT_ID:02X}"
        )
    pos = 2
    (n_frames,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    (n_atoms,) = struct.unpack_from("<H", blob, pos)
    pos += 2
    atoms: list[MotionAtom] = []
    for _ in range(n_atoms):
        if pos + 6 > len(blob):
            raise ValueError("lapose atom stream truncated mid-atom")
        atom_class_raw = blob[pos]
        pos += 1
        try:
            atom_class = AtomClass(atom_class_raw)
        except ValueError as exc:
            raise ValueError(f"unknown atom_class: {atom_class_raw}") from exc
        (frame_idx,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        param_len = blob[pos]
        pos += 1
        expected = ATOM_CLASS_PARAM_BYTES[atom_class]
        if param_len != expected:
            raise ValueError(
                f"atom_class {atom_class.name} requires {expected} param bytes; got {param_len}"
            )
        params = bytes(blob[pos : pos + param_len])
        pos += param_len
        atoms.append(MotionAtom(atom_class=atom_class, frame_idx=frame_idx, params=params))
    if pos != len(blob):
        raise ValueError(f"lapose atom stream trailing bytes: pos={pos} total={len(blob)}")
    return LaposeAtomStream(n_frames=n_frames, atoms=atoms)


def compute_lapose_atom_stream_bytes(stream: LaposeAtomStream) -> int:
    """Return encoded byte size of ``stream`` without raising on overflow."""
    return len(encode_motion_atom_stream(stream, enforce_budget=False))


def is_no_op(stream: LaposeAtomStream) -> bool:
    """A stream with zero atoms is a no-op (wastes header bytes)."""
    return len(stream.atoms) == 0


def decode_yaw_rate_params(params: bytes) -> float:
    """YAW_RATE atom: int16 correction in scaled units (rad * 1e4)."""
    if len(params) != 2:
        raise ValueError(f"yaw_rate params must be 2 bytes; got {len(params)}")
    (raw,) = struct.unpack("<h", params)
    return float(raw) * 1e-4


def encode_yaw_rate_params(yaw_rate_correction: float) -> bytes:
    """Inverse of :func:`decode_yaw_rate_params`."""
    raw = int(round(yaw_rate_correction * 1e4))
    raw = max(-32768, min(32767, raw))
    return struct.pack("<h", raw)


def decode_class_token_params(params: bytes) -> int:
    """CLASS_TOKEN atom: u8 token id (0-255)."""
    if len(params) != 1:
        raise ValueError(f"class_token params must be 1 byte; got {len(params)}")
    return int(params[0])


def encode_class_token_params(token_id: int) -> bytes:
    """Inverse of :func:`decode_class_token_params`."""
    if not (0 <= token_id <= 255):
        raise ValueError(f"token_id must be in [0, 255]; got {token_id}")
    return bytes([token_id])


def decode_foveation_pull_params(params: bytes) -> tuple[float, float, float]:
    """FOVEATION_PULL atom: (cx, cy, amp) each int16 / 1e4."""
    if len(params) != 6:
        raise ValueError(f"foveation_pull params must be 6 bytes; got {len(params)}")
    cx_raw, cy_raw, amp_raw = struct.unpack("<hhh", params)
    return (cx_raw * 1e-4, cy_raw * 1e-4, amp_raw * 1e-4)


def encode_foveation_pull_params(cx: float, cy: float, amp: float) -> bytes:
    """Inverse of :func:`decode_foveation_pull_params`."""
    cx_raw = max(-32768, min(32767, int(round(cx * 1e4))))
    cy_raw = max(-32768, min(32767, int(round(cy * 1e4))))
    amp_raw = max(-32768, min(32767, int(round(amp * 1e4))))
    return struct.pack("<hhh", cx_raw, cy_raw, amp_raw)


__all__ = [
    "ATOM_CLASS_PARAM_BYTES",
    "AtomCandidate",
    "AtomClass",
    "LAPOSE_MOTION_ATOM_FORMAT_ID",
    "LAPOSE_MOTION_ATOM_VERSION",
    "LaposeAtomAllocator",
    "LaposeAtomStream",
    "MAX_ENCODED_BYTES",
    "MotionAtom",
    "PR106_RESIDUAL_MAGIC",
    "compute_lapose_atom_stream_bytes",
    "decode_class_token_params",
    "decode_foveation_pull_params",
    "decode_motion_atom_stream",
    "decode_yaw_rate_params",
    "encode_class_token_params",
    "encode_foveation_pull_params",
    "encode_motion_atom_stream",
    "encode_yaw_rate_params",
    "is_no_op",
]
