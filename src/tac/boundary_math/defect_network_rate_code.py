# SPDX-License-Identifier: MIT
"""Lossless connected-defect recode of a ``PHAS1`` phase-carrier section.

This is a research-probe codec, not a shipped archive grammar.  It tests the
narrow rate hypothesis that the incumbent phase residual has a useful
per-connected-boundary zero mode.  The active boundary mask and class map are
the same out-of-band geometry inputs required by :mod:`phase_residual_carrier`.
In task #452's real probe they come from the GT cache, not from a shippable
receiver; the codec commits to them but does not make them receiver-derivable.

For each deterministic 8-connected component, the code stores one residual
zero mode followed by first differences along canonical raster order.  An
optional finite ``Z2`` orientation quotient maps a component residual sequence
``r`` to ``-r`` when its first non-zero entry is negative and stores one group
label bit.  Both transforms are exactly invertible.  A full decode regenerates
the incumbent residual stream and then uses the incumbent decoder, so equality
is tested at the decoded phase-field boundary rather than inferred from sizes.

The physics language is deliberately bounded: connected components and the
``Z2`` coding action are rigorous properties of this integer stream.  Calling
them a defect zero mode or finite gauging is an analogy to Benjamin, Lam, and
Luo (2026), "Chiral Tube Algebras I: Topological Defect Lines, Twisted Modules,
and Finite Gauging", arXiv:2607.07786, especially Sections 1.1 and 1.3.  No CFT
theorem is imported as a compression theorem.
"""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.boundary_math.phase_residual_carrier import (
    PHASE_CARRIER_MAGIC,
    decode_phase_carrier,
)
from tac.boundary_math.xi_spline_residual_coder import (
    decode_residual_matrix,
    encode_residual_matrix,
    measure_residual_schemes,
    residual_scheme_id,
)

DEFECT_TUBE_MAGIC = b"DTUB1\x00"


class DefectTubeRateCodeError(ValueError):
    """Raised when a candidate section is corrupt or not losslessly invertible."""


@dataclass(frozen=True)
class DefectTubeRateReport:
    incumbent_section_bytes: int
    candidate_section_bytes: int
    bytes_saved: int
    residual_count: int
    component_count: int
    component_base_bytes: int
    intracomponent_delta_bytes: int
    incumbent_residual_stream_bytes: int
    component_stream_bytes: int
    component_stream_delta_bytes: int
    group_label_bytes: int
    base_scheme: str
    delta_scheme: str
    z2_orientation_quotient: bool
    exact_residual_roundtrip: bool
    exact_phase_field_roundtrip: bool
    constant_component_fraction: float
    constant_pixel_fraction: float
    singleton_component_fraction: float
    sign_consistent_component_fraction: float
    canonical_sequence_orbit_reuse_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _IncumbentSection:
    header: dict[str, Any]
    xi_bytes: bytes
    residuals: np.ndarray
    residual_stream_bytes: int


def _geometry_sha256(
    active_masks: list[np.ndarray],
    class_maps: list[np.ndarray],
    *,
    n_frames: int,
    height: int,
    width: int,
    classes: Iterable[int],
) -> str:
    """Commit to the exact out-of-band geometry required by this probe codec."""

    if len(active_masks) != n_frames or len(class_maps) != n_frames:
        raise DefectTubeRateCodeError(
            "geometry frame count mismatch: "
            f"masks={len(active_masks)} maps={len(class_maps)} expected={n_frames}"
        )
    digest = hashlib.sha256()
    digest.update(struct.pack("<III", n_frames, height, width))
    ordered_classes = tuple(int(value) for value in classes)
    digest.update(struct.pack("<I", len(ordered_classes)))
    for value in ordered_classes:
        digest.update(struct.pack("<q", value))
    for p, (mask, class_map) in enumerate(zip(active_masks, class_maps, strict=True)):
        semantic_mask = np.asarray(mask, dtype=bool)
        semantic_map = np.asarray(class_map, dtype=np.int64)
        expected_shape = (height, width)
        if semantic_mask.shape != expected_shape or semantic_map.shape != expected_shape:
            raise DefectTubeRateCodeError(
                f"frame {p}: geometry shape mismatch; "
                f"mask={semantic_mask.shape} map={semantic_map.shape} expected={expected_shape}"
            )
        digest.update(np.packbits(semantic_mask.reshape(-1), bitorder="little").tobytes())
        digest.update(semantic_map.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _take(blob: bytes, off: int, n: int, what: str) -> tuple[bytes, int]:
    if n < 0 or off < 0 or off + n > len(blob):
        raise DefectTubeRateCodeError(f"truncated {what}")
    return blob[off : off + n], off + n


def _parse_incumbent(section: bytes) -> _IncumbentSection:
    if not section.startswith(PHASE_CARRIER_MAGIC):
        raise DefectTubeRateCodeError("incumbent section has bad PHAS1 magic")
    off = len(PHASE_CARRIER_MAGIC)
    raw, off = _take(section, off, 4, "incumbent header length")
    (hlen,) = struct.unpack("<I", raw)
    raw, off = _take(section, off, hlen, "incumbent header")
    header = json.loads(raw.decode("utf-8"))
    p = int(header["n_frames"])
    xi_bytes, off = _take(section, off, p * 6 * 2, "incumbent xi table")
    raw, off = _take(section, off, 4, "incumbent residual length")
    (rlen,) = struct.unpack("<I", raw)
    res_blob, off = _take(section, off, rlen, "incumbent residual stream")
    if off != len(section):
        raise DefectTubeRateCodeError("incumbent section has trailing bytes")
    n = int(header["residual_count"])
    residuals = (
        decode_residual_matrix(res_blob, int(header["residual_scheme_id"]), n, 1).reshape(-1)
        if n
        else np.zeros(0, dtype=np.int64)
    )
    return _IncumbentSection(
        header=header,
        xi_bytes=xi_bytes,
        residuals=residuals,
        residual_stream_bytes=len(res_blob),
    )


def _true_runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(row, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _component_indices(mask: np.ndarray) -> list[np.ndarray]:
    """Canonical 8-connected components as sorted flat pixel indices.

    A row-run union-find avoids a SciPy dependency and makes the component
    ordering explicit: components are ordered by their first raster pixel;
    pixels inside a component are in raster order.
    """

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise DefectTubeRateCodeError(f"component mask must be 2-D, got {m.shape}")
    h, w = m.shape
    parent: list[int] = []
    runs: list[tuple[int, int, int]] = []

    def new_run(y: int, x0: int, x1: int) -> int:
        rid = len(parent)
        parent.append(rid)
        runs.append((y, x0, x1))
        return rid

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    previous: list[int] = []
    for y in range(h):
        current: list[int] = []
        for x0, x1 in _true_runs(m[y]):
            rid = new_run(y, x0, x1)
            current.append(rid)
            for pid in previous:
                _, p0, p1 = runs[pid]
                if p1 + 1 < x0:
                    continue
                if p0 - 1 > x1:
                    break
                union(rid, pid)
        previous = current

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for rid, run in enumerate(runs):
        groups.setdefault(find(rid), []).append(run)
    components: list[np.ndarray] = []
    for group in groups.values():
        idx = np.concatenate(
            [np.arange(y * w + x0, y * w + x1 + 1, dtype=np.int64) for y, x0, x1 in group]
        )
        components.append(np.sort(idx))
    components.sort(key=lambda x: int(x[0]))
    return components


def _iter_component_ordinals(
    active_masks: list[np.ndarray], class_maps: list[np.ndarray], classes: Iterable[int]
) -> Iterable[tuple[int, int, np.ndarray, int]]:
    """Yield ``(frame, class, active-raster ordinals, frame-class count)``."""

    classes = tuple(int(c) for c in classes)
    if len(active_masks) != len(class_maps):
        raise DefectTubeRateCodeError("mask/class-map frame counts differ")
    for p, (mask, cmap) in enumerate(zip(active_masks, class_maps, strict=True)):
        mask = np.asarray(mask, dtype=bool)
        cmap = np.asarray(cmap)
        if mask.shape != cmap.shape:
            raise DefectTubeRateCodeError(f"frame {p}: mask/class-map shapes differ")
        for cls in classes:
            selected = mask & (cmap == cls)
            active_idx = np.flatnonzero(selected.reshape(-1))
            for component in _component_indices(selected):
                ordinals = np.searchsorted(active_idx, component)
                if not np.array_equal(active_idx[ordinals], component):
                    raise DefectTubeRateCodeError("component-to-active ordering is inconsistent")
                yield p, cls, ordinals.astype(np.int64), int(active_idx.size)


def _best_stream(values: np.ndarray) -> tuple[str, bytes]:
    matrix = np.asarray(values, dtype=np.int64).reshape(-1, 1)
    if matrix.size == 0:
        return "varint", b""
    sizes = measure_residual_schemes(matrix)
    scheme = min(sizes, key=lambda name: (sizes[name], name))
    return scheme, encode_residual_matrix(matrix, scheme)


def _pack_bits(bits: list[bool]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_bits(blob: bytes, n: int) -> np.ndarray:
    if len(blob) != (n + 7) // 8:
        raise DefectTubeRateCodeError("group-label bitstream length mismatch")
    return np.asarray([bool(blob[i // 8] & (1 << (i % 8))) for i in range(n)], dtype=bool)


def _component_streams(
    incumbent: _IncumbentSection,
    active_masks: list[np.ndarray],
    class_maps: list[np.ndarray],
    *,
    z2_orientation_quotient: bool,
) -> tuple[np.ndarray, np.ndarray, list[bool], dict[str, float | int]]:
    classes = tuple(int(c) for c in incumbent.header["classes"])
    expected_counts = incumbent.header["per_frame_class_counts"]
    cursor = 0
    bases: list[int] = []
    deltas: list[int] = []
    group_labels: list[bool] = []
    sizes: list[int] = []
    constant_components = 0
    constant_pixels = 0
    sign_consistent = 0
    canonical_sequences: set[bytes] = set()

    grouped: dict[tuple[int, int], list[np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for p, cls, ordinals, frame_class_count in _iter_component_ordinals(active_masks, class_maps, classes):
        grouped.setdefault((p, cls), []).append(ordinals)
        counts[(p, cls)] = frame_class_count

    for p in range(int(incumbent.header["n_frames"])):
        for ci, cls in enumerate(classes):
            n = int(expected_counts[p][ci])
            if counts.get((p, cls), 0) != n:
                raise DefectTubeRateCodeError(
                    f"frame {p} class {cls}: derived active count {counts.get((p, cls), 0)} != header {n}"
                )
            frame_values = incumbent.residuals[cursor : cursor + n]
            cursor += n
            for ordinals in grouped.get((p, cls), []):
                values = np.asarray(frame_values[ordinals], dtype=np.int64)
                if values.size == 0:
                    continue
                sign_flip = False
                if z2_orientation_quotient:
                    nz = values[values != 0]
                    sign_flip = bool(nz.size and nz[0] < 0)
                canonical = -values if sign_flip else values
                bases.append(int(canonical[0]))
                if canonical.size > 1:
                    deltas.extend(np.diff(canonical).astype(np.int64).tolist())
                group_labels.append(sign_flip)
                sizes.append(int(values.size))
                canonical_sequences.add(struct.pack("<I", values.size) + canonical.tobytes())
                if np.all(values == values[0]):
                    constant_components += 1
                    constant_pixels += int(values.size)
                if np.all(values >= 0) or np.all(values <= 0):
                    sign_consistent += 1
    if cursor != incumbent.residuals.size:
        raise DefectTubeRateCodeError("incumbent residual stream not fully assigned to components")
    component_count = len(bases)
    residual_count = int(incumbent.residuals.size)
    stats: dict[str, float | int] = {
        "component_count": component_count,
        "constant_component_fraction": constant_components / component_count if component_count else 1.0,
        "constant_pixel_fraction": constant_pixels / residual_count if residual_count else 1.0,
        "singleton_component_fraction": float(np.mean(np.asarray(sizes) == 1)) if sizes else 1.0,
        "sign_consistent_component_fraction": sign_consistent / component_count if component_count else 1.0,
        "canonical_sequence_orbit_reuse_fraction": (
            1.0 - len(canonical_sequences) / component_count if component_count else 0.0
        ),
    }
    return (
        np.asarray(bases, dtype=np.int64),
        np.asarray(deltas, dtype=np.int64),
        group_labels,
        stats,
    )


def encode_defect_tube_recode(
    incumbent_section: bytes,
    active_masks: list[np.ndarray],
    class_maps: list[np.ndarray],
    *,
    z2_orientation_quotient: bool = False,
) -> tuple[bytes, DefectTubeRateReport]:
    """Losslessly recode one real incumbent phase section."""

    incumbent = _parse_incumbent(incumbent_section)
    geometry_sha256 = _geometry_sha256(
        active_masks,
        class_maps,
        n_frames=int(incumbent.header["n_frames"]),
        height=int(incumbent.header["height"]),
        width=int(incumbent.header["width"]),
        classes=incumbent.header["classes"],
    )
    bases, deltas, labels, stats = _component_streams(
        incumbent,
        active_masks,
        class_maps,
        z2_orientation_quotient=z2_orientation_quotient,
    )
    base_scheme, base_blob = _best_stream(bases)
    delta_scheme, delta_blob = _best_stream(deltas)
    label_blob = _pack_bits(labels) if z2_orientation_quotient else b""
    h0 = incumbent.header
    header = {
        "version": 1,
        "n_frames": int(h0["n_frames"]),
        "height": int(h0["height"]),
        "width": int(h0["width"]),
        "q_step": float(h0["q_step"]),
        "classes": [int(c) for c in h0["classes"]],
        "anchor_predict": float(h0["anchor_predict"]),
        "annulus_band": float(h0["annulus_band"]),
        "gap_xi": str(h0["gap_xi"]),
        "pitch": float(h0["pitch"]),
        "residual_count": int(incumbent.residuals.size),
        "component_count": int(bases.size),
        "delta_count": int(deltas.size),
        "base_scheme_id": residual_scheme_id(base_scheme),
        "delta_scheme_id": residual_scheme_id(delta_scheme),
        "z2_orientation_quotient": bool(z2_orientation_quotient),
        "geometry_sha256": geometry_sha256,
    }
    hj = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    section = b"".join(
        (
            DEFECT_TUBE_MAGIC,
            struct.pack("<I", len(hj)),
            hj,
            incumbent.xi_bytes,
            struct.pack("<III", len(label_blob), len(base_blob), len(delta_blob)),
            label_blob,
            base_blob,
            delta_blob,
        )
    )
    rebuilt_phase, rebuilt_residuals = decode_defect_tube_to_phase_section(
        section, active_masks, class_maps
    )
    exact_residual = np.array_equal(rebuilt_residuals, incumbent.residuals)
    if not exact_residual:
        raise DefectTubeRateCodeError("NO-FAKE: candidate residual decode differs from incumbent")
    incumbent_fields = decode_phase_carrier(incumbent_section, active_masks, class_maps)
    candidate_fields = decode_phase_carrier(rebuilt_phase, active_masks, class_maps)
    exact_fields = all(
        np.array_equal(a, b) for a, b in zip(incumbent_fields, candidate_fields, strict=True)
    )
    if not exact_fields:
        raise DefectTubeRateCodeError("NO-FAKE: candidate phase fields differ from incumbent")
    report = DefectTubeRateReport(
        incumbent_section_bytes=len(incumbent_section),
        candidate_section_bytes=len(section),
        bytes_saved=len(incumbent_section) - len(section),
        residual_count=int(incumbent.residuals.size),
        component_count=int(stats["component_count"]),
        component_base_bytes=len(base_blob),
        intracomponent_delta_bytes=len(delta_blob),
        incumbent_residual_stream_bytes=incumbent.residual_stream_bytes,
        component_stream_bytes=len(base_blob) + len(delta_blob) + len(label_blob),
        component_stream_delta_bytes=(
            len(base_blob)
            + len(delta_blob)
            + len(label_blob)
            - incumbent.residual_stream_bytes
        ),
        group_label_bytes=len(label_blob),
        base_scheme=base_scheme,
        delta_scheme=delta_scheme,
        z2_orientation_quotient=bool(z2_orientation_quotient),
        exact_residual_roundtrip=exact_residual,
        exact_phase_field_roundtrip=exact_fields,
        constant_component_fraction=float(stats["constant_component_fraction"]),
        constant_pixel_fraction=float(stats["constant_pixel_fraction"]),
        singleton_component_fraction=float(stats["singleton_component_fraction"]),
        sign_consistent_component_fraction=float(stats["sign_consistent_component_fraction"]),
        canonical_sequence_orbit_reuse_fraction=float(stats["canonical_sequence_orbit_reuse_fraction"]),
    )
    return section, report


def _parse_candidate(section: bytes) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes]:
    if not section.startswith(DEFECT_TUBE_MAGIC):
        raise DefectTubeRateCodeError("bad DTUB1 magic")
    off = len(DEFECT_TUBE_MAGIC)
    raw, off = _take(section, off, 4, "candidate header length")
    (hlen,) = struct.unpack("<I", raw)
    raw, off = _take(section, off, hlen, "candidate header")
    header = json.loads(raw.decode("utf-8"))
    p = int(header["n_frames"])
    xi_bytes, off = _take(section, off, p * 6 * 2, "candidate xi table")
    raw, off = _take(section, off, 12, "candidate stream lengths")
    label_len, base_len, delta_len = struct.unpack("<III", raw)
    label_blob, off = _take(section, off, label_len, "candidate group labels")
    base_blob, off = _take(section, off, base_len, "candidate component bases")
    delta_blob, off = _take(section, off, delta_len, "candidate component deltas")
    if off != len(section):
        raise DefectTubeRateCodeError("candidate section has trailing bytes")
    return header, xi_bytes, label_blob, base_blob, delta_blob


def decode_defect_tube_to_phase_section(
    section: bytes,
    active_masks: list[np.ndarray],
    class_maps: list[np.ndarray],
) -> tuple[bytes, np.ndarray]:
    """Decode ``DTUB1`` and regenerate an equivalent ``PHAS1`` section."""

    header, xi_bytes, label_blob, base_blob, delta_blob = _parse_candidate(section)
    n_components = int(header["component_count"])
    n_deltas = int(header["delta_count"])
    bases = (
        decode_residual_matrix(base_blob, int(header["base_scheme_id"]), n_components, 1).reshape(-1)
        if n_components
        else np.zeros(0, dtype=np.int64)
    )
    deltas = (
        decode_residual_matrix(delta_blob, int(header["delta_scheme_id"]), n_deltas, 1).reshape(-1)
        if n_deltas
        else np.zeros(0, dtype=np.int64)
    )
    z2 = bool(header["z2_orientation_quotient"])
    if not z2 and label_blob:
        raise DefectTubeRateCodeError("non-Z2 candidate must not contain group-label bytes")
    labels = _unpack_bits(label_blob, n_components) if z2 else np.zeros(n_components, dtype=bool)
    classes = tuple(int(c) for c in header["classes"])
    geometry_sha256 = _geometry_sha256(
        active_masks,
        class_maps,
        n_frames=int(header["n_frames"]),
        height=int(header["height"]),
        width=int(header["width"]),
        classes=classes,
    )
    if geometry_sha256 != str(header["geometry_sha256"]):
        raise DefectTubeRateCodeError("out-of-band geometry commitment mismatch")
    grouped: dict[tuple[int, int], list[np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for p, cls, ordinals, count in _iter_component_ordinals(active_masks, class_maps, classes):
        grouped.setdefault((p, cls), []).append(ordinals)
        counts[(p, cls)] = count

    output: list[np.ndarray] = []
    per_frame_counts: list[list[int]] = []
    bi = di = 0
    for p in range(int(header["n_frames"])):
        frame_counts: list[int] = []
        for cls in classes:
            n = counts.get((p, cls), 0)
            values = np.zeros(n, dtype=np.int64)
            frame_counts.append(n)
            for ordinals in grouped.get((p, cls), []):
                if bi >= bases.size:
                    raise DefectTubeRateCodeError("component base stream exhausted early")
                length = int(ordinals.size)
                canonical = np.empty(length, dtype=np.int64)
                canonical[0] = bases[bi]
                if length > 1:
                    if di + length - 1 > deltas.size:
                        raise DefectTubeRateCodeError("component delta stream exhausted early")
                    canonical[1:] = canonical[0] + np.cumsum(deltas[di : di + length - 1])
                    di += length - 1
                values[ordinals] = -canonical if labels[bi] else canonical
                bi += 1
            output.append(values)
        per_frame_counts.append(frame_counts)
    if bi != bases.size or di != deltas.size:
        raise DefectTubeRateCodeError("candidate component streams not fully consumed")
    residuals = np.concatenate(output) if output else np.zeros(0, dtype=np.int64)
    if residuals.size != int(header["residual_count"]):
        raise DefectTubeRateCodeError("candidate residual count mismatch")

    residual_scheme, residual_blob = _best_stream(residuals)
    incumbent_header = {
        "version": 1,
        "n_frames": int(header["n_frames"]),
        "height": int(header["height"]),
        "width": int(header["width"]),
        "q_step": float(header["q_step"]),
        "classes": list(classes),
        "anchor_predict": float(header["anchor_predict"]),
        "annulus_band": float(header["annulus_band"]),
        "gap_xi": str(header["gap_xi"]),
        "pitch": float(header["pitch"]),
        "per_frame_class_counts": per_frame_counts,
        "residual_scheme_id": residual_scheme_id(residual_scheme),
        "residual_count": int(residuals.size),
    }
    hj = json.dumps(incumbent_header, separators=(",", ":")).encode("utf-8")
    phase_section = b"".join(
        (
            PHASE_CARRIER_MAGIC,
            struct.pack("<I", len(hj)),
            hj,
            xi_bytes,
            struct.pack("<I", len(residual_blob)),
            residual_blob,
        )
    )
    return phase_section, residuals


__all__ = [
    "DEFECT_TUBE_MAGIC",
    "DefectTubeRateCodeError",
    "DefectTubeRateReport",
    "decode_defect_tube_to_phase_section",
    "encode_defect_tube_recode",
]
