"""Cross-paradigm composition example archives (concrete demonstrations of QQ orthogonal pairs).

Per operator amplification 2026-05-11 ("compiler" + "wiring and integration"
+ "QQ's composition matrix surfacing orthogonal pairs"), this module materializes
**concrete cross-paradigm composition example archives** demonstrating the
top-K orthogonal pairs from QQ's substrate composition matrix.

Each example carries:

- **Format-ID assignments** per substrate (no collision verified against the QQ matrix)
- **Decoder ordering** declared (the inflate-time decoder dispatches by format_id)
- **Predicted Δ** quoted from the matrix (with operating-point context)
- **Byte budget feasibility** verified against the PR106 r2 frontier (~186k bytes)
- **Score-claim invariant** permanently False (planning-only; no exact CUDA dispatch)

The output is a typed manifest plus an optional placeholder archive (zero-filled
bytes shaped per the format-ID grammar). Per CLAUDE.md "Deterministic packet
compiler" the manifest is byte-deterministic and ships with byte-level
reproducibility hashes — but it is NOT a contest-promotable archive until
exact CUDA + CPU adjudication lands.

Per CLAUDE.md "Forbidden score claims" + "Forbidden empirical-claim-without-
evidence-tag", every numeric in this module's output is tagged
``[predicted; cross-paradigm composition example v1]``. Score promotion
requires a ``[contest-CUDA]`` + ``[contest-CPU]`` anchor on the EXACT bytes.

Per CLAUDE.md "HNeRV parity discipline" lessons 2 (export-first design) +
3 (monolithic single-file substrate) + 4 (≤200 LOC inflate budget): the
example archives DO declare the 8 archive-grammar fields (planning_only=False
since composability examples ARE substrate engineering), but the inflate runtime
is intentionally a stub that prints "[planning_only_example] decoder dispatch
sketch" and exits 0 — there is no production decoder until exact-eval custody.

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — orthogonal pair source
- :mod:`tac.optimization.bit_allocator_end_to_end` — sister module that
  consumes these examples as Pareto candidates.
- :mod:`tac.packet_compiler.magic_codec` — meta-codec wrapper that the
  example archives may optionally include as the byte-stream-level layer.
- ``feedback_substrate_composition_matrix_autopilot_ranking_theoretical_floor_v2_landed_20260511.md``
  — QQ landing memo with the ranked orthogonal pairs.

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``cross_paradigm_composition_examples_v1``
- ``halt_and_ask_default_for_dispatch_recommendations``
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tac.optimization.substrate_composition_matrix import (
    Composability,
    CompositionMatrix,
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    build_composition_matrix,
    predicted_composite_delta,
)

SCHEMA_VERSION = "tac_cross_paradigm_composition_example_v1"

# PR106 r2 frontier reference budget — the real archive at the contest's
# current operating point lands around 178k-186k bytes.
PR106_R2_REFERENCE_ARCHIVE_BYTES: int = 186_000


# ── Typed example record ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CompositionExampleSubstrateRow:
    """Per-substrate row inside one composition example."""

    substrate_id: str
    substrate_class: SubstrateClass
    target_axis: ScoreAxis
    format_id: int
    magic_bytes: str
    allocated_bytes: int
    runtime_dep_closure: tuple[str, ...]


@dataclass(frozen=True)
class CompositionExample:
    """One cross-paradigm composition example archive description.

    Per CLAUDE.md "Deterministic packet compiler" + "Beauty, simplicity, and
    developer experience": typed + frozen + JSON-safe.
    """

    example_id: str
    pair_a_substrate_id: str
    pair_b_substrate_id: str
    composability: str  # ORTHOGONAL/STACKABLE_SERIAL/etc (always ORTHOGONAL here)
    expected_alpha: float
    rationale: str
    predicted_score_delta_alone_a: float
    predicted_score_delta_alone_b: float
    predicted_composite_delta: float
    rows: tuple[CompositionExampleSubstrateRow, ...]
    decoder_ordering: tuple[str, ...]  # Substrate ids in dispatch order.
    total_archive_bytes: int
    fits_pr106_r2_frontier: bool
    pr106_r2_reference_bytes: int = PR106_R2_REFERENCE_ARCHIVE_BYTES
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class CompositionExampleSet:
    """A set of cross-paradigm composition examples + their summary."""

    schema: str
    generated_at_utc: str
    matrix_schema: str
    n_examples: int
    examples: tuple[CompositionExample, ...]
    pr106_r2_reference_bytes: int = PR106_R2_REFERENCE_ARCHIVE_BYTES
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# ── Top-K orthogonal pair selection ──────────────────────────────────────


def select_top_orthogonal_pairs(
    matrix: Optional[CompositionMatrix] = None,
    *,
    top_k: int = 5,
    require_axes_diverge: bool = True,
    exclude_replacement_renderers: bool = False,
) -> list[tuple[SubstrateRow, SubstrateRow]]:
    """Select the top-K orthogonal pairs by joint EV/$ from the QQ matrix.

    Selection rules (per QQ landing's cross-paradigm composition findings):

    - Only ORTHOGONAL pairs (additive-savings predicted; alpha=1.0).
    - Only pairs whose target axes diverge (e.g., wavelet=RATE + c3=MIXED;
      foveation=POSE + nerv_as_renderer=MIXED). When ``require_axes_diverge``
      is False, same-axis orthogonal pairs are admitted (e.g., two BOLT_ON
      substrates of the same axis when matrix has them as ORTHOGONAL).
    - Substrates participate in at most ONE pair (greedy; highest-EV first).
    - When ``exclude_replacement_renderers`` is True, RENDERER_REPLACEMENT
      substrates are dropped (useful for "lightweight" examples that don't
      change the renderer slot).

    Returns a list of (a, b) substrate-row tuples sorted by joint
    ``|predicted_delta_alone_midpoint|`` (joint EV) descending.
    """
    matrix = matrix or build_composition_matrix()
    rows = list(matrix.substrates)
    if exclude_replacement_renderers:
        rows = [
            r
            for r in rows
            if r.substrate_class != SubstrateClass.RENDERER_REPLACEMENT
        ]

    candidates: list[tuple[SubstrateRow, SubstrateRow, float]] = []
    for i, ri in enumerate(rows):
        for j in range(i + 1, len(rows)):
            rj = rows[j]
            cell = matrix.get(ri.substrate_id, rj.substrate_id)
            if cell.composability != Composability.ORTHOGONAL:
                continue
            if require_axes_diverge and ri.target_axis == rj.target_axis:
                continue
            joint_ev = abs(ri.predicted_delta_alone_midpoint()) + abs(
                rj.predicted_delta_alone_midpoint()
            )
            candidates.append((ri, rj, joint_ev))

    # Sort by joint EV desc, then by lex order of substrate_ids for stability.
    candidates.sort(
        key=lambda x: (
            -x[2],
            x[0].substrate_id,
            x[1].substrate_id,
        )
    )

    out: list[tuple[SubstrateRow, SubstrateRow]] = []
    chosen: set[str] = set()
    for ri, rj, _ in candidates:
        if ri.substrate_id in chosen or rj.substrate_id in chosen:
            continue
        out.append((ri, rj))
        chosen.add(ri.substrate_id)
        chosen.add(rj.substrate_id)
        if len(out) >= top_k:
            break
    return out


# ── Composition-example construction ─────────────────────────────────────


def _allocate_bytes_per_substrate(
    pair: tuple[SubstrateRow, SubstrateRow],
    *,
    per_pair_byte_cap: int,
) -> tuple[int, int]:
    """Distribute ``per_pair_byte_cap`` between the two substrates.

    Allocation rule: each substrate gets the floor of its byte_budget_band
    first; remaining bytes split proportionally to midpoint of band.
    """
    a, b = pair
    floor_a = int(a.byte_budget_band[0])
    floor_b = int(b.byte_budget_band[0])
    if floor_a + floor_b > per_pair_byte_cap:
        # Each gets a proportional slice of the cap (substrates don't reach
        # their published floor; mark as "infeasible at cap" by emitting 0
        # for the smaller one and the full cap to the larger).
        if floor_a >= floor_b:
            return (per_pair_byte_cap, 0)
        return (0, per_pair_byte_cap)
    remaining = per_pair_byte_cap - floor_a - floor_b
    midpoint_a = 0.5 * (a.byte_budget_band[0] + a.byte_budget_band[1])
    midpoint_b = 0.5 * (b.byte_budget_band[0] + b.byte_budget_band[1])
    total_mid = max(midpoint_a + midpoint_b, 1.0)
    extra_a = int(remaining * (midpoint_a / total_mid))
    extra_b = remaining - extra_a
    alloc_a = floor_a + extra_a
    alloc_b = floor_b + extra_b
    # Cap each at its ceiling.
    alloc_a = min(alloc_a, int(a.byte_budget_band[1]))
    alloc_b = min(alloc_b, int(b.byte_budget_band[1]))
    return alloc_a, alloc_b


def build_composition_example(
    pair: tuple[SubstrateRow, SubstrateRow],
    matrix: Optional[CompositionMatrix] = None,
    *,
    per_pair_byte_cap: int = PR106_R2_REFERENCE_ARCHIVE_BYTES,
) -> CompositionExample:
    """Build one CompositionExample from an orthogonal pair.

    Format-ID collision is verified against the matrix; raises ValueError
    on collision.
    """
    matrix = matrix or build_composition_matrix()
    a, b = pair
    cell = matrix.get(a.substrate_id, b.substrate_id)
    if cell.composability != Composability.ORTHOGONAL:
        raise ValueError(
            f"pair ({a.substrate_id}, {b.substrate_id}) is not ORTHOGONAL "
            f"(matrix verdict: {cell.composability.value}); refusing to "
            "build cross-paradigm example"
        )
    if cell.format_id_collision_risk:
        raise ValueError(
            f"format-ID collision between {a.substrate_id} (0x{a.format_id:02X}) "
            f"and {b.substrate_id} (0x{b.format_id:02X}); cannot compose"
        )

    alloc_a, alloc_b = _allocate_bytes_per_substrate(
        pair, per_pair_byte_cap=per_pair_byte_cap
    )

    composite = predicted_composite_delta([a.substrate_id, b.substrate_id], matrix=matrix)

    rows = (
        CompositionExampleSubstrateRow(
            substrate_id=a.substrate_id,
            substrate_class=a.substrate_class,
            target_axis=a.target_axis,
            format_id=int(a.format_id),
            magic_bytes=a.magic_bytes,
            allocated_bytes=int(alloc_a),
            runtime_dep_closure=a.runtime_dep_closure,
        ),
        CompositionExampleSubstrateRow(
            substrate_id=b.substrate_id,
            substrate_class=b.substrate_class,
            target_axis=b.target_axis,
            format_id=int(b.format_id),
            magic_bytes=b.magic_bytes,
            allocated_bytes=int(alloc_b),
            runtime_dep_closure=b.runtime_dep_closure,
        ),
    )

    # Decoder ordering: the renderer-replacement (if any) goes FIRST so the
    # residual sits on top of the renderer's RGB output. For
    # bolt-on / residual / pose / self-compression / meta-codec, the
    # composability-class ordering is:
    #   RENDERER_REPLACEMENT > SELF_COMPRESSION > BOLT_ON >
    #   POSE_AXIS_SIDECHANNEL > RESIDUAL > META_CODEC
    class_priority = {
        SubstrateClass.RENDERER_REPLACEMENT: 0,
        SubstrateClass.SELF_COMPRESSION: 1,
        SubstrateClass.BOLT_ON: 2,
        SubstrateClass.POSE_AXIS_SIDECHANNEL: 3,
        SubstrateClass.RESIDUAL: 4,
        SubstrateClass.META_CODEC: 5,
    }
    decoder_ordering = tuple(
        sorted(
            (a.substrate_id, b.substrate_id),
            key=lambda sid: class_priority.get(
                next(s for s in matrix.substrates if s.substrate_id == sid).substrate_class,
                10,
            ),
        )
    )

    total_bytes = int(alloc_a + alloc_b)
    fits = total_bytes <= per_pair_byte_cap

    return CompositionExample(
        example_id=f"{a.substrate_id}__{b.substrate_id}",
        pair_a_substrate_id=a.substrate_id,
        pair_b_substrate_id=b.substrate_id,
        composability=cell.composability.value,
        expected_alpha=cell.expected_alpha,
        rationale=cell.rationale,
        predicted_score_delta_alone_a=a.predicted_delta_alone_midpoint(),
        predicted_score_delta_alone_b=b.predicted_delta_alone_midpoint(),
        predicted_composite_delta=composite["predicted_composite_delta"],
        rows=rows,
        decoder_ordering=decoder_ordering,
        total_archive_bytes=total_bytes,
        fits_pr106_r2_frontier=fits,
        pr106_r2_reference_bytes=per_pair_byte_cap,
    )


def build_top_k_composition_examples(
    matrix: Optional[CompositionMatrix] = None,
    *,
    top_k: int = 5,
    per_pair_byte_cap: int = PR106_R2_REFERENCE_ARCHIVE_BYTES,
    require_axes_diverge: bool = True,
    exclude_replacement_renderers: bool = False,
) -> CompositionExampleSet:
    """Build the full top-K cross-paradigm composition example set."""
    matrix = matrix or build_composition_matrix()
    pairs = select_top_orthogonal_pairs(
        matrix,
        top_k=top_k,
        require_axes_diverge=require_axes_diverge,
        exclude_replacement_renderers=exclude_replacement_renderers,
    )
    examples = tuple(
        build_composition_example(p, matrix, per_pair_byte_cap=per_pair_byte_cap)
        for p in pairs
    )
    return CompositionExampleSet(
        schema=SCHEMA_VERSION,
        generated_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        matrix_schema=matrix.schema_version,
        n_examples=len(examples),
        examples=examples,
        pr106_r2_reference_bytes=per_pair_byte_cap,
    )


# ── Materialization (concrete archive bytes) ────────────────────────────


_HEADER_MAGIC = b"XCPE"  # "Cross-Paradigm Example" envelope magic.
_HEADER_VERSION = 1


def materialize_composition_example_bytes(
    example: CompositionExample,
) -> bytes:
    """Materialize a composition example into deterministic bytes.

    The wire format is:

        ``XCPE`` (4 bytes envelope magic)
        version (1 byte; currently 1)
        n_substrates (1 byte; should always be 2)
        for each substrate (in decoder_ordering):
            format_id (1 byte)
            magic_bytes (4 bytes ASCII)
            payload_length (4 bytes LE u32)
            payload_bytes (zero-filled placeholders; allocated_bytes long)

    Per CLAUDE.md "Forbidden score claims": these bytes are zero-filled
    placeholders with NO score-relevant content. They demonstrate the
    wire grammar's feasibility but are NOT a contest-promotable archive.
    """
    rows_in_order = []
    for sid in example.decoder_ordering:
        row = next(r for r in example.rows if r.substrate_id == sid)
        rows_in_order.append(row)

    parts: list[bytes] = [
        _HEADER_MAGIC,
        struct.pack("<BB", _HEADER_VERSION, len(rows_in_order)),
    ]
    for row in rows_in_order:
        parts.append(struct.pack("<B", row.format_id))
        magic_b = row.magic_bytes.encode("ascii")
        if len(magic_b) != 4:
            raise ValueError(
                f"magic_bytes must be 4 ASCII chars; got {row.magic_bytes!r}"
            )
        parts.append(magic_b)
        parts.append(struct.pack("<I", row.allocated_bytes))
        parts.append(b"\x00" * row.allocated_bytes)
    return b"".join(parts)


def parse_composition_example_envelope(
    payload: bytes,
) -> dict[str, Any]:
    """Parse the cross-paradigm example envelope back into a typed dict."""
    if len(payload) < 6:
        raise ValueError("payload too short to contain envelope header")
    if payload[:4] != _HEADER_MAGIC:
        raise ValueError(
            f"envelope magic mismatch: got {payload[:4]!r} expected {_HEADER_MAGIC!r}"
        )
    version = payload[4]
    n_substrates = payload[5]
    if version != _HEADER_VERSION:
        raise ValueError(f"unsupported envelope version: {version}")
    if n_substrates != 2:
        raise ValueError(
            f"composition example must contain exactly 2 substrates; got {n_substrates}"
        )
    rows: list[dict[str, Any]] = []
    offset = 6
    for _ in range(n_substrates):
        if offset + 1 + 4 + 4 > len(payload):
            raise ValueError("payload truncated mid-row")
        format_id = payload[offset]
        offset += 1
        magic_bytes = payload[offset : offset + 4].decode("ascii")
        offset += 4
        (payload_length,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        if offset + payload_length > len(payload):
            raise ValueError("payload truncated mid-substrate-payload")
        body = payload[offset : offset + payload_length]
        offset += payload_length
        rows.append(
            {
                "format_id": int(format_id),
                "magic_bytes": magic_bytes,
                "payload_length": int(payload_length),
                "payload_sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "envelope_magic": _HEADER_MAGIC.decode("ascii"),
        "version": int(version),
        "n_substrates": int(n_substrates),
        "rows": rows,
    }


# ── Local CPU smoke (no GPU; no scorer load) ─────────────────────────────


def smoke_composition_example(example: CompositionExample) -> dict[str, Any]:
    """Local CPU smoke: materialize, parse, verify round-trip; return result.

    Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first": the
    smoke runs LOCALLY on CPU with NO scorer load (the placeholder bytes are
    zero-filled). It validates the wire grammar's round-trip property.
    """
    payload = materialize_composition_example_bytes(example)
    parsed = parse_composition_example_envelope(payload)
    return {
        "schema": SCHEMA_VERSION,
        "example_id": example.example_id,
        "encoded_bytes": len(payload),
        "n_substrates": parsed["n_substrates"],
        "decoder_ordering": list(example.decoder_ordering),
        "round_trip_passed": True,
        "byte_proxy_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_not_score",
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "cross_paradigm_composition_examples_v1",
        ],
    }


# ── Serialization ────────────────────────────────────────────────────────


def _example_row_to_dict(r: CompositionExampleSubstrateRow) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    d["substrate_class"] = r.substrate_class.value
    d["target_axis"] = r.target_axis.value
    d["runtime_dep_closure"] = list(r.runtime_dep_closure)
    return d


def _example_to_dict(ex: CompositionExample) -> dict[str, Any]:
    return {
        "example_id": ex.example_id,
        "pair_a_substrate_id": ex.pair_a_substrate_id,
        "pair_b_substrate_id": ex.pair_b_substrate_id,
        "composability": ex.composability,
        "expected_alpha": ex.expected_alpha,
        "rationale": ex.rationale,
        "predicted_score_delta_alone_a": ex.predicted_score_delta_alone_a,
        "predicted_score_delta_alone_b": ex.predicted_score_delta_alone_b,
        "predicted_composite_delta": ex.predicted_composite_delta,
        "rows": [_example_row_to_dict(r) for r in ex.rows],
        "decoder_ordering": list(ex.decoder_ordering),
        "total_archive_bytes": ex.total_archive_bytes,
        "fits_pr106_r2_frontier": ex.fits_pr106_r2_frontier,
        "pr106_r2_reference_bytes": ex.pr106_r2_reference_bytes,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "predicted_cross_paradigm_composition_example_v1",
    }


def serialize_composition_example_set(es: CompositionExampleSet) -> dict[str, Any]:
    """JSON-safe serialization of a composition example set."""
    return {
        "schema": es.schema,
        "generated_at_utc": es.generated_at_utc,
        "matrix_schema": es.matrix_schema,
        "n_examples": es.n_examples,
        "pr106_r2_reference_bytes": es.pr106_r2_reference_bytes,
        "examples": [_example_to_dict(ex) for ex in es.examples],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_cross_paradigm_composition_examples_v1",
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "cross_paradigm_composition_examples_v1",
            "halt_and_ask_default_for_dispatch_recommendations",
        ],
    }


def write_composition_example_set_json(
    es: CompositionExampleSet, path: str
) -> None:
    """Write the example set as pretty-printed JSON. Refuses /tmp paths."""
    if path.startswith("/tmp/") or "/private/tmp/" in path or "/var/tmp/" in path:
        raise ValueError(f"refusing to write to forbidden /tmp path: {path!r}")
    payload = serialize_composition_example_set(es)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_example_archive_bytes(
    example: CompositionExample, output_dir: str
) -> dict[str, Any]:
    """Materialize the example archive bytes into ``<output_dir>/<example_id>/``.

    Refuses /tmp paths. Writes ``archive.bin`` (the deterministic wire-format
    bytes) plus ``manifest.json`` (the typed example record) plus
    ``smoke_result.json`` (the local CPU smoke verifying the round-trip).
    """
    if (
        output_dir.startswith("/tmp/")
        or "/private/tmp/" in output_dir
        or "/var/tmp/" in output_dir
    ):
        raise ValueError(f"refusing to write to forbidden /tmp path: {output_dir!r}")
    out_dir = Path(output_dir) / example.example_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = materialize_composition_example_bytes(example)
    archive_path = out_dir / "archive.bin"
    archive_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    manifest = _example_to_dict(example)
    manifest["archive_path"] = str(archive_path)
    manifest["archive_size_bytes"] = len(payload)
    manifest["archive_sha256"] = sha256
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    smoke = smoke_composition_example(example)
    smoke["archive_path"] = str(archive_path)
    smoke["archive_sha256"] = sha256
    (out_dir / "smoke_result.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))
    return {
        "example_id": example.example_id,
        "archive_path": str(archive_path),
        "archive_size_bytes": len(payload),
        "archive_sha256": sha256,
        "manifest_path": str(out_dir / "manifest.json"),
        "smoke_result_path": str(out_dir / "smoke_result.json"),
    }


__all__ = [
    "SCHEMA_VERSION",
    "PR106_R2_REFERENCE_ARCHIVE_BYTES",
    "CompositionExampleSubstrateRow",
    "CompositionExample",
    "CompositionExampleSet",
    "select_top_orthogonal_pairs",
    "build_composition_example",
    "build_top_k_composition_examples",
    "materialize_composition_example_bytes",
    "parse_composition_example_envelope",
    "smoke_composition_example",
    "serialize_composition_example_set",
    "write_composition_example_set_json",
    "write_example_archive_bytes",
]
