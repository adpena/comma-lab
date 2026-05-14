# SPDX-License-Identifier: MIT
"""Materialize a film_pose_conditioning × magic_codec composition probe archive.

Per grand council Cluster 1 verdict 2026-05-12 (BUILD-ONE-PROBE; subagent
``a6046518ed0ec4869``). The composition pair `film_pose_conditioning ×
magic_codec` is the **most orthogonal** cell in the substrate-composition
matrix:

* ``magic_codec`` operates on the WEIGHT-TENSOR axis (rate, format_id 0xF0,
  magic "MGIC").
* ``film_pose_conditioning`` is a BOLT_ON acting on the renderer-INPUT-pose
  axis (pose, format_id 0x81, magic "FILM"). It declares a small MLP weight
  bundle (byte_budget_band 2_000-8_000 bytes per
  ``tac.optimization.substrate_composition_matrix``).

The orthogonality means the composition is the highest-information probe for
whether the saturation hypothesis (PR106 r2 being entropy-saturated → magic
codec REGRESSES bytes) generalizes from the singleton case to multi-substrate
pairs. The dual-base prescription (PR106 r2 + A1) disambiguates "saturation
on this specific base" vs "magic-codec-broken on every base".

Per CLAUDE.md non-negotiables:

* ``score_claim`` permanently False.
* ``promotion_eligible`` permanently False.
* ``ready_for_exact_eval_dispatch`` permanently False.
* ``cuda_eval_worth_testing`` permanently False — the probe is BYTE-PROXY ONLY.
* No scorer load at materialization time.
* No ``/tmp`` paths persisted; output to ``--output-dir`` enforced via
  ``tac.output_path_policy``.
* Deterministic-bytes: same source + same flags → byte-identical output.
* Lane is research-only by construction (the FiLM section is a synthetic
  declared-bytes placeholder, not a trained MLP) per Catalog #124
  ``research_only=true`` opt-out for the archive-grammar requirement.

The probe answers ONE question:

    Does film_pose_conditioning × magic_codec show different byte behavior
    than magic_codec ALONE on the same base?

If empirical Δ ≥ -100 bytes on either base → BUILD-REMAINING-3-CELLS
recommendation (the composition matrix's predicted-Δ has structure beyond
the singleton's saturation signal).

If both bases regress → the saturation hypothesis is reinforced; the
composition matrix should be ABANDONED on saturated bases entirely.

CLAUDE.md gates honoured:

* ``check_artifact_lifecycle_compliance`` (Catalog #113): the emitted
  ``composition_runtime_manifest.json`` carries DERIVED_OUTPUT regen header
  (``generated_at`` + ``from_state_hash``).
* ``check_no_op_proof_promotes_to_blocker`` (Catalog #139): the runtime
  manifest lists the no-byte-closed-runtime + no-inflate-parity blockers.
* ``check_representation_lane_has_archive_grammar`` (Catalog #124): the
  composition lane declares ``research_only=true`` because the FiLM
  section is a probe-time synthesized weight bundle, not a trained MLP that
  ships in a contest packet.

Usage::

    python tools/materialize_film_pose_x_magic_codec_archive.py \\
        --source-archive experiments/results/.../pr106_r2/archive.zip \\
        --base-label pr106_r2 \\
        --output-dir experiments/results/b1_film_pose_x_magic_codec_<ts>/pr106_r2 \\
        --film-section-bytes 4096 \\
        --film-section-seed 20260512
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tac.output_path_policy import assert_not_temporary_output_dir
from tac.packet_compiler.magic_codec import (
    MagicCodecError,
    MagicCodecResult,
    StreamHint,
    encode_magic_codec,
)


# ── Composition constants ────────────────────────────────────────────────

# Per ``substrate_composition_matrix.py:545-558`` — FiLM pose conditioning
# is declared format_id 0x81 / magic "FILM" / byte budget 2_000-8_000.
_FILM_FORMAT_ID = 0x81
_FILM_MAGIC = b"FILM"
_FILM_BUDGET_MIN = 2_000
_FILM_BUDGET_MAX = 8_000

# Magic codec writes singleton/PR106 r2-compatible primitives 0xF0-0xF2.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMISSION_RUNTIME_PATH = (
    _REPO_ROOT / "submissions" / "magic_codec_pr106_r2" / "inflate.py"
)
_SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS = frozenset({0xF0, 0xF1, 0xF2})


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="materialize_film_pose_x_magic_codec_archive",
        description=(
            "Build the film_pose_conditioning × magic_codec composition "
            "probe archive on a single base. Byte-proxy ONLY — no score "
            "claim. Per grand council Cluster 1 verdict 2026-05-12."
        ),
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Source base archive (PR106 r2 or A1) to compose on.",
    )
    parser.add_argument(
        "--base-label",
        required=True,
        choices=["pr106_r2", "a1", "other"],
        help=(
            "Base identifier for manifest provenance. pr106_r2 is the "
            "saturated base; a1 is the non-saturated base."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory. MUST NOT be under /tmp (CLAUDE.md "
            "forbidden pattern). Canonical: experiments/results/<lane>_<ts>/."
        ),
    )
    parser.add_argument(
        "--film-section-bytes",
        type=int,
        default=4096,
        help=(
            "Synthetic FiLM section byte budget. Must be within the "
            f"declared band [{_FILM_BUDGET_MIN}, {_FILM_BUDGET_MAX}] from "
            "substrate_composition_matrix.py. Default 4096 = midpoint."
        ),
    )
    parser.add_argument(
        "--film-section-seed",
        type=int,
        default=20260512,
        help=(
            "Deterministic seed for the synthetic FiLM weight payload. "
            "Default 20260512 (probe date)."
        ),
    )
    parser.add_argument(
        "--magic-codec-stream-type",
        choices=("weight_tensor",),
        default="weight_tensor",
        help=(
            "Stream-type hint for the magic codec layer. Only "
            "'weight_tensor' is supported for the probe (matches the "
            "singleton-falsification baseline)."
        ),
    )
    parser.add_argument(
        "--magic-codec-selection-strategy",
        choices=("smallest_byte_count", "entropy_estimate"),
        default="smallest_byte_count",
        help="How the magic codec auto-selects among candidate primitives.",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Operator handle for manifest provenance. Default 'unknown'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the manifest WITHOUT writing any output. Useful for "
            "ranking the probe before committing the build."
        ),
    )
    return parser.parse_args(argv)


# ── FiLM section synthesis (deterministic, declared-bytes placeholder) ──


def _synthesize_film_section_bytes(
    *, target_bytes: int, seed: int
) -> bytes:
    """Generate a deterministic, declared-bytes FiLM section payload.

    The FiLM section is a SYNTHETIC weight bundle for the byte-proxy probe.
    It is NOT a trained MLP. It carries the FiLM format envelope so the
    composition manifest can verifiably attribute byte cost to the FiLM
    bolt-on layer separately from the magic_codec layer.

    Wire format (probe-only, NOT a contest-compliant runtime format)::

        bytes  0..4   : magic "FILM" (4 bytes)
        bytes  4..5   : format_id 0x81 (1 byte)
        bytes  5..6   : version 0x01 (1 byte)
        bytes  6..10  : section_byte_count uint32 LE (4 bytes)
        bytes 10..N   : deterministic int8 PRNG weight payload

    The PRNG is ``numpy.random.default_rng(seed)`` integers in [-7, 7] —
    keeps entropy below 4 bits/symbol so the FiLM section is itself a
    representative-looking small MLP weight stream rather than uniform
    random bytes. The choice has no bearing on the byte-proxy probe; it
    only ensures the section is a credible placeholder for a real trained
    FiLM bolt-on.
    """
    if not (_FILM_BUDGET_MIN <= target_bytes <= _FILM_BUDGET_MAX):
        raise SystemExit(
            f"--film-section-bytes={target_bytes} outside declared band "
            f"[{_FILM_BUDGET_MIN}, {_FILM_BUDGET_MAX}] for "
            "film_pose_conditioning (per substrate_composition_matrix.py)"
        )

    header = (
        _FILM_MAGIC
        + struct.pack("<BB", _FILM_FORMAT_ID, 0x01)
        + struct.pack("<I", target_bytes)
    )
    body_len = target_bytes - len(header)
    if body_len < 0:
        raise SystemExit(
            f"target_bytes={target_bytes} is smaller than FiLM header "
            f"({len(header)} bytes); pick a larger budget"
        )

    rng = np.random.default_rng(seed)
    body = rng.integers(low=-7, high=8, size=body_len, dtype=np.int8).tobytes()
    return header + body


# ── Magic codec layer (re-uses the singleton-build path) ────────────────


def _decode_dense_from_member_bytes(
    member_bytes: bytes, *, quantize_bits: int = 8
) -> np.ndarray:
    """Interpret an archive member's bytes as a dense int8 stream.

    Matches the singleton baseline at
    ``tools/materialize_magic_codec_archive.py::_decode_dense_from_member_bytes``
    for the weight_tensor stream-type with the default quantize_bits=8.
    Kept inline here so this probe tool is self-contained for review.
    """
    if quantize_bits != 8:
        raise SystemExit(
            f"probe only supports quantize_bits=8; got {quantize_bits}"
        )
    arr = np.frombuffer(member_bytes, dtype=np.int8).astype(np.int8, copy=True)
    return arr


def _assert_submission_runtime_can_roundtrip(
    member_name: str, member_bytes: bytes, result: MagicCodecResult
) -> None:
    """Fail closed unless the submission runtime can decode the magic codec
    payload byte-identically to the source bytes.

    Mirrors the parity gate in ``materialize_magic_codec_archive.py``.
    """
    if result.selected_primitive_id not in _SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS:
        raise MagicCodecError(
            f"{member_name}: selected primitive_id "
            f"0x{result.selected_primitive_id:02X} is not supported by "
            "submissions/magic_codec_pr106_r2/inflate.py"
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "_magic_codec_pr106_r2_inflate", _SUBMISSION_RUNTIME_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load runtime at {_SUBMISSION_RUNTIME_PATH}"
            )
        runtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime_module)
        decoded = runtime_module._decode_envelope_to_inner_bytes(result.payload)
    except Exception as exc:
        raise MagicCodecError(
            f"{member_name}: submission runtime refused selected magic-codec "
            f"payload: {exc}"
        ) from exc
    if decoded != member_bytes:
        raise MagicCodecError(
            f"{member_name}: submission runtime decode is not byte-identical "
            "to the source member; refusing non-parity materialization"
        )


def _apply_magic_codec_to_member(
    member_name: str,
    member_bytes: bytes,
    *,
    stream_type: str,
    selection_strategy: str,
) -> tuple[MagicCodecResult, dict[str, object]]:
    """Run the magic codec auto-selector on one base archive member."""
    dense = _decode_dense_from_member_bytes(member_bytes)
    result = encode_magic_codec(
        dense,
        hint=StreamHint(stream_type),
        selection_strategy=selection_strategy,
    )
    _assert_submission_runtime_can_roundtrip(member_name, member_bytes, result)
    row = {
        "member_name": member_name,
        "source_bytes": len(member_bytes),
        "dense_shape": list(dense.shape),
        "dense_dtype": str(dense.dtype),
        "magic_codec_payload_bytes": len(result.payload),
        "inner_primitive_byte_count": result.inner_primitive_byte_count,
        "selected_primitive": result.selected_primitive,
        "selected_primitive_id": result.selected_primitive_id,
        "selection_strategy": result.selection_strategy,
        "magic_codec_byte_delta": len(result.payload) - len(member_bytes),
        "selection_log": [
            {
                "primitive_name": c.primitive_name,
                "primitive_id": c.primitive_id,
                "encoded_bytes": len(c.encoded_bytes),
                "refused": c.refused,
                "refusal_reason": c.refusal_reason,
            }
            for c in result.selection_log
        ],
    }
    return result, row


# ── Composition archive emission ────────────────────────────────────────


def _emit_composition_archive(
    *,
    archive_path: Path,
    magic_codec_member_name: str,
    magic_codec_payload: bytes,
    film_section_bytes: bytes,
) -> None:
    """Write a deterministic monolithic ZIP archive carrying both sections.

    The archive grammar is the simplest possible monolithic packet: each
    composition layer occupies one ZIP member.

    * ``<member_name>`` — magic_codec envelope payload (rewrite of source).
    * ``film.bin`` — FiLM section bytes (bolt-on layer).

    The ZIP uses ``ZIP_STORED`` + fixed 1980-01-01 timestamps for
    deterministic-bytes per CLAUDE.md "Deterministic packet compiler".
    """
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info_mc = zipfile.ZipInfo(
            filename=magic_codec_member_name, date_time=(1980, 1, 1, 0, 0, 0)
        )
        zf.writestr(info_mc, magic_codec_payload)
        info_film = zipfile.ZipInfo(
            filename="film.bin", date_time=(1980, 1, 1, 0, 0, 0)
        )
        zf.writestr(info_film, film_section_bytes)


# ── Manifests ────────────────────────────────────────────────────────────


def _build_composition_runtime_manifest(
    *, source_hash: str, base_label: str
) -> dict[str, object]:
    """Emit the runtime-closure manifest per Catalog #139 + #124.

    All refusal flags permanently False (probe is byte-proxy ONLY).
    Lane is ``research_only=true`` per Catalog #124 opt-out — the FiLM
    section is a synthetic placeholder, not a trained MLP.
    """
    return {
        "schema": "composition_runtime_manifest.v1",
        # Catalog #113 DERIVED_OUTPUT regen header.
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "from_state_hash": source_hash,
        # Composition declaration.
        "composition_cell": (
            "orthogonal_pair__film_pose_conditioning__magic_codec"
        ),
        "base_label": base_label,
        "substrate_a": {
            "substrate_id": "film_pose_conditioning",
            "format_id": f"0x{_FILM_FORMAT_ID:02X}",
            "magic_bytes": "FILM",
            "byte_budget_band": [_FILM_BUDGET_MIN, _FILM_BUDGET_MAX],
            "target_axis": "POSE",
        },
        "substrate_b": {
            "substrate_id": "magic_codec",
            "format_id": "0xF0",
            "magic_bytes": "MGIC",
            "target_axis": "RATE",
        },
        # Catalog #124 (research_only opt-out — probe is not a trained MLP).
        "research_only": True,
        "research_only_rationale": (
            "FiLM section is a deterministic synthetic weight bundle for the "
            "byte-proxy composition probe per grand council Cluster 1 "
            "verdict 2026-05-12. NOT a trained MLP; NOT contest-shippable."
        ),
        # Runtime dep closure (planning-only).
        "runtime_dep_closure": [
            "numpy",
            "brotli",
            "lzma",
            "constriction",
            "repo_tac_required_until_vendored",
            "film_pose_conditioning_decoder_NOT_YET_BUILT",
        ],
        "inflate_runtime_loc_budget": 200,
        "decoder_module": "tac.packet_compiler.magic_codec",
        "magic_envelope": "MAGC",
        "primitive_id_namespace": "0xF0-0xFF",
        "submission_supported_primitive_ids": [
            f"0x{pid:02X}"
            for pid in sorted(_SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS)
        ],
        # Permanently-False refusal flags (byte-proxy probe).
        "runtime_tree_byte_closed": False,
        "score_aware_loss": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "rank_or_kill_eligible": False,
        # Catalog #139 first-class blockers (promoted, not just advisory).
        "blockers": [
            "film_pose_conditioning_decoder_not_built",
            "composition_inflate_parity_not_run",
            "no_byte_closed_runtime_packet_built",
            "research_adapter_runtime_depends_on_repo_tac_until_vendored",
            "film_section_synthetic_not_trained_mlp",
        ],
        # No-op proof (per Catalog #139): the probe IS the composition; no
        # claim is made that the inflate.sh runtime would consume the FiLM
        # section because no such runtime exists yet.
        "no_op_proof": {
            "runtime_consumes_film_section": False,
            "runtime_consumes_magic_codec_bytes": False,
            "no_op_detector_passed": False,
            "no_op_detector_status": (
                "advisory_only__byte_proxy_probe__no_inflate_runtime_exists"
            ),
        },
    }


def _build_selection_manifest(
    *,
    source: Path,
    source_hash: str,
    source_bytes: int,
    base_label: str,
    operator: str | None,
    member_name: str,
    magic_codec_row: dict[str, object],
    film_section_bytes: int,
    film_section_seed: int,
    composition_archive_path: Path,
    composition_archive_hash: str,
) -> dict[str, object]:
    """Build the full selection manifest with both layer attributions."""
    # Compute composition byte deltas (the probe's actual deliverable).
    mc_bytes = int(magic_codec_row["magic_codec_payload_bytes"])
    composition_total = int(composition_archive_path.stat().st_size)
    delta_vs_source = composition_total - source_bytes
    return {
        "schema": "film_pose_x_magic_codec_composition_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator or "unknown",
        "composition_cell": (
            "orthogonal_pair__film_pose_conditioning__magic_codec"
        ),
        "base_label": base_label,
        "source_archive_path": str(source),
        "source_archive_sha256": source_hash,
        "source_archive_size_bytes": source_bytes,
        "magic_codec_layer": magic_codec_row,
        "film_section_layer": {
            "format_id": f"0x{_FILM_FORMAT_ID:02X}",
            "magic_bytes": "FILM",
            "section_byte_count": film_section_bytes,
            "section_byte_budget_band": [_FILM_BUDGET_MIN, _FILM_BUDGET_MAX],
            "section_seed": film_section_seed,
            "section_kind": "synthetic_deterministic_int8_prng_placeholder",
            "section_role": (
                "byte_proxy_for_film_pose_conditioning_mlp_weights"
            ),
        },
        "composition_archive": {
            "archive_path": str(composition_archive_path),
            "archive_sha256": composition_archive_hash,
            "archive_total_bytes": composition_total,
            "magic_codec_member_name": member_name,
            "film_member_name": "film.bin",
        },
        "byte_delta_analysis": {
            "source_archive_bytes": source_bytes,
            "magic_codec_payload_bytes": mc_bytes,
            "film_section_bytes": film_section_bytes,
            "composition_archive_total_bytes": composition_total,
            "delta_vs_source_bytes": delta_vs_source,
            "magic_codec_layer_byte_delta": int(
                magic_codec_row["magic_codec_byte_delta"]
            ),
            "film_layer_byte_delta": film_section_bytes,
            # Zip overhead (header bytes per member) is non-payload but
            # part of the on-disk total; record it explicitly so the
            # interpretation in the landing memo is rigorous.
            "zip_overhead_bytes": (
                composition_total - mc_bytes - film_section_bytes
            ),
        },
        # All score-related flags permanently False — byte-proxy probe.
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "rank_or_kill_eligible": False,
        "research_only": True,
        "blockers": [
            "film_pose_conditioning_decoder_not_built",
            "composition_inflate_parity_not_run",
            "no_byte_closed_runtime_packet_built",
            "research_adapter_runtime_depends_on_repo_tac_until_vendored",
            "film_section_synthetic_not_trained_mlp",
        ],
    }


# ── Output dir guard ────────────────────────────────────────────────────


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse forbidden /tmp paths per CLAUDE.md non-negotiable."""
    try:
        assert_not_temporary_output_dir(
            output_dir,
            tool_name="materialize_film_pose_x_magic_codec_archive",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_output_dir(args.output_dir)

    source = args.source_archive.resolve()
    if not source.exists():
        raise SystemExit(f"source archive {source!s} does not exist")
    if not zipfile.is_zipfile(source):
        raise SystemExit(f"source {source!s} is not a valid ZIP archive")

    source_bytes_total = source.stat().st_size
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    # The probe ALWAYS processes the first member (matches the singleton
    # baseline at materialize_magic_codec_archive.py which iterates all
    # members in the same default mode). Both PR106 r2 and A1 are
    # monolithic single-member archives, so this is unambiguous.
    with zipfile.ZipFile(source, "r") as zf:
        members = zf.namelist()
        if not members:
            raise SystemExit(f"source archive {source!s} has no members")
        if len(members) > 1:
            raise SystemExit(
                f"source archive has {len(members)} members; the probe is "
                "monolithic-archive-only. Provide a single-member archive."
            )
        member_name = members[0]
        member_bytes = zf.read(member_name)

    # Layer 1: magic codec on the source member.
    try:
        mc_result, mc_row = _apply_magic_codec_to_member(
            member_name,
            member_bytes,
            stream_type=args.magic_codec_stream_type,
            selection_strategy=args.magic_codec_selection_strategy,
        )
    except MagicCodecError as exc:
        raise SystemExit(
            f"refusing to materialize composition probe: {exc}"
        ) from exc

    # Layer 2: synthesize the FiLM section.
    film_bytes = _synthesize_film_section_bytes(
        target_bytes=args.film_section_bytes,
        seed=args.film_section_seed,
    )

    out_dir = args.output_dir.resolve()
    composition_archive_path = out_dir / "composition_archive.zip"

    if args.dry_run:
        # Print a dry-run manifest using a hypothetical (uncreated) archive.
        dry_manifest = _build_selection_manifest(
            source=source,
            source_hash=source_hash,
            source_bytes=source_bytes_total,
            base_label=args.base_label,
            operator=args.operator,
            member_name=member_name,
            magic_codec_row=mc_row,
            film_section_bytes=len(film_bytes),
            film_section_seed=args.film_section_seed,
            composition_archive_path=composition_archive_path,
            composition_archive_hash="<dry-run-no-archive>",
        )
        # Patch the composition-archive size to the hypothetical zip-stored
        # total: 2 ZipInfo headers (~30 bytes each) + 2 names + 2 payloads.
        dry_total = (
            len(mc_result.payload)
            + len(film_bytes)
            + 30 * 2
            + len(member_name)
            + len("film.bin")
        )
        dry_manifest["composition_archive"]["archive_total_bytes"] = dry_total
        dry_manifest["byte_delta_analysis"][
            "composition_archive_total_bytes"
        ] = dry_total
        dry_manifest["byte_delta_analysis"]["delta_vs_source_bytes"] = (
            dry_total - source_bytes_total
        )
        dry_manifest["byte_delta_analysis"]["zip_overhead_bytes"] = (
            dry_total - len(mc_result.payload) - len(film_bytes)
        )
        json.dump(dry_manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_composition_archive(
        archive_path=composition_archive_path,
        magic_codec_member_name=member_name,
        magic_codec_payload=mc_result.payload,
        film_section_bytes=film_bytes,
    )
    composition_archive_hash = hashlib.sha256(
        composition_archive_path.read_bytes()
    ).hexdigest()

    # Write the selection manifest.
    manifest = _build_selection_manifest(
        source=source,
        source_hash=source_hash,
        source_bytes=source_bytes_total,
        base_label=args.base_label,
        operator=args.operator,
        member_name=member_name,
        magic_codec_row=mc_row,
        film_section_bytes=len(film_bytes),
        film_section_seed=args.film_section_seed,
        composition_archive_path=composition_archive_path,
        composition_archive_hash=composition_archive_hash,
    )
    (out_dir / "composition_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Write the runtime manifest (Catalog #113 DERIVED_OUTPUT format).
    runtime_manifest = _build_composition_runtime_manifest(
        source_hash=source_hash, base_label=args.base_label
    )
    (out_dir / "composition_runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n"
    )

    # Operator-facing summary.
    delta = manifest["byte_delta_analysis"]["delta_vs_source_bytes"]
    print(
        f"[film_pose_x_magic_codec] base={args.base_label} "
        f"src_bytes={source_bytes_total} → composition_bytes="
        f"{composition_archive_path.stat().st_size} "
        f"(Δ={delta:+} bytes; sha={composition_archive_hash[:8]})"
    )
    print(
        f"  magic_codec_layer: source={mc_row['source_bytes']} → "
        f"payload={mc_row['magic_codec_payload_bytes']} "
        f"(Δ={mc_row['magic_codec_byte_delta']:+} bytes; "
        f"primitive={mc_row['selected_primitive']})"
    )
    print(
        f"  film_section_layer: bytes={len(film_bytes)} "
        f"(Δ=+{len(film_bytes)} bytes; synthetic_placeholder)"
    )
    print(
        f"  manifest: {out_dir / 'composition_selection_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
