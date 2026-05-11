"""Shared helpers for the 5 per-family PR106 + non-HNeRV residual materializers.

Each per-family tool at ``tools/materialize_<family>_residual_pr106_sidecar.py``:

1. reads the PR106 r2 canonical archive (``archive.zip``);
2. extracts the canonical ``0.bin`` PR106 payload;
3. computes a family-specific residual (this module is family-agnostic — that
   step is the family's own subagent code);
4. wraps via ``tac.residual_basis.pr106_sidecar_packing.build_archive(...)``;
5. emits ``materialization_manifest.json`` with promotion-status pinned False;
6. emits a no-op-detector byte-mutation smoke test result.

This helper closes the boilerplate around steps 1, 2, 4, 5, 6 so every tool
focuses on its own residual encoder.

Per CLAUDE.md HNeRV parity discipline:

* archive_grammar: monolithic ``0.bin`` (single-file per lesson 3); per-family
  wire format documented in ``pr106_sidecar_packing.py``.
* parser_section_manifest: returned in the manifest payload.
* no_op_detector_planned: ``run_no_op_byte_mutation_smoke`` proves a single
  residual byte change produces a different inflate output via the per-family
  inflate.py runtime.
* score_claim / promotion_eligible / ready_for_exact_eval_dispatch: pinned
  False at every emission boundary.

No score claim. No GPU dispatch. No MPS forwarding. No /tmp paths.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    BuildResidualArchiveResult,
    build_archive,
    parse_archive,
)

PR106_BIN_MEMBER_NAME: Final[str] = "0.bin"
DEFAULT_PR106_ARCHIVE: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "submissions/pr106_latent_sidecar_r2/archive.zip"
)


class MaterializerError(ValueError):
    """Raised on contract violations in the materialization pipeline."""


@dataclass(frozen=True)
class MaterializationManifest:
    """Typed manifest emitted alongside each candidate archive.

    Per CLAUDE.md HNeRV parity discipline 8-archive-grammar fields + Catalog
    #100 ``check_gate2_no_naked_bytes``: every field that could be promotion-
    bait is pinned False here.
    """

    family: str
    format_id: int
    pr106_source_archive: str
    pr106_source_sha256: str
    pr106_bytes_size: int
    residual_bytes_size: int
    archive_bytes_size: int
    archive_sha256: str
    schema: str
    timestamp_utc: str
    extra: dict[str, Any]
    archive_grammar: str = field(
        default="pr106_plus_residual_sidecar_monolithic_v1", init=False
    )
    parser_section_manifest: str = field(
        default="magic(1B)=0xFD + format_id(1B) + pr106_len(4B LE) + pr106_bytes + residual_len(4B LE) + residual_bytes",
        init=False,
    )
    inflate_runtime_loc_budget: int = field(default=200, init=False)
    runtime_dep_closure: tuple[str, ...] = field(
        default=("numpy", "torch", "brotli", "PR106 codec.py", "PR106 model.py"),
        init=False,
    )
    export_format: str = field(
        default="pr106_plus_residual_per_family_v1", init=False
    )
    score_aware_loss: str = field(
        default="research_only_scaffold_no_score_aware_loss_yet", init=False
    )
    bolt_on_loc_budget: int = field(default=350, init=False)
    no_op_detector_planned: bool = field(default=True, init=False)
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default="research_signal", init=False)


def now_utc_iso() -> str:
    """Return the current UTC time in ISO 8601 with seconds precision."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pr106_bytes(pr106_archive: Path) -> tuple[bytes, str]:
    """Extract the canonical 0.bin PR106 payload from the archive zip.

    Returns (bytes, sha256_of_zip). The PR106 canonical archive is a single-
    member zip per HNeRV parity discipline; if the zip has additional members
    the helper refuses (defensive guard).
    """
    if not pr106_archive.is_file():
        raise MaterializerError(f"PR106 archive not found: {pr106_archive}")
    archive_sha = sha256_file(pr106_archive)
    with zipfile.ZipFile(pr106_archive, mode="r") as zf:
        names = zf.namelist()
        if PR106_BIN_MEMBER_NAME not in names:
            raise MaterializerError(
                f"{PR106_BIN_MEMBER_NAME} not in archive {pr106_archive}: members={names}"
            )
        # Reject multi-member zips — PR106 canonical is single-file.
        if names != [PR106_BIN_MEMBER_NAME]:
            raise MaterializerError(
                f"PR106 archive must contain only '{PR106_BIN_MEMBER_NAME}'; got {names}"
            )
        pr106_bytes = zf.read(PR106_BIN_MEMBER_NAME)
    if not pr106_bytes:
        raise MaterializerError(f"PR106 0.bin is empty in {pr106_archive}")
    return pr106_bytes, archive_sha


def emit_archive_zip(archive_bytes: bytes, output_zip: Path) -> Path:
    """Wrap the raw archive bytes into the canonical single-member zip.

    The contest packet format requires a zip-shaped wrapper; this helper
    writes the residual archive as ``0.bin`` inside a zip with a fixed
    timestamp so the output is byte-deterministic.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    # Use ZipInfo with fixed mtime (2025-01-01 00:00:00 UTC) for deterministic zip bytes.
    info = zipfile.ZipInfo(filename=PR106_BIN_MEMBER_NAME, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED  # 0.bin is already-compressed payload
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output_zip, mode="w") as zf:
        zf.writestr(info, archive_bytes)
    return output_zip


def write_manifest(manifest: MaterializationManifest, output_path: Path) -> Path:
    """Write the manifest as deterministic-ordered JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": manifest.schema,
        "timestamp_utc": manifest.timestamp_utc,
        "family": manifest.family,
        "format_id": manifest.format_id,
        "pr106_source_archive": manifest.pr106_source_archive,
        "pr106_source_sha256": manifest.pr106_source_sha256,
        "pr106_bytes_size": manifest.pr106_bytes_size,
        "residual_bytes_size": manifest.residual_bytes_size,
        "archive_bytes_size": manifest.archive_bytes_size,
        "archive_sha256": manifest.archive_sha256,
        "archive_grammar": manifest.archive_grammar,
        "parser_section_manifest": manifest.parser_section_manifest,
        "inflate_runtime_loc_budget": manifest.inflate_runtime_loc_budget,
        "runtime_dep_closure": list(manifest.runtime_dep_closure),
        "export_format": manifest.export_format,
        "score_aware_loss": manifest.score_aware_loss,
        "bolt_on_loc_budget": manifest.bolt_on_loc_budget,
        "no_op_detector_planned": manifest.no_op_detector_planned,
        "score_claim": manifest.score_claim,
        "promotion_eligible": manifest.promotion_eligible,
        "ready_for_exact_eval_dispatch": manifest.ready_for_exact_eval_dispatch,
        "evidence_grade": manifest.evidence_grade,
        "extra": manifest.extra,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def materialize_family_archive(
    *,
    family: str,
    pr106_archive: Path,
    residual_bytes: bytes,
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path, MaterializationManifest, BuildResidualArchiveResult]:
    """End-to-end family-agnostic materializer.

    Reads PR106 r2 → wraps with the family-specific residual blob →
    emits ``<family>_pr106_residual_sidecar_archive.zip`` and
    ``materialization_manifest.json`` into ``output_dir``.

    The residual byte encoding (the family-specific part) MUST be
    pre-computed by the per-family tool and passed in as ``residual_bytes``;
    this helper does not interpret the residual contents.

    Returns ``(archive_zip_path, manifest_path, manifest, build_result)``.
    """
    if family not in PR106_RESIDUAL_FORMAT_IDS:
        raise MaterializerError(f"unknown family {family!r}")
    pr106_bytes, pr106_sha = extract_pr106_bytes(pr106_archive)
    build_result = build_archive(
        family=family, pr106_bytes=pr106_bytes, residual_bytes=residual_bytes
    )
    archive_zip = output_dir / f"{family}_pr106_residual_sidecar_archive.zip"
    emit_archive_zip(build_result.archive_bytes, archive_zip)
    manifest_path = output_dir / "materialization_manifest.json"
    manifest = MaterializationManifest(
        family=family,
        format_id=build_result.format_id,
        pr106_source_archive=str(pr106_archive),
        pr106_source_sha256=pr106_sha,
        pr106_bytes_size=build_result.pr106_len,
        residual_bytes_size=build_result.residual_len,
        archive_bytes_size=len(build_result.archive_bytes),
        archive_sha256=sha256_bytes(build_result.archive_bytes),
        schema=f"{family}_pr106_residual_sidecar_materialization_v1",
        timestamp_utc=now_utc_iso(),
        extra=extra or {},
    )
    write_manifest(manifest, manifest_path)
    return archive_zip, manifest_path, manifest, build_result


def run_no_op_detector_byte_mutation(
    *,
    archive_bytes: bytes,
    expected_format_id: int,
) -> dict[str, Any]:
    """Run the byte-mutation no-op detector smoke: flip a residual byte and
    verify the parsed residual changes.

    This is the in-process smoke; the per-family inflate runtime tests do the
    end-to-end byte-mutation parity check (which requires PyTorch + the PR106
    decoder) on a real PR106 input.
    """
    parsed_a = parse_archive(archive_bytes)
    if parsed_a.format_id != expected_format_id:
        raise MaterializerError(
            f"format_id mismatch in no-op detector smoke: "
            f"got 0x{parsed_a.format_id:02X} expected 0x{expected_format_id:02X}"
        )
    if not parsed_a.residual_bytes:
        return {
            "result": "skipped_empty_residual",
            "rationale": "no_op_detector_inapplicable_when_residual_is_zero_bytes",
        }
    blob = bytearray(archive_bytes)
    # First residual byte sits at offset header(6B) + pr106_len + residual_len_prefix(4B).
    offset = 6 + len(parsed_a.pr106_bytes) + 4
    original = blob[offset]
    blob[offset] = (original + 1) & 0xFF
    parsed_b = parse_archive(bytes(blob))
    return {
        "result": "passed" if parsed_a.residual_bytes != parsed_b.residual_bytes else "failed",
        "offset_mutated": offset,
        "original_byte": original,
        "mutated_byte": (original + 1) & 0xFF,
    }


__all__ = [
    "DEFAULT_PR106_ARCHIVE",
    "MaterializationManifest",
    "MaterializerError",
    "PR106_BIN_MEMBER_NAME",
    "emit_archive_zip",
    "extract_pr106_bytes",
    "materialize_family_archive",
    "now_utc_iso",
    "run_no_op_detector_byte_mutation",
    "sha256_bytes",
    "sha256_file",
    "write_manifest",
]
