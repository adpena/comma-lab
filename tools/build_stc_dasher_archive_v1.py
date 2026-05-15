#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an STC-Dasher v1 substrate-agnostic archive bolt-on candidate.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #6 (Filler +
MacKay + Shannon). Substrate-agnostic rate-axis bit-shaver: applies to
ANY substrate's archive.

Usage
-----
::

    .venv/bin/python tools/build_stc_dasher_archive_v1.py \\
        --source-archive submissions/a1/archive.zip \\
        --out-dir experiments/results/stc_dasher_a1_v1_<timestamp> \\
        --sigma 0.0 \\
        --constraint-length 12 \\
        --context-length 2

What it does
------------
1. Reads the source archive bytes (substrate-agnostic; any ``.zip``).
2. Applies STC-Dasher v1 codec post-hoc to the archive bytes via
   :func:`tac.codecs.stc_dasher.encode_stream`.
3. Verifies byte-stable roundtrip (encode -> decode -> identity).
4. Wraps the encoded bytes in a deterministic ZIP envelope at
   ``<out-dir>/archive_stc_dasher_v1_<sha256_short>.zip``.
5. Emits a cost-band manifest at ``<out-dir>/build_manifest.json`` with
   ``predicted_delta_s_band: null`` + ``score_claim=false`` +
   ``byte_anchor_only=true`` per CLAUDE.md "Apples-to-apples evidence
   discipline" + Catalog #92 sister surfaces. The v1 scaffold intentionally
   preserves the original bytes in a residual envelope, so it is expected to
   be rate-negative until the Viterbi inverse / true entropy-coded payload
   lands.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline": this
tool produces a CANDIDATE archive only. NO score claim until a
contest-CUDA / contest-CPU eval anchor lands per the auth-eval
non-negotiables.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": all
output paths are under ``experiments/results/`` namespace.

Per CLAUDE.md "tac stays clean": this CLI delegates to
:mod:`tac.codecs.stc_dasher` library code; the CLI itself is a thin
build-and-pack wrapper.

Lane: ``lane_stc_dasher_scaffold_v1_20260515``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codecs.stc_dasher import (  # noqa: E402
    SCAFFOLD_ONLY,
    STC_DASHER_SCHEMA_VERSION,
    decode_stream,
    encode_stream,
)


@dataclass
class BuildManifest:
    """Cost-band manifest for STC-Dasher v1 candidate archives.

    Per CLAUDE.md "Bit-level deconstruction and entropy discipline" +
    Catalog #92 sister "evidence row archive bytes has provenance".
    """

    schema_version: str
    lane_id: str
    catalog_anchor: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_bytes: int
    candidate_archive_path: str
    candidate_archive_sha256: str
    candidate_archive_bytes: int
    delta_bytes: int
    encoded_payload_bytes: int
    syndrome_bytes: int
    overhead_bytes: int
    estimated_arithmetic_bits: float
    constraint_length: int
    context_length: int
    sigma: float
    sigma_int8: int
    roundtrip_byte_stable: bool
    predicted_delta_s_band: tuple[float, float] | None
    predicted_delta_s_band_after_viterbi_inverse: tuple[float, float]
    score_claim: bool
    byte_anchor_only: bool
    score_claim_possible_after_result_review: bool
    contest_axis_anchor: str
    custody_status: str
    cuda_eval_worth_testing: bool
    ready_for_exact_eval_dispatch: bool
    measured_config_status: str
    review_blockers: tuple[str, ...]
    built_at_utc: str
    scaffold_only: bool
    notes: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_deterministic_zip(out_path: Path, member_name: str, payload: bytes) -> None:
    """Pack a single payload into a ZIP with deterministic timestamp.

    Per CLAUDE.md FORBIDDEN_PATTERNS check 19
    `check_archive_builders_use_deterministic_zip`: use ``ZipInfo +
    writestr`` with fixed timestamp instead of ``ZipFile.write``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(out_path, "w") as zf:
        zf.writestr(info, payload)


def build_stc_dasher_candidate(
    *,
    source_archive: Path,
    out_dir: Path,
    sigma: float,
    constraint_length: int,
    context_length: int,
    payload_bit_ratio: int,
    notes: str = "",
    max_source_bytes: int = 65_536,
) -> BuildManifest:
    """Build one STC-Dasher v1 candidate archive + manifest.

    Parameters
    ----------
    max_source_bytes
        Fail-fast cap for v1 scaffold builds. This builder never truncates
        input bytes because a truncated source is not a byte-closed candidate
        for the supplied archive. Use ``0`` only for an intentional full-archive
        timing run.

    Returns the manifest object (also written to disk).
    """
    if not source_archive.exists():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_bytes_full = source_archive.read_bytes()
    src_sha = _sha256(src_bytes_full)
    src_len = len(src_bytes_full)
    if max_source_bytes > 0 and src_len > max_source_bytes:
        raise ValueError(
            f"source archive is {src_len} bytes, above max_source_bytes="
            f"{max_source_bytes}. Refusing slow scaffold build rather than "
            "emitting a truncated non-byte-closed candidate. Use "
            "--max-source-bytes 0 only for an intentional full-archive timing "
            "run; v1 is byte-anchor-only and expected rate-negative until the "
            "Viterbi inverse lands."
        )
    src_bytes = src_bytes_full

    enc_result = encode_stream(
        src_bytes,
        sigma,
        constraint_length=constraint_length,
        context_length=context_length,
        payload_bit_ratio=payload_bit_ratio,
    )

    # Mandatory roundtrip-byte-stable verification before manifest emission.
    dec_result = decode_stream(enc_result.encoded_bytes, sigma=sigma)
    roundtrip_ok = dec_result.decoded_bytes == src_bytes
    if not roundtrip_ok:
        raise RuntimeError(
            "STC-Dasher v1 roundtrip failed - refusing to emit candidate archive"
        )

    cand_member = "stc_dasher_v1.bin"
    cand_short = src_sha[:12]
    cand_path = out_dir / f"archive_stc_dasher_v1_{cand_short}.zip"
    _build_deterministic_zip(cand_path, cand_member, enc_result.encoded_bytes)
    cand_bytes_disk = cand_path.read_bytes()
    cand_sha = _sha256(cand_bytes_disk)

    delta = len(cand_bytes_disk) - src_len

    manifest = BuildManifest(
        schema_version=f"stc_dasher_build_manifest_v{STC_DASHER_SCHEMA_VERSION}",
        lane_id="lane_stc_dasher_scaffold_v1_20260515",
        catalog_anchor="grand_reunion_symposium_phase_f_composite_6",
        source_archive_path=str(source_archive.relative_to(REPO_ROOT))
        if source_archive.is_relative_to(REPO_ROOT)
        else str(source_archive),
        source_archive_sha256=src_sha,
        source_archive_bytes=src_len,
        candidate_archive_path=str(cand_path.relative_to(REPO_ROOT))
        if cand_path.is_relative_to(REPO_ROOT)
        else str(cand_path),
        candidate_archive_sha256=cand_sha,
        candidate_archive_bytes=len(cand_bytes_disk),
        delta_bytes=delta,
        encoded_payload_bytes=len(enc_result.encoded_bytes),
        syndrome_bytes=enc_result.n_syndrome_bytes,
        overhead_bytes=enc_result.overhead_bytes,
        estimated_arithmetic_bits=enc_result.estimated_arithmetic_bits,
        constraint_length=enc_result.constraint_length,
        context_length=enc_result.context_length,
        sigma=sigma,
        sigma_int8=enc_result.sigma_int8,
        roundtrip_byte_stable=roundtrip_ok,
        # v1 carries the source residual in the envelope, so this concrete
        # byte-closed artifact is expected to be rate-negative. Keep the
        # speculative post-Viterbi band separate to avoid score-claim bleed.
        predicted_delta_s_band=None,
        predicted_delta_s_band_after_viterbi_inverse=(-0.010, -0.030),
        # CLAUDE.md "Apples-to-apples evidence discipline": no score claim
        # until contest-CUDA / contest-CPU anchor lands.
        score_claim=False,
        byte_anchor_only=True,
        score_claim_possible_after_result_review=True,
        contest_axis_anchor="byte_anchor_only",
        custody_status="byte_anchor_only_no_score_yet",
        cuda_eval_worth_testing=False,
        ready_for_exact_eval_dispatch=False,
        measured_config_status="byte_anchor_only_pre_score_eval",
        review_blockers=(
            "no_contest_cuda_anchor_yet",
            "no_contest_cpu_anchor_yet",
            "scaffold_only_per_catalog_220",
            "v1_scaffold_carries_residual_envelope_for_byte_stability",
            "full_viterbi_inverse_council_gated",
        ),
        built_at_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        scaffold_only=bool(SCAFFOLD_ONLY),
        notes=notes
        or (
            "STC-Dasher v1 scaffold candidate. This concrete v1 envelope "
            "carries the residual bytes, so it is not expected to lower rate. "
            "The speculative [-0.010, -0.030] rate-axis band applies only "
            "after the full Viterbi inverse / true entropy-coded payload "
            "lands. Scaffold-only per Catalog #220."
        ),
    )

    manifest_path = out_dir / "build_manifest.json"
    manifest_dict: dict[str, Any] = asdict(manifest)
    manifest_dict["rate_negative_scaffold"] = manifest.delta_bytes > 0
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, sort_keys=True))

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an STC-Dasher v1 substrate-agnostic archive bolt-on "
            "candidate (Grand Reunion symposium Composite #6)."
        )
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Path to the substrate archive ZIP (e.g. submissions/a1/archive.zip).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "Output directory under experiments/results/ namespace per "
            "CLAUDE.md /tmp-forbidden discipline."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help=(
            "Rate-distortion control. v1 scaffold: only sigma=0 produces "
            "a byte-stable roundtrip."
        ),
    )
    parser.add_argument(
        "--constraint-length",
        type=int,
        default=12,
        help="STC parity-check matrix constraint length (Filler 2011 default 12).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2,
        help="Dasher AC context model length (MacKay 2003 default 2).",
    )
    parser.add_argument(
        "--payload-bit-ratio",
        type=int,
        default=4,
        help=(
            "STC payload-to-cover bit ratio. v1 supports only 4 because the "
            "envelope does not serialize the ratio yet."
        ),
    )
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=65_536,
        help=(
            "Fail-fast cap (bytes) on source archives for v1 scaffold builds. "
            "Set 0 only for an intentional full-archive timing run."
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Operator-supplied notes for the manifest body.",
    )
    args = parser.parse_args(argv)

    # Per CLAUDE.md /tmp-forbidden + Catalog #207 namespace discipline:
    # refuse out-dir outside experiments/results/.
    out_dir_resolved = args.out_dir.resolve()
    namespace_root = (REPO_ROOT / "experiments" / "results").resolve()
    if not out_dir_resolved.is_relative_to(namespace_root):
        print(
            f"ERROR: --out-dir must be under {namespace_root.relative_to(REPO_ROOT)}/ "
            f"per CLAUDE.md /tmp-forbidden + Catalog #207 namespace discipline. "
            f"Got: {out_dir_resolved}",
            file=sys.stderr,
        )
        return 2
    if args.payload_bit_ratio != 4:
        print(
            "ERROR: --payload-bit-ratio is fixed at 4 in STC-Dasher v1 because "
            "the envelope does not serialize this field yet; bump the schema "
            "before exposing non-default ratios.",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = build_stc_dasher_candidate(
            source_archive=args.source_archive,
            out_dir=args.out_dir,
            sigma=args.sigma,
            constraint_length=args.constraint_length,
            context_length=args.context_length,
            payload_bit_ratio=args.payload_bit_ratio,
            notes=args.notes,
            max_source_bytes=args.max_source_bytes,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[stc-dasher-v1] candidate built: {manifest.candidate_archive_path}\n"
        f"  source bytes:    {manifest.source_archive_bytes}\n"
        f"  candidate bytes: {manifest.candidate_archive_bytes}\n"
        f"  delta bytes:     {manifest.delta_bytes:+d}\n"
        f"  syndrome bytes:  {manifest.syndrome_bytes}\n"
        f"  AC bits (est):   {manifest.estimated_arithmetic_bits:.1f}\n"
        f"  roundtrip:       {manifest.roundtrip_byte_stable}\n"
        "  predicted dS:    none for v1 scaffold  [byte_anchor_only=true]\n"
        "  future dS band:  "
        f"{manifest.predicted_delta_s_band_after_viterbi_inverse}  "
        "[requires_viterbi_inverse=true]\n"
        f"  scaffold_only:   {manifest.scaffold_only}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
