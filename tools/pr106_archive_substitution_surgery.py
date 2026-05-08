#!/usr/bin/env python3
"""PR106 archive-substitution surgery: replace decoder_packed_brotli or
latents_and_sidecar_brotli sections in a PR106-shaped archive.

PR106 (BradyMeighan ``hnerv_lc_v2``, public score 0.1954) is a single-member
ZIP holding ``0.bin`` whose payload follows the FF-packed HNeRV grammar:

    0.bin = 0xFF + decoder_len(3 bytes LE)
              + decoder_packed_brotli[: decoder_len]
              + latents_and_sidecar_brotli[decoder_len :]

Unlike PR101's fixed-offset slice grammar, PR106 stores ``decoder_len``
in the header. That means **the decoder section can change length** in a
substitution and the runtime still parses correctly — provided the same
4-byte header is regenerated. This is the substrate-adaptive door
``Op_KLLatent`` and the cathedral's lgwin-tuning surface walk through.

Sister tool to ``pr101_archive_substitution_surgery.py`` with three
behavioral differences:

  1. **Variable-length decoder** — substitute_decoder() permits any
     positive-length replacement (header re-written automatically).
  2. **Variable-length tail** — substitute_latents_and_sidecar() permits
     any positive-length replacement.
  3. **Codex dispatch-gate contract** — SubstitutionReport carries
     ``score_affecting_payload_changed``, ``charged_bits_changed``,
     ``target_modes``, ``deployment_target`` per the 2026-05-07 codex
     parallel-session contract (see
     ``feedback_codex_parallel_session_packet_compiler_20260507.md``).

CLAUDE.md compliance: pure-CPU + zipfile + brotli (the FF-grammar parser
already shipped in ``tac.hnerv_lowlevel_packer``); no scorer load; no
MPS/CUDA. Outputs to ``experiments/results/<lane>/`` per durable-state
convention.

Usage::

    .venv/bin/python tools/pr106_archive_substitution_surgery.py verify \\
        --archive experiments/.../public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip

    .venv/bin/python tools/pr106_archive_substitution_surgery.py substitute-decoder \\
        --input-archive .../archive.zip \\
        --replacement-decoder-blob /path/to/new_decoder.brotli \\
        --output-archive experiments/results/<lane>/substituted_archive.zip \\
        --report experiments/results/<lane>/substitution_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.hnerv_lowlevel_packer import (  # noqa: E402
    HnervLowlevelPackError,
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
)

# PR106 layout constants (verified against
# ``experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip``
# 2026-05-07: single-member ZIP, member name ``0.bin``, payload 186_131 B
# starting with 0xFF + 3-byte LE decoder length).
PR106_INNER_MEMBER_NAME = "0.bin"
PR106_HEADER_LEN = 4  # 0xFF magic + 3-byte LE decoder length
PR106_HEADER_MAGIC = 0xFF

DEFAULT_TARGET_MODES = ["contest_exact_eval"]
DEFAULT_DEPLOYMENT_TARGET = "t4_contest_runtime"
REPORT_SCHEMA_VERSION = "pr106_archive_substitution_surgery.v2"


@dataclass
class SubstitutionReport:
    """Custody record for a PR106 surgery operation.

    Codex dispatch-gate contract fields (target_modes, deployment_target,
    score_affecting_payload_changed, charged_bits_changed) make this report
    machine-checkable for the dispatch ranker without re-deriving from
    SHA fields.
    """

    input_archive: str
    output_archive: str
    input_size_bytes: int
    output_size_bytes: int
    bytes_delta: int

    # PR106-specific section accounting
    inner_member_name: str
    inner_member_input_size: int
    inner_member_output_size: int

    decoder_input_len: int
    decoder_replacement_len: int
    decoder_len_delta: int

    latents_sidecar_input_len: int
    latents_sidecar_output_len: int
    latents_sidecar_len_delta: int

    # Custody — every section gets input/output SHA so a downstream
    # auditor can prove which sections moved and which were preserved.
    sha256_input_archive: str
    sha256_output_archive: str
    sha256_input_inner_member: str
    sha256_output_inner_member: str

    sha256_input_decoder: str
    sha256_output_decoder: str
    sha256_input_latents_sidecar: str
    sha256_output_latents_sidecar: str

    schema_version: str = REPORT_SCHEMA_VERSION
    evidence_grade: str = "archive_construction_only"
    score_claim: bool = False
    ready_for_exact_eval_dispatch: bool = False
    dispatch_attempted: bool = False

    archive_container_changed: bool = False
    archive_byte_count_changed: bool = False
    inner_member_payload_changed: bool = False
    decoder_payload_changed: bool = False
    latents_sidecar_payload_changed: bool = False
    decoder_preserved: bool = False
    latents_sidecar_preserved: bool = False
    header_decoder_len_matches_output_decoder: bool = False

    # Codex dispatch-gate contract (2026-05-07)
    target_modes: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODES))
    deployment_target: str = DEFAULT_DEPLOYMENT_TARGET
    score_affecting_payload_changed: bool = False
    charged_bits_changed: bool = False
    charged_byte_count_changed: bool = False
    exact_eval_blockers: list[str] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_inner_blob(archive_path: Path) -> bytes:
    """Extract the single inner blob (member ``0.bin``) from a PR106 archive."""
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != [PR106_INNER_MEMBER_NAME]:
            raise ValueError(
                f"archive {archive_path} has members {names!r}; expected "
                f"['{PR106_INNER_MEMBER_NAME}'] (PR106 layout)"
            )
        with zf.open(PR106_INNER_MEMBER_NAME) as fp:
            return fp.read()


def _write_pr106_archive(inner_blob: bytes, archive_path: Path) -> None:
    """Write a PR106-shaped archive: ZIP with single ``0.bin`` member.

    Stored uncompressed (the inner blob is already entropy-coded with
    brotli on every section). Fixed timestamp (1980-01-01) for
    deterministic-bytes reproducibility — note that PR106's own archives
    don't strip ZIP timestamps, but the cathedral's discipline is
    determinism-first.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=PR106_INNER_MEMBER_NAME)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, inner_blob)


def _build_report(
    *,
    input_archive: Path,
    output_archive: Path,
    input_inner: bytes,
    output_inner: bytes,
    input_packed: PackedHnervPayload,
    output_packed: PackedHnervPayload,
    target_modes: list[str],
    deployment_target: str,
    notes: list[str],
) -> SubstitutionReport:
    decoder_changed = input_packed.decoder_packed_brotli != output_packed.decoder_packed_brotli
    tail_changed = (
        input_packed.latents_and_sidecar_brotli
        != output_packed.latents_and_sidecar_brotli
    )
    payload_changed = decoder_changed or tail_changed
    input_archive_sha = _sha256(input_archive.read_bytes())
    output_archive_sha = _sha256(output_archive.read_bytes())
    input_inner_sha = _sha256(input_inner)
    output_inner_sha = _sha256(output_inner)
    archive_byte_count_changed = input_archive.stat().st_size != output_archive.stat().st_size
    exact_eval_blockers = [
        "exact_runtime_parity_not_supplied",
        "matching_lane_dispatch_claim_not_supplied",
        "contest_cuda_auth_eval_not_run",
    ]
    if not payload_changed:
        exact_eval_blockers.insert(0, "score_affecting_payload_not_changed")
    return SubstitutionReport(
        input_archive=str(input_archive),
        output_archive=str(output_archive),
        input_size_bytes=input_archive.stat().st_size,
        output_size_bytes=output_archive.stat().st_size,
        bytes_delta=output_archive.stat().st_size - input_archive.stat().st_size,
        inner_member_name=PR106_INNER_MEMBER_NAME,
        inner_member_input_size=len(input_inner),
        inner_member_output_size=len(output_inner),
        decoder_input_len=len(input_packed.decoder_packed_brotli),
        decoder_replacement_len=len(output_packed.decoder_packed_brotli),
        decoder_len_delta=(
            len(output_packed.decoder_packed_brotli)
            - len(input_packed.decoder_packed_brotli)
        ),
        latents_sidecar_input_len=len(input_packed.latents_and_sidecar_brotli),
        latents_sidecar_output_len=len(output_packed.latents_and_sidecar_brotli),
        latents_sidecar_len_delta=(
            len(output_packed.latents_and_sidecar_brotli)
            - len(input_packed.latents_and_sidecar_brotli)
        ),
        sha256_input_archive=input_archive_sha,
        sha256_output_archive=output_archive_sha,
        sha256_input_inner_member=input_inner_sha,
        sha256_output_inner_member=output_inner_sha,
        sha256_input_decoder=_sha256(input_packed.decoder_packed_brotli),
        sha256_output_decoder=_sha256(output_packed.decoder_packed_brotli),
        sha256_input_latents_sidecar=_sha256(input_packed.latents_and_sidecar_brotli),
        sha256_output_latents_sidecar=_sha256(output_packed.latents_and_sidecar_brotli),
        archive_container_changed=input_archive_sha != output_archive_sha,
        archive_byte_count_changed=archive_byte_count_changed,
        inner_member_payload_changed=input_inner_sha != output_inner_sha,
        decoder_payload_changed=decoder_changed,
        latents_sidecar_payload_changed=tail_changed,
        decoder_preserved=not decoder_changed,
        latents_sidecar_preserved=not tail_changed,
        header_decoder_len_matches_output_decoder=(
            output_inner[:1] == bytes([PR106_HEADER_MAGIC])
            and int.from_bytes(output_inner[1:4], "little")
            == len(output_packed.decoder_packed_brotli)
        ),
        target_modes=list(target_modes),
        deployment_target=deployment_target,
        score_affecting_payload_changed=payload_changed,
        # Historical field retained for callers that used it as a payload-change
        # sentinel. Use archive_container_changed / charged_byte_count_changed
        # for container and rate accounting.
        charged_bits_changed=payload_changed,
        charged_byte_count_changed=archive_byte_count_changed,
        exact_eval_blockers=exact_eval_blockers,
        notes=notes,
    )


def substitute_decoder(
    *,
    input_archive: Path,
    replacement_decoder: bytes,
    output_archive: Path,
    target_modes: list[str] | None = None,
    deployment_target: str = DEFAULT_DEPLOYMENT_TARGET,
) -> SubstitutionReport:
    """Substitute the decoder_packed_brotli section in a PR106 archive.

    Unlike PR101 surgery, PR106's FF-grammar header carries the decoder
    length so the replacement can be any positive number of bytes. The
    runtime parses the header, slices [4 : 4+decoder_len] as decoder, and
    [4+decoder_len :] as tail — the substitution simply rewrites the
    header to match the replacement length.

    Defensive guards: replacement must be non-empty, must not exceed the
    24-bit header limit (16,777,215 bytes), and must be valid brotli
    (verified by attempting decompression — failures raise instead of
    producing a corrupted archive).
    """
    if target_modes is None:
        target_modes = list(DEFAULT_TARGET_MODES)
    if not replacement_decoder:
        raise ValueError("replacement_decoder is empty; PR106 requires a positive-length decoder")
    if len(replacement_decoder) > 0xFFFFFF:
        raise ValueError(
            f"replacement_decoder length {len(replacement_decoder)} exceeds "
            f"PR106 24-bit header limit (16,777,215 bytes)"
        )

    input_inner = _read_inner_blob(input_archive)
    input_packed = parse_ff_packed_brotli_hnerv(input_inner)

    # Validate the replacement is brotli-decodable. Catches the common
    # operator error of supplying a raw .pt or unwrapped tensor file.
    import brotli
    try:
        brotli.decompress(replacement_decoder)
    except brotli.error as exc:
        raise ValueError(
            f"replacement_decoder is not brotli-decompressible: {exc}"
        ) from exc

    output_packed = PackedHnervPayload(
        header=b"",  # ignored by to_bytes(); regenerated from decoder length
        decoder_packed_brotli=replacement_decoder,
        latents_and_sidecar_brotli=input_packed.latents_and_sidecar_brotli,
    )
    output_inner = output_packed.to_bytes()
    _write_pr106_archive(output_inner, output_archive)

    notes = [
        "decoder_packed_brotli substituted; latents_and_sidecar_brotli preserved",
        f"input decoder len={len(input_packed.decoder_packed_brotli)}; "
        f"replacement decoder len={len(replacement_decoder)}; "
        f"delta={len(replacement_decoder) - len(input_packed.decoder_packed_brotli):+}",
    ]
    # Reparse to populate the header for the report (header is regenerated
    # by to_bytes(); reparsing exposes the canonical bytes).
    output_packed_canonical = parse_ff_packed_brotli_hnerv(output_inner)
    return _build_report(
        input_archive=input_archive,
        output_archive=output_archive,
        input_inner=input_inner,
        output_inner=output_inner,
        input_packed=input_packed,
        output_packed=output_packed_canonical,
        target_modes=target_modes,
        deployment_target=deployment_target,
        notes=notes,
    )


def substitute_latents_and_sidecar(
    *,
    input_archive: Path,
    replacement_tail: bytes,
    output_archive: Path,
    target_modes: list[str] | None = None,
    deployment_target: str = DEFAULT_DEPLOYMENT_TARGET,
) -> SubstitutionReport:
    """Substitute the latents_and_sidecar_brotli section.

    The tail is variable-length by design (read as ``payload[4+decoder_len :]``).
    Replacement length is unconstrained beyond non-empty.

    PR106's runtime then brotli-decompresses the tail and parses it as the
    HNeRV latent stream + per-frame metadata. Replacements that don't
    satisfy that grammar will fail at parse-time inside PR106's inflate.
    The defensive guard here is "non-empty + brotli-decodable"; structural
    correctness is the caller's responsibility.
    """
    if target_modes is None:
        target_modes = list(DEFAULT_TARGET_MODES)
    if not replacement_tail:
        raise ValueError("replacement_tail is empty; PR106 requires positive-length latents+sidecar")

    import brotli
    try:
        brotli.decompress(replacement_tail)
    except brotli.error as exc:
        raise ValueError(
            f"replacement_tail is not brotli-decompressible: {exc}"
        ) from exc

    input_inner = _read_inner_blob(input_archive)
    input_packed = parse_ff_packed_brotli_hnerv(input_inner)

    output_packed = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=input_packed.decoder_packed_brotli,
        latents_and_sidecar_brotli=replacement_tail,
    )
    output_inner = output_packed.to_bytes()
    _write_pr106_archive(output_inner, output_archive)

    notes = [
        "latents_and_sidecar_brotli substituted; decoder_packed_brotli preserved",
        f"input tail len={len(input_packed.latents_and_sidecar_brotli)}; "
        f"replacement tail len={len(replacement_tail)}; "
        f"delta={len(replacement_tail) - len(input_packed.latents_and_sidecar_brotli):+}",
    ]
    output_packed_canonical = parse_ff_packed_brotli_hnerv(output_inner)
    return _build_report(
        input_archive=input_archive,
        output_archive=output_archive,
        input_inner=input_inner,
        output_inner=output_inner,
        input_packed=input_packed,
        output_packed=output_packed_canonical,
        target_modes=target_modes,
        deployment_target=deployment_target,
        notes=notes,
    )


def verify_byte_layout(archive_path: Path) -> dict[str, object]:
    """Diagnostic: load a PR106 archive and report its FF-grammar layout."""
    blob = _read_inner_blob(archive_path)
    try:
        packed = parse_ff_packed_brotli_hnerv(blob)
    except HnervLowlevelPackError as exc:
        return {
            "archive_path": str(archive_path),
            "error": f"not a valid PR106 FF-grammar payload: {exc}",
        }
    return {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "inner_blob_size": len(blob),
        "header": {
            "len": PR106_HEADER_LEN,
            "magic_byte": f"0x{blob[0]:02x}",
            "decoder_len_field": int.from_bytes(blob[1:4], "little"),
        },
        "decoder_packed_brotli": {
            "offset": PR106_HEADER_LEN,
            "len": len(packed.decoder_packed_brotli),
            "sha256": _sha256(packed.decoder_packed_brotli),
        },
        "latents_and_sidecar_brotli": {
            "offset": PR106_HEADER_LEN + len(packed.decoder_packed_brotli),
            "len": len(packed.latents_and_sidecar_brotli),
            "sha256": _sha256(packed.latents_and_sidecar_brotli),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Inspect PR106 archive layout")
    p_verify.add_argument("--archive", type=Path, required=True)

    p_subst_dec = sub.add_parser(
        "substitute-decoder",
        help="Replace decoder_packed_brotli with new bytes",
    )
    p_subst_dec.add_argument("--input-archive", type=Path, required=True)
    p_subst_dec.add_argument("--replacement-decoder-blob", type=Path, required=True,
                             help="Path to a .bin file with the new brotli-encoded decoder bytes")
    p_subst_dec.add_argument("--output-archive", type=Path, required=True)
    p_subst_dec.add_argument("--report", type=Path, default=None)
    p_subst_dec.add_argument("--target-modes", type=str, default="contest_exact_eval",
                             help="Comma-separated target modes (default: contest_exact_eval)")
    p_subst_dec.add_argument("--deployment-target", type=str,
                             default=DEFAULT_DEPLOYMENT_TARGET)

    p_subst_tail = sub.add_parser(
        "substitute-latents-sidecar",
        help="Replace latents_and_sidecar_brotli with new bytes",
    )
    p_subst_tail.add_argument("--input-archive", type=Path, required=True)
    p_subst_tail.add_argument("--replacement-tail-blob", type=Path, required=True,
                              help="Path to a .bin file with the new brotli-encoded latents+sidecar bytes")
    p_subst_tail.add_argument("--output-archive", type=Path, required=True)
    p_subst_tail.add_argument("--report", type=Path, default=None)
    p_subst_tail.add_argument("--target-modes", type=str, default="contest_exact_eval")
    p_subst_tail.add_argument("--deployment-target", type=str,
                              default=DEFAULT_DEPLOYMENT_TARGET)

    args = parser.parse_args(argv)

    if args.cmd == "verify":
        layout = verify_byte_layout(args.archive)
        print(json.dumps(layout, indent=2, sort_keys=True))
        return 0

    if args.cmd == "substitute-decoder":
        replacement_bytes = args.replacement_decoder_blob.read_bytes()
        target_modes = [m.strip() for m in args.target_modes.split(",") if m.strip()]
        report = substitute_decoder(
            input_archive=args.input_archive,
            replacement_decoder=replacement_bytes,
            output_archive=args.output_archive,
            target_modes=target_modes,
            deployment_target=args.deployment_target,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0

    if args.cmd == "substitute-latents-sidecar":
        replacement_bytes = args.replacement_tail_blob.read_bytes()
        target_modes = [m.strip() for m in args.target_modes.split(",") if m.strip()]
        report = substitute_latents_and_sidecar(
            input_archive=args.input_archive,
            replacement_tail=replacement_bytes,
            output_archive=args.output_archive,
            target_modes=target_modes,
            deployment_target=args.deployment_target,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
