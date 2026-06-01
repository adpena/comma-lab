# SPDX-License-Identifier: MIT
"""Re-pack compact receiver archives through decoder-codec portfolios.

This is the post-training byte grammar actuator for compact carriers. It takes
an existing byte-closed archive, parses its charged receiver sections, mutates
only the decoder-state codec, rebuilds ``archive.zip``, and optionally runs the
generated receiver proof. It grants no score authority.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    PSV4_MAGIC,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    pack_archive as pack_selector_v4_archive,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    parse_archive as parse_selector_v4_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    PVQ_MAGIC,
)
from tac.substrates.pact_nerv_vq.archive import (
    pack_archive as pack_vq_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    parse_archive as parse_vq_archive,
)

COMPACT_DECODER_CODEC_SWEEP_SCHEMA = "compact_decoder_codec_sweep.v1"
COMPACT_DECODER_CODEC_VARIANT_SCHEMA = "compact_decoder_codec_sweep_variant.v1"
SUPPORTED_COMPACT_DECODER_CODECS: tuple[str, ...] = (
    "portfolio_auto",
    "int8_mixed",
    "int8_scale_bundled",
    "int4_mixed",
    "int4_scale_bundled",
    "int2_mixed",
    "int2_scale_bundled",
    "fp16_enveloped",
)


class CompactDecoderCodecSweepError(RuntimeError):
    """Raised for fail-closed compact codec sweep errors."""


def sweep_compact_decoder_codecs(
    *,
    source_archive_zip: str | Path,
    output_dir: str | Path,
    decoder_codecs: Sequence[str] = SUPPORTED_COMPACT_DECODER_CODECS,
    family: str = "auto",
    repo_root: str | Path = ".",
    run_receiver_proof: bool = True,
    retain_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Sweep charged decoder codecs for one compact archive.

    Args:
        source_archive_zip: existing contest-shaped ``archive.zip`` containing
            a compact ``0.bin`` payload.
        output_dir: durable SSD/local artifact root for variant archives.
        decoder_codecs: candidate decoder-state codecs to materialize.
        family: ``auto`` / ``pact_nerv_vq`` / ``pact_nerv_selector_v4``.
        repo_root: repository root used for runtime vendoring and relative
            proof paths.
        run_receiver_proof: when true, run each generated ``inflate.sh`` and
            emit the shared archive-bound candidate package.
        retain_receiver_proof_output: preserve inflated raw bytes instead of
            proof-and-delete. Defaults false for disk hygiene.
        receiver_proof_timeout_seconds: per-variant inflate proof timeout.
        allow_overwrite: allow a non-empty output directory.

    Returns:
        machine-readable false-authority sweep report.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    source = Path(source_archive_zip).expanduser().resolve(strict=False)
    if not source.is_file():
        raise CompactDecoderCodecSweepError(f"source archive missing: {source}")
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise CompactDecoderCodecSweepError(
            f"output dir is non-empty; pass allow_overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    source_bin = _read_zip_member(source, "0.bin")
    resolved_family = _resolve_family(source_bin, family=family)
    source_sha = sha256_file(source)
    rows = []
    for codec in _validated_codecs(decoder_codecs):
        rows.append(
            _materialize_variant(
                source_archive_zip=source,
                source_archive_sha256=source_sha,
                source_bin=source_bin,
                family=resolved_family,
                decoder_codec=codec,
                output_dir=out / _codec_dir_name(codec),
                repo_root=root,
                run_receiver_proof=run_receiver_proof,
                retain_receiver_proof_output=retain_receiver_proof_output,
                receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
            )
        )
    rows_sorted = sorted(rows, key=lambda row: int(row["archive_bytes"]))
    report = {
        "schema": COMPACT_DECODER_CODEC_SWEEP_SCHEMA,
        "source_archive_zip": source.as_posix(),
        "source_archive_sha256": source_sha,
        "source_archive_bytes": int(source.stat().st_size),
        "family": resolved_family,
        "decoder_codecs": list(_validated_codecs(decoder_codecs)),
        "run_receiver_proof": bool(run_receiver_proof),
        "variant_rows": rows_sorted,
        "best_variant": rows_sorted[0] if rows_sorted else None,
        "blockers": _dedupe(
            blocker
            for row in rows_sorted
            for blocker in list(row.get("blockers") or [])
        ),
        **FALSE_AUTHORITY,
    }
    report_path = out / "compact_decoder_codec_sweep_report.json"
    write_json(report_path, report)
    report["report_path"] = report_path.as_posix()
    write_json(report_path, report)
    return report


def _materialize_variant(
    *,
    source_archive_zip: Path,
    source_archive_sha256: str,
    source_bin: bytes,
    family: str,
    decoder_codec: str,
    output_dir: Path,
    repo_root: Path,
    run_receiver_proof: bool,
    retain_receiver_proof_output: bool,
    receiver_proof_timeout_seconds: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_bytes, num_pairs, decoder_codec_meta = _repack_bin(
        source_bin,
        family=family,
        decoder_codec=decoder_codec,
    )
    bin_path = output_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)
    submission_dir = output_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name=_substrate_pkg_name(family),
        repo_root=repo_root,
        vendor_shared_inflate_runtime=True,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = output_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    archive_sha = sha256_file(archive_zip_path)
    archive_bytes = int(archive_zip_path.stat().st_size)
    blockers: list[str] = [
        "full_video_mlx_scorer_replay_not_attached",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    package_path = None
    receiver_proof_path = None
    receiver_proof_passed = False
    if run_receiver_proof:
        package = emit_archive_bound_candidate_runtime_package(
            adapter_id=f"{family}_decoder_codec_sweep",
            candidate_family=f"{family}_decoder_codec_sweep",
            candidate_id_prefix=f"{family}_{_codec_dir_name(decoder_codec)}",
            transform_kind="compact_decoder_codec_repack",
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=output_dir,
            repo_root=repo_root,
            receiver_contract_kind=f"{family}_decode_only_receiver",
            proof_filename=f"{family}_{_codec_dir_name(decoder_codec)}_receiver_proof.json",
            candidate_label=f"{family}_{_codec_dir_name(decoder_codec)}",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes(num_pairs),
            retain_receiver_output=retain_receiver_proof_output,
            timeout_seconds=int(receiver_proof_timeout_seconds),
            runtime_adapter_manifest_extra={
                "schema": "compact_decoder_codec_sweep_runtime_adapter.v1",
                "source_archive_zip": source_archive_zip.as_posix(),
                "source_archive_sha256": source_archive_sha256,
                "decoder_codec": decoder_codec,
                "parsed_decoder_codec": decoder_codec_meta,
                "num_pairs": int(num_pairs),
            },
            candidate_row_schema="compact_decoder_codec_sweep_candidate_row.v1",
            wrapper_schema="compact_decoder_codec_sweep_adapter_package.v1",
            input_artifacts=[source_archive_zip.as_posix(), bin_path.as_posix()],
            extra_blockers=blockers,
            mlx_triage_argv=[
                "tools/sweep_compact_decoder_codecs.py",
                "--source-archive-zip",
                source_archive_zip.as_posix(),
                "--decoder-codec",
                decoder_codec,
            ],
        )
        package_path = (output_dir / "archive_bound_candidate_adapter_package.json").as_posix()
        receiver_proof = dict(package.get("receiver_proof") or {})
        receiver_proof_path = receiver_proof.get("proof_path")
        receiver_proof_passed = (
            receiver_proof.get("runtime_consumption_proof_passed") is True
        )
        blockers = _dedupe(
            [
                *blockers,
                *list(receiver_proof.get("blockers") or []),
            ]
        )
    else:
        blockers.append("receiver_proof_not_run")

    row = {
        **FALSE_AUTHORITY,
        "schema": COMPACT_DECODER_CODEC_VARIANT_SCHEMA,
        "family": family,
        "decoder_codec": decoder_codec,
        "parsed_decoder_codec": decoder_codec_meta,
        "source_archive_zip": source_archive_zip.as_posix(),
        "source_archive_sha256": source_archive_sha256,
        "archive_path": archive_zip_path.as_posix(),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "bin_path": bin_path.as_posix(),
        "bin_bytes": len(bin_bytes),
        "num_pairs": int(num_pairs),
        "receiver_proof_path": receiver_proof_path,
        "receiver_proof_passed": bool(receiver_proof_passed),
        "archive_bound_candidate_package_path": package_path,
        "charged_bits_changed": True,
        "score_affecting_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "blockers": blockers,
    }
    write_json(output_dir / "compact_decoder_codec_variant.json", row)
    return row


def _repack_bin(
    source_bin: bytes,
    *,
    family: str,
    decoder_codec: str,
) -> tuple[bytes, int, dict[str, Any]]:
    if family == "pact_nerv_vq":
        parsed = parse_vq_archive(source_bin)
        bin_bytes = pack_vq_archive(
            parsed.decoder_state_dict,
            parsed.codebook,
            parsed.indices,
            parsed.meta,
            schema_version=parsed.schema_version,
            decoder_codec=decoder_codec,
            indices_codec=parsed.indices_codec,
        )
        reparsed = parse_vq_archive(bin_bytes)
        return (
            bin_bytes,
            int(parsed.indices.shape[0]),
            _decoder_codec_meta(reparsed.meta),
        )
    if family == "pact_nerv_selector_v4":
        parsed = parse_selector_v4_archive(source_bin)
        bin_bytes = pack_selector_v4_archive(
            parsed.decoder_state_dict,
            parsed.latents,
            parsed.selector_bytes,
            parsed.meta,
            palette_size=parsed.palette_size,
            schema_version=parsed.schema_version,
            decoder_codec=decoder_codec,
        )
        reparsed = parse_selector_v4_archive(bin_bytes)
        return (
            bin_bytes,
            int(parsed.latents.shape[0]),
            _decoder_codec_meta(reparsed.meta),
        )
    raise CompactDecoderCodecSweepError(f"unsupported family: {family!r}")


def _decoder_codec_meta(meta: dict[str, Any]) -> dict[str, Any]:
    value = meta.get("_decoder_state_codec")
    return dict(value) if isinstance(value, dict) else {}


def _resolve_family(source_bin: bytes, *, family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in {"auto", "pact_nerv_vq", "pact_nerv_selector_v4"}:
        raise CompactDecoderCodecSweepError(f"unsupported family: {family!r}")
    if normalized != "auto":
        return normalized
    magic = source_bin[:4]
    if magic == PVQ_MAGIC:
        return "pact_nerv_vq"
    if magic == PSV4_MAGIC:
        return "pact_nerv_selector_v4"
    raise CompactDecoderCodecSweepError(f"cannot auto-detect family from magic {magic!r}")


def _read_zip_member(archive_zip: Path, member: str) -> bytes:
    with zipfile.ZipFile(archive_zip, "r") as zf:
        try:
            return zf.read(member)
        except KeyError as exc:
            raise CompactDecoderCodecSweepError(
                f"archive {archive_zip} missing member {member!r}"
            ) from exc


def _validated_codecs(values: Iterable[str]) -> tuple[str, ...]:
    rows = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not rows:
        raise CompactDecoderCodecSweepError("at least one decoder codec is required")
    unsupported = [value for value in rows if value not in SUPPORTED_COMPACT_DECODER_CODECS]
    if unsupported:
        raise CompactDecoderCodecSweepError(
            f"unsupported decoder codec(s): {unsupported}"
        )
    return tuple(dict.fromkeys(rows))


def _codec_dir_name(codec: str) -> str:
    return str(codec).replace("/", "_").replace(" ", "_")


def _substrate_pkg_name(family: str) -> str:
    if family in {"pact_nerv_vq", "pact_nerv_selector_v4"}:
        return family
    raise CompactDecoderCodecSweepError(f"unsupported family: {family!r}")


def _expected_receiver_output_bytes(num_pairs: int) -> int:
    return int(num_pairs) * 2 * int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3


def _dedupe(values: Iterable[Any]) -> list[Any]:
    rows: list[Any] = []
    for value in values:
        if value not in rows:
            rows.append(value)
    return rows


__all__ = [
    "COMPACT_DECODER_CODEC_SWEEP_SCHEMA",
    "SUPPORTED_COMPACT_DECODER_CODECS",
    "CompactDecoderCodecSweepError",
    "sweep_compact_decoder_codecs",
]
