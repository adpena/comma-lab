# SPDX-License-Identifier: MIT
"""Re-pack compact receiver archives through decoder-codec portfolios.

This is the post-training byte grammar actuator for compact carriers. It takes
an existing byte-closed archive, parses its charged receiver sections, mutates
only the decoder-state codec, rebuilds ``archive.zip``, and optionally runs the
generated receiver proof. It grants no score authority.
"""

from __future__ import annotations

import copy
import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.hi_nerv.archive import HIV1_MAGIC
from tac.substrates.hi_nerv.archive import (
    parse_archive as parse_hi_nerv_archive,
)
from tac.substrates.hi_nerv.archive import (
    repack_archive_decoder_codec as repack_hi_nerv_archive_decoder_codec,
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
COMPACT_DECODER_CODEC_REPLAY_ADJUDICATION_SCHEMA = (
    "compact_decoder_codec_replay_adjudication.v1"
)
SUPPORTED_COMPACT_DECODER_CODECS: tuple[str, ...] = (
    "portfolio_auto",
    "int8_mixed",
    "int8_scale_bundled",
    "int7_mixed",
    "int7_scale_bundled",
    "int6_mixed",
    "int6_scale_bundled",
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
        family: ``auto`` / ``pact_nerv_vq`` / ``pact_nerv_selector_v4`` /
            ``hi_nerv``.
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
    section_value_rows = _decoder_codec_section_value_rows(
        rows_sorted,
        source_archive_bytes=int(source.stat().st_size),
    )
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": f"{COMPACT_DECODER_CODEC_SWEEP_SCHEMA}.section_value_rows",
            "candidate_id": f"{resolved_family}_compact_decoder_codec_sweep",
            "family": resolved_family,
            "axis_tag": "[planning/control]",
            "section_value_rows": section_value_rows,
        }
    )
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
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "blockers": _dedupe(
            [
                *[
                    blocker
                    for row in rows_sorted
                    for blocker in list(row.get("blockers") or [])
                ],
                *list(byte_price_plan.get("blockers") or []),
            ]
        ),
        **FALSE_AUTHORITY,
    }
    report_path = out / "compact_decoder_codec_sweep_report.json"
    write_json(report_path, report)
    report["report_path"] = report_path.as_posix()
    write_json(report_path, report)
    return report


def adjudicate_compact_decoder_codec_sweep_with_replay(
    *,
    codec_sweep_report: Mapping[str, Any],
    source_replay_profile: Mapping[str, Any],
    best_codec_replay_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach full-video replay adjudication to an existing codec sweep report.

    This does not grant contest authority. It only supersedes the stale
    "replay not attached" blocker when a matching local replay gate exists and
    preserves byte-saving primitives for later full-stack composition.
    """

    family = str(codec_sweep_report.get("family") or "").strip().lower()
    if family == "pact_nerv_vq":
        from tac.substrates.pact_nerv_vq.competitiveness_gate import (
            build_pact_vq_competitiveness_gate,
        )

        gate = build_pact_vq_competitiveness_gate(
            codec_sweep_report=codec_sweep_report,
            source_replay_profile=source_replay_profile,
            best_codec_replay_profile=best_codec_replay_profile,
        )
    else:
        raise CompactDecoderCodecSweepError(
            f"replay adjudication unsupported for family {family!r}"
        )

    report = copy.deepcopy(dict(codec_sweep_report))
    best_variant = dict(report.get("best_variant") or {})
    pre_blockers = list(report.get("blockers") or [])
    pre_best_blockers = list(best_variant.get("blockers") or [])
    superseded = [
        "full_video_mlx_scorer_replay_not_attached",
        "codec_sweep_report_needs_replay_attachment",
    ]
    post_blockers = _dedupe(
        [
            *[
                blocker
                for blocker in pre_blockers
                if blocker not in superseded
            ],
            *[
                blocker
                for blocker in gate.get("blockers") or []
                if blocker not in superseded
            ],
        ]
    )
    post_best_blockers = _dedupe(
        [
            blocker
            for blocker in pre_best_blockers
            if blocker not in superseded
        ]
    )

    best_variant.update(
        {
            "blockers": post_best_blockers,
            "full_video_mlx_replay_attached": True,
            "replay_gate_schema": gate.get("schema"),
            "replay_gate_verdict": gate.get("verdict"),
            "preserve_rate_primitive": bool(gate.get("preserve_rate_primitive")),
            "exact_axis_blocked": bool(gate.get("exact_axis_blocked")),
            "exact_spend_candidate": bool(gate.get("exact_spend_candidate")),
            "demote_for_full_stack_portfolio": bool(
                gate.get("demote_for_full_stack_portfolio")
            ),
        }
    )
    report.update(
        {
            "schema": codec_sweep_report.get(
                "schema",
                COMPACT_DECODER_CODEC_SWEEP_SCHEMA,
            ),
            "replay_adjudication": {
                "schema": COMPACT_DECODER_CODEC_REPLAY_ADJUDICATION_SCHEMA,
                "family": family,
                "gate_schema": gate.get("schema"),
                "gate_verdict": gate.get("verdict"),
                "full_video_mlx_replay_attached": True,
                "superseded_blockers": superseded,
                "pre_adjudication_blockers": pre_blockers,
                "pre_adjudication_best_variant_blockers": pre_best_blockers,
                "preserve_rate_primitive": bool(gate.get("preserve_rate_primitive")),
                "exact_axis_blocked": bool(gate.get("exact_axis_blocked")),
                "exact_spend_candidate": bool(gate.get("exact_spend_candidate")),
                "demote_for_full_stack_portfolio": bool(
                    gate.get("demote_for_full_stack_portfolio")
                ),
                "deltas": gate.get("deltas"),
                "rate_axis": gate.get("rate_axis"),
                "recommended_next_actions": gate.get("recommended_next_actions"),
                **FALSE_AUTHORITY,
            },
            "best_variant": best_variant,
            "blockers": post_blockers,
            "full_video_mlx_replay_attached": True,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    return report


def adjudicate_compact_decoder_codec_sweep_with_replay_from_paths(
    *,
    codec_sweep_report_path: str | Path,
    source_replay_profile_path: str | Path,
    best_codec_replay_profile_path: str | Path,
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load replay-adjudication inputs from JSON paths and optionally write."""

    report = adjudicate_compact_decoder_codec_sweep_with_replay(
        codec_sweep_report=_load_json_mapping(codec_sweep_report_path),
        source_replay_profile=_load_json_mapping(source_replay_profile_path),
        best_codec_replay_profile=_load_json_mapping(best_codec_replay_profile_path),
    )
    if output_report_path is not None:
        out = Path(output_report_path).expanduser().resolve(strict=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = out.as_posix()
        write_json(out, report)
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


def _decoder_codec_section_value_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_archive_bytes: int,
) -> list[dict[str, Any]]:
    section_rows: list[dict[str, Any]] = []
    for row in rows:
        decoder_codec = str(row.get("decoder_codec") or "")
        byte_delta = int(row.get("archive_bytes") or 0) - int(source_archive_bytes)
        section_rows.append(
            {
                "row_id": f"{row.get('family')}_decoder_codec:{decoder_codec}",
                "section_id": f"decoder_codec:{decoder_codec}",
                "row_kind": (
                    "new_residual_or_sidecar"
                    if byte_delta > 0
                    else "existing_section_cut"
                ),
                "family": row.get("family"),
                "scope": "compact_decoder_codec_replacement",
                "byte_delta": byte_delta,
                "section_bytes": abs(byte_delta)
                or int(row.get("archive_bytes") or 0),
                "delta_nonrate_score": None,
                "axis_tag": "[planning/control]",
                "receiver_proof_status": (
                    "satisfied" if row.get("receiver_proof_passed") is True else "missing"
                ),
                "full_video_coverage": int(row.get("num_pairs") or 0) >= 600,
                "archive_sha256": row.get("archive_sha256"),
                "source_archive_sha256": row.get("source_archive_sha256"),
                "source_archive_bytes": int(source_archive_bytes),
                "candidate_archive_bytes": int(row.get("archive_bytes") or 0),
                "decoder_codec": decoder_codec,
                "parsed_decoder_codec": dict(row.get("parsed_decoder_codec") or {}),
                "blockers": list(row.get("blockers") or []),
                **FALSE_AUTHORITY,
            }
        )
    return section_rows


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
    if family == "hi_nerv":
        parsed = parse_hi_nerv_archive(source_bin)
        bin_bytes = repack_hi_nerv_archive_decoder_codec(
            source_bin,
            decoder_codec=decoder_codec,
        )
        reparsed = parse_hi_nerv_archive(bin_bytes)
        return (
            bin_bytes,
            int(parsed.latents_coarse.shape[0]),
            _decoder_codec_meta(reparsed.meta),
        )
    raise CompactDecoderCodecSweepError(f"unsupported family: {family!r}")


def _decoder_codec_meta(meta: dict[str, Any]) -> dict[str, Any]:
    value = meta.get("_decoder_state_codec")
    return dict(value) if isinstance(value, dict) else {}


def _resolve_family(source_bin: bytes, *, family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in {"auto", "pact_nerv_vq", "pact_nerv_selector_v4", "hi_nerv"}:
        raise CompactDecoderCodecSweepError(f"unsupported family: {family!r}")
    if normalized != "auto":
        return normalized
    magic = source_bin[:4]
    if magic == PVQ_MAGIC:
        return "pact_nerv_vq"
    if magic == PSV4_MAGIC:
        return "pact_nerv_selector_v4"
    if magic == HIV1_MAGIC:
        return "hi_nerv"
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
    if family in {"pact_nerv_vq", "pact_nerv_selector_v4", "hi_nerv"}:
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


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve(strict=False)
    if not p.is_file():
        raise CompactDecoderCodecSweepError(f"JSON input missing: {p}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CompactDecoderCodecSweepError(f"JSON input is not object: {p}")
    return payload


__all__ = [
    "COMPACT_DECODER_CODEC_REPLAY_ADJUDICATION_SCHEMA",
    "COMPACT_DECODER_CODEC_SWEEP_SCHEMA",
    "SUPPORTED_COMPACT_DECODER_CODECS",
    "CompactDecoderCodecSweepError",
    "adjudicate_compact_decoder_codec_sweep_with_replay",
    "adjudicate_compact_decoder_codec_sweep_with_replay_from_paths",
    "sweep_compact_decoder_codecs",
]
