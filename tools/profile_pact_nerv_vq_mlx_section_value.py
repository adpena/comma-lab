#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Price PVQ PACT-NeRV-VQ sections with full-video MLX scorer replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import write_json  # noqa: E402
from tac.submission_archive import write_deterministic_zip_member  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.pact_nerv_vq.section_value import (  # noqa: E402
    PVQ_SECTION_VALUE_PROFILE_SCHEMA,
    PVQ_SECTION_VALUE_SOURCE_SCHEMA,
    PVQ_SUPPORTED_SECTION_NAMES,
    neutralize_pvq_section,
    pvq_layout_report,
)
from tools.profile_pact_nerv_selector_v3_mlx_section_value import (  # noqa: E402
    DEFAULT_REFERENCE_CACHE,
    VariantSpec,
    _absent_section_row,
    _build_report,
    _extract_submission,
    _materialize_caches,
    _prepare_owned_dir,
    _read_archive_member,
    _resolve,
    _resolve_reference_cache_dir,
    _run_mlx_responses,
    _sha256_file,
)

OWNED_MARKER = ".pact_nerv_vq_mlx_section_value_owned.json"
DEFAULT_SECTIONS = ("decoder_qw", "codebooks_q", "selectors_rc", "residual_rc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--projection-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO_ROOT / "upstream",
        help=(
            "Pinned contest upstream snapshot consumed by inflate/cache "
            "materialization. Keep separate from --repo-root when running from "
            "an SSD code worktree."
        ),
    )
    parser.add_argument(
        "--video-names-file",
        type=Path,
        help=(
            "Optional file_list override for cache materialization; defaults "
            "to <upstream-dir>/public_test_video_names.txt."
        ),
    )
    parser.add_argument("--reference-cache-dir", type=Path, default=DEFAULT_REFERENCE_CACHE)
    parser.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--window-pairs", type=int, default=25)
    parser.add_argument("--scorer-batch-pairs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--allow-large-tensor-cache", action="store_true")
    parser.add_argument("--allow-batch-shape-research-signal", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preserve-existing-cache-reports",
        action="store_true",
        help=(
            "With --force on an owned output dir, keep mlx_caches, mlx_work, "
            "and mlx_cache_reports so archive-matched cache rows can be reused."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    repo_root = _resolve(args.repo_root, base=REPO_ROOT)
    archive = _resolve(args.archive, base=repo_root)
    output_dir = _resolve(args.output_dir, base=repo_root)
    upstream_dir = _resolve(args.upstream_dir, base=repo_root)
    video_names_file = (
        None
        if args.video_names_file is None
        else _resolve(args.video_names_file, base=repo_root)
    )
    reference_cache_dir = _resolve(args.reference_cache_dir, base=repo_root)
    resolved_reference_cache_dir = _resolve_reference_cache_dir(reference_cache_dir)
    projection_manifest = (
        None
        if args.projection_manifest is None
        else _resolve(args.projection_manifest, base=repo_root)
    )
    if args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.window_pairs < 1:
        raise SystemExit("--window-pairs must be >= 1")
    if args.scorer_batch_pairs < 1:
        raise SystemExit("--scorer-batch-pairs must be >= 1")
    if args.scorer_batch_pairs != 1 and not args.allow_batch_shape_research_signal:
        raise SystemExit(
            "--scorer-batch-pairs > 1 requires --allow-batch-shape-research-signal"
        )
    _prepare_owned_dir(
        output_dir,
        force=bool(args.force),
        preserve_cache_artifacts=bool(args.preserve_existing_cache_reports),
        owned_marker=OWNED_MARKER,
        tool_name=Path(__file__).name,
    )

    started = time.time()
    baseline_blob = _read_archive_member(archive, "0.bin")
    layout = pvq_layout_report(blob=baseline_blob)
    variants, absent_sections = _materialize_variants(
        archive=archive,
        baseline_blob=baseline_blob,
        output_dir=output_dir,
        requested_sections=[str(item).strip().lower() for item in args.sections],
    )
    cache_rows = _materialize_caches(
        variants=variants,
        output_dir=output_dir,
        repo_root=repo_root,
        upstream_dir=upstream_dir,
        video_names_file=video_names_file,
        max_pairs=int(args.max_pairs),
        inflate_timeout=int(args.inflate_timeout),
        allow_large_tensor_cache=bool(args.allow_large_tensor_cache),
    )
    payloads = _run_mlx_responses(
        variants=variants,
        output_dir=output_dir,
        repo_root=repo_root,
        upstream_dir=upstream_dir,
        reference_cache_dir=resolved_reference_cache_dir,
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        device=str(args.device),
        progress_every=int(args.progress_every),
        allow_batch_shape_research_signal=bool(args.allow_batch_shape_research_signal),
        response_family_prefix="pact_nerv_vq_section_value",
    )
    report = _build_report(
        raw_argv=raw_argv,
        variants=variants,
        absent_sections=absent_sections,
        cache_rows=cache_rows,
        payloads=payloads,
        layout=layout,
        output_dir=output_dir,
        repo_root=repo_root,
        upstream_dir=upstream_dir,
        video_names_file=video_names_file,
        archive=archive,
        projection_manifest=projection_manifest,
        reference_cache_dir=reference_cache_dir,
        resolved_reference_cache_dir=resolved_reference_cache_dir,
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        started=started,
        profile_schema=PVQ_SECTION_VALUE_PROFILE_SCHEMA,
        source_schema=PVQ_SECTION_VALUE_SOURCE_SCHEMA,
        layout_key="pvq_section_layout",
        residual_policy_schema="pact_nerv_vq_residual_admission_policy.v1",
        tool_path=Path(__file__),
    )
    report["family"] = "pact_nerv_vq"
    for row in report.get("section_value_rows", []):
        if isinstance(row, dict):
            row["family"] = "pact_nerv_vq"
    report_path = output_dir / "pact_nerv_vq_mlx_section_value_profile.json"
    write_json(report_path, report)
    compat_path = output_dir / "hprc_mlx_component_neutralization_profile.json"
    write_json(compat_path, report)
    print(
        json.dumps(
            {
                "schema": PVQ_SECTION_VALUE_PROFILE_SCHEMA,
                "report": report_path.as_posix(),
                "compat_report": compat_path.as_posix(),
                "variant_count": len(variants),
                "absent_sections": absent_sections,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _materialize_variants(
    *,
    archive: Path,
    baseline_blob: bytes,
    output_dir: Path,
    requested_sections: list[str],
) -> tuple[list[VariantSpec], list[dict[str, Any]]]:
    variants: list[VariantSpec] = []
    absent: list[dict[str, Any]] = []
    baseline_dir = output_dir / "variants" / "baseline"
    baseline_archive = baseline_dir / "archive.zip"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_repack = _write_zip_replacing_member(
        source_archive=archive,
        output_archive=baseline_archive,
        member_name="0.bin",
        replacement_bytes=baseline_blob,
        allow_overwrite=True,
    )
    write_json(baseline_dir / "zip_replacement_report.json", baseline_repack)
    _extract_submission(baseline_archive, baseline_dir / "submission")
    variants.append(
        _variant_row(
            variant_id="baseline",
            neutralized_section=None,
            archive_zip_path=baseline_archive,
            submission_dir=baseline_dir / "submission",
            bin_path=baseline_dir / "submission" / "0.bin",
            variant_dir=baseline_dir,
        )
    )
    for section in requested_sections:
        if section == "residual_rc":
            absent.append(_absent_section_row(section))
            continue
        if section not in PVQ_SUPPORTED_SECTION_NAMES:
            raise SystemExit(
                f"unsupported section {section!r}; "
                f"valid={(*PVQ_SUPPORTED_SECTION_NAMES, 'residual_rc')}"
            )
        if section == "receiver_state":
            absent.append(
                {
                    "section": section,
                    "status": (
                        "blocked_receiver_state_neutralization_would_"
                        "invalidate_decoder_shape"
                    ),
                    **FALSE_AUTHORITY,
                }
            )
            continue
        variant_id = f"neutralize_{section}"
        variant_dir = output_dir / "variants" / variant_id
        variant_archive = variant_dir / "archive.zip"
        variant_dir.mkdir(parents=True, exist_ok=True)
        neutralized = neutralize_pvq_section(baseline_blob, section)
        replacement = _write_zip_replacing_member(
            source_archive=archive,
            output_archive=variant_archive,
            member_name="0.bin",
            replacement_bytes=neutralized,
            allow_overwrite=True,
        )
        write_json(variant_dir / "zip_replacement_report.json", replacement)
        _extract_submission(variant_archive, variant_dir / "submission")
        variants.append(
            _variant_row(
                variant_id=variant_id,
                neutralized_section=section,
                archive_zip_path=variant_archive,
                submission_dir=variant_dir / "submission",
                bin_path=variant_dir / "submission" / "0.bin",
                variant_dir=variant_dir,
            )
        )
    return variants, absent


def _variant_row(
    *,
    variant_id: str,
    neutralized_section: str | None,
    archive_zip_path: Path,
    submission_dir: Path,
    bin_path: Path,
    variant_dir: Path,
) -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id,
        neutralized_section=neutralized_section,
        archive_zip_path=archive_zip_path,
        submission_dir=submission_dir,
        bin_path=bin_path,
        archive_bytes=archive_zip_path.stat().st_size,
        archive_sha256=_sha256_file(archive_zip_path),
        bin_sha256=_sha256_file(bin_path),
        variant_dir=variant_dir,
    )


def _write_zip_replacing_member(
    *,
    source_archive: Path,
    output_archive: Path,
    member_name: str,
    replacement_bytes: bytes,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_archive).expanduser().resolve(strict=False)
    output = Path(output_archive).expanduser().resolve(strict=False)
    if output.exists() and not allow_overwrite:
        raise FileExistsError(f"output archive exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    replaced: dict[str, Any] | None = None
    with zipfile.ZipFile(source, "r") as zin:
        names = {info.filename for info in zin.infolist() if not info.is_dir()}
        if member_name not in names:
            raise ValueError(f"ZIP member missing: {member_name}")
        with zipfile.ZipFile(output, "w") as zout:
            for info in sorted(zin.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                old = zin.read(info.filename)
                if info.filename == member_name:
                    write_deterministic_zip_member(
                        zout,
                        info.filename,
                        replacement_bytes,
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )
                    replaced = {
                        "member": info.filename,
                        "old_bytes": len(old),
                        "old_sha256": hashlib.sha256(old).hexdigest(),
                        "bytes": len(replacement_bytes),
                        "sha256": hashlib.sha256(replacement_bytes).hexdigest(),
                    }
                else:
                    write_deterministic_zip_member(
                        zout,
                        info.filename,
                        old,
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )
                    copied.append(
                        {
                            "member": info.filename,
                            "bytes": len(old),
                            "sha256": hashlib.sha256(old).hexdigest(),
                        }
                    )
    if replaced is None:
        raise ValueError(f"ZIP member was not replaced: {member_name}")
    return {
        "schema": "pact_nerv_vq_zip_member_replacement.v1",
        "source_archive": _file_row(source),
        "output_archive": _file_row(output),
        "replaced_member": replaced,
        "copied_members": copied,
        **FALSE_AUTHORITY,
    }


def _file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"profile_pact_nerv_vq_mlx_section_value failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
