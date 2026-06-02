#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Price PR95/HNeRV sections with full-video MLX scorer replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np  # noqa: E402

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    parse_pr95_public_archive_zip,
    write_pr95_public_archive_zip,
)
from tac.repo_io import write_json  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tools.profile_pact_nerv_selector_v3_mlx_section_value import (  # noqa: E402
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_UPSTREAM_DIR,
    VariantSpec,
    _absent_section_row,
    _build_report,
    _materialize_caches,
    _prepare_owned_dir,
    _resolve,
    _resolve_reference_cache_dir,
    _run_mlx_responses,
    _sha256_file,
)

PR95_HNERV_SECTION_VALUE_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
PR95_HNERV_SECTION_VALUE_SOURCE_SCHEMA = "pr95_hnerv_section_value_profile_source.v1"
OWNED_MARKER = ".pr95_hnerv_mlx_section_value_owned.json"
DEFAULT_SECTIONS = ("decoder_qw", "latents_rc")
SUPPORTED_SECTIONS = frozenset((*DEFAULT_SECTIONS, "rdo_plan", "receiver_state", "residual_rc"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--submission-dir",
        required=True,
        type=Path,
        help="PR95 runtime directory containing inflate.sh/inflate.py.",
    )
    parser.add_argument("--projection-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--video-names-file", type=Path)
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
    parser.add_argument("--preserve-existing-cache-reports", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    repo_root = _resolve(args.repo_root, base=REPO_ROOT)
    archive = _resolve(args.archive, base=repo_root)
    submission_dir = _resolve(args.submission_dir, base=repo_root)
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
    packet = parse_pr95_public_archive_zip(archive)
    layout = _pr95_layout_report(packet=packet, submission_dir=submission_dir)
    variants, absent_sections = _materialize_variants(
        archive=archive,
        packet=packet,
        submission_dir=submission_dir,
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
        response_family_prefix="pr95_hnerv_section_value",
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
        profile_schema=PR95_HNERV_SECTION_VALUE_PROFILE_SCHEMA,
        source_schema=PR95_HNERV_SECTION_VALUE_SOURCE_SCHEMA,
        layout_key="pr95_hnerv_section_layout",
        residual_policy_schema="pr95_hnerv_residual_admission_policy.v1",
        tool_path=Path(__file__),
    )
    report["family"] = "pr95_hnerv"
    report["submission_runtime"] = _runtime_row(submission_dir)
    for row in report.get("section_value_rows", []):
        if isinstance(row, dict):
            row["family"] = "pr95_hnerv"
    report_path = output_dir / "pr95_hnerv_mlx_section_value_profile.json"
    write_json(report_path, report)
    compat_path = output_dir / "hprc_mlx_component_neutralization_profile.json"
    write_json(compat_path, report)
    print(
        json.dumps(
            {
                "schema": PR95_HNERV_SECTION_VALUE_PROFILE_SCHEMA,
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
    packet: Any,
    submission_dir: Path,
    output_dir: Path,
    requested_sections: list[str],
) -> tuple[list[VariantSpec], list[dict[str, Any]]]:
    variants: list[VariantSpec] = []
    absent: list[dict[str, Any]] = []
    baseline_dir = output_dir / "variants" / "baseline"
    baseline_archive = baseline_dir / "archive.zip"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, baseline_archive)
    baseline_bin = baseline_dir / "0.bin"
    baseline_bin.write_bytes(_read_pr95_member_bytes(baseline_archive))
    variants.append(
        _variant_row(
            variant_id="baseline",
            neutralized_section=None,
            archive_zip_path=baseline_archive,
            submission_dir=submission_dir,
            bin_path=baseline_bin,
            variant_dir=baseline_dir,
        )
    )
    for section in requested_sections:
        if section not in SUPPORTED_SECTIONS:
            raise SystemExit(
                f"unsupported section {section!r}; valid={tuple(sorted(SUPPORTED_SECTIONS))}"
            )
        if section in {"residual_rc", "rdo_plan"}:
            absent.append(_absent_section_row(section))
            continue
        if section == "receiver_state":
            absent.append(
                {
                    "section": section,
                    "status": "blocked_receiver_state_neutralization_would_change_decoder_contract",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        variant_id = f"neutralize_{section}"
        variant_dir = output_dir / "variants" / variant_id
        variant_archive = variant_dir / "archive.zip"
        variant_dir.mkdir(parents=True, exist_ok=True)
        state_dict = packet.state_dict
        latents = packet.latents
        if section == "decoder_qw":
            state_dict = {
                name: np.zeros_like(value, dtype=np.float32)
                for name, value in packet.state_dict.items()
            }
        elif section == "latents_rc":
            latents = np.zeros_like(packet.latents, dtype=np.float32)
        write_pr95_public_archive_zip(
            state_dict,
            latents,
            meta=packet.meta,
            output_zip_path=variant_archive,
        )
        variant_bin = variant_dir / "0.bin"
        variant_bin.write_bytes(_read_pr95_member_bytes(variant_archive))
        variants.append(
            _variant_row(
                variant_id=variant_id,
                neutralized_section=section,
                archive_zip_path=variant_archive,
                submission_dir=submission_dir,
                bin_path=variant_bin,
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


def _pr95_layout_report(*, packet: Any, submission_dir: Path) -> dict[str, Any]:
    return {
        "schema": "pr95_hnerv_section_layout.v1",
        "member_name": packet.member_name,
        "member_bytes": packet.member_bytes,
        "member_sha256": packet.member_sha256,
        "n_pairs": int(packet.latents.shape[0]),
        "latent_dim": int(packet.latents.shape[1]),
        "state_dict_tensor_count": len(packet.state_dict),
        "meta": dict(packet.meta),
        "submission_runtime": _runtime_row(submission_dir),
        "supported_neutralization_sections": list(DEFAULT_SECTIONS),
        **FALSE_AUTHORITY,
    }


def _runtime_row(submission_dir: Path) -> dict[str, Any]:
    inflate = submission_dir / "inflate.sh"
    return {
        "schema": "pr95_hnerv_runtime_ref.v1",
        "submission_dir": submission_dir.as_posix(),
        "inflate_sh_path": inflate.as_posix(),
        "inflate_sh_exists": inflate.is_file(),
        "inflate_sh_sha256": _sha256_file(inflate) if inflate.is_file() else None,
        **FALSE_AUTHORITY,
    }


def _read_pr95_member_bytes(archive_zip: Path) -> bytes:
    import zipfile

    with zipfile.ZipFile(archive_zip) as zf:
        return zf.read("0.bin")


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"profile_pr95_hnerv_mlx_section_value failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
