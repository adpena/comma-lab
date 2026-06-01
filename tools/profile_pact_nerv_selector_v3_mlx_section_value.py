#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Price PSV3 PACT-NeRV sections with full-video MLX scorer replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np  # noqa: E402

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score  # noqa: E402
from tac.local_acceleration.mlx_response_windows import (  # noqa: E402
    split_mlx_scorer_response_windows,
)
from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MANIFEST_CACHE_INTEGRITY_MODE,
    MLXScorerResponseBatchJob,
    build_mlx_scorer_response_payload_batch,
    write_mlx_scorer_response_payload,
)
from tac.repo_io import write_json  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.pact_nerv_selector_v3.section_value import (  # noqa: E402
    PSV3_SECTION_VALUE_PROFILE_SCHEMA,
    PSV3_SECTION_VALUE_SOURCE_SCHEMA,
    PSV3_SUPPORTED_SECTION_NAMES,
    neutralize_psv3_section,
    psv3_layout_report,
    write_zip_replacing_member,
)

OWNED_MARKER = ".pact_nerv_selector_v3_mlx_section_value_owned.json"
DEFAULT_REFERENCE_CACHE = (
    REPO_ROOT / "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_SECTIONS = ("decoder_qw", "latents_rc", "selectors_rc", "residual_rc")


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    neutralized_section: str | None
    archive_zip_path: Path
    submission_dir: Path
    bin_path: Path
    archive_bytes: int
    archive_sha256: str
    bin_sha256: str
    variant_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--projection-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
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
    _prepare_owned_dir(output_dir, force=bool(args.force))

    started = time.time()
    baseline_blob = _read_archive_member(archive, "0.bin")
    layout = psv3_layout_report(blob=baseline_blob)
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
        reference_cache_dir=reference_cache_dir,
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        device=str(args.device),
        progress_every=int(args.progress_every),
        allow_batch_shape_research_signal=bool(args.allow_batch_shape_research_signal),
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
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        started=started,
    )
    report_path = output_dir / "pact_nerv_selector_v3_mlx_section_value_profile.json"
    write_json(report_path, report)
    compat_path = output_dir / "hprc_mlx_component_neutralization_profile.json"
    write_json(compat_path, report)
    print(
        json.dumps(
            {
                "schema": PSV3_SECTION_VALUE_PROFILE_SCHEMA,
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
    baseline_repack = write_zip_replacing_member(
        source_archive=archive,
        output_archive=baseline_archive,
        member_name="0.bin",
        replacement_bytes=baseline_blob,
        allow_overwrite=True,
    )
    write_json(baseline_dir / "zip_replacement_report.json", baseline_repack)
    _extract_submission(baseline_archive, baseline_dir / "submission")
    baseline_bin = baseline_dir / "submission" / "0.bin"
    variants.append(
        _variant_row(
            variant_id="baseline",
            neutralized_section=None,
            archive_zip_path=baseline_archive,
            submission_dir=baseline_dir / "submission",
            bin_path=baseline_bin,
            variant_dir=baseline_dir,
        )
    )
    for section in requested_sections:
        if section == "residual_rc":
            absent.append(_absent_section_row(section))
            continue
        if section not in PSV3_SUPPORTED_SECTION_NAMES:
            raise SystemExit(
                f"unsupported section {section!r}; valid={(*PSV3_SUPPORTED_SECTION_NAMES, 'residual_rc')}"
            )
        if section == "receiver_state":
            absent.append(
                {
                    "section": section,
                    "status": "blocked_receiver_state_neutralization_would_invalidate_decoder_shape",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        variant_id = f"neutralize_{section}"
        variant_dir = output_dir / "variants" / variant_id
        variant_archive = variant_dir / "archive.zip"
        variant_dir.mkdir(parents=True, exist_ok=True)
        neutralized = neutralize_psv3_section(baseline_blob, section)
        replacement = write_zip_replacing_member(
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


def _absent_section_row(section: str) -> dict[str, Any]:
    return {
        "section": section,
        "status": "absent_no_candidate_bytes_admitted",
        "admission_status": "demote_residual_token_variant",
        "admission_rule": "candidate_delta_nonrate + candidate_delta_rate < 0",
        "objective_delta": 0.0,
        **FALSE_AUTHORITY,
    }


def _materialize_caches(
    *,
    variants: list[VariantSpec],
    output_dir: Path,
    repo_root: Path,
    upstream_dir: Path,
    video_names_file: Path | None,
    max_pairs: int,
    inflate_timeout: int,
    allow_large_tensor_cache: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    tool = repo_root / "tools/materialize_mlx_scorer_cache_from_submission.py"
    for variant in variants:
        cache_dir = output_dir / "mlx_caches" / variant.variant_id
        work_dir = output_dir / "mlx_work" / variant.variant_id
        report_output = output_dir / "mlx_cache_reports" / f"{variant.variant_id}.json"
        cmd = [
            sys.executable,
            tool.as_posix(),
            "--archive",
            variant.archive_zip_path.as_posix(),
            "--submission-dir",
            variant.submission_dir.as_posix(),
            "--output-cache-dir",
            cache_dir.as_posix(),
            "--work-dir",
            work_dir.as_posix(),
            "--report-output",
            report_output.as_posix(),
            "--upstream-dir",
            upstream_dir.as_posix(),
            "--max-pairs",
            str(max_pairs),
            "--inflate-timeout",
            str(inflate_timeout),
            "--force",
        ]
        if video_names_file is not None:
            cmd.extend(["--video-names-file", video_names_file.as_posix()])
        if allow_large_tensor_cache:
            cmd.append("--allow-large-tensor-cache")
        completed = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"MLX cache materialization failed for {variant.variant_id}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        rows[variant.variant_id] = {
            "cache_dir": cache_dir.as_posix(),
            "work_dir": work_dir.as_posix(),
            "report_output": report_output.as_posix(),
            "argv": cmd,
            "upstream_dir": upstream_dir.as_posix(),
            "video_names_file": (
                None if video_names_file is None else video_names_file.as_posix()
            ),
            "stdout": completed.stdout.strip(),
            "stderr_tail": completed.stderr.strip()[-4000:],
        }
    return rows


def _run_mlx_responses(
    *,
    variants: list[VariantSpec],
    output_dir: Path,
    repo_root: Path,
    reference_cache_dir: Path,
    max_pairs: int,
    window_pairs: int,
    scorer_batch_pairs: int,
    device: str,
    progress_every: int,
    allow_batch_shape_research_signal: bool,
    response_family_prefix: str = "pact_nerv_selector_v3_section_value",
) -> dict[str, dict[str, Any]]:
    jobs = [
        MLXScorerResponseBatchJob(
            candidate_cache_dir=output_dir / "mlx_caches" / variant.variant_id,
            archive_size_bytes=variant.archive_bytes,
            output=output_dir / "mlx_responses" / f"{variant.variant_id}.json",
            components_dir=output_dir / "mlx_components" / variant.variant_id,
            response_family=f"{response_family_prefix}_{variant.variant_id}",
        )
        for variant in variants
    ]
    payloads = build_mlx_scorer_response_payload_batch(
        reference_cache_dir=reference_cache_dir,
        jobs=jobs,
        repo_root=repo_root,
        batch_pairs=int(scorer_batch_pairs),
        device_type=device,
        progress_every=progress_every,
        max_pairs=max_pairs,
        allow_gpu_research_signal=device == "gpu",
        allow_batch_shape_research_signal=allow_batch_shape_research_signal,
        allow_unaudited_candidate_cache_debug=True,
        cache_integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
    )
    out: dict[str, dict[str, Any]] = {}
    for variant, payload in zip(variants, payloads, strict=True):
        path = output_dir / "mlx_responses" / f"{variant.variant_id}.json"
        write_mlx_scorer_response_payload(payload, path)
        if int(scorer_batch_pairs) == 1 and payload.get("components", {}).get("artifacts"):
            _write_window_splits(
                payload=payload,
                output_dir=output_dir,
                variant_id=variant.variant_id,
                window_pairs=window_pairs,
            )
        out[variant.variant_id] = payload
    return out


def _build_report(
    *,
    raw_argv: list[str],
    variants: list[VariantSpec],
    absent_sections: list[dict[str, Any]],
    cache_rows: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    layout: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    upstream_dir: Path,
    video_names_file: Path | None,
    archive: Path,
    projection_manifest: Path | None,
    reference_cache_dir: Path,
    max_pairs: int,
    window_pairs: int,
    scorer_batch_pairs: int,
    started: float,
    profile_schema: str = PSV3_SECTION_VALUE_PROFILE_SCHEMA,
    source_schema: str = PSV3_SECTION_VALUE_SOURCE_SCHEMA,
    layout_key: str = "psv3_section_layout",
    residual_policy_schema: str = "pact_nerv_selector_v3_residual_admission_policy.v1",
    tool_path: Path | None = None,
) -> dict[str, Any]:
    baseline = payloads["baseline"]
    baseline_variant = variants[0]
    section_rows = [
        _section_value_row(
            baseline=baseline,
            baseline_variant=baseline_variant,
            variant=variant,
            payload=payloads[variant.variant_id],
            projection_manifest=projection_manifest,
        )
        for variant in variants
    ]
    for absent in absent_sections:
        if absent["section"] == "residual_rc":
            section_rows.append(
                {
                    "variant_id": "residual_absent_no_admission",
                    "neutralized_section": "residual_rc",
                    "archive_zip_bytes": int(baseline_variant.archive_bytes),
                    "archive_bytes_removed_vs_baseline": 0,
                    "delta_avg_posenet_dist": 0.0,
                    "delta_avg_segnet_dist": 0.0,
                    "delta_nonrate_score": 0.0,
                    "delta_rate_score": 0.0,
                    "delta_total_mlx_score_advisory": 0.0,
                    "marginal_status": "residual_section_absent_no_token_admitted",
                    "admission_status": "demote_residual_token_variant",
                    "family": "pact_nerv",
                    "projection_manifest_path": (
                        None if projection_manifest is None else projection_manifest.as_posix()
                    ),
                    **FALSE_AUTHORITY,
                }
            )
    return {
        "schema": profile_schema,
        "source_schema": source_schema,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "repo_root": repo_root.as_posix(),
        "upstream_dir": upstream_dir.as_posix(),
        "video_names_file": None if video_names_file is None else video_names_file.as_posix(),
        "tool_argv": [
            sys.executable,
            str((tool_path or Path(__file__)).resolve()),
            *raw_argv,
        ],
        "family": "pact_nerv",
        "projection_manifest_path": None if projection_manifest is None else projection_manifest.as_posix(),
        "candidate_archive": {
            "path": archive.as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        },
        "reference_cache_dir": reference_cache_dir.as_posix(),
        layout_key: layout,
        "cache_materialization_rows": cache_rows,
        "max_pairs": int(max_pairs),
        "window_pairs": int(window_pairs),
        "scorer_batch_pairs": int(scorer_batch_pairs),
        "batch_shape_research_signal": int(scorer_batch_pairs) != 1,
        "variant_rows": [
            {
                "variant_id": variant.variant_id,
                "neutralized_section": variant.neutralized_section,
                "archive_zip_path": variant.archive_zip_path.as_posix(),
                "archive_zip_bytes": int(variant.archive_bytes),
                "archive_zip_sha256": variant.archive_sha256,
                "bin_path": variant.bin_path.as_posix(),
                "bin_sha256": variant.bin_sha256,
                "cache_report": cache_rows[variant.variant_id]["report_output"],
                "mlx_response": (output_dir / "mlx_responses" / f"{variant.variant_id}.json").as_posix(),
            }
            for variant in variants
        ],
        "section_value_rows": section_rows,
        "absent_section_rows": absent_sections,
        "scope_status": {
            "section": "executed",
            "pair": "component arrays emitted by MLX response when singleton batch pairs are used",
            "batch": (
                "window JSON emitted from MLX component arrays"
                if int(scorer_batch_pairs) == 1
                else "batched scorer response executed as research signal only"
            ),
            "full_video": "executed" if int(max_pairs) >= 600 else "sampled_prefix_requires_full_video_rerun",
            "class_region": "blocked_missing_segnet_class_logit_or_label_surface_in_cache",
            "boundary": "blocked_missing_boundary_surface_in_cache",
        },
        "residual_admission_policy": {
            "schema": residual_policy_schema,
            "rule": "admit residual bytes only when measured_delta_nonrate + rate_cost < 0",
            "observed_residual_section": "absent",
            "default_action": "demote_unmeasured_or_absent_residual_tokens",
            **FALSE_AUTHORITY,
        },
        "blockers": [
            "mlx_local_response_is_advisory_not_score_authority",
            *([] if int(max_pairs) >= 600 else ["full_video_mlx_response_not_executed"]),
            *(
                ["batch_shape_research_signal_requires_singleton_rerun_before_promotion"]
                if int(scorer_batch_pairs) != 1
                else []
            ),
            "class_region_boundary_scopes_require_logits_or_boundary_cache_extension",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def _section_value_row(
    *,
    baseline: dict[str, Any],
    baseline_variant: VariantSpec,
    variant: VariantSpec,
    payload: dict[str, Any],
    projection_manifest: Path | None,
) -> dict[str, Any]:
    delta_pose = float(payload["avg_posenet_dist"]) - float(baseline["avg_posenet_dist"])
    delta_seg = float(payload["avg_segnet_dist"]) - float(baseline["avg_segnet_dist"])
    baseline_nonrate = _nonrate_score(baseline)
    variant_nonrate = _nonrate_score(payload)
    delta_nonrate = variant_nonrate - baseline_nonrate
    archive_bytes_removed = int(baseline_variant.archive_bytes) - int(variant.archive_bytes)
    delta_rate = float(payload["score_rate_contribution"]) - float(
        baseline["score_rate_contribution"]
    )
    total_delta = float(payload["canonical_score"]) - float(baseline["canonical_score"])
    section = variant.neutralized_section or "none"
    return {
        "variant_id": variant.variant_id,
        "neutralized_section": section,
        "family": "pact_nerv",
        "projection_manifest_path": None if projection_manifest is None else projection_manifest.as_posix(),
        "archive_zip_bytes": int(variant.archive_bytes),
        "archive_bytes_removed_vs_baseline": int(archive_bytes_removed),
        "delta_avg_posenet_dist": delta_pose,
        "delta_avg_segnet_dist": delta_seg,
        "delta_nonrate_score": delta_nonrate,
        "delta_rate_score": delta_rate,
        "delta_total_mlx_score_advisory": total_delta,
        "nonrate_score_value_per_removed_archive_kib": (
            delta_nonrate / max(archive_bytes_removed / 1024.0, 1.0e-9)
            if archive_bytes_removed > 0
            else None
        ),
        "rate_price_score_per_kib": (25.0 / ORIGINAL_VIDEO_BYTES) * 1024.0,
        "marginal_status": _marginal_status(delta_nonrate, archive_bytes_removed),
        **FALSE_AUTHORITY,
    }


def _write_window_splits(
    *,
    payload: dict[str, Any],
    output_dir: Path,
    variant_id: str,
    window_pairs: int,
) -> None:
    artifacts = payload.get("components", {}).get("artifacts", {})
    pose_path = artifacts.get("posenet_distortion", {}).get("path")
    seg_path = artifacts.get("segnet_distortion", {}).get("path")
    if not pose_path or not seg_path:
        return
    split_mlx_scorer_response_windows(
        response_payload=payload,
        posenet_distortion=np.load(pose_path),
        segnet_distortion=np.load(seg_path),
        output_dir=output_dir / "mlx_windows" / variant_id,
        window_pairs=max(1, int(window_pairs)),
        components_dir=output_dir / "mlx_window_components" / variant_id,
    )


def _nonrate_score(payload: dict[str, Any]) -> float:
    return contest_formula_score(
        seg_dist=float(payload["avg_segnet_dist"]),
        pose_dist=float(payload["avg_posenet_dist"]),
        archive_bytes=0,
    )


def _marginal_status(delta_nonrate: float, archive_bytes_removed: int) -> str:
    if archive_bytes_removed <= 0:
        return "no_archive_byte_savings"
    paid_rate_score = archive_bytes_removed * (25.0 / ORIGINAL_VIDEO_BYTES)
    if delta_nonrate <= 0.0:
        return "cut_candidate_distortion_nonworse"
    if delta_nonrate < paid_rate_score:
        return "cut_candidate_value_below_rate_price"
    return "protect_candidate_value_exceeds_rate_price"


def _extract_submission(archive: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)


def _read_archive_member(archive: Path, member: str) -> bytes:
    with zipfile.ZipFile(archive) as zf:
        return zf.read(member)


def _prepare_owned_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        marker = path / OWNED_MARKER
        if not force and any(path.iterdir()):
            raise SystemExit(f"output dir exists; pass --force: {path}")
        if force:
            if not marker.exists() and any(path.iterdir()):
                raise SystemExit(f"refusing --force on non-owned output dir: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / OWNED_MARKER).write_text(
        json.dumps({"schema": "owned_directory_marker.v1", "tool": Path(__file__).name})
        + "\n",
        encoding="utf-8",
    )


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"profile_pact_nerv_selector_v3_mlx_section_value failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
