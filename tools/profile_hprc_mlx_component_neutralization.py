#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile HPRC component neutralizations through the MLX scorer-response port.

The tool materializes valid archive/runtime variants, inflates them into fixed
scorer-input caches, runs the non-authoritative MLX scorer-response adapter, and
emits allocator rows. It deliberately refuses to create score authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
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
from tac.repo_io import sha256_file, write_json  # noqa: E402
from tac.substrates.hprc.archive import HprcSectionKind, parse_hprc_packet  # noqa: E402
from tac.substrates.hprc.archive_candidate import (  # noqa: E402
    HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING,
    export_hprc_archive_bytes,
)
from tac.substrates.hprc.learned_receiver import (  # noqa: E402
    COMPACT_RECEIVER_MODE,
    decode_compact_receiver_packet,
    neutralize_compact_receiver_section,
    transform_compact_receiver_residual,
)

SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
BACKLOG_SCHEMA = "hprc_scorer_ranked_residual_shrink_backlog.v1"
OWNED_MARKER = ".hprc_mlx_component_neutralization_owned.json"
DEFAULT_REFERENCE_CACHE = (
    REPO_ROOT / "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)
DEFAULT_SECTIONS = (
    "residual_rc",
    "decoder_qw",
    "latents_rc",
    "selectors_rc",
    "receiver_state",
)
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    neutralized_section: str | None
    archive_zip_path: Path
    submission_dir: Path
    hprc_bin_path: Path
    archive_bytes: int
    archive_sha256: str
    hprc_0bin_sha256: str
    variant_dir: Path


@dataclass(frozen=True)
class BaselineReuse:
    source_profile_path: Path
    source_cache_report_path: Path
    source_response_path: Path
    source_components_dir: Path | None
    source_cache_dir: Path | None
    source_profile: dict[str, Any]
    source_variant_row: dict[str, Any]
    source_payload: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / ".omx/research/hprc_compact_receiver_z8_reference_20260531T233001Z",
        help="HPRC materialization directory containing 0.bin, archive.zip, and submission/.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--reference-cache-dir", type=Path, default=DEFAULT_REFERENCE_CACHE)
    parser.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS))
    parser.add_argument(
        "--residual-transforms",
        nargs="*",
        default=[],
        help=(
            "Additional residual-token shrink transforms to materialize and score, "
            "for example threshold_abs_le=1 quant_step=2 keep_top_fraction=0.2."
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--window-pairs", type=int, default=1)
    parser.add_argument(
        "--scorer-batch-pairs",
        type=int,
        default=1,
        help=(
            "MLX scorer-response batch width. Values above 1 are faster but remain "
            "batch-shape research signal and require --allow-batch-shape-research-signal."
        ),
    )
    parser.add_argument(
        "--allow-batch-shape-research-signal",
        action="store_true",
        help=(
            "Permit --scorer-batch-pairs > 1. Batched rows are never promotion "
            "authority without singleton/full exact replay."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument(
        "--cache-materialization-mode",
        choices=("hprc-direct", "shell-inflate"),
        default="hprc-direct",
        help=(
            "How to build advisory MLX scorer caches for HPRC variants. "
            "hprc-direct avoids multi-GB raw scratch; shell-inflate preserves "
            "the older full inflate path for parity audits."
        ),
    )
    parser.add_argument(
        "--allow-large-tensor-cache",
        action="store_true",
        help="Permit full-video scorer tensor caches after SSD/storage preflight.",
    )
    parser.add_argument(
        "--reuse-baseline-profile",
        type=Path,
        help=(
            "Reuse the baseline MLX response from a prior HPRC profile when the "
            "baseline 0.bin SHA, reference cache, and requested pair count match. "
            "This skips recomputing the unchanged baseline scorer pass."
        ),
    )
    parser.add_argument(
        "--receiver-proof",
        action="store_true",
        help="Run full receiver proof for every emitted variant. Slow; prefer survivors only.",
    )
    parser.add_argument(
        "--skip-receiver-proof",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _resolve(args.repo_root, base=REPO_ROOT)
    candidate_dir = _resolve(args.candidate_dir, base=repo_root)
    output_dir = _resolve(args.output_dir, base=repo_root)
    reference_cache_dir = _resolve(args.reference_cache_dir, base=repo_root)
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
        allow_existing=bool(args.allow_existing_output_dir),
    )

    started = time.time()
    packet_path = _require_file(candidate_dir / "0.bin", "candidate 0.bin")
    packet = parse_hprc_packet(packet_path.read_bytes())
    decode_compact_receiver_packet(packet)
    section_kinds = [_section_kind(name) for name in args.sections]
    variants = _materialize_variants(
        packet_path=packet_path,
        packet_bytes=packet_path.read_bytes(),
        packet=packet,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
        repo_root=repo_root,
        section_kinds=section_kinds,
        residual_transforms=list(args.residual_transforms),
        skip_receiver_proof=(not bool(args.receiver_proof)) or bool(args.skip_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
    )
    baseline_reuse = _prepare_baseline_reuse(
        profile_path=(
            None
            if args.reuse_baseline_profile is None
            else _resolve(args.reuse_baseline_profile, base=repo_root)
        ),
        baseline_variant=variants[0],
        reference_cache_dir=reference_cache_dir,
        max_pairs=int(args.max_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
    )
    cache_rows = _materialize_mlx_caches(
        variants=variants,
        output_dir=output_dir,
        repo_root=repo_root,
        max_pairs=int(args.max_pairs),
        inflate_timeout=int(args.inflate_timeout),
        allow_large_tensor_cache=bool(args.allow_large_tensor_cache),
        cache_materialization_mode=str(args.cache_materialization_mode),
        baseline_reuse=baseline_reuse,
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
        baseline_reuse=baseline_reuse,
        allow_batch_shape_research_signal=bool(args.allow_batch_shape_research_signal),
    )
    report = _build_report(
        variants=variants,
        cache_rows=cache_rows,
        payloads=payloads,
        output_dir=output_dir,
        repo_root=repo_root,
        candidate_dir=candidate_dir,
        reference_cache_dir=reference_cache_dir,
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        started=started,
        baseline_reuse=baseline_reuse,
    )
    report_path = output_dir / "hprc_mlx_component_neutralization_profile.json"
    write_json(report_path, report)
    backlog = _build_shrink_backlog(report)
    backlog_path = output_dir / "hprc_scorer_ranked_residual_shrink_backlog.json"
    write_json(backlog_path, backlog)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "backlog": str(backlog_path),
                "variant_count": len(variants),
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def _prepare_owned_dir(path: Path, *, force: bool, allow_existing: bool) -> None:
    if path.exists():
        marker = path / OWNED_MARKER
        if force:
            if not marker.exists() and any(path.iterdir()):
                raise SystemExit(f"refusing --force on non-owned output dir: {path}")
            shutil.rmtree(path)
        elif not allow_existing and any(path.iterdir()):
            raise SystemExit(f"output dir exists; pass --allow-existing-output-dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / OWNED_MARKER).write_text(
        json.dumps({"schema": "owned_directory_marker.v1", "tool": Path(__file__).name})
        + "\n",
        encoding="utf-8",
    )


def _section_kind(name: str) -> HprcSectionKind:
    key = name.lower()
    try:
        return HprcSectionKind[key.upper()]
    except KeyError as exc:
        valid = [kind.name.lower() for kind in HprcSectionKind]
        raise SystemExit(f"unknown HPRC section {name!r}; valid={valid}") from exc


def _materialize_variants(
    *,
    packet_path: Path,
    packet_bytes: bytes,
    packet: Any,
    candidate_dir: Path,
    output_dir: Path,
    repo_root: Path,
    section_kinds: list[HprcSectionKind],
    residual_transforms: list[str],
    skip_receiver_proof: bool,
    retain_receiver_proof_output: bool,
) -> list[VariantSpec]:
    _require_file(candidate_dir / "archive.zip", "candidate archive.zip")
    baseline_dir = output_dir / "variants" / "baseline"
    baseline_archive, baseline_sha, baseline_bytes = export_hprc_archive_bytes(
        packet_bytes,
        baseline_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=not skip_receiver_proof,
        retain_receiver_proof_output=retain_receiver_proof_output,
    )
    variants = [
        VariantSpec(
            variant_id="baseline",
            neutralized_section=None,
            archive_zip_path=baseline_archive,
            submission_dir=baseline_dir / "submission",
            hprc_bin_path=baseline_dir / "0.bin",
            archive_bytes=int(baseline_bytes),
            archive_sha256=baseline_sha,
            hprc_0bin_sha256=sha256_file(baseline_dir / "0.bin"),
            variant_dir=baseline_dir,
        )
    ]
    for kind in section_kinds:
        variant_id = f"neutralize_{kind.name.lower()}"
        variant_dir = output_dir / "variants" / variant_id
        variant_packet = neutralize_compact_receiver_section(packet, kind)
        archive_zip, archive_sha, archive_bytes = export_hprc_archive_bytes(
            variant_packet,
            variant_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=not skip_receiver_proof,
            retain_receiver_proof_output=retain_receiver_proof_output,
        )
        variants.append(
            VariantSpec(
                variant_id=variant_id,
                neutralized_section=kind.name.lower(),
                archive_zip_path=archive_zip,
                submission_dir=variant_dir / "submission",
                hprc_bin_path=variant_dir / "0.bin",
                archive_bytes=int(archive_bytes),
                archive_sha256=archive_sha,
                hprc_0bin_sha256=sha256_file(variant_dir / "0.bin"),
                variant_dir=variant_dir,
            )
        )
    for transform in residual_transforms:
        variant_id = f"residual_transform_{_variant_slug(transform)}"
        variant_dir = output_dir / "variants" / variant_id
        variant_packet = transform_compact_receiver_residual(packet, transform=transform)
        archive_zip, archive_sha, archive_bytes = export_hprc_archive_bytes(
            variant_packet,
            variant_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=not skip_receiver_proof,
            retain_receiver_proof_output=retain_receiver_proof_output,
        )
        variants.append(
            VariantSpec(
                variant_id=variant_id,
                neutralized_section="residual_rc",
                archive_zip_path=archive_zip,
                submission_dir=variant_dir / "submission",
                hprc_bin_path=variant_dir / "0.bin",
                archive_bytes=int(archive_bytes),
                archive_sha256=archive_sha,
                hprc_0bin_sha256=sha256_file(variant_dir / "0.bin"),
                variant_dir=variant_dir,
            )
        )
    return variants


def _variant_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    if len(slug) <= 80:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:63].rstrip('_')}_{digest}"


def _prepare_baseline_reuse(
    *,
    profile_path: Path | None,
    baseline_variant: VariantSpec,
    reference_cache_dir: Path,
    max_pairs: int,
    scorer_batch_pairs: int,
) -> BaselineReuse | None:
    if profile_path is None:
        return None
    payload = _load_json_object(profile_path)
    if payload.get("schema") != SCHEMA:
        raise SystemExit(
            f"baseline reuse profile has wrong schema: {payload.get('schema')!r}"
        )
    if int(payload.get("max_pairs") or 0) != int(max_pairs):
        raise SystemExit(
            "baseline reuse requires identical max_pairs: "
            f"source={payload.get('max_pairs')} requested={max_pairs}"
        )
    source_reference = Path(str(payload.get("reference_cache_dir") or "")).expanduser()
    if source_reference != reference_cache_dir:
        raise SystemExit(
            "baseline reuse requires identical reference cache: "
            f"source={source_reference} requested={reference_cache_dir}"
        )
    variant_rows = payload.get("variant_rows")
    if not isinstance(variant_rows, list):
        raise SystemExit("baseline reuse profile is missing variant_rows")
    baseline_rows = [
        row
        for row in variant_rows
        if isinstance(row, dict) and row.get("variant_id") == "baseline"
    ]
    if len(baseline_rows) != 1:
        raise SystemExit("baseline reuse profile must contain exactly one baseline row")
    row = baseline_rows[0]
    if row.get("hprc_0bin_sha256") != baseline_variant.hprc_0bin_sha256:
        raise SystemExit(
            "baseline reuse refused: hprc_0bin_sha256 mismatch "
            f"source={row.get('hprc_0bin_sha256')} "
            f"current={baseline_variant.hprc_0bin_sha256}"
        )
    source_response_path = Path(str(row.get("mlx_response") or ""))
    source_cache_report_path = Path(str(row.get("cache_report") or ""))
    if not source_response_path.is_file():
        raise SystemExit(f"baseline reuse response missing: {source_response_path}")
    if not source_cache_report_path.is_file():
        raise SystemExit(f"baseline reuse cache report missing: {source_cache_report_path}")
    source_payload = _load_json_object(source_response_path)
    if int(source_payload.get("n_samples") or 0) != int(max_pairs):
        raise SystemExit(
            "baseline reuse response n_samples mismatch: "
            f"source={source_payload.get('n_samples')} requested={max_pairs}"
        )
    source_batch_pairs = int(source_payload.get("batch_pairs") or 1)
    if source_batch_pairs != int(scorer_batch_pairs):
        raise SystemExit(
            "baseline reuse response batch_pairs mismatch: "
            f"source={source_batch_pairs} requested={scorer_batch_pairs}. "
            "Run a matching baseline profile instead of mixing batch-shape evidence."
        )
    components = source_payload.get("components", {})
    artifacts = components.get("artifacts") if isinstance(components, dict) else None
    source_components_dir: Path | None = None
    if isinstance(artifacts, dict):
        paths = [
            Path(str(row.get("path")))
            for row in artifacts.values()
            if isinstance(row, dict) and row.get("path")
        ]
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"baseline reuse component artifact missing: {path}")
        if paths:
            source_components_dir = paths[0].parent
    cache_rows = payload.get("cache_materialization_rows")
    source_cache_dir = None
    if isinstance(cache_rows, dict):
        baseline_cache = cache_rows.get("baseline")
        if isinstance(baseline_cache, dict) and baseline_cache.get("cache_dir"):
            source_cache_dir = Path(str(baseline_cache["cache_dir"]))
    return BaselineReuse(
        source_profile_path=profile_path,
        source_cache_report_path=source_cache_report_path,
        source_response_path=source_response_path,
        source_components_dir=source_components_dir,
        source_cache_dir=source_cache_dir,
        source_profile=payload,
        source_variant_row=row,
        source_payload=source_payload,
    )


def _materialize_mlx_caches(
    *,
    variants: list[VariantSpec],
    output_dir: Path,
    repo_root: Path,
    max_pairs: int,
    inflate_timeout: int,
    allow_large_tensor_cache: bool,
    cache_materialization_mode: str,
    baseline_reuse: BaselineReuse | None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    tool = repo_root / "tools" / "materialize_mlx_scorer_cache_from_submission.py"
    for variant in variants:
        cache_dir = output_dir / "mlx_caches" / variant.variant_id
        work_dir = output_dir / "mlx_work" / variant.variant_id
        report_output = output_dir / "mlx_cache_reports" / f"{variant.variant_id}.json"
        if variant.variant_id == "baseline" and baseline_reuse is not None:
            report_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_reuse.source_cache_report_path, report_output)
            rows[variant.variant_id] = {
                "cache_dir": (
                    str(baseline_reuse.source_cache_dir)
                    if baseline_reuse.source_cache_dir is not None
                    else str(cache_dir)
                ),
                "work_dir": str(work_dir),
                "report_output": str(report_output),
                "argv": [],
                "stdout": "",
                "stderr_tail": "",
                "reused_from_profile": str(baseline_reuse.source_profile_path),
                "reused_from_cache_report": str(baseline_reuse.source_cache_report_path),
                "reused_baseline_cache": True,
            }
            continue
        cmd = _build_mlx_cache_materialization_command(
            tool=tool,
            variant=variant,
            cache_dir=cache_dir,
            work_dir=work_dir,
            report_output=report_output,
            max_pairs=max_pairs,
            inflate_timeout=inflate_timeout,
            cache_materialization_mode=cache_materialization_mode,
        )
        if allow_large_tensor_cache:
            cmd.append("--allow-large-tensor-cache")
        completed = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if completed.returncode != 0:
            raise SystemExit(
                "MLX cache materialization failed for "
                f"{variant.variant_id}:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        rows[variant.variant_id] = {
            "cache_dir": str(cache_dir),
            "work_dir": str(work_dir),
            "report_output": str(report_output),
            "argv": cmd,
            "stdout": completed.stdout.strip(),
            "stderr_tail": completed.stderr.strip()[-4000:],
        }
    return rows


def _build_mlx_cache_materialization_command(
    *,
    tool: Path,
    variant: VariantSpec,
    cache_dir: Path,
    work_dir: Path,
    report_output: Path,
    max_pairs: int,
    inflate_timeout: int,
    cache_materialization_mode: str,
) -> list[str]:
    if cache_materialization_mode not in {"hprc-direct", "shell-inflate"}:
        raise ValueError(f"unknown cache materialization mode: {cache_materialization_mode}")
    cmd = [
        sys.executable,
        str(tool),
        "--archive",
        str(variant.archive_zip_path),
        "--submission-dir",
        str(variant.submission_dir),
        "--output-cache-dir",
        str(cache_dir),
        "--work-dir",
        str(work_dir),
        "--report-output",
        str(report_output),
        "--max-pairs",
        str(max_pairs),
        "--local-acquisition-max-pairs",
        str(max_pairs),
        "--inflate-timeout",
        str(inflate_timeout),
        "--force",
    ]
    if cache_materialization_mode == "hprc-direct":
        cmd.append("--hprc-direct-cache")
    return cmd


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
    baseline_reuse: BaselineReuse | None,
    allow_batch_shape_research_signal: bool,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    job_variants: list[VariantSpec] = []
    if baseline_reuse is not None:
        baseline = variants[0]
        out["baseline"] = _copy_reused_baseline_response(
            baseline_reuse=baseline_reuse,
            baseline_variant=baseline,
            output_dir=output_dir,
        )
        job_variants = [variant for variant in variants if variant.variant_id != "baseline"]
    else:
        job_variants = list(variants)
    jobs = [
        MLXScorerResponseBatchJob(
            candidate_cache_dir=output_dir / "mlx_caches" / variant.variant_id,
            archive_size_bytes=variant.archive_bytes,
            output=output_dir / "mlx_responses" / f"{variant.variant_id}.json",
            components_dir=output_dir / "mlx_components" / variant.variant_id,
            response_family=f"hprc_component_neutralization_{variant.variant_id}",
        )
        for variant in job_variants
    ]
    payloads = (
        build_mlx_scorer_response_payload_batch(
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
        if jobs
        else []
    )
    for variant, payload in zip(job_variants, payloads, strict=True):
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


def _copy_reused_baseline_response(
    *,
    baseline_reuse: BaselineReuse,
    baseline_variant: VariantSpec,
    output_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(baseline_reuse.source_payload)
    source_archive_bytes = int(payload.get("archive_size_bytes") or 0)
    _retarget_payload_archive_bytes(payload, archive_bytes=baseline_variant.archive_bytes)
    artifacts = payload.get("components", {}).get("artifacts", {})
    if isinstance(artifacts, dict):
        component_dir = output_dir / "mlx_components" / "baseline"
        component_dir.mkdir(parents=True, exist_ok=True)
        for row in artifacts.values():
            if not isinstance(row, dict) or not row.get("path"):
                continue
            source = Path(str(row["path"]))
            target = component_dir / source.name
            shutil.copy2(source, target)
            row["path"] = str(target)
            row["reused_from_path"] = str(source)
            row["reused_from_profile"] = str(baseline_reuse.source_profile_path)
    payload["baseline_reuse"] = {
        "schema": "hprc_baseline_mlx_response_reuse.v1",
        "source_profile": str(baseline_reuse.source_profile_path),
        "source_response": str(baseline_reuse.source_response_path),
        "source_cache_report": str(baseline_reuse.source_cache_report_path),
        "source_archive_size_bytes": source_archive_bytes,
        "current_archive_size_bytes": int(baseline_variant.archive_bytes),
        "current_archive_sha256": baseline_variant.archive_sha256,
        "hprc_0bin_sha256": baseline_variant.hprc_0bin_sha256,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    path = output_dir / "mlx_responses" / "baseline.json"
    write_mlx_scorer_response_payload(payload, path)
    return payload


def _retarget_payload_archive_bytes(payload: dict[str, Any], *, archive_bytes: int) -> None:
    archive_bytes_int = int(archive_bytes)
    payload["archive_size_bytes"] = archive_bytes_int
    payload["rate_unscaled"] = archive_bytes_int / ORIGINAL_VIDEO_BYTES
    payload["score_rate_contribution"] = 25.0 * payload["rate_unscaled"]
    score = contest_formula_score(
        seg_dist=float(payload["avg_segnet_dist"]),
        pose_dist=float(payload["avg_posenet_dist"]),
        archive_bytes=archive_bytes_int,
    )
    payload["canonical_score"] = score
    payload["score_recomputed_from_components"] = score
    payload["canonical_score_source"] = "score_recomputed_from_components_retargeted_archive_bytes"


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


def _build_report(
    *,
    variants: list[VariantSpec],
    cache_rows: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    output_dir: Path,
    repo_root: Path,
    candidate_dir: Path,
    reference_cache_dir: Path,
    max_pairs: int,
    window_pairs: int,
    scorer_batch_pairs: int,
    started: float,
    baseline_reuse: BaselineReuse | None,
) -> dict[str, Any]:
    baseline = payloads["baseline"]
    baseline_variant = variants[0]
    baseline_cache_report = _load_cache_report(cache_rows["baseline"]["report_output"])
    full_video_executed = _full_video_mlx_response_executed(
        cache_report=baseline_cache_report,
        max_pairs=max_pairs,
    )
    local_acquisition_partial_raw = _local_acquisition_partial_raw_blocker(
        baseline_cache_report
    )
    section_rows = []
    for variant in variants:
        payload = payloads[variant.variant_id]
        section_rows.append(
            _section_value_row(
                baseline=baseline,
                baseline_variant=baseline_variant,
                variant=variant,
                payload=payload,
            )
        )
    pair_rows = _pair_rows(
        baseline_payload=baseline,
        variants=variants[1:],
        payloads=payloads,
        max_pairs=max_pairs,
    )
    return {
        "schema": SCHEMA,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "repo_root": str(repo_root),
        "candidate_dir": str(candidate_dir),
        "reference_cache_dir": str(reference_cache_dir),
        "tool_argv": [sys.executable, *sys.argv],
        "cache_materialization_rows": cache_rows,
        "baseline_reuse": _baseline_reuse_report(baseline_reuse),
        "receiver_mode": COMPACT_RECEIVER_MODE,
        "max_pairs": int(max_pairs),
        "window_pairs": int(window_pairs),
        "scorer_batch_pairs": int(scorer_batch_pairs),
        "batch_shape_research_signal": int(scorer_batch_pairs) != 1,
        "archive_byte_ceiling": {
            "sub019_zero_distortion_archive_bytes": int(
                HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING
            ),
            "rate_price_score_per_byte": float(25.0 / ORIGINAL_VIDEO_BYTES),
        },
        "variant_rows": [
            {
                "variant_id": variant.variant_id,
                "neutralized_section": variant.neutralized_section,
                "archive_zip_path": str(variant.archive_zip_path),
                "archive_zip_bytes": int(variant.archive_bytes),
                "archive_zip_sha256": variant.archive_sha256,
                "hprc_0bin_path": str(variant.hprc_bin_path),
                "hprc_0bin_sha256": variant.hprc_0bin_sha256,
                "cache_report": cache_rows[variant.variant_id]["report_output"],
                "mlx_response": str(output_dir / "mlx_responses" / f"{variant.variant_id}.json"),
            }
            for variant in variants
        ],
        "section_value_rows": section_rows,
        "pair_value_rows": pair_rows,
        "scope_status": {
            "section": "executed",
            "pair": "executed_from_mlx_component_arrays",
            "frame": "pair rows include frame indices; per-frame split awaits framewise scorer arrays",
            "batch": (
                "window JSON emitted from MLX component arrays"
                if int(scorer_batch_pairs) == 1
                else "batched scorer response executed as research signal only"
            ),
            "batch_windows": (
                "emitted_from_singleton_batch_pairs"
                if int(scorer_batch_pairs) == 1
                else "blocked_for_batched_research_signal_requires_singleton_rerun"
            ),
            "full_video": (
                "executed" if full_video_executed else
                "sampled_prefix_requires_full_video_rerun"
            ),
            "class_region": "blocked_missing_segnet_class_logit_or_label_surface_in_cache",
            "boundary": "blocked_missing_boundary_surface_in_cache",
        },
        "blockers": [
            "mlx_local_response_is_advisory_not_score_authority",
            *(
                []
                if full_video_executed
                else ["full_video_mlx_response_not_executed"]
            ),
            *(
                ["local_acquisition_partial_raw_not_full_video"]
                if local_acquisition_partial_raw
                else []
            ),
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


def _baseline_reuse_report(baseline_reuse: BaselineReuse | None) -> dict[str, Any]:
    if baseline_reuse is None:
        return {
            "schema": "hprc_baseline_mlx_response_reuse_report.v1",
            "enabled": False,
            "score_claim": False,
            "promotion_eligible": False,
        }
    return {
        "schema": "hprc_baseline_mlx_response_reuse_report.v1",
        "enabled": True,
        "source_profile": str(baseline_reuse.source_profile_path),
        "source_response": str(baseline_reuse.source_response_path),
        "source_cache_report": str(baseline_reuse.source_cache_report_path),
        "source_components_dir": (
            None
            if baseline_reuse.source_components_dir is None
            else str(baseline_reuse.source_components_dir)
        ),
        "source_cache_dir": (
            None
            if baseline_reuse.source_cache_dir is None
            else str(baseline_reuse.source_cache_dir)
        ),
        "validation": {
            "hprc_0bin_sha256_matched": True,
            "reference_cache_dir_matched": True,
            "max_pairs_matched": True,
            "component_artifacts_present": True,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _load_cache_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"cache report must be a JSON object: {path}")
    return payload


def _load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _full_video_mlx_response_executed(*, cache_report: dict[str, Any], max_pairs: int) -> bool:
    return (
        int(cache_report.get("raw_pair_count") or 0) >= 600
        and int(cache_report.get("cached_pair_count") or 0) >= 600
        and int(max_pairs) >= 600
    )


def _local_acquisition_partial_raw_blocker(cache_report: dict[str, Any]) -> bool:
    return (
        bool(cache_report.get("local_acquisition_partial_raw"))
        and int(cache_report.get("raw_pair_count") or 0) < 600
    )


def _section_value_row(
    *,
    baseline: dict[str, Any],
    baseline_variant: VariantSpec,
    variant: VariantSpec,
    payload: dict[str, Any],
) -> dict[str, Any]:
    delta_pose = float(payload["avg_posenet_dist"]) - float(baseline["avg_posenet_dist"])
    delta_seg = float(payload["avg_segnet_dist"]) - float(baseline["avg_segnet_dist"])
    baseline_nonrate = _nonrate_score(baseline)
    variant_nonrate = _nonrate_score(payload)
    delta_nonrate = variant_nonrate - baseline_nonrate
    archive_bytes_removed = int(baseline_variant.archive_bytes) - int(variant.archive_bytes)
    rate_score_delta = float(payload["score_rate_contribution"]) - float(
        baseline["score_rate_contribution"]
    )
    total_score_delta = float(payload["canonical_score"]) - float(baseline["canonical_score"])
    section = variant.neutralized_section or "none"
    rate_price = 25.0 / ORIGINAL_VIDEO_BYTES
    return {
        "variant_id": variant.variant_id,
        "neutralized_section": section,
        "archive_zip_bytes": int(variant.archive_bytes),
        "archive_bytes_removed_vs_baseline": int(archive_bytes_removed),
        "delta_avg_posenet_dist": delta_pose,
        "delta_avg_segnet_dist": delta_seg,
        "delta_nonrate_score": delta_nonrate,
        "delta_rate_score": rate_score_delta,
        "delta_total_mlx_score_advisory": total_score_delta,
        "nonrate_score_value_per_removed_archive_kib": (
            delta_nonrate / max(archive_bytes_removed / 1024.0, 1.0e-9)
            if archive_bytes_removed > 0
            else None
        ),
        "rate_price_score_per_kib": rate_price * 1024.0,
        "marginal_status": _marginal_status(delta_nonrate, archive_bytes_removed),
        **FALSE_AUTHORITY,
    }


def _pair_rows(
    *,
    baseline_payload: dict[str, Any],
    variants: list[VariantSpec],
    payloads: dict[str, dict[str, Any]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    baseline_pose, baseline_seg = _component_arrays(baseline_payload)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        payload = payloads[variant.variant_id]
        pose, seg = _component_arrays(payload)
        pair_window = payload.get("source_pair_window", [[0, 1], [0, 1]])
        for idx in range(min(len(pose), len(seg), max_pairs)):
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "neutralized_section": variant.neutralized_section,
                    "pair_row": idx,
                    "frame_indices": _frame_indices_for_row(pair_window, idx),
                    "delta_posenet_dist": float(pose[idx] - baseline_pose[idx]),
                    "delta_segnet_dist": float(seg[idx] - baseline_seg[idx]),
                    "delta_nonrate_score_pair_local": (
                        _nonrate_from_components(
                            seg=float(seg[idx]),
                            pose=float(pose[idx]),
                        )
                        - _nonrate_from_components(
                            seg=float(baseline_seg[idx]),
                            pose=float(baseline_pose[idx]),
                        )
                    ),
                    **FALSE_AUTHORITY,
                }
            )
    return rows


def _component_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifacts = payload.get("components", {}).get("artifacts", {})
    pose_path = artifacts.get("posenet_distortion", {}).get("path")
    seg_path = artifacts.get("segnet_distortion", {}).get("path")
    if not pose_path or not seg_path:
        raise SystemExit("MLX response did not emit component arrays")
    return np.load(pose_path).astype(np.float32), np.load(seg_path).astype(np.float32)


def _frame_indices_for_row(pair_window: Any, idx: int) -> list[int]:
    if isinstance(pair_window, list) and len(pair_window) == 2:
        start = pair_window[0]
        if isinstance(start, list) and len(start) == 2:
            return [int(start[0]) + 2 * idx, int(start[0]) + 2 * idx + 1]
    return [2 * idx, 2 * idx + 1]


def _nonrate_score(payload: dict[str, Any]) -> float:
    return _nonrate_from_components(
        seg=float(payload["avg_segnet_dist"]),
        pose=float(payload["avg_posenet_dist"]),
    )


def _nonrate_from_components(*, seg: float, pose: float) -> float:
    return contest_formula_score(seg_dist=seg, pose_dist=pose, archive_bytes=0)


def _marginal_status(delta_nonrate: float, archive_bytes_removed: int) -> str:
    if archive_bytes_removed <= 0:
        return "no_archive_byte_savings"
    paid_rate_score = archive_bytes_removed * (25.0 / ORIGINAL_VIDEO_BYTES)
    if delta_nonrate <= 0.0:
        return "cut_candidate_distortion_nonworse"
    if delta_nonrate < paid_rate_score:
        return "cut_candidate_value_below_rate_price"
    return "protect_candidate_value_exceeds_rate_price"


def _build_shrink_backlog(report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in report.get("section_value_rows", []):
        if row.get("neutralized_section") in {None, "none"}:
            continue
        rows.append(
            {
                "family": "hprc_compact_receiver",
                "stage": "pre_entropy_residual_or_receiver_section",
                "scope": "section",
                "section": row.get("neutralized_section"),
                "variant_id": row.get("variant_id"),
                "marginal_status": row.get("marginal_status"),
                "archive_bytes_removed_vs_baseline": row.get(
                    "archive_bytes_removed_vs_baseline"
                ),
                "delta_nonrate_score": row.get("delta_nonrate_score"),
                "delta_total_mlx_score_advisory": row.get("delta_total_mlx_score_advisory"),
                "next_materializer_task": _next_materializer_task(row),
                **FALSE_AUTHORITY,
            }
        )
    rows.sort(
        key=lambda item: (
            _marginal_priority(str(item["marginal_status"])),
            str(item["section"]) != "residual_rc",
            -int(item.get("archive_bytes_removed_vs_baseline") or 0),
        )
    )
    pair_scoped_candidates = _build_pair_scoped_residual_candidates(report)
    return {
        "schema": BACKLOG_SCHEMA,
        "source_profile_schema": report.get("schema"),
        "source_scope_status": report.get("scope_status"),
        "hard_archive_byte_ceiling": report.get("archive_byte_ceiling"),
        "rows": rows,
        "pair_scoped_residual_candidate_rows": pair_scoped_candidates,
        "allocator_rule": (
            "cut or recode bytes whose MLX/exact nonrate marginal is below "
            "25/original_video_bytes; promote only after receiver proof plus exact CPU/CUDA gate"
        ),
        "priority_note": (
            "residual_rc remains the first production shrink target unless scorer value exceeds "
            "the rate price on full-video exact-reduced replay"
        ),
        **FALSE_AUTHORITY,
    }


def _build_pair_scoped_residual_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    rate_price = float(
        report.get("archive_byte_ceiling", {}).get("rate_price_score_per_byte")
        or 25.0 / ORIGINAL_VIDEO_BYTES
    )
    section_rows = {
        str(row.get("variant_id")): row
        for row in report.get("section_value_rows", [])
        if row.get("variant_id")
    }
    pairs_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("pair_value_rows", []):
        variant_id = str(row.get("variant_id"))
        if not variant_id.startswith("residual_transform_threshold_abs_le_"):
            continue
        pairs_by_variant.setdefault(variant_id, []).append(row)

    out: list[dict[str, Any]] = []
    for variant_id, pair_rows in pairs_by_variant.items():
        threshold = _threshold_from_threshold_variant_id(variant_id)
        section_row = section_rows.get(variant_id)
        if threshold is None or section_row is None or not pair_rows:
            continue
        bytes_removed = int(section_row.get("archive_bytes_removed_vs_baseline") or 0)
        if bytes_removed <= 0:
            continue
        pair_rows = sorted(pair_rows, key=lambda row: int(row.get("pair_row") or 0))
        per_pair_bytes = bytes_removed / max(len(pair_rows), 1)
        per_pair_rate_price = per_pair_bytes * rate_price
        selected = [
            row
            for row in pair_rows
            if float(row.get("delta_nonrate_score_pair_local") or 0.0)
            <= per_pair_rate_price
        ]
        if not selected:
            continue
        selected_indices = sorted(int(row["pair_row"]) for row in selected)
        pair_ranges = _compress_int_ranges(selected_indices)
        transform = (
            f"threshold_abs_le_pairs={threshold}@{_format_int_ranges(pair_ranges)}"
        )
        estimated_bytes_removed = per_pair_bytes * len(selected)
        estimated_nonrate_delta = sum(
            float(row.get("delta_nonrate_score_pair_local") or 0.0)
            for row in selected
        )
        out.append(
            {
                "family": "hprc_compact_receiver",
                "stage": "pre_entropy_residual_tokens",
                "scope": "pair",
                "source_variant_id": variant_id,
                "residual_transform": transform,
                "threshold_abs_le": int(threshold),
                "selected_pair_count": len(selected),
                "protected_pair_count": len(pair_rows) - len(selected),
                "pair_ranges": [[int(start), int(end)] for start, end in pair_ranges],
                "estimated_archive_bytes_removed_vs_baseline": round(estimated_bytes_removed),
                "estimated_delta_nonrate_pair_local_sum": float(estimated_nonrate_delta),
                "estimated_delta_rate_score": float(-estimated_bytes_removed * rate_price),
                "selection_rule": (
                    "apply residual shrink only where pair-local MLX nonrate delta "
                    "is <= the uniform per-pair archive-rate price"
                ),
                "next_materializer_task": (
                    "rerun profile_hprc_mlx_component_neutralization.py with "
                    f"--residual-transforms {transform} and full-video direct HPRC cache"
                ),
                **FALSE_AUTHORITY,
            }
        )
    out.sort(
        key=lambda row: (
            -int(row["estimated_archive_bytes_removed_vs_baseline"]),
            int(row["protected_pair_count"]),
            str(row["source_variant_id"]),
        )
    )
    return out


def _threshold_from_threshold_variant_id(variant_id: str) -> int | None:
    prefix = "residual_transform_threshold_abs_le_"
    if not variant_id.startswith(prefix):
        return None
    try:
        return int(variant_id[len(prefix) :])
    except ValueError:
        return None


def _compress_int_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_unique = sorted({int(idx) for idx in indices})
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_unique[0]
    for idx in sorted_unique[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def _format_int_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def _marginal_priority(status: str) -> int:
    order = {
        "cut_candidate_distortion_nonworse": 0,
        "cut_candidate_value_below_rate_price": 1,
        "protect_candidate_value_exceeds_rate_price": 2,
        "no_archive_byte_savings": 3,
    }
    return order.get(status, 4)


def _next_materializer_task(row: dict[str, Any]) -> str:
    section = str(row.get("neutralized_section"))
    if section == "residual_rc":
        return "replace_raw_residual_grid_with_scorer_ranked_significance_and_learned_prior_coder"
    if section == "latents_rc":
        return "delta_code_latents_and_range_code_symbols_under_same_lambda"
    if section == "selectors_rc":
        return "sweep_selector_semantics_and_symbol_coder_before_cutting"
    if section == "decoder_qw":
        return "sweep_decoder_grid_basis_and_weight_quantization_under_receiver_proof"
    if section == "receiver_state":
        return "remove_or_recode_state_if_full_video_scorer_value_stays_below_rate_price"
    return "inspect_section_specific_materializer"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
