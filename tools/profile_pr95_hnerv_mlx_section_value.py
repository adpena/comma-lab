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
    parser.add_argument(
        "--baseline-cache-dir",
        type=Path,
        help=(
            "Optional trusted baseline candidate cache to reuse. Requires "
            "--baseline-cache-report and --baseline-mlx-response; archive "
            "bytes/SHA, pair count, scorer batch shape, and false-authority "
            "flags are validated before reuse."
        ),
    )
    parser.add_argument("--baseline-cache-report", type=Path)
    parser.add_argument("--baseline-mlx-response", type=Path)
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
    baseline_reuse_paths = _resolve_baseline_reuse_paths(args=args, base=repo_root)
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
    cache_rows: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    baseline_reuse: dict[str, Any] | None = None
    variants_to_execute = variants
    if baseline_reuse_paths is not None:
        baseline_variant = variants[0]
        baseline_reuse = _load_external_baseline_reuse(
            baseline_variant=baseline_variant,
            cache_dir=baseline_reuse_paths["cache_dir"],
            cache_report=baseline_reuse_paths["cache_report"],
            mlx_response=baseline_reuse_paths["mlx_response"],
            output_dir=output_dir,
            max_pairs=int(args.max_pairs),
            scorer_batch_pairs=int(args.scorer_batch_pairs),
        )
        cache_rows["baseline"] = baseline_reuse["cache_row"]
        payloads["baseline"] = baseline_reuse["payload"]
        variants_to_execute = [variant for variant in variants if variant.variant_id != "baseline"]
    if variants_to_execute:
        cache_rows.update(
            _materialize_caches(
                variants=variants_to_execute,
                output_dir=output_dir,
                repo_root=repo_root,
                upstream_dir=upstream_dir,
                video_names_file=video_names_file,
                max_pairs=int(args.max_pairs),
                inflate_timeout=int(args.inflate_timeout),
                allow_large_tensor_cache=bool(args.allow_large_tensor_cache),
            )
        )
        payloads.update(
            _run_mlx_responses(
                variants=variants_to_execute,
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
    if baseline_reuse is not None:
        report["baseline_reuse"] = baseline_reuse["metadata"]
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


def _resolve_baseline_reuse_paths(
    *,
    args: argparse.Namespace,
    base: Path,
) -> dict[str, Path] | None:
    supplied = {
        "cache_dir": args.baseline_cache_dir,
        "cache_report": args.baseline_cache_report,
        "mlx_response": args.baseline_mlx_response,
    }
    if not any(value is not None for value in supplied.values()):
        return None
    missing = [name for name, value in supplied.items() if value is None]
    if missing:
        raise SystemExit(
            "baseline reuse requires --baseline-cache-dir, "
            "--baseline-cache-report, and --baseline-mlx-response together; "
            f"missing={missing}"
        )
    return {name: _resolve(Path(value), base=base) for name, value in supplied.items()}


def _load_external_baseline_reuse(
    *,
    baseline_variant: VariantSpec,
    cache_dir: Path,
    cache_report: Path,
    mlx_response: Path,
    output_dir: Path,
    max_pairs: int,
    scorer_batch_pairs: int,
) -> dict[str, Any]:
    manifest = cache_dir / "manifest.json"
    for label, path in (
        ("baseline cache dir", cache_dir),
        ("baseline cache manifest", manifest),
        ("baseline cache report", cache_report),
        ("baseline MLX response", mlx_response),
    ):
        if label == "baseline cache dir":
            if not path.is_dir():
                raise ValueError(f"{label} is missing: {path}")
        elif not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    cache_report_payload = json.loads(cache_report.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    response_payload = json.loads(mlx_response.read_text(encoding="utf-8"))
    _validate_external_baseline_cache(
        baseline_variant=baseline_variant,
        cache_dir=cache_dir,
        manifest=manifest,
        cache_report=cache_report,
        cache_report_payload=cache_report_payload,
        manifest_payload=manifest_payload,
        max_pairs=max_pairs,
    )
    _validate_external_baseline_response(
        baseline_variant=baseline_variant,
        response_path=mlx_response,
        response_payload=response_payload,
        max_pairs=max_pairs,
        scorer_batch_pairs=scorer_batch_pairs,
    )
    write_json(output_dir / "mlx_responses" / "baseline.json", response_payload)
    return {
        "cache_row": {
            "cache_dir": cache_dir.as_posix(),
            "work_dir": None,
            "report_output": cache_report.as_posix(),
            "argv": [],
            "upstream_dir": None,
            "video_names_file": None,
            "stdout": "",
            "stderr_tail": "",
            "reused_external_baseline_cache_report": True,
            "reuse_integrity": {
                "archive_bytes": int(baseline_variant.archive_bytes),
                "archive_sha256": baseline_variant.archive_sha256,
                "cache_manifest": manifest.as_posix(),
                "cache_report": cache_report.as_posix(),
                "mlx_response": mlx_response.as_posix(),
            },
        },
        "payload": response_payload,
        "metadata": {
            "schema": "pr95_hnerv_external_baseline_reuse.v1",
            "status": "accepted_archive_hash_and_pair_shape_match",
            "source_cache_dir": cache_dir.as_posix(),
            "source_cache_manifest": manifest.as_posix(),
            "source_cache_report": cache_report.as_posix(),
            "source_mlx_response": mlx_response.as_posix(),
            "local_mlx_response_copy": (
                output_dir / "mlx_responses" / "baseline.json"
            ).as_posix(),
            "archive_bytes": int(baseline_variant.archive_bytes),
            "archive_sha256": baseline_variant.archive_sha256,
            "max_pairs": int(max_pairs),
            "scorer_batch_pairs": int(scorer_batch_pairs),
            **FALSE_AUTHORITY,
        },
    }


def _validate_external_baseline_cache(
    *,
    baseline_variant: VariantSpec,
    cache_dir: Path,
    manifest: Path,
    cache_report: Path,
    cache_report_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    max_pairs: int,
) -> None:
    archive = cache_report_payload.get("archive")
    if not isinstance(archive, dict):
        raise ValueError(f"baseline cache report lacks archive row: {cache_report}")
    _expect_archive_identity(
        label="baseline cache report",
        archive_bytes=archive.get("bytes"),
        archive_sha256=archive.get("sha256"),
        baseline_variant=baseline_variant,
    )
    _expect_archive_identity(
        label="baseline cache manifest",
        archive_bytes=cache_report_payload.get("archive", {}).get("bytes"),
        archive_sha256=manifest_payload.get("archive_sha256"),
        baseline_variant=baseline_variant,
    )
    reported_manifest = cache_report_payload.get("cache_manifest")
    if reported_manifest is not None and Path(str(reported_manifest)).resolve() != manifest.resolve():
        raise ValueError(
            "baseline cache report points at a different manifest: "
            f"{reported_manifest} != {manifest}"
        )
    if int(manifest_payload.get("pair_count", -1)) < int(max_pairs):
        raise ValueError(
            f"baseline cache has {manifest_payload.get('pair_count')} pairs; "
            f"need at least {max_pairs}"
        )
    for label, payload in (
        ("baseline cache report", cache_report_payload),
        ("baseline cache manifest", manifest_payload),
    ):
        _require_false_authority(label=label, payload=payload)
    if Path(str(cache_report_payload.get("output_cache_dir", cache_dir))).resolve() != cache_dir.resolve():
        raise ValueError(
            "baseline cache report output_cache_dir does not match "
            f"--baseline-cache-dir: {cache_report_payload.get('output_cache_dir')}"
        )


def _validate_external_baseline_response(
    *,
    baseline_variant: VariantSpec,
    response_path: Path,
    response_payload: dict[str, Any],
    max_pairs: int,
    scorer_batch_pairs: int,
) -> None:
    _expect_archive_identity(
        label="baseline MLX response",
        archive_bytes=response_payload.get("archive_size_bytes"),
        archive_sha256=response_payload.get("archive_sha256"),
        baseline_variant=baseline_variant,
    )
    if int(response_payload.get("n_samples", -1)) < int(max_pairs):
        raise ValueError(
            f"baseline MLX response has {response_payload.get('n_samples')} samples; "
            f"need at least {max_pairs}: {response_path}"
        )
    if int(response_payload.get("candidate_cache_pairs", -1)) < int(max_pairs):
        raise ValueError(
            "baseline MLX response candidate cache is not full enough: "
            f"{response_payload.get('candidate_cache_pairs')} < {max_pairs}"
        )
    if int(response_payload.get("reference_cache_pairs", -1)) < int(max_pairs):
        raise ValueError(
            "baseline MLX response reference cache is not full enough: "
            f"{response_payload.get('reference_cache_pairs')} < {max_pairs}"
        )
    if int(response_payload.get("batch_pairs", -1)) != int(scorer_batch_pairs):
        raise ValueError(
            "baseline MLX response batch shape mismatch: "
            f"{response_payload.get('batch_pairs')} != {scorer_batch_pairs}"
        )
    _require_false_authority(label="baseline MLX response", payload=response_payload)


def _expect_archive_identity(
    *,
    label: str,
    archive_bytes: Any,
    archive_sha256: Any,
    baseline_variant: VariantSpec,
) -> None:
    if int(archive_bytes or -1) != int(baseline_variant.archive_bytes):
        raise ValueError(
            f"{label} archive bytes mismatch: "
            f"{archive_bytes} != {baseline_variant.archive_bytes}"
        )
    if str(archive_sha256 or "") != str(baseline_variant.archive_sha256):
        raise ValueError(
            f"{label} archive sha256 mismatch: "
            f"{archive_sha256} != {baseline_variant.archive_sha256}"
        )


def _require_false_authority(*, label: str, payload: dict[str, Any]) -> None:
    for key in (
        "score_claim",
        "ready_for_exact_eval_dispatch",
        "promotable",
        "promotion_eligible",
        "rank_or_kill_eligible",
    ):
        if payload.get(key) is True:
            raise ValueError(f"{label} cannot be reused with {key}=true")


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
