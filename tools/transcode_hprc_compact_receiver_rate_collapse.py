#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize HPRC compact-receiver rate-collapse variants.

This is a queue-owned pre-replay step: losslessly entropy-wrap charged compact
receiver sections, measure byte-closed archive.zip bytes, then emit one best
archive export for downstream rate gating and replay. It never claims score.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.repo_io import ArtifactWriteError, sha256_bytes, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive import parse_hprc_packet  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY, export_hprc_archive_bytes  # noqa: E402
from tac.substrates.hprc.learned_receiver import decode_compact_receiver_packet  # noqa: E402
from tac.substrates.hprc.rate_collapse import (  # noqa: E402
    DEFAULT_IMPORTANCE_PROTECTED_RESIDUAL_TOKEN_COLLAPSE_SPEC,
    DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS,
    HPRC_RATE_COLLAPSE_REPORT_SCHEMA,
    parse_rate_collapse_sections,
    parse_residual_token_collapse_specs,
    rate_collapse_variant_groups,
    transcode_compact_receiver_importance_weighted_residual_tokens,
    transcode_compact_receiver_residual_tokens,
    transcode_compact_receiver_sections,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-result", type=Path)
    source.add_argument("--exact-bridge", type=Path)
    source.add_argument("--source-archive", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--sections",
        action="append",
        default=[],
        help="Comma/space separated section tokens. Default: compact semantic payload sections.",
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--target-rate-term",
        type=float,
        help=(
            "When lossy residual candidates are enabled, select the least-damaging "
            "candidate whose byte-closed archive rate term is below this target."
        ),
    )
    parser.add_argument(
        "--residual-collapse-schedule",
        action="append",
        default=[],
        help=(
            "Residual token collapse schedule entries like dz4_qd2 or 4:2. "
            f"Default: {','.join(DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS)}."
        ),
    )
    parser.add_argument(
        "--disable-lossy-residual-collapse",
        action="store_true",
        help="Only run lossless section entropy wrapping.",
    )
    parser.add_argument(
        "--enable-lossy-residual-collapse",
        action="store_true",
        help=(
            "Also materialize lossy residual-token candidates. This is implied "
            "by --target-rate-term and remains false-authority until replayed."
        ),
    )
    parser.add_argument(
        "--residual-importance-npy",
        type=Path,
        help=(
            "Optional residual-token importance surface. Low values are coarsened; "
            "high values use --importance-protected-spec. Shape must broadcast to "
            "frames x residual_grid_h x residual_grid_w x channels."
        ),
    )
    parser.add_argument(
        "--p19-posenet-null-pairs",
        type=Path,
        help=(
            "Optional P19 artifact. Selected PoseNet-null pairs are treated as "
            "lower-importance residual-token coarsening candidates."
        ),
    )
    parser.add_argument(
        "--p18-segnet-region-waterfill",
        type=Path,
        help=(
            "Optional P18 artifact. Listed SegNet-vulnerable regions are protected "
            "inside any P19-null pair before residual-token coarsening."
        ),
    )
    parser.add_argument("--importance-coarsen-quantile", type=float, default=0.10)
    parser.add_argument(
        "--importance-selection-domain",
        choices=("global_weighted", "eligible_low"),
        default="global_weighted",
        help=(
            "global_weighted ranks every residual token by the supplied importance "
            "surface; eligible_low only coarsens the explicit low-importance mask "
            "emitted by structured P18/P19 artifacts."
        ),
    )
    parser.add_argument(
        "--importance-protected-spec",
        default=DEFAULT_IMPORTANCE_PROTECTED_RESIDUAL_TOKEN_COLLAPSE_SPEC,
        help=(
            "Residual collapse spec used for high-importance tokens. Default is "
            f"{DEFAULT_IMPORTANCE_PROTECTED_RESIDUAL_TOKEN_COLLAPSE_SPEC}."
        ),
    )
    parser.add_argument("--skip-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--out-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        output_dir / "hprc_rate_collapse_report.json"
        if args.out_json is None
        else _resolve(args.out_json, repo_root=repo_root)
    )

    source = _load_source(args, repo_root=repo_root)
    source_archive = source["archive_path"]
    source_archive_bytes = int(source["archive_bytes"])
    source_archive_sha256 = str(source["archive_sha256"])
    source_packet = bytes(source["packet_bytes"])
    source_packet_sha256 = str(source["packet_sha256"])
    source_packet_path = str(source["packet_path"])

    sections = parse_rate_collapse_sections(args.sections)
    residual_importance_requested = (
        args.residual_importance_npy is not None
        or args.p19_posenet_null_pairs is not None
        or args.p18_segnet_region_waterfill is not None
    )
    lossy_residual_collapse_enabled = (
        (
            bool(args.enable_lossy_residual_collapse)
            or args.target_rate_term is not None
            or bool(args.residual_collapse_schedule)
            or residual_importance_requested
        )
        and not bool(args.disable_lossy_residual_collapse)
    )
    if residual_importance_requested and not lossy_residual_collapse_enabled:
        raise ValueError("residual importance inputs require lossy residual collapse to be enabled")
    importance_payload = (
        _load_residual_importance_payload(args, source_packet, repo_root=repo_root)
        if lossy_residual_collapse_enabled and residual_importance_requested
        else None
    )
    variants: list[dict[str, Any]] = [
        {
            "variant_id": "baseline",
            "sections": [],
            "archive_zip_path": source_archive.as_posix(),
            "archive_zip_sha256": source_archive_sha256,
            "archive_zip_bytes": int(source_archive_bytes),
            "archive_rate_term": contest_rate_term(int(source_archive_bytes)),
            "hprc_0bin_path": source_packet_path,
            "hprc_0bin_sha256": source_packet_sha256,
            "hprc_0bin_bytes": len(source_packet),
            "section_rows": [],
            "lossy_residual_token_collapse": False,
            "residual_token_collapse": None,
            "export_kind": "source_training_export",
            **FALSE_AUTHORITY,
        }
    ]
    packet_by_variant: dict[str, bytes] = {"baseline": source_packet}
    for variant_id, group in rate_collapse_variant_groups(sections):
        packet, section_rows = transcode_compact_receiver_sections(
            source_packet,
            sections=group,
            brotli_quality=int(args.brotli_quality),
        )
        packet_by_variant[variant_id] = packet
        variant_dir = output_dir / "variants" / variant_id
        archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
            packet,
            variant_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=False,
            retain_receiver_proof_output=False,
        )
        bin_path = variant_dir / "0.bin"
        variants.append(
            {
                "variant_id": variant_id,
                "sections": [kind.name.lower() for kind in group],
                "archive_zip_path": archive_path.as_posix(),
                "archive_zip_sha256": archive_sha,
                "archive_zip_bytes": int(archive_bytes),
                "archive_rate_term": contest_rate_term(int(archive_bytes)),
                "hprc_0bin_path": bin_path.as_posix(),
                "hprc_0bin_sha256": sha256_file(bin_path),
                "hprc_0bin_bytes": len(packet),
                "section_rows": section_rows,
                "lossy_residual_token_collapse": False,
                "residual_token_collapse": None,
                "export_kind": "rate_collapse_probe_export_without_receiver_proof",
                **FALSE_AUTHORITY,
            }
        )

    if lossy_residual_collapse_enabled:
        residual_specs = parse_residual_token_collapse_specs(
            args.residual_collapse_schedule or None
        )
        for spec in residual_specs:
            packet, section_rows, metrics = transcode_compact_receiver_residual_tokens(
                source_packet,
                spec=spec,
                sections=sections,
                brotli_quality=int(args.brotli_quality),
            )
            variant_id = spec.variant_id
            packet_by_variant[variant_id] = packet
            variant_dir = output_dir / "variants" / variant_id
            archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
                packet,
                variant_dir,
                repo_root=repo_root,
                emit_archive_bound_candidate_package=False,
                retain_receiver_proof_output=False,
            )
            bin_path = variant_dir / "0.bin"
            variants.append(
                {
                    "variant_id": variant_id,
                    "sections": [kind.name.lower() for kind in sections],
                    "archive_zip_path": archive_path.as_posix(),
                    "archive_zip_sha256": archive_sha,
                    "archive_zip_bytes": int(archive_bytes),
                    "archive_rate_term": contest_rate_term(int(archive_bytes)),
                    "hprc_0bin_path": bin_path.as_posix(),
                    "hprc_0bin_sha256": sha256_file(bin_path),
                    "hprc_0bin_bytes": len(packet),
                    "section_rows": section_rows,
                    "lossy_residual_token_collapse": True,
                    "residual_token_collapse": metrics,
                    "export_kind": "residual_token_rate_collapse_probe_export_without_receiver_proof",
                    **FALSE_AUTHORITY,
                }
            )
        if importance_payload is not None:
            high_spec = parse_residual_token_collapse_specs(args.importance_protected_spec)[0]
            for low_spec in residual_specs:
                packet, section_rows, metrics = (
                    transcode_compact_receiver_importance_weighted_residual_tokens(
                        source_packet,
                        low_importance_spec=low_spec,
                        high_importance_spec=high_spec,
                        importance=importance_payload["importance"],
                        coarsen_quantile=float(args.importance_coarsen_quantile),
                        eligible_mask=importance_payload.get("eligible_mask"),
                        selection_domain=str(args.importance_selection_domain),
                        sections=sections,
                        brotli_quality=int(args.brotli_quality),
                    )
                )
                variant_id = str(metrics["variant_id"])
                packet_by_variant[variant_id] = packet
                variant_dir = output_dir / "variants" / variant_id
                archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
                    packet,
                    variant_dir,
                    repo_root=repo_root,
                    emit_archive_bound_candidate_package=False,
                    retain_receiver_proof_output=False,
                )
                bin_path = variant_dir / "0.bin"
                variants.append(
                    {
                        "variant_id": variant_id,
                        "sections": [kind.name.lower() for kind in sections],
                        "archive_zip_path": archive_path.as_posix(),
                        "archive_zip_sha256": archive_sha,
                        "archive_zip_bytes": int(archive_bytes),
                        "archive_rate_term": contest_rate_term(int(archive_bytes)),
                        "hprc_0bin_path": bin_path.as_posix(),
                        "hprc_0bin_sha256": sha256_file(bin_path),
                        "hprc_0bin_bytes": len(packet),
                        "section_rows": section_rows,
                        "lossy_residual_token_collapse": True,
                        "importance_weighted_residual_token_collapse": True,
                        "residual_token_collapse": metrics,
                        "residual_importance_source": importance_payload["source"],
                        "residual_importance_selection_domain": str(
                            args.importance_selection_domain
                        ),
                        "export_kind": (
                            "importance_weighted_residual_token_rate_collapse_probe_"
                            "export_without_receiver_proof"
                        ),
                        **FALSE_AUTHORITY,
                    }
                )

    best, selection = _select_best_variant(
        variants,
        target_rate_term=args.target_rate_term,
    )
    best_is_lossy = bool(best.get("lossy_residual_token_collapse"))
    residual_transform = _residual_transform_for_variant(best)
    losslessness_kind = (
        "lossy_residual_token_collapse"
        if best_is_lossy
        else "lossless_section_entropy_transcode"
    )
    best_packet = packet_by_variant[str(best["variant_id"])]
    best_export_dir = output_dir / "best_archive_export"
    best_archive_path, best_archive_sha, best_archive_bytes = export_hprc_archive_bytes(
        best_packet,
        best_export_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=not bool(args.skip_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_output),
    )
    receiver_proof_path = best_export_dir / "receiver_proof" / "hprc_receiver_proof.json"
    adapter_package_path = best_export_dir / "archive_bound_candidate_adapter_package.json"
    best_delta = int(best_archive_bytes) - int(source_archive_bytes)
    cleanup_manifest_path = output_dir / "hprc_rate_collapse_cleanup_manifest.json"
    cleanup_manifest = _build_cleanup_manifest(
        output_dir=output_dir,
        report_path=report_path,
        variants=variants,
        best_variant_id=str(best["variant_id"]),
        best_export_dir=best_export_dir,
    )
    _write_json(cleanup_manifest_path, cleanup_manifest, allow_overwrite=bool(args.allow_overwrite))
    report = {
        "schema": HPRC_RATE_COLLAPSE_REPORT_SCHEMA,
        "source_input": {
            "kind": source["kind"],
            "path": source["input_path"].as_posix(),
            "schema": source.get("input_schema"),
            "custody_verified": bool(source.get("custody_verified")),
        },
        "training_result_path": (
            source["input_path"].as_posix() if source["kind"] == "training_result" else None
        ),
        "output_dir": output_dir.as_posix(),
        "source": {
            "archive_zip_path": source_archive.as_posix(),
            "archive_zip_sha256": source_archive_sha256,
            "archive_zip_bytes": int(source_archive_bytes),
            "hprc_0bin_path": source_packet_path,
            "hprc_0bin_sha256": source_packet_sha256,
            "hprc_0bin_bytes": len(source_packet),
        },
        "variant_count": len(variants),
        "variants": variants,
        "lossy_residual_collapse_enabled": lossy_residual_collapse_enabled,
        "residual_importance_enabled": importance_payload is not None,
        "residual_importance_source": (
            importance_payload["source"] if importance_payload is not None else None
        ),
        "residual_importance_selection_domain": (
            str(args.importance_selection_domain) if importance_payload is not None else None
        ),
        "best_variant_id": best["variant_id"],
        "best_variant_sections": best["sections"],
        "best_variant_selection": selection,
        "best_variant_lossy_residual_token_collapse": best_is_lossy,
        "best_variant_residual_transform": residual_transform,
        "best_variant_losslessness_kind": losslessness_kind,
        "rate_collapse_archive_bytes_delta": best_delta,
        "rate_collapse_archive_bytes_saved": max(0, -best_delta),
        "rate_collapse_win": best_delta < 0,
        "artifact": {
            "archive_path": best_archive_path.as_posix(),
            "archive_sha256": best_archive_sha,
            "archive_bytes": int(best_archive_bytes),
            "archive_rate_term": contest_rate_term(int(best_archive_bytes)),
            "hprc_0bin_path": (best_export_dir / "0.bin").as_posix(),
            "hprc_0bin_sha256": sha256_file(best_export_dir / "0.bin"),
            "receiver_proof_path": receiver_proof_path.as_posix(),
            "receiver_proof_present": receiver_proof_path.is_file(),
            "archive_bound_package_path": adapter_package_path.as_posix(),
            "archive_bound_package_present": adapter_package_path.is_file(),
            "rate_collapse_variant_id": best["variant_id"],
            "lossy_residual_token_collapse": best_is_lossy,
            "residual_token_collapse": best.get("residual_token_collapse"),
            "residual_transform": residual_transform,
            "losslessness_kind": losslessness_kind,
            **FALSE_AUTHORITY,
        },
        "receiver_proof_requested": not bool(args.skip_receiver_proof),
        "receiver_output_retained": bool(args.retain_receiver_output),
        "cleanup_manifest": {
            "path": cleanup_manifest_path.as_posix(),
            "sha256": sha256_file(cleanup_manifest_path),
            "variant_export_bytes": cleanup_manifest["variant_export_bytes"],
            "reclaimable_non_best_variant_bytes": cleanup_manifest[
                "reclaimable_non_best_variant_bytes"
            ],
        },
        "next_required_action": (
            "gate_archive_rate_before_local_replay"
            if best_export_dir.is_dir()
            else "repair_rate_collapse_export"
        ),
        **FALSE_AUTHORITY,
    }
    _write_json(report_path, report, allow_overwrite=bool(args.allow_overwrite))
    print(json.dumps({**report, "report_path": report_path.as_posix()}, sort_keys=True))
    return 0


def _build_cleanup_manifest(
    *,
    output_dir: Path,
    report_path: Path,
    variants: list[dict[str, Any]],
    best_variant_id: str,
    best_export_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    reclaimable = 0
    seen: set[Path] = set()
    for row in variants:
        variant_id = str(row.get("variant_id"))
        archive = Path(str(row.get("archive_zip_path") or ""))
        if not archive.is_absolute():
            archive = (output_dir / archive).resolve(strict=False)
        variant_dir = archive.parent if archive.name == "archive.zip" else archive
        if variant_dir in seen or not variant_dir.exists():
            continue
        seen.add(variant_dir)
        bytes_total = _tree_bytes(variant_dir)
        total += bytes_total
        is_best = variant_id == best_variant_id
        if not is_best:
            reclaimable += bytes_total
        rows.append(
            {
                "variant_id": variant_id,
                "path": variant_dir.as_posix(),
                "bytes": bytes_total,
                "retained": is_best,
                "disposition": "retain_best_export" if is_best else "certified_rebuildable_probe_export",
                "archive_zip_sha256": row.get("archive_zip_sha256"),
                "hprc_0bin_sha256": row.get("hprc_0bin_sha256"),
            }
        )
    best_bytes = _tree_bytes(best_export_dir) if best_export_dir.exists() else 0
    return {
        "schema": "hprc_rate_collapse_cleanup_manifest.v1",
        "output_dir": output_dir.as_posix(),
        "report_path": report_path.as_posix(),
        "best_variant_id": best_variant_id,
        "best_export_dir": best_export_dir.as_posix(),
        "best_export_bytes": best_bytes,
        "variant_export_bytes": total,
        "reclaimable_non_best_variant_bytes": reclaimable,
        "cleanup_policy": "retain_best_and_report; non-best variants are certified rebuildable by this report",
        "auto_delete_performed": False,
        "rows": rows,
        **FALSE_AUTHORITY,
    }


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                total += int(child.stat().st_size)
    return total


def _load_source(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    if args.training_result is not None:
        input_path = _resolve(args.training_result, repo_root=repo_root)
        training_result = _load_json_object(input_path)
        artifact = (
            training_result.get("artifact")
            if isinstance(training_result.get("artifact"), dict)
            else {}
        )
        archive_path = _artifact_path(artifact, "archive_path", repo_root=repo_root)
        expected_bytes = _positive_int(artifact.get("archive_bytes"))
        expected_sha = str(artifact.get("archive_sha256") or "")
        return _source_from_archive(
            archive_path,
            input_path=input_path,
            kind="training_result",
            input_schema=str(training_result.get("schema") or ""),
            expected_archive_bytes=expected_bytes,
            expected_archive_sha256=expected_sha,
            custody_verified=True,
        )
    if args.exact_bridge is not None:
        input_path = _resolve(args.exact_bridge, repo_root=repo_root)
        bridge = _load_json_object(input_path)
        archive = bridge.get("archive") if isinstance(bridge.get("archive"), dict) else {}
        archive_path = _artifact_path(archive, "path", repo_root=repo_root)
        archive_custody = (
            bridge.get("archive_custody")
            if isinstance(bridge.get("archive_custody"), dict)
            else {}
        )
        hprc_0bin_custody = (
            bridge.get("hprc_0bin_custody")
            if isinstance(bridge.get("hprc_0bin_custody"), dict)
            else {}
        )
        custody_verified = (
            bridge.get("ready_for_exact_eval_dispatch") is True
            and archive_custody.get("verified") is True
            and hprc_0bin_custody.get("verified") is True
        )
        if not custody_verified:
            raise ValueError(
                "exact bridge source is not custody-verified and exact-dispatch-ready"
            )
        return _source_from_archive(
            archive_path,
            input_path=input_path,
            kind="exact_bridge",
            input_schema=str(bridge.get("schema") or ""),
            expected_archive_bytes=_positive_int(archive.get("bytes")),
            expected_archive_sha256=str(archive.get("sha256") or ""),
            expected_packet_sha256=str(archive.get("hprc_0bin_sha256") or ""),
            custody_verified=True,
        )
    if args.source_archive is not None:
        input_path = _resolve(args.source_archive, repo_root=repo_root)
        return _source_from_archive(
            input_path,
            input_path=input_path,
            kind="source_archive",
            input_schema=None,
            custody_verified=False,
        )
    raise ValueError("one source is required")


def _source_from_archive(
    archive_path: Path,
    *,
    input_path: Path,
    kind: str,
    input_schema: str | None,
    expected_archive_bytes: int | None = None,
    expected_archive_sha256: str = "",
    expected_packet_sha256: str = "",
    custody_verified: bool,
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"source archive missing: {archive_path}")
    archive_bytes = int(archive_path.stat().st_size)
    archive_sha256 = sha256_file(archive_path)
    if expected_archive_bytes is not None and archive_bytes != int(expected_archive_bytes):
        raise ValueError(
            "source archive bytes mismatch: "
            f"expected={expected_archive_bytes} actual={archive_bytes}"
        )
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "source archive sha256 mismatch: "
            f"expected={expected_archive_sha256} actual={archive_sha256}"
        )
    packet_bytes, packet_sha, packet_path = _read_source_hprc_0bin(archive_path)
    if expected_packet_sha256 and packet_sha != expected_packet_sha256:
        raise ValueError(
            "source HPRC 0.bin sha256 mismatch: "
            f"expected={expected_packet_sha256} actual={packet_sha}"
        )
    return {
        "kind": kind,
        "input_path": input_path,
        "input_schema": input_schema,
        "archive_path": archive_path,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "packet_bytes": packet_bytes,
        "packet_sha256": packet_sha,
        "packet_path": packet_path,
        "custody_verified": custody_verified,
    }


def _read_source_hprc_0bin(source_archive: Path) -> tuple[bytes, str, str]:
    sibling = source_archive.parent / "0.bin"
    if sibling.is_file():
        payload = sibling.read_bytes()
        return payload, sha256_file(sibling), sibling.as_posix()
    with zipfile.ZipFile(source_archive) as archive:
        try:
            payload = archive.read("0.bin")
        except KeyError as exc:
            raise FileNotFoundError(
                f"source archive has no 0.bin and no sibling 0.bin: {source_archive}"
            ) from exc
    return payload, sha256_bytes(payload), f"{source_archive.as_posix()}::0.bin"


def _load_residual_importance_payload(
    args: argparse.Namespace,
    source_packet: bytes,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    sources = [
        args.residual_importance_npy is not None,
        args.p19_posenet_null_pairs is not None,
        args.p18_segnet_region_waterfill is not None,
    ]
    if not any(sources):
        return None
    if args.residual_importance_npy is not None and any(sources[1:]):
        raise ValueError(
            "--residual-importance-npy cannot be combined with P18/P19 artifact inputs"
        )
    if args.residual_importance_npy is not None:
        path = _resolve(args.residual_importance_npy, repo_root=repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"residual importance surface missing: {path}")
        importance = np.load(path, mmap_mode="r")
        return {
            "importance": np.asarray(importance, dtype=np.float32),
            "source": {
                "kind": "residual_importance_npy",
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": [int(dim) for dim in importance.shape],
            },
        }
    return _importance_from_p18_p19_artifacts(
        source_packet,
        p19_path=args.p19_posenet_null_pairs,
        p18_path=args.p18_segnet_region_waterfill,
        repo_root=repo_root,
    )


def _importance_from_p18_p19_artifacts(
    source_packet: bytes,
    *,
    p19_path: Path | None,
    p18_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    compact_packet = parse_hprc_packet(source_packet)
    compact = decode_compact_receiver_packet(compact_packet)
    q_shape = tuple(int(dim) for dim in compact.residual.q.shape)
    importance = np.ones(q_shape[:-1], dtype=np.float32)
    eligible_mask = np.ones(q_shape[:-1], dtype=bool) if p19_path is None else np.zeros(q_shape[:-1], dtype=bool)
    gop_size = max(1, int(compact_packet.config.gop_size))
    expected_pair_count = (q_shape[0] + gop_size - 1) // gop_size
    selected_pair_ids: set[int] = set()
    p19_record: dict[str, Any] | None = None
    p18_record: dict[str, Any] | None = None
    p18_protected_region_count = 0
    source_binding_blockers: list[str] = []

    if p19_path is not None:
        path = _resolve(p19_path, repo_root=repo_root)
        p19 = _load_json_object(path)
        if p19.get("schema") != "p19_posenet_null_pair_detection.v1":
            raise ValueError("P19 PoseNet-null artifact schema mismatch")
        if p19.get("n_pairs") is not None and int(p19["n_pairs"]) != expected_pair_count:
            raise ValueError("P19 PoseNet-null artifact n_pairs does not match HPRC packet")
        selected_pair_ids = {int(item) for item in p19.get("selected_pair_ids") or []}
        out_of_range = [
            pair_id
            for pair_id in selected_pair_ids
            if pair_id < 0 or pair_id >= expected_pair_count
        ]
        if out_of_range:
            raise ValueError("P19 PoseNet-null artifact contains out-of-range pair ids")
        for pair_id in selected_pair_ids:
            for frame_idx in _frames_for_pair(pair_id, frame_count=q_shape[0], gop_size=gop_size):
                importance[frame_idx, :, :] = 0.0
                eligible_mask[frame_idx, :, :] = True
        if not bool(eligible_mask.any()):
            raise ValueError("P19 PoseNet-null artifact selected no in-range HPRC frames")
        p19_record = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "selected_pair_count": len(selected_pair_ids),
            "source_archive": p19.get("source_archive"),
        }
        if isinstance(p19.get("source_archive"), dict):
            source_binding_blockers.append("p19_source_archive_is_proxy_not_hprc_source_archive")

    if p18_path is not None:
        path = _resolve(p18_path, repo_root=repo_root)
        p18 = _load_json_object(path)
        if p18.get("schema") != "p18_segnet_region_waterfill.v1":
            raise ValueError("P18 SegNet-region artifact schema mismatch")
        if (
            p18.get("n_pairs_available") is not None
            and int(p18["n_pairs_available"]) != expected_pair_count
        ):
            raise ValueError("P18 SegNet-region artifact n_pairs_available does not match HPRC packet")
        grid_h = q_shape[1]
        grid_w = q_shape[2]
        for row in p18.get("rows") or []:
            if not isinstance(row, dict):
                continue
            pair_id = int(row.get("pair_id", -1))
            if pair_id < 0:
                continue
            if pair_id >= expected_pair_count:
                raise ValueError("P18 SegNet-region artifact contains out-of-range pair ids")
            for region in row.get("regions256") or []:
                if not isinstance(region, dict) or not isinstance(region.get("box"), dict):
                    continue
                box = region["box"]
                x0 = _grid_start(float(box["x0"]), grid_w)
                x1 = _grid_end(float(box["x1"]), grid_w)
                y0 = _grid_start(float(box["y0"]), grid_h)
                y1 = _grid_end(float(box["y1"]), grid_h)
                if x1 <= x0 or y1 <= y0:
                    continue
                for frame_idx in _frames_for_pair(pair_id, frame_count=q_shape[0], gop_size=gop_size):
                    importance[frame_idx, y0:y1, x0:x1] = 2.0
                    eligible_mask[frame_idx, y0:y1, x0:x1] = False
                p18_protected_region_count += 1
        if (p18.get("rows") or []) and p18_protected_region_count <= 0:
            raise ValueError("P18 SegNet-region artifact mapped no protected HPRC residual cells")
        p18_record = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "protected_region_count": p18_protected_region_count,
            "source_p19_posenet_null_pairs": p18.get("source_p19_posenet_null_pairs"),
        }

    return {
        "importance": importance,
        "eligible_mask": eligible_mask,
        "source": {
            "kind": "p18_p19_scorer_region_artifacts",
            "residual_token_shape": list(q_shape),
            "importance_shape": [int(dim) for dim in importance.shape],
            "eligible_mask_shape": [int(dim) for dim in eligible_mask.shape],
            "eligible_cell_count": int(eligible_mask.sum()),
            "eligible_cell_fraction": float(eligible_mask.sum() / max(eligible_mask.size, 1)),
            "gop_size": gop_size,
            "expected_pair_count": expected_pair_count,
            "p19_posenet_null_pairs": p19_record,
            "p18_segnet_region_waterfill": p18_record,
            "source_binding_status": (
                "video_pair_count_compatible_proxy_priors"
                if source_binding_blockers
                else "video_pair_count_compatible"
            ),
            "source_binding_blockers": source_binding_blockers,
            "semantics": (
                "importance ranks residual cells; P19-null cells are cheapest, "
                "P18 SegNet-vulnerable cells are protected, and selection_domain "
                "controls whether coarsening is global-weighted or confined to "
                "the explicit low-importance eligible mask"
            ),
        },
    }


def _frames_for_pair(pair_id: int, *, frame_count: int, gop_size: int) -> range:
    start = int(pair_id) * int(gop_size)
    stop = min(frame_count, start + int(gop_size))
    if start < 0 or start >= frame_count:
        return range(0, 0)
    return range(start, stop)


def _grid_start(value: float, size: int) -> int:
    return max(0, min(size, int(np.floor(float(value) * float(size)))))


def _grid_end(value: float, size: int) -> int:
    return max(0, min(size, int(np.ceil(float(value) * float(size)))))


def _artifact_path(artifact: dict[str, Any], key: str, *, repo_root: Path) -> Path:
    raw = artifact.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"training result artifact missing {key}")
    path = _resolve(Path(raw), repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"training result artifact path missing: {path}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _select_best_variant(
    variants: list[dict[str, Any]],
    *,
    target_rate_term: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not variants:
        raise ValueError("rate-collapse produced no variants")
    if target_rate_term is not None:
        target = float(target_rate_term)
        passing = [
            row
            for row in variants
            if float(row.get("archive_rate_term", float("inf"))) < target
        ]
        if passing:
            best = min(
                passing,
                key=lambda row: (
                    _residual_metric(row, "residual_q_mse"),
                    _residual_metric(row, "residual_q_l1"),
                    int(row["archive_zip_bytes"]),
                    str(row["variant_id"]),
                ),
            )
            return best, {
                "schema": "hprc_rate_collapse_variant_selection.v1",
                "mode": "least_residual_damage_under_target_rate",
                "target_rate_term": target,
                "passing_variant_count": len(passing),
                "selected_variant_id": best["variant_id"],
            }
        best = min(
            variants,
            key=lambda row: (
                int(row["archive_zip_bytes"]),
                _residual_metric(row, "residual_q_mse"),
                str(row["variant_id"]),
            ),
        )
        return best, {
            "schema": "hprc_rate_collapse_variant_selection.v1",
            "mode": "no_variant_under_target_lowest_archive_bytes",
            "target_rate_term": target,
            "passing_variant_count": 0,
            "selected_variant_id": best["variant_id"],
        }
    best = min(variants, key=lambda row: (int(row["archive_zip_bytes"]), str(row["variant_id"])))
    return best, {
        "schema": "hprc_rate_collapse_variant_selection.v1",
        "mode": "lowest_archive_bytes",
        "target_rate_term": None,
        "passing_variant_count": None,
        "selected_variant_id": best["variant_id"],
    }


def _residual_transform_for_variant(row: dict[str, Any]) -> str:
    variant_id = str(row.get("variant_id") or "unknown")
    if bool(row.get("lossy_residual_token_collapse")):
        return f"hprc_lossy_residual_token_rate_collapse_{variant_id}"
    return "lossless_hprc_section_entropy_rate_collapse"


def _residual_metric(row: dict[str, Any], key: str) -> float:
    metrics = row.get("residual_token_collapse")
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _resolve(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: dict[str, Any], *, allow_overwrite: bool) -> None:
    expected = sha256_file(path) if path.is_file() and allow_overwrite else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected,
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, FileNotFoundError, ValueError) as exc:
        print(f"transcode_hprc_compact_receiver_rate_collapse failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
