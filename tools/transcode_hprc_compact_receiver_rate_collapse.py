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

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.repo_io import ArtifactWriteError, sha256_bytes, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY, export_hprc_archive_bytes  # noqa: E402
from tac.substrates.hprc.rate_collapse import (  # noqa: E402
    DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS,
    HPRC_RATE_COLLAPSE_REPORT_SCHEMA,
    parse_rate_collapse_sections,
    parse_residual_token_collapse_specs,
    rate_collapse_variant_groups,
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
    lossy_residual_collapse_enabled = (
        (bool(args.enable_lossy_residual_collapse) or args.target_rate_term is not None)
        and not bool(args.disable_lossy_residual_collapse)
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

    best, selection = _select_best_variant(
        variants,
        target_rate_term=args.target_rate_term,
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
        "best_variant_id": best["variant_id"],
        "best_variant_sections": best["sections"],
        "best_variant_selection": selection,
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
            **FALSE_AUTHORITY,
        },
        "receiver_proof_requested": not bool(args.skip_receiver_proof),
        "receiver_output_retained": bool(args.retain_receiver_output),
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
