#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile SNeRV SNAR1 receiver-valid section neutralizations.

This tool is intentionally scorer-free.  It materializes receiver-decodable
semantic neutralizations for optional SNAR1 sections, writes packet variants,
and emits false-authority section-value rows that can be priced by the shared
NeRV byte-price controller after scorer replay supplies non-rate deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    sha256_bytes,
    sha256_file,
    write_bytes_artifact,
    write_json_artifact,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    SECTION_ORDER,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.section_value import (  # noqa: E402
    NEUTRALIZABLE_SNERV_SECTIONS,
    SNERV_SNAR1_SECTION_VALUE_SCHEMA,
    neutralize_snerv_section,
)

PROFILE_SCHEMA = "snerv_snar1_section_value_profile.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Raw .snar packet or archive.zip containing member 0.bin.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--variant-output-dir",
        type=Path,
        help="Directory for baseline/neutralized .snar packets.",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=list(NEUTRALIZABLE_SNERV_SECTIONS),
        help=f"SNAR1 sections to neutralize. Default: {NEUTRALIZABLE_SNERV_SECTIONS}",
    )
    parser.add_argument("--step-map-bins", type=int, default=16)
    parser.add_argument(
        "--skip-receiver-decode",
        action="store_true",
        help="Do not decode neutralized packets after rebuild; adds blockers.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting output files only with matching expected SHA flags.",
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-variant-tree-present", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    output_json = _resolve(args.output_json)
    variant_output_dir = _resolve(
        args.variant_output_dir
        if args.variant_output_dir is not None
        else output_json.with_suffix("").with_name(f"{output_json.stem}_variants")
    )
    report = build_snerv_snar1_section_value_profile(
        input_path=_resolve(args.input_path),
        variant_output_dir=variant_output_dir,
        requested_sections=tuple(str(section) for section in args.sections),
        step_map_bins=int(args.step_map_bins),
        verify_receiver_decode=not bool(args.skip_receiver_decode),
        raw_argv=raw_argv,
        allow_overwrite=bool(args.allow_overwrite),
        expected_variant_tree_present=bool(args.expected_variant_tree_present),
    )
    result = write_json_artifact(
        output_json,
        report,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    print(
        json.dumps(
            {
                "schema": PROFILE_SCHEMA,
                "report": result.path,
                "bytes": result.bytes_written,
                "sha256": result.sha256,
                "variant_count": report["variant_count"],
                "section_value_row_count": len(report["section_value_rows"]),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def build_snerv_snar1_section_value_profile(
    *,
    input_path: Path,
    variant_output_dir: Path,
    requested_sections: tuple[str, ...],
    step_map_bins: int = 16,
    verify_receiver_decode: bool = True,
    raw_argv: list[str] | None = None,
    allow_overwrite: bool = False,
    expected_variant_tree_present: bool = False,
) -> dict[str, Any]:
    if int(step_map_bins) < 2:
        raise ValueError("step_map_bins must be >= 2")
    packet, input_kind = _read_snerv_packet(input_path)
    decoded = unpack_snerv_archive(packet)
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    if expected_variant_tree_present is False and any(variant_output_dir.iterdir()):
        raise FileExistsError(
            f"variant output dir is not empty; pass a fresh directory: {variant_output_dir}"
        )
    baseline_path = variant_output_dir / "baseline.snar"
    baseline_write = _write_packet_artifact_idempotent(baseline_path, packet)
    variants: list[dict[str, Any]] = [
        {
            "variant_id": "baseline",
            "section": None,
            "packet_path": baseline_write["path"],
            "packet_bytes": baseline_write["bytes_written"],
            "packet_sha256": baseline_write["sha256"],
            "receiver_decode_status": "baseline_not_neutralized",
            **FALSE_AUTHORITY,
        }
    ]
    section_rows: list[dict[str, Any]] = []
    blockers: list[str] = [
        "delta_nonrate_score_missing",
        "runtime_consumption_proof_missing_for_neutralized_packets",
        "paired_contest_cpu_cuda_auth_eval_missing",
    ]
    requested = _normalize_sections(requested_sections)
    for section in requested:
        neutralized = neutralize_snerv_section(
            packet,
            section,
            step_map_bins=int(step_map_bins),
            verify_receiver_decode=bool(verify_receiver_decode),
        )
        neutralized_packet = bytes(neutralized["packet"])
        variant_path = variant_output_dir / f"neutralized_{section}.snar"
        variant_write = _write_packet_artifact_idempotent(
            variant_path, neutralized_packet
        )
        variant = {
            key: value for key, value in neutralized.items() if key != "packet"
        }
        variant.update(
            {
                "variant_id": f"neutralized_{section}",
                "packet_path": variant_write["path"],
                "packet_bytes": variant_write["bytes_written"],
                "packet_sha256": variant_write["sha256"],
            }
        )
        variants.append(variant)
        row = dict(neutralized["section_value_row"])
        row.update(
            {
                "variant_packet_path": variant_write["path"],
                "neutralized_packet_sha256": variant_write["sha256"],
                "neutralized_packet_bytes": variant_write["bytes_written"],
            }
        )
        section_rows.append(row)
        blockers.extend(str(blocker) for blocker in row.get("blockers") or [])
    return {
        "schema": PROFILE_SCHEMA,
        "source_schema": SNERV_SNAR1_SECTION_VALUE_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/profile_snerv_snar1_section_value.py",
        "raw_argv": list(raw_argv or []),
        "input_path": input_path.as_posix(),
        "input_kind": input_kind,
        "input_bytes": len(packet),
        "input_sha256": sha256_bytes(packet),
        "variant_output_dir": variant_output_dir.as_posix(),
        "section_order": list(SECTION_ORDER),
        "neutralizable_sections": list(NEUTRALIZABLE_SNERV_SECTIONS),
        "requested_sections": list(requested),
        "step_map_bins": int(step_map_bins),
        "verify_receiver_decode": bool(verify_receiver_decode),
        "allow_report_overwrite_requested": bool(allow_overwrite),
        "decoded_metadata": dict(decoded.metadata),
        "baseline_section_bytes": {
            section: len(blob) for section, blob in decoded.sections.items()
        },
        "variant_count": len(variants),
        "variants": variants,
        "section_value_rows": section_rows,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _read_snerv_packet(input_path: Path) -> tuple[bytes, str]:
    source = Path(input_path).expanduser().resolve(strict=False)
    blob = source.read_bytes()
    if blob.startswith(b"SNAR1"):
        return blob, "raw_snar_packet"
    if source.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(source) as zf:
            return zf.read("0.bin"), "archive_zip_member_0_bin"
    raise ValueError(f"{source}: expected raw SNAR1 packet or archive.zip")


def _normalize_sections(sections: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for section in sections:
        clean = str(section).strip()
        if not clean:
            continue
        if clean not in NEUTRALIZABLE_SNERV_SECTIONS:
            raise ValueError(
                f"unsupported SNeRV section {clean!r}; "
                f"valid={NEUTRALIZABLE_SNERV_SECTIONS}"
            )
        if clean not in out:
            out.append(clean)
    return tuple(out)


def _write_packet_artifact_idempotent(path: Path, packet: bytes) -> dict[str, Any]:
    """Write a packet artifact, allowing only byte-identical existing files."""

    expected_sha = sha256_bytes(packet)
    if path.exists():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ArtifactWriteError(
                f"{path}: refusing to replace non-identical SNAR1 variant "
                f"expected={expected_sha} actual={actual_sha}"
            )
        return {
            "path": path.as_posix(),
            "bytes_written": int(path.stat().st_size),
            "sha256": actual_sha,
        }
    result = write_bytes_artifact(path, packet)
    return {
        "path": result.path,
        "bytes_written": result.bytes_written,
        "sha256": result.sha256,
    }


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve(strict=False)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
