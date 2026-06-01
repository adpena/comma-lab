#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Project compact renderer archives/blobs onto the HPRC section spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.hprc.representation_spine import (  # noqa: E402
    HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    build_packed_hnerv_spine_from_archive,
    build_pact_nerv_len_prefixed_spine_from_archive,
    build_pact_nerv_vq_spine_from_archive,
    build_pr95_hnerv_spine_from_archive,
    write_representation_spine_projection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=[family.value for family in HprcRepresentationFamily],
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Source archive.zip for pr95_hnerv, hnerv_packed, or pact_nerv_vq.",
    )
    parser.add_argument("--decoder-blob", type=Path)
    parser.add_argument("--latents-blob", type=Path)
    parser.add_argument("--codebooks-blob", type=Path)
    parser.add_argument("--selectors-blob", type=Path)
    parser.add_argument("--residual-blob", type=Path)
    parser.add_argument("--receiver-state-blob", type=Path)
    parser.add_argument("--manifest-extra-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--basename", default="hprc_representation_spine")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    family = HprcRepresentationFamily(args.family)
    archive = _resolve_optional(args.archive, repo_root)
    if family == HprcRepresentationFamily.PR95_HNERV:
        if archive is None:
            raise ValueError("--archive is required for pr95_hnerv")
        spine = build_pr95_hnerv_spine_from_archive(archive)
    elif family == HprcRepresentationFamily.HNERV_PACKED:
        if archive is None:
            raise ValueError("--archive is required for hnerv_packed")
        spine = build_packed_hnerv_spine_from_archive(archive)
    elif family == HprcRepresentationFamily.PACT_NERV_VQ:
        if archive is None:
            raise ValueError("--archive is required for pact_nerv_vq")
        spine = build_pact_nerv_vq_spine_from_archive(archive)
    elif family == HprcRepresentationFamily.PACT_NERV and archive is not None:
        spine = build_pact_nerv_len_prefixed_spine_from_archive(archive)
    else:
        decoder_blob_path = _resolve_optional(args.decoder_blob, repo_root)
        if decoder_blob_path is None:
            raise ValueError(f"--decoder-blob is required for generic family {family.value}")
        spine = build_generic_neural_spine_packet(
            family=family,
            decoder_blob=decoder_blob_path.read_bytes(),
            latents_blob=_read_optional_blob(args.latents_blob, repo_root),
            codebooks_blob=_read_optional_blob(args.codebooks_blob, repo_root),
            selectors_blob=_read_optional_blob(args.selectors_blob, repo_root),
            residual_blob=_read_optional_blob(args.residual_blob, repo_root),
            receiver_state_blob=_read_optional_blob(args.receiver_state_blob, repo_root),
            manifest_extra=_load_extra(args.manifest_extra_json, repo_root),
        )

    output_dir = _resolve(args.output_dir, repo_root)
    projection = write_representation_spine_projection(
        output_dir=output_dir,
        spine=spine,
        basename=str(args.basename),
    )
    report = {
        "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
        "projection": projection,
        "source_archive": None if archive is None else archive.as_posix(),
        "next_required_action": "route_projection_into_value_per_byte_acquisition",
        **FALSE_AUTHORITY,
    }
    out_json = _resolve_optional(args.out_json, repo_root)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def _resolve(path: Path, repo_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else repo_root / expanded


def _resolve_optional(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    return _resolve(path, repo_root)


def _read_optional_blob(path: Path | None, repo_root: Path) -> bytes:
    resolved = _resolve_optional(path, repo_root)
    return b"" if resolved is None else resolved.read_bytes()


def _load_extra(path: Path | None, repo_root: Path) -> dict[str, object]:
    resolved = _resolve_optional(path, repo_root)
    if resolved is None:
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--manifest-extra-json must contain a JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"build_hprc_representation_spine_projection failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
