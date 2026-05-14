#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Estimate compression-based dependence between archive sections and signals.

The tool is read-only with respect to input artifacts and emits JSON to stdout.
It computes deterministic compressor sizes, an algorithmic mutual-information
proxy ``C(x) + C(y) - C(xy)``, and normalized compression distance. These are
diagnostics only and never score authority, promotion authority, or exact-eval
dispatch readiness.
"""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import sys
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Compressor = Callable[[bytes], bytes]
EVIDENCE_GRADE = "proxy_diagnostic_kolmogorov_mi"


@dataclass(frozen=True)
class NamedBlob:
    name: str
    source: str
    data: bytes


def _compressor(name: str) -> Compressor:
    if name == "zlib":
        return lambda data: zlib.compress(data, level=9)
    if name == "bz2":
        return lambda data: bz2.compress(data, compresslevel=9)
    if name == "lzma":
        return lambda data: lzma.compress(
            data,
            format=lzma.FORMAT_XZ,
            check=lzma.CHECK_CRC32,
            preset=9,
        )
    raise ValueError(f"unsupported compressor {name!r}")


def _split_label_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=PATH or LABEL=ZIP::MEMBER, got {spec!r}")
    label, source = spec.split("=", 1)
    if not label:
        raise ValueError("label must be non-empty")
    if not source:
        raise ValueError("source must be non-empty")
    return label, source


def _read_source(source: str) -> bytes:
    if "::" in source:
        archive_raw, member = source.split("::", 1)
        if not archive_raw or not member:
            raise ValueError(f"bad zip-member source {source!r}")
        archive = Path(archive_raw)
        with zipfile.ZipFile(archive, "r") as zf:
            return zf.read(member)
    return Path(source).read_bytes()


def _load_named_blob(spec: str) -> NamedBlob:
    label, source = _split_label_spec(spec)
    return NamedBlob(name=label, source=source, data=_read_source(source))


def _pair_record(section: NamedBlob, signal: NamedBlob, compress: Compressor) -> dict[str, object]:
    cx = len(compress(section.data))
    cy = len(compress(signal.data))
    cxy = len(compress(section.data + signal.data))
    cyx = len(compress(signal.data + section.data))
    joint = min(cxy, cyx)
    denom = max(cx, cy, 1)
    ami = cx + cy - joint
    ncd = (joint - min(cx, cy)) / denom
    return {
        "section": section.name,
        "section_source": section.source,
        "signal": signal.name,
        "signal_source": signal.source,
        "raw_section_bytes": len(section.data),
        "raw_signal_bytes": len(signal.data),
        "compressed_section_bytes": cx,
        "compressed_signal_bytes": cy,
        "compressed_joint_bytes": joint,
        "algorithmic_mi_bytes": ami,
        "normalized_mi": max(0.0, ami / denom),
        "normalized_compression_distance": ncd,
        "evidence_grade": EVIDENCE_GRADE,
        "proxy": True,
        "proxy_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate deterministic compression-based MI/NCD between archive "
            "sections and scorer-relevant signal files."
        )
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Archive section bytes. ZIP members use LABEL=archive.zip::member.",
    )
    parser.add_argument(
        "--signal",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Scorer-relevant signal bytes. ZIP members use LABEL=archive.zip::member.",
    )
    parser.add_argument(
        "--compressor",
        choices=("zlib", "bz2", "lzma"),
        default="zlib",
        help="Deterministic stdlib compressor to use.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.section:
        parser.error("at least one --section is required")
    if not args.signal:
        parser.error("at least one --signal is required")

    try:
        sections = [_load_named_blob(spec) for spec in args.section]
        signals = [_load_named_blob(spec) for spec in args.signal]
        compress = _compressor(args.compressor)
        records = [
            _pair_record(section, signal, compress)
            for section in sections
            for signal in signals
        ]
    except Exception as exc:
        print(f"diagnose_kolmogorov_mi: {exc}", file=sys.stderr)
        return 2

    output = {
        "schema_version": 1,
        "tool": "tools/diagnose_kolmogorov_mi.py",
        "compressor": args.compressor,
        "records": records,
        "evidence_grade": EVIDENCE_GRADE,
        "proxy": True,
        "proxy_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "operator_metadata": {
            "operator_visible": True,
            "runbook": (
                "tools/diagnose_kolmogorov_mi.py --section LABEL=PATH "
                "--signal LABEL=PATH"
            ),
            "preflight_visibility": (
                "diagnostic JSON must remain proxy-only; do not promote to "
                "exact-eval readiness from KC3 compression dependence alone"
            ),
            "score_authority": "none",
        },
    }
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
