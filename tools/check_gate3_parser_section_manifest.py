#!/usr/bin/env python3
"""Gate 3 — Parser-section manifest gate (HNeRV-family monolithic packets).

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #3.

Rule: for monolithic HNeRV-family payloads (PR101/PR103/PR106-class
packets that have a single charged ZIP member parsed into internal
sections), require a parser-section manifest with:

  * ``offsets``    (list of int)
  * ``lengths``    (list of int)
  * ``section_names`` (list of str)
  * ``section_sha256s`` (list of str, hex)
  * ``entropy_estimates`` (list of float)
  * ``old_new_section_boundaries`` (object: each section name -> {old, new})

Detection (static):
  Scan ``experiments/results/**/build_manifest.json`` and
  ``reports/raw/**/manifest.json`` for manifests that are flagged
  monolithic-HNeRV via ANY of:

    * ``packet_family`` containing ``hnerv``
    * ``archive_layout=monolithic_hnerv``
    * ``representation_name`` containing ``hnerv``/``HNeRV``/``mnerv``
    * Filename path includes ``hnerv``

  For every such manifest, require ALL six fields above to be non-empty.

Live count on landing: typically 0 for synthetic candidates; existing
public PR101/103/106 intake manifests are vendored read-only and
exempted via the ``vendored_public_pr_intake=true`` waiver token.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

REQUIRED_PARSER_FIELDS: tuple[str, ...] = (
    "offsets",
    "lengths",
    "section_names",
    "section_sha256s",
    "entropy_estimates",
    "old_new_section_boundaries",
)

HNERV_FAMILY_TOKENS = (
    "hnerv",
    "HNeRV",
    "mnerv",
    "MNeRV",
    "monolithic_hnerv",
)


@dataclass
class Finding:
    manifest_rel: str
    representation: str
    reason: str


def _is_hnerv_family(manifest: dict, manifest_path: Path) -> bool:
    family = str(manifest.get("packet_family", "")).lower()
    if any(tok.lower() in family for tok in HNERV_FAMILY_TOKENS):
        return True
    layout = str(manifest.get("archive_layout", "")).lower()
    if "monolithic_hnerv" in layout:
        return True
    name = str(manifest.get("representation_name", "")).lower()
    if any(tok.lower() in name for tok in HNERV_FAMILY_TOKENS):
        return True
    parts = manifest_path.as_posix().lower()
    return "hnerv" in parts


def _has_vendored_waiver(manifest: dict) -> bool:
    if manifest.get("vendored_public_pr_intake") is True:
        return True
    return str(manifest.get("vendored_public_pr_intake", "")).lower() == "true"


def _missing_parser_fields(manifest: dict) -> list[str]:
    parser = manifest.get("parser_section_manifest")
    if isinstance(parser, dict):
        # Required fields can be at top level or nested in
        # parser_section_manifest.
        return [f for f in REQUIRED_PARSER_FIELDS if not parser.get(f)]
    return [f for f in REQUIRED_PARSER_FIELDS if not manifest.get(f)]


def _parser_payload(manifest: dict) -> dict[str, Any]:
    parser = manifest.get("parser_section_manifest")
    if isinstance(parser, dict):
        return parser
    return manifest


def _schema_errors(manifest: dict) -> list[str]:
    p = _parser_payload(manifest)
    errors: list[str] = []
    offsets = p.get("offsets")
    lengths = p.get("lengths")
    names = p.get("section_names")
    hashes = p.get("section_sha256s")
    entropy = p.get("entropy_estimates")
    boundaries = p.get("old_new_section_boundaries")
    arrays = {
        "offsets": offsets,
        "lengths": lengths,
        "section_names": names,
        "section_sha256s": hashes,
        "entropy_estimates": entropy,
    }
    for field, value in arrays.items():
        if not isinstance(value, list) or len(value) == 0:
            errors.append(f"{field} must be a non-empty list")
    if errors:
        return errors
    n = len(names)
    for field, value in arrays.items():
        if len(value) != n:
            errors.append(f"{field} length {len(value)} != section_names length {n}")
    if any(not isinstance(v, int) or v < 0 for v in offsets):
        errors.append("offsets must be non-negative integers")
    if any(not isinstance(v, int) or v <= 0 for v in lengths):
        errors.append("lengths must be positive integers")
    if any(not isinstance(v, str) or not v.strip() for v in names):
        errors.append("section_names must be non-empty strings")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v) for v in hashes):
        errors.append("section_sha256s must be 64-hex strings")
    if any(not isinstance(v, int | float) or v < 0 for v in entropy):
        errors.append("entropy_estimates must be non-negative numbers")
    if not isinstance(boundaries, dict):
        errors.append("old_new_section_boundaries must be an object")
    else:
        missing_names = [name for name in names if name not in boundaries]
        if missing_names:
            errors.append(
                "old_new_section_boundaries missing sections: "
                + ",".join(str(v) for v in missing_names)
            )
        for name in names:
            value = boundaries.get(name)
            if not isinstance(value, dict):
                continue
            for key in ("old", "new"):
                bounds = value.get(key)
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or not all(isinstance(v, int) and v >= 0 for v in bounds)
                    or bounds[0] > bounds[1]
                ):
                    errors.append(f"old_new_section_boundaries.{name}.{key} must be [start,end]")
    return errors


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    patterns = (
        "experiments/results/*/build_manifest.json",
        "experiments/results/*/*/build_manifest.json",
        "reports/raw/*/manifest.json",
        "reports/raw/*/*/manifest.json",
    )
    for pattern in patterns:
        for path in repo.glob(pattern):
            # Skip vendored public PR intakes.
            relpath = path.relative_to(repo).as_posix()
            if "public_pr" in relpath and "intake" in relpath:
                # Vendored snapshot; do not enforce
                continue
            if not path.is_file():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict):
                continue
            if not _is_hnerv_family(manifest, path):
                continue
            if _has_vendored_waiver(manifest):
                continue
            missing = _missing_parser_fields(manifest)
            schema_errors = [] if missing else _schema_errors(manifest)
            if not missing and not schema_errors:
                continue
            detail = (
                f"missing required parser-section fields: {','.join(missing)}"
                if missing
                else "invalid parser-section schema: " + "; ".join(schema_errors)
            )
            findings.append(
                Finding(
                    manifest_rel=relpath,
                    representation=str(
                        manifest.get("representation_name", "<unknown>")
                    ),
                    reason=(
                        f"HNeRV-family monolithic-packet manifest {detail}. "
                        f"Provide parser_section_manifest with offsets, "
                        f"lengths, section_names, section_sha256s, "
                        f"entropy_estimates, old_new_section_boundaries OR "
                        f"set vendored_public_pr_intake=true. Gate 3 "
                        f"(parser-section gate)."
                    ),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[gate3-parser-section-manifest] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.manifest_rel} representation={f.representation}: "
                f"{f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate3-parser-section-manifest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
