#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build a sanitized Cloudflare Pages bundle from reports/graphs/site.

The historical site directory contains useful generated artifacts, but some of
its JSON timelines preserve local operator paths for private custody. This
builder copies the site into a separate public bundle and redacts private ops
surfaces before the strict publish hygiene guard runs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.preflight import check_public_release_hygiene
from tools.audit_public_publish_links import audit_public_publish_links


DEFAULT_SOURCE = ROOT / "site"
DEFAULT_OUTPUT = ROOT / "public_site"
DEFAULT_MAX_ASSET_BYTES = 25 * 1024 * 1024


REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_comma_lab_github_url",
        re.compile(r"https://github\.com/adpena/comma-lab(?:/[^\s)\"'<>]*)?"),
        "https://github.com/adpena/tac",
    ),
    (
        "local_absolute_operator_path",
        re.compile(
            r"(?<![A-Za-z0-9_])/"
            r"(?:Users|home|private|tmp|teamspace|var/folders)"
            r"(?:/[^\s)\"'<>`]*)?"
        ),
        "${LOCAL_PATH_REDACTED}",
    ),
    (
        "private_lightning_studio_url",
        re.compile(r"https://lightning\.ai/[^/\s)]+/[^/\s)]+/studios/[^\s)\"'<>]*"),
        "${LIGHTNING_PRIVATE_URL_REDACTED}",
    ),
    (
        "vast_ssh_endpoint",
        re.compile(r"\bssh\d+\.vast\.ai(?::\d+)?\b"),
        "${VAST_SSH_REDACTED}",
    ),
    (
        "api_token",
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b"),
        "${TOKEN_REDACTED}",
    ),
    (
        "secret_env_assignment",
        re.compile(r"\b(VAST_API_KEY|LIGHTNING_API_KEY|CLOUDFLARE_API_TOKEN|OPENAI_API_KEY)\s*=\s*[^\s\"'`]+"),
        r"\1=${SECRET_REDACTED}",
    ),
    (
        "modal_call_id",
        re.compile(r"\bfc-[A-Z0-9]{20,}\b"),
        "${MODAL_ID_REDACTED}",
    ),
    (
        "modal_app_id",
        re.compile(r"\bap-[A-Za-z0-9]{10,}\b"),
        "${MODAL_ID_REDACTED}",
    ),
)


@dataclass(frozen=True)
class RedactionRecord:
    path: str
    label: str
    count: int


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"${{EXTERNAL_PATH}}/{path.name}"


def sanitize_text(text: str) -> tuple[str, list[tuple[str, int]]]:
    records: list[tuple[str, int]] = []
    out = text
    for label, pattern, replacement in REDACTIONS:
        out, count = pattern.subn(replacement, out)
        if count:
            records.append((label, count))
    return out, records


def _sanitize_file(path: Path, root: Path) -> list[RedactionRecord]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    sanitized, raw_records = sanitize_text(text)
    if sanitized != text:
        path.write_text(sanitized, encoding="utf-8")
    rel = path.relative_to(root).as_posix()
    return [RedactionRecord(path=rel, label=label, count=count) for label, count in raw_records]


def _asset_size_violations(root: Path, max_asset_bytes: int) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.stat().st_size > max_asset_bytes:
            violations.append(f"{path.relative_to(root).as_posix()}:{path.stat().st_size}")
    return violations


def _remove_oversized_assets(root: Path, max_asset_bytes: int) -> list[dict[str, object]]:
    omitted: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= max_asset_bytes:
            continue
        omitted.append({"path": path.relative_to(root).as_posix(), "bytes": size})
        path.unlink()
    return omitted


def build_public_site_bundle(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    *,
    max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    oversized_policy: str = "omit",
    strict_hygiene: bool = True,
    live_link_audit: bool = False,
) -> dict[str, object]:
    if oversized_policy not in {"omit", "fail"}:
        raise ValueError("oversized_policy must be 'omit' or 'fail'")
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"missing source site directory: {source}")
    if source == output or _is_relative_to(output, source):
        raise ValueError("output must not be inside source")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)

    redactions: list[RedactionRecord] = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            redactions.extend(_sanitize_file(path, output))

    omitted_oversized_assets: list[dict[str, object]] = []
    if oversized_policy == "omit":
        omitted_oversized_assets = _remove_oversized_assets(output, max_asset_bytes)
    size_violations = _asset_size_violations(output, max_asset_bytes)
    if size_violations:
        raise RuntimeError(
            "public site asset exceeds size limit:\n"
            + "\n".join(f"  - {item}" for item in size_violations)
        )

    hygiene_violations: list[str] = []
    manifest = {
        "schema_version": 1,
        "source": _display_path(source),
        "output": _display_path(output),
        "file_count": sum(1 for path in output.rglob("*") if path.is_file()),
        "redaction_count": sum(record.count for record in redactions),
        "redactions": [record.__dict__ for record in redactions],
        "omitted_oversized_assets": omitted_oversized_assets,
        "hygiene_violation_count": 0,
        "public_link_count": 0,
        "public_link_violation_count": 0,
        "public_link_live_audit": live_link_audit,
        "max_asset_bytes": max_asset_bytes,
        "oversized_policy": oversized_policy,
        "score_claim": False,
        "promotion_claim": False,
        "allowed_use": "public_cloudflare_pages_supplement_bundle",
    }
    (output / "public_site_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hygiene_violations = check_public_release_hygiene(
        repo_root=REPO_ROOT,
        strict=strict_hygiene,
        verbose=False,
        scan_paths=[output],
    )
    link_payload = audit_public_publish_links(
        [output],
        base_root=output,
        live=live_link_audit,
    )
    link_violations = [
        "{path}:{line}: {kind}: {url} ({detail})".format(**violation)
        for violation in link_payload["violations"]
    ]
    if link_violations and strict_hygiene:
        raise RuntimeError(
            "PUBLIC SITE LINK HYGIENE violations:\n"
            + "\n".join(f"  - {violation}" for violation in link_violations[:40])
        )
    manifest["hygiene_violation_count"] = len(hygiene_violations)
    manifest["public_link_count"] = int(link_payload["link_count"])
    manifest["public_link_violation_count"] = len(link_violations)
    (output / "public_site_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-asset-bytes", type=int, default=DEFAULT_MAX_ASSET_BYTES)
    parser.add_argument("--oversized-policy", choices=("omit", "fail"), default="omit")
    parser.add_argument("--no-strict-hygiene", action="store_true")
    parser.add_argument("--live-link-audit", action="store_true")
    args = parser.parse_args()

    manifest = build_public_site_bundle(
        args.source,
        args.output,
        max_asset_bytes=args.max_asset_bytes,
        oversized_policy=args.oversized_policy,
        strict_hygiene=not args.no_strict_hygiene,
        live_link_audit=args.live_link_audit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
