#!/usr/bin/env python3
"""Audit public publish surfaces for broken or private links.

The static mode is deterministic and suitable for preflight/tests. The live
mode intentionally performs unauthenticated requests only; it catches links
that look public in text but actually require an operator session.
"""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402

DEFAULT_PRIVATE_NETLOCS = (
    "github.com/adpena/comma-lab",
)

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yml",
    ".yaml",
}

URL_RE = re.compile(r"https?://[^\s<>'\")\]]+")


@dataclass(frozen=True)
class LinkRecord:
    path: str
    line: int
    url: str


@dataclass(frozen=True)
class LinkViolation:
    path: str
    line: int
    url: str
    kind: str
    detail: str


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_scan_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}:
                files.append(path)
    return sorted({path.resolve() for path in files})


def _clean_url(raw: str) -> str:
    return raw.rstrip(".,;:")


def extract_public_links(roots: Iterable[Path], *, base_root: Path | None = None) -> list[LinkRecord]:
    """Extract HTTP(S) links from text-like files under ``roots``."""
    root = base_root or Path.cwd()
    records: list[LinkRecord] = []
    for path in _iter_scan_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = _display_path(path, root)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in URL_RE.finditer(line):
                records.append(LinkRecord(path=rel, line=lineno, url=_clean_url(match.group(0))))
    return records


def _normalized_private_target(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc.lower()}{parsed.path}".rstrip("/")


def static_link_violations(
    links: Iterable[LinkRecord],
    *,
    private_netlocs: tuple[str, ...] = DEFAULT_PRIVATE_NETLOCS,
) -> list[LinkViolation]:
    """Return deterministic private-link violations without network access."""
    private_targets = tuple(target.lower().rstrip("/") for target in private_netlocs)
    violations: list[LinkViolation] = []
    for link in links:
        normalized = _normalized_private_target(link.url)
        for private_target in private_targets:
            if normalized == private_target or normalized.startswith(private_target + "/"):
                violations.append(
                    LinkViolation(
                        path=link.path,
                        line=link.line,
                        url=link.url,
                        kind="private_link",
                        detail=f"link points at private publish surface {private_target}",
                    )
                )
    return violations


def unauthenticated_status(url: str, *, timeout: float = 10.0) -> tuple[int | None, str]:
    """Fetch a URL without auth headers and return ``(status, detail)``."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pact-public-link-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), "HEAD"
    except urllib.error.HTTPError as exc:
        if exc.code not in {401, 403, 404, 405, 501}:
            return int(exc.code), "HEAD"
    except Exception as exc:  # pragma: no cover - exercised via injected checker in tests.
        return None, type(exc).__name__

    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "pact-public-link-audit/1", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), "GET"
    except urllib.error.HTTPError as exc:
        return int(exc.code), "GET"
    except Exception as exc:  # pragma: no cover - exercised via injected checker in tests.
        return None, type(exc).__name__


def live_link_violations(
    links: Iterable[LinkRecord],
    *,
    checker: Callable[[str], tuple[int | None, str]] = unauthenticated_status,
) -> list[LinkViolation]:
    """Return violations for links that are not publicly readable."""
    violations: list[LinkViolation] = []
    seen: dict[str, tuple[int | None, str]] = {}
    for link in links:
        if link.url not in seen:
            seen[link.url] = checker(link.url)
        status, method = seen[link.url]
        if status is None or status in {401, 403, 404} or (status is not None and status >= 500):
            violations.append(
                LinkViolation(
                    path=link.path,
                    line=link.line,
                    url=link.url,
                    kind="unauthenticated_link_failure",
                    detail=f"{method} returned {status if status is not None else 'no status'}",
                )
            )
    return violations


def audit_public_publish_links(
    roots: Iterable[Path],
    *,
    base_root: Path | None = None,
    live: bool = False,
    checker: Callable[[str], tuple[int | None, str]] = unauthenticated_status,
) -> dict[str, object]:
    roots = [root.resolve() for root in roots]
    base = (base_root or Path.cwd()).resolve()
    links = extract_public_links(roots, base_root=base)
    violations = static_link_violations(links)
    if live:
        violations.extend(live_link_violations(links, checker=checker))
    return {
        "schema_version": 1,
        "roots": [_display_path(root, base) for root in roots],
        "live": live,
        "link_count": len(links),
        "violation_count": len(violations),
        "violations": [asdict(violation) for violation in violations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    payload = audit_public_publish_links(args.roots, base_root=args.repo_root, live=args.live)
    if args.format == "json":
        print(json_text(payload), end="")
    elif payload["violation_count"]:
        print(
            "public publish link audit: FAIL "
            f"({payload['violation_count']} violation(s), {payload['link_count']} link(s) scanned)"
        )
        for violation in payload["violations"][:30]:
            print(
                "  - {path}:{line}: {kind}: {url} ({detail})".format(
                    **violation,
                )
            )
    else:
        mode = "live unauthenticated" if args.live else "static"
        print(f"public publish link audit: PASS ({mode}, {payload['link_count']} link(s) scanned)")

    return 1 if args.strict and payload["violation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
