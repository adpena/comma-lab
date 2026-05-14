#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch ALL scored PR archives + sources from the comma video compression challenge.

Per user mandate 2026-05-05 ("don't we have repo for all 090+ PRs? we should if
not as long as they innovated or beat scores"): the per-PR archive corpus is
the foundation for any post-contest reverse-engineering, methodological audit,
or paper figure. Without it, claims about what others did are speculation.

This is the HARDENED version (v2): no silent failures. Every archive-download
attempt is logged with explicit reason code; every PR with no recoverable
archive is flagged in the FETCH_SUMMARY with `archive_status` field so the
operator can manually triage. The fetcher walks GitHub releases for each PR's
head_repo as a fallback when the PR body lacks a direct .zip URL.

For each scored PR on the live leaderboard, fetch:
  1. PR metadata (created_at, author, score, name, head SHA)
  2. PR body (often contains archive download URL)
  3. archive.zip — searched in this order:
     a. direct .zip URL in PR body
     b. direct .zip URL in PR comments
     c. archive.zip path referenced by the PR body and present in the head tree
     d. GitHub release asset on the head_repo, ranked by PR/name/body relevance
     e. release asset in commaai/comma_video_compression_challenge releases
     f. flagged as needing manual triage (with explicit reason)
  4. source tree at the PR's head SHA (shallow clone of the contestant's fork)
  5. SHA-256 of the archive (for byte-exact provenance audits)

Output structure (per PR):
  experiments/results/public_pr_intake_full/public_pr<N>_intake_<DATE>_auto/
    pr_metadata.json          — leaderboard score, author, timestamps, head SHA
    pr_body.md                — full PR description
    archive.zip               — the actual scored archive (if recoverable)
    archive_provenance.json   — sha256, size, source URL, fetch attempts
    source/                   — shallow clone of contestant's fork at head SHA
    INTAKE_LOG.md             — what succeeded, what failed, what's missing

Usage:
    .venv/bin/python tools/fetch_all_public_pr_archives.py \\
        --output-dir experiments/results/public_pr_intake_full \\
        --skip-existing-archives \\
        --max-prs 60

    # Re-fetch only PRs missing archives (don't redo metadata):
    .venv/bin/python tools/fetch_all_public_pr_archives.py \\
        --skip-existing-archives \\
        --refetch-archives-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import read_json, sha256_file, write_json  # noqa: E402

COMMA_REPO = "commaai/comma_video_compression_challenge"


def _repo_relative_display(path: Path) -> str:
    """Render repo-local paths when possible, without failing on relative inputs."""
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _fetch_leaderboard() -> list[dict]:
    """Pull the live README and parse the leaderboard table. Raises on failure."""
    url = f"https://raw.githubusercontent.com/{COMMA_REPO}/master/README.md"
    readme = urllib.request.urlopen(url, timeout=15).read().decode()
    pat = re.compile(r"<td>\s*(\d+\.\d+)\s*</td>\s*<td>\s*([^<\s]+).*?<td>.*?pull/(\d+)", re.S)
    entries = [
        {"score": float(m.group(1)), "name": m.group(2), "pr_number": int(m.group(3))}
        for m in pat.finditer(readme)
    ]
    if not entries:
        raise RuntimeError("leaderboard parse returned 0 entries — README format may have changed")
    return entries


def _gh_api(path: str, raise_on_fail: bool = False) -> dict | list | None:
    """Call gh api. By default returns None on failure (logged); raise_on_fail to raise."""
    try:
        result = subprocess.run(
            ["gh", "api", path], capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        msg = f"gh api {path} failed: {type(e).__name__}: {str(e)[:200]}"
        if raise_on_fail:
            raise RuntimeError(msg) from e
        print(f"  WARN: {msg}", file=sys.stderr)
        return None


_ARCHIVE_URL_PATTERNS = [
    r"https?://github\.com/[^\s)\]]+/releases/download/[^\s)\]]+\.zip",
    r"https?://github\.com/user-attachments/files/[^\s)\]]+\.zip",
    r"https?://github\.com/[^\s)\]]+\.zip",
    r"https?://huggingface\.co/[^\s)\]]+/resolve/[^\s)\]]+\.zip",
    r"https?://(?:www\.)?dropbox\.com/[^\s)\]]+\.zip",
    r"https?://drive\.google\.com/[^\s)\]]+",
]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _extract_archive_urls_from_text(text: str) -> list[str]:
    """Extract direct archive URLs from PR markdown text in encounter order."""
    if not text:
        return []
    urls: list[str] = []
    for pat in _ARCHIVE_URL_PATTERNS:
        for m in re.finditer(pat, text):
            urls.append(m.group(0).rstrip(").,;]>"))
    return _dedupe_preserve_order(urls)


def _extract_archive_url_from_body(body: str) -> str | None:
    """Backward-compatible first-pass helper for callers that expect one URL."""
    urls = _extract_archive_urls_from_text(body)
    return urls[0] if urls else None


def _archive_path_hints_from_body(body: str) -> list[str]:
    """Extract repository-relative archive.zip paths from PR body prose."""
    if not body:
        return []
    candidates: list[str] = []
    for token in re.findall(r"`([^`]*archive\.zip)`", body):
        candidates.append(token.strip())
    for token in re.findall(r"(?<![\w/.-])([A-Za-z0-9_./-]+archive\.zip)", body):
        candidates.append(token.strip())

    paths: list[str] = []
    for candidate in candidates:
        if re.match(r"https?://", candidate):
            continue
        normalized = candidate.strip().strip(".,;:()[]<>")
        if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
            continue
        paths.append(normalized)
    return _dedupe_preserve_order(paths)


def _archive_context_tokens(*texts: str) -> set[str]:
    joined = " ".join(texts).lower().replace("-", "_")
    raw = re.findall(r"[a-z0-9]+", joined)
    stop = {
        "archive",
        "zip",
        "submission",
        "score",
        "pull",
        "pr",
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
        "under",
        "included",
        "github",
        "download",
        "release",
    }
    return {token for token in raw if len(token) >= 3 and token not in stop}


def _rank_archive_urls(urls: list[str], context_tokens: set[str], pr_num: int) -> list[str]:
    def score(index_url: tuple[int, str]) -> tuple[int, int]:
        index, url = index_url
        normalized = unquote(url).lower().replace("-", "_")
        token_hits = sum(1 for token in context_tokens if token in normalized)
        pr_hits = int(f"pr{pr_num}" in normalized or f"pull_{pr_num}" in normalized or f"pull-{pr_num}" in normalized)
        exact_archive = int(normalized.endswith("/archive.zip"))
        return (-(10 * token_hits + 5 * pr_hits + exact_archive), index)

    return [url for _, url in sorted(enumerate(_dedupe_preserve_order(urls)), key=score)]


def _walk_pr_comments_for_archive(pr_num: int) -> list[str]:
    """Read PR issue comments and return direct archive links."""
    comments = _gh_api(f"repos/{COMMA_REPO}/issues/{pr_num}/comments?per_page=100")
    if not comments or not isinstance(comments, list):
        return []
    urls: list[str] = []
    for comment in comments:
        urls.extend(_extract_archive_urls_from_text(comment.get("body") or ""))
    return _dedupe_preserve_order(urls)


def _walk_releases_for_archive(head_repo: str, pr_num: int, body: str, name: str = "") -> list[str]:
    """Second-pass: walk GitHub releases on the head_repo for archive.zip-like assets.
    Returns list of candidate URLs (caller picks/tries each)."""
    candidates = []
    if not head_repo:
        return candidates
    releases = _gh_api(f"repos/{head_repo}/releases?per_page=20")
    if not releases or not isinstance(releases, list):
        return candidates
    for rel in releases:
        for asset in rel.get("assets", []) or []:
            asset_name = (asset.get("name") or "").lower()
            url = asset.get("browser_download_url")
            if not url:
                continue
            # Prefer assets that look like an archive submission
            if asset_name.endswith(".zip") and (
                "archive" in asset_name
                or f"pr{pr_num}" in asset_name
                or f"pull-{pr_num}" in asset_name
                or "submission" in asset_name
            ):
                candidates.append(url)
            elif asset_name.endswith(".zip"):
                candidates.append(url)  # any .zip in releases is a candidate
    return _rank_archive_urls(candidates, _archive_context_tokens(body, name), pr_num)


def _walk_pr_commits_for_lfs(head_repo: str, head_sha: str, body: str = "") -> list[str]:
    """Third-pass: check the PR's head commit for archive.zip files."""
    if not head_repo or not head_sha:
        return []
    candidates = []
    path_hints = set(_archive_path_hints_from_body(body))
    # Look for archive.zip in the tree
    tree = _gh_api(f"repos/{head_repo}/git/trees/{head_sha}?recursive=1")
    if not tree or not isinstance(tree, dict):
        return candidates
    for entry in tree.get("tree", []) or []:
        path = entry.get("path", "")
        if path.endswith("archive.zip") and (path in path_hints or entry.get("size", 0) > 1000):
            # Direct download via raw.githubusercontent
            url = f"https://raw.githubusercontent.com/{head_repo}/{head_sha}/{path}"
            candidates.append(url)
    return _rank_archive_urls(candidates, _archive_context_tokens(body), 0)


def _download_file(url: str, dest: Path, max_bytes: int = 500_000_000) -> tuple[bool, str]:
    """Download with size cap. Returns (success, sha256_or_error)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "comma-pr-archive-fetcher/2.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_length = int(resp.headers.get("Content-Length", "0"))
            if content_length and content_length > max_bytes:
                return False, f"size {content_length} > cap {max_bytes}"
            h = hashlib.sha256()
            written = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        return False, f"exceeded cap {max_bytes} during stream"
                    f.write(chunk)
                    h.update(chunk)
            if written < 100:
                return False, f"suspiciously small ({written} bytes); likely an HTML error page"
            return True, h.hexdigest()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def _try_archive_candidates(candidates: list[str], dest: Path) -> tuple[bool, str, str | None, list[dict]]:
    """Try each candidate URL in order. Returns (success, status, sha256_or_None, attempts)."""
    attempts = []
    for url in candidates:
        ok, result = _download_file(url, dest)
        attempts.append({"url": url, "success": ok, "result": result})
        if ok:
            # Validate it's actually a ZIP
            try:
                with zipfile.ZipFile(dest) as z:
                    members = z.namelist()
                if not members:
                    attempts[-1]["zip_empty"] = True
                    dest.unlink()
                    continue
                attempts[-1]["zip_members"] = members[:5]
                return True, "downloaded", result, attempts
            except zipfile.BadZipFile:
                attempts[-1]["bad_zip"] = True
                dest.unlink(missing_ok=True)
                continue
    return False, "no_candidate_succeeded", None, attempts


def _fetch_one_pr(pr_num: int, score: float, name: str, output_root: Path,
                 skip_existing_archives: bool, refetch_archives_only: bool) -> dict:
    """Fetch all artifacts for one PR. Returns intake summary."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    pr_dir = output_root / f"public_pr{pr_num}_intake_{today}_auto"
    pr_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"# PR #{pr_num} intake — {name} (score {score})",
        f"Started UTC: {datetime.now(UTC).isoformat()}", "",
    ]

    # 1. PR metadata (skip if refetch-archives-only and metadata already exists)
    meta_path = pr_dir / "pr_metadata.json"
    if refetch_archives_only and meta_path.is_file():
        meta = read_json(meta_path)
        log_lines.append("  re-using existing pr_metadata.json")
        # We need raw API meta for head_repo / head_sha — load from what's stored
        head_repo = meta.get("head_repo")
        head_sha = meta.get("head_sha")
        body = (pr_dir / "pr_body.md").read_text() if (pr_dir / "pr_body.md").is_file() else ""
    else:
        api_meta = _gh_api(f"repos/{COMMA_REPO}/pulls/{pr_num}")
        if api_meta is None:
            log_lines.append("FATAL: gh api pull/{pr_num} returned no data")
            (pr_dir / "INTAKE_LOG.md").write_text("\n".join(log_lines))
            return {"pr": pr_num, "status": "fatal_no_metadata"}

        meta = {
            "pr_number": pr_num,
            "leaderboard_score": score,
            "leaderboard_name": name,
            "title": api_meta.get("title"),
            "author": (api_meta.get("user") or {}).get("login"),
            "created_at": api_meta.get("created_at"),
            "merged_at": api_meta.get("merged_at"),
            "closed_at": api_meta.get("closed_at"),
            "state": api_meta.get("state"),
            "head_sha": (api_meta.get("head") or {}).get("sha"),
            "head_repo": ((api_meta.get("head") or {}).get("repo") or {}).get("full_name"),
            "additions": api_meta.get("additions"),
            "deletions": api_meta.get("deletions"),
            "changed_files": api_meta.get("changed_files"),
        }
        write_json(meta_path, meta)
        log_lines.append(f"✓ pr_metadata.json (head_sha={meta['head_sha']}, author={meta['author']})")
        body = api_meta.get("body") or ""
        (pr_dir / "pr_body.md").write_text(body)
        log_lines.append(f"✓ pr_body.md ({len(body)} chars)")
        head_repo = meta["head_repo"]
        head_sha = meta["head_sha"]

    # 2. Archive — multi-pass URL discovery
    archive_dest = pr_dir / "archive.zip"
    if skip_existing_archives and archive_dest.is_file() and archive_dest.stat().st_size > 100:
        sha = sha256_file(archive_dest)
        log_lines.append(f"✓ archive.zip exists ({archive_dest.stat().st_size} bytes, sha256 {sha[:16]}...) — skipping")
        archive_status = "exists_skipped"
        archive_sha = sha
        archive_size = archive_dest.stat().st_size
        attempts = []
    else:
        candidates = []
        # Pass 1: direct URL in body
        for url1 in _extract_archive_urls_from_text(body):
            if url1 not in candidates:
                candidates.append(url1)
                log_lines.append(f"  body URL: {url1}")
        # Pass 2: direct URLs in PR comments
        for url_comment in _walk_pr_comments_for_archive(pr_num):
            if url_comment not in candidates:
                candidates.append(url_comment)
                log_lines.append(f"  comment URL: {url_comment}")
        # Pass 3: archive.zip in head SHA tree, body path hints first
        url_tree_list = _walk_pr_commits_for_lfs(head_repo, head_sha, body)
        for u in url_tree_list:
            if u not in candidates:
                candidates.append(u)
                log_lines.append(f"  in-tree archive: {u}")
        # Pass 4: walk releases on head_repo
        if head_repo:
            url2_list = _walk_releases_for_archive(head_repo, pr_num, body, name)
            for u in url2_list:
                if u not in candidates:
                    candidates.append(u)
                    log_lines.append(f"  head_repo release asset: {u}")
        # Pass 5: walk releases on commaai/main repo
        if not candidates:
            url3_list = _walk_releases_for_archive(COMMA_REPO, pr_num, body, name)
            for u in url3_list:
                if u not in candidates:
                    candidates.append(u)
                    log_lines.append(f"  comma release asset: {u}")

        if not candidates:
            archive_status = "no_url_found_in_any_pass"
            archive_sha = None
            archive_size = None
            attempts = []
            log_lines.append("✗ no archive URL found in body, comments, head SHA tree, head_repo releases, OR comma releases")
        else:
            ok, status, sha, attempts = _try_archive_candidates(candidates, archive_dest)
            if ok:
                archive_size = archive_dest.stat().st_size
                archive_sha = sha
                log_lines.append(f"✓ archive.zip downloaded ({archive_size} bytes, sha256 {sha[:16]}...)")
                archive_status = "downloaded"
            else:
                log_lines.append(f"✗ all {len(candidates)} candidates failed")
                for a in attempts:
                    log_lines.append(f"    {a['url']}: {a['result']}")
                archive_status = status
                archive_sha = None
                archive_size = None

    write_json(
        pr_dir / "archive_provenance.json",
        {
            "candidates_tried": [a["url"] for a in attempts] if attempts else [],
            "attempts": attempts,
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
            "status": archive_status,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    # 3. Source clone (shallow at head SHA) — skip if already cloned
    src_dir = pr_dir / "source"
    src_status = "skipped_exists" if src_dir.is_dir() else "skipped_no_head"
    if head_repo and head_sha and not src_dir.is_dir():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--no-tags",
                 f"https://github.com/{head_repo}.git", str(src_dir)],
                capture_output=True, text=True, timeout=120, check=True,
            )
            subprocess.run(
                ["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", head_sha],
                capture_output=True, text=True, timeout=60, check=False,
            )
            subprocess.run(
                ["git", "-C", str(src_dir), "checkout", "--detach", head_sha],
                capture_output=True, text=True, timeout=30, check=False,
            )
            log_lines.append(f"✓ source/ cloned from {head_repo} at {head_sha[:8]}")
            src_status = "cloned"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log_lines.append(f"✗ source clone failed: {type(e).__name__}: {str(e)[:200]}")
            src_status = "clone_failed"

    log_lines.append("")
    log_lines.append(f"Status: archive={archive_status} source={src_status}")
    (pr_dir / "INTAKE_LOG.md").write_text("\n".join(log_lines))

    return {
        "pr": pr_num,
        "score": score,
        "name": name,
        "status": "complete" if archive_status in ("downloaded", "exists_skipped") else "incomplete",
        "dir": _repo_relative_display(pr_dir),
        "archive_status": archive_status,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "source_status": src_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                       default=REPO / "experiments/results/public_pr_intake_full",
                       help="root dir for per-PR intake folders")
    parser.add_argument("--skip-existing-archives", action="store_true",
                       help="skip downloading if archive.zip already exists in intake dir")
    parser.add_argument("--refetch-archives-only", action="store_true",
                       help="re-use existing pr_metadata.json + pr_body.md; only re-attempt the archive download")
    parser.add_argument("--max-prs", type=int, default=60,
                       help="cap on number of PRs to fetch per run")
    parser.add_argument("--only-prs", type=str, default=None,
                       help="comma-separated PR numbers to fetch (overrides leaderboard walk)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_prs:
        prs = [{"pr_number": int(p), "score": 0.0, "name": "manual"}
               for p in args.only_prs.split(",")]
    else:
        try:
            prs = _fetch_leaderboard()
        except Exception as e:
            print(f"FATAL: leaderboard fetch failed: {e}", file=sys.stderr)
            return 2
        print(f"[fetch] leaderboard has {len(prs)} scored PRs")

    prs.sort(key=lambda p: p["score"])
    if args.max_prs:
        prs = prs[:args.max_prs]

    if args.dry_run:
        for p in prs:
            print(f"  PR #{p['pr_number']:>3}  score={p['score']:.3f}  {p['name']}")
        return 0

    results = []
    for i, p in enumerate(prs, 1):
        print(f"\n[{i}/{len(prs)}] PR #{p['pr_number']} — {p['name']} (score {p['score']})", flush=True)
        try:
            r = _fetch_one_pr(p["pr_number"], p["score"], p["name"],
                            args.output_dir, args.skip_existing_archives,
                            args.refetch_archives_only)
            results.append(r)
            print(f"  → {r['status']} | archive={r.get('archive_status')} src={r.get('source_status')}", flush=True)
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({"pr": p["pr_number"], "status": f"exception: {type(e).__name__}: {e}"})

    summary = {
        "schema": "fetch_all_public_pr_archives_v2",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "total_attempted": len(prs),
        "n_complete": sum(1 for r in results if r.get("status") == "complete"),
        "n_with_archive": sum(1 for r in results if r.get("archive_sha256")),
        "n_with_source": sum(1 for r in results if r.get("source_status") == "cloned" or r.get("source_status") == "skipped_exists"),
        "needs_manual_triage": [
            {"pr": r["pr"], "score": r.get("score"), "name": r.get("name"),
             "reason": r.get("archive_status"),
             "dir": r.get("dir")}
            for r in results if not r.get("archive_sha256")
        ],
        "results": results,
    }
    summary_path = args.output_dir / "FETCH_SUMMARY.json"
    write_json(summary_path, summary)
    print(f"\n[fetch] SUMMARY → {_repo_relative_display(summary_path)}")
    print(f"  attempted: {summary['total_attempted']}")
    print(f"  complete:  {summary['n_complete']}")
    print(f"  with archive: {summary['n_with_archive']}")
    print(f"  with source:  {summary['n_with_source']}")
    print(f"  needs manual triage: {len(summary['needs_manual_triage'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
