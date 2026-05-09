#!/usr/bin/env python3
"""Gate 5 — Runtime closure gate.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #5.

Rule: run the exact contest inflate signature in a clean environment
**before** dispatch (or before treating a public packet as method
evidence). Dependency closure failures (missing ``brotli``, wrong wrapper
signatures, hidden sidecars, local paths, CPU/CUDA mismatches) MUST be
classified as runtime blockers, not method negatives.

Detection (static):
  Two complementary scans:

  1. Build-manifest scan: any ``build_manifest.json`` that records a
     ``submission_dir_relpath`` AND claims ``ready_for_exact_eval_dispatch=
     true`` (or ``score_claim=true``) MUST also record:
       * ``runtime_manifest`` non-empty (reference to runtime tree
         description), OR
       * ``runtime_closure_verified`` true with non-empty
         ``runtime_closure_evidence``, OR
       * ``inflate_smoke_log`` non-empty (path to clean-env inflate log).

  2. Evidence-row scan: any row in a canonical evidence ledger that
     marks a public PR replay (``source`` contains
     ``public_pr<number>``) and claims a contest-CUDA negative MUST
     classify the cause: a row with
     ``contest_dispatch_verdict=measured_config_retired_*`` for a
     public-PR row must include a ``failure_class`` field whose value
     is one of:
       * ``runtime_blocker_dependency_missing``
       * ``runtime_blocker_wrapper_signature``
       * ``runtime_blocker_hidden_sidecar``
       * ``runtime_blocker_local_path``
       * ``runtime_blocker_cpu_cuda_mismatch``
       * ``method_negative`` (explicit "method actually failed")

     This forces public-PR replay failures to disambiguate runtime vs
     method, as the audit cites PR106 as a missing-brotli runtime
     failure that was almost classified as method evidence.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

EVIDENCE_FILES: tuple[str, ...] = (
    "reports/cathedral_autopilot_evidence.jsonl",
    "reports/raw/pr101_omega_opt_evidence.jsonl",
)

PUBLIC_PR_PATTERN = re.compile(r"public_pr\d+", re.IGNORECASE)

RECOGNIZED_FAILURE_CLASSES: tuple[str, ...] = (
    "runtime_blocker_dependency_missing",
    "runtime_blocker_wrapper_signature",
    "runtime_blocker_hidden_sidecar",
    "runtime_blocker_local_path",
    "runtime_blocker_cpu_cuda_mismatch",
    "runtime_blocker_other",
    "method_negative",
)


def _string_path_exists(repo: Path, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if text.startswith(("external:", "sha256:", "inline:")):
        return True
    path = Path(text)
    if not path.is_absolute():
        path = repo / path
    return path.exists()


def _nested_get(mapping: dict, *keys: str) -> object:
    cur: object = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _has_symlink_parent(repo: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return False
    cur = repo
    for part in rel.parts[:-1]:
        cur = cur / part
        if cur.is_symlink():
            return True
    return False


def _git_ignored_relpaths(repo: Path, relpaths: list[str]) -> set[str]:
    """Return relpaths ignored by git, excluding tracked force-added files."""
    if not relpaths or not (repo / ".git").exists():
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=repo,
            input="\n".join(relpaths) + "\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode not in (0, 1):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _manifest_claims_dispatch(manifest: dict) -> bool:
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        return True
    if manifest.get("score_claim") is True:
        return True
    return str(manifest.get("contest_dispatch_verdict", "")).lower() in ("positive", "frontier")


def _manifest_has_runtime_closure(manifest: dict, repo: Path) -> bool:
    rm = manifest.get("runtime_manifest")
    if _string_path_exists(repo, rm):
        return True
    if isinstance(rm, (dict, list)) and len(rm) > 0:
        return True
    for nested_manifest in (
        _nested_get(manifest, "provenance", "inflate_runtime_manifest"),
        _nested_get(manifest, "eval_data", "provenance", "inflate_runtime_manifest"),
        manifest.get("inflate_runtime_manifest"),
    ):
        if isinstance(nested_manifest, (dict, list)) and len(nested_manifest) > 0:
            return True
    if manifest.get("runtime_closure_verified") is True and (
        _string_path_exists(repo, manifest.get("runtime_closure_evidence"))
    ):
        return True
    return _string_path_exists(repo, manifest.get("inflate_smoke_log"))


def _scan_build_manifests(
    repo: Path,
    *,
    include_ignored_results: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    patterns = (
        "experiments/results/*/build_manifest.json",
        "experiments/results/*/*/build_manifest.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(repo.glob(pattern))
    relpaths_by_path = {
        path: path.relative_to(repo).as_posix()
        for path in paths
        if not _has_symlink_parent(repo, path)
    }
    ignored = (
        set()
        if include_ignored_results
        else _git_ignored_relpaths(repo, list(relpaths_by_path.values()))
    )
    for path, relpath in relpaths_by_path.items():
        if relpath in ignored:
            continue
        if "public_pr" in relpath and "intake" in relpath:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        if not _manifest_claims_dispatch(manifest):
            continue
        if _manifest_has_runtime_closure(manifest, repo):
            continue
        findings.append(
            Finding(
                file_rel=relpath,
                line_number=0,
                technique=str(
                    manifest.get(
                        "lane_id", manifest.get("technique", "<unknown>")
                    )
                ),
                reason=(
                    "build_manifest claims dispatch readiness "
                    "(ready_for_exact_eval_dispatch=true OR "
                    "score_claim=true OR positive verdict) but "
                    "lacks runtime_manifest, "
                    "runtime_closure_verified+evidence, or "
                    "inflate_smoke_log. Run inflate in clean env "
                    "BEFORE dispatch. Gate 5 (runtime closure)."
                ),
            )
        )
    return findings


def _scan_evidence_for_public_pr_negatives(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel in EVIDENCE_FILES:
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            source = str(row.get("source", ""))
            if not PUBLIC_PR_PATTERN.search(source):
                continue
            verdict = str(row.get("contest_dispatch_verdict", "")).lower()
            status = str(row.get("measured_config_status", "")).lower()
            negative = ("retired" in verdict or "negative" in verdict
                        or "retired" in status or "negative" in status)
            if not negative:
                continue
            failure_class = str(row.get("failure_class", "")).lower()
            if failure_class in RECOGNIZED_FAILURE_CLASSES:
                continue
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        "public-PR replay row marked negative/retired "
                        "but missing failure_class disambiguation "
                        "(must be one of "
                        f"{','.join(RECOGNIZED_FAILURE_CLASSES)}). "
                        "Don't conflate runtime blockers with method "
                        "negatives. Gate 5 (runtime closure)."
                    ),
                )
            )
    return findings


def scan(
    repo_root: Path | None = None,
    *,
    include_ignored_results: bool = False,
) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    findings.extend(
        _scan_build_manifests(repo, include_ignored_results=include_ignored_results)
    )
    findings.extend(_scan_evidence_for_public_pr_negatives(repo))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--include-ignored-results",
        action="store_true",
        help=(
            "also scan git-ignored experiments/results build manifests; "
            "default preflight scans durable committed/unignored custody surfaces"
        ),
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo, include_ignored_results=args.include_ignored_results)
    if findings:
        print(
            f"[gate5-runtime-closure] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel}:{f.line_number} technique={f.technique}: "
                f"{f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate5-runtime-closure] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
