#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical pre-dispatch codex adversarial-review wrapper - Catalog #271.

Operator-approved 2026-05-15: wire codex adversarial-review automation into
``tools/operator_authorize.py`` BEFORE every paid Modal/Lightning/Vast.ai
dispatch >$1 estimated cost. The codex review of Z3-G1 caught F1+F2 BEFORE
FULL CUDA dispatched but AFTER $0.59 smoke spend - this gate moves the review
BEFORE smoke.

Sister of:
- Catalog #243 (local pre-deploy harness): same canonical insertion point in
  ``tools/operator_authorize.py::_run_local_pre_deploy_check``.
- Catalog #167 (smoke-before-full): same dispatch-wrapper canonical-routing META.
- Catalog #199 (paired-env operator bypass discipline): same paired-env pattern.
- Catalog #245 (Modal call-id ledger): same fcntl-locked JSONL append-only store
  pattern.

Cache schema (.omx/state/codex_pre_dispatch_review_cache.jsonl):
  Append-only JSONL keyed on a composite ``(git_HEAD_sha, recipe_sha,
  trainer_sha)`` hash. Each row carries the verdict + findings + cache
  metadata. TTL: 1h (3600 sec) per entry. Concurrent claim safety via
  fcntl.flock(LOCK_EX) on .omx/state/.codex_pre_dispatch_review_cache.lock.

Verdict taxonomy (parsed from codex companion output):
  - approve         : No findings; safe to dispatch.
  - advisory        : Findings exist but none are CRITICAL/HIGH; logs but
                      does not block.
  - needs-attention : >=1 HIGH finding; refuses dispatch unless paired-env
                      bypass.
  - no-ship         : >=1 CRITICAL finding; refuses dispatch unless paired-env
                      bypass.
  - invocation-error: codex companion timed out / crashed / exited nonzero
                      WITHOUT producing severity-tagged findings; the review
                      did not complete. Per Catalog #281 (codex bfa2p1uex F1
                      fail-closed-on-companion-failure self-protection):
                      refuses dispatch unless paired-env bypass.

Paired-env bypass (per CLAUDE.md "Comment-only contracts are FORBIDDEN" +
Catalog #199 paired-env discipline):
  OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1
  OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE=<text>

Bare ``OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1`` (without paired
rationale) raises ``SystemExit(13)`` per Catalog #199 sister rule.

Cost gate (avoid burning codex tokens on cheap smokes):
  Only invoke codex review when recipe's
  ``cost_band.hand_calibrated_fallback_p50_usd`` (or computed band p50) > $1.
  Below threshold: log advisory, do not block.

Usage::

    .venv/bin/python tools/run_codex_review_for_dispatch.py \\
        --trainer experiments/train_substrate_X.py \\
        --recipe substrate_X_modal_t4_dispatch \\
        --estimated-cost-usd 5.00 \\
        --json-out /tmp/codex_review.json

Exit codes:
  0 = approve OR advisory (safe to dispatch) OR paired-env bypass active
  1 = needs-attention OR no-ship (refuse dispatch; codex found bugs)
  2 = invocation-error (refuse dispatch; codex companion timeout/crash/nonzero
      rc with no findings; review did not actually run; Catalog #281)
 13 = paired-env bypass invariant violated
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Canonical paths + constants
# ---------------------------------------------------------------------------

CACHE_PATH = REPO_ROOT / ".omx" / "state" / "codex_pre_dispatch_review_cache.jsonl"
CACHE_LOCK_PATH = REPO_ROOT / ".omx" / "state" / ".codex_pre_dispatch_review_cache.lock"
CACHE_TTL_SECONDS = 3600  # 1 hour

CODEX_COMPANION_SCRIPT_REL = (
    ".claude/plugins/cache/openai-codex/codex/1.0.3/scripts/codex-companion.mjs"
)
CODEX_COMPANION_SCRIPT = str(Path.home() / CODEX_COMPANION_SCRIPT_REL)
CODEX_COMPANION_SCRIPT_DISPLAY = f"~/{CODEX_COMPANION_SCRIPT_REL}"

# Cost gate threshold: don't burn codex tokens on cheap smokes
COST_GATE_THRESHOLD_USD = 1.00

# Verdict canonical set
VERDICTS_SAFE = frozenset({"approve", "advisory"})
VERDICTS_BLOCKING = frozenset({"needs-attention", "no-ship"})
# Catalog #281 — codex bfa2p1uex F1: invocation-error is a NEW verdict that
# represents companion timeout / crash / nonzero rc with no severity tokens.
# Treated as BLOCKING by default (CLI exit=2) because the codex review did
# not actually run; paired-env bypass per Catalog #199 still applies.
VERDICT_INVOCATION_ERROR = "invocation-error"
VERDICTS_ALL = VERDICTS_SAFE | VERDICTS_BLOCKING | frozenset({VERDICT_INVOCATION_ERROR})
VERDICT_RANK = {
    "approve": 0,
    "advisory": 1,
    "needs-attention": 2,
    "no-ship": 3,
    VERDICT_INVOCATION_ERROR: 4,
}

# Schema version for the cache JSONL rows
CACHE_SCHEMA_VERSION = "codex_pre_dispatch_review_cache_v1"

# Paired-env bypass tokens per Catalog #199 sister rule
BYPASS_VERDICT_ENV = "OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT"
BYPASS_RATIONALE_ENV = "OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE"

# Severity-to-verdict mapping. Higher severity findings dominate.
SEVERITY_CRITICAL_TOKENS = ("CRITICAL", "BLOCKER", "NO-SHIP", "no-ship")
SEVERITY_HIGH_TOKENS = ("HIGH", "high")
SEVERITY_LOW_TOKENS = ("LOW", "MEDIUM", "advisory", "ADVISORY")
_SEVERITY_SUFFIX_RE = r"(?:\s+(?:F?\d+|[A-Z]\d+))?\s*[:\-]"
SEVERITY_CRITICAL_RE = re.compile(
    rf"\b(?:CRITICAL|BLOCKER|NO[-_ ]SHIP)\b{_SEVERITY_SUFFIX_RE}",
    re.IGNORECASE,
)
SEVERITY_HIGH_RE = re.compile(rf"\bHIGH\b{_SEVERITY_SUFFIX_RE}", re.IGNORECASE)
SEVERITY_LOW_RE = re.compile(
    rf"\b(?:LOW|MEDIUM|ADVISORY)\b{_SEVERITY_SUFFIX_RE}",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CodexReviewResult:
    """Structured codex adversarial-review result."""

    verdict: str  # one of VERDICTS_ALL
    findings: list[str]
    cache_hit: bool
    cache_age_sec: int  # 0 if not from cache; positive if cached
    cache_key: str  # composite hash for traceability
    raw_output_excerpt: str  # first ~2000 chars of codex output for logs
    invoked_at_utc: str  # when this result was generated/cached
    elapsed_sec: float  # wall-clock for actual codex invocation (0 if cache hit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "verdict": self.verdict,
            "findings": list(self.findings),
            "cache_hit": self.cache_hit,
            "cache_age_sec": self.cache_age_sec,
            "cache_key": self.cache_key,
            "raw_output_excerpt": self.raw_output_excerpt,
            "invoked_at_utc": self.invoked_at_utc,
            "elapsed_sec": self.elapsed_sec,
        }


# ---------------------------------------------------------------------------
# fcntl-locked JSONL cache (sister pattern to Catalog #128/#131/#245)
# ---------------------------------------------------------------------------


@contextmanager
def _cache_lock() -> Iterator[None]:
    """fcntl.flock(LOCK_EX) on the canonical cache lock path."""
    CACHE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_LOCK_PATH.touch(exist_ok=True)
    fd = os.open(str(CACHE_LOCK_PATH), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _utc_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _utc_epoch() -> int:
    """Return current UTC epoch seconds."""
    return int(time.time())


def _compute_cache_key(
    git_head_sha: str,
    recipe_sha: str,
    trainer_sha: str,
    dirty_tree_fingerprint: str = "",
    untracked_relevant_fingerprint: str = "",
) -> str:
    """Composite SHA-256 of the inputs (truncated to 16 hex chars).

    Catalog #282 — codex bfa2p1uex F2 cache-key-includes-dirty-tree
    self-protection. Pre-fix, the key was only ``(git_head_sha, recipe_sha,
    trainer_sha)``. Codex companion is invoked with ``--scope working-tree``
    so dirty changes in shared helpers (``src/tac/preflight.py``,
    ``tools/*.py``, etc.) and untracked relevant files materially change
    what the review evaluates — but were INVISIBLE to the cache key. A
    cached approve from 1h ago could authorize a materially different
    working tree without rerunning review.
    """
    h = hashlib.sha256()
    h.update(git_head_sha.encode("utf-8"))
    h.update(b"|")
    h.update(recipe_sha.encode("utf-8"))
    h.update(b"|")
    h.update(trainer_sha.encode("utf-8"))
    h.update(b"|")
    h.update(dirty_tree_fingerprint.encode("utf-8"))
    h.update(b"|")
    h.update(untracked_relevant_fingerprint.encode("utf-8"))
    return h.hexdigest()[:16]


# Catalog #282 — relevant-file globs for untracked-fingerprint scan.
# Files matching these globs (relative to repo root) participate in the
# untracked-relevant fingerprint so untracked dispatch-critical changes
# invalidate the cache.
_UNTRACKED_RELEVANT_GLOBS: tuple[str, ...] = (
    "tools/*.py",
    "src/tac/*.py",
    "src/tac/preflight.py",
    "src/tac/substrates/**/*.py",
    "CLAUDE.md",
    "scripts/remote_lane_*.sh",
    "experiments/train_substrate_*.py",
)


def _git_dirty_tree_fingerprint(repo_root: Path) -> str:
    """Return SHA-256 (truncated 16 chars) of git diff against HEAD.

    Catalog #282 — captures both staged + unstaged changes via
    ``git diff --binary --cached HEAD`` then ``git diff --binary``. Empty
    string if git unavailable.
    """
    try:
        h = hashlib.sha256()
        for args in (
            ["git", "diff", "--binary", "--cached", "HEAD"],
            ["git", "diff", "--binary"],
        ):
            r = subprocess.run(
                args,
                cwd=str(repo_root),
                capture_output=True,
                timeout=15,
            )
            if r.returncode == 0:
                h.update(r.stdout or b"")
            h.update(b"|")
        return h.hexdigest()[:16]
    except (subprocess.SubprocessError, OSError):
        return "no-git-diff"


def _git_untracked_relevant_fingerprint(repo_root: Path) -> str:
    """Return SHA-256 (truncated 16 chars) of untracked relevant file paths+sizes.

    Catalog #282 — captures the set of untracked files matching
    ``_UNTRACKED_RELEVANT_GLOBS`` (path + size + content hash). An untracked
    helper that materially changes dispatch behavior must invalidate the
    cache. Empty string if git unavailable.
    """
    try:
        r = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return "no-git-untracked"
        untracked = sorted(line.strip() for line in r.stdout.splitlines() if line.strip())
    except (subprocess.SubprocessError, OSError):
        return "no-git-untracked"

    # Filter by canonical relevant globs
    import fnmatch as _fnmatch

    relevant: list[str] = []
    for rel in untracked:
        for pat in _UNTRACKED_RELEVANT_GLOBS:
            if _fnmatch.fnmatch(rel, pat):
                relevant.append(rel)
                break

    h = hashlib.sha256()
    for rel in relevant:
        path = repo_root / rel
        try:
            size = path.stat().st_size if path.exists() else 0
        except OSError:
            size = 0
        h.update(rel.encode("utf-8"))
        h.update(b"|")
        h.update(str(size).encode("utf-8"))
        h.update(b"|")
        try:
            if path.is_file() and size < 5_000_000:  # skip huge artifacts
                h.update(path.read_bytes())
        except OSError:
            pass
        h.update(b"\n")
    return h.hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """SHA-256 of file contents (truncated to 16 hex chars)."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _git_head_sha(repo_root: Path) -> str:
    """Resolve git HEAD SHA via subprocess (truncated to 16 hex chars)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:16]
    except (subprocess.SubprocessError, OSError):
        pass
    return "no-git"


def lookup_cached_review(
    cache_key: str,
    *,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    cache_path: Path | None = None,
) -> CodexReviewResult | None:
    """Look up a cached review by composite key; return None if missing/expired.

    Reads the entire JSONL file under the lock and returns the LATEST row
    matching the key whose age (now - invoked_at_epoch) is within TTL.
    """
    cp = cache_path or CACHE_PATH
    with _cache_lock():
        if not cp.exists():
            return None
        try:
            text = cp.read_text(encoding="utf-8")
        except OSError:
            return None
        latest_row: dict[str, Any] | None = None
        latest_epoch = -1
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("cache_key") != cache_key:
                continue
            row_epoch = int(row.get("invoked_at_epoch", 0))
            if row_epoch > latest_epoch:
                latest_epoch = row_epoch
                latest_row = row
        if latest_row is None:
            return None
        now = _utc_epoch()
        age = now - latest_epoch
        # TTL semantics: ttl_seconds=0 means "never reuse" (strict).
        # Otherwise reuse iff age < ttl_seconds.
        if ttl_seconds <= 0 or age >= ttl_seconds:
            return None
        verdict = latest_row.get("verdict", "advisory")
        if verdict not in VERDICTS_ALL:
            verdict = "advisory"
        return CodexReviewResult(
            verdict=verdict,
            findings=list(latest_row.get("findings", [])),
            cache_hit=True,
            cache_age_sec=age,
            cache_key=cache_key,
            raw_output_excerpt=str(latest_row.get("raw_output_excerpt", "")),
            invoked_at_utc=str(latest_row.get("invoked_at_utc", _utc_iso())),
            elapsed_sec=float(latest_row.get("elapsed_sec", 0.0)),
        )


def append_cached_review(
    result: CodexReviewResult,
    *,
    cache_path: Path | None = None,
) -> None:
    """Append a row to the cache JSONL (atomic via fcntl + append mode)."""
    cp = cache_path or CACHE_PATH
    cp.parent.mkdir(parents=True, exist_ok=True)
    row = result.as_dict()
    # Add invoked_at_epoch for fast TTL filtering on read
    row["invoked_at_epoch"] = _utc_epoch()
    serialized = json.dumps(row, sort_keys=True) + "\n"
    with _cache_lock(), cp.open("a", encoding="utf-8") as fh:
        # Append-only - never mutate prior rows (HISTORICAL_PROVENANCE per
        # Catalog #110/#113).
        fh.write(serialized)


# ---------------------------------------------------------------------------
# Codex companion invocation + verdict parsing
# ---------------------------------------------------------------------------


def codex_companion_available(*, script_path: str | None = None) -> bool:
    """Return True if the codex companion script is installed + node is in PATH."""
    target = Path(script_path or CODEX_COMPANION_SCRIPT)
    if not target.exists():
        return False
    return shutil.which("node") is not None


def _display_companion_script_path(script_path: str | None) -> str:
    return script_path if script_path is not None else CODEX_COMPANION_SCRIPT_DISPLAY


def parse_verdict_from_codex_output(output: str) -> tuple[str, list[str]]:
    """Parse the codex adversarial-review output to extract verdict + findings.

    Heuristics:
      1. Scan for explicit verdict markers (``VERDICT: <kind>``).
      2. Otherwise, classify by highest severity finding present:
         - any CRITICAL/BLOCKER token -> 'no-ship'
         - any HIGH token -> 'needs-attention'
         - any LOW/MEDIUM/advisory -> 'advisory'
         - none of the above -> 'approve'
      3. Findings: each line containing a severity token is captured (truncated
         to 200 chars per finding, max 50 findings).

    Returns (verdict, findings).
    """
    if not output:
        return ("approve", [])

    text = output

    # Step 1: explicit verdict marker
    explicit_re = re.compile(
        r"verdict\s*[:=]\s*(approve|advisory|needs[-_ ]attention|no[-_ ]ship)",
        re.IGNORECASE,
    )
    explicit_match = explicit_re.search(text)
    explicit_verdict: str | None = None
    if explicit_match:
        raw = explicit_match.group(1).lower()
        normalized = raw.replace("_", "-").replace(" ", "-")
        if normalized in VERDICTS_ALL:
            explicit_verdict = normalized

    # Step 2: collect findings by explicit severity markers. Do not treat
    # prose such as "HIGH frequency" as a blocking finding unless it has the
    # finding-style delimiter.
    findings: list[str] = []
    has_critical = False
    has_high = False
    has_low = False

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if SEVERITY_CRITICAL_RE.search(line_stripped):
            has_critical = True
            findings.append(line_stripped[:200])
        elif SEVERITY_HIGH_RE.search(line_stripped):
            has_high = True
            findings.append(line_stripped[:200])
        elif SEVERITY_LOW_RE.search(line_stripped):
            has_low = True
            findings.append(line_stripped[:200])
        if len(findings) >= 50:
            break

    # Step 3: derive verdict
    severity_verdict: str | None = None
    if has_critical:
        severity_verdict = "no-ship"
    elif has_high:
        severity_verdict = "needs-attention"
    elif has_low:
        severity_verdict = "advisory"

    if explicit_verdict:
        if severity_verdict and VERDICT_RANK[severity_verdict] > VERDICT_RANK[explicit_verdict]:
            return (severity_verdict, findings)
        return (explicit_verdict, findings)
    if severity_verdict:
        return (severity_verdict, findings)
    return ("approve", findings)


def invoke_codex_adversarial_review(
    *,
    trainer_path: Path,
    recipe_name: str,
    repo_root: Path,
    timeout_seconds: int = 600,
    script_path: str | None = None,
) -> tuple[str, float, int]:
    """Invoke ``codex-companion.mjs adversarial-review`` synchronously.

    Returns ``(stdout_text, elapsed_sec, returncode)``. The caller is
    responsible for parsing the verdict.

    Uses ``--wait`` flag so this call blocks until the review is complete
    (sister pattern: the canonical ``/codex:adversarial-review`` skill call).
    Focus text is built from the trainer + recipe context.

    Cost: ~30-90s per invocation depending on diff size.
    """
    target_script = script_path or CODEX_COMPANION_SCRIPT
    focus_text = (
        f"Pre-dispatch adversarial review for {recipe_name} "
        f"(trainer: {trainer_path.name}). "
        f"Look for: archive grammar drift, archive member name mismatches, "
        f"auth-eval CLI flag drift, scorer-at-inflate violations, "
        f"non-canonical inflate device, missing required-input file paths, "
        f"NotImplementedError in _full_main, deterministic-zip violations, "
        f"phantom-score directory naming. Verdict: approve | advisory | "
        f"needs-attention | no-ship."
    )
    cmd = [
        "node",
        str(target_script),
        "adversarial-review",
        "--wait",
        "--scope",
        "working-tree",
        focus_text,
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return (
            f"[codex-companion timeout after {timeout_seconds}s]",
            elapsed,
            124,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        elapsed = time.time() - start
        return (f"[codex-companion invocation error: {exc}]", elapsed, 2)

    elapsed = time.time() - start
    # Combine stdout + stderr for verdict parsing (codex may emit findings
    # to either stream).
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return (combined, elapsed, result.returncode)


# ---------------------------------------------------------------------------
# Top-level public API
# ---------------------------------------------------------------------------


def run_codex_review_for_dispatch(
    *,
    trainer_path: Path,
    recipe_path: Path,
    repo_root: Path | None = None,
    estimated_cost_usd: float = 0.0,
    cost_gate_threshold_usd: float = COST_GATE_THRESHOLD_USD,
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    timeout_seconds: int = 600,
    cache_path: Path | None = None,
    skip_cache: bool = False,
    no_cache_for_paid_dispatch: bool = False,
    script_path: str | None = None,
) -> CodexReviewResult:
    """Run a pre-dispatch codex adversarial review with caching + cost gate.

    Cost gate: if ``estimated_cost_usd <= cost_gate_threshold_usd``, returns
    an ``advisory`` result without invoking codex (don't burn tokens on cheap
    smokes). Bug class anchor: cheap smokes are cheap to re-run; the gate
    pays off on full-run + multi-thousand-dollar dispatches.

    Cache lookup: composite SHA-256 of ``(git_HEAD_sha, recipe_sha,
    trainer_sha)``; TTL ``cache_ttl_seconds`` (1h default). Cache hits are
    free + fast.

    Returns a structured ``CodexReviewResult``. Caller decides whether the
    verdict is blocking via ``result.verdict in VERDICTS_BLOCKING``.
    """
    root = Path(repo_root or REPO_ROOT)

    # Cost gate (avoid burning codex tokens on cheap smokes)
    if estimated_cost_usd <= cost_gate_threshold_usd:
        return CodexReviewResult(
            verdict="advisory",
            findings=[
                f"[cost-gate] estimated cost ${estimated_cost_usd:.2f} "
                f"<= threshold ${cost_gate_threshold_usd:.2f}; codex review "
                f"skipped (run with higher cost to invoke)"
            ],
            cache_hit=False,
            cache_age_sec=0,
            cache_key="cost-gated",
            raw_output_excerpt="",
            invoked_at_utc=_utc_iso(),
            elapsed_sec=0.0,
        )

    # Compute cache key. Catalog #282 — codex bfa2p1uex F2: include
    # dirty-tree + untracked-relevant fingerprints so working-tree changes in
    # shared helpers (preflight, runtime, untracked dispatch-critical files)
    # invalidate the cache. Codex companion is invoked with
    # ``--scope working-tree``; the cache key MUST mirror that scope.
    git_sha = _git_head_sha(root)
    recipe_sha = _file_sha256(recipe_path)
    trainer_sha = _file_sha256(trainer_path)
    dirty_fingerprint = _git_dirty_tree_fingerprint(root)
    untracked_fingerprint = _git_untracked_relevant_fingerprint(root)
    cache_key = _compute_cache_key(
        git_sha,
        recipe_sha,
        trainer_sha,
        dirty_tree_fingerprint=dirty_fingerprint,
        untracked_relevant_fingerprint=untracked_fingerprint,
    )

    # Cache lookup. Catalog #282 — for paid-dispatch path, skip cache by
    # default (operator-authorize calls with no_cache_for_paid_dispatch=True
    # via CLI flag) so a cached approve from minutes ago cannot authorize
    # a now-different working tree.
    if not skip_cache and not no_cache_for_paid_dispatch:
        cached = lookup_cached_review(
            cache_key,
            ttl_seconds=cache_ttl_seconds,
            cache_path=cache_path,
        )
        if cached is not None:
            return cached

    # Invoke codex
    if not codex_companion_available(script_path=script_path):
        # Refuse with needs-attention so the operator notices the missing
        # tooling rather than silently dispatching without review.
        return CodexReviewResult(
            verdict="needs-attention",
            findings=[
                "[codex-companion-missing] codex companion script not found "
                f"at {_display_companion_script_path(script_path)} OR `node` not in "
                'PATH; install per CLAUDE.md "Codex CLI invocation" section '
                "before paid dispatch"
            ],
            cache_hit=False,
            cache_age_sec=0,
            cache_key=cache_key,
            raw_output_excerpt="",
            invoked_at_utc=_utc_iso(),
            elapsed_sec=0.0,
        )

    output, elapsed, rc = invoke_codex_adversarial_review(
        trainer_path=trainer_path,
        recipe_name=recipe_path.stem,
        repo_root=root,
        timeout_seconds=timeout_seconds,
        script_path=script_path,
    )

    # Catalog #281 — codex bfa2p1uex F1: rc-first verdict promotion.
    # When the companion times out (rc=124), crashes (rc=2), or otherwise
    # exits nonzero WITHOUT producing a severity-tagged finding the parser
    # can recognize, the previous logic returned `approve` (because the
    # bracketed error string lacked CRITICAL/HIGH/LOW tokens), letting paid
    # dispatch continue with NO actual adversarial review. The fix promotes
    # nonzero rc to a blocking verdict BEFORE parsing text. Paired-env
    # bypass per Catalog #199 still applies via check_paired_bypass_env_or_exit.
    if rc != 0:
        verdict = VERDICT_INVOCATION_ERROR
        findings = [
            f"[codex-companion-rc] FAIL-CLOSED nonzero rc={rc} "
            f"(verdict={VERDICT_INVOCATION_ERROR}; codex review did not "
            "complete; per Catalog #281 fail-closed-on-companion-failure "
            "paired-env bypass via OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_"
            "VERDICT=1 + RATIONALE=<text> if intentional)",
        ]
        # If the companion DID surface findings (e.g. partial output before
        # timeout), preserve them so the operator can still see signal.
        _, parsed_findings = parse_verdict_from_codex_output(output)
        if parsed_findings:
            findings.extend(parsed_findings[:10])
    else:
        verdict, findings = parse_verdict_from_codex_output(output)

    result = CodexReviewResult(
        verdict=verdict,
        findings=findings,
        cache_hit=False,
        cache_age_sec=0,
        cache_key=cache_key,
        raw_output_excerpt=output[:2000],
        invoked_at_utc=_utc_iso(),
        elapsed_sec=elapsed,
    )

    # Cache the result for future invocations
    try:
        append_cached_review(result, cache_path=cache_path)
    except OSError:
        # Cache write failure is non-fatal; don't block dispatch on it.
        pass

    return result


# ---------------------------------------------------------------------------
# Paired-env bypass discipline (Catalog #199 sister pattern)
# ---------------------------------------------------------------------------


def check_paired_bypass_env_or_exit() -> bool:
    """Return True if the paired bypass env vars are SET + valid.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + Catalog #199
    paired-env discipline: ``OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1``
    REQUIRES paired ``OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE=<text>``.

    Returns:
      True  if bypass active (both env vars set + rationale non-empty)
      False if neither env var set (no bypass requested)

    Raises ``SystemExit(13)`` if bypass intent set without rationale.
    """
    intent = os.environ.get(BYPASS_VERDICT_ENV, "").strip()
    rationale = os.environ.get(BYPASS_RATIONALE_ENV, "").strip()
    if intent != "1":
        return False
    if not rationale:
        # Print the rationale-required message then exit with rc=13 so the
        # caller (operator_authorize.py) can distinguish paired-env-bypass
        # invariant violations from other failure classes.
        print(
            f"[run-codex-review] FATAL: {BYPASS_VERDICT_ENV}=1 requires "
            f"paired {BYPASS_RATIONALE_ENV}=<text> per CLAUDE.md "
            "paired-env discipline + Catalog #199 sister rule. Refusing "
            "to bypass codex review without rationale.",
            file=sys.stderr,
        )
        raise SystemExit(13)
    print(
        f"[run-codex-review] [CODEX REVIEW BYPASS ACTIVE] reason={rationale!r}",
        file=sys.stderr,
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-dispatch codex adversarial-review wrapper (Catalog #271). "
            "Returns 0 on safe verdict (approve/advisory), 1 on blocking "
            "verdict (needs-attention/no-ship), 2 on invocation error, "
            "13 on paired-env bypass invariant violation."
        )
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        required=True,
        help="Path to the trainer .py file to be dispatched",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to the recipe .yaml file under .omx/operator_authorize_recipes/",
    )
    parser.add_argument(
        "--estimated-cost-usd",
        type=float,
        default=0.0,
        help=(
            f"Estimated dispatch cost in USD; codex review only invoked when "
            f">${COST_GATE_THRESHOLD_USD:.2f} (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--cost-gate-threshold-usd",
        type=float,
        default=COST_GATE_THRESHOLD_USD,
        help=f"Override cost gate threshold (default: ${COST_GATE_THRESHOLD_USD:.2f})",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=CACHE_TTL_SECONDS,
        help=f"Cache TTL in seconds (default: {CACHE_TTL_SECONDS})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Codex companion invocation timeout (default: 600)",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Bypass cache lookup; always invoke codex",
    )
    parser.add_argument(
        "--no-cache-for-paid-dispatch",
        action="store_true",
        help=(
            "Disable cache reuse for paid-dispatch path (Catalog #282). "
            "Set by tools/operator_authorize.py before paid Modal/Lightning "
            "dispatch so a stale cached approve cannot authorize a "
            "different working tree."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write structured result JSON to this path",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override cache JSONL path (default: .omx/state/codex_pre_dispatch_review_cache.jsonl)",
    )
    parser.add_argument(
        "--script-path",
        type=str,
        default=None,
        help="Override codex companion script path (for testing)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo root for git HEAD resolution (testing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # Check paired-env bypass first (raises SystemExit(13) on bare intent)
    bypass_active = check_paired_bypass_env_or_exit()

    result = run_codex_review_for_dispatch(
        trainer_path=args.trainer,
        recipe_path=args.recipe,
        repo_root=args.repo_root,
        estimated_cost_usd=args.estimated_cost_usd,
        cost_gate_threshold_usd=args.cost_gate_threshold_usd,
        cache_ttl_seconds=args.cache_ttl_seconds,
        timeout_seconds=args.timeout_seconds,
        skip_cache=args.skip_cache,
        no_cache_for_paid_dispatch=args.no_cache_for_paid_dispatch,
        cache_path=args.cache_path,
        script_path=args.script_path,
    )

    # Emit human-readable summary to stderr
    print(
        f"[run-codex-review] verdict={result.verdict} "
        f"cache_hit={result.cache_hit} cache_age_sec={result.cache_age_sec} "
        f"elapsed_sec={result.elapsed_sec:.1f} "
        f"findings={len(result.findings)}",
        file=sys.stderr,
    )
    for finding in result.findings[:5]:
        print(f"  - {finding}", file=sys.stderr)

    # JSON output (always written if requested)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result.as_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    # Verdict -> exit code per Catalog #281 fail-closed-on-companion-failure:
    #   approve / advisory  -> 0 (safe to dispatch)
    #   needs-attention / no-ship -> 1 (refuse dispatch; findings exist)
    #   invocation-error -> 2 (refuse dispatch; review did not run)
    #   bypass active -> 0 (paired-env operator override active)
    if result.verdict in VERDICTS_SAFE:
        return 0
    if bypass_active:
        print(
            f"[run-codex-review] [CODEX REVIEW BYPASS] verdict={result.verdict} "
            f"would block but bypass is active; allowing dispatch",
            file=sys.stderr,
        )
        return 0
    if result.verdict == VERDICT_INVOCATION_ERROR:
        # Distinct exit code so callers (operator_authorize.py) can
        # distinguish review-did-not-run from review-found-bugs.
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
