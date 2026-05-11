#!/usr/bin/env python3
"""Dispatch a contest-faithful CPU auth eval on the comma_video_compression_challenge
fork via GitHub Actions.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" (commit b4919d24): every shippable submission archive must get an
authoritative CPU score on a 1:1 contest-compliant runner. The contest's
`eval.yml` workflow on `commaai/comma_video_compression_challenge` uses
`runner=ubuntu-latest` (Linux x86_64) — that IS the literal contest hardware.
A non-admin cannot trigger that workflow on the upstream repo, but we maintain a
fork (`adpena/comma_video_compression_challenge`) that exposes the same workflow
and same runner image. Free GitHub Actions minutes are the cheapest path AND
the literal contest hardware.

This helper:

  1. Verifies the input archive's SHA (fail-fast if mismatch).
  2. Uploads the archive as an asset on a fresh GitHub Release on the fork
     (Release tag scoped to the dispatch timestamp; one tag per dispatch).
  3. Calls the `eval.yml` `workflow_dispatch` with `submission_url` pointing
     at the release asset, `runner=ubuntu-latest`, `submission_name` unique.
  4. Polls until the workflow run completes (success or failure).
  5. Downloads the `eval-<submission_name>` artifact (contains `archive.zip`
     and `report.txt`).
  6. Parses `report.txt` for `pose_avg`, `seg_avg`, `rate`, `score`.
  7. Writes `contest_auth_eval.adjudicated.json` next to the archive with the
     full provenance + components + lane-tag `[contest-CPU]` +
     `hardware: github-actions-ubuntu-latest-x86_64`.

Output JSON keys:
  - archive_size_bytes (int)
  - archive_sha256 (str)
  - canonical_score (float)  — recomputed from components, not rounded report display
  - avg_segnet_dist (float)
  - avg_posenet_dist (float)
  - compression_rate (float)
  - score_recomputed_from_components (float)  — sanity check
  - device ("cpu")
  - hardware ("github-actions-ubuntu-latest-x86_64")
  - runner_os_release (str, parsed from log)
  - evaluate_py_sha256 (str)
  - workflow_run_id (int)
  - workflow_run_url (str)
  - release_tag (str)
  - asset_url (str)
  - lane_tag ("[contest-CPU]")
  - dispatched_at_utc (ISO 8601)
  - completed_at_utc (ISO 8601)
  - report_text (str)  — full report.txt contents

Exit codes:
  0 — completed; adjudicated JSON written
  2 — input validation error (missing archive, SHA mismatch, etc.)
  3 — workflow failed (timeout, eval crash, parse error)
  4 — gh CLI / GitHub API error

Usage:
  python tools/dispatch_cpu_eval_via_github_actions.py \\
    --archive-path experiments/results/.../archive.zip \\
    --archive-sha 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb \\
    --submission-name pr107_apogee_gha_cpu_20260508 \\
    --output-dir experiments/results/pr107_cpu_eval_gha_20260508/

Per CLAUDE.md "NEVER invent CLI flags" rule: every flag passed to `gh` is
verified against `gh workflow run --help` / `gh release create --help`. The
eval.yml inputs are read from the actual workflow file before dispatch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

UPSTREAM_FORK_REPO = "adpena/comma_video_compression_challenge"
EVAL_WORKFLOW_FILE = "eval.yml"
DEFAULT_RUNNER = "ubuntu-latest"
LANE_TAG = "[contest-CPU]"
HARDWARE_LABEL = "github-actions-ubuntu-latest-x86_64"
POLL_INTERVAL_SEC = 30
POLL_TIMEOUT_SEC = 60 * 45  # 45 min wall-clock budget; eval has 30 min internal


class AmbiguousSubmissionMatchError(RuntimeError):
    """Raised when artifact selection cannot uniquely resolve a submission.

    Bug class fixed (2026-05-09 codex round-2 HIGH 1): substring match on
    ``submission_name`` would silently pick the wrong artifact under
    concurrent dispatch (e.g. ``apogee`` vs ``apogee_stack_b100``). This
    exception fails CLOSED rather than mis-attribute a ``[contest-CPU]``
    score to the wrong archive.

    Carries the resolution context (the requested name + the candidate
    submission_dir tokens that matched/conflicted) so the operator can
    diagnose without re-running the workflow.
    """

    def __init__(
        self,
        submission_name: str,
        candidates: list[str],
        *,
        context: str = "",
    ) -> None:
        self.submission_name = submission_name
        self.candidates = list(candidates)
        self.context = context
        msg = (
            f"ambiguous submission match for {submission_name!r}: "
            f"{len(candidates)} candidate(s) {candidates!r}"
        )
        if context:
            msg += f" (context: {context})"
        super().__init__(msg)


# Patterns used to extract the canonical ``submission_dir`` token from a
# workflow log or report.txt body. The fork's eval pipeline writes
# ``submission_dir: submissions/<name>`` and may also surface the same path
# via ``--submission-dir submissions/<name>`` on the command line.
#
# The crucial discipline: we ONLY match the basename (Path(value).name) so
# substring overlap (``apogee`` vs ``apogee_stack_b100``) cannot cause a
# cross-attribution. Each match yields exactly one canonical name token.
_SUBMISSION_DIR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ``submission_dir: submissions/<name>`` (with or without trailing /)
    re.compile(r"submission_dir:\s*submissions/([A-Za-z0-9_.\-]+)/?\b"),
    # ``--submission-dir submissions/<name>`` and ``--submission-dir=submissions/<name>``
    re.compile(r"--submission-dir[=\s]+submissions/([A-Za-z0-9_.\-]+)/?\b"),
    # ``--submission-dir ./submissions/<name>`` (leading ./ tolerated)
    re.compile(r"--submission-dir[=\s]+\./submissions/([A-Za-z0-9_.\-]+)/?\b"),
    # ``submission_name=<name>`` literal (workflow_dispatch echo)
    re.compile(r"submission_name=([A-Za-z0-9_.\-]+)\b"),
)


def _validate_submission_name(submission_name: str) -> str:
    """Reject empty / whitespace / path-segment names; return the trimmed name.

    HIGH 1 reinforcement: an empty or whitespace name MUST NOT silently
    match ``submission_dir:`` lines via degenerate regex.
    """
    if submission_name is None:
        raise ValueError("submission_name is required (got None)")
    s = submission_name.strip()
    if not s:
        raise ValueError(
            f"submission_name must be non-empty / non-whitespace; got {submission_name!r}"
        )
    if "/" in s or "\\" in s:
        raise ValueError(
            f"submission_name must not contain path separators; got {submission_name!r} "
            "(use the bare name, not submissions/<name>)"
        )
    return s


def _extract_submission_dir_tokens(text: str) -> set[str]:
    """Return the set of distinct canonical submission_dir basenames in text.

    Each match is reduced to ``Path(<captured>).name`` so a path-suffix
    inside the captured group still yields a single canonical token.
    """
    tokens: set[str] = set()
    for pat in _SUBMISSION_DIR_PATTERNS:
        for match in pat.finditer(text):
            captured = match.group(1)
            if not captured:
                continue
            # Defensive: reduce to basename so any nested path stays clean.
            name = Path(captured).name
            if name:
                tokens.add(name)
    return tokens


def _text_mentions_submission_exactly(text: str, submission_name: str) -> bool:
    """Return True iff text contains an EXACT-identity match for submission_name.

    Exact-identity = at least one ``submission_dir:`` (or equivalent) token
    decodes to ``submission_name``. If the text contains tokens for multiple
    distinct submissions, we still accept iff our requested name is among
    them; the caller (``run_log_mentions_submission``) only needs to know
    whether the log discusses our submission. Artifact selection remains
    fail-closed on duplicate matching reports in ``download_artifact``.
    """
    submission_name = _validate_submission_name(submission_name)
    tokens = _extract_submission_dir_tokens(text)
    return submission_name in tokens


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_dependency_manifest_for_submission(
    submission_dir: Path | None,
    *,
    submission_name: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Return the same runtime custody manifest used by contest_auth_eval.

    GitHub Actions only returns ``report.txt`` for CPU evals, so the dispatcher
    attaches the local runtime manifest used to create the fork PR. The manifest
    is still non-promotional: it is custody metadata for same-runtime pairing
    and does not alter the score.
    """

    runtime_dir = submission_dir or repo_root / "submissions" / submission_name
    inflate_sh = runtime_dir / "inflate.sh"
    if not inflate_sh.is_file():
        return None

    module_path = repo_root / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location(
        "pact_dispatch_cpu_contest_auth_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_fn = getattr(module, "_runtime_dependency_manifest", None)
    if manifest_fn is None:
        return None
    return manifest_fn(
        inflate_sh,
        repo_root / "upstream",
        repo_root=repo_root,
    )


def run_gh(args: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command. Raises on non-zero exit by default."""
    result = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"[gh-error] cmd={' '.join(args)} rc={result.returncode}\n"
            f"  stdout={result.stdout!r}\n  stderr={result.stderr!r}\n"
        )
    return result


def verify_archive(archive_path: Path, expected_sha: str) -> int:
    if not archive_path.exists():
        sys.stderr.write(f"[fatal] archive not found: {archive_path}\n")
        sys.exit(2)
    actual_sha = sha256_of(archive_path)
    if actual_sha != expected_sha:
        sys.stderr.write(
            f"[fatal] archive SHA mismatch:\n"
            f"  expected: {expected_sha}\n  actual:   {actual_sha}\n"
        )
        sys.exit(2)
    # Verify it's a valid zip
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            if not members:
                sys.stderr.write(f"[fatal] archive {archive_path} is an empty zip\n")
                sys.exit(2)
    except zipfile.BadZipFile:
        sys.stderr.write(f"[fatal] archive {archive_path} is not a valid zip\n")
        sys.exit(2)
    return archive_path.stat().st_size


def submission_runtime_contract_error(submission_name: str, pr_number: str | None) -> str | None:
    """Return a fail-fast reason when the GHA workflow cannot see inflate.sh.

    The fork workflow downloads only ``archive.zip`` from ``submission_url``.
    The runtime files, especially ``inflate.sh``, must already exist under
    ``submissions/<submission_name>/`` in the checked-out repository. For new
    non-baseline submissions that means the caller must provide ``--pr-number``
    pointing at a fork PR whose merge ref contains that directory.
    """

    if submission_name == "baseline":
        return None
    if pr_number:
        return None
    return (
        "non-baseline GHA CPU eval requires --pr-number for a PR whose checkout "
        f"contains submissions/{submission_name}/inflate.sh; eval.yml downloads "
        "only archive.zip and does not provide runtime files from the release asset"
    )


def create_release_with_asset(
    archive_path: Path,
    release_tag: str,
    archive_sha: str,
    archive_size: int,
    repo: str,
) -> str:
    """Create a release on the fork and upload archive.zip as an asset.

    Returns the asset download URL.
    """
    notes = (
        f"Auto-created by tools/dispatch_cpu_eval_via_github_actions.py.\n\n"
        f"- archive_sha256: {archive_sha}\n"
        f"- archive_size_bytes: {archive_size}\n"
        f"- dispatched_at_utc: {dt.datetime.now(dt.UTC).isoformat()}\n"
        f"- purpose: CPU auth eval on contest-compliant Linux x86_64 runner\n"
    )
    result = run_gh(
        [
            "release",
            "create",
            release_tag,
            "-R",
            repo,
            "--title",
            f"CPU auth eval dispatch — {release_tag}",
            "--notes",
            notes,
            str(archive_path),
        ]
    )
    if result.returncode != 0:
        sys.stderr.write("[fatal] gh release create failed\n")
        sys.exit(4)
    # Read the asset URL from the freshly created release
    asset_q = run_gh(
        [
            "release",
            "view",
            release_tag,
            "-R",
            repo,
            "--json",
            "assets",
            "--jq",
            ".assets[] | select(.name == \"archive.zip\") | .url",
        ]
    )
    if asset_q.returncode != 0 or not asset_q.stdout.strip():
        sys.stderr.write("[fatal] could not read release asset url after create\n")
        sys.exit(4)
    return asset_q.stdout.strip()


def trigger_workflow(
    submission_name: str,
    submission_url: str,
    runner: str,
    repo: str,
    pr_number: str | None = None,
) -> int:
    """Trigger the eval.yml workflow_dispatch and return the run ID.

    The dispatch endpoint doesn't return the run ID, so we list runs
    immediately afterward and pick the most recent run for this workflow
    that is in_progress or queued.

    pr_number (optional): when provided, the workflow's actions/checkout step
    will use ``refs/pull/<n>/merge`` instead of master. This is required when
    the submission directory (e.g. ``submissions/apogee/``) lives on a fork
    branch that is not yet merged to master — without this the Evaluate step
    fails with ``inflate.sh not found``. Verified against the fork's
    ``eval.yml`` workflow inputs (the ``pr_number`` input IS declared upstream).
    """
    pre_runs = run_gh(
        [
            "run",
            "list",
            "-R",
            repo,
            "-w",
            EVAL_WORKFLOW_FILE,
            "-L",
            "20",
            "--json",
            "databaseId",
        ]
    )
    pre_ids: set[int] = set()
    if pre_runs.returncode == 0 and pre_runs.stdout.strip():
        pre_runs_json = json.loads(pre_runs.stdout)
        pre_ids = {int(row["databaseId"]) for row in pre_runs_json}

    dispatch_args = [
        "workflow",
        "run",
        EVAL_WORKFLOW_FILE,
        "-R",
        repo,
        "-f",
        f"submission_name={submission_name}",
        "-f",
        f"submission_url={submission_url}",
        "-f",
        f"runner={runner}",
    ]
    if pr_number:
        dispatch_args.extend(["-f", f"pr_number={pr_number}"])
    dispatch = run_gh(dispatch_args)
    if dispatch.returncode != 0:
        sys.stderr.write("[fatal] gh workflow run failed\n")
        sys.exit(4)

    # Poll for the new run ID. Under concurrent dispatches, "latest run after
    # workflow_dispatch" is not enough: another agent can dispatch in the same
    # second. Select the run whose workflow log mentions this submission_name.
    #
    # GitHub does not always expose ``gh run view --log`` while a run is still
    # in progress. If this dispatch created exactly one new eval.yml run, the
    # pre/post run-set delta is already a unique custody handle, so accept it
    # rather than waiting five minutes for logs that will only appear after the
    # job completes. If multiple new runs exist, keep the stricter log-token
    # discriminator and fail closed on ambiguity.
    deadline = time.monotonic() + 300
    seen_candidates: set[int] = set()
    while time.monotonic() < deadline:
        runs_q = run_gh(
            [
                "run",
                "list",
                "-R",
                repo,
                "-w",
                EVAL_WORKFLOW_FILE,
                "-L",
                "20",
                "--json",
                "databaseId,status,createdAt",
            ]
        )
        if runs_q.returncode == 0 and runs_q.stdout.strip():
            runs = json.loads(runs_q.stdout)
            new_ids = [
                int(row["databaseId"])
                for row in runs
                if int(row["databaseId"]) not in pre_ids
            ]
            for row in runs:
                rid = int(row["databaseId"])
                if rid in pre_ids:
                    continue
                seen_candidates.add(rid)
                if run_log_mentions_submission(rid, repo, submission_name):
                    return rid
            if len(new_ids) == 1:
                return new_ids[0]
        time.sleep(5)
    sys.stderr.write(
        "[fatal] could not identify matching workflow run within 300s; "
        f"submission_name={submission_name!r}; new_candidate_run_ids="
        f"{sorted(seen_candidates)}\n"
    )
    sys.exit(4)


def run_log_mentions_submission(run_id: int, repo: str, submission_name: str) -> bool:
    """Return True iff a run log names the requested submission EXACTLY.

    GitHub's workflow-dispatch API does not return the new run ID. The only
    reliable same-turn discriminator under concurrent dispatch is the workflow
    log, whose download/evaluate/comment steps include ``submission_name``.

    HIGH 1 fix (codex round-2 review, 2026-05-09): the previous substring
    match (``submission_name in result.stdout``) was prefix-unsafe.
    Dispatching ``apogee`` could attach the wrong run from a concurrent
    ``apogee_stack_b100`` dispatch because the substring ``"apogee"`` was
    present in any line mentioning the longer name. We now extract
    canonical ``submission_dir`` basename tokens from the log and compare
    by exact identity — never by substring.
    """
    submission_name = _validate_submission_name(submission_name)
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "-R", repo, "--log"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return _text_mentions_submission_exactly(result.stdout, submission_name)


def poll_run(run_id: int, repo: str) -> dict[str, Any]:
    """Poll until the run completes; return the final status JSON."""
    started = time.monotonic()
    last_step_logged = ""
    while True:
        elapsed = time.monotonic() - started
        if elapsed > POLL_TIMEOUT_SEC:
            sys.stderr.write(
                f"[fatal] poll timeout after {elapsed:.0f}s on run {run_id}\n"
            )
            sys.exit(3)
        q = run_gh(
            [
                "run",
                "view",
                str(run_id),
                "-R",
                repo,
                "--json",
                "status,conclusion,jobs,url,createdAt,updatedAt",
            ]
        )
        if q.returncode != 0:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        info = json.loads(q.stdout)
        status = info.get("status", "")
        # Surface progress
        for job in info.get("jobs", []):
            if job.get("name") == "test":
                for step in job.get("steps", []):
                    if step.get("status") == "in_progress":
                        name = step.get("name", "")
                        if name and name != last_step_logged:
                            print(
                                f"[poll +{elapsed:.0f}s] step in progress: {name}",
                                flush=True,
                            )
                            last_step_logged = name
        if status == "completed":
            return info
        time.sleep(POLL_INTERVAL_SEC)


def download_artifact(run_id: int, submission_name: str, repo: str, dest_dir: Path) -> Path:
    """Download the eval-<submission_name> artifact; return the path to report.txt.

    HIGH 1 fix (codex round-2 review, 2026-05-09): the fallback branch (when
    the named artifact download fails) downloads ALL artifacts then locates a
    matching ``report.txt`` by substring. Substring matching is prefix-unsafe
    (``apogee`` matches ``apogee_stack_b100``) and produces silent
    cross-attribution under concurrent dispatch. We now:

    1. Tokenize each candidate ``report.txt`` to extract its canonical
       ``submission_dir`` basename(s) via :func:`_extract_submission_dir_tokens`.
    2. Accept ONLY reports whose token set contains an EXACT match for
       ``submission_name``.
    3. If multiple distinct reports each carry the requested name (i.e., the
       artifact dump contains duplicates), raise
       :class:`AmbiguousSubmissionMatchError` rather than silently pick one.
    4. Empty / whitespace ``submission_name`` is rejected up front by
       :func:`_validate_submission_name`.
    """
    submission_name = _validate_submission_name(submission_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"eval-{submission_name}"
    result = run_gh(
        [
            "run",
            "download",
            str(run_id),
            "-R",
            repo,
            "-n",
            artifact_name,
            "-D",
            str(dest_dir),
        ]
    )
    if result.returncode != 0:
        sys.stderr.write(
            "[warn] named artifact download failed; downloading all artifacts "
            "and selecting by report submission_dir\n"
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
        dest_dir.mkdir(parents=True, exist_ok=True)
        all_result = run_gh(
            [
                "run",
                "download",
                str(run_id),
                "-R",
                repo,
                "-D",
                str(dest_dir),
            ]
        )
        if all_result.returncode != 0:
            sys.stderr.write("[fatal] gh run download failed\n")
            sys.exit(4)
    # Locate matching report.txt within dest_dir using EXACT identity match.
    # We tokenize each candidate report's canonical submission_dir tokens and
    # accept iff submission_name is in that set. Ambiguity (multiple reports
    # whose tokens include submission_name) is FAIL CLOSED.
    candidates: list[Path] = []
    for path in dest_dir.rglob("report.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tokens = _extract_submission_dir_tokens(text)
        if submission_name in tokens:
            candidates.append(path)
    if not candidates:
        sys.stderr.write(
            f"[fatal] matching report.txt for submission_name={submission_name!r} "
            f"not found in downloaded artifact(s) at {dest_dir}\n"
        )
        sys.exit(3)
    if len(candidates) > 1:
        # Distinct artifact paths each claiming our submission_name. This is
        # the FAIL-CLOSED branch — never silently pick one.
        raise AmbiguousSubmissionMatchError(
            submission_name,
            [str(p) for p in candidates],
            context=(
                f"download_artifact run_id={run_id} repo={repo}; "
                "multiple report.txt files in the same artifact bundle each "
                "carry an exact-identity match for the requested submission_name"
            ),
        )
    return candidates[0]


REPORT_PATTERNS = {
    "avg_posenet_dist": re.compile(
        r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"
    ),
    "avg_segnet_dist": re.compile(
        r"Average SegNet Distortion:\s*([0-9.eE+-]+)"
    ),
    "compression_rate": re.compile(r"Compression Rate:\s*([0-9.eE+-]+)"),
    "reported_final_score_display_rounded": re.compile(r"Final score:.*=\s*([0-9.eE+-]+)"),
    "n_samples": re.compile(r"Evaluation results over (\d+) samples"),
}


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text()
    parsed: dict[str, Any] = {"report_text": text}
    for key, pat in REPORT_PATTERNS.items():
        m = pat.search(text)
        if not m:
            sys.stderr.write(
                f"[fatal] could not parse {key} from report.txt:\n{text}\n"
            )
            sys.exit(3)
        val = m.group(1)
        parsed[key] = int(val) if key == "n_samples" else float(val)
    # Recompute score from components for sanity
    import math

    recomputed = (
        100.0 * parsed["avg_segnet_dist"]
        + math.sqrt(10.0 * parsed["avg_posenet_dist"])
        + 25.0 * parsed["compression_rate"]
    )
    parsed["score_recomputed_from_components"] = recomputed
    parsed["canonical_score_recomputed"] = recomputed
    parsed["canonical_score"] = recomputed
    parsed["canonical_score_source"] = "score_recomputed_from_components"
    display_score = parsed["reported_final_score_display_rounded"]
    drift = abs(recomputed - display_score)
    parsed["score_rounding_abs_delta"] = drift
    parsed["score_reported_rounded_differs_from_canonical"] = drift > 1e-12
    if drift > 0.02:  # report.txt rounds to 2 decimals
        sys.stderr.write(
            f"[warn] score drift {drift:.4f} > 0.02; reported={display_score} "
            f"recomputed={recomputed}\n"
        )
    return parsed


def fetch_log_for_runner(run_id: int, repo: str) -> str:
    """Fetch a snippet of the workflow log to capture runner OS info."""
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "-R", repo, "--log"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "<log fetch failed>"
    # Look for "Linux" / "ubuntu" / kernel info in early steps
    out = result.stdout
    for line in out.splitlines():
        ll = line.lower()
        if "ubuntu" in ll and ("22.04" in ll or "24.04" in ll or "20.04" in ll):
            return line.strip()
    # Fallback: first 2000 chars
    return out[:2000]


def write_adjudicated(
    output_path: Path,
    *,
    archive_path: Path,
    archive_sha: str,
    archive_size: int,
    parsed: dict[str, Any],
    run_id: int,
    run_url: str,
    release_tag: str,
    asset_url: str,
    runner_os_release: str,
    evaluate_py_sha: str,
    submission_name: str,
    dispatched_at: str,
    completed_at: str,
    repo: str,
    runtime_manifest: dict[str, Any] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "archive_relpath": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "canonical_score": parsed["canonical_score"],
        "canonical_score_recomputed": parsed["canonical_score_recomputed"],
        "canonical_score_source": parsed["canonical_score_source"],
        "reported_final_score_display_rounded": parsed[
            "reported_final_score_display_rounded"
        ],
        "score_rounding_abs_delta": parsed["score_rounding_abs_delta"],
        "score_reported_rounded_differs_from_canonical": parsed[
            "score_reported_rounded_differs_from_canonical"
        ],
        "avg_segnet_dist": parsed["avg_segnet_dist"],
        "avg_posenet_dist": parsed["avg_posenet_dist"],
        "compression_rate": parsed["compression_rate"],
        "score_recomputed_from_components": parsed[
            "score_recomputed_from_components"
        ],
        "n_samples": parsed["n_samples"],
        "device": "cpu",
        "hardware": HARDWARE_LABEL,
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "runner_os_release": runner_os_release,
        "evaluate_py_sha256": evaluate_py_sha,
        "runtime_tree_sha256": (
            runtime_manifest.get("runtime_tree_sha256")
            if isinstance(runtime_manifest, dict)
            else None
        ),
        "runtime_content_tree_sha256": (
            runtime_manifest.get("runtime_content_tree_sha256")
            if isinstance(runtime_manifest, dict)
            else None
        ),
        "inflate_runtime_manifest": runtime_manifest,
        "provenance": {
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
            "device": "cpu",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "hardware": HARDWARE_LABEL,
            "inflate_runtime_manifest": runtime_manifest,
        },
        "workflow_run_id": run_id,
        "workflow_run_url": run_url,
        "fork_repo": repo,
        "submission_name": submission_name,
        "release_tag": release_tag,
        "asset_url": asset_url,
        "lane_tag": LANE_TAG,
        "evidence_grade": "contest-CPU-1to1",
        "dispatched_at_utc": dispatched_at,
        "completed_at_utc": completed_at,
        "report_text": parsed["report_text"],
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dispatch contest-faithful CPU auth eval via fork GHA",
    )
    p.add_argument("--archive-path", required=True, type=Path)
    p.add_argument("--archive-sha", required=True, type=str)
    p.add_argument(
        "--submission-name",
        required=True,
        type=str,
        help="unique submission name (will appear under submissions/<name>/)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory to write adjudicated.json and downloaded report",
    )
    p.add_argument(
        "--repo",
        default=UPSTREAM_FORK_REPO,
        help=f"GitHub repo (default: {UPSTREAM_FORK_REPO})",
    )
    p.add_argument("--runner", default=DEFAULT_RUNNER)
    p.add_argument(
        "--release-tag",
        default=None,
        help="release tag to use (default: cpu-eval-<submission_name>-<utc-stamp>)",
    )
    p.add_argument(
        "--evaluate-py-path",
        type=Path,
        default=Path("upstream/evaluate.py"),
        help="path to local upstream/evaluate.py for SHA pinning",
    )
    p.add_argument(
        "--skip-dispatch",
        action="store_true",
        help="(diagnostic) only verify archive+upload; don't dispatch workflow",
    )
    p.add_argument(
        "--pr-number",
        type=str,
        default=None,
        help=(
            "optional PR number on the fork; when set the workflow's "
            "actions/checkout step uses refs/pull/<n>/merge instead of master "
            "(required when submission code lives on a PR branch). The fork's "
            "eval.yml exposes pr_number as an optional workflow_dispatch input."
        ),
    )
    p.add_argument(
        "--submission-dir",
        type=Path,
        default=None,
        help=(
            "local path to a submission directory (containing inflate.sh + "
            "runtime files). Used together with --auto-create-fork-pr to "
            "auto-publish the runtime to a fork PR before dispatch."
        ),
    )
    p.add_argument(
        "--auto-create-fork-pr",
        action="store_true",
        help=(
            "if --submission-name is non-baseline AND --pr-number is missing, "
            "auto-invoke tools/create_fork_pr_for_submission.py to create a "
            "draft fork PR providing submissions/<name>/ runtime, then use the "
            "returned PR number for the workflow dispatch. Requires "
            "--submission-dir."
        ),
    )
    args = p.parse_args()

    archive_size = verify_archive(args.archive_path, args.archive_sha)
    print(
        f"[ok] archive {args.archive_path} verified: "
        f"sha256={args.archive_sha} bytes={archive_size}",
        flush=True,
    )
    runtime_contract_error = submission_runtime_contract_error(
        args.submission_name,
        args.pr_number,
    )
    if runtime_contract_error and args.auto_create_fork_pr:
        if args.submission_dir is None:
            sys.stderr.write(
                "[fatal] --auto-create-fork-pr requires --submission-dir\n"
            )
            return 2
        # Auto-create the fork PR providing submissions/<name>/ runtime.
        helper = Path(__file__).resolve().parent / "create_fork_pr_for_submission.py"
        proc = subprocess.run(
            [
                sys.executable, str(helper),
                "--submission-dir", str(args.submission_dir),
                "--submission-name", args.submission_name,
                "--fork-repo", args.repo,
                "--archive-path", str(args.archive_path),
                "--archive-sha256", args.archive_sha,
                "--reuse-existing",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"[fatal] create_fork_pr_for_submission.py failed: "
                f"rc={proc.returncode}\n  stderr: {proc.stderr!r}\n"
            )
            return 2
        pr_number_str = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
        try:
            args.pr_number = str(int(pr_number_str))
        except (ValueError, TypeError):
            sys.stderr.write(
                f"[fatal] could not parse PR number from "
                f"create_fork_pr_for_submission.py output: {pr_number_str!r}\n"
            )
            return 2
        print(
            f"[ok] auto-created fork PR #{args.pr_number} for "
            f"submission {args.submission_name!r}",
            flush=True,
        )
        # Re-validate now that --pr-number is set.
        runtime_contract_error = submission_runtime_contract_error(
            args.submission_name,
            args.pr_number,
        )
    if runtime_contract_error:
        sys.stderr.write(f"[fatal] {runtime_contract_error}\n")
        return 2

    evaluate_py_sha = (
        sha256_of(args.evaluate_py_path)
        if args.evaluate_py_path.exists()
        else "<not-found>"
    )
    repo_root = Path(__file__).resolve().parents[1]
    runtime_manifest = runtime_dependency_manifest_for_submission(
        args.submission_dir,
        submission_name=args.submission_name,
        repo_root=repo_root,
    )

    dispatched_at = dt.datetime.now(dt.UTC).isoformat()
    release_tag = args.release_tag or (
        f"cpu-eval-{args.submission_name}-"
        + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    )

    asset_url = create_release_with_asset(
        archive_path=args.archive_path,
        release_tag=release_tag,
        archive_sha=args.archive_sha,
        archive_size=archive_size,
        repo=args.repo,
    )
    print(f"[ok] release asset uploaded: {asset_url}", flush=True)

    if args.skip_dispatch:
        print("[skip-dispatch] exiting before workflow trigger", flush=True)
        return 0

    run_id = trigger_workflow(
        submission_name=args.submission_name,
        submission_url=asset_url,
        runner=args.runner,
        repo=args.repo,
        pr_number=args.pr_number,
    )
    run_url = f"https://github.com/{args.repo}/actions/runs/{run_id}"
    print(f"[ok] workflow dispatched: run_id={run_id} url={run_url}", flush=True)

    info = poll_run(run_id, args.repo)
    completed_at = dt.datetime.now(dt.UTC).isoformat()
    if info.get("conclusion") != "success":
        sys.stderr.write(
            f"[fatal] workflow run {run_id} concluded "
            f"{info.get('conclusion')!r}; see {run_url}\n"
        )
        return 3

    with tempfile.TemporaryDirectory() as td:
        report_path = download_artifact(
            run_id, args.submission_name, args.repo, Path(td)
        )
        parsed = parse_report(report_path)
        # Copy report.txt into output_dir for posterity
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(report_path, args.output_dir / "report.txt")

    runner_os = fetch_log_for_runner(run_id, args.repo)

    out = write_adjudicated(
        args.output_dir / "contest_auth_eval.adjudicated.json",
        archive_path=args.archive_path,
        archive_sha=args.archive_sha,
        archive_size=archive_size,
        parsed=parsed,
        run_id=run_id,
        run_url=run_url,
        release_tag=release_tag,
        asset_url=asset_url,
        runner_os_release=runner_os,
        evaluate_py_sha=evaluate_py_sha,
        submission_name=args.submission_name,
        dispatched_at=dispatched_at,
        completed_at=completed_at,
        repo=args.repo,
        runtime_manifest=runtime_manifest,
    )
    print(
        f"[done] adjudicated.json written to {out}\n"
        f"  canonical_score = {parsed['canonical_score']}  {LANE_TAG}\n"
        f"  pose_avg = {parsed['avg_posenet_dist']}\n"
        f"  seg_avg  = {parsed['avg_segnet_dist']}\n"
        f"  rate     = {parsed['compression_rate']}\n"
        f"  recomputed = {parsed['score_recomputed_from_components']}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
