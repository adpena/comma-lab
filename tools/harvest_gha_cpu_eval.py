#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest a [contest-CPU] GHA eval.yml workflow_dispatch result.

Counterpart to ``tools/trigger_gha_cpu_eval.py``. Reads the dispatch_metadata.json
that the trigger phase wrote, polls the workflow run to completion, downloads
the ``eval-<submission_name>`` artifact, parses ``report.txt``, and writes
``submissions/<label>/contest_auth_eval.cpu.json`` per the A1 reference
schema (``submissions/a1/contest_auth_eval.cpu.json``).

What it does:
  1. Polls ``gh run view <run_id> --json status,conclusion`` until completed.
  2. On ``conclusion=success``: downloads the workflow artifact, parses the
     ``report.txt`` for ``avg_posenet_dist``/``avg_segnet_dist``/
     ``compression_rate``/``n_samples`` and recomputes the canonical score.
  3. Captures the runner OS line via ``gh run view <run_id> --log``.
  4. Writes ``submissions/<label>/contest_auth_eval.cpu.json`` (or the
     ``--output-cpu-json`` override) with these custody fields per
     CLAUDE.md "Apples-to-apples evidence discipline" + the A1 reference:
       - ``evidence_grade: "contest-CPU-1to1"`` (only when runner is
         ubuntu-latest); otherwise ``"advisory"``.
       - ``lane_tag: "[contest-CPU]"`` likewise gated by runner identity.
       - ``hardware: "github-actions-ubuntu-latest-x86_64"``.
       - ``score_recomputed_from_components`` (not the rounded report-display
         score) is the canonical score.
  5. Closes the lane-claim with a terminal status via
     ``tools/claim_lane_dispatch.py claim --force --status completed_gha_cpu_eval``
     (or ``failed_gha_cpu_eval_<reason>``).
  6. Appends a $0.00 cost-band anchor (GHA minutes are free for public repos)
     via ``tools/append_cost_band_anchor.py --platform github --gpu cpu``.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE": the ``[contest-CPU]`` tag is gated by the
runner being literally ``ubuntu-latest`` (Linux x86_64). The non-CPU runner
``linux-nvidia-t4`` available in eval.yml is a different axis and is NOT
tagged ``[contest-CPU]`` even though the same workflow accepts it.

Per CLAUDE.md "Forbidden score claims": no score is reported until
``report.txt`` is parsed; on poll-timeout / fail / parse-error the harvest
returns the relevant exit code and leaves the lane claim closed as
``failed_gha_cpu_eval_<reason>``.

Usage:
  python tools/harvest_gha_cpu_eval.py \\
    --dispatch-metadata experiments/results/gha_cpu_eval_<label>_<utc>/dispatch_metadata.json

Exit codes:
  0 — harvested success; contest_auth_eval.cpu.json written
  2 — input validation error
  3 — workflow conclusion != success (timeout / failure / cancelled)
  4 — gh CLI error / API error
  5 — report.txt parse error
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POLL_INTERVAL_SEC = 30
DEFAULT_POLL_TIMEOUT_SEC = 60 * 45  # 45 min; eval has 30 min internal timeout

# Patterns parse the upstream evaluate.py output format. Verified against
# upstream/evaluate.py:95-100. The "Final score:" display value is rounded to
# 2 decimals and must NOT be used as the canonical score — recompute from
# components per "Apples-to-apples evidence discipline".
_REPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "avg_posenet_dist": re.compile(r"Average PoseNet Distortion:\s*([0-9.eE+,-]+)"),
    "avg_segnet_dist": re.compile(r"Average SegNet Distortion:\s*([0-9.eE+,-]+)"),
    "compression_rate": re.compile(r"Compression Rate:\s*([0-9.eE+,-]+)"),
    "reported_final_score_display_rounded": re.compile(r"Final score:.*=\s*([0-9.eE+,-]+)"),
    "n_samples": re.compile(r"Evaluation results over (\d+) samples"),
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], check=False, capture_output=True, text=True)


def load_dispatch_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"VALIDATION_ERROR: dispatch_metadata not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"VALIDATION_ERROR: dispatch_metadata.json malformed: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("VALIDATION_ERROR: dispatch_metadata must be a JSON object")
    if data.get("schema") != "pact.gha_cpu_eval_dispatch_metadata.v1":
        raise SystemExit(
            "VALIDATION_ERROR: dispatch_metadata.json schema mismatch; "
            f"got schema={data.get('schema')!r}"
        )
    required = {
        "label",
        "submission_name",
        "lane_id",
        "instance_job_id",
        "archive_sha256",
        "archive_size_bytes",
        "fork_repo",
        "runner",
        "workflow_run_id",
    }
    missing = sorted(required - set(data))
    if missing:
        raise SystemExit(
            f"VALIDATION_ERROR: dispatch_metadata.json missing required fields: {missing}"
        )
    if data.get("workflow_run_id") is None:
        raise SystemExit(
            "VALIDATION_ERROR: dispatch_metadata.workflow_run_id is null — "
            "trigger phase failed before run id capture; nothing to harvest"
        )
    return data


def poll_run(
    run_id: int,
    repo: str,
    *,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    poll_timeout_sec: int = DEFAULT_POLL_TIMEOUT_SEC,
    monotonic: callable = time.monotonic,
    sleep: callable = time.sleep,
) -> dict[str, Any]:
    """Poll ``gh run view`` until ``status=completed``; return the JSON.

    ``monotonic`` and ``sleep`` are injected for unit tests that prefer to
    drive the loop without real wall-clock waits.
    """
    started = monotonic()
    last_step = ""
    while True:
        elapsed = monotonic() - started
        if elapsed > poll_timeout_sec:
            raise TimeoutError(f"poll timeout after {elapsed:.0f}s on run {run_id}")
        q = run_gh(
            [
                "run",
                "view",
                str(run_id),
                "-R",
                repo,
                "--json",
                "status,conclusion,jobs,url",
            ]
        )
        if q.returncode != 0:
            sleep(poll_interval_sec)
            continue
        try:
            info = json.loads(q.stdout)
        except json.JSONDecodeError:
            sleep(poll_interval_sec)
            continue
        if info.get("status") == "completed":
            return info
        for job in info.get("jobs", []) or []:
            if job.get("name") == "test":
                for step in job.get("steps", []) or []:
                    if step.get("status") == "in_progress":
                        name = step.get("name", "")
                        if name and name != last_step:
                            print(
                                f"[poll +{elapsed:.0f}s] step in progress: {name}",
                                flush=True,
                            )
                            last_step = name
        sleep(poll_interval_sec)


def download_artifact(run_id: int, submission_name: str, repo: str, dest_dir: Path) -> Path:
    """Download eval-<submission_name>; return the path to report.txt."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"eval-{submission_name}"
    res = run_gh(
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
    if res.returncode != 0:
        raise RuntimeError(
            f"gh run download failed for artifact {artifact_name!r}: "
            f"stderr={res.stderr!r}"
        )
    report = dest_dir / "report.txt"
    if not report.is_file():
        # Sometimes the artifact unzips into a nested directory.
        candidates = list(dest_dir.rglob("report.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"report.txt missing from artifact {artifact_name!r} at {dest_dir}"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"multiple report.txt files in artifact {artifact_name!r}: "
                f"{[str(p) for p in candidates]}"
            )
        report = candidates[0]
    return report


def parse_report(report_path: Path) -> dict[str, Any]:
    """Parse upstream evaluate.py report.txt and recompute canonical score."""
    text = report_path.read_text(encoding="utf-8")
    parsed: dict[str, Any] = {"report_text": text}
    for key, pat in _REPORT_PATTERNS.items():
        match = pat.search(text)
        if not match:
            raise ValueError(f"could not parse {key} from report.txt:\n{text}")
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key == "n_samples" else float(raw)

    recomputed = (
        100.0 * parsed["avg_segnet_dist"]
        + math.sqrt(10.0 * parsed["avg_posenet_dist"])
        + 25.0 * parsed["compression_rate"]
    )
    parsed["canonical_score"] = recomputed
    parsed["canonical_score_recomputed"] = recomputed
    parsed["score_recomputed_from_components"] = recomputed
    parsed["canonical_score_source"] = "score_recomputed_from_components"
    display = parsed["reported_final_score_display_rounded"]
    parsed["score_rounding_abs_delta"] = abs(recomputed - display)
    parsed["score_reported_rounded_differs_from_canonical"] = (
        parsed["score_rounding_abs_delta"] > 1e-12
    )
    return parsed


def fetch_runner_os_line(run_id: int, repo: str) -> str:
    res = run_gh(["run", "view", str(run_id), "-R", repo, "--log"])
    if res.returncode != 0:
        return ""
    for line in res.stdout.splitlines():
        if "Image: ubuntu-" in line:
            return line.strip()
        if "Operating System" in line and "Ubuntu" in line:
            return line.strip()
    return ""


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_contest_compliant_runner(runner: str) -> bool:
    """The ``[contest-CPU]`` tag is gated on runner identity == ubuntu-latest.

    Per CLAUDE.md "Submission auth eval ... ON 1:1 CONTEST-COMPLIANT
    HARDWARE": Linux x86_64 GHA ubuntu-latest IS the contest CI runner.
    The eval.yml workflow also accepts ``linux-nvidia-t4`` for CUDA runs;
    that does NOT count as ``[contest-CPU]``.
    """
    return runner == "ubuntu-latest"


def build_cpu_json(
    *,
    metadata: dict[str, Any],
    parsed: dict[str, Any],
    runner_os: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    """Build the contest_auth_eval.cpu.json payload per A1 reference."""
    runner = metadata.get("runner", "")
    compliant = is_contest_compliant_runner(runner)
    evidence_grade = "contest-CPU-1to1" if compliant else "advisory"
    lane_tag = "[contest-CPU]" if compliant else "[advisory]"
    hardware = (
        "github-actions-ubuntu-latest-x86_64"
        if compliant
        else f"github-actions-{runner}"
    )
    return {
        "archive_size_bytes": int(metadata.get("archive_size_bytes")),
        "archive_sha256": metadata.get("archive_sha256"),
        "asset_url": metadata.get("archive_url"),
        "avg_posenet_dist": parsed["avg_posenet_dist"],
        "avg_segnet_dist": parsed["avg_segnet_dist"],
        "canonical_score": parsed["canonical_score"],
        "canonical_score_recomputed": parsed["canonical_score_recomputed"],
        "canonical_score_source": parsed["canonical_score_source"],
        "completed_at_utc": completed_at_utc,
        "compression_rate": parsed["compression_rate"],
        "device": "cpu",
        "dispatched_at_utc": metadata.get("dispatched_at_utc"),
        "evidence_grade": evidence_grade,
        "fork_repo": metadata.get("fork_repo"),
        "hardware": hardware,
        "lane_tag": lane_tag,
        "n_samples": parsed["n_samples"],
        "release_tag": metadata.get("release_tag"),
        "report_text": parsed["report_text"],
        "reported_final_score_display_rounded": parsed[
            "reported_final_score_display_rounded"
        ],
        "runner": runner,
        "runner_arch": "x86_64" if compliant else "unknown",
        "runner_os_release": runner_os,
        "score_claim_valid": compliant,
        "score_recomputed_from_components": parsed["score_recomputed_from_components"],
        "score_reported_rounded_differs_from_canonical": parsed[
            "score_reported_rounded_differs_from_canonical"
        ],
        "score_rounding_abs_delta": parsed["score_rounding_abs_delta"],
        "submission_name": metadata.get("submission_name"),
        "workflow_run_id": int(metadata.get("workflow_run_id")),
        "workflow_run_url": metadata.get("workflow_run_url"),
    }


def close_lane_claim_terminal(
    *,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    status: str,
    notes: str,
) -> None:
    """Append a terminal row for the active claim. Best-effort (non-fatal)."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--force",
        "--lane-id",
        lane_id,
        "--platform",
        "github",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--status",
        status,
        "--notes",
        notes,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"[warn] terminal lane claim failed rc={proc.returncode}: "
            f"stderr={proc.stderr!r}\n"
        )


def append_cost_band_anchor(
    *,
    dispatch_label: str,
    wall_clock_sec: float,
) -> None:
    """GHA minutes are $0 for public repos. Best-effort (non-fatal)."""
    helper = REPO_ROOT / "tools" / "append_cost_band_anchor.py"
    if not helper.is_file():
        return
    cmd = [
        sys.executable,
        str(helper),
        "--dispatch-label",
        dispatch_label,
        "--trainer",
        "upstream/.github/workflows/eval.yml",
        "--platform",
        "github",
        "--gpu",
        "cpu",
        "--epochs",
        "0",
        "--batch-size",
        "16",
        "--actual-wall-clock-sec",
        f"{wall_clock_sec:.0f}",
        "--actual-cost-usd",
        "0.00",
        "--notes",
        "GHA CPU eval on public repo (free minutes); 0 USD cost; runner=ubuntu-latest",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"[warn] cost-band anchor append failed rc={proc.returncode}: "
            f"stderr={proc.stderr!r}\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest a [contest-CPU] GHA eval result.",
    )
    parser.add_argument(
        "--dispatch-metadata",
        required=True,
        type=Path,
        help="Path to dispatch_metadata.json written by trigger_gha_cpu_eval.py",
    )
    parser.add_argument(
        "--output-cpu-json",
        type=Path,
        default=None,
        help="Override destination for contest_auth_eval.cpu.json (default: submissions/<label>/contest_auth_eval.cpu.json)",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SEC,
    )
    parser.add_argument(
        "--poll-timeout-sec",
        type=int,
        default=DEFAULT_POLL_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--agent",
        default="claude:harvest_gha_cpu_eval",
        help="Agent name recorded in the terminal lane-claim row",
    )
    args = parser.parse_args(argv)

    metadata = load_dispatch_metadata(args.dispatch_metadata)
    label = metadata["label"]
    submission_name = metadata["submission_name"]
    lane_id = metadata["lane_id"]
    instance_job_id = metadata["instance_job_id"]
    repo = metadata["fork_repo"]
    run_id = int(metadata["workflow_run_id"])

    print(f"[harvest] polling run {run_id} on {repo} ...", flush=True)
    started = time.monotonic()
    try:
        info = poll_run(
            run_id,
            repo,
            poll_interval_sec=args.poll_interval_sec,
            poll_timeout_sec=args.poll_timeout_sec,
        )
    except TimeoutError as exc:
        sys.stderr.write(f"[fatal] {exc}\n")
        close_lane_claim_terminal(
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=args.agent,
            status="failed_gha_cpu_eval_poll_timeout",
            notes=f"poll timeout after {args.poll_timeout_sec}s on run {run_id}",
        )
        return 3

    conclusion = info.get("conclusion")
    if conclusion != "success":
        sys.stderr.write(
            f"[fatal] workflow run {run_id} concluded {conclusion!r}; "
            f"see https://github.com/{repo}/actions/runs/{run_id}\n"
        )
        close_lane_claim_terminal(
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=args.agent,
            status=f"failed_gha_cpu_eval_{conclusion or 'unknown'}",
            notes=f"workflow conclusion={conclusion!r}",
        )
        return 3

    completed_at_utc = _utc_now_iso()
    wall_clock_sec = max(time.monotonic() - started, 0.0)

    with tempfile.TemporaryDirectory() as td:
        try:
            report_path = download_artifact(run_id, submission_name, repo, Path(td))
        except (RuntimeError, FileNotFoundError) as exc:
            sys.stderr.write(f"[fatal] artifact download failed: {exc}\n")
            close_lane_claim_terminal(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status="failed_gha_cpu_eval_artifact_download",
                notes=str(exc),
            )
            return 4
        try:
            parsed = parse_report(report_path)
        except ValueError as exc:
            sys.stderr.write(f"[fatal] report.txt parse error: {exc}\n")
            close_lane_claim_terminal(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status="failed_gha_cpu_eval_parse_error",
                notes=str(exc)[:200],
            )
            return 5
        # Preserve a copy of report.txt next to dispatch_metadata.json.
        try:
            shutil.copy(report_path, args.dispatch_metadata.parent / "report.txt")
        except OSError:
            pass

    runner_os = fetch_runner_os_line(run_id, repo)
    cpu_json_record = build_cpu_json(
        metadata=metadata,
        parsed=parsed,
        runner_os=runner_os,
        completed_at_utc=completed_at_utc,
    )

    output_cpu_json = args.output_cpu_json or (
        REPO_ROOT / "submissions" / label / "contest_auth_eval.cpu.json"
    )
    output_cpu_json.parent.mkdir(parents=True, exist_ok=True)
    output_cpu_json.write_text(
        json.dumps(cpu_json_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    close_lane_claim_terminal(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=args.agent,
        status="completed_gha_cpu_eval",
        notes=(
            f"canonical_score={cpu_json_record['canonical_score']:.10f} "
            f"{cpu_json_record['lane_tag']} run_id={run_id} "
            f"hardware={cpu_json_record['hardware']}"
        ),
    )
    append_cost_band_anchor(
        dispatch_label=instance_job_id,
        wall_clock_sec=wall_clock_sec,
    )

    print(
        f"[done] {output_cpu_json}\n"
        f"  canonical_score = {cpu_json_record['canonical_score']:.10f}  {cpu_json_record['lane_tag']}\n"
        f"  pose_avg        = {cpu_json_record['avg_posenet_dist']}\n"
        f"  seg_avg         = {cpu_json_record['avg_segnet_dist']}\n"
        f"  rate            = {cpu_json_record['compression_rate']}\n"
        f"  hardware        = {cpu_json_record['hardware']}\n"
        f"  evidence_grade  = {cpu_json_record['evidence_grade']}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
