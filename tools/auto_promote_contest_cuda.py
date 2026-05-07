"""Auto-promotion: scan for contest-CUDA artifacts → auto-mark lane gates + auto-update reports.

Per priority-list item #4 from the cathedral roadmap rollup
(`feedback_automate_and_densify_intelligence_20260507`):

  > Dispatch auto-routing — when contest-CUDA score lands, auto-promote
  > gates + auto-update reports.

This tool walks `experiments/results/` looking for any directory containing
both:
  - an `archive.zip`
  - a contest-CUDA evidence file (one of:
    `pre_submission_compliance.contest_final.json`,
    `contest_auth_eval.adjudicated.json`)

For each match, it:
  1. Reads the contest-CUDA score from the evidence file
  2. Computes the lane id from the directory name
  3. Calls `tools.lane_maturity.py mark` to advance the lane through the
     gate ladder (impl_complete + real_archive_empirical + contest_cuda)
  4. Appends a record to `experiments/results/bilevel_atom_ledger.jsonl`
  5. Updates `reports/latest.md` if the new score beats the current
     frontier

Strict-scorer-rule: pure CPU + json + filesystem. No scorer load. The
contest-CUDA score is READ from disk, never computed.

Cross-references:

- Bilevel driver: `tools/run_bilevel_optimization.py`
- Lane maturity CLI: `tools/lane_maturity.py`
- Three orientations memo: `feedback_frontier_paper_floor_unknown_20260507`
"""

import argparse
import datetime as _dt
import json
import pathlib
import subprocess
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContestCudaArtifact:
    """One discovered (archive, evidence, score) triple."""
    lane_dir: str
    archive_path: str
    evidence_path: str
    score: float
    archive_bytes: int
    archive_sha256: str | None
    evidence_grade: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _try_read_score(evidence_path: pathlib.Path) -> tuple[float, dict] | None:
    """Try to extract a contest-CUDA score from a known evidence-file format."""
    try:
        data = json.loads(evidence_path.read_text())
    except Exception:
        return None
    # Pattern 1: contest_final compliance file with `auth_eval.score` (T4)
    if isinstance(data, dict):
        auth = data.get("auth_eval") or {}
        if isinstance(auth, dict):
            score = auth.get("score") or auth.get("score_total")
            if isinstance(score, (int, float)):
                return float(score), data
        # Pattern 2: top-level `score` key
        score = data.get("score") or data.get("score_total")
        if isinstance(score, (int, float)):
            return float(score), data
        # Pattern 3: adjudicated.json
        if "score" in data:
            try:
                return float(data["score"]), data
            except (TypeError, ValueError):
                pass
    return None


def discover_contest_cuda_artifacts(
    repo_root: pathlib.Path,
    *,
    results_subdir: str = "experiments/results",
) -> list[ContestCudaArtifact]:
    """Walk `experiments/results/` finding (archive, evidence, score) triples."""
    out: list[ContestCudaArtifact] = []
    base = repo_root / results_subdir
    if not base.exists():
        return out
    candidate_evidence_names = (
        "pre_submission_compliance.contest_final.json",
        "contest_auth_eval.adjudicated.json",
        "contest_final.json",
    )
    for lane_dir in sorted(base.iterdir()):
        if not lane_dir.is_dir():
            continue
        archive = lane_dir / "archive.zip"
        if not archive.exists():
            continue
        for candidate_name in candidate_evidence_names:
            evidence = lane_dir / candidate_name
            if not evidence.exists():
                continue
            parsed = _try_read_score(evidence)
            if parsed is None:
                continue
            score, blob = parsed
            sha = None
            if isinstance(blob, dict):
                sha = (
                    blob.get("archive_sha256")
                    or (blob.get("archive") or {}).get("sha256")
                    or (blob.get("auth_eval") or {}).get("archive_sha256")
                )
            out.append(ContestCudaArtifact(
                lane_dir=str(lane_dir),
                archive_path=str(archive),
                evidence_path=str(evidence),
                score=score,
                archive_bytes=archive.stat().st_size,
                archive_sha256=str(sha) if sha is not None else None,
                evidence_grade="[contest-CUDA]",
            ))
            break  # one evidence per lane_dir
    return out


# ---------------------------------------------------------------------------
# Lane gate auto-promotion
# ---------------------------------------------------------------------------

def lane_id_from_dir(lane_dir_path: str) -> str:
    """Convert ``experiments/results/lane_<id>_<UTC>`` to ``<id>``.

    For directories without a lane prefix (e.g.
    `pr103_repack_pr106_standalone_20260507`), use the directory basename
    as the lane id (truncated to first 64 chars to satisfy registry
    constraints).
    """
    name = pathlib.Path(lane_dir_path).name
    if name.startswith("lane_"):
        rest = name[len("lane_"):]
        # strip trailing _<UTC>
        parts = rest.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) >= 8 and parts[1][0:4].isdigit():
            return parts[0]
        return rest
    return name[:64]


def mark_lane_gates(
    repo_root: pathlib.Path,
    lane_id: str,
    artifact: ContestCudaArtifact,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Auto-mark impl_complete + real_archive_empirical + contest_cuda gates."""
    cli = repo_root / "tools/lane_maturity.py"
    if not cli.exists():
        return {"status": "BLOCKED", "reason": f"lane_maturity CLI missing at {cli}"}

    actions: list[dict[str, object]] = []
    for gate, evidence in (
        ("impl_complete", f"contest-CUDA archive at {artifact.archive_path}"),
        ("real_archive_empirical", artifact.archive_path),
        ("contest_cuda", f"{artifact.score} {artifact.evidence_grade} {artifact.evidence_path}"),
    ):
        cmd = [
            ".venv/bin/python", str(cli), "mark",
            "--lane-id", lane_id,
            "--gate", gate,
            "--evidence", evidence,
        ]
        if dry_run:
            actions.append({"gate": gate, "command": cmd, "executed": False})
            continue
        try:
            result = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=30,
            )
            actions.append({
                "gate": gate,
                "command": cmd,
                "executed": True,
                "rc": result.returncode,
                "stdout_tail": result.stdout[-200:] if result.stdout else "",
                "stderr_tail": result.stderr[-200:] if result.stderr else "",
            })
        except Exception as e:
            actions.append({
                "gate": gate,
                "command": cmd,
                "executed": False,
                "error": str(e)[:200],
            })
    return {"status": "ATTEMPTED", "lane_id": lane_id, "actions": actions}


# ---------------------------------------------------------------------------
# Atom ledger append
# ---------------------------------------------------------------------------

def append_to_atom_ledger(
    repo_root: pathlib.Path,
    artifact: ContestCudaArtifact,
    lane_id: str,
) -> pathlib.Path:
    ledger_path = repo_root / "experiments/results/bilevel_atom_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "tools/auto_promote_contest_cuda",
        "lane_id": lane_id,
        "lane_dir": artifact.lane_dir,
        "archive_path": artifact.archive_path,
        "archive_bytes": artifact.archive_bytes,
        "archive_sha256": artifact.archive_sha256,
        "evidence_path": artifact.evidence_path,
        "evidence_grade": artifact.evidence_grade,
        "contest_cuda_score": artifact.score,
    }
    with ledger_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return ledger_path


# ---------------------------------------------------------------------------
# Reports auto-update
# ---------------------------------------------------------------------------

def update_reports_latest_if_frontier(
    repo_root: pathlib.Path,
    artifact: ContestCudaArtifact,
    lane_id: str,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Prepend a one-line frontier note to reports/latest.md if score is best yet."""
    reports_path = repo_root / "reports/latest.md"
    if not reports_path.exists():
        return {"status": "no_reports_file", "path": str(reports_path)}
    current = reports_path.read_text()
    # Find any existing best score in the file
    import re
    score_pattern = re.compile(r"score\s*[:=]\s*(\d+\.\d+)\s*\[contest-CUDA")
    matches = score_pattern.findall(current)
    existing_best = min((float(m) for m in matches), default=float("inf"))
    if artifact.score >= existing_best:
        return {
            "status": "not_a_frontier",
            "lane_id": lane_id,
            "score": artifact.score,
            "current_best": existing_best,
        }
    note = (
        f"\n## Auto-promotion {_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
        f"**New frontier**: lane `{lane_id}` lands score **{artifact.score}** "
        f"`[contest-CUDA]` (vs prior best {existing_best}).\n\n"
        f"- archive: `{artifact.archive_path}` ({artifact.archive_bytes:,} B)\n"
        f"- evidence: `{artifact.evidence_path}`\n"
        f"- SHA-256: `{artifact.archive_sha256}`\n\n"
    )
    if dry_run:
        return {
            "status": "would_update",
            "lane_id": lane_id,
            "score": artifact.score,
            "current_best": existing_best,
            "note_preview": note[:500],
        }
    # Insert just after the document title (first line starting with `# `)
    lines = current.splitlines(keepends=True)
    insert_at = 1
    for i, line in enumerate(lines[:5]):
        if line.startswith("# "):
            insert_at = i + 1
            break
    new_text = "".join(lines[:insert_at]) + note + "".join(lines[insert_at:])
    reports_path.write_text(new_text)
    return {
        "status": "frontier_updated",
        "lane_id": lane_id,
        "score": artifact.score,
        "prior_best": existing_best,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Auto-promote contest-CUDA artifacts")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--dry-run", action="store_true", help="show what would happen, don't actually mark gates / edit reports")
    p.add_argument("--results-subdir", default="experiments/results")
    args = p.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else pathlib.Path(__file__).resolve().parent.parent
    artifacts = discover_contest_cuda_artifacts(repo_root, results_subdir=args.results_subdir)

    print(f"[auto-promote] discovered {len(artifacts)} contest-CUDA artifact(s)")
    for art in artifacts:
        lane_id = lane_id_from_dir(art.lane_dir)
        print(f"\n[auto-promote] lane: {lane_id}")
        print(f"  archive: {art.archive_path} ({art.archive_bytes:,} B)")
        print(f"  score:   {art.score} {art.evidence_grade}")
        print(f"  sha256:  {art.archive_sha256}")
        gate_result = mark_lane_gates(repo_root, lane_id, art, dry_run=args.dry_run)
        print(f"  gates:   {gate_result['status']}")
        if not args.dry_run:
            ledger_path = append_to_atom_ledger(repo_root, art, lane_id)
            print(f"  ledger:  appended to {ledger_path}")
        report_result = update_reports_latest_if_frontier(repo_root, art, lane_id, dry_run=args.dry_run)
        print(f"  reports: {report_result['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
