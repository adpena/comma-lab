#!/usr/bin/env python3
"""pointer_move_packet.py — an exact row lands, the apparatus writes every consequence.

Operator binding 2026-09-04: *"much of this requires a more permanent solution."*
Two exact contest-CUDA rows landed that day and MAIN hand-executed ~10 obligatory
consequences for each. Three of the hand steps went wrong in one day (a mistyped
archive sha, a truncated runtime-tree sha, a mis-computed rate-corner demand), and
the ``#316`` citation gate failed twice on the way. This tool executes the whole
list from the harvest payload so MAIN only REVIEWS.

Stages, in order, each refusing rather than proceeding on doubt:

  1 ROW        recompute S from the evaluator's components (never ``final_score``)
               and cross-check against the evaluator's OWN printed report.txt,
               parsing its thousands-separated byte count.
  2 SEAL       verify the archive on disk hashes to the scored sha at the scored size,
               and that the candidate seal (when one is given) names the same bytes.
  3 POINTER    refresh `.omx/state/canonical_frontier_pointer.json` from local state.
  4 SURFACES   regenerate the frontier citation surfaces and ASSERT the Catalog #316
               staleness gate passes strict. The gate failing is what cost MAIN two
               rounds; asserting it here means it can never be discovered later.
  5 LANE       register the lane if absent and mark impl_complete /
               real_archive_empirical / the axis gate with evidence from the receipts.
  6 CUSTODY    duplicate archive (+ seal) to the OTHER SSD tier, sha-verified after
               the copy, choosing the tier by measured free space.
  7 MEMO       render the pointer-move memo from a template: delta table vs the prior
               pointer, projection fidelity, the re-derived sub-0.12 arithmetic,
               custody, "what this does not claim", the equations leg.
  8 HOT-STATE  set the POINTER_LINE section of `.omx/state/main_hot_state.md`.
  9 EVENT      append `.omx/state/pointer_move_events.jsonl`.
 10 COMMIT     PRINT the serializer command (or run it with --commit). Default is to
               print, because the contract is that MAIN reviews the memo first.

Dry by default: nothing is written without ``--apply``. ``--repo-root`` points every
stage at one tree, which is what makes a replay against a scratch copy of the state
possible without touching live state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.pointer_move import (  # noqa: E402
    HarvestRefusal,
    PacketPlan,
    PriorAnchor,
    cross_check_against_report,
    load_json,
    pointer_move_event,
    render_memo,
    render_pointer_line,
    score_row_from_harvest,
    target_arithmetic,
)

VENV_PY = str(REPO / ".venv" / "bin" / "python")


def _tool(name: str) -> str:
    """Absolute path to a repo tool.

    Relative tool paths broke the first replay: the sub-tools ran with ``cwd`` set to
    the target repo root, and a scratch root has no ``tools/`` dir, so every stage
    exited non-zero and the packet reported success over a no-op. Absolute paths plus
    ``cwd=REPO`` keep the tool location and the WRITE location independent.
    """

    return str(REPO / "tools" / name)

#: SSD custody tiers in the repo's declared priority order. The packet copies the
#: scored archive to whichever tier does NOT already hold it, and picks by measured
#: free space when both are candidates — never by a remembered number.
SSD_TIERS = ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")

AXIS_LABELS = {
    "contest_cuda": "contest-CUDA",
    "contest_cpu": "contest-CPU",
}
AXIS_GATES = {
    "contest_cuda": "contest_cuda",
    "contest_cpu": "contest_cpu",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run(cmd: list[str], *, cwd: Path, apply: bool, label: str) -> dict[str, Any]:
    """Run one sub-tool, or record what would run. Never raises on rc."""

    record: dict[str, Any] = {"label": label, "argv": cmd, "applied": apply}
    if not apply:
        return record
    proc = subprocess.run(  # subprocess-no-check-OK: rc recorded and adjudicated by the caller
        cmd, cwd=str(cwd), capture_output=True, text=True
    )
    record["rc"] = proc.returncode
    record["stdout_tail"] = proc.stdout[-1200:]
    record["stderr_tail"] = proc.stderr[-1200:]
    return record


def prior_anchor_from_pointer(repo_root: Path, axis: str) -> PriorAnchor:
    """Read the pointer BEFORE the refresh; that is what the move is measured against.

    Read from disk rather than from a memo: a memo's "prior" line is a hand-copied
    number, and copying is the step that fails.
    """

    from tac.canonical_frontier_pointer import load_canonical_frontier_pointer_lenient

    pointer = load_canonical_frontier_pointer_lenient(repo_root=repo_root)
    if pointer is None:
        return PriorAnchor(label="no prior pointer", score=None, archive_bytes=None, d_seg=None, d_pose=None)
    anchor = getattr(pointer, f"our_local_frontier_{axis}", None)
    if anchor is None:
        return PriorAnchor(
            label=f"no prior {axis} anchor", score=None, archive_bytes=None, d_seg=None, d_pose=None
        )
    data = anchor.as_dict() if hasattr(anchor, "as_dict") else dict(anchor)
    extra = data.get("extra") or {}
    lane = data.get("lane_id") or extra.get("lane_id")
    return PriorAnchor(
        label=str(lane or data.get("archive_sha256", "")[:8] or "prior"),
        score=data.get("score"),
        archive_bytes=extra.get("archive_bytes"),
        d_seg=extra.get("avg_segnet_dist"),
        d_pose=extra.get("avg_posenet_dist"),
        lane_id=lane,
        archive_sha256=data.get("archive_sha256"),
    )


def _prior_components_from_mirror(repo_root: Path, prior: PriorAnchor) -> PriorAnchor:
    """Fill the prior anchor's d_seg / d_pose from its own anchor mirror when absent.

    The pointer's ``extra`` block does not always carry the distortion components, and
    the delta table needs them. They are read from the mirror the prior row wrote —
    never re-typed from a memo, and left as ``None`` when no mirror proves them.
    """

    if prior.d_seg is not None and prior.d_pose is not None:
        return prior
    if not prior.archive_sha256:
        return prior
    mirror_dir = repo_root / "experiments/results/modal_auth_eval_mirror"
    if not mirror_dir.is_dir():
        return prior
    for path in sorted(mirror_dir.glob("contest_auth_eval_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("archive_sha256") != prior.archive_sha256:
            continue
        prior.d_seg = prior.d_seg if prior.d_seg is not None else data.get("avg_segnet_dist")
        prior.d_pose = prior.d_pose if prior.d_pose is not None else data.get("avg_posenet_dist")
        if prior.archive_bytes is None:
            prior.archive_bytes = data.get("archive_size_bytes")
        break
    return prior


def verify_archive(archive: Path | None, row) -> list[str]:
    """Refuse unless the archive on disk IS the bytes that were scored."""

    problems: list[str] = []
    if archive is None:
        return ["no --archive given: the scored bytes were not re-verified on disk"]
    if not archive.is_file():
        return [f"archive not on disk: {archive}"]
    size = archive.stat().st_size
    if size != row.archive_bytes:
        problems.append(f"archive on disk is {size:,} B; the scored row is {row.archive_bytes:,} B")
    digest = _sha256(archive)
    if digest != row.archive_sha256:
        problems.append(f"archive on disk hashes {digest}; the scored row is {row.archive_sha256}")
    return problems


def verify_seal(seal_path: Path | None, row) -> list[str]:
    """Refuse unless the candidate seal names the exact scored archive."""

    if seal_path is None:
        return []
    if not seal_path.is_file():
        return [f"seal not on disk: {seal_path}"]
    try:
        doc = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"seal unreadable: {type(exc).__name__}: {exc}"]
    archive = doc.get("archive") or {}
    problems: list[str] = []
    sealed_sha = str(archive.get("sha256") or "").lower()
    if sealed_sha and sealed_sha != row.archive_sha256.lower():
        problems.append(f"seal pins archive {sealed_sha}; the scored row is {row.archive_sha256}")
    sealed_bytes = archive.get("bytes") or archive.get("size_bytes")
    if isinstance(sealed_bytes, int) and sealed_bytes != row.archive_bytes:
        problems.append(f"seal pins {sealed_bytes:,} B; the scored row is {row.archive_bytes:,} B")
    return problems


def pick_custody_tier(archive: Path | None) -> tuple[str | None, dict[str, Any]]:
    """Choose the SSD tier that is NOT already holding this archive, by free space.

    Duplicating custody onto the tier the payload already sits on is not a second
    copy; and picking a tier from memory is how a copy lands on a nearly-full volume.
    Both facts are MEASURED here at run time.
    """

    stats: dict[str, Any] = {}
    source_tier = None
    if archive is not None:
        for tier in SSD_TIERS:
            if str(archive).startswith(tier):
                source_tier = tier
    best: tuple[int, str] | None = None
    for tier in SSD_TIERS:
        root = Path(tier)
        if not root.is_dir():
            stats[tier] = {"mounted": False}
            continue
        usage = shutil.disk_usage(root)
        stats[tier] = {
            "mounted": True,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1 << 30), 1),
            "is_source_tier": tier == source_tier,
        }
        if tier == source_tier:
            continue
        if best is None or usage.free > best[0]:
            best = (usage.free, tier)
    stats["source_tier"] = source_tier
    return (best[1] if best else None), stats


def duplicate_custody(
    *, archive: Path | None, seal: Path | None, tier: str | None, subdir: str, apply: bool
) -> dict[str, Any]:
    """Copy the scored archive (+ seal) to the second tier and VERIFY the copy's sha.

    ALWAYS KEEP THE PAYLOAD, applied to redundancy: one SSD is one failure. The copy
    is re-hashed after landing, because an unverified copy is a belief, not a backup.
    """

    record: dict[str, Any] = {"tier": tier, "subdir": subdir, "applied": apply, "copies": []}
    if archive is None or tier is None:
        record["skipped"] = "no archive given" if archive is None else "no second tier available"
        return record
    target_dir = Path(tier) / subdir
    for source in [p for p in (archive, seal) if p is not None and p.is_file()]:
        target = target_dir / source.name
        entry = {"source": str(source), "target": str(target), "source_sha256": _sha256(source)}
        if apply:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entry["target_sha256"] = _sha256(target)
            entry["verified"] = entry["target_sha256"] == entry["source_sha256"]
        record["copies"].append(entry)
    return record


def next_move_number(repo_root: Path) -> int | None:
    """One past the highest move number in the events ledger, or None when unseeded.

    Returning None rather than 1 is deliberate: guessing an ordinal would put two memos
    under one number, and a ledger with a hole in it is worse than an explicit argument.
    """

    path = repo_root / ".omx/state/pointer_move_events.jsonl"
    if not path.is_file():
        return None
    highest = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line).get("move_number")
        except json.JSONDecodeError:
            continue
        if isinstance(value, int) and value > highest:
            highest = value
    return highest + 1 if highest else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", required=True, help="MODAL_REMOTE_RESULT.json from the fire")
    ap.add_argument("--repo-root", default=str(REPO), help="tree every stage writes into (default: this repo)")
    ap.add_argument("--lane-id", default=None, help="lane the fire claimed (default: read from the anchor mirror)")
    ap.add_argument("--call-id", default=None, help="Modal call id (default: read from the fire manifest)")
    ap.add_argument("--archive", default=None, help="the scored archive.zip, re-hashed against the row")
    ap.add_argument("--seal", default=None, help="candidate seal JSON pinning the scored archive")
    ap.add_argument(
        "--move-number",
        type=int,
        default=None,
        help="ordinal of this pointer move; default: one past the highest in "
        ".omx/state/pointer_move_events.jsonl. The ledger has to be seeded once with an "
        "explicit number because the first 25 moves predate it — after that the ordinal "
        "is derived and can no longer drift.",
    )
    ap.add_argument("--memo-path", default=None, help="default: .omx/research/<lane>_pointer_move_<n>_<date>.md")
    ap.add_argument("--custody-subdir", default=None, help="dir name under the second SSD tier")
    ap.add_argument("--headline", default="", help="memo title clause (prose, the arm's claim)")
    ap.add_argument("--mechanism-file", default=None, help="file with the mechanism paragraph")
    ap.add_argument("--not-claimed-file", default=None, help="file with the 'what this does not claim' text")
    ap.add_argument("--next-file", default=None, help="file with the 'next from here' text")
    ap.add_argument("--equations-leg", default="", help="one line for the equations leg")
    ap.add_argument("--projected-score", type=float, default=None, help="the pre-fire advisory projection")
    ap.add_argument("--projection-note", default="", help="one clause about the projection instrument")
    ap.add_argument("--lane-name", default=None, help="display name when the lane must be registered")
    ap.add_argument("--lane-phase", type=float, default=8.0, help="phase for a newly registered lane")
    ap.add_argument("--pointer-line-extra", default="", help="extra bullet for the hot-state POINTER_LINE")
    ap.add_argument(
        "--allow-non-improving",
        action="store_true",
        help="run the packet even when the row does NOT beat the prior pointer on its axis "
        "(a custody/bookkeeping row); the memo then states it plainly",
    )
    ap.add_argument(
        "--no-custody",
        action="store_true",
        help="do not copy the archive to the second SSD tier (replays must not write real custody)",
    )
    ap.add_argument("--apply", action="store_true", help="write; default is a dry plan")
    ap.add_argument("--commit", action="store_true", help="run the serializer instead of printing it")
    ap.add_argument("--json", action="store_true", help="emit the packet record as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if args.move_number is None:
        derived = next_move_number(repo_root)
        if derived is None:
            print(
                "REFUSED: no --move-number and .omx/state/pointer_move_events.jsonl carries "
                "none to count from. Pass the ordinal explicitly once; after that it is derived.",
                file=sys.stderr,
            )
            return 7
        args.move_number = derived
    harvest_path = Path(args.harvest).resolve()
    packet: dict[str, Any] = {
        "schema": "pointer_move_packet.v1",
        "at_utc": _utc(),
        "repo_root": str(repo_root),
        "harvest": str(harvest_path),
        "harvest_sha256": _sha256(harvest_path) if harvest_path.is_file() else None,
        "apply": bool(args.apply),
        "stages": {},
    }

    # ---- 1 ROW ---------------------------------------------------------------------
    payload = load_json(harvest_path)
    call_id = args.call_id
    if not call_id:
        manifest = harvest_path.parent / "FIRE_MANIFEST.json"
        if manifest.is_file():
            try:
                call_id = json.loads(manifest.read_text(encoding="utf-8")).get("stage5_call_id")
            except (OSError, json.JSONDecodeError):
                call_id = None
    try:
        row = score_row_from_harvest(payload, lane_id=args.lane_id, call_id=call_id)
    except HarvestRefusal as exc:
        print(f"REFUSED (row): {exc}", file=sys.stderr)
        return 2
    if not row.passed or row.validation_errors:
        print(
            f"REFUSED (row): passed={row.passed} validation_errors={list(row.validation_errors)}; "
            "a row the evaluator did not pass is not a pointer move.",
            file=sys.stderr,
        )
        return 2
    report_text = ((payload.get("artifacts") or {}).get("report.txt")) or ""
    cross = cross_check_against_report(row, report_text) if report_text else [
        "no evaluator report.txt in the harvest artifacts; the receipt was not cross-checked"
    ]
    packet["stages"]["1_row"] = {"row": row.to_dict(), "report_cross_check": cross}
    if cross and report_text:
        for problem in cross:
            print(f"REFUSED (row): {problem}", file=sys.stderr)
        return 2

    axis = row.axis or "contest_cuda"
    axis_label = AXIS_LABELS.get(axis, axis)
    # The axis label is DERIVED from the receipt, never a remembered string: an "n600"
    # suffix on a row that scored a different sample count would be a false authority
    # claim in the memo title, and the hardware token must name what actually ran.
    hardware_bits = []
    model = (row.gpu_model or "").strip()
    for token in ("T4", "A10G", "A100", "4090", "H100", "L40S"):
        if token.lower() in model.lower():
            hardware_bits.append(token)
            break
    else:
        if model:
            hardware_bits.append(model)
    if row.n_samples is not None:
        hardware_bits.append(f"n{row.n_samples}")
    axis_label_full = " ".join([axis_label, *hardware_bits])

    prior = _prior_components_from_mirror(repo_root, prior_anchor_from_pointer(repo_root, axis))
    beats = prior.score is None or row.score < prior.score
    arithmetic = target_arithmetic(row)
    plan = PacketPlan(
        row=row,
        prior=prior,
        arithmetic=arithmetic,
        move_number=int(args.move_number),
        beats_prior=beats,
        report_cross_check=cross,
    )
    if not beats and not args.allow_non_improving:
        print(
            f"REFUSED: S {row.score} does not beat the prior {axis} pointer {prior.score}. "
            "Pass --allow-non-improving for a custody/bookkeeping row.",
            file=sys.stderr,
        )
        packet["stages"]["plan"] = plan.to_dict()
        if args.json:
            print(json.dumps(packet, indent=2))
        return 3

    # ---- 2 SEAL --------------------------------------------------------------------
    archive = Path(args.archive).resolve() if args.archive else None
    seal = Path(args.seal).resolve() if args.seal else None
    seal_problems = verify_archive(archive, row) + verify_seal(seal, row)
    plan.seal_checks = seal_problems
    packet["stages"]["2_seal"] = {
        "archive": str(archive) if archive else None,
        "seal": str(seal) if seal else None,
        "problems": seal_problems,
    }
    hard_seal_failures = [p for p in seal_problems if not p.startswith("no --archive given")]
    if hard_seal_failures:
        for problem in hard_seal_failures:
            print(f"REFUSED (seal): {problem}", file=sys.stderr)
        return 4
    packet["plan"] = plan.to_dict()

    # ---- 3 POINTER -----------------------------------------------------------------
    packet["stages"]["3_pointer"] = _run(
        [VENV_PY, _tool("refresh_canonical_frontier.py"), "--no-update-upstream", "--no-print",
         "--repo-root", str(repo_root)],
        cwd=REPO,
        apply=args.apply,
        label="refresh pointer from local state",
    )
    if args.apply and packet["stages"]["3_pointer"].get("rc"):
        print(f"REFUSED (pointer): refresh rc={packet['stages']['3_pointer']['rc']}", file=sys.stderr)
        print(packet["stages"]["3_pointer"].get("stderr_tail", ""), file=sys.stderr)
        return 6

    # ---- 4 SURFACES ----------------------------------------------------------------
    packet["stages"]["4_surfaces"] = _run(
        [VENV_PY, _tool("scan_best_anchor_per_axis.py"), "--repo-root", str(repo_root),
         "--refresh-citation-surfaces"],
        cwd=REPO,
        apply=args.apply,
        label="regenerate frontier citation surfaces",
    )
    if args.apply:
        from tac.preflight import check_reports_latest_md_not_stale_vs_canonical_frontier

        violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
            strict=False, repo_root=repo_root
        )
        packet["stages"]["4_surfaces"]["catalog_316_violations"] = list(violations)
        if violations:
            for violation in violations:
                print(f"REFUSED (#316): {violation}", file=sys.stderr)
            if args.json:
                print(json.dumps(packet, indent=2))
            return 5

    # ---- 5 LANE --------------------------------------------------------------------
    lane_id = args.lane_id or row.lane_id
    isolated = repo_root != REPO
    lane_stage: dict[str, Any] = {"lane_id": lane_id, "steps": [], "isolated_replay": isolated}
    if isolated:
        # tools/lane_maturity.py and tools/main_hot_state.py resolve their state from
        # their OWN __file__, not from a --repo-root. Running them during a replay would
        # write LIVE state, which is the one thing a replay must never do. The argv is
        # still recorded so the replay can diff what WOULD run.
        lane_stage["skipped"] = "lane tools are not --repo-root aware; not run under an isolated root"
    if lane_id and not isolated:
        registry = repo_root / ".omx/state/lane_registry.json"
        known = False
        if registry.is_file():
            try:
                known = any(
                    lane.get("id") == lane_id
                    for lane in json.loads(registry.read_text(encoding="utf-8")).get("lanes", [])
                )
            except (OSError, json.JSONDecodeError):
                known = False
        if not known:
            lane_stage["steps"].append(
                _run(
                    [VENV_PY, _tool("lane_maturity.py"), "add-lane", lane_id,
                     "--name", args.lane_name or lane_id, "--phase", str(args.lane_phase)],
                    cwd=REPO, apply=args.apply, label="register lane",
                )
            )
        evidence_common = (
            f"{axis_label} S {row.score} @ {row.archive_bytes:,} B; archive sha "
            f"{row.archive_sha256}; harvest {harvest_path}"
        )
        for gate in ("impl_complete", "real_archive_empirical", AXIS_GATES.get(axis, "contest_cuda")):
            lane_stage["steps"].append(
                _run(
                    [VENV_PY, _tool("lane_maturity.py"), "mark", lane_id, "--gate", gate,
                     "--evidence", evidence_common],
                    cwd=REPO, apply=args.apply, label=f"mark {gate}",
                )
            )
    elif not lane_id:
        lane_stage["skipped"] = "no lane id (pass --lane-id)"
    if lane_id:
        lane_stage["planned_argv"] = [
            [VENV_PY, _tool("lane_maturity.py"), "mark", lane_id, "--gate", gate, "--evidence",
             f"{axis_label} S {row.score} @ {row.archive_bytes:,} B; archive sha "
             f"{row.archive_sha256}; harvest {harvest_path}"]
            for gate in ("impl_complete", "real_archive_empirical", AXIS_GATES.get(axis, "contest_cuda"))
        ]
    packet["stages"]["5_lane"] = lane_stage

    # ---- 6 CUSTODY -----------------------------------------------------------------
    tier, tier_stats = pick_custody_tier(archive)
    subdir = args.custody_subdir or f"{lane_id or 'pointer'}/custody_pointer{plan.move_number}"
    custody_apply = args.apply and not args.no_custody
    packet["stages"]["6_custody"] = {
        "tier_stats": tier_stats,
        "suppressed": bool(args.no_custody),
        **duplicate_custody(archive=archive, seal=seal, tier=tier, subdir=subdir, apply=custody_apply),
    }

    # ---- 7 MEMO --------------------------------------------------------------------
    def _text(path: str | None, default: str) -> str:
        return Path(path).read_text(encoding="utf-8") if path else default

    date_utc = packet["at_utc"][:10]
    custody_lines = [
        f"Harvest: `{harvest_path}` (sha `{packet['harvest_sha256']}`).",
    ]
    if archive:
        custody_lines.append(f"Archive: `{archive}` (sha `{row.archive_sha256}`, {row.archive_bytes:,} B).")
    if seal:
        custody_lines.append(f"Seal: `{seal}`.")
    for copy in packet["stages"]["6_custody"].get("copies", []):
        custody_lines.append(
            f"Second copy: `{copy['target']}` (sha verified: {copy.get('verified', 'planned')})."
        )
    if not beats:
        custody_lines.append(
            "This row does NOT beat the prior pointer on its axis; it is a custody row "
            "(--allow-non-improving)."
        )
    memo = render_memo(
        plan,
        date_utc=date_utc,
        axis_label=axis_label_full,
        headline=args.headline or "exact row",
        mechanism=_text(args.mechanism_file, "_(mechanism paragraph owed by the arm that produced the row)_"),
        custody_lines=custody_lines,
        not_claimed=_text(
            args.not_claimed_file,
            "_(the arm owes the explicit non-claims: the other contest axis, any publication, "
            "and any transfer of this number to another vehicle)_",
        ),
        equations_leg=args.equations_leg or "_(equations leg owed by the arm)_",
        projection=(
            {"projected_score": args.projected_score, "note": args.projection_note}
            if args.projected_score is not None
            else None
        ),
        next_from_here=_text(args.next_file, ""),
    )
    memo_path = Path(args.memo_path) if args.memo_path else (
        repo_root / ".omx/research" / f"{lane_id or 'pointer'}_pointer_move_{plan.move_number}_{date_utc.replace('-', '')}.md"
    )
    packet["stages"]["7_memo"] = {"path": str(memo_path), "bytes": len(memo.encode("utf-8"))}
    if args.apply:
        memo_path.parent.mkdir(parents=True, exist_ok=True)
        memo_path.write_text(memo, encoding="utf-8")
    packet["memo_text"] = memo

    # ---- 8 HOT-STATE ---------------------------------------------------------------
    pointer_line = render_pointer_line(plan, axis_label=axis_label_full, extra=args.pointer_line_extra)
    hot_state_file = repo_root / ".omx/tmp" / f"pointer_line_move_{plan.move_number}.txt"
    packet["stages"]["8_hot_state"] = {"content_file": str(hot_state_file), "content": pointer_line}
    if args.apply:
        hot_state_file.parent.mkdir(parents=True, exist_ok=True)
        hot_state_file.write_text(pointer_line + "\n", encoding="utf-8")
        packet["stages"]["8_hot_state"]["run"] = _run(
            [VENV_PY, _tool("main_hot_state.py"), "--set-section", "pointer_line",
             "--content-file", str(hot_state_file)],
            cwd=REPO, apply=not isolated, label="set hot-state POINTER_LINE",
        )
        if isolated:
            packet["stages"]["8_hot_state"]["skipped"] = (
                "main_hot_state.py is not --repo-root aware; not run under an isolated root"
            )

    # ---- 9 EVENT -------------------------------------------------------------------
    event = pointer_move_event(
        plan, axis_label=axis_label_full, memo_path=str(memo_path), at_utc=packet["at_utc"]
    )
    events_path = repo_root / ".omx/state/pointer_move_events.jsonl"
    packet["stages"]["9_event"] = {"path": str(events_path), "event": event}
    if args.apply:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    # ---- 10 COMMIT -----------------------------------------------------------------
    commit_files = [str(memo_path.relative_to(repo_root))] if memo_path.is_relative_to(repo_root) else []
    commit_msg = (
        f"pointer move {plan.move_number}: S {row.score} @ {row.archive_bytes:,} B "
        f"[{axis_label_full}] — packet-written [no-triality] [p0-ledger-ok]"
    )
    commit_cmd = [
        VENV_PY, _tool("subagent_commit_serializer.py"), "--message", commit_msg, "--files", *commit_files
    ]
    packet["stages"]["10_commit"] = {"argv": commit_cmd, "ran": bool(args.commit and args.apply)}
    if args.commit and args.apply and commit_files:
        shas = [f"{f}={hashlib.sha256((repo_root / f).read_bytes()).hexdigest()}" for f in commit_files]
        for sha in shas:
            commit_cmd += ["--expected-content-sha256", sha]
        packet["stages"]["10_commit"]["run"] = _run(
            commit_cmd, cwd=REPO, apply=True, label="serializer commit"
        )

    if args.json:
        print(json.dumps(packet, indent=2))
    else:
        print(f"{'APPLIED' if args.apply else 'DRY PLAN'}: pointer move {plan.move_number}")
        print(f"  S {row.score} @ {row.archive_bytes:,} B [{axis_label_full}]")
        print(f"  prior {prior.label}: {prior.score}  delta {plan.delta_score}")
        print(f"  memo -> {memo_path}")
        print(f"  custody tier -> {tier}/{subdir}")
        if not (args.commit and args.apply):
            if commit_files:
                print("  commit:  " + " ".join(commit_cmd))
            else:
                print("  commit:  (memo is outside --repo-root; commit it by hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
