#!/usr/bin/env python3
"""Closed-loop feedback sweep scaffold: rank -> dispatch-ready filter -> harvest.

This is retained as a recovered research scaffold, but it is fail-closed by
default. It must not turn Apogee/prediction/local-proxy rankings into paid jobs.
Each cycle:

    1. RANK — meta_lagrangian.MetaLagrangianSearch ranks candidates using
              current calibration anchors (proxy is closed-form, microseconds)
    2. FILTER — only candidates already marked ready_for_exact_eval_dispatch
                by exact-SHA non-proxy readiness evidence survive
    3. DISPATCH — surviving candidates fan out only when --allow-paid-dispatch
                  is explicit
    4. HARVEST — collect contest-CUDA scores + per-component (pose, seg)
                 distortion from each dispatch's contest_auth_eval.json
    5. RESEED — append every successful [contest-CUDA] result as a new
                empirical anchor, strengthening the calibration for cycle N+1
    6. CONVERGE — if best score in last 2 cycles is within --convergence-eps
                  of best ever, stop. Else loop.

Per CLAUDE.md cost discipline: --max-total-cost gates the entire sweep, and
--max-cost-per-cycle gates each cycle. Budget exhaustion stops the loop.

Per CLAUDE.md "Auth eval EVERYWHERE": only [contest-CUDA] tagged rows update
the anchors. Failed dispatches are recorded but do not pollute calibration.

Race mode narrows top-K and budget only. It does not drop exact-readiness gates.

Usage (typical apogee_intN sweep, ~$5 budget, 4 cycles):
    .venv/bin/python tools/feedback_loop_sweep.py \\
        --candidate-generator apogee_intN \\
        --anchors-path .omx/calibration/anchors_apogee_intN.json \\
        --top-k 16 --max-cycles 4 \\
        --max-total-cost 5.00 --max-cost-per-cycle 1.50 \\
        --provider lightning \\
        --output-dir reports/feedback_loop_$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from experiments.distortion_proxy_local import make_distortion_proxy  # type: ignore  # noqa: E402
from tac.optimizer.meta_lagrangian import (  # noqa: E402
    LagrangianConstraints,
    MetaLagrangianSearch,
)
from tac.predictor.score_band import load_calibration_anchors  # noqa: E402

BLOCKED_EVIDENCE_SEMANTICS = {
    "prediction_only_forensic",
    "local_proxy_prediction_forensic",
    "byte_only_forensic",
}


@dataclass
class CycleRecord:
    cycle: int
    started_utc: str
    elapsed_seconds: float
    n_candidates_ranked: int
    n_dispatched: int
    n_succeeded: int
    n_with_score: int
    best_score: float | None
    best_candidate_id: str | None
    cost_estimated: float
    cost_cumulative: float
    new_anchors_added: int
    converged: bool
    notes: list[str] = field(default_factory=list)


# ── Candidate generators ──────────────────────────────────────────────────


def _generator_apogee_intn() -> list[dict]:
    """Walk experiments/results/apogee_int<N>_repack_*/ for ready archives."""
    candidates = []
    for n in (4, 5, 6, 7, 8):
        repack_dirs = sorted(REPO.glob(f"experiments/results/apogee_int{n}_repack_*"))
        if not repack_dirs:
            continue
        repack_dir = repack_dirs[-1]
        meta_path = repack_dir / "repack_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        archive_path = repack_dir / f"apogee_int{n}_archive.zip"
        archive_bytes = meta.get("archive_size_bytes")
        rel_err = meta.get("rel_err_pct_per_weight")
        if archive_bytes is None or rel_err is None or not archive_path.is_file():
            continue
        candidates.append({
            "candidate_id": f"apogee_int{n}",
            "archive_bytes": archive_bytes,
            "rel_err_pct": rel_err,
            "n_layers": meta.get("n_intn_layers", 13),
            "lane_class": "apogee_intN",
            "archive_path": archive_path,
            "archive_sha256": meta.get("candidate_archive_sha256"),
            "ready_for_exact_eval_dispatch": False,
            "evidence_semantics": "byte_only_forensic",
            "dispatch_blockers": [
                "missing_contest_faithful_distortion_model",
                "missing_exact_sha_non_proxy_readiness_evidence",
            ],
            "score_claim": False,
        })
    return candidates


GENERATORS = {"apogee_intN": _generator_apogee_intn}


def _candidate_dispatch_blockers(candidate: dict) -> list[str]:
    blockers: list[str] = []
    if candidate.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("candidate_not_ready_for_exact_eval_dispatch")
    semantics = str(candidate.get("evidence_semantics") or "").strip().lower()
    if semantics in BLOCKED_EVIDENCE_SEMANTICS or "prediction" in semantics or "proxy" in semantics:
        blockers.append(f"blocked_evidence_semantics:{semantics or 'missing'}")
    if candidate.get("dispatch_blockers"):
        blockers.append("candidate_has_dispatch_blockers")
    if candidate.get("score_claim") is True and candidate.get("score_claim_verified") is not True:
        blockers.append("unverified_score_claim")
    return blockers


# ── Dispatch + harvest (in-process, replaces 3-binary chain) ──────────────


def _build_dispatch_cmd(candidate: dict, *, provider: str, lane_script: str,
                       label_prefix: str, max_dph: float, estimated_cost: float) -> list[str]:
    label = f"{label_prefix}_{candidate['candidate_id']}"
    band = candidate.get("predicted_band") or [
        candidate.get("band_low", 0.0), candidate.get("band_high", 1.0)
    ]
    if provider == "lightning":
        archive_path = candidate.get("archive_path")
        if not archive_path:
            raise ValueError(
                f"candidate {candidate['candidate_id']!r} missing archive_path "
                "for Lightning dispatch"
            )
        cmd = [
            sys.executable, str(REPO / "tools/lightning_dispatch_pr106_stack.py"),
            "--lane", str(candidate.get("lane_id") or candidate["candidate_id"]),
            "--archive", str(archive_path),
            "--predicted-low", str(band[0]),
            "--predicted-high", str(band[1]),
            "--job-name", label,
        ]
        gate_json = candidate.get("apogee_distortion_gate_json")
        if gate_json:
            cmd += ["--apogee-distortion-gate-json", str(gate_json)]
        _ = lane_script
        return cmd
    if provider == "vastai":
        return [
            sys.executable, str(REPO / "scripts/launch_lane_on_vastai.py"), "full",
            "--lane-script", lane_script, "--label", label,
            "--predicted-band", str(band[0]), str(band[1]),
            "--estimated-cost", str(estimated_cost),
            "--council-priority", "1", "--max-dph", str(max_dph),
        ]
    raise ValueError(f"unknown provider: {provider}")


def _fire_one(candidate: dict, *, provider: str, lane_script: str, label_prefix: str,
             max_dph: float, estimated_cost: float, timeout: float, dry_run: bool) -> dict:
    label = f"{label_prefix}_{candidate['candidate_id']}"
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    if dry_run:
        cmd = _build_dispatch_cmd(candidate, provider=provider, lane_script=lane_script,
                                  label_prefix=label_prefix, max_dph=max_dph,
                                  estimated_cost=estimated_cost)
        return {
            "candidate_id": candidate["candidate_id"], "label": label,
            "started_utc": started, "elapsed_seconds": 0.0,
            "returncode": 0, "stdout_tail": "DRY-RUN: " + " ".join(cmd),
            "stderr_tail": "", "score_json_path": None,
            "contest_cuda_score": None, "avg_pose_dist": None, "avg_seg_dist": None,
            "tag": "[dry-run]",
        }
    cmd = _build_dispatch_cmd(candidate, provider=provider, lane_script=lane_script,
                              label_prefix=label_prefix, max_dph=max_dph,
                              estimated_cost=estimated_cost)
    try:
        proc = subprocess.run(  # subprocess-no-check-OK: rc + tags captured for harvest
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
            cwd=str(REPO),
        )
        rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        out = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        err = f"TIMEOUT after {timeout}s"
    elapsed = time.monotonic() - t0

    score_json = None
    score = pose = seg = None
    matches = sorted(REPO.glob(f"experiments/results/*{label}*/contest_auth_eval.json"))
    if matches:
        score_json = str(matches[-1])
        try:
            payload = json.loads(Path(score_json).read_text())
            score = float(payload.get("final_score") or payload.get("contest_score") or payload.get("score"))
            pose = payload.get("avg_pose_dist") or payload.get("pose_dist")
            seg = payload.get("avg_seg_dist") or payload.get("seg_dist")
            pose = float(pose) if pose is not None else None
            seg = float(seg) if seg is not None else None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    tag = "[contest-CUDA]" if (rc == 0 and score is not None) else (
        f"[dispatch-failed rc={rc}]" if rc != 0 else "[no-score-json]"
    )
    return {
        "candidate_id": candidate["candidate_id"], "label": label,
        "started_utc": started, "elapsed_seconds": elapsed,
        "returncode": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:],
        "score_json_path": score_json, "contest_cuda_score": score,
        "avg_pose_dist": pose, "avg_seg_dist": seg, "tag": tag,
        "archive_sha256": candidate.get("archive_sha256"),
        "archive_bytes": candidate.get("archive_bytes"),
        "rel_err_pct": candidate.get("rel_err_pct"),
        "n_layers": candidate.get("n_layers"),
    }


def _reseed_anchors(harvested: list[dict], anchors_path: Path,
                   lossless_pose: float, lossless_seg: float) -> int:
    """Append [contest-CUDA] tagged results to the anchor file. Returns count added."""
    if not anchors_path.is_file():
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        anchors_path.write_text("[]")
    existing = json.loads(anchors_path.read_text())
    existing_labels = {a.get("lane_id") for a in existing}
    added = []
    for h in harvested:
        if h["tag"] != "[contest-CUDA]":  # CUSTODY_VALIDATOR_OK: legacy harvested JSONL prefilter; rows are diagnostic calibration anchors, not direct score promotion
            continue
        if h["label"] in existing_labels:
            continue
        if h["rel_err_pct"] is None or h["archive_bytes"] is None:
            continue
        # Round 2B B4 fix (2026-05-06, 83% confidence): record the SOURCE of
        # avg_pose_dist + avg_seg_dist so the predictor can downweight anchors
        # whose distortions came from the lossless fallback rather than from
        # an actual contest-CUDA measurement. Silent fill-in was the Q-FAITHFUL
        # ghost-evidence pattern. The source field is required for any future
        # weight or filter logic in the meta-Lagrangian engine.
        pose_source = "contest_cuda_measured" if h["avg_pose_dist"] is not None else "fallback_lossless"
        seg_source = "contest_cuda_measured" if h["avg_seg_dist"] is not None else "fallback_lossless"
        added.append({
            "lane_id": h["label"],
            "rel_err_pct_per_weight": h["rel_err_pct"],
            "archive_bytes": h["archive_bytes"],
            "contest_cuda_score": h["contest_cuda_score"],
            "avg_pose_dist": h["avg_pose_dist"] if h["avg_pose_dist"] is not None else lossless_pose,
            "avg_seg_dist": h["avg_seg_dist"] if h["avg_seg_dist"] is not None else lossless_seg,
            "avg_pose_dist_source": pose_source,
            "avg_seg_dist_source": seg_source,
            "rate_unscaled": h["archive_bytes"] / 37545489,
            "measured_utc": h["started_utc"],
            "job_id": h["label"],
            "archive_sha256": (h.get("archive_sha256") or "")[:16] or "unknown",
            "harvested_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    if added:
        anchors_path.write_text(json.dumps(existing + added, indent=2))
    return len(added)


# ── The closed loop ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-generator", choices=list(GENERATORS), default="apogee_intN")
    parser.add_argument("--anchors-path", type=Path, default=None,
                        help="defaults to .omx/calibration/anchors_<generator>.json")
    parser.add_argument("--top-k", type=int, default=8,
                        help="dispatch this many highest-ranked candidates per cycle")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--max-total-cost", type=float, default=5.00)
    parser.add_argument("--max-cost-per-cycle", type=float, default=1.50)
    parser.add_argument("--estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--max-dph", type=float, default=0.30)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--per-dispatch-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    parser.add_argument("--lane-script", default="scripts/remote_lane_apogee_intN.sh")
    parser.add_argument("--label-prefix", default="feedback_loop")
    parser.add_argument("--convergence-eps", type=float, default=0.0005,
                        help="if best-of-cycle improvement < this for 2 cycles, stop")
    parser.add_argument("--race-mode", action="store_true",
                        help="leaderboard-move mode: narrow top-K to 4, drop sanity gate, fire fast")
    parser.add_argument("--lossless-pose-dist", type=float, default=3.4e-5)
    parser.add_argument("--lossless-seg-dist", type=float, default=0.00067819)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="per-cycle JSONL + summary written here")
    parser.add_argument("--dry-run", action="store_true",
                        help="print dispatch commands per cycle without firing")
    parser.add_argument("--allow-paid-dispatch", action="store_true",
                        help="required for non-dry-run dispatch after exact-readiness filtering")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = args.anchors_path or (
        REPO / ".omx" / "calibration" / f"anchors_{args.candidate_generator}.json"
    )

    if args.race_mode:
        args.top_k = min(args.top_k, 4)
        args.max_cost_per_cycle = min(args.max_cost_per_cycle, 0.50)
        print("[feedback-loop] RACE MODE: top_k=4, max_cost_per_cycle=$0.50, exact-readiness gates still active")

    if not args.dry_run and not args.allow_paid_dispatch:
        print(
            "FATAL: non-dry-run feedback sweeps require --allow-paid-dispatch after exact-readiness filtering",
            file=sys.stderr,
        )
        return 2
    if args.allow_paid_dispatch:
        print(
            "FATAL: tools/feedback_loop_sweep.py paid dispatch is retired. "
            "This recovered scaffold has only shallow candidate checks and must "
            "not launch Lightning/Vast.ai jobs directly. Generate a ranked "
            "queue, promote it through tools/promote_optimizer_candidate_for_exact_eval.py, "
            "then dispatch through the audited exact-ready actuator.",
            file=sys.stderr,
        )
        return 2

    # Generator function
    generator = GENERATORS[args.candidate_generator]

    cycles: list[CycleRecord] = []
    cumulative_cost = 0.0
    best_ever = float("inf")
    best_ever_id: str | None = None
    cycles_without_improvement = 0

    for cycle_idx in range(1, args.max_cycles + 1):
        if cumulative_cost >= args.max_total_cost:
            print(f"[feedback-loop] BUDGET EXHAUSTED: cumulative ${cumulative_cost:.2f} >= cap ${args.max_total_cost:.2f}")
            break

        cycle_started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        cycle_t0 = time.monotonic()
        print(f"\n=== CYCLE {cycle_idx}/{args.max_cycles} START {cycle_started} ===")

        # 1. RANK — load anchors fresh each cycle (so reseeds take effect)
        anchors = load_calibration_anchors(anchors_path) if anchors_path.is_file() else []
        proxy = make_distortion_proxy(anchors_path)
        search = MetaLagrangianSearch(
            calibration_anchors=anchors, distortion_proxy=proxy,
            constraints=LagrangianConstraints(),
        )
        all_candidates = generator()
        if not all_candidates:
            print(f"[cycle {cycle_idx}] no candidates from generator — stopping")
            break
        evaluations = search.evaluate_all(all_candidates)
        # Use forensic rank (lagrangian-sorted) since sanity gate may block
        # all eligible_for_dispatch due to int4-falsification scar
        ranked = sorted(evaluations, key=lambda e: (e.lagrangian, e.candidate_id))
        top_k = ranked[:args.top_k]

        # Re-attach archive metadata for dispatch
        cand_lookup = {c["candidate_id"]: c for c in all_candidates}
        dispatch_set = []
        for ev in top_k:
            c = cand_lookup.get(ev.candidate_id, {}).copy()
            c["predicted_band"] = [ev.band_low, ev.band_high]
            c.setdefault("ready_for_exact_eval_dispatch", False)
            c.setdefault("evidence_semantics", "local_proxy_prediction_forensic")
            dispatch_set.append(c)

        blocked = {
            c["candidate_id"]: _candidate_dispatch_blockers(c)
            for c in dispatch_set
            if _candidate_dispatch_blockers(c)
        }
        if blocked:
            print(f"[cycle {cycle_idx}] exact-readiness filter blocked {len(blocked)} candidate(s):")
            for candidate_id, blockers in list(blocked.items())[:20]:
                print(f"  - {candidate_id}: {', '.join(blockers)}")
        dispatch_set = [c for c in dispatch_set if not _candidate_dispatch_blockers(c)]
        if not dispatch_set:
            print(f"[cycle {cycle_idx}] no exact-eval-ready candidates after readiness filtering — stopping")
            break

        cycle_cost_est = args.estimated_cost_per_dispatch * len(dispatch_set)
        if cycle_cost_est > args.max_cost_per_cycle:
            keep = max(1, int(args.max_cost_per_cycle / args.estimated_cost_per_dispatch))
            print(f"[cycle {cycle_idx}] cost ${cycle_cost_est:.2f} > per-cycle cap ${args.max_cost_per_cycle:.2f}; trimming top-K from {len(dispatch_set)} to {keep}")
            dispatch_set = dispatch_set[:keep]
            cycle_cost_est = args.estimated_cost_per_dispatch * len(dispatch_set)
        if cumulative_cost + cycle_cost_est > args.max_total_cost:
            keep = max(0, int((args.max_total_cost - cumulative_cost) / args.estimated_cost_per_dispatch))
            print(f"[cycle {cycle_idx}] would breach total budget; trimming to {keep}")
            dispatch_set = dispatch_set[:keep]
            cycle_cost_est = args.estimated_cost_per_dispatch * len(dispatch_set)
        if not dispatch_set:
            print(f"[cycle {cycle_idx}] no candidates after budget trim — stopping")
            break

        print(f"[cycle {cycle_idx}] ranked {len(evaluations)} candidates, dispatching top-{len(dispatch_set)} at est ${cycle_cost_est:.2f}")
        for c in dispatch_set:
            print(f"  → {c['candidate_id']}  archive_bytes={c.get('archive_bytes')}  rel_err={c.get('rel_err_pct')}%")

        # 2. DISPATCH — fan out concurrently
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(dispatch_set))) as ex:
            futures = {
                ex.submit(_fire_one, c, provider=args.provider, lane_script=args.lane_script,
                         label_prefix=f"{args.label_prefix}_c{cycle_idx}",
                         max_dph=args.max_dph,
                         estimated_cost=args.estimated_cost_per_dispatch,
                         timeout=args.per_dispatch_timeout_seconds,
                         dry_run=args.dry_run): c["candidate_id"]
                for c in dispatch_set
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"  [{cid}] EXCEPTION: {exc}")
                    continue
                results.append(r)
                score_str = f"score={r['contest_cuda_score']:.4f}" if r["contest_cuda_score"] is not None else "score=PENDING"
                print(f"  [{cid}] {r['tag']} elapsed={r['elapsed_seconds']:.1f}s {score_str}")

        # 3. HARVEST + write per-cycle JSONL
        cycle_jsonl = args.output_dir / f"cycle_{cycle_idx:02d}_harvested.jsonl"
        with open(cycle_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        cumulative_cost += cycle_cost_est

        # 4. RESEED — only [contest-CUDA] rows
        n_added = _reseed_anchors(results, anchors_path,
                                  lossless_pose=args.lossless_pose_dist,
                                  lossless_seg=args.lossless_seg_dist)

        # 5. CONVERGE check
        successful = [r for r in results if r["contest_cuda_score"] is not None]
        cycle_best = min((r["contest_cuda_score"] for r in successful), default=None)
        cycle_best_id = None
        if cycle_best is not None:
            cycle_best_id = next(r["candidate_id"] for r in successful if r["contest_cuda_score"] == cycle_best)
            improvement = best_ever - cycle_best
            if cycle_best < best_ever:
                best_ever = cycle_best
                best_ever_id = cycle_best_id
                if improvement < args.convergence_eps:
                    cycles_without_improvement += 1
                else:
                    cycles_without_improvement = 0
            else:
                cycles_without_improvement += 1
        else:
            cycles_without_improvement += 1
        converged = cycles_without_improvement >= 2

        record = CycleRecord(
            cycle=cycle_idx, started_utc=cycle_started,
            elapsed_seconds=time.monotonic() - cycle_t0,
            n_candidates_ranked=len(evaluations), n_dispatched=len(dispatch_set),
            n_succeeded=sum(1 for r in results if r["returncode"] == 0),
            n_with_score=len(successful), best_score=cycle_best,
            best_candidate_id=cycle_best_id,
            cost_estimated=cycle_cost_est, cost_cumulative=cumulative_cost,
            new_anchors_added=n_added, converged=converged,
        )
        cycles.append(record)
        print(f"[cycle {cycle_idx}] DONE — best={cycle_best} new_anchors={n_added} cumulative_cost=${cumulative_cost:.2f}")
        if converged:
            print(f"[cycle {cycle_idx}] CONVERGED — best score has not improved by ≥{args.convergence_eps} for 2 cycles")
            break

    # Final summary
    summary = {
        "schema": "feedback_loop_sweep_v1",
        "candidate_generator": args.candidate_generator,
        "anchors_path": str(anchors_path),
        "n_cycles_completed": len(cycles),
        "max_cycles_allowed": args.max_cycles,
        "best_score_overall": best_ever if best_ever != float("inf") else None,
        "best_candidate_overall": best_ever_id,
        "cumulative_cost": cumulative_cost,
        "max_total_cost": args.max_total_cost,
        "race_mode": args.race_mode,
        "dry_run": args.dry_run,
        "cycles": [
            {
                "cycle": r.cycle, "started_utc": r.started_utc,
                "elapsed_seconds": r.elapsed_seconds,
                "n_candidates_ranked": r.n_candidates_ranked,
                "n_dispatched": r.n_dispatched, "n_succeeded": r.n_succeeded,
                "n_with_score": r.n_with_score, "best_score": r.best_score,
                "best_candidate_id": r.best_candidate_id,
                "cost_estimated": r.cost_estimated, "cost_cumulative": r.cost_cumulative,
                "new_anchors_added": r.new_anchors_added, "converged": r.converged,
            }
            for r in cycles
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[feedback-loop] SUMMARY → {summary_path}")
    print(f"[feedback-loop] best overall: {best_ever_id} = {best_ever if best_ever != float('inf') else 'NONE'}")
    print(f"[feedback-loop] cycles completed: {len(cycles)}/{args.max_cycles}")
    print(f"[feedback-loop] total spend: ${cumulative_cost:.2f} / ${args.max_total_cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
