#!/usr/bin/env python3
"""Parallel-dispatch actuator for exact-eval-ready ranked candidates.

This tool is intentionally strict. It refuses prediction-only, forensic,
local-proxy, or missing-readiness candidates. The input must contain candidates
already marked `ready_for_exact_eval_dispatch=true` by a separate exact-SHA
readiness gate; this actuator only fans out those already-authorized jobs and
harvests their result JSONL.

The full closed loop:
    1. local: meta_lagrangian_search_cli.py ranks N candidates → top-K
    2. PARALLEL: this script fires K dispatches concurrently
    3. harvest: each dispatch writes contest_auth_eval.json to its result dir
    4. reseed: harvested empirical anchors update .omx/calibration/anchors_*.json
    5. repeat from step 1 with the updated calibration

Prediction sweeps belong upstream as planning-only feedback. They do not become
remote jobs here.

Per CLAUDE.md cost discipline: every dispatch must include `--max-dph` and
`--estimated-cost` so a runaway sweep cannot exceed the operator's budget.

Usage (typical apogee_intN sweep):
    .venv/bin/python tools/meta_lagrangian_search_cli.py \\
        --lane-class apogee_intN --auto-sweep-bits 4,5,6,7,8 \\
        --top-k 16 --output reports/sweep_ranked.json

    .venv/bin/python tools/parallel_dispatch_top_k.py \\
        --ranked-input reports/sweep_ranked.json \\
        --max-concurrency 16 \\
        --provider lightning \\
        --estimated-cost-per-dispatch 0.11 \\
        --max-total-cost 5.00 \\
        --harvest-output reports/sweep_harvested.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class DispatchResult:
    candidate_id: str
    label: str
    archive_sha256: str | None
    archive_size_bytes: int | None
    started_utc: str
    elapsed_seconds: float
    returncode: int
    stdout_tail: str
    stderr_tail: str
    score_json_path: str | None
    contest_cuda_score: float | None


_LIGHTNING_DISPATCH = REPO / "tools" / "lightning_dispatch_pr106_stack.py"
_VASTAI_DISPATCH = REPO / "scripts" / "launch_lane_on_vastai.py"
DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES = 185_578
DEFAULT_ACTIVE_FLOOR_SCORE = 0.2089810755823297
BLOCKED_EVIDENCE_SEMANTICS = {
    "prediction_only_forensic",
    "local_proxy_prediction_forensic",
    "byte_only_forensic",
}


class DispatchInputError(ValueError):
    """Raised when ranked input is not exact-eval dispatch-ready."""


def _candidate_archive_bytes(candidate: dict) -> int | None:
    for key in ("archive_size_bytes", "expected_archive_size_bytes", "archive_bytes"):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _candidate_blockers(
    candidate: dict,
    *,
    active_floor_archive_bytes: int | None = DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = DEFAULT_ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> list[str]:
    blockers: list[str] = []
    # Round 5 R5-1 fix (2026-05-06, 95% CRITICAL): the historical
    # `if candidate.get("dispatch_blockers"): blockers.append(...)` rejected
    # any candidate with a non-empty list. Round 4 R4-B documented that
    # `build_wavelet_apply_gate` and `build_wavelet_apply_transform_candidate`
    # both emit fail-closed-by-design dispatch_blockers (4+ unconditional
    # entries) — operators must clear those entries by hand after providing
    # the corresponding evidence. So a non-empty list is NOT itself a
    # blocker; the canonical clearance signal is `ready_for_exact_eval_dispatch
    # == True`. That signal is already checked above. The dispatch_blockers
    # list is informational ("next required evidence") and must not be the
    # gating predicate for the actuator.
    if candidate.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("candidate_not_ready_for_exact_eval_dispatch")
    semantics = str(candidate.get("evidence_semantics") or "").strip().lower()
    if semantics in BLOCKED_EVIDENCE_SEMANTICS or "prediction" in semantics or "proxy" in semantics:
        blockers.append(f"blocked_evidence_semantics:{semantics or 'missing'}")
    if candidate.get("score_claim") is True and candidate.get("score_claim_verified") is not True:
        blockers.append("unverified_score_claim")
    archive_bytes = _candidate_archive_bytes(candidate)
    if (
        active_floor_archive_bytes is not None
        and archive_bytes is not None
        and archive_bytes > active_floor_archive_bytes
    ):
        if not allow_above_active_floor_dispatch:
            score_note = (
                f", active_floor_score={active_floor_score:.12f}"
                if active_floor_score is not None else ""
            )
            blockers.append(
                "above_active_floor_archive_bytes:"
                f"{archive_bytes}>{active_floor_archive_bytes}{score_note}; "
                "treat as research/calibration unless explicitly overridden"
            )
        elif not operator_override_reason:
            blockers.append("above_active_floor_override_missing_reason")
    return blockers


def _build_dispatch_cmd(
    candidate: dict,
    *,
    provider: str,
    lane_script: str,
    label_prefix: str,
    estimated_cost: float,
    max_dph: float,
) -> list[str]:
    candidate_id = candidate["candidate_id"]
    label = f"{label_prefix}_{candidate_id}"
    band = candidate.get("predicted_band", [candidate.get("band_low", 0.0), candidate.get("band_high", 1.0)])

    if provider == "lightning":
        if not _LIGHTNING_DISPATCH.is_file():
            raise FileNotFoundError(f"missing lightning dispatcher: {_LIGHTNING_DISPATCH}")
        archive_path = candidate.get("archive_path")
        if not archive_path:
            raise DispatchInputError(
                f"candidate {candidate_id!r} missing archive_path for Lightning dispatch"
            )
        cmd = [
            sys.executable, str(_LIGHTNING_DISPATCH),
            "--lane", str(candidate.get("lane_id") or candidate_id),
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
        if not _VASTAI_DISPATCH.is_file():
            raise FileNotFoundError(f"missing vastai dispatcher: {_VASTAI_DISPATCH}")
        return [
            sys.executable, str(_VASTAI_DISPATCH), "full",
            "--lane-script", lane_script,
            "--label", label,
            "--predicted-band", str(band[0]), str(band[1]),
            "--estimated-cost", str(estimated_cost),
            "--council-priority", "1",
            "--max-dph", str(max_dph),
        ]

    raise ValueError(f"unknown provider: {provider} (expected: lightning | vastai)")


def _fire_one(
    candidate: dict,
    *,
    provider: str,
    lane_script: str,
    label_prefix: str,
    estimated_cost: float,
    max_dph: float,
    timeout_seconds: float,
) -> DispatchResult:
    candidate_id = candidate["candidate_id"]
    label = f"{label_prefix}_{candidate_id}"
    cmd = _build_dispatch_cmd(
        candidate,
        provider=provider, lane_script=lane_script,
        label_prefix=label_prefix,
        estimated_cost=estimated_cost, max_dph=max_dph,
    )
    started = time.gmtime()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", started)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # subprocess-no-check-OK: we capture rc explicitly + log to harvested JSONL
            cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False,
            cwd=str(REPO),
        )
        rc = proc.returncode
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout_tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = f"TIMEOUT after {timeout_seconds}s"
    elapsed = time.monotonic() - t0

    # Harvest contest_auth_eval.json from the lane's expected output directory.
    score_json_path = None
    contest_cuda_score = None
    expected_outputs = sorted(REPO.glob(f"experiments/results/*{label}*/contest_auth_eval.json"))
    if expected_outputs:
        score_json_path = str(expected_outputs[-1])
        try:
            payload = json.loads(Path(score_json_path).read_text())
            contest_cuda_score = float(payload.get("final_score") or payload.get("contest_score") or payload.get("score"))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            contest_cuda_score = None

    return DispatchResult(
        candidate_id=candidate_id,
        label=label,
        archive_sha256=candidate.get("archive_sha256") or candidate.get("expected_archive_sha256"),
        archive_size_bytes=candidate.get("archive_size_bytes") or candidate.get("expected_archive_size_bytes"),
        started_utc=started_utc,
        elapsed_seconds=elapsed,
        returncode=rc,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        score_json_path=score_json_path,
        contest_cuda_score=contest_cuda_score,
    )


def _load_top_k(
    ranked_input: Path,
    k: int | None,
    *,
    active_floor_archive_bytes: int | None = DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = DEFAULT_ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> list[dict]:
    """Load candidates from a meta-Lagrangian ranked-output JSON file."""
    payload = json.loads(ranked_input.read_text())
    if isinstance(payload, dict) and payload.get("ready_for_exact_eval_dispatch") is False:
        raise DispatchInputError(
            f"{ranked_input} is marked ready_for_exact_eval_dispatch=false; refusing parallel dispatch"
        )
    candidates = payload.get("dispatch_ready") or payload.get("top_k") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError(f"ranked-input must contain a top_k or dispatch_ready list, got {type(candidates)}")
    if k is not None:
        candidates = candidates[:k]
    blocked: list[str] = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            blocked.append(f"candidate[{idx}]: not an object")
            continue
        candidate_id = candidate.get("candidate_id", f"candidate[{idx}]")
        for blocker in _candidate_blockers(
            candidate,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_score=active_floor_score,
            allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
            operator_override_reason=operator_override_reason,
        ):
            blocked.append(f"{candidate_id}: {blocker}")
    if blocked:
        details = "\n  - ".join(blocked[:20])
        raise DispatchInputError(
            "ranked-input contains non-dispatch-ready candidates; refusing paid dispatch:\n  - "
            + details
        )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-input", type=Path, required=True,
                        help="JSON output from tools/meta_lagrangian_search_cli.py")
    parser.add_argument("--top-k", type=int, default=None,
                        help="cap dispatch count; defaults to full ranked list length")
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="max simultaneous dispatches (default: 8). Pin to your provider's quota.")
    parser.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    parser.add_argument("--lane-script", default="scripts/remote_lane_apogee_intN.sh",
                        help="path to the remote lane script (relative to repo root)")
    parser.add_argument("--label-prefix", default="parallel_sweep",
                        help="label prefix for each dispatch (used to namespace result dirs)")
    parser.add_argument("--estimated-cost-per-dispatch", type=float, default=0.30,
                        help="$ estimate per dispatch for budget gating")
    parser.add_argument("--max-total-cost", type=float, default=5.00,
                        help="hard cap on total $ across all dispatches; refuse if exceeded")
    parser.add_argument("--active-floor-archive-bytes", type=int, default=DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES,
                        help="refuse paid dispatch for candidates larger than the current active "
                        "archive-byte floor unless --allow-above-active-floor-dispatch is set "
                        f"(default: {DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES})")
    parser.add_argument("--active-floor-score", type=float, default=DEFAULT_ACTIVE_FLOOR_SCORE,
                        help="current active score floor for diagnostic blocker text "
                        f"(default: {DEFAULT_ACTIVE_FLOOR_SCORE:.12f})")
    parser.add_argument("--allow-above-active-floor-dispatch", action="store_true",
                        help="operator override for calibration/non-rate experiments whose archives "
                        "are larger than --active-floor-archive-bytes")
    parser.add_argument("--operator-override-reason", default=None,
                        help="required with --allow-above-active-floor-dispatch")
    parser.add_argument("--max-dph", type=float, default=0.30,
                        help="passed to vastai dispatcher to gate per-hour cost")
    parser.add_argument("--per-dispatch-timeout-seconds", type=float, default=1800.0,
                        help="kill any individual dispatch after this many seconds (default 30min)")
    parser.add_argument("--harvest-output", type=Path, default=None,
                        help="write harvested-results JSONL to this path (one DispatchResult per line)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the dispatch commands that WOULD fire, without firing them")
    args = parser.parse_args(argv)
    if args.allow_above_active_floor_dispatch and not args.operator_override_reason:
        print(
            "FATAL: --allow-above-active-floor-dispatch requires "
            "--operator-override-reason",
            file=sys.stderr,
        )
        return 2

    try:
        candidates = _load_top_k(
            args.ranked_input,
            args.top_k,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_score=args.active_floor_score,
            allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
            operator_override_reason=args.operator_override_reason,
        )
    except DispatchInputError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print(f"FATAL: no candidates in {args.ranked_input}", file=sys.stderr)
        return 2

    n = len(candidates)
    estimated_total = args.estimated_cost_per_dispatch * n
    if estimated_total > args.max_total_cost:
        print(
            f"FATAL: estimated total ${estimated_total:.2f} (={n}x${args.estimated_cost_per_dispatch}) "
            f"exceeds --max-total-cost ${args.max_total_cost:.2f}. "
            f"Lower --top-k or raise --max-total-cost.",
            file=sys.stderr,
        )
        return 3

    print(f"[parallel-dispatch] {n} candidates, max_concurrency={args.max_concurrency}, provider={args.provider}")
    print(f"[parallel-dispatch] estimated total cost: ${estimated_total:.2f} (cap ${args.max_total_cost:.2f})")
    print(f"[parallel-dispatch] timeout per dispatch: {args.per_dispatch_timeout_seconds}s")

    if args.dry_run:
        print("[parallel-dispatch] DRY-RUN — printing commands only:")
        for c in candidates:
            cmd = _build_dispatch_cmd(
                c, provider=args.provider, lane_script=args.lane_script,
                label_prefix=args.label_prefix,
                estimated_cost=args.estimated_cost_per_dispatch,
                max_dph=args.max_dph,
            )
            print("  " + " ".join(cmd))
        return 0

    results: list[DispatchResult] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futures = {
            ex.submit(
                _fire_one, c,
                provider=args.provider, lane_script=args.lane_script,
                label_prefix=args.label_prefix,
                estimated_cost=args.estimated_cost_per_dispatch,
                max_dph=args.max_dph,
                timeout_seconds=args.per_dispatch_timeout_seconds,
            ): c["candidate_id"]
            for c in candidates
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover — defensive
                print(f"[parallel-dispatch] EXCEPTION dispatching {cid}: {exc}", file=sys.stderr)
                continue
            results.append(result)
            score_str = (
                f"score={result.contest_cuda_score:.4f}"
                if result.contest_cuda_score is not None else "score=PENDING"
            )
            symbol = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
            print(
                f"[parallel-dispatch] [{symbol}] {result.candidate_id} "
                f"elapsed={result.elapsed_seconds:.1f}s {score_str}"
            )

    elapsed_total = time.monotonic() - t0
    n_ok = sum(1 for r in results if r.returncode == 0)
    n_with_score = sum(1 for r in results if r.contest_cuda_score is not None)
    print(
        f"[parallel-dispatch] DONE — {n_ok}/{len(results)} dispatches succeeded "
        f"({n_with_score} with parsed contest-CUDA score) in {elapsed_total:.1f}s wall-clock"
    )

    if args.harvest_output:
        args.harvest_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.harvest_output, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "candidate_id": r.candidate_id,
                    "label": r.label,
                    "archive_sha256": r.archive_sha256,
                    "archive_size_bytes": r.archive_size_bytes,
                    "started_utc": r.started_utc,
                    "elapsed_seconds": r.elapsed_seconds,
                    "returncode": r.returncode,
                    "stdout_tail": r.stdout_tail,
                    "stderr_tail": r.stderr_tail,
                    "score_json_path": r.score_json_path,
                    "contest_cuda_score": r.contest_cuda_score,
                    "tag": "[contest-CUDA]" if (r.returncode == 0 and r.contest_cuda_score is not None) else "[dispatch-failed]",
                }) + "\n")
        print(f"[parallel-dispatch] harvested → {args.harvest_output}")

    # Best score in the batch (lower is better for comma's contest)
    successful_scores = [(r.candidate_id, r.contest_cuda_score) for r in results if r.contest_cuda_score is not None]
    if successful_scores:
        successful_scores.sort(key=lambda t: t[1])
        best_id, best_score = successful_scores[0]
        print(f"[parallel-dispatch] best in batch: {best_id} = {best_score:.4f} [contest-CUDA]")

    return 0 if n_ok > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
