#!/usr/bin/env python3
"""V3 closed-loop compiler — STEP 1: normalize an exact-eval artifact into typed
candidate-action evidence + a machine-readable campaign decision.

This is the thin-loop ENTRY POINT of the evaluator-action waterfilling compiler
(operator directive 2026-06-09). It turns a just-landed exact-eval JSON (a
``hi_nerv_backend_only_exact_eval.v1`` artifact from the B2 bridge, or any
contest_auth_eval result carrying d_seg/d_pose/bytes) into:

  * ``candidate_action_evaluation_<tag>.v1.json``  -- the checkpoint-as-action row
    (vehicle, base/candidate sha, d_seg, d_pose, bytes, score, ΔS-vs-frontier,
    pays_rent verdict). The checkpoint IS an action atom; the frontier is the base.
  * ``campaign_decision_<tag>.v1.json``  -- the UNIVERSAL verdict
    (apply_campaign_decision: hard-fail / patch-bridge / prep-cuda / continue /
    inspect, binding constraint named) PLUS the substrate NEXT-ACTION route
    (route_substrate_next_action: the operator's 5+1 case decision tree).

Hard rules honored (CLAUDE.md + operator):
  * ΔS is computed from the EXACT contest-score formula, never from proxy telemetry.
  * Frontier is read from the canonical pointer (NEVER hardcoded literal).
  * advisory axis tag preserved verbatim; promotion/score-claim stay False.
  * NEVER auto-kills (Forbidden premature KILL) -- a high score routes to INSPECT.
  * base_archive_sha256 mismatch => the candidate is flagged STALE (not silently
    scored against the wrong base).

Reusable logic lives in ``tac.optimization.harvest_evidence``; this file is a thin
CLI per CLAUDE.md "tac stays clean; thin CLIs delegate to tac modules".

Usage:
  .venv/bin/python tools/ingest_exact_eval_to_candidate.py \\
      --exact-eval-json <run>/hi_nerv_backend_only_ep250_exact_eval.json \\
      --tag r3_ep250 \\
      --checkpoint-trajectory-dir <run>/checkpoints \\
      --output-dir <run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.optimization.harvest_evidence import (  # noqa: E402
    VEHICLE_HI_NERV,
    apply_campaign_decision,
    build_candidate_action_evaluation_row,
    build_harvest_receipt,
    route_substrate_next_action,
)

_DEFAULT_FRONTIER_POINTER = ".omx/state/canonical_frontier_pointer.json"


def _authority_tier(axis_tag: str) -> str:
    """Classify a result by AUTHORITY TIER (operator discipline 2026-06-09): the
    roadmap only updates on exact eval, never on proxy/advisory. The tier makes the
    authority explicit as a first-class field so a row can never be mistaken for
    promotion-grade evidence it is not.

      telemetry_proxy   : training-loss / PSNR / margin proxy (mechanism, NOT score)
      exact_cpu_advisory: evaluate.py on local macOS CPU/MLX (real number, NON-promotable)
      contest_cpu       : evaluate.py on Linux x86_64 (the public-leaderboard axis)
      contest_cuda      : evaluate.py on NVIDIA T4/equivalent (the CUDA promotion axis)
    Promotion grade requires BOTH contest_cpu AND contest_cuda on the same bytes.
    """
    tag = str(axis_tag).lower()
    if "contest-cuda" in tag or "contest_cuda" in tag:
        return "contest_cuda"
    if "contest-cpu" in tag or "contest_cpu" in tag:
        return "contest_cpu"
    if "macos" in tag or "mlx" in tag or "advisory" in tag:
        return "exact_cpu_advisory"
    return "telemetry_proxy"


def _metric_family(schema: str, *, has_d_seg: bool, has_d_pose: bool,
                   has_bytes: bool, ran_evaluate_py: bool) -> str:
    """Classify the METRIC FAMILY (operator anti-'metric-laundering' discipline 2026-06-09).

    authority_tier alone is insufficient: a 21.74 dB PSNR row and an evaluate.py score row
    can share schema shape. metric_family makes the KIND of number explicit so a capacity/
    proxy row can never be structurally mistaken for an exact-score row:

      exact_evaluate    : upstream evaluate.py on the full archive (d_seg/d_pose/bytes + report)
      exact_pair_scorer : the repo's own scorer over inflated frames, per-pair (diagnosis, advisory)
      psnr_capacity     : RGB/Y reconstruction PSNR (carrier capacity, NOT score)
      scorer_proxy      : MLX/training scorer surrogate (e.g. boundary-hinge; NOT official d_seg)
      telemetry_loss    : raw training loss
    """
    s = str(schema).lower()
    if ran_evaluate_py and has_d_seg and has_d_pose and has_bytes and "exact_eval" in s:
        return "exact_evaluate"
    if "per_pair" in s or "pair_scorer" in s:
        return "exact_pair_scorer"
    if "psnr" in s or "recon_fit" in s or "capacity" in s:
        return "psnr_capacity"
    if "scorer" in s or "proxy" in s:
        return "scorer_proxy"
    return "telemetry_loss"


def _read_frontier(pointer_path: Path, axis: str) -> tuple[float, str, str]:
    """Read the canonical frontier (score, grade, archive_sha) for an axis.

    NEVER hardcoded -- the pointer file is the source of truth. The archive sha
    is the CURRENT base sha the candidate is scored against (STALE detection).
    """
    data = json.loads(pointer_path.read_text())
    key = f"our_local_frontier_{axis}"  # e.g. our_local_frontier_contest_cpu
    node = data.get(key)
    if not isinstance(node, dict) or "score" not in node:
        raise KeyError(
            f"frontier pointer {pointer_path} missing {key}.score "
            f"(available: {sorted(data)})"
        )
    return (
        float(node["score"]),
        str(node.get("evidence_grade", f"[{axis}]")),
        str(node.get("archive_sha256", "")),
    )


def _extract_distortions(ev: dict[str, Any]) -> tuple[float, float, int]:
    """Pull (d_seg, d_pose, archive_bytes) robustly from an exact-eval JSON."""
    b2r = ev.get("b2", {}).get("b2_result", {}) if isinstance(ev.get("b2"), dict) else {}

    def _first(*candidates: Any) -> Any:
        for c in candidates:
            if c is not None:
                return c
        return None

    d_seg = _first(
        ev.get("avg_segnet_dist"),
        b2r.get("avg_segnet_dist"),
        (ev.get("score_seg_contribution") / 100.0)
        if ev.get("score_seg_contribution") is not None
        else None,
    )
    d_pose = _first(ev.get("avg_posenet_dist"), b2r.get("avg_posenet_dist"))
    archive_bytes = _first(
        ev.get("export", {}).get("archive_bytes") if isinstance(ev.get("export"), dict) else None,
        b2r.get("archive_size_bytes"),
        ev.get("archive_bytes"),
    )
    if d_seg is None or d_pose is None or archive_bytes is None:
        raise ValueError(
            "exact-eval JSON missing one of avg_segnet_dist / avg_posenet_dist / "
            f"archive_bytes (got d_seg={d_seg} d_pose={d_pose} bytes={archive_bytes})"
        )
    return float(d_seg), float(d_pose), int(archive_bytes)


def _bridge_ok(ev: dict[str, Any]) -> bool:
    """Did the export->inflate->evaluate bridge succeed (vs a mechanical failure)?"""
    if ev.get("pipeline_works") is True:
        return True
    b2 = ev.get("b2", {})
    if isinstance(b2, dict) and b2.get("b2_returncode") == 0:
        return True
    return False


def _detect_proxy_divergence(traj_dir: Path | None) -> dict[str, Any]:
    """Scan checkpoint *.meta.json selection metrics for a divergence signature.

    Diverged := final/last-epoch selection metric is materially WORSE (>1.5x) than
    the best (min). A *descending* run keeps its best near the end; a *diverging*
    run captures 'best' early then climbs.
    """
    out: dict[str, Any] = {
        "scanned": False,
        "diverged": False,
        "best_value": None,
        "final_value": None,
        "best_epoch": None,
        "final_epoch": None,
    }
    if traj_dir is None or not traj_dir.is_dir():
        return out
    rows: list[tuple[int, float, bool]] = []
    for meta in sorted(traj_dir.glob("*.meta.json")):
        try:
            m = json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        val = m.get("checkpoint_selection_metric_value", m.get("loss"))
        ep = m.get("global_epoch")
        if val is None or ep is None:
            continue
        rows.append((int(ep), float(val), bool(m.get("is_final", False))))
    if not rows:
        return out
    out["scanned"] = True
    best_ep, best_val, _ = min(rows, key=lambda r: r[1])
    finals = [r for r in rows if r[2]]
    final_ep, final_val, _ = finals[0] if finals else max(rows, key=lambda r: r[0])
    out.update(
        best_value=best_val,
        best_epoch=best_ep,
        final_value=final_val,
        final_epoch=final_ep,
        diverged=bool(final_val > best_val * 1.5),
    )
    return out


def ingest_exact_eval(
    *,
    exact_eval_json: Path,
    tag: str,
    output_dir: Path,
    frontier_pointer: Path,
    frontier_axis: str = "contest_cpu",
    vehicle: str = VEHICLE_HI_NERV,
    candidate_kind: str = "checkpoint",
    base_archive_sha256: str | None = None,
    baseline_d_seg: float | None = None,
    baseline_d_pose: float | None = None,
    checkpoint_trajectory_dir: Path | None = None,
) -> dict[str, Any]:
    """Normalize one exact-eval artifact into the two typed V3 rows + write them."""
    ev = json.loads(exact_eval_json.read_text())
    d_seg, d_pose, archive_bytes = _extract_distortions(ev)
    axis_tag = str(ev.get("axis_tag") or ev.get("lane_tag") or "[macOS-CPU advisory]")

    export = ev.get("export", {}) if isinstance(ev.get("export"), dict) else {}
    candidate_sha = str(export.get("archive_sha256") or ev.get("archive_sha256") or "unknown")
    ckpt = ev.get("checkpoint", {}) if isinstance(ev.get("checkpoint"), dict) else {}
    checkpoint_label = (
        str(ckpt.get("global_epoch")) if ckpt.get("global_epoch") is not None else tag
    )

    frontier_score, frontier_grade, frontier_archive_sha = _read_frontier(
        frontier_pointer, frontier_axis
    )

    # The frontier IS the base the candidate is scored against; its archive sha is
    # the current base sha. STALE := the caller pinned an expected base that no
    # longer matches the current frontier base (the candidate's ΔS would be against
    # the wrong base). For a fresh checkpoint-vs-frontier ingest with no pin, the
    # base is the frontier by construction -> never stale.
    base_sha = frontier_archive_sha or f"frontier:{frontier_axis}"
    bridge_ok = _bridge_ok(ev)

    receipt = build_harvest_receipt(
        vehicle=vehicle,
        run_dir=str(exact_eval_json.parent),
        checkpoint_label=checkpoint_label,
        checkpoint_sha256=str(ckpt.get("ema_state_path") or ckpt.get("live_state_path") or "unknown"),
        export_status="passed" if export else ("passed" if bridge_ok else "failed"),
        archive_path=str(export.get("archive_path") or ""),
        archive_sha256=candidate_sha,
        archive_bytes=archive_bytes,
        sidecar_exported=bool(export.get("sidecar_exported", False)),
        pay_rent_gate_active=True,
        inflate_status="passed" if bridge_ok else "failed",
        evaluate_status="passed" if bridge_ok else "failed",
        result_json_path=str(exact_eval_json),
    )

    candidate_eval = build_candidate_action_evaluation_row(
        vehicle=vehicle,
        candidate_kind=candidate_kind,
        candidate_archive_sha256=candidate_sha,
        candidate_d_seg=d_seg,
        candidate_d_pose=d_pose,
        candidate_bytes=archive_bytes,
        base_archive_sha256=base_sha,
        base_score=frontier_score,
        checkpoint_label=checkpoint_label,
        axis_tag=axis_tag,
    )
    # STALE flag: if a base sha was pinned and the candidate's recorded base
    # differs, the ΔS is against the wrong base -> mark stale (do not auto-trust).
    candidate_eval["stale_base_mismatch"] = bool(
        base_archive_sha256 is not None
        and base_archive_sha256 != candidate_eval["base_archive_sha256"]
    )
    # Explicit AUTHORITY TIER + METRIC FAMILY (operator anti-metric-laundering discipline):
    # authority_tier = WHERE the number was produced; metric_family = WHAT KIND of number.
    # A capacity/proxy row can never be structurally mistaken for an exact-score row.
    candidate_eval["authority_tier"] = _authority_tier(axis_tag)
    candidate_eval["metric_family"] = _metric_family(
        str(ev.get("schema", "")),
        has_d_seg=True,  # _extract_distortions raised if absent
        has_d_pose=True,
        has_bytes=archive_bytes is not None,
        ran_evaluate_py=bridge_ok,
    )
    candidate_eval["has_d_seg"] = True
    candidate_eval["has_d_pose"] = True
    candidate_eval["has_archive_bytes"] = archive_bytes is not None
    candidate_eval["has_evaluate_py_report"] = bool(bridge_ok)

    decision = apply_campaign_decision(
        receipt=receipt,
        candidate_eval=candidate_eval,
        frontier_score=frontier_score,
    )

    divergence = _detect_proxy_divergence(checkpoint_trajectory_dir)
    next_action = route_substrate_next_action(
        d_seg=d_seg,
        d_pose=d_pose,
        baseline_d_seg=baseline_d_seg,
        baseline_d_pose=baseline_d_pose,
        proxy_total_diverged=bool(divergence.get("diverged")),
        eval_bridge_ok=bridge_ok,
    )
    decision["next_action"] = next_action
    _authority = _authority_tier(axis_tag)
    _mf = candidate_eval["metric_family"]
    _has_full = bool(archive_bytes is not None and bridge_ok)  # d_seg/d_pose always present here
    decision["authority_tier"] = _authority
    decision["metric_family"] = _mf
    # SCORE-roadmap update requires ALL of: contest axis + exact_evaluate metric + full fields.
    # This is the firewall: a 21.74 dB psnr_capacity row (exact_cpu_advisory) is structurally
    # barred from changing the score roadmap even though it is a real measurement.
    decision["score_roadmap_update_eligible"] = bool(
        _authority in ("contest_cpu", "contest_cuda")
        and _mf == "exact_evaluate"
        and _has_full
    )
    # MECHANISM update (which experiment to run next) is allowed from any real measurement
    # (advisory exact eval, per-pair scorer, capacity PSNR, scorer proxy) -- never promotion.
    decision["mechanism_update_eligible"] = bool(
        _mf in ("exact_evaluate", "exact_pair_scorer", "psnr_capacity", "scorer_proxy")
    )
    # PROMOTION requires the PAIRED dual-axis (contest_cpu AND contest_cuda on the SAME archive
    # sha256); a single-axis ingest row can never establish it.
    decision["promotion_update_eligible"] = False
    decision["promotion_requires"] = (
        "paired contest_cpu AND contest_cuda exact_evaluate on the same archive sha256"
    )
    # Back-compat alias (older consumers read roadmap_update_eligible).
    decision["roadmap_update_eligible"] = decision["score_roadmap_update_eligible"]
    decision["proxy_divergence"] = divergence
    decision["frontier"] = {
        "axis": frontier_axis,
        "score": frontier_score,
        "evidence_grade": frontier_grade,
        "pointer": str(frontier_pointer),
    }
    decision["advisory_axis_caveat"] = (
        f"candidate axis={axis_tag}; frontier axis=[{frontier_axis}]; ΔS compares a "
        f"local-advisory score to a contest frontier (NOT a promotion-eligible claim)."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cae_path = output_dir / f"candidate_action_evaluation_{tag}.v1.json"
    dec_path = output_dir / f"campaign_decision_{tag}.v1.json"
    cae_path.write_text(json.dumps(candidate_eval, indent=2, sort_keys=True) + "\n")
    dec_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    return {
        "candidate_action_evaluation": candidate_eval,
        "campaign_decision": decision,
        "candidate_action_evaluation_path": str(cae_path),
        "campaign_decision_path": str(dec_path),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exact-eval-json", required=True, type=Path)
    p.add_argument("--tag", required=True, help="e.g. r3_ep250")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--frontier-pointer", type=Path, default=Path(_DEFAULT_FRONTIER_POINTER))
    p.add_argument("--frontier-axis", default="contest_cpu", choices=["contest_cpu", "contest_cuda"])
    p.add_argument("--vehicle", default=VEHICLE_HI_NERV)
    p.add_argument("--candidate-kind", default="checkpoint")
    p.add_argument("--base-archive-sha256", default=None)
    p.add_argument("--baseline-d-seg", type=float, default=None)
    p.add_argument("--baseline-d-pose", type=float, default=None)
    p.add_argument("--checkpoint-trajectory-dir", type=Path, default=None)
    args = p.parse_args(argv)

    result = ingest_exact_eval(
        exact_eval_json=args.exact_eval_json,
        tag=args.tag,
        output_dir=args.output_dir,
        frontier_pointer=args.frontier_pointer,
        frontier_axis=args.frontier_axis,
        vehicle=args.vehicle,
        candidate_kind=args.candidate_kind,
        base_archive_sha256=args.base_archive_sha256,
        baseline_d_seg=args.baseline_d_seg,
        baseline_d_pose=args.baseline_d_pose,
        checkpoint_trajectory_dir=args.checkpoint_trajectory_dir,
    )
    cae = result["candidate_action_evaluation"]
    dec = result["campaign_decision"]
    na = dec["next_action"]
    div = dec["proxy_divergence"]
    print("=== V3 ingest: exact eval -> candidate action ===")
    print(f"  d_seg={result['d_seg']:.6f}  d_pose={result['d_pose']:.6f}  bytes={result['archive_bytes']:,}")
    print(f"  candidate_score={cae['candidate_score']:.5f}  frontier={dec['frontier']['score']:.5f} [{dec['frontier']['axis']}]")
    print(f"  ΔS_vs_frontier={cae['delta_score_total']:+.5f}  pays_rent={cae['pays_rent']}  verdict={cae['verdict']}")
    print(f"  UNIVERSAL decision={dec['decision']}  reason={dec['reason']}")
    if "binding_constraint" in dec:
        bc = dec["binding_constraint"]
        print(f"    binding={bc['binding_constraint']}  shares: seg={bc['shares']['seg']:.2f} pose={bc['shares']['pose']:.2f} rate={bc['shares']['rate']:.2f}")
    if div.get("scanned"):
        print(f"  proxy trajectory: best={div['best_value']:.3f}@ep{div['best_epoch']} final={div['final_value']:.3f}@ep{div['final_epoch']} diverged={div['diverged']}")
    print(f"  NEXT-ACTION case={na['case']}  route={na['route']}")
    print(f"    seg_state={na['seg_state']}  pose_state={na['pose_state']}  reason={na['reason']}")
    print(f"  wrote: {result['candidate_action_evaluation_path']}")
    print(f"  wrote: {result['campaign_decision_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
