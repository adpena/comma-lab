#!/usr/bin/env python3
"""Aggregate Phase A ablation anchors into a single operator-readable summary.

Phase A council prescription (from `.omx/research/grand_council_extreme_rigor_track_1_20260508.md`)
maps to 7 ablation lanes:

    A0  — MDL/Bayesian baseline
    A1  — score-gradient supervision PR101 fine-tune
    A2  — sensitivity-aware quantisation (Xavier-L2; FALSIFIED -3,635 B vs uniform)
    A3-alt — Mallat wavelet importance (incremental_improvement_insufficient)
    A4  — ChARM 2020 hyperprior (toy + hand-parametric PR101 configs retired;
           learned co-design remains live)
    A4-alt — Filler STC pose codec (LANDED at 75c99b84; -400 B vs PD-V2 smooth-walk)
    A5  — frame-conditional bit budget (LANDED; best η=2.0 saves -1,278 B)
    A6  — Selfcomp block-FP × hyperprior compose (LANDED at 97fbfef2;
            BEATS both standalones, does NOT beat brotli; +35,891 B)

Each lane writes ``build_manifest.json`` under ``experiments/results/<lane>_<timestamp>/``
with byte / rel_err / score-axis / evidence-grade / dispatch-blocker fields per the
canonical schema. This tool walks every Phase A manifest, tags each by lane, and
emits a Pareto-front table:

    archive_bytes vs rel_err vs score-axis-marginal, by lane × evidence_grade.

Usage:
    .venv/bin/python tools/phase_a_pareto_summary.py
    .venv/bin/python tools/phase_a_pareto_summary.py --output reports/phase_a_pareto_20260508.md
    .venv/bin/python tools/phase_a_pareto_summary.py --json experiments/results/phase_a_pareto.json

Output is deterministic + sorted: alphabetic lane id, then archive_bytes asc, then
rel_err asc. Run is read-only (no commits, no GPU spend).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_geometry import (  # noqa: E402
    PlannerAxisMarginals,
    TargetByteBudget,
    planner_axis_marginals,
    target_byte_budget_for_score,
)

# Canonical lane name → human label mapping.
LANE_PATTERNS: dict[str, str] = {
    "track1_phase_a0_": "A0_mdl_baseline",
    "track1_phase_a1_": "A1_score_gradient",
    "track1_phase_a2_packet_ladder": "A2_packet_ladder",
    "track1_phase_a2_sensitivity_quant": "A2_xavier_l2_sensitivity",
    "pr101_sensitivity_aware_quant": "A2_xavier_l2_sensitivity",
    "pr101_sensitivity_aware_mallat_wavelet": "A3_alt_mallat_wavelet",
    "pr101_pose_filler_stc": "A4_alt_filler_stc_pose",
    "pr101_frame_conditional_bit": "A5_frame_conditional_bits",
    "charm_50k_toy_substrate": "A4_charm_hyperprior_toy",
    "pr101_charm_real_substrate_probe": "A4_charm_real_pr101_probe",
    "pr101_a6_blockfp_hyperprior": "A6_selfcomp_blockfp_hyperprior",
    "cross_paradigm_admm_x_op1_finalizer": "Cross_paradigm_ADMM_x_Op1",
    "admm_x_lossy_coarsening": "ADMM_lossy_coarsening_baseline",
}

# PR101 brotli baseline (the byte-anchor most lanes compare against).
PR101_BROTLI_BYTES = 178_144

# Planning target carried from the 2026-05-08 sub-0.17 solver memo. These are
# prediction-only floor assumptions, not score evidence.
SUBTARGET_SCORE = 0.170
SUBTARGET_CPU_D_SEG_FLOOR = 6.0e-4
SUBTARGET_CPU_D_POSE_FLOOR = 3.5e-5

# PR107/apogee-ish operating point used only to expose CPU/CUDA planner-axis
# separation in this summary. The helper reports prediction-only marginals.
AXIS_ADVISOR_CUDA_D_SEG = 6.88e-4
AXIS_ADVISOR_CUDA_D_POSE = 1.74e-4
AXIS_ADVISOR_ARCHIVE_BYTES = 178_392

NONCOMPARABLE_BYTE_BUDGET_LANES = {
    "A0_mdl_baseline",
    "A1_score_gradient",
    "A4_charm_hyperprior_toy",
    "A4_charm_real_pr101_probe",
    "A4_alt_filler_stc_pose",
}


@dataclass
class PhaseAEntry:
    lane: str
    manifest_path: Path
    archive_bytes: int | None
    rel_err: float | None
    rel_err_form: str | None
    evidence_grade: str | None
    score_claim: bool
    ready_for_exact_eval_dispatch: bool
    dispatch_blockers: tuple[str, ...]
    dispatch_status: str | None
    timestamp: str | None
    delta_vs_uniform_bytes: int | None = None
    delta_vs_brotli_bytes: int | None = None
    byte_budget_comparable: bool = False
    subtarget_gap_bytes: int | None = None
    meets_subtarget_byte_budget: bool | None = None
    notes: list[str] = field(default_factory=list)


def classify_lane(rel_path: str) -> str:
    """Return canonical lane id from a path under experiments/results/."""
    for pattern, lane in LANE_PATTERNS.items():
        if pattern in rel_path:
            return lane
    return "UNCLASSIFIED"


def parse_manifest(manifest_path: Path) -> PhaseAEntry | None:
    """Load + extract canonical fields from a build_manifest.json."""
    rel = str(manifest_path.relative_to(REPO_ROOT))
    lane = classify_lane(rel)
    if lane == "UNCLASSIFIED":
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Different lanes use different field names; canonicalise.
    archive_bytes = (
        data.get("archive_bytes")
        or data.get("empirical_archive_bytes")
        or data.get("archive_bytes_total")
        or data.get("estimated_archive_bytes")
        or data.get("archive_bytes_new")
        or data.get("fstc_blob_bytes")
        or data.get("best_archive_bytes")
    )
    best_model = data.get("best_model")
    if archive_bytes is None and isinstance(best_model, dict):
        archive_bytes = best_model.get("archive_estimate_bytes")
    if archive_bytes is None:
        # Try sweep entries under "results" or "rows" keys.
        for key in ("results", "rows", "sweep", "entries"):
            seq = data.get(key)
            if isinstance(seq, list) and seq:
                # Take the smallest archive_bytes from the sweep.
                bytes_list = [
                    e.get("archive_bytes")
                    or e.get("empirical_archive_bytes")
                    or e.get("archive_bytes_new")
                    or e.get("fstc_blob_bytes")
                    for e in seq
                    if isinstance(e, dict)
                ]
                bytes_list = [b for b in bytes_list if isinstance(b, int)]
                if bytes_list:
                    archive_bytes = min(bytes_list)
                    break

    rel_err = (
        data.get("rel_err")
        or data.get("rms_rel_err")
        or data.get("avg_rel_err")
        or data.get("max_rel_err")
    )
    rel_err_form = data.get("rel_err_form") or data.get("rel_err_mode") or "unknown"

    evidence_grade = data.get("evidence_grade") or data.get("evidence_tag")
    score_claim = bool(data.get("score_claim", False))
    ready = bool(data.get("ready_for_exact_eval_dispatch", False))
    dispatch_blockers = tuple(data.get("dispatch_blockers", []) or [])
    dispatch_status = data.get("dispatch_status") or data.get("status")

    timestamp = (
        data.get("started_at_utc")
        or data.get("build_timestamp_utc")
        or data.get("created_at_utc")
        or manifest_path.parent.name.rsplit("_", 1)[-1]
    )

    entry = PhaseAEntry(
        lane=lane,
        manifest_path=manifest_path,
        archive_bytes=archive_bytes,
        rel_err=rel_err,
        rel_err_form=rel_err_form,
        evidence_grade=evidence_grade,
        score_claim=score_claim,
        ready_for_exact_eval_dispatch=ready,
        dispatch_blockers=dispatch_blockers,
        dispatch_status=dispatch_status,
        timestamp=timestamp,
    )

    # Compute deltas if archive_bytes is known.
    if archive_bytes is not None and lane != "A4_alt_filler_stc_pose":
        entry.delta_vs_brotli_bytes = archive_bytes - PR101_BROTLI_BYTES

    # Lane-specific notes.
    if lane == "A2_xavier_l2_sensitivity":
        entry.notes.append("FALSIFIED proxy (-3,635 B regression vs uniform)")
    if lane == "A3_alt_mallat_wavelet":
        entry.notes.append("incremental_improvement_insufficient (Mallat > Xavier in 2/4)")
    if lane == "A4_charm_hyperprior_toy":
        entry.notes.append("toy config measured negative; learned co-design only")
    if lane == "A4_charm_real_pr101_probe":
        best = data.get("best_model", {})
        delta = best.get("delta_vs_brotli_optuna_bytes") if isinstance(best, dict) else None
        if isinstance(delta, int):
            entry.notes.append(
                f"hand-parametric ChARM roundtrips exactly but loses to brotli ({delta:+,} B)"
            )
        else:
            entry.notes.append("hand-parametric ChARM probe; no score claim")
    if lane == "A1_score_gradient":
        entry.notes.append(
            "first Modal config retired on macOS CPU advisory; reactivation needs constrained fine-tune"
        )
    if lane == "ADMM_lossy_coarsening_baseline":
        entry.notes.append("Path B baseline; -28 KB savings at 4-5% rel_err")
    if lane == "A4_alt_filler_stc_pose":
        entry.notes.append("pose-codec byte anchor; not comparable to full PR101 archive bytes")
    if lane == "A5_frame_conditional_bits":
        best_delta = data.get("best_archive_delta_bytes")
        if isinstance(best_delta, int):
            entry.notes.append(f"best frame-conditioned latent delta {best_delta:+,} B")
        entry.notes.append(
            "eta=4 complexity runtime packet collapsed on macOS CPU advisory; use score-domain allocation next"
        )
    if lane == "Cross_paradigm_ADMM_x_Op1":
        entry.notes.append(
            "advisory score 0.328444: rate improves but SegNet collapse dominates"
        )
    if lane == "A6_selfcomp_blockfp_hyperprior":
        delta = data.get("delta_pr101_brotli_baseline")
        if isinstance(delta, int):
            entry.notes.append(
                f"measured proxy loses to PR101 brotli ({delta:+,} B); family not killed"
            )

    return entry


def find_phase_a_manifests() -> list[Path]:
    """Walk experiments/results/ and return every build_manifest.json that
    classifies as a Phase A lane.
    """
    out: list[Path] = []
    results_root = REPO_ROOT / "experiments" / "results"
    if not results_root.is_dir():
        return out
    for manifest in results_root.rglob("build_manifest.json"):
        rel = str(manifest.relative_to(REPO_ROOT))
        if classify_lane(rel) != "UNCLASSIFIED":
            out.append(manifest)
    real_charm_report = REPO_ROOT / "reports" / "pr101_charm_real_substrate_probe.json"
    if real_charm_report.is_file():
        out.append(real_charm_report)
    return sorted(out)


def deduplicate_keep_best(entries: list[PhaseAEntry]) -> list[PhaseAEntry]:
    """For each lane, keep the entry with smallest archive_bytes (or first if no
    bytes). This collapses sweep-explosion to a single Pareto point per lane.
    """
    by_lane: dict[str, PhaseAEntry] = {}
    for e in entries:
        cur = by_lane.get(e.lane)
        if cur is None:
            by_lane[e.lane] = e
            continue
        # Prefer entries with archive_bytes set, then smaller archive_bytes.
        if e.archive_bytes is None:
            continue
        if cur.archive_bytes is None or e.archive_bytes < cur.archive_bytes:
            by_lane[e.lane] = e
    return sorted(by_lane.values(), key=lambda x: x.lane)


def build_subtarget_budget() -> TargetByteBudget:
    """Closed-form byte budget for the Phase A sub-0.17 planning target."""
    return target_byte_budget_for_score(
        target_score=SUBTARGET_SCORE,
        d_seg_floor=SUBTARGET_CPU_D_SEG_FLOOR,
        d_pose_floor=SUBTARGET_CPU_D_POSE_FLOOR,
        current_archive_bytes=PR101_BROTLI_BYTES,
    )


def build_axis_advisors() -> tuple[PlannerAxisMarginals, PlannerAxisMarginals]:
    """Return CUDA-internal and CPU-leaderboard planner-axis marginals."""
    kwargs = {
        "cuda_d_seg": AXIS_ADVISOR_CUDA_D_SEG,
        "cuda_d_pose": AXIS_ADVISOR_CUDA_D_POSE,
        "archive_bytes": AXIS_ADVISOR_ARCHIVE_BYTES,
        "archive_class": "hnerv",
    }
    return (
        planner_axis_marginals(target_axis="cuda_internal", **kwargs),
        planner_axis_marginals(target_axis="cpu_leaderboard", **kwargs),
    )


def annotate_planning_targets(
    entries: list[PhaseAEntry],
    *,
    budget: TargetByteBudget | None = None,
) -> list[PhaseAEntry]:
    """Annotate byte-comparable rows with the subtarget byte-budget gap."""
    target_budget = budget or build_subtarget_budget()
    max_bytes = target_budget.max_archive_bytes
    for e in entries:
        comparable = (
            e.archive_bytes is not None
            and e.lane not in NONCOMPARABLE_BYTE_BUDGET_LANES
            and max_bytes is not None
        )
        e.byte_budget_comparable = bool(comparable)
        if comparable:
            assert e.archive_bytes is not None
            assert max_bytes is not None
            e.subtarget_gap_bytes = e.archive_bytes - max_bytes
            e.meets_subtarget_byte_budget = e.archive_bytes <= max_bytes
        else:
            e.subtarget_gap_bytes = None
            e.meets_subtarget_byte_budget = None
    return entries


def _format_budget_gap(e: PhaseAEntry) -> str:
    if e.subtarget_gap_bytes is None:
        return "n/a"
    return f"{e.subtarget_gap_bytes:+,}"


def render_markdown(entries: list[PhaseAEntry]) -> str:
    """Emit the operator-facing Phase A summary in Markdown."""
    budget = build_subtarget_budget()
    cuda_axis, cpu_axis = build_axis_advisors()
    annotate_planning_targets(entries, budget=budget)
    feasibility = "true" if budget.feasible_under_floors else "false"
    lines = [
        "# Phase A Cross-Ablation Pareto Summary",
        "",
        f"Generated by `tools/phase_a_pareto_summary.py` from {len(entries)} unique-lane manifests.",
        "",
        f"PR101 brotli baseline: **{PR101_BROTLI_BYTES:,} B** (the byte-anchor most lanes target).",
        "",
        "## Solver planning targets",
        "",
        f"- Subtarget: **{SUBTARGET_SCORE:.3f}** with CPU-floor assumptions "
        f"`d_seg={SUBTARGET_CPU_D_SEG_FLOOR:.1e}`, "
        f"`d_pose={SUBTARGET_CPU_D_POSE_FLOOR:.1e}`. "
        f"Max archive bytes: **{budget.max_archive_bytes:,} B**; "
        f"required savings vs PR101 brotli: **{budget.required_savings_bytes:,} B**; "
        f"floor-feasible: `{feasibility}`. "
        f"Evidence grade: `{budget.evidence_grade}`.",
        f"- Axis advisor at `d_seg_cuda={AXIS_ADVISOR_CUDA_D_SEG:.2e}`, "
        f"`d_pose_cuda={AXIS_ADVISOR_CUDA_D_POSE:.2e}`, "
        f"`B={AXIS_ADVISOR_ARCHIVE_BYTES:,}`: "
        f"`target_axis={cuda_axis.target_axis}` priority `{cuda_axis.priority_axis}` "
        f"(seg={cuda_axis.seg_marginal:.2f}, pose={cuda_axis.pose_marginal:.2f}); "
        f"`target_axis={cpu_axis.target_axis}` priority `{cpu_axis.priority_axis}` "
        f"(seg={cpu_axis.seg_marginal:.2f}, pose={cpu_axis.pose_marginal:.2f}). "
        f"Evidence grade: `{cpu_axis.evidence_grade}`.",
        "",
        "| Lane | Archive bytes | Δ vs brotli | Sub-0.17 byte gap | rel_err | Evidence grade | Dispatch ready | Notes |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for e in entries:
        bytes_cell = f"{e.archive_bytes:,}" if e.archive_bytes is not None else "—"
        delta_cell = (
            f"{e.delta_vs_brotli_bytes:+,}"
            if e.delta_vs_brotli_bytes is not None
            else "—"
        )
        gap_cell = _format_budget_gap(e)
        rel_err_cell = (
            f"{e.rel_err:.4f} ({e.rel_err_form})"
            if isinstance(e.rel_err, (int, float))
            else "—"
        )
        ev = (e.evidence_grade or "—").replace("|", "/").strip()
        ready_cell = "✓" if e.ready_for_exact_eval_dispatch else "—"
        notes_cell = "; ".join(e.notes) if e.notes else "—"
        lines.append(
            f"| {e.lane} | {bytes_cell} | {delta_cell} | {gap_cell} | {rel_err_cell} | "
            f"{ev} | {ready_cell} | {notes_cell} |"
        )
    lines.extend([
        "",
        "## Class-level findings",
        "",
        "- **Decision 3 weight-domain proxies are exhausted.** A2 Xavier-L2 (-3,635 B regression "
        "vs uniform) + A3-alt Mallat wavelet (incremental over Xavier, still loses to uniform "
        "at high budgets) both fail. Future Decision 3 reactivation must use score-domain "
        "(Hessian-trace, score-gradient) or byte-domain (compression-hardness) proxies.",
        "- **PR101 substrate is near-iid at brotli's compressor.** Multiple lanes show ~1-3% "
        "above the iid Shannon floor; magnitude-based concentration of bits cannot break that.",
        "- **A4 hand-parametric ChARM on real PR101 loses by +28,601 B.** The "
        "real-substrate range-coder probe roundtrips exactly, but static/delta/"
        "previous-symbol PMFs are not competitive with PR101 brotli. Learned "
        "co-designed ChARM remains live.",
        "- **A1 first Modal config is a measured-config negative.** Training/build completed, "
        "but exact CUDA was skipped by DALI/NVDEC preflight and local macOS CPU advisory "
        "scored 3.721654. Reactivation needs constrained fine-tune, not exact eval of this archive.",
        "- **A5 and Cross-paradigm byte savings are scorer-unsafe at current configs.** "
        "A5 eta=4 complexity allocation and Cross-paradigm ADMM x Op1 both need score-domain "
        "or SegNet-boundary-aware allocation before new exact-eval spend.",
        "",
        "## Open lanes",
        "",
        "- A4-alt (Filler STC pose codec): byte-anchor landed; representative "
        "pose-distribution only, not a PR101 monolithic archive rewrite.",
        "- A4 (ChARM/hyperprior): toy and hand-parametric PR101 configs are "
        "measured-config negatives. Next variant must be learned/co-designed "
        "with the substrate or produce a runtime-consumed packet before exact "
        "eval spend.",
        "- A5 (frame-conditional bit budget): runtime side-info path landed, but "
        "the eta=4 complexity schedule is retired after advisory collapse. Next "
        "variant needs score-domain q-bit allocation.",
        "- Cross-paradigm ADMM x Op1: measured config retired after macOS CPU "
        "advisory SegNet collapse; next variant needs lower-distortion trust region "
        "or scorer-aware allocation.",
        "- A6 (Selfcomp block-FP × hyperprior compose): LANDED at `97fbfef2`. "
        "Best compose B=64, sq=uint8 = 214,035 B; BEATS blockfp-only "
        "(-34,607 B) AND hyperprior-only (-18,356 B); does NOT beat PR101 "
        "brotli baseline (+35,891 B). Verdict "
        "`incremental_improvement_insufficient` (NOT killed per CLAUDE.md "
        "kill-as-last-resort); reactivation criteria are documented in "
        "`.omx/research/pr101_a6_selfcomp_blockfp_hyperprior_measured_negative_20260508_codex.md` "
        "with PR106 substrate ranked first, then learned hyper-decoder MLP, "
        "cross-tensor grouping, joint-AC over scale stream, and "
        "compose-after-lossy_coarsening.",
        "",
    ])
    return "\n".join(lines) + "\n"


def render_json(entries: list[PhaseAEntry]) -> str:
    """Emit machine-readable Pareto summary."""
    budget = build_subtarget_budget()
    cuda_axis, cpu_axis = build_axis_advisors()
    annotate_planning_targets(entries, budget=budget)
    out = {
        "schema_version": "phase_a_pareto_v1",
        "pr101_brotli_baseline_bytes": PR101_BROTLI_BYTES,
        "planning_targets": {
            "subtarget_byte_budget": asdict(budget),
            "axis_advisors": {
                "cuda_internal": asdict(cuda_axis),
                "cpu_leaderboard": asdict(cpu_axis),
            },
        },
        "lane_count": len(entries),
        "lanes": [],
    }
    for e in entries:
        out["lanes"].append({
            "lane": e.lane,
            "archive_bytes": e.archive_bytes,
            "rel_err": e.rel_err,
            "rel_err_form": e.rel_err_form,
            "delta_vs_brotli_bytes": e.delta_vs_brotli_bytes,
            "evidence_grade": e.evidence_grade,
            "score_claim": e.score_claim,
            "ready_for_exact_eval_dispatch": e.ready_for_exact_eval_dispatch,
            "dispatch_blockers": list(e.dispatch_blockers),
            "dispatch_status": e.dispatch_status,
            "byte_budget_comparable": e.byte_budget_comparable,
            "subtarget_gap_bytes": e.subtarget_gap_bytes,
            "meets_subtarget_byte_budget": e.meets_subtarget_byte_budget,
            "timestamp": e.timestamp,
            "manifest_path": str(e.manifest_path.relative_to(REPO_ROOT)),
            "notes": e.notes,
        })
    return json.dumps(out, indent=2, sort_keys=False) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Markdown summary path (default: stdout).",
    )
    p.add_argument(
        "--json", type=Path, default=None,
        help="Optional JSON output path for machine consumption.",
    )
    args = p.parse_args()

    manifests = find_phase_a_manifests()
    raw_entries = []
    for m in manifests:
        e = parse_manifest(m)
        if e is not None:
            raw_entries.append(e)
    entries = annotate_planning_targets(deduplicate_keep_best(raw_entries))

    md = render_markdown(entries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"[ok] wrote {args.output} ({len(entries)} unique lanes from "
              f"{len(manifests)} manifests)", file=sys.stderr)
    else:
        sys.stdout.write(md)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(render_json(entries), encoding="utf-8")
        print(f"[ok] wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
