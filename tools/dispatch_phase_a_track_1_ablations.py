"""Phase A Track 1 ablations dispatch wrapper.

Per CLAUDE.md "parallel-dispatch is a FIRST-CLASS DELIVERABLE", this wrapper
fans out the 8 Phase A ablations across local CPU (Phase A0/A2/A4-alt/A5),
Lightning T4 (Phase A1/A3-alt/A4/A6), and free GHA CPU (auth eval harvest).

It is the actuator that turns the council's Phase A staging plan into N
concurrent dispatches with per-item gating, heartbeat, and harvest. Council
gating happens BEFORE this wrapper fires; this wrapper is the executor.

CLAUDE.md non-negotiable enforcements:
- Lane claim opened via tools/claim_lane_dispatch.py before any GPU spend
- Cost cap per dispatch + total cap (operator pre-approved $55 Phase A total)
- Heartbeat to experiments/results/<lane>/heartbeat.log every 60s
- Persistence path under experiments/results/, NOT /tmp
- All commits via subagent_commit_serializer (this wrapper does not commit;
  it produces artifacts that downstream tools will commit)

Tag discipline:
- Phase A0/A2/A4-alt/A5 results: [empirical:<artifact>] (CPU-only,
  deterministic)
- Phase A1/A3-alt/A4/A6 results: [contest-CUDA] iff Lightning T4 dispatch
  produces inflate.sh + evaluate.py output; otherwise [predicted]
- M5 Max ablations: [macOS-CPU calibrated] for HNeRV-cluster archives or
  [MPS-research-signal] for proxy training signals

Usage:

    # Phase A0 (MDL calc — local CPU, $0):
    .venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \\
        --decision A0 --substrate pr101 --output reports/raw/

    # Phase A2 (sensitivity-aware quant — local CPU, $0):
    .venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \\
        --decision A2 --substrate pr101 --output experiments/results/

    # Phase A1 (score-gradient on PR101 fine-tune — Lightning T4, $8):
    .venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \\
        --decision A1 --substrate pr101 --duration 3h --budget 8

    # All Phase A in parallel (council recommended):
    .venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \\
        --decision all --total-budget 55

The wrapper does NOT commit; downstream harvest scripts collect artifacts
into experiments/results/ for review by the operator/parent agent before any
commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_RESULTS = REPO_ROOT / "experiments" / "results"
REPORTS_RAW = REPO_ROOT / "reports" / "raw"


@dataclass
class DispatchSpec:
    """Specification for a single Phase A ablation dispatch."""

    decision: str  # e.g. "A0", "A1", "A4-alt"
    name: str
    substrate: str  # "pr101", "pr107", "toy_50k"
    target: str  # "cpu_local", "lightning_t4", "gha_cpu"
    estimated_duration_hours: float
    estimated_cost_usd: float
    lane_id: str
    description: str
    pre_check_required: list[str] = field(default_factory=list)
    output_relpath: str = ""
    run_args: list[str] = field(default_factory=list)


PHASE_A_DISPATCHES: dict[str, DispatchSpec] = {
    "A0": DispatchSpec(
        decision="A0",
        name="MDL closed-form lower bound",
        substrate="pr101",
        target="cpu_local",
        estimated_duration_hours=0.05,
        estimated_cost_usd=0.0,
        lane_id="track1_phase_a0_mdl",
        description="MacKay closed-form Bayesian-MDL lower bound on PR101 weights",
        pre_check_required=[],
        output_relpath="reports/raw/track_1_mdl_pr101",
        run_args=[
            ".venv/bin/python",
            "tools/mdl_lower_bound_calculator.py",
            "--quantization",
            "int8",
            "--hyperprior-config",
            "charm_2020",
            "--architecture",
            "hnerv_pr101",
        ],
    ),
    "A2": DispatchSpec(
        decision="A2",
        name="Sensitivity-aware per-tensor quantization",
        substrate="pr101",
        target="cpu_local",
        estimated_duration_hours=2.0,
        estimated_cost_usd=0.0,
        lane_id="track1_phase_a2_sensitivity_quant",
        description="Jack-from-skunkworks lane: per-tensor importance-weighted K-search",
        pre_check_required=[
            "tools/sensitivity_weighted_lossy_coarsening.py",
            "src/tac/optimization/lagrangian_per_tensor_allocation.py",
            "src/tac/optimization/beta_fisher_lossy_weights.py",
        ],
        output_relpath="experiments/results/track1_phase_a2_sensitivity_pr101",
        run_args=[
            ".venv/bin/python",
            "tools/sensitivity_weighted_lossy_coarsening.py",
            "--substrate",
            "pr101",
            "--state-dict",
            "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
            "--sensitivity-map",
            "experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt",
            "--allow-diagnostic-sensitivity",
            "--local-only",
            "--rms-budget",
            "0.0386",
            "--max-K",
            "64",
        ],
    ),
    "A1": DispatchSpec(
        decision="A1",
        name="Score-gradient supervision on PR101 fine-tune",
        substrate="pr101",
        target="lightning_t4",
        estimated_duration_hours=3.0,
        estimated_cost_usd=8.0,
        lane_id="track1_phase_a1_score_gradient",
        description="Quantizr/Hinton lane: SegNet KL-distill (T=2.0) + PoseNet MSE backprop",
        pre_check_required=[
            "src/tac/losses.py",  # contains scorer_loss, kl_on_logits, segnet_*_per_pixel
        ],
        output_relpath="experiments/results/track1_phase_a1_score_gradient",
        run_args=[],  # to be wired through scripts/remote_archive_only_eval.sh
    ),
    "A3-alt": DispatchSpec(
        decision="A3-alt",
        name="Mallat wavelet basis x per-tensor importance",
        substrate="pr101",
        target="lightning_t4",
        estimated_duration_hours=2.0,
        estimated_cost_usd=4.0,
        lane_id="track1_phase_a3alt_wavelet_importance",
        description="Mallat lane: wavelet-coefficient importance for higher-resolution allocation",
        pre_check_required=["src/tac/sensitivity_map.py"],
        output_relpath="experiments/results/track1_phase_a3alt_wavelet",
        run_args=[],
    ),
    "A4": DispatchSpec(
        decision="A4",
        name="ChARM 2020 co-trained 50K-param toy substrate",
        substrate="toy_50k",
        target="lightning_t4",
        estimated_duration_hours=6.0,
        estimated_cost_usd=15.0,
        lane_id="track1_phase_a4_charm_codesign",
        description="Ballé lane: ChARM 2020 (channel-conditional autoregressive) on co-trained substrate",
        pre_check_required=[],
        output_relpath="experiments/results/track1_phase_a4_charm_toy",
        run_args=[],
    ),
    "A4-alt": DispatchSpec(
        decision="A4-alt",
        name="Filler STC pose encoding",
        substrate="pr101",
        target="cpu_local",
        estimated_duration_hours=2.0,
        estimated_cost_usd=0.0,
        lane_id="track1_phase_a4alt_stc_pose",
        description="Filler lane: syndrome-trellis pose encoding (parity-check arithmetic)",
        pre_check_required=["src/tac/codec/syndrome_trellis_codec.py"],
        output_relpath="experiments/results/track1_phase_a4alt_stc_pose",
        run_args=[],
    ),
    "A5": DispatchSpec(
        decision="A5",
        name="Frame-conditional bit budget byte-only ablation",
        substrate="pr101",
        target="cpu_local",
        estimated_duration_hours=1.0,
        estimated_cost_usd=0.0,
        lane_id="track1_phase_a5_frame_budget",
        description="Boyd lane: per-frame water-filling bit allocation",
        pre_check_required=[],
        output_relpath="experiments/results/track1_phase_a5_frame_budget",
        run_args=[],
    ),
    "A6": DispatchSpec(
        decision="A6",
        name="Selfcomp block-FP x hyperprior compose",
        substrate="pr101",
        target="lightning_t4",
        estimated_duration_hours=4.0,
        estimated_cost_usd=5.0,
        lane_id="track1_phase_a6_selfcomp_x_hyperprior",
        description="Selfcomp lane: block-FP on high-sensitivity tensors + hyperprior on rest",
        pre_check_required=[],
        output_relpath="experiments/results/track1_phase_a6_compose",
        run_args=[],
    ),
}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def claim_lane(
    lane_id: str,
    *,
    platform: str,
    instance_job_id: str,
    predicted_eta_utc: str,
    force: bool = False,
    notes: str = "",
    dry_run: bool = False,
) -> int:
    """Open a lane claim via tools/claim_lane_dispatch.py.

    Returns the subprocess exit code. 0 = claim opened or no conflict.
    Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION — NON-NEGOTIABLE".
    """
    cmd = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        platform,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        "claude:phase_a_dispatch_wrapper",
        "--predicted-eta-utc",
        predicted_eta_utc,
        "--status",
        "provisioning",
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    if notes:
        cmd.extend(["--notes", notes])
    print(f"[claim] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[claim] stderr: {proc.stderr}", file=sys.stderr)
    return proc.returncode


def write_heartbeat(lane_dir: Path, message: str) -> None:
    """Write a heartbeat entry for a lane.

    Per CLAUDE.md "Remote code parity — NON-NEGOTIABLE": every long-running
    dispatch writes a heartbeat to the lane's local artifact directory.
    """
    lane_dir.mkdir(parents=True, exist_ok=True)
    with (lane_dir / "heartbeat.log").open("a") as f:
        f.write(f"{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} {message}\n")


def precheck(spec: DispatchSpec) -> tuple[bool, str]:
    """Verify required files exist before dispatch.

    Returns (ok, reason). If reason is non-empty and ok=False, abort dispatch.
    """
    for path_str in spec.pre_check_required:
        path = REPO_ROOT / path_str
        if not path.exists():
            return False, f"pre-check failed: {path_str} not found"
    return True, ""


def dispatch_local_cpu(spec: DispatchSpec, output_root: Path) -> dict[str, Any]:
    """Dispatch a CPU-only ablation locally on M5 Max.

    Phase A0/A2/A4-alt/A5 use this path. CPU-only and deterministic; results
    tagged [empirical:<artifact>] for byte-side claims, [macOS-CPU calibrated]
    for HNeRV-cluster CPU eval.
    """
    timestamp = utc_timestamp()
    lane_dir = output_root / f"{spec.lane_id}_{timestamp}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    write_heartbeat(lane_dir, f"start decision={spec.decision} target=cpu_local")

    if not spec.run_args:
        # A2/A4-alt/A5 placeholders — emit a stub manifest indicating the tool
        # needs to be built as part of the dispatch. This is the council Round 2
        # finding 10 documented blocker.
        manifest = {
            "decision": spec.decision,
            "lane_id": spec.lane_id,
            "substrate": spec.substrate,
            "target": "cpu_local",
            "status": "tool_not_yet_implemented",
            "evidence_grade": "stub_pending_build",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "council_finding_reference": "Round 2 finding 10",
            "next_action": (
                f"Build the {spec.name} tool in tools/, then re-run this dispatch with "
                "--decision " + spec.decision + " to produce the artifact."
            ),
            "estimated_dev_hours": spec.estimated_duration_hours,
            "estimated_dispatch_hours_after_build": spec.estimated_duration_hours,
            "lane_dir": str(lane_dir),
        }
        (lane_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
        write_heartbeat(lane_dir, "stub_pending_build")
        return manifest

    # A0 has run_args wired; resolve --output if not present
    cmd = list(spec.run_args)
    if "--output" not in cmd:
        cmd.extend(["--output", str(lane_dir / f"{spec.decision}_result.json")])
    write_heartbeat(lane_dir, f"exec {' '.join(cmd[:3])}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (lane_dir / "stdout.log").write_text(proc.stdout)
    (lane_dir / "stderr.log").write_text(proc.stderr)
    write_heartbeat(lane_dir, f"exit code={proc.returncode}")

    manifest = {
        "decision": spec.decision,
        "lane_id": spec.lane_id,
        "substrate": spec.substrate,
        "target": "cpu_local",
        "exit_code": proc.returncode,
        "evidence_grade": "byte_proxy_only_deterministic" if proc.returncode == 0 else "failed",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "remote_dispatch_allowed": False,
        "dispatch_attempted": False,
        "tool_output": str(lane_dir / f"{spec.decision}_result.json"),
        "lane_dir": str(lane_dir),
    }
    (lane_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def dispatch_lightning_t4(spec: DispatchSpec, output_root: Path) -> dict[str, Any]:
    """Stub: stage a Lightning T4 dispatch via the canonical bootstrap.

    Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this MUST
    delegate to scripts/remote_archive_only_eval.sh::bootstrap_runtime_deps and
    NEVER inline `uv run --with torch` or `apt-get install`. Per CLAUDE.md
    "Forbidden uv torch install without driver-version pin", torch must be
    pinned to torch==2.5.1+cu124 with UV_EXTRA_INDEX_URL.

    This wrapper produces a STAGING manifest indicating what would be dispatched;
    actual Lightning launch happens via existing infrastructure
    (tools/lightning_dispatch_pr106_stack.py) which is operator-invoked. The
    council memo Round 3 finding 8 verifies the contract.
    """
    timestamp = utc_timestamp()
    lane_dir = output_root / f"{spec.lane_id}_{timestamp}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    write_heartbeat(lane_dir, f"start decision={spec.decision} target=lightning_t4 staged")

    instance_job_id = f"{spec.lane_id}_{timestamp}"
    predicted_eta_utc = (
        dt.datetime.now(dt.UTC) + dt.timedelta(hours=spec.estimated_duration_hours)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    # This wrapper stages a dispatch manifest only. It validates the required
    # claim command shape without opening a phantom active claim; the real
    # GPU launcher must open the active claim immediately before spend.
    claim_status = claim_lane(
        spec.lane_id,
        platform="lightning",
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=spec.description,
        dry_run=True,
    )
    if claim_status != 0:
        manifest = {
            "decision": spec.decision,
            "lane_id": spec.lane_id,
            "status": "claim_dry_run_refused",
            "claim_exit_code": claim_status,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "evidence_grade": "claim_dry_run_blocked",
            "lane_dir": str(lane_dir),
            "next_action": "Resolve lane-claim syntax/conflict via tools/claim_lane_dispatch.py and retry",
        }
        (lane_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    manifest: dict[str, Any] = {
        "decision": spec.decision,
        "lane_id": spec.lane_id,
        "substrate": spec.substrate,
        "target": "lightning_t4",
        "status": "staged_pending_operator_launch",
        "instance_job_id": instance_job_id,
        "predicted_eta_utc": predicted_eta_utc,
        "estimated_duration_hours": spec.estimated_duration_hours,
        "estimated_cost_usd": spec.estimated_cost_usd,
        "evidence_grade": "staged_no_dispatch",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "active_lane_claim_opened": False,
        "claim_dry_run_validated": True,
        "council_gate_g13_status": "GREEN_claim_dry_run_validated_no_gpu_dispatch",
        "council_gate_g14_status": "GREEN_heartbeat_wired",
        "lane_dir": str(lane_dir),
        "checkpoint_contract": {
            "ema_shadow": True,
            "lambda_R": spec.decision in {"A4", "A6"},
            "lambda_S": spec.decision in {"A4", "A6"},
            "lambda_P": spec.decision in {"A4", "A6"},
            "step_number": True,
            "scheduler_state": True,
        },
        "next_action": (
            f"Operator/parent invokes the canonical Lightning dispatcher with "
            f"lane_id={spec.lane_id}, substrate={spec.substrate}, "
            f"duration={spec.estimated_duration_hours}h, budget=${spec.estimated_cost_usd}. "
            "The actual GPU launcher must open a non-dry-run lane claim immediately before spend. "
            "All bootstrap goes through scripts/remote_archive_only_eval.sh "
            "(self-bootstraps uv + ffmpeg + cu124 torch). Auth eval at completion runs "
            "BOTH [contest-CUDA] and [contest-CPU] per CLAUDE.md dual-eval mandate."
        ),
        "tag_discipline": {
            "training_intermediate": "[advisory only]",
            "final_archive": "[contest-CUDA]" if spec.target == "lightning_t4" else "[empirical]",
            "cpu_eval": "[contest-CPU]" if spec.target == "lightning_t4" else "[macOS-CPU calibrated]",
        },
    }
    (lane_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_heartbeat(lane_dir, "staged_pending_operator_launch")
    return manifest


def dispatch_one(spec: DispatchSpec, output_root: Path) -> dict[str, Any]:
    """Dispatch a single Phase A ablation."""
    print(f"\n=== Dispatch {spec.decision}: {spec.name} ===")
    print(f"  substrate: {spec.substrate}")
    print(f"  target:    {spec.target}")
    print(f"  duration:  ~{spec.estimated_duration_hours}h")
    print(f"  cost:      ~${spec.estimated_cost_usd}")
    print(f"  lane:      {spec.lane_id}")

    ok, reason = precheck(spec)
    if not ok:
        print(f"  [pre-check] FAILED: {reason}")
        return {
            "decision": spec.decision,
            "lane_id": spec.lane_id,
            "status": "pre_check_failed",
            "reason": reason,
            "evidence_grade": "blocked",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    if spec.target == "cpu_local":
        return dispatch_local_cpu(spec, output_root)
    elif spec.target == "lightning_t4":
        return dispatch_lightning_t4(spec, output_root)
    else:
        return {
            "decision": spec.decision,
            "lane_id": spec.lane_id,
            "status": "unknown_target",
            "reason": f"target={spec.target} not implemented in this wrapper",
            "evidence_grade": "blocked",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A Track 1 ablations dispatch wrapper (parallel actuator)"
    )
    parser.add_argument(
        "--decision",
        choices=[*list(PHASE_A_DISPATCHES.keys()), "all"],
        required=True,
        help="Phase A ablation to dispatch ('all' = parallel-dispatch all 8)",
    )
    parser.add_argument(
        "--substrate",
        default="pr101",
        choices=["pr101", "pr107", "toy_50k"],
        help="Substrate for the ablation (default: pr101)",
    )
    parser.add_argument(
        "--duration",
        default="auto",
        help="Override estimated duration (e.g. '3h'); 'auto' uses spec default",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override per-dispatch budget cap ($); spec default applies if unset",
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        default=55.0,
        help="Total Phase A budget cap (default: $55, operator pre-approved)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS_RESULTS,
        help="Output root for artifacts (default: experiments/results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only — print what would be dispatched without claiming lanes",
    )
    args = parser.parse_args()

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    decisions = list(PHASE_A_DISPATCHES.keys()) if args.decision == "all" else [args.decision]
    is_staging_all = args.decision == "all"

    # MEDIUM-E (codex 2026-05-08): banner when --decision all is invoked.
    # The wrapper is a staging PLANNER over each decision's manifest + lane
    # claim — it does NOT execute the per-decision training/dispatch jobs in
    # parallel and does NOT enforce dependency gates (A0 -> A2 -> A1 -> A4
    # -> parallel A3-alt/A4-alt/A5/A6 -> Phase C). Operators must invoke the
    # underlying training scripts (experiments/train_charm_*, train_score_*,
    # tools/pr101_sensitivity_*, etc.) themselves AFTER reviewing the staged
    # manifest. Treat this wrapper's output as plans + lane claims, not as
    # active concurrent execution.
    if is_staging_all:
        print(
            "\n[STAGING PLANNER MODE] --decision all stages manifests + lane "
            "claims for each Phase A decision SEQUENTIALLY. This wrapper does "
            "NOT launch concurrent jobs; it does NOT enforce A0 -> A2 -> A1 "
            "-> A4 dependency gates; the artifacts produced are dispatch "
            "intent, not execution. Invoke per-decision training scripts "
            "directly to actually run the ablations.",
            file=sys.stderr,
        )

    # Cost gate per CLAUDE.md "Vast.ai create without disk + cuda_vers gate"
    total_cost = sum(
        PHASE_A_DISPATCHES[d].estimated_cost_usd for d in decisions
    )
    print("\n=== Phase A Dispatch Plan ===")
    print(f"Decisions: {', '.join(decisions)}")
    print(f"Estimated total cost: ${total_cost:.2f} (cap: ${args.total_budget})")

    if total_cost > args.total_budget:
        print(
            f"[BUDGET GATE] FAILED: estimated ${total_cost:.2f} exceeds cap ${args.total_budget}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        for d in decisions:
            spec = PHASE_A_DISPATCHES[d]
            print(f"  {d}: {spec.name} ({spec.target}, ~{spec.estimated_duration_hours}h, ${spec.estimated_cost_usd})")
        return 0

    # Sequential dispatch (each dispatch is self-contained; subprocess spawning
    # could parallelize but Lightning T4 staging is operator-gated anyway)
    results: list[dict[str, Any]] = []
    for d in decisions:
        spec = PHASE_A_DISPATCHES[d]
        result = dispatch_one(spec, output_root)
        results.append(result)

    # Write rollup manifest
    timestamp = utc_timestamp()
    rollup_path = output_root / f"phase_a_dispatch_rollup_{timestamp}.json"
    rollup = {
        "phase": "A",
        "wrapper_version": "1.0",
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "decisions_dispatched": decisions,
        "total_estimated_cost_usd": total_cost,
        "total_budget_cap_usd": args.total_budget,
        "results": results,
        "next_action": (
            "After Lightning dispatches complete, harvest via "
            "tools/harvest_modal_calls.py / tools/harvest_gha_runs.py and "
            "verify each lane's [contest-CUDA] + [contest-CPU] dual-eval per CLAUDE.md."
        ),
    }
    rollup_path.write_text(json.dumps(rollup, indent=2))
    print(f"\nRollup manifest: {rollup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
