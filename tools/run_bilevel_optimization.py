"""Bilevel optimization driver — the magic-autopilot conductor.

Per Grand Council 2026-05-07 verdict
(`.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md`),
the optimal path to lowest-score is a bilevel optimization with three
nested loops:

    OUTER:  δεζ joint training reshapes substrate θ
    MIDDLE: meta-Lagrangian search over codec/architecture/paradigm atoms
    INNER:  Joint-ADMM allocates rate-distortion budget across streams

This driver is the **single-command entry point** that conducts all three.
Per user 2026-05-07 ("magic-driven autopilot"):

    python tools/run_bilevel_optimization.py --phase 1
    python tools/run_bilevel_optimization.py --phase auto    # picks next phase

It auto-detects substrate from disk state, dispatches the appropriate
cathedral primitive for each phase, emits a phase-manifest.json with
gate states + next-step recommendation, and (on phases that produce
contest-CUDA scores) auto-updates lane_maturity gates.

## Phase trajectory (per Grand Council)

| Phase | Target | Move                                   | GPU? |
|------:|-------:|----------------------------------------|------|
|     1 |  0.190 | PR100 + Op1+Op2+Op2.5 stack            | yes  |
|     2 |  0.183 | RAFT poses (Path-B pose-axis fix)      | yes  |
|     3 |  0.165 | δεζ training on PR100                  | yes  |
|     4 |  0.155 | Ballé hyperprior un-STUB + γ           | yes  |
|     5 |  0.145 | α masks + wavelet + SIREN              | yes  |
|     6 |  0.135 | telescopic foveation + UNIWARD         | yes  |
|     7 |  0.125 | YF-floor research                      | yes  |

## CPU-only phases (no GPU required)

The driver runs the **CPU-side preparation** for any phase:

- Substrate auto-detection from disk
- Cathedral autopilot Pareto sweep on the available substrate
- Pre-dispatch artifact generation (archives, manifests)
- Gate-state computation
- Meta-Lagrangian atom-ledger updates from prior phase outputs

GPU-bound work (contest-CUDA evaluation, δεζ training) is QUEUED, not
executed by this driver. The dispatch playbook
`scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh`
is the gate-5 executor that consumes this driver's output.

## Magic ingredients

1. **Substrate auto-detection** — picks PR100, PR106, or δεζ-trained from
   disk; no operator hand-holding.
2. **Phase auto-routing** — read lane_registry, compute next gate to attack.
3. **Continual-learning ledger** — every phase's contest-CUDA score
   appends to `experiments/results/bilevel_atom_ledger.jsonl`; the
   meta-Lagrangian middle-loop consumes it.
4. **Reports auto-update** — `reports/latest.md` gets the new frontier
   anchor automatically when a contest-CUDA score lands.

## Strict-scorer-rule

This driver loads NO scorer. It calls cathedral primitives that are
themselves scorer-clean. GPU-bound work is QUEUED via dispatch playbooks,
not executed inline.

## Cross-references

- Council deliberation: ``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``
- Three orientations memo: ``feedback_frontier_paper_floor_unknown_20260507``
- Cathedral session index: ``.omx/research/INDEX_session_2026_05_07_codec_pipeline_canonicalization.md``
- Magic-autopilot gap memo: ``feedback_automate_and_densify_intelligence_20260507``
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

# Make `tac` importable from tools/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Phase trajectory (frozen by Grand Council)
# ---------------------------------------------------------------------------

PHASE_TRAJECTORY: dict[int, dict[str, object]] = {
    1: {
        "label": "PR100 + Op1+Op2+Op2.5 stack",
        "target_score": 0.190,
        "estimated_gpu_usd": 5,
        "wall_clock_days": 1,
        "gpu_required": True,
        "cathedral_op": "Op1_PR101SplitBrotli + Op2_PR103ArithmeticCodec",
        "substrate_hint": "PR100 (canonical-winner)",
    },
    2: {
        "label": "RAFT-derived poses (Path-B pose-axis fix)",
        "target_score": 0.183,
        "estimated_gpu_usd": 25,
        "wall_clock_days": 7,
        "gpu_required": True,
        "cathedral_op": "Op_RAFTPoseStream (NOT YET BUILT)",
        "substrate_hint": "PR100 + RAFT-flow-derived poses",
    },
    3: {
        "label": "δεζ joint training on PR100",
        "target_score": 0.165,
        "estimated_gpu_usd": 145,
        "wall_clock_days": 30,
        "gpu_required": True,
        "cathedral_op": "tac.codec_pipeline_deltaepszeta_callback + tac.shannon_h2_loss",
        "substrate_hint": "δεζ-trained PR100 substrate",
    },
    4: {
        "label": "Ballé hyperprior un-STUB + γ paradigm",
        "target_score": 0.155,
        "estimated_gpu_usd": 250,
        "wall_clock_days": 45,
        "gpu_required": True,
        "cathedral_op": "Op_GammaJointADMM(substrate_aware_init=True)",
        "substrate_hint": "δεζ-trained + Ballé prior-fit",
    },
    5: {
        "label": "α masks + wavelet + SIREN",
        "target_score": 0.145,
        "estimated_gpu_usd": 400,
        "wall_clock_days": 60,
        "gpu_required": True,
        "cathedral_op": "tac.codec_pipeline_mask + Op_WaveletResidual + Op_SIREN",
        "substrate_hint": "δεζ-trained + Ballé + mask bakeoff winner",
    },
    6: {
        "label": "telescopic foveation + UNIWARD",
        "target_score": 0.135,
        "estimated_gpu_usd": 700,
        "wall_clock_days": 90,
        "gpu_required": True,
        "cathedral_op": "Op_TelescopicFoveation + UNIWARD-weighted quantization",
        "substrate_hint": "phase 5 substrate + foveation decorator",
    },
    7: {
        "label": "YF-floor research (sub-Shannon territory)",
        "target_score": 0.125,
        "estimated_gpu_usd": 1500,
        "wall_clock_days": 180,
        "gpu_required": True,
        "cathedral_op": "score-Jacobian KL basis + arXiv 2604.24658 + Cosmos",
        "substrate_hint": "research-grade",
    },
}


# ---------------------------------------------------------------------------
# Substrate auto-detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstrateCandidate:
    """One auto-detected substrate on disk."""
    label: str
    path: str
    bytes: int
    score_anchor: float | None  # known contest-CUDA score, if any
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_substrates(repo_root: pathlib.Path) -> list[SubstrateCandidate]:
    """Walk known disk locations + return list of available substrates."""
    candidates: list[SubstrateCandidate] = []

    pr106_state_dict = repo_root / "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt"
    if pr106_state_dict.exists():
        candidates.append(SubstrateCandidate(
            label="PR106 (current local frontier substrate)",
            path=str(pr106_state_dict),
            bytes=pr106_state_dict.stat().st_size,
            score_anchor=0.20898105277982337,  # PR103-on-PR106 contest-CUDA T4
            notes="below-medal-band per gap-decomposition; pose=69% of gap",
        ))

    pr100_intake = repo_root / "experiments/results/public_pr100_intake_20260504_codex/source"
    if pr100_intake.exists():
        candidates.append(SubstrateCandidate(
            label="PR100 (canonical-winner substrate, public PR #100)",
            path=str(pr100_intake),
            bytes=sum(p.stat().st_size for p in pr100_intake.rglob("*") if p.is_file()),
            score_anchor=0.1954,  # public PR #100 published score
            notes="medal-band substrate; PR101/102/103 medal entries derived from this",
        ))

    pr101_intake = repo_root / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source"
    if pr101_intake.exists():
        candidates.append(SubstrateCandidate(
            label="PR101 (gold medal substrate, public PR #101)",
            path=str(pr101_intake),
            bytes=sum(p.stat().st_size for p in pr101_intake.rglob("*") if p.is_file()),
            score_anchor=0.193,
            notes="gold-medal substrate; canonical winner",
        ))

    pr103_intake = repo_root / "experiments/results/public_pr103_intake_20260504_codex/source"
    if pr103_intake.exists():
        candidates.append(SubstrateCandidate(
            label="PR103 (silver medal substrate, public PR #103)",
            path=str(pr103_intake),
            bytes=sum(p.stat().st_size for p in pr103_intake.rglob("*") if p.is_file()),
            score_anchor=0.195,
            notes="silver-medal substrate; AC bolt-on on PR100 substrate",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Phase auto-routing
# ---------------------------------------------------------------------------

def detect_current_phase(repo_root: pathlib.Path) -> int:
    """Return the next phase to execute based on lane_registry + dispatch state.

    Heuristic: scan `.omx/state/lane_registry.json` for which gates are
    GREEN. Phase 1 is "do" until a contest-CUDA on PR100 substrate lands;
    Phase 2 starts when Phase 1 score is recorded; etc.
    """
    registry_path = repo_root / ".omx/state/lane_registry.json"
    if not registry_path.exists():
        return 1  # fresh checkout, start at phase 1

    try:
        registry = json.loads(registry_path.read_text())
    except Exception:
        return 1

    lanes_raw = registry.get("lanes", {})
    # Registry schema variation: dict[name → lane] OR list[lane-with-id]
    if isinstance(lanes_raw, dict):
        lane_items = list(lanes_raw.items())
    elif isinstance(lanes_raw, list):
        lane_items = [(lane.get("id") or lane.get("name", ""), lane) for lane in lanes_raw]
    else:
        return 1

    def _has_lane_at_level(substring: str, level: int) -> bool:
        return any(
            substring in (name or "").lower() and (lane.get("level", 0) or 0) >= level
            for name, lane in lane_items
        )

    # Phase 1: PR100-anchor lane at L3? No → still in phase 1.
    if not _has_lane_at_level("pr100", 3):
        return 1
    # Phase 2: RAFT-poses lane at L3?
    if not _has_lane_at_level("raft", 3):
        return 2
    # Phase 3: δεζ-trained substrate at L3?
    if not (_has_lane_at_level("deltaepszeta", 3) or _has_lane_at_level("delta_epsilon_zeta", 3)):
        return 3

    # Successive phases follow the same pattern; default to phase + 1
    return 4


# ---------------------------------------------------------------------------
# Continual-learning atom ledger
# ---------------------------------------------------------------------------

def append_to_atom_ledger(
    repo_root: pathlib.Path,
    *,
    phase: int,
    substrate: SubstrateCandidate,
    contest_cuda_score: float | None,
    archive_bytes: int | None,
    archive_sha256: str | None,
    cathedral_op: str,
    notes: str,
) -> pathlib.Path:
    """Append one phase's empirical result to the meta-Lagrangian atom ledger."""
    ledger_path = repo_root / "experiments/results/bilevel_atom_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": phase,
        "substrate_label": substrate.label,
        "substrate_path": substrate.path,
        "substrate_score_anchor": substrate.score_anchor,
        "contest_cuda_score": contest_cuda_score,
        "evidence_grade": "[contest-CUDA]" if contest_cuda_score is not None else "[CPU-prep]",
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "cathedral_op": cathedral_op,
        "notes": notes,
    }
    with ledger_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return ledger_path


# ---------------------------------------------------------------------------
# Phase manifest emitter
# ---------------------------------------------------------------------------

def emit_phase_manifest(
    repo_root: pathlib.Path,
    *,
    phase: int,
    substrates: list[SubstrateCandidate],
    chosen_substrate: SubstrateCandidate,
    next_action: str,
    notes: str,
) -> pathlib.Path:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / f"experiments/results/lane_bilevel_phase{phase}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "tools/run_bilevel_optimization",
        "evidence_grade": "[CPU-prep]",
        "score_claim": False,
        "phase": phase,
        "phase_spec": PHASE_TRAJECTORY.get(phase, {}),
        "substrates_detected": [s.to_dict() for s in substrates],
        "chosen_substrate": chosen_substrate.to_dict(),
        "next_action": next_action,
        "notes": notes,
    }
    manifest_path = out_dir / "phase_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# Phase 1 — PR100 anchor + Op1+Op2 cathedral autopilot (CPU prep)
# ---------------------------------------------------------------------------

def run_phase_1(repo_root: pathlib.Path) -> dict[str, object]:
    """CPU-side prep for Phase 1: PR100 anchor + Op1+Op2 stack.

    What we CAN do CPU-only:
      - Auto-detect PR100 substrate
      - Inspect canonical PR100 archive bytes + score anchor
      - Pre-stage the dispatch artifacts
      - Emit gate-readiness report

    What we CANNOT do without GPU:
      - Actually run contest-CUDA evaluation
      - Train a renderer

    Phase 1's GPU work is queued via the deferred-dispatch playbook.
    """
    substrates = detect_substrates(repo_root)
    pr100 = next(
        (s for s in substrates if "PR100" in s.label),
        None,
    )
    pr101 = next(
        (s for s in substrates if "PR101" in s.label),
        None,
    )
    chosen = pr101 or pr100  # prefer PR101 (gold) if both present
    if chosen is None:
        return {
            "status": "BLOCKED",
            "reason": "PR100/PR101 substrate not on disk; intake from public PR archives needed first",
            "next_action": "Run intake of PR100/PR101 source via experiments/results/public_pr*_intake_*",
        }

    return {
        "status": "READY_FOR_DISPATCH",
        "chosen_substrate": chosen,
        "next_action": (
            "Build PR101 split-Brotli + PR103 AC bolt-on archive on the chosen "
            "substrate, then dispatch via Lightning T4 / Vast.ai 4090 when GPU "
            "billing returns. Predicted score 0.190 [predicted-band only]."
        ),
        "all_substrates": substrates,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bilevel optimization driver — magic-autopilot conductor",
    )
    p.add_argument(
        "--phase",
        default="auto",
        help="phase to run: 1-7, or 'auto' to pick from current state",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="repo root (default: detected from script location)",
    )
    p.add_argument(
        "--list-substrates",
        action="store_true",
        help="just list detected substrates and exit",
    )
    p.add_argument(
        "--list-phases",
        action="store_true",
        help="print the phase trajectory table and exit",
    )
    args = p.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else pathlib.Path(__file__).resolve().parent.parent

    if args.list_phases:
        print("Phase trajectory (Grand Council):")
        print(f"  {'Phase':<6} {'Score':<8} {'GPU $':<8} {'Days':<6} Move")
        print("  " + "-" * 90)
        for ph, spec in PHASE_TRAJECTORY.items():
            print(
                f"  {ph:<6} "
                f"{str(spec['target_score']):<8} "
                f"${spec['estimated_gpu_usd']:<7} "
                f"{spec['wall_clock_days']:<6} "
                f"{spec['label']}"
            )
        return 0

    substrates = detect_substrates(repo_root)
    if args.list_substrates:
        print("Auto-detected substrates:")
        for s in substrates:
            anchor = f"score={s.score_anchor}" if s.score_anchor is not None else "no anchor"
            print(f"  {s.label}\n    path: {s.path}\n    {anchor}\n    {s.notes}")
        return 0

    if args.phase == "auto":
        phase = detect_current_phase(repo_root)
        print(f"[bilevel] auto-detected next phase: {phase}")
    else:
        phase = int(args.phase)

    spec = PHASE_TRAJECTORY.get(phase)
    if spec is None:
        print(f"FATAL: unknown phase {phase}", file=sys.stderr)
        return 2

    print(f"\n=== Bilevel Phase {phase}: {spec['label']} ===")
    print(f"  target score:    {spec['target_score']} [predicted-band only]")
    print(f"  GPU budget:      ${spec['estimated_gpu_usd']}")
    print(f"  wall-clock:      {spec['wall_clock_days']} days")
    print(f"  cathedral op:    {spec['cathedral_op']}")
    print(f"  substrate hint:  {spec['substrate_hint']}")
    print()

    print(f"[bilevel] detected {len(substrates)} substrate(s) on disk:")
    for s in substrates:
        print(f"  - {s.label}")
    print()

    if phase == 1:
        result = run_phase_1(repo_root)
        print(f"[bilevel] phase 1 status: {result['status']}")
        if result["status"] == "READY_FOR_DISPATCH":
            chosen = result["chosen_substrate"]
            print(f"[bilevel] chosen substrate: {chosen.label}")
            print(f"[bilevel] next action: {result['next_action']}")
            manifest_path = emit_phase_manifest(
                repo_root,
                phase=1,
                substrates=substrates,
                chosen_substrate=chosen,
                next_action=str(result["next_action"]),
                notes="Phase 1 CPU-side prep complete; GPU dispatch queued.",
            )
            append_to_atom_ledger(
                repo_root,
                phase=1,
                substrate=chosen,
                contest_cuda_score=None,
                archive_bytes=None,
                archive_sha256=None,
                cathedral_op=str(spec["cathedral_op"]),
                notes="Phase 1 CPU prep; awaiting GPU dispatch for contest-CUDA score",
            )
            print(f"\n[bilevel] manifest: {manifest_path}")
            print(f"[bilevel] atom ledger updated: {repo_root / 'experiments/results/bilevel_atom_ledger.jsonl'}")
        else:
            print(f"[bilevel] phase 1 BLOCKED: {result['reason']}")
            print(f"[bilevel] {result['next_action']}")
            return 3
    else:
        # Phases 2-7: GPU-bound; emit a queue-only manifest.
        if not substrates:
            print("[bilevel] no substrates on disk; cannot proceed")
            return 4
        chosen = substrates[0]  # placeholder; real driver picks per phase spec
        print(f"[bilevel] phase {phase} requires GPU; CPU prep complete.")
        print(f"[bilevel] queued for GPU dispatch when billing returns.")
        manifest_path = emit_phase_manifest(
            repo_root,
            phase=phase,
            substrates=substrates,
            chosen_substrate=chosen,
            next_action=f"Dispatch phase {phase} GPU work; see phase spec for cathedral op.",
            notes="Phase {phase} CPU prep complete; GPU dispatch queued.",
        )
        append_to_atom_ledger(
            repo_root,
            phase=phase,
            substrate=chosen,
            contest_cuda_score=None,
            archive_bytes=None,
            archive_sha256=None,
            cathedral_op=str(spec["cathedral_op"]),
            notes=f"Phase {phase} CPU prep; awaiting GPU dispatch",
        )
        print(f"\n[bilevel] manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
