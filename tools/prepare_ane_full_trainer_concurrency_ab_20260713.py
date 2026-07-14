#!/usr/bin/env python3
"""Prepare, but never fire, the held n24 ANE/full-trainer concurrency A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path

from tac.witness_dsl.ane_unlock_followup_policy_20260713 import (
    ARMS,
    compile_ane_trainer_concurrency_ticket,
)
from tac.witness_dsl.spec_throughput_component_timer_20260713 import (
    compile_throughput_component_timer_ticket,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "experiments/results/ane_full_trainer_concurrency_ab_20260713"
GT_N24 = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n24.npz"
SIBLING_GO_PACKET = REPO / ".omx/research/GO_PACKET_inloop_component_timer_20260713.md"
SIBLING_MEASUREMENTS = REPO / ".omx/research/throughput_fresh_eyes_measurements_20260713.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _launcher_argv(out_dir: Path, arm: str) -> list[str]:
    purpose = (
        "HELD operator-GO n24 real-state ANE/full-trainer concurrency A/B; "
        f"arm={arm}; identical four-epoch CE-exact component timer; MEANS only"
    )
    return [
        ".venv/bin/python",
        "tools/launch_witness_run.py",
        "--gt-cache",
        str(GT_N24.relative_to(REPO)),
        "--num-pairs",
        "24",
        "--epochs",
        "4",
        "--config",
        "throughput_component_timer_solo_20260713",
        "--out-dir",
        str(out_dir.relative_to(REPO)),
        "--label",
        f"ane_concurrency_{arm}_n24_20260713",
        "--purpose",
        purpose,
        "--no-dashboard",
    ]


def _sidecar_argv(out_dir: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/bench_ane_unlock_followup_20260713.py",
        "--bootstrap-offline",
        "--mode",
        "sidecar",
        "--duration-s",
        "900",
        "--resume-sidecar",
        "--out-dir",
        str(out_dir.relative_to(REPO)),
    ]


def build_manifest(out_dir: Path) -> dict[str, object]:
    missing = [str(path) for path in (GT_N24, SIBLING_GO_PACKET, SIBLING_MEASUREMENTS) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"GO packet REFUSE: missing canonical input(s): {missing}")
    arms: dict[str, object] = {}
    compile_blockers: list[dict[str, str]] = []
    for arm in ARMS:
        arm_dir = out_dir / arm
        ticket = compile_ane_trainer_concurrency_ticket(arm)
        launcher = _launcher_argv(arm_dir, arm)
        row: dict[str, object] = {
            "ticket": ticket.to_dict(),
            "launcher_argv": launcher,
            "launcher_command": shlex.join(launcher),
        }
        try:
            compiled = compile_throughput_component_timer_ticket(
                str(GT_N24.relative_to(REPO)),
                num_pairs=24,
                epochs=4,
                out_dir=str(arm_dir.relative_to(REPO)),
                variant="solo_control",
            )
            trainer_argv = compiled.typed.to_program().compile_trainer_argv()
            row.update({
                "compile_status": "COMPLETE",
                "compiled_program_name": compiled.name,
                "compiled_trainer_argv_sha256": hashlib.sha256(
                    json.dumps(trainer_argv, sort_keys=False, separators=(",", ":")).encode()
                ).hexdigest(),
            })
        except Exception as exc:
            blocker = {"arm": arm, "error_type": type(exc).__name__, "error": str(exc)[:8000]}
            compile_blockers.append(blocker)
            row.update({
                "compile_status": "BLOCKED_NOT_GO_READY",
                "compiled_program_name": None,
                "compiled_trainer_argv_sha256": None,
                "compile_blocker": blocker,
            })
        if arm == "trainer_plus_ane_sidecar":
            sidecar_dir = out_dir / "frozen_teacher_sidecar"
            sidecar = _sidecar_argv(sidecar_dir)
            row["sidecar_argv"] = sidecar
            row["sidecar_command"] = shlex.join(sidecar)
            row["ordering"] = (
                "start/resume sidecar; wait for sidecar_receipt.json status=in_progress; "
                "then invoke the governed launcher; never bypass launch_witness_run.py"
            )
        arms[arm] = row
    status = "BLOCKED_TIMER_DSL_NOT_GO_READY" if compile_blockers else "HELD_OPERATOR_GO_REQUIRED"
    return {
        "schema": "ane_full_trainer_concurrency_ab_go_packet.v1",
        "status": status,
        "lane_id": "lane_ane_unlock_followup_20260713",
        "axis": "[n24 real-state macOS MLX/CoreML timing advisory] NON-PROMOTABLE",
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
        "n_real_states": 24,
        "n600_transfer_rule": "timing may be labeled n24-linear-extrapolation x25; never n600 measured",
        "treatment_invariant": "same typed trainer config, seed, cache, epochs, and telemetry; only frozen teacher sidecar differs",
        "acceptance": {
            "teacher_degradation_strictly_less_than_fraction": 0.05,
            "mlx_trainer_degradation_strictly_less_than_fraction": 0.05,
            "ane_placement_required_for_architecture_accept": True,
        },
        "derived_payoff": {
            "settled_forward_replacement_upper_case_speedup": 2.293,
            "label": "DERIVED from prior measured full-float32 CoreML forward and assumed teacher fraction; not this A/B",
            "actual_training_loop_payoff": "BLOCKED pending sibling in-loop timer plus this held A/B",
        },
        "sibling_custody": {
            str(SIBLING_GO_PACKET.relative_to(REPO)): _sha256(SIBLING_GO_PACKET),
            str(SIBLING_MEASUREMENTS.relative_to(REPO)): _sha256(SIBLING_MEASUREMENTS),
        },
        "arms": arms,
        "compile_blockers": compile_blockers,
        "verdict_scope": "the sibling-owned four-epoch component-timer DSL on the current main worktree",
        "req_R": (
            "the sibling owner must restore a typed four-epoch schedule whose LADDER lane and movable arm "
            "windows end strictly before Muon, after which this preparer and both governed --dry-run commands pass"
            if compile_blockers else
            "fresh lane claim plus explicit operator GO, then the exact two bounded governed arms"
        ),
        "operator_instruction": (
            "Do not GO while status is BLOCKED. Once the typed compile blocker is cleared, one GO authorizes "
            "these two bounded arms in sequence. Claim/recheck the lane immediately before fire; run the solo "
            "arm to completion, then the sidecar arm. The preparer itself cannot launch."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    manifest = build_manifest(out_dir)
    path = out_dir / "go_packet.json"
    _atomic_json(path, manifest)
    print(json.dumps({"status": manifest["status"], "path": str(path), "arms": list(manifest["arms"])}, indent=2))
    return 2 if str(manifest["status"]).startswith("BLOCKED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
