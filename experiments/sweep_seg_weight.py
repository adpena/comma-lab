#!/usr/bin/env python3
"""Lane T2-RATIO: bounded sweep above the historical SegNet weight cap.

INTENTIONAL_OVERRIDE: CLAUDE.md marks ``segnet_loss_weight > 100`` as
forbidden for ordinary lanes. T2-RATIO intentionally overrides that cap only
inside this bounded grid, with PoseNet-loss-floor protection. If PoseNet loss
exceeds 2x the baseline at an epoch check, the trial reverts to the last good
checkpoint and skips the remaining epochs.

Predicted band: [1.00, 1.15] [advisory only].
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SEG_WEIGHT_GRID: tuple[int, ...] = (120, 150, 200, 300, 500)
POSE_FLOOR_MULTIPLIER = 2.0
RESULTS_FILENAME = "sweep_seg_weight_results.json"


@dataclass(frozen=True)
class PoseFloorDecision:
    triggered: bool
    current_posenet_loss: float
    baseline_posenet_loss: float
    threshold: float
    reverted_to_checkpoint: str | None = None
    skip_remaining_epochs: bool = False


@dataclass(frozen=True)
class TrialConfig:
    profile: str
    tag_prefix: str
    precomputed: str | None
    device: str
    check_every: int
    baseline_posenet_loss: float | None
    train_module: str
    auth_eval_masks: str | None
    auth_eval_poses: str | None
    auth_eval_upstream_dir: str | None
    auth_eval_during_sweep: bool
    max_frames: int | None


class SubprocessTrialState:
    """Tiny state object used by the PoseNet-floor guard."""

    def __init__(self, trial_dir: Path) -> None:
        self.trial_dir = trial_dir
        self.last_good_checkpoint: Path | None = None
        self.reverted_to_checkpoint: Path | None = None

    def revert_to_checkpoint(self, checkpoint: Path) -> None:
        self.reverted_to_checkpoint = checkpoint
        if checkpoint.exists():
            dst = self.trial_dir / "pose_floor_reverted_checkpoint.pt"
            shutil.copy2(checkpoint, dst)


def maybe_revert_for_posenet_floor(
    training_state: object,
    *,
    current_posenet_loss: float,
    baseline_posenet_loss: float,
    multiplier: float = POSE_FLOOR_MULTIPLIER,
) -> PoseFloorDecision:
    """Revert a trial when PoseNet loss breaches the configured floor.

    ``training_state`` is intentionally duck-typed for unit testing and for
    lightweight orchestration. It must expose ``last_good_checkpoint`` and may
    expose ``revert_to_checkpoint(path)`` or ``revert_to_last_good_checkpoint()``.
    """
    threshold = float(baseline_posenet_loss) * float(multiplier)
    if not math.isfinite(current_posenet_loss) or current_posenet_loss > threshold:
        checkpoint = getattr(training_state, "last_good_checkpoint", None)
        if checkpoint is not None and hasattr(training_state, "revert_to_checkpoint"):
            training_state.revert_to_checkpoint(Path(checkpoint))
        elif hasattr(training_state, "revert_to_last_good_checkpoint"):
            training_state.revert_to_last_good_checkpoint()
        return PoseFloorDecision(
            triggered=True,
            current_posenet_loss=float(current_posenet_loss),
            baseline_posenet_loss=float(baseline_posenet_loss),
            threshold=threshold,
            reverted_to_checkpoint=str(checkpoint) if checkpoint is not None else None,
            skip_remaining_epochs=True,
        )
    return PoseFloorDecision(
        triggered=False,
        current_posenet_loss=float(current_posenet_loss),
        baseline_posenet_loss=float(baseline_posenet_loss),
        threshold=threshold,
        skip_remaining_epochs=False,
    )


def select_best_trial(trials: Sequence[Mapping]) -> Mapping:
    """Select the completed trial with the minimum authoritative score."""
    candidates = [
        t for t in trials
        if t.get("auth_score") is not None or t.get("proxy_score") is not None
    ]
    if not candidates:
        raise ValueError("no trials with auth_score or proxy_score")

    def key(trial: Mapping) -> float:
        score = trial.get("auth_score")
        if score is None:
            score = trial.get("proxy_score")
        return float(score)

    return min(candidates, key=key)


def _latest_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    latest = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        latest = json.loads(line)
    return latest


def _read_best_meta(trial_dir: Path, tag: str) -> dict | None:
    path = trial_dir / f"renderer_{tag}_best_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_auth_eval_result(trial_dir: Path) -> dict | None:
    for candidate in (
        trial_dir / "auth_eval_on_best.json",
        trial_dir / "auth_eval_renderer.json",
        trial_dir / "contest_auth_eval.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text())
    log = trial_dir / "auth_eval.log"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line.split("RESULT_JSON:", 1)[1].strip())
    return None


def run_training_trial(
    seg_weight: int,
    trial_dir: Path,
    epochs: int,
    config: TrialConfig,
) -> dict:
    """Run one sweep trial via the canonical renderer trainer."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{config.tag_prefix}_seg{seg_weight}"
    state = SubprocessTrialState(trial_dir)
    baseline_pose = config.baseline_posenet_loss
    log_path = trial_dir / "train.log"
    resume_from: Path | None = None

    completed_epochs = 0
    status = "completed"
    floor_decision: PoseFloorDecision | None = None

    for chunk_end in range(config.check_every, epochs + config.check_every, config.check_every):
        target_epochs = min(chunk_end, epochs)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            config.train_module,
            "--profile",
            config.profile,
            "--tag",
            tag,
            "--output-dir",
            str(trial_dir),
            "--epochs",
            str(target_epochs),
            "--eval-every",
            str(config.check_every),
            "--segnet-weight",
            str(seg_weight),
            "--device",
            config.device,
        ]
        if config.precomputed:
            cmd += ["--precomputed", config.precomputed]
        if resume_from is not None:
            cmd += ["--resume-from", str(resume_from)]
        if config.max_frames is not None:
            cmd += ["--max-frames", str(config.max_frames)]
        if config.auth_eval_during_sweep:
            if config.auth_eval_masks:
                cmd += ["--auth-eval-masks", config.auth_eval_masks]
            if config.auth_eval_poses:
                cmd += ["--auth-eval-poses", config.auth_eval_poses]
            if config.auth_eval_upstream_dir:
                cmd += ["--auth-eval-upstream-dir", config.auth_eval_upstream_dir]
        else:
            cmd.append("--no-auth-eval-on-best")

        with log_path.open("a") as log:
            log.write("\n$ " + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        completed_epochs = target_epochs
        if proc.returncode != 0:
            status = "failed"
            break

        telemetry = _latest_jsonl(trial_dir / f"{tag}_telemetry.jsonl")
        state_checkpoint = trial_dir / f"training_state_{tag}.pt"
        if telemetry and state_checkpoint.exists():
            current_pose = float(telemetry.get("eval_pose", 0.0))
            if baseline_pose is None and current_pose > 0:
                baseline_pose = current_pose
            if baseline_pose is not None and current_pose > 0:
                floor_decision = maybe_revert_for_posenet_floor(
                    state,
                    current_posenet_loss=current_pose,
                    baseline_posenet_loss=baseline_pose,
                )
                if floor_decision.triggered:
                    status = "pose_floor_reverted"
                    break
                state.last_good_checkpoint = state_checkpoint
        resume_from = state_checkpoint if state_checkpoint.exists() else resume_from
        if target_epochs >= epochs:
            break

    best_meta = _read_best_meta(trial_dir, tag) or {}
    auth = _read_auth_eval_result(trial_dir) or {}
    auth_score = auth.get("final_score") or auth.get("score") or auth.get("auth_score")
    result = {
        "lane": "T2-RATIO",
        "seg_weight": seg_weight,
        "status": status,
        "epochs_requested": epochs,
        "epochs_completed": completed_epochs,
        "profile": config.profile,
        "tag": tag,
        "trial_dir": str(trial_dir),
        "checkpoint": str(trial_dir / f"renderer_{tag}_best_fp32.pt"),
        "proxy_score": best_meta.get("scorer"),
        "auth_score": auth_score,
        "posenet_floor": asdict(floor_decision) if floor_decision else None,
    }
    (trial_dir / "trial_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


TrialRunner = Callable[[int, Path, int], dict]


def run_sweep(
    *,
    output_dir: Path,
    trial_runner: TrialRunner,
    epochs: int,
    seg_weights: Iterable[int] = SEG_WEIGHT_GRID,
) -> Path:
    """Run all T2-RATIO trials and write aggregate JSON results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for seg_weight in seg_weights:
        trial_dir = output_dir / f"seg_weight_{seg_weight}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        result = dict(trial_runner(int(seg_weight), trial_dir, int(epochs)))
        result.setdefault("seg_weight", int(seg_weight))
        trials.append(result)
        (trial_dir / "trial_result.json").write_text(json.dumps(result, indent=2) + "\n")

    best = select_best_trial(trials) if trials else None
    payload = {
        "lane": "T2-RATIO",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intentional_override": (
            "INTENTIONAL_OVERRIDE: segnet_loss_weight > 100 is bounded here "
            "by PoseNet-loss-floor protection."
        ),
        "predicted_band": [1.00, 1.15],
        "score_tag": "[advisory only]",
        "seg_weights": [int(w) for w in seg_weights],
        "epochs": int(epochs),
        "pose_floor_multiplier": POSE_FLOOR_MULTIPLIER,
        "best_trial": dict(best) if best is not None else None,
        "trials": trials,
    }
    result_path = output_dir / RESULTS_FILENAME
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/t2ratio_sweep"))
    p.add_argument("--epochs", type=int, default=250, choices=range(200, 301),
                   metavar="[200-300]")
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--profile", type=str, default="proven_baseline")
    p.add_argument("--tag-prefix", type=str, default="t2ratio")
    p.add_argument("--precomputed", type=str, default="experiments/precomputed_local")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--baseline-posenet-loss", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--train-module", type=str, default="tac.experiments.train_renderer")
    p.add_argument("--auth-eval-during-sweep", action="store_true",
                   help="Run train_renderer's built-in CUDA auth eval per trial.")
    p.add_argument("--auth-eval-masks", type=str, default=None)
    p.add_argument("--auth-eval-poses", type=str, default=None)
    p.add_argument("--auth-eval-upstream-dir", type=str, default="upstream")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrialConfig(
        profile=args.profile,
        tag_prefix=args.tag_prefix,
        precomputed=args.precomputed,
        device=args.device,
        check_every=args.check_every,
        baseline_posenet_loss=args.baseline_posenet_loss,
        train_module=args.train_module,
        auth_eval_masks=args.auth_eval_masks,
        auth_eval_poses=args.auth_eval_poses,
        auth_eval_upstream_dir=args.auth_eval_upstream_dir,
        auth_eval_during_sweep=bool(args.auth_eval_during_sweep),
        max_frames=args.max_frames,
    )

    def runner(seg_weight: int, trial_dir: Path, epochs: int) -> dict:
        return run_training_trial(seg_weight, trial_dir, epochs, config)

    result_path = run_sweep(
        output_dir=args.output_dir,
        trial_runner=runner,
        epochs=args.epochs,
    )
    print(f"T2-RATIO sweep results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
