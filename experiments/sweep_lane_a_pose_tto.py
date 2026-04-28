"""Bayesian sweep for Lane A pose TTO.

Lane A baseline (hand-tuned, scored 1.15 [contest-CUDA] on 2026-04-27):
    --steps 500 --batch-pairs 8 --tto-lr 0.01 --posetto-noise-std 0.5

This sweep replaces those four hand-tuned numbers with a TPE-driven search.
Predicted improvement band: [0.95, 1.15] — Bayesian search should find at
least Lane A baseline, often better, since the hand-pick was an educated
guess (no formal study), and the proxy-auth gap means small TTO-config
changes can move auth score 0.05-0.20 points.

Search-space bounds — LITERATURE-INFORMED (no arbitrariness):
    tto_steps           int  [200, 1500]
        Lower 200: below this, PoseNet phase-transition (~80-100 steps)
        hasn't completed for non-warm-start runs (CLAUDE.md "TTO step
        curve" empirical, Vast.ai 4090). Lane A warm-starts so 200 is
        a reasonable floor. Upper 1500: SegNet plateau is reached by
        300-500 steps; further steps are wasted compute on this arch.
    batch_pairs         int  [4, 16]
        Lower 4: VRAM minimum that still gives stable EfficientNet-B2 grad
        signal. Upper 16: above this OOMs on 4090 24GB
        (memory: feedback_pose_tto_oom_4090).
    tto_lr              loguniform [1e-3, 1e-1]
        Lane A baseline 0.01 sits in the middle; widen log-decade either
        way so we sample both more conservative and more aggressive lrs.
    posetto_noise_std   uniform [0.1, 1.0]
        Hotz-Fridrich roundtrip noise. Lane A baseline 0.5; sweep 0.1-1.0
        to test whether the proxy-auth gap shrinks with stronger/weaker noise.
    eval_roundtrip      categorical [True]   # FIXED — CLAUDE.md non-negotiable.

Sampler: TPE (Bergstra et al. 2011). Acquisition: EI (Optuna default).
Pruner: MedianPruner (n_startup_trials=5, n_warmup_steps=10).

Provenance: study.db SQLite + trial_history.jsonl + lane_a_sweep_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tac.sweep_runner import BayesianSweep, OptunaTrialDispatcher  # noqa: E402

LANE_A_TEMPLATE = REPO_ROOT / "scripts" / "remote_lane_a_sweep.sh"

# Citation-backed bounds; see module docstring for rationale.
LANE_A_SEARCH_SPACE: dict[str, tuple] = {
    "tto_steps":          ("int", 200, 1500),
    "batch_pairs":        ("int", 4, 16),
    "tto_lr":             ("loguniform", 1e-3, 1e-1),
    "posetto_noise_std":  ("uniform", 0.1, 1.0),
    "eval_roundtrip":     ("fixed", True),  # CLAUDE.md non-negotiable
    "device":             ("fixed", "cuda"),  # CLAUDE.md non-negotiable
}

LANE_A_PREDICTED_BAND = (0.95, 1.15)
LANE_A_BASELINE_HAND_TUNED = 1.15


def build_sweep(output_dir: Path, n_trials: int) -> BayesianSweep:
    """Instantiate the Lane A sweep with all canonical settings.

    Pure factory — no side effects beyond what BayesianSweep does in its
    __post_init__ (search-space validation + non-negotiable enforcement).
    """
    return BayesianSweep(
        name="lane_a_pose_tto",
        script_template=LANE_A_TEMPLATE,
        search_space=LANE_A_SEARCH_SPACE,
        n_trials=n_trials,
        objective="auth_score",
        direction="minimize",
        output_dir=output_dir,
        predicted_band=LANE_A_PREDICTED_BAND,
    )


def _emit_optimized_remote_script(
    output_dir: Path,
    best_params: dict,
    best_value: float,
) -> Path:
    """After the sweep finishes, write a 'best-config' remote script.

    This is the artifact the operator ships for the OFFICIAL Lane A-Sweep
    measurement: the same template but with the Bayesian-optimal parameter
    values baked in. One canonical script, one canonical archive.
    """
    template_text = LANE_A_TEMPLATE.read_text()
    rendered = template_text
    for name, value in best_params.items():
        tok = f"__PARAM_{name.upper()}__"
        if isinstance(value, bool):
            rendered = rendered.replace(tok, "1" if value else "0")
        else:
            rendered = rendered.replace(tok, str(value))
    # Tag this is the 'optimized' build, not a per-trial script.
    rendered = rendered.replace("__SWEEP_NAME__", "lane_a_optimized")
    rendered = rendered.replace("__SWEEP_TRIAL_NUMBER__", "BEST")

    out_path = REPO_ROOT / "scripts" / "remote_lane_a_optimized.sh"
    out_path.write_text(rendered)
    out_path.chmod(0o755)

    sidecar = out_path.with_suffix(".params.json")
    sidecar.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_value": best_value,
                "sweep_name": "lane_a_pose_tto",
                "predicted_band": list(LANE_A_PREDICTED_BAND),
                "baseline_hand_tuned": LANE_A_BASELINE_HAND_TUNED,
            },
            indent=2,
            default=str,
        )
    )
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian sweep over Lane A pose TTO hyperparameters",
    )
    p.add_argument("--n-trials", type=int, default=30,
                   help="Total Optuna trials (default 30 — fits in $7-10 budget)")
    p.add_argument("--objective", type=str, default="auth_score",
                   help="Metric to optimize (provenance only; parser is fixed)")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments" / "results" / "sweep_lane_a",
                   help="Where trial scripts + history + study.db are written")
    p.add_argument("--smoke", action="store_true",
                   help="Run local smoke (NO GPU; tests plumbing only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = build_sweep(args.output_dir, n_trials=args.n_trials)
    sweep.objective = args.objective

    dispatcher = OptunaTrialDispatcher(sweep)

    if args.smoke:
        summary = dispatcher.local_smoke()
        print(f"[lane-a-sweep] SMOKE complete (no GPU). best_value={summary['best_value']:.4f}")
    else:
        # Real sweep: each trial dispatches to remote, results are read from
        # sidecar JSON files. The result_callback EXECUTES the rendered
        # trial script (synchronous bash invocation) BEFORE parsing the
        # sidecar — codex Round 2 finding #2 fix (parse-before-execute bug).
        import subprocess as _sp
        def _callback(script_path: Path, params: dict) -> float:
            # Execute the trial script synchronously. The remote_lane_a_sweep.sh
            # wrapper provides the workspace + GPU env; we just need to run
            # the rendered trial inline so its sidecar .result.json exists
            # before parse_remote_result is called.
            res = _sp.run(["bash", str(script_path)], check=False)
            if res.returncode != 0:
                # Fail loud — Optuna will mark this trial as FAILED
                raise RuntimeError(
                    f"trial script {script_path} exited {res.returncode}; "
                    f"sidecar .result.json may not exist."
                )
            return sweep.parse_remote_result(script_path)

        summary = dispatcher.dispatch_only(_callback)

    # Persist canonical results JSON (consumed by the wrapper script).
    results_path = args.output_dir / "lane_a_sweep_results.json"
    results_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[lane-a-sweep] wrote {results_path}")

    if not args.smoke and summary.get("best_params"):
        opt_script = _emit_optimized_remote_script(
            args.output_dir,
            summary["best_params"],
            summary["best_value"],
        )
        print(f"[lane-a-sweep] wrote optimized remote script: {opt_script}")

    band = summary.get("predicted_band")
    if band and summary.get("best_value") is not None:
        bv = summary["best_value"]
        if bv > band[1]:
            print(f"[lane-a-sweep] WARN: best_value {bv:.4f} ABOVE predicted band {band}")
        elif bv < band[0]:
            print(f"[lane-a-sweep] OK: best_value {bv:.4f} BELOW predicted lower bound — verify")
        else:
            print(f"[lane-a-sweep] OK: best_value {bv:.4f} within predicted band {band}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
