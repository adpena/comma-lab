#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""$0 read-only real-artifact probe for the n=1 costate warm-start cluster."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        tmp = Path(f.name)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from tac.witness_control.costate_requential_curriculum import (
        REQUENTIAL_RECEIPT_SCHEMA,
        walkforward_requential_backtest,
    )
    from tac.witness_control.costate_society_diagnostics import (
        SOCIETY_RECEIPT_SCHEMA,
        diagnose_mechanism_society,
        diagnose_router_ulp_robustness,
    )
    from tac.witness_control.costate_warmstart_cluster import (
        BLOCK_MULTIPLIERS,
        RECEIPT_SCHEMA,
        posterior_solve_mlx_fp32,
        walkforward_backtest,
    )
    from tac.witness_control.lambda_net import backtest, read_trajectory

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve()
    if run_dir == out.parent or run_dir in out.parents:
        raise SystemExit("refusing to write the receipt inside the sacred run directory")
    daemon = run_dir / "daemon.log"
    if not daemon.is_file():
        raise SystemExit(f"missing trajectory log: {daemon}")
    traj = read_trajectory(run_dir)
    baseline_names = (
        "A_ridge_solve", "Q_priormean_iso", "P_priormean_aniso",
        "T_gp_costate_posterior",
    )
    baselines = {}
    for name in baseline_names:
        report, _ = backtest(traj, architecture=name, seed=args.seed)
        baselines[name] = report.to_dict()

    u = walkforward_backtest(traj)
    u_p = walkforward_backtest(traj, prior_modes=("P_priormean_aniso",))
    u_uniform = walkforward_backtest(traj, block_multipliers=(1.0, 1.0, 1.0, 1.0))
    u_clip = walkforward_backtest(traj, difference_clip=True)
    u_closed = walkforward_backtest(traj, aggregate_constraint=True)
    requential_uniform = walkforward_requential_backtest(traj, strategy="uniform")
    requential_disagreement = walkforward_requential_backtest(
        traj, strategy="disagreement")
    society = diagnose_mechanism_society(traj, seed=args.seed)
    router_robustness = diagnose_router_ulp_robustness(traj, seed=args.seed)

    mlx = {"status": "UNAVAILABLE"}
    try:
        # Parity is exercised on the final Q posterior inputs through the focused test;
        # availability is recorded here without pretending MLX is score authority.
        import mlx.core as mx
        mx.eval(mx.array([0.0]))
        mlx = {"status": "AVAILABLE_TESTED_BY_FOCUSED_PARITY",
               "callable": posterior_solve_mlx_fp32.__name__}
    except (ImportError, RuntimeError) as exc:
        mlx = {"status": "UNAVAILABLE_IN_CURRENT_RUNTIME", "reason": str(exc)}

    owned_sources = (
        REPO / "src/tac/witness_control/costate_warmstart_cluster.py",
        REPO / "src/tac/witness_control/costate_requential_curriculum.py",
        REPO / "src/tac/witness_control/costate_society_diagnostics.py",
    )
    module = owned_sources[0]
    deterministic = {
        "schema": RECEIPT_SCHEMA,
        "axis_tag": "[macOS advisory] NON-PROMOTABLE",
        "score_claim": False,
        "pointer_delta": None,
        "seed": args.seed,
        "run_custody": {
            "run_dir": str(run_dir),
            "daemon_log": str(daemon),
            "daemon_sha256": _sha256(daemon),
            "n_verdicts": traj.n_verdicts,
            "n_loss_rows": len(traj.loss_terms),
            "n_levers": len(traj.lever_names),
        },
        "module_sha256": _sha256(module),
        "owned_source_sha256": {
            str(path.relative_to(REPO)): _sha256(path) for path in owned_sources
        },
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "baselines": baselines,
        "U_hierarchical_physics_residual": u.to_dict(),
        "U_P_anisotropic_prior_ablation": u_p.to_dict(),
        "U_uniform_precision_ablation": u_uniform.to_dict(),
        "U_vr_clipped_difference_ablation": u_clip.to_dict(),
        "U_closed_organ_T_persistence_constraint": u_closed.to_dict(),
        "R_requential_uniform_replay": requential_uniform.to_dict(),
        "R_requential_disagreement_replay": requential_disagreement.to_dict(),
        "requential_receipt_schema": REQUENTIAL_RECEIPT_SCHEMA,
        "mechanism_society": society.to_dict(),
        "router_ulp_robustness": router_robustness.to_dict(),
        "society_receipt_schema": SOCIETY_RECEIPT_SCHEMA,
        "mlx_parity_surface": mlx,
        "adoption": {
            "status": "PROVISIONAL_INSTANCE_ONLY",
            "graduation_min_independent_trajectories": 3,
            "synthetic_substitution_allowed": False,
        },
    }
    deterministic_bytes = json.dumps(
        deterministic, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        **deterministic,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "deterministic_result_sha256": hashlib.sha256(deterministic_bytes).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        },
        "preregistered_block_multipliers": list(BLOCK_MULTIPLIERS),
    }
    _atomic_json(out, payload)
    print(json.dumps({
        "out": str(out),
        "u_wf_mae": u.walkforward_mae_model,
        "persistence_wf_mae": u.walkforward_mae_persistence,
        "u_clip_wf_mae": u_clip.walkforward_mae_model,
        "u_closed_wf_mae": u_closed.walkforward_mae_model,
        "requential_uniform_wf_mae": requential_uniform.walkforward_mae_model,
        "requential_disagreement_wf_mae": requential_disagreement.walkforward_mae_model,
        "folds": u.n_folds,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
