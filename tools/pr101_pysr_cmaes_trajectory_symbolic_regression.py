#!/usr/bin/env python3
"""PySR symbolic regression on CMA-ES / Optuna PR101 brotli-param trajectory.

Implements the WIRE verdict from the cmaes-adjacent-libraries hypernerd
subagent (memo
``feedback_cmaes_adjacent_libraries_hypernerd_verdict_20260507.md``):
"PySR symbolic regression on cmaes trajectory data → free closed-form
surrogate (highest expected-info-gain-per-hour item)."

Pipeline:
  1. Ingest CMA-ES + Optuna atom-ledger JSONL trajectories from
     ``experiments/results/{cma_pr101_*,optuna_pr101_*}/``.
  2. Build (q, lgwin, lgblock) -> bytes_out matrix; deduplicate identical
     param triplets (keep min-bytes representative); 80/20 train/test split.
  3. Run PySR with model_selection="best", binary {+,-,*,/},
     unary {log, exp, sqrt}, deterministic seed.
  4. Validate held-out R^2 + RMSE.
  5. Cross-check the symbolic optimum (gradient descent on the closed-form
     surrogate over the integer grid) against the empirical best.
  6. Emit a forensic report JSON + an atom-ledger JSONL stamped with the
     CPU-prep PySR-symbolic-regression evidence grade.

CLAUDE.md compliance:
  - Pure CPU (PySR runs Julia SymbolicRegression.jl in-process); no GPU,
    no scorer load, no archive mutation.
  - Per ``forbidden_premature_class_level_falsification``: any negative
    surrogate-fit result tagged MEASURED_CONFIG_NOT_DISPATCHABLE; a
    surrogate-fit failure does NOT falsify cmaes/PySR class.
  - Per ``forbidden_CPU_MPS_derived_dispatch_readiness_flag``:
    ready_for_exact_eval_dispatch=False on every emitted row.
  - score_claim=False, family_falsified=False,
    falsification_scope="symbolic_surrogate_fit_only_no_score_test".
  - evidence_grade="[CPU-prep faithful PySR-symbolic-regression test]".

Usage::

    .venv/bin/python tools/pr101_pysr_cmaes_trajectory_symbolic_regression.py \
        --output-dir experiments/results/pr101_pysr_symbolic_regression_<TS> \
        --niterations 50 --population-size 30 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "tools/pr101_pysr_cmaes_trajectory_symbolic_regression.py"
REPORT_SCHEMA = "pr101_pysr_cmaes_symbolic_regression_report_v1"
LEDGER_ROW_SCHEMA = "pr101_pysr_cmaes_symbolic_regression_eval_v1"
EVIDENCE_GRADE = "[CPU-prep faithful PySR-symbolic-regression test]"
EVIDENCE_SEMANTICS = "cpu_pysr_symbolic_regression_forensic"
TARGET_MODES = ["contest_exact_eval_planning"]
DEPLOYMENT_TARGET = "desktop_research"
DISPATCH_BLOCKERS = [
    "cpu_only_symbolic_regression",
    "no_archive_substitution_performed",
    "no_score_affecting_payload_change_proof",
    "missing_byte_closed_archive_manifest",
    "missing_exact_cuda_auth_eval",
    "symbolic_surrogate_is_planning_signal_only",
]
FALSIFICATION_SCOPE = "symbolic_surrogate_fit_only_no_score_test"

# Hypernerd-verdict-cited trajectory roots. These already exist on disk
# from CMA-ES + Optuna sweeps that landed 2026-05-07.
DEFAULT_TRAJECTORY_GLOBS = [
    "experiments/results/cma_pr101_real_substrate_*/cma_pr101_atom_ledger.jsonl",
    "experiments/results/optuna_pr101_*/optuna_atom_ledger.jsonl",
]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Trajectory:
    """A single (q, lgwin, lgblock, bytes_out) trial."""

    q: int
    lgwin: int
    lgblock: int
    bytes_out: int
    source_file: str

    def key(self) -> tuple[int, int, int]:
        return (self.q, self.lgwin, self.lgblock)


@dataclass
class SurrogateFit:
    """A single PySR-fit equation row."""

    equation: str
    complexity: int
    loss: float
    score: float
    sympy_format: str | None = None


@dataclass
class FitReport:
    schema: str
    tool: str
    timestamp_utc: str
    n_trajectories_total: int
    n_trajectories_unique: int
    n_train: int
    n_test: int
    train_r2: float
    test_r2: float
    test_rmse: float
    test_mae: float
    best_equation: str
    best_complexity: int
    pareto_front: list[SurrogateFit]
    empirical_best: dict[str, Any]
    surrogate_optimum: dict[str, Any]
    pysr_config: dict[str, Any]
    sources: list[str]
    evidence_grade: str = EVIDENCE_GRADE
    evidence_semantics: str = EVIDENCE_SEMANTICS
    target_modes: list[str] = field(default_factory=lambda: list(TARGET_MODES))
    deployment_target: str = DEPLOYMENT_TARGET
    dispatch_blockers: list[str] = field(
        default_factory=lambda: list(DISPATCH_BLOCKERS)
    )
    score_claim: bool = False
    family_falsified: bool = False
    falsification_scope: str = FALSIFICATION_SCOPE
    ready_for_exact_eval_dispatch: bool = False
    field_selection_ready_for_exact_eval_dispatch: bool = False
    charged_bits_changed: bool = False
    score_affecting_payload_changed: bool = False
    dispatch_attempted: bool = False
    dispatchable: bool = False
    notes: str = (
        "PySR symbolic regression on CMA-ES + Optuna PR101 brotli-param "
        "trajectory data; CPU-only forensic planning; surrogate is a "
        "closed-form prediction of bytes_out(q, lgwin, lgblock) on the "
        "PR101 substrate. Surrogate-derived optima MUST be validated "
        "against the actual brotli encoder before any dispatch."
    )


def discover_trajectory_files(
    repo_root: Path, globs: list[str]
) -> list[Path]:
    files: list[Path] = []
    for g in globs:
        for p in sorted(repo_root.glob(g)):
            if p.is_file():
                files.append(p)
    return files


def parse_trajectory_jsonl(path: Path) -> list[Trajectory]:
    out: list[Trajectory] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = row.get("op_params") or {}
            q = params.get("brotli_quality")
            lgwin = params.get("brotli_lgwin")
            lgblock = params.get("brotli_lgblock")
            bytes_out = row.get("bytes_out")
            if q is None or lgwin is None or lgblock is None or bytes_out is None:
                continue
            try:
                out.append(
                    Trajectory(
                        q=int(q),
                        lgwin=int(lgwin),
                        lgblock=int(lgblock),
                        bytes_out=int(bytes_out),
                        source_file=str(path.relative_to(REPO_ROOT)),
                    )
                )
            except (TypeError, ValueError):
                continue
    return out


def deduplicate_trajectories(
    trajectories: list[Trajectory],
) -> list[Trajectory]:
    """Keep the min-bytes representative per (q, lgwin, lgblock) triplet.

    Multiple sweeps may have evaluated the same triplet; brotli is
    deterministic given parameters so they should agree, but if not the
    minimum is the achievable bytes (no random noise in brotli).
    """
    best: dict[tuple[int, int, int], Trajectory] = {}
    for t in trajectories:
        k = t.key()
        if k not in best or t.bytes_out < best[k].bytes_out:
            best[k] = t
    return sorted(best.values(), key=lambda t: t.key())


def build_xy(
    trajectories: list[Trajectory],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X and target y for PySR.

    Features: q, lgwin, lgblock (all integer-domain).
    Target: bytes_out.
    """
    X = np.asarray(
        [[float(t.q), float(t.lgwin), float(t.lgblock)] for t in trajectories],
        dtype=np.float64,
    )
    y = np.asarray([float(t.bytes_out) for t in trajectories], dtype=np.float64)
    feature_names = ["q", "lgwin", "lgblock"]
    return X, y, feature_names


def train_test_split(
    X: np.ndarray, y: np.ndarray, train_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_train = max(1, int(round(train_frac * n)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan") if ss_res > 1e-12 else 1.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    niterations: int,
    population_size: int,
    seed: int,
    procs: int,
) -> tuple[Any, list[SurrogateFit]]:
    """Fit PySR; return the model + Pareto front."""
    # Lazy import — keep top-level import light.
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        population_size=population_size,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt"],
        model_selection="best",
        verbosity=0,
        progress=False,
        random_state=seed,
        deterministic=True,
        parallelism="serial" if procs <= 1 else "multiprocessing",
        procs=procs,
        # Bias toward simple expressions; bytes_out is integer-domain
        # so even fairly simple polynomials suffice for a surrogate.
        maxsize=25,
        warm_start=False,
        temp_equation_file=False,
    )
    model.fit(X_train, y_train, variable_names=feature_names)
    eqns = model.equations_
    pareto: list[SurrogateFit] = []
    for _, row in eqns.iterrows():
        eq_str = str(row["equation"])
        try:
            sympy_str = str(row.get("sympy_format", ""))
        except Exception:
            sympy_str = None
        pareto.append(
            SurrogateFit(
                equation=eq_str,
                complexity=int(row["complexity"]),
                loss=float(row["loss"]),
                score=float(row["score"]),
                sympy_format=sympy_str,
            )
        )
    return model, pareto


def find_surrogate_optimum_grid(
    model: Any,
    q_range: tuple[int, int],
    lgwin_range: tuple[int, int],
    lgblock_range: tuple[int, int],
) -> dict[str, Any]:
    """Sweep the integer grid of (q, lgwin, lgblock); return surrogate-min."""
    triplets: list[tuple[int, int, int]] = []
    for q in range(q_range[0], q_range[1] + 1):
        for lgw in range(lgwin_range[0], lgwin_range[1] + 1):
            for lgb in range(lgblock_range[0], lgblock_range[1] + 1):
                triplets.append((q, lgw, lgb))
    X = np.asarray(triplets, dtype=np.float64)
    yhat = model.predict(X)
    best_idx = int(np.argmin(yhat))
    return {
        "q": int(triplets[best_idx][0]),
        "lgwin": int(triplets[best_idx][1]),
        "lgblock": int(triplets[best_idx][2]),
        "predicted_bytes_out": float(yhat[best_idx]),
        "n_grid_points": len(triplets),
    }


def write_atom_ledger(
    ledger_path: Path,
    fit_report: FitReport,
) -> None:
    """Append one planning row per Pareto-front equation."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, fit in enumerate(fit_report.pareto_front):
        atom_id = f"pr101_pysr_symbolic/eq_{idx:02d}"
        row = {
            "schema": LEDGER_ROW_SCHEMA,
            "atom_id": atom_id,
            "tool": TOOL_NAME,
            "timestamp_utc": fit_report.timestamp_utc,
            "cathedral_op": "tac.codec_pipeline.Op1_PR101SplitBrotli",
            "family": "symbolic_regression_surrogate",
            "family_group": (
                "symbolic_regression_surrogate:Op1_PR101SplitBrotli"
            ),
            "pareto_scope": "tac.codec_pipeline.Op1_PR101SplitBrotli",
            "pareto_frontier": True,  # All emitted rows are on the front.
            "complexity": fit.complexity,
            "loss": fit.loss,
            "score": fit.score,  # PySR's per-equation score, NOT contest score.
            "equation": fit.equation,
            "sympy_format": fit.sympy_format,
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_semantics": EVIDENCE_SEMANTICS,
            "deployment_target": DEPLOYMENT_TARGET,
            "target_modes": list(TARGET_MODES),
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
            "ready_for_exact_eval_dispatch": False,
            "field_selection_ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "family_falsified": False,
            "falsification_scope": FALSIFICATION_SCOPE,
            "charged_bits_changed": False,
            "score_affecting_payload_changed": False,
            "dispatch_attempted": False,
            "dispatchable": False,
            "test_r2": fit_report.test_r2,
            "test_rmse": fit_report.test_rmse,
            "n_trajectories_unique": fit_report.n_trajectories_unique,
            "notes": (
                "PySR symbolic-regression Pareto-front equation; CPU-only "
                "forensic planning row. The 'score' here is PySR's "
                "complexity-vs-loss tradeoff metric, NOT a contest score."
            ),
        }
        rows.append(row)
    with ledger_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_report_json(
    report_path: Path,
    fit_report: FitReport,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": fit_report.schema,
        "tool": fit_report.tool,
        "timestamp_utc": fit_report.timestamp_utc,
        "evidence_grade": fit_report.evidence_grade,
        "evidence_semantics": fit_report.evidence_semantics,
        "target_modes": fit_report.target_modes,
        "deployment_target": fit_report.deployment_target,
        "dispatch_blockers": fit_report.dispatch_blockers,
        "score_claim": fit_report.score_claim,
        "family_falsified": fit_report.family_falsified,
        "falsification_scope": fit_report.falsification_scope,
        "ready_for_exact_eval_dispatch": (
            fit_report.ready_for_exact_eval_dispatch
        ),
        "field_selection_ready_for_exact_eval_dispatch": (
            fit_report.field_selection_ready_for_exact_eval_dispatch
        ),
        "charged_bits_changed": fit_report.charged_bits_changed,
        "score_affecting_payload_changed": (
            fit_report.score_affecting_payload_changed
        ),
        "dispatch_attempted": fit_report.dispatch_attempted,
        "dispatchable": fit_report.dispatchable,
        "trajectory_sources": fit_report.sources,
        "n_trajectories_total": fit_report.n_trajectories_total,
        "n_trajectories_unique": fit_report.n_trajectories_unique,
        "n_train": fit_report.n_train,
        "n_test": fit_report.n_test,
        "metrics": {
            "train_r2": fit_report.train_r2,
            "test_r2": fit_report.test_r2,
            "test_rmse": fit_report.test_rmse,
            "test_mae": fit_report.test_mae,
        },
        "best_equation": fit_report.best_equation,
        "best_complexity": fit_report.best_complexity,
        "pareto_front": [
            {
                "complexity": f.complexity,
                "loss": f.loss,
                "score": f.score,
                "equation": f.equation,
                "sympy_format": f.sympy_format,
            }
            for f in fit_report.pareto_front
        ],
        "empirical_best": fit_report.empirical_best,
        "surrogate_optimum": fit_report.surrogate_optimum,
        "pysr_config": fit_report.pysr_config,
        "notes": fit_report.notes,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PySR symbolic regression on PR101 brotli-param "
            "CMA-ES/Optuna trajectory data."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory for fit report + atom ledger; created if "
            "missing. Use experiments/results/pr101_pysr_*_<TS>."
        ),
    )
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=None,
        help=(
            "Glob (relative to repo root) for trajectory JSONL files; "
            "may be repeated. Defaults to CMA-ES + Optuna PR101 sweeps."
        ),
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=50,
        help="PySR niterations (default 50).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="PySR population_size (default 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default 0).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train/test split fraction (default 0.8).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=1,
        help=(
            "PySR processes (default 1 = serial; >1 enables Julia "
            "multiprocessing)."
        ),
    )
    parser.add_argument(
        "--q-range",
        type=int,
        nargs=2,
        default=(1, 11),
        help="Search range for brotli quality (default 1 11).",
    )
    parser.add_argument(
        "--lgwin-range",
        type=int,
        nargs=2,
        default=(10, 24),
        help="Search range for brotli lgwin (default 10 24).",
    )
    parser.add_argument(
        "--lgblock-range",
        type=int,
        nargs=2,
        default=(16, 24),
        help="Search range for brotli lgblock (default 16 24).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir: Path = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    globs = args.trajectory_glob or list(DEFAULT_TRAJECTORY_GLOBS)
    sources = discover_trajectory_files(REPO_ROOT, globs)
    if not sources:
        print(
            "ERROR: no trajectory JSONL files matched globs:",
            globs,
            file=sys.stderr,
        )
        return 2

    all_trajectories: list[Trajectory] = []
    for src in sources:
        all_trajectories.extend(parse_trajectory_jsonl(src))

    if len(all_trajectories) < 12:
        print(
            f"ERROR: only {len(all_trajectories)} trajectory rows; need >= "
            "12 for an 80/20 split with PySR.",
            file=sys.stderr,
        )
        return 3

    unique = deduplicate_trajectories(all_trajectories)
    X, y, feature_names = build_xy(unique)
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, args.train_frac, args.seed
    )

    print(
        f"[pr101_pysr] sources={len(sources)} "
        f"total_trajectories={len(all_trajectories)} "
        f"unique={len(unique)} "
        f"train={X_train.shape[0]} test={X_test.shape[0]}"
    )
    t0 = time.time()
    model, pareto = fit_pysr(
        X_train,
        y_train,
        feature_names,
        niterations=args.niterations,
        population_size=args.population_size,
        seed=args.seed,
        procs=args.procs,
    )
    elapsed = time.time() - t0
    print(f"[pr101_pysr] PySR fit elapsed={elapsed:.1f}s")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test) if X_test.shape[0] > 0 else np.array([])
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred) if X_test.shape[0] > 0 else float("nan")
    test_rmse = rmse(y_test, y_test_pred) if X_test.shape[0] > 0 else float("nan")
    test_mae = mae(y_test, y_test_pred) if X_test.shape[0] > 0 else float("nan")

    best = model.get_best()
    best_equation = str(best["equation"])
    best_complexity = int(best["complexity"])

    # Empirical best
    emp_best = min(unique, key=lambda t: t.bytes_out)
    empirical_best = {
        "q": emp_best.q,
        "lgwin": emp_best.lgwin,
        "lgblock": emp_best.lgblock,
        "bytes_out": emp_best.bytes_out,
        "source_file": emp_best.source_file,
    }

    surrogate_optimum = find_surrogate_optimum_grid(
        model,
        q_range=tuple(args.q_range),
        lgwin_range=tuple(args.lgwin_range),
        lgblock_range=tuple(args.lgblock_range),
    )

    fit_report = FitReport(
        schema=REPORT_SCHEMA,
        tool=TOOL_NAME,
        timestamp_utc=_utc_now(),
        n_trajectories_total=len(all_trajectories),
        n_trajectories_unique=len(unique),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        train_r2=train_r2,
        test_r2=test_r2,
        test_rmse=test_rmse,
        test_mae=test_mae,
        best_equation=best_equation,
        best_complexity=best_complexity,
        pareto_front=pareto,
        empirical_best=empirical_best,
        surrogate_optimum=surrogate_optimum,
        pysr_config={
            "niterations": args.niterations,
            "population_size": args.population_size,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "procs": args.procs,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["log", "exp", "sqrt"],
            "model_selection": "best",
            "deterministic": True,
            "maxsize": 25,
        },
        sources=[str(p.relative_to(REPO_ROOT)) for p in sources],
    )

    report_path = output_dir / "pr101_pysr_symbolic_regression_report.json"
    ledger_path = output_dir / "pr101_pysr_atom_ledger.jsonl"
    write_report_json(report_path, fit_report)
    write_atom_ledger(ledger_path, fit_report)

    print(
        f"[pr101_pysr] report -> {report_path.relative_to(REPO_ROOT)}\n"
        f"[pr101_pysr] ledger -> {ledger_path.relative_to(REPO_ROOT)}\n"
        f"[pr101_pysr] best_equation: {best_equation}\n"
        f"[pr101_pysr] complexity={best_complexity} "
        f"train_r2={train_r2:.4f} test_r2={test_r2:.4f} "
        f"test_rmse={test_rmse:.2f} test_mae={test_mae:.2f}\n"
        f"[pr101_pysr] empirical_best: q={empirical_best['q']} "
        f"lgwin={empirical_best['lgwin']} lgblock={empirical_best['lgblock']} "
        f"-> {empirical_best['bytes_out']} B\n"
        f"[pr101_pysr] surrogate_optimum: q={surrogate_optimum['q']} "
        f"lgwin={surrogate_optimum['lgwin']} "
        f"lgblock={surrogate_optimum['lgblock']} -> "
        f"{surrogate_optimum['predicted_bytes_out']:.1f} B (predicted)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
