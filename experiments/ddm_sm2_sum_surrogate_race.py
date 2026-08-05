"""SM2 scorer-free entropy+SMEVR SUM surrogate race on live token streams.

This tool closes the ddm_rg5 residual that the two-arm
``entropy`` vs ``smevr_surrogate`` race omitted the third arm:
an affine SUM surrogate using both marginal entropy and the temporal-delta
SMEVR proxy.  It never calls the frozen scorer or ``upstream/evaluate.py``;
the authority surface is the real lossless R7 SMEVR coder bytes on already
materialized token streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ddm_r7_token_coder import encode_token_codes  # noqa: E402


DEFAULT_TOKEN_NPY = (
    Path("/Volumes/VertigoDataTier/pact/ddm_sb1_20260804")
    / "B_rt1_subfinal_tokens"
    / "sub_final_tokens.npy"
)
DEFAULT_RG5_ROWS = REPO_ROOT / ".omx" / "research" / "ddm_rg5_rows_20260801.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / ".omx" / "research" / "ddm_sm2_20260805"


@dataclass(frozen=True)
class FitResult:
    name: str
    target: str
    predictors: tuple[str, ...]
    intercept: float
    coefficients: dict[str, float]
    n_rows: int
    rmse: float
    mae: float
    max_abs_error: float
    r2: float | None

    def predict_one(self, row: Mapping[str, Any]) -> float:
        value = self.intercept
        for predictor in self.predictors:
            value += self.coefficients[predictor] * float(row[predictor])
        return float(value)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "predictors": list(self.predictors),
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "n_rows": self.n_rows,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "r2": self.r2,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _as_numeric_matrix(
    rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    usable: list[Mapping[str, Any]] = []
    for row in rows:
        if target not in row:
            continue
        if all(predictor in row for predictor in predictors):
            usable.append(row)
    if not usable:
        raise ValueError(f"no usable rows for target={target} predictors={predictors}")
    x = np.ones((len(usable), len(predictors) + 1), dtype=np.float64)
    for col, predictor in enumerate(predictors, start=1):
        x[:, col] = [float(row[predictor]) for row in usable]
    y = np.asarray([float(row[target]) for row in usable], dtype=np.float64)
    return x, y


def fit_affine(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    predictors: Sequence[str],
    target: str,
) -> FitResult:
    x, y = _as_numeric_matrix(rows, predictors, target)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = None if ss_tot <= 0.0 else float(1.0 - float(np.sum(residual**2)) / ss_tot)
    return FitResult(
        name=name,
        target=target,
        predictors=tuple(predictors),
        intercept=float(beta[0]),
        coefficients={predictor: float(beta[i + 1]) for i, predictor in enumerate(predictors)},
        n_rows=int(len(y)),
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs,
        r2=r2,
    )


def quadratic_delta_fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expanded: list[dict[str, float]] = []
    for row in rows:
        if not {"d_surr_entropy", "d_surr_smevr", "d_bytes"} <= set(row):
            continue
        e = float(row["d_surr_entropy"])
        s = float(row["d_surr_smevr"])
        expanded.append(
            {
                "d_surr_entropy": e,
                "d_surr_smevr": s,
                "d_surr_entropy_sq": e * e,
                "d_surr_smevr_sq": s * s,
                "d_surr_cross": e * s,
                "d_bytes": float(row["d_bytes"]),
            }
        )
    if not expanded:
        raise ValueError("no usable delta rows for quadratic fit")
    linear = fit_affine(
        "delta_sum_affine",
        expanded,
        predictors=("d_surr_entropy", "d_surr_smevr"),
        target="d_bytes",
    )
    quadratic = fit_affine(
        "delta_sum_quadratic",
        expanded,
        predictors=(
            "d_surr_entropy",
            "d_surr_smevr",
            "d_surr_entropy_sq",
            "d_surr_smevr_sq",
            "d_surr_cross",
        ),
        target="d_bytes",
    )
    ratio = quadratic.rmse / linear.rmse if linear.rmse > 0 else math.nan
    return {
        "linear": linear.to_json(),
        "quadratic": quadratic.to_json(),
        "quadratic_rmse_over_linear_rmse": float(ratio),
        "quadratic_rmse_reduction_fraction": float(1.0 - ratio),
    }


def soft_hist_entropy_bits_np(vals: np.ndarray, levels: int, temp: float = 0.15) -> float:
    """NumPy equivalent of train_tr1_partition_renderer_mlx._soft_hist_entropy_bits."""

    flat = np.asarray(vals, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    scale = float(levels - 1)
    x01 = np.clip((flat + 1.0) * 0.5, 0.0, 1.0) * scale
    centers = np.arange(levels, dtype=np.float64)
    logits = -((x01[:, None] - centers[None, :]) ** 2) / max(float(temp), 1e-6)
    logits -= np.max(logits, axis=1, keepdims=True)
    soft = np.exp(logits)
    soft /= np.sum(soft, axis=1, keepdims=True)
    p = np.mean(soft, axis=0) + 1e-12
    return float(-np.sum(p * np.log2(p)))


def token_surrogates(codes: np.ndarray, *, levels: int, temp: float = 0.15) -> dict[str, float]:
    q = np.asarray(codes)
    if q.ndim < 2:
        raise ValueError(f"codes must include a pair/frame axis, got shape {q.shape}")
    values = q.astype(np.float64) / float(levels - 1) * 2.0 - 1.0
    entropy = soft_hist_entropy_bits_np(values, levels=levels, temp=temp)
    if values.shape[0] < 2:
        smevr = 0.0
    else:
        deltas = 0.5 * (values[1:] - values[:-1])
        smevr = soft_hist_entropy_bits_np(deltas, levels=levels, temp=temp)
    return {
        "surr_entropy_bits": float(entropy),
        "surr_smevr_surrogate_bits": float(smevr),
    }


def smevr_bytes(codes: np.ndarray, *, levels: int, codec: str) -> int:
    return len(encode_token_codes(np.asarray(codes, dtype=np.uint8), levels=levels, codec=codec))


def base_rate_row(
    codes: np.ndarray,
    *,
    levels: int,
    codec: str,
    temp: float,
    row_id: str,
    probe: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_id": row_id,
        "probe": probe,
        "token_shape": list(np.asarray(codes).shape),
        "codec": codec,
        "levels": int(levels),
        "smevr_bytes": int(smevr_bytes(codes, levels=levels, codec=codec)),
    }
    row.update(token_surrogates(codes, levels=levels, temp=temp))
    return row


def full_transform_rows(
    codes: np.ndarray,
    *,
    levels: int,
    codec: str,
    temp: float,
    random_perms: int,
    seed: int,
) -> list[dict[str, Any]]:
    q = np.asarray(codes, dtype=np.uint8)
    transforms: list[tuple[str, np.ndarray]] = [("identity", np.arange(q.shape[0]))]
    transforms.append(("reverse_pairs", np.arange(q.shape[0] - 1, -1, -1)))
    for shift in (1, 7, 60, 123):
        transforms.append((f"roll_pairs_{shift}", np.roll(np.arange(q.shape[0]), int(shift))))
    rng = np.random.default_rng(seed)
    for idx in range(random_perms):
        transforms.append((f"random_perm_seed_{seed}_{idx}", rng.permutation(q.shape[0])))

    rows: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for row_id, order in transforms:
        row = base_rate_row(
            q[order],
            levels=levels,
            codec=codec,
            temp=temp,
            row_id=row_id,
            probe="live_full_pair_permutation",
        )
        if base is None:
            base = dict(row)
        row["d_bytes"] = int(row["smevr_bytes"]) - int(base["smevr_bytes"])
        row["d_surr_entropy"] = (
            float(row["surr_entropy_bits"]) - float(base["surr_entropy_bits"])
        )
        row["d_surr_smevr"] = (
            float(row["surr_smevr_surrogate_bits"])
            - float(base["surr_smevr_surrogate_bits"])
        )
        rows.append(row)
    return rows


def cell_rows(
    codes: np.ndarray,
    *,
    levels: int,
    codec: str,
    temp: float,
    cell_limit: int | None = None,
) -> list[dict[str, Any]]:
    q = np.asarray(codes, dtype=np.uint8)
    if q.ndim != 4:
        raise ValueError(f"expected token shape [pairs, rows, cols, channels], got {q.shape}")
    rows: list[dict[str, Any]] = []
    count = 0
    for rr in range(q.shape[1]):
        for cc in range(q.shape[2]):
            sub = q[:, rr : rr + 1, cc : cc + 1, :]
            row = base_rate_row(
                sub,
                levels=levels,
                codec=codec,
                temp=temp,
                row_id=f"cell_r{rr:02d}_c{cc:02d}",
                probe="live_per_cell",
            )
            row["cell_row"] = int(rr)
            row["cell_col"] = int(cc)
            rows.append(row)
            count += 1
            if cell_limit is not None and count >= cell_limit:
                return rows
    return rows


def tile_rows(
    codes: np.ndarray,
    *,
    levels: int,
    codec: str,
    temp: float,
    tile_h: int,
    tile_w: int,
) -> list[dict[str, Any]]:
    q = np.asarray(codes, dtype=np.uint8)
    if q.ndim != 4:
        raise ValueError(f"expected token shape [pairs, rows, cols, channels], got {q.shape}")
    rows: list[dict[str, Any]] = []
    for r0 in range(0, q.shape[1], tile_h):
        for c0 in range(0, q.shape[2], tile_w):
            r1 = min(r0 + tile_h, q.shape[1])
            c1 = min(c0 + tile_w, q.shape[2])
            sub = q[:, r0:r1, c0:c1, :]
            row = base_rate_row(
                sub,
                levels=levels,
                codec=codec,
                temp=temp,
                row_id=f"tile_r{r0:02d}_{r1:02d}_c{c0:02d}_{c1:02d}",
                probe="live_spatial_tile",
            )
            row["tile_r0"] = int(r0)
            row["tile_r1"] = int(r1)
            row["tile_c0"] = int(c0)
            row["tile_c1"] = int(c1)
            rows.append(row)
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_metrics(pred: Sequence[float], actual: Sequence[float]) -> dict[str, float | None]:
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    pearson = _pearson(x, y)
    spearman = _pearson(_rankdata(x), _rankdata(y))
    return {"pearson": pearson, "spearman": spearman}


def score_model_on_rows(
    fit: FitResult,
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    usable = [row for row in rows if target in row and all(p in row for p in fit.predictors)]
    pred = [fit.predict_one(row) for row in usable]
    actual = [float(row[target]) for row in usable]
    residual = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    metrics = rank_metrics(pred, actual)
    metrics.update(
        {
            "n_rows": len(usable),
            "rmse": float(np.sqrt(np.mean(residual**2))) if usable else None,
            "mae": float(np.mean(np.abs(residual))) if usable else None,
            "max_abs_error": float(np.max(np.abs(residual))) if usable else None,
        }
    )
    return metrics


def fixed_sum_rows(rows: Sequence[Mapping[str, Any]], *, delta: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if delta:
        for row in rows:
            if "d_surr_entropy" not in row or "d_surr_smevr" not in row:
                continue
            copy = dict(row)
            copy["d_surr_fixed_sum"] = float(row["d_surr_entropy"]) + float(row["d_surr_smevr"])
            out.append(copy)
    else:
        for row in rows:
            copy = dict(row)
            copy["surr_fixed_sum_bits"] = (
                float(row["surr_entropy_bits"]) + float(row["surr_smevr_surrogate_bits"])
            )
            out.append(copy)
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]], target: str) -> dict[str, Any]:
    values = np.asarray([float(row[target]) for row in rows if target in row], dtype=np.float64)
    if values.size == 0:
        return {"n_rows": 0}
    return {
        "n_rows": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def build_result(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token_path = Path(args.tokens)
    rg5_path = Path(args.rg5_rows)
    q = np.load(token_path)
    if q.dtype != np.uint8:
        q = q.astype(np.uint8)

    rg5_rows = load_jsonl(rg5_path)
    delta_rows = [
        row
        for row in rg5_rows
        if {"d_bytes", "d_surr_entropy", "d_surr_smevr"} <= set(row)
    ]
    absolute_rows = [
        row
        for row in rg5_rows
        if {"smevr_bytes", "surr_entropy_bits", "surr_smevr_surrogate_bits"} <= set(row)
    ]
    if not delta_rows:
        raise ValueError("RG5 row file contains no byte-delta rows")
    if not absolute_rows:
        raise ValueError("RG5 row file contains no absolute-byte rows")

    delta_rows_sum = fixed_sum_rows(delta_rows, delta=True)
    absolute_rows_sum = fixed_sum_rows(absolute_rows, delta=False)
    fits = {
        "delta_entropy": fit_affine(
            "delta_entropy",
            delta_rows,
            predictors=("d_surr_entropy",),
            target="d_bytes",
        ),
        "delta_smevr": fit_affine(
            "delta_smevr",
            delta_rows,
            predictors=("d_surr_smevr",),
            target="d_bytes",
        ),
        "delta_fixed_sum": fit_affine(
            "delta_fixed_sum",
            delta_rows_sum,
            predictors=("d_surr_fixed_sum",),
            target="d_bytes",
        ),
        "delta_sum_affine": fit_affine(
            "delta_sum_affine",
            delta_rows,
            predictors=("d_surr_entropy", "d_surr_smevr"),
            target="d_bytes",
        ),
        "absolute_entropy": fit_affine(
            "absolute_entropy",
            absolute_rows,
            predictors=("surr_entropy_bits",),
            target="smevr_bytes",
        ),
        "absolute_smevr": fit_affine(
            "absolute_smevr",
            absolute_rows,
            predictors=("surr_smevr_surrogate_bits",),
            target="smevr_bytes",
        ),
        "absolute_fixed_sum": fit_affine(
            "absolute_fixed_sum",
            absolute_rows_sum,
            predictors=("surr_fixed_sum_bits",),
            target="smevr_bytes",
        ),
        "absolute_sum_affine": fit_affine(
            "absolute_sum_affine",
            absolute_rows,
            predictors=("surr_entropy_bits", "surr_smevr_surrogate_bits"),
            target="smevr_bytes",
        ),
    }

    live_full = full_transform_rows(
        q,
        levels=args.levels,
        codec=args.codec,
        temp=args.temp,
        random_perms=args.random_perms,
        seed=args.seed,
    )
    live_cells = cell_rows(
        q,
        levels=args.levels,
        codec=args.codec,
        temp=args.temp,
        cell_limit=args.cell_limit,
    )
    live_tiles = tile_rows(
        q,
        levels=args.levels,
        codec=args.codec,
        temp=args.temp,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
    )
    live_cells_sum = fixed_sum_rows(live_cells, delta=False)
    live_tiles_sum = fixed_sum_rows(live_tiles, delta=False)
    live_full_sum = fixed_sum_rows(live_full, delta=True)

    fit_json = {name: fit.to_json() for name, fit in fits.items()}
    quadratic = quadratic_delta_fit(delta_rows)
    live_metrics = {
        "full_pair_permutation_deltas": {
            "delta_entropy": score_model_on_rows(fits["delta_entropy"], live_full, target="d_bytes"),
            "delta_smevr": score_model_on_rows(fits["delta_smevr"], live_full, target="d_bytes"),
            "delta_fixed_sum": score_model_on_rows(
                fits["delta_fixed_sum"], live_full_sum, target="d_bytes"
            ),
            "delta_sum_affine": score_model_on_rows(
                fits["delta_sum_affine"], live_full, target="d_bytes"
            ),
        },
        "per_cell_absolute_bytes": {
            "absolute_entropy": score_model_on_rows(
                fits["absolute_entropy"], live_cells, target="smevr_bytes"
            ),
            "absolute_smevr": score_model_on_rows(
                fits["absolute_smevr"], live_cells, target="smevr_bytes"
            ),
            "absolute_fixed_sum": score_model_on_rows(
                fits["absolute_fixed_sum"], live_cells_sum, target="smevr_bytes"
            ),
            "absolute_sum_affine": score_model_on_rows(
                fits["absolute_sum_affine"], live_cells, target="smevr_bytes"
            ),
        },
        "spatial_tile_absolute_bytes": {
            "absolute_entropy": score_model_on_rows(
                fits["absolute_entropy"], live_tiles, target="smevr_bytes"
            ),
            "absolute_smevr": score_model_on_rows(
                fits["absolute_smevr"], live_tiles, target="smevr_bytes"
            ),
            "absolute_fixed_sum": score_model_on_rows(
                fits["absolute_fixed_sum"], live_tiles_sum, target="smevr_bytes"
            ),
            "absolute_sum_affine": score_model_on_rows(
                fits["absolute_sum_affine"], live_tiles, target="smevr_bytes"
            ),
        },
    }

    all_rows: list[dict[str, Any]] = []
    for row in live_full:
        copy = dict(row)
        copy["pred_delta_entropy_bytes"] = fits["delta_entropy"].predict_one(copy)
        copy["pred_delta_smevr_bytes"] = fits["delta_smevr"].predict_one(copy)
        copy["pred_delta_sum_affine_bytes"] = fits["delta_sum_affine"].predict_one(copy)
        all_rows.append(copy)
    for source_rows, source_rows_sum, probe_name in (
        (live_cells, live_cells_sum, "live_per_cell"),
        (live_tiles, live_tiles_sum, "live_spatial_tile"),
    ):
        sum_by_id = {row["row_id"]: row for row in source_rows_sum}
        for row in source_rows:
            copy = dict(row)
            copy["pred_entropy_bytes"] = fits["absolute_entropy"].predict_one(copy)
            copy["pred_smevr_bytes"] = fits["absolute_smevr"].predict_one(copy)
            copy["pred_sum_affine_bytes"] = fits["absolute_sum_affine"].predict_one(copy)
            fixed_row = sum_by_id[copy["row_id"]]
            copy["surr_fixed_sum_bits"] = fixed_row["surr_fixed_sum_bits"]
            copy["pred_fixed_sum_bytes"] = fits["absolute_fixed_sum"].predict_one(fixed_row)
            copy["probe"] = probe_name
            all_rows.append(copy)

    pair_perm_deltas = [row for row in live_full if row["row_id"] != "identity"]
    entropy_blind_abs = [
        abs(float(row["d_surr_entropy"]))
        for row in pair_perm_deltas
    ]
    result = {
        "schema": "ddm_sm2_sum_surrogate_race.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "score_axis": "none",
            "scorer_forwards": 0,
            "upstream_evaluate_calls": 0,
            "archive_mutations": 0,
            "byte_authority": "lossless R7 encode_token_codes(codec=smevr)",
        },
        "inputs": {
            "token_npy": str(token_path),
            "token_npy_sha256": sha256_file(token_path),
            "token_shape": list(q.shape),
            "token_dtype": str(q.dtype),
            "token_min": int(np.min(q)),
            "token_max": int(np.max(q)),
            "rg5_rows": str(rg5_path),
            "rg5_rows_sha256": sha256_file(rg5_path),
        },
        "config": {
            "levels": int(args.levels),
            "codec": args.codec,
            "temp": float(args.temp),
            "random_perms": int(args.random_perms),
            "seed": int(args.seed),
            "tile_h": int(args.tile_h),
            "tile_w": int(args.tile_w),
            "cell_limit": args.cell_limit,
        },
        "fit": {
            "rg5_delta_rows": len(delta_rows),
            "rg5_absolute_rows": len(absolute_rows),
            "models": fit_json,
            "quadratic_check": quadratic,
        },
        "live": {
            "base_full_smevr_bytes": int(live_full[0]["smevr_bytes"]),
            "base_full_entropy_bits": float(live_full[0]["surr_entropy_bits"]),
            "base_full_smevr_surrogate_bits": float(live_full[0]["surr_smevr_surrogate_bits"]),
            "full_pair_permutation_rows": len(live_full),
            "per_cell_rows": len(live_cells),
            "spatial_tile_rows": len(live_tiles),
            "summaries": {
                "full_d_bytes": summarize_rows(pair_perm_deltas, "d_bytes"),
                "full_d_surr_entropy": summarize_rows(pair_perm_deltas, "d_surr_entropy"),
                "full_d_surr_smevr": summarize_rows(pair_perm_deltas, "d_surr_smevr"),
                "per_cell_smevr_bytes": summarize_rows(live_cells, "smevr_bytes"),
                "tile_smevr_bytes": summarize_rows(live_tiles, "smevr_bytes"),
            },
            "pair_permutation_entropy_blind_max_abs_delta_bits": (
                float(max(entropy_blind_abs)) if entropy_blind_abs else None
            ),
            "metrics": live_metrics,
        },
        "routing": {
            "verdict_scope": "FORMULATION: scorer-free token-stream byte surrogate selection",
            "migration_policy": "queue C_sum_affine behind current two-arm entropy/smevr surfaces; no hot-swap",
        },
    }
    return result, all_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default=str(DEFAULT_TOKEN_NPY))
    parser.add_argument("--rg5-rows", default=str(DEFAULT_RG5_ROWS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--levels", type=int, default=16)
    parser.add_argument("--codec", default="smevr")
    parser.add_argument("--temp", type=float, default=0.15)
    parser.add_argument("--random-perms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=865)
    parser.add_argument("--tile-h", type=int, default=4)
    parser.add_argument("--tile-w", type=int, default=4)
    parser.add_argument("--cell-limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result, rows = build_result(args)
    result_path = out_dir / "SM2_RESULTS.json"
    rows_path = out_dir / "SM2_ROWS.jsonl"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(rows_path, rows)
    print(json.dumps({"result": str(result_path), "rows": str(rows_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
