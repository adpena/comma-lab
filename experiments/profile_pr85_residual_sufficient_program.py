#!/usr/bin/env python3
"""Profile smaller sufficient programs for the PR85 mask token stream.

This is a local planning profiler only. It searches for non-arbitrary residual
representations of the decoded PR85 QMA9 mask token tensor: exact predictors,
mod-5 residuals, sparse nonzero maps, and row-span atoms. It does not build an
archive, run a scorer, dispatch GPU work, or claim a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_PATH = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_PROFILE_JSON = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_token_source_profile.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr85_residual_sufficient_program_20260504_codex"

SCHEMA = "pr85_residual_sufficient_program_profile_v1"
TOOL = "experiments/profile_pr85_residual_sufficient_program.py"
CONTEST_RATE_LAMBDA = 25.0 / 37_545_489.0


class ResidualProgramError(ValueError):
    """Raised when token custody or profiler inputs are inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts64 = np.asarray(counts, dtype=np.float64)
    total = float(counts64.sum())
    if total <= 0.0:
        return 0.0
    probs = counts64[counts64 > 0.0] / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return 0.0 if abs(entropy) < 1e-15 else entropy


def _read_source_profile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResidualProgramError(f"invalid token profile JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ResidualProgramError("token profile JSON must be an object")
    return payload


def load_storage_tokens(
    token_path: Path,
    profile_json: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load PR85 QMA9 tokens in recorded storage order ``N,W,H``."""

    profile = _read_source_profile(profile_json)
    token_source = profile.get("token_source")
    if not isinstance(token_source, Mapping):
        raise ResidualProgramError("profile missing token_source object")
    shape_raw = token_source.get("shape")
    if (
        not isinstance(shape_raw, list)
        or len(shape_raw) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape_raw)
    ):
        raise ResidualProgramError(f"invalid token shape: {shape_raw!r}")
    if token_source.get("dtype") != "uint8":
        raise ResidualProgramError(f"expected uint8 tokens, got {token_source.get('dtype')!r}")
    shape = tuple(int(value) for value in shape_raw)
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    actual_bytes = int(token_path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ResidualProgramError(
            f"token byte count mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    expected_sha = token_source.get("sha256")
    actual_sha = _sha256_file(token_path)
    if isinstance(expected_sha, str) and expected_sha and expected_sha != actual_sha:
        raise ResidualProgramError(f"token SHA mismatch: expected {expected_sha}, got {actual_sha}")
    tokens = np.memmap(token_path, mode="r", dtype=np.uint8, shape=shape)
    return tokens, profile


def storage_nwh_to_render_nhw(tokens: np.ndarray) -> np.ndarray:
    """Normalize PR85 QMA9 storage ``N,W,H`` tokens to renderer ``N,H,W``."""

    if tokens.ndim != 3:
        raise ResidualProgramError(f"expected 3D token tensor, got {tokens.ndim}D")
    return np.ascontiguousarray(np.transpose(tokens, (0, 2, 1)))


def _predictor(tokens: np.ndarray, name: str) -> np.ndarray:
    pred = np.zeros_like(tokens, dtype=np.uint8)
    if name == "absolute_zero":
        return pred
    if name == "time_prev_zero_first":
        if tokens.shape[0] > 1:
            pred[1:] = tokens[:-1]
        return pred
    if name == "left_zero_border":
        if tokens.shape[2] > 1:
            pred[:, :, 1:] = tokens[:, :, :-1]
        return pred
    if name == "up_zero_border":
        if tokens.shape[1] > 1:
            pred[:, 1:, :] = tokens[:, :-1, :]
        return pred
    if name == "time_prev_then_left_border":
        if tokens.shape[0] > 1:
            pred[1:] = tokens[:-1]
        if tokens.shape[2] > 1:
            pred[:, :, 0] = tokens[:, :, 0]
        return pred
    raise ResidualProgramError(f"unknown predictor: {name}")


def mod5_residual(tokens: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if tokens.shape != prediction.shape:
        raise ResidualProgramError("prediction shape mismatch")
    return ((tokens.astype(np.int16) - prediction.astype(np.int16)) % 5).astype(np.uint8)


def _frame_quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "p050": float(np.quantile(values, 0.50)),
        "p090": float(np.quantile(values, 0.90)),
        "p099": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _row_span_stats(nonzero: np.ndarray) -> dict[str, Any]:
    if nonzero.ndim != 3:
        raise ResidualProgramError("nonzero mask must be NHW")
    frame_count, height, width = (int(v) for v in nonzero.shape)
    row_has = nonzero.any(axis=2)
    span_count = int(row_has.sum())
    if span_count:
        x = np.arange(width, dtype=np.int32)
        mins = np.where(nonzero, x[None, None, :], width).min(axis=2)
        maxs = np.where(nonzero, x[None, None, :], -1).max(axis=2)
        widths = (maxs[row_has] - mins[row_has] + 1).astype(np.int64, copy=False)
    else:
        widths = np.array([], dtype=np.int64)
    rows_per_frame = row_has.sum(axis=1).astype(np.int64, copy=False)
    changed_per_frame = nonzero.reshape(frame_count, height * width).sum(axis=1).astype(
        np.int64,
        copy=False,
    )
    top_frames = sorted(
        (
            {
                "changed_pixels": int(changed_per_frame[index]),
                "frame": int(index),
                "rows_with_changes": int(rows_per_frame[index]),
            }
            for index in range(frame_count)
        ),
        key=lambda row: (-int(row["changed_pixels"]), int(row["frame"])),
    )[:12]
    return {
        "changed_pixels_per_frame": _frame_quantiles(changed_per_frame),
        "frame_count": frame_count,
        "height": height,
        "row_span_count": span_count,
        "row_span_width_quantiles": _frame_quantiles(widths),
        "rows_per_frame_quantiles": _frame_quantiles(rows_per_frame),
        "top_changed_frames": top_frames,
        "width": width,
    }


def residual_program_record(
    tokens: np.ndarray,
    *,
    predictor_name: str,
    charged_mask_bytes: int,
) -> dict[str, Any]:
    prediction = _predictor(tokens, predictor_name)
    residual = mod5_residual(tokens, prediction)
    counts = np.bincount(residual.reshape(-1), minlength=5)[:5].astype(np.int64, copy=False)
    total = int(counts.sum())
    zero_count = int(counts[0])
    nonzero_count = total - zero_count
    nonzero_counts = counts[1:]
    symbol_entropy = _entropy_from_counts(counts)
    binary_counts = np.array([zero_count, nonzero_count], dtype=np.int64)
    binary_entropy = _entropy_from_counts(binary_counts)
    nonzero_entropy = _entropy_from_counts(nonzero_counts)
    raw_entropy_bytes = symbol_entropy * total / 8.0
    split_entropy_bytes = binary_entropy * total / 8.0 + nonzero_entropy * nonzero_count / 8.0
    nonzero = residual != 0
    spans = _row_span_stats(nonzero)
    # Fixed-width atom estimates are intentionally simple and pessimistic. They
    # price a row-span map plus exact residual symbols inside each changed span.
    span_header_bytes_fixed16 = spans["row_span_count"] * 6
    span_payload_symbol_lower_bound = nonzero_entropy * nonzero_count / 8.0
    span_program_lower_bound_bytes = span_header_bytes_fixed16 + span_payload_symbol_lower_bound
    best_lower_bound = min(raw_entropy_bytes, split_entropy_bytes, span_program_lower_bound_bytes)
    saved = float(charged_mask_bytes - best_lower_bound)
    return {
        "best_lower_bound_bytes": float(best_lower_bound),
        "charged_mask_bytes": int(charged_mask_bytes),
        "counts": {str(i): int(v) for i, v in enumerate(counts.tolist())},
        "dispatch_unlocked": False,
        "estimated_bytes_saved_vs_charged_mask": saved,
        "evidence_grade": "empirical/local_residual_sufficient_statistics",
        "non_arbitrary_basis": (
            "exact decoded PR85 QMA9 token tensor plus deterministic predictor "
            f"{predictor_name!r}"
        ),
        "nonzero_fraction": float(nonzero_count / total) if total else 0.0,
        "nonzero_symbol_entropy_bits": float(nonzero_entropy),
        "planning_only": True,
        "predictor": predictor_name,
        "rate_score_delta_lower_bound": float(-saved * CONTEST_RATE_LAMBDA),
        "raw_residual_entropy_bits_per_token": float(symbol_entropy),
        "raw_residual_entropy_bytes": float(raw_entropy_bytes),
        "score_claim": False,
        "sparse_split_entropy_bytes": float(split_entropy_bytes),
        "sparse_zero_map_entropy_bits_per_token": float(binary_entropy),
        "span_program_fixed16_lower_bound_bytes": float(span_program_lower_bound_bytes),
        "span_program_fixed16_overhead_bytes": int(span_header_bytes_fixed16),
        "span_program_fixed16_payload_entropy_bytes": float(span_payload_symbol_lower_bound),
        "row_span_atoms": spans,
        "token_count": total,
        "zero_fraction": float(zero_count / total) if total else 0.0,
    }


def build_profile(
    *,
    token_path: Path,
    profile_json: Path,
    predictors: Sequence[str] = (
        "absolute_zero",
        "time_prev_zero_first",
        "left_zero_border",
        "up_zero_border",
        "time_prev_then_left_border",
    ),
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    storage_tokens, source_profile = load_storage_tokens(token_path, profile_json)
    render_tokens = storage_nwh_to_render_nhw(storage_tokens)
    if render_tokens.size and (int(render_tokens.min()) < 0 or int(render_tokens.max()) > 4):
        raise ResidualProgramError("token class values outside 0..4")
    mask_identity = source_profile.get("mask_segment_identity")
    if not isinstance(mask_identity, Mapping) or int(mask_identity.get("bytes", 0)) <= 0:
        raise ResidualProgramError("profile missing positive mask_segment_identity.bytes")
    charged_mask_bytes = int(mask_identity["bytes"])
    records = [
        residual_program_record(
            render_tokens,
            predictor_name=predictor,
            charged_mask_bytes=charged_mask_bytes,
        )
        for predictor in predictors
    ]
    ranked = sorted(
        records,
        key=lambda row: (
            -float(row["estimated_bytes_saved_vs_charged_mask"]),
            str(row["predictor"]),
        ),
    )
    best = ranked[0]
    if float(best["estimated_bytes_saved_vs_charged_mask"]) > 0.0:
        next_action = "prototype_byte_closed_coder_for_best_residual_program"
    else:
        next_action = "do_not_dispatch_residual_program_without_model_or_atom_refinement"
    return {
        "axis_convention": {
            "input_storage": "N,W,H from PR85 QMA9 token source",
            "profile_order": "N,H,W render order",
        },
        "charged_baseline": {
            "contest_rate_lambda_points_per_byte": CONTEST_RATE_LAMBDA,
            "mask_segment_bytes": charged_mask_bytes,
            "mask_segment_sha256": mask_identity.get("sha256"),
            "qma9_bits_per_token_charged": float(charged_mask_bytes * 8.0 / render_tokens.size),
        },
        "dispatch_performed": False,
        "evidence_grade": "empirical/local_residual_sufficient_statistics",
        "input_profile": {
            "path": str(profile_json),
            "sha256": _sha256_file(profile_json),
            "schema": source_profile.get("schema"),
        },
        "input_token_source": {
            "bytes": int(token_path.stat().st_size),
            "dtype": "uint8",
            "path": str(token_path),
            "render_order_sha256": hashlib.sha256(render_tokens.tobytes(order="C")).hexdigest(),
            "render_shape": [int(v) for v in render_tokens.shape],
            "sha256": _sha256_file(token_path),
            "storage_shape": [int(v) for v in storage_tokens.shape],
        },
        "planning_only": True,
        "recommendations": [
            {
                "action": next_action,
                "basis": (
                    "best deterministic residual sufficient-statistic lower bound "
                    f"comes from {best['predictor']}"
                ),
                "dispatch_unlocked": False,
                "predictor": best["predictor"],
                "score_claim": False,
            },
            {
                "action": "if_training_on_fast_gpu_use_this_profile_as_lossless_target_density",
                "basis": (
                    "top changed frames, row spans, and residual nonzero maps provide a "
                    "non-arbitrary curriculum/atom density field for learned mask coders"
                ),
                "dispatch_unlocked": False,
                "score_claim": False,
            },
        ],
        "recorded_at_utc": recorded_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "residual_programs": ranked,
        "schema": SCHEMA,
        "score_claim": False,
        "tool": TOOL,
    }


def render_markdown(profile: Mapping[str, Any], *, top_k: int = 8) -> str:
    baseline = profile["charged_baseline"]
    lines = [
        "# PR85 Residual Sufficient-Program Profile",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        f"- charged_qma9_mask_bytes: {baseline['mask_segment_bytes']}",
        f"- charged_qma9_bits_per_token: {baseline['qma9_bits_per_token_charged']:.9f}",
        f"- token_source_sha256: `{profile['input_token_source']['sha256']}`",
        f"- render_order_sha256: `{profile['input_token_source']['render_order_sha256']}`",
        "",
        "## Residual Programs",
        "",
        "| rank | predictor | zero frac | nonzero frac | best lb bytes | est saved bytes | rate-score delta lb | row spans |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(profile["residual_programs"][:top_k], start=1):
        lines.append(
            "| {rank} | `{predictor}` | {zero:.6f} | {nonzero:.6f} | {lb:.3f} | {saved:.3f} | {delta:.9f} | {spans} |".format(
                rank=rank,
                predictor=row["predictor"],
                zero=float(row["zero_fraction"]),
                nonzero=float(row["nonzero_fraction"]),
                lb=float(row["best_lower_bound_bytes"]),
                saved=float(row["estimated_bytes_saved_vs_charged_mask"]),
                delta=float(row["rate_score_delta_lower_bound"]),
                spans=int(row["row_span_atoms"]["row_span_count"]),
            )
        )
    lines.extend(["", "## Recommendations", ""])
    for row in profile["recommendations"]:
        lines.append(f"- `{row['action']}`: {row['basis']}")
    lines.append(
        "\nThese are local sufficient-statistic bounds only. They can select a coder or training curriculum, but cannot claim score until a byte-closed archive passes exact CUDA auth eval.\n"
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--predictors",
        default="absolute_zero,time_prev_zero_first,left_zero_border,up_zero_border,time_prev_then_left_border",
        help="Comma-separated deterministic predictors to profile.",
    )
    parser.add_argument("--output-json-name", default="pr85_residual_sufficient_program_profile.json")
    parser.add_argument("--output-md-name", default="pr85_residual_sufficient_program_profile.md")
    parser.add_argument("--recorded-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    predictors = tuple(part.strip() for part in args.predictors.split(",") if part.strip())
    if not predictors:
        raise SystemExit("--predictors must include at least one predictor")
    profile = build_profile(
        token_path=args.token_path,
        profile_json=args.profile_json,
        predictors=predictors,
        recorded_at_utc=args.recorded_at_utc,
    )
    output_json = args.output_dir / args.output_json_name
    output_md = args.output_dir / args.output_md_name
    _write_json(output_json, profile)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(profile), encoding="utf-8")
    print(
        json.dumps(
            {
                "dispatch_performed": False,
                "output_json": str(output_json),
                "output_md": str(output_md),
                "planning_only": True,
                "score_claim": False,
                "top_predictor": profile["residual_programs"][0]["predictor"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
