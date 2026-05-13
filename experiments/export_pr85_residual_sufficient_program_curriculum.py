#!/usr/bin/env python3
"""Export PR85 residual sufficient-program density as a training curriculum.

This is a local planning/export tool only. It consumes the PR85 residual
sufficient-program profile plus the decoded QMA9 token source, then emits a
compact JSON/NPZ substrate for learned or native mask coders. It does not build
an archive, run a scorer, dispatch GPU work, or claim a score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib import format as np_format


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_JSON = (
    REPO_ROOT
    / "experiments/results/pr85_residual_sufficient_program_20260504_codex/"
    "pr85_residual_sufficient_program_profile.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/pr85_residual_sufficient_program_20260504_codex/"
    "curriculum_density"
)

TOOL = "experiments/export_pr85_residual_sufficient_program_curriculum.py"
SCHEMA = "pr85_residual_sufficient_program_curriculum_density_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
KNOWN_PREDICTORS = {
    "absolute_zero",
    "time_prev_zero_first",
    "left_zero_border",
    "up_zero_border",
    "time_prev_then_left_border",
}


class CurriculumExportError(ValueError):
    """Raised when a profile, token source, or export contract is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CurriculumExportError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CurriculumExportError(f"{path} must contain a JSON object")
    return payload


def _positive_int_list(value: Any, *, field: str, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise CurriculumExportError(f"invalid {field}: {value!r}")
    return tuple(int(item) for item in value)


def _select_predictor(
    profile: Mapping[str, Any],
    predictor: str | None,
) -> tuple[str, int, Mapping[str, Any]]:
    programs = profile.get("residual_programs")
    if not isinstance(programs, list) or not programs:
        raise CurriculumExportError("profile missing nonempty residual_programs")
    rows: list[Mapping[str, Any]] = []
    for row in programs:
        if not isinstance(row, Mapping) or not isinstance(row.get("predictor"), str):
            raise CurriculumExportError("each residual_program row must include predictor")
        rows.append(row)

    predictor_id = predictor or str(rows[0]["predictor"])
    if predictor_id not in KNOWN_PREDICTORS:
        raise CurriculumExportError(f"unknown predictor: {predictor_id}")
    for index, row in enumerate(rows, start=1):
        if row["predictor"] == predictor_id:
            return predictor_id, index, row
    raise CurriculumExportError(f"predictor {predictor_id!r} is not present in profile")


def _token_source_from_profile(
    profile: Mapping[str, Any],
    *,
    profile_json: Path,
    token_path_override: Path | None,
) -> tuple[Path, tuple[int, int, int], dict[str, Any]]:
    source = profile.get("input_token_source")
    if not isinstance(source, Mapping):
        raise CurriculumExportError("profile missing input_token_source object")

    token_value = token_path_override or source.get("path")
    if token_value is None:
        raise CurriculumExportError("profile has no input_token_source.path; pass --token-path")
    token_path = (
        Path(token_value)
        if isinstance(token_value, Path)
        else _resolve_path(str(token_value), base_dir=profile_json.parent)
    )
    if not token_path.is_file():
        raise CurriculumExportError(f"token source does not exist: {token_path}")

    storage_shape = _positive_int_list(
        source.get("storage_shape") or source.get("shape"),
        field="input_token_source.storage_shape",
        length=3,
    )
    if source.get("dtype") not in (None, "uint8"):
        raise CurriculumExportError(f"expected uint8 token source, got {source.get('dtype')!r}")
    expected_bytes = int(np.prod(storage_shape, dtype=np.int64))
    actual_bytes = int(token_path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise CurriculumExportError(
            f"token byte count mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    expected_sha = source.get("sha256")
    actual_sha = _sha256_file(token_path)
    if isinstance(expected_sha, str) and expected_sha and expected_sha != actual_sha:
        raise CurriculumExportError(f"token SHA mismatch: expected {expected_sha}, got {actual_sha}")

    custody = {
        "bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha,
        "path": _repo_rel(token_path),
        "sha256": actual_sha,
        "sha256_match": expected_sha in (None, "", actual_sha),
        "storage_shape_nwh": [int(value) for value in storage_shape],
    }
    return token_path, storage_shape, custody


def load_render_tokens(token_path: Path, storage_shape_nwh: tuple[int, int, int]) -> np.ndarray:
    """Load uint8 PR85 tokens and normalize storage ``N,W,H`` to render ``N,H,W``."""

    storage = np.memmap(token_path, mode="r", dtype=np.uint8, shape=storage_shape_nwh)
    render = np.asarray(np.transpose(storage, (0, 2, 1)))
    if render.size and (int(render.min()) < 0 or int(render.max()) > 4):
        raise CurriculumExportError("token class values outside 0..4")
    return render


def predict_tokens(tokens_nhw: np.ndarray, predictor_id: str) -> np.ndarray:
    """Return the deterministic predictor tensor used by the residual profile."""

    if predictor_id not in KNOWN_PREDICTORS:
        raise CurriculumExportError(f"unknown predictor: {predictor_id}")
    prediction = np.zeros_like(tokens_nhw, dtype=np.uint8)
    if predictor_id == "absolute_zero":
        return prediction
    if predictor_id == "time_prev_zero_first":
        if tokens_nhw.shape[0] > 1:
            prediction[1:] = tokens_nhw[:-1]
        return prediction
    if predictor_id == "left_zero_border":
        if tokens_nhw.shape[2] > 1:
            prediction[:, :, 1:] = tokens_nhw[:, :, :-1]
        return prediction
    if predictor_id == "up_zero_border":
        if tokens_nhw.shape[1] > 1:
            prediction[:, 1:, :] = tokens_nhw[:, :-1, :]
        return prediction
    if predictor_id == "time_prev_then_left_border":
        if tokens_nhw.shape[0] > 1:
            prediction[1:] = tokens_nhw[:-1]
        if tokens_nhw.shape[2] > 1:
            prediction[:, :, 0] = tokens_nhw[:, :, 0]
        return prediction
    raise AssertionError(f"unreachable predictor: {predictor_id}")


def residual_nonzero_mask(tokens_nhw: np.ndarray, predictor_id: str) -> tuple[np.ndarray, np.ndarray]:
    prediction = predict_tokens(tokens_nhw, predictor_id)
    residual = ((tokens_nhw.astype(np.int16) - prediction.astype(np.int16)) % 5).astype(
        np.uint8
    )
    return residual, residual != 0


def _top_frame_records(
    frame_counts: np.ndarray,
    rows_with_changes: np.ndarray,
    *,
    frame_pixels: int,
    top_frame_count: int,
) -> list[dict[str, Any]]:
    order = sorted(
        range(int(frame_counts.size)),
        key=lambda frame: (-int(frame_counts[frame]), int(frame)),
    )[:top_frame_count]
    return [
        {
            "density": float(frame_counts[frame] / frame_pixels) if frame_pixels else 0.0,
            "frame": int(frame),
            "nonzero_count": int(frame_counts[frame]),
            "rows_with_changes": int(rows_with_changes[frame]),
        }
        for frame in order
    ]


def _row_span_arrays(
    nonzero: np.ndarray,
    row_counts: np.ndarray,
    *,
    max_row_spans: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
    frame_count, height, width = (int(value) for value in nonzero.shape)
    frame_idx, row_idx = np.nonzero(row_counts > 0)
    total_available = int(frame_idx.size)
    if max_row_spans <= 0 or total_available == 0:
        empty = {
            "row_span_frame": np.array([], dtype=np.int32),
            "row_span_row": np.array([], dtype=np.int32),
            "row_span_x0": np.array([], dtype=np.int32),
            "row_span_x1": np.array([], dtype=np.int32),
            "row_span_nonzero_count": np.array([], dtype=np.int32),
            "row_span_density": np.array([], dtype=np.float32),
        }
        return empty, [], total_available

    order = sorted(
        range(total_available),
        key=lambda i: (-int(row_counts[frame_idx[i], row_idx[i]]), int(frame_idx[i]), int(row_idx[i])),
    )[:max_row_spans]
    selected_frame = frame_idx[order].astype(np.int32, copy=False)
    selected_row = row_idx[order].astype(np.int32, copy=False)
    x0 = np.empty(len(order), dtype=np.int32)
    x1 = np.empty(len(order), dtype=np.int32)
    counts = np.empty(len(order), dtype=np.int32)
    density = np.empty(len(order), dtype=np.float32)
    records: list[dict[str, Any]] = []
    for out_index, (frame, row) in enumerate(zip(selected_frame.tolist(), selected_row.tolist())):
        xs = np.flatnonzero(nonzero[int(frame), int(row)])
        left = int(xs[0])
        right = int(xs[-1])
        count = int(row_counts[int(frame), int(row)])
        row_density = float(count / width) if width else 0.0
        x0[out_index] = left
        x1[out_index] = right
        counts[out_index] = count
        density[out_index] = row_density
        records.append(
            {
                "density": row_density,
                "frame": int(frame),
                "nonzero_count": count,
                "row": int(row),
                "x0": left,
                "x1": right,
            }
        )
    arrays = {
        "row_span_frame": selected_frame,
        "row_span_row": selected_row,
        "row_span_x0": x0,
        "row_span_x1": x1,
        "row_span_nonzero_count": counts,
        "row_span_density": density,
    }
    return arrays, records, total_available


def _array_to_npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np_format.write_array(handle, np.asarray(array), allow_pickle=False)
    return handle.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            archive.writestr(info, _array_to_npy_bytes(np.asarray(arrays[name])))


def build_curriculum_export(
    *,
    profile_json: Path,
    token_path: Path | None = None,
    predictor: str | None = None,
    top_frame_count: int = 32,
    max_row_spans: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if top_frame_count <= 0:
        raise CurriculumExportError("--top-frame-count must be positive")
    if max_row_spans < 0:
        raise CurriculumExportError("--max-row-spans must be nonnegative")

    profile = _read_json_object(profile_json)
    predictor_id, predictor_rank, predictor_row = _select_predictor(profile, predictor)
    token_source_path, storage_shape, token_custody = _token_source_from_profile(
        profile,
        profile_json=profile_json,
        token_path_override=token_path,
    )
    tokens = load_render_tokens(token_source_path, storage_shape)
    render_tokens = np.ascontiguousarray(tokens)
    residual, nonzero = residual_nonzero_mask(render_tokens, predictor_id)
    frame_count, height, width = (int(value) for value in nonzero.shape)
    frame_pixels = height * width
    row_counts = nonzero.sum(axis=2, dtype=np.int32)
    frame_counts = row_counts.sum(axis=1, dtype=np.int32)
    rows_with_changes = (row_counts > 0).sum(axis=1, dtype=np.int32)
    total_nonzero = int(frame_counts.sum())
    token_count = int(nonzero.size)
    frame_density = (
        frame_counts.astype(np.float32) / np.float32(frame_pixels) if frame_pixels else np.zeros(frame_count, dtype=np.float32)
    )
    row_density = (
        row_counts.astype(np.float32) / np.float32(width) if width else np.zeros_like(row_counts, dtype=np.float32)
    )
    frame_weight = np.zeros(frame_count, dtype=np.float32)
    if total_nonzero > 0:
        frame_weight = frame_counts.astype(np.float32) / np.float32(total_nonzero)
    row_weight = np.zeros_like(row_density, dtype=np.float32)
    if total_nonzero > 0:
        row_weight = row_counts.astype(np.float32) / np.float32(total_nonzero)

    residual_counts = np.bincount(residual.reshape(-1), minlength=5)[:5].astype(
        np.int64,
        copy=False,
    )
    top_frames = _top_frame_records(
        frame_counts,
        rows_with_changes,
        frame_pixels=frame_pixels,
        top_frame_count=min(top_frame_count, frame_count),
    )
    span_arrays, row_span_records, row_span_total_available = _row_span_arrays(
        nonzero,
        row_counts,
        max_row_spans=max_row_spans,
    )

    top_indices = np.array([row["frame"] for row in top_frames], dtype=np.int32)
    arrays: dict[str, np.ndarray] = {
        "frame_density": frame_density.astype(np.float32, copy=False),
        "frame_index": np.arange(frame_count, dtype=np.int32),
        "frame_nonzero_count": frame_counts.astype(np.int32, copy=False),
        "frame_sampling_weight": frame_weight.astype(np.float32, copy=False),
        "residual_symbol_counts": residual_counts,
        "row_density": row_density.astype(np.float32, copy=False),
        "row_nonzero_count": row_counts.astype(np.int32, copy=False),
        "row_sampling_weight": row_weight.astype(np.float32, copy=False),
        "shape_nhw": np.array([frame_count, height, width], dtype=np.int32),
        "top_frame_density": frame_density[top_indices].astype(np.float32, copy=False),
        "top_frame_index": top_indices,
        "top_frame_nonzero_count": frame_counts[top_indices].astype(np.int32, copy=False),
        "top_frame_rows_with_changes": rows_with_changes[top_indices].astype(np.int32, copy=False),
        **span_arrays,
    }

    profile_sha = _sha256_file(profile_json)
    render_sha = _sha256_bytes(render_tokens.tobytes(order="C"))
    expected_render_sha = profile.get("input_token_source", {}).get("render_order_sha256")
    charged_baseline = profile.get("charged_baseline") if isinstance(profile.get("charged_baseline"), Mapping) else {}
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "gpu_required": False,
        "dispatch_unlocked": False,
        "evidence_grade": "empirical/local_curriculum_density_export",
        "source_profile_recorded_at_utc": profile.get("recorded_at_utc"),
        "axis_convention": {
            "input_storage": "N,W,H from PR85 QMA9 token source",
            "export_order": "N,H,W render order",
        },
        "predictor": {
            "id": predictor_id,
            "profile_rank": predictor_rank,
            "profile_zero_fraction": predictor_row.get("zero_fraction"),
            "profile_nonzero_fraction": predictor_row.get("nonzero_fraction"),
            "profile_best_lower_bound_bytes": predictor_row.get("best_lower_bound_bytes"),
        },
        "density": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "token_count": token_count,
            "nonzero_count": total_nonzero,
            "nonzero_fraction": float(total_nonzero / token_count) if token_count else 0.0,
            "residual_symbol_counts": {str(i): int(value) for i, value in enumerate(residual_counts.tolist())},
            "row_span_total_available": row_span_total_available,
            "row_span_record_count": len(row_span_records),
            "row_span_record_cap": int(max_row_spans),
        },
        "curriculum": {
            "frame_sort": "nonzero_count_desc_frame_asc",
            "row_span_sort": "row_nonzero_count_desc_frame_asc_row_asc",
            "top_frame_count": len(top_frames),
            "top_frames": top_frames,
            "row_span_records": row_span_records,
            "weight_basis": "residual nonzero density for selected predictor",
        },
        "custody": {
            "source_profile": {
                "bytes": int(profile_json.stat().st_size),
                "path": _repo_rel(profile_json),
                "schema": profile.get("schema"),
                "sha256": profile_sha,
            },
            "token_source": {
                **token_custody,
                "render_order_sha256": render_sha,
                "expected_render_order_sha256": expected_render_sha,
                "render_order_sha256_match": expected_render_sha in (None, "", render_sha),
                "render_shape_nhw": [frame_count, height, width],
            },
            "charged_baseline": {
                "mask_segment_bytes": charged_baseline.get("mask_segment_bytes"),
                "mask_segment_sha256": charged_baseline.get("mask_segment_sha256"),
            },
            "npz": None,
        },
        "artifact_contract": {
            "arrays_npz": {
                name: {
                    "dtype": str(np.asarray(array).dtype),
                    "shape": [int(value) for value in np.asarray(array).shape],
                }
                for name, array in sorted(arrays.items())
            },
            "score_claim": False,
            "dispatch": False,
        },
    }
    return summary, arrays


def write_export(
    *,
    profile_json: Path,
    output_dir: Path,
    token_path: Path | None = None,
    predictor: str | None = None,
    top_frame_count: int = 32,
    max_row_spans: int = 0,
    output_json_name: str = "pr85_residual_sufficient_program_curriculum_density.json",
    output_npz_name: str = "pr85_residual_sufficient_program_curriculum_density.npz",
) -> dict[str, Any]:
    summary, arrays = build_curriculum_export(
        profile_json=profile_json,
        token_path=token_path,
        predictor=predictor,
        top_frame_count=top_frame_count,
        max_row_spans=max_row_spans,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / output_npz_name
    write_deterministic_npz(output_npz, arrays)
    summary["custody"]["npz"] = {
        "bytes": int(output_npz.stat().st_size),
        "path": _repo_rel(output_npz),
        "sha256": _sha256_file(output_npz),
    }
    output_json = output_dir / output_json_name
    _write_json(output_json, summary)
    summary["output_json"] = {
        "bytes": int(output_json.stat().st_size),
        "path": _repo_rel(output_json),
        "sha256": _sha256_file(output_json),
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--token-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictor", default=None, help="Predictor id from the residual profile; defaults to the ranked-best row.")
    parser.add_argument("--top-frame-count", type=int, default=32)
    parser.add_argument("--max-row-spans", type=int, default=0)
    parser.add_argument(
        "--output-json-name",
        default="pr85_residual_sufficient_program_curriculum_density.json",
    )
    parser.add_argument(
        "--output-npz-name",
        default="pr85_residual_sufficient_program_curriculum_density.npz",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = write_export(
            profile_json=args.profile_json,
            token_path=args.token_path,
            output_dir=args.output_dir,
            predictor=args.predictor,
            top_frame_count=int(args.top_frame_count),
            max_row_spans=int(args.max_row_spans),
            output_json_name=args.output_json_name,
            output_npz_name=args.output_npz_name,
        )
    except CurriculumExportError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "dispatch": False,
                "dispatch_performed": False,
                "json": summary["output_json"],
                "npz": summary["custody"]["npz"],
                "planning_only": True,
                "predictor_id": summary["predictor"]["id"],
                "score_claim": False,
                "top_frame_count": summary["curriculum"]["top_frame_count"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
