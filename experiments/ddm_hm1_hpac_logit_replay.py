"""ddm_hm1 phase A -- teacher-forced HPAC logit replay on the hv1 frontier archive.

The shipped decoder (``runtime.residual_archive.decode_production_tokens``) runs the
integer HPAC network once per group per frame, adds a 25x5 boundary/argmax residual
table to the raw logits, quantizes, softmaxes, and hands the result to an rc64 range
coder.  The decode is exact, so feeding the already-decoded token field back in as the
causal state reproduces the identical logits without needing the range coder at all.

This module does exactly that and RETAINS the raw pre-correction logits.  Those logits
are the reusable instrument for pricing model capacity: any additive-logit correction
table -- the shipped one, a larger one, or none -- can then be evaluated offline against
the true tokens without re-running the network.

Byte-identity gate: applying the shipped table to the retained logits must reproduce
``corrected_quantized_logit_sha256`` and ``corrected_cdf_input_sha256`` from the wc1
decode receipt, and the accumulated ``-log2 p[actual]`` must reproduce ddm_dc1's
measured HPAC cross-entropy.  If it does not, nothing downstream is admissible.

Axis: ``[macOS-CPU advisory / scorer-free byte measurement]``.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# The shipped decoder pins four threads inside inflate.py; match it so the replay
# exercises the same reduction order as the receiver it is claiming identity with.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "4")

DEFAULT_PREPARED = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815"
    "/prepared/hv1_base_control"
)
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815"
    "/runs/base_optimized_n600_r3/output/.f26_decode_checkpoints"
    "/tokens_cpu_stage_complete.u8"
)
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")

FRONTIER_ARCHIVE_SHA256 = (
    "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
)
FRONTIER_ARCHIVE_BYTES = 182_759
DECODED_TOKEN_SHA256 = (
    "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
)
# From ddm_wc1's native decode receipt, re-verified by ddm_dc1's byte-identical replay.
EXPECTED_CORRECTED_LOGIT_SHA256 = (
    "562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7"
)
EXPECTED_CDF_INPUT_SHA256 = (
    "dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7"
)
# ddm_dc1 retained/hpac_cross_entropy_n600.json
EXPECTED_CROSS_ENTROPY_BYTES = 112_109.57757858819

LOGIT_SCALE = 8  # runtime HPAC_LOGIT_PRECISION; logits are exact multiples of 1/8.


class ReplayError(RuntimeError):
    """Raised when the replay cannot be proven identical to the shipped decode."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _import_runtime(prepared: Path) -> tuple[Any, Any, Path]:
    """Import the prepared receiver tree exactly as ``inflate.py`` does."""
    prepared = prepared.resolve()
    if str(prepared) not in sys.path:
        sys.path.insert(0, str(prepared))
    from runtime import residual_archive

    renderer_dir = prepared / "cpr1"
    if str(renderer_dir) not in sys.path:
        sys.path.insert(0, str(renderer_dir))
    import importlib

    renderer = importlib.import_module("inflate")
    return residual_archive, renderer, renderer_dir


def _probability_from_corrected(corrected: np.ndarray, precision: int) -> np.ndarray:
    """Byte-identical copy of ``residual_archive._probability_table``."""
    quantized = np.clip(
        np.rint(np.asarray(corrected, dtype=np.float32) * precision),
        -32768,
        32767,
    ).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def replay(
    prepared: Path,
    tokens_path: Path,
    outdir: Path,
    frames: int | None,
    progress_every: int,
) -> dict[str, Any]:
    import torch

    residual_archive, renderer, renderer_dir = _import_runtime(prepared)
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1

    archive_path = prepared / "archive.zip"
    archive_sha = _sha256_file(archive_path)
    if archive_sha != FRONTIER_ARCHIVE_SHA256:
        raise ReplayError(
            f"archive is not the hv1 frontier: {archive_sha} != {FRONTIER_ARCHIVE_SHA256}"
        )
    if archive_path.stat().st_size != FRONTIER_ARCHIVE_BYTES:
        raise ReplayError("archive size does not match the hv1 frontier")

    height, width = int(renderer.EVAL_H), int(renderer.EVAL_W)
    total_frames = int(renderer.N)
    plane = height * width
    expected_tokens = total_frames * plane
    if tokens_path.stat().st_size != expected_tokens:
        raise ReplayError(
            f"token field is {tokens_path.stat().st_size} B, expected {expected_tokens}"
        )
    token_field = np.memmap(tokens_path, dtype=np.uint8, mode="r").reshape(
        total_frames, height, width
    )

    limit = total_frames if frames is None else int(frames)
    if limit <= 0 or limit > total_frames:
        raise ReplayError("frame limit is outside the real n600 field")
    is_full_field = limit == total_frames
    if is_full_field:
        field_sha = _sha256_file(tokens_path)
        if field_sha != DECODED_TOKEN_SHA256:
            raise ReplayError(
                f"token field sha {field_sha} != wc1 receipt {DECODED_TOKEN_SHA256}"
            )

    parts = residual_archive.read_residual_archive(archive_path)
    device = torch.device("cpu")
    base_hpac = materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual_archive._sparse_class(renderer_dir)(model, height, width)

    group_positions: list[np.ndarray] = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_positions.append(flat)
    covered = np.concatenate(group_positions)
    if covered.size != plane or np.unique(covered).size != plane:
        raise ReplayError("group masks do not partition the frame exactly once")

    group_index = np.empty(plane, dtype=np.uint8)
    device_positions: list[Any] = []
    for group, flat in enumerate(group_positions):
        group_index[flat] = group
        device_positions.append(torch.from_numpy(flat).to(device))

    outdir.mkdir(parents=True, exist_ok=True)
    logits_path = outdir / f"base_logits_int16_n{limit}.i16"
    boundary_path = outdir / f"boundary_bucket_n{limit}.u8"
    logits_out = np.memmap(
        logits_path, dtype=np.int16, mode="w+", shape=(limit, plane, 5)
    )
    boundary_out = np.memmap(
        boundary_path, dtype=np.uint8, mode="w+", shape=(limit, plane)
    )

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    cross_entropy_bits = 0.0
    minimum_probability = 1.0
    started = time.perf_counter()

    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros((1, height, width), dtype=torch.long, device=device)
        for frame in range(limit):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = residual_archive._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(plane, 4, dtype=np.uint8)
            boundary_out[frame] = boundary
            truth_flat = token_field[frame].reshape(-1).astype(np.int64)
            frame_logits = np.empty((plane, 5), dtype=np.float32)

            for group, flat in enumerate(group_positions):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                frame_logits[flat] = base_logits

                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat].astype(np.int64) * residual_archive.NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = _probability_from_corrected(
                    corrected, int(renderer.HPAC_LOGIT_PRECISION)
                )
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )

                actual = truth_flat[flat]
                chosen = probability[np.arange(actual.size), actual].astype(np.float64)
                minimum_probability = min(minimum_probability, float(chosen.min()))
                cross_entropy_bits -= float(np.log2(chosen).sum())

                current.reshape(-1)[device_positions[group]] = torch.from_numpy(
                    actual
                ).to(device)

            # int16 retention is lossless only if every raw logit is an exact
            # multiple of 1/8.  Check the assembled frame once rather than per group.
            scaled = frame_logits * LOGIT_SCALE
            as_int = np.rint(scaled)
            if not np.array_equal(scaled, as_int):
                raise ReplayError(
                    "raw logits are not exact multiples of 1/8; int16 retention "
                    "would be lossy"
                )
            if as_int.min() < -32768 or as_int.max() > 32767:
                raise ReplayError("raw logits do not fit int16")
            logits_out[frame] = as_int.astype(np.int16)

            rebuilt = current[0].to(device="cpu", dtype=torch.uint8).numpy()
            if not np.array_equal(rebuilt, token_field[frame]):
                raise ReplayError(f"teacher-forced state diverged at frame {frame}")
            previous = current
            if progress_every and frame % progress_every == 0:
                print(
                    f"  frame {frame}/{limit} xe={cross_entropy_bits / 8:,.0f}B "
                    f"t={time.perf_counter() - started:.0f}s",
                    flush=True,
                )

    logits_out.flush()
    boundary_out.flush()
    del logits_out, boundary_out
    group_path = outdir / "group_index.u8"
    group_path.write_bytes(group_index.tobytes())

    cross_entropy_bytes = cross_entropy_bits / 8.0
    corrected_sha = corrected_digest.hexdigest()
    cdf_sha = cdf_digest.hexdigest()
    identity = {
        "corrected_quantized_logit_sha256": corrected_sha,
        "corrected_cdf_input_sha256": cdf_sha,
        "matches_wc1_corrected_logits": (
            is_full_field and corrected_sha == EXPECTED_CORRECTED_LOGIT_SHA256
        ),
        "matches_wc1_cdf_input": (
            is_full_field and cdf_sha == EXPECTED_CDF_INPUT_SHA256
        ),
        "cross_entropy_bytes_matches_dc1": (
            is_full_field
            and abs(cross_entropy_bytes - EXPECTED_CROSS_ENTROPY_BYTES) < 1e-3
        ),
    }
    if is_full_field and not all(
        identity[key]
        for key in (
            "matches_wc1_corrected_logits",
            "matches_wc1_cdf_input",
            "cross_entropy_bytes_matches_dc1",
        )
    ):
        raise ReplayError(
            "full-field replay is NOT byte-identical to the shipped decode; "
            f"identity={identity}"
        )

    report = {
        "schema": "ddm_hm1_hpac_logit_replay.v1",
        "axis": "[macOS-CPU advisory / scorer-free byte measurement]",
        "score_claim": False,
        "promotable": False,
        "archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": FRONTIER_ARCHIVE_BYTES,
        },
        "tokens_path": str(tokens_path),
        "frames": limit,
        "is_full_field": is_full_field,
        "symbols": limit * plane,
        "height": height,
        "width": width,
        "groups": len(group_positions),
        "shipped_token_stream_bytes": len(parts.token_stream),
        "shipped_hpac_model_bytes": len(parts.hpac_blob),
        "shipped_table_states": int(parts.table.codes.shape[0]),
        "shipped_table_bits": int(parts.table.bits),
        "shipped_table_scale": float(parts.table.scale),
        "hpac_cross_entropy_bytes": cross_entropy_bytes,
        "minimum_probability_seen": minimum_probability,
        "identity": identity,
        "retained": {
            "base_logits_int16": str(logits_path),
            "boundary_bucket": str(boundary_path),
            "group_index": str(group_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="prefix length; omit for the full n600 field (the only verdict scale)",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    report = replay(
        args.prepared,
        args.tokens,
        args.outdir,
        args.frames,
        args.progress_every,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    destination = args.report or (
        args.outdir / f"logit_replay_n{report['frames']}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
