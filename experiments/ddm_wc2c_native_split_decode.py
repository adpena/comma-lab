#!/usr/bin/env python3
"""ddm_wc2c - native split token decode: C does the integer model, numpy keeps the corrector.

THE PROBLEM.  ``runtime/f26_inflate.py:435-441`` hard-refuses ``native-hpac``
because the fused ``f26_hpac_decode_frame`` runs its own probability table and
RC64 coder, leaving no seam for the shipped decode-time probability corrector.
An unpatched native path would decode a different field, so the refusal is
correct.  jg5's token stage is 1,341.540 s of a 1,419.900 s inflate
[contest-CUDA T4] against a CUDA residual of [822, 1302] s -- REFUSE -- so the
refusal is also the submission's critical path.

WHY THIS SHAPE AND NOT A C PORT OF THE CORRECTOR.  MEASURED, this arm, n=12
frame prefix, 2,280 iterations, [macOS-CPU advisory]
(``ddm_wc2c_token_stage_profile.py``):

    sparse_selected_logits   60.78%  |  corrector_coding_row    18.55%
    frame_context             3.59%  |  corrector_observe        9.46%
    ------------------------ 64.37%  |  corrector_group_state    4.33%
    integer model                    |  ---------------------- 32.34%
                                     |  float64 corrector
    probability + digests + RC64 + host/device transfers:  ~3.3%

Lowering the integer model alone has a ceiling of **2.807x**; the derived PASS
bar is 1.804x and the WARN bar is 1.096x (``ddm_wc2`` GO section).  So the
integer half clears the bar on its own, and it carries NO floating-point
identity hazard -- every value in it is an exact integer op.  The corrector half
is 2,121 lines of float64 across four modules whose identity rests on IEEE
``+ - * /`` in a hand-fixed summation order (``free_corrector.py:266-279``); its
measured failure mode under a reordered reduction is S = 27.83, a
desynchronised decoder, not a rounding wobble.  Doing the safe 64% first is the
smaller, honest, provable step; the corrector port remains available for the
remaining 3.6x and must be argued on its own measured merits.

WHAT IS PROVEN, NOT ASSUMED.  Nothing here is admissible on a speed argument.
The driver reproduces, bit for bit, the shipping receipt's
``corrected_quantized_logit_sha256``, ``corrected_cdf_input_sha256``,
``decoded_token_sha256`` and RC64 ``decoder_bit_position``.  A single differing
byte is a REFUSAL regardless of speed.

AXIS.  ``[macOS-CPU advisory]``.  Never a score.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CANDIDATE = Path("/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5")

# jg5 pointer body, read at source from
# /Volumes/APDataStore/pact/ddm_jg5/t4_row_r1/harvested_artifacts/contest_auth_eval.stdout.log
JG5_ANCHORS = {
    "archive_sha256": "f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e",
    "archive_bytes": 180_625,
    "decoded_token_sha256": (
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
    ),
    "corrected_quantized_logit_sha256": (
        "8269fe1aad031620b18051ad784d877bc9e6e9a4a71e775e78681955c4eec4dd"
    ),
    "corrected_cdf_input_sha256": (
        "370a5e2a85ccbb1e598c84333cc851f0a8c352091fde272160826b4b04e46000"
    ),
    "raw_sha256": "6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883",
    "decoder_bit_position": 910_837,
}


class Wc2cSplitError(RuntimeError):
    """A precondition, identity, or contract invariant of the split path failed."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _import_candidate(candidate: Path):
    candidate = candidate.resolve()
    if not (candidate / "runtime" / "f26_inflate.py").is_file():
        raise Wc2cSplitError(f"no candidate runtime tree at {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    import runtime.f26_hpac_native as native_binding
    import runtime.f26_inflate as f26_inflate
    import runtime.residual_archive as residual_archive

    return residual_archive, f26_inflate, native_binding


def bind_split_symbols(library: Any) -> Any:
    """Declare the ddm_wc2c split entry points on an already-loaded CDLL.

    The shipping ``_load_library`` binds only the fused surface.  Declaring the
    split symbols here rather than editing that function keeps the proof
    independent of the staged runtime tree: the same library object drives both
    paths, so a fused-vs-split comparison cannot be confounded by two loads.
    """
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    model_p = ctypes.c_void_p

    library.f26_hpac_frame_begin.argtypes = [
        ctypes.c_void_p, model_p, u8p, ctypes.c_int32, u8p
    ]
    library.f26_hpac_frame_begin.restype = ctypes.c_int
    library.f26_hpac_group_logits.argtypes = [
        ctypes.c_void_p, model_p, u8p, ctypes.c_int32, f32p, i32p, i32p
    ]
    library.f26_hpac_group_logits.restype = ctypes.c_int
    library.f26_hpac_group_commit.argtypes = [
        ctypes.c_void_p, model_p, ctypes.c_int32, u8p, u8p
    ]
    library.f26_hpac_group_commit.restype = ctypes.c_int
    library.f26_hpac_group_output_count.argtypes = [model_p, ctypes.c_int32]
    library.f26_hpac_group_output_count.restype = ctypes.c_int32
    library.f26_hpac_timing_reset.argtypes = []
    library.f26_hpac_timing_reset.restype = None
    library.f26_hpac_dispatch_path.argtypes = []
    library.f26_hpac_dispatch_path.restype = ctypes.c_char_p
    return library


# The five slots f26_hpac_last_timing accumulates, in order.  Read-only telemetry
# that cannot change a decoded byte, so it is default-ON: a default-off gauge is
# a gauge nobody reads.
NATIVE_TIMING_SLOTS = (
    "prepare_frame_context",
    "conv_state_reset",
    "group_model",
    "group_corrected_rows",
    "group_commit_and_conv_update",
)


def read_native_timing(library: Any) -> dict[str, float]:
    buffer = (ctypes.c_double * 5)()
    library.f26_hpac_last_timing(buffer)
    return {name: float(buffer[index]) for index, name in enumerate(NATIVE_TIMING_SLOTS)}


def _pointer(array: np.ndarray, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def decode_split_native(
    candidate: Path,
    frames: int,
    library_path: Path,
    token_output: Path | None = None,
) -> dict[str, Any]:
    """Decode a frame prefix (or the full field) through the split native path."""
    residual_archive, f26_inflate, native_binding = _import_candidate(candidate)
    import torch
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator

    torch.set_num_threads(4)
    device = torch.device("cpu")

    archive_path = candidate / "archive.zip"
    renderer_dir = candidate / "cpr1"
    parts = residual_archive.read_residual_archive(archive_path)
    if parts.table is None:
        raise Wc2cSplitError("archive carries no residual correction table")
    renderer = f26_inflate._load_renderer(renderer_dir)

    setup_started = time.perf_counter()
    base_hpac = residual_archive.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    sparse = residual_archive._sparse_class(renderer_dir)(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    optimize_sparse_evaluator(sparse)
    buffers = native_binding._build_model_buffers(parts, renderer, sparse)
    library = bind_split_symbols(native_binding._load_library(library_path))
    setup_seconds = time.perf_counter() - setup_started

    height = int(renderer.EVAL_H)
    width = int(renderer.EVAL_W)
    plane = height * width
    total_frames = min(int(frames), int(renderer.N))
    groups = int(buffers.native.groups)

    rc64_library = os.environ.get("CPR1_RC64_LIBRARY")
    if not rc64_library:
        raise Wc2cSplitError("CPR1_RC64_LIBRARY is required")
    coder = residual_archive.NativeDecoder(Path(rc64_library), parts.token_stream)

    payload = np.frombuffer(parts.token_stream, dtype=np.uint8).copy()
    native_decoder = library.f26_rc64_create(
        _pointer(payload, ctypes.c_uint8), payload.size
    )
    if not native_decoder:
        raise Wc2cSplitError("native RC64 decoder allocation failed")

    corrector = FreeCorrector(plane)
    model_ref = ctypes.byref(buffers.native)

    output_counts = [
        int(library.f26_hpac_group_output_count(model_ref, group))
        for group in range(groups)
    ]
    if min(output_counts) <= 0:
        raise Wc2cSplitError("native reported a non-positive group output count")
    widest = max(output_counts)
    corrected_buffer = np.empty((widest, 5), dtype=np.float32)
    predicted_buffer = np.empty(widest, dtype=np.int32)
    flat_buffer = np.empty(widest, dtype=np.int32)

    # The group plans the pure-Python loop uses, gathered once so the native
    # flat-position order can be checked against them on the first frame.  This
    # is a BOUNDARY check, not a per-group one: it runs on frame 0 only, because
    # the loop executes 114,000 times and anything per-group is a budget item.
    expected_flat = []
    for mask in renderer.group_masks(device):
        expected_flat.append(
            np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int32)
        )
    if len(expected_flat) != groups:
        raise Wc2cSplitError(
            f"native reports {groups} groups; the renderer plans {len(expected_flat)}"
        )

    tokens = np.zeros((total_frames, height, width), dtype=np.uint8)
    previous = np.zeros(plane, dtype=np.uint8)
    current = np.zeros(plane, dtype=np.uint8)
    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    precision = renderer.HPAC_LOGIT_PRECISION

    library.f26_hpac_timing_reset()
    decode_started = time.perf_counter()
    for frame in range(total_frames):
        if frame:
            boundary = residual_archive._boundary_buckets(
                previous.reshape(height, width)
            ).reshape(-1)
        else:
            boundary = np.full(plane, 4, dtype=np.uint8)
        boundary = np.ascontiguousarray(boundary, dtype=np.uint8)
        corrector.begin_frame(boundary)

        status = library.f26_hpac_frame_begin(
            native_decoder,
            model_ref,
            _pointer(previous, ctypes.c_uint8),
            ctypes.c_int32(frame),
            _pointer(current, ctypes.c_uint8),
        )
        if status:
            raise Wc2cSplitError(f"f26_hpac_frame_begin failed with status {status}")

        for group in range(groups):
            count = output_counts[group]
            status = library.f26_hpac_group_logits(
                native_decoder,
                model_ref,
                _pointer(boundary, ctypes.c_uint8),
                ctypes.c_int32(group),
                _pointer(corrected_buffer, ctypes.c_float),
                _pointer(predicted_buffer, ctypes.c_int32),
                _pointer(flat_buffer, ctypes.c_int32),
            )
            if status:
                raise Wc2cSplitError(
                    f"f26_hpac_group_logits(frame={frame}, group={group}) "
                    f"failed with status {status}"
                )
            corrected = corrected_buffer[:count]
            predicted = predicted_buffer[:count].astype(np.int64)
            flat_positions = flat_buffer[:count]
            if frame == 0 and not np.array_equal(flat_positions, expected_flat[group]):
                raise Wc2cSplitError(
                    f"native flat-position order differs from the renderer plan "
                    f"at group {group}"
                )

            corrected_digest.update(
                np.ascontiguousarray(corrected, dtype="<f4").tobytes()
            )
            probability = residual_archive._probability_table(corrected, precision)
            cdf_digest.update(
                np.ascontiguousarray(probability, dtype="<f4").tobytes()
            )
            state = corrector.group_state(probability, predicted, flat_positions)
            symbols = coder.decode(corrector.coding_row(state)).astype(np.int64)
            corrector.observe(state, symbols)

            commit = np.ascontiguousarray(symbols, dtype=np.uint8)
            status = library.f26_hpac_group_commit(
                native_decoder,
                model_ref,
                ctypes.c_int32(group),
                _pointer(commit, ctypes.c_uint8),
                _pointer(current, ctypes.c_uint8),
            )
            if status:
                raise Wc2cSplitError(
                    f"f26_hpac_group_commit(frame={frame}, group={group}) "
                    f"failed with status {status}"
                )

        tokens[frame] = current.reshape(height, width)
        corrector.end_frame(current)
        previous, current = current, previous
    decode_seconds = time.perf_counter() - decode_started

    library.f26_rc64_destroy(native_decoder)

    if token_output is not None:
        token_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = token_output.with_name(f".{token_output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(tokens.tobytes())
        os.replace(temporary, token_output)

    full_field = total_frames == int(renderer.N)
    report: dict[str, Any] = {
        "schema": "ddm_wc2c_native_split_decode.v1",
        "axis": "[macOS-CPU advisory]",
        "candidate": str(candidate.resolve()),
        "archive_sha256": _sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "native_library": str(library_path.resolve()),
        "native_library_sha256": _sha256_file(library_path),
        "rc64_library": str(Path(rc64_library).resolve()),
        "rc64_library_sha256": _sha256_file(Path(rc64_library)),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "frames": total_frames,
        "frames_total_in_archive": int(renderer.N),
        "full_field": full_field,
        "groups": groups,
        "iterations": total_frames * groups,
        "setup_seconds": setup_seconds,
        "decode_seconds": decode_seconds,
        "seconds_per_iteration": decode_seconds / (total_frames * groups),
        "decode_path": library.f26_hpac_dispatch_path().decode("ascii"),
        "native_stage_seconds": read_native_timing(library),
        "model_manifest_sha256": buffers.manifest_sha256,
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "decoder_bit_position": int(coder.bit_position),
        "token_bytes": int(tokens.size),
        "token_output": str(token_output) if token_output else None,
    }
    if full_field:
        checks = {
            key: {
                "expected": JG5_ANCHORS[key],
                "measured": report[key],
                "match": report[key] == JG5_ANCHORS[key],
            }
            for key in (
                "corrected_quantized_logit_sha256",
                "corrected_cdf_input_sha256",
                "decoded_token_sha256",
                "decoder_bit_position",
            )
        }
        report["identity_vs_jg5_t4_receipt"] = checks
        report["identity_verdict"] = (
            "PASS" if all(item["match"] for item in checks.values()) else "REFUSE"
        )
    else:
        report["identity_vs_jg5_t4_receipt"] = None
        report["identity_verdict"] = "PREFIX_SCOPE_ONLY"
        report["verdict_scope"] = (
            "frame prefix; digests are prefix digests and cannot be compared to the "
            "n600 receipt anchors"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--token-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = decode_split_native(
        candidate=args.candidate,
        frames=args.frames,
        library_path=args.library,
        token_output=args.token_output,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["identity_verdict"] != "REFUSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
