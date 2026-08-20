#!/usr/bin/env python3
"""ddm_wc2c - the shipping split token decoder: native integer model, numpy corrector.

STAGED AS ``runtime/f26_split_token_decoder.py`` inside the candidate runtime
tree by ``experiments/ddm_wc2c_stage_native_split_runtime.py``.  This file is the
repo-canonical source; the staged copy differs only by the recorded
package-relative import rewrite, and the stager asserts the rewrite count so a
silent divergence fails the build instead of shipping.

WHAT IT REPLACES.  ``runtime/residual_archive.decode_production_tokens`` runs the
whole token stage in Python on top of a torch sparse evaluator.  On the shipping
axis that stage is 1,341.540 s of a 1,419.900 s inflate [contest-CUDA T4, jg5],
against a CUDA residual of [822, 1302] s -- a REFUSE.  This module keeps every
decoded value identical and moves only the integer HPAC model into C.

WHAT IT DELIBERATELY DOES NOT DO.  It does not touch the probability table, the
free probability corrector, or the RC64 coder.  Those stay in the exact numpy and
ctypes code the shipping receipt was produced with.  The corrector is float64 on
its decision path and its identity depends on a hand-fixed summation order
(``free_corrector.py:266-279``); the measured cost of getting that wrong is
S = 27.83, a desynchronised decoder.  The integer model has no such hazard, which
is precisely why it is the half that was lowered.

RESUMABILITY, stated exactly.  Resume granularity is the WHOLE TOKEN STAGE -- the
same granularity the shipping python path already has through ``f26_inflate``'s
token checkpoint -- and NOT per frame.  The reason is measured, not chosen:
``runtime/entropy/rc64_backend.c`` exports only create / decode / destroy /
bit_position, with no state save or restore, so the coder cannot resume
mid-stream.  Per-frame resume therefore requires an RC64 state pair added to that
backend plus a proof that adding it changed no decoded byte -- a separate landing
with its own burden.  It is recorded here as an OWED item rather than claimed.
The rolling field is written to a memmap so a completed stage survives a later
crash in the render.

AXIS.  Emits no score.  The caller's exact eval is the only authority.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class SplitTokenDecodeError(RuntimeError):
    """A precondition, contract, or resume invariant of the split decode failed."""


def _pointer(array: np.ndarray, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def bind_split_symbols(library: Any) -> Any:
    """Declare the split entry points on an already-loaded native library."""
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
    library.f26_hpac_dispatch_path.argtypes = []
    library.f26_hpac_dispatch_path.restype = ctypes.c_char_p
    library.f26_hpac_thread_count.argtypes = []
    library.f26_hpac_thread_count.restype = ctypes.c_int32
    return library


def decode_split_tokens(
    parts: Any,
    runtime: Any,
    code_dir: Path,
    device: Any,
    *,
    checkpoint_dir: Path,
    frame_limit: int | None = None,
) -> tuple[Any, dict[str, object]]:
    """Decode the full F26 token field through the split native path.

    Signature and return shape match
    ``runtime.residual_archive.decode_production_tokens`` so the caller in
    ``f26_inflate`` swaps one call, not a pipeline.

    ``frame_limit`` preserves the advisory prefix-inflation path that
    ``f26_inflate`` routed exclusively through the native decoder
    (``"advisory prefix inflation requires the resumable native token path"``).
    Dropping it would have silently decoded all 600 frames when a prefix was
    asked for -- a behaviour regression that no test would have caught because
    the identity proof only ever runs the full field.

    ``device`` is accepted for signature compatibility and DELIBERATELY not used
    to place the token model: this path is CPU-native by construction.  That is
    the point rather than an oversight.  On the T4 the shipping python loop makes
    two host<->device round trips plus one small kernel launch per group, 114,000
    times, and its measured 11.77 ms per iteration is SLOWER than the same loop
    on a local CPU.  The report records ``token_device: "cpu"`` so a run on a
    CUDA box cannot be mistaken for a GPU token decode.
    """
    import torch

    from .f26_hpac_native import _build_model_buffers, _load_library
    from .free_corrector import FreeCorrector
    from .hpac_inference import optimize_sparse_evaluator
    from .ihs2 import materialize_ihs1
    from .residual_archive import (
        NativeDecoder,
        ResidualArchiveError,
        _boundary_buckets,
        _probability_table,
        _sparse_class,
    )

    library_text = os.environ.get("F26_HPAC_NATIVE_LIBRARY")
    if not library_text:
        raise SplitTokenDecodeError("F26_HPAC_NATIVE_LIBRARY is required")
    library_path = Path(library_text).resolve()
    if not library_path.is_file():
        raise SplitTokenDecodeError(f"native library does not exist: {library_path}")
    rc64_text = os.environ.get("CPR1_RC64_LIBRARY")
    if not rc64_text:
        raise ResidualArchiveError("RC64 decoding requires CPR1_RC64_LIBRARY")

    started = time.time()
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    # The torch model is built only to derive the native buffers; it never
    # evaluates a frame on this path.  Kept on CPU because the buffer extraction
    # reads plain numpy and a device round trip would buy nothing.
    model = runtime.load_hpac(base_hpac, torch.device("cpu"))
    sparse = _sparse_class(code_dir)(model, runtime.EVAL_H, runtime.EVAL_W)
    optimize_sparse_evaluator(sparse)
    buffers = _build_model_buffers(parts, runtime, sparse)
    library = bind_split_symbols(_load_library(library_path))
    del model, sparse

    height = int(runtime.EVAL_H)
    width = int(runtime.EVAL_W)
    plane = height * width
    total_frames = int(runtime.N if frame_limit is None else frame_limit)
    if total_frames <= 0 or total_frames > int(runtime.N):
        raise SplitTokenDecodeError("frame_limit is outside the real n600 field")
    groups = int(buffers.native.groups)
    model_ref = ctypes.byref(buffers.native)

    coder = NativeDecoder(Path(rc64_text), parts.token_stream)
    payload = np.frombuffer(parts.token_stream, dtype=np.uint8).copy()
    native_decoder = library.f26_rc64_create(
        _pointer(payload, ctypes.c_uint8), payload.size
    )
    if not native_decoder:
        raise SplitTokenDecodeError("native RC64 decoder allocation failed")

    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    token_path = checkpoint_dir / "tokens_rolling.u8"
    if token_path.exists():
        token_path.unlink()
    tokens = np.memmap(
        token_path, mode="w+", dtype=np.uint8, shape=(total_frames, height, width)
    )
    corrector = FreeCorrector(plane)
    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    previous = np.zeros(plane, dtype=np.uint8)

    output_counts = [
        int(library.f26_hpac_group_output_count(model_ref, group))
        for group in range(groups)
    ]
    if min(output_counts) <= 0:
        raise SplitTokenDecodeError("native reported a non-positive group output count")
    widest = max(output_counts)

    # The flat-position order the corrector's context indexing assumes.  The
    # native side supplies logits AND positions together, so an internal
    # reordering would stay self-consistent for the model while silently
    # feeding the corrector the wrong per-pixel context -- a wrong-field bug
    # with no exception.  Checked on frame 0 only: the loop runs 114,000 times
    # and nothing per-group belongs in a stage on a wall-clock knife edge.
    expected_flat = [
        np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int32)
        for mask in runtime.group_masks(torch.device("cpu"))
    ]
    if len(expected_flat) != groups:
        raise SplitTokenDecodeError(
            f"native reports {groups} groups; the renderer plans {len(expected_flat)}"
        )
    corrected_buffer = np.empty((widest, 5), dtype=np.float32)
    predicted_buffer = np.empty(widest, dtype=np.int32)
    flat_buffer = np.empty(widest, dtype=np.int32)
    current = np.zeros(plane, dtype=np.uint8)
    precision = runtime.HPAC_LOGIT_PRECISION

    for frame in range(total_frames):
        boundary = (
            _boundary_buckets(previous.reshape(height, width)).reshape(-1)
            if frame
            else np.full(plane, 4, dtype=np.uint8)
        )
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
            raise SplitTokenDecodeError(f"frame_begin({frame}) status {status}")

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
                raise SplitTokenDecodeError(
                    f"group_logits(frame={frame}, group={group}) status {status}"
                )
            if frame == 0 and not np.array_equal(
                flat_buffer[:count], expected_flat[group]
            ):
                raise SplitTokenDecodeError(
                    f"native flat-position order differs from the renderer plan "
                    f"at group {group}"
                )
            corrected = corrected_buffer[:count]
            corrected_digest.update(
                np.ascontiguousarray(corrected, dtype="<f4").tobytes()
            )
            probability = _probability_table(corrected, precision)
            cdf_digest.update(np.ascontiguousarray(probability, dtype="<f4").tobytes())
            state = corrector.group_state(
                probability, predicted_buffer[:count].astype(np.int64), flat_buffer[:count]
            )
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
                raise SplitTokenDecodeError(
                    f"group_commit(frame={frame}, group={group}) status {status}"
                )

        tokens[frame] = current.reshape(height, width)
        corrector.end_frame(current)
        previous, current = current, previous

    tokens.flush()
    elapsed = time.time() - started
    field = np.ascontiguousarray(np.asarray(tokens))
    library.f26_rc64_destroy(native_decoder)

    return torch.from_numpy(field), {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "adapter_sha256": "",
        "token_codec": "rc64",
        "decoder_bit_position": int(coder.bit_position),
        "token_decoder": "native-hpac-split",
        "token_device": "cpu",
        "frames_decoded": total_frames,
        "decode_path": library.f26_hpac_dispatch_path().decode("ascii"),
        "decode_threads": int(library.f26_hpac_thread_count()),
        "native_library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
    }
