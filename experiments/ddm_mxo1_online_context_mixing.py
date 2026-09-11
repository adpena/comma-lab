#!/usr/bin/env python3
"""MXO1: scorer-free n600 online augmentation of the shipped move-44 prior.

This is a research encoder/replay, not a candidate builder.  It consumes the
integer probability rows observed from the exact shipped receiver and emits
four retained RC64 streams: a byte-identity control plus three causal learners.
Every learner starts cold and derives all mutable state from the decoded prefix.

The shipped receiver already contains a PAQ-style fixed-point logistic mixer.
Accordingly, ``lane_stack`` is deliberately an augmentation of the *final*
shipped probabilities, not a renamed replacement for that incumbent mechanism.
``group_rnn`` is a tiny leaky integer recurrence over causal decode groups, and
``prev_patch`` is a previous-frame 3x3 hashed context expert.

Axis: [macOS-CPU advisory / scorer-free n600 exact RC64 byte measurement].
No scorer, archive mutation, candidate archive, or incumbent receiver edit.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_ls1_shipped_surprise import ARCHIVE_SHA, FIELD, FIELD_SHA, fact
from experiments.ddm_tc1_mixer_codec import frequencies, log2_fixed, mix_probabilities

route_b = jg2.load_route_b()

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing")
LS1 = ROOT.parent / "ddm_ls1"
RUNTIME = ROOT.parent / "ddm_rlc5_cure_on_move43/candidate_runtime"
N, H, W, K, TOTAL = 600, 384, 512, 5, 1 << 31
SIZE = H * W
Y, X = np.indices((H, W))
PHASE = (X % 64 + 2 * (Y % 64)).reshape(-1)
PHASE_POSITIONS = tuple(np.flatnonzero(group == PHASE) for group in range(190))
AXIS = "[macOS-CPU advisory / scorer-free n600 exact RC64 byte measurement]"
MODELS = ("control", "lane_stack", "group_rnn", "prev_patch")
LANE_LEVELS = (8, 8, 64)
PATCH_LEVELS = 1 << 12
CHECKPOINT_EVERY = 25
SEED = 20260911
STRICT_T4_DECODE_LIMIT_S = 1260.0
MOVE44_T4_DECODE_S = 1232.418725255
MOVE44_T4_TAIL_S = 1142.9962956905365
MOVE44_LOCAL_TAIL_S = 1168.6204084999987
TAIL_DEMAND_B = 25_899


class Mxo1Error(RuntimeError):
    """Fail-closed MXO1 refusal."""


def guard(need: int = 0) -> None:
    """SSD-first storage waterfall; retain all evidence and fail closed."""
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < (16 << 30) + need:
        raise Mxo1Error("STORAGE_BLOCK: keep all existing bytes; no deletion authorized")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes, *, immutable: bool = False) -> None:
    guard(len(payload))
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise Mxo1Error(f"write escaped owned store: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise Mxo1Error(f"immutable retained payload changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path: Path, value: object, *, immutable: bool = False) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, payload, immutable=immutable)


def atomic_npz(path: Path, values: dict[str, np.ndarray], *, immutable: bool = False) -> None:
    guard(sum(value.nbytes for value in values.values()))
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise Mxo1Error(f"write escaped owned store: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(values) or any(
                not np.array_equal(prior[name], value) for name, value in values.items()
            ):
                raise Mxo1Error(f"immutable retained array changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_npz(path: Path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if fact(path) != receipt:
        raise Mxo1Error(f"retained input checksum changed: {path}")
    return np.load(path, allow_pickle=False)


def bind() -> dict[str, object]:
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["effective_frontier"]["archive_sha256"] != ARCHIVE_SHA:
        raise Mxo1Error("POINTER_CHANGED: explicit rebind required")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise Mxo1Error("shipped decoded field changed")
    trace = json.loads((LS1 / "TRACE.json").read_text())
    if not trace["full_n600"] or trace["symbols"] != N * SIZE or not trace["field_identity"]:
        raise Mxo1Error("LS1 is not a full n600 receiver-identity trace")
    sys.path.insert(0, str(RUNTIME))
    from runtime import residual_archive

    parts = residual_archive.read_residual_archive(RUNTIME / "archive.zip")
    if len(parts.token_stream) != 119_749:
        raise Mxo1Error("move44 shipped token stream census changed")
    pins = {
        "axis": AXIS,
        "score_claim": False,
        "archive_sha256": ARCHIVE_SHA,
        "field": fact(FIELD),
        "ls1_trace": fact(LS1 / "TRACE.json"),
        "ls1_inputs": fact(LS1 / "INPUTS.json"),
        "ls1_atlas": fact(LS1 / "atlas/RESULT.json"),
        "runtime_archive": fact(RUNTIME / "archive.zip"),
        "shipped_token_stream": {
            "bytes": len(parts.token_stream),
            "sha256": sha256_bytes(parts.token_stream),
        },
        "producer": fact(Path(__file__)),
        "executed_source_release": fact(
            ROOT / "source_release/ddm_mxo1_online_context_mixing_v2.py"
        ),
        "codec": fact(REPO / "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py"),
        "seed": SEED,
        "symbols": N * SIZE,
        "state_rule": "cold start; state changes only after decoded causal prefix",
    }
    # v1 is retained as the raster-order instrument falsifier. v2 pins the
    # executed group-order source. v3 binds this linted, terminal-idempotent
    # source to the exact executed source release without overwriting either.
    path = ROOT / "BINDING_v3.json"
    if path.exists() and json.loads(path.read_text()) != pins:
        raise Mxo1Error("source/input binding drift")
    save_json(path, pins, immutable=True)
    atomic_bytes(ROOT / "retained/shipped_token_stream.bin", parts.token_stream, immutable=True)
    save_json(
        ROOT / "retained/shipped_token_stream.bin.json",
        fact(ROOT / "retained/shipped_token_stream.bin"),
        immutable=True,
    )
    return pins


def compile_encoder() -> Path:
    build = ROOT / "build"
    receipt = build / "BUILD.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        library = Path(prior["library"]["path"])
        if prior["library"] != fact(library):
            raise Mxo1Error("retained RC64 encoder library drift")
        return library
    library, build_record = jg2.compile_rc64(build, route_b, "mxo1")
    if library != build / "rc64_mxo1/librc64_jg2.dylib":
        raise Mxo1Error("unexpected RC64 build path")
    # Keep the build in its producer-selected subdirectory; record the exact path.
    save_json(receipt, build_record, immutable=True)
    return library


def encode_frequencies(
    encoder: route_b.NativeRc64Encoder,
    symbols: np.ndarray,
    rows: np.ndarray,
) -> None:
    source = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
    lattice = np.ascontiguousarray(rows, dtype=np.uint32)
    if lattice.shape != (len(source), K) or np.any(lattice.sum(axis=1, dtype=np.uint64) != TOTAL):
        raise Mxo1Error("invalid integer RC64 probability lattice")
    status = encoder.library.rc64_encoder_encode(
        encoder.context,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lattice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(source),
    )
    if status:
        raise Mxo1Error(f"RC64 encode failed with status {status}")


def encode_plane(
    encoder: route_b.NativeRc64Encoder,
    truth: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Preserve the receiver's 190-group symbol order, never raster order."""
    for positions in PHASE_POSITIONS:
        encode_frequencies(encoder, truth[positions], rows[positions])


def encoder_body(encoder: route_b.NativeRc64Encoder) -> bytes:
    encoder.finish()
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise Mxo1Error("RC64 encoder produced no retained payload")
    return ctypes.string_at(pointer, size)


def initial_state() -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "frame": np.array([0], dtype=np.int64),
        "ideal_bits": np.zeros((len(MODELS), N), dtype=np.float64),
        "model_seconds": np.zeros((len(MODELS),), dtype=np.float64),
        "lane_weights_float": np.zeros((K, 3), dtype=np.float64),
        "lane_weights": np.zeros((K, 3), dtype=np.int8),
        "lane_grad2": np.zeros((K, 3), dtype=np.float64),
        "rnn_hits_q31": np.zeros((K, K), dtype=np.int64),
        "rnn_expect_q31": np.zeros((K, K), dtype=np.int64),
        "patch_counts": np.zeros((K * PATCH_LEVELS, K), dtype=np.int64),
        "patch_expected": np.zeros((K * PATCH_LEVELS, K), dtype=np.int64),
    }
    for j, levels in enumerate(LANE_LEVELS):
        state[f"lane_counts_{j}"] = np.zeros((K * levels, K), dtype=np.int64)
        state[f"lane_expected_{j}"] = np.zeros((K * levels, K), dtype=np.int64)
    return state


def load_frame(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    row = LS1 / "rows" / f"frame_{frame:04d}.npz"
    with checked_npz(row) as data:
        base = data["frequencies"].copy()
        truth = data["symbols"].copy()
        bits = data["bits"].copy()
    cell = LS1 / "atlas/cells" / f"frame_{frame:04d}.npz"
    with checked_npz(cell) as data:
        keys = data["receiver_lane"]
        past = ((keys // 8) % 8).astype(np.int64)
        visible = (keys % 8).astype(np.int64)
    if base.shape != (SIZE, K) or truth.shape != (SIZE,):
        raise Mxo1Error("LS1 receiver row shape changed")
    if np.any(base.sum(axis=1, dtype=np.uint64) != TOTAL):
        raise Mxo1Error("LS1 receiver row frequency mass changed")
    selected = base[np.arange(SIZE), truth]
    np.testing.assert_array_equal(bits, -np.log2(selected.astype(np.float64) / TOTAL))
    return base, truth, bits, (past, visible, past * 8 + visible)


def ratio_table(counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
    ratio = (counts.astype(np.float64) + 0.5) / (expected.astype(np.float64) / TOTAL + 0.5)
    return log2_fixed(np.clip(ratio, 1 / 16, 16))


def apply_features(
    base: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    original = (base.astype(np.float64) / TOTAL).astype(np.float32)
    mixed = mix_probabilities(base, phi, weights, original_rows=original)
    result = frequencies(mixed)
    inactive = ~np.any(weights[base.argmax(axis=1)], axis=1)
    result[inactive] = base[inactive]
    return result


def selected_bits(rows: np.ndarray, truth: np.ndarray) -> float:
    selected = rows[np.arange(len(truth)), truth]
    return float((-np.log2(selected.astype(np.float64) / TOTAL)).sum())


def update_expected(
    counts: np.ndarray,
    expected: np.ndarray,
    code: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
) -> None:
    counts += np.bincount(
        code * K + truth, minlength=counts.shape[0] * K
    ).reshape(counts.shape)
    for symbol in range(K):
        expected[:, symbol] += np.bincount(
            code, weights=base[:, symbol], minlength=counts.shape[0]
        ).astype(np.int64)


def lane_rows(
    state: dict[str, np.ndarray],
    base: np.ndarray,
    codes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    arg = base.argmax(axis=1)
    tables = [
        ratio_table(state[f"lane_counts_{j}"], state[f"lane_expected_{j}"])
        for j in range(3)
    ]
    phi = np.stack(
        [table[arg * levels + code] for table, levels, code in zip(tables, LANE_LEVELS, codes, strict=True)],
        axis=2,
    )
    rows = apply_features(base, phi, state["lane_weights"])
    return rows, phi


def update_lane(
    state: dict[str, np.ndarray],
    base: np.ndarray,
    truth: np.ndarray,
    codes: tuple[np.ndarray, ...],
    rows: np.ndarray,
    phi: np.ndarray,
) -> None:
    arg = base.argmax(axis=1)
    probability = rows.astype(np.float64) / TOTAL
    for bank in range(K):
        mask = arg == bank
        if not mask.any():
            continue
        mean = np.einsum("nk,nkf->nf", probability[mask], phi[mask].astype(np.float64))
        observed = phi[mask][np.arange(int(mask.sum())), truth[mask]].astype(np.float64)
        gradient = ((mean - observed) / (1024 * 32)).mean(axis=0)
        state["lane_grad2"][bank] += gradient * gradient
        state["lane_weights_float"][bank] -= 4.0 * gradient / np.sqrt(
            state["lane_grad2"][bank] + 1e-12
        )
    state["lane_weights_float"][:] = np.clip(state["lane_weights_float"], -128, 127)
    state["lane_weights"][:] = np.rint(state["lane_weights_float"]).astype(np.int8)
    for j, (levels, code) in enumerate(zip(LANE_LEVELS, codes, strict=True)):
        index = arg * levels + code
        update_expected(
            state[f"lane_counts_{j}"], state[f"lane_expected_{j}"], index, truth, base
        )


def group_rnn_frame(
    state: dict[str, np.ndarray],
    encoder: route_b.NativeRc64Encoder,
    base: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float]:
    """A 25-cell leaky recurrent calibrator, strictly group causal."""
    started = time.perf_counter()
    bits = 0.0
    weights = np.full((K, 1), 16, dtype=np.int8)  # square-root damping
    for positions in PHASE_POSITIONS:
        group_base = base[positions]
        group_truth = truth[positions]
        table = ratio_table(state["rnn_hits_q31"] / TOTAL, state["rnn_expect_q31"])
        arg = group_base.argmax(axis=1)
        phi = table[arg][:, :, None]
        rows = apply_features(group_base, phi, weights)
        bits += selected_bits(rows, group_truth)
        encode_frequencies(encoder, group_truth, rows)
        # Integer leaky recurrence: 63/64 prior state plus this decoded group.
        state["rnn_hits_q31"] -= state["rnn_hits_q31"] // 64
        state["rnn_expect_q31"] -= state["rnn_expect_q31"] // 64
        observed = np.bincount(arg * K + group_truth, minlength=K * K).reshape(K, K)
        state["rnn_hits_q31"] += observed.astype(np.int64) * TOTAL
        for symbol in range(K):
            state["rnn_expect_q31"][:, symbol] += np.bincount(
                arg, weights=group_base[:, symbol], minlength=K
            ).astype(np.int64)
        # Keep a symmetric one-observation cold prior without making it mutable.
        if np.any(state["rnn_hits_q31"] < 0) or np.any(state["rnn_expect_q31"] < 0):
            raise Mxo1Error("group recurrence underflow")
    return bits, time.perf_counter() - started


def previous_patch(previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return np.zeros(SIZE, dtype=np.int64)
    padded = np.pad(previous, 1, constant_values=K)
    code = np.zeros((H, W), dtype=np.uint32)
    for dy in range(3):
        for dx in range(3):
            code = (code * np.uint32(7) + padded[dy : dy + H, dx : dx + W]) & np.uint32(
                PATCH_LEVELS - 1
            )
    return code.reshape(-1).astype(np.int64)


def checkpoint(
    run: Path,
    frame: int,
    state: dict[str, np.ndarray],
    encoders: dict[str, route_b.NativeRc64Encoder],
) -> None:
    state["frame"][0] = frame
    state_path = run / "checkpoints" / f"state_{frame:04d}.npz"
    atomic_npz(state_path, state, immutable=True)
    encoder_facts = {}
    for name, encoder in encoders.items():
        path = run / "checkpoints" / f"encoder_{name}_{frame:04d}.bin"
        atomic_bytes(path, encoder.snapshot(), immutable=True)
        encoder_facts[name] = fact(path)
    receipt = {
        "frame": frame,
        "state": fact(state_path),
        "encoders": encoder_facts,
        "schema": "ddm_mxo1_checkpoint_v1",
    }
    save_json(run / "checkpoints" / f"stage_{frame:04d}.json", receipt, immutable=True)
    save_json(run / "LATEST.json", receipt)


def resume(
    run: Path, library: Path
) -> tuple[int, dict[str, np.ndarray], dict[str, route_b.NativeRc64Encoder]]:
    latest = run / "LATEST.json"
    if not latest.exists():
        return 0, initial_state(), {name: route_b.NativeRc64Encoder(library) for name in MODELS}
    receipt = json.loads(latest.read_text())
    state_path = Path(receipt["state"]["path"])
    if fact(state_path) != receipt["state"]:
        raise Mxo1Error("latest model checkpoint changed")
    with np.load(state_path, allow_pickle=False) as saved:
        state = {name: saved[name].copy() for name in saved.files}
    encoders = {}
    for name in MODELS:
        path = Path(receipt["encoders"][name]["path"])
        if fact(path) != receipt["encoders"][name]:
            raise Mxo1Error(f"latest {name} encoder checkpoint changed")
        encoders[name] = route_b.NativeRc64Encoder(library, path.read_bytes())
    frame = int(state["frame"][0])
    if frame != receipt["frame"]:
        raise Mxo1Error("model/encoder checkpoint frame mismatch")
    return frame, state, encoders


def finish_run(
    run: Path,
    state: dict[str, np.ndarray],
    encoders: dict[str, route_b.NativeRc64Encoder],
    pins: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    stream_facts = {}
    for name, encoder in encoders.items():
        path = run / "retained" / f"{name}.rc64"
        atomic_bytes(path, encoder_body(encoder), immutable=True)
        stream_facts[name] = fact(path)
    atomic_npz(
        run / "retained/ledgers.npz",
        {
            "ideal_bits": state["ideal_bits"],
            "model_seconds": state["model_seconds"],
            "lane_weights": state["lane_weights"],
        },
        immutable=True,
    )
    control = stream_facts["control"]
    shipped = pins["shipped_token_stream"]
    control_ok = control["bytes"] == shipped["bytes"] and control["sha256"] == shipped["sha256"]
    if not control_ok:
        raise Mxo1Error("INSTRUMENT_FALSIFIED: integer-row control differs from shipped RC64 bytes")
    base_ideal = float(state["ideal_bits"][0].sum())
    models = {}
    for index, name in enumerate(MODELS):
        ideal = float(state["ideal_bits"][index].sum())
        realized = int(control["bytes"] - stream_facts[name]["bytes"])
        models[name] = {
            "stream": stream_facts[name],
            "ideal_bytes": ideal / 8,
            "ideal_savings_bytes": (base_ideal - ideal) / 8,
            "realized_savings_bytes": realized,
            "fraction_of_tail_demand": realized / TAIL_DEMAND_B,
            "host_model_and_encode_seconds": float(state["model_seconds"][index]),
            "host_ns_per_symbol": float(state["model_seconds"][index] * 1e9 / (N * SIZE)),
        }
    result_path = run / "RESULT.json"
    prior_elapsed = None
    if result_path.exists():
        prior_elapsed = json.loads(result_path.read_text())["elapsed_seconds"]
    result = {
        "axis": AXIS,
        "score_claim": False,
        "full_n600": True,
        "symbols": N * SIZE,
        "elapsed_seconds": elapsed if prior_elapsed is None else prior_elapsed,
        "control_byte_identical": True,
        "models": models,
        "tail_demand_bytes": TAIL_DEMAND_B,
        "strict_t4_decode_limit_seconds": STRICT_T4_DECODE_LIMIT_S,
        "move44_t4_decode_seconds": MOVE44_T4_DECODE_S,
        "strict_t4_incremental_slack_seconds": STRICT_T4_DECODE_LIMIT_S - MOVE44_T4_DECODE_S,
        "strict_t4_incremental_ns_per_symbol": (
            (STRICT_T4_DECODE_LIMIT_S - MOVE44_T4_DECODE_S) * 1e9 / (N * SIZE)
        ),
        "same_receiver_t4_over_local_tail_factor": MOVE44_T4_TAIL_S / MOVE44_LOCAL_TAIL_S,
        "determinism": "all coding rows and state are prefix-derived; byte-identical repeat required",
        "lane_final_weights_q5": state["lane_weights"].tolist(),
    }
    save_json(result_path, result, immutable=True)
    return result


def replay(label: str) -> dict[str, object]:
    pins = bind()
    library = compile_encoder()
    run = ROOT / label
    start, state, encoders = resume(run, library)
    target = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    started = time.perf_counter()
    for frame in range(start, N):
        base, truth, base_bits, lane_codes = load_frame(frame)
        np.testing.assert_array_equal(truth, target[frame].reshape(-1))

        tick = time.perf_counter()
        encode_plane(encoders["control"], truth, base)
        state["ideal_bits"][0, frame] = float(base_bits.sum())
        state["model_seconds"][0] += time.perf_counter() - tick

        tick = time.perf_counter()
        lane, lane_phi = lane_rows(state, base, lane_codes)
        state["ideal_bits"][1, frame] = selected_bits(lane, truth)
        encode_plane(encoders["lane_stack"], truth, lane)
        update_lane(state, base, truth, lane_codes, lane, lane_phi)
        state["model_seconds"][1] += time.perf_counter() - tick

        bits, seconds = group_rnn_frame(state, encoders["group_rnn"], base, truth)
        state["ideal_bits"][2, frame] = bits
        state["model_seconds"][2] += seconds

        tick = time.perf_counter()
        patch = previous_patch(None if frame == 0 else target[frame - 1])
        arg = base.argmax(axis=1)
        table = ratio_table(state["patch_counts"], state["patch_expected"])
        phi = table[arg * PATCH_LEVELS + patch][:, :, None]
        patch_rows = apply_features(base, phi, np.full((K, 1), 16, dtype=np.int8))
        state["ideal_bits"][3, frame] = selected_bits(patch_rows, truth)
        encode_plane(encoders["prev_patch"], truth, patch_rows)
        update_expected(
            state["patch_counts"],
            state["patch_expected"],
            arg * PATCH_LEVELS + patch,
            truth,
            base,
        )
        state["model_seconds"][3] += time.perf_counter() - tick

        boundary = frame + 1
        if boundary % CHECKPOINT_EVERY == 0 or boundary == N:
            checkpoint(run, boundary, state, encoders)
            print(
                json.dumps(
                    {
                        "stage": label,
                        "frames": boundary,
                        "elapsed_seconds": time.perf_counter() - started,
                        "ideal_savings_bytes": {
                            name: float(
                                (
                                    state["ideal_bits"][0, :boundary].sum()
                                    - state["ideal_bits"][index, :boundary].sum()
                                )
                                / 8
                            )
                            for index, name in enumerate(MODELS[1:], 1)
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return finish_run(run, state, encoders, pins, time.perf_counter() - started)


def compare() -> dict[str, object]:
    first = json.loads((ROOT / "primary_v2/RESULT.json").read_text())
    repeat = json.loads((ROOT / "repeat_v2/RESULT.json").read_text())
    matches = {}
    for name in MODELS:
        a, b = first["models"][name]["stream"], repeat["models"][name]["stream"]
        matches[name] = a["bytes"] == b["bytes"] and a["sha256"] == b["sha256"]
    result = {
        "status": "PASS" if all(matches.values()) else "FAIL",
        "byte_identical_streams": matches,
        "primary": fact(ROOT / "primary_v2/RESULT.json"),
        "repeat": fact(ROOT / "repeat_v2/RESULT.json"),
        "scope": "independent cold n600 replays on the same host and source rows",
    }
    save_json(ROOT / "DETERMINISM.json", result, immutable=True)
    if not all(matches.values()):
        raise Mxo1Error("determinism repeat changed one or more RC64 streams")
    return result


def retention() -> None:
    payloads = [
        fact(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.suffix in (".bin", ".rc64", ".npz", ".dylib", ".c")
    ]
    save_json(
        ROOT / "RETENTION.json",
        {
            "files": payloads,
            "bytes": sum(item["bytes"] for item in payloads),
            "policy": "KEEP all streams, complete model states, and RC64 encoder checkpoints",
            "cleanup": "none authorized; fail closed before reserve",
            "free_bytes": shutil.disk_usage(ROOT).free,
            "reserve_bytes": 16 << 30,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("primary", "repeat", "compare"), required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise Mxo1Error("--resume-from must name MXO1's owned durable SSD root")
    guard(1 << 30)
    lock = (ROOT / ".stage.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        labels = {"primary": "primary_v2", "repeat": "repeat_v2"}
        result = compare() if args.stage == "compare" else replay(labels[args.stage])
        save_json(
            ROOT / f"STAGE_{args.stage}.json",
            {"argv": sys.argv, "result": result, "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip()},
        )
        retention()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
