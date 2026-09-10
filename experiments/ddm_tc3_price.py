#!/usr/bin/env python3
"""TC3 real n600 preparation, box-constrained calibration, and retained twins.

All results are scorer-free. No candidate is a score until MAIN's exact row.
Each stage rereads the live pointer. Resumes bind inputs, source and native code.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from experiments import ddm_tc3_geometry as geometry
from experiments import ddm_tc3_mixer as codec
from experiments import ddm_tc3_trace as ref

io, base = ref.jg2, codec.base
ROOT = ref.ROOT / "move39"
N, H, W, K = 600, 384, 512, 5
ORDER = np.argsort(base.GROUP, kind="stable")
POSITIONS = [ORDER[base.GROUP[ORDER] == g] for g in range(190)]
KEEP = 2.0**-14


def fact(path):
    return io.file_fact(Path(path))


def load_arrays(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if fact(path) != receipt:
        raise ValueError("retained array digest changed: " + str(path))
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_arrays(path, values):
    receipt = ref.arrays(path, values)
    ref.record(path.with_suffix(".json"), receipt)
    return receipt


def binding(stage, extra=None):
    inputs = ref.prepare()
    work = ROOT / stage
    value = {
        "inputs": fact(ROOT / "INPUTS.json"),
        "sources": [fact(Path(m.__file__)) for m in (ref, codec, base, geometry, io)],
        "native_wrapper": fact(io.ROUTE_B),
        "trace_binding": json.loads((ROOT / "trace/LATEST.json").read_text())["binding"],
        "producer": fact(Path(__file__)),
        "extra": extra,
        "archive": inputs["archive_sha256"],
        "field": inputs["field"],
        "seed": ref.SEED,
        "score_claim": False,
        "axis": ref.AXIS,
    }
    path = work / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("stage input/source binding changed: " + stage)
    ref.record(path, value)
    return value


def check_inherited(old, current):
    """Stage-specific extra inputs differ; the computation and field must not."""
    for key in current:
        if key != "extra" and old.get(key) != current[key]:
            raise ValueError("inherited stage binding changed: " + key)


def source_frame(frame, expected_trace_binding):
    path = ROOT / "trace/frames" / f"frame_{frame:04d}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if fact(path) != receipt["payload"] or receipt["source_field_sha256"] != ref.FIELD_SHA:
        raise ValueError("source trace frame custody changed")
    if receipt["binding"] != expected_trace_binding:
        raise ValueError("source trace producer/native lineage changed")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def prepare(stop):
    if not 0 < stop <= N:
        raise ValueError("invalid preparation frame boundary")
    pin = binding("prepare_v2")
    work = ROOT / "prepare_v2"
    weights = (ref.ROOT / "retained/tc1_weights.bin").read_bytes()
    a = bytes([1]) + weights + bytes(5)
    b = np.zeros((5, 8), dtype=np.int8)
    b[:, :7] = np.frombuffer(weights, dtype=np.int8).reshape(5, 7)
    mixers = [codec.LaneMixer(a), codec.LaneMixer(bytes([2]) + b.tobytes())]
    start = 0
    totals = np.zeros((2, 4))
    latest = work / "LATEST.json"
    if latest.exists():
        last = json.loads(latest.read_text())
        if last["binding"] != pin:
            raise ValueError("prepare resume binding changed")
        state = load_arrays(Path(last["state"]))
        start = int(state["frame"][0])
        totals = state["totals"]
        for j, mixer in enumerate(mixers):
            mixer.restore({k[3:]: v for k, v in state.items() if k.startswith(f"m{j}_")})
            if mixer.frame != start:
                raise ValueError("prepare mixer frame drift")
    field = io.load_tokens(ROOT / "field.u8")
    if start > stop:
        raise ValueError("requested preparation stop precedes checkpoint")
    started = time.monotonic()
    for frame in range(start, stop):
        ready = ROOT / "trace/frames" / f"frame_{frame:04d}.json"
        deadline = time.monotonic() + 2400
        while not ready.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("source frame did not arrive; previous prepare state retained")
            time.sleep(2)
        data = source_frame(frame, pin["trace_binding"])
        plane = field[frame]
        previous = None if frame == 0 else field[frame - 1]
        np.testing.assert_array_equal(data["tokens"], plane)
        truth = plane.reshape(-1)
        values = {"truth": truth.copy()}
        for j, mixer in enumerate(mixers):
            mixer.begin_frame()
            phi = np.empty((H * W, K, 1 if j == 0 else 8), dtype=np.int16)
            freq = np.empty((H * W, K), dtype=np.int64)
            for positions in POSITIONS:
                pp, ff, original = mixer.features(data["raw_rows"][positions], positions, plane, previous)
                if j == 0:
                    np.testing.assert_array_equal(original, data["rows"][positions])
                else:
                    np.testing.assert_array_equal(
                        base.mix_probabilities(ff, pp, mixer.weights, original), data["rows"][positions]
                    )
                phi[positions], freq[positions] = pp, ff
                mixer.observe(positions, truth[positions])
            values[f"phi{j}"] = phi
            values[f"freq{j}"] = freq.astype(np.uint32)
            values[f"bins{j}"] = mixer.bins.copy()
            bits = -np.log2(freq[np.arange(H * W), truth].astype(float) / base.TOTAL)
            keep = (1 - freq.max(axis=1).astype(float) / base.TOTAL >= KEEP) | (truth != freq.argmax(axis=1))
            values[f"keep{j}"] = keep
            totals[j] += [bits.sum(), bits[keep].sum(), bits[~keep].sum(), keep.sum()]
            mixer.end_frame(plane, previous)
        save_arrays(work / "frames" / f"frame_{frame:04d}.npz", values)
        state = {"frame": np.array([frame + 1]), "totals": totals.copy()}
        for j, mixer in enumerate(mixers):
            state.update({f"m{j}_" + k: v for k, v in mixer.snapshot().items()})
        path = work / "states" / f"state_{frame + 1:04d}.npz"
        save_arrays(path, state)
        ref.record(latest, {"binding": pin, "frame": frame + 1, "state": str(path)})
        print(
            json.dumps({"stage": "prepare_v2", "frame": frame + 1, "elapsed": time.monotonic() - started}), flush=True
        )
    return ref.record(
        work / ("RESULT.json" if stop == N else f"PARTIAL_{stop:04d}.json"),
        {
            "binding": pin,
            "frames": stop,
            "positions": stop * H * W,
            "totals": totals.tolist(),
            "full_n600": stop == N,
            "source_tc1_rows_exact": True,
        },
    )


def objective(variant):
    j = 0 if variant == "A" else 1
    F = 1 if j == 0 else 8
    work = ROOT / "fit" / variant
    source = REPO / "experiments/ddm_tc1_logistic_bound.c"
    generated = work / "logistic.c"
    ref.blob(generated, source.read_bytes().replace(b"#define F 7", f"#define F {F}".encode()))
    library = work / "liblogistic.dylib"
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(generated),
        "-o",
        str(library),
    ]
    if not library.exists():
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        ref.record(
            work / "BUILD.json",
            {"source": fact(source), "generated": fact(generated), "library": fact(library), "argv": command},
        )
    else:
        build = json.loads((work / "BUILD.json").read_text())
        if build != {"source": fact(source), "generated": fact(generated), "library": fact(library), "argv": command}:
            raise ValueError("logistic native build drift")
    chunks = {k: [] for k in ("phi", "logp", "truth", "group")}
    for frame in range(N):
        data = load_arrays(ROOT / "prepare_v2/frames" / f"frame_{frame:04d}.npz")
        keep = data[f"keep{j}"]
        freq = data[f"freq{j}"][keep]
        chunks["phi"].append(data[f"phi{j}"][keep])
        chunks["logp"].append(np.log(freq.astype(float) / base.TOTAL))
        chunks["truth"].append(data["truth"][keep])
        chunks["group"].append(freq.argmax(axis=1).astype(np.uint8))
    arrays = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in chunks.items()}
    chunks.clear()
    lib = ctypes.CDLL(str(library))

    def ptr(dtype):
        return np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")

    lib.tc1_objective.argtypes = [
        ctypes.c_size_t,
        ptr(np.float64),
        ptr(np.int16),
        ptr(np.uint8),
        ptr(np.uint8),
        ptr(np.float64),
        ctypes.c_int,
        ptr(np.float64),
        ptr(np.float64),
    ]
    lib.tc1_objective.restype = ctypes.c_double

    def call(weights, derivatives=True):
        grad = np.zeros((K, F))
        hess = np.zeros((K, F, F))
        weights = np.ascontiguousarray(weights, dtype=float)
        loss = lib.tc1_objective(
            len(arrays["truth"]),
            arrays["logp"],
            arrays["phi"],
            arrays["truth"],
            arrays["group"],
            weights,
            int(derivatives),
            grad,
            hess,
        )
        if not all(np.isfinite(v).all() for v in (np.asarray(loss), grad, hess)):
            raise ValueError("nonfinite logistic objective")
        return loss, grad, hess

    return call, arrays


def fit(variant):
    prepared = json.loads((ROOT / "prepare_v2/RESULT.json").read_text())
    trace = json.loads((ROOT / "trace/RESULT.json").read_text())
    if not prepared["full_n600"] or not trace["full_control_byte_identical"]:
        raise ValueError("full source control and preparation required")
    pin = binding("fit/" + variant, extra=fact(ROOT / "prepare_v2/RESULT.json"))
    check_inherited(prepared["binding"], pin)
    work = ROOT / "fit" / variant
    call, arrays = objective(variant)
    F = 1 if variant == "A" else 8
    weights = np.zeros((K, F))
    if variant == "B":
        weights[:, :7] = (
            np.frombuffer((ref.ROOT / "retained/tc1_weights.bin").read_bytes(), dtype=np.int8).reshape(K, 7) / 32
        )
    baseline = call(weights, False)[0]
    last = work / "STATE.json"
    start = 0
    if last.exists():
        state = json.loads(last.read_text())
        if state["binding"] != pin:
            raise ValueError("fit restart binding changed")
        start = state["next_iteration"]
        weights = np.array(state["weights"])
    low, high = -4.0, 127 / 32
    for iteration in range(start, 60):
        loss, grad, hess = call(weights)
        gap = float(np.sum(grad * (weights - np.where(grad >= 0, low, high))))
        ref.record(
            work / f"ITER_{iteration:03d}.json",
            {"weights": weights.tolist(), "loss_nats": loss, "gradient": grad.tolist(), "gap_nats": gap},
        )
        print(
            json.dumps(
                {
                    "stage": "fit",
                    "variant": variant,
                    "iteration": iteration,
                    "kept_gain_bytes": (baseline - loss) / math.log(2) / 8,
                    "gap_bytes": gap / math.log(2) / 8,
                }
            ),
            flush=True,
        )
        if gap / math.log(2) / 8 < 0.01:
            break
        direction = np.zeros_like(weights)
        for bank in range(K):
            direction[bank] = np.linalg.solve(
                hess[bank] + np.eye(F) * max(1e-8, np.trace(hess[bank]) * 1e-10), -grad[bank]
            )
        accepted = False
        for trial_index, scale in enumerate(2.0 ** -np.arange(20)):
            candidate = np.clip(weights + scale * direction, low, high)
            trial_loss = call(candidate, False)[0]
            ref.record(
                work / f"TRIAL_{iteration:03d}_{trial_index:02d}.json",
                {"weights": candidate.tolist(), "loss_nats": trial_loss, "scale": float(scale)},
            )
            if trial_loss <= loss + 1e-4 * float(np.sum(grad * (candidate - weights))):
                weights = candidate
                accepted = True
                break
        if not accepted:
            raise RuntimeError("box Newton step stalled; retain and inspect")
        ref.record(last, {"binding": pin, "next_iteration": iteration + 1, "weights": weights.tolist()})
    loss, grad, _ = call(weights)
    gap = float(np.sum(grad * (weights - np.where(grad >= 0, low, high))))
    rounded = np.clip(np.rint(weights * 32), -128, 127).astype(np.int8)
    ref.blob(work / "weights_i8.bin", rounded.tobytes())
    save_arrays(work / "weights_float.npz", {"weights": weights})
    return ref.record(
        work / "RESULT.json",
        {
            "binding": pin,
            "weights": weights.tolist(),
            "rounded": rounded.tolist(),
            "kept_positions": len(arrays["truth"]),
            "kept_continuous_gain_bytes": (baseline - loss) / math.log(2) / 8,
            "final_gap_bytes": gap / math.log(2) / 8,
            "converged": gap / math.log(2) / 8 < 0.01,
            "scope": "retained fitting subset only, fixed features and bounded shared weights; finite-family numerical tangent; not full-n600 optimality or a universal geometry bound",
            "equation_anchor": "lane_boundary_context_map_bound_v1",
            "full_n600_pricing": "real twins next",
        },
    )


def encode(variant):
    weights_path = ROOT / "fit" / variant / "weights_i8.bin"
    fit_result = json.loads((ROOT / "fit" / variant / "RESULT.json").read_text())
    if not fit_result["converged"]:
        raise ValueError("joint calibration not converged")
    pin = binding(
        "encode/" + variant,
        extra={"weights": fact(weights_path), "fit_result": fact(ROOT / "fit" / variant / "RESULT.json")},
    )
    check_inherited(fit_result["binding"], pin)
    j = 0 if variant == "A" else 1
    F = 1 if j == 0 else 8
    weights = np.frombuffer(weights_path.read_bytes(), dtype=np.int8).reshape(K, F)
    np.testing.assert_array_equal(weights, np.array(fit_result["rounded"], dtype=np.int8))
    float_weights = np.array(fit_result["weights"])
    work = ROOT / "encode" / variant
    route = io.load_route_b()
    build_file = work / "BUILD.json"
    if build_file.exists():
        build = json.loads(build_file.read_text())
        for key in ("base_source", "generated", "library"):
            if fact(build[key]["path"]) != build[key]:
                raise ValueError("codec native build drift")
        library = Path(build["library"]["path"])
    else:
        library, build = io.compile_rc64(work, route, "tc3_" + variant)
        ref.record(build_file, build)
    state = None
    start = 0
    totals = np.zeros(3)
    last = work / "LATEST.json"
    if last.exists():
        receipt = json.loads(last.read_text())
        if receipt["binding"] != pin:
            raise ValueError("encode resume binding drift")
        state = load_arrays(Path(receipt["state"]))
        start = int(state["frame"][0])
        totals = state["totals"]
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[f"enc{i}"].tobytes()) for i in range(2)
    ]
    for frame in range(start, N):
        data = load_arrays(ROOT / "prepare_v2/frames" / f"frame_{frame:04d}.npz")
        source = source_frame(frame, pin["trace_binding"])
        original = source["rows" if j == 0 else "raw_rows"]
        phi, freq, truth = data[f"phi{j}"], data[f"freq{j}"].astype(np.int64), data["truth"]
        rows = base.mix_probabilities(freq, phi, weights, original)
        if j == 0:
            inactive = np.all(phi[:, :, 0] == 0, axis=1)
            rows[inactive] = original[inactive]
        ff = base.frequencies(rows)
        integer_bits = -np.log2(ff[np.arange(H * W), truth].astype(float) / base.TOTAL).sum()
        z = (
            np.log(freq.astype(float) / base.TOTAL)
            + np.sum(phi.astype(float) * float_weights[freq.argmax(axis=1), None, :], axis=2) * math.log(2) / base.Q
        )
        top = z.max(axis=1)
        norm = np.log(np.exp(z - top[:, None]).sum(axis=1)) + top
        continuous_bits = (norm - z[np.arange(H * W), truth]).sum() / math.log(2)
        source_freq = base.frequencies(source["rows"])
        source_bits = -np.log2(source_freq[np.arange(H * W), truth].astype(float) / base.TOTAL).sum()
        totals += [source_bits, continuous_bits, integer_bits]
        for enc in encoders:
            enc.encode(truth[ORDER].astype(np.int32), rows[ORDER])
        state = {"frame": np.array([frame + 1]), "totals": totals.copy()}
        state.update({f"enc{i}": np.frombuffer(enc.snapshot(), dtype=np.uint8) for i, enc in enumerate(encoders)})
        path = work / "states" / f"state_{frame + 1:04d}.npz"
        save_arrays(path, state)
        ref.record(last, {"binding": pin, "state": str(path)})
        if (frame + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "encode",
                        "variant": variant,
                        "frame": frame + 1,
                        "ideal_bytes": (totals[0] - totals[2]) / 8,
                    }
                ),
                flush=True,
            )
    riders = []
    for i, enc in enumerate(encoders):
        envelope = enc.finish()
        ref.blob(work / f"twin{i}.envelope", envelope)
        size = int(enc.library.rc64_encoder_size(enc.context))
        raw = ctypes.string_at(enc.library.rc64_encoder_data(enc.context), size)
        ref.blob(work / f"twin{i}.rc64", raw)
        counted = (
            ((ref.ROOT / "retained/tc1_weights.bin").read_bytes() + weights.tobytes()) if j == 0 else weights.tobytes()
        )
        rider = codec.MAGIC + bytes([j + 1]) + counted + raw
        ref.blob(work / f"twin{i}.rider", rider)
        riders.append(rider)
    if riders[0] != riders[1]:
        raise ValueError("independent twin encodes differ")
    return ref.record(
        work / "RESULT.json",
        {
            "binding": pin,
            "variant": variant,
            "full_n600": True,
            "positions": N * H * W,
            "twin_byte_identical": True,
            "totals_bits": totals.tolist(),
            "full_continuous_gain_bytes": (totals[0] - totals[1]) / 8,
            "full_integer_gain_bytes": (totals[0] - totals[2]) / 8,
            "source_rider_bytes": 40 + (ref.ROOT / "retained/shipped_token_stream.rc64").stat().st_size,
            "candidate_rider": fact(work / "twin0.rider"),
            "net_tail_saved_bytes": 40
            + (ref.ROOT / "retained/shipped_token_stream.rc64").stat().st_size
            - len(riders[0]),
            "archive_created": False,
            "score_claim": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "encode"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--stop-after", type=int, default=N)
    args = parser.parse_args()
    if args.resume_from.resolve() != ref.ROOT.resolve():
        raise ValueError("resume root differs from assigned arm store")
    result = (
        prepare(args.stop_after)
        if args.stage == "prepare"
        else (fit(args.variant) if args.stage == "fit" else encode(args.variant))
    )
    print(json.dumps(result, sort_keys=True))
