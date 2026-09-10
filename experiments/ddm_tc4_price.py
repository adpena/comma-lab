#!/usr/bin/env python3
"""TC4 scorer-free retained identity, five-map preparation, fitting, and twins.

Every stage pins move41; all encoded bytes and atomic resume states stay on SSD.
Old40 coefficients are frozen. New coefficients are fitted on KEEP then priced
on every symbol, as in TC2/TC3; no full-population optimum is claimed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as io
from experiments import ddm_tc1_mixer_codec as base
from experiments import ddm_tc3_geometry as geometry
from experiments import ddm_tc3_mixer as tc3
from experiments import ddm_tc4_maps as maps

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40")
ARCHIVE_SHA = "299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936"
FIELD_SHA = "b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5"
N, H, W, K = 600, 384, 512, 5
AXIS = "[exact bytes, scorer-free macOS-CPU, n600]"
ORDER = np.argsort(base.GROUP, kind="stable")
POSITIONS = [ORDER[base.GROUP[ORDER] == g] for g in range(190)]
CAP = 6 * 1024**3


def fact(path):
    return io.file_fact(Path(path))


def storage(path, need=0):
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside arm store")
    used = sum(p.stat().st_size for p in ROOT.parent.rglob("*") if p.is_file())
    if used + need > CAP or shutil.disk_usage(ROOT).free < need + 8 * 1024**3:
        raise RuntimeError("STORAGE_BLOCK: keep all bytes, no uncertified deletion")
    path.parent.mkdir(parents=True, exist_ok=True)


def record(path, value):
    storage(path, 65536)
    io.atomic_json(path, value)
    return value


def blob(path, payload):
    storage(path, len(payload))
    io.persist_immutable_bytes(path, payload, label="TC4 retained bytes")
    return fact(path)


def save(path, values):
    storage(path, sum(v.nbytes for v in values.values()))
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError("immutable array changed: " + str(path))
    else:
        temp = path.with_suffix(".new")
        with temp.open("wb") as out:
            np.savez_compressed(out, **values)
            out.flush()
            os.fsync(out.fileno())
        temp.replace(path)
    record(path.with_suffix(".json"), fact(path))
    return fact(path)


def arrays(path, source=False):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if source and "payload" in receipt:
        receipt = receipt["payload"]
    if fact(path) != receipt:
        raise ValueError("array custody changed: " + str(path))
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def pin(stage):
    pointer = (REPO / ".omx/state/canonical_frontier_pointer.json").read_bytes()
    p = json.loads(pointer)
    if p["effective_frontier"]["archive_sha256"] != ARCHIVE_SHA:
        raise RuntimeError("POINTER_MOVED: explicit field rebind required")
    blob(ROOT / "pointer" / (stage.replace("/", "_") + "_" + hashlib.sha256(pointer).hexdigest() + ".json"), pointer)
    if io.sha256_file(SOURCE / "candidate_runtime/archive.zip") != ARCHIVE_SHA:
        raise ValueError("source archive mismatch")
    if io.sha256_file(SOURCE / "field.u8") != FIELD_SHA:
        raise ValueError("source field mismatch")
    # Pin actual counted old40 weights and exact raw RC64, not summary lengths.
    member = io.read_archive_member(SOURCE / "candidate_runtime/archive.zip")
    parts = io.split_member(member)
    rider = parts["tail"][96:]
    config, raw = tc3.unpack_rider(rider)
    if config[0] != 1 or len(parts["tail"]) != 119675:
        raise ValueError("expected move41 TC3 A tail")
    blob(ROOT / "source/archive.zip", (SOURCE / "candidate_runtime/archive.zip").read_bytes())
    blob(ROOT / "source/old40.bin", config[1:])
    blob(ROOT / "source/raw.rc64", raw)
    blob(ROOT / "source/rider.bin", rider)
    blob(ROOT / "source/prefix.bin", member[: -len(rider)])
    sources = [fact(Path(m.__file__)) for m in (io, base, tc3, geometry, maps)]
    sources += [fact(Path(__file__)), fact(io.ROUTE_B), fact(REPO / "experiments/ddm_tc1_logistic_bound.c")]
    binding = {
        "archive_sha256": ARCHIVE_SHA,
        "field_sha256": FIELD_SHA,
        "source_files": sources,
        "old40": fact(ROOT / "source/old40.bin"),
        "source_manifests": [
            fact(SOURCE / rel) for rel in ("trace/RESULT.json", "prepare_v2/RESULT.json", "INPUTS.json")
        ],
        "runtime_files": [
            fact(p)
            for p in sorted((SOURCE / "candidate_runtime").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        ],
        "seed": 20260910,
        "score_claim": False,
        "axis": AXIS,
        "equation_anchor": "lane_boundary_context_map_bound_v1",
    }
    path = ROOT / stage / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("stage source binding changed: " + stage)
    record(path, binding)
    return binding


def inherited(stage, current):
    old = json.loads((ROOT / stage / "INPUTS.json").read_text())
    if old != current:
        raise ValueError("inherited stage binding changed: " + stage)


def native(work):
    route = io.load_route_b()
    path = work / "BUILD.json"
    if path.exists():
        build = json.loads(path.read_text())
        for k in ("base_source", "generated", "library"):
            if fact(build[k]["path"]) != build[k]:
                raise ValueError("native build drift")
        lib = Path(build["library"]["path"])
    else:
        storage(path, 4 * 1024**2)
        lib, build = io.compile_rc64(work, route, "tc4")
        record(path, build)
    return route, lib


def restore(work, binding):
    if not (work / "LATEST.json").exists():
        return {}, 0
    latest = json.loads((work / "LATEST.json").read_text())
    if latest["binding"] != binding:
        raise ValueError("resume binding mismatch")
    state = arrays(Path(latest["state"]))
    frame = int(state["frame"][0])
    if frame != latest["frame"]:
        raise ValueError("LATEST and checkpoint frames disagree")
    return state, frame


def checkpoint(work, binding, frame, values):
    path = work / "states" / f"stage_{frame:04d}.npz"
    save(path, dict(frame=np.array([frame]), **values))
    record(work / "LATEST.json", {"binding": binding, "frame": frame, "state": str(path)})


def finish_twins(work, encoders):
    result = []
    for i, encoder in enumerate(encoders):
        envelope = encoder.finish()
        blob(work / f"twin{i}.envelope", envelope)
        size = int(encoder.library.rc64_encoder_size(encoder.context))
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)
        blob(work / f"twin{i}.rc64", raw)
        result.append(raw)
    if result[0] != result[1]:
        raise ValueError("independent encoders disagree")
    return result


def control(stop):
    binding = pin("control")
    work = ROOT / "control"
    route, lib = native(work)
    state, start = restore(work, {"inputs": binding, "native": fact(work / "BUILD.json")})
    if not 0 <= start <= stop <= N:
        raise ValueError("requested control stop precedes resume state")
    encoders = [route.NativeRc64Encoder(lib, state[f"enc{i}"].tobytes() if state else None) for i in range(2)]
    weights = np.frombuffer((ROOT / "source/old40.bin").read_bytes(), dtype=np.int8)[35:].reshape(K, 1)
    trace_binding = json.loads((SOURCE / "trace/RESULT.json").read_text())["binding"]
    field = np.memmap(SOURCE / "field.u8", dtype=np.uint8, mode="r", shape=(N, H, W))
    for frame in range(start, stop):
        receipt = json.loads((SOURCE / "trace/frames" / f"frame_{frame:04d}.json").read_text())
        if receipt["binding"] != trace_binding or receipt["source_field_sha256"] != FIELD_SHA:
            raise ValueError("source frame producer or field drift")
        source = arrays(SOURCE / "trace/frames" / f"frame_{frame:04d}.npz", True)
        prep = arrays(SOURCE / "prepare_v2/frames" / f"frame_{frame:04d}.npz")
        np.testing.assert_array_equal(prep["truth"], source["tokens"].reshape(-1))
        np.testing.assert_array_equal(source["tokens"], field[frame])
        rows = base.mix_probabilities(prep["freq0"].astype(np.int64), prep["phi0"], weights, source["rows"])
        inactive = np.all(prep["phi0"][:, :, 0] == 0, axis=1)
        rows[inactive] = source["rows"][inactive]
        save(
            work / "frames" / f"frame_{frame:04d}.npz",
            {"rows": rows, "truth": prep["truth"], "lane_bins": prep["bins0"].reshape(H, W)},
        )
        for encoder in encoders:
            encoder.encode(prep["truth"][ORDER].astype(np.int32), rows[ORDER])
        if (frame + 1) % 20 == 0 or frame + 1 == stop:
            checkpoint(
                work,
                {"inputs": binding, "native": fact(work / "BUILD.json")},
                frame + 1,
                {f"enc{i}": np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(encoders)},
            )
            print(json.dumps({"stage": "control", "frame": frame + 1}), flush=True)
    raw = finish_twins(work / f"finish_{stop:04d}", encoders)
    identical = raw[0] == (ROOT / "source/raw.rc64").read_bytes()
    if stop == N and not identical:
        raise ValueError("full move41 raw stream identity failed")
    return record(
        work / ("RESULT.json" if stop == N else f"PARTIAL_{stop}.json"),
        {
            "binding": binding,
            "frames": stop,
            "twin_identical": True,
            "source_raw_identical": identical,
            "raw_bytes": len(raw[0]),
            "source_tail_bytes": 119675,
        },
    )


def prepare(stop):
    binding = pin("prepare")
    inherited("control", binding)
    control_result = json.loads((ROOT / "control/RESULT.json").read_text())
    if not control_result["source_raw_identical"]:
        raise ValueError("full identity must precede map pricing")
    work = ROOT / "prepare"
    state, start = restore(work, binding)
    if not 0 <= start <= stop <= N:
        raise ValueError("requested preparation stop precedes resume state")
    mixer = maps.ContextMixer(bytes([31]) + (ROOT / "source/old40.bin").read_bytes() + bytes(25))
    if state:
        for j in range(5):
            mixer.counts[j], mixer.expected[j] = state[f"c{j}"], state[f"e{j}"]
    previous = (
        None if not start else arrays(ROOT / "control/frames" / f"frame_{start - 1:04d}.npz")["truth"].reshape(H, W)
    )
    for frame in range(start, stop):
        data = arrays(ROOT / "control/frames" / f"frame_{frame:04d}.npz")
        truth, rows = data["truth"], data["rows"]
        plane = truth.reshape(H, W)
        codes = maps.context_maps(plane, maps.previous_map(previous), data["lane_bins"], complete=True)
        mixer.tables = [
            base.log2_fixed(np.clip((c + 0.5) / (e / base.TOTAL + 0.5), 1 / 16, 16))
            for c, e in zip(mixer.counts, mixer.expected, strict=True)
        ]
        phi, freq = mixer.features_from_codes(rows, codes)
        keep = (1 - freq.max(axis=1) / base.TOTAL >= 2.0**-14) | (truth != freq.argmax(axis=1))
        save(work / "frames" / f"frame_{frame:04d}.npz", {"phi": phi, "keep": keep, "codes": codes})
        mixer.update_counts(truth, codes, freq)
        checkpoint(
            work,
            binding,
            frame + 1,
            {**{f"c{j}": mixer.counts[j] for j in range(5)}, **{f"e{j}": mixer.expected[j] for j in range(5)}},
        )
        previous = plane
        if (frame + 1) % 20 == 0:
            print(json.dumps({"stage": "prepare", "frame": frame + 1}), flush=True)
    return record(work / ("RESULT.json" if stop == N else f"PARTIAL_{stop}.json"), {"binding": binding, "frames": stop})


def objective(selected, work):
    source = REPO / "experiments/ddm_tc1_logistic_bound.c"
    generated = work / "logistic.c"
    blob(generated, source.read_bytes().replace(b"#define F 7", f"#define F {len(selected)}".encode()))
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
        record(
            work / "FIT_BUILD.json",
            {"source": fact(source), "generated": fact(generated), "library": fact(library), "argv": command},
        )
    else:
        build = json.loads((work / "FIT_BUILD.json").read_text())
        if build != {"source": fact(source), "generated": fact(generated), "library": fact(library), "argv": command}:
            raise ValueError("fit library drift")
    chunks = {k: [] for k in ("phi", "logp", "truth", "group")}
    for frame in range(N):
        data = arrays(ROOT / "prepare/frames" / f"frame_{frame:04d}.npz")
        source_data = arrays(ROOT / "control/frames" / f"frame_{frame:04d}.npz")
        keep = data["keep"]
        freq = base.frequencies(source_data["rows"][keep])
        chunks["phi"].append(data["phi"][keep][:, :, selected])
        chunks["logp"].append(np.log(freq / base.TOTAL))
        chunks["truth"].append(source_data["truth"][keep])
        chunks["group"].append(freq.argmax(axis=1).astype(np.uint8))
    values = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in chunks.items()}
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
        grad, hess = np.zeros((K, len(selected))), np.zeros((K, len(selected), len(selected)))
        loss = lib.tc1_objective(
            len(values["truth"]),
            values["logp"],
            values["phi"],
            values["truth"],
            values["group"],
            np.ascontiguousarray(weights),
            int(derivatives),
            grad,
            hess,
        )
        if not all(np.isfinite(v).all() for v in (np.asarray(loss), grad, hess)):
            raise ValueError("nonfinite objective")
        return loss, grad, hess

    return call, len(values["truth"])


def fit(mask):
    binding = pin(f"fit/{mask}")
    inherited("prepare", binding)
    inherited("control", binding)
    selected = [j for j in range(5) if mask & (1 << j)]
    if json.loads((ROOT / "prepare/RESULT.json").read_text())["frames"] != N:
        raise ValueError("full preparation required")
    work = ROOT / "fit" / str(mask)
    call, count = objective(selected, work)
    fit_binding = {"inputs": binding, "native": fact(work / "FIT_BUILD.json")}
    weights = np.zeros((K, len(selected)))
    initial = call(weights, False)[0]
    last = work / "STATE.json"
    start = 0
    if last.exists():
        state = json.loads(last.read_text())
        if state["binding"] != fit_binding:
            raise ValueError("fit resume drift")
        start, weights = state["iteration"], np.array(state["weights"])
    for iteration in range(start, 80):
        loss, grad, hess = call(weights)
        gap = float(np.sum(grad * (weights - np.where(grad >= 0, -4.0, 127 / 32))))
        record(
            work / f"ITER_{iteration:03d}.json",
            {"weights": weights.tolist(), "loss": loss, "gap_bytes": gap / math.log(2) / 8},
        )
        print(
            json.dumps(
                {
                    "stage": "fit",
                    "mask": mask,
                    "iteration": iteration,
                    "gain_bytes": (initial - loss) / math.log(2) / 8,
                    "gap_bytes": gap / math.log(2) / 8,
                }
            ),
            flush=True,
        )
        if gap / math.log(2) / 8 < 0.01:
            break
        direction = np.stack(
            [
                np.linalg.solve(h + np.eye(len(selected)) * max(1e-8, np.trace(h) * 1e-10), -g)
                for g, h in zip(grad, hess, strict=True)
            ]
        )
        for trial, scale in enumerate(2.0 ** -np.arange(20)):
            candidate = np.clip(weights + scale * direction, -4.0, 127 / 32)
            candidate_loss = call(candidate, False)[0]
            record(
                work / f"TRIAL_{iteration:03d}_{trial:02d}.json",
                {"weights": candidate.tolist(), "loss": candidate_loss},
            )
            if candidate_loss <= loss + 1e-4 * float(np.sum(grad * (candidate - weights))):
                weights = candidate
                break
        else:
            raise RuntimeError("fit stalled; retain and inspect")
        record(last, {"binding": fit_binding, "iteration": iteration + 1, "weights": weights.tolist()})
    loss, grad, _ = call(weights)
    gap = float(np.sum(grad * (weights - np.where(grad >= 0, -4.0, 127 / 32)))) / math.log(2) / 8
    rounded = np.clip(np.rint(weights * 32), -128, 127).astype(np.int8)
    blob(work / "weights.bin", rounded.tobytes())
    return record(
        work / "RESULT.json",
        {
            "binding": binding,
            "selected": selected,
            "weights": weights.tolist(),
            "rounded": rounded.tolist(),
            "kept_positions": count,
            "kept_gain_bytes": (initial - loss) / math.log(2) / 8,
            "gap_bytes": gap,
            "converged": gap < 0.01,
            "scope": "KEEP subset fixed-feature numerical tangent; no full population optimum",
        },
    )


def encode(mask):
    binding = pin(f"encode/{mask}")
    for stage in ("control", "prepare", f"fit/{mask}"):
        inherited(stage, binding)
    work = ROOT / "encode" / str(mask)
    fitted = json.loads((ROOT / "fit" / str(mask) / "RESULT.json").read_text())
    if not fitted["converged"]:
        raise ValueError("unconverged fit")
    selected = fitted["selected"]
    if selected != [j for j in range(5) if mask & (1 << j)] or fitted["binding"] != binding:
        raise ValueError("fit mask or binding mismatch")
    weights = np.array(fitted["rounded"], dtype=np.int8)
    if weights.tobytes() != (ROOT / "fit" / str(mask) / "weights.bin").read_bytes():
        raise ValueError("retained weights differ from shipped rounded weights")
    floats = np.array(fitted["weights"])
    route, lib = native(work)
    state, start = restore(work, {"binding": binding, "fit": fitted, "native": fact(work / "BUILD.json")})
    encoders = [route.NativeRc64Encoder(lib, state[f"enc{i}"].tobytes() if state else None) for i in range(2)]
    totals = state["totals"] if state else np.zeros(3)
    for frame in range(start, N):
        src = arrays(ROOT / "control/frames" / f"frame_{frame:04d}.npz")
        data = arrays(ROOT / "prepare/frames" / f"frame_{frame:04d}.npz")
        truth, original, phi = src["truth"], src["rows"], data["phi"][:, :, selected]
        freq = base.frequencies(original)
        rows = maps.mix_rows(freq, phi, weights, original)
        integer = base.frequencies(rows)
        z = (
            np.log(freq / base.TOTAL)
            + np.sum(phi * floats[freq.argmax(axis=1), None, :], axis=2) * math.log(2) / base.Q
        )
        top = z.max(axis=1)
        norm = np.log(np.exp(z - top[:, None]).sum(axis=1)) + top
        totals += [
            -np.log2(freq[np.arange(H * W), truth] / base.TOTAL).sum(),
            (norm - z[np.arange(H * W), truth]).sum() / math.log(2),
            -np.log2(integer[np.arange(H * W), truth] / base.TOTAL).sum(),
        ]
        for encoder in encoders:
            encoder.encode(truth[ORDER].astype(np.int32), rows[ORDER])
        if (frame + 1) % 20 == 0:
            checkpoint(
                work,
                {"binding": binding, "fit": fitted, "native": fact(work / "BUILD.json")},
                frame + 1,
                dict(
                    totals=totals.copy(),
                    **{f"enc{i}": np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(encoders)},
                ),
            )
            print(json.dumps({"stage": "encode", "mask": mask, "frame": frame + 1}), flush=True)
    raws = finish_twins(work, encoders)
    riders, archives = [], []
    for i, raw in enumerate(raws):
        rider = maps.MAGIC + bytes([mask]) + (ROOT / "source/old40.bin").read_bytes() + weights.tobytes() + raw
        riders.append(blob(work / f"twin{i}.rider", rider))
        member = (ROOT / "source/prefix.bin").read_bytes() + rider
        path = work / f"twin{i}.zip"
        storage(path, len(member) + 1024)
        io.pack_archive(member, path)
        if io.read_archive_member(path) != member:
            raise ValueError("archive parse mismatch")
        archives.append(fact(path))
    if archives[0]["sha256"] != archives[1]["sha256"]:
        raise ValueError("archive twins differ")
    result = {
        "binding": binding,
        "mask": mask,
        "frames": N,
        "positions": N * H * W,
        "totals_bits": totals.tolist(),
        "continuous_gain_bytes": (totals[0] - totals[1]) / 8,
        "integer_gain_bytes": (totals[0] - totals[2]) / 8,
        "tail_bytes": 96 + riders[0]["bytes"],
        "tail_saved_bytes": 119675 - 96 - riders[0]["bytes"],
        "archive_bytes": archives[0]["bytes"],
        "archive_saved_bytes": 180154 - archives[0]["bytes"],
        "fitted_weight_bytes": 40 + weights.size,
        "riders": riders,
        "archives": archives,
        "twin_identical": True,
        "score_claim": False,
    }
    return record(work / "RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("control", "prepare", "fit", "encode"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--mask", type=int, default=1)
    parser.add_argument("--stop-after", type=int, default=N)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.mask < 32 or not 0 < args.stop_after <= N:
        raise ValueError("invalid explicit resume root/mask/stop")
    result = (
        control(args.stop_after)
        if args.stage == "control"
        else prepare(args.stop_after)
        if args.stage == "prepare"
        else fit(args.mask)
        if args.stage == "fit"
        else encode(args.mask)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
