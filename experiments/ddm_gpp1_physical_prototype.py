#!/usr/bin/env python3
# ruff: noqa: I001
"""Physically encode and decode the GPP1 context over exact move-44 rows.

This scorer-free n600 prototype consumes the immutable LS1 observations of the
shipping RLC1 probability rows.  A copied receiver module adds one cold-start
online calibration bank keyed by a retained causal public-model belief.  Three
counted one-byte strength variants are encoded with twin RC64 encoders, packed
into retained prototype archives, and decoded back to the token field.

The prototype does not integrate RGB/token interleaving into public inflate.py,
does not claim that the public model weights are free under rule 118, and is not
a candidate archive or score.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import shutil
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
from experiments.ddm_ls1_shipped_surprise import FIELD, FIELD_SHA, H, K, N, TOTAL, W

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_gpp1/prototype")
GPP1 = ROOT.parent
V1_SOURCE = ROOT / "source/ddm_gpp1_physical_prototype_v1.py"
LS1 = ROOT.parent.parent / "ddm_ls1"
SOURCE_RUNTIME = ROOT.parent.parent / "ddm_rlc5_cure_on_move43/candidate_runtime"
ARCHIVE_SHA = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
STREAM_SHA = "c499972a33dac497412c18839b632a8e1bbf75d518528039db0e2aca80c8eb13"
RUNTIME_SOURCE = REPO / "experiments/ddm_gpp1_runtime_prior.py"
WEIGHTS = (8, 16, 32)
MODELS = ("control", "w8", "w16", "w32")
PHASE_Y, PHASE_X = np.indices((H, W))
PHASE = (PHASE_X % 64 + 2 * (PHASE_Y % 64)).reshape(-1)
PHASE_POSITIONS = tuple(np.flatnonzero(group == PHASE) for group in range(190))
PLANE = H * W
SEED = 20260911
RESERVE_BYTES = 16 << 30
TAIL_DEMAND_BYTES = 25899
STRICT_DECODE_LIMIT_SECONDS = 1260.0
MOVE44_T4_INFLATE_SECONDS = 1232.418725255
T4_OVER_LOCAL_TAIL = 1142.9962956905365 / 1168.6204084999987
AXIS = "[macOS-CPU advisory / scorer-free n600 exact RC64 byte measurement]"


class PrototypeError(RuntimeError):
    """Fail-closed physical-prototype error."""


def fact(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def guard(path: Path, need: int = 0) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise PrototypeError(f"write escaped owned prototype store: {path}")
    if shutil.disk_usage(ROOT).free < RESERVE_BYTES + need:
        raise PrototypeError("STORAGE_BLOCK: retain all evidence; no deletion authorized")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_bytes(path: Path, payload: bytes, *, immutable: bool = True) -> None:
    guard(path, len(payload))
    if path.exists() and immutable:
        if path.read_bytes() != payload:
            raise PrototypeError(f"immutable retained payload changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path: Path, value: object, *, immutable: bool = True) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        immutable=immutable,
    )


def atomic_npz(path: Path, values: dict[str, np.ndarray], *, immutable: bool = True) -> None:
    guard(path, sum(value.nbytes for value in values.values()))
    if path.exists() and immutable:
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(values) or any(
                not np.array_equal(prior[name], value) for name, value in values.items()
            ):
                raise PrototypeError(f"immutable retained array changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_npz(path: Path) -> dict[str, np.ndarray]:
    receipt = json.loads(path.with_suffix(".json").read_text())
    expected = receipt.get("payload", receipt)
    if fact(path) != expected:
        raise PrototypeError(f"retained input checksum changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def copy_runtime() -> dict[str, object]:
    copied: dict[str, object] = {}
    for source in sorted(SOURCE_RUNTIME.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix in {".pyc", ".so", ".dylib"}:
            continue
        relative = source.relative_to(SOURCE_RUNTIME)
        destination = ROOT / "runtime_copy" / relative
        atomic_bytes(destination, source.read_bytes())
        source_fact, copy_fact = fact(source), fact(destination)
        if any(source_fact[name] != copy_fact[name] for name in ("bytes", "sha256")):
            raise PrototypeError("runtime copy identity failed")
        copied[str(relative)] = {"source": source_fact, "copy": copy_fact}
    receiver_module = ROOT / "runtime_copy/runtime/gpp1_prior.py"
    atomic_bytes(receiver_module, RUNTIME_SOURCE.read_bytes())
    copied["runtime/gpp1_prior.py"] = {
        "source": fact(RUNTIME_SOURCE),
        "copy": fact(receiver_module),
    }
    return copied


def bind() -> tuple[dict[str, object], object, object]:
    if fact(SOURCE_RUNTIME / "archive.zip")["sha256"] != ARCHIVE_SHA:
        raise PrototypeError("move-44 source archive changed")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise PrototypeError("move-44 decoded field changed")
    inference = json.loads((GPP1 / "INFERENCE_RESULT.json").read_text())
    oracle = json.loads((GPP1 / "oracle/RESULT.json").read_text())
    if not inference["complete_n600"] or not oracle["full_n600"] or not oracle["prototype_required"]:
        raise PrototypeError("complete triggered GPP1 bound is required")
    sources = copy_runtime()
    runtime_copy = ROOT / "runtime_copy"
    sys.path.insert(0, str(runtime_copy))
    from runtime import residual_archive
    from runtime.gpp1_prior import GenericMotionMixer

    parts = residual_archive.read_residual_archive(runtime_copy / "archive.zip")
    stream = parts.token_stream
    if len(stream) != 119749 or hashlib.sha256(stream).hexdigest() != STREAM_SHA:
        raise PrototypeError("shipped move-44 token stream changed")
    pins = {
        "axis": AXIS,
        "score_claim": False,
        "archive": fact(SOURCE_RUNTIME / "archive.zip"),
        "field": fact(FIELD),
        "ls1_trace": fact(LS1 / "TRACE.json"),
        "ls1_atlas": fact(LS1 / "atlas/RESULT.json"),
        "belief_inference": fact(GPP1 / "INFERENCE_RESULT.json"),
        "belief_oracle": fact(GPP1 / "oracle/RESULT.json"),
        "collision_recovery": fact(GPP1 / "collision_retained/RECOVERY.json"),
        "runtime_sources": sources,
        "producer": fact(Path(__file__)),
        "receiver_module": fact(RUNTIME_SOURCE),
        "seed": SEED,
        "strengths_q5": WEIGHTS,
        "state_rule": "cold KT tables; update only after a complete decoded plane",
        "retention": "KEEP all streams, checkpoints, archives, decoded field, and copied runtime",
    }
    path = ROOT / "INPUTS.json"
    if path.exists():
        prior = json.loads(path.read_text())
        normalized_pins = json.loads(json.dumps(pins, sort_keys=True))
        stable_prior = {key: value for key, value in prior.items() if key != "producer"}
        stable_current = {key: value for key, value in normalized_pins.items() if key != "producer"}
        if stable_prior != stable_current:
            raise PrototypeError("prototype input binding changed")
        if prior["producer"] != pins["producer"]:
            if fact(V1_SOURCE) != {
                **prior["producer"],
                "path": str(V1_SOURCE),
            }:
                raise PrototypeError("previous prototype source was not retained byte-identically")
            save_json(
                ROOT / "source/PROVENANCE_MIGRATION.json",
                {
                    "reason": "terminal-resume validation added after the measured run",
                    "numerical_or_receiver_change": False,
                    "previous_producer": prior["producer"],
                    "previous_source_retained": fact(V1_SOURCE),
                    "current_producer": pins["producer"],
                },
            )
        pins = prior
    else:
        save_json(path, pins)
    atomic_bytes(ROOT / "retained/shipped.rc64", stream)
    return pins, parts, GenericMotionMixer


def build_encoder() -> Path:
    receipt = ROOT / "build/BUILD.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        library = Path(prior["library"]["path"])
        if prior["library"] != fact(library):
            raise PrototypeError("RC64 build changed")
        return library
    route = jg2.load_route_b()
    library, build = jg2.compile_rc64(ROOT / "build", route, "gpp1")
    save_json(receipt, build)
    return library


def encode_frequencies(encoder, symbols: np.ndarray, rows: np.ndarray) -> None:
    source = np.ascontiguousarray(symbols, dtype=np.int32)
    lattice = np.ascontiguousarray(rows, dtype=np.uint32)
    if lattice.shape != (len(source), K) or np.any(lattice.sum(axis=1, dtype=np.uint64) != TOTAL):
        raise PrototypeError("invalid RC64 lattice")
    status = encoder.library.rc64_encoder_encode(
        encoder.context,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lattice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(source),
    )
    if status:
        raise PrototypeError(f"RC64 encoder failed with status {status}")


def encode_plane_integer(encoder, truth: np.ndarray, rows: np.ndarray) -> None:
    for positions in PHASE_POSITIONS:
        encode_frequencies(encoder, truth[positions], rows[positions])


def encode_plane_float(encoder, truth: np.ndarray, rows: np.ndarray) -> None:
    for positions in PHASE_POSITIONS:
        encoder.encode(truth[positions], rows[positions])


def load_frame(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = checked_npz(LS1 / "rows" / f"frame_{frame:04d}.npz")
    base, truth, bits = row["frequencies"], row["symbols"], row["bits"]
    belief = checked_npz(GPP1 / "beliefs" / f"pair_{frame:04d}.npz")["belief_motion_boundary16"].reshape(-1)
    if base.shape != (PLANE, K) or truth.shape != (PLANE,) or belief.shape != (PLANE,):
        raise PrototypeError("frame payload shape changed")
    if np.any(base.sum(axis=1, dtype=np.uint64) != TOTAL):
        raise PrototypeError("frame probability mass changed")
    return base, truth, bits, belief


def state_arrays(frame: int, mixers: dict[str, object], ideal_bits: np.ndarray) -> dict[str, np.ndarray]:
    values = {"frame": np.array([frame], dtype=np.int64), "ideal_bits": ideal_bits}
    for name, mixer in mixers.items():
        values.update({f"{name}_{key}": value for key, value in mixer.snapshot().items()})
    return values


def checkpoint(frame: int, mixers: dict[str, object], ideal_bits: np.ndarray, encoders: dict[str, object]) -> None:
    stage = ROOT / "checkpoints" / f"state_{frame:04d}.npz"
    atomic_npz(stage, state_arrays(frame, mixers, ideal_bits))
    encoder_facts = {}
    for name, encoder in encoders.items():
        path = ROOT / "checkpoints" / f"encoder_{name}_{frame:04d}.bin"
        atomic_bytes(path, encoder.snapshot())
        encoder_facts[name] = fact(path)
    receipt = {
        "schema": "ddm_gpp1.physical_checkpoint.v1",
        "frame": frame,
        "state": fact(stage),
        "encoders": encoder_facts,
    }
    save_json(ROOT / "checkpoints" / f"stage_{frame:04d}.json", receipt)
    save_json(ROOT / "LATEST.json", receipt, immutable=False)


def resume(library: Path, mixer_class) -> tuple[int, dict[str, object], np.ndarray, dict[str, object]]:
    route = jg2.load_route_b()
    names = ["control", *[f"w{weight}_{twin}" for weight in WEIGHTS for twin in range(2)]]
    latest = ROOT / "LATEST.json"
    mixers = {f"w{weight}": mixer_class(weight) for weight in WEIGHTS}
    if not latest.exists():
        return (
            0,
            mixers,
            np.zeros((len(MODELS), N), dtype=np.float64),
            {name: route.NativeRc64Encoder(library) for name in names},
        )
    receipt = json.loads(latest.read_text())
    if receipt["state"] != fact(Path(receipt["state"]["path"])):
        raise PrototypeError("latest state checkpoint changed")
    with np.load(receipt["state"]["path"], allow_pickle=False) as saved:
        frame = int(saved["frame"][0])
        ideal_bits = saved["ideal_bits"].copy()
        for name, mixer in mixers.items():
            mixer.restore({key: saved[f"{name}_{key}"] for key in ("weight_q5", "counts", "expected")})
    encoders = {}
    for name in names:
        path = Path(receipt["encoders"][name]["path"])
        if fact(path) != receipt["encoders"][name]:
            raise PrototypeError("latest encoder checkpoint changed")
        encoders[name] = route.NativeRc64Encoder(library, path.read_bytes())
    if frame != receipt["frame"]:
        raise PrototypeError("state/encoder checkpoint frame mismatch")
    return frame, mixers, ideal_bits, encoders


def encoder_body(encoder) -> bytes:
    encoder.finish()
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise PrototypeError("RC64 encoder produced no payload")
    return ctypes.string_at(pointer, size)


def validate_terminal_result(result: dict[str, object]) -> None:
    """Validate a completed checkpoint without replacing measured wall times."""
    if result["inputs"] != fact(ROOT / "INPUTS.json"):
        raise PrototypeError("terminal result input receipt changed")
    for variant in result["variants"].values():
        stream = variant["stream"]
        if stream != fact(Path(stream["path"])):
            raise PrototypeError("terminal result stream changed")
        archive = variant["archive"]
        if archive is not None and archive["archive"] != fact(Path(archive["archive"]["path"])):
            raise PrototypeError("terminal result archive changed")
    decoded = result["winner_receiver"]["decoded_field"]
    if decoded != fact(Path(decoded["path"])) or decoded["sha256"] != FIELD_SHA:
        raise PrototypeError("terminal result decoded field changed")
    migration = json.loads((ROOT / "source/PROVENANCE_MIGRATION.json").read_text())
    if migration["previous_producer"] != result["producer"]:
        raise PrototypeError("terminal result producer is not covered by source migration")


def pack_archives(parts, streams: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    original = jg2.read_archive_member(SOURCE_RUNTIME / "archive.zip")
    sections = jg2.split_member(original)
    if sections["tail"][96:160] != b"RLC1" + bytes(parts.tc1_weights):
        raise PrototypeError("move-44 RLC1 rider layout changed")
    outputs: dict[str, dict[str, object]] = {}
    for weight in WEIGHTS:
        name = f"w{weight}"
        stream = Path(streams[name]["path"]).read_bytes()
        rider = b"GPP1" + bytes([1, weight]) + b"RLC1" + bytes(parts.tc1_weights) + stream
        atomic_bytes(ROOT / f"retained/{name}.rider", rider)
        changed = dict(sections)
        changed["tail"] = sections["tail"][:96] + rider
        member = jg2.join_member(changed)
        for section_name in ("header", "hpac", "semantic", "carrier"):
            if changed[section_name] != sections[section_name]:
                raise PrototypeError("prototype changed an untargeted archive section")
        atomic_bytes(ROOT / f"retained/{name}.member", member)
        archive = ROOT / f"retained/{name}.prototype.zip"
        if not archive.exists():
            jg2.pack_archive(member, archive)
        if jg2.read_archive_member(archive) != member:
            raise PrototypeError("prototype archive parse-back failed")
        outputs[name] = {
            "archive": fact(archive),
            "rider": fact(ROOT / f"retained/{name}.rider"),
            "member": fact(ROOT / f"retained/{name}.member"),
            "archive_delta_bytes": archive.stat().st_size - 180406,
            "stream_savings_bytes": 119749 - len(stream),
            "prototype_header_bytes": 6,
        }
    return outputs


def decode_stream(name: str, weight: int, stream: bytes, library: Path, mixer_class) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "runtime_copy"))
    from runtime.entropy.rc64 import NativeDecoder

    mixer = mixer_class(weight)
    decoder = NativeDecoder(library, stream)
    decoded = np.empty((N, H, W), dtype=np.uint8)
    model_seconds = 0.0
    started = time.perf_counter()
    for frame in range(N):
        base, truth, _, belief = load_frame(frame)
        tick = time.perf_counter()
        table = mixer.table()
        probabilities = mixer.coding(base, belief, table)
        current = np.empty(PLANE, dtype=np.uint8)
        for positions in PHASE_POSITIONS:
            current[positions] = decoder.decode(probabilities[positions]).astype(np.uint8)
        mixer.observe(base, belief, current)
        model_seconds += time.perf_counter() - tick
        if not np.array_equal(current, truth):
            raise PrototypeError(f"prototype receiver diverged at frame {frame}")
        decoded[frame] = current.reshape(H, W)
    elapsed = time.perf_counter() - started
    path = ROOT / f"retained/{name}.decoded.u8"
    atomic_bytes(path, decoded.tobytes())
    decoded_fact = fact(path)
    if decoded_fact["sha256"] != FIELD_SHA:
        raise PrototypeError("prototype receiver field differs")
    return {
        "decoded_field": decoded_fact,
        "token_field_byte_diff": 0,
        "full_n600": True,
        "receiver_model_and_rc64_seconds_local_cpu": model_seconds,
        "receiver_total_replay_seconds_local_cpu": elapsed,
        "decoder_bit_position": decoder.bit_position,
    }


def run() -> dict[str, object]:
    pins, parts, mixer_class = bind()
    random.seed(SEED)
    np.random.seed(SEED)
    library = build_encoder()
    start, mixers, ideal_bits, encoders = resume(library, mixer_class)
    result_path = ROOT / "RESULT.json"
    if start == N and result_path.exists():
        result = json.loads(result_path.read_text())
        validate_terminal_result(result)
        print(
            json.dumps(
                {
                    "stage": "terminal_resume_validated",
                    "frame": start,
                    "result": fact(result_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result
    target = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    model_seconds = dict.fromkeys(MODELS, 0.0)
    started = time.perf_counter()
    for frame in range(start, N):
        base, truth, bits, belief = load_frame(frame)
        np.testing.assert_array_equal(truth, target[frame].reshape(-1))
        tick = time.perf_counter()
        encode_plane_integer(encoders["control"], truth, base)
        ideal_bits[0, frame] = float(bits.sum())
        model_seconds["control"] += time.perf_counter() - tick
        for model_index, weight in enumerate(WEIGHTS, 1):
            name = f"w{weight}"
            tick = time.perf_counter()
            table = mixers[name].table()
            probabilities = mixers[name].coding(base, belief, table)
            frequency = (probabilities.astype(np.float64) * TOTAL).astype(np.int64)
            np.maximum(frequency, 1, out=frequency)
            winner = probabilities.argmax(axis=1)
            frequency[np.arange(PLANE), winner] += TOTAL - frequency.sum(axis=1)
            ideal_bits[model_index, frame] = float(
                -np.log2(frequency[np.arange(PLANE), truth].astype(np.float64) / TOTAL).sum()
            )
            for twin in range(2):
                encode_plane_float(encoders[f"{name}_{twin}"], truth, probabilities)
            mixers[name].observe(base, belief, truth)
            model_seconds[name] += time.perf_counter() - tick
        boundary = frame + 1
        if boundary % 25 == 0:
            checkpoint(boundary, mixers, ideal_bits, encoders)
            print(
                json.dumps(
                    {
                        "stage": "physical_encode",
                        "frames": boundary,
                        "elapsed_seconds": time.perf_counter() - started,
                        "ideal_incremental_savings_bytes": {
                            name: float((ideal_bits[0, :boundary].sum() - ideal_bits[index, :boundary].sum()) / 8)
                            for index, name in enumerate(MODELS[1:], 1)
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    streams: dict[str, dict[str, object]] = {}
    control = encoder_body(encoders["control"])
    atomic_bytes(ROOT / "retained/control.rc64", control)
    streams["control"] = fact(ROOT / "retained/control.rc64")
    if streams["control"]["sha256"] != STREAM_SHA or streams["control"]["bytes"] != 119749:
        raise PrototypeError("INSTRUMENT_FALSIFIED: control differs from shipped stream")
    for weight in WEIGHTS:
        name = f"w{weight}"
        bodies = []
        for twin in range(2):
            body = encoder_body(encoders[f"{name}_{twin}"])
            path = ROOT / f"retained/{name}.twin{twin}.rc64"
            atomic_bytes(path, body)
            bodies.append(fact(path))
        if bodies[0]["sha256"] != bodies[1]["sha256"]:
            raise PrototypeError(f"{name} twin streams differ")
        streams[name] = bodies[0]
    archives = pack_archives(parts, streams)
    winner = min(MODELS[1:], key=lambda name: (archives[name]["archive"]["bytes"], name))
    winner_weight = WEIGHTS[MODELS[1:].index(winner)]
    decode = decode_stream(
        winner,
        winner_weight,
        Path(streams[winner]["path"]).read_bytes(),
        library,
        mixer_class,
    )
    inference = json.loads((GPP1 / "INFERENCE_RESULT.json").read_text())
    strict_slack = STRICT_DECODE_LIMIT_SECONDS - MOVE44_T4_INFLATE_SECONDS
    projected_model_t4 = inference["model_wall_seconds_local_cpu"] * T4_OVER_LOCAL_TAIL
    variants = {}
    for index, name in enumerate(MODELS):
        stream = streams[name]
        variants[name] = {
            "stream": stream,
            "realized_stream_savings_bytes": 119749 - stream["bytes"],
            "ideal_stream_savings_bytes": float((ideal_bits[0].sum() - ideal_bits[index].sum()) / 8),
            "local_model_and_encode_seconds_this_process": model_seconds[name],
            "archive": None if name == "control" else archives[name],
        }
    result = {
        "axis": AXIS,
        "score_claim": False,
        "candidate_archive": False,
        "full_n600": True,
        "symbols": N * PLANE,
        "control_byte_identical": True,
        "variants": variants,
        "winner": winner,
        "winner_receiver": decode,
        "winner_archive_savings_bytes": 180406 - archives[winner]["archive"]["bytes"],
        "winner_fraction_of_demand": (180406 - archives[winner]["archive"]["bytes"]) / TAIL_DEMAND_BYTES,
        "determinism": {
            "candidate_encoder_twins_byte_identical": True,
            "model_repeat_pairs_byte_identical": [1, 100, 300, 599],
            "receiver_token_field_byte_identical": True,
            "cross_host_model_belief_identity_measured": False,
        },
        "wall": {
            "model_n600_local_cpu_seconds": inference["model_wall_seconds_local_cpu"],
            "prototype_receiver_mix_and_rc64_local_cpu_seconds": decode["receiver_model_and_rc64_seconds_local_cpu"],
            "strict_t4_incremental_slack_seconds": strict_slack,
            "model_t4_seconds_projection_not_measurement": projected_model_t4,
            "projected_model_over_strict_slack_seconds": projected_model_t4 - strict_slack,
            "verdict": "FAILS_STRICT_RECEIVER_WALL",
        },
        "receiver_boundary": (
            "exact token-stream decode over copied receiver mixing code and exact shipped "
            "RLC1 rows; RGB/token interleaving and public inflate.py integration are not built"
        ),
        "rule118_boundary": (
            "prototype only; public generic weight exclusion remains an operator publication decision"
        ),
        "inputs": fact(ROOT / "INPUTS.json"),
        "producer": fact(Path(__file__)),
    }
    save_json(ROOT / "RESULT.json", result)
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)
    return result


def self_test() -> None:
    assert len(PHASE_POSITIONS) == 190
    merged = np.concatenate(PHASE_POSITIONS)
    assert len(merged) == PLANE and len(np.unique(merged)) == PLANE
    assert tuple(MODELS[1:]) == tuple(f"w{weight}" for weight in WEIGHTS)
    print("ddm_gpp1 physical prototype self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "self-test"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        if args.resume_from is None or args.resume_from.resolve() != ROOT.resolve():
            raise PrototypeError("--resume-from must name the canonical prototype store")
        run()
