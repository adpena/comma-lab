"""Retained real-field checks against the sealed native primitive implementations.

Geometry checks replay every causal group of explicitly selected real planes.
The arithmetic check covers initialization only unless real prefix probability
rows are provided. This is neither a renderer test nor an authoritative score.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ddm_mrs1_python_primitives as primitive


def fact(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_array(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)
    return fact(path)


def build(source, destination):
    command = ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(destination)]
    if not destination.exists():
        subprocess.run(command, check=True, timeout=30)
    return {"command": command, "source": fact(source), "binary": fact(destination)}


class NativeGeometry:
    """Explicit-path oracle adapter; no environment-dependent implementation choice."""

    def __init__(self, library, previous, config):
        self.library = ctypes.CDLL(str(library))
        integers = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        octets = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.library.rlc_new.argtypes = [integers, ctypes.c_void_p]
        self.library.rlc_new.restype = ctypes.c_void_p
        self.library.rlc_free.argtypes = [ctypes.c_void_p]
        self.library.rlc_free.restype = None
        self.library.rlc_contexts.argtypes = [ctypes.c_void_p, ctypes.c_int, integers, octets]
        self.library.rlc_contexts.restype = None
        self.library.rlc_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, integers, integers]
        self.library.rlc_observe.restype = None
        values = np.array(primitive.parse_geometry_config(config), dtype=np.int64)
        previous = None if previous is None else np.ascontiguousarray(previous, dtype=np.uint8)
        self.handle = self.library.rlc_new(values, None if previous is None else previous.ctypes.data)
        if not self.handle:
            raise MemoryError("native geometry allocation failed")

    def contexts(self, positions):
        output = np.empty(len(positions), dtype=np.uint8)
        self.library.rlc_contexts(self.handle, len(positions), positions, output)
        return output

    def observe(self, positions, symbols):
        self.library.rlc_observe(self.handle, len(positions), positions, np.ascontiguousarray(symbols, dtype=np.int64))

    def close(self):
        self.library.rlc_free(self.handle)
        self.handle = None


class DecoderState(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64),
                ("code", ctypes.c_uint64), ("data", ctypes.c_void_p),
                ("size", ctypes.c_size_t), ("bit_position", ctypes.c_size_t),
                ("error", ctypes.c_int)]


def check_range(library, payload, root, probability_path):
    lib = ctypes.CDLL(str(library))
    lib.rc64_decoder_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.rc64_decoder_create.restype = ctypes.c_void_p
    lib.rc64_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.rc64_decoder_destroy.restype = None
    lib.rc64_decoder_decode_probabilities.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.rc64_decoder_decode_probabilities.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(payload)
    handle = lib.rc64_decoder_create(buffer, len(payload))
    if not handle:
        raise MemoryError("native arithmetic allocation failed")
    python = primitive.ArithmeticDecoder(payload)
    outputs, count = [], 0
    try:
        native = ctypes.cast(handle, ctypes.POINTER(DecoderState)).contents
        initial = {name: int(getattr(native, name)) for name in ("low", "high", "code", "bit_position")}
        if any(getattr(python, name) != value for name, value in initial.items()):
            raise ValueError("arithmetic initialization differs")
        if probability_path is not None:
            probabilities = np.ascontiguousarray(np.load(probability_path, allow_pickle=False), dtype=np.float32)
            if probabilities.ndim != 2 or probabilities.shape[1] != 5 or not 0 < len(probabilities) <= 1_000_000:
                raise ValueError("prefix probabilities must have shape [1..1000000, 5]")
            outputs.append(save_array(root / "real_prefix_probabilities.npy", probabilities))
            expected = np.empty(len(probabilities), dtype=np.int32)
            status = lib.rc64_decoder_decode_probabilities(handle, probabilities.ctypes.data, len(expected), expected.ctypes.data)
            if status:
                raise ValueError(f"native arithmetic decode refused: {status}")
            actual = python.decode(probabilities)
            outputs.extend((save_array(root / "range_native.npy", expected), save_array(root / "range_python.npy", actual)))
            if not np.array_equal(actual, expected):
                raise ValueError("arithmetic decoded symbols differ")
            count = len(actual)
        final = {name: int(getattr(native, name)) for name in initial}
        equal = all(getattr(python, name) == value for name, value in final.items())
        if not equal:
            raise ValueError("arithmetic final state differs")
        result = {"state_equal": equal, "initial": initial, "final": final,
                  "decoded_symbols": count, "outputs": outputs,
                  "scope": "actual prefix probabilities" if count else "actual stream initialization only"}
        save_json(root / "range.json", result)
        return result
    finally:
        lib.rc64_decoder_destroy(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.resume_from.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 100 * 1024 * 1024:
        raise ValueError("need 100 MiB of free retention space")
    if not args.frames or len(args.frames) > 32 or any(f < 0 or f >= 600 for f in args.frames):
        raise ValueError("supply between 1 and 32 actual frame indices")
    binding = {"runtime_archive": fact(args.runtime / "archive.zip"), "field": fact(args.field),
               "frames": args.frames, "source": fact(Path(primitive.__file__)), "harness": fact(Path(__file__)),
               "seed": args.seed, "probabilities": None if args.probabilities is None else fact(args.probabilities)}
    binding["oracle_sources"] = [fact(args.runtime / "runtime/rlc1_geometry.c"),
                                 fact(args.runtime / "runtime/entropy/rc64_backend.c")]
    if args.checkpoint_receipt is not None:
        checkpoint = json.loads(args.checkpoint_receipt.read_text())
        if (checkpoint["path"] != str(args.field.resolve())
                or checkpoint["sha256"] != binding["field"]["sha256"]
                or max(args.frames) >= checkpoint["frame"]):
            raise ValueError("field or selected frames differ from the checkpoint receipt")
        binding["checkpoint_receipt"] = checkpoint
    manifest = root / "binding.json"
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError("resume inputs or source changed; use a new output directory")
    save_json(manifest, binding)
    builds = [build(args.runtime / "runtime/rlc1_geometry.c", root / "geometry.so"),
              build(args.runtime / "runtime/entropy/rc64_backend.c", root / "range.so")]
    with zipfile.ZipFile(args.runtime / "archive.zip") as archive:
        outer = archive.read("p")
    (root / "archive_member_p.bin").write_bytes(outer)
    header = struct.unpack_from("<4sBBBBHHH", outer)
    if header[:2] != (b"RX1M", 1) or not (header[4] & 128):
        raise ValueError("archive does not contain the expected counted mixer")
    tail = outer[14 + sum(header[5:]) + 96:]
    if tail[:5] != b"RLC1\x01":
        raise ValueError("unexpected token mixer framing")
    config, payload = tail[45:64], tail[64:]
    if args.checkpoint_receipt is not None and hashlib.sha256(payload).hexdigest() != checkpoint["binding"]["stream"]:
        raise ValueError("checkpoint token stream differs from this archive")
    (root / "geometry_config.bin").write_bytes(config)
    (root / "arithmetic_stream.bin").write_bytes(payload)
    lines = set()

    def trace(frame, event, _argument):
        if event == "line" and frame.f_code.co_filename == primitive.__file__:
            lines.add(frame.f_lineno)
        return trace

    sys.settrace(trace)
    try:
        arithmetic = check_range(root / "range.so", payload, root, args.probabilities)
        field = np.load(args.field, allow_pickle=False)
        token_array = field["tokens"] if hasattr(field, "files") and "tokens" in field.files else None
        rows = []
        try:
            for frame in args.frames:
                receipt = root / f"frame_{frame:03d}.json"
                if receipt.exists():
                    previous_result = json.loads(receipt.read_text())
                    if not previous_result["equal"] or any(fact(Path(row["path"])) != row for row in previous_result["outputs"]):
                        raise ValueError("retained geometry receipt or payload differs")
                    rows.append(previous_result)
                    continue
                start = time.monotonic()
                def plane(index):
                    value = token_array[index] if token_array is not None else field[str(index)] if hasattr(field, "files") else field[index]
                    return np.asarray(value, dtype=np.uint8)

                current = plane(frame)
                previous = None if frame == 0 else plane(frame - 1)
                outputs = [save_array(root / f"frame_{frame:03d}_input.npy", current)]
                if previous is not None:
                    outputs.append(save_array(root / f"frame_{frame:03d}_previous.npy", previous))
                reference, native = primitive.CausalGeometry(previous, config), NativeGeometry(root / "geometry.so", previous, config)
                expected, actual = np.empty((384, 512), np.uint8), np.empty((384, 512), np.uint8)
                try:
                    for positions in primitive.GROUP_POSITIONS:
                        expected.ravel()[positions] = native.contexts(positions)
                        actual.ravel()[positions] = reference.contexts(positions)
                        symbols = current.ravel()[positions]
                        native.observe(positions, symbols)
                        reference.observe(positions, symbols)
                finally:
                    native.close()
                outputs.extend((save_array(root / f"frame_{frame:03d}_native.npy", expected),
                                save_array(root / f"frame_{frame:03d}_python.npy", actual)))
                mismatches = int(np.count_nonzero(actual != expected))
                result = {"frame": frame, "positions": int(actual.size), "groups": 190,
                          "equal": mismatches == 0, "mismatches": mismatches,
                          "seconds_with_coverage": time.monotonic() - start, "outputs": outputs}
                save_json(receipt, result)
                rows.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                if mismatches:
                    raise ValueError("native/Python geometry mismatch")
        finally:
            if hasattr(field, "close"):
                field.close()
    finally:
        sys.settrace(None)
        coverage_path = root / "primitive_lines.json"
        prior_lines = json.loads(coverage_path.read_text())["lines"] if coverage_path.exists() else []
        save_json(coverage_path, {"file": str(Path(primitive.__file__).resolve()), "lines": sorted(lines | set(prior_lines))})
    files = [fact(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "complete.json"
             and not p.name.startswith("._") and not p.name.endswith(".pending")]
    retained = sum(row["bytes"] for row in files)
    if retained > 100 * 1024 * 1024:
        raise ValueError("retention exceeds 100 MiB; keep artifacts and report blocker")
    save_json(root / "complete.json", {"geometry": rows, "arithmetic": arithmetic, "builds": builds,
              "retained_bytes": retained, "files": files, "score_claim": False,
              "axis": "[macOS-CPU advisory / primitive parity only]", "authority": "instance"})


if __name__ == "__main__":
    main()
