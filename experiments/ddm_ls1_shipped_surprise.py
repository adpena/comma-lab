#!/usr/bin/env python3
"""Observe the unchanged move-44 public token decoder; retain every coding row.

Research only, n600, scorer-free. Copies sources, never edits an incumbent tree.
All outputs and complete receiver restart state live in the owned SSD store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ls1")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime")
FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
ENCODE = SOURCE.parent / "encode/RESULT.json"
ARCHIVE_SHA = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
N, H, W, K, TOTAL = 600, 384, 512, 5, 1 << 31
SEED = 20260910
AXIS = "[macOS-CPU advisory / scorer-free n600 receiver probabilities]"


def fact(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(4 << 20), b""):
            digest.update(data)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def admit(path, need=0):
    """Only our fresh output tree is writable; all retained evidence stays kept."""
    if not Path(path).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write escaped ddm_ls1")
    if shutil.disk_usage(ROOT).free < (16 << 30) + need:
        raise RuntimeError("STORAGE_BLOCK: retain all evidence; no deletion authorized")
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def record(path, value):
    admit(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def arrays(path, values):
    """Retain payload first; an immutable replay must reproduce every array."""
    admit(path, sum(v.nbytes for v in values.values()))
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError(f"retained replay differs: {path}")
    else:
        temporary = path.with_suffix(".npz.new")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    record(path.with_suffix(".json"), fact(path))


def prepare():
    """Pin the encoder binding, archive and unchanged runtime copy before work."""
    field = fact(FIELD)
    receipt = json.loads(ENCODE.read_text())
    bound = receipt["binding"]["binding"]["field"]
    if field["sha256"] != FIELD_SHA or field["bytes"] != N * H * W:
        raise ValueError("wrong shipped field")
    if any(field[k] != bound[k] for k in ("bytes", "sha256")) or fact(bound["path"]) != bound:
        raise ValueError("encoder field custody mismatch")
    if fact(SOURCE / "archive.zip")["sha256"] != ARCHIVE_SHA:
        raise ValueError("wrong move44 archive")
    copied = {}
    for source in sorted(SOURCE.rglob("*")):
        if not source.is_file() or source.suffix not in (".py", ".c", ".sh", ".zip", ".json", ".txt", ".md"):
            continue
        relative = source.relative_to(SOURCE)
        destination = ROOT / "runtime_copy" / relative
        admit(destination, source.stat().st_size)
        if not destination.exists():
            shutil.copyfile(source, destination)
        src, dst = fact(source), fact(destination)
        if src["sha256"] != dst["sha256"]:
            raise ValueError("runtime source drift")
        copied[str(relative)] = dict(source=src, copy=dst)
    pins = dict(field=field, encoder_receipt=fact(ENCODE), sources=copied,
                producer=fact(Path(__file__)), seed=SEED, axis=AXIS,
                score_claim=False, cleanup="KEEP: all bulk is required retained evidence; no scratch raw render or candidates",
                reserve_bytes=16 << 30, git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    path = ROOT / "INPUTS.json"
    if path.exists():
        old = json.loads(path.read_text())
        for name in ("field", "encoder_receipt", "sources", "producer", "seed"):
            if old[name] != pins[name]:
                raise ValueError("restart inputs changed: " + name)
    else:
        record(path, pins)
    return pins


def trace(resume_from):
    """Decode real RC64 bytes and observe only after the receiver chooses symbols."""
    if Path(resume_from).resolve() != (ROOT / "receiver_checkpoints").resolve():
        raise ValueError("resume state must be this arm's receiver checkpoints")
    pins = prepare()
    import torch

    from experiments import ddm_jg2_tail_reencode as jg2
    from experiments.ddm_tc1_public_proof import build_libraries

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = ROOT / "runtime_copy"
    builds = build_libraries(runtime, ROOT / "native")
    lib = ROOT / "native/geometry.dylib"
    command = ["cc", "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off",
               "-fno-fast-math", str(runtime / "runtime/rlc1_geometry.c"), "-o", str(lib)]
    build_record = ROOT / "native/geometry.json"
    if not build_record.exists():
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (ROOT / "native/geometry.build.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        record(build_record, dict(argv=command, library=fact(lib)))
    elif json.loads(build_record.read_text())["library"] != fact(lib):
        raise ValueError("geometry library changed")
    os.environ["RLC1_GEOMETRY_LIBRARY"] = str(lib)
    os.environ["TC1_RECEIVER_CHECKPOINT_DIR"] = str(resume_from)
    os.environ["TC1_RECEIVER_STOP_AFTER"] = "600"
    residual, renderer, code_dir = jg2.load_runtime(runtime)
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.rlc1_mixer import LaneMixer
    from runtime.tc1_shared_mixer import frequencies

    target = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    observer = dict(freq=np.empty((H * W, K), dtype=np.uint32),
                    symbols=np.empty(H * W, dtype=np.uint8),
                    seen=np.zeros(H * W, dtype=bool), positions=None)
    original_coding, original_decode, original_end = LaneMixer.coding, NativeDecoder.decode, LaneMixer.end_frame

    def coding(self, rows, positions, plane, previous):
        result = original_coding(self, rows, positions, plane, previous)
        if observer["positions"] is not None or observer["seen"][positions].any():
            raise ValueError("observer group sequence")
        observer["positions"] = positions.copy()
        return result

    def decode(self, probabilities):
        symbols = original_decode(self, probabilities)
        positions = observer["positions"]
        if positions is None or len(positions) != len(symbols):
            raise ValueError("decoder observer missing positions")
        observer["freq"][positions] = frequencies(probabilities).astype(np.uint32)
        observer["symbols"][positions] = symbols
        observer["seen"][positions] = True
        observer["positions"] = None
        return symbols

    def end(self, plane, previous):
        frame = self.frame
        original_end(self, plane, previous)
        if not observer["seen"].all() or observer["positions"] is not None:
            raise ValueError("incomplete observed plane")
        np.testing.assert_array_equal(plane, target[frame])
        np.testing.assert_array_equal(observer["symbols"].reshape(H, W), plane)
        freq = observer["freq"]
        if not np.all(freq.sum(axis=1, dtype=np.uint64) == TOTAL):
            raise ValueError("frequency mass mismatch")
        selected = freq[np.arange(H * W), observer["symbols"]]
        bits = -np.log2(selected.astype(np.float64) / TOTAL)
        arrays(ROOT / "rows" / f"frame_{frame:04d}.npz", dict(
            frequencies=freq.copy(), symbols=observer["symbols"].copy(), bits=bits))
        observer["seen"].fill(False)
        if (frame + 1) % 25 == 0:
            print(json.dumps(dict(observed_pairs=frame + 1)), flush=True)

    LaneMixer.coding, NativeDecoder.decode, LaneMixer.end_frame = coding, decode, end
    parts = residual.read_residual_archive(runtime / "archive.zip")
    if len(parts.token_stream) != 119749 or len(parts.tc1_weights) != 60:
        raise ValueError("move44 stream/config census differs")
    tokens, report = residual.decode_production_tokens(parts, renderer, code_dir, torch.device("cpu"))
    arrays(ROOT / "decoded.npz", dict(tokens=tokens.numpy()))
    if hashlib.sha256(tokens.numpy().tobytes()).hexdigest() != FIELD_SHA:
        raise ValueError("full decoded field differs")
    bits, row_facts = 0.0, []
    for frame in range(N):
        path = ROOT / "rows" / f"frame_{frame:04d}.npz"
        row_fact = fact(path)
        if row_fact != json.loads(path.with_suffix(".json").read_text()):
            raise ValueError("row hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            bits += float(data["bits"].sum())
        row_facts.append(row_fact)
    difference = len(parts.token_stream) - bits / 8
    if abs(difference) / len(parts.token_stream) > 0.005:
        raise ValueError("INSTRUMENT_FALSIFIED: stream reconciliation exceeds 0.5 percent")
    record(ROOT / "TRACE.json", dict(axis=AXIS, score_claim=False, full_n600=True,
        symbols=N * H * W, bits=bits, ideal_bytes=bits / 8, stream_bytes=len(parts.token_stream),
        framing_bytes=difference, tail_bytes=119909, prefix_bytes=96, rider_bytes=64,
        field_identity=True, public_decoder_report=report, builds=builds, rows=row_facts,
        inputs=fact(ROOT / "INPUTS.json"), field=pins["field"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True)
    trace(parser.parse_args().resume_from)
