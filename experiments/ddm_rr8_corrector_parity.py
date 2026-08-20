#!/usr/bin/env python3
"""ddm_rr8 - capture, differential-parity and bench the C free corrector.

WHY A TRACE HARNESS AND NOT JUST A FULL DECODE.  The identity gate for this port is the
whole n600 field, and that run costs ~15 minutes.  Iterating a 2,000-line C port against a
15-minute oracle is how a port takes a week.  So the instrument is split:

  ``capture``  run the REAL decoder once, recording every corrector input and the symbols
               the RC64 coder actually returned, for the first K frames.  Real inputs, not
               synthesised ones -- a corrector fed random probabilities exercises none of the
               cold-context, saturation or recency paths that decide identity.
  ``parity``   replay that trace through the shipped Python corrector and the C corrector in
               LOCKSTEP, comparing the emitted row bit-for-bit at EVERY group, and comparing
               the live tables at every frame boundary.
  ``bench``    time both on the same trace, same host, same process.

WHY STATE IS COMPARED AND NOT ONLY OUTPUT.  A corrector whose tables have already diverged
still emits an identical row wherever both sides are cold, because a cold cell's multiplier
is exactly 1.0 by construction.  An output-only comparison would therefore pass for a long
time after the run is already lost.  ``parity`` compares the 13 members' counts/hits/phat_q,
the 4,000x13 mixer weights, all three within-miss tables and the per-pixel run field.

WHAT PASSING HERE DOES AND DOES NOT PROVE.  It proves the C reproduces the Python on the
REAL first K frames, tables included.  It does NOT replace the full-field run: the trace
cannot exercise a table state that only 600 frames of accumulation reaches (the recency
halving at 4,096, the weight clamp, the deep surprise bins).  The full n600 decode against
the four published anchors remains the gate; this is what makes reaching it affordable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TREE = REPO / "submissions" / "robust_current" / "jg5_sub015_runtime" / "runtime"
PLANE = 384 * 512
NUM_CLASSES = 5

#: ``f26_corrector_table`` selectors, mirrored from the C.
TABLE_FAMILY_COUNTS = 0
TABLE_FAMILY_HITS = 1
TABLE_FAMILY_PHAT = 2
TABLE_WEIGHTS = 3
TABLE_MISS_COUNTS = 4
TABLE_MISS_EXPECT = 5
TABLE_MISS_SEEN = 6
TABLE_RUN = 7


class ParityError(RuntimeError):
    """The C corrector and the shipped Python corrector disagree."""


class _CaptureStop(Exception):
    """Controlled unwind once the requested frame count is recorded."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _import_tree(tree: Path):
    """Import the shipped runtime package READ-ONLY, exactly as ``inflate.py`` does."""
    tree = tree.resolve()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    import runtime.free_corrector as free_corrector

    return free_corrector


# --- capture --------------------------------------------------------------------------


class _RecordingCorrector:
    """Wraps the real corrector, recording its inputs and the decoder's symbols.

    It DELEGATES every call unchanged, so the decode it records is the decode that would
    have happened.  Recording a corrector by re-deriving its inputs afterwards would record
    a different run.
    """

    def __init__(self, inner, output: Path, frames: int) -> None:
        self.inner = inner
        self.output = output
        self.frames = frames
        self.frame_index = 0
        self._reset()

    def _reset(self) -> None:
        self.boundary: np.ndarray | None = None
        self.prob: list[np.ndarray] = []
        self.predicted: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.symbols: list[np.ndarray] = []

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        self.boundary = np.asarray(boundary_flat, dtype=np.uint8).reshape(-1).copy()
        self.inner.begin_frame(boundary_flat)

    def group_state(self, probability, predicted, positions):
        self.prob.append(np.ascontiguousarray(probability, dtype=np.float32).copy())
        self.predicted.append(np.asarray(predicted, dtype=np.int64).reshape(-1).copy())
        self.positions.append(np.asarray(positions, dtype=np.int64).reshape(-1).copy())
        return self.inner.group_state(probability, predicted, positions)

    def coding_row(self, state):
        return self.inner.coding_row(state)

    def observe(self, state, symbols) -> None:
        self.symbols.append(np.asarray(symbols, dtype=np.int64).reshape(-1).copy())
        self.inner.observe(state, symbols)

    def end_frame(self, tokens_flat: np.ndarray) -> None:
        tokens = np.asarray(tokens_flat, dtype=np.uint8).reshape(-1).copy()
        sizes = np.array([part.shape[0] for part in self.prob], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        path = self.output / f"frame_{self.frame_index:04d}.npz"
        np.savez(
            path,
            boundary=self.boundary,
            tokens=tokens,
            offsets=offsets,
            prob=np.concatenate(self.prob, axis=0),
            predicted=np.concatenate(self.predicted),
            positions=np.concatenate(self.positions),
            symbols=np.concatenate(self.symbols),
        )
        self.inner.end_frame(tokens_flat)
        self.frame_index += 1
        self._reset()
        if self.frame_index >= self.frames:
            raise _CaptureStop(f"captured {self.frame_index} frames")


def capture(tree: Path, output: Path, frames: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    free_corrector = _import_tree(tree)
    from runtime.f26_inflate import inflate_archive

    original = free_corrector.FreeCorrector
    recorder: dict[str, _RecordingCorrector] = {}

    def factory(plane: int):
        wrapper = _RecordingCorrector(original(plane), output, frames)
        recorder["value"] = wrapper
        return wrapper

    free_corrector.FreeCorrector = factory
    started = time.perf_counter()
    try:
        inflate_archive(
            tree / "archive.zip",
            output / "unused.raw",
            renderer_dir=tree / "cpr1",
            device_name="cpu",
            num_threads=4,
            checkpoint_dir=output / ".checkpoints",
        )
    except _CaptureStop:
        pass
    finally:
        free_corrector.FreeCorrector = original
    elapsed = time.perf_counter() - started

    written = sorted(output.glob("frame_*.npz"))
    manifest = {
        "schema": "ddm_rr8_corrector_trace.v1",
        "tree": str(tree),
        "frames_requested": frames,
        "frames_written": len(written),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in written
        ],
        "capture_seconds": elapsed,
        "notes": [
            "Real decoder inputs and REAL RC64 symbols, recorded by delegation so the "
            "recorded decode is the decode that would have happened.",
            "Advisory instrument only. It carries no score and moves no byte.",
        ],
    }
    _atomic_json(output / "trace_manifest.json", manifest)
    return manifest


# --- replay ---------------------------------------------------------------------------


def _load_frames(trace: Path) -> list[dict[str, np.ndarray]]:
    frames = []
    for path in sorted(trace.glob("frame_*.npz")):
        with np.load(path) as data:
            frames.append({key: data[key] for key in data.files})
    if not frames:
        raise ParityError(f"no captured frames under {trace}")
    return frames


def _python_corrector(tree: Path):
    free_corrector = _import_tree(tree)
    return free_corrector.FreeCorrector(PLANE)


def _native_corrector(tree: Path, library: Path):
    """Load the SHIPPED wrapper file, unpatched.

    ``native_free_corrector`` resolves its sibling corrector modules relatively in-tree and
    absolutely from a checkout, so the file exercised here is byte-identical to the one the
    stager copies into the candidate tree -- including its config-drift guard, which is the
    part most worth testing.
    """
    _import_tree(tree)
    source = (REPO / "runtime-rs" / "native" / "f26-corrector").resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import native_free_corrector as binding

    return binding.NativeFreeCorrector(PLANE, library)


def _python_tables(corrector) -> dict[str, np.ndarray]:
    """The live Python tables, keyed identically to the C selectors."""
    tables: dict[str, np.ndarray] = {}
    for position, family in enumerate(corrector.families):
        tables[f"counts/{position}"] = family.counts
        tables[f"hits/{position}"] = family.hits
        tables[f"phat/{position}"] = family.phat_q
    tables["weights"] = corrector.weights.reshape(-1)
    tables["miss_counts"] = corrector._miss_counts.reshape(-1)
    tables["miss_expect"] = corrector._miss_expect.reshape(-1)
    tables["miss_seen"] = corrector._miss_seen
    tables["run"] = corrector.run
    return tables


def _native_tables(corrector, n_families: int) -> dict[str, np.ndarray]:
    tables: dict[str, np.ndarray] = {}
    for position in range(n_families):
        tables[f"counts/{position}"] = corrector.table(TABLE_FAMILY_COUNTS, position)
        tables[f"hits/{position}"] = corrector.table(TABLE_FAMILY_HITS, position)
        tables[f"phat/{position}"] = corrector.table(TABLE_FAMILY_PHAT, position)
    tables["weights"] = corrector.table(TABLE_WEIGHTS)
    tables["miss_counts"] = corrector.table(TABLE_MISS_COUNTS)
    tables["miss_expect"] = corrector.table(TABLE_MISS_EXPECT)
    tables["miss_seen"] = corrector.table(TABLE_MISS_SEEN)
    tables["run"] = corrector.table(TABLE_RUN)
    return tables


def parity(tree: Path, trace: Path, library: Path) -> dict[str, Any]:
    frames = _load_frames(trace)
    python = _python_corrector(tree)
    native = _native_corrector(tree, library)

    groups_checked = 0
    rows_checked = 0
    for frame_index, frame in enumerate(frames):
        python.begin_frame(frame["boundary"])
        native.begin_frame(frame["boundary"])
        offsets = frame["offsets"]
        for group in range(offsets.size - 1):
            start, stop = int(offsets[group]), int(offsets[group + 1])
            probability = frame["prob"][start:stop]
            predicted = frame["predicted"][start:stop]
            positions = frame["positions"][start:stop]
            symbols = frame["symbols"][start:stop]

            py_state = python.group_state(probability, predicted, positions)
            nat_state = native.group_state(probability, predicted, positions)
            py_row = python.coding_row(py_state)
            nat_row = native.coding_row(nat_state)

            # BYTE comparison, not np.allclose.  The RC64 backend turns a row into an
            # integer frequency with ``(uint64_t)(value * 2**31)``, so one float32 ULP moves
            # a frequency by up to 128 counts and desynchronises the decoder from there on.
            # "Close" is not a category this gate has.
            if py_row.shape != nat_row.shape or py_row.dtype != nat_row.dtype:
                raise ParityError(
                    f"frame {frame_index} group {group}: shape/dtype mismatch "
                    f"{py_row.shape}/{py_row.dtype} vs {nat_row.shape}/{nat_row.dtype}"
                )
            if py_row.tobytes() != nat_row.tobytes():
                differing = np.flatnonzero(
                    py_row.view(np.uint32).reshape(-1) != nat_row.view(np.uint32).reshape(-1)
                )
                first = int(differing[0])
                raise ParityError(
                    f"frame {frame_index} group {group}: coding_row differs at "
                    f"{differing.size} of {py_row.size} float32 slots; first at flat index "
                    f"{first} (row {first // NUM_CLASSES}, class {first % NUM_CLASSES}): "
                    f"python={py_row.reshape(-1)[first]!r} native={nat_row.reshape(-1)[first]!r}"
                )

            python.observe(py_state, symbols)
            native.observe(nat_state, symbols)
            groups_checked += 1
            rows_checked += stop - start

        python.end_frame(frame["tokens"])
        native.end_frame(frame["tokens"])

        py_tables = _python_tables(python)
        nat_tables = _native_tables(native, len(python.families))
        if set(py_tables) != set(nat_tables):
            raise ParityError("table key sets differ between the two correctors")
        for key in sorted(py_tables):
            left = np.asarray(py_tables[key], dtype=np.int64).reshape(-1)
            right = np.asarray(nat_tables[key], dtype=np.int64).reshape(-1)
            if left.size != right.size:
                raise ParityError(
                    f"frame {frame_index}: table {key} size {left.size} vs {right.size}"
                )
            if not np.array_equal(left, right):
                differing = np.flatnonzero(left != right)
                first = int(differing[0])
                raise ParityError(
                    f"frame {frame_index}: table {key} differs at {differing.size} of "
                    f"{left.size} cells; first at {first}: "
                    f"python={left[first]} native={right[first]}"
                )

    native.close()
    return {
        "schema": "ddm_rr8_corrector_parity.v1",
        "verdict": "IDENTICAL",
        "frames": len(frames),
        "groups_checked": groups_checked,
        "rows_checked": int(rows_checked),
        "float32_slots_checked": int(rows_checked) * NUM_CLASSES,
        "tables_compared_per_frame": len(_python_tables(python)),
        "notes": [
            "coding_row compared as BYTES at every group; tables compared at every frame "
            "boundary. A cold cell emits exactly 1.0 on both sides, so an output-only "
            "comparison could pass on already-diverged tables -- hence the state check.",
            "This is a TRACE parity result on the first frames, not the full-field gate. "
            "The n600 identity run against the four published anchors is the gate.",
        ],
    }


# --- bench ----------------------------------------------------------------------------


def bench(tree: Path, trace: Path, library: Path, repeats: int) -> dict[str, Any]:
    frames = _load_frames(trace)

    def run(make) -> float:
        corrector = make()
        started = time.perf_counter()
        for frame in frames:
            corrector.begin_frame(frame["boundary"])
            offsets = frame["offsets"]
            for group in range(offsets.size - 1):
                start, stop = int(offsets[group]), int(offsets[group + 1])
                state = corrector.group_state(
                    frame["prob"][start:stop],
                    frame["predicted"][start:stop],
                    frame["positions"][start:stop],
                )
                corrector.coding_row(state)
                corrector.observe(state, frame["symbols"][start:stop])
            corrector.end_frame(frame["tokens"])
        elapsed = time.perf_counter() - started
        close = getattr(corrector, "close", None)
        if close:
            close()
        return elapsed

    python_times = [run(lambda: _python_corrector(tree)) for _ in range(repeats)]
    native_times = [run(lambda: _native_corrector(tree, library)) for _ in range(repeats)]
    python_best = min(python_times)
    native_best = min(native_times)

    rows = sum(int(frame["offsets"][-1]) for frame in frames)
    return {
        "schema": "ddm_rr8_corrector_bench.v1",
        "axis": "[macOS-CPU advisory]",
        "frames": len(frames),
        "positions_per_repeat": rows,
        "repeats": repeats,
        "python_seconds": python_times,
        "native_seconds": native_times,
        "python_best_seconds": python_best,
        "native_best_seconds": native_best,
        "speedup_best": python_best / native_best,
        "speedup_median": float(np.median(python_times) / np.median(native_times)),
        "break_even": {
            "frame_b_narrow": 2.03,
            "frame_a_wide": 2.77,
            "source": "ddm_cd1 6.4, pre-registered against the CI wall",
        },
        "notes": [
            "PORT SCOPE ONLY: group_state + coding_row + observe, the three calls ddm_cd1 "
            "measured at 917.929 s on T4. begin_frame/end_frame are included in the loop "
            "because both implementations pay them; they are 0.8 s of the 1,280 s stage.",
            "This is a LOCAL ratio on an M5 Max. ddm_cd1 6.3 MEASURED the T4 container vCPU "
            "at 4.35x slower than this core on exactly this numpy, so the local ratio is a "
            "direction, never the shipping number. Only a T4 row prices the port.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "parity", "bench"))
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "capture":
        result = capture(args.tree, args.trace, args.frames)
    else:
        if args.library is None:
            parser.error(f"{args.mode} needs --library")
        if args.mode == "parity":
            result = parity(args.tree, args.trace, args.library)
        else:
            result = bench(args.tree, args.trace, args.library, args.repeats)

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.report:
        _atomic_json(args.report, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
