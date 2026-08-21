#!/usr/bin/env python3
"""ddm_fx5 -- lockstep Python-vs-C parity for the E1 19-member corrector.

WHY THIS EXISTS, AND WHAT IT IS NOT.

``ddm_fx5`` extends the rc2 receiver's free corrector from ``ddm_fx2``'s frozen
13-member D1 build to its raced 19-member E1 build.  The extension lands in TWO
places that must agree bit-for-bit: ``runtime/fx2_model_axis_corrector.py`` (the
Python reference) and ``runtime/f26_corrector_native.c`` (the C the receiver
actually runs).  Four new rule cases were transcribed by hand from the Python
closures into the C ``family_rule_index`` switch.  A transcription slip there is
exactly the ``ddm_rr2`` failure shape -- the row still looks plausible, the coder
desynchronises, and the score comes back 27.83.

The DEFINITIVE gate is the full n600 decode of the candidate archive: the encoder
runs the PYTHON corrector and the receiver runs the C one, so a byte-identical
decoded token field proves Python == C over 600 frames of real accumulated state.
This script is not that gate and does not claim to be.  It is the CHEAP
PRE-CHECK that stops a hand-transcription bug from costing a 14-minute decode --
the same split ``ddm_rr8`` made for the original port, for the same reason.

WHAT IT DRIVES.  Real corrector inputs are only available from a real decode, and
a real decode of the candidate under the 13-member law would desynchronise.  So
this drives SYNTHETIC but STRUCTURED state: probability rows spanning the
confident/uncertain range, symbols that both agree and disagree with the model
argmax, and a wavefront group order taken from the receiver's OWN rule
``group(x, y) = (x & 63) + 2 * (y & 63)``.  That exercises every feature the 19
rules index on -- cls, ubin, agree1, agree2, run, boundary, spatial, spatial4,
homog -- across cold cells, warm cells and the recency window.

WHAT IT PROVES AND WHAT IT DOES NOT.  Passing proves the two implementations
compute the same coding row and carry the same tables on the inputs driven here,
which is where a mis-transcribed rule index shows up immediately.  It does NOT
exercise a table state only 600 frames of accumulation reaches (the 4,096 recency
halving, the weight clamp, the deep surprise bins).  The full-field decode
remains the gate.

Comparison is BYTE-EXACT, never ``allclose``: the RC64 backend turns a row into a
frequency with ``(uint64_t)(value * 2**31)``, so one float32 ULP moves a
frequency by up to 128 counts and desynchronises the decoder from there on.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PLANE_H, PLANE_W = 384, 512
PLANE = PLANE_H * PLANE_W
NUM_CLASSES = 5

# C table selectors, mirroring ddm_rr8_corrector_parity.
TABLE_FAMILY_COUNTS = 0
TABLE_FAMILY_HITS = 1
TABLE_FAMILY_PHAT = 2
TABLE_WEIGHTS = 3
TABLE_MISS_COUNTS = 4
TABLE_MISS_EXPECT = 5
TABLE_MISS_SEEN = 6
TABLE_RUN = 7


class ParityError(SystemExit):
    """Python and C disagree; the candidate must not be sealed."""


def _import_tree(tree: Path):
    """Import ``runtime`` from the candidate tree, ahead of any sibling copy."""
    root = str(Path(tree).resolve())
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for name in [n for n in sys.modules if n == "runtime" or n.startswith("runtime.")]:
        del sys.modules[name]
    return importlib.import_module("runtime.free_corrector")


def wavefront_groups() -> list[np.ndarray]:
    """The receiver's own 190 causal groups: ``group(x,y) = (x&63) + 2*(y&63)``."""
    ys, xs = np.mgrid[0:PLANE_H, 0:PLANE_W]
    index = (xs & 63) + 2 * (ys & 63)
    flat = index.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_index = flat[order]
    boundaries = np.flatnonzero(np.diff(sorted_index)) + 1
    return np.split(order.astype(np.int64), boundaries)


def synth_frame(rng: np.random.Generator, frame: int) -> dict[str, np.ndarray]:
    """One frame of structured inputs: probabilities, symbols, boundary buckets."""
    # Probabilities spanning the whole confidence range.  A stream that is only
    # confident never reaches the deep surprise bins the new members index on.
    logits = rng.normal(0.0, 1.0 + 2.0 * ((frame % 3) + 1), size=(PLANE, NUM_CLASSES))
    logits[:, 0] += 3.0  # the real field is class-0 heavy (Road ~23%, and it is the hub)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    predicted = prob.argmax(axis=1).astype(np.int64)
    # ~0.2% disagreement, matching the shipped body's 99.81% hit rate, so the
    # miss sector and the hit sector are both exercised at realistic weight.
    flip = rng.random(PLANE) < 0.002
    symbols = predicted.copy()
    symbols[flip] = (symbols[flip] + 1 + rng.integers(0, NUM_CLASSES - 1, flip.sum())) % NUM_CLASSES

    boundary = rng.integers(0, 5, size=PLANE).astype(np.int64)
    return {"prob": prob, "predicted": predicted, "symbols": symbols, "boundary": boundary}


def python_tables(corrector) -> dict[str, np.ndarray]:
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


def native_tables(corrector, n_families: int) -> dict[str, np.ndarray]:
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


def run_parity(tree: Path, library: Path, frames: int, seed: int) -> dict[str, Any]:
    free_corrector = _import_tree(tree)
    from runtime import native_free_corrector as binding

    python = free_corrector.FreeCorrector(PLANE)
    native = binding.NativeFreeCorrector(PLANE, Path(library))

    n_families = len(python.families)
    if n_families != 19:
        raise ParityError(
            f"REFUSING: the tree carries {n_families} members, not the 19 this arm "
            "builds. Point --tree at the ddm_fx5 candidate runtime."
        )

    groups = wavefront_groups()
    rng = np.random.default_rng(seed)
    rows_checked = 0
    groups_checked = 0

    for frame in range(frames):
        data = synth_frame(rng, frame)
        python.begin_frame(data["boundary"])
        native.begin_frame(data["boundary"])

        for positions in groups:
            probability = np.ascontiguousarray(data["prob"][positions])
            predicted = np.ascontiguousarray(data["predicted"][positions])
            symbols = np.ascontiguousarray(data["symbols"][positions])

            py_state = python.group_state(probability, predicted, positions)
            nat_state = native.group_state(probability, predicted, positions)
            py_row = python.coding_row(py_state)
            nat_row = native.coding_row(nat_state)

            if py_row.shape != nat_row.shape or py_row.dtype != nat_row.dtype:
                raise ParityError(
                    f"frame {frame}: coding_row shape/dtype mismatch "
                    f"{py_row.shape}/{py_row.dtype} vs {nat_row.shape}/{nat_row.dtype}"
                )
            if py_row.tobytes() != nat_row.tobytes():
                left = py_row.reshape(-1)
                right = nat_row.reshape(-1)
                differing = np.flatnonzero(
                    py_row.view(np.uint32).reshape(-1) != nat_row.view(np.uint32).reshape(-1)
                )
                first = int(differing[0])
                raise ParityError(
                    f"PARITY FAILED at frame {frame}: coding_row differs at "
                    f"{differing.size} of {left.size} float32 slots; first at flat "
                    f"index {first} (row {first // NUM_CLASSES}, class "
                    f"{first % NUM_CLASSES}): python={left[first]!r} "
                    f"native={right[first]!r}. A mis-transcribed rule index in "
                    "family_rule_index is the first thing to check."
                )

            python.observe(py_state, symbols)
            native.observe(nat_state, symbols)
            groups_checked += 1
            rows_checked += positions.size

        python.end_frame(data["symbols"].astype(np.uint8))
        native.end_frame(data["symbols"].astype(np.uint8))

        py_tables = python_tables(python)
        nat_tables = native_tables(native, n_families)
        if set(py_tables) != set(nat_tables):
            raise ParityError("table key sets differ between the two correctors")
        for key in sorted(py_tables):
            left = np.asarray(py_tables[key], dtype=np.int64).reshape(-1)
            right = np.asarray(nat_tables[key], dtype=np.int64).reshape(-1)
            if left.size != right.size:
                raise ParityError(
                    f"frame {frame}: table {key} size {left.size} vs {right.size}"
                )
            if not np.array_equal(left, right):
                differing = np.flatnonzero(left != right)
                first = int(differing[0])
                raise ParityError(
                    f"PARITY FAILED at frame {frame}: table {key} differs at "
                    f"{differing.size} of {left.size} cells; first at {first}: "
                    f"python={left[first]} native={right[first]}"
                )

    native.close()
    return {
        "schema": "ddm_fx5_parity_e1.v1",
        "verdict": "IDENTICAL",
        "tree": str(tree),
        "library": str(library),
        "members": n_families,
        "frames": frames,
        "seed": seed,
        "groups_checked": groups_checked,
        "rows_checked": rows_checked,
        "comparison": "byte-exact on coding_row and on every family/mixer/miss/run table",
        "scope": (
            "structured synthetic inputs; the full n600 decode of the candidate "
            "archive remains the definitive Python-vs-C gate"
        ),
        "axis": "[macOS-CPU advisory / implementation parity only]",
        "score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", required=True, help="the ddm_fx5 candidate runtime dir")
    parser.add_argument("--library", required=True, help="the built E1 .so")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    result = run_parity(Path(args.tree), Path(args.library), args.frames, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
