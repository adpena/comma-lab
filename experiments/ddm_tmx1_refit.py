"""ddm_tmx1 — refit the tail coder's 40 counted int8 mixer weights on the CURRENT field.

The shipped 60 B RLC1 rider carries ``variant(1) + 40 int8 weights + 19 B geometry``.
The 40 weights are tc1's 35 shared-mixer coefficients (5 winning classes x 7 features:
spatial2, spatial3, previous, run, rowband, temperature, hit) fitted on the MOVE-32
field, plus tc3's 5 lane coefficients fitted on the MOVE-40 field.  The field last moved
at move 43.  This tool refits all 40 on the field the coder actually codes today.

It is a FITTING device, not an evidence device.  Three things make that honest:

* the objective is the coder's OWN ideal code length -- the RC64 frequency the shipped
  ``frequencies`` builder assigns to the symbol actually coded -- evaluated through the
  shipped cascade (``SharedMixer`` then ``LaneMixer``) imported from the priced runtime
  copy rather than re-derived;
* the price is never taken from here.  The authority is the real 600-frame encode in
  ``experiments/ddm_tmx1_mixer_price.py``;
* the held-out fold (frame parity) exists so an in-sample fit cannot pass as a win.

Two approximations are named rather than hidden.  (1) The objective runs on a retained
1-in-32 systematic sample of the 117,964,800 coded symbols, scaled by the stride.  (2) The
lane table is carried from the control encode; it is built from tc1's OUTPUT, so a change
in the 35 tc1 weights moves it at second order.  Both are fitting-side only: the exact
archive comes from the real encode, which recomputes everything.

Axis ``[macOS-CPU advisory; a fitting surrogate, scorer-free]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_tmx1/fit")
CONTROL = Path("/Volumes/VertigoDataTier/pact/ddm_tmx1/price/control47")
FIELD = Path("/Volumes/APDataStore/pact/ddm_hpr1/inputs/field.u8")
N, H, W, K, F = 600, 384, 512, 5, 7
BINS = 9
TOTAL = 1 << 31
LOW, HIGH = -128, 127


class FitError(RuntimeError):
    """A fitting input is not what the shipped object says it is."""


def load_shipped(runtime: Path):
    """Import the SHIPPED mixer math out of the priced runtime copy."""
    sys.path.insert(0, str(runtime))
    import importlib

    base = importlib.import_module("runtime.tc1_shared_mixer")
    if (base.K, base.F, base.H, base.W, base.TOTAL) != (K, F, H, W, TOTAL):
        raise FitError("shipped mixer geometry is not the one this tool assumes")
    return base


def coder_frequencies(values: np.ndarray) -> np.ndarray:
    """RC64's own frequency build, vectorised without the shipped guard raises.

    Identical arithmetic to ``tc1_shared_mixer.frequencies``: float32 row -> double,
    scaled by 2**31, truncated, floored at one, WINNER balanced to the total.  Checked
    against the shipped function before any fitting runs.
    """
    work = np.asarray(values, dtype=np.float32).astype(np.float64)
    freq = np.maximum((work * TOTAL).astype(np.int64), 1)
    winner = work.argmax(axis=1)
    freq[np.arange(len(freq)), winner] += TOTAL - freq.sum(axis=1)
    return freq


def replay_contexts(base, frames: np.ndarray, positions: np.ndarray) -> dict:
    """Rebuild every sampled position's five tc1 contexts with the SHIPPED function."""
    field = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    starts = np.searchsorted(frames, np.arange(N))
    ends = np.searchsorted(frames, np.arange(N), side="right")
    out = {name: np.empty(len(frames), dtype=np.int64) for name in base.LEVELS}
    run = np.zeros((H, W), dtype=np.uint8)
    for frame in range(N):
        plane = np.asarray(field[frame])
        previous = None if frame == 0 else np.asarray(field[frame - 1])
        lo, hi = int(starts[frame]), int(ends[frame])
        if hi > lo:
            values = base.contexts(plane, previous, run, positions[lo:hi])
            for name in base.LEVELS:
                out[name][lo:hi] = values[name]
        run = (
            np.zeros((H, W), dtype=np.uint8)
            if frame == 0
            else np.where(plane == previous, np.minimum(run + 1, 7), 0).astype(np.uint8)
        )
    return out


def build_phi(base, rows, frames, contexts, tables):
    """Assemble phi exactly as ``SharedMixer.features`` does, for every sampled position."""
    freq = coder_frequencies(rows)
    arg = freq.argmax(axis=1)
    starts = np.searchsorted(frames, np.arange(N))
    ends = np.searchsorted(frames, np.arange(N), side="right")
    sizes = [K * levels * K for levels in base.LEVELS.values()]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    phi = np.zeros((len(frames), K, F), dtype=np.int16)
    for frame in range(N):
        lo, hi = int(starts[frame]), int(ends[frame])
        if hi <= lo:
            continue
        packed = tables[frame]
        for j, (name, levels) in enumerate(base.LEVELS.items()):
            table = packed[offsets[j] : offsets[j + 1]].reshape(K * levels, K)
            phi[lo:hi, :, j] = table[arg[lo:hi] * levels + contexts[name][lo:hi]]
    phi[:, :, 5] = base.log2_fixed(freq.astype(np.float64) / TOTAL)
    phi[np.arange(len(frames)), arg, 6] = base.Q
    return phi, freq, arg


def build_lane(frames, bins, lane_tables) -> np.ndarray:
    """Per sample, the lane table row for EVERY possible tc1-output winner: (n, K, K)."""
    starts = np.searchsorted(frames, np.arange(N))
    ends = np.searchsorted(frames, np.arange(N), side="right")
    out = np.zeros((len(frames), K, K), dtype=np.int16)
    for frame in range(N):
        lo, hi = int(starts[frame]), int(ends[frame])
        if hi <= lo:
            continue
        table = lane_tables[frame].reshape(K * BINS, K)
        for k in range(K):
            out[lo:hi, k] = table[k * BINS + bins[lo:hi]]
    out[bins == 8] = 0
    return out


class Objective:
    """The shipped two-stage cascade's ideal bits over one fixed set of samples."""

    def __init__(self, base, phi, freq, arg, lane, truth):
        self.base = base
        self.phi = np.ascontiguousarray(phi, dtype=np.int32)
        self.freq = np.ascontiguousarray(freq, dtype=np.float64) / TOTAL
        self.arg = np.ascontiguousarray(arg, dtype=np.int64)
        self.lane = np.ascontiguousarray(lane, dtype=np.int32)
        self.truth = np.ascontiguousarray(truth, dtype=np.int64)
        self.index = np.arange(len(truth))
        self.step = base.Q * base.SCALE
        self.calls = 0

    def __call__(self, weights) -> float:
        weights = np.asarray(weights, dtype=np.int64)
        w35 = weights[:35].reshape(K, F)
        w5 = weights[35:].reshape(K, 1)
        if not np.all(np.any(w35, axis=1)):
            return float("inf")  # an all-zero bank routes to a fallback this rail does not price
        self.calls += 1
        exponent = np.einsum("nkf,nf->nk", self.phi, w35[self.arg].astype(np.int32))
        exponent -= exponent.max(axis=1, keepdims=True)
        integer, fraction = np.divmod(exponent.astype(np.int64), self.step)
        raw = self.freq * np.ldexp(self.base.POW2[fraction], integer.astype(np.int32))
        raw /= raw.sum(axis=1, keepdims=True)
        mixed = np.maximum(raw, np.finfo(np.float32).tiny).astype(np.float32)
        # --- the lane stage, on tc1's output, exactly as LaneMixer.coding runs it
        f1 = coder_frequencies(mixed)
        a1 = f1.argmax(axis=1)
        extra = self.lane[self.index, a1]
        exponent = extra * w5[a1].astype(np.int32)
        exponent -= exponent.max(axis=1, keepdims=True)
        integer, fraction = np.divmod(exponent.astype(np.int64), self.step)
        raw = f1.astype(np.float64) / TOTAL * np.ldexp(self.base.POW2[fraction], integer.astype(np.int32))
        raw /= raw.sum(axis=1, keepdims=True)
        result = np.maximum(raw, np.finfo(np.float32).tiny).astype(np.float32)
        inactive = ~np.any(extra, axis=1) | (w5[a1, 0] == 0)
        result[inactive] = mixed[inactive]
        final = coder_frequencies(result)
        return float(-np.log2(final[self.index, self.truth].astype(np.float64) / TOTAL).sum())


def coordinate_search(objective, start, sweeps: int, steps, log: list):
    """Deterministic coordinate descent on the int8 grid; every accepted move retained."""
    best = np.asarray(start, dtype=np.int64).copy()
    value = objective(best)
    log.append({"sweep": -1, "coordinate": None, "value": value, "weights": best.tolist()})
    for sweep in range(sweeps):
        moved = False
        for coordinate in range(len(best)):
            trials = []
            for step in steps:
                candidate = best.copy()
                candidate[coordinate] = int(np.clip(best[coordinate] + step, LOW, HIGH))
                if candidate[coordinate] == best[coordinate]:
                    continue
                trials.append((objective(candidate), candidate))
            if not trials:
                continue
            trials.sort(key=lambda item: item[0])
            if trials[0][0] < value - 1e-9:
                value, best = trials[0][0], trials[0][1]
                moved = True
                log.append({"sweep": sweep, "coordinate": coordinate, "value": value,
                            "weight": int(best[coordinate])})
        if not moved:
            break
    return best, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--search-stride", type=int, default=4,
                        help="thin the retained trace by this factor for the SEARCH only")
    parser.add_argument("--steps", type=int, nargs="+", default=[-3, -2, -1, 1, 2, 3])
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    started = time.time()
    runtime = CONTROL / "runtime_copy"
    base = load_shipped(runtime)
    price = json.loads((CONTROL / "PRICE.json").read_text())
    stride = int(price["fit_trace"]["stride"])
    with np.load(CONTROL / "retained/fit_trace.npz", allow_pickle=False) as data:
        trace = {k: data[k] for k in data.files}
    rows_sha = hashlib.sha256(trace["rows"].tobytes()).hexdigest()
    shipped_rider = (CONTROL / "retained/rider_config.bin").read_bytes()
    shipped = np.frombuffer(shipped_rider[1:41], dtype=np.int8).astype(np.int64)

    probe = trace["rows"][:200000]
    if not np.array_equal(base.frequencies(probe), coder_frequencies(probe)):
        raise FitError("the vectorised frequency build disagrees with the shipped one")

    frames = trace["frame"].astype(np.int64)
    if np.any(np.diff(frames) < 0):
        raise FitError("the retained trace is not in frame order")
    positions = trace["pos"].astype(np.int64)
    contexts = replay_contexts(base, frames, positions)
    phi, freq, arg = build_phi(base, trace["rows"], frames, contexts, trace["tables"])
    lane = build_lane(frames, trace["bins"].astype(np.int64), trace["lane_tables"])
    truth = trace["truth"].astype(np.int64)
    even = np.flatnonzero((frames & 1) == 0)
    odd = np.flatnonzero((frames & 1) == 1)

    def make(subset):
        return Objective(base, phi[subset], freq[subset], arg[subset], lane[subset], truth[subset])

    full = make(slice(None))
    on_even, on_odd = make(even), make(odd)
    thin = args.search_stride
    search_even, search_odd = make(even[::thin]), make(odd[::thin])
    search_all = make(np.arange(len(truth))[::thin])
    shipped_bits = {"all": full(shipped), "even": on_even(shipped), "odd": on_odd(shipped)}

    # THE OFFLINE CASCADE'S OWN CONTROL.  Scaled by the sampling stride, the surrogate's
    # bits under the SHIPPED weights must reproduce the EXACT in-loop ideal bits the
    # control encode accumulated from the same rows the arithmetic coder was fed.  A
    # structural error anywhere in the replay -- a context, a table, a stage order --
    # shows up here, and nothing downstream is admissible if it does not agree.
    exact = float(price["exact_ideal_bits"]["total"])
    replay = shipped_bits["all"] * stride
    agreement = abs(replay - exact) / exact
    if agreement > 0.01:
        raise FitError(
            f"OFFLINE_CASCADE_CONTROL_FAILED: replayed {replay:.1f} bits against the loop's "
            f"exact {exact:.1f} ({agreement:.4%}); the refit is not admissible"
        )

    results, logs = {}, {}
    for name, fit_search, held in (("even", search_even, on_odd), ("odd", search_odd, on_even)):
        logs[name] = []
        fitted, _ = coordinate_search(fit_search, shipped, args.sweeps, args.steps, logs[name])
        results[name] = {
            "weights": fitted.tolist(),
            "fit_bits": (on_even if name == "even" else on_odd)(fitted),
            "fit_bits_shipped": shipped_bits[name],
            "held_out_bits": held(fitted),
            "held_out_bits_shipped": shipped_bits["odd" if name == "even" else "even"],
        }
    logs["all"] = []
    final, _ = coordinate_search(search_all, shipped, args.sweeps, args.steps, logs["all"])
    payload = bytes(np.clip(final, LOW, HIGH).astype(np.int8))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights_i8.bin").write_bytes(payload)
    (args.out / "search_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    final_bits = {"all": full(final), "even": on_even(final), "odd": on_odd(final)}
    held_delta = sum(
        results[name]["held_out_bits"] - results[name]["held_out_bits_shipped"] for name in ("even", "odd")
    )
    report = {
        "schema": "ddm_tmx1_refit.v1",
        "control": str(CONTROL),
        "control_archive_sha256": price["binding"]["base_archive"]["sha256"],
        "trace_rows_sha256": rows_sha,
        "samples": int(len(truth)),
        "sample_stride": stride,
        "search_stride": thin,
        "sweeps": args.sweeps,
        "steps": args.steps,
        "shipped_weights": shipped.tolist(),
        "shipped_weights_sha256": hashlib.sha256(bytes(shipped.astype(np.int8))).hexdigest(),
        "shipped_bits": shipped_bits,
        "folds": results,
        "final_weights": final.tolist(),
        "final_weights_sha256": hashlib.sha256(payload).hexdigest(),
        "final_bits": final_bits,
        "exact_shipped_stream_bits": exact,
        "offline_cascade_control": {"replayed_bits": replay, "exact_bits": exact,
                                    "relative_gap": agreement,
                                    "rule": "stride-scaled surrogate must match the loop's exact bits"},
        "in_sample_delta_bytes_estimate": (final_bits["all"] - shipped_bits["all"]) / 8.0 * stride,
        "held_out_delta_bytes_estimate": held_delta / 8.0 * stride,
        "objective_calls": {"full": full.calls, "even": on_even.calls, "odd": on_odd.calls},
        "axis": "[macOS-CPU advisory; a fitting surrogate, scorer-free]",
        "score_claim": False,
        "wall_seconds": round(time.time() - started, 3),
    }
    (args.out / "REFIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=2))
    for name, value in results.items():
        print(name, "held-out bits", value["held_out_bits"], "vs shipped", value["held_out_bits_shipped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
