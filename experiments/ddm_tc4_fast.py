"""Incremental TC4 maps, exactly the same bins and counted weights.

Scratch is derived solely from decoded group prefixes and resets every frame.
The CLI runs real-field equivalence controls; it does not price or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from experiments import ddm_tc4_maps as maps

H, W, K = maps.H, maps.W, maps.K


def highest_bit(value):
    """Integer floor(log2(value)); caller handles the zero sentinel."""
    work = np.asarray(value, dtype=np.uint64).copy()
    result = np.zeros(work.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        upper = work >> np.uint64(shift)
        take = upper != 0
        work = np.where(take, upper, work)
        result += take * shift
    return result


class FastContextMixer(maps.ContextMixer):
    """Same TC4 model; replace full-plane map scans with incremental scratch."""

    def begin_frame(self):
        super().begin_frame()
        self.row_length = np.zeros(H * W, dtype=np.uint8)
        self.transitions = np.zeros((H // 64, W), dtype=np.uint64)

    def context_codes(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        yy, xx = positions // W, positions % W
        plane = self.old.geometry.plane.reshape(-1)
        available = xx % 64 != 0
        left_pos = np.maximum(positions - 1, 0)
        left = plane[left_pos]
        available &= left < K
        run = self.row_length[left_pos]
        bucket = np.searchsorted([1, 2, 4, 8, 16, 32, 63], run, side="left")
        run_map = np.where(available, left * 8 + bucket, 40)

        block, remainder = yy // 64, yy % 64
        all_blocks = np.arange(H // 64)[:, None]
        before = (np.left_shift(np.uint64(1), remainder.astype(np.uint64)) - np.uint64(1))[None, :]
        bits = self.transitions[:, xx]
        bits = np.where(all_blocks < block, bits, np.where(all_blocks == block, bits & before, np.uint64(0)))
        last_block = np.max(np.where(bits != 0, all_blocks, -1), axis=0)
        chosen = bits[np.maximum(last_block, 0), np.arange(len(positions))]
        endpoint = np.maximum(last_block, 0) * 64 + highest_bit(chosen)
        lower = endpoint * W + xx
        upper = np.maximum(endpoint - 1, 0) * W + xx
        vertical = np.where(last_block >= 0, plane[upper] * K + plane[lower], 25)

        above = np.maximum(positions - W, 0)
        lane = np.where(remainder != 0, self.lane_bins.reshape(-1)[above], 9)
        movable = available & (left == 3)
        state = np.where(available, 0, 8)
        state[movable] = 1 + np.searchsorted([1, 2, 4, 8, 16, 32], run[movable], side="left")
        horizon = ((yy >= 128) & (yy < 320)) * 9 + state
        return np.stack((run_map, vertical, self.temporal.reshape(-1)[positions], lane, horizon), axis=1).astype(np.uint8)

    def coding(self, rows, positions, plane, previous):
        original = self.old.coding(rows, positions, plane, previous)
        if self.temporal is None:
            self.temporal = maps.previous_map(previous)
        codes = self.context_codes(positions)
        phi, freq = self.features_from_codes(original, codes)
        self.codes[positions], self.freq[positions] = codes, freq
        self.lane_bins.reshape(-1)[positions] = self.old.bins[positions]
        return maps.mix_rows(freq, phi[:, :, self.selected], self.weights, original)

    def observe(self, positions, symbols):
        # The inherited group guard verifies exact sequence and alphabet first.
        super().observe(positions, symbols)
        positions = np.asarray(positions, dtype=np.int64)
        yy, xx = positions // W, positions % W
        plane = self.old.geometry.plane.reshape(-1)
        left = np.maximum(positions - 1, 0)
        same = (xx % 64 != 0) & (plane[left] == plane[positions])
        self.row_length[positions] = np.where(same, self.row_length[left] + 1, 1)
        # An observation can complete the pair with either known neighbour.
        for lower_y in (yy, yy + 1):
            valid = (lower_y > 0) & (lower_y < H)
            y, x = lower_y[valid], xx[valid]
            low, high = plane[y * W + x], plane[(y - 1) * W + x]
            transition = (low < K) & (high < K) & (low != high)
            y, x = y[transition], x[transition]
            bit = np.left_shift(np.uint64(1), (y % 64).astype(np.uint64))
            np.bitwise_or.at(self.transitions, (y // 64, x), bit)


def validate():
    """Retain per-frame controls and complete states on 32 random + endpoints."""
    from experiments import ddm_tc4_price as price

    root = price.ROOT.parent / "fast_controls"
    root.mkdir(parents=True, exist_ok=True)
    binding = {
        "producer": price.fact(Path(__file__)),
        "dependencies": [price.fact(Path(m.__file__)) for m in (maps, price, price.base, price.tc3, price.geometry)],
        "field": price.fact(price.SOURCE / "field.u8"),
        "archive": price.fact(price.SOURCE / "candidate_runtime/archive.zip"),
        "source_manifest": price.fact(price.SOURCE / "prepare_v2/RESULT.json"),
        "seed": 20260910,
        "score_claim": False,
    }
    if binding["field"]["sha256"] != price.FIELD_SHA or binding["archive"]["sha256"] != price.ARCHIVE_SHA:
        raise ValueError("source custody mismatch")
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["effective_frontier"]["archive_sha256"] != price.ARCHIVE_SHA:
        raise ValueError("POINTER_MOVED: rebind before validation")

    def reserve(need):
        used = sum(p.stat().st_size for p in root.parent.rglob("*") if p.is_file())
        if used + need > price.CAP or shutil.disk_usage(root).free < need + 8 * 1024**3:
            raise RuntimeError("STORAGE_BLOCK: retain everything")

    def record(path, value):
        reserve(65536)
        price.io.atomic_json(path, value)

    old40 = (price.ROOT / "source/old40.bin").read_bytes()
    config = bytes([31]) + old40 + bytes([1]) * 25
    reserve(len(config))
    price.io.persist_immutable_bytes(root / "control_config.bin", config, label="retained implementation-control weights")
    selected = sorted({0, 599, *np.random.default_rng(20260910).choice(600, 32, replace=False).tolist()})
    binding["selected"] = selected
    binding["config"] = price.fact(root / "control_config.bin")
    manifest = root / "INPUTS.json"
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError("validation source changed: retain and use a new generation")
    record(manifest, binding)
    field = np.memmap(price.SOURCE / "field.u8", dtype=np.uint8, mode="r", shape=(600, H, W))
    mixer = FastContextMixer(config)
    resumed = False
    times = []
    for frame in selected:
        result_path, state_path = root / f"frame_{frame:04d}.json", root / f"state_{frame:04d}.npz"
        if result_path.exists():
            receipt = json.loads(result_path.read_text())
            if receipt["binding"] != binding or receipt["state"] != price.fact(state_path) or not receipt["passed"]:
                raise ValueError("completed control custody changed")
            with np.load(state_path, allow_pickle=False) as saved:
                mixer.restore({key: saved[key] for key in saved.files})
            times.append(receipt["seconds"])
            resumed = True
            continue
        source = price.arrays(price.SOURCE / "trace/frames" / f"frame_{frame:04d}.npz", True)
        baseline = price.arrays(price.ROOT / "control/frames" / f"frame_{frame:04d}.npz")
        np.testing.assert_array_equal(source["tokens"], field[frame])
        # TC3 preparation's old counts do not depend on its five extra weights.
        # Restore the exact pre-frame state and explicitly rebind counted old40.
        mixer.old = maps.LaneMixer(bytes([1]) + old40)
        if frame:
            prior = price.arrays(price.SOURCE / "prepare_v2/states" / f"state_{frame:04d}.npz")
            old_state = {key[3:]: value for key, value in prior.items() if key.startswith("m0_")}
            old_state["config"] = np.frombuffer(bytes([1]) + old40, dtype=np.uint8).copy()
            mixer.old.restore(old_state)
        plane = field[frame]
        previous = None if frame == 0 else field[frame - 1]
        offline = maps.context_maps(plane, maps.previous_map(previous), baseline["lane_bins"], complete=True)
        mixer.begin_frame()
        phi, freq = mixer.features_from_codes(baseline["rows"], offline)
        expected = maps.mix_rows(freq, phi, mixer.weights, baseline["rows"])
        prefix = np.full((H, W), K, dtype=np.uint8)
        started = time.monotonic()
        for positions in price.POSITIONS:
            actual = mixer.coding(source["raw_rows"][positions], positions, prefix, previous)
            np.testing.assert_array_equal(mixer.codes[positions], offline[positions])
            np.testing.assert_array_equal(mixer.freq[positions], freq[positions])
            np.testing.assert_array_equal(actual, expected[positions])
            symbols = plane.reshape(-1)[positions]
            mixer.observe(positions, symbols)
            prefix.reshape(-1)[positions] = symbols
        mixer.end_frame(plane, previous)
        elapsed = time.monotonic() - started
        state = mixer.snapshot()
        fresh = FastContextMixer(config)
        fresh.restore(state)
        for key, value in state.items():
            np.testing.assert_array_equal(fresh.snapshot()[key], value)
        reserve(sum(v.nbytes for v in state.values()))
        temp = state_path.with_suffix(".new")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **state)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(state_path)
        receipt = {
            "binding": binding,
            "frame": frame,
            "passed": True,
            "groups": 190,
            "maps": 5,
            "state": price.fact(state_path),
            "seconds": elapsed,
            "plane_sha256": hashlib.sha256(plane.tobytes()).hexdigest(),
        }
        record(result_path, receipt)
        times.append(elapsed)
        print(json.dumps({"frame": frame, "seconds": elapsed, "passed": True}), flush=True)
    result = {"binding": binding, "passed": True, "frames": len(selected), "seconds": times, "resumed": resumed,
              "scope": "API map/probability/state identity on real sampled frames; no savings, score, or n600 runtime claim"}
    record(root / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    expected_root = Path("/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/fast_controls")
    if args.resume_from.resolve() != expected_root:
        raise ValueError("wrong control resume root")
    print(json.dumps(validate()), flush=True)
