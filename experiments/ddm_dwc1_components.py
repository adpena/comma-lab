"""Replay shipped causal geometry/map methods on retained real decoded fields.

This measures map work only: no probability rows are invented and no entropy
decode, calibration, renderer, or contest score is claimed. Run after the public
timings, never alongside them. Every sampled frame retains emitted maps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "4"
sys.dont_write_bytecode = True

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock")
SOURCES = {
    "tc3": Path("/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/candidate_runtime"),
    "tc4": Path("/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41/candidate_runtime"),
}
IDENTITY = SOURCES["tc3"].parent / "PUBLIC_IDENTITY.json"
MODULES = ("tc1_shared_mixer.py", "tc3_geometry.py", "tc3_mixer.py",
           "tc4_maps.py", "tc4_fast.py")
SHAPE = (600, 384, 512)
COMPONENT_BUDGET = 1024**3
TOTAL_BUDGET = 8 * 1024**3


def fact(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def occupied(root):
    """Count each retained hardlinked payload once by filesystem and inode."""
    seen, total = set(), 0
    for path in root.rglob("*"):
        if path.is_file():
            info = path.stat()
            key = (info.st_dev, info.st_ino)
            if key not in seen:
                seen.add(key)
                total += info.st_size
    return total


def reserve(root, additional):
    """Reserve worst-case next output; bytes are retained on every failure."""
    component_used = occupied(root)
    total_used = occupied(ROOT)
    facts = {"component_bytes": component_used, "total_bytes": total_used,
             "reserved_next_bytes": additional, "component_budget": COMPONENT_BUDGET,
             "total_budget": TOTAL_BUDGET}
    if (component_used + additional > COMPONENT_BUDGET
            or total_used + additional > TOTAL_BUDGET
            or shutil.disk_usage(ROOT).free < 40 * 1024**3 + additional):
        save(root / "STORAGE_BLOCKED.json", facts)
        raise RuntimeError("storage reservation blocked; every existing payload retained")
    return facts


def snapshot():
    try:
        result = subprocess.run(["ps", "-axo", "pid,ppid,%cpu,command"],
                                capture_output=True, text=True, timeout=5)
        rows = result.stdout.splitlines()
        return {"load_average": list(os.getloadavg()), "quiesced": False,
                "process_inventory_returncode": result.returncode,
                "process_inventory_error": result.stderr, "process_inventory": rows,
                "concurrent_process_count": max(0, len(rows) - 1) if result.returncode == 0 else None,
                "normalization": None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"load_average": list(os.getloadavg()), "quiesced": False,
                "concurrent_process_count": None, "error": repr(exc), "normalization": None}


def frame_ids(sample, seed):
    if not 32 <= sample <= 100:
        raise ValueError("sample must be 32..100")
    rng = np.random.default_rng(seed)
    return [int(rng.choice(chunk)) for chunk in np.array_split(np.arange(600), sample)]


def load_modules(role, root):
    """Import byte-identical bounded copies under a role-specific package name."""
    destination = root / "sources" / role
    destination.mkdir(parents=True, exist_ok=True)
    names = MODULES[:3] if role == "tc3" else MODULES
    sources = {}
    for name in names:
        source = SOURCES[role] / "runtime" / name
        target = destination / name
        source_fact = fact(source)
        if not target.exists():
            shutil.copyfile(source, target)
        target_fact = fact(target)
        if target_fact["sha256"] != source_fact["sha256"]:
            raise RuntimeError("copied source changed; refusing drift")
        sources[name] = {"source": source_fact, "copy": target_fact}
    package_name = "dwc1_" + role
    package = types.ModuleType(package_name)
    package.__path__ = [str(destination)]
    sys.modules[package_name] = package
    geometry = importlib.import_module(package_name + ".tc3_geometry")
    mixer = importlib.import_module(package_name + ".tc3_mixer")
    fast = None if role == "tc3" else importlib.import_module(package_name + ".tc4_fast")
    return geometry, mixer, fast, sources


def replay_frame(role, modules, truth, previous):
    """Invoke actual methods on exact causal map state, excluding probabilities.

    For tc4 we materialize only probability-independent state from begin_frame
    and coding. old.pending/old.bins are the coding method's group handoff. The
    real old LaneMixer.observe consumes that handoff and updates LaneGeometry.
    No synthetic probabilities, weights, or learned counts enter this replay.
    """
    geometry, mixer_module, fast_module = modules
    counters = {"lane_init_seconds": 0.0, "lane_contexts_seconds": 0.0,
                "lane_observe_seconds": 0.0, "tc4_previous_map_seconds": 0.0,
                "tc4_context_codes_seconds": 0.0, "tc4_observe_inclusive_seconds": 0.0,
                "tc4_old_observe_seconds": 0.0, "tc4_state_setup_seconds": 0.0}
    started = time.perf_counter()
    lane = geometry.LaneGeometry(previous)
    counters["lane_init_seconds"] = time.perf_counter() - started
    bins_all = np.empty((384, 512), dtype=np.uint8)
    codes_all = np.empty((384, 512, 5), dtype=np.uint8) if role == "tc4" else None
    fast = None
    if role == "tc4":
        started = time.perf_counter()
        fast = fast_module.FastContextMixer.__new__(fast_module.FastContextMixer)
        old = mixer_module.LaneMixer.__new__(mixer_module.LaneMixer)
        old.geometry, old.pending = lane, None
        old.bins = np.empty(384 * 512, dtype=np.uint8)
        fast.old = old
        fast.row_length = np.zeros(384 * 512, dtype=np.uint8)
        fast.transitions = np.zeros((384 // 64, 512), dtype=np.uint64)
        fast.lane_bins = np.full((384, 512), 9, dtype=np.uint8)
        counters["tc4_state_setup_seconds"] = time.perf_counter() - started
        original_observe = old.observe

        def timed_old_observe(positions, symbols):
            started = time.perf_counter()
            original_observe(positions, symbols)
            counters["tc4_old_observe_seconds"] += time.perf_counter() - started

        old.observe = timed_old_observe
    visits = 0
    for positions in geometry.POSITIONS:
        symbols = truth.reshape(-1)[positions].astype(np.int64)
        started = time.perf_counter()
        bins = lane.contexts(positions)
        counters["lane_contexts_seconds"] += time.perf_counter() - started
        bins_all.reshape(-1)[positions] = bins
        if fast is None:
            started = time.perf_counter()
            lane.observe(positions, symbols)
            counters["lane_observe_seconds"] += time.perf_counter() - started
        else:
            fast.old.pending = positions.copy()
            fast.old.bins[positions] = bins
            if visits == 0:
                started = time.perf_counter()
                fast.temporal = fast_module.maps.previous_map(previous)
                counters["tc4_previous_map_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            codes = fast.context_codes(positions)
            counters["tc4_context_codes_seconds"] += time.perf_counter() - started
            codes_all.reshape(-1, 5)[positions] = codes
            fast.lane_bins.reshape(-1)[positions] = bins
            started = time.perf_counter()
            fast.observe(positions, symbols)
            counters["tc4_observe_inclusive_seconds"] += time.perf_counter() - started
        visits += len(positions)
    np.testing.assert_array_equal(lane.plane, truth)
    if visits != 384 * 512 or lane.group != len(geometry.POSITIONS) or lane.pending:
        raise RuntimeError("incomplete causal frame replay")
    counters["tc4_observe_incremental_seconds"] = (
        counters["tc4_observe_inclusive_seconds"] - counters["tc4_old_observe_seconds"])
    if role == "tc3":
        counters["tc3_predictor_seconds"] = (counters["lane_init_seconds"]
            + counters["lane_contexts_seconds"] + counters["lane_observe_seconds"])
    else:
        counters["tc4_additional_maps_seconds"] = (counters["tc4_previous_map_seconds"]
            + counters["tc4_context_codes_seconds"] + counters["tc4_observe_incremental_seconds"])
    arrays = {"lane_bins": bins_all}
    if codes_all is not None:
        arrays["tc4_codes"] = codes_all
        arrays["tc4_temporal"] = fast.temporal
    return arrays, {"times": counters, "token_visits": visits,
                    "group_calls": len(geometry.POSITIONS), "exact_final_plane": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    root = args.resume_from.resolve()
    if root != ROOT / "components":
        raise ValueError("unexpected component retention root")
    root.mkdir(parents=True, exist_ok=True)
    ids = frame_ids(args.sample, args.seed)
    reserve(root, 150 * 1024**2 + args.sample * 2 * 2 * 1024**2)
    original = json.loads(IDENTITY.read_text())["decoded_field"]
    if fact(original["path"]) != original or original["bytes"] != int(np.prod(SHAPE)):
        raise RuntimeError("retained source field differs from public identity receipt")
    source_field = Path(original["path"])
    retained_field = root / "tokens.u8"
    if not retained_field.exists():
        temporary = retained_field.with_suffix(".new")
        shutil.copyfile(source_field, temporary)
        temporary.replace(retained_field)
    if fact(retained_field)["sha256"] != original["sha256"]:
        raise RuntimeError("retained field drift")
    field = np.memmap(retained_field, mode="r", dtype=np.uint8, shape=SHAPE)
    binding = {"schema": "ddm_dwc1.components_binding.v1", "seed": args.seed,
               "frame_ids": ids, "selection": "one seeded random frame per equal temporal stratum",
               "source_identity": fact(IDENTITY), "field": fact(retained_field),
               "producer": fact(Path(__file__)), "numpy": np.__version__,
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "platform": platform.platform(), "host": platform.node(),
               "threads": {key: os.environ[key] for key in THREAD_KEYS}, "sources": {}}
    modules = {}
    for role in SOURCES:
        geometry, mixer, fast, source_facts = load_modules(role, root)
        modules[role] = (geometry, mixer, fast)
        binding["sources"][role] = source_facts
    binding_path = root / "BINDING.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError("component resume binding drift")
    save(binding_path, binding)
    binding_sha = fact(binding_path)["sha256"]
    rows = []
    for frame in ids:
        truth = np.asarray(field[frame])
        previous = None if frame == 0 else np.asarray(field[frame - 1])
        if np.any(truth >= 5) or (previous is not None and np.any(previous >= 5)):
            raise RuntimeError("real decoded field contains an invalid class")
        for role in SOURCES:
            receipt_path = root / role / f"frame_{frame:04d}.json"
            if receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                if (row["role"] != role or row["frame"] != frame
                        or row["binding_sha256"] != binding_sha
                        or row["token_visits"] != 384 * 512
                        or row["group_calls"] != 190
                        or any(not np.isfinite(value) or value < 0 for value in row["times"].values())
                        or fact(row["output"]["path"]) != row["output"]):
                    raise RuntimeError("retained component output drift")
                rows.append(row)
                continue
            reserve(root, 4 * 1024**2)
            before = snapshot()
            arrays, result = replay_frame(role, modules[role], truth, previous)
            output = receipt_path.with_suffix(".npz")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                # An interrupted receipt write never authorizes payload overwrite.
                with np.load(output, allow_pickle=False) as old:
                    if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                        raise RuntimeError("orphan retained maps differ from replay")
            else:
                temporary = output.with_suffix(".npz.new")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(output)
            row = {"role": role, "frame": frame, "binding_sha256": binding_sha,
                   "output": fact(output), **result,
                   "concurrency_before": before, "concurrency_after": snapshot()}
            save(receipt_path, row)
            rows.append(row)
            save(root / "LATEST.json", {"role": role, "frame": frame,
                 "completed_rows": len(rows), "receipt": fact(receipt_path)})
            print(json.dumps({"role": role, "frame": frame, "times": result["times"]}), flush=True)
    summary = {}
    for role in SOURCES:
        selected = [row for row in rows if row["role"] == role]
        tokens = sum(row["token_visits"] for row in selected)
        seconds = {key: sum(row["times"][key] for row in selected) for key in selected[0]["times"]}
        summary[role] = {"frames": len(selected), "token_visits": tokens,
                         "seconds": seconds,
                         "seconds_per_million_tokens": {key: value * 1e6 / tokens for key, value in seconds.items()},
                         "n600_projected_seconds": {key: value * int(np.prod(SHAPE)) / tokens for key, value in seconds.items()}}
    save(root / "RESULT.json", {"schema": "ddm_dwc1.components.v1", "binding": binding,
         "axis": "[macOS-CPU advisory]", "score_claim": False, "summary": summary,
         "scope": "exact shipped geometry and map methods, real causal decoded fields; excludes probability calibration, mix_rows, learned-count updates, entropy decode and rendering",
         "tc4_incremental_method": "outer shipped FastContextMixer.observe wall minus nested shipped LaneMixer.observe wall on the same call; wrapper overhead remains",
         "normalization": None, "quiesced": False, "retention": reserve(root, 0),
         "n600_status": "projection from the recorded stratified frame sample, not n600 timing"})


if __name__ == "__main__":
    main()
