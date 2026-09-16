#!/usr/bin/env python3
"""Exhaustive pair-keyed int4 RGB head-bias search on retained move52 bytes.

Research only. All candidate camera bytes are retained losslessly as a Cartesian
channel bank, not discarded after measuring a length or an objective. The frozen
CPU scorer is used without reducing the 4096-vector lattice.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO), str(REPO / "src")]
STORE = Path("/Volumes/APDataStore/pact/ddm_psa2")
PSA1 = Path("/Volumes/APDataStore/pact/ddm_psa1")
RAW = Path("/Volumes/APDataStore/pact/ddm_pd4/parseback/0.raw")
PIN = "ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e"
AXIS = "[macOS-CPU advisory, frozen CPU SegNet/PoseNet, DALI GT lineage]"
SEED = 20260916
GRID = sorted(itertools.product(range(-8, 8), repeat=3), key=lambda q: (sum(abs(x) for x in q), q))


def fact(path: Path) -> dict:
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha}


def load(path: Path):
    return json.loads(path.read_text())


def capacity(additional: int = 0) -> dict:
    total = 0
    for path in STORE.rglob("*"):
        if any(p.startswith("._") or p.endswith(".pending") for p in path.parts):
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    result = {"retained_bytes": total, "free_bytes": shutil.disk_usage(STORE.parent).free,
                  "limit_bytes": 3 << 30, "reserve_bytes": 8 << 30}
    if total + additional + (2 << 20) > result["limit_bytes"]:
        raise ValueError("RETENTION_LIMIT: preserve all bytes and block")
    if result["free_bytes"] < result["reserve_bytes"] + additional + (2 << 20):
        raise ValueError("STORAGE_RESERVE: preserve all bytes and block")
    return result


def retain(path: Path, blob: bytes) -> dict:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise ValueError("write outside owned store")
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError(f"immutable checkpoint conflict: {path}")
        return fact(path)
    capacity(len(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists() and pending.read_bytes() != blob:
        raise ValueError(f"interrupted payload needs custody: {pending}")
    with pending.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    return fact(path)


def save(path: Path, value) -> dict:
    return retain(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def array(path: Path, **values) -> dict:
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return retain(path, buffer.getvalue())


def checked(expected: dict) -> None:
    if fact(Path(expected["path"])) != expected:
        raise ValueError(f"source or checkpoint drift: {expected['path']}")


def section():
    import ddm_jrx2_renderer as renderer

    renderer.c.STORE = STORE
    return renderer.section()


def prepare() -> None:
    import ddm_pd4_pose_directed_pass4 as pd4

    print(json.dumps({"storage_before_heavy": capacity(16 << 20)}), flush=True)
    if (STORE / "BINDING.json").exists():
        prior = load(STORE / "BINDING.json")
        checked(prior["raw"])
        for row in prior["runtime"]:
            checked(row["source"])
            checked(row["copy"])
        print("resumed verified completed prepare", flush=True)
        return
    source = PSA1 / "price_runtime"
    runtime = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
            runtime.append({"source": fact(path), "copy": retain(STORE / "price_runtime" / path.relative_to(source), path.read_bytes())})
    if fact(STORE / "price_runtime/archive.zip")["sha256"] != PIN:
        raise ValueError("not move52 archive")
    receipts = pd4.bind_move52(verify_raw=True, raw=RAW)
    prior = load(PSA1 / "SAMPLE.json")
    pools, sources = set(), []
    for arm in ("ddm_pd4", "ddm_pd5"):
        path = STORE.parent / arm / "sheets/priced_rows.jsonl"
        sources.append(fact(path))
        pools.update(int(json.loads(line)["pair"]) for line in path.read_text().splitlines())
    inherited = list(prior["pairs"])
    extra = sorted(pools - set(inherited), key=lambda p: (-prior["population_counts"][p], p))[:24]
    pairs = inherited + extra
    if len(inherited) != 36 or len(pairs) != 60 or len(set(pairs)) != 60:
        raise ValueError("charter roster not 36 plus24 distinct pairs")
    save(STORE / "SAMPLE.json", {"pairs": pairs, "inherited36": inherited, "heavy24": extra,
         "counts": prior["population_counts"], "source": fact(PSA1 / "SAMPLE.json"), "pool_sources": sources,
         "selection": "psa1 exact36 plus24 heaviest residual pairs in pd4/pd5 priced rows, excluding36"})
    save(STORE / "BINDING.json", {"runtime": runtime, "receipts": receipts, "raw": fact(RAW),
         "producer": fact(Path(__file__)), "axis": AXIS, "score_claim": False, "seed": SEED})


def instrument(with_pose: bool = False):
    import ddm_pd4_pose_directed_pass4 as pd4
    import ddm_pp1_pose_actuation as pp1

    for row in load(STORE / "BINDING.json")["runtime"]:
        checked(row["copy"])
    pd4.bind_move52(raw=RAW)
    # Load only the copied runtime. Parent module imports are path-sensitive.
    pp1.POINTER_TREE = STORE / "price_runtime"
    pp1.POINTER_ARCHIVE = pp1.POINTER_TREE / "archive.zip"
    body = pp1.load_body(with_raw=True, with_segnet=True)
    return body, pp1.build_pose_instrument(body.raw) if with_pose else None


def bank(body, pair: int, scales):
    """Lossless Cartesian factorization of every lattice candidate's camera bytes."""
    import ddm_fe1_frame_embedding_search as fe1
    import numpy as np
    import torch
    import torch.nn.functional as functional

    root = STORE / "pairs" / f"pair_{pair:03d}"
    done = root / "BANK.json"
    if done.exists():
        receipt = load(done)
        for item in receipt["artifacts"]:
            checked(item)
        return np.load(root / "camera_bank.npz")["camera"]
    print(json.dumps({"pair": pair, "stage": "bank", "storage_before_heavy": capacity(64 << 20)}), flush=True)
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = body.semantic.head.register_forward_hook(capture)
    try:
        control = fe1.render_pair(body, pair)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("head invoked unexpected number of times")
    head = captured[0]
    artifacts = [array(root / "head.npz", head=head.numpy(), scales=scales)]
    artifacts.append(array(root / "control.npz", camera=control))
    if not np.array_equal(control[0], body.raw[2 * pair + 1]):
        raise ValueError("literal renderer does not reproduce pinned raw")
    camera = np.empty((16, 874, 1164, 3), dtype=np.uint8)
    with torch.inference_mode():
        for i, code in enumerate(range(-8, 8)):
            offset = torch.from_numpy((scales * code).astype(np.float32)).view(1, 3, 1, 1)
            native = torch.sigmoid(head + offset) * 255.0
            rendered = functional.interpolate(native, size=(874, 1164), mode="bilinear", align_corners=False).clamp(0, 255).round()
            camera[i] = rendered.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
    artifacts.append(array(root / "camera_bank.npz", camera=camera))
    if not np.array_equal(camera[8], control[0]):
        raise ValueError("zero bias bank differs from literal control")
    controls = []
    for code in ((-8, 7, 1), (2, -3, 0)):
        offset = torch.from_numpy((scales * np.asarray(code)).astype(np.float32)).view(1, 3, 1, 1)

        def add_bias(_module, _inputs, output, offset=offset):
            return output + offset

        handle = body.semantic.head.register_forward_hook(add_bias)
        try:
            literal = fe1.render_pair(body, pair)[0]
        finally:
            handle.remove()
        composed = np.stack([camera[q + 8, :, :, c] for c, q in enumerate(code)], axis=-1)
        # Literal payload is exactly represented by the persisted bank if this holds.
        if not np.array_equal(literal, composed):
            artifacts.append(array(root / f"factorization_failure_{len(controls)}.npz", literal=literal))
            raise ValueError("channel factorization differs from real hooked render")
        controls.append({"codes": code, "raw_sha256": hashlib.sha256(literal.tobytes()).hexdigest()})
    save(done, {"pair": pair, "artifacts": artifacts, "zero_identity": True, "literal_controls": controls,
         "all_candidate_payloads_retained": "camera_bank.npz[channel_code+8,:,:,channel] concatenated in RGB order",
         "vectors": 4096, "axis": AXIS})
    return camera


def control() -> None:
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_jg1_seg_solve as jg1
    import ddm_up2_shipping_pose_solve as up2
    import numpy as np
    import torch

    _container, runs, *_ = section()
    scales = np.asarray(runs["head.weight"]["scales"], dtype=np.float32)
    body, inst = instrument(with_pose=True)
    roster = load(STORE / "SAMPLE.json")
    base_path = STORE.parent / "ddm_pd4/base/pose_base_move52.npy"
    base = np.load(base_path)
    rows = []
    for pair in roster["pairs"]:
        dest = STORE / "pairs" / f"pair_{pair:03d}"
        prior = dest / "CONTROL.json"
        if prior.exists():
            row = load(prior)
            for artifact in row["artifacts"]:
                checked(artifact)
            rows.append(row)
            continue
        camera = bank(body, pair, scales)
        labels = jg1.argmax_from_camera_frames(body.net, camera[8][None])[0]
        label_fact = array(dest / "base_argmax.npz", argmax=labels)
        if int(np.count_nonzero(labels != body.gt[pair])) != roster["counts"][pair]:
            raise ValueError("zero argmax disagrees with move52")
        index = np.asarray([pair], dtype=np.int64)
        codes = np.asarray(inst.state.codes, dtype=np.int32)[index]
        with torch.inference_mode():
            coeff = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
            frame0 = up2.render_frame0(coeff, inst.state, index, differentiable=False)
            camera0 = frame0.to(torch.uint8).permute(0, 2, 3, 1).numpy()
        pose_frame = array(dest / "base_frame0.npz", frame0=camera0)
        if not np.array_equal(camera0[0], body.raw[2 * pair]):
            raise ValueError("repaired frame0 forward does not reproduce shipped raw")
        value = float(br1.evaluate_codes(inst, pair, codes)[0])
        error = abs(value - float(base[pair]))
        row = {"pair": pair, "d_pose": value, "baseline": float(base[pair]), "abs_error": error,
                   "band_abs": 2.2642595483754955e-8, "passed": error <= 2.2642595483754955e-8,
                   "seg_cells": roster["counts"][pair], "artifacts": [label_fact, pose_frame]}
        save(prior, row)
        if not row["passed"]:
            raise ValueError("base pose outside measured tolerance gate")
        rows.append(row)
        print(json.dumps({"control_pair": pair, "pose_abs_error": error, "storage": capacity()}), flush=True)
    if len(rows) != 60 or not all(row["passed"] for row in rows):
        raise ValueError("incomplete control")
    save(STORE / "CONTROL.json", {"k": 60, "n": 60, "rows": rows, "scales": scales.tolist(),
         "source_pose": fact(base_path), "base_mean": float(base.mean()), "axis": AXIS, "score_claim": False})


def search(pairs: list[int], stop_after_blocks: int | None) -> None:
    import ddm_jg1_seg_solve as jg1
    import numpy as np

    gate = load(STORE / "CONTROL.json")
    if gate["k"] != 60 or gate["n"] != 60 or not all(r["passed"] for r in gate["rows"]):
        raise ValueError("60-pair base control not passed")
    _container, runs, *_ = section()
    scales = np.asarray(runs["head.weight"]["scales"], dtype=np.float32)
    body, _ = instrument()
    roster = load(STORE / "SAMPLE.json")
    if not pairs or any(p not in roster["pairs"] for p in pairs):
        raise ValueError("pairs outside the fixed60")
    completed = 0
    for pair in pairs:
        root = STORE / "pairs" / f"pair_{pair:03d}"
        if (root / "SEARCH.json").exists():
            continue
        camera = bank(body, pair, scales)
        all_rows = []
        for start in range(0, 4096, 16):
            block = root / f"grid_{start:04d}.json"
            if block.exists():
                prior = load(block)
                checked(prior["bank"])
                if [tuple(r["codes"]) for r in prior["rows"]] != GRID[start:start + 16]:
                    raise ValueError("completed block lattice order differs")
                all_rows.extend(prior["rows"])
                continue
            print(json.dumps({"pair": pair, "grid_start": start, "storage_before_heavy": capacity(1 << 20)}), flush=True)
            started = time.monotonic()
            rows = []
            for index in range(start, start + 16):
                code = GRID[index]
                frame = np.stack([camera[q + 8, :, :, c] for c, q in enumerate(code)], axis=-1)
                labels = jg1.argmax_from_camera_frames(body.net, frame[None])[0]
                cells = int(np.count_nonzero(labels != body.gt[pair]))
                rows.append({"index": index, "codes": code, "cells": cells,
                                 "camera_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            save(block, {"rows": rows, "wall_seconds": time.monotonic() - started,
                            "bank": fact(root / "camera_bank.npz"), "producer": fact(Path(__file__))})
            all_rows.extend(rows)
            completed += 1
            print(json.dumps({"pair": pair, "grid_done": start + 16, "best_cells": min(r["cells"] for r in all_rows),
                              "block_wall_seconds": time.monotonic() - started}), flush=True)
            if stop_after_blocks is not None and completed >= stop_after_blocks:
                return
        if len(all_rows) != 4096 or {r["index"] for r in all_rows} != set(range(4096)):
            raise ValueError("incomplete lattice")
        best = min(all_rows, key=lambda r: (r["cells"], sum(abs(q) for q in r["codes"]), tuple(r["codes"])))
        zero = next(r for r in all_rows if r["codes"] == [0, 0, 0] or r["codes"] == (0, 0, 0))
        if zero["cells"] != roster["counts"][pair]:
            raise ValueError("zero argmax differs from pinned residual")
        frame = np.stack([camera[q + 8, :, :, c] for c, q in enumerate(best["codes"])], axis=-1)
        labels = jg1.argmax_from_camera_frames(body.net, frame[None])[0]
        selected = array(root / "selected.npz", frame1=frame, argmax=labels, codes=np.asarray(best["codes"], dtype=np.int8))
        save(root / "SEARCH.json", {"pair": pair, "best": best, "zero": zero, "vectors": 4096,
             "credit_cells": zero["cells"] - best["cells"], "scales": scales.tolist(), "selected": selected,
             "axis": AXIS, "score_claim": False, "pose_resolved": False})


def main() -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "control", "search"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--pairs")
    parser.add_argument("--stop-after-blocks", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE.resolve():
        raise ValueError("wrong resume store")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "control":
        control()
    else:
        pairs = [int(p) for p in args.pairs.split(",")] if args.pairs else load(STORE / "SAMPLE.json")["pairs"]
        search(pairs, args.stop_after_blocks)


if __name__ == "__main__":
    main()
