#!/usr/bin/env python3
"""Retained, resumable pd4 field-control search on jrd1's fixed random draw.

This is only row-1 proposal generation/replay, not the joint exchange probe.
The pd4/pd1 renderer, frozen scorers and jg5 pose refinement remain unchanged.
All rendered proposal frames and solver results are retained by content hash.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import stat
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO), str(REPO / "src")]
import ddm_jrd1_byte_preflight as custody

STORE = Path("/Volumes/APDataStore/pact/ddm_jrx1")
SEARCH = STORE / "search_v2"
SAMPLE = REPO / ".omx/research/ddm_jrd1_20260916/SAMPLE.json"
SAMPLE_SHA = "b905cbabf06ca4a9dc78c1f9ed49a929e0c15d5f2b580a55fa769c23a5256613"
PD4 = Path("/Volumes/APDataStore/pact/ddm_pd4")
SEED = 20260916
custody.STORE = STORE
if not hasattr(custody, "_jrx1_original_retain"):
    custody._jrx1_original_retain = custody.retain
_serial_retain = custody._jrx1_original_retain


def stable_storage() -> dict:
    """A rename during a census retries the census; it never drops a payload."""
    while True:
        total = 0
        try:
            for path in STORE.rglob("*"):
                observed = path.stat()
                if stat.S_ISREG(observed.st_mode):
                    total += observed.st_size
            break
        except FileNotFoundError:
            # Atomic .pending -> final renames are normal, not missing custody.
            continue
    return {"retained_bytes": total, "free_bytes": shutil.disk_usage(STORE).free,
            "limit_bytes": custody.LIMIT, "reserve_bytes": custody.RESERVE}


def locked_retain(path: Path, payload: bytes) -> dict:
    """Serialize capacity-check plus payload commit across this arm's workers."""
    STORE.mkdir(parents=True, exist_ok=True)
    with (STORE / ".custody.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _serial_retain(path, payload)


custody.storage = stable_storage
custody.retain = locked_retain


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_json(path: Path, value: object) -> dict:
    return custody.retain(path, (json.dumps(value, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n").encode())


def load_checked(path: Path) -> dict:
    return json.loads(path.read_text())


def guard() -> dict:
    STORE.mkdir(parents=True, exist_ok=True)
    if custody.fact(SAMPLE)["sha256"] != SAMPLE_SHA:
        raise ValueError("fixed sample changed")
    pointer = load_checked(REPO / ".omx/state/canonical_frontier_pointer.json")
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != custody.ARCHIVE_SHA:
        raise ValueError("POINTER_MOVED: re-pin before proceeding")
    import ddm_pd4_pose_directed_pass4 as pd4
    binding = pd4.bind_move52(raw=PD4 / "parseback/0.raw", verify_raw=True)
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError("charter requires child niceness zero")
    custody.check_capacity(32 << 20)
    print(json.dumps({"storage": custody.storage(), "binding": binding}), flush=True)
    return {"sample": load_checked(SAMPLE), "scorer_binding": binding}


def search(resume_from: Path, pairs: list[int]) -> None:
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pd1_pose_directed as pd1
    import ddm_pd4_pose_directed_pass4 as pd4
    import ddm_pp1_pose_actuation as pp1
    import ddm_up2_shipping_pose_solve as up2
    import numpy as np
    import torch

    if resume_from.resolve() != STORE.resolve():
        raise ValueError("resume-from must be this arm's owned store")
    checked = guard()
    sample = checked["sample"]
    if any(pair not in sample["pairs"] for pair in pairs) or len(pairs) != len(set(pairs)):
        raise ValueError("pairs must be distinct members of the preregistered draw")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    pp1.set_threads(2)
    base = np.load(PD4 / "base/pose_base_move52.npy")
    pose_unit = pd4.pose_s_per_pair_unit(float(base.mean()))
    seg_cell = pd4.seg_s_per_cell()
    sources = [Path(m.__file__) for m in (pd1, pd4, pp1, fe1, jg5, br1, up2, jg1)] + [Path(__file__)]
    sources.extend(p for p in pd4.MOVE52_TREE.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and not p.name.startswith("._"))
    sys.path.insert(0, str(REPO / "upstream"))
    try:
        import modules
    finally:
        sys.path.pop(0)
    sources.extend([Path(modules.__file__), Path(modules.segnet_sd_path),
                    Path(modules.posenet_sd_path), jg1.DEFAULT_GT_DALI, up2.DEFAULT_DALI_GT])
    binding = {str(p): custody.fact(p) for p in sources}
    binding["scorer_binding"] = checked["scorer_binding"]
    binding["base_pose"] = custody.fact(PD4 / "base/pose_base_move52.npy")
    binding["sample"] = custody.fact(SAMPLE)
    save_json(SEARCH / "BINDING.json", binding)
    binding_sha = digest(json.dumps(binding, sort_keys=True).encode())

    original_render, original_refine = fe1.render_pair, jg5.refine_pair
    original_load, original_instrument = pp1.load_body, pp1.build_pose_instrument
    cache = {}

    def retained_render(body, pair):
        frame = original_render(body, pair)
        raw = frame.tobytes()
        sha = digest(raw)
        destination = SEARCH / "frames" / (sha + ".zlib")
        if not destination.exists():
            custody.check_capacity(len(raw))
            custody.retain(destination, zlib.compress(raw, level=6))
        else:
            if digest(zlib.decompress(destination.read_bytes())) != sha:
                raise ValueError("retained render mismatch")
        save_json(destination.with_suffix(".json"), {
            "raw_sha256": sha, "shape": list(frame.shape), "dtype": str(frame.dtype),
            "payload": custody.fact(destination), "raw_bytes": len(raw)})
        return frame

    def retained_refine(inst, pair, start_codes, **kwargs):
        # Full immutable proposal boundary: an interrupted solve repeats that one
        # solve from its pinned inputs. Completed proposals are never recomputed.
        frame = np.asarray(inst.raw[2 * pair + 1])
        key = {"pair": pair, "frame_sha": digest(frame.tobytes()),
               "start_codes": np.asarray(start_codes).astype(int).tolist(), "solver": kwargs,
               "binding_sha": binding_sha}
        name = digest(json.dumps(key, sort_keys=True).encode())
        path = SEARCH / "solves" / (name + ".json")
        checksum = path.with_suffix(".sha.json")
        if path.exists() and checksum.exists():
            if custody.fact(path) != load_checked(checksum):
                raise ValueError("pose checkpoint payload changed")
            receipt = load_checked(path)
            if receipt["inputs"] != key:
                raise ValueError("pose resume identity mismatch")
            return receipt["result"]
        save_json(path.with_suffix(".inputs.json"), key)
        result = original_refine(inst, pair, start_codes, **kwargs)
        save_json(path, {"inputs": key, "result": result})
        save_json(checksum, custody.fact(path))
        return result

    def loaded_body(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = original_load(**kwargs)
        return cache[key]

    def loaded_instrument(raw):
        if "instrument" not in cache:
            cache["instrument"] = original_instrument(raw)
        return cache["instrument"]

    fe1.render_pair, jg5.refine_pair = retained_render, retained_refine
    pp1.load_body, pp1.build_pose_instrument = loaded_body, loaded_instrument
    try:
        for pair in pairs:
            root = SEARCH / f"pair_{pair:03d}"
            if (root / "DONE.json").exists():
                done = load_checked(root / "DONE.json")
                if done["binding_sha"] != binding_sha:
                    raise ValueError("completed pair source binding changed")
                for artifact in done["artifacts"]:
                    if custody.fact(Path(artifact["path"])) != artifact:
                        raise ValueError("completed pair artifact changed")
                print(json.dumps({"pair": pair, "status": "retained_complete"}), flush=True)
                continue
            print(json.dumps({"pair": pair, "storage": custody.storage()}), flush=True)
            custody.check_capacity(96 << 20)
            # Preserve interrupted JSONL and replay into a fresh attempt, using
            # only completed immutable solver checkpoints from earlier attempts.
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt_{len(list(root.glob('attempt_*'))):03d}"
            attempt.mkdir()
            args = pd1.build_parser().parse_args([
                "search", "--pairs", str(pair), "--base-pose", str(PD4 / "base/pose_base_move52.npy"),
                "--base-band-abs", "2.2642595483754955e-08", "--cells", "16", "--deltas=-1,1",
                "--interior-radius", "3", "--max-cells", "2", "--refine", "12",
                "--bits-per-token", "12.923", "--outer-rounds", "40", "--max-gn-iterations", "400",
                "--threads", "2", "--out-dir", str(attempt / "single")])
            # Never use pd1's resume flag: it mistakes the first completed row for
            # a complete pair. Cached full proposal solves make a replay harmless.
            pd1.cmd_search(args)
            rows = [json.loads(s) for s in (attempt / "single/search_b_0.jsonl").read_text().splitlines()]
            singles = {pd4.proposal_key(row): row for row in rows}
            anchor = max(singles.values(), key=lambda r: -r["credit_d_pose"] * pose_unit
                         - r["d_cells"] * seg_cell) if singles else None
            if anchor is not None:
                anchor_path = attempt / "ANCHOR.json"
                save_json(anchor_path, {str(pair): anchor})
                args = pd4.build_parser().parse_args([
                    "cluster-search", "--pairs", str(pair), "--anchors", str(anchor_path),
                    "--base-pose", str(PD4 / "base/pose_base_move52.npy"),
                    "--base-band-abs", "2.2642595483754955e-08", "--raw", str(PD4 / "parseback/0.raw"),
                    "--deltas=-1,1", "--interior-radius", "3", "--neighbour-radius", "1",
                    "--max-cells", "2", "--refine", "12", "--outer-rounds", "40",
                    "--max-gn-iterations", "400", "--threads", "2", "--out-dir", str(attempt / "cluster")])
                pd4.cmd_cluster_search(args)
            receipts = [custody.fact(p) for p in sorted(attempt.rglob("*")) if p.is_file()
                        and not p.name.startswith("._")]
            save_json(root / "DONE.json", {"pair": pair, "single_proposals": len(singles),
                      "binding_sha": binding_sha, "completed_attempt": str(attempt),
                      "selection_extension": "fixed K keeps best single anchor even when benefit is nonpositive",
                      "cluster_searched": anchor is not None, "artifacts": receipts,
                      "status": "PROPOSALS_READY_UNPRICED" if singles else "NO_ADMISSIBLE_SINGLE_ANCHOR",
                      "axis": "[macOS-CPU advisory]", "score_claim": False})
    finally:
        fe1.render_pair, jg5.refine_pair = original_render, original_refine
        pp1.load_body, pp1.build_pose_instrument = original_load, original_instrument


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--pairs", default="", help="optional subset of fixed draw; empty means all 24")
    args = parser.parse_args()
    pairs = [int(p) for p in args.pairs.split(",")] if args.pairs else load_checked(SAMPLE)["pairs"]
    search(args.resume_from, pairs)


if __name__ == "__main__":
    main()
