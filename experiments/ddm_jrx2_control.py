#!/usr/bin/env python3
"""Repaired jrx2 row-1 instrument; no renderer/joint row is implemented here.

Replay identical pd4 sheet fields for calibration, preserve the random K24
denominator separately, and reuse jrx1's real encoder and winner selection.
Every new payload belongs to ddm_jrx2; predecessor stores are read-only.
"""

from __future__ import annotations

import argparse
import contextlib
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
import ddm_jrx1_field_control as control
import ddm_jrx1_price_control as price
import ddm_pd4_pose_directed_pass4 as pd4
import numpy as np

STORE = Path("/Volumes/APDataStore/pact/ddm_jrx2")
PREDECESSOR = Path("/Volumes/APDataStore/pact/ddm_jrx1/search_v2")
REFERENCE = REPO / ".omx/research/ddm_jrx1_20260916/PD4_REFERENCE_AUDIT.json"
AXIS = "[macOS-CPU advisory, frozen CPU-torch SegNet/PoseNet, DALI GT lineage]"
load = control.load_checked
save = control.save_json
fact = control.custody.fact


def retained_bytes(root: Path) -> int:
    """Skip transient pending files and ExFAT stubs; retry atomic rename races."""
    while True:
        try:
            total = 0
            for path in root.rglob("*"):
                if path.name.startswith("._") or path.name.endswith((".pending", ".new")):
                    continue
                info = path.stat()
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
            return total
        except FileNotFoundError:
            continue


def storage() -> dict:
    return {
        "retained_bytes": retained_bytes(STORE),
        "free_bytes": shutil.disk_usage(STORE).free,
        "limit_bytes": control.custody.LIMIT,
        "reserve_bytes": control.custody.RESERVE,
    }


def configure() -> None:
    control.STORE = control.custody.STORE = price.STORE = STORE
    control.SEARCH = PREDECESSOR
    control.stable_storage = control.custody.storage = storage
    price.ROOT = STORE / "prices"
    price.job_bytes = retained_bytes


def calibration_rows() -> list[dict]:
    reference = load(REFERENCE)
    price.checked_fact(reference["source"])
    source = [json.loads(x) for x in Path(reference["source"]["path"]).read_text().splitlines()]
    rows = []
    for wanted in reference["rows"]:
        found = [r for r in source if r["pair"] == wanted["pair"] and r["edits"] == wanted["edits"]]
        if len(found) != 1:
            raise ValueError("calibration proposal identity is not unique")
        rows.append(found[0])
    if len(rows) != 12:
        raise ValueError("calibration denominator changed")
    return rows


def bind() -> dict:
    configure()
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError("charter requires child niceness zero")
    original = load(PREDECESSOR / "BINDING.json")
    for value in original.values():
        if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
            price.checked_fact(value)
    current = load(REPO / ".omx/state/canonical_frontier_pointer.json")
    if current["our_local_frontier_contest_cuda"]["archive_sha256"] != control.custody.ARCHIVE_SHA:
        raise ValueError("POINTER_MOVED")
    control.custody.check_capacity(64 << 20)
    print(json.dumps({"storage_before_heavy": storage()}), flush=True)
    binding = {
        "predecessor_binding": fact(PREDECESSOR / "BINDING.json"),
        "sources": [fact(Path(p)) for p in (__file__, control.__file__, price.__file__)],
        "reference": fact(REFERENCE),
        "seed": control.SEED,
        "axis": AXIS,
        "score_claim": False,
    }
    save(STORE / "BINDING.json", binding)
    return binding


@contextlib.contextmanager
def instrument(binding: dict):
    """Retain real renders and immutable completed jg5 solves, as in jrx1."""
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pp1_pose_actuation as pp1
    import torch

    random.seed(control.SEED)
    np.random.seed(control.SEED)
    torch.manual_seed(control.SEED)
    torch.use_deterministic_algorithms(True)
    pd4.bind_move52(raw=control.PD4 / "parseback/0.raw", verify_raw=True)
    pp1.set_threads(2)
    originals = fe1.render_pair, jg5.refine_pair, pp1.load_body, pp1.build_pose_instrument
    cache = {}
    binding_sha = control.digest(json.dumps(binding, sort_keys=True).encode())

    def render(body, pair):
        frame = originals[0](body, pair)
        raw = frame.tobytes()
        sha = control.digest(raw)
        destination = STORE / "frames" / (sha + ".zlib")
        if not destination.exists():
            control.locked_retain(destination, zlib.compress(raw, 6))
        if control.digest(zlib.decompress(destination.read_bytes())) != sha:
            raise ValueError("render retention mismatch")
        save(
            destination.with_suffix(".json"),
            {
                "raw_sha256": sha,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "raw_bytes": len(raw),
                "payload": fact(destination),
            },
        )
        return frame

    def refine(inst, pair, start_codes, **kwargs):
        key = {
            "pair": pair,
            "frame_sha": control.digest(np.asarray(inst.raw[2 * pair + 1]).tobytes()),
            "start_codes": np.asarray(start_codes).astype(int).tolist(),
            "solver": kwargs,
            "binding_sha": binding_sha,
        }
        name = control.digest(json.dumps(key, sort_keys=True).encode())
        path = STORE / "solves" / (name + ".json")
        checksum = path.with_suffix(".sha.json")
        if path.exists() and checksum.exists():
            price.checked_fact(load(checksum))
            receipt = load(path)
            if receipt["inputs"] != key:
                raise ValueError("solve identity mismatch")
            return receipt["result"]
        save(path.with_suffix(".inputs.json"), key)
        result = originals[1](inst, pair, start_codes, **kwargs)
        save(path, {"inputs": key, "result": result})
        save(checksum, fact(path))
        return result

    def body(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = originals[2](**kwargs)
        return cache[key]

    def pose(raw):
        if "pose" not in cache:
            cache["pose"] = originals[3](raw)
        return cache["pose"]

    fe1.render_pair, jg5.refine_pair = render, refine
    pp1.load_body, pp1.build_pose_instrument = body, pose
    try:
        yield
    finally:
        fe1.render_pair, jg5.refine_pair, pp1.load_body, pp1.build_pose_instrument = originals


def measure(pairs: list[int], binding: dict) -> None:
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pp1_pose_actuation as pp1

    references = {r["pair"]: r for r in calibration_rows()}
    if not pairs or any(p not in references and p not in (502, 547) for p in pairs):
        raise ValueError("measurement scope outside calibration and two named nulls")
    with instrument(binding):
        body = pp1.load_body(with_raw=True, with_segnet=True)
        inst = pp1.build_pose_instrument(body.raw)
        codes = np.asarray(inst.state.codes, dtype=np.int32)
        base = np.load(control.PD4 / "base/pose_base_move52.npy")
        mean = float(base.mean())
        for p in pairs:
            root = STORE / "measure" / f"pair_{p:03d}"
            done = root / "DONE.json"
            if done.exists():
                for entry in load(done)["artifacts"]:
                    price.checked_fact(entry)
                continue
            print(json.dumps({"pair": p, "storage_before_heavy": storage()}), flush=True)
            control.custody.check_capacity(96 << 20)
            if p in (502, 547):
                anchor = (
                    {"pair": p, "cell": [129, 273], "old": 2, "new": 1}
                    if p == 502
                    else {"pair": p, "cell": [343, 231], "old": 4, "new": 3}
                )
                anchors = root / "ANCHOR.json"
                save(anchors, {str(p): anchor})
                attempt = root / f"attempt_{len(list(root.glob('attempt_*'))):03d}"
                args = pd4.build_parser().parse_args(
                    [
                        "cluster-search",
                        "--pairs",
                        str(p),
                        "--anchors",
                        str(anchors),
                        "--base-pose",
                        str(control.PD4 / "base/pose_base_move52.npy"),
                        "--base-band-abs",
                        "2.2642595483754955e-08",
                        "--raw",
                        str(control.PD4 / "parseback/0.raw"),
                        "--deltas=-1,1",
                        "--interior-radius",
                        "3",
                        "--neighbour-radius",
                        "1",
                        "--max-cells",
                        "2",
                        "--refine",
                        "12",
                        "--outer-rounds",
                        "40",
                        "--max-gn-iterations",
                        "400",
                        "--threads",
                        "2",
                        "--out-dir",
                        str(attempt),
                    ]
                )
                pd4.cmd_cluster_search(args)
                artifacts = [fact(x) for x in attempt.rglob("*") if x.is_file() and not x.name.startswith("._")]
                save(done, {"pair": p, "artifacts": artifacts, "completed_attempt": str(attempt)})
                continue
            reference = references[p]
            fe1.restore_pair_codes(body, p)
            before = fe1.flips_pair(fe1.argmax_pair(body, p), body, p)
            dp0 = float(br1.evaluate_codes(inst, p, codes[p][None])[0])
            if abs(dp0 - float(base[p])) > 2.2642595483754955e-08:
                raise ValueError("base pose outside retained batch band")
            for rr, cc, old, new in pd4.row_edits(reference):
                if int(body.tokens[p][rr, cc]) != old:
                    raise ValueError("calibration source symbol changed")
                body.tokens[p][rr, cc] = new
            frame = fe1.render_pair(body, p)
            after = fe1.flips_pair(jg1.argmax_from_camera_frames(body.net, frame)[0], body, p)
            fe1.restore_pair_codes(body, p)
            overlay = pp1.MemoryOverlayRaw(body.raw, {2 * p + 1: frame[0]})
            moved = br1.Instrument(inst.state, overlay, inst.targets, inst.posenet, inst.blow, inst.gram, inst.bmat)
            resolved = jg5.refine_pair(
                moved,
                p,
                codes[p],
                dd_threshold=jg5.materiality_dd_threshold(mean),
                outer_rounds=40,
                max_gn_iterations=400,
            )
            delta = float(resolved["final_d_pose"]) - dp0
            row = {
                "pair": p,
                "edits": reference["edits"],
                "d_cells": after - before,
                "credit_d_pose": delta,
                "d_pose_base": dp0,
                "d_pose_resolved": resolved["final_d_pose"],
                "carrier_codes": resolved["codes"],
                "benefit_S": -delta * pd4.pose_s_per_pair_unit(mean) - (after - before) * pd4.seg_s_per_cell(),
                "numerator": "pd4 matched first-order pose plus seg; not a finite composition score",
                "axis": AXIS,
                "score_claim": False,
            }
            entry = save(root / "ROW.json", row)
            save(done, {"pair": p, "artifacts": [entry]})
            print(json.dumps(row), flush=True)


def prepare_calibration() -> None:
    refs = calibration_rows()
    sheets = load(control.PD4 / "sheets/SHEETS.json")
    plans = []
    for sheet in sorted({r["sheet"] for r in refs}):
        rows = [r for r in sheets["manifest"] if r["sheet"] == sheet]
        name = f"calibration{sheet}"
        price.prepare_job(name, rows)
        with (
            np.load(control.PD4 / f"sheets/sheet_{sheet:02d}.npz", allow_pickle=False) as original,
            np.load(price.ROOT / name / "field.npz", allow_pickle=False) as replay,
        ):
            if any(not np.array_equal(original[str(p)], replay[str(p)]) for p in range(600)):
                raise ValueError("calibration field is not identical to the retained pd4 sheet")
        plans.append({"name": name, "sheet": sheet, "pairs": [r["pair"] for r in refs if r["sheet"] == sheet]})
    save(
        STORE / "CALIBRATION_PLAN.json",
        {
            "plans": plans,
            "rows": refs,
            "denominator": "original complete pd4 sheet per_frame_bits minus move52 control",
            "k": 12,
            "n": 12,
        },
    )


def discovery() -> None:
    sample, pool = price.read_proposals()
    mean = float(np.load(control.PD4 / "base/pose_base_move52.npy").mean())
    for p in (502, 547):
        done = load(STORE / "measure" / f"pair_{p:03d}/DONE.json")
        for entry in done["artifacts"]:
            price.checked_fact(entry)
        path = Path(done["completed_attempt"]) / "cluster_0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["benefit_S"] = (
                -row["credit_d_pose"] * pd4.pose_s_per_pair_unit(mean) - row["d_cells"] * pd4.seg_s_per_cell()
            )
        pool[p] = sorted(rows, key=lambda r: -r["benefit_S"])[:6]
    finite = [p for p in sample if pool[p]]
    save(
        STORE / "DISCOVERY_DENOMINATOR.json",
        {
            "pairs": sample,
            "finite": finite,
            "null_pairs": [p for p in sample if not pool[p]],
            "denominator": 24,
            "k": len(finite),
            "policy": "nulls have no changed token and no price; no replacement draw",
        },
    )
    price.read_proposals = lambda: (finite, pool)
    price.sheets()


def gate() -> None:
    if fact(price.TAIL_CONTROL)["sha256"] != price.TAIL_CONTROL_SHA:
        raise ValueError("base price receipt drift")
    base = load(price.TAIL_CONTROL)
    rows = []
    for expected in calibration_rows():
        p = expected["pair"]
        done = load(price.ROOT / f"calibration{expected['sheet']}/DONE.json")
        price.checked_fact(done["encode"])
        encoded = load(Path(done["encode"]["path"]))
        measure_root = STORE / "measure" / f"pair_{p:03d}"
        for entry in load(measure_root / "DONE.json")["artifacts"]:
            price.checked_fact(entry)
        observed = load(measure_root / "ROW.json")
        bits = encoded["per_frame_bits"][p] - base["per_frame_bits"][p]
        price_error = abs(bits / expected["real_delta_bits"] - 1)
        credit_error = abs(observed["benefit_S"] / expected["benefit_S"] - 1)
        rows.append(
            {
                "pair": p,
                "reference_bits": expected["real_delta_bits"],
                "bits": bits,
                "reference_credit": expected["benefit_S"],
                "credit": observed["benefit_S"],
                "price_relative_error": price_error,
                "credit_relative_error": credit_error,
                "passed": price_error <= 0.10 and credit_error <= 0.15,
            }
        )
    k = sum(row["passed"] for row in rows)
    save(
        STORE / "ROW1_GATE.json",
        {
            "rows": rows,
            "k": k,
            "n": len(rows),
            "passed": k / len(rows) >= 0.8,
            "axis": AXIS,
            "score_claim": False,
            "failures": [r["pair"] for r in rows if not r["passed"]],
        },
    )
    print(json.dumps({"k": k, "n": len(rows), "passed": k / len(rows) >= 0.8}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-calibration", "measure", "discovery", "encode", "winners", "gate"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE.resolve():
        raise ValueError("wrong resume store")
    binding = bind()
    {
        "prepare-calibration": prepare_calibration,
        "measure": lambda: measure([int(p) for p in args.pairs.split(",") if p], binding),
        "discovery": discovery,
        "encode": lambda: price.encode(args.name),
        "winners": price.winners,
        "gate": gate,
    }[args.stage]()


if __name__ == "__main__":
    main()
