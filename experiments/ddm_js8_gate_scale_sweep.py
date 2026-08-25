#!/usr/bin/env python3
"""Tune JS8's counted gate amplitude on the retained stratified n32 bank.

Selection uses only realized SegNet plus exact archive rate because QS5 frame-0
pose compensation has not yet been compiled.  Every gate, archive, receiver
render, scorer input, logit field, argmax field, and pose vector is retained.
The selected row remains a TOY-BRACKET proposal, never an n600 verdict.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js8_implicit_edge_conditioning as build
from experiments import ddm_js8_stratified_advisory_screen as screen
from experiments.ddm_ec1_runtime import js8_edge_state_conditioner as js8_runtime

OUTPUT: Final = build.BULK_ROOT / "scale_sweep_v1"
SCREEN_RESULT: Final = build.BULK_ROOT / "screen_v1/SCREEN_RESULT.json"
SCALES: Final = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5)


class JS8SweepError(RuntimeError):
    """The immutable bank, candidate payload, or selection invariant failed."""


def main() -> int:
    if OUTPUT.exists():
        raise JS8SweepError(f"immutable sweep output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source = json.loads(SCREEN_RESULT.read_text())
    if source.get("status") != "TOY_BRACKET_COMPLETE_NO_VERDICT":
        raise JS8SweepError("source screen status differs")
    pair_ids = np.load(build.BULK_ROOT / "screen_v1/retained/pair_ids.int64.npy", allow_pickle=False)
    target_frames = np.load(build.BULK_ROOT / "screen_v1/retained/target_camera.uint8.npy", mmap_mode="r")
    base_frames = np.load(build.BULK_ROOT / "screen_v1/retained/base_camera.uint8.npy", mmap_mode="r")
    target_argmax = np.load(build.BULK_ROOT / "screen_v1/retained/target_argmax.npy", mmap_mode="r")
    target_pose = np.load(build.BULK_ROOT / "screen_v1/retained/target_pose.npy", mmap_mode="r")
    if pair_ids.shape != (32,) or target_frames.shape[0] != 32 or base_frames.shape[0] != 32:
        raise JS8SweepError("retained n32 bank geometry differs")
    weights = np.load(
        build.BULK_ROOT / "build_v1/retained/conditioning/edge_weights.float32.npy",
        allow_pickle=False,
    )
    ec1_blob = build.EC2_MODULE.read_bytes()
    segnet, posenet = screen.load_scorers()  # SCORER_LOADER_ORDER_OK: screen-local wrapper (ddm_js8_stratified_advisory_screen:152) returns (segnet, posenet); unpack matches its verified order
    torch.set_num_threads(4)
    rows = []
    for scale in SCALES:
        label = f"scale_{str(scale).replace('.', 'p')}"
        root = OUTPUT / label
        gate_blob = js8_runtime.serialize_gate(weights, adapter_scale=scale)
        screen.atomic_bytes(root / "retained/js8_edge_gate.br", gate_blob)
        decoded, decoded_scale, header = js8_runtime.parse_gate(gate_blob)
        screen.atomic_npy(root / "retained/js8_edge_gate.parseback.float32.npy", decoded)
        archive = build.deterministic_archive(ec1_blob, gate_blob)
        repeat = build.deterministic_archive(ec1_blob, gate_blob)
        if archive != repeat:
            raise JS8SweepError(f"archive repeat differs at scale {scale}")
        screen.atomic_bytes(root / "retained/archive.zip", archive)
        screen.atomic_bytes(root / "retained/archive.repeat.zip", repeat)
        archive_record = screen.file_record(root / "retained/archive.zip")
        build.adapt_runtime(root, archive_record)

        pre_r, master = screen.render_active(pair_ids, ec1_blob, gate_blob)
        frames = np.asarray(base_frames).copy()
        frames[:, 1] = master
        scored = screen.score_pairs(frames, segnet, posenet)
        payload_arrays = {
            "active_pre_r.float32.npy": pre_r,
            "active_camera_uncompensated.uint8.npy": frames,
            **{f"active_{name}.npy": value for name, value in scored.items()},
        }
        payloads = {
            "gate": screen.file_record(root / "retained/js8_edge_gate.br"),
            "gate_parseback": screen.file_record(root / "retained/js8_edge_gate.parseback.float32.npy"),
            "archive": archive_record,
            "archive_repeat": screen.file_record(root / "retained/archive.repeat.zip"),
        }
        for name, value in payload_arrays.items():
            path = root / "retained" / name
            screen.atomic_npy(path, np.asarray(value))
            payloads[name] = screen.file_record(path)
        flips = int(np.count_nonzero(scored["argmax"] != target_argmax))
        d_seg = flips / int(np.prod(target_argmax.shape))
        d_pose = float(np.mean((scored["pose"] - target_pose) ** 2))
        seg_term = 100.0 * d_seg
        rate_term = 25.0 * archive_record["bytes"] / screen.RATE_DENOMINATOR
        row = {
            "label": label,
            "adapter_scale": decoded_scale,
            "gate_header": header,
            "archive_bytes": archive_record["bytes"],
            "archive_delta_bytes_vs_mc36": archive_record["bytes"] - build.BASE_ARCHIVE_BYTES,
            "flips": flips,
            "d_seg": d_seg,
            "d_pose_uncompensated": d_pose,
            "seg_term": seg_term,
            "pose_term_uncompensated": math.sqrt(10.0 * d_pose),
            "rate_term": rate_term,
            "seg_plus_rate_selection_value": seg_term + rate_term,
            "uncompensated_sample_S": seg_term + math.sqrt(10.0 * d_pose) + rate_term,
            "payloads": payloads,
            "runtime": str(root / "adapted_runtime"),
        }
        rows.append(row)
    selected = min(rows, key=lambda row: (row["seg_plus_rate_selection_value"], row["archive_bytes"]))
    base = source["rows"]["base"]
    result = {
        "schema": "ddm_js8_gate_scale_sweep.v1",
        "status": "TOY_BRACKET_TUNED_FULL_N600_OWED",
        "axis": source["axis"],
        "selection_mode": "minimum realized n32 SegNet term plus exact archive rate; pose excluded until QS5 compensation",
        "pair_ids": pair_ids.tolist(),
        "rows": rows,
        "base": base,
        "selected": selected,
        "selected_delta_vs_base": {
            "flips": selected["flips"] - base["flips"],
            "seg_term": selected["seg_term"] - base["seg_term"],
            "rate_term": selected["rate_term"] - base["rate_term"],
            "seg_plus_rate": selected["seg_plus_rate_selection_value"] - (base["seg_term"] + base["rate_term"]),
        },
        "boundaries": {
            "full_n600_confirmation": False,
            "candidate_full_receiver_decode": False,
            "pose_compensation": False,
            "score_claim": False,
            "pointer_moved": False,
            "verdict_scope": "TOY-BRACKET tuning only; no instance/formulation/family verdict",
        },
        "next_fire": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "JS8 scorer-slot owner / MAIN",
            "consumer_store": str(build.BULK_ROOT / "full_n600_v1"),
            "fire_trigger": "one full-n600 scorer slot is explicitly owned; compile QS5 frame-0 compensation for the selected retained scale, then exact receiver decode and matched n600 scoring in chunks <=120",
        },
    }
    screen.atomic_json(OUTPUT / "SWEEP_RESULT.json", result)
    screen.atomic_json(
        build.LOGICAL_ROOT / "SWEEP_POINTER.json", {"result": screen.file_record(OUTPUT / "SWEEP_RESULT.json")}
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "selected", "selected_delta_vs_base", "boundaries", "next_fire")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
