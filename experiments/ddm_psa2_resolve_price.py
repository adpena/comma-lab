#!/usr/bin/env python3
"""Resolved-pose and full counted PSA1 bias prices for completed psa2 searches.

No runtime receiver change or dispatch. All outputs remain research artifacts.
The pose evaluator's disk cache makes deterministic interrupted solves replayable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm_psa2_rgb_bias as p


def resolve(pairs: list[int]) -> None:
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pp1_pose_actuation as pp1
    import ddm_up2_shipping_pose_solve as up2
    import numpy as np
    import torch

    p.section()
    body, inst = p.instrument(with_pose=True)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    mean = p.load(p.STORE / "CONTROL.json")["base_mean"]
    for pair in pairs:
        root = p.STORE / "pairs" / f"pair_{pair:03d}"
        if (root / "RESOLVED.json").exists():
            prior = p.load(root / "RESOLVED.json")
            p.checked(prior["selected"])
            continue
        row = p.load(root / "SEARCH.json")
        if row["vectors"] != 4096:
            raise ValueError("no partial-lattice pose admission")
        p.checked(row["selected"])
        print(json.dumps({"pair": pair, "stage": "resolve", "storage_before_heavy": p.capacity(32 << 20)}), flush=True)
        with np.load(root / "selected.npz") as selected:
            frame1 = selected["frame1"]
        overlay = pp1.MemoryOverlayRaw(body.raw, {2 * pair + 1: frame1})
        moved = br1.Instrument(inst.state, overlay, inst.targets, inst.posenet, inst.blow, inst.gram, inst.bmat)
        original = br1.evaluate_codes
        source_sha = p.fact(Path(__file__))["sha256"]
        inputs = {"pair": pair, "search": p.fact(root / "SEARCH.json"), "source_sha256": source_sha,
                  "jg5": p.fact(Path(jg5.__file__)), "br1": p.fact(Path(br1.__file__)),
                  "start_codes": codes[pair].tolist(), "outer_rounds": 40, "max_gn_iterations": 400,
                  "dd_threshold": jg5.materiality_dd_threshold(mean)}
        p.save(root / "SOLVE_INPUTS.json", inputs)

        def cached(target, pair_id, block, moved=moved, pair=pair, root=root, original=original):
            if target is not moved or pair_id != pair:
                raise ValueError("pose cache used outside its bound pair and render")
            block = np.asarray(block, dtype=np.int32)
            key = hashlib.sha256(block.tobytes() + str(block.shape).encode()).hexdigest()
            dest = root / "pose_evaluations" / (key + ".npz")
            receipt = dest.with_suffix(".json")
            if receipt.exists():
                p.checked(p.load(receipt))
                with np.load(dest) as bank:
                    if not np.array_equal(bank["codes"], block):
                        raise ValueError("pose cache identity collision")
                    return bank["values"].copy()
            values = original(target, pair_id, block)
            p.save(receipt, p.array(dest, codes=block, values=values))
            return values

        br1.evaluate_codes = cached
        try:
            result = jg5.refine_pair(moved, pair, codes[pair], dd_threshold=inputs["dd_threshold"],
                                     outer_rounds=40, max_gn_iterations=400)
        finally:
            br1.evaluate_codes = original
        refined = np.asarray(result["codes"], dtype=np.int32)[None]
        actual = float(original(moved, pair, refined)[0])
        base_control = p.load(root / "CONTROL.json")
        replay_difference = abs(actual - result["final_d_pose"])
        replay = {"candidate_batch_d_pose": result["final_d_pose"], "batch1_d_pose": actual,
                  "abs_difference": replay_difference, "band_abs": base_control["band_abs"]}
        p.save(root / "POSE_REPLAY.json", replay)
        if replay_difference > base_control["band_abs"]:
            raise ValueError("selected resolved pose replay outside measured batch band")
        index = np.asarray([pair], dtype=np.int64)
        with torch.inference_mode():
            coeff = up2.codes_to_coefficients(refined, inst.state.coefficient_scales)
            frame0 = up2.render_frame0(coeff, inst.state, index, differentiable=False)
            camera0 = frame0.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
        selected_fact = p.array(root / "resolved_pair.npz", frame0=camera0, frame1=frame1,
                                carrier_codes=refined[0], bias_codes=np.asarray(row["best"]["codes"], dtype=np.int8))
        dp0 = base_control["d_pose"]
        p.save(root / "RESOLVED.json", {"pair": pair, "selected": selected_fact, "solver": result,
            "bias_codes": row["best"]["codes"], "credit_cells": row["credit_cells"],
            "d_pose_base": dp0, "d_pose_resolved": actual, "delta_d_pose": actual - dp0,
            "pose_replay": replay,
            "zero_bias_control": not any(row["best"]["codes"]), "axis": p.AXIS, "score_claim": False})
        print(json.dumps({"resolved_pair": pair, "delta_d_pose": actual - dp0,
                          "credit_cells": row["credit_cells"], "stop": result["stop_reason"]}), flush=True)


def instance(pair: int, codes: list[int], scales, rc1) -> dict:
    import numpy as np

    values = np.asarray(codes, dtype=np.int32)
    if values.shape != (3,) or values.min() < -8 or values.max() > 7:
        raise ValueError("not a three-channel int4 action")
    metadata = struct.pack("<HBHHH", pair, 1, 3, 1, 3) + np.asarray(scales, dtype="<f2").tobytes() + b"\0"
    return {"pair": pair, "metadata": metadata, "packed": rc1.pack_signed_codes(values, 4), "groups": [values]}


def price(label: str, pairs: list[int]) -> None:
    import brotli
    import ddm_jg2_tail_reencode as jg2
    import ddm_psa1_byte_price as psa1
    import ddm_ren2_restore_init as ren2
    import ddm_up3_carrier_splice as splice
    import numpy as np

    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError("invalid price label")
    root = p.STORE / "prices" / label
    if (root / "PRICE.json").exists():
        prior = p.load(root / "PRICE.json")
        if prior["pairs"] != sorted(pairs):
            raise ValueError("price label already binds a different set")
        for entry in prior["artifacts"]:
            p.checked(entry)
        return
    print(json.dumps({"price": label, "storage_before_heavy": p.capacity(8 << 20)}), flush=True)
    container, runs, rc1, residual, mixer, receiver = p.section()
    runtime = p.STORE / "price_runtime"
    body = splice.parse_shipped_body(runtime, verify_sha=False)
    codes = np.asarray(body.codes, dtype=np.int32).copy()
    selected = []
    for pair in sorted(pairs):
        row = p.load(p.STORE / "pairs" / f"pair_{pair:03d}/RESOLVED.json")
        p.checked(row["selected"])
        codes[pair] = row["solver"]["codes"]
        selected.append(instance(pair, row["bias_codes"], runs["head.weight"]["scales"], rc1))
    raw = psa1.serialize("B3", selected, 0)
    aux = psa1.serialize("B3", selected, 1)
    artifacts = [p.retain(root / "section.raw", raw), p.retain(root / "section.metadata", aux),
                 p.array(root / "carrier_codes.npz", codes=codes)]
    ids, _, groups = psa1.parse_section(raw, rc1, mixer)
    expected = [v for row in selected for v in row["groups"]]
    if ids != sorted(pairs) or any(not np.array_equal(a, b) for a, b in zip(groups, expected, strict=True)):
        raise ValueError("bias section round trip differs")
    metadata, source = mixer.split_body(container["body"], container["template"])
    _, descriptors = mixer.plan_metadata(metadata, container["template"])
    _, extra, _ = psa1.parse_section(aux, rc1, mixer)
    weights = ren2.measure_mixer_weights(container["rider"])
    shape = p.load(p.PSA1 / "inputs/jrd1_bytes.json")["container"]
    twins = {"q11": [], "sm1": [], "carrier": []}
    for repeat in range(2):
        dest = root / f"repeat{repeat}"
        # q9/lgwin16 is the source carrier shape; refuse control drift rather than search containers.
        built = splice.build_archive(body, codes, runtime_dir=runtime, container_options=[(body.ck2_carrier, 9, 16)])
        artifact = p.retain(dest / "carrier.zip", built["archive_bytes"])
        artifacts.append(artifact)
        twins["carrier"].append(artifact)
        if not pairs and artifact["sha256"] != p.PIN:
            raise ValueError("carrier null rebuild is not move52; shape needs re-derivation")
        sections = jg2.split_member(jg2.read_archive_member(Path(artifact["path"])))
        br = brotli.compress(raw, quality=11)
        artifacts.append(p.retain(dest / "section.q11.br", br))
        if brotli.decompress(br) != raw:
            raise ValueError("standalone bias decode differs")
        whole = jg2.join_member(sections) + br + struct.pack("<4sI", b"PSAX", len(br))
        artifacts.append(p.retain(dest / "q11.member.p", whole))
        archive = dest / "q11.research.zip"
        if not archive.exists():
            jg2.pack_archive(whole, archive)
        if jg2.read_archive_member(archive) != whole:
            raise ValueError("counted q11 archive differs")
        twins["q11"].append(p.fact(archive))
        artifacts.append(twins["q11"][-1])
        payload, encoded, _, _ = mixer.walk(descriptors + extra, weights, source=source + expected)
        artifacts.append(p.retain(dest / "extended.range.bin", payload))
        if not pairs:
            payload_length = mixer.HEADER.unpack_from(container["rider"])[-1]
            if payload != container["rider"][-payload_length:]:
                raise ValueError("empty extension changed the original arithmetic stream")
        if any(not np.array_equal(a, b) for a, b in zip(encoded, source + expected, strict=True)):
            raise ValueError("real SM1 walker symbols differ")
        rider = mixer.HEADER.pack(b"SM1X", 1, 24, 0, len(payload)) + metadata + aux + weights.tobytes() + payload
        artifacts.append(p.retain(dest / "rider.sm1x", rider))
        psa1.verify_sm1x(rider, container, rc1, mixer, source + expected, sorted(pairs))
        staged = splice._ck2_interleave_planes(rider) if container["ck2_semantic"] else rider
        artifacts.append(p.retain(dest / "staged.bin", staged))
        semantic = brotli.compress(staged, quality=shape["quality"], lgwin=shape["lgwin"])
        artifacts.append(p.retain(dest / "semantic.br", semantic))
        fields = list(jg2.RX1_HEADER.unpack(sections["header"]))
        fields[6] = len(semantic)
        sections["header"] = jg2.RX1_HEADER.pack(*fields)
        sections["semantic"] = semantic
        whole = jg2.join_member(sections)
        artifacts.append(p.retain(dest / "sm1.member.p", whole))
        archive = dest / "sm1.research.zip"
        if not archive.exists():
            jg2.pack_archive(whole, archive)
        twins["sm1"].append(p.fact(archive))
        artifacts.append(twins["sm1"][-1])
        if jg2.read_archive_member(archive) != whole:
            raise ValueError("counted SM1 archive member differs")
    if any(x[0]["sha256"] != x[1]["sha256"] for x in twins.values()):
        raise ValueError("price twins differ")
    p.save(root / "PRICE.json", {"pairs": sorted(pairs), "twins": twins, "artifacts": artifacts,
           "delta_bytes_q11": twins["q11"][0]["bytes"] - 179332,
           "delta_bytes_sm1": twins["sm1"][0]["bytes"] - 179332,
           "carrier_delta_bytes": twins["carrier"][0]["bytes"] - 179332,
           "axis": p.AXIS, "research_only": True, "receiver_implemented": False, "score_claim": False})


def main() -> None:
    import random

    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("resolve", "price"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.resume_from.resolve() != p.STORE.resolve():
        raise ValueError("wrong resume store")
    pairs = [int(x) for x in args.pairs.split(",") if x]
    roster = p.load(p.STORE / "SAMPLE.json")["pairs"]
    if len(set(pairs)) != len(pairs) or any(x not in roster for x in pairs):
        raise ValueError("pairs outside fixed60 or duplicated")
    random.seed(p.SEED)
    np.random.seed(p.SEED)
    torch.manual_seed(p.SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "resolve":
        resolve(pairs or roster)
    else:
        price(args.label, pairs)


if __name__ == "__main__":
    main()
