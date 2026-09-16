#!/usr/bin/env python3
"""Literal all600 collateral census of a counted psa2 research action.

This exercises the actual renderer and repaired carrier with parsed bias values.
It is not the conditional public receiver or a scorer/evaluation result. Every
rendered byte is checked against an already persisted frame, or retained on error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm_psa2_rgb_bias as p


def census(label: str) -> None:
    import brotli
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg2_tail_reencode as jg2
    import ddm_psa1_byte_price as psa1
    import ddm_up2_shipping_pose_solve as up2
    import numpy as np
    import torch

    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError("invalid price label")
    root = p.STORE / "census" / label
    price_root = p.STORE / "prices" / label
    price = p.load(price_root / "PRICE.json")
    for artifact in price["artifacts"]:
        p.checked(artifact)
    inputs = {"price": p.fact(price_root / "PRICE.json"), "producer": p.fact(Path(__file__)),
              "rgb_producer": p.fact(Path(p.__file__)), "base_binding": p.fact(p.STORE / "BINDING.json")}
    p.save(root / "INPUTS.json", inputs)
    if (root / "CENSUS.json").exists():
        return
    print(json.dumps({"census": label, "storage_before_heavy": p.capacity(16 << 20)}), flush=True)
    _, runs, rc1, _, mixer, _ = p.section()
    scales = np.asarray(runs["head.weight"]["scales"], dtype=np.float32)
    member = jg2.read_archive_member(Path(price["twins"]["q11"][0]["path"]))
    magic, length = struct.unpack("<4sI", member[-8:])
    if magic != b"PSAX" or length > len(member) - 8:
        raise ValueError("counted standalone bias trailer differs")
    raw = brotli.decompress(member[-8-length:-8])
    if raw != (price_root / "section.raw").read_bytes():
        raise ValueError("counted archive bias differs from retained section")
    if psa1.HEADER.unpack_from(raw)[2:4] != (psa1.FORMS["B3"], 0):
        raise ValueError("not a packed B3 bias section")
    ids, _, groups = psa1.parse_section(raw, rc1, mixer)
    if ids != price["pairs"] or len(groups) != len(ids):
        raise ValueError("bias section key census differs")
    # B3 has one fixed descriptor per pair; check the actual stored scales too.
    cursor = psa1.HEADER.size
    for _ in ids:
        pair, nvec, size, cols, ns = struct.unpack_from("<HBHHH", raw, cursor)
        actual_scales = np.frombuffer(raw[cursor+9:cursor+15], dtype="<f2").astype(np.float32)
        if nvec != 1 or size != 3 or cols != 1 or ns != 3 or not np.array_equal(actual_scales, scales):
            raise ValueError("counted RGB scale descriptor differs")
        cursor += 18
    if cursor != len(raw):
        raise ValueError("unexpected B3 record size")
    bias = {pair: np.asarray(code, dtype=np.int32) for pair, code in zip(ids, groups, strict=True)}
    body, inst = p.instrument(with_pose=True)
    with np.load(price_root / "carrier_codes.npz") as bank:
        codes = bank["codes"].copy()
    base_codes = np.asarray(inst.state.codes, dtype=np.int32)
    unedited = [x for x in range(600) if x not in bias]
    if not np.array_equal(codes[unedited], base_codes[unedited]):
        raise ValueError("carrier codes spill onto unedited pairs")
    rows = []
    for pair in range(600):
        receipt = root / "rows" / f"pair_{pair:03d}.json"
        if receipt.exists():
            rows.append(p.load(receipt))
            continue
        if pair % 100 == 0:
            print(json.dumps({"pair": pair, "storage_before_heavy": p.capacity(8 << 20)}), flush=True)
        code = bias.get(pair)

        def hook(_module, _inputs, output, code=code):
            if code is None:
                return output
            offset = torch.from_numpy((scales * code).astype(np.float32)).view(1, 3, 1, 1)
            return output + offset

        handle = body.semantic.head.register_forward_hook(hook)
        try:
            camera1 = fe1.render_pair(body, pair)[0]
        finally:
            handle.remove()
        index = np.asarray([pair], dtype=np.int64)
        with torch.inference_mode():
            coeff = up2.codes_to_coefficients(codes[index], inst.state.coefficient_scales)
            frame0 = up2.render_frame0(coeff, inst.state, index, differentiable=False)
            camera0 = frame0.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
        if pair in bias:
            resolved = p.load(p.STORE / "pairs" / f"pair_{pair:03d}/RESOLVED.json")
            p.checked(resolved["selected"])
            with np.load(resolved["selected"]["path"]) as selected:
                matches = np.array_equal(camera0, selected["frame0"]) and np.array_equal(camera1, selected["frame1"])
            payload = resolved["selected"]
        else:
            matches = np.array_equal(camera0, body.raw[2*pair]) and np.array_equal(camera1, body.raw[2*pair+1])
            payload = {"base_raw": p.load(p.STORE / "BINDING.json")["raw"], "frames": [2*pair, 2*pair+1]}
        row = {"pair": pair, "edited": pair in bias, "literal_match": bool(matches),
               "frame0_sha256": hashlib.sha256(camera0.tobytes()).hexdigest(),
               "frame1_sha256": hashlib.sha256(camera1.tobytes()).hexdigest(), "retained_payload": payload}
        if not matches:
            row["unexpected_payload"] = p.array(root / "mismatches" / f"pair_{pair:03d}.npz", frame0=camera0, frame1=camera1)
        p.save(receipt, row)
        if not matches:
            raise ValueError(f"literal census mismatch on pair {pair}")
        rows.append(row)
    if len(rows) != 600 or not all(x["literal_match"] for x in rows):
        raise ValueError("incomplete all600 census")
    p.save(root / "CENSUS.json", {"n_pairs": 600, "edited_pairs": ids,
           "unedited_bit_identical": len(unedited), "unedited_denominator": len(unedited),
           "edited_match_retained": len(ids), "rows": rows, "inputs": inputs,
           "surface": "literal research renderer; public receiver not implemented", "axis": p.AXIS, "score_claim": False})


def main() -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != p.STORE.resolve():
        raise ValueError("wrong census store")
    random.seed(p.SEED)
    np.random.seed(p.SEED)
    torch.manual_seed(p.SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    census(args.label)


if __name__ == "__main__":
    main()
