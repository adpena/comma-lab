#!/usr/bin/env python3
"""Row-2 SegNet collateral falsifier, guarded by repaired row-1 calibration.

This measures the per-pair gradient-proposed, nearest int4 code action and its
all-600 SegNet collateral. It does not implement a resolved joint exchange row.
All changed renders are retained losslessly as XORs against pinned move52 raw.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO), str(REPO / "src")]
import ddm_jrx2_control as c
import numpy as np

ROOT = c.STORE / "renderer"
NAMES = ("head.weight", "blocks.3.dw.weight", "blocks.3.pw.weight")


def retain(path: Path, payload: bytes) -> dict:
    return c.control.locked_retain(path, payload)


def array(path: Path, **values) -> dict:
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return retain(path, stream.getvalue())


def binding() -> dict:
    original = c.bind()
    gate = c.load(c.STORE / "ROW1_GATE.json")
    if not gate["passed"] or gate["n"] != 12 or gate["k"] / gate["n"] < 0.8:
        raise ValueError("ROW1_NOT_PASSED: no renderer measurement")
    reference = c.load(c.price.REFERENCE)
    for relative, expected in reference["runtime_sources"].items():
        observed = c.fact(c.STORE / "price_runtime" / relative)
        if any(observed[key] != expected[key] for key in ("sha256", "bytes")):
            raise ValueError("copied runtime drift")
    result = {
        "control_binding": original,
        "gate": c.fact(c.STORE / "ROW1_GATE.json"),
        "source": c.fact(Path(__file__)),
        "axis": c.AXIS,
        "score_claim": False,
        "helpers": [
            c.fact(REPO / "experiments" / name)
            for name in (
                "ddm_ren2_restore_init.py",
                "ddm_rw1_renderer_edge_foldback.py",
                "ddm_jg2_tail_reencode.py",
                "ddm_up3_carrier_splice.py",
            )
        ],
    }
    c.save(ROOT / "BINDING.json", result)
    return result


def section():
    import ddm_ren2_restore_init as ren2

    ren2.POINTER_TREE = c.STORE / "price_runtime"
    container = ren2.read_shipped_semantic_container()
    residual, mixer, _inflate, receiver = ren2._receiver_modules()
    rc1 = importlib.import_module("rc1_adaptive_model_sections")
    body, template = container["body"], container["template"]
    offsets, cursor = [], 10

    def read(_kind, count):
        nonlocal cursor
        offsets.append(cursor)
        result = body[cursor : cursor + count]
        cursor += count
        return result

    plan = rc1.walk_sm3r(read, template, tuple(body[4:8]))
    if cursor != len(body) or container["sz1_split"]:
        raise ValueError("unsupported or unconsumed semantic representation")
    runs, index = {}, 1
    for name, value in template.items():
        if value.ndim < 2:
            index += 1
            continue
        pruned = name in rc1.ROW_PRUNE_NAMES
        index += int(pruned)
        scale_item, scale_offset = plan[index], offsets[index]
        index += 1
        item, offset = plan[index], offsets[index]
        index += 1
        if name not in NAMES:
            continue
        if pruned or item["kind"] != "codes" or int(item["bits"]) != 4:
            raise ValueError("actuator is not the reference unpruned int4 run")
        codes = np.asarray(
            rc1.unpack_signed_codes(body[offset : offset + item["length"]], item["count"], 4), dtype=np.int32
        )
        scales = np.frombuffer(body[scale_offset : scale_offset + scale_item["length"]], dtype="<f2").astype(np.float32)
        shape = tuple(template[name].shape)
        runs[name] = {"offset": offset, "length": item["length"], "codes": codes, "scales": scales, "shape": shape}
    if set(runs) != set(NAMES):
        raise ValueError("renderer actuator census mismatch")
    return container, runs, rc1, residual, mixer, receiver


def pack(label: str, action: dict | None) -> dict:
    """The same SM1S/CK2/Brotli/ZIP chain as jrd1, with all twin bytes kept."""
    import brotli
    import ddm_jg2_tail_reencode as jg2
    import ddm_ren2_restore_init as ren2
    from ddm_up3_carrier_splice import _ck2_interleave_planes

    container, runs, rc1, residual, mixer, receiver = section()
    body = container["body"]
    if action is not None:
        run = runs[action["tensor"]]
        codes = run["codes"].copy()
        k = action["flat_index"]
        if codes[k] != action["old"] or abs(action["new"] - action["old"]) != 1 or not -8 <= action["new"] <= 7:
            raise ValueError("not one nearest representable int4 step")
        codes[k] = action["new"]
        encoded = rc1.pack_signed_codes(codes, 4)
        retain(ROOT / label / "code_run.bin", encoded)
        body = body[: run["offset"]] + encoded + body[run["offset"] + run["length"] :]
    source = c.STORE / "price_runtime/archive.zip"
    parts = residual.read_residual_archive(source)
    sections = jg2.split_member(jg2.read_archive_member(source))
    weights = ren2.measure_mixer_weights(container["rider"])
    reference = c.load(Path("/Volumes/APDataStore/pact/ddm_jrd1/renderer/RESULT.json"))
    shape = reference["container"]
    twins = []
    for repeat in range(2):
        dest = ROOT / label / f"repeat{repeat}"
        retain(dest / "body.sm3r", body)
        rider, payload, metadata = mixer.encode(body, container["template"], weights)
        retain(dest / "rider.sm1s", rider)
        retain(dest / "range_payload.bin", payload)
        retain(dest / "metadata.bin", metadata)
        if receiver.restore_sm1_semantic(rider, container["template"]) != body:
            raise ValueError("semantic parse-back differs")
        staged = _ck2_interleave_planes(rider) if container["ck2_semantic"] else rider
        retain(dest / "staged.bin", staged)
        member = brotli.compress(staged, quality=shape["quality"], lgwin=shape["lgwin"])
        retain(dest / "member.br", member)
        changed = dict(sections)
        header = list(jg2.RX1_HEADER.unpack(sections["header"]))
        header[6] = len(member)
        changed["header"], changed["semantic"] = jg2.RX1_HEADER.pack(*header), member
        whole = jg2.join_member(changed)
        retain(dest / "member.p", whole)
        archive = dest / "archive.zip"
        if not archive.exists():
            c.control.custody.check_capacity(len(whole) + 4096)
            jg2.pack_archive(whole, archive)
        if jg2.read_archive_member(archive) != whole:
            raise ValueError("packed archive differs")
        parsed = residual.read_residual_archive(archive)
        if bytes(parsed.semantic_blob) != rider:
            raise ValueError("archive semantic mismatch")
        for name in ("hpac_blob", "carrier_blob", "token_stream", "tc1_weights", "residual_payload"):
            if getattr(parsed, name) != getattr(parts, name):
                raise ValueError("unrelated archive section changed")
        twins.append(c.fact(archive))
    if twins[0]["sha256"] != twins[1]["sha256"]:
        raise ValueError("renderer twins disagree")
    if action is None and twins[0]["sha256"] != c.control.custody.ARCHIVE_SHA:
        raise ValueError("renderer null archive not move52")
    result = {
        "action": action,
        "archives": twins,
        "delta_bytes": twins[0]["bytes"] - 179332,
        "score_claim": False,
        "pose_resolved": False,
    }
    c.save(ROOT / label / "PACK.json", result)
    return result


def proposals() -> None:
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_pp1_pose_actuation as pp1
    import ddm_rw1_renderer_edge_foldback as rw1
    import torch

    c.pd4.bind_move52(raw=c.control.PD4 / "parseback/0.raw", verify_raw=True)
    pp1.set_threads(2)
    torch.manual_seed(c.control.SEED)
    torch.use_deterministic_algorithms(True)
    body = pp1.load_body(with_raw=True, with_segnet=True)
    _container, runs, *_ = section()
    pack("control", None)
    parameters = dict(body.semantic.named_parameters())
    for name in NAMES:
        run = runs[name]
        scale = np.broadcast_to(run["scales"].reshape((-1,) + (1,) * (len(run["shape"]) - 1)), run["shape"])
        realized = run["codes"].reshape(run["shape"]) * scale
        if not np.array_equal(realized.astype(np.float32), parameters[name].detach().numpy()):
            raise ValueError("parsed codes and loaded renderer weights differ")
        parameters[name].requires_grad_(True)
    for p in c.load(c.control.SAMPLE)["pairs"]:
        dest = ROOT / f"pair_{p:03d}"
        if (dest / "PROPOSAL.json").exists():
            continue
        print(json.dumps({"pair": p, "storage_before_heavy": c.storage()}), flush=True)
        baseline = fe1.render_pair(body, p)
        if not np.array_equal(baseline[0], body.raw[2 * p + 1]):
            retain(dest / "FAILED_CONTROL.zlib", zlib.compress(baseline.tobytes(), 6))
            raise ValueError("render does not reproduce pinned raw")
        pred = jg1.argmax_from_camera_frames(body.net, baseline)[0]
        mask = (pred != body.gt[p]) & (body.tokens[p] == body.gt[p])
        base_fact = array(dest / "base.npz", argmax=pred, residual_mask=mask)
        if not mask.any():
            c.save(
                dest / "PROPOSAL.json",
                {"pair": p, "action": None, "reason": "no residual at correct tokens", "artifacts": [base_fact]},
            )
            continue
        for parameter in parameters.values():
            parameter.grad = None
        tokens = torch.from_numpy(body.tokens[p : p + 1].astype(np.int64))
        frame = body.semantic(tokens, torch.tensor([p]))
        logits = rw1._seg_logits(body.net, rw1._exact_r_camera(frame))
        errors = rw1._expected_flip(logits, torch.from_numpy(body.gt[p : p + 1].astype(np.int64)), rw1.TAU_REFERENCE)
        loss = errors[0][torch.from_numpy(mask)].sum()
        loss.backward()
        gradients, options = {}, []
        for name in NAMES:
            run = runs[name]
            scale = np.broadcast_to(run["scales"].reshape((-1,) + (1,) * (len(run["shape"]) - 1)), run["shape"])
            grad = parameters[name].grad.detach().numpy() * scale
            gradients[name] = grad.copy()
            flat = grad.ravel()
            move = -np.sign(flat).astype(np.int32)
            valid = np.isfinite(flat) & (move != 0) & (run["codes"] + move >= -8) & (run["codes"] + move <= 7)
            for k in np.flatnonzero(valid):
                options.append((float(flat[k] * move[k]), name, int(k), int(move[k])))
        gradient_fact = array(dest / "gradients.npz", **gradients)
        if not options:
            raise ValueError("nonempty residual has no finite representable gradient action")
        directional, name, k, delta = min(options)
        action = {
            "tensor": name,
            "flat_index": k,
            "old": int(runs[name]["codes"][k]),
            "new": int(runs[name]["codes"][k]) + delta,
        }
        packed = pack(f"pair_{p:03d}", action)
        c.save(
            dest / "PROPOSAL.json",
            {
                "pair": p,
                "action": action,
                "pack": packed,
                "predicted_surrogate_delta": directional,
                "targeted_cells": int(mask.sum()),
                "selection": "minimum gradient dot nearest allowed one-code int4 step; fixed scales; rw1 tau0.1",
                "scope": "correct-token SegNet residual; gradient ranks only; realized result unmeasured",
                "artifacts": [base_fact, gradient_fact],
            },
        )


def score(pairs: list[int]) -> None:
    import ddm_jg1_seg_solve as jg1
    import ddm_pp1_pose_actuation as pp1
    import torch

    c.pd4.bind_move52(raw=c.control.PD4 / "parseback/0.raw", verify_raw=True)
    pp1.set_threads(4)
    torch.manual_seed(c.control.SEED)
    torch.use_deterministic_algorithms(True)
    body = pp1.load_body(with_raw=True, with_segnet=True)
    fixed_sample = c.load(c.control.SAMPLE)["pairs"]
    if not pairs or len(pairs) != len(set(pairs)) or any(p not in fixed_sample for p in pairs):
        raise ValueError("score scope is outside K24")
    for pair in pairs:
        dest = ROOT / f"pair_{pair:03d}"
        proposal = c.load(dest / "PROPOSAL.json")
        if proposal["action"] is None:
            continue
        for entry in proposal["pack"]["archives"]:
            c.price.checked_fact(entry)
        archive_sha = proposal["pack"]["archives"][0]["sha256"]
        shared = ROOT / "collateral_by_archive" / archive_sha
        model = jg1.load_semantic_renderer(
            Path(proposal["pack"]["archives"][0]["path"]), c.STORE / "price_runtime/runtime"
        )
        for start in range(0, 600, 25):
            output = shared / f"chunk_{start:03d}.json"
            if output.exists():
                chunk = c.load(output)
                for entry in chunk["payloads"] + [chunk["base_argmax"]]:
                    c.price.checked_fact(entry)
                continue
            print(json.dumps({"candidate_pair": pair, "pair": start, "storage_before_heavy": c.storage()}), flush=True)
            c.control.custody.check_capacity(128 << 20)
            indices = list(range(start, min(start + 25, 600)))
            base_path = ROOT / "base_argmax" / f"chunk_{start:03d}.npz"
            if base_path.exists():
                with np.load(base_path, allow_pickle=False) as data:
                    base_predictions = data["argmax"]
            else:
                base_predictions = np.stack(
                    [jg1.argmax_from_camera_frames(body.net, np.asarray(body.raw[2 * p + 1])[None])[0] for p in indices]
                )
                array(base_path, argmax=base_predictions)
            deltas, predictions, rows = [], [], []
            for offset, p in enumerate(indices):
                frame = jg1.render_frame1(model, body.tokens[p : p + 1], np.array([p]))
                deltas.append(np.bitwise_xor(frame[0], np.asarray(body.raw[2 * p + 1])))
                pred = jg1.argmax_from_camera_frames(body.net, frame)[0]
                predictions.append(pred)
                before, after = base_predictions[offset] != body.gt[p], pred != body.gt[p]
                rows.append(
                    {
                        "pair": p,
                        "archive_sha256": archive_sha,
                        "base_errors": int(before.sum()),
                        "errors": int(after.sum()),
                        "delta_cells": int(after.sum()) - int(before.sum()),
                        "new_errors": int((after & ~before).sum()),
                        "fixed_errors": int((before & ~after).sum()),
                        "raw_sha256": c.control.digest(frame.tobytes()),
                        "axis": c.AXIS,
                        "score_claim": False,
                    }
                )
            retained = array(
                shared / f"chunk_{start:03d}.npz", xor_frames=np.stack(deltas), argmax=np.stack(predictions)
            )
            with np.load(retained["path"], allow_pickle=False) as data:
                stored_xor, stored_argmax = data["xor_frames"], data["argmax"]
                for offset, p in enumerate(indices):
                    restored = np.bitwise_xor(stored_xor[offset], body.raw[2 * p + 1])
                    if c.control.digest(restored.tobytes()) != rows[offset]["raw_sha256"]:
                        raise ValueError("delta frame retention failed roundtrip")
                    if not np.array_equal(stored_argmax[offset], predictions[offset]):
                        raise ValueError("argmax retention failed roundtrip")
            c.save(
                output,
                {
                    "rows": rows,
                    "payloads": [retained],
                    "base_argmax": c.fact(base_path),
                    "retention": "lossless XOR against pinned move52 raw; no materialized render discarded",
                },
            )
        rows = [
            row
            for start in range(0, 600, 25)
            for row in c.load(shared / f"chunk_{start:03d}.json")["rows"]
        ]
        target_credit = -rows[pair]["delta_cells"]
        collateral = sum(row["delta_cells"] for row in rows if row["pair"] != pair)
        result = {
            "pair": pair,
            "pairs_scored": 600,
            "scorer_batch": 1,
            "checkpoint_chunk": 25,
            "target_credit_cells": target_credit,
            "other599_net_collateral_cells": collateral,
            "all600_delta_cells": collateral - target_credit,
            "collateral_exceeds_target_seg_credit": collateral > target_credit,
            "delta_archive_bytes": proposal["pack"]["delta_bytes"],
            "shared_collateral": str(shared),
            "archive_sha256": archive_sha,
            "axis": c.AXIS,
            "score_claim": False,
            "resolved_pose": None,
            "boundary": "SegNet collateral falsifier only; not a resolved total-score exchange row",
        }
        c.save(dest / "COLLATERAL.json", result)
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("proposals", "score"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--pairs", default="")
    args = parser.parse_args()
    if args.resume_from.resolve() != c.STORE.resolve():
        raise ValueError("wrong resume store")
    binding()
    if args.stage == "proposals":
        proposals()
    else:
        score([int(p) for p in args.pairs.split(",") if p])


if __name__ == "__main__":
    main()
