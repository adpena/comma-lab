"""Retain per-layer 3-bit starting packets, before the unresolved seg/pose stage.

These are INITIAL QUANTIZATION candidates, not re-solved or scored candidates.
The current shipping parser and SM1 coder are used. Every layer's other tensors,
row selection, token field, HPAC and carrier remain unchanged. No score claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_ntb2_control as control
from experiments import ddm_rlc1_run as landed
from experiments.ddm_rc1_model_section_adaptive_recode import ck2_interleave
from experiments.ddm_sm3_semantic_representation import quantized_components

ROOT = control.ROOT.parent / "renderer_seeds"


def retain(path, payload):
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside renderer store")
    if shutil.disk_usage(ROOT.parent).free < (40 << 30) + len(payload):
        raise RuntimeError("STORAGE_BLOCK: keep all payloads")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ntb2 initial precision packet")
    return landed.fact(path)


def run():
    import brotli
    import numpy as np
    import torch

    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != control.BASE_SHA:
        raise ValueError("POINTER_MOVED")
    runtime = ROOT / "runtime_copy"
    sources = []
    for source in sorted((control.ROOT / "source_runtime").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
            sources.append(retain(runtime / source.relative_to(control.ROOT / "source_runtime"), source.read_bytes()))
    if landed.fact(runtime / "archive.zip")["sha256"] != control.BASE_SHA:
        raise ValueError("base archive changed")
    rx, renderer, _ = landed.io.load_runtime(runtime)
    from runtime import rc1_adaptive_model_sections as rc1
    from runtime import sm1_semantic_mixer as sm1

    template = renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH).state_dict()
    parts = rx.read_residual_archive(runtime / "archive.zip")
    body = sm1.restore_semantic(parts.semantic_blob, template)
    state = renderer.unpack_variant_semantic_or_none(body, template)
    renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH).load_state_dict(state, strict=True)
    metadata, _ = sm1.split_body(body, template)
    _, descriptors = sm1.plan_metadata(metadata, template)
    offset = 10

    def read(_kind, length):
        nonlocal offset
        value = body[offset : offset + length]
        offset += length
        return value

    plan = rc1.walk_sm3r(read, template, tuple(body[4:8]))
    if offset != len(body):
        raise ValueError("semantic census did not close")
    code_indices = [i for i, item in enumerate(plan) if item["kind"] == "codes"]
    weights_at = sm1.HEADER.size + len(metadata)
    weights = np.frombuffer(parts.semantic_blob[weights_at : weights_at + 24], dtype=np.int8)
    member = landed.io.split_member(landed.io.read_archive_member(runtime / "archive.zip"))
    binding = {
        "sources": sources,
        "producer": landed.fact(Path(__file__)),
        "quantizer": landed.fact(REPO / "experiments/ddm_sm3_semantic_representation.py"),
        "seed": 20260911,
        "score_claim": False,
        "scorer_run": False,
        "status": "INITIAL_PACKET_ONLY; not re-solved; not optimized for scorer; no admissibility claim",
        "cleanup": "all packets kept, stage replay immutable, 40 GiB reserve",
    }
    retain(ROOT / "INPUTS.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())
    table = []
    roster = [None] + [i for i, d in enumerate(descriptors) if d["bits"] == 4]
    for group in roster:
        name = "control" if group is None else descriptors[group]["name"]
        target = ROOT / name
        changed = [dict(item) for item in plan]
        expected = None
        if group is not None:
            desc = descriptors[group]
            value = state[name]
            if "selected_rows" in desc:
                value = value[desc["selected_rows"]]
            restored, scales, codes = quantized_components(name, value, bits=3)
            expected = state[name].clone()
            if "selected_rows" in desc:
                expected[desc["selected_rows"]] = restored
            else:
                expected = restored
            idx = code_indices[group]
            changed[idx - 1]["blob"] = scales.tobytes()
            changed[idx]["blob"] = rc1.pack_signed_codes(codes, 3)
            depth = bytearray(changed[0]["blob"])
            shift = 4 * (group % 2)
            depth[group // 2] = (depth[group // 2] & ~(15 << shift)) | (3 << shift)
            changed[0]["blob"] = bytes(depth)
        new_body = body[:10] + b"".join(item["blob"] for item in changed)
        retain(target / "body.sm3r", new_body)
        decoded = renderer.unpack_variant_semantic_or_none(new_body, template)
        for key, value in state.items():
            torch.testing.assert_close(decoded[key], expected if key == name else value, rtol=0, atol=0)
        if group is not None and torch.equal(decoded[name], state[name]):
            raise ValueError("precision cut changed no values")
        archives = []
        for twin in range(2):
            rider, payload, meta = sm1.encode(new_body, template, weights)
            retain(target / f"rider.twin{twin}.bin", rider)
            retain(target / f"range.twin{twin}.bin", payload)
            retain(target / f"metadata.twin{twin}.bin", meta)
            if sm1.restore_semantic(rider, template) != new_body:
                raise ValueError("SM1 inverse failed")
            outer = ck2_interleave(rider)
            retain(target / f"outer.twin{twin}.ck2", outer)
            section = brotli.compress(outer, quality=10, lgwin=16)
            retain(target / f"section.twin{twin}.br", section)
            sections = dict(member)
            sections["semantic"] = section
            h = list(rx.RX1_MODEL_HEADER.unpack(sections["header"]))
            h[6] = len(section)
            sections["header"] = rx.RX1_MODEL_HEADER.pack(*h)
            packed = landed.io.join_member(sections)
            retain(target / f"member.twin{twin}.bin", packed)
            archive = target / f"archive.twin{twin}.zip"
            landed.ROOT = ROOT
            landed.pack(packed, archive, "stored", None)
            parsed = rx.read_residual_archive(archive)
            if parsed.semantic_blob != rider:
                raise ValueError("archive semantic parser differs")
            for field in ("hpac_blob", "carrier_blob", "token_stream", "residual_payload", "tc1_weights"):
                if getattr(parsed, field) != getattr(parts, field):
                    raise ValueError("unrelated component changed")
            archives.append(landed.fact(archive))
        if archives[0]["sha256"] != archives[1]["sha256"]:
            raise ValueError("initial packet twins differ")
        if group is None and archives[0]["sha256"] != control.BASE_SHA:
            raise ValueError("SM1 container control differs from move44")
        row = {
            "layer": name,
            "semantic_bytes": len(section),
            "twins": archives,
            "delta_bytes_before_resolve": archives[0]["bytes"] - 180406,
            "d_seg": None,
            "d_pose": None,
            "delta_s": None,
            "resolved": False,
            "score_claim": False,
            "status": "INITIAL_PACKET_ONLY",
        }
        retain(target / "RECEIPT.json", (json.dumps(row, indent=2, sort_keys=True) + "\n").encode())
        table.append(row)
        print(json.dumps({"layer": name, "bytes_before_resolve": archives[0]["bytes"]}), flush=True)
    retain(ROOT / "INITIAL_PRICES.json", (json.dumps(table, indent=2, sort_keys=True) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("wrong resume root")
    run()


if __name__ == "__main__":
    main()
