#!/usr/bin/env python3
"""Resume HM1 n600 materialization or Range closure from retained payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_hm1_frame_dim_curve as hm1
import torch


def load_cell(hp, output: Path, candidate: str) -> hm1.Cell:
    root = output / "retained/candidates" / candidate
    receipt = json.loads((root / "model_receipt.json").read_text(encoding="utf-8"))
    raw_path = root / "hpac.raw"
    if hp.file_record(raw_path) != receipt["hpac_raw"]:
        raise hm1.HM1Error(f"HPAC payload custody failed: {candidate}")
    return hm1.Cell(
        candidate,
        int(receipt["frame_dim"]),
        tuple(int(value) for value in receipt["kept_dimensions"]),
        tuple(int(value) for value in receipt["dropped_dimensions"]),
        raw_path.read_bytes(),
        root / "checkpoint.pt",
        root / "hpac.xz",
    )


def run(output: Path, candidate: str, stage: str) -> dict[str, object]:
    hp = hm1._hp3()
    cell = load_cell(hp, output, candidate)
    code_root = output / "retained/codes" / candidate
    if stage == "materialize":
        cache_record = hp.file_record(hp.CACHE)
        if cache_record["sha256"] != hp.CACHE_SHA256:
            raise hm1.HM1Error("canonical cache SHA-256 changed")
        packer, inflate = hp.configure_sources()
        cache = torch.load(hp.CACHE, map_location="cpu", weights_only=False)["seg"][: hm1.FRAME_COUNT]
        manifest = hm1.materialize_frames(
            hp=hp,
            packer=packer,
            inflate=inflate,
            cell=cell,
            cache=cache.to(torch.uint8),
            frame_ids=list(range(hm1.FRAME_COUNT)),
            root=code_root,
            stage_frames=24,
            contiguous=True,
        )
        return {
            "stage": stage,
            "candidate": candidate,
            "frames": manifest["frames"],
            "complete": manifest["complete"],
            "manifest": hp.file_record(code_root / "code_manifest.json"),
        }
    manifest = json.loads((code_root / "code_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("tokens") != hm1.FULL_TOKEN_COUNT:
        raise hm1.HM1Error(f"n600 codes are incomplete: {candidate}")
    encoded = hp.encode_monolithic_checkpoint(output, candidate, manifest)
    return {"stage": stage, "candidate": candidate, "complete": True, "encoded": encoded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", choices=("materialize", "encode"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.candidate, args.stage), indent=2))


if __name__ == "__main__":
    main()
