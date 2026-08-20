#!/usr/bin/env python3
"""Finish one fully materialized HM1 n120 cell from retained payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_hm1_frame_dim_curve as hm1


def run(output: Path, candidate: str) -> dict[str, object]:
    hp = hm1._hp3()
    root = output / "retained/candidates" / candidate
    selection_root = output / "retained/selection" / candidate
    receipt = json.loads((root / "model_receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((selection_root / "code_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("tokens") != hm1.SAMPLE_TOKEN_COUNT:
        raise hm1.HM1Error(f"cell is not fully materialized: {candidate}")
    raw_path = root / "hpac.raw"
    if hp.file_record(raw_path) != receipt["hpac_raw"]:
        raise hm1.HM1Error(f"HPAC payload custody failed: {candidate}")
    if manifest.get("hpac_sha256") != receipt["hpac_raw"]["sha256"]:
        raise hm1.HM1Error(f"code/model identity changed: {candidate}")
    cell = hm1.Cell(
        candidate,
        int(receipt["frame_dim"]),
        tuple(int(value) for value in receipt["kept_dimensions"]),
        tuple(int(value) for value in receipt["dropped_dimensions"]),
        raw_path.read_bytes(),
        root / "checkpoint.pt",
        root / "hpac.xz",
    )
    row = hm1.encode_sample(hp=hp, cell=cell, manifest=manifest, root=selection_root)
    hp.replace_json(selection_root / "result.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    row = run(args.output, args.candidate)
    print(
        json.dumps(
            {
                "candidate": row["candidate"],
                "sample_range": row["sample_range"],
                "sample_decode_exact": row["sample_decode_exact"],
                "projected_n600_range_joint_bytes": row["projected_n600_range_joint_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
