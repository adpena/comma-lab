#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed assembler for the int8 training-rungs local evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = "ef2c097f98f74dbd16e77c6f7b60f05e0a630b6bd65ee55bf334336c4549c965"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--b-custody", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    backend, a3, b = (_load(path) for path in (args.backend, args.a3, args.b))
    b_custody = _load(args.b_custody)
    if backend.get("mlx", {}).get("native_quantized_conv_supported") is not False:
        raise ValueError("backend receipt does not establish A1=no native quantized conv")
    if a3.get("provenance", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("A3 checkpoint custody mismatch")
    bm = b.get("measurement", {})
    if (
        b.get("status") != "MEASURED"
        or bm.get("n_pairs") != 600
        or bm.get("n600_evidence") is not True
        or b.get("provenance", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("B is not terminal n600 evidence on the preregistered checkpoint")
    packet = b.get("provenance", {}).get("packet", {})
    if packet.get("parse_back_equals_direct_int8_dequant") is not True:
        raise ValueError("B lacks parser/direct-int8 equality")
    if b_custody.get("receipt_sha256") != _sha256(args.b):
        raise ValueError("B custody addendum does not bind the terminal receipt bytes")
    payload = {
        "schema": "int8_training_rungs_local_bundle.v1",
        "assembled_at_utc": _utc(),
        "lane_id": "lane_int8_training_rungs_20260713",
        "research_only": True,
        "training_launched": False,
        "pointer_delta": "ZERO",
        "labels": {
            "A1": "SOURCE_VERIFIED",
            "A2": "BLOCKED_NOT_MEASURED_TICKET",
            "A3": "BLOCKED_NOT_MEASURED_TICKET" if a3.get("status") != "MEASURED" else "MEASURED",
            "B": "MEASURED_N600",
            "QAT_outcome": "UNMEASURED_TICKET",
        },
        "summary": {
            "A1_mlx_native_quantized_conv": False,
            "A1_mlx_version": backend["mlx"]["installed_version"],
            "A2_ane_forward_latency_ms": backend["coreml_ane"]["ane_forward_latency_ms"],
            "A2_status": backend["coreml_ane"]["status"],
            "A3_status": a3["status"],
            "A3_argmax_flips": None if a3.get("quality") is None else a3["quality"]["argmax_flip_count_vs_fp32"],
            "A3_global_gradient_cosine": None if a3.get("quality") is None else a3["quality"]["global_gradient_cosine"],
            "B_d_seg_fp32_ema": bm["d_seg_fp32_ema"],
            "B_d_seg_parsed_int8": bm["d_seg_parsed_int8"],
            "B_gap_int8_minus_fp32": bm["d_seg_gap_int8_minus_fp32"],
            "B_seg_score_unit_gap_100x": bm["seg_score_unit_gap_100x"],
            "QAT_ticket": bool(b.get("qat_ticket", {}).get("ticketed")),
        },
        "artifacts": {
            "backend_support": _ref(args.backend),
            "a3_quality": _ref(args.a3),
            "b_posthoc_gap": _ref(args.b),
            "b_posthoc_gap_custody": _ref(args.b_custody),
        },
        "verdict_scope": (
            "A1 installed MLX public API; A2/A3 contained local toolchain blockers; B exact v7.5.2 "
            "EMA, first real n600, parsed LVLS1, macOS CPU/numpy advisory Seg-only; no achieved QAT, "
            "d_pose, contest score, or promotion claim"
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
