#!/usr/bin/env python3
"""Retain scorer-free PR130 Seg-axis evidence and price measured levers.

This audit does not run SegNet.  It re-reduces already-retained chroma and
decoder-target evidence, materializes the stage-07 counterfactual archive, and
prices two current-vehicle levers against exact complete-archive bytes.  The
missing PR130 candidate-error edge decomposition remains a scorer-gated job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")


SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_OUT = SSD_ROOT / "ddm_sg2_20260810/source_audit_v3"
DEFAULT_BASE = SSD_ROOT / "ddm_pr130_reproduce_20260809/reproduction/archive.zip"
DEFAULT_STAGE07 = (
    SSD_ROOT
    / "pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints"
    / "semantic_renderer_w96_b4_qat4_12k.pt"
)
DEFAULT_STAGE08 = (
    SSD_ROOT
    / "pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints"
    / "semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)
DEFAULT_CHROMA_RECEIPT = Path(
    ".omx/research/ddm_rm1_20260808/chroma_siting_sensitivity.json"
)
DEFAULT_AV_CACHE = SSD_ROOT / "ddm_chroma_dali_av_20260809/gt_cache_av.pt"  # GT_LINEAGE_OK: a seg-axis SOURCE AUDIT that binds both lineages side by side (DALI on the next line) in order to compare them; the AV binding is the audit's subject, not its objective
DEFAULT_DALI_CACHE = SSD_ROOT / "ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
DEFAULT_SD1_RESULT = Path(
    ".omx/research/ddm_sd1_semantic_20260809/SD1_RESULTS.json"
)
EXPECTED_STAGE07_SHA256 = (
    "1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf"
)
EXPECTED_STAGE08_SHA256 = sd1.EXPECTED_CHECKPOINT_SHA256
EXPECTED_CHROMA_RECEIPT_SHA256 = (
    "f1e5d1321ec6bfb47daf984da41737c0a9e95ec2bfdf1fb1e59feb72b3690563"
)
EXPECTED_AV_CACHE_SHA256 = (
    "837b5852dc71ded7ffd20f59f0e8192a4ce753fe1a7b36882ed8e09f211e1f99"
)
EXPECTED_DALI_CACHE_SHA256 = (
    "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
)
EXPECTED_SD1_ARCHIVE_SHA256 = (
    "010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67"
)
EXPECTED_PR130_REPORTED_DSEG = 0.00028609
EDGE_CLASSES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"resume artifact differs: {path}")
    atomic_write_bytes(path, payload)
    return artifact(path)


def require_sha(path: Path, expected: str) -> dict[str, Any]:
    record = artifact(path)
    if record["sha256"] != expected:
        raise ValueError(
            f"source pin differs for {path}: {record['sha256']} != {expected}"
        )
    return record


def require_ssd(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SSD_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"evidence must remain below {SSD_ROOT}: {resolved}") from error
    return resolved


def edge_table(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    if first.shape != second.shape or first.dtype != torch.uint8:
        raise ValueError("edge inputs must be equally shaped uint8 tensors")
    matrix = torch.zeros((len(EDGE_CLASSES), len(EDGE_CLASSES)), dtype=torch.int64)
    for source in range(len(EDGE_CLASSES)):
        source_mask = first == source
        for target in range(len(EDGE_CLASSES)):
            matrix[source, target] = torch.count_nonzero(
                source_mask & (second == target)
            )
    disagreements = int(torch.count_nonzero(first != second))
    symmetric = []
    for left in range(len(EDGE_CLASSES)):
        for right in range(left + 1, len(EDGE_CLASSES)):
            count = int(matrix[left, right] + matrix[right, left])
            symmetric.append(
                {
                    "edge": f"{EDGE_CLASSES[left]}<->{EDGE_CLASSES[right]}",
                    "pixels": count,
                    "share_of_disagreements": count / disagreements,
                }
            )
    symmetric.sort(key=lambda row: (-row["pixels"], row["edge"]))
    return {
        "population_pixels": first.numel(),
        "disagreement_pixels": disagreements,
        "disagreement_fraction": disagreements / first.numel(),
        "directed_matrix_rows_first_columns_second": matrix.tolist(),
        "symmetric_edges": symmetric,
        "road_participating_pixels": sum(
            row["pixels"] for row in symmetric if row["edge"].startswith("Road<->")
        ),
    }


def load_state(path: Path) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"unsupported checkpoint: {path}")
    return checkpoint, checkpoint["state_dict"]


def semantic_price(
    *,
    old_dseg: float,
    new_dseg: float,
    old_archive_bytes: int,
    new_archive_bytes: int,
) -> dict[str, Any]:
    delta_dseg = new_dseg - old_dseg
    delta_bytes = new_archive_bytes - old_archive_bytes
    seg_delta_s = 100.0 * delta_dseg
    rate_delta_s = 25.0 * delta_bytes / sd1.ORIGINAL_BYTES
    return {
        "old_d_seg": old_dseg,
        "new_d_seg": new_dseg,
        "delta_d_seg": delta_dseg,
        "old_archive_bytes": old_archive_bytes,
        "new_archive_bytes": new_archive_bytes,
        "delta_archive_bytes": delta_bytes,
        "seg_delta_s": seg_delta_s,
        "rate_delta_s": rate_delta_s,
        "semantic_leg_delta_s": seg_delta_s + rate_delta_s,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--stage07", type=Path, default=DEFAULT_STAGE07)
    parser.add_argument("--stage08", type=Path, default=DEFAULT_STAGE08)
    parser.add_argument("--chroma-receipt", type=Path, default=DEFAULT_CHROMA_RECEIPT)
    parser.add_argument("--av-cache", type=Path, default=DEFAULT_AV_CACHE)
    parser.add_argument("--dali-cache", type=Path, default=DEFAULT_DALI_CACHE)
    parser.add_argument("--sd1-result", type=Path, default=DEFAULT_SD1_RESULT)
    parser.add_argument("--minimum-free-bytes", type=int, default=200_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = require_ssd(args.out_dir)
    resume_path = require_ssd(args.resume_from)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_dir).free < args.minimum_free_bytes:
        raise RuntimeError("storage preflight refused: insufficient SSD free space")

    source_pins = {
        "audit_source": artifact(Path(__file__).resolve()),
        "base_archive": require_sha(
            args.base_archive, sd1.EXPECTED_BASE_ARCHIVE_SHA256
        ),
        "stage07_checkpoint": require_sha(args.stage07, EXPECTED_STAGE07_SHA256),
        "stage08_checkpoint": require_sha(args.stage08, EXPECTED_STAGE08_SHA256),
        "chroma_receipt": require_sha(
            args.chroma_receipt, EXPECTED_CHROMA_RECEIPT_SHA256
        ),
        "av_cache": require_sha(args.av_cache, EXPECTED_AV_CACHE_SHA256),
        "dali_cache": require_sha(args.dali_cache, EXPECTED_DALI_CACHE_SHA256),
        "sd1_result": artifact(args.sd1_result),
    }
    fingerprints = {
        name: record["sha256"] for name, record in source_pins.items()
    }
    configuration = {
        "out_dir": str(out_dir),
        "resume_from": str(resume_path),
        "base_archive": str(args.base_archive.resolve()),
        "stage07": str(args.stage07.resolve()),
        "stage08": str(args.stage08.resolve()),
        "chroma_receipt": str(args.chroma_receipt.resolve()),
        "av_cache": str(args.av_cache.resolve()),
        "dali_cache": str(args.dali_cache.resolve()),
        "sd1_result": str(args.sd1_result.resolve()),
        "minimum_free_bytes": args.minimum_free_bytes,
        "seed": None,
        "determinism": "no RNG is used; archive builder is deterministic",
    }
    if resume_path.exists():
        progress = json.loads(resume_path.read_text())
        if progress.get("fingerprints") != fingerprints:
            raise ValueError("resume fingerprints differ")
        if progress.get("configuration") != configuration:
            raise ValueError("resume configuration differs")
    else:
        progress = {
            "schema": "ddm_sg2_source_audit.progress.v1",
            "created_at_utc": utc_now(),
            "argv": sys.argv,
            "configuration": configuration,
            "fingerprints": fingerprints,
            "completed_stages": [],
        }
        atomic_write_json(resume_path, progress)

    chroma = json.loads(args.chroma_receipt.read_text())
    pair_ids = np.asarray(
        [int(row["pair_idx"]) for row in chroma["rows"]], dtype="<i2"
    )
    pair_id_artifact = retain_bytes(
        out_dir / "retained/chroma_siting/pair_ids.int16le", pair_ids.tobytes()
    )
    disagreements = sum(
        int(row["argmax_disagree_pixels"]) for row in chroma["rows"]
    )
    chroma_denominator = sum(int(row["argmax_pixels"]) for row in chroma["rows"])
    chroma_fraction = disagreements / chroma_denominator
    if chroma_fraction != float(chroma["pooled_argmax_disagree_frac"]):
        raise ValueError("chroma receipt scalar does not re-reduce exactly")
    chroma_result = {
        "axis": "[macOS-CPU advisory]",
        "population_kind": "seeded stratified-random source-frame chroma-siting perturbation",
        "sampled_pairs": len(chroma["rows"]),
        "population_pairs": 600,
        "denominator_pixels": chroma_denominator,
        "disagreement_pixels": disagreements,
        "disagreement_fraction": chroma_fraction,
        "fraction_of_rounded_pr130_reported_d_seg": (
            chroma_fraction / EXPECTED_PR130_REPORTED_DSEG
        ),
        "positive_control_centered_path_byte_identical_to_upstream": bool(
            chroma["positive_control"]["centered_path_byte_identical_to_upstream"]
        ),
        "pair_ids": pair_ids.astype(int).tolist(),
        "pair_ids_artifact": pair_id_artifact,
        "verdict": (
            "REPRODUCED_AS_A_HYPOTHETICAL_SOURCE_DECODER_PERTURBATION; "
            "NOT_PR130_CANDIDATE_ERROR_AND_NOT_DALI_VS_AV_CAUSAL_ATTRIBUTION"
        ),
    }
    progress["completed_stages"] = sorted(
        set(progress["completed_stages"]) | {"chroma_siting_reduction"}
    )
    atomic_write_json(resume_path, progress)

    av = torch.load(args.av_cache, map_location="cpu", weights_only=False)["seg"]
    dali = torch.load(args.dali_cache, map_location="cpu", weights_only=False)["seg"]
    decoder_edges = edge_table(av, dali)
    decoder_edges.update(
        {
            "axis": "[same-host Tesla T4; AV-vs-DALI retained target-cache reduction]",
            "av_cache": source_pins["av_cache"],
            "dali_cache": source_pins["dali_cache"],
            "fraction_of_rounded_pr130_reported_d_seg": (
                decoder_edges["disagreement_fraction"]
                / EXPECTED_PR130_REPORTED_DSEG
            ),
            "verdict": (
                "MEASURED_SOURCE_TARGET_DECODER_DIFFERENCE; "
                "NOT_PR130_CANDIDATE_ERROR_DECOMPOSITION"
            ),
        }
    )
    progress["completed_stages"] = sorted(
        set(progress["completed_stages"]) | {"decoder_edge_reduction"}
    )
    atomic_write_json(resume_path, progress)

    base = sd1.read_base_archive(args.base_archive)
    if len(base.archive_bytes) != sd1.EXPECTED_BASE_ARCHIVE_BYTES:
        raise ValueError("base archive byte count differs")
    stage07_checkpoint, stage07_state = load_state(args.stage07)
    stage08_checkpoint, _ = load_state(args.stage08)
    allocation = dict.fromkeys(sd1.quantized_names(stage07_state), 4)
    stage07_semantic, stage07_expected = sd1.pack_semantic_state(
        stage07_state, allocation, legacy_int4=True
    )
    stage07_archive = sd1.rebuild_archive(base, stage07_semantic)
    stage07_repeat = sd1.rebuild_archive(base, stage07_semantic)
    if stage07_archive != stage07_repeat:
        raise ValueError("independent stage07 archive builds differ")
    parsed_semantic = sd1.semantic_blob_from_archive(stage07_archive, base)
    if parsed_semantic != stage07_semantic:
        raise ValueError("stage07 semantic archive parse-back differs")
    decoded_stage07, decoded_allocation, format_name = sd1.unpack_semantic_state(
        parsed_semantic, stage07_state
    )
    sm3.assert_state_equal(stage07_expected, decoded_stage07)
    stage07_dir = out_dir / "retained/stage07_counterfactual"
    stage07_retained = {
        "semantic_payload": retain_bytes(
            stage07_dir / "semantic.bin", stage07_semantic
        ),
        "archive": retain_bytes(stage07_dir / "archive.zip", stage07_archive),
        "archive_repeat": retain_bytes(
            stage07_dir / "archive.repeat.zip", stage07_repeat
        ),
        "decoded_state": retain_bytes(
            stage07_dir / "decoded_state.sm3state", sm3.state_wire(decoded_stage07)
        ),
    }
    stage07_dseg = float(stage07_checkpoint["result"]["quantized_exact_seg"])
    stage08_dseg = float(stage08_checkpoint["result"]["quantized_exact_seg"])
    tail_qat_price = semantic_price(
        old_dseg=stage07_dseg,
        new_dseg=stage08_dseg,
        old_archive_bytes=len(stage07_archive),
        new_archive_bytes=len(base.archive_bytes),
    )
    tail_qat_price.update(
        {
            "lever": "stage07_to_stage08_expected_flip_tail_qat",
            "axis": (
                "[checkpoint-recorded CUDA n600 research cache plus exact scorer-free "
                "complete-archive bytes]"
            ),
            "format": format_name,
            "decoded_allocation": dict(decoded_allocation),
            "retained": stage07_retained,
            "status": "MEASURED_AND_ALREADY_CONSUMED_BY_PR130_BASE",
        }
    )
    progress["completed_stages"] = sorted(
        set(progress["completed_stages"]) | {"stage07_archive_and_tail_price"}
    )
    atomic_write_json(resume_path, progress)

    sd1_result = json.loads(args.sd1_result.read_text())
    q4 = next(
        row
        for row in sd1_result["n600_measurements"]
        if row["candidate_id"] == "uniform_q4_legacy_n600"
    )
    mixed = next(
        row
        for row in sd1_result["n600_measurements"]
        if row["candidate_id"] == "selected_mixed_n600"
    )
    mixed_archive = Path(mixed["archive_path"])
    require_sha(mixed_archive, EXPECTED_SD1_ARCHIVE_SHA256)
    mixed_price = semantic_price(
        old_dseg=float(q4["d_seg"]),
        new_dseg=float(mixed["d_seg"]),
        old_archive_bytes=int(q4["archive_bytes"]),
        new_archive_bytes=int(mixed["archive_bytes"]),
    )
    mixed_price.update(
        {
            "lever": "sd1_selected_mixed_q3q4",
            "axis": sd1_result["axis"],
            "retained_archive": artifact(mixed_archive),
            "public_receiver_readable": bool(
                mixed["receiver_status"] == "PUBLIC_RECEIVER_CLOSED"
            ),
            "status": "MEASURED_SEMANTIC_LEG_WIN_POSE_UNMEASURED",
        }
    )

    result = {
        "schema": "ddm_sg2_source_audit.result.v1",
        "completed_at_utc": utc_now(),
        "axis": "[scorer-free audit with explicitly labeled reused measurement axes]",
        "score_claim": False,
        "pointer_moved": False,
        "run": {
            "argv": sys.argv,
            "configuration": configuration,
        },
        "source_pins": source_pins,
        "pr130_reported_seg": {
            "rounded_d_seg": EXPECTED_PR130_REPORTED_DSEG,
            "seg_contribution_s": 100.0 * EXPECTED_PR130_REPORTED_DSEG,
            "implied_mismatches_at_n600": (
                EXPECTED_PR130_REPORTED_DSEG * 600 * 384 * 512
            ),
            "status": "NOT_EXACTLY_RECONSTRUCTIBLE_FROM_ROUNDED_SCALAR",
            "candidate_argmax_payload_status": "NOT_RETAINED_IN_SEARCHED_SCOPE",
            "per_edge_decomposition_status": (
                "UNMEASURED_SCORER_GATED_AND_SOURCE_PAYLOAD_NOT_RETAINED"
            ),
        },
        "chroma_siting_906": chroma_result,
        "av_dali_source_target_edges": decoder_edges,
        "exchange_rate": {
            "rate_s_per_1000_bytes": 25_000.0 / sd1.ORIGINAL_BYTES,
            "original_video_bytes": sd1.ORIGINAL_BYTES,
        },
        "priced_levers": [tail_qat_price, mixed_price],
        "boundaries": [
            "No SegNet, PoseNet, candidate rendering, or contest evaluator ran in this audit.",
            "The AV-DALI edge table decomposes source-target decoder disagreement, not PR130 candidate error.",
            "The tail-QAT d_seg values are checkpoint-recorded on the training cache axis; only its archive bytes were newly materialized here.",
            "The mixed q3/q4 semantic-leg price excludes pose and therefore is not a full-score row.",
        ],
    }
    result_path = out_dir / "SG2_SOURCE_AUDIT_RESULT.json"
    atomic_write_json(result_path, result)
    progress["completed_stages"] = sorted(
        set(progress["completed_stages"]) | {"final_result"}
    )
    progress["completed_at_utc"] = utc_now()
    progress["result"] = artifact(result_path)
    atomic_write_json(resume_path, progress)
    print(json.dumps({"result": artifact(result_path), "progress": artifact(resume_path)}))


if __name__ == "__main__":
    main()
