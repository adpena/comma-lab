#!/usr/bin/env python3
"""Retained counted-mask oracle upper bound for MF1 native-stage repairs.

This scope-reduced instrument reuses the byte-identical, verified n32 baseline
payloads from MF1 measurement v2 and the full-n600 localization/mask payloads
from measurement v3. It materializes only three new candidate families, each
editing the exact native-stage manufactured support and paying at least the
real Brotli-q11 mask length. Every candidate field is retained before metrics
are assembled. No receiver, archive, upstream file, Modal lane, or Metal lane
is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import ddm_mf1_manufactured_seg_repair as mf1
import numpy as np

LOCALIZATION_ROOT = Path("/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/measurement_v3")
BASELINE_ROOT = Path("/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/measurement_v2")
DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/oracle_native_v1")


def source_facts() -> dict[str, Any]:
    localization_manifest = json.loads((LOCALIZATION_ROOT / "LOCALIZATION_MANIFEST.json").read_text())
    facts = mf1.source_facts(include_probe=True)
    facts.update(
        {
            "oracle_instrument": mf1.file_fact(Path(__file__).resolve()),
            "localization": mf1.file_fact(LOCALIZATION_ROOT / "LOCALIZATION.json"),
            "localization_manifest": mf1.file_fact(LOCALIZATION_ROOT / "LOCALIZATION_MANIFEST.json"),
            "candidate_mask_raw": mf1.file_fact(LOCALIZATION_ROOT / "retained/native_manufactured_mask.n600.packbits"),
            "candidate_mask_brotli": mf1.file_fact(
                LOCALIZATION_ROOT / "retained/native_manufactured_mask.n600.brotli_q11"
            ),
            "candidate_mask_brotli_repeat": mf1.file_fact(
                LOCALIZATION_ROOT / "retained/native_manufactured_mask.n600.repeat.brotli_q11"
            ),
            "prototype": mf1.file_fact(
                LOCALIZATION_ROOT / "retained/decoder_derived_prototypes.float32.npy",
                expected_sha256=localization_manifest["prototype_payload"]["sha256"],
            ),
            "selection": mf1.file_fact(
                LOCALIZATION_ROOT / "retained/stratified_random_n32_selection.npz",
                expected_sha256=localization_manifest["selection_payload"]["sha256"],
            ),
            "baseline_result": mf1.file_fact(BASELINE_ROOT / "PROBE_RESULT.json"),
            "baseline_manifest": mf1.file_fact(BASELINE_ROOT / "PROBE_MANIFEST.json"),
        }
    )
    return facts


def load_selection() -> np.ndarray:
    with np.load(LOCALIZATION_ROOT / "retained/stratified_random_n32_selection.npz") as selection:
        pairs = selection["pair_indices"].astype(np.int64)
    if pairs.size != 32 or np.unique(pairs).size != 32:
        raise mf1.Mf1Error("oracle selection is not 32 unique pairs")
    return pairs


def baseline_payloads(pairs: np.ndarray) -> list[dict[str, Any]]:
    manifest = json.loads((BASELINE_ROOT / "PROBE_MANIFEST.json").read_text())
    result = json.loads((BASELINE_ROOT / "PROBE_RESULT.json").read_text())
    if result["pair_indices"] != pairs.tolist():
        raise mf1.Mf1Error("reused baseline selection differs from oracle selection")
    expected = {Path(fact["path"]).name: fact for fact in manifest["baseline_payloads"]}
    rows = []
    for pair in pairs:
        path = BASELINE_ROOT / f"probe_n32/baseline/pairs/pair_{int(pair):04d}.npz"
        fact = expected.get(path.name)
        if fact is None:
            raise mf1.Mf1Error(f"baseline manifest has no row for pair {int(pair)}")
        rows.append(mf1.file_fact(path, expected_sha256=fact["sha256"]))
    return rows


def run_probe(store: Path, *, resume_from: Path | None = None) -> None:
    mf1.storage_preflight(store, required_free_bytes=2_000_000_000)
    mf1.recover_incomplete_temps(store)
    facts = source_facts()
    pairs = load_selection()
    baseline_facts = baseline_payloads(pairs)
    localization = json.loads((LOCALIZATION_ROOT / "LOCALIZATION.json").read_text())
    payload_fact = localization["candidate_family"]["payload"]["compressed"]
    payload_bytes = int(payload_fact["bytes"])
    if payload_fact["sha256"] != facts["candidate_mask_brotli"]["sha256"]:
        raise mf1.Mf1Error("candidate mask localization/source binding drifted")
    if facts["candidate_mask_brotli"]["sha256"] != facts["candidate_mask_brotli_repeat"]["sha256"]:
        raise mf1.Mf1Error("candidate mask deterministic repeat drifted")

    binding = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    checkpoint_path = resume_from if resume_from is not None else store / "ORACLE_CHECKPOINT.json"
    if resume_from is not None and not checkpoint_path.is_file():
        raise mf1.Mf1Error(f"explicit oracle resume checkpoint is absent: {checkpoint_path}")
    checkpoint_path = mf1.require_apdatastore_path(checkpoint_path, label="oracle checkpoint")
    if checkpoint_path.parent != store.resolve():
        raise mf1.Mf1Error("oracle --resume-from must name the checkpoint inside --store")
    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.is_file()
        else {
            "schema": "ddm_mf1.oracle_checkpoint.v1",
            "source_binding_sha256": binding,
            "candidates": {},
        }
    )
    if checkpoint.get("source_binding_sha256") != binding:
        raise mf1.Mf1Error("oracle checkpoint source binding drifted")

    prototypes = np.load(LOCALIZATION_ROOT / "retained/decoder_derived_prototypes.float32.npy")
    tokens = np.memmap(
        mf1.MST1 / "retained/inputs/tokens_cpu_stage_complete.u8",
        dtype=np.uint8,
        mode="r",
        shape=(mf1.N, mf1.H, mf1.W),
    )
    gt = np.load(mf1.MST1 / "retained/inputs/gt_argmax_n600.npy", mmap_mode="r")
    terminal = np.load(mf1.MST1 / "retained/inputs/cuda_terminal_argmax_n600.npy", mmap_mode="r")
    gt_pose6 = np.load(mf1.GT_POSE6, mmap_mode="r")
    raw = np.memmap(
        mf1.CPU_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(mf1.N * 2, mf1.CAM_H, mf1.CAM_W, 3),
    )
    native_manufactured = mf1.load_packbits(
        mf1.MST1 / "retained/attribution_masks/earliest_manufactured_native_render_head.n600.packbits"
    )
    final_manufactured = (np.asarray(terminal) != np.asarray(gt)) & (np.asarray(tokens) == np.asarray(gt))
    torch, segnet, posenet = mf1.load_scorers()

    candidate_root = store / "candidates"
    for tag, alpha in mf1.PROBE_CANDIDATES:
        admitted = {int(row["pair"]) for row in checkpoint["candidates"].get(tag, [])}
        expected = {int(row["pair"]): row["payload"]["sha256"] for row in checkpoint["candidates"].get(tag, [])}
        for pair_value in pairs:
            pair = int(pair_value)
            output = candidate_root / tag / "pairs" / f"pair_{pair:04d}.npz"
            if output.is_file():
                mf1.existing_pair_fact(output, expected_sha256=expected.get(pair))
                if pair not in admitted:
                    checkpoint["candidates"].setdefault(tag, []).append(
                        {"pair": pair, "payload": mf1.file_fact(output)}
                    )
                    mf1.atomic_json(checkpoint_path, checkpoint)
                continue
            _, _, offset = mf1.chunk_bounds(pair)
            native = np.asarray(np.load(mf1.chunk_dir(pair) / "native_rgb.float32.npy", mmap_mode="r")[offset]).astype(
                np.float32, copy=True
            )
            fields = mf1.candidate_fields(
                torch,
                segnet,
                posenet,
                native_chw=native,
                tokens=np.asarray(tokens[pair]),
                slave_hwc=np.asarray(raw[2 * pair]),
                global_prototypes=prototypes,
                edit_mask=native_manufactured[pair],
                alpha=alpha,
            )
            fact = mf1.atomic_npz(output, **fields)
            checkpoint["candidates"].setdefault(tag, []).append({"pair": pair, "payload": fact})
            mf1.atomic_json(checkpoint_path, checkpoint)
            print(f"{tag} retained pair {pair}", flush=True)

    baseline_argmax = []
    baseline_pose6 = []
    for pair in pairs:
        path = BASELINE_ROOT / f"probe_n32/baseline/pairs/pair_{int(pair):04d}.npz"
        with np.load(path, allow_pickle=False) as values:
            baseline_argmax.append(values["segnet_argmax_uint8"].copy())
            baseline_pose6.append(values["pose6_float32"].copy())
    baseline_argmax_array = np.stack(baseline_argmax)
    baseline_pose6_array = np.stack(baseline_pose6)
    subset_gt = np.asarray(gt[pairs])
    subset_tokens = np.asarray(tokens[pairs])
    subset_gt_pose6 = np.asarray(gt_pose6[pairs])
    subset_final_manufactured = final_manufactured[pairs]
    subset_native_manufactured = native_manufactured[pairs]
    baseline_error = baseline_argmax_array != subset_gt
    baseline_dseg = float(baseline_error.mean())
    baseline_dpose = float(np.mean((baseline_pose6_array - subset_gt_pose6) ** 2))
    baseline = {
        "d_seg": baseline_dseg,
        "d_pose": baseline_dpose,
        "seg_errors": int(baseline_error.sum()),
        "cpu_manufactured_errors": int((baseline_error & (subset_tokens == subset_gt)).sum()),
        "t4_final_manufactured_support": int(subset_final_manufactured.sum()),
        "native_manufactured_support": int(subset_native_manufactured.sum()),
        "native_manufactured_still_wrong_cpu": int((subset_native_manufactured & baseline_error).sum()),
        "pose_contribution": math.sqrt(10.0 * baseline_dpose),
    }

    candidate_rows = []
    for tag, alpha in mf1.PROBE_CANDIDATES:
        candidate_argmax = []
        candidate_pose6 = []
        candidate_payloads = []
        for pair in pairs:
            path = candidate_root / tag / "pairs" / f"pair_{int(pair):04d}.npz"
            candidate_payloads.append(mf1.file_fact(path))
            with np.load(path, allow_pickle=False) as values:
                candidate_argmax.append(values["segnet_argmax_uint8"].copy())
                candidate_pose6.append(values["pose6_float32"].copy())
        argmax = np.stack(candidate_argmax)
        pose6 = np.stack(candidate_pose6)
        error = argmax != subset_gt
        dseg = float(error.mean())
        dpose = float(np.mean((pose6 - subset_gt_pose6) ** 2))
        fixed = baseline_error & ~error
        introduced = ~baseline_error & error
        native_baseline_wrong = subset_native_manufactured & baseline_error
        native_fixed = native_baseline_wrong & ~error
        native_persisting = native_baseline_wrong & error
        per_class = []
        for class_index, class_name in enumerate(mf1.CLASSES):
            support = subset_gt == class_index
            per_class.append(
                {
                    "class": class_name,
                    "support_pixels": int(support.sum()),
                    "baseline_errors": int((baseline_error & support).sum()),
                    "candidate_errors": int((error & support).sum()),
                    "fixed": int((fixed & support).sum()),
                    "introduced": int((introduced & support).sum()),
                    "native_manufactured_fixed": int((native_fixed & support).sum()),
                }
            )
        seg_delta_s = 100.0 * (dseg - baseline_dseg)
        pose_delta_s = math.sqrt(10.0 * dpose) - math.sqrt(10.0 * baseline_dpose)
        rate_delta_s_lower_bound = mf1.RATE_S_PER_BYTE * payload_bytes
        candidate_rows.append(
            {
                "tag": tag,
                "alpha": alpha,
                "d_seg": dseg,
                "delta_d_seg": dseg - baseline_dseg,
                "seg_delta_s": seg_delta_s,
                "d_pose": dpose,
                "delta_d_pose": dpose - baseline_dpose,
                "pose_delta_s": pose_delta_s,
                "distortion_delta_s": seg_delta_s + pose_delta_s,
                "compressed_payload_bytes": payload_bytes,
                "archive_byte_delta": "UNMEASURED: receiver/container not integrated",
                "rate_delta_s_lower_bound": rate_delta_s_lower_bound,
                "joint_delta_s_lower_bound": seg_delta_s + pose_delta_s + rate_delta_s_lower_bound,
                "errors_fixed": int(fixed.sum()),
                "errors_introduced": int(introduced.sum()),
                "net_error_delta": int(error.sum() - baseline_error.sum()),
                "native_manufactured_fixed": int(native_fixed.sum()),
                "native_manufactured_persisting": int(native_persisting.sum()),
                "per_class": per_class,
                "payloads": candidate_payloads,
            }
        )

    result = {
        "schema": "ddm_mf1.oracle_native_result.v1",
        "axis": mf1.AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "verdict_population": "seeded stratified-random n32 subset; exact native-stage manufactured support only",
        "pair_indices": pairs.tolist(),
        "source_facts": facts,
        "baseline": baseline,
        "candidates": candidate_rows,
        "byte_cost": {
            "compressed_payload_bytes": payload_bytes,
            "archive_delta_bytes": "UNMEASURED: receiver/container not integrated",
            "interpretation": "real full-n600 Brotli-q11 address payload is a strict lower bound before container overhead",
        },
        "authority_boundary": (
            "Real R, uint8, frozen CPU SegNet and PoseNet on n32. No n600 scorer row, "
            "candidate archive, contest score, or pointer move."
        ),
    }
    result_fact = mf1.atomic_json(store / "ORACLE_RESULT.json", result)
    manifest = {
        "schema": "ddm_mf1.oracle_manifest.v1",
        "result": result_fact,
        "checkpoint": mf1.file_fact(checkpoint_path),
        "baseline_payloads": baseline_facts,
        "candidate_payload_count": len(mf1.PROBE_CANDIDATES) * len(pairs),
        "complete": True,
    }
    mf1.atomic_json(store / "ORACLE_MANIFEST.json", manifest)


def verify(store: Path) -> None:
    manifest = json.loads((store / "ORACLE_MANIFEST.json").read_text())
    mf1.file_fact(Path(manifest["result"]["path"]), expected_sha256=manifest["result"]["sha256"])
    for fact in manifest["baseline_payloads"]:
        mf1.file_fact(Path(fact["path"]), expected_sha256=fact["sha256"])
    checkpoint_path = Path(manifest["checkpoint"]["path"])
    mf1.file_fact(checkpoint_path, expected_sha256=manifest["checkpoint"]["sha256"])
    checkpoint = json.loads(checkpoint_path.read_text())
    expected_tags = {tag for tag, _ in mf1.PROBE_CANDIDATES}
    if set(checkpoint["candidates"]) != expected_tags:
        raise mf1.Mf1Error("oracle checkpoint candidate tags drifted")
    if any(len(rows) != 32 for rows in checkpoint["candidates"].values()):
        raise mf1.Mf1Error("oracle checkpoint has an incomplete candidate")
    keys = [(tag, int(row["pair"])) for tag, rows in checkpoint["candidates"].items() for row in rows]
    if len(keys) != 96 or len(set(keys)) != 96:
        raise mf1.Mf1Error("oracle checkpoint is not 3 x 32 unique candidate payloads")
    for rows in checkpoint["candidates"].values():
        for row in rows:
            mf1.file_fact(Path(row["payload"]["path"]), expected_sha256=row["payload"]["sha256"])
    receipt = {
        "schema": "ddm_mf1.oracle_verification.v1",
        "status": "COMPLETE",
        "result_sha256": manifest["result"]["sha256"],
        "manifest_sha256": mf1.sha256_file(store / "ORACLE_MANIFEST.json"),
        "candidate_payloads_verified": 96,
    }
    mf1.atomic_json(store / "ORACLE_COMPLETED_VERIFICATION.json", receipt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("probe", "verify"))
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Darwin":
        raise mf1.Mf1Error("this charter authorizes the macOS CPU advisory lane only")
    if args.stage == "probe":
        run_probe(args.store, resume_from=args.resume_from)
    else:
        if args.resume_from is not None:
            raise mf1.Mf1Error("verify discovers its checkpoint from the manifest")
        verify(args.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
