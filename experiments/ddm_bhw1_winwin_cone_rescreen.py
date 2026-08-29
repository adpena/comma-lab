#!/usr/bin/env python3
"""Re-screen retained trade-space families with fcd1's exact B/H/W classifier.

This arm does not invent a coding argmax.  ``prepare`` inventories the retained
family objects and screens only LD1, whose model stayed byte-identical to DX2
and whose exact final coding-argmax field is already retained.  Families whose
own coding argmax was never persisted remain reduced-scope inventory rows with
an explicit re-derivation cost.

The largest retained LD1 B pool is materialized as a complete edited field and
an edit-plane payload.  Its LD1-induced subset (cells absent from the DX2 B
pool) is retained separately so an inherited fcd1 opportunity cannot be passed
off as family-specific evidence.  The real byte verdict is delegated to
``ddm_jg2_tail_reencode.py``; ``summarize`` refuses unless that instrument first
reproduces the family-base stream byte-identically and then emits a trustworthy
full-n600 joint re-encode.  Every payload is retained under APDataStore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_fcd1_field_for_coder_diagonal import (
    classify_pool,
    write_candidate_payload,
)
from tac.candidate_seal import CONSISTENT, check_pin_consistency, repin_receiver

AP_ROOT = Path("/Volumes/APDataStore/pact")
STORE = AP_ROOT / "ddm_bhw1_winwin_cone_rescreen"
LOCAL_RECEIPTS = REPO / ".omx/tmp/arm_receipts_local"

N, H, W = 600, 384, 512
POSITIONS = N * H * W
AXIS = "[macOS-CPU frozen-scorer advisory]"
MEASUREMENT_SURFACE = "scorer-free exact B/H/W labels and real joint re-encode bytes"
S_PER_BYTE = 25.0 / 37_545_489.0
LAW_MINIMUM_B_SHARE = 0.001
MINIMUM_FREE_BYTES = 8 << 30
GB1_BYTES = 180_215

TOKENS = (
    AP_ROOT
    / "ddm_tb2_token_bit_attribution/measurement_v1/retained/fields/decoded_tokens_instrumented.u8"
)
CODING_ARGMAX = (
    AP_ROOT
    / "ddm_df1_dddb_field/measurement_v1/retained/fields/position_coding_argmax.u8.bin"
)
GT = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")
DX2_RUNTIME = AP_ROOT / "ddm_dx2/r7/candidate_runtime_dx2"

LD1 = LOCAL_RECEIPTS / "ddm_ld1_lane_lossy_drop_exchange/measurement_v1"
LD1_RATE_CURVE = LD1 / "RATE_CURVE.json"
LD1_TAGS = (
    "lane2road_topcost_k002500",
    "lane2road_topcost_k005000",
    "lane2road_topcost_k010000",
    "lane2road_topcost_k020000",
    "lane2road_topcost_k040000",
    "lane2road_topcost_k060000",
)

JF2 = AP_ROOT / "ddm_jf2_terminal_diagonal_harvest/retained"
OE1 = LOCAL_RECEIPTS / "ddm_oe1_online_escape_member"
AE1 = Path("/Volumes/VertigoDataTier/pact/ddm_ae1_anti_predicted_excess/measurement_v2")

PINS = {
    "tokens": (POSITIONS, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "coding_argmax": (POSITIONS, "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e"),
    "gt": (POSITIONS + 128, "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"),
}

DF1_REDERIVATION = {
    "reference_store": str(AP_ROOT / "ddm_df1_dddb_field/measurement_v1"),
    "measured_stage_count": 30,
    "measured_total_elapsed_seconds": 923.5094490868505,
    "measured_total_stage_artifact_bytes": 2_025_181_467,
    "scope": "one n600 final-coding-row trajectory on the shipped DX2 object",
    "extension_required": (
        "adapt experiments/ddm_df1_drop_field.py to the family's actual model, corrector, "
        "and target-field trajectory; its current constants are hardwired to DX2"
    ),
}


class Bhw1Error(RuntimeError):
    """A custody, scope, or real-byte gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def require_pin(path: Path, key: str) -> dict[str, Any]:
    size, digest = PINS[key]
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise Bhw1Error(f"custody pin failed for {key}: {path}")
    return file_fact(path)


def verify_fact(fact: dict[str, Any]) -> dict[str, Any]:
    path = Path(fact["path"])
    current = file_fact(path)
    if current != fact:
        raise Bhw1Error(f"retained object drifted: {path}")
    return current


def storage_preflight(store: Path) -> dict[str, Any]:
    if not store.resolve().is_relative_to(AP_ROOT.resolve()):
        raise Bhw1Error(f"store must remain under {AP_ROOT}: {store}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise Bhw1Error(f"storage preflight: {free} B free < {MINIMUM_FREE_BYTES} B")
    return {
        "path": str(store),
        "free_bytes": free,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "status": "PASS",
    }


def bounded_argmax_files(root: Path) -> list[dict[str, Any]]:
    """Name-based bounded census; semantic suitability is adjudicated separately."""
    return [
        file_fact(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "argmax" in path.name.lower()
    ]


def load_ld1_rows() -> list[dict[str, Any]]:
    rate_curve = json.loads(LD1_RATE_CURVE.read_text())
    rate_by_tag = {row["tag"]: row for row in rate_curve["rows"]}
    rows: list[dict[str, Any]] = []
    for tag in LD1_TAGS:
        rung_path = LD1 / "retained/rungs" / tag / "RUNG.json"
        rung = json.loads(rung_path.read_text())
        rate = rate_by_tag[tag]
        field = verify_fact(rung["candidate_field"])
        archive = verify_fact(rate["candidate_archive"])
        rows.append(
            {
                "tag": tag,
                "tokens_changed": int(rung["tokens_changed"]),
                "field": field,
                "archive": archive,
                "prior_real_delta_bytes_vs_dx2": int(rate["archive_delta_bytes"]),
                "rung_receipt": file_fact(rung_path),
                "rate_receipt": verify_fact(rate["rate_receipt"]),
            }
        )
    return rows


def load_jf2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in sorted(path for path in JF2.iterdir() if path.is_dir()):
        receipt_path = root / "MEASURE_RESULT.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        selected_source = receipt["model"]["winner"]["payload"]
        selected_model = root / "retained/model" / Path(selected_source["path"]).name
        rows.append(
            {
                "tag": root.name,
                "receipt": file_fact(receipt_path),
                "archive": file_fact(root / "retained/candidate_archive.zip"),
                "field": file_fact(root / "retained/decoded_tokens.u8"),
                "selected_model": file_fact(selected_model),
            }
        )
    if len(rows) != 7:
        raise Bhw1Error(f"JF2 retained row census drifted: expected 7, found {len(rows)}")
    return rows


def family_inventory(ld1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jf2_rows = load_jf2_rows()
    oe1_result_path = OE1 / "RESULT.json"
    oe1_result = json.loads(oe1_result_path.read_text())
    ae1_result_path = AE1 / "RESULT.json"
    ae1_result = json.loads(ae1_result_path.read_text())
    predictor_argmax = verify_fact(ae1_result["source_binding"]["sources"]["predictor_argmax"])

    return [
        {
            "family": "dg2 diagonal",
            "objects": [row for row in jf2_rows if row["tag"] in {"k040000", "k060000"}],
            "search_space_verified_at_source": "joint field/model rate-versus-realized-distortion trade",
            "win_win_aware": False,
            "coding_argmax_census": bounded_argmax_files(JF2),
            "rescreenable_at_zero_cost": False,
            "scope": "REDUCED: family-specific final coding argmax was not persisted",
            "rederivation": {
                **DF1_REDERIVATION,
                "objects": 2,
                "projected_elapsed_seconds_at_df1_reference": 2 * DF1_REDERIVATION["measured_total_elapsed_seconds"],
                "projected_stage_artifact_bytes_at_df1_reference": 2 * DF1_REDERIVATION["measured_total_stage_artifact_bytes"],
                "overlap_note": "these two rows are already members of the seven-row JF2 store",
            },
        },
        {
            "family": "jf1/jf2 terminal diagonal",
            "objects": jf2_rows,
            "search_space_verified_at_source": "terminal byte winners followed by same-axis Seg/Pose refusal",
            "win_win_aware": False,
            "coding_argmax_census": bounded_argmax_files(JF2),
            "rescreenable_at_zero_cost": False,
            "scope": "REDUCED: seven refit-model final coding argmax fields were not persisted",
            "rederivation": {
                **DF1_REDERIVATION,
                "objects": 7,
                "projected_elapsed_seconds_at_df1_reference": 7 * DF1_REDERIVATION["measured_total_elapsed_seconds"],
                "projected_stage_artifact_bytes_at_df1_reference": 7 * DF1_REDERIVATION["measured_total_stage_artifact_bytes"],
            },
        },
        {
            "family": "oe1 zero-stored causal escape",
            "objects": [
                {
                    "tag": row["rung"],
                    "member": verify_fact(row["member"]),
                    "stream": verify_fact(row["stream"]),
                    "decoded_tokens": verify_fact(row["decoded_tokens"]),
                }
                for row in oe1_result["rows"]
            ],
            "search_space_verified_at_source": "lossless rate redistribution, not a rate/distortion trade",
            "win_win_aware": False,
            "coding_argmax_census": bounded_argmax_files(OE1),
            "rescreenable_at_zero_cost": False,
            "scope": "REDUCED: the five online-mixture final coding argmax fields were not persisted",
            "rederivation": {
                "mechanism_extension": (
                    "persist each rung's final coding-row argmax inside experiments/ddm_oe1_online_escape_member.py; "
                    "DX2's df1 producer does not contain OE1's online expert"
                ),
                "measured_original_five_rung_elapsed_seconds": float(oe1_result["elapsed_seconds"]),
                "minimum_new_final_argmax_payload_bytes": 5 * POSITIONS,
                "cost_status": "bounded lower cost; implementation review and checkpoint-schema extension still required",
            },
        },
        {
            "family": "ld1 lossy Lane",
            "objects": ld1_rows,
            "search_space_verified_at_source": "explicit rate-versus-token-truth trade over six nested Lane-to-Road fields",
            "win_win_aware": False,
            "coding_argmax": require_pin(CODING_ARGMAX, "coding_argmax"),
            "coding_argmax_census": bounded_argmax_files(LD1),
            "rescreenable_at_zero_cost": True,
            "scope": "FULL for the six registered LD1 fields under their unchanged DX2 model",
            "rederivation": None,
        },
        {
            "family": "ae1 anti-predicted excess",
            "objects": {
                "result": file_fact(ae1_result_path),
                "manifest": file_fact(AE1 / "MANIFEST.json"),
                "predictor_argmax_unsuitable_for_BHW": predictor_argmax,
            },
            "search_space_verified_at_source": "gross-allocation plus signalling/static-overlay rate accounting; no physical static-overlay token coder",
            "win_win_aware": False,
            "coding_argmax_census": bounded_argmax_files(AE1),
            "rescreenable_at_zero_cost": False,
            "scope": (
                "REDUCED: FS2 predictor argmax is pre-corrector and is not AE1's final coding argmax; "
                "the two static overlays never produced a finite RC64 object"
            ),
            "rederivation": {
                "cost_status": "not honestly costable as replay-only",
                "required_first": "build a physical receiver/coder for each retained static formulation, then persist its final coding argmax",
                "mechanism_extension": True,
            },
        },
    ]


def classify_ld1_row(row: dict[str, Any], retained: Path) -> dict[str, Any]:
    tokens = np.memmap(Path(row["field"]["path"]), dtype=np.uint8, mode="r", shape=(N, H, W))
    argmax = np.memmap(CODING_ARGMAX, dtype=np.uint8, mode="r", shape=(N, H, W))
    gt = np.load(GT, mmap_mode="r", allow_pickle=False)
    totals: Counter[str] = Counter()
    coordinate_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    for frame in range(N):
        classes = classify_pool(tokens[frame], argmax[frame], gt[frame])
        for name, mask in classes.items():
            totals[name] += int(np.count_nonzero(mask))
        yy, xx = np.nonzero(classes["benefit"])
        if yy.size:
            coordinate_parts.append(
                np.stack(
                    [np.full(yy.size, frame, dtype=np.int32), yy.astype(np.int32), xx.astype(np.int32)],
                    axis=1,
                )
            )
            old_parts.append(np.asarray(tokens[frame][yy, xx], dtype=np.uint8))
            new_parts.append(np.asarray(argmax[frame][yy, xx], dtype=np.uint8))
    coords = np.concatenate(coordinate_parts, axis=0) if coordinate_parts else np.empty((0, 3), dtype=np.int32)
    old = np.concatenate(old_parts) if old_parts else np.empty(0, dtype=np.uint8)
    new = np.concatenate(new_parts) if new_parts else np.empty(0, dtype=np.uint8)
    pool = retained / "coordinates" / f"{row['tag']}.benefit.frame_y_x_old_new.npz"
    atomic_npz(pool, coords=coords, old=old, new=new)
    disagreement = int(totals["benefit"] + totals["harm"] + totals["wash"])
    return {
        **row,
        "B_benefit": int(totals["benefit"]),
        "H_harm": int(totals["harm"]),
        "W_wash": int(totals["wash"]),
        "disagreement_positions": disagreement,
        "B_share_of_disagreement": float(totals["benefit"] / disagreement) if disagreement else None,
        "benefit_pool": file_fact(pool),
        "_coords": coords,
        "_old": old,
        "_new": new,
        "_tokens": tokens,
    }


def stage_runtime(source: Path, archive: Path, destination: Path) -> dict[str, Any]:
    """Stage one retained family archive without mutating its source runtime."""
    if destination.exists():
        verdict = check_pin_consistency(destination)
        if verdict.verdict != CONSISTENT:
            raise Bhw1Error(f"existing staged runtime is not pin-consistent: {verdict.summary()}")
        if sha256_file(destination / "archive.zip") != sha256_file(archive):
            raise Bhw1Error("existing staged runtime binds different archive bytes")
        return {
            "runtime": str(destination),
            "archive": file_fact(destination / "archive.zip"),
            "pin_consistency": verdict.verdict,
            "resumed": True,
        }
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise Bhw1Error(f"partial runtime exists; inspect before retry: {temporary}")
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    shutil.copy2(archive, temporary / "archive.zip")
    repin = repin_receiver(temporary)
    verdict = check_pin_consistency(temporary)
    if verdict.verdict != CONSISTENT:
        raise Bhw1Error(f"staged runtime refused after re-pin: {verdict.summary()}")
    os.replace(temporary, destination)
    return {
        "runtime": str(destination),
        "archive": file_fact(destination / "archive.zip"),
        "pin_consistency": verdict.verdict,
        "repin_changed": repin.changed,
        "resumed": False,
    }


def run_prepare(args: argparse.Namespace) -> int:
    store = Path(args.store)
    preflight = storage_preflight(store)
    source_inputs = {
        "tokens": require_pin(TOKENS, "tokens"),
        "coding_argmax": require_pin(CODING_ARGMAX, "coding_argmax"),
        "gt": require_pin(GT, "gt"),
    }
    gt = np.load(GT, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, H, W) or gt.dtype != np.uint8:
        raise Bhw1Error(f"GT field shape/dtype drifted: {gt.shape} {gt.dtype}")

    ld1_rows = load_ld1_rows()
    inventory = family_inventory(ld1_rows)
    classified = [classify_ld1_row(row, store / "retained") for row in ld1_rows]
    top = max(classified, key=lambda row: int(row["B_benefit"]))
    top_coords = top.pop("_coords")
    top_old = top.pop("_old")
    top_new = top.pop("_new")
    top_tokens = top.pop("_tokens")
    top_payload = write_candidate_payload(
        name=f"{top['tag']}_gt_benefit",
        selected=np.ones(int(top["B_benefit"]), dtype=bool),
        coords=top_coords,
        old=top_old,
        new=top_new,
        tokens=top_tokens,
        retained=store / "retained",
    )
    base_tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))
    base_values = np.asarray(
        base_tokens[top_coords[:, 0], top_coords[:, 1], top_coords[:, 2]],
        dtype=np.uint8,
    )
    ld1_induced = base_values == top_new
    if not np.any(ld1_induced):
        raise Bhw1Error("top LD1 B cone contains no family-induced cells beyond the DX2 base")
    induced_payload = write_candidate_payload(
        name=f"{top['tag']}_ld1_induced_gt_benefit",
        selected=ld1_induced,
        coords=top_coords,
        old=top_old,
        new=top_new,
        tokens=top_tokens,
        retained=store / "retained",
    )
    for row in classified:
        row.pop("_coords", None)
        row.pop("_old", None)
        row.pop("_new", None)
        row.pop("_tokens", None)

    base_archive = Path(top["archive"]["path"])
    staged = stage_runtime(DX2_RUNTIME, base_archive, store / "runtimes" / top["tag"])
    result = {
        "schema": "ddm_bhw1_prepare.v1",
        "axis": AXIS,
        "measurement_surface": MEASUREMENT_SURFACE,
        "score_claim": False,
        "promotable": False,
        "implementation": {
            "driver": file_fact(Path(__file__)),
            "classifier": file_fact(REPO / "experiments/ddm_fcd1_field_for_coder_diagonal.py"),
            "real_joint_reencoder": file_fact(REPO / "experiments/ddm_jg2_tail_reencode.py"),
        },
        "storage_preflight": preflight,
        "source_inputs": source_inputs,
        "inventory": inventory,
        "rescreened_family": "ld1 lossy Lane",
        "classification_rows": classified,
        "top_cone": {
            "family": "ld1 lossy Lane",
            "tag": top["tag"],
            "B_benefit": top["B_benefit"],
            "base_archive": top["archive"],
            "base_field": top["field"],
            "gt_benefit_payload": top_payload,
            "DX2_inherited_B": int(top["B_benefit"] - np.count_nonzero(ld1_induced)),
            "LD1_induced_B": int(np.count_nonzero(ld1_induced)),
            "LD1_induced_B_share_of_disagreement": float(
                np.count_nonzero(ld1_induced) / int(top["disagreement_positions"])
            ),
            "ld1_induced_gt_benefit_payload": induced_payload,
            "staged_runtime": staged,
        },
        "next_stage": (
            "run ddm_jg2 control against the staged family base, then encode the retained "
            "full and LD1-induced GT-benefit edit payloads, all n600 with checkpointing and resume"
        ),
    }
    atomic_json(store / "PREPARE.json", result)
    print(
        json.dumps(
            {
                "rescreened_family": result["rescreened_family"],
                "rows": [
                    {"tag": row["tag"], "B": row["B_benefit"], "H": row["H_harm"], "W": row["W_wash"]}
                    for row in classified
                ],
                "top": {
                    "tag": top["tag"],
                    "B": top["B_benefit"],
                    "DX2_inherited_B": result["top_cone"]["DX2_inherited_B"],
                    "LD1_induced_B": result["top_cone"]["LD1_induced_B"],
                    "LD1_induced_B_share_of_disagreement": result["top_cone"][
                        "LD1_induced_B_share_of_disagreement"
                    ],
                },
            },
            indent=2,
        )
    )
    return 0


def store_manifest(store: Path) -> dict[str, Any]:
    rows = [
        file_fact(path)
        for path in sorted(store.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json" and not path.name.endswith(".partial")
    ]
    return {
        "schema": "ddm_bhw1_manifest.v1",
        "root": str(store),
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["bytes"]) for row in rows),
        "artifacts": rows,
    }


def run_summarize(args: argparse.Namespace) -> int:
    store = Path(args.store)
    storage_preflight(store)
    prepare_path = store / "PREPARE.json"
    if not prepare_path.is_file():
        raise Bhw1Error("PREPARE.json missing")
    prepare = json.loads(prepare_path.read_text())
    top = prepare["top_cone"]
    full_tag = f"bhw1_{top['tag']}_gt_benefit"
    induced_tag = f"bhw1_{top['tag']}_ld1_induced_gt_benefit"
    receipts = store / "reencode/retained"
    control_path = receipts / "S1_control_600.json"
    full_encode_path = receipts / f"S1_encode_{full_tag}.json"
    induced_encode_path = receipts / f"S1_encode_{induced_tag}.json"
    if not control_path.is_file() or not full_encode_path.is_file() or not induced_encode_path.is_file():
        raise Bhw1Error("real re-encode receipts are incomplete")
    control = json.loads(control_path.read_text())
    if not control.get("byte_identical") or int(control.get("frames", 0)) != N:
        raise Bhw1Error("family-base n600 inverse-coder control is not byte-identical")
    base = verify_fact(top["base_archive"])

    def validate_encode(path: Path, expected_edits: int) -> dict[str, Any]:
        encode = json.loads(path.read_text())
        if not encode.get("delta_trustworthy") or int(encode.get("frames", 0)) != N:
            raise Bhw1Error(f"candidate real re-encode is not trustworthy n600: {path}")
        if encode.get("pointer_archive") != top["base_archive"]:
            raise Bhw1Error(f"candidate re-encode did not use the retained family base: {path}")
        if int(encode.get("tokens_changed", -1)) != expected_edits:
            raise Bhw1Error(f"candidate edit count drifted from its exact B pool: {path}")
        candidate = verify_fact(encode["candidate_archive"])
        delta = int(candidate["bytes"] - base["bytes"])
        if delta != int(encode["archive_delta_bytes"]):
            raise Bhw1Error(f"candidate archive delta disagrees with retained archive stats: {path}")
        return {
            "B_benefit": expected_edits,
            "H_harm": 0,
            "W_wash": 0,
            "encode_receipt": file_fact(path),
            "candidate_archive": candidate,
            "real_marginal_bytes_vs_family_base": delta,
            "real_marginal_bits_per_edit": 8.0 * delta / expected_edits,
            "real_delta_S_rate_vs_family_base": delta * S_PER_BYTE,
            "real_marginal_bytes_vs_gb1": int(candidate["bytes"] - GB1_BYTES),
        }

    full = validate_encode(full_encode_path, int(top["B_benefit"]))
    induced = validate_encode(induced_encode_path, int(top["LD1_induced_B"]))
    family_specific_negative_rate = int(induced["real_marginal_bytes_vs_family_base"]) < 0
    family_specific_mass = float(top["LD1_induced_B_share_of_disagreement"])
    prediction_confirmed = family_specific_negative_rate and family_specific_mass >= LAW_MINIMUM_B_SHARE
    full_rate_opening = int(full["real_marginal_bytes_vs_family_base"]) < 0
    result = {
        "schema": "ddm_bhw1_real_reencode_result.v1",
        "axis": AXIS,
        "measurement_surface": MEASUREMENT_SURFACE,
        "score_claim": False,
        "promotable": False,
        "family": "ld1 lossy Lane",
        "family_base_tag": top["tag"],
        "control": file_fact(control_path),
        "family_base_archive": base,
        "gb1_archive_bytes": GB1_BYTES,
        "full_top_cone_including_DX2_inherited_cells": full,
        "LD1_induced_cone_only": induced,
        "law_test": {
            "minimum_B_share_of_disagreement": LAW_MINIMUM_B_SHARE,
            "observed_LD1_induced_B_share_of_disagreement": family_specific_mass,
            "negative_real_marginal_bytes": family_specific_negative_rate,
            "prediction_confirmed": prediction_confirmed,
        },
        "distortion_legs": {
            "realized_d_seg": None,
            "realized_d_pose": None,
            "status": "UNMEASURED_NO_SCORER_LANE_OWNERSHIP",
        },
        "verdict": (
            "INHERITED_FULL_CONE_RATE_OPENING_DISTORTION_UNMEASURED_LAW_PREDICTION_FALSIFIED_ON_MASS"
            if full_rate_opening
            else "NO_FULL_CONE_RATE_OPENING_LAW_PREDICTION_FALSIFIED_ON_MASS"
        ),
        "law_scope": (
            "confirmed beyond fcd1 at the charter's >=0.1% mass threshold"
            if prediction_confirmed
            else "falsified at the only zero-cost re-screenable family: LD1-induced B is below 0.1%; "
            "the full cone is separate because it mostly inherits DX2 cells"
        ),
    }
    atomic_json(store / "REAL_REENCODE_RESULT.json", result)
    atomic_json(store / "MANIFEST.json", store_manifest(store))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=STORE)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare").set_defaults(func=run_prepare)
    sub.add_parser("summarize").set_defaults(func=run_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
