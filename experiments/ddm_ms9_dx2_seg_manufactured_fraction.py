#!/usr/bin/env python3
"""Scorer-free n600 DX2 manufactured-Seg field replay.

This instrument does not run SegNet.  It joins three already-retained exact
fields: contest-CUDA DALI GT, the decoded semantic labels, and a retained
contest-CUDA component-only argmax field.  The latter belongs to FX5, but the
FX5 and DX2 contest-CUDA receipts bind their inflated 0.raw payloads to the
same SHA-256, so the argmax field is also an exact field replay for DX2.

Every measured support mask is retained as a packed-bit payload.  The run is
stage-resumable: completed masks are hash-checked from CHECKPOINT.json and each
new payload is written atomically before the checkpoint advances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SHAPE = (600, 384, 512)
N_PIXELS = int(np.prod(SHAPE))
RATE_DENOMINATOR_BYTES = 37_545_489
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
MIN_FREE_BYTES = 2 << 30
REPO_ROOT = Path(__file__).resolve().parents[1]
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")

DX2_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/archive.zip")
DX2_RECEIPT = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1/MODAL_REMOTE_RESULT.json")
FX5_RECEIPT = Path("/Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/MODAL_REMOTE_RESULT.json")
GT_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")
LABEL_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r2/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
ARGMAX_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/payloads_r8/fx5_e1_argmax_n600.npy"
)
ARGMAX_PROVENANCE = REPO_ROOT / (
    ".omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8_preflight/FIRE_ORDER.json"
)
RT1_ARGMAX_FIELD = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/argmax_base.npy")
RT1_LABEL_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/"
    "coders/s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction")

EXPECTED = {
    "dx2_archive_sha256": ("976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"),
    "dx2_archive_bytes": 180_368,
    "fx5_archive_sha256": ("4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"),
    "d_seg_report_8dp": 0.00020139,
    "d_pose_report_8dp": 0.00000637,
    "score_recomputed": 0.14821987563243377,
    "t4_raw_sha256": ("6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883"),
    "t4_raw_bytes": 3_662_409_600,
    "gt_sha256": ("91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"),
    "label_sha256": ("cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "argmax_sha256": ("e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34"),
    "rt1_argmax_sha256": ("2aeb1e6be0f7c6ab8191b790204d8df0ae5fdce7ef2ecc5b2d18a715f1a674c4"),
    "rt1_label_sha256": ("9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"),
}


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def decode_json_maybe(value: Any) -> Any:
    while isinstance(value, str):
        value = json.loads(value)
    return value


def load_receipt(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return value


def nested_artifact(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    value = decode_json_maybe(receipt["artifacts"][name])
    if not isinstance(value, dict):
        raise ValueError(f"artifact {name!r} is not an object")
    return value


def inflated_raw_fact(receipt: dict[str, Any]) -> dict[str, Any]:
    artifact = nested_artifact(receipt, "inflated_outputs_manifest.json")
    payload = artifact.get("payload", artifact)
    files = payload["files"]
    if len(files) != 1 or files[0]["relative_path"] != "0.raw":
        raise ValueError("expected one 0.raw entry in inflated manifest")
    return files[0]


def inflate_report(receipt: dict[str, Any]) -> dict[str, Any]:
    stdout = receipt["artifacts"]["contest_auth_eval.stdout.log"]
    if not isinstance(stdout, str):
        raise ValueError("contest-auth stdout artifact is not text")
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("{") and '"schema": "ddm_f26p_inflate_report.v1"' in line:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    if len(rows) != 1:
        raise ValueError("expected one ddm_f26p_inflate_report.v1 row")
    return rows[0]


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"byte mismatch for {path}: {size} != {expected_bytes}")
    sha = sha256_file(path)
    if sha != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {path}: {sha} != {expected_sha}")
    return {"path": str(path), "bytes": size, "sha256": sha}


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = np.load(GT_FIELD, mmap_mode="r")
    labels = np.memmap(LABEL_FIELD, dtype=np.uint8, mode="r", shape=SHAPE)
    argmax = np.load(ARGMAX_FIELD, mmap_mode="r")
    rt1_argmax = np.load(RT1_ARGMAX_FIELD, mmap_mode="r")
    rt1_labels = np.memmap(RT1_LABEL_FIELD, dtype=np.uint8, mode="r", shape=SHAPE)
    for name, field in (
        ("gt", gt),
        ("labels", labels),
        ("argmax", argmax),
        ("rt1_argmax", rt1_argmax),
        ("rt1_labels", rt1_labels),
    ):
        if tuple(field.shape) != SHAPE or field.dtype != np.uint8:
            raise ValueError(f"{name} has {field.shape}/{field.dtype}, expected {SHAPE}/uint8")
    return gt, labels, argmax, rt1_argmax, rt1_labels


MaskFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def mask_specs() -> dict[str, MaskFn]:
    base: dict[str, MaskFn] = {
        "final_error": lambda gt, labels, argmax: argmax != gt,
        "representation_error": lambda gt, labels, argmax: labels != gt,
        "manufactured_final_error": lambda gt, labels, argmax: (argmax != gt) & (labels == gt),
        "representation_survived_final_error": lambda gt, labels, argmax: (argmax != gt) & (labels != gt),
        "representation_corrected": lambda gt, labels, argmax: (argmax == gt) & (labels != gt),
        "final_changed_from_label": lambda gt, labels, argmax: argmax != labels,
        "wrong_to_wrong_change": lambda gt, labels, argmax: (argmax != gt) & (labels != gt) & (argmax != labels),
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        stem = class_name.lower()
        base[f"class_{class_id}_{stem}_final_error"] = lambda gt, labels, argmax, c=class_id: (gt == c) & (argmax != gt)
        base[f"class_{class_id}_{stem}_representation_error"] = lambda gt, labels, argmax, c=class_id: (
            (gt == c) & (labels != gt)
        )
        base[f"class_{class_id}_{stem}_manufactured_final_error"] = lambda gt, labels, argmax, c=class_id: (
            (gt == c) & (argmax != gt) & (labels == gt)
        )
        base[f"class_{class_id}_{stem}_representation_survived_final_error"] = lambda gt, labels, argmax, c=class_id: (
            (gt == c) & (argmax != gt) & (labels != gt)
        )
        base[f"class_{class_id}_{stem}_gt_support"] = lambda gt, labels, argmax, c=class_id: gt == c
        base[f"class_{class_id}_{stem}_label_support"] = lambda gt, labels, argmax, c=class_id: labels == c
        base[f"class_{class_id}_{stem}_argmax_support"] = lambda gt, labels, argmax, c=class_id: argmax == c
    return base


def rt1_mask_specs() -> dict[str, MaskFn]:
    return {
        "rt1_positive_control_final_error": lambda gt, labels, argmax: argmax != gt,
        "rt1_positive_control_representation_error": lambda gt, labels, argmax: labels != gt,
        "rt1_positive_control_argmax_changed_from_label": lambda gt, labels, argmax: argmax != labels,
    }


def write_packed_mask(
    path: Path,
    fn: MaskFn,
    gt: np.ndarray,
    labels: np.ndarray,
    argmax: np.ndarray,
    chunk_pairs: int,
) -> int:
    tmp = path.with_name(path.name + ".partial")
    count = 0
    with tmp.open("wb") as handle:
        for start in range(0, SHAPE[0], chunk_pairs):
            stop = min(start + chunk_pairs, SHAPE[0])
            mask = fn(gt[start:stop], labels[start:stop], argmax[start:stop])
            if mask.dtype != np.bool_:
                raise TypeError(f"mask {path.name} is not bool")
            count += int(np.count_nonzero(mask))
            packed = np.packbits(mask.reshape(-1), bitorder="little")
            handle.write(packed.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return count


def resume_or_materialize_masks(
    out: Path,
    binding: dict[str, Any],
    gt: np.ndarray,
    labels: np.ndarray,
    argmax: np.ndarray,
    rt1_labels: np.ndarray,
    rt1_argmax: np.ndarray,
    chunk_pairs: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    checkpoint_path = out / "CHECKPOINT.json"
    checkpoint: dict[str, Any] = {
        "schema": "ddm_ms9_field_replay_checkpoint_v1",
        "source_binding": binding,
        "completed": {},
    }
    if checkpoint_path.exists():
        loaded = load_receipt(checkpoint_path)
        if loaded.get("source_binding") != binding:
            raise ValueError("existing checkpoint is bound to different source fields")
        checkpoint = loaded

    masks_dir = out / "retained" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    manifests: list[dict[str, Any]] = []
    packed_bytes = (N_PIXELS + 7) // 8

    grouped_specs = (
        (mask_specs(), labels, argmax),
        (rt1_mask_specs(), rt1_labels, rt1_argmax),
    )
    for specs, group_labels, group_argmax in grouped_specs:
        for name, fn in specs.items():
            path = masks_dir / f"{name}.n600.packbits"
            done = checkpoint["completed"].get(name)
            if done is not None:
                fact = verify_file(path, done["sha256"], packed_bytes)
                count = int(done["count"])
            else:
                count = write_packed_mask(path, fn, gt, group_labels, group_argmax, chunk_pairs)
                fact = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                if fact["bytes"] != packed_bytes:
                    raise ValueError(f"packed mask has wrong byte count: {path}")
                checkpoint["completed"][name] = {**fact, "count": count}
                atomic_json(checkpoint_path, checkpoint)
            counts[name] = count
            manifests.append(
                {
                    **fact,
                    "name": name,
                    "count_true": count,
                    "format": "numpy.packbits",
                    "bitorder": "little",
                    "logical_shape": list(SHAPE),
                    "logical_dtype": "bool",
                    "padding_bits": packed_bytes * 8 - N_PIXELS,
                }
            )
    return counts, manifests


def iou(field: np.ndarray, gt: np.ndarray, class_id: int, chunk_pairs: int) -> float:
    intersection = 0
    union = 0
    for start in range(0, SHAPE[0], chunk_pairs):
        stop = min(start + chunk_pairs, SHAPE[0])
        pred_c = field[start:stop] == class_id
        gt_c = gt[start:stop] == class_id
        intersection += int(np.count_nonzero(pred_c & gt_c))
        union += int(np.count_nonzero(pred_c | gt_c))
    return intersection / union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-pairs", type=int, default=16)
    args = parser.parse_args()
    if args.chunk_pairs <= 0:
        raise ValueError("--chunk-pairs must be positive")
    args.out = args.out.resolve()
    if not args.out.is_relative_to(VERTIGO_ROOT.resolve()):
        raise ValueError(f"--out must remain on the Vertigo SSD tier: {VERTIGO_ROOT}")
    if shutil.disk_usage(args.out.parent).free < MIN_FREE_BYTES:
        raise RuntimeError(f"Vertigo free space is below the {MIN_FREE_BYTES}-byte preflight floor")
    args.out.mkdir(parents=True, exist_ok=True)

    sources = {
        "dx2_archive": verify_file(
            DX2_ARCHIVE,
            EXPECTED["dx2_archive_sha256"],
            EXPECTED["dx2_archive_bytes"],
        ),
        "gt": verify_file(GT_FIELD, EXPECTED["gt_sha256"], N_PIXELS + 128),
        "labels": verify_file(LABEL_FIELD, EXPECTED["label_sha256"], N_PIXELS),
        "argmax": verify_file(ARGMAX_FIELD, EXPECTED["argmax_sha256"], N_PIXELS + 128),
        "rt1_argmax": verify_file(RT1_ARGMAX_FIELD, EXPECTED["rt1_argmax_sha256"], N_PIXELS + 128),
        "rt1_labels": verify_file(RT1_LABEL_FIELD, EXPECTED["rt1_label_sha256"], N_PIXELS),
        "dx2_receipt": file_fact(DX2_RECEIPT),
        "fx5_receipt": file_fact(FX5_RECEIPT),
        "argmax_provenance": file_fact(ARGMAX_PROVENANCE),
        "instrument": file_fact(Path(__file__).resolve()),
    }
    dx2_receipt = load_receipt(DX2_RECEIPT)
    fx5_receipt = load_receipt(FX5_RECEIPT)
    argmax_provenance = load_receipt(ARGMAX_PROVENANCE)
    argmax_rows = [
        row for row in argmax_provenance["commands"] if row.get("purpose") == "recover_existing_fx5_base_argmax"
    ]
    if len(argmax_rows) != 1:
        raise ValueError("expected one retained FX5 argmax provenance row")
    argmax_binding = argmax_rows[0]["satisfied_by"]
    if (
        argmax_binding["sha256"] != EXPECTED["argmax_sha256"]
        or argmax_binding["bytes"] != N_PIXELS + 128
        or argmax_binding["shape"] != list(SHAPE)
        or argmax_binding["source_object_sha256"] != EXPECTED["fx5_archive_sha256"]
    ):
        raise ValueError("retained FX5 argmax provenance binding drifted")
    frontier_values = {
        "d_seg": dx2_receipt["avg_segnet_dist"],
        "d_pose": dx2_receipt["avg_posenet_dist"],
        "score": dx2_receipt["score_recomputed_from_components"],
    }
    expected_frontier_values = {
        "d_seg": EXPECTED["d_seg_report_8dp"],
        "d_pose": EXPECTED["d_pose_report_8dp"],
        "score": EXPECTED["score_recomputed"],
    }
    if frontier_values != expected_frontier_values:
        raise ValueError(f"DX2 frontier pin drifted: {frontier_values} != {expected_frontier_values}")
    if fx5_receipt["expected_archive_sha256"] != EXPECTED["fx5_archive_sha256"]:
        raise ValueError("FX5 receipt is bound to a different archive")
    dx2_raw = inflated_raw_fact(dx2_receipt)
    fx5_raw = inflated_raw_fact(fx5_receipt)
    dx2_inflate = inflate_report(dx2_receipt)
    if dx2_inflate["token_decoder"]["decoded_token_sha256"] != EXPECTED["label_sha256"]:
        raise ValueError("DX2 T4 receipt decoded a different semantic-label field")
    for name, fact in (("DX2", dx2_raw), ("FX5", fx5_raw)):
        if fact["sha256"] != EXPECTED["t4_raw_sha256"]:
            raise ValueError(f"{name} T4 raw SHA does not match the registered identity")
        if fact["bytes"] != EXPECTED["t4_raw_bytes"]:
            raise ValueError(f"{name} T4 raw bytes do not match the registered identity")

    gt, labels, argmax, rt1_argmax, rt1_labels = load_fields()
    for name, field in (
        ("gt", gt),
        ("labels", labels),
        ("argmax", argmax),
        ("rt1_argmax", rt1_argmax),
        ("rt1_labels", rt1_labels),
    ):
        minimum = int(np.min(field))
        maximum = int(np.max(field))
        if minimum < 0 or maximum >= len(CLASS_NAMES):
            raise ValueError(f"{name} class range [{minimum}, {maximum}] is outside [0, 4]")
    binding = {
        "shape": list(SHAPE),
        "gt_sha256": sources["gt"]["sha256"],
        "label_sha256": sources["labels"]["sha256"],
        "argmax_sha256": sources["argmax"]["sha256"],
        "rt1_argmax_sha256": sources["rt1_argmax"]["sha256"],
        "rt1_label_sha256": sources["rt1_labels"]["sha256"],
        "argmax_provenance_sha256": sources["argmax_provenance"]["sha256"],
        "dx2_t4_raw_sha256": dx2_raw["sha256"],
        "fx5_t4_raw_sha256": fx5_raw["sha256"],
        "mask_definition_version": 2,
    }
    counts, masks = resume_or_materialize_masks(
        args.out,
        binding,
        gt,
        labels,
        argmax,
        rt1_labels,
        rt1_argmax,
        args.chunk_pairs,
    )

    final_errors = counts["final_error"]
    manufactured = counts["manufactured_final_error"]
    survived = counts["representation_survived_final_error"]
    if manufactured + survived != final_errors:
        raise AssertionError("manufactured and representation-survived masks are not additive")
    if final_errors != 23_757:
        raise AssertionError(f"DX2 exact numerator drifted: {final_errors} != 23757")

    rt1_final = counts["rt1_positive_control_final_error"]
    rt1_label = counts["rt1_positive_control_representation_error"]
    rt1_render_vs_label = counts["rt1_positive_control_argmax_changed_from_label"]
    if (rt1_final, rt1_label, rt1_render_vs_label) != (34_938, 1_717, 33_743):
        raise AssertionError("retained RT1 positive-control counts drifted")

    bytes_per_flip = (100 / N_PIXELS) / (25 / RATE_DENOMINATOR_BYTES)
    class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        stem = class_name.lower()
        gt_count = counts[f"class_{class_id}_{stem}_gt_support"]
        class_final = counts[f"class_{class_id}_{stem}_final_error"]
        class_manufactured = counts[f"class_{class_id}_{stem}_manufactured_final_error"]
        class_survived = counts[f"class_{class_id}_{stem}_representation_survived_final_error"]
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": gt_count,
                "gt_fraction": gt_count / N_PIXELS,
                "final_errors": class_final,
                "final_error_share": class_final / final_errors,
                "representation_errors": counts[f"class_{class_id}_{stem}_representation_error"],
                "manufactured_final_errors": class_manufactured,
                "representation_survived_final_errors": class_survived,
                "manufactured_fraction": class_manufactured / class_final,
                "manufactured_byte_equivalent_ceiling": class_manufactured * bytes_per_flip,
                "label_iou_vs_gt": iou(labels, gt, class_id, args.chunk_pairs),
                "argmax_iou_vs_gt": iou(argmax, gt, class_id, args.chunk_pairs),
            }
        )

    contest_eval = nested_artifact(dx2_receipt, "contest_auth_eval.json")
    if contest_eval["canonical_score"] != EXPECTED["score_recomputed"]:
        raise ValueError("nested DX2 contest-eval score pin drifted")
    gt_lineage = contest_eval["gt_lineage"]
    if gt_lineage["lineage"] != "DALI_NVDEC" or not gt_lineage["is_authority_lineage"]:
        raise ValueError("DX2 contest receipt is not bound to authority DALI GT")
    receipt = {
        "schema": "ddm_ms9_dx2_seg_manufactured_fraction_v1",
        "status": "MEASURED_EXACT_FIELD_REPLAY_PARTIAL_STAGE_SPLIT",
        "axis": "[contest-CUDA T4 component-only exact field replay]",
        "score_claim": False,
        "n_pairs": SHAPE[0],
        "shape": list(SHAPE),
        "denominator_pixels": N_PIXELS,
        "source_binding": binding,
        "sources": sources,
        "raw_identity_join": {
            "dx2": dx2_raw,
            "fx5": fx5_raw,
            "identity_holds": dx2_raw["sha256"] == fx5_raw["sha256"],
            "scope": "exact T4 inflated 0.raw bytes; transfers retained FX5 argmax to DX2",
        },
        "dx2_t4_decoded_token_binding": {
            "decoded_token_sha256": dx2_inflate["token_decoder"]["decoded_token_sha256"],
            "token_codec": dx2_inflate["token_decoder"]["token_codec"],
            "pair_count": dx2_inflate["pair_count"],
        },
        "frontier_pin": {
            "archive_sha256": sources["dx2_archive"]["sha256"],
            "archive_bytes": sources["dx2_archive"]["bytes"],
            "d_seg_report_8dp": dx2_receipt["avg_segnet_dist"],
            "d_pose_report_8dp": dx2_receipt["avg_posenet_dist"],
            "score_recomputed_from_components": dx2_receipt["score_recomputed_from_components"],
            "contest_eval_canonical_score": contest_eval["canonical_score"],
        },
        "definition": {
            "manufactured_final_error": "(argmax != GT) and (label == GT)",
            "representation_survived_final_error": "(argmax != GT) and (label != GT)",
            "additive_identity": "final_error = manufactured + representation_survived",
            "reason": (
                "argmax!=label is not additive because downstream rendering also "
                "corrects representation errors and can change one wrong label to another"
            ),
        },
        "counts": counts,
        "fractions": {
            "manufactured_of_final": manufactured / final_errors,
            "representation_survived_of_final": survived / final_errors,
            "changed_from_label_over_final_nonadditive_diagnostic": counts["final_changed_from_label"] / final_errors,
        },
        "score_arithmetic": {
            "exact_seg_term_from_numerator": 100 * final_errors / N_PIXELS,
            "bytes_per_flip_equivalent": bytes_per_flip,
            "total_seg_byte_equivalent_ceiling": final_errors * bytes_per_flip,
            "manufactured_byte_equivalent_ceiling": manufactured * bytes_per_flip,
            "representation_survived_byte_equivalent_ceiling": survived * bytes_per_flip,
            "ceiling_warning": ("oracle-equivalent upper bounds only; not a mechanism or a byte-saving plan"),
        },
        "classes": class_rows,
        "rt1_retained_positive_control": {
            "status": "REPLAY_ONLY_NO_FRESH_SCORER",
            "final_errors": rt1_final,
            "representation_errors": rt1_label,
            "argmax_changes_from_label": rt1_render_vs_label,
            "matches_rt1_receipt": True,
        },
        "stage_custody": {
            "representation_label": sources["labels"],
            "terminal_argmax": sources["argmax"],
            "gt": sources["gt"],
            "gt_lineage": gt_lineage,
            "render_R_uint8_combined": {
                "retained_local_payload": False,
                "remote_manifest_sha256": dx2_raw["sha256"],
                "remote_manifest_bytes": dx2_raw["bytes"],
                "note": "this arm did not rematerialize the T4 raw output",
            },
            "render_R_uint8_individual_fields": {
                "measured": False,
                "reason": "no scorer slot and no retained progressive DX2 fields",
            },
            "argmax_logits_or_margins": {
                "measured": False,
                "reason": "the retained exact field is argmax-only",
            },
        },
        "retained_masks": masks,
        "determinism": {
            "randomness": "none",
            "chunk_pairs": args.chunk_pairs,
            "argv": sys.argv,
            "python": sys.version,
            "numpy": np.__version__,
            "minimum_free_bytes_preflight": MIN_FREE_BYTES,
            "resume_checkpoint": str(args.out / "CHECKPOINT.json"),
        },
    }
    atomic_json(args.out / "MS9_FIELD_REPLAY.json", receipt)
    atomic_json(
        args.out / "MASK_MANIFEST.json",
        {
            "schema": "ddm_ms9_mask_manifest_v1",
            "source_binding": binding,
            "masks": masks,
        },
    )
    atomic_json(
        args.out / "MS9_FIRE_ORDER.json",
        {
            "schema": "ddm_ms9_progressive_stage_split_fire_order_v1",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN",
            "consumer": "next DX2 Seg-cure selector",
            "consumer_store": str(args.out / "fresh_stage_split_r1"),
            "current_blocker": (
                "ms9 does not own the sole n600 scorer lane and no retained DX2 "
                "progressive render/R/uint8/logit field set exists"
            ),
            "fire_trigger": [
                "MAIN records an explicit exclusive n600 scorer-lane grant for ddm_ms9",
                "all previously admitted n600 scorer jobs are terminal",
                "the consumer store is fresh and empty",
                "storage and provenance preflight passes",
                "the new instrument reproduces RT1 34938/1717/33743 and DX2 23757",
            ],
            "ordered_actions": [
                {
                    "order": 1,
                    "action": "claim the unique scorer lane in the live authority store",
                    "owner": "MAIN",
                },
                {
                    "order": 2,
                    "action": (
                        "build a fresh DX2 progressive instrument in actual operator order; "
                        "retain native render, camera uint8, evaluator-resized RGB, logits, "
                        "argmax, and every measured mask"
                    ),
                    "admission": "RT1 and DX2 terminal positive controls pass exactly",
                },
                {
                    "order": 3,
                    "action": "fire full-n600 local advisories serially through the canonical firer only",
                    "argv_pattern": [
                        ".venv/bin/python",
                        "tools/fire_local_advisory.py",
                        "--runtime-dir",
                        "{instrumented_runtime_dir}",
                        "--archive",
                        str(DX2_ARCHIVE),
                        "--attempt-dir",
                        "{fresh_stage_attempt_dir}",
                        "--label",
                        "ddm_ms9_dx2_{stage}_n600",
                    ],
                    "note": (
                        "placeholders are intentionally non-executable until the new stage runtime "
                        "exists and MAIN seals its paths"
                    ),
                },
            ],
            "forbidden": [
                "hand-launched scorer",
                "parallel n600 scorer runs",
                "Modal",
                "Metal authority",
                "discarding any materialized field or mask",
                "reusing HV1 stage fractions as DX2 measurements",
            ],
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
