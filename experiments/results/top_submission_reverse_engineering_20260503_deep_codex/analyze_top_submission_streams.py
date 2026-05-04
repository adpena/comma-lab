#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
CONTEST_ORIGINAL_BYTES = 37_545_489


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pact_unpack_renderer_payload", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zip_members(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            rows.append(
                {
                    "name": info.filename,
                    "compress_type": info.compress_type,
                    "compress_size": info.compress_size,
                    "file_size": info.file_size,
                    "crc": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(data),
                    "date_time": list(info.date_time),
                    "external_attr": info.external_attr,
                }
            )
    return rows


def read_single_payload(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"{path} has {len(names)} members; expected single payload")
        return names[0], zf.read(names[0])


def charged_slices(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, start: int, end: int, codec: str) -> None:
        data = payload[start:end]
        rows.append(
            {
                "name": name,
                "offset": start,
                "end": end,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "magic_hex": data[:8].hex(),
                "codec": codec,
            }
        )

    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        add("p3_header", 0, cursor, "header")
        add("masks.mkv.br", cursor, cursor + mask_len, "brotli")
        cursor += mask_len
        add("renderer.bin.br", cursor, cursor + model_len, "brotli")
        cursor += model_len
        add("seg_tile_actions.br", cursor, cursor + actions_len, "brotli")
        cursor += actions_len
        add("optimized_poses.bin.br", cursor, len(payload), "brotli_qp1")
    elif payload.startswith(b"P4"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, model_len, dict_len, actions_len = struct.unpack_from("<IHHH", payload, 2)
        add("p4_header", 0, cursor, "header")
        add("masks.mkv.br", cursor, cursor + mask_len, "brotli")
        cursor += mask_len
        add("renderer.bin.br", cursor, cursor + model_len, "brotli")
        cursor += model_len
        add("seg_tile_action_dict.br", cursor, cursor + dict_len, "brotli")
        cursor += dict_len
        add("seg_tile_actions.br", cursor, cursor + actions_len, "brotli")
        cursor += actions_len
        add("optimized_poses.bin.br", cursor, len(payload), "brotli_qp1")
    elif payload.startswith(b"P5"):
        cursor = 2 + struct.calcsize("<IHHHH")
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from("<IHHHH", payload, 2)
        add("p5_header", 0, cursor, f"header_record_count={record_count}")
        add("masks.mkv.br", cursor, cursor + mask_len, "brotli")
        cursor += mask_len
        add("renderer.bin.br", cursor, cursor + model_len, "brotli")
        cursor += model_len
        add("seg_tile_action_dict.br", cursor, cursor + dict_len, "brotli")
        cursor += dict_len
        add("seg_tile_actions.packed24.br", cursor, cursor + actions_len, "brotli")
        cursor += actions_len
        add("optimized_poses.bin.br", cursor, len(payload), "brotli_qp1")
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
        add("p6_header", 0, cursor, f"header_record_count={record_count}")
        add("masks.mkv.br", cursor, cursor + mask_len, "brotli")
        cursor += mask_len
        add("renderer.bin.br", cursor, cursor + model_len, "brotli")
        cursor += model_len
        add("seg_tile_actions.delta_varint.br", cursor, cursor + actions_len, "brotli")
        cursor += actions_len
        add("optimized_poses.bin.br", cursor, len(payload), "brotli_qp1")
    else:
        # Public fixed slices. PR75 is mask, model, actions, pose. PR67 older
        # archives are mask, model, pose and will not validate the action slice.
        mask_len = 219_472
        model_len = 56_034 if len(payload) >= 276_641 else 55_965
        actions_len = 236 if len(payload) == 276_641 else 0
        cursor = 0
        add("masks.mkv.br", cursor, cursor + mask_len, "brotli")
        cursor += mask_len
        add("renderer.bin.br", cursor, cursor + model_len, "brotli")
        cursor += model_len
        if actions_len:
            add("seg_tile_actions.br", cursor, cursor + actions_len, "brotli")
            cursor += actions_len
        add("optimized_poses.bin.br", cursor, len(payload), "brotli_qp1")
    return rows


def decode_action_records(raw: bytes | None) -> list[dict[str, int]]:
    if not raw:
        return []
    if len(raw) % 4 != 0:
        return [{"decode_error": len(raw)}]  # type: ignore[list-item]
    records = []
    for offset in range(0, len(raw), 4):
        records.append(
            {
                "index": offset // 4,
                "pair_index": int.from_bytes(raw[offset:offset + 2], "little"),
                "tile_id": raw[offset + 2],
                "action_id": raw[offset + 3],
            }
        )
    return records


def summarize_actions(records: list[dict[str, int]]) -> dict[str, Any]:
    if not records or "decode_error" in records[0]:
        return {"record_count": 0, "decode_error": records[0].get("decode_error") if records else None}
    pairs = [r["pair_index"] for r in records]
    tiles = [r["tile_id"] for r in records]
    actions = [r["action_id"] for r in records]
    return {
        "record_count": len(records),
        "unique_pairs": len(set(pairs)),
        "unique_tiles": len(set(tiles)),
        "unique_actions": len(set(actions)),
        "pair_min": min(pairs),
        "pair_max": max(pairs),
        "tile_min": min(tiles),
        "tile_max": max(tiles),
        "action_min": min(actions),
        "action_max": max(actions),
        "records_sha256": sha256_bytes(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def member_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": sha256_bytes(data), "magic_hex": data[:8].hex()}
        for name, data in sorted(decoded.items())
    }


def score_from_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    # contest_auth_eval.json appears in a few nearby schemas; keep this tolerant.
    pose = data.get("posenet_dist") or data.get("avg_posenet_dist") or data.get("average_posenet_distortion")
    seg = data.get("segnet_dist") or data.get("avg_segnet_dist") or data.get("average_segnet_distortion")
    score = data.get("canonical_score") or data.get("score_recomputed_from_components") or data.get("score") or data.get("final_score")
    archive_bytes = data.get("archive_bytes") or data.get("archive_size_bytes") or data.get("submission_file_size")
    if pose is None:
        pose = data.get("metrics", {}).get("posenet_dist")
    if seg is None:
        seg = data.get("metrics", {}).get("segnet_dist")
    if score is None and pose is not None and seg is not None and archive_bytes is not None:
        score = 100 * float(seg) + math.sqrt(10 * float(pose)) + 25 * int(archive_bytes) / CONTEST_ORIGINAL_BYTES
    return {
        "path": str(path),
        "score": score,
        "posenet_dist": pose,
        "segnet_dist": seg,
        "archive_bytes": archive_bytes,
    }


def analyze_archive(label: str, path: Path, unpacker: Any, eval_json: Path | None = None) -> dict[str, Any]:
    member_name, payload = read_single_payload(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    actions = decode_action_records(decoded.get("seg_tile_actions.bin"))
    return {
        "label": label,
        "archive_path": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_bytes(path.read_bytes()),
        "zip_members": zip_members(path),
        "payload_member": member_name,
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "payload_format": header.get("payload_format"),
        "charged_slices": charged_slices(payload),
        "decoded_members": member_summary(decoded),
        "actions": summarize_actions(actions),
        "action_records": actions,
        "score": score_from_json(eval_json),
    }


def pose_metrics(a: bytes | None, b: bytes | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    aa = np.frombuffer(a, dtype=np.float16).astype(np.float32)
    bb = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    n = min(len(aa), len(bb))
    diff = aa[:n] - bb[:n]
    return {
        "lhs_values": int(len(aa)),
        "rhs_values": int(len(bb)),
        "compared_values": int(n),
        "equal_prefix": bool(np.array_equal(aa[:n], bb[:n])),
        "max_abs": float(np.max(np.abs(diff))) if n else None,
        "mean_abs": float(np.mean(np.abs(diff))) if n else None,
        "rms": float(np.sqrt(np.mean(diff * diff))) if n else None,
        "changed_values": int(np.count_nonzero(diff)) if n else 0,
    }


def byte_metrics(a: bytes | None, b: bytes | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    changed = sum(1 for i in range(n) if a[i] != b[i])
    first_diff = next((i for i in range(n) if a[i] != b[i]), None)
    return {
        "lhs_bytes": len(a),
        "rhs_bytes": len(b),
        "compared_bytes": n,
        "byte_delta_lhs_minus_rhs": len(a) - len(b),
        "equal_prefix": a[:n] == b[:n],
        "changed_prefix_bytes": changed,
        "first_diff_offset": first_diff,
    }


def compare(lhs: dict[str, Any], rhs: dict[str, Any], decoded_cache: dict[str, dict[str, bytes]]) -> dict[str, Any]:
    ldec = decoded_cache[lhs["label"]]
    rdec = decoded_cache[rhs["label"]]
    lrecords = {(r["pair_index"], r["tile_id"], r["action_id"]) for r in lhs["action_records"] if "pair_index" in r}
    rrecords = {(r["pair_index"], r["tile_id"], r["action_id"]) for r in rhs["action_records"] if "pair_index" in r}
    lpairtile = {(r["pair_index"], r["tile_id"]) for r in lhs["action_records"] if "pair_index" in r}
    rpairtile = {(r["pair_index"], r["tile_id"]) for r in rhs["action_records"] if "pair_index" in r}
    member_cmp: dict[str, Any] = {}
    for name in sorted(set(ldec) | set(rdec)):
        la = ldec.get(name)
        rb = rdec.get(name)
        member_cmp[name] = {
            "lhs_present": la is not None,
            "rhs_present": rb is not None,
            "lhs_bytes": len(la) if la is not None else None,
            "rhs_bytes": len(rb) if rb is not None else None,
            "sha_equal": sha256_bytes(la) == sha256_bytes(rb) if la is not None and rb is not None else False,
            "byte_metrics": byte_metrics(la, rb),
        }
    return {
        "lhs": lhs["label"],
        "rhs": rhs["label"],
        "archive_byte_delta_lhs_minus_rhs": lhs["archive_bytes"] - rhs["archive_bytes"],
        "payload_byte_delta_lhs_minus_rhs": lhs["payload_bytes"] - rhs["payload_bytes"],
        "member_comparison": member_cmp,
        "pose_metrics": pose_metrics(ldec.get("optimized_poses.bin"), rdec.get("optimized_poses.bin")),
        "action_exact_overlap": len(lrecords & rrecords),
        "action_pairtile_overlap": len(lpairtile & rpairtile),
        "lhs_action_records_not_rhs": sorted(lrecords - rrecords)[:200],
        "rhs_action_records_not_lhs": sorted(rrecords - lrecords)[:200],
    }


def write_action_csv(path: Path, analyses: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "index", "pair_index", "tile_id", "action_id"])
        writer.writeheader()
        for item in analyses:
            for rec in item["action_records"]:
                if "pair_index" in rec:
                    writer.writerow({"label": item["label"], **rec})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", action="append", nargs=3, metavar=("LABEL", "PATH", "EVAL_JSON_OR_NONE"))
    args = parser.parse_args()

    unpacker = load_unpacker()
    analyses = []
    decoded_cache: dict[str, dict[str, bytes]] = {}
    for label, raw_path, raw_eval in args.archive or []:
        path = Path(raw_path)
        eval_json = None if raw_eval == "none" else Path(raw_eval)
        member_name, payload = read_single_payload(path)
        header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
        decoded_cache[label] = decoded
        item = analyze_archive(label, path, unpacker, eval_json)
        analyses.append(item)

    comparisons = []
    by_label = {item["label"]: item for item in analyses}
    for lhs_label in by_label:
        for rhs_label in ("public_pr75", "legacy_public_pr67", "c088_top40_p3", "c089_staged_top25_ampminus1_p3"):
            if lhs_label != rhs_label and rhs_label in by_label:
                comparisons.append(compare(by_label[lhs_label], by_label[rhs_label], decoded_cache))

    out = {
        "schema": "top_submission_reverse_engineering_stream_analysis_v1",
        "repo_root": str(REPO_ROOT),
        "unpacker": str(UNPACKER_PATH),
        "archives": analyses,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stream_analysis.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    write_action_csv(args.output_dir / "action_records.csv", analyses)

    summary_rows = []
    for item in analyses:
        score = item.get("score") or {}
        summary_rows.append(
            {
                "label": item["label"],
                "archive_bytes": item["archive_bytes"],
                "archive_sha256": item["archive_sha256"],
                "payload_bytes": item["payload_bytes"],
                "payload_format": item["payload_format"],
                "action_records": item["actions"].get("record_count"),
                "score": score.get("score") if score else None,
                "pose": score.get("posenet_dist") if score else None,
                "seg": score.get("segnet_dist") if score else None,
            }
        )
    with (args.output_dir / "archive_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
