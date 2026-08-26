# SPDX-License-Identifier: MIT
"""ddm_qo1 -- repair-stream optimal-form byte race on the fz4 ``sub_final`` base.

This is a scorer-slot-safe arm: it never launches a full n600 evaluate.  It
parses the real shipped ``sub_final`` archive, races real lossless coders over
the actual F0PR1 coefficient body, re-runs pair selection using measured
per-pair pose deltas, stages the best byte-closed candidate archive, and checks
the changed pairs through the actual receiver plus frozen CPU scorer.

Axis: [macOS-CPU frozen-scorer advisory] for the <=32 changed-pair forwards;
full n600 remains queued for the scorer owner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

import ddm_r7_token_coder as r7
from ddm_js1_staging_discriminator import Scorer, decode_gt_frames

from tac.optimization.ddm_ix2_archive_container import build_payload, parse_payload
from tac.optimization.frame0_pose_repair_stream import (
    decode_pose_repair_stream,
    encode_pose_repair_stream,
    section_ledger,
)
from tac.submission_chain import build_byte_ledger, stage_submission

RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7541458627114951
LIVE_BEST_BYTES = 358_084
LIVE_D_SEG = 0.00431179
LIVE_D_POSE = 0.00071459
SEG_TERM = 100.0 * LIVE_D_SEG
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_zip_payload(archive: Path) -> bytes:
    with zipfile.ZipFile(archive) as zf:
        return zf.read("0.bin")


def _load_damage_table(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        return {int(row["pair"]): row for row in csv.DictReader(f)}


def _score(*, d_pose_mean: float, archive_bytes: int) -> float:
    return SEG_TERM + math.sqrt(10.0 * d_pose_mean) + RATE_PER_BYTE * archive_bytes


def _raw_coef_body(coefs: dict[int, np.ndarray]) -> bytes:
    return b"".join(np.asarray(coefs[p]).astype("<i2", copy=False).tobytes() for p in sorted(coefs))


def _smevr_nibble_frame(raw_body: bytes, *, n_pairs: int, k: int) -> bytes:
    raw = np.frombuffer(raw_body, dtype=np.uint8)
    nibbles = np.empty(raw.size * 2, dtype=np.uint8)
    nibbles[0::2] = raw >> 4
    nibbles[1::2] = raw & 15
    codes = nibbles.reshape(n_pairs, 3, k * k * 4, 1)
    frame = r7.encode_token_codes(codes, levels=16, codec="smevr")
    restored = r7.decode_token_codes(frame)
    if not np.array_equal(restored, codes):
        raise RuntimeError("SMEVR nibble round-trip failed")
    return frame


def _coder_race(coefs: dict[int, np.ndarray], *, k: int, seg_h: int, seg_w: int) -> list[dict[str, Any]]:
    raw_body = _raw_coef_body(coefs)
    legacy_overhead = 8 + 9 + 4 * len(coefs)
    smevr_frame = _smevr_nibble_frame(raw_body, n_pairs=len(coefs), k=k)
    rows = [
        {
            "coder": "raw_int16",
            "body_bytes": len(raw_body),
            "section_bytes": len(encode_pose_repair_stream(coefs, k=k, seg_h=seg_h, seg_w=seg_w,
                                                            coder="raw_int16")),
            "receiver_supported": True,
        },
        {
            "coder": "lzma1_raw",
            "body_bytes": len(lzma.compress(raw_body, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)),
            "section_bytes": len(encode_pose_repair_stream(coefs, k=k, seg_h=seg_h, seg_w=seg_w,
                                                            coder="lzma1_raw")),
            "receiver_supported": True,
        },
        {
            "coder": "brotli_q11",
            "body_bytes": len(brotli.compress(raw_body, quality=11)),
            "section_bytes": len(encode_pose_repair_stream(coefs, k=k, seg_h=seg_h, seg_w=seg_w,
                                                            coder="brotli_q11")),
            "receiver_supported": True,
        },
        {
            "coder": "smevr_nibble",
            "body_bytes": len(smevr_frame),
            "section_bytes": legacy_overhead + len(smevr_frame),
            "receiver_supported": False,
            "note": "real SMEVR token coder over repair-body nibbles; not a shipped F0PR coder",
        },
        {
            "coder": "pair_bitpack",
            "body_bytes": (
                len(encode_pose_repair_stream(coefs, k=k, seg_h=seg_h, seg_w=seg_w,
                                              coder="pair_bitpack"))
                - legacy_overhead
            ),
            "section_bytes": len(encode_pose_repair_stream(coefs, k=k, seg_h=seg_h, seg_w=seg_w,
                                                            coder="pair_bitpack")),
            "receiver_supported": True,
            "note": "lossless adaptive signed bit width per repaired pair",
        },
    ]
    best_bytes = min(row["section_bytes"] for row in rows)
    for row in rows:
        row["delta_bytes_vs_best"] = int(row["section_bytes"] - best_bytes)
    return sorted(rows, key=lambda row: (row["section_bytes"], row["coder"]))


def _load_probe_candidates(paths: list[Path], coef_paths: list[Path]) -> dict[int, tuple[float, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(path.read_text())["rows"])
    coef_store: dict[str, np.ndarray] = {}
    for path in coef_paths:
        coef_store.update(dict(np.load(path)))
    out: dict[int, tuple[float, np.ndarray]] = {}
    for row in rows:
        if int(row["k"]) != 6 or not row.get("best_coef_saved"):
            continue
        pair = int(row["pair"])
        key = f"p{pair}_k6"
        if key in coef_store:
            out[pair] = (float(row["d_pose_camera_best"]), coef_store[key].astype(np.int16))
    return out


def _waterfill_select(
    *,
    candidates: dict[int, tuple[float, np.ndarray]],
    base_pose: dict[int, float],
    bulk: bytes,
    base_sections: list[bytes],
    current_payload_bytes: int,
    archive_bytes: int,
) -> dict[str, Any]:
    values = dict(base_pose)

    def trial_score(selected: list[int]) -> tuple[float, int, float, bytes]:
        for p in range(600):
            values[p] = base_pose[p]
        coefs = {p: candidates[p][1] for p in selected}
        for p in selected:
            values[p] = candidates[p][0]
        blob = encode_pose_repair_stream(coefs, k=6, seg_h=384, seg_w=512) if coefs else b""
        candidate_payload = build_payload(bulk, [*base_sections, blob]) if coefs else build_payload(
            bulk, base_sections)
        bytes_out = archive_bytes - current_payload_bytes + len(candidate_payload)
        mean_pose = sum(values.values()) / 600.0
        return _score(d_pose_mean=mean_pose, archive_bytes=bytes_out), bytes_out, mean_pose, blob

    order = sorted(
        candidates,
        key=lambda p: base_pose[p] - candidates[p][0],
        reverse=True,
    )
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    best_s, best_bytes, best_pose, best_blob = trial_score(selected)
    first_rejected: dict[str, Any] | None = None
    for pair in order:
        trial = selected + [pair]
        s_new, bytes_new, pose_new, blob_new = trial_score(trial)
        row = {
            "pair": pair,
            "gain_d_pose": base_pose[pair] - candidates[pair][0],
            "trial_S": s_new,
            "trial_archive_bytes": bytes_new,
            "accepted": s_new < best_s,
        }
        trace.append(row)
        if s_new < best_s:
            selected = trial
            best_s, best_bytes, best_pose, best_blob = s_new, bytes_new, pose_new, blob_new
        else:
            first_rejected = row
            break
    return {
        "selection_order": order,
        "selected_pairs": selected,
        "first_rejected": first_rejected,
        "trace": trace,
        "section_ledger": section_ledger(best_blob),
        "archive_bytes_pred": best_bytes,
        "d_pose_mean_pred_from_probe": best_pose,
        "S_pred_from_probe": best_s,
        "blob": best_blob,
    }


def _runtime_source(base_runtime: Path, out_dir: Path) -> Path:
    runtime = out_dir / "runtime_v4d_qo1"
    runtime.mkdir(parents=True, exist_ok=True)
    for path in base_runtime.iterdir():
        if path.suffix in {".py", ".sh"}:
            shutil.copy2(path, runtime / path.name)
    shutil.copy2(REPO / "experiments" / "inflate_runner_v4d.py", runtime / "inflate_runner.py")
    return runtime


def _stage_and_accept(
    *,
    payload: bytes,
    selected_pairs: list[int],
    base_pose: dict[int, float],
    archive_dest: Path,
    runtime: Path,
    gt_mkv: Path,
) -> dict[str, Any]:
    runtime_files = tuple(path.name for path in sorted(runtime.iterdir()) if path.suffix in {".py", ".sh"})
    archive = stage_submission(payload, dest=archive_dest, runtime_src=runtime, runtime_files=runtime_files)
    ledger = build_byte_ledger(
        archive,
        joint_names=("config", "renderer", "selector", "pose_warp", "frame0_pose_repair"),
    )
    ext = archive_dest / "archive"
    if ext.exists():
        shutil.rmtree(ext)
    ext.mkdir()
    from tac.submission_archive import safe_extract_zip

    safe_extract_zip(archive, ext)

    sys.path.insert(0, str(archive_dest))
    sys.modules.pop("inflate_runner", None)
    import inflate_runner as run

    dec = run.Decoder(ext)
    scorer = Scorer(4)
    wanted = set()
    for pair in selected_pairs:
        wanted.update({2 * pair, 2 * pair + 1})
    gt_frames = decode_gt_frames(gt_mkv, wanted)

    values = dict(base_pose)
    control_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        f1 = dec.f1(pair)
        f0 = dec.f0(pair, f1)
        gt = np.stack([gt_frames[2 * pair], gt_frames[2 * pair + 1]])
        d_pose = scorer.d_pose(scorer.pose_out(gt), scorer.pose_out(np.stack([f0, f1])))
        if len(control_rows) < 8:
            seg_with = scorer.seg_argmax(np.stack([f0, f1]))
            seg_without = scorer.seg_argmax(np.stack([np.zeros_like(f0), f1]))
            seg_equal = bool(np.array_equal(seg_with, seg_without))
        else:
            seg_equal = None
        values[pair] = d_pose
        control_rows.append({"pair": pair, "d_pose": d_pose, "seg_equal_to_zero_f0_control": seg_equal})

    d_pose_mean = sum(values.values()) / 600.0
    s_pred = _score(d_pose_mean=d_pose_mean, archive_bytes=archive.stat().st_size)
    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "byte_ledger": ledger.as_dict(),
        "d_pose_mean_pred_from_changed_pair_forwards": d_pose_mean,
        "S_pred_advisory": s_pred,
        "delta_vs_live_best": s_pred - LIVE_BEST_S,
        "changed_pair_forwards": control_rows,
        "seg_control_n": sum(row["seg_equal_to_zero_f0_control"] is not None for row in control_rows),
        "seg_control_all_equal": all(
            row["seg_equal_to_zero_f0_control"] is True
            for row in control_rows
            if row["seg_equal_to_zero_f0_control"] is not None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=Path(
        "/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip"))
    ap.add_argument("--damage-table", type=Path, default=Path(
        ".omx/research/ddm_fz4_20260804/fz4_per_pair_damage_table.csv"))
    ap.add_argument("--probe-json", type=Path, action="append", default=[
        Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/pu2_tail_solve_a.json"),
        Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/pu2_tail_solve_b.json"),
    ])
    ap.add_argument("--coefs-npz", type=Path, action="append", default=[
        Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/pu2_tail_coefs_a.npz"),
        Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/pu2_tail_coefs_b.npz"),
    ])
    ap.add_argument("--base-runtime", type=Path, default=Path(
        "/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/runtime"))
    ap.add_argument("--gt-mkv", type=Path, default=Path("upstream/videos/0.mkv"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/Volumes/VertigoDataTier/pact/ddm_qo1_20260804"))
    ap.add_argument("--repo-receipt", type=Path, default=Path(
        ".omx/research/ddm_qo1_repair_stream_optimal_form_20260804.json"))
    ap.add_argument("--kcrop-json", type=Path, default=Path(
        "/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/kcrop_probe/kcrop_results.json"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_zip_payload(args.archive)
    bulk, sections = parse_payload(payload)
    current = decode_pose_repair_stream(sections[-1])
    current_coefs = current.coefs
    damage = _load_damage_table(args.damage_table)
    base_pose = {p: float(damage[p]["pu2_live_d_pose"]) for p in range(600)}
    current_pose = {
        p: float(damage[p]["sub_final_d_pose_after_repair"])
        for p in range(600)
    }
    candidates = _load_probe_candidates(args.probe_json, args.coefs_npz)
    waterfill = _waterfill_select(
        candidates=candidates,
        base_pose=base_pose,
        bulk=bulk,
        base_sections=list(sections[:-1]),
        current_payload_bytes=len(payload),
        archive_bytes=args.archive.stat().st_size,
    )
    best_blob = waterfill.pop("blob")
    best_payload = build_payload(bulk, [*sections[:-1], best_blob])
    runtime = _runtime_source(args.base_runtime, args.out_dir)
    accepted = _stage_and_accept(
        payload=best_payload,
        selected_pairs=waterfill["selected_pairs"],
        base_pose=base_pose,
        archive_dest=args.out_dir / "sub_auto_pairbit",
        runtime=runtime,
        gt_mkv=args.gt_mkv,
    )

    current_score = _score(d_pose_mean=sum(current_pose.values()) / 600.0,
                           archive_bytes=args.archive.stat().st_size)
    coder_race = _coder_race(current_coefs, k=current.k, seg_h=current.seg_h, seg_w=current.seg_w)
    kcrop = json.loads(args.kcrop_json.read_text()) if args.kcrop_json.exists() else None
    receipt = {
        "schema": "ddm_qo1_repair_stream_optimal_form.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] for <=32 changed-pair forwards; no full n600",
        "score_claim": False,
        "source_archive": str(args.archive),
        "source_archive_bytes": args.archive.stat().st_size,
        "source_archive_sha256": _sha256(args.archive),
        "source_f0pr_section_ledger": section_ledger(sections[-1]),
        "live_best_baseline": {
            "S": LIVE_BEST_S,
            "archive_bytes": LIVE_BEST_BYTES,
            "d_seg": LIVE_D_SEG,
            "d_pose": LIVE_D_POSE,
            "axis": "[macOS-CPU advisory]",
        },
        "recomputed_current_from_damage_table": {
            "S": current_score,
            "d_pose_mean": sum(current_pose.values()) / 600.0,
            "archive_bytes": args.archive.stat().st_size,
            "delta_vs_live_best": current_score - LIVE_BEST_S,
        },
        "coder_race_on_current_21_pair_body": coder_race,
        "selective_pair_waterfill": waterfill,
        "best_candidate": accepted,
        "kcrop_uniform_k1_to_k6_small_batch": kcrop,
        "boundaries": [
            "No full-n600 scorer job launched; sg4/fleet owner must fire queued verdict.",
            "d_seg is held constant by construction and by an 8-pair SegNet zero-frame0 control.",
            "Uniform k<=4 rows are cropped-current-coef variants, not new quant-aware lower-k solves.",
            "SMEVR row is a real token-coder round trip over repair-body nibbles, but not a shipped F0PR coder.",
        ],
    }
    args.repo_receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    (args.out_dir / "ddm_qo1_repair_stream_optimal_form_20260804.json").write_text(
        json.dumps(receipt, indent=1) + "\n"
    )
    print(json.dumps({
        "repo_receipt": str(args.repo_receipt),
        "candidate_archive": accepted["archive"],
        "candidate_bytes": accepted["archive_bytes"],
        "candidate_S_pred": accepted["S_pred_advisory"],
        "delta_vs_live_best": accepted["delta_vs_live_best"],
        "selected_pairs": waterfill["selected_pairs"],
        "f0pr_section": waterfill["section_ledger"],
        "seg_control_all_equal": accepted["seg_control_all_equal"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
