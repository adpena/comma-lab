#!/usr/bin/env python3
"""Build and validate the fcd3 pair-screened fcd1 candidate ladder.

``screen`` is deliberately scorer-free.  It joins the retained fcd2 per-pair
baseline, GN, and refinement banks, applies the chartered pose thresholds, and
materialises every selected field through fcd1's real field-builder path.

``summarize`` consumes real full-stream re-encode receipts.  It refuses unless
the full-set selector reproduces the fcd1 union archive byte-identically and
then stages one public runtime per rung.  Retained fcd2 pose values are only a
screen: a rung still needs a fresh candidate-bound solve before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import ddm_fcd1_field_for_coder_diagonal as fcd1

STORE = Path("/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal")
FCD3 = STORE / "fcd3_pose_screened_reselection"
FCD2_SCHUR = STORE / "fcd2_distortion_legs" / "union" / "schur"
POOL = STORE / "retained" / "coordinates" / "benefit_pool.frame_y_x_old_new_assignment.npz"
UNION_FIELD = STORE / "retained" / "fields" / "union.tokens.u8"
UNION_ARCHIVE = STORE / "reencode" / "retained" / "candidate_fcd1_union.zip"
BASE_RUNTIME = STORE / "runtimes" / "base_jt21"
SOURCE_CONTROL = STORE / "reencode" / "retained" / "S1_control_600.json"

POOL_SHA256 = "cc09fd9d4cb9a7253df30dbe38d5f60e33ee9e62c8217d9d0b1276ea5c2b5042"
UNION_FIELD_SHA256 = "7988b14811e532e751e1986a85d27aa32410e4d41b07e73ff126ed51a08d2bde"
UNION_ARCHIVE_SHA256 = "c45ab4e687d1a598b2c2191e5c4bf176bb1c12b24748795434cd109eb9a3aa6b"
BASE_ARCHIVE_SHA256 = "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3"
BASE_ARCHIVE_BYTES = 180_192
N_PAIRS = 600
THRESHOLDS = (("tau_1e-8", 1e-8), ("tau_1e-7", 1e-7), ("tau_1e-6", 1e-6))
S_PER_BYTE = 25.0 / 37_545_489.0
AXIS = "[macOS-CPU frozen-scorer advisory screen; exact real re-encode bytes]"


class Fcd3Error(RuntimeError):
    """A retained-input, selection, or byte-identity gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_sha(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected:
        raise Fcd3Error(f"custody pin failed: {path}")
    return file_fact(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_rows(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return per-pair minima and refinement stop rows after complete-bank checks."""
    best: dict[int, dict[str, Any]] = {}
    refinement: dict[int, dict[str, Any]] = {}
    by_family: dict[str, set[int]] = {"gn": set(), "refine": set()}
    for path in paths:
        family = "refine" if "refine_shard" in str(path) else "gn"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                pair = int(row["pair"])
                if pair in by_family[family]:
                    raise Fcd3Error(f"duplicate {family} pair {pair} at {path}:{line_number}")
                by_family[family].add(pair)
                candidate = dict(row)
                candidate["bank_family"] = family
                candidate["bank_path"] = str(path)
                if pair not in best or float(candidate["final_d_pose"]) < float(best[pair]["final_d_pose"]):
                    best[pair] = candidate
                if family == "refine":
                    refinement[pair] = candidate
    expected = set(range(N_PAIRS))
    for family, observed in by_family.items():
        if observed != expected:
            missing = sorted(expected - observed)
            raise Fcd3Error(f"{family} bank is not exactly n600; missing={missing[:8]}")
    return best, refinement


def concentration(values: np.ndarray) -> dict[str, Any]:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    ordered = np.sort(positive)[::-1]
    total = float(ordered.sum())
    cumulative = np.cumsum(ordered)
    curve = []
    for k in (1, 5, 10, 25, 50, 100, 200, 400, 600):
        curve.append({"worst_k_pairs": k, "fraction_of_positive_excess": float(cumulative[k - 1] / total) if total else 0.0})
    capture: dict[str, int | None] = {}
    for fraction in (0.5, 0.8, 0.9, 0.95, 0.99):
        capture[str(fraction)] = int(np.searchsorted(cumulative, fraction * total) + 1) if total else None
    return {"positive_excess_sum": total, "curve": curve, "pairs_needed_for_fraction": capture}


def reuse_control(destination_root: Path) -> dict[str, Any]:
    source = json.loads(SOURCE_CONTROL.read_text())
    if not source.get("byte_identical") or int(source.get("frames", 0)) != N_PAIRS:
        raise Fcd3Error("the retained fcd1 n600 inverse-coder control is not admissible")
    source_stream = Path(source["stream"]["path"])
    if file_fact(source_stream) != source["stream"]:
        raise Fcd3Error("the retained fcd1 control stream drifted")
    destination_stream = destination_root / "work" / "tail_control_600.bin"
    destination_stream.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_stream, destination_stream)
    source_fact = file_fact(SOURCE_CONTROL)
    source["stream"] = file_fact(destination_stream)
    source["fcd3_reuse"] = {
        "reason": "same exact jt21 body, token field, decoder, and already-proven n600 inverse",
        "source_receipt": source_fact,
        "fresh_control_run": False,
    }
    destination = destination_root / "retained" / "S1_control_600.json"
    atomic_json(destination, source)
    return {"receipt": file_fact(destination), "stream": file_fact(destination_stream), "source": source_fact}


def run_screen(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if not out.is_relative_to(STORE.resolve()):
        raise Fcd3Error(f"fcd3 output must remain under the existing consumer store: {out}")
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < (8 << 30):
        raise Fcd3Error(f"storage preflight failed: {free} B free")

    pool_fact = require_sha(POOL, POOL_SHA256)
    union_field_fact = require_sha(UNION_FIELD, UNION_FIELD_SHA256)
    union_archive_fact = require_sha(UNION_ARCHIVE, UNION_ARCHIVE_SHA256)
    baseline_root = FCD2_SCHUR / "baseline"
    base_pose_path = baseline_root / "d_pose_per_pair_base_odd_frames.npy"
    union_pose_path = baseline_root / "d_pose_per_pair_union.npy"
    base_pose = np.load(base_pose_path, allow_pickle=False).astype(np.float64)
    union_pose = np.load(union_pose_path, allow_pickle=False).astype(np.float64)
    if base_pose.shape != (N_PAIRS,) or union_pose.shape != (N_PAIRS,):
        raise Fcd3Error("baseline per-pair vectors are not n600")

    gn_paths = sorted(FCD2_SCHUR.glob("gn_shard[0-4]/rows.jsonl"))
    refine_paths = sorted(FCD2_SCHUR.glob("refine_shard[0-4]/rows.jsonl"))
    if len(gn_paths) != 5 or len(refine_paths) != 5:
        raise Fcd3Error("expected exactly five GN and five refinement shard banks")
    best_rows, refinement_rows = read_rows(gn_paths + refine_paths)
    best_pose = np.asarray([float(best_rows[p]["final_d_pose"]) for p in range(N_PAIRS)])

    pool = np.load(POOL, allow_pickle=False)
    coords = np.asarray(pool["coords"], dtype=np.int32)
    old = np.asarray(pool["old"], dtype=np.uint8)
    new = np.asarray(pool["new"], dtype=np.uint8)
    if coords.shape != (5_268, 3) or old.shape != (5_268,) or new.shape != (5_268,):
        raise Fcd3Error("benefit pool shape drifted")
    edit_count = np.bincount(coords[:, 0], minlength=N_PAIRS).astype(np.int64)
    active = edit_count > 0
    improved = union_pose < base_pose
    if int(improved.sum()) != 8:
        raise Fcd3Error(f"measured-improved pair count drifted: {int(improved.sum())} != 8")
    if np.any(improved & ~active):
        raise Fcd3Error("a measured-improved pair has no fcd1 edit to retain")

    table_rows: list[dict[str, Any]] = []
    for pair in range(N_PAIRS):
        selected = best_rows[pair]
        table_rows.append(
            {
                "pair": pair,
                "edit_positions": int(edit_count[pair]),
                "uncompensated_d_pose": float(union_pose[pair]),
                "per_pair_base_d_pose": float(base_pose[pair]),
                "uncompensated_delta_d_pose": float(union_pose[pair] - base_pose[pair]),
                "measured_improved_uncompensated": bool(improved[pair]),
                "best_compensated_d_pose": float(best_pose[pair]),
                "best_compensated_excess_vs_base": float(best_pose[pair] - base_pose[pair]),
                "best_bank": selected["bank_family"],
                "best_bank_path": selected["bank_path"],
                "stop_reason": selected.get("stop_reason", "gn_bank_best_pre_refinement"),
                "refinement_stop_reason": refinement_rows[pair].get("stop_reason"),
            }
        )
    table_path = out / "screen" / "per_pair_screen.jsonl"
    atomic_jsonl(table_path, table_rows)

    retained = out / "retained"
    tokens = np.memmap(fcd1.TOKENS, dtype=np.uint8, mode="r", shape=(N_PAIRS, fcd1.H, fcd1.W))
    payload_rows: list[dict[str, Any]] = []
    selection_masks: dict[str, np.ndarray] = {}

    full_pair_mask = active.copy()
    selection_masks["fullset_identity"] = full_pair_mask
    for name, tau in THRESHOLDS:
        # The charter says <= but separately requires pair-level ties to be dropped
        # conservatively.  Strict < implements that tie rule; the eight measured
        # uncompensated improvements are forced in exactly as preregistered.
        pair_mask = active & ((best_pose < base_pose + tau) | improved)
        selection_masks[name] = pair_mask

    for name, pair_mask in selection_masks.items():
        selected_positions = np.isin(coords[:, 0], np.flatnonzero(pair_mask))
        payload = fcd1.write_candidate_payload(
            name=name,
            selected=selected_positions,
            coords=coords,
            old=old,
            new=new,
            tokens=tokens,
            retained=retained,
        )
        payload["selected_pairs"] = [int(v) for v in np.flatnonzero(pair_mask)]
        payload["threshold"] = None if name == "fullset_identity" else dict(THRESHOLDS)[name]
        payload_rows.append(payload)

    fullset = next(row for row in payload_rows if row["name"] == "fullset_identity")
    if fullset["candidate_field"]["sha256"] != union_field_fact["sha256"]:
        raise Fcd3Error("full-set pair filter did not reproduce the fcd1 union field")

    vectors_path = out / "screen" / "screen_vectors.npz"
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = vectors_path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            edit_count=edit_count,
            base_pose=base_pose,
            union_pose=union_pose,
            best_compensated_pose=best_pose,
            measured_improved=improved,
            **{f"keep_{name}": mask for name, mask in selection_masks.items()},
        )
    os.replace(temporary, vectors_path)

    reencode_control = reuse_control(out / "reencode")
    result = {
        "schema": "ddm_fcd3_screen.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "screening_only": True,
        "implementation": file_fact(Path(__file__)),
        "storage": {"path": str(out), "free_bytes_before": free, "minimum_free_bytes": 8 << 30},
        "inputs": {
            "benefit_pool": pool_fact,
            "union_field": union_field_fact,
            "union_archive": union_archive_fact,
            "base_pose_per_pair": file_fact(base_pose_path),
            "union_pose_per_pair": file_fact(union_pose_path),
            "gn_banks": [file_fact(p) for p in gn_paths],
            "refinement_banks": [file_fact(p) for p in refine_paths],
        },
        "denominators": {"pairs": N_PAIRS, "active_pairs": int(active.sum()), "B_positions": int(edit_count.sum()), "measured_improved_pairs": int(improved.sum())},
        "per_pair_table": file_fact(table_path),
        "vectors": file_fact(vectors_path),
        "concentration": {
            "uncompensated_positive_excess": concentration(union_pose - base_pose),
            "best_compensated_positive_excess": concentration(best_pose - base_pose),
        },
        "rungs": payload_rows,
        "fullset_field_identity": {"passed": True, "reference": union_field_fact, "observed": fullset["candidate_field"]},
        "reused_inverse_control": reencode_control,
        "next_stage": "real n600 joint re-encode of fullset_identity and all three tau rungs",
    }
    atomic_json(out / "screen" / "SCREEN.json", result)
    print(json.dumps({"rungs": [{"name": r["name"], "pairs": r["active_pairs"], "positions": r["tokens_changed"]} for r in payload_rows], "improved_pairs": int(improved.sum())}, indent=2))
    return 0


def run_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    screen_path = out / "screen" / "SCREEN.json"
    if not screen_path.is_file():
        raise Fcd3Error("SCREEN.json is missing")
    screen = json.loads(screen_path.read_text())
    vectors = np.load(Path(screen["vectors"]["path"]), allow_pickle=False)
    base_pose = np.asarray(vectors["base_pose"], dtype=np.float64)
    best_compensated_pose = np.asarray(vectors["best_compensated_pose"], dtype=np.float64)
    base_d_pose = float(base_pose.mean())
    rows_by_name = {row["name"]: row for row in screen["rungs"]}
    names = ["fullset_identity", *(name for name, _ in THRESHOLDS)]
    rows: list[dict[str, Any]] = []
    for name in names:
        receipt_path = out / "reencode" / "retained" / f"S1_encode_fcd3_{name}.json"
        if not receipt_path.is_file():
            raise Fcd3Error(f"real re-encode receipt missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("frames") != N_PAIRS or not receipt.get("delta_trustworthy"):
            raise Fcd3Error(f"untrustworthy or partial real re-encode: {receipt_path}")
        if receipt.get("tokens_changed") != rows_by_name[name]["tokens_changed"]:
            raise Fcd3Error(f"edit count drift for {name}")
        pointer = receipt.get("pointer_archive", {})
        if pointer.get("sha256") != BASE_ARCHIVE_SHA256 or pointer.get("bytes") != BASE_ARCHIVE_BYTES:
            raise Fcd3Error(f"{name} was not encoded against jt21")
        archive = Path(receipt["candidate_archive"]["path"])
        if file_fact(archive) != receipt["candidate_archive"]:
            raise Fcd3Error(f"retained archive drifted for {name}")
        runtime = fcd1.stage_runtime(BASE_RUNTIME, archive, out / "runtimes" / f"candidate_{name}")
        delta = int(receipt["archive_delta_bytes"])
        screening_pose: dict[str, Any] | None = None
        screening_net: float | None = None
        if name != "fullset_identity":
            keep = np.asarray(vectors[f"keep_{name}"], dtype=bool)
            screened_d_pose = float(np.where(keep, best_compensated_pose, base_pose).mean())
            pose_delta = math.sqrt(10.0 * screened_d_pose) - math.sqrt(10.0 * base_d_pose)
            screening_pose = {
                "evidence_role": "retained-bank screening projection only; not publish evidence",
                "base_d_pose": base_d_pose,
                "screened_d_pose": screened_d_pose,
                "delta_S_pose": pose_delta,
            }
            screening_net = delta * S_PER_BYTE + pose_delta
        rows.append(
            {
                "name": name,
                "threshold": rows_by_name[name].get("threshold"),
                "pairs_kept": int(rows_by_name[name]["active_pairs"]),
                "B_positions_kept": int(rows_by_name[name]["tokens_changed"]),
                "archive_bytes": int(receipt["archive_bytes_candidate"]),
                "archive_delta_bytes_vs_jt21": delta,
                "projected_rate_delta_S": delta * S_PER_BYTE,
                "screening_pose_projection": screening_pose,
                "screening_projected_net_delta_S_excluding_unknown_seg": screening_net,
                "candidate_archive": receipt["candidate_archive"],
                "runtime": runtime,
                "publish_verdict": "SCREEN_ONLY_FRESH_SOLVE_REQUIRED" if name != "fullset_identity" else "IDENTITY_CONTROL",
            }
        )
    identity = rows[0]
    if identity["candidate_archive"]["sha256"] != UNION_ARCHIVE_SHA256:
        raise Fcd3Error("full-set selector failed to reproduce the union archive byte-identically")
    rung_rows = rows[1:]
    best = min(
        rung_rows,
        key=lambda row: (
            row["screening_projected_net_delta_S_excluding_unknown_seg"],
            row["threshold"],
        ),
    )
    result = {
        "schema": "ddm_fcd3_reencode_summary.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "rows": rows,
        "fullset_archive_identity": {"passed": True, "expected_sha256": UNION_ARCHIVE_SHA256, "observed": identity["candidate_archive"]},
        "best_rung": best["name"],
        "best_rung_rule": "most negative rate-plus-retained-screen-pose delta; realized seg is unknown and a fresh exact-object solve is still mandatory",
    }
    atomic_json(out / "reencode" / "REAL_REENCODE_SUMMARY.json", result)
    print(json.dumps({"best_rung": best["name"], "rows": [{"name": r["name"], "pairs": r["pairs_kept"], "positions": r["B_positions_kept"], "bytes": r["archive_bytes"]} for r in rows]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=FCD3)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("screen").set_defaults(func=run_screen)
    sub.add_parser("summarize").set_defaults(func=run_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
