#!/usr/bin/env python3
"""Adjudicate a contest_auth_eval.json result without regex-log scraping.

Remote lane scripts should treat ``experiments/contest_auth_eval.py`` as the
source of truth and read its JSON artifact directly. This helper validates the
artifact against the exact archive bytes that were evaluated, updates the lane
provenance, and emits a few shell-safe KEY=VALUE lines for completion logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"FATAL: contest_auth_eval.json missing numeric {key!r}")
    out = float(value)
    if not math.isfinite(out):
        raise SystemExit(f"FATAL: contest_auth_eval.json {key!r} is not finite: {value!r}")
    return out


def _status(score: float, predicted_low: float, predicted_high: float, hard_kill_above: float) -> str:
    if score > hard_kill_above:
        return "HARD_KILL_REGRESSION"
    if predicted_low <= score <= predicted_high:
        return "IN_PREDICTED_BAND"
    return "OUT_OF_PREDICTED_BAND"


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    contest_json = Path(args.contest_json)
    archive = Path(args.archive)
    provenance_path = Path(args.provenance)

    if not contest_json.is_file():
        raise SystemExit(f"FATAL: contest auth JSON not found: {contest_json}")
    if not archive.is_file():
        raise SystemExit(f"FATAL: evaluated archive not found: {archive}")

    payload = json.loads(contest_json.read_text())
    score_recomputed = _require_number(payload, "score_recomputed_from_components")
    final_score = _require_number(payload, "final_score")
    if not (0.0 < score_recomputed < args.max_sane_score):
        raise SystemExit(
            "FATAL: score_recomputed_from_components outside sane range "
            f"(0,{args.max_sane_score}): {score_recomputed}"
        )

    n_samples = payload.get("n_samples")
    if n_samples != args.required_samples:
        raise SystemExit(
            f"FATAL: contest_auth_eval.json n_samples={n_samples!r}, "
            f"expected {args.required_samples}"
        )

    actual_archive_bytes = archive.stat().st_size
    actual_archive_sha256 = _sha256(archive)
    payload_archive_bytes = payload.get("archive_size_bytes")
    if payload_archive_bytes != actual_archive_bytes:
        raise SystemExit(
            f"FATAL: contest_auth_eval archive_size_bytes={payload_archive_bytes!r}, "
            f"actual archive bytes={actual_archive_bytes}"
        )

    eval_provenance = payload.get("provenance")
    if not isinstance(eval_provenance, dict):
        raise SystemExit("FATAL: contest_auth_eval.json missing provenance object")
    device = eval_provenance.get("device")
    if device != args.required_device:
        raise SystemExit(
            f"FATAL: contest_auth_eval provenance.device={device!r}, "
            f"expected {args.required_device!r}"
        )
    payload_archive_sha256 = eval_provenance.get("archive_sha256") or payload.get("archive_sha256")
    if payload_archive_sha256 != actual_archive_sha256:
        raise SystemExit(
            "FATAL: contest_auth_eval archive_sha256 does not match evaluated archive: "
            f"json={payload_archive_sha256!r} actual={actual_archive_sha256}"
        )

    if args.result_copy:
        result_copy = Path(args.result_copy)
        result_copy.parent.mkdir(parents=True, exist_ok=True)
        result_copy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        result_copy = contest_json

    predicted_low, predicted_high = args.predicted_band
    lane_status = _status(score_recomputed, predicted_low, predicted_high, args.hard_kill_above)
    gpu_t4_match = eval_provenance.get("gpu_t4_match")
    evidence_grade = "A++ contest T4" if gpu_t4_match is True else "A score-grade"

    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
    else:
        provenance = {}

    baseline_archive_bytes = args.baseline_archive_bytes
    archive_delta_bytes = None
    if baseline_archive_bytes is not None:
        archive_delta_bytes = actual_archive_bytes - baseline_archive_bytes

    provenance.update(
        {
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stacked_archive_bytes": actual_archive_bytes,
            "final_archive_bytes": actual_archive_bytes,
            "baseline_archive_bytes": baseline_archive_bytes,
            "archive_delta_bytes": archive_delta_bytes,
            "contest_cuda_score": score_recomputed,
            "contest_cuda_score_recomputed": score_recomputed,
            "contest_cuda_score_reported_rounded": final_score,
            "contest_cuda_score_source": "contest_auth_eval.json:score_recomputed_from_components",
            "contest_cuda_result_json": str(result_copy),
            "contest_cuda_n_samples": n_samples,
            "contest_cuda_archive_sha256": actual_archive_sha256,
            "contest_cuda_archive_bytes": actual_archive_bytes,
            "contest_cuda_device": device,
            "contest_cuda_gpu_model": eval_provenance.get("gpu_model"),
            "contest_cuda_gpu_t4_match": gpu_t4_match,
            "evidence_grade": evidence_grade,
            "score_tag": "[contest-CUDA]",
            "result_tag": "[contest-CUDA]",
            "score_delta_vs_baseline": score_recomputed - args.baseline_score,
            args.delta_key: score_recomputed - args.baseline_score,
            "hard_kill_triggered": score_recomputed > args.hard_kill_above,
            "lane_status": lane_status,
        }
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    shutil.move(str(tmp_path), str(provenance_path))

    return {
        "score_recomputed": score_recomputed,
        "score_reported_rounded": final_score,
        "lane_status": lane_status,
        "hard_kill_triggered": score_recomputed > args.hard_kill_above,
        "archive_sha256": actual_archive_sha256,
        "archive_bytes": actual_archive_bytes,
        "gpu_model": eval_provenance.get("gpu_model"),
        "gpu_t4_match": gpu_t4_match,
        "evidence_grade": evidence_grade,
        "result_json": str(result_copy),
        "provenance": str(provenance_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contest-json", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--result-copy")
    p.add_argument("--baseline-score", type=float, required=True)
    p.add_argument("--baseline-archive-bytes", type=int)
    p.add_argument("--predicted-band", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    p.add_argument("--hard-kill-above", type=float, required=True)
    p.add_argument("--delta-key", default="score_delta_vs_baseline")
    p.add_argument("--required-device", default="cuda")
    p.add_argument("--required-samples", type=int, default=600)
    p.add_argument("--max-sane-score", type=float, default=10.0)
    args = p.parse_args()

    result = adjudicate(args)
    print(f"SCORE_RECOMPUTED={result['score_recomputed']:.15g}")
    print(f"SCORE_REPORTED_ROUNDED={result['score_reported_rounded']:.15g}")
    print(f"LANE_STATUS={result['lane_status']}")
    print(f"HARD_KILL_TRIGGERED={int(result['hard_kill_triggered'])}")
    print(f"ARCHIVE_SHA256={result['archive_sha256']}")
    print(f"ARCHIVE_BYTES={result['archive_bytes']}")
    print(f"GPU_T4_MATCH={json.dumps(result['gpu_t4_match'])}")
    print(f"EVIDENCE_GRADE={result['evidence_grade']}")
    print(f"RESULT_JSON={result['result_json']}")
    print(f"ADJUDICATION_JSON: {json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
