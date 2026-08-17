#!/usr/bin/env python3
"""Select a retained HPAC checkpoint by a declared joint-score proxy.

The selector is scorer-free. It joins periodic checkpoint epochs to the exact
training-log telemetry and minimizes the rate term plus the MC36 top-1 error
proxy. Pose is absent from this telemetry, so the receipt is an advisory input
to endpoint adjudication, never a contest-score or promotion receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "hpac_checkpoint_joint_proxy_selection.v1"
RATE_NUMERATOR = 25.0
RATE_DENOMINATOR = 37_545_489.0
SEG_PROXY_COEFFICIENT = 100.0
_CHECKPOINT_RE = re.compile(r"epoch_(\d+)\.pt\Z")


class SelectionError(RuntimeError):
    """Checkpoint telemetry cannot support an unambiguous selection."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _telemetry_rows(log_text: str, *, source: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        log_text.splitlines(), start=1
    ):
        if "{" not in line or "estimated_joint_bytes" not in line or "top1_error" not in line:
            continue
        try:
            raw = json.loads(line[line.index("{") :])
            raw_epoch = raw["epoch"]
            raw_joint_bytes = raw["estimated_joint_bytes"]
            if isinstance(raw_epoch, bool) or isinstance(raw_joint_bytes, bool):
                raise ValueError("boolean integer field")
            epoch = int(raw_epoch)
            joint_bytes = int(raw_joint_bytes)
            top1_error = float(raw["top1_error"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SelectionError(f"malformed selection telemetry at line {line_number}") from exc
        if (
            (isinstance(raw_epoch, float) and raw_epoch != epoch)
            or (isinstance(raw_joint_bytes, float) and raw_joint_bytes != joint_bytes)
        ):
            raise SelectionError(f"nonintegral selection telemetry at line {line_number}")
        if epoch in rows:
            raise SelectionError(f"duplicate selection telemetry for epoch {epoch}")
        if joint_bytes <= 0 or not math.isfinite(top1_error) or not 0.0 <= top1_error <= 1.0:
            raise SelectionError(f"invalid selection telemetry at epoch {epoch}")
        rows[epoch] = {
            "epoch": epoch,
            "phase": raw.get("phase"),
            "estimated_joint_bytes": joint_bytes,
            "top1_error": top1_error,
            "source_line": line_number,
        }
    if not rows:
        raise SelectionError(f"no joint-byte/top1 telemetry in {source}")
    return rows


def select(log_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    log_path = log_path.expanduser().resolve(strict=True)
    checkpoint_dir = checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise SelectionError(f"checkpoint directory is not a directory: {checkpoint_dir}")
    log_bytes = log_path.read_bytes()
    telemetry = _telemetry_rows(log_bytes.decode("utf-8", errors="replace"), source=log_path)
    checkpoints: dict[int, Path] = {}
    for path in checkpoint_dir.iterdir():
        if not path.is_file() or path.name.startswith("._"):
            continue
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch in checkpoints:
            raise SelectionError(f"duplicate periodic checkpoint for epoch {epoch}")
        if path.stat().st_size < 1:
            raise SelectionError(f"periodic checkpoint is empty at epoch {epoch}: {path}")
        resolved = path.resolve(strict=True)
        if resolved.parent != checkpoint_dir:
            raise SelectionError(f"periodic checkpoint resolves outside its store: {path}")
        checkpoints[epoch] = resolved

    rate_coefficient = RATE_NUMERATOR / RATE_DENOMINATOR
    candidates: list[dict[str, Any]] = []
    for epoch in sorted(set(telemetry) & set(checkpoints)):
        row = telemetry[epoch]
        checkpoint = checkpoints[epoch]
        candidates.append(
            {
                **row,
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "joint_proxy": (
                    rate_coefficient * row["estimated_joint_bytes"]
                    + SEG_PROXY_COEFFICIENT * row["top1_error"]
                ),
            }
        )
    if not candidates:
        raise SelectionError(
            "no epoch has both telemetry and a retained periodic checkpoint"
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["joint_proxy"],
            row["estimated_joint_bytes"],
            row["top1_error"],
            row["epoch"],
        ),
    )
    selected_path = Path(selected["checkpoint_path"])
    rate_only = min(
        candidates,
        key=lambda row: (row["estimated_joint_bytes"], row["top1_error"], row["epoch"]),
    )
    seg_proxy_cost_bytes = (
        selected["estimated_joint_bytes"] - rate_only["estimated_joint_bytes"]
    )
    return {
        "schema": SCHEMA,
        "authority": "[macOS-MPS advisory telemetry proxy; no contest score]",
        "score_claim": False,
        "selection_formula": {
            "expression": "(25/37545489)*estimated_joint_bytes + 100*top1_error",
            "rate_coefficient_per_byte": rate_coefficient,
            "seg_proxy_coefficient": SEG_PROXY_COEFFICIENT,
            "pose_term_included": False,
        },
        # ddm_oa2 observability (score-neutral; does NOT change the selection).
        # On this vehicle the MC36 token labels are FIXED and the HPAC codec is
        # lossless (decoded spatial-token SHA-256 is the losslessness authority),
        # so d_seg is decode-invariant across every checkpoint and `top1_error`
        # carries no score effect: rate is the only axis a checkpoint can move.
        # These fields expose what the seg-proxy term costs on the axis that does
        # move, so a criterion change can be adjudicated on measured evidence
        # rather than argued. Changing the criterion is a design decision and is
        # deliberately NOT taken here.
        "rate_only_optimum": {
            "epoch": rate_only["epoch"],
            "estimated_joint_bytes": rate_only["estimated_joint_bytes"],
            "top1_error": rate_only["top1_error"],
            "checkpoint_path": rate_only["checkpoint_path"],
            "differs_from_selected": rate_only["epoch"] != selected["epoch"],
            "seg_proxy_cost_bytes": seg_proxy_cost_bytes,
            "seg_proxy_cost_score": seg_proxy_cost_bytes * rate_coefficient,
            "boundary": (
                "advisory estimated bytes, not serialized archive bytes; the "
                "ordering it induces is the claim, the magnitude is an estimate"
            ),
        },
        "source_log": {
            "path": str(log_path),
            "bytes": len(log_bytes),
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
        },
        "checkpoint_dir": str(checkpoint_dir),
        "telemetry_epoch_count": len(telemetry),
        "checkpoint_epoch_count": len(checkpoints),
        "joined_candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "selected_checkpoint": _file_record(selected_path),
        "consumer": "task #1058 endpoint identity race and composed-row adjudication",
        "boundary": (
            "selection is only a rate-plus-MC36-label proxy; identity, receiver, Pose, "
            "archive, and exact T4 gates remain owed"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output_path = args.out.expanduser().resolve(strict=False)
        input_log = args.log.expanduser().resolve(strict=False)
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=False)
        if output_path in (input_log, checkpoint_dir) or checkpoint_dir in output_path.parents:
            raise SelectionError("output receipt must be outside the input log and checkpoint store")
        result = select(args.log, args.checkpoint_dir)
        _atomic_json(output_path, result)
    except (OSError, SelectionError) as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc)}), file=os.sys.stderr)
        return 2
    print(json.dumps(result["selected"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
