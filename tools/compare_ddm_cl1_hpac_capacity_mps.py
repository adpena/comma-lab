#!/usr/bin/env python3
"""Compare matched RX2 CPU/MPS parity runs and retain real IHS1 packs.

This is a training-instrument comparison, never score authority.  Both
six-epoch endpoints are packed twice through RX2's real CPU IHS1 pack path;
every raw/XZ payload is retained and SHA-256 inventoried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments import ddm_rx2_mc36_identity_race as race  # noqa: E402
from tools import train_ddm_cl1_hpac_capacity as trainer  # noqa: E402

AXIS = "[macOS CPU-vs-MPS training parity; not score authority]"
METRICS = ("bpp", "top1_error", "estimated_joint_bytes")
SSD_ROOTS = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))
REFERENCE_TRAINER = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
REFERENCE_TRAINER_SHA256 = "8392a9b9f2d303698de59e627fa489a792ab0b0b38170cebd425f9310162059e"
RACE_PACKER = REPO_ROOT / "experiments/ddm_rx2_mc36_identity_race.py"
RACE_PACKER_SHA256 = "5f9cd39f4338a33f8bbca5052508b0989114b15d61ebf521629922e03d7b1d57"


class ParityError(RuntimeError):
    """Fail-closed error for a malformed or unmatched parity endpoint."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_pins() -> None:
    expected = {
        REFERENCE_TRAINER: REFERENCE_TRAINER_SHA256,
        RACE_PACKER: RACE_PACKER_SHA256,
    }
    observed = {path: _sha256_file(path) for path in expected}
    mismatches = {
        str(path): {"expected": digest, "observed": observed[path]}
        for path, digest in expected.items()
        if observed[path] != digest
    }
    if mismatches:
        raise ParityError(f"parity authority source drift: {mismatches}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParityError(f"JSON root is not an object: {path}")
    return value


def _completed_elapsed_s(path: Path) -> float:
    receipt = _read_json(path)
    try:
        elapsed_s = float(receipt["elapsed_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParityError(f"done receipt has no numeric elapsed_s: {path}") from exc
    if (
        receipt.get("schema") != "detached_local_process_done.v2"
        or receipt.get("rc") != 0
        or not math.isfinite(elapsed_s)
        or elapsed_s <= 0
    ):
        raise ParityError(f"done receipt is not a successful measured launch: {path}")
    return elapsed_s


def _latest_live_cpu_epoch(path: Path) -> int:
    epochs: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "estimated_joint_bytes" not in line or "{" not in line:
            continue
        try:
            row = json.loads(line[line.index("{") :])
            epoch = int(row["epoch"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if row.get("phase") in {"continuous", "discrete_qat"}:
            epochs.append(epoch)
    if not epochs:
        raise ParityError(f"live CPU log has no evaluated training epoch: {path}")
    latest = max(epochs)
    if not 0 <= latest <= 60:
        raise ParityError(f"live CPU epoch is outside 0..60: {latest}")
    return latest


def _race_projection(*, cpu_elapsed_s: float, mps_elapsed_s: float, live_cpu_epoch: int) -> dict[str, Any]:
    if cpu_elapsed_s <= 0 or mps_elapsed_s <= 0 or not 0 <= live_cpu_epoch <= 60:
        raise ParityError("race projection inputs are outside their admitted ranges")
    cpu_s_per_epoch = cpu_elapsed_s / 6.0
    mps_s_per_epoch = mps_elapsed_s / 6.0
    cpu_remaining_hours = (60 - live_cpu_epoch) * cpu_s_per_epoch / 3600.0
    mps_full_hours = 60 * mps_s_per_epoch / 3600.0
    finish_margin_hours = cpu_remaining_hours - mps_full_hours
    return {
        "measured_cpu_s_per_epoch": cpu_s_per_epoch,
        "measured_mps_s_per_epoch": mps_s_per_epoch,
        "measured_port_speedup": cpu_s_per_epoch / mps_s_per_epoch,
        "live_cpu_epoch_at_projection": live_cpu_epoch,
        "projected_cpu_remaining_hours": cpu_remaining_hours,
        "projected_full_mps_hours": mps_full_hours,
        "finish_margin_hours": finish_margin_hours,
        "one_subtraction": "projected_cpu_remaining_hours - projected_full_mps_hours",
        "mps_finishes_first_if_cadence_holds": finish_margin_hours > 0,
        "projection_not_measurement": True,
    }


def _require_ssd_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    for root in SSD_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return
    raise ParityError(f"--output-root must live on an admitted SSD tier: {path}")


def _validate_result(value: dict[str, Any], *, expected_device: str) -> None:
    identity = value.get("run_identity", {})
    config = value.get("config", {})
    history = value.get("history")
    if (
        value.get("schema") != "ddm_cl1_hpac_capacity_trainer_result.v1"
        or value.get("score_claim") is not False
        or config.get("profile") != "rx2_mc36"
        or config.get("epochs") != 6
        or config.get("qat_fraction") != 0.5
        or config.get("eval_every") != 2
        or config.get("device") != expected_device
        or identity.get("mps_trained") is not (expected_device == "mps")
        or identity.get("port_mode") != f"parity-{expected_device}"
        or not isinstance(history, list)
    ):
        raise ParityError(f"malformed parity-{expected_device} trainer result")
    if [row.get("epoch") for row in history] != [1, 2, 4, 6]:
        raise ParityError(f"parity-{expected_device} history is not the sealed [1,2,4,6] trajectory")
    if sum(row.get("phase") == "discrete_qat" for row in history) < 2:
        raise ParityError(f"parity-{expected_device} history lacks two evaluated QAT epochs")


def _validate_matched_config(cpu: dict[str, Any], mps: dict[str, Any]) -> None:
    ignored = {"device", "save", "out"}
    cpu_config = {key: value for key, value in cpu["config"].items() if key not in ignored}
    mps_config = {key: value for key, value in mps["config"].items() if key not in ignored}
    if cpu_config != mps_config:
        differing = sorted(
            key for key in set(cpu_config) | set(mps_config) if cpu_config.get(key) != mps_config.get(key)
        )
        raise ParityError(f"CPU/MPS parity configs differ outside device/output paths: {differing}")


def _validate_checkpoint(path: Path, *, expected_device: str) -> dict[str, Any]:
    checkpoint = race.torch.load(path, map_location="cpu", weights_only=False)
    identity = checkpoint.get("run_identity", {})
    config = identity.get("training_config", {})
    if (
        checkpoint.get("schema") != "ddm_cl1_hpac_capacity_checkpoint.v2"
        or checkpoint.get("epoch") != 6
        or checkpoint.get("phase") != "discrete_qat"
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or config.get("profile") != "rx2_mc36"
        or config.get("device") != expected_device
        or identity.get("mps_trained") is not (expected_device == "mps")
    ):
        raise ParityError(f"malformed parity-{expected_device} endpoint checkpoint")
    if trainer._causal_state_sha256(checkpoint) != checkpoint.get("causal_state_sha256"):
        raise ParityError(f"parity-{expected_device} checkpoint causal hash does not verify")
    return checkpoint


def _relative_delta(cpu: float, mps: float) -> float:
    if not math.isfinite(cpu) or not math.isfinite(mps):
        raise ParityError("parity trajectory contains a non-finite metric")
    denominator = abs(cpu)
    if denominator == 0.0:
        return 0.0 if mps == 0.0 else math.inf
    return abs(mps - cpu) / denominator


def _trajectory(cpu: dict[str, Any], mps: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    maxima = dict.fromkeys(METRICS, 0.0)
    for cpu_row, mps_row in zip(cpu["history"], mps["history"], strict=True):
        if cpu_row["epoch"] != mps_row["epoch"] or cpu_row["phase"] != mps_row["phase"]:
            raise ParityError("CPU/MPS parity trajectory epochs or phases differ")
        deltas = {metric: _relative_delta(float(cpu_row[metric]), float(mps_row[metric])) for metric in METRICS}
        for metric, value in deltas.items():
            maxima[metric] = max(maxima[metric], value)
        rows.append(
            {
                "epoch": cpu_row["epoch"],
                "phase": cpu_row["phase"],
                "cpu": {metric: cpu_row[metric] for metric in METRICS},
                "mps": {metric: mps_row[metric] for metric in METRICS},
                "relative_divergence": deltas,
            }
        )
    return rows, maxima


def _pack_twice(checkpoint: Path, root: Path) -> dict[str, Any]:
    first = race._pack_terminal_ihs1(checkpoint, root / "repeat_1")
    second = race._pack_terminal_ihs1(checkpoint, root / "repeat_2")
    deterministic = all(
        first[name][field] == second[name][field] for name in ("raw", "xz") for field in ("bytes", "sha256")
    )
    if not deterministic:
        raise ParityError(f"IHS1 pack repeat changed bytes for {checkpoint}")
    return {
        "checkpoint": race.file_record(checkpoint),
        "repeat_1": first,
        "repeat_2": second,
        "deterministic_repeat_exact": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--cpu-result", type=Path, required=True)
    value.add_argument("--mps-result", type=Path, required=True)
    value.add_argument("--cpu-checkpoint", type=Path, required=True)
    value.add_argument("--mps-checkpoint", type=Path, required=True)
    value.add_argument("--cpu-done-receipt", type=Path, required=True)
    value.add_argument("--mps-done-receipt", type=Path, required=True)
    value.add_argument("--live-cpu-log", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    _validate_source_pins()
    _require_ssd_output(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ParityError("--output-root must be fresh or empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cpu = _read_json(args.cpu_result)
    mps = _read_json(args.mps_result)
    _validate_result(cpu, expected_device="cpu")
    _validate_result(mps, expected_device="mps")
    _validate_matched_config(cpu, mps)
    _validate_checkpoint(args.cpu_checkpoint, expected_device="cpu")
    _validate_checkpoint(args.mps_checkpoint, expected_device="mps")
    trajectory, maxima = _trajectory(cpu, mps)
    projection = _race_projection(
        cpu_elapsed_s=_completed_elapsed_s(args.cpu_done_receipt),
        mps_elapsed_s=_completed_elapsed_s(args.mps_done_receipt),
        live_cpu_epoch=_latest_live_cpu_epoch(args.live_cpu_log),
    )
    packs = {
        "cpu": _pack_twice(args.cpu_checkpoint, args.output_root / "retained/cpu_endpoint"),
        "mps": _pack_twice(args.mps_checkpoint, args.output_root / "retained/mps_endpoint"),
    }
    result = {
        "schema": "ddm_cl1_hpac_capacity_mps_parity.v1",
        "axis": AXIS,
        "score_claim": False,
        "cpu_result": race.file_record(args.cpu_result),
        "mps_result": race.file_record(args.mps_result),
        "trajectory": trajectory,
        "max_relative_divergence": maxima,
        "race_projection": projection,
        "timing_receipts": {
            "cpu": race.file_record(args.cpu_done_receipt),
            "mps": race.file_record(args.mps_done_receipt),
            "live_cpu_log": race.file_record(args.live_cpu_log),
        },
        "endpoint_real_ihs1": packs,
        "mps_kernel_coverage": (
            "PASS: parity-mps completed model+STE+backward+optimizer+EMA with PYTORCH_ENABLE_MPS_FALLBACK=0"
        ),
        "authority_boundary": "MPS is a training research signal; CPU IHS1 pack is serialization authority",
        "source_pins": {
            "reference_trainer": {
                "path": str(REFERENCE_TRAINER),
                "sha256": REFERENCE_TRAINER_SHA256,
            },
            "real_ihs1_packer": {
                "path": str(RACE_PACKER),
                "sha256": RACE_PACKER_SHA256,
            },
        },
        "all_materialized_payloads_retained": True,
        "argv": [sys.executable, *sys.argv],
    }
    race.atomic_json(args.output_root / "PARITY_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
