#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the three preregistered regularized-max family probes without proxies.

The real n600 target logits can answer the target-side sparsemax/Cole--Hopf
questions.  Receiver claims require three additional, canonically registered
surfaces: a logits-to-uint8 preimage adapter, frozen rank-4 valid-cell
prototypes, and an Aurenhammer min-generator comparator.  This tool refuses to
invent any of them.  Missing surfaces produce an explicit N-A receipt, never a
synthetic comparison or a negative family verdict.

No scorer forward is launched unless a receiver candidate exists.  Therefore
the cached hard labels characterize the target surface, while the frozen CPU
Torch scorer remains the terminal HARD_ACCEPT authority for any future
receiver-complete rerun.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SURFACE_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
)
DEFAULT_LOGITS = DEFAULT_SURFACE_ROOT / "teacher_logits_n600/gt_segnet_logits.f16"
DEFAULT_LABELS = DEFAULT_SURFACE_ROOT / "targets_n600/gt_segnet_argmax.u8"
DEFAULT_REGISTRY = REPO / ".omx/state/canonical_equations_registry.jsonl"

N_PAIRS = 600
N_CLASSES = 5
HEIGHT = 384
WIDTH = 512
TOTAL_PIXELS = N_PAIRS * HEIGHT * WIDTH
UNIT_SCALE = 1.0

LOGITS_BYTES = 1_179_648_000
LOGITS_SHA256 = "41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52"
LABELS_BYTES = 117_964_800
LABELS_SHA256 = "36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68"

REQUIRED_REGISTRATIONS = {
    "preimage_adapter": "regmax_logits_to_uint8_preimage_v1",
    "valid_cell_prototypes": "rank4_valid_cell_prototypes_v1",
    "aurenhammer_comparator": "aurenhammer_min_generator_lp_v1",
    "principal_cell_fixture": "rank4_principal_cell_inequality_fixture_v1",
}

SPARSEMAX_PROBE = "probe_sparsemax_margin_band_preimage_ab_v1"
TROPICAL_PROBE = "probe_tropical_residuation_principal_cell_representative_v1"
HOPFIELD_PROBE = "probe_entropy_hopfield_preprox_uint8_v1"

SPARSEMAX_FALSIFIER = (
    "FALSIFIED if the matched sparsemax arm fails to improve hard accepts, exact-oracle "
    "calls, or bytes versus entropy/Cole-Hopf on identical cells and budget."
)
TROPICAL_FALSIFIER = (
    "FALSIFIED if the gauge-fixed principal representative changes the hard cell, is "
    "longer after the same coder, or requires uncounted state."
)
HOPFIELD_FALSIFIER = (
    "FALSIFIED if one frozen-prototype memory-prox step does not improve hard-accept "
    "count or exact-call cost versus no-prox on identical cells and budget."
)


class ProbeError(RuntimeError):
    """Fail-closed input or custody error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sparsemax(logits: np.ndarray, *, scale: float = UNIT_SCALE) -> np.ndarray:
    """Euclidean simplex projection along the last axis.

    ``scale`` is the coefficient of ``0.5 ||p||^2``.  At unit scale this is
    exactly one-hot iff the top-one/top-two logit margin is at least one.
    """

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ProbeError("sparsemax requires a final class axis of length >=2")
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0:
        raise ProbeError("sparsemax inputs and positive scale must be finite")
    scaled = values / float(scale)
    ordered = np.sort(scaled, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ordered, axis=-1)
    ranks = np.arange(1, values.shape[-1] + 1, dtype=np.float64)
    active = 1.0 + ranks * ordered > cumulative
    support = np.sum(active, axis=-1)
    threshold = (np.take_along_axis(cumulative, (support - 1)[..., None], axis=-1)[..., 0] - 1.0) / support
    projected = np.maximum(scaled - threshold[..., None], 0.0)
    return projected


def softmax(logits: np.ndarray, *, scale: float = UNIT_SCALE) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0:
        raise ProbeError("softmax inputs and positive scale must be finite")
    shifted = values / float(scale)
    shifted -= np.max(shifted, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


@dataclass
class _Aggregate:
    count: int = 0
    entropy_debt_sum: float = 0.0
    sparsemax_debt_sum: float = 0.0
    entropy_support_sum: int = 0
    sparsemax_support_sum: int = 0
    sparsemax_exact_one_hot: int = 0

    def add(
        self,
        entropy_debt: np.ndarray,
        sparsemax_debt: np.ndarray,
        sparsemax_support: np.ndarray,
    ) -> None:
        n = int(entropy_debt.size)
        self.count += n
        self.entropy_debt_sum += float(np.sum(entropy_debt, dtype=np.float64))
        self.sparsemax_debt_sum += float(np.sum(sparsemax_debt, dtype=np.float64))
        self.entropy_support_sum += N_CLASSES * n
        self.sparsemax_support_sum += int(np.sum(sparsemax_support, dtype=np.int64))
        self.sparsemax_exact_one_hot += int(np.count_nonzero(sparsemax_support == 1))

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "pixels": 0,
                "entropy_target_debt_mean": None,
                "sparsemax_target_debt_mean": None,
                "entropy_support_size_mean": None,
                "sparsemax_support_size_mean": None,
                "sparsemax_exact_one_hot_fraction": None,
            }
        return {
            "pixels": self.count,
            "entropy_target_debt_mean": self.entropy_debt_sum / self.count,
            "sparsemax_target_debt_mean": self.sparsemax_debt_sum / self.count,
            "entropy_support_size_mean": self.entropy_support_sum / self.count,
            "sparsemax_support_size_mean": self.sparsemax_support_sum / self.count,
            "sparsemax_exact_one_hot_fraction": self.sparsemax_exact_one_hot / self.count,
        }


@dataclass
class _PredictionAccumulator:
    overall: _Aggregate = field(default_factory=_Aggregate)
    by_class: list[_Aggregate] = field(
        default_factory=lambda: [_Aggregate() for _ in range(N_CLASSES)]
    )
    by_stratum: dict[str, _Aggregate] = field(
        default_factory=lambda: {
            "high_margin_interior_ge_1": _Aggregate(),
            "boundary_annulus_or_tie_lt_1": _Aggregate(),
        }
    )
    by_class_stratum: dict[str, list[_Aggregate]] = field(
        default_factory=lambda: {
            "high_margin_interior_ge_1": [_Aggregate() for _ in range(N_CLASSES)],
            "boundary_annulus_or_tie_lt_1": [_Aggregate() for _ in range(N_CLASSES)],
        }
    )
    fp16_argmax_vs_hard_label_mismatches: int = 0

    def add(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
            raise ProbeError(f"logit chunk must be (pixels,{N_CLASSES}), got {logits.shape}")
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        if target.shape != (logits.shape[0],) or np.any((target < 0) | (target >= N_CLASSES)):
            raise ProbeError("hard target labels are missing, out of range, or shape-incompatible")

        entropy = softmax(logits)
        sparse = sparsemax(logits)
        rows = np.arange(target.size)
        entropy_debt = 1.0 - entropy[rows, target]
        sparse_debt = 1.0 - sparse[rows, target]
        sparse_support = np.count_nonzero(sparse > 0.0, axis=-1)
        ordered = np.sort(np.asarray(logits, dtype=np.float64), axis=-1)
        high_margin = ordered[:, -1] - ordered[:, -2] >= UNIT_SCALE
        self.fp16_argmax_vs_hard_label_mismatches += int(
            np.count_nonzero(np.argmax(logits, axis=-1) != target)
        )

        self.overall.add(entropy_debt, sparse_debt, sparse_support)
        for class_id in range(N_CLASSES):
            class_mask = target == class_id
            self.by_class[class_id].add(
                entropy_debt[class_mask], sparse_debt[class_mask], sparse_support[class_mask]
            )
        for name, mask in (
            ("high_margin_interior_ge_1", high_margin),
            ("boundary_annulus_or_tie_lt_1", ~high_margin),
        ):
            self.by_stratum[name].add(
                entropy_debt[mask], sparse_debt[mask], sparse_support[mask]
            )
            for class_id in range(N_CLASSES):
                both = mask & (target == class_id)
                self.by_class_stratum[name][class_id].add(
                    entropy_debt[both], sparse_debt[both], sparse_support[both]
                )

    def result(self) -> dict[str, Any]:
        return {
            "overall": self.overall.result(),
            "per_class": {str(i): value.result() for i, value in enumerate(self.by_class)},
            "per_stratum": {key: value.result() for key, value in self.by_stratum.items()},
            "per_class_within_stratum": {
                key: {str(i): value.result() for i, value in enumerate(values)}
                for key, values in self.by_class_stratum.items()
            },
            "fp16_argmax_vs_hard_label_mismatches": self.fp16_argmax_vs_hard_label_mismatches,
        }


def measure_prediction_surface(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Measure prediction-map quantities on arrays with class-last logits."""

    accumulator = _PredictionAccumulator()
    accumulator.add(np.asarray(logits).reshape(-1, N_CLASSES), np.asarray(labels).reshape(-1))
    return accumulator.result()


def _registered_surfaces(path: Path) -> dict[str, bool]:
    if not path.is_file():
        raise ProbeError(f"canonical equation registry is missing: {path}")
    text = path.read_text(errors="strict")
    return {name: equation_id in text for name, equation_id in REQUIRED_REGISTRATIONS.items()}


def _n_a_receipt(
    *,
    probe_id: str,
    falsifier: str,
    blockers: list[str],
    common: dict[str, Any],
    partial_measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **common,
        "probe_id": probe_id,
        "verdict": "N-A",
        "verdict_scope": "matched preregistered formulation on this branch",
        "falsifier": falsifier,
        "falsifier_evaluated": False,
        "blockers": blockers,
        "hard_cpu_torch_oracle": {
            "authority": "terminal HARD_ACCEPT only after decoded uint8 candidate parse-back",
            "invoked": False,
            "reason": "no receiver candidate exists without the missing registered surfaces",
        },
        "hard_accepts": None,
        "exact_oracle_calls": None,
        "candidate_bytes_same_coder": None,
        "partial_measurement": partial_measurement,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    logits_path = args.logits.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    for path, expected_bytes, expected_sha, name in (
        (logits_path, LOGITS_BYTES, LOGITS_SHA256, "logits"),
        (labels_path, LABELS_BYTES, LABELS_SHA256, "hard labels"),
    ):
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ProbeError(f"real n600 {name} bytes mismatch at {path}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ProbeError(f"real n600 {name} sha256 mismatch: {actual_sha}")

    registrations = _registered_surfaces(registry_path)
    logits = np.memmap(
        logits_path, dtype=np.float16, mode="r", shape=(N_PAIRS, N_CLASSES, HEIGHT, WIDTH)
    )
    labels = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    accumulator = _PredictionAccumulator()
    for pair_id in range(N_PAIRS):
        class_last = np.moveaxis(np.asarray(logits[pair_id], dtype=np.float32), 0, -1)
        accumulator.add(class_last.reshape(-1, N_CLASSES), np.asarray(labels[pair_id]).reshape(-1))
    prediction = accumulator.result()
    if prediction["overall"]["pixels"] != TOTAL_PIXELS:
        raise ProbeError("full n600 pixel count drifted")

    common = {
        "schema": "regmax_family_probe_receipt.v1",
        "created_at_utc": args.timestamp,
        "git_head": _git_head(),
        "axis": "[macOS-CPU advisory] target-cache analysis; no score authority",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "inputs": {
            "logits": {"path": str(logits_path), "bytes": LOGITS_BYTES, "sha256": LOGITS_SHA256},
            "hard_labels": {"path": str(labels_path), "bytes": LABELS_BYTES, "sha256": LABELS_SHA256},
            "shape": [N_PAIRS, N_CLASSES, HEIGHT, WIDTH],
            "scale": UNIT_SCALE,
            "registry": {"path": str(registry_path), "sha256": _sha256(registry_path)},
        },
        "registered_surface_presence": registrations,
    }

    sparse_blockers = []
    if not registrations["preimage_adapter"]:
        sparse_blockers.append(
            f"missing canonical registration {REQUIRED_REGISTRATIONS['preimage_adapter']}"
        )
    sparse_partial = {
        "status": "MEASURED_TARGET_SURFACE_ONLY",
        "target_debt_definition": "mean(1 - p[target_hard_label])",
        "strata": {
            "high_margin_interior_ge_1": "fp16 top1-top2 margin >= unit scale",
            "boundary_annulus_or_tie_lt_1": "fp16 top1-top2 margin < unit scale",
        },
        "prediction_maps": prediction,
        "preregistered_exact_one_hot_prediction": 0.9733,
        "prediction_absolute_error": abs(
            prediction["overall"]["sparsemax_exact_one_hot_fraction"] - 0.9733
        ),
        "authority_boundary": (
            "target-only; cached labels are not a decoded candidate HARD_ACCEPT and no "
            "receiver/byte benefit is claimed"
        ),
    }
    sparse_receipt = _n_a_receipt(
        probe_id=SPARSEMAX_PROBE,
        falsifier=SPARSEMAX_FALSIFIER,
        blockers=sparse_blockers,
        common=common,
        partial_measurement=sparse_partial,
    )

    tropical_blockers = [
        f"missing canonical registration {REQUIRED_REGISTRATIONS[name]}"
        for name in ("aurenhammer_comparator", "principal_cell_fixture")
        if not registrations[name]
    ]
    tropical_receipt = _n_a_receipt(
        probe_id=TROPICAL_PROBE,
        falsifier=TROPICAL_FALSIFIER,
        blockers=tropical_blockers,
        common=common,
        partial_measurement={
            "status": "NOT_RUN_NO_MATCHED_COMPARATOR",
            "reason": (
                "The branch has the exact rank-4 head law but no registered A,b principal-cell "
                "fixture and no Aurenhammer same-coder comparator. Constructing either here "
                "would redesign the preregistered probe."
            ),
        },
    )

    hopfield_blockers = [
        f"missing canonical registration {REQUIRED_REGISTRATIONS[name]}"
        for name in ("valid_cell_prototypes", "preimage_adapter")
        if not registrations[name]
    ]
    hopfield_receipt = _n_a_receipt(
        probe_id=HOPFIELD_PROBE,
        falsifier=HOPFIELD_FALSIFIER,
        blockers=hopfield_blockers,
        common=common,
        partial_measurement={
            "status": "NOT_RUN_NO_FROZEN_PROTOTYPES_OR_MATCHED_PREIMAGE",
            "reason": (
                "No frozen rank-4 valid-cell prototype artifact or typed logits-to-uint8 "
                "adapter is registered; deriving prototypes from this measurement would "
                "change the as-written treatment."
            ),
        },
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = [sparse_receipt, tropical_receipt, hopfield_receipt]
    manifest_rows = []
    for receipt in receipts:
        path = output_dir / f"{receipt['probe_id']}.json"
        payload = _canonical_json(receipt)
        path.write_bytes(payload)
        manifest_rows.append(
            {"probe_id": receipt["probe_id"], "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = {
        **common,
        "schema": "regmax_family_probe_manifest.v1",
        "elapsed_seconds": time.monotonic() - started,
        "receipts": manifest_rows,
        "verdicts": {receipt["probe_id"]: receipt["verdict"] for receipt in receipts},
        "overall_verdict": "N-A_MATCHED_RECEIVER_SURFACES_ABSENT",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_payload = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_payload)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest_sha256={hashlib.sha256(manifest_payload).hexdigest()}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timestamp",
        default=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    return parser


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
