#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the preregistered regularized-max probes against custodied surfaces.

The tool consumes the real n600 logit/label cache and the three SHA-bound
prerequisite receipts.  It executes every composition that those surfaces
actually define and fails closed where the class/feature-space treatments still
lack a typed pullback to RGB bytes.  In particular, a generic bounded RGB
preimage adapter is not silently treated as a class-logit-to-RGB inverse.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.prereq_surfaces import (  # noqa: E402
    MATCHED_ADAPTER_SCHEMA,
    PROTOTYPE_BANK_SCHEMA,
    build_frozen_rank4_prototype_bank,
    compare_affine_cell_representatives_same_coder,
)

DEFAULT_SURFACE_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
)
DEFAULT_LOGITS = DEFAULT_SURFACE_ROOT / "teacher_logits_n600/gt_segnet_logits.f16"
DEFAULT_LABELS = DEFAULT_SURFACE_ROOT / "targets_n600/gt_segnet_argmax.u8"
DEFAULT_PREREQ_DIR = REPO / ".omx/research/prereq_surfaces_flush_20260720"
DEFAULT_WEIGHTS = Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors")

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

PULLBACK_BLOCKER = "MISSING_TYPED_CLASS_OR_RANK4_FEATURE_TO_RGB_PREIMAGE_PULLBACK"
PULLBACK_SCOPE = (
    "matched class-logit/rank-4 treatment -> camera RGB bytes -> parse-back -> "
    "fresh frozen CPU-Torch SegNet HARD_ACCEPT composition only; individual "
    "prerequisite surfaces remain valid"
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


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _load_prerequisite_receipts(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read prerequisite manifest: {manifest_path}") from exc
    if manifest.get("schema") != "prereq_surfaces_flush_receipt_manifest.v1":
        raise ProbeError("prerequisite manifest schema mismatch")
    declared = manifest.get("receipts")
    if not isinstance(declared, dict):
        raise ProbeError("prerequisite manifest receipt map is missing")
    receipts: dict[str, Any] = {}
    for filename, custody in declared.items():
        path = root / filename
        if (
            not isinstance(custody, dict)
            or not path.is_file()
            or path.stat().st_size != custody.get("bytes")
            or _sha256(path) != custody.get("sha256")
        ):
            raise ProbeError(f"prerequisite receipt custody mismatch: {filename}")
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbeError(f"prerequisite receipt is not ASCII JSON: {filename}") from exc
        if not isinstance(value, dict):
            raise ProbeError(f"prerequisite receipt must be a mapping: {filename}")
        receipts[filename] = value
    surface1 = receipts.get("surface_1_matched_preimage_adapter.json")
    surface2 = receipts.get("surface_2_rank4_prototype_bank.json")
    surface3 = receipts.get("surface_3_same_coder_comparator.json")
    if not isinstance(surface1, dict) or surface1.get("schema") != MATCHED_ADAPTER_SCHEMA:
        raise ProbeError("surface-1 matched adapter receipt is absent or malformed")
    if not isinstance(surface2, dict) or surface2.get("schema") != PROTOTYPE_BANK_SCHEMA:
        raise ProbeError("surface-2 prototype receipt is absent or malformed")
    if not isinstance(surface3, dict) or not surface3.get("same_coder"):
        raise ProbeError("surface-3 same-coder receipt is absent or malformed")
    return {
        "root": str(root),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "surface_1": surface1,
        "surface_2": surface2,
        "surface_3": surface3,
    }


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


def _composition_blocked_receipt(
    *,
    probe_id: str,
    falsifier: str,
    common: dict[str, Any],
    decomposition: dict[str, Any],
    treatment_space: str,
) -> dict[str, Any]:
    return {
        **common,
        "probe_id": probe_id,
        "verdict": "BLOCKED_NOT_MEASURED",
        "verdict_scope": PULLBACK_SCOPE,
        "falsifier": falsifier,
        "falsifier_evaluated": False,
        "blockers": [
            {
                "code": PULLBACK_BLOCKER,
                "treatment_space": treatment_space,
                "adapter_input_space": "camera RGB or scorer-plane RGB continuous proposal",
                "missing_map": "treatment space to camera/scorer-plane RGB continuous proposal",
                "reason": (
                    "surface 1 preserves a supplied RGB proposal through the exact bounded "
                    "uint8 solve, but it does not derive RGB from class probabilities or "
                    "rank-4 SegNet-head features"
                ),
                "verdict_scope": PULLBACK_SCOPE,
            }
        ],
        "hard_cpu_torch_oracle": {
            "authority": "fresh frozen SegNet on parsed RGB bytes",
            "invoked": False,
            "reason": "no well-typed RGB candidate exists for this treatment",
        },
        "hard_accepts": None,
        "exact_oracle_calls": None,
        "candidate_bytes_same_coder": None,
        "per_stratum_decomposition": decomposition,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    logits_path = args.logits.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    for path, expected_bytes, expected_sha, name in (
        (logits_path, LOGITS_BYTES, LOGITS_SHA256, "logits"),
        (labels_path, LABELS_BYTES, LABELS_SHA256, "hard labels"),
    ):
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ProbeError(f"real n600 {name} bytes mismatch at {path}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ProbeError(f"real n600 {name} sha256 mismatch: {actual_sha}")

    prerequisite = _load_prerequisite_receipts(args.prerequisite_dir)
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
            "prerequisite_manifest": prerequisite["manifest"],
            "frozen_segnet_weights": {
                "path": str(args.weights.expanduser().resolve(strict=True)),
                "sha256": _sha256(args.weights.expanduser().resolve(strict=True)),
            },
        },
        "surface_presence": {
            "matched_preimage_adapter": True,
            "frozen_rank4_prototype_bank": True,
            "same_coder_comparator": True,
        },
    }

    measured_fraction = prediction["overall"]["sparsemax_exact_one_hot_fraction"]
    preregistered_fraction = 0.9733309173583984
    sparse_decomposition = {
        "status": "MEASURED_N600_TARGET_SURFACE_NOT_RECEIVER_VERDICT",
        "target_debt_definition": "mean(1 - p[target_hard_label])",
        "strata": {
            "high_margin_interior_ge_1": "fp16 top1-top2 margin >= unit scale",
            "boundary_annulus_or_tie_lt_1": "fp16 top1-top2 margin < unit scale",
        },
        "prediction_maps": prediction,
        "preregistered_exact_one_hot_fraction": preregistered_fraction,
        "measured_exact_one_hot_fraction": measured_fraction,
        "delta_vs_preregistered_fraction": measured_fraction - preregistered_fraction,
        "beats_preregistered_fraction": measured_fraction > preregistered_fraction,
        "authority_boundary": (
            "target-only; cached labels are not a decoded candidate HARD_ACCEPT and no "
            "receiver/byte benefit is claimed"
        ),
    }
    sparse_receipt = _composition_blocked_receipt(
        probe_id=SPARSEMAX_PROBE,
        falsifier=SPARSEMAX_FALSIFIER,
        common=common,
        decomposition=sparse_decomposition,
        treatment_space="five-class sparsemax/Cole-Hopf probability field",
    )

    bank = build_frozen_rank4_prototype_bank(args.weights)
    comparator = compare_affine_cell_representatives_same_coder(bank)
    if comparator != prerequisite["surface_3"]:
        raise ProbeError("fresh same-coder comparator is not byte-identical to custodied surface 3")
    representatives = comparator["representatives"]
    tropical = representatives["tropical_residuation_principal"]
    aurenhammer = representatives["aurenhammer_min_generator_lp"]
    zero_sum = representatives["zero_sum_min_norm"]
    tropical_falsified = (
        not tropical["exact_cell_identity"]
        or tropical["coded_bytes"] > aurenhammer["coded_bytes"]
    )
    tropical_receipt = {
        **common,
        "probe_id": TROPICAL_PROBE,
        "verdict": "FALSIFIED_FORMULATION" if tropical_falsified else "SURVIVES_FORMULATION",
        "verdict_scope": (
            "tropical principal representative as a minimum-byte representative of the "
            "five frozen-head strict prototype cells through PDW2 plus Brotli-q11"
        ),
        "falsifier": TROPICAL_FALSIFIER,
        "falsifier_evaluated": True,
        "same_coder_comparator": comparator,
        "candidate_bytes_same_coder": {
            "tropical_principal": tropical["coded_bytes"],
            "aurenhammer_min_generator": aurenhammer["coded_bytes"],
            "zero_sum_min_norm": zero_sum["coded_bytes"],
            "tropical_minus_aurenhammer": tropical["coded_bytes"] - aurenhammer["coded_bytes"],
            "tropical_minus_zero_sum": tropical["coded_bytes"] - zero_sum["coded_bytes"],
        },
        "per_stratum_decomposition": {
            "per_frozen_cell": {
                str(class_id): {
                    "prototype_margin": float(bank.margins[class_id]),
                    "expected_class": class_id,
                    "tropical_class": tropical["prototype_cell_labels"][class_id],
                    "aurenhammer_class": aurenhammer["prototype_cell_labels"][class_id],
                    "zero_sum_class": zero_sum["prototype_cell_labels"][class_id],
                }
                for class_id in range(N_CLASSES)
            },
            "packet_bytes_are_joint": True,
            "joint_packet_rationale": "PDW2 serializes one five-cell affine complex, not separable per-cell packets",
        },
        "hard_cpu_torch_oracle": {
            "invoked": False,
            "reason": "probe 2 is exact frozen affine-head cell identity plus real coder bytes; it makes no RGB score claim",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }

    hopfield_receipt = _composition_blocked_receipt(
        probe_id=HOPFIELD_PROBE,
        falsifier=HOPFIELD_FALSIFIER,
        common=common,
        treatment_space="rank-4 frozen SegNet-head quotient features",
        decomposition={
            "status": "FROZEN_BANK_MEASURED_BUT_RECEIVER_COMPOSITION_ABSENT",
            "per_frozen_cell": {
                str(class_id): {
                    "prototype_label": int(bank.labels[class_id]),
                    "prototype_margin": float(bank.margins[class_id]),
                }
                for class_id in range(N_CLASSES)
            },
            "rank": int(bank.receipt["rank"]),
            "prototype_sha256": bank.receipt["prototype_sha256"],
        },
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = [sparse_receipt, tropical_receipt, hopfield_receipt]
    manifest_rows = []
    for receipt in receipts:
        path = output_dir / f"{receipt['probe_id']}.json"
        payload = _canonical_json(receipt)
        if path.exists():
            raise ProbeError(f"refusing to overwrite preserved receipt: {path}")
        _atomic_write(path, payload)
        manifest_rows.append(
            {"probe_id": receipt["probe_id"], "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = {
        **common,
        "schema": "regmax_family_probe_manifest.v1",
        "elapsed_seconds": time.monotonic() - started,
        "receipts": manifest_rows,
        "verdicts": {receipt["probe_id"]: receipt["verdict"] for receipt in receipts},
        "overall_verdict": "P2_FALSIFIED_PRINCIPAL_MIN_BYTE_P1_P3_BLOCKED_ON_TYPED_RGB_PULLBACK",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_payload = _canonical_json(manifest)
    if manifest_path.exists():
        raise ProbeError(f"refusing to overwrite preserved receipt: {manifest_path}")
    _atomic_write(manifest_path, manifest_payload)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest_sha256={hashlib.sha256(manifest_payload).hexdigest()}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--prerequisite-dir", type=Path, default=DEFAULT_PREREQ_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
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
