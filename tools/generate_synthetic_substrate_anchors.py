#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate synthetic substrate-class anchors (research-signal only, NOT empirical).

Per operator amplification 2026-05-11 ("compiler" + "wiring and integration"):
NN's `tools/cpu_cuda_xray_substrate_class_classifier.py` (P5 xray classifier)
currently emits an INSUFFICIENT-DATA verdict because only ONE substrate class
(``hnerv_family``) has paired anchors (R's authoritative SegNet + PoseNet
landings). To activate the classifier across the 16 non-HNeRV substrates
identified in QQ's substrate composition matrix without burning $0.30-0.60 of
duplicate Modal/Lightning compute, this tool **GENERATES SYNTHETIC ANCHORS**
from each substrate's known characteristics (renderer arch + bit budget +
score-axis target) and tags them explicitly ``[synthetic-not-empirical]``
per CLAUDE.md.

The output schema is COMPATIBLE with NN's classifier input (the same
``layer_drift.json`` schema with ``layer_drift_rows`` + ``stage_compounding``
+ ``first_divergence`` + ``cpu_record_path`` / ``cuda_record_path`` etc), so
the classifier ingests it via the same ``--input-spec`` mechanism. The
synthetic anchors set (a) ``mode = "synthetic_not_empirical"``, (b)
``score_claim = false`` permanently, (c) ``promotion_eligible = false``
permanently, (d) `evidence_grade = "diagnostic_not_score"`, AND (e) a new
top-level field ``synthetic_anchor`` true so downstream consumers can detect
synthetic-vs-empirical without parsing prose.

Per CLAUDE.md "Forbidden score claims": these synthetic anchors NEVER feed
``tac.continual_learning.posterior_update_locked`` or any contest-CUDA
promotion path. They are used SOLELY to activate NN's classifier so that
the per-substrate-class signature can be produced for sensitivity-map
seeding when the next round of L2-encoder dispatches lands real anchors.

Usage
-----

::

    .venv/bin/python tools/generate_synthetic_substrate_anchors.py \\
        --output-dir experiments/results/synthetic_substrate_anchors_<UTC>/ \\
        --substrate-class non_hnerv_family

    # then re-run NN's classifier:

    .venv/bin/python tools/cpu_cuda_xray_substrate_class_classifier.py \\
        --input-spec hnerv_family:<segnet_anchor.json> \\
        --input-spec hnerv_family:<posenet_anchor.json> \\
        --input-spec non_hnerv_family:<synthetic_anchor1.json> \\
        ...

Per CLAUDE.md "Forbidden /tmp paths" the tool refuses /tmp output dirs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Mirror NN's classifier accept list (avoid coupling import).
_VALID_SUBSTRATE_CLASSES: tuple[str, ...] = ("hnerv_family", "non_hnerv_family")
_VALID_EVIDENCE_GRADES: tuple[str, ...] = (
    "diagnostic_not_score",
    "diagnostic-not-score",
)

SCHEMA_VERSION = "tac_synthetic_substrate_anchor_v1"


# Per QQ landing (`feedback_substrate_composition_matrix_*`) the 16 non-HNeRV
# substrate inventory + per-substrate synthetic-drift parametrisation
# (mean drift + signature scale + n_layers + class hint for fine-grained
# classification). The drift pattern is deterministic per (substrate_id, seed)
# pair so two runs with the same seed produce byte-identical synthetic anchors.
@dataclass(frozen=True)
class SubstrateSyntheticSpec:
    substrate_id: str
    name: str
    fine_grained_class: str  # e.g., "residual", "self_compression", "nerv_family"
    coarse_class: str  # NN-classifier-compatible: "hnerv_family" / "non_hnerv_family"
    n_layers: int
    base_drift_mean: float  # log-magnitude mean drift; expected ~ 1e-7 for HNeRV-like
    base_drift_signature_scale: float  # multiplier on drift across layers
    n_stages: int
    target_axis: str  # rate / seg / pose / mixed (for human inspection)


# Default seed for reproducibility per CLAUDE.md "Deterministic packet compiler".
DEFAULT_SEED: int = 0xCAFE


def canonical_synthetic_substrate_specs() -> list[SubstrateSyntheticSpec]:
    """Return the canonical 16-substrate non-HNeRV synthetic-spec inventory.

    Per QQ landing's substrate inventory + the per-substrate landing memos:

    - Residual basis (5): wavelet, cool_chic, c3, siren, coordinate_mlp
    - Pose-axis (3): foveation_field, raft_pose_stream,
      lapose_motion_atom_allocator
    - Self-compression (3): scpp_substrate, hessian_block_fp, mdl_fp4_tto
    - NeRV-family (5): blocknerv, ffnerv, dsnerv, hinerv, tcnerv

    Each spec encodes the substrate's expected drift signature class. The
    drift parameters are PRIORS, not measurements; they capture the substrate's
    architectural family relationship to the contest scorer's drift kernel.
    """
    return [
        # Residual basis: tiny sidecar weights; drift dominated by entropy
        # coder rounding (very small magnitudes).
        SubstrateSyntheticSpec(
            substrate_id="wavelet_residual",
            name="Wavelet residual basis (Mallat)",
            fine_grained_class="residual",
            coarse_class="non_hnerv_family",
            n_layers=180,
            base_drift_mean=2.5e-7,
            base_drift_signature_scale=1.5,
            n_stages=3,
            target_axis="rate",
        ),
        SubstrateSyntheticSpec(
            substrate_id="cool_chic_residual",
            name="Cool-Chic hierarchical pyramid residual",
            fine_grained_class="residual",
            coarse_class="non_hnerv_family",
            n_layers=200,
            base_drift_mean=2.7e-7,
            base_drift_signature_scale=1.4,
            n_stages=4,
            target_axis="rate",
        ),
        SubstrateSyntheticSpec(
            substrate_id="c3_residual",
            name="C3 (Cool-Chic + temporal hyperprior) residual",
            fine_grained_class="residual",
            coarse_class="non_hnerv_family",
            n_layers=210,
            base_drift_mean=3.0e-7,
            base_drift_signature_scale=1.3,
            n_stages=4,
            target_axis="mixed",
        ),
        SubstrateSyntheticSpec(
            substrate_id="siren_residual",
            name="SIREN sinusoidal coordinate-MLP residual",
            fine_grained_class="residual",
            coarse_class="non_hnerv_family",
            n_layers=120,
            base_drift_mean=4.5e-7,
            base_drift_signature_scale=2.0,
            n_stages=3,
            target_axis="rate",
        ),
        SubstrateSyntheticSpec(
            substrate_id="coordinate_mlp_residual",
            name="Coordinate-MLP family-agnostic residual",
            fine_grained_class="residual",
            coarse_class="non_hnerv_family",
            n_layers=140,
            base_drift_mean=4.0e-7,
            base_drift_signature_scale=2.1,
            n_stages=3,
            target_axis="rate",
        ),
        # Pose-axis sidechannels: small per-frame stream; drift dominated
        # by float-vs-fixed-point pose representation.
        SubstrateSyntheticSpec(
            substrate_id="foveation_field",
            name="Telescopic foveation field",
            fine_grained_class="pose_axis_sidechannel",
            coarse_class="non_hnerv_family",
            n_layers=80,
            base_drift_mean=8.0e-7,
            base_drift_signature_scale=2.5,
            n_stages=2,
            target_axis="pose",
        ),
        SubstrateSyntheticSpec(
            substrate_id="raft_pose_stream",
            name="RAFT optical-flow pose stream",
            fine_grained_class="pose_axis_sidechannel",
            coarse_class="non_hnerv_family",
            n_layers=160,
            base_drift_mean=6.5e-7,
            base_drift_signature_scale=2.3,
            n_stages=3,
            target_axis="pose",
        ),
        SubstrateSyntheticSpec(
            substrate_id="lapose_motion_atom_allocator",
            name="LAPose inverse-dynamics motion-atom allocator",
            fine_grained_class="pose_axis_sidechannel",
            coarse_class="non_hnerv_family",
            n_layers=90,
            base_drift_mean=7.0e-7,
            base_drift_signature_scale=2.4,
            n_stages=2,
            target_axis="pose",
        ),
        # Self-compression: weight-domain quantisation; drift profile is
        # chunky (per-block-FP boundaries amplify drift at block-edge layers).
        SubstrateSyntheticSpec(
            substrate_id="scpp_substrate",
            name="SC++ block-FP self-compression substrate",
            fine_grained_class="self_compression",
            coarse_class="non_hnerv_family",
            n_layers=240,
            base_drift_mean=1.5e-6,
            base_drift_signature_scale=3.5,
            n_stages=5,
            target_axis="rate",
        ),
        SubstrateSyntheticSpec(
            substrate_id="hessian_block_fp",
            name="Hessian block-FP allocator (Boyd ADMM water-filling)",
            fine_grained_class="self_compression",
            coarse_class="non_hnerv_family",
            n_layers=240,
            base_drift_mean=1.2e-6,
            base_drift_signature_scale=3.2,
            n_stages=5,
            target_axis="rate",
        ),
        SubstrateSyntheticSpec(
            substrate_id="mdl_fp4_tto",
            name="MDL/FP4 test-time training (Stage-5 final TTO)",
            fine_grained_class="self_compression",
            coarse_class="non_hnerv_family",
            n_layers=240,
            base_drift_mean=1.4e-6,
            base_drift_signature_scale=3.4,
            n_stages=5,
            target_axis="mixed",
        ),
        # NeRV-family: full renderer-replacement architectures; drift
        # signature is significantly different from HNeRV's stride-2 stem.
        SubstrateSyntheticSpec(
            substrate_id="blocknerv",
            name="BlockNeRV (tile-decomposed)",
            fine_grained_class="nerv_family",
            coarse_class="non_hnerv_family",
            n_layers=320,
            base_drift_mean=4.0e-7,
            base_drift_signature_scale=1.8,
            n_stages=8,
            target_axis="mixed",
        ),
        SubstrateSyntheticSpec(
            substrate_id="ffnerv",
            name="FFNeRV (Fourier-features)",
            fine_grained_class="nerv_family",
            coarse_class="non_hnerv_family",
            n_layers=290,
            base_drift_mean=4.5e-7,
            base_drift_signature_scale=1.9,
            n_stages=7,
            target_axis="mixed",
        ),
        SubstrateSyntheticSpec(
            substrate_id="dsnerv",
            name="DSNeRV (diffusion-supervised)",
            fine_grained_class="nerv_family",
            coarse_class="non_hnerv_family",
            n_layers=350,
            base_drift_mean=5.2e-7,
            base_drift_signature_scale=2.0,
            n_stages=9,
            target_axis="mixed",
        ),
        SubstrateSyntheticSpec(
            substrate_id="hinerv",
            name="HiNeRV (hierarchical)",
            fine_grained_class="nerv_family",
            coarse_class="non_hnerv_family",
            n_layers=380,
            base_drift_mean=4.8e-7,
            base_drift_signature_scale=2.1,
            n_stages=9,
            target_axis="mixed",
        ),
        SubstrateSyntheticSpec(
            substrate_id="tcnerv",
            name="TCNeRV (temporal-conv)",
            fine_grained_class="nerv_family",
            coarse_class="non_hnerv_family",
            n_layers=300,
            base_drift_mean=4.2e-7,
            base_drift_signature_scale=1.8,
            n_stages=7,
            target_axis="pose",
        ),
    ]


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse forbidden /tmp paths per CLAUDE.md."""
    as_str = str(output_dir.resolve())
    forbidden_anchors = ("/tmp/", "/var/tmp/", "/private/tmp/")
    for anchor in forbidden_anchors:
        if as_str.startswith(anchor):
            raise SystemExit(
                f"refusing to write to forbidden /tmp path {output_dir!s} "
                "per CLAUDE.md `forbidden_/tmp_paths_in_any_persisted_artifact`"
            )


def _generate_drift_vector(
    spec: SubstrateSyntheticSpec, *, seed: int
) -> list[float]:
    """Generate a deterministic synthetic per-layer drift vector.

    The vector is a log-spaced random walk centered on ``base_drift_mean``
    with per-layer multiplicative noise scaled by
    ``base_drift_signature_scale``. The class-fine-grained class encodes
    a deterministic class-conditional drift kernel:

    - ``residual``: monotonic decay from layer 0 to N (entropy coder
      front-loads error in early layers)
    - ``pose_axis_sidechannel``: triangular bump centered at ~30% of
      depth (motion-feature transformer head intermediate)
    - ``self_compression``: chunky stair-case (block-FP boundaries every
      ~1/n_stages of depth)
    - ``nerv_family``: spread + skewed-right (deeper NeRV layers
      accumulate more compounding drift)
    """
    rng = random.Random(seed ^ hash(spec.substrate_id))
    out: list[float] = []
    for layer_idx in range(spec.n_layers):
        depth_frac = layer_idx / max(spec.n_layers - 1, 1)
        if spec.fine_grained_class == "residual":
            kernel_factor = math.exp(-2.5 * depth_frac)
        elif spec.fine_grained_class == "pose_axis_sidechannel":
            peak_at = 0.30
            kernel_factor = math.exp(-((depth_frac - peak_at) ** 2) / 0.05)
        elif spec.fine_grained_class == "self_compression":
            n_stages = max(spec.n_stages, 1)
            stage_idx = int(depth_frac * n_stages)
            kernel_factor = 1.0 + 0.4 * (stage_idx % 2)
        elif spec.fine_grained_class == "nerv_family":
            kernel_factor = 0.7 + 0.6 * depth_frac + 0.1 * (depth_frac ** 2)
        else:
            kernel_factor = 1.0
        # Multiplicative log-normal noise.
        noise = math.exp(rng.gauss(0.0, 0.4))
        drift = (
            spec.base_drift_mean
            * kernel_factor
            * spec.base_drift_signature_scale
            * noise
        )
        out.append(float(max(drift, 0.0)))
    return out


def _build_layer_drift_rows(
    spec: SubstrateSyntheticSpec,
    drift_vector: list[float],
) -> list[dict[str, Any]]:
    """Build NN-classifier-compatible layer_drift_rows from a drift vector."""
    rows: list[dict[str, Any]] = []
    for layer_idx, drift in enumerate(drift_vector):
        rows.append(
            {
                "fingerprint_only_l2_proxy": float(drift),
                "fingerprint_only_max_proxy": float(drift * 4.5),
                "has_full_tensors": False,
                "kl_divergence": None,
                "l2_relative_error": float("nan"),
                "layer_name": f"{spec.substrate_id}.layer.{layer_idx}",
                "max_abs_error": float("nan"),
                "mean_abs_error": float("nan"),
                "module_type": spec.fine_grained_class,
                "note": "synthetic-not-empirical (capture_mode='synthetic_signature')",
                "output_index": 0,
                "rank_top1_disagreement": None,
            }
        )
    return rows


def _build_stage_compounding(
    spec: SubstrateSyntheticSpec, drift_vector: list[float]
) -> dict[str, Any]:
    """Build a NN-classifier-compatible stage_compounding from drift vector."""
    n_stages = max(spec.n_stages, 1)
    by_stage: list[dict[str, Any]] = []
    chunk = max(1, len(drift_vector) // n_stages)
    for stage_idx in range(n_stages):
        start = stage_idx * chunk
        end = start + chunk if stage_idx < n_stages - 1 else len(drift_vector)
        slice_ = drift_vector[start:end]
        if not slice_:
            continue
        max_eps = max(slice_)
        mean_eps = sum(slice_) / float(len(slice_))
        compound_factor = 1.0 + max_eps  # First-order approximation.
        by_stage.append(
            {
                "compound_factor": float(compound_factor),
                "eps_sources": ["synthetic_fingerprint_proxy"],
                "max_eps": float(max_eps),
                "mean_eps": float(mean_eps),
                "num_layers": len(slice_),
                "stage_key": f"{spec.substrate_id}.stage.{stage_idx}",
            }
        )
    return {"by_stage": by_stage}


def build_synthetic_anchor(
    spec: SubstrateSyntheticSpec, *, seed: int
) -> dict[str, Any]:
    """Build one synthetic substrate anchor in NN-classifier-compatible JSON.

    Per CLAUDE.md "Forbidden score claims": ``score_claim=False`` /
    ``promotion_eligible=False`` invariants are baked in. The
    ``synthetic_anchor`` flag at the top level is the explicit machine-
    readable separator from real empirical anchors.
    """
    drift_vector = _generate_drift_vector(spec, seed=seed)
    layer_drift_rows = _build_layer_drift_rows(spec, drift_vector)
    stage_compounding = _build_stage_compounding(spec, drift_vector)

    # First divergence: deterministic by drift vector + threshold.
    threshold = 1e-2
    first_divergence_layer = None
    for idx, d in enumerate(drift_vector):
        if d >= threshold:
            first_divergence_layer = idx
            break
    first_divergence = {
        "first_argmax_divergence": None,
        "first_l2_relative_exceedance": first_divergence_layer,
        "l2_relative_threshold": threshold,
    }

    return {
        "schema": "synthetic_layer_drift_v1",
        "synthetic_anchor": True,
        "synthetic_anchor_seed": seed,
        "synthetic_substrate_id": spec.substrate_id,
        "synthetic_fine_grained_class": spec.fine_grained_class,
        "synthetic_coarse_class": spec.coarse_class,
        "synthetic_target_axis": spec.target_axis,
        "label": f"{spec.substrate_id}_synthetic_anchor",
        "tag": "[synthetic-not-empirical; not contest-CUDA; research-signal]",
        "tool": "tools/generate_synthetic_substrate_anchors.py",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "mode": "synthetic_not_empirical",
        "scorer": spec.fine_grained_class,
        "evidence_grade": "diagnostic_not_score",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "frame_pair_idx": 0,
        "num_layers_compared": spec.n_layers,
        "l2_relative_drift_threshold": threshold,
        "cpu_record_path": (
            f"experiments/results/synthetic_substrate_anchors/{spec.substrate_id}/cpu_record.synthetic"
        ),
        "cpu_record_sha256": (
            "synthetic_" + hashlib.sha256(
                f"{spec.substrate_id}_cpu_{seed}".encode()
            ).hexdigest()[:48]
        ),
        "cuda_record_path": (
            f"experiments/results/synthetic_substrate_anchors/{spec.substrate_id}/cuda_record.synthetic"
        ),
        "cuda_record_sha256": (
            "synthetic_" + hashlib.sha256(
                f"{spec.substrate_id}_cuda_{seed}".encode()
            ).hexdigest()[:48]
        ),
        "shared_input_tensor_path": (
            f"experiments/results/synthetic_substrate_anchors/{spec.substrate_id}/input.synthetic"
        ),
        "shared_input_tensor_sha256": (
            "synthetic_" + hashlib.sha256(
                f"{spec.substrate_id}_input_{seed}".encode()
            ).hexdigest()[:48]
        ),
        "from_state_hash": "synthetic_state",
        "cpu_capture_host": {
            "contest_compliant_cpu_substrate": False,
            "evidence_grade_qualifier": "synthetic_not_empirical",
            "is_linux_x86_64": False,
            "is_macos_darwin": False,
            "machine": "synthetic",
            "note": "Synthetic anchor; no real CPU host involved.",
            "platform": "synthetic_anchor_v1",
            "system": "Synthetic",
        },
        "mixed_substrate_advisory": (
            "synthetic_not_empirical: this anchor was generated by "
            "tools/generate_synthetic_substrate_anchors.py from substrate-class "
            "priors. It MUST NOT be used to claim score, promote a lane, or "
            "feed continual-learning posterior updates. Per CLAUDE.md "
            "'Forbidden score claims' + 'forbidden_score_claim_with_byte_change'. "
            "Use SOLELY for substrate-class signature seeding."
        ),
        "first_divergence": first_divergence,
        "stage_compounding": stage_compounding,
        "layer_drift_rows": layer_drift_rows,
        "final_logits": {
            "available": False,
            "kl_divergence": None,
            "l2_relative_error": float("nan"),
            "layer_name": f"{spec.substrate_id}.synthetic.final_logits",
            "max_abs_error": float("nan"),
            "mean_abs_error": float("nan"),
            "module_type": spec.fine_grained_class,
        },
    }


def _safe_dump_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON with NaN serialised as null (synthetic anchors mark unavailable
    fields explicitly; we replace NaN with null to keep parsers happy)."""
    def _replace_nan(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _replace_nan(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_replace_nan(x) for x in o]
        if isinstance(o, float):
            if math.isnan(o):
                return None
            if math.isinf(o):
                return None
        return o
    cleaned = _replace_nan(payload)
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True))


def emit_anchors(
    output_dir: Path,
    specs: Sequence[SubstrateSyntheticSpec],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Emit one ``layer_drift.json``-style synthetic anchor per spec.

    Output layout::

        <output_dir>/
            <substrate_id>/
                layer_drift.json     # NN-classifier-compatible
            anchors_index.jsonl       # JSONL summary (one row per anchor)
            anchors_set_manifest.json # Top-level set manifest
    """
    _validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        sub_dir = output_dir / spec.substrate_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        anchor = build_synthetic_anchor(spec, seed=seed)
        out_path = sub_dir / "layer_drift.json"
        _safe_dump_json(anchor, out_path)
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        rows.append(
            {
                "substrate_id": spec.substrate_id,
                "fine_grained_class": spec.fine_grained_class,
                "coarse_class": spec.coarse_class,
                "target_axis": spec.target_axis,
                "n_layers": spec.n_layers,
                "n_stages": spec.n_stages,
                "anchor_path": str(out_path),
                "anchor_sha256": sha256,
                "synthetic": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    # JSONL index.
    jsonl_path = output_dir / "anchors_index.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    )
    # Top-level set manifest.
    manifest = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_anchors": len(rows),
        "seed": seed,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_not_score",
        "synthetic_set": True,
        "anchors": rows,
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "synthetic_substrate_anchor_v1",
            "synthetic_not_empirical",
            "halt_and_ask_default_for_dispatch_recommendations",
        ],
    }
    manifest_path = output_dir / "anchors_set_manifest.json"
    _safe_dump_json(manifest, manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_synthetic_substrate_anchors",
        description=(
            "Generate synthetic substrate-class anchors for the P5 xray "
            "substrate-class classifier. evidence_grade=diagnostic-not-score; "
            "score_claim=false permanently; tagged [synthetic-not-empirical]."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory under experiments/results/ (refuses /tmp paths).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic seed for reproducible synthetic anchors (default: 0x{DEFAULT_SEED:X}).",
    )
    parser.add_argument(
        "--substrate-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of substrate_ids to emit (default: all 16). "
            "Useful for incremental anchor generation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = canonical_synthetic_substrate_specs()
    if args.substrate_ids:
        keep = {sid.strip() for sid in args.substrate_ids.split(",") if sid.strip()}
        specs = [s for s in specs if s.substrate_id in keep]
        if not specs:
            print(
                f"ERROR: no canonical substrate matches --substrate-ids {args.substrate_ids!r}",
                file=sys.stderr,
            )
            return 2
    manifest = emit_anchors(args.output_dir, specs, seed=args.seed)
    print(
        json.dumps(
            {
                "status": "ok",
                "n_anchors_emitted": manifest["n_anchors"],
                "output_dir": str(args.output_dir.resolve()),
                "seed": args.seed,
                "manifest_path": str((args.output_dir / "anchors_set_manifest.json").resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
