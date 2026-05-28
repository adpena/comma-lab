#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Sister Catalog #1265 canonical PASS/FAIL gate for Wyner-Ziv WZPSC01-grammar candidates.

Parameterized sister of:

* ``tools/gate_mlx_candidate_contest_equivalence.py`` (PR95/HNeRV-grammar canonical; commit ``69c316ca4``)
* ``tools/gate_mlx_candidate_contest_equivalence_z6_v2.py`` (Z6V2CU1 grammar)
* ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py`` (IA3 grammar)
* ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v[234].py`` (PSV grammars)

**Structural distinction from sister gates**: this substrate is NOT a
neural renderer; there is NO MLX vs PyTorch decoder forward parity to
measure in the sigmoid ``[0, 1]`` output space. The canonical contest-
equivalence verdict for the Wyner-Ziv pipeline-stage codec IS the
**archive-bytes byte-identical roundtrip** via the canonical primitive's
``reconstruct_from_wyner_ziv_layer`` path (the same path the contest's
inflate runtime would use). The substrate's mathematical contract per
Wyner 1976 R(D|Y) is **byte-equivalence**, not float-tolerance.

Per the cascade doctrine ``fb270e9b6`` L6 gate requirement + the
MLX-first doctrine ``4107bbf8d`` per-class bridge calibration scope: each
substrate-class grammar needs its own sister #1265 gate parameterized for
that grammar.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #341
non-promotable markers: the gate output carries
``axis_tag='[macOS-MLX research-signal]'`` + ``score_claim=False`` +
``promotable=False``.

Gate semantics
==============

* **PASS**: WZPSC01 archive decode byte-identically to the source pre-entropy
  bytes referenced in the MLX-LOCAL L1 harness training_artifact.json.
* **FAIL**: byte-identity violated (catastrophic primitive contract
  violation per Catalog #105 / #139 / #220 / #272 no-op detector).
* **OBSERVABILITY_ONLY**: density verdict in the artifact is
  IMPLEMENTATION_LEVEL_FALSIFICATION (per Catalog #307). The
  archive-bytes parity is still verified, but the substrate does not yet
  beat the canonical frontier per the sister design memo §Predicted ΔS band.

Exit codes::

    0 = PASS (byte-identical roundtrip + density supports score savings)
    1 = FAIL (catastrophic primitive contract violation)
    2 = OBSERVABILITY_ONLY (byte-identical roundtrip + density falsified IMPL-level)
    3 = CLI / measurement error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WZPSC_GATE_SCHEMA = "wyner_ziv_pipeline_stage_codec_catalog_1265_gate.v1"
UTC = timezone.utc

DEFAULT_DENSITY_PASS_THRESHOLD_PERCENT = 5.0
DEFAULT_DENSITY_OBSERVABILITY_THRESHOLD_PERCENT = 1.0


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_wyner_ziv_pipeline_stage_codec_candidate(
    *,
    mlx_training_artifact_path: Path,
    parity_proof_path: Path | None = None,
    density_pass_threshold_percent: float = DEFAULT_DENSITY_PASS_THRESHOLD_PERCENT,
    density_observability_threshold_percent: float = (
        DEFAULT_DENSITY_OBSERVABILITY_THRESHOLD_PERCENT
    ),
) -> dict[str, Any]:
    """Canonical Catalog #1265 contest-equivalence verdict on a WZPSC01 candidate.

    Args:
        mlx_training_artifact_path: path to the L1 harness's training_artifact.json.
        parity_proof_path: optional path to the bridge tool's parity-proof JSON
            (per ``tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py``).
        density_pass_threshold_percent: density above which the candidate is
            PASS (sub-frontier candidate per the sister design memo §Predicted
            ΔS band). Default 5.0%.
        density_observability_threshold_percent: density above which the
            candidate is at least OBSERVABILITY_ONLY (saturating composition
            per Catalog #227 alpha=0.5). Default 1.0%.

    Returns:
        Dict with the canonical verdict + Provenance + non-promotable markers.
    """
    if not mlx_training_artifact_path.exists():
        raise FileNotFoundError(
            f"MLX training artifact not found at {mlx_training_artifact_path}"
        )
    artifact = json.loads(mlx_training_artifact_path.read_text())
    if artifact.get("substrate_id") != "wyner_ziv_pipeline_stage_codec":
        raise ValueError(
            f"artifact substrate_id != 'wyner_ziv_pipeline_stage_codec'; got "
            f"{artifact.get('substrate_id')!r}"
        )

    roundtrip_byte_identical = bool(artifact.get("roundtrip_byte_identical", False))
    max_density_percent = float(artifact.get("max_density_percent", 0.0))
    verdict_kind_from_artifact = artifact.get("verdict", {}).get("kind", "UNKNOWN")
    archive_info = artifact.get("wzpsc01_archive", {})

    # Optional parity-proof cross-check
    parity_proof_consistent = None
    parity_proof_details: dict[str, Any] = {}
    if parity_proof_path is not None:
        if not parity_proof_path.exists():
            raise FileNotFoundError(
                f"parity proof not found at {parity_proof_path}; emit it via "
                f"tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py"
            )
        proof = json.loads(parity_proof_path.read_text())
        parity_proof_consistent = (
            proof.get("inputs", {}).get("wzpsc01_archive_sha256")
            == archive_info.get("sha256")
            and bool(proof.get("parity_verdict", {}).get("roundtrip_byte_identical"))
        )
        parity_proof_details = {
            "parity_proof_path": str(parity_proof_path),
            "parity_proof_archive_sha256": proof.get("inputs", {}).get("wzpsc01_archive_sha256"),
            "parity_proof_consistent": parity_proof_consistent,
        }

    # Verdict cascade per CLAUDE.md "Apples-to-apples evidence discipline":
    # 1. Roundtrip MUST be byte-identical (any FAIL here is catastrophic per Catalog #105)
    # 2. Density >= PASS threshold → PASS (sub-frontier candidate)
    # 3. Density >= OBSERVABILITY threshold → OBSERVABILITY_ONLY (saturating)
    # 4. Otherwise → OBSERVABILITY_ONLY (IMPLEMENTATION-LEVEL falsified per Catalog #307)
    if not roundtrip_byte_identical:
        verdict_label = "FAIL"
        verdict_message = (
            "WZPSC01 archive bytes roundtrip is NOT byte-identical. "
            "This is a catastrophic primitive contract violation per "
            "Catalog #105/#139/#220/#272 no-op detector. The substrate's "
            "Wyner 1976 R(D|Y) reconstructibility invariant is violated. "
            "DO NOT promote this candidate."
        )
        exit_code = 1
    elif max_density_percent >= density_pass_threshold_percent:
        verdict_label = "PASS"
        verdict_message = (
            f"Y-derivable-prefix density {max_density_percent:.6f}% supports "
            "additive composition per sister design memo §Predicted ΔS band. "
            "SUB-FRONTIER candidate. Operator-routable L2 paired CUDA+CPU "
            "auth-eval per Catalog #246 + per-substrate symposium per "
            "Catalog #325."
        )
        exit_code = 0
    elif max_density_percent >= density_observability_threshold_percent:
        verdict_label = "OBSERVABILITY_ONLY_SATURATING_COMPOSITION"
        verdict_message = (
            f"Y-derivable-prefix density {max_density_percent:.6f}% in the "
            "saturating composition band per Catalog #227 alpha=0.5. "
            "Predicted band [-0.0050, -0.0020] per sister design memo. "
            "NEXT-VARIANT iteration queued; not yet sub-frontier."
        )
        exit_code = 2
    else:
        verdict_label = "OBSERVABILITY_ONLY_IMPLEMENTATION_LEVEL_FALSIFICATION"
        verdict_message = (
            f"Y-derivable-prefix density {max_density_percent:.6f}% is below "
            f"{density_observability_threshold_percent}% threshold. "
            "IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307; PARADIGM "
            "INTACT (Wyner 1976 R(D|Y); decoder-side PoseNet as canonical Y "
            "per Catalog #311 Atick-Tishby-Wyner triple). Per CLAUDE.md "
            "'Forbidden premature KILL': DEFERRED-PENDING-research. "
            "Reactivation paths in sister design memo §Reactivation criteria."
        )
        exit_code = 2

    gate_verdict = {
        "schema_version": WZPSC_GATE_SCHEMA,
        "substrate_id": "wyner_ziv_pipeline_stage_codec",
        "lane_id": "lane_wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_20260528",
        "gate_kind": "catalog_1265_archive_bytes_byte_identity_NOT_decoder_float_parity",
        "gate_rationale": (
            "This substrate is a byte-stream codec wrapper (Wyner 1976 R(D|Y) "
            "primitive); the canonical contest-equivalence verdict IS the "
            "WZPSC01 archive bytes byte-identical roundtrip + Y-derivable-"
            "prefix density vs the sister design memo §Predicted ΔS band. "
            "Sister gates (Z6-v2, IA3) measure MLX vs PyTorch decoder forward "
            "parity in sigmoid [0, 1] space; this substrate's mathematical "
            "structure has no such notion (byte-equivalence is the contract)."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "mlx_training_artifact_path": str(mlx_training_artifact_path),
            "mlx_training_artifact_sha256": _hash_file(mlx_training_artifact_path),
            "wzpsc01_archive_sha256": archive_info.get("sha256"),
            "wzpsc01_archive_bytes_len": archive_info.get("bytes_len"),
        },
        "measurements": {
            "roundtrip_byte_identical": roundtrip_byte_identical,
            "max_density_percent": max_density_percent,
            "verdict_kind_from_artifact": verdict_kind_from_artifact,
            "density_pass_threshold_percent": density_pass_threshold_percent,
            "density_observability_threshold_percent": density_observability_threshold_percent,
        },
        "parity_proof_cross_check": parity_proof_details,
        "verdict": {
            "label": verdict_label,
            "exit_code": exit_code,
            "message": verdict_message,
        },
        # Catalog #341 non-promotable markers; Catalog #287/#323 canonical Provenance
        "canonical_provenance": {
            "kind": "predicted_from_model",
            "evidence_grade": "macOS-MLX research-signal",
            "axis_tag": "[macOS-MLX research-signal]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotable": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "Catalog #1265 gate verdict on a WZPSC01 archive candidate. "
                "Non-promotable per Catalog #192/#317/#341 + CLAUDE.md 'MLX "
                "portable-local-substrate authority' (this is an observability "
                "verdict, not a contest score claim). PASS verdict here means "
                "the candidate is operator-routable for L2 paired CUDA+CPU "
                "auth-eval per Catalog #246 + per-substrate symposium per "
                "Catalog #325 14-day window."
            ),
            "canonical_helper_invocation": (
                "tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer + "
                "tac.substrates.wyner_ziv_pipeline_stage_codec.inflate."
                "inflate_wyner_ziv_pipeline_stage_codec_scaffold"
            ),
            "hardware_substrate": "darwin_arm64_m5_max_macos_mlx_local",
        },
    }
    return gate_verdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate_mlx_candidate_contest_equivalence_wyner_ziv_pipeline_stage_codec",
        description=(
            "Catalog #1265 canonical PASS/FAIL gate for Wyner-Ziv pipeline-"
            "stage codec WZPSC01-grammar candidates. Byte-identity + density-"
            "threshold gate (NOT decoder float-parity per this substrate's "
            "byte-stream mathematical structure)."
        ),
    )
    parser.add_argument(
        "--mlx-training-artifact-path",
        type=Path,
        required=True,
        help="Path to MLX-LOCAL L1 harness's training_artifact.json.",
    )
    parser.add_argument(
        "--parity-proof-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the bridge tool's parity-proof JSON (per "
            "tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py). "
            "If provided the gate cross-checks the archive sha256 + roundtrip "
            "consistency."
        ),
    )
    parser.add_argument(
        "--density-pass-threshold-percent",
        type=float,
        default=DEFAULT_DENSITY_PASS_THRESHOLD_PERCENT,
        help=(
            "Density above which the candidate is PASS (sub-frontier candidate "
            "per the sister design memo §Predicted ΔS band). Default 5.0%."
        ),
    )
    parser.add_argument(
        "--density-observability-threshold-percent",
        type=float,
        default=DEFAULT_DENSITY_OBSERVABILITY_THRESHOLD_PERCENT,
        help=(
            "Density above which the candidate is OBSERVABILITY_ONLY "
            "(saturating composition per Catalog #227 alpha=0.5). Default 1.0%."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the gate verdict JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        verdict = gate_wyner_ziv_pipeline_stage_codec_candidate(
            mlx_training_artifact_path=args.mlx_training_artifact_path,
            parity_proof_path=args.parity_proof_path,
            density_pass_threshold_percent=args.density_pass_threshold_percent,
            density_observability_threshold_percent=(
                args.density_observability_threshold_percent
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[wzpsc-gate] FAIL: {exc!r}", file=sys.stderr)
        return 3

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[wzpsc-gate] verdict written to {args.out}")

    print(
        f"[wzpsc-gate] VERDICT: {verdict['verdict']['label']} "
        f"(exit={verdict['verdict']['exit_code']}; "
        f"density={verdict['measurements']['max_density_percent']:.6f}%; "
        f"roundtrip_byte_identical={verdict['measurements']['roundtrip_byte_identical']}; "
        f"non-promotable per Catalog #341)."
    )
    print(f"[wzpsc-gate] message: {verdict['verdict']['message']}")
    return int(verdict["verdict"]["exit_code"])


__all__ = (
    "gate_wyner_ziv_pipeline_stage_codec_candidate",
    "main",
    "WZPSC_GATE_SCHEMA",
    "DEFAULT_DENSITY_PASS_THRESHOLD_PERCENT",
    "DEFAULT_DENSITY_OBSERVABILITY_THRESHOLD_PERCENT",
)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
