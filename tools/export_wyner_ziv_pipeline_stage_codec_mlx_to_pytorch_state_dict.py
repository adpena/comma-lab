#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec MLX → PyTorch archive-bytes bridge tool.

PER-SUBSTRATE INDIVIDUALLY-FRACTAL canonical engineering pass per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 REINFORCED 2026-05-28.

**Structural distinction from sister bridges**: this substrate is NOT a
neural renderer; the archive is a byte-stream
``(main_compressed, side_compressed_baked, meta_json)`` per WZPSC01 grammar
(see :mod:`tac.substrates.wyner_ziv_pipeline_stage_codec.archive`). There is
NO MLX state_dict to bridge to a PyTorch state_dict. The canonical
"bridge" for this substrate is the **archive-bytes parity proof**: the same
WZPSC01 archive bytes MUST decode byte-identically via the canonical
primitive ``tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer``
regardless of host (M5 Max MLX, Linux x86_64 CPU, NVIDIA CUDA — they all
run the same numpy-portable inflate path per HNeRV parity L4 ≤2 deps).

Sister bridges (for context, NOT shared code per UNIQUE-AND-COMPLETE-PER-
METHOD):

* ``tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py``
  — bridges MLX latents+ego_vecs+film_gen state_dict to PyTorch (neural
  substrate)
* ``tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py`` — bridges
  MLX IA3 state_dict to PyTorch (neural substrate)
* ``tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py``
  — THIS tool: bridges WZPSC01 archive bytes parity (byte-stream substrate)

Canonical bridge contract:

::

    MLX-LOCAL L1 harness emits training_artifact.json + WZPSC01 archive (.bin)
      |
      v   :func:`export_wyner_ziv_pipeline_stage_codec_archive_bytes_parity`
      v
    Archive-bytes parity proof JSON + numpy-portable inflate sanity
      |
      v   :mod:`tools.gate_mlx_candidate_contest_equivalence_wyner_ziv_pipeline_stage_codec`
      v
    Catalog #1265 contest-equivalence verdict (PASS / FAIL / OBSERVABILITY_ONLY)

Per CLAUDE.md non-negotiables PRESERVED:

* "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
  — bridge output is non-promotable until paired CUDA+CPU auth-eval.
* "MLX portable-local-substrate authority" — bridge produces
  ``evidence_grade='macOS-MLX research-signal'`` artifacts.
* "Bugs must be permanently fixed AND self-protected against" via canonical
  Provenance per Catalog #287/#323 + Tier A markers per Catalog #341.
* Catalog #110/#113 APPEND-ONLY: NEW parity-proof JSON; never mutates.
* HNeRV parity L4: ≤200 LOC inflate; bridge tool body ≤350 LOC (substrate
  engineering exception per L7).
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
WZPSC_BRIDGE_SCHEMA = "wyner_ziv_pipeline_stage_codec_archive_bytes_parity_bridge.v1"
UTC = timezone.utc


def _hash_file(path: Path) -> str:
    """Compute sha256 of a file's bytes (canonical fingerprint per Catalog #323)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_wyner_ziv_pipeline_stage_codec_archive_bytes_parity(
    *,
    mlx_training_artifact_path: Path,
    output_parity_proof: Path,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Bridge: verify WZPSC01 archive bytes parity + emit canonical Provenance proof.

    Per the canonical bridge contract above: this substrate has NO MLX
    state_dict to bridge — the archive is byte-stream per WZPSC01 grammar.
    The "bridge" verifies that the archive bytes emitted by the MLX-LOCAL
    L1 harness decode byte-identically via the canonical primitive's
    reconstruct path (the same path the contest's inflate runtime would
    use).

    Args:
        mlx_training_artifact_path: path to the MLX-LOCAL L1 harness's
            ``training_artifact.json`` emitted by
            :func:`tac.substrates.wyner_ziv_pipeline_stage_codec.trainer._full_main`.
        output_parity_proof: path to write the canonical parity-proof JSON.
        overwrite: if True, overwrite existing output_parity_proof.

    Returns:
        Dict matching the parity-proof JSON's schema (also written to
        ``output_parity_proof``).

    Raises:
        FileNotFoundError: ``mlx_training_artifact_path`` does not exist,
            OR the archive .bin path declared in the artifact does not exist.
        ValueError: artifact JSON is malformed / missing required keys.
        RuntimeError: archive bytes roundtrip is NOT byte-identical
            (catastrophic primitive contract violation).
    """
    if not mlx_training_artifact_path.exists():
        raise FileNotFoundError(
            f"MLX training artifact not found at {mlx_training_artifact_path}; "
            "run `python -m tac.substrates.wyner_ziv_pipeline_stage_codec.trainer "
            "--full --output-dir <out>` first."
        )

    artifact = json.loads(mlx_training_artifact_path.read_text())
    if artifact.get("substrate_id") != "wyner_ziv_pipeline_stage_codec":
        raise ValueError(
            f"artifact substrate_id != 'wyner_ziv_pipeline_stage_codec'; got "
            f"{artifact.get('substrate_id')!r}"
        )

    archive_info = artifact.get("wzpsc01_archive", {})
    archive_path_str = archive_info.get("saved_to")
    if not archive_path_str:
        raise ValueError("artifact missing 'wzpsc01_archive.saved_to' field")
    archive_path = Path(archive_path_str)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"archive bytes not found at {archive_path}; the MLX-LOCAL "
            "L1 harness may have been run with a temporary output-dir."
        )

    # Re-derive Y from the canonical source declared in the artifact + verify
    # byte-identical roundtrip via the same canonical primitive path the
    # contest's inflate runtime would use.
    from tac.codec.wyner_ziv_layer import derive_side_info_from_canonical_source
    from tac.substrates.wyner_ziv_pipeline_stage_codec.inflate import (
        inflate_wyner_ziv_pipeline_stage_codec_scaffold,
    )

    best_source = artifact.get("best_source", "math_constants")
    if best_source == "none":
        best_source = "math_constants"
    y_bytes = derive_side_info_from_canonical_source(best_source)

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    t_inflate_start = time.time()
    inflated = inflate_wyner_ziv_pipeline_stage_codec_scaffold(
        archive_bytes=archive_bytes,
        side_info_y=y_bytes,
    )
    inflate_seconds = time.time() - t_inflate_start
    reconstructed = inflated["reconstructed_pre_entropy_bytes"]

    # The "parity proof" here is: the same archive bytes that the MLX-LOCAL
    # harness emitted decode byte-identically AT THE SAME source bytes
    # length, with the SAME source bytes sha256.
    source_bytes_sha256 = artifact["base_substrate"]["pre_entropy_bytes_sha256"]
    source_bytes_len = artifact["base_substrate"]["pre_entropy_bytes_len"]
    reconstructed_sha256 = hashlib.sha256(reconstructed).hexdigest()
    roundtrip_byte_identical = (
        len(reconstructed) == source_bytes_len
        and reconstructed_sha256 == source_bytes_sha256
    )
    if not roundtrip_byte_identical:
        raise RuntimeError(
            f"archive bytes roundtrip is NOT byte-identical: "
            f"source_len={source_bytes_len} reconstructed_len={len(reconstructed)} "
            f"source_sha={source_bytes_sha256[:16]} "
            f"reconstructed_sha={reconstructed_sha256[:16]}. This is a "
            "catastrophic primitive contract violation per Catalog #105/"
            "#139/#220/#272 no-op detector."
        )

    parity_proof = {
        "schema_version": WZPSC_BRIDGE_SCHEMA,
        "substrate_id": "wyner_ziv_pipeline_stage_codec",
        "lane_id": "lane_wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_20260528",
        "bridge_kind": "archive_bytes_parity_NOT_state_dict_bridge",
        "bridge_rationale": (
            "This substrate is a byte-stream codec wrapper (no neural state_dict); "
            "the canonical bridge IS the archive-bytes byte-identical roundtrip "
            "via the same canonical primitive path the contest inflate runtime "
            "would use. Sister bridges (Z6-v2, IA3) bridge MLX state_dict to "
            "PyTorch state_dict; this substrate's mathematical structure has no "
            "such bridge requirement."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "mlx_training_artifact_path": str(mlx_training_artifact_path),
            "mlx_training_artifact_sha256": _hash_file(mlx_training_artifact_path),
            "wzpsc01_archive_path": str(archive_path),
            "wzpsc01_archive_sha256": archive_sha256,
            "wzpsc01_archive_bytes_len": len(archive_bytes),
            "side_info_source_used_for_roundtrip": best_source,
        },
        "parity_verdict": {
            "roundtrip_byte_identical": roundtrip_byte_identical,
            "source_bytes_len": source_bytes_len,
            "source_bytes_sha256": source_bytes_sha256,
            "reconstructed_bytes_len": len(reconstructed),
            "reconstructed_bytes_sha256": reconstructed_sha256,
            "inflate_seconds": inflate_seconds,
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
                "Archive-bytes parity verified via the canonical primitive's "
                "reconstruct_from_wyner_ziv_layer path. Non-promotable per "
                "Catalog #192/#317/#341 + CLAUDE.md 'MLX portable-local-"
                "substrate authority' (this is a parity proof, not a contest "
                "score claim). Promotion to a contest score claim requires "
                "L2 paired CUDA+CPU auth-eval per Catalog #246 + per-substrate "
                "symposium per Catalog #325 14-day window."
            ),
            "canonical_helper_invocation": (
                "tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer + "
                "tac.substrates.wyner_ziv_pipeline_stage_codec.inflate."
                "inflate_wyner_ziv_pipeline_stage_codec_scaffold"
            ),
            "hardware_substrate": "darwin_arm64_m5_max_macos_mlx_local",
        },
    }

    if output_parity_proof.exists() and not overwrite:
        raise FileExistsError(
            f"parity proof already exists at {output_parity_proof} and "
            "overwrite=False"
        )
    output_parity_proof.parent.mkdir(parents=True, exist_ok=True)
    output_parity_proof.write_text(
        json.dumps(parity_proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return parity_proof


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict",
        description=(
            "Wyner-Ziv pipeline-stage codec MLX → PyTorch archive-bytes bridge "
            "tool. Verifies byte-identical roundtrip + emits canonical "
            "Provenance proof (NOT a state_dict bridge; this substrate is "
            "byte-stream codec wrapper)."
        ),
    )
    parser.add_argument(
        "--mlx-training-artifact-path",
        type=Path,
        required=True,
        help=(
            "Path to the MLX-LOCAL L1 harness's training_artifact.json "
            "emitted by tac.substrates.wyner_ziv_pipeline_stage_codec.trainer"
            "._full_main."
        ),
    )
    parser.add_argument(
        "--output-parity-proof",
        type=Path,
        required=True,
        help="Path to write the canonical parity-proof JSON.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to overwrite an existing parity proof file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        proof = export_wyner_ziv_pipeline_stage_codec_archive_bytes_parity(
            mlx_training_artifact_path=args.mlx_training_artifact_path,
            output_parity_proof=args.output_parity_proof,
            overwrite=not args.no_overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"[wzpsc-bridge] FAIL: {exc!r}", file=sys.stderr)
        return 1

    print(
        f"[wzpsc-bridge] OK: parity proof written to {args.output_parity_proof} "
        f"(roundtrip_byte_identical={proof['parity_verdict']['roundtrip_byte_identical']}; "
        f"archive_sha={proof['inputs']['wzpsc01_archive_sha256'][:16]}; "
        f"non-promotable per Catalog #341)."
    )
    return 0


__all__ = (
    "export_wyner_ziv_pipeline_stage_codec_archive_bytes_parity",
    "main",
    "WZPSC_BRIDGE_SCHEMA",
)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
