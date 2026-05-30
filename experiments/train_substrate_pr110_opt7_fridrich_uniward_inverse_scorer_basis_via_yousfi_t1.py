#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:MLX_first_substrate_trainer_does_not_use_pytorch_cuda_autocast_fp16_primitive_per_operator_mlx_first_canonical_doctrine_2026_05_30_or_substrate_uses_documented_alternative_precision_strategy
# NO_GRAD_WAIVED:mlx_substrate_trainer_uses_no_autograd_at_eval_canonical_mlx_lazy_eval_pattern_per_operator_mlx_first_canonical_doctrine_2026_05_30
# TF32_WAIVED:mlx_substrate_trainer_runs_on_mlx_macos_advisory_no_pytorch_cuda_tf32_per_operator_mlx_first_canonical_doctrine_2026_05_30
# TORCH_COMPILE_WAIVED:mlx_substrate_trainer_uses_mlx_lazy_eval_not_pytorch_torch_compile_per_operator_mlx_first_canonical_doctrine_2026_05_30
"""PR110-OPT-7 via Yousfi-T1 substrate trainer — L1 PROMOTION canonical entry point.

Per CLAUDE.md "MLX portable-local-substrate authority" + operator standing
directive 2026-05-30 verbatim *"30k plus epoch runs all for free on MLX
for proving... MLX to the full extent possible"*: this trainer runs
MLX-first on macOS-MLX advisory. Per Catalog #127/#192/#317/#341:
non-promotable by construction; macOS-CPU advisory / macOS-MLX
research-signal only. Paired-CUDA RATIFICATION per Catalog #246 on
1:1 contest-compliant hardware is the operator-attended dispatch path
per Phase D recipe.

Per deferred-items feeder audit ``46aa6ad86`` Phase E TOP-1 op-routable
PR110-OPT-7 L1 promotion via Yousfi-T1 enablement: this trainer composes
the 5 LANDED canonical primitives (alaska / Yousfi-T1 A+B+C / PR110-OPT-7
L0 SCAFFOLD) into ONE coherent substrate per HNeRV-parity-or-greater
binding-depth discipline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# Canonical Tier 1 + Tier 2 + Tier 3 manifest per Catalog #151 + #270.
# The substrate is MLX-first so the canonical Tier 1 fp16/tf32/torch.compile
# flags are PYTORCH-canonical and DO NOT apply; substrate adopts canonical
# MLX-lazy-eval per CLAUDE.md "MLX portable-local-substrate authority".
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, object]] = {
    "--enable-autocast-fp16": {
        "default": False,
        "env": "ENABLE_AUTOCAST_FP16",
        "satisfied_by_profile": [],
        "rationale": "Substrate is MLX-first; PyTorch autocast does not apply",
    },
    "--enable-tf32": {
        "default": False,
        "env": "ENABLE_TF32",
        "satisfied_by_profile": [],
        "rationale": "Substrate is MLX-first; PyTorch TF32 does not apply",
    },
    "--enable-torch-compile": {
        "default": False,
        "env": "ENABLE_TORCH_COMPILE",
        "satisfied_by_profile": [],
        "rationale": "Substrate is MLX-first; uses canonical MLX lazy eval",
    },
    "--no-grad-eval": {
        "default": True,
        "env": "NO_GRAD_EVAL",
        "satisfied_by_profile": [],
        "rationale": "Substrate uses MLX no-autograd at eval per CLAUDE.md",
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _emit_substrate_archive(
    output_dir: Path,
    *,
    config_dict: dict,
    result_summary: dict,
    pr110_base_archive_path: Path | None,
) -> Path:
    """Emit OPT7VYT1 archive per the canonical 4-section grammar.

    Per Catalog #146 + #220 + #272: archive header + 4 length-prefixed zlib
    sections + optional PR110 base inline. The L1 PROMOTION emits sections
    serialized as JSON payloads; L2 INTEGRATION substrate-engineering wave
    replaces with full canonical per-pair binary serialization.
    """
    import struct
    import zlib

    from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.archive_grammar import (
        ARCHIVE_MAGIC,
        ARCHIVE_VERSION,
        pack_header,
    )

    archive_dir = output_dir / "submission" / "submission_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_bin_path = archive_dir / "0.bin"

    # Serialize each of the 4 canonical sections as zlib-compressed JSON
    sections_payloads = [
        json.dumps(
            result_summary.get("pose_vulnerability_summary", {}),
            sort_keys=True,
        ).encode("utf-8"),
        json.dumps(
            {"alaska_color_slice": list(result_summary.get("alaska_color_slice", []))},
            sort_keys=True,
        ).encode("utf-8"),
        json.dumps(
            result_summary.get("inverse_scorer_basis_summary", {}),
            sort_keys=True,
        ).encode("utf-8"),
        json.dumps(
            result_summary.get("chroma_perturbation_summary", {}),
            sort_keys=True,
        ).encode("utf-8"),
    ]

    pr110_base_sha256_prefix = bytes(16)  # placeholder; L2 wave wires real
    header = pack_header(
        version=ARCHIVE_VERSION,
        alaska_color_branch_index=9,  # Y0_UV canonical default
        basis_strategy_index=3,  # JOINT canonical default
        chroma_strategy_index=3,  # JOINT canonical default
        pr110_base_sha256_prefix=pr110_base_sha256_prefix,
    )

    pr110_base_bytes = b""
    if pr110_base_archive_path is not None and pr110_base_archive_path.is_file():
        pr110_base_bytes = pr110_base_archive_path.read_bytes()

    with archive_bin_path.open("wb") as f:
        f.write(header)
        for payload in sections_payloads:
            compressed = zlib.compress(payload, level=9) if payload else b""
            f.write(struct.pack("<I", len(compressed)))
            f.write(compressed)
        f.write(pr110_base_bytes)

    archive_bytes = archive_bin_path.stat().st_size

    # Write inflate.sh + inflate.py
    inflate_sh = archive_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'exec uv run "$HERE/inflate.py" "$1" "$2" "$3"\n'
    )
    inflate_sh.chmod(0o755)

    # Copy canonical inflate.py from the substrate package
    canonical_inflate = (
        REPO_ROOT
        / "src/tac/substrates/pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1/inflate.py"
    )
    (archive_dir / "inflate.py").write_text(canonical_inflate.read_text())
    (archive_dir / "inflate.py").chmod(0o755)

    return archive_bin_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--vulnerable-pair-budget", type=int, default=4)
    parser.add_argument(
        "--alaska-color-branch",
        type=str,
        default="Y0_UV",
        help="Canonical alaska ColorBranchSliceStrategy value (default Y0_UV)",
    )
    parser.add_argument(
        "--inverse-scorer-basis-strategy",
        type=str,
        default="uniward_inverse_joint_scorer_basis_linear_combination",
        help="PR110-OPT-7 InverseScorerBasisStrategy value",
    )
    parser.add_argument(
        "--chroma-perturbation-strategy",
        type=str,
        default="joint_atick_redlich_linear_combination",
        help="Yousfi-T1 Deliverable C ChromaPerturbationStrategy value",
    )
    parser.add_argument(
        "--chroma-perturbation-magnitude",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--use-canonical-pose-vulnerability-anchor",
        action="store_true",
        default=False,
        help="Use canonical Yousfi-T1 600-pair fp64 anchor (default: synthetic)",
    )
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pr110-base-archive-path",
        type=Path,
        default=None,
        help="Optional PR110 base archive to inline-embed in OPT7VYT1 archive",
    )
    # Canonical Tier 1 flags per Catalog #151 + #270 (MLX-first; flags
    # documented but no-op since substrate is MLX-native).
    parser.add_argument("--enable-autocast-fp16", action="store_true", default=False)
    parser.add_argument("--enable-tf32", action="store_true", default=False)
    parser.add_argument("--enable-torch-compile", action="store_true", default=False)
    parser.add_argument("--no-grad-eval", action="store_true", default=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)

    # Import substrate (lazy to keep parser quick)
    from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1 import (
        PR110OPT7ViaYousfiT1Config,
        apply_substrate_to_pr110_canonical,
        verify_canonical_helper_invocation,
    )

    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] {_utc_now()} START "
        f"n_pairs={args.n_pairs} vulnerable_pair_budget={args.vulnerable_pair_budget} "
        f"alaska_color_branch={args.alaska_color_branch} "
        f"basis_strategy={args.inverse_scorer_basis_strategy} "
        f"chroma_strategy={args.chroma_perturbation_strategy}"
    )

    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=args.n_pairs,
        vulnerable_pair_budget=args.vulnerable_pair_budget,
        alaska_color_branch=args.alaska_color_branch,
        inverse_scorer_basis_strategy=args.inverse_scorer_basis_strategy,
        chroma_perturbation_strategy=args.chroma_perturbation_strategy,
        chroma_perturbation_magnitude=args.chroma_perturbation_magnitude,
        use_canonical_pose_vulnerability_anchor=args.use_canonical_pose_vulnerability_anchor,
        rng_seed=args.rng_seed,
    )

    t_start = time.time()
    result = apply_substrate_to_pr110_canonical(config, repo_root=str(REPO_ROOT))
    elapsed = time.time() - t_start

    # Slot EEE NO FAKE IMPLEMENTATIONS verification
    verdict = verify_canonical_helper_invocation(result)
    if verdict["substantive_distinctness_verdict"] != "PASS":
        print(
            f"[pr110-opt7-via-yousfi-t1-l1-trainer] SLOT_EEE_VERIFICATION_FAILED: "
            f"{json.dumps(verdict, indent=2)}",
            file=sys.stderr,
        )
        return 3

    # Build canonical result summary
    result_summary = {
        "schema_version": "pr110_opt7_via_yousfi_t1_l1_promotion_v1",
        "generated_at_utc": _utc_now(),
        "config": {
            "n_pairs": config.n_pairs,
            "vulnerable_pair_budget": config.vulnerable_pair_budget,
            "alaska_color_branch": config.alaska_color_branch,
            "inverse_scorer_basis_strategy": config.inverse_scorer_basis_strategy,
            "chroma_perturbation_strategy": config.chroma_perturbation_strategy,
            "chroma_perturbation_magnitude": config.chroma_perturbation_magnitude,
            "use_canonical_pose_vulnerability_anchor": (
                config.use_canonical_pose_vulnerability_anchor
            ),
            "rng_seed": config.rng_seed,
        },
        "tier_a_canonical_routing_markers": {
            "predicted_delta_adjustment": result.predicted_delta_adjustment,
            "promotable": result.promotable,
            "axis_tag": result.axis_tag,
            "verdict": result.verdict,
        },
        "pose_vulnerability_summary": dict(result.pose_vulnerability_summary),
        "alaska_color_slice": list(result.alaska_color_slice),
        "inverse_scorer_basis_summary": dict(result.inverse_scorer_basis_summary),
        "chroma_perturbation_summary": dict(result.chroma_perturbation_summary),
        "posenet_surrogate_summary": dict(result.posenet_surrogate_summary),
        "per_pair_selected_indices": list(result.per_pair_selected_indices),
        "canonical_helpers_invoked": dict(result.canonical_helpers_invoked),
        "cross_reference_matrix": dict(result.cross_reference_matrix),
        "canonical_provenance": dict(result.canonical_provenance),
        "slot_eee_verification": verdict,
        "elapsed_seconds": elapsed,
        "evidence_grade": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
    }

    # Write canonical training stats per Catalog #305 observability surface
    stats_path = args.output_dir / "training_stats.json"
    stats_path.write_text(json.dumps(result_summary, indent=2, sort_keys=True))

    # Emit OPT7VYT1 archive per Catalog #146 contest contract
    archive_path = _emit_substrate_archive(
        args.output_dir,
        config_dict=result_summary["config"],
        result_summary=result_summary,
        pr110_base_archive_path=args.pr110_base_archive_path,
    )

    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] DONE elapsed={elapsed:.2f}s "
        f"helpers_invoked={verdict['invocation_count']}/5 "
        f"substantive={verdict['substantive_distinctness_verdict']} "
        f"archive={archive_path} "
        f"archive_bytes={archive_path.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
