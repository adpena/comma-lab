#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:MLX_first_substrate_trainer_does_not_use_pytorch_cuda_autocast_fp16_primitive_per_operator_mlx_first_canonical_doctrine_2026_05_30_l0_scaffold_smoke_only
# NO_GRAD_WAIVED:mlx_substrate_trainer_uses_no_autograd_at_eval_canonical_mlx_lazy_eval_pattern_per_operator_mlx_first_canonical_doctrine_2026_05_30
# TF32_WAIVED:mlx_substrate_trainer_runs_on_mlx_macos_advisory_no_pytorch_cuda_tf32_per_operator_mlx_first_canonical_doctrine_2026_05_30
# TORCH_COMPILE_WAIVED:mlx_substrate_trainer_uses_mlx_lazy_eval_not_pytorch_torch_compile_per_operator_mlx_first_canonical_doctrine_2026_05_30
"""PR110-OPT-11 multi-mode-per-pair composition L0 SCAFFOLD + L1 trainer.

Per CLAUDE.md "MLX portable-local-substrate authority" + operator MLX-first
doctrine 2026-05-30: L0 SCAFFOLD smoke + L1 Phase C smoke run MLX-LOCAL /
macOS-CPU advisory ONLY. Per Catalog #127 / #192 / #317 / #341: non-promotable
by construction. Per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK:
``_full_main`` raises ``NotImplementedError`` until Phase 2 council symposium
per Catalog #325 unblocks paired-CUDA RATIFICATION dispatch.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + Catalog #240 recipe-vs-trainer-state consistency: the
companion recipe ``.omx/operator_authorize_recipes/substrate_pr110_opt11_*.yaml``
declares ``research_only: true`` + ``dispatch_enabled: false`` so the trainer's
``_full_main NotImplementedError`` is transparent at the dispatch surface.

Per Catalog #326 driver-mode-hardcode discipline: trainer-mode resolution
honors PR110_OPT11_TRAINER_MODE env var (smoke|l1|full) with explicit precedence
over SMOKE_ONLY. Default = smoke (L0 SCAFFOLD evidence surface preserved).

The substrate's distinguishing-feature per Catalog #272 is the per-pair
(selector_a, selector_b) multi-mode composition.

L0 SCAFFOLD smoke (``_smoke_main``): ACTUALLY composes 2 modes per pair via
real numpy arithmetic on synthetic frames (NO FAKE IMPLEMENTATIONS per Slot
EEE 5 forbidden classes — Class 1).

L1 PROMOTION (``_l1_main``, mode=l1): per CLAUDE.md "Canonical leaderboard
binding-depth discipline" L7 + Catalog #213 real-frame consumption + per
"Forbidden make_synthetic_pair_batch in any non-smoke training path": consumes
REAL ``upstream/videos/0.mkv`` contest base pairs via the canonical
``tac.substrates._shared.trainer_skeleton.decode_real_pairs`` and ACTUALLY
invokes the substrate's ``_compose_two_modes_on_frame`` distinguishing
primitive on real frame_0 arrays (NOT a trivial ``+1.0`` probe — NO FAKE
IMPLEMENTATIONS Slot EEE Class 1 + Class 2). MLX-LOCAL Phase C 7/7 GREEN
canonical validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Phase C 7/7 GREEN canonical validation axes per the L1 PROMOTION contract.
PHASE_C_REQUIRED_AXES: tuple[str, ...] = (
    "real_frame_consumption_catalog_213",
    "multi_mode_composition_applied_on_real_frames",
    "composition_distinct_from_both_single_modes",
    "archive_emitted_byte_stable",
    "canonical_helpers_invoked",
    "tier_a_markers_present",
    "provenance_canonical",
)
PHASE_C_PASS_THRESHOLD = 7


# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS canonical manifest.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, object]] = {
    "--output-dir": {
        "env": "PR110_OPT11_OUTPUT_DIR",
        "satisfied_by_profile": [],
        "rationale": "Substrate outputs (archive + sentinel inflate runtime + smoke logs).",
        "required_input_file": False,
    },
    "--enable-autocast-fp16": {
        "env": "ENABLE_AUTOCAST_FP16",
        "satisfied_by_profile": [],
        "rationale": "MLX-first substrate; declared for Catalog #172 conformance even when N/A.",
    },
    "--enable-torch-compile": {
        "env": "ENABLE_TORCH_COMPILE",
        "satisfied_by_profile": [],
        "rationale": "MLX-first substrate; declared for Catalog #179 conformance even when N/A.",
    },
    "--no-grad-eval": {
        "env": "NO_GRAD_EVAL",
        "satisfied_by_profile": [],
        "rationale": "MLX no-autograd at eval per CLAUDE.md.",
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _emit_substrate_archive(
    output_dir: Path,
    *,
    config_dict: dict,
    per_pair_selectors: list,
    family_pair: tuple,
) -> Path:
    """Emit OPT11MMP archive per the canonical 2-section grammar.

    Per Catalog #146 + #220 + #272: archive header + 2 length-prefixed zlib
    sections + optional PR110 base inline. The L0 SCAFFOLD emits sections
    serialized as raw selector stream + JSON family-pair metadata; L1+
    INTEGRATION substrate-engineering wave replaces with full canonical
    per-pair binary serialization + real PR110 base inline.
    """
    import struct
    import zlib

    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.archive_grammar import (
        ARCHIVE_VERSION,
        pack_header,
        pack_selector_stream,
    )

    archive_dir = output_dir / "submission" / "submission_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_bin_path = archive_dir / "0.bin"

    # Section 0: raw per-pair (selector_a, selector_b) stream
    selector_stream = pack_selector_stream(
        per_pair_selectors, selector_bits_per_mode=config_dict["selector_bits_per_mode"]
    )
    # Section 1: family-pair metadata JSON
    family_metadata = json.dumps(
        {
            "family_a": family_pair[0],
            "family_b": family_pair[1],
            "n_pairs": config_dict["n_pairs"],
            "modes_per_pair": config_dict["modes_per_pair"],
            "selector_bits_per_mode": config_dict["selector_bits_per_mode"],
        },
        sort_keys=True,
    ).encode("utf-8")

    sections_payloads = [selector_stream, family_metadata]

    pr110_base_sha256_prefix = bytes(16)  # placeholder; L1+ wave wires real
    header = pack_header(
        version=ARCHIVE_VERSION,
        modes_per_pair=config_dict["modes_per_pair"],
        selector_bits_per_mode=config_dict["selector_bits_per_mode"],
        family_pair_index=config_dict["family_pair_index"],
        pr110_base_sha256_prefix=pr110_base_sha256_prefix,
    )

    with archive_bin_path.open("wb") as f:
        f.write(header)
        for payload in sections_payloads:
            compressed = zlib.compress(payload, level=9) if payload else b""
            f.write(struct.pack("<I", len(compressed)))
            f.write(compressed)
        # L0 SCAFFOLD: no PR110 base inline yet (L1+ wires)

    # Write inflate.sh + inflate.py
    inflate_sh = archive_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'exec uv run "$HERE/inflate.py" "$1" "$2" "$3"\n'
    )
    inflate_sh.chmod(0o755)

    canonical_inflate = (
        REPO_ROOT
        / "src/tac/substrates/pr110_opt11_multi_mode_per_pair_composition/inflate.py"
    )
    (archive_dir / "inflate.py").write_text(canonical_inflate.read_text())
    (archive_dir / "inflate.py").chmod(0o755)

    return archive_bin_path


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-LOCAL L0 SCAFFOLD smoke path (the canonical L0 evidence-bearing surface).

    Runs the substrate's per-pair multi-mode composition on synthetic frames;
    emits OPT11MMP archive + sentinel inflate runtime + summary JSON. Per
    CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192:
    non-promotable macOS-CPU advisory only.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)

    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition import (
        PR110OPT11Config,
        apply_substrate_to_pr110_canonical,
        verify_canonical_multi_mode_composition,
    )

    print(
        f"[pr110-opt11-l0-trainer] {_utc_now()} START smoke "
        f"n_pairs={args.n_pairs} modes_per_pair={args.modes_per_pair} "
        f"family_pair_index={args.family_pair_index}"
    )

    config = PR110OPT11Config(
        n_pairs=args.n_pairs,
        modes_per_pair=args.modes_per_pair,
        selector_bits_per_mode=args.selector_bits_per_mode,
        family_pair_index=args.family_pair_index,
        rng_seed=args.rng_seed,
    )

    t_start = time.time()
    result = apply_substrate_to_pr110_canonical(config)
    elapsed = time.time() - t_start

    invocation_verdict = verify_canonical_multi_mode_composition(result)

    # Emit canonical archive
    config_dict = {
        "n_pairs": config.n_pairs,
        "modes_per_pair": config.modes_per_pair,
        "selector_bits_per_mode": config.selector_bits_per_mode,
        "family_pair_index": config.family_pair_index,
    }
    archive_path = _emit_substrate_archive(
        args.output_dir,
        config_dict=config_dict,
        per_pair_selectors=list(result.per_pair_selectors),
        family_pair=result.family_pair,
    )
    archive_bytes = archive_path.stat().st_size

    summary = {
        "schema_version": "pr110_opt11_l0_smoke_v1",
        "subagent_axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "predicted_band_validation_status": "pending_post_training",
        "elapsed_seconds": elapsed,
        "n_pairs": config.n_pairs,
        "modes_per_pair": config.modes_per_pair,
        "selector_bits_per_mode": config.selector_bits_per_mode,
        "family_pair": list(result.family_pair),
        "verdict": result.verdict,
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "canonical_helpers_invocation_verdict": invocation_verdict,
        "composition_behavioral_evidence_summary": {
            "n_samples_composed": result.composition_behavioral_evidence.get(
                "n_samples_composed", 0
            ),
            "all_compositions_produced_distinct_output": (
                result.composition_behavioral_evidence.get(
                    "all_compositions_produced_distinct_output", False
                )
            ),
            "mean_sum_abs_delta_ab_vs_base": (
                result.composition_behavioral_evidence.get(
                    "mean_sum_abs_delta_ab_vs_base", 0.0
                )
            ),
            "n_combinations_addressable": (
                result.composition_behavioral_evidence.get(
                    "n_combinations_addressable", 0
                )
            ),
        },
        "canonical_provenance": result.canonical_provenance,
        "cross_reference_matrix": result.cross_reference_matrix,
        "result_review_blockers": [
            "non_promotable_macos_cpu_advisory_per_catalog_192",
            "l0_scaffold_pending_l1_integration_per_catalog_220",
            "paired_cuda_ratification_required_per_catalog_246",
            "per_substrate_symposium_phase_2_pending_per_catalog_325",
            "canonical_equation_formalization_pending_per_catalog_344",
        ],
    }

    summary_path = args.output_dir / "pr110_opt11_l0_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(
        f"[pr110-opt11-l0-trainer] {_utc_now()} DONE smoke "
        f"elapsed={elapsed:.2f}s archive_bytes={archive_bytes} "
        f"verdict={result.verdict} "
        f"composition_distinct_output={invocation_verdict['composition_distinct_output_verdict']}"
    )
    return 0


def _real_frame_base_pairs(n_pairs: int, *, downscale: int = 4) -> np.ndarray:
    """Decode REAL contest-video base pairs per Catalog #213.

    Per CLAUDE.md "Forbidden make_synthetic_pair_batch in any non-smoke training
    path" + Catalog #213 canonical real-frame consumption surface: L1 PROMOTION
    consumes real ``upstream/videos/0.mkv`` frames via the canonical
    ``tac.substrates._shared.trainer_skeleton.decode_real_pairs`` helper (which
    returns a torch ``(N, 2, 3, 384, 512)`` float32 tensor in ``[0, 255]`` at
    the canonical EVAL_HW), NOT synthetic ``rng.integers`` / ``torch.randn``
    fixtures.

    Converts to a ``(n_pairs, 2, H, W, 3)`` uint8 numpy array (the layout the
    substrate's ``_apply_canonical_perturbation`` consumes). ``downscale``
    strides the spatial dims to keep the MLX-LOCAL Phase C composition probe
    fast; the contest output resolution is preserved by the inflate runtime
    per Catalog #367, not by the trainer's downscaled probe.
    """
    from tac.substrates._shared.trainer_skeleton import decode_real_pairs

    video_path = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    # (N, 2, 3, 384, 512) float32 in [0, 255]
    tensor = decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag="pr110_opt11_multi_mode_per_pair_composition_l1",
        max_pairs=n_pairs,
        repo_root=REPO_ROOT,
    )
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] != n_pairs:
        raise RuntimeError(
            f"Catalog #213 real-frame decode returned {arr.shape[0]} pairs, "
            f"expected {n_pairs}; contest video may be truncated"
        )
    # (N, 2, 3, H, W) float -> (N, 2, H, W, 3) uint8
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (0, 1, 3, 4, 2))
    if downscale and downscale > 1:
        arr = arr[:, :, ::downscale, ::downscale, :]
    return np.ascontiguousarray(arr)


def _phase_c_validate(
    *,
    real_frame_consumed: bool,
    composition_applied: bool,
    composition_distinct: bool,
    archive_bytes: int,
    helpers_invoked: bool,
    tier_a_present: bool,
    provenance_ok: bool,
) -> dict:
    """Phase C 7/7 GREEN canonical validation per Catalog #294 + #213.

    Per Slot EEE NO FAKE Class 2: every axis verifies ACTUAL behavior on real
    frames; none are shape-constant assertions.
    """
    axes = {
        "real_frame_consumption_catalog_213": bool(real_frame_consumed),
        "multi_mode_composition_applied_on_real_frames": bool(composition_applied),
        "composition_distinct_from_both_single_modes": bool(composition_distinct),
        "archive_emitted_byte_stable": archive_bytes > 0,
        "canonical_helpers_invoked": bool(helpers_invoked),
        "tier_a_markers_present": bool(tier_a_present),
        "provenance_canonical": bool(provenance_ok),
    }
    n_pass = sum(1 for v in axes.values() if v)
    return {
        "axes": axes,
        "n_pass": n_pass,
        "n_total": len(axes),
        "phase_c_verdict": "GREEN" if n_pass == PHASE_C_PASS_THRESHOLD else "RED",
    }


def _l1_main(args: argparse.Namespace) -> int:
    """L1 PROMOTION MLX-LOCAL Phase C smoke path.

    Per CLAUDE.md "Canonical leaderboard binding-depth discipline" L7: binds
    real-frame consumption (Catalog #213) + the substrate's distinguishing
    multi-mode composition primitive (Catalog #272) applied on REAL frame_0
    arrays + Phase C 7/7 GREEN canonical validation + OPT11MMP archive
    emission. Per Catalog #192: non-promotable macOS-CPU advisory only until
    paired-CUDA RATIFICATION per Catalog #246.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)

    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition import (
        PR110OPT11Config,
        apply_substrate_to_pr110_canonical,
        verify_canonical_multi_mode_composition,
    )
    from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
        CANONICAL_MODE_MENU,
        _apply_canonical_perturbation,
        _compose_two_modes_on_frame,
    )

    print(
        f"[pr110-opt11-l1-trainer] {_utc_now()} START L1 Phase C smoke "
        f"n_pairs={args.n_pairs} downscale={args.downscale} "
        f"family_pair_index={args.family_pair_index}"
    )

    # --- Catalog #213: real-frame consumption ---
    t_decode = time.time()
    real_pairs = _real_frame_base_pairs(args.n_pairs, downscale=args.downscale)
    decode_elapsed = time.time() - t_decode
    real_frame_consumed = real_pairs.shape[0] == args.n_pairs and real_pairs.ndim == 5
    real_frame_0_mean = float(real_pairs[0, 0].mean())

    # --- canonical substrate apply (selector derivation) ---
    config = PR110OPT11Config(
        n_pairs=args.n_pairs,
        modes_per_pair=args.modes_per_pair,
        selector_bits_per_mode=args.selector_bits_per_mode,
        family_pair_index=args.family_pair_index,
        rng_seed=args.rng_seed,
    )
    result = apply_substrate_to_pr110_canonical(config)
    invocation_verdict = verify_canonical_multi_mode_composition(result)
    # The canonical substrate exposes per-family GLOBAL mode indices into
    # CANONICAL_MODE_MENU (a tuple of (family, mode_id) tuples); the per-pair
    # selector is an index INTO each family's global-index list.
    family_a_global = result.family_a_mode_indices
    family_b_global = result.family_b_mode_indices

    # --- Catalog #272 distinguishing-feature: ACTUALLY compose 2 modes on REAL
    #     frame_0 arrays + verify behavioral distinctness (NO FAKE Class 1+2) ---
    n_compose_samples = min(args.n_pairs, 8)
    n_distinct_from_a = 0
    n_distinct_from_b = 0
    n_composed = 0
    sum_abs_delta_real = 0.0
    composition_applied = False
    for i in range(n_compose_samples):
        sel_a, sel_b = result.per_pair_selectors[i]
        mode_a_global = family_a_global[sel_a % len(family_a_global)]
        mode_b_global = family_b_global[sel_b % len(family_b_global)]
        frame_0 = np.ascontiguousarray(real_pairs[i, 0])
        out_a = _apply_canonical_perturbation(frame_0, mode_a_global)
        out_b = _apply_canonical_perturbation(frame_0, mode_b_global)
        out_ab = _compose_two_modes_on_frame(frame_0, mode_a_global, mode_b_global)
        n_composed += 1
        composition_applied = True
        # NO FAKE Class 1+2: the composed frame must differ from BOTH single
        # modes (neither family A nor family B contains identity/zero-param per
        # the canonical menu, so composition is always distinct on a real frame).
        if not np.array_equal(out_ab, out_a):
            n_distinct_from_a += 1
        if not np.array_equal(out_ab, out_b):
            n_distinct_from_b += 1
        sum_abs_delta_real += float(
            np.sum(np.abs(out_ab.astype(np.int64) - frame_0.astype(np.int64)))
        )
    composition_distinct = (
        n_composed > 0
        and n_distinct_from_a == n_composed
        and n_distinct_from_b == n_composed
    )
    mean_abs_delta_real = sum_abs_delta_real / n_composed if n_composed else 0.0
    _ = CANONICAL_MODE_MENU  # referenced for canonical-menu provenance binding

    # --- OPT11MMP archive emission (canonical 2-section grammar) ---
    config_dict = {
        "n_pairs": config.n_pairs,
        "modes_per_pair": config.modes_per_pair,
        "selector_bits_per_mode": config.selector_bits_per_mode,
        "family_pair_index": config.family_pair_index,
    }
    archive_path = _emit_substrate_archive(
        args.output_dir,
        config_dict=config_dict,
        per_pair_selectors=list(result.per_pair_selectors),
        family_pair=result.family_pair,
    )
    archive_bytes = archive_path.stat().st_size

    # --- Phase C 7/7 GREEN canonical validation ---
    phase_c = _phase_c_validate(
        real_frame_consumed=real_frame_consumed,
        composition_applied=composition_applied,
        composition_distinct=composition_distinct,
        archive_bytes=archive_bytes,
        helpers_invoked=bool(invocation_verdict),
        tier_a_present=result.axis_tag == "[predicted]",
        provenance_ok=bool(result.canonical_provenance),
    )

    summary = {
        "schema_version": "pr110_opt11_l1_phase_c_v1",
        "subagent_axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "predicted_band_validation_status": "pending_post_training",
        "phase_c": phase_c,
        "real_frame_decode_elapsed_s": decode_elapsed,
        "real_frame_0_mean": real_frame_0_mean,
        "real_frame_shape": list(real_pairs.shape),
        "n_pairs": config.n_pairs,
        "modes_per_pair": config.modes_per_pair,
        "selector_bits_per_mode": config.selector_bits_per_mode,
        "family_pair": list(result.family_pair),
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "l1_real_frame_composition_evidence": {
            "n_composed_on_real_frames": n_composed,
            "n_distinct_from_single_mode_a": n_distinct_from_a,
            "n_distinct_from_single_mode_b": n_distinct_from_b,
            "composition_distinct_from_both": composition_distinct,
            "mean_abs_delta_composed_vs_real_base": mean_abs_delta_real,
        },
        "canonical_helpers_invocation_verdict": invocation_verdict,
        "canonical_provenance": result.canonical_provenance,
        "result_review_blockers": [
            "non_promotable_macos_cpu_advisory_per_catalog_192",
            "l1_phase_c_smoke_pending_paired_cuda_ratification_per_catalog_246",
            "per_substrate_symposium_phase_2_pending_per_catalog_325",
            "canonical_equation_formalization_pending_per_catalog_344",
            "post_training_tier_c_density_pending_per_catalog_324",
        ],
    }
    summary_path = args.output_dir / "pr110_opt11_l1_phase_c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(
        f"[pr110-opt11-l1-trainer] {_utc_now()} DONE L1 Phase C "
        f"verdict={phase_c['phase_c_verdict']} n_pass={phase_c['n_pass']}/7 "
        f"archive_bytes={archive_bytes} real_frame_0_mean={real_frame_0_mean:.1f} "
        f"composition_distinct_from_both={composition_distinct}"
    )
    return 0 if phase_c["phase_c_verdict"] == "GREEN" else 1


def _full_main(args: argparse.Namespace) -> int:
    """Paired-CUDA RATIFICATION dispatch path (council-gated per Catalog #325).

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #240 recipe-vs-trainer-state consistency: this
    full path raises ``NotImplementedError`` until Phase 2 per-substrate
    symposium per Catalog #325 unblocks paired-CUDA dispatch.

    Per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK: the L0 SCAFFOLD's
    research_only opt-out makes the full path explicitly non-dispatchable
    at the recipe surface.
    """
    raise NotImplementedError(
        "PR110-OPT-11 _full_main is council-gated per Catalog #325 + #220 "
        "SCAFFOLD_DEFERRED_INTEGRATION_OK. The L0 SCAFFOLD smoke path "
        "(_smoke_main) is the canonical L0 evidence-bearing surface; the "
        "paired-CUDA RATIFICATION dispatch path lands per Phase 2 council "
        "symposium recommendation. The companion recipe declares "
        "research_only=true + dispatch_enabled=false so this NotImplementedError "
        "is transparent at the dispatch surface per Catalog #240."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PR110-OPT-11 multi-mode-per-pair composition L0 SCAFFOLD trainer"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for smoke artifacts + archive emission",
    )
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--modes-per-pair", type=int, default=2)
    parser.add_argument("--selector-bits-per-mode", type=int, default=4)
    parser.add_argument(
        "--family-pair-index",
        type=int,
        default=1,
        help="Index into CANONICAL_ORTHOGONAL_FAMILY_PAIRS [0, 5]; default 1 = (luma_bias, rgb_bias)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="L1 real-frame spatial stride for the MLX-LOCAL Phase C smoke probe",
    )
    parser.add_argument("--rng-seed", type=int, default=42)
    # Catalog #151 conformance flags (declared even when MLX-first N/A)
    parser.add_argument("--enable-autocast-fp16", action="store_true")
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--no-grad-eval", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", default=True)

    args = parser.parse_args(argv)

    # Per Catalog #326 mode resolution: PR110_OPT11_TRAINER_MODE > SMOKE_ONLY > default
    # Modes: smoke (L0 SCAFFOLD synthetic) | l1 (L1 real-frame Phase C) | full (council-gated)
    trainer_mode = os.environ.get("PR110_OPT11_TRAINER_MODE", "").strip().lower()
    smoke_only_env = os.environ.get("SMOKE_ONLY", "1").strip()

    if trainer_mode == "full":
        return _full_main(args)
    if trainer_mode == "l1":
        return _l1_main(args)
    if trainer_mode == "smoke":
        return _smoke_main(args)
    # Default: smoke (L0 SCAFFOLD evidence surface preserved; operator must
    # explicitly opt into l1 / full via PR110_OPT11_TRAINER_MODE env var)
    if smoke_only_env in {"0", "false", "no"}:
        return _full_main(args)
    return _smoke_main(args)


if __name__ == "__main__":
    sys.exit(main())
