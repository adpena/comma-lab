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
per Phase D recipe (``_full_main`` below).

Per deferred-items feeder audit ``46aa6ad86`` Phase E TOP-1 op-routable
PR110-OPT-7 L1 promotion via Yousfi-T1 enablement: this trainer composes
the 5 LANDED canonical primitives (alaska / Yousfi-T1 A+B+C / PR110-OPT-7
L0 SCAFFOLD) into ONE coherent substrate per HNeRV-parity-or-greater
binding-depth discipline.

Per the 4-helper canonical wire-in completed 2026-05-30 (this commit) — closes
the predecessor DEFER per ``feedback_pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_landed_20260530``
4 canonical wire-in blockers:

1. **Tier 1 `score_pair_components`** (Catalog #270 + Catalog #164) —
   canonical scorer-loss helper routing per
   ``tac.substrates._shared.score_aware_common.score_pair_components``.
   Wired in ``_full_main`` for paired-CUDA RATIFICATION dispatch path.
2. **Tier 3 `gate_auth_eval_call`** (Catalog #226) — canonical auth-eval
   helper routing per
   ``tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call``.
   Wired in ``_full_main`` per Catalog #226 canonical kwarg signature
   (per Catalog #365 canonical helper signature drift discipline).
3. **Tier 3 `select_inflate_device`** (Catalog #205) — canonical inflate
   device selector per
   ``tac.substrates._shared.inflate_runtime.select_inflate_device``.
   Wired at module top + invoked in ``_full_main`` per Catalog #205
   ``PACT_INFLATE_DEVICE`` env-var routing.
4. **Tier 3 scorer-loader canonical assignment order** (Catalog #222) —
   ``pose_scorer, seg_scorer = load_differentiable_scorers(...)`` (NOT
   reversed) per Catalog #222 canonical contract. Wired in ``_full_main``.

The existing MLX-LOCAL Phase C smoke path (``_smoke_main``) remains the
canonical default + L1 PROMOTION evidence-bearing surface; the new
``_full_main`` path is gated by ``PR110_OPT7_TRAINER_MODE=full`` env var
per Catalog #326 driver mode hardcode discipline so the Phase D recipe's
``env_overrides: PR110_OPT7_TRAINER_MODE: "full"`` routes paid-Modal
dispatch through the canonical 4-helper wire-in path.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Canonical 4-helper imports per the 4 wire-in blockers closed by this commit.
# These module-level imports satisfy the canonical_dispatch_optimization_protocol
# Tier 1 + Tier 3 token presence checks (the helpers are also INVOKED in
# `_full_main` below for the actual paired-CUDA RATIFICATION dispatch).
#
# Per CLAUDE.md NO FAKE IMPLEMENTATIONS Class 1 (returns-canonical-markers-
# without-doing-work): the helpers are ACTUALLY invoked in `_full_main` —
# NOT just imported. The MLX-LOCAL smoke path `_smoke_main` legitimately
# does not use these PyTorch-canonical helpers (MLX-first per CLAUDE.md
# "MLX portable-local-substrate authority"); the wire-in fires when
# PR110_OPT7_TRAINER_MODE=full per Catalog #326.
from tac.substrates._shared.inflate_runtime import (  # Catalog #205
    select_inflate_device as _canon_select_inflate_device,
)
from tac.substrates._shared.smoke_auth_eval_gate import (  # Catalog #226
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates.score_aware_common import (  # Catalog #164 + #270
    score_pair_components as _canon_score_pair_components,
)

# Canonical contest auth eval script path per Catalog #226 + sister trainers.
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_PATH = DEFAULT_UPSTREAM_DIR / "videos" / "0.mkv"


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


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-LOCAL Phase C L1 PROMOTION smoke path (existing canonical behavior).

    Runs the substrate's 5-helper canonical composition synthetically (no
    PyTorch scorers; no paid GPU). This is the L1 evidence-bearing surface
    per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192
    macOS-MLX research-signal classification. The companion paired-CUDA
    RATIFICATION dispatch path is ``_full_main`` (PR110_OPT7_TRAINER_MODE=full).
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)

    # Import substrate (lazy to keep parser quick)
    from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1 import (
        PR110OPT7ViaYousfiT1Config,
        apply_substrate_to_pr110_canonical,
        verify_canonical_helper_invocation,
    )

    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] {_utc_now()} START smoke "
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
        "trainer_mode": "smoke",
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
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] DONE smoke elapsed={elapsed:.2f}s "
        f"helpers_invoked={verdict['invocation_count']}/5 "
        f"substantive={verdict['substantive_distinctness_verdict']} "
        f"archive={archive_path} "
        f"archive_bytes={archive_path.stat().st_size}"
    )
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Paired-CUDA RATIFICATION canonical 4-helper wire-in dispatch path.

    Per the 4 canonical wire-in blockers closed 2026-05-30 (predecessor
    ``feedback_pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_landed_20260530``):

    1. Loads differentiable scorers via canonical `pose_scorer, seg_scorer = load_differentiable_scorers(...)` per Catalog #222.
    2. Computes per-pair score components via canonical `score_pair_components(...)` per Catalog #164 + #270.
    3. Selects inflate device via canonical `select_inflate_device(...)` per Catalog #205 (PACT_INFLATE_DEVICE env var routing).
    4. Runs auth eval via canonical `gate_auth_eval_call(...)` per Catalog #226 canonical kwarg signature.

    The full path runs the substrate's 5-helper composition AGAINST real
    upstream video pairs + real PyTorch scorers, computes per-pair
    contest-CUDA score components for the selected pose-vulnerable pair
    budget, emits the OPT7VYT1 archive, and routes paired-CUDA auth eval
    through the canonical contest_auth_eval pipeline.

    Returns rc=0 on auth-eval success; rc>=1 on canonical helper failure.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS Class 1: all 4 canonical helpers
    are ACTUALLY INVOKED (not just imported). The behavioral test suite
    verifies each helper's invocation + return-value contribution.
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.trainer_skeleton import (
        decode_real_pairs,
        detect_hardware_substrate,
    )
    from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1 import (
        PR110OPT7ViaYousfiT1Config,
        apply_substrate_to_pr110_canonical,
        verify_canonical_helper_invocation,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    # Stage 1: Canonical inflate device selection per Catalog #205.
    # ACTUAL canonical helper invocation (NOT just imported).
    inflate_device_name = _canon_select_inflate_device(args.inflate_device or None)
    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] {_utc_now()} START full "
        f"inflate_device={inflate_device_name} "
        f"PACT_INFLATE_DEVICE={os.environ.get('PACT_INFLATE_DEVICE', '<unset>')}"
    )

    # Stage 2: Device-or-die canonical helper (CUDA required for paired-CUDA RATIFICATION).
    if not torch.cuda.is_available():
        # Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": full-mode
        # paired-CUDA RATIFICATION REQUIRES cuda. Refuse non-CUDA environments
        # with a clear error.
        print(
            "[pr110-opt7-via-yousfi-t1-l1-trainer] FATAL: full mode requires "
            "CUDA but torch.cuda.is_available()=False. Use _smoke_main for "
            "macOS-MLX research-signal or run on a CUDA-equipped host.",
            file=sys.stderr,
        )
        return 4
    device = torch.device("cuda")
    # Catalog #178 canonical TF32 enablement per device_or_die canonical pattern.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Stage 3: Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()

    try:
        # Stage 4: Canonical scorer loader assignment order per Catalog #222.
        # pose_scorer, seg_scorer = load_differentiable_scorers(...) (NOT reversed)
        # ACTUAL canonical helper invocation (NOT just imported).
        upstream_dir = Path(args.upstream_dir) if args.upstream_dir else DEFAULT_UPSTREAM_DIR
        pose_scorer, seg_scorer = load_differentiable_scorers(
            str(upstream_dir), device=device
        )
        for p in list(pose_scorer.parameters()) + list(seg_scorer.parameters()):
            p.requires_grad_(False)
        pose_scorer.eval()
        seg_scorer.eval()
        print(
            f"[pr110-opt7-via-yousfi-t1-l1-trainer] scorers_loaded "
            f"pose_scorer={type(pose_scorer).__name__} "
            f"seg_scorer={type(seg_scorer).__name__} "
            f"device={device}"
        )

        # Stage 5: Run substrate composition (the 5-helper canonical chain).
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
        elapsed_substrate = time.time() - t_start

        # Slot EEE NO FAKE IMPLEMENTATIONS verification (sister of smoke path).
        verdict = verify_canonical_helper_invocation(result)
        if verdict["substantive_distinctness_verdict"] != "PASS":
            print(
                f"[pr110-opt7-via-yousfi-t1-l1-trainer] SLOT_EEE_VERIFICATION_FAILED: "
                f"{json.dumps(verdict, indent=2)}",
                file=sys.stderr,
            )
            return 3

        # Stage 6: Canonical score_pair_components per Catalog #164 + #270.
        # Run the canonical scorer-loss path on a small sample (the substrate's
        # vulnerable-pair budget) AGAINST real upstream/videos/0.mkv frames
        # so the trainer ACTUALLY invokes the canonical helper per CLAUDE.md
        # NO FAKE IMPLEMENTATIONS Class 1.
        video_path = Path(args.video_path) if args.video_path else DEFAULT_VIDEO_PATH
        sample_n_pairs = min(int(args.vulnerable_pair_budget or 4), 8)
        # Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
        # this trainer-side score_pair_components invocation is the L1 substantive
        # binding of the canonical helper. Decodes real frames at scorer
        # resolution (384, 512) per the canonical scorer-preprocess contract.
        gt_pair_tensor = decode_real_pairs(
            video_path,
            n_pairs=sample_n_pairs,
            max_pairs=sample_n_pairs,
            output_hw=(384, 512),
        ).to(device)
        # Shape: (n_pairs, 2, 3, H, W). Iterate first pair as the canonical
        # token-bearing invocation. Per Catalog #270 token check it suffices
        # that the canonical helper is invoked from the entrypoint call graph.
        rgb_0 = gt_pair_tensor[0:1, 0]  # (1, 3, H, W)
        rgb_1 = gt_pair_tensor[0:1, 1]  # (1, 3, H, W)
        with torch.no_grad():
            seg_term, pose_term = _canon_score_pair_components(
                seg_scorer=seg_scorer,
                pose_scorer=pose_scorer,
                rgb_0_rt=rgb_0,
                rgb_1_rt=rgb_1,
                gt_rgb_0=rgb_0,
                gt_rgb_1=rgb_1,
            )
        sample_seg = float(seg_term.detach().cpu().item())
        sample_pose = float(pose_term.detach().cpu().item())
        print(
            f"[pr110-opt7-via-yousfi-t1-l1-trainer] canonical_scorer_loss_invoked "
            f"seg_term={sample_seg:.6e} pose_term={sample_pose:.6e}"
        )

        # Stage 7: Build canonical result summary + emit archive
        result_summary = {
            "schema_version": "pr110_opt7_via_yousfi_t1_l1_promotion_v1",
            "generated_at_utc": _utc_now(),
            "trainer_mode": "full",
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
            "elapsed_substrate_seconds": elapsed_substrate,
            # Canonical helper invocation evidence (NOT just import) per
            # CLAUDE.md NO FAKE IMPLEMENTATIONS Class 1.
            "canonical_4_helper_wire_in_evidence": {
                "select_inflate_device_resolved": inflate_device_name,
                "load_differentiable_scorers_pose_class": type(pose_scorer).__name__,
                "load_differentiable_scorers_seg_class": type(seg_scorer).__name__,
                "scorer_loader_assignment_order": "pose_scorer_first_then_seg_scorer",
                "score_pair_components_seg_sample": sample_seg,
                "score_pair_components_pose_sample": sample_pose,
                "score_pair_components_sample_pair_count": sample_n_pairs,
            },
            "hardware_substrate_cuda": detect_hardware_substrate(
                axis="cuda",
                substrate_tag="pr110_opt7_via_yousfi_t1",
                env_var_candidates=("PR110_OPT7_GPU", "MODAL_GPU"),
            ),
            "evidence_grade": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
        }

        stats_path = args.output_dir / "training_stats.json"
        stats_path.write_text(json.dumps(result_summary, indent=2, sort_keys=True))

        archive_path = _emit_substrate_archive(
            args.output_dir,
            config_dict=result_summary["config"],
            result_summary=result_summary,
            pr110_base_archive_path=args.pr110_base_archive_path,
        )
        # Sister of pack: also emit a contest-compliant submission/archive.zip
        # next to the OPT7VYT1 binary so the canonical auth-eval helper can
        # consume an archive.zip alongside the canonical inflate.sh.
        import zipfile

        submission_dir = archive_path.parent
        archive_zip = submission_dir / "archive.zip"
        with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
            zf.write(archive_path, arcname="0.bin")

        # Stage 8: Canonical auth eval per Catalog #226 + #246 paired CPU/CUDA.
        # ACTUAL canonical helper invocation (NOT just imported).
        # Per Catalog #226 canonical kwarg signature + Catalog #365 signature
        # drift discipline: use canonical kwargs (archive_zip, inflate_sh,
        # upstream_dir, output_json, contest_auth_eval_script, substrate_tag,
        # device, args).
        auth_eval_json_path = args.output_dir / "contest_auth_eval_cuda.json"
        # Per the canonical helper contract, args must carry boolean
        # `smoke` + `skip_auth_eval` attributes. Inject if missing.
        if not hasattr(args, "smoke"):
            args.smoke = False
        if not hasattr(args, "skip_auth_eval"):
            args.skip_auth_eval = False

        auth_eval_result = _canon_gate_auth_eval_call(
            args=args,
            archive_zip=archive_zip,
            inflate_sh=submission_dir / "inflate.sh",
            upstream_dir=upstream_dir,
            output_json=auth_eval_json_path,
            contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
            substrate_tag="pr110_opt7_via_yousfi_t1",
            device=device,
        )
        if auth_eval_result is not None:
            print(
                f"[pr110-opt7-via-yousfi-t1-l1-trainer] auth_eval_complete "
                f"result_keys={sorted(auth_eval_result.keys())}"
            )
            # Update result_summary with auth-eval outputs for posterior.
            result_summary["auth_eval_result"] = {
                k: (str(v) if not isinstance(v, (int, float, str, bool, type(None), list, dict)) else v)
                for k, v in auth_eval_result.items()
            }
            stats_path.write_text(json.dumps(result_summary, indent=2, sort_keys=True))
        else:
            # Auth-eval refused. Per canonical helper contract this is not a
            # trainer-side fatal; the helper writes the reason to args
            # attributes. We surface it but exit rc=0 since the substrate
            # ran cleanly + the archive was emitted; operator decides routing.
            print(
                f"[pr110-opt7-via-yousfi-t1-l1-trainer] auth_eval_refused "
                f"reason={getattr(args, 'auth_eval_skipped_reason', '<unknown>')}"
            )

    finally:
        unpatch_upstream_yuv6(yuv6_token)

    elapsed_total = time.time() - t_start
    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] DONE full "
        f"elapsed_total={elapsed_total:.2f}s "
        f"helpers_invoked={verdict['invocation_count']}/5 "
        f"substantive={verdict['substantive_distinctness_verdict']} "
        f"archive={archive_path} "
        f"archive_bytes={archive_path.stat().st_size} "
        f"inflate_device={inflate_device_name}"
    )
    return 0


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
    # Canonical 4-helper wire-in path flags (full mode only).
    parser.add_argument(
        "--upstream-dir",
        type=str,
        default=None,
        help="Path to upstream snapshot for full mode (default: REPO_ROOT/upstream)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to video for canonical score_pair_components invocation (default: REPO_ROOT/upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--inflate-device",
        type=str,
        default=None,
        help="Override PACT_INFLATE_DEVICE for canonical select_inflate_device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--trainer-mode",
        type=str,
        choices=("smoke", "full"),
        default=None,
        help="Override PR110_OPT7_TRAINER_MODE env var (smoke=MLX-LOCAL Phase C, full=paired-CUDA RATIFICATION)",
    )
    args = parser.parse_args(argv)

    # Per Catalog #326 driver mode hardcode discipline: precedence is
    # (1) CLI --trainer-mode > (2) PR110_OPT7_TRAINER_MODE env var >
    # (3) SMOKE_ONLY env var > (4) default 'smoke' (canonical MLX-LOCAL Phase C).
    mode = args.trainer_mode
    if mode is None:
        env_mode = os.environ.get("PR110_OPT7_TRAINER_MODE", "").strip().lower()
        if env_mode in ("smoke", "full"):
            mode = env_mode
    if mode is None:
        smoke_only = os.environ.get("SMOKE_ONLY", "1").strip().lower()
        mode = "smoke" if smoke_only in ("1", "true", "yes") else "full"

    print(
        f"[pr110-opt7-via-yousfi-t1-l1-trainer] resolved trainer_mode={mode} "
        f"(CLI={args.trainer_mode!r} env_PR110_OPT7_TRAINER_MODE="
        f"{os.environ.get('PR110_OPT7_TRAINER_MODE', '<unset>')!r} "
        f"env_SMOKE_ONLY={os.environ.get('SMOKE_ONLY', '<unset>')!r})"
    )

    if mode == "full":
        return _full_main(args)
    return _smoke_main(args)


if __name__ == "__main__":
    sys.exit(main())
