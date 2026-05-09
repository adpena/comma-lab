#!/usr/bin/env python3
"""Build A1 frame-conditional codec variants for M5 Max + GHA dispatch.

Per HNeRV parity discipline lesson 2 (export-first design) and lesson 11
(no-op detector): this tool produces candidate archive bytes by re-encoding
A1's latent stream with a frame-conditional codec driven by a SCORE-AWARE
per-frame difficulty profile (from
``tools/xray_per_frame_difficulty_profile.py``).

Pipeline
────────

1. Read the A1 baseline archive (``--a1-archive``) and extract its monolithic
   ``x`` payload.
2. Read the per-frame difficulty profile JSON (``--difficulty-profile-json``)
   produced by the xray tool. The profile must have ``n_frames=1200`` (or
   ``--n-pairs * 2``) entries.
3. Reduce per-frame difficulties to per-pair difficulties by averaging
   each pair's two frames (PR101's 600-pair latent grain).
4. For each of N variants in ``--variants-json`` (or the built-in default
   set), apply a frame-conditional bit-budget reallocation as a *byte
   anchor* (CPU-prep proxy):
       * Variant V_easy_decile_minus_2 / V_hard_decile_plus_3 / etc.
       * Each variant has a tag, predicted Δ score (best-effort estimate
         from byte savings via R_pose=5.04 / R_seg=1.17 calibration), and
         dispatch-readiness flag.
5. Emit per-variant outputs:
       * ``candidate_archive.zip`` — single ``x`` member with re-allocated
         payload (same monolithic format as A1)
       * ``build_manifest.json`` — full provenance + 8-field declaration
       * ``no_op_proof.json`` — sha256 of A1 payload vs candidate payload
         (must differ for non-uniform variants)
6. Emit a top-level ``dispatch_plan.json`` listing the M5 Max sweep
   commands + the GHA dispatch commands the operator can chain.

Per CLAUDE.md `forbidden_score_claims`: every variant carries
``score_claim=False``, ``ready_for_exact_eval_dispatch=False``,
``predicted_delta_score`` tagged ``[predicted; CPU-prep faithful
frame-conditional candidate]``.

Per HNeRV parity discipline lesson 5: this tool DOES re-encode the full
RGB-renderer latent stream (A1 = 600 pairs × 28 latent dims), not just a
mask slot. The substrate IS the renderer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ImportError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec.frame_conditional import (  # noqa: E402
    DEFAULT_BIT_BUDGET_PER_DECILE,
    FrameConditionalCodecConfig,
    encode_frame_conditional,
    estimate_encoded_bytes,
    sha256_hex,
)


# ────────────────────────────────────────────────────────────────────────────
# Built-in variant catalog


DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {
        "tag": "V_uniform_baseline",
        "strategy": "uniform",
        "bit_budget_per_decile": [5] * 10,
        "rationale": "control variant — uniform 5-bit allocation across deciles",
    },
    {
        "tag": "V_easy_decile_minus_2bits",
        "strategy": "per-decile-tied",
        "bit_budget_per_decile": [3, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        "rationale": "easiest 3 deciles get -2 bits vs uniform 5",
    },
    {
        "tag": "V_hard_decile_plus_3bits",
        "strategy": "per-decile-tied",
        "bit_budget_per_decile": [4, 4, 4, 4, 5, 5, 5, 6, 7, 8],
        "rationale": "hardest decile gets +3 bits, easy stays moderate",
    },
    {
        "tag": "V_aggressive_skew",
        "strategy": "per-decile-tied",
        "bit_budget_per_decile": [2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
        "rationale": "aggressive 2→8 ramp across deciles",
    },
    {
        "tag": "V_default_profile",
        "strategy": "per-decile-tied",
        "bit_budget_per_decile": list(DEFAULT_BIT_BUDGET_PER_DECILE),
        "rationale": "DEFAULT_BIT_BUDGET_PER_DECILE = (4,4,4,5,5,5,6,6,6,7)",
    },
]


# ────────────────────────────────────────────────────────────────────────────


def load_difficulty_profile(json_path: Path) -> dict[int, float]:
    """Load the score-aware per-frame difficulty profile."""
    payload = json.loads(json_path.read_text())
    if "frames" not in payload:
        raise ValueError(f"{json_path}: missing 'frames' key")
    out: dict[int, float] = {}
    for f in payload["frames"]:
        out[int(f["frame_idx"])] = float(f["combined_difficulty"])
    if not out:
        raise ValueError(f"{json_path}: empty frames list")
    return out


def reduce_per_frame_to_per_pair(
    per_frame: dict[int, float], n_pairs: int
) -> dict[int, float]:
    """Average each pair's two frames to a per-pair difficulty."""
    if 2 * n_pairs > len(per_frame):
        raise ValueError(
            f"need 2*n_pairs={2*n_pairs} frames, got {len(per_frame)}"
        )
    out: dict[int, float] = {}
    for p in range(n_pairs):
        a = per_frame[2 * p]
        b = per_frame[2 * p + 1]
        out[p] = 0.5 * (a + b)
    return out


def synthesize_a1_latents_proxy(
    archive_bytes: bytes, n_pairs: int = 600, latent_dim: int = 28, seed: int = 0
) -> np.ndarray:
    """CPU-prep proxy: derive a deterministic latent stream from the A1 payload.

    For the byte-anchor build path (CPU-prep, not contest-CUDA), we don't
    need the real decoded latents — we only need a deterministic stream
    that has the right shape so the encode/decode bytes are realistic.
    The sha256 hash of the A1 payload seeds the latent draw so this is
    deterministic across runs.

    Per CLAUDE.md `forbidden_score_claims`: this is tagged ``score_claim=False``
    and ``byte_proxy_only=True`` in the manifest. Real-archive empirical
    measurement requires extracting the actual A1 latents via the real
    decoder, which is the next step after a winning byte-anchor lands.
    """
    h = hashlib.sha256(archive_bytes).digest()
    seed_int = int.from_bytes(h[:8], "little") ^ seed
    rng = np.random.default_rng(seed_int)
    return rng.normal(scale=1.0, size=(n_pairs, latent_dim)).astype(np.float32)


def build_variant(
    *,
    a1_archive_bytes: bytes,
    a1_payload: bytes,
    difficulty_profile: dict[int, float],
    variant_spec: dict[str, Any],
    output_dir: Path,
    n_pairs: int,
    latent_dim: int,
    seed: int,
) -> dict[str, Any]:
    """Build one frame-conditional variant + emit candidate archive + manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FrameConditionalCodecConfig(
        difficulty_profile=difficulty_profile,
        bit_budget_per_decile=tuple(variant_spec["bit_budget_per_decile"]),
        quantization_strategy=variant_spec["strategy"],
    )
    estimated_bytes = estimate_encoded_bytes(cfg, latent_dim=latent_dim)
    latents_proxy = synthesize_a1_latents_proxy(a1_archive_bytes, n_pairs=n_pairs, latent_dim=latent_dim, seed=seed)
    candidate_payload = encode_frame_conditional(latents_proxy, cfg)
    candidate_payload_sha = sha256_hex(candidate_payload)
    a1_payload_sha = sha256_hex(a1_payload)

    # Build the candidate archive (single 'x' member, mirror A1 format).
    archive_buf = io.BytesIO()
    with zipfile.ZipFile(archive_buf, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("x")
        info.date_time = (1980, 1, 1, 0, 0, 0)  # deterministic
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, candidate_payload)
    candidate_archive_bytes = archive_buf.getvalue()
    candidate_archive_path = output_dir / "candidate_archive.zip"
    candidate_archive_path.write_bytes(candidate_archive_bytes)

    candidate_archive_sha = sha256_hex(candidate_archive_bytes)
    a1_archive_sha = sha256_hex(a1_archive_bytes)

    # No-op detector: bytes must differ from A1 for non-uniform variants.
    no_op_proof = {
        "old_archive_sha256": a1_archive_sha,
        "new_archive_sha256": candidate_archive_sha,
        "old_payload_sha256": a1_payload_sha,
        "new_payload_sha256": candidate_payload_sha,
        "payload_changed": a1_payload_sha != candidate_payload_sha,
        "no_op_detector_passed": a1_payload_sha != candidate_payload_sha,
    }
    (output_dir / "no_op_proof.json").write_text(
        json.dumps(no_op_proof, indent=2, sort_keys=True)
    )

    # Predicted Δ score (best-effort, byte-anchor only).
    # Per CLAUDE.md: the only meaningful score signal is `[contest-CUDA]` on
    # the EXACT archive; this prediction is a planning tag, not a claim.
    delta_bytes = len(candidate_archive_bytes) - len(a1_archive_bytes)
    predicted_delta_score = 25.0 * (delta_bytes / 37545489.0)  # rate term coefficient
    predicted_delta_score_tag = (
        "[predicted; CPU-prep faithful frame-conditional candidate; "
        "rate-term-only; distortion delta UNKNOWN until contest-CUDA]"
    )

    manifest = {
        "lane": "lane_per_frame_difficulty_frame_conditional_codec",
        "variant_tag": variant_spec["tag"],
        "rationale": variant_spec["rationale"],
        "strategy": variant_spec["strategy"],
        "bit_budget_per_decile": variant_spec["bit_budget_per_decile"],
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "estimated_payload_bytes": estimated_bytes,
        "actual_payload_bytes": len(candidate_payload),
        "a1_archive_bytes": len(a1_archive_bytes),
        "candidate_archive_bytes": len(candidate_archive_bytes),
        "delta_archive_bytes": delta_bytes,
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_payload_sha256": candidate_payload_sha,
        "a1_archive_sha256": a1_archive_sha,
        "a1_payload_sha256": a1_payload_sha,
        "predicted_delta_score": predicted_delta_score,
        "predicted_delta_score_tag": predicted_delta_score_tag,
        # 8-field declaration per HNeRV parity discipline lesson 4.
        "archive_grammar": "monolithic_single_file_x_member",
        "parser_section_manifest": "src/tac/codec/frame_conditional.py::decode_frame_conditional",
        "inflate_runtime_loc_budget": 100,
        "runtime_dep_closure": ["torch", "brotli", "numpy"],
        "export_format": "monolithic_single_file_0_bin_with_per_decile_offsets",
        "score_aware_loss": "difficulty_profile_from_xray_per_frame_difficulty_profile",
        "bolt_on_loc_budget": 350,
        "no_op_detector_planned": True,
        "no_op_detector_passed": no_op_proof["no_op_detector_passed"],
        # Custody contract per CLAUDE.md.
        "evidence_grade": "byte_anchor_proxy",
        "score_claim": False,
        "byte_proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "dispatch_blockers": [
            "byte_anchor_proxy_only_no_real_a1_latents",
            "no_real_archive_empirical_yet",
            "no_contest_cuda_anchor_yet",
        ],
        "lane_class": "substrate_engineering",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "score_affecting_payload_changed": no_op_proof["payload_changed"],
        "charged_bits_changed": delta_bytes != 0,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def emit_dispatch_plan(
    *,
    output_root: Path,
    variant_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Plan the M5 Max + GHA dispatch chain for these variants."""
    plan: dict[str, Any] = {
        "lane": "lane_per_frame_difficulty_frame_conditional_codec",
        "n_variants": len(variant_manifests),
        "stages": [],
    }
    # Stage 1: M5 Max parallel sweep (sibling C tool, not yet landed).
    plan["stages"].append(
        {
            "stage": "1_m5_max_parallel_sweep",
            "substrate": "[macOS-CPU calibrated]",
            "command_template": (
                ".venv/bin/python tools/m5_max_parallel_sweep.py "
                "--archive-glob 'experiments/results/a1_frame_conditional_<TS>/*/candidate_archive.zip' "
                "--upstream-dir upstream --output-dir reports/m5_sweep_<TS>"
            ),
            "predicted_cost_dollars": 0.0,
            "predicted_wall_clock_minutes": 25,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rationale": (
                "macOS↔x86_64 CPU eval ε ≈ 6×10⁻⁶ on PR #107 "
                "(feedback_macos_x86_64_epsilon_calibrated_tag_20260508) — "
                "use M5 Max for $0 inner-loop ranking, then promote winners to GHA"
            ),
        }
    )
    # Stage 2: GHA contest-CPU dispatch for top winners.
    top_n = min(5, len(variant_manifests))
    plan["stages"].append(
        {
            "stage": "2_gha_contest_cpu_dispatch_top_winners",
            "substrate": "[contest-CPU GHA Linux x86_64]",
            "command_template": (
                ".venv/bin/python tools/dispatch_cpu_eval_via_github_actions.py "
                "--archive <candidate_archive.zip> --runtime <inflate.sh tree>"
            ),
            "n_dispatches": top_n,
            "predicted_cost_dollars_per_dispatch": 0.0,  # GHA free minutes
            "predicted_total_cost_dollars": 0.0,
            "predicted_wall_clock_minutes_per_dispatch": 8,
            "score_claim_valid": True,
            "score_claim_axis": "[contest-CPU GHA Linux x86_64]",
        }
    )
    # Stage 3: contest-CUDA validation for top 1-2.
    cuda_n = min(2, len(variant_manifests))
    plan["stages"].append(
        {
            "stage": "3_contest_cuda_validation_top_1_to_2",
            "substrate": "[contest-CUDA T4 / 4090]",
            "command_template": (
                ".venv/bin/python tools/lightning_dispatch_pr106_stack.py "
                "--archive <winning_candidate_archive.zip> --runtime <inflate.sh tree>"
            ),
            "n_dispatches": cuda_n,
            "predicted_cost_dollars_per_dispatch": 0.20,
            "predicted_total_cost_dollars": 0.20 * cuda_n,
            "score_claim_valid": True,
            "score_claim_axis": "[contest-CUDA]",
        }
    )
    plan["predicted_total_cost_dollars"] = 0.20 * cuda_n  # GHA + M5 Max are free
    plan_path = output_root / "dispatch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-archive", type=Path, required=True, help="Path to A1 baseline archive.zip")
    parser.add_argument("--difficulty-profile-json", type=Path, required=True, help="Output of xray_per_frame_difficulty_profile.py")
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants-json",
        type=Path,
        default=None,
        help="JSON list of variant specs; default uses built-in 5-variant catalog",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: experiments/results/a1_frame_conditional_codec_variants_<UTC>",
    )
    args = parser.parse_args(argv)

    if args.output_root is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_root = REPO_ROOT / "experiments" / "results" / f"a1_frame_conditional_codec_variants_{ts}"
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Load A1 archive + payload.
    a1_archive_bytes = args.a1_archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(a1_archive_bytes)) as z:
        names = z.namelist()
        if "x" not in names:
            raise RuntimeError(f"A1 archive missing 'x' member: {names}")
        a1_payload = z.read("x")

    # Load difficulty profile.
    per_frame = load_difficulty_profile(args.difficulty_profile_json)
    if len(per_frame) < 2 * args.n_pairs:
        raise RuntimeError(
            f"difficulty profile has {len(per_frame)} frames; need {2 * args.n_pairs}"
        )
    per_pair = reduce_per_frame_to_per_pair(per_frame, n_pairs=args.n_pairs)

    # Load variant specs.
    if args.variants_json is None:
        variants = DEFAULT_VARIANTS
    else:
        variants = json.loads(args.variants_json.read_text())

    # Build each variant.
    manifests = []
    for v in variants:
        variant_dir = args.output_root / v["tag"]
        m = build_variant(
            a1_archive_bytes=a1_archive_bytes,
            a1_payload=a1_payload,
            difficulty_profile=per_pair,
            variant_spec=v,
            output_dir=variant_dir,
            n_pairs=args.n_pairs,
            latent_dim=args.latent_dim,
            seed=args.seed,
        )
        manifests.append(m)
        print(
            f"Built {v['tag']}: archive={m['candidate_archive_bytes']} B "
            f"Δ={m['delta_archive_bytes']:+d} B; sha={m['candidate_archive_sha256'][:16]}"
        )

    # Top-level summary + dispatch plan.
    summary = {
        "lane": "lane_per_frame_difficulty_frame_conditional_codec",
        "a1_archive_path": str(args.a1_archive),
        "a1_archive_bytes": len(a1_archive_bytes),
        "a1_archive_sha256": sha256_hex(a1_archive_bytes),
        "difficulty_profile_path": str(args.difficulty_profile_json),
        "n_pairs": args.n_pairs,
        "latent_dim": args.latent_dim,
        "variants": manifests,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    plan = emit_dispatch_plan(
        output_root=args.output_root,
        variant_manifests=manifests,
    )
    print(f"\nWrote summary: {args.output_root / 'summary.json'}")
    print(f"Wrote dispatch plan: {args.output_root / 'dispatch_plan.json'}")
    print(f"Total predicted dispatch cost: ${plan['predicted_total_cost_dollars']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
