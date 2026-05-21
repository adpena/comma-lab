# SPDX-License-Identifier: MIT
"""Trainer for the pre-trained driving prior substrate.

Phase 1: distill the codebook OFFLINE from a public dashcam dataset (or
deterministic synthetic — for scaffold smoke). This step writes a frozen
codebook artifact under ``experiments/results/<lane>/codebook.bin``.

Phase 2: load the codebook, train the small contest-overfit renderer + per-pair
int8 residual against the contest video with the score-aware Lagrangian
(eval-roundtrip + Atick-Redlich cooperative-receiver + soft codebook prior).

Phase 3: pack the codebook + renderer + residual + meta into a DP1 archive
and run contest-CUDA + contest-CPU auth eval (both axes per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA").

Phase 2 (this landing) wires :func:`_full_main`. Real Comma2k19 distillation
runs through ``tac.substrates.pretrained_driving_prior.Comma2k19FrameIterator``
when ``--dataset-name=comma2k19`` and ``--comma2k19-chunks-dir <path>``;
synthetic stub remains the default for tests / structural smoke.

Catalog #146 inflate.sh contract: the trainer's ``_write_runtime`` emits
the contest 3-positional-arg ``inflate.sh <archive_dir> <output_dir> <file_list>``.
Catalog #151: ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` declares every required
flag so operator wrappers thread env-vars correctly.
Catalog #152: ``required_input_file=True`` flags trigger pre-dispatch
filesystem validation in the operator wrapper.
Catalog #164: scorer preprocess routed through canonical
``tac.substrates.score_aware_common.score_pair_components``.
Catalog #178: TF32 matmul routed through canonical
``tac.substrates._shared.trainer_skeleton.device_or_die``.
Catalog #190: ``hardware_substrate`` dynamically detected — never hardcoded.
Catalog #209: every call to :func:`distill_codebook` in this file is
constructed via ``Comma2k19FrameIterator`` so the leakage guard runs first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrate_registry import SubstrateContract, register_substrate

REPO_ROOT = Path(__file__).resolve().parent.parent

# CLAUDE.md Catalog #151 + #152: declare required flags so operator wrappers
# thread env-vars + pre-dispatch file validation runs.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "DPP_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "generator_command": "ls upstream/videos/0.mkv",
    },
    "--output-dir": {
        "env": "DPP_OUTPUT_DIR",
        "default": "experiments/results/lane_pretrained_driving_prior_lane_scaffold_20260513",
        "required_input_file": False,
    },
    "--upstream-dir": {
        "env": "DPP_UPSTREAM_DIR",
        "default": "upstream",
        "required_input_file": False,
    },
    "--device": {
        "env": "DPP_DEVICE",
        "default": "cuda",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "DPP_EPOCHS",
        "default": "2000",
        "required_input_file": False,
    },
    "--batch-size": {
        "env": "DPP_BATCH_SIZE",
        "default": "1",
        "required_input_file": False,
    },
    "--dataset-name": {
        "env": "DPP_DATASET_NAME",
        "default": "synthetic_test",
        "required_input_file": False,
        "satisfied_by_profile": ["smoke", "comma2k19_full"],
    },
    "--enable-autocast-fp16": {
        "env": "DPP_ENABLE_AUTOCAST_FP16",
        "default": "0",
        "required_input_file": False,
    },
    "--comma2k19-chunks-dir": {
        "env": "DPP_COMMA2K19_CHUNKS_DIR",
        "default": "",
        "required_input_file": False,
        # Required only when --dataset-name=comma2k19 (validated at parse time).
        "satisfied_by_profile": ["smoke", "comma2k19_full"],
    },
    # Catalog #213 auto-download canonical-helper wire-up. When
    # --comma2k19-chunks-dir is empty AND --dataset-name=comma2k19, the
    # trainer constructs a Comma2k19LocalCache rooted at --cache-dir and
    # the log-incremental schedule streams chunks on-demand. This unblocks
    # Phase 2 dispatch without the operator manually provisioning a chunks
    # dir on every Modal/Vast.ai worker.
    "--cache-dir": {
        "env": "DPP_CACHE_DIR",
        "default": "",  # empty = use default_cache_dir() = ~/.cache/tac/comma2k19_chunks
        "required_input_file": False,
        "satisfied_by_profile": ["smoke", "comma2k19_full"],
    },
    "--max-disk-gb": {
        "env": "DPP_MAX_DISK_GB",
        "default": "100.0",
        "required_input_file": False,
    },
    "--log-incremental-base": {
        "env": "DPP_LOG_INCREMENTAL_BASE",
        "default": "2",
        "required_input_file": False,
    },
    "--log-incremental-max-chunks": {
        "env": "DPP_LOG_INCREMENTAL_MAX_CHUNKS",
        "default": "80",
        "required_input_file": False,
    },
    "--log-incremental-quality-threshold": {
        "env": "DPP_LOG_INCREMENTAL_QUALITY_THRESHOLD",
        "default": "0.005",
        "required_input_file": False,
    },
    # WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — procedural codebook
    # replacement variant flags per `dp1_paired_smoke_dispatch_pre_authorization_
    # checklist_20260520` OP-ROUTABLE #2 + parent design memo §4 paired-smoke
    # recipe spec. Six env-var flags wire the 3 recipe variants (original /
    # procedural / null_exploit_control). Catalog #151 + #152 + #324 sister.
    "--enable-procedural-codebook-replacement": {
        "env": "DPP_PROCEDURAL_CODEBOOK_REPLACEMENT",
        "default": "0",
        "required_input_file": False,
    },
    "--procedural-codebook-seed-hex": {
        "env": "DPP_PROCEDURAL_CODEBOOK_SEED_HEX",
        "default": "",
        "required_input_file": False,
    },
    "--procedural-codebook-generator-kind": {
        "env": "DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND",
        "default": "pcg64",
        "required_input_file": False,
    },
    "--procedural-codebook-null-exploit-control": {
        "env": "DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL",
        "default": "0",
        "required_input_file": False,
    },
    "--procedural-codebook-validate-domain": {
        "env": "DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN",
        "default": "1",
        "required_input_file": False,
    },
    "--procedural-variant-provenance-path": {
        "env": "DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH",
        "default": "",
        "required_input_file": False,
    },
    "--procedural-variant-distillation-skip": {
        "env": "DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP",
        "default": "0",
        "required_input_file": False,
    },
}


# Score-aware constants (mirrored from sister substrates).
EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer resolution (height, width) for eval pair tensors."""

CONTEST_NORMALIZER: float = 37_545_489.0
"""Contest score's bytes-per-rate normalizer (Σ pair bytes)."""

N_PAIRS_FULL: int = 600
"""Default Phase 2 contest pair count (matching upstream evaluate.py)."""

CONTEST_AUTH_EVAL_SCRIPT: Path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL: Path = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

LANE_ID_PHASE_2: str = "lane_pretrained_driving_prior_phase_2_20260514"


def build_argparser() -> argparse.ArgumentParser:
    """Build the canonical argparse for the trainer.

    Every flag in ``TIER_1_OPERATOR_REQUIRED_FLAGS`` must appear here as a
    ``parser.add_argument(...)``; the Catalog #12 ``preflight_arity`` check
    enforces caller-side flag subsets so this contract is bidirectional.
    """
    parser = argparse.ArgumentParser(
        description="Train the pre-trained driving prior substrate (DP1).",
    )
    parser.add_argument("--video-path", default="upstream/videos/0.mkv")
    parser.add_argument(
        "--output-dir",
        default="experiments/results/lane_pretrained_driving_prior_lane_scaffold_20260513",
    )
    parser.add_argument("--upstream-dir", default="upstream")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dataset-name",
        default="synthetic_test",
        choices=("synthetic_test", "comma2k19", "comma10k", "comma10k19", "bdd100k"),
    )
    parser.add_argument(
        "--allow-bdd100k-dataset-images",
        action="store_true",
        help="Opt in to BDD100K UC-Berkeley non-commercial-research dataset images.",
    )
    parser.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        help="Wrap forward in torch.autocast(fp16) + GradScaler (Tier 1 engineering win).",
    )
    # Catalog #178 sister: TF32 fast-math (substrate-wide pattern; defaults on for CUDA).
    parser.add_argument(
        "--enable-tf32",
        action="store_true",
        default=True,
        help="Enable TF32 matmul/conv kernels on Ampere/Hopper (default on).",
    )
    # Catalog #172 sister: optional torch.compile for the renderer.
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Wrap the renderer with torch.compile (Inductor; Tier 2 engineering).",
    )
    parser.add_argument(
        "--enable-gt-scorer-cache",
        action="store_true",
        default=False,
        help=(
            "F3 GTScorerCache: pre-compute the GT scorer outputs once and "
            "reuse across the hot loop (~50%% scorer compute savings; "
            "mathematically identical to GT-forward path). Catalog #228. "
            "Default OFF preserves byte-faithful historical behavior; opt-in "
            "is operator-routed via this flag for the substrates that have "
            "been F3-wired (PDP wire-in landed 2026-05-14 commit 0916332eb)."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke path (CPU OK).")
    parser.add_argument(
        "--codebook-path",
        default="",
        help="Optional pre-distilled codebook .bin path. If empty, distill at start.",
    )
    parser.add_argument(
        "--dataset-frames-dir",
        default="",
        help="Optional directory of pre-extracted dashcam frame .png/.jpg files.",
    )
    parser.add_argument(
        "--comma2k19-chunks-dir",
        default="",
        help=(
            "Path to a Comma2k19 chunk directory (MIT-licensed; "
            "github.com/commaai/comma2k19). Required when "
            "--dataset-name=comma2k19 UNLESS --cache-dir is supplied to "
            "auto-download via Comma2k19LocalCache. The Comma2k19FrameIterator "
            "runs the Catalog #209 contest-video-leakage guard before any "
            "decode."
        ),
    )
    # Catalog #213 — Comma2k19 auto-download canonical cache.
    parser.add_argument(
        "--cache-dir",
        default="",
        help=(
            "Local cache dir for auto-downloaded Comma2k19 chunks "
            "(MIT-licensed; SHA-256-verified). Empty = use "
            "~/.cache/tac/comma2k19_chunks (default_cache_dir()). When set "
            "with --dataset-name=comma2k19 and an empty --comma2k19-chunks-dir, "
            "the trainer auto-downloads via Comma2k19LocalCache + uses the "
            "log-incremental schedule to feed distillation."
        ),
    )
    parser.add_argument(
        "--max-disk-gb",
        type=float,
        default=100.0,
        help=(
            "Maximum disk usage for the Comma2k19 cache before LRU "
            "eviction kicks in. The cache refuses new fetches that "
            "would exceed this even after evicting all stored entries."
        ),
    )
    parser.add_argument(
        "--log-incremental-base",
        type=int,
        default=2,
        help=(
            "Exponential base for the log-incremental chunk schedule. "
            "Default 2 yields the canonical doubling schedule "
            "[1, 2, 4, 8, 16, 32, 64, 80]."
        ),
    )
    parser.add_argument(
        "--log-incremental-max-chunks",
        type=int,
        default=80,
        help=(
            "Cap on the number of cumulative chunks the log-incremental "
            "schedule visits. Default 80 = full Comma2k19 corpus size."
        ),
    )
    parser.add_argument(
        "--log-incremental-quality-threshold",
        type=float,
        default=0.005,
        help=(
            "Marginal-improvement plateau threshold for early-stop. "
            "When (quality[n-1] - quality[n]) < this threshold and at "
            "least 3 schedule steps have run, distillation stops. "
            "Set to 0.0 to disable early-stop."
        ),
    )
    parser.add_argument(
        "--disable-log-incremental",
        action="store_true",
        help=(
            "Disable the log-incremental schedule and fall back to a single "
            "distill_codebook() call on the operator-supplied "
            "--comma2k19-chunks-dir. Used when the operator wants exact "
            "control over the chunk count (e.g. for byte-stable replays)."
        ),
    )
    # Operator pivot 2026-05-14: streaming + log mode (NO permanent disk cache).
    parser.add_argument(
        "--use-streamer",
        action="store_true",
        help=(
            "OPERATOR PIVOT 2026-05-14: route Comma2k19 chunk access through "
            "Comma2k19LocalStreamer (streaming + JSONL log; NO permanent disk "
            "cache) instead of Comma2k19LocalCache. Mutually exclusive with "
            "--cache-dir. Operator pivot from auto-download to streaming + log."
        ),
    )
    parser.add_argument(
        "--stream-log-dir",
        default="",
        help=(
            "Directory for the date-rotated JSONL stream-access log (defaults "
            "to ~/.cache/tac/comma2k19_stream_logs). Required when "
            "--use-streamer is set."
        ),
    )
    parser.add_argument(
        "--ram-buffer-gb",
        type=float,
        default=2.0,
        help=(
            "In-memory streaming buffer cap in gigabytes. LRU eviction once "
            "the cap is reached. There is NO permanent disk cache in streamer "
            "mode."
        ),
    )
    parser.add_argument(
        "--streamer-frames-per-chunk",
        type=int,
        default=256,
        help=(
            "Max frames decoded per streamed chunk in --use-streamer mode "
            "(cost+memory). Streamed bytes are discarded after decode."
        ),
    )
    parser.add_argument(
        "--stream-chunking-mode",
        default="frame_range",
        choices=(
            "frame_range",
            "motion_class",
            "entropy",
            "saliency",
            "byte_size",
            "temporal_window",
        ),
        help=(
            "Dynamic streamer chunk planner. frame_range preserves source "
            "order; motion/entropy/saliency prioritize metadata-scored chunks."
        ),
    )
    parser.add_argument(
        "--stream-frame-range-size",
        type=int,
        default=256,
        help="Frame window used by frame_range/motion/entropy/saliency chunking.",
    )
    parser.add_argument(
        "--stream-byte-size-target",
        type=int,
        default=0,
        help="Target bytes per chunk for byte_size chunking; 0 uses streamer default.",
    )
    parser.add_argument(
        "--stream-temporal-window-sec",
        type=float,
        default=0.0,
        help="Seconds per chunk for temporal_window chunking; 0 uses 1 second.",
    )
    parser.add_argument(
        "--stream-motion-threshold",
        type=float,
        default=None,
        help="Optional minimum metadata score for motion_class chunking.",
    )
    parser.add_argument(
        "--stream-entropy-threshold",
        type=float,
        default=None,
        help="Optional minimum metadata score for entropy chunking.",
    )
    parser.add_argument(
        "--stream-saliency-topk",
        type=int,
        default=None,
        help="Optional top-k metadata chunks for saliency chunking.",
    )
    parser.add_argument(
        "--max-distillation-frames",
        type=int,
        default=4096,
        help=(
            "Hard cap on frames decoded for codebook distillation (cost+memory)."
        ),
    )
    parser.add_argument(
        "--max-distillation-chunks",
        type=int,
        default=8,
        help=(
            "Cap on Comma2k19 chunks visited during distillation "
            "(each chunk ~60 sec @ 20 Hz)."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=N_PAIRS_FULL,
        help="Cap on contest pairs decoded for the score-aware loop (default 600).",
    )
    parser.add_argument(
        "--val-pair-count",
        type=int,
        default=64,
        help="Held-out validation pair count.",
    )
    parser.add_argument(
        "--val-every-epochs",
        type=int,
        default=50,
        help="Validation cadence (epochs).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0xDA5C)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="Canonical EMA decay (CLAUDE.md non-negotiable; default 0.997).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="Per-pixel additive noise during training (zero at eval).",
    )
    parser.add_argument(
        "--alpha-rate", type=float, default=25.0, help="Score-domain rate weight."
    )
    parser.add_argument(
        "--beta-seg", type=float, default=100.0, help="SegNet score-domain weight."
    )
    parser.add_argument(
        "--gamma-pose",
        type=float,
        default=math.sqrt(10.0),
        help="PoseNet score-domain sqrt-weight (contest formula).",
    )
    parser.add_argument(
        "--pose-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Operating-point tilt; 1.0 = contest formula. PR106 r2 has 2.71x "
            "pose marginal sensitivity at saturated operating points."
        ),
    )
    parser.add_argument(
        "--delta-prior",
        type=float,
        default=0.05,
        help="Codebook soft-prior weight in the joint Lagrangian.",
    )
    parser.add_argument(
        "--per-pair-bytes",
        type=int,
        default=12,
        help="Per-pair int8 residual budget bytes.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=3,
        help="DrivingPriorRenderer hidden depth.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="DrivingPriorRenderer hidden dim.",
    )
    parser.add_argument("--skip-archive-build", action="store_true")
    parser.add_argument("--skip-auth-eval", action="store_true")
    parser.add_argument(
        "--full-cpu",
        action="store_true",
        help=(
            "Permit the full training path on CPU (research / advisory). MUST "
            "be paired with --advisory-cpu-explicitly-waived per Catalog #197 "
            "+ CLAUDE.md MPS / non-1:1-CPU non-negotiable."
        ),
    )
    parser.add_argument(
        "--advisory-cpu-explicitly-waived",
        action="store_true",
        help=(
            "Operator attestation that this CPU run is research / advisory "
            "ONLY (NOT [contest-CPU] / NOT promotion-eligible). Required "
            "with --full-cpu per Catalog #197."
        ),
    )
    # WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — procedural codebook
    # replacement variant per OP-ROUTABLE #2 of `dp1_paired_smoke_dispatch_
    # pre_authorization_checklist_20260520`. When enabled, the canonical
    # `pack_archive(...)` output is post-processed via
    # `tac.substrates.pretrained_driving_prior.distillation_procedural_variant.
    # compose_with_procedural_codebook(...)` so the codebook section bytes
    # are replaced by `brotli(seed_bytes)`. Inflate re-derives via
    # `tac.procedural_codebook_generator.derive_codebook_from_seed(...)`.
    # Catalog #324 predicted_band_validation_status=pending_post_training is
    # emitted in the readiness manifest so first paired Modal T4 smoke is the
    # canonical anchor for canonical equation #26.
    parser.add_argument(
        "--enable-procedural-codebook-replacement",
        action="store_true",
        help=(
            "Replace the Comma2k19-distilled codebook section with a "
            "deterministic procedural seed (32 bytes by default) per "
            "canonical equation #26. Inflate runtime re-derives the "
            "codebook bytes from the seed."
        ),
    )
    parser.add_argument(
        "--procedural-codebook-seed-hex",
        default="",
        help=(
            "32-byte procedural seed in hex (64 hex chars). Empty = deterministic "
            "default seed derived from --seed (so byte-stable test runs work). "
            "Operator-supplied seeds must match the canonical equation #26 "
            "domain-of-validity 8-256 bytes."
        ),
    )
    parser.add_argument(
        "--procedural-codebook-generator-kind",
        default="pcg64",
        choices=("xorshift", "lcg", "pcg64"),
        help=(
            "PRNG kind for codebook derivation (default pcg64). Sister of "
            "tac.procedural_codebook_generator.SUPPORTED_GENERATOR_KINDS."
        ),
    )
    parser.add_argument(
        "--procedural-codebook-null-exploit-control",
        action="store_true",
        help=(
            "Recipe #3 NULL-EXPLOIT-CONTROL: emit a procedural variant whose "
            "seed bytes are all-zero. The null-exploit control measures the "
            "score-axis impact of a degenerate seed; sister of the canonical "
            "PCG64-seeded variant per parent design memo §4 recipe #3."
        ),
    )
    parser.add_argument(
        "--procedural-codebook-validate-domain",
        action="store_true",
        default=True,
        help=(
            "When enabled, call "
            "tac.substrates.pretrained_driving_prior.distillation_procedural_variant."
            "verify_procedural_codebook_in_domain(...) before composing the "
            "variant. Default ON; opt-out for diagnostic runs only."
        ),
    )
    parser.add_argument(
        "--procedural-variant-provenance-path",
        default="",
        help=(
            "Optional path to write the canonical Provenance + variant manifest "
            "JSON (Catalog #323 sister). When empty, writes alongside the "
            "archive at <output_dir>/procedural_variant_provenance.json."
        ),
    )
    parser.add_argument(
        "--procedural-variant-distillation-skip",
        action="store_true",
        help=(
            "Skip the Comma2k19-derived distillation step when the variant "
            "replaces the codebook entirely. Use with --enable-procedural-"
            "codebook-replacement; the canonical Comma2k19 cache + log-"
            "incremental feeder are NOT consulted, so the variant is "
            "structurally OOD by construction (Catalog #209/#213 sister)."
        ),
    )
    return parser


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197 coupled-flag validator: --full-cpu requires waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived "
            "(Catalog #197). The non-CUDA path cannot produce promotion-"
            "eligible artifacts; the explicit attestation makes that "
            "operator-visible."
        )


def _resolve_procedural_seed_bytes(args: argparse.Namespace) -> bytes:
    """Resolve the 32-byte procedural seed per WAVE-3-DP1-DISPATCH-READY-EXTENSION.

    Precedence (parent design memo §4 paired-smoke recipe #2 + #3):

    1. ``--procedural-codebook-null-exploit-control`` (recipe #3) → 32
       zero bytes. The all-zero seed measures the score-axis impact of a
       degenerate seed; the variant module's
       :func:`verify_seed_mutation_changes_codebook_bytes` invariant still
       holds because PRG output is not constant.
    2. ``--procedural-codebook-seed-hex <hex>`` (operator-supplied seed)
       → parsed verbatim. Must be 16-512 hex chars (8-256 bytes); raises
       ``SystemExit`` on malformed input. Domain-of-validity per canonical
       equation #26 is 8-256 bytes.
    3. Default → ``hashlib.sha256(f"dpp-procedural-seed:{args.seed}".encode())``
       gives a deterministic 32-byte seed for byte-stable test runs.

    Returns: 32-byte (or operator-sized) seed bytes.

    Raises: ``SystemExit`` on malformed --procedural-codebook-seed-hex.
    """
    if args.procedural_codebook_null_exploit_control:
        return b"\x00" * 32
    if args.procedural_codebook_seed_hex:
        hex_str = args.procedural_codebook_seed_hex.strip().lower()
        if not all(c in "0123456789abcdef" for c in hex_str):
            raise SystemExit(
                f"ERROR: --procedural-codebook-seed-hex contains non-hex "
                f"chars: {args.procedural_codebook_seed_hex!r}"
            )
        if not (16 <= len(hex_str) <= 512) or (len(hex_str) % 2) != 0:
            raise SystemExit(
                f"ERROR: --procedural-codebook-seed-hex length {len(hex_str)} "
                f"outside canonical equation #26 domain-of-validity "
                f"[16, 512] hex chars (8-256 bytes); must be even-length."
            )
        return bytes.fromhex(hex_str)
    return hashlib.sha256(
        f"dpp-procedural-seed:{args.seed}".encode("utf-8")
    ).digest()


def _apply_procedural_codebook_replacement(
    *,
    args: argparse.Namespace,
    canonical_archive_bytes: bytes,
    seed_bytes: bytes,
    output_dir: Path,
) -> bytes:
    """Apply procedural codebook replacement post pack_archive.

    Per WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 OP-ROUTABLE #2 of
    ``dp1_paired_smoke_dispatch_pre_authorization_checklist_20260520``.

    Delegates to
    :func:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant.compose_with_procedural_codebook`
    which swaps the DP1 archive's ``codebook_blob`` for ``brotli(seed_bytes)``
    and rewrites the header ``codebook_len`` field. Renderer / residual /
    meta sections are preserved byte-for-byte.

    Catalog #344 cross-reference: when ``--procedural-codebook-validate-domain``
    is set (default ON), invokes
    :func:`verify_procedural_codebook_in_domain` BEFORE composition; the
    helper gracefully falls back to constant comparison when sister
    ``tac.canonical_equations.validate_context_is_in_domain`` has not yet
    landed.

    Catalog #323 canonical Provenance: writes the variant manifest to
    ``--procedural-variant-provenance-path`` (or default
    ``<output_dir>/procedural_variant_provenance.json``). The manifest
    carries ``score_claim=False`` + ``promotion_eligible=False`` +
    ``axis_tag=[predicted]`` per CLAUDE.md "Forbidden empirical-claim-
    without-evidence-tag" non-negotiable.

    Args:
        args: argparse namespace.
        canonical_archive_bytes: DP1 archive bytes emitted by ``pack_archive``.
        seed_bytes: Resolved procedural seed bytes.
        output_dir: Trainer output directory (default provenance path root).

    Returns:
        Post-replacement archive bytes. Header ``codebook_len`` is rewritten.
    """
    from tac.substrates.pretrained_driving_prior.distillation_procedural_variant import (
        CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        compose_with_procedural_codebook,
        verify_procedural_codebook_in_domain,
    )

    if args.procedural_codebook_validate_domain:
        if not verify_procedural_codebook_in_domain(
            CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        ):
            raise SystemExit(
                "ERROR: procedural codebook context "
                f"{CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT!r} fails IN-DOMAIN "
                "verification per canonical equation #26 + Catalog #344. "
                "Pass --no-procedural-codebook-validate-domain (or unset env "
                "DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN) for diagnostic runs."
            )

    canonical_len = len(canonical_archive_bytes)
    new_archive_bytes = compose_with_procedural_codebook(
        original_archive_bytes=canonical_archive_bytes,
        seed_bytes=seed_bytes,
        generator_kind=args.procedural_codebook_generator_kind,
    )
    new_len = len(new_archive_bytes)
    bytes_saved = canonical_len - new_len
    predicted_delta_s = -25.0 * bytes_saved / 37_545_489.0
    print(
        f"[full] procedural codebook replacement: "
        f"{canonical_len} B -> {new_len} B (saved {bytes_saved} B; "
        f"predicted ΔS={predicted_delta_s:+.6f})"
    )

    # Catalog #323 canonical Provenance sidecar manifest.
    provenance_path = (
        Path(args.procedural_variant_provenance_path)
        if args.procedural_variant_provenance_path
        else output_dir / "procedural_variant_provenance.json"
    )
    provenance_payload = {
        "schema": "dp1_procedural_variant_provenance_v1",
        "schema_landing": "wave_3_dp1_dispatch_ready_extension_20260520",
        "canonical_equation_id": (
            "procedural_codebook_from_seed_compression_savings_v1"
        ),
        "canonical_equation_context": CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        "seed_size_bytes": int(len(seed_bytes)),
        "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
        "generator_kind": args.procedural_codebook_generator_kind,
        "null_exploit_control": bool(
            args.procedural_codebook_null_exploit_control
        ),
        "validate_domain": bool(args.procedural_codebook_validate_domain),
        "distillation_skip": bool(args.procedural_variant_distillation_skip),
        "canonical_archive_bytes": int(canonical_len),
        "post_replacement_archive_bytes": int(new_len),
        "archive_bytes_saved": int(bytes_saved),
        "predicted_delta_s_contest_rate": float(predicted_delta_s),
        "predicted_band_validation_status": "pending_post_training",
        "evidence_grade": "[predicted]",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[predicted]",
        "lane_id": "lane_dp1_procedural_codebook_replacement_variant_20260520",
        "design_memo": (
            ".omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md"
        ),
        "checklist_memo": (
            "feedback_dp1_paired_smoke_dispatch_pre_authorization_checklist_landed_20260520.md"
        ),
        "wave_3_landing_memo": (
            "feedback_dp1_dispatch_ready_extension_landed_20260520.md"
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return new_archive_bytes


def _maybe_set_tf32(enable: bool) -> None:
    """Enable TF32 fast-math kernels on Ampere/Hopper (Catalog #178).

    The canonical helper :func:`tac.substrates._shared.trainer_skeleton.device_or_die`
    always enables TF32 on CUDA per Catalog #178; this function is kept as a
    no-op-friendly wrapper so the smoke path (which doesn't go through
    ``device_or_die``) can also benefit.
    """
    if not enable:
        return
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _torch_version_string() -> str:
    try:
        import torch

        return f"{torch.__version__}"
    except Exception:
        return "unknown"


def _build_local_cache(args: argparse.Namespace):
    """Construct the canonical Comma2k19LocalCache per Catalog #213.

    Honors ``--cache-dir`` (env ``DPP_CACHE_DIR``) override; falls back to
    :func:`default_cache_dir`. Always disk-budget-aware via
    ``--max-disk-gb``. Returns the cache instance ready to fetch chunks.
    """
    from tac.substrates.pretrained_driving_prior import (
        Comma2k19LocalCache,
        default_cache_dir,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    return Comma2k19LocalCache(
        cache_dir=cache_dir,
        max_disk_gb=float(args.max_disk_gb),
    )


def _build_frame_iterator(args: argparse.Namespace):
    """Construct the canonical Comma2k19FrameIterator per Catalog #209.

    All callers in this trainer go through this single helper so the
    contest-video-leakage guard fires once, in a path STRICT preflight
    can audit. NEVER call ``distill_codebook(frames=<raw iterator>)``
    directly from this trainer.

    Mode resolution (priority order):

    1. If ``--dataset-name=comma2k19`` AND ``--comma2k19-chunks-dir`` is set:
       use the operator-supplied chunks dir verbatim (existing behavior).
    2. If ``--dataset-name=comma2k19`` AND no explicit chunks dir but the
       trainer is wired to use the cache: the auto-download path goes
       through :func:`_log_incremental_distillation_path` instead and this
       helper is not called. Callers should branch on
       ``_use_auto_download_cache(args)`` before invoking this helper.
    3. If ``--dataset-name=bdd100k``: error out (Phase 3 scope).
    4. Default: deterministic synthetic stub.
    """
    from tac.substrates.pretrained_driving_prior import Comma2k19FrameIterator

    if args.dataset_name == "comma2k19":
        if not args.comma2k19_chunks_dir:
            raise SystemExit(
                "ERROR: --dataset-name=comma2k19 with no --comma2k19-chunks-dir "
                "AND no --cache-dir auto-download path; this code path should "
                "have been routed via _log_incremental_distillation_path() "
                "instead. Either pass --comma2k19-chunks-dir <path> (download "
                "an MIT-licensed chunk from github.com/commaai/comma2k19) OR "
                "set --cache-dir + leave --disable-log-incremental off."
            )
        # COMMA2K19_LEAKAGE_VERIFIED_OK:routed-via-Comma2k19FrameIterator-which-runs-check_no_contest_video_leakage-internally
        return Comma2k19FrameIterator(
            chunks_dir=Path(args.comma2k19_chunks_dir),
            max_chunks=args.max_distillation_chunks,
            max_frames_per_chunk=max(
                32, args.max_distillation_frames // max(1, args.max_distillation_chunks)
            ),
        )
    if args.dataset_name == "bdd100k":
        raise SystemExit(
            "ERROR: --dataset-name=bdd100k is not yet wired in Phase 2; "
            "Comma2k19FrameIterator does not yet have a BDD100K backend. "
            "Use --dataset-name=comma2k19 + --comma2k19-chunks-dir or "
            "--dataset-name=synthetic_test."
        )
    # Default: deterministic synthetic stub (tests + CI structural smoke).
    # COMMA2K19_LEAKAGE_VERIFIED_OK:synthetic-mode-does-not-touch-disk
    return Comma2k19FrameIterator(
        synthetic=True,
        n_frames=min(args.max_distillation_frames, 1024),
        seed=args.seed,
    )


def _use_auto_download_cache(args: argparse.Namespace) -> bool:
    """Return True iff the trainer should use the Comma2k19LocalCache.

    Acceptance criteria:

    1. ``--dataset-name=comma2k19`` (the cache only knows MIT-licensed chunks)
    2. No explicit ``--comma2k19-chunks-dir`` passed (otherwise the operator
       wants their own chunks tree)
    3. ``--disable-log-incremental`` is False (operator can opt out)
    4. ``--use-streamer`` is False (operator routes through cache, not stream)
    """
    return (
        args.dataset_name == "comma2k19"
        and not args.comma2k19_chunks_dir
        and not getattr(args, "disable_log_incremental", False)
        and not getattr(args, "use_streamer", False)
    )


def _use_streamer(args: argparse.Namespace) -> bool:
    """Return True iff the trainer should route through Comma2k19LocalStreamer.

    Operator pivot 2026-05-14: streaming + log mode is the NEW canonical path.

    Acceptance criteria:
    1. ``--use-streamer`` is explicitly set
    2. ``--dataset-name=comma2k19`` (the streamer only knows MIT-licensed chunks)
    3. No explicit ``--comma2k19-chunks-dir`` (operator chose streaming over disk)
    4. ``--disable-log-incremental`` is False (streaming mode IS the log-incremental
       schedule; the two are coupled)
    """
    return (
        bool(getattr(args, "use_streamer", False))
        and args.dataset_name == "comma2k19"
        and not args.comma2k19_chunks_dir
        and not getattr(args, "disable_log_incremental", False)
    )


def _validate_dataset_source_args(
    args: argparse.Namespace,
    *,
    codebook_path: Path | None,
) -> None:
    """Fail closed on unsupported or ambiguous DP1 dataset-source modes."""
    if args.dataset_name in {"comma10k", "comma10k19"}:
        raise SystemExit(
            "ERROR: comma10k is a semantic-segmentation image dataset/source "
            "for SegNet-prior work, not a DP1 dashcam-video pretraining source. "
            "Use --dataset-name=comma2k19 for DP1 video priors, or build a "
            "separate SegNet-prior adapter with its own source-custody contract."
        )
    if args.dataset_name == "bdd100k":
        raise SystemExit(
            "ERROR: --dataset-name=bdd100k is declared as future/opt-in "
            "planning surface but is not trainer-wired for DP1. Use "
            "--dataset-name=comma2k19 for real pretraining or synthetic_test "
            "for structural smoke."
        )
    if args.dataset_name not in {"synthetic_test", "comma2k19"}:
        raise SystemExit(f"ERROR: unsupported DP1 dataset_name={args.dataset_name!r}")
    if bool(getattr(args, "use_streamer", False)) and bool(args.cache_dir):
        raise SystemExit(
            "ERROR: --use-streamer and --cache-dir are mutually exclusive; "
            "choose either streaming provenance or local-cache provenance."
        )
    if args.dataset_name == "comma2k19":
        modes = [
            bool(codebook_path is not None and codebook_path.is_file()),
            bool(args.comma2k19_chunks_dir),
            bool(_use_auto_download_cache(args)),
            bool(_use_streamer(args)),
        ]
        if sum(int(flag) for flag in modes) != 1:
            raise SystemExit(
                "ERROR: real DP1 comma2k19 runs require exactly one source "
                "mode: --codebook-path, --comma2k19-chunks-dir, cache-mode "
                "(no chunks dir, log-incremental enabled), or --use-streamer."
            )


def _dp1_dataset_source_mode(
    args: argparse.Namespace,
    *,
    codebook_path: Path | None,
) -> str:
    if codebook_path is not None and codebook_path.is_file():
        return "prebuilt_codebook"
    if args.dataset_name == "synthetic_test":
        return "synthetic"
    if bool(args.comma2k19_chunks_dir):
        return "local_chunks"
    if _use_streamer(args):
        return "stream_log"
    if _use_auto_download_cache(args):
        return "local_cache"
    return "unknown"


def _build_dp1_dataset_source_manifest(
    args: argparse.Namespace,
    *,
    book_metadata: dict[str, Any],
    codebook_path: Path | None,
    schedule_log: list[Any],
) -> dict[str, Any]:
    """Build and return the canonical DP1 dataset-source manifest."""
    from tac.substrates.pretrained_driving_prior import (
        STREAMER_DEFAULT_LOG_DIR,
        build_dp1_dataset_source,
        default_cache_dir,
    )

    source_mode = _dp1_dataset_source_mode(args, codebook_path=codebook_path)
    source = build_dp1_dataset_source(
        dataset_name=str(args.dataset_name),
        source_mode=source_mode,
        distillation_mode=(
            "log_incremental"
            if schedule_log
            else ("prebuilt_codebook" if source_mode == "prebuilt_codebook" else "single_pass")
        ),
        seed=int(args.seed),
        max_distillation_frames=int(args.max_distillation_frames),
        max_distillation_chunks=int(args.max_distillation_chunks),
        codebook_metadata=book_metadata,
        chunks_dir=Path(args.comma2k19_chunks_dir) if args.comma2k19_chunks_dir else None,
        cache_dir=(
            Path(args.cache_dir)
            if args.cache_dir
            else (default_cache_dir() if source_mode == "local_cache" else None)
        ),
        stream_log_dir=(
            Path(args.stream_log_dir)
            if args.stream_log_dir
            else (STREAMER_DEFAULT_LOG_DIR if source_mode == "stream_log" else None)
        ),
        codebook_path=codebook_path if codebook_path and codebook_path.is_file() else None,
        schedule_log=schedule_log,
    )
    return source.to_dict()


def _build_local_streamer(args: argparse.Namespace):
    """Construct the canonical Comma2k19LocalStreamer per Catalog #214.

    All streamer construction goes through this single helper so the
    JSONL log dir, RAM buffer cap, and contest-video-leakage guard are
    centralized. NEVER instantiate Comma2k19LocalStreamer directly in
    trainer code outside this helper.
    """
    from tac.substrates.pretrained_driving_prior import Comma2k19LocalStreamer

    log_dir = args.stream_log_dir or None  # None → DEFAULT_LOG_DIR
    # COMMA2K19_LEAKAGE_VERIFIED_OK:routed-via-Comma2k19LocalStreamer-which-runs-_verify_chunk_id_safe-internally
    return Comma2k19LocalStreamer(
        log_dir=Path(log_dir).expanduser() if log_dir else None,
        ram_buffer_gb=float(args.ram_buffer_gb),
        dispatch_label=Path(args.output_dir).name if args.output_dir else None,
    )


def _build_dynamic_chunking_strategy(args: argparse.Namespace):
    """Build the streamer chunking strategy from explicit CLI flags."""
    from tac.substrates.pretrained_driving_prior import DynamicChunkingStrategy

    return DynamicChunkingStrategy(
        mode=str(args.stream_chunking_mode),
        frame_range_size=int(args.stream_frame_range_size)
        if args.stream_frame_range_size
        else None,
        motion_threshold=args.stream_motion_threshold,
        entropy_threshold=args.stream_entropy_threshold,
        saliency_topk=args.stream_saliency_topk,
        byte_size_target=int(args.stream_byte_size_target)
        if args.stream_byte_size_target
        else None,
        temporal_window_sec=float(args.stream_temporal_window_sec)
        if args.stream_temporal_window_sec
        else None,
    )


def _log_incremental_streaming_path(
    args: argparse.Namespace,
):
    """Distill a codebook via the streaming-mode log-incremental schedule.

    Operator pivot 2026-05-14: this is the streaming-mode companion to
    :func:`_log_incremental_distillation_path`. Same schedule + plateau
    logic; different chunk source (stream + JSONL log instead of cache).
    """
    from tac.substrates.pretrained_driving_prior import (
        DistillationConfig,
        LogIncrementalSchedule,
        log_incremental_distillation_streaming,
    )

    streamer = _build_local_streamer(args)
    chunking_strategy = _build_dynamic_chunking_strategy(args)
    chunk_specs = streamer.plan_chunks(
        chunking_strategy,
        video_metadata={
            "frames_per_chunk": int(args.streamer_frames_per_chunk),
        },
    )
    chunk_ids = [spec.chunk_id for spec in chunk_specs]
    schedule = LogIncrementalSchedule(
        base=int(args.log_incremental_base),
        initial_chunks=1,
        max_chunks=int(args.log_incremental_max_chunks),
        quality_plateau_threshold=float(args.log_incremental_quality_threshold),
    )
    distill_cfg_template = DistillationConfig(
        dataset_name="comma2k19",
        random_seed=int(args.seed),
        max_frames=int(args.max_distillation_frames),
    )
    return log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        chunk_ids=chunk_ids,
        distill_cfg_template=distill_cfg_template,
        frames_per_chunk=int(args.streamer_frames_per_chunk),
        extra_provenance={
            "dynamic_chunking_strategy": chunking_strategy.to_dict(),
            "dynamic_chunking_plan_count": len(chunk_specs),
            "dynamic_chunking_plan_preview": [
                spec.to_dict() for spec in chunk_specs[:16]
            ],
        },
    )


def _log_incremental_distillation_path(
    args: argparse.Namespace,
):
    """Distill a codebook via the canonical log-incremental schedule.

    Per Catalog #213 the auto-download cache is the SOLE entry point for
    chunk fetches; per Catalog #209 the leakage guard fires inside
    Comma2k19FrameIterator before any decode. Returns
    ``(DashcamCodebook, schedule_log)`` where ``schedule_log`` is the
    per-step ScheduleStepResult list (continual-learning anchor candidates).
    """
    from tac.substrates.pretrained_driving_prior import (
        DistillationConfig,
        LogIncrementalSchedule,
        log_incremental_distillation,
    )

    cache = _build_local_cache(args)
    schedule = LogIncrementalSchedule(
        base=int(args.log_incremental_base),
        initial_chunks=1,
        max_chunks=int(args.log_incremental_max_chunks),
        quality_plateau_threshold=float(args.log_incremental_quality_threshold),
    )
    distill_cfg_template = DistillationConfig(
        dataset_name="comma2k19",
        random_seed=int(args.seed),
        max_frames=int(args.max_distillation_frames),
    )
    return log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        distill_cfg_template=distill_cfg_template,
    )


def _archive_bytes_proxy_closed_form(num_pairs: int, per_pair_bytes: int) -> float:
    """Closed-form rate proxy for the joint Lagrangian.

    Uses fixed bands per L1 scaffold archive sizing: 5 KB codebook,
    25 KB renderer (FP16 + brotli), then num_pairs * per_pair_bytes for
    the residual, plus 1 KB meta. This is a TRAINING PROXY only; the
    auth eval reads the actual archive byte count.
    """
    return float(5 * 1024 + 25 * 1024 + 1024 + num_pairs * per_pair_bytes)


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke path: distill synthetic codebook + pack archive + parse roundtrip.

    No GPU required; no real training. Used by CI to verify the scaffold's
    structural contracts before any GPU spend. Per CLAUDE.md "Forbidden
    `make_synthetic_pair_batch` calls in any non-smoke training path", real
    training MUST use the contest video pyav decode (handled in _full_main).
    """
    from pathlib import Path

    from tac.substrates.pretrained_driving_prior import (
        DistillationConfig,
        DrivingPriorRenderer,
        DrivingPriorRendererConfig,
        build_readiness_manifest,
        distill_codebook,
        pack_archive,
        parse_archive,
        serialize_codebook,
    )

    print("[dpp-smoke] distilling synthetic codebook (deterministic; $0)")
    cfg = DistillationConfig(
        dataset_name="synthetic_test", random_seed=0xDA5C, max_frames=128
    )
    # Catalog #209 path: even the smoke uses Comma2k19FrameIterator (synthetic
    # mode) so the canonical iterator class is the SOLE entry point to the
    # distiller. The leakage guard is no-op for synthetic mode.
    smoke_args = argparse.Namespace(
        dataset_name="synthetic_test",
        comma2k19_chunks_dir="",
        max_distillation_frames=128,
        max_distillation_chunks=1,
        seed=cfg.random_seed,
    )
    frames_iter = _build_frame_iterator(smoke_args)
    book = distill_codebook(cfg, frames=iter(frames_iter))
    print(
        f"[dpp-smoke] codebook validated; provenance="
        f"{book.metadata['dataset_provenance']!r}; "
        f"license_tags={book.metadata['license_tags']}"
    )

    renderer_cfg = DrivingPriorRendererConfig(hidden_dim=32, num_hidden_layers=2)
    renderer = DrivingPriorRenderer(renderer_cfg)
    num_pairs = 4
    per_pair_bytes = 8
    residual = bytes([0] * (num_pairs * per_pair_bytes))
    # Catalog #210 provenance propagation — every DP1 archive carries
    # forensic metadata so downstream replay can audit dataset origin,
    # license attribution, codebook reproducibility, and tampering
    # detection.
    meta = {
        "residual_int8_scale": 64.0,
        "prior_inflate_strength": 1.0,
        "hidden_dim": renderer_cfg.hidden_dim,
        "num_hidden_layers": renderer_cfg.num_hidden_layers,
        # Per Catalog #210: the codebook's own metadata is the
        # authoritative source for these keys; we surface them at the
        # archive level so a tool reading just the meta blob can audit.
        "license_tags": book.metadata.get("license_tags", []),
        "dataset_provenance": book.metadata.get("dataset_provenance", ""),
        "distillation_version": book.metadata.get("distillation_version", ""),
        "random_seed": book.metadata.get("random_seed", cfg.random_seed),
        "basis_sha256": book.metadata.get("basis_sha256", ""),
        "num_frames_used": book.metadata.get("num_frames_used", 0),
    }
    packed = pack_archive(
        book,
        renderer.state_dict(),
        residual,
        meta,
        num_pairs=num_pairs,
        output_height=renderer_cfg.output_height,
        output_width=renderer_cfg.output_width,
        per_pair_bytes=per_pair_bytes,
    )
    parsed = parse_archive(packed)
    print(
        f"[dpp-smoke] archive pack/parse roundtrip: {len(packed)} bytes; "
        f"pairs={parsed.num_pairs}; header={28}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_bin = out_dir / "archive_dir" / "0.bin"
    archive_bin.parent.mkdir(parents=True, exist_ok=True)
    archive_bin.write_bytes(packed)
    archive_zip = out_dir / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packed)
    smoke_archive = out_dir / "smoke_archive.bin"
    smoke_archive.write_bytes(packed)
    codebook_bytes = serialize_codebook(book)
    codebook_path = out_dir / "codebook.bin"
    codebook_path.write_bytes(codebook_bytes)
    manifest = build_readiness_manifest(
        archive_path=str(archive_zip),
        codebook_path=str(codebook_path),
        archive_bytes=len(packed),
        codebook_bytes=len(codebook_bytes),
    )
    archive_sha256 = hashlib.sha256(packed).hexdigest()
    archive_zip_bytes = archive_zip.read_bytes()
    archive_zip_sha256 = hashlib.sha256(archive_zip_bytes).hexdigest()
    manifest["archive_sha256"] = archive_sha256
    manifest["archive_zip_bytes"] = len(archive_zip_bytes)
    manifest["archive_zip_sha256"] = archive_zip_sha256
    manifest["training_mode"] = "smoke"
    manifest["result"] = {
        "training_mode": "smoke",
        "archive_bytes": len(packed),
        "archive_sha256": archive_sha256,
        "archive_zip_bytes": len(archive_zip_bytes),
        "archive_zip_sha256": archive_zip_sha256,
        "archive_zip_path": str(archive_zip),
        "archive_bin_path": str(archive_bin),
        "codebook_path": str(codebook_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[dpp-smoke] wrote smoke archive: {smoke_archive}")
    print(f"[dpp-smoke] wrote archive.zip: {archive_zip}")
    print(f"[dpp-smoke] wrote manifest.json: {out_dir / 'manifest.json'}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training path: real distillation + score-aware joint training.

    Stages (mirrors the canonical balle_renderer trainer + Phase 2 council
    memo):

    1.  Pin seeds + select device via canonical
        :func:`tac.substrates._shared.trainer_skeleton.device_or_die`.
    2.  Patch upstream rgb_to_yuv6 globally (PR #95/#106 differentiable scorer
        contract; required BEFORE scorer construction per HNeRV parity L8).
    3.  Load differentiable SegNet + PoseNet via
        :func:`tac.scorer.load_differentiable_scorers`.
    4.  Distill the codebook (or load from ``--codebook-path``) via
        :func:`distill_codebook` consuming a
        :class:`Comma2k19FrameIterator`. Catalog #209 enforces this routing.
    5.  Decode real contest pairs via
        :func:`tac.substrates._shared.trainer_skeleton.decode_real_pairs`
        (NEVER ``make_synthetic_pair_batch`` per CLAUDE.md FORBIDDEN_PATTERNS).
    6.  Build :class:`DrivingPriorRenderer` + EMA shadow (decay 0.997 per
        CLAUDE.md "EMA — NON-NEGOTIABLE").
    7.  Stage 1 prior-only: train against ``DashcamPriorLoss`` for warmup so
        the renderer starts inside the dashcam manifold; SegNet/PoseNet
        scorers are loaded but unused.
    8.  Stage 2 frozen-prior: SegNet + PoseNet take over; the prior is FROZEN
        (codebook buffers are non-trainable by construction; no codebook
        gradient ever).
    9.  Stage 3 joint: full Lagrangian rate + seg + pose + prior with EMA
        update after every ``optimizer.step`` and snapshot+restore validation.
    10. Save EMA shadow as the inference checkpoint at every val-improvement.
    11. Build the DP1 archive from the EMA shadow + per-pair int8 residual
        derived from the validation-time pair MSE delta.
    12. Emit Catalog #146 contest-compliant ``inflate.sh`` + ``inflate.py``
        runtime alongside the bytes.
    13. Run CUDA auth eval; tag ``[contest-CUDA]`` only if Linux x86_64
        + recognized GPU substrate (Catalog #190); otherwise tag
        ``[advisory only]``.
    14. Append continual-learning anchor via
        :func:`posterior_update_locked` (Catalog #128) and cost-band anchor
        when ``DPP_ACTUAL_COST_USD`` is provided (Catalog #175).
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.smoke_auth_eval_gate import (
        gate_auth_eval_call as _canon_gate_auth_eval_call,
    )

    # F3 GTScorerCache canonical helper per Council omnibus Decision 13
    # PROCEED Option C 2026-05-14 + Time-Traveler default-OFF amendment.
    # Substrate-side score_aware_loss accepts F3 kwargs as of the same
    # commit batch; trainer-side wire-in defaults OFF (substrate authors
    # opt in via --enable-gt-scorer-cache flag).
    from tac.substrates._shared.trainer_skeleton import (
        build_optimized_training_context as _canon_build_optimized_training_context,
    )
    from tac.substrates._shared.trainer_skeleton import (
        decode_real_pairs as _canon_decode_real_pairs,
    )
    from tac.substrates._shared.trainer_skeleton import (
        detect_hardware_substrate as _canon_detect_hardware_substrate,
    )
    from tac.substrates._shared.trainer_skeleton import (
        device_or_die as _canon_device_or_die,
    )
    from tac.substrates._shared.trainer_skeleton import (
        pin_seeds as _canon_pin_seeds,
    )
    from tac.substrates.pretrained_driving_prior import (
        DashcamPriorLoss,
        DistillationConfig,
        DrivingPriorLossWeights,
        DrivingPriorRenderer,
        DrivingPriorRendererConfig,
        DrivingPriorScoreAwareLoss,
        PriorApplicationWeights,
        build_readiness_manifest,
        distill_codebook,
        pack_archive,
        serialize_codebook,
    )
    from tac.training import EMA

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    # 1. Seeds + device
    _canon_pin_seeds(args.seed)
    device = _canon_device_or_die(
        args.device,
        smoke=False,
        substrate_tag="pretrained_driving_prior",
        allow_full_cpu=bool(args.full_cpu),
    )
    _stage(f"device_resolved_{device}")

    # 2. Patch upstream YUV6 BEFORE scorer construction
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 3. Differentiable scorers
        upstream_dir = Path(args.upstream_dir)
        posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 4. Codebook (load OR distill — both routed through Catalog #209)
        codebook_path = Path(args.codebook_path) if args.codebook_path else None
        _validate_dataset_source_args(args, codebook_path=codebook_path)
        log_incremental_schedule_log: list[Any] = []
        if codebook_path is not None and codebook_path.is_file():
            from tac.substrates.pretrained_driving_prior import parse_codebook

            book = parse_codebook(codebook_path.read_bytes())
            print(f"[full] loaded codebook from {codebook_path}")
        elif _use_streamer(args):
            # OPERATOR PIVOT 2026-05-14: Catalog #214 streaming + JSONL log path.
            # Streams chunks dynamically (no permanent disk cache); decodes
            # frames; logs every access to a JSONL ledger; discards bytes.
            print(
                f"[full] log-incremental STREAMING distillation; base="
                f"{args.log_incremental_base}, max_chunks="
                f"{args.log_incremental_max_chunks}, threshold="
                f"{args.log_incremental_quality_threshold}, "
                f"stream_log_dir={args.stream_log_dir or '~/.cache/tac/comma2k19_stream_logs'}"
            )
            book, log_incremental_schedule_log = _log_incremental_streaming_path(args)
            print(
                f"[full] log-incremental streaming distillation complete "
                f"({len(log_incremental_schedule_log)} steps; "
                f"final chunk_count="
                f"{log_incremental_schedule_log[-1].chunk_count if log_incremental_schedule_log else 0}; "
                f"early_stopped="
                f"{any(s.early_stopped for s in log_incremental_schedule_log)})"
            )
        elif _use_auto_download_cache(args):
            # Catalog #213 auto-download path with log-incremental schedule.
            # Streams chunks on-demand from cache; runs codebook distillation
            # in exponentially-growing chunk batches with plateau early-stop.
            print(
                f"[full] log-incremental distillation; base="
                f"{args.log_incremental_base}, max_chunks="
                f"{args.log_incremental_max_chunks}, threshold="
                f"{args.log_incremental_quality_threshold}"
            )
            book, log_incremental_schedule_log = _log_incremental_distillation_path(
                args
            )
            print(
                f"[full] log-incremental distillation complete "
                f"({len(log_incremental_schedule_log)} steps; "
                f"final chunk_count="
                f"{log_incremental_schedule_log[-1].chunk_count if log_incremental_schedule_log else 0}; "
                f"early_stopped="
                f"{any(s.early_stopped for s in log_incremental_schedule_log)})"
            )
        else:
            print(f"[full] distilling codebook from dataset={args.dataset_name!r}")
            distill_cfg = DistillationConfig(
                dataset_name=args.dataset_name,
                random_seed=args.seed,
                max_frames=args.max_distillation_frames,
            )
            frame_iter = _build_frame_iterator(args)
            book = distill_codebook(distill_cfg, frames=iter(frame_iter))
            print(
                f"[full] distilled codebook; provenance="
                f"{book.metadata.get('dataset_provenance')!r}; "
                f"frames_used={book.metadata.get('num_frames_used')}"
            )
        dataset_source_manifest = _build_dp1_dataset_source_manifest(
            args,
            book_metadata=dict(book.metadata),
            codebook_path=codebook_path,
            schedule_log=log_incremental_schedule_log,
        )
        book.metadata["dataset_source_manifest"] = dataset_source_manifest
        if dataset_source_manifest["reproducibility_blockers"]:
            print(
                "[full] WARN dataset source reproducibility blockers: "
                + ", ".join(dataset_source_manifest["reproducibility_blockers"]),
                file=sys.stderr,
            )
        _stage("codebook_ready")

        # 5. Decode real contest pairs
        video_path = Path(args.video_path)
        print(f"[full] decoding {args.max_pairs} pairs from {video_path}")
        pair_tensor = _canon_decode_real_pairs(
            video_path,
            n_pairs=args.max_pairs,
            substrate_tag="pretrained_driving_prior",
            max_pairs=args.max_pairs,
        )
        n_pairs = int(pair_tensor.shape[0])
        pair_tensor = pair_tensor.to(device)
        _stage(f"pairs_decoded_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
        val_indices = torch.arange(
            val_idx_start, n_pairs, device=device, dtype=torch.long
        )

        # 6. Renderer + EMA
        renderer_cfg = DrivingPriorRendererConfig(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        renderer = DrivingPriorRenderer(renderer_cfg).to(device)
        # Tier-1 O3: torch.compile wrap (no-op when flag=False).
        from tac.training_optimization import compile_with_fallback as _compile_with_fallback
        renderer = _compile_with_fallback(
            renderer,
            enabled=bool(getattr(args, "enable_torch_compile", False)),
            mode="default",
            fallback_on_error=True,
        )
        ema = EMA(renderer, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian
        prior_loss_module = DashcamPriorLoss(
            book,
            PriorApplicationWeights(eval_resolution=EVAL_HW),
            device=str(device),
        ).to(device)
        weights = DrivingPriorLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            delta_prior=args.delta_prior,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = DrivingPriorScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            prior_loss=prior_loss_module,
            weights=weights,
        )
        _stage("lagrangian_built")

        # F3 GTScorerCache wire-in per F3-BACKPORT-VQVAE-PDP + Council omnibus
        # Decision 13 PROCEED Option C 2026-05-14 + Time-Traveler default-OFF
        # amendment. The canonical helper respects ``args.enable_gt_scorer_cache``
        # (default False); when False, returns ``gt_cache=None`` and the loss
        # falls back to the GT-forward path - byte-faithful to historical
        # behavior. When True, pre-computes GT PoseNet + SegNet outputs once
        # (~50%% scorer compute savings; mathematically identical).
        opt_ctx = _canon_build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=pair_tensor,
            substrate_model=renderer,
            device=device,
        )
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

        # 8. Optimizer + cosine schedule
        optimizer = torch.optim.AdamW(
            renderer.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        archive_bytes_proxy_val = _archive_bytes_proxy_closed_form(
            n_pairs, args.per_pair_bytes
        )
        archive_bytes_proxy = torch.tensor(
            archive_bytes_proxy_val, dtype=torch.float32, device=device
        )

        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = output_dir / "best.pt"
        n_train = int(train_indices.shape[0])
        nan_strike = 0

        for epoch in range(args.epochs):
            renderer.train()
            perm = train_indices[torch.randperm(n_train, device=device)]
            epoch_loss_sum = 0.0
            epoch_batches = 0
            # Tier-1 O2: autocast wrap (no-op when flag=False).
            from tac.training_optimization import autocast_aware_forward as _autocast_aware_forward
            for start in range(0, n_train, args.batch_size):
                idx = perm[start : start + args.batch_size]
                if idx.numel() == 0:
                    continue
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16,
                    device=device,
                ):
                    # Render each pair separately and stack — DrivingPriorRenderer
                    # renders one pair at a time per its current API.
                    rgb_0_list: list[torch.Tensor] = []
                    rgb_1_list: list[torch.Tensor] = []
                    for pair_idx in idx.tolist():
                        rgb_0_one, rgb_1_one = renderer.render_pair(int(pair_idx), n_pairs)
                        rgb_0_list.append(rgb_0_one)
                        rgb_1_list.append(rgb_1_one)
                    rgb_0 = torch.cat(rgb_0_list, dim=0) * 255.0  # to [0, 255] for scorer
                    rgb_1 = torch.cat(rgb_1_list, dim=0) * 255.0
                    gt = pair_tensor[idx]  # (B, 2, 3, H, W) in [0, 255]
                    gt_0 = gt[:, 0]
                    gt_1 = gt[:, 1]
                    # F3 GTScorerCache lookup (per-pair-index batched).
                    # When gt_cache is None (default-OFF amendment), the
                    # three kwargs stay None and the dispatch helper falls
                    # back to GT-forward (byte-faithful to historical).
                    gt_pose_batch = gt_seg_batch = None
                    gt_seg_already_probs = None
                    if gt_cache is not None:
                        gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                            idx, device=device
                        )
                        gt_seg_already_probs = gt_cache.seg_already_probs
                    loss, _parts = loss_fn(
                        rgb_0,
                        rgb_1,
                        gt_0,
                        gt_1,
                        archive_bytes_proxy,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=gt_pose_batch,
                        gt_seg_batch=gt_seg_batch,
                        gt_seg_already_probs=gt_seg_already_probs,
                    )
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[full] WARN non-finite loss epoch={epoch} batch={start} "
                        f"strike={nan_strike}/3",
                        file=sys.stderr,
                    )
                    if nan_strike >= 3:
                        raise RuntimeError(
                            "NaN watchdog tripped; aborting (preserving EMA)."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                nan_strike = 0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        renderer.parameters(), max_norm=args.grad_clip
                    )
                optimizer.step()
                ema.update(renderer)
                epoch_loss_sum += float(loss.detach().item())
                epoch_batches += 1
            scheduler.step()
            avg_loss = epoch_loss_sum / max(1, epoch_batches)

            # Validation + best-ckpt selection (snapshot+restore EMA)
            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                orig_state = {
                    k: v.detach().clone() for k, v in renderer.state_dict().items()
                }
                ema.apply(renderer)
                renderer.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for vidx in val_indices.tolist():
                        rgb_0_v, rgb_1_v = renderer.render_pair(int(vidx), n_pairs)
                        rgb_0_v = rgb_0_v * 255.0
                        rgb_1_v = rgb_1_v * 255.0
                        gt_v = pair_tensor[vidx : vidx + 1]
                        # F3 cache lookup for the single val pair (same
                        # primitive as train; default-OFF preserved).
                        val_pose_batch = val_seg_batch = None
                        val_seg_already_probs = None
                        if gt_cache is not None:
                            vidx_tensor = torch.tensor(
                                [int(vidx)], dtype=torch.long, device=device
                            )
                            val_pose_batch, val_seg_batch = gt_cache.lookup(
                                vidx_tensor, device=device
                            )
                            val_seg_already_probs = gt_cache.seg_already_probs
                        val_loss, _vparts = loss_fn(
                            rgb_0_v,
                            rgb_1_v,
                            gt_v[:, 0],
                            gt_v[:, 1],
                            archive_bytes_proxy,
                            apply_eval_roundtrip=True,
                            noise_std=0.0,
                            gt_pose_batch=val_pose_batch,
                            gt_seg_batch=val_seg_batch,
                            gt_seg_already_probs=val_seg_already_probs,
                        )
                        val_loss_sum += float(val_loss.detach().item())
                        val_batches += 1
                val_lag = val_loss_sum / max(1, val_batches)
                renderer.load_state_dict(orig_state)
                renderer.train()
                print(
                    f"[full] epoch {epoch + 1}/{args.epochs} train_avg={avg_loss:.6f} "
                    f"val_lag={val_lag:.6f} (best={best_val_lag:.6f} @ ep{best_epoch + 1})"
                )
                if val_lag < best_val_lag and math.isfinite(val_lag):
                    best_val_lag = val_lag
                    best_epoch = epoch
                    ema_state = ema.state_dict()
                    torch.save(
                        {
                            "state_dict": {
                                k: v.detach().cpu() for k, v in ema_state.items()
                            },
                            "config": asdict(renderer_cfg),
                            "ema_decay": args.ema_decay,
                            "best_val_lagrangian": val_lag,
                            "best_epoch": int(epoch),
                            "saved_at_utc": _utc_now_iso(),
                        },
                        ckpt_best_path,
                    )

        train_elapsed_sec = time.time() - train_started_at
        _stage(f"train_complete_elapsed_{int(train_elapsed_sec)}s")

        if not ckpt_best_path.is_file():
            print(
                "[full] WARN no improving val checkpoint observed; "
                "saving end-of-training EMA",
                file=sys.stderr,
            )
            ema_state = ema.state_dict()
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in ema_state.items()
                    },
                    "config": asdict(renderer_cfg),
                    "ema_decay": args.ema_decay,
                    "best_val_lagrangian": best_val_lag,
                    "best_epoch": int(args.epochs - 1),
                    "saved_at_utc": _utc_now_iso(),
                    "fallback_end_of_training_save": True,
                },
                ckpt_best_path,
            )

        # 11. Build DP1 archive from EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = output_dir / "archive.zip"
        payload_0bin_sha = ""
        payload_0bin_bytes = 0
        if not args.skip_archive_build:
            print(f"[full] building DP1 archive from {ckpt_best_path}")
            ckpt_obj = torch.load(
                ckpt_best_path, map_location="cpu", weights_only=False
            )
            ema_state_loaded = ckpt_obj["state_dict"]
            # Per-pair int8 residual: tiny calibration encoding the delta
            # between EMA-render and GT at each validation pair (expressed
            # as a global RGB offset; bounded by per_pair_bytes budget).
            cpu_renderer = DrivingPriorRenderer(renderer_cfg).cpu().eval()
            cpu_renderer.load_state_dict(ema_state_loaded, strict=True)
            residual_payload = bytearray()
            with torch.no_grad():
                for pair_idx in range(n_pairs):
                    rgb_0_p, rgb_1_p = cpu_renderer.render_pair(pair_idx, n_pairs)
                    rgb_0_255 = (rgb_0_p * 255.0).clamp(0.0, 255.0)
                    gt = pair_tensor[pair_idx, 0].cpu().float()
                    # Mean RGB delta per pair (3 floats), tiled across the
                    # per_pair_bytes budget. Quantize to int8 at scale=64.
                    delta_rgb = (gt - rgb_0_255[0]).mean(dim=(1, 2))
                    delta_q = (
                        (delta_rgb * 64.0 / 255.0)
                        .clamp(-127.0, 127.0)
                        .to(torch.int8)
                    )
                    pair_bytes = bytes((int(v) & 0xFF) for v in delta_q.tolist())
                    pair_bytes = (pair_bytes * (args.per_pair_bytes // 3 + 1))[
                        : args.per_pair_bytes
                    ]
                    residual_payload.extend(pair_bytes)

            # Catalog #210 provenance propagation — codebook metadata
            # surfaces to archive metadata so downstream replay tools can
            # audit dataset origin, license attribution, codebook
            # reproducibility, and tampering detection.
            meta = {
                "residual_int8_scale": 64.0,
                "prior_inflate_strength": 1.0,
                "hidden_dim": renderer_cfg.hidden_dim,
                "num_hidden_layers": renderer_cfg.num_hidden_layers,
                "best_val_lagrangian": best_val_lag if math.isfinite(best_val_lag) else None,
                "best_epoch": best_epoch,
                "lane_id": LANE_ID_PHASE_2,
                # Catalog #210 surfaces:
                "license_tags": book.metadata.get("license_tags", []),
                "dataset_provenance": book.metadata.get(
                    "dataset_provenance", ""
                ),
                "distillation_version": book.metadata.get(
                    "distillation_version", ""
                ),
                "random_seed": book.metadata.get("random_seed", 0),
                "basis_sha256": book.metadata.get("basis_sha256", ""),
                "num_frames_used": book.metadata.get("num_frames_used", 0),
                "dataset_source_manifest": dataset_source_manifest,
            }
            # WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — procedural
            # variant meta-flags. Inflate runtime reads these to decide
            # whether to re-derive the codebook via
            # tac.procedural_codebook_generator.derive_codebook_from_seed.
            # Per Catalog #324 the variant emits
            # predicted_band_validation_status=pending_post_training so the
            # first paired Modal T4 smoke is the canonical anchor for
            # canonical equation #26.
            procedural_seed_bytes: bytes | None = None
            if args.enable_procedural_codebook_replacement:
                procedural_seed_bytes = _resolve_procedural_seed_bytes(args)
                meta["procedural_codebook_variant_active"] = True
                meta["procedural_codebook_seed_hex"] = (
                    procedural_seed_bytes.hex()
                )
                meta["procedural_codebook_generator_kind"] = (
                    args.procedural_codebook_generator_kind
                )
                meta["procedural_codebook_null_exploit_control"] = bool(
                    args.procedural_codebook_null_exploit_control
                )
                meta["predicted_band_validation_status"] = (
                    "pending_post_training"
                )
                meta["canonical_equation_id"] = (
                    "procedural_codebook_from_seed_compression_savings_v1"
                )
                meta["canonical_equation_context"] = (
                    "comma2k19_ood_derived_basis_replacement"
                )
            ema_state_torch = dict(ema_state_loaded)
            bin_bytes = pack_archive(
                book,
                ema_state_torch,
                bytes(residual_payload),
                meta,
                num_pairs=n_pairs,
                output_height=renderer_cfg.output_height,
                output_width=renderer_cfg.output_width,
                per_pair_bytes=args.per_pair_bytes,
            )
            # WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — apply
            # procedural codebook replacement post pack_archive. The variant
            # module's compose_with_procedural_codebook swaps codebook_blob
            # bytes for brotli(seed) and rewrites the DP1 header
            # codebook_len field. Renderer / residual / meta sections are
            # preserved byte-for-byte.
            if procedural_seed_bytes is not None:
                bin_bytes = _apply_procedural_codebook_replacement(
                    args=args,
                    canonical_archive_bytes=bin_bytes,
                    seed_bytes=procedural_seed_bytes,
                    output_dir=output_dir,
                )
            (output_dir / "0.bin").write_bytes(bin_bytes)
            payload_0bin_bytes = len(bin_bytes)
            payload_0bin_sha = _sha256_bytes(bin_bytes)
            print(
                f"[full] wrote 0.bin ({payload_0bin_bytes} B; sha={payload_0bin_sha[:12]})"
            )

            submission_dir = output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes
            )
            archive_bytes = archive_zip_path.stat().st_size
            archive_sha = _sha256_bytes(archive_zip_path.read_bytes())
            _stage(f"archive_built_bytes_{archive_bytes}")

        # 12. CUDA auth eval (when not opted out)
        # 12. CUDA auth eval — canonical helper (Catalog #226 self-protect)
        contest_cuda_score: float | None = None
        auth_eval_path: Path | None = None
        if (
            not args.skip_auth_eval
            and archive_zip_path.is_file()
            and CONTEST_AUTH_EVAL_SCRIPT.is_file()
        ):
            print("[full] launching CUDA auth eval")
            auth_eval_path = output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=output_dir / "submission" / "inflate.sh",
                upstream_dir=upstream_dir,
                output_json=auth_eval_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="pretrained_driving_prior",
                device=device,
                full_cpu_active=bool(args.full_cpu),
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full] [contest-CUDA] = {contest_cuda_score} "
                    f"(axis={auth_result['auth_eval_score_axis']}, "
                    f"lane_tag={auth_result['auth_eval_lane_tag']}, "
                    f"archive_sha={archive_sha[:12]})"
                )
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior anchor (Catalog #128 atomic-locked)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="pretrained_driving_prior",
                    provenance_path=output_dir / "provenance.json",
                    env_var_candidates=("DPP_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=detected_substrate,
                    architecture_class=LANE_ID_PHASE_2,
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=(
                        f"DP1 phase 2 first-anchor; epochs={args.epochs}; "
                        f"dataset={args.dataset_name}"
                    ),
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(
                    f"[full] posterior_update_locked failed: {exc}",
                    file=sys.stderr,
                )

        # 14. Cost-band anchor (best-effort; never fail the run on this).
        cost_band_anchor_appended = False
        cost_band_anchor_skip_reason: str | None = None
        actual_cost_usd: float | None = None
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost_usd = parse_actual_cost_usd(
                os.environ.get("DPP_ACTUAL_COST_USD"),
                field_name="DPP_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            cost_band_anchor_skip_reason = f"invalid_DPP_ACTUAL_COST_USD:{exc}"
        if (
            COST_BAND_TOOL.is_file()
            and train_elapsed_sec > 0
            and actual_cost_usd is not None
        ):
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(COST_BAND_TOOL),
                        "--dispatch-label",
                        f"pretrained_driving_prior_{_utc_now_iso()}",
                        "--trainer",
                        "experiments/train_substrate_pretrained_driving_prior.py",
                        "--platform",
                        os.environ.get("DPP_PLATFORM", "modal"),
                        "--gpu",
                        os.environ.get("DPP_GPU", "tesla_t4"),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--actual-wall-clock-sec",
                        str(train_elapsed_sec),
                        "--actual-cost-usd",
                        str(actual_cost_usd),
                        "--notes",
                        "DP1 Phase 2 first-anchor",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    cost_band_anchor_appended = True
                else:
                    cost_band_anchor_skip_reason = (
                        f"append_failed_rc_{proc.returncode}:"
                        f"{(proc.stderr or proc.stdout)[-300:]}"
                    )
            except Exception as exc:
                cost_band_anchor_skip_reason = f"append_failed:{exc}"
        else:
            if actual_cost_usd is None and cost_band_anchor_skip_reason is None:
                cost_band_anchor_skip_reason = "missing_DPP_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        codebook_bytes = serialize_codebook(book)
        codebook_path_out = output_dir / "codebook.bin"
        codebook_path_out.write_bytes(codebook_bytes)
        manifest = build_readiness_manifest(
            archive_path=str(archive_zip_path),
            codebook_path=str(codebook_path_out),
            archive_bytes=int(archive_bytes),
            codebook_bytes=len(codebook_bytes),
        )
        manifest["archive_sha256"] = archive_sha
        manifest["payload_0bin_sha256"] = payload_0bin_sha
        manifest["payload_0bin_bytes"] = payload_0bin_bytes
        manifest["contest_cuda_score"] = contest_cuda_score
        manifest["cost_band_anchor_appended"] = cost_band_anchor_appended
        manifest["cost_band_anchor_skip_reason"] = cost_band_anchor_skip_reason
        manifest["train_elapsed_sec"] = float(train_elapsed_sec)
        manifest["lane_id"] = LANE_ID_PHASE_2
        if contest_cuda_score is not None:
            manifest["evidence_grade"] = "[contest-CUDA]"
            manifest["score_claim"] = True
            manifest["score_claim_valid"] = True
            manifest["ready_for_exact_eval_dispatch"] = False
            manifest["promotion_eligible"] = False
            manifest["dispatch_blockers"] = ["contest_cpu_eval_not_run_on_linux_x86_64"]

        provenance = {
            "schema": "dpp_phase_2_provenance_v1",
            "generated_at": _utc_now_iso(),
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
            "lane_id": LANE_ID_PHASE_2,
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "n_pairs_decoded": n_pairs,
            "n_train_pairs": int(train_indices.shape[0]),
            "n_val_pairs": int(val_indices.shape[0]),
            "best_val_lagrangian": (
                best_val_lag if math.isfinite(best_val_lag) else None
            ),
            "best_epoch": int(best_epoch),
            "train_elapsed_sec": float(train_elapsed_sec),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "payload_0bin_sha256": payload_0bin_sha,
            "payload_0bin_bytes": payload_0bin_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": str(auth_eval_path) if auth_eval_path else None,
            "cost_band_anchor_appended": cost_band_anchor_appended,
            "cost_band_anchor_skip_reason": cost_band_anchor_skip_reason,
            "stage_log": stage_log,
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "custody_status": "ci-rebuildable",
            "codebook_provenance": book.metadata,
            "dataset_source_manifest": dataset_source_manifest,
            # Catalog #213 log-incremental schedule log (empty unless the
            # auto-download cache path fired). Each step is a continual-learning
            # anchor candidate per Catalog #128 (frame_count vs codebook_quality
            # posterior).
            "log_incremental_schedule_log": [
                step.to_dict() for step in log_incremental_schedule_log
            ],
        }
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


def _write_runtime(submission_dir: Path) -> None:
    """Emit Catalog #146-compliant inflate.sh + inflate.py runtime tree.

    Vendors the canonical shared inflate runtime + the DP1 substrate
    package (codebook / archive / inflate / architecture) into the
    submission directory so :func:`inflate_one_video` can reload the
    archive bytes deterministically on a clean GPU host.
    """
    from tac.substrates._shared.trainer_skeleton import (
        vendor_shared_inflate_runtime,
    )

    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir
        / "src"
        / "tac"
        / "substrates"
        / "pretrained_driving_prior"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = (
        REPO_ROOT / "src" / "tac" / "substrates" / "pretrained_driving_prior"
    )
    for name in ("architecture.py", "archive.py", "codebook.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# pretrained_driving_prior contest-compliant inflate (DP1; Phase 2)\n"
        "# Catalog #146 contract: $1=archive_dir $2=output_dir $3=file_list\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" '
        '"$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""DP1 contest-compliant inflate runtime."""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.pretrained_driving_prior.inflate import (\n"
        "    inflate_one_video,\n"
        ")\n"
        "from tac.substrates._shared.inflate_runtime import (\n"
        "    raw_output_path,\n"
        "    select_inflate_device,\n"
        ")\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    src_path = archive_dir / '0.bin'\n"
        "    if not src_path.is_file():\n"
        "        src_path = archive_dir / 'x'\n"
        "    archive_bytes = src_path.read_bytes()\n"
        "    device = select_inflate_device()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        inflate_one_video(\n"
        "            archive_bytes,\n"
        "            raw_output_path(output_dir, line),\n"
        "            device=device,\n"
        "        )\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Catalog #19 deterministic zip: ZipInfo + writestr + fixed timestamp."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

PRETRAINED_DRIVING_PRIOR_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="pretrained_driving_prior",
    lane_id="lane_pretrained_driving_prior_phase_2_20260514",
    target_modes=("contest_one_video_replay", "production_generalized", "production_edge_adaptive", "research_substrate",),
    deployment_target="comma_ai_production",
    council_verdict_provenance=(
        ".omx/research/dpp_phase_2_training_design_20260514.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "DP1 monolithic single-file 0.bin: header (DP1 magic + version + length-prefixed) + Comma2k19-distilled codebook (fp16 + brotli) + per-frame index codebook lookups + license_tags + dataset_provenance + distillation_version + random_seed + basis_sha256 + num_frames_used (forensic provenance per Catalog #210)"
    ),
    parser_section_manifest={
        "header": "DP1_magic_and_version_length_prefixed",
        "codebook_weights": "fp16_brotli_blob",
        "frame_index_lookups": "uint16_brotli_per_frame",
        "license_tags": "json_inline",
        "dataset_provenance": "json_inline",
        "distillation_version": "json_inline",
        "random_seed": "json_inline",
        "basis_sha256": "json_inline",
        "num_frames_used": "json_inline",
    },
    inflate_runtime_loc_budget=160,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1750,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=True,
    recipe_research_only=False,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.5,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_209_no_contest_video_leakage_in_distillation_callers",
        "catalog_210_dp1_codebook_provenance_metadata_present",
        "catalog_211_dp1_composition_routes_through_canonical_helper",
        "catalog_213_comma2k19_downloads_route_through_canonical_cache",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "DP1 is the canonical pretraining lane reused across A1/PR101/HDM8/YUCR/TT5L/sane_hnerv; sensitivity captured by codebook entropy + per-frame lookup entropy"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on codebook + uint16 lookups; per-substream not per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "tools/probe_dp1_disambiguator.py (planned); codebook-size + license-tag verifier"
        ),
    },
)


@register_substrate(PRETRAINED_DRIVING_PRIOR_SUBSTRATE_CONTRACT)



def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    _validate_full_cpu_flags(args)
    if args.dataset_name in {"comma10k", "comma10k19"}:
        raise SystemExit(
            "ERROR: comma10k is not a DP1 dashcam-video pretraining source; "
            "use comma2k19 for DP1 or a separate SegNet-prior lane."
        )
    _maybe_set_tf32(args.enable_tf32)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(main())
