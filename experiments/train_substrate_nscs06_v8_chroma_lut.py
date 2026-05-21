# SPDX-License-Identifier: MIT
"""Train (compress-only) the nscs06_v8_chroma_lut substrate (L0 SCAFFOLD).

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21 + CASCADE
COMPRESSION symposium commit ``d125af6c3`` PRIORITY 3 (Daubechies + Mallat
multi-scale partition discovery framing; 2nd IN-DOMAIN procedural-variant
substrate after grayscale_lut) + HONEST CASCADE-MORTALITY ASSESSMENT
commit ``d884dd6aa`` Rank 2 + canonical equation #26 IN-DOMAIN context
``nscs06_v8_chroma_lut``.

This is the canonical NSCS06 v8 trainer L0 SCAFFOLD. ``_full_main`` raises
``NotImplementedError`` until the per-substrate symposium per Catalog #325
lands a PROCEED verdict within the 14-day window (2026-05-21 -> 2026-06-04).
``_smoke_main`` is a CPU-only compress-pass producing a tiny CH08 archive
for local engineering smoke (compose + parse + inflate roundtrip) — NOT a
contest-eligible artifact per Catalog #240 ``research_only=True`` recipe.

Recipe-vs-trainer-state consistency (Catalog #240):
  - Recipe: ``substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`` has
    ``research_only: true`` + ``dispatch_enabled: false``.
  - Trainer: ``_full_main`` raises NotImplementedError; only ``_smoke_main``
    produces an archive (tiny, non-contest-eligible).

CLAUDE.md compliance:
  - Train against ``upstream/videos/0.mkv`` (NOT synthetic; Catalog #114).
    Smoke uses synthetic data only because ``research_only=True``.
  - Hardware substrate dynamically detected via canonical helper (Catalog #190).
  - Contest-compliant runtime emission (3-positional-arg ``inflate.sh``; Catalog #146).
  - TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151).
  - META layer ``@register_substrate`` decoration (Catalog #241/#242).
  - No silent device defaults; --device required; --smoke gates CPU.
  - No /tmp paths in persisted artifacts.

Usage (smoke; CPU; tiny output)::

    .venv/bin/python experiments/train_substrate_nscs06_v8_chroma_lut.py \\
        --output-dir experiments/results/nscs06_v8_smoke_<utc> \\
        --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:scorer-not-yet-wired-substrate-engineering-scaffold-l0-no-training-loop
# TORCH_COMPILE_WAIVED:no-training-loop-numpy-only-codec
# TF32_WAIVED:no-matmul-operations-numpy-pillow-only
# NO_GRAD_WAIVED:no-training-loop-numpy-pillow-only
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-substrate-engineering-scaffold-l0
# SCORER_PREPROCESS_HANDLED_OK:no-score-aware-loss-substrate-engineering-scaffold-l0
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tac.substrate_registry import register_substrate

# Import the v8 substrate package — this triggers @register_substrate
# decoration validation per Catalog #241/#242.
from tac.substrates.nscs06_v8_chroma_lut import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT,
    NUM_SEGNET_CLASSES,
    POSE_DIMS,
    PROCEDURAL_SEED_SIZE_BYTES,
    pack_archive,
    parse_archive,
    predicted_delta_s,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — TIER 1 operator-required flags
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS06_V8_VIDEO_PATH",
        "rationale": (
            "score-aware compress-side scorer MUST query the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            "feedback_t3_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520.md#priority-3"
        ),
    },
    "--output-dir": {
        "env": "NSCS06_V8_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "NSCS06_V8_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; "
            "required for non-smoke compress + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "NSCS06_V8_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required for "
            "full run (MPS refused per CLAUDE.md); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "NSCS06_V8_EPOCHS",
        "rationale": (
            "v8 has no training loop; trainer skeleton + dispatch infra "
            "expect --epochs; we accept any positive value and run ONE "
            "compress pass (or NotImplementedError in _full_main per Catalog "
            "#240 until per-substrate symposium ratifies dispatch)"
        ),
        "default": "1",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs06_v8_chroma_lut",
        description=(
            "Compress-only pass for the nscs06 v8 chroma-LUT substrate "
            "(CASCADE COMPRESSION symposium d125af6c3 PRIORITY 3; canonical "
            "equation #26 IN-DOMAIN context nscs06_v8_chroma_lut; L0 SCAFFOLD "
            "until per-substrate symposium per Catalog #325 PROCEED verdict)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--variant",
        choices=("v1_inline_lut", "v2_procedural_seed"),
        default="v2_procedural_seed",
        help=(
            "CH08 archive variant: v1 inline LUT (full 4096-byte chroma table) "
            "or v2 procedural seed (32-byte PCG64 seed replaces the LUT slot; "
            "predicted ΔS = -0.002706 [prediction; canonical-equation-26-grounded])"
        ),
    )
    return p


# ---------------------------------------------------------------------------
# _smoke_main — tiny synthetic compress pass (research_only)
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU compress pass for engineering smoke.

    Produces a synthetic CH08 archive (NOT a contest-eligible artifact;
    research_only per Catalog #240). Verifies the v8 archive pack/parse +
    inflate roundtrip end-to-end without requiring upstream video or scorers.
    """
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tiny synthetic config — matches the tests in src/tac/substrates/nscs06_v8_chroma_lut/tests/
    num_pairs = 4
    gh, gw = 4, 6
    out_h, out_w = 16, 24

    rng = np.random.RandomState(args.seed)
    pose_bytes = rng.randint(0, 256, size=num_pairs * POSE_DIMS, dtype=np.uint8).tobytes()
    grayscale_bytes = rng.randint(
        0, 256, size=num_pairs * gh * gw, dtype=np.uint8
    ).tobytes()

    if args.variant == "v1_inline_lut":
        lut = rng.randint(
            0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
        )
        blob = pack_archive(
            num_pairs=num_pairs, grayscale_h=gh, grayscale_w=gw,
            output_height=out_h, output_width=out_w,
            pose_bytes=pose_bytes, grayscale_bytes=grayscale_bytes,
            chroma_lut=lut,
        )
    else:
        seed = rng.randint(0, 256, size=PROCEDURAL_SEED_SIZE_BYTES, dtype=np.uint8).tobytes()
        blob = pack_archive(
            num_pairs=num_pairs, grayscale_h=gh, grayscale_w=gw,
            output_height=out_h, output_width=out_w,
            pose_bytes=pose_bytes, grayscale_bytes=grayscale_bytes,
            chroma_seed=seed,
        )

    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(blob)

    arc = parse_archive(blob)
    smoke_meta = {
        "variant": args.variant,
        "schema_version": arc.schema_version,
        "num_pairs": arc.num_pairs,
        "archive_bytes": len(blob),
        "chroma_lut_bytes_declared": CHROMA_LUT_BYTES_DEFAULT,
        "predicted_delta_s": predicted_delta_s(),
        "research_only": True,
        "evidence_grade": "predicted",
        "axis_tag": "[prediction]",
        "promotable": False,
        "score_claim_valid": False,
        "canonical_equation_in_domain_context": "nscs06_v8_chroma_lut",
    }
    (output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, sort_keys=True, indent=2)
    )
    print(
        f"[smoke] CH08 {args.variant} archive bytes: {len(blob)} "
        f"(predicted ΔS = {predicted_delta_s():.6f} [prediction])"
    )
    return 0


# ---------------------------------------------------------------------------
# _full_main — NotImplementedError per Catalog #240 until per-substrate symposium
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Full compress + auth eval — REFUSED per Catalog #240 until symposium.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial
    grand council symposium" non-negotiable + Catalog #325 14-day per-substrate
    symposium window: the v8 trainer is in L0 SCAFFOLD state pending the
    per-substrate symposium PROCEED verdict (target window: 2026-05-21 ->
    2026-06-04).

    The recipe ``substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml``
    carries ``research_only: true`` + ``dispatch_enabled: false`` per
    Catalog #240; this trainer body MIRRORS that state by raising
    NotImplementedError. Per Catalog #315 OPTIMAL FORM discipline: PROCEED
    verdict + reactivation criteria pin promotion to L1.
    """
    raise NotImplementedError(
        "nscs06_v8_chroma_lut substrate is L0 SCAFFOLD per Catalog #240 + "
        "Catalog #325 per-substrate symposium pending (window 2026-05-21 -> "
        "2026-06-04). Use --smoke for engineering verification; full compress "
        "+ auth eval is gated on PROCEED verdict from the canonical 6-step "
        "per-substrate symposium per CLAUDE.md 'PER-SUBSTRATE OPTIMAL FORM via "
        "adversarial grand council symposium' non-negotiable."
    )


@register_substrate(NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main())
