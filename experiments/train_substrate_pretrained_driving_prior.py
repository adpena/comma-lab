# NO_GRAD_WAIVED:scaffold-only-trainer-Phase-2-not-yet-implemented-no-eval-scorer-forwards-on-this-path
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

This file is a SCAFFOLD — it wires the canonical helpers, declares the
Catalog #151 ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` manifest, and the
smoke path exercises codebook distillation + pack + parse. Real training
fires Phase 2 when operator + cost gates are green.

Catalog #146 inflate.sh contract: the trainer's ``_write_runtime`` emits
the contest 3-positional-arg ``inflate.sh <archive_dir> <output_dir> <file_list>``.
Catalog #151: ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` declares every required
flag so operator wrappers thread env-vars correctly.
Catalog #152: ``required_input_file=True`` flags trigger pre-dispatch
filesystem validation in the operator wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from typing import Any

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
}


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
        choices=("synthetic_test", "comma2k19", "bdd100k"),
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
    return parser


def _maybe_set_tf32(enable: bool) -> None:
    """Enable TF32 fast-math kernels on Ampere/Hopper (Catalog #178)."""
    if not enable:
        return
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


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
    book = distill_codebook(cfg)
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
    meta = {
        "residual_int8_scale": 64.0,
        "hidden_dim": renderer_cfg.hidden_dim,
        "num_hidden_layers": renderer_cfg.num_hidden_layers,
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
    manifest["archive_sha256"] = archive_sha256
    manifest["training_mode"] = "smoke"
    manifest["result"] = {
        "training_mode": "smoke",
        "archive_bytes": len(packed),
        "archive_sha256": archive_sha256,
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
    """Full training path. SCAFFOLD: declares the contract but does not yet
    fire the score-aware loop against real contest pairs.

    Production landing fills in:

    * Load codebook (distill if missing; else load from --codebook-path)
    * Build DrivingPriorRenderer with EMA per CLAUDE.md non-negotiable
    * Decode real contest pairs via ``decode_real_pairs`` (canonical helper)
    * Load differentiable scorers (per L1 HNeRV parity)
    * Train with DrivingPriorScoreAwareLoss
    * Build DP1 archive at best EMA checkpoint
    * Write Catalog #146 contest-compliant inflate.sh + inflate.py
    * Run contest-CUDA + contest-CPU auth eval (per CLAUDE.md submission rule)
    * Append posterior anchor via tac.cost_band_calibration.append_anchor

    For the L0 scaffold landing, full path raises NotImplementedError so the
    operator-authorize wrapper cannot accidentally fire a no-op full dispatch.
    """
    raise NotImplementedError(
        "DPP full training is L0 SCAFFOLD. Use --smoke for the scaffold "
        "structural test. Full training lands when the Phase 2 design "
        "memo (.omx/research/dpp_phase_2_training_design_<DATE>.md) is "
        "council-approved and the codebook distillation has been run "
        "against real Comma2k19 frames (operator-gated)."
    )


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    _maybe_set_tf32(args.enable_tf32)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(main())
