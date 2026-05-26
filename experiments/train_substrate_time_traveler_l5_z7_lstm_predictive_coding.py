# SPDX-License-Identifier: MIT
"""Z7 GRU recurrent predictive-coding trainer/export scaffold.

Z7 is the GRU-bound recurrent successor to Z6 Candidate 4c in the
Time-Traveler L5 predictive-coding staircase. This file is intentionally a
false-authority prebuild surface:

- ``_smoke_main`` instantiates the GRU predictor, validates the Z6-compatible
  forward signature, checks identity mode and gradient flow, then writes a
  non-authoritative smoke JSON.
- ``_full_main`` runs a byte-closed prebuild training/export smoke against real
  contest-video pairs, emits Z7PCWM1 ``0.bin`` / deterministic ``archive.zip`` /
  scorer-free runtime tree, and keeps every authority flag false until the Z7
  Wave N+1 council, beta-IB anchor, same-bytes disambiguator, and exact eval
  are ready.

No score claim, promotion claim, or paid dispatch authority is created here.
"""
# AUTOCAST_FP16_WAIVED:research-only-prebuild-scaffold-not-on-paid-dispatch-path-recipe-research_only-true-dispatch_enabled-false-canonical-autocast-backport-pending-Z6-4c-paired-exact-eval-and-Wave-N-plus-1-council

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F

from tac.substrates._shared.trainer_skeleton import decode_real_pairs as _decode_real_pairs
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding import (
    GruRecurrentPredictor,
    Z7GruPredictiveCodingConfig,
    Z7GruPredictiveCodingSubstrate,
    pack_archive,
)
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.inflate import (
    inflate_one_video,
)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
SUBSTRATE_ID = "time_traveler_l5_z7_lstm_predictive_coding"
LANE_ID = "lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z7_GRU_VIDEO_PATH",
        "rationale": (
            "Path to upstream/videos/0.mkv decoded into per-pair frames; "
            "required for non-smoke training at Wave N+1 build."
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z7_GRU_OUTPUT_DIR",
        "rationale": "Output directory for smoke stats and future trainer artifacts.",
        "default": None,
    },
    "--epochs": {
        "env": "Z7_GRU_EPOCHS",
        "rationale": "Training epoch count; smoke validates architecture only.",
        "default": "100",
    },
    "--batch-size": {
        "env": "Z7_GRU_BATCH_SIZE",
        "rationale": "Per-step pair count for the future full trainer.",
        "default": "4",
    },
    "--lr": {
        "env": "Z7_GRU_LR",
        "rationale": "AdamW base learning rate for the future full trainer.",
        "default": "5e-4",
    },
    "--gru-hidden-dim": {
        "env": "Z7_GRU_HIDDEN_DIM",
        "rationale": "GRU hidden-state width; default 128 per Z7 symposium prior.",
        "default": "128",
    },
    "--gru-num-layers": {
        "env": "Z7_GRU_NUM_LAYERS",
        "rationale": "Stacked GRUCell layer count; default 1 for first prebuild.",
        "default": "1",
    },
    "--ego-source": {
        "env": "Z7_GRU_EGO_SOURCE",
        "rationale": (
            "Future ego-source selector: posenet_projection or scorer_logit_compressed."
        ),
        "default": "posenet_projection",
    },
    "--ego-motion-dim": {
        "env": "Z7_GRU_EGO_MOTION_DIM",
        "rationale": "Ego-motion vector dimension; default 8 matches Z6 baseline.",
        "default": "8",
    },
    "--identity-predictor": {
        "env": "Z7_GRU_IDENTITY_PREDICTOR",
        "rationale": "Ablation mode: return z_prev unchanged with zero parameters.",
        "default": "false",
    },
    "--stateful": {
        "env": "Z7_GRU_STATEFUL",
        "rationale": "Whether recurrent state persists across the 600-pair sequence.",
        "default": "true",
    },
    "--beta-ib": {
        "env": "Z7_GRU_BETA_IB",
        "rationale": (
            "Beta-IB parameter; placeholder only until C6 IBPS Phase 2 empirical "
            "anchor lands."
        ),
        "default": "1.0",
    },
    "--latent-dim": {
        "env": "Z7_GRU_LATENT_DIM",
        "rationale": "Latent dimensionality for the byte-closed Z7PCWM1 packet.",
        "default": "24",
    },
    "--max-pairs": {
        "env": "Z7_GRU_MAX_PAIRS",
        "rationale": (
            "Number of real contest frame-pairs to train/export in prebuild "
            "full_main. Use 600 only for a future ratified full run."
        ),
        "default": "8",
    },
    "--decoder-embed-dim": {
        "env": "Z7_GRU_DECODER_EMBED_DIM",
        "rationale": "Decoder embedding width for the Z6-compatible RGB renderer.",
        "default": "32",
    },
    "--decoder-channels": {
        "env": "Z7_GRU_DECODER_CHANNELS",
        "rationale": "Comma-separated decoder channels, e.g. 32,24,16,12.",
        "default": "32,24,16,12",
    },
    "--decoder-num-upsample-blocks": {
        "env": "Z7_GRU_DECODER_UPSAMPLE_BLOCKS",
        "rationale": "PixelShuffle upsample block count for the RGB decoder.",
        "default": "4",
    },
    "--decoder-initial-grid-h": {
        "env": "Z7_GRU_DECODER_INITIAL_GRID_H",
        "rationale": "Decoder initial latent grid height.",
        "default": "24",
    },
    "--decoder-initial-grid-w": {
        "env": "Z7_GRU_DECODER_INITIAL_GRID_W",
        "rationale": "Decoder initial latent grid width.",
        "default": "32",
    },
    "--output-height": {
        "env": "Z7_GRU_OUTPUT_HEIGHT",
        "rationale": "Training/render height; full score-aware runs should use 384.",
        "default": "384",
    },
    "--output-width": {
        "env": "Z7_GRU_OUTPUT_WIDTH",
        "rationale": "Training/render width; full score-aware runs should use 512.",
        "default": "512",
    },
    "--inflate-verify": {
        "env": "Z7_GRU_INFLATE_VERIFY",
        "rationale": (
            "Run scorer-free inflate on the emitted packet as a local runtime check."
        ),
        "default": "false",
    },
    "--emit-static-control": {
        "env": "Z7_GRU_EMIT_STATIC_CONTROL",
        "rationale": (
            "Emit an identity/static-capacity control archive.zip with the same "
            "contest archive byte count as the recurrent packet."
        ),
        "default": "true",
    },
    "--loss-mode": {
        "env": "Z7_GRU_LOSS_MODE",
        "rationale": (
            "Training loss: proxy keeps the local MSE smoke path; score_aware "
            "loads frozen differentiable scorers at compress time only."
        ),
        "default": "proxy",
        "choices": ("proxy", "score_aware"),
    },
    "--context-conditioning-mode": {
        "env": "Z7_GRU_CONTEXT_CONDITIONING_MODE",
        "rationale": (
            "Opt-in decoder-context branch. 'none' preserves the Z7-GRU "
            "baseline; 'latent_affine' lets recurrent context modulate decoder "
            "latents and is stored byte-closed in the encoder section."
        ),
        "default": "none",
        "choices": ("none", "latent_affine"),
    },
    "--context-affine-strength": {
        "env": "Z7_GRU_CONTEXT_AFFINE_STRENGTH",
        "rationale": "Bounded affine modulation strength for latent_affine mode.",
        "default": "0.125",
    },
    "--upstream-dir": {
        "env": "Z7_GRU_UPSTREAM_DIR",
        "rationale": "Upstream scorer/runtime directory for score-aware training.",
        "default": str(REPO_ROOT / "upstream"),
    },
    "--alpha-rate": {
        "env": "Z7_GRU_ALPHA_RATE",
        "rationale": "Rate weight in the score-aware training Lagrangian.",
        "default": "25.0",
    },
    "--beta-seg": {
        "env": "Z7_GRU_BETA_SEG",
        "rationale": "SegNet weight in the score-aware training Lagrangian.",
        "default": "100.0",
    },
    "--gamma-pose": {
        "env": "Z7_GRU_GAMMA_POSE",
        "rationale": "PoseNet sqrt-distance weight in the score-aware loss.",
        "default": str(math.sqrt(10.0)),
    },
    "--noise-std": {
        "env": "Z7_GRU_NOISE_STD",
        "rationale": "Optional score-aware RGB training noise in [0, 255] units.",
        "default": "0.0",
    },
    "--smoke": {
        "env": "Z7_GRU_SMOKE",
        "rationale": "Run the cheap predictor-only smoke path instead of export main.",
        "default": "false",
    },
    "--device": {
        "env": "Z7_GRU_DEVICE",
        "rationale": "Smoke compute device: cpu, mps, or cuda when available.",
        "default": "cpu",
    },
}


def _env_default(meta: dict[str, Any]) -> Any:
    env = meta.get("env")
    return os.environ.get(env, meta.get("default")) if env else meta.get("default")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Z7 GRU trainer SCAFFOLD. Smoke validates prebuild only."
    )
    for flag, meta in TIER_1_OPERATOR_REQUIRED_FLAGS.items():
        default = _env_default(meta)
        if flag in {
            "--identity-predictor",
            "--stateful",
            "--inflate-verify",
            "--emit-static-control",
            "--smoke",
        }:
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=_boolish(default),
                help=meta.get("rationale", ""),
            )
        else:
            kwargs: dict[str, Any] = {"default": default, "help": meta.get("rationale", "")}
            if "choices" in meta:
                kwargs["choices"] = meta["choices"]
            parser.add_argument(flag, **kwargs)
    return parser


def _select_device(requested: str) -> torch.device:
    req = str(requested).strip().lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "mps" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _resolve_smoke_config(args: argparse.Namespace) -> Z7GruPredictiveCodingConfig:
    return Z7GruPredictiveCodingConfig(
        latent_dim=24,
        ego_motion_dim=int(args.ego_motion_dim),
        gru_hidden_dim=int(args.gru_hidden_dim),
        gru_num_layers=int(args.gru_num_layers),
        stateful=_boolish(args.stateful),
        identity_predictor=_boolish(args.identity_predictor),
        beta_ib=float(args.beta_ib),
        context_conditioning_mode=str(
            getattr(args, "context_conditioning_mode", "none")
        ),
        context_affine_strength=float(
            getattr(args, "context_affine_strength", "0.125")
        ),
    )


def _nonidentity_probe_config(
    cfg: Z7GruPredictiveCodingConfig,
) -> Z7GruPredictiveCodingConfig:
    """Return the same smoke config with the trainable GRU path enabled."""

    return Z7GruPredictiveCodingConfig(
        latent_dim=cfg.latent_dim,
        ego_motion_dim=cfg.ego_motion_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        gru_num_layers=cfg.gru_num_layers,
        stateful=cfg.stateful,
        identity_predictor=False,
        beta_ib=cfg.beta_ib,
        num_pairs=cfg.num_pairs,
        context_conditioning_mode=cfg.context_conditioning_mode,
        context_affine_strength=cfg.context_affine_strength,
    )


def _parse_channels(value: Any) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not channels:
        raise ValueError("--decoder-channels must include at least one integer")
    return channels


def _resolve_full_config(args: argparse.Namespace) -> Z7GruPredictiveCodingConfig:
    max_pairs = int(args.max_pairs)
    if max_pairs <= 0 or max_pairs > 600:
        raise ValueError("--max-pairs must be in [1, 600]")
    return Z7GruPredictiveCodingConfig(
        latent_dim=int(args.latent_dim),
        ego_motion_dim=int(args.ego_motion_dim),
        gru_hidden_dim=int(args.gru_hidden_dim),
        gru_num_layers=int(args.gru_num_layers),
        stateful=_boolish(args.stateful),
        identity_predictor=_boolish(args.identity_predictor),
        beta_ib=float(args.beta_ib),
        num_pairs=max_pairs,
        decoder_embed_dim=int(args.decoder_embed_dim),
        decoder_initial_grid_h=int(args.decoder_initial_grid_h),
        decoder_initial_grid_w=int(args.decoder_initial_grid_w),
        decoder_channels=_parse_channels(args.decoder_channels),
        decoder_num_upsample_blocks=int(args.decoder_num_upsample_blocks),
        output_height=int(args.output_height),
        output_width=int(args.output_width),
        context_conditioning_mode=str(
            getattr(args, "context_conditioning_mode", "none")
        ),
        context_affine_strength=float(
            getattr(args, "context_affine_strength", "0.125")
        ),
    )


def _ego_motion_from_pairs(pairs_unit: torch.Tensor, ego_motion_dim: int) -> torch.Tensor:
    """Derive deterministic low-rate pair motion features without scorer loads."""

    if pairs_unit.dim() != 5 or pairs_unit.shape[1] != 2:
        raise ValueError("pairs_unit must have shape (N, 2, 3, H, W)")
    frame_delta = pairs_unit[:, 1] - pairs_unit[:, 0]
    feats = [
        frame_delta.mean(dim=(1, 2, 3)),
        frame_delta.abs().mean(dim=(1, 2, 3)),
        frame_delta.square().mean(dim=(1, 2, 3)).sqrt(),
        pairs_unit[:, 0].mean(dim=(1, 2, 3)),
        pairs_unit[:, 1].mean(dim=(1, 2, 3)),
    ]
    channel_mean_delta = frame_delta.mean(dim=(2, 3))
    base = torch.cat([torch.stack(feats, dim=1), channel_mean_delta], dim=1)
    if base.shape[1] < ego_motion_dim:
        repeats = (ego_motion_dim + base.shape[1] - 1) // base.shape[1]
        base = base.repeat(1, repeats)
    base = base[:, :ego_motion_dim]
    mean = base.mean(dim=0, keepdim=True)
    std = base.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (base - mean) / std


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deterministic_comment(length: int) -> bytes:
    if length < 0 or length > 0xFFFF:
        raise ValueError(f"ZIP comment length {length} outside u16 range")
    chunks: list[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < length:
        chunks.append(
            hashlib.sha256(
                f"z7-static-control-zip-comment:{counter}".encode()
            ).hexdigest().encode("ascii")
        )
        counter += 1
    return b"".join(chunks)[:length]


def _archive_zip_bytes(*, bin_bytes: bytes, comment: bytes = b"") -> bytes:
    """Return deterministic archive.zip bytes containing only ``0.bin``."""

    fixed_ts = (2026, 1, 1, 0, 0, 0)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)
        zf.comment = comment
    return bio.getvalue()


def _normalize_loss_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in {"proxy", "score_aware"}:
        raise ValueError("--loss-mode must be one of: proxy, score_aware")
    return mode


def _estimate_archive_bytes_proxy(
    model: Z7GruPredictiveCodingSubstrate,
    cfg: Z7GruPredictiveCodingConfig,
) -> int:
    """Cheap byte proxy for score-aware training before final packing."""

    params = model.num_parameters_breakdown()
    weights_bytes = max(
        1_500,
        int(
            (
                int(params["predictor"])
                + int(params.get("context_conditioner", 0))
                + int(params["decoder"])
                + int(params["latent_init"])
                + int(params["residuals"])
            )
            * 2
            * 0.6
        ),
    )
    tensor_bytes = (
        cfg.num_pairs * cfg.latent_dim
        + cfg.num_pairs * cfg.ego_motion_dim
        + cfg.latent_dim
    )
    return int(weights_bytes + tensor_bytes + 1_500)


def _build_score_aware_loss(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """Load compress-time scorers and build the Z7 score-aware loss."""

    from tac.scorer import load_differentiable_scorers
    from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.score_aware_loss import (
        Z7GruPredictiveCodingScoreAwareLoss,
        Z7PredictiveCodingLossWeights,
    )

    posenet, segnet = load_differentiable_scorers(
        Path(args.upstream_dir),
        device=device,
    )
    for param in list(posenet.parameters()) + list(segnet.parameters()):
        param.requires_grad_(False)
    posenet.eval()
    segnet.eval()
    weights = Z7PredictiveCodingLossWeights(
        alpha_rate=float(args.alpha_rate),
        beta_seg=float(args.beta_seg),
        gamma_pose=float(args.gamma_pose),
        beta_ib=float(args.beta_ib),
    )
    return Z7GruPredictiveCodingScoreAwareLoss(
        seg_scorer=segnet,
        pose_scorer=posenet,
        weights=weights,
    )


def _build_archive_zip(
    archive_zip_path: Path,
    *,
    bin_bytes: bytes,
    comment: bytes = b"",
) -> None:
    """Write deterministic archive.zip containing ONLY the data payload."""

    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    archive_zip_path.write_bytes(_archive_zip_bytes(bin_bytes=bin_bytes, comment=comment))


def _context_conditioner_state_dict(
    model: Z7GruPredictiveCodingSubstrate,
) -> dict[str, torch.Tensor]:
    """Return the byte-closed context-conditioner stream for Z7PCWM1."""

    if model.context_conditioner is None:
        return {}
    return {
        key: value.detach().cpu()
        for key, value in model.context_conditioner.state_dict().items()
    }


def _write_runtime(submission_dir: Path) -> None:
    """Emit scorer-free contest inflate runtime for the Z7PCWM1 packet."""

    submission_dir.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
    ):
        pkg_init.parent.mkdir(parents=True, exist_ok=True)
        pkg_init.write_text("", encoding="utf-8")

    z7_src = REPO_ROOT / "src" / "tac" / "substrates" / (
        "time_traveler_l5_z7_lstm_predictive_coding"
    )
    z7_dst = submission_dir / "src" / "tac" / "substrates" / (
        "time_traveler_l5_z7_lstm_predictive_coding"
    )
    z7_dst.mkdir(parents=True, exist_ok=True)
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(z7_src / name, z7_dst / name)
    (z7_dst / "__init__.py").write_text(
        "\"\"\"Z7 runtime package (inflate-time only; no scorer imports).\"\"\"\n",
        encoding="utf-8",
    )

    z6_src = REPO_ROOT / "src" / "tac" / "substrates" / "time_traveler_l5_z6"
    z6_dst = submission_dir / "src" / "tac" / "substrates" / "time_traveler_l5_z6"
    z6_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(z6_src / "architecture.py", z6_dst / "architecture.py")
    (z6_dst / "__init__.py").write_text(
        "\"\"\"Z6 decoder dependency for Z7 inflate runtime.\"\"\"\n",
        encoding="utf-8",
    )

    shared_src = REPO_ROOT / "src" / "tac" / "substrates" / "_shared"
    shared_dst = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dst.mkdir(parents=True, exist_ok=True)
    (shared_dst / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(shared_src / "inflate_runtime.py", shared_dst / "inflate_runtime.py")

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Z7 GRU recurrent predictive-coding inflate runtime. No scorer imports.\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec \"${PYTHON:-python3}\" \"$HERE/inflate.py\" "
        "\"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)
    (submission_dir / "inflate.py").write_text(
        "#!/usr/bin/env python\n"
        "\"\"\"Z7 contest inflate runtime shim. No scorer imports.\"\"\"\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.inflate "
        "import main_cli\n\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main_cli())\n",
        encoding="utf-8",
    )


def _static_control_archive_pair(
    *,
    model: Z7GruPredictiveCodingSubstrate,
    cfg: Z7GruPredictiveCodingConfig,
    meta: dict[str, Any],
    target_archive_zip_bytes: int,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build same-archive-byte identity/static-capacity control bytes.

    The evaluator's archive byte term uses ``archive.zip`` size. The control
    therefore pads the standard ZIP comment, not a second member, to match the
    recurrent packet's byte budget while keeping the parsed ``0.bin`` monolithic
    and runtime-consumable.
    """

    control_cfg = replace(cfg, identity_predictor=True)
    control_meta = dict(meta)
    control_meta.update(
        {
            "paired_control_mode": "identity_predictor_static_capacity_control",
            "paired_control_archive_byte_policy": "same_archive_zip_bytes",
            "paired_control_score_claim": False,
            "paired_control_promotion_eligible": False,
        }
    )
    control_bytes = pack_archive(
        _context_conditioner_state_dict(model),
        model.decoder.state_dict(),
        {},
        model.latent_init.detach().cpu(),
        model.residuals.detach().cpu(),
        model.ego_motion_buffer.detach().cpu(),
        control_meta,
        config=control_cfg,
    )
    control_zip = _archive_zip_bytes(bin_bytes=control_bytes)
    comment_len = int(target_archive_zip_bytes) - len(control_zip)
    if comment_len < 0:
        raise RuntimeError(
            "Z7 static-control archive.zip exceeded recurrent packet byte budget: "
            f"control={len(control_zip)} recurrent={target_archive_zip_bytes}"
        )
    control_zip = _archive_zip_bytes(
        bin_bytes=control_bytes,
        comment=_deterministic_comment(comment_len),
    )
    if len(control_zip) != int(target_archive_zip_bytes):
        raise RuntimeError(
            "failed to byte-match Z7 static control archive.zip: "
            f"control={len(control_zip)} recurrent={target_archive_zip_bytes}"
        )
    control_stats = {
        "mode": "identity_predictor_static_capacity_control",
        "archive_bin_bytes": len(control_bytes),
        "archive_bin_sha256": _sha256_bytes(control_bytes),
        "archive_zip_bytes": len(control_zip),
        "archive_zip_sha256": _sha256_bytes(control_zip),
        "same_archive_zip_bytes_as_recurrent": True,
        "zip_comment_padding_bytes": comment_len,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
    }
    return control_bytes, control_zip, control_stats


def _smoke_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "results" / (
            f"z7_gru_scaffold_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_smoke_config(args)
    device = _select_device(args.device)
    predictor = GruRecurrentPredictor(cfg).to(device)

    print(f"[z7_gru_scaffold] smoke mode; output_dir={output_dir}")
    print(f"[z7_gru_scaffold] device={device} requested={args.device}")
    print(f"[z7_gru_scaffold] config={cfg}")
    print(f"[z7_gru_scaffold] predictor={predictor.to_z6_compatible_signature()}")
    print(f"[z7_gru_scaffold] num_parameters={predictor.num_parameters()}")

    z_prev = torch.randn(4, cfg.latent_dim, device=device)
    ego = torch.randn(4, cfg.ego_motion_dim, device=device)
    out = predictor(z_prev, ego)
    assert out.shape == z_prev.shape, f"shape mismatch: {out.shape} != {z_prev.shape}"

    identity_cfg = Z7GruPredictiveCodingConfig(
        latent_dim=cfg.latent_dim,
        ego_motion_dim=cfg.ego_motion_dim,
        identity_predictor=True,
    )
    identity = GruRecurrentPredictor(identity_cfg).to(device)
    identity_out = identity(z_prev, ego)
    assert torch.allclose(identity_out, z_prev)
    assert identity.num_parameters() == 0

    gradient_predictor = predictor
    if cfg.identity_predictor:
        gradient_predictor = GruRecurrentPredictor(_nonidentity_probe_config(cfg)).to(
            device
        )
    gradient_predictor.reset_state(4, device=device)
    z_grad = torch.randn(4, cfg.latent_dim, device=device, requires_grad=True)
    e_grad = torch.randn(4, cfg.ego_motion_dim, device=device, requires_grad=True)
    out_grad = gradient_predictor(z_grad, e_grad)
    out_grad.sum().backward()
    assert z_grad.grad is not None and z_grad.grad.abs().sum() > 0
    assert e_grad.grad is not None and e_grad.grad.abs().sum() > 0

    recurrence_predictor = gradient_predictor
    recurrence_predictor.eval()
    recurrence_predictor.reset_state(1, device=device)
    z_one = torch.randn(1, cfg.latent_dim, device=device)
    e_one = torch.randn(1, cfg.ego_motion_dim, device=device)
    with torch.no_grad():
        out_1 = recurrence_predictor(z_one, e_one)
        out_2 = recurrence_predictor(z_one, e_one)
    stateful_recurrence_delta = float((out_2 - out_1).abs().sum().item())

    stats = {
        "schema_version": 1,
        "name": "z7_gru_scaffold_smoke_stats",
        "substrate_id": SUBSTRATE_ID,
        "lane_id": LANE_ID,
        "num_parameters": predictor.num_parameters(),
        "config": {
            "latent_dim": cfg.latent_dim,
            "ego_motion_dim": cfg.ego_motion_dim,
            "gru_hidden_dim": cfg.gru_hidden_dim,
            "gru_num_layers": cfg.gru_num_layers,
            "stateful": cfg.stateful,
            "identity_predictor": cfg.identity_predictor,
            "beta_ib": cfg.beta_ib,
            "ego_source": args.ego_source,
            "context_conditioning_mode": cfg.context_conditioning_mode,
            "context_affine_strength": cfg.context_affine_strength,
        },
        "sanity_checks_passed": [
            "instantiation",
            "forward_pass_shape",
            "identity_predictor_returns_z_prev",
            "per_pair_master_gradient_compatible_catalog_810",
            "stateful_gru_recurrence_smoke",
        ],
        "stateful_recurrence_delta": stateful_recurrence_delta,
        "evidence_grade": "smoke_scaffold_only_NOT_promotable",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "result_review_blockers": [
            "scaffold_smoke_validates_gru_predictor_signature_only_not_training",
            "full_main_export_smoke_available_but_not_score_authority",
            "z7_trained_packet_and_auth_eval_absent_prebuild",
            "wave_n_plus_1_council_required_per_z7_symposium",
            "paired_exact_eval_json_required_for_z7_disambiguator",
            "no_contest_cuda_pair",
        ],
        "device": str(device),
        "requested_device": str(args.device),
        "utc_now": datetime.now(UTC).isoformat(),
    }
    stats_path = output_dir / "z7_gru_scaffold_smoke_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"[z7_gru_scaffold] stats written: {stats_path}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    total_started_at = time.perf_counter()
    stage_wall_seconds: dict[str, float] = {}
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "results" / (
            f"z7_gru_prebuild_export_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_full_config(args)
    device = _select_device(args.device)
    loss_mode = _normalize_loss_mode(args.loss_mode)
    torch.manual_seed(711)

    stage_started_at = time.perf_counter()
    real_pairs = _decode_real_pairs(
        Path(args.video_path),
        n_pairs=600,
        max_pairs=cfg.num_pairs,
        substrate_tag=SUBSTRATE_ID,
        repo_root=REPO_ROOT,
    )
    pairs_unit = real_pairs.to(device=device, dtype=torch.float32) / 255.0
    if tuple(pairs_unit.shape[-2:]) != (cfg.output_height, cfg.output_width):
        flat = pairs_unit.reshape(-1, 3, pairs_unit.shape[-2], pairs_unit.shape[-1])
        resized = F.interpolate(
            flat,
            size=(cfg.output_height, cfg.output_width),
            mode="bilinear",
            align_corners=False,
        )
        pairs_unit = resized.reshape(
            cfg.num_pairs,
            2,
            3,
            cfg.output_height,
            cfg.output_width,
        )
    stage_wall_seconds["decode_resize_seconds"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    model = Z7GruPredictiveCodingSubstrate(cfg).to(device)
    with torch.no_grad():
        model.ego_motion_buffer.copy_(
            _ego_motion_from_pairs(pairs_unit.detach().cpu(), cfg.ego_motion_dim).to(
                device=device,
                dtype=model.ego_motion_buffer.dtype,
            )
        )
    stage_wall_seconds["model_init_ego_seconds"] = time.perf_counter() - stage_started_at

    archive_bytes_proxy = torch.tensor(
        float(_estimate_archive_bytes_proxy(model, cfg)),
        device=device,
    )
    score_aware_loss: torch.nn.Module | None = None
    if loss_mode == "score_aware":
        stage_started_at = time.perf_counter()
        score_aware_loss = _build_score_aware_loss(args=args, device=device)
        stage_wall_seconds["score_aware_scorer_load_seconds"] = (
            time.perf_counter() - stage_started_at
        )
        score_aware_loss.train()
        print("[z7_gru_prebuild] loss_mode=score_aware; scorers loaded for training")
    else:
        stage_wall_seconds["score_aware_scorer_load_seconds"] = 0.0

    epochs = max(1, int(args.epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    losses: list[dict[str, float | int]] = []
    target0 = pairs_unit[:, 0]
    target1 = pairs_unit[:, 1]
    train_started_at = time.perf_counter()
    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rgb_0, rgb_1, latents = model.reconstruct_all_pairs()
        residual_loss = model.residuals.pow(2).mean()
        latent_smoothness = (
            (latents[1:] - latents[:-1]).pow(2).mean()
            if latents.shape[0] > 1
            else latents.new_tensor(0.0)
        )
        if score_aware_loss is not None:
            loss, parts = score_aware_loss(
                reconstructed_rgb_0=rgb_0 * 255.0,
                reconstructed_rgb_1=rgb_1 * 255.0,
                gt_rgb_0=target0 * 255.0,
                gt_rgb_1=target1 * 255.0,
                archive_bytes_proxy=archive_bytes_proxy,
                residuals=model.residuals,
                latents=latents,
                apply_eval_roundtrip=True,
                noise_std=float(args.noise_std),
            )
            loss_record: dict[str, float | int] = {
                "epoch": epoch,
                "loss": float(loss.item()),
            }
            for key, value in parts.items():
                loss_record[key] = float(value.detach().cpu())
        else:
            recon_loss = (rgb_0 - target0).pow(2).mean() + (
                rgb_1 - target1
            ).pow(2).mean()
            loss = recon_loss + float(args.beta_ib) * 1e-3 * (
                residual_loss + latent_smoothness
            )
            loss_record = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon": float(recon_loss.item()),
                "residual": float(residual_loss.item()),
                "latent_smoothness": float(latent_smoothness.item()),
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_record["wall_seconds"] = time.perf_counter() - epoch_started_at
        losses.append(loss_record)
    train_total_seconds = time.perf_counter() - train_started_at
    stage_wall_seconds["train_total_seconds"] = train_total_seconds

    stage_started_at = time.perf_counter()
    meta = {
        **model.decoder_metadata(),
        "schema": "z7_gru_prebuild_full_main_export_v1",
        "substrate_id": SUBSTRATE_ID,
        "lane_id": LANE_ID,
        "target_video": str(Path(args.video_path)),
        "target_pairs": cfg.num_pairs,
        "epochs": epochs,
        "batch_size_flag": int(args.batch_size),
        "lr": float(args.lr),
        "loss_mode": loss_mode,
        "score_aware_scorer_loss_used": loss_mode == "score_aware",
        "ego_source": str(args.ego_source),
        "ego_motion_source": "real_video_pair_delta_proxy_no_scorer_load",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
    }
    archive_bytes = pack_archive(
        _context_conditioner_state_dict(model),
        model.decoder.state_dict(),
        model.predictor.state_dict(),
        model.latent_init.detach().cpu(),
        model.residuals.detach().cpu(),
        model.ego_motion_buffer.detach().cpu(),
        meta,
        config=cfg,
    )
    bin_path = output_dir / "0.bin"
    bin_path.write_bytes(archive_bytes)
    archive_zip_path = output_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
    archive_zip_bytes = archive_zip_path.read_bytes()

    static_control: dict[str, Any] | None = None
    static_control_bytes: bytes | None = None
    if _boolish(args.emit_static_control):
        static_control_bytes, static_control_zip_bytes, static_control = (
            _static_control_archive_pair(
                model=model,
                cfg=cfg,
                meta=meta,
                target_archive_zip_bytes=len(archive_zip_bytes),
            )
        )
        control_dir = output_dir / "static_capacity_control"
        control_dir.mkdir(parents=True, exist_ok=True)
        control_bin_path = control_dir / "0.bin"
        control_zip_path = control_dir / "archive.zip"
        control_bin_path.write_bytes(static_control_bytes)
        control_zip_path.write_bytes(static_control_zip_bytes)
        static_control.update(
            {
                "archive_bin_path": str(control_bin_path),
                "archive_zip_path": str(control_zip_path),
            }
        )

    submission_dir = output_dir / "submission_runtime"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
    stage_wall_seconds["export_packaging_seconds"] = (
        time.perf_counter() - stage_started_at
    )

    inflate_verify: dict[str, Any] | None = None
    stage_started_at = time.perf_counter()
    if _boolish(args.inflate_verify):
        verify_raw = output_dir / "inflate_verify" / "0.raw"
        frames = inflate_one_video(archive_bytes, verify_raw, device=str(device))
        inflate_verify = {
            "raw_path": str(verify_raw),
            "frames_written": frames,
            "raw_bytes": verify_raw.stat().st_size,
            "raw_sha256": _sha256_bytes(verify_raw.read_bytes()),
        }
        if static_control_bytes is not None and static_control is not None:
            control_raw = output_dir / "inflate_verify" / "static_control.raw"
            control_frames = inflate_one_video(
                static_control_bytes,
                control_raw,
                device=str(device),
            )
            control_raw_bytes = control_raw.read_bytes()
            control_sha = _sha256_bytes(control_raw_bytes)
            recurrent_raw_bytes = verify_raw.read_bytes()
            byte_differences = sum(
                a != b for a, b in zip(recurrent_raw_bytes, control_raw_bytes)
            ) + abs(len(recurrent_raw_bytes) - len(control_raw_bytes))
            static_control.update(
                {
                    "inflate_verify_raw_path": str(control_raw),
                    "inflate_verify_frames_written": control_frames,
                    "inflate_verify_raw_bytes": control_raw.stat().st_size,
                    "inflate_verify_raw_sha256": control_sha,
                    "runtime_output_changed_vs_recurrent": (
                        control_sha != inflate_verify["raw_sha256"]
                    ),
                    "runtime_output_byte_differences_vs_recurrent": byte_differences,
                }
            )
    stage_wall_seconds["inflate_verify_seconds"] = time.perf_counter() - stage_started_at

    final_loss = losses[-1] if losses else {"loss": None}
    total_wall_seconds = time.perf_counter() - total_started_at
    stage_wall_seconds["total_wall_seconds"] = total_wall_seconds
    timing_smoke = {
        "schema": "z7_timing_smoke_v1",
        "axis": "[local-trainer-timing advisory]",
        "device": str(device),
        "loss_mode": loss_mode,
        "epochs": epochs,
        "num_pairs": cfg.num_pairs,
        "stage_wall_seconds": {
            key: float(value) for key, value in stage_wall_seconds.items()
        },
        "seconds_per_epoch": float(train_total_seconds / max(1, epochs)),
        "seconds_per_pair_epoch": float(
            train_total_seconds / max(1, epochs * cfg.num_pairs)
        ),
        "pairs_per_second_epoch": float(
            (epochs * cfg.num_pairs) / max(train_total_seconds, 1e-12)
        ),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_paid_dispatch": False,
    }
    stats = {
        "schema_version": 1,
        "name": "z7_gru_prebuild_full_main_export_stats",
        "substrate_id": SUBSTRATE_ID,
        "lane_id": LANE_ID,
        "config": {
            "latent_dim": cfg.latent_dim,
            "ego_motion_dim": cfg.ego_motion_dim,
            "gru_hidden_dim": cfg.gru_hidden_dim,
            "gru_num_layers": cfg.gru_num_layers,
            "stateful": cfg.stateful,
            "identity_predictor": cfg.identity_predictor,
            "beta_ib": cfg.beta_ib,
            "num_pairs": cfg.num_pairs,
            **model.decoder_metadata(),
        },
        "losses": losses,
        "final_loss": final_loss.get("loss"),
        "final_loss_proxy": final_loss.get("loss") if loss_mode == "proxy" else None,
        "final_loss_score_aware": (
            final_loss.get("loss") if loss_mode == "score_aware" else None
        ),
        "archive_bin_path": str(bin_path),
        "archive_bin_bytes": len(archive_bytes),
        "archive_bin_sha256": _sha256_bytes(archive_bytes),
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_bytes": archive_zip_path.stat().st_size,
        "archive_zip_sha256": _sha256_bytes(archive_zip_bytes),
        "static_capacity_control": static_control,
        "submission_runtime_dir": str(submission_dir),
        "inflate_verify": inflate_verify,
        "timing_smoke": timing_smoke,
        "loss_mode": loss_mode,
        "score_aware_scorer_loss_used": loss_mode == "score_aware",
        "archive_bytes_proxy_for_training": int(archive_bytes_proxy.detach().cpu()),
        "num_parameters": model.num_parameters(),
        "num_parameters_breakdown": model.num_parameters_breakdown(),
        "evidence_grade": (
            "score_aware_training_export_smoke_NOT_promotable"
            if loss_mode == "score_aware"
            else "prebuild_training_export_smoke_NOT_promotable"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "result_review_blockers": [
            (
                "score_aware_trained_packet_not_auth_eval_validated"
                if loss_mode == "score_aware"
                else "not_score_aware_scorer_loss"
            ),
            "no_paired_exact_eval_json",
            "wave_n_plus_1_council_required_per_z7_symposium",
            "c6_beta_ib_anchor_required",
            (
                "same_archive_bytes_disambiguator_exact_eval_required"
                if static_control
                else "same_archive_bytes_disambiguator_control_missing"
            ),
            "no_contest_cuda_pair",
        ],
        "utc_now": datetime.now(UTC).isoformat(),
    }
    stats_path = output_dir / "z7_gru_prebuild_full_main_export_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"[z7_gru_prebuild] stats written: {stats_path}")
    print(f"[z7_gru_prebuild] archive_zip={archive_zip_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    # Explicit attribute reassignments so preflight dead-resolver gate sees
    # the canonical TIER_1_OPERATOR_REQUIRED_FLAGS-derived argparse names at
    # static analysis time (per comprehensive bug audit cascade 2026-05-26).
    args.context_conditioning_mode = getattr(args, "context_conditioning_mode", "none")
    args.context_affine_strength = getattr(args, "context_affine_strength", 0.125)
    if _boolish(args.smoke):
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
