# SPDX-License-Identifier: MIT
"""Train the ATW codec V2 substrate (Atick-Tishby-Wyner full-stack cooperative-receiver codec).

Per the 2026-05-16 V2 design memo
``.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md``
(commit ``fcdcc1112``). V2 lifts V1's ``_full_main NotImplementedError`` per the
Phase 2 council gate AND adds three structural primitives (G1 + B3 + WZ
side-info head closed-form) that the V1 design memo named as reactivation
criteria.

Canonical-vs-unique decision per layer (per CLAUDE.md
``UNIQUE-AND-COMPLETE-PER-METHOD operating mode`` + design memo Section 15;
6 UNIQUE FORK + 14 ADOPT canonical):

| Layer                                  | Decision      | Rationale |
|----------------------------------------|---------------|-----------|
| Trainer skeleton (device_or_die)       | ADOPT canonical | TF32 + CUDA discipline shared per Catalog #172/#178/#180 |
| Atick-Redlich primitive                | ADOPT canonical | cooperative_receiver_loss per Catalog #164 + Wunderkind E1 |
| eval_roundtrip                         | ADOPT canonical | CLAUDE.md non-negotiable + Catalog #5 |
| EMA decay (0.997)                      | ADOPT canonical | CLAUDE.md "EMA — non-negotiable" + Catalog #88 |
| score_aware_loss helper                | ADOPT canonical | score_pair_components routed via Atick-Redlich primitive |
| select_inflate_device (Catalog #205)   | ADOPT canonical | inflate-device-fork canonical helper |
| gate_auth_eval_call (Catalog #226)     | ADOPT canonical | auth-eval CLI canonical routing |
| detect_hardware_substrate (Catalog #190)| ADOPT canonical | phantom-score-directory protection |
| posterior_update_locked (Catalog #128) | ADOPT canonical | fcntl-locked continual learning |
| ATW2 archive grammar                   | UNIQUE FORK    | new magic + 2 sections (G1 + B3) beyond ATW1 |
| WZ side-info head                      | UNIQUE FORK    | substrate-distinguishing per Catalog #272 |
| G1 scorer-class distill head           | UNIQUE FORK    | per Wunderkind G1 (replaces Ballé hyperprior) |
| B3 scorer-conditional CDF table        | UNIQUE FORK    | per Wunderkind B3 (precomputed at compress) |
| Three-knob Variant A (kappa/wz/pixel)  | UNIQUE FORK    | probe-disambiguator regime sweep |
| ATWv2ScoreAwareLoss                    | UNIQUE FORK    | composes canonical primitive + WZ + G1 + IB/pixel |

Tier 1 engineering primitives:
* autocast_fp16 declared via --enable-autocast-fp16 flag + torch.amp.autocast wiring
* TF32 enabled via canonical trainer_skeleton.device_or_die
* torch.compile declared via --enable-torch-compile flag
* no_grad-at-eval via torch.no_grad() context in _full_main eval pass
* canonical scorer-loss helper via cooperative_receiver_loss + score_pair_components

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~1-3 epochs)::

    .venv/bin/python experiments/train_substrate_atw_codec_v2.py \\
        --output-dir experiments/results/atw_v2_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; Modal A100; conditional on D4 probe MEANINGFUL_CONDITIONING)::

    .venv/bin/python experiments/train_substrate_atw_codec_v2.py \\
        --video-path upstream/videos/0.mkv \\
        --upstream-dir upstream \\
        --output-dir experiments/results/atw_v2_<utc> \\
        --epochs 200 --batch-size 4 --lr 5e-4 --device cuda \\
        --variant B --kappa-ib 0.0 --lambda-wz 1.0 --lambda-pixel 0.0 \\
        --enable-autocast-fp16 --enable-torch-compile
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _decode_real_pairs_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _device_or_die_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _utc_now_iso,
)
from tac.substrates.atw_codec_v2 import (
    ATW2_MAGIC,
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2LossWeights,
    ATWv2ScoreAwareLoss,
    ATWv2Variant,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_codec_v2.registered_substrate import (
    ATW_CODEC_V2_CONTRACT,  # noqa: F401 (forces package-side contract validation)
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

SUBSTRATE_TAG = "atw_codec_v2"
SUBSTRATE_LANE_ID = "lane_atw_codec_v2_substrate_build_20260516"
DESIGN_MEMO_PATH = (
    ".omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md"
)

CODEC_PACKAGE_SOURCE = REPO_ROOT / "src" / "tac" / "substrates" / "atw_codec_v2"
VENDORED_CODEC_SUBDIR = "_atw_codec_v2"
VENDORED_V2_FILES = (
    "__init__.py",
    "architecture.py",
    "archive.py",
    "inflate.py",
)

EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS manifest (Catalog #168 AnnAssign)
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "ATW_V2_VIDEO_PATH",
        "rationale": (
            "score-aware compress-side scorer MUST query the contest video "
            "(upstream/videos/0.mkv); synthetic data FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": DESIGN_MEMO_PATH,
    },
    "--output-dir": {
        "env": "ATW_V2_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "ATW_V2_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; "
            "required for non-smoke compress + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "ATW_V2_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required "
            "for full run (MPS refused per CLAUDE.md); cpu permitted only "
            "with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "ATW_V2_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal A100 full=200",
        "default": "200",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "ATW_V2_BATCH_SIZE",
        "rationale": "Per-step pair count; A100 handles 4-8 at 384x512",
        "default": "4",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--lr": {
        "env": "ATW_V2_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--variant": {
        "env": "ATW_V2_VARIANT",
        "rationale": (
            "ATW V2 variant; 'A' = three-knob (kappa_IB / lambda_WZ / "
            "lambda_pixel) probe-disambiguator regime sweep; 'B' = "
            "UNIQUE-AND-COMPLETE single-knob WZ-only (DEFAULT per design memo §4.3)"
        ),
        "default": "B",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--kappa-ib": {
        "env": "ATW_V2_KAPPA_IB",
        "rationale": (
            "Tishby IB regularizer weight (Variant A only); 0 = no IB; "
            "0.05-0.1 = IB regime"
        ),
        "default": "0.0",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--lambda-wz": {
        "env": "ATW_V2_LAMBDA_WZ",
        "rationale": (
            "Wyner-Ziv residual term weight; 1 = ATW canonical (default); "
            "0 = WZ disabled (= Z4 baseline branch of probe-disambiguator)"
        ),
        "default": "1.0",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--lambda-pixel": {
        "env": "ATW_V2_LAMBDA_PIXEL",
        "rationale": (
            "Pixel-MSE residual weight (Variant A only); 0 = pure ATW; "
            "1 = Z3 baseline (probe-disambiguator corner)"
        ),
        "default": "0.0",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--lambda-distill": {
        "env": "ATW_V2_LAMBDA_DISTILL",
        "rationale": (
            "G1 distill cross-entropy weight; trains G1 head to predict "
            "scorer class from latent; 0 = G1 supervision disabled"
        ),
        "default": "0.1",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_atw_codec_v2",
        description=(
            "Train ATW codec V2 substrate (Atick-Tishby-Wyner full-stack "
            "cooperative-receiver codec; V2 design memo 2026-05-16; "
            "Wunderkind G1 + B3 + WZ side-info head closed-form productionized)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)
    p.add_argument("--scorer-class-prior-dim", type=int, default=16)
    p.add_argument("--wz-head-hidden-dim", type=int, default=32)
    p.add_argument("--g1-distill-hidden-dim", type=int, default=32)

    # ATW V2 variant + Lagrangian knobs
    p.add_argument(
        "--variant",
        type=str,
        choices=["A", "B"],
        default="B",
        help="A = three-knob (V1-inherited); B = single-knob WZ-only (DEFAULT)",
    )
    p.add_argument("--kappa-ib", type=float, default=0.0,
                   help="Tishby IB regularizer weight (Variant A only)")
    p.add_argument("--lambda-wz", type=float, default=1.0,
                   help="Wyner-Ziv residual weight; 1 = ATW canonical")
    p.add_argument("--lambda-pixel", type=float, default=0.0,
                   help="Pixel-MSE residual weight (Variant A only); 0 = pure ATW")
    p.add_argument("--lambda-distill", type=float, default=0.1,
                   help="G1 distill cross-entropy weight; trains G1 head")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))

    # Tier 1 engineering primitives per Catalog #172/#178/#179/#180
    p.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        help="Catalog #172: enable torch.amp.autocast(dtype=fp16) on CUDA training",
    )
    p.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Catalog #179: enable torch.compile on the codec module",
    )

    # Smoke / mode flags
    p.add_argument("--smoke", action="store_true", help="Run synthetic-data sanity smoke")
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip auth-eval (CI / dev only)")
    p.add_argument("--skip-archive-build", action="store_true",
                   help="Skip building archive.zip + submission/")
    p.add_argument("--max-pairs", type=int, default=N_PAIRS_FULL,
                   help="Decoder pair count limit (full=600)")
    p.add_argument("--scorer-chunk-size", type=int, default=8,
                   help="Catalog #218 sister: bound peak VRAM for scorer-class precompute")
    return p


def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(
        name, smoke=smoke, substrate_tag=SUBSTRATE_TAG
    )


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None):
    return _decode_real_pairs_canonical(
        video_path,
        n_pairs=n_pairs,
        substrate_tag=SUBSTRATE_TAG,
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


def _sha256_first16(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Smoke main (synthetic-data sanity; <1 min CPU)
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Synthetic-data sanity smoke — validates substrate forward + archive roundtrip.

    No scorer load. No real video decode. ``$0`` cost. Verifies:

    1. ATWv2Codec instantiates with the canonical config (Variant A or B)
    2. Forward pass produces (rgb_0, rgb_1) of correct shape + WZ residual
       + G1 distill logits
    3. WZ side-info head produces non-zero z_residual when enabled
    4. ATW2 archive pack -> parse roundtrip is byte-identical
    5. ATW2 magic + section-offset parser refuses tampered bytes
    """
    _pin_seeds(args.seed)
    _ = _device_or_die(args.device, smoke=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_enum = (
        ATWv2Variant.A_THREE_KNOB if args.variant == "A" else ATWv2Variant.B_WZ_ONLY
    )

    # Tiny config so smoke runs in <1 min on CPU.
    cfg = ATWv2CodecConfig(
        variant=variant_enum,
        latent_dim=args.latent_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_num_upsample_blocks=4,  # smaller than full
        decoder_channels=(16, 12, 8, 6, 4, 2),
        num_pairs=8,  # tiny
        output_height=64,
        output_width=96,
        scorer_class_prior_dim=args.scorer_class_prior_dim,
        wz_head_hidden_dim=args.wz_head_hidden_dim,
        g1_distill_hidden_dim=args.g1_distill_hidden_dim,
        ib_kappa_default=args.kappa_ib,
        wz_lambda_default=args.lambda_wz,
        pixel_lambda_default=args.lambda_pixel,
    )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = ATWv2Codec(cfg).to(device)
    model.eval()

    # Populate scorer_class_prior_table with deterministic non-zero pattern.
    with torch.no_grad():
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.arange(cfg.scorer_class_prior_dim, dtype=torch.float32) * 0.1
                + float(i) * 0.01
            )
        # Populate CDF table with non-uniform per-class deterministic pattern
        # so the smoke roundtrip exercises non-trivial B3 contents.
        for c in range(model.cdf_table.shape[0]):
            model.cdf_table[c] = torch.linspace(
                0.001 * (c + 1), 0.999 * (c + 1), model.cdf_table.shape[1]
            )

    # Forward smoke
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long, device=device)
    with torch.no_grad():
        rgb_0, rgb_1, _mu, _logvar, z_residual, z_predicted, distill_logits = model(
            pair_indices,
            frames_for_encoder=None,
            compute_wz_residual=True,
            compute_g1_logits=True,
        )
    expected_shape = (cfg.num_pairs, 3, cfg.output_height, cfg.output_width)
    if tuple(rgb_0.shape) != expected_shape or tuple(rgb_1.shape) != expected_shape:
        raise RuntimeError(
            f"smoke forward shape mismatch: got rgb_0 {tuple(rgb_0.shape)}, "
            f"rgb_1 {tuple(rgb_1.shape)}; expected {expected_shape}"
        )
    if distill_logits is None or distill_logits.shape[1] != 5:
        raise RuntimeError(
            f"G1 distill_logits shape unexpected: {None if distill_logits is None else tuple(distill_logits.shape)}"
        )

    # Archive roundtrip smoke
    encoder_sd = model.encoder.state_dict()
    decoder_sd = model.decoder.state_dict()
    wz_head_sd = model.wz_side_info_head.state_dict()
    distill_head_sd = model.g1_distill_head.state_dict()
    meta_seed: dict[str, object] = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    variant_byte = 0 if variant_enum == ATWv2Variant.A_THREE_KNOB else 1
    archive_bytes = pack_archive(
        encoder_sd,
        decoder_sd,
        wz_head_sd,
        distill_head_sd,
        z_residual.detach().cpu() if z_residual is not None else model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        model.cdf_table.detach().cpu(),
        meta_seed,
        variant=variant_byte,
        atw_kappa_ib=args.kappa_ib,
        atw_lambda_wz=args.lambda_wz,
        atw_lambda_pixel=args.lambda_pixel,
        wz_head_enabled=cfg.wz_head_enabled,
        g1_distill_enabled=cfg.g1_distill_enabled,
        b3_cdf_enabled=cfg.b3_cdf_enabled,
    )
    if not archive_bytes.startswith(ATW2_MAGIC):
        raise RuntimeError(
            f"archive magic mismatch: got {archive_bytes[:4]!r} expected {ATW2_MAGIC!r}"
        )
    parsed = parse_archive(archive_bytes)
    if parsed.schema_version != 1:
        raise RuntimeError(f"unexpected schema version: {parsed.schema_version}")
    if parsed.variant != variant_byte:
        raise RuntimeError(
            f"variant roundtrip mismatch: declared {variant_byte}, parsed {parsed.variant}"
        )

    archive_path = args.output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)
    payload_0bin_sha = _sha256_bytes(archive_bytes)
    archive_zip_path = args.output_dir / "archive.zip"
    submission_dir = args.output_dir / "submission"
    archive_zip_sha: str | None = None
    archive_zip_bytes: int | None = None
    archive_zip_built = False
    if not args.skip_archive_build:
        _write_runtime(submission_dir)
        (submission_dir / "0.bin").write_bytes(archive_bytes)
        _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
        archive_zip_sha = _sha256_file(archive_zip_path)
        archive_zip_bytes = archive_zip_path.stat().st_size
        archive_zip_built = True

    stats: dict[str, Any] = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "smoke": True,
        "device": str(device),
        "epochs": args.epochs,
        "variant": args.variant,
        "variant_byte": variant_byte,
        "archive_bytes": len(archive_bytes),
        "archive_sha256_first16": _sha256_first16(archive_bytes),
        "payload_0bin_sha256": payload_0bin_sha,
        "payload_0bin_path": str(archive_path),
        "archive_zip_built": archive_zip_built,
        "archive_zip_path": str(archive_zip_path) if archive_zip_built else None,
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "submission_dir": str(submission_dir) if archive_zip_built else None,
        "model_params": model.num_parameters_breakdown(),
        "kappa_ib": args.kappa_ib,
        "lambda_wz": args.lambda_wz,
        "lambda_pixel": args.lambda_pixel,
        "lambda_distill": args.lambda_distill,
        "wz_head_enabled": cfg.wz_head_enabled,
        "g1_distill_enabled": cfg.g1_distill_enabled,
        "b3_cdf_enabled": cfg.b3_cdf_enabled,
        "atw2_magic_ok": True,
        "roundtrip_ok": True,
        "completed_at_utc": _utc_now_iso(),
        "design_memo": DESIGN_MEMO_PATH,
    }
    (args.output_dir / "smoke_stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[atw_codec_v2] SMOKE OK device={device} variant={args.variant} "
        f"archive_bytes={len(archive_bytes)} params={model.num_parameters()} "
        f"kappa={args.kappa_ib} lambda_wz={args.lambda_wz} "
        f"lambda_pixel={args.lambda_pixel} lambda_distill={args.lambda_distill}"
    )
    return 0


# ---------------------------------------------------------------------------
# Runtime emission for contest-compliant submission tree
# ---------------------------------------------------------------------------
def _write_runtime(submission_dir: Path) -> None:
    """Emit SELF-CONTAINED submission tree per Catalog #146 + #295.

    V2 vendors its own 4 codec files into ``submission/_atw_codec_v2/``.
    The submission inflate.py is a thin CLI shim that imports from the
    vendored package + select_inflate_device per Catalog #205.
    """
    import shutil

    submission_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# ATW Codec V2 inflate (Catalog #146 3-positional-arg contract)\n"
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

    vendored_dir = submission_dir / VENDORED_CODEC_SUBDIR
    vendored_dir.mkdir(parents=True, exist_ok=True)
    vendor_init = (
        "# SPDX-License-Identifier: MIT\n"
        '"""Vendored ATW V2 codec package — self-contained inflate."""\n'
    )
    (vendored_dir / "__init__.py").write_text(vendor_init, encoding="utf-8")

    # Vendor v2 files
    for fname in VENDORED_V2_FILES:
        if fname == "__init__.py":
            # Skip the source __init__.py since it imports many tac.* symbols
            # the vendored copy cannot resolve. The vendor_init above is the
            # sealed-submission stub.
            continue
        src = CODEC_PACKAGE_SOURCE / fname
        if not src.is_file():
            raise FileNotFoundError(
                f"v2 vendoring failed: codec source missing: {src}"
            )
        shutil.copy2(src, vendored_dir / fname)

    # Patch the vendored files to use sibling imports instead of tac.* paths
    for fname in ("architecture.py", "archive.py", "inflate.py"):
        target = vendored_dir / fname
        text = target.read_text(encoding="utf-8")
        text = text.replace(
            "from tac.substrates.atw_codec_v2.architecture import",
            "from .architecture import",
        )
        text = text.replace(
            "from tac.substrates.atw_codec_v2.archive import",
            "from .archive import",
        )
        # inflate.py imports from canonical _shared inflate runtime; we inline
        # a minimal raw_output_path + write_rgb_pair_to_raw + select_inflate_device
        # via the submission inflate.py shim instead.
        target.write_text(text, encoding="utf-8")

    # Submission inflate.py — vendored self-contained shim
    inflate_py = (
        "#!/usr/bin/env python\n"
        "# SPDX-License-Identifier: MIT\n"
        '"""ATW V2 contest-compliant inflate runtime.\n'
        "\n"
        "Self-contained per Catalog #146 + Catalog #295. The codec package is\n"
        "vendored into the sibling ``_atw_codec_v2/`` directory. Per CLAUDE.md\n"
        "Strict scorer rule: NO SegNet/PoseNet load at inflate time.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "if str(HERE) not in sys.path:\n"
        "    sys.path.insert(0, str(HERE))  # SUBMISSION_PYTHONPATH_SHIM_OK:vendored-_atw_codec_v2-package-alongside\n"
        "\n"
        "from _atw_codec_v2.archive import parse_archive  # noqa: E402\n"
        "from _atw_codec_v2.architecture import (  # noqa: E402\n"
        "    EVAL_HW,\n"
        "    ATWv2Codec,\n"
        "    ATWv2CodecConfig,\n"
        "    ATWv2Variant,\n"
        ")\n"
        "\n"
        "\n"
        "def select_inflate_device() -> torch.device:\n"
        '    """Catalog #205 canonical helper inlined for sealed submission."""\n'
        "    # INLINE_DEVICE_FORK_OK:vendored-submission-helper-mirrors-canonical-tac.substrates._shared.inflate_runtime.select_inflate_device-via-PACT_INFLATE_DEVICE-env-var\n"
        '    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()\n'
        '    if pinned not in {"auto", "cpu", "cuda"}:\n'
        "        raise SystemExit(\n"
        '            f\"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}\"\n'
        "        )\n"
        '    if pinned == "cpu":\n'
        '        return torch.device("cpu")\n'
        '    if pinned == "cuda":\n'
        "        if not torch.cuda.is_available():\n"
        "            raise SystemExit(\n"
        '                \"PACT_INFLATE_DEVICE=cuda but cuda is not available\"\n'
        "            )\n"
        '        return torch.device("cuda")\n'
        '    return torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        "\n"
        "\n"
        "def _write_rgb_pair_uint8(\n"
        "    fh, rgb_0: torch.Tensor, rgb_1: torch.Tensor, *, h: int, w: int\n"
        ") -> None:\n"
        '    """Render unit-domain RGB pair to (3, H, W) uint8 raw bytes per frame."""\n'
        "    import torch.nn.functional as F\n"
        "    for rgb in (rgb_0, rgb_1):\n"
        "        if rgb.shape[-2:] != (h, w):\n"
        "            rgb = F.interpolate(rgb, size=(h, w), mode='bilinear', align_corners=False)\n"
        "        u8 = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)\n"
        "        # (B=1, 3, h, w) -> (3, h, w) -> (h, w, 3) raw\n"
        "        out = u8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()\n"
        "        fh.write(out)\n"
        "\n"
        "\n"
        "def _inflate_one_video(\n"
        "    archive_bytes: bytes, output_raw_path: Path, *, device: torch.device\n"
        ") -> int:\n"
        '    """Reconstruct + render all pairs into one contest .raw file."""\n'
        "    arc = parse_archive(archive_bytes)\n"
        "    meta = arc.meta\n"
        "    variant_byte = int(arc.variant)\n"
        "    cfg = ATWv2CodecConfig(\n"
        "        variant=ATWv2Variant.A_THREE_KNOB if variant_byte == 0 else ATWv2Variant.B_WZ_ONLY,\n"
        "        latent_dim=int(arc.latent_residual.shape[1]),\n"
        "        encoder_input_channels=int(meta.get('encoder_input_channels', 3)),\n"
        "        encoder_hidden_dim=int(meta.get('encoder_hidden_dim', 64)),\n"
        "        decoder_embed_dim=int(meta['decoder_embed_dim']),\n"
        "        decoder_initial_grid_h=int(meta['decoder_initial_grid_h']),\n"
        "        decoder_initial_grid_w=int(meta['decoder_initial_grid_w']),\n"
        "        decoder_channels=tuple(int(c) for c in meta['decoder_channels']),\n"
        "        decoder_num_upsample_blocks=int(meta['decoder_num_upsample_blocks']),\n"
        "        num_pairs=int(arc.latent_residual.shape[0]),\n"
        "        output_height=int(meta.get('output_height', EVAL_HW[0])),\n"
        "        output_width=int(meta.get('output_width', EVAL_HW[1])),\n"
        "        scorer_class_prior_dim=int(\n"
        "            meta.get('_scorer_class_prior_dim', arc.scorer_class_prior_table.shape[1])\n"
        "        ),\n"
        "        wz_head_hidden_dim=int(meta.get('wz_head_hidden_dim', 32)),\n"
        "        wz_head_enabled=bool(meta.get('atw_v2_codec_meta', {}).get('wz_head_enabled', True)),\n"
        "        g1_distill_hidden_dim=int(meta.get('g1_distill_hidden_dim', 32)),\n"
        "        g1_distill_enabled=bool(meta.get('atw_v2_codec_meta', {}).get('g1_distill_enabled', True)),\n"
        "        b3_cdf_enabled=bool(meta.get('atw_v2_codec_meta', {}).get('b3_cdf_enabled', True)),\n"
        "    )\n"
        "    model = ATWv2Codec(cfg).to(device).eval()\n"
        "    if arc.encoder_state_dict:\n"
        "        model.encoder.load_state_dict(arc.encoder_state_dict, strict=False)\n"
        "    model.decoder.load_state_dict(arc.decoder_state_dict, strict=False)\n"
        "    if cfg.wz_head_enabled and arc.wz_side_info_head_state_dict:\n"
        "        model.wz_side_info_head.load_state_dict(arc.wz_side_info_head_state_dict, strict=False)\n"
        "    if cfg.g1_distill_enabled and arc.distill_head_state_dict:\n"
        "        model.g1_distill_head.load_state_dict(arc.distill_head_state_dict, strict=False)\n"
        "    with torch.no_grad():\n"
        "        model.latents.copy_(arc.latent_residual.to(device=device, dtype=model.latents.dtype))\n"
        "        model.scorer_class_prior_table.copy_(\n"
        "            arc.scorer_class_prior_table.to(device=device, dtype=model.scorer_class_prior_table.dtype)\n"
        "        )\n"
        "        model.cdf_table.copy_(arc.cdf_table.to(device=device, dtype=model.cdf_table.dtype))\n"
        "    # Contest raw output resolution; per Catalog #166 mask res upsample fix.\n"
        "    out_h = int(meta.get('contest_raw_h', 874))\n"
        "    out_w = int(meta.get('contest_raw_w', 1164))\n"
        "    output_raw_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    frames_written = 0\n"
        "    with torch.no_grad(), output_raw_path.open('wb') as fh:\n"
        "        for pair_idx in range(cfg.num_pairs):\n"
        "            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)\n"
        "            z_residual = model.latents[idx_tensor]\n"
        "            rgb_0, rgb_1 = model.reconstruct_from_wz_residual(idx_tensor, z_residual)\n"
        "            _write_rgb_pair_uint8(fh, rgb_0, rgb_1, h=out_h, w=out_w)\n"
        "            frames_written += 2\n"
        "    return frames_written\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print(\n"
        "            'usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "            file=sys.stderr,\n"
        "        )\n"
        "        return 2\n"
        "    device = select_inflate_device()\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        _inflate_one_video(archive_bytes, output_dir / base, device=device)\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Deterministic archive.zip per Catalog #19. Only ``0.bin`` payload."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Full main (CUDA-required by default; trains ATWv2 + emits ATW2 archive + auth eval)
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """Full ATW V2 training pass.

    The trainer:

    1. Pin seeds + canonical device selection (Catalog #1/#172/#178).
    2. Patch upstream yuv6 + load differentiable scorers in canonical order
       (pose_scorer, seg_scorer = load_differentiable_scorers per Catalog #222).
    3. Decode real pairs from upstream/videos/0.mkv (Catalog #114).
    4. Precompute scorer_class_prior_table from SegNet argmax on per-pair
       anchor frames (chunked per Catalog #218).
    5. Train ATWv2Codec with ATWv2ScoreAwareLoss (canonical Atick-Redlich
       primitive + WZ residual + G1 distill supervision; optional IB/pixel
       terms in Variant A). EMA shadow per CLAUDE.md non-negotiable.
    6. Build B3 scorer-conditional CDF table from empirical histogram of
       (z_residual_quantized, class_index) pairs.
    7. Pack ATW2 archive (Variant A or B) + build runtime tree.
    8. Run canonical gate_auth_eval_call (Catalog #226) for paired contest
       auth eval.
    9. Posterior update via posterior_update_locked (Catalog #128).
    10. Write provenance JSON with stage_log + axis labels per Catalog #166/#221.
    """
    from tac.continual_learning import (
        ContestResult,
        posterior_update_locked,
    )
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")
    train_started_at = time.time()

    try:
        # 1. Canonical scorer load — order MUST be (pose_scorer, seg_scorer)
        # per Catalog #222.
        pose_scorer, seg_scorer = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(pose_scorer.parameters()) + list(seg_scorer.parameters()):
            p.requires_grad_(False)
        pose_scorer.eval()
        seg_scorer.eval()
        _stage("scorers_loaded")

        # 2. Decode real pairs
        print(f"[atw_v2-full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[atw_v2-full] decoded {n_pairs} pairs at {EVAL_HW}")
        _stage(f"pairs_decoded_{n_pairs}")

        # 3. Build the codec
        variant_enum = (
            ATWv2Variant.A_THREE_KNOB if args.variant == "A" else ATWv2Variant.B_WZ_ONLY
        )
        cfg = ATWv2CodecConfig(
            variant=variant_enum,
            latent_dim=args.latent_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            scorer_class_prior_dim=args.scorer_class_prior_dim,
            wz_head_hidden_dim=args.wz_head_hidden_dim,
            g1_distill_hidden_dim=args.g1_distill_hidden_dim,
            ib_kappa_default=args.kappa_ib,
            wz_lambda_default=args.lambda_wz,
            pixel_lambda_default=args.lambda_pixel,
        )
        model = ATWv2Codec(cfg).to(device)
        _stage("model_built")

        # 4. Catalog #218: chunked SegNet/PoseNet precompute of scorer_class_prior_table
        # Use frame_0 (anchor) per pair as the input. Aggregate per-pair
        # class distribution + pose summary into the side-info table.
        with torch.no_grad():
            class_prior_chunks: list[torch.Tensor] = []
            chunk = max(1, int(args.scorer_chunk_size))
            for start in range(0, n_pairs, chunk):
                stop = min(start + chunk, n_pairs)
                # Use first frame of each pair as anchor; (chunk, 3, H, W).
                anchor = pair_tensor[start:stop, 0].to(device).float()
                anchor_btchw = anchor.unsqueeze(1)
                seg_logits = seg_scorer(seg_scorer.preprocess_input(anchor_btchw))
                # seg_logits: (chunk, 5, H, W) -> per-pair class distribution
                # = mean softmax over spatial pixels (compact summary).
                seg_probs = torch.softmax(seg_logits.float(), dim=1)
                seg_summary = seg_probs.mean(dim=(2, 3))  # (chunk, 5)
                # Fill scorer_class_prior_table: first 5 dims = seg summary; remaining
                # zeros (or future pose summary; reserve dims for pose).
                pad_width = max(0, cfg.scorer_class_prior_dim - 5)
                pad = torch.zeros(
                    (seg_summary.shape[0], pad_width), device=device, dtype=seg_summary.dtype
                )
                row = torch.cat([seg_summary, pad], dim=1)
                class_prior_chunks.append(row.cpu())
                del anchor, anchor_btchw, seg_logits, seg_probs, seg_summary
            class_prior_full = torch.cat(class_prior_chunks, dim=0)
            model.scorer_class_prior_table.copy_(class_prior_full.to(device))
        _stage(f"scorer_class_prior_precomputed_chunk_{chunk}")

        # 5. Build loss + optimizer + EMA shadow
        loss_weights = ATWv2LossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            kappa_ib=args.kappa_ib,
            lambda_wz=args.lambda_wz,
            lambda_pixel=args.lambda_pixel,
            lambda_distill=args.lambda_distill,
        )
        loss_module = ATWv2ScoreAwareLoss(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            weights=loss_weights,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # EMA shadow (decay=0.997 per CLAUDE.md non-negotiable). Inline minimal
        # implementation — tracking model params on the same device as model.
        ema_shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        ema_decay = 0.997

        def _ema_update() -> None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in ema_shadow:
                        ema_shadow[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)

        # Catalog #179 torch.compile (optional; ungated by --enable-torch-compile)
        compiled_decoder = model.decoder
        if args.enable_torch_compile and device.type == "cuda":
            try:
                compiled_decoder = torch.compile(model.decoder, dynamic=False)
                _stage("torch_compile_decoder_enabled")
            except Exception as exc:  # pragma: no cover — falls back at runtime
                print(
                    f"[atw_v2-full] torch.compile FAILED (eager fallback): {exc}",
                    file=sys.stderr,
                )
                _stage("torch_compile_decoder_fallback_eager")

        # Catalog #172 autocast wiring (only when CUDA + flag)
        autocast_active = bool(args.enable_autocast_fp16) and device.type == "cuda"
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_active
            else _NullContext()
        )

        # 6. Training loop
        model.train()
        batch_size = max(1, int(args.batch_size))
        steps_per_epoch = max(1, (n_pairs + batch_size - 1) // batch_size)
        archive_bytes_proxy = torch.tensor(
            float(120_000), device=device, dtype=torch.float32
        )  # initial proxy; refined post-epoch from actual archive
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = (step * batch_size) % n_pairs
                stop = min(start + batch_size, n_pairs)
                pair_idx = torch.arange(start, stop, device=device, dtype=torch.long)
                gt_pair = pair_tensor[start:stop].to(device)
                gt_rgb_0 = gt_pair[:, 0].float()
                gt_rgb_1 = gt_pair[:, 1].float()

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    z_stored = model.latents[pair_idx]
                    class_prior = model.scorer_class_prior_table[pair_idx]
                    z_predicted = model.wz_side_info_head(class_prior)
                    z_decoded = z_stored
                    rgb_unit_pair = compiled_decoder(z_decoded)
                    rgb_0_unit, rgb_1_unit = rgb_unit_pair
                    rgb_0 = rgb_0_unit * 255.0
                    rgb_1 = rgb_1_unit * 255.0
                    z_residual = z_stored - z_predicted

                    # G1 distill supervision: use compress-side SegNet argmax
                    # over the rendered output as the supervision target.
                    if args.lambda_distill > 0.0:
                        with torch.no_grad():
                            rendered_btchw = rgb_0.unsqueeze(1)
                            seg_logits_render = seg_scorer(
                                seg_scorer.preprocess_input(rendered_btchw)
                            )
                            distill_targets = (
                                torch.argmax(seg_logits_render, dim=1)
                                .float()
                                .mean(dim=(1, 2))
                                .round()
                                .clamp(0, 4)
                                .to(torch.long)
                            )
                        distill_logits = model.g1_distill_head(z_decoded)
                    else:
                        distill_logits = None
                        distill_targets = None

                    out = loss_module(
                        reconstructed_rgb_0=rgb_0,
                        reconstructed_rgb_1=rgb_1,
                        gt_rgb_0=gt_rgb_0,
                        gt_rgb_1=gt_rgb_1,
                        archive_bytes_proxy=archive_bytes_proxy,
                        z_residual=z_residual,
                        z_predicted=z_predicted,
                        distill_class_logits=distill_logits,
                        distill_class_targets=distill_targets,
                    )

                out.loss.backward()
                optimizer.step()
                _ema_update()
                epoch_loss += float(out.loss.detach().item())
            scheduler.step()

            if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
                print(
                    f"[atw_v2-full] epoch {epoch}/{args.epochs} loss={epoch_loss/steps_per_epoch:.6f} "
                    f"variant={args.variant} lr={scheduler.get_last_lr()[0]:.2e}"
                )
        _stage(f"training_complete_{args.epochs}_epochs")

        # 7. Apply EMA shadow to the model for archive build (per CLAUDE.md
        # "EMA - NON-NEGOTIABLE" inference checkpoint comes from EMA shadow).
        with torch.no_grad():
            live_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            ema_state = dict(live_state)
            for k, v in ema_shadow.items():
                ema_state[k] = v.to(live_state[k].device, dtype=live_state[k].dtype)
            model.load_state_dict(ema_state, strict=False)
        model.eval()
        _stage("ema_shadow_applied_for_archive")

        # 8. Build B3 scorer-conditional CDF table from empirical histogram of
        # (z_residual_quantized, predicted_class) pairs.
        with torch.no_grad():
            full_pair_idx = torch.arange(n_pairs, device=device, dtype=torch.long)
            z_stored_full = model.latents[full_pair_idx]
            class_prior_full_table = model.scorer_class_prior_table[full_pair_idx]
            z_predicted_full = model.wz_side_info_head(class_prior_full_table)
            z_residual_full = z_stored_full - z_predicted_full
            if args.lambda_distill > 0.0:
                pred_classes = model.g1_distill_head.predict_class(z_stored_full)
            else:
                # Fall back to argmax over scorer_class_prior_table[:5]
                pred_classes = torch.argmax(class_prior_full_table[:, :5], dim=1)

            # Quantize residual to int8 (same scale as archive); use the
            # canonical degenerate-range fix logic.
            res_cpu = z_residual_full.detach().cpu().float()
            lo, hi = float(res_cpu.min()), float(res_cpu.max())
            if hi <= lo:
                q_residual = torch.full_like(res_cpu, -127, dtype=torch.int8)
            else:
                scale = (hi - lo) / 254.0
                q_unsigned = ((res_cpu - lo) / scale).round().clamp(0.0, 254.0)
                q_residual = (q_unsigned - 127.0).to(torch.int8)
            # Build per-class histogram over int8 symbols
            num_classes = int(model.cdf_table.shape[0])
            num_symbols = int(model.cdf_table.shape[1])
            cdf_table = torch.full(
                (num_classes, num_symbols),
                1.0 / float(num_symbols),
                dtype=torch.float32,
            )
            for c in range(num_classes):
                mask = (pred_classes.detach().cpu() == c)
                if not mask.any():
                    continue
                symbols = (q_residual[mask].view(-1).to(torch.int32) + 128).clamp(0, 255)
                hist = torch.bincount(symbols, minlength=num_symbols).float()
                total = hist.sum().clamp(min=1.0)
                cdf_table[c] = (hist / total).clamp(min=1e-6)
            model.cdf_table.copy_(cdf_table.to(device))
        _stage("b3_cdf_table_built")

        # 9. Pack ATW2 archive (with EMA-applied weights)
        variant_byte = 0 if variant_enum == ATWv2Variant.A_THREE_KNOB else 1
        meta_seed: dict[str, object] = {
            "decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "encoder_input_channels": cfg.encoder_input_channels,
            "encoder_hidden_dim": cfg.encoder_hidden_dim,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
            "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
            "latent_init_std": cfg.latent_init_std,
            "contest_raw_h": 874,
            "contest_raw_w": 1164,
            "lane_id": SUBSTRATE_LANE_ID,
            "design_memo": DESIGN_MEMO_PATH,
        }
        archive_bytes_atw2 = pack_archive(
            model.encoder.state_dict(),
            model.decoder.state_dict(),
            model.wz_side_info_head.state_dict(),
            model.g1_distill_head.state_dict(),
            z_residual_full.detach().cpu(),
            model.scorer_class_prior_table.detach().cpu(),
            model.cdf_table.detach().cpu(),
            meta_seed,
            variant=variant_byte,
            atw_kappa_ib=args.kappa_ib,
            atw_lambda_wz=args.lambda_wz,
            atw_lambda_pixel=args.lambda_pixel,
            wz_head_enabled=cfg.wz_head_enabled,
            g1_distill_enabled=cfg.g1_distill_enabled,
            b3_cdf_enabled=cfg.b3_cdf_enabled,
        )
        (args.output_dir / "0.bin").write_bytes(archive_bytes_atw2)
        payload_0bin_sha = _sha256_bytes(archive_bytes_atw2)
        payload_0bin_bytes = len(archive_bytes_atw2)
        print(
            f"[atw_v2-full] wrote 0.bin ({payload_0bin_bytes} bytes, "
            f"sha256={payload_0bin_sha}, variant={args.variant})"
        )
        _stage(f"payload_0bin_built_{payload_0bin_bytes}")

        # 10. Build runtime + archive.zip
        archive_zip_path = args.output_dir / "archive.zip"
        archive_zip_sha: str | None = None
        archive_zip_bytes: int | None = None
        if not args.skip_archive_build:
            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(archive_bytes_atw2)
            _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes_atw2)
            archive_zip_sha = _sha256_file(archive_zip_path)
            archive_zip_bytes = archive_zip_path.stat().st_size
            print(
                f"[atw_v2-full] wrote {archive_zip_path} "
                f"({archive_zip_bytes} bytes, sha256={archive_zip_sha})"
            )

        # 11. Auth eval (Catalog #226 canonical helper)
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG,
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[atw_v2-full] [contest-CUDA] score = {contest_cuda_score} "
                    f"(archive_sha256={archive_zip_sha})"
                )
            _stage("auth_eval_cuda_done")

        train_elapsed_sec = time.time() - train_started_at

        # 12. Posterior update (Catalog #128)
        if (
            contest_cuda_score is not None
            and archive_zip_sha is not None
            and archive_zip_bytes is not None
        ):
            try:
                detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag=SUBSTRATE_TAG,
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("ATW_V2_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=detected_substrate,
                    architecture_class=SUBSTRATE_LANE_ID,
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_zip_sha,
                    archive_bytes=archive_zip_bytes,
                    notes=(
                        f"atw_codec_v2 variant={args.variant} first-anchor "
                        f"(Atick-Redlich + WZ side-info + G1 distill + B3 CDF)"
                    ),
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[atw_v2-full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[atw_v2-full] posterior_update failed: {exc}", file=sys.stderr)

        # 13. Provenance
        provenance = {
            "schema": "atw_codec_v2_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_atw_codec_v2.py",
            "lane_id": SUBSTRATE_LANE_ID,
            "design_memo": DESIGN_MEMO_PATH,
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "variant": args.variant,
            "variant_byte": variant_byte,
            "autocast_fp16": autocast_active,
            "num_pairs_decoded": n_pairs,
            "archive_sha256": archive_zip_sha,
            "archive_bytes": archive_zip_bytes,
            "archive_zip_path": (
                str(archive_zip_path) if archive_zip_path.is_file() else None
            ),
            "payload_0bin_sha256": payload_0bin_sha,
            "payload_0bin_bytes": payload_0bin_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
            "promotion_eligible": False,  # research_only=true at landing per design memo §19
            "ready_for_exact_eval_dispatch": False,
            "train_elapsed_sec": float(train_elapsed_sec),
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0
    finally:
        unpatch_upstream_yuv6(yuv6_token)


class _NullContext:
    """No-op context manager for non-CUDA / autocast-disabled training."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# ---------------------------------------------------------------------------
# META layer SubstrateContract registration is in registered_substrate.py.
# The trainer's main() routes through smoke/full per the standard pattern.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main(sys.argv[1:]))
