# SPDX-License-Identifier: MIT
"""Train the SABOR boundary-only renderer substrate (PAIR T+OPT3 2026-05-13).

Operator-callable training script per the Grand Council O1 first-principles
hypothesis (Shannon LEAD): the contest scorer's SegNet distortion is
``(argmax(out1) != argmax(out2)).float().mean()`` — only logit ORDERING
affects the score. The φ1 SABOR audit (2026-05-13) empirically confirmed
99.27% argmax-stable pixels at ε=32 RGB perturbation. SABOR consumes that
free-byte capacity by factoring each frame into (a) a sparse boundary mask +
high-fidelity boundary RGB and (b) per-class mean + per-pair bias + FiLM
refinement for the interior.

Substrate design references:

* ``src/tac/substrates/sabor_boundary_only_renderer/__init__.py`` — full
  archive grammar + 13 HNeRV parity-discipline lesson declarations.
* ``.omx/research/sabor_boundary_audit_20260513.md`` — empirical capacity audit.
* ``tools/measure_segnet_argmax_stable_interior.py`` — the measurement tool.

Council-binding CLAUDE.md non-negotiables honored end-to-end:

* Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic
  data; Catalog #114).
* Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract).
* ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
* ``apply_eval_roundtrip_during_training`` inside the per-batch loop.
* ``tac.training.EMA(decay=0.997)``; inference checkpoint = EMA shadow,
  NEVER live weights.
* AdamW lr cosine annealing; grad clip 1.0; NaN watchdog.
* End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS; CPU permitted only with ``--smoke``.
* Catalog #128 atomic posterior update.
* Catalog #146 contest-compliant inflate.sh / inflate.py emission.
* Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS declared as ``AnnAssign``
  (Catalog #168 META gate).
* Catalog #190 dynamic ``hardware_substrate`` detection.

Usage (smoke; CPU, tiny config, no scorer load)::

    .venv/bin/python experiments/train_substrate_sabor_boundary_only_renderer.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sabor_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_sabor_boundary_only_renderer.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sabor_full_<utc> \\
        --epochs 2000 --batch-size 32 --lr 1e-3 --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-in-_run_val_loop
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

EVAL_HW: tuple[int, int] = (384, 512)
N_PAIRS_FULL: int = 600
CONTEST_NORMALIZER: float = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest (Catalog #168 AnnAssign form for AST gate)
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "SABOR_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/sabor_boundary_audit_20260513.md#5-verdict"
        ),
    },
    "--output-dir": {
        "env": "SABOR_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "SABOR_EPOCHS",
        "rationale": (
            "training epochs; council target 2000 for full SABOR (boundary-only "
            "is cheaper per-epoch than full-renderer substrates)"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "SABOR_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "SABOR_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_sabor_boundary_only_renderer",
        description=(
            "Train SABOR boundary-only renderer substrate end-to-end "
            "(Council F O1; PAIR T+OPT3 2026-05-13)."
        ),
    )

    # ---- TIER_1 required ----
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")

    # ---- Training hyperparameters ----
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-pair-count", type=int, default=16)
    p.add_argument("--val-every-epochs", type=int, default=25)
    p.add_argument("--max-decoded-pairs", type=int, default=N_PAIRS_FULL)

    # ---- Substrate architecture knobs ----
    p.add_argument("--edge-threshold", type=float, default=0.04,
                   help="Canny-style gradient-magnitude threshold (phi1 audit boundary fraction band 1-3 percent).")
    p.add_argument("--refinement-hidden", type=int, default=32)
    p.add_argument("--refinement-blocks", type=int, default=2)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--bias-dim", type=int, default=3)

    # ---- Lagrangian weights ----
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    p.add_argument("--delta-boundary", type=float, default=0.1,
                   help="Boundary-consistency regularizer weight.")

    # ---- Modes ----
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load); proves wiring.")
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")

    # ---- Tier-1 engineering flags (waivers above) ----
    p.add_argument("--enable-autocast-fp16", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true")
    p.add_argument("--enable-tf32", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Helpers (canonical-skeleton-bound wrappers)
# ---------------------------------------------------------------------------


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None = None):
    return _decode_real_pairs_canonical(
        video_path, n_pairs=n_pairs, substrate_tag="sabor",
        max_pairs=max_pairs, repo_root=REPO_ROOT,
    )


def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(name, smoke=smoke, substrate_tag="sabor")


def _archive_bytes_proxy_closed_form(model, num_frames: int, h: int, w: int):
    """Closed-form bytes proxy for the rate term.

    Two dominant rate components: (a) decoder state_dict (fp16 + brotli),
    (b) boundary mask + boundary RGB (sparse — depends on edge_threshold).
    We use a conservative proxy assuming 2% boundary-pixel fraction.
    """
    import torch

    n_decoder = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith("class_means")
    )
    boundary_pixels = int(0.02 * num_frames * h * w)  # 2% conservative
    decoder_bytes = n_decoder * 2  # fp16
    class_means_bytes = model.cfg.num_seg_classes * 3
    mask_bytes = (num_frames * h * w + 7) // 8  # packbits
    boundary_rgb_bytes = boundary_pixels * 3
    seg_argmax_bytes = num_frames * h * w  # uint8
    bytes_proxy = float(
        decoder_bytes + class_means_bytes + mask_bytes
        + boundary_rgb_bytes + seg_argmax_bytes
    )
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Inflate-time runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------


def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py``.

    Catalog #146: 3-positional-arg contract, ``set -euo pipefail``, no
    runtime network/dep fetches, no scorer imports.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# SABOR boundary-only renderer contest-compliant inflate.\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
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

    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"SABOR contest-compliant inflate runtime.\n"
        "\n"
        "Reads archive_dir/0.bin via the packaged substrate parser; for each\n"
        "entry in file_list emits one contest .raw frame stream. No scorer\n"
        "imports (strict-scorer-rule contract).\n"
        "\"\"\"\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.sabor_boundary_only_renderer.inflate import inflate_one_video\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        dst = output_dir / f'{base}.raw'\n"
        "        inflate_one_video(archive_bytes, dst, device_str='cpu')\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(
    archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path
) -> None:
    """Deterministic charged archive.zip containing only the data packet."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)
        # archive.zip is the charged data packet. Runtime files live beside it
        # in submission_dir and are evaluated through --inflate-sh custody.


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke; proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.sabor_boundary_only_renderer import (
        SaborBoundaryOnlyConfig,
        SaborBoundaryOnlyRenderer,
        detect_boundary_mask_canny_segnet,
        pack_archive,
        parse_archive,
    )

    _pin_seeds(args.seed)

    cfg = SaborBoundaryOnlyConfig(
        num_pairs=4,
        output_height=24,
        output_width=32,
        edge_threshold=args.edge_threshold,
        refinement_hidden=args.refinement_hidden,
        refinement_blocks=args.refinement_blocks,
        embedding_dim=args.embedding_dim,
        bias_dim=args.bias_dim,
    )
    device = _device_or_die(args.device, smoke=True)
    model = SaborBoundaryOnlyRenderer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = cfg.num_pairs * 2
    h, w = cfg.output_height, cfg.output_width

    print(f"[smoke] SABOR params: {model.num_parameters():,}")

    # Synthetic pairs for smoke ONLY (Catalog #114: synthetic non-smoke FORBIDDEN).
    gt_pairs = torch.rand(cfg.num_pairs, 2, 3, h, w, device=device)

    # Compute boundary mask + boundary RGB up-front (precomputed at archive time).
    boundary_mask = torch.zeros(num_frames, h, w, dtype=torch.bool, device=device)
    boundary_rgb_targets = torch.zeros(cfg.num_pairs, 2, 3, h, w, device=device)
    segnet_argmax = torch.randint(0, cfg.num_seg_classes, (num_frames, h, w),
                                  dtype=torch.long, device=device)
    for pi in range(cfg.num_pairs):
        for fi in range(2):
            mask_bhw = detect_boundary_mask_canny_segnet(
                gt_pairs[pi, fi].unsqueeze(0),
                segnet_argmax[2 * pi + fi].unsqueeze(0),
                edge_threshold=args.edge_threshold,
            )
            boundary_mask[2 * pi + fi] = mask_bhw[0]
            boundary_rgb_targets[pi, fi] = gt_pairs[pi, fi]
    print(f"[smoke] boundary count: {int(boundary_mask.sum())} "
          f"({100.0 * float(boundary_mask.sum()) / boundary_mask.numel():.2f}%)")

    seg_pair_5d = segnet_argmax.view(cfg.num_pairs, 2, h, w)
    mask_pair_5d = boundary_mask.view(cfg.num_pairs, 2, h, w)

    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(
            idx, mask_pair_5d, boundary_rgb_targets, seg_pair_5d,
        )
        # Trivial smoke loss — just to prove gradient flow
        loss = ((rgb_0 - gt_pairs[:, 0]).abs().mean()
                + (rgb_1 - gt_pairs[:, 1]).abs().mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    # Build a smoke archive and roundtrip to verify the grammar is consistent.
    bp_count = int(boundary_mask.sum().item())
    boundary_rgb_flat = torch.zeros(bp_count, 3, dtype=torch.uint8, device=device)
    cm = model.quantize_class_means_for_archive().cpu()
    sd_cpu = {
        k: v.detach().cpu() for k, v in model.runtime_state_dict_for_archive().items()
    }
    blob = pack_archive(
        decoder_state_dict=sd_cpu,
        class_means=cm,
        boundary_mask=boundary_mask.cpu(),
        boundary_rgb_flat=boundary_rgb_flat.cpu(),
        segnet_argmax=segnet_argmax.to(torch.uint8).cpu(),
        meta={"smoke": True, "edge_threshold": args.edge_threshold},
        num_pairs=cfg.num_pairs,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        num_seg_classes=cfg.num_seg_classes,
        refinement_hidden=cfg.refinement_hidden,
        refinement_blocks=cfg.refinement_blocks,
        embedding_dim=cfg.embedding_dim,
        bias_dim=cfg.bias_dim,
        edge_threshold=cfg.edge_threshold,
    )
    arc = parse_archive(blob)
    print(f"[smoke] archive bytes: {len(blob)}; roundtrip OK (boundary_count="
          f"{arc.boundary_pixel_count})")

    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "smoke": True,
        "archive_bytes": len(blob),
        "boundary_count": bp_count,
    }, ckpt_path)
    print(f"[smoke] wrote {ckpt_path}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end)
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point — requires CUDA + score-aware scorers."""
    import torch
    import torch.nn.functional as F

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.sabor_boundary_only_renderer import (
        SaborBoundaryOnlyConfig,
        SaborBoundaryOnlyLossWeights,
        SaborBoundaryOnlyRenderer,
        SaborBoundaryOnlyScoreAwareLoss,
        detect_boundary_mask_canny_segnet,
        pack_archive,
    )
    from tac.training import EMA

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")

    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # Decode real frame pairs (Catalog #114; NOT synthetic).
        pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_decoded_pairs,
        ).to(device)
        n_pairs = int(pair_tensor.shape[0])
        h, w = EVAL_HW
        # The canonical decoder returns [0, 255]; SABOR loss expects [0, 1] for
        # the rendered output (it rescales to 255 internally). Convert GT to [0, 1].
        gt_pairs_unit = pair_tensor / 255.0
        num_frames = n_pairs * 2
        _stage(f"pairs_decoded_{n_pairs}")

        # Precompute SegNet argmax + boundary mask + boundary RGB at GT time.
        # The boundary mask + RGB + argmax are stored in the archive (not learned).
        seg_argmax = torch.zeros(num_frames, h, w, dtype=torch.long, device=device)
        with torch.no_grad():
            for pi in range(n_pairs):
                for fi in range(2):
                    # SegNet expects (B, T=2, C, H, W) at scorer-resolution; we
                    # pass the per-frame argmax via a singleton pair.
                    one_frame = gt_pairs_unit[pi, fi].unsqueeze(0) * 255.0  # (1, 3, H, W)
                    pair_btchw = torch.stack([one_frame, one_frame], dim=1)
                    seg_logits = segnet(segnet.preprocess_input(pair_btchw))
                    if seg_logits.dim() == 4:
                        # (B, C, H_s, W_s) — resize to (h, w) if needed.
                        if tuple(seg_logits.shape[-2:]) != (h, w):
                            seg_logits = F.interpolate(
                                seg_logits, size=(h, w), mode="bilinear",
                                align_corners=False,
                            )
                        argmax = seg_logits.argmax(dim=1).squeeze(0)
                    else:
                        # Fallback: treat as (B, C) global logits (smoke stubs).
                        argmax = torch.zeros(h, w, dtype=torch.long, device=device)
                    seg_argmax[2 * pi + fi] = argmax
        _stage("segnet_argmax_precomputed")

        # Compute boundary mask = Canny union SegNet-4nbr-disagreement.
        boundary_mask = torch.zeros(num_frames, h, w, dtype=torch.bool, device=device)
        for pi in range(n_pairs):
            for fi in range(2):
                bm = detect_boundary_mask_canny_segnet(
                    gt_pairs_unit[pi, fi].unsqueeze(0),
                    seg_argmax[2 * pi + fi].unsqueeze(0),
                    edge_threshold=args.edge_threshold,
                )
                boundary_mask[2 * pi + fi] = bm[0]
        bp_count = int(boundary_mask.sum().item())
        bp_fraction = bp_count / max(1, num_frames * h * w)
        print(
            f"[full] boundary mask: count={bp_count:,} fraction={bp_fraction:.4f}"
        )
        _stage(f"boundary_mask_built_{bp_count}_pixels")

        # boundary_rgb_target: the high-fidelity RGB at boundary pixels.
        # Stored verbatim in the archive (uint8). At inflate time the
        # renderer overlays these onto the texture-filled interior.
        boundary_rgb_target = gt_pairs_unit.clone()  # (n_pairs, 2, 3, H, W) in [0, 1]

        # Build train/val split (last val_pair_count pairs held out).
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
        val_indices = torch.arange(val_idx_start, n_pairs, device=device, dtype=torch.long)

        # Build model.
        cfg = SaborBoundaryOnlyConfig(
            num_pairs=n_pairs,
            output_height=h,
            output_width=w,
            edge_threshold=args.edge_threshold,
            refinement_hidden=args.refinement_hidden,
            refinement_blocks=args.refinement_blocks,
            embedding_dim=args.embedding_dim,
            bias_dim=args.bias_dim,
        )
        model = SaborBoundaryOnlyRenderer(cfg).to(device)
        # Initialize class_means from per-class GT mean RGB for warm start.
        _initialize_class_means(model, gt_pairs_unit, seg_argmax)
        print(f"[full] SABOR params: {model.num_parameters():,}")
        _stage("model_built_with_class_mean_init")

        # EMA shadow.
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # Lagrangian.
        weights = SaborBoundaryOnlyLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            delta_boundary_consistency=args.delta_boundary,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = SaborBoundaryOnlyScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights,
        )
        _stage("lagrangian_built")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"
        archive_bytes_proxy = _archive_bytes_proxy_closed_form(
            model, num_frames, h, w,
        )
        nan_strike = 0
        max_nan_strikes = 3
        n_train = int(train_indices.shape[0])
        bs = max(1, args.batch_size)
        boundary_mask_pair = boundary_mask.view(n_pairs, 2, h, w)
        seg_argmax_pair = seg_argmax.view(n_pairs, 2, h, w)

        for epoch in range(args.epochs):
            model.train()
            perm = train_indices[torch.randperm(n_train, device=device)]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                if idx.numel() == 0:
                    continue
                mask_b = boundary_mask_pair[idx]
                rgb_targets_b = boundary_rgb_target[idx]
                seg_b = seg_argmax_pair[idx]
                rgb_0, rgb_1 = model(idx, mask_b, rgb_targets_b, seg_b)
                gt = gt_pairs_unit[idx]
                loss, _ = loss_fn(
                    rgb_0, rgb_1, gt[:, 0], gt[:, 1], archive_bytes_proxy,
                    mask_b, rgb_targets_b,
                    apply_eval_roundtrip=True, noise_std=args.noise_std,
                )
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[full] WARN: non-finite loss epoch {epoch} batch "
                        f"{start} strike {nan_strike}/{max_nan_strikes}",
                        file=sys.stderr,
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                nan_strike = 0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(model)
                epoch_loss += float(loss.detach().item())
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)

            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                val_lag = _run_val_loop(
                    model, ema, loss_fn, gt_pairs_unit, boundary_mask_pair,
                    boundary_rgb_target, seg_argmax_pair, val_indices,
                    archive_bytes_proxy, args.noise_std,
                )
                print(
                    f"[full] epoch {epoch + 1}/{args.epochs} "
                    f"train_avg={avg_loss:.6f} val_lag={val_lag:.6f} "
                    f"(best={best_val_lag:.6f}@{best_epoch + 1})"
                )
                if val_lag < best_val_lag and math.isfinite(val_lag):
                    best_val_lag = val_lag
                    best_epoch = epoch
                    ema_state = ema.state_dict()
                    torch.save({
                        "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                        "config": asdict(cfg),
                        "ema_decay": args.ema_decay,
                        "best_val_lagrangian": val_lag,
                        "best_epoch": int(epoch),
                        "saved_at_utc": _utc_now_iso(),
                    }, ckpt_best_path)
            elif (epoch + 1) % max(1, args.val_every_epochs // 4) == 0:
                print(f"[full] epoch {epoch + 1}/{args.epochs} train_avg={avg_loss:.6f}")

        train_elapsed = time.time() - train_started_at
        _stage(f"train_complete_elapsed_{int(train_elapsed)}s")

        if not ckpt_best_path.is_file():
            print("[full] WARN: no improving val checkpoint; saving end-of-train EMA.",
                  file=sys.stderr)
            ema_state = ema.state_dict()
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                "config": asdict(cfg),
                "ema_decay": args.ema_decay,
                "best_val_lagrangian": best_val_lag,
                "best_epoch": int(args.epochs - 1),
                "saved_at_utc": _utc_now_iso(),
                "fallback_end_of_training_save": True,
            }, ckpt_best_path)

        # Build the SBO1 archive bytes from EMA shadow.
        archive_sha = ""
        archive_bytes_total = 0
        archive_zip_path = args.output_dir / "archive.zip"
        contest_cuda_score: float | None = None
        auth_eval_result_path: Path | None = None

        if not args.skip_archive_build:
            ema_ckpt = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            cpu_model = SaborBoundaryOnlyRenderer(cfg).to("cpu")
            cpu_model.load_state_dict(ema_ckpt["state_dict"], strict=False)
            cm_q = cpu_model.quantize_class_means_for_archive()
            # Boundary RGB flat: read every boundary-True pixel from the GT
            # (boundary RGB is verbatim stored; the renderer overlays it).
            boundary_rgb_flat = _flatten_boundary_rgb(
                gt_pairs_unit.cpu(), boundary_mask_pair.cpu()
            )
            sd_cpu = cpu_model.runtime_state_dict_for_archive()
            meta = {
                "schema": "sabor_meta_v1",
                "boundary_fraction": bp_fraction,
                "boundary_pixel_count": bp_count,
                "edge_threshold": args.edge_threshold,
                "lane_id": "lane_sabor_boundary_only_renderer_substrate_20260513",
            }
            bin_bytes = pack_archive(
                decoder_state_dict=sd_cpu,
                class_means=cm_q,
                boundary_mask=boundary_mask.cpu(),
                boundary_rgb_flat=boundary_rgb_flat,
                segnet_argmax=seg_argmax.to(torch.uint8).cpu(),
                meta=meta,
                num_pairs=cfg.num_pairs,
                output_height=cfg.output_height,
                output_width=cfg.output_width,
                num_seg_classes=cfg.num_seg_classes,
                refinement_hidden=cfg.refinement_hidden,
                refinement_blocks=cfg.refinement_blocks,
                embedding_dim=cfg.embedding_dim,
                bias_dim=cfg.bias_dim,
                edge_threshold=cfg.edge_threshold,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes_total = len(bin_bytes)
            print(f"[full] wrote 0.bin ({archive_bytes_total} bytes, "
                  f"sha256={archive_sha})")

            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir,
            )
            _stage(f"archive_built_bytes_{archive_bytes_total}")

        # CUDA auth eval — canonical helper (Catalog #226 self-protect).
        if not args.skip_auth_eval and archive_zip_path.is_file():
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="sabor",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full] [contest-CUDA] score = {contest_cuda_score} "
                    f"(axis={auth_result['auth_eval_score_axis']}, "
                    f"lane_tag={auth_result['auth_eval_lane_tag']}, "
                    f"archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        # Posterior update (Catalog #128).
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )
                _hw_sub = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="sabor",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("SABOR_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_hw_sub,
                    architecture_class="lane_sabor_boundary_only_renderer_substrate_20260513",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes_total,
                    notes=f"SABOR first-anchor; epochs={args.epochs}",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}",
                      file=sys.stderr)

        # Provenance manifest.
        provenance = {
            "schema": "sabor_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_sabor_boundary_only_renderer.py",
            "lane_id": "lane_sabor_boundary_only_renderer_substrate_20260513",
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": n_pairs,
            "best_val_lagrangian": (
                best_val_lag if math.isfinite(best_val_lag) else None
            ),
            "best_epoch": int(best_epoch),
            "train_elapsed_sec": float(train_elapsed),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes_total,
            "boundary_pixel_count": bp_count,
            "boundary_fraction": bp_fraction,
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
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "predicted_band": [0.165, 0.185],
            "predicted_band_basis": (
                "Council F O1 + sabor_boundary_audit_20260513 capacity proof"
            ),
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {args.output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_val_loop(
    model, ema, loss_fn, gt_pairs_unit, boundary_mask_pair, boundary_rgb_target,
    seg_argmax_pair, val_indices, archive_bytes_proxy, noise_std,
) -> float:
    """Snapshot+restore eval forward — Catalog #88 + Catalog #180."""
    import torch

    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    model.eval()
    try:
        with torch.inference_mode():
            mask_b = boundary_mask_pair[val_indices]
            rgb_t = boundary_rgb_target[val_indices]
            seg_b = seg_argmax_pair[val_indices]
            rgb_0, rgb_1 = model(val_indices, mask_b, rgb_t, seg_b)
            gt = gt_pairs_unit[val_indices]
            loss, _ = loss_fn(
                rgb_0, rgb_1, gt[:, 0], gt[:, 1], archive_bytes_proxy,
                mask_b, rgb_t, apply_eval_roundtrip=True, noise_std=noise_std,
            )
            return float(loss.detach().item())
    finally:
        model.load_state_dict(orig_state)
        model.train()


def _initialize_class_means(model, gt_pairs_unit, seg_argmax) -> None:
    """Initialize class_means from per-class GT mean RGB on first frames."""
    import torch

    with torch.no_grad():
        device = model.class_means.device
        n_pairs = int(gt_pairs_unit.shape[0])
        flat_rgb = gt_pairs_unit.view(n_pairs * 2, 3, -1).permute(0, 2, 1).reshape(-1, 3)
        flat_seg = seg_argmax.view(-1)
        new_means = model.class_means.detach().clone().to(device)
        for c in range(model.cfg.num_seg_classes):
            mask = flat_seg == c
            if int(mask.sum().item()) > 0:
                new_means[c] = flat_rgb[mask].mean(dim=0).to(device)
        model.class_means.copy_(new_means.clamp(0.0, 1.0))


def _flatten_boundary_rgb(gt_pairs_unit, boundary_mask_pair):
    """Extract uint8 RGB at every True boundary pixel, row-major across frames."""
    import torch

    n_pairs = int(gt_pairs_unit.shape[0])
    out_rows: list[torch.Tensor] = []
    for pi in range(n_pairs):
        for fi in range(2):
            mask_hw = boundary_mask_pair[pi, fi]  # (H, W) bool
            rgb_chw = gt_pairs_unit[pi, fi]  # (3, H, W) in [0, 1]
            if not bool(mask_hw.any().item()):
                continue
            sel = rgb_chw.permute(1, 2, 0)[mask_hw]  # (n_b, 3) in [0, 1]
            q = (sel.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            out_rows.append(q)
    if not out_rows:
        return torch.zeros(0, 3, dtype=torch.uint8)
    return torch.cat(out_rows, dim=0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
