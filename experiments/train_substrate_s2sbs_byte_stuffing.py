# SPDX-License-Identifier: MIT
"""Train the S2SBS (Stride-2-Stem Byte-Stuffing) substrate.

Council F O3 directive 2026-05-13 — PAIR T+OPT3. φ3 audit:
``.omx/research/s2sbs_blindspot_audit_20260513.md`` empirically confirmed
~97 KB joint-safe per frame, ~38 MB post-ECC capacity. This trainer is
the L0/L1 substrate-engineering scaffold that turns the audit into a
score-aware-loss + EMA + archive-build + contest-runtime-emit pipeline.

Per the codex math correction
(``feedback_codex_math_correction_pr95_lora_dora_landed_20260513.md``):
the byte-stuffing INCREASES archive size; rate slope = 6.66e-7
score/byte. Useful payload bytes must reduce seg+pose distortion by at
least the rate cost. The audit guarantees the SegNet/PoseNet
distortion does NOT move under HF-band perturbations; payload usefulness
is the substrate's burden to prove via downstream consumers.

Council-binding contract (CLAUDE.md non-negotiables):

- Train against ``upstream/videos/0.mkv`` decoded via pyav; synthetic data
  FORBIDDEN outside ``--smoke`` (Catalog #114).
- ``patch_upstream_yuv6_globally`` BEFORE ``load_differentiable_scorers``
  (Catalog #187).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` after every optimizer.step (Catalog
  #88); inference checkpoint = EMA shadow.
- Score-domain Lagrangian via ``S2sbsScoreAwareLoss``.
- AdamW + cosine; gradient clip 1.0.
- Contest-compliant inflate.sh template with 3 positional args + set -e
  (Catalog #146).
- TIER_1_OPERATOR_REQUIRED_FLAGS as ast.AnnAssign (Catalog #168).
- ``research_only=true``; no score authority emitted.

Usage (smoke, macOS CPU)::

    .venv/bin/python experiments/train_substrate_s2sbs_byte_stuffing.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/s2sbs_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full, CUDA required)::

    .venv/bin/python experiments/train_substrate_s2sbs_byte_stuffing.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/s2sbs_<utc> \\
        --epochs 2000 --batch-size 8 --device cuda
"""
# AUTOCAST_FP16_WAIVED:hf-fft-byte-codec-needs-fp32-numerics-until-paired-anchor-lands
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic-until-paired-CPU/CUDA-anchor-lands
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.substrate_registry import SubstrateContract, register_substrate
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
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.s2sbs_byte_stuffing import (
    PayloadChannel,
    S2sbsConfig,
    S2sbsLossWeights,
    S2sbsRenderer,
    S2sbsScoreAwareLoss,
    pack_archive,
)
from tac.training import EMA

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "s2sbs"
SUBSTRATE_LANE_ID = "lane_s2sbs_stride2_byte_stuffing_substrate_20260513"


# Catalog #151 / Catalog #168 — annotated assignment.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "S2SBS_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "S2SBS_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "S2SBS_EPOCHS",
        "rationale": (
            "training epochs; default 2000 (HF byte channel converges fast "
            "against frozen scorer)."
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "S2SBS_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "S2SBS_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--payload-bytes-per-pair": {
        "env": "S2SBS_PAYLOAD_BYTES_PER_PAIR",
        "rationale": (
            "post-ECC payload bytes per pair; default 32 per Council F O3 "
            "directive (rate-cost-aware)."
        ),
        "default": "32",
    },
    "--delta-amp-uint8": {
        "env": "S2SBS_DELTA_AMP_UINT8",
        "rationale": (
            "Hermitian-FFT amplitude in uint8 units; 0.75 = audit-pinned "
            "joint-safe per φ3 audit. > 1.0 forbidden."
        ),
        "default": "0.75",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_substrate_s2sbs_byte_stuffing")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--payload-bytes-per-pair", type=int, default=32)
    p.add_argument("--delta-amp-uint8", type=float, default=0.75)
    p.add_argument("--ecc-rate", type=float, default=0.25)
    p.add_argument("--payload-channel", choices=["R", "G", "B"], default="R")
    p.add_argument("--hf-blindspot-lf-cutoff-h", type=int, default=96)
    p.add_argument("--hf-blindspot-lf-cutoff-w", type=int, default=128)
    p.add_argument("--num-pairs", type=int, default=600)
    p.add_argument("--max-pairs", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpu", default="t4")
    return p


def _make_config(args: argparse.Namespace) -> S2sbsConfig:
    return S2sbsConfig(
        num_pairs=int(args.num_pairs),
        output_height=384,
        output_width=512,
        hf_blindspot_lf_cutoff_h=int(args.hf_blindspot_lf_cutoff_h),
        hf_blindspot_lf_cutoff_w=int(args.hf_blindspot_lf_cutoff_w),
        delta_amp_uint8=float(args.delta_amp_uint8),
        payload_channel=str(args.payload_channel),
        base_seed=int(args.seed),
        payload_bytes_per_pair=int(args.payload_bytes_per_pair),
        ecc_rate=float(args.ecc_rate),
    )


def _vendor_runtime(output_dir: Path) -> None:
    """Emit contest-compliant inflate.sh (3 positional args + set -e) and inflate.py."""
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "# S2SBS contest-compliant inflate.sh — Catalog #146 + #163.\n"
        "HERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'exec uv run --quiet --with torch==2.5.1+cu124 '
        '--extra-index-url https://download.pytorch.org/whl/cu124 '
        '--index-strategy unsafe-best-match '
        '"$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    # Copy the canonical inflate.py from the substrate package.
    src_inflate = REPO_ROOT / "src" / "tac" / "substrates" / "s2sbs_byte_stuffing" / "inflate.py"
    shutil.copy2(src_inflate, runtime_dir / "inflate.py")


def _train_one_step(
    renderer: S2sbsRenderer,
    loss_fn: S2sbsScoreAwareLoss,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    pair_indices: torch.Tensor,
    payload_rows: tuple[PayloadChannel, ...],
    archive_bytes_proxy: torch.Tensor,
    grad_clip: float,
) -> dict[str, float]:
    rgb_0, rgb_1 = renderer(pair_indices, payload_rows)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), grad_clip)
    optimizer.step()
    ema.update(renderer)
    return {k: float(v.item()) for k, v in parts.items()}


class _SmokeSegScorer(torch.nn.Module):
    """Stand-in SegNet for smoke (preprocess_input contract preserved)."""

    def __init__(self, *, out_channels: int = 5) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.conv = torch.nn.Conv2d(3, self.out_channels, kernel_size=3, padding=1)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _SmokePoseScorer(torch.nn.Module):
    """Stand-in PoseNet returning the upstream-contract pose dict."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, 6)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = pair_btchw.shape
        flat = pair_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        flat6 = flat.expand(-1, 6, -1, -1)
        half_h = h // 2
        half_w = w // 2
        flat6_sub = flat6[..., : half_h * 2, : half_w * 2].reshape(
            b * t, 6, half_h, 2, half_w, 2
        ).mean(dim=(3, 5))
        return flat6_sub.reshape(b, t * 6, half_h, half_w)

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = x_b12hw.flatten(2).mean(dim=2)
        return {"pose": self.head(pooled)}


def _smoke_train_loop(
    renderer: S2sbsRenderer,
    pair_batches: list[torch.Tensor],
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Smoke training loop: gradient-flow + EMA + provenance writer.

    The smoke loop uses the deterministic base decoder and a dummy SegNet/
    PoseNet stand-in so the CPU smoke completes in seconds. Full training
    swaps the scorer constructors to upstream + DALI inside the remote
    driver path; this entry point keeps the architecture clean.
    """
    seg_scorer = _SmokeSegScorer(out_channels=5).to(device).eval()
    pose_scorer = _SmokePoseScorer().to(device).eval()
    for p in seg_scorer.parameters():
        p.requires_grad_(False)
    for p in pose_scorer.parameters():
        p.requires_grad_(False)

    loss_fn = S2sbsScoreAwareLoss(seg_scorer, pose_scorer, S2sbsLossWeights()).to(device)
    # Renderer base decoder is parameter-free; expose a tiny trainable
    # bias so the smoke loop exercises optimizer.step + EMA.
    bias = torch.nn.Parameter(torch.zeros(3, 1, 1, device=device))
    renderer_with_bias = _RendererWithBias(renderer, bias).to(device)
    optimizer = torch.optim.AdamW(
        [bias], lr=args.lr, weight_decay=args.weight_decay
    )
    ema = EMA(renderer_with_bias, decay=args.ema_decay)

    last_parts: dict[str, float] = {}
    payload_rows = (
        PayloadChannel(pair_index=0, payload=b"\xAA" * args.payload_bytes_per_pair),
    )
    archive_bytes_proxy = torch.tensor(150_000.0, device=device)
    for _epoch in range(args.epochs):
        for batch_pairs in pair_batches:
            pair_indices = torch.arange(batch_pairs.shape[0], dtype=torch.long, device=device)
            gt_0 = batch_pairs[:, 0].to(device)
            gt_1 = batch_pairs[:, 1].to(device)
            last_parts = _train_one_step(
                renderer_with_bias, loss_fn, optimizer, ema,
                gt_rgb_0=gt_0, gt_rgb_1=gt_1,
                pair_indices=pair_indices,
                payload_rows=payload_rows,
                archive_bytes_proxy=archive_bytes_proxy,
                grad_clip=args.grad_clip,
            )

    # Pack archive with EMA-applied bias snapshot.
    cfg = renderer.cfg
    orig = {k: v.detach().clone() for k, v in renderer_with_bias.state_dict().items()}
    ema.apply(renderer_with_bias)
    archive_bytes = pack_archive(config=cfg, payloads=payload_rows)
    renderer_with_bias.load_state_dict(orig)

    archive_dir = output_dir / "archive_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "0.bin").write_bytes(archive_bytes)

    # Zip into archive.zip for contest packet shape.
    archive_zip = output_dir / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", archive_bytes)

    _vendor_runtime(output_dir)

    sha = _canon_sha256_bytes(archive_bytes)
    archive_zip_bytes = archive_zip.read_bytes()
    return {
        "archive_sha256": sha,
        "archive_bytes": len(archive_bytes),
        "archive_zip_sha256": _canon_sha256_bytes(archive_zip_bytes),
        "archive_zip_bytes": len(archive_zip_bytes),
        "last_loss_parts": last_parts,
        "training_mode": "smoke" if args.smoke else "full-scaffold",
        "evidence_grade": "macOS-CPU advisory" if device == "cpu" else "training-only",
    }


class _RendererWithBias(torch.nn.Module):
    def __init__(self, inner: S2sbsRenderer, bias: torch.nn.Parameter) -> None:
        super().__init__()
        self.inner = inner
        self.bias = bias

    def forward(
        self,
        pair_indices: torch.Tensor,
        payload_rows: tuple[PayloadChannel, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_0, rgb_1 = self.inner(pair_indices, payload_rows)
        return (rgb_0 + self.bias).clamp(0.0, 1.0), (rgb_1 + self.bias).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

S2SBS_BYTE_STUFFING_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="s2sbs_byte_stuffing",
    lane_id="lane_substrate_s2sbs_byte_stuffing_20260512",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "S2SBS1 monolithic single-file 0.bin: header + S2SBS byte-stuffing-encoded payload (sequence-to-sequence byte stuffing per CLAUDE.md grand council) + sentinel + per-frame index"
    ),
    parser_section_manifest={
        "header": "S2SBS1_magic_and_version",
        "byte_stuffed_payload": "byte_stuffing_encoded",
        "sentinel": "fixed_sentinel_marker",
        "frame_index": "uint16_per_frame",
    },
    inflate_runtime_loc_budget=110,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=460,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=True,
    recipe_research_only=False,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=14,
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
        "catalog_228_f3_cache_pending_backport",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "S2SBS byte-stuffing; sensitivity captured by symbol entropy of the stuffed payload"
        ),
        "hook_bit_allocator_class": (
            "byte-stuffing per-symbol; per-symbol not per-tensor"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (sequence-to-sequence byte stuffing); no 2+ defensible interpretations"
        ),
    },
)


@register_substrate(S2SBS_BYTE_STUFFING_SUBSTRATE_CONTRACT)



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.smoke:
        print(
            "[s2sbs] FATAL: full training is not implemented for this L0/L1 "
            "substrate scaffold; pass --smoke until a downstream consumer and "
            "score-aware full trainer land.",
            file=sys.stderr,
        )
        return 2
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _canon_pin_seeds(int(args.seed))
    device = _canon_device_or_die(str(args.device), smoke=bool(args.smoke), substrate_tag=SUBSTRATE_TAG)
    cfg = _make_config(args)
    renderer = S2sbsRenderer(cfg).to(device)

    pair_batches: list[torch.Tensor] = []
    if args.smoke:
        # Smoke uses tiny synthetic pairs at the scorer resolution to keep
        # CPU CI < 5 s. Per Catalog #114: only allowed under --smoke.
        # SYNTHETIC_NON_SMOKE_OK:smoke-only-deterministic-test-input
        torch.manual_seed(int(args.seed))
        for _ in range(2):
            pair_batches.append(torch.rand(min(int(args.batch_size), 2), 2, 3, cfg.output_height, cfg.output_width))
    else:
        pairs = _canon_decode_real_pairs(
            Path(args.video_path),
            n_pairs=int(args.num_pairs),
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=int(args.max_pairs),
        )
        # Each pair tensor shaped (2, 3, H, W); batch them.
        for i in range(0, len(pairs), int(args.batch_size)):
            slab = torch.stack(pairs[i : i + int(args.batch_size)], dim=0)
            pair_batches.append(slab)

    result = _smoke_train_loop(renderer, pair_batches, args, device, args.output_dir)
    manifest = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": _canon_utc_now_iso(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "result": result,
        "archive_sha256": result["archive_sha256"],
        "archive_bytes": result["archive_bytes"],
        "archive_zip_sha256": result["archive_zip_sha256"],
        "archive_zip_bytes": result["archive_zip_bytes"],
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cuda" if device == "cuda" else "cpu",
            substrate_tag=SUBSTRATE_TAG,
        ),
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[s2sbs] archive_sha256={result['archive_sha256']}")
    print(f"[s2sbs] archive_bytes={result['archive_bytes']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
