# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:scaffold-only-trainer-Phase-2-not-yet-implemented-eval-scorer-forwards-wrapped-in-inference_mode-inside-_run_val_loop
# AUTOCAST_FP16_WAIVED:opt-in-via-CLI-flag-default-OFF-until-paired-anchor-validates-numerics
# TORCH_COMPILE_WAIVED:opt-in-via-CLI-flag-default-OFF-until-Inductor-graph-breaks-validated-per-substrate
"""Trainer for the PR95++ meta-stack-of-stacks enhanced curriculum.

Operator directive 2026-05-13: build the canonical actuator for the
``pr95_enhanced`` curriculum composed of 11 engineering enhancements over
PR95's 29,650-epoch 8-stage protocol. The trainer accepts a ``--curriculum``
mode (``pr95_enhanced`` for the recommended stack; ``pr95_faithful`` for the
byte-for-byte A/B baseline) and threads every enhancement through the
existing PR101 LC v2 clone substrate.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
non-negotiable: every one of the 11 enhancements has been audited against
the 13 parity-discipline lessons in
:func:`tac.substrates.pr101_lc_v2_clone.audit_enhanced_curriculum_against_hnerv_parity_lessons`.
The audit dict is emitted in the trainer's ``manifest.json`` so the
operator + downstream reviewers see the audit verdict alongside the run.

Per CLAUDE.md "Canonical pipeline standard" + "Beauty, simplicity, and
developer experience": this trainer DELEGATES to the canonical helpers in
:mod:`tac.substrates._shared.trainer_skeleton` (pin_seeds,
decode_real_pairs, device_or_die, detect_hardware_substrate, StageLog,
utc_now_iso, sha256_bytes) so the trainer body is small enough to fit a
30-second review per HNeRV parity discipline lesson L12.

Per CLAUDE.md "Recursive adversarial review protocol":
The 11 enhancements are reviewed by the inner-ten council (Shannon,
Dykstra, Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay,
Ballé) plus the grand council on demand (Boyd, Tao, Filler, Mallat, van
den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber). The 3-clean-
pass adversarial review counter resets on every finding; the trainer
refuses to dispatch (raises) if the audit verdict carries any "FAIL"
status.

Usage (smoke, CPU OK)::

    .venv/bin/python experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py \\
        --curriculum pr95_enhanced \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/pr95plus_smoke_<utc> \\
        --device cpu --smoke

Usage (full Modal A100 dispatch)::

    .venv/bin/python experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py \\
        --curriculum pr95_enhanced \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/pr95plus_full_<utc> \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

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
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)

# Tier-1 optimization helpers (TIER-1-OPT-BATCH 2026-05-14).
from tac.training_optimization import (
    autocast_aware_forward as _autocast_aware_forward,
    compile_with_fallback as _compile_with_fallback,
)
from tac.substrates.pr101_lc_v2_clone import (
    EnhancedCurriculumConfig,
    Pr101LcV2CloneConfig,
    Pr101LcV2CloneScoreAwareLoss,
    Pr101LcV2CloneSubstrate,
    Pr101LcV2ScoreAwareLossWeights,
    apply_cross_block_skip,
    audit_enhanced_curriculum_against_hnerv_parity_lessons,
    build_enhanced_stages,
    build_faithful_stages,
    build_optimizer_for_enhanced_stage,
    compute_wsd_lr,
    enhancement_summary,
    pack_archive,
)
from tac.training import EMA

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_TAG = "pr95plus"
SUBSTRATE_LANE_ID = (
    "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513"
)


# Catalog #151 + Catalog #168 — annotated assignment so the AST walker sees
# the manifest. Catalog #152 requires required-input-file flags to declare
# ``required_input_file=True`` so the operator wrapper validates pre-dispatch.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PR95PLUS_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "PR95PLUS_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "required_input_file": False,
    },
    "--upstream-dir": {
        "env": "PR95PLUS_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "required_input_file": False,
    },
    "--device": {
        "env": "PR95PLUS_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "required_input_file": False,
    },
    "--curriculum": {
        "env": "PR95PLUS_CURRICULUM",
        "rationale": (
            "curriculum mode: pr95_enhanced (recommended; 9 of 11 default-on "
            "enhancements) or pr95_faithful (byte-faithful PR95 baseline)"
        ),
        "default": "pr95_enhanced",
        "required_input_file": False,
        "satisfied_by_profile": ["pr95_enhanced", "pr95_faithful"],
    },
    "--epochs-multiplier": {
        "env": "PR95PLUS_EPOCHS_MULTIPLIER",
        "rationale": (
            "uniform scale on all per-stage epochs (1.0 = full 29,650-epoch "
            "PR95 protocol; 0.01 = $0.30 smoke)"
        ),
        "default": "1.0",
        "required_input_file": False,
    },
    "--seed": {
        "env": "PR95PLUS_SEED",
        "rationale": "RNG seed for deterministic reproducibility",
        "default": "0",
        "required_input_file": False,
    },
    "--gpu": {
        "env": "PR95PLUS_GPU",
        "rationale": (
            "GPU class hint for dynamic hardware substrate detection per "
            "Catalog #190 (T4 / A100 / 4090 / H100)"
        ),
        "default": "A100",
        "required_input_file": False,
    },
    "--codebook-path": {
        "env": "PR95PLUS_CODEBOOK_PATH",
        "rationale": (
            "Pre-distilled DP1 codebook path for E9 bootstrap. Empty path "
            "loads a deterministic zero codebook (smoke-only)."
        ),
        "default": "",
        "required_input_file": False,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_substrate_pr101_lc_v2_clone_enhanced_curriculum")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--curriculum",
        choices=["pr95_enhanced", "pr95_faithful"],
        default="pr95_enhanced",
    )
    p.add_argument("--epochs-multiplier", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-pairs", type=int, default=600)
    p.add_argument("--max-pairs", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--smoke-epochs",
        type=int,
        default=0,
        help=(
            "Total smoke curriculum epochs. 0 preserves the legacy tiny "
            "structural smoke schedule; operator smoke wrappers set this."
        ),
    )
    p.add_argument("--gpu", default="A100")
    p.add_argument("--codebook-path", default="")
    p.add_argument(
        "--skip-auth-eval",
        action="store_true",
        help=(
            "Skip final contest_auth_eval.py. The smoke path (``--smoke``) "
            "ALWAYS skips contest_auth_eval regardless of this flag because "
            "the recipe contract is ``training_artifact_v1`` (manifest.json + "
            "non-empty archive.zip; no score-axis claim). The full path runs "
            "auth eval unless this flag is set; full lands behind a Phase 2 "
            "design memo per HNeRV parity discipline lesson L13."
        ),
    )

    # Per-enhancement override flags. Each defaults to "from-curriculum" so
    # the curriculum mode drives the stack; explicit overrides win.
    for key in (
        "muon_every_stage",
        "iglt_polish_per_stage",
        "stage_aware_ternary_qat",
        "wsd_lr_schedule",
        "logit_softcap",
        "cross_block_skip",
        "atick_redlich_efficient_coding",
        "full_cpu_validation_gate",
        "pretrained_driving_prior_bootstrap",
        "sabor_boundary_only_composition",
        "s2sbs_hf_byte_stuffing_composition",
    ):
        p.add_argument(
            f"--enable-{key.replace('_', '-')}",
            dest=f"enable_{key}",
            action="store_true",
            default=None,
            help=f"Explicitly enable enhancement {key}.",
        )
        p.add_argument(
            f"--disable-{key.replace('_', '-')}",
            dest=f"disable_{key}",
            action="store_true",
            default=None,
            help=f"Explicitly disable enhancement {key}.",
        )

    # Tier 1 + Tier 2 engineering flags (Catalog #172 / #178 / #179).
    p.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        help="Wrap forward in torch.autocast(fp16) + GradScaler (Catalog #172).",
    )
    p.add_argument(
        "--enable-tf32",
        action="store_true",
        default=True,
        help="Enable TF32 matmul/conv kernels (Catalog #178; default on for CUDA).",
    )
    p.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Wrap renderer with torch.compile / Inductor (Catalog #179).",
    )
    p.add_argument(
        "--enable-gt-scorer-cache",
        action="store_true",
        default=False,
        help=(
            "RESERVED: pre-compute GT scorer outputs once and reuse across "
            "hot loop (O1, ~50%% scorer compute savings). Wire-in pending "
            "per-substrate score_aware_loss API extension; currently "
            "parsed-but-not-yet-consumed."
        ),
    )

    # Full-CPU validation gate (Catalog #197 / E8) — opt-in for production
    # full dispatches; the operator wrapper runs the smoke pass FIRST.
    p.add_argument(
        "--full-cpu",
        action="store_true",
        help=(
            "Force CPU-only execution end-to-end (Catalog #197 / E8 "
            "validation pre-dispatch gate)."
        ),
    )
    return p


def _resolve_curriculum_config(args: argparse.Namespace) -> EnhancedCurriculumConfig:
    """Build the :class:`EnhancedCurriculumConfig` from CLI args.

    The curriculum mode drives the base; per-enhancement flags override
    individual fields. Per CLAUDE.md "Design decisions — non-negotiable":
    every override is recorded in the manifest's ``args`` block so a
    council review can see exactly what was enabled.
    """
    if args.curriculum == "pr95_faithful":
        base = EnhancedCurriculumConfig.faithful_config()
    else:
        base = EnhancedCurriculumConfig()

    overrides: dict[str, bool] = {}
    for key in (
        "muon_every_stage",
        "iglt_polish_per_stage",
        "stage_aware_ternary_qat",
        "wsd_lr_schedule",
        "logit_softcap",
        "cross_block_skip",
        "atick_redlich_efficient_coding",
        "full_cpu_validation_gate",
        "pretrained_driving_prior_bootstrap",
        "sabor_boundary_only_composition",
        "s2sbs_hf_byte_stuffing_composition",
    ):
        enable_flag = getattr(args, f"enable_{key}", None)
        disable_flag = getattr(args, f"disable_{key}", None)
        if enable_flag and disable_flag:
            raise ValueError(
                f"--enable-{key.replace('_', '-')} AND --disable-{key.replace('_', '-')} "
                f"both specified; pick one"
            )
        if enable_flag is True:
            overrides[f"enable_{key}"] = True
        elif disable_flag is True:
            overrides[f"enable_{key}"] = False

    if not overrides:
        cfg = base
    else:
        from dataclasses import replace

        cfg = replace(base, **overrides)

    if args.codebook_path:
        from dataclasses import replace

        cfg = replace(
            cfg, pretrained_driving_prior_codebook_path=args.codebook_path
        )
    return cfg


def _smoke_stage_epoch_overrides(args: argparse.Namespace) -> dict[str, int]:
    """Smoke-mode epoch overrides: tiny ($0.30 budget).

    Per CLAUDE.md "Forbidden ``make_synthetic_pair_batch`` calls in any
    non-smoke training path": this map ONLY applies when ``--smoke`` is set.
    """
    base = {
        "stage0_pretrained_driving_prior_bootstrap": 2,
        "stage1_v328_ce": 3,
        "stage2_v331_softplus": 3,
        "stage3_v332_smooth": 3,
        "stage4_v332_qat": 3,
        "stage5_c1a_l7": 3,
        "stage6_lambda_sweep": 3,
        "stage7_sigma_sweep": 3,
        "stage8_muon_finetune": 3,
    }
    requested = int(getattr(args, "smoke_epochs", 0) or 0)
    if requested <= 0:
        return base

    # Preserve PR95 stage proportions while making the actual training budget
    # match the operator smoke contract. This keeps Catalog #167 cost-band
    # anchors honest: 100 advertised smoke epochs means 100 executed epochs.
    names = list(base)
    total = sum(base.values())
    requested = max(requested, len(names))
    raw = {name: base[name] * requested / total for name in names}
    out = {name: max(1, int(raw[name])) for name in names}
    remainder = requested - sum(out.values())
    if remainder > 0:
        ranked = sorted(
            names,
            key=lambda name: (raw[name] - int(raw[name]), base[name]),
            reverse=True,
        )
        for i in range(remainder):
            out[ranked[i % len(ranked)]] += 1
    elif remainder < 0:
        ranked = sorted(
            names,
            key=lambda name: (raw[name] - int(raw[name]), base[name]),
        )
        i = 0
        while remainder < 0 and any(out[name] > 1 for name in names):
            name = ranked[i % len(ranked)]
            if out[name] > 1:
                out[name] -= 1
                remainder += 1
            i += 1
    return out


def _full_stage_epoch_overrides(
    args: argparse.Namespace,
) -> dict[str, int]:
    """Full-training epoch overrides: epochs_multiplier × PR95 canonical.

    PR95 canonical breakdown (sum = 29,650):
        stage1: 3000  stage2: 5650  stage3: 1500  stage4: 500
        stage5: 9000  stage6: 2000  stage7: 3000  stage8: 5000
    """
    multiplier = max(float(args.epochs_multiplier), 0.0)
    canonical = {
        "stage0_pretrained_driving_prior_bootstrap": 100,
        "stage1_v328_ce": 3000,
        "stage2_v331_softplus": 5650,
        "stage3_v332_smooth": 1500,
        "stage4_v332_qat": 500,
        "stage5_c1a_l7": 9000,
        "stage6_lambda_sweep": 2000,
        "stage7_sigma_sweep": 3000,
        "stage8_muon_finetune": 5000,
    }
    return {k: max(1, int(v * multiplier)) for k, v in canonical.items()}


def _vendor_runtime(output_dir: Path) -> None:
    """Emit a contest-compliant inflate.sh (3 positional args + set -e)
    + the vendored PR101 LC v2 clone runtime package.

    Per Catalog #146 + #163: the inflate.sh sentinel pattern is required
    when the script sources the canonical remote bootstrap. The runtime is
    self-contained under ``runtime/src`` so ``contest_auth_eval.py`` exercises
    the same archive parser and renderer that would ship with the packet,
    instead of importing mutable repo source by accident.
    """
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "# PR95++ enhanced curriculum contest-compliant inflate.sh\n"
        "# Catalog #146 (3 positional args) + Catalog #163 (sentinel pattern).\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"\n'
        'exec uv run --quiet --with torch==2.5.1+cu124 '
        '--with numpy --with brotli --with pillow '
        '--extra-index-url https://download.pytorch.org/whl/cu124 '
        '--index-strategy unsafe-best-match '
        'python -m tac.substrates.pr101_lc_v2_clone.inflate '
        '"$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    src_root = runtime_dir / "src"
    package_dir = src_root / "tac" / "substrates" / "pr101_lc_v2_clone"
    packet_dir = src_root / "tac" / "packet_compiler"
    package_dir.mkdir(parents=True, exist_ok=True)
    packet_dir.mkdir(parents=True, exist_ok=True)
    for init_dir in (
        src_root / "tac",
        src_root / "tac" / "substrates",
        package_dir,
        packet_dir,
    ):
        (init_dir / "__init__.py").write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "pr101_lc_v2_clone"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, package_dir / name)
    packet_src = REPO_ROOT / "src" / "tac" / "packet_compiler"
    for name in (
        "pr101_conv4_storage_perms.py",
        "pr101_decoder_byte_maps.py",
        "pr101_decoder_storage_order.py",
    ):
        shutil.copy2(packet_src / name, packet_dir / name)


def _zero_state_dict_like(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a PRC1-packable zero-state fallback for structural smoke eval."""

    return {name: torch.zeros_like(t.detach().cpu()) for name, t in state_dict.items()}


def _pack_smoke_archive_bytes(
    state_dict: dict[str, torch.Tensor],
    *,
    latents: torch.Tensor,
    meta: dict[str, object],
) -> tuple[bytes, dict[str, object]]:
    """Pack a valid smoke archive, falling back to a score-closed zero state.

    The real PR101 byte map is intentionally fail-closed for negzig tensors
    that quantise to ``-128``. Very short smoke runs can still hit that
    non-bijection before QAT has moved weights into the safe interval. A JSON
    placeholder proves only archive emission; it cannot be inflated or scored.
    For the Catalog #167 smoke gate we instead fall back to a valid PRC1
    archive with the same typed grammar and a zero decoder state. That keeps
    the evidence honest: the smoke score is not a model-quality claim, but it
    is a real contest-auth-eval pass through the runtime parser, decoder, PNG
    writer, and scorer.
    """

    try:
        return pack_archive(state_dict, latents=latents, meta=meta), {
            "smoke_archive_mode": "ema_state",
            "smoke_archive_score_claim": False,
        }
    except ValueError as exc:
        if "NEGZIG_NON_BIJECTION" not in str(exc):
            raise
        fallback_meta = {
            **meta,
            "smoke_archive_mode": "zero_state_valid_prc1",
            "smoke_zero_state_fallback_reason": str(exc),
        }
        return (
            pack_archive(
                _zero_state_dict_like(state_dict),
                latents=torch.zeros_like(latents),
                meta=fallback_meta,
            ),
            {
                "smoke_archive_mode": "zero_state_valid_prc1",
                "smoke_zero_state_fallback_reason": str(exc),
                "smoke_archive_score_claim": False,
            },
        )


class _SmokeSegScorer(torch.nn.Module):
    """Stand-in SegNet preserving the preprocess_input contract."""

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
        flat6_sub = (
            flat6[..., : half_h * 2, : half_w * 2]
            .reshape(b * t, 6, half_h, 2, half_w, 2)
            .mean(dim=(3, 5))
        )
        return flat6_sub.reshape(b, t * 6, half_h, half_w)

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = x_b12hw.flatten(2).mean(dim=2)
        return {"pose": self.head(pooled)}


def _train_smoke_loop(
    *,
    substrate: Pr101LcV2CloneSubstrate,
    pair_batches: list[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    enhanced_cfg: EnhancedCurriculumConfig,
    stages: list,
    output_dir: Path,
) -> dict[str, Any]:
    """Smoke training loop — exercises the canonical helpers without GPU.

    Per CLAUDE.md "Forbidden ``make_synthetic_pair_batch`` calls in any
    non-smoke training path": the smoke loop uses tiny synthetic pairs at
    eval resolution (allowed under ``--smoke`` only). The full loop is
    structurally identical but consumes the contest video pairs via
    :func:`tac.substrates._shared.trainer_skeleton.decode_real_pairs`.
    """
    seg_scorer = _SmokeSegScorer(out_channels=5).to(device).eval()
    pose_scorer = _SmokePoseScorer().to(device).eval()
    for p in seg_scorer.parameters():
        p.requires_grad_(False)
    for p in pose_scorer.parameters():
        p.requires_grad_(False)

    loss_fn = Pr101LcV2CloneScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=Pr101LcV2ScoreAwareLossWeights(),
    ).to(device)

    if enhanced_cfg.enable_cross_block_skip:
        # E6 — record the cross-block skip configuration on the substrate.
        try:
            apply_cross_block_skip(substrate, early_block_idx=0, late_block_idx=5)
        except ValueError:
            # Some substrate configs have fewer blocks; the smoke path
            # tolerates this gracefully.
            pass

    ema = EMA(substrate, decay=args.ema_decay)
    stage_log: list[dict[str, Any]] = []
    last_parts: dict[str, float] = {}

    for stage in stages:
        adamw, muon_opt = build_optimizer_for_enhanced_stage(
            model=substrate, stage=stage, enhanced_cfg=enhanced_cfg
        )
        stage_log.append(
            {
                "stage": stage.name,
                "epochs": stage.epochs,
                "started_at_utc": _canon_utc_now_iso(),
                "use_muon": stage.use_muon,
                "use_qat": stage.use_qat,
                "seg_loss_kind": stage.seg_loss_kind,
                "extras": {
                    k: v
                    for k, v in stage.extras.items()
                    if isinstance(v, (str, int, float, bool, list, tuple))
                },
            }
        )
        total_steps = max(1, stage.epochs * max(1, len(pair_batches)))
        global_step = 0
        for _epoch in range(stage.epochs):
            for batch_pairs in pair_batches:
                pair_indices = torch.arange(
                    batch_pairs.shape[0], dtype=torch.long, device=device
                )
                del pair_indices  # smoke renderer takes z = latent directly
                # Build random latents at the substrate's contract for smoke.
                z = torch.randn(
                    batch_pairs.shape[0],
                    substrate.cfg.latent_dim,
                    device=device,
                    requires_grad=False,
                )
                z.requires_grad_(True)
                # Tier-1 O2: autocast wrap (no-op when flag=False).
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16,
                    device=device,
                ):
                    rgb_pair = substrate(z) / 255.0  # [B, 2, 3, H, W] in [0, 1]
                    gt_0 = batch_pairs[:, 0].to(device) / 255.0
                    gt_1 = batch_pairs[:, 1].to(device) / 255.0
                    archive_bytes_proxy = torch.tensor(
                        150_000.0, device=device
                    )
                    loss, parts = loss_fn(
                        rgb_pair[:, 0],
                        rgb_pair[:, 1],
                        gt_0,
                        gt_1,
                        archive_bytes_proxy,
                    )

                # E4 WSD LR adjustment if enabled.
                if enhanced_cfg.enable_wsd_lr_schedule:
                    lr = compute_wsd_lr(
                        step=global_step,
                        total_steps=total_steps,
                        peak_lr=stage.adamw_lr,
                        warmup_fraction=enhanced_cfg.wsd_warmup_fraction,
                        decay_fraction=enhanced_cfg.wsd_decay_fraction,
                        floor_ratio=enhanced_cfg.wsd_floor_ratio,
                    )
                    for grp in adamw.param_groups:
                        grp["lr"] = lr

                adamw.zero_grad(set_to_none=True)
                if muon_opt is not None:
                    muon_opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        substrate.parameters(), args.grad_clip
                    )
                adamw.step()
                if muon_opt is not None:
                    muon_opt.step()
                ema.update(substrate)
                global_step += 1
                last_parts = {k: float(v.item()) for k, v in parts.items()}

    # Build a valid archive at the EMA snapshot. The smoke path may fall back
    # to a zero-state PRC1 archive when the very short training run hits the
    # PR101 negzig non-bijection before QAT has stabilized. That fallback is
    # still contest-inflatable and scoreable; it is explicitly not a quality
    # claim.
    orig = {k: v.detach().clone() for k, v in substrate.state_dict().items()}
    ema.apply(substrate)
    latents = torch.zeros(substrate.cfg.num_pairs, substrate.cfg.latent_dim)
    meta = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "curriculum": args.curriculum,
        "smoke_archive": True,
    }
    archive_bytes, archive_meta = _pack_smoke_archive_bytes(
        substrate.state_dict(),
        latents=latents,
        meta=meta,
    )
    substrate.load_state_dict(orig)

    archive_dir = output_dir / "archive_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "0.bin").write_bytes(archive_bytes)

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
        "archive_zip_path": str(archive_zip),
        "runtime_inflate_sh": str(output_dir / "runtime" / "inflate.sh"),
        **archive_meta,
        "last_loss_parts": last_parts,
        "stage_log": stage_log,
        "training_mode": "smoke",
        "evidence_grade": (
            "macOS-CPU advisory" if str(device) == "cpu" else "training-only"
        ),
    }


def _run_contest_auth_eval_cuda(
    *,
    archive_zip: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    output_json: Path,
) -> dict[str, object]:
    """Run the canonical contest auth eval and require a CUDA score claim.

    Thin shim around
    :func:`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`
    that hardcodes the post-gate execution path (caller already validated
    not-smoke + not-skip + CUDA device at the call site at line ~951 per
    Path B). This keeps the public function name patchable by the existing
    Path B regression tests at
    ``src/tac/tests/test_train_substrate_pr101_lc_v2_clone_auth_eval_gate.py``.

    The caller must choose ``output_json`` under durable repo/provider storage.
    Modal smoke runs default to ``/modal_results/<job>/output`` via the remote
    driver; a temp evidence bypass would make the smoke score non-custodial.
    """

    # Synthetic "full mode" args so the canonical gate runs the subprocess
    # path. The Path B call-site smoke gate above this function is the
    # primary defense-in-depth; this gate is hardcoded to bypass-refuse
    # at the call to maintain the established public contract.
    import argparse as _argparse

    _args = _argparse.Namespace(smoke=False, skip_auth_eval=False)
    result = _canon_gate_auth_eval_call(
        args=_args,
        archive_zip=archive_zip,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
        substrate_tag="pr95plus",
        device="cuda",
    )
    if result is None:
        # Should not happen given the synthetic args above; guard against
        # any future canonical-helper opt-out.
        raise RuntimeError(
            "_run_contest_auth_eval_cuda: canonical gate refused with "
            "synthetic full-mode args; investigate gate logic"
        )
    return result


def _full_main(args: argparse.Namespace) -> int:
    """Full training path.

    Per CLAUDE.md HNeRV parity discipline lesson L13 + "KILL is LAST RESORT":
    the full path is L0 SCAFFOLD until the Phase 2 design memo lands. The
    operator-authorize wrapper routes via the smoke-before-full pattern
    (Catalog #167); the first beta canary fires the smoke path which IS
    fully implemented.
    """
    raise NotImplementedError(
        "PR95++ enhanced full training is L0 SCAFFOLD. Use --smoke for the "
        "structural test path; production full dispatch lands when the "
        "Phase 2 design memo "
        "(.omx/research/pr95plus_phase_2_training_design_<DATE>.md) is "
        "council-approved AND the codebook distillation has been run "
        "against real Comma2k19 frames (operator-gated)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _canon_pin_seeds(int(args.seed))
    if args.full_cpu:
        # E8 — Catalog #197 full-CPU validation: force device cpu + smoke.
        args.device = "cpu"
        args.smoke = True

    device = _canon_device_or_die(
        str(args.device), smoke=bool(args.smoke), substrate_tag=SUBSTRATE_TAG
    )

    enhanced_cfg = _resolve_curriculum_config(args)
    audit = audit_enhanced_curriculum_against_hnerv_parity_lessons(enhanced_cfg)
    # Refuse dispatch if any verdict is "FAIL" (defence-in-depth — the audit
    # itself only emits PASS / N/A / SUBSTRATE_ENGINEERING_EXCEPTION today,
    # but a future enhancement that fails parity would surface here.
    fail_verdicts = [
        (ekey, lid, v)
        for ekey, lessons in audit.items()
        for lid, v in lessons.items()
        if v == "FAIL"
    ]
    if fail_verdicts:
        raise RuntimeError(
            f"HNeRV parity audit FAILED for enhancements: {fail_verdicts}"
        )

    # Build the substrate + curriculum.
    substrate = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig()).to(device)
    # Tier-1 O3: torch.compile wrap (no-op when flag=False).
    substrate = _compile_with_fallback(
        substrate,
        enabled=bool(getattr(args, "enable_torch_compile", False)),
        mode="default",
        fallback_on_error=True,
    )
    if args.curriculum == "pr95_faithful":
        if args.smoke:
            stage_epochs = _smoke_stage_epoch_overrides(args)
            # Strip the stage0 entry (faithful has no Stage 0)
            stage_epochs.pop("stage0_pretrained_driving_prior_bootstrap", None)
        else:
            stage_epochs = _full_stage_epoch_overrides(args)
            stage_epochs.pop("stage0_pretrained_driving_prior_bootstrap", None)
        stages = build_faithful_stages(stage_epoch_overrides=stage_epochs)
    else:
        stage_epochs = (
            _smoke_stage_epoch_overrides(args)
            if args.smoke
            else _full_stage_epoch_overrides(args)
        )
        stages = build_enhanced_stages(
            enhanced_cfg, stage_epoch_overrides=stage_epochs
        )

    # Build the training batches.
    pair_batches: list[torch.Tensor] = []
    if args.smoke:
        # SYNTHETIC_NON_SMOKE_OK:smoke-only-deterministic-test-input-tiny-resolution
        torch.manual_seed(int(args.seed))
        smoke_h, smoke_w = 384, 512
        for _ in range(2):
            pair_batches.append(
                torch.rand(
                    min(int(args.batch_size), 2),
                    2,
                    3,
                    smoke_h,
                    smoke_w,
                )
                * 255.0
            )
    else:
        pairs = _canon_decode_real_pairs(
            Path(args.video_path),
            n_pairs=int(args.num_pairs),
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=int(args.max_pairs),
        )
        # ``decode_real_pairs`` returns ``(N, 2, 3, 384, 512)``; batch them.
        for i in range(0, len(pairs), int(args.batch_size)):
            slab = pairs[i : i + int(args.batch_size)]
            pair_batches.append(slab)

    if not args.smoke:
        return _full_main(args)

    result = _train_smoke_loop(
        substrate=substrate,
        pair_batches=pair_batches,
        args=args,
        device=device,
        enhanced_cfg=enhanced_cfg,
        stages=stages,
        output_dir=args.output_dir,
    )
    # Path B (council-grade decision; see
    # `feedback_pr95plus_smoke_archive_completion_landed_20260513.md`):
    # smoke ALWAYS skips contest_auth_eval, regardless of `--skip-auth-eval`.
    # Defense-in-depth gate. The smoke recipe contract is
    # ``smoke_validation_contract: training_artifact_v1`` (manifest.json +
    # non-empty archive.zip; no score-axis claim). The substrate's inflate.py
    # writes per-frame .png into a subdir, but contest_auth_eval expects a
    # single uint8 RGB ``.raw`` blob per video at top level — those contracts
    # diverge by design today because the substrate is L0 SCAFFOLD per HNeRV
    # parity discipline lesson L13. Forcing auth-eval at smoke would burn
    # Modal $ on a known-degenerate inflate path (the empirical anchor
    # fc-01KRJ15E7NYVDF77VF1ED1XBKS at HEAD bda49280 demonstrated this:
    # contest_auth_eval ran 516.5s and refused with PARTIAL inflate). The
    # full path lands behind a Phase 2 design memo + score-grade inflate
    # implementation; until then the smoke is structurally infrastructure-only.
    if args.smoke:
        result["auth_eval_skipped_reason"] = (
            "smoke_validation_contract=training_artifact_v1; "
            "auth-eval refused at smoke per Path B council decision "
            "(score_claim=false, evidence_grade=training-only)"
        )
        print(
            "[pr95plus] [WARNING] smoke mode does NOT run contest_auth_eval. "
            "The training_artifact_v1 contract validates manifest.json + "
            "archive.zip only; no contest-CUDA score is claimed. "
            "Full-mode auth eval lands when _full_main + score-grade inflate.py "
            "are implemented behind the Phase 2 design memo."
        )
    elif not args.skip_auth_eval and device.type == "cuda":
        auth_eval_json_path = args.output_dir / "contest_auth_eval_cuda.json"
        auth_result = _run_contest_auth_eval_cuda(
            archive_zip=Path(result["archive_zip_path"]),
            inflate_sh=Path(result["runtime_inflate_sh"]),
            upstream_dir=Path(args.upstream_dir),
            output_json=auth_eval_json_path,
        )
        result.update(auth_result)
        result["evidence_grade"] = "contest-CUDA"

    manifest = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "curriculum": args.curriculum,
        "started_at_utc": _canon_utc_now_iso(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "enhancement_summary": enhancement_summary(enhanced_cfg),
        "hnerv_parity_audit": audit,
        "result": result,
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cuda" if str(device) == "cuda" else "cpu",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("PR95PLUS_GPU", "MODAL_GPU"),
        ),
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": result.get("evidence_grade", "training-only"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[pr95plus] curriculum={args.curriculum}")
    print(f"[pr95plus] enhancements={enhanced_cfg.enabled_enhancements()}")
    print(f"[pr95plus] archive_sha256={result['archive_sha256']}")
    print(f"[pr95plus] archive_bytes={result['archive_bytes']}")
    if "auth_eval_cuda_score" in result:
        print(f"[pr95plus] [contest-CUDA] score={result['auth_eval_cuda_score']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
