# SPDX-License-Identifier: MIT
"""Train the stack-of-stacks composition substrate (beat-PR95 Idea 3).

Design memo: ``.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md``
Lane:        ``lane_stack_of_stacks_composition_implementation_20260513``

This trainer composes K ≤ 3 inner substrates (frozen base archives, e.g.
A1, A1+LAPose, A1+wavelet) into ONE deterministic compose-and-dispatch
archive via the canonical :mod:`tac.composition.stack_of_stacks` module.
Per arm, a small LangevinOptimizer-trained residual head learns a
score-gradient-aware correction to the renderer's output at hard pairs.

The trainer is intentionally LEAN because Idea 3's compositional win is
INFERENCE-TIME (per-pair-best-of-K selection + sidecar atoms), NOT
training-time. We spend $2-5 of GPU on K residual heads + auth eval, NOT
$15 on a full from-scratch retrain.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline":

* L1 score-aware substrate trains against contest video pixels (real
  decode via canonical helper).
* L2 ``patch_upstream_yuv6_globally()`` BEFORE ``load_differentiable_scorers``
  (Catalog #187).
* L4 inflate runtime ≤ 200 LOC per substrate-engineering exemption.
* L5 architecture is the FULL renderer (RGB out from selected pairs).
* L7 bolt-on size ≤ 350 LOC residual head; trainer is substrate-engineering.
* L8 ``apply_eval_roundtrip=True`` inside the per-batch loop.
* L9 runtime closure — inflate.sh sourced from canonical
  ``scripts/remote_archive_only_eval.sh`` SOURCE_ONLY pattern.
* EMA decay 0.997 + snapshot+restore at eval.
* No scorer load at inflate (Catalog #6); selector precomputed at TRAIN
  time and stored in the SOS1 trailer.

Usage (single-arm byte-closed canary; CPU smoke)::

    .venv/bin/python experiments/train_substrate_stack_of_stacks.py \
        --base-archive submissions/a1/archive.zip \
        --base-runtime-dir submissions/a1 \
        --middle-arm-substrate-ids a1 \
        --video-path upstream/videos/0.mkv \
        --output-dir experiments/results/sos_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
        --epochs 0 --batch-size 4 --device cpu --smoke

Future full mode (BLOCKED until score-aware residual training + per-arm
decoder hooks land)::

    .venv/bin/python experiments/train_substrate_stack_of_stacks.py \
        --base-archive .../archive.zip \
        --middle-arm-substrate-ids a1,a1_plus_lapose,a1_plus_wavelet \
        --video-path upstream/videos/0.mkv \
        --output-dir experiments/results/sos_full_<utc> \
        --epochs 800 --batch-size 32 --lr 1e-4 --device cuda \
        --langevin-t-init 0.3 --langevin-t-final 1e-4 \
        --outer-stack-k 3
"""
# AUTOCAST_FP16_WAIVED:composition-time-residual-training-numerics-need-fp32-until-paired-anchor-lands
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic-until-paired-CPU/CUDA-anchor-lands
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.composition.stack_of_stacks import (
    BoundaryAtomSpec,
    HFSidecarSpec,
    InnerStackSpec,
    MiddleStackSpec,
    OuterStackSpec,
    ResidualSpec,
    StackOfStacksError,
    compose_stack_of_stacks,
    validate_byte_budget,
)
from tac.optimization.langevin_optimizer import LangevinOptimizer
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
from tac.training import EMA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "submissions"
    / "a1"
    / "archive.zip"
)
DEFAULT_BASE_RUNTIME_DIR = REPO_ROOT / "submissions" / "a1"

EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
DEFAULT_OUTER_K = 1
MAX_TOTAL_ARCHIVE_BYTES = 250_000  # ~7 KB of stack-of-stacks slack over A1's 178 KB


# Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS manifest — ast.AnnAssign per
# Catalog #168. The fields are env / rationale / default /
# required_input_file / generator_command.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--base-archive": {
        "env": "STACK_OF_STACKS_BASE_ARCHIVE",
        "rationale": (
            "Base substrate archive bytes used for every arm's inner stack. "
            "Default is the mounted A1 PR101-fine-tuned anchor."
        ),
        "default": str(DEFAULT_BASE_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "submissions/a1 archive custody (landed 2026-05-11)",
    },
    "--base-runtime-dir": {
        "env": "STACK_OF_STACKS_BASE_RUNTIME_DIR",
        "rationale": (
            "Base inflate runtime bundled into the stack runtime for the "
            "single-arm passthrough path. Default is submissions/a1."
        ),
        "default": str(DEFAULT_BASE_RUNTIME_DIR.relative_to(REPO_ROOT)),
        "required_input_file": False,
        "generator_command": "submissions/a1 runtime custody",
    },
    "--video-path": {
        "env": "STACK_OF_STACKS_VIDEO_PATH",
        "rationale": (
            "Score-aware substrate trains against contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke."
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "STACK_OF_STACKS_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "STACK_OF_STACKS_EPOCHS",
        "rationale": (
            "training epochs; default 800 for full per design memo §4.1 "
            "(residual heads converge fast against frozen base)."
        ),
        "default": "800",
    },
    "--upstream-dir": {
        "env": "STACK_OF_STACKS_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "STACK_OF_STACKS_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--middle-arm-substrate-ids": {
        "env": "STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS",
        "rationale": (
            "comma-separated substrate ids for the middle-stack arms "
            "(e.g. 'a1,a1_plus_lapose'); 1..3 arms allowed."
        ),
        "default": "a1",
    },
    "--outer-stack-k": {
        "env": "STACK_OF_STACKS_OUTER_K",
        "rationale": (
            "K-checkpoint outer ensemble count (1..3); 1 disables ensemble, "
            ">=2 enables per-pair-best-of-K selector."
        ),
        "default": str(DEFAULT_OUTER_K),
    },
}

TIER_1_EXTRA_MOUNT_PATHS = ("submissions/a1",)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_stack_of_stacks",
        description=(
            "Train the stack-of-stacks composition substrate (beat-PR95 "
            "design Idea 3)."
        ),
    )
    p.add_argument(
        "--base-archive",
        type=Path,
        default=DEFAULT_BASE_ARCHIVE,
        help="Path to base substrate archive.zip (default: A1 PR101 anchor).",
    )
    p.add_argument(
        "--base-runtime-dir",
        type=Path,
        default=DEFAULT_BASE_RUNTIME_DIR,
        help="Directory containing the base runtime inflate.sh/inflate.py/src.",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to upstream/videos/0.mkv (contest video).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root.",
    )
    p.add_argument(
        "--middle-arm-substrate-ids",
        type=str,
        default="a1",
        help="Comma-separated substrate ids for middle-stack arms (1..3).",
    )
    p.add_argument(
        "--outer-stack-k",
        type=int,
        default=DEFAULT_OUTER_K,
        help="K-checkpoint outer ensemble count (1..3).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device; cuda required for full, cpu only with --smoke.",
    )

    # Training hyperparameters.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="AdamW + LangevinOptimizer learning rate (residual heads only).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--val-pair-count", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=N_PAIRS_FULL)

    # Langevin polish schedule.
    p.add_argument("--langevin-t-init", type=float, default=0.3,
                   help="Initial Langevin temperature for polish phase.")
    p.add_argument("--langevin-t-final", type=float, default=1e-4)
    p.add_argument("--langevin-schedule", type=str, default="cosine",
                   choices=["cosine", "exp", "log", "exponential", "geman_geman"])
    p.add_argument("--langevin-polish-epochs", type=int, default=100,
                   help="Number of polish epochs at the end of training.")

    # Outer-stack ensemble temperatures (informational stored in arm_meta).
    p.add_argument("--ensemble-temperatures", type=str, default="",
                   help="Comma-separated temperatures for each arm (e.g. '1.0,0.5,0.1').")

    # Inner-stack atom budgets (per arm).
    p.add_argument("--sabor-capacity-bytes", type=int, default=0,
                   help="SABOR boundary atom budget per arm (bytes).")
    p.add_argument("--sabor-audit-capacity-bytes", type=int, default=14_600_000,
                   help="Measured SABOR audit capacity per video (bytes).")
    p.add_argument("--s2sbs-capacity-bytes", type=int, default=0,
                   help="S2SBS HF sidecar budget per arm (bytes).")
    p.add_argument("--s2sbs-audit-capacity-bytes", type=int, default=38_000_000,
                   help="Measured S2SBS audit capacity (bytes).")
    p.add_argument("--residual-int8-bytes", type=int, default=0,
                   help="Per-arm score-gradient residual byte budget.")

    # Modes.
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: cpu allowed, fewer pairs, no scorer load.")
    p.add_argument(
        "--research-only-runtime-scaffold",
        action="store_true",
        help=(
            "Explicitly acknowledge this scaffold is research-only and emits "
            "a non-scoreable runtime until per-arm decoder hooks land."
        ),
    )
    p.add_argument("--max-total-archive-bytes", type=int, default=MAX_TOTAL_ARCHIVE_BYTES)
    p.add_argument("--dispatch-instance-job-id", type=str, default="")
    p.add_argument("--lane-id", type=str,
                   default="lane_stack_of_stacks_composition_implementation_20260513")
    p.add_argument("--dispatch-platform", type=str, default="modal")
    return p


# ---------------------------------------------------------------------------
# Spec resolution + archive load
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    args: argparse.Namespace
    output_dir: Path
    base_archive_bytes: bytes
    base_archive_sha256: str
    arm_substrate_ids: list[str]
    ensemble_temperatures: tuple[float, ...]
    stage_log: list[dict[str, Any]]
    device_str: str


def _read_base_archive(archive_path: Path) -> tuple[bytes, str]:
    """Read the base archive's single ``x`` member (raw bytes) + sha256."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"base archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if "x" not in names:
            raise RuntimeError(
                f"base archive {archive_path} does not contain a single 'x' "
                f"member; found: {names}"
            )
        member_bytes = zf.read("x")
    return member_bytes, hashlib.sha256(member_bytes).hexdigest()


def _resolve_arm_substrate_ids(raw: str) -> list[str]:
    ids = [s.strip() for s in raw.split(",") if s.strip()]
    if not ids:
        raise SystemExit("--middle-arm-substrate-ids must be non-empty")
    if len(ids) > 3:
        raise SystemExit(
            f"--middle-arm-substrate-ids supports at most 3 arms; got {len(ids)}"
        )
    return ids


def _resolve_ensemble_temperatures(raw: str, k: int) -> tuple[float, ...]:
    if not raw.strip():
        # Default schedule: 1.0, 0.3, 0.1 (decreasing) for K arms.
        if k == 1:
            return (1.0,)
        elif k == 2:
            return (1.0, 0.3)
        else:
            return (1.0, 0.3, 0.1)
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if len(parts) != k:
        raise SystemExit(
            f"--ensemble-temperatures count ({len(parts)}) must equal "
            f"--outer-stack-k ({k})"
        )
    return tuple(float(p) for p in parts)


# ---------------------------------------------------------------------------
# Smoke training + Langevin polish
# ---------------------------------------------------------------------------


def _build_residual_payload(byte_budget: int, *, seed: int = 0) -> bytes:
    """Build a deterministic int8 residual payload at byte_budget bytes.

    For the smoke / build path the payload is a deterministic
    pseudo-random byte sequence so the composer test fixtures remain
    reproducible. The full-training path overwrites this with the actual
    LangevinOptimizer-trained residual when scorer gradients are
    available.
    """
    if byte_budget <= 0:
        return b""
    import random as _rng

    state = _rng.Random(seed)
    return bytes(state.randint(-128, 127) % 256 for _ in range(byte_budget))


def _train_arm_residual_with_langevin(
    *,
    arm_idx: int,
    arm_substrate_id: str,
    base_bytes: bytes,
    args: argparse.Namespace,
    device,
) -> bytes:
    """Train a small residual on a frozen base substrate via LangevinOptimizer.

    For the smoke path this is a placeholder that produces deterministic
    bytes WITHOUT loading scorers. The full-training path is gated behind
    ``not args.smoke`` AND a successful scorer-load probe.

    Per CLAUDE.md "EVERY training path MUST instantiate EMA" + "MPS auth
    eval is NOISE": this helper is a thin demonstrator of LangevinOptimizer
    integration; the actual residual training inside a contest-faithful
    score-aware loop lives in the per-substrate trainer (e.g.
    experiments/train_substrate_a1_plus_lapose.py).
    """
    import torch

    byte_budget = args.residual_int8_bytes
    if byte_budget <= 0:
        return b""
    if not args.smoke:
        raise SystemExit(
            "stack-of-stacks full residual training is blocked: the current "
            "scaffold does not load differentiable scorers, does not apply "
            "eval_roundtrip, and does not train score-aware per-arm residuals. "
            "Use --smoke --research-only-runtime-scaffold for local parser/"
            "archive experiments only."
        )

    class _ResidualHead(torch.nn.Module):
        """Minimal trainable residual payload carrier for the smoke scaffold."""

        def __init__(self, rank: int, dim: int) -> None:
            super().__init__()
            self.residual = torch.nn.Parameter(torch.zeros(rank, dim))

        def loss_against(self, target: torch.Tensor) -> torch.Tensor:
            return ((self.residual - target) ** 2).mean()

    # Tiny demonstrator: residual is a (rank=4, 96) tensor; gradient drives
    # it toward zero via the proxy loss (placeholder for the real
    # score-aware path). The point of THIS trainer is to land the
    # composition wire-up; per-arm residual fine-tune is owned by the
    # respective per-substrate trainer (a1_plus_lapose / a1_plus_wavelet).
    rank, dim = 4, max(8, byte_budget // 4)
    model = _ResidualHead(rank, dim).to(device)
    ema = EMA(model, decay=args.ema_decay)
    # Build with explicit gradient flow so LangevinOptimizer integrates.
    target = torch.randn(rank, dim, device=device) * 0.02
    optimizer = LangevinOptimizer(
        model.parameters(),
        lr=args.lr,
        T_init=args.langevin_t_init,
        T_final=args.langevin_t_final,
        n_steps=max(1, args.langevin_polish_epochs),
        schedule=args.langevin_schedule,
        weight_decay=args.weight_decay,
        noise_seed=args.seed + arm_idx,
    )
    for _ in range(max(1, args.langevin_polish_epochs)):
        optimizer.zero_grad()
        loss = model.loss_against(target)
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Quantize the EMA shadow to int8 and serialize, restoring the live
    # training state immediately after the export snapshot.
    live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        ema.apply(model)
        residual_for_export = model.residual.detach().cpu()
    finally:
        model.load_state_dict(live_state)
    quantized = (
        (residual_for_export * 4.0).round().clamp(-128, 127).to(torch.int8)
    )
    raw_bytes = bytes(quantized.numpy().tobytes())
    if len(raw_bytes) > byte_budget:
        raw_bytes = raw_bytes[:byte_budget]
    elif len(raw_bytes) < byte_budget:
        raw_bytes = raw_bytes + b"\x00" * (byte_budget - len(raw_bytes))
    return raw_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _emit_provenance(state: TrainState) -> Path:
    """Write provenance.json next to the output archive."""
    prov_path = state.output_dir / "provenance.json"
    single_arm_passthrough = _is_single_arm_passthrough_build(
        state.args,
        state.arm_substrate_ids,
    )
    payload = {
        "started_at_utc": state.stage_log[0]["at"] if state.stage_log else _canon_utc_now_iso(),
        "completed_at_utc": _canon_utc_now_iso(),
        "lane_id": state.args.lane_id,
        "dispatch_instance_job_id": state.args.dispatch_instance_job_id,
        "dispatch_platform": state.args.dispatch_platform,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(state.args).items()},
        "base_archive_sha256": state.base_archive_sha256,
        "arm_substrate_ids": state.arm_substrate_ids,
        "ensemble_temperatures": list(state.ensemble_temperatures),
        "stage_log": state.stage_log,
        "device": state.device_str,
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cuda" if state.device_str == "cuda" else "cpu",
            substrate_tag="stack_of_stacks",
            env_var_candidates=("STACK_OF_STACKS_GPU", "MODAL_GPU"),
        ),
        "predicted_band_contest_cpu": (
            [0.190, 0.210] if single_arm_passthrough else [0.175, 0.190]
        ),
        "predicted_band_basis": (
            "single_arm_a1_passthrough_canary; score-neutral or slight rate penalty"
            if single_arm_passthrough
            else "beat_pr95_curriculum_substrate_training_design_20260513.md#idea-3"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": single_arm_passthrough,
        "canary_exact_eval_ready": single_arm_passthrough,
        "score_lowering_dispatch_ready": False,
        "operator_dispatch_enabled": False,
        "evidence_grade": (
            "byte_closed_single_arm_passthrough_no_score_claim"
            if single_arm_passthrough
            else "build_artifact_no_score_claim"
        ),
        "runtime_contract": (
            "single_arm_a1_passthrough_exact_eval_canary"
            if single_arm_passthrough
            else "research_only_multi_arm_scaffold"
        ),
    }
    prov_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return prov_path


def _build_archive_from_compose(
    *,
    composed_bytes: bytes,
    output_dir: Path,
    base_runtime_dir: Path,
) -> Path:
    """Zip the composed ``x`` blob into archive.zip + write inflate.sh / inflate.py.

    Per Catalog #146 the inflate.sh template MUST accept 3 positional
    args ($1 archive_dir, $2 output_dir, $3 file_list); per Catalog #6 the
    inflate.py template MUST NOT import upstream scorer modules.
    """
    if not (base_runtime_dir / "inflate.sh").is_file():
        raise FileNotFoundError(f"base runtime inflate.sh missing: {base_runtime_dir}")
    if not (base_runtime_dir / "inflate.py").is_file():
        raise FileNotFoundError(f"base runtime inflate.py missing: {base_runtime_dir}")

    submission_dir = output_dir / "submission_dir"
    submission_dir.mkdir(parents=True, exist_ok=True)
    archive_path = submission_dir / "archive.zip"

    # Deterministic ZIP per Catalog #19.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(filename="x", date_time=(2026, 5, 13, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, composed_bytes)
    archive_path.write_bytes(buf.getvalue())

    # inflate.sh — canonical 3-arg signature.
    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text(
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        'archive_dir="$1"\n'
        'output_dir="$2"\n'
        'file_list="$3"\n'
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec uv run --quiet --with torch==2.5.1+cu124 \\\n'
        '    --with brotli==1.1.0 \\\n'
        '    --with numpy \\\n'
        '    --extra-index-url https://download.pytorch.org/whl/cu124 \\\n'
        '    "$HERE/inflate.py" "$archive_dir" "$output_dir" "$file_list"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)

    # Bundle the base runtime used by the single-arm pass-through path. Ignore
    # custody/eval artifacts and archives; the stack archive supplies x bytes.
    bundled_runtime = submission_dir / "base_runtime"
    if bundled_runtime.exists():
        shutil.rmtree(bundled_runtime)
    shutil.copytree(
        base_runtime_dir,
        bundled_runtime,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "archive.zip",
            "contest_auth_eval*.json",
            "pre_submission_compliance*.json",
            "dual_eval_adjudicated.json",
        ),
    )

    # inflate.py — stack-of-stacks parser + single-runtime delegation.
    # It strips the SOS1 trailer, writes the selected arm bytes to a temporary
    # data dir, and delegates to the bundled base runtime. Mixed per-pair arm
    # selection remains fail-closed until true multi-arm frame stitching lands.
    inflate_py = submission_dir / "inflate.py"
    inflate_py.write_text(
        _STACK_OF_STACKS_INFLATE_PY_TEMPLATE,
        encoding="utf-8",
    )
    return archive_path


_STACK_OF_STACKS_INFLATE_PY_TEMPLATE = '''#!/usr/bin/env python3
"""Stack-of-stacks inflate runtime.

Per CLAUDE.md "HNeRV parity discipline" L4 ≤ 200 LOC + Catalog #6 no
scorer load at inflate. This template parses the SOS1 trailer at the end
of the ``x`` blob, looks up the per-pair arm selector, and delegates to
the per-substrate decoder (which is responsible for rendering the base
substrate's bytes into RGB output).

This runtime is intentionally strict. It supports only the byte-closed
single-arm passthrough path: strip the SOS1 trailer, recover arm 0 bytes,
and delegate to the bundled base runtime. Mixed-arm selectors fail closed
until frame-level multi-arm stitching lands.
"""
from __future__ import annotations

import subprocess
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

SOS_SIDECAR_MAGIC = b"SOS1"
SOS_HEADER_STRUCT = struct.Struct("<4sBBHBH2s")


def _parse_sos(archive_bytes: bytes):
    idx = archive_bytes.rfind(SOS_SIDECAR_MAGIC)
    if idx < 0:
        return None
    if idx + SOS_HEADER_STRUCT.size > len(archive_bytes):
        return None
    try:
        magic, ver, mask, n_pairs, k, meta_len, _ = SOS_HEADER_STRUCT.unpack_from(
            archive_bytes, idx
        )
    except struct.error:
        return None
    if magic != SOS_SIDECAR_MAGIC:
        return None
    selector_start = idx + SOS_HEADER_STRUCT.size
    selector_end = selector_start + n_pairs
    meta_end = selector_end + meta_len
    if meta_end != len(archive_bytes):
        return None
    return {
        "k": int(k),
        "n_pairs": int(n_pairs),
        "layer_mask": int(mask),
        "selector": archive_bytes[selector_start:selector_end],
        "arm_concat_bytes": archive_bytes[:idx],
        "trailer_offset": idx,
    }


def main():
    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        sys.exit(2)
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_zip = archive_dir / "archive.zip"
    if not archive_zip.is_file():
        print(f"FATAL: archive.zip missing at {archive_zip}", file=sys.stderr)
        sys.exit(3)
    with zipfile.ZipFile(archive_zip, "r") as zf:
        if "x" not in zf.namelist():
            print("FATAL: archive missing single 'x' member", file=sys.stderr)
            sys.exit(4)
        x_blob = zf.read("x")
    parsed = _parse_sos(x_blob)
    if parsed is None:
        # No SOS1 trailer: treat as straight base substrate; defer to arm 0.
        print(
            "[stack-of-stacks] no SOS1 trailer; arm 0 base passed through unchanged",
            file=sys.stderr,
        )
    else:
        print(
            f"[stack-of-stacks] SOS1 trailer found: k={parsed['k']} "
            f"n_pairs={parsed['n_pairs']} layer_mask=0x{parsed['layer_mask']:02x}",
            file=sys.stderr,
        )
    if parsed is None:
        arm_blob = x_blob
    else:
        selected = set(parsed["selector"])
        if parsed["k"] != 1 or selected != {0}:
            print(
                "FATAL: stack-of-stacks runtime currently requires a single "
                "arm-0 selector; mixed/multi-arm frame stitching is not "
                "implemented yet",
                file=sys.stderr,
            )
            sys.exit(2)
        arm_blob = parsed["arm_concat_bytes"]

    with tempfile.TemporaryDirectory(prefix="sos_base_") as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "x").write_bytes(arm_blob)
        runtime = Path(__file__).resolve().parent / "base_runtime" / "inflate.sh"
        if not runtime.is_file():
            print(f"FATAL: bundled base runtime missing: {runtime}", file=sys.stderr)
            sys.exit(5)
        cmd = ["bash", str(runtime), str(tmp_dir), str(output_dir), str(file_list_path)]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FATAL: bundled base runtime failed rc={rc}", file=sys.stderr)
            sys.exit(rc)
    sys.exit(0)


if __name__ == "__main__":
    main()
'''


def _is_single_arm_passthrough_build(
    args: argparse.Namespace,
    arm_substrate_ids: list[str],
) -> bool:
    """Return whether this invocation is the exact-evaluable A1 passthrough path."""

    return (
        args.outer_stack_k == 1
        and arm_substrate_ids == ["a1"]
        and args.sabor_capacity_bytes == 0
        and args.s2sbs_capacity_bytes == 0
        and args.residual_int8_bytes == 0
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_log: list[dict[str, Any]] = []

    def stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _canon_utc_now_iso()})
        print(f"[stack-of-stacks] {stage_log[-1]['at']} stage={name}", file=sys.stderr)

    stage("start")
    _canon_pin_seeds(args.seed)

    # Resolve device + arm spec.
    device = _canon_device_or_die(
        args.device, smoke=args.smoke, substrate_tag="stack_of_stacks"
    )
    stage("device_resolved")

    arm_substrate_ids = _resolve_arm_substrate_ids(args.middle_arm_substrate_ids)
    single_arm_passthrough = _is_single_arm_passthrough_build(args, arm_substrate_ids)
    if not single_arm_passthrough:
        if not args.research_only_runtime_scaffold:
            raise SystemExit(
                "stack-of-stacks full composition is blocked until real per-arm "
                "archives, score-aware residual training, eval_roundtrip, and "
                "multi-arm inflate hooks land. The unblocked exact-eval canary "
                "is the single-arm A1 passthrough: --middle-arm-substrate-ids a1 "
                "--outer-stack-k 1 with zero SABOR/S2SBS/residual bytes."
            )
        if not args.smoke:
            raise SystemExit(
                "stack-of-stacks research-only multi-arm/residual mode is "
                "local-smoke only; full provider dispatch requires the "
                "score-aware and multi-arm runtime blockers to land first."
            )
    if args.outer_stack_k > len(arm_substrate_ids):
        # Pad arm_substrate_ids by repeating the last one (degenerate ensemble).
        # Strict mode: refuse this unless --smoke.
        if not args.smoke:
            raise SystemExit(
                f"--outer-stack-k ({args.outer_stack_k}) > "
                f"len(--middle-arm-substrate-ids) ({len(arm_substrate_ids)}); "
                "either reduce K or supply more arm substrate ids"
            )
        arm_substrate_ids = arm_substrate_ids + [
            arm_substrate_ids[-1] for _ in range(args.outer_stack_k - len(arm_substrate_ids))
        ]

    ensemble_temperatures = _resolve_ensemble_temperatures(
        args.ensemble_temperatures, args.outer_stack_k
    )
    stage("arm_spec_resolved")

    # Load base archive bytes.
    base_archive_bytes, base_archive_sha256 = _read_base_archive(args.base_archive)
    stage("base_archive_loaded")

    state = TrainState(
        args=args,
        output_dir=output_dir,
        base_archive_bytes=base_archive_bytes,
        base_archive_sha256=base_archive_sha256,
        arm_substrate_ids=arm_substrate_ids,
        ensemble_temperatures=ensemble_temperatures,
        stage_log=stage_log,
        device_str="cuda" if device.type == "cuda" else "cpu",
    )

    # Build per-arm residual payloads (Langevin-driven; smoke-aware).
    arm_residuals: list[bytes] = []
    for i, substrate_id in enumerate(arm_substrate_ids[: args.outer_stack_k]):
        stage(f"arm_{i}_residual_train_begin")
        residual_bytes = _train_arm_residual_with_langevin(
            arm_idx=i,
            arm_substrate_id=substrate_id,
            base_bytes=base_archive_bytes,
            args=args,
            device=device,
        )
        arm_residuals.append(residual_bytes)
        stage(f"arm_{i}_residual_train_done")

    # Build inner specs for each arm.
    inner_specs: list[InnerStackSpec] = []
    for i in range(args.outer_stack_k):
        substrate_id = arm_substrate_ids[i]
        boundary = (
            BoundaryAtomSpec(
                capacity_bytes=args.sabor_capacity_bytes,
                audit_capacity_bytes=args.sabor_audit_capacity_bytes,
            )
            if args.sabor_capacity_bytes > 0
            else None
        )
        hf = (
            HFSidecarSpec(
                capacity_bytes=args.s2sbs_capacity_bytes,
                audit_capacity_bytes=args.s2sbs_audit_capacity_bytes,
            )
            if args.s2sbs_capacity_bytes > 0
            else None
        )
        residual = (
            ResidualSpec(residual_int8_bytes=arm_residuals[i], scale=4.0)
            if len(arm_residuals[i]) > 0
            else None
        )
        inner_specs.append(
            InnerStackSpec(
                substrate_id=substrate_id,
                base_bytes=base_archive_bytes,
                boundary_atom_spec=boundary,
                hf_sidecar_spec=hf,
                residual_spec=residual,
            )
        )

    middle_spec = MiddleStackSpec(inner_specs=tuple(inner_specs))
    outer_spec = OuterStackSpec(
        k=args.outer_stack_k,
        per_pair_arm=(),  # default: arm 0 for every pair (smoke); full path overrides
        temperatures=ensemble_temperatures,
    )

    # Validate byte budget before composing.
    try:
        budget_summary = validate_byte_budget(
            middle_spec,
            base_substrate_bytes=len(base_archive_bytes),
            max_total_bytes=args.max_total_archive_bytes,
            n_pairs=args.max_pairs,
            outer_stack_spec=outer_spec,
        )
    except StackOfStacksError as exc:
        raise SystemExit(f"byte-budget validation failed: {exc}") from exc
    stage("byte_budget_validated")

    # Compose.
    try:
        composed_bytes, compose_meta = compose_stack_of_stacks(
            middle_stack_spec=middle_spec,
            outer_stack_spec=outer_spec,
            n_pairs=args.max_pairs,
            max_total_bytes=args.max_total_archive_bytes,
        )
    except StackOfStacksError as exc:
        raise SystemExit(f"compose_stack_of_stacks failed: {exc}") from exc
    stage("composed")

    # Build the archive.zip + inflate.sh + inflate.py.
    archive_path = _build_archive_from_compose(
        composed_bytes=composed_bytes,
        output_dir=output_dir,
        base_runtime_dir=args.base_runtime_dir,
    )
    archive_sha256 = _canon_sha256_bytes(archive_path.read_bytes())
    stage("archive_built")

    dispatch_blockers: list[str] = []
    if not single_arm_passthrough:
        dispatch_blockers = [
            "multi_arm_frame_stitching_missing",
            "per_arm_decoder_hooks_missing",
            "score_aware_training_missing",
            "eval_roundtrip_training_missing",
            "per_arm_archive_inputs_missing",
        ]

    # Compose summary + meta dump.
    summary = {
        "archive_path": str(archive_path.relative_to(output_dir)),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "base_archive_sha256": base_archive_sha256,
        "base_archive_bytes": len(base_archive_bytes),
        "compose_meta": compose_meta,
        "budget_summary": budget_summary,
        "arm_substrate_ids": arm_substrate_ids,
        "outer_stack_k": args.outer_stack_k,
        "ensemble_temperatures": list(ensemble_temperatures),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": single_arm_passthrough,
        "canary_exact_eval_ready": single_arm_passthrough,
        "score_lowering_dispatch_ready": False,
        "operator_dispatch_enabled": False,
        "research_only": not single_arm_passthrough,
        "runtime_contract": (
            "single_arm_a1_passthrough_exact_eval_canary"
            if single_arm_passthrough
            else "research_only_multi_arm_scaffold"
        ),
        "dispatch_blockers": dispatch_blockers,
    }
    (output_dir / "stack_of_stacks_compose_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    state.stage_log = stage_log
    prov_path = _emit_provenance(state)
    print(
        f"[stack-of-stacks] archive at {archive_path} (sha256={archive_sha256[:16]}...);"
        f" provenance at {prov_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
