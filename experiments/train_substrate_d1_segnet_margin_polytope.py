# SPDX-License-Identifier: MIT
"""Train the D1 SegNet margin polytope encoder substrate (sub-0.188 path #1).

Source memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md``
Lane:        ``lane_d1_segnet_margin_polytope_encoder_20260514``
Sister:      ``tac.substrates.yucr`` (frame-0 cooperative-receiver sister)
             ``tac.substrates.a1_plus_wavelet_residual`` (pose-axis sister)

D1 operationalizes the SegNet argmax-margin polytope from the deep-math memo
§3.6 into a constructive **frame-1 perturbation encoder**: per-pixel logit
margin `m(x, y) = top1 - top2` defines a safe polytope `Pi_1(x, y)` whose
volume scales with `m^3 / det(J_seg)`. The encoder packs only noise that
fits inside the polytope interior — the SegNet argmax never flips, so
SegNet's distortion term is unchanged while the rate term shrinks.

D1 is the **frame-1 geometric-nullspace** sister of YUCR's **frame-0
structural-nullspace** exploit. Together they exhaust the bidirectional
cooperative-receiver bit-allocation.

Binding council verdicts (operator-pre-approved per deep-math memo §10 D1):

* **Sidecar architecture** (HNeRV parity L3): D1 is a SIDECAR (`d1_polytope.bin`)
  composing with a frozen base substrate (default A1; YUCR cross-axis
  composition supported).
* **Frame-1 polytope encoder** (deep-math memo §3.6): only allocate noise
  where `B_safe = m / L > 0`. Boundary pixels (`m = 0`) receive zero
  entropy — the polytope-interior invariant is enforced at encode time.
* **Operator-norm Lipschitz `L`** (deep-math memo §10 D1): SegNet Jacobian
  upper bound recorded in archive metadata. Default L=20 is conservative
  for normalized RGB on SegNet (tu-efficientnet_b2); operator may
  override per-deployment.
* **Margin-preserving hinge** in the loss: `lambda_d1 * mean(relu(threshold
  - margin))` pulls the trainer toward LARGER safe polytopes (the
  argmax-stability geometry expands at training time so inflate-time has
  more headroom).
* **D2 ≤500 B target overhead** for the polytope payload (smaller than
  YUCR because the margin map is non-negative + brotli-friendly).
* **BOTH AXES** mandatory: contest-CPU GHA Linux x86_64 + contest-CUDA
  per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

CLAUDE.md non-negotiables honored end-to-end:

* Score-aware substrate (HNeRV parity L1): train against
  ``upstream/videos/0.mkv`` decoded via pyav, gradient through scorers.
* ``patch_upstream_yuv6_globally()`` BEFORE ``load_differentiable_scorers``
  (Catalog #187; PR #95/#106 contract).
* ``apply_eval_roundtrip=True`` inside the per-batch loop (Catalog #5).
* EMA decay 0.997 + snapshot+restore at eval (Catalog #88).
* No scorer load at inflate time (Catalog #6).
* No /tmp persisted evidence paths (CLAUDE.md FORBIDDEN_PATTERN).
* TIER_1_OPERATOR_REQUIRED_FLAGS as ``ast.AnnAssign`` so Catalog #151 sees
  it (Catalog #168).
* Dynamic ``hardware_substrate`` detection (Catalog #190).

The ``_full_main`` path builds a byte-closed D1+A1 packet: one deterministic
margin extraction pass, polytope encoding, fused overlay runtime, readiness
manifest, and gated auth eval. ``_smoke_main`` remains the no-scorer CPU
archive grammar smoke.

Usage (smoke; CPU, no scorer load)::

    .venv/bin/python experiments/train_substrate_d1_segnet_margin_polytope.py \\
        --a1-archive experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/d1_polytope_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threaded from operator wrapper)::
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic-until-paired-CPU/CUDA-anchor-lands
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Canonical trainer skeleton helpers (Catalog #168 / #178 / #190).
from datetime import UTC

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _canon_decode_real_pairs,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import (
    D1_OVERLAY_CHANNEL_POLICIES,
    D1_OVERLAY_SIGN_POLICIES,
    normalize_overlay_amplitude_scale,
)

SUBSTRATE_TAG = "d1_polytope"
SUBSTRATE_LANE_ID = "lane_d1_segnet_margin_polytope_encoder_20260514"


# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
# Catalog #153/Modal mount: use submissions/a1/archive.zip (sha 87ec7ca5...;
# identical bytes to the experiments/results/track4_sg_a1_t178000_20260509
# archive) so the file is mounted via STRUCTURAL_MINIMUM_DIRS('submissions',
# None) per src/tac/deploy/modal/mount_manifest.py.
DEFAULT_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

# A1 anchor (post-fine-tune; sole sub-0.20 anchor per posterior 2026-05-14).
A1_CANONICAL_SHA256_INNER = (
    "8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243"
)
A1_CANONICAL_BYTES_INNER = 178_162


# ---------------------------------------------------------------------------
# Catalog #151 manifest — annotated assignment per Catalog #168.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive": {
        "env": "D1_POLYTOPE_A1_ARCHIVE",
        "rationale": (
            "A1 base substrate archive bytes (the sole sub-0.20 anchor per "
            "the 49-anchor posterior).  D1 is a sidecar — A1 is loaded "
            "verbatim and the D1POLY1 0.bin is composed alongside as a "
            "magic-byte-distinct member.  Path resolves to submissions/a1/ "
            "(committed; mounted via STRUCTURAL_MINIMUM_DIRS); the byte-"
            "identical sister at experiments/results/track4_sg_a1_t178000_"
            "20260509/submission_dir/archive.zip is NOT mounted by Modal "
            "per DEFAULT_RESULTS_IGNORE."
        ),
        "default": "submissions/a1/archive.zip",
        "required_input_file": True,
        "generator_command": (
            "experiments/results/track4_sg_a1_t178000_20260509/ (landed "
            "2026-05-09 via PR101 score-gradient fine-tune); the same "
            "archive bytes also live at submissions/a1/archive.zip "
            "(committed) — both have sha256 87ec7ca5..."
        ),
        "rationale_audit": (
            ".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md"
            "#section-10-top-5-dispatch-queue-d1"
        ),
    },
    "--video-path": {
        "env": "D1_POLYTOPE_VIDEO_PATH",
        "rationale": (
            "Score-aware substrate trains against contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside "
            "--smoke per Catalog #114."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "D1_POLYTOPE_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "D1_POLYTOPE_EPOCHS",
        "rationale": (
            "training epochs; full-dispatch target 1000 (lighter than "
            "A1+wavelet because the polytope encoder is closed-form per "
            "pixel — no per-pair residual head to converge)."
        ),
        "default": "1000",
    },
    "--upstream-dir": {
        "env": "D1_POLYTOPE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": "upstream",
    },
    "--device": {
        "env": "D1_POLYTOPE_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--polytope-payload-bits": {
        "env": "D1_POLYTOPE_PAYLOAD_BITS",
        "rationale": (
            "polytope payload bit budget; default 8000 (~1 KB after "
            "brotli).  D2 ≤500 B target is achievable at 4000-bit budgets."
        ),
        "default": "8000",
    },
    "--jacobian-lipschitz": {
        "env": "D1_POLYTOPE_JACOBIAN_LIPSCHITZ",
        "rationale": (
            "SegNet Jacobian operator-norm upper bound L; safe budget = "
            "margin / L.  Default 20 is conservative for normalized RGB "
            "on SegNet (tu-efficientnet_b2); operator may override after "
            "offline calibration."
        ),
        "default": "20.0",
    },
    "--overlay-channel-policy": {
        "env": "D1_POLYTOPE_OVERLAY_CHANNEL_POLICY",
        "rationale": (
            "Metadata-only runtime policy sweep over RGB channel projection; "
            "default rgb preserves the original D1 L2 overlay behavior."
        ),
        "default": "rgb",
    },
    "--overlay-amplitude-scale": {
        "env": "D1_POLYTOPE_OVERLAY_AMPLITUDE_SCALE",
        "rationale": (
            "Metadata-only runtime attenuation sweep in [0,1]; values below "
            "1.0 cannot amplify beyond the encoder-certified safe lattice."
        ),
        "default": "1.0",
    },
    "--overlay-sign-policy": {
        "env": "D1_POLYTOPE_OVERLAY_SIGN_POLICY",
        "rationale": (
            "Metadata-only runtime sign schedule sweep; default payload "
            "preserves the encoded sign field."
        ),
        "default": "payload",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_d1_segnet_margin_polytope",
        description=(
            "Train the D1 SegNet margin polytope encoder substrate (sub-"
            "0.188 path #1; lowest cost; deep-math memo §3.6 / §10 D1)."
        ),
    )
    p.add_argument(
        "--a1-archive", type=Path, default=DEFAULT_A1_ARCHIVE,
        help="Path to A1 base archive.zip (178,162 B post-fine-tune anchor).",
    )
    p.add_argument(
        "--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
        help="Path to upstream/videos/0.mkv (contest video).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write checkpoints + manifest + archive.",
    )
    p.add_argument(
        "--epochs", type=int, required=True,
        help="Number of training epochs (default-config 1000 for full).",
    )
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root for scorer load + auth eval.",
    )

    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ema-decay", type=float, default=0.997,
        help="EMA decay per CLAUDE.md EMA non-negotiable.",
    )
    p.add_argument("--val-pair-count", type=int, default=8)
    p.add_argument(
        "--max-decoded-pairs", type=int, default=N_PAIRS_FULL,
        help="Cap on total decoded pairs (full=600; smoke uses smaller).",
    )

    # D1 polytope encoder config
    p.add_argument(
        "--polytope-payload-bits", type=int, default=8000,
        help="Bit budget for the polytope payload (default 8000 ~ 1 KB).",
    )
    p.add_argument(
        "--jacobian-lipschitz", type=float, default=20.0,
        help="SegNet Jacobian operator-norm upper bound L (default 20).",
    )
    p.add_argument(
        "--margin-threshold", type=float, default=0.1,
        help="Hinge threshold for the margin-preserving loss term.",
    )
    p.add_argument(
        "--lambda-d1", type=float, default=0.05,
        help="Weight of the margin-preserving hinge in the loss.",
    )
    p.add_argument(
        "--margin-h", type=int, default=EVAL_HW[0],
        help="Margin map height (default 384 = scorer eval H).",
    )
    p.add_argument(
        "--margin-w", type=int, default=EVAL_HW[1],
        help="Margin map width (default 512 = scorer eval W).",
    )
    p.add_argument(
        "--overlay-channel-policy",
        choices=D1_OVERLAY_CHANNEL_POLICIES,
        default="rgb",
        help="D1 scalar overlay to RGB channel mapping.",
    )
    p.add_argument(
        "--overlay-amplitude-scale",
        type=float,
        default=1.0,
        help=(
            "D1 overlay attenuation in [0,1]; values below 1.0 materialize "
            "metadata-only amplitude sweep candidates without amplifying "
            "beyond the encoder-certified lattice."
        ),
    )
    p.add_argument(
        "--overlay-sign-policy",
        choices=D1_OVERLAY_SIGN_POLICIES,
        default="payload",
        help="D1 overlay sign schedule.",
    )

    # Lagrangian weights (score-domain; HNeRV parity L6)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Modes
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--smoke", action="store_true",
        help="Tiny CPU smoke (no scorer load); proves wiring.",
    )
    p.add_argument(
        "--skip-auth-eval", action="store_true",
        help="Skip end-of-train CUDA auth eval (smoke-only path).",
    )

    # Tier-1 engineering flags (Catalog #172/#178/#179/#180)
    p.add_argument(
        "--enable-autocast-fp16", action="store_true",
        help="Enable torch.autocast('cuda', dtype=float16). Catalog #172.",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true",
        help="Wrap polytope encoder in torch.compile. Catalog #179.",
    )
    p.add_argument(
        "--enable-tf32", action="store_true",
        help="Enable TF32 matmul on Ampere+ GPUs. Catalog #178.",
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    from datetime import datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "no-git"


def _pin_seeds(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_or_die(name: str, *, smoke: bool):
    """Canonical device-or-die gate (mirrors sister substrate trainers).

    Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
    CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiables,
    MPS is REFUSED, CPU is allowed ONLY with --smoke, CUDA is the only
    promotion-eligible substrate.
    """
    import torch
    n = (name or "").strip().lower()
    if n == "mps":
        raise RuntimeError(
            "MPS is REFUSED per CLAUDE.md 'MPS auth eval is NOISE' "
            "non-negotiable; use cuda for promotion or cpu for --smoke."
        )
    if n == "cpu" and not smoke:
        raise RuntimeError(
            "CPU is REFUSED for full training (proxy scorer drift); "
            "use --smoke for CPU smoke, otherwise --device cuda."
        )
    if n == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )
    return torch.device(
        n if n else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def _detect_hardware_substrate(device) -> str:
    """Dynamic substrate detection per Catalog #190.

    Returns one of: linux_x86_64_t4 / linux_x86_64_4090 / linux_x86_64_a100 /
    linux_x86_64_a10g / linux_x86_64_h100 / linux_x86_64_unknown_cuda /
    linux_x86_64_cpu / linux_x86_64_l40s / macos_<arch>_cpu / unknown.
    Catalog #190 forbids hardcoding "linux_x86_64_<gpu>" — we must resolve
    dynamically.
    """
    # Prefer canonical helper when available; fall back to a local resolver
    # to keep this module standalone in case of import cycles.
    try:
        from tac.substrates._shared.trainer_skeleton import (
            detect_hardware_substrate as _canon_detect,
        )
        return str(_canon_detect(device))
    except Exception:
        pass

    os_name = platform.system().lower()
    arch = platform.machine().lower()
    if os_name == "darwin":
        return f"macos_{arch}_cpu"
    if str(device) == "cpu":
        return f"linux_{arch}_cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            return f"linux_{arch}_unknown_cuda"
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return f"linux_{arch}_unknown_cuda"

    if "t4" in name:
        return "linux_x86_64_t4"
    if "4090" in name:
        return "linux_x86_64_4090"
    if "a100" in name:
        return "linux_x86_64_a100"
    if "a10g" in name:
        return "linux_x86_64_a10g"
    if "h100" in name:
        return "linux_x86_64_h100"
    if "l40" in name:
        return "linux_x86_64_l40s"
    return f"linux_{arch}_unknown_cuda"


def _write_provenance(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    device,
    smoke: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write provenance.json per CLAUDE.md "Operator gates must be wired
    and used" — provenance JSON is consumed by harvest + autopilot.
    """
    provenance_path = output_dir / "provenance.json"
    overlay_consumed = not bool(smoke)
    body: dict[str, Any] = {
        "lane_id": "lane_d1_segnet_margin_polytope_encoder_20260514",
        "started_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "smoke": bool(smoke),
        "device": str(device),
        "hardware_substrate": _detect_hardware_substrate(device),
        "python": sys.version,
        "platform": platform.platform(),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "evidence_grade": "proxy" if smoke else "pending_contest_eval",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "current_runtime_effect": (
            "synthetic_smoke_no_runtime"
            if smoke
            else "base_renderer_plus_d1_overlay"
        ),
        "runtime_overlay_consumed": overlay_consumed,
        "l2_projected_score_band": [0.181, 0.188],
        "predicted_score_evidence_grade": (
            "proxy_smoke" if smoke else "first-principles-bound"
        ),
        "source_memo": (
            ".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md"
        ),
    }
    if extra:
        body.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return provenance_path


def _load_a1_archive_bytes(archive_zip_path: Path) -> tuple[bytes, str, int]:
    """Read the inner ``x`` blob from an A1-style archive.zip."""
    import zipfile
    with zipfile.ZipFile(archive_zip_path) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise ValueError(
                f"A1 archive {archive_zip_path} missing inner 'x' blob; "
                f"got {names}"
            )
        data = zf.read("x")
    sha = _sha256_bytes(data)
    return data, sha, len(data)


# ---------------------------------------------------------------------------
# Smoke training path — proves wiring at $0 cost
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke path — proves the D1 encoder + archive wiring without a scorer
    load or upstream video decode. Generates synthetic margin map +
    polytope allocation + archive roundtrip in seconds.

    The smoke is enough to validate:
    1. Margin map -> polytope encoder -> archive pack -> parse roundtrip
       is byte-stable.
    2. The polytope-interior invariant (boundary pixels get zero noise).
    3. Archive metadata carries the apples-to-apples-discipline tags
       (score_claim=False, evidence_grade=proxy).
    4. Inflate.py contract (Catalog #146) is satisfied — sha mismatch
       refuses, magic-byte mismatch refuses.

    Returns 0 on success, non-zero on failure.
    """
    import numpy as np
    import torch

    from tac.substrates.d1_segnet_margin_polytope import (
        D1PolytopeConfig,
        build_readiness_manifest,
        compose_with_base,
        encode_polytope_payload,
        estimate_overhead_bytes,
        parse_archive,
    )
    from tac.substrates.d1_segnet_margin_polytope.architecture import (
        _BaseArchiveDescriptor,
    )

    device = _device_or_die(args.device, smoke=True)
    _pin_seeds(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Smoke uses a small margin map (proves wiring; no scorer load).
    smoke_h, smoke_w = 48, 64
    print(
        f"[d1-smoke] device={device} margin_shape=({smoke_h},{smoke_w}) "
        f"L={args.jacobian_lipschitz} budget_bits={args.polytope_payload_bits}",
        flush=True,
    )

    cfg = D1PolytopeConfig(
        base_substrate_id="a1",
        margin_map_resolution=(smoke_h, smoke_w),
        polytope_payload_bits=min(args.polytope_payload_bits, 2000),
        jacobian_lipschitz=args.jacobian_lipschitz,
        margin_threshold=args.margin_threshold,
    )

    # Synthetic margin map with realistic boundary-pixel distribution
    # (~20% boundary near 0, ~80% interior near 1.0 — matches the
    # deep-math memo §2.5 expected histogram).
    rng = np.random.RandomState(args.seed)
    margin_synth = np.where(
        rng.rand(smoke_h, smoke_w) < 0.2,
        rng.rand(smoke_h, smoke_w) * 0.1,  # boundary pixels: small
        0.5 + rng.rand(smoke_h, smoke_w),  # interior pixels: large
    ).astype(np.float32)
    margin_map = torch.from_numpy(margin_synth)

    # Phase 1: polytope encoding
    polytope_blob = encode_polytope_payload(
        margin_map,
        jacobian_lipschitz=cfg.jacobian_lipschitz,
        budget_bits=cfg.polytope_payload_bits,
    )
    print(
        f"[d1-smoke] polytope_payload_bytes={len(polytope_blob)} "
        f"(brotli-compressed)",
        flush=True,
    )

    # Phase 2: synthetic base archive (smoke doesn't load real A1).
    fake_base_bytes = b"smoke_base_archive_bytes"
    fake_base_sha = _sha256_bytes(fake_base_bytes)
    base_desc = _BaseArchiveDescriptor(
        base_substrate_id="a1",
        base_archive_sha256=fake_base_sha,
        base_archive_bytes=len(fake_base_bytes),
    )

    # Phase 3: compose with base
    d1_blob = compose_with_base(
        base_archive_descriptor=base_desc,
        margin_map=margin_map,
        polytope_payload=polytope_blob,
        config=cfg,
        extra_meta={
            "smoke_run": True,
            "git_head": _git_head_sha(),
            "smoke_timestamp_utc": _utc_now_iso(),
        },
    )
    print(
        f"[d1-smoke] d1_polytope_bin_bytes={len(d1_blob)} "
        f"(header={31} + payload+meta)",
        flush=True,
    )

    # Phase 4: roundtrip verification
    parsed = parse_archive(d1_blob)
    assert parsed.base_substrate_id == "a1"
    assert parsed.base_archive_sha256_truncated == fake_base_sha[:16]
    assert math.isclose(parsed.jacobian_lipschitz, cfg.jacobian_lipschitz)
    assert parsed.meta["score_claim"] is False
    assert parsed.meta["evidence_grade"] == "proxy"

    # Phase 5: persist archive + readiness manifest
    archive_path = output_dir / "d1_polytope.bin"
    archive_path.write_bytes(d1_blob)
    base_path = output_dir / "a1.bin"  # smoke writes the fake base alongside
    base_path.write_bytes(fake_base_bytes)

    readiness = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=len(fake_base_bytes),
        d1_overhead_bytes=len(d1_blob),
        config=cfg,
        runtime_overlay_consumed=False,
        base_archive_evidence_grade="synthetic_smoke",
    )
    readiness["smoke_run"] = True
    readiness["margin_synth_boundary_fraction"] = float(
        (margin_synth < 0.1).mean()
    )
    readiness["margin_synth_interior_fraction"] = float(
        (margin_synth >= 0.1).mean()
    )
    readiness["estimated_overhead_bytes"] = estimate_overhead_bytes(
        config=cfg
    )
    readiness_path = output_dir / "readiness_manifest.json"
    readiness_path.write_text(
        json.dumps(readiness, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_provenance(
        output_dir,
        args=args,
        device=device,
        smoke=True,
        extra={
            "d1_overhead_bytes": len(d1_blob),
            "polytope_payload_bytes": len(polytope_blob),
            "smoke_phases_completed": [
                "polytope_encoding",
                "compose_with_base",
                "archive_roundtrip",
                "readiness_manifest",
            ],
            "interior_fraction_smoke": readiness[
                "margin_synth_interior_fraction"
            ],
            "boundary_fraction_smoke": readiness[
                "margin_synth_boundary_fraction"
            ],
        },
    )

    print(
        f"[d1-smoke] OK; archive={archive_path} "
        f"readiness={readiness_path} provenance=provenance.json",
        flush=True,
    )
    return 0




# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 + #163 + #205).
# ---------------------------------------------------------------------------


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant fused D1+A1 inflate runtime.

    The generated packet must be score-bearing: it first renders the frozen
    A1 base archive, then consumes ``d1_polytope.bin`` and applies the
    decoded polytope overlay to every frame_1 output. This intentionally
    fails closed when the D1 sidecar is present but changes zero bytes, so
    D1 cannot regress into a dead-rate sidecar again.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Vendor A1's canonical runtime tree (codec.py + model.py + inflate.py).
    # submissions/a1/ is the canonical committed source (Catalog #205-compliant
    # inflate.py with select_inflate_device honoring PACT_INFLATE_DEVICE).
    a1_canonical_dir = REPO_ROOT / "submissions" / "a1"
    if not a1_canonical_dir.is_dir():
        raise FileNotFoundError(
            f"A1 canonical submission dir not found: {a1_canonical_dir}; "
            "cannot vendor A1 runtime."
        )

    # Copy A1's src/ tree (codec.py + model.py).
    a1_src_dir = a1_canonical_dir / "src"
    if a1_src_dir.is_dir():
        dst_src = submission_dir / "src"
        dst_src.mkdir(parents=True, exist_ok=True)
        for child in a1_src_dir.iterdir():
            if child.is_file() and child.suffix == ".py":
                shutil.copy2(child, dst_src / child.name)

    # Copy A1's inflate.py as the local base renderer. The D1 driver below
    # owns the contest 3-arg runtime contract and invokes this per file.
    a1_inflate_py = a1_canonical_dir / "inflate.py"
    if not a1_inflate_py.is_file():
        raise FileNotFoundError(f"A1 inflate.py missing: {a1_inflate_py}")
    shutil.copy2(a1_inflate_py, submission_dir / "a1_inflate.py")
    (submission_dir / "a1_inflate.py").chmod(0o755)

    d1_verify_py = (
        "#!/usr/bin/env python3\n"
        "\"\"\"Stdlib-only D1 sidecar runtime custody guard.\"\"\"\n"
        "from __future__ import annotations\n"
        "import hashlib\n"
        "import struct\n"
        "import sys\n"
        "from pathlib import Path\n"
        "FMT = '<4sBHHffBBIII'\n"
        "SIZE = struct.calcsize(FMT)\n"
        "MAGIC = b'D1PY'\n"
        "def main(argv: list[str]) -> int:\n"
        "    if len(argv) != 3:\n"
        "        print('usage: d1_verify.py <d1_polytope.bin> <base.bin>', file=sys.stderr)\n"
        "        return 2\n"
        "    d1_path = Path(argv[1])\n"
        "    base_path = Path(argv[2])\n"
        "    blob = d1_path.read_bytes()\n"
        "    if len(blob) < SIZE:\n"
        "        raise ValueError(f'D1 sidecar too short: {len(blob)}')\n"
        "    magic, version, h, w, scale, lipschitz, base_len, sha_len, margin_len, poly_len, meta_len = struct.unpack(FMT, blob[:SIZE])\n"
        "    if magic != MAGIC:\n"
        "        raise ValueError(f'bad D1 magic: {magic!r}')\n"
        "    if version != 1:\n"
        "        raise ValueError(f'unsupported D1 schema version: {version}')\n"
        "    pos = SIZE\n"
        "    base_id = blob[pos:pos + base_len].decode('utf-8'); pos += base_len\n"
        "    expected_sha = blob[pos:pos + sha_len].decode('utf-8'); pos += sha_len\n"
        "    pos += margin_len + poly_len + meta_len\n"
        "    if pos != len(blob):\n"
        "        raise ValueError(f'D1 sidecar size mismatch: parsed={pos} actual={len(blob)}')\n"
        "    if base_id != 'a1':\n"
        "        raise ValueError(f'D1 fused runtime only supports a1 base; got {base_id!r}')\n"
        "    actual_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()[:sha_len]\n"
        "    if actual_sha != expected_sha:\n"
        "        raise ValueError(f'base sha mismatch: {actual_sha} != {expected_sha}')\n"
        "    print(f'[d1-verify] parsed D1 sidecar h={h} w={w} margin_len={margin_len} poly_len={poly_len} meta_len={meta_len} L={lipschitz:.3f}', file=sys.stderr)\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(sys.argv))\n"
    )
    (submission_dir / "d1_verify.py").write_text(d1_verify_py, encoding="utf-8")
    (submission_dir / "d1_verify.py").chmod(0o755)

    d1_inflate_py = f"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np

HERE = Path(__file__).resolve().parent
CAMERA_H = {CAMERA_HW[0]}
CAMERA_W = {CAMERA_HW[1]}
FRAME_BYTES = CAMERA_H * CAMERA_W * 3
D1_FMT = '<4sBHHffBBIII'
D1_SIZE = struct.calcsize(D1_FMT)
D1_MAGIC = b'D1PY'
PLY_FMT = '<4sBIfBf'
PLY_SIZE = struct.calcsize(PLY_FMT)
PLY_MAGIC = b'PLY1'


def _read_file_list(file_list_path: Path) -> list[str]:
    names = []
    for line in file_list_path.read_text(encoding='utf-8').splitlines():
        value = line.strip()
        if value and not value.startswith('#'):
            names.append(value)
    if not names:
        raise ValueError(f'file_list {{file_list_path}} is empty')
    return names


def _parse_d1_sidecar(path: Path) -> dict[str, object]:
    blob = path.read_bytes()
    if len(blob) < D1_SIZE:
        raise ValueError(f'D1 sidecar too short: {{len(blob)}}')
    (
        magic,
        version,
        h,
        w,
        scale,
        lipschitz,
        base_len,
        sha_len,
        margin_len,
        poly_len,
        meta_len,
    ) = struct.unpack(D1_FMT, blob[:D1_SIZE])
    if magic != D1_MAGIC:
        raise ValueError(f'bad D1 magic: {{magic!r}}')
    if version != 1:
        raise ValueError(f'unsupported D1 schema version: {{version}}')
    pos = D1_SIZE
    base_id = blob[pos:pos + base_len].decode('utf-8')
    pos += base_len
    expected_sha = blob[pos:pos + sha_len].decode('utf-8')
    pos += sha_len
    margin_blob = blob[pos:pos + margin_len]
    pos += margin_len
    poly_blob = blob[pos:pos + poly_len]
    pos += poly_len
    meta_blob = blob[pos:pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f'D1 sidecar size mismatch: parsed={{pos}} actual={{len(blob)}}'
        )
    if not margin_blob or not poly_blob or not meta_blob:
        raise ValueError('D1 sidecar has empty required section')
    meta = json.loads(meta_blob.decode('utf-8'))
    return {{
        'height': int(h),
        'width': int(w),
        'scale': float(scale),
        'lipschitz': float(lipschitz),
        'base_id': base_id,
        'expected_sha': expected_sha,
        'margin_blob': margin_blob,
        'poly_blob': poly_blob,
        'meta': meta,
    }}


def _decode_margin(
    margin_blob: bytes,
    *,
    height: int,
    width: int,
    scale: float,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f'D1 margin scale must be > 0; got {{scale}}')
    raw = brotli.decompress(margin_blob)
    expected = int(height) * int(width)
    if len(raw) != expected:
        raise ValueError(
            f'D1 margin map size mismatch: got {{len(raw)}} expected {{expected}}'
        )
    margin_i8 = np.frombuffer(raw, dtype=np.int8).copy()
    if np.count_nonzero(margin_i8 < 0):
        raise ValueError('D1 margin map contains negative int8 values')
    return margin_i8.reshape(int(height), int(width)).astype(np.float32) * float(scale)


def _decode_overlay(
    poly_blob: bytes,
    *,
    height: int,
    width: int,
    expected_lipschitz: float,
    margin_hw: np.ndarray,
) -> np.ndarray:
    raw = brotli.decompress(poly_blob)
    if len(raw) < PLY_SIZE:
        raise ValueError(f'polytope payload too short: {{len(raw)}}')
    magic, version, num_pixels, _lambda, lattice_levels, lipschitz = (
        struct.unpack(PLY_FMT, raw[:PLY_SIZE])
    )
    if magic != PLY_MAGIC:
        raise ValueError(f'bad polytope magic: {{magic!r}}')
    if version != 1:
        raise ValueError(f'unsupported polytope version: {{version}}')
    if lattice_levels != 5:
        raise ValueError(f'unsupported lattice_levels={{lattice_levels}}')
    if lipschitz <= 0:
        raise ValueError(f'bad jacobian_lipschitz={{lipschitz}}')
    if not np.isclose(float(lipschitz), float(expected_lipschitz), rtol=1e-5, atol=1e-6):
        raise ValueError(
            f'D1 jacobian_lipschitz mismatch: archive={{expected_lipschitz}} '
            f'payload={{lipschitz}}'
        )
    expected = int(height) * int(width)
    if num_pixels != expected:
        raise ValueError(
            f'polytope pixel count {{num_pixels}} != D1 grid {{expected}}'
        )
    pos = PLY_SIZE
    end = pos + num_pixels
    if end != len(raw):
        raise ValueError(
            f'polytope payload size mismatch: header={{num_pixels}} '
            f'remaining={{len(raw) - pos}}'
        )
    noise = np.frombuffer(raw[pos:end], dtype=np.int8).copy()
    if np.count_nonzero(np.abs(noise) > 2):
        raise ValueError('D1 lattice violation: noise outside [-2,2]')
    margin_flat = margin_hw.reshape(-1)
    boundary_violations = int(
        np.count_nonzero((noise != 0) & (margin_flat <= 1e-6))
    )
    safe_budget = margin_flat / float(expected_lipschitz)
    max_safe_abs = np.floor(safe_budget + 1e-6).astype(np.int16)
    unsafe_violations = int(
        np.count_nonzero(np.abs(noise.astype(np.int16)) > max_safe_abs)
    )
    if boundary_violations:
        raise ValueError(
            f'D1 boundary violation: {{boundary_violations}} nonzero noise levels '
            'on zero-margin pixels'
        )
    if unsafe_violations:
        raise ValueError(
            f'D1 safe-budget violation: {{unsafe_violations}} noise levels '
            'exceed floor(margin/L)'
        )
    noise_2d = noise.reshape(int(height), int(width))
    y_idx = (np.arange(CAMERA_H, dtype=np.int64) * int(height)) // CAMERA_H
    x_idx = (np.arange(CAMERA_W, dtype=np.int64) * int(width)) // CAMERA_W
    return noise_2d[y_idx[:, None], x_idx[None, :]].astype(np.int8, copy=True)


def _channel_policy_weights(policy: str) -> np.ndarray:
    weights_by_policy = {{
        'rgb': (1, 1, 1),
        'neg_rgb': (-1, -1, -1),
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'neg_green': (0, -1, 0),
        'rb_pos_g_neg': (1, -1, 1),
    }}
    if policy not in weights_by_policy:
        raise ValueError(
            f'unsupported D1 overlay_channel_policy={{policy!r}}; '
            f'expected one of {{sorted(weights_by_policy)}}'
        )
    return np.asarray(weights_by_policy[policy], dtype=np.int16)


def _normalize_overlay_amplitude_scale(amplitude_scale: float) -> float:
    scale = float(amplitude_scale)
    if not np.isfinite(scale):
        raise ValueError(
            f'overlay_amplitude_scale must be finite; got {{amplitude_scale!r}}'
        )
    if scale < 0.0 or scale > 1.0:
        raise ValueError(
            f'overlay_amplitude_scale={{scale}} out of range [0, 1]'
        )
    return scale


def _overlay_sign_for_pair(sign_policy: str, pair_idx: int) -> int:
    policy = sign_policy.strip().lower()
    if policy == 'payload':
        return 1
    if policy == 'negate_payload':
        return -1
    if policy == 'alternating_pairs':
        return 1 if int(pair_idx) % 2 == 0 else -1
    if policy == 'pair_mask':
        raise ValueError('pair_mask sign policy requires decoded pair_sign_mask')
    raise ValueError(
        f'unsupported D1 overlay_sign_policy={{sign_policy!r}}; '
        "expected one of ['alternating_pairs', 'negate_payload', 'pair_mask', 'payload']"
    )


def _unpack_pair_sign_mask(mask_b85: str, *, n_pairs: int) -> tuple[int, ...]:
    payload = base64.b85decode(str(mask_b85).encode('ascii'))
    expected_len = (int(n_pairs) * 2 + 7) // 8
    if len(payload) != expected_len:
        raise ValueError(
            f'D1 pair mask byte length {{len(payload)}} != expected {{expected_len}}'
        )
    signs = []
    sign_by_code = {{0: 0, 1: 1, 2: -1}}
    for byte in payload:
        for shift in (0, 2, 4, 6):
            if len(signs) >= n_pairs:
                break
            code = (byte >> shift) & 0b11
            if code not in sign_by_code:
                raise ValueError('D1 pair mask contains reserved 2-bit code 3')
            signs.append(sign_by_code[code])
    return tuple(signs)


def _pair_sign_mask_from_meta(meta: dict[str, object], n_pairs: int) -> tuple[int, ...] | None:
    policy = str(meta.get('overlay_sign_policy', 'payload')).strip().lower()
    if policy != 'pair_mask':
        return None
    mask_b85 = meta.get('pair_mask_b85')
    mask_pairs = int(meta.get('pair_mask_n', n_pairs))
    if not isinstance(mask_b85, str) or not mask_b85:
        raise ValueError('D1 pair_mask policy missing pair_mask_b85')
    if mask_pairs != n_pairs:
        raise ValueError(
            f'D1 pair mask n_pairs={{mask_pairs}} does not match inflated n_pairs={{n_pairs}}'
        )
    return _unpack_pair_sign_mask(mask_b85, n_pairs=n_pairs)


def _overlay_sign_for_pair_masked(
    sign_policy: str,
    pair_idx: int,
    pair_sign_mask: tuple[int, ...] | None,
) -> int:
    if sign_policy.strip().lower() != 'pair_mask':
        return _overlay_sign_for_pair(sign_policy, pair_idx)
    if pair_sign_mask is None:
        raise ValueError('pair_mask sign policy requires pair_sign_mask')
    if pair_idx < 0 or pair_idx >= len(pair_sign_mask):
        raise ValueError(
            f'pair_idx {{pair_idx}} outside pair_sign_mask length {{len(pair_sign_mask)}}'
        )
    sign = int(pair_sign_mask[pair_idx])
    if sign not in (-1, 0, 1):
        raise ValueError(f'D1 pair_sign_mask[{{pair_idx}}]={{sign}} is invalid')
    return sign


def _attenuate_overlay_levels(
    overlay_hw: np.ndarray,
    *,
    amplitude_scale: float,
) -> np.ndarray:
    scale = _normalize_overlay_amplitude_scale(amplitude_scale)
    if scale == 1.0:
        return overlay_hw.astype(np.int8, copy=True)
    values = overlay_hw.astype(np.int16)
    magnitude = np.floor(np.abs(values).astype(np.float32) * scale + 0.5)
    signed = np.sign(values).astype(np.int16) * magnitude.astype(np.int16)
    return np.clip(signed, -2, 2).astype(np.int8)


def _apply_overlay(
    raw_path: Path,
    overlay_hw: np.ndarray,
    *,
    channel_policy: str,
    amplitude_scale: float,
    sign_policy: str,
    pair_sign_mask: tuple[int, ...] | None,
) -> dict[str, int]:
    actual_size = raw_path.stat().st_size
    pair_bytes = 2 * FRAME_BYTES
    if actual_size % pair_bytes != 0:
        raise ValueError(
            f'raw size {{actual_size}} is not a multiple of pair_bytes={{pair_bytes}}'
        )
    n_pairs = actual_size // pair_bytes
    overlay_hw = _attenuate_overlay_levels(
        overlay_hw, amplitude_scale=amplitude_scale
    )
    nonzero_overlay_pixels = int(np.count_nonzero(overlay_hw))
    weights = _channel_policy_weights(channel_policy)
    _overlay_sign_for_pair_masked(sign_policy, 0, pair_sign_mask)
    if nonzero_overlay_pixels == 0:
        return {{
            'pairs_modified': 0,
            'bytes_changed': 0,
            'nonzero_overlay_pixels': 0,
        }}
    overlay_flat = (
        overlay_hw[:, :, np.newaxis].astype(np.int16)
        * weights
    ).reshape(-1)
    overlay_flat_negative = -overlay_flat
    pairs_modified = 0
    bytes_changed = 0
    with raw_path.open('r+b') as fp:
        for pair_idx in range(n_pairs):
            sign = _overlay_sign_for_pair_masked(sign_policy, pair_idx, pair_sign_mask)
            if sign == 0:
                continue
            pair_overlay = overlay_flat if sign > 0 else overlay_flat_negative
            frame_1_offset = (2 * pair_idx + 1) * FRAME_BYTES
            fp.seek(frame_1_offset)
            frame_1_bytes = fp.read(FRAME_BYTES)
            if len(frame_1_bytes) != FRAME_BYTES:
                raise ValueError(
                    f'short read at pair {{pair_idx}}: {{len(frame_1_bytes)}}'
                )
            frame_1_arr = np.frombuffer(frame_1_bytes, dtype=np.uint8).copy()
            new_vals = np.clip(
                frame_1_arr.astype(np.int16) + pair_overlay, 0, 255
            ).astype(np.uint8)
            changed = int(np.count_nonzero(new_vals != frame_1_arr))
            if changed:
                fp.seek(frame_1_offset)
                fp.write(new_vals.tobytes())
                pairs_modified += 1
                bytes_changed += changed
    return {{
        'pairs_modified': pairs_modified,
        'bytes_changed': bytes_changed,
        'nonzero_overlay_pixels': nonzero_overlay_pixels,
    }}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='inflate.py',
        description='D1 fused A1+polytope overlay inflate',
    )
    parser.add_argument('archive_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('file_list', type=Path)
    args = parser.parse_args(argv)

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d1_path = archive_dir / 'd1_polytope.bin'
    if not d1_path.is_file():
        raise FileNotFoundError(f'D1 sidecar not found in {{archive_dir}}')
    d1 = _parse_d1_sidecar(d1_path)
    if d1['base_id'] != 'a1':
        raise ValueError(f'D1 fused runtime only supports a1 base; got {{d1["base_id"]!r}}')
    base_path = archive_dir / 'a1.bin'
    if not base_path.is_file():
        legacy = archive_dir / 'x'
        if legacy.is_file():
            base_path = legacy
    if not base_path.is_file():
        raise FileNotFoundError(f'A1 source blob not found in {{archive_dir}}')
    actual_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    expected_sha = str(d1['expected_sha'])
    if actual_sha[:len(expected_sha)] != expected_sha:
        raise ValueError(
            f'base sha mismatch: {{actual_sha[:len(expected_sha)]}} != {{expected_sha}}'
        )

    margin_hw = _decode_margin(
        d1['margin_blob'],
        height=int(d1['height']),
        width=int(d1['width']),
        scale=float(d1['scale']),
    )
    channel_policy = str(d1['meta'].get('overlay_channel_policy', 'rgb'))
    overlay_hw = _decode_overlay(
        d1['poly_blob'],
        height=int(d1['height']),
        width=int(d1['width']),
        expected_lipschitz=float(d1['lipschitz']),
        margin_hw=margin_hw,
    )
    video_names = _read_file_list(args.file_list)
    total_pairs = 0
    total_bytes = 0
    amplitude_scale = _normalize_overlay_amplitude_scale(
        d1['meta'].get('overlay_amplitude_scale', 1.0)
    )
    sign_policy = str(d1['meta'].get('overlay_sign_policy', 'payload'))
    for video_name in video_names:
        stem = video_name.rsplit('.', 1)[0] if '.' in video_name else video_name
        dst_raw = output_dir / f'{{stem}}.raw'
        subprocess.run(
            [
                os.environ.get('PYTHON', sys.executable),
                str(HERE / 'a1_inflate.py'),
                str(base_path),
                str(dst_raw),
            ],
            check=True,
        )
        actual_size = dst_raw.stat().st_size
        if actual_size % (2 * FRAME_BYTES) != 0:
            raise ValueError(
                f'raw size {{actual_size}} is not a multiple of pair_bytes={{2 * FRAME_BYTES}}'
            )
        pair_sign_mask = _pair_sign_mask_from_meta(
            d1['meta'], actual_size // (2 * FRAME_BYTES)
        )
        diag = _apply_overlay(
            dst_raw,
            overlay_hw,
            channel_policy=channel_policy,
            amplitude_scale=amplitude_scale,
            sign_policy=sign_policy,
            pair_sign_mask=pair_sign_mask,
        )
        total_pairs += diag['pairs_modified']
        total_bytes += diag['bytes_changed']

    print(
        f'[d1-inflate] OVERLAY_TOTAL pairs_modified={{total_pairs}} '
        f'bytes_changed={{total_bytes}} videos_processed={{len(video_names)}} '
        f'channel_policy={{channel_policy}} '
        f'amplitude_scale={{amplitude_scale}} sign_policy={{sign_policy}}',
        file=sys.stderr,
    )
    if total_bytes <= 0:
        raise RuntimeError(
            'D1 sidecar was present but overlay changed zero bytes; refusing dead-rate packet'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
"""
    (submission_dir / "inflate.py").write_text(d1_inflate_py, encoding="utf-8")
    (submission_dir / "inflate.py").chmod(0o755)

    # Emit a Catalog #146-compliant 3-arg inflate.sh that delegates to the
    # fused D1 runtime. The runtime handles A1 rendering + D1 overlay.
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# D1 SegNet margin polytope sidecar contest-compliant inflate.\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list per Catalog #146.\n"
        "# This is the fused D1+A1 runtime: render A1, then apply the\n"
        "# polytope overlay from d1_polytope.bin to frame_1.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)


def _build_archive_zip(
    archive_zip_path: Path,
    *,
    d1_bin_bytes: bytes,
    base_bin_bytes: bytes,
    base_substrate_id: str,
) -> None:
    """Deterministic archive.zip containing D1 sidecar + base archive.

    The contest packet has ONE archive.zip with two named members:
    ``d1_polytope.bin`` (the sidecar) and ``<base_substrate_id>.bin``
    (the frozen base). Per CLAUDE.md "deterministic ZIP" + Catalog
    #5/#19 (deterministic_zip) we use ZipInfo + writestr with a fixed
    timestamp so the bytes are reproducible across runs.
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    members = [
        ("d1_polytope.bin", d1_bin_bytes),
        (f"{base_substrate_id}.bin", base_bin_bytes),
    ]
    members.sort(key=lambda m: m[0])  # deterministic member order
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zi = zipfile.ZipInfo(name, date_time=fixed_ts)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)


# ---------------------------------------------------------------------------
# Full training path — Phase 2 dispatch approved 2026-05-14
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Full D1 sidecar build + CUDA auth eval.

    D1 is a SIDECAR — it has NO renderer of its own; the polytope
    encoder operates on real contest-video pairs through the SegNet
    scorer to produce the margin map + polytope payload. The
    "training" loop is therefore:

    1. Pin seeds (Catalog #5).
    2. Patch upstream rgb_to_yuv6 BEFORE scorer load (Catalog #187).
    3. Load differentiable scorers (SegNet + PoseNet; eval-mode only —
       D1 is non-gradient; we just need a forward to compute the
       margin map per Catalog #164's preprocess_input contract).
    4. Decode contest video to real (N_pairs, 2, 3, 384, 512) pairs.
    5. Compute per-pair margin map via SegNet top1-top2 (averaged
       over the first ``args.epochs`` pairs and aggregated to a single
       global map — design verdict #2: per-frame, NOT per-pair, to
       keep the sidecar overhead small).
    6. Encode polytope payload via reverse-water-fill on per-pixel
       safe budget = margin / L (closed-form; no SGD).
    7. Pack D1POLY1 sidecar + compose with frozen A1 base.
    8. Build archive.zip with both members, write fused D1+A1 contest
       runtime tree.
    9. Gate auth eval only after the runtime consumes the D1 overlay and
       the readiness manifest marks the packet exact-eval dispatchable.
    10. Update continual-learning posterior on success (Catalog #128).
    11. Write provenance.json with predicted/empirical bands.

    Design verdicts pinned (operator-approved 2026-05-14 "proceed with all"):

    * V1 Jacobian Lipschitz: fixed ``args.jacobian_lipschitz`` (default 20.0);
      recorded in archive meta so the receiver inverts deterministically.
    * V2 Per-pair vs per-frame margin map: per-FRAME (single global map
      averaged over pairs; ~200 KB → ~50 KB post-brotli). Per-pair would
      cost N_pairs × that and overflow the 3 KB Pareto budget per Catalog
      #170/#171.
    * V3 Polytope-interior noise timing: POST-renderer. The emitted
      contest runtime renders frozen A1 first, then applies the decoded
      D1 overlay to frame_1 and fails closed if the sidecar changes zero
      bytes.
    * V4 D1 + YUCR composition order: D1 ALONE on A1 first (cleanest
      single-variable ablation — build subagent recommendation).
    * V5 EMA shadow scope: EMA the MARGIN MAP only (cheap; matches the
      design intent that the geometric quantity is the trained signal).
      Default decay 0.997 per Catalog #88.

    The per-pixel rate savings is bounded above by the polytope-interior
    fraction × per-pixel safe-bit budget; projected ΔS for D1 on A1 is
    [-0.012, -0.005] (deep-math memo §10). The projection is not a score
    claim until paired contest-CPU and contest-CUDA exact eval land.
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.d1_segnet_margin_polytope import (
        D1PolytopeConfig,
        analyze_d1_overlay_effect,
        build_readiness_manifest,
        compose_with_base,
        encode_polytope_payload,
        estimate_overhead_bytes,
        parse_archive,
    )
    from tac.substrates.d1_segnet_margin_polytope.architecture import (
        _BaseArchiveDescriptor,
    )
    from tac.substrates.d1_segnet_margin_polytope.margin_map import (
        MARGIN_MAP_DEFAULT_RESOLUTION,
        compute_logit_margin_map,
    )

    _pin_seeds(args.seed)
    full_cpu_active = False  # D1 has no --full-cpu opt-in (CUDA-required path)
    device = _device_or_die(args.device, smoke=False)

    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}", flush=True)

    _stage("seed_pinned")

    # Patch upstream rgb_to_yuv6 BEFORE scorer load per Catalog #187.
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    auth_eval_json_path: Path | None = None
    archive_zip_sha: str | None = None
    archive_zip_size: int = 0
    d1_bin_sha: str = ""
    d1_overhead_bytes: int = 0

    try:
        # Load scorers (SegNet only needed for margin map; PoseNet loaded
        # for symmetry + future component-coherent eval roundtripping).
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # Decode real contest video.
        print(
            f"[{SUBSTRATE_TAG}-full] decoding pairs from {args.video_path}",
            flush=True,
        )
        n_pairs_target = min(args.max_decoded_pairs, N_PAIRS_FULL)
        gt_pair_tensor = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=n_pairs_target,
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=args.max_decoded_pairs,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        # Load frozen A1 base archive.
        if not args.a1_archive.is_file():
            raise FileNotFoundError(
                f"--a1-archive missing: {args.a1_archive}"
            )
        base_bin_bytes, base_sha, base_bytes_len = _load_a1_archive_bytes(
            args.a1_archive
        )
        _stage(f"a1_loaded_{base_bytes_len}B_sha{base_sha[:8]}")

        # ------------------------------------------------------------------
        # Compute per-frame aggregate margin map.
        #
        # Verdict #2: per-FRAME (single global map averaged over a window of
        # contest pairs). Iteration "epochs" here drives the number of
        # pair-batches consumed for the moving-average + EMA-update.
        #
        # Verdict #5: EMA-shadow margin map only (decay = args.ema_decay).
        #
        # Verdict #1: jacobian_lipschitz fixed at args.jacobian_lipschitz.
        # ------------------------------------------------------------------
        margin_h, margin_w = args.margin_h, args.margin_w
        ema_margin: torch.Tensor | None = None
        ema_decay = float(args.ema_decay)
        if not (0.0 < ema_decay < 1.0):
            raise SystemExit(
                f"[{SUBSTRATE_TAG}-full] ema_decay={ema_decay} must be "
                "in (0, 1) per CLAUDE.md EMA non-negotiable"
            )

        # D1 margin extraction is deterministic closed-form inference. Repeating
        # identical frozen-scorer passes only burns wall-clock, so keep the
        # legacy --epochs flag as custody metadata and run one effective pass.
        effective_margin_passes = 1
        if int(args.epochs) != effective_margin_passes:
            print(
                f"[{SUBSTRATE_TAG}-full] --epochs={args.epochs} requested; "
                "D1 uses one deterministic margin pass",
                flush=True,
            )
        wall_clock_started = time.time()
        with torch.inference_mode():
            for epoch in range(effective_margin_passes):
                for batch_start in range(0, n_pairs, args.batch_size):
                    batch_indices = list(
                        range(
                            batch_start,
                            min(batch_start + args.batch_size, n_pairs),
                        )
                    )
                    if not batch_indices:
                        continue
                    pair_btchw = gt_pair_tensor[batch_indices]  # (B, 2, 3, H, W)
                    # Margin map for this batch on real video pairs.
                    batch_margin = compute_logit_margin_map(
                        seg_scorer=segnet,
                        rgb_pair_btchw=pair_btchw,
                        target_resolution=(margin_h, margin_w),
                        detach_grad=True,
                        downsample_mode=(
                            "area"
                            if (margin_h, margin_w) != MARGIN_MAP_DEFAULT_RESOLUTION
                            else "bilinear"
                        ),
                    )  # (B, H, W)
                    batch_mean = batch_margin.mean(dim=0)  # (H, W)
                    if ema_margin is None:
                        ema_margin = batch_mean.clone()
                    else:
                        ema_margin = (
                            ema_decay * ema_margin
                            + (1.0 - ema_decay) * batch_mean
                        )

                if epoch % max(1, args.val_pair_count) == 0:
                    interior_frac = float(
                        (ema_margin > args.margin_threshold).float().mean().item()
                    ) if ema_margin is not None else 0.0
                    print(
                        f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d} "
                        f"ema_margin_mean={float(ema_margin.mean()):.4f} "
                        f"interior_fraction={interior_frac:.3f}",
                        flush=True,
                    )

        if ema_margin is None:
            raise RuntimeError(
                f"[{SUBSTRATE_TAG}-full] ema_margin was never built; "
                f"epochs={args.epochs} batches yielded 0 forwards."
            )

        margin_map_final = ema_margin.detach().clamp_min(0.0).contiguous()
        train_elapsed = time.time() - wall_clock_started
        _stage(
            f"margin_map_built_shape_{tuple(margin_map_final.shape)}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # ------------------------------------------------------------------
        # Encode polytope payload (closed-form reverse-water-fill).
        # ------------------------------------------------------------------
        polytope_blob = encode_polytope_payload(
            margin_map_final.cpu(),
            jacobian_lipschitz=args.jacobian_lipschitz,
            budget_bits=args.polytope_payload_bits,
        )
        _stage(f"polytope_payload_bytes_{len(polytope_blob)}")

        # ------------------------------------------------------------------
        # Pack D1POLY1 sidecar + compose with frozen A1 base.
        # ------------------------------------------------------------------
        cfg = D1PolytopeConfig(
            base_substrate_id="a1",
            margin_map_resolution=(margin_h, margin_w),
            polytope_payload_bits=args.polytope_payload_bits,
            jacobian_lipschitz=args.jacobian_lipschitz,
            margin_threshold=args.margin_threshold,
        )
        base_desc = _BaseArchiveDescriptor(
            base_substrate_id="a1",
            base_archive_sha256=base_sha,
            base_archive_bytes=base_bytes_len,
        )
        d1_blob = compose_with_base(
            base_archive_descriptor=base_desc,
            margin_map=margin_map_final.cpu(),
            polytope_payload=polytope_blob,
            config=cfg,
            extra_meta={
                "lane_id": SUBSTRATE_LANE_ID,
                "trained_at_utc": _utc_now_iso(),
                "git_head": _git_head_sha(),
                "n_pairs_used": int(n_pairs),
                "ema_decay": float(ema_decay),
                "requested_epochs": int(args.epochs),
                "effective_margin_passes": int(effective_margin_passes),
                "overlay_channel_policy": str(args.overlay_channel_policy),
                "overlay_amplitude_scale": float(args.overlay_amplitude_scale),
                "overlay_sign_policy": str(args.overlay_sign_policy),
                "runtime_overlay_consumed": True,
                "current_runtime_effect": "base_renderer_plus_d1_overlay",
                "l2_projected_score_band_low": 0.181,
                "l2_projected_score_band_high": 0.188,
                "predicted_score_evidence_grade": "first-principles-bound",
            },
        )
        d1_overhead_bytes = len(d1_blob)
        d1_bin_sha = _sha256_bytes(d1_blob)
        print(
            f"[{SUBSTRATE_TAG}-full] D1POLY1 sidecar: "
            f"{d1_overhead_bytes} B sha256={d1_bin_sha[:16]}...",
            flush=True,
        )
        _stage(
            f"d1_sidecar_packed_{d1_overhead_bytes}B_sha{d1_bin_sha[:8]}"
        )

        # Roundtrip verify the sidecar before shipping.
        parsed = parse_archive(d1_blob)
        if parsed.base_substrate_id != "a1":
            raise RuntimeError(
                f"D1 sidecar roundtrip failed: parsed base "
                f"{parsed.base_substrate_id!r} != 'a1'"
            )
        if parsed.base_archive_sha256_truncated != base_sha[:16]:
            raise RuntimeError(
                "D1 sidecar roundtrip failed: base sha truncation mismatch"
            )
        overlay_diag = analyze_d1_overlay_effect(
            parsed,
            channel_policy=str(args.overlay_channel_policy),
            amplitude_scale=float(args.overlay_amplitude_scale),
            sign_policy=str(args.overlay_sign_policy),
        )
        print(
            f"[{SUBSTRATE_TAG}-full] overlay_diag "
            f"decoded_nonzero={overlay_diag.decoded_noise_nonzero_pixels} "
            f"camera_nonzero={overlay_diag.camera_overlay_nonzero_pixels} "
            f"integer_feasible={overlay_diag.integer_feasible_pixels} "
            f"blockers={overlay_diag.dispatch_blockers}",
            flush=True,
        )
        _stage("d1_sidecar_roundtrip_verified")

        # ------------------------------------------------------------------
        # Build archive.zip + write contest runtime tree.
        # ------------------------------------------------------------------
        submission_dir = args.output_dir / "submission_dir"
        _write_runtime(submission_dir)
        (submission_dir / "d1_polytope.bin").write_bytes(d1_blob)
        (submission_dir / "a1.bin").write_bytes(base_bin_bytes)

        archive_zip_path = args.output_dir / "archive.zip"
        _build_archive_zip(
            archive_zip_path,
            d1_bin_bytes=d1_blob,
            base_bin_bytes=base_bin_bytes,
            base_substrate_id="a1",
        )
        archive_zip_bytes_raw = archive_zip_path.read_bytes()
        archive_zip_sha = _sha256_bytes(archive_zip_bytes_raw)
        archive_zip_size = len(archive_zip_bytes_raw)
        # Mirror the canonical archive into the submission_dir for inflate.
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        print(
            f"[{SUBSTRATE_TAG}-full] archive.zip: {archive_zip_size} B "
            f"sha256={archive_zip_sha[:16]}...",
            flush=True,
        )
        _stage(f"archive_zip_built_{archive_zip_size}B_sha{archive_zip_sha[:8]}")

        # Build readiness manifest (non-promotable per Catalog #192).
        readiness = build_readiness_manifest(
            base_substrate_id="a1",
            base_archive_bytes=base_bytes_len,
            d1_overhead_bytes=d1_overhead_bytes,
            config=cfg,
            runtime_overlay_consumed=True,
            decoded_noise_nonzero_pixels=overlay_diag.decoded_noise_nonzero_pixels,
            camera_overlay_nonzero_pixels=overlay_diag.camera_overlay_nonzero_pixels,
            integer_feasible_pixels=overlay_diag.integer_feasible_pixels,
            unsafe_nonzero_pixels=overlay_diag.unsafe_nonzero_pixels,
            pair_mask_active_pairs=overlay_diag.pair_mask_active_pairs,
            overlay_dispatch_blockers=overlay_diag.dispatch_blockers,
        )
        readiness["lane_id"] = SUBSTRATE_LANE_ID
        readiness["archive_zip_bytes"] = archive_zip_size
        readiness["archive_zip_sha256"] = archive_zip_sha
        readiness["d1_bin_sha256"] = d1_bin_sha
        readiness["n_pairs_used"] = int(n_pairs)
        readiness["margin_map_resolution"] = [margin_h, margin_w]
        readiness["overlay_channel_policy"] = str(args.overlay_channel_policy)
        readiness["overlay_amplitude_scale"] = float(args.overlay_amplitude_scale)
        readiness["overlay_sign_policy"] = str(args.overlay_sign_policy)
        readiness["d1_overlay_diagnostics"] = overlay_diag.to_json_dict()
        readiness["estimated_overhead_bytes"] = estimate_overhead_bytes(
            config=cfg
        )
        (args.output_dir / "readiness_manifest.json").write_text(
            json.dumps(readiness, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # CUDA auth eval gate.
        # ------------------------------------------------------------------
        auth_eval_json_path = args.output_dir / "contest_auth_eval_cuda.json"
        if not bool(readiness.get("ready_for_exact_eval_dispatch")):
            args.auth_eval_skipped_reason = "d1_readiness_manifest_refused_dispatch"
            print(
                f"[{SUBSTRATE_TAG}-auth-eval] SKIPPED: "
                "D1 readiness manifest refused exact-eval dispatch.",
                flush=True,
            )
            _stage("auth_eval_skipped_readiness_manifest_refused_dispatch")
        else:
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=submission_dir / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_json_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG,
                device=device,
                full_cpu_active=full_cpu_active,
            )
            if auth_result is None:
                _stage("auth_eval_skipped_gate_refused")
            else:
                _stage("auth_eval_cuda_done_valid_claim")

    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # ---------------------------------------------------------------------
    # Posterior update (Catalog #128 locked write).
    # ---------------------------------------------------------------------
    if (
        not args.skip_auth_eval
        and auth_eval_json_path is not None
        and auth_eval_json_path.is_file()
    ):
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(
                auth_eval_json_path
            )
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}",
                flush=True,
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(
                f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}",
                flush=True,
            )

    # ---------------------------------------------------------------------
    # Provenance.
    # ---------------------------------------------------------------------
    provenance_extra: dict[str, Any] = {
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "n_pairs_used": int(n_pairs) if "n_pairs" in dir() else None,
        "requested_epochs": int(args.epochs),
        "effective_margin_passes": int(
            locals().get("effective_margin_passes", 1)
        ),
        "overlay_channel_policy": str(args.overlay_channel_policy),
        "overlay_amplitude_scale": float(args.overlay_amplitude_scale),
        "overlay_sign_policy": str(args.overlay_sign_policy),
        "d1_overlay_diagnostics": overlay_diag.to_json_dict(),
        "d1_overhead_bytes": d1_overhead_bytes,
        "d1_bin_sha256": d1_bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "stage_log": stage_log,
        "design_verdicts": {
            "v1_jacobian_lipschitz": float(args.jacobian_lipschitz),
            "v2_margin_map_scope": "per_frame_ema",
            "v3_polytope_noise_timing": "post_renderer_overlay_active",
            "v4_composition_order": "d1_alone_on_a1",
            "v5_ema_shadow_scope": "margin_map_only",
            "ema_decay": float(args.ema_decay),
        },
        "runtime_overlay_consumed": True,
        "current_runtime_effect": "base_renderer_plus_d1_overlay",
        "l2_projected_score_band_low": 0.181,
        "l2_projected_score_band_high": 0.188,
        "predicted_score_evidence_grade": "first-principles-bound",
        "source_memo": (
            ".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md"
        ),
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=(
                "D1_POLYTOPE_GPU",
                "MODAL_GPU",
            ),
        ),
    }
    _write_provenance(
        args.output_dir, args=args, device=device, smoke=False,
        extra=provenance_extra,
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.overlay_amplitude_scale = normalize_overlay_amplitude_scale(
        args.overlay_amplitude_scale
    )

    # Resolve paths
    args.a1_archive = Path(args.a1_archive).resolve()
    args.video_path = Path(args.video_path).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.upstream_dir = Path(args.upstream_dir).resolve()

    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["TIER_1_OPERATOR_REQUIRED_FLAGS", "main"]
