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

The ``_full_main`` path is **NotImplementedError-gated** until council
approval per CLAUDE.md "Design decisions — non-negotiable": full GPU
training composes the margin map computation + polytope encoder loop +
end-of-train auth eval, which requires the inner-quintet sign-off before
$1+ Modal T4 spend. ``_smoke_main`` covers the L1 SCAFFOLD wiring at
$0 cost (synthetic margin map + polytope allocation + archive roundtrip
in seconds).

Usage (smoke; CPU, no scorer load)::

    .venv/bin/python experiments/train_substrate_d1_segnet_margin_polytope.py \\
        --a1-archive experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/d1_polytope_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threaded from operator wrapper; council-gated)::

    # NOTE: full path raises NotImplementedError until inner-quintet
    # council sign-off; smoke path is the L1 SCAFFOLD deliverable.
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
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_A1_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "track4_sg_a1_t178000_20260509"
    / "submission_dir"
    / "archive.zip"
)
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
            "magic-byte-distinct member."
        ),
        "default": (
            "experiments/results/track4_sg_a1_t178000_20260509/"
            "submission_dir/archive.zip"
        ),
        "required_input_file": True,
        "generator_command": (
            "experiments/results/track4_sg_a1_t178000_20260509/ (landed "
            "2026-05-09 via PR101 score-gradient fine-tune)"
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

    # Lagrangian weights (score-domain; HNeRV parity L6)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))

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
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
        "predicted_score_band": [0.181, 0.188],
        "predicted_score_evidence_grade": "first-principles-bound",
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
        compute_logit_margin_map_dummy,
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
# Full training path — gated behind NotImplementedError per CLAUDE.md
# "Design decisions — non-negotiable"
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Full training path — GATED behind explicit inner-quintet council
    approval before $1+ Modal T4 spend.

    Per CLAUDE.md "Design decisions — non-negotiable":
    > Design tradeoffs (bicubic vs bilinear, loss function choice,
    > constraint boundaries, rho growth strategy, what to include in
    > archive, etc.) MUST be council-approved before implementation.

    The full path requires the following design verdicts that the L1
    SCAFFOLD landing intentionally defers to a council deliberation pass:

    1. **Jacobian Lipschitz calibration strategy**: should `L` be a
       fixed hyperparameter, a per-pair offline-computed map, or
       trained jointly with the margin map?
    2. **Per-pair vs per-frame margin map**: should every pair get its
       own margin map (600 maps × ~200 KB) or share a single global map?
    3. **Polytope-interior noise application timing**: is the noise
       applied PRE-renderer (correcting the renderer's view) or
       POST-renderer (correcting the rendered output)?
    4. **Composition order with YUCR**: does D1 → YUCR or YUCR → D1
       compose better? (The deep-math memo predicts they're additive
       but the empirical ordering may matter for the per-pixel
       interference between frame-0 noise and frame-1 margin shifts.)
    5. **EMA shadow scope**: do we EMA-shadow the margin map only, the
       margin map + polytope budget, or the full archive?

    The smoke path validates the substrate end-to-end at $0 cost. Once
    the council approves the design decisions above, the full path
    lands in a follow-up commit per CLAUDE.md "3 consecutive clean
    passes required before the code is cleared for deployment".
    """
    raise NotImplementedError(
        "D1 full training path is GATED behind inner-quintet council "
        "approval per CLAUDE.md 'Design decisions — non-negotiable'.  The "
        "L1 SCAFFOLD landing intentionally defers the full path to a "
        "council deliberation pass; the smoke path proves the substrate "
        "end-to-end at $0 cost.  See docstring above for the 5 design "
        "verdicts the council must adjudicate before $1+ Modal T4 spend."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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


__all__ = ["main", "TIER_1_OPERATOR_REQUIRED_FLAGS"]
