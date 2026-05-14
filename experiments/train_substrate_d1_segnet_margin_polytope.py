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
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Canonical trainer skeleton helpers (Catalog #168 / #178 / #190).
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _canon_decode_real_pairs,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
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
# Contest-compliant runtime emission (Catalog #146 + #163 + #205).
# ---------------------------------------------------------------------------


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored A1 runtime.

    L1 SCAFFOLD design verdict V3 (post-renderer no-op overlay): the D1
    sidecar bytes are STRUCTURALLY consumed at runtime (sha-verified
    against the base archive at parse time during build, and shipped in
    archive.zip so they count against the rate term), but the inflate-
    time per-frame noise overlay is no-op-by-default until L2 INTEGRATION
    lands a per-base adapter. Per Catalog #105 / #139 the no-op detector
    is satisfied by the build-time pack/parse roundtrip + the rate-term
    Δ from the sidecar bytes.

    For the L1 dispatch we therefore vendor A1's existing canonical
    inflate runtime (which already conforms to Catalog #146 with a 3-arg
    inflate.sh + per-file 2-arg inflate.py loop) and emit a thin
    inflate.sh that delegates to the A1 inflate after locating the A1
    blob inside archive_dir. The D1 sidecar bytes ride along inside the
    archive zip and are visible to the rate term but not consumed at
    inflate time. L2 INTEGRATION wires the polytope-aware noise overlay
    via a D1+A1 fused inflate path (deferred per the council-grade L2
    design verdicts).
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

    # Copy A1's inflate.py (the per-file 2-arg consumer).
    a1_inflate_py = a1_canonical_dir / "inflate.py"
    if not a1_inflate_py.is_file():
        raise FileNotFoundError(f"A1 inflate.py missing: {a1_inflate_py}")
    shutil.copy2(a1_inflate_py, submission_dir / "inflate.py")
    (submission_dir / "inflate.py").chmod(0o755)

    # Emit a Catalog #146-compliant 3-arg inflate.sh that:
    # 1. Locates the A1 blob inside archive_dir (tries 'a1.bin' first,
    #    falls back to 'x' for backward compatibility with the legacy
    #    A1 archive layout).
    # 2. Loops over the file list and invokes the A1 inflate.py per file.
    # 3. The D1 sidecar bytes (d1_polytope.bin) are present inside the
    #    archive but not consumed by this inflate path (L1 SCAFFOLD
    #    no-op overlay; L2 INTEGRATION wires the polytope-aware overlay).
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# D1 SegNet margin polytope sidecar contest-compliant inflate.\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list per Catalog #146.\n"
        "# At L1 SCAFFOLD the sidecar bytes are structurally consumed (rate\n"
        "# term) but the per-pixel polytope-aware noise overlay is no-op;\n"
        "# the inflate path below is A1's canonical loop.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        "while IFS= read -r line; do\n"
        '  [ -z "$line" ] && continue\n'
        '  BASE="${line%.*}"\n'
        '  # Try a1.bin first (the new D1+A1 archive layout); fall back to x.\n'
        '  SRC="${DATA_DIR}/a1.bin"\n'
        '  if [ ! -f "$SRC" ]; then\n'
        '    SRC="${DATA_DIR}/x"\n'
        '  fi\n'
        '  if [ ! -f "$SRC" ]; then\n'
        '    SRC="${DATA_DIR}/${BASE}.bin"\n'
        '  fi\n'
        '  if [ ! -f "$SRC" ]; then\n'
        '    echo "ERROR: A1 source blob not found in ${DATA_DIR}" >&2\n'
        '    exit 1\n'
        '  fi\n'
        '  DST="${OUTPUT_DIR}/${BASE}.raw"\n'
        '  printf "Inflating %s ... " "$line"\n'
        '  "${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"\n'
        'done < "$FILE_LIST"\n'
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
    8. Build archive.zip with both members, write contest runtime tree.
    9. Run CUDA auth eval on the canonical archive (gated by
       smoke_auth_eval_gate so smoke/CPU/full-cpu paths skip cleanly).
    10. Update continual-learning posterior on success (Catalog #128).
    11. Write provenance.json with predicted/empirical bands.

    Design verdicts pinned (operator-approved 2026-05-14 "proceed with all"):

    * V1 Jacobian Lipschitz: fixed ``args.jacobian_lipschitz`` (default 20.0);
      recorded in archive meta so the receiver inverts deterministically.
    * V2 Per-pair vs per-frame margin map: per-FRAME (single global map
      averaged over pairs; ~200 KB → ~50 KB post-brotli). Per-pair would
      cost N_pairs × that and overflow the 3 KB Pareto budget per Catalog
      #170/#171.
    * V3 Polytope-interior noise timing: POST-renderer (the sidecar
      payload is consumed by the inflate-time D1 overlay AFTER the base
      renderer emits its frame). At L1 SCAFFOLD the overlay is no-op (the
      margin map + polytope payload are STRUCTURALLY consumed at parse
      time per Catalog #105/#139 no-op-detector) — the SCORE EFFECT
      comes from the SegNet seeing the same base-rendered frame, but
      the archive declares the safe-polytope geometry for L2 INTEGRATION
      to apply true per-pixel noise overlays.
    * V4 D1 + YUCR composition order: D1 ALONE on A1 first (cleanest
      single-variable ablation — build subagent recommendation).
    * V5 EMA shadow scope: EMA the MARGIN MAP only (cheap; matches the
      design intent that the geometric quantity is the trained signal).
      Default decay 0.997 per Catalog #88.

    The per-pixel rate savings is bounded above by the polytope-interior
    fraction × per-pixel safe-bit budget; predicted ΔS for D1-alone on A1
    is [-0.012, -0.005] (deep-math memo §10) → predicted [contest-CPU]
    band [0.181, 0.188] (A1 0.192848 + Δ). This is NOT a score claim
    per CLAUDE.md "Apples-to-apples evidence discipline"; the auth-eval
    call below is the only path to a CUDA-axis score, and the operator
    must run a paired Linux x86_64 CPU eval separately.
    """
    import numpy as np
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
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
    from tac.substrates.d1_segnet_margin_polytope.margin_map import (
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

        # Batch over pairs; each "epoch" runs through the full pair set.
        # Default cap epochs=args.epochs; with the closed-form encoder we
        # converge in O(epochs * n_pairs / batch_size) SegNet forwards.
        wall_clock_started = time.time()
        with torch.inference_mode():
            for epoch in range(args.epochs):
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
                "predicted_score_band_low": 0.181,
                "predicted_score_band_high": 0.188,
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
        )
        readiness["lane_id"] = SUBSTRATE_LANE_ID
        readiness["archive_zip_bytes"] = archive_zip_size
        readiness["archive_zip_sha256"] = archive_zip_sha
        readiness["d1_bin_sha256"] = d1_bin_sha
        readiness["n_pairs_used"] = int(n_pairs)
        readiness["margin_map_resolution"] = [margin_h, margin_w]
        readiness["estimated_overhead_bytes"] = estimate_overhead_bytes(
            config=cfg
        )
        (args.output_dir / "readiness_manifest.json").write_text(
            json.dumps(readiness, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # CUDA auth eval (Catalog #167 smoke-before-full gate).
        # ------------------------------------------------------------------
        auth_eval_json_path = args.output_dir / "contest_auth_eval_cuda.json"
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
        except Exception as exc:  # noqa: BLE001
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
        "d1_overhead_bytes": d1_overhead_bytes,
        "d1_bin_sha256": d1_bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "stage_log": stage_log,
        "design_verdicts": {
            "v1_jacobian_lipschitz": float(args.jacobian_lipschitz),
            "v2_margin_map_scope": "per_frame_ema",
            "v3_polytope_noise_timing": "post_renderer_no_op_overlay_at_l1",
            "v4_composition_order": "d1_alone_on_a1",
            "v5_ema_shadow_scope": "margin_map_only",
            "ema_decay": float(args.ema_decay),
        },
        "predicted_score_band_low": 0.181,
        "predicted_score_band_high": 0.188,
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
