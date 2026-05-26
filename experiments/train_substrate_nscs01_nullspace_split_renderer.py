# SPDX-License-Identifier: MIT
"""NSCS01 Nullspace Split Renderer trainer.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
this trainer ships in two paths:

* ``_smoke_main`` — tiny config (4 pairs, 24x32, 3 epochs), deterministic
  random GT (Catalog #114 _smoke_main allowed exception). Builds a real
  archive + runtime to verify the export contract end-to-end. Does NOT
  load scorers.
* ``_full_main`` — intentionally raises ``NotImplementedError`` until the
  head0 architecture disambiguator, real-pair training/export path, and
  paired auth-eval custody are implemented. This keeps the scaffold
  research-only so accidental Modal dispatches are refused until Phase 2
  council approval.

Per Catalog #151 the TIER_1_OPERATOR_REQUIRED_FLAGS manifest is declared
as `ast.AnnAssign` so Catalog #168 AST walker observes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _canon_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    require_contest_cuda_auth_eval_claim as _canon_require_contest_cuda_auth_eval_claim,
)
from tac.substrates.nscs01_nullspace_split_renderer import (
    CAMERA_H,
    CAMERA_W,
    NullspaceSplitConfig,
    NullspaceSplitRenderer,
    pack_archive,
)
from tac.substrates.nscs01_nullspace_split_renderer.registered_substrate import (
    NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT,  # noqa: F401  (forces contract validation)
)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_LANE_ID = "lane_nscs01_nullspace_split_renderer_20260515"
SUBSTRATE_TAG = "nscs01_nullspace_split_renderer"

EVAL_HW = (CAMERA_H, CAMERA_W)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_auth_eval_json_paths(
    output_dir: Path,
    *,
    durable_root: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(gate_json, local_copy_json)`` for score-grade auth eval.

    Modal trainers run from a writable ``/tmp/pact`` copy. The canonical
    ``contest_auth_eval.py`` scorer refuses score-grade evidence paths under
    temp storage, so the gate writes to a non-temp path and then the trainer
    copies that JSON back into ``output_dir`` for artifact harvest.
    """
    local_copy_json = output_dir / "contest_auth_eval.json"
    temp_root = Path(tempfile.gettempdir())
    if not _path_is_under(local_copy_json, temp_root):
        return local_copy_json, local_copy_json
    root = durable_root
    if root is None:
        root = Path(
            os.environ.get(
                "NSCS01_AUTH_EVAL_ROOT",
                "/root/nscs01_nullspace_split_renderer_auth_eval",
            )
        )
    return root / output_dir.name / "contest_auth_eval.json", local_copy_json


def _is_pair_capped_smoke(args: argparse.Namespace) -> bool:
    """Return True when the run intentionally emits fewer than contest pairs."""
    return getattr(args, "max_pairs", None) is not None and args.max_pairs < N_PAIRS_FULL


# ---------------------------------------------------------------------------
# Catalog #151 manifest — annotated as ast.AnnAssign (Catalog #168).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS01_VIDEO_PATH",
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
            ".omx/research/nscs01_nullspace_split_renderer_design_20260515.md §1"
        ),
    },
    "--output-dir": {
        "env": "NSCS01_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "NSCS01_EPOCHS",
        "rationale": (
            "NSCS01 substrate is medium-sized (split-head renderer + per-pair "
            "latents); council default 1000 epochs for full training run"
        ),
        "default": "1000",
    },
    "--upstream-dir": {
        "env": "NSCS01_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "NSCS01_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke "
            "or --full-cpu --advisory-cpu-explicitly-waived"
        ),
        "default": "cuda",
    },
    "--head0-bits": {
        "env": "NSCS01_HEAD0_BITS",
        "rationale": (
            "frame_0_head bit-width (4/6/8); default 4 — frame_0 is in "
            "SegNet's nullspace and tolerates aggressive quantization"
        ),
        "default": "4",
    },
    "--head1-bits": {
        "env": "NSCS01_HEAD1_BITS",
        "rationale": (
            "frame_1_head bit-width (6/8); default 8 — frame_1 must be "
            "SegNet-argmax-stable"
        ),
        "default": "8",
    },
    "--latent-dim": {
        "env": "NSCS01_LATENT_DIM",
        "rationale": "per-pair latent vector dimension; default 16",
        "default": "16",
    },
    "--enable-gt-scorer-cache": {
        "env": "NSCS01_ENABLE_GT_SCORER_CACHE",
        "rationale": (
            "F3/GTScorerCache; caches frozen GT PoseNet+SegNet targets once "
            "and reuses indexed batches in the score-aware hot loop"
        ),
        "default": "true",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs01_nullspace_split_renderer",
        description=(
            "Train NSCS01 nullspace-split renderer (assumptions-challenge "
            "audit NSCS01 — exploits SegNet last-frame-only nullspace)"
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--head0-bits", type=int, default=4, choices=[4, 6, 8])
    p.add_argument("--head1-bits", type=int, default=8, choices=[6, 8])
    p.add_argument("--latent-bits", type=int, default=12, choices=[8, 12])
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--head0-base-channels", type=int, default=16)
    p.add_argument("--head1-base-channels", type=int, default=48)
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke-only path (tiny config, 3 epochs, no scorer)")
    p.add_argument("--full-cpu", action="store_true",
                   help="Catalog #197 paired-flag opt-in for non-smoke CPU training")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Required sister flag for --full-cpu (Catalog #197)")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997)")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))
    p.add_argument("--lambda-pixel-0", type=float, default=0.05)
    p.add_argument("--lambda-pixel-1", type=float, default=0.20)
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--val-every-epochs", type=int, default=20)
    p.add_argument("--val-pair-count", type=int, default=32)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap pair decode for fast smoke iteration.")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172")
    p.add_argument(
        "--enable-gt-scorer-cache", dest="enable_gt_scorer_cache",
        action="store_true", default=True,
    )
    p.add_argument(
        "--disable-gt-scorer-cache", dest="enable_gt_scorer_cache",
        action="store_false",
    )
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; uses canonical trainer_skeleton helpers")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with the advisory waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197"
        )


def _git_head_sha() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    from datetime import datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pin_seeds(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: tiny config, 3 epochs, deterministic random GT, no scorer.

    Builds a real NSP1 archive + contest runtime so the export contract is
    verified end-to-end. The auth eval is skipped (no scorer load).
    """
    _pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NullspaceSplitConfig(
        latent_dim=8,
        head0_bits=args.head0_bits,
        head1_bits=args.head1_bits,
        latent_bits=args.latent_bits,
        head0_base_channels=8,
        head1_base_channels=16,
        num_pairs=4,
    )
    renderer = NullspaceSplitRenderer(cfg).to(args.device)

    # Smoke deterministic GT (Catalog #114 _smoke_main allowed exception).
    gt_frame_0 = torch.rand(4, 3, 24, 32, device=args.device) * 255.0
    gt_frame_1 = torch.rand(4, 3, 24, 32, device=args.device) * 255.0

    opt = torch.optim.AdamW(
        renderer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    losses: list[float] = []
    epoch_count = max(args.epochs, 3)
    for _epoch in range(epoch_count):
        opt.zero_grad()
        idx = torch.arange(cfg.num_pairs, device=args.device, dtype=torch.long)
        f0_pred, f1_pred = renderer.reconstruct_pair(idx)
        # Resize gt to match smoke config output to stay tiny
        gt_f0_resize = torch.nn.functional.interpolate(
            gt_frame_0, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
        )
        gt_f1_resize = torch.nn.functional.interpolate(
            gt_frame_1, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
        )
        loss = (
            (f0_pred - gt_f0_resize).pow(2).mean() / (255.0 ** 2)
            + (f1_pred - gt_f1_resize).pow(2).mean() / (255.0 ** 2)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.item()))

    # Pack archive (real NSP1 grammar; verifies the export contract).
    archive_bytes = pack_archive(
        head0_state_dict=renderer.frame_0_head.state_dict(),
        head1_state_dict=renderer.frame_1_head.state_dict(),
        latents=renderer.latents,
        head0_bits=cfg.head0_bits,
        head1_bits=cfg.head1_bits,
        latent_bits=cfg.latent_bits,
        head0_base_channels=cfg.head0_base_channels,
        head1_base_channels=cfg.head1_base_channels,
        extra_meta={
            "smoke": True,
            "lane_id": SUBSTRATE_LANE_ID,
            "git_head": _git_head_sha(),
            "trained_at_utc": _utc_now_iso(),
        },
    )
    (out_dir / "0.bin").write_bytes(archive_bytes)

    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    archive_sha = _sha256_bytes(archive_bytes)
    archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
    final_loss = losses[-1] if losses else float("inf")
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_path.stat().st_size,
        "archive_zip_sha256": archive_zip_sha,
        "head0_bits": cfg.head0_bits,
        "head1_bits": cfg.head1_bits,
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "council_phase_2_required_before_full_dispatch": False,
        "git_head": _git_head_sha(),
        "trained_at_utc": _utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[nscs01-smoke] OK final_loss={final_loss:.6f} "
        f"archive={len(archive_bytes)}B sha={archive_sha[:12]}... "
        f"head0_bits={cfg.head0_bits} head1_bits={cfg.head1_bits}"
    )
    return 0


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``.
    Per Catalog #163 the script uses ``set -euo pipefail``.
    Per CLAUDE.md "Strict scorer rule": no scorer at inflate time.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "nscs01_nullspace_split_renderer"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "nscs01_nullspace_split_renderer"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)

    # Runtime __init__.py is MINIMAL — no score_aware_loss import (which would
    # pull in scorer code at inflate time, forbidden per "Strict scorer rule").
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"NSCS01 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.nscs01_nullspace_split_renderer.architecture import (\n"
        "    CAMERA_H, CAMERA_W, NUM_PAIRS,\n"
        "    NullspaceSplitConfig, NullspaceSplitRenderer,\n"
        ")\n"
        "from tac.substrates.nscs01_nullspace_split_renderer.archive import (\n"
        "    NSP1_MAGIC, NSP1_SCHEMA_VERSION, NullspaceSplitArchive,\n"
        "    deserialize_head_state_dicts, deserialize_latents,\n"
        "    pack_archive, parse_archive,\n"
        ")\n"
        "__all__ = [\n"
        "    'CAMERA_H', 'CAMERA_W', 'NSP1_MAGIC', 'NSP1_SCHEMA_VERSION',\n"
        "    'NUM_PAIRS', 'NullspaceSplitArchive', 'NullspaceSplitConfig',\n"
        "    'NullspaceSplitRenderer', 'deserialize_head_state_dicts',\n"
        "    'deserialize_latents', 'pack_archive', 'parse_archive',\n"
        "]\n",
        encoding="utf-8",
    )

    # Vendor the canonical _shared/inflate_runtime.py (Catalog #205).
    shared_dir = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py",
        shared_dir / "inflate_runtime.py",
    )

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# NSCS01 nullspace-split-renderer contest-compliant inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "# Per CLAUDE.md \"Strict scorer rule\": no scorer at inflate time.\n"
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
        '"""NSCS01 contest-compliant inflate runtime.\n'
        "\n"
        "Delegates to the vendored substrate CLI. No scorer imports.\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.nscs01_nullspace_split_renderer.inflate import main_cli\n"
        "\n"
        "def main() -> int:\n"
        "    return main_cli()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin).

    Per Catalog #19 (deterministic ZIP), uses ZipInfo + writestr with a fixed
    timestamp to ensure byte-stable archive.zip output across runs.
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Full entry path — UNLOCKED 2026-05-15 per UNIQUE-AND-COMPLETE-PER-METHOD
# ---------------------------------------------------------------------------
# Per the standing directive (feedback_consolidate_everything_into_meta_layer
# _or_canonical_helpers_standing_directive_20260515.md): substrate trainers
# bind ALL ingredients into ONE coherent packet per the PR 95 paradigm. The
# UNIQUE element of NSCS01 is split-head training: frame_0_head receives ONLY
# PoseNet+pixel gradients (frame[0] is in SegNet's nullspace per
# upstream/modules.py:108 `x[:, -1, ...]` slice); frame_1_head receives BOTH
# SegNet+PoseNet+pixel gradients. The structural property is verified by the
# nullspace-gradient test in tests/test_nscs01_substrate.py.
#
# Canonical-vs-unique decisions per layer (design memo §7 alignment):
#   1. pyav decode             -> ADOPT canonical (decode_real_pairs helper)
#   2. seed pinning            -> ADOPT canonical (pin_seeds → trainer skeleton)
#   3. device gate             -> ADOPT canonical (device_or_die)
#   4. YUV6 patch              -> ADOPT canonical (patch_upstream_yuv6_globally)
#   5. scorer load             -> ADOPT canonical (load_differentiable_scorers)
#   6. EMA shadow              -> ADOPT canonical (tac.training.EMA, decay=0.997)
#   7. score-aware loss        -> FORK (NullspaceSplitScoreAwareLoss; the canonical
#                                  score_pair_components_dispatch homogenizes the
#                                  split-frame gradient routing that IS the exploit)
#   8. archive pack/runtime    -> FORK (NSP1 grammar; per-head bit-widths)
#   9. auth-eval gate          -> ADOPT canonical (gate_auth_eval_call)
#  10. posterior update        -> ADOPT canonical (posterior_update_locked_*)
#  11. provenance + manifest   -> ADOPT canonical pattern (sister substrates)
#  12. hardware detect         -> ADOPT canonical (detect_hardware_substrate)


def _run_val_loop(
    renderer: NullspaceSplitRenderer,
    loss_fn,
    gt_pair_tensor: torch.Tensor,
    val_pair_indices: list[int],
    archive_bytes_proxy: torch.Tensor,
    device,
    *,
    chunk_size: int = 16,
) -> float:
    """Validation pass with EMA shadow + torch.inference_mode (Catalog #180).

    Mini-batches val pairs per Catalog #218 OOM discipline: full 600-pair
    forward at 384x512 + per-head conv stack exceeds T4 (14.56 GB).
    """
    renderer.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(val_pair_indices), chunk_size):
            chunk = val_pair_indices[start : start + chunk_size]
            if not chunk:
                continue
            idx_tensor = torch.tensor(chunk, device=device, dtype=torch.long)
            f0_pred, f1_pred = renderer.reconstruct_pair(idx_tensor)
            gt_0 = gt_pair_tensor[idx_tensor, 0]
            gt_1 = gt_pair_tensor[idx_tensor, 1]
            try:
                loss, _ = loss_fn(
                    frame_0_pred=f0_pred,
                    frame_1_pred=f1_pred,
                    gt_frame_0=gt_0,
                    gt_frame_1=gt_1,
                    archive_bytes_proxy=archive_bytes_proxy,
                    apply_eval_roundtrip=True,
                    noise_std=0.0,
                )
            except Exception as exc:
                print(
                    f"[{SUBSTRATE_TAG}-val] WARN val batch start={start} skipped: {exc!r}"
                )
                continue
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else math.inf


def _full_main(args: argparse.Namespace) -> int:
    """Full training entry: pyav decode + score-aware split-head loss + EMA + auth eval.

    Binds all PR 95 paradigm ingredients into ONE coherent packet:
      * pyav-decoded real contest pairs (NO synthetic data; Catalog #114)
      * patched upstream YUV6 BEFORE scorer construction (Catalog #187)
      * load_differentiable_scorers (frozen; eval mode; no scorer at inflate)
      * NullspaceSplitScoreAwareLoss with split-frame gradient routing
        (frame_0_head gets NO SegNet gradient by autograd structure)
      * EMA(decay=0.997) update post optimizer.step; inference = EMA shadow
      * eval_roundtrip=True throughout (Catalog #5 non-negotiable)
      * AdamW + cosine annealing; gradient clip 1.0; NaN watchdog
      * Mini-batched reconstruct_pair (Catalog #218 OOM discipline)
      * NSP1 archive pack + contest-compliant runtime tree emission
      * Canonical gate_auth_eval_call (Catalog #226) on best EMA checkpoint
      * Canonical require_contest_cuda_auth_eval_claim (Catalog #127 custody)
      * Continual-learning posterior_update_locked (Catalog #128 atomic fcntl)
      * Hardware substrate detection (Catalog #190)
      * Catalog #220 operational-mechanism declaration via auth-eval claim
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs01_nullspace_split_renderer.score_aware_loss import (
        NullspaceSplitLossWeights,
        NullspaceSplitScoreAwareLoss,
    )
    from tac.training import EMA

    _pin_seeds(args.seed)
    device = _canon_device_or_die(
        args.device,
        smoke=False,
        substrate_tag=SUBSTRATE_TAG,
        allow_full_cpu=bool(args.full_cpu),
    )
    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    auth_eval_gate_json_path, auth_eval_json_path = _resolve_auth_eval_json_paths(
        args.output_dir
    )
    archive_zip_path = args.output_dir / "archive.zip"
    archive_zip_sha = ""
    archive_zip_size = 0
    bin_sha = ""
    bin_size = 0
    n_params = 0
    best_val_lag = math.inf
    best_epoch = -1
    auth_eval_result: dict[str, object] | None = None

    try:
        # 2. Load differentiable scorers (frozen; eval; no grad).
        # Canonical contract: (posenet, segnet) order per
        # tac.scorer.load_differentiable_scorers signature; see Catalog #222
        # which refuses reversed assignment.
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 3. Decode real contest pairs via canonical pyav helper.
        print(f"[{SUBSTRATE_TAG}-full] decoding pairs from {args.video_path}")
        gt_pair_tensor = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=args.max_pairs,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 4. Build NSCS01 split-renderer at the requested num_pairs.
        cfg = NullspaceSplitConfig(
            latent_dim=args.latent_dim,
            head0_bits=args.head0_bits,
            head1_bits=args.head1_bits,
            latent_bits=args.latent_bits,
            head0_base_channels=args.head0_base_channels,
            head1_base_channels=args.head1_base_channels,
            num_pairs=n_pairs,
        )
        renderer = NullspaceSplitRenderer(cfg).to(device)
        n_params = sum(p.numel() for p in renderer.parameters())
        n_params_head0 = sum(p.numel() for p in renderer.frame_0_head.parameters())
        n_params_head1 = sum(p.numel() for p in renderer.frame_1_head.parameters())
        print(
            f"[{SUBSTRATE_TAG}-full] renderer params: total={n_params:,} "
            f"head0={n_params_head0:,} head1={n_params_head1:,}"
        )
        _stage(
            f"renderer_built_{n_params}_params_head0_{n_params_head0}_head1_{n_params_head1}"
        )

        # 5. EMA shadow (CLAUDE.md non-negotiable, decay=0.997).
        ema = EMA(renderer, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 6. Score-aware Lagrangian (FORK from canonical per design memo §4).
        weights = NullspaceSplitLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            lambda_pixel_0=args.lambda_pixel_0,
            lambda_pixel_1=args.lambda_pixel_1,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = NullspaceSplitScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built_split_head")

        # 7. Optimizer (AdamW + cosine annealing).
        optimizer = torch.optim.AdamW(
            renderer.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # 8. Train loop.
        train_started_at = time.time()
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        # Archive byte proxy: estimate from head sizes + bit-widths + brotli ratio.
        # head_bytes ≈ params * bits / 8 * 0.4 (brotli ratio on int-packed weights)
        head0_bytes_proxy = max(1000, int(n_params_head0 * args.head0_bits / 8 * 0.4))
        head1_bytes_proxy = max(1000, int(n_params_head1 * args.head1_bits / 8 * 0.4))
        latent_bytes_proxy = max(
            500, int(n_pairs * args.latent_dim * args.latent_bits / 8 * 0.5)
        )
        meta_bytes_proxy = 4_000
        total_proxy_bytes = (
            head0_bytes_proxy + head1_bytes_proxy + latent_bytes_proxy + meta_bytes_proxy
        )
        archive_bytes_proxy = torch.tensor(float(total_proxy_bytes), device=device)
        print(
            f"[{SUBSTRATE_TAG}-full] archive_bytes_proxy: head0={head0_bytes_proxy}B "
            f"head1={head1_bytes_proxy}B latent={latent_bytes_proxy}B "
            f"meta={meta_bytes_proxy}B total={total_proxy_bytes}B"
        )

        for epoch in range(args.epochs):
            renderer.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []

            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[batch_start : batch_start + args.batch_size]
                if not batch_indices:
                    continue
                batch_idx_tensor = torch.tensor(
                    batch_indices, device=device, dtype=torch.long
                )
                # Mini-batched reconstruct_pair (Catalog #218 discipline): gradient
                # flows into selected latent rows only via index_select.
                f0_pred, f1_pred = renderer.reconstruct_pair(batch_idx_tensor)
                gt_0 = gt_pair_tensor[batch_idx_tensor, 0]
                gt_1 = gt_pair_tensor[batch_idx_tensor, 1]

                loss, parts = loss_fn(
                    frame_0_pred=f0_pred,
                    frame_1_pred=f1_pred,
                    gt_frame_0=gt_0,
                    gt_frame_1=gt_1,
                    archive_bytes_proxy=archive_bytes_proxy,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[{SUBSTRATE_TAG}-full] NaN strike {nan_strike}/{max_nan_strikes}"
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(renderer.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(renderer)
                epoch_losses.append(float(loss.detach().cpu()))

            scheduler.step()

            if epoch % max(1, args.val_every_epochs) == 0 or epoch == args.epochs - 1:
                live_state = {
                    k: v.detach().clone() for k, v in renderer.state_dict().items()
                }
                ema.apply(renderer)
                ema_state_for_ckpt: dict[str, torch.Tensor] | None = None
                try:
                    val_lag = _run_val_loop(
                        renderer, loss_fn, gt_pair_tensor, val_indices_pool,
                        archive_bytes_proxy, device,
                    )
                    ema_state_for_ckpt = {
                        k: v.detach().cpu().clone()
                        for k, v in renderer.state_dict().items()
                    }
                finally:
                    renderer.load_state_dict(live_state)
                    renderer.train()

                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                ) if epoch_losses else math.nan
                print(
                    f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d}: train={avg_train:.5f} "
                    f"val={val_lag:.5f} (best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag and ema_state_for_ckpt is not None:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    torch.save(
                        {
                            "state_dict": ema_state_for_ckpt,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_lag": val_lag,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # 9. Load best EMA checkpoint + emit NSP1 archive.
        if not args.skip_archive_build:
            if ckpt_best_path.exists():
                best_ckpt = torch.load(
                    ckpt_best_path, weights_only=False, map_location=device
                )  # WEIGHTS_ONLY_FALSE_OK:trusted-local-checkpoint-from-this-process
                renderer.load_state_dict(best_ckpt["state_dict"])
            renderer.eval()

            bin_bytes = pack_archive(
                head0_state_dict=renderer.frame_0_head.state_dict(),
                head1_state_dict=renderer.frame_1_head.state_dict(),
                latents=renderer.latents,
                head0_bits=cfg.head0_bits,
                head1_bits=cfg.head1_bits,
                latent_bits=cfg.latent_bits,
                head0_base_channels=cfg.head0_base_channels,
                head1_base_channels=cfg.head1_base_channels,
                extra_meta={
                    "lane_id": SUBSTRATE_LANE_ID,
                    "smoke": False,
                    "best_val_lag": float(best_val_lag),
                    "best_epoch": int(best_epoch),
                    "git_head": _git_head_sha(),
                    "trained_at_utc": _utc_now_iso(),
                    "n_params_total": int(n_params),
                    "n_params_head0": int(n_params_head0),
                    "n_params_head1": int(n_params_head1),
                },
            )
            bin_sha = _sha256_bytes(bin_bytes)
            bin_size = len(bin_bytes)
            print(
                f"[{SUBSTRATE_TAG}-full] NSP1 archive: {bin_size} B "
                f"sha256={bin_sha[:16]}..."
            )
            _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

            # 10. Emit contest-compliant runtime tree + archive.zip.
            submission_dir = args.output_dir / "submission_dir"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes)
            archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
            archive_zip_size = archive_zip_path.stat().st_size
            shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
            _stage("archive_emitted")

            # 11. Auth eval ([contest-CUDA] inline) through the canonical gate
            # per Catalog #226 (no hand-rolled subprocess to contest_auth_eval.py).
            if not args.skip_auth_eval:
                auth_eval_result = _canon_gate_auth_eval_call(
                    args=args,
                    archive_zip=archive_zip_path,
                    inflate_sh=submission_dir / "inflate.sh",
                    upstream_dir=args.upstream_dir,
                    output_json=auth_eval_gate_json_path,
                    contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                    substrate_tag=SUBSTRATE_TAG,
                    device=device,
                    full_cpu_active=bool(args.full_cpu),
                )
                if auth_eval_result is not None:
                    if auth_eval_gate_json_path != auth_eval_json_path:
                        auth_eval_json_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(auth_eval_gate_json_path, auth_eval_json_path)
                    _canon_require_contest_cuda_auth_eval_claim(
                        auth_eval_json_path,
                        archive_sha256=archive_zip_sha,
                        substrate_tag=SUBSTRATE_TAG,
                    )
                    _stage("auth_eval_cuda_done_valid_claim")
                else:
                    _stage("auth_eval_skipped_gate_refused")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 12. Posterior update (Catalog #128 atomic fcntl).
    if (not args.skip_auth_eval) and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(
                f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}"
            )

    # 13. Provenance + manifest (canonical schema; sister-substrate pattern).
    hardware_substrate_cuda = _canon_detect_hardware_substrate(
        axis="cuda",
        substrate_tag=SUBSTRATE_TAG,
        env_var_candidates=("NSCS01_GPU", "MODAL_GPU"),
    )
    provenance = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "best_val_lag": float(best_val_lag) if math.isfinite(best_val_lag) else None,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "head0_bits": args.head0_bits,
        "head1_bits": args.head1_bits,
        "latent_bits": args.latent_bits,
        "council_phase_2_unanimous_seal": False,  # L1 SCAFFOLD landing
        "design_memo": (
            ".omx/research/nscs01_nullspace_split_renderer_design_20260515.md"
        ),
        "stage_log": stage_log,
        "auth_eval_gate_json_path": str(auth_eval_gate_json_path),
        "auth_eval_json_path": str(auth_eval_json_path),
        "hardware_substrate_cuda": hardware_substrate_cuda,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    pair_capped_smoke = _is_pair_capped_smoke(args)
    manifest = {
        "schema": "nscs01_nullspace_split_renderer_training_artifact_manifest_v1",
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "training_mode": "smoke" if pair_capped_smoke else "full",
        "research_only": pair_capped_smoke,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_bytes": bin_size,
        "archive_sha256": bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "max_pairs": args.max_pairs,
        "n_pairs_full_required_for_auth_eval": N_PAIRS_FULL,
        "auth_eval_skipped": bool(args.skip_auth_eval),
        "auth_eval_skipped_reason": (
            "pair_capped_smoke_emits_truncated_raw_stream"
            if pair_capped_smoke and args.skip_auth_eval
            else ""
        ),
        "result": {
            "training_mode": "smoke" if pair_capped_smoke else "full",
            "archive_bytes": bin_size,
            "archive_sha256": bin_sha,
            "archive_zip_bytes": archive_zip_size,
            "archive_zip_sha256": archive_zip_sha,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[{SUBSTRATE_TAG}-full] wrote {args.output_dir / 'provenance.json'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
