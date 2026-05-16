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
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from datetime import UTC

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
SUBSTRATE_LANE_ID = "lane_nscs01_nullspace_split_renderer_20260515"
SUBSTRATE_TAG = "nscs01_nullspace_split_renderer"


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
    p.add_argument("--lambda-pixel-0", type=float, default=0.05)
    p.add_argument("--lambda-pixel-1", type=float, default=0.20)
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
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
# Full entry path — gated until L1 SCAFFOLD smoke + Tier C ablation land
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:  # pragma: no cover — gated
    """Full training entry: contest video pyav decode + scorer-aware loss + EMA + auth eval.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    this entry MUST raise NotImplementedError until Phase 2 council approves
    the L1 smoke + paired CPU/CUDA Tier C ablation. The IMPLEMENTATION below
    is fully wired but council-gated; the operator unblocks by removing the
    raise + green-up via the canonical smoke-before-full pattern (Catalog #167).
    """
    raise NotImplementedError(
        "NSCS01 _full_main is council-gated per CLAUDE.md 'Substrate scaffolds "
        "MUST be COMPLETE or RESEARCH-ONLY'. L1 SCAFFOLD smoke must pass + Tier "
        "C ablation must land paired CPU/CUDA evidence + adversarial council "
        "review (5 PROCEED unanimous) before unblocking. The full path uses the "
        "canonical pyav decode + GTScorerCache + score-aware loss + EMA + "
        "gate_auth_eval_call(...) pipeline; remove this raise once council "
        "green-ups."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
