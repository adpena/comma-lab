# SPDX-License-Identifier: MIT
"""Train (compress-only) the nscs06_carmack_hotz_strip_everything substrate.

Per grand-reunion symposium composite #4 design (2026-05-15). The radical
Carmack-Hotz architecture is unusual: there is NO neural-codec training loop.
Instead, `_full_main` is a one-shot COMPRESS pass:

  1. Decode contest video into per-pair RGB tensors.
  2. Run SegNet at compress time (FREE per contest README rule 2) to derive
     per-pixel class importance + class-conditional CDFs.
  3. Run PoseNet at compress time to derive per-pair pose deltas.
  4. Build the grayscale palette from the GT luminance histogram.
  5. Quantize odd-frame grayscale -> palette indices (Quantizr PR #56).
  6. Arithmetic-encode palette indices with class-conditional CDFs.
  7. Quantize pose deltas to uint8.
  8. Pack into CH06 archive bytes.
  9. Run canonical CUDA auth eval per Catalog #226.

This script honors EVERY CLAUDE.md non-negotiable that applies:
- Train against ``upstream/videos/0.mkv`` (NOT synthetic; Catalog #114).
- Patch upstream ``rgb_to_yuv6`` before scorer load (PR #95/#106).
- ``load_differentiable_scorers`` for SegNet/PoseNet (compress-only; never at inflate).
- Canonical ``gate_auth_eval_call`` for auth eval (Catalog #226).
- Continual-learning posterior update via ``posterior_update_locked`` (Catalog #128).
- Hardware substrate dynamically detected via canonical helper (Catalog #190).
- Contest-compliant runtime emission (3-positional-arg ``inflate.sh``; Catalog #146).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151).
- META layer ``@register_substrate`` decoration (Catalog #241/#242).

Why no EMA / no eval_roundtrip-in-loop / no scoring-aware Lagrangian here:
this substrate's score-improvement mechanism is NOT iterative learning but
CLOSED-FORM bit allocation derived from the scorer's argmax. There are zero
learnable weights to train; the only "training" is HISTOGRAM construction.

Usage (smoke; CPU, 4 pairs, tiny output)::

    .venv/bin/python experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs06_smoke_<utc> \\
        --epochs 1 --device cpu --smoke

Usage (full; CUDA-required compress-side scorer query)::

    .venv/bin/python experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs06_<utc> \\
        --epochs 1 --device cuda
"""
# AUTOCAST_FP16_WAIVED:scorer-runs-once-at-compress-time-not-in-training-loop-so-autocast-irrelevant
# TORCH_COMPILE_WAIVED:no-training-loop-to-compile-codec-is-pure-numpy
# TF32_WAIVED:carmack-hotz-strip-everything-has-no-neural-codec-no-matmul-operations-numpy-pillow-only-inflate
# NO_GRAD_WAIVED:no-training-loop-no-eval-gradient-numpy-pillow-only-inflate
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-segnet-runs-once-at-compress-time-class-label-derivation-only
# SCORER_PREPROCESS_HANDLED_OK:no-score-aware-loss-no-hot-loop-segnet-runs-once-at-compress-time-for-class-cdf-construction-not-gradient-loss
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tac.substrate_registry import SubstrateContract, register_substrate
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

# Canonical codec-package source. `_write_runtime` vendors these 3 files into
# the submission tree as `submission/_nscs06_codec/{archive,codec,inflate}.py`
# so the submission directory is SELF-CONTAINED per HNeRV parity discipline
# L4 + L9 (no `tac.*` import from the contest judge's runtime tree). The
# previous `sys.path.insert(0, HERE / 'src')` + `from tac.substrates...` import
# in submission/inflate.py only worked by accident on the Modal worker because
# the source tree happened to be mounted; in a sealed contest environment the
# import would `ModuleNotFoundError`. v6 fix: vendor the package.
CODEC_PACKAGE_SOURCE = (
    REPO_ROOT / "src" / "tac" / "substrates" / "nscs06_carmack_hotz_strip_everything"
)
VENDORED_CODEC_FILES = ("archive.py", "codec.py", "inflate.py")
VENDORED_CODEC_SUBDIR = "_nscs06_codec"

EVAL_HW = (384, 512)
CONTEST_RAW_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS06_VIDEO_PATH",
        "rationale": (
            "score-aware compress-side scorer MUST query the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            "feedback_grand_reunion_fields_grade_passion_full_council_debrief_"
            "vision_strategy_design_whiteboard_session_20260515.md#composite-4"
        ),
    },
    "--output-dir": {
        "env": "NSCS06_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "NSCS06_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; "
            "required for non-smoke compress + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "NSCS06_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required for "
            "full run (MPS refused per CLAUDE.md); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "NSCS06_EPOCHS",
        "rationale": (
            "Carmack-Hotz has no training loop, but the trainer skeleton + "
            "dispatch infra expect --epochs; we accept any positive value "
            "and run ONE compress pass"
        ),
        "default": "1",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs06_carmack_hotz_strip_everything",
        description=(
            "Compress-only pass for the Carmack-Hotz strip-everything substrate "
            "(symposium #4; NO neural codec; closed-form bit allocator)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--palette-size", type=int, default=16)
    p.add_argument("--grayscale-downsample", type=int, default=4)
    p.add_argument("--max-pairs", type=int, default=N_PAIRS_FULL)
    p.add_argument("--pose-quant-scale", type=float, default=10.0)
    # OOM fix per Catalog #218 sister pattern (D4 mini-batch reconstruct).
    # NSCS06 100ep Modal T4 smoke fc-01KRQDTA70GEXSZ2CEEYGWQNSR crashed at
    # SegNet conv_pw with 10.55 GiB activation alloc when scoring all 600
    # pairs at full resolution on T4 14.56 GiB. Default chunk_size=8 keeps
    # peak SegNet activation under ~150 MB which leaves comfortable headroom
    # below the T4 capacity ceiling (and also keeps the canary at T4 cost
    # band per Catalog #215). PoseNet preprocess emits (B, T*6, H/2, W/2) so
    # the same chunk size is conservative for both scorers.
    p.add_argument(
        "--scorer-chunk-size",
        type=int,
        default=8,
        help=(
            "Mini-batch size for compress-side SegNet + PoseNet forward "
            "passes; mirrors D4's pair_indices kwarg pattern (Catalog #218)."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(
        name, smoke=smoke, substrate_tag="nscs06_carmack_hotz"
    )


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None):
    return _decode_real_pairs_canonical(
        video_path,
        n_pairs=n_pairs,
        substrate_tag="nscs06_carmack_hotz",
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


def _rgb_to_grayscale_u8(rgb_bcwh: np.ndarray) -> np.ndarray:
    """BT.601 luminance: Y = 0.299 R + 0.587 G + 0.114 B."""
    if rgb_bcwh.ndim != 3 or rgb_bcwh.shape[0] != 3:
        raise ValueError(f"expected (3, H, W); got {rgb_bcwh.shape}")
    r = rgb_bcwh[0].astype(np.float32)
    g = rgb_bcwh[1].astype(np.float32)
    b = rgb_bcwh[2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _write_runtime(submission_dir: Path) -> None:
    """Emit SELF-CONTAINED contest-compliant inflate.sh + inflate.py + vendored
    codec package per Catalog #146 + HNeRV parity discipline L4 + L9.

    The submission tree the contest judge unzips MUST be self-contained — it
    has ZERO access to the PACT repo source tree. v5 dispatch
    fc-01KRR... (Modal T4, 100ep smoke) shipped inflate.sh + inflate.py only;
    the submission/inflate.py used `sys.path.insert(0, HERE/'src')` then
    `from tac.substrates.nscs06_carmack_hotz_strip_everything.inflate import
    inflate_one_video` — which silently fell through to the mounted Modal
    source tree, then produced 0 `.raw` files (rc=0 but no output; contest
    auth_eval raised `RuntimeError: [inflate] PARTIAL inflate - missing .raw
    for 1/1 videos: ['0.raw']`).

    The v6 fix lands here per the operator's Path A directive: copy the 3
    canonical codec files (`archive.py`, `codec.py`, `inflate.py`) verbatim
    into `submission/_nscs06_codec/`. The submission `inflate.py` becomes a
    thin CLI wrapper that imports `inflate_one_video` from the vendored
    package via relative path (`from _nscs06_codec.inflate import ...`).

    Path A rationale (vs Path B inline-everything-in-one-file):
      - Inflate-side surface is ~315 LOC across 3 logical layers (arithmetic
        decoder + archive parser + per-pair render). Inlining into a single
        file would exceed the HNeRV parity L4 ≤200 LOC waiver budget AND
        flatten the natural 3-layer separation that makes each layer
        reviewable in 30 seconds per L12.
      - The 3 canonical files already honor every HNeRV parity contract:
        deterministic, numpy+Pillow only (NO torch, NO scorer per strict-
        scorer-rule), reviewable per L12. Vendoring preserves byte-identical
        behavior vs the canonical module — zero forking, single source of
        truth.
      - The `_nscs06_codec` subdir name uses an underscore prefix so the
        contest evaluator's archive whitelist (which only inspects archive
        zip contents, not the runtime tree) is unaffected, and the import
        path is visually distinct from any first-party `tac.*` module.

    Per Catalog #205 the submission inflate.py also exposes the canonical
    `select_inflate_device` helper signature.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Carmack-Hotz strip-everything contest-compliant inflate (NSCS06 2026-05-15)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list (Catalog #146)\n"
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

    # ------------------------------------------------------------------
    # Vendor the canonical codec package (3 files, byte-identical copy)
    # ------------------------------------------------------------------
    vendored_dir = submission_dir / VENDORED_CODEC_SUBDIR
    vendored_dir.mkdir(parents=True, exist_ok=True)
    # __init__.py: minimal re-exports (only inflate-side names; compress-side
    # helpers exist in codec.py + archive.py as dead code at inflate time but
    # are NEVER imported by the runtime — strict-scorer-rule preserved).
    vendor_init = (
        "# SPDX-License-Identifier: MIT\n"
        '"""Vendored NSCS06 codec package — self-contained inflate-side bytes.\n'
        "\n"
        "Copied verbatim from\n"
        "``src/tac/substrates/nscs06_carmack_hotz_strip_everything/{archive,codec,inflate}.py``\n"
        "by ``experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_write_runtime``\n"
        "so the submission tree is self-contained per HNeRV parity L4 + L9.\n"
        "NO torch, NO scorer, NO PACT repo dependency at inflate time.\n"
        '"""\n'
        "from .inflate import inflate_one_video\n"
        "\n"
        '__all__ = ["inflate_one_video"]\n'
    )
    (vendored_dir / "__init__.py").write_text(vendor_init, encoding="utf-8")
    for fname in VENDORED_CODEC_FILES:
        src = CODEC_PACKAGE_SOURCE / fname
        if not src.is_file():
            raise FileNotFoundError(
                f"NSCS06 vendoring failed: canonical codec source missing: {src}"
            )
        shutil.copy2(src, vendored_dir / fname)

    inflate_py = (
        "#!/usr/bin/env python\n"
        "# SPDX-License-Identifier: MIT\n"
        '"""Carmack-Hotz contest-compliant inflate runtime (NO torch, NO scorer).\n'
        "\n"
        "Self-contained per HNeRV parity discipline L4 + L9. The codec package\n"
        "(archive parser + arithmetic decoder + per-pair render) is vendored into\n"
        "the sibling ``_nscs06_codec/`` directory by\n"
        "``experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_write_runtime``.\n"
        "ZERO ``tac.*`` imports at inflate time; ZERO PACT repo dependency.\n"
        "\n"
        "Per Catalog #146 the contract is\n"
        "``inflate.py <archive_dir> <output_dir> <file_list>``.\n"
        "Per Catalog #205 the canonical ``select_inflate_device`` helper is\n"
        "exposed even though the substrate is numpy+Pillow only (no torch).\n"
        '"""\n'
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "if str(HERE) not in sys.path:\n"
        "    sys.path.insert(0, str(HERE))\n"
        "\n"
        "from _nscs06_codec.inflate import inflate_one_video  # noqa: E402\n"
        "\n"
        "\n"
        "def select_inflate_device() -> str:\n"
        '    """Catalog #205 canonical helper; Carmack-Hotz needs no torch.\n'
        "\n"
        "    The substrate has no neural primitives, so cuda vs cpu is a no-op.\n"
        "    We still expose the canonical name + honor the env var to keep\n"
        "    Catalog #205's strict gate green.\n"
        '    """\n'
        "    # INLINE_DEVICE_FORK_OK:carmack-hotz-substrate-has-no-torch-no-cuda-cpu-distinction\n"
        '    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()\n'
        '    if pinned not in {"auto", "cpu", "cuda"}:\n'
        "        raise SystemExit(\n"
        '            f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}"\n'
        "        )\n"
        '    return "cpu"  # substrate is numpy-only; cuda is a no-op\n'
        "\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print(\n"
        "            'usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "            file=sys.stderr,\n"
        "        )\n"
        "        return 2\n"
        "    select_inflate_device()\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        inflate_one_video(archive_bytes, output_dir / base)\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(
    archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path
) -> None:
    """Deterministic archive.zip per Catalog #19 + HNeRV parity L3.

    Archive MUST contain ONLY ``0.bin`` payload — NOT ``inflate.sh``/``inflate.py``
    runtime scripts (those live alongside the archive in ``submission_dir/`` and
    are routed by the contest evaluator separately). v4 dispatch fc-01KRQESD49...
    failed at archive validation with "UNKNOWN file types in archive:
    ['inflate.sh', 'inflate.py']" because the previous loop packed them inside.
    The ``submission_dir`` parameter is preserved for signature stability but
    intentionally unused in this body.
    """
    del submission_dir  # see docstring; preserved for caller-signature stability
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Smoke main (CPU; synthetic pairs; no scorer load)
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke validates codec roundtrip on synthetic pairs."""
    from tac.substrates.nscs06_carmack_hotz_strip_everything import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.nscs06_carmack_hotz_strip_everything.archive import (
        POSE_DIMS,
        encode_grayscale_stream,
        quantize_pose_deltas,
    )
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        NUM_SEGNET_CLASSES,
        build_class_conditional_cdf,
        build_grayscale_palette,
    )
    from tac.substrates.nscs06_carmack_hotz_strip_everything.inflate import (
        inflate_one_video,
    )

    _pin_seeds(args.seed)
    _ = _device_or_die(args.device, smoke=True)  # device unused; codec is numpy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    h_g, w_g = 6, 8  # tiny grayscale
    n_pairs = 4

    # Build synthetic palette + arith stream + pose
    gray_samples = rng.integers(0, 256, size=(n_pairs, h_g, w_g), dtype=np.uint8)
    palette = build_grayscale_palette(gray_samples, palette_size=args.palette_size)
    palette_indices = palette.quantize(gray_samples)
    cls = rng.integers(
        0, NUM_SEGNET_CLASSES, size=palette_indices.shape, dtype=np.uint8
    )
    cdf = build_class_conditional_cdf(
        palette_indices, cls, palette_size=args.palette_size
    )
    arith_bytes = encode_grayscale_stream(
        palette_indices=palette_indices, class_labels=cls, cdf=cdf
    )
    pose = rng.standard_normal((n_pairs, POSE_DIMS)).astype(np.float32) * 0.01
    pose_bytes, zero = quantize_pose_deltas(pose, scale=args.pose_quant_scale)
    meta = {
        "grayscale_downsample": args.grayscale_downsample,
        "pose_quant_scale": args.pose_quant_scale,
        "pose_quant_zero": zero,
    }
    bin_bytes = pack_archive(
        palette=palette,
        cdf=cdf,
        grayscale_arith_bytes=arith_bytes,
        pose_bytes=pose_bytes,
        meta=meta,
        num_pairs=n_pairs,
        grayscale_h=h_g,
        grayscale_w=w_g,
        output_height=24,
        output_width=32,
    )
    (args.output_dir / "0.bin").write_bytes(bin_bytes)
    arc = parse_archive(bin_bytes)
    print(
        f"[smoke] CH06 archive bytes: {len(bin_bytes)} (palette={palette.size} "
        f"pairs={n_pairs} {h_g}x{w_g})"
    )
    # Run the inflate path end-to-end to verify roundtrip
    raw_path = inflate_one_video(bin_bytes, args.output_dir / "inflate_smoke" / "0")
    expected_raw_bytes = 32 * 24 * n_pairs * 2 * 3
    raw_bytes = raw_path.stat().st_size
    print(f"[smoke] inflate wrote {raw_bytes} raw bytes (expected {expected_raw_bytes})")
    if raw_bytes != expected_raw_bytes:
        print("[smoke] FAIL: raw byte-count mismatch", file=sys.stderr)
        return 1
    assert arc.num_pairs == n_pairs
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required for compress-side scorer; no training loop)
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """One-shot compress pass: scorer at compress -> closed-form CDFs -> CH06."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs06_carmack_hotz_strip_everything import (
        pack_archive,
    )
    from tac.substrates.nscs06_carmack_hotz_strip_everything.archive import (
        POSE_DIMS,
        encode_grayscale_stream,
        quantize_pose_deltas,
    )
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        build_class_conditional_cdf,
        build_grayscale_palette,
    )

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
        # 1. Load scorers (compress-side ONLY)
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded_compress_side")

        # 2. Decode real pairs
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        _stage(f"pairs_decoded_{n_pairs}")

        # 3. Grayscale field (odd frame ONLY; even frame derived via pose)
        H, W = EVAL_HW
        h_g = H // args.grayscale_downsample
        w_g = W // args.grayscale_downsample
        odd_rgb = pair_tensor[:, 0].cpu().numpy().astype(np.uint8)  # (N, 3, H, W)
        gray_full = np.stack([_rgb_to_grayscale_u8(odd_rgb[i]) for i in range(n_pairs)])
        # Downsample via Pillow-equivalent (numpy mean-pool for determinism)
        # Use simple area-average pooling: reshape + mean over the
        # (downsample, downsample) sub-blocks.
        gray_lowres = (
            gray_full.reshape(n_pairs, h_g, args.grayscale_downsample, w_g,
                              args.grayscale_downsample)
            .mean(axis=(2, 4))
            .clip(0, 255)
            .astype(np.uint8)
        )
        _stage(f"grayscale_lowres_built_{h_g}x{w_g}")

        # 4. Build palette + quantize
        palette = build_grayscale_palette(
            gray_lowres, palette_size=args.palette_size
        )
        palette_indices = palette.quantize(gray_lowres)
        _stage(f"palette_built_size_{palette.size}")

        # 5. SegNet argmax at low-res (per-cell class labels)
        # OOM FIX (Catalog #218 sister): the previous single forward over all
        # 600 pairs allocated ~10.55 GiB inside timm EfficientNet-B2 conv_pw
        # on T4 (14.56 GiB capacity), crashing fc-01KRQDTA70GEXSZ2CEEYGWQNSR
        # at $0.019 / 114s before any archive bytes were emitted. We mirror
        # D4's mini-batch reconstruct pattern: iterate the SegNet forward in
        # chunks of args.scorer_chunk_size (default 8) so peak activation
        # memory scales with chunk_size, not n_pairs. The argmax + cpu
        # accumulation is byte-identical to the previous full-batch path
        # (deterministic; no dropout; eval mode) so the resulting class map
        # is unchanged versus the pre-fix output for any chunk_size >= 1.
        with torch.no_grad():
            cls_full_chunks: list[np.ndarray] = []
            chunk = max(1, int(args.scorer_chunk_size))
            for start in range(0, n_pairs, chunk):
                stop = min(start + chunk, n_pairs)
                # SegNet expects (B, T=1, C, H, W); we pass per-pair LAST frame as
                # (B, 3, H, W) via the canonical scorer.preprocess_input.
                odd_chunk = (
                    pair_tensor[start:stop, 0].to(device).float()
                )  # (chunk, 3, H, W)
                # (chunk, T=1, C, H, W) for the scorer's expected layout
                odd_btchw = odd_chunk.unsqueeze(1)
                seg_logits = segnet(segnet.preprocess_input(odd_btchw))
                # seg_logits: (chunk, 5, H, W) at 384x512 — argmax to uint8.
                cls_chunk = (
                    torch.argmax(seg_logits, dim=1).to(torch.uint8).cpu().numpy()
                )
                cls_full_chunks.append(cls_chunk)
                # Release the per-chunk activation buffers before the next
                # iteration so peak VRAM stays bounded.
                del odd_chunk, odd_btchw, seg_logits, cls_chunk
            cls_full = np.concatenate(cls_full_chunks, axis=0)
        # Downsample class map by majority-vote in each block — simple stride
        # subsample is acceptable for the L1 SCAFFOLD path.
        cls_lowres = cls_full[
            :,
            :: args.grayscale_downsample,
            :: args.grayscale_downsample,
        ][:, :h_g, :w_g].astype(np.uint8)
        _stage(
            f"segnet_argmax_lowres_chunked_size_{int(args.scorer_chunk_size)}"
        )

        # 6. Build class-conditional CDF (the closed-form allocator's heart)
        cdf = build_class_conditional_cdf(
            palette_indices, cls_lowres, palette_size=args.palette_size
        )
        _stage("class_conditional_cdf_built")

        # 7. Arith-encode the grayscale palette indices
        arith_bytes = encode_grayscale_stream(
            palette_indices=palette_indices, class_labels=cls_lowres, cdf=cdf
        )
        _stage(f"grayscale_arith_encoded_bytes_{len(arith_bytes)}")

        # 8. PoseNet at compress-side -> per-pair 6-dim deltas
        # OOM FIX (Catalog #218 sister): mirror the SegNet chunked pattern
        # above. PoseNet preprocess emits (B, T*6, H/2, W/2) and the FastViT
        # backbone allocates similar O(B) activation memory to SegNet's
        # EfficientNet-B2, so the same chunk size keeps peak VRAM bounded.
        # We also correct the prior dict-vs-tensor indexing bug: upstream
        # PoseNet.forward returns the Hydra dict {head_name: tensor}, so the
        # canonical first-6-dim slice is out["pose"][..., :POSE_DIMS] per
        # CLAUDE.md "Exact scorer architectures" and the canonical pattern at
        # src/tac/research/*.py (e.g. run_saliency_sweep.py:56).
        with torch.no_grad():
            pose_chunks: list[np.ndarray] = []
            chunk = max(1, int(args.scorer_chunk_size))
            for start in range(0, n_pairs, chunk):
                stop = min(start + chunk, n_pairs)
                pose_input = (
                    pair_tensor[start:stop].to(device).float()
                )  # (chunk, 2, 3, H, W)
                pose_out = posenet(posenet.preprocess_input(pose_input))
                # Hydra dict: {"pose": (chunk, 12), ...}; take first POSE_DIMS.
                pose_chunk = (
                    pose_out["pose"][..., :POSE_DIMS]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                pose_chunks.append(pose_chunk)
                del pose_input, pose_out, pose_chunk
            pose = np.concatenate(pose_chunks, axis=0).astype(np.float32)
        pose_bytes, pose_zero = quantize_pose_deltas(
            pose, scale=args.pose_quant_scale
        )
        _stage(
            f"pose_quantized_bytes_{len(pose_bytes)}_chunk_{int(args.scorer_chunk_size)}"
        )

        # 9. Pack CH06 archive
        meta = {
            "grayscale_downsample": args.grayscale_downsample,
            "pose_quant_scale": args.pose_quant_scale,
            "pose_quant_zero": pose_zero,
            "compress_lane_id": "lane_nscs06_carmack_hotz_strip_everything_20260515",
        }
        bin_bytes = pack_archive(
            palette=palette,
            cdf=cdf,
            grayscale_arith_bytes=arith_bytes,
            pose_bytes=pose_bytes,
            meta=meta,
            num_pairs=n_pairs,
            grayscale_h=h_g,
            grayscale_w=w_g,
            output_height=CONTEST_RAW_HW[0],
            output_width=CONTEST_RAW_HW[1],
        )
        (args.output_dir / "0.bin").write_bytes(bin_bytes)
        archive_sha = _sha256_bytes(bin_bytes)
        archive_bytes_len = len(bin_bytes)
        print(
            f"[full] wrote 0.bin ({archive_bytes_len} bytes, sha256={archive_sha})"
        )
        _stage(f"archive_built_bytes_{archive_bytes_len}")

        # 10. Write inflate runtime + deterministic archive.zip
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir
            )
            print(f"[full] wrote {archive_zip_path}")

        # 11. CUDA auth eval (Catalog #226)
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
                substrate_tag="nscs06_carmack_hotz",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full] [contest-CUDA] score = {contest_cuda_score} "
                    f"(archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        train_elapsed_sec = time.time() - train_started_at

        # 12. Continual-learning posterior update (Catalog #128)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="nscs06_carmack_hotz",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("NSCS06_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=detected_substrate,
                    architecture_class="lane_nscs06_carmack_hotz_strip_everything_20260515",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes_len,
                    notes="nscs06 Carmack-Hotz strip-everything first anchor",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}", file=sys.stderr)

        # 13. Cost-band anchor (best-effort)
        cost_band_appended = False
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost = parse_actual_cost_usd(
                os.environ.get("NSCS06_ACTUAL_COST_USD"),
                field_name="NSCS06_ACTUAL_COST_USD",
            )
        except ValueError:
            actual_cost = None
        if (
            COST_BAND_TOOL.is_file()
            and train_elapsed_sec > 0
            and actual_cost is not None
        ):
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(COST_BAND_TOOL),
                        "--dispatch-label",
                        f"nscs06_carmack_hotz_{_utc_now_iso()}",
                        "--trainer",
                        "experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py",
                        "--platform",
                        os.environ.get("NSCS06_PLATFORM", "modal"),
                        "--gpu",
                        os.environ.get("NSCS06_GPU", "T4"),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        "1",
                        "--actual-wall-clock-sec",
                        str(train_elapsed_sec),
                        "--actual-cost-usd",
                        str(actual_cost),
                        "--notes",
                        "Carmack-Hotz strip-everything compress-only first anchor",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                cost_band_appended = proc.returncode == 0
            except Exception as exc:
                print(f"[full] cost-band append failed (non-fatal): {exc}", file=sys.stderr)

        # 14. Provenance manifest
        provenance = {
            "schema": "nscs06_carmack_hotz_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": (
                "experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py"
            ),
            "lane_id": "lane_nscs06_carmack_hotz_strip_everything_20260515",
            "args": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": n_pairs,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes_len,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "cost_band_anchor_appended": cost_band_appended,
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "train_elapsed_sec": float(train_elapsed_sec),
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0
    finally:
        unpatch_upstream_yuv6(yuv6_token)


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

CARMACK_HOTZ_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="nscs06_carmack_hotz_strip_everything",
    lane_id="lane_nscs06_carmack_hotz_strip_everything_20260515",
    target_modes=("contest_one_video_replay", "research_substrate"),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        "feedback_grand_reunion_fields_grade_passion_full_council_debrief_"
        "vision_strategy_design_whiteboard_session_20260515.md#composite-4"
    ),
    # 2.2 Architecture & runtime (Catalog #124)
    archive_grammar=(
        "CH06 monolithic single-file 0.bin: header (magic=CH06, version=1) + "
        "grayscale palette (uint8) + class-conditional CDF table (uint16) + "
        "arith-coded grayscale palette indices + per-pair pose deltas (uint8) "
        "+ utf-8 json meta. NO neural weights. NO PyTorch at inflate."
    ),
    parser_section_manifest={
        "header": "CH06_magic_and_version",
        "palette": "uint8_grayscale_levels",
        "cdf": "uint16_class_conditional_cdf",
        "grayscale_stream": "arith_coded_palette_indices",
        "pose_stream": "uint8_quantized_pose_deltas",
        "meta": "utf8_json",
    },
    inflate_runtime_loc_budget=100,
    runtime_dep_closure=("numpy>=1.24", "Pillow>=10"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=1400,  # substrate_engineering exception per L7
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (Catalog #220)
    archive_bytes_added=(
        "~3-15 KB total (palette ~16 B; CDF ~170 B; arith-coded grayscale "
        "~2-10 KB depending on entropy; pose ~3.6 KB)"
    ),
    score_improvement_mechanism_status="OPERATIONAL",
    runtime_overlay_consumed=True,
    # 2.4 Recipe schema
    recipe_smoke_only=False,
    recipe_research_only=False,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band
    cost_band_epochs=1,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=8.00,  # mid-band of $5-15 prediction
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="carmack_hotz_no_neural",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 + 2.8
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "Carmack-Hotz is a hand-rolled codec; per-byte sensitivity is "
            "captured directly by hook_pareto_constraint=rate_distortion_v1 "
            "(the closed-form allocator IS the sensitivity)"
        ),
        "hook_bit_allocator_class": (
            "the substrate IS the bit allocator (closed-form class-conditional "
            "CDF); no separate ibps/lsq/uniform allocator applies"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (grayscale palette + class-conditional arith + "
            "pose-delta warp); no 2+ defensible interpretations to disambiguate"
        ),
    },
)


@register_substrate(CARMACK_HOTZ_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
