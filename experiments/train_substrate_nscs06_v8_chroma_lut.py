# SPDX-License-Identifier: MIT
"""Train (compress-only) the nscs06_v8_chroma_lut substrate (L1 INTEGRATION).

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21 + CASCADE
COMPRESSION symposium commit ``d125af6c3`` PRIORITY 3 (Daubechies + Mallat
multi-scale partition discovery framing; 2nd IN-DOMAIN procedural-variant
substrate after grayscale_lut) + HONEST CASCADE-MORTALITY ASSESSMENT
commit ``d884dd6aa`` Rank 2 + canonical equation #26 IN-DOMAIN context
``nscs06_v8_chroma_lut``.

Phase 2 BUILD landing per OVERNIGHT-V (2026-05-21): `_full_main` is now a
one-shot compress + auth-eval pass mirroring the sister v7 trainer pattern
(``experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main``)
adapted to the v8 (level, class) chroma LUT shape. Per OVERNIGHT-T T1 GREEN
PROCEED-unconditional verdict (commit ``3ef1d8876``) + OVERNIGHT-A Phase 2
T2 DESIGN memo (commit ``29f92af8d``) Section 2.1 10-stage decomposition +
RATIFY-3 4 canonical helpers (commit ``20b6b59b3``).

Recipe-vs-trainer-state consistency (Catalog #240):
  - Recipe: ``substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`` now
    flips ``dispatch_enabled: true`` + ``research_only: false`` +
    ``NSCS06_V8_TRAINER_MODE: "full"`` atomically with this trainer landing.
  - Trainer: ``_full_main`` IMPLEMENTED (~200 LOC); ``_smoke_main`` preserved
    for engineering smoke (CPU/synthetic).

Phase 2 10-stage `_full_main` (per OVERNIGHT-A Phase 2 T2 DESIGN memo Section 2.1):
  1. seed pin + device-or-die + output_dir mkdir + RATIFY-3 Carmack-MVP pre-smoke
  2. upstream yuv6 patch + scorer load (compress-side ONLY; strict-scorer-rule)
  3. decode real pairs from upstream/videos/0.mkv via canonical helper
  4. per-pixel SegNet argmax (chunked per Catalog #218 OOM fix; sister v7 pattern)
  5. per-pixel grayscale quantization at full resolution (NEW for v8 vs v7 lowres)
  6. build (16, 5, 3) chroma LUT via canonical build_chroma_lut_from_ground_truth
  7. PoseNet at compress-side (chunked; per Catalog #218; sister v7 dict slice)
  8. RATIFY-3 build_per_assumption_ablation_ladder + Dykstra-feasibility verdict
  9. pack CH08 v2 archive (procedural-seed variant per canonical equation #26)
  10. canonical auth-eval helper gate_auth_eval_call + RATIFY-3 JSON ablation table

CLAUDE.md compliance:
  - Train against ``upstream/videos/0.mkv`` (NOT synthetic; Catalog #114).
  - Patch upstream ``rgb_to_yuv6`` before scorer load (PR #95/#106).
  - ``load_differentiable_scorers`` for SegNet/PoseNet (compress-only; never at inflate).
  - Canonical ``gate_auth_eval_call`` for auth eval (Catalog #226).
  - Continual-learning posterior update via ``posterior_update_locked`` (Catalog #128).
  - Hardware substrate dynamically detected via canonical helper (Catalog #190).
  - Contest-compliant runtime emission (3-positional-arg ``inflate.sh``; Catalog #146).
  - TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151).
  - META layer ``@register_substrate`` decoration (Catalog #241/#242).
  - Atomic recipe flip per Catalog #240 + Strict-flip atomicity rule.
  - No silent device defaults; --device required; --smoke gates CPU.
  - No /tmp paths in persisted artifacts.

Usage (smoke; CPU; tiny output)::

    .venv/bin/python experiments/train_substrate_nscs06_v8_chroma_lut.py \\
        --output-dir experiments/results/nscs06_v8_smoke_<utc> \\
        --device cpu --smoke

Usage (full; CUDA-required compress-side scorer query; Modal T4)::

    .venv/bin/python experiments/train_substrate_nscs06_v8_chroma_lut.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs06_v8_<utc> \\
        --epochs 1 --device cuda
"""
# AUTOCAST_FP16_WAIVED:scorer-runs-once-at-compress-time-not-in-training-loop-so-autocast-irrelevant-sister-v7-pattern
# TORCH_COMPILE_WAIVED:no-training-loop-to-compile-codec-is-pure-numpy-sister-v7-pattern
# TF32_WAIVED:v8-chroma-lut-has-no-neural-codec-no-matmul-operations-numpy-pillow-only-inflate
# NO_GRAD_WAIVED:no-training-loop-no-eval-gradient-numpy-pillow-only-inflate
# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop-segnet-runs-once-at-compress-time-class-label-derivation-only-sister-v7-pattern
# SCORER_PREPROCESS_HANDLED_OK:no-score-aware-loss-no-hot-loop-segnet-runs-once-at-compress-time-for-class-derivation-not-gradient-loss
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

from tac.substrate_registry import register_substrate
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

# Import the v8 substrate package — this triggers @register_substrate
# decoration validation per Catalog #241/#242.
from tac.substrates.nscs06_v8_chroma_lut import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT,
    NUM_SEGNET_CLASSES,
    POSE_DIMS,
    PROCEDURAL_SEED_SIZE_BYTES,
    build_chroma_lut_from_ground_truth,
    pack_archive,
    parse_archive,
    predicted_delta_s,
)
from tac.substrates.nscs06_v8_chroma_lut.revisions import (
    build_per_assumption_ablation_ladder,
    emit_per_assumption_ablation_table_json,
    run_carmack_mvp_first_pre_smoke_verification,
    verify_multi_scale_dykstra_feasibility,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

# Canonical codec-package source. `_write_runtime` vendors these files into
# the submission tree as `submission/_nscs06_v8_codec/{architecture,archive,inflate,procedural_variant}.py`
# so the submission directory is SELF-CONTAINED per HNeRV parity discipline
# L4 + L9 + Catalog #295 (no `tac.*` import from the contest judge's runtime
# tree). Sister of v7 _write_runtime VENDORED_CODEC_FILES pattern.
CODEC_PACKAGE_SOURCE = (
    REPO_ROOT / "src" / "tac" / "substrates" / "nscs06_v8_chroma_lut"
)
# The procedural_codebook_generator is the cross-substrate canonical helper
# (sister grayscale_lut + DP1 + VQ-VAE pattern). Vendor it under the codec
# subdir as well so the inflate runtime is fully self-contained.
PROCEDURAL_CODEBOOK_GENERATOR_SOURCE = (
    REPO_ROOT / "src" / "tac" / "procedural_codebook_generator.py"
)
VENDORED_CODEC_FILES = ("architecture.py", "archive.py", "inflate.py", "procedural_variant.py")
VENDORED_CODEC_SUBDIR = "_nscs06_v8_codec"

EVAL_HW = (384, 512)
CONTEST_RAW_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest — TIER 1 operator-required flags
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS06_V8_VIDEO_PATH",
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
            "feedback_t3_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520.md#priority-3"
        ),
    },
    "--output-dir": {
        "env": "NSCS06_V8_OUTPUT_DIR",
        "rationale": "custody location for archive + provenance + auth-eval JSON",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "NSCS06_V8_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for SegNet/PoseNet weights + evaluate.py; "
            "required for non-smoke compress + auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "NSCS06_V8_DEVICE",
        "rationale": (
            "compute device for compress-side scorer query; cuda required for "
            "full run (MPS refused per CLAUDE.md); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "NSCS06_V8_EPOCHS",
        "rationale": (
            "v8 has no training loop; trainer skeleton + dispatch infra "
            "expect --epochs; we accept any positive value and run ONE "
            "compress pass"
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
        prog="train_substrate_nscs06_v8_chroma_lut",
        description=(
            "Compress-only pass for the nscs06 v8 chroma-LUT substrate "
            "(CASCADE COMPRESSION symposium d125af6c3 PRIORITY 3; canonical "
            "equation #26 IN-DOMAIN context nscs06_v8_chroma_lut; Phase 2 "
            "BUILD landing per OVERNIGHT-V 2026-05-21)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--variant",
        choices=("v1_inline_lut", "v2_procedural_seed"),
        default="v2_procedural_seed",
        help=(
            "CH08 archive variant: v1 inline LUT (full 4096-byte chroma table) "
            "or v2 procedural seed (32-byte PCG64 seed replaces the LUT slot; "
            "predicted ΔS = -0.002706 [prediction; canonical-equation-26-grounded])"
        ),
    )
    # Catalog #218 sister: chunked SegNet + PoseNet forwards (default 8 keeps
    # peak VRAM < 150 MB on T4 14.56 GiB). Same default as v7.
    p.add_argument(
        "--scorer-chunk-size",
        type=int,
        default=8,
        help=(
            "Mini-batch size for compress-side SegNet + PoseNet forward passes; "
            "mirrors v7 chunked pattern + D4's pair_indices kwarg (Catalog #218)."
        ),
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap on pairs decoded (default: N_PAIRS_FULL=600).",
    )
    p.add_argument(
        "--pose-quant-scale",
        type=float,
        default=1.0,
        help="Pose delta uint8 quantization scale (default 1.0).",
    )
    p.add_argument(
        "--grayscale-downsample",
        type=int,
        default=8,
        help=(
            "Grayscale field downsample factor (default 8 -> 48x64 lowres "
            "for 384x512 EVAL_HW; sister v7 pattern)."
        ),
    )
    p.add_argument(
        "--skip-auth-eval",
        action="store_true",
        help="Skip the final canonical auth-eval (for local-CPU dry runs).",
    )
    p.add_argument(
        "--skip-archive-build",
        action="store_true",
        help="Skip the archive.zip build (for trainer-only smoke).",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers (canonical pattern; sister v7)
# ---------------------------------------------------------------------------
def _device_or_die(name: str, *, smoke: bool):
    return _device_or_die_canonical(
        name, smoke=smoke, substrate_tag="nscs06_v8_chroma_lut"
    )


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None):
    return _decode_real_pairs_canonical(
        video_path,
        n_pairs=n_pairs,
        substrate_tag="nscs06_v8_chroma_lut",
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _quantize_pose_deltas(pose: np.ndarray, *, scale: float) -> tuple[bytes, int]:
    """Quantize per-pair pose deltas to uint8 with sister v7 contract.

    Returns (pose_bytes, zero) where zero=128 (uint8 midpoint).
    """
    zero = 128
    arr = (pose * float(scale) + zero).clip(0, 255).astype(np.uint8)
    return arr.tobytes(), zero


def _write_runtime(submission_dir: Path) -> None:
    """Emit SELF-CONTAINED contest-compliant inflate.sh + inflate.py + vendored
    codec package per Catalog #146 + #295 + HNeRV parity discipline L4 + L9.

    Sister of v7 _write_runtime pattern. Vendors the 4 canonical codec files +
    procedural_codebook_generator.py into ``submission/_nscs06_v8_codec/``;
    the submission ``inflate.py`` becomes a thin CLI wrapper that imports
    ``inflate_one_video`` from the vendored package via relative path.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# NSCS06 v8 chroma-LUT contest-compliant inflate (Phase 2 BUILD 2026-05-21)\n"
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

    # Vendor the canonical codec package (4 files, byte-identical copy) +
    # the cross-substrate procedural_codebook_generator.py.
    vendored_dir = submission_dir / VENDORED_CODEC_SUBDIR
    vendored_dir.mkdir(parents=True, exist_ok=True)
    vendor_init = (
        "# SPDX-License-Identifier: MIT\n"
        '"""Vendored NSCS06 v8 codec package — self-contained inflate-side bytes.\n'
        "\n"
        "Copied verbatim from\n"
        "``src/tac/substrates/nscs06_v8_chroma_lut/{architecture,archive,inflate,procedural_variant}.py``\n"
        "+ ``src/tac/procedural_codebook_generator.py`` by\n"
        "``experiments/train_substrate_nscs06_v8_chroma_lut.py::_write_runtime``\n"
        "so the submission tree is self-contained per HNeRV parity L4 + L9 +\n"
        "Catalog #295. NO torch, NO scorer, NO PACT repo dependency at inflate time.\n"
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
                f"NSCS06 v8 vendoring failed: canonical codec source missing: {src}"
            )
        shutil.copy2(src, vendored_dir / fname)

    # Patch the vendored inflate.py to use a relative import for
    # derive_codebook_from_seed (the canonical source uses
    # `from tac.procedural_codebook_generator import ...`).
    vendored_inflate = vendored_dir / "inflate.py"
    text = vendored_inflate.read_text(encoding="utf-8")
    text = text.replace(
        "from tac.procedural_codebook_generator import derive_codebook_from_seed",
        "from .procedural_codebook_generator import derive_codebook_from_seed",
    )
    vendored_inflate.write_text(text, encoding="utf-8")
    # Vendor procedural_codebook_generator.py alongside.
    if not PROCEDURAL_CODEBOOK_GENERATOR_SOURCE.is_file():
        raise FileNotFoundError(
            f"NSCS06 v8 vendoring failed: procedural codebook generator missing: "
            f"{PROCEDURAL_CODEBOOK_GENERATOR_SOURCE}"
        )
    shutil.copy2(
        PROCEDURAL_CODEBOOK_GENERATOR_SOURCE, vendored_dir / "procedural_codebook_generator.py"
    )

    inflate_py = (
        "#!/usr/bin/env python\n"
        "# SPDX-License-Identifier: MIT\n"
        '"""NSCS06 v8 chroma-LUT contest-compliant inflate runtime (NO torch, NO scorer).\n'
        "\n"
        "Self-contained per HNeRV parity discipline L4 + L9 + Catalog #295.\n"
        "Per Catalog #146 the contract is ``inflate.py <archive_dir> <output_dir> <file_list>``.\n"
        "Per Catalog #205 the canonical ``select_inflate_device`` helper is exposed.\n"
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
        f"from {VENDORED_CODEC_SUBDIR}.inflate import inflate_one_video  # noqa: E402\n"
        "\n"
        "\n"
        "def select_inflate_device() -> str:\n"
        '    """Catalog #205 canonical helper; v8 is numpy + Pillow only."""\n'
        "    # INLINE_DEVICE_FORK_OK:v8-chroma-lut-substrate-has-no-torch-no-cuda-cpu-distinction-numpy-pillow-only-sister-v7-pattern\n"
        '    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()\n'
        '    if pinned not in {"auto", "cpu", "cuda"}:\n'
        '        raise SystemExit(\n'
        '            f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}"\n'
        '        )\n'
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
    archive_zip_path: Path, *, bin_bytes: bytes
) -> None:
    """Deterministic archive.zip per Catalog #19 + HNeRV parity L3.

    Archive MUST contain ONLY ``0.bin`` payload (NOT inflate.sh/inflate.py
    runtime scripts; those live alongside the archive in submission_dir/ and
    are routed by the contest evaluator separately). Sister of v7 pattern.
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# _smoke_main — tiny synthetic compress pass (research_only)
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU compress pass for engineering smoke.

    Produces a synthetic CH08 archive (NOT a contest-eligible artifact;
    research-only smoke). Verifies the v8 archive pack/parse + inflate
    roundtrip end-to-end without requiring upstream video or scorers.
    """
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tiny synthetic config — matches the tests in src/tac/substrates/nscs06_v8_chroma_lut/tests/
    num_pairs = 4
    gh, gw = 4, 6
    out_h, out_w = 16, 24

    rng = np.random.RandomState(args.seed)
    pose_bytes = rng.randint(0, 256, size=num_pairs * POSE_DIMS, dtype=np.uint8).tobytes()
    grayscale_bytes = rng.randint(
        0, 256, size=num_pairs * gh * gw, dtype=np.uint8
    ).tobytes()

    if args.variant == "v1_inline_lut":
        lut = rng.randint(
            0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
        )
        blob = pack_archive(
            num_pairs=num_pairs, grayscale_h=gh, grayscale_w=gw,
            output_height=out_h, output_width=out_w,
            pose_bytes=pose_bytes, grayscale_bytes=grayscale_bytes,
            chroma_lut=lut,
        )
    else:
        seed = rng.randint(0, 256, size=PROCEDURAL_SEED_SIZE_BYTES, dtype=np.uint8).tobytes()
        blob = pack_archive(
            num_pairs=num_pairs, grayscale_h=gh, grayscale_w=gw,
            output_height=out_h, output_width=out_w,
            pose_bytes=pose_bytes, grayscale_bytes=grayscale_bytes,
            chroma_seed=seed,
        )

    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(blob)

    arc = parse_archive(blob)
    smoke_meta = {
        "variant": args.variant,
        "schema_version": arc.schema_version,
        "num_pairs": arc.num_pairs,
        "archive_bytes": len(blob),
        "chroma_lut_bytes_declared": CHROMA_LUT_BYTES_DEFAULT,
        "predicted_delta_s": predicted_delta_s(),
        "research_only": True,
        "evidence_grade": "predicted",
        "axis_tag": "[prediction]",
        "promotable": False,
        "score_claim_valid": False,
        "canonical_equation_in_domain_context": "nscs06_v8_chroma_lut",
    }
    (output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, sort_keys=True, indent=2)
    )
    print(
        f"[smoke] CH08 {args.variant} archive bytes: {len(blob)} "
        f"(predicted ΔS = {predicted_delta_s():.6f} [prediction])"
    )
    return 0


# ---------------------------------------------------------------------------
# _full_main — Phase 2 BUILD landing 2026-05-21 (OVERNIGHT-V)
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Phase 2 BUILD: one-shot compress + auth-eval pass per OVERNIGHT-A 10-stage.

    Per OVERNIGHT-T T1 PROCEED-unconditional verdict (commit ``3ef1d8876``) +
    OVERNIGHT-A Phase 2 T2 DESIGN memo (commit ``29f92af8d``) Section 2.1
    10-stage decomposition + RATIFY-3 4 canonical helpers (commit ``20b6b59b3``).
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers

    # Stage 1: seed pin + device-or-die + output_dir mkdir + RATIFY-3 Carmack-MVP
    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")

    # RATIFY-3 REVISION #3 Carmack MVP-first 5-step pre-smoke verification
    # (BEFORE firing the paid GPU meter)
    pre_smoke_verdict = run_carmack_mvp_first_pre_smoke_verification()
    if not pre_smoke_verdict.ready_for_first_paired_smoke:
        print(
            f"[full] FATAL: Carmack MVP-first 5-step pre-smoke verification "
            f"FAILED (all_steps_passed={pre_smoke_verdict.all_steps_passed}); "
            f"refusing dispatch per RATIFY-3 REVISION #3 contract",
            file=sys.stderr,
        )
        for step in pre_smoke_verdict.steps:
            if not step.passed:
                print(f"  - {step.step_letter}: {step.details}", file=sys.stderr)
        return 25  # sister exit code aligned with driver Stage 1b defense
    _stage("ratify3_carmack_mvp_first_passed")

    # RATIFY-3 REVISION #2: Dykstra-feasibility check per Catalog #296
    dykstra_verdict = verify_multi_scale_dykstra_feasibility()
    if not dykstra_verdict.intersection_non_empty:
        print(
            f"[full] FATAL: Dykstra-feasibility intersection EMPTY "
            f"(is_additive={dykstra_verdict.is_additive}); refusing dispatch",
            file=sys.stderr,
        )
        return 26
    _stage("ratify3_dykstra_feasibility_passed")

    # RATIFY-3 REVISION #1: build ablation ladder (cited in metadata; the
    # actual paired smoke runs ONE arm per call; the canonical multi-arm
    # ablation is driven by the operator-authorize harness running this
    # trainer N times with different --variant + helper-pinned axis values
    # per OVERNIGHT-A Phase 2 council REVISION #4 default option (a)).
    ablation_ladder = build_per_assumption_ablation_ladder()
    _stage(f"ratify3_ablation_ladder_built_arms_{len(ablation_ladder.arms)}")

    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    train_started_at = time.time()
    try:
        # Stage 2: scorer load (compress-side ONLY; strict-scorer-rule)
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded_compress_side")

        # Stage 3: decode real pairs from upstream/videos/0.mkv
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        _stage(f"pairs_decoded_{n_pairs}")

        # Stage 4: per-pixel SegNet argmax at FULL resolution (chunked per Catalog #218)
        # v8 needs FULL-resolution class labels to build the (level, class) LUT
        # bins; v7's lowres path is replaced with full-resolution argmax here.
        with torch.no_grad():
            cls_full_chunks: list[np.ndarray] = []
            chunk = max(1, int(args.scorer_chunk_size))
            for start in range(0, n_pairs, chunk):
                stop = min(start + chunk, n_pairs)
                odd_chunk = (
                    pair_tensor[start:stop, 0].to(device).float()
                )  # (chunk, 3, H, W)
                odd_btchw = odd_chunk.unsqueeze(1)
                seg_logits = segnet(segnet.preprocess_input(odd_btchw))
                cls_chunk = (
                    torch.argmax(seg_logits, dim=1).to(torch.uint8).cpu().numpy()
                )
                cls_full_chunks.append(cls_chunk)
                del odd_chunk, odd_btchw, seg_logits, cls_chunk
            cls_full = np.concatenate(cls_full_chunks, axis=0)
        _stage(
            f"segnet_argmax_full_res_chunked_size_{int(args.scorer_chunk_size)}"
        )

        # Stage 5: per-pixel grayscale quantization at FULL resolution (for LUT
        # derivation) + a LOWRES downsampled grayscale stream for archive payload
        # (v8 archives the lowres grayscale; the inflate runtime re-upsamples
        # via Pillow BILINEAR per sister v8 inflate.py contract).
        H, W = EVAL_HW
        h_g = H // args.grayscale_downsample
        w_g = W // args.grayscale_downsample
        odd_rgb = pair_tensor[:, 0].cpu().numpy().astype(np.uint8)  # (N, 3, H, W)
        # BT.601 luma per pixel (full resolution; for LUT derivation)
        r_f = odd_rgb[:, 0].astype(np.float32)
        g_f = odd_rgb[:, 1].astype(np.float32)
        b_f = odd_rgb[:, 2].astype(np.float32)
        gray_full = (0.299 * r_f + 0.587 * g_f + 0.114 * b_f).clip(0, 255).astype(np.uint8)
        # Lowres grayscale via area-average pooling for the archive payload.
        gray_lowres = (
            gray_full.reshape(n_pairs, h_g, args.grayscale_downsample, w_g,
                              args.grayscale_downsample)
            .mean(axis=(2, 4))
            .clip(0, 255)
            .astype(np.uint8)
        )
        _stage(f"grayscale_full_and_lowres_built_{h_g}x{w_g}")

        # Stage 6: build (16, 5, 3) chroma LUT via canonical helper
        chroma_lut = build_chroma_lut_from_ground_truth(
            odd_rgb, cls_full,
            grayscale_levels=GRAYSCALE_LEVELS_DEFAULT,
            num_segnet_classes=NUM_SEGNET_CLASSES,
        )
        chroma_lut_sha = _sha256_bytes(chroma_lut.tobytes())
        _stage(f"chroma_lut_built_shape_{chroma_lut.shape}_sha_{chroma_lut_sha[:8]}")

        # Stage 7: PoseNet at compress-side (chunked; sister v7 dict slice)
        with torch.no_grad():
            pose_chunks: list[np.ndarray] = []
            chunk = max(1, int(args.scorer_chunk_size))
            for start in range(0, n_pairs, chunk):
                stop = min(start + chunk, n_pairs)
                pose_input = (
                    pair_tensor[start:stop].to(device).float()
                )  # (chunk, 2, 3, H, W)
                pose_out = posenet(posenet.preprocess_input(pose_input))
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
        pose_bytes, pose_zero = _quantize_pose_deltas(pose, scale=args.pose_quant_scale)
        _stage(
            f"pose_quantized_bytes_{len(pose_bytes)}_chunk_{int(args.scorer_chunk_size)}"
        )

        # Stage 5b: derive per-cell SegNet class labels at low-res for the
        # CH08 v3 cls_stream slot (Catalog #233 L1→L2 promotion canonical
        # 4-gate unblocker per T3 council #1335 REVISION #2 Yousfi BLOCKER).
        # NEAREST downsample (point-sample top-left pixel of each
        # `args.grayscale_downsample`-sized cell) is the canonical sister of
        # inflate.py's `Image.NEAREST` upsample; the pair forms a lossless
        # round-trip for the "uniform-cls" boundary invariant enforced by
        # `tests/test_cls_stream_wire_in.py::test_inflate_v3_with_uniform_class_matches_v2`.
        # cls_full shape (n_pairs, H, W) was built at Stage 4; lowres shape
        # MUST match Stage 5 gray_lowres (n_pairs, h_g, w_g).
        cls_lowres = cls_full[
            :,
            : h_g * args.grayscale_downsample : args.grayscale_downsample,
            : w_g * args.grayscale_downsample : args.grayscale_downsample,
        ]
        if cls_lowres.shape != (n_pairs, h_g, w_g):
            raise RuntimeError(
                f"cls_lowres shape {cls_lowres.shape} != ({n_pairs}, {h_g}, {w_g}) "
                f"— Stage 5b NEAREST downsample shape invariant violated"
            )
        cls_bytes = np.ascontiguousarray(cls_lowres, dtype=np.uint8).tobytes()
        cls_lowres_sha = _sha256_bytes(cls_bytes)
        _stage(
            f"cls_lowres_nearest_downsample_shape_{cls_lowres.shape}_sha_{cls_lowres_sha[:8]}"
        )

        # Stage 8: (already invoked above as RATIFY-3 pre-smoke + Dykstra)

        # Stage 9: pack CH08 v3 archive (procedural-seed + cls_stream per T3
        # council #1335 PROCEED_WITH_REVISIONS REVISION #2 Yousfi BLOCKER).
        # v3 supersedes the v2 emission per sister cls_stream wire-in landing
        # (commits 581b7b129 + 545beb35c); the v1 inline-LUT branch is
        # codec-incompatible with v3 cls_stream so it omits cls_bytes (the
        # cargo-cult #5 cls=0 uniform inflate fallback is preserved for v1).
        if args.variant == "v2_procedural_seed":
            # Default v2/v3: 32-byte PCG64 seed replaces the inline LUT slot.
            # Per canonical equation #26 closed form, predicted ΔS = -0.002706
            # (REPLACEMENT savings, unchanged). v3 adds cls_stream ADDITIVE
            # bytes; total rate-axis ΔS is the empirical question per the
            # paired Modal T4 dispatch decision (Catalog #246).
            # The seed is HASH-derived from the chroma LUT bytes so it is
            # deterministic AND distinguishing per Catalog #272.
            import hashlib
            seed = hashlib.sha256(chroma_lut.tobytes()).digest()[:PROCEDURAL_SEED_SIZE_BYTES]
            bin_bytes = pack_archive(
                num_pairs=n_pairs,
                grayscale_h=h_g,
                grayscale_w=w_g,
                output_height=CONTEST_RAW_HW[0],
                output_width=CONTEST_RAW_HW[1],
                pose_bytes=pose_bytes,
                grayscale_bytes=gray_lowres.tobytes(),
                pose_quant_scale=args.pose_quant_scale,
                chroma_seed=seed,
                cls_bytes=cls_bytes,
            )
            archive_variant_tag = "v3_procedural_seed_with_cls_stream"
        else:
            # v1: inline 4096-byte LUT (full chroma table; no compression).
            bin_bytes = pack_archive(
                num_pairs=n_pairs,
                grayscale_h=h_g,
                grayscale_w=w_g,
                output_height=CONTEST_RAW_HW[0],
                output_width=CONTEST_RAW_HW[1],
                pose_bytes=pose_bytes,
                grayscale_bytes=gray_lowres.tobytes(),
                pose_quant_scale=args.pose_quant_scale,
                chroma_lut=chroma_lut,
            )
            archive_variant_tag = "v1_inline_lut"

        (args.output_dir / "0.bin").write_bytes(bin_bytes)
        payload_0bin_sha = _sha256_bytes(bin_bytes)
        payload_0bin_bytes = len(bin_bytes)
        print(
            f"[full] wrote 0.bin "
            f"({payload_0bin_bytes} bytes, sha256={payload_0bin_sha}, variant={archive_variant_tag})"
        )
        _stage(f"payload_0bin_built_bytes_{payload_0bin_bytes}_{archive_variant_tag}")

        # Build deterministic archive.zip + submission runtime
        archive_zip_path = args.output_dir / "archive.zip"
        archive_zip_sha: str | None = None
        archive_zip_bytes: int | None = None
        submission_dir = args.output_dir / "submission"
        if not args.skip_archive_build:
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes)
            archive_zip_sha = _sha256_file(archive_zip_path)
            archive_zip_bytes = archive_zip_path.stat().st_size
            print(
                f"[full] wrote {archive_zip_path} "
                f"({archive_zip_bytes} bytes, sha256={archive_zip_sha})"
            )
            _stage(f"archive_zip_built_bytes_{archive_zip_bytes}")

        # Stage 10: canonical auth-eval helper (Catalog #226) + RATIFY-3 JSON table
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=submission_dir / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="nscs06_v8_chroma_lut",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result.get("auth_eval_cuda_score")
                if contest_cuda_score is not None:
                    print(
                        f"[full] [contest-CUDA] score = {contest_cuda_score} "
                        f"(archive_sha256={archive_zip_sha})"
                    )
            _stage("auth_eval_done")

        train_elapsed_sec = time.time() - train_started_at

        # RATIFY-3 REVISION #4: emit per-assumption ablation table JSON
        try:
            ablation_table_path = emit_per_assumption_ablation_table_json(
                ablation_ladder,
                repo_root=REPO_ROOT,
                multi_scale_verdict=dykstra_verdict,
                carmack_mvp_verdict=pre_smoke_verdict,
                extra_provenance={
                    "phase_2_build_landing_commit": _git_head_sha(),
                    "phase_2_build_lane_id": (
                        "lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521"
                    ),
                    "archive_variant_tag": archive_variant_tag,
                    "archive_sha256": archive_zip_sha,
                    "archive_bytes": archive_zip_bytes,
                    "auth_eval_cuda_score": contest_cuda_score,
                    "chroma_lut_sha256": chroma_lut_sha,
                },
            )
            _stage(f"ratify3_ablation_table_emitted_{ablation_table_path.name}")
            print(f"[full] RATIFY-3 ablation table -> {ablation_table_path}")
        except Exception as exc:
            print(f"[full] RATIFY-3 ablation table emission FAILED (non-fatal): {exc}", file=sys.stderr)

        # Continual-learning posterior update (Catalog #128)
        if (
            contest_cuda_score is not None
            and archive_zip_sha is not None
            and archive_zip_bytes is not None
        ):
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="nscs06_v8_chroma_lut",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("NSCS06_V8_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=detected_substrate,
                    architecture_class=(
                        "lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521"
                    ),
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_zip_sha,
                    archive_bytes=archive_zip_bytes,
                    notes=(
                        f"nscs06 v8 chroma_lut Phase 2 BUILD first paired anchor "
                        f"variant={archive_variant_tag} predicted_delta_s={predicted_delta_s():.6f}"
                    ),
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}", file=sys.stderr)

        # Cost-band anchor (best-effort)
        cost_band_appended = False
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost = parse_actual_cost_usd(
                os.environ.get("NSCS06_V8_ACTUAL_COST_USD"),
                field_name="NSCS06_V8_ACTUAL_COST_USD",
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
                        f"nscs06_v8_chroma_lut_{_utc_now_iso()}",
                        "--trainer",
                        "experiments/train_substrate_nscs06_v8_chroma_lut.py",
                        "--platform",
                        os.environ.get("NSCS06_V8_PLATFORM", "modal"),
                        "--gpu",
                        os.environ.get("NSCS06_V8_GPU", "T4"),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        "1",
                        "--actual-wall-clock-sec",
                        str(train_elapsed_sec),
                        "--actual-cost-usd",
                        str(actual_cost),
                        "--notes",
                        "nscs06 v8 chroma_lut Phase 2 BUILD first paired anchor",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                cost_band_appended = proc.returncode == 0
            except Exception as exc:
                print(f"[full] cost-band append failed (non-fatal): {exc}", file=sys.stderr)

        # Provenance manifest
        provenance = {
            "schema": "nscs06_v8_chroma_lut_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_nscs06_v8_chroma_lut.py",
            "lane_id": (
                "lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521"
            ),
            "args": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": n_pairs,
            "archive_sha256": archive_zip_sha,
            "archive_bytes": archive_zip_bytes,
            "archive_zip_path": str(archive_zip_path) if archive_zip_path.is_file() else None,
            "payload_0bin_sha256": payload_0bin_sha,
            "payload_0bin_bytes": payload_0bin_bytes,
            "archive_variant_tag": archive_variant_tag,
            "chroma_lut_sha256": chroma_lut_sha,
            "pose_quant_zero": pose_zero,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "predicted_delta_s_canonical_equation_26": predicted_delta_s(),
            "canonical_equation_in_domain_context": "nscs06_v8_chroma_lut",
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
            "ratify3_carmack_mvp_first_passed": pre_smoke_verdict.all_steps_passed,
            "ratify3_dykstra_is_additive": dykstra_verdict.is_additive,
            "ratify3_ablation_ladder_total_arms": len(ablation_ladder.arms),
            "phase_2_build_council_anchors": {
                "overnight_t_t1_proceed_unconditional": (
                    "council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521"
                ),
                "overnight_a_t2_design": (
                    "council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521"
                ),
                "per_substrate_symposium": (
                    "council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521"
                ),
                "ratify3_helpers_applied": "commit_20b6b59b3",
            },
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0
    finally:
        unpatch_upstream_yuv6(yuv6_token)


@register_substrate(NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main())
