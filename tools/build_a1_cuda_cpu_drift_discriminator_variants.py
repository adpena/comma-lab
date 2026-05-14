#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
r"""Build 4 A1 discriminator variants that isolate the CUDA-CPU drift mechanism.

The CUDA-CPU drift on HNeRV-cluster archives is empirically calibrated
(R_pose=5.04, R_seg=1.17 over PR100/101/102/103/105; commit 697bfe01,
``feedback_cuda_cpu_axis_profile_learning_layer_20260508``). The mechanism
behind this drift is currently 3-way ambiguous:

  1. **Loader-byte drift** — DALI (CUDA decoder, ground-truth side) vs
     PyAV (CPU decoder) produce different ground-truth bytes per frame;
     drift accumulates through SegNet/PoseNet.
  2. **Conv-kernel accumulation drift** — PoseNet's FastViT-T12
     RepMixer/conv path (and SegNet's EfficientNet-B2) accumulate FP32
     differently on CUDA vs CPU.
  3. **Hydra/head numerical sensitivity** — PoseNet's Hydra head
     (12-dim → first-6-used) has high condition number; tiny upstream
     perturbations are amplified.

This tool emits 4 variants of A1's submission_dir. The archive bytes are
**bit-identical** across all 4 variants — only ``inflate.py`` differs.
Each variant either (a) leaves the inflate output identical to A1 baseline,
or (b) injects a controlled perturbation that ISOLATES one of the three
hypothesised mechanisms.

The 4 variants:

  V_baseline                — A1 unchanged. Establishes the canonical 5.04 / 1.17 drift.
  V_loader_isolated         — Inflate decoder runs on CPU regardless of available device.
                              Both CUDA and CPU eval get IDENTICAL inflated frames; any
                              remaining CUDA-CPU score gap is from GT-loader (DALI vs PyAV)
                              + scorer-forward path drift, NOT inflate-decoder drift.
  V_conv_isolated           — ``torch.use_deterministic_algorithms(True)`` +
                              ``torch.backends.cudnn.deterministic = True`` +
                              ``benchmark = False`` set in inflate.py before decoder
                              forward. Forces deterministic conv kernels in inflate.
                              Doesn't directly probe upstream PoseNet conv kernels (we
                              can't modify upstream), but isolates whether inflate-time
                              conv-noise contributes to the asymmetry between CUDA-inflated
                              and CPU-inflated frames.
  V_hydra_isolated          — Pre-quantize inflate output frames to a coarser grid
                              (round to nearest multiple of 2 instead of nearest integer).
                              Washes out tiny upstream perturbations before they reach
                              the high-condition Hydra head. If drift narrows here,
                              head-amplification is the dominant mechanism.

Each variant gets its own ``experiments/results/a1_cuda_cpu_drift_discriminator_<variant>_<ts>/``
directory with a complete submission_dir suitable for both GHA CPU eval AND CUDA
exact-eval dispatch.

Per CLAUDE.md:
  - Public PR intake clones (PR101 source) READ-ONLY (Check 109).
  - All claims tagged ``[predicted; cuda-cpu-drift-discriminator on A1 substrate]``
    until paired CPU/CUDA results return ``[contest-CPU GHA Linux x86_64]``
    and ``[contest-CUDA <substrate>]``.
  - Per HNeRV-parity discipline lesson 11 (no-op detector): variants are
    inflate-only, so the new score MUST be re-measured (not predicted from
    archive bytes which are unchanged).
  - Per ``forbidden_premature_kill_without_research_exhaustion`` and the
    "kill as last resort" rule: if all 3 isolation variants fail to reduce
    drift, that is NEGATIVE-RESULT-IS-EVIDENCE for a 4th unmodeled
    mechanism — surfaced for operator review, not used to kill the
    discriminator family.
  - Upstream files (``upstream/evaluate.py``, ``upstream/frame_utils.py``,
    ``upstream/modules.py``) are NEVER modified.

Output layout per variant:
  experiments/results/a1_cuda_cpu_drift_discriminator_<variant>_<ts>/
    submission_dir/
      archive.zip           (bit-identical to A1)
      inflate.py            (variant-specific isolation)
      inflate.sh            (bit-identical to A1)
      src/codec.py          (bit-identical)
      src/model.py          (bit-identical)
    discriminator_manifest.json   (per-variant: submission_name, variant_id,
                                    expected_archive_sha, inflate_py_sha_old/new,
                                    isolation_spec, mechanism_hypothesis)

Cross-references:
  - Source memo: ``feedback_domain_exploitation_catalog_landed_20260509.md``
    (item 4 in top-5 highest EIG/$).
  - Mechanism deep-dive: ``feedback_cuda_cpu_pose_drift_mechanism_deep_dive_20260508.md``.
  - Per-architecture-class registry:
    ``src/tac/optimization/cuda_cpu_axis_profile_registry.py`` (commit 697bfe01).
  - Sister discriminator design: ``feedback_decoder_drift_third_axis_20260508.md``
    (Test C — controlled bit-noise injection).
  - A1 canonical CPU anchor: 0.19284757 [contest-CPU GHA Linux x86_64]
    (``experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# A1 canonical archive (the ONE that scored 0.19284 [contest-CPU GHA])
A1_SUBMISSION_DIR = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir"
)
A1_ARCHIVE_PATH = A1_SUBMISSION_DIR / "archive.zip"
A1_EXPECTED_ARCHIVE_SHA = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_EXPECTED_ARCHIVE_BYTES = 178262
A1_CANONICAL_CPU_SCORE = 0.19284757743677347
A1_CANONICAL_CPU_EVIDENCE = (
    "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json"
)


# ---------------------------------------------------------------------------
# Variant table.
#
# Each variant declares:
#   variant_id            — short id used in output paths and submission_name.
#   name                  — human-readable label.
#   mechanism_hypothesis  — which of the 3 mechanisms this variant ISOLATES.
#   isolation_spec        — structured spec describing the inflate.py mutation.
#   inflate_mutator       — name of the mutator function below (string handle so
#                           the table is JSON-serialisable for the manifest).
#   rationale             — why this variant is an isolation, not a no-op.
#
# The mutator strategies are:
#
#   _mutate_baseline                — emit A1's inflate.py byte-for-byte.
#   _mutate_loader_isolated         — force inflate decoder onto CPU regardless
#                                     of CUDA availability. Eliminates the
#                                     inflate-side conv-kernel asymmetry.
#   _mutate_conv_isolated           — set torch.use_deterministic_algorithms(True)
#                                     + cudnn.deterministic + cudnn.benchmark=False
#                                     before decoder forward. Tightens inflate-time
#                                     conv kernels.
#   _mutate_hydra_isolated          — pre-quantize inflated frames to a coarser
#                                     grid (round to nearest multiple of 2 in uint8).
#                                     Washes out tiny upstream perturbations.
#
# The mutators operate on the source text of A1's inflate.py. Anchor lines:
#   ANCHOR_DEVICE        = "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
#   ANCHOR_DECODER_BUILD = "    decoder = HNeRVDecoder("
#   ANCHOR_LOOP_HEAD     = "    with torch.inference_mode(), open(dst_raw, \"wb\") as fout:"
#   ANCHOR_REROUND_TAIL  = "                .round()"
# ---------------------------------------------------------------------------

ANCHOR_DEVICE = '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
ANCHOR_DECODER_BUILD = "    decoder = HNeRVDecoder("
ANCHOR_LOOP_HEAD = '    with torch.inference_mode(), open(dst_raw, "wb") as fout:'
ANCHOR_REROUND_TAIL = "                .round()"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "v_baseline",
        "name": "V_baseline: A1 unchanged (canonical 5.04 / 1.17 drift anchor)",
        "mechanism_hypothesis": "control",
        "isolation_spec": {
            "kind": "control",
            "modifications": [],
        },
        "inflate_mutator": "_mutate_baseline",
        "rationale": (
            "Control. Confirms the discriminator-builder pipeline produces a "
            "byte-identical inflate.py to A1, and re-measures the canonical "
            "CUDA-CPU score gap on the current GHA + CUDA substrates."
        ),
    },
    {
        "variant_id": "v_loader_isolated",
        "name": "V_loader_isolated: inflate decoder forced to CPU regardless of device",
        "mechanism_hypothesis": "loader_byte_drift",
        "isolation_spec": {
            "kind": "force_inflate_cpu",
            "modifications": [
                "device = torch.device(\"cpu\")  # DISCRIMINATOR: force inflate-side CPU",
            ],
        },
        "inflate_mutator": "_mutate_loader_isolated",
        "rationale": (
            "Forces the inflate-time HNeRVDecoder to run on CPU even when "
            "torch.cuda.is_available() returns True. This makes the inflated "
            "uint8 frames bit-identical between CUDA-eval and CPU-eval runs. "
            "If the canonical CUDA-CPU score gap (~0.033) shrinks in this "
            "variant, the residual gap is dominated by GT-loader drift "
            "(DaliVideoDataset vs AVVideoDataset on the ground-truth side) "
            "+ scorer-forward CUDA-CPU drift. If the gap stays at ~0.033, "
            "inflate-side conv-kernel asymmetry is NOT the dominant mechanism."
        ),
    },
    {
        "variant_id": "v_conv_isolated",
        "name": "V_conv_isolated: deterministic algorithms + cudnn deterministic in inflate",
        "mechanism_hypothesis": "conv_kernel_accumulation_drift",
        "isolation_spec": {
            "kind": "deterministic_kernels",
            "modifications": [
                "torch.use_deterministic_algorithms(True, warn_only=True)",
                "torch.backends.cudnn.deterministic = True",
                "torch.backends.cudnn.benchmark = False",
            ],
        },
        "inflate_mutator": "_mutate_conv_isolated",
        "rationale": (
            "Forces deterministic conv kernels and disables cuDNN benchmark "
            "auto-tuning in the inflate-time decoder. Doesn't reach upstream "
            "PoseNet/SegNet (those are pinned), but tightens the inflate-time "
            "decoder's CUDA conv numerics. If V_loader_isolated AND V_conv_isolated "
            "both narrow the gap, conv-kernel accumulation is part of the "
            "mechanism. If V_conv_isolated narrows BUT V_loader_isolated does not, "
            "the inflate-time conv asymmetry is the contributor (rare but possible)."
        ),
    },
    {
        "variant_id": "v_hydra_isolated",
        "name": "V_hydra_isolated: inflate output pre-quantized to multiples of 2",
        "mechanism_hypothesis": "hydra_head_numerical_sensitivity",
        "isolation_spec": {
            "kind": "coarse_quantize_inflate_output",
            "modifications": [
                # Replaces .round() with .div(2.0).round().mul(2.0) and clamps to uint8 range.
                "div(2.0).round().mul(2.0)",
            ],
        },
        "inflate_mutator": "_mutate_hydra_isolated",
        "rationale": (
            "Pre-quantizes inflated uint8 frames to nearest multiple of 2 (instead "
            "of nearest integer). This washes out tiny upstream perturbations of "
            "magnitude < 1 LSB before they reach the high-condition Hydra head in "
            "PoseNet. If R_pose drops from 5.04 to <2.0 here, head-amplification "
            "of small inputs is the dominant pose-drift mechanism. (Score will "
            "likely be WORSE than baseline on absolute terms — that's expected; "
            "the discriminator signal is the CUDA/CPU RATIO, not absolute score.)"
        ),
    },
]


# ---------------------------------------------------------------------------
# Mutator implementations.
# ---------------------------------------------------------------------------


def _mutate_baseline(template_text: str) -> str:
    """Return template unchanged."""
    return template_text


def _mutate_loader_isolated(template_text: str) -> str:
    """Replace ``device = torch.device("cuda" if ... else "cpu")`` with
    ``device = torch.device("cpu")  # DISCRIMINATOR: force inflate-side CPU``.
    """
    if ANCHOR_DEVICE not in template_text:
        raise RuntimeError(
            f"could not find ANCHOR_DEVICE line in inflate.py template; "
            f"expected: {ANCHOR_DEVICE!r}"
        )
    new_line = '    device = torch.device("cpu")  # DISCRIMINATOR: force inflate-side CPU'
    return template_text.replace(ANCHOR_DEVICE, new_line, 1)


def _mutate_conv_isolated(template_text: str) -> str:
    """Insert deterministic-kernel flags ABOVE the device line."""
    if ANCHOR_DEVICE not in template_text:
        raise RuntimeError(
            f"could not find ANCHOR_DEVICE line in inflate.py template; "
            f"expected: {ANCHOR_DEVICE!r}"
        )
    insert_block = (
        "    # DISCRIMINATOR v_conv_isolated: deterministic conv kernels in inflate.\n"
        "    torch.use_deterministic_algorithms(True, warn_only=True)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False\n"
    )
    return template_text.replace(
        ANCHOR_DEVICE,
        insert_block + ANCHOR_DEVICE,
        1,
    )


def _mutate_hydra_isolated(template_text: str) -> str:
    """Replace ``.round()`` (the per-frame uint8 conversion) with
    ``.div(2.0).round().mul(2.0).clamp(0, 255)``.

    The original chain in A1's inflate.py is::

        frames = (
            up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    We replace ``.round()`` (the indented line ending in ``.round()``) with
    ``.div(2.0).round().mul(2.0).clamp(0, 255)``. The ``.clamp(0, 255)`` is
    re-applied because rounding to multiples of 2 can briefly overshoot (e.g.
    255.7 → 256 → clamp back to 255 → mul by 2 → 510, etc.). We also keep the
    pre-existing ``.clamp(0, 255)`` upstream of the chain to bound the 254-255
    cliff into a stable basin.
    """
    if ANCHOR_REROUND_TAIL not in template_text:
        raise RuntimeError(
            f"could not find ANCHOR_REROUND_TAIL line in inflate.py template; "
            f"expected: {ANCHOR_REROUND_TAIL!r}"
        )
    new_line = (
        "                .div(2.0).round().mul(2.0).clamp(0, 255)  # DISCRIMINATOR: coarse-grid"
    )
    return template_text.replace(ANCHOR_REROUND_TAIL, new_line, 1)


# Map mutator function-handle string → callable. Manifest serializes the string
# only; the dispatch uses this map.
_MUTATORS: dict[str, Any] = {
    "_mutate_baseline": _mutate_baseline,
    "_mutate_loader_isolated": _mutate_loader_isolated,
    "_mutate_conv_isolated": _mutate_conv_isolated,
    "_mutate_hydra_isolated": _mutate_hydra_isolated,
}


# ---------------------------------------------------------------------------
# Build helpers.
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a repo-relative path when possible, else an absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_inflate_py(template_text: str, mutator_name: str) -> str:
    """Apply the named mutator to the template and return the new inflate.py
    source.

    Raises:
      KeyError when ``mutator_name`` is not registered.
      RuntimeError when the mutator's anchor lines are missing from the template.
    """
    mutator = _MUTATORS[mutator_name]
    return mutator(template_text)


def write_variant(
    variant: dict[str, Any],
    output_root: Path,
    timestamp: str,
    inflate_template: str,
) -> dict[str, Any]:
    variant_id = variant["variant_id"]
    out_dir = (
        output_root / f"a1_cuda_cpu_drift_discriminator_{variant_id}_{timestamp}"
    )
    sub_dir = out_dir / "submission_dir"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Copy archive.zip + inflate.sh + src/ from A1 (unchanged).
    shutil.copy2(A1_ARCHIVE_PATH, sub_dir / "archive.zip")
    inflate_sh_path = sub_dir / "inflate.sh"
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.sh", inflate_sh_path)
    inflate_sh_path.chmod(0o755)
    src_target = sub_dir / "src"
    src_target.mkdir(exist_ok=True)
    for fname in ("model.py", "codec.py"):
        shutil.copy2(A1_SUBMISSION_DIR / "src" / fname, src_target / fname)

    # Emit variant inflate.py.
    new_inflate = build_inflate_py(inflate_template, variant["inflate_mutator"])
    inflate_path = sub_dir / "inflate.py"
    inflate_path.write_text(new_inflate)

    # Verify archive integrity.
    archive_path = sub_dir / "archive.zip"
    archive_sha_actual = sha256_of(archive_path)
    if archive_sha_actual != A1_EXPECTED_ARCHIVE_SHA:
        raise RuntimeError(
            f"archive.zip SHA mismatch after copy: "
            f"expected={A1_EXPECTED_ARCHIVE_SHA} actual={archive_sha_actual}; "
            f"aborting variant {variant_id!r}"
        )
    archive_size_actual = archive_path.stat().st_size
    if archive_size_actual != A1_EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive.zip size mismatch after copy: "
            f"expected={A1_EXPECTED_ARCHIVE_BYTES} actual={archive_size_actual}; "
            f"aborting variant {variant_id!r}"
        )

    # Verify the variant ACTUALLY mutated the inflate.py (or, for baseline, did NOT).
    inflate_sha_old = sha256_of(A1_SUBMISSION_DIR / "inflate.py")
    inflate_sha_new = sha256_of(inflate_path)
    if variant_id == "v_baseline":
        if inflate_sha_new != inflate_sha_old:
            raise RuntimeError(
                "v_baseline variant produced inflate.py SHA different from "
                "A1's; baseline must be byte-identical."
            )
    else:
        if inflate_sha_new == inflate_sha_old:
            raise RuntimeError(
                f"variant {variant_id!r} produced inflate.py SHA IDENTICAL "
                f"to A1's baseline; mutator {variant['inflate_mutator']!r} did not apply."
            )

    inflate_lines_old = (A1_SUBMISSION_DIR / "inflate.py").read_text().count("\n")
    inflate_lines_new = inflate_path.read_text().count("\n")

    submission_name = f"a1_cuda_cpu_drift_discriminator_{variant_id}_{timestamp}"
    manifest = {
        "lane_id": "lane_avvideodataset_cuda_path_mechanism_discriminator",
        "schema_version": "a1_cuda_cpu_drift_discriminator_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "variant_id": variant_id,
        "variant_name": variant["name"],
        "variant_rationale": variant["rationale"],
        "mechanism_hypothesis": variant["mechanism_hypothesis"],
        "isolation_spec": variant["isolation_spec"],
        "inflate_mutator": variant["inflate_mutator"],
        "submission_name": submission_name,
        "archive_path": manifest_path(archive_path),
        "archive_sha256": archive_sha_actual,
        "archive_size_bytes": archive_size_actual,
        "archive_unchanged_from_a1": True,
        "inflate_py_path": manifest_path(inflate_path),
        "inflate_py_sha256_old": inflate_sha_old,
        "inflate_py_sha256_new": inflate_sha_new,
        "inflate_py_lines_old": inflate_lines_old,
        "inflate_py_lines_new": inflate_lines_new,
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_smoke_checked": False,
        "evidence_grade": (
            "[predicted; cuda-cpu-drift-discriminator on A1 substrate; pre-dispatch]"
        ),
        "dispatch_blockers": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
            "record runtime tree SHA and terminal dispatch claim row",
            "dispatch BOTH CPU (GHA) and CUDA (T4/4090/A100) per dual-eval mandate",
        ],
        "tag_discipline": {
            "before_eval": "[predicted; cuda-cpu-drift-discriminator on A1 substrate]",
            "after_cpu_eval": (
                "[contest-CPU GHA Linux x86_64] iff GHA dispatch produces eval.yml output"
            ),
            "after_cuda_eval": (
                "[contest-CUDA <substrate>] iff Vast.ai 4090 / Modal A100 / Lightning T4"
                " produces a clean exact-eval result"
            ),
            "macos_local": "[macOS-CPU advisory only] — dev-velocity smoke only, NOT authoritative",
        },
        # Forensics provenance
        "source_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "source_inflate_py_sha256": inflate_sha_old,
        "source_inflate_template_path": manifest_path(
            A1_SUBMISSION_DIR / "inflate.py"
        ),
        "a1_canonical_cpu_score_baseline": {
            "value": A1_CANONICAL_CPU_SCORE,
            "tag": "[contest-CPU GHA Linux x86_64]",
            "evidence_path": A1_CANONICAL_CPU_EVIDENCE,
        },
        # Decision rules for downstream registry update.
        "drift_decomposition_decision_rules": {
            "primary_mechanism_threshold": (
                "If R_pose drops from 5.04 to < 2.0 in any isolated variant, "
                "that mechanism is the primary contributor."
            ),
            "multi_mechanism_threshold": (
                "If multiple isolation experiments BOTH reduce drift, mechanisms "
                "are partially independent (multiplicative)."
            ),
            "fourth_mechanism_threshold": (
                "If NONE of the 3 isolated variants narrow drift, a 4th unmodeled "
                "mechanism exists; surface as operator decision per "
                "'forbidden_premature_kill_without_research_exhaustion' — DO NOT "
                "kill the discriminator family."
            ),
            "no_op_detector": (
                "Per HNeRV-parity discipline lesson 11: the inflate-only mutation "
                "for v_baseline must produce a SHA-identical inflate.py to A1's; "
                "the other 3 must NOT (write_variant raises if either invariant "
                "is violated)."
            ),
        },
    }
    (out_dir / "discriminator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[ok] {variant_id}: out={manifest_path(out_dir)} "
        f"submission_name={submission_name} mutator={variant['inflate_mutator']} "
        f"inflate_sha_new={inflate_sha_new[:12]}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments/results",
        help="parent directory under which to write per-variant directories",
    )
    p.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="UTC timestamp suffix (default: now in YYYYMMDDTHHMMSSZ form)",
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="subset of variant_ids to build (default: all)",
    )
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="print the variant table and exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="don't write files; just print what would be written",
    )
    p.add_argument(
        "--rollup-output",
        type=Path,
        default=None,
        help=(
            "optional path to write a discriminator_rollup.json summarizing all "
            "variants (paths, SHAs, mechanism hypotheses, dispatch wiring)"
        ),
    )
    args = p.parse_args()

    if args.list_variants:
        print(
            f"{'variant_id':<22} {'mechanism_hypothesis':<36} {'mutator':<28} description"
        )
        for v in VARIANTS:
            print(
                f"{v['variant_id']:<22} {v['mechanism_hypothesis']:<36} "
                f"{v['inflate_mutator']:<28} {v['name']}"
            )
        return 0

    # Validate A1 inputs exist.
    if not A1_ARCHIVE_PATH.exists():
        print(
            f"FATAL: A1 archive not found at {manifest_path(A1_ARCHIVE_PATH)}; "
            f"cannot build discriminator variants without canonical A1 substrate.",
            file=sys.stderr,
        )
        return 2
    template_path = A1_SUBMISSION_DIR / "inflate.py"
    if not template_path.exists():
        print(
            f"FATAL: A1 inflate.py template not found at {manifest_path(template_path)}.",
            file=sys.stderr,
        )
        return 2

    # Cross-check the archive SHA before any variant is built.
    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        print(
            f"FATAL: A1 archive SHA mismatch: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={actual_sha}; abort to avoid mis-tagging the discriminator.",
            file=sys.stderr,
        )
        return 2

    timestamp = args.timestamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    selected = (
        VARIANTS
        if args.variants is None
        else [v for v in VARIANTS if v["variant_id"] in set(args.variants)]
    )
    if not selected:
        print(
            f"FATAL: no variants matched --variants={args.variants!r}; "
            f"available: {[v['variant_id'] for v in VARIANTS]}",
            file=sys.stderr,
        )
        return 2

    template = template_path.read_text()
    rollup_entries: list[dict[str, Any]] = []

    for variant in selected:
        if args.dry_run:
            print(
                f"[dry-run] would build {variant['variant_id']} "
                f"(mutator={variant['inflate_mutator']})",
                flush=True,
            )
            continue
        manifest = write_variant(
            variant,
            args.output_root,
            timestamp,
            template,
        )
        rollup_entries.append(manifest)

    if args.rollup_output and rollup_entries:
        rollup = {
            "lane_id": "lane_avvideodataset_cuda_path_mechanism_discriminator",
            "schema_version": "a1_cuda_cpu_drift_discriminator_rollup_v1",
            "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "build_timestamp_suffix": timestamp,
            "n_variants": len(rollup_entries),
            "source_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
            "a1_canonical_cpu_score_baseline": {
                "value": A1_CANONICAL_CPU_SCORE,
                "tag": "[contest-CPU GHA Linux x86_64]",
                "evidence_path": A1_CANONICAL_CPU_EVIDENCE,
            },
            "variants": rollup_entries,
            "next_required_actions": [
                "claim lane via tools/claim_lane_dispatch.py",
                "dispatch CPU eval per variant via tools/dispatch_cpu_eval_via_github_actions.py",
                "dispatch CUDA eval per variant via Vast.ai 4090 / Modal A100 / Lightning T4",
                "harvest both axes per variant; compute observed R_pose / R_seg",
                "update src/tac/optimization/cuda_cpu_axis_profile_registry per discriminator verdict",
                "land memory file with mechanism verdict + downstream implications",
            ],
        }
        args.rollup_output.parent.mkdir(parents=True, exist_ok=True)
        args.rollup_output.write_text(
            json.dumps(rollup, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[ok] rollup written: {manifest_path(args.rollup_output)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
