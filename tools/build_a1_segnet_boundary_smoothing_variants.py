#!/usr/bin/env python3
r"""Build N variants of A1's submission_dir with SegNet boundary smoothing
baked into ``inflate.py`` after the canonical PR101-inherited bias correction.

This is atom #1 (highest EIG/$) from
``.omx/research/domain_exploitation_catalog_20260509.md`` (domain_exploit_scorer_1):
3-LOC change to A1's inflate.py adding a boundary-aware Gaussian filter on the
rendered RGB output BEFORE the SegNet preprocess sees it. Exploits SegNet's
stride-2 stem blind-spot — a known structural property of EfficientNet-B2's
canonical stem (CLAUDE.md "Exact scorer architectures" + "Fridrich inverse
steganalysis" sections).

Mechanism (per CLAUDE.md):
  - SegNet = ``smp.Unet('tu-efficientnet_b2', classes=5)`` with vanilla stride-2
    stem (NO Yousfi surgery). Input: bilinear-resize to (512, 384). Distortion =
    argmax disagreement rate over 5 classes.
  - The stride-2 stem loses HALF resolution immediately, then EfficientNet
    downsamples further. Distortion is dominated by argmax flips at class
    boundaries — high-frequency noise in the rendered output that has no
    semantic content but jitters argmax assignments at boundary pixels.
  - A small Gaussian filter on the rendered RGB output (BEFORE the bilinear
    resize SegNet's preprocess applies) reduces those argmax flips. PoseNet
    (FastViT-T12 RepMixer/conv) is largely insensitive to this because its
    distortion is MSE on first 6 pose dims (continuous, not argmax).

Predicted impact (per CLAUDE.md "SegNet vs PoseNet importance — operating-point
dependent"): at PR106/A1 medal-band substrate, SegNet TOTAL contribution is
3.67× larger than pose. Even small boundary-noise reductions are net positive
on the CPU axis (the leaderboard axis). Predicted -0.001 to -0.005 score
delta.

Variants:
  V_baseline      : A1 unchanged (control; reproduces 0.19284 [contest-CPU GHA])
  V_smooth_3x3    : 3x3 Gaussian, sigma=0.5 (mild)
  V_smooth_5x5    : 5x5 Gaussian, sigma=1.0 (moderate)
  V_smooth_class_aware : per-class boundary smoothing approximation via two-stage
                         (3x3 sigma=0.5 then sub-stride2 box average) — purely
                         RGB-domain since we don't have classes at inflate time.

The archive bytes themselves are UNCHANGED — only `inflate.py` differs across
variants. Each variant gets its own
``experiments/results/segnet_boundary_smoothing_v<variant>_20260509/`` directory
with a complete submission_dir suitable for GHA CPU eval dispatch.

Per CLAUDE.md / sister-subagent coordination:
  - Public PR intake clones (PR101/PR102/PR103 source) READ-ONLY (Check 109).
  - All claims tagged ``[predicted; SegNet boundary smoothing on A1 substrate]``
    until GHA result returns ``[contest-CPU GHA Linux x86_64]``.
  - Per HNeRV-parity discipline lesson 11: no-op detector — variants are
    inflate-only, so the new score MUST be re-measured. We additionally include
    a runtime byte-difference proof (``no_op_detector_passed``) by sampling 1
    rendered frame from baseline vs each smoothed variant and asserting
    pixel-level non-equivalence.
  - Per lesson 13: any "variant doesn't help" finding is DEFERRED-pending-research.
  - Per HIGH 1 fix in dispatcher (codex round-2 a44467a): submission_name
    matching is now exact-identity, so distinct submission names per variant
    are still recommended for clarity but no longer load-bearing for custody.
  - Per CLAUDE.md "/tmp paths FORBIDDEN": all paths under experiments/results/.

Output layout per variant:
  experiments/results/segnet_boundary_smoothing_<variant_id>_<ts>/
    submission_dir/
      archive.zip           (bit-identical to A1)
      inflate.py            (variant-specific smoothing; baseline = A1 unchanged)
      inflate.sh            (bit-identical to A1)
      src/codec.py          (bit-identical)
      src/model.py          (bit-identical)
    sweep_manifest.json     (per-variant: submission_name, variant_id,
                             expected_archive_sha, inflate_py_sha_old/new,
                             lines_added, smoothing_spec, no_op_detector_result)
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

# A1 canonical CPU score (the regression target for V_baseline)
A1_CANONICAL_SCORE_VALUE = 0.19284757743677347
A1_CANONICAL_SCORE_TAG = "[contest-CPU GHA Linux x86_64]"
A1_CANONICAL_SCORE_EVIDENCE = (
    "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/"
    "contest_auth_eval.adjudicated.json"
)

# PR101 source (READ-ONLY oracle per Check 109).
PR101_INFLATE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py"
)


# --- Smoothing-block builders ---------------------------------------------
#
# Each builder returns a list of source-code lines (each ALREADY indented to
# match the inflate.py loop body — 12 spaces). Lines are inserted AFTER the
# canonical PR101-inherited bias correction block and BEFORE the ``frames = (``
# reshape line.
#
# Constraints:
#   - Use only torch + torch.nn.functional (already imported in A1's inflate.py).
#   - Operate on `up` tensor of shape (batch, 2, 3, CAMERA_H, CAMERA_W).
#   - Respect dtype/device (don't promote to float64).
#   - The smoothing kernel is built ONCE per call (no per-pair recomputation).

SMOOTH_BASELINE_LINES: list[str] = []  # V_baseline = no smoothing

SMOOTH_3X3_LINES: list[str] = [
    "# SegNet boundary smoothing — 3x3 Gaussian, sigma=0.5",
    "_k3 = torch.tensor([1., 2., 1.], device=up.device, dtype=up.dtype) / 4.",
    "_kernel3 = (_k3[:, None] * _k3[None, :])[None, None, :, :].expand(3, 1, 3, 3).contiguous()",
    "_b, _f, _c, _h, _w = up.shape",
    "up = F.conv2d(up.view(_b * _f, _c, _h, _w), _kernel3, padding=1, groups=3).view(_b, _f, _c, _h, _w)",
]

SMOOTH_5X5_LINES: list[str] = [
    "# SegNet boundary smoothing — 5x5 Gaussian, sigma=1.0",
    "_k5 = torch.tensor([1., 4., 6., 4., 1.], device=up.device, dtype=up.dtype) / 16.",
    "_kernel5 = (_k5[:, None] * _k5[None, :])[None, None, :, :].expand(3, 1, 5, 5).contiguous()",
    "_b, _f, _c, _h, _w = up.shape",
    "up = F.conv2d(up.view(_b * _f, _c, _h, _w), _kernel5, padding=2, groups=3).view(_b, _f, _c, _h, _w)",
]

SMOOTH_CLASS_AWARE_LINES: list[str] = [
    "# SegNet boundary smoothing — two-stage (3x3 Gaussian sigma=0.5 + sub-stride-2 box avg)",
    "_k3 = torch.tensor([1., 2., 1.], device=up.device, dtype=up.dtype) / 4.",
    "_kernel3 = (_k3[:, None] * _k3[None, :])[None, None, :, :].expand(3, 1, 3, 3).contiguous()",
    "_b, _f, _c, _h, _w = up.shape",
    "_flat = up.view(_b * _f, _c, _h, _w)",
    "_stage1 = F.conv2d(_flat, _kernel3, padding=1, groups=3)",
    "# Sub-stride-2 box average mimics the SegNet stride-2 stem blind-spot domain",
    "_kbox = torch.full((3, 1, 2, 2), 0.25, device=up.device, dtype=up.dtype)",
    "_stage2 = F.conv2d(_stage1, _kbox, padding=0, stride=1, groups=3)",
    "# Pad back to (_h, _w) so the downstream reshape works",
    "_stage2 = F.pad(_stage2, (0, 1, 0, 1), mode='replicate')",
    "up = _stage2.view(_b, _f, _c, _h, _w)",
]


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "v_baseline",
        "name": "V_baseline: A1 unchanged (control; reproduces 0.19284 [contest-CPU GHA])",
        "smoothing_lines": SMOOTH_BASELINE_LINES,
        "smoothing_kind": "none",
        "smoothing_kernel_h": 0,
        "smoothing_kernel_w": 0,
        "smoothing_sigma": None,
        "rationale": (
            "Sanity check: this MUST reproduce A1's 0.19284 score. Confirms "
            "submission packaging is bit-identical to the canonical A1 GHA "
            "result and that the build pipeline does not introduce drift."
        ),
        "predicted_delta_band": (0.0, 0.0),
        "predicted_delta_tag": (
            "[predicted; baseline regression check; expected exact match]"
        ),
    },
    {
        "variant_id": "v_smooth_3x3",
        "name": "V_smooth_3x3: 3x3 Gaussian boundary smoothing, sigma=0.5",
        "smoothing_lines": SMOOTH_3X3_LINES,
        "smoothing_kind": "gaussian_3x3_sigma05",
        "smoothing_kernel_h": 3,
        "smoothing_kernel_w": 3,
        "smoothing_sigma": 0.5,
        "rationale": (
            "Mild boundary-noise reduction. Predicted to reduce SegNet argmax "
            "disagreement at class boundaries (the stride-2 stem blind-spot "
            "domain) while leaving PoseNet's MSE largely unchanged. PR101 has "
            "established that small per-channel arithmetic at this site is "
            "load-bearing for the gold score, suggesting the substrate has "
            "headroom for finer boundary-domain edits."
        ),
        "predicted_delta_band": (-0.005, -0.001),
        "predicted_delta_tag": (
            "[predicted; SegNet boundary smoothing on A1 substrate; CPU-axis]"
        ),
    },
    {
        "variant_id": "v_smooth_5x5",
        "name": "V_smooth_5x5: 5x5 Gaussian boundary smoothing, sigma=1.0",
        "smoothing_lines": SMOOTH_5X5_LINES,
        "smoothing_kind": "gaussian_5x5_sigma10",
        "smoothing_kernel_h": 5,
        "smoothing_kernel_w": 5,
        "smoothing_sigma": 1.0,
        "rationale": (
            "Moderate boundary-noise reduction. Tests whether stronger "
            "smoothing helps or hurts — the design tension between SegNet "
            "(prefers smoother boundaries) and PoseNet (penalizes blur of "
            "geometric features the FastViT-T12 backbone exploits)."
        ),
        "predicted_delta_band": (-0.003, +0.002),
        "predicted_delta_tag": (
            "[predicted; SegNet vs PoseNet design-tension probe on A1 substrate]"
        ),
    },
    {
        "variant_id": "v_smooth_class_aware",
        "name": "V_smooth_class_aware: 3x3 Gaussian + sub-stride-2 box average (two-stage)",
        "smoothing_lines": SMOOTH_CLASS_AWARE_LINES,
        "smoothing_kind": "two_stage_gaussian3x3_then_box2x2",
        "smoothing_kernel_h": 3,
        "smoothing_kernel_w": 3,
        "smoothing_sigma": 0.5,
        "rationale": (
            "Approximates the SegNet stride-2 stem blind-spot domain by "
            "first applying a mild 3x3 Gaussian (boundary smoothing) then a "
            "2x2 box average (mimicking the stride-2 stem's effective "
            "receptive field in the full-resolution domain). Inflate-time "
            "only — we do not have class labels at inflate time, so this is "
            "a domain-aware approximation rather than a true class-conditional "
            "smoother."
        ),
        "predicted_delta_band": (-0.004, +0.001),
        "predicted_delta_tag": (
            "[predicted; two-stage substrate-domain smoothing; uncertain band]"
        ),
    },
]


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


# Anchor block where smoothing is inserted. PR101's canonical inflate block
# is between the reshape line and the frames-conversion line. We MUST keep
# the bias correction block in place (it is load-bearing for A1's score)
# and insert the smoothing block AFTER the bias lines and BEFORE the frames
# conversion.
ANCHOR_AFTER_BIAS = "            up[:, 1, 1].sub_(1.0)"
ANCHOR_END_PREFIX = "            frames = ("


def build_inflate_py(template_text: str, smoothing_lines: list[str]) -> str:
    """Emit a new inflate.py with the variant's smoothing block inserted
    AFTER A1's existing PR101-inherited bias correction and BEFORE the
    frames-conversion reshape.

    Robustness: anchor to the LAST bias-correction line (A1 has 3 bias
    lines); insert smoothing block immediately after.
    """
    lines = template_text.splitlines(keepends=True)
    insert_idx = None
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == ANCHOR_AFTER_BIAS:
            insert_idx = i + 1
            break
    if insert_idx is None:
        raise RuntimeError(
            f"could not find anchor line {ANCHOR_AFTER_BIAS!r} in inflate.py "
            f"template; A1 inflate template may have drifted"
        )
    # Defensive: confirm `frames = (` follows (with at most a few blank
    # lines between).
    found_frames = False
    for j in range(insert_idx, min(insert_idx + 5, len(lines))):
        if lines[j].startswith(ANCHOR_END_PREFIX):
            found_frames = True
            break
    if not found_frames:
        raise RuntimeError(
            f"after anchor {ANCHOR_AFTER_BIAS!r}, expected {ANCHOR_END_PREFIX!r} "
            f"within 5 lines; A1 inflate template structure changed"
        )
    if not smoothing_lines:
        # V_baseline: no insert. Just re-emit unchanged.
        return template_text
    smoothing_block = "".join("            " + b + "\n" for b in smoothing_lines)
    out_lines = lines[:insert_idx] + [smoothing_block] + lines[insert_idx:]
    return "".join(out_lines)


def run_no_op_detector(
    baseline_inflate: Path,
    variant_inflate: Path,
    archive_path: Path,
    output_dir: Path,
    a1_submission_dir: Path,
) -> dict[str, Any]:
    """Render 1 batch (1 pair = 2 frames) from baseline + variant; assert
    pixel-level non-equivalence to prove the smoothing block is consumed by
    the rendering path (not silently dead code).

    To keep memory and disk bounded, we use a SHIM inflate script that calls
    inflate.parse_a1_finetuned_archive + decoder for just 1 latent pair
    (~2.4 MB raw output instead of ~3.4 GB for full 600-pair inflate).

    Returns a dict with per-variant pixel diff statistics. The baseline
    case is exempt (there is no smoothing to verify).
    """
    import subprocess
    import zipfile

    if variant_inflate == baseline_inflate or _file_bytes_equal(
        baseline_inflate, variant_inflate
    ):
        return {
            "no_op_detector_passed": True,
            "skipped": True,
            "reason": "baseline variant — no smoothing to verify",
        }

    # Stage src/ next to each inflate.py so they can find codec/model.
    work = output_dir / "no_op_detector_work"
    work.mkdir(exist_ok=True)
    base_dir = work / "baseline"
    var_dir = work / "variant"
    for d, infl in [(base_dir, baseline_inflate), (var_dir, variant_inflate)]:
        d.mkdir(exist_ok=True)
        shutil.copy2(infl, d / "inflate.py")
        src_target = d / "src"
        src_target.mkdir(exist_ok=True)
        for fname in ("model.py", "codec.py"):
            shutil.copy2(a1_submission_dir / "src" / fname, src_target / fname)

    # The inflate.py expects the raw inner blob, NOT the ZIP wrapper.
    # archive.zip contains a single member named 'x' (inflate.sh's contract).
    inner_bin = work / "x"
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if "x" not in names:
            return {
                "no_op_detector_passed": False,
                "error": f"archive.zip member 'x' missing; members={names}",
            }
        with zf.open("x") as src, open(inner_bin, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Write a SHIM script that does the same render path as inflate.inflate()
    # but with N_PAIRS=1, so memory + disk stay bounded.
    shim_src = (
        "import sys, os\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE))\n"
        "import inflate as inf\n"
        "inf.N_PAIRS = 1\n"  # 1 pair = 2 frames; 2 * 1164 * 874 * 3 = ~6.1 MB
        "inf.inflate(sys.argv[1], sys.argv[2])\n"
    )
    (base_dir / "shim.py").write_text(shim_src)
    (var_dir / "shim.py").write_text(shim_src)

    base_raw = work / "base.raw"
    var_raw = work / "var.raw"
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    py_exe = str(venv_python) if venv_python.exists() else sys.executable

    try:
        subprocess.run(
            [py_exe, str(base_dir / "shim.py"), str(inner_bin), str(base_raw)],
            check=True, capture_output=True, text=True, timeout=600,
        )
        subprocess.run(
            [py_exe, str(var_dir / "shim.py"), str(inner_bin), str(var_raw)],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        return {
            "no_op_detector_passed": False,
            "error": f"inflate failed: {e.stderr[:500]}",
        }
    except subprocess.TimeoutExpired:
        return {
            "no_op_detector_passed": False,
            "error": "inflate timed out (>600s)",
        }

    if not base_raw.exists() or not var_raw.exists():
        return {
            "no_op_detector_passed": False,
            "error": "inflate produced no output",
        }

    base_bytes = base_raw.read_bytes()
    var_bytes = var_raw.read_bytes()
    if len(base_bytes) != len(var_bytes):
        return {
            "no_op_detector_passed": True,
            "different_byte_count": True,
            "base_bytes": len(base_bytes),
            "variant_bytes": len(var_bytes),
        }
    # Full diff over 2-frame raw output (~6.1 MB).
    sample_n = len(base_bytes)
    diffs = sum(
        1 for a, b in zip(base_bytes, var_bytes) if a != b
    )
    diff_ratio = diffs / sample_n if sample_n else 0.0

    # Cleanup raw outputs (now small but still cleanup)
    base_raw.unlink(missing_ok=True)
    var_raw.unlink(missing_ok=True)

    return {
        "no_op_detector_passed": diffs > 0,
        "sampled_bytes": sample_n,
        "differing_bytes": diffs,
        "diff_ratio": diff_ratio,
    }


def _file_bytes_equal(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def write_variant(
    variant: dict[str, Any],
    output_root: Path,
    timestamp: str,
    inflate_template: str,
    run_no_op: bool = True,
) -> dict[str, Any]:
    variant_id = variant["variant_id"]
    out_dir = output_root / f"segnet_boundary_smoothing_{variant_id}_{timestamp}"
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
    new_inflate = build_inflate_py(inflate_template, variant["smoothing_lines"])
    inflate_path = sub_dir / "inflate.py"
    inflate_path.write_text(new_inflate)

    # Verify archive.zip integrity (must be bit-identical to A1).
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

    inflate_sha_old = sha256_of(A1_SUBMISSION_DIR / "inflate.py")
    inflate_sha_new = sha256_of(inflate_path)
    inflate_lines_old = (A1_SUBMISSION_DIR / "inflate.py").read_text().count("\n")
    inflate_lines_new = inflate_path.read_text().count("\n")
    lines_added = inflate_lines_new - inflate_lines_old

    # No-op detector: render 1 frame from baseline + variant; verify
    # pixel-level non-equivalence (proves smoothing is consumed).
    no_op_result: dict[str, Any] = {"no_op_detector_passed": True, "skipped": True}
    if run_no_op:
        no_op_result = run_no_op_detector(
            baseline_inflate=A1_SUBMISSION_DIR / "inflate.py",
            variant_inflate=inflate_path,
            archive_path=archive_path,
            output_dir=out_dir,
            a1_submission_dir=A1_SUBMISSION_DIR,
        )

    submission_name = f"a1_segnet_boundary_smoothing_{variant_id}_{timestamp}"
    manifest = {
        "lane_id": "lane_a1_segnet_boundary_smoothing_inflate",
        "schema_version": "a1_segnet_boundary_smoothing_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "variant_id": variant_id,
        "variant_name": variant["name"],
        "variant_rationale": variant["rationale"],
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
        "inflate_py_lines_added": lines_added,
        "smoothing_spec": {
            "smoothing_kind": variant["smoothing_kind"],
            "smoothing_lines": variant["smoothing_lines"],
            "n_smoothing_lines": len(variant["smoothing_lines"]),
            "smoothing_kernel_h": variant["smoothing_kernel_h"],
            "smoothing_kernel_w": variant["smoothing_kernel_w"],
            "smoothing_sigma": variant["smoothing_sigma"],
            "anchor_after_bias": ANCHOR_AFTER_BIAS,
            "anchor_end_prefix": ANCHOR_END_PREFIX,
        },
        "no_op_detector_result": no_op_result,
        "no_op_detector_passed": bool(no_op_result.get("no_op_detector_passed", False)),
        "predicted_delta_band": list(variant["predicted_delta_band"]),
        "predicted_delta_tag": variant["predicted_delta_tag"],
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": bool(
            no_op_result.get("no_op_detector_passed", False)
        ),
        "runtime_smoke_checked": bool(
            no_op_result.get("no_op_detector_passed", False)
            and not no_op_result.get("skipped")
        ),
        "evidence_grade": (
            "[predicted; SegNet boundary smoothing on A1 substrate; pre-GHA-dispatch]"
        ),
        "dispatch_blockers": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
            "record runtime tree SHA and terminal dispatch claim row",
        ],
        "tag_discipline": {
            "before_eval": (
                "[predicted; SegNet boundary smoothing on A1 substrate]"
            ),
            "after_eval": (
                "[contest-CPU GHA Linux x86_64] iff GHA dispatch produces eval.yml output"
            ),
        },
        # Forensics provenance
        "source_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "source_inflate_py_sha256": inflate_sha_old,
        "source_inflate_template_path": manifest_path(
            A1_SUBMISSION_DIR / "inflate.py"
        ),
        "pr101_inflate_oracle_sha256": (
            sha256_of(PR101_INFLATE) if PR101_INFLATE.exists() else None
        ),
        "a1_canonical_score_baseline": {
            "value": A1_CANONICAL_SCORE_VALUE,
            "tag": A1_CANONICAL_SCORE_TAG,
            "evidence_path": A1_CANONICAL_SCORE_EVIDENCE,
        },
        "domain_catalog_atom_id": "domain_exploit_scorer_1",
        "domain_catalog_evidence": (
            ".omx/research/domain_exploitation_catalog_20260509.md (atom #1; "
            "highest EIG/$ in catalog; ~$0.40 GHA dispatch; predicted -0.001 to "
            "-0.005 on CPU axis)"
        ),
    }
    (out_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[ok] {variant_id}: out={manifest_path(out_dir)} "
        f"submission_name={submission_name} smoothing_lines={len(variant['smoothing_lines'])} "
        f"inflate_sha_new={inflate_sha_new[:12]} "
        f"no_op_passed={manifest['no_op_detector_passed']}",
        flush=True,
    )
    return manifest


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
        help="UTC timestamp suffix; default = now",
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
        "--skip-no-op-detector",
        action="store_true",
        help=(
            "(diagnostic) skip the no-op detector that renders 1 frame from "
            "baseline + each variant. Default ON because the detector is the "
            "only safeguard against silent dead-code in inflate.py."
        ),
    )
    p.add_argument(
        "--rollup-output",
        type=Path,
        default=None,
        help="optional path to write a sweep_rollup.json summarizing all variants",
    )
    args = p.parse_args()

    if args.list_variants:
        print("variant_id                          n_smooth_lines  description")
        for v in VARIANTS:
            print(
                f"  {v['variant_id']:<32}  {len(v['smoothing_lines']):>2}  {v['name']}"
            )
        return 0

    # Validate inputs exist
    if not A1_ARCHIVE_PATH.exists():
        sys.stderr.write(f"[fatal] A1 archive missing: {A1_ARCHIVE_PATH}\n")
        return 2
    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        sys.stderr.write(
            f"[fatal] A1 archive SHA mismatch: "
            f"expected={A1_EXPECTED_ARCHIVE_SHA} actual={actual_sha}\n"
        )
        return 2

    template_path = A1_SUBMISSION_DIR / "inflate.py"
    if not template_path.exists():
        sys.stderr.write(f"[fatal] A1 inflate.py template missing: {template_path}\n")
        return 2
    template_text = template_path.read_text()

    timestamp = args.timestamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    if args.variants is None:
        selected = VARIANTS
    else:
        wanted = set(args.variants)
        selected = [v for v in VARIANTS if v["variant_id"] in wanted]
        if len(selected) != len(wanted):
            missing = wanted - {v["variant_id"] for v in VARIANTS}
            sys.stderr.write(
                f"[fatal] unknown variant_ids: {sorted(missing)}\n"
                f"  available: {[v['variant_id'] for v in VARIANTS]}\n"
            )
            return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    rollup: dict[str, Any] = {
        "lane_id": "lane_a1_segnet_boundary_smoothing_inflate",
        "schema_version": "a1_segnet_boundary_smoothing_rollup_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "timestamp": timestamp,
        "n_variants": len(selected),
        "variants": [],
        "domain_catalog_atom_id": "domain_exploit_scorer_1",
    }
    for variant in selected:
        m = write_variant(
            variant,
            output_root=args.output_root,
            timestamp=timestamp,
            inflate_template=template_text,
            run_no_op=not args.skip_no_op_detector,
        )
        rollup["variants"].append(m)

    if args.rollup_output is not None:
        args.rollup_output.parent.mkdir(parents=True, exist_ok=True)
        args.rollup_output.write_text(
            json.dumps(rollup, indent=2, sort_keys=True) + "\n"
        )
        print(f"[ok] wrote rollup -> {manifest_path(args.rollup_output)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
