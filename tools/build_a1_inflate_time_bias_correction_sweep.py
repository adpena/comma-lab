#!/usr/bin/env python3
r"""Build N variants of A1's submission_dir with different inflate-time bias
corrections baked into ``inflate.py``.

This is the highest-EIG/$ \$0-GPU experiment available right now per the
forensics dossier (`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`):

  - PR101 (gold, 0.193) carries inflate-time per-channel Y/V bias correction
    ``up[:,0,0].sub_(1.0); up[:,0,2].sub_(1.0); up[:,1,1].sub_(1.0)`` (from
    PR98's "decode-side channel postprocess" lineage).
  - PR103 (silver, 0.195) carries no bias correction.
  - PR102 (bronze, 0.195) is byte-identical to PR100 but ships
    ``up[:,0,0].add_(1.0)`` + ``DELTA_SCALE 0.0095``.

A1's existing inflate.py ALREADY inherits PR101's bias correction (lines
78-80). So the canonical sweep tests:

  V0  control          : no bias correction                  (expect WORSE than 0.19284)
  V1  baseline         : PR101 default                       (== 0.19284 [contest-CPU GHA])
  V2  half-magnitude   : ÷0.5 of PR101                       (smoothing test)
  V3  1.5x             : ×1.5 of PR101                       (overshoot test)
  V4  2x               : ×2.0 of PR101                       (large-overshoot test)
  V5  opposite-sign    : add(1.0) instead of sub(1.0)        (sign-flip test)
  V6  PR102-pattern    : up[:,0,0].add_(1.0) only            (PR102 bronze pattern)
  V7  PR101+R0+1       : PR101 + up[:,0,0].add_(0.5)         (PR101 stack with PR102 partial)
  V8  zero-frame0-only : sub frame 0 R/B only, leave frame 1 (component-isolated)
  V9  zero-frame1-only : sub frame 1 G only, leave frame 0   (component-isolated)
  V10 R-only-±1.0      : up[:,0,0].sub_(1.0) only            (single-channel)

The archive bytes themselves are UNCHANGED — only `inflate.py` differs across
variants. Each variant gets its own ``experiments/results/a1_bias_correction_sweep_<variant>_<ts>/``
directory with a complete submission_dir suitable for GHA CPU eval dispatch.

Per CLAUDE.md:
  - Public PR intake clones (PR101 source) READ-ONLY (Check 109).
  - All claims tagged ``[predicted; bias-correction-sweep on A1 substrate]``
    until GHA result returns ``[contest-CPU GHA Linux x86_64]``.
  - Per HNeRV-parity discipline lesson 11: no-op detector — variants are
    inflate-only, so the new score MUST be re-measured (not predicted from
    archive bytes which are unchanged).
  - Per lesson 13: any "variant doesn't work" finding is DEFERRED-pending-research.

Output layout per variant:
  experiments/results/a1_bias_correction_sweep_<variant>_<ts>/
    submission_dir/
      archive.zip           (bit-identical to A1)
      inflate.py            (variant-specific bias correction)
      inflate.sh            (bit-identical to A1)
      src/codec.py          (bit-identical)
      src/model.py          (bit-identical)
    sweep_manifest.json     (per-variant: submission_name, variant_id, expected_archive_sha,
                             inflate_py_sha_old/new, lines_changed, bias_spec)
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

# PR101 source (READ-ONLY oracle per Check 109).
PR101_INFLATE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py"
)


# Variants — each is (variant_id, name, bias_lines, sidecar_scale_override).
# bias_lines: a list of inflate.py source lines inserted INSIDE the loop
# at the canonical anchor `up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)`.
# sidecar_scale_override: optional float that replaces SIDECAR_DELTAS_X100/100.
#   None = use codec.py default (0.01).
#
# Each set of lines must be valid inside the PR101-style inflate loop.
VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "v0_control_no_bias",
        "name": "V0 control: no bias correction (expected to score WORSE than baseline)",
        "bias_lines": [],
        "rationale": "Removes A1's inherited PR101 bias correction. Tests whether bias is load-bearing for A1's score.",
    },
    {
        "variant_id": "v1_pr101_baseline",
        "name": "V1 baseline: A1's existing PR101-inherited bias (regression check)",
        "bias_lines": [
            "up[:, 0, 0].sub_(1.0)",
            "up[:, 0, 2].sub_(1.0)",
            "up[:, 1, 1].sub_(1.0)",
        ],
        "rationale": "Should match A1's known 0.19284 [contest-CPU GHA]. Sanity check the sweep wiring.",
    },
    {
        "variant_id": "v2_half_magnitude",
        "name": "V2 half-magnitude: PR101 bias × 0.5",
        "bias_lines": [
            "up[:, 0, 0].sub_(0.5)",
            "up[:, 0, 2].sub_(0.5)",
            "up[:, 1, 1].sub_(0.5)",
        ],
        "rationale": "Smoothing test — does A1's substrate prefer milder bias than PR101's QAT-finetuned substrate?",
    },
    {
        "variant_id": "v3_one_point_five_x",
        "name": "V3 1.5x magnitude: PR101 bias × 1.5",
        "bias_lines": [
            "up[:, 0, 0].sub_(1.5)",
            "up[:, 0, 2].sub_(1.5)",
            "up[:, 1, 1].sub_(1.5)",
        ],
        "rationale": "Overshoot test — does A1's substrate prefer stronger bias than PR101?",
    },
    {
        "variant_id": "v4_two_x",
        "name": "V4 2x magnitude: PR101 bias × 2.0",
        "bias_lines": [
            "up[:, 0, 0].sub_(2.0)",
            "up[:, 0, 2].sub_(2.0)",
            "up[:, 1, 1].sub_(2.0)",
        ],
        "rationale": "Larger overshoot — characterizes the bias magnitude curve.",
    },
    {
        "variant_id": "v5_opposite_sign",
        "name": "V5 opposite sign: add(1.0) instead of sub(1.0)",
        "bias_lines": [
            "up[:, 0, 0].add_(1.0)",
            "up[:, 0, 2].add_(1.0)",
            "up[:, 1, 1].add_(1.0)",
        ],
        "rationale": "Sign-flip test — confirms the bias direction is causal vs accidental on A1.",
    },
    {
        "variant_id": "v6_pr102_pattern",
        "name": "V6 PR102 pattern: only up[:,0,0].add_(1.0)",
        "bias_lines": [
            "up[:, 0, 0].add_(1.0)",
        ],
        "rationale": "PR102 bronze winning pattern (single-channel red bias of frame 0).",
    },
    {
        "variant_id": "v7_pr101_stack_pr102_red",
        "name": "V7 PR101 stacked with PR102 partial red",
        "bias_lines": [
            "up[:, 0, 0].sub_(1.0)",
            "up[:, 0, 2].sub_(1.0)",
            "up[:, 1, 1].sub_(1.0)",
            "up[:, 0, 0].add_(0.5)",
        ],
        "rationale": "Combine PR101's full bias with half of PR102's red shift. Tests stacking medal-band tricks.",
    },
    {
        "variant_id": "v8_frame0_only",
        "name": "V8 frame 0 only: sub frame 0 R/B, leave frame 1",
        "bias_lines": [
            "up[:, 0, 0].sub_(1.0)",
            "up[:, 0, 2].sub_(1.0)",
        ],
        "rationale": "Component-isolated: which frame's bias matters more on A1?",
    },
    {
        "variant_id": "v9_frame1_only",
        "name": "V9 frame 1 only: sub frame 1 G only",
        "bias_lines": [
            "up[:, 1, 1].sub_(1.0)",
        ],
        "rationale": "Component-isolated: frame 1's green-channel bias alone.",
    },
    {
        "variant_id": "v10_red_channel_only",
        "name": "V10 red channel only: sub frame 0 R only",
        "bias_lines": [
            "up[:, 0, 0].sub_(1.0)",
        ],
        "rationale": "Single-channel ablation of frame 0 red bias.",
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a repo-relative path when possible, else an absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# Anchor block where bias correction is inserted. PR101's canonical inflate
# block is between the reshape line and the frames-conversion line. We replace
# the entire region (the existing PR101 bias lines) with the variant's
# bias_lines so each variant carries exactly one bias-correction block.
ANCHOR_START = "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)"
ANCHOR_END_PREFIX = "            frames = ("


def build_inflate_py(template_text: str, bias_lines: list[str]) -> str:
    """Emit a new inflate.py with the variant's bias_lines replacing the
    canonical PR101 bias block.

    Robustness: we re-emit the whole inner loop to be safe — find the
    `up = up.reshape(...)` line, then keep everything up to (but not
    including) the next `        frames = (` line, replace what's between
    them with the variant bias.
    """
    lines = template_text.splitlines(keepends=True)
    start_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == ANCHOR_START and start_idx is None:
            start_idx = i
        elif start_idx is not None and ln.startswith(ANCHOR_END_PREFIX):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError(
            f"could not find anchor lines in inflate.py template "
            f"(start_idx={start_idx} end_idx={end_idx}); template may have drifted"
        )
    # Bias block: each line gets 12 spaces of indentation (matches PR101's
    # canonical inflate.py loop body).
    bias_block = "".join("            " + b + "\n" for b in bias_lines)
    out_lines = lines[: start_idx + 1] + [bias_block] + lines[end_idx:]
    return "".join(out_lines)


def write_variant(
    variant: dict[str, Any],
    output_root: Path,
    timestamp: str,
    inflate_template: str,
) -> dict[str, Any]:
    variant_id = variant["variant_id"]
    out_dir = output_root / f"a1_bias_correction_sweep_{variant_id}_{timestamp}"
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
    new_inflate = build_inflate_py(inflate_template, variant["bias_lines"])
    inflate_path = sub_dir / "inflate.py"
    inflate_path.write_text(new_inflate)

    archive_path = sub_dir / "archive.zip"
    archive_sha_actual = sha256_of(archive_path)
    if archive_sha_actual != A1_EXPECTED_ARCHIVE_SHA:
        raise RuntimeError(
            f"archive.zip SHA mismatch after copy: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={archive_sha_actual}; aborting variant {variant_id!r}"
        )
    archive_size_actual = archive_path.stat().st_size
    if archive_size_actual != A1_EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive.zip size mismatch after copy: expected={A1_EXPECTED_ARCHIVE_BYTES} "
            f"actual={archive_size_actual}; aborting variant {variant_id!r}"
        )

    inflate_sha_old = sha256_of(A1_SUBMISSION_DIR / "inflate.py")
    inflate_sha_new = sha256_of(inflate_path)
    inflate_lines_old = (A1_SUBMISSION_DIR / "inflate.py").read_text().count("\n")
    inflate_lines_new = inflate_path.read_text().count("\n")

    submission_name = (
        f"a1_bias_correction_sweep_{variant_id}_{timestamp}"
    )
    manifest = {
        "lane_id": "lane_a1_inflate_time_bias_correction_sweep",
        "schema_version": "a1_bias_correction_sweep_v1",
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
        "bias_spec": {
            "bias_lines": variant["bias_lines"],
            "n_bias_lines": len(variant["bias_lines"]),
            "anchor_start": ANCHOR_START,
            "anchor_end_prefix": ANCHOR_END_PREFIX,
        },
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_smoke_checked": False,
        "evidence_grade": "[predicted; bias-correction sweep on A1 substrate; pre-GHA-dispatch]",
        "dispatch_blockers": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
            "record runtime tree SHA and terminal dispatch claim row",
        ],
        "tag_discipline": {
            "before_eval": "[predicted; bias-correction sweep on A1 substrate]",
            "after_eval": "[contest-CPU GHA Linux x86_64] iff GHA dispatch produces eval.yml output",
        },
        # Forensics provenance
        "source_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "source_inflate_py_sha256": inflate_sha_old,
        "source_inflate_template_path": manifest_path(A1_SUBMISSION_DIR / "inflate.py"),
        "pr101_inflate_oracle_sha256": sha256_of(PR101_INFLATE),
        "a1_canonical_score_baseline": {
            "value": 0.19284757743677347,
            "tag": "[contest-CPU GHA Linux x86_64]",
            "evidence_path": "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json",
        },
    }
    (out_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[ok] {variant_id}: out={manifest_path(out_dir)} "
        f"submission_name={submission_name} bias_lines={len(variant['bias_lines'])} "
        f"inflate_sha_new={inflate_sha_new[:12]}",
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
        "--dry-run",
        action="store_true",
        help="don't write files; just print what would be written",
    )
    p.add_argument(
        "--rollup-output",
        type=Path,
        default=None,
        help="optional path to write a sweep_rollup.json summarizing all variants",
    )
    args = p.parse_args()

    if args.list_variants:
        print("variant_id                          n_bias_lines  description")
        for v in VARIANTS:
            print(f"  {v['variant_id']:<32}  {len(v['bias_lines']):>2}  {v['name']}")
        return 0

    # Validate inputs exist
    if not A1_ARCHIVE_PATH.exists():
        sys.stderr.write(f"[fatal] A1 archive missing: {A1_ARCHIVE_PATH}\n")
        return 2
    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        sys.stderr.write(
            f"[fatal] A1 archive SHA mismatch: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={actual_sha}\n"
        )
        return 2
    if not PR101_INFLATE.exists():
        sys.stderr.write(f"[fatal] PR101 inflate.py oracle missing: {PR101_INFLATE}\n")
        return 2

    # Use A1's existing inflate.py as the template (it has the PR101-style anchor).
    template_path = A1_SUBMISSION_DIR / "inflate.py"
    template_text = template_path.read_text()

    timestamp = args.timestamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    selected = args.variants or [v["variant_id"] for v in VARIANTS]
    selected_variants = [v for v in VARIANTS if v["variant_id"] in selected]
    if len(selected_variants) != len(selected):
        unknown = set(selected) - {v["variant_id"] for v in VARIANTS}
        sys.stderr.write(
            f"[fatal] unknown variant_id(s): {sorted(unknown)}; "
            f"valid: {[v['variant_id'] for v in VARIANTS]}\n"
        )
        return 2

    if args.dry_run:
        print("[dry-run] would build variants:")
        for v in selected_variants:
            print(f"  {v['variant_id']}: {v['name']}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    rollup: list[dict[str, Any]] = []
    for v in selected_variants:
        manifest = write_variant(v, args.output_root, timestamp, template_text)
        rollup.append({
            "variant_id": manifest["variant_id"],
            "submission_name": manifest["submission_name"],
            "out_dir": str(
                Path(manifest["inflate_py_path"]).parent.parent
            ),
            "archive_sha256": manifest["archive_sha256"],
            "inflate_py_sha256": manifest["inflate_py_sha256_new"],
            "n_bias_lines": manifest["bias_spec"]["n_bias_lines"],
        })

    rollup_path = args.rollup_output or (
        args.output_root / f"a1_bias_correction_sweep_rollup_{timestamp}.json"
    )
    rollup_payload = {
        "schema_version": "a1_bias_correction_sweep_rollup_v1",
        "lane_id": "lane_a1_inflate_time_bias_correction_sweep",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_variants": len(rollup),
        "variants": rollup,
        "a1_canonical_baseline_score": 0.19284757743677347,
        "a1_canonical_baseline_tag": "[contest-CPU GHA Linux x86_64]",
        "a1_canonical_baseline_evidence": "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json",
    }
    rollup_path.write_text(json.dumps(rollup_payload, indent=2, sort_keys=True) + "\n")
    print(
        f"\n[done] {len(rollup)} variants written; rollup={manifest_path(rollup_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
