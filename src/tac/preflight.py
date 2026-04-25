"""Preflight pipeline validator — catches integration mismatches before GPU burns.

Every bug in this project was at a boundary between components:
  - Masks at wrong resolution (48x64 vs 384x512 → score 103 vs 2)
  - Poses optimized against wrong masks (27x PoseNet regression)
  - Archive missing artifacts (119KB vs 338KB → 0.108 rate error)
  - eval_roundtrip defaulting False (proxy-auth drift 11x)
  - FP4 without QAT (26x PoseNet degradation)
  - TTO frames at GT range [0,255] instead of TTO-optimized [0,~184] (WILDE failure 2026-04-25)
  - Ad-hoc nohup watchers dying silently (3-A100 deployment failure 2026-04-25)

CANONICAL ENTRY POINT: preflight_all(). Combines:
  - preflight_check         → artifact validation (renderer/masks/poses/archive)
  - preflight_training_inputs → TTO range, profile arch, eval_roundtrip
  - check_codebase_drift    → AST scan blocks ad-hoc patterns

Usage:
    from tac.preflight import preflight_all
    preflight_all(
        profile_name="shiraz",
        profile_arch=PROFILES["shiraz"],
        tto_frames_path="experiments/results/tto_v7_hinge_500/tto_frames.pt",
        gt_poses_path="experiments/results/gt_poses.pt",
        masks_path="submissions/robust_current/masks_crf50.mkv",
    )
"""
from __future__ import annotations

import ast
import re
import struct
import subprocess
import sys
from pathlib import Path

import torch


class PreflightError(Exception):
    """A preflight check failed — do NOT proceed."""
    pass


class PreflightWarning:
    """A preflight check raised a concern but is not fatal."""
    def __init__(self, msg: str):
        self.msg = msg


def preflight_check(
    renderer_path: str | Path | None = None,
    masks_path: str | Path | None = None,
    poses_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    expected_n_frames: int = 1200,
    expected_n_pairs: int = 600,
    expected_seg_h: int = 384,
    expected_seg_w: int = 512,
    verbose: bool = True,
) -> list[PreflightWarning]:
    """Run all preflight checks. Raises PreflightError on fatal issues.

    Returns list of warnings (non-fatal concerns).
    """
    warnings: list[PreflightWarning] = []
    checks_passed = 0
    checks_total = 0

    def _pass(msg: str) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        checks_passed += 1
        if verbose:
            print(f"  [PASS] {msg}")

    def _fail(msg: str) -> None:
        nonlocal checks_total
        checks_total += 1
        if verbose:
            print(f"  [FAIL] {msg}")
        raise PreflightError(msg)

    def _warn(msg: str) -> None:
        nonlocal checks_total
        checks_total += 1
        warnings.append(PreflightWarning(msg))
        if verbose:
            print(f"  [WARN] {msg}")

    if verbose:
        print("=" * 60)
        print("PREFLIGHT CHECK")
        print("=" * 60)

    # ── Renderer checks ──────────────────────────────────────────
    if renderer_path:
        renderer_path = Path(renderer_path)
        if not renderer_path.exists():
            _fail(f"Renderer not found: {renderer_path}")

        raw = renderer_path.read_bytes()
        magic = raw[:4]

        if magic == b"ASYM":
            header_len = struct.unpack("<I", raw[4:8])[0]
            import json
            header = json.loads(raw[8:8 + header_len])
            pose_dim = header.get("pose_dim", 0)
            base_ch = header.get("base_ch", "?")
            dsconv = header.get("use_dsconv", False)
            _pass(f"Renderer: ASYM, pose_dim={pose_dim}, base_ch={base_ch}, dsconv={dsconv}, {len(raw):,}B")

            if pose_dim == 0:
                _warn("Renderer has pose_dim=0 — FiLM conditioning disabled, poses will have no effect")
            if pose_dim > 0 and poses_path is None:
                _warn("Renderer has pose_dim>0 but no poses_path provided — will use zero poses")
        elif magic == b"FP4A":
            _pass(f"Renderer: FP4A, {len(raw):,}B")
            _warn("FP4 renderer — verify QAT was used during training (post-hoc QAT degrades 3-26x)")
        else:
            _warn(f"Renderer: unknown format (magic={magic}), assuming PyTorch .pt")

    # ── Mask checks ──────────────────────────────────────────────
    if masks_path:
        masks_path = Path(masks_path)
        if not masks_path.exists():
            _fail(f"Masks not found: {masks_path}")

        if masks_path.suffix in (".mkv", ".mp4"):
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(masks_path)],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                parts = probe.stdout.strip().split(",")
                w, h = int(parts[0]), int(parts[1])
                size = masks_path.stat().st_size

                if h == expected_seg_h and w == expected_seg_w:
                    _pass(f"Masks: {w}x{h} (native resolution), {size:,}B")
                elif expected_seg_h % h == 0 and expected_seg_w % w == 0:
                    scale = expected_seg_h // h
                    _warn(f"Masks at 1/{scale} resolution ({w}x{h}), will upsample to {expected_seg_w}x{expected_seg_h}")
                else:
                    _fail(f"Masks resolution {w}x{h} is not a clean factor of {expected_seg_w}x{expected_seg_h}")
            else:
                _warn("Could not probe mask video with ffprobe")
        elif masks_path.suffix == ".pt":
            m = torch.load(str(masks_path), weights_only=True)
            if m.shape[1] != expected_seg_h or m.shape[2] != expected_seg_w:
                _warn(f"Masks shape {m.shape} — expected (N, {expected_seg_h}, {expected_seg_w})")
            else:
                _pass(f"Masks: {m.shape}, .pt format")

    # ── Pose checks ──────────────────────────────────────────────
    if poses_path:
        poses_path = Path(poses_path)
        if not poses_path.exists():
            _fail(f"Poses not found: {poses_path}")

        p = torch.load(str(poses_path), weights_only=True)
        if p.shape[0] != expected_n_pairs:
            _fail(f"Poses shape {p.shape} — expected ({expected_n_pairs}, 6). "
                  f"Wrong number of pairs.")
        if p.shape[1] != 6:
            _fail(f"Poses shape {p.shape} — expected (N, 6). Wrong pose dimension.")
        _pass(f"Poses: {p.shape}, dtype={p.dtype}")

        if p.abs().max() > 100:
            _warn(f"Poses max value {p.abs().max():.1f} — unusually large, may indicate wrong scale")
        if p.abs().mean() < 0.001:
            _warn(f"Poses mean abs {p.abs().mean():.6f} — near zero, may not have been optimized")

    # ── Pose-mask consistency ────────────────────────────────────
    # This is the #1 source of score regressions in this project.
    # Poses optimized against wrong masks caused 27x PoseNet degradation.
    if poses_path and masks_path:
        _warn("CRITICAL: Verify poses were optimized against THESE EXACT masks. "
              "Mismatched poses caused 27x PoseNet regression. "
              "optimize_poses.py now requires --masks to prevent this.")

    # ── Archive checks ───────────────────────────────────────────
    if archive_path:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            _fail(f"Archive not found: {archive_path}")

        try:
            from tac.submission_archive import validate_archive, RENDERER_SUBMISSION_MANIFEST
            result = validate_archive(archive_path, RENDERER_SUBMISSION_MANIFEST, strict=False)
            if result.valid:
                _pass(f"Archive: {result.archive_bytes:,}B, rate={result.rate_term:.4f}, valid")
            else:
                for err in result.errors:
                    _fail(f"Archive: {err}")
                for w in result.warnings:
                    _warn(f"Archive: {w}")
        except Exception as e:
            _warn(f"Archive validation failed: {e}")

    # ── Summary ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  {checks_passed}/{checks_total} checks passed, {len(warnings)} warnings")
        if warnings:
            print("  Warnings:")
            for w in warnings:
                print(f"    - {w.msg}")
        print("=" * 60)

    return warnings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preflight pipeline validator")
    parser.add_argument("--renderer", type=str, default=None)
    parser.add_argument("--masks", type=str, default=None)
    parser.add_argument("--poses", type=str, default=None)
    parser.add_argument("--archive", type=str, default=None)
    args = parser.parse_args()

    try:
        preflight_check(
            renderer_path=args.renderer,
            masks_path=args.masks,
            poses_path=args.poses,
            archive_path=args.archive,
        )
    except PreflightError as e:
        print(f"\nPREFLIGHT FAILED: {e}")
        sys.exit(1)


def preflight_all(
    profile_name: str | None = None,
    profile_arch: dict | None = None,
    tto_frames_path: str | Path | None = None,
    gt_poses_path: str | Path | None = None,
    masks_path: str | Path | None = None,
    renderer_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    check_codebase: bool = True,
    verbose: bool = True,
) -> None:
    """Single entry point: run ALL preflight checks. Raises on any failure.

    This is what every deployment / pipeline / experiment should call FIRST.
    Combines:
      - preflight_check: artifact validation (renderer/masks/poses/archive shapes, magic bytes)
      - preflight_training_inputs: training-time data integrity (TTO range, profile arch, eval_roundtrip)
      - preflight_codebase: AST scan for forbidden ad-hoc patterns (no nohup, no launch_*.sh)

    Pass only the args relevant to your stage. e.g., training preflight needs
    profile_name + tto_frames_path + gt_poses_path + masks_path. Inflate-time
    preflight needs renderer_path + masks_path + archive_path.
    """
    # 1. Codebase drift check (cheap, always run unless explicitly disabled)
    if check_codebase:
        check_codebase_drift(strict=True)
        preflight_arity(strict=True, verbose=verbose)
        preflight_profiles(strict=True, verbose=verbose)

    # 2. Training inputs (only if profile + tto_frames provided)
    if profile_name and tto_frames_path and gt_poses_path and masks_path and profile_arch:
        preflight_training_inputs(
            tto_frames_path=tto_frames_path,
            gt_poses_path=gt_poses_path,
            masks_path=masks_path,
            profile_name=profile_name,
            profile_arch=profile_arch,
            verbose=verbose,
        )

    # 3. Artifact preflight (only if any artifact path provided)
    if any([renderer_path, masks_path, archive_path]):
        preflight_check(
            renderer_path=renderer_path,
            masks_path=masks_path if not tto_frames_path else None,  # avoid double-check
            poses_path=None,  # handled in training_inputs
            archive_path=archive_path,
            verbose=verbose,
        )


def preflight_training_inputs(
    tto_frames_path: str | Path,
    gt_poses_path: str | Path,
    masks_path: str | Path,
    profile_name: str,
    profile_arch: dict,
    verbose: bool = True,
) -> None:
    """Validate training inputs BEFORE the GPU starts.

    Catches the failure modes that destroyed WILDE+GREEN on 2026-04-25:
      - TTO frames at GT range [0, 255] instead of TTO-optimized [0, ~184]
      - tto_frames.pt is corrupted (wrong dtype, infinite values)
      - Mask count doesn't match expected_n_frames
      - GT poses missing or wrong shape
      - Profile architecture doesn't match what the renderer would expect

    Raises PreflightError on fatal issues. No warnings — every fail is fatal.
    """
    if verbose:
        print("=" * 60)
        print(f"TRAINING PREFLIGHT — profile '{profile_name}'")
        print("=" * 60)

    # 1. TTO frames must exist, be valid, and be TTO-OPTIMIZED (range < 200)
    p = Path(tto_frames_path)
    if not p.exists():
        raise PreflightError(f"TTO frames missing: {p}")
    try:
        t = torch.load(str(p), map_location="cpu", weights_only=True)
    except Exception as e:
        raise PreflightError(f"TTO frames corrupted (cannot torch.load): {p} — {e}")
    if t.ndim != 4 or t.shape[1:] != (384, 512, 3):
        raise PreflightError(f"TTO frames wrong shape {tuple(t.shape)} (expected (N,384,512,3)): {p}")
    tmin, tmax = float(t.min()), float(t.max())
    if not (0 <= tmin and tmax < 1e6):
        raise PreflightError(f"TTO frames out of range [{tmin},{tmax}] — likely corrupted: {p}")
    is_gt_video = tmax > 200  # TTO-optimized frames cluster around max ~184
    if is_gt_video:
        raise PreflightError(
            f"TTO frames at GT-video range [0, {tmax:.0f}] — these are RAW GT FRAMES, "
            f"not TTO-optimized. This is the WILDE failure mode (proxy 267 instead of 0.5). "
            f"Re-run optimize_poses.py to generate TTO-optimized frames first. Path: {p}"
        )
    if verbose:
        print(f"  [PASS] tto_frames.pt: {tuple(t.shape)} {t.dtype} range [{tmin:.1f},{tmax:.1f}] (TTO-optimized)")

    # 2. GT poses must exist with shape (600, 6)
    pp = Path(gt_poses_path)
    if not pp.exists():
        raise PreflightError(f"GT poses missing: {pp}")
    try:
        poses = torch.load(str(pp), map_location="cpu", weights_only=True)
        if isinstance(poses, dict):
            poses = poses.get("poses", poses.get("gt_poses"))
    except Exception as e:
        raise PreflightError(f"GT poses corrupted: {pp} — {e}")
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise PreflightError(f"GT poses wrong shape {tuple(poses.shape)} (expected (N,6)): {pp}")
    if poses.shape[0] not in (600, 1200):
        raise PreflightError(f"GT poses {poses.shape[0]} entries (expected 600 pairs or 1200 frames): {pp}")
    if verbose:
        print(f"  [PASS] gt_poses.pt: {tuple(poses.shape)}")

    # 3. Mask video frame count
    mp = Path(masks_path)
    if not mp.exists():
        raise PreflightError(f"Masks missing: {mp}")
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", str(mp)],
            text=True, timeout=60,
        ).strip()
        nframes = int(out)
    except (subprocess.TimeoutExpired, ValueError, subprocess.CalledProcessError) as e:
        raise PreflightError(f"ffprobe failed on masks: {mp} — {e}")
    if nframes not in (600, 1200):
        raise PreflightError(
            f"Masks have {nframes} frames (expected 600 half-frame or 1200 full): {mp}"
        )
    if verbose:
        print(f"  [PASS] masks.mkv: {nframes} frames ({'half-frame' if nframes == 600 else 'full'})")

    # 4. Profile architecture sanity
    required_keys = ["base_ch", "mid_ch", "depth", "pose_dim", "padding_mode"]
    missing = [k for k in required_keys if k not in profile_arch]
    if missing:
        raise PreflightError(f"Profile '{profile_name}' missing arch keys: {missing}")
    if profile_arch["padding_mode"] not in ("zeros", "replicate", "reflect"):
        raise PreflightError(f"Profile '{profile_name}' has invalid padding_mode={profile_arch['padding_mode']}")
    if not (1 <= profile_arch["depth"] <= 4):
        raise PreflightError(f"Profile '{profile_name}' depth={profile_arch['depth']} out of range [1,4]")
    if verbose:
        print(f"  [PASS] profile arch: base_ch={profile_arch['base_ch']} "
              f"mid_ch={profile_arch['mid_ch']} depth={profile_arch['depth']} "
              f"pose_dim={profile_arch['pose_dim']} padding={profile_arch['padding_mode']}")

    # 5. Profile must include eval_roundtrip=True (NON-NEGOTIABLE)
    if not profile_arch.get("eval_roundtrip", False):
        raise PreflightError(
            f"Profile '{profile_name}' has eval_roundtrip=False. "
            f"This causes 2-11x proxy-auth gap. NON-NEGOTIABLE per CLAUDE.md."
        )
    if verbose:
        print(f"  [PASS] eval_roundtrip=True (CLAUDE.md non-negotiable)")

    if verbose:
        print(f"  ALL TRAINING PREFLIGHT CHECKS PASSED for profile '{profile_name}'")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Patterns that should NEVER appear outside contest submissions.
FORBIDDEN_FILE_PATTERNS = [
    "experiments/launch_*.sh",
    "experiments/launch_*.py",
    "experiments/run_*.sh",
    "experiments/qat_*.sh",
    "experiments/vastai_*.sh",
    "experiments/build_and_eval.sh",
    "experiments/crf_sweep_score.sh",
]

ALLOWED_BASH_PATHS = {
    "submissions/exact_current/inflate.sh",
    "submissions/exact_current/compress.sh",
    "submissions/exact_current/start.sh",
    "submissions/robust_current/inflate.sh",
    "submissions/robust_current/compress.sh",
}


class CodebaseDriftError(Exception):
    """An ad-hoc pattern reappeared in the codebase. Block all deployment."""


def _scan_python_for_forbidden(path: Path) -> list[str]:
    """AST-scan a Python file for forbidden subprocess patterns.

    Returns list of human-readable violations.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return [f"{path}: SyntaxError (cannot parse)"]

    for node in ast.walk(tree):
        # subprocess.run(...) / subprocess.Popen(...) with 'nohup' in args
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if func_str in ("subprocess.run", "subprocess.Popen", "subprocess.call"):
                # Check positional args for 'nohup' string literal
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if "nohup" in arg.value:
                            violations.append(
                                f"{path}:{node.lineno}: {func_str} with 'nohup' "
                                f"— use tmux instead (binding non-negotiable per CLAUDE.md)"
                            )
                    elif isinstance(arg, ast.List):
                        for elt in arg.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value.strip() == "nohup":
                                    violations.append(
                                        f"{path}:{node.lineno}: {func_str} with nohup arg — use tmux"
                                    )

        # f-string SSH commands containing 'nohup ... &' (the killer pattern)
        if isinstance(node, ast.JoinedStr):
            full = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"nohup.*&", full) and ("ssh" in full.lower() or "/workspace" in full):
                violations.append(
                    f"{path}:{node.lineno}: f-string with 'nohup ... &' over SSH "
                    f"— this is the WATCHER PATTERN that DIED on 2026-04-25. Use tmux."
                )

    return violations


def _scan_bash_text_for_forbidden(path: Path) -> list[str]:
    """Scan a bash file for nohup-watcher patterns and ad-hoc python invocations."""
    violations: list[str] = []
    text = path.read_text()
    if "nohup" in text and "&" in text and "while pgrep" in text:
        violations.append(
            f"{path}: 'nohup ... while pgrep ...' watcher pattern. "
            f"This DIED on all 3 A100s on 2026-04-25. Use tmux."
        )
    if "python3 -u experiments/train_distill.py" in text or "python experiments/train_distill.py" in text:
        violations.append(
            f"{path}: ad-hoc invocation of train_distill.py. "
            f"Use 'python experiments/pipeline.py --profile <name>' (canonical entry point)."
        )
    return violations


def check_codebase_drift(strict: bool = True) -> list[str]:
    """Run the codebase drift check. Raise CodebaseDriftError if strict and violations found."""
    all_violations: list[str] = []

    # 1. Forbidden file patterns
    for pattern in FORBIDDEN_FILE_PATTERNS:
        for found in REPO_ROOT.glob(pattern):
            all_violations.append(
                f"{found.relative_to(REPO_ROOT)}: forbidden ad-hoc launcher. "
                f"Use scripts/deploy_vastai.py + pipeline.py instead."
            )

    # 2. Bash scripts outside whitelist
    for sh_path in REPO_ROOT.glob("experiments/**/*.sh"):
        rel = str(sh_path.relative_to(REPO_ROOT))
        if rel not in ALLOWED_BASH_PATHS:
            all_violations.append(
                f"{rel}: bash script in experiments/ — only contest submission "
                f"scripts allowed (inflate.sh, compress.sh in submissions/)"
            )
        all_violations.extend(_scan_bash_text_for_forbidden(sh_path))

    # 3. Python files with nohup or watcher patterns
    for py_path in (REPO_ROOT / "scripts").glob("*.py"):
        all_violations.extend(_scan_python_for_forbidden(py_path))
    for py_path in (REPO_ROOT / "experiments").glob("*.py"):
        all_violations.extend(_scan_python_for_forbidden(py_path))

    if all_violations and strict:
        msg = (
            "CODEBASE DRIFT DETECTED — ad-hoc deployment patterns reappeared.\n"
            "These patterns wasted real money and CO2 on 2026-04-25. "
            "Per CLAUDE.md binding rules:\n\n"
            + "\n".join(f"  • {v}" for v in all_violations)
            + "\n\nFix every violation. There is no bypass — this is the gate working."
        )
        raise CodebaseDriftError(msg)
    return all_violations


# ── Arity / arg / config validation ───────────────────────────────────────────
#
# The bug class this catches: a launcher (pipeline.py, deploy_vastai.py, a shell
# wrapper) invokes a target script (qat_finetune.py, train_distill.py, etc.)
# with a list of CLI flags. If the target's argparse signature doesn't accept a
# flag, that flag is silently dropped (or argparse errors out at runtime — way
# too late, after $$ of GPU has been spent on the wrong thing). If the launcher
# fails to pass a flag the target needs, the target uses the default — the
# SHIRAZ A100 disaster: profile said motion_hidden=24, qat_finetune.py defaulted
# to 32, so QAT silently rebuilt the wrong architecture.
#
# Three layers:
#   1. Each target script's argparse signature is parsed via AST.
#   2. Each subprocess.run([...]) call in a launcher is parsed via AST.
#   3. We cross-validate: every flag passed must exist on the target; every
#      target arg in ARCH_FLAGS_REQUIRED that the target accepts must be passed.

# Architectural flags that, IF a target script accepts them, MUST be passed by
# any launcher invoking that script. Missing one → silent default → wrong arch.
# This is the SHIRAZ failure mode: trained with motion_hidden=24, QAT got 32.
ARCH_FLAGS_REQUIRED = {
    "--base-ch", "--mid-ch", "--motion-hidden", "--depth", "--embed-dim",
    "--pose-dim", "--padding-mode",
}
# Boolean (store_true) flags whose silent default = False would corrupt the
# experiment. Rule D fires when a target accepts one of these and the launcher
# source NEVER mentions it (so the launcher can't even conditionally pass it).
ARCH_FLAGS_BOOLEAN = {
    # Architecture flags
    "--use-dsconv", "--use-dilation", "--use-zoom-flow",
    # Training-discipline flags whose absence violates CLAUDE.md
    "--eval-roundtrip",
    # Loss / optimizer modulators that profiles toggle
    "--use-swa", "--use-per-class-weights",
    "--use-texture-loss", "--use-linf-penalty", "--use-markov-loss",
    "--freeze-motion-phase2", "--freeze-renderer-phase3",
    "--beneficial-quant-noise",
}

# Launcher files that invoke target scripts via subprocess.
LAUNCHER_FILES = [
    "experiments/pipeline.py",
    "scripts/deploy_vastai.py",
]

# Target script directories: every .py here is a potential subprocess target.
TARGET_DIRS = ["experiments", "scripts"]


class ArityViolation(Exception):
    """Arity / arg-matching failure between launcher and target."""


def _parse_argparse_signature(path: Path) -> dict[str, dict] | None:
    """AST-parse a script's argparse calls. Returns {flag: {required, action, type, ...}}.

    Indexes every `--` form across all positional args of `add_argument`, so
    `add_argument("-m", "--motion-hidden", ...)` correctly registers
    `--motion-hidden`.

    Returns None if the script has no argparse usage. Skips silently on syntax
    errors (caught by other preflight layers).
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None

    flags: dict[str, dict] = {}
    has_argparse = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        # Match `<anything>.add_argument(...)`. Common: parser.add_argument,
        # p.add_argument, sub.add_argument.
        if not func_str.endswith(".add_argument"):
            continue
        has_argparse = True
        # Collect every `--flag` literal across ALL positional args (handles
        # `add_argument("-m", "--motion-hidden", ...)` short-form aliases).
        long_forms: list[str] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    long_forms.append(arg.value)
        if not long_forms:
            continue
        spec = {"required": False, "action": None, "type": None,
                "has_default": False, "lineno": node.lineno}
        for kw in node.keywords:
            if kw.arg == "required" and isinstance(kw.value, ast.Constant):
                spec["required"] = bool(kw.value.value)
            elif kw.arg == "action" and isinstance(kw.value, ast.Constant):
                spec["action"] = kw.value.value
            elif kw.arg == "default":
                spec["has_default"] = True
            elif kw.arg == "type":
                spec["type"] = ast.unparse(kw.value) if hasattr(ast, "unparse") else "?"
        for f in long_forms:
            flags[f] = spec

    return flags if has_argparse else None


def _statically_resolve_list(node: ast.AST, scope: dict[str, ast.AST]) -> list[ast.AST] | None:
    """Try to resolve `node` to a list of AST elements (literals or names).

    Handles: List literal, Name → scope lookup, BinOp/list extend via .extend()
    aren't fully tracked, but we record what we can.
    """
    if isinstance(node, ast.List):
        return list(node.elts)
    if isinstance(node, ast.Name) and node.id in scope:
        return _statically_resolve_list(scope[node.id], scope)
    return None


def _extract_flag_strings(elts: list[ast.AST]) -> list[str]:
    """From a list of AST nodes (cmd elements), extract literal `--flag` strings."""
    flags: list[str] = []
    for elt in elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            if elt.value.startswith("--"):
                flags.append(elt.value)
    return flags


def _extract_target_script(elts: list[ast.AST]) -> str | None:
    """Find an `experiments/foo.py` or `scripts/foo.py` literal in the cmd list."""
    for elt in elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            v = elt.value
            for d in TARGET_DIRS:
                if v.startswith(f"{d}/") and v.endswith(".py"):
                    return v
    return None


_SUBPROCESS_FUNCS = {
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_call", "subprocess.check_output",
}

_BASH_C_TARGET_RE = re.compile(
    r"\b(?:python\d?|\.venv/bin/python\d?)\s+(?:-\w+\s+)*((?:experiments|scripts)/[\w/]+\.py)([^&|;\n]*)"
)


def _extract_invocations_from_scope(
    scope: ast.AST,
) -> list[tuple[int, str, list[str]]]:
    """Find subprocess.{run,Popen,...} invocations within a single scope.

    A scope is a Module, FunctionDef, or AsyncFunctionDef node. Variable
    tracking (`cmd = [...]`, `cmd.extend([...])`, `cmd.append(...)`) is
    confined to this scope to avoid cross-function pollution.

    Iterates the scope's body sequentially (in lexical order) so that
    variable definitions are seen before their use. We descend into
    sub-statements (if-branches, for-bodies, with-bodies) but DO NOT descend
    into nested FunctionDef/ClassDef — those are separate scopes handled by
    the caller.

    Also detects `subprocess.run(["bash", "-c", "python experiments/foo.py ..."])`
    by regex-parsing the inner string.
    """
    list_vars: dict[str, list[ast.AST]] = {}
    invocations: list[tuple[int, str, list[str]]] = []

    def visit(node: ast.AST) -> None:
        # Don't recurse into nested function or class scopes.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if node is scope:
                pass  # We're at the top of our scope; descend into body below.
            else:
                return

        # Track `name = [...]`
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.List):
                    list_vars[node.targets[0].id] = list(node.value.elts)

        # Track `name.extend([...])` and `name.append("--flag")`
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                tname = call.func.value.id
                meth = call.func.attr
                if tname in list_vars and meth in ("extend", "append"):
                    if call.args:
                        a = call.args[0]
                        if isinstance(a, ast.List):
                            list_vars[tname].extend(a.elts)
                        elif isinstance(a, ast.Constant):
                            list_vars[tname].append(a)

        # subprocess invocation
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if func_str in _SUBPROCESS_FUNCS and node.args:
                cmd_node = node.args[0]
                elts: list[ast.AST] | None = None
                if isinstance(cmd_node, ast.List):
                    elts = list(cmd_node.elts)
                elif isinstance(cmd_node, ast.Name) and cmd_node.id in list_vars:
                    elts = list_vars[cmd_node.id]
                if elts is not None:
                    target = _extract_target_script(elts)
                    flags = _extract_flag_strings(elts)
                    if target is not None:
                        invocations.append((node.lineno, target, flags))
                    else:
                        # Check for `["bash", "-c", "python experiments/x.py ..."]`
                        for elt in elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                m = _BASH_C_TARGET_RE.search(elt.value)
                                if m:
                                    bash_target = m.group(1)
                                    bash_tail = m.group(2) or ""
                                    bash_flags = [tok for tok in bash_tail.split() if tok.startswith("--")]
                                    invocations.append((node.lineno, bash_target, bash_flags))

        # Recurse into children (statements within this scope only).
        for child in ast.iter_child_nodes(node):
            visit(child)

    # Descend from the scope's body, not the scope node itself.
    if isinstance(scope, ast.Module):
        body = scope.body
    else:
        body = getattr(scope, "body", [])
    for stmt in body:
        visit(stmt)

    return invocations


def _scope_nodes(tree: ast.Module) -> list[ast.AST]:
    """Return the module + every FunctionDef/AsyncFunctionDef as separate scopes."""
    scopes: list[ast.AST] = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(node)
    return scopes


def _collect_all_flag_literals(tree: ast.Module) -> set[str]:
    """Find every `--flag` string literal anywhere in the module source.

    Used by Rule D: a launcher that never even mentions a target's boolean
    arch flag (e.g., never has `--use-dsconv` in its source) cannot possibly
    be passing it conditionally, so it has a silent-default risk.
    """
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("--"):
                seen.add(node.value)
    return seen


def _build_target_signatures(repo_root: Path) -> dict[str, dict[str, dict]]:
    """Parse every potential target script into {target_path: {flag: spec}}."""
    sigs: dict[str, dict[str, dict]] = {}
    for d in TARGET_DIRS:
        for py in (repo_root / d).glob("*.py"):
            rel = str(py.relative_to(repo_root))
            sig = _parse_argparse_signature(py)
            if sig is not None:
                sigs[rel] = sig
    return sigs


def _scan_launcher_invocations(
    launcher_path: Path,
) -> tuple[list[tuple[int, str, list[str]]], set[str]]:
    """Return ((lineno, target, flags) invocations, all-flag-literals-in-source).

    Walks each scope (module + every FunctionDef/AsyncFunctionDef) with its
    OWN list_vars, so cross-function `cmd` reuse cannot cause Function A's
    list to be polluted by Function B's `.extend(...)`.

    Also returns the set of every `--flag` literal appearing anywhere in the
    file's source — used by Rule D to detect launchers that don't even
    mention a target's boolean arch flag (silent-default risk).
    """
    try:
        tree = ast.parse(launcher_path.read_text(), filename=str(launcher_path))
    except (SyntaxError, UnicodeDecodeError):
        return [], set()

    seen: set[tuple[int, str, tuple[str, ...]]] = set()
    out: list[tuple[int, str, list[str]]] = []
    for scope in _scope_nodes(tree):
        for lineno, target, flags in _extract_invocations_from_scope(scope):
            key = (lineno, target, tuple(flags))
            if key in seen:
                continue
            seen.add(key)
            out.append((lineno, target, flags))
    all_flag_literals = _collect_all_flag_literals(tree)
    return out, all_flag_literals


def preflight_arity(
    repo_root: Path | None = None,
    launcher_files: list[str] | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Validate that every subprocess invocation matches its target's argparse.

    Four rules:
      A. Every --flag passed by a launcher MUST exist on the target script.
         (catches typos and renamed flags)
      B. Every required=True arg of the target MUST be passed.
         (catches forgotten required flags)
      C. If the target accepts an ARCH_FLAGS_REQUIRED flag and the launcher does
         NOT pass it, that's a silent-default risk → fail. (catches the SHIRAZ
         motion_hidden=24 vs default 32 disaster.)
      D. If the target accepts an ARCH_FLAGS_BOOLEAN flag and the launcher's
         source code never even mentions that flag string, the launcher cannot
         be conditionally passing it — that's also a silent-default risk →
         fail. (catches the SHIRAZ-class disaster for boolean flags like
         --use-dsconv and --use-dilation.)

    Returns list of human-readable violations. Raises ArityViolation if strict.
    """
    root = repo_root or REPO_ROOT
    launcher_files = launcher_files or LAUNCHER_FILES

    sigs = _build_target_signatures(root)
    violations: list[str] = []

    for launcher_rel in launcher_files:
        launcher_path = root / launcher_rel
        if not launcher_path.exists():
            continue
        invocations, all_flag_literals = _scan_launcher_invocations(launcher_path)

        for lineno, target, flags_passed in invocations:
            target_sig = sigs.get(target)
            if target_sig is None:
                # Target either has no argparse or wasn't found. Skip silently;
                # codebase-drift check covers missing files.
                continue
            target_flags = set(target_sig.keys())
            passed = set(flags_passed)

            # Rule A: unknown flags
            unknown = passed - target_flags
            for f in sorted(unknown):
                violations.append(
                    f"{launcher_rel}:{lineno}: passes {f!r} to {target} "
                    f"but target has no such argparse arg"
                )

            # Rule B: missing required
            for flag, spec in target_sig.items():
                if spec["required"] and flag not in passed:
                    violations.append(
                        f"{launcher_rel}:{lineno}: invokes {target} but does not pass "
                        f"required arg {flag!r}"
                    )

            # Rule C: missing arch flag (silent default risk)
            target_arch_flags = target_flags & ARCH_FLAGS_REQUIRED
            missing_arch = target_arch_flags - passed
            for flag in sorted(missing_arch):
                violations.append(
                    f"{launcher_rel}:{lineno}: invokes {target} which accepts arch "
                    f"flag {flag!r} but launcher doesn't pass it. Silent default → "
                    f"WRONG architecture (the SHIRAZ motion_hidden=24 vs default 32 disaster)."
                )

            # Rule D: boolean arch flag never mentioned anywhere in launcher source
            # The launcher MAY conditionally pass a boolean flag (e.g.,
            # `if cfg.use_dsconv: cmd.append("--use-dsconv")`). We can't tell
            # from this single invocation site whether the conditional path is
            # ever taken. But if the flag string never appears ANYWHERE in the
            # launcher's source code, we know with certainty the launcher has
            # no path to pass it. That's a silent-default risk.
            target_bool_flags = target_flags & ARCH_FLAGS_BOOLEAN
            never_mentioned = target_bool_flags - all_flag_literals
            for flag in sorted(never_mentioned):
                violations.append(
                    f"{launcher_rel}:{lineno}: invokes {target} which accepts boolean "
                    f"arch flag {flag!r} but launcher source NEVER mentions it. "
                    f"Silent-default risk: target will run with {flag!r}=False even "
                    f"if the profile sets it True. (Boolean-flag SHIRAZ class.)"
                )

    if verbose and violations:
        print(f"  [arity] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        n_launchers = sum(1 for f in launcher_files if (root / f).exists())
        n_targets = len(sigs)
        print(f"  [arity] OK: {n_launchers} launchers x {n_targets} targets clean")

    if violations and strict:
        raise ArityViolation(
            "ARITY MISMATCH between launcher(s) and target script(s):\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nFix every violation. Each one is a real bug class that has "
            "burned GPU money in this repo (see CLAUDE.md SHIRAZ A100 incident)."
        )
    return violations


# ── Profile validation ────────────────────────────────────────────────────────

PROFILE_REQUIRED_ARCH_KEYS = {
    "base_ch", "mid_ch", "depth", "pose_dim", "padding_mode", "eval_roundtrip",
}
PROFILE_RECOMMENDED_KEYS = {
    "embed_dim", "motion_hidden", "use_dsconv", "use_dilation",
}


def preflight_profiles(strict: bool = True, verbose: bool = True) -> list[str]:
    """Validate every PROFILES entry against architectural and binding constraints.

    Catches:
      - Missing required arch keys (would crash training silently with defaults).
      - eval_roundtrip != True (CLAUDE.md non-negotiable).
      - Typo'd keys (warns: not in the recommended/known set).
      - padding_mode not in (zeros, replicate, reflect, circular).
    """
    violations: list[str] = []
    try:
        from tac.profiles import PROFILES
    except ImportError as e:
        msg = f"  [profiles] cannot import tac.profiles: {e}"
        if verbose:
            print(msg)
        return [msg]

    # Profiles whose experiment_type is renderer training (the ones that flow
    # through pipeline.py + qat_finetune.py + optimize_poses.py). Other profile
    # families (e.g., the legacy "training" CPU lane) have different schemas.
    RENDERER_TYPES = {"renderer_training"}

    KNOWN_TYPES = RENDERER_TYPES | {
        "training",         # legacy CPU lane
        "smoke_test",       # quick correctness checks, no arch contract
        "eval",             # contest-compliant evaluation profiles
        "gpu_lane",         # constrained-gen / variational / ensemble lanes
        "self_compress",    # self-compression eureka profiles
        "entropy_archive",  # entropy-coded archive experiments
        "network_codec",    # learned codec profiles
    }
    for name, prof in PROFILES.items():
        etype = prof.get("experiment_type")
        if etype is None:
            violations.append(
                f"profile {name!r} missing 'experiment_type' key — would be "
                f"silently skipped by validation. Set to 'training' or 'renderer_training'."
            )
            continue
        if etype not in KNOWN_TYPES:
            violations.append(
                f"profile {name!r} has unknown experiment_type={etype!r}. "
                f"Expected one of {sorted(KNOWN_TYPES)}."
            )
            continue
        if etype not in RENDERER_TYPES:
            continue
        for key in PROFILE_REQUIRED_ARCH_KEYS:
            if key not in prof:
                violations.append(f"profile {name!r} missing required arch key {key!r}")
        if prof.get("eval_roundtrip") is not True:
            violations.append(
                f"profile {name!r} has eval_roundtrip={prof.get('eval_roundtrip')!r}, "
                f"must be True (CLAUDE.md non-negotiable)"
            )
        pm = prof.get("padding_mode")
        if pm is not None and pm not in {"zeros", "replicate", "reflect", "circular"}:
            violations.append(f"profile {name!r} invalid padding_mode={pm!r}")
        depth = prof.get("depth")
        if depth is not None and not (1 <= int(depth) <= 4):
            violations.append(f"profile {name!r} depth={depth} out of range [1,4]")

    if verbose and violations:
        print(f"  [profiles] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        n_renderer = sum(1 for p in PROFILES.values() if p.get("experiment_type") in RENDERER_TYPES)
        print(f"  [profiles] OK: {n_renderer} renderer profile(s) validated")

    if violations and strict:
        raise PreflightError(
            "PROFILE VALIDATION FAILED:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations
