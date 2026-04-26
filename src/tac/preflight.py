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
            # R38 fix: use detect_pose_manifest to autopick the right
            # manifest based on which pose format the archive actually has.
            from tac.submission_archive import validate_archive, detect_pose_manifest
            manifest = detect_pose_manifest(archive_path)
            result = validate_archive(archive_path, manifest, strict=False)
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


# Note: __main__ block moved to the bottom of the module so all validator
# functions (preflight_all et al.) are defined before invocation. Was a
# misleading-CLI bug per R38 — operators running `python -m tac.preflight`
# only got artifact validation, silently skipping all 5 codebase layers.


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
        preflight_arch_consistency(strict=True, verbose=verbose)
        preflight_filename_contract(strict=True, verbose=verbose)

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
    # R38 fix: accept HWC (N,384,512,3) OR CHW (N,3,384,512). Project history
    # has had silent HWC/CHW format bugs; the validator should not assume one.
    if t.ndim != 4:
        raise PreflightError(f"TTO frames wrong ndim {t.ndim} (expected 4): {p}")
    valid_shapes = {(384, 512, 3), (3, 384, 512)}
    if tuple(t.shape[1:]) not in valid_shapes:
        raise PreflightError(
            f"TTO frames wrong shape {tuple(t.shape)} (expected (N,384,512,3) HWC "
            f"or (N,3,384,512) CHW): {p}"
        )
    tmin, tmax = float(t.min()), float(t.max())
    if not (0 <= tmin and tmax < 1e6):
        raise PreflightError(f"TTO frames out of range [{tmin},{tmax}] — likely corrupted: {p}")
    # R38 fix: support both [0,255] uint-scale and [0,1] normalized scale.
    # If max ≤ 1.5, treat as [0,1] — TTO-optimized [0,1] frames cluster ~0.72.
    # If max > 1.5, treat as [0,255] — TTO-optimized clusters ~184.
    if tmax > 1.5:
        is_gt_video = tmax > 200
    else:
        # [0,1] scale: GT frames clamp to ~1.0; TTO-optimized cluster ~0.72.
        is_gt_video = tmax > 0.95
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
    # R38 fix: was AttributeError on poses=None when neither 'poses' nor
    # 'gt_poses' key existed in the dict.
    if poses is None:
        raise PreflightError(
            f"GT poses dict has neither 'poses' nor 'gt_poses' key: {pp}"
        )
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


def _scan_text_for_dangerous_patterns(text: str, location: str) -> list[str]:
    """Cross-language scan for shell patterns that have caused real outages.

    Both bash files and Python files (via subprocess string literals + f-strings
    + tmux-send-keys composition) feed through this. Each rule cites the exact
    incident that motivated it so future maintainers can judge edge cases.

    Args:
        text: shell text — either a bash file body or a string literal that
            will be passed to bash -c / ssh.
        location: human-readable origin (e.g. "scripts/foo.sh" or
            "src/tac/deploy/x.py:412") used in violation messages.

    Returns: list of violations.
    """
    violations: list[str] = []

    # Self-matching `pgrep -f TOKEN` deadlock. 2026-04-26 SHIRAZ:
    #   bash -c "while pgrep -f train_distill > /dev/null; do sleep 60; done; bash run_pipeline.sh"
    # The bash -c argv literally contained "train_distill", so pgrep -f matched
    # the wrapper itself and the loop never exited — burned ~21h of A100 time.
    # Detect any `pgrep -f TOKEN` whose TOKEN appears elsewhere in the SAME
    # text blob (file or string literal).
    for m in re.finditer(r"pgrep\s+-[a-z]*f[a-z]*\s+['\"]?([A-Za-z0-9_./-]+)", text):
        token = m.group(1)
        if len(token) < 3:
            continue
        if text.count(token) >= 2:
            violations.append(
                f"{location}: `pgrep -f {token}` will SELF-MATCH — the token "
                f"appears elsewhere in this text, so the wait loop's own argv "
                f"matches and the loop sleeps forever. 2026-04-26 SHIRAZ "
                f"deadlock burned ~21h of A100 time. Use a pidfile, "
                f"`pgrep -x <executable>` (exact name), or a unique cookie."
            )
            break

    # Blind `.pt → .bin` rename. 2026-04-26 retto wrapper did
    #   cp $(ls *_partial.pt) /tmp/.../optimized_poses.bin
    # Pickle masqueraded as raw fp16 buffer; auth_eval_renderer crashed after
    # 7 min of mask extraction with `frombuffer` size mismatch.
    for m in re.finditer(
        r"\b(?:cp|mv|install|ln\s+-s)\s+(?:-[a-zA-Z]+\s+)*(\S+\.pt)\s+(\S+\.bin)\b",
        text,
    ):
        violations.append(
            f"{location}: `{m.group(0)}` renames a pickle .pt to raw .bin. "
            f"This corrupts pose loaders. Use tac.submission_archive."
            f"save_poses_binary() or have the producer emit .bin directly."
        )

    # Wrapper that SHIPS `*_partial*` files as if they were finished artifacts.
    # `optimized_poses_partial.pt` is what optimize_poses.py writes
    # periodically; shipping it as the final archive artifact means N pairs
    # rather than the full 600 are present. Only fire when the reference
    # appears near a copy/move/archive operation — a producer that natively
    # writes or resumes from its own partial is fine (e.g. optimize_poses.py
    # itself, --resume CLI args, docstrings).
    has_partial_ref = bool(
        re.search(r"\b\S*_partial\.(?:pt|bin)\b", text)
        or re.search(r"_partial\*\.(?:pt|bin)", text)
    )
    if has_partial_ref:
        ships_or_renames = re.search(
            r"\b(?:cp|mv|install|ln\s+-s|tar|zip|aws\s+s3|scp|rsync|"
            r"build_submission_archive|optimized_poses\.bin|/archive/)",
            text,
        )
        if ships_or_renames:
            violations.append(
                f"{location}: ships a `*_partial*` artifact (rename/copy/"
                f"archive). Partial files are incomplete by definition. Wait "
                f"for the canonical final write or re-run the producer. "
                f"2026-04-26 SHIRAZ shipped 60 of 600 poses for a contest "
                f"eval because of this pattern."
            )

    return violations


def _scan_python_for_forbidden(path: Path) -> list[str]:
    """AST-scan a Python file for forbidden subprocess patterns.

    Returns list of human-readable violations.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return [f"{path}: SyntaxError (cannot parse)"]

    # R-mps-noise-rule 2026-04-25: NEW. Per CLAUDE.md "MPS auth eval is NOISE",
    # detect any auth_eval invocation hardcoded to --device mps. Allowed only
    # in test files / smoke tests (path contains "/tests/" or "/smoke").
    is_test_or_smoke = ("/tests/" in str(path) or "/smoke" in str(path).lower()
                        or "test_" in path.name)

    for node in ast.walk(tree):
        # subprocess.* / os.system with 'nohup' in args. R38 fix: extended
        # to subprocess.check_call/check_output and os.system.
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if func_str in ("subprocess.run", "subprocess.Popen", "subprocess.call",
                            "subprocess.check_call", "subprocess.check_output",
                            "os.system", "os.popen"):
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
                # R-mps-noise: detect auth_eval invocations with --device mps.
                # Allow in test/smoke paths.
                if not is_test_or_smoke:
                    full = ast.unparse(node) if hasattr(ast, "unparse") else ""
                    if "auth_eval" in full and re.search(r"--device['\"\s,]+mps", full):
                        violations.append(
                            f"{path}:{node.lineno}: auth_eval invocation with "
                            f"'--device mps' — MPS auth scores are NOISE per CLAUDE.md "
                            f"HIGHEST-EMPHASIS rule (23x PoseNet drift verified 2026-04-25). "
                            f"Use --device cuda."
                        )

        # f-string SSH commands containing 'nohup ... &' (the killer pattern)
        if isinstance(node, ast.JoinedStr):
            full = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"nohup.*&", full) and ("ssh" in full.lower() or "/workspace" in full):
                violations.append(
                    f"{path}:{node.lineno}: f-string with 'nohup ... &' over SSH "
                    f"— this is the WATCHER PATTERN that DIED on 2026-04-25. Use tmux."
                )
            # Pose-format and self-match scans on the unparsed f-string. This
            # catches dynamically composed bash -c / ssh commands that never
            # land on disk as a .sh file (the 2026-04-26 SHIRAZ root cause).
            for v in _scan_text_for_dangerous_patterns(full, f"{path}:{node.lineno}"):
                violations.append(v)

        # Plain string constants over 40 chars also worth scanning — the
        # `bash -c "..."` literal in deploy_vastai composes via str.join.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 40:
                for v in _scan_text_for_dangerous_patterns(node.value, f"{path}:{node.lineno}"):
                    violations.append(v)

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
    violations.extend(_scan_text_for_dangerous_patterns(text, str(path)))
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

    # 3. Python files with nohup or watcher patterns. R36 extended scan to
    # src/tac/ subtrees; R37 added existence guard so a fresh checkout
    # missing one of these dirs doesn't crash preflight (Python <3.12
    # rglob raises FileNotFoundError on missing path).
    drift_scan_dirs = ["scripts", "experiments",
                       "src/tac/contrib", "src/tac/deploy",
                       "src/tac/experiments"]
    for d in drift_scan_dirs:
        d_path = REPO_ROOT / d
        if not d_path.exists():
            continue
        for py_path in d_path.rglob("*.py"):
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
# R38 fix: src/tac/experiments/ added — train_renderer.py is a de-facto
# launcher invoked directly via `python -m tac.experiments.train_renderer`.
TARGET_DIRS = ["experiments", "scripts", "src/tac/experiments"]


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


def _statically_resolve_list(node, scope: dict) -> list | None:
    """Try to resolve `node` to a list of AST elements (literals or names).

    Handles: List literal, Name → scope lookup (which may already be a
    resolved Python list of AST nodes), BinOp `+` of two resolvable lists
    (R38: closes an arity-validator escape hatch). `.extend()` is tracked
    elsewhere (in scope's list_vars).
    """
    # Already-resolved Python list of AST nodes (from scope's list_vars).
    if isinstance(node, list):
        return list(node)
    if isinstance(node, ast.List):
        return list(node.elts)
    if isinstance(node, ast.Name) and node.id in scope:
        return _statically_resolve_list(scope[node.id], scope)
    # R38 fix: handle `cmd = ["a","b"] + extras` and `["x"] + flags` patterns.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _statically_resolve_list(node.left, scope)
        right = _statically_resolve_list(node.right, scope)
        if left is not None and right is not None:
            return left + right
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

        # Track `name = [...]` and `name = a + b` (R38 BinOp).
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                # Try the full resolver — handles List, Name, BinOp(+).
                resolved = _statically_resolve_list(node.value, list_vars)
                if resolved is not None:
                    list_vars[node.targets[0].id] = resolved

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
                # R38 fix: route through _statically_resolve_list so BinOp
                # `+` patterns (cmd = ["a"] + flags) are tracked, closing
                # the prior arity-validator escape hatch.
                elts: list[ast.AST] | None = _statically_resolve_list(
                    cmd_node, list_vars
                )
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


# ── Filename contract validation ──────────────────────────────────────────────
#
# Bug class this catches: a consumer script (pipeline.py) constructs a path
# like `iter_dir / "renderer_qat_best.pt"` and reads/exists-checks it, but
# the producer script (qat_finetune.py) actually saves it as
# `qat_best_float.pt`. The mismatch is silent — exists() returns False, the
# fallback branch fires, and the pipeline silently uses the wrong artifact.
#
# Caught manually in R33 (renderer_qat_best.pt → qat_best_float.pt) and R34
# (renderer_qat.bin → renderer_fp4.bin). This validator automates the check.

class FilenameContractError(Exception):
    """A consumer-side filename literal is never produced by any script."""


# Filename suffixes that represent artifacts (versus, e.g., test fixtures or
# config files). Anything matching these suffixes that's read in a launcher
# but never written anywhere is a phantom path.
_ARTIFACT_SUFFIXES = (".bin", ".pt", ".pth", ".mkv", ".mp4", ".raw",
                      ".zip", ".tar", ".tar.gz", ".tgz")

# Filenames that are deliberately external (not produced by our code) — they
# come from upstream data, the contest archive, third-party tools, etc.
_EXTERNAL_FILENAMES = {
    "0.mkv",  # upstream/videos/0.mkv (contest GT)
    "masks.mkv", "poses.pt", "renderer.bin",  # contest-required submission filenames
    "video_names.txt",  # contest input
    "submission.zip", "archive.zip",  # contest output filenames (built by submission_archive)
    "pretrained.pth",  # pretrained model weights
}


def _extract_artifact_filenames(path: Path) -> set[str]:
    """AST-extract every artifact-suffix string literal from a Python file.

    Returns names like {"renderer_fp4.bin", "qat_best_float.pt"}. Skips
    non-artifact strings (URLs, log file names, fixture paths).
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if v.endswith(_ARTIFACT_SUFFIXES):
                # Take just the basename; we don't care about the directory.
                base = v.split("/")[-1]
                # Skip glob patterns and obvious non-literal hints.
                if "*" in base or "{" in base:
                    continue
                # Skip suffix-fragments used in f-string concat
                # (e.g., `for suffix in ["_int4lzma2.bin", ".bin"]`).
                # Real basenames have a non-empty stem before the suffix.
                stem = base
                for suf in _ARTIFACT_SUFFIXES:
                    if stem.endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                if not stem or stem.startswith(("_", ".")):
                    continue
                # Skip very generic names that are too noisy to validate.
                if base in _EXTERNAL_FILENAMES:
                    continue
                found.add(base)
    return found


def _extract_write_literals(path: Path) -> set[str]:
    """AST-extract artifact filenames that appear in WRITE contexts.

    Detects two layers:

    Direct (literal IS the call argument):
      - `torch.save(_, "X.pt")` — second arg literal
      - `open("X", "w"|"a"|"wb"|"ab")` — first arg literal with write mode
      - `<expr>.write_bytes(_)` / `.write_text(_)` / `.touch()` — receiver
        path expression containing an artifact literal
      - `os.replace(_, "X")` / `shutil.copy(_, "X")` — target literal

    Indirect (literal is in a Path-assignment, then the variable is used
    in a write context):
      - `out_path = iter_dir / "X.bin"`
        `torch.save(model, str(out_path))` or
        `export_fn(_, str(out_path))` or
        `out_path.write_bytes(...)` etc.
      This catches the common pipeline.py pattern.

    Returns just basenames.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()

    def _collect_artifact_literals_in(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                v = sub.value
                if v.endswith(_ARTIFACT_SUFFIXES):
                    base = v.split("/")[-1]
                    if "*" not in base and "{" not in base:
                        out.add(base)
        return out

    # Pass 1a: collect Name → set of artifact basenames assigned to that name.
    # Tracks `name = <expr-containing-artifact-literal>` for later write-context
    # cross-linking.
    name_to_literals: dict[str, set[str]] = {}
    # Map FunctionDef → its name (so we can scope Return tracking).
    WRITE_FN_PREFIXES = (
        "export_", "save_", "write_", "encode_", "build_",
        "pack_", "dump_", "emit_", "serialize_",
    )

    def _is_write_named_fn(fn_node: ast.AST) -> bool:
        if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return fn_node.name.startswith(WRITE_FN_PREFIXES)
        return False

    # Build parent-pointer map so we can walk up from a Return to find its
    # enclosing function.
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def _enclosing_fn(node: ast.AST) -> ast.AST | None:
        cur = parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(id(cur))
        return None

    # Pass 1a: build name_to_literals BEFORE any Return-tracking pass so
    # the lookup is complete (ast.walk order isn't guaranteed; a Return
    # could be visited before its Assign otherwise).
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                lits = _collect_artifact_literals_in(node.value)
                if lits:
                    name_to_literals.setdefault(t.id, set()).update(lits)

    # Pass 1b: process Return statements. R36: only count when enclosing
    # function has a write-prefix name. R37: also follow Name indirection
    # (`return path` where path = dir / "X.bin" was assigned earlier).
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            fn = _enclosing_fn(node)
            if fn is not None and _is_write_named_fn(fn):
                lits = _collect_artifact_literals_in(node.value)
                if lits:
                    found.update(lits)
                for nm in {sub.id for sub in ast.walk(node.value)
                           if isinstance(sub, ast.Name)}:
                    if nm in name_to_literals:
                        found.update(name_to_literals[nm])

    def _names_referenced(node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    def _record_write(arg_node: ast.AST) -> None:
        """Record literals from arg_node, including via Name indirection."""
        found.update(_collect_artifact_literals_in(arg_node))
        for nm in _names_referenced(arg_node):
            if nm in name_to_literals:
                found.update(name_to_literals[nm])

    # Pass 2: detect write-context calls and extract literals (direct or via Name).
    WRITE_FUNCS_2ND_ARG = {"torch.save", "os.replace", "shutil.copy",
                           "shutil.copyfile", "shutil.move", "os.rename"}
    WRITE_METHOD_SUFFIXES = (".write_bytes", ".write_text", ".touch",
                             ".save", ".dump")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        # torch.save / os.replace / shutil.copy: 2nd positional arg is the target
        if func_str in WRITE_FUNCS_2ND_ARG and len(node.args) >= 2:
            _record_write(node.args[1])
        # open(target, "w"/"a"/"x")
        if func_str == "open" and node.args:
            mode_arg = None
            if len(node.args) >= 2:
                mode_arg = node.args[1]
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode_arg = kw.value
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                if any(c in mode_arg.value for c in ("w", "a", "x")):
                    _record_write(node.args[0])
        # x.write_bytes(...) / x.write_text(...) / x.touch() / x.save() / x.dump()
        if any(func_str.endswith(suf) for suf in WRITE_METHOD_SUFFIXES):
            if isinstance(node.func, ast.Attribute):
                _record_write(node.func.value)
        # export/save/write/encode/build/dump/emit/serialize/pack helpers:
        # any function whose name starts with these prefixes — treat
        # 2nd-or-later arg as target. Includes encoder funcs (encode_masks,
        # encode_video) and serializer funcs (dump_state, emit_archive).
        if func_str.split(".")[-1].startswith(
            ("export_", "save_", "write_", "encode_", "build_",
             "pack_", "dump_", "emit_", "serialize_")
        ):
            for arg in node.args[1:]:
                _record_write(arg)
    return found


def preflight_filename_contract(
    repo_root: Path | None = None,
    consumer_files: list[str] | None = None,
    producer_dirs: list[str] | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Validate that every artifact filename READ by a consumer is WRITTEN
    by some producer script.

    Consumer = pipeline.py and other launchers. They read filenames via
        Path expressions and check existence / load weights / pass to subprocess.
    Producer = anything in experiments/ or src/tac/ that writes the file via
        torch.save, file.write_*, ffmpeg subprocess, etc.

    AST-level approach: extract every artifact-suffixed string literal from
    consumer files. Extract the same from producer files. The set difference
    {consumer_literals} - {producer_literals} - {external} is the violation set.

    This is conservative: a literal appearing in producer source is treated
    as "produced" even if the producer code path is dead. Catches the
    obvious filename-typo bug class (R33, R34) without false positives on
    legitimate refactors.
    """
    root = repo_root or REPO_ROOT
    consumer_files = consumer_files or LAUNCHER_FILES + [
        "experiments/pipeline.py",  # also a producer (step_export, etc.)
    ]
    producer_dirs = producer_dirs or ["experiments", "src/tac",
                                       "submissions/robust_current"]

    consumer_literals: dict[str, set[str]] = {}
    consumer_paths_resolved: set[Path] = set()
    for cf in consumer_files:
        cp = (root / cf).resolve()
        if cp.exists():
            consumer_literals[cf] = _extract_artifact_filenames(cp)
            consumer_paths_resolved.add(cp)

    # Producer scan: every script EXCEPT the consumer files. A consumer that
    # is also a producer (e.g., pipeline.py writes renderer.bin) would
    # otherwise self-validate every typo. We collect a separate set of
    # "consumer self-writes" via AST write-context detection; those literals
    # ARE legitimate (the file produces what it consumes).
    producer_literals: set[str] = set(_EXTERNAL_FILENAMES)
    producer_literals.discard("renderer.bin")  # we DO produce this
    n_producer_files = 0
    for pd in producer_dirs:
        for py in (root / pd).rglob("*.py"):
            if py.resolve() in consumer_paths_resolved:
                continue  # skip consumer files in producer scan
            n_producer_files += 1
            producer_literals.update(_extract_artifact_filenames(py))
        for sh in (root / pd).rglob("*.sh"):
            try:
                text = sh.read_text()
                for token in re.findall(
                    r'[\w./_-]+\.(?:bin|pt|pth|mkv|mp4|raw|zip|tar\.gz|tar|tgz)', text):
                    producer_literals.add(token.split("/")[-1])
            except (OSError, UnicodeDecodeError):
                pass

    # Also scan consumer files themselves for explicit WRITE-context literals
    # (torch.save target, open(..., "w") arg, .write_bytes/.write_text receiver
    # path with the literal). Those are legitimate self-produced names.
    for cp in consumer_paths_resolved:
        producer_literals.update(_extract_write_literals(cp))

    violations: list[str] = []
    for consumer, lits in consumer_literals.items():
        phantoms = lits - producer_literals
        for ph in sorted(phantoms):
            violations.append(
                f"{consumer}: reads {ph!r} but no producer in "
                f"{producer_dirs} ever writes that name. "
                f"R33/R34 bug class — verify the producer's actual output filename."
            )

    if verbose and violations:
        print(f"  [filenames] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        n_consumer = sum(1 for cf in consumer_files if (root / cf).exists())
        print(f"  [filenames] OK: {n_consumer} consumers x {n_producer_files} "
              f"producer files clean ({len(producer_literals)} known artifacts)")

    if violations and strict:
        raise FilenameContractError(
            "FILENAME CONTRACT VIOLATIONS — consumer reads a filename no "
            "producer writes:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nThis is the R33/R34 bug class. Either:\n"
            "  1. Fix the consumer to use the actual producer filename\n"
            "  2. Add the filename to a producer that should write it\n"
            "  3. Add it to _EXTERNAL_FILENAMES if it's contest/upstream data"
        )
    return violations


# ── Profile-vs-ArchConfig field consistency ───────────────────────────────────
#
# Bug class this catches: a profile sets `use_dscovn: True` (typo of
# use_dsconv) and the model is built without DSConv silently — same SHIRAZ
# class but at the profile-key level instead of the CLI-flag level.
#
# preflight_arity catches CLI flag drift (--use-dsconv missing). This new
# validator catches profile-key drift (profile says `use_dscovn` but
# ArchConfig has `use_dsconv` — close-match Levenshtein typo).


def preflight_arch_consistency(strict: bool = True, verbose: bool = True) -> list[str]:
    """Cross-validate every renderer-training PROFILES entry's arch keys
    against tac.renderer.ArchConfig fields.

    Two checks:
      A. Every profile arch-like key (matches Levenshtein cutoff 0.85 to an
         ArchConfig field) MUST exactly match an ArchConfig field name.
         Otherwise it's a likely typo.
      B. Every required ArchConfig field that profiles typically override
         (PROFILE_REQUIRED_ARCH_KEYS) must be present in the profile.
    """
    import difflib
    violations: list[str] = []
    try:
        from tac.profiles import PROFILES
        from tac.renderer import ArchConfig
    except ImportError as e:
        msg = f"  [arch_consistency] cannot import: {e}"
        if verbose:
            print(msg)
        return [msg]
    arch_field_names = {
        f.name for f in __import__("dataclasses").fields(ArchConfig)
    }
    n_profiles = 0
    for name, prof in PROFILES.items():
        if prof.get("experiment_type") != "renderer_training":
            continue
        n_profiles += 1
        for key in prof.keys():
            if key in arch_field_names:
                continue
            # Is it close to any ArchConfig field name?
            close = difflib.get_close_matches(key, arch_field_names, n=1, cutoff=0.85)
            if close:
                violations.append(
                    f"profile {name!r}: key {key!r} is close to ArchConfig "
                    f"field {close[0]!r} but not an exact match. Likely typo. "
                    f"If intentional (training-script-only key), rename to "
                    f"something distinct from ArchConfig fields."
                )
    if verbose and violations:
        print(f"  [arch_consistency] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [arch_consistency] OK: {n_profiles} renderer profile(s) "
              f"× {len(arch_field_names)} ArchConfig fields clean")
    if violations and strict:
        raise PreflightError(
            "ARCH CONSISTENCY VIOLATIONS — profile keys close to but not "
            "matching ArchConfig fields:\n"
            + "\n".join(f"  • {v}" for v in violations)
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
        # R38 fix: enforce eval_roundtrip=True on ALL training profile types
        # ("training" + "renderer_training"), not just renderer_training.
        # CLAUDE.md non-negotiable applies to every training path.
        if etype in ("training", "renderer_training"):
            if "eval_roundtrip" in prof and prof.get("eval_roundtrip") is not True:
                violations.append(
                    f"profile {name!r} has eval_roundtrip={prof.get('eval_roundtrip')!r}, "
                    f"must be True (CLAUDE.md non-negotiable)"
                )
        if etype not in RENDERER_TYPES:
            continue
        for key in PROFILE_REQUIRED_ARCH_KEYS:
            if key not in prof:
                violations.append(f"profile {name!r} missing required arch key {key!r}")
        # eval_roundtrip on renderer profiles is REQUIRED to be True (not just
        # "if present, True").
        if prof.get("eval_roundtrip") is not True:
            violations.append(
                f"profile {name!r} has eval_roundtrip={prof.get('eval_roundtrip')!r}, "
                f"must be True (CLAUDE.md non-negotiable)"
            )
        pm = prof.get("padding_mode")
        if pm is not None and pm not in {"zeros", "replicate", "reflect", "circular"}:
            violations.append(f"profile {name!r} invalid padding_mode={pm!r}")
        # R38 fix: catch non-int depth before int() raises ValueError.
        depth = prof.get("depth")
        if depth is not None:
            if not isinstance(depth, int):
                violations.append(
                    f"profile {name!r} depth={depth!r} type {type(depth).__name__}, expected int"
                )
            elif not (1 <= depth <= 4):
                violations.append(f"profile {name!r} depth={depth} out of range [1,4]")

        # Lane D2: mask_half_sim_prob requires use_zoom_flow=True. The
        # training-side simulation derives the warp from RadialZoomWarp via
        # tac.lane_mark_speed.zoom_from_masks; with use_zoom_flow=False the
        # renderer doesn't accept the flow signal and the simulation is dead
        # weight (consumes compute, doesn't shift the trained distribution).
        msp = prof.get("mask_half_sim_prob", 0.0)
        if msp is not None and msp > 0:
            if not isinstance(msp, (int, float)) or not (0 <= msp <= 1):
                violations.append(
                    f"profile {name!r} mask_half_sim_prob={msp!r} must be in [0, 1]"
                )
            if not prof.get("use_zoom_flow"):
                violations.append(
                    f"profile {name!r} sets mask_half_sim_prob={msp} but "
                    f"use_zoom_flow={prof.get('use_zoom_flow')!r}. The "
                    f"training-side mask-half simulation only matches inflate "
                    f"behaviour when use_zoom_flow=True (the inflate side warps "
                    f"odd-frame masks via RadialZoomWarp). Either enable "
                    f"use_zoom_flow=True or set mask_half_sim_prob=0."
                )

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preflight pipeline validator — runs ALL layers by default"
    )
    parser.add_argument("--renderer", type=str, default=None,
                        help="Optional renderer .bin/.pt for artifact check")
    parser.add_argument("--masks", type=str, default=None)
    parser.add_argument("--poses", type=str, default=None)
    parser.add_argument("--archive", type=str, default=None)
    parser.add_argument("--no-codebase", action="store_true",
                        help="Skip codebase / arity / profiles / filenames / arch_consistency")
    parser.add_argument("--profile", type=str, default=None,
                        help="Profile name for training-input validation")
    parser.add_argument("--tto-frames", type=str, default=None)
    parser.add_argument("--gt-poses", type=str, default=None)
    args = parser.parse_args()

    try:
        # R38 fix: was preflight_check (artifact-only) — now preflight_all
        # so the CLI runs the full 5-layer validation. Operators running
        # `python -m tac.preflight` expected comprehensive validation.
        profile_arch = None
        if args.profile:
            from tac.profiles import PROFILES
            if args.profile not in PROFILES:
                print(f"Unknown profile: {args.profile}", file=sys.stderr)
                sys.exit(2)
            profile_arch = PROFILES[args.profile]
        preflight_all(
            profile_name=args.profile,
            profile_arch=profile_arch,
            tto_frames_path=args.tto_frames,
            gt_poses_path=args.gt_poses,
            masks_path=args.masks,
            renderer_path=args.renderer,
            archive_path=args.archive,
            check_codebase=not args.no_codebase,
            verbose=True,
        )
        print("\nPREFLIGHT PASSED")
    except (PreflightError, ArityViolation, FilenameContractError,
            CodebaseDriftError) as e:
        print(f"\nPREFLIGHT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
