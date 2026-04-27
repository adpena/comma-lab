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
        # 2026-04-27 codex R5-2 Finding #2: scanner flipped to strict after
        # all 19 known violations were fixed (12 dead resolvers in
        # train_renderer.py + 7 dead imports across test_fp4_quality /
        # train_distill / benchmark_mlx / train_renderer). The scanner now
        # blocks the same silent-default class it was added to prevent.
        # See feedback_dead_resolver_violations_20260427 memory entry +
        # test_preflight_dead_resolvers_strict_passes_on_real_codebase.
        preflight_dead_resolvers(strict=True, verbose=verbose)
        preflight_profiles(strict=True, verbose=verbose)
        preflight_arch_consistency(strict=True, verbose=verbose)
        preflight_filename_contract(strict=True, verbose=verbose)
        preflight_loader_format_safety(strict=True, verbose=verbose)
        preflight_canonical_checkpoints(strict=True, verbose=verbose)
        preflight_build_renderer_signature(strict=True, verbose=verbose)
        preflight_bootstrap_safety(strict=True, verbose=verbose)

        # 2026-04-27 codex R5-3 Finding #4: wire the 8 meta-bug checks
        # (FORBIDDEN PATTERNS / CLAUDE.md) into preflight_all WARN-ONLY.
        # All 8 currently have non-zero live-codebase counts; flipping any
        # to strict before fixing those would break Lane A's downstream
        # eval. After fixes #5/#6/#7/#8 in the same commit, current live
        # counts (per `python -c '... check_*(strict=False)'`):
        #   check_no_mps_fallback_default:           1
        #   check_shell_set_e_present:               1
        #   check_no_shell_zip_binary:               1
        #   check_no_pipefail_grep_q_trap:           1
        #   check_no_eval_roundtrip_false:           0  (eligible for strict
        #                                                once verified to
        #                                                stay at 0 across
        #                                                a Lane A cycle)
        #   check_no_scorer_load_at_inflate:        13
        #   check_training_scripts_have_auth_eval:   0  (eligible for strict
        #                                                once verified to
        #                                                stay at 0; AST-
        #                                                upgraded in same
        #                                                commit, no FPs)
        #   check_no_disable_eval_roundtrip_flag:   16
        # Promotion plan: each check flips to strict=True INDIVIDUALLY
        # after operator confirms zero live violations on a clean Lane A
        # cycle. Mirrors the pattern used for preflight_dead_resolvers
        # (commit add2def5: warn-only → fixes → strict-flip).
        check_no_mps_fallback_default(strict=False, verbose=verbose)
        check_shell_set_e_present(strict=False, verbose=verbose)
        check_no_shell_zip_binary(strict=False, verbose=verbose)
        check_no_pipefail_grep_q_trap(strict=False, verbose=verbose)
        check_no_eval_roundtrip_false(strict=False, verbose=verbose)
        check_no_scorer_load_at_inflate(strict=False, verbose=verbose)
        check_training_scripts_have_auth_eval(strict=False, verbose=verbose)
        check_no_disable_eval_roundtrip_flag(strict=False, verbose=verbose)

        # 2026-04-27 (this commit): wire the 3 follow-on meta-bug checks
        # (codex R5-3 #2 + #11 + NVDEC probe gap from commits a1128fd9 /
        # eef64293) into preflight_all WARN-ONLY. Live counts on this
        # codebase at wire-in time:
        #   check_no_pack_sparse_delta_approved_outside_promotion_tool: 0
        #     (eligible for strict promotion once verified to stay at 0
        #      across a Lane C cycle)
        #   check_inflate_sh_handles_br_centrally:                     0
        #     (robust_current/inflate.sh has the Stage 0 block per
        #      a1128fd9; exact_current/inflate.sh is a 5-line passthrough
        #      with no PYTHON_INFLATE dispatch — soft-skipped; eligible
        #      for strict promotion)
        #   check_remote_scripts_have_nvdec_probe:                     3
        #     (remote_pose_tto_bootstrap.sh, remote_setup_full.sh,
        #      remote_train_bootstrap.sh — all do GPU work; either add
        #      probe call or NO_NVDEC_NEEDED header)
        check_no_pack_sparse_delta_approved_outside_promotion_tool(strict=False, verbose=verbose)
        check_inflate_sh_handles_br_centrally(strict=False, verbose=verbose)
        check_remote_scripts_have_nvdec_probe(strict=False, verbose=verbose)

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

    # Ad-hoc remote bootstrap scripts in /tmp. The 2026-04-26 SHIRAZ deploy
    # repeatedly wrote /tmp/*.sh files that vanished on instance restart and
    # were never under version control. The canonical entry point is
    # `scripts/remote_train_bootstrap.sh <profile>` (rsynced with the repo).
    # Allow `/tmp/*.log`, `/tmp/foo.bin`, `/tmp/cache/...` etc — only fire on
    # bash/python shell files written to /tmp and then EXECUTED.
    if re.search(r"\b(bash|sh|python3?)\s+/tmp/[A-Za-z_][\w./]*\.(sh|py)\b", text):
        if "scripts/remote_train_bootstrap.sh" not in text:
            violations.append(
                f"{location}: executes a /tmp/*.{{sh,py}} script — ad-hoc "
                f"deploy scripts in /tmp vanish across instance restarts and "
                f"are not version-controlled. Use the canonical "
                f"`scripts/remote_train_bootstrap.sh <profile>` instead, or "
                f"add the path to scripts/ if it's a reusable tool."
            )

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


# ── Dead-resolver / dead-import validation ────────────────────────────────────
#
# Bug class this catches: code that reads a profile-derived value via
# `getattr(args, "X", DEFAULT)` (or `args.X`) but the script never actually
# resolves X into the argparse Namespace — so the silent default fires every
# time and the profile's value is dead. Caught manually three times in the
# 2026-04-27 R5 codex review:
#   - pose_dim: every SHIRAZ/DEN/WILDE/GREEN run silently trained pose_dim=0
#     (FiLM disabled) because parse_args never copied profile.pose_dim into
#     the Namespace. (Lane D incidental fix, commit 0746a803.)
#   - segnet_uncertainty_weighted_loss: imported in train_renderer but never
#     defined in tac.losses. Hidden by stale .pyc caches; would have crashed
#     Lane D at runtime. (Lane D R5, commit 46e2ab6d.)
#   - args.uncertainty_loss_floor: referenced at train_renderer:1614 with no
#     CLI flag and no resolver call. (Lane D R5.)
#
# This validator catches all three at preflight time so they never ship.

class DeadResolverViolation(Exception):
    """A script reads args.X with no flag + no resolver, OR imports a name
    that does not exist in the source module."""


def _flag_to_attr(flag: str) -> str:
    """Convert '--motion-hidden' to 'motion_hidden' (argparse default rule)."""
    return flag.lstrip("-").replace("-", "_")


def _collect_assigned_args_attrs(tree: ast.Module) -> set[str]:
    """Walk the AST for every `args.X = ...` (Assign) and `args.X += ...`
    (AugAssign) site. Returns the set of attribute names assigned anywhere
    in the module — this is the resolver-side ground truth."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "args"):
                    out.add(tgt.attr)
        elif isinstance(node, ast.AugAssign):
            tgt = node.target
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "args"):
                out.add(tgt.attr)
    return out


def _scan_python_for_dead_resolvers(
    path: Path,
    repo_root: Path,
) -> list[str]:
    """Find `getattr(args, 'X', ...)` references where X has neither a
    `--X` argparse flag in the same file nor an `args.X = ...` assignment
    anywhere in the same file.

    Conservative scope by design: only the literal getattr-with-args idiom
    is flagged. Plain `args.X` reads are too noisy (every CLI program reads
    its own args). The getattr form specifically encodes a silent-default
    contract that the bug class exploits.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    sig = _parse_argparse_signature(path) or {}
    flag_attrs = {_flag_to_attr(f) for f in sig.keys()}
    assigned_attrs = _collect_assigned_args_attrs(tree)
    known_attrs = flag_attrs | assigned_attrs

    rel = path.relative_to(repo_root) if path.is_absolute() else path
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        target_node = node.args[0]
        attr_node = node.args[1]
        if not (isinstance(target_node, ast.Name) and target_node.id == "args"):
            continue
        if not (isinstance(attr_node, ast.Constant)
                and isinstance(attr_node.value, str)):
            continue
        attr_name = attr_node.value
        if attr_name.startswith("_"):
            # Private-by-convention; usually internal helpers, skip.
            continue
        if attr_name in known_attrs:
            continue
        violations.append(
            f"{rel}:{node.lineno}: getattr(args, {attr_name!r}, ...) but no "
            f"--{attr_name.replace('_', '-')!r} argparse flag and no "
            f"`args.{attr_name} = ...` assignment found anywhere in the "
            f"file. DEAD RESOLVER: silent default reads will mask any "
            f"profile value the operator thinks they set. "
            f"(pose_dim / uncertainty_loss_floor bug class.)"
        )
    return violations


def _module_top_level_names(mod_path: Path) -> set[str]:
    """Return every name defined or re-exported at module top level.

    Handles: function/class defs, simple assignments, AnnAssign, ImportFrom
    re-exports, and Import. Does NOT execute the module.
    """
    try:
        tree = ast.parse(mod_path.read_text(), filename=str(mod_path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _resolve_tac_module_path(module: str, repo_root: Path) -> Path | None:
    """Resolve `tac.X.Y` to the on-disk file path. Handles package __init__
    and bare modules. Returns None if not found in this repo."""
    if not module.startswith("tac."):
        return None
    rel = module.replace(".", "/")
    candidate = repo_root / "src" / f"{rel}.py"
    if candidate.exists():
        return candidate
    candidate = repo_root / "src" / rel / "__init__.py"
    if candidate.exists():
        return candidate
    return None


def _is_resolvable_submodule(parent_module: str, name: str, repo_root: Path) -> bool:
    """True if `from <parent_module> import <name>` would resolve `name` as
    a submodule of <parent_module>. Handles e.g.
    `from tac.lossless import next_frame_coder` where next_frame_coder is
    a `.py` file inside src/tac/lossless/."""
    if not parent_module.startswith("tac."):
        return False
    parent_rel = parent_module.replace(".", "/")
    candidate = repo_root / "src" / parent_rel / f"{name}.py"
    if candidate.exists():
        return True
    candidate = repo_root / "src" / parent_rel / name / "__init__.py"
    return candidate.exists()


def _import_inside_try_handler(tree: ast.Module, target: ast.ImportFrom) -> bool:
    """True if `target` (an ImportFrom node) is lexically inside a `try:` body
    whose handlers catch ImportError (or bare except). Such imports are
    intentional graceful-fallback patterns and should not be flagged."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # Walk just the try-body (not the handlers / else / finally) for the target.
        for body_node in node.body:
            if any(child is target for child in ast.walk(body_node)):
                # Now check the handlers — at least one must catch ImportError
                # (or be a bare except).
                for handler in node.handlers:
                    if handler.type is None:
                        return True  # bare `except:`
                    # Handle `except ImportError`, `except (ImportError, ...)`,
                    # `except ModuleNotFoundError`, etc.
                    candidates: list[ast.AST] = []
                    if isinstance(handler.type, ast.Tuple):
                        candidates.extend(handler.type.elts)
                    else:
                        candidates.append(handler.type)
                    for c in candidates:
                        name = ast.unparse(c) if hasattr(ast, "unparse") else ""
                        if "ImportError" in name or "ModuleNotFoundError" in name:
                            return True
    return False


def _scan_python_for_dead_imports(path: Path, repo_root: Path) -> list[str]:
    """Find `from tac.X import Y` where Y is not defined at top level in
    tac.X AND Y is not a resolvable submodule. Skips imports inside
    try/except ImportError blocks (intentional graceful fallback).

    Catches the segnet_uncertainty_weighted_loss class — runtime
    NameError masked by stale .pyc caches.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    rel = path.relative_to(repo_root) if path.is_absolute() else path
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module:
            continue
        mod_path = _resolve_tac_module_path(node.module, repo_root)
        if mod_path is None:
            continue
        if _import_inside_try_handler(tree, node):
            continue
        defined = _module_top_level_names(mod_path)
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name in defined:
                continue
            if _is_resolvable_submodule(node.module, alias.name, repo_root):
                continue
            violations.append(
                f"{rel}:{node.lineno}: imports {alias.name!r} from "
                f"{node.module} but that name is NOT defined at the top "
                f"level of {mod_path.relative_to(repo_root)} and is not a "
                f"resolvable submodule. DEAD IMPORT: runtime NameError when "
                f".pyc cache is invalidated. "
                f"(segnet_uncertainty_weighted_loss bug class.)"
            )
    return violations


def preflight_dead_resolvers(
    repo_root: Path | None = None,
    target_dirs: list[str] | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Scan target scripts for dead-resolver and dead-import bug patterns.

    Two rules:
      A. Every `getattr(args, 'X', DEFAULT)` reference must have a corresponding
         `--X` argparse flag OR an explicit `args.X = ...` assignment somewhere
         in the same file. Otherwise the silent default masks profile values.
         (pose_dim / uncertainty_loss_floor bug class.)
      B. Every `from tac.X import Y` must resolve — Y must actually be defined
         at top level in tac.X. Otherwise stale .pyc caches mask a runtime
         NameError. (segnet_uncertainty_weighted_loss bug class.)

    Returns list of human-readable violations. Raises DeadResolverViolation
    if strict and any are found.
    """
    root = repo_root or REPO_ROOT
    target_dirs = target_dirs or TARGET_DIRS

    violations: list[str] = []
    n_scanned = 0

    for d in target_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for py in sorted(d_path.glob("*.py")):
            n_scanned += 1
            violations.extend(_scan_python_for_dead_resolvers(py, root))
            violations.extend(_scan_python_for_dead_imports(py, root))

    if verbose and violations:
        print(f"  [dead-resolvers] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [dead-resolvers] OK: {n_scanned} files scanned")

    if violations and strict:
        raise DeadResolverViolation(
            "DEAD-RESOLVER / DEAD-IMPORT violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nFix every violation. Each one is a real bug class that has "
            "burned GPU money in this repo (pose_dim, "
            "segnet_uncertainty_weighted_loss, uncertainty_loss_floor — "
            "2026-04-27 R5 codex review)."
        )
    return violations


# ── Meta-bug pattern checks ───────────────────────────────────────────────────
#
# Each check below catches a CLAUDE.md "FORBIDDEN PATTERNS" bug class that has
# bitten this project at least once and cost real GPU money. These are
# additive, defensive scaffolds: each starts in warn-only mode (strict=False)
# until it surfaces zero true-positive violations on the live codebase, then
# is flipped strict=True in preflight_all() (see TODO at end of section).
#
# Pattern → memory entry mapping:
#   1. MPS-fallback device default       → feedback_default_to_convenience_trap
#   2. set -uo pipefail (no -e)          → feedback_zip_dep_bootstrap_trap
#   3. shell `zip` binary                → feedback_zip_dep_bootstrap_trap
#   4. pipefail + grep -q SIGPIPE        → feedback_pipefail_grep_q_trap
#   5. eval_roundtrip=False              → CLAUDE.md "eval_roundtrip" rule
#   6. scorer load at inflate            → feedback_strict_scorer_rule
#   7. training script no auth eval      → CLAUDE.md "Auth eval EVERYWHERE"
#   8. --no-eval-roundtrip CLI flag      → Lane C R5 fix (commit 9d71ec5d)


class MetaBugViolation(Exception):
    """A meta-bug pattern (CLAUDE.md FORBIDDEN PATTERNS) detected."""


# Directories scanned for Python meta-bug patterns. Mirrors TARGET_DIRS but
# adds scripts/ for shell-adjacent Python launchers.
_META_PY_SCAN_DIRS = ["src/tac", "experiments", "scripts"]
# Directories scanned for shell meta-bug patterns.
_META_SH_SCAN_DIRS = ["scripts", "submissions/robust_current"]


def _iter_python_files(root: Path, dirs: list[str]) -> list[Path]:
    """Collect every .py file under `dirs` (recursively). Skips __pycache__."""
    out: list[Path] = []
    for d in dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def _iter_shell_files(root: Path, dirs: list[str]) -> list[Path]:
    """Collect every .sh file under `dirs` (recursively)."""
    out: list[Path] = []
    for d in dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.sh"):
            out.append(p)
    return sorted(out)


# Heredoc start: `<< [-]['"]?TOKEN['"]?` after a redirect-or-no-redirect operator.
# We deliberately accept all heredoc forms: <<TOKEN, <<-TOKEN, <<"TOKEN",
# <<'TOKEN', << TOKEN, <<- TOKEN, etc. The matched group 'token' is the bare
# delimiter (quotes/dashes stripped). Tab/space separation between << and the
# token is allowed by bash and by us.
_HEREDOC_START_RE = re.compile(
    r"""
    <<-?\s*                  # << or <<- with optional whitespace
    (?P<quote>['"])?         # optional opening quote
    (?P<token>[A-Za-z_][A-Za-z0-9_]*)  # delimiter token
    (?(quote)(?P=quote))     # closing quote (must match opener)
    """,
    re.VERBOSE,
)


def _mask_shell_heredocs(text: str) -> str:
    """Replace bash heredoc bodies with empty lines.

    Preserves total line count so reported lineno still matches the source.
    Without this, shell scanners would treat heredoc bodies (which can
    legitimately contain `set -uo pipefail`, `zip foo.zip bar`, `| grep -q`
    as Python/docs/embedded snippets) as executable shell.

    Behavior:
      • Detects every `<<TOKEN`, `<<-TOKEN`, `<< TOKEN`, `<<"TOKEN"`,
        `<<'TOKEN'` heredoc start on a non-comment line.
      • Skips ahead until the next line whose stripped content equals TOKEN
        (or, for `<<-TOKEN`, leading tabs are also stripped per bash).
      • Replaces lines BETWEEN the start (exclusive) and the terminator
        (exclusive) with empty strings. The start line itself is preserved
        (so a violation written on the start line — unusual — is still
        visible) and the terminator line is preserved.
      • If multiple heredocs start on the same line (rare: `cmd <<A <<B`),
        we mask both bodies in order, requiring A then B as terminators.
      • A heredoc with no terminator (eof) means everything from start+1
        to EOF is masked. This matches bash's runtime behavior of erroring,
        but for static analysis "treat as quoted" is the safer call than
        "treat as code".
    """
    lines = text.split("\n")
    out_lines = list(lines)  # mutable copy
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip comment-only lines for heredoc-start detection.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            i += 1
            continue
        # Find ALL heredoc starts on this line (e.g. `cmd <<A <<B`).
        # Order matters: bash reads bodies in left-to-right order.
        matches = list(_HEREDOC_START_RE.finditer(line))
        if not matches:
            i += 1
            continue
        # Tokens to consume in order. Track whether each was `<<-` form
        # (which strips leading TABS — not spaces — from the terminator
        # comparison per POSIX).
        pending: list[tuple[str, bool]] = []
        for m in matches:
            token = m.group("token")
            stripped_form = line[max(0, m.start() - 1):m.start() + 3].lstrip("<")
            # Simpler & robust: look at the raw match text from `<<` onward.
            raw = line[m.start():m.end()]
            is_dash = raw.startswith("<<-")
            pending.append((token, is_dash))
        j = i + 1
        while pending and j < len(lines):
            cand = lines[j]
            token, is_dash = pending[0]
            cmp = cand.lstrip("\t") if is_dash else cand
            if cmp == token:
                # Terminator hit — pop it; stop masking this body.
                pending.pop(0)
                j += 1
                continue
            # Mask this body line (preserve line count by emitting empty).
            out_lines[j] = ""
            j += 1
        # If pending non-empty here, we hit EOF without terminator.
        # Lines i+1..end are already masked above. Continue past current.
        i = j
    return "\n".join(out_lines)


# ── Check 1: MPS-fallback device default ──────────────────────────────────────


def _scan_python_for_mps_fallback(path: Path, repo_root: Path) -> list[str]:
    """Detect `... else "mps" ...` ternaries triggered when CUDA is missing.

    Two layers:
      A. AST: an IfExp where the test calls `.cuda.is_available()` and the
         orelse contains the literal `"mps"` (either directly or as a nested
         IfExp orelse leaf).
      B. Text: a regex backup catches one-liners that span multiple ternaries,
         covering the common `"cuda" if ... else "mps" if ... else "cpu"`.

    Tests / smoke files are skipped — they may legitimately probe MPS.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name or "/smoke" in rel_s.lower():
        return []
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    violations: list[str] = []

    def _orelse_mentions_mps(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and sub.value == "mps":
                return True
        return False

    def _test_checks_cuda(node: ast.AST) -> bool:
        s = ast.unparse(node) if hasattr(ast, "unparse") else ""
        return "cuda.is_available" in s or "torch.cuda.is_available" in s

    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            if _test_checks_cuda(node.test) and _orelse_mentions_mps(node.orelse):
                violations.append(
                    f"{rel}:{node.lineno}: ternary `cuda.is_available() ... "
                    f"else \"mps\" ...` — MPS-fallback device default. "
                    f"FORBIDDEN per CLAUDE.md (feedback_default_to_convenience_trap). "
                    f"Default to CUDA-required; raise on no-CUDA; provide "
                    f"explicit `--device cpu` opt-in."
                )

    # codex R5-3 #7: BoolOp (and/or) device-selection chains. Pattern:
    #   torch.cuda.is_available() and 'cuda' or torch.backends.mps.is_available() and 'mps' or 'cpu'
    # Has no IfExp anywhere — must AST-walk BoolOp explicitly. Rule (refined
    # to avoid the FP class `... or str(self.device) == "mps"` where "mps"
    # is INSIDE a Compare and never selected as a value):
    #   1. Walk top-level BoolOps (not nested inside Compare / Subscript /
    #      Call / etc. — only BoolOps that COULD evaluate to the string
    #      "mps" as a result).
    #   2. The BoolOp tree must contain a `cuda.is_available()` call.
    #   3. A string constant "mps" must appear as a DIRECT leaf operand of
    #      a BoolOp value subtree (i.e., reachable by following only
    #      BoolOp.values links, not by descending into Compare/Call/etc.).
    #      That's exactly the position where a fallback chain would put it.
    def _bool_value_leaves(node: ast.BoolOp):
        """Yield every node that can be the BoolOp's RESULT value."""
        for v in node.values:
            if isinstance(v, ast.BoolOp):
                yield from _bool_value_leaves(v)
            else:
                yield v

    def _tree_has_cuda_check(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                s = ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
                if "cuda.is_available" in s:
                    return True
        return False

    seen_boolop_lines: set[int] = set()
    # Find OUTERMOST BoolOps only (so a nested BoolOp inside another BoolOp
    # doesn't double-count). Easiest: collect parent links once.
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp):
            continue
        # Skip nested BoolOps — only flag the outermost.
        p = parents.get(id(node))
        if isinstance(p, ast.BoolOp):
            continue
        if node.lineno in seen_boolop_lines:
            continue
        if not _tree_has_cuda_check(node):
            continue
        # Check leaves: is "mps" a result-position string constant?
        # Note: an `IfExp` inside a BoolOp value is treated as a value
        # (we recurse the IfExp body+orelse for "mps").
        is_fallback = False
        for leaf in _bool_value_leaves(node):
            # leaf can itself be an IfExp / Constant / Call / etc.
            if isinstance(leaf, ast.Constant) and leaf.value == "mps":
                is_fallback = True
                break
            if isinstance(leaf, ast.IfExp):
                for sub in ast.walk(leaf):
                    if isinstance(sub, ast.Constant) and sub.value == "mps":
                        is_fallback = True
                        break
                if is_fallback:
                    break
        if is_fallback:
            violations.append(
                f"{rel}:{node.lineno}: BoolOp chain `... cuda.is_available() "
                f"... 'mps' ...` — MPS-fallback device default. FORBIDDEN per "
                f"CLAUDE.md (feedback_default_to_convenience_trap). Default "
                f"to CUDA-required; raise on no-CUDA; provide explicit "
                f"`--device cpu` opt-in."
            )
            seen_boolop_lines.add(node.lineno)

    # Text backup: one-line chains that AST already caught are deduped by lineno.
    seen_lines = {int(v.split(":")[1]) for v in violations}
    pat = re.compile(r'"cuda".*cuda\.is_available\(\).*else\s*"mps"')
    for i, line in enumerate(text.splitlines(), start=1):
        if i in seen_lines:
            continue
        if pat.search(line):
            violations.append(
                f"{rel}:{i}: chained ternary with `\"cuda\" if "
                f"cuda.is_available() else \"mps\"` — MPS-fallback device "
                f"default. FORBIDDEN per CLAUDE.md "
                f"(feedback_default_to_convenience_trap)."
            )
    return violations


def check_no_mps_fallback_default(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch the MPS-fallback device default bug class.

    Reference: feedback_default_to_convenience_trap (CLAUDE.md FORBIDDEN
    PATTERNS). Defaulting to "mps" when CUDA is unavailable produces silent
    drift (23x PoseNet error verified 2026-04-25). Default must be
    CUDA-required; opt-in to CPU/MPS only via explicit flag with banner.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for py in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_mps_fallback(py, root))

    if verbose and violations:
        print(f"  [no-mps-fallback] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-mps-fallback] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "MPS-FALLBACK DEFAULT violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nMPS auth eval is NOISE — see CLAUDE.md "
            "feedback_default_to_convenience_trap. Default to CUDA-required."
        )
    return violations


# ── Check 2: shell `set -uo pipefail` without `set -e` ────────────────────────


def _scan_shell_for_missing_set_e(path: Path, repo_root: Path) -> list[str]:
    """Find `set -` lines that include `u` or `o pipefail` but NOT `e`."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    # codex R5-3 #5: mask heredoc bodies so embedded Python/docs don't
    # register as executable shell.
    text = _mask_shell_heredocs(text)
    violations: list[str] = []
    # We accept any `set` line as long as somewhere in the file `set -e`
    # (or set -euo / -ex etc.) is present. Track presence first.
    has_e_anywhere = False
    set_lines: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("set "):
            continue
        # Drop comments after `#`
        no_comment = stripped.split("#", 1)[0].strip()
        # Detect short flags like `-e`, `-eu`, `-euo`
        m = re.match(r"set\s+-([a-zA-Z]+)", no_comment)
        if m and "e" in m.group(1):
            has_e_anywhere = True
        # Or `set -o errexit`
        if "errexit" in no_comment:
            has_e_anywhere = True
        set_lines.append((i, no_comment))

    if has_e_anywhere:
        return []

    for lineno, line in set_lines:
        # Only flag lines that USE u or pipefail (the dangerous combo).
        m = re.match(r"set\s+-([a-zA-Z]+)", line)
        flags = m.group(1) if m else ""
        uses_u = "u" in flags
        uses_pipefail = "o" in flags and "pipefail" in line
        if uses_u or uses_pipefail:
            violations.append(
                f"{rel}:{lineno}: `{line}` uses `u`/`pipefail` without `e`. "
                f"Silent failure cascade: a failing command does not abort "
                f"the script — empty captures pass to argparse and crash "
                f"30 minutes later. Use `set -euo pipefail`. "
                f"(feedback_zip_dep_bootstrap_trap.)"
            )
    return violations


def check_shell_set_e_present(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch `set -uo pipefail` without `set -e` shell footgun.

    Reference: feedback_zip_dep_bootstrap_trap. Without `-e`, a failing
    `zip` or `python` command does not abort the script. Empty captured
    variables flow downstream and crash auth_eval 30 minutes later.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for sh in _iter_shell_files(root, _META_SH_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_shell_for_missing_set_e(sh, root))

    if verbose and violations:
        print(f"  [set-e-required] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [set-e-required] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "SHELL `set -e` MISSING violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nUse `set -euo pipefail` (feedback_zip_dep_bootstrap_trap)."
        )
    return violations


# ── Check 3: shell `zip` binary use ───────────────────────────────────────────


# Match `zip` (whitespace-bounded) but NOT `zipfile`, `unzip`, `gunzip`,
# `gzip`, `bzip2`, `gunzip2`, etc. We require `zip` to appear after a
# command boundary (start of line, `;`, `&&`, `||`, `|`, `(`, or `$(`)
# OPTIONALLY preceded by env vars / sudo, and followed by whitespace.
_ZIP_BIN_RE = re.compile(
    r'(?:^|[;&|()`]|\$\()\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:sudo\s+)?zip(?=[\s\\])'
)


def _scan_shell_for_zip_binary(path: Path, repo_root: Path) -> list[str]:
    """Find use of the shell `zip` binary (which is missing on PyTorch
    container images). Allow `python -c '...zipfile...'` and `unzip`."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    # codex R5-3 #5: mask heredoc bodies so embedded Python (which often
    # imports zipfile) doesn't register as a shell `zip` invocation.
    text = _mask_shell_heredocs(text)
    violations: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip lines that are clearly invoking python with zipfile
        if "zipfile" in stripped:
            continue
        if _ZIP_BIN_RE.search(stripped):
            violations.append(
                f"{rel}:{i}: shell `zip` binary not present on PyTorch "
                f"container images. Use `python -c \"import zipfile; ...\"` "
                f"instead. (feedback_zip_dep_bootstrap_trap.)"
            )
    return violations


def check_no_shell_zip_binary(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch shell `zip` binary use (missing on PyTorch container images).

    Reference: feedback_zip_dep_bootstrap_trap. The PyTorch base container
    has no `zip` (but `unzip` is separate and OK). Use `python zipfile`.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for sh in _iter_shell_files(root, _META_SH_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_shell_for_zip_binary(sh, root))

    if verbose and violations:
        print(f"  [no-shell-zip] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-shell-zip] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "SHELL `zip` BINARY violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nUse `python -c \"import zipfile\"` "
            "(feedback_zip_dep_bootstrap_trap)."
        )
    return violations


# ── Check 4: pipefail + `grep -q` SIGPIPE trap ────────────────────────────────


# codex R5-3 #6: LHS commands that are SAFE upstream of `| grep -q`.
# echo/printf are shell builtins that do not SIGPIPE meaningfully — they
# write a fixed-size buffer once and exit. The capture-first remediation
# (`OUT=$(cmd 2>&1); echo "$OUT" | grep -q ...`) MUST be allowed; otherwise
# the scanner flags its own prescribed fix.
_SAFE_GREP_Q_UPSTREAM_CMDS = ("echo", "printf")


def _grep_q_lhs_is_safe(lhs: str) -> bool:
    """Return True iff the pipe LHS is a safe builtin (echo/printf).

    Strips leading whitespace and any inline `!`/negation chains (e.g.
    `if ! echo "$X"`), then checks if the first token is a safe cmd.
    """
    s = lhs.strip()
    # Drop common shell preludes that don't change the upstream cmd:
    #   `if ! `, `! `, `&& `, `|| `, `; `, `then `, `do `, `{ `
    # Accept any chain of these prefixes once, then look at the next token.
    prefix_re = re.compile(
        r"^(?:if\s+)?(?:then\s+)?(?:do\s+)?(?:while\s+)?"
        r"(?:!\s+)?(?:\{?\s*)"
    )
    s = prefix_re.sub("", s).lstrip()
    # Bare token check: first whitespace-separated token.
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\b", s)
    if not m:
        return False
    return m.group(1) in _SAFE_GREP_Q_UPSTREAM_CMDS


def _scan_shell_for_pipefail_grep_q(path: Path, repo_root: Path) -> list[str]:
    """Find `... | grep -q PATTERN` lines under `set -e`/`pipefail`.

    grep -q closes stdin after first match; the upstream cmd then SIGPIPEs;
    pipefail propagates that as failure; `set -e` aborts the script.
    Remediation: capture-first idiom (`OUT=$(cmd); echo "$OUT" | grep -q ...`).

    codex R5-3 #6 exemptions:
      • `echo "$VAR" | grep -q PAT` — echo is a builtin, no meaningful SIGPIPE.
      • `printf "..." | grep -q PAT` — same.
      • `grep -q PAT <<< "$VAR"` — here-string, no pipe at all.
      These forms are the prescribed fix for the bug class — flagging them
      would block the remediation.
    codex R5-3 #5: heredoc bodies are masked before scanning.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    text = _mask_shell_heredocs(text)
    # Only fire if file has set -e or pipefail somewhere.
    has_pipefail_or_e = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("set "):
            continue
        no_comment = stripped.split("#", 1)[0].strip()
        m = re.match(r"set\s+-([a-zA-Z]+)", no_comment)
        if m and "e" in m.group(1):
            has_pipefail_or_e = True
        if "pipefail" in no_comment or "errexit" in no_comment:
            has_pipefail_or_e = True
    if not has_pipefail_or_e:
        return []

    grep_q_re = re.compile(r"\|\s*grep\s+-[a-zA-Z]*q")
    here_string_re = re.compile(r"grep\s+-[a-zA-Z]*q\b[^|]*<<<")
    violations: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        # Skip comments
        if line.lstrip().startswith("#"):
            continue
        # Here-string form `grep -q PAT <<< "$VAR"` is exempt.
        if here_string_re.search(line):
            continue
        m = grep_q_re.search(line)
        if not m:
            continue
        # Inspect the LHS (everything before this `|`). If it ends with
        # `echo ...` or `printf ...`, exempt. Use rfind to handle chains
        # like `cmd1 | cmd2 | grep -q ...`: only the IMMEDIATE upstream
        # matters for SIGPIPE on grep -q.
        # m.start() points at the `|` of the `| grep -q` pattern; the
        # upstream cmd is whatever follows the previous pipe (or line start).
        upstream_end = m.start()
        prev_pipe = line.rfind("|", 0, upstream_end)
        upstream_start = prev_pipe + 1 if prev_pipe >= 0 else 0
        lhs = line[upstream_start:upstream_end]
        if _grep_q_lhs_is_safe(lhs):
            continue
        violations.append(
            f"{rel}:{i}: `| grep -q` under `set -e`/`pipefail` triggers "
            f"SIGPIPE on the upstream command — pipeline aborts the "
            f"script. Capture-first idiom: "
            f"`OUT=$(cmd 2>&1); echo \"$OUT\" | grep -q ...`. "
            f"(feedback_pipefail_grep_q_trap.)"
        )
    return violations


def check_no_pipefail_grep_q_trap(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch `pipefail + grep -q` SIGPIPE trap.

    Reference: feedback_pipefail_grep_q_trap. `cmd | grep -q PAT` under
    `set -euo pipefail` SIGPIPEs the upstream when grep stops reading
    after first match. Whole pipeline reports failure → script aborts.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for sh in _iter_shell_files(root, _META_SH_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_shell_for_pipefail_grep_q(sh, root))

    if verbose and violations:
        print(f"  [no-pipefail-grep-q] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-pipefail-grep-q] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "PIPEFAIL + GREP -Q violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nUse capture-first idiom (feedback_pipefail_grep_q_trap)."
        )
    return violations


# ── Check 5: eval_roundtrip=False anywhere ────────────────────────────────────


def _scan_python_for_eval_roundtrip_false(path: Path, repo_root: Path) -> list[str]:
    """Detect:
      A. `eval_roundtrip=False` keyword in any call.
      B. `def foo(..., eval_roundtrip: bool = False, ...)` default.
      C. `def foo(..., eval_roundtrip = False, ...)` default (untyped).
    Test/smoke files are exempt.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    violations: list[str] = []

    # A. Keyword-arg call sites: foo(..., eval_roundtrip=False, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "eval_roundtrip" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is False:
                    violations.append(
                        f"{rel}:{node.lineno}: call passes "
                        f"`eval_roundtrip=False`. NON-NEGOTIABLE per CLAUDE.md: "
                        f"every training path must use eval_roundtrip. Only "
                        f"escape hatch is env var TAC_ALLOW_NO_ROUNDTRIP=1."
                    )

    # B/C. Function defaults
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        # Combine positional and keyword-only with their defaults.
        all_args = list(args.args) + list(args.kwonlyargs)
        # positional defaults align to the TAIL of args.args
        pos_defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
        kw_defaults = list(args.kw_defaults)
        all_defaults = pos_defaults + kw_defaults
        for a, d in zip(all_args, all_defaults):
            if a.arg != "eval_roundtrip" or d is None:
                continue
            if isinstance(d, ast.Constant) and d.value is False:
                violations.append(
                    f"{rel}:{node.lineno}: function `{node.name}` defaults "
                    f"`eval_roundtrip=False`. NON-NEGOTIABLE per CLAUDE.md: "
                    f"default must be True; only escape hatch is env var "
                    f"TAC_ALLOW_NO_ROUNDTRIP=1."
                )
    return violations


def check_no_eval_roundtrip_false(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch eval_roundtrip=False anywhere (call site or function default).

    Reference: CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE". Without
    eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run
    without it is a wasted run.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for py in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_eval_roundtrip_false(py, root))

    if verbose and violations:
        print(f"  [no-eval-roundtrip-false] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-eval-roundtrip-false] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "eval_roundtrip=False violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\neval_roundtrip is non-negotiable (CLAUDE.md)."
        )
    return violations


# ── Check 6: scorer load at inflate time ──────────────────────────────────────


# Patterns indicating a scorer is being loaded at inflate time. Per
# feedback_strict_scorer_rule, NO scorers may be loaded at inflate.
_SCORER_LOAD_NAMES = (
    "load_scorers", "load_posenet", "load_segnet",
    "load_differentiable_scorers",
    "PoseNet(", "SegNet(",  # direct instantiation
)
_SCORER_NAME_LITERALS_RE = re.compile(r"\b(posenet|segnet)\b", re.IGNORECASE)


def _scan_inflate_for_scorer_load(path: Path, repo_root: Path) -> list[str]:
    """Detect scorer-load patterns in inflate*.py files."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        # .sh files won't parse — fall back to text scan.
        text = ""
        try:
            text = path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            return []
        tree = None

    violations: list[str] = []

    if tree is not None:
        # AST: detect imports / calls referencing scorer loaders.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "tac.scorer" in node.module or node.module == "tac.scorer":
                    for alias in node.names:
                        violations.append(
                            f"{rel}:{node.lineno}: imports {alias.name!r} from "
                            f"{node.module} at inflate time. Strict scorer rule "
                            f"(CLAUDE.md feedback_strict_scorer_rule): NO scorer "
                            f"load at inflate; ~73MB destroys the rate term."
                        )
            if isinstance(node, ast.Call):
                func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
                for name in ("load_scorers", "load_posenet", "load_segnet",
                             "load_differentiable_scorers"):
                    if func_str.endswith(name):
                        violations.append(
                            f"{rel}:{node.lineno}: calls `{func_str}(...)` at "
                            f"inflate time. Strict scorer rule (CLAUDE.md "
                            f"feedback_strict_scorer_rule): NO scorer load at "
                            f"inflate."
                        )
                        break
    else:
        # Shell file fallback: any line mentioning posenet.bin / segnet.bin
        # / safetensors.load near scorer keywords.
        for i, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if "scorer" in low and ("load" in low or "import" in low):
                violations.append(
                    f"{rel}:{i}: shell line references scorer load at "
                    f"inflate time (CLAUDE.md feedback_strict_scorer_rule)."
                )
    return violations


def check_no_scorer_load_at_inflate(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch any scorer load at inflate time.

    Reference: feedback_strict_scorer_rule (CLAUDE.md "Strict scorer rule").
    NO PoseNet/SegNet load at inflate — those weights would have to live
    in archive.zip per Yousfi PR #35, destroying the rate term.

    Scans `submissions/*/inflate*.py` and `submissions/*/inflate.sh`.
    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    submissions = root / "submissions"
    if submissions.exists():
        for sub_dir in sorted(submissions.iterdir()):
            if not sub_dir.is_dir():
                continue
            for p in sorted(sub_dir.glob("inflate*.py")):
                n_scanned += 1
                violations.extend(_scan_inflate_for_scorer_load(p, root))
            for p in sorted(sub_dir.glob("inflate*.sh")):
                n_scanned += 1
                violations.extend(_scan_inflate_for_scorer_load(p, root))

    if verbose and violations:
        print(f"  [no-scorer-at-inflate] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-scorer-at-inflate] OK: {n_scanned} inflate files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "SCORER LOAD AT INFLATE violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nNo scorer at inflate time (feedback_strict_scorer_rule)."
        )
    return violations


# ── Check 7: training scripts MUST end with auth eval ─────────────────────────


# codex R5-3 #8: tokens that mark a path string (literal) as referring to a
# RENDERER artifact (the only thing that requires auth-eval — LoRA adapters,
# postfilters, statistics tensors, etc. do not). Match is case-insensitive
# on the path basename. We deliberately exclude generic "best"/"state_dict"
# from this list — those are too broad and produced FPs (lora_best.pt
# was misclassified as a renderer in the regex era).
_RENDERER_PATH_TOKENS = ("renderer", "checkpoint", "fp4")
# Generic "model" matches lora_*.pt / postfilter_*.pt FALSELY less often
# than expected, but we keep "model" because a path like `model_best.pt`
# IS likely a renderer. To minimize FPs the renderer-detector ALSO requires
# the dict being saved to look like a model state (handled below).
_RENDERER_PATH_TOKENS_GENERIC = ("model",)


def _path_string_looks_like_renderer(s: str) -> bool:
    """Return True if a string literal references a renderer artifact path.

    Strict tokens (renderer/checkpoint/fp4) are sufficient on their own.
    The 'model' token is generic and only counts if combined with a
    typical artifact extension (.pt/.pth/.bin).
    """
    s_low = s.lower()
    if any(tok in s_low for tok in _RENDERER_PATH_TOKENS):
        return True
    if any(tok in s_low for tok in _RENDERER_PATH_TOKENS_GENERIC):
        if s_low.endswith((".pt", ".pth", ".bin")):
            return True
    return False


def _node_references_renderer_path(node: ast.AST) -> bool:
    """True if this AST subtree contains any string constant that matches
    `_path_string_looks_like_renderer`. Handles f-strings, joinedstr,
    Path() chains, and `output_dir / "renderer.bin"` BinOp expressions."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _path_string_looks_like_renderer(sub.value):
                return True
        # f-strings: ast.JoinedStr containing FormattedValue + Constant parts.
        if isinstance(sub, ast.JoinedStr):
            for part in sub.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    if _path_string_looks_like_renderer(part.value):
                        return True
    return False


def _call_is_auth_eval_subprocess(node: ast.Call) -> bool:
    """True if `node` is `subprocess.run([..., "auth_eval_renderer.py", ...])`
    or `subprocess.Popen(...)` / `subprocess.check_call(...)` with the same
    auth-eval script as a list element."""
    func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
    if not (func_str.endswith("subprocess.run") or func_str.endswith("subprocess.Popen")
            or func_str.endswith("subprocess.check_call") or func_str.endswith("subprocess.check_output")
            or func_str == "run" or func_str == "Popen"
            or func_str == "check_call" or func_str == "check_output"):
        return False
    for arg in node.args:
        if isinstance(arg, ast.List):
            for elt in arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if "auth_eval_renderer" in elt.value:
                        return True
        # Single-string form: subprocess.run("python auth_eval_renderer.py ...", shell=True)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if "auth_eval_renderer" in arg.value:
                return True
    return False


def _call_is_auth_eval_helper(node: ast.Call) -> bool:
    """True if `node` calls `auth_eval_renderer.main(...)`, `run_auth_eval(...)`,
    `auth_eval(...)`, or any function whose unparse ends with these names."""
    func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
    targets = ("auth_eval_renderer.main", "run_auth_eval", "auth_eval",
               "auth_eval_renderer", "auth_eval_on_best")
    for t in targets:
        if func_str == t or func_str.endswith("." + t):
            return True
    return False


def _argparse_defines_no_auth_eval_optout(tree: ast.Module) -> bool:
    """True if the script defines `--no-auth-eval-on-best` argparse flag
    (operator's explicit opt-out — satisfies the rule)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not func_str.endswith(".add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value == "--no-auth-eval-on-best":
                    return True
    return False


def _script_imports_auth_eval(tree: ast.Module) -> bool:
    """True if the script `import`s the auth_eval module (any form). An
    import without a CALL is dead code — the rule requires an actual
    invocation, but we keep this distinction so the violation message can
    say 'imported but never called' for clarity."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "auth_eval" in node.module:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "auth_eval" in alias.name:
                    return True
    return False


def _scan_training_script_for_auth_eval(path: Path, repo_root: Path) -> list[str]:
    """Flag a training script that saves a renderer checkpoint but does not
    invoke auth_eval after it.

    codex R5-3 #8: AST-based replacement of the regex token-grep. Old form
    counted any token-anywhere (comment, help string, dead import) as
    satisfying. New rule:
      • Find every torch.save() call. If args reference a path matching
        `_path_string_looks_like_renderer`, mark script as "saves a renderer".
      • Find every subprocess.run([..., "auth_eval_renderer.py", ...])
        OR direct call to auth_eval_renderer.main()/run_auth_eval()/etc.
      • If --no-auth-eval-on-best is defined, satisfied (operator opt-out).
      • A script that imports auth_eval but never calls it → violation
        (dead-import-class).
      • A script that saves a non-renderer (lora_best.pt, postfilter.pt,
        masks.pt, posenet_targets.bin, stats.pt) → no violation.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []

    saves_renderer = False
    has_auth_eval_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            # Detect `torch.save(...)` (the canonical form). Aliased forms
            # like `from torch import save; save(...)` are intentionally NOT
            # supported — the codebase uses `torch.save(...)` exclusively
            # and the broader pattern would produce FPs (e.g. `model.save()`).
            if func_str == "torch.save":
                # Inspect args for renderer-like path string.
                for arg in node.args:
                    if _node_references_renderer_path(arg):
                        saves_renderer = True
                        break
            if _call_is_auth_eval_subprocess(node) or _call_is_auth_eval_helper(node):
                has_auth_eval_call = True

    if not saves_renderer:
        return []
    if has_auth_eval_call:
        return []
    if _argparse_defines_no_auth_eval_optout(tree):
        return []
    # Dead-import refinement: distinguish "imports but never calls" from
    # "no reference at all" so the operator's fix is obvious.
    if _script_imports_auth_eval(tree):
        return [
            f"{rel}: training script saves a renderer checkpoint and "
            f"IMPORTS auth_eval but never CALLS it (dead import). Per "
            f"CLAUDE.md \"Auth eval EVERYWHERE\": every chained experiment "
            f"MUST end with a CUDA auth eval. Add an explicit "
            f"`subprocess.run([..., 'auth_eval_renderer.py', ...])` or "
            f"`run_auth_eval(...)` after the best save."
        ]
    return [
        f"{rel}: training script saves a renderer checkpoint but never "
        f"invokes auth_eval (no `subprocess.run([..., 'auth_eval_renderer.py', "
        f"...])`, no `run_auth_eval(...)`, no `auth_eval_renderer.main()`, "
        f"and no `--no-auth-eval-on-best` opt-out flag). Per CLAUDE.md "
        f"\"Auth eval EVERYWHERE\": every chained experiment MUST end with "
        f"a CUDA auth eval against its best checkpoint. Tracking only proxy "
        f"is a wasted run (proxy-auth gap can be 100-350x even on CUDA-CUDA)."
    ]


def check_training_scripts_have_auth_eval(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch training scripts that save a model but never auth-eval it.

    Reference: CLAUDE.md "Auth eval EVERYWHERE — NON-NEGOTIABLE". Scans
    `experiments/train_*.py` and `src/tac/experiments/train_*.py`.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    candidates: list[Path] = []
    for d in ("experiments", "src/tac/experiments"):
        d_path = root / d
        if not d_path.exists():
            continue
        for p in sorted(d_path.glob("train_*.py")):
            candidates.append(p)
    for p in candidates:
        n_scanned += 1
        violations.extend(_scan_training_script_for_auth_eval(p, root))

    if verbose and violations:
        print(f"  [training-needs-auth-eval] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [training-needs-auth-eval] OK: {n_scanned} training scripts scanned")

    if violations and strict:
        raise MetaBugViolation(
            "TRAINING SCRIPT MISSING AUTH EVAL violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nAuth eval EVERYWHERE (CLAUDE.md non-negotiable)."
        )
    return violations


# ── Check 8: --no-eval-roundtrip CLI flag definition ──────────────────────────


def _scan_python_for_disable_eval_roundtrip_flag(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect `add_argument("--no-eval-roundtrip"...)` literals."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not func_str.endswith(".add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value == "--no-eval-roundtrip":
                    violations.append(
                        f"{rel}:{node.lineno}: defines `--no-eval-roundtrip` "
                        f"argparse flag. FORBIDDEN per CLAUDE.md: eval_roundtrip "
                        f"is non-negotiable; the only escape hatch is env var "
                        f"TAC_ALLOW_NO_ROUNDTRIP=1. Remove the flag."
                    )
    return violations


def check_no_disable_eval_roundtrip_flag(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch `--no-eval-roundtrip` argparse definitions.

    Reference: Lane C R5 fix (commit 9d71ec5d removed --no-eval-roundtrip
    from optimize_uniward_delta.py). The only escape hatch is env var.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for py in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_disable_eval_roundtrip_flag(py, root))

    if verbose and violations:
        print(f"  [no-disable-eval-roundtrip-flag] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-disable-eval-roundtrip-flag] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "--no-eval-roundtrip CLI FLAG violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nRemove the flag (CLAUDE.md non-negotiable)."
        )
    return violations


# ── Check 9: pack_sparse_delta(compliance_status='approved') outside promo ───
#
# Reference: codex R5-3 finding #2. The library function
# `tac.uniward_delta.pack_sparse_delta` accepts `compliance_status='approved'`
# only when paired with the constant-time-HMAC `_internal_promotion_token`.
# The runtime check exists, but a static scan catches the same bug class
# earlier (preflight time vs runtime). Any caller passing the literal
# 'approved' (or the COMPLIANCE_APPROVED constant) outside the canonical
# promotion tool / test-fixture surface is a violation: the operator-controlled
# attestation flow goes exclusively through `tools/promote_lane_c_to_approved.py`,
# which patches the wire header in-place rather than re-packing.

# The single permitted non-test caller. Anything else passing the approved
# literal/constant to pack_sparse_delta is a violation.
_PACK_SPARSE_DELTA_APPROVED_PROMO_FILE = "tools/promote_lane_c_to_approved.py"
# Token names that mean "approved" to pack_sparse_delta. We accept both the
# string literal "approved" and references to the constant COMPLIANCE_APPROVED
# (which equals "approved"). Both must be considered violations outside the
# canonical promotion tool.
_APPROVED_LITERAL = "approved"
_APPROVED_CONST_NAMES = {"COMPLIANCE_APPROVED"}


def _resolve_pack_sparse_delta_aliases(tree: ast.AST) -> set[str]:
    """Collect every name `pack_sparse_delta` is bound to in this module.

    Handles:
      - `from tac.uniward_delta import pack_sparse_delta` → {"pack_sparse_delta"}
      - `from tac.uniward_delta import pack_sparse_delta as pkt` → {"pkt"}
      - `import tac.uniward_delta as uwd` → {"uwd.pack_sparse_delta"} (we
        track the module alias and match `<alias>.pack_sparse_delta`)
      - bare `pack_sparse_delta(` calls (defensive: also include the literal
        name even if the import is somewhere else / wildcard / re-export)
    Returns a set of acceptable callable string forms.
    """
    aliases: set[str] = {"pack_sparse_delta"}
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "uniward_delta" in mod or mod.endswith("tac"):
                for alias in node.names:
                    if alias.name == "pack_sparse_delta":
                        aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("tac.uniward_delta", "tac"):
                    module_aliases.add(alias.asname or alias.name)
    # Also accept attribute calls like `<module_alias>.pack_sparse_delta(...)`.
    aliases.update(f"{m}.pack_sparse_delta" for m in module_aliases)
    return aliases


def _call_func_str(call: ast.Call) -> str:
    """Render the function expression of a Call to its source-level string,
    handling Name + Attribute chains. Best-effort, returns "" on failure."""
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(call.func)
    except Exception:
        pass
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        # Walk chain to root Name.
        parts: list[str] = [call.func.attr]
        cur: ast.AST = call.func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _kwarg_value_is_approved(kw: ast.keyword) -> bool:
    """True if kw.value is the string literal 'approved' OR a Name whose
    id is in `_APPROVED_CONST_NAMES`. Conservative: anything we can't
    constant-fold (function call, ternary, formatted string) is treated as
    NOT approved (would otherwise be caught by the runtime gate).
    """
    v = kw.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value == _APPROVED_LITERAL
    if isinstance(v, ast.Name) and v.id in _APPROVED_CONST_NAMES:
        return True
    if isinstance(v, ast.Attribute) and v.attr in _APPROVED_CONST_NAMES:
        return True
    return False


def _scan_python_for_pack_sparse_delta_approved(
    path: Path, repo_root: Path
) -> list[str]:
    """Find every call to pack_sparse_delta(..., compliance_status='approved'/COMPLIANCE_APPROVED, ...)
    in `path`. The promotion-tool / test-fixture filter is applied by the
    caller (since this function returns ALL hits; the caller decides which
    are exempt)."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    try:
        tree = ast.parse(text, filename=str(rel))
    except SyntaxError:
        return []
    aliases = _resolve_pack_sparse_delta_aliases(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = _call_func_str(node)
        if func_str not in aliases:
            continue
        for kw in node.keywords:
            if kw.arg == "compliance_status" and _kwarg_value_is_approved(kw):
                violations.append(
                    f"{rel}:{node.lineno}: pack_sparse_delta(compliance_status="
                    f"'approved' | COMPLIANCE_APPROVED) outside the canonical "
                    f"promotion tool ({_PACK_SPARSE_DELTA_APPROVED_PROMO_FILE}). "
                    f"Lane C δ.bin promotion goes through "
                    f"tools/promote_lane_c_to_approved.py, which patches the "
                    f"wire header in-place after attestation verification — "
                    f"NOT by re-packing with compliance_status='approved'. "
                    f"(codex R5-3 #2.)"
                )
                break
    return violations


def check_no_pack_sparse_delta_approved_outside_promotion_tool(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch pack_sparse_delta(compliance_status='approved') outside the
    canonical promotion tool.

    Reference: codex R5-3 finding #2 + tac.lane_c_compliance INTERNAL_PROMOTION_TOKEN
    + tools/promote_lane_c_to_approved.py. The runtime check refuses to
    write 'approved' without a constant-time HMAC token; this static scan
    catches the same bug class at preflight time. Test fixtures that
    construct approved blobs (with the internal token) are explicitly
    permitted via path filter.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    # Scan both the standard meta dirs AND tools/ (the promotion tool lives
    # in tools/ and any future tools-side caller would be covered there too).
    scan_dirs = list(_META_PY_SCAN_DIRS) + ["tools"]
    for py in _iter_python_files(root, scan_dirs):
        n_scanned += 1
        rel = py.relative_to(root) if py.is_absolute() else py
        rel_str = rel.as_posix()
        # Exempt the canonical promotion tool itself.
        if rel_str == _PACK_SPARSE_DELTA_APPROVED_PROMO_FILE:
            continue
        # Exempt every test file under src/tac/tests/ (test fixtures legitimately
        # use the internal token to construct approved blobs for negative-path
        # coverage).
        if rel_str.startswith("src/tac/tests/") and rel.name.startswith("test_"):
            continue
        violations.extend(_scan_python_for_pack_sparse_delta_approved(py, root))

    if verbose and violations:
        print(
            f"  [no-pack-sparse-delta-approved] {len(violations)} violation(s) "
            f"across {n_scanned} files:"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-pack-sparse-delta-approved] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "PACK_SPARSE_DELTA APPROVED OUTSIDE PROMOTION TOOL violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nLane C promotion uses tools/promote_lane_c_to_approved.py "
            "(codex R5-3 #2)."
        )
    return violations


# ── Check 10: inflate.sh handles .br centrally before PYTHON_INFLATE dispatch ─
#
# Reference: codex R5-3 finding #11 + commit a1128fd9. Every PYTHON_INFLATE
# branch must see the archive in fully-decompressed form. The fix added a
# Stage 0 brotli decompression block BEFORE branch dispatch. This scanner
# enforces the block exists and is positioned correctly for any inflate.sh
# that performs PYTHON_INFLATE branch dispatch.

# Markers that identify the centralized brotli stage.
_BROTLI_BLOCK_MARKERS = ("brotli stage 0", "Stage 0")
# The brotli pull token: `--with brotli` is what the centralized block uses
# in the `uv run` invocation. This identifies the actual decompression path
# (vs a comment that mentions brotli without acting on it).
_BROTLI_WITH_TOKEN = "--with brotli"
# The br-file glob detector that triggers the block. Either the literal
# `compgen -G ...*.br` form OR a `*.br` file-test guard counts.
_BROTLI_BR_GLOB_TOKEN_RE = re.compile(r"\.br\b")
# Identifies the branch-dispatch line. Matches `if [ "$PYTHON_INFLATE" = ...`,
# `case "$PYTHON_INFLATE"`, etc.
_PYTHON_INFLATE_DISPATCH_RE = re.compile(
    r"""
    (
        \[\s*"\$PYTHON_INFLATE"   # if [ "$PYTHON_INFLATE" = ...
        |
        case\s+"\$PYTHON_INFLATE" # case "$PYTHON_INFLATE" in
    )
    """,
    re.VERBOSE,
)


def _scan_inflate_sh_for_centralized_brotli(
    path: Path, repo_root: Path
) -> list[str]:
    """Validate that path is either (a) a trivial passthrough with no
    PYTHON_INFLATE dispatch (PASS), or (b) contains a centralized brotli
    Stage 0 block BEFORE the PYTHON_INFLATE dispatch line (PASS), else
    a violation."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    lines = text.splitlines()
    # Locate first PYTHON_INFLATE dispatch line.
    dispatch_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        if _PYTHON_INFLATE_DISPATCH_RE.search(line):
            dispatch_lineno = i
            break

    if dispatch_lineno is None:
        # No branch dispatch → trivial passthrough. Skip.
        return []

    # Find the centralized brotli block. Three signals must all be present
    # AND must precede the dispatch line:
    #   1. A `Stage 0`/`brotli stage 0` marker (comment or echo).
    #   2. A `*.br` glob/file-test (compgen / [ -e ...*.br ] / similar).
    #   3. An `--with brotli` invocation of `uv run`.
    marker_lineno: int | None = None
    br_glob_lineno: int | None = None
    with_brotli_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        if i >= dispatch_lineno:
            break
        if marker_lineno is None and any(
            m.lower() in line.lower() for m in _BROTLI_BLOCK_MARKERS
        ):
            # Confirm it's a brotli marker (not e.g. "Stage 0" referring to
            # something else — require co-occurrence with "brotli" within ±10
            # lines OR on the same line).
            if "brotli" in line.lower():
                marker_lineno = i
            else:
                # Check ±10 line window for "brotli".
                window = "\n".join(
                    lines[max(0, i - 10): min(len(lines), i + 10)]
                ).lower()
                if "brotli" in window:
                    marker_lineno = i
        if br_glob_lineno is None and _BROTLI_BR_GLOB_TOKEN_RE.search(line):
            br_glob_lineno = i
        if with_brotli_lineno is None and _BROTLI_WITH_TOKEN in line:
            with_brotli_lineno = i

    violations: list[str] = []
    if marker_lineno is None or br_glob_lineno is None or with_brotli_lineno is None:
        # Block missing entirely. Determine which signal(s) are absent for
        # a precise diagnostic.
        missing: list[str] = []
        if marker_lineno is None:
            missing.append("'Stage 0'/'brotli stage 0' marker comment")
        if br_glob_lineno is None:
            missing.append("'*.br' file-glob guard")
        if with_brotli_lineno is None:
            missing.append("'--with brotli' uv-run invocation")
        violations.append(
            f"{rel}:{dispatch_lineno}: PYTHON_INFLATE dispatch present but "
            f"centralized brotli Stage 0 block is incomplete (missing: "
            f"{', '.join(missing)}). Every PYTHON_INFLATE branch must see "
            f"the archive in fully-decompressed form. Add the Stage 0 block "
            f"BEFORE the dispatch (codex R5-3 #11, commit a1128fd9)."
        )
        return violations

    # All three signals present BEFORE dispatch → PASS. Position is implicitly
    # validated by the loop (we stop at dispatch_lineno).

    # ALSO: detect the after-dispatch case. If a brotli block ALSO appears
    # AFTER dispatch (e.g. inside a branch arm) without the centralized one
    # before, the loop above already caught the centralized-missing case.
    # The spec calls out "probe-too-late" as a violation; the symmetric
    # check here is "brotli-block-too-late". We flag any --with brotli that
    # appears AFTER dispatch UNLESS a centralized one also exists before.
    # (Centralized-before passes; we have it, so no further work.)

    return violations


def check_inflate_sh_handles_br_centrally(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch inflate.sh files where .br decompression is missing or runs
    AFTER the PYTHON_INFLATE branch dispatch.

    Reference: codex R5-3 finding #11 + commit a1128fd9. Without the
    centralized Stage 0 block, any non-renderer PYTHON_INFLATE branch on
    a Lane B-alt archive fails later as a missing renderer.bin / masks.mkv
    with no actionable hint. Trivial passthrough inflate.sh (no
    PYTHON_INFLATE dispatch) is a soft pass — the block is unnecessary.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    submissions = root / "submissions"
    if submissions.exists():
        for sub_dir in sorted(submissions.iterdir()):
            if not sub_dir.is_dir():
                continue
            for p in sorted(sub_dir.glob("inflate.sh")):
                n_scanned += 1
                violations.extend(_scan_inflate_sh_for_centralized_brotli(p, root))

    if verbose and violations:
        print(
            f"  [inflate-br-central] {len(violations)} violation(s) across "
            f"{n_scanned} inflate.sh file(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [inflate-br-central] OK: {n_scanned} inflate.sh file(s) scanned")

    if violations and strict:
        raise MetaBugViolation(
            "INFLATE.SH BROTLI CENTRALIZATION violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nAdd Stage 0 brotli block before PYTHON_INFLATE dispatch "
            "(codex R5-3 #11, commit a1128fd9)."
        )
    return violations


# ── Check 11: scripts/remote_*.sh must run NVDEC probe at Stage 0 ────────────
#
# Reference: feedback_vastai_nvdec_host_variation + commit eef64293. NVDEC
# host availability is host-dependent on Vast.ai 4090s — same image, same
# driver, different host = `CUDA_ERROR_NO_DEVICE` from DALI's video MIXED
# operator. The probe catches the bad-host case in 5 seconds. Every remote
# script that does GPU work MUST run `scripts/probe_nvdec.sh` BEFORE any
# GPU spend (training, pose TTO, archive build, evaluate.py, nvidia-smi
# query against driver).

# Token strings that identify a probe invocation. We match all three
# documented forms in the spec.
_NVDEC_PROBE_TOKENS = (
    "scripts/probe_nvdec.sh",  # bash $WORKSPACE/scripts/probe_nvdec.sh
    "probe_nvdec.sh",          # bash probe_nvdec.sh (relative)
)
# Comment header that explicitly opts out of the requirement. Operator's
# declaration that this script does no DALI / NVDEC video work.
_NVDEC_OPT_OUT_TOKEN = "NO_NVDEC_NEEDED"
# GPU-work markers: presence of any of these tokens means a probe call MUST
# precede them in the file. We accept partial substrings (e.g.
# `train_renderer.py` matches both `train_renderer.py` and
# `src/tac/experiments/train_renderer.py`).
_NVDEC_GPU_WORK_MARKERS = (
    "train_renderer.py",
    "optimize_poses.py",
    "experiments/build_baseline_archive.py",
    "build_baseline_archive.py",
    "train_distill.py",
    "auth_eval_renderer.py",
    "evaluate.py",
    "nvidia-smi",
)


def _scan_remote_script_for_nvdec_probe(
    path: Path, repo_root: Path
) -> list[str]:
    """Validate that path either opts out via NO_NVDEC_NEEDED OR contains
    a probe call BEFORE any GPU-work marker."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()

    # Opt-out: search the first 30 lines for the NO_NVDEC_NEEDED comment
    # header. Anywhere in the file would also work, but a header makes the
    # operator's intent reviewable.
    header = "\n".join(lines[:30])
    if _NVDEC_OPT_OUT_TOKEN in header:
        return []

    # Find earliest GPU-work marker line (1-indexed; None if no GPU work).
    gpu_work_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        # Skip comment-only lines for marker detection.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if any(tok in line for tok in _NVDEC_GPU_WORK_MARKERS):
            gpu_work_lineno = i
            break

    # Find earliest probe-call line.
    probe_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if any(tok in line for tok in _NVDEC_PROBE_TOKENS):
            probe_lineno = i
            break

    violations: list[str] = []

    if gpu_work_lineno is None:
        # Script does no GPU work and didn't opt out. If the probe is also
        # absent that's fine — nothing to probe FOR. PASS.
        return []

    if probe_lineno is None:
        violations.append(
            f"{rel}:{gpu_work_lineno}: GPU-work marker present but no NVDEC "
            f"probe call. Add `bash \"$WORKSPACE/scripts/probe_nvdec.sh\"` "
            f"as Stage 0 (BEFORE any GPU spend), OR add a "
            f"`# {_NVDEC_OPT_OUT_TOKEN}` comment header to opt out. "
            f"(feedback_vastai_nvdec_host_variation, commit eef64293.)"
        )
        return violations

    if probe_lineno >= gpu_work_lineno:
        violations.append(
            f"{rel}:{probe_lineno}: NVDEC probe call appears AFTER first "
            f"GPU-work marker (line {gpu_work_lineno}). Probe MUST run "
            f"BEFORE any GPU spend so a bad-host case is caught in 5s "
            f"instead of after $0.20+ of work. Move the probe to the top "
            f"of the script, BEFORE the first GPU-work invocation. "
            f"(feedback_vastai_nvdec_host_variation, commit eef64293.)"
        )

    return violations


def check_remote_scripts_have_nvdec_probe(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch scripts/remote_*.sh that do GPU work without an NVDEC probe.

    Reference: feedback_vastai_nvdec_host_variation memory entry + commit
    eef64293. The probe catches bad Vast.ai hosts in 5 seconds; without
    it, training proceeds successfully and only fails at the eval stage,
    burning $0.20-$10 per occurrence (this happened TWICE on 2026-04-27).
    Scripts that do no DALI / NVDEC work can opt out via a
    `# NO_NVDEC_NEEDED` comment header.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        for p in sorted(scripts_dir.glob("remote_*.sh")):
            n_scanned += 1
            violations.extend(_scan_remote_script_for_nvdec_probe(p, root))

    if verbose and violations:
        print(
            f"  [remote-nvdec-probe] {len(violations)} violation(s) across "
            f"{n_scanned} remote_*.sh file(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [remote-nvdec-probe] OK: {n_scanned} remote_*.sh file(s) scanned")

    if violations and strict:
        raise MetaBugViolation(
            "REMOTE NVDEC PROBE violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nAdd Stage 0 NVDEC probe (feedback_vastai_nvdec_host_variation, "
            "commit eef64293)."
        )
    return violations


# NOTE: 2026-04-27 codex R5-3 Finding #4 — all 8 meta-bug checks are wired
# into preflight_all() above (warn-only). See the codex R5-3 #4 comment block
# in preflight_all() for live-violation counts and per-check promotion plan.
# The 3 follow-on checks (codex R5-3 #2 + #11 + NVDEC probe gap, commits
# a1128fd9 / eef64293 + this commit) are wired in the same block.


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
# .amrc = Yousfi council #8 lossless argmax-RLE mask codec (2026-04-26).
_ARTIFACT_SUFFIXES = (".bin", ".pt", ".pth", ".mkv", ".mp4", ".raw",
                      ".zip", ".tar", ".tar.gz", ".tgz", ".amrc")

# Filenames that are deliberately external (not produced by our code) — they
# come from upstream data, the contest archive, third-party tools, etc.
_EXTERNAL_FILENAMES = {
    "0.mkv",  # upstream/videos/0.mkv (contest GT)
    "masks.mkv", "masks.amrc",  # mask artifacts (av1 + lossless argmax-RLE)
    "poses.pt", "renderer.bin",  # contest-required submission filenames
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


def preflight_build_renderer_signature(strict: bool = True, verbose: bool = True) -> list[str]:
    """Validate that build_renderer() accepts every arch knob set by any
    renderer training profile. The 2026-04-26 DEN arch drift bug existed
    because build_renderer() didn't accept use_zoom_flow/use_dsconv/
    padding_mode/use_dilation/pose_dim — the resolver in train_renderer
    set the args.* fields correctly but the build_renderer call silently
    dropped them. Result: 1.2h of wasted GPU on a checkpoint that
    consumers couldn't load.

    This rule introspects build_renderer's signature and confirms every
    profile-declared arch field has a matching kwarg. Catches the bug
    at lint time, not 1 hour into a $0.30 GPU run.
    """
    violations: list[str] = []
    try:
        import inspect
        from tac.renderer import build_renderer
        from tac.profiles import PROFILES
    except ImportError as e:
        msg = f"  [build_renderer_sig] cannot import: {e}"
        if verbose:
            print(msg)
        return [msg]

    sig = inspect.signature(build_renderer)
    accepted = set(sig.parameters.keys())

    arch_flags = (
        "use_zoom_flow", "use_dsconv", "padding_mode", "use_dilation",
        "pose_dim", "base_ch", "mid_ch", "embed_dim", "motion_hidden", "depth",
    )
    for prof_name, prof in PROFILES.items():
        if prof.get("experiment_type") != "renderer_training":
            continue
        for flag in arch_flags:
            if flag in prof and flag not in accepted:
                violations.append(
                    f"profile {prof_name!r} declares arch flag {flag!r} but "
                    f"build_renderer() does NOT accept it as a kwarg. The "
                    f"value is silently dropped at the call site, causing "
                    f"arch drift between profile spec and saved checkpoint. "
                    f"Add {flag!r} to build_renderer's signature + forward "
                    f"to MaskRenderer/MotionPredictor/AsymmetricPairGenerator."
                )

    if verbose and violations:
        print(f"  [build_renderer_sig] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [build_renderer_sig] OK: build_renderer accepts all "
              f"{len(arch_flags)} arch kwargs")

    if violations and strict:
        raise PreflightError(
            "BUILD_RENDERER SIGNATURE VIOLATIONS:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


def preflight_canonical_checkpoints(strict: bool = True, verbose: bool = True) -> list[str]:
    """Validate that every training producer's emitted checkpoint name is
    in the canonical registry (tac.checkpoint_names.canonical_checkpoint_names).

    Without this, deploys aborted at Stage 4 of the bootstrap because the
    producer wrote `renderer_<profile>_best_fp32.pt` but the consumer probe
    only had `distill_*.pt`. We wasted a full DEN training run on 2026-04-26
    before realising this. Now: any new training script that emits a
    different name MUST be added to PRODUCER_OUTPUTS in checkpoint_names.py
    AND its filename MUST appear in canonical_checkpoint_names() output.
    """
    violations: list[str] = []
    try:
        from tac.checkpoint_names import (
            PRODUCER_OUTPUTS,
            canonical_checkpoint_names,
        )
    except ImportError as e:
        msg = f"  [canonical_checkpoints] cannot import tac.checkpoint_names: {e}"
        if verbose:
            print(msg)
        return [msg]

    # Build the set of all canonical names across all known profiles. Each
    # profile-specific name has a placeholder so we strip the profile and
    # check the suffix pattern.
    try:
        from tac.profiles import PROFILES
        profiles = sorted(PROFILES.keys())
    except ImportError:
        profiles = []

    all_canonical: set[str] = set(canonical_checkpoint_names(profile=None))
    for prof in profiles:
        all_canonical.update(canonical_checkpoint_names(profile=prof))

    for producer_path, expected_name in PRODUCER_OUTPUTS.items():
        # Substitute <profile> placeholder if present.
        if "<profile>" in expected_name:
            # Match against any profile-instantiated form.
            matched = any(
                name.startswith("renderer_") and name.endswith("_best_fp32.pt")
                for name in all_canonical
            )
        else:
            matched = expected_name in all_canonical
        if not matched:
            violations.append(
                f"checkpoint_names.PRODUCER_OUTPUTS[{producer_path!r}] = "
                f"{expected_name!r} but that name is NOT in "
                f"canonical_checkpoint_names() output. Update either the "
                f"producer's output naming or canonical_checkpoint_names() "
                f"to match. 2026-04-26 hardening: this catches the "
                f"renderer_<profile>_best_fp32.pt vs distill_*.pt mismatch "
                f"that wasted a DEN training run."
            )

    if verbose and violations:
        print(f"  [canonical_checkpoints] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [canonical_checkpoints] OK: {len(PRODUCER_OUTPUTS)} producer(s) "
              f"validated against {len(all_canonical)} canonical name(s)")

    if violations and strict:
        raise PreflightError(
            "CANONICAL CHECKPOINT NAMES VIOLATIONS:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


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

    # ── AMRC mask-file validation hook ──
    # If any archive directory under the repo has a masks.amrc artifact,
    # validate its magic + header. This catches a future regression where
    # a producer writes a malformed AMRC blob without anyone noticing.
    amrc_violations = _validate_amrc_artifacts(root)
    violations.extend(amrc_violations)
    if amrc_violations and verbose:
        for v in amrc_violations:
            print(f"    • [amrc] {v}")

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


def _validate_amrc_artifacts(root: Path) -> list[str]:
    """Walk the repo for any *.amrc files in archive-like directories and
    validate they begin with the AMRC magic bytes + a current version.

    Searches: submissions/robust_current/**/*.amrc and
    experiments/results/**/*.amrc (the conventional archive output dirs).
    Skips directories that don't exist (this preflight is non-fatal in
    those cases).
    """
    findings: list[str] = []
    candidate_dirs = [
        root / "submissions" / "robust_current",
        root / "experiments" / "results",
    ]
    try:
        from tac.lossless.argmax_codec import validate_amrc_file
    except ImportError as e:
        # Codec module not yet built — skip the check rather than fail
        # the whole preflight. The contract violation list will still
        # surface if a consumer reads masks.amrc but no producer writes it.
        findings.append(
            f"argmax_codec not importable ({e}); skipping AMRC validation"
        )
        return findings
    for d in candidate_dirs:
        if not d.exists():
            continue
        for amrc in d.rglob("*.amrc"):
            try:
                validate_amrc_file(amrc)
            except (ValueError, OSError) as e:
                findings.append(
                    f"{amrc}: invalid AMRC header — {e}"
                )
    return findings


# ── Loader format safety ──────────────────────────────────────────────────────
#
# Bug class this catches: a consumer (engineered_quant_noise.py,
# pair_difficulty_map.py, kaggle_auth_eval_renderer.py, etc.) imports a
# `load_renderer` helper that does a bare `torch.load(path, weights_only=False)`
# on a path whose actual on-disk format is one of our binary exports
# (FP4A/ASYM/DPSM/I4LZ). torch.load tries to interpret the magic bytes as
# pickle, fails, and crashes with "could not convert string to float: 'P4AV'"
# (DEN-V2 2026-04-26).
#
# Permanent fix: every `load_renderer`-style helper in the codebase MUST
# content-detect the format. This validator AST-scans for the unsafe pattern.


class LoaderFormatSafetyError(Exception):
    """A consumer would torch.load a file path that might be a non-pickle
    binary export (FP4A/ASYM/DPSM/I4LZ)."""


# Module-relative names of canonical content-detecting loaders. A function
# call resolved (statically) to one of these is treated as safe.
_SAFE_LOADER_QUALNAMES = frozenset({
    # Renderer loaders
    "load_renderer",  # the canonical one in precompute_gradient_corrections
    "load_any_renderer_checkpoint",
    "load_asymmetric_checkpoint_fp4",
    "load_asymmetric_checkpoint",
    "load_renderer_checkpoint",
    "detect_checkpoint_type",
    "load_int4_lzma2",
    # Pose loaders (use the same content-detect pattern; see submission_archive)
    "load_optimized_poses",
    "load_poses_binary",
})


def _scan_python_for_unsafe_renderer_loader(path: Path) -> list[str]:
    """AST-scan a Python file for two related anti-patterns:

      1. `def load_renderer(...)` whose body calls `torch.load(...)` directly
         on the checkpoint argument WITHOUT a content-magic dispatch beforehand.
         (Producer-side: the loader is unsafe.)
      2. Bare `torch.load(<some>.bin / "*.bin" / a variable spelled "checkpoint*")`
         outside of a function known to be content-detecting.
         (Consumer-side: the call site is unsafe.)

    Returns a list of human-readable violations. Empty if clean.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return [f"{path}: SyntaxError (cannot parse)"]

    violations: list[str] = []

    # --- Pattern 1: any function whose name matches a known loader-shape
    # MUST content-detect the format (or delegate to a safe loader). Original
    # rule only matched `load_renderer*`; Contrarian R2 V3 (2026-04-26)
    # showed a refactor to `load_checkpoint`/`load_model`/`load_weights`/
    # `_load_ckpt`/`restore_model` would silently bypass the gate. The
    # expanded set catches the realistic rename surface.
    SAFE_MAGIC_TOKENS = ("FP4A", "ASYM", "DPSM", "I4LZ", "PK\\x03\\x04")

    def _is_loader_name(name: str) -> bool:
        """Pattern 1 trigger: function names that are likely renderer/model
        loaders. Intentionally broad — a false positive is a 1-line magic
        check; a false negative is a DEN-V2-class production crash.

        Contrarian R2 V3 (2026-04-26): expanded from `load_renderer*` only
        to also catch `load_*`/`_load_*`/`restore_*` on model/renderer/
        checkpoint/ckpt/weights/net suffixes — i.e. the realistic rename
        surface that would silently bypass the original gate.

        Exclusions: training-state and optimizer-state loaders are NOT
        renderer artifacts (they're always pickle by construction —
        optimizer state isn't tensor-only), so we exempt those names to
        avoid noise.
        """
        n = name.lower()
        # Any `load_*` / `_load_*` / `restore_*` / `_restore_*`
        # whose suffix names a model/checkpoint-shaped object.
        loader_prefixes = ("load_", "_load_", "restore_", "_restore_")
        if not any(n.startswith(p) for p in loader_prefixes):
            return False
        # Explicitly NOT renderer loaders (they're always pickle by design).
        non_renderer_suffixes = (
            "training_state",
            "optimizer_state",
            "optimizer",
            "scheduler",
            "trainer_state",
        )
        if any(tok in n for tok in non_renderer_suffixes):
            return False
        # 2026-04-26 Mario R2 CRITICAL #1: explicit allowlist for known
        # non-renderer loaders that the broad pattern (#1 below) would
        # false-positive on. These are TRUSTED — they don't load the FP4
        # renderer artifact format. Adding here exempts the function from
        # Pattern 1 scan but consumers will still be caught by the call-site
        # scan (Pattern 2) if they ever pass a renderer.bin path.
        TRUSTED_NON_RENDERER_LOADERS = frozenset({
            "load_checkpoint_weights",     # train_distill.py — training resume
            "load_network_codec",          # network_codec.py — NeRV codec, not renderer
            "load_checkpoint_state_dict",  # ensemble.py — ensemble combiner
            "load_compressed_weights",     # generic int-quant deserializer
            "load_postfilter",             # postfilter (different artifact class)
        })
        if name in TRUSTED_NON_RENDERER_LOADERS:
            return False
        # Suffix must look model/renderer/checkpoint-shaped.
        loader_suffix_tokens = (
            "renderer",
            "model",
            "checkpoint",
            "ckpt",
            "weights",
            "net",
        )
        return any(tok in n for tok in loader_suffix_tokens)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_loader_name(node.name):
            continue
        body_src = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if not body_src:
            continue
        # Safe iff (a) the body mentions a known magic token, OR (b) the body
        # delegates to one of the canonical safe loaders.
        has_magic = any(tok in body_src for tok in SAFE_MAGIC_TOKENS)
        delegates = any(
            f"{nm}(" in body_src for nm in _SAFE_LOADER_QUALNAMES
            if nm != node.name  # don't credit self-recursion
        )
        # Also consider it safe if it explicitly content-checks via a magic
        # variable name pattern (e.g., `magic = raw[:4]`).
        does_magic_read = bool(
            re.search(r"\.read\(\s*4\s*\)", body_src)
            or re.search(r"\[\s*:\s*4\s*\]", body_src)
            or re.search(r"\b_PICKLE_MAGICS\b", body_src)
            or re.search(r"\b_RENDERER_PICKLE_MAGICS\b", body_src)
            or re.search(r"\b_looks_like_pytorch_pickle\b", body_src)
            or re.search(r"\b_looks_like_pickle\b", body_src)
        )
        if has_magic or delegates or does_magic_read:
            continue
        # Otherwise, look for a torch.load call in the body. If found AND
        # it uses weights_only=False (DEN-V2's exact failure mode — the
        # legacy pickle path that crashes cryptically on FP4A magic), the
        # function is unsafe. Calls with weights_only=True are tensor-only
        # state-dict loads and cannot trigger the FP4A pickle crash, so
        # they are not the DEN-V2 bug class.
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn_str = ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
            if fn_str not in ("torch.load", "torch.frombuffer"):
                continue
            # Check weights_only=False (the DEN-V2 failure mode).
            uses_legacy_pickle = False
            for kw in sub.keywords:
                if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is False:
                        uses_legacy_pickle = True
                        break
            if not uses_legacy_pickle:
                continue
            violations.append(
                f"{path}:{node.lineno}: function `{node.name}` calls "
                f"`{fn_str}(..., weights_only=False)` without "
                f"content-detecting the file format first. This is the "
                f"2026-04-26 DEN-V2 bug pattern: torch.load on an "
                f"FP4A/ASYM/DPSM/I4LZ .bin file crashes with 'could not "
                f"convert string to float'. (Detected via expanded "
                f"loader-name match — load_*/restore_*/_load_*/_restore_* "
                f"over renderer/model/checkpoint/ckpt/weights/state/net; "
                f"Contrarian R2 V3 fix.) Either add a magic-byte dispatch "
                f"(read first 4 bytes, branch on FP4A/ASYM/DPSM/I4LZ vs "
                f"PyTorch pickle) OR delegate to "
                f"experiments.precompute_gradient_corrections.load_renderer "
                f"(the canonical content-detecting loader)."
            )
            break  # one violation per function is enough

    # --- Pattern 2: any module-level (NOT inside a safe-named function) call
    # like `torch.load(<arg>)` where the arg is a Name spelled like a
    # checkpoint path. Skip calls that are inside a function we already know
    # is safe (i.e., one whose body had the magic check above).

    # Build a parent-pointer map.
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def _enclosing_fn(node: ast.AST) -> ast.FunctionDef | None:
        cur = parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(id(cur))
        return None

    # Pattern 2 is intentionally NARROW: only flag when the FIRST positional
    # arg looks SPECIFICALLY like a renderer-checkpoint variable (not just any
    # "ckpt" — that's a TTO batch checkpoint, an optimizer state, etc.) AND
    # the call uses `weights_only=False` (DEN-V2's exact failure mode — the
    # legacy pickle path).
    #
    # The Contrarian forced this narrowing: an over-broad rule that flags
    # every torch.load in the repo gets disabled, defeating the whole point.
    # The tight rule stays on, catches the real DEN-V2 class without
    # false-positing TTO checkpoint resume, training-state loads, etc.

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if fn_str != "torch.load":
            continue
        if not node.args:
            continue

        # Require weights_only=False (or absent → defaults vary; tighten by
        # requiring explicit False since that's the DEN-V2 failure mode).
        has_weights_only_false = False
        for kw in node.keywords:
            if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant):
                if kw.value.value is False:
                    has_weights_only_false = True
        if not has_weights_only_false:
            continue

        # The first positional must be a "renderer-like" reference:
        #   - a Name spelled with "renderer" (NOT just "checkpoint" / "ckpt"
        #     which is too broad)
        #   - OR a literal `.bin` filename
        #   - OR a Call whose unparsed text contains "renderer"
        first = node.args[0]
        looks_renderer = False
        if isinstance(first, ast.Name):
            ident = first.id.lower()
            if "renderer" in ident:
                looks_renderer = True
        elif isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value.endswith(".bin"):
                looks_renderer = True
        elif isinstance(first, ast.Call):
            sub_str = ast.unparse(first) if hasattr(ast, "unparse") else ""
            if "renderer" in sub_str.lower():
                looks_renderer = True
        if not looks_renderer:
            continue

        # If it's inside a function whose body has a magic check (covered by
        # Pattern 1's safe-classification logic), let Pattern 1 own it.
        enc = _enclosing_fn(node)
        if enc is not None:
            enc_src = ast.unparse(enc) if hasattr(ast, "unparse") else ""
            if any(tok in enc_src for tok in SAFE_MAGIC_TOKENS):
                continue
            if any(f"{nm}(" in enc_src for nm in _SAFE_LOADER_QUALNAMES):
                continue

        # Test files are allowed to construct intentionally-wrong inputs.
        if "/tests/" in str(path) or "test_" in path.name:
            continue

        violations.append(
            f"{path}:{node.lineno}: bare `torch.load(<renderer-like>, "
            f"weights_only=False)` with no content-magic dispatch. "
            f"Use experiments.precompute_gradient_corrections.load_renderer "
            f"(the canonical content-detecting loader) or "
            f"tac.renderer_export.load_any_renderer_checkpoint instead. "
            f"(Bug pattern: DEN-V2 2026-04-26 — torch.load on FP4A .bin "
            f"crashes cryptically.)"
        )

    return violations


def preflight_loader_format_safety(
    repo_root: Path | None = None,
    scan_dirs: list[str] | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Validate that every renderer checkpoint loader in the repo is
    content-detecting (NOT bare torch.load).

    Two scans per file:
      1. Every `def load_renderer*` body must do magic-byte dispatch OR
         delegate to a known safe loader.
      2. No bare `torch.load(<checkpoint-like>)` outside a safe loader.

    Skips test/smoke files (they construct intentionally-wrong inputs).

    Returns the list of violations found. If `strict` and non-empty, raises
    LoaderFormatSafetyError.
    """
    root = repo_root or REPO_ROOT
    scan_dirs = scan_dirs or [
        "experiments",
        "src/tac",
        "submissions/robust_current",
    ]

    all_violations: list[str] = []
    n_scanned = 0
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for py_path in d_path.rglob("*.py"):
            n_scanned += 1
            all_violations.extend(_scan_python_for_unsafe_renderer_loader(py_path))

    if verbose:
        if all_violations:
            print(f"  [loader-format] {len(all_violations)} violation(s) "
                  f"across {n_scanned} files:")
            for v in all_violations:
                print(f"    • {v}")
        else:
            print(f"  [loader-format] OK: {n_scanned} files clean — every "
                  f"renderer loader is content-detecting")

    if all_violations and strict:
        raise LoaderFormatSafetyError(
            "LOADER FORMAT SAFETY VIOLATIONS — a consumer would torch.load a "
            "path that might be a non-pickle binary export. This is the "
            "2026-04-26 DEN-V2 bug class:\n"
            + "\n".join(f"  • {v}" for v in all_violations)
            + "\n\nFix: use experiments.precompute_gradient_corrections."
            "load_renderer (the canonical content-detecting loader) or add "
            "magic-byte dispatch to your local helper. Suffix-based dispatch "
            "is forbidden — it is what burned us in DEN-V2 (FP4 .bin) and "
            "SHIRAZ (pickle .bin)."
        )
    return all_violations


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
    # 2026-04-26 hardening: every renderer profile MUST declare seed +
    # deterministic explicitly. tools/check_determinism.py refuses to run
    # without them. SHIRAZ launch crashed mid-deploy on this exact missing
    # key on 2026-04-26.
    "seed", "deterministic",
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

        # Fridrich council #1 (2026-04-26): dct_quant_weight bounds check.
        # Catches typo'd huge values (e.g. 50.0) that would dominate the loss
        # stack and starve the scorer signal. Reasonable range: 0 (off) to
        # 10.0 (heavy weight, larger than any other Fridrich aux loss in DEN).
        dqw = prof.get("dct_quant_weight")
        if dqw is not None:
            if not isinstance(dqw, (int, float)):
                violations.append(
                    f"profile {name!r} dct_quant_weight={dqw!r} type "
                    f"{type(dqw).__name__}, expected float"
                )
            elif not (0.0 <= float(dqw) <= 10.0):
                violations.append(
                    f"profile {name!r} dct_quant_weight={dqw} out of range "
                    f"[0.0, 10.0] — values >10 would overwhelm scorer signal "
                    f"and starve PoseNet/SegNet gradients."
                )

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


def preflight_bootstrap_safety(
    scripts_dir: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Scan scripts/*_bootstrap.sh for the silent-failure cascade patterns
    that nuked LANE-B (2026-04-26, 6.5h + ~$2 wasted).

    The LANE-B kill chain (post-mortem in feedback_zip_dep_bootstrap_trap.md):
      1. PyTorch container has no `zip` binary; shell `zip` failed.
      2. `set -uo pipefail` (no `-e`) didn't abort on the failure.
      3. Empty ARCHIVE_BYTES crashed auth_eval at the very end.

    This preflight catches #1 and #2 statically by reading every bootstrap
    script's source. Patterns enforced:

      A. `set -euo pipefail` (or any -e* form) — `-e` is non-negotiable.
      B. No bare `zip` shell command (use python `zipfile.ZipFile` instead).

    Each violation explains what went wrong and the canonical fix.

    Args:
        scripts_dir: directory containing *_bootstrap.sh (defaults to repo
            scripts/). Pass a different path for testing.
        strict: raise PreflightError on any violation.
        verbose: print summary.

    Returns:
        list of violation strings (may be empty).
    """
    import re
    from pathlib import Path as _Path

    if scripts_dir is None:
        # Repo root resolution — preflight.py lives in src/tac/, so up two.
        scripts_dir = _Path(__file__).resolve().parents[2] / "scripts"
    scripts_dir = _Path(scripts_dir)

    violations: list[str] = []
    if not scripts_dir.is_dir():
        msg = f"  [bootstrap] scripts dir not found: {scripts_dir}"
        if verbose:
            print(msg)
        return [msg]

    bootstraps = sorted(scripts_dir.glob("*_bootstrap.sh"))
    if not bootstraps:
        if verbose:
            print(f"  [bootstrap] no *_bootstrap.sh found in {scripts_dir}")
        return []

    # Match `set -e`, `set -eu`, `set -euo`, `set -ue`, etc. — any combination
    # that includes a literal `-e` flag (with or without -u / -o / pipefail).
    SET_E_RE = re.compile(r"^\s*set\s+-[a-z]*e[a-z]*(\s|$)", re.MULTILINE)

    for path in bootstraps:
        text = path.read_text()

        # Strip comments + heredocs lazily — we want code-line analysis only.
        code_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
        code = "\n".join(code_lines)

        # A. set -e flag present
        if not SET_E_RE.search(code):
            violations.append(
                f"{path.name}: missing `set -e` (any -e* flag) — silent "
                f"command failures will cascade. LANE-B died this way: "
                f"`zip` failed, script kept running, 6.5h of pose TTO "
                f"output got crashed at the very end. Use "
                f"`set -euo pipefail` (matches the other bootstraps)."
            )

        # B. No `zip` shell binary (PyTorch container doesn't ship it).
        # Match `zip ` at command position, not `zipfile`/`unzip`/`gzip`.
        bad = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
        if bad:
            violations.append(
                f"{path.name}: invokes `zip` shell binary (match: "
                f"{bad.group(0).strip()!r}). The PyTorch CUDA container "
                f"`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` does NOT "
                f"ship `zip` — the command will silently fail. Use python "
                f"`zipfile.ZipFile` instead (no apt dep, deterministic)."
            )

    if verbose and violations:
        print(f"  [bootstrap] {len(violations)} violation(s) across {len(bootstraps)} script(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [bootstrap] OK: {len(bootstraps)} bootstrap script(s) clean")

    if violations and strict:
        raise PreflightError(
            "BOOTSTRAP SCRIPT SAFETY FAILED (LANE-B kill chain):\n"
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
            CodebaseDriftError, LoaderFormatSafetyError) as e:
        print(f"\nPREFLIGHT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
