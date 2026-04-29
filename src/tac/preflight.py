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
import datetime as _dt
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

        # 2026-04-27 codex R5-3 Finding #4 + R5-3-r3: wire all 11 meta-bug
        # checks (FORBIDDEN PATTERNS / CLAUDE.md) into preflight_all STRICT.
        # Live-codebase counts went 40 → 0 across F1 (commit 7d2b5299), F2
        # (commit a94a9325), and the codex-round-4 probe-before-DALI fix.
        # Every entry point — pre-commit hook (tools/preflight_hook.py), CI
        # (.github/workflows/ci.yml), and any direct preflight_all caller —
        # now BLOCKS at commit/PR/run time on any new violation. The bug
        # classes that wasted days of GPU time + multiple rounds of council
        # rework are now structurally extinct. Reverting any of these fixes
        # will fail strict here.
        check_no_mps_fallback_default(strict=True, verbose=verbose)
        check_shell_set_e_present(strict=True, verbose=verbose)
        check_no_shell_zip_binary(strict=True, verbose=verbose)
        check_no_pipefail_grep_q_trap(strict=True, verbose=verbose)
        check_no_eval_roundtrip_false(strict=True, verbose=verbose)
        check_no_scorer_load_at_inflate(strict=True, verbose=verbose)
        check_training_scripts_have_auth_eval(strict=True, verbose=verbose)
        check_no_disable_eval_roundtrip_flag(strict=True, verbose=verbose)
        check_no_pack_sparse_delta_approved_outside_promotion_tool(strict=True, verbose=verbose)
        check_inflate_sh_handles_br_centrally(strict=True, verbose=verbose)
        check_remote_scripts_have_nvdec_probe(strict=True, verbose=verbose)

        # 2026-04-27 codex R5-r6: 5 new checks for round-6 findings.
        # Each guards a regression of the matching finding fix. All 5 land
        # at 0 live-codebase violations (verified post-fix), so they go
        # straight to strict=True per the Lane A → strict pattern.
        check_no_brittle_six_line_waiver_lookback(strict=True, verbose=verbose)
        check_kl_distill_uses_roundtripped_frames(strict=True, verbose=verbose)
        check_eval_roundtrip_gate_called_after_output_dir_resolution(strict=True, verbose=verbose)
        check_nvdec_probe_has_error_classification(strict=True, verbose=verbose)
        check_archive_builders_use_deterministic_zip(strict=True, verbose=verbose)

        # 2026-04-27 meta-bug audit (commit a57731a0): 12 NEW checks for
        # additional bug classes from session + memory. 4 land at 0 live
        # violations and go straight to strict; the other 8 have real
        # existing violations and stay warn-only until a cleanup pass.
        # Per-check live counts at wire-in time:
        #   check_vastai_create_has_label                 0  → STRICT
        #   check_waivers_specify_env_gate                0  → STRICT
        #   check_inflate_scorer_load_has_runtime_banner  0  → STRICT
        #   check_vastai_prompts_have_cost_cap            0  → STRICT
        #   check_vastai_create_writes_tracker            2  warn
        #   check_subagent_prompts_no_cpu_fallback        1  warn
        #   check_scores_have_lane_tag                   20  warn (run_log/findings cleanup)
        #   check_halfframe_archive_uses_trained_profile  2  warn
        #   check_profile_keys_have_resolvers            91  warn (real audit needed — same class as pose_dim)
        #   check_test_files_imports_resolve             25  warn (broken-test cleanup)
        #   check_uniward_delta_has_attestation_gate      6  warn
        #   check_remote_scripts_write_provenance         5  warn (Lane provenance write)
        check_vastai_create_has_label(strict=True, verbose=verbose)
        check_waivers_specify_env_gate(strict=True, verbose=verbose)
        check_inflate_scorer_load_has_runtime_banner(strict=True, verbose=verbose)
        check_vastai_prompts_have_cost_cap(strict=True, verbose=verbose)
        # 2026-04-27 final cleanup pass: 8 warn-only checks now at 0
        # live violations (commits eb985e40 + 17e5f903 + 676bf206 + this).
        # Promoted to strict — bug classes structurally extinct.
        check_vastai_create_writes_tracker(strict=True, verbose=verbose)
        check_subagent_prompts_no_cpu_fallback(strict=True, verbose=verbose)
        check_scores_have_lane_tag(strict=True, verbose=verbose)
        check_halfframe_archive_uses_trained_profile(strict=True, verbose=verbose)
        check_profile_keys_have_resolvers(strict=True, verbose=verbose)
        check_test_files_imports_resolve(strict=True, verbose=verbose)
        check_uniward_delta_has_attestation_gate(strict=True, verbose=verbose)
        check_remote_scripts_write_provenance(strict=True, verbose=verbose)

        # 2026-04-27 council forensics (findings.md "Lane G — really dead,
        # or bugged?"): forbid `F.kl_div(..., reduction="batchmean")` on
        # spatial tensors. The bug under-divides the per-pixel mean by
        # H × W (=196,608 for 384×512 SegNet), silently over-weighting
        # every caller. Lands at 0 live violations after the losses.py
        # fix → straight to strict per the Lane A pattern. See Check M
        # comment block above the function definition.
        check_kl_div_reduction_correct(strict=True, verbose=verbose)

        # 2026-04-27 forensic council (findings.md "Lane F regression"):
        # 29th meta-bug check. Forbid the silent-default-masquerading-as-
        # negative-result pattern (auto-discover from N hardcoded paths +
        # WARN-and-proceed instead of raise). Lane F (qat_finetune.py) +
        # Lane G (kl_distill_weight default) are the 2 known instances —
        # both fixed; live count after qat_finetune.py fix should be 0.
        # See Check N comment block above the function definition + memory
        # `feedback_silent_default_masquerading_as_negative_result`.
        check_no_silent_auto_discovery_with_warn(strict=True, verbose=verbose)

        # 2026-04-27: 3 new meta-bug checks (30, 31, 32) for DX hardening.
        # All STRICT after sweep-fix landed in this commit:
        # - Check 30 (executable-bit): Lane GH bug + 6 historical chmod'd
        # - Check 31 (predicted_band): 8 lane scripts patched with band metadata
        # - Check 32 (contest-cuda-tag): 8 lane scripts patched with [contest-CUDA]
        # Bootstraps + sweep orchestrators + auth-eval-only scripts EXEMPT
        # via EXEMPT_SUFFIXES list inside each check function.
        check_remote_scripts_executable_bit(strict=True, verbose=verbose)
        check_remote_scripts_record_predicted_band(strict=True, verbose=verbose)
        check_remote_scripts_tag_contest_cuda_at_completion(strict=True, verbose=verbose)

        # 2026-04-28: 2 more strict meta-bug checks (33, 34) from overnight
        # deploy failures. Both STRICT after the comment-stripping fix:
        # NVDEC 7/12 hosts bad → probe must be Stage 0.
        # Lane S motion.head 6-vs-4 mismatch → resume needs shape validation.
        check_remote_scripts_probe_nvdec_early(strict=True, verbose=verbose)
        check_resume_from_state_dict_shape_compat(strict=True, verbose=verbose)

        # 2026-04-28: 2 more strict meta-bug checks (35, 36) from observed
        # patterns this session:
        # - tmux kill-server kills OTHER lanes' sessions (would cascade-fail
        #   shared-host runs; caught myself doing this in quick_setup)
        # - unconditional ensurepip crashes on PyTorch containers with newer
        #   pip than the bundled wheels (setup_full bug, just fixed)
        check_no_tmux_kill_server_in_lane_scripts(strict=True, verbose=verbose)
        check_no_unconditional_ensurepip(strict=True, verbose=verbose)

        # 2026-04-28 evening: 2 more checks (37, 38) from today's overnight wave.
        # macOS resource forks crash auth_eval; SSH no-timeout hangs parent agent.
        # Both STRICT after setup_full purge-once landed (Check 37 satisfied
        # via canonical bootstrap path); SSH check has 0 violations (no
        # script in repo uses ssh — it's all parent-agent invoked).
        check_lane_scripts_strip_macos_resource_forks(strict=True, verbose=verbose)
        check_ssh_commands_have_connect_timeout(strict=True, verbose=verbose)

        # 2026-04-28 late: Check 39 — undeployed archive-artifact producers.
        # CATCHES the recurring "code-shipped-never-deployed" failure mode:
        # tools that produce a registered submission artifact and have a
        # __main__ entry but no scripts/remote_lane_*.sh invocation. Lane EC
        # sat unused 2 weeks because of this exact gap. Lands at 0 live
        # violations after exemption pass for kaggle_kernels (alternative
        # platform), library helpers (scorer_targets.py), and 2 dead lanes
        # (mini_tto_inflate, optimize_embedding) — straight to STRICT per
        # the Lane A pattern. References:
        # - project_lane_ec_engineered_corrections_20260428
        # - project_outstanding_work_and_stacks_20260428 TIER 3
        check_undeployed_archive_artifact_producers(strict=True, verbose=verbose)

        # 2026-04-28 late: Check 40 — FP4 hardware-disclosure markers.
        # CATCHES the bug class that destroyed Lane F lineage: production
        # FP4 paths without hardware-capability disclosure. Lane F V1=2.73,
        # V2=1.79, V3=1.85 were all simulated FakeQuantFP4 in FP32 — 4090
        # is CC 8.9 and NVFP4 needs Blackwell CC 10.0, so "FP4 architectural
        # hostility" was unverifiable. Lands at 0 live violations after
        # adding `# FP4_HARDWARE_DISCLOSED:` markers to the 3 actual
        # production sites (fp4_quantize.py, profile_fp4_layer_sensitivity.py,
        # qat_finetune.py). Straight to STRICT — bug class structurally
        # extinct. Reference: project_cosmos_deep_dive_addendum_20260428.
        # Lane F-V5 (hardware FP8 via torchao.float8) is the proper rescue
        # path for Ada/Lovelace+ (CC >= 8.9) hardware.
        check_fp4_production_paths_disclose_hardware(strict=True, verbose=verbose)

        # 2026-04-28 evening: Check 41 — remote_lane_*.sh heartbeat loop.
        # CATCHES the silent-non-start failure mode that wasted ~$2.50 today
        # on instances 35739770/35739771/35739773 (Lane W Iceland, Lane K
        # Denmark, Lane OS-V2 NC). SSH + clone succeeded but lane script
        # never executed; no heartbeat.log on disk meant no readiness
        # verification possible. Lands at 0 live violations with sweep
        # orchestrators exempted. Reference:
        # feedback_vastai_launch_returns_success_before_lane_starts.
        check_remote_lane_scripts_have_heartbeat(strict=True, verbose=verbose)

        # 2026-04-28: Check 42 — pose-projection train/inference parity.
        # CATCHES the BUG-1 class from Lane M-V2 audit: pose-projection
        # helpers used at OPTIMIZATION time but NOT at INFLATE time produce
        # train/inference distribution mismatches. Lane M-V2 lost 5h GPU +
        # $1.50 to this exact bug (PoseNet 0.076 = 15× Lane A baseline was
        # signal of the bug, not the architectural premise). 0 live
        # violations after waivers (BUG-1 marked WAIVED until V3-clean
        # lands, scorer_exploits gradient projection marked WAIVED for
        # different domain). STRICT.
        # Reference: project_lane_m_v2_audit_council_findings_20260428.
        check_pose_projection_train_inference_parity(strict=True, verbose=verbose)

        # 2026-04-28 PM: Check 43 — launcher tarball must include lane anchors.
        # CATCHES the bug class where a tarball --exclude pattern wins over
        # a lane script's anchor reference. 3 lanes lost 2026-04-28 PM
        # because lane_a_landed/ was excluded but archive_lane_a.zip is the
        # canonical anchor. STRICT @ 0 violations after launcher fix landed.
        check_launcher_tarball_includes_lane_anchors(strict=True, verbose=verbose)

        # 2026-04-29 AM: Check 66 (no-git-reset-hard-in-lane-scripts).
        # `git reset --hard origin/main` in lane Stage-1 wipes local-only
        # anchor files (archive_lane_a.zip, baseline dirs) that the launcher
        # just SCP'd. 5/6 TIER-1 lanes crashed 2026-04-29 from this bug.
        # STRICT @ 0 violations after stripping pattern from all 11 scripts.
        check_no_git_reset_hard_in_remote_lane_scripts(strict=True, verbose=verbose)

        # 2026-04-29 AM: Check 67 (python-files-compile) + Check 68 (shell-syntax)
        # + Check 69 (anchor files exist locally).
        # PROACTIVE: catches SyntaxError + IndentationError + bash syntax bugs
        # + missing anchor files BEFORE they ship to remote and crash deploy.
        # User demand: "preflight needs to include a python compile step of all so
        # we can identify any python errors without deploying" + "autodetect and
        # permanently prevent all bugs possible to anticipate".
        # 631 .py files compile in ~0.75s; 109 shell scripts in ~0.45s; 72
        # anchor refs scanned in <0.1s. Total proactive cost: ~1.3s.
        # Check 69 caught 8 real bugs on first run (Lane F-V5 + Lane J-IMP +
        # Lane J-JBL all referenced non-existent lane_g_v3_landed/iter_0/).
        check_python_files_compile(strict=True, verbose=verbose)
        check_shell_scripts_syntax_clean(strict=True, verbose=verbose)
        check_lane_anchor_files_exist_locally(strict=True, verbose=verbose)

        # 2026-04-29: Check 43 — controlled-baseline methodology for new
        # Tuna-2 lanes. WARN-ONLY initially because it only applies to
        # remote_lane scripts added/modified after 2026-04-29 and is a
        # methodology guard, not a current correctness blocker.
        check_remote_lane_scripts_have_controlled_baseline(strict=False, verbose=verbose)

        # 2026-04-28 evening: 4 NEW meta-bug checks (44, 45, 46, 47) for
        # test-assertion strength + archive-size discipline. Ref Round 22
        # bit-STE sign bug post-mortem (4 review rounds dismissed it because
        # the only assertion was `grad is not None`).
        # Per-check live counts at wire-in time + promotion plan:
        #   Check 44 (gradient-direction-tests-exist)             0  → STRICT
        #   Check 45 (loss-convergence-tests)                     0  → STRICT
        #   Check 46 (quantizer-roundtrip-tests)                  0  → STRICT (R25 promotion: 5 test files added covering archive_codec/entropy_archive/mask_entropy_coder/network_codec/semantic_quantization; quantization_audit waived as drift-MEASUREMENT module not a quantizer)
        #   Check 47 (lane-archive-size-assertion)                0  → STRICT
        check_gradient_direction_tests_exist(strict=True, verbose=verbose)
        check_test_assertion_strength_for_loss_functions(strict=True, verbose=verbose)
        check_quantizer_modules_have_round_trip_test(strict=True, verbose=verbose)
        check_lane_deploy_scripts_have_archive_size_assertion(strict=True, verbose=verbose)

        # 2026-04-28 evening: 3 NEW meta-bug checks (48, 49, 50) from the
        # killed-lanes forensic audit. Reference:
        # project_killed_lanes_forensic_audit_20260428.
        # All 3 ship WARN-only initially because the live codebase has real
        # violations the user may want to fix incrementally:
        # - Check 48 (orphan-src-tac-modules): catches Lane V class — silent
        #   modules added but never wired into a profile / CLI / script.
        # - Check 49 (profile-loss-mode-allowlist-parity): catches Lane J-JBL
        #   class — profile loss_mode value not in train_renderer.py
        #   _VALID_LOSS_MODES allowlist. Live count: 2 (posenet_embedding,
        #   segnet_kl in profiles that may not actively dispatch through
        #   train_renderer.py). Promote to STRICT after audit + fix.
        # - Check 50 (deploy-script-profile-exists): catches typo'd or
        #   missing PROFILES registrations. Live count: 4 (one false-positive
        #   in a comment, two profiles needing registration). Promote to
        #   STRICT after lane script cleanup.
        check_no_orphan_src_tac_modules(strict=False, verbose=verbose)
        check_profile_loss_modes_in_validator_allowlist(strict=False, verbose=verbose)
        check_deploy_script_profiles_exist_in_registry(strict=False, verbose=verbose)

        # 2026-04-28 deep DX hardening pass 2: 3 NEW meta-bug checks
        # (51, 52, 53) for silent-swallow / unchecked-subprocess /
        # operator-discoverability. All ship WARN-only initially. Promote
        # to STRICT after one-time cleanup pass per the established
        # warn-only → strict pattern (see Lane A pattern in checks 1-11).
        # Reference: feedback_deep_hardening_pass_2_patterns_20260428.
        # - Check 51 (no-bare-except): catches `except:` and
        #   `except Exception: pass`. Same class as the
        #   tools/fleet_dashboard_live.py bug fixed in this pass.
        # - Check 52 (subprocess-run-checked): catches subprocess.run()
        #   without check=True or returncode check. Same class as the
        #   LANE-B silent-cascade trap (feedback_zip_dep_bootstrap_trap)
        #   but at the Python level.
        # - Check 53 (tools-have-argparse): operator-discoverability:
        #   tools/*.py + scripts/*.py with __main__ entry must wire
        #   argparse or click for --help.
        check_no_bare_except(strict=False, verbose=verbose)
        # 2026-04-28 deep hardening pass 3: Checks 52 + 53 promoted to STRICT
        # after one-time cleanup pass. Subprocess: 31 violations triaged into
        # 2 classes — wrappers/best-effort (24 waivers with concrete reason)
        # vs real bugs (7 fail-loud `check=True` adds for ffmpeg/ffprobe pipes
        # in hybrid_inflate, optimize_poses, train_distill, benchmark_codecs,
        # variable_rate). Argparse: 7 violations — 5 hook/dispatcher waivers
        # + 1 real argparse add (check_determinism.py) + 1 thin-shim waiver.
        # Bug classes structurally extinct.
        check_subprocess_run_checked(strict=True, verbose=verbose)
        check_tools_have_argparse(strict=True, verbose=verbose)

        # 2026-04-28 evening: 2 NEW STRICT meta-bug checks (54, 55) for the
        # canonical NVDEC workflow. Today wasted ~$10 on 87% NVDEC_BAD
        # Vast.ai 4090 hosts before the 2-layer fix landed:
        # - Layer 1 DETECTION (commit 58e55890): scripts/probe_nvdec.sh
        #   --lightweight at setup_full.sh Stage 0.5 catches ~95% of
        #   NVDEC-missing hosts BEFORE the 5-minute DALI install.
        # - Layer 2 ACTION (commit 5acebb88-ish): launch_lane_on_vastai.py
        #   phase2-launch Stage 2 polls setup.log + auto-destroys NVDEC_BAD.
        # Per the user mandate ("we need to automate and canonicalize and
        # permanently guard against NVDEC issue"), both layers are now
        # structurally extinct bug classes — any future refactor that
        # drops the poll OR re-orders the probe AFTER DALI fails preflight.
        # Both checks land at 0 live violations → straight to STRICT per
        # the Lane A pattern.
        # Reference: feedback_canonical_nvdec_workflow_GUARD_20260428.
        check_phase2_launch_polls_setup_log(strict=True, verbose=verbose)
        check_setup_full_probe_before_dali(strict=True, verbose=verbose)

        # 2026-04-28: Check 56 — verify_vast_instances.py auto-destroy
        # path must enforce BOTH IDLE-stale-minutes AND SETUP-stale-
        # minutes. Without the SETUP timer, a TRULY hung setup_full.sh
        # accrues cost forever (no heartbeat ever lands → IDLE timer
        # never fires). Reference: feedback_setup_stuck_cost_leak_FIXED_20260428.
        # Lands at 0 live violations → straight to STRICT.
        check_verify_vast_setup_stuck_dual_threshold(
            strict=True, verbose=verbose,
        )

        # 2026-04-29: Check 57 RETIRED. The pattern it required
        # (`git fetch origin main && git reset --hard origin/main`) caused
        # the 2026-04-29 5/6-TIER-1-lane crash by wiping local-only anchor
        # files SCP'd by the launcher. Replaced by Check 66 which PROHIBITS
        # the destructive pattern. The launcher tarball is now the canonical
        # parity mechanism. (memory: feedback_git_reset_nukes_anchors_20260429)
        # check_lane_scripts_use_canonical_git_sync(strict=True, verbose=verbose)

        # 2026-04-28 deep hardening pass 3 dimension 2: 4 NEW meta-bug
        # checks (58, 59, 60, 61). All ship STRICT initially because they
        # land at 0 live violations on the current codebase per the Lane A
        # pattern (commit 7d2b5299). Reference:
        # feedback_deep_hardening_pass_3_patterns_20260428.
        # - Check 58 (launcher-max-dph-floor): forbid hardcoded --max-dph
        #   below 0.40, which over-restricts the host pool and starves the
        #   search after NVDEC_BAD attrition (today wasted ~$10).
        # - Check 59 (phase2-extract-cleanup): cmd_phase2_extract MUST call
        #   destroy_instance() on CUDA-probe failure to stop cost accrual.
        # - Check 60 (memory-md-size): MEMORY.md > 250 lines silently
        #   truncates context loading. Today's session triggered the
        #   200-line warning; 250 gives a 50-line buffer.
        # - Check 61 (bootstrap-provenance): canonical bootstrap scripts
        #   MUST write provenance.json (git_hash + gpu_name) for post-mortem
        #   traceability per feedback_canonical_remote_bootstraps.
        check_launcher_max_dph_floor(strict=True, verbose=verbose)
        check_phase2_extract_destroys_on_failure(strict=True, verbose=verbose)
        # Check 60 ships warn-only because MEMORY.md is a user-controlled
        # file and the operator should fix it on their own cadence (this
        # session: 234 lines, under the 250 ceiling — currently 0 violations).
        check_memory_md_size_under_ceiling(strict=False, verbose=verbose)
        check_canonical_bootstraps_write_provenance(strict=True, verbose=verbose)

        # 2026-04-28 Codex F5 (5-finding adversarial review): every lane
        # script that calls contest_auth_eval MUST either use the canonical
        # experiments/contest_auth_eval.py module (which has the F5 guard
        # for missing config.env) OR check PYTHON_INFLATE=renderer locally.
        # Lane RM-d burned 1+ hour discovering the canonical inflate env
        # was missing on the remote tarball. Lands at 0 live violations
        # post-F5 fix → ships STRICT immediately.
        check_lane_scripts_set_up_inflate_environment(strict=True, verbose=verbose)

        # 2026-04-28 Check 64 — lane scripts must have a recent E2E smoke
        # proof. Closes the structural gap that cost Lane RM-d 3.5h GPU:
        # 63 STATIC preflight checks above all guard CODE PATTERNS, none
        # actually run the deploy → inflate → contest_auth_eval pipeline
        # locally. Check 64 enforces every remote_lane_*.sh has an entry
        # in .omx/state/lane_e2e_smoke_proofs.json that is < 7 days old,
        # written by experiments/canonical_local_auth_eval_smoke.py.
        # Promoted to STRICT after the backfill landed all 70 existing
        # lanes at 0 live violations. Reference:
        # feedback_canonical_e2e_smoke_PERMANENT_GUARD_20260428.
        check_lane_scripts_have_e2e_smoke_proof(strict=True, verbose=verbose)

        # 2026-04-28 PM: Check 65 — lane CLASSES (not just per-lane scripts)
        # must have at least one complete-pipeline proof on file. Closes the
        # Lane RM-d structural gap: new lane classes shipping without ever
        # demonstrating dispatch → train → archive → auth_eval cycle. Ships
        # WARN-ONLY initially so the existing 70 lanes have a backfill window;
        # promotion plan (Lane A pattern): backfill .omx/state/
        # lane_class_proofs.json, then flip strict=True. Reference:
        # feedback_artifact_recovery_canonical_workflow_20260428.
        check_lane_classes_have_pipeline_proof(strict=False, verbose=verbose)

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
    if path.is_dir() or not path.is_file():
        return violations
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
        if sh_path.is_dir():
            continue  # recovered_*.sh is a directory, not a script
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
# is flipped strict=True at its preflight_all() call site near the top of
# preflight_all (the previously-referenced TODO block was removed 2026-04-27
# after every meta-bug check was promoted).
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

# 2026-04-27 codex R5-4 #2: the previous scanner only matched static
# `from tac.scorer import ...` and call names ending with known loader
# function names. Live inflate scripts deliberately bypassed this with
# `importlib.import_module("tac.scorer")` + `getattr(mod, "load_scorers")`,
# producing a FALSE-clean strict scan while real scorer-at-inflate code
# remained env-gated in production. The scanner now AST-walks for:
#   1. importlib.import_module("tac.scorer*")
#   2. importlib.util.find_spec("tac.scorer*")
#   3. __import__("tac.scorer*")
#   4. getattr(<expr>, "load_scorers"|"load_posenet"|...)
# AND respects an explicit `# SCORER_AT_INFLATE_WAIVED:<reason>` comment
# marker (SAME LINE only — codex R5-r6 #1 tightened this from a 6-line
# lookback because nearby markers could waive unrelated calls). Waived
# violations are counted separately and surfaced to the operator so the
# gate cannot be silently bypassed — strict means "no UNWAIVED violations".
_DYNAMIC_IMPORT_FUNCS = (
    "importlib.import_module",
    "importlib.util.find_spec",
    "__import__",
)
_GETATTR_LOADER_NAMES = frozenset({
    "load_scorers", "load_posenet", "load_segnet",
    "load_differentiable_scorers", "load_posenet_targets",
    "extract_gt_pose_targets",
})
_SCORER_MODULE_PREFIX = "tac.scorer"  # matches tac.scorer, tac.scorer_targets, …
_WAIVER_MARKER = "SCORER_AT_INFLATE_WAIVED"
# 2026-04-27 codex R5-r6 #1 fix: lookback is now SAME-LINE ONLY.
# The previous 6-line lookback meant a marker intended for one specific
# pending-ruling import could waive an UNRELATED scorer load inserted
# nearby (or above, in the same try-block). The failure message even
# said "3 lines" while the constant was 6 — operators couldn't audit
# what a marker actually covered. Same-line policy:
#   - Marker MUST be in a comment on the SAME line as the offending call.
#   - For multi-line statements (e.g., a getattr(...) split across lines),
#     each call on each line needs its own same-line marker because the
#     AST records lineno per-call.
#   - The legacy `# noqa: scorer-at-inflate` form is also recognised, but
#     ONLY on the same line.
# Same-line enforcement is the only policy that is auditable without a
# walker — every waiver is structurally attached to the specific call
# being waived. Block-style waivers (a marker comment above a try-block)
# are no longer recognised by the scanner; existing block markers must be
# moved onto each offending call line.
_WAIVER_LOOKBACK_LINES = 0  # SAME-LINE ONLY (was 6 → bug → fixed in R5-r6 #1)


def _line_is_waived(lines: list[str], lineno: int) -> bool:
    """Return True if `lineno` (1-based) carries an explicit waiver marker
    on the SAME line.

    A waiver is recognised only on the same line as the offending call
    (codex R5-r6 #1). The marker must appear inside a comment (`#`-anywhere
    on that line is fine; we never match inside a string literal because
    we only scan the post-# segment). We also accept the legacy
    `# noqa: scorer-at-inflate (...)` form so existing inflate scripts
    keep working — but only when it's on the same line.
    """
    if lineno <= 0 or lineno > len(lines):
        return False
    # Same-line only: do NOT walk preceding lines.
    line = lines[lineno - 1]
    if "#" not in line:
        return False
    comment = line[line.index("#"):]
    if _WAIVER_MARKER in comment:
        return True
    if "noqa: scorer-at-inflate" in comment:
        return True
    return False


def _string_constant_arg(call: ast.Call) -> str | None:
    """Return the first positional arg of `call` if it is a string literal,
    else None. Handles `import_module("tac.scorer")` and `getattr(m, "x")`
    forms — for getattr we want the SECOND arg (index 1), so callers
    pass `call.args[idx]` instead."""
    if not call.args:
        return None
    a = call.args[0]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value
    return None


def _scan_inflate_for_scorer_load(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect scorer-load patterns in inflate*.py files. Returns UNWAIVED
    violations only — waived hits are reported separately by the caller
    (see `_scan_inflate_for_scorer_load_with_waivers`)."""
    unwaived, _waived = _scan_inflate_for_scorer_load_with_waivers(path, repo_root)
    return unwaived


def _scan_inflate_for_scorer_load_with_waivers(
    path: Path, repo_root: Path,
) -> tuple[list[str], list[str]]:
    """Detect scorer-load patterns in inflate*.py files.

    Returns (unwaived_violations, waived_violations). Unwaived violations
    fail strict-mode preflight; waived violations are surfaced to the
    operator so the count of pending-ruling waivers is visible.
    """
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
            return [], []
        tree = None

    unwaived: list[str] = []
    waived: list[str] = []
    lines = text.splitlines()

    def _record(lineno: int, msg: str) -> None:
        full = f"{rel}:{lineno}: {msg}"
        if _line_is_waived(lines, lineno):
            waived.append(full)
        else:
            unwaived.append(full)

    if tree is not None:
        # AST walk: detect static imports, dynamic imports, getattr-loaders,
        # and direct loader-call patterns.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if (_SCORER_MODULE_PREFIX in node.module
                        or node.module == _SCORER_MODULE_PREFIX):
                    for alias in node.names:
                        _record(
                            node.lineno,
                            f"imports {alias.name!r} from {node.module} at "
                            f"inflate time. Strict scorer rule (CLAUDE.md "
                            f"feedback_strict_scorer_rule): NO scorer load "
                            f"at inflate; ~73MB destroys the rate term.",
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name and _SCORER_MODULE_PREFIX in alias.name:
                        _record(
                            node.lineno,
                            f"imports module {alias.name!r} at inflate time. "
                            f"Strict scorer rule (feedback_strict_scorer_rule).",
                        )
            if isinstance(node, ast.Call):
                func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""

                # 1. Direct loader call (load_scorers, load_posenet, …).
                for name in ("load_scorers", "load_posenet", "load_segnet",
                             "load_differentiable_scorers"):
                    if func_str.endswith(name) and func_str != f"_{name}":
                        # Skip the helper-name shadows like `_resolve_scorers`
                        # (renamed in inflate_postfilter.py to avoid the
                        # static endswith match — those are now caught via
                        # the getattr path below).
                        _record(
                            node.lineno,
                            f"calls `{func_str}(...)` at inflate time. "
                            f"Strict scorer rule (feedback_strict_scorer_rule): "
                            f"NO scorer load at inflate.",
                        )
                        break

                # 2. Dynamic imports: importlib.import_module("tac.scorer*"),
                #    importlib.util.find_spec("tac.scorer*"),
                #    __import__("tac.scorer*").
                if (func_str.endswith("import_module")
                        or func_str.endswith("find_spec")
                        or func_str == "__import__"
                        or func_str.endswith(".__import__")):
                    s = _string_constant_arg(node)
                    if s and _SCORER_MODULE_PREFIX in s:
                        _record(
                            node.lineno,
                            f"dynamic import `{func_str}({s!r})` at inflate "
                            f"time. Strict scorer rule "
                            f"(feedback_strict_scorer_rule). Add an explicit "
                            f"`# {_WAIVER_MARKER}:<reason>` marker if this is "
                            f"an env-gated pending-ruling path.",
                        )

                # 3. getattr(<x>, "load_scorers"|"load_posenet"|...) — the
                #    canonical companion to importlib.import_module that
                #    used to slip through the scanner.
                if (func_str == "getattr" or func_str.endswith(".getattr")):
                    if len(node.args) >= 2 and isinstance(
                        node.args[1], ast.Constant,
                    ) and isinstance(node.args[1].value, str):
                        attr = node.args[1].value
                        if attr in _GETATTR_LOADER_NAMES:
                            _record(
                                node.lineno,
                                f"getattr(..., {attr!r}) at inflate time "
                                f"resolves a scorer loader. Strict scorer rule "
                                f"(feedback_strict_scorer_rule). Add "
                                f"`# {_WAIVER_MARKER}:<reason>` if env-gated.",
                            )
    else:
        # Shell file fallback: any line mentioning posenet.bin / segnet.bin
        # / safetensors.load near scorer keywords.
        for i, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if "scorer" in low and ("load" in low or "import" in low):
                msg = (
                    f"{rel}:{i}: shell line references scorer load at "
                    f"inflate time (CLAUDE.md feedback_strict_scorer_rule)."
                )
                if _line_is_waived(lines, i):
                    waived.append(msg)
                else:
                    unwaived.append(msg)
    return unwaived, waived


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
    Returns list of UNWAIVED violations. Raises MetaBugViolation if strict
    and any unwaived hits remain. Waived hits (those marked with
    `# SCORER_AT_INFLATE_WAIVED:<reason>`) are surfaced in verbose output
    but do NOT block strict mode — operators can see exactly how many
    pending-ruling paths exist.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    waived: list[str] = []
    n_scanned = 0
    submissions = root / "submissions"
    if submissions.exists():
        for sub_dir in sorted(submissions.iterdir()):
            if not sub_dir.is_dir():
                continue
            for p in sorted(sub_dir.glob("inflate*.py")):
                n_scanned += 1
                u, w = _scan_inflate_for_scorer_load_with_waivers(p, root)
                violations.extend(u)
                waived.extend(w)
            for p in sorted(sub_dir.glob("inflate*.sh")):
                n_scanned += 1
                u, w = _scan_inflate_for_scorer_load_with_waivers(p, root)
                violations.extend(u)
                waived.extend(w)

    if verbose and violations:
        print(f"  [no-scorer-at-inflate] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-scorer-at-inflate] OK: {n_scanned} inflate files scanned")
    if verbose and waived:
        print(
            f"  [no-scorer-at-inflate] {len(waived)} WAIVED hit(s) "
            f"(env-gated, pending-ruling — visible to operator):"
        )
        for v in waived:
            print(f"    ◇ {v}")

    if violations and strict:
        raise MetaBugViolation(
            "SCORER LOAD AT INFLATE violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nNo scorer at inflate time (feedback_strict_scorer_rule). "
            + "If this is an env-gated pending-ruling path, add an explicit "
            + f"`# {_WAIVER_MARKER}:<reason>` comment marker on the SAME "
            + "line as the offending call (codex R5-r6 #1: block-level / "
            + "lookback markers are no longer recognised — every waiver must "
            + "be structurally attached to its specific call site)."
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


_PACK_APPROVED_FIXTURE_MARKER = "PACK_APPROVED_FIXTURE_OK"


def _is_test_or_fixture_path(rel: Path) -> bool:
    """Return True if `rel` (path relative to repo root) is a test or
    pytest-conftest file ANYWHERE in the tree.

    2026-04-27 codex R5-4 #3: previously the exemption for fixtures was
    `src/tac/tests/test_*.py` only. With the strict scanner now scanning
    `experiments/`, `scripts/`, and `tools/`, a legitimate
    integration-test fixture under any of those dirs that constructs an
    approved blob with the internal promotion token would block strict
    preflight. We now recognise tests broadly:
      • basename matches test_*.py / *_test.py
      • basename is conftest.py
      • path contains a `/tests/` segment
    """
    name = rel.name
    if name == "conftest.py":
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    parts = rel.parts
    if "tests" in parts or "test" in parts:
        return True
    return False


def _file_has_pack_approved_fixture_marker(path: Path) -> bool:
    """Return True if `path` contains a `# PACK_APPROVED_FIXTURE_OK` comment
    anywhere — the explicit waiver mechanism for legitimate fixtures that
    don't live in a recognised test directory.
    """
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return False
    for line in text.splitlines():
        if "#" not in line:
            continue
        if _PACK_APPROVED_FIXTURE_MARKER in line[line.index("#"):]:
            return True
    return False


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
    catches the same bug class at preflight time.

    Test fixtures that construct approved blobs (with the internal token)
    are permitted via two complementary mechanisms:
      1. Path-based: any `test_*.py` / `*_test.py` / `conftest.py` file,
         or any path containing a `/tests/` segment, is exempt (broader
         than the previous `src/tac/tests/` only filter — codex R5-4 #3).
      2. Marker-based: any file containing
         `# PACK_APPROVED_FIXTURE_OK` (anywhere) is exempt — explicit
         operator waiver for fixtures outside the standard test layout.

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
        # Exempt any test file (broad detection — codex R5-4 #3).
        if _is_test_or_fixture_path(rel):
            continue
        # Exempt any file with the explicit waiver marker.
        if _file_has_pack_approved_fixture_marker(py):
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
    # Refinement (2026-04-27):
    #   - `nvidia-smi --query-gpu=...` is a 100ms info read, NOT GPU spend.
    #   - Descriptive log/echo/printf lines (e.g. `log "=== Stage 3:
    #     evaluate.py ..."`) frequently mention marker names without
    #     actually invoking them. Exempt these.
    gpu_work_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        # Skip comment-only lines for marker detection.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Exempt `nvidia-smi --query-...` info queries (sub-100ms, no spend).
        if "nvidia-smi --query-" in line:
            continue
        # Exempt descriptive log/echo/printf lines (operator-readable text
        # mentioning marker names doesn't run them).
        first_token = stripped.split(None, 1)[0] if stripped else ""
        if first_token in ("log", "echo", "printf"):
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


# ─────────────────────────────────────────────────────────────────────────
# 2026-04-27 codex R5-r6: 5 new preflight checks for the round-6 findings.
# Each check guards against a regression of the matching finding fix:
#
#   A. check_no_brittle_six_line_waiver_lookback  — Finding #1 (waiver)
#   B. check_kl_distill_uses_roundtripped_frames   — Finding #2 (KL roundtrip)
#   C. check_eval_roundtrip_gate_called_after_output_dir_resolution
#                                                  — Finding #3 (gate ordering)
#   D. check_nvdec_probe_has_error_classification  — Finding #4 (probe)
#   E. check_archive_builders_use_deterministic_zip — Finding #5 (det. zip)
#
# All wired warn-only initially in preflight_all() (per the established
# Lane A → strict promotion pattern); flip to strict=True once live counts
# are zero and codex has signed off.
# ─────────────────────────────────────────────────────────────────────────


# ── Check A: waiver lookback must NOT exceed 1 line ──────────────────────────
def check_no_brittle_six_line_waiver_lookback(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Guard Finding #1: scanner waiver lookback constant must be 0 or 1.

    The previous lookback was 6 lines, which let a waiver intended for one
    pending-ruling import suppress an UNRELATED scorer load inserted
    nearby. The auditable fix is same-line-only (lookback 0). This check
    inspects `_WAIVER_LOOKBACK_LINES` in `src/tac/preflight.py` (this
    file) and refuses anything > 1.
    """
    root = repo_root or REPO_ROOT
    pf = root / "src" / "tac" / "preflight.py"
    violations: list[str] = []
    if not pf.exists():
        return violations
    text = pf.read_text()
    # Extract the `_WAIVER_LOOKBACK_LINES = N` literal via simple regex.
    m = re.search(
        r"_WAIVER_LOOKBACK_LINES\s*=\s*(\d+)", text,
    )
    if m is None:
        violations.append(
            f"{pf.relative_to(root)}: missing `_WAIVER_LOOKBACK_LINES` "
            f"constant (the waiver-lookback scanner can no longer be audited)."
        )
    else:
        n = int(m.group(1))
        if n > 1:
            violations.append(
                f"{pf.relative_to(root)}: _WAIVER_LOOKBACK_LINES = {n} "
                f"(must be 0 or 1 per codex R5-r6 #1; the previous 6-line "
                f"lookback let unrelated nearby loads ride a single waiver)."
            )

    if verbose and violations:
        print(
            f"  [waiver-lookback] {len(violations)} violation(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [waiver-lookback] OK")

    if violations and strict:
        raise MetaBugViolation(
            "WAIVER LOOKBACK violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check B: kl_distill_segnet_only must NOT receive raw renderer pairs ─────
_KL_DISTILL_FORBIDDEN_FIRST_ARGS = frozenset({"pairs", "rendered_pair", "rendered_pair_hwc"})


def _scan_python_for_kl_distill_raw_pairs(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect call sites of kl_distill_segnet_only(...) whose FIRST positional
    arg is a raw renderer-output variable (one of `pairs`, `rendered_pair`,
    `rendered_pair_hwc`). The contract requires the same eval-roundtripped
    frames the SegNet scoring path consumes (codex R5-r6 #2).

    The check is intentionally STRICT on naming — the in-repo recipe is
    `rendered_pair_hwc_rt` (or any name with `_rt` / `roundtripped` in it).
    Add a `# KL_RAW_PAIRS_OK:<reason>` marker on the call line if the
    raw pairs are intentional (e.g., a unit test verifying the contract).
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not func_str.endswith("kl_distill_segnet_only"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        # Extract simple name; skip complex expressions (which already permute
        # / view → presumed roundtripped).
        if not isinstance(first, ast.Name):
            continue
        if first.id not in _KL_DISTILL_FORBIDDEN_FIRST_ARGS:
            continue
        # Same-line waiver opt-out.
        ln = node.lineno
        if 0 < ln <= len(lines):
            comment_idx = lines[ln - 1].find("#")
            if comment_idx >= 0 and "KL_RAW_PAIRS_OK" in lines[ln - 1][comment_idx:]:
                continue
        violations.append(
            f"{rel}:{node.lineno}: `kl_distill_segnet_only({first.id}, ...)` "
            f"passes raw renderer output to the KL helper. Codex R5-r6 #2: "
            f"feed the SAME eval-roundtripped frames the SegNet scoring "
            f"path consumes (typical name: `rendered_pair_hwc_rt`). If "
            f"intentional, add `# KL_RAW_PAIRS_OK:<reason>` on this line."
        )
    return violations


def check_kl_distill_uses_roundtripped_frames(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Guard Finding #2: KL distillation must operate on roundtripped frames.

    Live failure mode: optimize_poses.py passed `pairs` (raw renderer
    output) to `kl_distill_segnet_only(...)`, while the SegNet scoring
    path used `simulate_eval_roundtrip(frames_chw, ...)` first. Lane G
    KL gradients pulled the renderer in the wrong direction relative to
    the scored loss path.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for sub in ("experiments", "src/tac/experiments"):
        d = root / sub
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            # Skip __pycache__, tests live in src/tac/tests not here.
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_python_for_kl_distill_raw_pairs(p, root))

    if verbose and violations:
        print(
            f"  [kl-roundtrip] {len(violations)} violation(s) across "
            f"{n_scanned} script(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [kl-roundtrip] OK: {n_scanned} script(s) scanned")

    if violations and strict:
        raise MetaBugViolation(
            "KL DISTILL ROUNDTRIP violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nFeed the SegNet path's simulate_eval_roundtrip output, "
            + "not raw renderer pairs (codex R5-r6 #2)."
        )
    return violations


# ── Check C: _enforce_eval_roundtrip(args) must follow output_dir resolution ─
def _scan_python_for_gate_before_output_dir(
    path: Path, repo_root: Path,
) -> list[str]:
    """Find scripts where `_enforce_eval_roundtrip(args)` is called BEFORE
    any line that writes to `args.output_dir = ...` or first reads
    `args.output_dir` (codex R5-r6 #3).

    Heuristic: scan for the first line that calls
    `_enforce_eval_roundtrip(args)` AND the first line that ASSIGNS
    `args.output_dir = ...` (which means the script is computing a default
    output dir at runtime). If the gate call comes first, the sidecar
    write is dropped (output_dir is None at gate time).

    Files that never assign args.output_dir (CLI default suffices) pass
    trivially.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()
    gate_lineno: int | None = None
    output_dir_assign_lineno: int | None = None
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Match a CALL of _enforce_eval_roundtrip(args), not its def.
        if "_enforce_eval_roundtrip(args" in line and not stripped.startswith("def "):
            if gate_lineno is None:
                gate_lineno = i
        # Match `args.output_dir = ...` assignments (default-resolution).
        # Use a simple substring; precise AST walk would be overkill here.
        if "args.output_dir = " in line or "args.output_dir=" in line:
            if output_dir_assign_lineno is None:
                output_dir_assign_lineno = i
    if gate_lineno is None or output_dir_assign_lineno is None:
        return []
    if gate_lineno < output_dir_assign_lineno:
        return [
            f"{rel}:{gate_lineno}: `_enforce_eval_roundtrip(args)` called "
            f"BEFORE `args.output_dir` resolution at line "
            f"{output_dir_assign_lineno}. Sidecar JSON will land at None / be "
            f"silently dropped. Move the gate call AFTER output_dir "
            f"resolution (codex R5-r6 #3)."
        ]
    return []


def check_eval_roundtrip_gate_called_after_output_dir_resolution(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Guard Finding #3: gate call must follow `args.output_dir = ...`."""
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for sub in ("experiments", "src/tac/experiments"):
        d = root / sub
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_python_for_gate_before_output_dir(p, root))

    if verbose and violations:
        print(
            f"  [gate-ordering] {len(violations)} violation(s) across "
            f"{n_scanned} script(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [gate-ordering] OK: {n_scanned} script(s) scanned")

    if violations and strict:
        raise MetaBugViolation(
            "GATE ORDERING violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nMove the _enforce_eval_roundtrip(args) call AFTER any "
            + "args.output_dir = ... default-resolution (codex R5-r6 #3)."
        )
    return violations


# ── Check D: NVDEC probe must have error classification ─────────────────────
def check_nvdec_probe_has_error_classification(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Guard Finding #4: probe_nvdec.sh must classify failures, not exit-2-all.

    Insists on the presence of the PROBE_CLASSIFICATION marker AND at least
    2 distinct exit codes for non-OK paths (so a fixture/dependency error
    cannot be misclassified as a missing-NVDEC host).
    """
    root = repo_root or REPO_ROOT
    probe = root / "scripts" / "probe_nvdec.sh"
    violations: list[str] = []
    if not probe.exists():
        violations.append(
            "scripts/probe_nvdec.sh: missing — no NVDEC probe at all. "
            "Restore the file (feedback_vastai_nvdec_host_variation)."
        )
    else:
        text = probe.read_text()
        if "PROBE_CLASSIFICATION:" not in text:
            violations.append(
                "scripts/probe_nvdec.sh: missing PROBE_CLASSIFICATION marker. "
                "Codex R5-r6 #4: the probe must print a classification token "
                "so bash can dispatch on NVDEC vs DALI vs FIXTURE failure."
            )
        # Look for at least 2 distinct exit codes besides 0 and 1 (1 == DALI
        # missing). Specifically expect 2 (NVDEC), 3 (DALI build), 4
        # (fixture), 5 (unknown). Settle for any 3 distinct from {2,3,4,5}.
        exits = set()
        for m in re.finditer(r"\bexit\s+([0-9]+)\b", text):
            n = int(m.group(1))
            if n in (2, 3, 4, 5):
                exits.add(n)
        if len(exits) < 2:
            violations.append(
                f"scripts/probe_nvdec.sh: only {len(exits)} distinct "
                f"non-NVDEC exit codes found (need >= 2). Add separate "
                f"exit codes for FIXTURE / DALI_BUILD / UNKNOWN (codex "
                f"R5-r6 #4)."
            )

    if verbose and violations:
        print(
            f"  [probe-classification] {len(violations)} violation(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [probe-classification] OK")

    if violations and strict:
        raise MetaBugViolation(
            "NVDEC PROBE CLASSIFICATION violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check E: archive builders must use deterministic zip ────────────────────
_DET_ZIP_OPT_OUT = "DETERMINISTIC_ZIP_OK"
_DET_ZIP_HINT_FNS = ("_deterministic_zip_write", "writestr", "ZipInfo")


def _scan_python_for_nondeterministic_zip(
    path: Path, repo_root: Path,
) -> list[str]:
    """Find archive builders that call `ZipFile.write(...)` without a
    deterministic-zip helper (codex R5-r6 #5). Files with the explicit
    `# DETERMINISTIC_ZIP_OK` marker or a wrapper helper opt out."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if _DET_ZIP_OPT_OUT in text:
        return []
    # If the file uses the deterministic helper OR uses ZipInfo+writestr
    # AT LEAST ONCE alongside any .write() calls, consider it OK.
    has_helper_or_zipinfo = any(h in text for h in _DET_ZIP_HINT_FNS)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        # Match `<x>.write(<path>, arcname=...)`-style calls, not just
        # `<x>.write(<path>)` since the latter is also bad. The signature
        # is `ZipFile.write(filename, arcname=None, compress_type=None, ...)`,
        # so the FIRST positional arg is a path-like (str or Path).
        if not func_str.endswith(".write"):
            continue
        # Only flag if this call is inside a `with ZipFile(...) as <x>` and
        # the receiver matches. Approximate: look for `zipfile.ZipFile`
        # imported in file. Skip otherwise.
        if "ZipFile" not in text:
            continue
        if has_helper_or_zipinfo:
            continue
        violations.append(
            f"{rel}:{node.lineno}: `{func_str}(...)` inside a ZipFile "
            f"context — non-deterministic (embeds source mtime + perm bits). "
            f"Codex R5-r6 #5: use a fixed-timestamp ZipInfo + writestr() "
            f"OR add `# DETERMINISTIC_ZIP_OK` marker if intentional."
        )
    return violations


def check_archive_builders_use_deterministic_zip(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Guard Finding #5: archive-build scripts produce byte-identical zips."""
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    # Cover experiments/*build*.py + experiments/results/lane_*_*/build*.py
    candidates: list[Path] = []
    if (root / "experiments").exists():
        candidates.extend(sorted((root / "experiments").rglob("build*.py")))
        candidates.extend(sorted((root / "experiments").rglob("*build_archive*.py")))
    # Dedupe
    candidates = sorted({p for p in candidates if p.is_file()})
    for p in candidates:
        if "__pycache__" in p.parts:
            continue
        n_scanned += 1
        violations.extend(_scan_python_for_nondeterministic_zip(p, root))

    if verbose and violations:
        print(
            f"  [det-zip] {len(violations)} violation(s) across "
            f"{n_scanned} archive-build script(s):"
        )
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [det-zip] OK: {n_scanned} archive-build script(s) scanned")

    if violations and strict:
        raise MetaBugViolation(
            "DETERMINISTIC ZIP violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nUse fixed-timestamp ZipInfo + writestr (codex R5-r6 #5)."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# ADDITIVE META-BUG SECTION (2026-04-27, post-R5-r6)
# ════════════════════════════════════════════════════════════════════════════
#
# 12 new static-detectable preflight checks for meta-bug classes that have
# bitten this project but were NOT covered by checks 1-18 (existing meta-bug
# section + R5-r6 codex-fix subagent additions).
#
# These checks live in their own additive section to avoid merge conflict with
# past codex-fix subagents that edited checks 14-18. As of 2026-04-27 they are
# all wired into preflight_all() at strict=True (call sites near the top of
# preflight_all). New additive checks should land here too, then promoted.
#
# Pattern → memory entry mapping:
#   A. vastai-create-no-label                → orphan-instance prevention (today)
#   B. vastai-create-no-tracker              → cost-tracker registration
#   C. subagent-prompt-allows-cpu-fallback   → CLAUDE.md device-required rule
#   D. score-without-cuda-tag                → CLAUDE.md auth-eval-everywhere
#   E. waiver-marker-no-env-gate-name        → strict-scorer-rule auditability
#   F. half-frame-archive-without-trained    → feedback_half_frame_breaks_posenet
#   G. profile-key-no-resolver-bidirectional → extends dead-resolver scanner
#   H. inflate-scorer-load-no-runtime-banner → CLAUDE.md strict-scorer-rule
#   I. test-files-broken-imports             → test-coverage hygiene
#   J. subagent-prompt-no-cost-cap           → feedback_vastai_cost_paranoia
#   K. uniward-delta-no-attestation-flag     → Lane C R5 attestation gate
#   L. remote-script-no-provenance-write     → canonical pipeline standard
#
# All twelve start strict=False and must be promoted manually after the live
# violation count is verified clean (per the established Lane A → strict
# promotion pattern documented in commit 7f2740e4).


# ── Check A: Vast.ai `create instance` invocation must include --label ───────


_VASTAI_CREATE_INSTANCE_RE = re.compile(
    r'["\']create["\']\s*,\s*["\']instance["\']'
)


def _scan_python_for_vastai_create_no_label(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect `vastai create instance` invocations missing `--label`.

    The Vast.ai web console + show_instances output identify hosts by label.
    Orphan instances (no label) cannot be killed in bulk, cannot be matched
    to an experiment, and accrue cost silently. Today's incident: instance
    35707822 ran for ~$0.05 unidentifiable.

    Looks for `["create", "instance", ...]` arg list (canonical CLI form
    used by `client.py` and `check_vastai.py`) and checks whether the same
    arg list also contains `"--label"`.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        # Pull the literal-string elements.
        strs = [
            elt.value for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if "create" not in strs or "instance" not in strs:
            continue
        # Confirm the order is "create" then "instance" — these are the
        # vastai positional args. We only flag the literal CLI pattern,
        # not "create instance" as separate words used elsewhere.
        try:
            ci = strs.index("create")
            if strs[ci + 1] != "instance":
                continue
        except (IndexError, ValueError):
            continue
        if "--label" not in strs:
            violations.append(
                f"{rel}:{node.lineno}: `vastai create instance` invocation "
                f"missing `--label`. Orphan instances cannot be matched to "
                f"experiments → silent cost accrual (incident 2026-04-27, "
                f"$0.05). Add `'--label', f'lane-X-{{experiment.name}}'` to "
                f"the arg list."
            )
    return violations


def check_vastai_create_has_label(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every `vastai create instance` call must pass `--label`.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for path in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_vastai_create_no_label(path, root))
    if verbose:
        if violations:
            print(f"  [vastai-label] {len(violations)} unlabeled instance create(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [vastai-label] OK: {n_scanned} python file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "VASTAI CREATE INSTANCE WITHOUT --label:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check B: Vast.ai create-instance must register to active-instance tracker


_VASTAI_TRACKER_PATH = ".omx/state/vastai_active_instances.json"


def _scan_python_for_vastai_create_no_tracker(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect `vastai create instance` not followed by tracker write.

    We look for the canonical `["create", "instance", ...]` arg list, then
    scan the next ~30 lines for either:
      - a literal mention of `vastai_active_instances` (any form), OR
      - a function-name match like `register_active_instance(`,
        `track_instance(`, `_record_instance(`.

    The tracker exists so a separate cleanup script can detect orphans
    even when the main launch process dies (e.g. user Ctrl-C between
    create + setup). Without it, we have no audit trail of what was
    spawned by what script.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        strs = [
            elt.value for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if "create" not in strs or "instance" not in strs:
            continue
        try:
            ci = strs.index("create")
            if strs[ci + 1] != "instance":
                continue
        except (IndexError, ValueError):
            continue

        # Scan from this line forward to end-of-file for tracker hooks.
        # Rationale (R6 refinement, 2026-04-27): waiting for the instance
        # ID frequently takes 30-90 lines (poll loop for actual_status
        # ==running, then SSH info). Restricting to a 30-line window
        # produced false positives in the canonical launch paths
        # (scripts/check_vastai.py, src/tac/deploy/vastai/client.py)
        # where the tracker call is wired correctly but appears far
        # below the `create instance` arg list. The hard rule we care
        # about: SOMEWHERE in the same function body, a tracker write
        # must occur. End-of-file is a safe over-approximation; the
        # call sites are short (~600 lines max) and only one `create
        # instance` per file in practice.
        start = node.lineno
        window = "\n".join(lines[start - 1:])
        if (
            "vastai_active_instances" in window
            or "register_active_instance" in window
            or "register_instance(" in window
            or "track_instance(" in window
            or "_record_instance(" in window
        ):
            continue  # tracker hook present
        violations.append(
            f"{rel}:{node.lineno}: `vastai create instance` not followed by "
            f"a tracker write anywhere in the file. Add a call to "
            f"`tac.vastai_tracker.register_instance(...)` so a cleanup "
            f"script can detect orphans. (Tracker file: "
            f"{_VASTAI_TRACKER_PATH}.)"
        )
    return violations


def check_vastai_create_writes_tracker(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every Vast.ai launch must register the instance ID to a tracker file.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for path in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_vastai_create_no_tracker(path, root))
    if verbose:
        if violations:
            print(f"  [vastai-tracker] {len(violations)} untracked launch(es):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [vastai-tracker] OK: {n_scanned} python file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "VASTAI CREATE INSTANCE WITHOUT TRACKER REGISTRATION:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check C: Subagent prompts allowing `--device cpu` fallback ───────────────


_DEVICE_CPU_FALLBACK_RE = re.compile(
    r"--device\s+cpu",
    re.IGNORECASE,
)
_DETERMINISTIC_BYTES_OK_RE = re.compile(
    r"deterministic[-_ ]bytes acceptable|byte[-_ ]match[ \w]*N/A|cpu fallback approved",
    re.IGNORECASE,
)


def _scan_for_cpu_fallback_in_subagent_prompts(
    path: Path, repo_root: Path,
) -> list[str]:
    """Find subagent-prompt files mentioning `--device cpu` without caveat.

    A subagent dispatch prompt that says "use --device cpu if CUDA fails"
    can produce non-byte-matching archive bytes. Today's Lane H CRF56 task
    hit this — caught at review, no real cost, but a permanent gate is
    structurally cheaper than catching it again.

    Path filter: .md files under .agents/ and prompts/, plus Python literal
    strings invoking `Agent(...)` with prompt= containing the phrase.
    Caveat regex `_DETERMINISTIC_BYTES_OK_RE` allows the phrase if the
    same file (or same paragraph, approximated by 5-line window) explicitly
    permits non-deterministic bytes.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    # Skip preflight + tests + this very file.
    if (
        "/tests/" in rel_s
        or "preflight.py" in rel_s
        or "test_" in path.name
        or rel_s.endswith("CLAUDE.md")
    ):
        return []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if not _DEVICE_CPU_FALLBACK_RE.search(text):
        return []

    violations: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if not _DEVICE_CPU_FALLBACK_RE.search(line):
            continue
        # Look for caveat in surrounding 5-line window.
        window_start = max(0, i - 5)
        window_end = min(len(lines), i + 5)
        window = "\n".join(lines[window_start:window_end])
        if _DETERMINISTIC_BYTES_OK_RE.search(window):
            continue
        violations.append(
            f"{rel}:{i}: `--device cpu` mention without "
            f"'deterministic-bytes acceptable' caveat in 5-line window. "
            f"CPU fallback in a byte-deterministic build path produces "
            f"non-matching archive bytes (CLAUDE.md FORBIDDEN PATTERNS). "
            f"Add the caveat or remove the cpu fallback."
        )
    return violations


def check_subagent_prompts_no_cpu_fallback(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Subagent prompts must not allow `--device cpu` without caveat.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    # Scan .agents/, prompts/, and src/tac/agents/ if it exists.
    scan_dirs = [".agents", "prompts", "src/tac"]
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for ext in ("*.md", "*.py"):
            for p in d_path.rglob(ext):
                if "__pycache__" in p.parts:
                    continue
                n_scanned += 1
                violations.extend(
                    _scan_for_cpu_fallback_in_subagent_prompts(p, root)
                )
    if verbose:
        if violations:
            print(f"  [cpu-fallback] {len(violations)} unguarded cpu-fallback prompt(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [cpu-fallback] OK: {n_scanned} prompt/source file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "SUBAGENT PROMPT ALLOWS --device cpu WITHOUT CAVEAT:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check D: Numeric scores in run_log/findings without lane tag ─────────────


_SCORE_LANE_TAGS = (
    "[contest-CUDA]", "[advisory only]", "[MPS-PROXY]",
    "[contest-compliant]", "[unlimited-compute]",
    "[scorer-at-inflate-noncompliant]", "[CUDA-PROXY]",
)
_SCORE_LINE_RE = re.compile(
    r"\b(?:auth|score|total)\s*[=:]\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE,
)


def _scan_doc_for_untagged_scores(path: Path, repo_root: Path) -> list[str]:
    """Find lines like 'auth = 0.36' lacking a lane tag.

    CLAUDE.md non-negotiable: every numeric score MUST carry a lane tag so
    operators can never confuse contest-CUDA truth with proxy/MPS noise.
    Today's run_log has 9 score lines, only 1 tagged (audit done).
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    violations: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not _SCORE_LINE_RE.search(line):
            continue
        # Skip lines that look like math / formulas (e.g. "score = 100*seg + ...")
        if any(op in line for op in ("100*", "sqrt", "* seg", "(seg")):
            continue
        # Skip lines describing the scoring formula itself.
        if "formula" in line.lower() or "scoring" in line.lower()[:20]:
            continue
        if any(tag in line for tag in _SCORE_LANE_TAGS):
            continue
        # Allow [N.NN-N.NN] range expressions that are obviously projections.
        if "projection" in line.lower() or "projected" in line.lower():
            continue
        violations.append(
            f"{rel}:{i}: numeric score without lane tag. "
            f"Add one of {_SCORE_LANE_TAGS} to the same line. "
            f"(CLAUDE.md non-negotiable, MPS-CUDA drift = 23x.)"
        )
    return violations


def check_scores_have_lane_tag(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every numeric score in run_log/findings/BATTLE_PLAN must be lane-tagged.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    targets = [
        ".ralph/run_log.md",
        ".omx/research/findings.md",
        "docs/BATTLE_PLAN.md",
    ]
    n_scanned = 0
    for t in targets:
        p = root / t
        if not p.exists():
            continue
        n_scanned += 1
        violations.extend(_scan_doc_for_untagged_scores(p, root))
    if verbose:
        if violations:
            print(f"  [score-tag] {len(violations)} untagged score line(s):")
            for v in violations[:20]:  # cap output
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    … (+{len(violations) - 20} more)")
        else:
            print(f"  [score-tag] OK: {n_scanned} doc file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "SCORE LINES WITHOUT LANE TAG:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check E: SCORER_AT_INFLATE_WAIVED markers must name an env-gate var ──────


_WAIVER_GENERIC_RE = re.compile(
    r"#\s*SCORER_AT_INFLATE_WAIVED\s*(?::\s*([^\n]*))?"
)
_WAIVER_ENVGATE_RE = re.compile(
    r"env-gated[-_]([A-Z_][A-Z0-9_]*)(?:\s*=\s*[^\s,]+)?",
    re.IGNORECASE,
)


def _scan_for_unspecific_waivers(path: Path, repo_root: Path) -> list[str]:
    """Detect SCORER_AT_INFLATE_WAIVED markers that lack an env-gate name.

    The waiver format is:
        # SCORER_AT_INFLATE_WAIVED:env-gated-INFLATE_TTO=1
    The reason MUST start with `env-gated-` and name a specific env var.
    Bare `# SCORER_AT_INFLATE_WAIVED` (no reason) or
    `# SCORER_AT_INFLATE_WAIVED:reason-without-env-gate` is rejected so
    operators can audit which env-vars enable scorer-at-inflate paths.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    violations: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = _WAIVER_GENERIC_RE.search(line)
        if not m:
            continue
        reason = (m.group(1) or "").strip()
        if not reason:
            violations.append(
                f"{rel}:{i}: bare `# SCORER_AT_INFLATE_WAIVED` with no "
                f"reason. Required form: "
                f"`# SCORER_AT_INFLATE_WAIVED:env-gated-<ENV_VAR_NAME>=<val>`."
            )
            continue
        if not _WAIVER_ENVGATE_RE.search(reason):
            violations.append(
                f"{rel}:{i}: waiver reason {reason!r} does not name an "
                f"env-gate. Required: 'env-gated-<ENV_VAR_NAME>[=val]' so "
                f"operators can audit which env-var enables this path."
            )
    return violations


def check_waivers_specify_env_gate(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every scorer-at-inflate waiver must name the env-gate that enables it.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    submissions_dir = root / "submissions"
    if submissions_dir.exists():
        for p in submissions_dir.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_for_unspecific_waivers(p, root))
    if verbose:
        if violations:
            print(f"  [waiver-envgate] {len(violations)} unspecific waiver(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [waiver-envgate] OK: {n_scanned} submission file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "SCORER_AT_INFLATE_WAIVED MARKERS WITHOUT ENV-GATE NAME:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check F: --half-frame archive build requires half-frame-trained renderer


def _scan_for_halfframe_without_trained_profile(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect --half-frame archive builds without a trained-for-it profile.

    Per memory `feedback_half_frame_breaks_posenet` (2026-04-27): the
    Quantizr half-frame trick BREAKS PoseNet on the dilated-h64 baseline
    (PoseNet=28.7, score 17.55) because that renderer's MotionPredictor uses
    `(e_t1 - e_t).abs()` and warped-even-mask zeroes the diff.

    Rule: any invocation of `build_baseline_archive.py --half-frame` MUST
    also pass `--profile <X>` where the profile dict has
    `mask_half_sim_prob > 0` OR `use_zoom_flow=True`. We can statically
    check the script-text only — the profile lookup happens at runtime
    via `tac.profiles.PROFILES[X]`. So we (a) extract the profile name
    from the same invocation arg list, (b) import PROFILES, (c) check
    the keys.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if "build_baseline_archive" not in text or "--half-frame" not in text:
        return []
    violations: list[str] = []
    # Try to load PROFILES; if unavailable, fall back to text-marker check.
    try:
        from tac.profiles import PROFILES as _PROFILES
    except Exception:
        _PROFILES = None
    # Find each --half-frame mention; near it, find --profile <name>.
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if "--half-frame" not in line:
            continue
        # Skip the argparse flag DEFINITION itself (false positive — this is
        # the file that introduces the flag, not a caller). Detect the
        # `add_argument("--half-frame"...)` pattern in the same line OR the
        # 2 preceding lines (multi-line argparse calls are common).
        defn_window = "\n".join(lines[max(0, i - 3): i])
        if "add_argument" in line or "add_argument" in defn_window:
            continue
        # Skip docstring / help-string occurrences inside a triple-quoted
        # block on the same line: these are not invocations.
        if '"""' in line and line.count('"""') >= 1 and "--half-frame" in line.split('"""')[-1]:
            # Inside a docstring tail — skip (heuristic).
            pass
        # Scan a 30-line window for --profile.
        window_start = max(0, i - 30)
        window_end = min(len(lines), i + 30)
        window = "\n".join(lines[window_start:window_end])
        prof_match = re.search(
            r"--profile[\s=]+['\"]?([A-Za-z0-9_]+)['\"]?", window
        )
        if not prof_match:
            violations.append(
                f"{rel}:{i}: `--half-frame` present but no `--profile` "
                f"in 30-line window. Half-frame archives REQUIRE a "
                f"renderer trained with mask_half_sim_prob>0 OR "
                f"use_zoom_flow=True (memory feedback_half_frame_breaks_posenet)."
            )
            continue
        prof_name = prof_match.group(1)
        if _PROFILES is None:
            # Best effort — name-based sanity check.
            if "half_frame" not in prof_name and "zoom" not in prof_name:
                violations.append(
                    f"{rel}:{i}: `--half-frame` with `--profile {prof_name}` "
                    f"— profile name does not contain 'half_frame' or "
                    f"'zoom'. Verify profile has mask_half_sim_prob>0 OR "
                    f"use_zoom_flow=True (PROFILES not importable in scan)."
                )
            continue
        prof = _PROFILES.get(prof_name)
        if prof is None:
            violations.append(
                f"{rel}:{i}: `--half-frame` with unknown profile {prof_name!r}."
            )
            continue
        ok = (
            prof.get("mask_half_sim_prob", 0) > 0
            or prof.get("use_zoom_flow", False) is True
        )
        if not ok:
            violations.append(
                f"{rel}:{i}: `--half-frame` with profile {prof_name!r} which "
                f"has mask_half_sim_prob=0 AND use_zoom_flow=False. This "
                f"BREAKS PoseNet (memory feedback_half_frame_breaks_posenet, "
                f"verified 2026-04-27 score 17.55). Use a profile with "
                f"either flag enabled (e.g., 'dilated_h64_half_frame')."
            )
    return violations


def check_halfframe_archive_uses_trained_profile(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """`--half-frame` archive builds must use a renderer trained for it.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    # Scan all python + shell scripts.
    for path in _iter_python_files(root, ["scripts", "experiments"]):
        n_scanned += 1
        violations.extend(_scan_for_halfframe_without_trained_profile(path, root))
    for path in _iter_shell_files(root, ["scripts"]):
        n_scanned += 1
        violations.extend(_scan_for_halfframe_without_trained_profile(path, root))
    if verbose:
        if violations:
            print(f"  [halfframe] {len(violations)} half-frame mismatch(es):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [halfframe] OK: {n_scanned} script file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "HALF-FRAME ARCHIVE WITHOUT HALF-FRAME-TRAINED RENDERER:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check G: profile keys without parse_args resolver (bidirectional) ────────


_PROFILE_KEY_EXEMPTIONS = frozenset({
    # Documentation / metadata keys — never resolved as flags.
    "_doc", "_notes", "_origin", "name", "description",
    # Pydantic / dataclass internal keys.
    "model_config", "Config",
    # Aliases for already-resolved keys (handled via downstream rename).
    "channels",  # alias for hidden_dim in some profiles
})


def _extract_profile_keys() -> set[str] | None:
    """Return the union of keys across all PROFILES dicts, or None if import fails."""
    try:
        from tac.profiles import PROFILES
    except Exception:
        return None
    keys: set[str] = set()
    for prof in PROFILES.values():
        if isinstance(prof, dict):
            keys.update(prof.keys())
    return keys - _PROFILE_KEY_EXEMPTIONS


def _scan_for_resolver_keys(text: str) -> set[str]:
    """Pull every `cfg.<KEY> = ...` assignment + `args.<KEY>` read.

    Also catches a wide variety of profile-key access patterns so a key
    used anywhere in the codebase (not just in train_renderer) counts as
    'resolved'. The intent is to flag keys that have ZERO consumers, which
    is the actual bug class — not to require a specific resolver pattern.

    Resolver detection patterns (any one is sufficient):
      cfg.X = …                             # assignment
      cfg.X                                 # bare read
      args.X                                # parsed-args read
      profile["X"] / profile.get("X")       # dict access (variants:
        prof / p / cfg / config / hp / params / arch_dict / arch / vals /
        opts / overrides)
      kwargs.get("X")
      setattr(_, "X", _) / getattr(_, "X")
      self.config.X / self.cfg.X / self._cfg.X / self._config.X
      def …(X: type = default)              # function/method parameter
      f(X=value)                            # keyword argument in call
      X: type = default                     # dataclass field declaration
      # PROFILE_KEY_RESOLVED:X              # explicit waiver marker
    """
    out: set[str] = set()
    for m in re.finditer(r"\bcfg\.([A-Za-z_][A-Za-z0-9_]*)\s*=", text):
        out.add(m.group(1))
    for m in re.finditer(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)\b", text):
        out.add(m.group(1))
    # `profile["<KEY>"]` / `profile.get("<KEY>")` / `prof["<KEY>"]` /
    # `p["<KEY>"]` / `cfg["<KEY>"]` — all dict-access patterns.
    # Extended to cover common alias names: `vals`, `opts`, `overrides`.
    for m in re.finditer(
        r'\b(?:profile|prof|p|cfg|config|hp|params|arch_dict|arch|vals|opts|overrides|profile_vals)'
        r'(?:\.get)?\s*[(\[]\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']',
        text,
    ):
        out.add(m.group(1))
    # Bare attribute access `cfg.<KEY>` (read, not assignment).
    for m in re.finditer(r"\bcfg\.([A-Za-z_][A-Za-z0-9_]*)\b", text):
        out.add(m.group(1))
    # `kwargs.get("<KEY>")` and `setattr(.., "<KEY>", ..)`.
    for m in re.finditer(r'\bkwargs\.get\s*\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', text):
        out.add(m.group(1))
    for m in re.finditer(r'\bsetattr\s*\([^,]+,\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', text):
        out.add(m.group(1))
    # `getattr(<x>, "<KEY>"...)` reads.
    for m in re.finditer(r'\bgetattr\s*\([^,]+,\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', text):
        out.add(m.group(1))
    # `self.config.X`, `self.cfg.X`, `self._cfg.X`, `self._config.X`
    # — dataclass-config reads (legitimate consumer pattern,
    # used by tac.contrib.domain_solvers, etc.).
    for m in re.finditer(
        r'\bself\.(?:_?config|_?cfg)\.([A-Za-z_][A-Za-z0-9_]*)\b',
        text,
    ):
        out.add(m.group(1))
    # Explicit waiver marker for cases the scanner can't reach
    # (e.g. dynamic load via ** spread). Format:
    #   `# PROFILE_KEY_RESOLVED:my_key`  (single key)
    #   `# PROFILE_KEY_RESOLVED:k1,k2,k3` (multiple keys)
    for m in re.finditer(
        r'#\s*PROFILE_KEY_RESOLVED:\s*([A-Za-z_][A-Za-z0-9_,\s]*)',
        text,
    ):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                out.add(k)
    # Walk the AST to find function/method parameter names and
    # dataclass field declarations. This catches the very common
    # consumption pattern where a function signature names the key
    # directly, e.g. `def train(scorer_weight: float = 20.0): …`.
    # Without AST parsing, the regex would have to be fragile.
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                out.add(arg.arg)
            if node.args.vararg:
                out.add(node.args.vararg.arg)
            if node.args.kwarg:
                out.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            # Dataclass-style field declarations:  `X: type = default`
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    out.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            out.add(t.id)
        elif isinstance(node, ast.Call):
            # `f(X=value)` — keyword arguments in calls.
            for kw in node.keywords:
                if kw.arg is not None:  # exclude **kwargs spreads
                    out.add(kw.arg)
    return out


def check_profile_keys_have_resolvers(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Bidirectional: every profile key must have a parse_args resolver.

    The existing dead-resolver scanner finds parse_args entries that have
    no profile mapping (orphan flags). This complementary check finds
    profile keys that have no parse_args resolver (silent default → bug
    cluster: pose_dim, blend_mode, etc.).
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    # If the provided repo_root has no profiles.py, skip — tests use this
    # path with a stub repo and we don't want them to pull live PROFILES.
    if not (root / "src" / "tac" / "profiles.py").exists():
        if verbose:
            print(f"  [profile-resolver] SKIP: {root}/src/tac/profiles.py not found")
        return []
    keys = _extract_profile_keys()
    if keys is None:
        if verbose:
            print(f"  [profile-resolver] SKIP: PROFILES not importable")
        return []
    # Resolver search: the profile-key consumer can be ANY file under
    # src/tac/ or experiments/. The original narrow list (train_renderer +
    # train_distill + training + build_renderer + profiles) missed legit
    # consumers — e.g. T_max is used by the cosine scheduler in
    # train_renderer.py:1356, but the regex matched the assignment not the
    # use. Cast a wide net: if a key appears as a dict-access ANYWHERE in
    # src/tac or experiments, count it as resolved. This makes the gate
    # find the actual bug class — keys with ZERO consumers — without
    # producing false positives on widely-used keys.
    resolver_search_dirs = ["src/tac", "experiments"]
    resolved: set[str] = set()
    for d in resolver_search_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            resolved.update(_scan_for_resolver_keys(text))
    resolver_files = ["src/tac/", "experiments/"]  # for error message
    missing = sorted(keys - resolved)
    for k in missing:
        violations.append(
            f"profile key {k!r} has no resolver in any of "
            f"{resolver_files}. Profiles with this key would silently use "
            f"the constructor default. Add `cfg.{k} = profile['{k}']` to "
            f"the resolver section."
        )
    if verbose:
        if violations:
            print(f"  [profile-resolver] {len(violations)} unresolved profile key(s):")
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    … (+{len(violations) - 20} more)")
        else:
            print(f"  [profile-resolver] OK: {len(keys)} profile key(s) all resolved")
    if violations and strict:
        raise MetaBugViolation(
            "PROFILE KEYS WITHOUT RESOLVERS:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check H: scorer-at-inflate path must print [strict-scorer-rule] banner ──


def _file_loads_scorer_at_inflate(path: Path) -> bool:
    """Quick check: does this inflate*.py contain a scorer-load call?"""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return False
    return any(
        keyword in text
        for keyword in (
            "load_scorers", "load_posenet", "load_segnet",
            "load_differentiable_scorers", "tac.scorer",
            "extract_gt_pose_targets", "load_posenet_targets",
        )
    )


def check_inflate_scorer_load_has_runtime_banner(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Inflate files loading scorers must print a [strict-scorer-rule] banner.

    Per CLAUDE.md strict-scorer-rule: any inflate-time scorer-load path
    is non-compliant and MUST print a runtime warning banner so the score
    can be properly tagged in the run-log. Static scan: every inflate*.py
    that imports/calls a scorer loader must contain a literal
    `print(...)` of a string containing `[strict-scorer-rule]`.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    submissions_dir = root / "submissions"
    if submissions_dir.exists():
        for p in submissions_dir.rglob("inflate*.py"):
            n_scanned += 1
            if not _file_loads_scorer_at_inflate(p):
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            if "[strict-scorer-rule]" not in text:
                rel = p.relative_to(root) if p.is_absolute() else p
                violations.append(
                    f"{rel}: file loads scorer at inflate time but never "
                    f"prints '[strict-scorer-rule]' banner. Add a "
                    f"`print('[strict-scorer-rule] ...', file=sys.stderr)` "
                    f"on the env-gated branch so the score can be tagged "
                    f"[scorer-at-inflate-noncompliant]."
                )
    if verbose:
        if violations:
            print(f"  [scorer-banner] {len(violations)} inflate file(s) lack runtime banner:")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [scorer-banner] OK: {n_scanned} inflate file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "INFLATE SCORER-LOAD WITHOUT RUNTIME BANNER:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check I: test files importing symbols that don't exist ──────────────────


def _resolve_module_to_path(module: str, repo_root: Path) -> Path | None:
    """Map dotted module name to .py file path under the repo."""
    parts = module.split(".")
    candidates = [
        repo_root / "src" / Path(*parts).with_suffix(".py"),
        repo_root / Path(*parts).with_suffix(".py"),
        repo_root / "src" / Path(*parts) / "__init__.py",
        repo_root / Path(*parts) / "__init__.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _collect_module_top_level_names(tree: ast.Module) -> set[str]:
    """Names defined at module top level (functions, classes, assignments)."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
    return out


def _collect_importorskip_modules(tree: ast.Module) -> set[str]:
    """Collect every module name passed to pytest.importorskip(...) at module top.

    Honors the canonical pytest pattern for tests of optional / pending
    dependencies:
        pytest.importorskip("tac.self_augmentation")
        from tac.self_augmentation import foo  # scanner accepts because of skip above

    A test file that opts in this way runs cleanly when the module lands and
    skips gracefully (with reason) when it's missing — matches industrial
    pytest workflow for in-flight subagent / staged work.
    """
    skipped: set[str] = set()
    for node in tree.body:
        # Look for `pytest.importorskip("X")` either as a bare expression
        # statement or as an assignment RHS (`mod = pytest.importorskip("X")`).
        call: ast.Call | None = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            getattr(node, "value", None), ast.Call
        ):
            call = node.value  # type: ignore[assignment]
        if call is None:
            continue
        func = call.func
        is_importorskip = (
            isinstance(func, ast.Attribute)
            and func.attr == "importorskip"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )
        if not is_importorskip:
            continue
        if not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            skipped.add(first.value)
    return skipped


def _has_module_level_skip(tree: ast.Module) -> bool:
    """Return True if module body contains `pytest.skip(..., allow_module_level=True)`.

    This is the canonical pytest pattern for "skip the whole module" — see
    pytest docs on pytest.skip + allow_module_level. The scanner walks the
    module body (including nested if/try blocks at the top level) for any
    such call. When found, ALL ImportFrom in the file are tolerated since
    pytest will refuse to collect the module before any inner import runs.
    """
    def _scan(stmts: list[ast.stmt]) -> bool:
        for node in stmts:
            call: ast.Call | None = None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
            if call is not None:
                func = call.func
                is_skip = (
                    isinstance(func, ast.Attribute)
                    and func.attr == "skip"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pytest"
                )
                if is_skip:
                    for kw in call.keywords:
                        if (
                            kw.arg == "allow_module_level"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            return True
            # Recurse into top-level if/try blocks (still "module level").
            if isinstance(node, ast.If) and (_scan(node.body) or _scan(node.orelse)):
                return True
            if isinstance(node, ast.Try):
                if _scan(node.body) or _scan(node.orelse) or _scan(node.finalbody):
                    return True
                for handler in node.handlers:
                    if _scan(handler.body):
                        return True
        return False
    return _scan(tree.body)


def _scan_test_file_for_dead_imports(
    path: Path, repo_root: Path,
) -> list[str]:
    """Catch broken test imports. Companion to existing dead-import scanner.

    Existing scanner skips test dirs because of fixture noise. But real
    failures hide there: test_yousfi_*, test_wavelet_variance have been
    broken for sessions. Scan ONLY test files, ONLY for ImportError-class
    issues (target module not found, target name not in module).

    Honors `pytest.importorskip("X")` at module top as a legitimate opt-out
    for tests of optional / in-flight modules — see _collect_importorskip_modules.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []

    if _has_module_level_skip(tree):
        # `pytest.skip(..., allow_module_level=True)` at module top — pytest
        # refuses to collect the module so no ImportFrom inside ever runs.
        return []

    importorskip_mods = _collect_importorskip_modules(tree)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        # Only check intra-project imports (start with tac, experiments,
        # comma_lab, scripts).
        mod = node.module
        if not (
            mod.startswith("tac")
            or mod.startswith("experiments")
            or mod.startswith("comma_lab")
            or mod.startswith("scripts")
        ):
            continue
        # Honor pytest.importorskip("X") opt-out: skip imports of X or any
        # submodule under X.
        if any(mod == m or mod.startswith(m + ".") for m in importorskip_mods):
            continue
        # Resolve module file.
        mod_path = _resolve_module_to_path(mod, repo_root)
        if mod_path is None or not mod_path.exists():
            violations.append(
                f"{rel}:{node.lineno}: imports from {mod!r} which does not "
                f"resolve to a file. Either delete the test or fix the "
                f"import (test has been silently broken)."
            )
            continue
        # For each name imported, check it's defined in target.
        try:
            target_text = mod_path.read_text()
            target_tree = ast.parse(target_text)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            continue
        defined = _collect_module_top_level_names(target_tree)
        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name in defined:
                continue
            # `from tac import preflight` is a valid submodule import even when
            # `preflight` isn't a top-level name in `tac/__init__.py`. Python
            # resolves it to `tac/preflight.py`. Accept the import if the
            # submodule file exists.
            sub_path = _resolve_module_to_path(f"{mod}.{name}", repo_root)
            if sub_path is not None and sub_path.exists():
                continue
            violations.append(
                f"{rel}:{node.lineno}: imports {name!r} from {mod!r} "
                f"but {mod} does not define it. Test will ImportError "
                f"at collection time (silently skipped)."
            )
    return violations


def check_test_files_imports_resolve(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Tests file imports must resolve to actually-defined symbols.

    Per the historical pattern: test files have been silently broken for
    sessions because the existing dead-import scanner skips test dirs.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    test_dir = root / "src" / "tac" / "tests"
    if test_dir.exists():
        for p in test_dir.rglob("test_*.py"):
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_test_file_for_dead_imports(p, root))
    if verbose:
        if violations:
            print(f"  [test-imports] {len(violations)} broken test import(s):")
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    … (+{len(violations) - 20} more)")
        else:
            print(f"  [test-imports] OK: {n_scanned} test file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "TEST FILE IMPORTS DO NOT RESOLVE:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check J: subagent dispatch prompts must mention cost cap ────────────────


_VASTAI_PROMPT_RE = re.compile(r"\b(?:vast\.?ai|Vast\.?ai)\b")
_COST_GUARD_RE = re.compile(
    r"\$\s*\d|cost cap|budget|\$24 hard cap|destroy.*instance",
    re.IGNORECASE,
)


def _scan_for_vastai_prompt_no_cost_cap(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect agent prompts/dispatches mentioning Vast.ai with no cost guard."""
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if (
        "/tests/" in rel_s
        or "test_" in path.name
        or "preflight.py" in rel_s
        or rel_s.endswith("CLAUDE.md")
        or rel_s.endswith("MEMORY.md")
        or "memory/" in rel_s
        or rel_s.startswith(".memory/")
    ):
        return []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if not _VASTAI_PROMPT_RE.search(text):
        return []
    # Whole-file granularity for agent prompts: if the file mentions
    # vast.ai but never mentions a cost guard, flag it once.
    if _COST_GUARD_RE.search(text):
        return []
    # Find the first line that mentions vast.ai for the violation lineno.
    lineno = 1
    for i, line in enumerate(text.splitlines(), start=1):
        if _VASTAI_PROMPT_RE.search(line):
            lineno = i
            break
    return [
        f"{rel}:{lineno}: file dispatches/discusses Vast.ai work without "
        f"any cost-cap mention (no '$', 'budget', 'cost cap', or "
        f"'destroy instance'). Per feedback_vastai_cost_paranoia: "
        f"every Vast.ai dispatch MUST name a $ cap and a destroy condition."
    ]


def check_vastai_prompts_have_cost_cap(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Subagent prompts mentioning Vast.ai must mention a cost cap.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    scan_dirs = [".agents", "prompts"]
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.md"):
            n_scanned += 1
            violations.extend(_scan_for_vastai_prompt_no_cost_cap(p, root))
    if verbose:
        if violations:
            print(f"  [vastai-cost-cap] {len(violations)} unguarded vastai prompt(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [vastai-cost-cap] OK: {n_scanned} prompt file(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "VASTAI PROMPTS WITHOUT COST CAP:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check K: --with-uniward-delta requires --allow-pending-compliance/attestation


def _scan_for_uniward_delta_without_attestation(
    path: Path, repo_root: Path,
) -> list[str]:
    """Detect --with-uniward-delta usage without the compliance gate.

    Per Lane C R5 (commit ef8a9a1b): every UNIWARD δ injection MUST pass
    one of:
      - --allow-pending-compliance (operator override, recorded)
      - <attestation file present at canonical path>
    Static check: if a script invokes build_baseline_archive with
    --with-uniward-delta, it must ALSO pass --allow-pending-compliance OR
    have an explicit comment referencing the attestation file path.

    Refinement (R6 cleanup, 2026-04-27): the file that DEFINES the flag
    (experiments/build_baseline_archive.py) is excluded — every mention
    inside it is either the argparse definition, a help string, or an
    error message. The flag's compliance enforcement is implemented IN
    that file (the gate). So the rule: scan only CALLERS, not the file
    that owns the argparse definition. We detect ownership by looking
    for a top-level `add_argument("--with-uniward-delta"` line.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    if "/tests/" in rel_s or "test_" in path.name:
        return []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if "--with-uniward-delta" not in text:
        return []
    # Skip the file that DEFINES the flag (and thus enforces the gate
    # internally). Every textual occurrence inside that file is either
    # the argparse spec, the help string, or an internal error/comment —
    # never an actual subprocess call to itself.
    if 'add_argument("--with-uniward-delta"' in text or "add_argument('--with-uniward-delta'" in text:
        return []
    violations: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if "--with-uniward-delta" not in line:
            continue
        # Skip occurrences inside an obvious string literal (help text,
        # error message). Heuristic: line contains the flag preceded by
        # an opening quote AND followed (within the line) by a closing
        # quote, with no `subprocess` / `Popen` / shell-call markers.
        stripped = line.strip()
        is_comment_only = stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'")
        if is_comment_only and "subprocess" not in line and "Popen" not in line:
            continue
        # Scan a 30-line window.
        window_start = max(0, i - 30)
        window_end = min(len(lines), i + 30)
        window = "\n".join(lines[window_start:window_end])
        if (
            "--allow-pending-compliance" in window
            or "lane_c_compliance_attestations" in window
            or "verify_attestation_for_blob" in window
        ):
            continue
        violations.append(
            f"{rel}:{i}: `--with-uniward-delta` without "
            f"`--allow-pending-compliance` OR an attestation file reference "
            f"in 30-line window. Per Lane C R5 (commit ef8a9a1b): δ.bin "
            f"injection requires explicit compliance gate."
        )
    return violations


def check_uniward_delta_has_attestation_gate(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """`--with-uniward-delta` invocations must include compliance gate.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for path in _iter_python_files(root, ["scripts", "experiments"]):
        n_scanned += 1
        violations.extend(_scan_for_uniward_delta_without_attestation(path, root))
    for path in _iter_shell_files(root, ["scripts"]):
        n_scanned += 1
        violations.extend(_scan_for_uniward_delta_without_attestation(path, root))
    if verbose:
        if violations:
            print(f"  [uniward-attestation] {len(violations)} ungated δ invocation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [uniward-attestation] OK: {n_scanned} script(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "UNIWARD DELTA WITHOUT COMPLIANCE GATE:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check L: Vast.ai remote scripts must write provenance.json ──────────────


def _shell_script_writes_provenance(path: Path) -> bool:
    """True if this shell script writes provenance.json (any form)."""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return False
    return "provenance.json" in text or "PROVENANCE_JSON" in text


def check_remote_scripts_write_provenance(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every `scripts/remote_*.sh` must write provenance.json.

    Per CLAUDE.md canonical pipeline standard + memory
    `feedback_canonical_remote_bootstraps`: every remote run produces a
    provenance.json so a fresh agent can reconstruct the experiment.
    Lanes A/B/D/G shipped without it.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    scripts_dir = root / "scripts"
    if not scripts_dir.exists():
        if verbose:
            print(f"  [provenance] SKIP: scripts/ not found")
        return []
    n_scanned = 0
    for p in sorted(scripts_dir.glob("remote_*.sh")):
        n_scanned += 1
        if not _shell_script_writes_provenance(p):
            rel = p.relative_to(root) if p.is_absolute() else p
            violations.append(
                f"{rel}: remote script does not write provenance.json. "
                f"Per feedback_canonical_remote_bootstraps: every remote "
                f"run must emit provenance + heartbeat + run_record."
            )
    if verbose:
        if violations:
            print(f"  [provenance] {len(violations)} remote script(s) missing provenance:")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [provenance] OK: {n_scanned} remote script(s) scanned")
    if violations and strict:
        raise MetaBugViolation(
            "REMOTE SCRIPTS WITHOUT PROVENANCE.JSON:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check M: F.kl_div(reduction="batchmean") on spatial tensors ────────────
#
# Bug class (2026-04-27 council forensics, findings.md "Lane G — really
# dead, or bugged?"): `F.kl_div(..., reduction="batchmean")` divides only
# by the batch dim. On a (B, C, H, W) segmentation logit tensor that
# under-divides the canonical per-pixel mean by H × W (= 196,608 for
# 384 × 512 SegNet). The same surface API silently fits two completely
# different objectives depending on tensor shape — exactly the silent-
# default class CLAUDE.md FORBIDDEN PATTERNS warns against.
#
# Live failure: every caller of `kl_distill_segnet_only` passing
# `kl_distill_weight=1.0` (DEN/SHIRAZ/WILDE/Lane-D training profiles,
# Lane G pose TTO v1/v2) ran with a ~5000× over-weighted KL term.
#
# Defense: forbid `reduction="batchmean"` outright in the scanned dirs,
# require a `# KL_BATCHMEAN_OK:<reason>` waiver marker on the call line
# justifying that the input is provably a flat (B, num_classes) classifier
# tensor (the only shape for which `batchmean` matches the user's intent).
# Mirrors the existing `# KL_RAW_PAIRS_OK:<reason>` waiver pattern from
# Check B above.


def _scan_python_for_kl_div_batchmean(path: Path, repo_root: Path) -> list[str]:
    """Detect any `F.kl_div(..., reduction="batchmean")` call without a
    same-line `# KL_BATCHMEAN_OK:<reason>` waiver marker.

    Heuristic: matches calls whose function reference ends in `kl_div`
    (covers `F.kl_div`, `torch.nn.functional.kl_div`, and bare
    `kl_div` after `from torch.nn.functional import kl_div`). Only the
    exact `reduction="batchmean"` keyword form is flagged — positional
    `reduction` is rare in this API but still caught when the value is
    a string constant `"batchmean"`.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        except Exception:
            func_str = ""
        # Match `F.kl_div`, `torch.nn.functional.kl_div`, bare `kl_div`.
        if not (func_str == "kl_div" or func_str.endswith(".kl_div")):
            continue
        # Look for reduction=... keyword.
        is_batchmean = False
        for kw in node.keywords:
            if kw.arg == "reduction" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value == "batchmean":
                is_batchmean = True
                break
        # Positional fallback: kl_div(input, target, size_average, reduce, reduction)
        # The 5th positional (index 4) is `reduction`. We only flag if it is a
        # string literal "batchmean"; anything else (variable, missing) is fine.
        if not is_batchmean and len(node.args) >= 5:
            arg5 = node.args[4]
            if isinstance(arg5, ast.Constant) and arg5.value == "batchmean":
                is_batchmean = True
        if not is_batchmean:
            continue
        # Same-line waiver opt-out.
        ln = node.lineno
        if 0 < ln <= len(lines):
            comment_idx = lines[ln - 1].find("#")
            if comment_idx >= 0 and "KL_BATCHMEAN_OK" in lines[ln - 1][comment_idx:]:
                continue
        violations.append(
            f"{rel}:{node.lineno}: `F.kl_div(..., reduction=\"batchmean\")` "
            f"detected. On a spatial (B, C, H, W) tensor this under-divides "
            f"the canonical per-pixel mean by H × W (=196,608 for 384x512 "
            f"SegNet — see findings.md \"Lane G — really dead, or bugged?\"). "
            f"Use `F.kl_div(..., reduction=\"none\").sum(dim=1).mean()` for "
            f"per-pixel-per-class mean (canonical pattern: "
            f"`kl_distill_scorer_loss` line 622+646). If the input is "
            f"provably a flat (B, num_classes) classifier tensor and "
            f"`batchmean` is intended, add `# KL_BATCHMEAN_OK:<reason>` "
            f"on this line."
        )
    return violations


def check_kl_div_reduction_correct(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid `F.kl_div(..., reduction="batchmean")` without explicit waiver.

    See module-level Check M comment + findings.md
    "## 2026-04-27 Council forensics: Lane G — really dead, or bugged?"
    for the full math derivation. The scanner walks `src/tac/`,
    `experiments/`, `submissions/`, and `scripts/` for offending calls
    and requires a same-line `# KL_BATCHMEAN_OK:<reason>` marker as the
    only opt-out (mirrors the `# KL_RAW_PAIRS_OK:<reason>` pattern in
    Check B above).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    scan_dirs = ["src/tac", "experiments", "scripts", "submissions"]
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_python_for_kl_div_batchmean(p, root))

    if verbose and violations:
        print(f"  [no-kl-div-batchmean] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-kl-div-batchmean] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "F.kl_div(reduction=\"batchmean\") on spatial tensors:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nSee findings.md \"Lane G — really dead, or bugged?\" "
            "for the math (1/H/W silent under-division). Use "
            "`reduction=\"none\"` → `.sum(dim=1).mean()` (canonical pattern "
            "in kl_distill_scorer_loss line 622+646), OR add a same-line "
            "`# KL_BATCHMEAN_OK:<reason>` marker if the input is provably "
            "a flat (B, num_classes) classifier tensor."
        )
    return violations


# 2026-04-27 audit: the 12 checks listed in the previous TODO block are now
# wired into preflight_all() at strict=True (see lines ~316-330 above), the
# violation counts above ran clean. TODO removed; if a future check needs to
# be deferred, add it directly to preflight_all() at strict=False with a
# one-line note linking the audit that promotes it.


# ════════════════════════════════════════════════════════════════════════════
# Check N (29th meta-bug): silent-default-masquerading-as-negative-result
# ════════════════════════════════════════════════════════════════════════════
#
# The bug class — a CLI flag is missing, the script auto-discovers from a list
# of N hardcoded fallback paths, none exist, the script prints a `[WARN] ...`
# line and proceeds with a silent default (None / zero / empty). The operator
# sees the script "succeed" but the produced artifact was trained against the
# wrong inputs. The result then enters the council deliberation as if it were
# a real negative result, leading to "this lane is dead" misjudgments.
#
# Real-world incidents (2 in 2 days, 2026-04-27):
#   • Lane G v1 — `kl_distill_weight` defaulted to 5e-6 with batchmean reduction;
#     reported "KL distill killed PoseNet" when in fact the gradient was 5000x
#     over-weighted. (See findings.md "Lane G — really dead, or bugged?")
#   • Lane F v1 — `qat_finetune.py` had no `--poses` arg, auto-discovered from
#     `experiments/results/gt_poses.pt` + `upstream/gt_poses.pt`, neither
#     existed, printed `[WARN] ... will use zero poses` and proceeded. Renderer
#     was QAT-trained against zero poses, deployed against real poses, +58%
#     PoseNet regression reported as "FP4 quantization is dead." (See findings.md
#     "Lane F regression — bugged or dead?")
#
# The structural fix: forbid the pattern `for x in [Path(...), Path(...)]:
# if x.exists(): ... ; print("[WARN] ... not found"); return None` (or
# equivalent). Either RAISE (preferred) or document the silent fallback with
# an `# AUTO_DISCOVERY_OK:<reason>` waiver marker on the loop or warn line.
#
# Detection (combined AST + text):
#   1. AST-find every `for ... in [<list of Path/str literals>]:` loop body
#      that contains `if <var>.exists():` AND a `break` / `return` / assignment.
#   2. Look in the *same containing function* for a subsequent print-or-log call
#      whose string argument contains `[WARN]` (case-insensitive `WARN`).
#   3. If the function does NOT raise / sys.exit after the warn, flag it.
#   4. Same-line waiver `# AUTO_DISCOVERY_OK:<reason>` on either the for loop
#      header OR the warn line opts out.


_AUTO_DISCOVERY_WAIVER_TOKEN = "AUTO_DISCOVERY_OK"


def _line_has_waiver(lines: list[str], lineno: int) -> bool:
    """Return True if `lines[lineno-1]` has a `# AUTO_DISCOVERY_OK:` comment."""
    if not (0 < lineno <= len(lines)):
        return False
    src_line = lines[lineno - 1]
    comment_idx = src_line.find("#")
    if comment_idx < 0:
        return False
    return _AUTO_DISCOVERY_WAIVER_TOKEN in src_line[comment_idx:]


def _function_has_waiver(lines: list[str], fn_node: ast.AST) -> bool:
    """Return True if any line in the function body has the waiver marker.

    Permissive: a single waiver anywhere in the function exempts the function.
    Lets callers waive a multi-line auto-discovery block without picking the
    exact line."""
    if not hasattr(fn_node, "lineno") or not hasattr(fn_node, "end_lineno"):
        return False
    start, end = fn_node.lineno, fn_node.end_lineno or fn_node.lineno
    for ln in range(start, end + 1):
        if _line_has_waiver(lines, ln):
            return True
    return False


def _list_is_path_candidates(node: ast.AST) -> bool:
    """Detect `[Path("..."), Path(...) / "...", Path(cfg.x) / "..."]` literal.

    The list must contain >=2 elements that are either Path() calls, BinOp /
    expressions involving Path, or string literals that look like file paths
    (have a "/" or end with .pt/.bin/.json). Lenient on the exact form to
    handle the patterns we've seen in the wild.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    if len(node.elts) < 2:
        return False
    n_path_like = 0
    for elt in node.elts:
        # Path("...") or Path(...) / "..."
        text = ""
        try:
            text = ast.unparse(elt) if hasattr(ast, "unparse") else ""
        except Exception:
            pass
        if "Path(" in text:
            n_path_like += 1
            continue
        # bare string with path-like marker
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            v = elt.value
            if "/" in v or v.endswith((".pt", ".bin", ".json", ".mkv", ".pth", ".safetensors")):
                n_path_like += 1
    return n_path_like >= 2


def _function_warns_then_proceeds(fn_node: ast.AST, lines: list[str]) -> tuple[bool, int]:
    """Scan a function body for a `print/log("[WARN] ...")`-style call that is
    NOT followed by `raise` / `sys.exit` / `SystemExit` in the same function.

    Returns (has_silent_warn, warn_lineno). `has_silent_warn` is True when the
    warn is unguarded. lineno=0 when no warn found.
    """
    warn_calls: list[tuple[int, ast.Call]] = []
    raise_or_exit_after: dict[int, bool] = {}

    # First pass: collect warn print/log calls.
    for sub in ast.walk(fn_node):
        if not isinstance(sub, ast.Call):
            continue
        # Only top-of-string literal argument check.
        for a in sub.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                # Case-insensitive: `[WARN]`, `WARNING`, `WARN:`.
                up = a.value.upper()
                if "[WARN]" in up or "WARNING:" in up or up.startswith("WARN "):
                    # Filter by function name to avoid false positives like
                    # `assert "[WARN]" in ...` (those are ast.Compare, not Call).
                    func_text = ""
                    try:
                        func_text = ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
                    except Exception:
                        pass
                    # Match `print`, `*.print`, `log`, `*.log`, `_warn`,
                    # `*.warn`, `*.warning`, `logger.warning`, etc.
                    func_lower = func_text.lower()
                    if any(tok in func_lower for tok in (
                        "print", "log", "warn", "_warn", "echo",
                    )):
                        warn_calls.append((sub.lineno, sub))
                        break

    if not warn_calls:
        return (False, 0)

    # Second pass: for each warn, check if a `raise` / `sys.exit` / `SystemExit`
    # appears in the function body AFTER the warn line.
    for warn_ln, _ in warn_calls:
        guarded = False
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Raise) and sub.lineno > warn_ln:
                guarded = True
                break
            if isinstance(sub, ast.Call):
                func_text = ""
                try:
                    func_text = ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
                except Exception:
                    pass
                if sub.lineno > warn_ln and func_text in (
                    "sys.exit", "exit", "SystemExit", "os._exit",
                ):
                    guarded = True
                    break
        raise_or_exit_after[warn_ln] = guarded

    # Silent warn = warn that is NOT guarded.
    for warn_ln, _ in warn_calls:
        if not raise_or_exit_after.get(warn_ln, False):
            return (True, warn_ln)
    return (False, 0)


def _scan_python_for_silent_auto_discovery(path: Path, repo_root: Path) -> list[str]:
    """Detect the silent-default-masquerading-as-negative-result pattern.

    Looks for functions containing BOTH:
      (a) a `for x in [<list of >=2 Path-like literals>]:` loop body that
          conditionally uses the candidate via `<x>.exists()`, AND
      (b) somewhere later in the same function, an unguarded `print/log/warn`
          call whose first string literal contains `[WARN]` / `WARNING:` /
          `WARN ` (case-insensitive), with no following `raise` / `sys.exit`.

    The opt-out marker is `# AUTO_DISCOVERY_OK:<reason>` placed anywhere in
    the offending function (typically on the for-loop header or the warn line).
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    # Skip test files (they intentionally exercise the bug pattern).
    if "tests" in rel.parts or rel.name.startswith("test_"):
        return []
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []
    lines = text.splitlines()
    violations: list[str] = []

    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _function_has_waiver(lines, fn_node):
            continue

        # (a) Find a for-loop with Path-list iter + .exists() check in body.
        has_path_list_loop = False
        loop_lineno = 0
        for sub in ast.walk(fn_node):
            if not isinstance(sub, ast.For):
                continue
            if not _list_is_path_candidates(sub.iter):
                continue
            # Body must contain `<var>.exists()` Attribute call.
            uses_exists = False
            for body_sub in ast.walk(sub):
                if isinstance(body_sub, ast.Call):
                    func_text = ""
                    try:
                        func_text = ast.unparse(body_sub.func) if hasattr(ast, "unparse") else ""
                    except Exception:
                        pass
                    if func_text.endswith(".exists"):
                        uses_exists = True
                        break
            if uses_exists:
                has_path_list_loop = True
                loop_lineno = sub.lineno
                break

        if not has_path_list_loop:
            continue

        # (b) Function must contain an unguarded warn call.
        has_silent_warn, warn_lineno = _function_warns_then_proceeds(fn_node, lines)
        if not has_silent_warn:
            continue

        violations.append(
            f"{rel}:{loop_lineno}: function `{fn_node.name}` uses Path-list "
            f"auto-discovery (loop at line {loop_lineno}) followed by an unguarded "
            f"`[WARN]` print at line {warn_lineno}, with no `raise`/`sys.exit` "
            f"after it. This is the SILENT-DEFAULT-MASQUERADING bug class — "
            f"the script proceeds with a wrong default that produces an invalid "
            f"result without operator awareness. See findings.md "
            f"\"Lane F regression — bugged or dead?\" (2026-04-27) and memory "
            f"`feedback_silent_default_masquerading_as_negative_result`. Fix: "
            f"raise SystemExit on missing input OR add a same-function "
            f"`# AUTO_DISCOVERY_OK:<reason>` marker on the loop or warn line."
        )

    return violations


def check_no_silent_auto_discovery_with_warn(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """29th meta-bug check: silent-default-masquerading-as-negative-result.

    Catches functions that auto-discover from a list of hardcoded paths,
    print a `[WARN]` line when none exist, and proceed without raising.
    The operator sees the script "succeed" but the artifact was built with
    the wrong inputs — leading to "this lane is dead" misjudgments.

    Real-world incidents:
      • Lane F v1 (qat_finetune.py): auto-discovered gt_poses.pt from 2 paths,
        printed [WARN] ... will use zero poses, trained renderer with wrong
        conditioning. Reported as "FP4 quantization is dead." (BUGGED.)
      • Lane G v1 (kl_distill_weight defaulted with batchmean reduction):
        same class — silent bad default reported as "KL distill is dead."

    Reference: findings.md "Lane F regression — bugged or dead?" + memory
    `feedback_silent_default_masquerading_as_negative_result`.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for py in _iter_python_files(root, _META_PY_SCAN_DIRS):
        n_scanned += 1
        violations.extend(_scan_python_for_silent_auto_discovery(py, root))

    if verbose and violations:
        print(f"  [no-silent-auto-discovery] {len(violations)} violation(s) across {n_scanned} files:")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print(f"  [no-silent-auto-discovery] OK: {n_scanned} files scanned")

    if violations and strict:
        raise MetaBugViolation(
            "SILENT-DEFAULT-MASQUERADING-AS-NEGATIVE-RESULT violations:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nThis is the 2-in-2-days bug class (Lane G + Lane F, "
            "2026-04-27). The pattern is: missing CLI flag → auto-discover "
            "from N hardcoded paths → none exist → print [WARN] → proceed "
            "with silent default → operator sees the result land as a "
            "negative outcome. Fix: replace the auto-discovery + warn with "
            "an explicit `--<flag>` argument that RAISES on missing input. "
            "Documented opt-out: same-function `# AUTO_DISCOVERY_OK:<reason>` "
            "marker. See findings.md and memory "
            "`feedback_silent_default_masquerading_as_negative_result`."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check N+1 (30th meta-bug): remote scripts must have executable bit
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-27: Lane GH script was committed without +x bit; the audit subagent
# caught it via test_remote_lane_gh_script.py::test_script_is_executable.
# This preflight check generalizes the protection.
def check_remote_scripts_executable_bit(strict: bool = False, verbose: bool = False) -> list[str]:
    """Every scripts/remote_*.sh must have the executable bit set so
    ``bash`` invocations + tmux dispatch work without requiring chmod first.
    """
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [executable-bit] OK: scripts dir not found, skipped")
        return violations
    for script_path in sorted(scripts_dir.glob("remote_*.sh")):
        st = script_path.stat()
        if not (st.st_mode & 0o111):
            violations.append(
                f"{script_path}: not executable (mode {oct(st.st_mode)}). "
                f"Run `chmod +x {script_path}` to fix. Required for bash + "
                f"tmux dispatch."
            )
    if verbose:
        n_scripts = len(list(scripts_dir.glob("remote_*.sh")))
        if violations:
            print(f"  [executable-bit] {len(violations)} violation(s) across {n_scripts} remote_*.sh")
        else:
            print(f"  [executable-bit] OK: {n_scripts} remote_*.sh script(s) all executable")
    if strict and violations:
        raise PreflightError(
            "REMOTE SCRIPT EXECUTABLE BIT VIOLATIONS — at least one "
            f"scripts/remote_*.sh lacks +x bit:\n  • " + "\n  • ".join(violations) +
            "\nFix: chmod +x on each. Required so bash dispatch works "
            "without manual chmod intervention. (Lane GH bug 2026-04-27.)"
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check N+2 (31st meta-bug): remote scripts must record predicted_band
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-27: every remote_*.sh script today documents predicted_band in
# provenance JSON for empirical calibration of council intuition. Without
# the metadata, post-hoc analysis can't answer "did this lane land within
# the council's predicted range?" — losing crucial signal.
def check_remote_scripts_record_predicted_band(strict: bool = False, verbose: bool = False) -> list[str]:
    """Every scripts/remote_*.sh that emits provenance.json AND runs a
    LANE EXPERIMENT (not a bootstrap or sweep orchestrator) must include
    a 'predicted_band' field for empirical calibration.
    """
    # Exempt: bootstraps (utility, no per-experiment band), sweep orchestrators
    # (band depends on which trial wins), pure auth-eval reruns (diagnostic).
    EXEMPT_SUFFIXES = (
        "_bootstrap.sh", "_setup_full.sh", "_setup.sh",
        "_sweep.sh", "_optimized.sh",  # Bayesian sweep machinery
        "_auth_eval_only.sh",  # diagnostic rerun
    )
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [predicted-band] OK: scripts dir not found, skipped")
        return violations
    n_scripts = 0
    n_with_provenance = 0
    for script_path in sorted(scripts_dir.glob("remote_*.sh")):
        if any(script_path.name.endswith(suf) for suf in EXEMPT_SUFFIXES):
            continue
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        if "provenance.json" not in text and "PROVENANCE" not in text:
            continue  # script doesn't emit provenance, exempt
        n_with_provenance += 1
        if "predicted_band" not in text:
            violations.append(
                f"{script_path}: emits provenance.json but no "
                f"'predicted_band' metadata. Add to the python json.dump "
                f"block: `'predicted_band': [LOW, HIGH]`. Required for "
                f"council prediction calibration."
            )
    if verbose:
        if violations:
            print(f"  [predicted-band] {len(violations)}/{n_with_provenance} provenance-emitting script(s) missing predicted_band")
        else:
            print(f"  [predicted-band] OK: {n_with_provenance}/{n_scripts} remote_*.sh script(s) record predicted_band")
    if strict and violations:
        raise PreflightError(
            "PREDICTED BAND METADATA VIOLATIONS — at least one "
            f"scripts/remote_*.sh emits provenance but lacks predicted_band:\n  • "
            + "\n  • ".join(violations) +
            "\nFix: add 'predicted_band': [LOW, HIGH] to each provenance JSON "
            "for council calibration. (no-signal-loss CLAUDE.md rule.)"
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check N+3 (32nd meta-bug): remote scripts must tag completion [contest-CUDA]
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-27: per CLAUDE.md FORBIDDEN PATTERNS rule, every score must carry
# a lane tag (contest-CUDA / advisory / MPS-PROXY). Remote script completion
# logs are the canonical place for the tag. Currently checked only via
# per-script test files — generalize via preflight.
def check_remote_scripts_tag_contest_cuda_at_completion(strict: bool = False, verbose: bool = False) -> list[str]:
    """Every scripts/remote_*.sh that runs contest_auth_eval AND IS A
    LANE EXPERIMENT (not a bootstrap or sweep orchestrator) must include
    '[contest-CUDA]' literal in the completion log line so reports are
    self-tagging per CLAUDE.md score-tag rule.
    """
    EXEMPT_SUFFIXES = (
        "_bootstrap.sh", "_setup_full.sh", "_setup.sh",
        "_sweep.sh", "_optimized.sh",
        "_auth_eval_only.sh",
    )
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [completion-tag] OK: scripts dir not found, skipped")
        return violations
    n_scripts = 0
    n_with_eval = 0
    for script_path in sorted(scripts_dir.glob("remote_*.sh")):
        if any(script_path.name.endswith(suf) for suf in EXEMPT_SUFFIXES):
            continue
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        if "contest_auth_eval" not in text:
            continue
        n_with_eval += 1
        # Look for [contest-CUDA] tag literal anywhere in the script.
        if "[contest-CUDA]" not in text:
            violations.append(
                f"{script_path}: invokes contest_auth_eval but completion "
                f"log lacks '[contest-CUDA]' tag. Add the literal string "
                f"to the LANE_X_DONE log line so produced scores are "
                f"self-tagging per CLAUDE.md score-tag rule."
            )
    if verbose:
        if violations:
            print(f"  [completion-tag] {len(violations)}/{n_with_eval} eval script(s) missing [contest-CUDA] tag")
        else:
            print(f"  [completion-tag] OK: {n_with_eval}/{n_scripts} remote_*.sh script(s) tag completion")
    if strict and violations:
        raise PreflightError(
            "COMPLETION TAG VIOLATIONS — at least one scripts/remote_*.sh "
            f"runs contest_auth_eval but lacks '[contest-CUDA]' tag:\n  • "
            + "\n  • ".join(violations) +
            "\nFix: add '[contest-CUDA]' literal to the completion log line "
            "(LANE_X_DONE marker) so scores are self-tagging."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 33 (33rd meta-bug): remote scripts must NVDEC-probe at Stage 0
#                          BEFORE any GPU-spend operations
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: 7/12 overnight Vast.ai instances had compute-CUDA but
# missing NVDEC. The probe correctly catches this BUT only after Stage 4
# of setup_full.sh, which has already done apt + pip + DALI install
# (~5-10 min wasted). The probe MUST run at Stage 0 of every lane script
# so failures cost <30 seconds, not >5 minutes.
#
# This check verifies that every lane script's `bash $WORKSPACE/scripts/probe_nvdec.sh`
# call appears EARLY (before pip install / archive build / training).
def check_remote_scripts_probe_nvdec_early(strict: bool = False, verbose: bool = False) -> list[str]:
    """Every scripts/remote_lane_*.sh that does GPU work must call
    `bash $WORKSPACE/scripts/probe_nvdec.sh` BEFORE Stage 1 (training,
    archive build, mask extraction). NVDEC failures should cost <30s,
    not >5 min of wasted bootstrap.
    """
    EXEMPT_SUFFIXES = (
        "_bootstrap.sh", "_setup_full.sh", "_setup.sh",
        "_sweep.sh", "_optimized.sh", "_auth_eval_only.sh",
    )
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [nvdec-early] OK: scripts dir not found, skipped")
        return violations
    n_scripts = 0
    n_with_probe = 0
    for script_path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        if any(script_path.name.endswith(suf) for suf in EXEMPT_SUFFIXES):
            continue
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        # Strip comment-only lines so header docstrings don't false-positive
        # the GPU-marker scan. Use line-based filtering: lines starting with #.
        non_comment_lines = []
        char_offset = 0
        non_comment_text_chars = []
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # replace comment with same-length spaces to preserve indices
                non_comment_text_chars.append(" " * len(line))
            else:
                non_comment_text_chars.append(line)
        scan_text = "\n".join(non_comment_text_chars)
        # Find first probe_nvdec.sh call line and any GPU-cost line
        probe_idx = scan_text.find("probe_nvdec.sh")
        # GPU-cost markers: training launch, archive rebuild, mask extract
        # Match with $PYBIN/$PYTHON prefix or `python` to ensure it's an
        # executable invocation, not a doc reference.
        gpu_markers = [
            "experiments/train_renderer", "experiments/qat_finetune",
            "experiments/optimize_poses", "experiments/build_baseline_archive",
            "experiments/contest_auth_eval",
        ]
        first_gpu_idx = min(
            (scan_text.find(m) for m in gpu_markers if scan_text.find(m) >= 0),
            default=-1,
        )
        if probe_idx < 0:
            violations.append(
                f"{script_path}: no `probe_nvdec.sh` call. Add Stage 0 "
                f"NVDEC probe before any GPU-cost operation."
            )
            continue
        n_with_probe += 1
        if first_gpu_idx >= 0 and first_gpu_idx < probe_idx:
            violations.append(
                f"{script_path}: probe_nvdec.sh appears AFTER GPU-cost "
                f"command (probe@{probe_idx}, first GPU op@{first_gpu_idx}). "
                f"Move probe to Stage 0 BEFORE any GPU spend."
            )
    if verbose:
        if violations:
            print(f"  [nvdec-early] {len(violations)}/{n_scripts} script(s) violate early-probe rule")
        else:
            print(f"  [nvdec-early] OK: {n_with_probe}/{n_scripts} lane script(s) probe NVDEC at Stage 0")
    if strict and violations:
        raise PreflightError(
            "EARLY NVDEC PROBE VIOLATIONS — at least one lane script "
            f"doesn't probe NVDEC at Stage 0:\n  • " + "\n  • ".join(violations) +
            "\nFix: add `bash $WORKSPACE/scripts/probe_nvdec.sh || exit 2` "
            "BEFORE any train/qat/eval/archive command. Per memory "
            "feedback_vastai_nvdec_host_variation, NVDEC failure rate is "
            "~30-50% across host pools; early-probe saves $0.05-0.10 per bad host."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 34 (34th meta-bug): remote scripts that --resume-from a checkpoint
#                          must STATE_DICT-shape-validate the checkpoint
#                          against the profile-built model BEFORE GPU spend
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: Lane S overnight dispatch crashed at training launch with
# motion.head shape mismatch (Lane A renderer has 6-channel motion.head,
# Lane S profile builds 4-channel). The resume failed AFTER 5+ minutes
# of mask extraction + scorer cache build (~$0.05 wasted).
#
# This check looks for `--resume-from <path>` in lane scripts and ensures
# either:
#   (a) the script does a pre-flight shape validation BEFORE training launch
#       (e.g., `python -c "torch.load(...); model.load_state_dict(...)"`)
#   (b) the script uses the canonical resume-and-validate helper
#       `experiments/validate_resume_shapes.py` (TODO if doesn't exist)
def check_resume_from_state_dict_shape_compat(strict: bool = False, verbose: bool = False) -> list[str]:
    """Every lane script using `--resume-from <ckpt>` must shape-validate
    the checkpoint against the profile-built model BEFORE the heavy
    bootstrap (mask extraction, scorer cache).
    """
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [resume-shape] OK: scripts dir not found, skipped")
        return violations
    n_scripts = 0
    n_with_resume = 0
    for script_path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        text = script_path.read_text(errors="ignore")
        n_scripts += 1
        if "--resume-from" not in text:
            continue
        n_with_resume += 1
        # Look for any shape-validation marker:
        # - "load_state_dict" (inline pyc verification)
        # - "validate_resume_shapes" (canonical tool)
        # - "shape" + "validate" within 200 chars of --resume-from
        validation_markers = [
            "load_state_dict",
            "validate_resume_shapes",
            "validate_shape",
            "shape_compat",
            "_shape_check",
        ]
        has_validation = any(m in text for m in validation_markers)
        if not has_validation:
            violations.append(
                f"{script_path}: --resume-from present but no state_dict "
                f"shape validation. Add a pre-flight `python -c 'import torch; "
                f"torch.load(\"$RESUME_PATH\")'` + model.load_state_dict() "
                f"check BEFORE the heavy training launch. Lane S motion.head "
                f"6-vs-4 mismatch wasted ~$0.05 + 5 min when this was missing."
            )
    if verbose:
        if violations:
            print(f"  [resume-shape] {len(violations)}/{n_with_resume} resume-using script(s) lack shape validation")
        else:
            print(f"  [resume-shape] OK: {n_with_resume}/{n_scripts} lane script(s) shape-validate resumes")
    if strict and violations:
        raise PreflightError(
            "RESUME STATE_DICT SHAPE VALIDATION VIOLATIONS — at least one "
            f"lane script uses --resume-from but doesn't shape-validate:\n  • "
            + "\n  • ".join(violations) +
            "\nFix: add a pre-flight shape check before training launch. "
            "Lane S motion.head bug 2026-04-28."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 35 (35th meta-bug): scripts must NOT call `tmux kill-server`
#                          (kills OTHER lanes' tmux sessions on shared host)
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: I caught myself writing `tmux kill-server` in a quick_setup
# inline command — it would have killed any other lane's tmux session
# running on a shared Vast.ai instance. The canonical safe alternative
# is `tmux kill-session -t <session_name>` for the specific session, or
# just rely on `tmux new-session -d` to NOT clobber existing.
def check_no_tmux_kill_server_in_lane_scripts(strict: bool = False, verbose: bool = False) -> list[str]:
    """Scripts must NOT call ``tmux kill-server`` — kills ALL tmux
    sessions on the host, not just the lane's. Use
    ``tmux kill-session -t <name>`` instead, or rely on the absence
    of a same-named session.
    """
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    n_scripts = 0
    for script_path in sorted(scripts_dir.glob("remote_*.sh")):
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        non_comment_text = "\n".join(
            line if not line.lstrip().startswith("#") else " " * len(line)
            for line in text.split("\n")
        )
        if "tmux kill-server" in non_comment_text:
            violations.append(
                f"{script_path}: calls 'tmux kill-server' which kills ALL "
                f"tmux sessions on the host. Use 'tmux kill-session -t <name>' "
                f"for specific session, or rely on tmux new-session's existing-"
                f"session detection."
            )
    if verbose:
        if violations:
            print(f"  [no-tmux-kill-server] {len(violations)}/{n_scripts} script(s) violate")
        else:
            print(f"  [no-tmux-kill-server] OK: {n_scripts} script(s) clean")
    if strict and violations:
        raise PreflightError(
            "TMUX KILL-SERVER VIOLATIONS — at least one remote script "
            f"calls 'tmux kill-server':\n  • " + "\n  • ".join(violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 36 (36th meta-bug): scripts must NOT unconditionally call
#                          `python -m ensurepip --upgrade`
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: setup_full Stage 2 hit `subprocess.CalledProcessError` because
# the PyTorch container ships pip 26.x but ensurepip carries pip 24.0 wheels
# — ensurepip refuses to "upgrade" to an older version. The canonical fix
# is to skip ensurepip if pip is already importable.
def check_no_unconditional_ensurepip(strict: bool = False, verbose: bool = False) -> list[str]:
    """Scripts must guard ensurepip --upgrade with an `if ! python -c
    "import pip"` check, or skip ensurepip entirely on PyTorch containers
    that ship newer pip than the bundled wheels.
    """
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    n_scripts = 0
    for script_path in sorted(scripts_dir.glob("remote_*.sh")):
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        non_comment_text = "\n".join(
            line if not line.lstrip().startswith("#") else " " * len(line)
            for line in text.split("\n")
        )
        if "ensurepip" not in non_comment_text:
            continue
        # Find the line(s) containing ensurepip and check if guarded.
        # Look for `if ! ... pip` or `import pip` within 5 lines BEFORE.
        lines = non_comment_text.split("\n")
        for i, line in enumerate(lines):
            if "ensurepip" in line and "--upgrade" in line:
                window = "\n".join(lines[max(0, i-5):i+1])
                if "import pip" not in window and "if !" not in window:
                    violations.append(
                        f"{script_path}:{i+1}: unconditional 'ensurepip "
                        f"--upgrade'. Wrap with `if ! \"$PYBIN\" -c \"import "
                        f"pip\" 2>/dev/null; then ensurepip; fi`. The "
                        f"PyTorch container ships pip 26.x; bundled "
                        f"wheels (pip 24.0) trigger downgrade-refusal crash."
                    )
    if verbose:
        if violations:
            print(f"  [no-uncond-ensurepip] {len(violations)}/{n_scripts} script(s) violate")
        else:
            print(f"  [no-uncond-ensurepip] OK: {n_scripts} script(s) clean")
    if strict and violations:
        raise PreflightError(
            "UNCONDITIONAL ENSUREPIP VIOLATIONS — at least one script "
            f"calls ensurepip --upgrade without a pip-check guard:\n  • "
            + "\n  • ".join(violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 37 (37th meta-bug): lane scripts that contest_auth_eval must
#                          first remove macOS AppleDouble resource forks
#                          (`._*.mkv`) from upstream/videos to prevent
#                          contest-CUDA "extra-file contamination" failure.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-27: Lane F-V2 auth eval CRASHED at completion because the SCP'd
# tarball brought macOS `._0.mkv` AppleDouble files alongside `0.mkv`.
# `experiments/contest_auth_eval.py::_validate_uncompressed_dir` raises
# "uncompressed-dir contamination" with exit non-zero. Lost ~30s of GPU
# spend per occurrence + cognitive load to debug.
def check_lane_scripts_strip_macos_resource_forks(strict: bool = False, verbose: bool = False) -> list[str]:
    """Lane scripts running contest_auth_eval must ensure macOS AppleDouble
    files are purged from upstream/videos. Two valid patterns:
    (a) lane script does its own `rm -f upstream/videos/._*.mkv` before eval
    (b) setup_full.sh purges them once at bootstrap (and lane script depends
        on setup_full having been run via canonical bootstrap)
    """
    EXEMPT_SUFFIXES = (
        "_bootstrap.sh", "_setup_full.sh", "_setup.sh",
        "_sweep.sh", "_optimized.sh", "_auth_eval_only.sh",
    )
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    # Check if setup_full.sh has the canonical purge — if so, lane scripts
    # that follow the canonical bootstrap path are exempt.
    setup_full = scripts_dir / "remote_setup_full.sh"
    setup_full_purges = False
    if setup_full.exists():
        setup_text = setup_full.read_text(errors="ignore")
        setup_full_purges = (
            "find" in setup_text and "upstream/videos" in setup_text
            and "'._*'" in setup_text
        ) or "rm -f upstream/videos/._" in setup_text
    n_scripts = 0
    n_with_eval = 0
    for script_path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        if any(script_path.name.endswith(suf) for suf in EXEMPT_SUFFIXES):
            continue
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        if "contest_auth_eval" not in text:
            continue
        n_with_eval += 1
        # Look for `rm -f` of `._*.mkv` or equivalent before contest_auth_eval
        # OR a `find ... -name '._*'` cleanup. Permissive — accept any of:
        # - `rm -f upstream/videos/._*.mkv`
        # - `rm -f upstream/videos/._*`
        # - `find upstream/videos -name '._*' -delete`
        cleanup_markers = [
            "rm -f upstream/videos/._",
            "rm -f \"upstream/videos/._",
            "find upstream/videos -name '._",
            "find upstream/videos -name \"._",
            "find upstream/videos -type f -name '._",
        ]
        # If setup_full purges, AND this script sources env.sh / runs after
        # setup_full, accept that as satisfying the rule.
        depends_on_setup_full = "source" in text and "env.sh" in text
        if setup_full_purges and depends_on_setup_full:
            continue
        if not any(m in text for m in cleanup_markers):
            violations.append(
                f"{script_path}: invokes contest_auth_eval but doesn't strip "
                f"macOS AppleDouble files (._*.mkv) from upstream/videos. "
                f"Add `rm -f upstream/videos/._*.mkv` BEFORE the eval call to "
                f"prevent contamination-error crashes (Lane F-V2 bug 2026-04-27)."
            )
    if verbose:
        if violations:
            print(f"  [strip-resource-forks] {len(violations)}/{n_with_eval} eval script(s) lack ._* cleanup")
        else:
            print(f"  [strip-resource-forks] OK: {n_with_eval}/{n_scripts} script(s) strip macOS resource forks")
    if strict and violations:
        raise PreflightError(
            "macOS RESOURCE FORK CLEANUP VIOLATIONS — at least one lane "
            f"script invokes contest_auth_eval without ._* cleanup:\n  • "
            + "\n  • ".join(violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 38 (38th meta-bug): SSH commands in shell scripts must specify
#                          ConnectTimeout to prevent infinite hangs.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-27/28: SSH commands without ConnectTimeout can hang for 60+
# seconds when the host is briefly unreachable. In overnight wave loops,
# this stalls the parent agent + accumulates dead connections. Standard
# is `-o ConnectTimeout=10`.
def check_ssh_commands_have_connect_timeout(strict: bool = False, verbose: bool = False) -> list[str]:
    """Bash scripts using `ssh -o` for remote execution must specify
    `ConnectTimeout=N` to prevent indefinite hangs on bad hosts.
    """
    violations: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    n_scripts = 0
    n_with_ssh = 0
    for script_path in sorted(scripts_dir.glob("*.sh")):
        n_scripts += 1
        text = script_path.read_text(errors="ignore")
        non_comment_text = "\n".join(
            line if not line.lstrip().startswith("#") else " " * len(line)
            for line in text.split("\n")
        )
        # Look for `ssh ` invocations (remote execution) — but not in
        # `ssh-keygen`, `ssh-add`, etc.
        # Scan line by line for executable ssh calls
        lines = non_comment_text.split("\n")
        has_ssh_call = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # Match `ssh ` (with space) but not `sshd`, `ssh-keygen`, `ssh-add`,
            # or `ssh://`. Also accept `ssh\` (continuation).
            if "ssh " in line or stripped.endswith("ssh"):
                # Filter out common non-execution forms
                if any(m in line for m in ("ssh://", "sshd", "ssh-keygen", "ssh-add", "ssh-copy-id", "$SSH_", '"ssh"', "'ssh'")):
                    continue
                # Must be an ssh that includes flags or remote target
                if "@" not in line and "-p " not in line and "-o " not in line:
                    continue  # not a remote-exec form
                has_ssh_call = True
                if "ConnectTimeout" not in line:
                    # Check next 3 lines (for line continuations via \)
                    window = "\n".join(lines[i:min(i+3, len(lines))])
                    if "ConnectTimeout" not in window:
                        violations.append(
                            f"{script_path}:{i+1}: SSH command without "
                            f"`ConnectTimeout=N`. Add `-o ConnectTimeout=10` "
                            f"to prevent indefinite hangs on bad hosts."
                        )
        if has_ssh_call:
            n_with_ssh += 1
    if verbose:
        if violations:
            print(f"  [ssh-timeout] {len(violations)}/{n_with_ssh} ssh-using script(s) violate")
        else:
            print(f"  [ssh-timeout] OK: {n_with_ssh}/{n_scripts} script(s) use SSH timeouts")
    if strict and violations:
        raise PreflightError(
            "SSH CONNECT TIMEOUT VIOLATIONS — at least one script uses "
            f"`ssh` without ConnectTimeout:\n  • " + "\n  • ".join(violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 39 (2026-04-28): undeployed archive-artifact producers
#
# CATCHES the recurring "code-shipped-never-deployed" failure mode. Pattern:
# a tool exists at experiments/precompute_*.py or src/tac/*.py that writes a
# filename listed in submission_archive.py's artifact registry; it has a
# __main__ entry; it has tests; but no scripts/remote_lane_*.sh ever invokes
# it. Such a tool is dead code from the lab's perspective — burned engineering
# hours that never reach a Vast.ai score measurement.
#
# Concrete instances this would have caught:
#   - Lane EC engineered corrections — sat unused 2 weeks (Apr 14 → Apr 28).
#     33KB precompute_gradient_corrections.py + 60KB trick_stack.py
#     composition + 444-line test, all shipping `gradient_corrections.bin`,
#     never deployed until hand-flagged this session.
#   - Per project_outstanding_work_and_stacks_20260428: Lane Ω-V2, SI-V2,
#     LR-V2, LM-V2, MOS — same pattern, varying severity.
# ════════════════════════════════════════════════════════════════════════════

# Artifact filenames recognized as "submission archive output". Mirrored from
# submission_archive.py's required_files() mapping; if that registry changes,
# update both places. (A future hardening: parse the AST at runtime.)
_ARCHIVE_ARTIFACT_FILENAMES = frozenset({
    "renderer.bin",
    "masks.mkv",
    "masks.amrc",
    "optimized_poses.pt",
    "optimized_poses.bin",
    "optimized_embedding.pt",
    "poses.pt",
    "corrections.bin",
    "gradient_corrections.bin",
    "mini_segnet.bin",
    "mini_posenet.bin",
    "posenet_targets.bin",
    "zoom_scalars.bin",
    "foveation_params.bin",
})

# Producers we exempt from the "must be deployed" rule. These are either the
# registry itself, library helpers consumed inline by already-deployed
# pipelines, canonical entry points referenced through indirection
# (subprocess via deploy_vastai.py, pipeline.py, etc.) that grep wouldn't
# catch, or historically-dead lanes preserved for archeology. EVERY
# EXEMPTION needs a one-line WHY comment.
_DEPLOY_SCANNER_EXEMPT_PRODUCERS = frozenset({
    # Registry itself — it's the source of truth, not a producer
    "src/tac/submission_archive.py",
    # Renderer export is invoked from training scripts, never standalone
    "src/tac/renderer_export.py",
    # Pose TTO is invoked through pipeline.py compress + remote_pose_tto_bootstrap.sh
    "src/tac/optimize_poses.py",
    "experiments/optimize_poses.py",
    # Mask codec is invoked from compress.sh + canonical archive builders
    "src/tac/mask_codec.py",
    "src/tac/mask_entropy_coder.py",
    # AMRC lossless mask codec — invoked from compress.sh
    "src/tac/lossless/argmax_codec.py",
    # Pipeline is itself the orchestrator
    "experiments/pipeline.py",
    # Mini scorer training is the deployed lane's entry point itself
    "experiments/train_mini_scorer.py",
    # Library used by precompute_corrections.py, domain_solvers.py,
    # trick_stack.py — not a standalone tool
    "src/tac/scorer_targets.py",
    # ARCHEOLOGY: mini-scorer inflate path — strict-scorer-rule (CLAUDE.md)
    # forbids scorers at inflate time; mini-scorer lane is dead by policy.
    "experiments/mini_tto_inflate.py",
    # ARCHEOLOGY: embedding-loss TTO produced auth 0.61 on 2026-04-15 but was
    # superseded by pose TTO + KL distill collapse; preserved for reference.
    "experiments/optimize_embedding.py",
    # Canonical local E2E auth-eval smoke (Check 64). Mentions 'masks.amrc' in
    # its archive whitelist string but is itself a LOCAL preflight tool, not
    # a producer — it never writes the artifact, only validates archives that
    # contain it. Invoked by operators before lane dispatch + by Check 64.
    "experiments/canonical_local_auth_eval_smoke.py",
})

# Directory prefixes that run on alternative platforms (NOT Vast.ai), so
# absence from scripts/remote_lane_*.sh is expected.
_DEPLOY_SCANNER_EXEMPT_DIR_PREFIXES = (
    # Kaggle kernels run via `kaggle kernels push`, not via remote_lane scripts
    "experiments/kaggle_kernels/",
)


def _scan_repo_for_artifact_producers(
    artifact_name: str, repo_root: Path,
) -> list[Path]:
    """Find .py files that LIKELY write `artifact_name` to disk.

    Heuristic: file mentions the literal filename in source AND has at least
    one of {open(...,"wb"), torch.save, .write_bytes, np.save, pickle.dump,
    json.dump, zipfile.write*, brotli, gzip}. False positives (mentions in a
    docstring/comment) are rare and harmless — the deploy check filters them
    out by requiring __main__ + non-deployment.
    """
    producers: list[Path] = []
    write_markers = (
        "open(", "torch.save(", ".write_bytes(", "np.save(",
        "pickle.dump(", "json.dump(", "zipfile.", "brotli.", "gzip.",
        ".write(", "np.tofile(",
    )
    quoted_names = (f'"{artifact_name}"', f"'{artifact_name}'")
    for py in _iter_python_files(repo_root, ["src/tac", "experiments"]):
        # Skip tests + caches
        rel = py.relative_to(repo_root) if py.is_absolute() else py
        rel_s = str(rel)
        if "/tests/" in rel_s or "/__pycache__/" in rel_s:
            continue
        try:
            text = py.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        if not any(q in text for q in quoted_names):
            continue
        if not any(m in text for m in write_markers):
            continue
        producers.append(py)
    return producers


def _producer_has_main_entry(py: Path) -> bool:
    """True if file is a script (has __main__) — i.e., not a pure library."""
    try:
        text = py.read_text(errors="ignore")
    except (FileNotFoundError, PermissionError):
        return False
    return (
        'if __name__ == "__main__"' in text
        or "if __name__ == '__main__'" in text
    )


def _producer_is_deployed(
    py: Path, artifact: str, repo_root: Path,
) -> bool:
    """True if any deployment surface references the producer or its output.

    Three-signal OR across three deployment surfaces:
      1. scripts/remote_lane_*.sh — Vast.ai lane scripts
      2. scripts/remote_*_bootstrap.sh — canonical bootstraps (parameterized)
      3. src/tac/deploy/**/*.py — Vast.ai/Modal/Kaggle deployment registries

    For each surface we accept a match on producer's basename, producer's
    full repo path, or the artifact filename itself (covers inline producers
    like `python -c "..." > foo.bin`).
    """
    name = py.name
    rel_s = str(py.relative_to(repo_root) if py.is_absolute() else py)

    def _has_ref(t: str) -> bool:
        return name in t or rel_s in t or artifact in t

    scripts_dir = repo_root / "scripts"
    if scripts_dir.is_dir():
        for pattern in ("remote_lane_*.sh", "remote_*_bootstrap.sh"):
            for sh in scripts_dir.glob(pattern):
                try:
                    if _has_ref(sh.read_text(errors="ignore")):
                        return True
                except (FileNotFoundError, PermissionError):
                    continue

    # Surface 3: deploy registries — train_joint_pair.py is invoked
    # transparently through src/tac/deploy/vastai/experiments.py, etc.
    deploy_dir = repo_root / "src" / "tac" / "deploy"
    if deploy_dir.is_dir():
        for dp in deploy_dir.rglob("*.py"):
            try:
                if _has_ref(dp.read_text(errors="ignore")):
                    return True
            except (FileNotFoundError, PermissionError):
                continue
    return False


def check_undeployed_archive_artifact_producers(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch the 'code-shipped-never-deployed' bug class.

    For every filename registered in submission_archive.py's artifact mapping,
    find tools that produce it (write the filename to disk) and have a
    __main__ entry. If none of those tools is referenced by any
    scripts/remote_lane_*.sh (or remote_*_bootstrap.sh), we have a never-
    deployed lane — engineering hours that never produce a measured score.

    Reference: project_lane_ec_engineered_corrections_20260428 (sat 2 weeks
    unused). Reference: project_outstanding_work_and_stacks_20260428 TIER 3.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    seen_producers: set[Path] = set()

    for artifact in sorted(_ARCHIVE_ARTIFACT_FILENAMES):
        for py in _scan_repo_for_artifact_producers(artifact, root):
            if py in seen_producers:
                continue
            seen_producers.add(py)
            rel_s = str(py.relative_to(root) if py.is_absolute() else py)
            if rel_s in _DEPLOY_SCANNER_EXEMPT_PRODUCERS:
                continue
            if any(rel_s.startswith(p) for p in _DEPLOY_SCANNER_EXEMPT_DIR_PREFIXES):
                continue  # alternative-platform producer (e.g., Kaggle)
            if not _producer_has_main_entry(py):
                continue  # pure library — OK
            if _producer_is_deployed(py, artifact, root):
                continue
            violations.append(
                f"{rel_s}: writes '{artifact}' via __main__ entry but no "
                f"scripts/remote_lane_*.sh (or remote_*_bootstrap.sh) "
                f"invokes it. This is the 'code-shipped-never-deployed' "
                f"pattern (Lane EC sat unused 2 weeks). Either: (a) add a "
                f"remote_lane_*.sh that runs it; (b) add the file path to "
                f"_DEPLOY_SCANNER_EXEMPT_PRODUCERS in preflight.py with a "
                f"WHY comment if the producer is library-only or invoked "
                f"through indirection."
            )

    if verbose:
        if violations:
            print(f"  [undeployed-producers] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print("  [undeployed-producers] OK: every artifact-producer __main__ has a remote_lane_*.sh invocation")

    if violations and strict:
        raise MetaBugViolation(
            "UNDEPLOYED ARCHIVE-ARTIFACT PRODUCERS:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 40 (2026-04-28): hardware-quantization capability disclosure
#
# CATCHES the bug class that destroyed Lane F lineage: emitting FP4 archives
# / running FakeQuantFP4 in production code paths WITHOUT disclosing that
# FP4 hardware acceleration requires Blackwell (CC 10.0) and our reference
# 4090 hardware (CC 8.9) only supports SIMULATED FP4 via FakeQuantFP4.
#
# Memory ref: project_cosmos_deep_dive_addendum_20260428.
# Lane F V1=2.73, V2=1.79, V3=1.85 all generated by FakeQuantFP4 simulation
# with NO hardware backing. The "FP4 architecturally hostile" conclusion
# was unverifiable — could be simulation noise, not architectural. FP8 IS
# hardware-supported on 4090 via torchao.float8 (Lane F-V5 rescue path).
# ════════════════════════════════════════════════════════════════════════════

# Files that emit FP4 archives or use FakeQuantFP4 in production paths
# (not tests, not docs). Either each must add a disclosure marker, or be
# exempted here with a WHY comment.
_FP4_DISCLOSURE_EXEMPT = frozenset({
    # Library that defines the simulation primitive itself; the simulation
    # IS the unit of the docstring there.
    "src/tac/quantization.py",
    # Library export of FP4A format; format definition not a runtime path.
    "src/tac/renderer_export.py",
})


def _scan_for_fp4_production_paths(repo_root: Path) -> list[str]:
    """Scan for FP4 production paths missing hardware-disclosure markers.

    A "production path" is a non-test .py file under src/tac/ or experiments/
    that ACTUALLY INSTANTIATES quantization (not just reads/validates the
    archive format). Detection signals:
      (a) constructor call `FakeQuantFP4(...)` (instantiation, not import), OR
      (b) function call `fake_quant_fp4(...)` (lowercase apply form)
    AND does NOT contain a hardware-disclosure marker:
      - "[SIMULATED-FP4]" string literal
      - "[ADVISORY-FP4]" string literal
      - "compute_capability" reference (any form)
      - "get_device_capability" reference
      - "assert_quantization_hardware_supported" reference
      - "# FP4_HARDWARE_DISCLOSED:" comment marker

    Reading FP4A magic bytes (loaders/validators/registries) does NOT count
    as a production path; the magic-byte check is a passive format detection
    that doesn't make hardware-FP4 claims.
    """
    violations: list[str] = []
    # Constructor-call patterns indicating actual quantization instantiation
    # (regex-aware): `FakeQuantFP4(`, `FakeQuantFP4.apply(`, `fake_quant_fp4(`
    instantiation_re = re.compile(
        r"\b(?:FakeQuantFP4\s*\(|FakeQuantFP4\.apply\s*\(|fake_quant_fp4\s*\()"
    )
    disclosure_markers = (
        "[SIMULATED-FP4]",
        "[ADVISORY-FP4]",
        "compute_capability",
        "get_device_capability",
        "assert_quantization_hardware_supported",
        "# FP4_HARDWARE_DISCLOSED:",
    )
    for py in _iter_python_files(repo_root, ["src/tac", "experiments"]):
        rel = py.relative_to(repo_root) if py.is_absolute() else py
        rel_s = str(rel)
        if "/tests/" in rel_s or "/__pycache__/" in rel_s:
            continue
        if rel_s in _FP4_DISCLOSURE_EXEMPT:
            continue
        try:
            text = py.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        if not instantiation_re.search(text):
            continue
        if any(m in text for m in disclosure_markers):
            continue
        violations.append(
            f"{rel_s}: instantiates FakeQuantFP4 in a production path "
            f"without a hardware-disclosure marker. FP4 hardware "
            f"acceleration requires Blackwell (CC 10.0); 4090 (CC 8.9) "
            f"only supports simulated FP4. Either: (a) add a runtime print "
            f"'[SIMULATED-FP4] hardware capability < 10.0 — FP4 is "
            f"simulated via FakeQuantFP4'; (b) add `# FP4_HARDWARE_DISCLOSED: "
            f"<reason>` comment near the FakeQuantFP4 call; (c) call "
            f"`assert_quantization_hardware_supported('fp4', device, "
            f"allow_simulation=True)` from tac.quantization. Reference: "
            f"project_cosmos_deep_dive_addendum_20260428."
        )
    return violations


def check_fp4_production_paths_disclose_hardware(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch undeclared simulated-FP4 production paths.

    Reference: project_cosmos_deep_dive_addendum_20260428 (4090 is CC 8.9,
    NVFP4 needs CC 10.0; Lane F results were all simulated FakeQuantFP4
    with no hardware backing).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations = _scan_for_fp4_production_paths(root)

    if verbose:
        if violations:
            print(f"  [fp4-hw-disclose] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print("  [fp4-hw-disclose] OK: every FP4 production path discloses hardware reality")

    if violations and strict:
        raise MetaBugViolation(
            "FP4 PRODUCTION PATHS MISSING HARDWARE DISCLOSURE:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 42 (2026-04-28): train/inference pose-projection parity (BUG-1 class)
#
# CATCHES the bug class found by Lane M-V2 audit (memory:
# project_lane_m_v2_audit_council_findings_20260428): a pose-projection
# helper used at OPTIMIZATION time but NOT at INFLATE time, so the optimizer
# solves a different problem than what gets evaluated. The Lane M-V2 case:
#
#   optimize_poses.py: _project_to_renderer_pose(cond) → [zoom, 0,0,0,0,0]
#   inflate_renderer: <not called>; uses raw saved tensor → [zoom, baseline]
#
# Optimizer was driving a model conditioned on zero-pad; inflate evaluated
# with frozen-baseline-pad. The 0.076 PoseNet result was signal of the bug,
# not of the architectural premise. ~$1.50 + 5h GPU wasted before audit.
#
# This check enforces: any pose-projection helper (regex
# `_project.*pose|project_pose`) defined in experiments/ must EITHER be
# called from submissions/robust_current/inflate_renderer.py OR have an
# explicit `# PROJECT_PARITY_WAIVED:<reason>` marker near its definition.
# ════════════════════════════════════════════════════════════════════════════


def _scan_pose_projection_helpers(repo_root: Path) -> list[tuple[str, int]]:
    """Find pose-projection helper definitions in optimize/training scripts.

    Returns list of (file_path, lineno) where a candidate helper is defined.
    Pattern: `def _project*pose*` or `def project_*_pose` or `def *_pose_pad*`.
    """
    helpers: list[tuple[str, int]] = []
    pattern = re.compile(
        r"^def\s+(_?project_\w*pose\w*|project_\w*_pose|_?\w*_pose_pad\w*)\s*\(",
    )
    for py in _iter_python_files(repo_root, ["experiments", "src/tac"]):
        rel = py.relative_to(repo_root) if py.is_absolute() else py
        rel_s = str(rel)
        if "/tests/" in rel_s or "/inflate_renderer.py" in rel_s:
            continue
        try:
            text = py.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = pattern.match(line.lstrip())
            if m:
                helpers.append((rel_s, i))
    return helpers


def _inflate_calls_helper(helper_name: str, repo_root: Path) -> bool:
    """True iff inflate_renderer.py calls a function matching `helper_name`."""
    inflate_path = repo_root / "submissions" / "robust_current" / "inflate_renderer.py"
    if not inflate_path.exists():
        return False
    try:
        text = inflate_path.read_text(errors="ignore")
    except (FileNotFoundError, PermissionError):
        return False
    # Match either direct call `helper_name(` or import `from X import helper_name`
    return bool(re.search(rf"\b{re.escape(helper_name)}\s*\(", text)) or bool(
        re.search(rf"\bimport\s+\w+\s*,?\s*{re.escape(helper_name)}", text)
    ) or bool(re.search(rf"from\s+\S+\s+import.*\b{re.escape(helper_name)}", text))


def _has_parity_waiver(file_path: Path, def_lineno: int) -> bool:
    """Look for `# PROJECT_PARITY_WAIVED:` marker within 15 lines of def.

    Window is 15 because waiver comments often span multiple lines for
    explanation (e.g., the BUG-1 waiver at optimize_poses.py:752 needs
    7+ lines to reference the audit + V3-clean fix path).
    """
    try:
        lines = file_path.read_text(errors="ignore").splitlines()
    except (FileNotFoundError, PermissionError):
        return False
    start = max(0, def_lineno - 15)
    end = min(len(lines), def_lineno + 6)
    return any("PROJECT_PARITY_WAIVED:" in line for line in lines[start:end])


# ════════════════════════════════════════════════════════════════════════════
# Check 43 (2026-04-28): launcher tarball must include lane anchor paths
# ════════════════════════════════════════════════════════════════════════════


def check_launcher_tarball_includes_lane_anchors(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch the bug class where lane scripts reference anchor files that
    are EXCLUDED from the launcher's tarball.

    Reference: 2026-04-28 PM, 3 lanes (Ω-V2, EC, SAUG-V2) launched OK via
    launcher V4 split-mode but FAILED on remote because tarball excluded
    `experiments/results/lane_a_landed/` (3.4GB) — losing the canonical
    700KB `archive_lane_a.zip` anchor that all lanes reference. ~$1.50
    wasted across 3 destroyed instances.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []

    scripts_dir = root / "scripts"
    launcher = root / "scripts" / "launch_lane_on_vastai.py"
    if not scripts_dir.is_dir() or not launcher.exists():
        if verbose:
            print(f"  [tarball-anchor-parity] OK: launcher or scripts dir missing — skipping")
        return violations

    # Collect anchor paths referenced in remote_lane_*.sh
    anchor_paths: set[str] = set()
    pattern = re.compile(
        r'(?:ANCHOR_\w+|LANE_\w*ARCHIVE\w*|LANE_\w*POSES\w*|LANE_\w*MASKS\w*|LANE_\w*RENDERER\w*)='
        r'(?:"|\$\{[^:}]+:-)?(experiments/results/[\w./_-]+)'
    )
    for sh in scripts_dir.glob("remote_lane_*.sh"):
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        for m in pattern.finditer(text):
            anchor_paths.add(m.group(1))

    if not anchor_paths:
        if verbose:
            print(f"  [tarball-anchor-parity] OK: no anchor paths to check")
        return violations

    # Parse launcher includes + excludes
    try:
        ltext = launcher.read_text(errors="ignore")
    except (FileNotFoundError, PermissionError):
        return violations

    includes: set[str] = set()
    excludes: list[str] = []
    for line in ltext.splitlines():
        s = line.strip()
        m_ex = re.match(r'"--exclude=([^"]+)"', s)
        if m_ex:
            excludes.append(m_ex.group(1))
            continue
        m_inc = re.match(r'"(experiments/[^"]+)",?$', s)
        if m_inc:
            includes.add(m_inc.group(1))

    for ap in sorted(anchor_paths):
        if ap in includes:
            continue
        # If any exclude pattern would match the anchor path → violation
        # (unless an include exactly overrides)
        excluded = False
        for ex in excludes:
            ex_clean = ex.rstrip("*").rstrip("/")
            if not ex_clean:
                continue
            if ap.startswith(ex_clean):
                if any(ap == inc or ap.startswith(inc.rstrip("/") + "/") for inc in includes):
                    continue
                excluded = True
                break
        if excluded:
            violations.append(
                f"{ap}: referenced as anchor in scripts/remote_lane_*.sh but "
                f"EXCLUDED from launcher tarball. Lanes deployed via "
                f"scripts/launch_lane_on_vastai.py will FAIL on remote. "
                f"Add to includes list in `build_tarball()` OR remove the "
                f"parent --exclude pattern."
            )

    if verbose:
        if violations:
            print(f"  [tarball-anchor-parity] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [tarball-anchor-parity] OK: {len(anchor_paths)} anchor path(s) all in tarball")

    if violations and strict:
        raise MetaBugViolation(
            "LAUNCHER TARBALL MISSING LANE ANCHORS:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


def check_lane_anchor_files_exist_locally(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Check 69 — every `ANCHOR_*` / `LANE_*_ARCHIVE` path referenced in
    `remote_lane_*.sh` must EXIST in the local working tree.

    Check 43 verifies launcher tarball INCLUDES the path. This check is
    complementary: if the file doesn't exist locally, the tarball ships
    nothing, the lane crashes on remote with `[ -f "$ANCHOR_..." ]` failure.

    Skips paths that are env-overridable to placeholders (ANCHOR_FOO=${BAR:-})
    when no resolvable default exists in the file.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [anchor-exists-locally] OK: scripts/ missing — skipping")
        return []

    pattern = re.compile(
        r'(?:ANCHOR_\w+|LANE_\w*ARCHIVE\w*|LANE_\w*POSES\w*|LANE_\w*MASKS\w*|LANE_\w*RENDERER\w*)='
        r'(?:"|\$\{[^:}]+:-)?(experiments/results/[\w./_-]+|submissions/[\w./_-]+|upstream/[\w./_-]+)'
    )

    violations: list[str] = []
    n_checked = 0
    for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        for m in pattern.finditer(text):
            anchor_path = m.group(1)
            n_checked += 1
            full = root / anchor_path
            if not full.exists():
                violations.append(
                    f"{sh.relative_to(root)}: ANCHOR `{anchor_path}` does NOT "
                    f"exist locally — launcher tarball will ship nothing, "
                    f"lane will crash at `[ -f $ANCHOR_... ]` check on remote."
                )

    if verbose:
        if violations:
            print(f"  [anchor-exists-locally] {len(violations)} violation(s) "
                  f"({n_checked} anchor refs scanned):")
            for v in violations[:20]:
                print(f"    • {v}")
        else:
            print(f"  [anchor-exists-locally] OK: {n_checked} anchor refs all exist locally")

    if violations and strict:
        raise MetaBugViolation(
            "LANE ANCHOR FILES DO NOT EXIST LOCALLY:\n"
            + "\n".join(f"  • {v}" for v in violations[:20])
        )
    return violations


def check_python_files_compile(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    scan_dirs: tuple[str, ...] = ("src/tac", "scripts", "experiments", "tools"),
) -> list[str]:
    """Check 67 — every `.py` file in `scan_dirs` must parse + compile.

    PROACTIVE: catches SyntaxError + IndentationError + obvious typos
    BEFORE they ship to a remote and crash the lane after 5 minutes of
    deploy. Uses `py_compile.compile(doraise=True)` which exercises the
    full grammar without importing the module (so no import side-effects).

    2026-04-29: added per user demand "preflight needs to include a python
    compile step of all so we can identify any python errors without
    deploying" + "autodetect and permanently prevent all bugs possible
    to anticipate".

    Skips: __pycache__, .venv, .git, .pytest_cache, build/, dist/, node_modules.
    """
    import py_compile

    root = repo_root or REPO_ROOT
    skip_parts = {"__pycache__", ".venv", ".git", ".pytest_cache",
                  "build", "dist", "node_modules", ".eggs"}

    violations: list[str] = []
    n_compiled = 0
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for py in d_path.rglob("*.py"):
            if any(p in skip_parts for p in py.parts):
                continue
            try:
                py_compile.compile(str(py), doraise=True)
                n_compiled += 1
            except py_compile.PyCompileError as e:
                violations.append(
                    f"{py.relative_to(root)}: {type(e).__name__}: "
                    f"{str(e).strip()[:200]}"
                )
            except (SyntaxError, IndentationError) as e:
                violations.append(
                    f"{py.relative_to(root)}: {type(e).__name__} at "
                    f"line {e.lineno}: {e.msg}"
                )
            except Exception as e:  # pragma: no cover  unexpected
                violations.append(
                    f"{py.relative_to(root)}: {type(e).__name__}: {e}"
                )

    if verbose:
        if violations:
            print(f"  [python-compile] {len(violations)} violation(s) "
                  f"({n_compiled} files compiled OK):")
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    ... and {len(violations) - 20} more")
        else:
            print(f"  [python-compile] OK: {n_compiled} files compile clean")

    if violations and strict:
        raise MetaBugViolation(
            "PYTHON FILES FAIL TO COMPILE — would crash on import at deploy:\n"
            + "\n".join(f"  • {v}" for v in violations[:20])
            + (f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else "")
        )
    return violations


def check_shell_scripts_syntax_clean(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    scan_dirs: tuple[str, ...] = ("scripts", "submissions", "experiments", "tools"),
) -> list[str]:
    """Check 68 — every `*.sh` file in `scan_dirs` must pass `bash -n`.

    PROACTIVE bash syntax check (no execution). Catches unclosed quotes,
    bad heredocs, unmatched braces — bugs that would otherwise crash 30s
    into a remote deploy.

    Skips: directories that happen to end in .sh (recovered_*.sh, etc.)
    """
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if not bash:
        if verbose:
            print(f"  [shell-syntax] SKIP: bash not found on PATH")
        return []

    root = repo_root or REPO_ROOT
    skip_parts = {"__pycache__", ".venv", ".git", ".pytest_cache",
                  "build", "dist", "node_modules"}

    violations: list[str] = []
    n_checked = 0
    for d in scan_dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        for sh in d_path.rglob("*.sh"):
            if sh.is_dir() or not sh.is_file():
                continue
            if any(p in skip_parts for p in sh.parts):
                continue
            try:
                proc = subprocess.run(
                    [bash, "-n", str(sh)],
                    capture_output=True, text=True, timeout=10,
                )
                n_checked += 1
                if proc.returncode != 0:
                    err = proc.stderr.strip().splitlines()
                    msg = err[0] if err else f"non-zero exit {proc.returncode}"
                    violations.append(
                        f"{sh.relative_to(root)}: bash syntax error: {msg[:200]}"
                    )
            except subprocess.TimeoutExpired:
                violations.append(
                    f"{sh.relative_to(root)}: bash -n timed out (10s)"
                )
            except Exception as e:  # pragma: no cover
                violations.append(
                    f"{sh.relative_to(root)}: {type(e).__name__}: {e}"
                )

    if verbose:
        if violations:
            print(f"  [shell-syntax] {len(violations)} violation(s) "
                  f"({n_checked} scripts checked):")
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    ... and {len(violations) - 20} more")
        else:
            print(f"  [shell-syntax] OK: {n_checked} scripts pass `bash -n`")

    if violations and strict:
        raise MetaBugViolation(
            "SHELL SCRIPTS FAIL `bash -n` SYNTAX CHECK:\n"
            + "\n".join(f"  • {v}" for v in violations[:20])
        )
    return violations


def check_no_git_reset_hard_in_remote_lane_scripts(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Check 44 — `git reset --hard origin/main` in remote_lane_*.sh wipes
    local-only anchor files (archive_lane_a.zip, baseline dirs, etc.) that
    the launcher just SCP'd. The tarball IS the parity mechanism — never
    re-sync from origin/main on the remote.

    2026-04-29 incident: 5/6 TIER-1 lanes crashed at Stage 1 with
    "FATAL: missing Lane G v3 anchor archive" because canonical git-sync
    pattern (introduced today) ran `git reset --hard origin/main` after
    extract, deleting the local-only anchor archives the launcher had
    bundled. ~$1.50 wasted, 0 training output.

    Detects executable `git fetch ... && git reset --hard ...` lines
    (ignores comments). Returns violations; raises MetaBugViolation if strict.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [no-git-reset-hard] OK: scripts/ missing — skipping")
        return []

    violations: list[str] = []
    for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Match executable `git reset --hard` (not in comments)
            if re.search(r"\bgit\s+reset\s+--hard\b", line):
                violations.append(
                    f"{sh.relative_to(root)}:{lineno}: executable `git reset --hard` "
                    f"wipes local-only anchor files SCP'd by launcher. "
                    f"Trust the tarball — remove this line."
                )

    if verbose:
        if violations:
            print(f"  [no-git-reset-hard] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [no-git-reset-hard] OK: 0 lane scripts run `git reset --hard`")

    if violations and strict:
        raise MetaBugViolation(
            "LANE SCRIPTS RUNNING `git reset --hard` WILL WIPE LOCAL-ONLY ANCHORS:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nRemove the `git fetch + git reset --hard` block from each script. "
            "The launcher tarball is the canonical parity mechanism."
        )
    return violations


def check_pose_projection_train_inference_parity(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch pose-projection helpers used asymmetrically (BUG-1 class).

    Reference: project_lane_m_v2_audit_council_findings_20260428.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    helpers = _scan_pose_projection_helpers(root)
    for rel_s, lineno in helpers:
        # Extract helper name from def line
        try:
            text = (root / rel_s).read_text(errors="ignore")
            line = text.splitlines()[lineno - 1].lstrip()
            m = re.match(r"def\s+(\w+)\s*\(", line)
            if not m:
                continue
            helper_name = m.group(1)
        except (FileNotFoundError, PermissionError, IndexError):
            continue
        if _has_parity_waiver(root / rel_s, lineno):
            continue
        if _inflate_calls_helper(helper_name, root):
            continue
        violations.append(
            f"{rel_s}:{lineno}: pose-projection helper `{helper_name}` is "
            f"defined in an optimization script but never called from "
            f"submissions/robust_current/inflate_renderer.py. This is the "
            f"BUG-1 class from Lane M-V2 audit "
            f"(project_lane_m_v2_audit_council_findings_20260428): the "
            f"optimizer projects pose tensors one way, inflate evaluates "
            f"with raw saved tensors → train/inference distribution mismatch. "
            f"Either: (a) call the same helper from inflate_renderer.py to "
            f"ensure parity; (b) add `# PROJECT_PARITY_WAIVED: <reason>` "
            f"comment near the def if the helper is intentionally one-sided."
        )

    if verbose:
        if violations:
            print(f"  [pose-parity] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [pose-parity] OK: every pose-projection helper has parity or waiver")

    if violations and strict:
        raise MetaBugViolation(
            "POSE-PROJECTION TRAIN/INFERENCE PARITY VIOLATIONS:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 41 (2026-04-28): remote_lane_*.sh scripts must have heartbeat loop
#
# CATCHES the bug class that wasted ~$2.50 on 2026-04-28: 3 Vast.ai instances
# (W Iceland 35739770, K Denmark 35739771, OS-V2 NC 35739773) where SSH +
# repo clone succeeded but the lane script never invoked, leaving no
# heartbeat.log on disk and no GPU activity. The launcher reported success
# because clone completed, masking the actual non-execution.
#
# Memory ref: feedback_vastai_launch_returns_success_before_lane_starts.
#
# This check enforces that every remote_lane_*.sh script:
#   (a) defines HEARTBEAT (or LOG_DIR + heartbeat.log path), AND
#   (b) writes to that path in a backgrounded loop
# So a watchdog (or future post-launch verifier) can poll the on-disk
# heartbeat freshness as the canonical readiness signal.
#
# Sweep orchestrators (`*_sweep.sh`) are exempt because they delegate to
# per-trial scripts that have their own heartbeats.
# ════════════════════════════════════════════════════════════════════════════

# Sweep / orchestrator scripts that delegate heartbeat to sub-scripts
_HEARTBEAT_EXEMPT_SUFFIXES = (
    "_sweep.sh",
    # Lane A-Sweep template + orchestrator (per file docstring)
    "remote_lane_a_optimized.sh",
)


def _scan_remote_lane_scripts_missing_heartbeat(repo_root: Path) -> list[str]:
    """Scan remote_lane_*.sh for missing heartbeat-write pattern.

    Required pattern: file mentions 'heartbeat' (case-insensitive) AND has
    one of:
      - `>> "$HEARTBEAT"` (canonical pattern)
      - `>> $HEARTBEAT`
      - `>> "$LOG_DIR/heartbeat.log"`
      - any `heartbeat.log` write

    Sweep orchestrators are exempted via _HEARTBEAT_EXEMPT_SUFFIXES.
    """
    violations: list[str] = []
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    write_patterns = (
        '>> "$HEARTBEAT"',
        ">> $HEARTBEAT",
        '>> "$LOG_DIR/heartbeat.log"',
        ">> heartbeat.log",
        '"heartbeat.log"',
        "'heartbeat.log'",
    )
    for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
        name = sh.name
        if any(name.endswith(suf) for suf in _HEARTBEAT_EXEMPT_SUFFIXES):
            continue
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        if "heartbeat" not in text.lower():
            violations.append(
                f"{sh.relative_to(repo_root)}: no `heartbeat` reference. "
                f"Lane scripts MUST write a heartbeat.log so the launcher "
                f"and watchdog can verify the lane actually started "
                f"(memory: feedback_vastai_launch_returns_success_before_lane_starts). "
                f"Use the canonical pattern from "
                f"scripts/remote_lane_lm_zero_cost_poses.sh: "
                f"`HEARTBEAT=\"$LOG_DIR/heartbeat.log\"` + a backgrounded "
                f"`while true; do echo ... >> \"$HEARTBEAT\"; sleep 60; done &` "
                f"loop. If this is an orchestrator that delegates heartbeat "
                f"to per-trial sub-scripts, add the basename to "
                f"_HEARTBEAT_EXEMPT_SUFFIXES with a WHY comment."
            )
            continue
        if not any(p in text for p in write_patterns):
            violations.append(
                f"{sh.relative_to(repo_root)}: mentions 'heartbeat' but no "
                f"actual heartbeat-write pattern detected (expected one of: "
                f"`>> \"$HEARTBEAT\"`, `>> $HEARTBEAT`, `>> \"$LOG_DIR/heartbeat.log\"`, "
                f"or any `heartbeat.log` write). Add the canonical write "
                f"loop or update _HEARTBEAT_EXEMPT_SUFFIXES."
            )
    return violations


def check_remote_lane_scripts_have_heartbeat(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch lane scripts missing heartbeat-write pattern.

    Reference: feedback_vastai_launch_returns_success_before_lane_starts.
    Lane W/K/OS-V2 (2026-04-28) silently never started despite SSH + clone
    success, wasting ~$2.50. Heartbeat.log freshness is the only ground-
    truth readiness signal.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations = _scan_remote_lane_scripts_missing_heartbeat(root)

    if verbose:
        if violations:
            print(f"  [lane-heartbeat] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print("  [lane-heartbeat] OK: every remote_lane_*.sh writes a heartbeat (or is sweep-exempt)")

    if violations and strict:
        raise MetaBugViolation(
            "REMOTE LANE SCRIPTS MISSING HEARTBEAT:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 43 (2026-04-29): new remote_lane_*.sh scripts need controlled baseline
#
# Tuna-2 methodology: every new lane should identify a minimal-change
# controlled baseline so a negative/positive result isolates one mechanism.
# Scope is intentionally date-gated to scripts added/modified after
# 2026-04-29. For tracked files we ask git for the latest followed commit; for
# untracked or temp-repo tests we fall back to file mtime.
# ════════════════════════════════════════════════════════════════════════════

_CONTROLLED_BASELINE_CUTOFF = _dt.datetime(2026, 4, 29, tzinfo=_dt.timezone.utc)


def _parse_git_iso_datetime(text: str) -> _dt.datetime | None:
    text = text.strip()
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _remote_lane_script_changed_after_cutoff(
    sh: Path,
    repo_root: Path,
    cutoff: _dt.datetime,
) -> bool:
    rel = sh.relative_to(repo_root)
    changed_at = None
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-1",
                "--follow",
                "--format=%aI",
                "--",
                str(rel),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            changed_at = _parse_git_iso_datetime(proc.stdout)
    except (OSError, subprocess.SubprocessError):
        changed_at = None

    if changed_at is None:
        try:
            changed_at = _dt.datetime.fromtimestamp(
                sh.stat().st_mtime,
                tz=_dt.timezone.utc,
            )
        except FileNotFoundError:
            return False
    if changed_at.tzinfo is None:
        changed_at = changed_at.replace(tzinfo=_dt.timezone.utc)
    return changed_at > cutoff


def _scan_remote_lane_scripts_missing_controlled_baseline(
    repo_root: Path,
    cutoff: _dt.datetime = _CONTROLLED_BASELINE_CUTOFF,
) -> list[str]:
    violations: list[str] = []
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
        if not _remote_lane_script_changed_after_cutoff(sh, repo_root, cutoff):
            continue
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        if "controlled_baseline" in text:
            continue
        violations.append(
            f"{sh.relative_to(repo_root)}: missing `controlled_baseline` "
            f"metadata. New Tuna-2 remote lane scripts added/modified after "
            f"2026-04-29 should name a minimal-change controlled baseline "
            f"(docs/lane_methodology.md) so lane comparisons isolate one "
            f"mechanism."
        )
    return violations


def check_remote_lane_scripts_have_controlled_baseline(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Warn when future remote lane scripts omit controlled_baseline metadata."""
    root = repo_root or REPO_ROOT
    violations = _scan_remote_lane_scripts_missing_controlled_baseline(root)

    if verbose:
        if violations:
            print(f"  [controlled-baseline] {len(violations)} warning(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                "  [controlled-baseline] OK: qualifying remote_lane_*.sh "
                "scripts declare controlled_baseline"
            )

    if violations and strict:
        raise MetaBugViolation(
            "REMOTE LANE SCRIPTS MISSING CONTROLLED BASELINE:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 44 (2026-04-28): autograd.Function backward tests must check grad
#                        DIRECTION / VALUE, not just `grad is not None`.
#
# Round 22 review meta-bug: the bit-STE Round 12/13/14/18 reviews silently
# passed because the only assertion on `bits.grad` was
# `assert bits.grad is not None`. A SIGN bug (positive grad pushing bits
# down instead of up) hid for 4 review rounds before Round 21 finally caught
# it. The CLAUDE.md anti-arbitrariness rule: gradient-correctness tests must
# pin a number, a sign, or a comparison to a reference — finiteness is not
# a correctness gate.
#
# This check scans `src/tac/tests/test_*.py`. For each test that mentions
# any class extending `torch.autograd.Function`, we require the test to
# also assert at least one of:
#   - a numeric value on `.grad` (e.g., `pytest.approx(-0.04, ...)`)
#   - a sign / comparison on `.grad` (e.g., `grad < 0`, `grad.item() > 0`)
#   - a tensor comparison (e.g., `torch.allclose(grad, expected)`)
#
# Same-line waiver:
#   `# GRADIENT_DIRECTION_NOT_REQUIRED:<reason>`
# ════════════════════════════════════════════════════════════════════════════

_GRAD_DIRECTION_WAIVER_TOKEN = "GRADIENT_DIRECTION_NOT_REQUIRED:"

# Patterns that indicate a real gradient-direction / value assertion.
# We keep this conservative: any of these substrings in the same test
# function as a `.grad` reference satisfies the gate.
_GRAD_DIRECTION_PATTERNS = (
    "pytest.approx",
    "approx(",  # e.g., `approx(-0.04)` after `from pytest import approx`
    "torch.allclose",
    "allclose(",
    "torch.testing.assert_close",
    "assert_close(",
    ".grad <",
    ".grad >",
    ".grad.item() <",
    ".grad.item() >",
    ".grad ==",
    ".grad.item() ==",
    ".grad !=",
    ".sign()",
    "torch.sign",
    "loss_decrease",  # canonical pattern: assert loss after grad-step lower
    "loss_after",
    "loss_before",
    "torch.equal",
    # Convergence via SGD step: `final <= initial` / `initial >= final`.
    # Same idea as the loss-convergence patterns in Check 45 but specific
    # to autograd.Function tests that take a manual gradient step.
    "final <= initial",
    "initial >= final",
    "final < initial",
    "initial > final",
    # Indexed grad value/sign checks: `.grad[i].item() == X`,
    # `.grad[i] < 0`, etc. Catches the canonical Round 22 pattern where
    # specific elements are anchored. Use a regex below as well.
)

# Regex: indexed-grad value/sign check, e.g.
#   `w.grad[0].item() == 1.0` or `bits.grad[1] < 0` or `w.grad[i, j] >= ...`
_GRAD_DIRECTION_REGEX = re.compile(
    r"\.grad\[[^\]]*\](?:\.item\(\))?\s*(?:==|!=|<=|>=|<|>)"
)

# Regex: magnitude check on a grad value, e.g.
#   `abs(bits.grad.item()) < 1e-3` or `torch.abs(w.grad).max() < 0.5`
# We use a permissive lookahead: any `abs(...)` containing `.grad` somewhere
# inside, followed (within ~80 chars) by a comparison operator.
_GRAD_MAGNITUDE_REGEX = re.compile(
    r"abs\([^\n]*\.grad[^\n]{0,80}?[<>=!]=?"
    r"|\.grad\.abs\(\)[^=<>!\n]{0,80}?[<>=!]"
)


def _scan_test_file_for_grad_direction(
    path: Path, repo_root: Path
) -> list[str]:
    """For every test_* function that touches an autograd.Function backward,
    flag if it does not assert grad direction / value.

    Heuristic:
      1. Find the imports/use of `torch.autograd.Function` subclasses in the
         file (grep for `(torch.autograd.Function)` in class defs OR
         `<Name>.apply(` calls where Name was bound to such a class earlier
         in the file or imported).
      2. For each top-level `def test_*` function: if the function body
         references `.grad` AND any of the autograd.Function symbols, then
         require one of `_GRAD_DIRECTION_PATTERNS` to also appear in the
         function body. Otherwise FLAG.
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    rel_s = str(rel)
    # Only scan test files
    if "tests/" not in rel_s and "/tests/" not in rel_s:
        return []
    if not path.name.startswith("test_"):
        return []

    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    # 1. Collect names that are autograd.Function subclasses, either defined
    #    here or imported. Conservative: any imported name from a *quant*,
    #    *ste*, *self_compress*, *frozen_bit*, *fp4*, *fp8*, *learnable_bit*
    #    module is suspicious. Plus any class def that subclasses
    #    `torch.autograd.Function` or `Function` directly.
    autograd_function_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_repr = ast.unparse(base) if hasattr(ast, "unparse") else ""
                if "autograd.Function" in base_repr or base_repr == "Function":
                    autograd_function_names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            # Heuristic: imports from known STE-bearing modules.
            if any(tok in mod for tok in (
                "quantization", "quant", "ste", "self_compress",
                "frozen_bit", "fp4", "fp8", "learnable_bit",
            )):
                for alias in node.names:
                    name = alias.asname or alias.name
                    # Likely STE / Function-shaped name
                    if (
                        "STE" in name
                        or name.endswith("Quantize")
                        or name.endswith("Quant")
                        or name.endswith("FakeQuant")
                        or "Function" in name
                    ):
                        autograd_function_names.add(name)

    if not autograd_function_names:
        return []

    # Build a quick line-table for waiver detection.
    lines = text.splitlines()

    violations: list[str] = []

    # 2. Walk top-level test_* functions.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue

        # Get function source-text body (use line range).
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno)
        body_lines = lines[start:end]
        body_text = "\n".join(body_lines)

        # Same-line waiver on the def-line itself counts.
        def_line = lines[start] if start < len(lines) else ""
        if _GRAD_DIRECTION_WAIVER_TOKEN in def_line:
            continue

        # Same-line waiver on ANY line inside the function body counts.
        if _GRAD_DIRECTION_WAIVER_TOKEN in body_text:
            continue

        # Does the function reference any autograd.Function symbol?
        touches_function = any(
            name in body_text for name in autograd_function_names
        )
        if not touches_function:
            continue

        # Does the function reference `.grad`?
        if ".grad" not in body_text:
            continue

        # Check: does the body include any direction / value assertion?
        has_direction = (
            any(pat in body_text for pat in _GRAD_DIRECTION_PATTERNS)
            or bool(_GRAD_DIRECTION_REGEX.search(body_text))
            or bool(_GRAD_MAGNITUDE_REGEX.search(body_text))
        )
        if has_direction:
            continue

        # Acceptable also: a numeric `.grad` index/comparison via subscript
        # like `grad[0].item() == ...`. The patterns above cover this via
        # `pytest.approx` / `==` / `<` / `>` etc.

        violations.append(
            f"{rel}:{node.lineno}: test '{node.name}' touches an autograd."
            f"Function backward but only checks `grad is not None` / "
            f"`isfinite(grad)` — NO direction/value assertion. Add one of: "
            f"`pytest.approx(...)`, `torch.allclose(...)`, `assert grad < 0`, "
            f"or a loss-decrease check after a gradient step. "
            f"(Round 22 bit-STE sign bug hid for 4 review rounds because "
            f"of this exact gap.) Waive with same-line "
            f"`# {_GRAD_DIRECTION_WAIVER_TOKEN}<reason>`."
        )

    return violations


def check_gradient_direction_tests_exist(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Backward tests for autograd.Function must check grad direction/value.

    Reference: Round 22 bit-STE sign-bug post-mortem. The Round 12/13/14/18
    council reviews dismissed the sign bug because the only `bits.grad`
    assertion was `is not None`. Round 21 caught it via a hand-derived
    numeric value. Structural fix: every test that exercises an autograd.
    Function backward MUST assert sign, value, or a reference comparison.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    test_dir = root / "src" / "tac" / "tests"
    if test_dir.exists():
        for p in sorted(test_dir.rglob("test_*.py")):
            if "__pycache__" in p.parts:
                continue
            n_scanned += 1
            violations.extend(_scan_test_file_for_grad_direction(p, root))

    if verbose:
        if violations:
            print(
                f"  [grad-direction-tests] {len(violations)} violation(s) "
                f"across {n_scanned} test file(s):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    … (+{len(violations) - 20} more)")
        else:
            print(
                f"  [grad-direction-tests] OK: {n_scanned} test file(s) "
                f"scanned"
            )

    if violations and strict:
        raise MetaBugViolation(
            "GRADIENT-DIRECTION TESTS MISSING:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
            + "\n\nRound 22 bit-STE sign bug hid for 4 review rounds because "
            "the only assertion was `grad is not None`. Add direction/value "
            "checks (pytest.approx, torch.allclose, sign comparison, or a "
            "post-step loss-decrease check). Waive on the def-line with "
            f"`# {_GRAD_DIRECTION_WAIVER_TOKEN}<reason>`."
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 45 (2026-04-28): tests of *loss* functions/classes must include at
#                        least one convergence (loss-decrease) check.
#
# Companion to Check 44. Loss functions can return finite values that are
# nonetheless un-minimisable (gradient pointing the wrong way). Tests that
# only assert `loss.shape == ()` or `torch.isfinite(loss)` cannot detect
# this. CLAUDE.md anti-arbitrariness: a loss-function test must demonstrate
# the loss DECREASES under gradient descent (or has a known minimum at a
# known point).
#
# Same-line waiver: `# LOSS_CONVERGENCE_NOT_REQUIRED:<reason>`
# ════════════════════════════════════════════════════════════════════════════

_LOSS_CONVERGENCE_WAIVER_TOKEN = "LOSS_CONVERGENCE_NOT_REQUIRED:"

# Patterns that indicate a real loss-decrease / convergence assertion at the
# FILE level. If a loss-touching file has any of these patterns, we accept
# that file as a whole. Conservative on purpose: false-negatives are okay,
# false-positives (telling a clean test it's broken) are not.
_LOSS_CONVERGENCE_PATTERNS = (
    "loss_after",
    "loss_before",
    "loss_decrease",
    "loss_initial",
    "loss_final",
    "after_step",
    "before_step",
    "minimize",  # any reference to a minimisation / minimisable claim
    "monotonic",
    "decreases",
    ".step()",  # SGD/Adam step → loss recomputed → can compare
    "torch.optim",
    "gradient descent",  # docstring marker
    "GD step",
    "convergence",
    # Numeric anchor patterns (loss known to equal X at known input).
    "pytest.approx",
    "approx(",
    "torch.allclose",
    "torch.equal",
    "assert_close",
)


def _scan_test_file_for_loss_convergence(
    path: Path, repo_root: Path
) -> list[str]:
    """For each test file whose name contains 'loss' as a token (case-
    insensitive), require that the file as a whole demonstrates a
    convergence check.

    "Loss" must appear as a token, not as a fragment ("lossless" does NOT
    qualify — that's the lossless-coding test family, not a loss-function
    test).
    """
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    name = path.name.lower()
    if not name.startswith("test_"):
        return []
    # Tokenize on '_' / '.' boundaries; require "loss" as its own token.
    # 'lossless' / 'lossy' fragments do NOT count (different bug class).
    tokens = re.split(r"[_.]", name)
    if "loss" not in tokens:
        return []

    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    # Same-line waiver anywhere in the file for the WHOLE file.
    if _LOSS_CONVERGENCE_WAIVER_TOKEN in text:
        return []

    # File-level acceptance: any of the convergence patterns in the file.
    if any(pat in text for pat in _LOSS_CONVERGENCE_PATTERNS):
        return []

    # Find the first def test_* line for the violation lineno.
    lineno = 1
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("def test_"):
            lineno = i
            break

    return [
        f"{rel}:{lineno}: loss-function test file has no convergence check "
        f"(no loss_after/loss_before pattern, no `.step()`, no "
        f"`pytest.approx` / `torch.allclose` numeric anchor). A loss "
        f"function can return finite values whose gradient still points the "
        f"wrong way — finiteness is NOT a correctness gate. Add a "
        f"loss-decrease assertion or a known-minimum numeric check. "
        f"Waive with `# {_LOSS_CONVERGENCE_WAIVER_TOKEN}<reason>` anywhere "
        f"in the file."
    ]


def check_test_assertion_strength_for_loss_functions(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Tests of *loss* functions must include a convergence / numeric anchor.

    Companion to Check 44. A finite-but-wrong-direction loss is a known
    failure mode (Lane B 6.5h proxy-MSE-only TTO produced 0.0007 proxy /
    0.246 auth = 350× gap). Convergence tests catch this in seconds.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    test_dir = root / "src" / "tac" / "tests"
    if test_dir.exists():
        # Glob is permissive ("loss" anywhere); the scanner enforces the
        # "loss as a token (not fragment)" rule and discards "lossless" /
        # "lossy" filenames that are not loss-function tests.
        for p in sorted(test_dir.rglob("test_*loss*.py")):
            if "__pycache__" in p.parts:
                continue
            v = _scan_test_file_for_loss_convergence(p, root)
            # Only count files actually validated (token check passed).
            # Determine that by re-running the token check inline.
            tokens = re.split(r"[_.]", p.name.lower())
            if "loss" in tokens:
                n_scanned += 1
            violations.extend(v)

    if verbose:
        if violations:
            print(
                f"  [loss-convergence-tests] {len(violations)} violation(s) "
                f"across {n_scanned} loss-test file(s):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
        else:
            print(
                f"  [loss-convergence-tests] OK: {n_scanned} loss-test file(s) scanned"
            )

    if violations and strict:
        raise MetaBugViolation(
            "LOSS-FUNCTION TESTS MISSING CONVERGENCE CHECK:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 46 (2026-04-28): every public quantizer / encoder needs a roundtrip
#                        test (`unquantize(quantize(x))` ≈ `x`).
#
# A quantizer that silently drops dynamic range (or saturates / shifts) can
# pass forward-shape and finiteness tests but corrupt the artifact at
# inflate time. Roundtrip tests catch the failure mode in seconds.
#
# Same-line waiver: `# ROUNDTRIP_NOT_REQUIRED:<reason>`
# ════════════════════════════════════════════════════════════════════════════

_ROUNDTRIP_WAIVER_TOKEN = "ROUNDTRIP_NOT_REQUIRED:"

# File-name globs: which modules count as "quantizer / encoder" producers.
_QUANTIZER_FILE_PATTERNS = (
    "*quant*.py",
    "*codec*.py",
    "*entropy*.py",
)

# Substrings in the corresponding test file that count as a roundtrip
# assertion (decode/encode pair on the SAME tensor with allclose / equal).
_ROUNDTRIP_PATTERNS = (
    "torch.allclose",
    "allclose(",
    "torch.equal",
    "torch.testing.assert_close",
    "assert_close(",
    "round_trip",
    "roundtrip",
    "round-trip",
    "decode(encode",
    "encode(decode",
    "unquantize(quantize",
    "dequantize(quantize",
    "decompress(compress",
    "decompress_archive",
    "inverse_transform",
)


def _module_basename(p: Path) -> str:
    return p.stem


def _quantizer_modules(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    src_dir = repo_root / "src" / "tac"
    if not src_dir.exists():
        return out
    seen: set[Path] = set()
    for pattern in _QUANTIZER_FILE_PATTERNS:
        for p in src_dir.glob(pattern):
            if p in seen:
                continue
            if p.name.startswith("test_"):
                continue
            seen.add(p)
            out.append(p)
    return sorted(out)


def _has_public_class_or_function(path: Path) -> bool:
    """True iff the module exposes at least one public top-level class or def."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if not node.name.startswith("_"):
                return True
    return False


def _find_test_files_for_module(
    module_path: Path, repo_root: Path
) -> list[Path]:
    """Return test files that import from this module."""
    test_dir = repo_root / "src" / "tac" / "tests"
    if not test_dir.exists():
        return []
    mod_basename = _module_basename(module_path)
    needle = f"tac.{mod_basename}"
    out: list[Path] = []
    # Direct convention: test_<basename>.py
    direct = test_dir / f"test_{mod_basename}.py"
    if direct.exists():
        out.append(direct)
    # Anything else that imports `tac.<basename>`
    for p in test_dir.rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        if p in out:
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if needle in text:
            out.append(p)
    return out


def _scan_quantizer_for_roundtrip_test(
    module_path: Path, repo_root: Path
) -> list[str]:
    rel = module_path.relative_to(repo_root) if module_path.is_absolute() else module_path
    if not _has_public_class_or_function(module_path):
        return []

    # File-level waiver on the module itself.
    try:
        mod_text = module_path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    if _ROUNDTRIP_WAIVER_TOKEN in mod_text:
        return []

    test_files = _find_test_files_for_module(module_path, repo_root)
    if not test_files:
        return [
            f"{rel}: quantizer/encoder module has no test file at "
            f"src/tac/tests/test_{_module_basename(module_path)}.py and no "
            f"other test imports it. Add a roundtrip test "
            f"(`assert torch.allclose(decode(encode(x)), x, atol=...)`) or "
            f"waive in the module with "
            f"`# {_ROUNDTRIP_WAIVER_TOKEN}<reason>`."
        ]

    # If ANY test file for this module has a roundtrip pattern, accept.
    for tf in test_files:
        try:
            ttext = tf.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if _ROUNDTRIP_WAIVER_TOKEN in ttext:
            return []
        if any(pat in ttext for pat in _ROUNDTRIP_PATTERNS):
            return []

    return [
        f"{rel}: quantizer/encoder module has tests "
        f"({', '.join(t.name for t in test_files)}) but no roundtrip "
        f"assertion (no `torch.allclose`, no `decode(encode(...))`, no "
        f"`roundtrip` substring, no `assert_close`). Add "
        f"`assert torch.allclose(unquantize(quantize(x)), x, atol=...)` or "
        f"waive on the module with `# {_ROUNDTRIP_WAIVER_TOKEN}<reason>`."
    ]


def check_quantizer_modules_have_round_trip_test(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Every public quantizer / encoder needs a `decode(encode(x)) ≈ x` test.

    Reference: archive measurement disasters (2026-04-21) — quantizers
    silently dropped dynamic range, passing forward-shape tests but
    corrupting the inflated artifact. Roundtrip tests catch this in seconds.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for mod in _quantizer_modules(root):
        n_scanned += 1
        violations.extend(_scan_quantizer_for_roundtrip_test(mod, root))

    if verbose:
        if violations:
            print(
                f"  [quantizer-roundtrip-tests] {len(violations)} violation(s) "
                f"across {n_scanned} quantizer/encoder module(s):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
        else:
            print(
                f"  [quantizer-roundtrip-tests] OK: {n_scanned} module(s) scanned"
            )

    if violations and strict:
        raise MetaBugViolation(
            "QUANTIZER MODULES MISSING ROUNDTRIP TEST:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 47 (2026-04-28): scripts/remote_lane_*.sh that build an archive must
#                        ASSERT the archive size BEFORE calling
#                        contest_auth_eval / inflate.sh.
#
# Reference: Lane B class disasters where archive composition silently
# changed the rate term (renderer-only 119 KB instead of 338 KB full
# submission → 0.108 rate error per CLAUDE.md). The shell idiom
# `ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" ...) && [ "$ARCHIVE_BYTES" -gt N ]`
# OR a Python `os.path.getsize(...) >= N` assertion catches the failure
# mode at compose time, not after a $0.50 eval.
#
# Same-line waiver: `# ARCHIVE_SIZE_NOT_REQUIRED:<reason>`
# ════════════════════════════════════════════════════════════════════════════

_ARCHIVE_SIZE_WAIVER_TOKEN = "ARCHIVE_SIZE_NOT_REQUIRED:"

# Substrings that indicate the script BUILDS an archive (vs. just consuming
# one for eval). If none of these patterns appear, the script is exempt.
_ARCHIVE_BUILD_MARKERS = (
    "build_archive",
    "submission_archive",
    "ZipFile(",
    "zipfile.ZipFile",
    "zip.write",
    "z.write(",
    "shutil.copy",  # often used to assemble an archive directory
)

# Substrings that indicate auth eval / inflate is being invoked.
_AUTH_EVAL_MARKERS = (
    "contest_auth_eval",
    "auth_eval_renderer",
    "inflate.sh",
    "evaluate.py",
)

# Substrings that satisfy the size-assertion gate. Either a shell-side
# `[ "$X" -gt N ]` / `-le 0` style check OR a Python-side numeric compare.
_ARCHIVE_SIZE_ASSERTION_PATTERNS = (
    # Shell numeric guards on a captured size variable.
    'ARCHIVE_BYTES',
    'ARCHIVE_SIZE',
    "stat -c '%s'",
    'stat -c "%s"',
    "stat -f '%z'",
    'stat -f "%z"',
    "wc -c",
    "du -b",
    "du -sb",
    # Python-side: os.path.getsize / Path(...).stat().st_size etc with a
    # numeric compare (we use the `assert ... getsize` substring as the
    # gate; printing alone is NOT enough, but most scripts that check size
    # also assert).
    "assert os.path.getsize",
    "assert os.stat",
    "raise SystemExit",  # often used as size gate in inline Python
    "size empty or zero",  # canonical lane_a_optimized phrasing
    "refusing to call auth_eval",
    " -le 0",
    " -lt ",
    " -gt ",
)


def _scan_remote_lane_for_archive_size_assertion(
    path: Path, repo_root: Path
) -> list[str]:
    rel = path.relative_to(repo_root) if path.is_absolute() else path
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    # File-level waiver.
    if _ARCHIVE_SIZE_WAIVER_TOKEN in text:
        return []

    builds = any(m in text for m in _ARCHIVE_BUILD_MARKERS)
    evals = any(m in text for m in _AUTH_EVAL_MARKERS)
    if not (builds and evals):
        return []

    # Does the script include any size-assertion pattern?
    if any(pat in text for pat in _ARCHIVE_SIZE_ASSERTION_PATTERNS):
        return []

    # Find the first auth-eval marker line for the violation lineno.
    lineno = 1
    for i, line in enumerate(text.splitlines(), start=1):
        if any(m in line for m in _AUTH_EVAL_MARKERS):
            lineno = i
            break

    return [
        f"{rel}:{lineno}: lane script builds an archive AND invokes auth "
        f"eval, but does not assert archive byte-size before the eval call. "
        f"Add a guard like:\n"
        f"      ARCHIVE_BYTES=$(stat -c '%s' \"$ARCHIVE\" 2>/dev/null || stat -f '%z' \"$ARCHIVE\")\n"
        f"      [ \"$ARCHIVE_BYTES\" -gt 0 ] || {{ echo 'FATAL: archive empty'; exit 2; }}\n"
        f"    Lane B-class disasters (renderer-only 119 KB vs 338 KB full "
        f"submission) cost 0.108 rate points per CLAUDE.md. "
        f"Waive with `# {_ARCHIVE_SIZE_WAIVER_TOKEN}<reason>`."
    ]


def check_lane_deploy_scripts_have_archive_size_assertion(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Lane scripts that build an archive must assert size before auth eval.

    Reference: CLAUDE.md "Auth eval measurement — non-negotiable" — every
    auth eval MUST use the EXACT archive that will be submitted, and the
    archive size must be reported. Lane B's 119 KB renderer-only archive
    silently inflated the rate term by 0.108 across multiple sessions. A
    one-line shell guard catches this at compose time.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        for p in sorted(scripts_dir.glob("remote_lane_*.sh")):
            n_scanned += 1
            violations.extend(
                _scan_remote_lane_for_archive_size_assertion(p, root)
            )

    if verbose:
        if violations:
            print(
                f"  [lane-archive-size] {len(violations)} violation(s) "
                f"across {n_scanned} remote_lane_*.sh file(s):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
        else:
            print(
                f"  [lane-archive-size] OK: {n_scanned} remote_lane_*.sh file(s) scanned"
            )

    if violations and strict:
        raise MetaBugViolation(
            "LANE SCRIPTS MISSING ARCHIVE-SIZE ASSERTION:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 48: orphan src/tac modules (no profile / CLI / script reference) ──
#
# CATCHES: a contributor adds src/tac/new_thing.py, never wires it into a
# profile, CLI flag, or deploy script — silent dead code that bloats the
# wheel and confuses future agents about what's actually shipped. Live at
# session start (2026-04-28 evening): unknown count; check ships warn-only
# initially because the audit is a real cleanup task, not a regression
# blocker. Promotion to STRICT once the violation count is driven to 0.
#
# Reference: project_killed_lanes_forensic_audit_20260428 (Lane V channel
# bug shipped because the 88K DSConv path was orphaned from real testing).


# Modules that are intentionally library-only (imported by other tac
# modules but not user-facing via a profile / CLI / script). Excluding
# these prevents false positives — they're EXPECTED to be referenced only
# via Python imports, not via a deploy script or profile knob.
_ORPHAN_CHECK_EXEMPT_MODULES = {
    "__init__", "__main__",
    # Top-level entry / config modules (referenced by name from many places,
    # not via `tac.<name>` import — exempt from this check's grep heuristic).
    "profiles", "preflight", "cli", "entrypoints", "__main__",
    # Library helpers / utilities (imported by other tac.* modules)
    "bootstrap_codegen", "checkpoint_names", "cost_tracker",
    "data", "models", "checkpoint", "evaluate", "parametrize_strip",
}


def check_no_orphan_src_tac_modules(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catch src/tac/*.py modules with no profile / CLI / script reference.

    For every src/tac/<name>.py (excluding tests/, tools/, experiments/, and
    library-only exempts), at least one of the following must reference it:
      1. An import inside src/tac/profiles.py (e.g., a profile knob calls it)
      2. An import inside src/tac/experiments/train_renderer.py (CLI dispatch)
      3. A `from tac.<name>` or `import tac.<name>` in any
         scripts/remote_lane_*.sh's inline Python OR any experiments/*.py
         actively used by remote scripts.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    src_tac = root / "src" / "tac"
    if not src_tac.is_dir():
        return []

    # Enumerate candidate modules.
    candidates: list[str] = []
    for p in sorted(src_tac.glob("*.py")):
        stem = p.stem
        if stem in _ORPHAN_CHECK_EXEMPT_MODULES:
            continue
        if stem.startswith("_"):
            continue
        candidates.append(stem)

    # Build the union of all reference texts.
    profiles_text = ""
    train_text = ""
    profiles_path = src_tac / "profiles.py"
    train_path = src_tac / "experiments" / "train_renderer.py"
    try:
        profiles_text = profiles_path.read_text()
    except (OSError, UnicodeDecodeError):
        pass
    try:
        train_text = train_path.read_text()
    except (OSError, UnicodeDecodeError):
        pass

    scripts_text = ""
    scripts_dir = root / "scripts"
    if scripts_dir.is_dir():
        for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
            try:
                scripts_text += sh.read_text() + "\n"
            except (OSError, UnicodeDecodeError):
                continue
    experiments_text = ""
    exp_dir = root / "experiments"
    if exp_dir.is_dir():
        for py in sorted(exp_dir.glob("*.py")):
            try:
                experiments_text += py.read_text() + "\n"
            except (OSError, UnicodeDecodeError):
                continue

    haystack = profiles_text + "\n" + train_text + "\n" + scripts_text + "\n" + experiments_text
    violations: list[str] = []
    for name in candidates:
        # Match `tac.<name>` (import / from-import) — covers all 4 reference types.
        pattern = rf"\btac\.{re.escape(name)}\b"
        if not re.search(pattern, haystack):
            violations.append(
                f"src/tac/{name}.py: no reference in profiles.py / train_renderer.py / "
                f"remote_lane_*.sh / experiments/*.py — orphan module suspected. "
                f"If intentional library-only helper, add to _ORPHAN_CHECK_EXEMPT_MODULES."
            )

    if verbose:
        if violations:
            print(
                f"  [no-orphan-src-tac] {len(violations)} violation(s) "
                f"across {len(candidates)} candidate module(s):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    … and {len(violations) - 20} more")
        else:
            print(
                f"  [no-orphan-src-tac] OK: {len(candidates)} module(s) all referenced"
            )

    if violations and strict:
        raise MetaBugViolation(
            "ORPHAN SRC/TAC MODULES (no profile / CLI / script reference):\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 49: every profile loss_mode must be in train_renderer validator ──
#
# CATCHES: the Lane J-JBL exit class. A profile sets loss_mode="jbl" but
# train_renderer.py's _VALID_LOSS_MODES allowlist (~line 888) doesn't
# include "jbl" — the validator raises SystemExit at boot, the lane exits
# unexpectedly. Lane J-JBL hit this on 2026-04-28; ~$0.05 burned + 1
# debugging cycle. Catching at preflight time means the violation surfaces
# at commit/PR, not after deploy.
#
# Reference: project_killed_lanes_forensic_audit_20260428 (Lane J-JBL section).


def check_profile_loss_modes_in_validator_allowlist(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch profile.loss_mode values not in train_renderer.py allowlist.

    Iterate every PROFILES entry; if it sets loss_mode, the value MUST
    appear in _VALID_LOSS_MODES inside train_renderer.py. Otherwise the
    validator raises at boot and the lane exits silently.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    train_path = root / "src" / "tac" / "experiments" / "train_renderer.py"
    profiles_path = root / "src" / "tac" / "profiles.py"
    if not train_path.is_file() or not profiles_path.is_file():
        return []

    try:
        train_text = train_path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    # Extract the _VALID_LOSS_MODES tuple via regex (multi-line tuple support).
    m = re.search(
        r"_VALID_LOSS_MODES\s*=\s*\(([^)]*)\)",
        train_text,
        re.DOTALL,
    )
    if not m:
        # Allowlist not present — can't validate. Treat as a warning, not a
        # failure (the allowlist itself is enforced by code review).
        if verbose:
            print(
                "  [profile-loss-mode-allowlist] WARN: _VALID_LOSS_MODES "
                "tuple not found in train_renderer.py — skipping check"
            )
        return []
    allowlist_raw = m.group(1)
    allowed = set(re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', allowlist_raw))

    # Import-time profile loading is fragile from preflight (heavy deps).
    # Static-scan profiles.py for `"loss_mode":\s*"<value>"` literal pairs.
    try:
        profiles_text = profiles_path.read_text()
    except (OSError, UnicodeDecodeError):
        return []

    violations: list[str] = []
    seen_values: set[str] = set()
    # Match e.g. `"loss_mode": "jbl"` or `"loss_mode":"jbl"`.
    for m2 in re.finditer(
        r'["\']loss_mode["\']\s*:\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']',
        profiles_text,
    ):
        val = m2.group(1)
        seen_values.add(val)
    for val in sorted(seen_values):
        if val not in allowed:
            violations.append(
                f"profiles.py declares loss_mode={val!r} but "
                f"train_renderer.py _VALID_LOSS_MODES = {sorted(allowed)} "
                f"does NOT include it. Profile will SystemExit at boot. "
                f"Add {val!r} to _VALID_LOSS_MODES OR remove from profile."
            )

    if verbose:
        if violations:
            print(
                f"  [profile-loss-mode-allowlist] {len(violations)} "
                f"violation(s) — allowed: {sorted(allowed)}"
            )
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [profile-loss-mode-allowlist] OK: profile loss_mode "
                f"values {sorted(seen_values)} all in allowlist {sorted(allowed)}"
            )

    if violations and strict:
        raise MetaBugViolation(
            "PROFILE LOSS_MODE NOT IN VALIDATOR ALLOWLIST:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations


# ── Check 50: every deploy script --profile X must reference a real profile ──
#
# CATCHES: a deploy script passes `--profile some_typo` that never existed
# in PROFILES; train_renderer.py raises KeyError after 5+ minutes of setup
# burn (NVDEC probe, env init, package install). Catching at preflight
# means the violation surfaces at commit time, before any GPU spend.
#
# Reference: project_killed_lanes_forensic_audit_20260428 (Lane H-V3
# revival authoring required this check to land before the launch).


def check_deploy_script_profiles_exist_in_registry(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Catch deploy scripts whose --profile X references unknown PROFILES.

    For every scripts/remote_lane_*.sh file, extract every `--profile X`
    invocation and verify X exists as a key in PROFILES (parsed statically
    from src/tac/profiles.py — no Python import to keep preflight cheap).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    profiles_path = root / "src" / "tac" / "profiles.py"
    if not profiles_path.is_file():
        return []

    try:
        profiles_text = profiles_path.read_text()
    except (OSError, UnicodeDecodeError):
        return []

    # Static-extract every PROFILES key. Match patterns like:
    #   "h_v3_joint_halfframe": H_V3_JOINT_HALFFRAME,
    # Inside the PROFILES dict.
    # Conservative: just extract every double-quoted key from the file. A
    # false-positive registration is acceptable (profile MIGHT exist); a
    # false-negative is the bug class we want to catch.
    registered: set[str] = set()
    # Find the PROFILES = { ... } block bounds (best-effort: from PROFILES = { to the matching closing brace).
    pm = re.search(r"PROFILES\s*=\s*\{", profiles_text)
    if pm:
        # Walk char-by-char to find the matching brace.
        start = pm.end() - 1
        depth = 0
        end = len(profiles_text)
        for i in range(start, len(profiles_text)):
            c = profiles_text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = profiles_text[start:end]
        # Extract keys: matching `"<name>":` at start of trimmed lines (best-effort).
        for m2 in re.finditer(r'^\s*["\']([a-z_][a-z0-9_]*)["\']\s*:', block, re.MULTILINE):
            registered.add(m2.group(1))

    if not registered:
        if verbose:
            print(
                "  [deploy-script-profile-exists] WARN: failed to "
                "extract PROFILES keys — skipping check"
            )
        return []

    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return []

    violations: list[str] = []
    n_scanned = 0
    for sh in sorted(scripts_dir.glob("remote_lane_*.sh")):
        n_scanned += 1
        try:
            text = sh.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = sh.relative_to(root)
        # Match `--profile <name>` (next-token). Also match
        # `--profile=<name>` and bash interpolations starting with $.
        for m3 in re.finditer(
            r"--profile[\s=]+([a-zA-Z0-9_\$\{\}-]+)",
            text,
        ):
            ref = m3.group(1)
            # Skip bash interpolations (operator-supplied at runtime).
            if "$" in ref:
                continue
            # Skip dynamic placeholders.
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", ref):
                continue
            if ref not in registered:
                violations.append(
                    f"{rel}: --profile {ref!r} not in PROFILES registry. "
                    f"Add to src/tac/profiles.py PROFILES dict OR fix typo. "
                    f"Available: {sorted(registered)[:5]}…"
                )

    if verbose:
        if violations:
            print(
                f"  [deploy-script-profile-exists] {len(violations)} "
                f"violation(s) across {n_scanned} remote_lane_*.sh:"
            )
            for v in violations[:20]:
                print(f"    • {v}")
        else:
            print(
                f"  [deploy-script-profile-exists] OK: {n_scanned} "
                f"remote_lane_*.sh scanned, all --profile X resolve in PROFILES"
            )

    if violations and strict:
        raise MetaBugViolation(
            "DEPLOY SCRIPT --profile X REFERENCES MISSING PROFILE:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 51: bare `except:` and `except Exception: pass` ─────────────────────
#
# CATCHES the silent-swallow bug class: any handler that catches all
# exceptions without logging or re-raising hides bugs forever. We saw this
# in tools/fleet_dashboard_live.py (commit on 2026-04-28) where a
# `try: tag = cmd.split("--tag")[1].strip().split()[0]; except: pass` masked
# real failures. This check forbids:
#   - Bare `except:` (catches BaseException including KeyboardInterrupt)
#   - `except Exception: pass` (silent-swallow with no log)
#
# Allowed:
#   - Specific exceptions: `except IndexError:`, `except (OSError, ValueError):`
#   - Bare `except Exception` followed by logging / re-raise / clear handling
#
# Exemptions: tests/, vendored upstream/, this preflight.py file itself
# (where regex pattern strings include `except:` literal text), and any line
# with a SAME-LINE waiver marker `# noqa: E722` or `# silent-swallow-OK:`.
#
# Reference: feedback_deep_hardening_pass_2_patterns_20260428 +
# 2026-04-28 deep DX hardening pass.

_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*(?:#.*)?$")
_EXCEPT_EXCEPTION_PASS_RE = re.compile(
    r"^\s*except\s+Exception\s*(?:as\s+\w+)?\s*:\s*pass\s*(?:#.*)?$"
)


def check_no_bare_except(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid bare except: and `except Exception: pass`.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    skip_dirs = {
        "tests", "test", "upstream", "node_modules", ".venv", "venv",
        "build", "dist", "__pycache__",
    }
    for py_path in sorted(root.rglob("*.py")):
        # Skip the preflight file itself (contains regex patterns like
        # `except:` as string literals that would false-positive).
        if py_path.resolve() == Path(__file__).resolve():
            continue
        # Skip vendored / test / build dirs.
        rel_parts = py_path.relative_to(root).parts
        if any(p in skip_dirs for p in rel_parts):
            continue
        # Only scan src/tac, scripts/, tools/, experiments/.
        top = rel_parts[0] if rel_parts else ""
        if top not in {"src", "scripts", "tools", "experiments"}:
            continue
        try:
            text = py_path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        n_scanned += 1
        for i, line in enumerate(text.splitlines(), start=1):
            # Honor same-line waiver markers.
            if "# noqa: E722" in line or "# silent-swallow-OK" in line:
                continue
            if _BARE_EXCEPT_RE.match(line):
                rel = py_path.relative_to(root)
                violations.append(
                    f"{rel}:{i}: bare `except:` — catches BaseException "
                    f"including KeyboardInterrupt. Use specific exception type "
                    f"OR add `# noqa: E722` if intentional."
                )
            elif _EXCEPT_EXCEPTION_PASS_RE.match(line):
                rel = py_path.relative_to(root)
                violations.append(
                    f"{rel}:{i}: `except Exception: pass` silently swallows "
                    f"errors. Log the exception OR catch a specific subclass "
                    f"OR add `# silent-swallow-OK: <reason>`."
                )

    if verbose:
        if violations:
            print(
                f"  [no-bare-except] {len(violations)} violation(s) "
                f"across {n_scanned} files:"
            )
            for v in violations[:10]:
                print(f"    • {v}")
            if len(violations) > 10:
                print(f"    … and {len(violations) - 10} more")
        else:
            print(f"  [no-bare-except] OK: {n_scanned} files clean")

    if violations and strict:
        raise MetaBugViolation(
            "BARE EXCEPT / SILENT-SWALLOW VIOLATIONS:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 52: subprocess.run() returncode must be checked ─────────────────────
#
# CATCHES the silent-success bug class: `result = subprocess.run(...)` with
# no `.returncode` check downstream, no `check=True`, and no explicit
# discard. This is exactly how the LANE-B `set -uo pipefail` cascade hid
# silent failures (memory: feedback_zip_dep_bootstrap_trap). At Python
# level the equivalent is:
#
#   result = subprocess.run([...])
#   # ... no result.returncode check anywhere ...
#
# Allowed:
#   - `subprocess.run([...], check=True)` — raises CalledProcessError
#   - `r = subprocess.run([...]); if r.returncode != 0: ...` — explicit
#   - `subprocess.run([...], check=False)` — explicit opt-out
#   - Same-line `# subprocess-no-check-OK: <reason>` waiver
#
# Heuristic: scan for `subprocess.run(` and verify ONE of:
#   1. `check=True` in the call's parens (single-line)
#   2. The return value is captured AND `.returncode` appears within the
#      next 50 lines.
#   3. Same-line waiver.
#
# This is intentionally a loose check (warn-only initially) because perfect
# AST analysis of variable lifetimes is brittle. Promote to strict after
# a one-time cleanup pass.

def check_subprocess_run_checked(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Warn on `subprocess.run(...)` without check=True or returncode check.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    skip_dirs = {
        "tests", "test", "upstream", "node_modules", ".venv", "venv",
        "build", "dist", "__pycache__",
    }
    for py_path in sorted(root.rglob("*.py")):
        if py_path.resolve() == Path(__file__).resolve():
            continue
        rel_parts = py_path.relative_to(root).parts
        if any(p in skip_dirs for p in rel_parts):
            continue
        top = rel_parts[0] if rel_parts else ""
        if top not in {"src", "scripts", "tools", "experiments"}:
            continue
        try:
            text = py_path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        n_scanned += 1
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "subprocess.run(" not in line:
                continue
            # Skip pure-comment lines (text mention, not actual call site).
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Same-line waiver
            if "# subprocess-no-check-OK" in line:
                continue
            # Common safe patterns on the same line
            if "check=True" in line or "check = True" in line:
                continue
            # Multi-line call: scan next 8 lines for check=True
            window = "\n".join(lines[i:i + 8])
            if "check=True" in window or "check = True" in window:
                continue
            if "check=False" in window or "check = False" in window:
                # Explicit opt-out — accept (operator made an active choice).
                continue
            # If the return value is captured (e.g., `r =` or `result =`),
            # look forward up to 50 lines for a `.returncode` reference.
            assignment = re.match(r"^\s*(\w+)\s*=\s*subprocess\.run", line)
            if assignment:
                varname = assignment.group(1)
                lookahead = "\n".join(lines[i:i + 50])
                if f"{varname}.returncode" in lookahead:
                    continue
                if f"{varname}.check_returncode" in lookahead:
                    continue
            else:
                # Not assigned — if the call discards the result and is in a
                # context where failures don't matter (e.g., bootstrap script),
                # the operator should waive explicitly.
                pass
            rel = py_path.relative_to(root)
            violations.append(
                f"{rel}:{i + 1}: subprocess.run() without check=True or "
                f"returncode check. Use check=True OR capture + check "
                f"`.returncode` OR add `# subprocess-no-check-OK: <reason>`."
            )

    if verbose:
        if violations:
            print(
                f"  [subprocess-run-checked] {len(violations)} violation(s) "
                f"across {n_scanned} files (warn-only — promote after cleanup):"
            )
            for v in violations[:10]:
                print(f"    • {v}")
            if len(violations) > 10:
                print(f"    … and {len(violations) - 10} more")
        else:
            print(f"  [subprocess-run-checked] OK: {n_scanned} files clean")

    if violations and strict:
        raise MetaBugViolation(
            "SUBPROCESS.RUN WITHOUT CHECK= VIOLATIONS:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 53: tools/*.py must have non-empty --help ───────────────────────────
#
# CATCHES the operator-discoverability bug: a tool ships without argparse
# wired up, so `--help` either errors or prints nothing. Operators then
# can't find the tool's options without reading the source. This check
# verifies every executable script under tools/ AND scripts/*.py
# (excluding bootstrap shell scripts) accepts `--help` AND emits non-empty
# output.
#
# Heuristic: STATIC scan only (no subprocess invocation at preflight time
# because that would require imports to succeed and may have side effects).
# Verify that the file contains either:
#   - `argparse.ArgumentParser(`
#   - `import argparse` AND `add_argument(`
#   - `import click` (click auto-generates --help)
#   - Same-line `# no-argparse-OK: <reason>` waiver in a top-level comment
#
# Skipped: __init__.py, anything starting with `_`.

def check_tools_have_argparse(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify tools/*.py have argparse / click for --help discoverability.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0
    for tools_dir_name in ("tools", "scripts"):
        tools_dir = root / tools_dir_name
        if not tools_dir.is_dir():
            continue
        for py_path in sorted(tools_dir.glob("*.py")):
            if py_path.name.startswith("_") or py_path.name == "__init__.py":
                continue
            try:
                text = py_path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            n_scanned += 1
            # Same-line waiver in any top-level comment within the first 30 lines.
            head = "\n".join(text.splitlines()[:30])
            if "# no-argparse-OK" in head:
                continue
            # Must have a `__main__` entry to be a CLI.
            if "__name__" not in text or "__main__" not in text:
                continue  # library helper, not a CLI
            has_argparse = "ArgumentParser(" in text or (
                "import argparse" in text and "add_argument(" in text
            )
            has_click = "import click" in text or "from click" in text
            if not (has_argparse or has_click):
                rel = py_path.relative_to(root)
                violations.append(
                    f"{rel}: __main__ entry but no argparse/click — operators "
                    f"can't discover options via --help. Add an "
                    f"argparse.ArgumentParser OR `# no-argparse-OK: <reason>` "
                    f"in the top docstring."
                )

    if verbose:
        if violations:
            print(
                f"  [tools-have-argparse] {len(violations)} violation(s) "
                f"across {n_scanned} CLI scripts:"
            )
            for v in violations[:10]:
                print(f"    • {v}")
            if len(violations) > 10:
                print(f"    … and {len(violations) - 10} more")
        else:
            print(f"  [tools-have-argparse] OK: {n_scanned} CLI scripts clean")

    if violations and strict:
        raise MetaBugViolation(
            "CLI SCRIPTS WITHOUT --help:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 54 (54th meta-bug): scripts/launch_lane_on_vastai.py phase2-launch
#                          MUST call _poll_setup_log_for_outcome OR
#                          honor a skip_post_verify opt-in. Closes the
#                          "phase2-launch returns success before lane
#                          starts" regression class.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: Today wasted ~$10 on 87% NVDEC_BAD Vast.ai 4090 hosts because
# phase2-launch returned success the moment SSH+tmux backgrounded the lane
# wrapper — but setup_full.sh would then crash on Stage 4 NVDEC probe and
# the operator only learned about it 5+ minutes later via heartbeat.
#
# Fix landed in two layers:
#   Layer 1 DETECTION (commit 58e55890): scripts/probe_nvdec.sh
#     --lightweight at setup_full.sh Stage 0.5 catches ~95% of
#     NVDEC-missing hosts BEFORE the 5-minute DALI install.
#   Layer 2 ACTION  (commit 5acebb88-ish): launch_lane_on_vastai.py
#     phase2-launch Stage 2 polls setup.log via
#     _poll_setup_log_for_outcome() and auto-destroys NVDEC_BAD hosts.
#
# Without Layer 2, the canonical workflow regresses to fire-and-forget
# silent-failure mode. This check makes Layer 2 structurally permanent:
# any future refactor that drops the post-launch poll fails preflight.
#
# Memory: feedback_canonical_nvdec_workflow_GUARD_20260428,
#         feedback_vastai_launch_returns_success_before_lane_starts.

def check_phase2_launch_polls_setup_log(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid phase2-launch refactors that drop the post-launch outcome poll.

    The launcher's phase2-launch must either:
      (a) call ``_poll_setup_log_for_outcome(host, port, instance_id, ...)``
          to detect NVDEC_BAD / SETUP_COMPLETE on the lane host, OR
      (b) honor a ``skip_post_verify`` opt-in (``getattr(args,
          "skip_post_verify", False)``) for explicit fire-and-forget.

    Closes the "phase2-launch returns success before lane starts"
    regression class (see feedback_canonical_nvdec_workflow_GUARD_20260428).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    target = root / "scripts" / "launch_lane_on_vastai.py"
    violations: list[str] = []

    if not target.is_file():
        if verbose:
            print(f"  [phase2-launch-poll] SKIP: {target} not present")
        return violations

    try:
        text = target.read_text()
    except (OSError, UnicodeDecodeError) as e:
        violations.append(f"{target.relative_to(root)}: cannot read — {e}")
        if strict:
            raise MetaBugViolation(violations[0])
        return violations

    try:
        tree = ast.parse(text, filename=str(target))
    except SyntaxError as e:
        violations.append(
            f"{target.relative_to(root)}: SyntaxError ({e}) — cannot AST-scan"
        )
        if strict:
            raise MetaBugViolation(violations[0])
        return violations

    # Locate the cmd_phase2_launch function definition.
    target_func: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_phase2_launch":
            target_func = node
            break

    if target_func is None:
        violations.append(
            f"{target.relative_to(root)}: cmd_phase2_launch function not "
            f"found — the launcher must define a phase2-launch subcommand "
            f"that polls setup.log for NVDEC_BAD outcomes."
        )
    else:
        has_poll_call = False
        has_skip_opt_in = False
        for sub in ast.walk(target_func):
            if isinstance(sub, ast.Call):
                func_str = (
                    ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
                )
                # Match _poll_setup_log_for_outcome(...) or any call whose
                # function name ends with that token (allows future module
                # qualification, e.g. helpers._poll_setup_log_for_outcome).
                if (
                    func_str == "_poll_setup_log_for_outcome"
                    or func_str.endswith("._poll_setup_log_for_outcome")
                ):
                    has_poll_call = True
                # Match getattr(args, "skip_post_verify", False) opt-in
                # (any 3-arg getattr whose 2nd literal is the flag name).
                if (
                    func_str == "getattr"
                    and len(sub.args) >= 2
                    and isinstance(sub.args[1], ast.Constant)
                    and sub.args[1].value == "skip_post_verify"
                ):
                    has_skip_opt_in = True
        if not (has_poll_call and has_skip_opt_in):
            missing = []
            if not has_poll_call:
                missing.append("_poll_setup_log_for_outcome(...) call")
            if not has_skip_opt_in:
                missing.append('getattr(args, "skip_post_verify", False) opt-in')
            violations.append(
                f"{target.relative_to(root)}: cmd_phase2_launch missing "
                f"{' AND '.join(missing)}. Closes the "
                f"phase2-launch-returns-success-before-lane-starts regression "
                f"class. See feedback_canonical_nvdec_workflow_GUARD_20260428."
            )

    if verbose:
        if violations:
            print(
                f"  [phase2-launch-poll] {len(violations)} violation(s):"
            )
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [phase2-launch-poll] OK: cmd_phase2_launch polls "
                f"setup.log AND honors skip_post_verify opt-in"
            )

    if violations and strict:
        raise MetaBugViolation(
            "PHASE2-LAUNCH POLL VIOLATIONS — the launcher's phase2-launch "
            "must call _poll_setup_log_for_outcome AND honor a "
            "skip_post_verify opt-in. Without the poll, NVDEC-bad hosts "
            "burn $0.05-0.10 each (today's wave: ~$10 on 87% NVDEC_BAD).\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 55 (55th meta-bug): scripts/remote_setup_full.sh MUST invoke
#                          probe_nvdec.sh --lightweight at Stage 0.5
#                          BEFORE Stage 3 nvidia-dali-cuda120 install.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: companion to Check 54. The deep DALI-based NVDEC probe at
# Stage 4 runs AFTER a 5-minute `pip install nvidia-dali-cuda120` in
# Stage 3, costing $0.05+ per bad-NVDEC host. The lightweight pre-probe
# at Stage 0.5 dlopens libnvcuvid.so + cuvidGetDecoderCaps via ctypes —
# DALI-free, ~3s, catches ~95% of NVDEC-missing hosts BEFORE the heavy
# install.
#
# This check enforces ordering: if the script defines BOTH
# probe_nvdec.sh --lightweight AND a nvidia-dali-cuda120 install, the
# probe must come FIRST. A script that has neither is exempt (opt-out:
# the canonical setup is the only one in-tree, but third-party variants
# may not need DALI).

def check_setup_full_probe_before_dali(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid setup_full.sh refactors that move the lightweight NVDEC
    probe AFTER the DALI install (defeating the savings purpose).

    Scans ``scripts/remote_setup_full.sh`` for the FIRST occurrence of:
      - ``probe_nvdec.sh --lightweight``  → line N1
      - ``nvidia-dali-cuda120`` install OR ``Stage 3`` marker  → line N2

    Asserts N1 < N2. A file with neither is exempt (no DALI install ⇒
    no savings to defeat).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    target = root / "scripts" / "remote_setup_full.sh"
    violations: list[str] = []

    if not target.is_file():
        if verbose:
            print(f"  [setup-full-probe-order] SKIP: {target} not present")
        return violations

    try:
        text = target.read_text(errors="ignore")
    except OSError as e:
        violations.append(f"{target.relative_to(root)}: cannot read — {e}")
        if strict:
            raise MetaBugViolation(violations[0])
        return violations

    # Strip comment-only lines so docstring references don't count for
    # the ordering check (preserve line indices via space-padding).
    scan_lines: list[str] = []
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            scan_lines.append(" " * len(line))
        else:
            scan_lines.append(line)

    # Match `probe_nvdec.sh` (allowing intervening quotes/whitespace from
    # `bash "$WORKSPACE/scripts/probe_nvdec.sh" --lightweight`) followed by
    # `--lightweight` flag anywhere on the same line.
    probe_re = re.compile(r"probe_nvdec\.sh[\"'\s]*--lightweight\b")
    probe_line: int | None = None
    dali_line: int | None = None
    for i, line in enumerate(scan_lines, start=1):
        if probe_line is None and probe_re.search(line):
            probe_line = i
        if dali_line is None and (
            "nvidia-dali-cuda120" in line
            or "=== Stage 3" in line
        ):
            dali_line = i

    # Opt-out: neither marker present ⇒ no DALI savings to defeat.
    if probe_line is None and dali_line is None:
        if verbose:
            print(
                f"  [setup-full-probe-order] OK: {target.relative_to(root)} "
                f"has neither probe nor DALI install (opt-out)"
            )
        return violations

    if probe_line is None:
        violations.append(
            f"{target.relative_to(root)}: nvidia-dali-cuda120 install "
            f"present (line {dali_line}) but no `probe_nvdec.sh "
            f"--lightweight` Stage 0.5 pre-probe. Add the lightweight "
            f"probe BEFORE Stage 3 to save $0.05+/bad-NVDEC host."
        )
    elif dali_line is None:
        # Probe but no DALI — fine, nothing to defeat.
        if verbose:
            print(
                f"  [setup-full-probe-order] OK: probe present (line "
                f"{probe_line}); no DALI install to defeat"
            )
        return violations
    elif probe_line >= dali_line:
        violations.append(
            f"{target.relative_to(root)}: `probe_nvdec.sh --lightweight` "
            f"at line {probe_line} runs AFTER nvidia-dali-cuda120 install "
            f"at line {dali_line} — defeats the savings purpose. Move "
            f"probe to Stage 0.5 BEFORE Stage 3 DALI install. See "
            f"feedback_canonical_nvdec_workflow_GUARD_20260428."
        )

    if verbose:
        if violations:
            print(
                f"  [setup-full-probe-order] {len(violations)} violation(s):"
            )
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [setup-full-probe-order] OK: probe@line{probe_line} "
                f"runs BEFORE DALI@line{dali_line}"
            )

    if violations and strict:
        raise MetaBugViolation(
            "SETUP_FULL NVDEC PROBE ORDER VIOLATIONS — the lightweight "
            "NVDEC pre-probe must run BEFORE Stage 3 DALI install. "
            "Without it, every bad-NVDEC host pays the 5-minute DALI "
            "install cost before failing.\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 56 (56th meta-bug): scripts/verify_vast_instances.py auto-destroy
#                          path must use BOTH IDLE stale-minutes AND
#                          SETUP setup-stale-minutes thresholds.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: companion to the R31 cross-cutting SETUP-stuck cost-leak
# fix. The verify script's --auto-destroy-stale path originally only
# fired on IDLE/CRASHED — but a TRULY hung setup_full.sh (deadlocked,
# no heartbeat ever written) is classified SETUP, not IDLE. The IDLE
# stale-minutes threshold compares heartbeat freshness; with no
# heartbeat, that comparison never fires, so the instance accrues
# cost silently forever.
#
# This check enforces the dual-threshold pattern: any future refactor
# that drops EITHER the IDLE timer OR the SETUP timer fails preflight.
# Class of bug: heuristic-based health classifier with no timeout for
# the in-flight SETUP state.

def check_verify_vast_setup_stuck_dual_threshold(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid verify_vast_instances.py refactors that drop the
    SETUP-stale or IDLE-stale half of the dual-threshold auto-destroy.

    Scans ``scripts/verify_vast_instances.py`` for:
      1. CLI flag definition: ``--setup-stale-minutes``
      2. CLI flag definition: ``--stale-minutes``
      3. Auto-destroy path consults SETUP age (``setup_age_minutes``
         or ``setup_stale_minutes`` referenced inside the
         ``auto_destroy_stale`` branch)
      4. Auto-destroy path consults IDLE/CRASHED classification

    A repo missing the file is exempt (skip).

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    target = root / "scripts" / "verify_vast_instances.py"
    violations: list[str] = []

    if not target.is_file():
        if verbose:
            print(
                f"  [verify-vast-dual-threshold] SKIP: "
                f"{target} not present"
            )
        return violations

    try:
        text = target.read_text(errors="ignore")
    except OSError as e:
        violations.append(f"{target.relative_to(root)}: cannot read — {e}")
        if strict:
            raise MetaBugViolation(violations[0])
        return violations

    # 1. CLI flag definitions.
    if '"--setup-stale-minutes"' not in text and "'--setup-stale-minutes'" not in text:
        violations.append(
            f"{target.relative_to(root)}: missing CLI flag "
            f"`--setup-stale-minutes` definition. Without it, SETUP-"
            f"stuck instances (deadlocked setup_full.sh, never write "
            f"heartbeat) accrue cost silently forever — the IDLE "
            f"timer never fires because there's no heartbeat to be "
            f"stale. See feedback_setup_stuck_cost_leak_FIXED_20260428."
        )
    if '"--stale-minutes"' not in text and "'--stale-minutes'" not in text:
        violations.append(
            f"{target.relative_to(root)}: missing CLI flag "
            f"`--stale-minutes` definition (IDLE heartbeat-age "
            f"threshold). Half of the dual-threshold pattern."
        )

    # 2. Locate the auto-destroy block. Tolerate either snake_case
    # (args.auto_destroy_stale) or hyphenated CLI form references in
    # comments/strings; only the snake_case attribute matters.
    if "args.auto_destroy_stale" not in text:
        violations.append(
            f"{target.relative_to(root)}: missing "
            f"`args.auto_destroy_stale` branch — the auto-destroy "
            f"path is the only place the dual-threshold matters."
        )
    else:
        # Slice from the auto_destroy_stale branch onwards. We don't
        # need exact AST analysis — substring presence in the rest of
        # the file is sufficient evidence the path consults each
        # threshold.
        idx = text.find("args.auto_destroy_stale")
        tail = text[idx:]

        # 3. SETUP-side: must reference either the per-health setup
        # age field OR the CLI flag.
        if (
            "setup_age_minutes" not in tail
            and "setup_stale_minutes" not in tail
        ):
            violations.append(
                f"{target.relative_to(root)}: auto-destroy branch "
                f"doesn't reference `setup_age_minutes` or "
                f"`setup_stale_minutes` — SETUP-stuck instances "
                f"will leak cost. Add a stuck-SETUP filter to the "
                f"to_destroy list."
            )

        # 4. IDLE-side: must still classify on IDLE/CRASHED.
        if '"IDLE"' not in tail and "'IDLE'" not in tail:
            violations.append(
                f"{target.relative_to(root)}: auto-destroy branch "
                f"doesn't reference the IDLE classification — half of "
                f"the dual-threshold pattern is gone."
            )

    if verbose:
        if violations:
            print(
                f"  [verify-vast-dual-threshold] "
                f"{len(violations)} violation(s):"
            )
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [verify-vast-dual-threshold] OK: "
                f"--stale-minutes (IDLE) AND --setup-stale-minutes "
                f"(SETUP) both wired into auto-destroy"
            )

    if violations and strict:
        raise MetaBugViolation(
            "VERIFY_VAST_INSTANCES DUAL-THRESHOLD VIOLATIONS — the "
            "auto-destroy path must use BOTH --stale-minutes (IDLE "
            "heartbeat freshness) AND --setup-stale-minutes "
            "(SETUP first-seen age). Dropping either half re-introduces "
            "the SETUP-stuck cost-leak class.\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 57 (57th meta-bug): scripts/remote_lane_*.sh git-sync MUST use the
#                          canonical fetch+reset pattern, NOT bare
#                          `git pull --ff-only`.
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: Lane Q-FAITHFUL (highest-EV lane, predicted [0.40, 0.80]) crashed
# with `FATAL: git pull failed -- remote has uncommitted/conflicting changes`
# after Vast.ai reused a workspace from a prior failed deploy. `git pull
# --ff-only` aborts on uncommitted local junk; the canonical fix is
#
#   git fetch origin main && git reset --hard origin/main
#
# which discards local divergence and syncs to origin/main exactly. ANY future
# refactor that re-introduces bare `git pull --ff-only` (without a SAME-LINE
# `# GIT_SYNC_OPT_OUT:<reason>` waiver) fails preflight at commit/PR time.
#
# This check enforces:
#   1. Any lane script that performs git sync (uses `git pull`, `git fetch`,
#      OR `git reset` against origin) MUST use the canonical fetch+reset
#      pattern.
#   2. Bare `git pull --ff-only` is FORBIDDEN unless a SAME-LINE waiver
#      `# GIT_SYNC_OPT_OUT:<reason>` is present.
#   3. Lane scripts that do NO git sync at all are exempt (they trust the
#      parent launcher to deploy a clean checkout).
#
# Live count after Fix 1 (canonical-pattern landing): 0.

def check_lane_scripts_use_canonical_git_sync(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Forbid lane scripts from using fragile `git pull --ff-only` which
    aborts on stale Vast.ai workspaces. Require the canonical
    `git fetch origin main && git reset --hard origin/main` pattern.

    Scans ``scripts/remote_lane_*.sh``.

    Waiver: same-line ``# GIT_SYNC_OPT_OUT:<reason>`` marker on the bare
    `git pull --ff-only` line.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    violations: list[str] = []

    if not scripts_dir.is_dir():
        if verbose:
            print(
                f"  [canonical-git-sync] SKIP: "
                f"{scripts_dir} not present"
            )
        return violations

    lane_scripts = sorted(scripts_dir.glob("remote_lane_*.sh"))
    if not lane_scripts:
        if verbose:
            print(
                f"  [canonical-git-sync] SKIP: "
                f"no remote_lane_*.sh scripts found"
            )
        return violations

    # Accept both bare form and `git -C <path>` form (e.g.,
    # `git -C "$WORKSPACE" fetch origin main`).
    import re as _re
    canonical_re_a = _re.compile(r"\bgit(?:\s+-C\s+\S+)?\s+fetch\s+origin\s+main\b")
    canonical_re_b = _re.compile(r"\bgit(?:\s+-C\s+\S+)?\s+reset\s+--hard\s+origin/main\b")
    waiver_substr = "# GIT_SYNC_OPT_OUT:"

    for script in lane_scripts:
        try:
            text = script.read_text(errors="ignore")
        except OSError as e:
            violations.append(
                f"{script.relative_to(root)}: cannot read — {e}"
            )
            continue

        # Walk lines and flag any non-comment `git pull --ff-only` that
        # lacks a same-line waiver. Track waivered lines separately so a
        # file-level waiver also exempts the file-level canonical-pattern
        # check below.
        offending_lines: list[tuple[int, str]] = []
        file_has_waiver = False
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.lstrip()
            # Skip pure-comment lines — they're documentation, not code.
            if stripped.startswith("#"):
                continue
            if "git pull --ff-only" not in raw_line:
                continue
            # Same-line waiver allows opt-out.
            if waiver_substr in raw_line:
                file_has_waiver = True
                continue
            offending_lines.append((lineno, raw_line.strip()))

        if offending_lines:
            for lineno, line in offending_lines:
                violations.append(
                    f"{script.relative_to(root)}:{lineno}: bare "
                    f"`git pull --ff-only` is FORBIDDEN — replace with "
                    f"`git fetch origin main && git reset --hard origin/main` "
                    f"or add same-line `# GIT_SYNC_OPT_OUT:<reason>` waiver. "
                    f"Line: {line}"
                )
            continue

        # If a same-line waiver was found, the operator has explicitly
        # opted out of the canonical pattern — exempt the file.
        if file_has_waiver:
            continue

        # If the script does ANY git sync (pull/fetch/reset against origin),
        # enforce that the canonical pattern is present.
        does_git_sync = (
            "git pull" in text
            or "git fetch" in text
            or ("git reset" in text and "origin" in text)
        )
        if not does_git_sync:
            # Lane script trusts parent launcher — fine.
            continue

        if not (canonical_re_a.search(text) and canonical_re_b.search(text)):
            violations.append(
                f"{script.relative_to(root)}: performs git sync but does "
                f"NOT use the canonical `git fetch origin main && "
                f"git reset --hard origin/main` pattern. Stale Vast.ai "
                f"workspaces will crash on bare `git pull --ff-only` "
                f"(memory: feedback_canonical_git_sync_pattern_20260428)."
            )

    if verbose:
        if violations:
            print(
                f"  [canonical-git-sync] "
                f"{len(violations)} violation(s):"
            )
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [canonical-git-sync] OK: "
                f"all {len(lane_scripts)} lane script(s) either skip git "
                f"sync OR use canonical fetch+reset pattern"
            )

    if violations and strict:
        raise MetaBugViolation(
            "CANONICAL GIT SYNC VIOLATIONS — lane scripts must use "
            "`git fetch origin main && git reset --hard origin/main` "
            "(NOT bare `git pull --ff-only` which crashes on stale "
            "Vast.ai workspaces).\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 58 (58th meta-bug): launcher offer-search --max-dph must NOT be
#                          hardcoded below 0.40, which would over-restrict
#                          the host pool and starve the search (today's
#                          NVDEC_BAD on 87% of 4090s burned ~$10 because
#                          the survivor pool was tiny).
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: deep hardening pass 3 dimension 2. The launcher's
# argparse default (`p1.add_argument("--max-dph", type=float, default=0.50)`)
# is broad enough that the search returns ~5 offers reliably. But operators
# (or downstream calling scripts) sometimes hardcode a tighter cap to chase
# cheaper instances; this check forbids that for any value below 0.40 so the
# survivor pool is always > ~3 hosts even after NVDEC_BAD attrition.
#
# Static scan only: looks for `--max-dph <value>` and `max_dph=<value>` in
# scripts/launch_lane_on_vastai.py and any caller under scripts/. Same-line
# `# MAX_DPH_OK:<reason>` waiver allowed for known-safe cases.

def check_launcher_max_dph_floor(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    floor: float = 0.40,
) -> list[str]:
    """Forbid hardcoded launcher --max-dph below the floor (default 0.40).

    Scans scripts/launch_lane_on_vastai.py + scripts/*.sh for hardcoded
    --max-dph or max_dph= values; flags any below the floor without a
    same-line `# MAX_DPH_OK:<reason>` waiver.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    violations: list[str] = []
    if not scripts_dir.is_dir():
        if verbose:
            print(f"  [launcher-max-dph-floor] SKIP: {scripts_dir} not present")
        return violations

    import re as _re
    pat_cli = _re.compile(r"--max-dph[= ]([0-9]+\.?[0-9]*)")
    pat_kw = _re.compile(r"\bmax_dph\s*=\s*([0-9]+\.?[0-9]*)")
    # argparse default like `default=0.30` (only when --max-dph is on the same line)
    pat_default = _re.compile(r"--max-dph.*?\bdefault\s*=\s*([0-9]+\.?[0-9]*)")
    waiver = "# MAX_DPH_OK:"

    targets = sorted(scripts_dir.glob("launch_lane_on_vastai.py")) + sorted(
        scripts_dir.glob("*.sh")
    )
    for path in targets:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.lstrip()
            if stripped.startswith("#"):
                continue
            if waiver in raw_line:
                continue
            matched = False
            for pat in (pat_cli, pat_kw, pat_default):
                m = pat.search(raw_line)
                if not m:
                    continue
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                if val < floor:
                    violations.append(
                        f"{path.relative_to(root)}:{lineno}: hardcoded "
                        f"--max-dph={val} is below the {floor} floor — too few "
                        f"hosts after NVDEC_BAD attrition. Raise the cap or add "
                        f"same-line `{waiver}<reason>` waiver. Line: {raw_line.strip()}"
                    )
                    matched = True
                    break  # don't double-report the same line
            if matched:
                continue

    if verbose:
        if violations:
            print(f"  [launcher-max-dph-floor] {len(violations)} violation(s):")
            for v in violations[:10]:
                print(f"    • {v}")
        else:
            print(
                f"  [launcher-max-dph-floor] OK: no hardcoded --max-dph below "
                f"{floor} across launcher + lane scripts"
            )

    if violations and strict:
        raise MetaBugViolation(
            f"LAUNCHER --max-dph BELOW FLOOR ({floor}) — pool too small for "
            f"NVDEC attrition. Raise cap or waive.\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 59 (59th meta-bug): launcher cmd_phase2_extract MUST auto-destroy
#                          the instance on CUDA-probe failure (idle cost
#                          accrues otherwise).
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: deep hardening pass 3 dimension 2. The launcher's
# phase2-extract calls `lightweight_nvdec_probe(host, port)` and on failure
# MUST call `destroy_instance(instance_id)` (unless --no-destroy-on-fail
# is explicitly set). Today's session lost an instance ~$0.05 because an
# earlier version of the function let the operator's terminal session end
# without destroying.
#
# Static scan: parse cmd_phase2_extract function body and verify both
# `lightweight_nvdec_probe` AND `destroy_instance` are referenced inside
# the function.

def check_phase2_extract_destroys_on_failure(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify cmd_phase2_extract destroys the instance on CUDA-probe failure.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    launcher = root / "scripts" / "launch_lane_on_vastai.py"
    violations: list[str] = []
    if not launcher.exists():
        if verbose:
            print("  [phase2-extract-cleanup] SKIP: launcher not present")
        return violations
    try:
        text = launcher.read_text()
    except OSError:
        return violations
    # Find the function body for cmd_phase2_extract
    import re as _re
    m = _re.search(
        r"def cmd_phase2_extract\([^)]*\)[^:]*:\n((?:    [^\n]*\n|\n)+)",
        text,
    )
    if not m:
        violations.append(
            "scripts/launch_lane_on_vastai.py: cmd_phase2_extract function "
            "definition not found — has the launcher been refactored? Update "
            "this check or restore the function."
        )
    else:
        body = m.group(1)
        if "lightweight_nvdec_probe" not in body:
            violations.append(
                "scripts/launch_lane_on_vastai.py:cmd_phase2_extract: missing "
                "`lightweight_nvdec_probe(...)` call — Stage 2 CUDA probe is "
                "the canonical NVDEC_BAD detection step."
            )
        if "destroy_instance" not in body:
            violations.append(
                "scripts/launch_lane_on_vastai.py:cmd_phase2_extract: missing "
                "`destroy_instance(...)` call — failed CUDA probe must auto-"
                "destroy the instance to stop cost accrual (unless "
                "--no-destroy-on-fail is set explicitly)."
            )

    if verbose:
        if violations:
            print(f"  [phase2-extract-cleanup] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                "  [phase2-extract-cleanup] OK: cmd_phase2_extract probes NVDEC "
                "AND destroys on failure"
            )

    if violations and strict:
        raise MetaBugViolation(
            "PHASE2-EXTRACT MUST AUTO-DESTROY ON CUDA-PROBE FAILURE.\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 60 (60th meta-bug): MEMORY.md must stay under 250 lines (warns
#                          when exceeded — the auto-memory file accumulates
#                          across sessions and silently bloats context).
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: deep hardening pass 3 dimension 2. The Claude Code system
# message warns at 200 lines: "Only part of it was loaded. Keep index
# entries to one line under ~200 chars; move detail into topic files." We
# adopt 250 as a soft ceiling (50-line buffer) so the operator gets warned
# before the loader truncates context silently.
#
# Heuristic: hunt for MEMORY.md under either Claude home (`~/.claude/...`)
# or repo root. Flag if line count > ceiling.

def check_memory_md_size_under_ceiling(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    ceiling: int = 250,
) -> list[str]:
    """Warn when MEMORY.md exceeds the soft line-count ceiling.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    import os
    candidates: list[Path] = []
    home_memory = (
        Path.home()
        / ".claude" / "projects"
        / "-Users-adpena-Projects-pact" / "memory" / "MEMORY.md"
    )
    if home_memory.exists():
        candidates.append(home_memory)
    root = repo_root or REPO_ROOT
    repo_memory = root / "MEMORY.md"
    if repo_memory.exists():
        candidates.append(repo_memory)

    violations: list[str] = []
    for path in candidates:
        try:
            n = sum(1 for _ in path.open("r", errors="ignore"))
        except OSError:
            continue
        if n > ceiling:
            violations.append(
                f"{path}: {n} lines (> {ceiling} ceiling). Consolidate index "
                f"entries to one line each (move detail into topic files), or "
                f"prune obsolete entries to keep context windows from "
                f"silently truncating the file."
            )

    if verbose:
        if violations:
            print(f"  [memory-md-size] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            if candidates:
                print(
                    f"  [memory-md-size] OK: {len(candidates)} MEMORY.md file(s) "
                    f"all under {ceiling} lines"
                )
            else:
                print("  [memory-md-size] SKIP: no MEMORY.md found")

    if violations and strict:
        raise MetaBugViolation(
            f"MEMORY.md EXCEEDS {ceiling}-LINE CEILING.\n"
            + "\n".join(f"  • {v}" for v in violations[:5])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 61 (61st meta-bug): canonical lane bootstraps (remote_train_bootstrap.sh
#                          + remote_pose_tto_bootstrap.sh) MUST write
#                          provenance.json (git_hash + gpu_name + cost_cap +
#                          predicted_band fields).
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: deep hardening pass 3 dimension 2. Memory:
# `feedback_canonical_remote_bootstraps`. The 2 canonical bootstrap scripts
# (and any new variants) MUST write a provenance.json file at the START of
# their run so post-mortem analysis on Vast.ai instances has a deterministic
# anchor. Lane scripts (remote_lane_*.sh) call these bootstraps; the
# bootstrap is responsible for writing provenance.
#
# Static scan: look for `provenance.json` writes in canonical bootstrap
# scripts. Currently warn-only since broader audit needed for all
# remote_lane_*.sh that bypass the canonical bootstraps.

def check_canonical_bootstraps_write_provenance(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify canonical bootstrap scripts write provenance.json.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    violations: list[str] = []
    if not scripts_dir.is_dir():
        if verbose:
            print("  [bootstrap-provenance] SKIP: scripts/ not present")
        return violations
    canonical = [
        "remote_train_bootstrap.sh",
        "remote_pose_tto_bootstrap.sh",
        "remote_pose_tto_only_bootstrap.sh",
    ]
    n_checked = 0
    for name in canonical:
        path = scripts_dir / name
        if not path.exists():
            continue
        n_checked += 1
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "provenance.json" not in text:
            violations.append(
                f"scripts/{name}: does not write provenance.json — required "
                f"for post-mortem traceability per "
                f"feedback_canonical_remote_bootstraps."
            )
            continue
        # Look for the required fields anywhere in the script body.
        required_fields = ["git_hash", "gpu_name"]
        missing = [f for f in required_fields if f not in text]
        if missing:
            violations.append(
                f"scripts/{name}: provenance.json write is present but missing "
                f"fields {missing}. Required: git_hash, gpu_name. "
                f"Recommended: cost_cap, predicted_band (lane-specific)."
            )

    if verbose:
        if violations:
            print(f"  [bootstrap-provenance] {len(violations)} violation(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(
                f"  [bootstrap-provenance] OK: {n_checked} canonical bootstrap(s) "
                f"write provenance.json with required fields"
            )

    if violations and strict:
        raise MetaBugViolation(
            "CANONICAL BOOTSTRAPS MUST WRITE provenance.json.\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ════════════════════════════════════════════════════════════════════════════
# Check 63 (63rd meta-bug): every lane script that calls contest_auth_eval.py
#                          MUST verify config.env exists with PYTHON_INFLATE=
#                          renderer BEFORE the call (or rely on the canonical
#                          guard inside contest_auth_eval.py itself, which
#                          F5 added).
# ════════════════════════════════════════════════════════════════════════════
#
# 2026-04-28: Codex F5. Lane RM-d ran 1+ hour pose TTO, built archive, then
# crashed at Stage 3 contest_auth_eval because submissions/robust_current/
# config.env was not on the remote (the launcher tarball silently excluded
# .env files). inflate.sh fell into its ffmpeg path and tried to open
# extracted/0.mkv which never exists in a renderer-archive layout.
#
# The canonical fix is now layered:
#  1. scripts/launch_lane_on_vastai.py includes .env in the tarball suffix list
#  2. experiments/contest_auth_eval.py hard-fails if config.env is missing
#  3. THIS CHECK ensures lane scripts call the GUARDED contest_auth_eval (not
#     a stale local copy) and don't try to bypass the guard.
#
# Static scan: grep every scripts/remote_lane_*.sh for `contest_auth_eval`
# and verify the script either (a) calls the canonical
# experiments/contest_auth_eval.py (which has the guard) OR (b) has its own
# `config.env` / `PYTHON_INFLATE` precondition check before the eval call.
#
# Live count at wire-in: 0 (verified post-F5 fix). Ships STRICT.

def check_lane_scripts_set_up_inflate_environment(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Verify lane scripts that call contest_auth_eval set up the env correctly.

    Every scripts/remote_lane_*.sh that invokes contest_auth_eval MUST
    either:
      (a) Call experiments/contest_auth_eval.py (which has the F5 guard
          for missing config.env), OR
      (b) Have its own pre-check that verifies submissions/robust_current/
          config.env exists with PYTHON_INFLATE=renderer.

    Catches the F5 bug class: lanes that train + build archive successfully
    but crash at Stage 3 because the inflate environment is incomplete.
    Reference: feedback_codex_review_5_findings_FIXED_20260428 +
    Lane RM-d 0.mkv crash post-mortem.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    violations: list[str] = []
    n_scanned = 0
    if not scripts_dir.is_dir():
        if verbose:
            print("  [lane-inflate-env] SKIP: scripts/ not present")
        return violations

    canonical_module_substr = "experiments/contest_auth_eval.py"
    canonical_guard_grep = "PYTHON_INFLATE=renderer"

    for path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        n_scanned += 1
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        # Skip scripts that do NOT invoke contest_auth_eval at all
        if "contest_auth_eval" not in text:
            continue
        # Acceptance path (a): calls the canonical experiments/contest_auth_eval.py
        # which has the F5 guard built in.
        if canonical_module_substr in text:
            continue
        # Acceptance path (b): has its own PYTHON_INFLATE=renderer pre-check
        if canonical_guard_grep in text:
            continue
        # Otherwise this lane bypasses both guards — flag it.
        rel = str(path.relative_to(root))
        violations.append(
            f"{rel}: calls contest_auth_eval but neither (a) routes through "
            f"experiments/contest_auth_eval.py (which has the F5 config.env "
            f"guard) nor (b) checks PYTHON_INFLATE=renderer locally. The lane "
            f"may train successfully then crash at Stage 3 with extracted/0.mkv "
            f"missing. See Codex F5 (2026-04-28)."
        )

    if verbose:
        if violations:
            print(f"  [lane-inflate-env] {len(violations)} violation(s) across "
                  f"{n_scanned} remote_lane_*.sh file(s):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [lane-inflate-env] OK: {n_scanned} remote_lane_*.sh "
                  f"scripts checked; all set up inflate env correctly")

    if violations and strict:
        raise MetaBugViolation(
            "LANE SCRIPTS MUST SET UP INFLATE ENV (Codex F5 2026-04-28).\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ── Check 64: lane scripts must have a recent E2E smoke proof ─────────────────
#
# Reference: feedback_canonical_e2e_smoke_PERMANENT_GUARD_20260428.
#
# The structural gap this check closes: 63 STRICT preflight checks before
# Check 64 are STATIC analysis — code-pattern guards. None of them actually
# run the deploy → inflate → contest_auth_eval pipeline locally. A lane can
# pass every static check and still ship to Vast.ai with a broken pipeline.
#
# Lane RM-d (2026-04-28) is the canonical example: trained 3.5h on Vast.ai,
# built archive successfully, then crashed at Stage 3 because the inflate.sh
# ffmpeg path tried to read extracted/0.mkv (file that never exists in a
# renderer archive). The F5 fix in contest_auth_eval.py closes that specific
# bug, but the structural gap — "we never proved the lane will actually
# inflate end-to-end before dispatch" — remained.
#
# Check 64 enforces: every scripts/remote_lane_*.sh must have an entry in
# .omx/state/lane_e2e_smoke_proofs.json that is < 7 days old. The proof is
# written by experiments/canonical_local_auth_eval_smoke.py, which runs the
# full pipeline locally against a known-good fixture archive.
#
# Operators MUST run the smoke before dispatching a new lane. Without a
# proof, the preflight FAILS, blocking the dispatch.


SMOKE_PROOFS_REL = ".omx/state/lane_e2e_smoke_proofs.json"
SMOKE_PROOF_MAX_AGE_DAYS = 7


def check_lane_scripts_have_e2e_smoke_proof(
    repo_root: Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Verify every scripts/remote_lane_*.sh has a recent E2E smoke proof.

    A smoke proof is an entry in .omx/state/lane_e2e_smoke_proofs.json
    written by experiments/canonical_local_auth_eval_smoke.py. Each proof
    asserts the lane's archive would inflate cleanly through the canonical
    pipeline (extract → whitelist → renderer-magic → masks → config.env →
    inflate.sh dispatch → inflate_renderer.py imports → upstream/evaluate.py
    arity → GT video present → launcher includes .env).

    Acceptance paths per lane:
      (a) Proof exists with timestamp_utc < SMOKE_PROOF_MAX_AGE_DAYS old.
      (b) Lane script has same-line `# E2E_SMOKE_OPT_OUT:<reason>` comment
          (for lanes that genuinely cannot be smoke-tested locally — e.g.
          require 60GB GPU memory for archive build).

    Otherwise the lane FAILS this check.

    Returns list of violations. Raises MetaBugViolation if strict and any.
    """
    import datetime as _dt
    import json as _json

    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    proofs_path = root / SMOKE_PROOFS_REL
    violations: list[str] = []
    n_scanned = 0
    n_proven = 0
    n_waived = 0

    if not scripts_dir.is_dir():
        if verbose:
            print("  [e2e-smoke-proof] SKIP: scripts/ not present")
        return violations

    # Load proofs file (may not exist on a fresh repo). A missing file means
    # ZERO proofs — every lane will violate. That is by design: the operator
    # must run canonical_local_auth_eval_smoke.py at least once.
    proofs: dict = {}
    if proofs_path.exists():
        try:
            proofs = _json.loads(proofs_path.read_text())
            if not isinstance(proofs, dict):
                proofs = {}
        except (_json.JSONDecodeError, OSError):
            proofs = {}

    now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(days=SMOKE_PROOF_MAX_AGE_DAYS)

    for path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        n_scanned += 1
        lane_name = path.stem  # e.g. "remote_lane_g_v3_corrected_kl_weight"
        rel = str(path.relative_to(root))

        # Acceptance path (b): same-line opt-out waiver
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            text = ""
        if "# E2E_SMOKE_OPT_OUT:" in text:
            # Require a non-empty reason after the colon (anchor: at least 4
            # chars to discourage `# E2E_SMOKE_OPT_OUT:.` placeholder).
            import re as _re
            m = _re.search(r"#\s*E2E_SMOKE_OPT_OUT:\s*(\S.*)", text)
            if m and len(m.group(1).strip()) >= 4:
                n_waived += 1
                continue
            violations.append(
                f"{rel}: has '# E2E_SMOKE_OPT_OUT:' marker but no reason "
                f"(must be at least 4 chars)"
            )
            continue

        # Acceptance path (a): proof exists + recent
        proof = proofs.get(lane_name)
        if proof is None:
            violations.append(
                f"{rel}: no smoke proof in {SMOKE_PROOFS_REL} "
                f"(run: python experiments/canonical_local_auth_eval_smoke.py "
                f"--lane {lane_name})"
            )
            continue

        ts_str = proof.get("timestamp_utc")
        if not ts_str:
            violations.append(
                f"{rel}: proof exists but missing 'timestamp_utc' field "
                f"(corrupt proof — re-run smoke)"
            )
            continue

        try:
            # Parse the canonical UTC ISO timestamp written by the smoke tool.
            ts = _dt.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
            ts = ts.replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            violations.append(
                f"{rel}: proof has malformed timestamp_utc={ts_str!r} "
                f"(re-run smoke)"
            )
            continue

        if ts < cutoff:
            age_days = (now - ts).days
            violations.append(
                f"{rel}: smoke proof too old ({age_days} days, max "
                f"{SMOKE_PROOF_MAX_AGE_DAYS}). Re-run: python "
                f"experiments/canonical_local_auth_eval_smoke.py --lane "
                f"{lane_name}"
            )
            continue

        n_proven += 1

    if verbose:
        if violations:
            print(f"  [e2e-smoke-proof] {len(violations)} violation(s) across "
                  f"{n_scanned} remote_lane_*.sh file(s) "
                  f"(proven={n_proven} waived={n_waived}):")
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    ... and {len(violations) - 20} more")
        else:
            print(f"  [e2e-smoke-proof] OK: {n_scanned} remote_lane_*.sh "
                  f"scripts checked (proven={n_proven} waived={n_waived})")

    if violations and strict:
        raise MetaBugViolation(
            "LANE SCRIPTS MUST HAVE E2E SMOKE PROOF (Check 64 — closes the "
            "static-vs-pipeline gap that cost Lane RM-d 3.5h GPU on the "
            "0.mkv crash). Run:\n"
            "  python experiments/canonical_local_auth_eval_smoke.py "
            "--backfill-all\n"
            "to regenerate proofs for every lane.\n\nViolations:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
        )
    return violations


# ----------------------------------------------------------------------------
# Check 65 — lane-class auto-scan for pipeline proof
# ----------------------------------------------------------------------------
# Background: Lane RM-d (2026-04-28) crashed at the auth_eval stage AFTER 3.5h
# of training on a remote Vast.ai instance. The crash exposed a structural
# gap: while we have ~64 STATIC preflight checks for code patterns, no check
# verifies that a NEW LANE CLASS (e.g., the first "renderer-replacement" or
# "pose-replacement" lane) actually completed a full
# dispatch → train → archive → auth_eval cycle anywhere on record. New lane
# classes can ship into the codebase, run for hours on Vast.ai, and crash at
# auth_eval — and no preflight catches that BEFORE the GPU spend.
#
# Check 65 enforces: every lane CLASS in scripts/remote_lane_*.sh must have at
# least one proof in .omx/state/lane_class_proofs.json showing a complete
# pipeline cycle. The proof can come from (a) a real production deploy that
# landed an authoritative score or (b) a `--proof-only` Modal/local dry-run
# that demonstrated the pipeline end-to-end.

LANE_CLASS_PROOFS_REL = ".omx/state/lane_class_proofs.json"

# Mapping from filename keyword to canonical lane class. Edit here when new
# classes emerge; the scanner picks the FIRST match in declaration order, so
# put more-specific keywords above generic ones.
_LANE_CLASS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("pose_tto", "pose-tto"),
    ("pose_replacement", "pose-replacement"),
    ("posenet_distill", "pose-distill"),
    ("renderer_replacement", "renderer-replacement"),
    ("renderer_distill", "renderer-distill"),
    ("halfframe", "halfframe-mask"),
    ("entropy_archive", "entropy-archive"),
    ("archive_codec", "archive-codec"),
    ("cool_chic", "cool-chic-sidecar"),
    ("self_compress", "self-compress"),
    ("uniward", "uniward-distortion"),
    ("calibrated_pe", "calibrated-pe"),
    ("hessian", "hessian-bit-allocator"),
    ("lagrangian", "lagrangian-rate-distortion"),
    ("kl_distill", "kl-distill"),
    ("kl_weight", "kl-distill"),
    ("kldistill", "kl-distill"),
    ("fp4_qat", "fp4-qat"),
    ("fp8", "fp8-quant"),
    ("mae", "mae-pretrain"),
    ("optimized", "renderer-optimized"),
    ("sweep", "sweep-orchestrator"),
    ("rescue", "rescue-recovery"),
    ("training", "training-baseline"),
    ("smoke", "smoke-only"),
)


def _classify_lane_script(path: Path) -> str:
    """Return the canonical lane class for a remote_lane_*.sh path.

    Heuristic: lowercase the stem, normalize separators, and pick the first
    matching keyword from _LANE_CLASS_KEYWORDS. Falls back to "uncategorized"
    so the check always assigns a class (the proof still has to exist).
    """
    stem = path.stem.lower().replace("-", "_")
    for kw, cls in _LANE_CLASS_KEYWORDS:
        if kw in stem:
            return cls
    return "uncategorized"


def check_lane_classes_have_pipeline_proof(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify every lane CLASS has at least one complete-pipeline proof.

    Acceptance: a class is "proven" when ``.omx/state/lane_class_proofs.json``
    contains an entry like::

        {
          "renderer-replacement": {
            "proven_by_lane": "lane_d_v3_full_engineering",
            "proof_kind": "production-deploy",       // or "modal-dry-run"
            "score": 1.05,                           // optional but recommended
            "score_lane_tag": "[contest-CUDA]",      // CLAUDE.md non-neg
            "timestamp_utc": "2026-04-28T22:07:00Z",
            "notes": "Lane G v3 corrected KL weight, archive 694 KB"
          },
          ...
        }

    A new lane CLASS without a proof = FAIL. This catches the Lane RM-d class
    of bug PERMANENTLY: the first time a brand-new lane class ships, the
    operator MUST register a proof or the launcher refuses to deploy.

    SHIPS WARN-ONLY initially (strict=False) so the existing 70 lanes have
    a backfill window. Promotion plan: backfill _LANE_CLASS_PROOFS_REL with
    one proof per existing class (~10-15 entries), then flip strict=True via
    the standard Lane A → strict pattern.
    """
    import json as _json

    root = repo_root or REPO_ROOT
    scripts_dir = root / "scripts"
    proofs_path = root / LANE_CLASS_PROOFS_REL
    violations: list[str] = []

    if not scripts_dir.is_dir():
        if verbose:
            print("  [lane-class-proof] SKIP: scripts/ not present")
        return violations

    # Collect (class -> example_lane) for every remote_lane_*.sh.
    classes: dict[str, list[str]] = {}
    for path in sorted(scripts_dir.glob("remote_lane_*.sh")):
        cls = _classify_lane_script(path)
        classes.setdefault(cls, []).append(path.stem)

    if not classes:
        if verbose:
            print("  [lane-class-proof] SKIP: no remote_lane_*.sh found")
        return violations

    # Load proofs (missing file => zero proofs => every class violates).
    proofs: dict = {}
    if proofs_path.exists():
        try:
            data = _json.loads(proofs_path.read_text())
            if isinstance(data, dict):
                proofs = data
        except (_json.JSONDecodeError, OSError):
            proofs = {}

    n_proven = 0
    for cls, lanes in sorted(classes.items()):
        proof = proofs.get(cls)
        if not proof or not isinstance(proof, dict):
            example = lanes[0]
            violations.append(
                f"lane class {cls!r} has no proof in {LANE_CLASS_PROOFS_REL} "
                f"(example lane: {example}). Register one via Modal "
                f"(experiments/modal_auth_eval.py) or canonical local smoke."
            )
            continue
        # Soft schema: require proven_by_lane + timestamp_utc at minimum.
        if not proof.get("proven_by_lane"):
            violations.append(
                f"lane class {cls!r} proof missing 'proven_by_lane' field"
            )
            continue
        if not proof.get("timestamp_utc"):
            violations.append(
                f"lane class {cls!r} proof missing 'timestamp_utc' field"
            )
            continue
        n_proven += 1

    if verbose:
        if violations:
            print(
                f"  [lane-class-proof] {len(violations)} violation(s) across "
                f"{len(classes)} lane class(es) (proven={n_proven}):"
            )
            for v in violations[:20]:
                print(f"    • {v}")
            if len(violations) > 20:
                print(f"    ... and {len(violations) - 20} more")
        else:
            print(
                f"  [lane-class-proof] OK: {len(classes)} lane class(es) "
                f"all proven"
            )

    if violations and strict:
        raise MetaBugViolation(
            "LANE CLASSES MUST HAVE PIPELINE PROOF (Check 65 — closes the "
            "Lane RM-d class of bug). New lane classes shipping without a "
            "complete dispatch → train → archive → auth_eval proof on file. "
            "Add an entry to .omx/state/lane_class_proofs.json — see check "
            "docstring for schema.\n\nViolations:\n"
            + "\n".join(f"  • {v}" for v in violations[:50])
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
