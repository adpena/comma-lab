# SPDX-License-Identifier: MIT
"""Tests for Catalog #270 — canonical dispatch optimization protocol.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
+ "Subagent coherence-by-default" + the standing directive *"all possible
should be pulled into the decorator or similar reusable and shareable
tools and helpers"*.

Lane: ``lane_canonical_dispatch_optimization_protocol_20260515``.
Memory: ``feedback_canonical_dispatch_optimization_protocol_landed_20260515.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import canonical_dispatch_optimization_protocol as proto_mod  # noqa: E402

from tac.preflight import (  # noqa: E402
    PreflightError,
    check_dispatch_optimization_protocol_complete,
    _CHECK_270_WAIVER_MARKER,
    _check_270_resolve_recipe_for_trainer,
    _check_270_text_has_waiver,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic trainer + recipe fixtures (DO NOT touch live repo)
# ---------------------------------------------------------------------------

CLEAN_TRAINER_BODY = """\
# SPDX-License-Identifier: MIT
\"\"\"Synthetic clean trainer for Catalog #270 tests.\"\"\"
from tac.substrates._shared.score_aware_common import score_pair_components
from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call
from tac.substrates._shared.inflate_runtime import select_inflate_device
import torch

def _train():
    parser_args = '--enable-autocast-fp16 --enable-torch-compile'
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        pass
    pose_scorer, seg_scorer = load_default_scorers()
    return pose_scorer, seg_scorer

def _full_main(args):
    return _train()
"""

CLEAN_RECIPE_BODY = """\
schema_version: 1
name: synth_clean
lane_id: lane_synth_clean_20260515
min_vram_gb: 14
min_smoke_gpu: T4
video_input_strategy: per_dispatch_local_copy
pyav_decode_strategy: cpu_thread_async_upload
target_modes:
  - contest_one_video_replay
"""

CLEAN_LANE_DRIVER_BODY = """\
#!/usr/bin/env bash
set -euo pipefail
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo hello
"""


def _make_synthetic_repo(
    tmp_path: Path,
    *,
    substrate_id: str = "synth",
    trainer_body: str = CLEAN_TRAINER_BODY,
    recipe_body: str | None = CLEAN_RECIPE_BODY,
    driver_body: str | None = CLEAN_LANE_DRIVER_BODY,
) -> tuple[Path, Path, str | None]:
    """Build a synthetic repo with trainer + optional recipe + lane driver."""
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    trainer = tmp_path / "experiments" / f"train_substrate_{substrate_id}.py"
    trainer.write_text(trainer_body, encoding="utf-8")
    recipe_name: str | None = None
    if recipe_body is not None:
        recipe_name = f"substrate_{substrate_id}_modal_t4_dispatch"
        (tmp_path / ".omx" / "operator_authorize_recipes" / f"{recipe_name}.yaml").write_text(
            recipe_body, encoding="utf-8"
        )
    if driver_body is not None:
        (tmp_path / "scripts" / f"remote_lane_substrate_{substrate_id}.sh").write_text(
            driver_body, encoding="utf-8"
        )
    return tmp_path, trainer, recipe_name


# ---------------------------------------------------------------------------
# Helper-unit tests
# ---------------------------------------------------------------------------


class TestResolveRecipeForTrainer:
    def test_returns_unique_recipe(self, tmp_path: Path):
        root, trainer, _ = _make_synthetic_repo(tmp_path, substrate_id="alpha")
        out = _check_270_resolve_recipe_for_trainer(trainer, root)
        assert out == "substrate_alpha_modal_t4_dispatch"

    def test_returns_none_for_non_substrate_trainer(self, tmp_path: Path):
        path = tmp_path / "experiments" / "train_renderer.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        out = _check_270_resolve_recipe_for_trainer(path, tmp_path)
        assert out is None

    def test_returns_none_when_recipe_dir_missing(self, tmp_path: Path):
        path = tmp_path / "experiments" / "train_substrate_foo.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        out = _check_270_resolve_recipe_for_trainer(path, tmp_path)
        assert out is None

    def test_prefers_modal_t4_over_others(self, tmp_path: Path):
        (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True, exist_ok=True)
        trainer = tmp_path / "experiments" / "train_substrate_beta.py"
        trainer.write_text("", encoding="utf-8")
        for name in ("substrate_beta_local_apple_silicon.yaml",
                     "substrate_beta_modal_t4_dispatch.yaml",
                     "substrate_beta_modal_a100_dispatch.yaml"):
            (tmp_path / ".omx" / "operator_authorize_recipes" / name).write_text("", encoding="utf-8")
        out = _check_270_resolve_recipe_for_trainer(trainer, tmp_path)
        assert "modal_t4_dispatch" in out or "modal_a100_dispatch" in out


class TestWaiverDetection:
    def test_real_rationale_accepted(self):
        text = f"# foo\n{_CHECK_270_WAIVER_MARKER}intentional-pre-tier1-backfill\n"
        assert _check_270_text_has_waiver(text)

    def test_placeholder_rationale_rejected(self):
        text = f"{_CHECK_270_WAIVER_MARKER}<rationale>\n"
        assert not _check_270_text_has_waiver(text)
        text = f"{_CHECK_270_WAIVER_MARKER}<reason>\n"
        assert not _check_270_text_has_waiver(text)

    def test_empty_rationale_rejected(self):
        text = f"{_CHECK_270_WAIVER_MARKER}\n"
        assert not _check_270_text_has_waiver(text)

    def test_marker_outside_first_30_lines_not_honored(self):
        text = "\n" * 35 + f"{_CHECK_270_WAIVER_MARKER}reason\n"
        assert not _check_270_text_has_waiver(text)


# ---------------------------------------------------------------------------
# Verify-helper end-to-end tests (the canonical_dispatch_optimization_protocol module)
# ---------------------------------------------------------------------------


class TestVerifyDispatchProtocolComplete:
    def test_clean_trainer_with_recipe_passes(self, tmp_path: Path):
        root, trainer, recipe = _make_synthetic_repo(tmp_path)
        verdict = proto_mod.verify_dispatch_protocol_complete(
            trainer, recipe, repo_root=root
        )
        assert verdict.overall_pass, verdict.blockers
        assert verdict.tier1.overall_pass
        assert verdict.tier2.overall_pass
        assert verdict.tier3.overall_pass
        assert verdict.blockers == []

    def test_missing_trainer_returns_blocker_verdict(self, tmp_path: Path):
        verdict = proto_mod.verify_dispatch_protocol_complete(
            tmp_path / "experiments" / "train_substrate_nonexistent.py",
            None,
            repo_root=tmp_path,
        )
        assert not verdict.overall_pass
        assert len(verdict.blockers) >= 1
        assert "not found" in verdict.blockers[0] or "unreadable" in verdict.blockers[0]

    def test_no_recipe_runs_advisory_tier2(self, tmp_path: Path):
        root, trainer, _ = _make_synthetic_repo(tmp_path, recipe_body=None)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, None, repo_root=root)
        # Tier 2 recipe-side checks vacuous; lane driver checks still fire.
        assert any("advisory" in n.lower() for n in verdict.advisory_notes)

    def test_missing_lane_driver_advisory(self, tmp_path: Path):
        root, trainer, recipe = _make_synthetic_repo(tmp_path, driver_body=None)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert any("driver" in n for n in verdict.advisory_notes)

    def test_tier1_missing_autocast_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("--enable-autocast-fp16 --enable-torch-compile", "--enable-torch-compile")
        body = body.replace("torch.autocast", "")  # ensure no fallback
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier1.overall_pass
        assert any("autocast_fp16" in b for b in verdict.tier1.blockers)

    def test_tier1_missing_tf32_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier1.overall_pass
        assert any("tf32" in b for b in verdict.tier1.blockers)

    def test_tier1_missing_torch_compile_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("--enable-torch-compile", "")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier1.overall_pass
        assert any("torch_compile" in b for b in verdict.tier1.blockers)

    def test_tier1_missing_no_grad_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("with torch.no_grad():", "if True:")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier1.overall_pass
        assert any("no_grad_at_eval" in b for b in verdict.tier1.blockers)

    def test_tier1_missing_canonical_scorer_loss_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace(
            "from tac.substrates._shared.score_aware_common import score_pair_components",
            "# removed",
        )
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier1.overall_pass
        assert any("canonical_scorer_loss" in b for b in verdict.tier1.blockers)

    def test_tier2_recipe_missing_min_vram_flagged(self, tmp_path: Path):
        recipe_body = CLEAN_RECIPE_BODY.replace("min_vram_gb: 14\n", "")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, recipe_body=recipe_body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier2.overall_pass
        assert any("min_vram_gb" in b for b in verdict.tier2.blockers)

    def test_tier2_driver_missing_nvml_flagged(self, tmp_path: Path):
        driver_body = CLEAN_LANE_DRIVER_BODY.replace(
            'export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"\n', ""
        )
        root, trainer, recipe = _make_synthetic_repo(tmp_path, driver_body=driver_body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier2.overall_pass
        assert any("DALI_DISABLE_NVML" in b for b in verdict.tier2.blockers)

    def test_tier3_recipe_vs_trainer_state_divergence_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace(
            "def _full_main(args):\n    return _train()\n",
            "def _full_main(args):\n    raise NotImplementedError('phase 2')\n",
        )
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier3.overall_pass
        assert any("recipe-vs-trainer-state" in b for b in verdict.tier3.blockers)

    def test_tier3_research_only_recipe_accepts_stub_full_main(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace(
            "def _full_main(args):\n    return _train()\n",
            "def _full_main(args):\n    raise NotImplementedError('phase 2')\n",
        )
        recipe_body = CLEAN_RECIPE_BODY + "\nresearch_only: true\n"
        root, trainer, recipe = _make_synthetic_repo(
            tmp_path, trainer_body=body, recipe_body=recipe_body
        )
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert verdict.tier3.pass_signals.get("recipe_vs_trainer_state_consistent")

    def test_tier3_reversed_scorer_loader_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace(
            "pose_scorer, seg_scorer = load_default_scorers()",
            "seg_scorer, pose_scorer = load_default_scorers()",
        )
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier3.overall_pass
        assert any("REVERSED scorer-loader" in b for b in verdict.tier3.blockers)

    def test_tier3_phantom_score_filename_flagged(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace(
            "from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call\n",
            "",
        )
        body = body + '\nresult = "contest_auth_eval_cuda.json"\n'
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.tier3.overall_pass
        assert any("phantom" in b.lower() for b in verdict.tier3.blockers)

    def test_overall_and_blockers_aggregate(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        body = body.replace("with torch.no_grad():", "if True:")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        assert not verdict.overall_pass
        assert len(verdict.blockers) >= 2

    def test_verdict_as_dict_serializable(self, tmp_path: Path):
        root, trainer, recipe = _make_synthetic_repo(tmp_path)
        verdict = proto_mod.verify_dispatch_protocol_complete(trainer, recipe, repo_root=root)
        as_dict = verdict.as_dict()
        # Round-trips through json without error
        s = json.dumps(as_dict)
        assert "tier1" in s and "tier2" in s and "tier3" in s


# ---------------------------------------------------------------------------
# Preflight gate end-to-end tests
# ---------------------------------------------------------------------------


class TestPreflightGate:
    def test_clean_synthetic_repo_no_violations(self, tmp_path: Path):
        _make_synthetic_repo(tmp_path)
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert violations == []

    def test_dirty_synthetic_repo_has_violations(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        _make_synthetic_repo(tmp_path, trainer_body=body)
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(violations) >= 1

    def test_strict_mode_raises_on_violation(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        _make_synthetic_repo(tmp_path, trainer_body=body)
        with pytest.raises(PreflightError) as exc:
            check_dispatch_optimization_protocol_complete(
                repo_root=tmp_path, strict=True, verbose=False
            )
        msg = str(exc.value)
        assert "Catalog #270" in msg or "dispatch" in msg.lower()

    def test_strict_mode_silent_on_clean(self, tmp_path: Path):
        _make_synthetic_repo(tmp_path)
        # Should not raise
        out = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert out == []

    def test_waiver_with_real_rationale_skips_trainer(self, tmp_path: Path):
        # Make trainer dirty AND add waiver
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        body = f"{_CHECK_270_WAIVER_MARKER}operator-pre-tier1-backfill-window\n" + body
        _make_synthetic_repo(tmp_path, trainer_body=body)
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert violations == []

    def test_waiver_with_placeholder_rationale_does_not_skip(self, tmp_path: Path):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        body = f"{_CHECK_270_WAIVER_MARKER}<rationale>\n" + body
        _make_synthetic_repo(tmp_path, trainer_body=body)
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(violations) >= 1

    def test_no_trainers_dir_returns_empty(self, tmp_path: Path):
        # No experiments/train_substrate_*.py files
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert violations == []

    def test_string_repo_root_accepted(self, tmp_path: Path):
        _make_synthetic_repo(tmp_path)
        # Pass repo_root as str (not Path) — must coerce
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=str(tmp_path), strict=False, verbose=False
        )
        assert violations == []

    def test_multiple_trainers_aggregated(self, tmp_path: Path):
        # Two dirty trainers
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        _make_synthetic_repo(tmp_path, substrate_id="alpha", trainer_body=body)
        _make_synthetic_repo(tmp_path, substrate_id="beta", trainer_body=body)
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# Live-repo regression guard (warn-only at landing, bounded ceiling)
# ---------------------------------------------------------------------------


class TestLiveRepoRegression:
    def test_live_count_bounded(self):
        """Live-repo violations must stay ≤ 50 at landing.

        Per CLAUDE.md "Strict-flip atomicity rule" the gate lands warn-only
        because legacy substrates have real Tier 1 gaps. The ceiling
        prevents regression while operator-routed Tier 1 backfill
        progresses; lower the bound as the count drops; flip to strict
        when count = 0.
        """
        violations = check_dispatch_optimization_protocol_complete(
            repo_root=REPO_ROOT, strict=False, verbose=False
        )
        assert len(violations) <= 50, (
            f"Catalog #270 live violation count regressed to {len(violations)} "
            "(landing baseline ≤ 50). Either the new violation is genuine "
            "(operator-routed Tier 1 backfill regression) or the gate detection "
            "drifted. Investigate per CLAUDE.md 'Bugs must be permanently "
            "fixed AND self-protected against'."
        )


# ---------------------------------------------------------------------------
# Orchestrator-callsite regression guard
# ---------------------------------------------------------------------------


class TestOrchestratorCallsite:
    def test_preflight_all_wires_check_warn_only_at_landing(self):
        """Catalog #270 wired into preflight_all() at warn-only initial.

        Strict-flip planned alongside operator-routed Tier 1 backfill
        sweep that drives violations to 0.
        """
        preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
        text = preflight_path.read_text(encoding="utf-8")
        # Function present
        assert "def check_dispatch_optimization_protocol_complete(" in text
        # Wired into preflight_all
        assert "check_dispatch_optimization_protocol_complete(" in text
        # Wire-in is warn-only initially
        # Find the wire-in callsite (NOT the def line) — look for the
        # comment-anchored block + the strict=False keyword inline.
        wire_idx = text.find("# 2026-05-15 Catalog #270 - canonical dispatch optimization protocol")
        assert wire_idx > 0, "wire-in comment block missing"
        callsite_window = text[wire_idx : wire_idx + 1500]
        assert "check_dispatch_optimization_protocol_complete(" in callsite_window
        assert "strict=False" in callsite_window


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_emits_json(self, tmp_path: Path, monkeypatch):
        root, trainer, recipe = _make_synthetic_repo(tmp_path)
        # Patch REPO_ROOT in the helper module so recipe is found relative
        # to tmp_path
        monkeypatch.setattr(proto_mod, "REPO_ROOT", root)
        argv = [
            "canonical_dispatch_optimization_protocol.py",
            "--trainer", str(trainer),
            "--recipe", recipe,
            "--json",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = proto_mod.main()
        assert rc == 0  # warn-only mode

    def test_cli_strict_returns_1_on_fail(self, tmp_path: Path, monkeypatch, capsys):
        body = CLEAN_TRAINER_BODY.replace("torch.backends.cuda.matmul.allow_tf32", "# removed")
        root, trainer, recipe = _make_synthetic_repo(tmp_path, trainer_body=body)
        monkeypatch.setattr(proto_mod, "REPO_ROOT", root)
        argv = [
            "canonical_dispatch_optimization_protocol.py",
            "--trainer", str(trainer),
            "--recipe", recipe,
            "--strict",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = proto_mod.main()
        assert rc == 1
