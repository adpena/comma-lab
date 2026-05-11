"""Tests for the meta-bug preflight checks (CLAUDE.md FORBIDDEN PATTERNS).

Each pattern below has bitten this project at least once and cost real GPU
money. Tests confirm the scanner catches the bad form and passes the clean
form. NEW FILE — does not extend test_preflight_dead_resolvers.py (that file
is being modified by a parallel subagent).
"""
from __future__ import annotations

import concurrent.futures
import inspect
import json
import os
import time
import textwrap
from pathlib import Path

import pytest

import tac.preflight as preflight_mod
from tac.eval_roundtrip_gate import enforce_eval_roundtrip
from tac.preflight import (
    REPO_ROOT,
    MetaBugViolation,
    PreflightError,
    _scan_doc_for_untagged_scores,
    _scan_for_cpu_fallback_in_subagent_prompts,
    _scan_for_halfframe_without_trained_profile,
    _scan_for_uniward_delta_without_attestation,
    _scan_for_unspecific_waivers,
    _scan_for_vastai_prompt_no_cost_cap,
    _scan_inflate_for_scorer_load,
    _scan_inflate_sh_for_centralized_brotli,
    _scan_lightning_ssh_static_policy,
    _scan_python_for_disable_eval_roundtrip_flag,
    _scan_python_for_eval_roundtrip_false,
    _scan_python_for_kl_div_batchmean,
    _scan_python_for_mps_fallback,
    _scan_python_for_pack_sparse_delta_approved,
    _scan_python_for_silent_auto_discovery,
    _scan_python_for_vastai_create_no_label,
    _scan_python_for_vastai_create_no_tracker,
    _scan_remote_script_for_nvdec_probe,
    _scan_remote_script_for_plain_cmg3a_dispatch,
    _scan_remote_script_for_plain_pmg_dispatch,
    _scan_shell_for_missing_set_e,
    _scan_shell_for_pipefail_grep_q,
    _scan_shell_for_pipefail_tee_pipestatus_loss,
    _scan_shell_for_zip_binary,
    _scan_submission_for_provider_or_cpu_score_leakage,
    _scan_test_file_for_dead_imports,
    _scan_training_script_for_auth_eval,
    check_cmg3a_remote_dispatch_requires_pose_safety,
    check_evidence_row_has_falsification_scope_when_negative,
    check_halfframe_archive_uses_trained_profile,
    check_inflate_scorer_load_has_runtime_banner,
    check_inflate_sh_handles_br_centrally,
    check_kl_div_reduction_correct,
    check_lightning_exact_eval_manifest_runtime_closure,
    check_lightning_exact_eval_runner_bootstraps_dali,
    check_lightning_ssh_static_policy,
    check_no_active_mcp_server_config,
    check_no_compromised_lightning_supply_chain,
    check_no_disable_eval_roundtrip_flag,
    check_no_eval_roundtrip_false,
    check_no_live_mcp_processes,
    check_no_mps_fallback_default,
    check_no_pack_sparse_delta_approved_outside_promotion_tool,
    check_no_pipefail_grep_q_trap,
    check_no_pipefail_tee_pipestatus_loss,
    check_no_raw_zip_extractall,
    check_no_scorer_load_at_inflate,
    check_no_shell_zip_binary,
    check_no_shadowed_module_import_used_before_local_import,
    check_no_silent_auto_discovery_with_warn,
    check_no_submission_provider_or_cpu_score_leakage,
    check_pmg_remote_dispatch_requires_geometry_escape,
    check_profile_keys_have_resolvers,
    check_public_release_hygiene,
    check_remote_scripts_have_nvdec_probe,
    check_remote_scripts_write_provenance,
    check_scores_have_lane_tag,
    check_shell_set_e_present,
    check_subagent_prompts_no_cpu_fallback,
    check_test_files_imports_resolve,
    check_training_scripts_have_auth_eval,
    check_uniward_delta_has_attestation_gate,
    check_vastai_create_has_label,
    check_vastai_create_writes_tracker,
    check_vastai_prompts_have_cost_cap,
    check_waivers_specify_env_gate,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip("\n"))


def _stub_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo root."""
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ─── Check: shadowed import before use ────────────────────────────────────


class TestShadowedImportBeforeUse:
    def test_direct_shadowed_import_before_use_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "bad_shadow.py", """
            from tac.losses import _hwc_to_chw

            def train(frame):
                out = _hwc_to_chw(frame)
                from tac.losses import _hwc_to_chw
                return out
        """)

        violations = check_no_shadowed_module_import_used_before_local_import(
            repo_root=root,
            strict=False,
            verbose=False,
        )
        assert len(violations) == 1
        assert "bad_shadow.py:5" in violations[0]
        assert "_hwc_to_chw" in violations[0]

    def test_local_import_before_use_is_clean(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "good_shadow.py", """
            from tac.losses import _hwc_to_chw

            def train(frame):
                from tac.losses import _hwc_to_chw
                return _hwc_to_chw(frame)
        """)

        assert check_no_shadowed_module_import_used_before_local_import(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_nested_function_import_does_not_shadow_outer_scope(
        self, tmp_path: Path,
    ) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "nested_shadow.py", """
            from tac.losses import _hwc_to_chw

            def outer(frame):
                out = _hwc_to_chw(frame)

                def inner(x):
                    from tac.losses import _hwc_to_chw
                    return _hwc_to_chw(x)

                return inner(out)
        """)

        assert check_no_shadowed_module_import_used_before_local_import(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_incremental_cache_invalidates_changed_file(
        self, tmp_path: Path,
    ) -> None:
        root = _stub_repo(tmp_path)
        target = root / "src" / "tac" / "cached_shadow.py"
        _write(target, """
            from tac.losses import _hwc_to_chw

            def train(frame):
                return _hwc_to_chw(frame)
        """)

        assert check_no_shadowed_module_import_used_before_local_import(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []
        assert (
            root / ".omx" / "cache" / "shadowed_import_before_use_clean.json"
        ).exists()

        _write(target, """
            from tac.losses import _hwc_to_chw

            def train(frame):
                out = _hwc_to_chw(frame)
                from tac.losses import _hwc_to_chw
                return out
        """)

        violations = check_no_shadowed_module_import_used_before_local_import(
            repo_root=root,
            strict=False,
            verbose=False,
        )
        assert len(violations) == 1
        assert "cached_shadow.py" in violations[0]


# ─── Check: CMG3A Pose-collapse dispatch guard ──────────────────────────────


class TestCmg3aPoseSafetyDispatchGuard:
    def test_plain_target_body_remote_dispatch_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_bad_cmg3a.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            python experiments/build_cmg3_adaptive_runs_candidate.py \\
              --frontier-archive anchor.zip \\
              --decoded-mask-array masks.npy \\
              --output-dir out \\
              --target-body-bytes 166000
        """)
        violations = _scan_remote_script_for_plain_cmg3a_dispatch(script, root)
        assert violations
        with pytest.raises(MetaBugViolation, match="CMG3A REMOTE DISPATCH"):
            check_cmg3a_remote_dispatch_requires_pose_safety(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_field_policy_remote_dispatch_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_good_cmg3a.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            python experiments/build_cmg3_adaptive_runs_candidate.py \\
              --frontier-archive anchor.zip \\
              --decoded-mask-array masks.npy \\
              --output-dir out \\
              --target-body-bytes 166000 \\
              --field-policy-json pose_safe_plan.json \\
              --field-policy-id top0128
        """)
        assert _scan_remote_script_for_plain_cmg3a_dispatch(script, root) == []
        assert check_cmg3a_remote_dispatch_requires_pose_safety(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_multimask_run_count_needs_review_marker(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_multimask.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            python experiments/build_c067_multimask_reconciler_candidate.py \\
              --plan-json plan.json \\
              --frontier-archive anchor.zip \\
              --output-dir out \\
              --target-extra-runs 72000
        """)
        assert _scan_remote_script_for_plain_cmg3a_dispatch(script, root)

        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            python experiments/build_c067_multimask_reconciler_candidate.py \\
              --plan-json plan.json \\
              --frontier-archive anchor.zip \\
              --output-dir out \\
              --target-extra-runs 72000 \\
              # CMG3A_POSE_COLLAPSE_REVIEWED: exact negatives reviewed; dispatch only with new pose-safe stack rationale
        """)
        assert _scan_remote_script_for_plain_cmg3a_dispatch(script, root) == []


# ─── Check: PMG row-span exact-negative dispatch guard ──────────────────────


class TestPmgGeometryEscapeDispatchGuard:
    def test_plain_pmg_archive_only_dispatch_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_bad_pmg.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            export ARCHIVE_PATH="$PWD/experiments/results/pmg_hotspot_candidate/archive.zip"
            bash scripts/remote_archive_only_eval.sh
        """)
        violations = _scan_remote_script_for_plain_pmg_dispatch(script, root)
        assert violations
        with pytest.raises(MetaBugViolation, match="PMG REMOTE DISPATCH"):
            check_pmg_remote_dispatch_requires_geometry_escape(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_geometry_escape_reviewed_pmg_dispatch_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_good_pmg.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            # PMG_GEOMETRY_ESCAPE_REVIEWED: learned pose-conditioned decoder replaces raw row-span geometry
            python scripts/launch_lightning_batch_job.py submit-exact-eval \\
              --archive experiments/results/pmg_hotspot_pose_safe/archive.zip \\
              --artifact experiments/results/predictive_mask_grammar_runtime_readiness_20260502/predictive_mask_grammar_runtime_readiness_plan.json
        """)
        assert _scan_remote_script_for_plain_pmg_dispatch(script, root) == []
        assert check_pmg_remote_dispatch_requires_geometry_escape(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_historical_negative_replay_guard_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "remote_lane_old_pmg_replay.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${ALLOW_REPLAY_EXACT_NEGATIVE_PMG:-0}" != "1" ]]; then
              echo "PMG_EXACT_NEGATIVE_REPLAY_GUARD: refusing accidental replay"
              exit 88
            fi
            export ARCHIVE_PATH="$PWD/experiments/results/pmg_hotspot_candidate/archive.zip"
            bash scripts/remote_archive_only_eval.sh
        """)
        assert _scan_remote_script_for_plain_pmg_dispatch(script, root) == []


# ─── Check 0: Lightning PyPI compromise guard ───────────────────────────────


class TestNoCompromisedLightningSupplyChain:
    def test_bad_lightning_pin_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "requirements.txt", """
            lightning==2.6.3
        """)
        with pytest.raises(MetaBugViolation, match="COMPROMISED LIGHTNING"):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    @pytest.mark.parametrize(
        "spec",
        [
            "lightning==2.6.1",
            "lightning>=2.6.2",
            "lightning~=2.6",
            "lightning<2.6.4",
            "lightning[extra]==2.6.3",
            "lightning @ https://files.pythonhosted.org/packages/x/lightning-2.6.3-py3-none-any.whl",
        ],
    )
    def test_any_pypi_lightning_dependency_is_caught(self, tmp_path: Path, spec: str) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "requirements.txt", spec + "\n")
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_uv_lock_split_name_version_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "uv.lock", """
            [[package]]
            name = "lightning"
            version = "2.6.3"
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_waiver_text_cannot_suppress_known_bad_pin(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "requirements.txt", """
            lightning==2.6.3  # lightning-pypi-ok: this must not waive malware
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_bare_lightning_install_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "install_lightning.sh", """
            #!/usr/bin/env bash
            uv pip install lightning
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_lightning_sdk_install_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "install_lightning_sdk.sh", """
            #!/usr/bin/env bash
            uv pip install lightning-sdk
        """)
        v = check_no_compromised_lightning_supply_chain(
            repo_root=root,
            site_packages_roots=[],
            strict=True,
            verbose=False,
        )
        assert v == []

    def test_lightning_version_probe_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "probe_lightning.py", """
            import subprocess
            subprocess.run(["lightning", "--version"], check=True)
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    @pytest.mark.parametrize(
        "command",
        [
            ".venv/bin/lightning connect studio --name demo",
            "LIGHTNING=.venv/bin/lightning",
            "$LIGHTNING cp lit://workspace/out out",
            "lightning list studios",
        ],
    )
    def test_lightning_console_script_is_caught_in_tools(self, tmp_path: Path, command: str) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "tools" / "stale_lightning_cli.sh", f"""
            #!/usr/bin/env bash
            {command}
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_installed_bad_dist_info_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        site = tmp_path / "site-packages"
        dist = site / "lightning-2.6.2.dist-info"
        _write(dist / "METADATA", """
            Name: lightning
            Version: 2.6.2
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[site],
                strict=True,
                verbose=False,
            )

    def test_installed_clean_bare_lightning_dist_info_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        site = tmp_path / "site-packages"
        dist = site / "lightning-2.6.4.dist-info"
        _write(dist / "METADATA", """
            Name: lightning
            Version: 2.6.4
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[site],
                strict=True,
                verbose=False,
            )

    def test_installed_bad_lightning_init_hash_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _stub_repo(tmp_path)
        site = tmp_path / "site-packages"
        bad_init = site / "lightning" / "__init__.py"
        _write(bad_init, "# malicious import-time trigger\n")
        original_sha256_file = preflight_mod._sha256_file

        def fake_sha256(path: Path) -> str | None:
            if path == bad_init:
                return "2d4e21d2e78d0868ce7894487e67c67f929d8d81d78c5b07a3ad225b13eae890"
            return original_sha256_file(path)

        monkeypatch.setattr(preflight_mod, "_sha256_file", fake_sha256)

        with pytest.raises(MetaBugViolation, match="malicious lightning/__init__"):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[site],
                strict=True,
                verbose=False,
            )

    def test_cached_bad_lightning_wheel_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        cache = tmp_path / "pip-cache"
        _write(cache / "wheels" / "lightning-2.6.3-py3-none-any.whl", "placeholder\n")

        with pytest.raises(MetaBugViolation, match=r"cached Lightning 2\.6\.2/2\.6\.3"):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                package_cache_roots=[cache],
                strict=True,
                verbose=False,
            )

    def test_cached_bad_pytorch_lightning_sdist_is_caught_nested(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        cache = tmp_path / "uv-cache"
        _write(cache / "archive-v0" / "a" / "b" / "pytorch_lightning-2.6.2.tar.gz", "placeholder\n")

        with pytest.raises(MetaBugViolation, match=r"cached Lightning 2\.6\.2/2\.6\.3"):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                package_cache_roots=[cache],
                strict=True,
                verbose=False,
            )

    def test_package_json_postinstall_ioc_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "docs" / "package.json", '{"scripts":{"postinstall":"node setup.mjs"}}\n')

        with pytest.raises(MetaBugViolation, match=r"package\.json postinstall"):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_reported_lightning_iocs_are_registered(self) -> None:
        assert "d2815d425ae08cc627f1db69009442165f8bbc64b7e9157e2ff9d7aab02094d4" in (
            preflight_mod._MINI_SHAI_HULUD_IOC_SHA256
        )
        assert "2d4e21d2e78d0868ce7894487e67c67f929d8d81d78c5b07a3ad225b13eae890" in (
            preflight_mod._MINI_SHAI_HULUD_IOC_SHA256
        )
        assert "3071422c3294e7b61cb490c57c48c8dea569bacf12e57a078293b6547d7586d3" in (
            preflight_mod._MINI_SHAI_HULUD_IOC_SHA256
        )
        assert "56070a9d8de0c0ffb1ec5c309953cf4679432df5a78df9aeb020fbb73d2be9fb" in (
            preflight_mod._MINI_SHAI_HULUD_IOC_SHA256
        )

    @pytest.mark.parametrize(
        "rel_path",
        [
            ".claude/router_runtime.js",
            ".claude/settings.json",
            ".vscode/tasks.json",
        ],
    )
    def test_planted_repo_ioc_path_is_caught(self, tmp_path: Path, rel_path: str) -> None:
        root = _stub_repo(tmp_path)
        _write(root / rel_path, """
            // planted by a compromised dependency
        """)
        with pytest.raises(MetaBugViolation):
            check_no_compromised_lightning_supply_chain(
                repo_root=root,
                site_packages_roots=[],
                strict=True,
                verbose=False,
            )

    def test_cloud_deploy_never_executes_lightning_cli_for_version_probe(self) -> None:
        source = (Path(__file__).parents[3] / "src" / "tac" / "deploy" / "cloud_deploy.py").read_text()
        assert 'subprocess.run(["lightning", "--version"]' not in source
        assert "importlib.metadata.version(\"lightning-sdk\")" in source
        assert "from tac.deploy.build_bundle import build as build_bundle" in source
        assert "train_tac.py" not in source

    def test_build_bundle_uses_canonical_repo_root_and_deploy_config(self) -> None:
        source = (Path(__file__).parents[3] / "src" / "tac" / "deploy" / "build_bundle.py").read_text()
        assert "REPO = repo_root()" in source
        assert "from tac.deploy.deploy_config import EXPERIMENT_SCRIPT" in source
        assert "experiments/train_tac.py" not in source
        assert "TAC_UPSTREAM_DIR" in source
        assert "TAC_MODELS_DIR" in source
        assert "TAC_RESULTS_DIR" in source
        for stale_flag in (
            "--archive",
            "--gt-video",
            "--saliency",
            "--models-dir",
            "--upstream-dir",
            "--resume-from",
        ):
            assert stale_flag not in source

    def test_modal_archive_contains_no_retired_train_tac_launchers(self) -> None:
        archive_dir = Path(__file__).parents[3] / "src" / "tac" / "deploy" / "modal" / "archive"
        offenders = []
        for path in archive_dir.glob("*.py"):
            if path.name == "__init__.py":
                continue
            source = path.read_text()
            if "experiments/train_tac.py" in source or "/app/train_tac.py" in source:
                offenders.append(path.relative_to(Path(__file__).parents[3]).as_posix())
        assert offenders == []

    def test_operator_surfaces_do_not_route_to_retired_train_tac(self) -> None:
        root = Path(__file__).parents[3]
        operator_surfaces = [
            root / "README.md",
            root / "src" / "tac" / "profiles.py",
            root / "docs" / "research_roadmap.md",
            root / "data" / "artifacts" / "job_queue.json",
        ]
        offenders = []
        for path in operator_surfaces:
            source = path.read_text()
            stale_tokens = (
                "experiments/train_tac.py",
                "python train_tac.py",
                "experiments/modal_h96_deploy.py",
                "experiments/train_postfilter_qat_ema.py",
            )
            if any(token in source for token in stale_tokens):
                offenders.append(path.relative_to(root).as_posix())
        assert offenders == []


class TestLightningExactEvalDaliBootstrap:
    def test_exact_eval_dali_bootstrap_preflight_passes_current_runner(self) -> None:
        assert check_lightning_exact_eval_runner_bootstraps_dali(
            strict=True,
            verbose=False,
        ) == []

    def test_exact_eval_validator_rejects_missing_dali_preflight(self) -> None:
        from tac.deploy.lightning.batch_jobs import LightningAdjudicationSpec, LightningBatchJobSpec

        spec = LightningBatchJobSpec(
            name="bad",
            machine="T4",
            command=(
                "scripts/scan_lightning_supply_chain.py && "
                "python experiments/contest_auth_eval.py --device cuda && "
                "echo LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK && "
                "cp contest_auth_eval.json ."
            ),
            role="exact_cuda_eval",
            expected_archive_sha256="a" * 64,
            expected_archive_size_bytes=123,
            adjudication=LightningAdjudicationSpec(
                baseline_score=1.2,
                predicted_band_low=1.0,
                predicted_band_high=1.4,
                regression_threshold=1.6,
            ),
        )
        with pytest.raises(ValueError, match="DALI runner preflight"):
            spec.validate()


class TestLightningExactEvalManifestRuntimeClosure:
    def test_manifest_runtime_closure_preflight_tracks_cu124_torchvision_pin(self) -> None:
        assert check_lightning_exact_eval_manifest_runtime_closure(
            strict=True,
            verbose=False,
        ) == []


# ─── Check 0c: Lightning SSH static policy guard ────────────────────────────


class TestLightningSshStaticPolicy:
    def test_lightning_script_disabling_host_key_checking_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "lightning_bad.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            ssh -o StrictHostKeyChecking=no lightning-pact true
        """)

        violations = _scan_lightning_ssh_static_policy(script, root)

        assert any("StrictHostKeyChecking" in item for item in violations)

    def test_lightning_script_null_known_hosts_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "tools" / "lightning_bad.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            scp -o UserKnownHostsFile=/dev/null artifact lightning-pact:/tmp/
        """)

        violations = check_lightning_ssh_static_policy(root, strict=False, verbose=False)

        assert any("known_hosts" in item for item in violations)

    def test_lightning_runbook_bare_provider_target_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "docs" / "runbooks" / "lightning_bad.md", """
            ```bash
            export LIGHTNING_SSH_TARGET=ssh.lightning.ai
            ssh ssh.lightning.ai true
            ```
        """)

        with pytest.raises(MetaBugViolation, match="LIGHTNING SSH STATIC POLICY"):
            check_lightning_ssh_static_policy(root, strict=True, verbose=False)

    def test_lightning_ssh_alias_config_doc_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "docs" / "runbooks" / "lightning_good.md", """
            ```sshconfig
            Host lightning-pact
              HostName ssh.lightning.ai
              User <studio-ssh-user>
              IdentityFile ~/.ssh/lightning_pact
              IdentitiesOnly yes
              BatchMode yes
              StrictHostKeyChecking accept-new
            ```
        """)

        assert check_lightning_ssh_static_policy(root, strict=True, verbose=False) == []

    def test_lightning_provider_host_constant_for_alias_generation_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "configure_lightning_ssh.py", """
            DEFAULT_HOST = "ssh.lightning.ai"
            def render():
                return f"HostName {DEFAULT_HOST}"
        """)

        assert check_lightning_ssh_static_policy(root, strict=True, verbose=False) == []

    def test_lightning_deploy_tree_is_scanned(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "src" / "tac" / "deploy" / "lightning" / "bad.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            ssh -o StrictHostKeyChecking=no lightning-pact true
        """)

        violations = check_lightning_ssh_static_policy(root, strict=False, verbose=False)

        assert any("src/tac/deploy/lightning/bad.sh" in item for item in violations)

    def test_vast_scripts_are_out_of_scope_for_lightning_ssh_policy(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "launch_lane_on_vastai.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            ssh -o StrictHostKeyChecking=no root@ssh5.vast.ai true
        """)

        assert check_lightning_ssh_static_policy(root, strict=True, verbose=False) == []


# ─── Check 0d: Submission provider/CPU score leakage guard ──────────────────


class TestSubmissionProviderCpuScoreLeakage:
    def test_submission_helper_provider_host_and_cpu_score_are_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "submissions" / "robust_current" / "download_and_eval.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            REMOTE_HOST="${REMOTE_HOST:-alice@ssh.lightning.ai}"
            SCP_ARGS=(-o StrictHostKeyChecking=no)
            python3 "$SELF_DIR/runner.py" evaluate \\
              --upstream-dir upstream \\
              --skip-compress \\
              --device cpu
        """)

        violations = _scan_submission_for_provider_or_cpu_score_leakage(script, root)

        assert any("provider hostname" in item for item in violations)
        assert any("host-key" in item for item in violations)
        assert any("--device cpu/mps" in item for item in violations)
        with pytest.raises(MetaBugViolation, match="SUBMISSION PROVIDER/CPU SCORE"):
            check_no_submission_provider_or_cpu_score_leakage(root, strict=True, verbose=False)

    def test_submission_helper_cuda_alias_path_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "submissions" / "robust_current" / "download_and_eval.sh"
        _write(script, """
            #!/usr/bin/env bash
            set -euo pipefail
            REMOTE_TARGET="${REMOTE_TARGET:?set ssh alias}"
            SCP_ARGS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
            python3 "$SELF_DIR/runner.py" evaluate \\
              --upstream-dir upstream \\
              --skip-compress \\
              --device cuda
        """)

        assert _scan_submission_for_provider_or_cpu_score_leakage(script, root) == []
        assert check_no_submission_provider_or_cpu_score_leakage(root, strict=True, verbose=False) == []


# ─── Check 0b: MCP server config remains disabled ───────────────────────────


class TestNoActiveMcpServerConfig:
    def test_empty_json_mcp_servers_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / ".claude" / "mcp.json", """
            {"mcpServers": {}}
        """)
        assert check_no_active_mcp_server_config(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_active_json_mcp_servers_are_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / ".claude" / "mcp.json", """
            {
              "mcpServers": {
                "chrome": {"command": "chrome-devtools-mcp"}
              }
            }
        """)
        with pytest.raises(MetaBugViolation, match="ACTIVE MCP"):
            check_no_active_mcp_server_config(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_codex_toml_mcp_server_section_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / ".codex" / "config.toml", """
            [mcp_servers.roblox]
            command = "rbx-studio-mcp"
        """)
        with pytest.raises(MetaBugViolation, match="ACTIVE MCP"):
            check_no_active_mcp_server_config(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_json_helper_token_outside_mcp_servers_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / ".cursor" / "settings.json", """
            {
              "mcpServers": {},
              "terminal.integrated.env.osx": {
                "MCP_COMMAND": "chrome-devtools-mcp"
              }
            }
        """)
        with pytest.raises(MetaBugViolation, match="ACTIVE MCP"):
            check_no_active_mcp_server_config(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_repo_vscode_mcp_config_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / ".vscode" / "mcp.json", """
            {
              "mcpServers": {
                "roblox": {"command": "rbx-studio-mcp"}
              }
            }
        """)
        with pytest.raises(MetaBugViolation, match="ACTIVE MCP"):
            check_no_active_mcp_server_config(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_live_repo_has_no_repo_owned_mcp_config(self) -> None:
        assert check_no_active_mcp_server_config(strict=False, verbose=False) == []


class TestNoLiveMcpProcesses:
    def test_live_helper_process_is_caught(self) -> None:
        rows = [
            "101 /Users/adpena/.cargo/bin/rbx-studio-mcp --stdio",
            "102 npm exec chrome-devtools-mcp@latest --channel stable",
            "103 /bin/zsh -c 'npx chrome-devtools-mcp@latest --channel stable'",
            "104 python -m model.context --stdio",
        ]
        with pytest.raises(MetaBugViolation, match="LIVE MCP"):
            check_no_live_mcp_processes(
                process_rows=rows,
                strict=True,
                verbose=False,
            )

    def test_unrelated_processes_pass(self) -> None:
        rows = [
            "201 /usr/libexec/colorsyncd",
            "202 /bin/zsh -c ps -axo pid=,command=",
            "203 find /Users/adpena -iname *mcp* -o -iname *model.context*",
            "204 /bin/zsh -c find /Users/adpena -iname '*chrome-devtools-mcp*'",
            "205 rg -n chrome-devtools-mcp AGENTS.md scripts",
            "206 python -c 'print(\"model.context\")'",
        ]
        assert check_no_live_mcp_processes(
            process_rows=rows,
            strict=True,
            verbose=False,
        ) == []


# ─── Check 1: MPS-fallback device default ────────────────────────────────────


class TestNoMpsFallbackDefault:
    def test_classic_ternary_chain_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad.py"
        _write(script, """
            import torch
            def pick():
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                return device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v
        assert any("MPS-fallback" in s for s in v)

    def test_env_get_default_with_mps_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_env.py"
        _write(script, """
            import os
            import torch
            def pick():
                # env.get default is the IfExp inside a Call — AST should still catch it
                d = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "mps")
                return d
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v

    def test_plain_cuda_module_check_with_mps_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_plain_cuda.py"
        _write(script, """
            from torch import cuda
            def pick():
                return "cuda" if cuda.is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v

    def test_aliased_cuda_module_check_with_mps_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_alias_cuda.py"
        _write(script, """
            from torch import cuda as torch_cuda
            def pick():
                return "cuda" if torch_cuda.is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v

    def test_getattr_cuda_module_check_with_mps_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_getattr_cuda.py"
        _write(script, """
            import torch
            def pick():
                return "cuda" if getattr(torch, "cuda").is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) >= 1, v

    def test_cuda_required_default_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good.py"
        _write(script, """
            import torch
            def pick():
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA required")
                return "cuda"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], v

    def test_explicit_cpu_optin_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_optin.py"
        _write(script, """
            import argparse
            def pick():
                p = argparse.ArgumentParser()
                p.add_argument("--device", default="cuda")
                args = p.parse_args()
                return args.device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], v

    def test_mps_backend_probe_without_cuda_fallback_skips_ast(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "mps_probe.py"
        _write(script, """
            import torch
            def probe():
                return torch.backends.mps.is_available()
        """)

        def fail_parse(*args, **kwargs):
            raise AssertionError("pure MPS probe should not force AST parsing")

        monkeypatch.setattr(preflight_mod.ast, "parse", fail_parse)

        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], v

    def test_torch_alias_cuda_fallback_still_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "torch_alias_fallback.py"
        _write(script, """
            import torch as th

            def pick_device():
                return "cuda" if th.cuda.is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert len(v) == 1, v
        assert "MPS-fallback" in v[0]

    def test_test_files_are_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_dev.py"
        _write(script, """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], "test files should be skipped"

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        with pytest.raises(MetaBugViolation):
            check_no_mps_fallback_default(repo_root=root, strict=True, verbose=False)

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import torch
            d = "cuda" if torch.cuda.is_available() else "mps"
        """)
        v = check_no_mps_fallback_default(
            repo_root=root,
            strict=False,
            verbose=False,
        )
        assert len(v) >= 1

    def test_source_index_path_catches_formatted_cuda_attr(
        self,
        tmp_path: Path,
    ) -> None:
        from tac.source_index import source_index_context

        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad_wrapped_cuda_attr.py", """
            import torch
            d = (
                "cuda" if torch.cuda
                    .is_available() else "mps"
            )
        """)
        with source_index_context(root):
            v = check_no_mps_fallback_default(
                repo_root=root,
                strict=False,
                verbose=False,
            )
        assert len(v) >= 1

    def test_source_index_shared_scan_reuses_python_predicate_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tac.source_index import source_index_context

        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad_shared_predicates.py", """
            import argparse
            import torch

            device = "cuda" if torch.cuda.is_available() else "mps"

            def train(*, eval_roundtrip=False):
                return device

            parser = argparse.ArgumentParser()
            parser.add_argument("--no-eval-roundtrip", action="store_true")
        """)
        calls = {"mps": 0, "eval": 0, "disable": 0}
        original_mps = preflight_mod._scan_python_for_mps_fallback
        original_eval = preflight_mod._scan_python_for_eval_roundtrip_false
        original_disable = preflight_mod._scan_python_for_disable_eval_roundtrip_flag

        def counting_mps(*args, **kwargs):
            calls["mps"] += 1
            return original_mps(*args, **kwargs)

        def counting_eval(*args, **kwargs):
            calls["eval"] += 1
            return original_eval(*args, **kwargs)

        def counting_disable(*args, **kwargs):
            calls["disable"] += 1
            return original_disable(*args, **kwargs)

        monkeypatch.setattr(preflight_mod, "_scan_python_for_mps_fallback", counting_mps)
        monkeypatch.setattr(
            preflight_mod,
            "_scan_python_for_eval_roundtrip_false",
            counting_eval,
        )
        monkeypatch.setattr(
            preflight_mod,
            "_scan_python_for_disable_eval_roundtrip_flag",
            counting_disable,
        )

        with source_index_context(root):
            mps = check_no_mps_fallback_default(
                repo_root=root,
                strict=False,
                verbose=False,
            )
            eval_roundtrip = check_no_eval_roundtrip_false(
                repo_root=root,
                strict=False,
                verbose=False,
            )
            disable = check_no_disable_eval_roundtrip_flag(
                repo_root=root,
                strict=False,
                verbose=False,
            )

        assert len(mps) >= 1
        assert len(eval_roundtrip) >= 1
        assert len(disable) >= 1
        assert calls == {"mps": 1, "eval": 1, "disable": 1}

    def test_source_index_shared_scan_is_single_build_under_parallel_callers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tac.source_index import source_index_context

        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad_shared_predicates.py", """
            import argparse
            import torch

            device = "cuda" if torch.cuda.is_available() else "mps"

            def train(*, eval_roundtrip=False):
                return device

            parser = argparse.ArgumentParser()
            parser.add_argument("--no-eval-roundtrip", action="store_true")
        """)
        calls = {"mps": 0, "eval": 0, "disable": 0}
        original_mps = preflight_mod._scan_python_for_mps_fallback
        original_eval = preflight_mod._scan_python_for_eval_roundtrip_false
        original_disable = preflight_mod._scan_python_for_disable_eval_roundtrip_flag

        def counting_mps(*args, **kwargs):
            calls["mps"] += 1
            time.sleep(0.02)
            return original_mps(*args, **kwargs)

        def counting_eval(*args, **kwargs):
            calls["eval"] += 1
            time.sleep(0.02)
            return original_eval(*args, **kwargs)

        def counting_disable(*args, **kwargs):
            calls["disable"] += 1
            time.sleep(0.02)
            return original_disable(*args, **kwargs)

        monkeypatch.setattr(preflight_mod, "_scan_python_for_mps_fallback", counting_mps)
        monkeypatch.setattr(
            preflight_mod,
            "_scan_python_for_eval_roundtrip_false",
            counting_eval,
        )
        monkeypatch.setattr(
            preflight_mod,
            "_scan_python_for_disable_eval_roundtrip_flag",
            counting_disable,
        )

        with source_index_context(root) as index:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                results = list(
                    pool.map(
                        lambda _i: preflight_mod._meta_python_shared_scan(root, index),
                        range(3),
                    )
                )

        assert len({id(result) for result in results}) == 1
        assert calls == {"mps": 1, "eval": 1, "disable": 1}

    def test_rg_candidate_prefilter_catches_formatted_cuda_attr(
        self,
        tmp_path: Path,
    ) -> None:
        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_wrapped_cuda_attr.py"
        _write(bad, """
            import torch
            d = (
                "cuda" if torch.cuda
                    .is_available() else "mps"
            )
        """)
        _write(root / "experiments" / "mps_literal_only.py", """
            DEVICE = "mps"
        """)
        _write(root / "experiments" / "cuda_availability_only.py", """
            import torch
            CUDA_OK = torch.cuda.is_available()
        """)

        candidates = preflight_mod._rg_mps_fallback_candidate_files(
            root,
            ["experiments"],
        )
        if candidates is None:
            pytest.skip("ripgrep candidate prefilter unavailable")
        assert bad.resolve() in candidates
        assert (
            root / "experiments" / "mps_literal_only.py"
        ).resolve() not in candidates
        assert (
            root / "experiments" / "cuda_availability_only.py"
        ).resolve() not in candidates

        v = check_no_mps_fallback_default(repo_root=root, strict=False, verbose=False)
        assert len(v) >= 1

    def test_clean_check_cache_skips_mps_rescan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", raising=False)
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "good.py", """
            import torch
            def pick():
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA required")
                return "cuda"
        """)
        assert check_no_mps_fallback_default(repo_root=root, strict=True, verbose=False) == []

        def fail_scan(*args, **kwargs):
            raise AssertionError("clean MPS cache hit should skip rescanning Python ASTs")

        monkeypatch.setattr(preflight_mod, "_scan_python_for_mps_fallback", fail_scan)
        assert check_no_mps_fallback_default(repo_root=root, strict=True, verbose=False) == []

    def test_deleted_path_during_scan_is_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "missing.py"
        assert _scan_python_for_mps_fallback(script, root) == []

    def test_generic_source_iterator_skips_experiment_results(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        source = root / "experiments" / "build_candidate.py"
        result = root / "experiments" / "results" / "candidate" / "inflate.py"
        _write(source, "print('source')\n")
        _write(result, "print('generated')\n")

        paths = preflight_mod._iter_python_files(root, ["experiments"])

        assert source in paths
        assert result not in paths

    def test_python_ast_cache_reused_across_source_scanners(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_cache_probe.py"
        _write(
            script,
            """
            import torch

            def pick_device():
                return "cuda" if torch.cuda.is_available() else "mps"

            def train(*, eval_roundtrip=True):
                return pick_device()
            """,
        )
        preflight_mod._cached_python_text_and_tree.cache_clear()
        original_parse = preflight_mod.ast.parse
        calls = 0

        def counting_parse(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_parse(*args, **kwargs)

        monkeypatch.setattr(preflight_mod.ast, "parse", counting_parse)

        assert len(_scan_python_for_mps_fallback(script, root)) >= 1
        assert _scan_python_for_eval_roundtrip_false(script, root) == []
        assert _scan_python_for_disable_eval_roundtrip_flag(script, root) == []
        assert calls == 1
        preflight_mod._cached_python_text_and_tree.cache_clear()


# ─── Check 2: shell `set -e` required ────────────────────────────────────────


class TestShellSetEPresent:
    def test_set_uo_pipefail_no_e_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -uo pipefail
            ARCHIVE=$(zip out.zip in)
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, v
        assert "set -e" in v[0] or "without `e`" in v[0]

    def test_set_u_alone_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_u.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -u
            X=
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, v

    def test_set_euo_pipefail_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            ARCHIVE=$(python -c "import zipfile; ...")
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], v

    def test_set_e_alone_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_e.sh"
        _write(sh, """
            #!/bin/bash
            set -e
            echo hi
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", """
            #!/usr/bin/env bash
            set -uo pipefail
        """)
        with pytest.raises(MetaBugViolation):
            check_shell_set_e_present(repo_root=root, strict=True, verbose=False)


# ─── Check 3: shell `zip` binary ─────────────────────────────────────────────


class TestNoShellZipBinary:
    def test_zip_invocation_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            zip archive.zip renderer.bin masks.mkv poses.pt
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert len(v) == 1, v
        assert "zip" in v[0] and "zipfile" in v[0]

    def test_python_zipfile_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python -c "import zipfile; zipfile.ZipFile('archive.zip','w').write('renderer.bin')"
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_unzip_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_unzip.sh"
        _write(sh, """
            #!/usr/bin/env bash
            unzip archive.zip -d /tmp
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_zipfile_keyword_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_keyword.sh"
        _write(sh, """
            #!/bin/bash
            python3 my_zipfile_tool.py
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_comments_are_ignored(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "comments.sh"
        _write(sh, """
            #!/bin/bash
            # we used to call zip here, now we use python
            python -c "import zipfile"
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", "zip out.zip in\n")
        with pytest.raises(MetaBugViolation):
            check_no_shell_zip_binary(repo_root=root, strict=True, verbose=False)


# ─── Check 4: pipefail + grep -q SIGPIPE trap ────────────────────────────────


class TestNoPipefailGrepQTrap:
    def test_pipefail_grep_q_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            vastai logs INSTANCE | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, v
        assert "SIGPIPE" in v[0] or "grep -q" in v[0]

    def test_capture_first_idiom_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: `echo "$VAR" | grep -q PAT` is the prescribed
        remediation for the SIGPIPE bug class. Scanner MUST NOT flag it
        — otherwise it blocks its own fix."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(vastai logs INSTANCE 2>&1)
            echo "$OUT" | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"echo | grep -q is the safe form (echo is a builtin, no "
            f"meaningful SIGPIPE) — must not be flagged; got {v}"
        )

    def test_printf_capture_idiom_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: printf is also a builtin — same exemption as echo."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_printf.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(some_cmd)
            printf "%s" "$OUT" | grep -q "ready"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"printf | grep -q must not be flagged (codex R5-3 #6); got {v}"
        )

    def test_here_string_form_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: `grep -q PAT <<< "$VAR"` is the here-string form
        — no pipe at all, so SIGPIPE cannot occur."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_here_string.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            OUT=$(some_cmd)
            grep -q "ready" <<< "$OUT"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"here-string `grep -q PAT <<< \"$OUT\"` has no pipe — must not "
            f"be flagged (codex R5-3 #6); got {v}"
        )

    def test_if_negated_echo_passes(self, tmp_path: Path) -> None:
        """codex R5-3 #6: real-world remote_setup_full.sh form
        `if ! echo "$X" | grep -q PAT` — echo is still a builtin."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_if_neg.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            X=$(some_cmd)
            if ! echo "$X" | grep -q "needle"; then
                echo "missing"
            fi
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"`if ! echo ... | grep -q` is safe (echo is a builtin) — must "
            f"not be flagged (codex R5-3 #6); got {v}"
        )

    def test_unsafe_external_cmd_still_flagged(self, tmp_path: Path) -> None:
        """codex R5-3 #6: unsafe form (external cmd LHS) must STILL fire.
        The exemption is narrow — only echo/printf/here-string."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "still_bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            unzip -l archive.zip | grep -q postfilter.pt
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, (
            f"`unzip | grep -q` is the original bug class — must STILL be "
            f"flagged after the codex R5-3 #6 echo/printf exemption; got {v}"
        )

    def test_no_pipefail_no_violation(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "no_pipefail.sh"
        _write(sh, """
            #!/bin/bash
            # No set -e, no pipefail — grep -q is safe here.
            cat foo | grep -q "bar"
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], v

    def test_grep_without_q_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "grep_no_q.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cmd | grep "pattern" > /dev/null
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            cmd | grep -q "x"
        """)
        with pytest.raises(MetaBugViolation):
            check_no_pipefail_grep_q_trap(repo_root=root, strict=True, verbose=False)


class TestNoPipefailTeePipeStatusLoss:
    def test_tee_pipestatus_under_errexit_pipefail_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_tee.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python remote_eval.py 2>&1 | tee eval.log
            EVAL_RC=${PIPESTATUS[0]}
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert len(v) == 1, v
        assert "PIPESTATUS" in v[0]

    def test_set_plus_e_guarded_tee_pipestatus_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_tee.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            set +e
            python remote_eval.py 2>&1 | tee eval.log
            EVAL_RC=${PIPESTATUS[0]}
            set -e
            exit "${EVAL_RC}"
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert v == [], v

    def test_direct_pipestatus_use_before_capture_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_direct_use.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            set +e
            python remote_eval.py 2>&1 | tee eval.log
            if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                set -e
                exit "${PIPESTATUS[0]}"
            fi
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert len(v) == 1, v
        assert "immediately" in v[0]

    def test_missing_strict_restore_after_capture_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_no_restore.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            set +e
            python remote_eval.py 2>&1 | tee eval.log
            EVAL_RC=${PIPESTATUS[0]}
            if [ "${EVAL_RC}" -ne 0 ]; then
                exit "${EVAL_RC}"
            fi
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert len(v) == 1, v
        assert "restored" in v[0]

    def test_split_pipe_to_tee_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "bad_split_pipe.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python remote_eval.py |
                tee eval.log
            EVAL_RC=${PIPESTATUS[0]}
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert len(v) == 1, v

    def test_tee_tail_pipeline_guarded_capture_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_tee_tail.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            set +e
            python remote_eval.py 2>&1 | tee eval.log | tail -20
            PIPE_RC=("${PIPESTATUS[@]}")
            set -e
            if [ "${PIPE_RC[0]}" -ne 0 ]; then
                exit "${PIPE_RC[0]}"
            fi
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert v == [], v

    def test_set_plus_o_pipefail_guarded_tee_pipestatus_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "good_pipefail_off.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            set +o pipefail
            python remote_eval.py 2>&1 | tee eval.log
            EVAL_RC=${PIPESTATUS[0]}
            set -o pipefail
            exit "${EVAL_RC}"
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert v == [], v

    def test_tee_without_pipestatus_capture_is_not_this_bug_class(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "plain_tee.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            echo "progress" | tee progress.log
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert v == [], v

    def test_tee_pipestatus_inside_heredoc_is_not_flagged(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "doc_tee.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat > README <<'EOF'
            Bad example:
                python remote_eval.py | tee eval.log
                EVAL_RC=${PIPESTATUS[0]}
            EOF
        """)
        v = _scan_shell_for_pipefail_tee_pipestatus_loss(sh, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "bad_tee.sh", """
            #!/usr/bin/env bash
            set -euo pipefail
            cmd | tee out.log
            RC=${PIPESTATUS[0]}
        """)
        with pytest.raises(MetaBugViolation):
            check_no_pipefail_tee_pipestatus_loss(
                repo_root=root, strict=True, verbose=False,
            )


# ─── Check 5: eval_roundtrip=False ───────────────────────────────────────────


class TestNoEvalRoundtripFalse:
    def test_kwarg_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_call.py"
        _write(script, """
            def go():
                train(model, eval_roundtrip=False)
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v
        assert "eval_roundtrip=False" in v[0]

    def test_default_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_def.py"
        _write(script, """
            def train(model, eval_roundtrip: bool = False):
                pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v
        assert "default" in v[0].lower() or "defaults" in v[0].lower()

    def test_default_true_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good.py"
        _write(script, """
            def train(model, eval_roundtrip: bool = True):
                pass
            def go():
                train(m, eval_roundtrip=True)
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert v == [], v

    def test_eval_roundtrip_prefilter_skips_unrelated_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_unrelated_false.py"
        _write(script, """
            def train(model, eval_roundtrip: bool = True):
                unrelated_flag = False
                return unrelated_flag
        """)

        def fail_parse(*args, **kwargs):
            raise AssertionError("unrelated False should not force AST parsing")

        monkeypatch.setattr(preflight_mod.ast, "parse", fail_parse)

        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert v == [], v

    def test_multiline_default_false_still_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_multiline_default.py"
        _write(script, """
            def train(
                model,
                eval_roundtrip: bool =
                    False,
            ):
                pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v
        assert "eval_roundtrip=False" in v[0]

    def test_kwonly_default_false_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "kwonly.py"
        _write(script, """
            def train(model, *, eval_roundtrip=False):
                pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert len(v) == 1, v

    def test_test_files_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_x.py"
        _write(script, """
            def fn(eval_roundtrip=False): pass
        """)
        v = _scan_python_for_eval_roundtrip_false(script, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", "go(eval_roundtrip=False)\n")
        with pytest.raises(MetaBugViolation):
            check_no_eval_roundtrip_false(repo_root=root, strict=True, verbose=False)

    def test_shared_meta_scan_parallelizes_large_candidate_sets(self, monkeypatch) -> None:
        monkeypatch.delenv("PACT_META_PYTHON_SCAN_WORKERS", raising=False)
        assert preflight_mod._meta_python_shared_scan_worker_count(63) == 1
        assert preflight_mod._meta_python_shared_scan_worker_count(64) == 2

        monkeypatch.setenv("PACT_META_PYTHON_SCAN_WORKERS", "2")
        assert preflight_mod._meta_python_shared_scan_worker_count(64) == 2

        monkeypatch.setenv("PACT_META_PYTHON_SCAN_WORKERS", "999")
        assert preflight_mod._meta_python_shared_scan_worker_count(64) == 16

        monkeypatch.setenv("PACT_META_PYTHON_SCAN_WORKERS", "bad")
        assert preflight_mod._meta_python_shared_scan_worker_count(64) == 2


# ─── Check 6: scorer load at inflate ─────────────────────────────────────────


class TestNoScorerLoadAtInflate:
    def test_load_scorers_at_inflate_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            from tac.scorer import load_scorers
            def main():
                segnet, posenet = load_scorers()
                return segnet, posenet
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert len(v) >= 1, v
        assert any("scorer" in s.lower() for s in v)

    def test_renderer_only_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import torch
            def main():
                model = torch.load("renderer.bin", map_location="cpu")
                return model
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert v == [], v

    def test_load_posenet_call_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate_renderer.py"
        _write(inf, """
            from somewhere import load_posenet
            def main():
                pose = load_posenet()
        """)
        v = _scan_inflate_for_scorer_load(inf, root)
        assert len(v) >= 1, v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "submissions" / "robust_current" / "inflate.py", """
            from tac.scorer import load_segnet
        """)
        with pytest.raises(MetaBugViolation):
            check_no_scorer_load_at_inflate(repo_root=root, strict=True, verbose=False)

    def test_check_no_submissions_dir_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        # No submissions dir at all → vacuously OK.
        v = check_no_scorer_load_at_inflate(repo_root=root, strict=True, verbose=False)
        assert v == []


# ─── Check 7: training scripts must auth-eval ────────────────────────────────


class TestTrainingScriptsHaveAuthEval:
    def test_save_without_auth_eval_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_thing.py"
        _write(script, """
            import torch
            def go():
                model = build()
                torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "auth" in v[0].lower()

    def test_save_with_subprocess_auth_eval_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_good.py"
        _write(script, """
            import subprocess, torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                subprocess.run(["python", "auth_eval_renderer.py", "--ckpt", "renderer_best.pt"])
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_with_tac_auth_eval_import_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_imp.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval
            def go():
                torch.save(model, "renderer.pt")
                run_auth_eval("renderer.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_no_save_skips_check(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_smoke.py"
        _write(script, """
            def go():
                print("just a test, no save")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_explicit_optout_flag_satisfies(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_optout.py"
        _write(script, """
            # Operator may pass --no-auth-eval-on-best to skip; the rule is
            # satisfied by the existence of the flag (operator made an
            # explicit choice).
            import torch
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-auth-eval-on-best", action="store_true")
            torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_default_on_optout_flag_does_not_satisfy(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_default_on_optout.py"
        _write(script, """
            import torch
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-auth-eval-on-best", action="store_true", default=True)
            torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "auth_eval" in v[0]

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "train_bad.py", """
            import torch
            torch.save(m.state_dict(), "renderer_best.pt")
        """)
        with pytest.raises(MetaBugViolation):
            check_training_scripts_have_auth_eval(repo_root=root, strict=True, verbose=False)


# ─── Check 8: --no-eval-roundtrip CLI flag ──────────────────────────────────


class TestNoDisableEvalRoundtripFlag:
    def test_disable_flag_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_argparse.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip", action="store_true")
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert len(v) == 1, v
        assert "--no-eval-roundtrip" in v[0]

    def test_clean_argparse_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_argparse.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--eval-roundtrip", action="store_true", default=True)
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert v == [], v

    def test_test_files_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "tests" / "test_thing.py"
        _write(script, """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip")
        """)
        v = _scan_python_for_disable_eval_roundtrip_flag(script, root)
        assert v == [], v

    def test_check_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--no-eval-roundtrip")
        """)
        with pytest.raises(MetaBugViolation):
            check_no_disable_eval_roundtrip_flag(repo_root=root, strict=True, verbose=False)


# ─── codex R5-3 #5: heredoc masking for shell scanners ──────────────────────


class TestHeredocMasking:
    """Pin that bash heredoc bodies are NOT scanned as executable shell.

    Bug class (pre-fix): a heredoc embedding Python, docs, or generated
    shell text with lines like `set -uo pipefail`, `zip out.zip ...`,
    or `| grep -q` would be flagged by the shell scanners as if the
    heredoc body were code. Fix: `_mask_shell_heredocs(text)` zeroes
    out heredoc bodies (preserving line numbers) before regex scan.
    """

    # ----- _scan_shell_for_missing_set_e under heredoc -----

    def test_set_uo_pipefail_inside_heredoc_is_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Heredoc body containing the bug pattern must NOT trigger.
        Real-world case: scripts that emit shell snippets via heredoc."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat > /tmp/inner.sh <<'INNER'
            #!/usr/bin/env bash
            set -uo pipefail
            X=$(echo hi)
            INNER
            echo done
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], (
            f"`set -uo pipefail` inside a heredoc is documentation, not "
            f"executable shell — must not be flagged; got {v}"
        )

    def test_set_uo_pipefail_outside_heredoc_still_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Control: same pattern OUTSIDE a heredoc must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_bad.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -uo pipefail
            X=
        """)
        v = _scan_shell_for_missing_set_e(sh, root)
        assert len(v) == 1, (
            f"Control: same pattern OUTSIDE heredoc must still flag; got {v}"
        )

    # ----- _scan_shell_for_zip_binary under heredoc -----

    def test_zip_inside_python_heredoc_is_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        """Common idiom: `python3 <<'PY' ... import zipfile ... PY`. The
        word 'zip' appearing in Python code must not be treated as a
        shell `zip` invocation."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "python_emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python3 <<'PY'
            # generate archive in pure python — no shell zip needed
            zip out.zip in
            print("zip out.zip in")
            PY
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert v == [], (
            f"`zip out.zip` inside python heredoc is Python code, not shell "
            f"— must not be flagged; got {v}"
        )

    def test_zip_outside_heredoc_still_flagged(self, tmp_path: Path) -> None:
        """Control: shell-level `zip` must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_zip.sh"
        _write(sh, """
            #!/bin/bash
            zip archive.zip renderer.bin
        """)
        v = _scan_shell_for_zip_binary(sh, root)
        assert len(v) == 1, v

    # ----- _scan_shell_for_pipefail_grep_q under heredoc -----

    def test_grep_q_inside_heredoc_is_not_flagged(self, tmp_path: Path) -> None:
        """A heredoc body containing `cmd | grep -q` (e.g. emitted snippet
        for documentation or downstream script) must not be flagged."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "doc_emit.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat > README <<'EOF'
            Example bad pattern:
                some_cmd | grep -q "needle"
            EOF
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert v == [], (
            f"`cmd | grep -q` inside heredoc is documentation, not shell "
            f"— must not be flagged; got {v}"
        )

    def test_grep_q_outside_heredoc_still_flagged(self, tmp_path: Path) -> None:
        """Control: real `cmd | grep -q` outside heredoc must STILL be caught."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "real_grep_q.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            unzip -l archive.zip | grep -q renderer.bin
        """)
        v = _scan_shell_for_pipefail_grep_q(sh, root)
        assert len(v) == 1, v

    # ----- heredoc edge cases -----

    def test_dash_heredoc_terminator_with_leading_tabs(
        self, tmp_path: Path,
    ) -> None:
        """`<<-TOKEN` strips leading tabs from the terminator. Mask must
        recognize tab-indented terminators so the body is correctly bounded."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "dash_hd.sh"
        # Note: tabs are intentional inside the heredoc body for <<- form.
        sh.parent.mkdir(parents=True, exist_ok=True)
        sh.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cat <<-EOF\n"
            "\tset -uo pipefail\n"
            "\tEOF\n"
            "echo done\n"
        )
        v = _scan_shell_for_missing_set_e(sh, root)
        assert v == [], (
            f"<<-EOF with tab-stripped terminator: body must be masked; got {v}"
        )

    def test_unterminated_heredoc_masks_to_eof(self, tmp_path: Path) -> None:
        """A heredoc without a terminator (script error) — we mask to EOF
        so we conservatively avoid scanning the unterminated body."""
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "unterminated.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            cat <<NEVER_CLOSED
            zip out.zip in
            set -uo pipefail
        """)
        v_zip = _scan_shell_for_zip_binary(sh, root)
        v_set = _scan_shell_for_missing_set_e(sh, root)
        assert v_zip == [], v_zip
        assert v_set == [], v_set


# ─── codex R5-3 #7: MPS BoolOp chain detection ──────────────────────────────


class TestMpsBoolOpDetection:
    """Pin that BoolOp (and/or) device-selection chains are caught.

    Bug class: `cuda.is_available() and 'cuda' or mps.is_available() and 'mps'
    or 'cpu'` is the same MPS-fallback pattern as the IfExp ternary, but
    has no IfExp anywhere — must AST-walk BoolOp explicitly.
    """

    def test_classic_and_or_chain_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "boolop_bad.py"
        _write(script, """
            import torch
            def pick():
                return (torch.cuda.is_available() and 'cuda'
                        or torch.backends.mps.is_available() and 'mps'
                        or 'cpu')
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert any("BoolOp" in s or "MPS-fallback" in s for s in v), v
        assert len(v) >= 1, v

    def test_nested_parenthesized_boolop_is_caught(
        self, tmp_path: Path,
    ) -> None:
        """Inline cuda check + nested parens: the outer `or` is the
        result-position BoolOp containing the cuda check AND 'mps'."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "boolop_nested.py"
        _write(script, """
            import torch
            def pick():
                return ((torch.cuda.is_available() and 'cuda')
                        or (torch.backends.mps.is_available() and 'mps')
                        or 'cpu')
        """)
        v = _scan_python_for_mps_fallback(script, root)
        # Nested case: outer `or` BoolOp contains the cuda call subtree
        # AND has 'mps' as a leaf via the middle inner BoolOp's value.
        assert len(v) >= 1, v

    def test_pure_mps_literal_without_cuda_check_passes(
        self, tmp_path: Path,
    ) -> None:
        """Someone explicitly choosing MPS for an MPS-only test (e.g.
        `device = 'mps'` literal) must NOT flag — the rule targets the
        FALLBACK pattern, not deliberate MPS use."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "explicit_mps.py"
        _write(script, """
            def make_mps_only_test():
                device = "mps"  # we are deliberately on MPS for this probe
                return device
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], (
            f"Explicit `device = 'mps'` literal (no cuda check) must NOT "
            f"flag — only the FALLBACK pattern is forbidden; got {v}"
        )

    def test_compare_with_mps_string_is_not_a_fallback(
        self, tmp_path: Path,
    ) -> None:
        """Real FP from training.py: `use_autocast = (cuda.is_available()
        and 'cuda') or str(self.device) == 'mps'`. The literal 'mps' is
        inside a Compare — never selected as the result value. Must NOT
        flag."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "compare_mps.py"
        _write(script, """
            import torch
            class T:
                device = None
                def use_autocast(self):
                    return ((str(self.device).startswith("cuda")
                             and torch.cuda.is_available())
                            or str(self.device) == "mps")
        """)
        v = _scan_python_for_mps_fallback(script, root)
        assert v == [], (
            f"`... or str(dev) == 'mps'` is a Compare, not a fallback "
            f"value — must NOT flag; got {v}"
        )


# ─── codex R5-3 #8: training-script auth-eval AST upgrade ───────────────────


class TestTrainingAuthEvalAstUpgrade:
    """Pin the AST-based auth-eval check (replaces the old token-grep).

    Bug class (pre-fix): the regex form was both too narrow (missed
    multiline saves, variable paths) and too broad (matched any auth_eval
    token in a comment / help string / dead import). The new AST walker:
      1. Saves a renderer (path token: renderer/checkpoint/fp4/model.pt).
      2. Calls auth_eval (subprocess.run, .main(), helper) — or has the
         --no-auth-eval-on-best opt-out flag.
      Failure to satisfy 2 → violation. Imports without calls = dead-code
      violation.
    """

    def test_save_non_renderer_does_not_flag(self, tmp_path: Path) -> None:
        """torch.save(stats, 'stats.pt') — not a renderer, no violation."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_stats.py"
        _write(script, """
            import torch
            def go():
                torch.save({"loss": 0.5}, "stats.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], (
            f"`stats.pt` is not a renderer (no renderer/checkpoint/fp4 "
            f"token) — must NOT flag; got {v}"
        )

    def test_save_lora_does_not_flag(self, tmp_path: Path) -> None:
        """train_lora_tto.py-style: `torch.save(state, 'lora_best.pt')`.
        LoRA is not a renderer — must not flag."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_lora.py"
        _write(script, """
            import torch
            def go():
                torch.save(lora_state, output_dir / "lora_best.pt")
                torch.save(lora_state, output_dir / "lora_final.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_postfilter_does_not_flag(self, tmp_path: Path) -> None:
        """train_postfilter_on_renderer.py-style: postfilter saves are
        not renderer saves."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_pf.py"
        _write(script, """
            import torch
            def go():
                torch.save(postfilter.state_dict(), output_dir / "postfilter_best.pt")
                torch.save(postfilter_int8, output_dir / "postfilter_int8.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_subprocess_auth_eval_passes(
        self, tmp_path: Path,
    ) -> None:
        """Renderer save FOLLOWED by subprocess auth eval — satisfied."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_good_subp.py"
        _write(script, """
            import subprocess, torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                subprocess.run([
                    "python", "auth_eval_renderer.py",
                    "--ckpt", "renderer_best.pt",
                ])
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_optout_flag_passes(self, tmp_path: Path) -> None:
        """Renderer save with --no-auth-eval-on-best argparse opt-out: the
        operator made an explicit choice — satisfied."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_optout2.py"
        _write(script, """
            import argparse, torch
            p = argparse.ArgumentParser()
            p.add_argument("--no-auth-eval-on-best", action="store_true")
            torch.save(model.state_dict(), "renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_with_helper_call_passes(self, tmp_path: Path) -> None:
        """Direct call to `run_auth_eval(...)` satisfies the rule."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_helper.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                run_auth_eval("renderer_best.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v

    def test_save_renderer_imports_but_never_calls_auth_eval_flags(
        self, tmp_path: Path,
    ) -> None:
        """Dead-import-class: importing auth_eval but never calling it is
        STILL a violation — the import alone does not run the eval."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_dead_import.py"
        _write(script, """
            import torch
            from tac.auth_eval import run_auth_eval  # dead import
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
                # forgot to call run_auth_eval(...) — still violates
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "dead import" in v[0].lower() or "imports" in v[0].lower()

    def test_save_renderer_no_auth_eval_at_all_flags(
        self, tmp_path: Path,
    ) -> None:
        """No reference at all: violation."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_silent.py"
        _function_body = """
            import torch
            def go():
                torch.save(model.state_dict(), "renderer_best.pt")
        """
        _write(script, _function_body)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v
        assert "auth_eval" in v[0].lower() or "auth eval" in v[0].lower()

    def test_save_via_pathlib_join_with_renderer_token_flags(
        self, tmp_path: Path,
    ) -> None:
        """`output_dir / "renderer_fp4.bin"` BinOp — AST walker descends into
        BinOp.right.constant.value. Must catch this real-world form."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_pathlib.py"
        _write(script, """
            import torch
            from pathlib import Path
            def go():
                output_dir = Path("/tmp")
                torch.save(model, output_dir / "renderer_fp4.bin")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, (
            f"`output_dir / 'renderer_fp4.bin'` BinOp must be detected as "
            f"a renderer save — the constant is a child of BinOp.right; "
            f"got {v}"
        )

    def test_fstring_renderer_path_flags(self, tmp_path: Path) -> None:
        """f-string renderer paths: `f'{out}/renderer_ep{epoch}.pt'`. The
        AST walker must descend into JoinedStr to find the renderer token."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_fstring.py"
        _write(script, """
            import torch
            def go(epoch, out):
                torch.save(model, f"{out}/renderer_ep{epoch}.pt")
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert len(v) == 1, v

    def test_save_renderer_with_main_call_passes(self, tmp_path: Path) -> None:
        """`auth_eval_renderer.main()` direct invocation satisfies."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "train_main_call.py"
        _write(script, """
            import torch
            import auth_eval_renderer
            def go():
                torch.save(model, "renderer_best.pt")
                auth_eval_renderer.main()
        """)
        v = _scan_training_script_for_auth_eval(script, root)
        assert v == [], v


# ─── Check 9: pack_sparse_delta(compliance_status='approved') outside promo ──


class TestNoPackSparseDeltaApprovedOutsidePromotionTool:
    """codex R5-3 #2: pack_sparse_delta accepts compliance_status='approved'
    only with the constant-time-HMAC _internal_promotion_token. The runtime
    check exists; this static scan catches the same bug class earlier.
    """

    def test_call_in_promotion_tool_passes(self, tmp_path: Path) -> None:
        # The promotion tool itself is the canonical caller; even if it
        # someday calls pack_sparse_delta(approved) directly, that's
        # explicitly permitted by path filter.
        root = _stub_repo(tmp_path)
        (root / "tools").mkdir(parents=True, exist_ok=True)
        promo = root / "tools" / "promote_lane_c_to_approved.py"
        _write(promo, """
            from tac.uniward_delta import pack_sparse_delta
            def main():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='approved',
                    _internal_promotion_token=token,
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=False, verbose=False,
        )
        # The promotion-tool path is exempt by check_*; sub-scanner sees
        # the violation but check_* filters it out.
        assert v == [], v

    def test_call_in_test_file_passes(self, tmp_path: Path) -> None:
        # Test fixtures legitimately use the internal token to construct
        # approved blobs for negative-path coverage.
        root = _stub_repo(tmp_path)
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        test = root / "src" / "tac" / "tests" / "test_lane_c_attestation.py"
        _write(test, """
            from tac.uniward_delta import pack_sparse_delta, COMPLIANCE_APPROVED
            def test_promote():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status=COMPLIANCE_APPROVED,
                    _internal_promotion_token=token,
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v

    def test_call_in_arbitrary_experiment_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_promote.py"
        _write(bad, """
            from tac.uniward_delta import pack_sparse_delta
            def evil():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='approved',
                )
        """)
        v = _scan_python_for_pack_sparse_delta_approved(bad, root)
        assert len(v) >= 1, v
        assert any("compliance_status=" in s for s in v)

    def test_call_with_const_alias_is_caught(self, tmp_path: Path) -> None:
        # Catch the COMPLIANCE_APPROVED constant form (which equals 'approved').
        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_const.py"
        _write(bad, """
            from tac.uniward_delta import pack_sparse_delta, COMPLIANCE_APPROVED
            def evil():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status=COMPLIANCE_APPROVED,
                )
        """)
        v = _scan_python_for_pack_sparse_delta_approved(bad, root)
        assert len(v) >= 1, v

    def test_call_with_alias_import_is_caught(self, tmp_path: Path) -> None:
        # `from tac.uniward_delta import pack_sparse_delta as pkt` aliasing.
        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_alias.py"
        _write(bad, """
            from tac.uniward_delta import pack_sparse_delta as pkt
            def evil():
                blob = pkt(delta, cost, l_inf_budget=4.0, compliance_status='approved')
        """)
        v = _scan_python_for_pack_sparse_delta_approved(bad, root)
        assert len(v) >= 1, v

    def test_call_with_no_kwarg_passes(self, tmp_path: Path) -> None:
        # Defaults to pending_ruling — nothing to flag.
        root = _stub_repo(tmp_path)
        ok = root / "experiments" / "ok_default.py"
        _write(ok, """
            from tac.uniward_delta import pack_sparse_delta
            def ok():
                blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
        """)
        v = _scan_python_for_pack_sparse_delta_approved(ok, root)
        assert v == [], v

    def test_call_with_pending_or_rejected_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        ok = root / "experiments" / "ok_pending.py"
        _write(ok, """
            from tac.uniward_delta import pack_sparse_delta
            def ok():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='pending_ruling',
                )
                blob2 = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='rejected',
                )
        """)
        v = _scan_python_for_pack_sparse_delta_approved(ok, root)
        assert v == [], v

    def test_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "evil.py"
        _write(bad, """
            from tac.uniward_delta import pack_sparse_delta
            blob = pack_sparse_delta(
                d, c, l_inf_budget=4.0, compliance_status='approved'
            )
        """)
        with pytest.raises(MetaBugViolation):
            check_no_pack_sparse_delta_approved_outside_promotion_tool(
                repo_root=root, strict=True, verbose=False,
            )


# ─── Check 10: inflate.sh handles .br centrally before PYTHON_INFLATE ────────


class TestInflateShHandlesBrCentrally:
    """codex R5-3 #11: every PYTHON_INFLATE branch must see the archive
    fully decompressed. Centralized Stage 0 brotli block before dispatch.
    """

    def test_centralized_block_before_dispatch_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sub = root / "submissions" / "ok_sub"
        sub.mkdir(parents=True, exist_ok=True)
        sh = sub / "inflate.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            ARCHIVE_DIR="$1"
            # Stage 0: brotli decompression
            if compgen -G "$ARCHIVE_DIR"/*.br > /dev/null 2>&1; then
              uv run --with brotli python -c 'import brotli'
            fi
            if [ "$PYTHON_INFLATE" = "renderer" ]; then
              echo do work
            fi
        """)
        v = _scan_inflate_sh_for_centralized_brotli(sh, root)
        assert v == [], v

    def test_missing_brotli_block_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sub = root / "submissions" / "bad_missing"
        sub.mkdir(parents=True, exist_ok=True)
        sh = sub / "inflate.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            ARCHIVE_DIR="$1"
            if [ "$PYTHON_INFLATE" = "renderer" ]; then
              echo do work
            fi
        """)
        v = _scan_inflate_sh_for_centralized_brotli(sh, root)
        assert len(v) >= 1, v
        assert any("brotli" in s.lower() or "Stage 0" in s for s in v)

    def test_brotli_block_after_dispatch_is_caught(self, tmp_path: Path) -> None:
        # If --with brotli + .br + Stage 0 marker all appear AFTER the
        # dispatch line, the pre-dispatch loop never sees them → reported
        # as missing. Same outcome, so this exercises the position check.
        root = _stub_repo(tmp_path)
        sub = root / "submissions" / "bad_after"
        sub.mkdir(parents=True, exist_ok=True)
        sh = sub / "inflate.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            ARCHIVE_DIR="$1"
            if [ "$PYTHON_INFLATE" = "renderer" ]; then
              # Stage 0: brotli (TOO LATE — inside the branch)
              if compgen -G "$ARCHIVE_DIR"/*.br > /dev/null 2>&1; then
                uv run --with brotli python -c 'import brotli'
              fi
            fi
        """)
        v = _scan_inflate_sh_for_centralized_brotli(sh, root)
        assert len(v) >= 1, v

    def test_passthrough_inflate_sh_passes(self, tmp_path: Path) -> None:
        # No PYTHON_INFLATE dispatch → trivial passthrough → no Stage 0
        # block needed. PASS.
        root = _stub_repo(tmp_path)
        sub = root / "submissions" / "passthrough"
        sub.mkdir(parents=True, exist_ok=True)
        sh = sub / "inflate.sh"
        _write(sh, """
            #!/usr/bin/env bash
            set -euo pipefail
            python3 inflate.py "$@"
        """)
        v = _scan_inflate_sh_for_centralized_brotli(sh, root)
        assert v == [], v

    def test_real_robust_current_inflate_sh_passes(self) -> None:
        """The actual submissions/robust_current/inflate.sh in this repo
        was fixed in commit a1128fd9. It MUST pass."""
        from tac.preflight import REPO_ROOT
        path = REPO_ROOT / "submissions" / "robust_current" / "inflate.sh"
        if not path.exists():
            pytest.skip("submissions/robust_current/inflate.sh not present in this checkout")
        v = _scan_inflate_sh_for_centralized_brotli(path, REPO_ROOT)
        assert v == [], (
            "submissions/robust_current/inflate.sh has the Stage 0 block "
            f"per a1128fd9 — should pass, but got: {v}"
        )

    def test_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sub = root / "submissions" / "bad"
        sub.mkdir(parents=True, exist_ok=True)
        sh = sub / "inflate.sh"
        _write(sh, """
            #!/usr/bin/env bash
            if [ "$PYTHON_INFLATE" = "renderer" ]; then
              echo work
            fi
        """)
        with pytest.raises(MetaBugViolation):
            check_inflate_sh_handles_br_centrally(
                repo_root=root, strict=True, verbose=False,
            )


# ─── Check 11: scripts/remote_*.sh must run NVDEC probe at Stage 0 ───────────


class TestRemoteScriptsHaveNvdecProbe:
    """feedback_vastai_nvdec_host_variation + commit eef64293. NVDEC host
    availability is host-dependent on Vast.ai 4090s; the probe catches
    bad-host cases in 5s vs failing at the eval stage after $0.20+ of work.
    """

    def test_probe_at_top_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_ok.sh"
        _write(sh, """
            #!/bin/bash
            set -euo pipefail
            bash "$WORKSPACE/scripts/probe_nvdec.sh" || exit 2
            python experiments/build_baseline_archive.py --foo
            python -m tac.experiments.train_renderer.py --bar
        """)
        v = _scan_remote_script_for_nvdec_probe(sh, root)
        assert v == [], v

    def test_no_probe_call_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_bad_missing.sh"
        _write(sh, """
            #!/bin/bash
            set -euo pipefail
            python experiments/optimize_poses.py --foo
        """)
        v = _scan_remote_script_for_nvdec_probe(sh, root)
        assert len(v) >= 1, v
        assert any("no NVDEC probe" in s for s in v)

    def test_probe_after_gpu_work_is_caught(self, tmp_path: Path) -> None:
        # Probe MUST precede the first GPU-work marker. Probe at line 5
        # AFTER `train_renderer.py` at line 3 is a "probe-too-late" violation.
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_bad_late.sh"
        _write(sh, """
            #!/bin/bash
            set -euo pipefail
            python src/tac/experiments/train_renderer.py --foo
            echo "now probing — too late"
            bash "$WORKSPACE/scripts/probe_nvdec.sh" || exit 2
        """)
        v = _scan_remote_script_for_nvdec_probe(sh, root)
        assert len(v) >= 1, v
        assert any("AFTER" in s for s in v)

    def test_no_nvdec_needed_opt_out_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_optout.sh"
        _write(sh, """
            #!/bin/bash
            # NO_NVDEC_NEEDED — this script does no DALI / NVDEC video work
            set -euo pipefail
            python experiments/build_baseline_archive.py --no-video --foo
        """)
        v = _scan_remote_script_for_nvdec_probe(sh, root)
        assert v == [], v

    def test_no_gpu_work_passes(self, tmp_path: Path) -> None:
        # Script does no GPU work and didn't opt out → no probe needed.
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_quiet.sh"
        _write(sh, """
            #!/bin/bash
            set -euo pipefail
            echo "just a deploy / sync script"
            rsync -a foo bar
        """)
        v = _scan_remote_script_for_nvdec_probe(sh, root)
        assert v == [], v

    def test_real_lane_a_pose_tto_passes(self) -> None:
        """The actual scripts/remote_lane_a_pose_tto.sh was wired in
        commit eef64293. It MUST pass."""
        from tac.preflight import REPO_ROOT
        path = REPO_ROOT / "scripts" / "remote_lane_a_pose_tto.sh"
        if not path.exists():
            pytest.skip("scripts/remote_lane_a_pose_tto.sh not present in this checkout")
        v = _scan_remote_script_for_nvdec_probe(path, REPO_ROOT)
        assert v == [], (
            "scripts/remote_lane_a_pose_tto.sh was wired with the probe "
            f"in commit eef64293 — should pass, but got: {v}"
        )

    def test_strict_raises(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        sh = root / "scripts" / "remote_bad.sh"
        _write(sh, """
            #!/bin/bash
            python experiments/optimize_poses.py --foo
        """)
        with pytest.raises(MetaBugViolation):
            check_remote_scripts_have_nvdec_probe(
                repo_root=root, strict=True, verbose=False,
            )


# ─── codex R5-3 #4: preflight_all wires every meta-bug check ────────────────


class TestPreflightAllInvokesMetaBugChecks:
    """Source-grep regression: pin that preflight_all() actually invokes
    every meta-bug check function. Without this test, a future refactor
    could silently drop a check from preflight_all() — operators would
    get zero protection while believing it was wired.
    """

    def test_preflight_all_invokes_all_meta_bug_checks(self) -> None:
        # Source-grep preflight_all to verify each check is referenced.
        src = inspect.getsource(preflight_mod.preflight_all)

        required_checks = [
            "check_no_mps_fallback_default",
            "check_shell_set_e_present",
            "check_no_shell_zip_binary",
            "check_no_pipefail_grep_q_trap",
            "check_no_pipefail_tee_pipestatus_loss",
            "check_no_eval_roundtrip_false",
            "check_no_scorer_load_at_inflate",
            "check_training_scripts_have_auth_eval",
            "check_no_disable_eval_roundtrip_flag",
            # Follow-on (codex R5-3 #2 + #11 + NVDEC probe gap):
            "check_no_pack_sparse_delta_approved_outside_promotion_tool",
            "check_inflate_sh_handles_br_centrally",
            "check_remote_scripts_have_nvdec_probe",
            # codex R5-r6: 5 new checks for round-6 findings (warn-only).
            "check_no_brittle_six_line_waiver_lookback",
            "check_kl_distill_uses_roundtripped_frames",
            "check_eval_roundtrip_gate_called_after_output_dir_resolution",
            "check_nvdec_probe_has_error_classification",
            "check_archive_builders_use_deterministic_zip",
            "check_public_release_hygiene",
            "check_lightning_exact_eval_runner_bootstraps_dali",
            "check_dispatch_cli_shell_hazards",
            "check_reverse_engineering_tree_curation",
            "check_feature_flags_have_live_objective_effect",
            "check_no_fastvit_attention_compounding_claim",
            "check_no_auth_eval_optout_help_text_consumer_unverified",
        ]
        missing = [c for c in required_checks if c not in src]
        assert missing == [], (
            f"preflight_all() does not invoke {missing}. "
            f"codex R5-3 #4: every meta-bug check must be wired into "
            f"preflight_all (warn-only is OK, but must be called)."
        )

    def test_preflight_all_calls_meta_checks_strict(self) -> None:
        """Post-cleanup (commits 7d2b5299 + a94a9325): all 11 meta-bug
        checks now have ZERO live-codebase violations and are wired
        strict=True. Reverting any check to warn-only here means the
        bug class can silently land in a future commit — the gate must
        stay strict.

        If a future codebase change introduces a real violation that
        can't be fixed (e.g., a new submission strategy that requires
        scorer-at-inflate by design), the operator should add an
        explicit opt-out marker (NO_NVDEC_NEEDED, # noqa: scorer-at-
        inflate, etc.) recognized by the relevant scanner — NOT flip
        the check back to warn-only.
        """
        src = inspect.getsource(preflight_mod.preflight_all)
        meta_checks = [
            "check_no_mps_fallback_default",
            "check_shell_set_e_present",
            "check_no_shell_zip_binary",
            "check_no_pipefail_grep_q_trap",
            "check_no_pipefail_tee_pipestatus_loss",
            "check_no_eval_roundtrip_false",
            "check_no_scorer_load_at_inflate",
            "check_training_scripts_have_auth_eval",
            "check_no_disable_eval_roundtrip_flag",
            "check_no_pack_sparse_delta_approved_outside_promotion_tool",
            "check_inflate_sh_handles_br_centrally",
            "check_remote_scripts_have_nvdec_probe",
            # codex R5-r6: 5 new checks promoted directly to strict=True
            # because all landed at 0 live violations post-fix.
            "check_no_brittle_six_line_waiver_lookback",
            "check_kl_distill_uses_roundtripped_frames",
            "check_eval_roundtrip_gate_called_after_output_dir_resolution",
            "check_nvdec_probe_has_error_classification",
            "check_archive_builders_use_deterministic_zip",
            "check_lightning_exact_eval_runner_bootstraps_dali",
            "check_dispatch_cli_shell_hazards",
            "check_reverse_engineering_tree_curation",
            "check_feature_flags_have_live_objective_effect",
        ]
        for chk in meta_checks:
            # Find the line invoking this check and confirm strict=True.
            for line in src.splitlines():
                if chk + "(" in line and "def " not in line:
                    assert "strict=True" in line, (
                        f"{chk} must be invoked with strict=True in "
                        f"preflight_all (the bug class is structurally "
                        f"closed by 7d2b5299 + a94a9325); found: "
                        f"{line.strip()}. Add an opt-out marker for any "
                        f"new legitimate violation rather than reverting "
                        f"to warn-only."
                    )
                    break


# ─── 2026-05-02: public release hygiene for Apogee supplement surfaces ─────


class TestPublicReleaseHygiene:
    def test_public_release_hygiene_catches_private_ops_surface(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "README.md", """
            Notebook source: /Users/adpena/Projects/pact/reports/private.ipynb
            API key: sk-thisIsNotARealTokenButItIsLongEnough12345
            Debug job: https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge/app
            Vast shell: ssh4.vast.ai:25850
        """)
        with pytest.raises(MetaBugViolation, match="PUBLIC RELEASE HYGIENE"):
            check_public_release_hygiene(
                repo_root=root,
                strict=True,
                verbose=False,
                scan_paths=["README.md"],
            )

    def test_public_release_hygiene_allows_hosting_placeholders(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "docs" / "release.md", """
            Supplement: ${LIGHTNING_SUPPLEMENT_URL}
            Site: ${CLOUDFLARE_PAGES_URL}
            Public landing page: https://apogee.example.org/supplement
        """)
        violations = check_public_release_hygiene(
            repo_root=root,
            strict=True,
            verbose=False,
            scan_paths=["docs"],
        )
        assert violations == []

    def test_public_release_hygiene_explicit_scan_paths_override_default_exemptions(
        self,
        tmp_path: Path,
    ) -> None:
        root = _stub_repo(tmp_path)
        publish = root / "experiments" / "results" / "public_release_view"
        _write(
            publish / "README.md",
            "private notebook: /Users/example/private.ipynb\n",
        )

        with pytest.raises(MetaBugViolation, match="PUBLIC RELEASE HYGIENE"):
            check_public_release_hygiene(
                repo_root=root,
                strict=True,
                verbose=False,
                scan_paths=[publish],
            )


# ─── codex R5-4 #1: NVDEC probe must not fail on missing PyAV ────────────────


class TestNvdecProbeNoPyavDep:
    """The NVDEC probe used to import PyAV (`av`) inside the same block it
    used to diagnose NVDEC failure. On a fresh container without PyAV the
    import failed and the probe exit-2'd, falsely telling operators the
    host had no NVDEC. Fix: embed a tiny pre-built MP4 as base64; decode
    via stdlib `base64`. Tests pin both invariants.
    """

    PROBE_PATH = (
        Path(__file__).resolve().parents[3] / "scripts" / "probe_nvdec.sh"
    )

    def test_probe_does_not_import_pyav(self) -> None:
        text = self.PROBE_PATH.read_text()
        # The base64 fixture appears in the heredoc; it must NOT pull `av`.
        assert "import av" not in text, (
            "probe_nvdec.sh imports `av` (PyAV) — that turns a missing "
            "Python dep into a fake 'host has no NVDEC' exit-2. Use the "
            "embedded base64 fixture instead (codex R5-4 #1)."
        )
        # The replacement path must use base64 + the embedded constant.
        assert "TINY_MP4_B64" in text, (
            "probe_nvdec.sh must contain the embedded TINY_MP4_B64 "
            "fixture (codex R5-4 #1)."
        )
        assert "import base64" in text, (
            "probe_nvdec.sh must use stdlib `base64` to decode the "
            "embedded fixture."
        )

    def test_embedded_fixture_decodes_to_valid_mp4(self) -> None:
        """The embedded fixture must round-trip to a valid MP4 (1.5-2 KiB)."""
        import base64 as _b64
        import re

        text = self.PROBE_PATH.read_text()
        m = re.search(r"TINY_MP4_B64\s*=\s*\((.*?)\)", text, re.DOTALL)
        assert m, "TINY_MP4_B64 tuple not found in probe_nvdec.sh"
        parts = re.findall(r"'([^']+)'", m.group(1))
        assert len(parts) >= 2, f"expected multi-line tuple, got {len(parts)}"
        data = _b64.b64decode("".join(parts))
        # MP4 fixture should be small (1-3 KiB) and start with the ftyp box.
        assert 500 < len(data) < 4096, (
            f"embedded MP4 size suspicious: {len(data)} bytes"
        )
        # MP4 ftyp box: bytes 4-7 are 'ftyp'.
        assert data[4:8] == b"ftyp", (
            f"embedded fixture is not an MP4 (ftyp box missing): "
            f"first 16 bytes = {data[:16]!r}"
        )


# ─── codex R5-4 #2: scorer scanner detects dynamic imports + waivers ─────────


class TestScorerScannerDynamicImports:
    """The static scanner used to only catch `from tac.scorer import ...`
    and direct loader-name calls. Live inflate scripts deliberately used
    `importlib.import_module("tac.scorer")` + `getattr(mod, "load_scorers")`
    to bypass the scanner — strict mode reported clean while real scorer-
    at-inflate code remained env-gated. The new scanner detects the
    dynamic forms AND respects an explicit
    `# SCORER_AT_INFLATE_WAIVED:<reason>` comment marker so legitimate
    pending-ruling paths can be acknowledged rather than hidden.
    """

    def test_importlib_import_module_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod = importlib.import_module("tac.scorer")
        """)
        u, w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("import_module" in s for s in u), u
        assert w == [], w

    def test_importlib_util_find_spec_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib.util
            spec = importlib.util.find_spec("tac.scorer_targets")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("find_spec" in s for s in u), u

    def test_dunder_import_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            mod = __import__("tac.scorer")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("__import__" in s for s in u), u

    def test_getattr_loader_string_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod = importlib.import_module("tac.scorer")
            fn = getattr(mod, "load_scorers")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        # Two violations expected: import_module + getattr.
        assert len(u) >= 2, u
        assert any("getattr" in s and "load_scorers" in s for s in u), u

    def test_getattr_extract_gt_pose_targets_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            fn = getattr(somemod, "extract_gt_pose_targets")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("extract_gt_pose_targets" in s for s in u), u

    def test_waiver_marker_same_line(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod = importlib.import_module("tac.scorer")  # SCORER_AT_INFLATE_WAIVED:env-gated-INFLATE_TTO=1
        """)
        u, w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert u == [], (
            f"same-line waiver marker should suppress strict failure; got: {u}"
        )
        assert any("import_module" in s for s in w), w

    def test_waiver_marker_above_call_does_NOT_suppress(self, tmp_path: Path) -> None:
        """codex R5-r6 #1 NEGATIVE: marker on a previous line must NOT
        waive (lookback is 0 / same-line only). Pinned previously by
        `test_waiver_marker_within_lookback` which expected the OLD
        6-line-lookback behaviour."""
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            # SCORER_AT_INFLATE_WAIVED:env-gated-INFLATE_TTO=1
            # extra context line 1
            # extra context line 2
            mod = importlib.import_module("tac.scorer")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("import_module" in s for s in u), (
            f"codex R5-r6 #1: marker on previous lines must NOT waive; "
            f"got unwaived={u}"
        )

    def test_legacy_noqa_marker_only_recognised_on_same_line(
        self, tmp_path: Path,
    ) -> None:
        """The legacy `# noqa: scorer-at-inflate` marker is still
        recognised, but ONLY on the same line as the offending call
        (codex R5-r6 #1)."""
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        # Same-line: waived.
        inf_same = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf_same, """
            import importlib
            mod = importlib.import_module("tac.scorer")  # noqa: scorer-at-inflate (env-gated, pending-ruling)
        """)
        u, w = _scan_inflate_for_scorer_load_with_waivers(inf_same, root)
        assert u == [], f"same-line legacy noqa must suppress; got: {u}"
        assert len(w) == 1, w

        # Above call (block-level): NOT waived under same-line policy.
        inf_above = root / "submissions" / "robust_current" / "inflate2.py"
        _write(inf_above, """
            import importlib
            # noqa: scorer-at-inflate (env-gated, pending-ruling)
            mod = importlib.import_module("tac.scorer")
        """)
        u_above, _w_above = _scan_inflate_for_scorer_load_with_waivers(
            inf_above, root,
        )
        assert any("import_module" in s for s in u_above), (
            f"codex R5-r6 #1: legacy noqa above-call must NOT waive; "
            f"got unwaived={u_above}"
        )

    def test_marker_inside_string_is_not_a_waiver(self, tmp_path: Path) -> None:
        """The marker must be inside a comment, not a string literal."""
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            x = "SCORER_AT_INFLATE_WAIVED:fake"
            import importlib
            mod = importlib.import_module("tac.scorer")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        # Note: our heuristic treats `# inside a string` literally — the
        # `_line_is_waived` helper looks for `#` in the line text and
        # only checks the post-# segment. The string here has no `#`,
        # so no false-positive waiver. This pins that property.
        assert any("import_module" in s for s in u), u

    def test_check_strict_passes_with_only_waived(self, tmp_path: Path) -> None:
        from tac.preflight import check_no_scorer_load_at_inflate

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod = importlib.import_module("tac.scorer")  # SCORER_AT_INFLATE_WAIVED:env-gated-INFLATE_TTO=1
        """)
        # Strict mode must NOT raise — only waived hits remain.
        v = check_no_scorer_load_at_inflate(repo_root=root, strict=True, verbose=False)
        assert v == [], (
            f"strict mode should pass with only waived violations; got: {v}"
        )


# ─── codex R5-4 #3: pack_sparse_delta promotion gate broader fixture ────────


class TestPromotionGateFixtureExemptions:
    """The previous exemption was `src/tac/tests/test_*.py` only. The
    strict scanner now also scans `experiments/`, `scripts/`, and
    `tools/` — a legitimate fixture or integration test under any of
    those that constructs an approved blob with the internal token would
    block strict preflight. The exemption now covers test files broadly
    AND respects an explicit `# PACK_APPROVED_FIXTURE_OK` marker.
    """

    def test_experiments_test_file_is_exempt(self, tmp_path: Path) -> None:
        from tac.preflight import (
            check_no_pack_sparse_delta_approved_outside_promotion_tool,
        )

        root = _stub_repo(tmp_path)
        # A legit integration-test fixture sitting in experiments/.
        test = root / "experiments" / "test_some_pack_approved_workflow.py"
        _write(test, """
            from tac.uniward_delta import pack_sparse_delta
            def test_fixture():
                blob = pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='approved',
                    _internal_promotion_token=token,
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=True, verbose=False,
        )
        assert v == [], (
            f"test_*.py under experiments/ should be exempt (codex R5-4 "
            f"#3); got {v}"
        )

    def test_conftest_is_exempt(self, tmp_path: Path) -> None:
        from tac.preflight import (
            check_no_pack_sparse_delta_approved_outside_promotion_tool,
        )

        root = _stub_repo(tmp_path)
        conf = root / "experiments" / "conftest.py"
        _write(conf, """
            from tac.uniward_delta import pack_sparse_delta
            def make_fixture():
                return pack_sparse_delta(
                    delta, cost,
                    l_inf_budget=4.0,
                    compliance_status='approved',
                    _internal_promotion_token=token,
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=True, verbose=False,
        )
        assert v == [], f"conftest.py should be exempt; got {v}"

    def test_tests_dir_segment_is_exempt(self, tmp_path: Path) -> None:
        from tac.preflight import (
            check_no_pack_sparse_delta_approved_outside_promotion_tool,
        )

        root = _stub_repo(tmp_path)
        # File NOT named test_*.py but inside a /tests/ segment.
        f = root / "experiments" / "tests" / "shared_helpers.py"
        _write(f, """
            from tac.uniward_delta import pack_sparse_delta
            def helper():
                return pack_sparse_delta(
                    d, c, l_inf_budget=4.0, compliance_status='approved',
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=True, verbose=False,
        )
        assert v == [], (
            f"any path containing /tests/ should be exempt; got {v}"
        )

    def test_explicit_marker_is_exempt(self, tmp_path: Path) -> None:
        from tac.preflight import (
            check_no_pack_sparse_delta_approved_outside_promotion_tool,
        )

        root = _stub_repo(tmp_path)
        # A scripts-side fixture that's NOT a test file but carries the
        # explicit waiver marker.
        s = root / "scripts" / "build_negative_path_fixture.py"
        _write(s, """
            # PACK_APPROVED_FIXTURE_OK — builds an approved blob for the
            # downstream attestation-rejection negative-path test.
            from tac.uniward_delta import pack_sparse_delta
            def build():
                return pack_sparse_delta(
                    d, c, l_inf_budget=4.0,
                    compliance_status='approved',
                    _internal_promotion_token=token,
                )
        """)
        v = check_no_pack_sparse_delta_approved_outside_promotion_tool(
            repo_root=root, strict=True, verbose=False,
        )
        assert v == [], (
            f"file with `# PACK_APPROVED_FIXTURE_OK` marker should be "
            f"exempt; got {v}"
        )

    def test_unmarked_non_test_file_still_blocks(self, tmp_path: Path) -> None:
        """The exemption broadening must NOT silently let real violations
        through. A non-test, non-fixture, non-marked file in scripts/
        with `compliance_status='approved'` is still a violation."""
        from tac.preflight import (
            MetaBugViolation,
            check_no_pack_sparse_delta_approved_outside_promotion_tool,
        )

        root = _stub_repo(tmp_path)
        bad = root / "scripts" / "evil_promote.py"
        _write(bad, """
            from tac.uniward_delta import pack_sparse_delta
            def evil():
                return pack_sparse_delta(
                    d, c, l_inf_budget=4.0, compliance_status='approved',
                )
        """)
        with pytest.raises(MetaBugViolation):
            check_no_pack_sparse_delta_approved_outside_promotion_tool(
                repo_root=root, strict=True, verbose=False,
            )


# ─── codex R5-4 #4: centralised eval_roundtrip gate ──────────────────────────


class TestEvalRoundtripGate:
    """The per-script `_enforce_eval_roundtrip(args)` helper used to be a
    duplicated copy in 16 different scripts. It only checked the env var
    when args.eval_roundtrip was already False — so a leftover
    TAC_ALLOW_NO_ROUNDTRIP=1 in a shell session silently relaxed later
    runs without any per-run acknowledgement. The fix centralises the
    helper into `tac.eval_roundtrip_gate` and warns whenever the env var
    is present, regardless of the args value.
    """

    def setup_method(self) -> None:
        # Always start each test with a clean env.
        import os
        self._saved = os.environ.pop("TAC_ALLOW_NO_ROUNDTRIP", None)

    def teardown_method(self) -> None:
        import os
        os.environ.pop("TAC_ALLOW_NO_ROUNDTRIP", None)
        if self._saved is not None:
            os.environ["TAC_ALLOW_NO_ROUNDTRIP"] = self._saved

    def _make_args(self, eval_roundtrip: bool):
        class _A:
            pass
        a = _A()
        a.eval_roundtrip = eval_roundtrip
        return a

    def test_true_no_env_is_clean(self, capsys) -> None:
        from tac.eval_roundtrip_gate import enforce_eval_roundtrip

        r = enforce_eval_roundtrip(
            self._make_args(True), write_provenance=False,
        )
        assert r.eval_roundtrip is True
        assert r.env_var_present is False
        assert r.proceeded_via_escape_hatch is False
        # No banner printed.
        captured = capsys.readouterr()
        assert "DANGER" not in captured.err
        assert "WARNING" not in captured.err

    def test_true_with_env_present_warns(self, capsys) -> None:
        os.environ["TAC_ALLOW_NO_ROUNDTRIP"] = "1"
        r = enforce_eval_roundtrip(
            self._make_args(True), write_provenance=False,
        )
        assert r.eval_roundtrip is True
        assert r.env_var_present is True
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "TAC_ALLOW_NO_ROUNDTRIP" in captured.err

    def test_false_with_env_present_warns_and_proceeds(self, capsys) -> None:
        os.environ["TAC_ALLOW_NO_ROUNDTRIP"] = "1"
        r = enforce_eval_roundtrip(
            self._make_args(False), write_provenance=False,
        )
        assert r.eval_roundtrip is False
        assert r.proceeded_via_escape_hatch is True
        captured = capsys.readouterr()
        assert "DANGER" in captured.err

    def test_false_no_env_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            enforce_eval_roundtrip(
                self._make_args(False), write_provenance=False,
            )

    def test_provenance_written_when_output_dir_given(self, tmp_path: Path) -> None:
        out = tmp_path / "run_dir"
        r = enforce_eval_roundtrip(
            self._make_args(True), output_dir=out, write_provenance=True,
        )
        sidecar = out / "eval_roundtrip_gate.json"
        assert sidecar.exists()
        d = json.loads(sidecar.read_text())
        assert d["eval_roundtrip"] is True
        assert d["env_var_name"] == "TAC_ALLOW_NO_ROUNDTRIP"
        assert d["env_var_present"] == r.env_var_present

    def test_keyword_form_works_without_args(self) -> None:
        """Programmatic callers can pass `eval_roundtrip=...` directly."""
        from tac.eval_roundtrip_gate import enforce_eval_roundtrip

        r = enforce_eval_roundtrip(
            eval_roundtrip=True, write_provenance=False,
        )
        assert r.eval_roundtrip is True

    def test_neither_args_nor_kwarg_raises_value_error(self) -> None:
        from tac.eval_roundtrip_gate import enforce_eval_roundtrip

        with pytest.raises(ValueError):
            enforce_eval_roundtrip(write_provenance=False)

    # ─── codex R5-r6 #3: deferred sidecar via callback ───────────────────────

    def test_callback_defers_sidecar_until_write_now(self, tmp_path: Path) -> None:
        """Sidecar must only be written when write_sidecar_now() is called
        (codex R5-r6 #3). The gate API was extended so live scripts can
        defer the sidecar write until AFTER they resolve their default
        timestamped output_dir.
        """
        from tac.eval_roundtrip_gate import enforce_eval_roundtrip

        out = tmp_path / "deferred_run_dir"
        captured = {"resolved": None}

        def _callback() -> Path:
            # Simulates the script computing its timestamped default path.
            captured["resolved"] = out
            return out

        r = enforce_eval_roundtrip(
            self._make_args(True),
            output_dir_callback=_callback,
            write_provenance=True,
        )
        # Sidecar NOT yet written — callback hasn't been invoked.
        assert not (out / "eval_roundtrip_gate.json").exists()
        assert r._sidecar_written is False

        # Operator calls .write_sidecar_now() AFTER resolving output_dir.
        path = r.write_sidecar_now()
        assert path is not None
        assert path.exists()
        assert r._sidecar_written is True

    def test_callback_with_explicit_override_takes_precedence(
        self, tmp_path: Path,
    ) -> None:
        from tac.eval_roundtrip_gate import enforce_eval_roundtrip

        out_a = tmp_path / "from_callback"
        out_b = tmp_path / "from_override"

        r = enforce_eval_roundtrip(
            self._make_args(True),
            output_dir_callback=lambda: out_a,
            write_provenance=True,
        )
        path = r.write_sidecar_now(output_dir=out_b)
        assert path is not None
        assert path.parent == out_b
        assert (out_b / "eval_roundtrip_gate.json").exists()
        assert not (out_a / "eval_roundtrip_gate.json").exists()


# ─── codex R5-r6 #1: same-line waiver enforcement ───────────────────────────


class TestWaiverSameLineOnly:
    """The waiver scanner used to honour markers on any of the previous 6
    lines. That meant a marker intended for one specific pending-ruling
    import could waive an UNRELATED scorer load inserted nearby. The fix
    tightens the lookback to SAME-LINE ONLY so every waiver is structurally
    attached to its specific call site.
    """

    def test_same_line_waiver_works(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod = importlib.import_module("tac.scorer")  # SCORER_AT_INFLATE_WAIVED:env-gated
        """)
        u, w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert u == [], (
            f"same-line marker must waive; got unwaived={u}"
        )
        assert len(w) == 1, w

    def test_marker_one_line_above_does_NOT_waive(self, tmp_path: Path) -> None:
        """Codex R5-r6 #1 NEGATIVE TEST: marker on the line ABOVE the
        offending call must NOT suppress strict failure (lookback is 0)."""
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            # SCORER_AT_INFLATE_WAIVED:env-gated
            mod = importlib.import_module("tac.scorer")
        """)
        u, _w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        assert any("import_module" in s for s in u), (
            f"marker on previous line must NOT waive (codex R5-r6 #1); "
            f"got unwaived={u}"
        )

    def test_one_waivered_one_unwaivered_within_6_lines(
        self, tmp_path: Path,
    ) -> None:
        """Critical regression test for Finding #1: the previous 6-line
        lookback let a single waiver suppress BOTH the intended call AND
        an unrelated nearby call. With same-line policy, the waivered
        call must be suppressed and the un-waivered call must still
        appear in unwaived violations.
        """
        from tac.preflight import _scan_inflate_for_scorer_load_with_waivers

        root = _stub_repo(tmp_path)
        inf = root / "submissions" / "robust_current" / "inflate.py"
        _write(inf, """
            import importlib
            mod1 = importlib.import_module("tac.scorer")  # SCORER_AT_INFLATE_WAIVED:env-gated
            mod2 = importlib.import_module("tac.scorer_targets")
        """)
        u, w = _scan_inflate_for_scorer_load_with_waivers(inf, root)
        # mod1 waived, mod2 NOT waived — exactly one unwaived violation
        # mentioning scorer_targets.
        assert any("scorer_targets" in s for s in u), (
            f"un-waivered call MUST still surface; got unwaived={u}"
        )
        assert not any(
            "tac.scorer'" in s.replace('"', "'") and "scorer_targets" not in s
            for s in u
        ), (
            f"the intended same-line waiver must suppress its specific "
            f"call; got unwaived={u}"
        )
        assert any("tac.scorer" in s and "scorer_targets" not in s for s in w), w

    def test_lookback_constant_is_zero(self) -> None:
        """The constant must be 0 (or 1) — the brittle 6-line lookback
        was the root cause of Finding #1.
        """
        from tac.preflight import _WAIVER_LOOKBACK_LINES

        assert _WAIVER_LOOKBACK_LINES <= 1, (
            f"_WAIVER_LOOKBACK_LINES={_WAIVER_LOOKBACK_LINES} re-introduces "
            f"the codex R5-r6 #1 brittleness; must be 0 or 1."
        )


# ─── codex R5-r6 #2: KL distill uses roundtripped frames ────────────────────


class TestKlDistillUsesRoundtrippedFrames:
    """Lane G burned $0.85 because optimize_poses.py passed raw `pairs`
    (renderer output) to kl_distill_segnet_only(...) while the SegNet
    scoring path had already eval-roundtripped the same frames. The KL
    gradients pulled the renderer in the wrong direction relative to the
    scored loss path — net result: pose TTO HURT the score (proxy 0.0007
    vs auth 0.246 PoseNet, 350x gap; final auth 2.40 vs 0.90 baseline).
    """

    def test_raw_pairs_first_arg_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_kl_distill_raw_pairs

        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_kl.py"
        _write(bad, """
            from tac.losses import kl_distill_segnet_only
            def step(pairs, gt, segnet):
                kl, _ = kl_distill_segnet_only(pairs, gt, segnet, temperature=2.0)
                return kl
        """)
        v = _scan_python_for_kl_distill_raw_pairs(bad, root)
        assert any("pairs" in s for s in v), v

    def test_roundtripped_first_arg_is_clean(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_kl_distill_raw_pairs

        root = _stub_repo(tmp_path)
        good = root / "experiments" / "good_kl.py"
        _write(good, """
            from tac.losses import kl_distill_segnet_only
            def step(rendered_pair_hwc_rt, gt, segnet):
                kl, _ = kl_distill_segnet_only(rendered_pair_hwc_rt, gt, segnet, temperature=2.0)
                return kl
        """)
        v = _scan_python_for_kl_distill_raw_pairs(good, root)
        assert v == [], f"roundtripped variable must pass; got: {v}"

    def test_complex_first_arg_is_clean(self, tmp_path: Path) -> None:
        """Inline expressions (.permute, .view, etc.) are presumed to be
        intentional reshape of roundtripped frames — pass."""
        from tac.preflight import _scan_python_for_kl_distill_raw_pairs

        root = _stub_repo(tmp_path)
        good = root / "experiments" / "good_kl_inline.py"
        _write(good, """
            from tac.losses import kl_distill_segnet_only
            def step(frames_chw, gt, segnet):
                kl, _ = kl_distill_segnet_only(
                    frames_chw.permute(0, 2, 3, 1).contiguous(), gt, segnet,
                    temperature=2.0,
                )
                return kl
        """)
        v = _scan_python_for_kl_distill_raw_pairs(good, root)
        assert v == [], f"inline-permute first arg must pass; got: {v}"

    def test_same_line_marker_opts_out(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_kl_distill_raw_pairs

        root = _stub_repo(tmp_path)
        ok = root / "experiments" / "kl_test_fixture.py"
        _write(ok, """
            from tac.losses import kl_distill_segnet_only
            def fixture(pairs, gt, segnet):
                kl, _ = kl_distill_segnet_only(pairs, gt, segnet, temperature=2.0)  # KL_RAW_PAIRS_OK:test fixture
                return kl
        """)
        v = _scan_python_for_kl_distill_raw_pairs(ok, root)
        assert v == [], (
            f"same-line KL_RAW_PAIRS_OK marker must opt out; got: {v}"
        )

    def test_check_function_runs_without_strict_failure(self, tmp_path: Path) -> None:
        """Sanity: top-level check function callable in warn-only mode."""
        from tac.preflight import check_kl_distill_uses_roundtripped_frames

        root = _stub_repo(tmp_path)
        # Empty repo — check passes trivially.
        v = check_kl_distill_uses_roundtripped_frames(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v


# ─── codex R5-r6 #3: gate ordering preflight ────────────────────────────────


class TestGateAfterOutputDirResolution:
    """The eval_roundtrip_gate sidecar JSON used to be silently dropped
    when args.output_dir was None at call time. Live scripts now resolve
    args.output_dir BEFORE calling _enforce_eval_roundtrip(args) so the
    sidecar lands in the timestamped run dir.
    """

    def test_gate_before_output_dir_assignment_is_caught(
        self, tmp_path: Path,
    ) -> None:
        from tac.preflight import _scan_python_for_gate_before_output_dir

        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "bad_order.py"
        _write(bad, """
            def main():
                args = parse_args()
                _enforce_eval_roundtrip(args)
                if args.output_dir is None:
                    args.output_dir = "experiments/results/x"
        """)
        v = _scan_python_for_gate_before_output_dir(bad, root)
        assert any("BEFORE" in s for s in v), v

    def test_gate_after_output_dir_assignment_is_clean(
        self, tmp_path: Path,
    ) -> None:
        from tac.preflight import _scan_python_for_gate_before_output_dir

        root = _stub_repo(tmp_path)
        good = root / "experiments" / "good_order.py"
        _write(good, """
            def main():
                args = parse_args()
                if args.output_dir is None:
                    args.output_dir = "experiments/results/x"
                _enforce_eval_roundtrip(args)
        """)
        v = _scan_python_for_gate_before_output_dir(good, root)
        assert v == [], f"gate AFTER assignment must pass; got: {v}"

    def test_no_assignment_means_clean(self, tmp_path: Path) -> None:
        """If the script never resolves a default output_dir (CLI default
        suffices), the order doesn't matter."""
        from tac.preflight import _scan_python_for_gate_before_output_dir

        root = _stub_repo(tmp_path)
        ok = root / "experiments" / "no_assign.py"
        _write(ok, """
            def main():
                args = parse_args()
                _enforce_eval_roundtrip(args)
                output_dir = Path(args.output_dir)
        """)
        v = _scan_python_for_gate_before_output_dir(ok, root)
        assert v == [], v


# ─── codex R5-r6 #4: NVDEC probe error classification ───────────────────────


class TestNvdecProbeErrorClassification:
    """The probe used to map EVERY exception to exit 2 ("kill the host").
    A corrupt fixture or DALI build error therefore triggered the same
    operator response (destroy + relaunch) as a genuine missing-NVDEC
    host. The fix dispatches on a PROBE_CLASSIFICATION token to distinct
    exit codes (NVDEC=2, DALI=3, FIXTURE=4, UNKNOWN=5).
    """

    def test_probe_has_classification_marker(self) -> None:
        from tac.preflight import check_nvdec_probe_has_error_classification

        # Run against the live repo — should pass because we just landed
        # the codex R5-r6 #4 fix.
        v = check_nvdec_probe_has_error_classification(
            strict=False, verbose=False,
        )
        assert v == [], (
            f"live probe_nvdec.sh must pass classification check; got: {v}"
        )

    def test_missing_classification_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import check_nvdec_probe_has_error_classification

        root = _stub_repo(tmp_path)
        probe = root / "scripts" / "probe_nvdec.sh"
        probe.write_text(
            "#!/bin/bash\nset -e\nexit 0\n",
        )
        v = check_nvdec_probe_has_error_classification(
            repo_root=root, strict=False, verbose=False,
        )
        assert any("PROBE_CLASSIFICATION" in s for s in v), v

    def test_only_one_distinct_exit_code_is_caught(
        self, tmp_path: Path,
    ) -> None:
        from tac.preflight import check_nvdec_probe_has_error_classification

        root = _stub_repo(tmp_path)
        probe = root / "scripts" / "probe_nvdec.sh"
        probe.write_text(
            "#!/bin/bash\nset -e\n# PROBE_CLASSIFICATION:OK\nexit 2\n",
        )
        v = check_nvdec_probe_has_error_classification(
            repo_root=root, strict=False, verbose=False,
        )
        assert any("distinct" in s for s in v), v

    def test_missing_probe_file_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import check_nvdec_probe_has_error_classification

        root = _stub_repo(tmp_path)
        v = check_nvdec_probe_has_error_classification(
            repo_root=root, strict=False, verbose=False,
        )
        assert any("missing" in s.lower() for s in v), v


# ─── codex R5-r6 #5: deterministic-zip preflight ────────────────────────────


class TestArchiveBuildersUseDeterministicZip:
    """build_brotli_from_lane_a.py used `ZipFile.write(path, arcname=...)`
    which embeds the source-file mtime + perm bits. Two consecutive runs
    with identical inputs produced different archive bytes → no SHA
    anchoring possible. The fix uses fixed-timestamp ZipInfo + writestr.
    """

    def test_raw_zipfile_write_is_caught(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_nondeterministic_zip

        root = _stub_repo(tmp_path)
        bad = root / "experiments" / "build_bad.py"
        _write(bad, """
            import zipfile
            from pathlib import Path
            def main():
                with zipfile.ZipFile("/tmp/x.zip", "w") as z:
                    z.write("/tmp/a.bin", arcname="a.bin")
                    z.write("/tmp/b.bin", arcname="b.bin")
        """)
        v = _scan_python_for_nondeterministic_zip(bad, root)
        assert v != []
        assert any(".write" in s for s in v), v

    def test_deterministic_helper_is_clean(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_nondeterministic_zip

        root = _stub_repo(tmp_path)
        good = root / "experiments" / "build_good.py"
        _write(good, """
            import zipfile
            def _deterministic_zip_write(z, arcname, src):
                info = zipfile.ZipInfo(arcname, date_time=(2026, 4, 27, 0, 0, 0))
                z.writestr(info, b"data")
            def main():
                with zipfile.ZipFile("/tmp/x.zip", "w") as z:
                    z.write("/tmp/a.bin", arcname="a.bin")
        """)
        v = _scan_python_for_nondeterministic_zip(good, root)
        assert v == [], (
            f"file using ZipInfo helper must pass even with .write() calls; "
            f"got: {v}"
        )

    def test_explicit_optout_marker_is_clean(self, tmp_path: Path) -> None:
        from tac.preflight import _scan_python_for_nondeterministic_zip

        root = _stub_repo(tmp_path)
        ok = root / "experiments" / "build_optout.py"
        _write(ok, """
            # DETERMINISTIC_ZIP_OK — operator opt-out for this builder
            import zipfile
            def main():
                with zipfile.ZipFile("/tmp/x.zip", "w") as z:
                    z.write("/tmp/a.bin", arcname="a.bin")
        """)
        v = _scan_python_for_nondeterministic_zip(ok, root)
        assert v == [], v

    def test_check_function_runs(self, tmp_path: Path) -> None:
        """Top-level check is invokable in warn-only mode."""
        from tac.preflight import check_archive_builders_use_deterministic_zip

        root = _stub_repo(tmp_path)
        v = check_archive_builders_use_deterministic_zip(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v

    def test_check_covers_repo_compress_archive_script(self, tmp_path: Path) -> None:
        from tac.preflight import check_archive_builders_use_deterministic_zip

        root = _stub_repo(tmp_path)
        _write(root / "scripts" / "compress_archive.py", """
            import zipfile
            def main():
                with zipfile.ZipFile("archive.zip", "w") as z:
                    z.write("renderer.bin", arcname="renderer.bin")
        """)
        v = check_archive_builders_use_deterministic_zip(
            repo_root=root, strict=False, verbose=False,
        )
        assert any("scripts/compress_archive.py" in s for s in v), v

    def test_check_catches_compress_sh_inline_zipfile_write(self, tmp_path: Path) -> None:
        from tac.preflight import check_archive_builders_use_deterministic_zip

        root = _stub_repo(tmp_path)
        _write(root / "submissions" / "robust_current" / "compress.sh", """
            python3 - <<'PY'
            import zipfile
            with zipfile.ZipFile("archive.zip", "w") as zf:
                zf.write("renderer.bin", "renderer.bin")
            PY
        """)
        v = check_archive_builders_use_deterministic_zip(
            repo_root=root, strict=False, verbose=False,
        )
        assert any("compress.sh" in s for s in v), v

    def test_check_catches_raw_zip_extractall(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        bad_source = """
            import zipfile
            def main():
                with zipfile.ZipFile("archive.zip", "r") as zf:
                    zf.""" + """extractall("out")
        """
        _write(root / "submissions" / "robust_current" / "runner.py", bad_source)
        v = check_no_raw_zip_extractall(repo_root=root, strict=False, verbose=False)
        assert any("runner.py" in s for s in v), v
        from tac.source_index import source_index_context

        with source_index_context(root):
            indexed = check_no_raw_zip_extractall(
                repo_root=root,
                strict=False,
                verbose=False,
            )
        assert indexed == v

    def test_check_allows_canonical_safe_zip_extractor(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        helper_source = """
            def safe_extract_zip(zf, out):
                zf.""" + """extractall(out)
        """
        _write(root / "src" / "tac" / "submission_archive.py", helper_source)
        v = check_no_raw_zip_extractall(repo_root=root, strict=False, verbose=False)
        assert v == [], v

    def test_deterministic_zip_helper_produces_byte_identical_archives(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end regression: invoking the helper twice with identical
        inputs must produce byte-identical archive bytes (codex R5-r6 #5).
        """
        import hashlib
        import zipfile

        # Reproduce the helper inline so this test does not depend on the
        # build_brotli_from_lane_a.py harness staying importable.
        DET_DT = (2026, 4, 27, 0, 0, 0)
        DET_ATTR = (0o644 & 0xFFFF) << 16
        DET_SYS = 3

        def write_det(z: zipfile.ZipFile, arcname: str, data: bytes) -> None:
            info = zipfile.ZipInfo(filename=arcname, date_time=DET_DT)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = DET_ATTR
            info.create_system = DET_SYS
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

        payload_a = b"renderer-bytes-payload-A" * 100
        payload_b = b"masks-payload-B" * 200
        entries = sorted(
            [("renderer.bin.br", payload_a), ("masks.mkv", payload_b)],
            key=lambda kv: kv[0],
        )

        def build(out: Path) -> str:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for arc, data in entries:
                    write_det(z, arc, data)
            h = hashlib.sha256()
            h.update(out.read_bytes())
            return h.hexdigest()

        out1 = tmp_path / "a.zip"
        out2 = tmp_path / "b.zip"
        sha1 = build(out1)
        # Sleep a tiny bit so any wall-clock-based mtime drift would surface.
        import time
        time.sleep(1)
        sha2 = build(out2)
        assert sha1 == sha2, (
            f"deterministic-zip helper produced different archives across "
            f"two runs: {sha1} vs {sha2}. Codex R5-r6 #5 regression."
        )

    def test_nondeterministic_zfile_write_DOES_drift(
        self, tmp_path: Path,
    ) -> None:
        """Sanity for the test above: confirm vanilla ZipFile.write WOULD
        drift across runs (so we know the deterministic helper is doing
        something meaningful, not coincidentally producing the same bytes).
        """
        import hashlib
        import time
        import zipfile

        src = tmp_path / "input.bin"
        src.write_bytes(b"x" * 1024)
        out1 = tmp_path / "a.zip"
        out2 = tmp_path / "b.zip"

        with zipfile.ZipFile(out1, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(src, arcname="input.bin")
        # Touch the source mtime so the second zip embeds a different date.
        time.sleep(2)
        new_mtime = src.stat().st_mtime + 60
        os_utime_supported = True
        try:
            import os
            os.utime(src, (new_mtime, new_mtime))
        except OSError:
            os_utime_supported = False
        if not os_utime_supported:
            pytest.skip("os.utime not supported on this platform")
        with zipfile.ZipFile(out2, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(src, arcname="input.bin")

        sha1 = hashlib.sha256(out1.read_bytes()).hexdigest()
        sha2 = hashlib.sha256(out2.read_bytes()).hexdigest()
        # The vanilla path SHOULD have drifted because mtime changed.
        # If the platform happens to truncate to 2-second precision, the
        # difference still surfaces because we waited 2s.
        assert sha1 != sha2, (
            f"sanity: vanilla ZipFile.write must produce different bytes "
            f"when source mtime changes (got identical {sha1}). If this "
            f"asserts cleanly, the deterministic-zip test above is "
            f"vacuous on this platform."
        )


# ════════════════════════════════════════════════════════════════════════════
# ADDITIVE META-BUG TEST SECTION (12 new checks, post-R5-r6)
# ════════════════════════════════════════════════════════════════════════════
#
# Tests for the 12 new meta-bug checks added in the additive section of
# preflight.py. Each test class covers ONE check with offending +
# clean snippets. NEW classes only — does NOT touch any existing class
# above (those are owned by the codex-fix subagent).


# ─── Check A: Vast.ai create instance must include --label ──────────────────


class TestVastaiCreateHasLabel:
    def test_unlabeled_create_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "bad_launch.py"
        _write(script, """
            def launch():
                args = [
                    "create", "instance", "12345",
                    "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
                    "--disk", "40",
                ]
                run_vastai(args)
        """)
        v = _scan_python_for_vastai_create_no_label(script, root)
        assert len(v) == 1, v
        assert "--label" in v[0]

    def test_labeled_create_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "good_launch.py"
        _write(script, """
            def launch():
                args = [
                    "create", "instance", "12345",
                    "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
                    "--disk", "40",
                    "--label", "lane-X-experiment",
                ]
                run_vastai(args)
        """)
        v = _scan_python_for_vastai_create_no_label(script, root)
        assert v == [], v

    def test_unrelated_create_instance_not_flagged(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "innocent.py"
        _write(script, """
            words = ["create", "instances", "of", "a", "thing"]
        """)
        v = _scan_python_for_vastai_create_no_label(script, root)
        assert v == [], v


# ─── Check B: Vast.ai create must register tracker ─────────────────────────


class TestVastaiCreateWritesTracker:
    def test_no_tracker_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "bad.py"
        _write(script, """
            def launch():
                args = ["create", "instance", "12345", "--label", "x"]
                run_vastai(args)
                # ... 30 lines without tracker write ...
                pass
        """)
        v = _scan_python_for_vastai_create_no_tracker(script, root)
        assert len(v) == 1, v

    def test_tracker_write_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "scripts" / "good.py"
        _write(script, """
            def launch():
                args = ["create", "instance", "12345", "--label", "x"]
                instance_id = run_vastai(args)
                with open(".omx/state/vastai_active_instances.json", "r+") as f:
                    pass
        """)
        v = _scan_python_for_vastai_create_no_tracker(script, root)
        assert v == [], v


# ─── Check C: subagent prompts no --device cpu fallback ────────────────────


class TestSubagentPromptsNoCpuFallback:
    def test_unguarded_cpu_fallback_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        prompt = root / ".agents" / "task.md"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text(
            "Build the archive. If CUDA fails, run --device cpu instead.\n"
        )
        v = _scan_for_cpu_fallback_in_subagent_prompts(prompt, root)
        assert len(v) == 1, v

    def test_caveated_cpu_fallback_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        prompt = root / ".agents" / "task.md"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text(
            "Build the archive. --device cpu is OK here, "
            "deterministic-bytes acceptable for this analysis-only run.\n"
        )
        v = _scan_for_cpu_fallback_in_subagent_prompts(prompt, root)
        assert v == [], v

    def test_no_cpu_mention_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        prompt = root / ".agents" / "task.md"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("Build the archive on CUDA.\n")
        v = _scan_for_cpu_fallback_in_subagent_prompts(prompt, root)
        assert v == [], v


# ─── Check D: scores in run_log/findings must be lane-tagged ───────────────


class TestScoresHaveLaneTag:
    def test_untagged_score_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        log = root / ".ralph" / "run_log.md"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Result: auth = 0.36 (great progress)\n")
        v = _scan_doc_for_untagged_scores(log, root)
        assert len(v) == 1, v
        assert "lane tag" in v[0]

    def test_contest_cuda_tag_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        log = root / ".ralph" / "run_log.md"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Result: auth = 0.36 [contest-CUDA]\n")
        v = _scan_doc_for_untagged_scores(log, root)
        assert v == [], v

    def test_mps_proxy_tag_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        log = root / ".ralph" / "run_log.md"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Smoke: auth = 2.26 [MPS-PROXY]\n")
        v = _scan_doc_for_untagged_scores(log, root)
        assert v == [], v

    def test_formula_line_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        log = root / ".ralph" / "run_log.md"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("score = 100*seg + sqrt(10*pose) + rate\n")
        v = _scan_doc_for_untagged_scores(log, root)
        assert v == [], v


# ─── Check E: SCORER_AT_INFLATE_WAIVED must name env-gate ─────────────────


class TestWaiversSpecifyEnvGate:
    def test_bare_waiver_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("module = importlib.import_module('tac.scorer')  # SCORER_AT_INFLATE_WAIVED\n")
        v = _scan_for_unspecific_waivers(f, root)
        assert len(v) == 1, v
        assert "no" in v[0].lower() and "reason" in v[0].lower()

    def test_no_envgate_reason_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("foo()  # SCORER_AT_INFLATE_WAIVED:approved-by-yousfi\n")
        v = _scan_for_unspecific_waivers(f, root)
        assert len(v) == 1, v
        assert "env-gate" in v[0].lower()

    def test_proper_envgate_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("foo()  # SCORER_AT_INFLATE_WAIVED:env-gated-INFLATE_TTO=1\n")
        v = _scan_for_unspecific_waivers(f, root)
        assert v == [], v


# ─── Check F: half-frame archive needs trained renderer ───────────────────


class TestHalfframeArchiveTrainedProfile:
    def test_halfframe_no_profile_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "bad.sh"
        f.write_text("python experiments/build_baseline_archive.py --half-frame --output foo.zip\n")
        v = _scan_for_halfframe_without_trained_profile(f, root)
        assert len(v) >= 1, v

    def test_halfframe_with_zoom_profile_name_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "good.sh"
        f.write_text(
            "python experiments/build_baseline_archive.py "
            "--profile dilated_h64_half_frame --half-frame --output foo.zip\n"
        )
        v = _scan_for_halfframe_without_trained_profile(f, root)
        # PROFILES may or may not import; either way, name has 'half_frame' so passes.
        assert v == [], v

    def test_halfframe_with_module_profile_constant_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "experiments" / "arch_like.py"
        f.write_text(
            "\n".join([
                'PROFILE = "q_faithful_dilated_88k"',
                'cmd = f"""',
                'python experiments/build_baseline_archive.py \\',
                '  --device cuda --crf 50 --half-frame \\',
                '  --output masks.zip',
                'python src/tac/experiments/train_renderer.py \\',
                '  --profile {PROFILE} \\',
                '  --device cuda',
                '"""',
            ])
        )
        v = _scan_for_halfframe_without_trained_profile(f, root)
        assert v == [], v

    def test_no_halfframe_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "ok.sh"
        f.write_text("python experiments/build_baseline_archive.py --output foo.zip\n")
        v = _scan_for_halfframe_without_trained_profile(f, root)
        assert v == [], v


# ─── Check G: profile keys must have resolvers (bidirectional) ────────────


class TestProfileKeysHaveResolvers:
    def test_returns_list_without_error(self) -> None:
        # Smoke test on the live codebase — we just want it to RUN.
        v = check_profile_keys_have_resolvers(strict=False, verbose=False)
        assert isinstance(v, list)

    def test_strict_with_clean_state_passes(self, tmp_path: Path) -> None:
        # If PROFILES doesn't import, the check returns [] and never raises.
        # That's fine: the test confirms strict-mode raises only on real
        # violations.
        v = check_profile_keys_have_resolvers(
            repo_root=tmp_path, strict=True, verbose=False,
        )
        # tmp_path has no profiles.py → keys = None → returns []
        assert v == []


# ─── Check H: inflate scorer-load must print runtime banner ─────────────


class TestInflateScorerLoadBanner:
    def test_scorer_load_no_banner_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate_bad.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("from tac.scorer import load_scorers\n")
        v = check_inflate_scorer_load_has_runtime_banner(
            repo_root=root, strict=False, verbose=False,
        )
        assert len(v) == 1, v

    def test_scorer_load_with_banner_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate_good.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            "from tac.scorer import load_scorers\n"
            "print('[strict-scorer-rule] env-gated path active', file=sys.stderr)\n"
        )
        v = check_inflate_scorer_load_has_runtime_banner(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v

    def test_no_scorer_load_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "submissions" / "x" / "inflate.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("import sys\nprint('hello')\n")
        v = check_inflate_scorer_load_has_runtime_banner(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v


# ─── Check I: test files must have resolvable imports ────────────────────


class TestTestFilesImportsResolve:
    def test_import_from_nonexistent_module_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        f = root / "src" / "tac" / "tests" / "test_broken.py"
        f.write_text("from tac.does_not_exist import frobnicate\n")
        v = _scan_test_file_for_dead_imports(f, root)
        assert len(v) == 1, v
        assert "does not resolve" in v[0]

    def test_import_statement_from_nonexistent_module_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        f = root / "src" / "tac" / "tests" / "test_broken_import.py"
        f.write_text("import tac.does_not_exist as missing\n")
        v = _scan_test_file_for_dead_imports(f, root)
        assert len(v) == 1, v
        assert "imports module" in v[0]
        assert "does_not_exist" in v[0]

    def test_import_undefined_symbol_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        # Create a real module with limited names.
        mod = root / "src" / "tac" / "real_module.py"
        mod.parent.mkdir(parents=True, exist_ok=True)
        mod.write_text("def foo():\n    pass\n")
        # And a test that imports a symbol not in real_module.
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        f = root / "src" / "tac" / "tests" / "test_broken2.py"
        f.write_text("from tac.real_module import bar\n")
        v = _scan_test_file_for_dead_imports(f, root)
        assert len(v) == 1, v
        assert "does not define" in v[0]

    def test_valid_import_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "real_module.py"
        mod.parent.mkdir(parents=True, exist_ok=True)
        mod.write_text("def foo():\n    pass\n")
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        f = root / "src" / "tac" / "tests" / "test_ok.py"
        f.write_text("from tac.real_module import foo\n")
        v = _scan_test_file_for_dead_imports(f, root)
        assert v == [], v

    def test_top_level_tests_dir_is_scanned(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_broken_top_level.py").write_text(
            "from tac.does_not_exist import frobnicate\n"
        )
        violations = check_test_files_imports_resolve(
            repo_root=root,
            strict=False,
            verbose=False,
        )
        assert len(violations) == 1
        assert "test_broken_top_level.py" in violations[0]

    def test_third_party_import_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        f = root / "src" / "tac" / "tests" / "test_thirdparty.py"
        f.write_text("import torch\nfrom torch.nn import Module\nimport pytest\n")
        v = _scan_test_file_for_dead_imports(f, root)
        assert v == [], v

    def test_relative_repo_root_with_source_index_passes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "real_module.py"
        mod.parent.mkdir(parents=True, exist_ok=True)
        mod.write_text("def foo():\n    pass\n")
        tests_dir = root / "src" / "tac" / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_ok.py").write_text("from tac.real_module import foo\n")

        from tac.source_index import source_index_context

        monkeypatch.chdir(root)
        with source_index_context(Path(".")):
            assert check_test_files_imports_resolve(
                repo_root=Path("."),
                strict=True,
                verbose=False,
            ) == []

    def test_incremental_cache_invalidates_changed_test_import(
        self,
        tmp_path: Path,
    ) -> None:
        root = _stub_repo(tmp_path)
        mod = root / "src" / "tac" / "real_module.py"
        mod.parent.mkdir(parents=True, exist_ok=True)
        mod.write_text("def foo():\n    pass\n")
        tests_dir = root / "src" / "tac" / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_file = tests_dir / "test_cached_import.py"
        test_file.write_text("from tac.real_module import foo\n")

        assert check_test_files_imports_resolve(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []
        assert (
            root / ".omx" / "cache" / "test_imports_resolve_clean.json"
        ).exists()

        test_file.write_text("from tac.real_module import bar\n")
        violations = check_test_files_imports_resolve(
            repo_root=root,
            strict=False,
            verbose=False,
        )
        assert len(violations) == 1
        assert "real_module" in violations[0]
        assert "bar" in violations[0]


# ─── Check J: vastai prompts must mention cost cap ──────────────────────


class TestVastaiPromptsHaveCostCap:
    def test_unguarded_vastai_prompt_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        p = root / ".agents" / "task.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Launch a Vast.ai 4090 and run the experiment.\n")
        v = _scan_for_vastai_prompt_no_cost_cap(p, root)
        assert len(v) == 1, v

    def test_dollar_cost_cap_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        p = root / ".agents" / "task.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "Launch a Vast.ai 4090 and run the experiment. "
            "Budget: $5 hard cap. Destroy instance on completion.\n"
        )
        v = _scan_for_vastai_prompt_no_cost_cap(p, root)
        assert v == [], v

    def test_no_vastai_mention_skipped(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        p = root / ".agents" / "task.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Run locally on M5 Max.\n")
        v = _scan_for_vastai_prompt_no_cost_cap(p, root)
        assert v == [], v


# ─── Check K: --with-uniward-delta needs attestation gate ──────────────


class TestUniwardDeltaAttestationGate:
    def test_unattested_delta_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "lane_c_bad.sh"
        f.write_text(
            "python experiments/build_baseline_archive.py "
            "--with-uniward-delta delta.bin --output foo.zip\n"
        )
        v = _scan_for_uniward_delta_without_attestation(f, root)
        assert len(v) == 1, v

    def test_allow_pending_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "lane_c_good.sh"
        f.write_text(
            "python experiments/build_baseline_archive.py "
            "--with-uniward-delta delta.bin --allow-pending-compliance "
            "--output foo.zip\n"
        )
        v = _scan_for_uniward_delta_without_attestation(f, root)
        assert v == [], v

    def test_attestation_path_reference_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        f = root / "scripts" / "lane_c_attested.sh"
        f.write_text(
            "# attestation lives in .omx/state/lane_c_compliance_attestations/<sha>.json\n"
            "python experiments/build_baseline_archive.py "
            "--with-uniward-delta delta.bin --output foo.zip\n"
        )
        v = _scan_for_uniward_delta_without_attestation(f, root)
        assert v == [], v


# ─── Check L: remote scripts must write provenance.json ────────────────


class TestRemoteScriptsWriteProvenance:
    def test_no_provenance_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        scripts = root / "scripts"
        scripts.mkdir(exist_ok=True)
        f = scripts / "remote_lane_z_bad.sh"
        f.write_text("#!/bin/bash\nset -e\necho 'launching'\n")
        v = check_remote_scripts_write_provenance(
            repo_root=root, strict=False, verbose=False,
        )
        assert len(v) >= 1, v

    def test_provenance_present_passes(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        scripts = root / "scripts"
        scripts.mkdir(exist_ok=True)
        f = scripts / "remote_lane_z_good.sh"
        f.write_text(
            "#!/bin/bash\nset -e\n"
            'echo "{\\"git_sha\\":\\"abc\\"}" > provenance.json\n'
        )
        v = check_remote_scripts_write_provenance(
            repo_root=root, strict=False, verbose=False,
        )
        assert v == [], v

    def test_strict_raises_on_violation(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        scripts = root / "scripts"
        scripts.mkdir(exist_ok=True)
        f = scripts / "remote_lane_z_bad.sh"
        f.write_text("#!/bin/bash\nset -e\necho 'launching'\n")
        with pytest.raises(MetaBugViolation):
            check_remote_scripts_write_provenance(
                repo_root=root, strict=True, verbose=False,
            )


# ─── Smoke: all 12 new checks importable + callable ────────────────────


class TestNewMetaBugChecksSmoke:
    def test_all_12_checks_callable(self) -> None:
        """Confirms every new check function is importable + runnable on
        the live codebase without crashing."""
        for fn in [
            check_vastai_create_has_label,
            check_vastai_create_writes_tracker,
            check_subagent_prompts_no_cpu_fallback,
            check_scores_have_lane_tag,
            check_waivers_specify_env_gate,
            check_halfframe_archive_uses_trained_profile,
            check_profile_keys_have_resolvers,
            check_inflate_scorer_load_has_runtime_banner,
            check_test_files_imports_resolve,
            check_vastai_prompts_have_cost_cap,
            check_uniward_delta_has_attestation_gate,
            check_remote_scripts_write_provenance,
        ]:
            v = fn(strict=False, verbose=False)
            assert isinstance(v, list), f"{fn.__name__} must return list"


# ════════════════════════════════════════════════════════════════════════════
# ADDITIVE META-BUG TEST SECTION (Check M, post-2026-04-27 council forensics)
# ════════════════════════════════════════════════════════════════════════════
#
# Tests for `check_kl_div_reduction_correct` — the scanner that forbids
# `F.kl_div(..., reduction="batchmean")` on spatial (B, C, H, W) tensors.
# Bug class: under-divides the per-pixel mean by H x W (=196,608 for
# 384x512 SegNet). See findings.md "## 2026-04-27 Council forensics:
# Lane G — really dead, or bugged?" + losses.py Check M comment.


class TestKlDivReductionCorrect:
    """Pin: `F.kl_div(..., reduction="batchmean")` is forbidden in scanned
    dirs unless a same-line `# KL_BATCHMEAN_OK:<reason>` waiver exists."""

    def test_batchmean_kwarg_is_caught(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "bad.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q, T):
                return F.kl_div(log_p, q, reduction="batchmean") * (T * T)
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert len(v) == 1, v
        assert "batchmean" in v[0]
        assert "196,608" in v[0] or "196608" in v[0]

    def test_torch_nn_functional_kl_div_form_is_caught(self, tmp_path: Path) -> None:
        """Long-form import `torch.nn.functional.kl_div(..., reduction='batchmean')`."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_long.py"
        _write(script, """
            import torch
            def kl_loss(log_p, q):
                return torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert len(v) == 1, v

    def test_bare_kl_div_form_is_caught(self, tmp_path: Path) -> None:
        """Bare `kl_div(...)` after `from torch.nn.functional import kl_div`."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "bad_bare.py"
        _write(script, """
            from torch.nn.functional import kl_div
            def kl_loss(log_p, q):
                return kl_div(log_p, q, reduction="batchmean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert len(v) == 1, v

    def test_canonical_per_pixel_pattern_passes(self, tmp_path: Path) -> None:
        """The canonical fix from findings.md (mirrors `kl_distill_scorer_loss`):
        `reduction="none" → .sum(dim=1) → .mean()`. Must NOT be flagged.
        """
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "good.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q, T):
                kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)
                return kl_per_pixel.mean() * (T * T)
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_reduction_mean_passes(self, tmp_path: Path) -> None:
        """`reduction="mean"` is also fine (over-divides by C vs the canonical
        per-pixel-per-class pattern, but is not the catastrophic batchmean
        bug). Scanner is targeted at `batchmean` specifically."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "good_mean.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                return F.kl_div(log_p, q, reduction="mean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_waiver_marker_suppresses_violation(self, tmp_path: Path) -> None:
        """Same-line `# KL_BATCHMEAN_OK:<reason>` opt-out works (mirrors
        the `# KL_RAW_PAIRS_OK:<reason>` pattern from Check B)."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "waived.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                # Flat (B, num_classes) classifier tensor, not spatial.
                return F.kl_div(log_p, q, reduction="batchmean")  # KL_BATCHMEAN_OK:flat-classifier-tensor
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_waiver_marker_must_be_in_comment(self, tmp_path: Path) -> None:
        """`KL_BATCHMEAN_OK` appearing in a string literal (not a comment)
        does NOT count as a waiver — only same-line comment markers do."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "fake_waiver.py"
        # The token appears in a string literal, NOT in a comment after #.
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                msg = "KL_BATCHMEAN_OK"  # red herring in a different position
                return F.kl_div(log_p, q, reduction="batchmean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        # The actual `kl_div(..., batchmean)` call line has no `# KL_BATCHMEAN_OK`
        # comment, so it must still be flagged.
        assert len(v) == 1, v

    def test_unrelated_kl_div_call_not_flagged(self, tmp_path: Path) -> None:
        """`F.kl_div(...)` with default reduction (not specified) → not flagged."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "default_reduction.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                return F.kl_div(log_p, q)
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_unrelated_function_not_flagged(self, tmp_path: Path) -> None:
        """A `*.kl_div_something()` with batchmean must NOT be flagged
        (only exact `.kl_div` is matched)."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "other_func.py"
        _write(script, """
            import some_lib as S
            def kl_loss(x, y):
                # Not torch's kl_div — different function name.
                return S.kl_div_something(x, y, reduction="batchmean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_vendored_public_intake_tree_is_skipped(self, tmp_path: Path) -> None:
        """Public-PR intake mirrors are forensics, not our source to mutate."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "results" / "public_pr95_intake_20260504_codex" / "source" / "bad.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                return F.kl_div(log_p, q, reduction="batchmean")
        """)
        v = _scan_python_for_kl_div_batchmean(script, root)
        assert v == [], v

    def test_check_returns_zero_violations_on_live_codebase(self) -> None:
        """Live count gate: post-fix, the entire scanned codebase must
        produce 0 `batchmean`-on-spatial violations. If this asserts
        non-zero, a new offender has been introduced — fix it OR add
        the explicit `# KL_BATCHMEAN_OK:<reason>` waiver."""
        v = check_kl_div_reduction_correct(strict=False, verbose=False)
        assert v == [], (
            f"Live codebase has {len(v)} `batchmean`-on-spatial KL violation(s). "
            f"Fix them OR add the `# KL_BATCHMEAN_OK:<reason>` waiver. "
            f"Violations:\n" + "\n".join(f"  • {x}" for x in v)
        )

    def test_strict_mode_raises_metabugviolation(self, tmp_path: Path) -> None:
        """`strict=True` must raise `MetaBugViolation` on any violation,
        consistent with the other Check N strict-mode contracts."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "bad_strict.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q):
                return F.kl_div(log_p, q, reduction="batchmean")
        """)
        with pytest.raises(MetaBugViolation):
            check_kl_div_reduction_correct(
                repo_root=root, strict=True, verbose=False,
            )

    def test_strict_mode_passes_on_clean_repo(self, tmp_path: Path) -> None:
        """`strict=True` must NOT raise on a clean repo."""
        root = _stub_repo(tmp_path)
        script = root / "src" / "tac" / "good_strict.py"
        _write(script, """
            import torch.nn.functional as F
            def kl_loss(log_p, q, T):
                kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)
                return kl_per_pixel.mean() * (T * T)
        """)
        # No exception → pass.
        v = check_kl_div_reduction_correct(
            repo_root=root, strict=True, verbose=False,
        )
        assert v == [], v


# ════════════════════════════════════════════════════════════════════════════
# ADDITIVE META-BUG TEST SECTION (Check N, post-2026-04-27 Lane F forensic)
# ════════════════════════════════════════════════════════════════════════════
#
# Tests for `check_no_silent_auto_discovery_with_warn` — the 29th meta-bug
# scanner that catches the silent-default-masquerading-as-negative-result
# pattern. Bug class: missing CLI flag → auto-discover from N hardcoded
# paths → none exist → print [WARN] → proceed silently → operator sees the
# result land as a negative outcome and concludes "this lane is dead."
#
# Real-world incidents (2-in-2-days, 2026-04-27):
#   • Lane F v1 (qat_finetune.py) — auto-discovered gt_poses.pt, fell back
#     to zero poses, +58% PoseNet regression reported as "FP4 quant is dead."
#   • Lane G v1 (kl_distill_weight) — silent-batchmean over-weighting,
#     reported as "KL distill killed PoseNet" when it was a 5000x bug.
#
# See findings.md "Lane F regression — bugged or dead?" + memory
# `feedback_silent_default_masquerading_as_negative_result`.


class TestNoSilentAutoDiscoveryWithWarn:
    """Pin: forbid the `for x in [Path(...), Path(...)]: ... print('[WARN]')`
    + proceed-without-raise pattern. Either RAISE or add an
    `# AUTO_DISCOVERY_OK:<reason>` waiver."""

    def test_lane_f_qat_finetune_pattern_is_caught(self, tmp_path: Path) -> None:
        """The exact pre-fix Lane F pattern from qat_finetune.py."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "bad_qat.py"
        _write(script, """
            from pathlib import Path

            def load_poses(cfg):
                poses = None
                for poses_path in [
                    Path("experiments/results/gt_poses.pt"),
                    Path(cfg.upstream_dir) / "gt_poses.pt",
                ]:
                    if poses_path.exists():
                        poses = poses_path
                        break
                if poses is None:
                    print("[WARN] Renderer has pose_dim>0 but no poses_path provided — will use zero poses")
                return poses
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert len(v) == 1, v
        assert "load_poses" in v[0]
        assert "AUTO_DISCOVERY_OK" in v[0]

    def test_raise_after_loop_passes(self, tmp_path: Path) -> None:
        """Same auto-discovery, but raises SystemExit on no-match → CLEAN."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_raise.py"
        _write(script, """
            from pathlib import Path

            def load_poses(cfg):
                poses = None
                for poses_path in [
                    Path("experiments/results/gt_poses.pt"),
                    Path(cfg.upstream_dir) / "gt_poses.pt",
                ]:
                    if poses_path.exists():
                        poses = poses_path
                        break
                if poses is None:
                    raise SystemExit("FATAL: no poses found, pass --poses explicitly")
                return poses
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert v == [], v

    def test_sys_exit_after_warn_passes(self, tmp_path: Path) -> None:
        """A `print('[WARN]')` + `sys.exit(...)` is guarded → CLEAN."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "good_sysexit.py"
        _write(script, """
            import sys
            from pathlib import Path

            def load_poses(cfg):
                poses = None
                for poses_path in [
                    Path("a.pt"),
                    Path("b.pt"),
                ]:
                    if poses_path.exists():
                        poses = poses_path
                        break
                if poses is None:
                    print("[WARN] no poses")
                    sys.exit(2)
                return poses
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert v == [], v

    def test_function_waiver_marker_suppresses_violation(self, tmp_path: Path) -> None:
        """Same-function `# AUTO_DISCOVERY_OK:<reason>` opt-out works."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "waived.py"
        _write(script, """
            from pathlib import Path

            def load_optional_resource(cfg):
                # AUTO_DISCOVERY_OK:resource is genuinely optional, fallback is documented
                resource = None
                for path in [
                    Path("a.bin"),
                    Path("b.bin"),
                ]:
                    if path.exists():
                        resource = path
                        break
                if resource is None:
                    print("[WARN] optional resource not found, continuing")
                return resource
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert v == [], v

    def test_no_warn_call_is_clean(self, tmp_path: Path) -> None:
        """Auto-discovery + return None silently (no [WARN] print) → CLEAN.

        This is a different bug class (silent return) but is NOT caught by
        this scanner — only the WARN-then-proceed flavor. Documented gap."""
        root = _stub_repo(tmp_path)
        script = root / "experiments" / "no_warn.py"
        _write(script, """
            from pathlib import Path
            def load_poses(cfg):
                for path in [Path("a.pt"), Path("b.pt")]:
                    if path.exists():
                        return path
                return None
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert v == [], v

    def test_test_files_are_skipped(self, tmp_path: Path) -> None:
        """Test files (paths containing `tests` or filenames starting `test_`)
        are skipped — they intentionally exercise the bug pattern."""
        root = _stub_repo(tmp_path)
        # filename starts with test_
        (root / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
        script = root / "src" / "tac" / "tests" / "test_pattern.py"
        _write(script, """
            from pathlib import Path
            def load_poses(cfg):
                for poses_path in [Path("a.pt"), Path("b.pt")]:
                    if poses_path.exists():
                        return poses_path
                print("[WARN] no poses")
                return None
        """)
        v = _scan_python_for_silent_auto_discovery(script, root)
        assert v == [], "test files must be skipped"

    def test_check_strict_raises_on_violation(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            from pathlib import Path
            def load_poses(cfg):
                poses = None
                for poses_path in [Path("a.pt"), Path("b.pt")]:
                    if poses_path.exists():
                        poses = poses_path
                        break
                if poses is None:
                    print("[WARN] no poses found, using zero poses")
                return poses
        """)
        with pytest.raises(MetaBugViolation):
            check_no_silent_auto_discovery_with_warn(repo_root=root, strict=True, verbose=False)

    def test_check_warn_only_returns_list(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "experiments" / "bad.py", """
            from pathlib import Path
            def load_poses(cfg):
                poses = None
                for poses_path in [Path("a.pt"), Path("b.pt")]:
                    if poses_path.exists():
                        poses = poses_path
                        break
                if poses is None:
                    print("[WARN] no poses found")
                return poses
        """)
        v = check_no_silent_auto_discovery_with_warn(repo_root=root, strict=False, verbose=False)
        assert len(v) == 1, v

    def test_check_returns_zero_on_live_codebase(self) -> None:
        """Live count gate: post-fix the entire codebase produces 0 violations.
        If non-zero, a new offender has been introduced — fix it OR add the
        `# AUTO_DISCOVERY_OK:<reason>` waiver. (qat_finetune.py was the only
        known instance and was fixed in the same Lane F-V2 patch.)"""
        v = check_no_silent_auto_discovery_with_warn(strict=False, verbose=False)
        assert v == [], (
            f"Live codebase has {len(v)} silent-auto-discovery violation(s). "
            f"See findings.md 'Lane F regression — bugged or dead?' (2026-04-27).\n"
            + "\n".join(f"  • {x}" for x in v)
        )

    def test_qat_finetune_post_fix_passes(self) -> None:
        """Specific regression test: `experiments/qat_finetune.py` must NOT
        be flagged after the Bug 1 fix (the pre-fix file WOULD have been
        flagged). This is the one offender we fixed in the same patch."""
        target = REPO_ROOT / "experiments" / "qat_finetune.py"
        assert target.exists(), f"qat_finetune.py missing: {target}"
        v = _scan_python_for_silent_auto_discovery(target, REPO_ROOT)
        assert v == [], (
            "qat_finetune.py is being flagged — Bug 1 fix has regressed.\n"
            + "\n".join(f"  • {x}" for x in v)
        )


class TestEvidenceFalsificationScopeGuard:
    def test_positive_cpu_build_family_false_without_scope_passes(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "reports" / "cathedral_autopilot_evidence.jsonl", """
            {"technique":"phase4_orchestrator_smoke","evidence_grade":"[CPU-build]","family_falsified":false,"score_claim":false,"ready_for_exact_eval_dispatch":false}
        """)
        assert check_evidence_row_has_falsification_scope_when_negative(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_exact_negative_family_false_without_scope_fails(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "reports" / "cathedral_autopilot_evidence.jsonl", """
            {"technique":"lossy_config","evidence_grade":"[contest-CUDA A-negative]","family_falsified":false,"score_claim":false}
        """)
        with pytest.raises(PreflightError, match="FALSIFICATION SCOPE"):
            check_evidence_row_has_falsification_scope_when_negative(
                repo_root=root,
                strict=True,
                verbose=False,
            )

    def test_exact_negative_family_false_with_scope_passes(
        self, tmp_path: Path
    ) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "reports" / "cathedral_autopilot_evidence.jsonl", """
            {"technique":"lossy_config","evidence_grade":"[contest-CUDA A-negative]","family_falsified":false,"falsification_scope":"measured_config_only"}
        """)
        assert check_evidence_row_has_falsification_scope_when_negative(
            repo_root=root,
            strict=True,
            verbose=False,
        ) == []

    def test_family_falsified_true_still_fails(self, tmp_path: Path) -> None:
        root = _stub_repo(tmp_path)
        _write(root / "reports" / "cathedral_autopilot_evidence.jsonl", """
            {"technique":"bad_family_kill","evidence_grade":"[CPU-build]","family_falsified":true,"falsification_scope":"family"}
        """)
        with pytest.raises(PreflightError, match="FALSIFICATION SCOPE"):
            check_evidence_row_has_falsification_scope_when_negative(
                repo_root=root,
                strict=True,
                verbose=False,
            )
