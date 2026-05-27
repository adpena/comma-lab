# SPDX-License-Identifier: MIT
"""Tests for the Phase 9 canonical PR-submission full-lifecycle CLI (Layer 7).

Sister of ``test_submission_compliance.py`` (Phase 6) + ``test_submission_linter.py``
(Phase 5) + ``test_submission_bundle.py`` (Phase 4) at the end-to-end
orchestration sub-surface.

Strategy: orchestration tests monkeypatch the 6 layer helpers (imported into
the CLI module namespace) to return synthetic verdicts so exit-code routing +
sidecar emission + the Phase 8 gate integration are verified deterministically
WITHOUT paid dispatch or real trainers. Pure-function tests exercise the
parsers / attribution-self-lint / paired-env discipline directly.

Per CLAUDE.md "Executing actions with care": every test asserts the CLI NEVER
fires a gh command. Per user_pr_attribution memory: the attribution-self-lint
regression guards Claude/Anthropic/first-person-plural/emdash discipline.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import tools.operator_pr_submission_full_lifecycle as cli
from tac.submission_packet.builder import (
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    DependencyClosureManifest,
    SubmissionBundleResult,
)
from tac.submission_packet.compliance import (
    COMPLIANCE_SCHEMA_VERSION,
    ComplianceVerdict,
)
from tac.submission_packet.linter import (
    LINTER_SCHEMA_VERSION,
    LintFinding,
    LintVerdict,
)
from tac.submission_packet.paired_auth_eval import (
    PAIRED_AUTH_EVAL_SCHEMA_VERSION,
    PairedAuthEvalVerdict,
    PairedAuthEvalVerdictKind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _make_bundle(tmp_path: Path, lane_id: str = "lane_p9_test") -> SubmissionBundleResult:
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("inflate.sh", "#!/bin/bash\nset -euo pipefail\necho ok\n"),
        ("inflate.py", "# minimal inflate stub\n"),
        ("README.md", "# PR submission\n\n## Submission\n\n@SajayR (PR #56)\n"),
        ("report.txt", "placeholder\n"),
        ("PR_BODY.md", "# PR\n\n@SajayR (PR #56): HNeRV substrate [contest-CUDA]\n"),
    ):
        (submission_dir / name).write_text(body)
    archive = submission_dir / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", b"hello world")
    (submission_dir / "archive_manifest.json").write_text("{}\n")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    now = _utc()
    return SubmissionBundleResult(
        schema_version=SUBMISSION_BUNDLE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id="p9_substrate",
        archive_sha256=sha,
        archive_bytes=archive.stat().st_size,
        submission_dir=str(submission_dir),
        inflate_sh_path=str(submission_dir / "inflate.sh"),
        inflate_py_path=str(submission_dir / "inflate.py"),
        inflate_py_loc=1,
        inflate_py_loc_budget=DEFAULT_INFLATE_PY_LOC_BUDGET,
        inflate_py_loc_waiver_rationale=None,
        readme_md_path=str(submission_dir / "README.md"),
        report_txt_path=str(submission_dir / "report.txt"),
        archive_manifest_path=str(submission_dir / "archive_manifest.json"),
        dependency_closure_manifest=DependencyClosureManifest(
            declared_dependencies=("numpy",),
            dependency_budget=DEFAULT_INFLATE_DEPS_BUDGET,
            within_budget=True,
            numpy_portable=True,
        ),
        select_inflate_device_routing="inline_with_waiver",
        pythonpath_self_containment_status="clean",
        vendor_pythonpath_self_containment=False,
        runtime_dep_closure=("numpy",),
        measurement_utc=now,
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; submission-bundle-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_submission_bundle",
        canonical_equation_id="submission_bundle_canonical_helper_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
        canonical_provenance={"axis_tag": "[predicted]"},
        written_at_utc=now,
        written_pid=1,
        written_host="test",
    )


def _lint_verdict(*, clean: bool, error_count: int = 0) -> LintVerdict:
    findings: tuple[LintFinding, ...] = ()
    if not clean:
        findings = (
            LintFinding(
                surface="pr_body",
                severity="error",
                rule="catalog_forbidden_token",
                file_path="PR_BODY.md",
                line_number=1,
                matched_text="Claude",
                fix_suggestion="Remove forbidden token per user_pr_attribution.",
            ),
        )
        error_count = max(error_count, 1)
    return LintVerdict(
        schema_version=LINTER_SCHEMA_VERSION,
        overall_clean=clean,
        findings=findings,
        surfaces_scanned=("inflate_py", "pr_body"),
        error_count=error_count,
        warn_count=0,
        info_count=0,
        target_repo="commaai/comma_video_compression_challenge",
        measurement_utc=_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; submission-linter-canonical]",
        canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
        canonical_equation_id="submission_linter_canonical_helper_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
        canonical_provenance={"axis_tag": "[predicted]"},
    )


def _compliance_verdict(*, clean: bool) -> ComplianceVerdict:
    return ComplianceVerdict(
        schema_version=COMPLIANCE_SCHEMA_VERSION,
        lane_id="lane_p9_test",
        substrate_id="p9_substrate",
        archive_sha256="a" * 64,
        archive_bytes=1024,
        submission_dir="/tmp/x",
        overall_clean=clean,
        contest_final_strict=True,
        submission_score_axis="contest_cuda",
        total_checks=1,
        passed_count=1 if clean else 0,
        error_count=0 if clean else 1,
        warning_count=0,
        all_checks=(),
        error_checks=(),
        operator_gated_remaining=(),
        catalog_gate_protection_summary={},
        forbidden_macos_axis_detected=False,
        json_report_path="/tmp/x/compliance_report.json",
        measurement_utc=_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; submission-compliance-canonical]",
        canonical_helper_invocation="tac.submission_packet.enforce_contest_compliance",
        canonical_equation_id="submission_compliance_canonical_helper_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
        canonical_provenance={"axis_tag": "[predicted]"},
    )


def _paired_verdict(*, kind: str) -> PairedAuthEvalVerdict:
    is_pass = kind == PairedAuthEvalVerdictKind.PAIRED_PASS.value
    is_cuda_partial = kind == PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CUDA_ONLY.value
    cuda_present = is_pass or is_cuda_partial
    # Empty sha is only valid for BLOCKED_PRE_DISPATCH; any verdict where a
    # dispatch fired (PAIRED_PASS / PAIRED_PARTIAL_*) carries the sha.
    sha_known = kind != PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value
    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id="lane_p9_test",
        substrate_id="p9_substrate",
        archive_sha256_paired="a" * 64 if sha_known else "",
        archive_bytes=1024,
        submission_dir="/tmp/x",
        verdict=kind,
        verdict_rationale="synthetic test verdict for phase 9 lifecycle CLI routing",
        cuda_score=0.205 if cuda_present else None,
        cuda_axis_tag="[contest-CUDA]" if cuda_present else "[missing]",
        cuda_hardware_substrate="linux_x86_64_modal_t4" if cuda_present else "",
        cuda_call_id="fc-test" if cuda_present else "",
        cuda_seg_distortion=None,
        cuda_pose_distortion=None,
        cuda_rate_term=None,
        cuda_auth_eval_json_path="",
        cuda_elapsed_seconds=0.0,
        cuda_cost_usd=0.0,
        cpu_score=0.192 if is_pass else None,
        cpu_axis_tag="[contest-CPU]" if is_pass else "[missing]",
        cpu_hardware_substrate="linux_x86_64_modal_cpu" if is_pass else "",
        cpu_call_id="fc-test-cpu" if is_pass else "",
        cpu_seg_distortion=None,
        cpu_pose_distortion=None,
        cpu_rate_term=None,
        cpu_auth_eval_json_path="",
        cpu_elapsed_seconds=0.0,
        cpu_cost_usd=0.0,
        cuda_cpu_gap=None,
        cost_band="smoke",
        budget_usd=0.30,
        total_cost_usd=0.0,
        measurement_utc=_utc(),
        axis_tag="[contest-CUDA; contest-CPU]" if is_pass else "[predicted]",
        score_claim=False,
        promotable=is_pass,
        evidence_grade=(
            "[contest-CUDA; contest-CPU; paired-axis-empirical]"
            if is_pass
            else (
                "[contest-CUDA; paired-axis-cpu-missing]"
                if is_cuda_partial
                else "[predicted; paired-axis-not-yet-dispatched]"
            )
        ),
        canonical_helper_invocation="tac.submission_packet.plan_paired_auth_eval",
        canonical_equation_id="paired_auth_eval_canonical_helper_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform="modal",
        cuda_gpu="T4",
        cpu_target="linux_x86_64_modal",
        dry_run=True,
        forbidden_macos_axis_detected=False,
        canonical_provenance={"axis_tag": "[contest-CUDA; contest-CPU]"},
    )


def _patch_layers(
    monkeypatch,
    tmp_path: Path,
    *,
    bundle_raises: Exception | None = None,
    lint_clean: bool = True,
    compliance_clean: bool = True,
    paired_kind: str = PairedAuthEvalVerdictKind.PAIRED_PASS.value,
    gate_violations: list[str] | None = None,
):
    """Monkeypatch all 6 layer helpers + the Phase 8 gate in the CLI namespace."""
    bundle = _make_bundle(tmp_path)

    monkeypatch.setattr(cli, "build_compression_pipeline", lambda **kw: _FakePipeline())
    monkeypatch.setattr(
        cli, "build_archive_grammar_from_compression_pipeline_result",
        lambda **kw: _FakeGrammar(bundle.archive_sha256),
    )

    def _fake_bundle(**kw):
        if bundle_raises is not None:
            raise bundle_raises
        return bundle

    monkeypatch.setattr(cli, "build_submission_bundle", _fake_bundle)
    monkeypatch.setattr(
        cli, "lint_submission_bundle",
        lambda *a, **kw: _lint_verdict(clean=lint_clean),
    )
    monkeypatch.setattr(
        cli, "enforce_contest_compliance",
        lambda **kw: _compliance_verdict(clean=compliance_clean),
    )
    monkeypatch.setattr(
        cli, "plan_paired_auth_eval",
        lambda **kw: _paired_verdict(kind=paired_kind),
    )
    monkeypatch.setattr(
        cli, "_run_layer_6_gate",
        lambda submission_dir, repo_root: list(gate_violations or []),
    )
    return bundle


class _FakePipeline:
    substrate_id = "p9_substrate"


class _FakeGrammar:
    def __init__(self, sha: str) -> None:
        self.archive_sha256 = sha
        self.section_specs = (object(),)


def _args(tmp_path: Path, **overrides):
    """Build a parsed argparse.Namespace for run_full_lifecycle."""
    parser = cli._build_parser()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"x")
    base = [
        "--lane-id", "lane_p9_test",
        "--substrate-trainer", "experiments/train_substrate_grayscale_lut.py",
        "--recipe-path", ".omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml",
        "--archive-path", str(archive),
        "--output-dir", str(tmp_path / "submission_dir"),
        "--predecessors", "@SajayR:56:HNeRV_substrate",
    ]
    ns = parser.parse_args(base)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# Pure-function unit tests: predecessor parse
# ---------------------------------------------------------------------------
def test_parse_predecessors_valid():
    rows, errors = cli._parse_predecessors(["@SajayR:56:HNeRV_substrate"])
    assert errors == []
    assert rows == [{"handle": "SajayR", "pr_number": "56", "slug": "HNeRV_substrate"}]


def test_parse_predecessors_multiple():
    rows, errors = cli._parse_predecessors(
        ["@SajayR:56:HNeRV", "@AaronLeslie138:95:fec_curriculum"]
    )
    assert errors == []
    assert len(rows) == 2
    assert rows[1]["handle"] == "AaronLeslie138"


def test_parse_predecessors_malformed():
    rows, errors = cli._parse_predecessors(["NoAtSign:56:slug"])
    assert rows == []
    assert len(errors) == 1
    assert "malformed" in errors[0]


def test_parse_predecessors_none():
    rows, errors = cli._parse_predecessors(None)
    assert rows == []
    assert errors == []


def test_parse_predecessors_slug_with_colons():
    rows, errors = cli._parse_predecessors(["@h:101:slug:with:colons"])
    assert errors == []
    assert rows[0]["slug"] == "slug:with:colons"


# ---------------------------------------------------------------------------
# Attribution self-lint regression (CRITICAL per user_pr_attribution memory)
# ---------------------------------------------------------------------------
def test_attribution_markdown_clean():
    md = cli._build_attribution_chain_markdown(
        [{"handle": "SajayR", "pr_number": "56", "slug": "HNeRV"}],
        "commaai/comma_video_compression_challenge",
    )
    assert cli._scan_forbidden_pr_tokens(md) == []
    assert "@SajayR" in md
    assert "#56" in md


def test_scan_forbidden_token_claude():
    findings = cli._scan_forbidden_pr_tokens("This was built by Claude.")
    assert any("Claude" in f for f in findings)


def test_scan_forbidden_token_anthropic():
    findings = cli._scan_forbidden_pr_tokens("Anthropic helped here.")
    assert any("Anthropic" in f for f in findings)


def test_scan_forbidden_co_authored():
    findings = cli._scan_forbidden_pr_tokens("Co-Authored-By: someone")
    assert any("Co-Authored" in f for f in findings)


def test_scan_forbidden_first_person_plural():
    findings = cli._scan_forbidden_pr_tokens("we built our submission")
    assert any("first-person-plural" in f for f in findings)


def test_scan_forbidden_emdash():
    findings = cli._scan_forbidden_pr_tokens("this — that")
    assert any("emdash" in f for f in findings)


def test_scan_forbidden_clean_text():
    assert cli._scan_forbidden_pr_tokens("- @SajayR (PR #56): HNeRV substrate") == []


def test_attribution_no_predecessors():
    md = cli._build_attribution_chain_markdown([], "commaai/x")
    assert cli._scan_forbidden_pr_tokens(md) == []
    assert "Standalone" in md


# ---------------------------------------------------------------------------
# Paired-env discipline (Catalog #199)
# ---------------------------------------------------------------------------
def test_paired_env_inactive_when_unset(monkeypatch):
    monkeypatch.delenv(cli._ENV_CONFIRMED, raising=False)
    monkeypatch.delenv(cli._ENV_BUDGET, raising=False)
    active, note = cli._execute_paired_env_active()
    assert active is False
    assert note is None


def test_paired_env_active_when_paired(monkeypatch):
    monkeypatch.setenv(cli._ENV_CONFIRMED, "1")
    monkeypatch.setenv(cli._ENV_BUDGET, "5.00")
    active, note = cli._execute_paired_env_active()
    assert active is True
    assert "5.00" in note


def test_paired_env_bare_confirmed_rejected(monkeypatch):
    monkeypatch.setenv(cli._ENV_CONFIRMED, "1")
    monkeypatch.delenv(cli._ENV_BUDGET, raising=False)
    active, note = cli._execute_paired_env_active()
    assert active is False
    assert note is not None and "without paired" in note


def test_paired_env_non_numeric_budget_rejected(monkeypatch):
    monkeypatch.setenv(cli._ENV_CONFIRMED, "1")
    monkeypatch.setenv(cli._ENV_BUDGET, "notanumber")
    active, note = cli._execute_paired_env_active()
    assert active is False
    assert "not a numeric" in note


def test_paired_env_zero_budget_rejected(monkeypatch):
    monkeypatch.setenv(cli._ENV_CONFIRMED, "1")
    monkeypatch.setenv(cli._ENV_BUDGET, "0")
    active, note = cli._execute_paired_env_active()
    assert active is False
    assert "> 0" in note


# ---------------------------------------------------------------------------
# Orchestration: exit-code routing
# ---------------------------------------------------------------------------
def test_packet_clean_path_reaches_operator_gated(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path)
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_OPERATOR_GATED
    assert report["lifecycle_verdict"] == "OPERATOR-GATED"
    assert report["gh_commands_emitted"] is True
    assert report["gh_commands_fired"] is False


def test_lint_violations_exit_1(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path, lint_clean=False)
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_LINT_VIOLATIONS
    assert report["lifecycle_verdict"] == "LINT-VIOLATIONS"
    # Compliance + paired layers must NOT have run after lint failure.
    assert "layer_4_compliance" not in report["layers"]


def test_compliance_errors_exit_2(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path, compliance_clean=False)
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_COMPLIANCE_ERRORS
    assert report["lifecycle_verdict"] == "COMPLIANCE-ERRORS"
    assert "layer_5_paired_auth_eval" not in report["layers"]


def test_missing_paired_axis_exit_3(monkeypatch, tmp_path):
    _patch_layers(
        monkeypatch, tmp_path,
        paired_kind=PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CUDA_ONLY.value,
    )
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_MISSING_PAIRED_AXIS
    assert report["lifecycle_verdict"] == "MISSING-PAIRED-AXIS"


def test_gate_violation_exit_5(monkeypatch, tmp_path):
    _patch_layers(
        monkeypatch, tmp_path,
        gate_violations=["[Catalog #370] synthetic gate violation"],
    )
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_CLI_ERROR
    assert report["lifecycle_verdict"] == "CATALOG-370-GATE-VIOLATION"


def test_bundle_failure_exit_5(monkeypatch, tmp_path):
    from tac.submission_packet.builder import SubmissionBundleError
    _patch_layers(monkeypatch, tmp_path, bundle_raises=SubmissionBundleError("boom"))
    code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert code == cli.EXIT_CLI_ERROR
    assert report["layers"]["layer_2_builder"]["ok"] is False


def test_malformed_predecessor_exit_5(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path)
    code, report = cli.run_full_lifecycle(_args(tmp_path, predecessors=["bad_spec"]))
    assert code == cli.EXIT_CLI_ERROR
    assert report["layers"]["predecessor_parse"]["ok"] is False


def test_missing_archive_exit_5(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path)
    ns = _args(tmp_path, archive_path=Path("/nonexistent/archive.zip"))
    code, report = cli.run_full_lifecycle(ns)
    assert code == cli.EXIT_CLI_ERROR
    assert report["layers"]["preflight"]["ok"] is False


# ---------------------------------------------------------------------------
# Sidecar emission to canonical Phase 8 gate filenames
# ---------------------------------------------------------------------------
def test_sidecars_emitted_to_canonical_filenames(monkeypatch, tmp_path):
    bundle = _patch_layers(monkeypatch, tmp_path)
    cli.run_full_lifecycle(_args(tmp_path))
    sub = Path(bundle.submission_dir)
    assert (sub / cli._SIDECAR_BUNDLE).is_file()
    assert (sub / cli._SIDECAR_LINT).is_file()
    assert (sub / cli._SIDECAR_COMPLIANCE).is_file()
    assert (sub / cli._SIDECAR_PAIRED).is_file()


def test_bundle_sidecar_contains_clean_token(monkeypatch, tmp_path):
    bundle = _patch_layers(monkeypatch, tmp_path)
    cli.run_full_lifecycle(_args(tmp_path))
    text = (Path(bundle.submission_dir) / cli._SIDECAR_BUNDLE).read_text()
    payload = json.loads(text)
    # Phase 4 bundle sidecar carries promotable/score_claim canonical markers.
    assert payload["score_claim"] is False
    assert payload["promotable"] is False


def test_lint_sidecar_overall_clean_token(monkeypatch, tmp_path):
    bundle = _patch_layers(monkeypatch, tmp_path)
    cli.run_full_lifecycle(_args(tmp_path))
    text = (Path(bundle.submission_dir) / cli._SIDECAR_LINT).read_text()
    assert '"overall_clean": true' in text


def test_paired_sidecar_paired_pass_token(monkeypatch, tmp_path):
    bundle = _patch_layers(monkeypatch, tmp_path)
    cli.run_full_lifecycle(_args(tmp_path))
    text = (Path(bundle.submission_dir) / cli._SIDECAR_PAIRED).read_text()
    assert '"verdict": "PAIRED_PASS"' in text


def test_canonical_sidecar_names_match_phase_8_gate():
    # The exact filenames the Catalog #370 gate searches for.
    from tac.preflight import (
        _CHECK_370_PHASE_4_SIDECAR_FILENAMES,
        _CHECK_370_PHASE_5_SIDECAR_FILENAMES,
        _CHECK_370_PHASE_6_SIDECAR_FILENAMES,
        _CHECK_370_PHASE_7_SIDECAR_FILENAMES,
    )
    assert cli._SIDECAR_BUNDLE in _CHECK_370_PHASE_4_SIDECAR_FILENAMES
    assert cli._SIDECAR_LINT in _CHECK_370_PHASE_5_SIDECAR_FILENAMES
    assert cli._SIDECAR_COMPLIANCE in _CHECK_370_PHASE_6_SIDECAR_FILENAMES
    assert cli._SIDECAR_PAIRED in _CHECK_370_PHASE_7_SIDECAR_FILENAMES


# ---------------------------------------------------------------------------
# NO auto-gh-execution regression guard
# ---------------------------------------------------------------------------
def test_no_auto_gh_execution_anywhere_in_report(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path)
    _code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert report["gh_commands_fired"] is False
    # gh commands are emitted as text only, never executed.
    og = report["operator_gated_commands"]
    assert "gh release create" in og["step_1_host_archive"]
    assert "gh pr create" in og["step_2_create_pr"]


def test_source_does_not_subprocess_gh():
    src = Path(cli.__file__).read_text()
    # The CLI must NEVER subprocess / os.system / Popen anything (zero process
    # spawn) per CLAUDE.md "Executing actions with care".
    for forbidden in (
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
    ):
        assert forbidden not in src, f"CLI must not invoke {forbidden}"


# ---------------------------------------------------------------------------
# --execute paired-env discipline
# ---------------------------------------------------------------------------
def test_execute_without_paired_env_keeps_layer_5_plan_only(monkeypatch, tmp_path):
    monkeypatch.delenv(cli._ENV_CONFIRMED, raising=False)
    monkeypatch.delenv(cli._ENV_BUDGET, raising=False)
    _patch_layers(monkeypatch, tmp_path)
    code, report = cli.run_full_lifecycle(_args(tmp_path, execute=True, dry_run=False))
    # Execute mode without paired-env: Layer 5 stays plan-only; still reaches gate.
    assert code == cli.EXIT_OPERATOR_GATED
    assert "remains plan-only" in report["layers"]["layer_5_paired_auth_eval"]["paired_env"]


def test_execute_bare_confirmed_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(cli._ENV_CONFIRMED, "1")
    monkeypatch.delenv(cli._ENV_BUDGET, raising=False)
    _patch_layers(monkeypatch, tmp_path)
    code, report = cli.run_full_lifecycle(_args(tmp_path, execute=True, dry_run=False))
    assert code == cli.EXIT_CLI_ERROR
    assert report["layers"]["layer_5_paired_auth_eval"]["ok"] is False


def test_dry_run_is_default():
    parser = cli._build_parser()
    ns = parser.parse_args([
        "--lane-id", "x",
        "--substrate-trainer", "a.py",
        "--recipe-path", "b.yaml",
        "--archive-path", "c.zip",
        "--output-dir", "d/",
    ])
    assert ns.dry_run is True
    assert ns.execute is False


# ---------------------------------------------------------------------------
# --json output schema
# ---------------------------------------------------------------------------
def test_json_output_schema(monkeypatch, tmp_path, capsys):
    _patch_layers(monkeypatch, tmp_path)
    ns = _args(tmp_path, json=True)
    code = cli.main_from_namespace(ns) if hasattr(cli, "main_from_namespace") else None
    if code is None:
        # main() path
        code, report = cli.run_full_lifecycle(ns)
        payload = report
    else:
        payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "operator_pr_submission_full_lifecycle_v1"
    assert "layers" in payload
    assert payload["gh_commands_fired"] is False


def test_main_json_emits_valid_json(monkeypatch, tmp_path, capsys):
    bundle = _make_bundle(tmp_path)
    monkeypatch.setattr(cli, "build_compression_pipeline", lambda **kw: _FakePipeline())
    monkeypatch.setattr(
        cli, "build_archive_grammar_from_compression_pipeline_result",
        lambda **kw: _FakeGrammar(bundle.archive_sha256),
    )
    monkeypatch.setattr(cli, "build_submission_bundle", lambda **kw: bundle)
    monkeypatch.setattr(cli, "lint_submission_bundle", lambda *a, **kw: _lint_verdict(clean=True))
    monkeypatch.setattr(cli, "enforce_contest_compliance", lambda **kw: _compliance_verdict(clean=True))
    monkeypatch.setattr(
        cli, "plan_paired_auth_eval",
        lambda **kw: _paired_verdict(kind=PairedAuthEvalVerdictKind.PAIRED_PASS.value),
    )
    monkeypatch.setattr(cli, "_run_layer_6_gate", lambda submission_dir, repo_root: [])
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"x")
    argv = [
        "--lane-id", "lane_p9_test",
        "--substrate-trainer", "experiments/train_substrate_grayscale_lut.py",
        "--recipe-path", ".omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml",
        "--archive-path", str(archive),
        "--output-dir", str(tmp_path / "submission_dir"),
        "--predecessors", "@SajayR:56:HNeRV",
        "--json",
    ]
    code = cli.main(argv)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == cli.EXIT_OPERATOR_GATED
    assert payload["lifecycle_verdict"] == "OPERATOR-GATED"


# ---------------------------------------------------------------------------
# Integration: all 6 sister helpers + Phase 8 gate importable
# ---------------------------------------------------------------------------
def test_all_six_layer_helpers_importable():
    assert callable(cli.build_compression_pipeline)
    assert callable(cli.build_archive_grammar_from_compression_pipeline_result)
    assert callable(cli.build_submission_bundle)
    assert callable(cli.lint_submission_bundle)
    assert callable(cli.enforce_contest_compliance)
    assert callable(cli.plan_paired_auth_eval)


def test_phase_8_gate_importable_lazily():
    from tac.preflight import (
        check_no_pr_submission_without_canonical_compliance_verdict,
    )
    assert callable(check_no_pr_submission_without_canonical_compliance_verdict)


def test_exit_code_taxonomy_distinct():
    codes = {
        cli.EXIT_PACKET_CLEAN,
        cli.EXIT_LINT_VIOLATIONS,
        cli.EXIT_COMPLIANCE_ERRORS,
        cli.EXIT_MISSING_PAIRED_AXIS,
        cli.EXIT_OPERATOR_GATED,
        cli.EXIT_CLI_ERROR,
    }
    assert len(codes) == 6
    assert cli.EXIT_PACKET_CLEAN == 0
    assert cli.EXIT_OPERATOR_GATED == 4


# ---------------------------------------------------------------------------
# Layer 5 directs to dispatch_modal_paired_auth_eval.py for paid execution
# ---------------------------------------------------------------------------
def test_layer_5_directs_to_canonical_dispatch_tool(monkeypatch, tmp_path):
    _patch_layers(monkeypatch, tmp_path)
    _code, report = cli.run_full_lifecycle(_args(tmp_path))
    assert "dispatch_modal_paired_auth_eval.py" in report["layers"]["layer_5_paired_auth_eval"]["directs_to"]


# ---------------------------------------------------------------------------
# Live-repo regression guard: CLI is importable + parser builds
# ---------------------------------------------------------------------------
def test_live_repo_cli_help_builds():
    parser = cli._build_parser()
    assert parser.prog == "operator_pr_submission_full_lifecycle"


def test_live_repo_required_args_enforced():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing required args


def test_dispatch_modal_paired_tool_exists():
    assert (cli.REPO_ROOT / "tools" / "dispatch_modal_paired_auth_eval.py").is_file()
