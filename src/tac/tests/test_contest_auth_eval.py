# SPDX-License-Identifier: MIT
"""Tests for experiments/contest_auth_eval.py — the canonical generic
contest-compliant auth evaluator.

Covers parsing + the structural promises the tool makes:
  - Parses upstream/evaluate.py's report.txt format correctly
  - Score components match the reported final score
  - Refuses missing inflate.sh / archive / upstream
  - Records full provenance
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cae() -> ModuleType:
    """Load contest_auth_eval as a module without depending on it being
    installed (it's a top-level experiments/ script, not a package member)."""
    spec = importlib.util.spec_from_file_location(
        "contest_auth_eval", REPO / "experiments" / "contest_auth_eval.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(cae):
    """Sanity: the module imports and exposes the expected entry points."""
    assert hasattr(cae, "main")
    assert hasattr(cae, "_parse_report")
    assert hasattr(cae, "_run_inflate")
    assert hasattr(cae, "_run_upstream_evaluate")
    assert cae.SCHEMA_VERSION == 1


def test_parse_report_baseline_format(cae, tmp_path: Path):
    """Parses the canonical upstream/evaluate.py report shape.

    Uses self-consistent values where the formula 100*seg + sqrt(10*pose) +
    25*rate exactly reproduces the reported final score. The historical
    "0.9001 baseline" run_log entry actually has a 0.108-gap between
    reported and recomputed — that's a separate baseline-provenance bug
    being investigated, NOT a parser bug. Council R3 #6 hardening
    correctly surfaces such formula divergences.
    """
    # Self-consistent: 100*0.002 + sqrt(10*0.0107) + 25*0.009 = 0.200 + 0.327 + 0.225 = 0.752
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.01070000
  Average SegNet Distortion: 0.00200000
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00900
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.7521
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    result = cae._parse_report(rp, archive_size=337748)

    assert result["schema_version"] == 1
    assert result["final_score"] == pytest.approx(0.7521, abs=1e-4)
    assert result["avg_posenet_dist"] == pytest.approx(0.0107, abs=1e-4)
    assert result["avg_segnet_dist"] == pytest.approx(0.0020, abs=1e-4)
    assert result["rate_unscaled"] == pytest.approx(0.009, abs=1e-4)
    assert result["score_seg_contribution"] == pytest.approx(0.20, abs=1e-3)
    assert result["score_pose_contribution"] == pytest.approx(0.327, abs=1e-3)
    assert result["score_rate_contribution"] == pytest.approx(0.225, abs=1e-3)
    assert result["canonical_score"] == pytest.approx(
        result["score_recomputed_from_components"]
    )
    assert result["canonical_score_source"] == "score_recomputed_from_components"
    assert result["reported_final_score_display_rounded"] == pytest.approx(
        result["final_score"]
    )
    assert result["n_samples"] == 600
    assert result["archive_size_bytes"] == 337748


def test_parse_report_rejects_score_formula_divergence(cae, tmp_path: Path):
    """Council R3 #6: if the reported final doesn't match the recomputed
    score within 0.01, raise loud — protects against upstream changing
    the formula weights (which would silently shift our scores)."""
    # 100*0.0024 + sqrt(10*0.0107) + 25*0.00899 = 0.24 + 0.327 + 0.2248 = 0.792
    # but reported as 0.9001 — formula divergence of 0.108 must raise.
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.0024
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00899
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.9001
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    with pytest.raises(RuntimeError, match="formula divergence"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_rejects_nan(cae, tmp_path: Path):
    """Council R3 #5: NaN / inf parses as float() but is not a valid score."""
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: nan
  Average SegNet Distortion: 0.0024
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.009
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = nan
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    # NaN slips past the strict number regex (won't match) so the parser
    # raises "could not parse" — also acceptable. Either way: LOUD refusal.
    with pytest.raises(RuntimeError, match=r"non-finite|could not parse"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_rejects_wrong_n_samples(cae, tmp_path: Path):
    """Council R3 #3 backstop: if the .raw byte-count check missed a partial
    inflate, the n_samples mismatch in the report catches it."""
    # 599 samples instead of 600 — partial inflate slipped through
    text = """=== Evaluation results over 599 samples ===
  Average PoseNet Distortion: 0.01
  Average SegNet Distortion: 0.002
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.009
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.74
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    with pytest.raises(RuntimeError, match="expected 600 samples"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_score_recomputation_consistent(cae, tmp_path: Path):
    """The recomputed score from components should be CLOSE to the reported
    final (within a small tolerance for rounding in the source report)."""
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.00240
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00899
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.7919
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    result = cae._parse_report(rp, archive_size=337748)

    recomputed = result["score_recomputed_from_components"]
    # Within 0.01 of the reported final
    assert abs(recomputed - result["final_score"]) < 0.01, (
        f"recomputed={recomputed:.4f} reported={result['final_score']:.4f}"
    )
    assert result["canonical_score"] == pytest.approx(recomputed)
    assert result["canonical_score_source"] == "score_recomputed_from_components"


def test_parse_report_marks_reported_score_as_display_rounded(cae, tmp_path: Path):
    """The upstream report prints a rounded final score; ranking tools must use
    the recomputed component score instead."""
    # 100*0.001 + sqrt(10*0.004) + 25*0.003 = 0.375, reported as 0.38.
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.004
  Average SegNet Distortion: 0.001
  Submission file size: 3 bytes
  Original uncompressed size: 1000 bytes
  Compression Rate: 0.003
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.38
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)

    result = cae._parse_report(rp, archive_size=3)

    assert result["final_score"] == pytest.approx(0.38)
    assert result["canonical_score"] == pytest.approx(0.375)
    assert result["score_recomputed_from_components"] == pytest.approx(0.375)
    assert result["reported_final_score_display_rounded"] == pytest.approx(0.38)
    assert result["score_rounding_abs_delta"] == pytest.approx(0.005)
    assert result["score_reported_rounded_differs_from_canonical"] is True


def test_parse_report_raw_stdout_not_treated_as_path(cae):
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.00240
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00899
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.7919
"""
    result = cae._parse_report(text, archive_size=337748, source="stdout")

    assert result["score_recomputed_from_components"] == pytest.approx(0.7920010743750296)


def test_parse_report_uses_exact_byte_rate_not_rounded_printed_rate(cae, tmp_path: Path):
    """The report prints Compression Rate to 8 decimals; custody score uses bytes."""
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00003236
  Average SegNet Distortion: 0.00064260
  Submission file size: 186780 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00497477
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.21
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)

    result = cae._parse_report(rp, archive_size=186780)

    exact_rate = 186780 / 37545489
    assert result["rate_unscaled"] == pytest.approx(exact_rate)
    assert result["rate_unscaled_reported_rounded"] == pytest.approx(0.00497477)
    assert result["score_rate_contribution"] == pytest.approx(25 * exact_rate)
    assert abs(result["score_rate_contribution"] - 25 * 0.00497477) > 1e-8


def test_upstream_evaluate_records_elapsed_seconds(cae, tmp_path: Path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "evaluate.py").write_text("print('unused')\n")
    video_names = tmp_path / "names.txt"
    video_names.write_text("0.mkv\n")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "archive.zip").write_bytes(b"archive")

    captured: dict[str, object] = {}

    def fake_run(*_args, **_kwargs):
        captured["env"] = dict(_kwargs.get("env") or {})
        report = submission / "report.txt"
        report.write_text("""=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.00240
  Submission file size: 7 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00000019
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.57
""")
        return subprocess.CompletedProcess(
            args=["python", "evaluate.py"],
            returncode=0,
            stdout=report.read_text(),
            stderr="",
        )

    monkeypatch.setattr(cae, "_validate_uncompressed_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cae.subprocess, "run", fake_run)

    result = cae._run_upstream_evaluate(
        upstream,
        submission,
        tmp_path / "videos",
        video_names,
        "cuda",
    )

    assert "evaluate_elapsed_seconds" in result
    assert result["evaluate_elapsed_seconds"] >= 0.0
    assert captured["env"]["DALI_DISABLE_NVML"] == "1"


def test_contest_auth_eval_source_records_inflate_budget_fields() -> None:
    text = (REPO / "experiments" / "contest_auth_eval.py").read_text()

    assert "inflate_timeout_seconds" in text
    assert "evaluate_timeout_seconds" in text
    assert "inflate_elapsed_seconds" in text
    assert "contest_auth_eval_elapsed_seconds" in text
    assert "inflate_runtime_manifest" in text
    assert "runtime_tree_sha256" in text
    assert "effective_inflate_python" in text


def test_json_out_implies_durable_work_dir_source_contract() -> None:
    text = (REPO / "experiments" / "contest_auth_eval.py").read_text()

    assert 'elif args.json_out is not None and not args.allow_temp_work_dir:' in text
    assert 'f"{json_out.stem}_workdir"' in text
    assert "cleanup = False" in text


def test_cuda_t4_auth_eval_contract_is_exact_eval_not_promotion(cae):
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": True,
        },
    )

    assert contract["lane_tag"] == "[contest-CUDA]"
    assert contract["score_axis"] == "contest_cuda"
    assert contract["exact_cuda_eval_complete"] is True
    assert contract["score_claim_valid"] is True
    assert contract["score_claim"] is True
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert "pre_submission_compliance_check_not_recorded" in contract["promotion_blockers"]


def test_linux_cpu_auth_eval_contract_stays_non_promotional(cae):
    contract = cae._auth_eval_evidence_contract(
        "cpu",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": False,
        },
    )

    assert contract["lane_tag"] == "[contest-CPU]"
    assert contract["score_axis"] == "contest_cpu"
    assert contract["score_claim"] is True
    assert contract["score_claim_valid"] is True
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert "pre_submission_compliance_check_not_recorded" in contract["promotion_blockers"]


def test_evidence_contract_tags_cpu_as_leaderboard_reproduction(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cpu",
        600,
        {"gpu_t4_match": False, "platform_system": "Linux", "platform_machine": "x86_64"},
    )

    assert contract["evidence_grade"] == "contest-CPU"
    assert contract["lane_tag"] == "[contest-CPU]"
    assert contract["score_axis"] == "contest_cpu"
    assert contract["cpu_leaderboard_reproduction_eligible"] is True
    assert contract["score_claim_valid"] is True
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False


def test_evidence_contract_demotes_modal_cpu_advisory_even_on_linux_x86(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cpu",
        600,
        {
            "gpu_t4_match": False,
            "modal_auth_eval_advisory_only": True,
            "platform_machine": "x86_64",
            "platform_system": "Linux",
        },
    )

    assert contract["evidence_grade"] == "B"
    assert contract["lane_tag"] == "[diagnostic-auth-eval]"
    assert contract["score_axis"] == "diagnostic_cpu"
    assert contract["cpu_leaderboard_reproduction_eligible"] is False
    assert (
        "modal_training_wrapper_auth_eval_advisory_only"
        in contract["diagnostic_blockers"]
    )


def test_evidence_contract_downgrades_macos_cpu_to_advisory(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cpu",
        600,
        {"platform_system": "Darwin", "platform_machine": "arm64"},
    )

    assert contract["evidence_grade"] == "macOS-CPU advisory"
    assert contract["lane_tag"] == "[macOS-CPU advisory]"
    assert contract["score_axis"] == "cpu_advisory"
    assert contract["cpu_leaderboard_reproduction_eligible"] is False
    assert contract["hardware_compliance_blocker"] == "contest_cpu_requires_linux_x86_64"


def test_evidence_contract_keeps_cuda_t4_exact_eval_non_promotional(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": True,
        },
    )

    assert contract["evidence_grade"] == "contest-CUDA"
    assert contract["lane_tag"] == "[contest-CUDA]"
    assert contract["score_axis"] == "contest_cuda"
    assert contract["exact_cuda_eval_complete"] is True
    assert contract["score_claim"] is True
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["score_claim_valid"] is True


def test_runtime_dependency_manifest_hashes_fixed_inflate_files(cae, tmp_path: Path):
    runtime = tmp_path / "submission"
    upstream = tmp_path / "upstream"
    runtime.mkdir()
    upstream.mkdir()
    inflate_sh = runtime / "inflate.sh"
    renderer = runtime / "inflate_renderer.py"
    cpp_decoder = runtime / "range_mask_codec.cpp"
    ignored_payload = runtime / "renderer.bin"
    inflate_sh.write_text("#!/bin/sh\npython inflate_renderer.py\n")
    renderer.write_text("print('v1')\n")
    cpp_decoder.write_text("// decoder v1\n")
    ignored_payload.write_bytes(b"archive payload belongs in archive.zip")
    (upstream / "evaluate.py").write_text("print('eval')\n")

    manifest_a = cae._runtime_dependency_manifest(inflate_sh, upstream)
    cpp_decoder.write_text("// decoder v2\n")
    manifest_b = cae._runtime_dependency_manifest(inflate_sh, upstream)

    files = {entry["relative_path"]: entry for entry in manifest_a["files"]}
    assert "inflate.sh" in files
    assert "inflate_renderer.py" in files
    assert "range_mask_codec.cpp" in files
    assert "renderer.bin" not in files
    assert manifest_a["upstream_evaluate_py"]["sha256"]
    assert manifest_a["runtime_tree_sha256"] != manifest_b["runtime_tree_sha256"]


def test_runtime_dependency_manifest_hashes_repo_local_tac_imports(cae, tmp_path: Path):
    runtime = tmp_path / "submission"
    upstream = tmp_path / "upstream"
    repo = tmp_path / "repo"
    tac = repo / "src" / "tac"
    runtime.mkdir()
    upstream.mkdir()
    tac.mkdir(parents=True)
    inflate_sh = runtime / "inflate.sh"
    inflate_py = runtime / "inflate_renderer.py"
    helper = tac / "runtime_helper.py"
    inflate_sh.write_text("#!/bin/sh\npython inflate_renderer.py\n")
    inflate_py.write_text("from tac.runtime_helper import decode\n")
    (tac / "__init__.py").write_text("")
    helper.write_text("def decode():\n    return 'v1'\n")
    (upstream / "evaluate.py").write_text("print('eval')\n")

    manifest_a = cae._runtime_dependency_manifest(inflate_sh, upstream, repo_root=repo)
    helper.write_text("def decode():\n    return 'v2'\n")
    manifest_b = cae._runtime_dependency_manifest(inflate_sh, upstream, repo_root=repo)

    tac_manifest = manifest_a["repo_local_tac_import_manifest"]
    files = {entry["relative_path"]: entry for entry in tac_manifest["files"]}
    assert "src/tac/runtime_helper.py" in files
    assert tac_manifest["discovery"] == "static_ast_recursive_import_closure"
    assert manifest_a["runtime_tree_sha256"] != manifest_b["runtime_tree_sha256"]


def test_runtime_dependency_manifest_hashes_transitive_tac_imports(cae, tmp_path: Path):
    runtime = tmp_path / "submission"
    upstream = tmp_path / "upstream"
    repo = tmp_path / "repo"
    tac = repo / "src" / "tac"
    runtime.mkdir()
    upstream.mkdir()
    tac.mkdir(parents=True)
    inflate_sh = runtime / "inflate.sh"
    inflate_py = runtime / "inflate_renderer.py"
    helper = tac / "helper.py"
    leaf = tac / "leaf.py"
    inflate_sh.write_text("#!/bin/sh\npython inflate_renderer.py\n")
    inflate_py.write_text("from tac.helper import decode\n")
    (tac / "__init__.py").write_text("")
    helper.write_text("from tac.leaf import value\n\ndef decode():\n    return value\n")
    leaf.write_text("value = 'v1'\n")
    (upstream / "evaluate.py").write_text("print('eval')\n")

    manifest_a = cae._runtime_dependency_manifest(inflate_sh, upstream, repo_root=repo)
    leaf.write_text("value = 'v2'\n")
    manifest_b = cae._runtime_dependency_manifest(inflate_sh, upstream, repo_root=repo)

    files = {
        entry["relative_path"]
        for entry in manifest_a["repo_local_tac_import_manifest"]["files"]
    }
    assert "src/tac/helper.py" in files
    assert "src/tac/leaf.py" in files
    assert manifest_a["runtime_tree_sha256"] != manifest_b["runtime_tree_sha256"]


def test_record_inflate_runtime_artifacts_captures_packed_payload_summary(cae, tmp_path: Path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    summary = extracted / "renderer_payload_unpack_summary.json"
    summary.write_text(json.dumps({"members": [{"name": "renderer.bin", "bytes": 56093}]}))
    prov: dict[str, object] = {}

    cae._record_inflate_runtime_artifacts(prov, tmp_path, extracted)

    artifacts = prov["inflate_runtime_artifacts"]
    packed = artifacts["renderer_payload_unpack_summary"]
    assert packed["sha256"] == cae._sha256(summary, prefix=0)
    assert packed["payload"]["members"][0]["name"] == "renderer.bin"
    assert (tmp_path / "provenance.json").is_file()


def test_record_inflated_output_artifacts_hashes_raw_outputs(cae, tmp_path: Path):
    inflated = tmp_path / "inflated"
    inflated.mkdir()
    names = tmp_path / "video_names.txt"
    names.write_text("0.mkv\n")
    raw = inflated / "0.raw"
    raw.write_bytes(b"abc123")
    prov: dict[str, object] = {}

    manifest = cae._record_inflated_output_artifacts(prov, tmp_path, inflated, names)

    assert manifest["schema"] == "contest_auth_eval_inflated_output_manifest_v1"
    assert manifest["raw_file_count"] == 1
    assert manifest["total_bytes"] == 6
    assert manifest["files"][0]["relative_path"] == "0.raw"
    assert manifest["files"][0]["sha256"] == cae._sha256(raw, prefix=0)
    assert len(manifest["aggregate_sha256"]) == 64
    recorded = prov["inflated_output_manifest"]
    assert recorded["payload"]["aggregate_sha256"] == manifest["aggregate_sha256"]
    assert (tmp_path / "inflated_outputs_manifest.json").is_file()
    assert (tmp_path / "provenance.json").is_file()


def test_run_inflate_defaults_python_to_current_interpreter(
    cae,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local exact-eval must run shell-configured Python inflaters in this venv.

    Public PR101/A1-style packets import dependencies such as brotli from the
    evaluator environment. If contest_auth_eval is launched with
    `.venv/bin/python`, the inflate subprocess should inherit that interpreter
    unless the caller explicitly overrides `PYTHON`, `PYTHON_BIN`,
    `PACT_PYTHON_BIN`, or `UV_PYTHON`.
    """
    captured: dict[str, str] = {}
    video_names = tmp_path / "names.txt"
    video_names.write_text("0.mkv\n")
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n")
    archive_dir = tmp_path / "archive"
    inflated_dir = tmp_path / "inflated"
    archive_dir.mkdir()

    def fake_run(cmd, *, timeout, check, env):
        captured.update(env)
        raw_path = Path(cmd[3]) / "0.raw"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("wb") as f:
            f.truncate(1164 * 874 * 1200 * 3)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.delenv("PYTHON", raising=False)
    monkeypatch.delenv("PYTHON_BIN", raising=False)
    monkeypatch.delenv("PACT_PYTHON_BIN", raising=False)
    monkeypatch.delenv("UV_PYTHON", raising=False)
    monkeypatch.setattr(cae.subprocess, "run", fake_run)

    cae._run_inflate(inflate_sh, archive_dir, inflated_dir, video_names, timeout=5)

    assert captured["PYTHON"] == sys.executable
    assert captured["PYTHON_BIN"] == sys.executable
    assert captured["PACT_PYTHON_BIN"] == sys.executable
    assert captured["UV_PYTHON"] == sys.executable


def test_run_inflate_applies_diagnostic_env_to_inflate_only(
    cae,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    video_names = tmp_path / "names.txt"
    video_names.write_text("0.mkv\n")
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n")
    archive_dir = tmp_path / "archive"
    inflated_dir = tmp_path / "inflated"
    archive_dir.mkdir()

    def fake_run(cmd, *, timeout, check, env):
        captured.update(env)
        raw_path = Path(cmd[3]) / "0.raw"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("wb") as f:
            f.truncate(1164 * 874 * 1200 * 3)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(cae.subprocess, "run", fake_run)

    cae._run_inflate(
        inflate_sh,
        archive_dir,
        inflated_dir,
        video_names,
        timeout=5,
        extra_env={"CUDA_VISIBLE_DEVICES": ""},
    )

    assert captured["CUDA_VISIBLE_DEVICES"] == ""


def test_auth_artifact_output_outside_work_dir_resolves_relative_to_cwd(
    cae,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "work"
    cwd = tmp_path / "repo"
    work_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    resolved = cae._resolve_auth_artifact_output_under_work_dir(
        work_dir,
        Path("queue/artifact.json"),
        label="scorer-input hash artifact",
        allow_outside_work_dir=True,
    )

    assert resolved == cwd / "queue" / "artifact.json"
    assert not str(resolved).startswith(str(work_dir))


def test_parse_inflate_env_overrides_allowlist(cae) -> None:
    parsed = cae._parse_inflate_env_overrides(
        ["CUDA_VISIBLE_DEVICES=", "PACT_FORCE_INFLATE_DEVICE=cpu", "INFLATE_MODE=probe"]
    )

    assert parsed == {
        "CUDA_VISIBLE_DEVICES": "",
        "INFLATE_MODE": "probe",
        "PACT_FORCE_INFLATE_DEVICE": "cpu",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cae._parse_inflate_env_overrides(["CUDA_VISIBLE_DEVICES"])
    with pytest.raises(ValueError, match="not allowed"):
        cae._parse_inflate_env_overrides(["AWS_SECRET_ACCESS_KEY=nope"])


def test_inflate_device_policy_builds_diagnostic_env(cae) -> None:
    env, blockers = cae._inflate_env_for_device_policy("cpu", {})

    assert env == {
        "CUDA_VISIBLE_DEVICES": "",
        "PACT_INFLATE_DEVICE": "cpu",
    }
    assert blockers == ["inflate_device_policy_cpu"]

    env, blockers = cae._inflate_env_for_device_policy(
        "cuda",
        {"INFLATE_MODE": "probe"},
    )
    assert env == {
        "INFLATE_MODE": "probe",
        "PACT_INFLATE_DEVICE": "cuda",
    }
    assert blockers == [
        "inflate_device_policy_cuda",
        "inflate_env_overrides_present",
    ]

    with pytest.raises(ValueError, match="conflicting PACT_INFLATE_DEVICE"):
        cae._inflate_env_for_device_policy("cpu", {"PACT_INFLATE_DEVICE": "cuda"})


def test_auth_eval_evidence_contract_demotes_inflate_env_diagnostic(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": True,
        },
        diagnostic_blockers=["inflate_env_overrides_present"],
    )

    assert contract["evidence_grade"] == "B"
    assert contract["score_axis"] == "diagnostic_cuda"
    assert contract["score_claim"] is False
    assert contract["exact_cuda_eval_complete"] is False
    assert contract["diagnostic_blockers"] == ["inflate_env_overrides_present"]


def test_auth_eval_evidence_contract_marks_non_t4_cuda_as_diagnostic_axis(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": False,
        },
    )

    assert contract["evidence_grade"] == "B"
    assert contract["lane_tag"] == "[diagnostic-auth-eval]"
    assert contract["score_axis"] == "diagnostic_cuda"
    assert contract["score_claim"] is False


def test_auth_eval_evidence_contract_accepts_linux_a100_cuda_axis(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": False,
            "gpu_model": "NVIDIA A100-SXM4-40GB",
        },
    )

    assert contract["evidence_grade"] == "contest-CUDA"
    assert contract["score_axis"] == "contest_cuda"
    assert contract["score_claim"] is True
    assert contract["exact_cuda_eval_complete"] is True


def test_auth_eval_evidence_contract_requires_linux_for_cuda_axis(cae) -> None:
    contract = cae._auth_eval_evidence_contract(
        "cuda",
        600,
        {
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "gpu_t4_match": False,
            "gpu_model": "NVIDIA A100-SXM4-40GB",
        },
    )

    assert contract["evidence_grade"] == "B"
    assert contract["lane_tag"] == "[diagnostic-auth-eval]"
    assert contract["score_axis"] == "diagnostic_cuda"
    assert contract["score_claim"] is False


def test_expected_runtime_tree_hash_mismatch_fails_closed(cae) -> None:
    prov = {
        "inflate_runtime_manifest": {
            "runtime_tree_sha256": "a" * 64,
        }
    }
    with pytest.raises(RuntimeError, match="runtime tree hash mismatch"):
        cae._validate_expected_runtime_tree(prov, "b" * 64)


def test_parse_report_malformed_raises(cae, tmp_path: Path):
    """A malformed report (missing required fields) must raise loudly."""
    rp = tmp_path / "bad.txt"
    rp.write_text("garbage with no fields")
    with pytest.raises(RuntimeError, match="could not parse"):
        cae._parse_report(rp, archive_size=1000)


def test_extract_archive_zip_slip_protection(cae, tmp_path: Path):
    """A malicious archive with `../` paths must NOT escape the dest dir."""
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../escaped.txt", b"pwned")
    with pytest.raises(RuntimeError, match=r"NONCANONICAL|zip-slip"):
        cae._extract_archive(bad_zip, tmp_path / "dest")


def test_extract_archive_rejects_prefix_traversal_member(cae, tmp_path: Path):
    """`../dest_evil/p` must not pass a string-prefix destination check."""
    bad_zip = tmp_path / "evil_prefix.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../dest_evil/p", b"pwned")
    with pytest.raises(RuntimeError, match=r"NONCANONICAL|zip-slip"):
        cae._extract_archive(bad_zip, tmp_path / "dest")
    assert not (tmp_path / "dest_evil" / "p").exists()


def test_extract_archive_rejects_backslash_member(cae, tmp_path: Path):
    bad_zip = tmp_path / "evil_backslash.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr(r"..\\p", b"pwned")
    with pytest.raises(RuntimeError, match="BACKSLASH"):
        cae._extract_archive(bad_zip, tmp_path / "dest")


def test_extract_archive_normal(cae, tmp_path: Path):
    """Normal archive extracts cleanly."""
    src = tmp_path / "good.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("renderer.bin", b"fake renderer")
        z.writestr("masks.mkv", b"fake masks")
    members = cae._extract_archive(src, tmp_path / "dest")
    assert sorted(members) == ["masks.mkv", "renderer.bin"]
    assert (tmp_path / "dest" / "renderer.bin").exists()
    assert (tmp_path / "dest" / "masks.mkv").exists()


def test_archive_member_validator_accepts_charged_pr89_final_bias_member(cae):
    """PR89-style `fb` is a charged archive atom, not a housekeeping sidecar."""
    cae._validate_archive_members(["x", "fb"])


def test_archive_member_validator_accepts_charged_pr86_hpac_members(cae):
    """PR86's HPAC replay stores charged model/state blobs as compressed .pt files."""
    cae._validate_archive_members(
        ["master.pt.gz", "slave.pt.gz", "hpac.pt.ppmd", "tokens.bin", "meta.pt"]
    )


def test_archive_member_validator_accepts_charged_pr85_qma9_members(cae):
    """PR85 bridge archives expose QMA9 as a charged mask payload member."""
    cae._validate_archive_members(
        ["masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin"]
    )


def test_archive_member_validator_accepts_segmap_lct_members(cae):
    """SegMap/LCT archive members are charged payloads, not debug sidecars."""
    cae._validate_archive_members(
        ["segmap_weights.tar.xz", "grayscale.mkv", "optimized_poses.pt", "class_targets.fp16"]
    )


def test_archive_member_validator_accepts_brotli_logical_members(cae):
    """Diet-packed logical members may carry a final .br compression suffix."""
    cae._validate_archive_members(
        ["masks.mkv.br", "grayscale.mkv.br", "optimized_poses.pt.br", "payload.bin.br"]
    )


def test_archive_member_validator_rejects_unknown_brotli_logical_member(cae):
    with pytest.raises(RuntimeError, match="UNKNOWN file types"):
        cae._validate_archive_members(["debug.dat.br"])


def test_archive_member_validator_still_rejects_unknown_extensionless_debug_member(cae):
    with pytest.raises(RuntimeError, match="UNKNOWN file types"):
        cae._validate_archive_members(["x", "debug"])


def test_main_refuses_missing_archive(cae, tmp_path: Path):
    """--archive must point to an existing file."""
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(tmp_path / "nonexistent.zip"),
        "--inflate-sh", str(REPO / "submissions" / "robust_current" / "inflate.sh"),
        "--upstream-dir", str(REPO / "upstream"),
    ]
    with pytest.raises(SystemExit, match="--archive does not exist"):
        cae.main()


def test_main_rejects_nonpositive_scorer_hash_batch_before_path_resolution(cae, tmp_path: Path):
    sys.argv = [
        "contest_auth_eval.py",
        "--archive",
        str(tmp_path / "nonexistent.zip"),
        "--inflate-sh",
        str(tmp_path / "inflate.sh"),
        "--upstream-dir",
        str(tmp_path / "upstream"),
        "--scorer-input-cache-hashes-out",
        str(tmp_path / "hashes.json"),
        "--scorer-input-cache-hash-batch-pairs",
        "0",
    ]
    with pytest.raises(SystemExit, match="scorer-input-cache-hash-batch-pairs must be >= 1"):
        cae.main()


def test_main_refuses_missing_inflate(cae, tmp_path: Path):
    """--inflate-sh must point to an existing file."""
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip eocd
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(fake_archive),
        "--inflate-sh", str(tmp_path / "nonexistent.sh"),
        "--upstream-dir", str(REPO / "upstream"),
    ]
    with pytest.raises(SystemExit, match="--inflate-sh does not exist"):
        cae.main()


def test_main_refuses_missing_upstream(cae, tmp_path: Path):
    """--upstream-dir must contain evaluate.py."""
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    fake_inflate = tmp_path / "inflate.sh"
    fake_inflate.write_text("#!/bin/bash\necho ok\n")
    fake_inflate.chmod(0o755)
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(fake_archive),
        "--inflate-sh", str(fake_inflate),
        "--upstream-dir", str(tmp_path),
    ]
    with pytest.raises(SystemExit, match=r"missing evaluate\.py"):
        cae.main()


def test_help_includes_canonical_one_liner(cae):
    """The docstring should make the canonical contest flow obvious."""
    doc = cae.__doc__
    assert doc is not None
    assert "archive.zip" in doc
    assert "inflate.sh" in doc
    assert "upstream/evaluate.py" in doc
    assert "CANONICAL" in doc
