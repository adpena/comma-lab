# SPDX-License-Identifier: MIT
"""Tests for the A1 CUDA-CPU drift discriminator verdict analyzer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "analyze_a1_cuda_cpu_drift_discriminator_verdict.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "a1_drift_discriminator_verdict_tool", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_pair(
    tool,
    variant_id: str,
    mechanism_hypothesis: str,
    *,
    cpu_pose: float = 3.286e-5,
    cuda_pose: float = 1.65e-4,
    cpu_seg: float = 5.6e-4,
    cuda_seg: float = 6.6e-4,
    cpu_score: float = 0.193,
    cuda_score: float = 0.226,
):
    return tool.VariantPair(
        variant_id=variant_id,
        mechanism_hypothesis=mechanism_hypothesis,
        archive_sha256="deadbeef" * 8,
        cpu_pose=cpu_pose,
        cuda_pose=cuda_pose,
        cpu_seg=cpu_seg,
        cuda_seg=cuda_seg,
        cpu_score=cpu_score,
        cuda_score=cuda_score,
        cpu_evidence_grade="contest-CPU-1to1",
        cuda_evidence_grade="contest-CUDA-1to1",
        cpu_evidence_path="/tmp/test/cpu.json",
        cuda_evidence_path="/tmp/test/cuda.json",
    )


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------


def test_validate_eval_tag_accepts_contest_cpu_record(tmp_path: Path) -> None:
    tool = load_tool()
    record = {
        "evidence_grade": "contest-CPU-1to1",
        "lane_tag": "[contest-CPU]",
    }
    # Must not raise.
    tool._validate_eval_tag(record, tmp_path / "cpu.json", axis="cpu")


def test_validate_eval_tag_accepts_contest_cuda_record(tmp_path: Path) -> None:
    tool = load_tool()
    record = {
        "evidence_grade": "contest-CUDA-1to1",
        "lane_tag": "[contest-CUDA]",
    }
    tool._validate_eval_tag(record, tmp_path / "cuda.json", axis="cuda")


def test_validate_eval_tag_refuses_macos_cpu_advisory(tmp_path: Path) -> None:
    tool = load_tool()
    record = {
        "evidence_grade": "macOS-CPU advisory only",
        "lane_tag": "[macOS-CPU advisory only]",
    }
    with pytest.raises(ValueError, match="non-1:1"):
        tool._validate_eval_tag(record, tmp_path / "macos.json", axis="cpu")


def test_validate_eval_tag_refuses_mps_proxy(tmp_path: Path) -> None:
    tool = load_tool()
    record = {
        "evidence_grade": "MPS-PROXY",
        "lane_tag": "[MPS-PROXY]",
    }
    with pytest.raises(ValueError, match="non-1:1"):
        tool._validate_eval_tag(record, tmp_path / "mps.json", axis="cpu")


def test_validate_eval_tag_refuses_contest_cpu_advisory_even_with_contest_cpu_substring(
    tmp_path: Path,
) -> None:
    """A '[contest-CPU advisory]' record contains 'contest-CPU' as a substring,
    but is NOT contest-faithful — must be refused."""
    tool = load_tool()
    record = {
        "evidence_grade": "contest-CPU advisory",
        "lane_tag": "[contest-CPU advisory]",
    }
    with pytest.raises(ValueError, match="non-1:1"):
        tool._validate_eval_tag(record, tmp_path / "advisory.json", axis="cpu")


def test_validate_eval_tag_refuses_macos_calibrated_even_though_dev_velocity(
    tmp_path: Path,
) -> None:
    """[macOS-CPU calibrated] is a valid dev-velocity tag, but the discriminator
    needs full 1:1 GHA hardware — refuse for verdict computation."""
    tool = load_tool()
    record = {
        "evidence_grade": "macos calibrated",
        "lane_tag": "[macOS-CPU calibrated]",
    }
    with pytest.raises(ValueError, match="non-1:1"):
        tool._validate_eval_tag(record, tmp_path / "calibrated.json", axis="cpu")


def test_validate_eval_tag_refuses_advisory_only(tmp_path: Path) -> None:
    tool = load_tool()
    record = {"evidence_grade": "advisory only", "lane_tag": "[advisory only]"}
    with pytest.raises(ValueError, match="non-1:1"):
        tool._validate_eval_tag(record, tmp_path / "advisory.json", axis="cuda")


def test_validate_eval_tag_refuses_missing_authoritative_marker(tmp_path: Path) -> None:
    tool = load_tool()
    record = {
        "evidence_grade": "smoke",
        "lane_tag": "[smoke]",
    }
    with pytest.raises(ValueError, match="missing authoritative cuda tag"):
        tool._validate_eval_tag(record, tmp_path / "smoke.json", axis="cuda")


# ---------------------------------------------------------------------------
# VariantPair math
# ---------------------------------------------------------------------------


def test_variant_pair_score_gap_and_ratios() -> None:
    tool = load_tool()
    p = _make_pair(
        tool, "v_baseline", "control",
        cuda_pose=1.65e-4, cpu_pose=3.286e-5,
        cuda_seg=6.6e-4, cpu_seg=5.6e-4,
        cuda_score=0.226, cpu_score=0.193,
    )
    assert p.score_gap() == pytest.approx(0.033, rel=1e-3)
    assert p.r_pose() == pytest.approx(5.02, rel=2e-2)
    assert p.r_seg() == pytest.approx(1.179, rel=2e-2)


def test_variant_pair_handles_near_zero_pose_without_div_by_zero() -> None:
    tool = load_tool()
    p = _make_pair(tool, "v_baseline", "control", cpu_pose=0.0)
    # Should not raise; should clamp denominator
    val = p.r_pose()
    assert val > 0


# ---------------------------------------------------------------------------
# compute_verdict — single-mechanism cases
# ---------------------------------------------------------------------------


def test_compute_verdict_primary_loader_mechanism_identified() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    # Loader-isolated drops R_pose dramatically (e.g. cuda_pose now ~ cpu_pose).
    loader = _make_pair(
        tool, "v_loader_isolated", "loader_byte_drift",
        cuda_pose=4e-5, cpu_pose=3.286e-5,
    )
    conv = _make_pair(tool, "v_conv_isolated", "conv_kernel_accumulation_drift")
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "PRIMARY_MECHANISM_IDENTIFIED"
    findings = {f["variant_id"]: f for f in verdict["isolation_findings"]}
    assert findings["v_loader_isolated"]["verdict"] == "PRIMARY_MECHANISM"
    assert findings["v_conv_isolated"]["verdict"] in (
        "MECHANISM_NOT_DOMINANT",
        "CONTRIBUTING_MECHANISM",
    )
    spec = verdict["registry_update_spec"]
    fields = [e["field"] for e in spec["fields_to_add_or_update"]]
    assert "loader_drift_correction" in fields


def test_compute_verdict_primary_conv_mechanism_identified() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    loader = _make_pair(tool, "v_loader_isolated", "loader_byte_drift")
    conv = _make_pair(
        tool, "v_conv_isolated", "conv_kernel_accumulation_drift",
        cuda_pose=4e-5,
    )
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "PRIMARY_MECHANISM_IDENTIFIED"
    fields = [e["field"] for e in verdict["registry_update_spec"]["fields_to_add_or_update"]]
    assert "conv_kernel_determinism_required" in fields


def test_compute_verdict_primary_hydra_mechanism_identified() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    loader = _make_pair(tool, "v_loader_isolated", "loader_byte_drift")
    conv = _make_pair(tool, "v_conv_isolated", "conv_kernel_accumulation_drift")
    hydra = _make_pair(
        tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity",
        cuda_pose=4e-5,
    )
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "PRIMARY_MECHANISM_IDENTIFIED"
    fields = [e["field"] for e in verdict["registry_update_spec"]["fields_to_add_or_update"]]
    assert "head_quantize_post_inference_dtype" in fields


# ---------------------------------------------------------------------------
# compute_verdict — multi-mechanism + negative-result cases
# ---------------------------------------------------------------------------


def test_compute_verdict_multi_mechanism_primary_when_two_drop_below_threshold() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    loader = _make_pair(
        tool, "v_loader_isolated", "loader_byte_drift", cuda_pose=4e-5
    )
    conv = _make_pair(
        tool, "v_conv_isolated", "conv_kernel_accumulation_drift",
        cuda_pose=5e-5,
    )
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "MULTI_MECHANISM_PRIMARY"


def test_compute_verdict_fourth_mechanism_when_no_isolation_helps() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    # All isolations produce same R_pose as baseline.
    loader = _make_pair(tool, "v_loader_isolated", "loader_byte_drift")
    conv = _make_pair(tool, "v_conv_isolated", "conv_kernel_accumulation_drift")
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "FOURTH_MECHANISM_HYPOTHESIS"
    # Per CLAUDE.md kill-as-last-resort: rationale must NOT use 'kill' / 'falsified'
    # language for this verdict.
    assert "do NOT kill" in verdict["verdict_rationale"]
    assert "operator" in verdict["verdict_rationale"]


def test_compute_verdict_inconclusive_no_baseline() -> None:
    tool = load_tool()
    loader = _make_pair(tool, "v_loader_isolated", "loader_byte_drift")
    verdict = tool.compute_verdict([loader])
    assert verdict["verdict"] == "INCONCLUSIVE_NO_BASELINE"


def test_compute_verdict_inconclusive_when_variants_missing() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    loader = _make_pair(
        tool, "v_loader_isolated", "loader_byte_drift", cuda_pose=4e-5
    )
    verdict = tool.compute_verdict([baseline, loader])
    assert verdict["verdict"] == "INCONCLUSIVE_VARIANTS_MISSING"


def test_compute_verdict_contributing_only_when_modest_drops_no_primary() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    # 40% drop → contributing but stays above primary threshold.
    loader = _make_pair(
        tool, "v_loader_isolated", "loader_byte_drift",
        cuda_pose=1.0e-4,  # ratio drops from ~5 to ~3
    )
    conv = _make_pair(tool, "v_conv_isolated", "conv_kernel_accumulation_drift")
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    assert verdict["verdict"] == "MULTI_MECHANISM_CONTRIBUTING_ONLY"


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------


def test_load_variant_pair_round_trip(tmp_path: Path) -> None:
    tool = load_tool()
    vdir = tmp_path / "variant"
    vdir.mkdir()
    manifest = {
        "variant_id": "v_baseline",
        "mechanism_hypothesis": "control",
        "archive_sha256": "abc123" * 10 + "abcd",
        "isolation_spec": {"kind": "control"},
    }
    (vdir / "discriminator_manifest.json").write_text(json.dumps(manifest))
    cpu_record = {
        "archive_sha256": manifest["archive_sha256"],
        "canonical_score": 0.193,
        "avg_posenet_dist": 3.3e-5,
        "avg_segnet_dist": 5.6e-4,
        "evidence_grade": "contest-CPU-1to1",
        "lane_tag": "[contest-CPU]",
    }
    cuda_record = {
        "archive_sha256": manifest["archive_sha256"],
        "canonical_score": 0.226,
        "avg_posenet_dist": 1.65e-4,
        "avg_segnet_dist": 6.6e-4,
        "evidence_grade": "contest-CUDA-1to1",
        "lane_tag": "[contest-CUDA]",
    }
    cpu_path = tmp_path / "cpu.json"
    cuda_path = tmp_path / "cuda.json"
    cpu_path.write_text(json.dumps(cpu_record))
    cuda_path.write_text(json.dumps(cuda_record))
    pair = tool.load_variant_pair(vdir, cpu_path, cuda_path)
    assert pair.variant_id == "v_baseline"
    assert pair.score_gap() == pytest.approx(0.033, rel=1e-3)


def test_load_variant_pair_refuses_sha_mismatch(tmp_path: Path) -> None:
    tool = load_tool()
    vdir = tmp_path / "variant"
    vdir.mkdir()
    manifest = {
        "variant_id": "v_baseline",
        "mechanism_hypothesis": "control",
        "archive_sha256": "aaaa" * 16,
        "isolation_spec": {"kind": "control"},
    }
    (vdir / "discriminator_manifest.json").write_text(json.dumps(manifest))
    cpu_record = {
        "archive_sha256": "bbbb" * 16,
        "canonical_score": 0.193,
        "avg_posenet_dist": 3.3e-5,
        "avg_segnet_dist": 5.6e-4,
        "evidence_grade": "contest-CPU-1to1",
        "lane_tag": "[contest-CPU]",
    }
    cuda_record = dict(cpu_record)
    cuda_record["evidence_grade"] = "contest-CUDA-1to1"
    cuda_record["lane_tag"] = "[contest-CUDA]"
    cuda_record["archive_sha256"] = manifest["archive_sha256"]
    cpu_path = tmp_path / "cpu.json"
    cuda_path = tmp_path / "cuda.json"
    cpu_path.write_text(json.dumps(cpu_record))
    cuda_path.write_text(json.dumps(cuda_record))
    with pytest.raises(ValueError, match="archive_sha256 mismatch"):
        tool.load_variant_pair(vdir, cpu_path, cuda_path)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_includes_per_variant_table_and_verdict() -> None:
    tool = load_tool()
    baseline = _make_pair(tool, "v_baseline", "control")
    loader = _make_pair(
        tool, "v_loader_isolated", "loader_byte_drift", cuda_pose=4e-5
    )
    conv = _make_pair(tool, "v_conv_isolated", "conv_kernel_accumulation_drift")
    hydra = _make_pair(tool, "v_hydra_isolated", "hydra_head_numerical_sensitivity")
    verdict = tool.compute_verdict([baseline, loader, conv, hydra])
    md = tool.render_markdown(verdict)
    assert "PRIMARY_MECHANISM_IDENTIFIED" in md
    assert "v_baseline" in md
    assert "v_loader_isolated" in md
    assert "loader_drift_correction" in md
