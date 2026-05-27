# SPDX-License-Identifier: MIT
"""Tests for tac.submission_packet.paired_auth_eval (Phase 7 Layer 5).

Per Phase 1 audit specification memo §3 Phase 6 / Layer 5 acceptance contract:
  - plan_paired_auth_eval dry-run plan emission verified for canonical
    Linux x86_64 substrates AND macOS-CPU advisory non-promotion
  - PairedAuthEvalVerdict frozen-dataclass invariants enforced
  - Catalog #192 macOS-CPU structural non-promotion verified
  - Catalog #226 canonical gate_auth_eval_call routing preserved
  - Catalog #245 + #339 + #360 silent-no-spawn extinction routed via
    canonical helpers (not subprocess directly)
  - Catalog #341 Tier-A routing markers (predicted_delta_adjustment=0.0
    + promotable=False + axis_tag=[predicted]) verified
  - Catalog #335 cathedral consumer canonical contract compliance
  - Catalog #323 canonical Provenance umbrella threading verified
  - End-to-end Phase 4 SubmissionBundleResult -> Phase 7 PairedAuthEvalVerdict
    round-trip + CLI smoke
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.submission_packet import (
    PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID,
    PAIRED_AUTH_EVAL_SCHEMA_VERSION,
    PHASE_7_LAYER_VERSION,
    DependencyClosureManifest,
    PairedAuthEvalError,
    PairedAuthEvalVerdict,
    PairedAuthEvalVerdictKind,
    SubmissionBundleResult,
    SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    derive_paired_auth_eval_provenance,
    plan_paired_auth_eval,
    reconstruct_verdict_from_disk,
)
from tac.submission_packet.paired_auth_eval import (
    _CANONICAL_CUDA_GPU_CLASSES,
    _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES,
    _CANONICAL_PER_HOUR_RATES_USD,
    _COST_BAND_BUDGET_USD,
    _CPU_TARGET_TO_HARDWARE_SUBSTRATE,
    _CUDA_PLATFORM_TO_HARDWARE_SUBSTRATE_PREFIX,
    _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU,
    _EVIDENCE_GRADE_CPU_ONLY,
    _EVIDENCE_GRADE_CUDA_ONLY,
    _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING,
    _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY,
    _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS,
    _PLACEHOLDER_RATIONALES,
    _build_blocked_pre_dispatch_verdict,
    _build_dry_run_plan_verdict,
    _derive_evidence_grade,
    _estimate_per_axis_cost,
    _is_macos_hardware_substrate,
    _resolve_cpu_hardware_substrate,
    _resolve_cuda_hardware_substrate,
    _utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_canonical_bundle(
    *,
    tmp_path: Path,
    lane_id: str = "lane_phase_7_test",
    substrate_id: str = "phase_7_test_substrate",
) -> SubmissionBundleResult:
    """Build a minimal canonical SubmissionBundleResult fixture."""
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir(parents=True, exist_ok=True)
    archive_bytes = b"mock archive bytes for phase 7 tests"
    sha = hashlib.sha256(archive_bytes).hexdigest()
    (submission_dir / "archive.zip").write_bytes(archive_bytes)
    (submission_dir / "inflate.sh").write_text("#!/bin/bash\necho ok\n")
    (submission_dir / "inflate.py").write_text("# stub\n")
    (submission_dir / "README.md").write_text("# test\n")
    (submission_dir / "report.txt").write_text("placeholder\n")
    (submission_dir / "archive_manifest.json").write_text("{}")
    dep_manifest = DependencyClosureManifest(
        declared_dependencies=("numpy",),
        dependency_budget=2,
        within_budget=True,
        numpy_portable=True,
    )
    return SubmissionBundleResult(
        schema_version=SUBMISSION_BUNDLE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_sha256=sha,
        archive_bytes=len(archive_bytes),
        submission_dir=str(submission_dir),
        inflate_sh_path=str(submission_dir / "inflate.sh"),
        inflate_py_path=str(submission_dir / "inflate.py"),
        inflate_py_loc=1,
        inflate_py_loc_budget=200,
        inflate_py_loc_waiver_rationale=None,
        readme_md_path=str(submission_dir / "README.md"),
        report_txt_path=str(submission_dir / "report.txt"),
        archive_manifest_path=str(submission_dir / "archive_manifest.json"),
        dependency_closure_manifest=dep_manifest,
        select_inflate_device_routing="canonical_helper",
        pythonpath_self_containment_status="clean",
        vendor_pythonpath_self_containment=False,
        runtime_dep_closure=("numpy",),
        measurement_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; submission-bundle-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_submission_bundle",
        canonical_equation_id=SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
    )


def _make_canonical_verdict(
    *,
    bundle: SubmissionBundleResult,
    verdict_kind: str = "BLOCKED_PRE_DISPATCH",
    rationale: str = "DRY-RUN canonical test verdict",
    promotable: bool = False,
    score_claim: bool = False,
    cuda_score: float | None = None,
    cpu_score: float | None = None,
    cuda_hardware: str = "linux_x86_64_modal_t4",
    cpu_hardware: str = "linux_x86_64_modal_cpu",
    cuda_call_id: str = "",
    cpu_call_id: str = "",
    forbidden_macos: bool = False,
    axis_tag: str = "[predicted]",
    evidence_grade: str = _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING,
    dry_run: bool = True,
) -> PairedAuthEvalVerdict:
    """Build a canonical PairedAuthEvalVerdict for invariant tests."""
    cuda_axis_tag = "[contest-CUDA]" if cuda_score is not None else "[missing]"
    if cpu_score is not None:
        cpu_axis_tag = "[macOS-CPU advisory]" if forbidden_macos else "[contest-CPU]"
    else:
        cpu_axis_tag = "[missing]"
    cuda_cpu_gap = (
        float(cuda_score - cpu_score)
        if (cuda_score is not None and cpu_score is not None)
        else None
    )
    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id=bundle.lane_id,
        substrate_id=bundle.substrate_id,
        archive_sha256_paired=(
            "" if verdict_kind == "BLOCKED_PRE_DISPATCH" else bundle.archive_sha256
        ),
        archive_bytes=bundle.archive_bytes,
        submission_dir=bundle.submission_dir,
        verdict=verdict_kind,
        verdict_rationale=rationale,
        cuda_score=cuda_score,
        cuda_axis_tag=cuda_axis_tag,
        cuda_hardware_substrate=cuda_hardware,
        cuda_call_id=cuda_call_id,
        cuda_seg_distortion=None,
        cuda_pose_distortion=None,
        cuda_rate_term=None,
        cuda_auth_eval_json_path="",
        cuda_elapsed_seconds=0.0,
        cuda_cost_usd=0.0,
        cpu_score=cpu_score,
        cpu_axis_tag=cpu_axis_tag,
        cpu_hardware_substrate=cpu_hardware,
        cpu_call_id=cpu_call_id,
        cpu_seg_distortion=None,
        cpu_pose_distortion=None,
        cpu_rate_term=None,
        cpu_auth_eval_json_path="",
        cpu_elapsed_seconds=0.0,
        cpu_cost_usd=0.0,
        cuda_cpu_gap=cuda_cpu_gap,
        cost_band="smoke",
        budget_usd=1.0,
        total_cost_usd=0.0,
        measurement_utc=_utc_now_iso(),
        axis_tag=axis_tag,
        score_claim=score_claim,
        promotable=promotable,
        evidence_grade=evidence_grade,
        canonical_helper_invocation="tac.submission_packet.plan_paired_auth_eval",
        canonical_equation_id=PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform="modal",
        cuda_gpu="T4",
        cpu_target="linux_x86_64_modal",
        dry_run=dry_run,
        forbidden_macos_axis_detected=forbidden_macos,
        canonical_provenance={},
    )


# ---------------------------------------------------------------------------
# Module-level constants pinned
# ---------------------------------------------------------------------------


class TestModuleConstantsPinned:
    """Module-level constants pinned (Catalog #185 sister)."""

    def test_schema_version_pinned(self) -> None:
        assert PAIRED_AUTH_EVAL_SCHEMA_VERSION == "submission_paired_auth_eval_v1_20260526"

    def test_phase_layer_version_pinned(self) -> None:
        assert PHASE_7_LAYER_VERSION == (
            "phase_7_submission_paired_auth_eval_canonical_landed_20260526"
        )

    def test_canonical_equation_id_pinned(self) -> None:
        assert PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID == (
            "paired_auth_eval_canonical_helper_consolidation_savings_v1"
        )

    def test_verdict_kinds_canonical_set(self) -> None:
        kinds = {v.value for v in PairedAuthEvalVerdictKind}
        expected = {
            "PAIRED_PASS",
            "PAIRED_PARTIAL_CUDA_ONLY",
            "PAIRED_PARTIAL_CPU_ONLY",
            "BLOCKED_PRE_DISPATCH",
            "BLOCKED_HARVEST",
            "BLOCKED_AXIS_MISMATCH",
            "BLOCKED_HARDWARE_NON_COMPLIANT",
        }
        assert kinds == expected

    def test_canonical_cuda_gpu_classes(self) -> None:
        # Per Catalog #215 + sister Catalog #244
        assert "T4" in _CANONICAL_CUDA_GPU_CLASSES
        assert "A100" in _CANONICAL_CUDA_GPU_CLASSES
        assert "H100" in _CANONICAL_CUDA_GPU_CLASSES
        assert "4090" in _CANONICAL_CUDA_GPU_CLASSES

    def test_canonical_linux_x86_64_cpu_substrates(self) -> None:
        # Per Catalog #192 1:1 contest-compliant hardware
        assert "linux_x86_64_modal_cpu" in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES
        assert "linux_x86_64_vastai_cpu" in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES
        assert "linux_x86_64_gha_cpu" in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES
        # macOS NOT in canonical set per Catalog #192
        assert "macos_arm64" not in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES
        assert "darwin_arm64_m5_max_macos_cpu_advisory" not in _CANONICAL_LINUX_X86_64_CPU_SUBSTRATES

    def test_forbidden_macos_hardware_tokens(self) -> None:
        # Per Catalog #192 forbidden authoritative axes
        assert "macos_arm64" in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
        assert "darwin_arm64" in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
        assert "apple_silicon" in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
        assert "macos_cpu" in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS

    def test_cost_band_budget_envelope(self) -> None:
        # Per Catalog #270 cost-band envelope
        assert _COST_BAND_BUDGET_USD["smoke"] == 1.00
        assert _COST_BAND_BUDGET_USD["full"] == 5.00

    def test_placeholder_rationales_canonical(self) -> None:
        # Per Catalog #287 placeholder rejection
        assert "<rationale>" in _PLACEHOLDER_RATIONALES
        assert "<reason>" in _PLACEHOLDER_RATIONALES
        assert "" in _PLACEHOLDER_RATIONALES

    def test_evidence_grade_tokens_canonical(self) -> None:
        # Per Catalog #287 / #323
        assert "[contest-CUDA; contest-CPU; paired-axis-empirical]" == (
            _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU
        )
        assert "[macOS-CPU advisory; paired-non-promotable]" == (
            _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY
        )


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHardwareSubstrateResolvers:
    """_resolve_cuda_hardware_substrate / _resolve_cpu_hardware_substrate."""

    def test_resolve_cuda_modal_t4(self) -> None:
        assert _resolve_cuda_hardware_substrate(platform="modal", gpu="T4") == (
            "linux_x86_64_modal_t4"
        )

    def test_resolve_cuda_modal_a100(self) -> None:
        assert _resolve_cuda_hardware_substrate(platform="modal", gpu="A100") == (
            "linux_x86_64_modal_a100"
        )

    def test_resolve_cuda_vastai_4090(self) -> None:
        assert _resolve_cuda_hardware_substrate(platform="vastai", gpu="4090") == (
            "linux_x86_64_vastai_4090"
        )

    def test_resolve_cuda_lightning_t4(self) -> None:
        assert _resolve_cuda_hardware_substrate(platform="lightning", gpu="T4") == (
            "linux_x86_64_lightning_t4"
        )

    def test_resolve_cpu_modal(self) -> None:
        assert _resolve_cpu_hardware_substrate("linux_x86_64_modal") == (
            "linux_x86_64_modal_cpu"
        )

    def test_resolve_cpu_gha(self) -> None:
        assert _resolve_cpu_hardware_substrate("linux_x86_64_gha") == (
            "linux_x86_64_gha_cpu"
        )

    def test_resolve_cpu_darwin_advisory(self) -> None:
        # macOS advisory permitted at the resolver layer; structural refusal at verdict
        assert _resolve_cpu_hardware_substrate("darwin_arm64_advisory") == (
            "darwin_arm64_m5_max_macos_cpu_advisory"
        )


class TestMacosHardwareDetector:
    """_is_macos_hardware_substrate per Catalog #192."""

    def test_macos_arm64_detected(self) -> None:
        assert _is_macos_hardware_substrate("macos_arm64") is True

    def test_darwin_arm64_detected(self) -> None:
        assert _is_macos_hardware_substrate("darwin_arm64") is True

    def test_apple_silicon_detected(self) -> None:
        assert _is_macos_hardware_substrate("apple_silicon") is True

    def test_canonical_linux_x86_64_modal_t4_not_macos(self) -> None:
        assert _is_macos_hardware_substrate("linux_x86_64_modal_t4") is False

    def test_canonical_linux_x86_64_modal_cpu_not_macos(self) -> None:
        assert _is_macos_hardware_substrate("linux_x86_64_modal_cpu") is False

    def test_canonical_linux_x86_64_gha_cpu_not_macos(self) -> None:
        assert _is_macos_hardware_substrate("linux_x86_64_gha_cpu") is False


class TestCostEstimation:
    """_estimate_per_axis_cost per Catalog #270."""

    def test_modal_t4_smoke_cost_estimate(self) -> None:
        # T4 modal = $0.59/h * 0.25h smoke = $0.1475
        cost = _estimate_per_axis_cost(platform="modal", gpu_or_cpu="T4", cost_band="smoke")
        assert 0.10 < cost < 0.20

    def test_modal_a100_smoke_cost_estimate(self) -> None:
        # A100 modal = $3.40/h * 0.25h smoke = $0.85
        cost = _estimate_per_axis_cost(platform="modal", gpu_or_cpu="A100", cost_band="smoke")
        assert 0.80 < cost < 0.90

    def test_modal_cpu_smoke_cost_estimate(self) -> None:
        # CPU modal = $0.06/h * 0.25h smoke = $0.015
        cost = _estimate_per_axis_cost(platform="modal", gpu_or_cpu="cpu", cost_band="smoke")
        assert 0.01 < cost < 0.05

    def test_full_band_more_expensive_than_smoke(self) -> None:
        smoke = _estimate_per_axis_cost(platform="modal", gpu_or_cpu="T4", cost_band="smoke")
        full = _estimate_per_axis_cost(platform="modal", gpu_or_cpu="T4", cost_band="full")
        assert full > smoke


class TestEvidenceGradeDerivation:
    """_derive_evidence_grade per Catalog #287 / #323."""

    def test_paired_pass_promotable_yields_canonical(self) -> None:
        grade = _derive_evidence_grade(
            verdict="PAIRED_PASS",
            cuda_hardware="linux_x86_64_modal_t4",
            cpu_hardware="linux_x86_64_modal_cpu",
            promotable=True,
        )
        assert grade == _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU

    def test_macos_axis_forces_advisory_grade(self) -> None:
        grade = _derive_evidence_grade(
            verdict="PAIRED_PASS",
            cuda_hardware="linux_x86_64_modal_t4",
            cpu_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
            promotable=False,
        )
        assert grade == _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY

    def test_partial_cuda_only(self) -> None:
        grade = _derive_evidence_grade(
            verdict="PAIRED_PARTIAL_CUDA_ONLY",
            cuda_hardware="linux_x86_64_modal_t4",
            cpu_hardware="linux_x86_64_modal_cpu",
            promotable=False,
        )
        assert grade == _EVIDENCE_GRADE_CUDA_ONLY

    def test_partial_cpu_only(self) -> None:
        grade = _derive_evidence_grade(
            verdict="PAIRED_PARTIAL_CPU_ONLY",
            cuda_hardware="linux_x86_64_modal_t4",
            cpu_hardware="linux_x86_64_modal_cpu",
            promotable=False,
        )
        assert grade == _EVIDENCE_GRADE_CPU_ONLY

    def test_default_pending(self) -> None:
        grade = _derive_evidence_grade(
            verdict="BLOCKED_PRE_DISPATCH",
            cuda_hardware="linux_x86_64_modal_t4",
            cpu_hardware="linux_x86_64_modal_cpu",
            promotable=False,
        )
        assert grade == _EVIDENCE_GRADE_NON_PROMOTABLE_PENDING


class TestProvenanceDerivation:
    """derive_paired_auth_eval_provenance canonical fields pinned."""

    def test_provenance_canonical_fields(self) -> None:
        prov = derive_paired_auth_eval_provenance(
            lane_id="lane_test",
            substrate_id="test_sub",
            archive_sha256="a" * 64,
            measurement_utc=_utc_now_iso(),
            cuda_platform="modal",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
        )
        # Catalog #341 defaults
        assert prov["axis_tag"] == "[predicted]"
        assert prov["score_claim"] is False
        assert prov["promotable"] is False
        # Catalog #190 canonical helper invocation
        assert prov["canonical_helper_invocation"] == (
            "tac.submission_packet.plan_paired_auth_eval"
        )
        # Catalog #344 canonical equation
        assert prov["canonical_equation_id"] == PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID
        assert prov["canonical_equation_status"] == "FORMALIZATION_PENDING"
        assert prov["schema_version"] == PAIRED_AUTH_EVAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# PairedAuthEvalVerdict invariants (Catalog #127 + #192 + #221 + #341 + #287 + #323)
# ---------------------------------------------------------------------------


class TestPairedAuthEvalVerdictInvariants:
    """Frozen-dataclass __post_init__ invariants per canonical contract."""

    def test_canonical_blocked_pre_dispatch_construction(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        v = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="canonical synthetic blocker",
        )
        assert v.verdict == "BLOCKED_PRE_DISPATCH"
        assert v.score_claim is False
        assert v.promotable is False

    def test_invalid_schema_version_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="schema_version"):
            PairedAuthEvalVerdict(
                schema_version="wrong_version",
                lane_id=bundle.lane_id,
                substrate_id=bundle.substrate_id,
                archive_sha256_paired="",
                archive_bytes=bundle.archive_bytes,
                submission_dir=bundle.submission_dir,
                verdict="BLOCKED_PRE_DISPATCH",
                verdict_rationale="test",
                cuda_score=None,
                cuda_axis_tag="[missing]",
                cuda_hardware_substrate="linux_x86_64_modal_t4",
                cuda_call_id="",
                cuda_seg_distortion=None,
                cuda_pose_distortion=None,
                cuda_rate_term=None,
                cuda_auth_eval_json_path="",
                cuda_elapsed_seconds=0.0,
                cuda_cost_usd=0.0,
                cpu_score=None,
                cpu_axis_tag="[missing]",
                cpu_hardware_substrate="linux_x86_64_modal_cpu",
                cpu_call_id="",
                cpu_seg_distortion=None,
                cpu_pose_distortion=None,
                cpu_rate_term=None,
                cpu_auth_eval_json_path="",
                cpu_elapsed_seconds=0.0,
                cpu_cost_usd=0.0,
                cuda_cpu_gap=None,
                cost_band="smoke",
                budget_usd=1.0,
                total_cost_usd=0.0,
                measurement_utc=_utc_now_iso(),
                axis_tag="[predicted]",
                score_claim=False,
                promotable=False,
                evidence_grade=_EVIDENCE_GRADE_NON_PROMOTABLE_PENDING,
                canonical_helper_invocation="tac.submission_packet.plan_paired_auth_eval",
                canonical_equation_id=PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID,
                canonical_equation_status="FORMALIZATION_PENDING",
                cuda_platform="modal",
                cuda_gpu="T4",
                cpu_target="linux_x86_64_modal",
                dry_run=True,
                forbidden_macos_axis_detected=False,
                canonical_provenance={},
            )

    def test_invalid_verdict_kind_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="verdict"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="INVALID_VERDICT_KIND",
                rationale="test rationale",
            )

    def test_placeholder_rationale_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="substantive"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="BLOCKED_PRE_DISPATCH",
                rationale="<rationale>",
            )

    def test_short_rationale_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="substantive"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="BLOCKED_PRE_DISPATCH",
                rationale="ab",
            )

    def test_empty_rationale_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="substantive"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="BLOCKED_PRE_DISPATCH",
                rationale="",
            )

    def test_invalid_sha256_length_rejected_for_non_blocked_pre_dispatch(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        # Build a valid verdict, then attempt to re-construct with bad sha
        valid = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="PAIRED_PARTIAL_CUDA_ONLY",
            rationale="bundle synthetic with bad sha test",
            cuda_score=0.226,
            cuda_call_id="fc-cuda-test",
        )
        with pytest.raises(ValueError, match="archive_sha256_paired"):
            PairedAuthEvalVerdict(**{**valid.as_dict(), "archive_sha256_paired": "badsha"})

    def test_negative_score_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PARTIAL_CUDA_ONLY",
                rationale="negative score test",
                cuda_score=-0.5,
                cuda_hardware="linux_x86_64_modal_t4",
            )

    def test_nan_score_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="finite"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PARTIAL_CUDA_ONLY",
                rationale="nan score test",
                cuda_score=float("nan"),
            )

    def test_invalid_cost_band_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="canonical synthetic",
        )
        with pytest.raises(ValueError, match="cost_band"):
            PairedAuthEvalVerdict(
                **{**verdict.as_dict(), "cost_band": "invalid_band"}
            )

    def test_invalid_cuda_gpu_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="canonical synthetic",
        )
        with pytest.raises(ValueError, match="cuda_gpu"):
            PairedAuthEvalVerdict(
                **{**verdict.as_dict(), "cuda_gpu": "RTX_3090"}
            )

    def test_invalid_cpu_target_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="canonical synthetic",
        )
        with pytest.raises(ValueError, match="cpu_target"):
            PairedAuthEvalVerdict(
                **{**verdict.as_dict(), "cpu_target": "windows_10_cpu"}
            )

    def test_macos_axis_with_promotable_true_rejected(self, tmp_path: Path) -> None:
        """Catalog #192 + CLAUDE.md non-negotiable: macOS axis forces promotable=False."""
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="Catalog #192"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PASS",
                rationale="macos with promotable=True attempted",
                cpu_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                forbidden_macos=True,
                promotable=True,
                cuda_score=0.226,
                cpu_score=0.193,
                cuda_call_id="fc-test-cuda",
                cpu_call_id="fc-test-cpu",
                axis_tag="[contest-CUDA; contest-CPU]",
                evidence_grade=_EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU,
            )

    def test_forbidden_macos_mismatch_with_hardware_rejected(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="macOS"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="BLOCKED_HARDWARE_NON_COMPLIANT",
                rationale="macOS hardware but forbidden flag missing",
                cpu_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                forbidden_macos=False,  # MISSING flag despite hardware
            )

    def test_promotable_true_without_paired_pass_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="PAIRED_PASS"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PARTIAL_CUDA_ONLY",
                rationale="promotable=True attempted on partial verdict",
                cuda_score=0.226,
                cuda_call_id="fc-test",
                promotable=True,
                evidence_grade=_EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU,
                axis_tag="[contest-CUDA; contest-CPU]",
            )

    def test_score_claim_requires_promotable(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="score_claim"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PASS",
                rationale="score_claim without promotable test",
                cuda_score=0.226,
                cpu_score=0.193,
                cuda_call_id="fc-cuda",
                cpu_call_id="fc-cpu",
                score_claim=True,
                promotable=False,
            )

    def test_paired_pass_requires_canonical_axis_tag_when_promotable(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="axis_tag"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PASS",
                rationale="paired pass promotable wrong axis tag",
                cuda_score=0.226,
                cpu_score=0.193,
                cuda_call_id="fc-cuda",
                cpu_call_id="fc-cpu",
                promotable=True,
                score_claim=True,
                axis_tag="[predicted]",
                evidence_grade=_EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU,
            )

    def test_canonical_evidence_grade_validated(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="evidence_grade"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="BLOCKED_PRE_DISPATCH",
                rationale="evidence grade invalid",
                evidence_grade="[invalid; not-canonical]",
            )

    def test_canonical_equation_id_pinned(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="canonical",
        )
        with pytest.raises(ValueError, match="canonical_equation_id"):
            PairedAuthEvalVerdict(
                **{**verdict.as_dict(), "canonical_equation_id": "wrong_id"}
            )

    def test_paired_pass_requires_cuda_call_id_when_not_dry_run(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cuda_call_id"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PASS",
                rationale="paired pass without call_ids",
                cuda_score=0.226,
                cpu_score=0.193,
                cuda_call_id="",
                cpu_call_id="fc-cpu",
                dry_run=False,
            )

    def test_paired_pass_requires_cpu_call_id_when_not_dry_run(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cpu_call_id"):
            _make_canonical_verdict(
                bundle=bundle,
                verdict_kind="PAIRED_PASS",
                rationale="paired pass without cpu call_id",
                cuda_score=0.226,
                cpu_score=0.193,
                cuda_call_id="fc-cuda",
                cpu_call_id="",
                dry_run=False,
            )

    def test_as_dict_round_trip(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = _make_canonical_verdict(
            bundle=bundle,
            verdict_kind="BLOCKED_PRE_DISPATCH",
            rationale="round-trip test canonical",
        )
        d = verdict.as_dict()
        # Reconstruct from dict
        v2 = PairedAuthEvalVerdict(**d)
        assert v2.verdict == verdict.verdict
        assert v2.archive_sha256_paired == verdict.archive_sha256_paired
        assert v2.lane_id == verdict.lane_id


# ---------------------------------------------------------------------------
# plan_paired_auth_eval canonical helper happy paths
# ---------------------------------------------------------------------------


class TestPlanPairedAuthEvalDryRun:
    """plan_paired_auth_eval dry-run plan emission verified."""

    def test_canonical_linux_dry_run_blocked_pre_dispatch(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        # Dry-run mode emits BLOCKED_PRE_DISPATCH per the plan-only contract
        assert verdict.verdict == "BLOCKED_PRE_DISPATCH"
        assert verdict.dry_run is True
        assert verdict.cuda_hardware_substrate == "linux_x86_64_modal_t4"
        assert verdict.cpu_hardware_substrate == "linux_x86_64_modal_cpu"
        assert verdict.forbidden_macos_axis_detected is False
        assert verdict.promotable is False
        assert verdict.score_claim is False
        assert verdict.axis_tag == "[predicted]"

    def test_macos_advisory_dry_run_blocked_hardware_non_compliant(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="darwin_arm64_advisory",
            dry_run=True,
        )
        assert verdict.verdict == "BLOCKED_HARDWARE_NON_COMPLIANT"
        assert verdict.forbidden_macos_axis_detected is True
        assert verdict.cpu_hardware_substrate == "darwin_arm64_m5_max_macos_cpu_advisory"
        assert verdict.cpu_axis_tag == "[macOS-CPU advisory]"
        assert verdict.promotable is False
        assert verdict.evidence_grade == _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY

    def test_full_band_higher_estimated_cost(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        v_smoke = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        v_full = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="full",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        assert v_full.budget_usd > v_smoke.budget_usd

    def test_excessive_gpu_with_small_budget_blocked_pre_dispatch(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        # H100 + smoke = 6.25 * 0.25 = $1.56 > $0.30 budget
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="H100",
            cpu_target="linux_x86_64_modal",
            budget_usd=0.30,
            dry_run=True,
        )
        assert verdict.verdict == "BLOCKED_PRE_DISPATCH"
        assert "exceeds budget" in verdict.verdict_rationale
        assert verdict.dry_run is True

    def test_canonical_provenance_threaded(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        # Catalog #323 canonical Provenance umbrella
        assert "axis_tag" in verdict.canonical_provenance
        assert verdict.canonical_provenance["axis_tag"] == "[predicted]"
        assert verdict.canonical_provenance["score_claim"] is False
        assert verdict.canonical_provenance["promotable"] is False
        assert verdict.canonical_provenance["canonical_equation_id"] == (
            PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID
        )

    def test_archive_bytes_preserved(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        assert verdict.archive_bytes == bundle.archive_bytes
        assert verdict.submission_dir == bundle.submission_dir

    def test_lane_substrate_lineage_preserved(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(
            tmp_path=tmp_path,
            lane_id="lane_pr111_candidate_20260601",
            substrate_id="nscs06_v8_chroma_lut",
        )
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        assert verdict.lane_id == "lane_pr111_candidate_20260601"
        assert verdict.substrate_id == "nscs06_v8_chroma_lut"


class TestPlanPairedAuthEvalValidation:
    """plan_paired_auth_eval argument validation paths."""

    def test_non_bundle_result_rejected(self) -> None:
        with pytest.raises(ValueError, match="SubmissionBundleResult"):
            plan_paired_auth_eval(
                submission_bundle_result="not_a_bundle",  # type: ignore[arg-type]
            )

    def test_invalid_cost_band_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cost_band"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cost_band="megasmoke",
            )

    def test_invalid_cuda_gpu_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cuda_gpu"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cuda_gpu="RTX_3090",
            )

    def test_invalid_cuda_platform_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cuda_platform"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cuda_platform="aws_ec2",
            )

    def test_invalid_cpu_target_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="cpu_target"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cpu_target="windows_cpu",
            )

    def test_negative_budget_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="budget_usd"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                budget_usd=-1.0,
            )

    def test_execute_without_operator_approval_rejected(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(PairedAuthEvalError, match="operator_approved_handle"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                dry_run=False,  # execute mode
                operator_approved_handle=None,
            )

    def test_execute_with_operator_approval_raises_operator_gated(
        self, tmp_path: Path
    ) -> None:
        """Execute branch is operator-gated per CLAUDE.md."""
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        with pytest.raises(PairedAuthEvalError, match="operator-gated"):
            plan_paired_auth_eval(
                submission_bundle_result=bundle,
                dry_run=False,
                operator_approved_handle="adpena:2026-05-26T00:00:00Z",
            )


# ---------------------------------------------------------------------------
# reconstruct_verdict_from_disk
# ---------------------------------------------------------------------------


def _write_auth_eval_json(
    path: Path,
    *,
    archive_sha: str,
    final_score: float = 0.226,
    seg: float = 0.067,
    pose: float = 1e-5,
    rate: float = 0.119,
) -> Path:
    """Write a canonical contest_auth_eval.json fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "final_score": final_score,
        "seg_distortion": seg,
        "pose_distortion": pose,
        "rate_term": rate,
        "archive_sha256": archive_sha,
        "score_axis": "contest_cuda",
    }
    path.write_text(json.dumps(payload))
    return path


class TestReconstructVerdictFromDisk:
    """reconstruct_verdict_from_disk happy + failure paths."""

    def test_paired_pass_reconstruction(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = _write_auth_eval_json(
            tmp_path / "cuda" / "contest_auth_eval_cuda.json",
            archive_sha=bundle.archive_sha256,
            final_score=0.226,
        )
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu" / "contest_auth_eval_cpu.json",
            archive_sha=bundle.archive_sha256,
            final_score=0.193,
        )
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda-test",
            cpu_call_id="fc-cpu-test",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="linux_x86_64_modal",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.02,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "PAIRED_PASS"
        assert verdict.promotable is True
        assert verdict.score_claim is True
        assert verdict.axis_tag == "[contest-CUDA; contest-CPU]"
        assert verdict.cuda_score == 0.226
        assert verdict.cpu_score == 0.193
        assert abs(verdict.cuda_cpu_gap - 0.033) < 0.001
        assert verdict.cuda_call_id == "fc-cuda-test"
        assert verdict.cpu_call_id == "fc-cpu-test"
        assert verdict.evidence_grade == _EVIDENCE_GRADE_CONTEST_CUDA_PLUS_CPU
        assert verdict.archive_sha256_paired == bundle.archive_sha256
        assert verdict.forbidden_macos_axis_detected is False

    def test_axis_mismatch_detected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = _write_auth_eval_json(
            tmp_path / "cuda.json",
            archive_sha=bundle.archive_sha256,
        )
        # CPU axis claims DIFFERENT archive_sha
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu.json",
            archive_sha="b" * 64,
        )
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda",
            cpu_call_id="fc-cpu",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="linux_x86_64_modal",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.02,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "BLOCKED_AXIS_MISMATCH"
        assert verdict.promotable is False
        assert verdict.score_claim is False

    def test_macos_axis_blocked_hardware_non_compliant(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = _write_auth_eval_json(
            tmp_path / "cuda.json",
            archive_sha=bundle.archive_sha256,
        )
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu.json",
            archive_sha=bundle.archive_sha256,
        )
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda",
            cpu_call_id="fc-cpu",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="darwin_arm64_advisory",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.0,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "BLOCKED_HARDWARE_NON_COMPLIANT"
        assert verdict.forbidden_macos_axis_detected is True
        assert verdict.promotable is False
        assert verdict.score_claim is False

    def test_partial_cuda_only(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = _write_auth_eval_json(
            tmp_path / "cuda.json",
            archive_sha=bundle.archive_sha256,
            final_score=0.226,
        )
        # CPU JSON exists but has no final_score
        cpu_json = tmp_path / "cpu.json"
        cpu_json.write_text(json.dumps({"failed": True, "archive_sha256": bundle.archive_sha256}))
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda",
            cpu_call_id="fc-cpu",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="linux_x86_64_modal",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.02,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "PAIRED_PARTIAL_CUDA_ONLY"
        assert verdict.promotable is False
        assert verdict.cuda_score == 0.226
        assert verdict.cpu_score is None

    def test_partial_cpu_only(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = tmp_path / "cuda.json"
        cuda_json.write_text(json.dumps({"failed": True, "archive_sha256": bundle.archive_sha256}))
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu.json",
            archive_sha=bundle.archive_sha256,
            final_score=0.193,
        )
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda",
            cpu_call_id="fc-cpu",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="linux_x86_64_modal",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.02,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "PAIRED_PARTIAL_CPU_ONLY"
        assert verdict.promotable is False
        assert verdict.cpu_score == 0.193
        assert verdict.cuda_score is None

    def test_both_axes_failed_blocked_harvest(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = tmp_path / "cuda.json"
        cuda_json.write_text(json.dumps({"failed": True}))
        cpu_json = tmp_path / "cpu.json"
        cpu_json.write_text(json.dumps({"failed": True}))
        verdict = reconstruct_verdict_from_disk(
            submission_bundle_result=bundle,
            cuda_auth_eval_json_path=cuda_json,
            cpu_auth_eval_json_path=cpu_json,
            cuda_call_id="fc-cuda",
            cpu_call_id="fc-cpu",
            cost_band="smoke",
            cuda_gpu="T4",
            cuda_platform="modal",
            cpu_target="linux_x86_64_modal",
            budget_usd=1.0,
            cuda_cost_usd=0.15,
            cpu_cost_usd=0.02,
            cuda_elapsed_seconds=120.0,
            cpu_elapsed_seconds=900.0,
        )
        assert verdict.verdict == "BLOCKED_HARVEST"
        assert verdict.promotable is False
        assert verdict.score_claim is False

    def test_missing_cuda_json_raises(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu.json",
            archive_sha=bundle.archive_sha256,
        )
        with pytest.raises(PairedAuthEvalError, match="cuda_auth_eval"):
            reconstruct_verdict_from_disk(
                submission_bundle_result=bundle,
                cuda_auth_eval_json_path=tmp_path / "nonexistent_cuda.json",
                cpu_auth_eval_json_path=cpu_json,
                cuda_call_id="fc-cuda",
                cpu_call_id="fc-cpu",
                cost_band="smoke",
                cuda_gpu="T4",
                cuda_platform="modal",
                cpu_target="linux_x86_64_modal",
                budget_usd=1.0,
                cuda_cost_usd=0.0,
                cpu_cost_usd=0.0,
                cuda_elapsed_seconds=0.0,
                cpu_elapsed_seconds=0.0,
            )

    def test_unparseable_json_raises(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        cuda_json = tmp_path / "cuda.json"
        cuda_json.write_text("not valid json{")
        cpu_json = _write_auth_eval_json(
            tmp_path / "cpu.json",
            archive_sha=bundle.archive_sha256,
        )
        with pytest.raises(PairedAuthEvalError, match="parseable"):
            reconstruct_verdict_from_disk(
                submission_bundle_result=bundle,
                cuda_auth_eval_json_path=cuda_json,
                cpu_auth_eval_json_path=cpu_json,
                cuda_call_id="fc-cuda",
                cpu_call_id="fc-cpu",
                cost_band="smoke",
                cuda_gpu="T4",
                cuda_platform="modal",
                cpu_target="linux_x86_64_modal",
                budget_usd=1.0,
                cuda_cost_usd=0.0,
                cpu_cost_usd=0.0,
                cuda_elapsed_seconds=0.0,
                cpu_elapsed_seconds=0.0,
            )

    def test_non_bundle_result_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="SubmissionBundleResult"):
            reconstruct_verdict_from_disk(
                submission_bundle_result="not_a_bundle",  # type: ignore[arg-type]
                cuda_auth_eval_json_path=tmp_path / "x",
                cpu_auth_eval_json_path=tmp_path / "y",
                cuda_call_id="",
                cpu_call_id="",
                cost_band="smoke",
                cuda_gpu="T4",
                cuda_platform="modal",
                cpu_target="linux_x86_64_modal",
                budget_usd=1.0,
                cuda_cost_usd=0.0,
                cpu_cost_usd=0.0,
                cuda_elapsed_seconds=0.0,
                cpu_elapsed_seconds=0.0,
            )


# ---------------------------------------------------------------------------
# Phase 4 integration round-trip
# ---------------------------------------------------------------------------


class TestPhase4IntegrationRoundTrip:
    """Phase 4 SubmissionBundleResult -> Phase 7 PairedAuthEvalVerdict round-trip."""

    def test_bundle_archive_sha_threaded_to_verdict(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        # Dry-run plan-only verdict preserves the bundle sha for sha-locked invariant tracking
        # (the verdict carries the sha bundle that BOTH axes would run against; this is the
        # canonical contract that downstream consumers consume to verify the plan matches the
        # final dispatch). Note the dry-run verdict_kind is BLOCKED_HARDWARE_NON_COMPLIANT when
        # macOS is detected, else BLOCKED_PRE_DISPATCH per the plan-only contract.
        assert verdict.archive_sha256_paired == bundle.archive_sha256
        # Bundle sha preserved in lane lineage
        assert verdict.lane_id == bundle.lane_id
        assert verdict.archive_bytes == bundle.archive_bytes

    def test_bundle_lane_substrate_preserved_in_provenance(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(
            tmp_path=tmp_path,
            lane_id="lane_canonical_test_20260526",
            substrate_id="canonical_test_substrate",
        )
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        assert verdict.canonical_provenance["lane_id"] == bundle.lane_id
        assert verdict.canonical_provenance["substrate_id"] == bundle.substrate_id


# ---------------------------------------------------------------------------
# Cathedral consumer (Catalog #335)
# ---------------------------------------------------------------------------


class TestCathedralConsumer:
    """Phase 7 cathedral consumer (Catalog #335 canonical contract)."""

    def test_consumer_module_imports(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        assert consumer.CONSUMER_NAME == "paired_auth_eval_consumer"
        assert consumer.CONSUMER_VERSION == "1.0.0"

    def test_consumer_validates_canonical_contract(self) -> None:
        from tac.cathedral.consumer_contract import validate_consumer_module
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        result = validate_consumer_module(consumer)
        assert result.contract_compliant is True
        assert result.consumer_tier.name == "TIER_A_OBSERVABILITY_ONLY"

    def test_consumer_hooks_canonical(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        from tac.cathedral.consumer_contract import HookNumber
        assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
        assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in consumer.CONSUMER_HOOK_NUMBERS
        assert HookNumber.PROBE_DISAMBIGUATOR in consumer.CONSUMER_HOOK_NUMBERS

    def test_consumer_unknown_metadata_returns_neutral(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        result = consumer.consume_candidate({"lane_id": "test"})
        assert result["readiness_verdict"] == "UNKNOWN"
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_consumer_paired_pass_verdict(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "PAIRED_PASS",
                "forbidden_macos_axis_detected": False,
                "cuda_score": 0.226,
                "cpu_score": 0.193,
                "cuda_cpu_gap": 0.033,
                "promotable": True,
                "archive_sha256_paired": "a" * 64,
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "PAIRED_PASS"
        assert "PAIRED_PASS" in result["rationale"]
        assert result["predicted_delta_adjustment"] == 0.0
        # Tier A: always promotable=False regardless of paired_pass
        assert result["promotable"] is False

    def test_consumer_macos_forbidden_verdict(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "BLOCKED_HARDWARE_NON_COMPLIANT",
                "forbidden_macos_axis_detected": True,
                "promotable": False,
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "FORBIDDEN_MACOS_AXIS"
        assert "Catalog #192" in result["rationale"]

    def test_consumer_partial_cuda_only(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "PAIRED_PARTIAL_CUDA_ONLY",
                "cuda_score": 0.226,
                "cpu_score": None,
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "PARTIAL_CUDA_ONLY"

    def test_consumer_partial_cpu_only(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "PAIRED_PARTIAL_CPU_ONLY",
                "cpu_score": 0.193,
                "cuda_score": None,
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "PARTIAL_CPU_ONLY"

    def test_consumer_blocked_pre_dispatch(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "BLOCKED_PRE_DISPATCH",
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "BLOCKED_PRE_DISPATCH"

    def test_consumer_blocked_harvest(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "BLOCKED_HARVEST",
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "BLOCKED_HARVEST"

    def test_consumer_blocked_axis_mismatch(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        candidate = {
            "paired_auth_eval_verdict": {
                "verdict": "BLOCKED_AXIS_MISMATCH",
            }
        }
        result = consumer.consume_candidate(candidate)
        assert result["readiness_verdict"] == "BLOCKED_AXIS_MISMATCH"
        assert "Catalog #127" in result["rationale"]

    def test_consumer_update_from_anchor_noop(self) -> None:
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        # Phase 7 scope: observability-only; update_from_anchor is a no-op
        consumer.update_from_anchor({"any": "anchor"})  # MUST NOT raise


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


class TestCLISubprocess:
    """tools/paired_auth_eval_cli.py CLI surface."""

    def test_cli_help_exits_zero(self) -> None:
        cli = REPO_ROOT / "tools" / "paired_auth_eval_cli.py"
        proc = subprocess.run(
            [sys.executable, str(cli), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env={**dict(__import__("os").environ), "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream:{REPO_ROOT}"},
        )
        assert proc.returncode == 0
        assert "paired" in proc.stdout.lower() or "paired" in proc.stderr.lower()

    def test_cli_missing_arg_exits_nonzero(self) -> None:
        cli = REPO_ROOT / "tools" / "paired_auth_eval_cli.py"
        proc = subprocess.run(
            [sys.executable, str(cli)],
            capture_output=True,
            text=True,
            check=False,
            env={**dict(__import__("os").environ), "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream:{REPO_ROOT}"},
        )
        assert proc.returncode != 0

    def test_cli_bad_bundle_path_exits_cli_error(self, tmp_path: Path) -> None:
        cli = REPO_ROOT / "tools" / "paired_auth_eval_cli.py"
        proc = subprocess.run(
            [sys.executable, str(cli), "--from-submission-bundle", "/nonexistent/bundle.json"],
            capture_output=True,
            text=True,
            check=False,
            env={**dict(__import__("os").environ), "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream:{REPO_ROOT}"},
        )
        # EXIT_CLI_ERROR = 6
        assert proc.returncode == 6

    def test_cli_dry_run_with_canonical_bundle_emits_verdict(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        bundle_json_path = tmp_path / "bundle.json"
        bundle_json_path.write_text(json.dumps(bundle.as_dict()))
        cli = REPO_ROOT / "tools" / "paired_auth_eval_cli.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--from-submission-bundle",
                str(bundle_json_path),
                "--cost-band",
                "smoke",
                "--cuda-gpu",
                "T4",
                "--cpu-target",
                "linux_x86_64_modal",
                "--dry-run",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**dict(__import__("os").environ), "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream:{REPO_ROOT}"},
        )
        # BLOCKED_PRE_DISPATCH dry-run -> exit code 1
        assert proc.returncode == 1
        # Verify JSON output is parseable
        data = json.loads(proc.stdout)
        assert data["verdict"] == "BLOCKED_PRE_DISPATCH"
        assert data["dry_run"] is True
        assert data["cuda_hardware_substrate"] == "linux_x86_64_modal_t4"

    def test_cli_execute_without_operator_approved_rejected(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        bundle_json_path = tmp_path / "bundle.json"
        bundle_json_path.write_text(json.dumps(bundle.as_dict()))
        cli = REPO_ROOT / "tools" / "paired_auth_eval_cli.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--from-submission-bundle",
                str(bundle_json_path),
                "--execute",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**dict(__import__("os").environ), "PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream:{REPO_ROOT}"},
        )
        # Should fail at the "--execute requires --operator-approved" check
        assert proc.returncode == 6
        assert "operator-approved" in proc.stderr.lower() or "operator_approved" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Catalog #192 macOS-CPU regression guards (10th apples-to-apples)
# ---------------------------------------------------------------------------


class TestCatalog192MacosRefusal:
    """Catalog #192 + CLAUDE.md 'Submission auth eval BOTH CPU AND CUDA' regression guards."""

    def test_darwin_arm64_cpu_target_yields_forbidden_axis(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cpu_target="darwin_arm64_advisory",
            dry_run=True,
        )
        assert verdict.forbidden_macos_axis_detected is True
        assert verdict.promotable is False
        assert verdict.score_claim is False
        assert verdict.cpu_axis_tag == "[macOS-CPU advisory]"
        assert verdict.evidence_grade == _EVIDENCE_GRADE_PAIRED_MACOS_ADVISORY

    def test_canonical_linux_x86_64_targets_not_macos(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        for cpu_target in (
            "linux_x86_64_modal",
            "linux_x86_64_vastai",
            "linux_x86_64_lightning",
            "linux_x86_64_gha",
        ):
            verdict = plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cpu_target=cpu_target,
                dry_run=True,
            )
            assert verdict.forbidden_macos_axis_detected is False, (
                f"cpu_target {cpu_target} unexpectedly flagged as macOS"
            )


# ---------------------------------------------------------------------------
# Catalog #341 Tier A canonical-routing markers
# ---------------------------------------------------------------------------


class TestCatalog341TierARoutingMarkers:
    """Catalog #341 Tier A canonical-routing markers enforced."""

    def test_default_provenance_tier_a_canonical(self) -> None:
        prov = derive_paired_auth_eval_provenance(
            lane_id="lane_test",
            substrate_id="sub_test",
            archive_sha256="a" * 64,
            measurement_utc=_utc_now_iso(),
            cuda_platform="modal",
            cuda_gpu="T4",
            cpu_target="linux_x86_64_modal",
        )
        # Default Tier A markers
        assert prov["axis_tag"] == "[predicted]"
        assert prov["score_claim"] is False
        assert prov["promotable"] is False

    def test_dry_run_verdict_always_tier_a(self, tmp_path: Path) -> None:
        bundle = _make_canonical_bundle(tmp_path=tmp_path)
        verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cpu_target="linux_x86_64_modal",
            dry_run=True,
        )
        assert verdict.axis_tag == "[predicted]"
        assert verdict.score_claim is False
        assert verdict.promotable is False


# ---------------------------------------------------------------------------
# Live-repo regression guards (Catalog #185 sister)
# ---------------------------------------------------------------------------


class TestLiveRepoRegressionGuards:
    """Live-repo regression guards verify Phase 7 importability stays clean."""

    def test_phase_7_module_importable_via_package(self) -> None:
        from tac.submission_packet import (
            PAIRED_AUTH_EVAL_SCHEMA_VERSION,
            PHASE_7_LAYER_VERSION,
            PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID,
            PairedAuthEvalError,
            PairedAuthEvalVerdict,
            PairedAuthEvalVerdictKind,
            derive_paired_auth_eval_provenance,
            plan_paired_auth_eval,
            reconstruct_verdict_from_disk,
        )
        # Just verify symbols import successfully
        assert PAIRED_AUTH_EVAL_SCHEMA_VERSION is not None

    def test_phase_7_in_package_all(self) -> None:
        from tac import submission_packet as pkg
        assert "PAIRED_AUTH_EVAL_SCHEMA_VERSION" in pkg.__all__
        assert "PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID" in pkg.__all__
        assert "PHASE_7_LAYER_VERSION" in pkg.__all__
        assert "PairedAuthEvalVerdict" in pkg.__all__
        assert "PairedAuthEvalVerdictKind" in pkg.__all__
        assert "plan_paired_auth_eval" in pkg.__all__
        assert "reconstruct_verdict_from_disk" in pkg.__all__

    def test_phase_7_cathedral_consumer_auto_discoverable(self) -> None:
        """Phase 7 cathedral consumer is auto-discoverable per Catalog #335."""
        from tac.cathedral.consumer_contract import validate_consumer_module
        import tac.cathedral_consumers.paired_auth_eval_consumer as consumer
        result = validate_consumer_module(consumer)
        assert result.contract_compliant is True
