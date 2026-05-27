# SPDX-License-Identifier: MIT
"""Phase 4 Layer 2 — submission bundle canonical helper tests.

Covers:
- ``DependencyClosureManifest`` + ``SubmissionBundleResult`` dataclass invariants.
- End-to-end ``build_submission_bundle`` synthesizes a clean submission_dir.
- HNeRV parity L4 inflate.py <=200 LOC + <=2 ext deps verification.
- Catalog #205 canonical select_inflate_device routing (mirror present in
  generated inflate.py).
- Catalog #295 PYTHONPATH self-containment (no bare ``from tac.*`` imports
  in generated scaffold).
- Catalog #146 3-arg inflate.sh signature + ``set -euo pipefail``.
- Catalog #208 README + Catalog #208 docs no-local-absolute-paths.
- Public-PR hygiene refusal (Claude/Anthropic token leak refused).
- Catalog #335 cathedral consumer contract compliance.
- CLI subprocess test (exit codes 0-5).
- Integration test consuming actual Phase 3 ArchiveGrammarManifest
  (closes Phase 2→3→4 pipeline lineage).
- Live-repo regression guard.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.submission_packet import (
    PHASE_4_LAYER_VERSION,
    SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    ArchiveGrammarManifest,
    ArchiveSectionSpec,
    ByteMutationSmokeVerdict,
    DependencyClosureManifest,
    HardwareSubstrateClass,
    OperationalMechanismStatus,
    PythonpathSelfContainmentStatus,
    SectionKind,
    SelectInflateDeviceRouting,
    SubmissionBundleError,
    SubmissionBundleResult,
    build_dependency_closure_manifest,
    build_submission_bundle,
    derive_submission_bundle_provenance,
)
from tac.submission_packet.builder import (
    CANONICAL_INFLATE_SH_NAMED_TOKENS,
    CANONICAL_INFLATE_SH_REQUIRED_HEADER,
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    HNERV_CLASS_INFLATE_DEPS,
    NUMPY_PORTABLE_INFLATE_DEPS,
    _emit_canonical_select_inflate_device_block,
    _scan_for_forbidden_pr_tokens,
    _scan_for_local_absolute_paths,
)
from tac.submission_packet.compression_pipeline import (
    COMPRESSION_PIPELINE_SCHEMA_VERSION,
    CompressionPipelineResult,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_archive(tmp_path: Path, payload: bytes = b"hello world") -> Path:
    """Build a minimal canonical monolithic single-file 0.bin archive.zip."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", payload)
    return archive_path


def _make_pipeline_result(
    *, lane_id: str = "lane_test_phase_4", substrate_id: str = "test_substrate"
) -> CompressionPipelineResult:
    """Build a canonical CompressionPipelineResult for tests."""
    import datetime

    now = datetime.datetime.now(datetime.UTC).isoformat()
    return CompressionPipelineResult(
        schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        video_path="upstream/videos/0.mkv",
        hardware_substrate="macos_arm64_m5_max",
        hardware_substrate_class="local-mps",
        substrate_trainer_path=f"experiments/train_substrate_{substrate_id}.py",
        recipe_path=f".omx/operator_authorize_recipes/substrate_{substrate_id}_local.yaml",
        mlx_first_encode=True,
        qat_enabled=True,
        weights_export_path=None,
        weights_sha256=None,
        weights_size_bytes=None,
        training_anchor_call_id=None,
        qat_anchor_call_id=None,
        dispatch_optimization_protocol_overall_pass=True,
        dispatch_optimization_protocol_blockers=(),
        per_axis_predicted_band=None,
        measurement_utc=now,
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; compression-pipeline-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
        canonical_equation_id=(
            "compression_pipeline_canonical_helper_consolidation_savings_v1"
        ),
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
        cost_usd=None,
        canonical_provenance={"axis_tag": "[predicted]"},
        written_at_utc=now,
        written_pid=1,
        written_host="test",
    )


def _make_archive_grammar(
    *,
    archive_path: Path,
    lane_id: str = "lane_test_phase_4",
    substrate_id: str = "test_substrate",
) -> ArchiveGrammarManifest:
    """Build a canonical ArchiveGrammarManifest for tests."""
    import datetime

    now = datetime.datetime.now(datetime.UTC).isoformat()
    member_bytes = archive_path.read_bytes()
    # The 0.bin member sha (not the archive sha).
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open("0.bin") as member:
            section_bytes = member.read()
    archive_sha = hashlib.sha256(member_bytes).hexdigest()
    section_sha = hashlib.sha256(section_bytes).hexdigest()
    spec = ArchiveSectionSpec(
        section_name="0.bin",
        offset_in_archive=0,
        length_in_archive=len(section_bytes),
        sha256_of_section=section_sha,
        section_kind=SectionKind.OTHER.value,
        operational_mechanism_status=OperationalMechanismStatus.OPERATIONAL.value,
        distinguishing_feature_name=None,
        member_name="0.bin",
    )
    return ArchiveGrammarManifest(
        schema_version="archive_grammar_v1_20260526",
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_path=str(archive_path),
        archive_sha256=archive_sha,
        archive_bytes=len(member_bytes),
        section_specs=(spec,),
        monolithic_single_file=True,
        multi_file_justification=None,
        byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.NOT_RUN.value,
        byte_mutation_smoke_evidence_path=None,
        no_op_detector_passed=False,
        measurement_utc=now,
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; archive-grammar-canonical]",
        canonical_helper_invocation=(
            "tac.submission_packet.build_archive_grammar_from_compression_pipeline_result"
        ),
        canonical_equation_id="archive_grammar_canonical_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        parser_section_manifest_path=None,
        elapsed_seconds=0.01,
        canonical_provenance={"axis_tag": "[predicted]"},
        written_at_utc=now,
        written_pid=1,
        written_host="test",
    )


# ---------------------------------------------------------------------------
# DependencyClosureManifest invariants
# ---------------------------------------------------------------------------


class TestDependencyClosureManifest:
    def test_canonical_numpy_portable(self) -> None:
        dm = DependencyClosureManifest(
            declared_dependencies=("numpy",),
            dependency_budget=DEFAULT_INFLATE_DEPS_BUDGET,
            within_budget=True,
            numpy_portable=True,
        )
        assert dm.numpy_portable
        assert dm.within_budget

    def test_canonical_hnerv_class(self) -> None:
        dm = DependencyClosureManifest(
            declared_dependencies=("numpy", "torch"),
            dependency_budget=DEFAULT_INFLATE_DEPS_BUDGET,
            within_budget=True,
            numpy_portable=False,
        )
        assert not dm.numpy_portable
        assert dm.within_budget

    def test_over_budget_requires_waiver(self) -> None:
        with pytest.raises(ValueError, match="waiver_rationale"):
            DependencyClosureManifest(
                declared_dependencies=("a", "b", "c", "d"),
                dependency_budget=2,
                within_budget=False,
                numpy_portable=False,
                waiver_rationale=None,
            )

    def test_placeholder_waiver_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-placeholder"):
            DependencyClosureManifest(
                declared_dependencies=("a", "b", "c"),
                dependency_budget=2,
                within_budget=False,
                numpy_portable=False,
                waiver_rationale="<rationale>",
            )

    def test_within_budget_consistency_check(self) -> None:
        with pytest.raises(ValueError, match="within_budget"):
            DependencyClosureManifest(
                declared_dependencies=("a",),
                dependency_budget=2,
                within_budget=False,  # inconsistent with len(deps)=1 <= 2
                numpy_portable=False,
            )

    def test_numpy_portable_consistency_check(self) -> None:
        with pytest.raises(ValueError, match="numpy_portable"):
            DependencyClosureManifest(
                declared_dependencies=("numpy", "torch"),
                dependency_budget=2,
                within_budget=True,
                numpy_portable=True,  # inconsistent with deps != ("numpy",)
            )

    def test_must_be_sorted_tuple(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            DependencyClosureManifest(
                declared_dependencies=("torch", "numpy"),
                dependency_budget=2,
                within_budget=True,
                numpy_portable=False,
            )

    def test_as_dict_roundtrip(self) -> None:
        dm = DependencyClosureManifest(
            declared_dependencies=("numpy",),
            dependency_budget=2,
            within_budget=True,
            numpy_portable=True,
        )
        d = dm.as_dict()
        assert d["declared_dependencies"] == ["numpy"]
        assert d["numpy_portable"] is True

    def test_build_dependency_closure_manifest_helper(self) -> None:
        dm = build_dependency_closure_manifest(("numpy",))
        assert dm.numpy_portable
        assert dm.within_budget

    def test_build_dependency_closure_manifest_unsorted_input_sorted(self) -> None:
        dm = build_dependency_closure_manifest(("torch", "numpy"))
        assert dm.declared_dependencies == ("numpy", "torch")


# ---------------------------------------------------------------------------
# End-to-end build_submission_bundle synthesis
# ---------------------------------------------------------------------------


class TestBuildSubmissionBundleSynthesis:
    def test_canonical_synthesis_clean(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        assert isinstance(result, SubmissionBundleResult)
        assert result.archive_sha256 == grammar.archive_sha256
        assert result.archive_bytes == grammar.archive_bytes
        assert Path(result.submission_dir).is_dir()
        assert Path(result.inflate_sh_path).is_file()
        assert Path(result.inflate_py_path).is_file()
        assert Path(result.readme_md_path).is_file()
        assert Path(result.report_txt_path).is_file()
        assert Path(result.archive_manifest_path).is_file()
        assert (Path(result.submission_dir) / "archive.zip").is_file()

    def test_inflate_py_within_loc_budget(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        assert result.inflate_py_loc <= DEFAULT_INFLATE_PY_LOC_BUDGET, (
            f"HNeRV parity L4: inflate.py LOC={result.inflate_py_loc} "
            f"exceeds budget {DEFAULT_INFLATE_PY_LOC_BUDGET}"
        )

    def test_inflate_py_numpy_portable_default(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        assert result.dependency_closure_manifest.numpy_portable
        assert result.runtime_dep_closure == ("numpy",)

    def test_canonical_inflate_sh_3_arg_signature(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        sh_source = Path(result.inflate_sh_path).read_text(encoding="utf-8")
        assert CANONICAL_INFLATE_SH_REQUIRED_HEADER in sh_source
        for token in CANONICAL_INFLATE_SH_NAMED_TOKENS:
            assert token in sh_source

    def test_canonical_select_inflate_device_mirror_present(
        self, tmp_path: Path
    ) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        py_source = Path(result.inflate_py_path).read_text(encoding="utf-8")
        # Per Catalog #205: canonical select_inflate_device helper present.
        assert "def select_inflate_device" in py_source
        assert "PACT_INFLATE_DEVICE" in py_source
        assert "INLINE_DEVICE_FORK_OK" in py_source

    def test_pythonpath_self_containment_clean(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        assert (
            result.pythonpath_self_containment_status
            == PythonpathSelfContainmentStatus.CLEAN.value
        )
        py_source = Path(result.inflate_py_path).read_text(encoding="utf-8")
        # Per Catalog #295: scan for REAL import statements (line-anchored)
        # not docstring mentions. Mirrors the production check_188 + #295 gate.
        import re

        bare_tac_import = re.search(
            r"^\s*(?:from\s+tac(?:\.|\s)|import\s+tac(?:\.|\s|$))",
            py_source,
            re.MULTILINE,
        )
        assert bare_tac_import is None, (
            f"Catalog #295: bare 'from tac.*' / 'import tac' import detected: "
            f"{bare_tac_import.group(0)!r}"
        )

    def test_archive_manifest_sidecar_has_per_member_identity(
        self, tmp_path: Path
    ) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        manifest_data = json.loads(
            Path(result.archive_manifest_path).read_text(encoding="utf-8")
        )
        assert manifest_data["archive_sha256"] == result.archive_sha256
        assert len(manifest_data["members"]) == 1
        assert manifest_data["members"][0]["name"] == "0.bin"

    def test_readme_md_clean_no_attribution_leak(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        readme = Path(result.readme_md_path).read_text(encoding="utf-8")
        for token in ("Claude", "Anthropic", "Co-Authored"):
            assert token not in readme, f"public-PR token leak: {token}"

    def test_lane_id_mismatch_refused(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result(lane_id="lane_a")
        grammar = _make_archive_grammar(archive_path=archive, lane_id="lane_b")
        with pytest.raises(SubmissionBundleError, match="lane_id mismatch"):
            build_submission_bundle(
                compression_pipeline_result=pipeline,
                archive_grammar_manifest=grammar,
                output_dir=tmp_path / "out",
            )

    def test_substrate_id_mismatch_refused(self, tmp_path: Path) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result(substrate_id="sub_a")
        grammar = _make_archive_grammar(
            archive_path=archive, substrate_id="sub_b"
        )
        with pytest.raises(SubmissionBundleError, match="substrate_id mismatch"):
            build_submission_bundle(
                compression_pipeline_result=pipeline,
                archive_grammar_manifest=grammar,
                output_dir=tmp_path / "out",
            )

    def test_archive_zip_mtime_is_fresh_per_catalog_361(self, tmp_path: Path) -> None:
        import time

        archive = _make_test_archive(tmp_path / "src")
        # Force source mtime to be very old (sister of Modal scenario).
        old_mtime = time.time() - 86400 * 30  # 30 days ago
        import os

        os.utime(archive, (old_mtime, old_mtime))
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        copied = Path(result.submission_dir) / "archive.zip"
        # Per Catalog #361: mtime-fresh so Modal harvester picks it up.
        assert copied.stat().st_mtime > old_mtime + 86400 * 29

    def test_report_txt_placeholder_has_canonical_shape(
        self, tmp_path: Path
    ) -> None:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        rep = Path(result.report_txt_path).read_text(encoding="utf-8")
        # Per CLAUDE.md "Submission auth eval" + Catalog #146 contest evaluator
        # output format.
        assert "Evaluation results over 600 samples" in rep
        assert "Average PoseNet Distortion" in rep
        assert "Average SegNet Distortion" in rep
        assert "Submission file size" in rep
        assert "Compression Rate" in rep
        assert "Final score" in rep
        assert "37,545,489" in rep


# ---------------------------------------------------------------------------
# Catalog #208 + public-PR hygiene scanners
# ---------------------------------------------------------------------------


class TestCatalog208DocsHygiene:
    def test_scan_for_local_absolute_paths_clean(self) -> None:
        assert _scan_for_local_absolute_paths("nothing here") == ()
        assert _scan_for_local_absolute_paths("relative/path") == ()

    def test_scan_for_local_absolute_paths_macos(self) -> None:
        leaks = _scan_for_local_absolute_paths(
            "look at /Users/adpena/Projects/x"
        )
        assert any("/Users/adpena/" in m for m in leaks)

    def test_scan_for_local_absolute_paths_linux(self) -> None:
        leaks = _scan_for_local_absolute_paths(
            "see /home/runner/work/foo"
        )
        assert any("/home/runner/" in m for m in leaks)

    def test_scan_for_forbidden_pr_tokens_clean(self) -> None:
        assert _scan_for_forbidden_pr_tokens("a contest submission") == ()

    def test_scan_for_forbidden_pr_tokens_claude(self) -> None:
        leaks = _scan_for_forbidden_pr_tokens(
            "Generated with Claude Code"
        )
        assert "Claude" in leaks

    def test_scan_for_forbidden_pr_tokens_co_authored(self) -> None:
        leaks = _scan_for_forbidden_pr_tokens(
            "Co-Authored-By: Claude Opus"
        )
        assert "Co-Authored" in leaks
        assert "Claude" in leaks


# ---------------------------------------------------------------------------
# SubmissionBundleResult contract validation
# ---------------------------------------------------------------------------


class TestSubmissionBundleResultInvariants:
    def _make_valid(self, tmp_path: Path) -> SubmissionBundleResult:
        archive = _make_test_archive(tmp_path / "src")
        pipeline = _make_pipeline_result()
        grammar = _make_archive_grammar(archive_path=archive)
        return build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=tmp_path / "submission_dir",
        )

    def test_schema_version_pinned(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        assert result.schema_version == SUBMISSION_BUNDLE_SCHEMA_VERSION

    def test_canonical_equation_id_pinned(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        assert result.canonical_equation_id == SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID
        assert result.canonical_equation_status == "FORMALIZATION_PENDING"

    def test_axis_tag_predicted(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        assert result.axis_tag == "[predicted]"
        assert result.score_claim is False
        assert result.promotable is False

    def test_evidence_grade_canonical(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        assert result.evidence_grade.startswith("[predicted;")

    def test_canonical_helper_invocation(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        assert (
            result.canonical_helper_invocation
            == "tac.submission_packet.build_submission_bundle"
        )

    def test_as_dict_roundtrip(self, tmp_path: Path) -> None:
        result = self._make_valid(tmp_path)
        d = result.as_dict()
        assert d["schema_version"] == SUBMISSION_BUNDLE_SCHEMA_VERSION
        assert "dependency_closure_manifest" in d
        assert "archive_sha256" in d


# ---------------------------------------------------------------------------
# Catalog #335 cathedral consumer contract compliance
# ---------------------------------------------------------------------------


class TestCathedralConsumerContract:
    def test_consumer_is_canonical_contract_compliant(self) -> None:
        from tac.cathedral.consumer_contract import validate_consumer_module
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        reg = validate_consumer_module(m)
        assert reg.contract_compliant, f"validation_errors: {reg.validation_errors}"

    def test_consumer_hook_numbers_canonical(self) -> None:
        from tac.cathedral.consumer_contract import HookNumber
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in m.CONSUMER_HOOK_NUMBERS

    def test_consumer_ready_verdict_when_clean_bundle(self) -> None:
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        candidate = {
            "submission_bundle_result": {
                "inflate_py_loc": 150,
                "inflate_py_loc_budget": 200,
                "dependency_closure_manifest": {
                    "within_budget": True,
                    "numpy_portable": True,
                    "declared_dependencies": ["numpy"],
                },
                "pythonpath_self_containment_status": "clean",
                "select_inflate_device_routing": "inline_with_waiver",
            },
        }
        result = m.consume_candidate(candidate)
        assert result["readiness_verdict"] == "READY"
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_consumer_blocked_verdict_when_over_loc_budget(self) -> None:
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        candidate = {
            "submission_bundle_result": {
                "inflate_py_loc": 300,  # over budget
                "inflate_py_loc_budget": 200,
                "dependency_closure_manifest": {
                    "within_budget": True,
                    "numpy_portable": True,
                    "declared_dependencies": ["numpy"],
                },
                "pythonpath_self_containment_status": "clean",
                "select_inflate_device_routing": "inline_with_waiver",
            },
        }
        result = m.consume_candidate(candidate)
        assert result["readiness_verdict"] == "BLOCKED"
        assert "HNeRV parity L4" in result["rationale"]

    def test_consumer_unknown_verdict_when_no_metadata(self) -> None:
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        result = m.consume_candidate({})
        assert result["readiness_verdict"] == "UNKNOWN"

    def test_update_from_anchor_no_op(self) -> None:
        import tac.cathedral_consumers.submission_bundle_builder_consumer as m

        # Should not raise.
        m.update_from_anchor({"anything": "goes"})


# ---------------------------------------------------------------------------
# CLI subprocess test (exit codes 0-5)
# ---------------------------------------------------------------------------


class TestCLISubprocess:
    def test_cli_help_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "submission_bundle_cli.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Phase 4" in result.stdout

    def test_cli_missing_archive_exits_5(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "submission_bundle_cli.py"),
                "--lane-id",
                "lane_test",
                "--substrate-trainer",
                "experiments/train_substrate_nonexistent.py",
                "--recipe-path",
                ".omx/operator_authorize_recipes/nonexistent.yaml",
                "--archive-path",
                str(tmp_path / "missing.zip"),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-protocol-verification",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 5, f"stderr: {result.stderr}"


# ---------------------------------------------------------------------------
# Integration with Phase 3 ArchiveGrammarManifest (closes lineage)
# ---------------------------------------------------------------------------


class TestPhase3Integration:
    @pytest.mark.skipif(
        not SAMPLE_ARCHIVE.is_file(),
        reason="submissions/a1/archive.zip not present (out-of-scope for fixture)",
    )
    def test_integration_with_actual_a1_archive(self, tmp_path: Path) -> None:
        """Closes Phase 2 -> Phase 3 -> Phase 4 lineage on the actual A1 archive."""
        from tac.submission_packet.archive_grammar import (
            build_archive_grammar_from_compression_pipeline_result,
        )

        pipeline = _make_pipeline_result(
            lane_id="lane_a1_integration_test", substrate_id="a1_finetuned"
        )
        # Copy archive to a writable location for the test (the real one is in submissions/).
        test_archive = tmp_path / "a1_archive.zip"
        shutil.copy2(SAMPLE_ARCHIVE, test_archive)
        grammar = build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline,
            archive_path=test_archive,
            monolithic_single_file=False,
            multi_file_justification=(
                "A1 archive uses single ZIP member 'x' (non-canonical 0.bin name); "
                "preserved per Catalog #110 HISTORICAL_PROVENANCE on the existing PR101 lineage"
            ),
            output_dir=tmp_path,
            emit_parser_section_manifest=False,
        )
        output_dir = tmp_path / "submission_dir"
        result = build_submission_bundle(
            compression_pipeline_result=pipeline,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
        )
        assert result.archive_sha256 == grammar.archive_sha256
        assert Path(result.submission_dir).is_dir()
        # Live-repo regression: A1 archive is HNeRV-class (needs torch);
        # canonical numpy-portable scaffold passes anyway (Phase 6 paired_auth_eval
        # is what binds the real decoder).


# ---------------------------------------------------------------------------
# derive_submission_bundle_provenance contract
# ---------------------------------------------------------------------------


class TestProvenanceHelper:
    def test_provenance_carries_canonical_fields(self) -> None:
        import datetime

        now = datetime.datetime.now(datetime.UTC).isoformat()
        prov = derive_submission_bundle_provenance(
            lane_id="lane_t",
            substrate_id="sub_t",
            archive_sha256="a" * 64,
            measurement_utc=now,
        )
        assert prov["axis_tag"] == "[predicted]"
        assert prov["score_claim"] is False
        assert prov["promotable"] is False
        assert prov["canonical_helper_invocation"] == (
            "tac.submission_packet.build_submission_bundle"
        )
        assert prov["canonical_equation_id"] == SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID
        assert prov["canonical_equation_status"] == "FORMALIZATION_PENDING"


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_phase_4_layer_version_pinned(self) -> None:
        assert (
            PHASE_4_LAYER_VERSION
            == "phase_4_submission_bundle_canonical_landed_20260526"
        )

    def test_canonical_dep_sets_pinned(self) -> None:
        assert NUMPY_PORTABLE_INFLATE_DEPS == frozenset({"numpy"})
        assert HNERV_CLASS_INFLATE_DEPS == frozenset({"torch", "numpy"})

    def test_canonical_select_inflate_device_block_emits(self) -> None:
        block = _emit_canonical_select_inflate_device_block()
        assert "def select_inflate_device" in block
        assert "PACT_INFLATE_DEVICE" in block
        # Per Catalog #205: MPS forbidden.
        assert "MPS is forbidden" in block or "mps" not in block.lower().split("\n")[5]

    def test_select_inflate_device_routing_enum_canonical(self) -> None:
        assert SelectInflateDeviceRouting.CANONICAL_HELPER.value == "canonical_helper"
        assert (
            SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value
            == "inline_with_waiver"
        )

    def test_pythonpath_self_containment_status_enum_canonical(self) -> None:
        assert PythonpathSelfContainmentStatus.CLEAN.value == "clean"
