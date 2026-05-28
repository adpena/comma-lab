# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

import comma_lab.scheduler.frontier_rate_attack_bootstrap as frontier_bootstrap
from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    MATERIALIZER_EXACT_READINESS_BRIDGE_STEP_ID,
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_HARVEST_STEP_ID,
    MATERIALIZER_SUBMISSION_CLOSURE_STEP_ID,
)
from comma_lab.scheduler.byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    ARCHIVE_ZIP_REPACK_TARGET_KIND,
    DQS1_PAIRSET_TARGET_KIND,
    FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND,
    FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from comma_lab.scheduler.frontier_final_rate_attack_autoloop import (
    POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA,
    POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA,
    POST_FEEDBACK_PORTFOLIO_COVERAGE_PREFLIGHT_SCHEMA,
    execute_post_feedback_child_queues,
    select_post_feedback_child_queue_artifacts,
)
from comma_lab.scheduler.frontier_rate_attack_bootstrap import (
    ARCHIVE_RATE_ATTACK_SUPPORTED_TARGET_KINDS,
    BOOTSTRAP_SCHEMA,
    DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA,
    DERIVED_SECTION_MANIFEST_BATCH_SCHEMA,
    TARGET_COVERAGE_SCHEMA,
    FrontierRateAttackBootstrapError,
    archive_record,
    build_frontier_rate_attack_payloads,
    derive_archive_section_recode_manifests,
    derive_packet_member_merge_contract,
    parse_archive_spec,
    resolve_current_frontier_archive,
)
from comma_lab.scheduler.frontier_rate_attack_target_profile import (
    TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA,
    build_frontier_target_optimization_profile,
)
from comma_lab.scheduler.json_identity import stable_json_sha256
from tac.optimization.family_agnostic_materializers import (
    RENDERER_PAYLOAD_DFL1_MEMBER_NAMES,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
)
from tac.repo_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
QUEUE_BUILDER_TOOL = REPO_ROOT / "tools" / "build_frontier_final_rate_attack_queue.py"


AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _write_archive(path: Path, *, member_name: str = "renderer.bin") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"frontier-bytes")
    return path


def _write_robust_like_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("renderer.bin", b"renderer-payload" * 32)
        archive.writestr("masks.mkv", b"mask-payload" * 24)
        archive.writestr("optimized_poses.pt", b"pose-payload" * 16)
    return path


def _write_stored_archive(path: Path, *, member_name: str, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)
    return path


def _write_fp11_feca_submission(path: Path, *, marker_prefix_padding: int = 0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inflate.py").write_text("# runtime placeholder\n", encoding="utf-8")
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    encoder = path / "encoder"
    encoder.mkdir(parents=True, exist_ok=True)
    (encoder / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py").write_text(
        "# encoder placeholder\n",
        encoding="utf-8",
    )
    payload = (
        b"FP11"
        + (3).to_bytes(4, "little")
        + b"src"
        + (8).to_bytes(2, "little")
        + (b"x" * marker_prefix_padding)
        + b"FECaDATA"
        + b"dqs1"
    )
    return _write_stored_archive(path / "archive.zip", member_name="p", payload=payload)


def _write_fp11_source_brotli_recode_submission(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inflate.py").write_text("# runtime placeholder\n", encoding="utf-8")
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    src = path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "codec.py").write_text("DECODER_BLOB_LEN = 2\n", encoding="utf-8")
    payload = (
        b"FP11"
        + (3).to_bytes(4, "little")
        + b"abc"
        + (8).to_bytes(2, "little")
        + b"FECaDATA"
        + b"dqs1"
    )
    return _write_stored_archive(path / "archive.zip", member_name="p", payload=payload)


def _load_frontier_queue_builder_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_frontier_final_rate_attack_queue_tool",
        QUEUE_BUILDER_TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_portfolio_coverage_payload() -> dict[str, object]:
    return {
        "schema": "frontier_rate_attack_portfolio_coverage.v1",
        "queue_id": "feedback-refresh",
        "pathway_count": 5,
        "bound_pathway_count": 2,
        "rows": [],
        "deferred_registry_target_bindings": [
            {
                "target_kind": DQS1_PAIRSET_TARGET_KIND,
                "pathway_id": "dqs1_local_first_feedback",
                "binding_status": "bound",
                "artifact_paths": ["dqs1_followup_queue.json"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "target_kind": INVERSE_SCORER_CELL_TARGET_KIND,
                "pathway_id": "inverse_steganalysis_acquisition",
                "binding_status": "bound",
                "artifact_paths": ["targeted_component_correction_queue.json"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ],
        "coverage_ready_for_bounded_local_followup": True,
        "allowed_use": "queue_owned_frontier_rate_attack_portfolio_planning_only",
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key in AUTHORITY_KEYS:
        assert payload[key] is False


def test_frontier_bootstrap_builds_queue_with_experiment_metadata(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_unit",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        ),
        include_optional_target_blockers=True,
        local_cpu_concurrency=2,
    )

    bootstrap = payloads["bootstrap"]
    queue = payloads["queue"]
    backlog = payloads["backlog"]
    contexts = payloads["contexts"]
    _assert_false_authority(record)
    _assert_false_authority(bootstrap)
    assert bootstrap["schema"] == BOOTSTRAP_SCHEMA
    assert bootstrap["executable_target_count"] == 2
    assert bootstrap["experiment_count"] == 2
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 2}
    assert backlog["backlog_row_count"] == 2
    assert len(contexts["rows"]) == 2

    omitted = {row["target_kind"]: row for row in bootstrap["target_omissions"]}
    assert ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND in omitted
    assert TENSOR_FACTORIZE_TARGET_KIND in omitted
    assert omitted[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["score_claim"] is False

    for experiment in queue["experiments"]:
        metadata = experiment["metadata"]["frontier_rate_attack_bootstrap"]
        assert metadata["schema"] == BOOTSTRAP_SCHEMA
        assert metadata["archive_labels"] == ["frontier"]
        assert metadata["target_kinds"] == [
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        ]
        assert metadata["score_claim"] is False
        assert metadata["ready_for_exact_eval_dispatch"] is False
        assert metadata["exact_readiness_followup_requested"] is False


def test_frontier_bootstrap_default_targets_include_archive_zip_repack(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_default_targets",
        archive_records=[record],
        results_root=tmp_path / "results",
        include_optional_target_blockers=False,
    )

    assert ARCHIVE_ZIP_REPACK_TARGET_KIND in payloads["bootstrap"]["executable_target_kinds"]


def test_frontier_bootstrap_target_coverage_accounts_for_registry_executables(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_target_coverage",
        archive_records=[record],
        results_root=tmp_path / "results",
        include_optional_target_blockers=True,
    )

    coverage = payloads["target_coverage"]
    assert coverage["schema"] == TARGET_COVERAGE_SCHEMA
    assert coverage["coverage_complete"] is True
    assert coverage["unclassified_executable_candidate_target_kinds"] == []
    assert coverage["archive_rate_supported_target_kinds"] == list(
        ARCHIVE_RATE_ATTACK_SUPPORTED_TARGET_KINDS
    )
    assert ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND in coverage[
        "context_omitted_target_kinds"
    ]
    assert TENSOR_FACTORIZE_TARGET_KIND in coverage["context_omitted_target_kinds"]
    assert FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND in coverage[
        "context_omitted_target_kinds"
    ]
    deferred = {
        row["target_kind"]: row
        for row in coverage["deferred_registry_target_rows"]
    }
    assert DQS1_PAIRSET_TARGET_KIND in deferred
    assert deferred[DQS1_PAIRSET_TARGET_KIND]["deferred_to"] == (
        "dqs1_local_first_feedback_cycle"
    )
    assert INVERSE_SCORER_CELL_TARGET_KIND in deferred
    assert deferred[INVERSE_SCORER_CELL_TARGET_KIND]["deferred_to"] == (
        "inverse_steganalysis_acquisition_chain"
    )
    _assert_false_authority(coverage)


def test_frontier_bootstrap_binds_selector_context_recode_when_source_runtime_present(
    tmp_path: Path,
) -> None:
    archive_path = _write_fp11_feca_submission(tmp_path / "submission_dir")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_selector_context",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND,),
        include_optional_target_blockers=False,
        allow_overwrite=True,
    )

    row = payloads["contexts"]["rows"][0]
    context = row["context"]
    assert row["target_kind"] == FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND
    assert context["source_submission_dir"].endswith("submission_dir")
    assert context["output_dir"].endswith("feca_selector_context_recode")
    assert context["selector_codec_families"] == ["fec10_adaptive_blend"]
    assert context["downstream_materializer_targets"] == [ARCHIVE_ZIP_REPACK_TARGET_KIND]


def test_frontier_bootstrap_binds_selector_context_recode_when_feca_marker_is_late(
    tmp_path: Path,
) -> None:
    archive_path = _write_fp11_feca_submission(
        tmp_path / "submission_dir",
        marker_prefix_padding=5000,
    )
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_selector_context_late_marker",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND,),
        include_optional_target_blockers=False,
        allow_overwrite=True,
    )

    assert payloads["bootstrap"]["executable_target_kinds"] == [
        FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND
    ]
    assert payloads["bootstrap"]["target_omissions"] == []


def test_frontier_bootstrap_binds_fp11_source_brotli_recode_context(
    tmp_path: Path,
) -> None:
    archive_path = _write_fp11_source_brotli_recode_submission(tmp_path / "submission_dir")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_fp11_source_brotli",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,),
        include_optional_target_blockers=False,
        allow_overwrite=True,
    )

    row = payloads["contexts"]["rows"][0]
    context = row["context"]
    assert row["target_kind"] == FP11_SOURCE_BROTLI_RECODE_TARGET_KIND
    assert context["source_submission_dir"].endswith("submission_dir")
    assert context["output_dir"].endswith("fp11_source_brotli_recode")
    assert context["brotli_qualities"] == list(range(1, 12))
    assert context["brotli_lgwins"] == ["none", *list(range(16, 25))]
    assert payloads["bootstrap"]["target_omissions"] == []


def test_frontier_bootstrap_propagates_declared_video_scope_to_queue(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"declared contest target")
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    profile = build_frontier_target_optimization_profile(
        repo_root=tmp_path,
        target_profile_id="unit_contest_video",
        target_mode="contest_video_overfit",
        target_video_paths=("upstream/videos/0.mkv",),
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_target_scope",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
        include_optional_target_blockers=False,
        member_name="payload.bin",
        target_optimization_profile=profile,
        require_target_profile_ready=True,
    )

    target_metadata = payloads["bootstrap"]["target_optimization_profile_metadata"]
    assert target_metadata["schema"] == TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA
    assert target_metadata["target_profile_id"] == "unit_contest_video"
    assert target_metadata["target_mode"] == "contest_video_overfit"
    assert target_metadata["declared_overfit_allowed"] is True
    assert target_metadata["target_video_paths"] == ["upstream/videos/0.mkv"]
    assert payloads["contexts"]["target_optimization_profile"] == target_metadata
    assert payloads["contexts"]["rows"][0]["context"][
        "target_optimization_profile"
    ] == target_metadata
    assert payloads["backlog"]["rows"][0]["target_profile_id"] == "unit_contest_video"
    experiment_metadata = payloads["queue"]["experiments"][0]["metadata"]
    assert (
        experiment_metadata["frontier_target_optimization_profile"]
        == target_metadata
    )
    _assert_false_authority(target_metadata)


def test_frontier_bootstrap_refuses_unready_required_target_profile(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    profile = build_frontier_target_optimization_profile(
        repo_root=tmp_path,
        target_profile_id="unit_corpus_missing",
        target_mode="corpus_generalization",
        target_video_paths=("upstream/videos/0.mkv",),
    )

    with pytest.raises(
        FrontierRateAttackBootstrapError,
        match="target optimization profile is not ready",
    ):
        build_frontier_rate_attack_payloads(
            repo_root=tmp_path,
            queue_id="frontier_final_rate_attack_unready_target_scope",
            archive_records=[record],
            results_root=tmp_path / "results",
            target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
            include_optional_target_blockers=False,
            member_name="payload.bin",
            target_optimization_profile=profile,
            require_target_profile_ready=True,
        )


def test_frontier_bootstrap_binds_merge_and_dfl1_contexts_per_archive(
    tmp_path: Path,
) -> None:
    archive_path = _write_robust_like_archive(tmp_path / "archive.zip")
    record = archive_record(
        label="robust_current",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    contract = derive_packet_member_merge_contract(
        archive_records=[record],
        output_path=tmp_path / "merge_contract.json",
        repo_root=tmp_path,
        zip_compression_methods=("stored", "deflated"),
        zip_compresslevels=(1, 9),
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_receivers",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            PACKET_MEMBER_MERGE_TARGET_KIND,
            RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
        merge_contract=contract["merge_contract_path"],
        include_exact_readiness_followup=True,
        source_work_queue_path=tmp_path / "materializer_work_queue.json",
    )

    bootstrap = payloads["bootstrap"]
    contexts = {
        row["target_kind"]: row["context"]
        for row in payloads["contexts"]["rows"]
    }
    assert contract["schema"] == DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA
    assert contract["ready_archive_count"] == 1
    assert bootstrap["executable_target_kinds"] == [
        PACKET_MEMBER_MERGE_TARGET_KIND,
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    ]
    merge_context = contexts[PACKET_MEMBER_MERGE_TARGET_KIND]
    assert merge_context["sweep_archive_specs"] == [f"robust_current={record['path']}"]
    assert merge_context["all_members"] is True
    assert merge_context["member_selection"] == "all_members"
    assert merge_context["merge_contract"] == contract["merge_contract_path"]
    dfl1_context = contexts[RENDERER_PAYLOAD_DFL1_TARGET_KIND]
    assert dfl1_context["member_names"] == list(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES)
    assert dfl1_context["payload_member_name"] == "p"
    queue_path = tmp_path / "queue.json"
    _write_json(queue_path, payloads["queue"])
    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_expands_recompress_across_multi_member_archives(
    tmp_path: Path,
) -> None:
    archive_path = _write_robust_like_archive(tmp_path / "archive.zip")
    record = archive_record(
        label="robust_current",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_recompress_all_members",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
        include_optional_target_blockers=False,
    )

    assert payloads["bootstrap"]["executable_target_count"] == 3
    assert payloads["bootstrap"]["executable_target_kinds"] == [
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    ]
    contexts = [row["context"] for row in payloads["contexts"]["rows"]]
    assert [context["member_name"] for context in contexts] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
    ]
    assert all(
        context["sweep_archive_specs"] == [f"robust_current={record['path']}"]
        for context in contexts
    )


def test_frontier_bootstrap_omits_receiver_targets_per_archive_without_freezing_portfolio(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="x")
    record = archive_record(
        label="single_member",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_receiver_omissions",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            PACKET_MEMBER_MERGE_TARGET_KIND,
            RENDERER_PAYLOAD_DFL1_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
    )

    bootstrap = payloads["bootstrap"]
    assert bootstrap["executable_target_kinds"] == [
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    ]
    omissions = {
        (row["target_kind"], row["archive_label"]): row
        for row in bootstrap["target_omissions"]
    }
    assert (
        "packet_member_merge_requires_at_least_two_members"
        in omissions[(PACKET_MEMBER_MERGE_TARGET_KIND, "single_member")]["blockers"]
    )
    assert any(
        blocker.startswith("renderer_payload_dfl1_missing_members:")
        for blocker in omissions[
            (RENDERER_PAYLOAD_DFL1_TARGET_KIND, "single_member")
        ]["blockers"]
    )


def test_frontier_bootstrap_can_append_exact_readiness_followups(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_exact_followup",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
        include_optional_target_blockers=False,
        member_name="payload.bin",
        source_work_queue_path=tmp_path / "materializer_work_queue.json",
        include_exact_readiness_followup=True,
        exact_readiness_followup_require_ready=True,
    )

    bootstrap = payloads["bootstrap"]
    queue = payloads["queue"]
    experiment = queue["experiments"][0]
    metadata = experiment["metadata"]["frontier_rate_attack_bootstrap"]

    assert bootstrap["exact_readiness_followup_requested"] is True
    assert bootstrap["exact_readiness_followup_require_ready"] is True
    assert metadata["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is True
    assert experiment["metadata"]["exact_readiness_followup_skipped_reason"] is None
    assert [step["id"] for step in experiment["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_SUBMISSION_CLOSURE_STEP_ID,
        MATERIALIZER_EXACT_READINESS_BRIDGE_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]
    harvest_step = experiment["steps"][1]
    bridge_step = experiment["steps"][3]
    sweep_arg = harvest_step["command"][
        harvest_step["command"].index("--sweep-manifest") + 1
    ]
    assert sweep_arg.startswith(f"{experiment['id']}=")
    assert sweep_arg.endswith("packet_member_recompress_v1/sweep.json")
    assert "--work-queue" not in harvest_step["command"]
    assert "--exact-readiness-require-ready" in bridge_step["command"]


def test_frontier_bootstrap_exact_followup_request_harvests_sweep_candidates(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_exact_followup_skip",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
        include_optional_target_blockers=False,
        member_name="payload.bin",
        source_work_queue_path=tmp_path / "materializer_work_queue.json",
        include_exact_readiness_followup=True,
    )

    experiment = payloads["queue"]["experiments"][0]
    assert experiment["metadata"]["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is True
    assert experiment["metadata"]["exact_readiness_followup_skipped_reason"] is None
    assert [step["id"] for step in experiment["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_SUBMISSION_CLOSURE_STEP_ID,
        MATERIALIZER_EXACT_READINESS_BRIDGE_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]


def test_frontier_bootstrap_refuses_exact_followup_without_work_queue_custody(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    with pytest.raises(
        FrontierRateAttackBootstrapError,
        match="requires source_work_queue_path",
    ):
        build_frontier_rate_attack_payloads(
            repo_root=tmp_path,
            queue_id="frontier_final_rate_attack_missing_custody",
            archive_records=[record],
            results_root=tmp_path / "results",
            target_kinds=(PACKET_MEMBER_RECOMPRESS_TARGET_KIND,),
            include_optional_target_blockers=False,
            include_exact_readiness_followup=True,
        )


def test_frontier_bootstrap_cli_writes_valid_queue(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip")
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_final_rate_attack_queue.py",
            "--no-current-frontier",
            "--archive",
            f"frontier={archive_path}",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_final_rate_attack_unit",
            "--target-kind",
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            "--no-optional-target-blockers",
            "--include-exact-readiness-followup",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    queue_path = output_dir / "experiment_queue.json"
    bootstrap_path = output_dir / "frontier_rate_attack_bootstrap.json"
    assert queue_path.exists()
    assert bootstrap_path.exists()
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert bootstrap["archive_count"] == 1
    assert bootstrap["archives"][0]["sha256"] == sha256_file(archive_path)
    assert bootstrap["target_optimization_profile_metadata"]["target_mode"] == (
        "contest_video_overfit"
    )
    assert bootstrap["target_optimization_profile_metadata"][
        "declared_overfit_allowed"
    ] is True
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue["experiments"][0]["metadata"][
        "frontier_target_optimization_profile"
    ]["target_mode"] == "contest_video_overfit"
    assert queue["experiments"][0]["metadata"][
        "exact_readiness_followup_enabled"
    ] is True
    assert queue["experiments"][0]["metadata"][
        "exact_readiness_followup_skipped_reason"
    ] is None
    assert [step["id"] for step in queue["experiments"][0]["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_SUBMISSION_CLOSURE_STEP_ID,
        MATERIALIZER_EXACT_READINESS_BRIDGE_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]
    harvest_step = queue["experiments"][0]["steps"][1]
    assert "--sweep-manifest" in harvest_step["command"]
    assert "--chain-manifest" not in harvest_step["command"]

    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_cli_defaults_to_end_to_end_rate_attack_queue(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip", member_name="payload.bin")
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_final_rate_attack_queue.py",
            "--no-current-frontier",
            "--archive",
            f"frontier={archive_path}",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_final_rate_attack_default_e2e",
            "--target-kind",
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            "--member-name",
            "payload.bin",
            "--no-optional-target-blockers",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    bootstrap = json.loads(
        (output_dir / "frontier_rate_attack_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    queue = json.loads((output_dir / "experiment_queue.json").read_text(encoding="utf-8"))
    experiment = queue["experiments"][0]
    assert bootstrap["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_requested"] is True
    assert experiment["metadata"]["exact_readiness_followup_enabled"] is True
    assert [step["id"] for step in experiment["steps"]] == [
        MATERIALIZER_EXECUTION_STEP_ID,
        MATERIALIZER_HARVEST_STEP_ID,
        MATERIALIZER_SUBMISSION_CLOSURE_STEP_ID,
        MATERIALIZER_EXACT_READINESS_BRIDGE_STEP_ID,
        MATERIALIZER_DISPATCH_PLAN_STEP_ID,
    ]
    refresh_command = bootstrap["operator_commands"]["refresh_feedback_after_execute"]
    assert refresh_command[1] == "tools/build_frontier_rate_attack_feedback_refresh.py"
    assert "--materializer-feedback" in refresh_command
    assert refresh_command[
        refresh_command.index("--materializer-feedback") + 1
    ].endswith("final_rate_attack_signal_harvest.json")
    assert refresh_command[
        refresh_command.index("--frontier-artifact-root") + 1
    ].endswith("final_rate_attack_signal_harvest.json")
    assert refresh_command[
        refresh_command.index("--local-cpu-eureka-root") + 1
    ].endswith("final_rate_attack_signal_harvest.json")
    assert "--skip-raw-retention-plan" in refresh_command
    assert "--skip-mlx-retention-plan" in refresh_command


def test_frontier_bootstrap_execute_observer_signal_artifacts_are_planner_owned(
    tmp_path: Path,
) -> None:
    module = _load_frontier_queue_builder_tool_module()
    output_dir = tmp_path / "out"
    observation = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "frontier_final_rate_attack_signal_unit",
        "queue_sha256": "a" * 64,
        "healthy": True,
        "status_counts": {"succeeded": 1},
        "blockers": [],
        "state_watermark": {
            "queue_id": "frontier_final_rate_attack_signal_unit",
            "updated_at_utc": "2026-05-27T00:00:00Z",
        },
        "succeeded_artifact_steps": [
            {
                "experiment_id": "materializer_work_frontier_rate_attack_zip_header",
                "step_id": "build_materializer_submission_closure",
                "resource_kind": "local_cpu",
                "candidate_ids": ["zip_header_candidate"],
                "expected_artifacts": [
                    {
                        "path": str(tmp_path / "submission_closure.json"),
                        "optimizer_candidate_queue_materializer_rows": [
                            {
                                "candidate_id": "zip_header_candidate",
                                "target_kind": PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
                                "materializer_id": "family_agnostic_materializer",
                                "receiver_contract_kind": "archive_runtime_consumption",
                                "receiver_contract_satisfied": True,
                                "serialized_archive_delta_status": "realized_saving",
                                "serialized_archive_delta_savings_realized": True,
                                "serialized_archive_delta_realized_saved_bytes": 7,
                                "serialized_archive_delta_source_archive_bytes": 101,
                                "serialized_archive_delta_candidate_archive_bytes": 94,
                                "score_claim": False,
                                "promotion_eligible": False,
                                "rank_or_kill_eligible": False,
                                "ready_for_exact_eval_dispatch": False,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    observe_result = {
        "returncode": 0,
        "stdout": json.dumps(observation),
        "stderr": "",
    }

    result = module._write_execution_observer_signal_artifacts(
        output_dir=output_dir,
        observe_result=observe_result,
    )

    observer_path = output_dir / "observer_revalidation.json"
    signal_path = output_dir / "materializer_signal_observations.jsonl"
    harvest_path = output_dir / "final_rate_attack_signal_harvest.json"
    assert result["artifacts"]["observer_revalidation"].endswith("observer_revalidation.json")
    assert observer_path.exists()
    assert signal_path.exists()
    assert harvest_path.exists()

    rows = [
        json.loads(line)
        for line in signal_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows
    materializer_rows = [
        row
        for row in rows
        if row["observation_kind"] in {
            "family_agnostic_materializer_empirical_observation",
            "materializer_chain_archive_delta",
        }
    ]
    assert materializer_rows
    assert all(row["score_claim"] is False for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)

    harvest = json.loads(harvest_path.read_text(encoding="utf-8"))
    assert harvest["schema"] == module.SIGNAL_HARVEST_SCHEMA
    assert harvest["score_claim"] is False
    assert harvest["promotion_eligible"] is False
    assert harvest["ready_for_exact_eval_dispatch"] is False
    assert harvest["queue_healthy"] is True
    assert harvest["materializer_rate_positive_count"] >= 1
    assert harvest["materializer_saved_bytes_sum"] >= 7


def test_post_execute_feedback_refresh_command_is_bounded_to_queue_signal(
    tmp_path: Path,
) -> None:
    module = _load_frontier_queue_builder_tool_module()

    command = module._post_execute_feedback_refresh_command(
        queue_id="frontier_final_rate_attack_unit",
        output_dir=tmp_path / "out",
        results_root=tmp_path / "results",
        signal_harvest_path=tmp_path / "out" / "final_rate_attack_signal_harvest.json",
        action_summary="none",
        candidate_limit=4,
        local_cpu_concurrency=2,
        max_files_per_root=32,
        include_retention_plan=False,
    )

    assert command[1] == "tools/build_frontier_rate_attack_feedback_refresh.py"
    assert command[command.index("--action-summary") + 1] == "none"
    assert command[command.index("--candidate-limit") + 1] == "4"
    assert command[command.index("--local-cpu-concurrency") + 1] == "2"
    assert command[command.index("--max-files-per-root") + 1] == "32"
    assert command[command.index("--materializer-feedback") + 1].endswith(
        "final_rate_attack_signal_harvest.json"
    )
    assert command[command.index("--frontier-artifact-root") + 1].endswith(
        "final_rate_attack_signal_harvest.json"
    )
    assert command[command.index("--local-cpu-eureka-root") + 1].endswith(
        "final_rate_attack_signal_harvest.json"
    )
    assert "--skip-raw-retention-plan" in command
    assert "--skip-mlx-retention-plan" in command


def test_post_feedback_child_queue_selection_is_keyed_and_bounded(
    tmp_path: Path,
) -> None:
    first = tmp_path / "operation_chain_compiler_queue.json"
    second = tmp_path / "autonomous_chain_optimization_queue.json"
    _write_json(first, {"schema": "experiment_queue.v1", "queue_id": "first"})
    _write_json(second, {"schema": "experiment_queue.v1", "queue_id": "second"})

    selected = select_post_feedback_child_queue_artifacts(
        {
            "random_queue": str(tmp_path / "random_queue.json"),
            "autonomous_chain_optimization_queue": str(second),
            "operation_chain_compiler_queue": str(first),
        },
        repo_root=tmp_path,
        limit=1,
    )

    assert selected == [
        {
            "artifact_key": "operation_chain_compiler_queue",
            "queue_path": "operation_chain_compiler_queue.json",
        }
    ]


def test_post_feedback_child_queue_selection_prefers_runnable_before_frozen(
    tmp_path: Path,
) -> None:
    operation = tmp_path / "operation_chain_compiler_queue.json"
    autonomous = tmp_path / "autonomous_chain_optimization_queue.json"
    receiver = tmp_path / "receiver_repair_queue.json"
    _write_json(
        operation,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "operation",
            "experiments": [{"id": "op", "status": "queued", "steps": []}],
        },
    )
    _write_json(
        autonomous,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "autonomous",
            "experiments": [{"id": "auto", "status": "frozen", "steps": []}],
        },
    )
    _write_json(
        receiver,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "receiver",
            "experiments": [{"id": "rx", "status": "queued", "steps": []}],
        },
    )

    selected = select_post_feedback_child_queue_artifacts(
        {
            "autonomous_chain_optimization_queue": str(autonomous),
            "operation_chain_compiler_queue": str(operation),
            "receiver_repair_queue": str(receiver),
        },
        repo_root=tmp_path,
        limit=2,
    )

    assert [row["artifact_key"] for row in selected] == [
        "operation_chain_compiler_queue",
        "receiver_repair_queue",
    ]


def test_post_feedback_child_queue_selection_runs_posterior_followup_before_waterfill(
    tmp_path: Path,
) -> None:
    followup = tmp_path / "repair_posterior_acquisition_followup_queue.json"
    waterfill = tmp_path / "repair_budget_waterfill_queue.json"
    _write_json(
        followup,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "posterior_followup",
            "experiments": [{"id": "posterior", "status": "queued", "steps": []}],
        },
    )
    _write_json(
        waterfill,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "waterfill",
            "experiments": [{"id": "waterfill", "status": "queued", "steps": []}],
        },
    )

    selected = select_post_feedback_child_queue_artifacts(
        {
            "repair_budget_waterfill_queue": str(waterfill),
            "repair_posterior_acquisition_followup_queue": str(followup),
        },
        repo_root=tmp_path,
        limit=1,
    )

    assert selected == [
        {
            "artifact_key": "repair_posterior_acquisition_followup_queue",
            "queue_path": "repair_posterior_acquisition_followup_queue.json",
        }
    ]


def test_post_feedback_child_queue_execution_preserves_deferred_frozen_plan(
    tmp_path: Path,
) -> None:
    operation = tmp_path / "operation_chain_compiler_queue.json"
    receiver = tmp_path / "receiver_repair_queue.json"
    autonomous = tmp_path / "autonomous_chain_optimization_queue.json"
    payloads = {
        operation: {
            "schema": "experiment_queue.v1",
            "queue_id": "operation",
            "experiments": [{"id": "op", "status": "queued", "steps": []}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        receiver: {
            "schema": "experiment_queue.v1",
            "queue_id": "receiver",
            "experiments": [{"id": "rx", "status": "queued", "steps": []}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        autonomous: {
            "schema": "experiment_queue.v1",
            "queue_id": "autonomous",
            "experiments": [
                {
                    "id": "auto",
                    "status": "frozen",
                    "metadata": {
                        "queue_actuation_blockers": [
                            "no_receiver_closed_saved_bytes_available",
                            "no_targeted_component_correction_rows",
                        ],
                    },
                    "steps": [
                        {
                            "id": "inspect",
                            "command": ["python", "-c", "print('inspect')"],
                            "telemetry": {
                                "input_artifact_paths": [
                                    "receiver_closed_correction_budget.json"
                                ]
                            },
                        }
                    ],
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }
    for path, payload in payloads.items():
        _write_json(path, payload)

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        queue_arg = command[command.index("--queue") + 1] if "--queue" in command else ""
        queue_payload = payloads.get(tmp_path / queue_arg)
        if "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 1,
                    "success_count": 1,
                    "failure_count": 0,
                }
            )
        elif "observe" in command and queue_payload is not None:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": queue_payload["queue_id"],
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"succeeded": 1},
                    "blockers": [],
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={
            "operation_chain_compiler_queue": str(operation),
            "receiver_repair_queue": str(receiver),
            "autonomous_chain_optimization_queue": str(autonomous),
        },
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=2,
        run_command=fake_run,
    )

    assert [row["artifact_key"] for row in report["selected_queues"]] == [
        "operation_chain_compiler_queue",
        "receiver_repair_queue",
    ]
    assert report["activation_plan_count"] == 0
    assert report["deferred_activation_plan_count"] == 1
    assert report["deferred_activation_posterior_append_report_count"] == 1
    assert report["activation_posterior_append_report_count"] == 1
    assert report["activation_posterior_appended_count"] == 1
    deferred = report["deferred_activation_plans"][0]
    assert deferred["artifact_key"] == "autonomous_chain_optimization_queue"
    activation_plan = json.loads(
        (tmp_path / deferred["activation_plan_path"]).read_text(encoding="utf-8")
    )
    assert activation_plan["schema"] == POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA
    assert activation_plan["blocked_experiment_count"] == 1
    assert {
        action["activation_action"] for action in activation_plan["activation_actions"]
    } == {
        "materialize_receiver_closed_rate_budget_credit",
        "harvest_targeted_component_response_rows",
        "thaw_queue_definition_after_prerequisite_evidence_lands",
    }
    append_report = json.loads(
        (tmp_path / deferred["activation_posterior_append_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert append_report["appended_count"] == 1
    posterior_path = tmp_path / ".omx/state/repair_campaign_stackability_posterior.jsonl"
    posterior_rows = [
        json.loads(line)
        for line in posterior_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(posterior_rows) == 1
    assert posterior_rows[0]["typed_response_id"] == "activation_plan:autonomous:auto"


def test_post_feedback_child_queue_execution_refuses_missing_required_portfolio_coverage(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_requires_portfolio",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [{"id": "op", "status": "queued", "steps": []}],
    }
    _write_json(queue, queue_payload)
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> dict[str, object]:
        commands.append(command)
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"operation_chain_compiler_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
        require_portfolio_coverage=True,
    )

    assert commands == []
    assert report["candidate_queue_count"] == 1
    assert report["selected_queue_count"] == 0
    assert report["executed_queue_count"] == 0
    assert report["preflight_blocked_execution"] is True
    preflight = report["portfolio_coverage_preflight"]
    assert preflight["schema"] == POST_FEEDBACK_PORTFOLIO_COVERAGE_PREFLIGHT_SCHEMA
    assert preflight["valid"] is False
    assert preflight["blockers"] == [
        "frontier_rate_attack_portfolio_coverage_artifact_missing"
    ]
    assert report["portfolio_coverage_preflight_blocker_count"] == 1
    _assert_false_authority(report)
    assert (tmp_path / report["artifact_path"]).exists()


def test_post_feedback_child_queue_execution_accepts_required_portfolio_coverage(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    portfolio = tmp_path / "frontier_rate_attack_portfolio_coverage.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_with_portfolio",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [],
    }
    _write_json(queue, queue_payload)
    _write_json(portfolio, _valid_portfolio_coverage_payload())

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        if "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 1,
                    "success_count": 1,
                    "failure_count": 0,
                }
            )
        elif "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "child_queue_with_portfolio",
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"succeeded": 1},
                    "blockers": [],
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={
            "operation_chain_compiler_queue": str(queue),
            "frontier_rate_attack_portfolio_coverage": str(portfolio),
        },
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
        require_portfolio_coverage=True,
    )

    assert report["portfolio_coverage_preflight_valid"] is True
    assert report["preflight_blocked_execution"] is False
    assert report["selected_queue_count"] == 1
    assert report["executed_queue_count"] == 1
    assert report["failed_command_count"] == 0
    preflight = report["portfolio_coverage_preflight"]
    assert preflight["artifact_path"] == "frontier_rate_attack_portfolio_coverage.json"
    assert preflight["coverage_ready_for_bounded_local_followup"] is True
    assert preflight["bound_pathway_count"] == 2
    assert {
        row["target_kind"]: row["binding_status"]
        for row in preflight["deferred_binding_statuses"]
    } == {
        DQS1_PAIRSET_TARGET_KIND: "bound",
        INVERSE_SCORER_CELL_TARGET_KIND: "bound",
    }
    assert report["queue_runs"][0]["observer_revalidation_valid"] is True


def test_post_feedback_child_queue_execution_preserves_observer_artifacts(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_unit",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [],
    }
    _write_json(queue, queue_payload)
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> dict[str, object]:
        commands.append(command)
        stdout = ""
        if "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "child_queue_unit",
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"succeeded": 1},
                    "blockers": [],
                }
            )
        elif "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 1,
                    "success_count": 1,
                    "failure_count": 0,
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"operation_chain_compiler_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
    )

    assert report["schema"] == POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA
    assert report["selected_queue_count"] == 1
    assert report["failed_command_count"] == 0
    assert report["observer_revalidation_failed_count"] == 0
    assert report["stalled_queue_count"] == 0
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    run = report["queue_runs"][0]
    assert run["artifact_key"] == "operation_chain_compiler_queue"
    assert run["queue_healthy"] is True
    assert run["observer_revalidation_valid"] is True
    assert run["observer_revalidation_blockers"] == []
    assert run["steps_started"] == 1
    assert run["progress_made"] is True
    assert run["observer_revalidation_path"].endswith(
        "post_execute_feedback_child_queue_observations/"
        "operation_chain_compiler_queue/observer_revalidation.json"
    )
    assert (tmp_path / run["observer_revalidation_path"]).exists()
    worker_command = commands[2]
    assert worker_command[worker_command.index("--max-steps") + 1] == "3"


def test_post_feedback_child_queue_revalidation_rejects_unhealthy_observer(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_unhealthy",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [],
    }
    _write_json(queue, queue_payload)

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        if "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "child_queue_unhealthy",
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": False,
                    "blocker_count": 1,
                    "blockers": [
                        "experiment_queue_observation_artifact_postcondition_failures:1"
                    ],
                    "succeeded_artifact_failure_steps": [
                        {
                            "experiment_id": "bad_child",
                            "step_id": "materializer_harvest",
                        }
                    ],
                    "status_counts": {"succeeded": 1},
                }
            )
        elif "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 1,
                    "success_count": 1,
                    "failure_count": 0,
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"operation_chain_compiler_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
    )

    run = report["queue_runs"][0]
    assert report["observer_revalidation_failed_count"] == 1
    assert run["queue_healthy"] is False
    assert run["observer_revalidation_valid"] is False
    assert run["progress_made"] is False
    assert run["observer_revalidation_blockers"] == [
        "observer_queue_unhealthy",
        (
            "observer_blocker:"
            "experiment_queue_observation_artifact_postcondition_failures:1"
        ),
        "observer_blocker_count_nonzero",
        "observer_artifact_postcondition_failures_present",
    ]
    assert run["observer_revalidation"]["observed_healthy"] is False
    assert run["observer_revalidation"]["observed_blocker_count"] == 1
    assert run["observer_revalidation"]["observed_artifact_failure_count"] == 1


def test_post_feedback_child_queue_execution_reports_stalled_queued_work(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_stalled",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [],
    }
    _write_json(queue, queue_payload)

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        if "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 0,
                    "success_count": 0,
                    "failure_count": 0,
                }
            )
        elif "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "child_queue_stalled",
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"queued": 2},
                    "blockers": [],
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"operation_chain_compiler_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
    )

    run = report["queue_runs"][0]
    assert report["stalled_queue_count"] == 1
    assert run["steps_started"] == 0
    assert run["progress_made"] is False
    assert run["progress_blockers"] == [
        "child_queue_worker_started_zero_steps_with_queued_work"
    ]
    assert run["queue_healthy"] is True


def test_post_feedback_child_queue_execution_classifies_frozen_child_queue(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "autonomous_chain_optimization_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_frozen",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [
            {
                "id": "blocked_until_receiver_ready",
                "status": "frozen",
                "priority": 10,
                "tags": ["encoder-repair-allocator", "segnet-posenet-waterfill"],
                "metadata": {
                    "queue_actuation_blockers": [
                        "no_targeted_component_correction_response_rows",
                        "no_receiver_closed_saved_bytes_available",
                        "exact_auth_eval_required_before_score_or_promotion_claim",
                    ],
                },
                "steps": [
                    {
                        "id": "materialize",
                        "command": ["python", "-c", "print('frozen')"],
                        "telemetry": {
                            "input_artifact_paths": [
                                "targeted_component_correction_response_harvest.json",
                                "receiver_closed_correction_budget.json",
                            ],
                            "artifact_paths": ["repair_budget_waterfill.json"],
                        },
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": "repair_budget_waterfill.json",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    _write_json(queue, queue_payload)

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        if "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 0,
                    "success_count": 0,
                    "failure_count": 0,
                }
            )
        elif "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "child_queue_frozen",
                    "queue_sha256": stable_json_sha256(queue_payload),
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"frozen": 1},
                    "blockers": [],
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"autonomous_chain_optimization_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
    )

    run = report["queue_runs"][0]
    assert report["stalled_queue_count"] == 0
    assert report["frozen_noop_queue_count"] == 1
    assert run["steps_started"] == 0
    assert run["progress_made"] is False
    assert run["progress_blockers"] == [
        "child_queue_remaining_work_frozen_by_definition"
    ]
    assert run["queue_status_counts"] == {"frozen": 1}
    assert report["activation_plan_count"] == 1
    assert run["activation_plan_path"].endswith(
        "post_execute_feedback_child_queue_observations/"
        "autonomous_chain_optimization_queue/activation_plan.json"
    )
    activation_plan_path = tmp_path / run["activation_plan_path"]
    activation_plan = json.loads(activation_plan_path.read_text(encoding="utf-8"))
    assert activation_plan["schema"] == POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA
    assert activation_plan["score_claim"] is False
    assert activation_plan["promotion_eligible"] is False
    assert activation_plan["blocked_experiment_count"] == 1
    assert {
        action["activation_action"] for action in activation_plan["activation_actions"]
    } == {
        "harvest_targeted_component_response_rows",
        "materialize_receiver_closed_rate_budget_credit",
        "route_byte_closed_candidate_to_exact_auth_eval_handoff",
        "thaw_queue_definition_after_prerequisite_evidence_lands",
    }
    step_refs = activation_plan["blocked_experiments"][0]["step_evidence_refs"][0]
    assert step_refs["telemetry_input_artifact_paths"] == [
        "targeted_component_correction_response_harvest.json",
        "receiver_closed_correction_budget.json",
    ]
    assert step_refs["postcondition_paths"] == ["repair_budget_waterfill.json"]
    assert report["activation_learning_signal_report_count"] == 1
    signal_report_path = tmp_path / run["activation_learning_signal_report_path"]
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    assert signal_report["schema"] == REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    assert signal_report["blocked_signal_count"] == 1
    signal = signal_report["learning_signal_rows"][0]
    assert signal["learning_signal_kind"] == "blocked_child_queue_activation_plan"
    assert signal["local_planning_update"]["recommended_acquisition_policy"] == (
        "increase_priority_for_targeted_component_response_harvest"
    )
    assert signal["local_planning_update"]["planner_feature_vector"][
        "has_receiver_closed_budget_request"
    ] is True
    assert report["activation_posterior_append_report_count"] == 1
    assert report["activation_posterior_appended_count"] == 1
    append_report_path = tmp_path / run["activation_posterior_append_report_path"]
    append_report = json.loads(append_report_path.read_text(encoding="utf-8"))
    assert append_report["appended_count"] == 1
    posterior_path = tmp_path / ".omx/state/repair_campaign_stackability_posterior.jsonl"
    posterior_rows = [
        json.loads(line)
        for line in posterior_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(posterior_rows) == 1
    assert posterior_rows[0]["typed_response_id"] == (
        "activation_plan:child_queue_frozen:blocked_until_receiver_ready"
    )


def test_post_feedback_child_queue_execution_revalidates_observer_identity(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "operation_chain_compiler_queue.json"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "child_queue_identity",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [],
    }
    _write_json(queue, queue_payload)

    def fake_run(command: list[str]) -> dict[str, object]:
        stdout = ""
        if "run-worker" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_worker_result.v1",
                    "steps_started": 1,
                    "success_count": 1,
                    "failure_count": 0,
                }
            )
        elif "observe" in command:
            stdout = json.dumps(
                {
                    "schema": "experiment_queue_observation.v1",
                    "queue_id": "wrong_child_queue",
                    "queue_sha256": "0" * 64,
                    "observe_read_only": True,
                    "healthy": True,
                    "status_counts": {"succeeded": 1},
                    "blockers": [],
                }
            )
        return {
            "command": command,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    report = execute_post_feedback_child_queues(
        repo_root=tmp_path,
        feedback_artifacts={"operation_chain_compiler_queue": str(queue)},
        output_dir=tmp_path / "out",
        max_steps=3,
        max_parallel=1,
        limit=4,
        run_command=fake_run,
    )

    run = report["queue_runs"][0]
    assert report["observer_revalidation_failed_count"] == 1
    assert run["observer_revalidation_valid"] is False
    assert run["observer_revalidation_blockers"] == [
        "observer_queue_id_mismatch",
        "observer_queue_sha256_mismatch",
    ]
    assert run["progress_made"] is False
    assert run["observer_revalidation"]["expected_queue_sha256"] == stable_json_sha256(
        queue_payload
    )
    assert run["observer_revalidation"]["observer_revalidation_sha256"]


def test_frontier_bootstrap_results_root_probe_requires_readable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_frontier_queue_builder_tool_module()
    original_read_text = Path.read_text

    def deny_probe_read(path: Path, *args: object, **kwargs: object) -> str:
        if path.name.startswith(".pact_write_read_probe_"):
            raise PermissionError("read denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_probe_read)

    assert module._can_create_child(tmp_path) is False
    assert not any(tmp_path.glob(".pact_write_read_probe_*"))


def test_frontier_bootstrap_cli_derives_merge_contract_for_receiver_targets(
    tmp_path: Path,
) -> None:
    archive_path = _write_robust_like_archive(tmp_path / "archive.zip")
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_final_rate_attack_queue.py",
            "--no-current-frontier",
            "--archive",
            f"robust_current={archive_path}",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_final_rate_attack_receiver_cli",
            "--target-kind",
            PACKET_MEMBER_MERGE_TARGET_KIND,
            "--target-kind",
            RENDERER_PAYLOAD_DFL1_TARGET_KIND,
            "--no-optional-target-blockers",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "derived_packet_member_merge_contract" in payload["artifacts"]
    derived_path = output_dir / "derived_packet_member_merge_contract.json"
    assert derived_path.exists()
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert derived["ready_archive_count"] == 1
    bootstrap = json.loads(
        (output_dir / "frontier_rate_attack_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    assert bootstrap["executable_target_kinds"] == [
        PACKET_MEMBER_MERGE_TARGET_KIND,
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    ]
    queue_path = output_dir / "experiment_queue.json"
    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_derives_per_archive_section_manifest_target(
    tmp_path: Path,
) -> None:
    decoder_section = brotli.compress(b"decoder payload" * 16, quality=3)
    tail_section = brotli.compress(b"latent payload" * 16, quality=3)
    payload = bytes([0xFF]) + len(decoder_section).to_bytes(3, "little") + decoder_section + tail_section
    archive_path = _write_stored_archive(tmp_path / "archive.zip", member_name="x", payload=payload)
    record = archive_record(
        label="pr106_like",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    derived = derive_archive_section_recode_manifests(
        archive_records=[record],
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
    )

    assert derived["schema"] == DERIVED_SECTION_MANIFEST_BATCH_SCHEMA
    assert derived["ready_manifest_count"] == 1
    row = derived["rows"][0]
    assert row["score_claim"] is False
    assert row["ready_for_materializer_target"] is True
    assert row["selected_section_names"] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_sections",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,),
        include_optional_target_blockers=False,
        section_manifest_by_archive_label={"pr106_like": row["section_manifest_path"]},
        section_names_by_archive_label={"pr106_like": tuple(row["selected_section_names"])},
    )

    bootstrap = payloads["bootstrap"]
    contexts = payloads["contexts"]
    assert bootstrap["executable_target_kinds"] == [ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]
    assert bootstrap["target_omissions"] == []
    context = contexts["rows"][0]["context"]
    assert context["sweep_archive_specs"] == [f"pr106_like={record['path']}"]
    assert context["section_manifest"] == row["section_manifest_path"]
    assert context["section_names"] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    queue_path = tmp_path / "queue.json"
    _write_json(queue_path, payloads["queue"])
    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_omits_section_target_without_brotli_sections(
    tmp_path: Path,
) -> None:
    payload = bytes([0xFF]) + (7).to_bytes(3, "little") + b"decoder" + b"tail"
    archive_path = _write_stored_archive(tmp_path / "archive.zip", member_name="x", payload=payload)
    record = archive_record(
        label="opaque_pr106",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    derived = derive_archive_section_recode_manifests(
        archive_records=[record],
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
    )

    assert derived["ready_manifest_count"] == 0
    row = derived["rows"][0]
    assert row["ready_for_materializer_target"] is False
    assert "section_manifest_has_no_brotli_decompressible_sections" in row["blockers"]

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_sections_blocked",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
        section_manifest_by_archive_label={"opaque_pr106": row["section_manifest_path"]},
        section_names_by_archive_label={"opaque_pr106": tuple(row["selected_section_names"])},
    )

    omissions = payloads["bootstrap"]["target_omissions"]
    assert omissions[0]["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert omissions[0]["archive_label"] == "opaque_pr106"
    assert (
        "archive_section_entropy_recode_requires_brotli_decompressible_section"
        in omissions[0]["blockers"]
    )
    assert payloads["bootstrap"]["executable_target_kinds"] == [
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
    ]


def test_frontier_bootstrap_refuses_shared_section_manifest_for_multi_archive_sweep(
    tmp_path: Path,
) -> None:
    first = _write_stored_archive(
        tmp_path / "first.zip",
        member_name="x",
        payload=b"\xff\x01\x00\x00a",
    )
    second = _write_stored_archive(
        tmp_path / "second.zip",
        member_name="x",
        payload=b"\xff\x01\x00\x00b",
    )
    records = [
        archive_record(
            label="first",
            archive_path=first,
            repo_root=tmp_path,
            source_kind="unit_test",
        ),
        archive_record(
            label="second",
            archive_path=second,
            repo_root=tmp_path,
            source_kind="unit_test",
        ),
    ]
    manifest = tmp_path / "shared_section_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "unit_test_section_manifest.v1",
            "member": {"name": "x"},
            "sections": [
                {
                    "name": "one_byte",
                    "offset": 4,
                    "length": 1,
                    "sha256": "unused",
                }
            ],
        },
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_shared_section_refusal",
        archive_records=records,
        results_root=tmp_path / "results",
        target_kinds=(
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
        section_manifest=manifest.as_posix(),
        section_names=("one_byte",),
    )

    assert payloads["bootstrap"]["executable_target_kinds"] == [
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
    ]
    omissions = payloads["bootstrap"]["target_omissions"]
    assert omissions[0]["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert omissions[0]["archive_labels"] == ["first", "second"]
    assert omissions[0]["score_claim"] is False
    assert omissions[0]["ready_for_exact_eval_dispatch"] is False
    assert (
        "archive_section_entropy_recode_requires_per_archive_section_manifest_for_multi_archive_sweep"
        in omissions[0]["blockers"]
    )


def test_frontier_bootstrap_parse_archive_spec_checks_sha_and_members(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "candidate.zip", member_name="masks.mkv")

    record = parse_archive_spec(f"candidate={archive_path}", repo_root=tmp_path)

    assert record["label"] == "candidate"
    assert record["sha256"] == sha256_file(archive_path)
    assert record["zip_member_count"] == 1
    assert record["zip_members"][0]["name"] == "masks.mkv"
    _assert_false_authority(record)


def test_resolve_current_frontier_archive_from_auth_request(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path / "experiments" / "results" / "candidate" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(archive)
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "score": 0.123,
                "evidence_grade": "[contest-CPU]",
                "hardware_substrate": "linux_x86_64_cpu",
                "measured_at_utc": "2026-05-25T00:00:00Z",
                "extra": {"archive_bytes": archive.stat().st_size},
            }
        },
    )
    request_path = (
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval_cpu"
        / "job"
        / "modal_cpu_auth_eval_local_request.json"
    )
    _write_json(
        request_path,
        {
            "archive_path": archive.as_posix(),
            "archive_sha256": digest,
            "archive_size_bytes": archive.stat().st_size,
        },
    )

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
    )

    assert resolution["archive_sha256"] == digest
    assert resolution["archive_record"]["path"] == (
        "experiments/results/candidate/submission_dir/archive.zip"
    )
    assert resolution["match"]["request_path"] == (
        "experiments/results/modal_auth_eval_cpu/job/modal_cpu_auth_eval_local_request.json"
    )


def test_resolve_current_frontier_archive_skips_unreadable_request_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_archive = _write_archive(
        tmp_path / "experiments" / "results" / "candidate" / "submission_dir" / "archive.zip"
    )
    unreadable_archive = _write_archive(
        tmp_path / "external" / "stale" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(valid_archive)
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "score": 0.123,
                "evidence_grade": "[contest-CPU]",
                "hardware_substrate": "linux_x86_64_cpu",
                "measured_at_utc": "2026-05-25T00:00:00Z",
                "extra": {"archive_bytes": valid_archive.stat().st_size},
            }
        },
    )
    _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval_cpu"
        / "stale"
        / "modal_cpu_auth_eval_local_request.json",
        {
            "archive_path": unreadable_archive.as_posix(),
            "archive_sha256": digest,
            "archive_size_bytes": valid_archive.stat().st_size,
        },
    )
    real_sha256_file = frontier_bootstrap.sha256_file

    def guarded_sha256(path: str | Path) -> str:
        if Path(path) == unreadable_archive:
            raise PermissionError("simulated external storage permission failure")
        return real_sha256_file(path)

    monkeypatch.setattr(frontier_bootstrap, "sha256_file", guarded_sha256)

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
    )

    assert resolution["archive_sha256"] == digest
    assert resolution["archive_record"]["path"] == (
        "experiments/results/candidate/submission_dir/archive.zip"
    )
    assert resolution["match"]["source"] == "default_submission_archive_search"


def test_resolve_current_frontier_archive_disambiguates_duplicate_sha_by_auth_score(
    tmp_path: Path,
) -> None:
    first_archive = _write_archive(
        tmp_path / "runs" / "first" / "submission_dir" / "archive.zip"
    )
    second_archive = _write_archive(
        tmp_path / "runs" / "second" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(first_archive)
    assert sha256_file(second_archive) == digest
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "score": 0.123456789,
                "evidence_grade": "[contest-CPU]",
                "hardware_substrate": "linux_x86_64_cpu",
                "measured_at_utc": "2026-05-25T00:00:00Z",
                "extra": {"archive_bytes": first_archive.stat().st_size},
            }
        },
    )
    request_root = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    for label, archive, score in (
        ("first", first_archive, 0.123456789),
        ("second", second_archive, 0.123455789),
    ):
        run_dir = request_root / label
        _write_json(
            run_dir / "modal_cpu_auth_eval_local_request.json",
            {
                "archive_path": archive.as_posix(),
                "archive_sha256": digest,
                "archive_size_bytes": archive.stat().st_size,
                "expected_runtime_tree_sha256": f"{1 if label == 'first' else 2}" * 64,
            },
        )
        _write_json(
            run_dir / "contest_auth_eval.json",
            {
                "canonical_score": score,
                "provenance": {
                    "archive_sha256": digest,
                    "archive_size_bytes": archive.stat().st_size,
                },
            },
        )

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
        request_search_roots=(request_root,),
    )

    assert resolution["archive_record"]["path"] == (
        "runs/first/submission_dir/archive.zip"
    )
    assert resolution["match"]["disambiguation"]["strategy"] == (
        "auth_eval_canonical_score_matches_frontier_pointer"
    )


def test_resolve_current_frontier_archive_from_default_submission_dir(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path
        / "experiments"
        / "results"
        / "frontier_candidate"
        / "submission_dir"
        / "archive.zip"
    )
    digest = sha256_file(archive)
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "score": 0.123,
                "evidence_grade": "[contest-CPU]",
                "hardware_substrate": "linux_x86_64_cpu",
                "measured_at_utc": "2026-05-25T00:00:00Z",
                "extra": {"archive_bytes": archive.stat().st_size},
            }
        },
    )

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
    )

    assert resolution["archive_sha256"] == digest
    assert resolution["archive_record"]["path"] == (
        "experiments/results/frontier_candidate/submission_dir/archive.zip"
    )
    assert resolution["match"]["source"] == "default_submission_archive_search"
    assert resolution["match"]["request_path"] is None


def test_resolve_current_frontier_archive_fails_closed_on_duplicate_matches(
    tmp_path: Path,
) -> None:
    first = _write_archive(
        tmp_path / "experiments" / "results" / "candidate_a" / "submission_dir" / "archive.zip"
    )
    second = _write_archive(
        tmp_path / "experiments" / "results" / "candidate_b" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(first)
    assert sha256_file(second) == digest
    (first.parent / "inflate.sh").write_text("first runtime\n", encoding="utf-8")
    (second.parent / "inflate.sh").write_text("second runtime\n", encoding="utf-8")
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "extra": {"archive_bytes": first.stat().st_size},
            }
        },
    )
    for name, archive in (("a", first), ("b", second)):
        _write_json(
            tmp_path
            / "experiments"
            / "results"
            / "modal_auth_eval_cpu"
            / name
            / "request.json",
            {
                "archive_path": archive.as_posix(),
                "archive_sha256": digest,
                "archive_size_bytes": archive.stat().st_size,
            },
        )

    with pytest.raises(FrontierRateAttackBootstrapError, match="ambiguous"):
        resolve_current_frontier_archive(
            repo_root=tmp_path,
            pointer_path=pointer_path,
            frontier_axis="contest_cpu",
        )
