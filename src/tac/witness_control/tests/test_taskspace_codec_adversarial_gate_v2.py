# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tac.contest_score import (
    UNCOMPRESSED_SIZE_BYTES,
    compute_contest_score,
)
from tac.witness_control import taskspace_codec_adversarial_gate_v2 as gate
from tac.witness_dsl.dynamic_frontier_target import DynamicFrontierTargetSnapshot


def _snapshot(root: Path, *, target: float = 0.5) -> DynamicFrontierTargetSnapshot:
    pointer = root / ".omx/state/canonical_frontier_pointer.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{}\n")
    stat = pointer.stat()
    return DynamicFrontierTargetSnapshot(
        pointer_path=str(pointer),
        pointer_bytes=stat.st_size,
        pointer_sha256=gate.sha256_bytes(pointer.read_bytes()),
        pointer_device=stat.st_dev,
        pointer_inode=stat.st_ino,
        pointer_mtime_ns=stat.st_mtime_ns,
        last_refreshed_utc=datetime.now(UTC).isoformat(),
        source_snapshot_at_utc=datetime.now(UTC).isoformat(),
        target_score=target,
        selected_axis="contest-CPU",
        selected_source="fixture",
        selected_source_kind="our_exact",
        selected_score_precision="full",
        selected_custody="fixture",
        selected_evidence_grade="[fixture]",
        selected_archive_sha256="a" * 64,
        selected_lane_id="fixture",
        selected_hardware_substrate="fixture",
        selection_rule="fixture minimum",
    )


def _patch_frontier(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    target: float = 0.5,
) -> DynamicFrontierTargetSnapshot:
    snapshot = _snapshot(root, target=target)

    def load(*, repo_root: Path | str = ".", now_utc_iso: str | None = None):
        del now_utc_iso
        assert Path(repo_root) == root
        return snapshot

    def verify(value, *, now_utc_iso: str | None = None):
        del now_utc_iso
        assert value == snapshot
        return value

    monkeypatch.setattr(gate, "load_dynamic_frontier_target", load)
    monkeypatch.setattr(gate, "verify_dynamic_frontier_target_snapshot", verify)
    return snapshot


def _write_json(path: Path, value: object) -> Path:
    path.write_bytes(gate.canonical_json(value))
    return path


def _seal(
    tmp_path: Path,
    *,
    campaign_id: str = "campaign-a",
    representation: str = gate.PROGRAM_RESIDUAL_LAYERED,
) -> Path:
    path = tmp_path / f"{campaign_id}.seal.json"
    receipt = gate.seal_campaign(
        campaign_id=campaign_id,
        requested_representation=representation,
        repo_root=tmp_path,
        expected_repo_root=tmp_path,
        output_path=path,
    )
    assert receipt["status"] == "ADMIT"
    return path


def _admitted_pre_encode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    campaign_id: str = "campaign-a",
    config_bytes: bytes = b'{"config":"a"}\n',
) -> tuple[Path, Path]:
    seal = _seal(tmp_path, campaign_id=campaign_id)
    config = tmp_path / f"{campaign_id}.config.json"
    config.write_bytes(config_bytes)
    identity = _write_json(tmp_path / f"{campaign_id}.g58.json", {"schema": gate.G58_IDENTITY_SCHEMA})
    terminal = _write_json(tmp_path / f"{campaign_id}.terminal.json", {"terminal": True})
    outer = _write_json(tmp_path / f"{campaign_id}.outer.json", {"outer": True})
    monkeypatch.setattr(
        gate,
        "_validate_g58_pre_encode",
        lambda *_args, **_kwargs: (
            {"strict_production_evidence": {"analytic_only": True}},
            [],
        ),
    )
    monkeypatch.setattr(
        gate,
        "_validate_program_producer_config",
        lambda *_args, **_kwargs: [],
    )
    output = tmp_path / f"{campaign_id}.pre_encode.json"
    receipt = gate.admit_pre_encode(
        campaign_seal_path=seal,
        producer_config_path=config,
        g58_identity_receipt_path=identity,
        g58_terminal_stage_chain_path=terminal,
        g58_outer_proof_path=outer,
        output_path=output,
    )
    assert receipt["status"] == "ADMIT"
    return output, config


def _encoded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    campaign_id: str = "campaign-a",
) -> tuple[Path, Path, Path, Path]:
    pre_encode, config = _admitted_pre_encode(
        monkeypatch,
        tmp_path,
        campaign_id=campaign_id,
    )
    archive = tmp_path / f"{campaign_id}.zip"
    archive.write_bytes(b"counted-archive")
    raw = tmp_path / f"{campaign_id}.raw"
    raw.write_bytes(b"decoded-raw")
    output = tmp_path / f"{campaign_id}.encode.json"
    receipt = gate.admit_encode(
        pre_encode_receipt_path=pre_encode,
        archive_path=archive,
        decoded_raw_path=raw,
        output_path=output,
    )
    assert receipt["status"] == "ADMIT"
    return output, archive, raw, config


class _DummyAxis:
    value = "CPU"


class _DummyRun:
    def __init__(
        self,
        *,
        archive: Path,
        raw: Path,
        report: Path,
        d_seg: float,
        d_pose: float,
    ) -> None:
        self.archive_sha256 = gate.artifact_identity(archive, label="archive")["sha256"]
        self.archive_nbytes = archive.stat().st_size
        self.raw_sha256 = gate.artifact_identity(raw, label="raw")["sha256"]
        self.raw_nbytes = raw.stat().st_size
        self.report_sha256 = gate.artifact_identity(report, label="report")["sha256"]
        self.avg_segnet_dist = d_seg
        self.avg_posenet_dist = d_pose
        self.report_component_recomputed_score = compute_contest_score(
            d_seg,
            d_pose,
            archive.stat().st_size,
        )
        self.execution_axis = _DummyAxis()
        self.identity_sha256 = "d" * 64


def _post_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    d_seg: float = 0.0,
    d_pose: float = 0.0,
    blocker: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path]:
    encode, archive, raw, config = _encoded(monkeypatch, tmp_path)
    eval_receipt = _write_json(tmp_path / "official_run.json", {"strict": "fixture parser"})
    report = tmp_path / "report.txt"
    report.write_text("fixture report\n")
    run = _DummyRun(
        archive=archive,
        raw=raw,
        report=report,
        d_seg=d_seg,
        d_pose=d_pose,
    )

    class Parser:
        @classmethod
        def from_receipt_bytes(cls, payload: bytes):
            assert payload == eval_receipt.read_bytes()
            return run

    monkeypatch.setattr(gate, "OfficialEvaluationRunReceiptV1", Parser)
    eval_identity = gate.artifact_identity(eval_receipt, label="eval receipt")
    integration_paths: list[Path] = []
    blocker_path = None
    if blocker is None:
        for hook in gate.INTEGRATION_HOOKS:
            integration_paths.append(
                _write_json(
                    tmp_path / f"{hook}.json",
                    {
                        "schema": gate.INTEGRATION_SCHEMA,
                        "status": "INTEGRATED",
                        "campaign_id": "campaign-a",
                        "hook": hook,
                        "exact_eval_receipt_sha256": eval_identity["sha256"],
                    },
                )
            )
    else:
        blocker_path = _write_json(tmp_path / "blocker.json", blocker)
    output = tmp_path / "post_eval.json"
    gate.admit_post_eval(
        encode_receipt_path=encode,
        eval_receipt_path=eval_receipt,
        eval_report_path=report,
        integration_receipt_paths=integration_paths,
        blocker_path=blocker_path,
        output_path=output,
    )
    return output, archive, raw, config


def test_seal_refuses_caller_target_pointer_and_wrong_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _patch_frontier(monkeypatch, tmp_path)
    receipt = gate.seal_campaign(
        campaign_id="fake-frontier",
        requested_representation=gate.PROGRAM_RESIDUAL_LAYERED,
        repo_root=tmp_path / "caller-controlled",
        expected_repo_root=tmp_path,
        output_path=tmp_path / "refused.json",
        asserted_target_score=100.0,
        asserted_pointer_sha256="f" * 64,
    )
    assert receipt["candidate_admission"] is False
    assert "CALLER_REPOSITORY_ROOT_DIFFERS_FROM_LAUNCHER_ROOT" in receipt["refusals"]
    assert "CALLER_TARGET_DIFFERS_FROM_DYNAMIC_FRONTIER" in receipt["refusals"]
    assert "CALLER_POINTER_SHA_DIFFERS_FROM_DYNAMIC_FRONTIER" in receipt["refusals"]
    assert receipt["frontier"]["pointer_sha256"] == snapshot.pointer_sha256


def test_canonical_score_preserves_upstream_divide_then_multiply_and_strict_boundary() -> None:
    d_seg = 0.001
    d_pose = 0.0001
    archive_bytes = 1
    rate = float(archive_bytes) / UNCOMPRESSED_SIZE_BYTES
    upstream_order = 100 * d_seg + (d_pose * 10) ** 0.5 + 25 * rate
    score = compute_contest_score(d_seg, d_pose, archive_bytes)
    assert score == upstream_order
    assert not score < score


def test_refused_pre_encode_blocks_encode_and_receipt_is_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    seal = _seal(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    refused_path = tmp_path / "pre_encode_refused.json"
    refused = gate.admit_pre_encode(
        campaign_seal_path=seal,
        producer_config_path=config,
        output_path=refused_path,
    )
    assert refused["status"] == "REFUSE"
    archive = tmp_path / "a.zip"
    raw = tmp_path / "a.raw"
    archive.write_bytes(b"a")
    raw.write_bytes(b"r")
    later = gate.admit_encode(
        pre_encode_receipt_path=refused_path,
        archive_path=archive,
        decoded_raw_path=raw,
        output_path=tmp_path / "encode_refused.json",
    )
    assert later["status"] == "REFUSE"
    assert "PREDECESSOR_NOT_LIVE_CANDIDATE_ADMITTED" in later["refusals"]


def test_representation_archive_raw_and_symlink_switches_refuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    pre_encode, _ = _admitted_pre_encode(monkeypatch, tmp_path)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"raw")
    switched = gate.admit_encode(
        pre_encode_receipt_path=pre_encode,
        archive_path=archive,
        decoded_raw_path=raw,
        asserted_representation=gate.DIRECT_TASK_LAYERED_CONTROL,
        output_path=tmp_path / "representation_switch.json",
    )
    assert "REQUESTED_REPRESENTATION_SWITCH" in switched["refusals"]
    link = tmp_path / "archive-link.zip"
    link.symlink_to(archive)
    symlinked = gate.admit_encode(
        pre_encode_receipt_path=pre_encode,
        archive_path=link,
        decoded_raw_path=raw,
        output_path=tmp_path / "symlink_refused.json",
    )
    assert symlinked["status"] == "REFUSE"
    encoded, archive, _raw, _config = _encoded(monkeypatch, tmp_path, campaign_id="switch-object")
    archive.write_bytes(b"mutated-after-encode")
    fake_eval = _write_json(tmp_path / "fake_eval.json", {"schema": gate.EXACT_EVAL_SCHEMA})
    report = tmp_path / "fake_report.txt"
    report.write_text("fake\n")
    post = gate.admit_post_eval(
        encode_receipt_path=encoded,
        eval_receipt_path=fake_eval,
        eval_report_path=report,
        output_path=tmp_path / "mutated_archive_post.json",
    )
    assert post["status"] == "REFUSE"


def test_fabricated_exact_eval_strings_cannot_admit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    encoded, archive, raw, _ = _encoded(monkeypatch, tmp_path)
    report = _write_json(
        tmp_path / "fabricated_report.json",
        {
            "schema": gate.EXACT_REPORT_SCHEMA,
            "d_seg": 0.0,
            "d_pose": 0.0,
        },
    )
    fabricated = _write_json(
        tmp_path / "fabricated_eval.json",
        {
            "schema": gate.EXACT_EVAL_SCHEMA,
            "status": "EXACT_EVAL_COMPLETE",
            "evaluation_entrypoint": "upstream/evaluate.py",
            "authority_axis": "contest-CPU",
            "archive": gate.artifact_identity(archive, label="archive"),
            "decoded_raw": gate.artifact_identity(raw, label="raw"),
            "report": gate.artifact_identity(report, label="report"),
        },
    )
    receipt = gate.admit_post_eval(
        encode_receipt_path=encoded,
        eval_receipt_path=fabricated,
        eval_report_path=report,
        output_path=tmp_path / "fabricated_refused.json",
    )
    assert receipt["candidate_admission"] is False
    assert any("OfficialEvaluationRunReceiptV1" in item for item in receipt["refusals"])


@pytest.mark.parametrize("mutated_object", ["archive", "report"])
def test_mutated_archive_or_report_object_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutated_object: str,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    encoded, archive, raw, _ = _encoded(monkeypatch, tmp_path)
    eval_receipt = _write_json(tmp_path / "strict_run.json", {"strict": True})
    report = tmp_path / "report.txt"
    report.write_text("original report\n")
    run = _DummyRun(
        archive=archive,
        raw=raw,
        report=report,
        d_seg=0.0,
        d_pose=0.0,
    )

    class Parser:
        @classmethod
        def from_receipt_bytes(cls, _payload: bytes):
            return run

    monkeypatch.setattr(gate, "OfficialEvaluationRunReceiptV1", Parser)
    if mutated_object == "archive":
        archive.write_bytes(b"archive changed after ENCODE")
    else:
        report.write_text("report changed after strict receipt\n")
    receipt = gate.admit_post_eval(
        encode_receipt_path=encoded,
        eval_receipt_path=eval_receipt,
        eval_report_path=report,
        output_path=tmp_path / f"{mutated_object}_refused.json",
    )
    assert receipt["status"] == "REFUSE"
    assert receipt["candidate_admission"] is False


def test_empty_not_killed_prose_or_missing_integration_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    encoded, archive, raw, _ = _encoded(monkeypatch, tmp_path)
    eval_receipt = _write_json(tmp_path / "run.json", {"strict": True})
    report = tmp_path / "report.txt"
    report.write_text("report\n")
    run = _DummyRun(
        archive=archive,
        raw=raw,
        report=report,
        d_seg=0.0,
        d_pose=0.0,
    )

    class Parser:
        @classmethod
        def from_receipt_bytes(cls, _payload: bytes):
            return run

    monkeypatch.setattr(gate, "OfficialEvaluationRunReceiptV1", Parser)
    blocker = _write_json(
        tmp_path / "invalid_blocker.json",
        {
            "schema": gate.BLOCKER_SCHEMA,
            "campaign_id": "campaign-a",
            "owner": "owner",
            "missing_hooks": list(gate.INTEGRATION_HOOKS),
            "not_killed": [],
            "verdict_scope": {"level": "FORMULATION", "name": "narrow"},
            "next_executable_action": ["python", "next.py"],
            "exact_eval_receipt_sha256": gate.artifact_identity(
                eval_receipt,
                label="eval",
            )["sha256"],
            "prose": "prose is not authority",
        },
    )
    receipt = gate.admit_post_eval(
        encode_receipt_path=encoded,
        eval_receipt_path=eval_receipt,
        eval_report_path=report,
        blocker_path=blocker,
        output_path=tmp_path / "blocker_refused.json",
    )
    assert receipt["status"] == "REFUSE"
    assert receipt["learning_admission"] is False
    missing = gate.admit_post_eval(
        encode_receipt_path=encoded,
        eval_receipt_path=eval_receipt,
        eval_report_path=report,
        output_path=tmp_path / "missing_integration_refused.json",
    )
    assert missing["status"] == "REFUSE"


def test_uncompetitive_g57_row_cannot_reach_promotion_with_axis_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path, target=0.172)
    post_path, archive, _raw, _config = _post_eval(
        monkeypatch,
        tmp_path,
        d_seg=0.17946555,
        d_pose=45.10546494,
    )
    post = gate.reopen_receipt(post_path)[1]
    assert post["status"] == "LEARNING_ONLY"
    pre_public_path = tmp_path / "pre_public_refused.json"
    pre_public = gate.admit_pre_public_closure(
        post_eval_receipt_path=post_path,
        asserted_archive_path=archive,
        output_path=pre_public_path,
    )
    assert pre_public["status"] == "REFUSE"
    fake_axis_only = _write_json(
        tmp_path / "axis_only.json",
        {
            "schema": gate.PUBLIC_AUTH_SCHEMA,
            "authority_axis": "contest-CPU",
        },
    )
    promotion = gate.admit_pre_promotion(
        pre_public_receipt_path=pre_public_path,
        public_auth_receipt_path=fake_axis_only,
        output_path=tmp_path / "promotion_refused.json",
    )
    assert promotion["candidate_admission"] is False


def test_retrospective_g57_is_never_launcher_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    request = _write_json(tmp_path / "g57_request.json", {"schema": "g57.request"})
    old_receipt = _write_json(tmp_path / "g57_receipt.json", {"schema": "g57.receipt"})
    path = tmp_path / "retrospective.json"
    receipt = gate.audit_g57_retrospective(
        campaign_id="g57",
        requested_representation=gate.DIRECT_TASK_LAYERED_CONTROL,
        repo_root=tmp_path,
        g57_request_path=request,
        g57_receipt_path=old_receipt,
        output_path=path,
    )
    assert receipt["mode"] == gate.RETROSPECTIVE_ONLY
    assert receipt["candidate_admission"] is False
    with pytest.raises(
        gate.AdversarialGateError,
        match="does not terminate at CAMPAIGN_SEAL",
    ):
        gate.require_live_admission_receipt(
            path,
            expected_stage=gate.PRE_ENCODE,
            expected_repo_root=tmp_path,
        )


def test_real_g58_fixture_identity_reopens_but_production_outer_chain_is_owed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tac.witness_dsl.tests import (
        test_taskspace_selected_preimage_operand_adapter_v1 as g58_tests,
    )

    _patch_frontier(monkeypatch, tmp_path)
    fixture = g58_tests._fixture(tmp_path / "g58-fixture")
    adapter = g58_tests._adapter(fixture)
    identity = tmp_path / "g58-fixture-identity.json"
    adapter.publish_pre_encode_identity_receipt(identity)
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    receipt = gate.admit_pre_encode(
        campaign_seal_path=_seal(tmp_path),
        producer_config_path=config,
        g58_identity_receipt_path=identity,
        output_path=tmp_path / "fixture_refused.json",
    )
    assert receipt["evidence"]["adapter_schema_reopened"] is True
    assert "G58_PRODUCTION_TERMINAL_CHAIN_AND_OUTER_PROOF_OWED" in receipt["refusals"]
    assert receipt["candidate_admission"] is False


def test_strict_g58_learned_custody_reaches_separate_program_config_blocker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    seal = _seal(tmp_path)
    seal_identity = gate.artifact_identity(seal, label="seal")
    identity = _write_json(tmp_path / "identity.json", {"schema": gate.G58_IDENTITY_SCHEMA})
    terminal = _write_json(tmp_path / "terminal.json", {"terminal": True})
    outer = _write_json(tmp_path / "outer.json", {"outer": True})
    config = _write_json(tmp_path / "config.json", {"config": True})

    def reopen(**_kwargs):
        return {
            "schema": gate.PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
            "status": "ADMIT",
            "campaign_receipt": seal_identity,
            "learned_factor_section_ids": ["learned.factor"],
            "analytic_only": False,
        }

    monkeypatch.setattr(
        gate,
        "reopen_program_residual_production_pre_encode_evidence",
        reopen,
    )
    receipt = gate.admit_pre_encode(
        campaign_seal_path=seal,
        producer_config_path=config,
        g58_identity_receipt_path=identity,
        g58_terminal_stage_chain_path=terminal,
        g58_outer_proof_path=outer,
        output_path=tmp_path / "learned_refused.json",
    )
    assert "G58_LEARNED_FACTOR_DECODER_SOURCE_CUSTODY_NOT_REOPENED" not in receipt["refusals"]
    assert "PROGRAM_PRODUCER_CONFIG_SCHEMA_OWED" in receipt["refusals"]


def test_analytic_g58_still_refuses_until_real_program_runner_config_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    seal = _seal(tmp_path)
    seal_identity = gate.artifact_identity(seal, label="seal")
    identity = _write_json(tmp_path / "identity.json", {"schema": gate.G58_IDENTITY_SCHEMA})
    terminal = _write_json(tmp_path / "terminal.json", {"terminal": True})
    outer = _write_json(tmp_path / "outer.json", {"outer": True})
    config = _write_json(tmp_path / "config.json", {"looks_plausible": True})

    monkeypatch.setattr(
        gate,
        "reopen_program_residual_production_pre_encode_evidence",
        lambda **_kwargs: {
            "schema": gate.PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
            "status": "ADMIT",
            "campaign_receipt": seal_identity,
            "learned_factor_section_ids": [],
            "analytic_only": True,
        },
    )
    receipt = gate.admit_pre_encode(
        campaign_seal_path=seal,
        producer_config_path=config,
        g58_identity_receipt_path=identity,
        g58_terminal_stage_chain_path=terminal,
        g58_outer_proof_path=outer,
        output_path=tmp_path / "program_config_owed.json",
    )
    assert receipt["status"] == "REFUSE"
    assert "PROGRAM_PRODUCER_CONFIG_SCHEMA_OWED" in receipt["refusals"]


def test_research_only_public_receipts_cannot_mint_promotion_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    post_path, archive, _raw, _config = _post_eval(monkeypatch, tmp_path)
    pre_public_path = tmp_path / "pre_public.json"
    pre_public = gate.admit_pre_public_closure(
        post_eval_receipt_path=post_path,
        asserted_archive_path=archive,
        output_path=pre_public_path,
    )
    assert pre_public["status"] == "ADMIT"
    bundle = _write_json(tmp_path / "strict_research_bundle.json", {"research_only": True})
    monkeypatch.setattr(
        gate,
        "_verify_public_auth",
        lambda **_kwargs: {"typed_research_receipts_reopened": True},
    )
    receipt = gate.admit_pre_promotion(
        pre_public_receipt_path=pre_public_path,
        public_auth_receipt_path=bundle,
        output_path=tmp_path / "authority_owed.json",
    )
    assert receipt["status"] == "REFUSE"
    assert receipt["candidate_admission"] is False
    assert "AUTHORITY_EMITTER_OWED_SEALED_C0B_OR_CONTEST_OWNER" in receipt["refusals"]


def test_prior_cycle_receipt_config_representation_and_root_cannot_be_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    receipt_path, config = _admitted_pre_encode(monkeypatch, tmp_path, campaign_id="cycle-a")
    with pytest.raises(gate.AdversarialGateError, match="another campaign cycle"):
        gate.require_live_admission_receipt(
            receipt_path,
            expected_stage=gate.PRE_ENCODE,
            expected_campaign_id="cycle-b",
            expected_repo_root=tmp_path,
            expected_representation=gate.PROGRAM_RESIDUAL_LAYERED,
            expected_config_path=config,
        )
    changed_config = tmp_path / "changed-config.json"
    changed_config.write_text('{"config":"b"}\n')
    with pytest.raises(gate.AdversarialGateError, match="producer config mismatch"):
        gate.require_live_admission_receipt(
            receipt_path,
            expected_stage=gate.PRE_ENCODE,
            expected_campaign_id="cycle-a",
            expected_repo_root=tmp_path,
            expected_representation=gate.PROGRAM_RESIDUAL_LAYERED,
            expected_config_path=changed_config,
        )
    with pytest.raises(gate.AdversarialGateError, match="representation mismatch"):
        gate.require_live_admission_receipt(
            receipt_path,
            expected_stage=gate.PRE_ENCODE,
            expected_campaign_id="cycle-a",
            expected_repo_root=tmp_path,
            expected_representation=gate.DIRECT_TASK_LAYERED_CONTROL,
            expected_config_path=config,
        )
    other = tmp_path / "other"
    with pytest.raises(gate.AdversarialGateError, match="different repository root"):
        gate.require_live_admission_receipt(
            receipt_path,
            expected_stage=gate.PRE_ENCODE,
            expected_campaign_id="cycle-a",
            expected_repo_root=other,
        )


def test_direct_research_launchers_do_not_create_a_candidate_gate_cycle() -> None:
    from tools import build_taskspace_layered_public_closure as closure_tool
    from tools.run_taskspace_lossy_selected_plane_codec_n600 import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "config.json",
            "--output-root",
            "out",
        ]
    )
    assert args.config == Path("config.json")
    source = Path(closure_tool.__file__).read_text(encoding="utf-8")
    assert "--adversarial-pre-encode-receipt" not in source
    assert "--adversarial-pre-public-receipt" not in source
    assert "stage_exact_eval(" in source
    assert "--adversarial-pre-promotion-receipt" in source


def test_direct_exact_eval_staging_precedes_post_eval_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from tools import build_taskspace_layered_public_closure as closure_tool

    config = tmp_path / "config.json"
    config.write_text("{}\n")
    preview = tmp_path / "preview.zip"
    preview.write_bytes(b"preview")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        closure_tool,
        "build_preview",
        lambda _config: {
            "archive_preview": {"path": str(preview)},
            "receipt_path": str(tmp_path / "build-receipt.json"),
        },
    )

    def stage_exact_eval(path: Path, receipt: object):
        observed.update(path=path, receipt=receipt)
        return tmp_path / "submission" / "archive.zip", {"status": "STAGED"}

    monkeypatch.setattr(closure_tool, "stage_exact_eval", stage_exact_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(closure_tool.__file__),
            "--config",
            str(config),
            "--stage-exact-eval",
        ],
    )
    assert closure_tool.main() == 0
    assert observed["path"] == preview
    assert '"exact_eval_staging_receipt": {"status": "STAGED"}' in capsys.readouterr().out


def test_live_guard_rejects_a_stage_skipping_receipt_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    pre_encode_path, _ = _admitted_pre_encode(monkeypatch, tmp_path)
    predecessor_identity, predecessor = gate.reopen_receipt(pre_encode_path)
    snapshot = gate._snapshot_from_receipt(predecessor)
    body = gate._receipt_common(
        stage=gate.PRE_PROMOTION,
        mode=gate.LIVE,
        campaign_id=predecessor["campaign_id"],
        representation=predecessor["requested_representation"],
        repo_root=predecessor["repo_root"],
        frontier=snapshot,
        predecessor={
            **predecessor_identity,
            "body_sha256": predecessor["body_sha256"],
            "stage": predecessor["stage"],
        },
    )
    body.update(status="ADMIT", candidate_admission=True)
    skipped = tmp_path / "stage-skipping-pre-promotion.json"
    gate._write_receipt(skipped, body)
    with pytest.raises(
        gate.AdversarialGateError,
        match="skipped or repeated",
    ):
        gate.require_live_admission_receipt(
            skipped,
            expected_stage=gate.PRE_PROMOTION,
            expected_campaign_id=predecessor["campaign_id"],
            expected_repo_root=tmp_path,
            expected_representation=gate.PROGRAM_RESIDUAL_LAYERED,
        )


def test_receipt_publication_is_write_once_not_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_frontier(monkeypatch, tmp_path)
    path = _seal(tmp_path)
    original = path.read_bytes()
    path.write_text('{"mutated":true}\n')
    with pytest.raises(gate.AdversarialGateError, match="write-once"):
        gate.seal_campaign(
            campaign_id="campaign-a",
            requested_representation=gate.PROGRAM_RESIDUAL_LAYERED,
            repo_root=tmp_path,
            expected_repo_root=tmp_path,
            output_path=path,
        )
    assert path.read_bytes() != original


def test_artifact_verification_uses_one_descriptor_stable_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"stable":true}\n')
    row = gate.artifact_identity(path, label="artifact")
    original = gate._stable_read_with_identity
    calls = 0

    def one_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            path.write_text('{"swapped":true}\n')
            raise AssertionError("identity and payload were acquired by separate reads")
        return original(*args, **kwargs)

    monkeypatch.setattr(gate, "_stable_read_with_identity", one_read)
    identity, payload = gate._verify_artifact(row, label="artifact")
    assert calls == 1
    assert identity == row
    assert payload == b'{"stable":true}\n'
