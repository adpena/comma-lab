from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.ep725_lossless_xcodec_recode import inspect_source_zip, parse_ep725_lvls1
from tac.witness_dsl.ep725_population_global_recode_v2 import (
    SOURCE_ARCHIVE_SHA256,
    SOURCE_MEMBER_SHA256,
    parse_population_global_member,
)
from tools.materialize_ep725_population_global_recode_v2 import (
    DEFAULT_G20_CONTROL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_ARCHIVE,
    DEFAULT_SOURCE_RUNTIME,
    MaterializePopulationGlobalRecodeError,
    materialize,
)

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = DEFAULT_OUTPUT_DIR / "receipt.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_materializer_refuses_without_reviewed_execution() -> None:
    args = argparse.Namespace(
        execute_reviewed=False,
        source_archive=DEFAULT_SOURCE_ARCHIVE,
        source_runtime=DEFAULT_SOURCE_RUNTIME,
        g20_control=DEFAULT_G20_CONTROL,
        output_dir=DEFAULT_OUTPUT_DIR,
        resume_from=None,
    )
    with pytest.raises(MaterializePopulationGlobalRecodeError, match="execute-reviewed"):
        materialize(args)


def test_durable_real_n600_receipt_and_archive_are_exact() -> None:
    if not RECEIPT.is_file() or not DEFAULT_SOURCE_ARCHIVE.is_file():
        pytest.skip("reviewed G25 materialization or frozen SSD source is unavailable")
    receipt = json.loads(RECEIPT.read_bytes())
    assert receipt["schema"] == "tac.ep725_population_global_recode.v2"
    assert receipt["truth"] == {
        "candidate_claim": False,
        "contest_cpu_cuda_same_bytes_owed": True,
        "exact_eval_invoked": False,
        "full_n600_quantized_state_used": True,
        "full_n600_runtime_output_replay_owed": True,
        "pointer_moved": False,
        "promotion_eligible": False,
        "public_payload_reused": False,
        "research_only": True,
        "score_claim": False,
    }
    artifact_path = ROOT / receipt["artifact"]["path"]
    artifact = artifact_path.read_bytes()
    assert len(artifact) == receipt["artifact"]["bytes"] == 80_238
    assert _sha256(artifact) == receipt["artifact"]["sha256"]
    assert receipt["complete_object_controls"]["selected_control_name"] == "g25_v2"
    assert receipt["exact_delta"]["versus_g20_archive_bytes"] == -789
    assert receipt["exact_delta"]["versus_source_archive_bytes"] == -3_600

    with __import__("zipfile").ZipFile(__import__("io").BytesIO(artifact)) as archive:
        assert archive.namelist() == ["0.bin"]
        selected = parse_population_global_member(archive.read("0.bin"))
        assert archive.testzip() is None
    source_bytes = DEFAULT_SOURCE_ARCHIVE.read_bytes()
    source_profile = inspect_source_zip(
        source_bytes,
        expected_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_member_sha256=SOURCE_MEMBER_SHA256,
    )
    source = parse_ep725_lvls1(source_profile.member_bytes, require_source_form=True)
    assert selected.base_order == source.base_order
    assert all(np.array_equal(selected.base_quantized[name], source.base_quantized[name]) for name in source.base_order)
    assert np.array_equal(selected.code_quantized, source.code_quantized)
    assert selected.pose_bytes == source.pose_bytes


def test_receipt_preserves_nonadditive_whole_object_hyperedges() -> None:
    if not RECEIPT.is_file():
        pytest.skip("reviewed G25 materialization is unavailable")
    receipt = json.loads(RECEIPT.read_bytes())
    assert receipt["search"]["selection_surface"] == "exact complete archive.zip bytes"
    assert receipt["search"]["converged_whole_cycle"] is True
    assert receipt["search"]["points_measured"] == 6_669
    assert len(receipt["search"]["stages"]) == 20
    assert receipt["search"]["stages"][-1]["converged_whole_cycle"] is True
    for hyperedge in receipt["interaction_hyperedges"]:
        assert hyperedge["effect_observation_kind"] == "INDIVISIBLE_HYPEREDGE"
        assert hyperedge["additive_attribution_forbidden"] is True
        corners = hyperedge["corners"]
        expected = (
            corners["11"]["archive_bytes"]
            - corners["10"]["archive_bytes"]
            - corners["01"]["archive_bytes"]
            + corners["00"]["archive_bytes"]
        )
        assert hyperedge["interaction_archive_bytes"] == expected
        assert all(len(corners[name]["archive_sha256"]) == 64 for name in ("00", "10", "01", "11"))
    action = receipt["substitutive_action"]
    assert action["action_semantics"] == "REQUANTIZE_STORAGE"
    assert action["section_marginal_attribution_forbidden"] is True
    assert action["exact_effect"]["delta_archive_bytes"] == -789
    assert (
        action["consumer_contract"]["g19_v1_receipt_ingest_status"]
        == "SCHEMA_EXTENSION_REQUIRED_G19_V1_ACCEPTS_G20_ONLY"
    )


def test_preserved_stage_checkpoints_form_a_resume_chain() -> None:
    checkpoint_dir = DEFAULT_OUTPUT_DIR / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("stage_cycle*.json"))
    if not checkpoints:
        pytest.skip("reviewed G25 materialization is unavailable")
    stages = [json.loads(path.read_bytes()) for path in checkpoints]
    assert len(stages) == 20
    for left, right in itertools.pairwise(stages):
        state = left["next_resume_state"]
        assert state["config"] == right["before"]["config"]
        assert state["points_measured"] < right["next_resume_state"]["points_measured"]
    assert stages[-1]["converged_whole_cycle"] is True
