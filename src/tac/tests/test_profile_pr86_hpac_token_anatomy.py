from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "profile_pr86_hpac_token_anatomy.py"
PR86_DIR = REPO / "experiments/results/public_pr86_intake_20260504_codex"
PR85_PROFILE = (
    REPO
    / "experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json"
)
PR86_REPLAY_DIR = (
    REPO
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "profile_pr86_hpac_token_anatomy_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pr86_like_zip(path: Path, *, extra: bool = False) -> None:
    payloads = {
        "master.pt.gz": b"master",
        "slave.pt.gz": b"slave",
        "hpac.pt.ppmd": b"hpac",
        "tokens.bin": b"\x00\x00\x00\x00",
        "meta.pt": b"meta",
    }
    if extra:
        payloads["sidecar.bin"] = b"not allowed"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in payloads.items():
            zf.writestr(name, data)


def test_member_contract_is_deterministic_for_exact_member_set(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    _write_pr86_like_zip(archive)

    first = script.inspect_archive_members(archive, expected_bytes=None, expected_sha256=None)
    second = script.inspect_archive_members(archive, expected_bytes=None, expected_sha256=None)

    assert first == second
    assert first["sidecar_assumption_status"] == "passed_exact_required_member_set"
    assert first["promotable_member_contract"] is True
    assert first["unexpected_members"] == []
    assert [row["name"] for row in first["members"]] == list(script.REQUIRED_PR86_MEMBERS)


def test_member_contract_fails_closed_on_unexpected_sidecar(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    _write_pr86_like_zip(archive, extra=True)

    report = script.inspect_archive_members(archive, expected_bytes=None, expected_sha256=None)

    assert report["sidecar_assumption_status"] == "failed_closed"
    assert report["promotable_member_contract"] is False
    assert report["unexpected_members"] == ["sidecar.bin"]


def test_real_pr86_source_contract_classifies_submitted_tokens_as_raw() -> None:
    script = _load_script()

    contract = script.analyze_source_contract(PR86_DIR)

    token = contract["token_hpac_decode_contract"]
    assert token["training_objective"] == "residual_tokens"
    assert token["archive_compute_gt_tokens_call_present"] is True
    assert token["archive_write_tokens_second_arg"] == "gt"
    assert token["inflate_reconstructs_residuals"] is False
    assert token["submitted_archive_token_encoding"] == "raw_tokens"
    assert token["range_decoder_api_present"] is True
    assert token["categorical_perfect_false_present"] is True
    assert token["explicit_16384_grid_in_archive_or_inflate"] is False
    assert token["readme_mentions_16384_grid"] is True


def test_build_report_is_non_promotable_and_keys_current_replay_branch() -> None:
    script = _load_script()

    report = script.build_report(
        pr86_dir=PR86_DIR,
        archive=PR86_DIR / "archive.zip",
        pr85_profile=PR85_PROFILE,
        replay_dir=PR86_REPLAY_DIR,
        inspect_payloads=False,
    )

    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["promotable"] is False
    assert report["source_archive"]["bytes"] == 207579
    assert (
        report["source_archive"]["sha256"]
        == "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
    )
    assert (
        report["archive_member_contract"]["sidecar_assumption_status"]
        == "passed_exact_required_member_set"
    )
    assert report["current_exact_replay_status"]["status"] in {
        "archive_validator_whitelist_blocked",
        "score_json_present",
        "auth_log_present_without_score_json",
        "ARTIFACT_INFRA_FAILURE",
    }
    actions = report["recommended_next_actions_by_exact_replay_outcome"][
        "current_observed_replay_branch"
    ]
    assert actions
    assert all("dispatch" not in action.lower() for action in actions[:2])


def test_pr85_hpac_transplant_byte_math_is_reported() -> None:
    script = _load_script()

    report = script.build_report(
        pr86_dir=PR86_DIR,
        archive=PR86_DIR / "archive.zip",
        pr85_profile=PR85_PROFILE,
        replay_dir=PR86_REPLAY_DIR,
        inspect_payloads=False,
    )

    opportunities = {
        row["id"]: row for row in report["transplant_opportunities_onto_pr85"]
    }
    hpac = opportunities["hpac_reencode_pr85_mask_tokens"]
    assert hpac["drop_in_status"] == "not_drop_in"
    assert hpac["gross_byte_math"]["pr85_mask_segment_bytes"] == 159011
    assert hpac["gross_byte_math"]["pr86_hpac_tokens_meta_bytes"] == 143642
    assert hpac["gross_byte_math"]["gross_saved_bytes_if_same_contract"] == 15369
    assert hpac["score_claim"] is False
