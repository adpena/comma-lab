# SPDX-License-Identifier: MIT
"""Tests for the V8 learned-compression Faiss scaffold."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from tac.substrates.v8_learned_compression_faiss.archive import (
    V8ArchiveError,
    build_raw_frame_archive,
    parse_v8_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
TOOLS_DIR = REPO_ROOT / "tools"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _trainer():
    return importlib.import_module("train_substrate_v8_learned_compression_faiss")


def _load_inflate_module():
    path = REPO_ROOT / "submissions" / "v8_learned_compression_faiss" / "inflate.py"
    spec = importlib.util.spec_from_file_location("_test_v8_faiss_inflate", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_v8_faiss_inflate"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _recipe_path() -> Path:
    return (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/"
        "substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml"
    )


def _load_recipe_yaml() -> dict:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_recipe_path().read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_trainer_smoke_writes_fail_closed_manifest(tmp_path: Path) -> None:
    trainer = _trainer()
    args = trainer._build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--smoke",
            "--num-pairs",
            "4",
            "--categorical-groups",
            "8",
            "--codebook-size",
            "16",
        ]
    )
    rc = trainer._smoke_main(args)
    assert rc == 0
    manifest_path = tmp_path / "v8_smoke_results.json"
    assert manifest_path.is_file()
    text = manifest_path.read_text(encoding="utf-8")
    assert "score_claim" in text
    manifest = trainer.build_smoke_manifest(args, observed_at_utc="2026-05-20T00:00:00Z")
    assert manifest["research_only"] is True
    assert manifest["dispatch_enabled"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["local_implementation_status"] == "byte_closed_fixture_export_available"
    assert manifest["score_aware_roundtrip_contract"]["score_claim"] is False
    assert manifest["score_aware_roundtrip_contract"]["promotion_eligible"] is False
    assert manifest["score_aware_roundtrip_contract"]["ready_for_exact_eval_dispatch"] is False
    assert manifest["remaining_promotion_blockers"] == [
        "real_contest_video_scorer_training_not_run",
        "exact_cuda_auth_eval_missing",
        "catalog_324_tier_c_validation_missing",
        "modal_dispatch_recipe_remains_research_only_and_disabled",
    ]


def test_trainer_manifest_uses_corrected_canonical_helpers(tmp_path: Path) -> None:
    trainer = _trainer()
    args = trainer._build_parser().parse_args(["--output-dir", str(tmp_path)])
    manifest = trainer.build_smoke_manifest(args, observed_at_utc="2026-05-20T00:00:00Z")
    helpers = manifest["canonical_helpers"]
    assert helpers["score_aware"] == (
        "tac.substrates.score_aware_common.score_pair_components_dispatch"
    )
    assert helpers["pq_mi"] == (
        "tac.optimization.faiss_ivf_pq_atw_channel.compute_pq_mi_verdict"
    )
    assert helpers["provenance_archive_member"] == (
        "tac.provenance.build_provenance_for_archive_member"
    )
    helper_text = " ".join(str(value) for value in helpers.values())
    assert "tac.substrates._shared.score_aware_common" not in helper_text
    assert "build_provenance_for_contest_archive_byte_member" not in helper_text


def test_trainer_full_mode_writes_local_fixture_without_score_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _trainer()
    monkeypatch.setenv("V8_FAISS_TRAINER_MODE", "full")
    args = trainer._build_parser().parse_args(["--output-dir", str(tmp_path)])
    assert trainer._resolve_trainer_mode(args) == "full"
    assert trainer._full_main(args) == 0
    manifest = (tmp_path / "v8_smoke_results.json").read_text(encoding="utf-8")
    assert "local_byte_closed_export" in manifest
    assert "learned_compression_export" in manifest
    assert "score_claim" in manifest
    assert (tmp_path / "v8_fixture.v8f").is_file()
    assert (tmp_path / "v8_fixture.raw").is_file()


def test_trainer_full_mode_accepts_supplied_fixture_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _trainer()
    monkeypatch.setenv("V8_FAISS_TRAINER_MODE", "full")
    missing_args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "missing"), "--require-fixture-payload"]
    )
    with pytest.raises(FileNotFoundError, match="fixture-payload was not supplied"):
        trainer._full_main(missing_args)

    payload = bytes(range(24))
    payload_path = tmp_path / "fixture.raw"
    payload_path.write_bytes(payload)
    args = trainer._build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "supplied"),
            "--fixture-frames",
            "2",
            "--fixture-height",
            "2",
            "--fixture-width",
            "2",
            "--fixture-payload",
            str(payload_path),
            "--require-fixture-payload",
        ]
    )
    assert trainer._full_main(args) == 0
    manifest = json.loads(
        (tmp_path / "supplied" / "v8_smoke_results.json").read_text(encoding="utf-8")
    )
    source = manifest["local_byte_closed_export"]["raw_fixture_source"]
    assert source["kind"] == "supplied_raw_rgb_fixture"
    assert source["bytes"] == len(payload)
    _header, decoded = parse_v8_archive((tmp_path / "supplied" / "v8_fixture.v8f").read_bytes())
    assert decoded == payload


def test_trainer_local_export_records_categorical_and_hyperprior_contract(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    args = trainer._build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--fixture-frames",
            "6",
            "--fixture-height",
            "2",
            "--fixture-width",
            "2",
            "--categorical-groups",
            "3",
            "--codebook-size",
            "8",
        ]
    )
    assert trainer._smoke_main(args) == 0
    manifest = json.loads((tmp_path / "v8_smoke_results.json").read_text(encoding="utf-8"))
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "deterministic_categorical_posterior_codewords" in manifest[
        "implemented_local_blockers"
    ]
    export = manifest["local_byte_closed_export"]
    assert export["categorical_codeword_count"] == 6
    assert export["categorical_groups"] == 3
    assert export["codebook_size"] == 8
    assert len(export["scale_hyperprior"]) == 3
    assert export["header"]["payload_bytes"] == 6 * 2 * 2 * 3
    assert "exact_cuda_auth_eval_missing" in manifest["remaining_promotion_blockers"]
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from tac.substrates.v8_learned_compression_faiss.smoke import (
        V8_EXPORT_MAGIC,
        parse_v8_export_payload,
    )

    learned_path = Path(manifest["learned_compression_export"]["payload_path"])
    learned_bytes = learned_path.read_bytes()
    assert learned_bytes.startswith(V8_EXPORT_MAGIC)
    learned_header = parse_v8_export_payload(learned_bytes)
    assert learned_header["custody"]["score_claim"] is False
    assert learned_header["custody"]["promotion_eligible"] is False
    assert learned_header["metrics"]["eval_roundtrip_hook"]["scorer_loads"] is False
    assert any(blob["name"] == "categorical_codewords" for blob in learned_header["blobs"])
    assert any(blob["name"] == "scale_hyperprior_scales" for blob in learned_header["blobs"])


def test_operator_recipe_is_research_only_and_disabled() -> None:
    recipe = _recipe_path()
    text = recipe.read_text(encoding="utf-8")
    assert "research_only: true" in text
    assert "dispatch_enabled: false" in text
    assert "exact_cuda_auth_eval_missing" in text
    assert "required_input_files_trainer: experiments/train_substrate_v8_learned_compression_faiss.py" in text


def test_operator_recipe_structured_non_promotional_contract() -> None:
    recipe = _load_recipe_yaml()
    assert recipe["research_only"] is True
    assert recipe["dispatch_enabled"] is False
    assert set(recipe["target_modes"]) == {"research_substrate"}
    assert "contest_exact_eval" not in recipe["target_modes"]
    assert recipe["predicted_band_kind"] == "predicted_score_band"
    assert recipe["predicted_band_axis"] == "contest-CPU"
    assert recipe["predicted_band_validation_status"] == "pending_post_training"

    dispatch_blockers = set(recipe["dispatch_blockers"])
    assert {
        "real_contest_video_scorer_training_not_run",
        "exact_cuda_auth_eval_missing",
        "catalog_324_tier_c_validation_missing",
        "modal_dispatch_recipe_remains_research_only_and_disabled",
    }.issubset(dispatch_blockers)

    promotion_blockers = set(recipe["pre_promotion_blockers"])
    assert {
        "v8_learned_byte_closed_archive_missing",
        "v8_contest_runtime_custody_missing",
        "v8_exact_cuda_auth_eval_missing",
        "v8_exact_cpu_cuda_axis_pair_missing",
        "v8_score_claim_provenance_missing",
    }.issubset(promotion_blockers)


def test_operator_authorize_refuses_even_if_dispatch_flag_is_flipped_without_promotion_evidence() -> None:
    import operator_authorize as oa

    recipe = oa._load_recipe("substrate_v8_learned_compression_faiss_modal_a100_smoke")
    refusal = oa._recipe_dispatch_refusal(recipe)
    assert refusal is not None
    assert "dispatch_enabled=false" in refusal
    assert "exact_cuda_auth_eval_missing" in refusal

    recipe.raw["dispatch_enabled"] = True
    recipe.raw["dispatch_blockers"] = []
    refusal_after_bad_flip = oa._recipe_dispatch_refusal(recipe)
    assert refusal_after_bad_flip is not None
    assert "pre_promotion_blockers still declared" in refusal_after_bad_flip
    assert "v8_exact_cuda_auth_eval_missing" in refusal_after_bad_flip


def test_local_pre_deploy_check_does_not_certify_v8_as_dispatchable() -> None:
    import local_pre_deploy_check as lpdc

    ok, message = lpdc.check_recipe_status_consistent_with_trainer_state(
        REPO_ROOT / "experiments/train_substrate_v8_learned_compression_faiss.py",
        "substrate_v8_learned_compression_faiss_modal_a100_smoke",
    )
    assert ok is False
    assert "trainer `_full_main` is implemented but recipe is still non-dispatchable" in message
    assert "do not use local_pre_deploy_check as dispatch proof" in message


def test_predicted_band_audit_treats_v8_recipe_as_research_only_not_score_authority() -> None:
    from tac.optimization.tier_c_density_post_training_validator import (
        validate_recipe_predicted_band,
    )

    verdict = validate_recipe_predicted_band(_recipe_path())
    assert verdict.has_predicted_band is True
    assert verdict.is_valid is True
    assert verdict.is_research_only is True
    assert verdict.validation_status == "research_only"


def test_readiness_assessment_consumes_disabled_v8_recipe() -> None:
    import asymptotic_pursuit_candidate_readiness_assessment as readiness

    candidate = readiness.assess_candidate(
        "v8_learned_compression_faiss",
        repo_root=REPO_ROOT,
    )
    assert candidate.recipe_path == _recipe_path()
    assert candidate.research_only is True
    assert candidate.dispatch_enabled is False
    assert "exact_cuda_auth_eval_missing" in candidate.dispatch_blockers
    assert candidate.predicted_band_kind == "predicted_score_band"
    assert candidate.predicted_band_axis == "contest-CPU"
    assert candidate.predicted_band_validation_status == "pending_post_training"
    assert "RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL" in (
        candidate.blocking_issues
    )
    assert "RECIPE_DISPATCH_BLOCKER:exact_cuda_auth_eval_missing" in (
        candidate.blocking_issues
    )
    assert candidate.readiness_verdict != "READY"


def test_v8_archive_fixture_parses_and_inflates_deterministically(tmp_path: Path) -> None:
    mod = _load_inflate_module()
    payload = bytes(range(12))
    archive = build_raw_frame_archive(payload, frame_count=1, height=2, width=2, channels=3)
    assert archive == build_raw_frame_archive(
        payload, frame_count=1, height=2, width=2, channels=3
    )
    parsed, decoded = parse_v8_archive(archive)
    assert decoded == payload
    assert parsed.frame_count == 1
    assert parsed.height == 2
    assert parsed.width == 2
    assert parsed.channels == 3

    header = mod.parse_v8_archive_header(archive)
    assert header["magic"] == "V8FAISS1"
    assert header["research_only"] is True
    assert header["dispatch_enabled"] is False
    assert header["score_claim"] is False
    assert header["promotion_eligible"] is False
    assert header["ready_for_exact_eval_dispatch"] is False

    with mock.patch.dict(os.environ, {"PACT_INFLATE_DEVICE": "cpu"}, clear=False):
        assert mod.select_inflate_device() == "cpu"
    src = tmp_path / "x"
    dst = tmp_path / "x.raw"
    src.write_bytes(archive)
    with mock.patch.dict(os.environ, {"PACT_INFLATE_DEVICE": "cpu"}, clear=False):
        mod.inflate(src, dst)
    assert dst.read_bytes() == payload


def test_v8_archive_malformed_inputs_fail_closed(tmp_path: Path) -> None:
    mod = _load_inflate_module()
    payload = bytes(range(12))
    archive = bytearray(
        build_raw_frame_archive(payload, frame_count=1, height=2, width=2, channels=3)
    )

    with pytest.raises(ValueError, match="too short"):
        mod.parse_v8_archive_header(b"V8FAISS1")
    with pytest.raises(ValueError, match="magic mismatch"):
        mod.parse_v8_archive_header(b"WRONG000" + bytes(archive[8:]))

    bad_version = bytearray(archive)
    bad_version[9] = 2
    with pytest.raises(V8ArchiveError, match="unsupported V8 archive version"):
        parse_v8_archive(bytes(bad_version))

    bad_payload_len = bytearray(archive)
    bad_payload_len[20:24] = (999).to_bytes(4, "big")
    with pytest.raises(V8ArchiveError, match="payload length"):
        parse_v8_archive(bytes(bad_payload_len))

    bad_sha = bytearray(archive)
    bad_sha[-1] ^= 0xFF
    with pytest.raises(V8ArchiveError, match="sha256 mismatch"):
        parse_v8_archive(bytes(bad_sha))

    src = tmp_path / "bad.v8f"
    dst = tmp_path / "bad.raw"
    src.write_bytes(bytes(bad_sha))
    with pytest.raises(V8ArchiveError):
        mod.inflate(src, dst)
    assert not dst.exists()


def test_inflate_sh_consumes_safe_file_list_and_rejects_traversal(tmp_path: Path) -> None:
    payload = b"\x01\x02\x03\x04"
    archive = build_raw_frame_archive(payload, frame_count=1, height=2, width=2, channels=1)
    archive_dir = tmp_path / "archive"
    output_dir = tmp_path / "out"
    archive_dir.mkdir()
    (archive_dir / "tiny.v8f").write_bytes(archive)
    file_list = tmp_path / "files.txt"
    file_list.write_text("tiny.mkv\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "submissions/v8_learned_compression_faiss/inflate.sh"),
            str(archive_dir),
            str(output_dir),
            str(file_list),
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert (output_dir / "tiny.raw").read_bytes() == payload

    bad_file_list = tmp_path / "bad_files.txt"
    bad_file_list.write_text("../tiny.mkv\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "submissions/v8_learned_compression_faiss/inflate.sh"),
            str(archive_dir),
            str(output_dir),
            str(bad_file_list),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unsafe V8 file_list entry" in result.stderr


def test_v8_runtime_stays_non_submission_ready_after_decoder_lands() -> None:
    mod = _load_inflate_module()
    payload = b"\x00\x01\x02"
    archive = build_raw_frame_archive(payload, frame_count=1, height=1, width=1, channels=3)
    header = mod.parse_v8_archive_header(archive)
    assert header["research_only"] is True
    assert header["dispatch_enabled"] is False
    assert header["score_claim"] is False
    assert header["promotion_eligible"] is False
    assert header["ready_for_exact_eval_dispatch"] is False

    source = (REPO_ROOT / "submissions/v8_learned_compression_faiss/inflate.py").read_text(
        encoding="utf-8"
    )
    forbidden_runtime_imports = ("tac.scorer", "upstream.modules", "requests", "urllib")
    for token in forbidden_runtime_imports:
        assert token not in source

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tac.substrates.v8_learned_compression_faiss.archive; "
                "print('torch' in sys.modules)"
            ),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
    )
    assert import_check.stdout.strip() == "False"
