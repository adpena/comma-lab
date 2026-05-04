from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.pr86_hpac_codec import (
    EXPECTED_PR86_MEMBERS,
    EXPECTED_PR86_TOKENS_SHA256,
    DEFAULT_PR86_ARCHIVE,
    DEFAULT_PR86_MERGED_SOURCE_DIR,
    HPACMini,
    Pr86ArchiveContract,
    Pr86HpacReplayError,
    analyze_pr86_current_source_context,
    collect_dependency_report,
    decode_symbols_hpac_with_prev_context,
    decode_tokens_hpac,
    encode_symbols_hpac_with_prev_context,
    default_source_artifact_paths,
    encode_tokens_hpac,
    read_pr86_archive,
    run_pr86_hpac_probability_variant_matrix,
    run_pr86_hpac_replay,
    sha256_bytes,
    supported_hpac_probability_variant_names,
)


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "replay_pr86_hpac_tokens.py"


def _load_cli_script():
    spec = importlib.util.spec_from_file_location("replay_pr86_hpac_tokens_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _torch_bytes(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _gzip_torch(payload) -> bytes:
    return gzip.compress(_torch_bytes(payload), compresslevel=9)


def _synthetic_contract(tokens_blob: bytes) -> Pr86ArchiveContract:
    return Pr86ArchiveContract(
        expected_archive_bytes=None,
        expected_archive_sha256=None,
        expected_member_bytes={},
        expected_tokens_sha256=sha256_bytes(tokens_blob),
    )


def _write_synthetic_hpac_archive(
    tmp_path: Path,
    *,
    corrupt_tokens_sha: bool = False,
    probability_variant: str = "source_float64_perfect_false",
) -> tuple[Path, Pr86ArchiveContract]:
    pyppmd = pytest.importorskip("pyppmd")
    pytest.importorskip("constriction")
    torch.manual_seed(1234)
    model = HPACMini(num_pairs=2, P=2, delta=1, ch=4, d_film=2, use_spm=False).eval()
    tokens = np.array(
        [
            [
                [0, 1, 2, 3],
                [4, 0, 1, 2],
                [3, 4, 0, 1],
                [2, 3, 4, 0],
            ],
            [
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
            ],
        ],
        dtype=np.uint8,
    )
    tokens_blob, _encode_report = encode_tokens_hpac(
        model,
        tokens,
        P=2,
        delta=1,
        probability_variant=probability_variant,
    )
    contract = _synthetic_contract(tokens_blob)
    if corrupt_tokens_sha:
        contract = Pr86ArchiveContract(
            expected_archive_bytes=None,
            expected_archive_sha256=None,
            expected_member_bytes={},
            expected_tokens_sha256="0" * 64,
        )

    hpac_raw = _torch_bytes(model.state_dict())
    hpac_ppmd = pyppmd.compress(hpac_raw, max_order=4, mem_size=16 << 20)
    meta = {
        "N": 2,
        "mode": "hpac",
        "P": 2,
        "delta": 1,
        "ch": 4,
        "hpac_d_film": 2,
        "use_spm": False,
        "tokens_bpp": len(tokens_blob) * 8 / tokens.size,
        "H": 4,
        "W": 4,
    }
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("master.pt.gz", _gzip_torch({"master": torch.tensor([1.0])}))
        zf.writestr("slave.pt.gz", _gzip_torch({"slave": torch.tensor([2.0])}))
        zf.writestr("hpac.pt.ppmd", hpac_ppmd)
        zf.writestr("tokens.bin", tokens_blob)
        zf.writestr("meta.pt", _torch_bytes(meta))
    return archive, contract


def test_dependency_report_records_queue_behavior() -> None:
    report = collect_dependency_report()

    assert report["installed_versions"]["constriction"]
    assert report["installed_versions"]["pyppmd"]
    assert report["behavior_self_test"]["same_order_roundtrip_ok"] is True
    assert report["behavior_self_test"]["compressed_dtype"] == "uint32"
    assert report["status"] in {"passed", "passed_with_version_drift"}


def test_probability_variant_names_are_explicit_and_mark_source_contract() -> None:
    names = supported_hpac_probability_variant_names()

    assert names == (
        "source_float64_perfect_false",
        "source_float32_perfect_false",
        "source_float64_perfect_true",
        "source_float32_perfect_true",
    )


def test_encode_symbols_hpac_separates_residual_symbols_from_prev_context() -> None:
    pytest.importorskip("constriction")
    torch.manual_seed(1234)
    model = HPACMini(num_pairs=2, P=2, delta=1, ch=4, d_film=2, use_spm=False).eval()
    raw = np.array(
        [
            [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1], [2, 3, 4, 0]],
            [[1, 1, 2, 2], [3, 3, 4, 4], [0, 0, 1, 1], [2, 2, 3, 3]],
        ],
        dtype=np.uint8,
    )
    prev = np.zeros_like(raw)
    prev[1:] = raw[:-1]
    residual = ((raw.astype(np.int16) - prev.astype(np.int16)) % 5).astype(np.uint8)

    blob, encode_report = encode_symbols_hpac_with_prev_context(
        model,
        residual,
        prev,
        P=2,
        delta=1,
    )
    decoded_residual, decode_report = decode_symbols_hpac_with_prev_context(
        model,
        blob,
        prev,
        P=2,
        delta=1,
    )

    assert encode_report["symbol_context_contract"].startswith("symbols_nhw")
    assert decode_report["status"] == "passed"
    assert np.array_equal(decoded_residual, residual)
    assert np.array_equal(((decoded_residual.astype(np.int16) + prev.astype(np.int16)) % 5), raw)


def test_current_merged_source_context_classifies_pr86_hpac_contract() -> None:
    if not DEFAULT_PR86_MERGED_SOURCE_DIR.is_dir():
        pytest.skip("current merged PR86 source intake is not present")

    report = analyze_pr86_current_source_context(DEFAULT_PR86_MERGED_SOURCE_DIR)

    assert report["status"] == "current_merged_source"
    assert report["intake_summary"]["head_sha"] == "0eabe354f09b7490fd1cbb2b05a9102ab528d4d4"
    assert report["intake_summary"]["merge_commit_sha"] == "14bcede815306415a0005c3cd98804151bce4049"
    assert report["intake_summary"]["archive_identity_matches_cached_bytes"] is True
    assert report["token_semantics"]["training_objective"] == "residual_tokens"
    assert report["token_semantics"]["submitted_archive_token_encoding"] == "raw_tokens"
    assert report["probability_model_contract"]["archive_probability_numpy_dtype"] == "float64"
    assert report["probability_model_contract"]["inflate_probability_numpy_dtype"] == "float64"
    assert report["probability_model_contract"]["archive_categorical_perfect_false"] is True
    assert report["probability_model_contract"]["inflate_categorical_perfect_false"] is True
    assert report["probability_model_contract"]["explicit_16384_grid_in_archive_or_inflate"] is False
    assert report["source_files_identical_to_stale_cache"]["inflate.py"] is True


def test_archive_reader_fails_closed_on_zip_slip_unknown_and_duplicate_members(tmp_path: Path) -> None:
    contract = Pr86ArchiveContract(
        expected_archive_bytes=None,
        expected_archive_sha256=None,
        expected_member_bytes={},
        expected_tokens_sha256=None,
    )
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for name in EXPECTED_PR86_MEMBERS:
            zf.writestr(name, b"x")
        zf.writestr("../escape", b"bad")
        zf.writestr("tokens.bin", b"duplicate")

    with pytest.raises(Pr86HpacReplayError) as excinfo:
        read_pr86_archive(archive, contract=contract)

    assert excinfo.value.stage == "archive_member_contract"
    assert excinfo.value.reason == "duplicate_zip_members"
    assert excinfo.value.context["duplicate_member_names"] == ["tokens.bin"]


def test_synthetic_hpac_decode_reencode_byte_parity_passes(tmp_path: Path) -> None:
    archive, contract = _write_synthetic_hpac_archive(tmp_path)

    report = run_pr86_hpac_replay(archive, contract=contract, source_artifacts=())

    assert report["status"] == "passed"
    assert report["byte_parity_achieved"] is True
    assert report["dispatch_unlocked"] is False
    assert report["hpac_reencode"]["byte_exact_reencode"] is True
    assert report["probability_variant"]["name"] == "source_float64_perfect_false"
    assert report["probability_variant"]["source_contract"] is True
    assert report["tokens_bin"]["sha256_matches_expected"] is True


def test_synthetic_hpac_off_contract_variant_can_probe_without_dispatch_unlock(tmp_path: Path) -> None:
    archive, contract = _write_synthetic_hpac_archive(
        tmp_path,
        probability_variant="source_float32_perfect_true",
    )

    report = run_pr86_hpac_replay(
        archive,
        contract=contract,
        source_artifacts=(),
        probability_variant="source_float32_perfect_true",
    )

    assert report["status"] == "passed"
    assert report["byte_parity_achieved"] is True
    assert report["dispatch_unlocked"] is False
    assert report["probability_variant"]["source_contract"] is False
    assert report["hpac_reencode"]["byte_exact_reencode"] is True


def test_synthetic_probability_variant_matrix_reports_fail_closed_without_source_contract_parity(
    tmp_path: Path,
) -> None:
    archive, contract = _write_synthetic_hpac_archive(
        tmp_path,
        probability_variant="source_float32_perfect_true",
    )

    report = run_pr86_hpac_probability_variant_matrix(
        archive,
        contract=contract,
        source_artifacts=(),
        variants=("source_float32_perfect_true",),
    )

    assert report["status"] == "failed_closed"
    assert report["failure_reason"] == "no_source_contract_variant_full_decode_byte_exact_reencode"
    assert report["byte_parity_variants"] == ["source_float32_perfect_true"]
    assert report["source_contract_byte_parity_variants"] == []
    assert report["dispatch_unlocked"] is False


def test_synthetic_hpac_fails_closed_on_tokens_sha_mismatch(tmp_path: Path) -> None:
    archive, contract = _write_synthetic_hpac_archive(tmp_path, corrupt_tokens_sha=True)

    report = run_pr86_hpac_replay(archive, contract=contract, source_artifacts=())

    assert report["status"] == "failed_closed"
    assert report["failure_stage"] == "tokens_bin_contract"
    assert report["failure_reason"] == "tokens_sha256_mismatch"
    assert report["dispatch_unlocked"] is False


def test_cli_writes_json_for_synthetic_fixture(tmp_path: Path, capsys) -> None:
    archive, _contract = _write_synthetic_hpac_archive(tmp_path)
    out = tmp_path / "report.json"
    script = _load_cli_script()

    assert script.main(
        [
            "--archive",
            str(archive),
            "--allow-non-pr86-archive",
            "--no-source-artifacts",
            "--no-current-source-context",
            "--json-out",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["byte_parity_achieved"] is True
    assert payload["dispatch_unlocked"] is False


def test_cli_accepts_probability_variant_selection(tmp_path: Path, capsys) -> None:
    archive, _contract = _write_synthetic_hpac_archive(
        tmp_path,
        probability_variant="source_float32_perfect_true",
    )
    out = tmp_path / "variant_report.json"
    script = _load_cli_script()

    assert script.main(
        [
            "--archive",
            str(archive),
            "--allow-non-pr86-archive",
            "--no-source-artifacts",
            "--no-current-source-context",
            "--probability-variant",
            "source_float32_perfect_true",
            "--json-out",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["probability_variant"]["name"] == "source_float32_perfect_true"
    assert payload["probability_variant"]["source_contract"] is False


def test_cli_worker_report_scrubs_volatile_fields() -> None:
    script = _load_cli_script()

    payload = script._deterministic_worker_report(
        {
            "recorded_at_utc": "now",
            "elapsed_sec": 1.23,
            "nested": [{"elapsed_sec": 2.34, "stable": True}],
        }
    )

    assert payload == {"nested": [{"stable": True}]}


def test_real_pr86_artifact_smoke_fails_at_known_hpac_decode_blocker() -> None:
    pytest.importorskip("pyppmd")
    pytest.importorskip("constriction")
    if not DEFAULT_PR86_ARCHIVE.is_file():
        pytest.skip("public PR86 intake archive is not present")

    report = run_pr86_hpac_replay(
        DEFAULT_PR86_ARCHIVE,
        source_artifacts=default_source_artifact_paths(),
        max_frames=1,
    )

    assert report["status"] == "failed_closed"
    assert report["failure_stage"] == "submitted_tokens_decode"
    assert report["failure_reason"] == "hpac_entropy_decode_contract_mismatch"
    assert report["current_source_context"]["status"] == "current_merged_source"
    assert report["current_source_context"]["intake_summary"]["head_sha"] == (
        "0eabe354f09b7490fd1cbb2b05a9102ab528d4d4"
    )
    assert report["contract_mismatch_diagnostic"]["raw_vs_residual_classification"] == "raw_tokens"
    assert report["contract_mismatch_diagnostic"]["explicit_16384_grid_in_archive_or_inflate"] is False
    assert report["failure_context"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert report["tokens_bin"]["sha256"] == EXPECTED_PR86_TOKENS_SHA256
    assert report["dispatch_unlocked"] is False
