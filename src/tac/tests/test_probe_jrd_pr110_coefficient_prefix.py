# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/probe_jrd_pr110_coefficient_prefix.py"
spec = importlib.util.spec_from_file_location("probe_jrd_pr110_test", TOOL)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _populate_runtime(submission: Path) -> None:
    for relative in tool.RUNTIME_RELATIVE_FILES:
        path = submission / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture runtime {relative}\n")


def _proof_path(recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else tool.REPO / path


def test_refuse_unsafe_scope_requires_results_child(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"x")
    with pytest.raises(ValueError, match="output must be a child"):
        tool.refuse_unsafe_scope(archive, tmp_path / "out")


def test_section_summary_separates_last_safe_from_best_rate() -> None:
    baseline = {
        "archive_zip_bytes": 100,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "eval_pairs": 1,
        "axis": "[macOS-CPU advisory]",
    }
    rows = [
        {"bits_removed": 1, "archive_zip_bytes": 90, "d_seg": 0.1, "d_pose": 0.2},
        {"bits_removed": 2, "archive_zip_bytes": 95, "d_seg": 0.09, "d_pose": 0.2},
        {"bits_removed": 3, "archive_zip_bytes": 80, "d_seg": 0.11, "d_pose": 0.2},
    ]
    summary = tool._section_summary(
        section="a", family="uniform", baseline=baseline, rows=rows
    )
    assert summary["last_safe"]["bits_removed"] == 2
    assert summary["best_byte_safe"]["bits_removed"] == 1
    assert summary["eval_pairs"] == 1
    assert summary["axis"] == "[macOS-CPU advisory]"
    assert "not an n600" in summary["verdict_scope"]


def test_section_summary_calls_n600_scope_n600() -> None:
    baseline = {
        "archive_zip_bytes": 100,
        "d_seg": 0.1,
        "d_pose": 0.2,
        "eval_pairs": 600,
        "axis": "[macOS-CPU advisory]",
    }
    rows = [
        {"bits_removed": 1, "archive_zip_bytes": 90, "d_seg": 0.1, "d_pose": 0.2}
    ]
    summary = tool._section_summary(
        section="a", family="uniform", baseline=baseline, rows=rows
    )
    assert "all 600 contest pairs" in summary["verdict_scope"]
    assert "not an n600" not in summary["verdict_scope"]


def test_resume_fails_closed_on_fingerprint_change(tmp_path: Path) -> None:
    out = tmp_path / "run"
    tool.atomic_json(
        out / "resume/state.json",
        {"fingerprint": "old", "rows": [], "completed_sections": []},
    )
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        tool._load_resume(out, "new")


def test_parse_args_bounds_pair_counts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tool.parse_args(
            [
                "--archive",
                str(tmp_path / "a.zip"),
                "--out-dir",
                str(tmp_path / "out"),
                "--final-eval-pairs",
                "601",
            ]
        )
    with pytest.raises(SystemExit, match="2"):
        tool.parse_args(
            [
                "--archive",
                str(tmp_path / "a.zip"),
                "--out-dir",
                str(tmp_path / "out"),
                "--final-eval-pairs",
                "24",
            ]
        )


def test_controls_require_stable_positive_and_both_negative_component_changes() -> None:
    baseline = {
        "archive_zip_bytes": 100,
        "archive_zip_sha256": "a",
        "d_seg": 0.1,
        "d_pose": 0.2,
        "raw_sha256": "raw-a",
        "raw_bytes": 12,
    }
    positive = dict(baseline)
    negative = {**baseline, "archive_zip_sha256": "b", "d_seg": 0.2, "d_pose": 0.3}
    tool.validate_controls(baseline, positive, negative)
    with pytest.raises(RuntimeError, match="both scorer components"):
        tool.validate_controls(baseline, positive, {**negative, "d_pose": 0.2})
    with pytest.raises(RuntimeError, match="positive-repeat"):
        tool.validate_controls(baseline, {**positive, "d_seg": 0.11}, negative)


def test_final_positive_repeat_includes_rendered_raw_hash() -> None:
    baseline = {
        "archive_zip_bytes": 100,
        "archive_zip_sha256": "archive",
        "d_seg": 0.1,
        "d_pose": 0.2,
        "raw_sha256": "raw-a",
        "raw_bytes": 12,
    }
    tool.validate_positive_repeat(baseline, dict(baseline))
    with pytest.raises(RuntimeError, match="final positive-repeat"):
        tool.validate_positive_repeat(
            baseline, {**baseline, "raw_sha256": "raw-b"}
        )


def test_measure_final_controls_requires_and_returns_n600_negative_canary() -> None:
    calls: list[str] = []

    def measure(label: str, payload: bytes) -> dict[str, object]:
        calls.append(label)
        negative = label == "negative_all_decoder_coefficients_zero_final"
        return {
            "archive_zip_bytes": len(payload),
            "archive_zip_sha256": "negative" if negative else "baseline",
            "d_seg": 0.2 if negative else 0.1,
            "d_pose": 0.3 if negative else 0.2,
            "raw_sha256": "negative-raw" if negative else "baseline-raw",
            "raw_bytes": 12,
            "eval_pairs": 600,
        }

    baseline, positive, negative = tool.measure_final_controls(
        measure, b"base", b"zero"
    )
    assert calls == [
        "baseline_final",
        "baseline_final_positive_repeat",
        "negative_all_decoder_coefficients_zero_final",
    ]
    assert baseline == positive
    assert negative["archive_zip_sha256"] == "negative"

    def one_pair_measure(label: str, payload: bytes) -> dict[str, object]:
        return {**measure(label, payload), "eval_pairs": 1}

    with pytest.raises(RuntimeError, match="exactly 600 pairs"):
        tool.measure_final_controls(one_pair_measure, b"base", b"zero")


def test_atomic_bytes_replaces_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "candidate.zip"
    target.write_bytes(b"old")
    tool.atomic_bytes(target, b"complete-new-payload")
    assert target.read_bytes() == b"complete-new-payload"
    assert not list(tmp_path.glob(".*.tmp"))


def test_runtime_inflate_executes_twice_and_matches_in_process_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    _populate_runtime(submission)
    inflate = submission / "inflate.py"
    inflate.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n"
    )
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    candidate = receipt_dir / "candidate.zip"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"raw!")
    proof = tool.verify_candidate_runtime_inflate(
        candidate_path=candidate,
        submission_dir=submission,
        receipt_dir=receipt_dir,
        expected_raw_sha256=tool.sha256_bytes(b"raw!"),
        expected_raw_bytes=4,
        scratch_root=tmp_path / "scratch-root",
    )
    assert proof["bit_exact"] is True
    assert proof["scratch_cleaned_on_success"] is True
    assert [row["raw_sha256"] for row in proof["passes"]] == [
        tool.sha256_bytes(b"raw!"),
        tool.sha256_bytes(b"raw!"),
    ]
    assert _proof_path(proof["passes"][0]["log_path"]).is_file()
    assert _proof_path(proof["passes"][1]["log_path"]).is_file()
    assert tool.reusable_runtime_inflate_proof(
        proof,
        candidate_path=candidate,
        submission_dir=submission,
        expected_raw_sha256=tool.sha256_bytes(b"raw!"),
        expected_raw_bytes=4,
    )


def test_runtime_inflate_refuses_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    _populate_runtime(submission)
    (submission / "inflate.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n"
    )
    candidate = tmp_path / "candidate.zip"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"raw!")
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    with pytest.raises(RuntimeError, match="differs from in-process"):
        tool.verify_candidate_runtime_inflate(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt_dir,
            expected_raw_sha256=tool.sha256_bytes(b"nope"),
            expected_raw_bytes=4,
            scratch_root=tmp_path / "scratch-root",
        )
    proof = tool.json.loads((receipt_dir / "runtime_inflate_proof.json").read_text())
    assert proof["bit_exact"] is False
    assert proof["failure_cleanup_disposition"].startswith("retained fail-closed")
    assert {Path(row["path"]).name for row in proof["retained_failure_artifacts"]} == {
        "pass_1.raw",
        "x",
    }
    first_attempt = Path(proof["scratch_attempt_path"])
    first_log = _proof_path(proof["passes"][0]["log_path"])
    first_log_bytes = first_log.read_bytes()
    assert first_attempt.is_dir()
    with pytest.raises(RuntimeError, match="differs from in-process"):
        tool.verify_candidate_runtime_inflate(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt_dir,
            expected_raw_sha256=tool.sha256_bytes(b"nope"),
            expected_raw_bytes=4,
            scratch_root=tmp_path / "scratch-root",
        )
    second = tool.json.loads((receipt_dir / "runtime_inflate_proof.json").read_text())
    assert Path(second["scratch_attempt_path"]) != first_attempt
    assert first_attempt.is_dir()
    assert _proof_path(second["passes"][0]["log_path"]) != first_log
    assert first_log.read_bytes() == first_log_bytes


def test_runtime_inflate_nonzero_exit_authenticates_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    _populate_runtime(submission)
    (submission / "inflate.py").write_text(
        "import sys\n"
        "print('intentional crash evidence')\n"
        "raise SystemExit(7)\n"
    )
    candidate = tmp_path / "candidate.zip"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"raw!")
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    with pytest.raises(RuntimeError, match="failed rc=7"):
        tool.verify_candidate_runtime_inflate(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt_dir,
            expected_raw_sha256=tool.sha256_bytes(b"raw!"),
            expected_raw_bytes=4,
            scratch_root=tmp_path / "scratch-root",
        )
    proof = tool.json.loads((receipt_dir / "runtime_inflate_proof.json").read_text())
    assert len(proof["passes"]) == 1
    row = proof["passes"][0]
    assert row["returncode"] == 7
    log_path = _proof_path(row["log_path"])
    assert log_path.is_file()
    assert tool.sha256_file(log_path) == row["log_sha256"]
    assert b"intentional crash evidence" in log_path.read_bytes()


def test_fifo_runtime_proof_reuse_binds_runtime_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    _populate_runtime(submission)
    inflate = submission / "inflate.py"
    inflate.write_text("print('runtime')\n")
    candidate = tmp_path / "candidate.zip"
    candidate.write_bytes(b"zip-bytes")
    logs = []
    passes = []
    for index in (1, 2):
        log = tmp_path / f"pass_{index}.log"
        log.write_text(f"pass={index}\n")
        logs.append(log)
        passes.append(
            {
                "pass": index,
                "returncode": 0,
                "execution_error": None,
                "reader_daemon": True,
                "reader_alive_after_join": False,
                "raw_sha256": "a" * 64,
                "raw_bytes": 4,
                "log_path": str(log),
                "log_sha256": tool.sha256_file(log),
            }
        )
    proof = {
        "schema": "jrd_pr110_runtime_inflate_fifo_proof.v1",
        "candidate_sha256": tool.sha256_file(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "expected_in_process_raw_sha256": "a" * 64,
        "expected_raw_bytes": 4,
        "runtime_inflate_py": {
            "path": str(inflate),
            "bytes": inflate.stat().st_size,
            "sha256": tool.sha256_file(inflate),
        },
        "submission_runtime": tool._runtime_custody(submission),
        "passes": passes,
        "streaming_fifo_no_bulk_raw_materialized": True,
        "bit_exact": True,
        "scratch_cleaned_on_success": True,
    }

    def reusable() -> bool:
        return tool.reusable_runtime_inflate_proof(
            proof,
            candidate_path=candidate,
            submission_dir=submission,
            expected_raw_sha256="a" * 64,
            expected_raw_bytes=4,
        )

    assert reusable() is True
    transitive = (
        submission
        / "encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py"
    )
    original_transitive = transitive.read_text()
    transitive.write_text(original_transitive + "runtime mutation\n")
    assert reusable() is False
    transitive.write_text(original_transitive)
    proof["runtime_inflate_py"]["sha256"] = "0" * 64
    assert reusable() is False
    proof["runtime_inflate_py"]["sha256"] = tool.sha256_file(inflate)
    proof["streaming_fifo_no_bulk_raw_materialized"] = False
    assert reusable() is False
    proof["streaming_fifo_no_bulk_raw_materialized"] = True
    logs[0].unlink()
    assert reusable() is False


def test_fifo_runtime_proof_reuse_rejects_mutated_log_and_reader_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    _populate_runtime(submission)
    inflate = submission / "inflate.py"
    inflate.write_text("print('runtime')\n")
    candidate = tmp_path / "candidate.zip"
    candidate.write_bytes(b"zip-bytes")
    log_paths = []
    rows = []
    for index in (1, 2):
        log = tmp_path / f"pass_{index}.log"
        log.write_text(f"pass={index}\n")
        log_paths.append(log)
        rows.append(
            {
                "pass": index,
                "returncode": 0,
                "execution_error": None,
                "reader_daemon": True,
                "reader_alive_after_join": False,
                "raw_sha256": "b" * 64,
                "raw_bytes": 4,
                "log_path": str(log),
                "log_sha256": tool.sha256_file(log),
            }
        )
    proof = {
        "schema": "jrd_pr110_runtime_inflate_fifo_proof.v1",
        "candidate_sha256": tool.sha256_file(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "expected_in_process_raw_sha256": "b" * 64,
        "expected_raw_bytes": 4,
        "runtime_inflate_py": {
            "path": str(inflate),
            "bytes": inflate.stat().st_size,
            "sha256": tool.sha256_file(inflate),
        },
        "submission_runtime": tool._runtime_custody(submission),
        "passes": rows,
        "streaming_fifo_no_bulk_raw_materialized": True,
        "bit_exact": True,
        "scratch_cleaned_on_success": True,
    }
    kwargs = {
        "candidate_path": candidate,
        "submission_dir": submission,
        "expected_raw_sha256": "b" * 64,
        "expected_raw_bytes": 4,
    }
    assert tool.reusable_runtime_inflate_proof(proof, **kwargs) is True
    log_paths[1].write_text("tampered\n")
    assert tool.reusable_runtime_inflate_proof(proof, **kwargs) is False
    log_paths[1].write_text("pass=2\n")
    proof["passes"][0]["reader_alive_after_join"] = True
    assert tool.reusable_runtime_inflate_proof(proof, **kwargs) is False
    proof["passes"][0]["reader_alive_after_join"] = False
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_log = outside / "pass_1.log"
    outside_log.write_text("pass=1\n")
    proof["passes"][0]["log_path"] = str(outside_log)
    proof["passes"][0]["log_sha256"] = tool.sha256_file(outside_log)
    assert tool.reusable_runtime_inflate_proof(proof, **kwargs) is False


def test_runtime_versions_bind_numerical_dependencies() -> None:
    versions = tool._runtime_versions()
    assert versions["python_executable_sha256"]
    assert set(versions["packages"]) == {
        "numpy",
        "torch",
        "safetensors",
        "Brotli",
        "constriction",
    }


def test_runtime_custody_binds_live_feca_transitive_encoder_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    submission = tmp_path / "submission"
    for index, relative in enumerate(tool.RUNTIME_RELATIVE_FILES):
        path = submission / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime-{index}\n")
    expected = {
        "encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py",
        "encoder/build_pr101_frame_exploit_selector_packet_markov.py",
    }
    assert expected.issubset(set(tool.RUNTIME_RELATIVE_FILES))
    before = tool._runtime_custody(submission)["tree_sha256"]
    for relative in sorted(expected):
        path = submission / relative
        original = path.read_text()
        path.write_text(original + "mutation\n")
        assert tool._runtime_custody(submission)["tree_sha256"] != before
        path.write_text(original)
