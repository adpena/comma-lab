# SPDX-License-Identifier: MIT
"""Regression tests for `_resolve_pr101_codec_dir` in `tools/build_pr101_finetuned_archive.py`.

PR101 intake clones (`experiments/results/public_pr101_intake_*/source/...`) place
codec.py + model.py at `submissions/hnerv_ft_microcodec/src/`, NOT at the source
dir root. The Modal A1 dispatcher (`experiments/modal_phase_a1_score_gradient_pr101.py`)
ships the entire intake source dir, so the build stage MUST resolve the nested
layout. The original `_stage_submission_dir` impl (commit pre-fix) only checked
the flat layout, causing rc=1 on Modal call_id `fc-01KR4TVY14SWW0VN07XT1B4Y2Q`
after a successful 200-epoch training run (1825 GPU-seconds wasted).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_SCRIPT_PATH = _REPO_ROOT / "tools" / "build_pr101_finetuned_archive.py"
_HARVEST_SCRIPT_PATH = _REPO_ROOT / "tools" / "harvest_modal_calls.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "build_pr101_finetuned_archive", _BUILD_SCRIPT_PATH,
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_harvest_module():
    spec = importlib.util.spec_from_file_location(
        "harvest_modal_calls", _HARVEST_SCRIPT_PATH,
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_harvest_modal_summary_merge_preserves_unrelated_call_rows() -> None:
    mod = _load_harvest_module()

    merged = mod.merge_modal_harvest_summary_rows(
        [
            {"label": "old_a", "call_id": "fc-a", "status": "already_harvested"},
            {"label": "old_b", "call_id": "fc-b", "status": "already_harvested"},
        ],
        [
            {"label": "new_b", "call_id": "fc-b", "status": "harvested"},
            {"label": "new_c", "call_id": "fc-c", "status": "harvested"},
        ],
    )

    assert merged == [
        {"label": "old_a", "call_id": "fc-a", "status": "already_harvested"},
        {"label": "new_b", "call_id": "fc-b", "status": "harvested"},
        {"label": "new_c", "call_id": "fc-c", "status": "harvested"},
    ]


def test_resolve_codec_dir_flat_layout(tmp_path: Path) -> None:
    """Flat layout: codec.py + model.py at root of pr101_source_dir."""
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# codec\n")
    (tmp_path / "model.py").write_text("# model\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == tmp_path


def test_resolve_codec_dir_nested_pr101_intake_layout(tmp_path: Path) -> None:
    """Canonical PR101 intake layout: nested under submissions/hnerv_ft_microcodec/src/."""
    mod = _load_build_module()
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# codec\n")
    (nested / "model.py").write_text("# model\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == nested


def test_resolve_codec_dir_missing_raises_systemexit(tmp_path: Path) -> None:
    """Empty pr101_source_dir must raise SystemExit naming both attempted layouts."""
    mod = _load_build_module()
    with pytest.raises(SystemExit) as exc_info:
        mod._resolve_pr101_codec_dir(tmp_path)
    msg = str(exc_info.value)
    assert "FATAL" in msg
    assert "submissions/hnerv_ft_microcodec/src" in msg or "submissions" in msg


def test_resolve_codec_dir_partial_flat_falls_through_to_missing(tmp_path: Path) -> None:
    """Flat codec.py without model.py at root must NOT count as flat layout."""
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# codec\n")
    # model.py missing → flat layout rejected; nested also missing
    with pytest.raises(SystemExit):
        mod._resolve_pr101_codec_dir(tmp_path)


def test_resolve_codec_dir_partial_nested_falls_through_to_missing(tmp_path: Path) -> None:
    """Nested codec.py without model.py must NOT count as nested layout."""
    mod = _load_build_module()
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# codec\n")
    # nested model.py missing → rejected
    with pytest.raises(SystemExit):
        mod._resolve_pr101_codec_dir(tmp_path)


def test_resolve_codec_dir_prefers_flat_when_both_present(tmp_path: Path) -> None:
    """If a directory has both flat and nested codec.py, flat takes precedence
    (caller passed pr101_source_dir at the right level on purpose).
    """
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# flat\n")
    (tmp_path / "model.py").write_text("# flat\n")
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# nested\n")
    (nested / "model.py").write_text("# nested\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == tmp_path


def test_finetuned_inflate_sh_uses_portable_python_fallback() -> None:
    """Generated A1 runtime must not call bare `python`.

    macOS and several clean Linux images do not provide `python`, while the
    contest/runtime path can still use `python3`. Local custody evals may bind
    `PYTHON=.venv/bin/python` when packet dependencies live in the repo venv.
    """
    mod = _load_build_module()
    assert '"${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"' in mod.INFLATE_SH_NO_DEAD_K
    assert 'python "$HERE/inflate.py" "$SRC" "$DST"' not in mod.INFLATE_SH_NO_DEAD_K


def test_build_manifest_records_old_new_sha_and_no_op_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_build_module()
    monkeypatch.setattr(mod, "DECODER_BLOB_LEN", 3)
    monkeypatch.setattr(mod, "LATENT_BLOB_LEN", 2)
    monkeypatch.setattr(mod, "FIXED_STATE_SCHEMA", [("w", (1,))])
    monkeypatch.setattr(mod, "encode_decoder_compact", lambda _sd, brotli_quality: b"NEW")

    import tac.pr101_split_brotli_codec as codec_mod

    monkeypatch.setattr(
        codec_mod,
        "decode_decoder_compact",
        lambda _blob: {"w": torch.tensor([1.0])},
    )

    state_dict = tmp_path / "checkpoint.pt"
    torch.save({"w": torch.tensor([1.0])}, state_dict)

    source_archive = tmp_path / "source_archive.zip"
    with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"OLDLASIDE")

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "codec.py").write_text("# codec\n")
    (source_dir / "model.py").write_text("# model\n")

    out_dir = tmp_path / "out"
    rc = mod.main(
        [
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--pr101-source-dir",
            str(source_dir),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["score_affecting_payload_changed"] is True
    assert manifest["source_archive_sha256"]
    assert manifest["archive_sha256"]
    no_op = manifest["no_op_detector"]
    assert no_op["score_affecting_payload_changed"] is True
    assert no_op["decoder_payload_changed"] is True
    assert no_op["latent_payload_preserved"] is True
    assert no_op["sidecar_payload_preserved"] is True
    old_new = manifest["old_new_sha_metadata"]
    assert old_new["source_archive_sha256"] == manifest["source_archive_sha256"]
    assert old_new["new_archive_sha256"] == manifest["archive_sha256"]
    assert old_new["source_latent_blob_sha256"] == old_new["new_latent_blob_sha256"]
    assert old_new["source_sidecar_blob_sha256"] == old_new["new_sidecar_blob_sha256"]


def test_harvest_modal_calls_handles_none_elapsed_and_stdout_tail() -> None:
    text = _HARVEST_SCRIPT_PATH.read_text()

    assert "PLAN ONLY: pass --execute" in text
    assert "artifacts_dir.mkdir(exist_ok=True)" in text
    assert "elapsed_raw = result.get(\"elapsed_seconds\")" in text
    assert "isinstance(elapsed_raw, (int, float))" in text
    assert "result.get(\"stdout_tail\", \"\") or \"\"" in text
    assert "append_modal_training_cost_anchor" in text
    assert "append_modal_training_terminal_claim" in text
    assert "_append_terminal_claim_evidence" in text
    assert "failed_modal_training_invalid_artifacts" in text
    assert '"cost_band_anchor": cost_anchor' in text
    assert '"terminal_claim": terminal_claim' in text
    assert '"terminal_evidence": terminal_evidence' in text
    assert "failed_modal_training_function_timeout" in text
    assert "_safe_harvest_artifact_path" in text


def test_harvest_modal_calls_no_execute_is_read_only(tmp_path: Path) -> None:
    lane_dir = tmp_path / "experiments" / "results" / "lane_demo_modal"
    lane_dir.mkdir(parents=True)
    (lane_dir / "modal_metadata.json").write_text(
        json.dumps(
            {
                "label": "demo_modal_lane",
                "call_id": "fc-demo",
                "dispatched_at": "2026-05-13T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    other_lane_dir = tmp_path / "experiments" / "results" / "lane_other_modal"
    other_lane_dir.mkdir(parents=True)
    (other_lane_dir / "modal_metadata.json").write_text(
        json.dumps(
            {
                "label": "other_modal_lane",
                "call_id": "fc-other",
                "dispatched_at": "2026-05-13T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_HARVEST_SCRIPT_PATH),
            "--repo-root",
            str(tmp_path),
            "--call-id",
            "fc-demo",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0
    assert "PLAN ONLY: pass --execute" in proc.stdout
    assert "Found 1 dispatched lanes" in proc.stdout
    assert "demo_modal_lane" in proc.stdout
    assert "other_modal_lane" not in proc.stdout
    assert not (tmp_path / "experiments" / "results" / "_modal_harvest_summary.json").exists()


def test_harvest_modal_calls_from_ledger_execute_runs_harvest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_harvest_module()
    calls: list[object] = []

    def fake_print_from_ledger_view(
        repo_root: Path,
        *,
        call_ids: frozenset[str],
    ) -> None:
        calls.append(("ledger_view", repo_root, call_ids))

    def fake_harvest_modal_calls(
        *,
        repo_root: Path,
        summary_output: Path,
        get_timeout_seconds: float,
        call_ids: frozenset[str],
    ) -> list[dict[str, object]]:
        calls.append(
            ("harvest", repo_root, summary_output, get_timeout_seconds, call_ids)
        )
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text("[]", encoding="utf-8")
        return []

    monkeypatch.setattr(mod, "_print_from_ledger_view", fake_print_from_ledger_view)
    monkeypatch.setattr(mod, "harvest_modal_calls", fake_harvest_modal_calls)

    rc = mod.main(
        [
            "--from-ledger",
            "--execute",
            "--repo-root",
            str(tmp_path),
            "--summary-output",
            "reports/summary.json",
            "--get-timeout-seconds",
            "7",
            "--call-id",
            "fc-demo",
        ]
    )

    assert rc == 0
    assert calls[0] == ("ledger_view", tmp_path.resolve(), frozenset({"fc-demo"}))
    assert calls[1][0] == "harvest"
    assert calls[1][1] == tmp_path.resolve()
    assert calls[1][4] == frozenset({"fc-demo"})
    assert calls[1][2] == tmp_path.resolve() / "reports" / "summary.json"
    assert calls[1][3] == 7


def test_harvest_modal_calls_appends_call_id_ledger_terminal_event(
    tmp_path: Path,
) -> None:
    mod = _load_harvest_module()
    from tac.deploy.modal.call_id_ledger import (
        query_by_call_id,
        query_unharvested,
        register_dispatched_call_id,
    )

    ledger = tmp_path / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    lock = ledger.with_suffix(ledger.suffix + ".lock")
    register_dispatched_call_id(
        call_id="fc-demo",
        lane_id="lane_demo",
        label="demo",
        path=ledger,
        lock_path=lock,
    )

    result = mod._append_call_id_ledger_terminal_event(
        repo_root=tmp_path,
        metadata={
            "call_id": "fc-demo",
            "lane_id": "lane_demo",
            "label": "demo",
            "platform": "modal",
            "gpu": "T4",
            "dispatched_at": "2026-05-15T00:00:00Z",
        },
        harvested={"rc": 0, "elapsed_seconds": 12.5, "n_artifacts": 3},
        terminal_claim={
            "appended": True,
            "status": "completed_modal_training_recovered_no_score_claim",
        },
        agent="pytest",
    )

    assert result["appended"] is True
    assert query_unharvested(path=ledger) == []
    rows = query_by_call_id("fc-demo", path=ledger)
    assert [row["status"] for row in rows] == ["dispatched", "harvested"]
    assert rows[-1]["rc"] == 0
    assert rows[-1]["elapsed_seconds"] == 12.5

    second = mod._append_call_id_ledger_terminal_event(
        repo_root=tmp_path,
        metadata={"call_id": "fc-demo"},
        harvested={"rc": 0},
        terminal_claim={"appended": True, "status": "completed_modal_training_recovered_no_score_claim"},
        agent="pytest",
    )
    assert second["already_terminal"] is True
    assert len(query_by_call_id("fc-demo", path=ledger)) == 2


def test_harvest_modal_calls_supplements_lossy_terminal_call_id_row(
    tmp_path: Path,
) -> None:
    mod = _load_harvest_module()
    from tac.deploy.modal.call_id_ledger import (
        query_by_call_id,
        register_dispatched_call_id,
        update_call_id_outcome,
    )

    ledger = tmp_path / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    lock = ledger.with_suffix(ledger.suffix + ".lock")
    register_dispatched_call_id(
        call_id="fc-lossy",
        lane_id="lane_demo",
        label="demo",
        path=ledger,
        lock_path=lock,
    )
    update_call_id_outcome(
        call_id="fc-lossy",
        status="failed",
        path=ledger,
        lock_path=lock,
        lane_id="lane_demo",
        label="demo",
    )

    result = mod._append_call_id_ledger_terminal_event(
        repo_root=tmp_path,
        metadata={
            "call_id": "fc-lossy",
            "lane_id": "lane_demo",
            "label": "demo",
            "platform": "modal",
            "gpu": "T4",
        },
        harvested={
            "rc": 1,
            "elapsed_seconds": 170.1,
            "archive_sha256": "a" * 64,
            "archive_bytes": 123,
        },
        terminal_claim={"appended": True, "status": "failed_modal_training_rc_1"},
        agent="pytest",
    )

    assert result["appended"] is True
    rows = query_by_call_id("fc-lossy", path=ledger)
    assert [row["status"] for row in rows] == ["dispatched", "failed", "failed"]
    assert rows[-1]["rc"] == 1
    assert rows[-1]["elapsed_seconds"] == 170.1
    assert rows[-1]["archive_sha256"] == "a" * 64
    assert rows[-1]["archive_bytes"] == 123

    second = mod._append_call_id_ledger_terminal_event(
        repo_root=tmp_path,
        metadata={"call_id": "fc-lossy"},
        harvested={"rc": 1, "elapsed_seconds": 170.1},
        terminal_claim={"appended": True, "status": "failed_modal_training_rc_1"},
        agent="pytest",
    )
    assert second["already_terminal"] is True


def test_harvest_modal_calls_rejects_unsafe_artifact_paths(tmp_path: Path) -> None:
    mod = _load_harvest_module()
    root = tmp_path / "harvested_artifacts"

    assert mod._safe_harvest_artifact_path(root, "nested/result.json") == (
        root / "nested" / "result.json"
    )
    for unsafe in (
        ".",
        "../escape.json",
        "/tmp/escape.json",
        "nested/../../escape.json",
    ):
        with pytest.raises(mod.UnsafeModalArtifactPath):
            mod._safe_harvest_artifact_path(root, unsafe)


def test_harvest_modal_calls_refuses_partial_artifact_write_success(tmp_path: Path) -> None:
    mod = _load_harvest_module()
    root = tmp_path / "harvested_artifacts"

    with pytest.raises(mod.ModalArtifactWriteError) as exc_info:
        mod._write_modal_artifacts(
            artifacts_dir=root,
            artifacts={
                "ok/result.json": b"{}",
                "bad/not-bytes.json": {"not": "bytes"},
            },
        )

    assert not (root / "ok" / "result.json").exists()
    assert exc_info.value.errors[0]["relative_path"] == "bad/not-bytes.json"
    assert exc_info.value.errors[0]["error_type"] == "TypeError"


def test_harvest_modal_calls_refuses_non_string_artifact_keys(tmp_path: Path) -> None:
    mod = _load_harvest_module()
    root = tmp_path / "harvested_artifacts"

    with pytest.raises(mod.ModalArtifactWriteError) as exc_info:
        mod._write_modal_artifacts(
            artifacts_dir=root,
            artifacts={
                7: b"{}",
            },
        )

    assert exc_info.value.errors[0]["relative_path"] == "7"
    assert exc_info.value.errors[0]["error_type"] == "TypeError"
    assert not (root / "7").exists()


def test_harvest_modal_calls_appends_terminal_claim_evidence_once(tmp_path: Path) -> None:
    mod = _load_harvest_module()
    repo = tmp_path
    out_dir = repo / "experiments" / "results" / "lane_demo_modal"
    out_dir.mkdir(parents=True)
    claim = {
        "appended": True,
        "lane_id": "lane_demo",
        "instance_job_id": "job_demo",
        "status": "failed_modal_training_rc_1",
    }

    first = mod._append_terminal_claim_evidence(
        repo_root=repo,
        out_dir=out_dir,
        terminal_claim=claim,
    )
    second = mod._append_terminal_claim_evidence(
        repo_root=repo,
        out_dir=out_dir,
        terminal_claim=claim,
    )

    evidence = repo / "reports" / "cathedral_autopilot_evidence.jsonl"
    lines = evidence.read_text(encoding="utf-8").splitlines()
    assert first["appended"] is True
    assert second["already_covered"] is True
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["covered_terminal_claims"] == [
        {
            "lane_id": "lane_demo",
            "instance_job_id": "job_demo",
            "status": "failed_modal_training_rc_1",
        }
    ]


def test_harvest_modal_calls_terminal_evidence_preserves_inline_auth_eval_signal(
    tmp_path: Path,
) -> None:
    mod = _load_harvest_module()
    repo = tmp_path
    out_dir = repo / "experiments" / "results" / "lane_demo_modal"
    out_dir.mkdir(parents=True)
    claim = {
        "appended": True,
        "lane_id": "lane_demo",
        "instance_job_id": "job_demo",
        "status": "completed_modal_training_recovered_with_contest_cuda_auth_eval",
        "recovered_auth_eval": {
            "auth_eval_score": 90.78433465890384,
            "auth_eval_score_axis": "contest_cuda",
            "auth_eval_score_claim_valid": True,
            "auth_eval_exact_cuda_complete": True,
        },
    }

    result = mod._append_terminal_claim_evidence(
        repo_root=repo,
        out_dir=out_dir,
        terminal_claim=claim,
    )

    evidence = repo / "reports" / "cathedral_autopilot_evidence.jsonl"
    row = json.loads(evidence.read_text(encoding="utf-8").splitlines()[0])
    assert result["appended"] is True
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["recovered_auth_eval"]["auth_eval_score_axis"] == "contest_cuda"
    assert (
        row["measured_config_status"]
        == "terminal_dispatch_claim_preserved_with_inline_auth_eval"
    )
    assert (
        "terminal_training_harvest_preserves_inline_auth_eval_but_is_not_rank_authority"
        in row["dispatch_blockers"]
    )


def test_harvest_modal_calls_refuses_evidence_when_terminal_claim_failed(tmp_path: Path) -> None:
    mod = _load_harvest_module()
    repo = tmp_path
    out_dir = repo / "experiments" / "results" / "lane_demo_modal"
    out_dir.mkdir(parents=True)

    result = mod._append_terminal_claim_evidence(
        repo_root=repo,
        out_dir=out_dir,
        terminal_claim={
            "appended": False,
            "reason": "terminal_claim_failed:RuntimeError:boom",
            "lane_id": "lane_demo",
            "instance_job_id": "job_demo",
            "status": "failed_modal_training_rc_1",
        },
    )

    assert result == {
        "appended": False,
        "reason": "terminal_claim_not_appended",
        "lane_id": "lane_demo",
        "instance_job_id": "job_demo",
        "status": "failed_modal_training_rc_1",
    }
    assert not (repo / "reports" / "cathedral_autopilot_evidence.jsonl").exists()


def test_harvest_modal_calls_repolls_generated_nonterminal_summary_only(
    tmp_path: Path,
) -> None:
    mod = _load_harvest_module()
    out_dir = tmp_path / "lane_demo_modal"
    artifacts = out_dir / "harvested_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "_harvest_summary.json").write_text(
        json.dumps({"status": "not_ready"}),
        encoding="utf-8",
    )

    assert mod._already_harvested(out_dir, artifacts) is False

    (artifacts / "_harvest_summary.json").write_text(
        json.dumps({"status": "function_timeout"}),
        encoding="utf-8",
    )
    assert mod._already_harvested(out_dir, artifacts) is True

    (artifacts / "_harvest_summary.json").write_text(
        json.dumps({"status": "artifact_write_failed_retryable"}),
        encoding="utf-8",
    )
    assert mod._already_harvested(out_dir, artifacts) is False


def test_harvest_modal_calls_incomplete_claim_marker_keeps_local_result_signal(
    tmp_path: Path,
) -> None:
    mod = _load_harvest_module()
    out_dir = tmp_path / "lane_demo_modal"
    artifacts = out_dir / "harvested_artifacts"
    artifacts.mkdir(parents=True)
    (out_dir / "modal_training_terminal_claim.json").write_text(
        json.dumps(
            {
                "appended": False,
                "reason": "metadata_missing_lane_id_or_instance_job_id",
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "_harvest_summary.json").write_text(
        json.dumps(
            {
                "rc": 137,
                "elapsed_seconds": 33.0,
                "timed_out": False,
                "n_artifacts": 55,
                "crash_kind": "RC_137",
            }
        ),
        encoding="utf-8",
    )

    assert mod._already_harvested(out_dir, artifacts) is True
