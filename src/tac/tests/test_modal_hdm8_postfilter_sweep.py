# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "modal_hdm8_postfilter_sweep.py"


def _load_module():
    pytest.importorskip("modal", reason="modal SDK not installed")
    spec = importlib.util.spec_from_file_location("modal_hdm8_postfilter_sweep_mod", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_hdm8_postfilter_sweep_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_modes_preserves_comma_rgb_modes() -> None:
    mod = _load_module()

    modes = mod.parse_modes(
        modes="none;even_rgb_bias:2,-1,-1;even_grain_chroma:2.0"
    )

    assert modes == ["none", "even_rgb_bias:2,-1,-1", "even_grain_chroma:2.0"]


def test_parse_modes_requires_none_baseline() -> None:
    mod = _load_module()

    with pytest.raises(SystemExit, match="must include 'none'"):
        mod.parse_modes(modes="even_bias:1;even_bias:2")


def test_parse_modes_from_proxy_json(tmp_path: Path) -> None:
    mod = _load_module()
    proxy = tmp_path / "proxy.json"
    proxy.write_text(
        json.dumps(
            {
                "modes": [
                    {"mode": "none"},
                    {"mode": "even_rgb_bias:2,-1,-1"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mod.parse_modes(modes_from_json=str(proxy)) == [
        "none",
        "even_rgb_bias:2,-1,-1",
    ]


def test_dry_run_writes_request_without_claim_or_remote(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fake archive bytes")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        mod,
        "record_dispatch_claim",
        lambda **_kwargs: pytest.fail("claim attempted"),
    )

    class NoRemote:
        @staticmethod
        def remote(*_args):
            pytest.fail("remote attempted")

        @staticmethod
        def spawn(*_args):
            pytest.fail("spawn attempted")

    monkeypatch.setattr(mod, "run_hdm8_postfilter_sweep_t4", NoRemote)
    mod.main(
        str(archive),
        str(out_dir),
        n_pairs=2,
        modes="none;even_bias:1",
        include_per_pair=True,
        dry_run=True,
    )

    request = json.loads(
        (out_dir / "modal_hdm8_postfilter_sweep_local_request.json").read_text()
    )
    assert request["axis"] == mod.AXIS
    assert request["dispatch_attempted"] is False
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False
    assert request["modes"] == ["none", "even_bias:1"]
    assert request["include_per_pair"] is True
    assert request["source_transparency"]["schema"] == "tac_source_transparency_v1"
    assert request["source_transparency"]["release_contract"][
        "include_in_submission_packets"
    ] is True


def test_recover_dry_run_dir_without_spawn_manifest_is_clean_skip(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "dryrun"
    out_dir.mkdir()
    request = {
        "schema_version": 1,
        "modal_dispatch_mode": "dry_run",
        "dispatch_attempted": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    (out_dir / "modal_hdm8_postfilter_sweep_local_request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )

    summary = mod.recover_detached(output_dir=out_dir)

    assert summary["status"] == "dry_run_no_remote_call"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    persisted = json.loads(
        (out_dir / "modal_hdm8_postfilter_sweep_recover_summary.json").read_text()
    )
    assert persisted["status"] == "dry_run_no_remote_call"


def test_recover_cancelled_provider_call_persists_terminal_summary(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    out_dir = tmp_path / "cancelled"
    out_dir.mkdir()
    (out_dir / "modal_hdm8_postfilter_sweep_spawn.json").write_text(
        json.dumps(
            {
                "schema_version": "modal_hdm8_postfilter_sweep_spawn_v1",
                "call_id": "fc-cancelled",
                "lane_id": "hdm8_cancelled_probe",
                "instance_job_id": "hdm8_cancelled_probe_job",
                "claim_agent": "codex:test",
                "score_claim": False,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    class CancelledCall:
        def get(self, *, timeout):
            assert timeout == 0.0
            raise RuntimeError("Function call was cancelled by user. | bad\ncell")

    terminal_calls: list[dict[str, object]] = []
    monkeypatch.setattr(mod, "_function_call_from_id", lambda call_id: CancelledCall())
    monkeypatch.setattr(
        mod,
        "terminal_dispatch_claim",
        lambda **kwargs: terminal_calls.append(kwargs),
    )

    summary = mod.recover_detached(output_dir=out_dir)

    assert summary["status"] == "cancelled_provider_function_call"
    assert summary["passed"] is False
    assert summary["returncode"] == 1
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["provider_exception_type"] == "RuntimeError"
    assert "cancelled by user" in summary["provider_exception_message"]
    assert summary["lane_id"] == "hdm8_cancelled_probe"
    assert summary["instance_job_id"] == "hdm8_cancelled_probe_job"
    assert summary["terminal_claim_closed"] is True
    assert (
        summary["terminal_claim_status"]
        == "cancelled_modal_hdm8_postfilter_sweep_no_score_claim"
    )
    persisted = json.loads(
        (out_dir / "modal_hdm8_postfilter_sweep_recover_summary.json").read_text()
    )
    assert persisted == summary
    assert terminal_calls
    assert (
        terminal_calls[0]["status"]
        == "cancelled_modal_hdm8_postfilter_sweep_no_score_claim"
    )
    assert "score_claim=false" in terminal_calls[0]["notes"]
    assert "|" not in terminal_calls[0]["notes"]
    assert "\n" not in terminal_calls[0]["notes"]


def test_recover_cancelled_provider_call_can_skip_claim_close(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    out_dir = tmp_path / "cancelled_no_close"
    out_dir.mkdir()
    (out_dir / "modal_hdm8_postfilter_sweep_spawn.json").write_text(
        json.dumps({"call_id": "fc-cancelled"}),
        encoding="utf-8",
    )

    class CancelledCall:
        def get(self, *, timeout):
            raise RuntimeError("Function call was cancelled by user.")

    monkeypatch.setattr(mod, "_function_call_from_id", lambda call_id: CancelledCall())
    monkeypatch.setattr(
        mod,
        "terminal_dispatch_claim",
        lambda **_kwargs: pytest.fail("terminal claim should be skipped"),
    )

    summary = mod.recover_detached(output_dir=out_dir, no_close_claim=True)

    assert summary["status"] == "cancelled_provider_function_call"
    assert summary["passed"] is False
    assert summary["terminal_claim_status"] == "skipped_no_close_claim"


def test_recover_invalid_non_dict_result_fails_and_closes_claim(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    out_dir = tmp_path / "invalid_result"
    out_dir.mkdir()
    (out_dir / "modal_hdm8_postfilter_sweep_spawn.json").write_text(
        json.dumps(
            {
                "call_id": "fc-invalid",
                "lane_id": "hdm8_invalid_probe",
                "instance_job_id": "hdm8_invalid_probe_job",
                "claim_agent": "codex:test",
                "local_request": {
                    "archive_sha256": "a" * 64,
                    "archive_size_bytes": 123,
                },
            }
        ),
        encoding="utf-8",
    )

    class InvalidCall:
        def get(self, *, timeout):
            return ["not", "a", "dict"]

    terminal_calls: list[dict[str, object]] = []
    monkeypatch.setattr(mod, "_function_call_from_id", lambda call_id: InvalidCall())
    monkeypatch.setattr(
        mod,
        "terminal_dispatch_claim",
        lambda **kwargs: terminal_calls.append(kwargs),
    )

    summary = mod.recover_detached(output_dir=out_dir)

    assert summary["status"] == "invalid_result"
    assert summary["passed"] is False
    assert summary["returncode"] == 5
    assert summary["axis"] == mod.AXIS
    assert summary["archive_sha256"] == "a" * 64
    assert summary["archive_size_bytes"] == 123
    assert summary["terminal_claim_closed"] is True
    assert (
        summary["terminal_claim_status"]
        == "failed_modal_hdm8_postfilter_sweep_invalid_result_no_score_claim"
    )
    assert terminal_calls
    assert (
        terminal_calls[0]["status"]
        == "failed_modal_hdm8_postfilter_sweep_invalid_result_no_score_claim"
    )


def test_non_dry_run_requires_claim_fields_before_remote(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fake archive bytes")

    monkeypatch.setattr(
        mod,
        "record_dispatch_claim",
        lambda **_kwargs: pytest.fail("claim attempted"),
    )

    class NoRemote:
        @staticmethod
        def remote(*_args):
            pytest.fail("remote attempted")

    monkeypatch.setattr(mod, "run_hdm8_postfilter_sweep_t4", NoRemote)
    with pytest.raises(SystemExit, match="requires --lane-id and --instance-job-id"):
        mod.main(
            str(archive),
            str(tmp_path / "out"),
            n_pairs=2,
            modes="none;even_bias:1",
        )


def test_validate_sweep_payload_accepts_per_pair_cuda_proxy() -> None:
    mod = _load_module()
    payload = {
        "axis": mod.AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_sha256": "a" * 64,
        "archive_bytes": 123,
        "n_pairs": 2,
        "modes": [
            {
                "mode": "none",
                "n_pairs": 2,
                "avg_posenet_dist": 0.1,
                "avg_segnet_dist": 0.01,
                "score_proxy": 0.5,
                "delta_vs_none": 0.0,
                "pair_posenet_dist": [0.1, 0.1],
                "pair_segnet_dist": [0.01, 0.01],
            },
            {
                "mode": "even_bias:1",
                "n_pairs": 2,
                "avg_posenet_dist": 0.08,
                "avg_segnet_dist": 0.01,
                "score_proxy": 0.45,
                "delta_vs_none": -0.05,
                "pair_posenet_dist": [0.07, 0.09],
                "pair_segnet_dist": [0.01, 0.01],
            },
        ],
        "best": {"mode": "even_bias:1"},
    }

    errors = mod._validate_sweep_payload(
        payload,
        expected_modes=["none", "even_bias:1"],
        expected_n_pairs=2,
        include_per_pair=True,
        expected_archive_sha256="a" * 64,
        expected_archive_size_bytes=123,
    )

    assert errors == []


def test_validate_sweep_payload_rejects_score_claims_and_missing_pairs() -> None:
    mod = _load_module()
    payload = {
        "axis": mod.AXIS,
        "score_claim": True,
        "promotion_eligible": False,
        "archive_sha256": "a" * 64,
        "archive_bytes": 123,
        "n_pairs": 2,
        "modes": [
            {
                "mode": "none",
                "n_pairs": 2,
                "avg_posenet_dist": 0.1,
                "avg_segnet_dist": 0.01,
                "score_proxy": 0.5,
                "delta_vs_none": 0.0,
                "pair_posenet_dist": [0.1],
                "pair_segnet_dist": [0.01, 0.01],
            }
        ],
        "best": {"mode": "none"},
    }

    errors = mod._validate_sweep_payload(
        payload,
        expected_modes=["none"],
        expected_n_pairs=2,
        include_per_pair=True,
        expected_archive_sha256="a" * 64,
        expected_archive_size_bytes=123,
    )

    assert "score_claim must be false" in errors
    assert any("pair_posenet_dist length mismatch" in error for error in errors)
