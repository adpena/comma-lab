# SPDX-License-Identifier: MIT
"""NO-FAKE fail-closed tests for SNeRV decoder mode assignment probes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.analysis import snerv_decoder_mode_assignment_probe as probe


def test_parse_mode_plan_accepts_heuristic_and_explicit_modes() -> None:
    label, modes = probe.parse_mode_plan("magnitude", levels=1)
    assert label == "magnitude_heuristic"
    assert modes is None

    label, modes = probe.parse_mode_plan("fp16,i4,0", levels=1)
    assert label == "explicit_zero1_int41_fp161"
    assert modes == ("fp16", "int4", "zero")

    label, modes = probe.parse_mode_plan("f32,i4,0", levels=1)
    assert label == "explicit_zero1_int41_fp321"
    assert modes == ("fp32", "int4", "zero")


def test_parse_mode_plan_rejects_wrong_count() -> None:
    with pytest.raises(probe.SnervDecoderModeProbeError, match="expected 6"):
        probe.parse_mode_plan("fp16,int4,int4", levels=2)


def test_probe_is_fail_closed_and_selects_best_verified_candidate(monkeypatch) -> None:
    calls = []

    def fake_run_snerv_advisory(**kwargs):
        calls.append(kwargs)
        modes = kwargs["decoder_payload_mixed_modes"]
        if modes is None:
            return _fake_result(
                label="heuristic",
                source="magnitude_heuristic",
                histogram={"zero": 0, "int2": 0, "int4": 2, "int8": 0, "fp16": 1},
                score=1.7,
                verified=True,
            )
        return _fake_result(
            label="explicit",
            source="explicit",
            histogram={"zero": 1, "int2": 0, "int4": 1, "int8": 0, "fp16": 1},
            score=1.4,
            verified=True,
        )

    monkeypatch.setattr(probe, "run_snerv_advisory", fake_run_snerv_advisory)

    payload = probe.run_snerv_decoder_mode_assignment_probe(
        mode_plans=("magnitude_heuristic", "fp16,int4,zero"),
        n_pairs=1,
        levels=1,
        bits_per_coeff=2.0,
    )

    assert payload["schema"] == probe.SCHEMA
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["exact_or_full_video_launched"] is False
    assert payload["best_plan_label"] == "explicit_zero1_int41_fp161"
    assert payload["best_plan_score_linf_advisory"] == 1.4
    assert "full_600_pair_receiver_replay_missing" in payload["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in payload["blockers"]
    assert calls[0]["decoder_payload_codec"] == "mixed_magnitude_symmetric"
    assert calls[0]["decoder_payload_mixed_modes"] is None
    assert calls[1]["decoder_payload_mixed_modes"] == ("fp16", "int4", "zero")
    assert payload["candidates"][1]["mode_assignment_source"] == "explicit"
    assert payload["candidates"][1]["mode_histogram"]["zero"] == 1


def test_unverified_candidates_do_not_become_best(monkeypatch) -> None:
    def fake_run_snerv_advisory(**_kwargs):
        return _fake_result(
            label="bad",
            source="explicit",
            histogram={"zero": 0, "int2": 0, "int4": 3, "int8": 0, "fp16": 0},
            score=0.1,
            verified=False,
        )

    monkeypatch.setattr(probe, "run_snerv_advisory", fake_run_snerv_advisory)

    payload = probe.run_snerv_decoder_mode_assignment_probe(
        mode_plans=("int4,int4,int4",),
        n_pairs=1,
        levels=1,
    )

    assert payload["best_plan_label"] is None
    assert "no_receiver_replay_verified_candidate" in payload["blockers"]
    assert payload["candidates"][0]["score_claim"] is False


def test_probe_exports_receiver_packets_when_requested(tmp_path, monkeypatch) -> None:
    def fake_run_snerv_advisory(**_kwargs):
        return _fake_result(
            label="explicit",
            source="explicit",
            histogram={"zero": 1, "int2": 1, "int4": 1, "int8": 0, "fp16": 0},
            score=1.2,
            verified=True,
            packet=b"receiver-packet",
        )

    monkeypatch.setattr(probe, "run_snerv_advisory", fake_run_snerv_advisory)

    payload = probe.run_snerv_decoder_mode_assignment_probe(
        mode_plans=("zero,int2,int4",),
        n_pairs=1,
        levels=1,
        receiver_packet_dir=tmp_path / "packets",
    )

    candidate = payload["candidates"][0]
    packet_path = candidate["receiver_archive_packet_path"]
    assert packet_path is not None
    assert candidate["receiver_archive_packet_export"]["path"] == packet_path
    assert candidate["receiver_archive_packet_export"]["bytes"] == len(
        b"receiver-packet"
    )
    assert candidate["receiver_archive_packet_export"]["contest_archive_zip"] is False
    assert (
        candidate["receiver_archive_packet_export"]["kind"]
        == "snerv_receiver_packet_snar1_not_contest_archive_zip"
    )
    assert candidate["receiver_archive_packet_is_contest_archive_zip"] is False
    assert candidate["archive_path"] is None
    assert candidate["candidate_archive_path"] is None
    assert candidate["contest_archive_zip_path"] is None
    assert candidate["blockers"] == [
        "full_600_pair_receiver_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "not_packaged_as_contest_archive_zip",
    ]
    assert (tmp_path / "packets").is_dir()
    assert Path(packet_path).read_bytes() == b"receiver-packet"


def test_probe_packet_export_requires_raw_receiver_bytes(tmp_path, monkeypatch) -> None:
    def fake_run_snerv_advisory(**_kwargs):
        return _fake_result(
            label="redacted",
            source="explicit",
            histogram={"zero": 0, "int2": 0, "int4": 3, "int8": 0, "fp16": 0},
            score=1.2,
            verified=True,
            packet=None,
        )

    monkeypatch.setattr(probe, "run_snerv_advisory", fake_run_snerv_advisory)

    payload = probe.run_snerv_decoder_mode_assignment_probe(
        mode_plans=("int4,int4,int4",),
        n_pairs=1,
        levels=1,
        receiver_packet_dir=tmp_path / "packets",
    )

    candidate = payload["candidates"][0]
    assert candidate["receiver_archive_packet_path"] is None
    assert candidate["receiver_archive_packet_export"] == {}
    assert candidate["archive_path"] is None
    assert candidate["candidate_archive_path"] is None
    assert candidate["contest_archive_zip_path"] is None
    assert "receiver_packet_export_requested_but_raw_packet_missing" in candidate[
        "blockers"
    ]
    assert "receiver_packet_export_requested_but_raw_packet_missing" in payload[
        "blockers"
    ]


def _fake_result(
    *,
    label: str,
    source: str,
    histogram: dict[str, int],
    score: float,
    verified: bool,
    packet: bytes | None = b"fake-packet",
):
    packet_sha = probe.hashlib.sha256(packet or b"").hexdigest()
    payload = {
        "decoder_payload_header": {
            "mode_assignment_source": source,
            "mode_histogram": histogram,
            "payload_bytes": 27,
        },
        "decoder_bytes": 123,
        "receiver_archive_packet_bytes": 456,
        "receiver_archive_packet": {"sha256": packet_sha, "redacted": True},
        "archive_bytes_total": 456,
        "receiver_archive_replay_verified": verified,
        "receiver_archive_replay_error": None if verified else "fake failure",
        "d_seg_mean_linf": 0.01,
        "d_pose_mean_linf": 0.02,
        "score_linf": score,
        "d_seg_mean_l2": 0.03,
        "d_pose_mean_l2": 0.04,
        "score_l2": score + 0.5,
        "rate_term": 0.001,
        "archive_byte_closure_blockers": [
            "full_600_pair_receiver_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
            "not_packaged_as_contest_archive_zip",
        ],
    }
    attrs = {"as_jsonable": lambda: {**payload, "source_label": label}}
    if packet is not None:
        attrs["receiver_archive_packet"] = packet
    return SimpleNamespace(**attrs)
