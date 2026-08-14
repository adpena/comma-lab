from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments import ddm_js8_seg_stack_compensated_rerun as js8
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _record(name: str, digest: str) -> dict[str, object]:
    return {"path": f"/retained/{name}", "bytes": 17, "sha256": digest * 64}


def _rows(count: int = js8.EVENTS) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exact_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []
    for ordinal in range(count):
        proposal_id = f"ec1_{ordinal:04d}_id"
        event = _record(f"{proposal_id}.ec1p", "a")
        tokens = _record(f"{proposal_id}.npy", "b")
        exact_rows.append(
            {
                "proposal_id": proposal_id,
                "ordinal": ordinal,
                "pair": 7,
                "site_count": 1,
                "source_class": 2,
                "target_class": 0,
                "event_type_id": 1,
                "net_flip_gain_base_minus_candidate": 2 if ordinal < 19 else -1,
                "delta_d_pose_global_n600": 1e-9,
                "payloads": {
                    "event": event,
                    "candidate_tokens": tokens,
                    "candidate_master": _record("master.npy", "c"),
                    "candidate_pair": _record("pair.npy", "d"),
                    "indices": _record("indices.npy", "e"),
                    "candidate_scorer": {
                        "seg_input": _record("seg_input.npy", "f"),
                        "seg_logits": _record("seg_logits.npy", "1"),
                        "seg_argmax": _record("seg_argmax.npy", "2"),
                        "pose_input": _record("pose_input.npy", "3"),
                        "pose_output6": _record("pose_output.npy", "4"),
                    },
                },
            }
        )
        proposal_rows.append(
            {
                "proposal_id": proposal_id,
                "pair": 7,
                "site_count": 1,
                "source_class": "Undrivable",
                "target_class": "Road",
                "event_type": "boundary_offset",
                "source_archive_sha256": js8.CP135_ARCHIVE_SHA256,
                "consumer_payloads": {
                    "event.ec1p": event,
                    "candidate_tokens.uint8.npy": tokens,
                },
            }
        )
    return exact_rows, proposal_rows


def test_exact_join_and_reach_summary_are_denominator_closed() -> None:
    exact, proposals = _rows()
    joined = js8.join_rows(exact, proposals)
    summary = js8.summarize(joined)
    assert summary["event_count"] == 200
    assert summary["positive_event_count"] == 19
    assert summary["optimistic_positive_singleton_reach_flips"] == 38
    assert summary["all_singletons_net_flip_gain"] == 38 - 181
    assert summary["optimistic_zero_pose_tax_seg_score_improvement"] == pytest.approx(
        100 * 38 / js8.PIXELS
    )


def test_join_fails_closed_on_payload_identity_mismatch() -> None:
    exact, proposals = _rows()
    proposals[12] = copy.deepcopy(proposals[12])
    proposals[12]["consumer_payloads"]["event.ec1p"]["sha256"] = "9" * 64  # type: ignore[index]
    with pytest.raises(js8.JS8Error, match="event bytes differ"):
        js8.join_rows(exact, proposals)


def test_honest_ceiling_seals_no_fire_below_1000_flips() -> None:
    exact, proposals = _rows()
    decision = js8.decide(js8.summarize(js8.join_rows(exact, proposals)))
    assert decision["sealed_no_fire"] is True
    assert decision["disposition"] == "FOLDED_EXACT_REACH_CEILING"
    assert decision["next_route"] == "implicit_joint_distortion_conditioning"


def test_ceiling_does_not_block_a_hypothetical_reach_above_floor() -> None:
    summary = {"optimistic_positive_singleton_reach_flips": 1000}
    decision = js8.decide(summary)
    assert decision["sealed_no_fire"] is False
    assert decision["disposition"] == "READY_FOR_PER_EVENT_COMPENSATION_AND_JOINT_REMEASURE"


def test_js8_runner_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=js8.REPO,
        strict=False,
        roots=("experiments/ddm_js8_seg_stack_compensated_rerun.py",),
    )
    assert findings == []


def test_default_output_is_on_the_primary_ssd_tier() -> None:
    assert js8.DEFAULT_OUTPUT.is_relative_to(Path("/Volumes/VertigoDataTier/pact"))
