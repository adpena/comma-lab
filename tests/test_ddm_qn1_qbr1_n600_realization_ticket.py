# SPDX-License-Identifier: MIT
"""Guards for the QN1 QBR1 n600 realization ticket generator.

Every guard exercises real behaviour: the refusals are asserted by message, the emitted argv is
parsed back through the module's own argparse (so an invented flag fails the suite), and the score
law is cross-checked against two independently recorded numbers (the burn's own ``rate_exact`` and
the CLAUDE.md AFR1 rate).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from experiments import ddm_qn1_qbr1_n600_realization_ticket as qn1


def _seed_row(seed: int, s_hat: float, *, win: bool = True, pose: bool = True) -> dict[str, Any]:
    return {
        "seed": seed,
        "control_S_hat": s_hat + 1.0,
        "treatment_S_hat": s_hat,
        "delta_treatment_minus_control": -1.0,
        "treatment_win": win,
        "treatment_pose_corner_pass": pose,
    }


def _adjudication(rows: list[dict[str, Any]], *, disposition: str = qn1.LIVE_OUTCOME) -> dict[str, Any]:
    return {
        "schema": qn1.ADJUDICATION_SCHEMA_ID,
        "disposition": disposition,
        "treatment_wins": sum(1 for row in rows if row["treatment_win"]),
        "treatment_pose_corner_passes": sum(1 for row in rows if row["treatment_pose_corner_pass"]),
        "seed_rows": rows,
        "source_results": [
            {
                "path": f"/Volumes/APDataStore/pact/x/runs/seed_{row['seed']}/{qn1.TREATMENT_ARM}/RESULT.json",
                "bytes": 1,
                "sha256": "a" * 64,
            }
            for row in rows
        ],
    }


# --------------------------------------------------------------------------------------------------
# chunk plan
# --------------------------------------------------------------------------------------------------


def test_chunk_plan_partitions_600_into_20_contiguous_30_pair_spans() -> None:
    plan = qn1.chunk_plan()
    assert len(plan) == 20
    assert [row["pairs"] for row in plan] == [30] * 20
    assert plan[0]["first_pair"] == 0
    assert plan[-1]["last_pair"] == 599
    covered: list[int] = []
    for row in plan:
        covered.extend(range(row["first_pair"], row["last_pair"] + 1))
    assert covered == list(range(600))
    assert plan[0]["payload_name"] == "scorer_pairs_0000_0029.npz"
    assert plan[-1]["payload_name"] == "scorer_pairs_0570_0599.npz"


def test_chunk_plan_handles_a_ragged_tail_without_dropping_pairs() -> None:
    plan = qn1.chunk_plan(n=70, chunk_pairs=30)
    assert [row["pairs"] for row in plan] == [30, 30, 10]
    assert plan[-1]["last_pair"] == 69


def test_chunk_plan_refuses_degenerate_sizes() -> None:
    with pytest.raises(qn1.QN1Error):
        qn1.chunk_plan(n=0)
    with pytest.raises(qn1.QN1Error):
        qn1.chunk_plan(chunk_pairs=0)


# --------------------------------------------------------------------------------------------------
# score law
# --------------------------------------------------------------------------------------------------


def test_score_law_reproduces_the_burn_recorded_rate_exact() -> None:
    # MEASURED: the QBR1 milestone for seed_20260902/control_native100 step 2000 records
    # archive_bytes_exact 106626 and rate_exact 0.07099787673560465.
    row = qn1.score_law(d_seg=0.0, d_pose=0.0, archive_bytes=106_626)
    assert row["rate"] == pytest.approx(0.07099787673560465, abs=1e-15)
    assert row["S"] == pytest.approx(0.07099787673560465, abs=1e-15)
    assert row["byte_feasible"] is True


def test_score_law_reproduces_the_afr1_rate_from_claude_md() -> None:
    # CLAUDE.md goal banner: afr1 rate = 0.11985594327989708 at 180,002 archive bytes.
    row = qn1.score_law(d_seg=0.0, d_pose=0.0, archive_bytes=qn1.AFR1_ARCHIVE_BYTES)
    assert row["rate"] == pytest.approx(0.11985594327989708, abs=1e-15)
    assert row["byte_feasible"] is False


def test_score_law_composes_all_three_terms() -> None:
    row = qn1.score_law(d_seg=0.001, d_pose=0.0004, archive_bytes=106_626)
    assert row["seg_term"] == pytest.approx(0.1)
    assert row["pose_term"] == pytest.approx(0.06324555320336758)
    assert row["S"] == pytest.approx(row["seg_term"] + row["pose_term"] + row["rate"])
    assert row["delta_vs_0_12"] == pytest.approx(row["S"] - 0.12)


def test_score_law_refuses_impossible_inputs() -> None:
    with pytest.raises(qn1.QN1Error):
        qn1.score_law(d_seg=0.0, d_pose=0.0, archive_bytes=0)
    with pytest.raises(qn1.QN1Error):
        qn1.score_law(d_seg=-1e-9, d_pose=0.0, archive_bytes=100)
    with pytest.raises(qn1.QN1Error):
        qn1.score_law(d_seg=0.0, d_pose=-1e-9, archive_bytes=100)


def test_falsifier_row_is_a_regime_marker_and_says_so() -> None:
    """The QXR1 falsifier does NOT by itself clear 0.12; the ticket must not imply that it does."""
    row = qn1.falsifier_row(archive_bytes=106_626)
    assert row["falsifier_d_seg"] == 0.01
    assert row["falsifier_d_pose"] == 1.25e-4
    assert row["falsifier_byte_ceiling"] == 137_986
    assert row["falsifier_S_if_exactly_met"] == pytest.approx(
        1.0 + (10.0 * 1.25e-4) ** 0.5 + 25.0 * 106_626 / qn1.RATE_DENOMINATOR
    )
    assert row["falsifier_alone_clears_0_12"] is False
    assert row["falsifier_is_a_regime_marker_not_the_target"] is True
    required = row["d_seg_required_for_0_12_at_the_falsifier_pose"]
    assert required == pytest.approx(
        (0.12 - 25.0 * 106_626 / qn1.RATE_DENOMINATOR - (10.0 * 1.25e-4) ** 0.5) / 100.0
    )
    assert 0.0 < required < 0.01
    assert row["falsifier_d_seg_over_required_d_seg"] == pytest.approx(0.01 / required)
    assert row["status"] == "UNTESTED_UNTIL_MAIN_SCORER_FIRE"
    assert row["no_distortion_transfer"] is True


# --------------------------------------------------------------------------------------------------
# adjudication outcome refusals + winner selection
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("disposition", [qn1.MIXED_OUTCOME, qn1.CLOSED_OUTCOME, "SOMETHING_ELSE", None])
def test_select_winner_refuses_every_non_live_outcome(disposition: str | None) -> None:
    rows = [_seed_row(20260902, 0.4)]
    with pytest.raises(qn1.QN1Error, match="fires only on"):
        qn1.select_winner(_adjudication(rows, disposition=disposition))  # type: ignore[arg-type]


def test_select_winner_refuses_a_drifted_schema() -> None:
    payload = _adjudication([_seed_row(20260902, 0.4)])
    payload["schema"] = "ddm_qbr1_adjudication_result.v2"
    with pytest.raises(qn1.QN1Error, match="schema drifted"):
        qn1.select_winner(payload)


def test_select_winner_refuses_when_no_cell_passes_its_own_pose_corner() -> None:
    rows = [
        _seed_row(20260902, 0.40, win=True, pose=False),
        _seed_row(20260903, 0.41, win=True, pose=False),
        _seed_row(20260904, 0.42, win=False, pose=True),
    ]
    with pytest.raises(qn1.QN1Error, match="no single cell is buyable"):
        qn1.select_winner(_adjudication(rows))


def test_select_winner_refuses_empty_seed_rows() -> None:
    payload = _adjudication([])
    payload["disposition"] = qn1.LIVE_OUTCOME
    with pytest.raises(qn1.QN1Error, match="no seed rows"):
        qn1.select_winner(payload)


def test_select_winner_picks_the_lowest_S_hat_among_eligible_cells() -> None:
    rows = [
        _seed_row(20260902, 0.40),
        _seed_row(20260903, 0.31),  # lowest overall but pose fails below
        _seed_row(20260904, 0.35),
    ]
    rows[1]["treatment_pose_corner_pass"] = False
    winner = qn1.select_winner(_adjudication(rows))
    assert winner["seed"] == 20260904
    assert winner["cell_id"] == f"seed_20260904_{qn1.TREATMENT_ARM}"
    assert winner["eligible_seeds"] == [20260902, 20260904]
    assert winner["treatment_S_hat_n32"] == pytest.approx(0.35)
    assert winner["n32_numbers_are_selection_statistics_only"] is True


def test_select_winner_breaks_exact_ties_on_the_lowest_seed() -> None:
    rows = [_seed_row(20260904, 0.33), _seed_row(20260902, 0.33)]
    assert qn1.select_winner(_adjudication(rows))["seed"] == 20260902


def test_resolve_winner_result_refuses_when_the_source_result_is_not_named_once() -> None:
    payload = _adjudication([_seed_row(20260902, 0.4)])
    payload["source_results"] = []
    winner = {"seed": 20260902, "cell_id": f"seed_20260902_{qn1.TREATMENT_ARM}"}
    with pytest.raises(qn1.QN1Error, match="exactly one"):
        qn1.resolve_winner_result(payload, winner)


def test_milestone_from_result_requires_completion_and_the_5000_endpoint() -> None:
    endpoint = {"step": 5000}
    with pytest.raises(qn1.QN1Error, match="not complete"):
        qn1.milestone_from_result(
            {"schema": qn1.RESULT_SCHEMA_ID, "complete": False, "milestones": [endpoint]}
        )
    with pytest.raises(qn1.QN1Error, match="schema drifted"):
        qn1.milestone_from_result({"schema": "other", "complete": True, "milestones": [endpoint]})
    with pytest.raises(qn1.QN1Error, match="not 5000"):
        qn1.milestone_from_result(
            {"schema": qn1.RESULT_SCHEMA_ID, "complete": True, "milestones": [{"step": 4000}]}
        )
    assert (
        qn1.milestone_from_result({"schema": qn1.RESULT_SCHEMA_ID, "complete": True, "milestones": [endpoint]})
        is endpoint
    )


# --------------------------------------------------------------------------------------------------
# claim + custody guards
# --------------------------------------------------------------------------------------------------


def test_scorer_claim_refuses_the_placeholder_and_foreign_lanes() -> None:
    with pytest.raises(qn1.QN1Error, match="placeholder"):
        qn1.assert_scorer_claim_id(qn1.CLAIM_PLACEHOLDER)
    with pytest.raises(qn1.QN1Error, match="placeholder"):
        qn1.assert_scorer_claim_id("   ")
    with pytest.raises(qn1.QN1Error, match="QN1-owned lane id"):
        qn1.assert_scorer_claim_id("ddm_br2_scorer_20260903")
    assert qn1.assert_scorer_claim_id("ddm_qn1_scorer_20260903") == "ddm_qn1_scorer_20260903"


def test_output_root_guard_refuses_the_live_burn_custody_tree() -> None:
    for target in ("runs", "authorized_configs", "sealed_configs", "."):
        with pytest.raises(qn1.QN1Error, match="never sit inside the live burn custody root"):
            qn1.assert_output_root(qn1.AP_BURN_ROOT / target)


def test_output_root_guard_refuses_anything_outside_qn1_custody(tmp_path: Path) -> None:
    with pytest.raises(qn1.QN1Error, match="may only write under"):
        qn1.assert_output_root(tmp_path)
    assert qn1.assert_output_root(qn1.OUTPUT_ROOT / "nested") == (qn1.OUTPUT_ROOT / "nested").resolve()


def test_realization_root_guard_refuses_the_live_burn_custody_tree(tmp_path: Path) -> None:
    """The ticket must never NAME the burn's tree as the realization output, even though QN1 never writes it."""
    with pytest.raises(qn1.QN1Error, match="realization output root"):
        qn1.assert_not_burn_custody(qn1.AP_BURN_ROOT / "runs", what="the QN1 realization output root")
    assert qn1.assert_not_burn_custody(tmp_path, what="x") == tmp_path.resolve()
    assert qn1.assert_not_burn_custody(qn1.REALIZATION_ROOT, what="x") == qn1.REALIZATION_ROOT.resolve()


def test_preregistration_guard_rederives_the_burn_contract(tmp_path: Path) -> None:
    good = {
        "schema": "ddm_qbr1_preregistered_adjudication.v1",
        "population_n": 600,
        "endpoint_step": 5000,
        "no_n600_buy_before_sign_repeats": True,
    }
    path = tmp_path / "ADJUDICATION_SCHEMA.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert qn1.assert_preregistration(path)["endpoint_step"] == 5000
    for key, value, pattern in (
        ("schema", "ddm_qbr1_preregistered_adjudication.v2", "schema drifted"),
        ("population_n", 32, "is not the n600 buy"),
        ("endpoint_step", 4000, "endpoint_step"),
        ("no_n600_buy_before_sign_repeats", False, "no_n600_buy_before_sign_repeats"),
    ):
        path.write_text(json.dumps({**good, key: value}), encoding="utf-8")
        with pytest.raises(qn1.QN1Error, match=pattern):
            qn1.assert_preregistration(path)


# --------------------------------------------------------------------------------------------------
# object binding
# --------------------------------------------------------------------------------------------------


def _container(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "reencode_payloads.tar"
    with tarfile.open(path, mode="w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def _milestone(container: Path, *, archive: bytes, packet: bytes) -> dict[str, Any]:
    return {
        "schema": qn1.MILESTONE_SCHEMA_ID,
        "step": 5000,
        "archive_bytes_exact": len(archive),
        "rate_exact": 25.0 * len(archive) / qn1.RATE_DENOMINATOR,
        "reencode": {
            "archive": {
                "container": str(container),
                "container_member": "archive.zip",
                "bytes": len(archive),
                "sha256": qn1.sha256_bytes(archive),
            },
            "packet": {
                "container": str(container),
                "container_member": "packet.qbf",
                "bytes": len(packet),
                "sha256": qn1.sha256_bytes(packet),
            },
        },
    }


def test_bind_object_refuses_a_drifted_milestone_schema(tmp_path: Path) -> None:
    container = _container(tmp_path, {"archive.zip": b"x", "packet.qbf": b"y"})
    milestone = _milestone(container, archive=b"x", packet=b"y")
    milestone["schema"] = "ddm_qbr1_realized_milestone.v2"
    with pytest.raises(qn1.QN1Error, match="milestone schema drifted"):
        qn1.bind_object(milestone)


def test_bind_object_refuses_a_missing_reencode_block(tmp_path: Path) -> None:
    container = _container(tmp_path, {"archive.zip": b"x", "packet.qbf": b"y"})
    milestone = _milestone(container, archive=b"x", packet=b"y")
    del milestone["reencode"]
    with pytest.raises(qn1.QN1Error, match="no reencode block"):
        qn1.bind_object(milestone)


def test_bind_object_refuses_when_the_retained_archive_sha_drifts(tmp_path: Path) -> None:
    container = _container(tmp_path, {"archive.zip": b"x", "packet.qbf": b"y"})
    milestone = _milestone(container, archive=b"x", packet=b"y")
    milestone["reencode"]["archive"]["sha256"] = "0" * 64
    with pytest.raises(qn1.QN1Error, match="bound archive bytes or SHA-256 differ"):
        qn1.bind_object(milestone)


def test_bind_object_refuses_when_the_rate_does_not_follow_the_score_law(tmp_path: Path) -> None:
    container = _container(tmp_path, {"archive.zip": b"x", "packet.qbf": b"y"})
    milestone = _milestone(container, archive=b"x", packet=b"y")
    milestone["rate_exact"] = 0.5
    with pytest.raises(qn1.QN1Error, match="not the score law"):
        qn1.bind_object(milestone)


def test_bind_object_refuses_an_absent_container_member(tmp_path: Path) -> None:
    """Regression: an absent member used to escape as an untyped KeyError instead of refusing."""
    container = _container(tmp_path, {"archive.zip": b"x"})
    milestone = _milestone(container, archive=b"x", packet=b"y")
    with pytest.raises(qn1.QN1Error, match="container member is absent"):
        qn1.bind_object(milestone)


def test_bind_object_refuses_a_non_regular_container_member(tmp_path: Path) -> None:
    path = tmp_path / "reencode_payloads.tar"
    with tarfile.open(path, mode="w") as tar:
        info = tarfile.TarInfo("archive.zip")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        member = tarfile.TarInfo("packet.qbf")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"y"))
    milestone = _milestone(path, archive=b"x", packet=b"y")
    with pytest.raises(qn1.QN1Error, match="container member is unreadable"):
        qn1.bind_object(milestone)


# --------------------------------------------------------------------------------------------------
# fire-order argv: parsed back through this module's own argparse
# --------------------------------------------------------------------------------------------------


def test_realization_argv_is_verbatim_and_parses_through_the_real_argparse(tmp_path: Path) -> None:
    argv = qn1.realization_argv(
        runner=Path("/repo/experiments/ddm_qn1_qbr1_n600_realization_ticket.py"),
        output=Path("/out/root"),
        ticket_path=tmp_path / "FIRE_ORDER.json",
        claim_id="ddm_qn1_scorer_20260903",
        python=Path("/repo/.venv/bin/python"),
    )
    assert argv[0] == "/repo/.venv/bin/python"
    assert argv[1] == "/repo/experiments/ddm_qn1_qbr1_n600_realization_ticket.py"
    assert argv[2] == "realize"
    # The never-invent-flags proof: the module's own parser accepts every emitted flag.
    parsed = qn1.build_parser().parse_args(argv[2:])
    assert parsed.action == "realize"
    assert parsed.output == Path("/out/root")
    assert parsed.resume_from == Path("/out/root")
    assert parsed.scorer_claim_id == "ddm_qn1_scorer_20260903"
    assert parsed.launch_authorized is True
    assert parsed.ticket == tmp_path / "FIRE_ORDER.json"


def test_realize_flag_names_match_the_br2_protocol_source() -> None:
    """Never-invent-a-flag: QN1's realize flags are BR2's own, read from BR2's argparse source."""
    source = qn1.BR2_RUNNER.read_text(encoding="utf-8")
    for flag in ("--output", "--resume-from", "--scorer-claim-id", "--launch-authorized"):
        assert f'realize.add_argument("{flag}"' in source, flag
    # QN1 adds exactly one flag BR2 does not have: the ticket that carries the object.
    assert '"--ticket"' not in source
    parsed = qn1.build_parser().parse_args(
        ["realize", "--ticket", "/t.json", "--resume-from", "/o", "--scorer-claim-id", "ddm_qn1_x"]
    )
    assert parsed.launch_authorized is False
    assert parsed.output == qn1.REALIZATION_ROOT


def test_ticket_and_dry_run_subcommands_reject_unknown_flags() -> None:
    with pytest.raises(SystemExit):
        qn1.build_parser().parse_args(["ticket", "--not-a-real-flag", "x"])
    with pytest.raises(SystemExit):
        qn1.build_parser().parse_args(["realize", "--scorer-claim-id", "ddm_qn1_x"])  # missing --resume-from


# --------------------------------------------------------------------------------------------------
# realize refusals (reachable without torch, a scorer, or any payload)
# --------------------------------------------------------------------------------------------------


def _write_ticket(path: Path, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "schema": qn1.SCHEMA,
        "mode": "LIVE",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "realization": {
            "scorer_claim_id": "ddm_qn1_scorer_20260903",
            "chunk_plan": qn1.chunk_plan(),
            "output_root": str(path.parent),
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_realize_refuses_without_explicit_launch_authorization(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json")
    with pytest.raises(qn1.QN1Error, match="explicit launch authorization"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path,
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=False,
        )


def test_realize_refuses_a_resume_root_that_is_not_the_output_root(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json")
    with pytest.raises(qn1.QN1Error, match="must name the exact QN1 realization output root"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path / "elsewhere",
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=True,
        )


def test_realize_refuses_a_dry_run_ticket(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json", mode="DRY_RUN_PLUMBING", disposition="DRY_RUN_NOT_FIREABLE")
    with pytest.raises(qn1.QN1Error, match="dry-run tickets are not fireable"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path,
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=True,
        )


def test_realize_refuses_a_ticket_schema_drift(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json", schema="ddm_qn1_n600_realization_fire_order.v2")
    with pytest.raises(qn1.QN1Error, match="ticket schema drifted"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path,
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=True,
        )


def test_realize_refuses_an_output_root_the_ticket_did_not_seal(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json")
    ticket_payload = json.loads(ticket.read_text(encoding="utf-8"))
    ticket_payload["realization"]["output_root"] = "/some/other/root"
    ticket.write_text(json.dumps(ticket_payload), encoding="utf-8")
    with pytest.raises(qn1.QN1Error, match="not the output root sealed in the ticket"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path,
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=True,
        )


def test_realize_refuses_an_output_root_inside_the_live_burn_custody(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json")
    burn = qn1.AP_BURN_ROOT / "runs"
    with pytest.raises(qn1.QN1Error, match="never sit inside the live burn custody root"):
        qn1.realize(
            ticket_path=ticket,
            output=burn,
            resume_from=burn,
            claim_id="ddm_qn1_scorer_20260903",
            launch_authorized=True,
        )


def test_realize_refuses_a_claim_id_that_does_not_match_the_sealed_ticket(tmp_path: Path) -> None:
    ticket = _write_ticket(tmp_path / "t.json")
    with pytest.raises(qn1.QN1Error, match="does not match the sealed ticket"):
        qn1.realize(
            ticket_path=ticket,
            output=tmp_path,
            resume_from=tmp_path,
            claim_id="ddm_qn1_scorer_other",
            launch_authorized=True,
        )


# --------------------------------------------------------------------------------------------------
# aggregate: the n600 denominators and the BR2 equivalence
# --------------------------------------------------------------------------------------------------


def _pair_rows(*, seg_errors_each: int, pose_sse_each: float, class_names: list[str]) -> list[dict[str, Any]]:
    pixels = qn1.H * qn1.W
    rows = []
    for pair_id in range(qn1.N):
        per_class = [
            {"class_id": index, "class_name": name, "target_pixels": pixels if index == 0 else 0, "errors": 0}
            for index, name in enumerate(class_names)
        ]
        per_class[0]["errors"] = seg_errors_each
        rows.append(
            {
                "pair_id": pair_id,
                "seg_errors": seg_errors_each,
                "seg_pixels": pixels,
                "pose_squared_error_sum": pose_sse_each,
                "pose_values": 6,
                "per_class": per_class,
            }
        )
    return rows


def test_aggregate_applies_the_score_law_to_the_ticket_archive_bytes() -> None:
    names = ["a", "b", "c", "d", "e"]
    rows = _pair_rows(seg_errors_each=1_000, pose_sse_each=0.006, class_names=names)
    out = qn1.aggregate(rows, archive_bytes=106_626, class_names=names)
    assert out["d_seg"] == pytest.approx(1_000 / (qn1.H * qn1.W))
    assert out["d_pose"] == pytest.approx(0.001)
    assert out["rate"] == pytest.approx(0.07099787673560465, abs=1e-15)
    assert out["seg_pixels"] == qn1.N * qn1.H * qn1.W
    assert out["pose_values"] == qn1.N * 6
    assert out["per_class"][0]["conditional_d_seg"] == pytest.approx(out["d_seg"])
    assert out["per_class"][1]["conditional_d_seg"] is None


def test_aggregate_refuses_a_broken_per_class_partition() -> None:
    names = ["a", "b", "c", "d", "e"]
    rows = _pair_rows(seg_errors_each=10, pose_sse_each=0.0, class_names=names)
    rows[0]["per_class"][0]["errors"] = 11
    with pytest.raises(qn1.QN1Error, match="do not partition"):
        qn1.aggregate(rows, archive_bytes=106_626, class_names=names)


def test_aggregate_refuses_a_denominator_that_is_not_n600() -> None:
    names = ["a", "b", "c", "d", "e"]
    rows = _pair_rows(seg_errors_each=10, pose_sse_each=0.0, class_names=names)[:-1]
    with pytest.raises(qn1.QN1Error, match="differ from n600"):
        qn1.aggregate(rows, archive_bytes=106_626, class_names=names)


def test_aggregate_matches_br2_aggregate_when_the_archive_matches_br2() -> None:
    """QN1 must not drift from the exemplar's arithmetic; only the rate term is parameterised."""
    pytest.importorskip("torch")
    from experiments import ddm_br2_born_object_scorer_realization as br2

    names = list(br2.CLASS_NAMES)
    rows = _pair_rows(seg_errors_each=1_234, pose_sse_each=0.0042, class_names=names)
    mine = qn1.aggregate(rows, archive_bytes=br2.ARCHIVE_BYTES, class_names=names)
    theirs = br2.aggregate(rows)
    for key in ("d_seg", "d_pose", "seg_term", "pose_term", "distortion", "rate", "S"):
        assert mine[key] == pytest.approx(theirs[key], abs=1e-15), key
    assert qn1.H == br2.H and qn1.W == br2.W and qn1.N == br2.N
    assert qn1.CHUNK_PAIRS == br2.CHUNK_PAIRS
    assert qn1.RATE_DENOMINATOR == br2.RATE_DENOMINATOR


# --------------------------------------------------------------------------------------------------
# live-custody integration (skipped when the burn store is not mounted)
# --------------------------------------------------------------------------------------------------

_CUSTODY = qn1.DRY_RUN_CELL_ROOT.exists() and qn1.OUTPUT_ROOT.parent.exists()


@pytest.mark.skipif(not _CUSTODY, reason="QBR1 burn custody or the Vertigo tier is not mounted")
def test_dry_run_binds_the_live_cell_and_fires_every_refusal() -> None:
    receipt = qn1.dry_run(output=qn1.OUTPUT_ROOT, cell_root=qn1.DRY_RUN_CELL_ROOT, milestone_path=None)
    assert receipt["schema"] == qn1.DRY_RUN_SCHEMA
    assert receipt["plumbing_only"] is True
    assert receipt["cell_arm_is_a_control_not_a_treatment"] is True
    assert receipt["score_claim"] is False
    assert receipt["scorer_invocations"] == 0
    assert receipt["all_refusals_fired"] is True
    assert receipt["refusals_fired"] == receipt["refusals_total"]
    binding = receipt["binding"]
    assert binding["latent_records"] == qn1.N
    assert binding["receiver_packet_bit_identity"] is True
    assert binding["receiver_archive_roundtrip_bit_identity"] is True
    assert binding["byte_identical_to_scored_ancestor"] is False
    assert len(binding["decoded_field_digest"]) == 64
    assert receipt["chunk_plan_chunks"] == 20
    assert receipt["wrote_only_under"] == str(qn1.OUTPUT_ROOT.resolve())


@pytest.mark.skipif(not _CUSTODY, reason="QBR1 burn custody or the Vertigo tier is not mounted")
def test_dry_run_ticket_is_never_fireable() -> None:
    ticket = json.loads((qn1.OUTPUT_ROOT / "DRY_RUN_FIRE_ORDER.json").read_text(encoding="utf-8"))
    assert ticket["mode"] == "DRY_RUN_PLUMBING"
    assert ticket["disposition"] == "DRY_RUN_NOT_FIREABLE"
    assert ticket["adjudication"]["synthetic"] is True
    assert ticket["dry_run"]["cell_arm_is_a_control_not_a_treatment"] is True
    assert ticket["score_claim"] is False
    assert ticket["n32_advisory_numbers_are_not_transferred_to_n600"] is True
