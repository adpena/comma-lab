#!/usr/bin/env python3
"""Close JS8's gen-1 EC1 alphabet from the retained exact T4 singleton census.

The JS8 charter requests a local n600 advisory calibration before per-event
QS5 compensation and greedy joint composition.  Full-corpus recall found a
strictly stronger existing measurement: VD1 scored the same 200 proposal
payloads with the exact public receiver and frozen T4 scorers at n600, while
retaining every event payload, token plane, decoded frame, and scorer tensor.

This runner verifies that object identity, retains the consumed receipts,
emits the complete per-event reach table and curve, and applies JS8's stated
honest-ceiling rule.  It does not run a scorer, materialize a candidate, infer
singleton additivity as a composed result, or claim a contest score.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_js8_20260813")
VD1_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results")
VD1_FINAL: Final = VD1_ROOT / "FINAL_RESULT.json"
VD1_EVENTS: Final = VD1_ROOT / "EVENT_RESULTS.jsonl"
PROPOSALS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/"
    "follow_on/realized_acceptance_200/proposal_index.jsonl"
)
VD1_MEMO: Final = REPO / ".omx/research/ddm_vd1_census_verdict_20260812.md"
JS7_MEMO: Final = REPO / ".omx/research/ddm_js7_exact_row_verdict_20260812.md"
QS5_MEMO: Final = REPO / ".omx/research/ddm_qs5_resolve_compensation_20260813.md"
QS5_COMPILER: Final = REPO / "experiments/ddm_qs1_frame0_schur_coupled_solve.py"
QS5_T4_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/dispatch/"
    "ddm_qs5_dual_axis_20260813_r2/QS5_T4_REMOTE_RESULT.json"
)

EXPECTED_SHA256: Final = {
    VD1_FINAL: "6c53628184f55722f87fcb7e3dadc8b6c9a70025a804e00cfcbecb6674004973",
    VD1_EVENTS: "a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3",
    PROPOSALS: "599a3ac0a9c7d7e62c162fcee595194d6d3cd79685d0ceabab92e0231bd9d47e",
    VD1_MEMO: "72e37a18a874ac7bf6931730772f79392137a5857ae823b794d33a160252d30f",
    JS7_MEMO: "e887b217c8f75f43e5e950976b905778e5873d8a391b13ebdd90e1b65393053c",
    QS5_MEMO: "d401dc48209e5ba9bb2be7b7802734247135398916098dcdf0281afa56329632",
    QS5_COMPILER: "11cf05984af00e5d93edfd148b37a0f8d5aa9d3a7efa5a9eb802d3ccb58fc39c",
    QS5_T4_RECEIPT: "a4c717e2cd994c45f08eff30489d0a86ff993a305c4e9ded4b6b80bf878a01d2",
}

AXIS: Final = "[contest-CUDA T4 exact-upstream affected-pair n600 delta]"
CP135_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_ARCHIVE_BYTES: Final = 186_252
CP135_SCORE: Final = 0.16195513827824176
CP135_DPOSE_CHARTER_PIN: Final = 6.885642960696714e-6
N: Final = 600
H: Final = 384
W: Final = 512
PIXELS: Final = N * H * W
EVENTS: Final = 200
MIN_ROUTING_REACH: Final = 1_000
GOAL_BACKWARDS_FLIPS: Final = 4_314
MIN_FREE_BYTES: Final = 8 * 1024**3
CLASSES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EVENT_NAMES: Final = {
    0: "absolute_seed",
    1: "boundary_offset",
    2: "island_birth",
    3: "island_death",
    4: "lane_program_delta",
}


class JS8Error(RuntimeError):
    """A pinned-input, custody, retention, or routing invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows
    )
    return atomic_bytes(path, payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_payload_record(record: dict[str, Any], label: str) -> None:
    required = {"path", "bytes", "sha256"}
    if set(record) != required:
        raise JS8Error(f"{label}: payload record keys differ: {sorted(record)}")
    digest = str(record["sha256"])
    if int(record["bytes"]) <= 0 or len(digest) != 64:
        raise JS8Error(f"{label}: malformed payload record")


def require_local_payload(record: dict[str, Any], label: str) -> None:
    validate_payload_record(record, label)
    path = Path(str(record["path"]))
    if not path.is_file() or file_record(path) != record:
        raise JS8Error(f"{label}: local retained payload differs: {path}")


def validate_pins() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise JS8Error(f"missing pinned input: {path}")
        record = file_record(path)
        if record["sha256"] != expected:
            raise JS8Error(f"pinned input changed: {path}")
        records[path.name] = record
    return records


def join_rows(
    exact_rows: list[dict[str, Any]], proposal_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prove that VD1 measured precisely the JS8 EC1 proposal alphabet."""
    if len(exact_rows) != EVENTS or len(proposal_rows) != EVENTS:
        raise JS8Error(f"expected {EVENTS} rows, got {len(exact_rows)} / {len(proposal_rows)}")
    joined: list[dict[str, Any]] = []
    for ordinal, (exact, proposal) in enumerate(zip(exact_rows, proposal_rows, strict=True)):
        event_id = str(exact["proposal_id"])
        if int(exact["ordinal"]) != ordinal or proposal["proposal_id"] != event_id:
            raise JS8Error(f"row {ordinal}: proposal identity/order differs")
        scalar_pairs = (
            (int(exact["pair"]), int(proposal["pair"]), "pair"),
            (int(exact["site_count"]), int(proposal["site_count"]), "site_count"),
            (
                int(exact["source_class"]),
                CLASSES.index(str(proposal["source_class"])),
                "source_class",
            ),
            (
                int(exact["target_class"]),
                CLASSES.index(str(proposal["target_class"])),
                "target_class",
            ),
            (
                int(exact["event_type_id"]),
                {name: key for key, name in EVENT_NAMES.items()}[str(proposal["event_type"])],
                "event_type",
            ),
        )
        for measured, proposed, label in scalar_pairs:
            if measured != proposed:
                raise JS8Error(f"row {ordinal}: {label} differs")
        if proposal.get("source_archive_sha256") != CP135_ARCHIVE_SHA256:
            raise JS8Error(f"row {ordinal}: CP135 source archive differs")

        exact_payloads = exact["payloads"]
        proposal_payloads = proposal["consumer_payloads"]
        for exact_name, proposal_name in (
            ("event", "event.ec1p"),
            ("candidate_tokens", "candidate_tokens.uint8.npy"),
        ):
            measured_record = exact_payloads[exact_name]
            proposed_record = proposal_payloads[proposal_name]
            validate_payload_record(measured_record, f"row {ordinal} exact {exact_name}")
            validate_payload_record(proposed_record, f"row {ordinal} proposal {proposal_name}")
            if (
                measured_record["bytes"] != proposed_record["bytes"]
                or measured_record["sha256"] != proposed_record["sha256"]
            ):
                raise JS8Error(f"row {ordinal}: {exact_name} bytes differ")

        for name, record in exact_payloads.items():
            if name == "candidate_scorer":
                for scorer_name, scorer_record in record.items():
                    validate_payload_record(
                        scorer_record, f"row {ordinal} candidate_scorer.{scorer_name}"
                    )
            else:
                validate_payload_record(record, f"row {ordinal} {name}")

        gain = int(exact["net_flip_gain_base_minus_candidate"])
        joined.append(
            {
                "schema": "ddm_js8_exact_singleton_reach.v1",
                "ordinal": ordinal,
                "proposal_id": event_id,
                "pair": int(exact["pair"]),
                "event_type": EVENT_NAMES[int(exact["event_type_id"])],
                "source_class": CLASSES[int(exact["source_class"])],
                "target_class": CLASSES[int(exact["target_class"])],
                "site_count": int(exact["site_count"]),
                "exact_net_flip_gain": gain,
                "positive_singleton_reach": max(gain, 0),
                "exact_singleton_seg_score_delta": -100.0 * gain / PIXELS,
                "delta_d_pose_global_n600_uncompensated": float(
                    exact["delta_d_pose_global_n600"]
                ),
                "pose_compensation": {
                    "status": "NOT_RUN_CEILING_BRANCH_PRECEDED_COMPOSITION",
                    "mechanism_evidence": "QS5 r2 exact-object compensation landed below base",
                    "optimistic_zero_pose_tax_reach": max(gain, 0),
                },
                "authority": AXIS,
                "exact_measurement": exact,
                "source_proposal": proposal,
            }
        )
    return joined


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [int(row["exact_net_flip_gain"]) for row in rows]
    positive = [row for row in rows if int(row["positive_singleton_reach"]) > 0]
    reach_by_pair: defaultdict[int, int] = defaultdict(int)
    count_by_pair: Counter[int] = Counter()
    for row in positive:
        pair = int(row["pair"])
        reach_by_pair[pair] += int(row["positive_singleton_reach"])
        count_by_pair[pair] += 1
    total_reach = sum(max(gain, 0) for gain in gains)
    signs = Counter(1 if gain > 0 else 0 if gain == 0 else -1 for gain in gains)
    return {
        "schema": "ddm_js8_reach_summary.v1",
        "axis": AXIS,
        "score_claim": False,
        "selection_mode": "full 200-event alphabet; exact singleton affected-pair n600 census",
        "event_count": len(rows),
        "affected_pairs": sorted({int(row["pair"]) for row in rows}),
        "positive_event_count": len(positive),
        "zero_event_count": signs[0],
        "harmful_event_count": signs[-1],
        "all_singletons_net_flip_gain": sum(gains),
        "optimistic_positive_singleton_reach_flips": total_reach,
        "positive_reach_by_pair": {
            str(pair): {"events": count_by_pair[pair], "flips": reach_by_pair[pair]}
            for pair in sorted(reach_by_pair)
        },
        "optimistic_zero_pose_tax_seg_score_improvement": 100.0 * total_reach / PIXELS,
        "routing_floor_flips": MIN_ROUTING_REACH,
        "routing_floor_shortfall_flips": MIN_ROUTING_REACH - total_reach,
        "routing_floor_factor_short": MIN_ROUTING_REACH / total_reach if total_reach else None,
        "goal_backwards_required_flips_charter": GOAL_BACKWARDS_FLIPS,
        "goal_backwards_shortfall_flips": GOAL_BACKWARDS_FLIPS - total_reach,
        "goal_backwards_fraction_reached": total_reach / GOAL_BACKWARDS_FLIPS,
        "composition_interactions_measured": False,
        "ceiling_kind": (
            "optimistic additive singleton reach; sufficient for the charter's reroute gate, "
            "not a proof against nonlinear composition synergy"
        ),
    }


def reach_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["positive_singleton_reach"]), int(row["ordinal"])),
    )
    cumulative = 0
    curve: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, 1):
        cumulative += int(row["positive_singleton_reach"])
        curve.append(
            {
                "rank": rank,
                "proposal_id": row["proposal_id"],
                "ordinal": row["ordinal"],
                "pair": row["pair"],
                "exact_net_flip_gain": row["exact_net_flip_gain"],
                "cumulative_optimistic_positive_singleton_reach": cumulative,
                "cumulative_optimistic_seg_score_improvement": 100.0 * cumulative / PIXELS,
            }
        )
    return curve


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    reach = int(summary["optimistic_positive_singleton_reach_flips"])
    if reach >= MIN_ROUTING_REACH:
        disposition = "READY_FOR_PER_EVENT_COMPENSATION_AND_JOINT_REMEASURE"
        sealed = False
        reason = "exact singleton reach meets the charter's 1000-flip continuation floor"
    else:
        disposition = "FOLDED_EXACT_REACH_CEILING"
        sealed = True
        reason = (
            f"exact T4 optimistic positive singleton reach is {reach} flips, below the "
            f"charter's {MIN_ROUTING_REACH}-flip reroute floor"
        )
    return {
        "schema": "ddm_js8_sealed_no_fire_order.v1",
        "sealed_no_fire": sealed,
        "disposition": disposition,
        "reason": reason,
        "verdict_scope": (
            "INSTANCE: gen-1 EC1 200-event alphabet on CP135; not a FAMILY negative on "
            "sparse events or nonlinear composition"
        ),
        "candidate_materialized": False,
        "compensation_executed": False,
        "joint_composition_executed": False,
        "byte_close_executed": False,
        "scorer_launched": False,
        "modal_launched": False,
        "pointer_moved": False,
        "next_route": "implicit_joint_distortion_conditioning",
    }


def retain_input(source: Path, destination: Path) -> dict[str, Any]:
    payload = source.read_bytes()
    record = atomic_bytes(destination, payload)
    if record["sha256"] != EXPECTED_SHA256[source]:
        raise JS8Error(f"retained input copy differs: {source}")
    return record


def write_checkpoint(output: Path, stage: str, value: dict[str, Any]) -> dict[str, Any]:
    return atomic_json(output / "checkpoints" / f"stage_{stage}.json", value)


def execute(output: Path, resume_from: Path | None = None) -> dict[str, Any]:
    output = output.resolve()
    if resume_from is not None and resume_from.resolve() not in {
        output,
        (output / "checkpoints").resolve(),
    }:
        raise JS8Error("--resume-from must name the JS8 output or checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_BYTES:
        raise JS8Error(f"storage preflight failed: {usage.free} < {MIN_FREE_BYTES}")

    lock_path = output / "RUN.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        input_records = validate_pins()
        final_receipt = json.loads(VD1_FINAL.read_text())
        if (
            final_receipt.get("status") != "COMPLETE"
            or final_receipt.get("axis") != AXIS
            or final_receipt.get("event_count") != EVENTS
            or final_receipt.get("n600_denominator") != N
            or final_receipt["base_archive"]
            != {
                "bytes": CP135_ARCHIVE_BYTES,
                "path": final_receipt["base_archive"]["path"],
                "sha256": CP135_ARCHIVE_SHA256,
            }
            or not all(final_receipt["retention"].values())
        ):
            raise JS8Error("VD1 final receipt is incomplete or differs from JS8 pins")

        retained_dir = output / "retained" / "consumed_inputs"
        retained_inputs = {
            str(source): retain_input(source, retained_dir / source.name)
            for source in EXPECTED_SHA256
        }
        preflight = {
            "schema": "ddm_js8_preflight.v1",
            "axis": AXIS,
            "score_claim": False,
            "output": str(output),
            "resume_from": str(resume_from.resolve()) if resume_from else None,
            "storage": {"free_bytes": usage.free, "required_bytes": MIN_FREE_BYTES},
            "pinned_inputs": input_records,
            "retained_input_copies": retained_inputs,
            "provenance": {
                "argv": sys.argv,
                "runner": file_record(Path(__file__)),
                "source_git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                "consumed_measurement": final_receipt["provenance"],
            },
            "scorer_slot_claimed": False,
            "reason_no_scorer_claim": "stronger exact T4 full-alphabet census already retained",
        }
        atomic_json(output / "00_PREFLIGHT.json", preflight)
        write_checkpoint(output, "00_source_preflight", preflight)

        exact_rows = load_jsonl(VD1_EVENTS)
        proposal_rows = load_jsonl(PROPOSALS)
        joined = join_rows(exact_rows, proposal_rows)
        for ordinal, proposal in enumerate(proposal_rows):
            for name in ("event.ec1p", "candidate_tokens.uint8.npy"):
                require_local_payload(
                    proposal["consumer_payloads"][name], f"row {ordinal} local {name}"
                )
        join_receipt = {
            "schema": "ddm_js8_exact_census_join.v1",
            "joined_rows": len(joined),
            "proposal_id_order_sha256": hashlib.sha256(
                "\n".join(str(row["proposal_id"]) for row in joined).encode()
            ).hexdigest(),
            "event_and_candidate_token_payload_identity_exact": True,
            "all_local_source_event_and_token_payloads_reverified": True,
            "all_exact_t4_remote_field_records_present": True,
            "remote_payload_custody": final_receipt["retention"],
            "remote_volume_run_root": final_receipt["retention"]["volume_run_root"],
        }
        atomic_json(output / "10_EXACT_CENSUS_JOIN.json", join_receipt)
        write_checkpoint(output, "10_exact_census_join", join_receipt)

        table_record = atomic_jsonl(output / "PER_EVENT_REACH.jsonl", joined)
        curve_record = atomic_jsonl(output / "REACH_CURVE.jsonl", reach_curve(joined))
        summary = summarize(joined)
        summary["per_event_reach"] = table_record
        summary["reach_curve"] = curve_record
        summary_record = atomic_json(output / "20_REACH_SUMMARY.json", summary)
        write_checkpoint(output, "20_reach_table", {**summary, "receipt": summary_record})

        decision = decide(summary)
        decision["consumed_exact_t4_census"] = file_record(VD1_EVENTS)
        decision["qs5_compensation_evidence"] = {
            "compiler": file_record(QS5_COMPILER),
            "memo": file_record(QS5_MEMO),
            "t4_receipt": file_record(QS5_T4_RECEIPT),
            "boundary": (
                "mechanism measured on QS5 exact object; not instantiated per EC1 event because "
                "the exact seg-reach reroute gate fired first"
            ),
        }
        no_fire_record = atomic_json(output / "SEALED_NO_FIRE_ORDER.json", decision)

        queue = {
            "schema": "ddm_js8_next_action.v1",
            "action": "implicit_joint_distortion_conditioning",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "js1 stage-1 / MAIN #995 successor",
            "consumer_store": (
                "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/"
            ),
            "fire_trigger": (
                "ps135 SOLVE has landed; js1 stage 0 retains the shipping-base per-edge "
                "decomposition; no explicit edge mask ships; every complete semantic/pose/archive "
                "candidate is retained and jointly priced against an equal-parameter control"
            ),
            "negative_guards": [
                "do not iterate the gen-1 EC1 alphabet",
                "do not add an explicit edge-label stream",
                "do not reopen additive implicit-context rate calibration",
            ],
        }
        queue_record = atomic_json(output / "NEXT_ACTION.json", queue)
        write_checkpoint(
            output,
            "30_no_fire_and_reroute",
            {"decision": decision, "no_fire": no_fire_record, "next_action": queue_record},
        )

        final = {
            "schema": "ddm_js8_final_result.v1",
            "status": "COMPLETE_FOLDED_EXACT_REACH_CEILING",
            "axis": AXIS,
            "score_claim": False,
            "cp135": {
                "archive_sha256": CP135_ARCHIVE_SHA256,
                "archive_bytes": CP135_ARCHIVE_BYTES,
                "exact_score": CP135_SCORE,
                "d_pose_charter_pin": CP135_DPOSE_CHARTER_PIN,
            },
            "summary": summary,
            "decision": decision,
            "next_action": queue,
            "retention": {
                "output": str(output),
                "all_consumed_receipts_copied": True,
                "per_event_reach_retained": table_record,
                "reach_curve_retained": curve_record,
                "source_exact_fields_retained_at": final_receipt["retention"]["volume_run_root"],
                "new_candidate_payloads_materialized": False,
            },
            "measured": (
                "recalled exact T4 n600 singleton Seg/Pose deltas for all 200 identical EC1 events"
            ),
            "not_measured": (
                "per-event QS5 compensation, a jointly composed stack, archive bytes, complete S, "
                "contest-CPU, or any new exact row"
            ),
            "pointer_moved": False,
        }
        final_record = atomic_json(output / "FINAL_RESULT.json", final)
        write_checkpoint(output, "90_final", {**final, "receipt": final_record})
        return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume/revalidate the atomic stage checkpoints under this JS8 store.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute(args.output, args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
