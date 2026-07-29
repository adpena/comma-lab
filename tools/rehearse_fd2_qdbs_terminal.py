#!/usr/bin/env python
"""Bounded stale rehearsal of the FD2 QDBS terminal search.

This consumes the real stale FD2 optimizer checkpoint and its scorer-derived
first moment, but deliberately uses a deterministic test hard oracle.  It
exercises schedule precommitment and the complete compile/parse/consume/action
chain without invoking a scorer, n600 replay, or a promotion-capable path.
"""

from __future__ import annotations

import argparse
import json
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.fd2_qdbs_terminal import (
    MAX_CANDIDATE_EVALUATIONS,
    STALE_REHEARSAL_AUTHORITY_MARKER,
    ConsumedDescriptionCandidate,
    DescriptionDelta,
    DescriptionHardOracleCallbacks,
    DescriptionProposal,
    ParsedDescriptionCandidate,
    ProposalClass,
    QDBSAuthorityMode,
    RealizedJointAction,
    apply_description_proposal,
    run_fd2_qdbs_terminal,
)

DEFAULT_CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_ws3_w_joint_exact_history_20260724T132200Z/checkpoints/"
    "01_residual_bucket_realized_acceptance_intra_global000004.npz"
)
DEFAULT_FD2_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_fd2_20260728/fd2_disambiguation_receipt.json")

_MAGIC = b"FD2QDBS1"
_ISLAND_COUNT = 326
_LANE_COUNT = 24
_TEMPLATE_COUNT = 18


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_scorer_proposals(
    first_moment: np.ndarray, active_indices: tuple[int, ...]
) -> tuple[tuple[DescriptionProposal, ...], tuple[DescriptionProposal, ...]]:
    ranked = sorted(
        active_indices,
        key=lambda index: (-abs(float(first_moment[index])), index),
    )
    selected = ranked[:32]
    if len(selected) != 32 or any(float(first_moment[index]) == 0.0 for index in selected):
        raise RuntimeError("stale scorer first moment lacks 32 nonzero active coordinates")

    def signed_delta(index: int) -> DescriptionDelta:
        return DescriptionDelta(index, -1 if float(first_moment[index]) > 0.0 else 1)

    singletons = tuple(
        DescriptionProposal(
            identity=f"fd2_scorer_singleton_rank{rank:02d}_coord{index:03d}",
            proposal_class=ProposalClass.SCORER_SINGLETON,
            deltas=(signed_delta(index),),
            signal_label="stale_fd2_scorer_first_moment_abs",
            signal_value=abs(float(first_moment[index])),
        )
        for rank, index in enumerate(selected[:16])
    )
    groups: list[DescriptionProposal] = []
    for group in range(8):
        group_indices = selected[16 + 2 * group : 18 + 2 * group]
        deltas = tuple(sorted(signed_delta(index) for index in group_indices))
        groups.append(
            DescriptionProposal(
                identity=(
                    f"fd2_scorer_group_rank{group:02d}_"
                    + "_".join(f"coord{index:03d}" for index in sorted(group_indices))
                ),
                proposal_class=ProposalClass.SCORER_GROUP,
                deltas=deltas,
                signal_label="stale_fd2_scorer_first_moment_group_abs_sum",
                signal_value=sum(abs(float(first_moment[index])) for index in group_indices),
            )
        )
    return singletons, tuple(groups)


def _test_callbacks(
    target: np.ndarray,
) -> DescriptionHardOracleCallbacks:
    def compile_archive(theta: np.ndarray, _proposal: DescriptionProposal | None) -> bytes:
        return _MAGIC + struct.pack(">I", theta.size) + np.asarray(theta, dtype=">i8").tobytes()

    def parse_archive(payload: bytes) -> ParsedDescriptionCandidate:
        if len(payload) < 12 or payload[:8] != _MAGIC:
            raise RuntimeError("test description packet magic differs")
        size = struct.unpack(">I", payload[8:12])[0]
        expected = 12 + 8 * size
        if len(payload) != expected:
            raise RuntimeError("test description packet length differs")
        theta = np.frombuffer(payload[12:], dtype=">i8").astype(np.int64)
        return ParsedDescriptionCandidate(
            realized_theta=theta,
            archive_sha256=sha256(payload).hexdigest(),
            exact_parseback=True,
            value=payload,
        )

    def consume_archive(parsed: ParsedDescriptionCandidate) -> ConsumedDescriptionCandidate:
        return ConsumedDescriptionCandidate(
            realized_theta=parsed.realized_theta,
            archive_sha256=parsed.archive_sha256,
            exact_consumption=True,
            value=parsed.value,
        )

    def evaluate(
        consumed: ConsumedDescriptionCandidate,
        evaluation_idempotency_key: str,
    ) -> RealizedJointAction:
        delta = np.asarray(consumed.realized_theta, dtype=np.float64) - target
        payload = consumed.value
        if not isinstance(payload, bytes):
            raise RuntimeError("test receiver did not retain exact bytes")
        return RealizedJointAction(
            d_seg=float(np.mean(delta * delta, dtype=np.float64)) / 1_000_000.0,
            d_pose=0.001,
            archive_sha256=consumed.archive_sha256,
            archive_bytes=len(payload),
            sample_count=1,
            authority_marker=STALE_REHEARSAL_AUTHORITY_MARKER,
            custody_digest=None,
            evaluation_idempotency_key=evaluation_idempotency_key,
            realized=True,
        )

    return DescriptionHardOracleCallbacks(
        compile_archive=compile_archive,
        parse_archive=parse_archive,
        consume_archive=consume_archive,
        evaluate_joint_action_idempotent=evaluate,
    )


def rehearse(checkpoint: Path, fd2_receipt: Path, *, seed: int) -> dict[str, Any]:
    if not checkpoint.is_file() or not fd2_receipt.is_file():
        raise FileNotFoundError("stale FD2 checkpoint/receipt custody is unavailable")
    receipt = json.loads(fd2_receipt.read_text())
    if receipt.get("schema") != "ddm_fd2_posenull_gn_disambiguation.v1":
        raise RuntimeError("stale FD2 receipt schema differs")
    if receipt.get("diagnostics_gn_step1", {}).get("active_parameter_count") != 344:
        raise RuntimeError("stale FD2 active description count differs")
    if receipt.get("active_groups") != ["island_worldsheet", "shared_template_dof"]:
        raise RuntimeError("stale FD2 active groups differ")
    with np.load(checkpoint, allow_pickle=False) as state:
        theta = np.asarray(state["theta"], dtype=np.float64)
        first_moment = np.asarray(state["first_moment"], dtype=np.float64)
        typed_config_hash = str(state["__ddmjd_typed_config_hash"].item())
    expected_total = _ISLAND_COUNT + _LANE_COUNT + _TEMPLATE_COUNT
    if theta.shape != (expected_total,) or first_moment.shape != theta.shape:
        raise RuntimeError("stale FD2 description layout differs")
    active_indices = tuple(range(_ISLAND_COUNT)) + tuple(range(_ISLAND_COUNT + _LANE_COUNT, expected_total))
    if len(active_indices) != 344:
        raise AssertionError("stale FD2 active coordinate count differs")
    active_set = frozenset(active_indices)
    singletons, groups = _build_scorer_proposals(first_moment, active_indices)
    base_theta = np.rint(theta).astype(np.int64)
    inactive_indices = tuple(index for index in range(expected_total) if index not in active_set)
    if np.any(base_theta[np.asarray(inactive_indices, dtype=np.int64)] != 0):
        raise RuntimeError("stale FD2 base has nonzero inactive coordinates")
    target = base_theta.copy()
    for proposal in singletons + groups:
        target = apply_description_proposal(target, proposal)

    result = run_fd2_qdbs_terminal(
        base_theta,
        singletons,
        groups,
        _test_callbacks(target),
        active_indices=active_indices,
        seed=seed,
        authority_mode=QDBSAuthorityMode.STALE_REHEARSAL,
    )
    if result.candidate_evaluations != MAX_CANDIDATE_EVALUATIONS:
        raise AssertionError("rehearsal candidate budget differs")
    inactive_control_coordinates = sorted(
        {
            delta.index
            for proposal in result.schedule.random_controls
            for delta in proposal.deltas
            if delta.index not in active_set
        }
    )
    if inactive_control_coordinates:
        raise AssertionError("matched random controls escaped the active coordinate set")
    best_trace = min(
        (trace for trace in result.traces if trace.strict_realized_improvement),
        key=lambda trace: trace.action.action,
    )
    return {
        "schema": "ddm_eg1_qdbs_stale_rehearsal.v2",
        "generated_by": "tools/rehearse_fd2_qdbs_terminal.py",
        "rehearsal_command": (
            "PYTHONPATH=src uv run --no-project python tools/rehearse_fd2_qdbs_terminal.py --max-candidate-evals 48"
        ),
        "evidence_axis": "[deterministic stale mechanism rehearsal]",
        "authority_mode": result.authority_mode.value,
        "status": result.status.value,
        "authority_blocker": (
            "Deterministic test hard oracle and compact rehearsal codec were used; "
            "the production DDM compiler, public receiver, frozen scorers, and "
            "full-n600 upstream evaluator were not invoked."
        ),
        "candidate_schedule": {
            "seed": result.schedule.seed,
            "schedule_sha256": result.schedule.schedule_sha256,
            "active_indices_count": len(result.schedule.active_indices),
            "active_indices_sha256": result.schedule.active_indices_sha256,
            "base_inactive_nonzero_count": 0,
            "inactive_control_coordinates": inactive_control_coordinates,
            "scorer_derived_signed_singletons": 16,
            "scorer_derived_grouped": 8,
            "matched_integer_random_controls": 24,
            "max_candidate_evaluations": MAX_CANDIDATE_EVALUATIONS,
        },
        "shared_base_evaluations": result.shared_base_evaluations,
        "candidate_evaluations": result.candidate_evaluations,
        "base_action_test_oracle": {
            "archive_sha256": result.base_archive_sha256,
            "archive_bytes": result.base_action.archive_bytes,
            "joint_action": result.base_action.action,
        },
        "best_strict_improvement": {
            "identity": best_trace.identity,
            "archive_sha256": best_trace.archive_sha256,
            "archive_bytes": best_trace.action.archive_bytes,
            "delta_joint_action_test_oracle": best_trace.delta_vs_base,
        },
        "governed_handoff_identity": result.governed_handoff_identity,
        "governed_handoff_eligible": result.governed_handoff_eligible,
        "outer_governor_required": True,
        "promotion_allowed": result.promotion_allowed,
        "score_claim": False,
        "pointer_moved": False,
        "resume_ledger": {
            "required_for_production": True,
            "used_by_stale_rehearsal": False,
        },
        "production_contract_exercised": [
            "explicit active coordinate set with zero-valued inactive seed coordinates",
            "precommit all scorer and active-set matched-random proposals before evaluation",
            "compile exact candidate bytes and bind action SHA-256 plus byte count",
            "exact parse-back of realized description coordinates",
            "exact receiver-consumption certification",
            "fresh realized joint-action callback",
            "strict candidate action less than shared base action",
            "synthetic result remains nonpromotable and requires an outer governor",
        ],
        "stale_substrate": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "fd2_receipt": str(fd2_receipt),
            "fd2_receipt_sha256": _file_sha256(fd2_receipt),
            "typed_config_hash": typed_config_hash,
            "description_coordinates_total": expected_total,
            "active_description_coordinates": len(active_indices),
            "proposal_signal": "checkpoint first_moment from stale scorer-trained FD2 state",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fd2-receipt", type=Path, default=DEFAULT_FD2_RECEIPT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--max-candidate-evals",
        type=int,
        choices=(MAX_CANDIDATE_EVALUATIONS,),
        default=MAX_CANDIDATE_EVALUATIONS,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            rehearse(args.checkpoint, args.fd2_receipt, seed=args.seed),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
