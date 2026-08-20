#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the bounded synthetic G+A receiver-composition receipt.

This tool is deliberately structural.  It rebuilds its exact P/G/A fixture,
executes the receiver twice, runs a matched A-only counterfactual, and emits a
canonical hash-bound receipt.  It never invokes a scorer or claims that the
synthetic archive is a contest candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.coupled_preimage_program import (  # noqa: E402
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
    compile_coupled_preimage_program,
)
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.generative_taskspace_correction import (  # noqa: E402
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
    compile_generative_taskspace_correction,
)
from tac.witness_dsl.taskspace_pair_fragment_receiver import (  # noqa: E402
    build_a_only_counterfactual_fragment_archive,
    build_taskspace_pair_fragment_archive,
    parse_taskspace_pair_fragment_receipt,
    receive_taskspace_pair_fragment,
)

SCHEMA = "tac.g2_a2_composition_receipt.envelope.v1"
BODY_SCHEMA = "tac.g2_a2_composition_receipt.v1"
DEFAULT_OUTPUT = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/g2_a2_composition_receipt_v1.json"
)
IMPLEMENTATION_PATHS = (
    "src/tac/witness_dsl/generative_taskspace_receiver.py",
    "src/tac/witness_dsl/coupled_preimage_pair_adapter.py",
    "src/tac/witness_dsl/taskspace_pair_fragment_receiver.py",
    "src/tac/witness_dsl/pair_population_envelope.py",
    "tools/materialize_g2_a2_composition_receipt.py",
)


class CompositionReceiptError(RuntimeError):
    """Fixture, replay, pointer, or durable-write proof failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CompositionReceiptError("receipt is not finite canonical ASCII JSON") from exc


def _state() -> PredictorSemanticStateV1:
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    labels[:, 72:144] = 1
    labels[:, 144:230] = 2
    labels[:, 230:315] = 3
    labels[:, 315:] = 4
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(20,),
        labels=labels,
        pose6_codes=np.arange(6, dtype=np.int16).reshape(1, 6),
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (12, 24, 36),
            (48, 60, 72),
            (84, 96, 108),
            (120, 132, 144),
            (156, 168, 180),
        )
    )


def _g_packet(state: PredictorSemanticStateV1) -> bytes:
    program = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(20, "Lane", "birth", "box", 1, 35, 30, 48, 49),),
        realization_profile=_profile(),
    )
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8(1)
    teacher = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=_sha256(memoryview(target).cast("B")),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )
    return compile_generative_taskspace_correction(state, program, teacher_evidence=teacher).packet


def _a_packet(state: PredictorSemanticStateV1, g_packet: bytes, *, variant: int) -> bytes:
    decoded_g = apply_generative_taskspace_correction(g_packet, predictor_state=state)
    controls = (
        Frame1AnchoredY0FibreControlV1(
            source_pair_id=20,
            shift_y_i16=1,
            shift_x_i16=-2,
            rgb_delta_i16=(3 + variant, -4, 5),
        ),
    )
    return compile_coupled_preimage_program(
        state,
        decoded_g,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    ).packet


def _file_custody(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        path = repo_root / relative
        payload = path.read_bytes()
        rows.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    return rows


def build_receipt(
    *,
    repo_root: Path,
    frontier: DynamicFrontierTargetSnapshot,
) -> bytes:
    """Rebuild the complete bounded proof and return canonical envelope bytes."""

    if type(frontier) is not DynamicFrontierTargetSnapshot:
        raise CompositionReceiptError("frontier must be an exact live snapshot")
    verify_dynamic_frontier_target_snapshot(frontier)
    state = _state()
    g_packet = _g_packet(state)
    baseline_a = _a_packet(state, g_packet, variant=0)
    counterfactual_a = _a_packet(state, g_packet, variant=7)
    baseline_archive = build_taskspace_pair_fragment_archive(
        g_packet,
        baseline_a,
        predictor_state=state,
    )
    baseline = receive_taskspace_pair_fragment(baseline_archive, predictor_state=state)
    replay = receive_taskspace_pair_fragment(baseline_archive, predictor_state=state)
    counterfactual_archive = build_a_only_counterfactual_fragment_archive(
        baseline_archive,
        counterfactual_a,
        predictor_state=state,
    )
    counterfactual = receive_taskspace_pair_fragment(counterfactual_archive, predictor_state=state)
    parsed_receipt = parse_taskspace_pair_fragment_receipt(baseline.receipt.to_receipt_bytes())
    if baseline.receipt != replay.receipt or baseline.receipt != parsed_receipt:
        raise CompositionReceiptError("baseline receipt failed deterministic replay/parse-back")
    causal = {
        "exact_g_bytes_fixed": baseline.receipt.sections[0] == counterfactual.receipt.sections[0],
        "exact_a_bytes_changed": baseline.receipt.sections[1] != counterfactual.receipt.sections[1],
        "scorer_y1_fixed": baseline.receipt.scorer_y1_sha256 == counterfactual.receipt.scorer_y1_sha256,
        "camera_y1_fixed": baseline.receipt.camera_frame_sha256_chronological[1::2]
        == counterfactual.receipt.camera_frame_sha256_chronological[1::2],
        "scorer_y0_changed": baseline.receipt.scorer_y0_sha256 != counterfactual.receipt.scorer_y0_sha256,
        "camera_y0_changed": baseline.receipt.camera_frame_sha256_chronological[0::2]
        != counterfactual.receipt.camera_frame_sha256_chronological[0::2],
    }
    if not all(causal.values()):
        raise CompositionReceiptError("matched A-only causal proof is incomplete")
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompositionReceiptError("git HEAD cannot be bound") from exc
    body = {
        "schema": BODY_SCHEMA,
        "git_head_before_root_serialization": git_head,
        "dynamic_frontier_snapshot": asdict(frontier),
        "predictor_fixture": {
            "class": "deterministic_synthetic_external_PredictorSemanticStateV1",
            "binding_sha256": state.binding_sha256,
            "source_pair_ids": list(state.source_pair_ids),
            "counted_predictor_section_present": False,
        },
        "exact_packets": {
            "g_bytes": len(g_packet),
            "g_sha256": _sha256(g_packet),
            "baseline_a_bytes": len(baseline_a),
            "baseline_a_sha256": _sha256(baseline_a),
            "counterfactual_a_bytes": len(counterfactual_a),
            "counterfactual_a_sha256": _sha256(counterfactual_a),
        },
        "baseline": baseline.receipt.as_dict(),
        "a_only_counterfactual": counterfactual.receipt.as_dict(),
        "matched_causal_proof": causal,
        "implementation_custody": _file_custody(repo_root),
        "truth": {
            "bounded_pair_composition_complete": True,
            "deterministic_double_decode": True,
            "strict_receipt_parse_back": True,
            "source_custodied_counted_p_present": False,
            "n600_evidence": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_archive_eligible": False,
            "standalone_runtime_closed": False,
            "complete_payload_lineage": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        },
        "next_blocker": (
            "replace synthetic external V1 P with a source-custodied counted V2 predictor, then execute the "
            "same P/G/A archive path on bounded real pairs without inventing Pose6"
        ),
    }
    body_bytes = _canonical_json(body)
    return _canonical_json({"schema": SCHEMA, "body": body, "body_sha256": _sha256(body_bytes)})


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CompositionReceiptError("output exists with different or unsafe bytes")
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise CompositionReceiptError("installed receipt differs from generated bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    before = load_dynamic_frontier_target(repo_root=REPO)
    payload = build_receipt(repo_root=REPO, frontier=before)
    after = load_dynamic_frontier_target(repo_root=REPO)
    if after != before:
        raise CompositionReceiptError("dynamic frontier pointer changed during receipt materialization")
    _write_once_or_equal(args.output, payload)
    if args.stdout:
        sys.stdout.buffer.write(payload + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
