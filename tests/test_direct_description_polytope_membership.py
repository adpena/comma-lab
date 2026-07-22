from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    STREAM_ORDER,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    compile_chart_archive,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError, _sha256
from tac.optimization.direct_description_polytope_membership import (
    DirectDescriptionPolytopeMembershipCheckpointV1,
    DirectDescriptionPolytopeMembershipConfigV1,
    membership_strata_counts,
    stream_decode_digest,
)


def _synthetic_z(n_pairs: int = 64) -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(pair_id, plane_id, 96, 112, 128))
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 3, -2, 1, -4, 2, -1))
            for stratum_index, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum_index * 64, (stratum_index + 1) * 64):
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, 0, 0, 0))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, 0, 1, 2, 3, 4, 5))
    return DirectDescriptionChartZV1(
        n_pairs=n_pairs,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def test_membership_strata_decomposes_slack_and_escapes() -> None:
    cells = np.asarray([[[0, 0], [1, 1]]], dtype=np.uint8)
    margins = np.asarray([[[0.05, 0.25], [0.75, 2.0]]], dtype=np.float32)
    exact = np.asarray([[[True, False], [False, False]]])
    member = np.asarray([[[True, True], [False, True]]])
    rows = membership_strata_counts(exact, member, cells, margins)
    overall = rows["overall"]["all"]
    assert overall == {
        "sites": 4,
        "rgb_pixels_exact": 1,
        "same_c1_argmax_cell": 3,
        "slack_rescued_inexact_sites": 2,
        "argmax_cell_escapes": 1,
    }
    assert rows["target_margin"]["[0,0.1)"]["sites"] == 1
    assert rows["target_margin"]["[1,inf)"]["same_c1_argmax_cell"] == 1


def test_stream_decode_digest_covers_every_pair_and_repeats() -> None:
    receiver = receive_chart_archive(compile_chart_archive(_synthetic_z()).archive)
    first = stream_decode_digest(receiver, n_pairs=64)
    second = stream_decode_digest(receiver, n_pairs=64)
    assert first == second
    assert first["pairs_decoded"] == 64
    assert first["chunks_decoded"] == 6
    assert first["max_described_chunks_resident"] == 1


def test_checkpoint_is_canonical_tamper_evident_and_archive_bound() -> None:
    archive = compile_chart_archive(_synthetic_z()).archive
    config = DirectDescriptionPolytopeMembershipConfigV1(
        scorer_threads=4,
        target_receipt_path="target.json",
        target_receipt_sha256="1" * 64,
        upstream_root="/abs/upstream",
    )
    argv = ("python3", "tools/run_direct_description_polytope_membership.py")
    history = ({"stage_index": 0, "stage_name": "membership_n64", "archive_sha256": _sha256(archive)},)
    checkpoint = DirectDescriptionPolytopeMembershipCheckpointV1(
        config=config.model_dump(mode="json", by_alias=True),
        config_sha256=config.typed_config_hash(),
        dsl_compile_hash=config.dsl_compile_hash(),
        semantic_argv=argv,
        semantic_argv_sha256=_sha256("\0".join(argv).encode()),
        completed_stage_index=0,
        completed_stage_name="membership_n64",
        next_stage_index=1,
        described_pairs=64,
        current_archive_b64=base64.b64encode(archive).decode(),
        current_archive_sha256=_sha256(archive),
        current_archive_bytes=len(archive),
        stage_history=history,
    )
    payload = checkpoint.to_bytes()
    assert DirectDescriptionPolytopeMembershipCheckpointV1.from_bytes(payload) == checkpoint
    envelope = json.loads(payload)
    envelope["body"]["described_pairs"] = 65
    with pytest.raises((DirectDescriptionError, ValueError)):
        DirectDescriptionPolytopeMembershipCheckpointV1.from_bytes(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        )


def test_config_refuses_relative_upstream_root() -> None:
    with pytest.raises(ValueError):
        DirectDescriptionPolytopeMembershipConfigV1(
            scorer_threads=4,
            target_receipt_path="target.json",
            target_receipt_sha256="1" * 64,
            upstream_root="relative/upstream",
        )
