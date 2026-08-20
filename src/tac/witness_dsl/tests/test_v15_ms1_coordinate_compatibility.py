# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.contest_score import compute_contest_score
from tac.witness_dsl.v15_ms1_coordinate_compatibility import (
    AuditPaths,
    SelectedPreimagePair,
    audit_coordinate_compatibility,
    byte_ceiling_for_score,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_byte_ceiling_is_exact_at_current_ms1_operating_point() -> None:
    d_seg = 0.0001519690619574653
    d_pose = 0.00010184327939026322
    ceiling = byte_ceiling_for_score(0.172, d_seg, d_pose)
    assert ceiling == 187_562
    assert compute_contest_score(d_seg, d_pose, ceiling) <= 0.172
    assert compute_contest_score(d_seg, d_pose, ceiling + 1) > 0.172
    assert byte_ceiling_for_score(0.15, d_seg, d_pose) == 154_522


def test_selected_preimage_pair_requires_independent_exact_planes() -> None:
    frame0 = np.zeros((384, 512, 3), dtype=np.uint8)
    frame1 = np.ones((384, 512, 3), dtype=np.uint8)
    pair = SelectedPreimagePair(pair_index=599, frame0=frame0, frame1=frame1)
    assert pair.frame0.shape == (384, 512, 3)
    assert not pair.frame0.flags.writeable
    assert not np.shares_memory(pair.frame0, pair.frame1)
    same_source = SelectedPreimagePair(pair_index=0, frame0=frame0, frame1=frame0)
    assert not np.shares_memory(same_source.frame0, same_source.frame1)


def test_canonical_receipt_audit_falsifies_native_cast_without_heavy_render() -> None:
    evidence = Path(
        "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/prepare_receipt.json"
    )
    if not evidence.is_file():
        # `pytest.skip`, never a bare `return`: a bare return reports PASS on a host without the
        # evidence tier mounted, so the suite goes green while executing nothing (ddm_sd1, found
        # by review 2026-08-20). A skip is visible in the summary line; a return is not.
        pytest.skip(f"C1 evidence tier not mounted: {evidence}")
    receipt = audit_coordinate_compatibility(AuditPaths.canonical(REPO_ROOT), strict_v15_parse=False)

    # The frontier is read from the canonical pointer, never asserted as a literal. CLAUDE.md
    # forbids hardcoded frontier scores precisely because they rot: this line used to assert
    # 0.172 and went red the day the pointer moved to 0.14839100138338618, reading as a code
    # regression when the only thing that changed was a legitimately better score. What IS
    # invariant, and what this now checks, is that the receipt agrees with the pointer.
    pointer = json.loads((REPO_ROOT / ".omx/state/canonical_frontier_pointer.json").read_text())
    assert receipt["frontier"]["effective_score_to_beat"] == pointer["effective_frontier"]["score"]

    assert receipt["artifacts"]["ms1"]["state_identity"].startswith("UNCHANGED_C1_CAMERA_RAW")
    assert receipt["compatibility"]["outer_camera_coordinate"]["verdict"] == "EXACT_SHARED"
    assert receipt["compatibility"]["latent_or_field_cast"]["verdict"] == "FALSIFIED"
    assert receipt["existing_exact_seam"]["verdict"] == "IMPLEMENTED_NONPROMOTABLE_BASELINE"

    # Same reasoning one level down: the byte ceiling is a FUNCTION of the live frontier and the
    # receipt's own measured distortion, so derive it rather than pinning the value it happened to
    # take at score 0.172. This still fails if the economics block and the frontier disagree.
    economics = receipt["economics"]
    assert economics["max_archive_bytes_at_effective_frontier"] == byte_ceiling_for_score(
        receipt["frontier"]["effective_score_to_beat"],
        economics["d_seg"],
        economics["d_pose"],
    )
    assert receipt["authority"]["score_claim"] is False
