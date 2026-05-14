# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer_grayscale.py"
SPEC = importlib.util.spec_from_file_location("inflate_renderer_grayscale", MODULE_PATH)
assert SPEC is not None
inflate_grayscale = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inflate_grayscale)


def _sha(classes: torch.Tensor) -> str:
    return hashlib.sha256(classes.to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()


def _payload(*, candidate: torch.Tensor, expected: torch.Tensor, partial: bool = False) -> bytes:
    runs = [
        (0, 0, 1, 2, 2),
        (0, 1, 0, 1, 4),
    ]
    header = {
        "schema": "alpha4_residual_repair_amr1_v1",
        "magic": "AMR1",
        "shape": [int(value) for value in candidate.shape],
        "source_mask_u8_sha256": _sha(expected),
        "candidate_mask_u8_sha256": _sha(candidate),
        "record_struct": ">IHHHB",
        "record_count": len(runs),
        "selection": {"partial_repair": partial},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out.extend(b"AMR1")
    out.extend(struct.pack(">I", len(header_bytes)))
    out.extend(header_bytes)
    for run in runs:
        out.extend(struct.pack(">IHHHB", *run))
    return bytes(out)


def test_applies_amr1_residual_repair_and_verifies_source_sha() -> None:
    candidate = torch.zeros((1, 2, 4), dtype=torch.int64)
    expected = candidate.clone()
    expected[0, 0, 1:3] = 2
    expected[0, 1, 0] = 4

    repaired = inflate_grayscale._apply_amr1_repair(
        candidate,
        _payload(candidate=candidate, expected=expected),
        source_name="unit.amr1",
    )

    assert torch.equal(repaired, expected)


def test_rejects_candidate_sha_mismatch() -> None:
    candidate = torch.zeros((1, 2, 4), dtype=torch.int64)
    expected = candidate.clone()
    expected[0, 0, 1:3] = 2
    expected[0, 1, 0] = 4
    payload = _payload(candidate=candidate, expected=expected)
    mutated_candidate = candidate.clone()
    mutated_candidate[0, 0, 0] = 1

    with pytest.raises(RuntimeError, match="candidate SHA mismatch"):
        inflate_grayscale._apply_amr1_repair(
            mutated_candidate,
            payload,
            source_name="unit.amr1",
        )


def test_partial_repair_does_not_require_source_sha_match() -> None:
    candidate = torch.zeros((1, 2, 4), dtype=torch.int64)
    expected = candidate.clone()
    expected[0, 0, 1:3] = 2
    expected[0, 1, 0] = 4
    payload = _payload(candidate=candidate, expected=expected, partial=True)

    repaired = inflate_grayscale._apply_amr1_repair(
        candidate,
        payload,
        source_name="unit.amr1",
    )

    assert torch.equal(repaired, expected)
