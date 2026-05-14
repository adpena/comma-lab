# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments.build_pr85_qh1_model_candidate import build_candidates, build_qh1_payload
from tac.pr85_bundle import expand_pr85_bundle_to_runtime_members
from tac.qh0_renderer_codec import reconstruct_qh1_payload


REPO = Path(__file__).resolve().parents[3]


def test_build_qh1_payload_reconstructs_source_with_selected_record() -> None:
    source = b"QH0" + b"A" * 32 + b"B" * 80 + b"C" * 16
    records = [
        {
            "name": "synthetic.weight",
            "offset": 35,
            "bytes": 80,
            "best_probe_delta_vs_record_bytes": -8,
        }
    ]

    qh1, meta = build_qh1_payload(
        source,
        records,
        max_records=1,
        min_record_saving=1,
        patch_codecs=("zlib",),
        base_codecs=("zlib",),
    )

    assert qh1.startswith(b"QH1")
    assert reconstruct_qh1_payload(qh1) == source
    assert meta["selected_record_count"] == 1
    assert meta["reconstructs_exact_qh0"] is True


def test_build_candidates_records_byte_screen_without_dispatch(tmp_path: Path) -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    summary = build_candidates(
        archive,
        tmp_path,
        max_records_values=(1,),
        min_record_saving=16,
    )

    best = summary["best_candidate"]
    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert best["runtime_compatibility"]["dispatchable_now"] is False
    assert best["qh1"]["reconstructs_exact_qh0"] is True
    with zipfile.ZipFile(REPO / best["archive_path"], "r") as zf:
        raw = zf.read("x")
    rebuilt = expand_pr85_bundle_to_runtime_members(raw).members["renderer.bin"]
    with zipfile.ZipFile(archive, "r") as zf:
        source_x = zf.read("x")
    source = expand_pr85_bundle_to_runtime_members(source_x).members["renderer.bin"]
    assert reconstruct_qh1_payload(rebuilt) == source
