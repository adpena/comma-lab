from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyze_or_build_pr85_qfq4_model_serializer_candidate import (
    build_or_block_probe,
    qfq4_runtime_compatibility,
)


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]


def test_qfq4_runtime_compatibility_fails_closed_without_pr85_loader(tmp_path: Path) -> None:
    replay = tmp_path / "inflate.py"
    replay.write_text(
        'def load_compact_archive_bundle(data_dir):\n'
        '    path = data_dir / "x"\n'
        "def get_decoded_state_dict_custom(payload_data, device):\n"
        '    if payload_data[:3] in (b"QH0", b"QM0"):\n'
        "        return {}\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate_renderer.py").write_text("# QH0/QM0 only\n", encoding="utf-8")
    (runtime / "unpack_renderer_payload.py").write_text(
        'path = data_dir / "x"\n',
        encoding="utf-8",
    )

    compat = qfq4_runtime_compatibility(
        replay_inflate_py=replay,
        robust_current_dir=runtime,
    )

    assert compat["runtime_can_decode_without_edits"] is False
    assert compat["dispatch_unlocked"] is False
    assert compat["public_pr85_replay_qfq4_model_loader"] is False
    assert "public_pr85_replay_missing_QFQ4_model_loader" in compat["blockers"]


def test_real_pr85_qfq4_serializer_probe_blocks_without_candidate_archive(tmp_path: Path) -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    pr90_probe = REPO / "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json"
    if not archive.is_file() or not pr90_probe.is_file():
        pytest.skip("public PR85/PR90 intake artifacts are not present")

    summary = build_or_block_probe(
        archive=archive,
        out_dir=tmp_path,
        pr90_payload_probe_json=pr90_probe,
        qrow_policies=("shifted_int8_rows",),
    )

    assert summary["built_candidate_count"] == 0
    assert summary["dispatch"] is False
    assert summary["candidate_manifests"] == []
    assert summary["dispatch_unlocked"] is False
    assert summary["exact_eval_readiness"]["ready"] is False
    assert summary["blocker_class"] == "tensor_parity_failed_and_runtime_incompatible"
    assert summary["structured_blocker_json"] == str((tmp_path / "dispatch_blocker.json").resolve())
    assert not list(tmp_path.rglob("archive.zip"))

    blocker = json.loads((tmp_path / "dispatch_blocker.json").read_text(encoding="utf-8"))
    assert blocker["dispatch"] is False
    assert blocker["candidate_archive_emitted"] is False
    assert blocker["best_formula_only_candidate"]["candidate_id"] == "qfq4_pr85_shifted_int8_rows"
    assert blocker["decoded_tensor_parity_gate"]["decoded_tensor_parity"] is False
    assert blocker["runtime_output_parity_gate"]["passed"] is False
    assert "local_renderer_output_parity_on_source_vs_candidate" in blocker["required_before_dispatch"]

    best = summary["best_screened_candidate"]
    assert best["candidate_id"] == "qfq4_pr85_shifted_int8_rows"
    assert best["outer_pr85_model_delta_bytes_vs_source"] < 0
    assert best["decoded_tensor_parity"]["decoded_tensor_parity"] is False
    assert best["decoded_tensor_parity"]["mismatch_count"] == 1
    mismatch = best["decoded_tensor_parity"]["mismatches"][0]
    assert mismatch["name"] == "frame1_head.block1.film_proj.weight"
    assert mismatch["changed_elements"] == 4726
    assert mismatch["max_abs_diff"] == pytest.approx(6.103515625e-05)
    assert "pr85_runtime_missing_qfq4_loader" in best["build_blockers"]
