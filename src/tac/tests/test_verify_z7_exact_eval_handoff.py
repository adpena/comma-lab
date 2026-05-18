# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import stat
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import verify_z7_exact_eval_handoff as handoff  # noqa: E402


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _write_zip(path: Path, payload: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", payload)
    return path.stat().st_size, _sha256(path)


def _fixture_repo(tmp_path: Path, *, num_pairs: int = 600) -> Path:
    recurrent_bytes, recurrent_sha = _write_zip(
        tmp_path / "runs/z7/archive.zip",
        b"recurrent-z7-payload",
    )
    static_bytes, static_sha = _write_zip(
        tmp_path / "runs/z7/static_capacity_control/archive.zip",
        b"staticctrl-z7payload",
    )
    runtime_dir = tmp_path / "runs/z7/submission_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR)
    (runtime_dir / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    stats = {
        "archive_zip_bytes": recurrent_bytes,
        "archive_zip_path": "runs/z7/archive.zip",
        "archive_zip_sha256": recurrent_sha,
        "config": {"num_pairs": num_pairs},
        "lane_id": handoff.LANE_ID,
        "loss_mode": "score_aware",
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
        "score_aware_scorer_loss_used": True,
        "score_claim": False,
        "static_capacity_control": {
            "archive_zip_bytes": static_bytes,
            "archive_zip_path": "runs/z7/static_capacity_control/archive.zip",
            "archive_zip_sha256": static_sha,
            "promotion_eligible": False,
            "ready_for_paid_dispatch": False,
            "runtime_output_byte_differences_vs_recurrent": 7,
            "runtime_output_changed_vs_recurrent": True,
            "same_archive_zip_bytes_as_recurrent": recurrent_bytes == static_bytes,
            "score_claim": False,
        },
        "submission_runtime_dir": "runs/z7/submission_runtime",
        "substrate_id": handoff.SUBSTRATE_ID,
    }
    (tmp_path / "runs/z7/stats.json").write_text(
        json.dumps(stats),
        encoding="utf-8",
    )
    return tmp_path


def test_z7_handoff_blocks_current_one_pair_packet_but_keeps_plan_commands() -> None:
    payload = handoff.build_packet(repo_root=REPO_ROOT)

    assert payload["current_pair_count"] == 1
    assert payload["required_pair_count"] == 600
    assert payload["ready_for_exact_eval_handoff"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["provider_dispatch_attempted"] is False
    assert payload["lane_claim_opened"] is False
    assert payload["result_review_blockers"] == [
        "z7_exact_handoff_current_packet_not_600_pairs"
    ]
    assert payload["same_archive_zip_bytes"] is True
    assert payload["runtime_output_changed_vs_recurrent"] is True
    assert set(payload["modal_plan_commands_for_current_packet"]) == {
        "recurrent_paired_contest_cpu_cuda",
        "static_control_paired_contest_cpu_cuda",
    }
    for command in payload["modal_plan_commands_for_current_packet"].values():
        assert "tools/dispatch_modal_paired_auth_eval.py" in command
        assert "--execute" not in command


def test_z7_handoff_ready_for_ratified_full_pair_packet(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is True
    assert payload["result_review_blockers"] == []
    assert payload["current_pair_count"] == 600
    assert payload["runtime_custody"]["inflate_sh_executable"] is True
    assert set(payload["modal_execute_commands_after_ratified_full_packet"]) == {
        "recurrent_paired_contest_cpu_cuda",
        "static_control_paired_contest_cpu_cuda",
    }
    for command in payload["modal_execute_commands_after_ratified_full_packet"].values():
        assert "--execute" in command
        assert "--expected-runtime-tree-sha256 auto" in command
        assert "--skip-axis-if-promotable-anchor-exists" in command


def test_z7_handoff_refuses_false_authority_stats_and_hides_plan_commands(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["score_claim"] = True
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert "z7_exact_handoff_stats_score_claim_not_false" in payload[
        "result_review_blockers"
    ]
    assert payload["modal_plan_commands_for_current_packet"] == {}
    assert payload["modal_execute_commands_after_ratified_full_packet"] == {}
