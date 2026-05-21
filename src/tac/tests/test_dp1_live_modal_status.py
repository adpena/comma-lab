# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.optimization.dp1_live_modal_status import (
    build_dp1_live_modal_status,
    build_dp1_modal_call_status,
    parse_dp1_live_modal_log,
    render_markdown,
)


def _baseline_log() -> str:
    return """[lane-dpp] 2026-05-21T03:18:11Z stage_0b_dispatch_claim_verified lane=lane_base job=job_base
[archive-only-eval] 2026-05-21T03:18:11Z Stage 1: bootstrap runtime deps
[archive-only-eval] 2026-05-21T03:18:22Z Stage 4: Phase 2 full training (DPP_RUN_FULL=1)
[dpp-smoke] archive pack/parse roundtrip: 11972 bytes; pairs=4; header=28
[dpp-smoke] wrote archive.zip: /modal_results/base/output/archive.zip
does not have a deterministic implementation
"""


def _procedural_log() -> str:
    return """[lane-dpp] 2026-05-21T03:20:37Z stage_0b_dispatch_claim_verified lane=lane_proc job=job_proc
[archive-only-eval] 2026-05-21T03:20:41Z Stage 3: smoke distill + pack + parse
[full] procedural codebook replacement: 12438 B -> 3889 B (saved 8549 B; predicted delta_s=-0.005692)
[dpp-smoke] archive pack/parse roundtrip: 3889 bytes; pairs=4; header=28
[dpp-smoke] wrote manifest.json: /modal_results/proc/output/manifest.json
[archive-only-eval] 2026-05-21T03:20:48Z Stage 4: Phase 2 full training (DPP_RUN_FULL=1)
"""


def test_parse_dp1_live_modal_log_extracts_stage_and_smoke(tmp_path: Path) -> None:
    log = tmp_path / "procedural.log"
    log.write_text(_procedural_log(), encoding="utf-8")

    parsed = parse_dp1_live_modal_log(log, variant="procedural", repo_root=tmp_path)

    assert parsed["variant"] == "procedural"
    assert parsed["dispatch_claim"]["lane_id"] == "lane_proc"
    assert parsed["stage4_full_training_started"] is True
    assert parsed["smoke_roundtrip"]["archive_bytes"] == 3889
    assert parsed["procedural_codebook_replacement"]["saved_bytes"] == 8549
    assert parsed["procedural_codebook_replacement"]["predicted_delta_s"] == -0.005692
    assert parsed["written_artifacts"]["manifest.json"].endswith("manifest.json")


def test_build_dp1_live_modal_status_compares_smoke_bytes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.log"
    procedural = tmp_path / "procedural.log"
    baseline.write_text(_baseline_log(), encoding="utf-8")
    procedural.write_text(_procedural_log(), encoding="utf-8")

    status = build_dp1_live_modal_status(
        baseline_log=baseline,
        procedural_log=procedural,
        repo_root=tmp_path,
    )

    assert status["score_claim"] is False
    assert status["promotion_eligible"] is False
    assert status["status"] == "running"
    assert status["blockers"] == []
    assert status["smoke_delta_bytes_procedural_minus_baseline"] == 3889 - 11972
    assert "Procedural Replacement" in render_markdown(status)


def test_build_dp1_live_modal_status_reports_finished_when_both_return_zero(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.log"
    procedural = tmp_path / "procedural.log"
    baseline.write_text(_baseline_log() + "[archive-only-eval] finished in 12.0s rc=0\n", encoding="utf-8")
    procedural.write_text(_procedural_log() + "[archive-only-eval] finished in 13.0s rc=0\n", encoding="utf-8")

    status = build_dp1_live_modal_status(
        baseline_log=baseline,
        procedural_log=procedural,
        repo_root=tmp_path,
    )

    assert status["status"] == "finished"
    assert status["blockers"] == []


def _metadata(path: Path, *, variant: str, call_id: str) -> Path:
    path.write_text(
        """{
  "call_id": "%s",
  "label": "dp1_%s",
  "lane_id": "lane_%s",
  "live_volume": "comma-train-lane-results",
  "live_volume_prefix": "dp1_%s/",
  "dispatched_at": "2026-05-21T03:16:00",
  "max_seconds": 5400,
  "mounted_code_git_head": "abc123",
  "mounted_code_git_branch": "main",
  "working_tree_dirty": false,
  "working_tree_dirty_paths_count": 0,
  "sentinel_files_local_sha256": {"a.py": "0"}
}
"""
        % (call_id, variant, variant, variant),
        encoding="utf-8",
    )
    return path


def test_build_dp1_modal_call_status_reports_running_on_short_poll_timeout(
    tmp_path: Path,
) -> None:
    baseline = _metadata(tmp_path / "baseline.json", variant="baseline", call_id="fc-base")
    procedural = _metadata(
        tmp_path / "procedural.json",
        variant="procedural",
        call_id="fc-proc",
    )

    class _RunningCall:
        def get(self, timeout: float | None = None) -> dict:
            raise TimeoutError

    status = build_dp1_modal_call_status(
        baseline_metadata=baseline,
        procedural_metadata=procedural,
        repo_root=tmp_path,
        function_call_from_id=lambda _call_id: _RunningCall(),
    )

    assert status["schema"] == "dp1_live_modal_call_status_v1"
    assert status["status"] == "running"
    assert status["ready_for_training_harvest"] is False
    assert status["score_claim"] is False
    assert status["baseline"]["poll"]["status"] == "running_or_pending"
    assert "keep polling" in status["next_action"]


def test_build_dp1_modal_call_status_reports_ready_for_training_harvest(
    tmp_path: Path,
) -> None:
    baseline = _metadata(tmp_path / "baseline.json", variant="baseline", call_id="fc-base")
    procedural = _metadata(
        tmp_path / "procedural.json",
        variant="procedural",
        call_id="fc-proc",
    )

    class _FinishedCall:
        def __init__(self, call_id: str) -> None:
            self.call_id = call_id

        def get(self, timeout: float | None = None) -> dict:
            return {
                "returncode": 0,
                "elapsed_seconds": 12.5,
                "timed_out": False,
                "artifacts": {f"{self.call_id}/archive.zip": b"zip"},
                "stdout_tail": "finished",
            }

    status = build_dp1_modal_call_status(
        baseline_metadata=baseline,
        procedural_metadata=procedural,
        repo_root=tmp_path,
        function_call_from_id=_FinishedCall,
    )

    assert status["status"] == "ready_for_training_harvest"
    assert status["blockers"] == []
    assert status["ready_for_training_harvest"] is True
    assert status["ready_for_exact_eval_dispatch"] is False
    assert status["baseline"]["poll"]["artifact_count"] == 1
    assert "run tools/harvest_modal_calls.py" in render_markdown(status)
