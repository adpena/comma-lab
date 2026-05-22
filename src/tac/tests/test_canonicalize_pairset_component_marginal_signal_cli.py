# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.canonical_equations.registry import load_registry_events_lenient

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _pairset_acquisition() -> dict[str, object]:
    def row(
        candidate_id: str,
        *,
        rank: int,
        dropped_pair_index: int,
        selected_pair_indices: list[int],
    ) -> dict[str, object]:
        return {
            "schema": "decoder_q_pairset_acquisition_candidate.v1",
            **_false_authority(),
            "acquisition_id": candidate_id,
            "acquisition_rank": rank,
            "selector_kind": "drop_one_from_best",
            "selected_pair_count": len(selected_pair_indices),
            "selected_pair_indices": selected_pair_indices,
            "payload_bytes": 40,
            "rate_delta": -0.00000066585895312,
            "acquisition_operation": {
                "op": "drop_one",
                "dropped_pair_index": dropped_pair_index,
                "dropped_pair_rank": rank,
            },
            "predicted_score_mean": 0.195,
            "predicted_score_source": "fixture_non_authoritative",
        }

    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "candidates": [
            row(
                "pairset_drop_one_rank001_pair0101",
                rank=1,
                dropped_pair_index=101,
                selected_pair_indices=[202, 303],
            ),
            row(
                "pairset_drop_one_rank002_pair0202",
                rank=2,
                dropped_pair_index=202,
                selected_pair_indices=[101, 303],
            ),
        ],
    }


def _observation_row(tmp_path: Path) -> dict[str, object]:
    auth_path = tmp_path / "pairset_drop_one_rank001_pair0101_auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "evidence_grade": "contest-CPU",
                "score_claim_valid": True,
                "canonical_score": 0.192,
                "provenance": {
                    "archive_sha256": "a" * 64,
                    "inflated_output_manifest": {
                        "payload": {"aggregate_sha256": "b" * 64}
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        "candidate_id": "pairset_drop_one_rank001_pair0101",
        "sweep_config_id": "contest_cpu_exact_candidate",
        "optimization_pass_id": "exact_cpu_calibration",
        "family": "decoder_q_selective_dqs1",
        "observed_axis": "contest_cpu",
        "evidence_tag": "[contest-CPU]",
        "evidence_grade": "contest-CPU",
        "observed_score_or_delta": 0.192,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "c" * 64,
        "raw_output_or_cache_sha256": "b" * 64,
        "component_deltas": {
            "segnet_delta": 0.0,
            "posenet_delta": 0.0,
            "rate_delta": -0.00000066585895312,
        },
        "source_artifact_path": auth_path.as_posix(),
        "selected_pair_indices": [202, 303],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_helper_writes_canonical_portfolio_and_registers_equation(
    tmp_path: Path,
) -> None:
    pairset_path = tmp_path / "pairset.json"
    observation_path = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "out"
    registry_path = tmp_path / "registry.jsonl"
    _write_json(pairset_path, _pairset_acquisition())
    _write_jsonl(observation_path, [_observation_row(tmp_path)])

    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "canonicalize_pairset_component_marginal_signal.py"
            ),
            "--incumbent-score",
            "0.205",
            "--incumbent-score-by-axis",
            "contest_cpu=0.1921",
            "--pairset-acquisition",
            str(pairset_path),
            "--observation-jsonl",
            str(observation_path),
            "--output-dir",
            str(output_dir),
            "--register-equation",
            "--registry-path",
            str(registry_path),
            "--registry-lock-path",
            str(registry_path.with_suffix(".lock")),
            "--top-actions",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(result.stdout)
    summary = json.loads((output_dir / "action_summary.json").read_text())
    portfolio = json.loads((output_dir / "portfolio.json").read_text())
    assert stdout["schema"] == "pairset_component_marginal_canonicalization_summary.v1"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["equation_registration"]["registered"] is True
    model = summary["pairset_component_marginal_model"]
    refs = model["canonical_signal_refs"]
    assert model["active"] is True
    assert "pairset_component_marginal" in refs["xray_primitives"]
    assert "pairset_component_marginal_score_decomposition_v1" in refs[
        "canonical_equations"
    ]
    assert "tac.master_gradient_consumers.per_pair_difficulty_atlas" in refs[
        "master_gradient_consumers"
    ]
    assert portfolio["score_claim"] is False
    assert (output_dir / "portfolio.md").is_file()
    assert (output_dir / "action_summary.md").is_file()
    events = load_registry_events_lenient(registry_path)
    assert [event["equation_id"] for event in events] == [
        "pairset_component_marginal_score_decomposition_v1"
    ]


def test_helper_fails_closed_when_component_model_inactive(tmp_path: Path) -> None:
    pairset_path = tmp_path / "pairset.json"
    observation_path = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "out"
    _write_json(pairset_path, _pairset_acquisition())
    _write_jsonl(observation_path, [])

    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "canonicalize_pairset_component_marginal_signal.py"
            ),
            "--incumbent-score",
            "0.205",
            "--pairset-acquisition",
            str(pairset_path),
            "--observation-jsonl",
            str(observation_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "pairset component marginal model inactive" in result.stderr
    assert not output_dir.exists()
