from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "offline_exact_eval_bandit_replay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("offline_exact_eval_bandit_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contest_payload(score: float, *, device: str = "cuda", samples: int = 600) -> dict:
    return {
        "schema_version": 1,
        "final_score": score,
        "score_recomputed_from_components": score,
        "avg_posenet_dist": 0.01,
        "avg_segnet_dist": 0.002,
        "rate_unscaled": 0.01,
        "archive_size_bytes": 12345,
        "n_samples": samples,
        "provenance": {
            "device": device,
            "tool": "experiments/contest_auth_eval.py",
            "archive_sha256": "a" * 64,
            "gpu_t4_match": True,
        },
    }


def _write_jsonl(path: Path, cards: list[dict]) -> None:
    path.write_text("".join(json.dumps(card, sort_keys=True) + "\n" for card in cards))


def test_exact_cuda_only_becomes_reward_and_proxy_records_stay_no_claim(tmp_path: Path) -> None:
    module = _load_module()
    cards = [
        {
            "card_id": "exact_a",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "a"},
            "proxy_score": 1.0,
            "contest_auth_eval_json": _contest_payload(1.0),
        },
        {
            "card_id": "exact_b",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "b"},
            "proxy_score": 2.0,
            "contest_auth_eval_json": _contest_payload(2.0),
        },
        {
            "card_id": "cpu_low_proxy",
            "family": "beta",
            "candidate_config": {"lane": "beta", "variant": "cpu"},
            "proxy_score": 0.05,
            "contest_auth_eval_json": _contest_payload(0.01, device="cpu"),
        },
        {
            "card_id": "candidate_mid",
            "family": "beta",
            "candidate_config": {"lane": "beta", "variant": "mid"},
            "proxy_score": 0.8,
        },
    ]
    path = tmp_path / "cards.jsonl"
    _write_jsonl(path, cards)

    report = module.build_report([path], seed=17)

    assert report["format"] == "offline_exact_eval_bandit_replay_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["exact_reward_count"] == 2
    assert {row["card_ids"][0] for row in report["historical_rewards"]} == {"exact_a", "exact_b"}
    assert any("non_cuda_device" in reason for reason in report["non_reward_reasons"])

    ranked_by_id = {row["card_ids"][0]: row for row in report["ranking"]}
    cpu_row = ranked_by_id["cpu_low_proxy"]
    assert cpu_row["reward_source"] == "none"
    assert cpu_row["score_claim"] is False
    assert cpu_row["promotion_eligible"] is False
    assert cpu_row["result_tag"] == "[proxy-only advisory]"
    assert cpu_row["exact_cuda_required_before_score_claim"] is True
    assert "proxy_score" in cpu_row["advisory_features_used"]


def test_ranking_is_deterministic_across_input_order(tmp_path: Path) -> None:
    module = _load_module()
    exact_cards = [
        {
            "card_id": "exact_a",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "a"},
            "proxy_score": 1.0,
            "contest_auth_eval_json": _contest_payload(1.0),
        },
        {
            "card_id": "exact_b",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "b"},
            "proxy_score": 2.0,
            "contest_auth_eval_json": _contest_payload(2.0),
        },
    ]
    candidates = [
        {
            "card_id": "candidate_low",
            "family": "gamma",
            "candidate_config": {"lane": "gamma", "variant": "low"},
            "proxy_score": 0.7,
        },
        {
            "card_id": "candidate_high",
            "family": "gamma",
            "candidate_config": {"lane": "gamma", "variant": "high"},
            "proxy_score": 1.6,
        },
    ]
    path_a = tmp_path / "cards_a.jsonl"
    path_b = tmp_path / "cards_b.jsonl"
    _write_jsonl(path_a, exact_cards + candidates)
    _write_jsonl(path_b, list(reversed(exact_cards + candidates)))

    report_a = module.build_report([path_a], seed=5)
    report_b = module.build_report([path_b], seed=5)

    assert [row["candidate_config_sha256"] for row in report_a["ranking"]] == [
        row["candidate_config_sha256"] for row in report_b["ranking"]
    ]
    assert [row["card_ids"] for row in report_a["ranking"]] == [row["card_ids"] for row in report_b["ranking"]]
    assert report_a["ranking"][0]["card_ids"] == ["candidate_low"]


def test_candidate_config_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "bad.jsonl"
    _write_jsonl(
        path,
        [
            {
                "card_id": "bad_sha",
                "candidate_config": {"lane": "alpha"},
                "candidate_config_sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(module.ReplayBuildError, match="candidate config SHA mismatch"):
        module.build_report([path], seed=1)


def test_adjudication_json_is_accepted_as_exact_cuda_reward(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "cards.json"
    path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "card_id": "adjudicated_forensic",
                        "family": "omega",
                        "candidate_config": {"lane": "omega", "variant": "r1"},
                        "adjudication_json": {
                            "score_recomputed": 0.9,
                            "score_reported_rounded": 0.9,
                            "evidence_grade": "A-negative scoped forensic",
                            "archive_sha256": "b" * 64,
                            "archive_bytes": 456,
                            "promotion_eligible": False,
                            "allowed_use": ["forensic", "no_promotion"],
                        },
                    }
                ]
            }
        )
    )

    report = module.build_report([path], seed=3)

    assert report["exact_reward_count"] == 1
    assert report["historical_rewards"][0]["reward_source"] == "adjudication_json"
    assert report["historical_rewards"][0]["best_observed_exact_cuda_score"] == 0.9
    assert report["historical_rewards"][0]["score_claim"] is False
    assert report["ranking"] == []


def test_cli_writes_deterministic_report(tmp_path: Path) -> None:
    cards = [
        {
            "card_id": "exact_a",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "a"},
            "proxy_score": 1.0,
            "contest_auth_eval_json": _contest_payload(1.0),
        },
        {
            "card_id": "candidate",
            "family": "alpha",
            "candidate_config": {"lane": "alpha", "variant": "candidate"},
            "proxy_score": 0.9,
        },
    ]
    path = tmp_path / "cards.jsonl"
    output = tmp_path / "report.json"
    _write_jsonl(path, cards)

    first = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(path), "--seed", "11", "--output", str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    first_payload = json.loads(first.stdout)
    first_file = output.read_text()

    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(path), "--seed", "11", "--output", str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert output.read_text() == first_file
    assert json.loads(second.stdout) == first_payload
    assert first_payload["ranking"][0]["score_claim"] is False
