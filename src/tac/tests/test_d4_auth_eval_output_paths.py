# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

from experiments.train_substrate_d4_wyner_ziv_frame_0 import (
    _build_parser,
    _resolve_auth_eval_json_paths,
    _validate_auth_eval_pair_scope,
)


def test_d4_auth_eval_json_uses_non_temp_gate_path_for_modal_tmp_workspace() -> None:
    output_dir = Path(tempfile.gettempdir()) / "pact" / "d4_results" / "output"
    gate_json, local_copy_json = _resolve_auth_eval_json_paths(
        output_dir,
        durable_root=Path("/durable/d4_auth_eval"),
    )

    assert gate_json == Path("/durable/d4_auth_eval/output/contest_auth_eval.json")
    assert local_copy_json == output_dir / "contest_auth_eval.json"


def test_d4_auth_eval_json_stays_in_output_dir_when_output_is_durable() -> None:
    durable_output = Path.cwd() / "experiments" / "results" / "d4_test_output"
    gate_json, local_copy_json = _resolve_auth_eval_json_paths(
        durable_output,
        durable_root=Path("/durable/d4_auth_eval"),
    )

    assert gate_json == durable_output / "contest_auth_eval.json"
    assert local_copy_json == durable_output / "contest_auth_eval.json"


def test_d4_capped_pair_smoke_must_skip_auth_eval() -> None:
    args = _build_parser().parse_args(
        [
            "--video-path",
            "upstream/videos/0.mkv",
            "--output-dir",
            "experiments/results/d4_unit",
            "--max-pairs",
            "200",
        ],
    )

    try:
        _validate_auth_eval_pair_scope(args)
    except SystemExit as exc:
        assert "--skip-auth-eval" in str(exc)
        assert "truncated raw outputs" in str(exc)
    else:
        raise AssertionError("capped D4 auth eval should fail closed")


def test_d4_full_pair_or_skip_auth_eval_pair_scope_is_allowed() -> None:
    parser = _build_parser()
    full_args = parser.parse_args(
        [
            "--video-path",
            "upstream/videos/0.mkv",
            "--output-dir",
            "experiments/results/d4_unit",
            "--max-pairs",
            "600",
        ],
    )
    skip_args = parser.parse_args(
        [
            "--video-path",
            "upstream/videos/0.mkv",
            "--output-dir",
            "experiments/results/d4_unit",
            "--max-pairs",
            "200",
            "--skip-auth-eval",
        ],
    )

    _validate_auth_eval_pair_scope(full_args)
    _validate_auth_eval_pair_scope(skip_args)


def test_d4_pair_capped_training_writes_false_authority_manifest() -> None:
    trainer = Path("experiments/train_substrate_d4_wyner_ziv_frame_0.py").read_text(
        encoding="utf-8",
    )

    assert '"schema": "d4_wzf0_training_artifact_manifest_v1"' in trainer
    assert '"training_mode": "smoke" if pair_capped_smoke else "full"' in trainer
    assert '"research_only": pair_capped_smoke' in trainer
    assert '"score_claim": False' in trainer
    assert '"score_claim_valid": False' in trainer
    assert '"promotion_eligible": False' in trainer
    assert '"ready_for_exact_eval_dispatch": False' in trainer
    assert '"pair_capped_smoke_emits_truncated_raw_stream"' in trainer
    assert 'args.output_dir / "manifest.json"' in trainer
