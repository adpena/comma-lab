# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments.train_substrate_hi_nerv_mlx_local import (
    DIRECT_TRAINER_CANONICALIZATION_SCHEMA,
    DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA,
    TRAINER_SCHEMA,
    _build_parser,
    _build_staged_scorer_curriculum,
    _coder_qat_config_from_args,
    _config_from_args,
    _curriculum_stages_from_args,
    _direct_trainer_canonicalization_contract,
    _full_main,
    _metadata_safe,
    _pose_student_input_channels,
    _prioritized_pair_indices_from_args,
    _prioritized_pair_training_lineage_metadata,
    _prioritized_pair_training_metadata,
    _receiver_cache_quality_manifest_summary,
    _resolve_output_dir,
)
from tac.substrates._shared.mlx_score_aware.adapter import (
    DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
)


def test_hinerv_mlx_trainer_binds_modelsize_row_and_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--modelsize-row",
            "hi_nerv_local_small",
            "--num-pairs",
            "7",
            "--decoder-channels",
            "9,8,7,6,5,4,3",
            "--latent-dim-coarse",
            "11",
            "--output-height",
            "96",
            "--output-width",
            "128",
        ]
    )

    cfg = _config_from_args(args)

    assert cfg.num_pairs == 7
    assert cfg.latent_dim_coarse == 11
    assert cfg.latent_dim_mid == 15
    assert cfg.latent_dim_fine == 18
    assert cfg.embed_dim == 48
    assert cfg.decoder_channels == (9, 8, 7, 6, 5, 4, 3)
    assert cfg.output_height == 96
    assert cfg.output_width == 128


def test_hinerv_mlx_trainer_coder_qat_config_is_real_and_validated() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--coder-qat",
            "--coder-qat-bits",
            "4",
            "--coder-qat-quant-residual-weight",
            "0.25",
            "--coder-qat-magnitude-weight",
            "0.125",
            "--coder-qat-delta-weight",
            "0.0625",
        ]
    )

    cfg = _coder_qat_config_from_args(args)

    assert cfg.enabled is True
    assert cfg.quant_bits == 4
    assert cfg.quant_residual_weight == pytest.approx(0.25)
    assert cfg.magnitude_weight == pytest.approx(0.125)
    assert cfg.delta_weight == pytest.approx(0.0625)


def test_hinerv_mlx_trainer_pose_student_channels_match_preprocess() -> None:
    assert _pose_student_input_channels("rgb") == 3
    assert _pose_student_input_channels("pr95_yuv6") == 6

    with pytest.raises(ValueError, match="pose_student_input_preprocess"):
        _pose_student_input_channels("not_real")


def test_hinerv_mlx_trainer_builds_staged_scorer_curriculum() -> None:
    stages = _build_staged_scorer_curriculum(
        epochs=100,
        recon_fraction=0.75,
        segnet_fraction=0.15,
        final_recon_weight=0.25,
        segnet_lr_scale=0.3,
        final_lr_scale=0.1,
    )

    assert [stage.name for stage in stages] == [
        "hi_nerv_receiver_fit_recon_scaffold",
        "hi_nerv_segnet_last_frame_admission",
        "hi_nerv_joint_scorer_waterfill_finetune",
    ]
    assert [(stage.start_epoch, stage.end_epoch) for stage in stages] == [
        (0, 75),
        (75, 90),
        (90, 100),
    ]
    assert stages[0].loss_weights == {
        "recon": 1.0,
        "distill": 0.0,
        "pose_distill": 0.0,
    }
    assert stages[1].loss_weights == {
        "recon": 1.0,
        "distill": 1.0,
        "pose_distill": 0.0,
    }
    assert stages[2].loss_weights == {
        "recon": 0.25,
        "distill": 1.0,
        "pose_distill": 1.0,
    }
    assert stages[1].lr_scale == pytest.approx(0.3)
    assert stages[2].lr_scale == pytest.approx(0.1)


def test_hinerv_mlx_trainer_staged_curriculum_from_args_and_validation() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "80",
            "--staged-scorer-curriculum",
            "--staged-scorer-recon-fraction",
            "0.5",
            "--staged-scorer-segnet-fraction",
            "0.25",
        ]
    )

    stages = _curriculum_stages_from_args(args)

    assert stages is not None
    assert [(stage.start_epoch, stage.end_epoch) for stage in stages] == [
        (0, 40),
        (40, 60),
        (60, 80),
    ]
    with pytest.raises(ValueError, match="epochs >= 3"):
        _build_staged_scorer_curriculum(
            epochs=2,
            recon_fraction=0.5,
            segnet_fraction=0.25,
            final_recon_weight=0.25,
            segnet_lr_scale=0.3,
            final_lr_scale=0.1,
        )


def test_hinerv_mlx_trainer_rejects_local_output_without_opt_in(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(
        ["--smoke", "--output-dir", str(tmp_path / "local")]
    )

    with pytest.raises(StorageTierError, match="local_disk_tier_disabled"):
        _resolve_output_dir(args)


def test_hinerv_mlx_trainer_allows_explicit_local_smoke_output(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "local"),
            "--allow-local-output-dir",
        ]
    )

    output, storage = _resolve_output_dir(args)

    assert output == (tmp_path / "local").resolve(strict=False)
    assert output.is_dir()
    assert storage["schema"] == "hi_nerv_mlx_trainer_explicit_output_preflight.v1"
    assert storage["score_claim"] is False
    assert storage["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_parser_requires_mode() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])

    assert TRAINER_SCHEMA == "hi_nerv_mlx_score_aware_trainer.v1"


def test_hinerv_direct_trainer_canonicalization_contract_blocks_authority() -> None:
    contract = _direct_trainer_canonicalization_contract(mode="full")

    assert contract["schema"] == DIRECT_TRAINER_CANONICALIZATION_SCHEMA
    assert contract["canonical_runner_entrypoint"] == (
        "tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv"
    )
    assert contract["direct_trainer_role"] == (
        "runner_subprocess_or_research_smoke_only"
    )
    assert contract["planner_row_required"] is True
    assert contract["planner_row_id"] is None
    assert contract["source_parity_contract_consumed"] is False
    assert contract["pr95_prelaunch_gate_consumed"] is False
    assert contract["trainer_launch_allowed"] is False
    assert "hinerv_direct_trainer_missing_planner_row_id" in contract["blockers"]
    assert "hinerv_direct_trainer_local_cpu_replay_gate_not_bound" in contract[
        "blockers"
    ]
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_hinerv_direct_full_refuses_before_score_aware_trainer_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def import_tripwire(name: str, *args: object, **kwargs: object) -> object:
        if name == "tac.substrates._shared.mlx_score_aware":
            raise AssertionError("run_mlx_score_aware_full_main must not be reached")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_tripwire)
    args = _build_parser().parse_args(["--full"])

    assert _full_main(args) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["schema"] == DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA
    assert payload["mode"] == "full"
    assert payload["training_executed"] is False
    assert payload["export_executed"] is False
    assert payload["trainer_launch_allowed"] is False
    assert payload["allowed_direct_research_mode"] == "--smoke"
    assert (
        "hinerv_direct_full_trainer_launch_blocked_by_canonicalization_contract"
        in payload["blockers"]
    )
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_optimizer_choices_match_adapter() -> None:
    default_args = _build_parser().parse_args(["--full"])
    assert default_args.optimizer_kind == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND

    for optimizer_kind in ("rmsprop", "lion", "adafactor", "muon", "pact_muon_adamw"):
        args = _build_parser().parse_args(
            ["--full", "--optimizer-kind", optimizer_kind]
        )
        assert args.optimizer_kind == optimizer_kind

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--full", "--optimizer-kind", "definitely_not_optimizer"]
        )


def test_hinerv_mlx_trainer_parses_prioritized_pair_controls(
    tmp_path: Path,
) -> None:
    pair_file = tmp_path / "sample_generalization_gate.json"
    pair_file.write_text(
        '{"sample_generalization_gate":{"hard_pair_coverage":'
        '{"prioritized_pair_indices":[9,4,9]}}}',
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--full",
            "--prioritized-pair-indices",
            "3,4,3",
            "--prioritized-pair-indices-file",
            str(pair_file),
        ]
    )

    pair_indices = _prioritized_pair_indices_from_args(args)
    metadata = _prioritized_pair_training_metadata(pair_indices)

    assert pair_indices == (3, 4, 9)
    assert metadata["schema"] == "hi_nerv_direct_trainer_prioritized_pair_training.v1"
    assert metadata["enabled"] is True
    assert metadata["pair_indices"] == [3, 4, 9]
    assert (
        metadata["pair_index_domain"]
        == "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
    )
    assert metadata["arbitrary_source_pair_hydration"] is False
    assert metadata["target_hydration_pair_indices_consumed"] is False
    assert metadata["requires_num_pairs_covering_pair_ids"] is True
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_hinerv_prioritized_pair_lineage_metadata_has_no_canonical_authority() -> None:
    metadata = _prioritized_pair_training_lineage_metadata((4, 1))

    assert metadata["enabled"] is True
    assert metadata["pair_indices"] == [4, 1]
    assert (
        metadata["pair_index_domain"]
        == "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
    )
    assert metadata["arbitrary_source_pair_hydration"] is False
    assert metadata["target_hydration_pair_indices_consumed"] is False
    assert metadata["requires_num_pairs_covering_pair_ids"] is True
    assert metadata["canonical_authority_surface"] == (
        "TrainingArtifact top-level false-authority fields"
    )
    for forbidden in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "score_claim_valid",
    ):
        assert forbidden not in metadata


def test_hinerv_mlx_trainer_rejects_out_of_range_prioritized_pairs() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--num-pairs",
            "4",
            "--prioritized-pair-indices",
            "3,4",
        ]
    )

    with pytest.raises(ValueError, match="out-of-range"):
        _prioritized_pair_indices_from_args(args)


def test_hinerv_mlx_trainer_forwards_prioritized_pairs_to_harness() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]

    assert run_calls
    assert any(
        any(keyword.arg == "prioritized_pair_indices" for keyword in call.keywords)
        for call in run_calls
    )


def test_hinerv_mlx_trainer_metadata_safe_drops_nested_authority_keys() -> None:
    payload = {
        "storage": {
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "selected_workload_root": "/Volumes/VertigoDataTier/pact/x",
            "children": [{"rank_or_kill_eligible": False, "keep": "yes"}],
        },
        "keep_top": True,
    }

    safe = _metadata_safe(payload)

    assert "score_claim" not in safe["storage"]
    assert "ready_for_exact_eval_dispatch" not in safe["storage"]
    assert safe["storage"]["selected_workload_root"].endswith("/x")
    assert safe["storage"]["children"] == [{"keep": "yes"}]
    assert safe["keep_top"] is True


def test_hinerv_mlx_trainer_parses_post_export_receiver_cache_quality_gate() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--post-export-receiver-cache-quality-gate",
            "--receiver-cache-quality-max-pairs",
            "4",
            "--receiver-cache-quality-batch-pairs",
            "2",
            "--receiver-cache-quality-min-segnet-dynamic-range",
            "8",
            "--receiver-cache-quality-reference-cache-dir",
            "/Volumes/VertigoDataTier/pact/ref_cache",
        ]
    )

    assert args.post_export_receiver_cache_quality_gate is True
    assert args.receiver_cache_quality_max_pairs == 4
    assert args.receiver_cache_quality_batch_pairs == 2
    assert args.receiver_cache_quality_min_segnet_dynamic_range == pytest.approx(8.0)
    assert args.receiver_cache_quality_reference_cache_dir.as_posix().endswith(
        "/ref_cache"
    )


def test_hinerv_receiver_cache_quality_summary_drops_authority_keys() -> None:
    summary = _receiver_cache_quality_manifest_summary(
        {
            "report_path": "/Volumes/VertigoDataTier/pact/run/report.json",
            "archive_path": "/Volumes/VertigoDataTier/pact/run/archive.zip",
            "archive_sha256": "a" * 64,
            "candidate_cache_dir": "/Volumes/VertigoDataTier/pact/run/cache",
            "quality_gate_path": "/Volumes/VertigoDataTier/pact/run/gate.json",
            "quality_gate_passed": False,
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
            "score_claim": False,
            "quality_gate": {
                "verdict": "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW",
                "distance_to_reference": {"segnet_last_rgb_mae": 3.0},
                "stats": {
                    "candidate_segnet_last_rgb": {
                        "dynamic_range": 4.0,
                        "std": 1.5,
                    }
                },
                "score_claim": False,
            },
        }
    )

    assert summary is not None
    assert summary["schema"] == "hi_nerv_receiver_cache_quality_summary.v1"
    assert summary["quality_gate_passed"] is False
    assert summary["quality_gate_verdict"] == "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW"
    assert "score_claim" not in summary
    assert summary["candidate_segnet_last_rgb_stats"]["dynamic_range"] == pytest.approx(
        4.0
    )
