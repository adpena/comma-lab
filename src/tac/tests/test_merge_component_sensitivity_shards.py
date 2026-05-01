from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from experiments.merge_component_sensitivity_shards import (
    ComponentSensitivityShardMergeError,
    main,
    merge_component_sensitivity_shards,
)
from experiments.profile_component_sensitivity import (
    COMPONENT_OUTPUTS,
    FINITE_DIFFERENCE_SHARD_SCHEMA,
    _channel_ref_payload,
    _channel_ref_sha256,
)
from tac.sensitivity_map import load_sensitivity_map, save_sensitivity_map


ALL_REFS = [
    ("layer0.weight", 0),
    ("layer0.weight", 1),
    ("layer1.weight", 0),
    ("layer1.weight", 1),
]


def _empty_values() -> dict[str, torch.Tensor]:
    return {
        "layer0.weight": torch.zeros(2, dtype=torch.float32),
        "layer1.weight": torch.zeros(2, dtype=torch.float32),
    }


def _write_shard(
    root: Path,
    *,
    shard_index: int,
    shard_count: int = 2,
    assigned_refs: list[tuple[str, int]] | None = None,
) -> None:
    root.mkdir(parents=True)
    assigned = assigned_refs
    if assigned is None:
        start = len(ALL_REFS) * shard_index // shard_count
        end = len(ALL_REFS) * (shard_index + 1) // shard_count
        assigned = ALL_REFS[start:end]
    shard = {
        "schema": FINITE_DIFFERENCE_SHARD_SCHEMA,
        "is_shard": True,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_channel_count": len(assigned),
        "all_channel_count": len(ALL_REFS),
        "assigned_channel_refs": _channel_ref_payload(assigned),
        "all_channel_refs": _channel_ref_payload(ALL_REFS),
        "all_channel_sha256": _channel_ref_sha256(ALL_REFS),
        "assigned_channel_sha256": _channel_ref_sha256(assigned),
        "partition": "contiguous_sorted_conv_channel_refs_v1",
        "merge_required_for_certification_handoff": True,
    }
    summary = {
        "tool": "experiments/profile_component_sensitivity.py",
        "checkpoint": "renderer.bin",
        "video_mkv": "video.mkv",
        "masks_mkv": "masks.mkv",
        "poses": "poses.bin",
        "upstream_dir": "upstream",
        "pair_weights": None,
        "n_pairs_total": 600,
        "n_pairs_selected": 600,
        "n_pairs_calibration": 480,
        "n_pairs_holdout": 120,
        "split_seed": 123,
        "finite_difference_epsilon": 0.001,
        "device": "cuda",
        "component_response_path": "direct_renderer_tensor_inprocess_scorer",
        "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
        "promotion_requested": True,
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "canonical_scorer_path": False,
        "evidence_grade": "diagnostic_cuda_direct_renderer_finite_difference",
        "finite_difference_shard": shard,
        "finite_difference_merge": None,
        "certification_handoff_eligible": False,
    }
    (root / "component_sensitivity_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for component_index, component in enumerate(COMPONENT_OUTPUTS):
        values = _empty_values()
        holdout_values = _empty_values()
        for key, channel in assigned:
            values[key][channel] = 10.0 * (component_index + 1) + shard_index + channel
            holdout_values[key][channel] = 100.0 * (component_index + 1) + shard_index + channel
        save_sensitivity_map(
            root / f"{component}_sensitivity_map.pt",
            values,
            metadata={**summary, "component": component, "scorer_target": component},
        )
        save_sensitivity_map(
            root / f"{component}_holdout_sensitivity_map.pt",
            holdout_values,
            metadata={
                **summary,
                "component": component,
                "scorer_target": component,
                "split": "holdout",
            },
        )


def test_merge_component_sensitivity_shards_writes_exact_outputs(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0)
    _write_shard(shard1, shard_index=1)

    out = tmp_path / "merged"
    validation = merge_component_sensitivity_shards(
        shard_dirs=[shard1, shard0],
        output_dir=out,
        expected_shard_count=2,
    )

    assert validation["coverage"] == "exactly_once"
    assert validation["promotion_eligible"] is False
    assert validation["score_claim"] is False
    assert validation["certification_handoff_eligible"] is True
    assert (out / "component_sensitivity_shard_merge_validation.json").is_file()
    summary = json.loads((out / "component_sensitivity_profile_summary.json").read_text())
    assert summary["merge_tool"] == "experiments/merge_component_sensitivity_shards.py"
    assert summary["promotion_eligible"] is False
    assert summary["score_claim"] is False
    assert summary["finite_difference_shard"]["is_shard"] is False
    assert summary["finite_difference_shard"]["merge_required_for_certification_handoff"] is False
    assert summary["finite_difference_merge"]["source_shard_indices"] == [0, 1]

    merged, metadata = load_sensitivity_map(out / "posenet_sensitivity_map.pt")
    assert metadata["finite_difference_shard"] == summary["finite_difference_shard"]
    assert torch.equal(merged["layer0.weight"], torch.tensor([10.0, 11.0]))
    assert torch.equal(merged["layer1.weight"], torch.tensor([11.0, 12.0]))

    holdout, holdout_metadata = load_sensitivity_map(out / "posenet_holdout_sensitivity_map.pt")
    assert holdout_metadata["split"] == "holdout"
    assert torch.equal(holdout["layer0.weight"], torch.tensor([100.0, 101.0]))
    assert torch.equal(holdout["layer1.weight"], torch.tensor([101.0, 102.0]))


def test_merge_component_sensitivity_shards_rejects_missing_by_default(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "shard0"
    _write_shard(shard0, shard_index=0)

    with pytest.raises(ComponentSensitivityShardMergeError, match="missing finite-difference"):
        merge_component_sensitivity_shards(
            shard_dirs=[shard0],
            output_dir=tmp_path / "merged",
            expected_shard_count=2,
        )


def test_merge_component_sensitivity_shards_allow_incomplete_marks_non_handoff(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "shard0"
    _write_shard(shard0, shard_index=0)

    out = tmp_path / "merged"
    validation = merge_component_sensitivity_shards(
        shard_dirs=[shard0],
        output_dir=out,
        expected_shard_count=2,
        allow_incomplete=True,
    )

    assert validation["coverage"] == "incomplete"
    assert validation["missing_shard_indices"] == [1]
    assert validation["missing_channel_count"] == 2
    assert validation["certification_handoff_eligible"] is False
    summary = json.loads((out / "component_sensitivity_profile_summary.json").read_text())
    assert summary["certification_handoff_eligible"] is False
    assert summary["finite_difference_shard"]["merge_required_for_certification_handoff"] is True

    merged, _metadata = load_sensitivity_map(out / "combined_sensitivity_map.pt")
    assert torch.equal(merged["layer0.weight"], torch.tensor([30.0, 31.0]))
    assert torch.equal(merged["layer1.weight"], torch.zeros(2))


def test_merge_component_sensitivity_shards_rejects_duplicate_shard_index(
    tmp_path: Path,
) -> None:
    shard0a = tmp_path / "shard0a"
    shard0b = tmp_path / "shard0b"
    _write_shard(shard0a, shard_index=0)
    _write_shard(shard0b, shard_index=0)

    with pytest.raises(ComponentSensitivityShardMergeError, match="duplicate finite-difference shard index"):
        merge_component_sensitivity_shards(
            shard_dirs=[shard0a, shard0b],
            output_dir=tmp_path / "merged",
            expected_shard_count=2,
        )


def test_merge_component_sensitivity_shards_rejects_duplicate_channel(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0, assigned_refs=[ALL_REFS[0], ALL_REFS[1]])
    _write_shard(shard1, shard_index=1, assigned_refs=[ALL_REFS[1], ALL_REFS[2], ALL_REFS[3]])

    with pytest.raises(ComponentSensitivityShardMergeError, match="duplicate finite-difference shard channel"):
        merge_component_sensitivity_shards(
            shard_dirs=[shard0, shard1],
            output_dir=tmp_path / "merged",
            expected_shard_count=2,
        )


def test_merge_component_sensitivity_shards_cli_repeated_shard_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(shard0, shard_index=0)
    _write_shard(shard1, shard_index=1)

    rc = main(
        [
            "--shard-dir",
            str(shard0),
            "--shard-dir",
            str(shard1),
            "--output-dir",
            str(tmp_path / "merged"),
            "--expected-shard-count",
            "2",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"] == "exactly_once"
    assert payload["promotion_eligible"] is False
    assert payload["score_claim"] is False
