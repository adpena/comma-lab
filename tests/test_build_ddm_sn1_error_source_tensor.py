from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/build_ddm_sn1_error_source_tensor.py"
SPEC = importlib.util.spec_from_file_location("build_ddm_sn1_error_source_tensor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_lazy_g1_mask_is_deterministic() -> None:
    value = object.__new__(MODULE.LazyG1Mask)
    value._rows = {3: (((1, 1), (3, 1), (3, 3), (1, 3)),)}
    value.pair_count = 4
    value.max_slots = 1
    value.knots = 1
    value.templates = 1
    value.payload_sha256 = "0" * 64
    first = value[3]
    second = value[3]
    assert np.array_equal(first, second)
    assert first.dtype == np.bool_
    assert int(first.sum()) == 9


def test_first_rung_order_is_solve_first() -> None:
    assert MODULE.first_rung_and_move("NEVER_DESCRIBED")[0] == "VOCABULARY"
    assert MODULE.first_rung_and_move("DESCRIBED_BUT_REALIZATION_LOST")[0] == "CHART_OR_PARAMETER"
    assert MODULE.first_rung_and_move("STRUCTURALLY_HARD_IRREDUCIBLE")[0] == "POINT_CORRECTION_LAST"


def test_atomic_gzip_jsonl_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    rows = [{"b": 2, "a": 1}, {"kind": "tail"}]
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"

    first = MODULE.atomic_gzip_jsonl(first_path, rows)
    second = MODULE.atomic_gzip_jsonl(second_path, rows)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["sha256"] == second["sha256"]
    assert first["compression"] == "gzip_level_9_mtime_0"
    assert first["row_count"] == 2
    with gzip.open(first_path, "rt") as handle:
        assert handle.read() == '{"a":1,"b":2}\n{"kind":"tail"}\n'


def test_storage_resume_identity_excludes_volatile_free_bytes() -> None:
    before = {
        "status": "PASS",
        "selected_root": "/Volumes/VertigoDataTier/pact",
        "observed_free_bytes": 100,
    }
    after = {**before, "observed_free_bytes": 20}

    assert MODULE.stable_storage_identity(before) == MODULE.stable_storage_identity(after)
    assert "observed_free_bytes" not in MODULE.stable_storage_identity(before)
    assert MODULE.stable_resume_identity({"storage_preflight": before}) == (
        MODULE.stable_resume_identity({"storage_preflight": after})
    )


def test_survival_wall_merge_uses_counts_not_mean_of_fractions() -> None:
    def row(sites: int, errors: int) -> dict[str, object]:
        summary = {
            "sites": sites,
            "errors": errors,
            "error_fraction": errors / sites,
        }
        return {
            "all_classes": summary,
            "by_target_class": dict.fromkeys(MODULE.CLASS_NAMES, summary),
        }

    merged = MODULE.merge_survival_wall_149([row(10, 1), row(90, 18)])
    assert merged["all_classes"] == {
        "sites": 100,
        "errors": 19,
        "error_fraction": 0.19,
    }
    assert merged["by_target_class"]["Road"] == merged["all_classes"]


def test_pose6_summary_closes_coordinate_accounting() -> None:
    batches = [
        {
            "batch_size": 2,
            "pose6": {
                "coordinate_count": 12,
                "painted_vs_gt_sse": 3.0,
                "gt_cache_replay_max_abs": 0.125,
            },
        },
        {
            "batch_size": 1,
            "pose6": {
                "coordinate_count": 6,
                "painted_vs_gt_sse": 1.5,
                "gt_cache_replay_max_abs": 0.25,
            },
        },
    ]
    summary = MODULE.summarize_pose6_product(batches)
    assert summary["pair_count"] == 3
    assert summary["coordinate_count"] == 18
    assert summary["d_pose_first_six_mse"] == 0.25
    assert summary["gt_cache_replay_max_abs"] == 0.25


def test_stage_checkpoint_externalization_is_lossless_and_resumable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    checkpoints = output / "stage_checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "batch.json").write_text('{"closed":true}\n')
    config = SimpleNamespace(
        scratch_directory=tmp_path / "ssd",
        source_sha256={
            "v19c_final_archive_path": "1" * 64,
            "target_cache_path": "2" * 64,
        },
    )
    implementation = {"bundle_sha256": "3" * 64}
    first = MODULE.externalize_stage_checkpoints(
        config=config,
        config_hash="4" * 64,
        implementation=implementation,
        checkpoint_directory=checkpoints,
        output_directory=output,
    )
    assert checkpoints.is_symlink()
    assert (checkpoints / "batch.json").read_text() == '{"closed":true}\n'
    assert first["bytes"] == len('{"closed":true}\n')
    second = MODULE.externalize_stage_checkpoints(
        config=config,
        config_hash="4" * 64,
        implementation=implementation,
        checkpoint_directory=checkpoints,
        output_directory=output,
    )
    assert second["tree_sha256"] == first["tree_sha256"]


def test_new_stage_checkpoint_tree_starts_on_ssd_symlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoints = output / "stage_checkpoints"
    config = SimpleNamespace(scratch_directory=tmp_path / "ssd")
    implementation = {"bundle_sha256": "3" * 64}
    MODULE.prepare_stage_checkpoint_directory(
        config=config,
        config_hash="4" * 64,
        implementation=implementation,
        checkpoint_directory=checkpoints,
    )
    assert checkpoints.is_symlink()
    assert checkpoints.resolve().parent == (tmp_path / "ssd").resolve()
