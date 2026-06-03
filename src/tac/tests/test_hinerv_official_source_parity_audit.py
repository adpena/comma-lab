# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.hinerv_official_source_parity_audit import (
    FORWARD_PARITY_ARTIFACT_SCHEMA,
    SCHEMA,
    build_hinerv_official_source_parity_audit,
    render_hinerv_official_source_parity_markdown,
    summarize_hinerv_official_source_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_hinerv_official_source_audit_blocks_until_numeric_forward_artifact(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path)

    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["official_source_markers_present"] is True
    assert report["local_receiver_bindings_present"] is True
    assert report["official_forward_parity_proven"] is False
    assert report["blockers"] == ["hinerv_official_forward_parity_missing"]
    assert report["official_forward_parity_artifact_row"]["status"] == "missing"
    assert all(row["status"] == "present" for row in report["official_file_rows"])
    assert all(row["all_markers_present"] for row in report["official_marker_group_rows"])

    summary = summarize_hinerv_official_source_audit(report)
    assert summary["official_source_markers_present"] is True
    assert summary["local_receiver_bindings_present"] is True
    assert summary["official_forward_parity_proven"] is False
    assert "hinerv_official_forward_parity_missing" in summary["blockers"]

    md = render_hinerv_official_source_parity_markdown(report)
    assert "HiNeRV Official Source-Parity Audit" in md
    assert "official forward parity proven: `False`" in md


def test_hinerv_official_source_audit_fails_closed_on_missing_markers(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path)
    (official / "models/layers.py").write_text("class FeatureGrid: pass\n", encoding="utf-8")

    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    assert report["official_source_markers_present"] is False
    assert "hinerv_official_source_marker_missing:official_convnext_decoder" in report["blockers"]
    assert "hinerv_official_forward_parity_missing" in report["blockers"]


def test_hinerv_official_source_audit_rejects_boolean_only_forward_pass(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path)
    artifact = tmp_path / "forged.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": FORWARD_PARITY_ARTIFACT_SCHEMA,
                "official_forward_parity_passed": True,
                "component_rows": [
                    {
                        "component_id": "core_hierarchical_renderer",
                        "source_forward_parity_proven": True,
                    },
                    {
                        "component_id": "patch_dataset_path",
                        "source_forward_parity_proven": True,
                    },
                    {
                        "component_id": "prune_quant_codec",
                        "source_forward_parity_proven": True,
                    },
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        official_forward_parity_artifact_path=artifact,
        generated_utc="20260603T000000Z",
    )

    row = report["official_forward_parity_artifact_row"]
    assert row["status"] == "present"
    assert row["parity_passed"] is False
    assert "hinerv_official_forward_parity_weight_manifest_missing" in row["blockers"]
    assert "hinerv_official_forward_parity_source_replay_missing" in row["blockers"]
    assert any(
        blocker.startswith("numeric_max_abs_error_missing:core_hierarchical_renderer")
        for blocker in row["blockers"]
    )
    assert report["official_forward_parity_proven"] is False
    assert "hinerv_official_forward_parity_missing" in report["blockers"]


def test_hinerv_official_source_audit_accepts_hash_backed_numeric_replay(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path)
    artifact = tmp_path / "numeric.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": FORWARD_PARITY_ARTIFACT_SCHEMA,
                "official_weight_manifest": {
                    "state_dict_sha256": "1" * 64,
                    "state_dict_key_count": 7,
                },
                "source_forward_replay": {
                    "backend": "official_torch_vs_mlx",
                    "input_bundle_sha256": "2" * 64,
                },
                "official_forward_parity_passed": True,
                "official_forward_parity_falsified": False,
                "component_rows": [
                    _numeric_component("core_hierarchical_renderer"),
                    _numeric_component("patch_dataset_path"),
                    _numeric_component("prune_quant_codec"),
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        official_forward_parity_artifact_path=artifact,
        generated_utc="20260603T000000Z",
    )

    assert report["official_forward_parity_artifact_row"]["parity_passed"] is True
    assert report["official_forward_parity_artifact_row"]["blockers"] == []
    assert report["official_forward_parity_proven"] is True
    assert report["blockers"] == []
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def _numeric_component(component_id: str) -> dict[str, object]:
    return {
        "component_id": component_id,
        "source_forward_parity_proven": True,
        "max_abs_error": 0.0,
        "tolerance": 1.0e-6,
        "input_sha256": "3" * 64,
        "official_output_sha256": "4" * 64,
        "portable_output_sha256": "4" * 64,
        "official_weight_sha256": "5" * 64,
    }


def _write_minimal_official_hinerv_repo(tmp_path: Path) -> Path:
    root = tmp_path / "HiNeRV"
    (root / "models").mkdir(parents=True)
    (root / "compression").mkdir(parents=True)
    (root / "cfgs/models").mkdir(parents=True)
    (root / "models/hinerv.py").write_text(
        """
class HiNeRVEncoding:
    def __init__(self):
        self.grids.append(FeatureGrid((1, 2), init_scale=0.01))
        self.grid_expands.append(GridTrilinear3D((1, 2, 3)))

class HiNeRVUpsampler:
    pass

class HiNeRVDecoder:
    def __init__(self):
        self.blocks = []
        self.blocks.append(HiNeRVUpsampler())

class HiNeRV:
    pass

group.add_argument('--enc-grid-level')
group.add_argument('--base-grid-size')
group.add_argument('--base-grid-level')
group.add_argument('--base-grid-level-scale')
group.add_argument('--enc-grid-size')
group.add_argument('--enc-grid-level-scale')
""",
        encoding="utf-8",
    )
    (root / "models/layers.py").write_text(
        """
class FeatureGrid: pass
class GridTrilinear3D: pass
class ConvNeXtBlock: pass
class ConvNeXtBlockLessNorm: pass
class Upsample: pass
""",
        encoding="utf-8",
    )
    (root / "datasets.py").write_text(
        "group.add_argument('--patch-size')\ndef load_all_patches(): pass\ncached = 'patch'\npatch_size = [1, 2, 3]\n",
        encoding="utf-8",
    )
    (root / "cfgs/models/uvg-hinerv-s_1920x1080.txt").write_text(
        "--base-grid-size 150 -1 -1 2 --base-grid-level 2 "
        "--base-grid-level-scale 2. 1. 1. 0.5\n"
        "--enc-grid-size -1 4 --enc-grid-level 3 "
        "--enc-grid-level-scale 2. 0.5\n",
        encoding="utf-8",
    )
    (root / "hinerv_compress.py").write_text(
        """
from compression.prune_utils import set_pruning
from compression.quant_utils import set_quantization, QuantNoise
from compression.prune_utils import PruningMask
group.add_argument('--quant-level')
""",
        encoding="utf-8",
    )
    (root / "compression/quant_utils.py").write_text(
        "class QuantNoise: pass\ndef set_quantization(): pass\n",
        encoding="utf-8",
    )
    (root / "compression/codec_utils.py").write_text(
        "torchac.encode_float_cdf(cdf, inverse)\ntorchac.decode_float_cdf(cdf, stream)\n",
        encoding="utf-8",
    )
    (root / "compression/prune_utils.py").write_text(
        "class PruningMask: pass\ndef set_pruning(): pass\n",
        encoding="utf-8",
    )
    return root
