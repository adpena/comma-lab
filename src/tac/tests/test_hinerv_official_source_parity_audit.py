# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.hinerv_official_source_parity_audit import (
    FORWARD_PARITY_ARTIFACT_SCHEMA,
    SCHEMA,
    build_hinerv_official_forward_parity_artifact,
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
    assert "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF" in report[
        "local_binding_marker_row"
    ]["present_markers"]
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
    assert "| `core_hierarchical_renderer` | `False` | `False` |" in md


def test_hinerv_official_forward_parity_artifact_round_trips_falsification(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path)

    artifact = build_hinerv_official_forward_parity_artifact(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
    assert artifact["official_forward_parity_passed"] is False
    assert artifact["official_forward_parity_falsified"] is True
    assert artifact["source_forward_replay"]["backend"] == (
        "official_torch_cpu_full_hinerv_forward"
    )
    assert artifact["source_forward_replay"]["replay_ran"] is True
    assert len(artifact["source_forward_replay"]["input_bundle_sha256"]) == 64
    assert len(artifact["source_forward_replay"]["official_output_sha256"]) == 64
    assert artifact["source_forward_replay"]["official_output_shape"] == [1, 3, 1, 2, 2]
    assert artifact["source_forward_replay"]["full_hinerv_forward_parity_proven"] is True
    assert artifact["source_forward_replay"]["max_abs_error"] == 0.0
    assert artifact["source_forward_replay"]["blockers"] == []
    assert len(artifact["source_forward_replay"]["portable_output_sha256"]) == 64
    assert artifact["official_weight_manifest"]["state_dict_key_count"] > 0
    assert len(artifact["official_weight_manifest"]["state_dict_sha256"]) == 64
    grid_rows = {
        row["component_id"]: row
        for row in artifact["numeric_subcomponent_rows"]
    }
    assert grid_rows["official_grid_trilinear3d"]["source_forward_parity_proven"] is True
    assert grid_rows["official_grid_trilinear3d"]["full_hinerv_forward_parity_proven"] is False
    assert grid_rows["official_grid_trilinear3d"]["max_abs_error"] <= 1.0e-6
    assert len(grid_rows["official_grid_trilinear3d"]["official_output_sha256"]) == 64
    assert len(grid_rows["official_grid_trilinear3d"]["portable_output_sha256"]) == 64
    assert grid_rows["official_grid_trilinear3d"]["blockers"] == []
    assert grid_rows["official_patch_index_path"]["source_forward_parity_proven"] is True
    assert grid_rows["official_patch_index_path"]["full_hinerv_forward_parity_proven"] is False
    assert grid_rows["official_patch_index_path"]["max_abs_error"] == 0.0
    assert grid_rows["official_patch_index_path"]["output_hashes_bit_identical"] is True
    assert grid_rows["official_patch_index_path"]["blockers"] == []
    assert (
        grid_rows["official_patch_dataset_video_dataset"][
            "source_forward_parity_proven"
        ]
        is True
    )
    assert (
        grid_rows["official_patch_dataset_video_dataset"][
            "full_hinerv_forward_parity_proven"
        ]
        is False
    )
    assert (
        grid_rows["official_patch_dataset_video_dataset"]["backend"]
        == "official_dataset_video_dataset_vs_numpy"
    )
    assert grid_rows["official_patch_dataset_video_dataset"]["max_abs_error"] == 0.0
    assert grid_rows["official_patch_dataset_video_dataset"]["blockers"] == []
    artifact_states = {row["component_id"]: row for row in artifact["component_rows"]}
    assert artifact_states["core_hierarchical_renderer"][
        "source_forward_parity_proven"
    ] is True
    assert artifact_states["core_hierarchical_renderer"][
        "source_forward_parity_falsified"
    ] is False
    assert artifact_states["patch_dataset_path"]["source_forward_parity_proven"] is True
    assert artifact_states["patch_dataset_path"]["source_forward_parity_falsified"] is False
    assert artifact_states["patch_dataset_path"]["classification"] == (
        "official_patch_dataset_source_parity_proven"
    )
    assert artifact_states["prune_quant_codec"]["source_forward_parity_falsified"] is True

    artifact_path = tmp_path / "forward_parity.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260603T000000Z",
    )

    assert report["official_forward_parity_proven"] is False
    assert report["official_forward_parity_artifact_row"]["parity_passed"] is False
    assert report["official_forward_parity_artifact_row"]["parity_falsified"] is True
    assert report["official_forward_parity_artifact_row"][
        "falsification_accepted"
    ] is True
    assert (
        "hinerv_official_forward_parity_artifact_not_passing"
        not in report["official_forward_parity_artifact_row"]["blockers"]
    )
    states = {row["component_id"]: row for row in report["component_state_rows"]}
    assert states["core_hierarchical_renderer"]["classification"] == (
        "official_source_forward_parity_proven"
    )
    assert states["core_hierarchical_renderer"]["source_forward_parity_proven"] is True
    assert states["core_hierarchical_renderer"]["source_forward_parity_falsified"] is False
    assert states["core_hierarchical_renderer"]["official_source_forward_replay"][
        "replay_ran"
    ] is True
    assert states["patch_dataset_path"]["classification"] == (
        "official_source_forward_parity_proven"
    )
    assert states["patch_dataset_path"]["source_forward_parity_proven"] is True
    assert "hinerv_official_forward_parity_artifact_falsifies_parity" not in (
        states["core_hierarchical_renderer"]["blockers"]
    )
    summary = summarize_hinerv_official_source_audit(report)
    replay_rows = {
        row["component_id"]: row
        for row in summary["numeric_subcomponent_replays"]
    }
    assert replay_rows["official_grid_trilinear3d"]["source_forward_parity_proven"] is True
    assert replay_rows["official_grid_trilinear3d"][
        "full_hinerv_forward_parity_proven"
    ] is False
    assert replay_rows["official_patch_index_path"]["source_forward_parity_proven"] is True
    assert replay_rows["official_patch_index_path"][
        "full_hinerv_forward_parity_proven"
    ] is False
    assert replay_rows["official_patch_index_path"]["max_abs_error"] == 0.0
    assert replay_rows["official_patch_dataset_video_dataset"][
        "source_forward_parity_proven"
    ] is True
    assert replay_rows["official_patch_dataset_video_dataset"]["max_abs_error"] == 0.0


def test_hinerv_official_patch_dataset_replay_detects_source_mismatch(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(
        tmp_path,
        corrupt_dataset_patch=True,
    )

    artifact = build_hinerv_official_forward_parity_artifact(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    replay_rows = {
        row["component_id"]: row
        for row in artifact["numeric_subcomponent_rows"]
    }
    replay = replay_rows["official_patch_dataset_video_dataset"]
    assert replay["source_forward_parity_proven"] is False
    assert replay["max_abs_error"] > 0.0
    assert "hinerv_official_patch_dataset_source_replay_mismatch" in replay[
        "blockers"
    ]
    states = {row["component_id"]: row for row in artifact["component_rows"]}
    assert states["patch_dataset_path"]["source_forward_parity_falsified"] is True
    assert "hinerv_patch_dataset_source_forward_replay_failed" in states[
        "patch_dataset_path"
    ]["blockers"]


def test_hinerv_official_core_replay_rejects_unmapped_state(
    tmp_path: Path,
) -> None:
    official = _write_minimal_official_hinerv_repo(tmp_path, unmapped_core=True)

    artifact = build_hinerv_official_forward_parity_artifact(
        official_repo_dir=official,
        repo_root=REPO_ROOT,
        generated_utc="20260603T000000Z",
    )

    replay = artifact["source_forward_replay"]
    assert replay["replay_ran"] is True
    assert replay["full_hinerv_forward_parity_proven"] is False
    assert replay["portable_output_sha256"] is None
    assert "hinerv_official_state_not_mappable_to_local_portable_core" in replay[
        "blockers"
    ]
    states = {row["component_id"]: row for row in artifact["component_rows"]}
    assert states["core_hierarchical_renderer"]["source_forward_parity_proven"] is False
    assert states["core_hierarchical_renderer"]["source_forward_parity_falsified"] is True
    assert "hinerv_core_hierarchical_renderer_source_forward_replay_failed" in states[
        "core_hierarchical_renderer"
    ]["blockers"]


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


def _write_minimal_official_hinerv_repo(
    tmp_path: Path,
    *,
    corrupt_dataset_patch: bool = False,
    unmapped_core: bool = False,
) -> Path:
    root = tmp_path / "HiNeRV"
    (root / "models").mkdir(parents=True)
    (root / "compression").mkdir(parents=True)
    (root / "cfgs/models").mkdir(parents=True)
    tiny_model_source = (
        """
import torch

class _Group:
    def add_argument(self, *args, **kwargs):
        return None

group = _Group()

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

class TinyHiNeRV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor([0.125], dtype=torch.float32))

    def forward(self, input):
        idx = input['idx'].to(torch.float32)
        return (
            torch.ones((idx.shape[0], 3, 1, 2, 2), dtype=torch.float32)
            * (self.weight + self.bias).view(1, 1, 1, 1, 1)
        ).contiguous(memory_format=torch.channels_last_3d)
"""
        if unmapped_core
        else """
import torch

class _Group:
    def add_argument(self, *args, **kwargs):
        return None

group = _Group()

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

class TinyHiNeRV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

    def forward(self, input):
        idx = input['idx'].to(torch.float32)
        return (
            torch.ones((idx.shape[0], 3, 1, 2, 2), dtype=torch.float32)
            * self.weight.view(1, 1, 1, 1, 1)
        ).contiguous(memory_format=torch.channels_last_3d)
"""
    )
    (root / "models/hinerv.py").write_text(
        tiny_model_source
        + """

def build_model(args, logger, input):
    return TinyHiNeRV()

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
    patch_corruption = (
        "            patch = torch.zeros_like(patch)\n"
        if corrupt_dataset_patch
        else ""
    )
    (root / "datasets.py").write_text(
        f"""
import math
import os
import torch
import torchvision

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, logger, root, name, crop=[-1, -1], resize=[-1, -1], patch_size=[1, -1, -1], cached='none'):
        self.logger = logger
        self.root = os.path.expanduser(root)
        self.name = name
        self.img_paths = sorted([f for f in os.listdir(os.path.join(self.root, self.name)) if not f.startswith(".")])
        self.raw_size = self.get_raw_size()
        self.crop = tuple(crop[d] if crop[d] != -1 else self.raw_size[d] for d in range(2))
        self.resize = tuple(resize[d] if resize[d] != -1 else self.crop[d] for d in range(2))
        self.video_size = (len(self.img_paths), self.resize[0], self.resize[1])
        self.patch_size = tuple(patch_size[d] if patch_size[d] != -1 else self.video_size[d] for d in range(3))
        assert all(self.video_size[d] % self.patch_size[d] == 0 for d in range(3))
        self.num_patches = tuple(self.video_size[d] // self.patch_size[d] for d in range(3))
        assert cached in ['none', 'image', 'patch']
        self.cached = cached
        self.load_cache()

    def load_cache(self):
        if self.cached == 'image' or self.cached == 'patch':
            self.image_cached = self.load_all_images()
        else:
            self.image_cached = None
        if self.cached == 'patch':
            self.patch_cached = self.load_all_patches()
            self.image_cached = None
        else:
            self.patch_cached = None

    def get_raw_size(self):
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[0]))
        return img.shape[1:3]

    def load_image(self, idx):
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[idx]))
        img = torchvision.transforms.functional.center_crop(img, self.crop)
        img = torchvision.transforms.functional.resize(img, self.resize, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
        return img

    def load_patch(self, idx):
        patches = []
        h = idx[1] * self.patch_size[1]
        w = idx[2] * self.patch_size[2]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.image_cached[t] if self.image_cached is not None else self.load_image(t)
            patch = image[:, None, h: h + self.patch_size[1], w: w + self.patch_size[2]]
{patch_corruption}            patches.append(patch)
        return torch.concatenate(patches, dim=1)

    def load_all_images(self):
        return {{t: self.load_image(t) for t in range(self.video_size[0])}}

    def load_all_patches(self):
        patches = {{}}
        for t in range(self.num_patches[0]):
            for h in range(self.num_patches[1]):
                for w in range(self.num_patches[2]):
                    patches[(t, h, w)] = self.load_patch((t, h, w))
        return patches

    def get_patch(self, idx):
        return self.patch_cached[idx] if self.patch_cached is not None else self.load_patch(idx)

    def __len__(self):
        return math.prod(self.num_patches)

    def __getitem__(self, idx):
        idx_thw = (
            idx // (self.num_patches[1] * self.num_patches[2]),
            (idx % (self.num_patches[1] * self.num_patches[2])) // self.num_patches[2],
            (idx % (self.num_patches[1] * self.num_patches[2])) % self.num_patches[2],
        )
        patch = self.get_patch(idx_thw)
        return torch.tensor(idx_thw, dtype=int), torch.clone(patch).float() / 255.

def load_all_patches():
    return None

def set_dataset_args(parser):
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--patch-size')

cached = 'patch'
patch_size = [1, 2, 3]
""",
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
