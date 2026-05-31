# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from comma_lab.scheduler.scorer_region_selector_chain_queue import (
    build_scorer_region_selector_chain_queue,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.scorer_region_waterfill import (
    FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA,
    P18_SEGNET_REGION_WATERFILL_SCHEMA,
    build_frame1_region_waterfill_runtime_patch,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _submission(root: Path) -> Path:
    submission = root / "submission"
    (submission / "src").mkdir(parents=True)
    (submission / "archive.zip").write_bytes(b"fake")
    (submission / "inflate.py").write_text(
        "import sys\n"
        "from model import HNeRVDecoder  # type: ignore[import-not-found]\n"
        "def f():\n"
        "    n_pairs = int(meta[\"n_pairs\"])\n"
        "        for i in range(0, n_pairs, 16):\n"
        "            j = min(i + 16, n_pairs)\n"
        "            rounded = apply_pr101_selector_to_frames(\n"
        "                rounded,\n"
        "                selector_kind,\n"
        "                selector_codes,\n"
        "                selector_specs,\n"
        "                pair_start=i,\n"
        "            )\n"
        "            frames = rounded.to(torch.uint8)\n",
        encoding="utf-8",
    )
    return submission


def _p18(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "rows": [
                {
                    "pair_id": 0,
                    "regions256": [
                        {
                            "box": {"x0": 0.0, "y0": 0.0, "x1": 0.25, "y1": 0.25},
                            "class_id": 0,
                        },
                        {
                            "box": {"x0": 0.25, "y0": 0.0, "x1": 0.5, "y1": 0.25},
                            "class_id": 1,
                        },
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    return path


def test_frame1_region_patch_uses_class_conditioned_bulk_fill(tmp_path: Path) -> None:
    submission = _submission(tmp_path)
    p18 = _p18(tmp_path / "p18.json")

    payload = build_frame1_region_waterfill_runtime_patch(
        repo_root=tmp_path,
        source_submission_dir=submission,
        segnet_region_waterfill=p18,
        output_submission_dir=tmp_path / "patched",
        regions_per_pair=2,
        rgb_delta=(-1, -1, -1),
        class_rgb_delta_table={0: (3, -1, -1), 1: (-2, -2, -2)},
        overwrite=True,
    )

    assert payload["schema"] == FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA
    assert payload["bulk_fill_policy"]["segnet_class_conditioned"] is True
    assert payload["bulk_fill_policy"]["value_selection"] == "segnet_class_id_to_rgb_delta_table"
    assert payload["class_rgb_delta_table"] == {"0": [3, -1, -1], "1": [-2, -2, -2]}
    patch_source = (tmp_path / "patched" / "src" / "region_waterfill_patch.py").read_text(
        encoding="utf-8"
    )
    assert "(3, -1, -1), 0" in patch_source
    assert "(-2, -2, -2), 1" in patch_source
    assert "rgb_delta" in patch_source


def test_chain_queue_passes_class_conditioned_bulk_fill_flags(tmp_path: Path) -> None:
    submission = _submission(tmp_path)
    p18 = _p18(tmp_path / "p18.json")

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="class_conditioned_chain",
        source_submission_dir=submission,
        output_root=tmp_path / "out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        segnet_region_masks=p18,
        materialize_receiver_patch=True,
        receiver_patch_rgb_delta=(-1, -1, -1),
        receiver_patch_class_rgb_deltas={0: (3, -1, -1), 1: (-2, -2, -2)},
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "materialize_frame1_region_waterfill_runtime_patch"
    )
    assert command.count("--class-rgb-delta") == 2
    assert "0:3,-1,-1" in command
    assert "1:-2,-2,-2" in command
    policy = queue["metadata"]["receiver_patch_bulk_fill_policy"]
    assert policy["segnet_class_conditioned"] is True
    assert policy["class_rgb_delta_table"] == {"0": [3, -1, -1], "1": [-2, -2, -2]}
