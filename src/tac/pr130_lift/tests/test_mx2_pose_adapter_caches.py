from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.build_mx2_pose_adapter_caches import build_master_cache


def test_build_master_cache_extracts_second_frame_per_pair(tmp_path: Path) -> None:
    pairs = 2
    h = 2
    w = 3
    raw_frames = np.arange(pairs * 2 * h * w * 3, dtype=np.uint8).reshape(
        pairs * 2, h, w, 3
    )
    raw = tmp_path / "0.raw"
    raw.write_bytes(raw_frames.tobytes())
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"checkpoint-identity")
    out = tmp_path / "OUR_SURFACE_MASTERS.pt"
    receipt = tmp_path / "receipt.json"

    report = build_master_cache(
        raw,
        checkpoint,
        out,
        receipt,
        pairs=pairs,
        camera_h=h,
        camera_w=w,
        chunk_pairs=1,
    )

    payload = torch.load(out, map_location="cpu", weights_only=False)
    expected = np.ascontiguousarray(raw_frames[[1, 3]].transpose(0, 3, 1, 2))
    assert payload["source_checkpoint"] == str(checkpoint.resolve())
    assert torch.equal(payload["masters"], torch.from_numpy(expected))
    assert report["masters"]["shape"] == [pairs, 3, h, w]
    assert report["surface_source"] == "tq1c parent inflated RGB raw frame_1 per pair"
    assert receipt.exists()
