#!/usr/bin/env python
"""Shared utilities for the v62 codec training pipeline.

Public surface (used by master.py / slave.py / hpac.py):
  - Image-size constants: SEGNET_IN_W, SEGNET_IN_H, CAMERA_W, CAMERA_H
  - Data loader: preload_rgb_pairs_av (PyAV + yuv420_to_rgb, NVDEC-matching)
"""
from __future__ import annotations
import logging
import math
import sys
from pathlib import Path

import av
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
SEGNET_IN_H, SEGNET_IN_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164

# =============================================================================
# Data
# =============================================================================
def preload_rgb_pairs_av(video_dir: Path, names_file: Path) -> torch.Tensor:
    """v41: use yuv420_to_rgb (matches NVDEC/eval pixel format), not PyAV's rgb24.
    Eliminates the train/real eval gap caused by ffmpeg swscale ≠ NVDEC color conversion."""
    from frame_utils import yuv420_to_rgb
    logging.info("Preloading RGB pairs via PyAV + yuv420_to_rgb (NVDEC-matching)...")
    files = [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]
    out = []
    for fn in files:
        container = av.open(str(video_dir / fn))
        frames = [yuv420_to_rgb(fr).numpy() for fr in container.decode(video=0)]
        container.close()
        arr = np.stack(frames)
        N = (arr.shape[0] // 2) * 2
        arr = arr[:N].reshape(N // 2, 2, arr.shape[1], arr.shape[2], 3)
        out.append(torch.from_numpy(arr))
    return torch.cat(out, dim=0).contiguous()

