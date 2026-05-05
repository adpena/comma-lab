# Source Generated with Decompyle++
# File: model.cpython-312.pyc (Python 3.12)

'''HNeRV-style decoder: 229K params, single-video memorization.

Per-frame-pair latent (28-d) -> 6 upsample stages -> 384x512 RGB pair.

Each stage: Conv(in, out*4, 3x3) + PixelShuffle(2) + bilinear-skip + sin().
Final: dilated-conv refine residual + sigmoid RGB heads (separate frame 0 and 1).
'''
import torch
from torch.nn import nn

functional
<NODE:12> = None
