# code callouts

Small, measured implementation details tied to the major score and rigor changes.

## Learned post-filter selector

- file: `submissions/robust_current/inflate.sh`
- why it matters: The promoted inflate path can route into the tiny learned post-filter.

```bash
    if [ "$PYTHON_INFLATE" = "postfilter" ]; then
      echo "Inflating (canonical + learned post-filter) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run --with av --with torch --with numpy python "$SELF_DIR/inflate_postfilter.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE" \
        "${POSTFILTER_PATH:-$SELF_DIR/postfilter_int8.pt}"
      break
```

## Runtime payload includes learned assets

- file: `src/comma_lab/install.py`
- why it matters: The honest installed payload explicitly includes the post-filter script and weights.

```bash
        "inflate.sh",
        "inflate.py",
    ),
    "robust_current": (
        "archive.zip",
        "inflate.sh",
        "inflate.py",
        "inflate_postfilter.py",
        "inflate_grain_mask.py",
        "postfilter_int8.pt",
        "config.env",
```

## Shipped post-filter module

- file: `submissions/robust_current/inflate_postfilter.py`
- why it matters: The filter is a tiny residual CNN loaded from shipped int8 weights.

```bash
#!/usr/bin/env python
"""Inflate path with learned post-filter applied after bicubic upscale.

The post-filter is a tiny CNN (3,203 params, 7.5KB int8) trained directly
against the scorer's loss function via backprop. It learns to correct the
decoded video to maximize PoseNet+SegNet scores.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av


DEFAULT_POSTFILTER_META = {
    "variant": "residual",
    "hidden": 16,
    "kernel": 3,
}


# ============================================================
# Post-filter model (matches training architecture)
# ============================================================
class PostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
```
