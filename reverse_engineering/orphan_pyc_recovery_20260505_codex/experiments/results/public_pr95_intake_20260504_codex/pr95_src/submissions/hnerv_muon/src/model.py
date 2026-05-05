# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``15:1: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``model.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr95_intake_20260504_codex/pr95_src/submissions/hnerv_muon/src/model.py'
__recovery_spec__ = 'model.recovery_spec.json'
__recovery_ast_error__ = '15:1: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
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

"""
