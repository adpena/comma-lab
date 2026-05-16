# SPDX-License-Identifier: MIT
"""Vendored tac.* shim for apogee_v2 inflate runtime self-containment.

Per CLAUDE.md HNeRV parity discipline L4 + L9 + Catalog #295
(NSCS06 v5 bug-class anchor; commit 0b50ceceb): submissions/<id>/inflate.py
MUST NOT depend on the operator's working tree to resolve `from tac.*`
imports at runtime. The apogee_v2 inflate decoder requires
`tac.water_filling_codec_v2.decode_omega_w_v2` (which transitively
requires `tac.arithmetic_qint_codec.decode_qints_arithmetic`) to
parse OWV2-encoded weight tensors back to float32.

This vendored package contains ONLY the decode-side modules needed at
inflate time. NO scorer code, NO torch.nn modules beyond what
`water_filling_codec_v2` itself imports (which is just torch tensors —
no SegNet/PoseNet/scorer loading).

Sister of:
  - submissions/nscs06_carmack_hotz_strip_everything/_nscs06_codec/
    (canonical NSCS06 v6 reference pattern @ commit 90bca47ff)
  - submissions/apogee_intN/src/codec.py + model.py
    (canonical sibling vendoring pattern)
"""
