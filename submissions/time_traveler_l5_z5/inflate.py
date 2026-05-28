#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard hierarchical predictive coding inflate runtime entry-point.

Delegates to the vendored substrate CLI per the canonical NSCS01 fail-closed-
template pattern (Catalog #295 self-containment) + Catalog #205 canonical
`select_inflate_device` + Catalog #146 contest 3-positional-arg signature.

This checked-in file is a TEMPLATE / research-dev surface. The trainer's
``_write_runtime`` overwrites it in the emitted ``submission_dir`` AND copies
the substrate codec package +
``tac.substrates._shared.inflate_runtime`` into ``submission_dir/src/tac/...``
so the trainer-emitted submission tree is truly self-contained per CLAUDE.md
HNeRV parity discipline L4 + L9 + Catalog #295 (canonical-template pattern;
sister of nscs01_nullspace_split_renderer/inflate.py).

The fail-closed ``RuntimeError`` below guarantees this checked-in template
cannot itself fire a contest dispatch — it raises BEFORE any ``sys.path.insert(...)``
can resolve onto the operator's working tree. Only the trainer-emitted
``submission_dir/inflate.py`` (which vendors ``src/tac/...`` alongside via
``experiments/train_substrate_time_traveler_l5_z5_mlx_local.py``'s
``_write_runtime``) actually executes inflate work.

Per Catalog #325 + the per-substrate symposium memo (council_t2_z5_rao_
ballard_hinton_distilled_per_substrate_symposium_20260528.md) the Z5
substrate is PROCEED_WITH_REVISIONS pending anchor 3/3 identity-predictor
disambiguator per Catalog #308; paired-CUDA RATIFICATION authority is gated
on (a) per-substrate symposium re-convene PROCEED-unconditional + (b)
operator-explicit per-PR per `[[pr-creation-requires-explicit-operator-
authorization-with-adversarial-negative-findings-audit-standing-directive-
20260528]]` + (c) Catalog #246 1:1 contest-compliant hardware per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

Canonical inflate runtime LOC: ~50 LOC fail-closed shim (under HNeRV parity
L4 ≤200 LOC substrate-engineering waiver budget; the heavy decoder lifting
happens in the vendored ``tac.substrates.time_traveler_l5_z5.inflate``).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
vendored_src = HERE / "src"
if not vendored_src.is_dir():
    raise RuntimeError(
        "Z5 Rao-Ballard submission runtime is not packaged: missing vendored src/. "
        "Use the trainer-emitted submission_dir artifact before any contest run "
        "(experiments/train_substrate_time_traveler_l5_z5_mlx_local.py::_write_runtime "
        "vendors src/tac/substrates/time_traveler_l5_z5/* + "
        "src/tac/substrates/_shared/inflate_runtime.py into submission_dir/src/tac/...)."
    )
sys.path.insert(0, str(vendored_src))  # SUBMISSION_PYTHONPATH_SHIM_OK:z5-checked-in-template-fail-closes-via-RuntimeError-above-when-src-missing-trainer-emitted-submission-dir-vendors-src-tac-alongside-per-write-runtime-per-catalog-295-canonical-nscs01-fail-closed-waiver-pattern

from tac.substrates.time_traveler_l5_z5.inflate import main_cli  # noqa: E402


def main() -> int:
    return main_cli()


if __name__ == "__main__":
    sys.exit(main())
