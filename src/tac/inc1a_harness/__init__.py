# SPDX-License-Identifier: MIT
"""increment-1a HARNESS — the paint-free MASK-level DECOUPLING PARTITION SCREEN.

Redefined per crucible-3 P3b F1 (``SYNTHESIS_v2_v8_20260709.md`` §A.4):
1a measures the composite tropical-argmax **MASK d_seg vs the GT SegNet argmax L\\***
on ``gt_n600.npz`` — **PAINT-FREE (no RGB, no R, no paint)** — run as an A/B against
a matched-compute shared-head CONTROL arm. Pre-registered KILL: the decoupled arm's
mask d_seg must beat the control's by > ``delta_mask``. Both arms paint-free ⇒ the
flat-paint 0.0064 confound is EXCLUDED by construction. This is a NECESSARY-CONDITION
proxy (mask-optimal ≠ score-optimal, risk-3); the through-R SUFFICIENT test is 1b.

This package is HARNESS-ONLY. It does NOT train the arms (a later governed EVENT) and
it does NOT edit the sibling ``boundary_math`` / ``curriculum_dsl`` surfaces — it
CONSUMES their built primitives (``power_diagram_argmax``, ``solve_head_offsets``,
``signed_distance_fields``, ``bitmask_dseg.d_seg_reference``,
``perclass_verdict.per_class_flip_stats``, the analytic carrier builders).

Binding clause A (dedup / geometry-first): every harness output states its geometric
home; the assembler stores NOTHING derivable — it is deterministic code = rule-118 FREE
(no video-derived table is emitted). Binding clause B (§I): every carrier is tagged
GEOMETRIC-MINIMAL or KKT-WATERFILLED.

Pointer contest-CPU 0.19110 UNMOVED — this harness is MEANS. The smoke number is a
``[analytic-generators, no-trained-fields]`` harness validation + a first honest floor
row, NOT the 1a verdict (which needs trained decoupled fields + a matched control arm).
"""

from __future__ import annotations

from tac.inc1a_harness.composite_assembler import (
    CarrierField,
    CompositeResult,
    assemble_fields,
    compose_partition,
    reconcile_partition,
)
from tac.inc1a_harness.decoupling_screen import (
    DELTA_MASK_FRAME_SAMPLING_FLOOR,
    ArmResult,
    ControlArmSpec,
    KillVerdict,
    evaluate_kill,
    matched_control_spec,
    operative_delta_mask,
)
from tac.inc1a_harness.mask_dseg_meter import MaskDsegResult, measure_mask_dseg

__all__ = [
    "DELTA_MASK_FRAME_SAMPLING_FLOOR",
    "ArmResult",
    "CarrierField",
    "CompositeResult",
    "ControlArmSpec",
    "KillVerdict",
    "MaskDsegResult",
    "assemble_fields",
    "compose_partition",
    "evaluate_kill",
    "matched_control_spec",
    "measure_mask_dseg",
    "operative_delta_mask",
    "reconcile_partition",
]
