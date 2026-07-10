# SPDX-License-Identifier: MIT
"""through_r — the canonical through-R measurement harness + scaffold assembler.

CANONICALIZATION UNIT 1 (#388): the two most-rebuilt patterns in the campaign, unified so
subagents CONSUME one authority instead of re-deriving (and re-risking the flip-resolution
bug class). Three modules:

* :mod:`resolution_chain` — THE authoritative pinned R resolution chain (camera 874x1164,
  SegNet 384x512, seq_len 2), source-verified against ``upstream/evaluate.py`` +
  ``upstream/modules.py`` + :mod:`tac.contest_eval_contract`; the R first-half operator
  (render-grid -> bicubic UP -> uint8 camera); ``describe()`` provenance dump; and an
  explicit WH-vs-HW transposition-hazard guard.
* :mod:`harness` — :func:`measure_through_r`: candidate frames -> R -> frozen CPU-torch
  SegNet argmax -> per-class + aggregate d_seg vs cached ``lstars`` (n600 by default,
  toy-refusing; ``backend='cpu-torch'`` the only authority; chunked per the OOM law).
* :mod:`scaffold_assembler` — the canonical composite-argmax assembler (Laguerre/tropical
  argmax composition, pluggable ``b_c``, bounded reconciliation). The former
  :mod:`tac.inc1a_harness.composite_assembler` now re-exports from here.

Authority: realized-through-R is CPU-torch-authority-grade but ``[macOS-CPU advisory .
NON-PROMOTABLE]`` — the pointer (contest-CPU 0.19110) moves ONLY through a byte-closed
``upstream/evaluate.py`` exact row. This package is MEANS (apparatus), never a lever.
"""

from __future__ import annotations

from tac.through_r.compare import (
    LabelStackCompare,
    compare_label_stack_to_lstars,
)
from tac.through_r.harness import (
    DEFAULT_GT_CACHE,
    DEFAULT_VERDICT_BATCH,
    N600,
    SUPPORTED_BACKENDS,
    THROUGH_R_LABEL,
    ThroughRHarnessError,
    ThroughRResult,
    load_frozen_segnet,
    load_gt_lstars,
    measure_through_r,
)
from tac.through_r.palette_realization import (
    ArmResult,
    DecisionGeometry,
    FlipDecomposition,
    PaletteRealizationError,
    classify_color,
    decompose_flips,
    global_mean_palette,
    map_decision_geometry,
    mixing_robust_palette,
    naive_mean_palette,
    paint_camera_res_uint8,
    paint_render_grid,
    paint_seg_grid,
    realize_argmax_no_R,
    run_arm,
)
from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_HW,
    CAMERA_SIZE_WH,
    CAMERA_W,
    SCORER_INPUT_SIZE_WH,
    SEG_H,
    SEG_HW,
    SEG_W,
    SEQ_LEN,
    ResolutionChainError,
    contest_faithful_R_numpy,
    describe,
    read_upstream_constants,
    render_grid_to_camera_uint8,
    verify_against_upstream,
)
from tac.through_r.scaffold_assembler import (
    BC_MODES,
    N_SEG_CLASSES,
    CarrierField,
    CompositeResult,
    Inc1aAssemblerError,
    ScaffoldAssemblerError,
    assemble_fields,
    compose_partition,
    reconcile_partition,
)

__all__ = [
    "BC_MODES",
    "CAMERA_H",
    "CAMERA_HW",
    "CAMERA_SIZE_WH",
    "CAMERA_W",
    "DEFAULT_GT_CACHE",
    "DEFAULT_VERDICT_BATCH",
    "N600",
    "N_SEG_CLASSES",
    "SCORER_INPUT_SIZE_WH",
    "SEG_H",
    "SEG_HW",
    "SEG_W",
    "SEQ_LEN",
    "SUPPORTED_BACKENDS",
    "THROUGH_R_LABEL",
    "ArmResult",
    "CarrierField",
    "CompositeResult",
    "DecisionGeometry",
    "FlipDecomposition",
    "Inc1aAssemblerError",
    "LabelStackCompare",
    "PaletteRealizationError",
    "ResolutionChainError",
    "ScaffoldAssemblerError",
    "ThroughRHarnessError",
    "ThroughRResult",
    "assemble_fields",
    "classify_color",
    "compare_label_stack_to_lstars",
    "compose_partition",
    "contest_faithful_R_numpy",
    "decompose_flips",
    "describe",
    "global_mean_palette",
    "load_frozen_segnet",
    "load_gt_lstars",
    "map_decision_geometry",
    "measure_through_r",
    "mixing_robust_palette",
    "naive_mean_palette",
    "paint_camera_res_uint8",
    "paint_render_grid",
    "paint_seg_grid",
    "read_upstream_constants",
    "realize_argmax_no_R",
    "reconcile_partition",
    "render_grid_to_camera_uint8",
    "run_arm",
    "verify_against_upstream",
]
