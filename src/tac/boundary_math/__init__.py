# SPDX-License-Identifier: MIT
"""boundary_math — the seg-axis core of the evaluator-equivalence quotient compiler.

Source spec: ``.omx/research/closed_spec_boundary_math_system_of_equations_20260610.md``.

The seg-scored object is the SegNet **argmax partition** ``L*`` on the 600 frame1
images (``upstream/modules.py``: SegNet reads ``x[:,-1,...]``, 5 classes, resized to
384x512, raw [0,255]).  ``d_seg`` is the per-pixel argmax-flip RATE vs ``L*``.

This package replaces dense O(area) data structures + smooth-surrogate search with
the RIGHT data structures + a real combinatorial SOLVE:

- :mod:`tac.boundary_math.partition` — region adjacency graph + connected components
  of ``L*`` (contiguity).  The O(boundary) representation.
- :mod:`tac.boundary_math.dense_raster_lzma_baseline` — exact dense-label
  RAW-LZMA2 baseline; not a boundary-edge codec.
- :mod:`tac.boundary_math.context_partition_codec` — context/range-coded
  partition-store replacement for the dense baseline.
- :mod:`tac.boundary_math.bitmask_dseg` — exact ``d_seg = popcount(XOR)/N`` over the
  5 class masks (the bitmask functional that evaluate.py actually scores).
- :mod:`tac.boundary_math.margin_polytope` — per-pixel margin-polytope free-budget
  ``b(p) = m(p)/||g_p||`` (the system of linear inequalities, §4).
- :mod:`tac.boundary_math.region_merge` — the MDL region-merge SOLVE at the
  waterfilling water level ``lambda* = 1.27 B/flip`` (§10).  A real weighted
  greedy graph-cut over the RAG, NOT a parameter sweep, NOT a search.

NO FAKE: every function performs the named work on the real partition array; tests
verify BEHAVIOR (roundtrip identity, popcount-vs-reference, merge-drops-the-predicted
regions, polytope-budget correctness), never constants.
"""

from __future__ import annotations

from tac.boundary_math.bitmask_dseg import (
    class_masks_from_argmax,
    d_seg_bitmask,
    d_seg_reference,
)
from tac.boundary_math.dense_raster_lzma_baseline import (
    ContourCode,
    DenseRasterLzmaCode,
    decode_partition,
    encode_partition,
    partition_description_bytes,
)
from tac.boundary_math.legal_frame_bridge import (
    ClassPalette,
    fit_palette_gt_region_mean,
    rasterize_palette_frame,
)
from tac.boundary_math.lever_b_generator import (
    GeneratorConfig,
    ResidualComponentStats,
    generator_argmax,
    load_generator_npz,
    residual_component_stats,
    save_generator_npz,
)
from tac.boundary_math.lane_headstart import (
    LANE_CLASS,
    SUB015_LANE_DSEG,
    SUB019_LANE_DSEG,
    LaneCenterline,
    LaneHeadstartCorpusResult,
    apply_lane_residual,
    build_lane_headstart,
    compute_lane_residual,
    deserialize_and_rasterize,
    estimate_base_bytes,
    fit_centerlines,
    gauge_cost_cell,
    lane_base_from_argmax,
    rasterize_centerlines,
    residual_support_frac,
    serialize_centerlines_delta,
)
from tac.boundary_math.margin_conditional_residual import (
    BYTES_PER_SCORE,
    SEG_VALUE_PER_FLIP,
    WATERLINE_BYTES_PER_FLIP,
    ResidualCodeCost,
    WaterfillResult,
    boundary_set_from_margin,
    class_bits_conditional,
    conditional_position_bits,
    decode_residual,
    encode_residual,
    measure_code_cost,
    unconditional_position_bits,
    waterfill_select,
)
from tac.boundary_math.margin_polytope import (
    PerPixelFreeBudget,
    free_budget_from_margin_jacobian,
)
from tac.boundary_math.partition import (
    Region,
    RegionAdjacencyGraph,
    build_region_adjacency_graph,
    connected_components,
)
from tac.boundary_math.region_merge import (
    WATER_LEVEL_BYTES_PER_FLIP,
    RegionMergePlan,
    solve_mdl_region_merge,
)
from tac.boundary_math.seg_core import (
    SegCoreResult,
    build_and_measure_lstar,
    decode_gt_frame1_pairs,
    load_real_segnet,
    segnet_argmax_and_margin,
    solve_candidate_against_lstar,
)

__all__ = [
    "BYTES_PER_SCORE",
    "LANE_CLASS",
    "SEG_VALUE_PER_FLIP",
    "SUB015_LANE_DSEG",
    "SUB019_LANE_DSEG",
    "WATERLINE_BYTES_PER_FLIP",
    "WATER_LEVEL_BYTES_PER_FLIP",
    "ClassPalette",
    "ContourCode",
    "DenseRasterLzmaCode",
    "GeneratorConfig",
    "LaneCenterline",
    "LaneHeadstartCorpusResult",
    "PerPixelFreeBudget",
    "Region",
    "apply_lane_residual",
    "build_lane_headstart",
    "compute_lane_residual",
    "deserialize_and_rasterize",
    "estimate_base_bytes",
    "fit_centerlines",
    "gauge_cost_cell",
    "lane_base_from_argmax",
    "rasterize_centerlines",
    "residual_support_frac",
    "serialize_centerlines_delta",
    "RegionAdjacencyGraph",
    "RegionMergePlan",
    "ResidualCodeCost",
    "ResidualComponentStats",
    "SegCoreResult",
    "WaterfillResult",
    "boundary_set_from_margin",
    "build_and_measure_lstar",
    "build_region_adjacency_graph",
    "class_bits_conditional",
    "class_masks_from_argmax",
    "conditional_position_bits",
    "connected_components",
    "d_seg_bitmask",
    "d_seg_reference",
    "decode_gt_frame1_pairs",
    "decode_partition",
    "decode_residual",
    "encode_partition",
    "encode_residual",
    "fit_palette_gt_region_mean",
    "free_budget_from_margin_jacobian",
    "generator_argmax",
    "load_generator_npz",
    "load_real_segnet",
    "measure_code_cost",
    "partition_description_bytes",
    "rasterize_palette_frame",
    "residual_component_stats",
    "save_generator_npz",
    "segnet_argmax_and_margin",
    "solve_candidate_against_lstar",
    "solve_mdl_region_merge",
    "unconditional_position_bits",
    "waterfill_select",
]
