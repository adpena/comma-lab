# SPDX-License-Identifier: MIT
"""Canonical example compress-time passes for the tac.compress_time_optimization
namespace.

Importing this package registers the example passes into
``_REGISTERED_PASSES`` via the ``@compress_time_pass`` decorator side effect.
The examples are TOY implementations (zero bytes_added; trivial state
mutations) so they are testable without GPU / large model state.
"""

from tac.compress_time_optimization.examples.example_passes import (  # noqa: F401
    bisect_int8_scale_per_block_example,
    coord_search_fec6_k16_palette8_example,
    multipass_quant_depth_3_example,
    raw_quant_example,
    sa_fec6_selector_indices_example,
    tto_pose_per_pair_example,
)

__all__ = [
    "bisect_int8_scale_per_block_example",
    "coord_search_fec6_k16_palette8_example",
    "multipass_quant_depth_3_example",
    "raw_quant_example",
    "sa_fec6_selector_indices_example",
    "tto_pose_per_pair_example",
]
