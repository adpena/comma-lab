# SPDX-License-Identifier: MIT
"""Canonical example inflate-time post-processing passes for the
tac.inflate_time_post_processing namespace.

Importing this package registers the example passes into
``_REGISTERED_PASSES`` via the ``@inflate_time_post_filter`` decorator
side effect. The examples are TOY implementations (identity transforms;
zero frames_processed) so they are testable without GPU / large model
state / cv2 / torch dependencies.
"""

from tac.inflate_time_post_processing.examples.example_passes import (  # noqa: F401
    bilateral_denoise_per_frame_example,
    lanczos_upscale_384_to_874_example,
    learned_unet_4block_per_frame_example,
    multi_pass_inflate_7_variants_example,
    nlm_denoise_per_pair_example,
    raw_inflate_example,
)

__all__ = [
    "bilateral_denoise_per_frame_example",
    "lanczos_upscale_384_to_874_example",
    "learned_unet_4block_per_frame_example",
    "multi_pass_inflate_7_variants_example",
    "nlm_denoise_per_pair_example",
    "raw_inflate_example",
]
