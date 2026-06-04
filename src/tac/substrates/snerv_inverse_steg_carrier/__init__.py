# SPDX-License-Identifier: MIT
"""SNeRV inverse-steganalysis carrier — super-small-rate-by-design.

A spectra-preserving SNeRV (arXiv 2501.01681) carrier that STORES only the low-frequency
(coarse) orthonormal-DWT approximation and GENERATES the high-frequency detail
with a small decoder, then places the stored LF bits with the proven L-inf
pose-Fisher-led inverse-steganalysis allocator (``allocation.py``) via the EXACT
DWT synthesis adjoint (``dwt.py``, rel-residual 0.0).

This is the canonical cure for the "Z8 disease" (the raw wavelet detail blob that
was ~99.5% of the Z8 archive): here the byte-heavy detail is *generated*, not
stored.

Layers (per the inverse-steganalysis full-stack design memo §4):
  * ``dwt`` — L2/L3 orthonormal 2D DWT + the exact synthesis adjoint (G3).
  * ``carrier`` — L2 store-LF / generate-HF carrier + ridge-fit HF decoder.
  * ``allocation`` — L3 L-inf pose-Fisher allocation pushed into the LF domain.

NON-PROMOTABLE: all numbers measured by this substrate are ``[macOS-CPU advisory]``
(Catalog #341/#192/#127/#323); paired CPU+CUDA (Catalog #246) reserved for operator
authorization. The receiver NEVER loads the scorer (``tac.contest_eval_contract``).
"""

from __future__ import annotations

from tac.substrates.snerv_inverse_steg_carrier.allocation import (
    LfSaliency,
    SnervAllocationError,
    allocate_lf_linf,
    push_pixel_saliency_to_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    DEFAULT_SNERV_MODEL_SIZE,
    SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF,
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
    HighFrequencyRestorer,
    MultiResolutionFusionUnit,
    SnervCarrierError,
    SnervFrameCode,
    SnervModelSizeConfig,
    SnervTemporalExtension,
    decode_frame,
    dequantize_lf,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    fit_hf_decoder_weighted_least_squares,
    generate_hf_from_lf,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    DEFAULT_WAVELET,
    DWT_MODE,
    SnervDwtError,
    WaveletPyramid,
    dwt2_multilevel,
    dwt2_native_synthesis_adjoint,
    idwt2_multilevel,
    lf_coeff_count,
    synthesis_adjoint_residual,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    FALSE_AUTHORITY as OFFICIAL_SNERV_HFR_FALSE_AUTHORITY,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
    OFFICIAL_SNERV_HFR_SOURCE_SHA,
    SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF,
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
    OfficialHfrHeadsOutput,
    OfficialSnervHfrError,
    conv2d_nchw,
    conv2d_nchw_mlx,
    leaky_relu01,
    leaky_relu01_mlx,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    OFFICIAL_SNERV_MFU_SOURCE,
    OFFICIAL_SNERV_RB_SOURCE,
    OFFICIAL_SNERV_T_MFU_SOURCE,
    SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF,
    ConvTranspose2dShapeSpec,
    NchwShape,
    OfficialConvTranspose2dNchw,
    OfficialMfuShapeTrace,
    OfficialResidualBlockNoBN,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuError,
    OfficialSnervMfuForwardOutput,
    OfficialSnervMfuSpec,
    ResidualBlocksWithInputConvSpec,
    TensorSpec,
    concat_nchw_arrays,
    concat_nchw_mlx,
    concat_nchw_specs,
    conv_transpose2d_nchw,
    conv_transpose2d_nchw_mlx,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    OFFICIAL_SNERV_T_TUB_EVIDENCE_SCOPE,
    OFFICIAL_SNERV_T_TUB_SCHEMA,
    OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
    OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS,
    OfficialOutput2FusionResult,
    OfficialOutput2FusionShape,
    OfficialTubError,
    OfficialTubGraphInputs,
    OfficialTubShapeMetadata,
    official_output2_fusion_mlx,
    official_output2_fusion_numpy,
    official_output2_fusion_shape,
    official_output2_fusion_torch,
    prepare_official_tub_graph_inputs,
)

__all__ = [
    "DEFAULT_SNERV_MODEL_SIZE",
    "DEFAULT_WAVELET",
    "DWT_MODE",
    "OFFICIAL_SNERV_HFR_FALSE_AUTHORITY",
    "OFFICIAL_SNERV_HFR_SOURCE_CONTRACT",
    "OFFICIAL_SNERV_HFR_SOURCE_SHA",
    "OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS",
    "OFFICIAL_SNERV_MFU_SOURCE",
    "OFFICIAL_SNERV_RB_SOURCE",
    "OFFICIAL_SNERV_T_MFU_SOURCE",
    "OFFICIAL_SNERV_T_SOURCE_SHA",
    "OFFICIAL_SNERV_T_TUB_EVIDENCE_SCOPE",
    "OFFICIAL_SNERV_T_TUB_SCHEMA",
    "OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT",
    "OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS",
    "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
    "SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF",
    "SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER",
    "SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF",
    "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
    "SNERV_SPECTRA_PRESERVING_ADAPTER",
    "ConvTranspose2dShapeSpec",
    "HfGenerationDecoder",
    "HighFrequencyRestorer",
    "LfSaliency",
    "MultiResolutionFusionUnit",
    "NchwShape",
    "OfficialConv2dNchw",
    "OfficialConvTranspose2dNchw",
    "OfficialHfrConvBlock",
    "OfficialHfrHeads",
    "OfficialHfrHeadsOutput",
    "OfficialMfuShapeTrace",
    "OfficialOutput2FusionResult",
    "OfficialOutput2FusionShape",
    "OfficialResidualBlockNoBN",
    "OfficialResidualBlocksWithInputConv",
    "OfficialSnervHfrError",
    "OfficialSnervMfu",
    "OfficialSnervMfuError",
    "OfficialSnervMfuForwardOutput",
    "OfficialSnervMfuSpec",
    "OfficialTubError",
    "OfficialTubGraphInputs",
    "OfficialTubShapeMetadata",
    "ResidualBlocksWithInputConvSpec",
    "SnervAllocationError",
    "SnervCarrierError",
    "SnervDwtError",
    "SnervFrameCode",
    "SnervModelSizeConfig",
    "SnervTemporalExtension",
    "TensorSpec",
    "WaveletPyramid",
    "allocate_lf_linf",
    "concat_nchw_arrays",
    "concat_nchw_mlx",
    "concat_nchw_specs",
    "conv2d_nchw",
    "conv2d_nchw_mlx",
    "conv_transpose2d_nchw",
    "conv_transpose2d_nchw_mlx",
    "decode_frame",
    "dequantize_lf",
    "dwt2_multilevel",
    "dwt2_native_synthesis_adjoint",
    "encode_frame_lf",
    "fit_hf_decoder_least_squares",
    "fit_hf_decoder_weighted_least_squares",
    "generate_hf_from_lf",
    "idwt2_multilevel",
    "leaky_relu01",
    "leaky_relu01_mlx",
    "lf_coeff_count",
    "official_output2_fusion_mlx",
    "official_output2_fusion_numpy",
    "official_output2_fusion_shape",
    "official_output2_fusion_torch",
    "prepare_official_tub_graph_inputs",
    "push_pixel_saliency_to_lf",
    "quantize_lf",
    "synthesis_adjoint_residual",
]
