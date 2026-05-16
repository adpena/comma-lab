# SPDX-License-Identifier: MIT
"""Z3-G1 entropy-coded v2 substrate package.

Per `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md`:
operator-approved 2026-05-15 reactivation of v1
(`lane_z3_g1_scorer_softmax_hyperprior_gating_20260515`, research_only=true,
deferred per F1 codex finding empirical confirmation that
`hyperprior_weights_int8` + `w_hat_int8` slots ship empty `b""`).

v2 introduces a NEW magic + grammar (`Z3G2`) that REPLACES the empty Z3HV2
slots with TWO entropy-coded streams shipped at the wire-byte level:

1. **sigma_table_blob** (~300-500B brotli'd): the 5x28 = 140 int8 sigma
   values, brotli-compressed at quality 11.
2. **class_index_blob** (~200-400B): the 600 per-pair dominant SegNet class
   indices, encoded via constriction's QueueEncoder under a per-class prior
   CDF derived from frequency counts over the 600 pairs. The class prior CDF
   (5 * uint16 counts = 10B) ships alongside the encoded stream.

Score movement is unranked until full-frame ``inflate.sh`` mutation proof and
paired CPU+CUDA exact eval exist. The rate-side calculation remains a planning
hypothesis: section ~1986B vs A1 latent_blob 15387B ⇒ ~13.4 KB savings ⇒ rate
contribution -0.0089 before any distortion-axis measurement.

Architecture (v2 entropy-coded grammar):

    For each pair p in range(600):
        class_p = mode of SegNet(GT_frame_p).argmax(1) per pixel  (in [0, 4])
        sigma_p = sigma_table[class_p, :]   (28-dim per-dim scale)
        AC-encode residual_p under N(0, sigma_p^2)

    Archive payload (Z3G2):
        - Z3G2 header (~40B fixed)
        - sigma_table_blob: brotli(140 int8 sigma) (~300B)
        - class_prior_cdf_blob: 5 * uint16 = 10B (frequency counts)
        - class_index_blob: AC-coded class indices (~200-400B)
        - residual_blob: brotli(600*28 int8 residual) (~1200B)
        - per_dim_affine: 2 * 28 * float32 = 224B (offset + scale)

The v2 substitution is contest-legal at the compress-side only:
- COMPRESS-SIDE: SegNet runs on GT frame (FREE per CLAUDE.md "Strict scorer
  rule" rule #2 — compress-side scorer use is FREE). Class indices ship
  via the Z3G2 sidecar.
- INFLATE-SIDE: NO scorer load. The 4-byte magic Z3G2 distinguishes from
  Z3HV2 (Z3V2); decoder unpacks sigma table + class prior CDF + AC-decodes
  class indices + AC-decodes per-pair residual under class-conditional
  Gaussian prior.

Per Catalog #220 this remains ``score_improvement_mechanism_status=RESEARCH_ONLY``
with ``runtime_overlay_consumed=False`` until full-frame ``inflate.sh``
mutation proof and paired exact eval land. The current evidence is parser /
intermediate structural consumption:

1. Encoder/decoder roundtrip test (encode known sigma + class indices →
   decode → assert identity).
2. Byte-mutation smoke per Catalog #139 (tool mutates one byte each in
   sigma_table_blob and class_index_blob and asserts parser/intermediate
   tensors change — extincts the F1 phantom-class without claiming full-frame
   output proof).
3. Archive grammar parser symmetry (split_z3g2_payload_bytes round-trip).

Per Catalog #240: dispatch_enabled=true requires implementation_complete; v2
stays research-only while `_full_main` raises NotImplementedError.

NO score claim until paired CUDA + CPU auth eval lands per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".

LOC budget: <= 350 LOC per HNeRV parity discipline L7 (v2 IS a bolt-on;
A1 weights frozen).
"""
from __future__ import annotations

from tac.substrates.z3_g1_entropy_coded_v2.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G1EntropyCodedV2Config,
    Z3G2EntropyCodedScorerClassGatingHead,
    compute_class_prior_cdf,
    g1_v2_per_pair_dominant_class_from_segnet_argmax,
)
from tac.substrates.z3_g1_entropy_coded_v2.archive import (
    Z3G2_HEADER_STRUCT,
    Z3G2_MAGIC,
    Z3G2_PER_DIM_AFFINE_LEN,
    Z3G2_VERSION,
    Z3G2EntropyCodedCompositionArchiveContract,
    Z3G2EntropyCodedSectionMeta,
    build_z3g2_composition_archive_contract,
    build_z3g2_payload_bytes,
    decode_z3g2_section,
    encode_z3g2_section,
    is_z3g2_payload,
    split_z3g2_payload_bytes,
)
from tac.substrates.z3_g1_entropy_coded_v2.inflate_consumer import (
    _class_conditional_arithmetic_decode,
    _unpack_class_prior_cdf,
    _unpack_sigma_table_entropy_coded,
    reconstruct_class_indices_and_sigma_table_from_z3g2_payload,
    select_inflate_device,
)
from tac.substrates.z3_g1_entropy_coded_v2.registered_substrate import (
    Z3_G1_ENTROPY_CODED_V2_CONTRACT,
)
from tac.substrates.z3_g1_entropy_coded_v2.score_aware_loss import (
    estimate_z3g2_section_overhead_bytes,
    g1_v2_residual_rate_bits_per_sample,
    z3_g1_v2_lagrangian,
)

__all__ = [
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "G1_NUM_SCORER_CLASSES",
    "Z3G2_HEADER_STRUCT",
    "Z3G2_MAGIC",
    "Z3G2_PER_DIM_AFFINE_LEN",
    "Z3G2_VERSION",
    "Z3_G1_ENTROPY_CODED_V2_CONTRACT",
    "Z3G1EntropyCodedV2Config",
    "Z3G2EntropyCodedCompositionArchiveContract",
    "Z3G2EntropyCodedScorerClassGatingHead",
    "Z3G2EntropyCodedSectionMeta",
    "_class_conditional_arithmetic_decode",
    "_unpack_class_prior_cdf",
    "_unpack_sigma_table_entropy_coded",
    "build_z3g2_composition_archive_contract",
    "build_z3g2_payload_bytes",
    "compute_class_prior_cdf",
    "decode_z3g2_section",
    "encode_z3g2_section",
    "estimate_z3g2_section_overhead_bytes",
    "g1_v2_per_pair_dominant_class_from_segnet_argmax",
    "g1_v2_residual_rate_bits_per_sample",
    "is_z3g2_payload",
    "reconstruct_class_indices_and_sigma_table_from_z3g2_payload",
    "select_inflate_device",
    "split_z3g2_payload_bytes",
    "z3_g1_v2_lagrangian",
]
