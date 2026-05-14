# SPDX-License-Identifier: MIT
"""Research-source bindings for planning-only optimization math.

The objects here are not evidence of contest score movement. They bind paper
ideas to local variables, charged-byte contracts, and fail-closed blockers so
math-inspired planners cannot silently turn citations into dispatch authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = 1
TOOL_NAME = "tac.optimization.research_basis"


class ResearchBasisError(ValueError):
    """Raised when a research-basis request is malformed."""


REQUIRED_SOURCE_FIELDS = (
    "title",
    "authors",
    "year",
    "venue_or_status",
    "url",
    "lineage",
    "local_paradigms",
    "local_variables",
    "contest_terms",
    "charged_byte_contract",
    "hardening_blockers",
)


RESEARCH_SOURCES: dict[str, dict[str, Any]] = {
    "balle_e2e_2017": {
        "title": "End-to-end Optimized Image Compression",
        "authors": ["Johannes Balle", "Valero Laparra", "Eero P. Simoncelli"],
        "year": 2017,
        "venue_or_status": "ICLR 2017 / arXiv:1611.01704",
        "url": "https://arxiv.org/abs/1611.01704",
        "lineage": ["learned_compression", "variational_rate_distortion"],
        "local_paradigms": ["field_equations", "mdl", "joint_codec_stack"],
        "local_variables": [
            "variational_action_delta",
            "description_length_delta_bytes",
            "d_score_d_epsilon",
        ],
        "contest_terms": ["archive_bytes", "seg_dist", "pose_dist"],
        "charged_byte_contract": "Any learned decoder, latent, hyperparameter, or selector that affects inflate output must be inside archive.zip or fixed contest code.",
        "hardening_blockers": [
            "deterministic_quantization_contract",
            "byte_closed_archive_manifest",
            "exact_cuda_auth_eval",
        ],
    },
    "balle_hyperprior_2018": {
        "title": "Variational image compression with a scale hyperprior",
        "authors": [
            "Johannes Balle",
            "David Minnen",
            "Saurabh Singh",
            "Sung Jin Hwang",
            "Nick Johnston",
        ],
        "year": 2018,
        "venue_or_status": "ICLR 2018 / arXiv:1802.01436",
        "url": "https://arxiv.org/abs/1802.01436",
        "lineage": ["learned_compression", "hyperprior_entropy_model"],
        "local_paradigms": ["entropy_rate", "joint_codec_stack", "mdl"],
        "local_variables": [
            "conditional_entropy_floor_bits",
            "best_conditional_model_label",
            "model_byte_delta",
        ],
        "contest_terms": ["archive_bytes"],
        "charged_byte_contract": "Hyperprior side information is charged unless it is fixed contest runtime code.",
        "hardening_blockers": [
            "cross_platform_entropy_parameter_determinism",
            "runtime_parity_proof",
            "exact_cuda_auth_eval",
        ],
    },
    "fridrich_pq_wetpaper_2004": {
        "title": "Perturbed Quantization Steganography with Wet Paper Codes",
        "authors": ["Jessica Fridrich", "Miroslav Goljan", "David Soukal"],
        "year": 2004,
        "venue_or_status": "ACM Multimedia and Security Workshop 2004",
        "url": "https://doi.org/10.1145/1022431.1022435",
        "lineage": ["fridrich_lab", "wet_paper_codes", "sender_side_information"],
        "local_paradigms": ["categorical_masks", "field_equations", "entropy_rate"],
        "local_variables": [
            "atom_amplitude_epsilon",
            "selector_logits",
            "interaction_kernel",
            "conditional_groups",
        ],
        "contest_terms": ["archive_bytes", "seg_dist"],
        "charged_byte_contract": "Selection channels and side information must be reproducible from charged archive payload or fixed contest code.",
        "hardening_blockers": [
            "selection_channel_custody",
            "decoder_roundtrip_parity",
            "exact_stacked_archive_cuda_eval",
        ],
    },
    "fridrich_stc_2011": {
        "title": "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes",
        "authors": ["Tomas Filler", "Jan Judas", "Jessica Fridrich"],
        "year": 2011,
        "venue_or_status": "IEEE TIFS 2011",
        "url": "https://doi.org/10.1109/TIFS.2011.2134094",
        "lineage": ["fridrich_lab", "additive_distortion", "syndrome_trellis_codes"],
        "local_paradigms": ["field_equations", "meta_lagrangian", "pareto_kkt"],
        "local_variables": [
            "expected_total_score_delta",
            "expected_seg_dist_delta",
            "expected_pose_dist_delta",
            "byte_delta",
        ],
        "contest_terms": ["archive_bytes", "seg_dist", "pose_dist"],
        "charged_byte_contract": "Per-atom distortion costs are planning weights until encoded in a byte-closed archive and scored exactly.",
        "hardening_blockers": [
            "measured_atom_response",
            "nonadditive_interaction_review",
            "exact_cuda_auth_eval",
        ],
    },
    "yousfi_onehot_jpeg_2020": {
        "title": "An Intriguing Struggle of CNNs in JPEG Steganalysis and the OneHot Solution",
        "authors": ["Yassine Yousfi", "Jessica Fridrich"],
        "year": 2020,
        "venue_or_status": "IEEE Signal Processing Letters 2020",
        "url": "https://doi.org/10.1109/LSP.2020.2993959",
        "lineage": ["fridrich_lab", "yousfi", "categorical_dct_statistics"],
        "local_paradigms": ["categorical_masks", "openpilot_labels", "entropy_rate"],
        "local_variables": [
            "symbol_counts",
            "conditional_groups",
            "family_group",
            "pareto_scope",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Categorical labels or DCT-style one-hot codes must be either derived by fixed inflate code or carried as charged bytes.",
        "hardening_blockers": [
            "label_source_custody",
            "deterministic_decoder_contract",
            "exact_cuda_auth_eval",
        ],
    },
    "yousfi_cost_polarization_2023": {
        "title": "Cost polarization by dequantizing for JPEG steganography",
        "authors": ["Edgar Kaziakhmedov", "Yassine Yousfi", "Eli Dworetzky", "Jessica Fridrich"],
        "year": 2023,
        "venue_or_status": "Electronic Imaging 2023",
        "url": "https://doi.org/10.2352/EI.2023.35.4.MWSF-374",
        "lineage": ["fridrich_lab", "yousfi", "cost_polarization"],
        "local_paradigms": ["field_equations", "categorical_masks", "entropy_rate"],
        "local_variables": [
            "frechet_derivatives",
            "selector_logits",
            "confidence",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Any dequantized or reconstructed precover signal used for selection must be reproducible from contest-legal inputs.",
        "hardening_blockers": [
            "precover_proxy_nonpromotable_until_archive_consumed",
            "atom_response_calibration",
            "exact_cuda_auth_eval",
        ],
    },
    "lapose_2026": {
        "title": "LA-Pose: Latent Action Pretraining Meets Pose Estimation",
        "authors": [
            "Zhengqing Wang",
            "Saurabh Nair",
            "Prajwal Chidananda",
            "Pujith Kachana",
            "Samuel Li",
            "Matthew Brown",
            "Yasutaka Furukawa",
        ],
        "year": 2026,
        "venue_or_status": "CVPR 2026 / arXiv:2604.27448",
        "url": "https://arxiv.org/abs/2604.27448",
        "lineage": ["latent_action_pretraining", "driving_video_pose"],
        "local_paradigms": ["la_pose", "pose", "optimal_transport"],
        "local_variables": [
            "expected_pose_dist_delta",
            "pose_atom",
            "family_group",
            "d_pose_dist_d_epsilon",
        ],
        "contest_terms": ["pose_dist"],
        "charged_byte_contract": "Latent-action priors may rank compression-time atoms, but any scored pose stream or model output must be byte-closed and deterministic.",
        "hardening_blockers": [
            "pose_target_custody",
            "deterministic_init_pose_cli",
            "exact_cuda_auth_eval",
        ],
    },
    "foveated_telepresence_2025": {
        "title": "Foveated Compression for Immersive Telepresence Visualization",
        "authors": ["Max Schwarz", "Sven Behnke"],
        "year": 2025,
        "venue_or_status": "IEEE TELEPRESENCE 2025 / arXiv:2510.19848",
        "url": "https://arxiv.org/abs/2510.19848",
        "lineage": ["foveated_video_compression", "spatial_rate_allocation"],
        "local_paradigms": ["telescopic_foveation", "field_equations", "meta_lagrangian"],
        "local_variables": [
            "byte_allocator_lambdas",
            "expected_seg_dist_delta",
            "expected_pose_dist_delta",
            "interaction_assumptions",
        ],
        "contest_terms": ["archive_bytes", "seg_dist", "pose_dist"],
        "charged_byte_contract": "Foveation fields or gaze/center policies must be derivable from fixed contest code or charged archive payload.",
        "hardening_blockers": [
            "foveation_field_manifest",
            "pose_seg_component_gate",
            "exact_cuda_auth_eval",
        ],
    },
    "foveated_diffusion_2026": {
        "title": "Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation",
        "authors": ["Brian Chao", "Lior Yariv", "Howard Xiao", "Gordon Wetzstein"],
        "year": 2026,
        "venue_or_status": "arXiv:2603.23491",
        "url": "https://arxiv.org/abs/2603.23491",
        "lineage": ["foveated_generation", "mixed_resolution_tokens", "spatial_rate_allocation"],
        "local_paradigms": ["foveation", "telescopic_foveation", "field_equations", "meta_lagrangian"],
        "local_variables": [
            "foveation_token_density",
            "byte_allocator_lambdas",
            "expected_seg_dist_delta",
            "interaction_assumptions",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Mixed-resolution token or foveation-density policies are planning priors until their selectors and payload effects are byte-closed in archive.zip.",
        "hardening_blockers": [
            "mixed_resolution_token_runtime_contract",
            "archive_byte_accounting_required",
            "exact_cuda_auth_eval",
        ],
    },
    "nerv_2021": {
        "title": "NeRV: Neural Representations for Videos",
        "authors": ["Hao Chen", "Bo He", "Hanyu Wang", "Yixuan Ren", "Ser-Nam Lim", "Abhinav Shrivastava"],
        "year": 2021,
        "venue_or_status": "NeurIPS 2021 / arXiv:2110.13903",
        "url": "https://arxiv.org/abs/2110.13903",
        "lineage": ["neural_video_representation", "inr_video_compression"],
        "local_paradigms": ["alpha", "hnerv", "mask_payload"],
        "local_variables": [
            "decoder_entrypoint",
            "bytes_charged",
            "expected_seg_dist_delta",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Any NeRV decoder weights and mask latents used at inflate must be charged in the archive.",
        "hardening_blockers": [
            "decode_validation_required",
            "inflate_budget_required",
            "exact_cuda_auth_eval",
        ],
    },
    "hinerv_2023": {
        "title": "HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation",
        "authors": ["HiNeRV authors"],
        "year": 2023,
        "venue_or_status": "NeurIPS 2023 / arXiv:2306.09818",
        "url": "https://arxiv.org/abs/2306.09818",
        "lineage": ["neural_video_representation", "hierarchical_inr"],
        "local_paradigms": ["alpha", "hnerv", "mask_payload"],
        "local_variables": [
            "family_group",
            "expected_seg_dist_delta",
            "interaction_assumptions",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Hierarchical embeddings and side information are charged unless fixed in contest code.",
        "hardening_blockers": [
            "hierarchy_decode_roundtrip",
            "side_info_byte_accounting",
            "exact_cuda_auth_eval",
        ],
    },
    "geometric_visual_servo_ot_2026": {
        "title": "Geometric Visual Servo Via Optimal Transport",
        "authors": ["Ethan Canzini", "Simon Pope", "Ashutosh Tiwari"],
        "year": 2026,
        "venue_or_status": "Control Engineering Practice 2026 / arXiv:2506.02768v2",
        "url": "https://arxiv.org/abs/2506.02768",
        "lineage": ["optimal_transport", "se3_geodesic_flow", "visual_servoing"],
        "local_paradigms": ["pose", "optimal_transport", "field_equations", "meta_lagrangian"],
        "local_variables": [
            "geodesic_transport_cost",
            "expected_pose_dist_delta",
            "interaction_assumptions",
            "pareto_scope",
        ],
        "contest_terms": ["pose_dist", "archive_bytes"],
        "charged_byte_contract": "Geometric transport costs can rank pose/foveation atoms, but any pose stream, flow field, or selector consumed by inflate must be deterministic and charged.",
        "hardening_blockers": [
            "se3_transport_units_required",
            "runtime_pose_consumption_proof",
            "exact_cuda_auth_eval",
        ],
    },
    "telescope_2026": {
        "title": "Telescope: Long-Range Object Detection via Learnable Hyperbolic Foveation",
        "authors": ["Telescope authors"],
        "year": 2026,
        "venue_or_status": "arXiv:2604.06332",
        "url": "https://arxiv.org/abs/2604.06332",
        "lineage": ["hyperbolic_foveation", "spatial_rate_allocation"],
        "local_paradigms": ["telescopic_foveation", "field_equations", "mask_payload"],
        "local_variables": [
            "geometry_priors",
            "byte_allocator_lambdas",
            "expected_seg_dist_delta",
            "expected_pose_dist_delta",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Foveation parameters and inverse-warp policy must be deterministic and charged when used at inflate.",
        "hardening_blockers": [
            "foveation_archive_runtime_proof",
            "component_gate_required",
            "exact_cuda_auth_eval",
        ],
    },
    "lyra2_2026": {
        "title": "Lyra 2.0 long-horizon video synthesis mechanisms",
        "authors": ["NVIDIA Lyra 2.0 authors"],
        "year": 2026,
        "venue_or_status": "arXiv:2604.13036",
        "url": "https://arxiv.org/abs/2604.13036",
        "lineage": ["self_augmentation", "framepack", "canonical_coordinate_warping"],
        "local_paradigms": ["field_equations", "foveation", "pose", "mask_payload"],
        "local_variables": [
            "interaction_assumptions",
            "geometry_priors",
            "expected_pose_dist_delta",
            "expected_seg_dist_delta",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Self-augmentation may train selectors, but any coordinate warps or framepack policies consumed at inflate must be charged.",
        "hardening_blockers": [
            "roundtrip_aug_manifest",
            "coordinate_warp_decode_parity",
            "exact_cuda_auth_eval",
        ],
    },
    "mae_2021": {
        "title": "Masked Autoencoders Are Scalable Vision Learners",
        "authors": ["Kaiming He", "Xinlei Chen", "Saining Xie", "Yanghao Li", "Piotr Dollar", "Ross Girshick"],
        "year": 2021,
        "venue_or_status": "CVPR 2022 / arXiv:2111.06377",
        "url": "https://arxiv.org/abs/2111.06377",
        "lineage": ["masked_modeling", "asymmetric_encoder_decoder"],
        "local_paradigms": ["alpha", "mask_payload", "self_augmentation"],
        "local_variables": [
            "decoder_entrypoint",
            "bytes_charged",
            "expected_seg_dist_delta",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Sparse-visible-mask policies are charged payload decisions unless fixed and derivable at inflate.",
        "hardening_blockers": [
            "mask_sparsity_decode_parity",
            "pose_seg_component_gate",
            "exact_cuda_auth_eval",
        ],
    },
    "awq_2024": {
        "title": "AWQ: Activation-aware Weight Quantization",
        "authors": ["Ji Lin", "Jiaming Tang", "Haotian Tang", "Shang Yang", "Xingyu Dang", "Song Han"],
        "year": 2024,
        "venue_or_status": "MLSys 2024 / arXiv:2306.00978",
        "url": "https://arxiv.org/abs/2306.00978",
        "lineage": ["activation_aware_quantization", "mixed_precision"],
        "local_paradigms": ["sensitivity", "entropy_rate", "joint_codec_stack"],
        "local_variables": [
            "score_per_byte_marginal",
            "scorer_term_targeted",
            "byte_allocator_lambdas",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Protected-channel scales, masks, and quantization metadata must be charged or fixed.",
        "hardening_blockers": [
            "score_sensitivity_map_required",
            "deterministic_quantized_decode",
            "exact_cuda_auth_eval",
        ],
    },
    "hawq_v3_2020": {
        "title": "HAWQ-V3: Dyadic Neural Network Quantization",
        "authors": ["HAWQ-V3 authors"],
        "year": 2020,
        "venue_or_status": "arXiv:2011.10680",
        "url": "https://arxiv.org/abs/2011.10680",
        "lineage": ["hessian_aware_quantization", "mixed_precision"],
        "local_paradigms": ["sensitivity", "meta_lagrangian", "joint_codec_stack"],
        "local_variables": [
            "score_per_byte_marginal",
            "confidence",
            "pareto_objectives",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Mixed-precision decisions and scales must be deterministic archive payload or fixed code.",
        "hardening_blockers": [
            "hessian_or_fisher_custody",
            "byte_measured_quantized_stream",
            "exact_cuda_auth_eval",
        ],
    },
    "compressai": {
        "title": "CompressAI entropy-model and learned-compression reference implementation",
        "authors": ["InterDigital AI Lab contributors"],
        "year": 2020,
        "venue_or_status": "OSS reference / CompressAI documentation",
        "url": "https://interdigitalinc.github.io/CompressAI/",
        "lineage": ["learned_compression_oss", "entropy_models"],
        "local_paradigms": ["entropy_rate", "joint_codec_stack", "alpha"],
        "local_variables": [
            "conditional_entropy_floor_bits",
            "codec_kind",
            "bytes_charged",
        ],
        "contest_terms": ["archive_bytes"],
        "charged_byte_contract": "Reference entropy models may guide compression-time tooling; runtime dependencies and side info must be contest-legal and charged.",
        "hardening_blockers": [
            "dependency_policy_review",
            "deterministic_runtime_contract",
            "exact_cuda_auth_eval",
        ],
    },
    "constriction_ans": {
        "title": "constriction entropy coders",
        "authors": ["Bamler Lab contributors"],
        "year": 2022,
        "venue_or_status": "OSS entropy-coding library",
        "url": "https://github.com/bamler-lab/constriction",
        "lineage": ["ans_range_coding", "entropy_coder_implementation"],
        "local_paradigms": ["entropy_rate", "aq_huffman", "joint_codec_stack"],
        "local_variables": [
            "entropy_floor_bits",
            "gap_to_best_conditional_floor_bytes",
            "bytes_charged",
        ],
        "contest_terms": ["archive_bytes"],
        "charged_byte_contract": "Coder tables and compressed streams are charged; external libraries cannot be sidecar dependencies.",
        "hardening_blockers": [
            "rust_or_python_runtime_policy",
            "bit_exact_cross_platform_roundtrip",
            "exact_cuda_auth_eval",
        ],
    },
    "dworetzky_fridrich_detector_batch_2025": {
        "title": "Secure Payload Scaling in Detector-Informed Batch Steganography: The Mismatched Detectors Case",
        "authors": ["Eli Dworetzky", "Jessica Fridrich"],
        "year": 2025,
        "venue_or_status": "IH&MMSec 2025",
        "url": "https://doi.org/10.1145/3733102.3733134",
        "lineage": ["fridrich_lab", "detector_informed_batch_steganography", "mismatched_detectors"],
        "local_paradigms": ["meta_lagrangian", "bayesian_experimental_design", "field_equations"],
        "local_variables": [
            "expected_information_gain_nats",
            "confidence",
            "interaction_assumptions",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Detector-informed allocation can rank atoms, but the selected archive payload and any detector-derived tables must be charged or fixed.",
        "hardening_blockers": [
            "scorer_detector_mismatch_measurement",
            "candidate_archive_custody",
            "exact_cuda_auth_eval",
        ],
    },
    "kaziakhmedov_fridrich_lies_2025": {
        "title": "Effect of Acquisition Noise Outliers on Steganalysis",
        "authors": ["Edgar Kaziakhmedov", "Jessica Fridrich", "Patrick Bas"],
        "year": 2025,
        "venue_or_status": "IH&MMSec 2025",
        "url": "https://ws2.binghamton.edu/fridrich/Research/LIEs_final_v2_pdfa.pdf",
        "lineage": ["fridrich_lab", "acquisition_noise_outliers", "local_influence"],
        "local_paradigms": ["sensitivity", "field_equations", "categorical_masks"],
        "local_variables": [
            "score_per_byte_marginal",
            "hard_pair_support",
            "selector_logits",
        ],
        "contest_terms": ["seg_dist", "pose_dist"],
        "charged_byte_contract": "Outlier-aware selection maps are planning priors until their chosen edits are encoded in a charged archive.",
        "hardening_blockers": [
            "contest_video_noise_proxy_required",
            "component_sensitivity_custody",
            "exact_cuda_auth_eval",
        ],
    },
    "flavc_2025": {
        "title": "FLAVC: Learned Video Compression with Feature Level Attention",
        "authors": ["FLAVC authors"],
        "year": 2025,
        "venue_or_status": "CVPR 2025",
        "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_FLAVC_Learned_Video_Compression_with_Feature_Level_Attention_CVPR_2025_paper.html",
        "lineage": ["learned_video_compression", "feature_level_attention"],
        "local_paradigms": ["entropy_rate", "joint_codec_stack", "field_equations"],
        "local_variables": [
            "codec_kind",
            "conditional_entropy_floor_bits",
            "score_per_byte_marginal",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Feature attention latents, entropy tables, and decoder weights must be charged if consumed by inflate.",
        "hardening_blockers": [
            "feature_latent_byte_accounting",
            "decoder_runtime_budget",
            "exact_cuda_auth_eval",
        ],
    },
    "msnerv_2025": {
        "title": "MSNeRV: Neural Video Representation with Multi-Scale Feature Fusion",
        "authors": ["MSNeRV authors"],
        "year": 2025,
        "venue_or_status": "arXiv:2506.15276",
        "url": "https://arxiv.org/abs/2506.15276",
        "lineage": ["neural_video_representation", "multi_scale_feature_fusion"],
        "local_paradigms": ["hnerv", "alpha", "joint_codec_stack"],
        "local_variables": [
            "family_group",
            "codec_kind",
            "expected_seg_dist_delta",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Multi-scale feature tensors and fusion weights are charged archive payload unless fixed in contest runtime.",
        "hardening_blockers": [
            "multi_scale_decode_parity",
            "inflate_budget_required",
            "exact_cuda_auth_eval",
        ],
    },
    "balle_overfitted_wasserstein_2025": {
        "title": "Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion",
        "authors": ["Johannes Balle", "Fabian Versari", "Emilien Dupont", "Hyunjik Kim", "Matthias Bauer"],
        "year": 2025,
        "venue_or_status": "CVPR 2025",
        "url": "https://www.openaccess.thecvf.com/content/CVPR2025/html/Balle_Good_Cheap_and_Fast_Overfitted_Image_Compression_with_Wasserstein_Distortion_CVPR_2025_paper.html",
        "lineage": ["overfitted_compression", "wasserstein_distortion", "per_instance_codec"],
        "local_paradigms": ["mdl", "self_compressing_nn", "foveation"],
        "local_variables": [
            "description_length_delta_bytes",
            "expected_seg_dist_delta",
            "expected_pose_dist_delta",
        ],
        "contest_terms": ["seg_dist", "pose_dist", "archive_bytes"],
        "charged_byte_contract": "Overfitted model parameters are archive payload; perceptual Wasserstein losses are only priors until mapped to SegNet/PoseNet.",
        "hardening_blockers": [
            "contest_distortion_mapping_required",
            "charged_model_bytes_required",
            "exact_cuda_auth_eval",
        ],
    },
    "compression_as_adaptation_2026": {
        "title": "Compression as Adaptation",
        "authors": ["Compression as Adaptation authors"],
        "year": 2026,
        "venue_or_status": "arXiv:2603.07615",
        "url": "https://arxiv.org/abs/2603.07615",
        "lineage": ["adaptation_as_compression", "per_instance_model_adaptation"],
        "local_paradigms": ["self_compressing_nn", "mdl", "bayesian_experimental_design"],
        "local_variables": [
            "model_byte_delta",
            "data_byte_delta",
            "expected_information_gain_nats",
        ],
        "contest_terms": ["archive_bytes", "seg_dist", "pose_dist"],
        "charged_byte_contract": "Adapted parameters and any frozen model dependence must be contest-legal and charged or fixed.",
        "hardening_blockers": [
            "sidecar_model_forbidden",
            "runtime_budget_required",
            "exact_cuda_auth_eval",
        ],
    },
    "rdc_universal_2025": {
        "title": "Universal Rate-Distortion-Classification Representations",
        "authors": ["RDC authors"],
        "year": 2025,
        "venue_or_status": "arXiv:2504.09025",
        "url": "https://arxiv.org/abs/2504.09025",
        "lineage": ["rate_distortion_classification", "task_oriented_compression"],
        "local_paradigms": ["categorical_masks", "semantic_labels", "meta_lagrangian"],
        "local_variables": [
            "class_support",
            "expected_seg_dist_delta",
            "byte_delta",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Class representations, label maps, and codebooks must be charged or derived from fixed contest code.",
        "hardening_blockers": [
            "five_class_contract_instantiation",
            "label_codebook_custody",
            "exact_cuda_auth_eval",
        ],
    },
    "rdc_bernoulli_2026": {
        "title": "RDC Representation Theory for Bernoulli Sources",
        "authors": ["RDC Bernoulli authors"],
        "year": 2026,
        "venue_or_status": "arXiv:2601.11919",
        "url": "https://arxiv.org/abs/2601.11919",
        "lineage": ["rate_distortion_classification", "bernoulli_sources"],
        "local_paradigms": ["categorical_masks", "entropy_rate", "field_equations"],
        "local_variables": [
            "symbol_counts",
            "conditional_groups",
            "entropy_floor_bits",
        ],
        "contest_terms": ["seg_dist", "archive_bytes"],
        "charged_byte_contract": "Bernoulli/categorical abstractions must be instantiated on contest class streams with charged codebooks.",
        "hardening_blockers": [
            "contest_class_distribution_custody",
            "charged_codebook_required",
            "exact_cuda_auth_eval",
        ],
    },
}

DEFAULT_RESEARCH_BASIS_IDS = [
    "fridrich_stc_2011",
    "fridrich_pq_wetpaper_2004",
    "balle_e2e_2017",
    "balle_hyperprior_2018",
]

FAMILY_RESEARCH_BASIS_IDS: dict[str, list[str]] = {
    "alpha": ["fridrich_stc_2011", "fridrich_pq_wetpaper_2004", "yousfi_onehot_jpeg_2020"],
    "aq_huffman": ["constriction_ans", "balle_hyperprior_2018"],
    "bayesian_experimental_design": ["dworetzky_fridrich_detector_batch_2025", "compression_as_adaptation_2026"],
    "categorical": ["rdc_universal_2025", "rdc_bernoulli_2026", "yousfi_onehot_jpeg_2020", "yousfi_cost_polarization_2023"],
    "entropy": ["flavc_2025", "balle_hyperprior_2018", "compressai", "constriction_ans", "yousfi_onehot_jpeg_2020"],
    "foveation": ["telescope_2026", "foveated_diffusion_2026", "foveated_telepresence_2025", "lyra2_2026"],
    "gamma": ["balle_e2e_2017", "balle_hyperprior_2018", "compressai"],
    "hnerv": ["msnerv_2025", "nerv_2021", "hinerv_2023", "balle_hyperprior_2018", "fridrich_stc_2011"],
    "lapose": ["lapose_2026", "geometric_visual_servo_ot_2026"],
    "mask_payload": ["nerv_2021", "hinerv_2023", "mae_2021", "fridrich_pq_wetpaper_2004"],
    "meta_lagrangian": ["dworetzky_fridrich_detector_batch_2025", "geometric_visual_servo_ot_2026", "fridrich_stc_2011", "hawq_v3_2020", "awq_2024"],
    "mdl": ["balle_overfitted_wasserstein_2025", "compression_as_adaptation_2026"],
    "optimal_transport": ["geometric_visual_servo_ot_2026", "balle_overfitted_wasserstein_2025"],
    "pose": ["lapose_2026", "geometric_visual_servo_ot_2026", "foveated_telepresence_2025"],
    "self_augmentation": ["mae_2021", "lyra2_2026"],
    "semantic_labels": ["rdc_universal_2025", "yousfi_onehot_jpeg_2020"],
    "self_compressing_nn": ["compression_as_adaptation_2026", "balle_overfitted_wasserstein_2025"],
    "sensitivity": ["kaziakhmedov_fridrich_lies_2025", "awq_2024", "hawq_v3_2020"],
    "telescopic_foveation": ["telescope_2026", "foveated_diffusion_2026", "foveated_telepresence_2025"],
    "wavelet": ["fridrich_stc_2011", "fridrich_pq_wetpaper_2004"],
}


def research_basis_ids_for_family(*family_values: str) -> list[str]:
    """Return deterministic research-basis ids for one or more family labels."""

    out: list[str] = []
    for family in family_values:
        key = str(family or "").lower()
        for basis_id in FAMILY_RESEARCH_BASIS_IDS.get(key, []):
            if basis_id not in out:
                out.append(basis_id)
    if not out:
        out.extend(DEFAULT_RESEARCH_BASIS_IDS)
    return out


def research_basis_manifest(ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Return a deterministic, score-neutral manifest for research sources."""

    resolved_ids = list(ids or DEFAULT_RESEARCH_BASIS_IDS)
    if not resolved_ids:
        raise ResearchBasisError("research basis ids must be nonempty")
    seen: set[str] = set()
    sources = []
    for raw_id in resolved_ids:
        basis_id = str(raw_id or "").strip()
        if not basis_id:
            raise ResearchBasisError("research basis id must be nonempty")
        if basis_id in seen:
            continue
        if basis_id not in RESEARCH_SOURCES:
            raise ResearchBasisError(f"unknown research basis id: {basis_id}")
        seen.add(basis_id)
        source = _validated_source(basis_id, RESEARCH_SOURCES[basis_id])
        source["basis_id"] = basis_id
        sources.append(source)
    sources.sort(key=lambda source: (int(source["year"]), str(source["basis_id"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "source_count": len(sources),
        "sources": sources,
        "contract": (
            "Research priors may shape planning, features, and atom ranking, "
            "but they cannot promote, rank, or claim contest score without "
            "byte-closed exact CUDA archive evidence."
        ),
        "global_hardening_blockers": [
            "paper_claim_not_score_evidence",
            "charged_byte_contract_required",
            "deterministic_reproducibility_required",
            "exact_cuda_auth_eval_required",
        ],
    }


def _validated_source(basis_id: str, source: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in source]
    if missing:
        raise ResearchBasisError(
            f"{basis_id}: research source missing required fields: {', '.join(missing)}"
        )
    out = dict(source)
    if not str(out["title"]):
        raise ResearchBasisError(f"{basis_id}: title must be nonempty")
    if not isinstance(out["authors"], list) or not out["authors"]:
        raise ResearchBasisError(f"{basis_id}: authors must be a nonempty list")
    year = out["year"]
    if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
        raise ResearchBasisError(f"{basis_id}: year must be a valid integer")
    if not str(out["url"]).startswith("https://"):
        raise ResearchBasisError(f"{basis_id}: url must be https")
    for field in ("lineage", "local_paradigms", "local_variables", "contest_terms", "hardening_blockers"):
        if not isinstance(out[field], list) or not out[field]:
            raise ResearchBasisError(f"{basis_id}: {field} must be a nonempty list")
    if not str(out["charged_byte_contract"]):
        raise ResearchBasisError(f"{basis_id}: charged_byte_contract must be nonempty")
    return out


__all__ = [
    "DEFAULT_RESEARCH_BASIS_IDS",
    "FAMILY_RESEARCH_BASIS_IDS",
    "REQUIRED_SOURCE_FIELDS",
    "RESEARCH_SOURCES",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "ResearchBasisError",
    "research_basis_ids_for_family",
    "research_basis_manifest",
]
