"""PR95 hnerv_muon brownfield substrate — LoRA/DoRA adapters on frozen base.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 7
(substrate engineering excursion) + Catalog #124 (representation-lane
archive grammar at design time).

The base is PR95's published archive (178,309 bytes, SHA-256
4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4). We do not
retrain it (faithful retrain is $700+ per F1 forensic memo); we ADAPT it via
LoRA/DoRA low-rank adapters on selected layers.

Archive grammar: PR95 0.bin bytes UNCHANGED + LoRA TRAILER appended (see
`archive.py` for byte map).

Lane: `lane_pr95_artifact_lora_dora_surgery_20260513`.
Deconstruction memo: `.omx/research/pr95_artifact_deconstruction_20260513.md`.
F1 forensic anchor: `.omx/research/pr95_8stage_curriculum_forensic_20260513.md`.
"""

from .architecture import (
    DEFAULT_TIER_A_TARGETS,
    DEFAULT_TIER_B_TARGETS,
    DEFAULT_TIER_C_TARGETS,
    AdapterConfig,
    DoRAAdapter,
    LoRAAdapter,
    PR95LoRADoRADecoder,
)
from .archive import (
    LORA_TRAILER_MAGIC,
    LORA_TRAILER_VERSION,
    build_lora_archive,
    decode_lora_trailer,
    encode_lora_trailer,
    parse_lora_archive,
)
from .budget import (
    AdapterBreakEven,
    AdapterLayerBudget,
    adapter_break_even,
    adapter_raw_trailer_bytes,
    adapter_trainable_params,
    exact_pose_reduction_for_score_delta,
    rate_score_penalty_for_bytes,
    tier_c_layer_budgets,
    tier_c_raw_trailer_bytes,
    tier_c_trainable_params,
)

__all__ = [
    "DEFAULT_TIER_A_TARGETS",
    "DEFAULT_TIER_B_TARGETS",
    "DEFAULT_TIER_C_TARGETS",
    "LORA_TRAILER_MAGIC",
    "LORA_TRAILER_VERSION",
    "AdapterBreakEven",
    "AdapterConfig",
    "AdapterLayerBudget",
    "DoRAAdapter",
    "LoRAAdapter",
    "PR95LoRADoRADecoder",
    "adapter_break_even",
    "adapter_raw_trailer_bytes",
    "adapter_trainable_params",
    "build_lora_archive",
    "decode_lora_trailer",
    "encode_lora_trailer",
    "exact_pose_reduction_for_score_delta",
    "parse_lora_archive",
    "rate_score_penalty_for_bytes",
    "tier_c_layer_budgets",
    "tier_c_raw_trailer_bytes",
    "tier_c_trainable_params",
]
