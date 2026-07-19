"""Default-OFF typed contract for the yhat-native generator measurement arm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

POLICY_SCHEMA = "yhat_native_generator_policy.v1"
POLICY_NAME = "YhatNativeGenerator"
LANE_ID = "lane_yhat_native_generator_20260719"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)


class YhatNativeGeneratorPolicyError(ValueError):
    """Raised when this measurement-only policy is given live authority."""


@dataclass(frozen=True)
class YhatNativeGeneratorPolicy:
    """Sealed, argv-inert policy for proving yhat-native equivalence only."""

    schema: str = POLICY_SCHEMA
    name: str = POLICY_NAME
    version: int = 1
    lane_id: str = LANE_ID
    research_only: bool = True
    activation_state: str = "BUILT_NOT_ACTIVATED_RECEIVER_ARCHIVE_GATES_OWED"
    camera_hw: tuple[int, int] = CAMERA_HW
    scorer_hw: tuple[int, int] = SCORER_HW
    channels: int = 3
    posenet_plane_ownership: str = "both_re_realized_rgb_frames"
    segnet_plane_ownership: str = "shared_resized_re_realized_frame1_rgb"
    frozen_scorer_order: tuple[str, ...] = (
        "bilinear_rgb_resize_874x1164_to_384x512",
        "PoseNet.rgb_to_yuv6_on_both_resized_rgb_planes",
        "SegNet.shared_resized_frame1_rgb_plane",
    )
    deterministic_expander: str = "numpy_fp32_required"
    realization: str = "exact_uint8_lattice_and_rational_numerator_verification"
    counted_description_boundary: str = "compact_counted_description_only"
    free_expander_boundary: str = "generic_deterministic_expander_only"
    live_trainer_argv: tuple[str, ...] = ()
    overrides: tuple[tuple[str, str], ...] = ()
    epochs_delta: int = 0
    trainer_activation: bool = False
    live_v10_integration: bool = False
    launch: bool = False
    paid_dispatch: bool = False
    score_claim: bool = False
    promotion: bool = False
    promotion_eligible: bool = False
    pointer_movement: bool = False
    pointer_moved: bool = False
    value_provenance: tuple[tuple[str, str], ...] = (
        ("geometry", "upstream/modules.py; src/tac/optimization/uint8_lattice_feasibility.py"),
        ("scorer_order", "upstream/modules.py"),
        ("lattice_realization", "src/tac/optimization/uint8_lattice_feasibility.py"),
        (
            "n24_equivalence_receipt",
            ".omx/research/yhat_native_generator_20260719_receipt.json; "
            "external revision2 aggregate sha256="
            "1ad1cf84672c696b46f62ca8586bb29d5c70f55de5803902b6c37666e5b85c0f",
        ),
    )
    completed_gates: tuple[str, ...] = ("n24_exact_rational_plane_native_f32_ulp_receipt_closed_20260719",)
    owed_gates: tuple[str, ...] = (
        "compact_description_receiver_closure",
        "n600_decode_time_custody_within_30_minutes",
        "exact_archive_parse_back",
        "contest_cpu_replay_separate_axis",
        "contest_cuda_replay_separate_axis",
    )

    def validate(self) -> None:
        sealed = type(self)()
        changed = [field.name for field in fields(self) if getattr(self, field.name) != getattr(sealed, field.name)]
        if changed:
            raise YhatNativeGeneratorPolicyError("yhat-native policy fields are sealed; changed=" + ", ".join(changed))

    def compile_contract(self, **requested_authority: bool) -> dict[str, Any]:
        """Return a JSON-safe sealed contract; any requested authority fails closed."""

        self.validate()
        unknown = set(requested_authority) - {
            "trainer_activation",
            "live_v10_integration",
            "launch",
            "paid_dispatch",
            "score_claim",
            "promotion",
            "promotion_eligible",
            "pointer_movement",
            "pointer_moved",
        }
        if unknown:
            raise YhatNativeGeneratorPolicyError(f"unknown authority request: {sorted(unknown)}")
        attempted = sorted(name for name, enabled in requested_authority.items() if enabled)
        if attempted:
            raise YhatNativeGeneratorPolicyError(
                "yhat-native policy cannot authorize "
                + ", ".join(attempted)
                + "; receiver/archive measurement gates remain owed"
            )
        return asdict(self)

    def compile(self, **requested_authority: bool) -> dict[str, Any]:
        """Compatibility spelling for consumers that compile typed policies."""

        return self.compile_contract(**requested_authority)


__all__ = [
    "CAMERA_HW",
    "LANE_ID",
    "POLICY_NAME",
    "POLICY_SCHEMA",
    "SCORER_HW",
    "YhatNativeGeneratorPolicy",
    "YhatNativeGeneratorPolicyError",
]
