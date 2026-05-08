"""Typed runtime contract for cross-paradigm pipeline flags.

This module is intentionally deterministic and side-effect free: it inspects
operator-visible flags, checks required artifact paths, and reports warnings or
blockers. It never loads scorers, touches CUDA, dispatches jobs, or creates a
score claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SCHEMA = "cross_paradigm_runtime_contract_v1"

WIRED_ALPHA_MASK_CODECS = frozenset({"av1_monochrome", "argmax_rle"})
ALPHA_STUB_MASK_CODECS = frozenset({"nerv", "wavelet", "vqvae", "grayscale_lut"})

JsonScalar = str | int | float | bool | None

__all__ = [
    "ALPHA_STUB_MASK_CODECS",
    "SCHEMA",
    "WIRED_ALPHA_MASK_CODECS",
    "ArtifactRequirement",
    "CrossParadigmRuntimeContract",
    "RuntimeLaneContract",
    "build_cross_paradigm_runtime_contract",
    "raise_for_cross_paradigm_blockers",
]


@dataclass(frozen=True)
class ArtifactRequirement:
    field: str
    path: str
    required: bool
    exists: bool
    blocker: str

    def to_dict(self) -> dict[str, JsonScalar]:
        return {
            "field": self.field,
            "path": self.path,
            "required": self.required,
            "exists": self.exists,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class RuntimeLaneContract:
    paradigm: str
    lane_id: str
    enabled: bool
    status: str
    config: tuple[tuple[str, JsonScalar], ...] = ()
    artifacts: tuple[ArtifactRequirement, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    ready_for_runtime_dispatch: bool = False
    ready_for_exact_eval_dispatch: bool = False
    dispatch_attempted: bool = False
    score_claim_created: bool = False
    cuda_claim_created: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "lane_id": self.lane_id,
            "enabled": self.enabled,
            "status": self.status,
            "config": {key: value for key, value in self.config},
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "ready_for_runtime_dispatch": self.ready_for_runtime_dispatch,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "dispatch_attempted": self.dispatch_attempted,
            "score_claim_created": self.score_claim_created,
            "cuda_claim_created": self.cuda_claim_created,
        }


@dataclass(frozen=True)
class CrossParadigmRuntimeContract:
    lanes: tuple[RuntimeLaneContract, ...]

    @property
    def any_cross_paradigm_opt_in(self) -> bool:
        return any(lane.enabled for lane in self.lanes)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(warning for lane in self.lanes for warning in lane.warnings)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(blocker for lane in self.lanes for blocker in lane.blockers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "any_cross_paradigm_opt_in": self.any_cross_paradigm_opt_in,
            "ready_for_exact_eval_dispatch": False,
            "exact_eval_dispatch_attempted": False,
            "dispatch_attempted": False,
            "score_claim_created": False,
            "cuda_claim_created": False,
            "score_axis_contract": "dual_axis_cuda_and_cpu",
            "score_axis_contract_notes": (
                "Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA': any "
                "lane that emits a score band must distinguish CUDA-axis vs CPU-"
                "axis predictions. Both axes are required before exact-eval "
                "dispatch can be approved."
            ),
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }


def build_cross_paradigm_runtime_contract(
    cfg: Any,
    *,
    path_exists: Callable[[Path], bool] = Path.exists,
) -> CrossParadigmRuntimeContract:
    lanes: list[RuntimeLaneContract] = []

    mask_codec = str(_get(cfg, "mask_codec", "av1_monochrome"))
    if mask_codec != "av1_monochrome":
        lanes.append(_alpha_mask_codec_contract(mask_codec))

    use_sensitivity_weighted = bool(_get(cfg, "use_sensitivity_weighted", False))
    if use_sensitivity_weighted:
        lanes.append(
            _beta_owv3_contract(
                sensitivity_map_path=str(_get(cfg, "sensitivity_map_path", "")),
                bit_budget_ratio=float(_get(cfg, "owv3_bit_budget_ratio", 0.7)),
                protect_threshold=float(_get(cfg, "owv3_protect_threshold", 1e-3)),
                path_exists=path_exists,
            )
        )

    weight_compression = str(_get(cfg, "weight_compression", "fp4"))
    if weight_compression == "nwcs_sensitivity":
        lanes.append(
            _beta_nwcs_contract(
                weight_codec_path=str(_get(cfg, "weight_codec_path", "")),
                sensitivity_map_path=str(_get(cfg, "sensitivity_map_path", "")),
                path_exists=path_exists,
            )
        )

    if bool(_get(cfg, "use_joint_codec_stack", False)):
        lanes.append(
            _gamma_jcsp_contract(
                score_marginals_path=str(_get(cfg, "jcsp_score_marginals_path", "")),
                path_exists=path_exists,
            )
        )

    if bool(_get(cfg, "use_joint_scorer_aware", False)):
        lanes.append(
            _delta_joint_training_contract(
                config_path=str(_get(cfg, "joint_training_config_path", "")),
                path_exists=path_exists,
            )
        )

    if bool(_get(cfg, "use_raft_init", False)):
        lanes.append(_lapose_raft_contract())

    if bool(_get(cfg, "use_riemannian_tto", False)):
        lanes.append(_lapose_riemannian_contract())

    if bool(_get(cfg, "use_learnable_entropy", False)):
        lanes.append(_epsilon_entropy_contract())

    if bool(_get(cfg, "use_full_renderer_self_compress", False)):
        lanes.append(_zeta_self_compress_contract())

    return CrossParadigmRuntimeContract(lanes=tuple(lanes))


def raise_for_cross_paradigm_blockers(
    contract: CrossParadigmRuntimeContract,
) -> None:
    if not contract.blockers:
        return
    joined = "; ".join(contract.blockers)
    raise RuntimeError(
        "Cross-paradigm runtime contract blocked a non-dispatchable "
        f"configuration: {joined}"
    )


def _alpha_mask_codec_contract(mask_codec: str) -> RuntimeLaneContract:
    config = (("mask_codec", mask_codec),)
    if mask_codec in WIRED_ALPHA_MASK_CODECS:
        return RuntimeLaneContract(
            paradigm="alpha",
            lane_id=f"alpha_mask_codec:{mask_codec}",
            enabled=True,
            status="wired_runtime_path",
            config=config,
            warnings=(
                "alpha_mask_codec_runtime_wired_without_score_or_cuda_claim",
            ),
            ready_for_runtime_dispatch=True,
        )
    if mask_codec in ALPHA_STUB_MASK_CODECS:
        blocker = f"alpha_mask_codec_registered_but_not_wired:{mask_codec}"
        return RuntimeLaneContract(
            paradigm="alpha",
            lane_id=f"alpha_mask_codec:{mask_codec}",
            enabled=True,
            status="registered_but_not_wired",
            config=config,
            blockers=(blocker,),
        )
    blocker = f"alpha_mask_codec_unknown:{mask_codec}"
    return RuntimeLaneContract(
        paradigm="alpha",
        lane_id=f"alpha_mask_codec:{mask_codec}",
        enabled=True,
        status="unknown_mask_codec",
        config=config,
        blockers=(blocker,),
    )


def _beta_owv3_contract(
    *,
    sensitivity_map_path: str,
    bit_budget_ratio: float,
    protect_threshold: float,
    path_exists: Callable[[Path], bool],
) -> RuntimeLaneContract:
    artifact = _artifact_requirement(
        field="sensitivity_map_path",
        path=sensitivity_map_path,
        blocker="beta_sensitivity_map_artifact_missing",
        path_exists=path_exists,
    )
    blockers = _missing_artifact_blockers(artifact)
    return RuntimeLaneContract(
        paradigm="beta",
        lane_id="beta_owv3_sensitivity_weighted",
        enabled=True,
        status="wired_runtime_path" if not blockers else "blocked_missing_artifact",
        config=(
            ("use_sensitivity_weighted", True),
            ("owv3_bit_budget_ratio", bit_budget_ratio),
            ("owv3_protect_threshold", protect_threshold),
        ),
        artifacts=(artifact,),
        warnings=(
            "beta_owv3_runtime_wired_without_score_or_cuda_claim",
        ) if not blockers else (),
        blockers=blockers,
        ready_for_runtime_dispatch=not blockers,
    )


def _beta_nwcs_contract(
    *,
    weight_codec_path: str,
    sensitivity_map_path: str,
    path_exists: Callable[[Path], bool],
) -> RuntimeLaneContract:
    artifacts = (
        _artifact_requirement(
            field="weight_codec_path",
            path=weight_codec_path,
            blocker="beta_nwcs_weight_codec_artifact_missing",
            path_exists=path_exists,
        ),
        _artifact_requirement(
            field="sensitivity_map_path",
            path=sensitivity_map_path,
            blocker="beta_nwcs_sensitivity_map_artifact_missing",
            path_exists=path_exists,
        ),
    )
    blockers = _missing_artifact_blockers(*artifacts) + (
        "beta_nwcs_encoding_loop_not_wired",
    )
    return RuntimeLaneContract(
        paradigm="beta",
        lane_id="beta_nwcs_sensitivity_weighted",
        enabled=True,
        status="registered_but_not_wired",
        config=(("weight_compression", "nwcs_sensitivity"),),
        artifacts=artifacts,
        blockers=blockers,
    )


def _gamma_jcsp_contract(
    *,
    score_marginals_path: str,
    path_exists: Callable[[Path], bool],
) -> RuntimeLaneContract:
    artifact = _artifact_requirement(
        field="jcsp_score_marginals_path",
        path=score_marginals_path,
        blocker="gamma_jcsp_score_marginals_artifact_missing",
        path_exists=path_exists,
    )
    blockers = _missing_artifact_blockers(artifact) + (
        "gamma_jcsp_runtime_parity_not_proven",
        "gamma_jcsp_exact_eval_dispatch_requirements_missing",
    )
    return RuntimeLaneContract(
        paradigm="gamma",
        lane_id="gamma_joint_codec_stack",
        enabled=True,
        status="non_dispatchable_runtime_member",
        config=(("use_joint_codec_stack", True),),
        artifacts=(artifact,),
        blockers=blockers,
    )


def _delta_joint_training_contract(
    *,
    config_path: str,
    path_exists: Callable[[Path], bool],
) -> RuntimeLaneContract:
    artifact = _artifact_requirement(
        field="joint_training_config_path",
        path=config_path,
        blocker="delta_joint_training_config_artifact_missing",
        path_exists=path_exists,
    )
    blockers = _missing_artifact_blockers(artifact) + (
        "delta_joint_scorer_aware_dispatch_not_wired",
    )
    return RuntimeLaneContract(
        paradigm="delta",
        lane_id="delta_joint_scorer_aware_training",
        enabled=True,
        status="registered_but_not_wired",
        config=(("use_joint_scorer_aware", True),),
        artifacts=(artifact,),
        blockers=blockers,
    )


def _lapose_raft_contract() -> RuntimeLaneContract:
    return RuntimeLaneContract(
        paradigm="lapose",
        lane_id="lapose_raft_init",
        enabled=True,
        status="registered_but_not_wired",
        config=(("use_raft_init", True),),
        blockers=("lapose_raft_init_dispatch_not_wired",),
    )


def _lapose_riemannian_contract() -> RuntimeLaneContract:
    return RuntimeLaneContract(
        paradigm="lapose",
        lane_id="lapose_riemannian_tto",
        enabled=True,
        status="wired_runtime_path",
        config=(("use_riemannian_tto", True),),
        warnings=(
            "lapose_riemannian_runtime_wired_without_score_or_cuda_claim",
        ),
        ready_for_runtime_dispatch=True,
    )


def _epsilon_entropy_contract() -> RuntimeLaneContract:
    return RuntimeLaneContract(
        paradigm="epsilon",
        lane_id="epsilon_learnable_entropy",
        enabled=True,
        status="registered_but_not_wired",
        config=(("use_learnable_entropy", True),),
        blockers=("epsilon_learnable_entropy_dispatch_not_wired",),
    )


def _zeta_self_compress_contract() -> RuntimeLaneContract:
    return RuntimeLaneContract(
        paradigm="zeta",
        lane_id="zeta_full_renderer_self_compress",
        enabled=True,
        status="registered_but_not_wired",
        config=(("use_full_renderer_self_compress", True),),
        blockers=("zeta_full_renderer_self_compress_dispatch_not_wired",),
    )


def _artifact_requirement(
    *,
    field: str,
    path: str,
    blocker: str,
    path_exists: Callable[[Path], bool],
) -> ArtifactRequirement:
    exists = bool(path) and path_exists(Path(path))
    return ArtifactRequirement(
        field=field,
        path=path,
        required=True,
        exists=exists,
        blocker=blocker,
    )


def _missing_artifact_blockers(
    *artifacts: ArtifactRequirement,
) -> tuple[str, ...]:
    return tuple(
        artifact.blocker for artifact in artifacts
        if artifact.required and not artifact.exists
    )


def _get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)
