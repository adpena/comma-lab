#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare or execute the DDM PT1 continuous-paint ceiling meter.

The checked-in config is research-only and has ``execution_allowed=false``.
Consequently this delegated landing may emit only a SHA-bound
``PREPARED_NOT_EXECUTED`` receipt.  MAIN may review a successor config that
changes the authority bit and invoke ``--execute``; the execution path is
batch-checkpointed and refuses any window other than exact n600.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for value in (str(SRC_ROOT), str(REPO_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_continuous_paint_ceiling import (  # noqa: E402
    ContinuousPaintError,
    apply_global_channel_statistics,
    decompose_mechanisms,
    encode_global_channel_statistics,
    encode_stratum_spectrum_coefficients,
    measure_fitted_geometry_sdwl1,
    render_analytic_coverage_blend,
    render_hard_camera_placement,
    render_stratum_spectrum_match,
    resample_fields_at_pixel_centres,
    scorer_native_divergence_rows,
    sha256_array,
    signed_distance_fields,
    solve_stratum_spectrum_coefficients,
    split_curve_provenance,
    stage_transition,
    stratum_spectrum_components,
    stratum_spectrum_normal_equations,
    target_boundary_band,
)
from tac.optimization.ddm_description_vocabulary import (  # noqa: E402
    decode_boundary_worldsheet_spline,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.through_r.resolution_chain import (  # noqa: E402
    CAMERA_HW,
    SEG_HW,
    render_grid_to_camera_uint8,
)

EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
SCHEMA = "ddm_pt1_continuous_paint_ceiling.v1"
PREPARED_SCHEMA = "ddm_pt1_continuous_paint_ceiling_prepared_receipt.v1"
MEASURED_SCHEMA = "ddm_pt1_continuous_paint_ceiling_measurement_receipt.v1"
RATE_DUAL = 25 / 37_545_489


class PT1Config(BaseModel):
    """Strict, SHA-bound local measurement contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        populate_by_name=True,
    )

    schema_: Literal["DDMPT1ContinuousPaintCeilingConfigV1"] = Field(
        default="DDMPT1ContinuousPaintCeilingConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr
    seed: Literal[0] = 0
    pair_count: Literal[600] = 600
    batch_size: Literal[16] = 16
    target_cache_path: StrictStr
    target_cache_bytes: StrictInt = Field(gt=0)
    target_cache_sha256: StrictStr
    e2_receipt_path: StrictStr
    e2_receipt_sha256: StrictStr
    dv1_spline_path: StrictStr
    dv1_spline_sha256: StrictStr
    sdwl1_receipt_path: StrictStr
    sdwl1_receipt_sha256: StrictStr
    upstream_root: StrictStr
    scorer_modules_sha256: StrictStr
    segnet_weights_sha256: StrictStr
    palette_rgb_u8: tuple[
        tuple[StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt],
    ]
    analytic_softness: StrictFloat = Field(gt=0.0)
    boundary_dilation: StrictInt = Field(ge=0, le=4)
    scorer_native_layers: tuple[StrictStr, ...]
    amplitude_statistics_payload_bytes: Literal[30] = 30
    spectrum_payload_bytes: Literal[186] = 186
    survival_wall_receipt_path: StrictStr | None = None
    survival_wall_receipt_sha256: StrictStr | None = None
    expected_e2_paint_errors: Literal[3349482] = 3349482
    total_seg_sites: Literal[117964800] = 117964800
    box_floor_d_seg: Literal[0.0142] = 0.0142
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    execution_allowed: StrictBool = False
    pose_secondary_enabled: Literal[False] = False

    @model_validator(mode="after")
    def _validate_contract(self) -> PT1Config:
        for field in (
            "target_cache_sha256",
            "e2_receipt_sha256",
            "dv1_spline_sha256",
            "sdwl1_receipt_sha256",
            "scorer_modules_sha256",
            "segnet_weights_sha256",
        ):
            value = getattr(self, field)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field} must be lowercase SHA-256")
        wall_fields = (
            self.survival_wall_receipt_path,
            self.survival_wall_receipt_sha256,
        )
        if sum(value is not None for value in wall_fields) == 1:
            raise ValueError("survival-wall receipt path and SHA must be paired")
        if self.survival_wall_receipt_sha256 is not None:
            value = self.survival_wall_receipt_sha256
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    "survival_wall_receipt_sha256 must be lowercase SHA-256"
                )
        if not Path(self.target_cache_path).is_absolute():
            raise ValueError("target_cache_path must be absolute")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute")
        palette = np.asarray(self.palette_rgb_u8)
        if np.any((palette < 0) | (palette > 255)):
            raise ValueError("palette_rgb_u8 must contain uint8 values")
        if self.pair_count * SEG_HW[0] * SEG_HW[1] != self.total_seg_sites:
            raise ValueError("total_seg_sites does not close to exact n600 geometry")
        if not self.scorer_native_layers or len(set(self.scorer_native_layers)) != len(
            self.scorer_native_layers
        ):
            raise ValueError("scorer_native_layers must be a nonempty unique tuple")
        return self

    def hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ContinuousPaintError(f"regular non-symlink input is required: {path}")
    return path.read_bytes()


def _bound_bytes(path: Path, expected_sha256: str, name: str) -> bytes:
    payload = _read(path)
    observed = _sha256(payload)
    if observed != expected_sha256:
        raise ContinuousPaintError(
            f"{name} SHA-256 mismatch: {observed} != {expected_sha256}"
        )
    return payload


def _bound_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    payload = _bound_bytes(path, expected_sha256, name)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContinuousPaintError(f"{name} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ContinuousPaintError(f"{name} must be one JSON object")
    return value


def _independent_survival_wall(config: PT1Config) -> tuple[float, dict[str, str]]:
    """Load an independently measured hard-placement survival wall."""

    if (
        config.survival_wall_receipt_path is None
        or config.survival_wall_receipt_sha256 is None
    ):
        raise ContinuousPaintError(
            "execution requires an independent SHA-bound survival-wall receipt"
        )
    row = _bound_json(
        REPO_ROOT / config.survival_wall_receipt_path,
        config.survival_wall_receipt_sha256,
        "survival-wall receipt",
    )
    raw = row.get("measured_survival_wall_fraction")
    if raw is None and isinstance(row.get("mechanism_falsifier"), dict):
        raw = row["mechanism_falsifier"].get("measured_survival_wall_fraction")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ContinuousPaintError(
            "survival-wall receipt lacks measured_survival_wall_fraction"
        )
    wall = float(raw)
    if not 0.0 <= wall <= 1.0:
        raise ContinuousPaintError("independent survival wall is outside [0,1]")
    return wall, {
        "path": config.survival_wall_receipt_path,
        "sha256": config.survival_wall_receipt_sha256,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = rfc8785_canonicalize(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ContinuousPaintError(f"immutable output differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _source_binding() -> dict[str, Any]:
    paths = (
        Path(__file__),
        REPO_ROOT / "src/tac/optimization/ddm_continuous_paint_ceiling.py",
        REPO_ROOT / "src/tac/boundary_math/aa_sdf_observation_render.py",
        REPO_ROOT / "src/tac/boundary_math/phase_primitives.py",
        REPO_ROOT / "src/tac/boundary_math/texture_trunk.py",
        REPO_ROOT / "src/tac/optimization/ddm_dv2_sdwl1.py",
        REPO_ROOT / "src/tac/through_r/resolution_chain.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(_read(path)),
        }
        for path in paths
    }


def _four_clause_audit() -> dict[str, Any]:
    return {
        "schema": "ddm_four_clause_rate_doctrine.v1",
        "first_rung": True,
        "single_owner_facts": [
            {
                "fact": "already-counted decoded curve geometry",
                "owner": "DV1 boundary_worldsheet_spline",
                "dimension_home": "pair x scorer-column upper-Road separatrix",
                "first_rung": True,
            },
            {
                "fact": "target-derived boundary geometry absent from DV1",
                "owner": "fresh fitted SDWL1 sentence",
                "dimension_home": "pair x class partition and separatrix facts",
                "first_rung": True,
            },
            {
                "fact": "global RGB first and second moments",
                "owner": "PT1 global amplitude-statistics payload",
                "dimension_home": "one global set x RGB x mean/variance",
                "first_rung": True,
            },
            {
                "fact": "per-stratum through-R-passband texture coefficients",
                "owner": "PT1 stratum-spectrum payload",
                "dimension_home": "class x RGB x period group",
                "first_rung": True,
            },
        ],
        "streams": [
            {
                "member": "DV1 reused curves",
                "first_rung": True,
                "candidate_admissible": False,
                "non_redundancy": {
                    "conditioned_on": [],
                    "single_owner_facts": ["already-counted upper-Road separatrix"],
                },
                "audit_triple": {
                    "three_layer_decomposition": {
                        "descriptive_form": "continuous upper-Road separatrix",
                        "inherently_compact_dofs": "existing spline knots",
                        "coder": "existing DV1 selected joint-ground stream",
                    },
                    "scorer_visibility": {
                        "authority_surface": "SegNet frame1 after hard camera placement and R",
                        "status": "PENDING_N600_EXECUTION",
                    },
                    "sensitivity_priced_tolerance": {
                        "metric": "rank4 |margin|/||Delta w|| plus realized transition",
                        "status": "PENDING_N600_EXECUTION",
                    },
                },
                "verdict_scope": "reused DV1 curve sites only",
            },
            {
                "member": "fresh fitted SDWL1 geometry",
                "first_rung": True,
                "candidate_admissible": False,
                "non_redundancy": {
                    "conditioned_on": ["DV1 reused-curve site mask"],
                    "single_owner_facts": ["all remaining target-boundary sites"],
                },
                "audit_triple": {
                    "three_layer_decomposition": {
                        "descriptive_form": "target-derived class cells and separatrices",
                        "inherently_compact_dofs": "complete SDWL1 fact inventory",
                        "coder": "typed-section causal-delta arithmetic plus zlib9",
                    },
                    "scorer_visibility": {
                        "authority_surface": "SegNet frame1 after hard camera placement and R",
                        "status": "PENDING_N600_EXECUTION",
                    },
                    "sensitivity_priced_tolerance": {
                        "metric": "rank4 |margin|/||Delta w|| plus realized transition",
                        "status": "PENDING_N600_EXECUTION",
                    },
                },
                "verdict_scope": (
                    "complete fitted SDWL1 object; cannot be called zero-byte"
                ),
            },
            {
                "member": "global amplitude-statistics correction",
                "first_rung": True,
                "candidate_admissible": False,
                "non_redundancy": {
                    "conditioned_on": ["flat-paint geometry"],
                    "single_owner_facts": [
                        "global RGB first and second moments"
                    ],
                },
                "audit_triple": {
                    "three_layer_decomposition": {
                        "descriptive_form": "one global RGB affine",
                        "inherently_compact_dofs": "six float32 scalars",
                        "coder": "PT1AS1 raw little-endian payload, 30 bytes",
                    },
                    "scorer_visibility": {
                        "authority_surface": "frozen SegNet after exact R",
                        "status": "PENDING_N600_EXECUTION",
                    },
                    "sensitivity_priced_tolerance": {
                        "metric": "d_seg and per-layer Fisher-weighted divergence",
                        "status": "PENDING_N600_EXECUTION",
                    },
                },
                "verdict_scope": "global amplitude-statistics mechanism only",
            },
            {
                "member": "per-stratum spectrum correction",
                "first_rung": True,
                "candidate_admissible": False,
                "non_redundancy": {
                    "conditioned_on": [
                        "flat-paint geometry",
                        "global amplitude-statistics correction",
                    ],
                    "single_owner_facts": [
                        "per-stratum through-R-passband texture coefficients"
                    ],
                },
                "audit_triple": {
                    "three_layer_decomposition": {
                        "descriptive_form": (
                            "fixed texture-trunk period groups with per-stratum RGB weights"
                        ),
                        "inherently_compact_dofs": "45 float32 coefficients",
                        "coder": "PT1SP1 raw little-endian payload, 186 bytes",
                    },
                    "scorer_visibility": {
                        "authority_surface": "frozen SegNet after exact R",
                        "status": "PENDING_N600_EXECUTION",
                    },
                    "sensitivity_priced_tolerance": {
                        "metric": (
                            "d_seg, Fisher-weighted layer divergence, and trajectory depth"
                        ),
                        "status": "PENDING_N600_EXECUTION",
                    },
                },
                "verdict_scope": "local spectrum or region-ERF mechanism only",
            },
        ],
        "correction_policy": {
            "counted_correction_streams": [
                "PT1AS1 global amplitude-statistics payload",
                "PT1SP1 per-stratum spectrum payload",
            ],
            "description_owned_facts_reencoded": False,
            "status": "PASS_SEPARATE_DESCRIPTION_OWNERS",
        },
        "verdict_scope": (
            "PT1 prepared description streams; scorer tolerance rows remain unmeasured"
        ),
    }


def prepare(config: PT1Config, output_path: Path, semantic_argv: list[str]) -> Path:
    e2 = _bound_json(
        REPO_ROOT / config.e2_receipt_path,
        config.e2_receipt_sha256,
        "E2 receipt",
    )
    sdwl1 = _bound_json(
        REPO_ROOT / config.sdwl1_receipt_path,
        config.sdwl1_receipt_sha256,
        "SDWL1 receipt",
    )
    spline = _bound_bytes(
        REPO_ROOT / config.dv1_spline_path,
        config.dv1_spline_sha256,
        "DV1 spline",
    )
    spline_mask, spline_metadata = decode_boundary_worldsheet_spline(spline)
    if spline_mask.shape != (600, *SEG_HW):
        raise ContinuousPaintError("DV1 spline does not cover exact n600 scorer geometry")
    if e2.get("stream_stage_loss_attribution", {}).get("source_class") != (
        "(ii) live export realization-stage loss"
    ):
        raise ContinuousPaintError("E2 receipt lacks the canonical stage attribution")
    e2_paint_errors = sum(
        int(stream["stages"][0]["errors_introduced"])
        for stream in e2["stream_stage_loss_attribution"]["streams"]
    )
    if e2_paint_errors != config.expected_e2_paint_errors:
        raise ContinuousPaintError("E2 paint-floor count differs from the typed contract")
    source_custody = sdwl1.get("source_custody", {})
    if (
        source_custody.get("sha256") != config.target_cache_sha256
        or source_custody.get("bytes") != config.target_cache_bytes
    ):
        raise ContinuousPaintError("SDWL1 target-cache custody differs from PT1 config")
    prepared = {
        "schema": PREPARED_SCHEMA,
        "run_id": config.run_id,
        "status": "PREPARED_NOT_EXECUTED",
        "verdict": "NO_VERDICT_EXECUTION_FORBIDDEN",
        "verdict_scope": (
            "BUILD/PREPARE only; no n600 PT1 arm was scored and no formulation "
            "is confirmed or falsified"
        ),
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": EVIDENCE_AXIS,
        "first_rung": True,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "typed_config_sha256": config.hash(),
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "semantic_argv": semantic_argv,
        "input_custody": {
            "e2_receipt": {
                "path": config.e2_receipt_path,
                "sha256": config.e2_receipt_sha256,
                "paint_errors": e2_paint_errors,
                "paint_d_seg": f"{e2_paint_errors / config.total_seg_sites:.12f}",
            },
            "dv1_spline": {
                "path": config.dv1_spline_path,
                "sha256": config.dv1_spline_sha256,
                "pair_count": spline_metadata.pair_count,
                "temporal_stride": spline_metadata.temporal_stride,
                "horizontal_stride": spline_metadata.horizontal_stride,
            },
            "sdwl1_receipt": {
                "path": config.sdwl1_receipt_path,
                "sha256": config.sdwl1_receipt_sha256,
            },
            "target_cache": {
                "path": config.target_cache_path,
                "bytes": config.target_cache_bytes,
                "sha256": config.target_cache_sha256,
                "hash_policy": "verify_complete_file_before --execute; not reread by --prepare",
            },
        },
        "mechanism_arms": [
            {
                "arm": "hard_camera_placement",
                "primary": True,
                "first_rung": True,
                "mechanism": (
                    "canonical margin-ratio tie-localized continuous partition evaluated "
                    "at 874x1164 camera pixel centres; hard full-amplitude prototype "
                    "write; evaluator float bilinear down"
                ),
                "uint8_policy": "only exact prototype bytes cross uint8",
                "supersampling": False,
                "outcome": None,
            },
            {
                "arm": "analytic_coverage_blend",
                "primary": False,
                "first_rung": True,
                "mechanism": (
                    "analytic signed-distance coverage blend before uint8; "
                    "separate v14-adverse fixed-paint control"
                ),
                "uint8_policy": "blended RGB is rounded once at camera resolution",
                "supersampling": False,
                "outcome": None,
            },
            {
                "arm": "global_amplitude_statistics_match",
                "primary": False,
                "first_rung": True,
                "mechanism": (
                    "current flat-paint camera geometry unchanged; one global "
                    "per-channel mean/variance affine fitted to GT frame1"
                ),
                "uint8_policy": "global affine then one camera-resolution round",
                "counted_payload_bytes": config.amplitude_statistics_payload_bytes,
                "outcome": None,
            },
        ],
        "falsifiers": {
            "mechanism_primary": {
                "law": (
                    "recover at least (1-measured_survival_wall_fraction) of "
                    "adjudicated placement-attributable paint errors"
                ),
                "status": (
                    "PENDING_INDEPENDENT_WALL_RECEIPT"
                    if config.survival_wall_receipt_path is None
                    else "PENDING_N600_EXECUTION"
                ),
                "independent_wall_receipt": (
                    None
                    if config.survival_wall_receipt_path is None
                    else {
                        "path": config.survival_wall_receipt_path,
                        "sha256": config.survival_wall_receipt_sha256,
                    }
                ),
            },
            "box_secondary": {
                "threshold_d_seg": config.box_floor_d_seg,
                "law": "primary post-paint d_seg <= 0.0142",
                "status": "PENDING_N600_EXECUTION",
            },
            "mechanism_decomposition": {
                "required_partition": [
                    "sub_cell_placement",
                    "bn_se_amplitude_statistics",
                    "texture_prior_or_region_erf",
                    "class_interaction",
                ],
                "status": "PENDING_N600_EXECUTION",
            },
        },
        "description_cost_policy": {
            "reused_curve_delta_bytes": 0,
            "fresh_fitted_curve_delta_bytes": None,
            "fresh_charge": (
                "complete typed-section causal-delta SDWL1 outer-zlib bytes; "
                "a zero-byte claim is forbidden"
            ),
            "split_every_positive_row": True,
        },
        "pose_secondary": {
            "status": "PREPARED_NOT_ENABLED",
            "reason": (
                "xi-advected versus static paint requires an independently custodied "
                "decoder-side xi-to-camera warp and the fitted deterministic "
                "stratum-spectrum paint; flat advected paint is explicitly forbidden "
                "as pose-blind"
            ),
            "required_future_rows": [
                "static_amplitude_textured_paint",
                "xi_advected_amplitude_textured_paint",
                "PoseNet embedding delta",
                "PoseNet per-layer trajectory stability",
            ],
        },
        "diagnostic_variants": [
            {
                "variant": "stratum_spectrum_match",
                "first_rung": True,
                "geometry_changed": False,
                "seed": config.seed,
                "counted_payload_bytes": config.spectrum_payload_bytes,
                "basis": (
                    "existing texture_trunk periods 4/6/8 in the measured "
                    "through-R/stem-surviving passband"
                ),
                "fit": (
                    "global per-stratum RGB least-squares coefficients over the "
                    "fixed period groups; global amplitude statistics composed first"
                ),
                "purpose": "isolate local spectrum or region-ERF sensitivity",
                "outcome": None,
            }
        ],
        "scorer_native_diff_contract": {
            "schema": "ddm_pt1_scorer_native_diff_profile.v1",
            "required_for_every_evaluated_variant": True,
            "segnet_fields": [
                "layer",
                "channel_group",
                "spatial_band",
                "delta_norm",
                "delta_norm_relative",
                "fisher_weighted_delta",
                "trajectory_delta_norm_relative",
                "cross_batch_trajectory_delta_norm_relative",
            ],
            "layers": list(config.scorer_native_layers),
            "temporal_scope": (
                "all ordered consecutive frame1 samples, including preserved "
                "cross-batch endpoints"
            ),
            "pose_fields": [
                "embedding_delta",
                "trajectory_delta_norm_relative",
            ],
            "status": "PENDING_N600_EXECUTION",
        },
        "depth_of_first_divergence_contract": {
            "criterion": "exact deterministic delta_norm_relative > 0.0",
            "required_for_final_argmax_failures": True,
            "static_and_feature_trajectory": True,
            "status": "PENDING_N600_EXECUTION",
        },
        "rate_doctrine": _four_clause_audit(),
        "triality": {
            "dsl": "DDMPT1ContinuousPaintCeilingConfigV1",
            "dag": [
                "SHA-bound target and description inputs",
                "continuous local fields split by description owner",
                "hard camera placement OR separate analytic coverage control",
                "OR unchanged-geometry global amplitude-statistics match",
                "OR amplitude-statistics plus measured-passband stratum spectrum",
                "camera uint8",
                "official SegNet preprocess and argmax",
                "per-arm layer/trajectory diff, stage ledger, and falsifier",
            ],
            "equations": [
                "E_after=E_before+E_introduced-E_corrected",
                "t=M_p/(M_p+M_q)",
                "d_flip=|margin|/||Delta w||",
                "S=100*d_seg+sqrt(10*d_pose)+25*B/37545489",
            ],
        },
        "source_binding": _source_binding(),
        "execution_command_after_main_authorization": [
            sys.executable,
            str(Path(__file__).relative_to(REPO_ROOT)),
            "--config",
            str(config_path_from_argv(semantic_argv)),
            "--output",
            str(output_path.with_name("ddm_pt1_measurement_receipt.json")),
            "--execute",
        ],
    }
    _publish(output_path, prepared)
    return output_path


def config_path_from_argv(argv: list[str]) -> str:
    try:
        return argv[argv.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise ContinuousPaintError("semantic argv lacks --config") from exc


def _load_segnet(config: PT1Config) -> tuple[Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    upstream = Path(config.upstream_root).resolve()
    modules_path = upstream / "modules.py"
    if _sha256(_read(modules_path)) != config.scorer_modules_sha256:
        raise ContinuousPaintError("frozen scorer modules SHA-256 differs")
    sys.path.insert(0, str(upstream))
    try:
        import modules as upstream_modules
    finally:
        sys.path.pop(0)
    if Path(upstream_modules.__file__).resolve() != modules_path:
        raise ContinuousPaintError("frozen scorer imported from the wrong path")
    weights_path = Path(upstream_modules.segnet_sd_path).resolve()
    if _sha256(_read(weights_path)) != config.segnet_weights_sha256:
        raise ContinuousPaintError("SegNet weights SHA-256 differs")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = upstream_modules.SegNet().eval().cpu()
    model.load_state_dict(load_file(str(weights_path), device="cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "device": "cpu",
        "deterministic_algorithms": True,
        "modules_path": str(modules_path),
        "modules_sha256": config.scorer_modules_sha256,
        "segnet_weights_path": str(weights_path),
        "segnet_weights_sha256": config.segnet_weights_sha256,
        "batch_size": config.batch_size,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _score_with_activations(
    segnet: Any,
    rgb: np.ndarray,
    *,
    camera: bool,
    layers: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run SegNet and capture bounded 16x16 layer summaries via real hooks."""

    import torch
    import torch.nn.functional as F

    modules = dict(segnet.named_modules())
    missing = sorted(set(layers) - set(modules))
    if missing:
        raise ContinuousPaintError(f"scorer-native layers are absent: {missing}")
    captured: dict[str, np.ndarray] = {}
    handles = []

    def hook(name: str) -> Any:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                raise ContinuousPaintError(
                    f"scorer-native layer {name} emitted a non-tensor"
                )
            value = tensor.detach().float()
            if value.ndim == 4:
                value = F.adaptive_avg_pool2d(
                    value,
                    (min(16, value.shape[-2]), min(16, value.shape[-1])),
                )
            captured[name] = np.ascontiguousarray(value.cpu().numpy())

        return capture

    for name in layers:
        handles.append(modules[name].register_forward_hook(hook(name)))
    try:
        value = np.asarray(rgb)
        expected = CAMERA_HW if camera else SEG_HW
        if value.shape[1:3] != expected or value.shape[-1] != 3:
            raise ContinuousPaintError(
                "scorer-native RGB geometry differs from declared arm"
            )
        nchw = (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(0, 3, 1, 2)
            .float()
        )
        with torch.inference_mode():
            scorer_input = (
                segnet.preprocess_input(torch.stack((nchw, nchw), dim=1))
                if camera
                else nchw
            )
            cells = segnet(scorer_input).argmax(dim=1)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise ContinuousPaintError("scorer-native hook coverage is incomplete")
    ordered = {name: captured[name] for name in layers}
    return np.ascontiguousarray(cells.numpy().astype(np.uint8)), ordered


def _render_grid_batch_to_camera(rgb: np.ndarray) -> np.ndarray:
    """Apply the pinned bicubic-up and uint8 half of R to every sample."""

    value = np.asarray(rgb)
    if value.ndim != 4 or value.shape[1:3] != SEG_HW or value.shape[-1] != 3:
        raise ContinuousPaintError("render-grid RGB must be [pairs,384,512,3]")
    return np.ascontiguousarray(
        np.stack(
            [render_grid_to_camera_uint8(sample) for sample in value],
            axis=0,
        ),
        dtype=np.uint8,
    )


def _camera_batch_to_seg_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the scorer-owned bilinear geometry to camera RGB without a model."""

    import torch
    import torch.nn.functional as F

    value = np.asarray(rgb)
    if (
        value.ndim != 4
        or value.shape[1:3] != CAMERA_HW
        or value.shape[-1] != 3
        or value.dtype != np.uint8
    ):
        raise ContinuousPaintError("camera RGB must be [pairs,874,1164,3] uint8")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .float()
    )
    with torch.inference_mode():
        resized = F.interpolate(
            tensor,
            size=SEG_HW,
            mode="bilinear",
            align_corners=False,
        )
    return np.ascontiguousarray(
        resized.permute(0, 2, 3, 1).numpy(),
        dtype=np.float32,
    )


def _fit_streaming_global_statistics(
    *,
    labels: np.ndarray,
    target_camera: np.ndarray,
    palette: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_sum = np.zeros(3, dtype=np.float64)
    target_sum = np.zeros(3, dtype=np.float64)
    source_sumsq = np.zeros(3, dtype=np.float64)
    target_sumsq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        source = _render_grid_batch_to_camera(
            palette[np.asarray(labels[start:stop], dtype=np.uint8)]
        ).astype(
            np.float64
        )
        target = np.asarray(target_camera[start:stop], dtype=np.float64)
        source_sum += source.sum(axis=(0, 1, 2))
        target_sum += target.sum(axis=(0, 1, 2))
        source_sumsq += np.square(source).sum(axis=(0, 1, 2))
        target_sumsq += np.square(target).sum(axis=(0, 1, 2))
        count += int(np.prod(source.shape[:3]))
    source_mean = source_sum / count
    target_mean = target_sum / count
    source_var = np.maximum(source_sumsq / count - np.square(source_mean), 0.0)
    target_var = np.maximum(target_sumsq / count - np.square(target_mean), 0.0)
    source_std = np.sqrt(source_var)
    target_std = np.sqrt(target_var)
    if np.any(source_std < 1.0e-9):
        raise ContinuousPaintError("flat-paint global channel variance is zero")
    scale = target_std / source_std
    offset = target_mean - scale * source_mean
    return scale.astype(np.float32), offset.astype(np.float32)


def _cross_batch_trajectory(
    *,
    rows: list[dict[str, object]],
    candidate: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    previous_endpoint: Any | None,
    variant: str,
    layer_order: tuple[str, ...],
) -> None:
    if previous_endpoint is None:
        for row in rows:
            row["cross_batch_trajectory_delta_norm_relative"] = None
        return
    for index, row in enumerate(rows):
        layer = layer_order[index]
        candidate_delta = (
            candidate[layer][0].astype(np.float64)
            - previous_endpoint[f"{variant}_{index}"].astype(np.float64)
        )
        target_delta = (
            target[layer][0].astype(np.float64)
            - previous_endpoint[f"gt_{index}"].astype(np.float64)
        )
        row["cross_batch_trajectory_delta_norm_relative"] = float(
            np.linalg.norm((candidate_delta - target_delta).ravel())
            / max(np.linalg.norm(target_delta.ravel()), 1.0e-12)
        )


def _first_divergence_depth(
    rows: list[dict[str, object]],
    *,
    metric: str,
) -> dict[str, object]:
    available = [row.get(metric) is not None for row in rows]
    if not any(available):
        return {
            "metric": metric,
            "status": "UNAVAILABLE_NO_CONSECUTIVE_ENDPOINT",
            "first_divergent_layer_index": None,
            "first_divergent_layer": None,
            "shallowest_downstream_relay_input": None,
        }
    first = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get(metric) is not None and float(row[metric]) > 0.0
        ),
        None,
    )
    if first is None:
        return {
            "metric": metric,
            "status": "EXACT_MATCH_ALL_CAPTURED_DEPTHS",
            "first_divergent_layer_index": None,
            "first_divergent_layer": None,
            "shallowest_downstream_relay_input": rows[-1]["layer"],
        }
    return {
        "metric": metric,
        "status": (
            "STEM_DIVERGENCE_NO_DOWNSTREAM_ONLY_CORRECTION"
            if first == 0
            else "LATE_DIVERGENCE_FEATURE_RELAY_CANDIDATE"
        ),
        "first_divergent_layer_index": first,
        "first_divergent_layer": rows[first]["layer"],
        "shallowest_downstream_relay_input": (
            None if first == 0 else rows[first - 1]["layer"]
        ),
    }


def _depth_of_first_divergence(
    rows: list[dict[str, object]],
    *,
    final_argmax_errors: int,
) -> dict[str, object]:
    status = (
        "NOT_REQUIRED_FINAL_ARGMAX_MATCH"
        if final_argmax_errors == 0
        else "REQUIRED_FINAL_ARGMAX_FAILURE"
    )
    return {
        "schema": "ddm_pt1_depth_of_first_divergence.v1",
        "status": status,
        "criterion": "exact deterministic relative delta > 0.0",
        "final_argmax_errors": final_argmax_errors,
        "static": _first_divergence_depth(
            rows,
            metric="delta_norm_relative",
        ),
        "within_batch_feature_trajectory": _first_divergence_depth(
            rows,
            metric="trajectory_delta_norm_relative",
        ),
        "cross_batch_feature_trajectory": _first_divergence_depth(
            rows,
            metric="cross_batch_trajectory_delta_norm_relative",
        ),
    }


def execute(config: PT1Config, output_path: Path, semantic_argv: list[str]) -> Path:
    """Run exact n600 only; every batch is immutable and resumable."""

    if not config.execution_allowed:
        raise ContinuousPaintError(
            "execution refused: typed config has execution_allowed=false"
        )
    independent_wall, independent_wall_custody = _independent_survival_wall(config)
    target_path = Path(config.target_cache_path)
    if target_path.stat().st_size != config.target_cache_bytes:
        raise ContinuousPaintError("target cache byte count differs")
    target_payload_sha = hashlib.sha256()
    with target_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            target_payload_sha.update(chunk)
    if target_payload_sha.hexdigest() != config.target_cache_sha256:
        raise ContinuousPaintError("target cache SHA-256 differs")
    labels = open_stored_npy_memmap(target_path, "lstars")
    margins = open_stored_npy_memmap(target_path, "margins")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    target_camera = open_stored_npy_memmap(target_path, "gt_f1")
    if labels.shape != (600, *SEG_HW):
        raise ContinuousPaintError("target cache is not exact n600")
    spline_payload = _bound_bytes(
        REPO_ROOT / config.dv1_spline_path,
        config.dv1_spline_sha256,
        "DV1 spline",
    )
    described_curve_mask, _metadata = decode_boundary_worldsheet_spline(
        spline_payload
    )
    segnet, scorer_custody = _load_segnet(config)
    palette = np.asarray(config.palette_rgb_u8, dtype=np.uint8)
    stage_root = output_path.parent / "stage_checkpoints"
    stage_root.mkdir(parents=True, exist_ok=True)
    statistics_checkpoint = stage_root / "00_global_amplitude_statistics.json"
    if statistics_checkpoint.exists():
        statistics_row = json.loads(_read(statistics_checkpoint))
        if statistics_row.get("typed_config_sha256") != config.hash():
            raise ContinuousPaintError("global-statistics checkpoint config differs")
        scale = np.asarray(statistics_row["scale"], dtype=np.float32)
        offset = np.asarray(statistics_row["offset"], dtype=np.float32)
        statistics_payload = encode_global_channel_statistics(scale, offset)
        if _sha256(statistics_payload) != statistics_row["payload_sha256"]:
            raise ContinuousPaintError("global-statistics checkpoint payload differs")
    else:
        scale, offset = _fit_streaming_global_statistics(
            labels=labels,
            target_camera=target_camera,
            palette=palette,
            batch_size=config.batch_size,
        )
        statistics_payload = encode_global_channel_statistics(scale, offset)
        statistics_row = {
            "schema": "ddm_pt1_global_amplitude_statistics.v1",
            "typed_config_sha256": config.hash(),
            "scale": scale.tolist(),
            "offset": offset.tolist(),
            "payload_bytes": len(statistics_payload),
            "payload_sha256": _sha256(statistics_payload),
            "exact_parseback": True,
            "first_rung": True,
            "research_only": True,
            "score_claim": False,
        }
        _publish(statistics_checkpoint, statistics_row)
    spectrum_components = stratum_spectrum_components(seed=config.seed)
    spectrum_checkpoint = stage_root / "01_stratum_spectrum_coefficients.json"
    if spectrum_checkpoint.exists():
        spectrum_row = json.loads(_read(spectrum_checkpoint))
        if spectrum_row.get("typed_config_sha256") != config.hash():
            raise ContinuousPaintError("spectrum checkpoint config differs")
        spectrum_coefficients = np.asarray(
            spectrum_row["coefficients"],
            dtype=np.float32,
        )
        spectrum_payload = encode_stratum_spectrum_coefficients(
            spectrum_coefficients
        )
        if _sha256(spectrum_payload) != spectrum_row["payload_sha256"]:
            raise ContinuousPaintError("spectrum checkpoint payload differs")
    else:
        if np.any(scale <= 1.0e-9):
            raise ContinuousPaintError(
                "global statistics scale cannot condition the spectrum fit"
            )
        period_count = spectrum_components.shape[2]
        spectrum_gram = np.zeros(
            (5, 3, period_count, period_count),
            dtype=np.float64,
        )
        spectrum_rhs = np.zeros((5, 3, period_count), dtype=np.float64)
        for start in range(0, 600, config.batch_size):
            stop = min(start + config.batch_size, 600)
            target_render = _camera_batch_to_seg_rgb(
                np.ascontiguousarray(target_camera[start:stop], dtype=np.uint8)
            )
            target_pre_affine = (
                target_render - offset[None, None, None, :]
            ) / scale[None, None, None, :]
            batch_gram, batch_rhs = stratum_spectrum_normal_equations(
                np.ascontiguousarray(labels[start:stop], dtype=np.uint8),
                target_pre_affine,
                palette,
                seed=config.seed,
                components=spectrum_components,
            )
            spectrum_gram += batch_gram
            spectrum_rhs += batch_rhs
        spectrum_coefficients = solve_stratum_spectrum_coefficients(
            spectrum_gram,
            spectrum_rhs,
        )
        spectrum_payload = encode_stratum_spectrum_coefficients(
            spectrum_coefficients
        )
        spectrum_row = {
            "schema": "ddm_pt1_stratum_spectrum_coefficients.v1",
            "typed_config_sha256": config.hash(),
            "coefficients": spectrum_coefficients.tolist(),
            "payload_bytes": len(spectrum_payload),
            "payload_sha256": _sha256(spectrum_payload),
            "exact_parseback": True,
            "periods_render_px": [4.0, 6.0, 8.0],
            "basis_source": "tac.boundary_math.texture_trunk.default_band_spec",
            "through_r_surviving_only": True,
            "first_rung": True,
            "research_only": True,
            "score_claim": False,
        }
        _publish(spectrum_checkpoint, spectrum_row)
    if len(spectrum_payload) != config.spectrum_payload_bytes:
        raise ContinuousPaintError("spectrum payload byte contract differs")
    batch_rows: list[dict[str, Any]] = []
    for start in range(0, 600, config.batch_size):
        stop = min(start + config.batch_size, 600)
        checkpoint = stage_root / f"batch_{start:04d}_{stop:04d}.json"
        endpoint_path = stage_root / f"batch_{start:04d}_{stop:04d}.endpoints.npz"
        if checkpoint.exists():
            if not endpoint_path.is_file():
                raise ContinuousPaintError(
                    "scorer-native endpoint checkpoint is missing"
                )
            row = json.loads(_read(checkpoint))
            if row.get("typed_config_sha256") != config.hash():
                raise ContinuousPaintError("batch checkpoint config hash differs")
            with np.load(endpoint_path, allow_pickle=False) as endpoint:
                endpoint_hash = bytes(
                    endpoint["typed_config_sha256"].astype(np.uint8).tolist()
                ).decode("ascii")
            if endpoint_hash != config.hash():
                raise ContinuousPaintError(
                    "scorer-native endpoint checkpoint config differs"
                )
            batch_rows.append(row)
            continue
        target = np.ascontiguousarray(labels[start:stop], dtype=np.uint8)
        margin_batch = np.ascontiguousarray(
            margins[start:stop],
            dtype=np.float32,
        )
        fields = signed_distance_fields(target, margins=margin_batch)
        camera_fields = resample_fields_at_pixel_centres(fields)
        primary_rgb = render_hard_camera_placement(camera_fields, palette)
        secondary_rgb = render_analytic_coverage_blend(
            camera_fields,
            palette,
            softness=config.analytic_softness,
        )
        flat_camera_rgb = _render_grid_batch_to_camera(palette[target])
        statistics_rgb = apply_global_channel_statistics(
            flat_camera_rgb,
            scale,
            offset,
        )
        spectrum_render_rgb = render_stratum_spectrum_match(
            target,
            palette,
            spectrum_coefficients,
            seed=config.seed,
            components=spectrum_components,
        )
        spectrum_rgb = apply_global_channel_statistics(
            _render_grid_batch_to_camera(spectrum_render_rgb),
            scale,
            offset,
        )
        baseline_cells, baseline_activations = _score_with_activations(
            segnet,
            flat_camera_rgb,
            camera=True,
            layers=config.scorer_native_layers,
        )
        flat_camera_cells = baseline_cells
        primary_cells, primary_activations = _score_with_activations(
            segnet,
            primary_rgb,
            camera=True,
            layers=config.scorer_native_layers,
        )
        secondary_cells, secondary_activations = _score_with_activations(
            segnet,
            secondary_rgb,
            camera=True,
            layers=config.scorer_native_layers,
        )
        statistics_cells, statistics_activations = _score_with_activations(
            segnet,
            statistics_rgb,
            camera=True,
            layers=config.scorer_native_layers,
        )
        spectrum_cells, spectrum_activations = _score_with_activations(
            segnet,
            spectrum_rgb,
            camera=True,
            layers=config.scorer_native_layers,
        )
        target_cells, target_activations = _score_with_activations(
            segnet,
            np.ascontiguousarray(target_camera[start:stop], dtype=np.uint8),
            camera=True,
            layers=config.scorer_native_layers,
        )
        if not np.array_equal(target_cells, target):
            raise ContinuousPaintError(
                "SHA-bound target labels differ from frozen scorer on gt_f1"
            )
        native_profiles = {
            "e2_native_grid_paint_control": scorer_native_divergence_rows(
                candidate=baseline_activations,
                target=target_activations,
                margins=margin_batch,
            ),
            "hard_camera_placement": scorer_native_divergence_rows(
                candidate=primary_activations,
                target=target_activations,
                margins=margin_batch,
            ),
            "analytic_coverage_blend": scorer_native_divergence_rows(
                candidate=secondary_activations,
                target=target_activations,
                margins=margin_batch,
            ),
            "global_amplitude_statistics_match": scorer_native_divergence_rows(
                candidate=statistics_activations,
                target=target_activations,
                margins=margin_batch,
            ),
            "stratum_spectrum_match": scorer_native_divergence_rows(
                candidate=spectrum_activations,
                target=target_activations,
                margins=margin_batch,
            ),
        }
        previous_endpoint = None
        if start:
            prior_start = max(0, start - config.batch_size)
            previous_path = (
                stage_root
                / f"batch_{prior_start:04d}_{start:04d}.endpoints.npz"
            )
            if not previous_path.is_file():
                raise ContinuousPaintError(
                    "previous scorer-native endpoint checkpoint is missing"
                )
            previous_endpoint = np.load(previous_path, allow_pickle=False)
        activation_by_variant = {
            "baseline": baseline_activations,
            "hard": primary_activations,
            "analytic": secondary_activations,
            "statistics": statistics_activations,
            "spectrum": spectrum_activations,
        }
        profile_variant = {
            "e2_native_grid_paint_control": "baseline",
            "hard_camera_placement": "hard",
            "analytic_coverage_blend": "analytic",
            "global_amplitude_statistics_match": "statistics",
            "stratum_spectrum_match": "spectrum",
        }
        for name, rows in native_profiles.items():
            _cross_batch_trajectory(
                rows=rows,
                candidate=activation_by_variant[profile_variant[name]],
                target=target_activations,
                previous_endpoint=previous_endpoint,
                variant=profile_variant[name],
                layer_order=config.scorer_native_layers,
            )
        if previous_endpoint is not None:
            previous_endpoint.close()
        cells_by_variant = {
            "e2_native_grid_paint_control": baseline_cells,
            "hard_camera_placement": primary_cells,
            "analytic_coverage_blend": secondary_cells,
            "global_amplitude_statistics_match": statistics_cells,
            "stratum_spectrum_match": spectrum_cells,
        }
        divergence_depths = {
            name: _depth_of_first_divergence(
                rows,
                final_argmax_errors=int(
                    np.count_nonzero(cells_by_variant[name] != target)
                ),
            )
            for name, rows in native_profiles.items()
        }
        band = target_boundary_band(target, dilation=config.boundary_dilation)
        mechanisms = decompose_mechanisms(
            target=target,
            baseline=baseline_cells,
            primary_hard=primary_cells,
            statistics_control=flat_camera_cells,
            statistics_matched=statistics_cells,
            texture_probe=spectrum_cells,
            boundary_band=band,
        )
        provenance = split_curve_provenance(
            target_labels=target,
            described_curve_mask=np.ascontiguousarray(
                described_curve_mask[start:stop], dtype=bool
            ),
            dilation=config.boundary_dilation,
        )
        provenance_rows: dict[str, Any] = {}
        for name in (
            "already_described_curve_sites",
            "freshly_fitted_curve_sites",
        ):
            owner = provenance[name]
            provenance_rows[name] = {
                "first_rung": True,
                "primary_transition": stage_transition(
                    before=baseline_cells,
                    after=primary_cells,
                    target=target,
                    owner_mask=owner,
                ),
                "secondary_transition": stage_transition(
                    before=baseline_cells,
                    after=secondary_cells,
                    target=target,
                    owner_mask=owner,
                ),
                "statistics_transition": stage_transition(
                    before=flat_camera_cells,
                    after=statistics_cells,
                    target=target,
                    owner_mask=owner,
                ),
                "spectrum_transition": stage_transition(
                    before=baseline_cells,
                    after=spectrum_cells,
                    target=target,
                    owner_mask=owner,
                ),
            }
        row = {
            "schema": "ddm_pt1_continuous_paint_batch.v1",
            "typed_config_sha256": config.hash(),
            "pair_range": [start, stop],
            "first_rung": True,
            "baseline_transition": stage_transition(
                before=target,
                after=baseline_cells,
                target=target,
            ),
            "primary_transition": stage_transition(
                before=baseline_cells,
                after=primary_cells,
                target=target,
            ),
            "secondary_transition": stage_transition(
                before=baseline_cells,
                after=secondary_cells,
                target=target,
            ),
            "statistics_control_transition": stage_transition(
                before=baseline_cells,
                after=flat_camera_cells,
                target=target,
            ),
            "statistics_transition": stage_transition(
                before=flat_camera_cells,
                after=statistics_cells,
                target=target,
            ),
            "spectrum_transition": stage_transition(
                before=baseline_cells,
                after=spectrum_cells,
                target=target,
            ),
            "mechanisms": mechanisms.as_dict(),
            "curve_provenance": provenance_rows,
            "scorer_native_diff_profile": {
                "schema": "ddm_pt1_scorer_native_diff_profile.v1",
                "first_rung": True,
                "variants": native_profiles,
                "depth_of_first_divergence": divergence_depths,
                "trajectory_scope": (
                    "within-batch ordered samples plus exact previous-batch endpoint"
                ),
            },
            "hashes": {
                "baseline_cells": sha256_array(baseline_cells),
                "primary_camera_u8": sha256_array(primary_rgb),
                "primary_cells": sha256_array(primary_cells),
                "secondary_camera_u8": sha256_array(secondary_rgb),
                "secondary_cells": sha256_array(secondary_cells),
                "statistics_camera_u8": sha256_array(statistics_rgb),
                "statistics_cells": sha256_array(statistics_cells),
                "spectrum_camera_u8": sha256_array(spectrum_rgb),
                "spectrum_cells": sha256_array(spectrum_cells),
                "target": sha256_array(target),
            },
            "research_only": True,
            "score_claim": False,
        }
        endpoint_payload: dict[str, np.ndarray] = {
            "typed_config_sha256": np.frombuffer(
                config.hash().encode("ascii"),
                dtype=np.uint8,
            )
        }
        for index, layer in enumerate(config.scorer_native_layers):
            endpoint_payload[f"gt_{index}"] = target_activations[layer][-1]
            endpoint_payload[f"baseline_{index}"] = baseline_activations[layer][-1]
            endpoint_payload[f"hard_{index}"] = primary_activations[layer][-1]
            endpoint_payload[f"analytic_{index}"] = secondary_activations[layer][-1]
            endpoint_payload[f"statistics_{index}"] = statistics_activations[layer][-1]
            endpoint_payload[f"spectrum_{index}"] = spectrum_activations[layer][-1]
        temporary_endpoint = endpoint_path.with_name(
            f".{endpoint_path.name}.{os.getpid()}.tmp"
        )
        with temporary_endpoint.open("wb") as stream:
            np.savez_compressed(stream, **endpoint_payload)
        os.replace(temporary_endpoint, endpoint_path)
        _publish(checkpoint, row)
        batch_rows.append(row)

    def sum_path(*path: str) -> int:
        total = 0
        for row in batch_rows:
            value: Any = row
            for key in path:
                value = value[key]
            total += int(value)
        return total

    baseline_errors = sum_path("baseline_transition", "errors_after")
    primary_errors = sum_path("primary_transition", "errors_after")
    secondary_errors = sum_path("secondary_transition", "errors_after")
    statistics_errors = sum_path("statistics_transition", "errors_after")
    spectrum_errors = sum_path("spectrum_transition", "errors_after")
    if baseline_errors != config.expected_e2_paint_errors:
        raise ContinuousPaintError(
            f"PT1 baseline paint errors {baseline_errors} != E2 {config.expected_e2_paint_errors}"
        )
    fitted_debt = measure_fitted_geometry_sdwl1(labels, margins, poses)
    attributable = sum_path("mechanisms", "placement_attributable")
    recovered = sum_path("mechanisms", "placement_recovered")
    observed_wall = 1.0 - recovered / attributable if attributable else 1.0
    primary_dseg = primary_errors / config.total_seg_sites
    secondary_dseg = secondary_errors / config.total_seg_sites
    mechanism_bar = attributable > 0 and recovered >= (
        1.0 - independent_wall
    ) * attributable
    box_bar = primary_dseg <= config.box_floor_d_seg
    if attributable == 0:
        measured_verdict = "BLOCKED_NO_PLACEMENT_ATTRIBUTABLE_ERRORS"
    elif mechanism_bar:
        measured_verdict = "PRIMARY_MECHANISM_BAR_PASS"
    else:
        measured_verdict = "FORMULATION_SCOPED_MECHANISM_FALSIFIER_TRIPPED"
    measured = {
        "schema": MEASURED_SCHEMA,
        "run_id": config.run_id,
        "status": "MEASURED_ADVISORY_NOT_PROMOTABLE",
        "verdict": measured_verdict,
        "verdict_scope": (
            "FORMULATION: exact typed PT1 hard-placement and analytic-control "
            "composition on SHA-bound n600 inputs; not contest CPU/CUDA authority"
        ),
        "typed_config_sha256": config.hash(),
        "semantic_argv": semantic_argv,
        "evidence_axis": EVIDENCE_AXIS,
        "first_rung": True,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "scorer_custody": scorer_custody,
        "batch_count": len(batch_rows),
        "all_batches_checkpointed_and_preserved": True,
        "rows": [
            {
                "arm": "e2_native_grid_paint_control",
                "first_rung": True,
                "errors": baseline_errors,
                "d_seg": f"{baseline_errors / config.total_seg_sites:.12f}",
                "delta_bytes": 0,
            },
            {
                "arm": "hard_camera_placement",
                "first_rung": True,
                "errors": primary_errors,
                "d_seg": f"{primary_dseg:.12f}",
                "d_seg_vs_box_residual": f"{primary_dseg - config.box_floor_d_seg:.12f}",
                "delta_bytes_reused_curve_sites": 0,
                "delta_bytes_fitted_curve_object": fitted_debt.bytes,
            },
            {
                "arm": "analytic_coverage_blend",
                "first_rung": True,
                "errors": secondary_errors,
                "d_seg": f"{secondary_dseg:.12f}",
                "delta_bytes_reused_curve_sites": 0,
                "delta_bytes_fitted_curve_object": fitted_debt.bytes,
            },
            {
                "arm": "global_amplitude_statistics_match",
                "first_rung": True,
                "errors": statistics_errors,
                "d_seg": f"{statistics_errors / config.total_seg_sites:.12f}",
                "delta_bytes": len(statistics_payload),
                "payload_sha256": _sha256(statistics_payload),
                "geometry_changed": False,
            },
            {
                "arm": "stratum_spectrum_match",
                "diagnostic_only": True,
                "first_rung": True,
                "errors": spectrum_errors,
                "d_seg": f"{spectrum_errors / config.total_seg_sites:.12f}",
                "delta_bytes": len(spectrum_payload),
                "payload_sha256": _sha256(spectrum_payload),
                "geometry_changed": False,
                "seed": config.seed,
                "periods_render_px": [4.0, 6.0, 8.0],
            },
        ],
        "fitted_sdwl1_debt": {
            "bytes": fitted_debt.bytes,
            "sha256": fitted_debt.sha256,
            "semantic_sha256": fitted_debt.semantic_sha256,
            "exact_parseback": fitted_debt.exact_parseback,
            "described_scalar_facts": fitted_debt.described_scalar_facts,
        },
        "mechanism_falsifier": {
            "placement_attributable": attributable,
            "placement_recovered": recovered,
            "observed_primary_survival_wall_fraction": observed_wall,
            "independent_survival_wall_fraction": independent_wall,
            "independent_wall_custody": independent_wall_custody,
            "pass": mechanism_bar,
        },
        "mechanism_decomposition": {
            "sub_cell_placement": sum_path(
                "mechanisms", "sub_cell_placement"
            ),
            "bn_se_amplitude_statistics": sum_path(
                "mechanisms", "bn_se_amplitude_statistics"
            ),
            "texture_prior_or_region_erf": sum_path(
                "mechanisms", "texture_prior_or_region_erf"
            ),
            "class_interaction": sum_path(
                "mechanisms", "class_interaction"
            ),
            "disjoint_operational_attribution": True,
        },
        "box_falsifier": {
            "threshold_d_seg": config.box_floor_d_seg,
            "observed_primary_d_seg": primary_dseg,
            "residual": primary_dseg - config.box_floor_d_seg,
            "pass": box_bar,
        },
        "rate_doctrine": _four_clause_audit(),
        "scorer_native_diff_profile": {
            "schema": "ddm_pt1_scorer_native_diff_profile.v1",
            "batch_count": len(batch_rows),
            "all_variants_present": all(
                set(row["scorer_native_diff_profile"]["variants"])
                == {
                    "e2_native_grid_paint_control",
                    "hard_camera_placement",
                    "analytic_coverage_blend",
                    "global_amplitude_statistics_match",
                    "stratum_spectrum_match",
                }
                for row in batch_rows
            ),
            "all_consecutive_batch_boundaries_present": all(
                row["pair_range"][0] == 0
                or all(
                    layer["cross_batch_trajectory_delta_norm_relative"] is not None
                    for layers in row["scorer_native_diff_profile"]["variants"].values()
                    for layer in layers
                )
                for row in batch_rows
            ),
            "per_batch_rows": [
                row["scorer_native_diff_profile"] for row in batch_rows
            ],
        },
        "source_binding": _source_binding(),
    }
    _publish(output_path, measured)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    semantic_argv = list(sys.argv if argv is None else [str(Path(__file__)), *argv])
    try:
        config = PT1Config.model_validate_json(_read(args.config))
        if args.prepare:
            output = prepare(config, args.output, semantic_argv)
        else:
            output = execute(config, args.output, semantic_argv)
    except (ContinuousPaintError, OSError, ValueError) as exc:
        print(f"PT1 refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "sha256": _sha256(_read(output))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
