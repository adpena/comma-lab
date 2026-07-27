# SPDX-License-Identifier: MIT
"""G121-retained population wrapper for the exact post-G105 pose refitter.

G119 must never consume a trainer BEST or a semantic-only winner.  The public
entrypoint in this module lazily imports G121's strict physical opener and
iterates every row in ``g121_retained_prepose.json``.  Each row receives an
independent, verifier-accepted post-G105 refit.  The output is a joint
``(d_seg, d_pose, final archive bytes)`` ledger and its nondominated set; it is
not a contest score or a cross-stage winner.

G121 is deliberately a separate producer.  Until its typed opener lands under
the frozen module/API named in the G121 contract, this wrapper fails closed
before opening a G112 stage.  A caller cannot substitute a G120 BEST, a raw
G112 receipt, or an injected list of rows.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Final

from tac.witness_control.taskspace_post_g105_pose_refit_v1 import (
    ARCHIVE_DENOMINATOR_BYTES,
    GLOBAL_RANGE_BLOCKER_SCHEMA,
    MIN_FREE_BYTES,
    OPTIMIZER_VERDICT_SCOPE,
    PostG105PoseRefitConfigV1,
    PostG105PoseRefitError,
    PostG105PoseRefitResultV1,
    _canonical_json,
    _output_binding,
    _require_sha256,
    _seal,
    _strict_file_binding,
    _write_json_once,
    global_range_reactivation_blocker,
    run_post_g105_pose_refit,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import SSD_ROOTS

CONFIG_SCHEMA: Final = "tac.post_g105_pose_refit_population_config.v1"
JOINT_ROW_SCHEMA: Final = "tac.post_g105_pose_refit_joint_axis_row.v1"
JOINT_LEDGER_SCHEMA: Final = "tac.post_g105_pose_refit_joint_axis_ledger.v1"
G121_MODULE: Final = "tac.witness_control.taskspace_g121_resumable_stage_harvest_v1"
G121_OPENER: Final = "open_g121_retained_prepose_v1"
G121_MANIFEST_BASENAME: Final = "g121_retained_prepose.json"
G121_REQUIRED_SCHEMA: Final = "tac.g121_retained_prepose.v2"
G120_UNSAFE_PRODUCTION_COMMIT: Final = "f92813301fa20ed4098640c32ae5ea931f57376d"
G121_RETAIN_DISPOSITION: Final = "RETAIN_POST_G105_POSE"
EXACT_SEG_PIXEL_DENOMINATOR: Final = 600 * 384 * 512
_EXACT_OBSTRUCTION_RULE: Final = "100*k*target_denominator < target_numerator*pixel_denominator"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_UNSIGNED_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_UNSIGNED_INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")


class PostG105PoseRefitPopulationError(PostG105PoseRefitError):
    """The exhaustive G121 population or downstream joint ledger failed."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PostG105PoseRefitPopulationConfigV1:
    config_path: Path
    config_sha256: str
    run_id: str
    seed: int
    output_root: Path
    g121_retained_prepose: dict[str, object]
    target_capsule_receipt: dict[str, object]
    q_levels_candidates: tuple[int, ...]
    local_gauss_newton_stages: int
    finite_difference_q_steps: int
    damping: float
    trust_radius_q: int
    line_search_scales: tuple[float, ...]
    device: str
    torch_num_threads: int


@dataclass(frozen=True, slots=True)
class _RetainedStageV1:
    stage_tag: str
    row_identity_sha256: str
    disagreement_pixels: int
    pixel_denominator: int
    d_seg_wire: float
    live_target_score_decimal: str
    live_target_numerator: int
    live_target_denominator: int
    pointer_snapshot_identity_sha256: str
    postverified_pointer_identity_sha256: str
    g112_partition_receipt: dict[str, object]
    physical_stage_identity_sha256: str
    raw_row: dict[str, object]


@dataclass(frozen=True, slots=True)
class _OpenedRetainedPopulationV1:
    stages: tuple[_RetainedStageV1, ...]
    completion_receipt: dict[str, object]
    manifest_sha256: str
    pointer_snapshot_identity_sha256: str
    postverified_pointer_identity_sha256: str
    live_target_score_decimal: str
    live_target_numerator: int
    live_target_denominator: int


@dataclass(frozen=True, slots=True)
class PostG105PoseRefitPopulationResultV1:
    joint_ledger_path: Path
    joint_ledger_sha256: str
    retained_stage_count: int
    stage_results: tuple[PostG105PoseRefitResultV1, ...]


def seal_population_config(
    body: Mapping[str, object],
) -> dict[str, object]:
    return _seal(body, field="config_sha256")


def load_population_config(
    path: Path,
    *,
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
) -> PostG105PoseRefitPopulationConfigV1:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise PostG105PoseRefitPopulationError("config must not be a symlink")
    resolved = candidate.resolve()
    try:
        value = json.loads(resolved.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostG105PoseRefitPopulationError("population config is not readable strict JSON") from exc
    keys = {
        "schema",
        "run_id",
        "seed",
        "output_root",
        "g121_retained_prepose",
        "target_capsule_receipt",
        "q_levels_candidates",
        "local_gauss_newton_stages",
        "finite_difference_q_steps",
        "damping",
        "trust_radius_q",
        "line_search_scales",
        "device",
        "torch_num_threads",
        "research_only",
        "candidate_claim",
        "score_claim",
        "pointer_moved",
        "config_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise PostG105PoseRefitPopulationError("population config key set differs")
    config_sha = _require_sha256(value["config_sha256"], name="config")
    if _sha256(_canonical_json({key: item for key, item in value.items() if key != "config_sha256"})) != config_sha:
        raise PostG105PoseRefitPopulationError("config self-hash differs")
    if (
        value["schema"] != CONFIG_SCHEMA
        or type(value["run_id"]) is not str
        or _RUN_ID.fullmatch(value["run_id"]) is None
        or type(value["seed"]) is not int
        or value["research_only"] is not True
        or value["candidate_claim"] is not False
        or value["score_claim"] is not False
        or value["pointer_moved"] is not False
    ):
        raise PostG105PoseRefitPopulationError("population identity/false-authority fences differ")
    output = Path(str(value["output_root"])).expanduser()
    roots = tuple(item.expanduser().resolve() for item in allowed_output_roots)
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not roots
        or not any(output.resolve() == root or root in output.resolve().parents for root in roots)
    ):
        raise PostG105PoseRefitPopulationError("population output is outside the SSD storage waterfall")
    output = output.resolve()
    parent = next(
        (item for item in (output, *output.parents) if item.exists()),
        None,
    )
    if parent is None or parent.is_symlink() or not parent.is_dir() or shutil.disk_usage(parent).free < MIN_FREE_BYTES:
        raise PostG105PoseRefitPopulationError("population storage preflight failed")
    g121 = _strict_file_binding(
        value["g121_retained_prepose"],
        name="G121 retained-prepose manifest",
    )
    if Path(str(g121["path"])).name != G121_MANIFEST_BASENAME:
        raise PostG105PoseRefitPopulationError("G119 accepts only g121_retained_prepose.json, never BEST")
    q_levels = value["q_levels_candidates"]
    line_search = value["line_search_scales"]
    if (
        type(q_levels) is not list
        or not 1 <= len(q_levels) <= 8
        or any(type(item) is not int or not 1 <= item <= 32_767 for item in q_levels)
        or q_levels != sorted(set(q_levels))
        or type(line_search) is not list
        or not 1 <= len(line_search) <= 8
        or any(
            type(item) not in {int, float} or not math.isfinite(float(item)) or not 0.0 < float(item) <= 1.0
            for item in line_search
        )
        or float(line_search[0]) != 1.0
        or any(float(line_search[index]) <= float(line_search[index + 1]) for index in range(len(line_search) - 1))
        or type(value["local_gauss_newton_stages"]) is not int
        or not 1 <= value["local_gauss_newton_stages"] <= 8
        or type(value["finite_difference_q_steps"]) is not int
        or not 1 <= value["finite_difference_q_steps"] <= 1024
        or type(value["trust_radius_q"]) is not int
        or not 1 <= value["trust_radius_q"] <= 4096
        or type(value["damping"]) not in {int, float}
        or not math.isfinite(float(value["damping"]))
        or float(value["damping"]) <= 0.0
        or value["device"] not in {"cpu", "cuda"}
        or type(value["torch_num_threads"]) is not int
        or not 1 <= value["torch_num_threads"] <= 64
    ):
        raise PostG105PoseRefitPopulationError("population solver/controller config is outside bounds")
    return PostG105PoseRefitPopulationConfigV1(
        config_path=resolved,
        config_sha256=config_sha,
        run_id=value["run_id"],
        seed=value["seed"],
        output_root=output,
        g121_retained_prepose=g121,
        target_capsule_receipt=_strict_file_binding(
            value["target_capsule_receipt"],
            name="G109 target capsule receipt",
        ),
        q_levels_candidates=tuple(q_levels),
        local_gauss_newton_stages=value["local_gauss_newton_stages"],
        finite_difference_q_steps=value["finite_difference_q_steps"],
        damping=float(value["damping"]),
        trust_radius_q=value["trust_radius_q"],
        line_search_scales=tuple(float(item) for item in line_search),
        device=value["device"],
        torch_num_threads=value["torch_num_threads"],
    )


def _row_mapping(row: object) -> dict[str, object]:
    if hasattr(row, "to_dict") and callable(row.to_dict):
        value = row.to_dict()
    elif isinstance(row, Mapping):
        value = dict(row)
    else:
        raise PostG105PoseRefitPopulationError("typed G121 opener returned a row with no durable mapping")
    if type(value) is not dict:
        raise PostG105PoseRefitPopulationError("typed G121 row mapping is not exact dict")
    return value


def _exact_prepose_coordinates(
    value: Mapping[str, object],
) -> tuple[int, int, float, str, int, int, str, str]:
    """Recompute G121-v2 retention using only exact integer/rational custody."""

    public_wire = value.get("public_wire_seg")
    live_target = value.get("live_target")
    obstruction = value.get("prepose_obstruction")
    if (
        type(public_wire) is not dict
        or set(public_wire)
        != {
            "disagreement_pixels",
            "pixel_denominator",
            "d_seg_rational",
            "d_seg_display_float",
            "measurement_identity_sha256",
        }
        or type(live_target) is not dict
        or set(live_target)
        != {
            "score_decimal",
            "score_rational",
            "pointer_snapshot_identity_sha256",
            "postverified_pointer_identity_sha256",
        }
        or type(obstruction) is not dict
        or set(obstruction)
        != {
            "rule",
            "lhs",
            "rhs",
            "strict_distortion_open",
            "disposition",
        }
    ):
        raise PostG105PoseRefitPopulationError("G121-v2 row lacks exact public-wire/target/obstruction coordinates")
    disagreements = public_wire["disagreement_pixels"]
    pixel_denominator = public_wire["pixel_denominator"]
    d_seg_rational = public_wire["d_seg_rational"]
    d_seg_display = public_wire["d_seg_display_float"]
    target_decimal_text = live_target["score_decimal"]
    target_rational = live_target["score_rational"]
    if (
        type(disagreements) is not int
        or type(pixel_denominator) is not int
        or not 0 <= disagreements <= pixel_denominator
        or pixel_denominator != EXACT_SEG_PIXEL_DENOMINATOR
        or type(d_seg_rational) is not dict
        or set(d_seg_rational) != {"numerator", "denominator"}
        or d_seg_rational["numerator"] != disagreements
        or d_seg_rational["denominator"] != pixel_denominator
        or type(d_seg_display) is not float
        or not math.isfinite(d_seg_display)
        or d_seg_display != disagreements / pixel_denominator
    ):
        raise PostG105PoseRefitPopulationError("G121-v2 exact public-wire Seg coordinate differs")
    _require_sha256(
        public_wire["measurement_identity_sha256"],
        name="G121 public-wire measurement",
    )
    if (
        type(target_decimal_text) is not str
        or _UNSIGNED_DECIMAL.fullmatch(target_decimal_text) is None
        or type(target_rational) is not dict
        or set(target_rational) != {"numerator", "denominator"}
        or type(target_rational["numerator"]) is not int
        or type(target_rational["denominator"]) is not int
        or target_rational["numerator"] <= 0
        or target_rational["denominator"] <= 0
    ):
        raise PostG105PoseRefitPopulationError("G121-v2 exact live-target coordinate differs")
    try:
        decimal_target = Decimal(target_decimal_text)
    except InvalidOperation as exc:
        raise PostG105PoseRefitPopulationError("G121-v2 exact live-target Decimal is invalid") from exc
    target_fraction = Fraction(
        int(target_rational["numerator"]),
        int(target_rational["denominator"]),
    )
    if (
        not decimal_target.is_finite()
        or decimal_target <= 0
        or Fraction(decimal_target) != target_fraction
        or target_fraction.numerator != target_rational["numerator"]
        or target_fraction.denominator != target_rational["denominator"]
    ):
        raise PostG105PoseRefitPopulationError("G121-v2 Decimal/rational live target differs or is not reduced")
    pointer_sha = _require_sha256(
        live_target["pointer_snapshot_identity_sha256"],
        name="G121 pointer snapshot",
    )
    postverified_pointer_sha = _require_sha256(
        live_target["postverified_pointer_identity_sha256"],
        name="G121 postverified pointer",
    )
    lhs = 100 * disagreements * target_fraction.denominator
    rhs = target_fraction.numerator * pixel_denominator
    if (
        obstruction["rule"] != _EXACT_OBSTRUCTION_RULE
        or type(obstruction["lhs"]) is not str
        or _UNSIGNED_INTEGER.fullmatch(obstruction["lhs"]) is None
        or type(obstruction["rhs"]) is not str
        or _UNSIGNED_INTEGER.fullmatch(obstruction["rhs"]) is None
        or obstruction["lhs"] != str(lhs)
        or obstruction["rhs"] != str(rhs)
        or obstruction["strict_distortion_open"] is not (lhs < rhs)
        or obstruction["disposition"] != G121_RETAIN_DISPOSITION
        or not lhs < rhs
    ):
        raise PostG105PoseRefitPopulationError("G121-v2 retained row fails exact distortion-open cross-product")
    return (
        disagreements,
        pixel_denominator,
        disagreements / pixel_denominator,
        target_decimal_text,
        target_fraction.numerator,
        target_fraction.denominator,
        pointer_sha,
        postverified_pointer_sha,
    )


def _open_g121_retained_population(
    config: PostG105PoseRefitPopulationConfigV1,
) -> _OpenedRetainedPopulationV1:
    """Use only G121's physical opener; never parse/inject retained rows here."""

    try:
        module = importlib.import_module(G121_MODULE)
    except ImportError as exc:
        raise PostG105PoseRefitPopulationError(
            f"G121 retained-prepose implementation has not landed; required module={G121_MODULE}"
        ) from exc
    opener = getattr(module, G121_OPENER, None)
    if not callable(opener):
        raise PostG105PoseRefitPopulationError(
            f"G121 retained-prepose implementation lacks the frozen strict opener {G121_OPENER}(path, expected_sha256)"
        )
    opened = opener(
        Path(str(config.g121_retained_prepose["path"])),
        expected_sha256=str(config.g121_retained_prepose["sha256"]),
    )
    opened_schema = getattr(opened, "schema", None)
    rows = getattr(opened, "rows", None)
    exhaustive = getattr(opened, "exhaustive_enumeration_proven", None)
    completion = getattr(opened, "completion_receipt", None)
    manifest_path = getattr(opened, "manifest_path", None)
    manifest_sha256 = getattr(opened, "manifest_sha256", None)
    pointer_snapshot_identity_sha256 = getattr(
        opened,
        "pointer_snapshot_identity_sha256",
        None,
    )
    if (
        opened_schema != G121_REQUIRED_SCHEMA
        or not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not rows
        or exhaustive is not True
        or completion is None
        or not isinstance(manifest_path, Path)
        or manifest_path.expanduser().resolve() != Path(str(config.g121_retained_prepose["path"]))
        or _require_sha256(
            manifest_sha256,
            name="G121 retained-prepose manifest",
        )
        != config.g121_retained_prepose["sha256"]
    ):
        raise PostG105PoseRefitPopulationError(
            "G121 opener did not prove a nonempty exhaustive v2 retained population; "
            f"G120 {G120_UNSAFE_PRODUCTION_COMMIT} is not a safe production dependency"
        )
    pointer_snapshot_sha = _require_sha256(
        pointer_snapshot_identity_sha256,
        name="G121 pointer snapshot",
    )
    completion_binding = _strict_file_binding(
        completion,
        name="G121 exhaustive completion receipt",
    )
    normalized: list[_RetainedStageV1] = []
    seen: set[str] = set()
    exact_target_identity: tuple[str, int, int, str, str] | None = None
    for row in rows:
        value = _row_mapping(row)
        required = {
            "stage_tag",
            "row_identity_sha256",
            "public_wire_seg",
            "live_target",
            "prepose_obstruction",
            "physical_stage_identity",
            "physical_stage_identity_sha256",
            "pose_initializer_identity_sha256",
        }
        if not required.issubset(value):
            raise PostG105PoseRefitPopulationError("G121 retained row lacks G119 custody/value fields")
        row_identity = _require_sha256(
            value["row_identity_sha256"],
            name="G121 retained row",
        )
        stage_tag = value["stage_tag"]
        (
            disagreements,
            pixel_denominator,
            d_seg,
            target_decimal,
            target_numerator,
            target_denominator,
            row_pointer_sha,
            row_postverified_pointer_sha,
        ) = _exact_prepose_coordinates(value)
        physical = value["physical_stage_identity"]
        physical_sha = _require_sha256(
            value["physical_stage_identity_sha256"],
            name="G121 physical stage",
        )
        if (
            type(stage_tag) is not str
            or _RUN_ID.fullmatch(stage_tag) is None
            or type(physical) is not dict
            or row_identity in seen
        ):
            raise PostG105PoseRefitPopulationError("G121-v2 retained row identity/physical custody differs")
        row_target_identity = (
            target_decimal,
            target_numerator,
            target_denominator,
            row_pointer_sha,
            row_postverified_pointer_sha,
        )
        if exact_target_identity is None:
            exact_target_identity = row_target_identity
        elif exact_target_identity != row_target_identity:
            raise PostG105PoseRefitPopulationError("G121-v2 retained manifest mixes exact target/pointer identities")
        if row_pointer_sha != pointer_snapshot_sha:
            raise PostG105PoseRefitPopulationError("G121 opener pointer identity differs from retained row")
        # The G121 opener owns the full physical identity hash.  G119 still
        # reopens the exact G112 binding used by its candidate compiler.
        g112 = _strict_file_binding(
            physical.get("g112_partition_receipt"),
            name=f"G121 stage {stage_tag} G112 receipt",
        )
        pose_initializer = physical.get("g112_pose_initializer")
        if type(pose_initializer) is not dict or _require_sha256(
            pose_initializer.get("sha256"),
            name="G121 pose initializer",
        ) != value.get("pose_initializer_identity_sha256"):
            raise PostG105PoseRefitPopulationError("G121 row pose initializer identity differs")
        seen.add(row_identity)
        normalized.append(
            _RetainedStageV1(
                stage_tag=stage_tag,
                row_identity_sha256=row_identity,
                disagreement_pixels=disagreements,
                pixel_denominator=pixel_denominator,
                d_seg_wire=d_seg,
                live_target_score_decimal=target_decimal,
                live_target_numerator=target_numerator,
                live_target_denominator=target_denominator,
                pointer_snapshot_identity_sha256=row_pointer_sha,
                postverified_pointer_identity_sha256=(row_postverified_pointer_sha),
                g112_partition_receipt=g112,
                physical_stage_identity_sha256=physical_sha,
                raw_row=value,
            )
        )
    if exact_target_identity is None:
        raise AssertionError("nonempty retained rows lost exact target identity")
    # Preserve every typed retained row.  Sorting controls execution only and
    # does not reduce the population.
    return _OpenedRetainedPopulationV1(
        stages=tuple(
            sorted(
                normalized,
                key=lambda row: (row.stage_tag, row.row_identity_sha256),
            )
        ),
        completion_receipt=completion_binding,
        manifest_sha256=str(manifest_sha256),
        pointer_snapshot_identity_sha256=pointer_snapshot_sha,
        postverified_pointer_identity_sha256=exact_target_identity[4],
        live_target_score_decimal=exact_target_identity[0],
        live_target_numerator=exact_target_identity[1],
        live_target_denominator=exact_target_identity[2],
    )


def _stage_config(
    population: PostG105PoseRefitPopulationConfigV1,
    stage: _RetainedStageV1,
) -> PostG105PoseRefitConfigV1:
    stage_run_id = f"{population.run_id[:96]}.g121_{stage.row_identity_sha256[:16]}"
    return PostG105PoseRefitConfigV1(
        config_path=population.config_path,
        config_sha256=population.config_sha256,
        run_id=stage_run_id,
        seed=population.seed,
        output_root=(population.output_root / "stages" / f"{stage.stage_tag}.row_{stage.row_identity_sha256[:16]}"),
        g112_partition_receipt=stage.g112_partition_receipt,
        target_capsule_receipt=population.target_capsule_receipt,
        q_levels_candidates=population.q_levels_candidates,
        local_gauss_newton_stages=population.local_gauss_newton_stages,
        finite_difference_q_steps=population.finite_difference_q_steps,
        damping=population.damping,
        trust_radius_q=population.trust_radius_q,
        line_search_scales=population.line_search_scales,
        device=population.device,
        torch_num_threads=population.torch_num_threads,
    )


def _nondominated_indices(rows: Sequence[Mapping[str, object]]) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(rows):
        axes = (
            Fraction(
                int(row["d_seg_numerator"]),
                int(row["d_seg_denominator"]),
            ),
            float(row["d_pose_exact"]),
            int(row["final_archive_bytes"]),
        )
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_axes = (
                Fraction(
                    int(other["d_seg_numerator"]),
                    int(other["d_seg_denominator"]),
                ),
                float(other["d_pose_exact"]),
                int(other["final_archive_bytes"]),
            )
            if all(left <= right for left, right in zip(other_axes, axes, strict=True)) and any(
                left < right for left, right in zip(other_axes, axes, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def run_g121_retained_pose_population(
    *,
    config: PostG105PoseRefitPopulationConfigV1,
    resume_from: Path,
    command: Sequence[str],
) -> PostG105PoseRefitPopulationResultV1:
    if resume_from.expanduser().resolve() != config.output_root:
        raise PostG105PoseRefitPopulationError("--resume-from must equal the typed population output_root")
    opened_population = _open_g121_retained_population(config)
    stages = opened_population.stages
    results: list[PostG105PoseRefitResultV1] = []
    joint_rows: list[dict[str, object]] = []
    for stage in stages:
        stage_config = _stage_config(config, stage)
        result = run_post_g105_pose_refit(
            config=stage_config,
            resume_from=stage_config.output_root,
            command=command,
        )
        results.append(result)
        component_objective = (
            100.0 * stage.d_seg_wire
            + math.sqrt(10.0 * result.selected_pose_mse)
            + 25.0 * result.selected_archive_bytes / ARCHIVE_DENOMINATOR_BYTES
        )
        row_body: dict[str, object] = {
            "schema": JOINT_ROW_SCHEMA,
            "g121_row_identity_sha256": stage.row_identity_sha256,
            "stage_tag": stage.stage_tag,
            "physical_stage_identity_sha256": (stage.physical_stage_identity_sha256),
            "d_seg_numerator": stage.disagreement_pixels,
            "d_seg_denominator": stage.pixel_denominator,
            "d_seg_wire": stage.d_seg_wire,
            "d_seg_wire_is_display_only": True,
            "live_target_score_decimal": stage.live_target_score_decimal,
            "live_target_numerator": stage.live_target_numerator,
            "live_target_denominator": stage.live_target_denominator,
            "pointer_snapshot_identity_sha256": (stage.pointer_snapshot_identity_sha256),
            "postverified_pointer_identity_sha256": (stage.postverified_pointer_identity_sha256),
            "retention_recomputed_by_exact_cross_product": True,
            "legacy_retained_bool_consulted": False,
            "d_pose_exact": result.selected_pose_mse,
            "final_archive_bytes": result.selected_archive_bytes,
            "final_archive_sha256": result.selected_archive_sha256,
            "selected_q_levels": result.selected_q_levels,
            "component_objective_not_contest_score": component_objective,
            "optimizer_verdict_scope": OPTIMIZER_VERDICT_SCOPE,
            "global_xip2_range_optimality_claim": False,
            "global_range_reactivation_required": True,
            "global_range_reactivation_blocker": (global_range_reactivation_blocker()),
            "post_g105_refit_checkpoint": _output_binding(result.checkpoint_path),
            "post_g105_refit_run_receipt": _output_binding(result.run_receipt_path),
            "post_g105_refit_audit_receipt": _output_binding(result.audit_receipt_path),
            "exact_public_receiver_in_loop": True,
            "upstream_evaluate_py_run": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }
        joint_rows.append(_seal(row_body, field="joint_row_sha256"))
    pareto_indices = _nondominated_indices(joint_rows)
    ledger_body: dict[str, object] = {
        "schema": JOINT_LEDGER_SCHEMA,
        "run_id": config.run_id,
        "config": _output_binding(config.config_path),
        "g121_retained_prepose": config.g121_retained_prepose,
        "g121_completion_receipt": opened_population.completion_receipt,
        "g121_exhaustive_enumeration_proven": True,
        "g121_manifest_sha256": opened_population.manifest_sha256,
        "g121_pointer_snapshot_identity_sha256": (opened_population.pointer_snapshot_identity_sha256),
        "g121_postverified_pointer_identity_sha256": (opened_population.postverified_pointer_identity_sha256),
        "g121_live_target_score_decimal": (opened_population.live_target_score_decimal),
        "g121_live_target_numerator": (opened_population.live_target_numerator),
        "g121_live_target_denominator": (opened_population.live_target_denominator),
        "g120_unsafe_production_commit": G120_UNSAFE_PRODUCTION_COMMIT,
        "g120_v2_exact_rational_dependency_required": True,
        "defer_g115_wire_qat_rows_owned_by_g121_completion": True,
        "population_input": G121_MANIFEST_BASENAME,
        "legacy_or_semantic_best_consumed": False,
        "retained_stage_count": len(stages),
        "processed_stage_count": len(joint_rows),
        "every_retained_stage_processed": len(stages) == len(joint_rows),
        "axes": [
            "d_seg_numerator/d_seg_denominator",
            "d_pose_exact",
            "final_archive_bytes",
        ],
        "rows": joint_rows,
        "nondominated_joint_row_sha256": [joint_rows[index]["joint_row_sha256"] for index in pareto_indices],
        "cross_stage_winner_selected": False,
        "selection_deferred_to_whole_archive_evaluate": True,
        "optimizer_verdict_scope": OPTIMIZER_VERDICT_SCOPE,
        "global_xip2_range_optimality_claim": False,
        "global_range_reactivation_required": True,
        "global_range_reactivation_blocker_schema": (GLOBAL_RANGE_BLOCKER_SCHEMA),
        "upstream_evaluate_py_run": False,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    ledger = _seal(ledger_body, field="joint_ledger_sha256")
    path = config.output_root / "g119_post_g105_joint_axes.json"
    _write_json_once(path, ledger)
    return PostG105PoseRefitPopulationResultV1(
        joint_ledger_path=path,
        joint_ledger_sha256=str(ledger["joint_ledger_sha256"]),
        retained_stage_count=len(stages),
        stage_results=tuple(results),
    )


__all__ = [
    "CONFIG_SCHEMA",
    "G121_MANIFEST_BASENAME",
    "G121_MODULE",
    "G121_OPENER",
    "JOINT_LEDGER_SCHEMA",
    "JOINT_ROW_SCHEMA",
    "PostG105PoseRefitPopulationConfigV1",
    "PostG105PoseRefitPopulationError",
    "PostG105PoseRefitPopulationResultV1",
    "load_population_config",
    "run_g121_retained_pose_population",
    "seal_population_config",
]
