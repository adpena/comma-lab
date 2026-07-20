# SPDX-License-Identifier: MIT
"""Canonical score/rate law for the scoped Einstein--Kolmogorov n24 probe.

The pure arithmetic remains usable without filesystem state.  The explicit
``build_*``/``populate_*`` surface is the triality equation leg: it reads the
frozen aggregate measurement, binds its exact bytes through canonical
``Provenance``, and registers a research-only empirical anchor.  No score or
frontier authority is created by registration.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Literal

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "einstein_kolmogorov_crux_action_rate_contract_v1"
RESEARCH_ONLY_AXIS = "[macOS-CPU advisory]"
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_WEIGHT = 25
SEGMENTATION_WEIGHT = 100.0
POSE_RADICAND_WEIGHT = 10.0
SOURCE_MEASUREMENT = ".omx/research/einstein_kolmogorov_crux_measurement_20260719.json"
SOURCE_FRONTIER_MAGNITUDE = ".omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
BANKED_AB_V3_PATH = ".omx/research/einstein_kolmogorov_banked_n12_ab_20260720_v3.json"
BANKED_AB_V3_SHA256 = "9c5d636a76a9ef77bb29dec64e4221b098e449510f5f04c2f7218da885c63f0a"
RUNG_E_PATH = ".omx/research/constructive_solver_541_rung_e_n48_20260719.json"
RUNG_E_SHA256 = "d966e066bfd24deb0f7ad1fda865337ed2f108c03c93244c24dd592ac69682a9"
TRADE_CELLS_PATH = ".omx/research/seg_secant_rd_curve_n24_20260719_v2.json"
TRADE_CELLS_SHA256 = "28940965904e9238668de6350785ef0e12348275b64fab83b22901726b0d1f85"
JOINT_INVERSE_PATH = ".omx/research/joint_seg_pose_inverse_solve_receipt_n24_20260719.json"
JOINT_INVERSE_SHA256 = "7a6fdbdfb8f6084a6fd79bb0a63490335b22ae308774032fff7471bb4281e3e9"
EXACT_LATTICE_PATH = (
    "/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/aggregate_n600_receipt.json"
)
EXACT_LATTICE_SHA256 = "1509431e2f06963a0d711819d6fcee131fe44599d997ef137dbf4b352f2f2e60"
BANKED_AB_SUPERSEDED = (
    {
        "path": ".omx/research/einstein_kolmogorov_banked_n12_ab_20260720.json",
        "sha256": "9c031e016300768593d8848d265ed5d5c9fb915f335dd9024b3ba0b8ecfb904a",
        "reason": "strict total-byte equality boundary and immutable cleanup-successor custody were not closed",
    },
    {
        "path": ".omx/research/einstein_kolmogorov_banked_n12_ab_20260720_v2.json",
        "sha256": "9de355d7208f0256d9e137ace54c24cf0f9b3f6907dbb3753227f9c26807c7c7",
        "reason": "ephemeral output-root identity and receipt-to-chart provenance were not bound",
    },
)
C1_COMPONENT_DECIMAL_PLACES = 8
C1_ARCHIVE_PATH = (
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/archive.zip"
)
C1_CONTEST_CPU_EVAL_PATH = (
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/"
    "harvest_fc01KXXRAR/contest_auth_eval.json"
)
C1_CONTEST_CPU_VALIDATION_PATH = (
    "/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/"
    "harvest_fc01KXXRAR/modal_cpu_auth_eval_validation.json"
)
EXPECTED_SOURCE_RECEIPTS = {
    C1_ARCHIVE_PATH: {
        "sha256": "e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42",
        "bytes": 409_526_925,
    },
    C1_CONTEST_CPU_EVAL_PATH: {
        "sha256": "4ef77cc58c4232fc0bfa76f02c74ceb6c258e064707522a4b510e9fe06495e99",
    },
    C1_CONTEST_CPU_VALIDATION_PATH: {
        "sha256": "7230903c6fdad0474ccc4470160fa9dda36ddcd06af3ba5a8c454cfe4d414196",
    },
    RUNG_E_PATH: {"sha256": RUNG_E_SHA256},
    BANKED_AB_V3_PATH: {"sha256": BANKED_AB_V3_SHA256},
    TRADE_CELLS_PATH: {"sha256": TRADE_CELLS_SHA256},
    JOINT_INVERSE_PATH: {"sha256": JOINT_INVERSE_SHA256},
    EXACT_LATTICE_PATH: {"sha256": EXACT_LATTICE_SHA256},
}
EXPECTED_EXACT_ARCHIVE_ROW_IDS = {
    "c1_solved_distortion_n600_contest_cpu",
    "v10_rung_e_exact_two_plane_n48_local",
    "banked_n12_exact_receiver_control",
    "banked_n12_scorer_plane_precision_drop1",
}
EXPECTED_INVERSE_ROW_IDS = {
    "factor2_exact_lattice_frame1_n600",
    "joint_zero_band_n24",
}
C1_SOURCE_PROJECTION = {
    "pair_count": 600,
    "archive_bytes": 409_526_925,
    "archive_sha256": "e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42",
    "d_seg": 0.00015196,
    "d_pose": 0.00010184,
}
EXACT_LATTICE_SOURCE_PROJECTION = {
    "pair_count": 600,
    "d_seg": 9.663899739583334e-7,
    "d_pose": None,
    "archive_bytes": None,
    "exact_action": None,
    "frontier_relevant_distortion": True,
}


class InfeasibleByteBudgetError(ValueError):
    """The requested target is already exceeded before any counted bytes."""


def _nonnegative_real(value: float | int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _nonnegative_bytes(value: int, field: str = "archive_bytes") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def contest_action(*, d_seg: float | int, d_pose: float | int, archive_bytes: int) -> float:
    """Evaluate ``100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`` exactly in form."""
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    bytes_used = _nonnegative_bytes(archive_bytes)
    return (
        SEGMENTATION_WEIGHT * measured_seg
        + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
        + RATE_WEIGHT * bytes_used / RATE_DENOMINATOR_BYTES
    )


def frontier_feasible_at_zero_pose_and_rate(*, d_seg: float | int, target_action: float | int) -> bool:
    """Necessary frontier gate using the Seg term alone.

    ``False`` is a hard impossibility result: non-negative Pose and rate terms
    cannot rescue the row. ``True`` is only necessary, never sufficient.
    """

    measured_seg = _nonnegative_real(d_seg, "d_seg")
    target = _nonnegative_real(target_action, "target_action")
    return SEGMENTATION_WEIGHT * measured_seg < target


def _project_integral_population_bytes(*, mean_bytes_per_pair: float | int, pair_count: int) -> int:
    """Project a measured mean byte count only when the product is integral."""

    mean_bytes = _nonnegative_real(mean_bytes_per_pair, "mean_bytes_per_pair")
    population = _nonnegative_bytes(pair_count, "pair_count")
    if population == 0:
        raise ValueError("pair_count must be positive")
    projected = mean_bytes * population
    rounded = round(projected)
    if not math.isclose(projected, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("mean byte projection is not an integral population byte count")
    return int(rounded)


def inclusive_maximum_byte_budget(*, target_action: float | int, d_seg: float | int, d_pose: float | int) -> int:
    """Return the greatest byte count whose action is at most ``target_action``."""

    target = _nonnegative_real(target_action, "target_action")
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    non_rate = SEGMENTATION_WEIGHT * measured_seg + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
    slack = target - non_rate
    if slack < 0.0:
        raise InfeasibleByteBudgetError("target action is infeasible before the rate term")
    candidate = math.floor(slack * RATE_DENOMINATOR_BYTES / RATE_WEIGHT)
    while candidate >= 0 and contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate) > target:
        candidate -= 1
    while contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate + 1) <= target:
        candidate += 1
    return candidate


def maximum_byte_budget(*, target_action: float | int, d_seg: float | int, d_pose: float | int) -> int:
    """Return the greatest integral byte count that *strictly beats* a target.

    The strict integer ceiling is ``ceil(x) - 1``, not ``floor(x)`` at an
    equality boundary. The final comparisons use :func:`contest_action` so
    binary floating-point roundoff cannot silently admit an equal-score byte.
    """

    target = _nonnegative_real(target_action, "target_action")
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    non_rate = SEGMENTATION_WEIGHT * measured_seg + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
    slack = target - non_rate
    if slack <= 0.0:
        raise InfeasibleByteBudgetError("target action cannot be strictly beaten before the rate term")
    candidate = math.ceil(slack * RATE_DENOMINATOR_BYTES / RATE_WEIGHT) - 1
    while (
        candidate >= 0 and contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate) >= target
    ):
        candidate -= 1
    while contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate + 1) < target:
        candidate += 1
    if candidate < 0:
        raise InfeasibleByteBudgetError("target action cannot be strictly beaten with any non-negative byte count")
    return candidate


def fixed_byte_palette_delta(
    *,
    before_d_seg: float | int,
    before_d_pose: float | int,
    after_d_seg: float | int,
    after_d_pose: float | int,
    before_bytes: int,
    after_bytes: int,
) -> float:
    """Return the action delta for a zero-rate palette substitution.

    Refuses unequal byte counts so a rate change cannot be accidentally labelled
    as palette-only actuation.
    """
    before = _nonnegative_bytes(before_bytes, "before_bytes")
    after = _nonnegative_bytes(after_bytes, "after_bytes")
    if before != after:
        raise ValueError("fixed-byte palette actuation requires identical packet bytes")
    return contest_action(d_seg=after_d_seg, d_pose=after_d_pose, archive_bytes=after) - contest_action(
        d_seg=before_d_seg, d_pose=before_d_pose, archive_bytes=before
    )


@dataclass(frozen=True)
class MeasuredHardRReceipt:
    """A caller-supplied, scope-checked hard-R measurement row."""

    receipt_id: str
    verdict_scope: str
    d_seg: float
    d_pose: float
    archive_bytes: int
    axis: Literal["[macOS-CPU advisory]"] = RESEARCH_ONLY_AXIS
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if not self.receipt_id.strip() or not self.verdict_scope.strip():
            raise ValueError("receipt_id and verdict_scope must be non-empty")
        if self.axis != RESEARCH_ONLY_AXIS or self.research_only is not True:
            raise ValueError("this equation accepts only explicit research-only macOS advisory receipts")
        _nonnegative_real(self.d_seg, "d_seg")
        _nonnegative_real(self.d_pose, "d_pose")
        _nonnegative_bytes(self.archive_bytes)


@dataclass(frozen=True)
class DerivationEdge:
    source: str
    target: str
    relation: Literal["MEASURED_HARD_R_INPUT", "DERIVES", "SCOPES"]


@dataclass(frozen=True)
class ResearchOnlyDecision:
    """Non-promotable action/budget decision derived from one supplied receipt."""

    equation_id: str
    axis: Literal["[macOS-CPU advisory]"]
    verdict_scope: str
    measured_action: float
    maximum_bytes_at_target: int
    research_only: Literal[True]
    promotion_eligible: Literal[False]
    derivation_edges: tuple[DerivationEdge, ...]


def derive_research_only_decision(*, receipt: MeasuredHardRReceipt, target_action: float | int) -> ResearchOnlyDecision:
    """Compose ``measured hard-R receipt -> equation -> research-only decision``."""
    target = _nonnegative_real(target_action, "target_action")
    budget = maximum_byte_budget(target_action=target, d_seg=receipt.d_seg, d_pose=receipt.d_pose)
    decision_id = f"research_only_decision:{receipt.receipt_id}"
    return ResearchOnlyDecision(
        equation_id=EQUATION_ID,
        axis=receipt.axis,
        verdict_scope=receipt.verdict_scope,
        measured_action=contest_action(d_seg=receipt.d_seg, d_pose=receipt.d_pose, archive_bytes=receipt.archive_bytes),
        maximum_bytes_at_target=budget,
        research_only=True,
        promotion_eligible=False,
        derivation_edges=(
            DerivationEdge(receipt.receipt_id, EQUATION_ID, "MEASURED_HARD_R_INPUT"),
            DerivationEdge(EQUATION_ID, decision_id, "DERIVES"),
            DerivationEdge(receipt.verdict_scope, decision_id, "SCOPES"),
        ),
    )


def _load_scoped_measurement(path: str | Path) -> tuple[dict, dict, dict]:
    """Load the immutable n24 aggregate and return payload/source/winner rows."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "einstein_kolmogorov_crux_measurement.v2":
        raise ValueError("unexpected Einstein--Kolmogorov measurement schema")
    if payload.get("research_only") is not True or payload.get("score_claim") is not False:
        raise ValueError("canonical anchor requires an explicit research-only non-score receipt")
    scope = str(payload.get("verdict_scope", ""))
    if "n24" not in scope or "no contest-axis score" not in scope:
        raise ValueError("measurement verdict scope must retain n24 and non-contest boundaries")
    rows = payload.get("tournament")
    if not isinstance(rows, list):
        raise ValueError("measurement tournament must be a list")
    by_arm = {row.get("arm"): row for row in rows if isinstance(row, dict)}
    source = by_arm.get("source_per_pair_means")
    winner = by_arm.get("dspsa32_then_coordinate12")
    if not isinstance(source, dict) or not isinstance(winner, dict):
        raise ValueError("measurement must contain source and scoped winner rows")
    for row_name, row in (("source", source), ("winner", winner)):
        if not isinstance(row.get("hard_mismatch_px"), int):
            raise ValueError(f"{row_name} row lacks integral hard mismatch custody")
        if not isinstance(row.get("candidate_bytes"), int):
            raise ValueError(f"{row_name} row lacks integral byte custody")
        _nonnegative_real(row.get("d_seg"), f"{row_name}.d_seg")
        _nonnegative_bytes(row["candidate_bytes"], f"{row_name}.candidate_bytes")
    if winner["candidate_bytes"] != source["candidate_bytes"]:
        raise ValueError("fixed-label palette anchor requires identical packet bytes")
    if winner["hard_mismatch_px"] >= source["hard_mismatch_px"]:
        raise ValueError("measured scoped winner must strictly improve the in-run source control")
    correction = payload.get("operating_point_correction")
    if not isinstance(correction, dict):
        raise ValueError("measurement lacks the operating-point correction")
    if correction.get("verdict") != "WRONG_OPERATING_POINT_WALL_CHARACTERIZATION":
        raise ValueError("measurement operating-point verdict is not fail-closed")
    target_action = _nonnegative_real(correction.get("target_action"), "operating_point.target_action")
    measured_feasible = frontier_feasible_at_zero_pose_and_rate(
        d_seg=winner["d_seg"],
        target_action=target_action,
    )
    if measured_feasible or correction.get("frontier_feasible_even_at_zero_pose_zero_bytes") is not False:
        raise ValueError("measurement operating point must fail the Seg-only frontier necessity gate")
    if correction.get("n600_explicit_target_launch_eligible") is not False:
        raise ValueError("infeasible explicit-target operating point must not authorize n600 scaling")
    return payload, source, winner


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(commit: str, repo_relative_path: str) -> str:
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("historical git commit custody is malformed")
    completed = subprocess.run(
        ["git", "show", f"{commit}:{repo_relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("historical git blob custody is unavailable")
    return hashlib.sha256(completed.stdout).hexdigest()


def _resolve_source_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_artifact_row(row: dict, *, label: str, portable_absolute: bool = False) -> Path | None:
    path_value = row.get("path")
    sha256 = row.get("sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or sha256.lower() != sha256
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{label} path/SHA custody is malformed")
    path = _resolve_source_path(path_value)
    if not Path(path_value).is_absolute() and not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{label} relative path escapes the repository")
    if not path.is_file():
        if Path(path_value).is_absolute() and portable_absolute and row.get("portable_if_absent") is True:
            return None
        raise ValueError(f"{label} source artifact is absent")
    expected_bytes = row.get("bytes")
    if expected_bytes is not None and _nonnegative_bytes(expected_bytes, f"{label}.bytes") != path.stat().st_size:
        raise ValueError(f"{label} byte custody drift")
    if _sha256_file(path) != sha256:
        raise ValueError(f"{label} SHA custody drift")
    return path


def _validate_source_receipts(payload: dict) -> dict[str, dict]:
    rows = payload.get("source_receipts")
    if not isinstance(rows, list) or len(rows) != 8 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("frontier-magnitude chart requires exactly eight source receipts")
    paths = [row.get("path") for row in rows]
    if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("frontier-magnitude source-receipt paths must be unique strings")
    if set(paths) != set(EXPECTED_SOURCE_RECEIPTS):
        raise ValueError("frontier-magnitude source-receipt set drifted")
    by_path: dict[str, dict] = {}
    for index, row in enumerate(rows):
        expected = EXPECTED_SOURCE_RECEIPTS[row["path"]]
        if row.get("sha256") != expected["sha256"] or ("bytes" in expected and row.get("bytes") != expected["bytes"]):
            raise ValueError(f"source_receipts[{index}] frozen declaration drifted")
        _validate_artifact_row(row, label=f"source_receipts[{index}]", portable_absolute=True)
        by_path[row["path"]] = row
    return by_path


def _rows_by_exact_id(rows: object, *, expected_ids: set[str], label: str) -> dict[str, dict]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must be a list of mappings")
    point_ids = [row.get("point_id") for row in rows]
    if any(not isinstance(point_id, str) or not point_id for point_id in point_ids):
        raise ValueError(f"{label} point IDs must be non-empty strings")
    if len(set(point_ids)) != len(point_ids):
        raise ValueError(f"{label} point IDs must be unique")
    if set(point_ids) != expected_ids:
        raise ValueError(f"{label} row set drifted")
    return {row["point_id"]: row for row in rows}


def _load_json_source(
    source_rows: dict[str, dict],
    *,
    source_path: str,
    label: str,
) -> dict | None:
    source_row = source_rows.get(source_path)
    if not isinstance(source_row, dict):
        raise ValueError(f"{label} source receipt is absent")
    path = _resolve_source_path(source_path)
    if not path.is_file():
        if Path(source_path).is_absolute() and source_row.get("portable_if_absent") is True:
            return None
        raise ValueError(f"{label} source artifact is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} source payload must be a mapping")
    return payload


def _require_source_projection(row: dict, expected: dict, *, label: str) -> None:
    drifted = [key for key, value in expected.items() if row.get(key) != value]
    if drifted:
        raise ValueError(f"{label} drifted from source receipt fields: {', '.join(drifted)}")


def _decimal_action(*, d_seg: Decimal, d_pose: Decimal, archive_bytes: int) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        return (
            Decimal(100) * d_seg
            + (Decimal(10) * d_pose).sqrt()
            + Decimal(25) * Decimal(archive_bytes) / Decimal(RATE_DENOMINATOR_BYTES)
        )


def _rounded_component_interval(value: float | int, *, decimal_places: int) -> tuple[Decimal, Decimal]:
    center = Decimal(str(_nonnegative_real(value, "rounded_component")))
    half_ulp = Decimal(5).scaleb(-(decimal_places + 1))
    return center - half_ulp, center + half_ulp


def _decimal_strict_byte_budget(*, target: Decimal, d_seg: Decimal, d_pose: Decimal) -> int:
    with localcontext() as context:
        context.prec = 60
        slack = target - Decimal(100) * d_seg - (Decimal(10) * d_pose).sqrt()
        if slack <= 0:
            raise InfeasibleByteBudgetError("target action cannot be strictly beaten before the rate term")
        boundary = slack * Decimal(RATE_DENOMINATOR_BYTES) / Decimal(RATE_WEIGHT)
        candidate = int(boundary.to_integral_value(rounding=ROUND_CEILING)) - 1
        if not (
            _decimal_action(d_seg=d_seg, d_pose=d_pose, archive_bytes=candidate) < target
            and _decimal_action(d_seg=d_seg, d_pose=d_pose, archive_bytes=candidate + 1) >= target
        ):
            raise ValueError("decimal strict byte-cap bracketing failed")
        return candidate


def _validate_c1_source_custody(bank: dict, *, source_rows: dict[str, dict]) -> None:
    archive_source = source_rows.get(C1_ARCHIVE_PATH)
    eval_source = source_rows.get(C1_CONTEST_CPU_EVAL_PATH)
    validation_source = source_rows.get(C1_CONTEST_CPU_VALIDATION_PATH)
    if not all(isinstance(row, dict) for row in (archive_source, eval_source, validation_source)):
        raise ValueError("C1 archive/evaluation source receipts are incomplete")
    _require_source_projection(bank, C1_SOURCE_PROJECTION, label="C1 chart row")
    if (
        bank.get("archive_bytes") != archive_source.get("bytes")
        or bank.get("archive_sha256") != archive_source.get("sha256")
        or bank.get("pair_count") != 600
    ):
        raise ValueError("C1 chart archive custody drifted from its exact source receipt")
    eval_path = _resolve_source_path(C1_CONTEST_CPU_EVAL_PATH)
    validation_path = _resolve_source_path(C1_CONTEST_CPU_VALIDATION_PATH)
    if not eval_path.is_file() or not validation_path.is_file():
        return
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"), parse_float=Decimal)
    validation = json.loads(validation_path.read_text(encoding="utf-8"), parse_float=Decimal)
    reported_d_seg = evaluation.get("avg_segnet_dist")
    reported_d_pose = evaluation.get("avg_posenet_dist")
    if (
        not isinstance(reported_d_seg, Decimal)
        or not isinstance(reported_d_pose, Decimal)
        or reported_d_seg.as_tuple().exponent != -C1_COMPONENT_DECIMAL_PLACES
        or reported_d_pose.as_tuple().exponent != -C1_COMPONENT_DECIMAL_PLACES
    ):
        raise ValueError("C1 official evaluation components are not lexically reported to eight decimals")
    if (
        Decimal(str(bank.get("d_seg"))) != reported_d_seg
        or Decimal(str(bank.get("d_pose"))) != reported_d_pose
        or evaluation.get("archive_size_bytes") != bank.get("archive_bytes")
        or evaluation.get("n_samples") != bank.get("pair_count")
        or evaluation.get("score_axis") != "contest_cpu"
        or evaluation.get("provenance", {}).get("archive_sha256") != bank.get("archive_sha256")
        or Decimal(str(bank.get("derived_action_point_estimate_from_reported_centers")))
        != evaluation.get("score_recomputed_from_components")
    ):
        raise ValueError("C1 chart values drifted from the official contest-CPU evaluation")
    if (
        validation.get("passed") is not True
        or validation.get("archive_size_bytes") != bank.get("archive_bytes")
        or validation.get("expected_archive_sha256") != bank.get("archive_sha256")
        or validation.get("avg_segnet_dist") != reported_d_seg
        or validation.get("avg_posenet_dist") != reported_d_pose
        or validation.get("score_axis") != "contest_cpu"
    ):
        raise ValueError("C1 chart values drifted from the official policy-clamp receipt")


def _validate_c1_rounded_interval(bank: dict, *, target: float) -> None:
    if (
        bank.get("archive_measurement_label") != "MEASURED_EXACT_BYTES [contest-CPU custody]"
        or bank.get("distortion_measurement_label") != "MEASURED_ROUNDED_8DP [contest-CPU]"
    ):
        raise ValueError("C1 exact-byte/rounded-distortion authority labels drifted")
    d_seg_interval = _rounded_component_interval(bank.get("d_seg"), decimal_places=C1_COMPONENT_DECIMAL_PLACES)
    d_pose_interval = _rounded_component_interval(bank.get("d_pose"), decimal_places=C1_COMPONENT_DECIMAL_PLACES)
    recorded_components = bank.get("rounded_component_intervals")
    if not isinstance(recorded_components, dict):
        raise ValueError("C1 rounded-component interval custody is absent")
    for name, calculated in (("d_seg", d_seg_interval), ("d_pose", d_pose_interval)):
        recorded = recorded_components.get(name)
        if (
            not isinstance(recorded, dict)
            or Decimal(str(recorded.get("lower"))) != calculated[0]
            or Decimal(str(recorded.get("upper"))) != calculated[1]
            or recorded.get("closure") != "closed_half_ulp"
        ):
            raise ValueError(f"C1 {name} rounding interval drift")
    center_action = contest_action(
        d_seg=bank.get("d_seg"),
        d_pose=bank.get("d_pose"),
        archive_bytes=bank.get("archive_bytes"),
    )
    expected_seg_term = SEGMENTATION_WEIGHT * bank["d_seg"]
    expected_pose_term = math.sqrt(POSE_RADICAND_WEIGHT * bank["d_pose"])
    expected_rate_term = RATE_WEIGHT * bank["archive_bytes"] / RATE_DENOMINATOR_BYTES
    if not math.isclose(
        center_action,
        bank.get("derived_action_point_estimate_from_reported_centers"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in (
            (bank.get("derived_seg_term_from_reported_center"), expected_seg_term),
            (bank.get("derived_pose_term_from_reported_center"), expected_pose_term),
            (bank.get("rate_term_from_exact_archive_bytes"), expected_rate_term),
        )
    ):
        raise ValueError("C1 reported-center action arithmetic drift")
    action_interval = bank.get("derived_action_interval_from_rounded_components")
    lower_action = _decimal_action(
        d_seg=d_seg_interval[0],
        d_pose=d_pose_interval[0],
        archive_bytes=bank["archive_bytes"],
    )
    upper_action = _decimal_action(
        d_seg=d_seg_interval[1],
        d_pose=d_pose_interval[1],
        archive_bytes=bank["archive_bytes"],
    )
    if (
        not isinstance(action_interval, dict)
        or action_interval.get("label") != "DERIVED_INTERVAL_FROM_ROUNDED_COMPONENTS"
        or Decimal(str(action_interval.get("lower"))) != lower_action
        or Decimal(str(action_interval.get("upper"))) != upper_action
    ):
        raise ValueError("C1 action interval drift")
    target_decimal = Decimal(str(target))
    cap_min = _decimal_strict_byte_budget(
        target=target_decimal,
        d_seg=d_seg_interval[1],
        d_pose=d_pose_interval[1],
    )
    cap_max = _decimal_strict_byte_budget(
        target=target_decimal,
        d_seg=d_seg_interval[0],
        d_pose=d_pose_interval[0],
    )
    cap = bank.get("derived_strict_total_archive_cap")
    if (
        not isinstance(cap, dict)
        or cap.get("label") != "DERIVED_STRICT_TOTAL_ARCHIVE_CAP_INTERVAL"
        or cap.get("point_estimate_from_reported_centers")
        != maximum_byte_budget(target_action=target, d_seg=bank["d_seg"], d_pose=bank["d_pose"])
        or cap.get("interval_bytes") != [cap_min, cap_max]
        or cap.get("guaranteed_safe_cap_bytes") != cap_min
        or cap.get("scope") != "total archive.zip bytes, not a predictor-only budget"
    ):
        raise ValueError("C1 total-archive cap interval drift")
    if bank.get("frontier_relevant_distortion") is not frontier_feasible_at_zero_pose_and_rate(
        d_seg=bank["d_seg"], target_action=target
    ) or bank.get("inside_pointer_byte_box") is not (bank["archive_bytes"] <= cap_min):
        raise ValueError("C1 frontier relevance/byte-box classification drift")


def _validate_rung_e_source(row: dict, *, source_rows: dict[str, dict], target: float) -> None:
    receipt = _load_json_source(source_rows, source_path=RUNG_E_PATH, label="rung-E")
    if receipt is None:
        raise ValueError("repository-local rung-E receipt cannot be portable-only")
    archive = receipt.get("archive")
    hard_oracle = receipt.get("hard_oracle")
    pair_ids = receipt.get("pair_ids")
    oracle_pairs = hard_oracle.get("pairs") if isinstance(hard_oracle, dict) else None
    if (
        receipt.get("schema") != "v10_free_predictor_floor_rung_e.v1"
        or not isinstance(archive, dict)
        or not isinstance(hard_oracle, dict)
        or not isinstance(pair_ids, list)
        or not isinstance(oracle_pairs, list)
        or len(pair_ids) != len(oracle_pairs)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError("rung-E source receipt structure drifted")
    projected_bytes = (archive["bytes"] * 600 + len(pair_ids) - 1) // len(pair_ids)
    _require_source_projection(
        row,
        {
            "pair_count": len(pair_ids),
            "archive_bytes": archive["bytes"],
            "archive_sha256": archive["sha256"],
            "d_seg": hard_oracle["mean_d_seg"],
            "d_pose": hard_oracle["mean_d_pose"],
            "strict_bytes_to_beat_pointer_at_measured_distortion": maximum_byte_budget(
                target_action=target,
                d_seg=hard_oracle["mean_d_seg"],
                d_pose=hard_oracle["mean_d_pose"],
            ),
            "frontier_relevant_distortion": frontier_feasible_at_zero_pose_and_rate(
                d_seg=hard_oracle["mean_d_seg"],
                target_action=target,
            ),
            "derived_n600_linear_archive_bytes": projected_bytes,
            "derived_n600_action_at_unchanged_mean_distortion": contest_action(
                d_seg=hard_oracle["mean_d_seg"],
                d_pose=hard_oracle["mean_d_pose"],
                archive_bytes=projected_bytes,
            ),
        },
        label="rung-E chart row",
    )


def _validate_trade_source(
    chart_points: object,
    *,
    source_rows: dict[str, dict],
) -> dict[str, dict]:
    receipt = _load_json_source(source_rows, source_path=TRADE_CELLS_PATH, label="trade-cells")
    if receipt is None:
        raise ValueError("repository-local trade-cells receipt cannot be portable-only")
    measured_points = receipt.get("measured_points")
    if receipt.get("schema") != "seg_secant_rd_curve_composed.v1" or not isinstance(measured_points, list):
        raise ValueError("trade-cells source receipt structure drifted")
    source_by_id = _rows_by_exact_id(
        measured_points,
        expected_ids={
            "source_reference",
            "margin_m0p01",
            "margin_m0p03",
            "margin_m0p1",
            "margin_m0p3",
            "precision_drop1",
            "precision_drop2",
            "precision_drop3",
            "spatial_stride8",
            "spatial_stride16",
        },
        label="trade-cells source",
    )
    chart_by_id = _rows_by_exact_id(
        chart_points,
        expected_ids=set(source_by_id),
        label="trade-cells chart",
    )
    for point_id, source in source_by_id.items():
        _require_source_projection(
            chart_by_id[point_id],
            {
                "point_id": source["point_id"],
                "family": source["family"],
                "d_seg": source["d_seg"],
                "d_pose": source["d_pose"],
                "measured_mean_payload_bytes_per_pair": source["brotli_q11_bytes_per_pair"],
                "pose_violation_count": source["pose_violation_count"],
            },
            label=f"trade-cells chart row {point_id}",
        )
    return chart_by_id


def _validate_inverse_sources(
    chart_rows: object,
    *,
    source_rows: dict[str, dict],
    target: float,
) -> dict[str, dict]:
    by_id = _rows_by_exact_id(
        chart_rows,
        expected_ids=EXPECTED_INVERSE_ROW_IDS,
        label="exact inverse nonarchive chart",
    )
    lattice_row = by_id["factor2_exact_lattice_frame1_n600"]
    _require_source_projection(
        lattice_row,
        EXACT_LATTICE_SOURCE_PROJECTION,
        label="exact-lattice inverse chart row",
    )
    lattice = _load_json_source(source_rows, source_path=EXACT_LATTICE_PATH, label="exact-lattice")
    if lattice is not None:
        exact_lattice = lattice.get("exact_lattice")
        if not isinstance(exact_lattice, dict) or lattice.get("pairs") != 600:
            raise ValueError("exact-lattice source receipt structure drifted")
        receipt_projection = {
            "pair_count": lattice["pairs"],
            "d_seg": exact_lattice["d_seg"],
            "d_pose": None,
            "archive_bytes": None,
            "exact_action": None,
            "frontier_relevant_distortion": frontier_feasible_at_zero_pose_and_rate(
                d_seg=exact_lattice["d_seg"],
                target_action=target,
            ),
        }
        _require_source_projection(
            EXACT_LATTICE_SOURCE_PROJECTION,
            receipt_projection,
            label="frozen exact-lattice source projection",
        )
    joint = _load_json_source(source_rows, source_path=JOINT_INVERSE_PATH, label="joint inverse")
    if joint is None:
        raise ValueError("repository-local joint inverse receipt cannot be portable-only")
    curves = joint.get("measured_curves")
    if joint.get("schema") != "joint_seg_pose_inverse_rate_composed.v1" or not isinstance(curves, list):
        raise ValueError("joint inverse source receipt structure drifted")
    if len(curves) != 1 or not isinstance(curves[0], dict):
        raise ValueError("joint inverse source receipt must contain exactly one measured curve")
    source = curves[0]
    projected_bytes = _project_integral_population_bytes(
        mean_bytes_per_pair=source["bytes"],
        pair_count=600,
    )
    _require_source_projection(
        by_id["joint_zero_band_n24"],
        {
            "pair_count": source["pair_count"],
            "measured_mean_payload_bytes_per_pair": source["bytes"],
            "d_seg": source["d_seg"],
            "d_pose": source["d_pose"],
            "derived_n600_payload_bytes": projected_bytes,
            "derived_action_on_declared_payload_scope": contest_action(
                d_seg=source["d_seg"],
                d_pose=source["d_pose"],
                archive_bytes=projected_bytes,
            ),
            "strict_bytes_to_beat_pointer_at_measured_distortion": maximum_byte_budget(
                target_action=target,
                d_seg=source["d_seg"],
                d_pose=source["d_pose"],
            ),
            "frontier_relevant_distortion": frontier_feasible_at_zero_pose_and_rate(
                d_seg=source["d_seg"],
                target_action=target,
            ),
        },
        label="joint inverse chart row",
    )
    return by_id


def _validate_banked_v3_receipt(*, source_rows: dict[str, dict], chart_rows: dict[str, dict]) -> None:
    source = source_rows.get(BANKED_AB_V3_PATH)
    if not isinstance(source, dict) or source.get("sha256") != BANKED_AB_V3_SHA256:
        raise ValueError("frontier chart does not bind the authoritative banked v3 receipt")
    receipt_path = _resolve_source_path(BANKED_AB_V3_PATH)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    lifecycle = receipt.get("artifact_lifecycle")
    if (
        receipt.get("schema") != "v10_free_predictor_floor_banked_ab.v2"
        or receipt.get("supersedes") != list(BANKED_AB_SUPERSEDED)
        or not isinstance(lifecycle, dict)
        or lifecycle.get("cleanup_completed") is not True
        or lifecycle.get("cleanup_status") != "complete"
        or lifecycle.get("per_arm_stage_receipts_preserved_after_success") is not True
    ):
        raise ValueError("authoritative banked v3 receipt lacks mechanical supersession/lifecycle closure")
    for index, superseded in enumerate(BANKED_AB_SUPERSEDED):
        _validate_artifact_row(superseded, label=f"banked v3 superseded receipt {index}")
    rebuildable = lifecycle.get("rebuildable_from")
    cleanup_execution = lifecycle.get("cleanup_execution_custody")
    measurement_execution = receipt.get("execution_custody")
    if (
        not isinstance(rebuildable, dict)
        or not isinstance(cleanup_execution, dict)
        or not isinstance(measurement_execution, dict)
        or cleanup_execution.get("cross_version_cleanup_only") is not True
        or cleanup_execution.get("precleanup_tool_sha256") != rebuildable.get("tool_sha256")
        or measurement_execution.get("tool_sha256") != rebuildable.get("tool_sha256")
        or _git_blob_sha256(
            measurement_execution.get("git_head_before_measurement"),
            "tools/measure_v10_free_predictor_floor.py",
        )
        != measurement_execution.get("tool_sha256")
        or _git_blob_sha256(
            cleanup_execution.get("git_head"),
            "tools/measure_v10_free_predictor_floor.py",
        )
        != cleanup_execution.get("tool_sha256")
        or cleanup_execution.get("tool_sha256") == cleanup_execution.get("precleanup_tool_sha256")
    ):
        raise ValueError("authoritative banked v3 cross-version cleanup custody drifted")
    predecessor = lifecycle.get("precleanup_receipt")
    if not isinstance(predecessor, dict):
        raise ValueError("authoritative banked v3 receipt lacks its immutable predecessor")
    predecessor_path = _validate_artifact_row(predecessor, label="banked v3 precleanup receipt")
    if predecessor_path is None:
        raise ValueError("banked v3 predecessor cannot be portable-only")
    predecessor_payload = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor_lifecycle = predecessor_payload.get("artifact_lifecycle")
    if (
        predecessor_payload.get("schema") != receipt.get("schema")
        or not isinstance(predecessor_lifecycle, dict)
        or predecessor_lifecycle.get("cleanup_completed") is not False
        or predecessor_lifecycle.get("cleanup_status") != "pending"
    ):
        raise ValueError("banked v3 immutable predecessor state drifted")
    reconstructed = json.loads(json.dumps(receipt))
    reconstructed_lifecycle = reconstructed["artifact_lifecycle"]
    reconstructed_lifecycle["cleanup_completed"] = False
    reconstructed_lifecycle["cleanup_status"] = "pending"
    for key in ("cleanup_completed_at_utc", "cleanup_execution_custody", "precleanup_receipt"):
        reconstructed_lifecycle.pop(key, None)
    if reconstructed != predecessor_payload:
        raise ValueError("banked v3 final receipt is not an exact cleanup-only successor")
    stage_rows = lifecycle.get("durable_stage_receipts")
    if not isinstance(stage_rows, dict) or set(stage_rows) != {"control", "precision-drop"}:
        raise ValueError("banked v3 does not bind both durable arm stages")
    expected_stage_arm = {"control": "control", "precision-drop": "precision_drop"}
    for stage_id, arm_id in expected_stage_arm.items():
        row = stage_rows.get(stage_id)
        if not isinstance(row, dict):
            raise ValueError(f"banked v3 {stage_id} stage custody is malformed")
        stage_path = _validate_artifact_row(row, label=f"banked v3 {stage_id} stage")
        if stage_path is None:
            raise ValueError("banked v3 stages cannot be portable-only")
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if (
            stage.get("schema") != "v10_free_predictor_floor_banked_ab_stage.v1"
            or stage.get("arm_id") != arm_id
            or stage.get("stage_complete") is not True
        ):
            raise ValueError(f"banked v3 {stage_id} stage schema/completion drifted")
    arms = receipt.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("banked v3 receipt lacks its arm map")
    for chart_id, receipt_id in (
        ("banked_n12_exact_receiver_control", "control"),
        ("banked_n12_scorer_plane_precision_drop1", "precision_drop"),
    ):
        chart = chart_rows[chart_id]
        arm = arms.get(receipt_id)
        if not isinstance(arm, dict):
            raise ValueError(f"banked v3 receipt lacks arm {receipt_id}")
        expected_stage = json.loads(json.dumps(arm))
        expected_stage.pop("projection", None)
        stage_path = _resolve_source_path(
            stage_rows["precision-drop" if receipt_id == "precision_drop" else "control"]["path"]
        )
        if json.loads(stage_path.read_text(encoding="utf-8")) != expected_stage:
            raise ValueError(f"banked v3 durable stage drifted from final arm {receipt_id}")
        projection = arm.get("projection")
        hard_oracle = arm.get("hard_oracle")
        if not isinstance(projection, dict) or not isinstance(hard_oracle, dict):
            raise ValueError(f"banked v3 arm {receipt_id} lacks oracle/projection custody")
        expected = {
            "archive_bytes": arm["archive"]["bytes"],
            "archive_sha256": arm["archive"]["sha256"],
            "d_seg": hard_oracle["mean_d_seg"],
            "d_pose": hard_oracle["mean_d_pose"],
            "derived_n600_linear_archive_bytes": projection["projected_n600_archive_bytes"],
            "derived_n600_action_at_unchanged_mean_distortion": projection["projected_exact_objective"],
            "strict_bytes_to_beat_pointer_at_measured_distortion": projection[
                "strict_pointer_byte_cap_at_measured_distortion"
            ],
        }
        if any(chart.get(key) != value for key, value in expected.items()):
            raise ValueError(f"frontier chart drifted from banked v3 arm {receipt_id}")
    receipt_delta = receipt.get("treatment_minus_control")
    chart_delta = chart_rows["banked_n12_scorer_plane_precision_drop1"].get("treatment_minus_control")
    if (
        not isinstance(receipt_delta, dict)
        or not isinstance(chart_delta, dict)
        or chart_delta
        != {
            "archive_bytes": receipt_delta["archive_bytes"],
            "d_seg": receipt_delta["mean_d_seg"],
            "d_pose": receipt_delta["mean_d_pose"],
            "derived_n600_action": receipt_delta["projected_exact_objective"],
        }
    ):
        raise ValueError("frontier chart treatment delta drifted from banked v3 receipt")


def validate_frontier_magnitude_chart(
    path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
) -> dict:
    """Validate the exact-solver frontier chart and every score-law projection.

    The chart intentionally mixes separately labelled evidence axes. Validation
    checks arithmetic and scope; it does not promote local rows to contest
    authority or turn linear byte projections into archives.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "einstein_kolmogorov_frontier_magnitude_chart.v1":
        raise ValueError("unexpected frontier-magnitude chart schema")
    if (
        payload.get("research_only") is not True
        or payload.get("score_claim") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("frontier-magnitude chart must remain research-only and non-promotable")
    pointer = payload.get("pointer")
    if not isinstance(pointer, dict) or pointer.get("moved") is not False:
        raise ValueError("frontier-magnitude chart must preserve the pointer")
    target = _nonnegative_real(pointer.get("score"), "pointer.score")
    source_rows = _validate_source_receipts(payload)

    by_id = _rows_by_exact_id(
        payload.get("exact_archive_rows"),
        expected_ids=EXPECTED_EXACT_ARCHIVE_ROW_IDS,
        label="exact archive chart",
    )
    bank = by_id.get("c1_solved_distortion_n600_contest_cpu")
    rung_e = by_id.get("v10_rung_e_exact_two_plane_n48_local")
    banked_control = by_id.get("banked_n12_exact_receiver_control")
    banked_treatment = by_id.get("banked_n12_scorer_plane_precision_drop1")
    if not all(isinstance(row, dict) for row in (bank, rung_e, banked_control, banked_treatment)):
        raise ValueError("frontier-magnitude chart lacks an exact receiver control/treatment row")
    _validate_c1_source_custody(bank, source_rows=source_rows)
    _validate_c1_rounded_interval(bank, target=target)
    _validate_rung_e_source(rung_e, source_rows=source_rows, target=target)
    _validate_banked_v3_receipt(source_rows=source_rows, chart_rows=by_id)
    rung_projected_bytes = _nonnegative_bytes(
        rung_e.get("derived_n600_linear_archive_bytes"), "rung_e.derived_n600_linear_archive_bytes"
    )
    rung_projected_action = contest_action(
        d_seg=rung_e.get("d_seg"),
        d_pose=rung_e.get("d_pose"),
        archive_bytes=rung_projected_bytes,
    )
    if not math.isclose(
        rung_projected_action,
        rung_e.get("derived_n600_action_at_unchanged_mean_distortion"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("rung-E projected action arithmetic drift")
    if maximum_byte_budget(target_action=target, d_seg=rung_e["d_seg"], d_pose=rung_e["d_pose"]) != rung_e.get(
        "strict_bytes_to_beat_pointer_at_measured_distortion"
    ):
        raise ValueError("rung-E strict byte cap drift")
    for row, frontier_relevant in ((banked_control, True), (banked_treatment, False)):
        pair_count = _nonnegative_bytes(row.get("pair_count"), f"{row.get('point_id')}.pair_count")
        if pair_count == 0:
            raise ValueError("banked receiver row pair count must be positive")
        archive_bytes = _nonnegative_bytes(row.get("archive_bytes"), f"{row.get('point_id')}.archive_bytes")
        projected_bytes = (archive_bytes * 600 + pair_count - 1) // pair_count
        if projected_bytes != row.get("derived_n600_linear_archive_bytes"):
            raise ValueError(f"{row.get('point_id')} projected archive-byte arithmetic drift")
        action = contest_action(
            d_seg=row.get("d_seg"),
            d_pose=row.get("d_pose"),
            archive_bytes=projected_bytes,
        )
        if not math.isclose(
            action,
            row.get("derived_n600_action_at_unchanged_mean_distortion"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{row.get('point_id')} projected action arithmetic drift")
        if row.get("frontier_relevant_distortion") is not frontier_relevant:
            raise ValueError(f"{row.get('point_id')} frontier-distortion gate drift")
        if frontier_relevant:
            cap = maximum_byte_budget(target_action=target, d_seg=row["d_seg"], d_pose=row["d_pose"])
            if cap != row.get("strict_bytes_to_beat_pointer_at_measured_distortion"):
                raise ValueError(f"{row.get('point_id')} strict byte cap drift")
        elif (
            row.get("strict_bytes_to_beat_pointer_at_measured_distortion") != -1
            or contest_action(d_seg=row["d_seg"], d_pose=row["d_pose"], archive_bytes=0) < target
        ):
            raise ValueError(f"{row.get('point_id')} infeasible distortion classification drift")
    treatment_delta = banked_treatment.get("treatment_minus_control")
    if not isinstance(treatment_delta, dict) or (
        treatment_delta.get("archive_bytes") != banked_treatment["archive_bytes"] - banked_control["archive_bytes"]
        or not math.isclose(
            treatment_delta.get("d_seg"),
            banked_treatment["d_seg"] - banked_control["d_seg"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            treatment_delta.get("d_pose"),
            banked_treatment["d_pose"] - banked_control["d_pose"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            treatment_delta.get("derived_n600_action"),
            banked_treatment["derived_n600_action_at_unchanged_mean_distortion"]
            - banked_control["derived_n600_action_at_unchanged_mean_distortion"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("banked receiver matched A/B delta drift")

    trade = payload.get("trade_cells_curve")
    points = trade.get("points") if isinstance(trade, dict) else None
    trade_by_id = _validate_trade_source(points, source_rows=source_rows)
    if not isinstance(points, list):  # Kept explicit for static narrowing after fail-closed validation.
        raise ValueError("frontier-magnitude chart lacks trade-cells points")
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("trade-cells points must be mappings")
        projected_bytes = _project_integral_population_bytes(
            mean_bytes_per_pair=point.get("measured_mean_payload_bytes_per_pair"),
            pair_count=600,
        )
        if projected_bytes != point.get("derived_n600_payload_bytes"):
            raise ValueError(f"{point.get('point_id')} projected byte arithmetic drift")
        projected_action = contest_action(
            d_seg=point.get("d_seg"),
            d_pose=point.get("d_pose"),
            archive_bytes=projected_bytes,
        )
        if not math.isclose(
            projected_action,
            point.get("derived_action_on_declared_payload_scope"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{point.get('point_id')} projected action arithmetic drift")
        feasible = frontier_feasible_at_zero_pose_and_rate(d_seg=point.get("d_seg"), target_action=target)
        if feasible is not point.get("seg_only_frontier_necessary_gate"):
            raise ValueError(f"{point.get('point_id')} Seg-only gate drift")
    lead = trade_by_id.get("precision_drop1")
    if not isinstance(lead, dict) or lead.get("pose_violation_count") != 0:
        raise ValueError("frontier-magnitude trade-cells lead must retain the pose-clean control")

    _validate_inverse_sources(
        payload.get("exact_inverse_nonarchive_rows"),
        source_rows=source_rows,
        target=target,
    )

    gap = payload.get("exact_production_gap")
    if (
        not isinstance(gap, dict)
        or gap.get("blocker") != "NO_COMPLETE_N600_ARCHIVE_WITHIN_TOTAL_SCORE_BYTE_CAP"
        or gap.get("secondary_blocker") != "MISSING_ARBITRARY_NUMERATOR_PLANE_CODEC"
        or gap.get("secondary_blocker_condition")
        != "applies only if the joint interval-solver arbitrary-numerator representation is selected"
        or gap.get("rate_subproblem")
        != "compact predictor/program after debiting all fixed runtime, archive-framing, and Pose overhead"
    ):
        raise ValueError("frontier-magnitude chart must fail closed on the total-archive cap and conditional ABI gap")
    if payload.get("n600_trade_cells_launch_eligible") is not False:
        raise ValueError("frontier-magnitude chart cannot authorize an n600 trade-cells launch")
    return payload


def build_einstein_kolmogorov_crux_action_rate_contract_v1(
    *,
    measurement_path: str | Path = SOURCE_MEASUREMENT,
    frontier_chart_path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
) -> CanonicalEquation:
    """Build the hash-bound, research-only canonical equation and n24 anchor."""

    payload, source, winner = _load_scoped_measurement(measurement_path)
    frontier_chart = validate_frontier_magnitude_chart(frontier_chart_path)
    correction = payload["operating_point_correction"]
    measurement_path_str = str(measurement_path)
    measured_utc = str(payload["utc"])
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=measurement_path,
        reactivation_criteria=(
            "Research-only n24 fixed-label palette evidence. Reactivate for promotion only "
            "after a complete n600 archive is byte-closed and scored on a contest axis."
        ),
        measurement_axis=RESEARCH_ONLY_AXIS,
        hardware_substrate="macos_arm64",
        captured_at_utc=measured_utc,
    )
    frontier_utc = str(frontier_chart["written_at_utc"])
    frontier_provenance = build_provenance_for_research_sidecar(
        sidecar_path=frontier_chart_path,
        reactivation_criteria=(
            "Cross-axis exact-solver reuse and projected rate controls only. Reactivate "
            "for promotion after a complete n600 trade-cells archive is byte-closed and "
            "scored on a contest axis."
        ),
        measurement_axis="[cross-axis exact receipts; axes remain separate]",
        hardware_substrate="mixed; see frontier chart row custody",
        captured_at_utc=frontier_utc,
    )
    anchor = EmpiricalAnchor(
        anchor_id="einstein_kolmogorov_n24_fixed_label_palette_20260719",
        measurement_utc=measured_utc,
        inputs={
            "scope": payload["verdict_scope"],
            "source_candidate_sha256": source["candidate_sha256"],
            "source_packet_bytes": source["candidate_bytes"],
            "pair_count": payload["inputs"]["pair_count"],
            "scorer_pixels": payload["inputs"]["scorer_pixels"],
        },
        predicted_output={
            "fixed_byte_palette_winner_strictly_improves_source": True,
            "winner_frontier_feasible_at_zero_pose_zero_rate": False,
            "full_archive_or_contest_score_claim": False,
        },
        empirical_output={
            "source_hard_mismatch_px": source["hard_mismatch_px"],
            "source_d_seg": source["d_seg"],
            "winner_hard_mismatch_px": winner["hard_mismatch_px"],
            "winner_d_seg": winner["d_seg"],
            "winner_candidate_bytes": winner["candidate_bytes"],
            "winner_candidate_sha256": winner["candidate_sha256"],
            "winner_seg_term": SEGMENTATION_WEIGHT * winner["d_seg"],
            "target_action": correction["target_action"],
            "winner_frontier_feasible_at_zero_pose_zero_rate": False,
            "operating_point_verdict": correction["verdict"],
            "full_archive_or_contest_score_claim": False,
        },
        residual=0.0,
        source_artifact=measurement_path_str,
        measurement_method=(
            "PDW1 encode/decode/re-encode, factor-2 uint8 realization certificate, "
            "and singleton frozen CPU-Torch SegNet over all 24 packet pairs"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    frontier_rows = {row["point_id"]: row for row in frontier_chart["exact_archive_rows"]}
    trade_rows = {row["point_id"]: row for row in frontier_chart["trade_cells_curve"]["points"]}
    frontier_anchor = EmpiricalAnchor(
        anchor_id="einstein_kolmogorov_exact_solver_frontier_magnitude_20260720",
        measurement_utc=frontier_utc,
        inputs={
            "pointer": frontier_chart["pointer"],
            "exact_solver_receipt_count": len(frontier_chart["source_receipts"]),
            "trade_cells_point_count": len(trade_rows),
            "axes_are_not_promoted_or_inferred_equivalent": True,
        },
        predicted_output={
            "frontier_magnitude_exact_receiver_control_exists": True,
            "matched_receiver_positive_band_control_exists": True,
            "frontier_magnitude_pose_clean_trade_cells_control_exists": True,
            "complete_n600_in_box_archive_exists": False,
            "n600_trade_cells_launch_eligible": False,
        },
        empirical_output={
            "bank": {
                "archive_bytes": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["archive_bytes"],
                "d_seg": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["d_seg"],
                "d_pose": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["d_pose"],
                "derived_action_interval_from_rounded_components": frontier_rows[
                    "c1_solved_distortion_n600_contest_cpu"
                ]["derived_action_interval_from_rounded_components"],
                "derived_strict_total_archive_cap": frontier_rows["c1_solved_distortion_n600_contest_cpu"][
                    "derived_strict_total_archive_cap"
                ],
            },
            "exact_receiver_control": {
                "pair_count": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["pair_count"],
                "archive_bytes": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["archive_bytes"],
                "d_seg": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["d_seg"],
                "d_pose": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["d_pose"],
            },
            "matched_receiver_control": {
                "pair_count": frontier_rows["banked_n12_exact_receiver_control"]["pair_count"],
                "archive_bytes": frontier_rows["banked_n12_exact_receiver_control"]["archive_bytes"],
                "d_seg": frontier_rows["banked_n12_exact_receiver_control"]["d_seg"],
                "d_pose": frontier_rows["banked_n12_exact_receiver_control"]["d_pose"],
            },
            "matched_receiver_treatment": {
                "point_id": "banked_n12_scorer_plane_precision_drop1",
                "archive_bytes": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["archive_bytes"],
                "d_seg": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["d_seg"],
                "d_pose": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["d_pose"],
                "derived_n600_action": frontier_rows["banked_n12_scorer_plane_precision_drop1"][
                    "derived_n600_action_at_unchanged_mean_distortion"
                ],
            },
            "pose_clean_trade_cells_control": {
                "point_id": "precision_drop1",
                "d_seg": trade_rows["precision_drop1"]["d_seg"],
                "d_pose": trade_rows["precision_drop1"]["d_pose"],
                "derived_n600_payload_bytes": trade_rows["precision_drop1"]["derived_n600_payload_bytes"],
                "derived_action_on_declared_payload_scope": trade_rows["precision_drop1"][
                    "derived_action_on_declared_payload_scope"
                ],
            },
            "exact_production_blocker": frontier_chart["exact_production_gap"]["blocker"],
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=str(frontier_chart_path),
        measurement_method=(
            "Read-only SHA revalidation, matched n12 production receiver A/B, and exact "
            "score-law composition of the banked C1, Rung-E, zero-band joint-solve, "
            "exact-lattice, and trade-cells receipts"
        ),
        provenance=frontier_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Einstein--Kolmogorov scoped action/rate contract",
        one_line_summary=(
            "Exact contest action arithmetic with a hash-bound research-only n24 "
            "fixed-label wall plus matched exact-receiver frontier-magnitude anchor."
        ),
        latex_form=(
            r"S=100D_{seg}+\sqrt{10D_{pose}}+25B/37545489,\quad "
            r"B^{<}_{max}=\left\lceil(S_t-100D_{seg}-\sqrt{10D_{pose}})37545489/25\right\rceil-1"
        ),
        python_callable_module_path=("tac.canonical_equations.einstein_kolmogorov_crux_20260719:contest_action"),
        domain_of_validity={
            "action_contract": "frozen comma contest score arithmetic",
            "empirical_anchor_scope": payload["verdict_scope"],
            "anchor_measurement_sha256": provenance.source_sha256,
            "research_only": True,
            "promotion_eligible": False,
            "full_archive_claim": False,
            "operating_point_verdict": correction["verdict"],
            "n600_explicit_target_launch_eligible": False,
            "frontier_magnitude_chart_sha256": frontier_provenance.source_sha256,
            "frontier_magnitude_exact_production_blocker": frontier_chart["exact_production_gap"]["blocker"],
            "n600_trade_cells_launch_eligible": False,
        },
        units_in={
            "d_seg": "fraction",
            "d_pose": "mean squared error",
            "archive_bytes": "bytes",
        },
        units_out={"action": "score units", "maximum_byte_budget": "bytes"},
        empirical_anchors=(anchor, frontier_anchor),
        predicted_vs_empirical_residual={
            "fixed_byte_palette_no_regression": 0.0,
            "frontier_magnitude_chart_arithmetic": 0.0,
        },
        last_calibration_utc=frontier_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_einstein_kolmogorov_crux",
            "tac.optimization.einstein_kolmogorov_crux",
        ),
        canonical_producers=(measurement_path_str, str(frontier_chart_path)),
        provenance=provenance,
    )


def populate_einstein_kolmogorov_crux_action_rate_contract_v1(
    *,
    measurement_path: str | Path = SOURCE_MEASUREMENT,
    frontier_chart_path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
    path=None,
    lock_path=None,
    agent=None,
    subagent_id=None,
) -> CanonicalEquation:
    """Append the scoped law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1(
        measurement_path=measurement_path,
        frontier_chart_path=frontier_chart_path,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="n24 fixed-label wall plus exact-receiver frontier-magnitude A/B; research-only and non-promotable",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "SOURCE_FRONTIER_MAGNITUDE",
    "SOURCE_MEASUREMENT",
    "InfeasibleByteBudgetError",
    "MeasuredHardRReceipt",
    "ResearchOnlyDecision",
    "build_einstein_kolmogorov_crux_action_rate_contract_v1",
    "contest_action",
    "derive_research_only_decision",
    "fixed_byte_palette_delta",
    "frontier_feasible_at_zero_pose_and_rate",
    "inclusive_maximum_byte_budget",
    "maximum_byte_budget",
    "populate_einstein_kolmogorov_crux_action_rate_contract_v1",
    "validate_frontier_magnitude_chart",
]
