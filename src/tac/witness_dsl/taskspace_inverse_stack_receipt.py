"""Fail-closed research receipt for the task-space inverse codec stack.

The receipt joins the existing encoder-side teacher census and prior-signal
harvest to the live canonical frontier pointer and the canonical
equation/DAG/index/autopilot apparatus.  It does not build a candidate.  In
particular, an exact teacher output is not candidate payload lineage, n64 is a
bounded acquisition/timing diagnostic only, and no originality claim exists
without current-stack ``borrowed_substrate_accounting``.

The stack vocabulary is explicit:

``P``
    counted predictor program;
``G``
    counted compact task-space correction parameters;
``A``
    counted compact preimage controls producing distinct Y0/Y1;
``T``
    optional counted irreducible terminal quotient, admitted only after
    measured same-object matched-byte P/G/A controls fail on total score;
``E``
    encoder-only teacher/evaluator truth, never candidate payload.

This module is deterministic and scorer-free.  The canonical CLI performs the
expensive strict source reopen before asking this module to publish a receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tac.canonical_frontier_pointer import (
    POINTER_SCHEMA_VERSION,
    POINTER_STALE_SECONDS,
    CanonicalFrontierPointer,
    effective_frontier_score,
    load_canonical_frontier_pointer_strict,
    recompute_effective_frontier,
)
from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    RATE_COEFFICIENT,
    contest_score,
    target_byte_budget_for_score,
)

SCHEMA: Final = "tac.taskspace_inverse_codec_stack.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_inverse_codec_stack_receipt.v1"
TEACHER_RECEIPT_SCHEMA: Final = "tac.g1_teacher_atom_census_receipt.v1"
TEACHER_BODY_SCHEMA: Final = "tac.g1_teacher_atom_census.v1"
HARVEST_RECEIPT_SCHEMA: Final = "tac.g1_prior_signal_harvest_receipt.v1"
HARVEST_BODY_SCHEMA: Final = "tac.g1_prior_signal_harvest.v1"
C1_SCHEMA: Final = "tac.coupled_witness_raw_debt.v2"
C1_CANONICAL_FILE_SHA256: Final = "0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3"
C1_CANONICAL_AXIS: Final = "[Darwin-arm64 CPU advisory] NON-PROMOTABLE"
LANE_DAG_RECEIPT_SCHEMA: Final = "tac.original_taskspace_inverse_witness_codec_roadmap_receipt.v4"
LANE_DAG_BODY_SCHEMA: Final = "tac.original_taskspace_inverse_witness_codec_roadmap.v4"
ACCOUNTING_SCHEMA: Final = "tac.taskspace_inverse_borrowed_substrate_accounting.v1"

RESEARCH_ROOT: Final = Path(".omx/research/original_taskspace_inverse_witness_codec_20260725")
DEFAULT_TEACHER_CENSUS: Final = RESEARCH_ROOT / "g1_teacher_atom_census_n64_20260726.json"
DEFAULT_PRIOR_HARVEST: Final = RESEARCH_ROOT / "g1_prior_signal_harvest_v1.json"
DEFAULT_C1_ANCHOR: Final = RESEARCH_ROOT / "c1_live_target_debt_n600_batch16.json"
DEFAULT_LANE_DAG: Final = RESEARCH_ROOT / "roadmap_v4.json"
DEFAULT_HISTORICAL_ACCOUNTING: Final = RESEARCH_ROOT / "spine_refresh.json"
DEFAULT_CANONICAL_DAG: Final = Path(".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md")
DEFAULT_RESEARCH_INDEX: Final = Path(".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md")
DEFAULT_FRONTIER_POINTER: Final = Path(".omx/state/canonical_frontier_pointer.json")
DEFAULT_EQUATION_REGISTRY: Final = Path(".omx/state/canonical_equations_registry.jsonl")
DEFAULT_AUTOPILOT: Final = Path("tools/cathedral_autopilot.py")
DEFAULT_G_SOURCE: Final = Path("src/tac/witness_dsl/generative_taskspace_correction.py")
DEFAULT_PAIR_SOURCE: Final = Path("src/tac/witness_dsl/pair_population_envelope.py")
DEFAULT_A_SOURCE: Final = Path("src/tac/witness_dsl/coupled_preimage_program.py")
DEFAULT_OUTPUT: Final = RESEARCH_ROOT / "taskspace_inverse_codec_stack_receipt_v1.json"

REQUIRED_EQUATION_IDS: Final = (
    "ddm_receiver_support_pf2_causal_intersection_v1",
    "ddm_score_quotient_functional_v1",
    "predict_project_realization_admissibility_v1",
)
BASE_EXACT_BLOCKERS: Final = (
    "receiver_consumption_custody_absent",
    "n600_lineage_clean_p_g_a_optional_t_same_object_receipt_absent",
    "n600_receiver_closed_standalone_archive_absent",
    "reverse_causal_A_canonical_instance_packet_and_compile_receipt_absent",
    "reverse_causal_A_packet_not_yet_adapted_into_PairPopulation_counted_sections",
    "reverse_causal_A_complete_typed_control_derivation_and_archive_lineage_absent",
    "frame1_preimage_production_archive_parser_binding_absent",
    "frame0_y0_fibre_production_archive_parser_binding_absent",
    "complete_archive_payload_lineage_not_closed",
    "complete_candidate_archive_borrowed_substrate_accounting_absent",
    "standalone_clean_root_double_inflate_custody_absent",
    "exact_same_archive_contest_cpu_and_cuda_receipts_absent",
)
NEXT_EXECUTABLE_EDGE: Final = (
    "land the governed archive-member_to_strict-G-decode_to_evaluated-raw-output receiver-consumption receipt "
    "plus a matched G-only counterfactual; in parallel adapt the reverse-causal A packet into PairPopulation "
    "counted sections, then build one lineage-clean n600 P/G/A plus optional measured T same-object standalone "
    "archive receipt; n64 may time/acquire only"
)

_BODY_KEYS: Final = frozenset(
    {
        "schema",
        "authority",
        "source_reopen",
        "source_custody",
        "frontier_join",
        "conditional_c1_n600_byte_ceiling",
        "p_g_a_t_ownership",
        "borrowed_substrate_accounting",
        "readiness",
        "lineage",
        "canonical_apparatus",
        "exact_blockers",
        "next_executable_edge",
        "verdict",
    }
)


class TaskspaceInverseStackReceiptError(ValueError):
    """Raised when a stack receipt could launder subset or payload authority."""


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for content identities and publication."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(repo_root: Path, path: Path) -> str:
    resolved = Path(os.path.abspath(path))
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_stable_bytes(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    """Read one bounded regular-file snapshot through one no-follow descriptor."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise TaskspaceInverseStackReceiptError("O_NOFOLLOW is required for source custody")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TaskspaceInverseStackReceiptError(f"source could not be opened without following links: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspaceInverseStackReceiptError(f"source must be a regular file: {path}")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise TaskspaceInverseStackReceiptError(f"source has invalid byte size: {path} ({before.st_size})")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise TaskspaceInverseStackReceiptError(f"source exceeds byte bound while reading: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or total != before.st_size:
        raise TaskspaceInverseStackReceiptError(f"source changed during descriptor read: {path}")
    return b"".join(chunks)


def _custody(repo_root: Path, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": _display_path(repo_root, path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _load_json(repo_root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    resolved = _resolve(repo_root, path)
    payload = _read_stable_bytes(resolved)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TaskspaceInverseStackReceiptError(f"invalid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise TaskspaceInverseStackReceiptError(f"JSON root must be an object: {resolved}")
    return value, _custody(repo_root, resolved, payload), payload


def _load_jsonl(repo_root: Path, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    resolved = _resolve(repo_root, path)
    payload = _read_stable_bytes(resolved)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskspaceInverseStackReceiptError(f"invalid UTF-8 JSONL: {resolved}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise TaskspaceInverseStackReceiptError(f"invalid JSONL at {resolved}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TaskspaceInverseStackReceiptError(f"JSONL row must be an object: {resolved}:{line_number}")
        rows.append(row)
    return rows, _custody(repo_root, resolved, payload), payload


def _exact_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskspaceInverseStackReceiptError(f"{label} must be an object")
    return value


def _finite_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskspaceInverseStackReceiptError(f"{label} must be an exact JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TaskspaceInverseStackReceiptError(f"{label} must be finite and nonnegative")
    return result


def _strict_max_archive_bytes(*, target_score: float, d_seg: float, d_pose: float) -> int | None:
    """Largest nonnegative integer B whose fully recomposed score is strict.

    ``ceil(open_boundary)-1`` is only a provisional integer candidate: binary
    floating cancellation can otherwise return a byte count whose recomposed
    score is exactly equal to ``target_score``.  Ratcheting against the same
    contest-score implementation used by admission preserves the open level
    set rather than three independently rounded component constraints.
    """

    target = _finite_nonnegative_number(target_score, "strict target score")
    seg = _finite_nonnegative_number(d_seg, "strict d_seg")
    pose = _finite_nonnegative_number(d_pose, "strict d_pose")
    distortion = contest_score(seg, pose, 0)
    rate_term_budget = target - distortion
    if rate_term_budget <= 0.0:
        return None
    open_boundary = rate_term_budget * CONTEST_REFERENCE_BYTES / RATE_COEFFICIENT
    candidate = math.ceil(open_boundary) - 1

    # A correct provisional boundary is at most a few ulps/bytes away.  Refuse
    # a numerically incoherent surface rather than loop over an unbounded range.
    for _ in range(16):
        if candidate < 0:
            return None
        if contest_score(seg, pose, candidate) < target:
            break
        candidate -= 1
    else:
        raise TaskspaceInverseStackReceiptError("strict byte ceiling failed to close below target")
    for _ in range(16):
        if contest_score(seg, pose, candidate + 1) >= target:
            return candidate
        candidate += 1
    raise TaskspaceInverseStackReceiptError("strict byte ceiling failed to reach the maximal strict integer")


def _timestamp_is_stale(value: Any, *, now: datetime | None = None) -> bool:
    if not isinstance(value, str) or not value:
        return True
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    age_seconds = (current - observed).total_seconds()
    return age_seconds < -300 or age_seconds > POINTER_STALE_SECONDS


def _validated_live_effective_frontier(pointer: CanonicalFrontierPointer) -> Mapping[str, Any]:
    """Require exact min-recomposition plus a current official snapshot."""

    expected = recompute_effective_frontier(pointer)
    actual = pointer.effective_frontier
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise TaskspaceInverseStackReceiptError("canonical pointer has no recomposable effective frontier")
    if canonical_json_bytes(expected) != canonical_json_bytes(actual):
        raise TaskspaceInverseStackReceiptError(
            "serialized effective frontier differs from constituent minimum; refresh pointer"
        )
    # A fresh local wrapper timestamp cannot prove that no newer public score
    # exists.  Competitive planning therefore requires the official snapshot
    # used in the min to be independently current.
    if _timestamp_is_stale(pointer.upstream_leaderboard_snapshot_at_utc):
        raise TaskspaceInverseStackReceiptError(
            "official leaderboard snapshot is stale; refresh pointer before admission planning"
        )
    if actual.get("source") == "upstream_official_leaderboard" and (
        actual.get("snapshot_at_utc") != pointer.upstream_leaderboard_snapshot_at_utc
    ):
        raise TaskspaceInverseStackReceiptError("effective frontier winning snapshot timestamp differs")
    return actual


def _validate_c1_historical_anchor(
    *,
    repo_root: Path,
    path: Path,
    payload: bytes,
    receipt: Mapping[str, Any],
) -> None:
    """Admit only the one immutable historical v2 n600 measurement."""

    resolved = Path(os.path.abspath(_resolve(repo_root, path)))
    expected_path = Path(os.path.abspath(repo_root / DEFAULT_C1_ANCHOR))
    if resolved != expected_path or _sha256(payload) != C1_CANONICAL_FILE_SHA256:
        raise TaskspaceInverseStackReceiptError(
            "C1 v2 is admissible only at its canonical path and immutable file SHA-256"
        )
    if receipt.get("schema") != C1_SCHEMA:
        raise TaskspaceInverseStackReceiptError("C1 n600 anchor schema differs")
    _require_false_authority(receipt, "C1 n600 anchor")
    if receipt.get("axis") != C1_CANONICAL_AXIS:
        raise TaskspaceInverseStackReceiptError("C1 n600 anchor advisory axis differs")
    scorer_custody = _exact_mapping(receipt.get("scorer_custody"), "C1 scorer custody")
    if scorer_custody.get("device") != "cpu" or scorer_custody.get("deterministic_algorithms") is not True:
        raise TaskspaceInverseStackReceiptError("C1 scorer CPU/determinism custody differs")


def _current_stack_borrowed_substrate_accounting() -> dict[str, Any]:
    zero_payload_bytes = {
        "candidate_archive_bytes": 0,
        "candidate_source_bytes": 0,
        "candidate_weights_bytes": 0,
        "candidate_latents_bytes": 0,
        "candidate_selectors_bytes": 0,
        "candidate_payload_bytes": 0,
    }
    return {
        "schema": ACCOUNTING_SCHEMA,
        "scope": "current_research_stack_before_candidate_materialization",
        "candidate_archive_exists": False,
        "complete_candidate_archive_accounted": False,
        "named_input_inventory_present": True,
        "candidate_payload_bytes_total": 0,
        "inherited_candidate_payload_bytes_total": 0,
        "rows": [
            {
                "item_id": "hnerv_pr130",
                "provenance_class": "borrowed_external",
                "allowed_role": "research_and_mechanism_reference_only",
                "candidate_payload_included": False,
                **zero_payload_bytes,
            },
            {
                "item_id": "quarantined_149f_donor",
                "provenance_class": "borrowed_quarantined_signal",
                "allowed_role": "signal_only",
                "candidate_payload_included": False,
                **zero_payload_bytes,
            },
            {
                "item_id": "v9_v10_c2_internal",
                "provenance_class": "our_original_internal_code_and_mechanisms",
                "allowed_role": "current_stack_mechanism_source_not_complete_candidate",
                "candidate_payload_included": False,
                "inherited_candidate_bytes": 0,
                **zero_payload_bytes,
            },
            {
                "item_id": "pbr_gt_oracle_dense_teacher_E",
                "provenance_class": "target_derived_encoder_only_truth",
                "allowed_role": "encoder_only_teacher_evaluator_truth_never_ships",
                "candidate_payload_included": False,
                "candidate_payload_forbidden": True,
                **zero_payload_bytes,
            },
        ],
        "originality_proven": False,
        "originality_gate": "rebuild_accounting_from_every_byte_of_one_complete_candidate_archive",
    }


def _require_false_authority(body: Mapping[str, Any], label: str) -> None:
    expected = {
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if any(body.get(key) is not value for key, value in expected.items()):
        raise TaskspaceInverseStackReceiptError(f"{label} authority boundary differs")


def _validate_envelope(
    envelope: Mapping[str, Any],
    *,
    receipt_schema: str,
    body_schema: str,
    label: str,
) -> Mapping[str, Any]:
    if envelope.get("schema") != receipt_schema:
        raise TaskspaceInverseStackReceiptError(f"{label} receipt schema differs")
    body = _exact_mapping(envelope.get("body"), f"{label} body")
    if body.get("schema") != body_schema:
        raise TaskspaceInverseStackReceiptError(f"{label} body schema differs")
    if envelope.get("body_sha256") != _sha256(canonical_json_bytes(body)):
        raise TaskspaceInverseStackReceiptError(f"{label} body hash differs")
    _require_false_authority(body, label)
    return body


def _strict_reopen_sources(teacher: Mapping[str, Any], harvest: Mapping[str, Any]) -> None:
    """Invoke the canonical producers' full source-custody validators."""

    # Lazy imports keep this receipt module scorer-free and avoid imposing the
    # teacher census' NumPy receiver stack on ordinary importers.
    from tac.witness_dsl.g1_prior_signal_harvest import validate_envelope as validate_harvest
    from tools.measure_g1_teacher_atom_census import validate_receipt as validate_teacher

    validate_teacher(teacher, reopen_sources=True)
    validate_harvest(harvest, reopen_sources=True)


def _latest_equation_ids(events: list[dict[str, Any]]) -> tuple[str, ...]:
    latest: dict[str, Mapping[str, Any]] = {}
    for event in events:
        equation_id = event.get("equation_id")
        if isinstance(equation_id, str) and equation_id:
            latest[equation_id] = event
    return tuple(sorted(latest))


def _source_snapshot(repo_root: Path, path: Path) -> tuple[dict[str, Any], bytes]:
    resolved = _resolve(repo_root, path)
    payload = _read_stable_bytes(resolved)
    return _custody(repo_root, resolved, payload), payload


def build_stack_receipt(
    *,
    repo_root: Path,
    teacher_census_path: Path = DEFAULT_TEACHER_CENSUS,
    prior_harvest_path: Path = DEFAULT_PRIOR_HARVEST,
    c1_anchor_path: Path = DEFAULT_C1_ANCHOR,
    frontier_pointer_path: Path = DEFAULT_FRONTIER_POINTER,
    equation_registry_path: Path = DEFAULT_EQUATION_REGISTRY,
    lane_dag_path: Path = DEFAULT_LANE_DAG,
    canonical_dag_path: Path = DEFAULT_CANONICAL_DAG,
    research_index_path: Path = DEFAULT_RESEARCH_INDEX,
    autopilot_path: Path = DEFAULT_AUTOPILOT,
    g_source_path: Path = DEFAULT_G_SOURCE,
    pair_source_path: Path = DEFAULT_PAIR_SOURCE,
    a_source_path: Path = DEFAULT_A_SOURCE,
    historical_accounting_path: Path = DEFAULT_HISTORICAL_ACCOUNTING,
    strict_source_reopen: bool = True,
) -> dict[str, Any]:
    """Build one deterministic, non-promotable P/G/A/T stack receipt."""

    root = repo_root.resolve()
    teacher, teacher_custody, _ = _load_json(root, teacher_census_path)
    harvest, harvest_custody, _ = _load_json(root, prior_harvest_path)
    teacher_body = _validate_envelope(
        teacher,
        receipt_schema=TEACHER_RECEIPT_SCHEMA,
        body_schema=TEACHER_BODY_SCHEMA,
        label="teacher census",
    )
    harvest_body = _validate_envelope(
        harvest,
        receipt_schema=HARVEST_RECEIPT_SCHEMA,
        body_schema=HARVEST_BODY_SCHEMA,
        label="prior harvest",
    )
    if teacher_body.get("candidate_payload_allowed") is not False:
        raise TaskspaceInverseStackReceiptError("teacher census must remain candidate-forbidden")
    if harvest_body.get("candidate_payload_emitted") is not False:
        raise TaskspaceInverseStackReceiptError("prior harvest must not emit candidate payload")
    measurement = _exact_mapping(teacher_body.get("measurement"), "teacher measurement")
    per_pair = measurement.get("per_pair")
    if not isinstance(per_pair, list) or len(per_pair) != 64:
        raise TaskspaceInverseStackReceiptError("teacher census must remain the bounded n64 diagnostic")
    if strict_source_reopen:
        _strict_reopen_sources(teacher, harvest)

    c1, c1_custody, c1_payload = _load_json(root, c1_anchor_path)
    _validate_c1_historical_anchor(
        repo_root=root,
        path=c1_anchor_path,
        payload=c1_payload,
        receipt=c1,
    )
    c1_aggregate = _exact_mapping(c1.get("aggregate"), "C1 aggregate")
    if c1_aggregate.get("pair_count") != 600:
        raise TaskspaceInverseStackReceiptError("C1 anchor must cover the n600 decision surface")
    d_seg = _finite_nonnegative_number(c1_aggregate.get("mean_d_seg"), "C1 mean_d_seg")
    d_pose = _finite_nonnegative_number(c1_aggregate.get("mean_d_pose"), "C1 mean_d_pose")

    pointer_path = _resolve(root, frontier_pointer_path)
    loaded_pointer = load_canonical_frontier_pointer_strict(repo_root=root, path=pointer_path)
    if loaded_pointer.is_stale():
        raise TaskspaceInverseStackReceiptError(
            "canonical frontier pointer is stale; refresh before admission planning"
        )
    pointer_payload = _read_stable_bytes(pointer_path)
    try:
        pointer_data = json.loads(pointer_payload)
    except json.JSONDecodeError as exc:
        raise TaskspaceInverseStackReceiptError("canonical frontier pointer JSON differs") from exc
    if not isinstance(pointer_data, Mapping):
        raise TaskspaceInverseStackReceiptError("canonical frontier pointer root must be an object")
    pointer = CanonicalFrontierPointer.from_dict(pointer_data)
    if pointer.as_dict() != loaded_pointer.as_dict():
        raise TaskspaceInverseStackReceiptError(
            "canonical frontier pointer changed between strict load and custody read"
        )
    if pointer.is_stale():
        raise TaskspaceInverseStackReceiptError("custodied canonical frontier pointer is stale")
    if pointer.schema_version != POINTER_SCHEMA_VERSION:
        raise TaskspaceInverseStackReceiptError("canonical frontier pointer schema differs")
    effective_frontier = _validated_live_effective_frontier(pointer)
    target_score = effective_frontier_score(pointer)
    if target_score is None:
        raise TaskspaceInverseStackReceiptError("canonical pointer has no effective frontier")
    if target_score != _finite_nonnegative_number(effective_frontier.get("score"), "effective frontier score"):
        raise TaskspaceInverseStackReceiptError("effective frontier score differs after min recomposition")
    budget = target_byte_budget_for_score(
        target_score=target_score,
        d_seg_floor=d_seg,
        d_pose_floor=d_pose,
    )
    strict_max_archive_bytes = _strict_max_archive_bytes(
        target_score=target_score,
        d_seg=d_seg,
        d_pose=d_pose,
    )
    if strict_max_archive_bytes is None:
        ceiling_status = "NO_STRICT_NONNEGATIVE_ARCHIVE_BYTE_BUDGET"
        score_at_strict_max = None
        score_at_next_byte = None
    else:
        ceiling_status = "PREDICTION_ONLY_CONDITIONAL_BYTE_CEILING"
        score_at_strict_max = contest_score(d_seg, d_pose, strict_max_archive_bytes)
        score_at_next_byte = contest_score(d_seg, d_pose, strict_max_archive_bytes + 1)

    registry_path = _resolve(root, equation_registry_path)
    registry_events, registry_custody, _ = _load_jsonl(root, registry_path)
    equation_ids = _latest_equation_ids(registry_events)
    missing_equations = tuple(sorted(set(REQUIRED_EQUATION_IDS) - set(equation_ids)))

    lane_dag, lane_dag_custody, _ = _load_json(root, lane_dag_path)
    if lane_dag.get("schema") != LANE_DAG_RECEIPT_SCHEMA:
        raise TaskspaceInverseStackReceiptError("lane DAG receipt schema differs")
    lane_body = _exact_mapping(lane_dag.get("body"), "lane DAG body")
    if lane_body.get("schema") != LANE_DAG_BODY_SCHEMA:
        raise TaskspaceInverseStackReceiptError("lane DAG body schema differs")
    if lane_dag.get("body_sha256") != _sha256(canonical_json_bytes(lane_body)):
        raise TaskspaceInverseStackReceiptError("lane DAG body hash differs")
    mission = _exact_mapping(lane_body.get("mission"), "lane DAG mission")
    ignored_fixed_target = _finite_nonnegative_number(
        mission.get("strict_authoritative_target"),
        "lane DAG fixed strict_authoritative_target",
    )
    codec_gestalt = _exact_mapping(lane_body.get("codec_gestalt"), "lane DAG codec gestalt")
    information_classes = codec_gestalt.get("information_classes")
    if not isinstance(information_classes, list):
        raise TaskspaceInverseStackReceiptError("lane DAG lacks typed information classes")
    counted_classes = [
        row
        for row in information_classes
        if isinstance(row, Mapping) and row.get("class") == "counted_decoder_sufficient_statistic"
    ]
    encoder_classes = [
        row
        for row in information_classes
        if isinstance(row, Mapping) and row.get("class") == "encoder_only_truth_and_evidence"
    ]
    if len(counted_classes) != 1 or len(encoder_classes) != 1:
        raise TaskspaceInverseStackReceiptError("lane DAG P/G/A/T and encoder-truth classes differ")
    counted_members = counted_classes[0].get("members")
    if not isinstance(counted_members, list) or "optional irreducible terminal quotient T" not in counted_members:
        raise TaskspaceInverseStackReceiptError("lane DAG terminal quotient ownership differs")
    if counted_classes[0].get("ships") is not True or encoder_classes[0].get("ships") is not False:
        raise TaskspaceInverseStackReceiptError("lane DAG shipping ownership differs")
    build_dag = lane_body.get("build_dag")
    if not isinstance(build_dag, list):
        raise TaskspaceInverseStackReceiptError("lane DAG lacks typed build_dag")
    lane_dag_ids = tuple(
        str(row.get("id")) for row in build_dag if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    )
    required_lane_nodes = ("G1-COMPILE", "PX1-TYPE", "PX1-N600", "AUTHORITY")
    if not set(required_lane_nodes).issubset(lane_dag_ids):
        raise TaskspaceInverseStackReceiptError("lane DAG lacks required P/G/A authority nodes")

    canonical_dag_custody, canonical_dag_payload = _source_snapshot(root, canonical_dag_path)
    index_custody, index_payload = _source_snapshot(root, research_index_path)
    autopilot_custody, autopilot_payload = _source_snapshot(root, autopilot_path)
    g_custody, g_payload = _source_snapshot(root, g_source_path)
    pair_custody, pair_payload = _source_snapshot(root, pair_source_path)
    a_custody, a_payload = _source_snapshot(root, a_source_path)
    accounting, accounting_custody, _ = _load_json(root, historical_accounting_path)
    prior_accounting = accounting.get("borrowed_substrate_accounting")
    prior_accounting_present = isinstance(prior_accounting, Mapping)

    marker = SCHEMA.encode("utf-8")
    canonical_dag_writeback_present = marker in canonical_dag_payload
    research_index_writeback_present = marker in index_payload
    autopilot_consumer_present = marker in autopilot_payload
    g_module_present = b"class GenerativeCorrectionProgramV1" in g_payload
    pair_envelope_present = b"class PairPopulationEnvelope" in pair_payload
    reverse_causal_a_present = all(
        marker in a_payload
        for marker in (
            b"class CoupledPreimageProgramV1",
            b"class Frame1AnchoredY0FibreControlV1",
            b"class JointSharedSkeletonTwoFibreControlV1",
            b"def parse_coupled_preimage_program",
            b"def decode_coupled_preimage_program",
        )
    )

    apparatus_blockers = []
    if missing_equations:
        apparatus_blockers.append("required_canonical_equations_missing")
    if not canonical_dag_writeback_present:
        apparatus_blockers.append("canonical_dag_writeback_for_stack_receipt_missing")
    if not research_index_writeback_present:
        apparatus_blockers.append("canonical_research_index_entry_for_stack_receipt_missing")
    if not autopilot_consumer_present:
        apparatus_blockers.append("cathedral_autopilot_consumer_for_stack_schema_missing")
    if not strict_source_reopen:
        apparatus_blockers.append("strict_teacher_and_harvest_source_reopen_not_run")

    exact_blockers = list(BASE_EXACT_BLOCKERS)
    exact_blockers.extend(apparatus_blockers)

    borrowed_substrate_accounting = _current_stack_borrowed_substrate_accounting()

    authority = {
        "research_only": True,
        "score_claim": False,
        "candidate_score": None,
        "candidate_archive_emitted": False,
        "candidate_payload_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "originality_claim": False,
        "pointer_moved": False,
        "pointer_delta": None,
        "n600_is_only_decision_surface": True,
        "n64_is_bounded_non_promotable_timing_and_acquisition_diagnostic_only": True,
    }
    body = {
        "schema": SCHEMA,
        "authority": authority,
        "source_reopen": {
            "teacher_census": "STRICT_REOPEN_PASS" if strict_source_reopen else "NOT_RUN",
            "prior_signal_harvest": "STRICT_REOPEN_PASS" if strict_source_reopen else "NOT_RUN",
            "teacher_scope_pairs": 64,
            "teacher_scope_role": "bounded_non_promotable_timing_and_acquisition_diagnostic_only",
            "decision_surface_pairs": 600,
        },
        "source_custody": {
            "teacher_census": teacher_custody,
            "prior_signal_harvest": harvest_custody,
            "c1_n600_anchor": c1_custody,
            "frontier_pointer": _custody(root, pointer_path, pointer_payload),
            "equation_registry": registry_custody,
            "lane_dag": lane_dag_custody,
            "canonical_dag": canonical_dag_custody,
            "canonical_research_index": index_custody,
            "cathedral_autopilot": autopilot_custody,
            "g_module": g_custody,
            "pair_population_module": pair_custody,
            "coupled_preimage_A_module": a_custody,
            "historical_borrowed_substrate_accounting": accounting_custody,
        },
        "frontier_join": {
            "selection_rule": effective_frontier.get("selection_rule"),
            "effective_frontier": dict(effective_frontier),
            "role": "live_competitive_target_only_no_local_archive_authority",
            "strict_admission_rule": "candidate_score_must_be_strictly_less_than_live_effective_frontier",
            "roadmap_v4_fixed_target_ignored": True,
            "ignored_field": "body.mission.strict_authoritative_target",
            "ignored_fixed_target": ignored_fixed_target,
            "candidate_score": None,
            "candidate_delta_to_frontier": None,
        },
        "conditional_c1_n600_byte_ceiling": {
            "source_pair_count": 600,
            "source_axis": c1.get("axis"),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "score_geometry_function": "tac.score_geometry.target_byte_budget_for_score",
            "target_source": "canonical_frontier_pointer.effective_frontier",
            "target_score": budget.target_score,
            "distortion_floor_score": budget.distortion_floor_score,
            "rate_term_budget": budget.rate_term_budget,
            "strict_open_byte_boundary_formula": (
                "provisional_ceil(rate_term_budget*37545489/25)-1_then_ratchet_against_full_contest_score.v1"
            ),
            "planning_helper_max_archive_bytes": budget.max_archive_bytes,
            "max_archive_bytes": strict_max_archive_bytes,
            "score_at_max_archive_bytes": score_at_strict_max,
            "score_at_next_archive_byte": score_at_next_byte,
            "max_archive_bytes_is_strict": (score_at_strict_max is not None and score_at_strict_max < target_score),
            "next_archive_byte_is_not_strict": (score_at_next_byte is not None and score_at_next_byte >= target_score),
            "planning_helper_agrees_with_strict_ceiling": budget.max_archive_bytes == strict_max_archive_bytes,
            "feasible_under_floors": strict_max_archive_bytes is not None,
            "status": ceiling_status,
            "conditional_only": True,
            "candidate_archive_bytes": None,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "p_g_a_t_ownership": {
            "P": {
                "role": "counted_predictor_program",
                "source_status": "REOPENED_N64_STRUCTURAL_SOURCE",
                "candidate_lineage_proven": False,
            },
            "G": {
                "role": "counted_compact_taskspace_correction_parameters",
                "source_module_present": g_module_present,
                "candidate_instance_receipt_present": False,
            },
            "A": {
                "role": "counted_compact_preimage_controls_for_distinct_y0_y1",
                "pair_population_scaffold_present": pair_envelope_present,
                "reverse_causal_G_to_exact_Y1_to_Y0_given_Y1_reference_grammar_present": (reverse_causal_a_present),
                "callable_modes": [
                    "FRAME1_ANCHORED_Y0_FIBRE",
                    "JOINT_SHARED_SKELETON_TWO_FIBRE",
                ],
                "pose6_consumed_by_reference_materializer": False,
                "lineage_classification": "PRODUCER_DECLARED_UNVERIFIED",
                "expected_y0_and_chronological_identities_bound": True,
                "decoder_source_binding_scope": "direct_module_file_bytes_only_nontransitive.v1",
                "production_frame1_and_frame0_grammars_present": False,
            },
            "T": {
                "role": "optional_counted_irreducible_terminal_quotient",
                "admission_rule": (
                    "count_only_after_measured_same_object_matched_byte_P_G_A_controls_fail_to_improve_total_score"
                ),
                "instantiated": False,
                "candidate_bytes": None,
                "may_duplicate_preceding_owners": False,
            },
            "E": {
                "role": "encoder_only_teacher_evaluator_truth",
                "includes": ["PBR2", "target_labels", "obligation_IR", "hard_oracle_evidence", "dense_teacher_Y"],
                "candidate_payload_allowed": False,
                "candidate_bytes": 0,
            },
            "exact_output_policy": "exact_decoder_output_does_not_by_itself_prove_or_disprove_payload_lineage",
            "payload_lineage_policy": "only_source_closed_P_G_A_and_optional_measured_T_may_ship_E_never_ships",
        },
        "borrowed_substrate_accounting": borrowed_substrate_accounting,
        "readiness": {
            "P": {
                "status": "STRUCTURAL_SOURCE_REOPENED_N64_ONLY",
                "n600_decision_ready": False,
                "blockers": ["n600_counted_predictor_foreign_key_to_complete_stack_absent"],
            },
            "G": {
                "status": "SOURCE_MODULE_PRESENT_NO_N600_INSTANCE_RECEIPT"
                if g_module_present
                else "SOURCE_MODULE_ABSENT",
                "n600_decision_ready": False,
                "blockers": ["n600_lineage_clean_counted_G_packet_and_compile_receipt_absent"],
            },
            "A": {
                "status": (
                    "REVERSE_CAUSAL_REFERENCE_GRAMMARS_PRESENT_PRODUCTION_ARCHIVE_BINDING_ABSENT"
                    if pair_envelope_present and reverse_causal_a_present
                    else "PAIR_ENVELOPE_OR_REVERSE_CAUSAL_REFERENCE_GRAMMARS_ABSENT"
                ),
                "n600_decision_ready": False,
                "blockers": [
                    "reverse_causal_A_packet_not_yet_adapted_into_PairPopulation_counted_sections",
                    "reverse_causal_A_canonical_instance_packet_and_compile_receipt_absent",
                    "reverse_causal_A_complete_typed_control_derivation_and_archive_lineage_absent",
                    "frame1_preimage_production_archive_parser_binding_absent",
                    "frame0_y0_fibre_production_archive_parser_binding_absent",
                ],
            },
            "T": {
                "status": "NOT_INSTANTIATED_NO_MEASURED_IRREDUCIBLE_QUOTIENT_ADMITTED",
                "n600_decision_ready": False,
                "blockers": [
                    "same_object_measured_unrepresented_obligation_absent",
                    "matched_byte_analytic_and_dictionary_controls_absent",
                ],
            },
            "E": {
                "status": "ENCODER_ONLY_N64_TEACHER_REOPENED",
                "candidate_payload_allowed": False,
                "candidate_bytes": 0,
            },
            "complete_stack": {
                "status": "BLOCKED_RESEARCH_ONLY",
                "n600_decision_ready": False,
                "candidate_archive_ready": False,
                "promotion_ready": False,
            },
        },
        "lineage": {
            "teacher_exact_output": True,
            "teacher_payload_lineage": "target_derived_encoder_only_candidate_forbidden",
            "candidate_exact_output_observed": False,
            "complete_archive_payload_lineage_closed": False,
            "historical_borrowed_substrate_accounting_present": prior_accounting_present,
            "historical_accounting_covers_current_p_g_a_stack": False,
            "current_stack_borrowed_substrate_accounting_present": True,
            "complete_candidate_archive_accounted": False,
            "originality_proven": False,
            "rule": "named_inputs_are_accounted_but_no_originality_proof_until_one_complete_candidate_archive_is_byte_accounted",
        },
        "canonical_apparatus": {
            "equations": {
                "required_ids": list(REQUIRED_EQUATION_IDS),
                "present_ids": [equation_id for equation_id in REQUIRED_EQUATION_IDS if equation_id in equation_ids],
                "missing_ids": list(missing_equations),
                "joined": not missing_equations,
            },
            "lane_dag": {
                "required_nodes": list(required_lane_nodes),
                "present_nodes": [node for node in required_lane_nodes if node in lane_dag_ids],
                "joined": True,
                "fixed_target_field_authoritative": False,
            },
            "canonical_dag": {
                "read_joined": True,
                "stack_schema_writeback_present": canonical_dag_writeback_present,
            },
            "research_index": {
                "read_joined": True,
                "stack_schema_writeback_present": research_index_writeback_present,
            },
            "cathedral_autopilot": {
                "source_reopened": True,
                "stack_schema_consumer_present": autopilot_consumer_present,
                "projection": {
                    "technique": "taskspace_inverse_codec_stack",
                    "research_only": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_blockers": sorted(set(exact_blockers)),
                },
            },
            "integration_blockers": sorted(set(apparatus_blockers)),
        },
        "exact_blockers": sorted(set(exact_blockers)),
        "next_executable_edge": NEXT_EXECUTABLE_EDGE,
        "verdict": "REOPENED_EVIDENCE_STACK_BLOCKED_BEFORE_N600_CANDIDATE_AUTHORITY",
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "body": body,
        "body_sha256": _sha256(canonical_json_bytes(body)),
    }
    validate_stack_receipt(receipt, reopen_sources=False)
    return receipt


def validate_stack_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    reopen_sources: bool = False,
) -> None:
    """Reject any authority relaxation, stale body hash, or source drift."""

    if set(receipt) != {"schema", "body", "body_sha256"}:
        raise TaskspaceInverseStackReceiptError("stack receipt keys are not exact")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise TaskspaceInverseStackReceiptError("stack receipt schema differs")
    body = _exact_mapping(receipt.get("body"), "stack body")
    if set(body) != _BODY_KEYS or body.get("schema") != SCHEMA:
        raise TaskspaceInverseStackReceiptError("stack body schema or fields differ")
    if receipt.get("body_sha256") != _sha256(canonical_json_bytes(body)):
        raise TaskspaceInverseStackReceiptError("stack body hash differs")
    authority = _exact_mapping(body.get("authority"), "stack authority")
    required_authority = {
        "research_only": True,
        "score_claim": False,
        "candidate_score": None,
        "candidate_archive_emitted": False,
        "candidate_payload_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "originality_claim": False,
        "pointer_moved": False,
        "pointer_delta": None,
        "n600_is_only_decision_surface": True,
        "n64_is_bounded_non_promotable_timing_and_acquisition_diagnostic_only": True,
    }
    if dict(authority) != required_authority:
        raise TaskspaceInverseStackReceiptError("stack authority firewall differs")
    frontier = _exact_mapping(body.get("frontier_join"), "frontier join")
    frontier_effective = _exact_mapping(frontier.get("effective_frontier"), "joined effective frontier")
    joined_target = _finite_nonnegative_number(frontier_effective.get("score"), "joined effective frontier score")
    if (
        frontier.get("role") != "live_competitive_target_only_no_local_archive_authority"
        or frontier.get("candidate_score") is not None
        or frontier.get("candidate_delta_to_frontier") is not None
        or frontier.get("roadmap_v4_fixed_target_ignored") is not True
        or frontier.get("ignored_field") != "body.mission.strict_authoritative_target"
    ):
        raise TaskspaceInverseStackReceiptError("frontier join can claim candidate authority")
    _finite_nonnegative_number(frontier.get("ignored_fixed_target"), "ignored fixed roadmap target")
    source_custody = _exact_mapping(body.get("source_custody"), "source custody")
    c1_custody = _exact_mapping(source_custody.get("c1_n600_anchor"), "C1 source custody")
    ceiling = _exact_mapping(body.get("conditional_c1_n600_byte_ceiling"), "C1 byte ceiling")
    ceiling_target = _finite_nonnegative_number(ceiling.get("target_score"), "C1 byte-ceiling target")
    ceiling_d_seg = _finite_nonnegative_number(ceiling.get("d_seg"), "C1 byte-ceiling d_seg")
    ceiling_d_pose = _finite_nonnegative_number(ceiling.get("d_pose"), "C1 byte-ceiling d_pose")
    expected_budget = target_byte_budget_for_score(
        target_score=ceiling_target,
        d_seg_floor=ceiling_d_seg,
        d_pose_floor=ceiling_d_pose,
    )
    expected_strict_max = _strict_max_archive_bytes(
        target_score=ceiling_target,
        d_seg=ceiling_d_seg,
        d_pose=ceiling_d_pose,
    )
    expected_status = (
        "PREDICTION_ONLY_CONDITIONAL_BYTE_CEILING"
        if expected_strict_max is not None
        else "NO_STRICT_NONNEGATIVE_ARCHIVE_BYTE_BUDGET"
    )
    expected_score_at_max = (
        contest_score(ceiling_d_seg, ceiling_d_pose, expected_strict_max) if expected_strict_max is not None else None
    )
    expected_score_at_next = (
        contest_score(ceiling_d_seg, ceiling_d_pose, expected_strict_max + 1)
        if expected_strict_max is not None
        else None
    )
    if (
        joined_target != ceiling_target
        or ceiling.get("source_pair_count") != 600
        or ceiling.get("source_axis") != C1_CANONICAL_AXIS
        or c1_custody.get("path") != DEFAULT_C1_ANCHOR.as_posix()
        or c1_custody.get("sha256") != C1_CANONICAL_FILE_SHA256
        or ceiling.get("distortion_floor_score") != expected_budget.distortion_floor_score
        or ceiling.get("rate_term_budget") != expected_budget.rate_term_budget
        or ceiling.get("strict_open_byte_boundary_formula")
        != "provisional_ceil(rate_term_budget*37545489/25)-1_then_ratchet_against_full_contest_score.v1"
        or ceiling.get("planning_helper_max_archive_bytes") != expected_budget.max_archive_bytes
        or ceiling.get("max_archive_bytes") != expected_strict_max
        or ceiling.get("score_at_max_archive_bytes") != expected_score_at_max
        or ceiling.get("score_at_next_archive_byte") != expected_score_at_next
        or ceiling.get("max_archive_bytes_is_strict")
        is not (expected_score_at_max is not None and expected_score_at_max < ceiling_target)
        or ceiling.get("next_archive_byte_is_not_strict")
        is not (expected_score_at_next is not None and expected_score_at_next >= ceiling_target)
        or ceiling.get("planning_helper_agrees_with_strict_ceiling")
        is not (expected_budget.max_archive_bytes == expected_strict_max)
        or ceiling.get("feasible_under_floors") is not (expected_strict_max is not None)
        or ceiling.get("status") != expected_status
        or ceiling.get("conditional_only") is not True
        or ceiling.get("candidate_archive_bytes") is not None
        or ceiling.get("score_claim") is not False
        or ceiling.get("promotion_eligible") is not False
        or ceiling.get("ready_for_exact_eval_dispatch") is not False
    ):
        raise TaskspaceInverseStackReceiptError("conditional C1 ceiling can claim candidate authority")
    ownership = _exact_mapping(body.get("p_g_a_t_ownership"), "P/G/A/T ownership")
    preimage = _exact_mapping(ownership.get("A"), "A ownership")
    terminal = _exact_mapping(ownership.get("T"), "T ownership")
    teacher = _exact_mapping(ownership.get("E"), "E teacher ownership")
    if (
        preimage.get("role") != "counted_compact_preimage_controls_for_distinct_y0_y1"
        or preimage.get("pair_population_scaffold_present") is not True
        or preimage.get("reverse_causal_G_to_exact_Y1_to_Y0_given_Y1_reference_grammar_present") is not True
        or preimage.get("callable_modes") != ["FRAME1_ANCHORED_Y0_FIBRE", "JOINT_SHARED_SKELETON_TWO_FIBRE"]
        or preimage.get("pose6_consumed_by_reference_materializer") is not False
        or preimage.get("lineage_classification") != "PRODUCER_DECLARED_UNVERIFIED"
        or preimage.get("expected_y0_and_chronological_identities_bound") is not True
        or preimage.get("decoder_source_binding_scope") != "direct_module_file_bytes_only_nontransitive.v1"
        or preimage.get("production_frame1_and_frame0_grammars_present") is not False
    ):
        raise TaskspaceInverseStackReceiptError("A ownership/readiness boundary differs")
    if (
        terminal.get("role") != "optional_counted_irreducible_terminal_quotient"
        or terminal.get("admission_rule")
        != "count_only_after_measured_same_object_matched_byte_P_G_A_controls_fail_to_improve_total_score"
        or terminal.get("instantiated") is not False
        or terminal.get("candidate_bytes") is not None
        or terminal.get("may_duplicate_preceding_owners") is not False
    ):
        raise TaskspaceInverseStackReceiptError("terminal quotient ownership differs")
    accounting = _exact_mapping(body.get("borrowed_substrate_accounting"), "borrowed substrate accounting")
    if dict(accounting) != _current_stack_borrowed_substrate_accounting():
        raise TaskspaceInverseStackReceiptError("borrowed substrate authority boundary differs")
    lineage = _exact_mapping(body.get("lineage"), "lineage")
    if (
        teacher.get("candidate_payload_allowed") is not False
        or teacher.get("candidate_bytes") != 0
        or lineage.get("current_stack_borrowed_substrate_accounting_present") is not True
        or lineage.get("complete_candidate_archive_accounted") is not False
        or lineage.get("originality_proven") is not False
    ):
        raise TaskspaceInverseStackReceiptError("payload/originality boundary differs")
    readiness = _exact_mapping(body.get("readiness"), "readiness")
    complete = _exact_mapping(readiness.get("complete_stack"), "complete-stack readiness")
    if any(
        complete.get(key) is not False for key in ("n600_decision_ready", "candidate_archive_ready", "promotion_ready")
    ):
        raise TaskspaceInverseStackReceiptError("complete stack cannot be ready in this receipt")
    blockers = body.get("exact_blockers")
    if (
        not isinstance(blockers, list)
        or not set(BASE_EXACT_BLOCKERS).issubset(blockers)
        or body.get("next_executable_edge") != NEXT_EXECUTABLE_EDGE
    ):
        raise TaskspaceInverseStackReceiptError("stack receipt must retain exact blockers")
    if reopen_sources:
        if repo_root is None:
            raise TaskspaceInverseStackReceiptError("repo_root is required to reopen stack sources")
        rebuilt = build_stack_receipt(repo_root=repo_root, strict_source_reopen=True)
        if canonical_json_bytes(rebuilt.get("body")) != canonical_json_bytes(body):
            raise TaskspaceInverseStackReceiptError("stack body differs from canonical source reconstruction")


def write_once_receipt(path: Path, receipt: Mapping[str, Any], *, repo_root: Path) -> None:
    """Crash-atomically publish canonical bytes without overwriting a peer."""

    validate_stack_receipt(receipt, reopen_sources=False)
    payload = canonical_json_bytes(receipt) + b"\n"
    target = _resolve(repo_root.resolve(), path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = _read_stable_bytes(target)
        if existing != payload:
            raise TaskspaceInverseStackReceiptError(f"refusing to overwrite a different stack receipt: {target}")
        reopened = json.loads(existing)
        validate_stack_receipt(reopened, repo_root=repo_root, reopen_sources=True)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _read_stable_bytes(target) != payload:
                raise TaskspaceInverseStackReceiptError(f"concurrent stack receipt differs: {target}") from None
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    reopened_payload = _read_stable_bytes(target)
    if reopened_payload != payload:
        raise TaskspaceInverseStackReceiptError("published stack receipt bytes differ")
    validate_stack_receipt(json.loads(reopened_payload), repo_root=repo_root, reopen_sources=True)


__all__ = [
    "DEFAULT_OUTPUT",
    "RECEIPT_SCHEMA",
    "SCHEMA",
    "TaskspaceInverseStackReceiptError",
    "build_stack_receipt",
    "canonical_json_bytes",
    "validate_stack_receipt",
    "write_once_receipt",
]
