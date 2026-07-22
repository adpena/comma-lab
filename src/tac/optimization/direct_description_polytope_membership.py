# SPDX-License-Identifier: MIT
"""Polytope-membership rung and n600 closure for Task #603 DDM.

The membership rung compares the frozen SegNet argmax cell of a described
frame-1 plane with the cell of the exact C1 solved member.  It is a local
CPU-Torch advisory measurement, not a contest score.  The n600 stage is a
scorer-free, same-artifact describe/decode custody proof built on the existing
DDM chart grammar.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.analysis.segnet_boundary_marginals import boundary_mask_from_labels
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.direct_description_measurement_ladder import (
    PAIR_HW,
    ChartReceiverResultV1,
    DirectDescriptionTargetPlaneReceiptV1,
    compile_chart_archive,
    fit_chart_description,
    iter_target_plane_chunks,
    load_pose_target_codes,
    load_target_receipt,
    measure_quantity_bridge,
    parse_chart_archive,
    prove_sampled_noop_honesty,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import (
    POINTER_SCORE_TEXT,
    SEED,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _require_sha256,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_real_target_rung0 import _committed_source_custody

RESULT_SCHEMA: Final = "direct_description_polytope_membership_n600.v1"
CONFIG_SCHEMA: Final = "DirectDescriptionPolytopeMembershipConfigV1"
CHECKPOINT_SCHEMA: Final = "DirectDescriptionPolytopeMembershipCheckpointV1"
MEMBERSHIP_SCHEMA: Final = "direct_description_argmax_cell_membership.v1"
CELL_FAMILY: Final = "frozen_segnet_native_f32_batch16_argmax_cell.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-SegNet advisory]"
SCORER_BATCH_SIZE: Final = 16
MEMBERSHIP_PAIR_COUNTS: Final = (64, 256)
N600_PAIRS: Final = 600
STAGE_NAMES: Final = (
    "membership_n64",
    "membership_n256",
    "n600_same_artifact_archive_closure",
)
MARGIN_BANDS: Final = (
    (0.0, 0.1, "[0,0.1)"),
    (0.1, 0.5, "[0.1,0.5)"),
    (0.5, 1.0, "[0.5,1)"),
    (1.0, float("inf"), "[1,inf)"),
)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

CellOracle = Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray | None]]


def _fraction_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.000000000000"
    return f"{numerator / denominator:.12f}"


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(value).tobytes(order="C"))


def _compact_quantity_bridge(bridge: Mapping[str, Any]) -> dict[str, Any]:
    """Retain exact aggregate custody while hashing the ordered per-pair rows."""

    rows = list(bridge["per_pair_exact_agreement"])
    compact = {key: value for key, value in bridge.items() if key != "per_pair_exact_agreement"}
    compact["per_pair_exact_agreement"] = {
        "pair_count": len(rows),
        "ordered_pair_ids": [row["pair_id"] for row in rows],
        "rows_sha256": _sha256(rfc8785_canonicalize(rows)),
        "all_pairs_covered_once": [row["pair_id"] for row in rows] == list(range(len(rows))),
    }
    return compact


def _empty_counts() -> dict[str, int]:
    return {
        "sites": 0,
        "rgb_pixels_exact": 0,
        "same_c1_argmax_cell": 0,
        "slack_rescued_inexact_sites": 0,
        "argmax_cell_escapes": 0,
    }


def _accumulate_counts(
    counts: dict[str, int],
    mask: np.ndarray,
    *,
    exact: np.ndarray,
    member: np.ndarray,
) -> None:
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != exact.shape or member.shape != exact.shape:
        raise DirectDescriptionError("membership stratum geometry mismatch")
    counts["sites"] += int(np.count_nonzero(selected))
    counts["rgb_pixels_exact"] += int(np.count_nonzero(selected & exact))
    counts["same_c1_argmax_cell"] += int(np.count_nonzero(selected & member))
    counts["slack_rescued_inexact_sites"] += int(np.count_nonzero(selected & member & ~exact))
    counts["argmax_cell_escapes"] += int(np.count_nonzero(selected & ~member))


def _finalize_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    sites = int(counts["sites"])
    exact = int(counts["rgb_pixels_exact"])
    members = int(counts["same_c1_argmax_cell"])
    inexact = sites - exact
    return {
        **{key: int(value) for key, value in counts.items()},
        "rgb_pixel_exact_fraction": _fraction_text(exact, sites),
        "same_c1_argmax_cell_fraction": _fraction_text(members, sites),
        "membership_minus_exactness_fraction": f"{(members - exact) / sites:.12f}" if sites else "0.000000000000",
        "slack_rescue_fraction_of_inexact_sites": _fraction_text(int(counts["slack_rescued_inexact_sites"]), inexact),
        "argmax_cell_escape_fraction": _fraction_text(int(counts["argmax_cell_escapes"]), sites),
    }


def membership_strata_counts(
    exact: np.ndarray,
    member: np.ndarray,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count exactness-to-membership deltas by class, margin, and boundary."""

    exact_mask = np.asarray(exact, dtype=np.bool_)
    member_mask = np.asarray(member, dtype=np.bool_)
    cells = np.asarray(target_cells)
    margins = np.asarray(target_margins, dtype=np.float32)
    if (
        exact_mask.ndim != 3
        or member_mask.shape != exact_mask.shape
        or cells.shape != exact_mask.shape
        or margins.shape != exact_mask.shape
        or not np.isfinite(margins).all()
        or np.any(margins < 0)
        or np.any((cells < 0) | (cells >= len(CLASS_NAMES)))
    ):
        raise DirectDescriptionError("invalid membership mask/cell/margin batch")
    boundary = boundary_mask_from_labels(cells, dilation=1)
    result: dict[str, dict[str, dict[str, int]]] = {
        "overall": {"all": _empty_counts()},
        "boundary_role": {"boundary_codim1": _empty_counts(), "cell_interior": _empty_counts()},
        "target_class": {name: _empty_counts() for name in CLASS_NAMES},
        "target_margin": {name: _empty_counts() for _lo, _hi, name in MARGIN_BANDS},
    }
    _accumulate_counts(
        result["overall"]["all"], np.ones(exact_mask.shape, dtype=np.bool_), exact=exact_mask, member=member_mask
    )
    _accumulate_counts(result["boundary_role"]["boundary_codim1"], boundary, exact=exact_mask, member=member_mask)
    _accumulate_counts(result["boundary_role"]["cell_interior"], ~boundary, exact=exact_mask, member=member_mask)
    for class_id, class_name in enumerate(CLASS_NAMES):
        _accumulate_counts(result["target_class"][class_name], cells == class_id, exact=exact_mask, member=member_mask)
    for low, high, name in MARGIN_BANDS:
        _accumulate_counts(
            result["target_margin"][name],
            (margins >= low) & (margins < high),
            exact=exact_mask,
            member=member_mask,
        )
    return result


def _merge_strata_counts(
    target: dict[str, dict[str, dict[str, int]]],
    update: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> None:
    if set(target) != set(update):
        raise DirectDescriptionError("membership stratum families changed during streaming")
    for family, rows in target.items():
        if set(rows) != set(update[family]):
            raise DirectDescriptionError("membership strata changed during streaming")
        for name, counts in rows.items():
            for key in counts:
                counts[key] += int(update[family][name][key])


def _finalize_strata(
    counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {family: {name: _finalize_counts(row) for name, row in rows.items()} for family, rows in counts.items()}


def iter_target_scorer_batches(
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    n_pairs: int,
    *,
    batch_size: int = SCORER_BATCH_SIZE,
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Rebatch one source chunk at a time into canonical scorer batches."""

    if batch_size != SCORER_BATCH_SIZE:
        raise DirectDescriptionError("membership scorer batch size must remain canonical 16")
    batch = np.empty((batch_size, 2, *PAIR_HW, 3), dtype=np.uint8)
    batch_ids: list[int] = []
    cursor = 0
    for pair_ids, planes in iter_target_plane_chunks(receipt, n_pairs):
        source_cursor = 0
        while source_cursor < len(pair_ids):
            take = min(batch_size - cursor, len(pair_ids) - source_cursor)
            batch[cursor : cursor + take] = planes[source_cursor : source_cursor + take]
            batch_ids.extend(pair_ids[source_cursor : source_cursor + take])
            cursor += take
            source_cursor += take
            if cursor == batch_size:
                yield tuple(batch_ids), batch
                cursor = 0
                batch_ids = []
    if cursor:
        yield tuple(batch_ids), batch[:cursor]


def _load_segnet_oracle(
    upstream_root: Path,
    *,
    threads: int,
) -> tuple[CellOracle, dict[str, Any]]:
    """Load the real frozen SegNet without importing or shipping scorer weights."""

    root = Path(upstream_root).resolve()
    modules_path = root / "modules.py"
    if threads < 1 or not modules_path.is_file():
        raise DirectDescriptionError("membership SegNet custody is unavailable")
    sys.path.insert(0, str(root))
    try:
        import modules as upstream_modules
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise DirectDescriptionError("membership SegNet runtime imports are unavailable") from exc
    if Path(upstream_modules.__file__).resolve() != modules_path:
        raise DirectDescriptionError("membership SegNet imported a non-custodied modules.py")
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    weights_path = Path(upstream_modules.segnet_sd_path).resolve()
    if not weights_path.is_file():
        raise DirectDescriptionError("membership SegNet weights are missing")
    segnet = upstream_modules.SegNet().eval().to("cpu")
    segnet.load_state_dict(load_file(str(weights_path), device="cpu"))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    def oracle(pairs: np.ndarray, need_margin: bool) -> tuple[np.ndarray, np.ndarray | None]:
        value = np.asarray(pairs)
        if value.dtype != np.uint8 or value.ndim != 5 or value.shape[1:] != (2, *PAIR_HW, 3):
            raise DirectDescriptionError("membership scorer requires uint8 [B,2,384,512,3]")
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            logits = segnet(segnet.preprocess_input(tensor))
            cells = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            margin: np.ndarray | None = None
            if need_margin:
                top2 = torch.topk(logits, k=2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(cells), None if margin is None else np.ascontiguousarray(margin)

    return oracle, {
        "cell_family": CELL_FAMILY,
        "implementation": "upstream.modules.SegNet.native_cpu_torch",
        "modules_path": str(modules_path),
        "modules_sha256": _sha256(_read_regular_file_once(modules_path)),
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": _sha256(_read_regular_file_once(weights_path)),
        "batch_size": SCORER_BATCH_SIZE,
        "threads": threads,
        "seed": SEED,
        "deterministic_algorithms": True,
        "device": "cpu",
        "weights_shipped_in_archive": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def measure_argmax_cell_membership(
    receiver: ChartReceiverResultV1,
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    *,
    oracle: CellOracle,
    cached_lstars: np.ndarray,
) -> dict[str, Any]:
    """Measure same-C1-member argmax-cell fraction in bounded batches."""

    n_pairs = receiver.z.n_pairs
    cache = np.asarray(cached_lstars)
    if cache.shape != (600, *PAIR_HW) or cache.dtype != np.int64:
        raise DirectDescriptionError("cached C1 target cells must be int64 [600,384,512]")
    strata: dict[str, dict[str, dict[str, int]]] | None = None
    per_pair: list[dict[str, Any]] = []
    target_cell_digest = hashlib.sha256()
    described_cell_digest = hashlib.sha256()
    cached_cell_digest = hashlib.sha256()
    target_cache_matches = 0
    described_cache_matches = 0
    rgb_input_cell_agreements = 0
    total_sites = n_pairs * PAIR_HW[0] * PAIR_HW[1]
    observed = 0
    replay_checked = False
    for pair_ids, target in iter_target_scorer_batches(receipt, n_pairs):
        described = receiver.render_pairs(pair_ids)
        target_cells, target_margins = oracle(target, True)
        described_cells, described_margins = oracle(described, False)
        if target_margins is None or described_margins is not None:
            raise DirectDescriptionError("membership oracle margin contract mismatch")
        if not replay_checked:
            replay_target, replay_margins = oracle(target, True)
            replay_described, _ = oracle(described, False)
            if (
                replay_margins is None
                or not np.array_equal(target_cells, replay_target)
                or not np.array_equal(target_margins, replay_margins)
                or not np.array_equal(described_cells, replay_described)
            ):
                raise DirectDescriptionError("membership scorer deterministic replay failed")
            replay_checked = True
        cached = np.ascontiguousarray(cache[np.asarray(pair_ids, dtype=np.int64)])
        exact = np.all(described[:, 1] == target[:, 1], axis=-1)
        member = described_cells == target_cells
        batch_strata = membership_strata_counts(exact, member, target_cells, target_margins)
        if strata is None:
            strata = batch_strata
        else:
            _merge_strata_counts(strata, batch_strata)
        target_cache = target_cells == cached
        described_cache = described_cells == cached
        target_rgb = np.argmax(target[:, 1], axis=-1)
        described_rgb = np.argmax(described[:, 1], axis=-1)
        for local_index, pair_id in enumerate(pair_ids):
            sites = PAIR_HW[0] * PAIR_HW[1]
            exact_count = int(np.count_nonzero(exact[local_index]))
            member_count = int(np.count_nonzero(member[local_index]))
            per_pair.append(
                {
                    "pair_id": pair_id,
                    "rgb_pixels_exact": exact_count,
                    "same_c1_argmax_cell_sites": member_count,
                    "sites": sites,
                    "rgb_pixel_exact_fraction": _fraction_text(exact_count, sites),
                    "same_c1_argmax_cell_fraction": _fraction_text(member_count, sites),
                    "slack_rescued_inexact_sites": int(np.count_nonzero(member[local_index] & ~exact[local_index])),
                    "argmax_cell_escapes": sites - member_count,
                    "c1_target_matches_cached_lstar": int(np.count_nonzero(target_cache[local_index])),
                    "described_matches_cached_lstar": int(np.count_nonzero(described_cache[local_index])),
                }
            )
        target_cell_digest.update(target_cells.tobytes(order="C"))
        described_cell_digest.update(described_cells.tobytes(order="C"))
        cached_cell_digest.update(cached.tobytes(order="C"))
        target_cache_matches += int(np.count_nonzero(target_cache))
        described_cache_matches += int(np.count_nonzero(described_cache))
        rgb_input_cell_agreements += int(np.count_nonzero(target_rgb == described_rgb))
        observed += len(pair_ids)
    if observed != n_pairs or not replay_checked or strata is None:
        raise DirectDescriptionError("membership pair coverage incomplete")
    if [row["pair_id"] for row in per_pair] != list(range(n_pairs)):
        raise DirectDescriptionError("membership pair order is noncanonical")
    finalized = _finalize_strata(strata)
    overall = finalized["overall"]["all"]
    rgb_disagreements = total_sites - rgb_input_cell_agreements
    cell_escapes = int(overall["argmax_cell_escapes"])
    return {
        "schema": MEMBERSHIP_SCHEMA,
        "n_pairs": n_pairs,
        "archive_bytes": len(receiver.archive),
        "archive_sha256": _sha256(receiver.archive),
        "cell_family": CELL_FAMILY,
        "definition": (
            "frame1 site remains in the exact native-f32 frozen-SegNet argmax cell of the C1 solved member "
            "under canonical batch16 CPU-Torch arithmetic"
        ),
        "same_c1_argmax_cell_fraction": overall["same_c1_argmax_cell_fraction"],
        "argmax_cell_escape_fraction": overall["argmax_cell_escape_fraction"],
        "rgb_pixel_exact_fraction_frame1": overall["rgb_pixel_exact_fraction"],
        "membership_minus_exactness_fraction": overall["membership_minus_exactness_fraction"],
        "slack_rescued_inexact_sites": overall["slack_rescued_inexact_sites"],
        "slack_rescue_fraction_of_inexact_sites": overall["slack_rescue_fraction_of_inexact_sites"],
        "strata": finalized,
        "same_site_rgb_channel_argmax_diagnostic": {
            "definition": "tie-first RGB input-channel argmax; not SegNet",
            "disagreements": rgb_disagreements,
            "sites": total_sites,
            "disagreement_fraction": _fraction_text(rgb_disagreements, total_sites),
            "segnet_cell_escape_fraction": _fraction_text(cell_escapes, total_sites),
            "rgb_diagnostic_minus_segnet_escape_fraction": f"{(rgb_disagreements - cell_escapes) / total_sites:.12f}",
        },
        "c1_target_cache_crosscheck": {
            "cached_family": "gt_n600.lstars exact frozen target raster",
            "target_member_matches": target_cache_matches,
            "described_matches": described_cache_matches,
            "sites": total_sites,
            "target_member_match_fraction": _fraction_text(target_cache_matches, total_sites),
            "described_match_fraction": _fraction_text(described_cache_matches, total_sites),
            "target_cells_sha256": target_cell_digest.hexdigest(),
            "described_cells_sha256": described_cell_digest.hexdigest(),
            "cached_cells_sha256": cached_cell_digest.hexdigest(),
        },
        "per_pair": per_pair,
        "per_pair_rows_sha256": _sha256(rfc8785_canonicalize(per_pair)),
        "deterministic_first_batch_replay": True,
        "scorer_batch_size": SCORER_BATCH_SIZE,
        "max_source_chunks_resident": 1,
        "max_scorer_batches_resident": 1,
        "d_seg_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def stream_decode_digest(
    receiver: ChartReceiverResultV1,
    *,
    n_pairs: int,
    chunk_pairs: int = 12,
) -> dict[str, Any]:
    """Decode every requested pair while retaining one described chunk."""

    if receiver.z.n_pairs != n_pairs or chunk_pairs != 12:
        raise DirectDescriptionError("stream decode closure requires exact archive pair count and 12-pair chunks")
    digest = hashlib.sha256()
    pair_digests: list[str] = []
    chunks = 0
    for start in range(0, n_pairs, chunk_pairs):
        pair_ids = tuple(range(start, min(start + chunk_pairs, n_pairs)))
        described = receiver.render_pairs(pair_ids)
        digest.update(described.tobytes(order="C"))
        pair_digests.extend(_array_sha256(described[index]) for index in range(len(pair_ids)))
        chunks += 1
    return {
        "pairs_decoded": n_pairs,
        "chunks_decoded": chunks,
        "chunk_pairs": chunk_pairs,
        "receiver_output_sha256": digest.hexdigest(),
        "per_pair_sha256_tree": _sha256(rfc8785_canonicalize(pair_digests)),
        "max_described_chunks_resident": 1,
    }


class DirectDescriptionPolytopeMembershipConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionPolytopeMembershipConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_polytope_membership_n600_seed1234"] = "ddm_polytope_membership_n600_seed1234"
    seed: Literal[1234] = SEED
    membership_pair_counts: tuple[Literal[64], Literal[256]] = MEMBERSHIP_PAIR_COUNTS
    closure_pairs: Literal[600] = N600_PAIRS
    scorer_batch_size: Literal[16] = SCORER_BATCH_SIZE
    scorer_threads: StrictInt = Field(ge=1, le=16)
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    checkpoint_policy: Literal["atomic_preserve_every_stage"] = "atomic_preserve_every_stage"
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionPolytopeMembershipConfigV1:
        _require_sha256(self.target_receipt_sha256, "target_receipt_sha256")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute custody")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": RESULT_SCHEMA,
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class DirectDescriptionPolytopeMembershipProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_polytope_membership.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


class DirectDescriptionPolytopeMembershipCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionPolytopeMembershipCheckpointV1"] = Field(
        default=CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    completed_stage_index: StrictInt = Field(ge=0, le=2)
    completed_stage_name: StrictStr
    next_stage_index: StrictInt = Field(ge=1, le=3)
    described_pairs: StrictInt = Field(ge=64, le=600)
    current_archive_b64: StrictStr
    current_archive_sha256: StrictStr
    current_archive_bytes: StrictInt = Field(ge=1)
    stage_history: tuple[dict[str, Any], ...]
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionPolytopeMembershipCheckpointV1:
        for field in ("config_sha256", "dsl_compile_hash", "semantic_argv_sha256", "current_archive_sha256"):
            _require_sha256(getattr(self, field), field)
        config = DirectDescriptionPolytopeMembershipConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("membership checkpoint config identity mismatch")
        if self.next_stage_index != self.completed_stage_index + 1:
            raise ValueError("membership checkpoint continuation cursor mismatch")
        if STAGE_NAMES[self.completed_stage_index] != self.completed_stage_name:
            raise ValueError("membership checkpoint stage name mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("membership checkpoint argv identity mismatch")
        try:
            archive = base64.b64decode(self.current_archive_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("membership checkpoint archive base64 malformed") from exc
        if (
            base64.b64encode(archive).decode() != self.current_archive_b64
            or len(archive) != self.current_archive_bytes
            or _sha256(archive) != self.current_archive_sha256
        ):
            raise ValueError("membership checkpoint archive custody mismatch")
        receiver = receive_chart_archive(archive)
        if receiver.z.n_pairs != self.described_pairs:
            raise ValueError("membership checkpoint described-pair count mismatch")
        expected_pairs = (64, 256, 600)[self.completed_stage_index]
        if self.described_pairs != expected_pairs or len(self.stage_history) != self.next_stage_index:
            raise ValueError("membership checkpoint stage coverage mismatch")
        if self.stage_history[-1].get("archive_sha256") != self.current_archive_sha256:
            raise ValueError("membership checkpoint history/archive mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(rfc8785_canonicalize(body))})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionPolytopeMembershipCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("membership checkpoint JSON is malformed") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "body_sha256"}
            or rfc8785_canonicalize(value) != payload
        ):
            raise DirectDescriptionError("membership checkpoint envelope is noncanonical")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("membership checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_polytope_membership__stage{self.completed_stage_index:03d}_{self.completed_stage_name}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


@dataclass(frozen=True, slots=True)
class PolytopeMembershipStageRunV1:
    final_receiver: ChartReceiverResultV1
    stage_history: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool


def load_polytope_membership_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionPolytopeMembershipConfigV1,
    semantic_argv: Sequence[str],
) -> DirectDescriptionPolytopeMembershipCheckpointV1:
    checkpoint = DirectDescriptionPolytopeMembershipCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
    ):
        raise DirectDescriptionError("membership resume identity differs from governed run")
    return checkpoint


def run_polytope_membership_stages(
    config: DirectDescriptionPolytopeMembershipConfigV1,
    *,
    checkpoint_directory: Path,
    semantic_argv: Sequence[str],
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    pose_codes: np.ndarray,
    oracle: CellOracle,
    cached_lstars: np.ndarray,
    resume_from: Path | None = None,
    stop_after_stage_index: int | None = None,
) -> PolytopeMembershipStageRunV1:
    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("polytope membership requires typed semantic argv")
    history: list[dict[str, Any]] = []
    checkpoints: list[Path] = []
    start_stage = 0
    receiver: ChartReceiverResultV1 | None = None
    if resume_from is not None:
        checkpoint = load_polytope_membership_checkpoint(resume_from, config=config, semantic_argv=argv)
        receiver = receive_chart_archive(base64.b64decode(checkpoint.current_archive_b64, validate=True))
        history = [dict(row) for row in checkpoint.stage_history]
        start_stage = checkpoint.next_stage_index
    for stage_index in range(start_stage, 3):
        previous = receiver
        n_pairs = (64, 256, 600)[stage_index]
        z = fit_chart_description(receipt, pose_codes, n_pairs)
        first = compile_chart_archive(z).archive
        second = compile_chart_archive(z).archive
        if first != second or parse_chart_archive(first).archive != first:
            raise DirectDescriptionError("polytope membership compiler determinism/parse identity failed")
        receiver = receive_chart_archive(first)
        prefix_same = True
        if previous is not None:
            prefix_same = stream_decode_digest(previous, n_pairs=previous.z.n_pairs)[
                "receiver_output_sha256"
            ] == _prefix_decode_sha256(receiver, previous.z.n_pairs)
            if not prefix_same:
                raise DirectDescriptionError("larger DDM artifact changed an already-described prefix")
        bridge = _compact_quantity_bridge(measure_quantity_bridge(receiver, receipt, pose_codes))
        row: dict[str, Any] = {
            "stage_index": stage_index,
            "stage_name": STAGE_NAMES[stage_index],
            "described_pairs": n_pairs,
            "archive_bytes": len(receiver.archive),
            "archive_sha256": _sha256(receiver.archive),
            "compiler_determinism_x2": True,
            "parse_reencode_identical": True,
            "prefix_receiver_output_identical": prefix_same,
            "quantity_bridge": bridge,
        }
        if stage_index < 2:
            membership = measure_argmax_cell_membership(
                receiver,
                receipt,
                oracle=oracle,
                cached_lstars=cached_lstars,
            )
            row["stage_role"] = "frozen_argmax_cell_membership_rung"
            row["membership"] = membership
            row["measured_curve_point"] = {
                "archive_bytes": len(receiver.archive),
                "membership_fraction": membership["same_c1_argmax_cell_fraction"],
                "pose_code_completeness": bridge["pose_debt"]["pose6_coordinate_exact_fraction"],
            }
        else:
            decode_first = stream_decode_digest(receiver, n_pairs=N600_PAIRS)
            decode_second = stream_decode_digest(receiver, n_pairs=N600_PAIRS)
            if decode_first != decode_second:
                raise DirectDescriptionError("n600 streaming decode determinism x2 failed")
            noop = prove_sampled_noop_honesty(receiver.z)
            custody = dict(receiver.custody)
            closure = {
                "schema": "direct_description_n600_same_artifact_closure.v1",
                "archive_bytes": len(receiver.archive),
                "archive_sha256": _sha256(receiver.archive),
                "same_artifact_describe_decode": bridge["archive_sha256"] == _sha256(receiver.archive),
                "pairs_described": N600_PAIRS,
                "pairs_decoded": decode_first["pairs_decoded"],
                "all_pairs_covered_once": bridge["per_pair_exact_agreement"]["all_pairs_covered_once"],
                "decode_determinism_x2": True,
                "decode": decode_first,
                "resume_identity_owed_to_outer_run": True,
                "unique_home_coverage_bytes": custody["unique_home_coverage_bytes"],
                "all_archive_bytes_have_one_home": custody["all_archive_bytes_have_one_home"],
                "member_count": custody["member_count"],
                "sampled_noop_honesty": noop,
                "source_chunks_resident_during_bridge": 1,
                "described_chunks_resident_during_bridge": 1,
                "maximum_combined_chunks_resident": 2,
                "candidate_archive": False,
                "score_claim": False,
            }
            if not (
                closure["same_artifact_describe_decode"]
                and closure["pairs_decoded"] == N600_PAIRS
                and closure["all_pairs_covered_once"]
                and closure["all_archive_bytes_have_one_home"]
                and closure["unique_home_coverage_bytes"] == closure["archive_bytes"]
            ):
                raise DirectDescriptionError("n600 same-artifact closure predicate failed")
            row["stage_role"] = "same_artifact_n600_scorer_free_custody_closure"
            row["n600_closure"] = closure
        history.append(row)
        checkpoint = DirectDescriptionPolytopeMembershipCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            completed_stage_index=stage_index,
            completed_stage_name=STAGE_NAMES[stage_index],
            next_stage_index=stage_index + 1,
            described_pairs=n_pairs,
            current_archive_b64=base64.b64encode(receiver.archive).decode(),
            current_archive_sha256=_sha256(receiver.archive),
            current_archive_bytes=len(receiver.archive),
            stage_history=tuple(history),
        )
        checkpoints.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_stage_index is not None and stage_index >= stop_after_stage_index:
            break
    if receiver is None:
        raise DirectDescriptionError("polytope membership executed no stage")
    return PolytopeMembershipStageRunV1(receiver, tuple(history), tuple(checkpoints), len(history) == 3)


def _prefix_decode_sha256(receiver: ChartReceiverResultV1, n_pairs: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, n_pairs, 12):
        pair_ids = tuple(range(start, min(start + 12, n_pairs)))
        digest.update(receiver.render_pairs(pair_ids).tobytes(order="C"))
    return digest.hexdigest()


def _storage_preflight(output_directory: Path) -> dict[str, Any]:
    probe = Path(output_directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    required = 96 * 1024 * 1024
    if shutil.disk_usage(probe).free < required:
        raise DirectDescriptionError("polytope membership refuses: insufficient receipt space")
    return {
        "output_tier": str(probe.resolve()),
        "required_free_bytes": required,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": "/Volumes/VertigoDataTier/pact",
        "bulk_target_read_only": True,
        "status": "PASS",
    }


def run_polytope_membership_n600(
    config: DirectDescriptionPolytopeMembershipConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    """Run n64/n256 membership plus genuine scorer-free n600 closure."""

    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    if Path(receipt.upstream_repo_root).resolve() / "upstream" != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("membership upstream root differs from exact target custody")
    pose_codes = load_pose_target_codes(receipt)
    cache_path = Path(receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("membership cached cell source size/path custody mismatch")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("membership cached C1 cells are unavailable") from exc
    oracle, scorer_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    primary = run_polytope_membership_stages(
        config,
        checkpoint_directory=root / "primary_stage_receipts",
        semantic_argv=semantic_argv,
        receipt=receipt,
        pose_codes=pose_codes,
        oracle=oracle,
        cached_lstars=cached_lstars,
    )
    if not primary.complete or len(primary.checkpoint_paths) != 3:
        raise DirectDescriptionError("polytope membership primary run did not preserve every stage")
    resumed = run_polytope_membership_stages(
        config,
        checkpoint_directory=root / "resume_stage_receipts",
        semantic_argv=semantic_argv,
        receipt=receipt,
        pose_codes=pose_codes,
        oracle=oracle,
        cached_lstars=cached_lstars,
        resume_from=primary.checkpoint_paths[1],
    )
    if (
        not resumed.complete
        or primary.final_receiver.archive != resumed.final_receiver.archive
        or primary.stage_history != resumed.stage_history
    ):
        raise DirectDescriptionError("polytope membership resume is not bit-identical")
    final_archive = _publish_new_bytes(
        root / "ddm_polytope_membership_n600_final.not_a_candidate.zip.receipt-bytes",
        primary.final_receiver.archive,
    )
    stages = [dict(row) for row in primary.stage_history]
    curve = [stages[0]["measured_curve_point"], stages[1]["measured_curve_point"]]
    n256_membership = stages[1]["membership"]
    n600_closure = dict(stages[2]["n600_closure"])
    n600_closure["resume_identical"] = True
    n600_closure["resume_identity_owed_to_outer_run"] = False
    result = {
        "schema": RESULT_SCHEMA,
        "task": 603,
        "master_task": 578,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": "MEMBERSHIP_MEASURED_AND_N600_SAME_ARTIFACT_CLOSURE_GREEN",
        "verdict_scope": (
            "local frozen-SegNet batch16 argmax-cell membership at n64/n256 plus scorer-free n600 "
            "same-artifact archive closure; no contest score, candidate, promotion, or launch authority"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": {
            "module": _committed_source_custody("src/tac/optimization/direct_description_polytope_membership.py"),
            "cli": _committed_source_custody("tools/run_direct_description_polytope_membership.py"),
        },
        "reuse": {
            "#547/#549": "exact C1 solved member, native cell target, and pose-tube lineage reused read-only",
            "#580": "factor-2/full-resize solved-member surface reused through predecessor target bytes",
            "#602": "mdl_polytope_member_solve retained as the bounded member-search predecessor",
            "rungs1_to_3": "existing DDM chart grammar, receiver, quantity bridge, and no-op custody reused directly",
            "new_surface": "membership objective and n600 same-artifact closure only",
        },
        "target_custody": {
            "receipt_path": config.target_receipt_path,
            "receipt_sha256": config.target_receipt_sha256,
            "source_cache_path": str(cache_path),
            "source_cache_bytes": receipt.source_cache.bytes,
            "source_cache_sha256": receipt.source_cache.sha256,
            "source_cache_hash_reused_from_exact_receipt": True,
            "source_cache_mutated": False,
        },
        "scorer_custody": scorer_custody,
        "curve": curve,
        "membership": {
            "n64": stages[0]["membership"],
            "n256": n256_membership,
        },
        "exactness_to_membership_delta": {
            "n256_rgb_pixel_exact_fraction_frame1": n256_membership["rgb_pixel_exact_fraction_frame1"],
            "n256_same_c1_argmax_cell_fraction": n256_membership["same_c1_argmax_cell_fraction"],
            "n256_membership_minus_exactness_fraction": n256_membership["membership_minus_exactness_fraction"],
            "n256_slack_rescue_fraction_of_inexact_sites": n256_membership["slack_rescue_fraction_of_inexact_sites"],
            "per_stratum": n256_membership["strata"],
        },
        "rgb_diagnostic_to_membership": n256_membership["same_site_rgb_channel_argmax_diagnostic"],
        "n600_closure": n600_closure,
        "archive": {
            "path": str(final_archive),
            "bytes": len(primary.final_receiver.archive),
            "sha256": _sha256(primary.final_receiver.archive),
            "candidate_role": "not_a_candidate",
            "parse_reencode_identical": True,
            "compiler_determinism_x2": True,
            "custody": dict(primary.final_receiver.custody),
        },
        "resume": {
            "resumed_from_stage": 1,
            "terminal_archive_bit_identical": True,
            "terminal_history_bit_identical": True,
            "primary_checkpoint_sha256": [_sha256(_read_regular_file_once(path)) for path in primary.checkpoint_paths],
            "resume_checkpoint_sha256": [_sha256(_read_regular_file_once(path)) for path in resumed.checkpoint_paths],
            "all_primary_stage_checkpoints_preserved": True,
        },
        "blocker_delta": {
            "N600_SAME_ARTIFACT_ARCHIVE_CLOSURE": "RED_TO_GREEN_MEASURED_APPARATUS_SCOPE",
            "POLYTOPE_MEMBERSHIP_RUNG": "NEW_GREEN_LOCAL_FROZEN_SCORER_SCOPE",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "target_bulk_remains_read_only_on_ssd": True,
            "scratch_policy": "one source chunk plus one scorer/receiver batch; immutable small checkpoints",
            "certify_or_block": "no deletion or movement performed",
        },
        "main_landing_review_required": True,
    }
    receipt_path = _publish_new_bytes(
        root / "ddm_polytope_membership_n600_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


__all__ = [
    "DirectDescriptionPolytopeMembershipCheckpointV1",
    "DirectDescriptionPolytopeMembershipConfigV1",
    "DirectDescriptionPolytopeMembershipProgramV1",
    "iter_target_scorer_batches",
    "load_polytope_membership_checkpoint",
    "measure_argmax_cell_membership",
    "membership_strata_counts",
    "run_polytope_membership_n600",
    "run_polytope_membership_stages",
    "stream_decode_digest",
]
