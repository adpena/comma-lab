#!/usr/bin/env python3
"""Prepare scorer-free SCMDL field proposals from retained realized cells.

This arm does not score, encode, refit, or launch anything.  It compiles fresh
dense-field edits over the byte-custodied DX2/AFR1 class field, persists every
materialized field, and hands the results to the gate-2 scorer owner.  Ground
truth never enters the proposal path.  The retained JF/FCD fields are folded,
not copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "ddm_sfp1_scmdl_field_proposal.v1"
HANDOFF_SCHEMA = "ddm_sfp1_scmdl_gate2_handoff.v1"
SHAPE = (600, 384, 512)
PIXELS = int(np.prod(SHAPE))
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection")


@dataclass(frozen=True)
class SourcePin:
    source_id: str
    path: str
    sha256: str
    bytes: int
    role: str
    proposal_eligible: bool
    exclusion: str | None = None


@dataclass(frozen=True)
class SelectorSpec:
    pair_rank_limit: int | None
    boundary_distance_max: int
    position_axis: str | None = None
    position_cells: tuple[int, ...] = ()
    disagreement_required: bool = True


@dataclass(frozen=True)
class GEditSpec:
    operation: str
    transition_order: tuple[str, ...]
    contexts: tuple[str, ...]
    refit_required: bool
    stored_side_stream: bool = False


@dataclass(frozen=True)
class ProposalSpec:
    proposal_id: str
    rank: int
    rank_status: str
    x_edit: str
    assignment_source: str
    selector: SelectorSpec
    g_edit: GEditSpec | None
    refit_required: bool
    source_laws: tuple[str, ...]
    prior_cell_relation: str


SOURCE_PINS: tuple[SourcePin, ...] = (
    SourcePin(
        "afr1_dense_x",
        "/Volumes/VertigoDataTier/pact/ddm_xs1_cross_section_conditioning/measurement_v1/retained/input/dx2_tokens_decoded.u8",
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
        117_964_800,
        "current AFR1/DX2 byte-identical five-class dense field X",
        True,
    ),
    SourcePin(
        "mst1_cuda_terminal_argmax",
        "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/retained/inputs/cuda_terminal_argmax_n600.npy",
        "e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34",
        117_964_928,
        "realized terminal scorer argmax A; copied into durable arm custody",
        True,
    ),
    SourcePin(
        "g3_atlas_rows",
        "/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_n600.jsonl",
        "faaff7299d86aa49c97e25e9cce2eeb0201f64e919f110015d31708788bcec09",
        9_121_001,
        "realized Seg/Pose pair-cell rankings; advisory source vehicle",
        True,
    ),
    SourcePin(
        "g3_summary",
        "/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/summary.json",
        "7e1f3918964f35729bf969b56eef6aebc949b22373f61e190d74d5c4825fa918",
        5_550,
        "atlas denominator and top-k measurement contract",
        True,
    ),
    SourcePin(
        "msr1_characterize",
        "/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1/MSR1_CHARACTERIZE.json",
        "cf227543124f231bf932f9828afd7c4f89469080f51306f945b397651b9d9e9d",
        18_116,
        "realized stage law and source-custody receipt",
        True,
    ),
    SourcePin(
        "msr1_boundary_distance",
        "/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1/token_boundary_distance.n600.npy",
        "df0c931653964354b9bf7a4d13edd302517950f1dde6a619c9d7ba4c392a78cf",
        117_964_928,
        "receiver-field boundary geometry; no GT values consumed",
        True,
    ),
    SourcePin(
        "msr1_native_margin",
        "/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1/native_top1_margin.float32.n600.npy",
        "e165bbc6ee0d9fa5363bfd490827825b32cd9ff27c2d26b503220252cc9a779d",
        471_859_328,
        "scorer top1-minus-top2 margin custody; diagnostic pin only",
        True,
    ),
    SourcePin(
        "mi1_cell_tables",
        "/Volumes/APDataStore/pact/ddm_mi1_indicator_model_axis/measurement_v1/RETAIN_cell_tables.json",
        "070fe1024c3e920d22e24bf7475ef46bd0699b01ca6221530c089a5bf70c89aa",
        95_366,
        "realized coder cell residuals used only for position-cell ranking",
        True,
    ),
    SourcePin(
        "mi1_verify",
        "/Volumes/APDataStore/pact/ddm_mi1_indicator_model_axis/measurement_v1/VERIFY.json",
        "5c7b65bc1d690ff241a7fab8e0f75e8c3899821cf50d194b70aa7f8a687b3045",
        2_914,
        "MI1 custody and control receipt",
        True,
    ),
    SourcePin(
        "rr9_verdict",
        "/Volumes/APDataStore/pact/ddm_rr9_reorder_refit/VERDICT.json",
        "243c083973b037e39ab75b090a667d2d28a05eb20a207d269d1f959c9dcf2c82",
        2_565,
        "within-group neutral result and unmeasured cross-group refit cell",
        True,
    ),
    SourcePin(
        "wj1_join_result",
        "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/JOIN_RESULT.json",
        "253511041b2b03209ee2dd138c4c1b753c4b95604784e730c01c11e321693ef0",
        118_718,
        "cost-times-error law only; GT-conditioned masks are folded",
        False,
        "position lists are GT-conditioned and cannot seed an SFP1 proposal",
    ),
    SourcePin(
        "bhw2_jf2_result",
        "/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/jf2/JF2_RESULT.json",
        "11b69b1bec08b7bd3042a3a9da12d1fac31aada098d692c9975cbfc77e8368b5",
        13_909,
        "positive instrument control and fold evidence only",
        False,
        "B/H/W selectors consume GT and the retained JF field is not a proposal",
    ),
    SourcePin(
        "bhw2_positive_field",
        "/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/jf2/screen/k060000/retained/benefit_field.u8.bin",
        "da09731f140a0ddbd79520004a41cc4a77eab5efd85f3b3d012bf1e48756553a",
        117_964_800,
        "control-only field with retained real RC64 outcome",
        False,
        "control-only; never ranked or handed off as an SCMDL candidate",
    ),
    SourcePin(
        "bhw2_positive_stream",
        "/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/jf2/benefit_reencode/retained/streams/jf2_k060000.rc64",
        "ab7e327b46ed60da37b2de9f812ac8e68a230aaf5131771121c54d38365736bb",
        108_108,
        "known physical RC64 result for the positive instrument control",
        False,
        "control-only byte receipt",
    ),
)


LAW_FOLD_TABLE: tuple[dict[str, Any], ...] = (
    {
        "law": "g3 pair score-mass atlas",
        "disposition": "GENERATE",
        "candidate_family": "atlas-ranked terminal-consensus X edits",
        "reason": "realized scorer cells give a legal pair ranking; ranks remain PROJECTION",
    },
    {
        "law": "MST1/MSR1 stage and boundary split",
        "disposition": "GENERATE",
        "candidate_family": "boundary-one terminal-consensus X edits",
        "reason": "boundary geometry and realized terminal argmax are legal; GT arrays are not read",
    },
    {
        "law": "MI1 position model residual",
        "disposition": "GENERATE",
        "candidate_family": "position-cell joined terminal-consensus X edits",
        "reason": "receiver-derived position cells join a realized scorer disagreement map",
    },
    {
        "law": "RR9 reorder plus refit",
        "disposition": "GENERATE_G_EDIT",
        "candidate_family": "cross-group causal schedule refit",
        "reason": "within-group reorder is folded as byte-neutral; cross-group refit is a distinct open cell",
    },
    {
        "law": "WJ1 cost-times-error positions",
        "disposition": "FOLD",
        "candidate_family": None,
        "reason": "its retained position masks are GT-conditioned; the concentration law is evidence only",
    },
    {
        "law": "BHW2/JF2 benefit field",
        "disposition": "CONTROL_ONLY",
        "candidate_family": None,
        "reason": "known real byte outcome validates the instrument, but GT-conditioned B and the JF field cannot seed proposals",
    },
    {
        "law": "FCD1 same-field diagonal",
        "disposition": "FOLD",
        "candidate_family": None,
        "reason": "copying its retained field would reproduce a closed prior candidate cell",
    },
    {
        "law": "WWC1 token-GT cone",
        "disposition": "FOLD",
        "candidate_family": None,
        "reason": "token-GT is forbidden as proposal material and the measured cone is closed",
    },
)


def _sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _pin_map() -> dict[str, SourcePin]:
    return {pin.source_id: pin for pin in SOURCE_PINS}


def verify_sources() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for pin in SOURCE_PINS:
        path = Path(pin.path)
        if not path.is_file():
            raise FileNotFoundError(f"missing source {pin.source_id}: {path}")
        size = path.stat().st_size
        if size != pin.bytes:
            raise ValueError(f"size mismatch for {pin.source_id}: {size} != {pin.bytes}")
        actual = _sha256(path)
        if actual != pin.sha256:
            raise ValueError(f"sha mismatch for {pin.source_id}: {actual} != {pin.sha256}")
        verified.append({**asdict(pin), "verified": True})
    return verified


def _atlas_ranked_pairs(path: Path) -> list[int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 600:
        raise ValueError(f"atlas denominator must be 600, got {len(rows)}")
    rows.sort(key=lambda row: (int(row["score_rank"]), int(row["pair_index"])))
    pair_ids = [int(row["pair_index"]) for row in rows]
    if sorted(pair_ids) != list(range(600)):
        raise ValueError("atlas pair ids are not exactly 0..599")
    return pair_ids


def _rank_position_cells(table_path: Path, axis: str, count: int) -> tuple[int, ...]:
    table = json.loads(table_path.read_text())["tables"][axis]
    gains = [
        (index, (float(base) - float(fitted)) / 8.0)
        for index, (base, fitted) in enumerate(zip(table["base_bits"], table["fitted_bits"]))
    ]
    return tuple(index for index, _ in sorted(gains, key=lambda item: (-item[1], item[0]))[:count])


def proposal_specs() -> tuple[ProposalSpec, ...]:
    g_edit = GEditSpec(
        operation="refit_cross_group_causal_schedule",
        transition_order=(),
        contexts=("source_class", "target_class", "boundary_distance", "position_cell"),
        refit_required=True,
    )
    return (
        ProposalSpec(
            "sfp1_p01_atlas24_boundary1",
            1,
            "PROJECTION",
            "assign realized terminal argmax where selector is true",
            "mst1_cuda_terminal_argmax",
            SelectorSpec(24, 1),
            g_edit,
            True,
            ("g3 pair score-mass atlas", "MST1/MSR1 stage and boundary split", "RR9 reorder plus refit"),
            "fresh X edit; no retained candidate field is read",
        ),
        ProposalSpec(
            "sfp1_p02_atlas64_boundary1",
            2,
            "PROJECTION",
            "assign realized terminal argmax where selector is true",
            "mst1_cuda_terminal_argmax",
            SelectorSpec(64, 1),
            g_edit,
            True,
            ("g3 pair score-mass atlas", "MST1/MSR1 stage and boundary split", "RR9 reorder plus refit"),
            "fresh superset cell; top24 result is an explicit prefix",
        ),
        ProposalSpec(
            "sfp1_p03_mi1_patch12_boundary1",
            3,
            "PROJECTION",
            "assign realized terminal argmax where selector is true",
            "mst1_cuda_terminal_argmax",
            SelectorSpec(None, 1, "patch192", ()),
            g_edit,
            True,
            ("MI1 position model residual", "MST1/MSR1 stage and boundary split", "RR9 reorder plus refit"),
            "fresh X edit; position law is joined to current realized scorer disagreements",
        ),
    )


def validate_proposal(spec: ProposalSpec) -> None:
    encoded = json.dumps(asdict(spec), sort_keys=True).lower()
    for forbidden in ("token_gt", "gt_argmax", "ground_truth", "exception_stream", "address_stream"):
        if forbidden in encoded:
            raise ValueError(f"forbidden proposal material {forbidden!r} in {spec.proposal_id}")
    if spec.rank_status != "PROJECTION":
        raise ValueError(f"rank must be PROJECTION for {spec.proposal_id}")
    if spec.assignment_source != "mst1_cuda_terminal_argmax":
        raise ValueError(f"assignment must come from realized terminal argmax for {spec.proposal_id}")
    if spec.g_edit is not None and spec.g_edit.stored_side_stream:
        raise ValueError(f"G edit cannot create a stored side stream for {spec.proposal_id}")


def _position_map(axis: str) -> np.ndarray:
    rows = np.arange(SHAPE[1], dtype=np.int16)[:, None]
    cols = np.arange(SHAPE[2], dtype=np.int16)[None, :]
    if axis == "patch192":
        return (rows // 32) * 16 + cols // 32
    if axis == "tile48":
        return (rows // 64) * 8 + cols // 64
    if axis == "subtile4":
        return ((rows % 64) // 32) * 2 + (cols % 64) // 32
    raise ValueError(f"unsupported position axis {axis!r}")


def selector_mask(
    spec: SelectorSpec,
    frame_index: int,
    x_frame: np.ndarray,
    target_frame: np.ndarray,
    boundary_frame: np.ndarray,
    ranked_pairs: Sequence[int],
) -> np.ndarray:
    if spec.pair_rank_limit is not None and frame_index not in set(ranked_pairs[: spec.pair_rank_limit]):
        return np.zeros_like(x_frame, dtype=bool)
    mask = boundary_frame <= spec.boundary_distance_max
    if spec.disagreement_required:
        mask = mask & (x_frame != target_frame)
    if spec.position_axis:
        cells = np.isin(_position_map(spec.position_axis), np.asarray(spec.position_cells))
        mask = mask & cells
    return mask


def _transition_counts(x: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    old = x[mask].astype(np.int16)
    new = target[mask].astype(np.int16)
    packed = old * 5 + new
    counts = np.bincount(packed, minlength=25)
    return {
        f"{source}->{destination}": int(counts[source * 5 + destination])
        for source in range(5)
        for destination in range(5)
        if source != destination and counts[source * 5 + destination]
    }


def _merge_counts(total: dict[str, int], part: Mapping[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + int(value)


def delta_is_subset(
    base: np.ndarray, smaller: np.ndarray, larger: np.ndarray
) -> tuple[bool, bool, int, int]:
    """Return subset and assignment agreement for two candidate deltas."""
    smaller_count = 0
    larger_count = 0
    subset = True
    assignments_agree = True
    for frame in range(base.shape[0]):
        base_frame = np.asarray(base[frame])
        small_frame = np.asarray(smaller[frame])
        large_frame = np.asarray(larger[frame])
        small_delta = small_frame != base_frame
        large_delta = large_frame != base_frame
        smaller_count += int(small_delta.sum())
        larger_count += int(large_delta.sum())
        subset = subset and not bool(np.any(small_delta & ~large_delta))
        assignments_agree = assignments_agree and not bool(
            np.any(small_delta & (small_frame != large_frame))
        )
    return subset, assignments_agree, smaller_count, larger_count


def _write_field(
    output: Path,
    x: np.ndarray,
    target: np.ndarray,
    boundary: np.ndarray,
    spec: ProposalSpec | None,
    ranked_pairs: Sequence[int],
) -> dict[str, Any]:
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    changed = 0
    transitions: dict[str, int] = {}
    with temp.open("wb") as handle:
        for frame in range(SHAPE[0]):
            x_frame = np.asarray(x[frame])
            if spec is None:
                out = x_frame
                mask = np.zeros_like(x_frame, dtype=bool)
            else:
                target_frame = np.asarray(target[frame])
                mask = selector_mask(
                    spec.selector,
                    frame,
                    x_frame,
                    target_frame,
                    np.asarray(boundary[frame]),
                    ranked_pairs,
                )
                out = x_frame.copy()
                out[mask] = target_frame[mask]
                changed += int(mask.sum())
                _merge_counts(transitions, _transition_counts(x_frame, target_frame, mask))
            payload = np.ascontiguousarray(out, dtype=np.uint8).tobytes()
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if temp.stat().st_size != PIXELS:
        raise ValueError(f"materialized field has {temp.stat().st_size} bytes, expected {PIXELS}")
    temp.replace(output)
    return {
        "path": str(output),
        "bytes": PIXELS,
        "sha256": digest.hexdigest(),
        "changed_sites": changed,
        "transition_counts": dict(sorted(transitions.items())),
    }


def _resume_or_write(
    output: Path,
    expected: Mapping[str, Any] | None,
    writer: Any,
) -> dict[str, Any]:
    if output.exists():
        if not output.is_file():
            raise ValueError(f"output path is not a regular file: {output}")
        if expected and output.stat().st_size == expected.get("bytes") and _sha256(output) == expected.get("sha256"):
            return dict(expected)
        raise ValueError(f"existing output is not checkpoint-identical; preserve and adjudicate: {output}")
    return writer()


def _schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA,
        "title": "SCMDL scorer-free dense-field proposal",
        "type": "object",
        "required": [
            "proposal_id",
            "rank",
            "rank_status",
            "x_edit",
            "assignment_source",
            "selector",
            "refit_required",
        ],
        "properties": {
            "proposal_id": {"type": "string"},
            "rank": {"type": "integer", "minimum": 1},
            "rank_status": {"const": "PROJECTION"},
            "x_edit": {"type": "string"},
            "assignment_source": {"const": "mst1_cuda_terminal_argmax"},
            "selector": {"type": "object"},
            "g_edit": {"type": ["object", "null"]},
            "refit_required": {"type": "boolean"},
        },
        "forbidden_material": [
            "token-GT values or masks as selectors/assignments",
            "retained JF/FCD candidate fields",
            "stored address or exception side streams",
        ],
        "coded_object": "the complete changed 600x384x512 uint8 X field",
    }


def _storage_preflight(output_root: Path, candidate_count: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    required = PIXELS * (candidate_count + 2) + (64 << 20)
    receipt = {
        "schema": "ddm_sfp1_storage_preflight.v1",
        "tier": str(output_root),
        "free_bytes": usage.free,
        "required_bytes": required,
        "passed": usage.free >= required,
        "policy": "retain all complete fields; atomic partials are deleted only after a verified final exists",
    }
    _atomic_json(output_root / "STORAGE_PREFLIGHT.json", receipt)
    if not receipt["passed"]:
        raise OSError(f"insufficient storage: {usage.free} < {required}")
    return receipt


def _copy_argmax_to_custody(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha:
            raise ValueError(f"existing custody payload is not source-identical; preserve and adjudicate: {destination}")
    else:
        temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with source.open("rb") as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(temp) != expected_sha:
            raise ValueError("durable argmax copy failed byte-identity")
        temp.replace(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": _sha256(destination)}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def build(output_root: Path = OUTPUT_ROOT, resume_from: Path | None = None) -> dict[str, Any]:
    specs = proposal_specs()
    for spec in specs:
        validate_proposal(spec)
    _storage_preflight(output_root, len(specs))
    verified_sources = verify_sources()
    _atomic_json(output_root / "SOURCE_CUSTODY.json", {"sources": verified_sources})
    _atomic_json(output_root / "LAW_FOLD_TABLE.json", list(LAW_FOLD_TABLE))
    _atomic_json(output_root / "SCHEMA.json", _schema_document())

    pins = _pin_map()
    durable_argmax = _copy_argmax_to_custody(
        Path(pins["mst1_cuda_terminal_argmax"].path),
        output_root / "custody" / "cuda_terminal_argmax_n600.npy",
        pins["mst1_cuda_terminal_argmax"].sha256,
    )
    checkpoint_path = resume_from or output_root / "RUN_CHECKPOINT.json"
    old_checkpoint: dict[str, Any] = {}
    if checkpoint_path.is_file():
        old_checkpoint = json.loads(checkpoint_path.read_text())
    completed: dict[str, Any] = dict(old_checkpoint.get("completed", {}))

    x = np.memmap(pins["afr1_dense_x"].path, dtype=np.uint8, mode="r", shape=SHAPE)
    target = np.load(durable_argmax["path"], mmap_mode="r").reshape(SHAPE)
    boundary = np.load(pins["msr1_boundary_distance"].path, mmap_mode="r").reshape(SHAPE)
    ranked_pairs = _atlas_ranked_pairs(Path(pins["g3_atlas_rows"].path))
    mi1_path = Path(pins["mi1_cell_tables"].path)
    hydrated_specs: list[ProposalSpec] = []
    for spec in specs:
        if spec.selector.position_axis == "patch192":
            selector = SelectorSpec(
                spec.selector.pair_rank_limit,
                spec.selector.boundary_distance_max,
                "patch192",
                _rank_position_cells(mi1_path, "patch192", 12),
                spec.selector.disagreement_required,
            )
            spec = ProposalSpec(**{**asdict(spec), "selector": selector, "g_edit": spec.g_edit})
        hydrated_specs.append(spec)

    null_path = output_root / "controls" / "null_empty_proposal.u8"
    null_path.parent.mkdir(parents=True, exist_ok=True)
    null_result = _resume_or_write(
        null_path,
        completed.get("null_empty_proposal"),
        lambda: _write_field(null_path, x, target, boundary, None, ranked_pairs),
    )
    if null_result["sha256"] != pins["afr1_dense_x"].sha256 or null_result["changed_sites"] != 0:
        raise ValueError("null proposal is not byte-identical to X")
    completed["null_empty_proposal"] = null_result
    _atomic_json(
        checkpoint_path,
        {"schema": "ddm_sfp1_resume.v1", "stage": "null_control_complete", "completed": completed},
    )

    candidates: list[dict[str, Any]] = []
    for spec in hydrated_specs:
        output = output_root / "candidates" / f"{spec.proposal_id}.u8"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = _resume_or_write(
            output,
            completed.get(spec.proposal_id),
            lambda spec=spec, output=output: _write_field(output, x, target, boundary, spec, ranked_pairs),
        )
        if result["changed_sites"] <= 0 or result["sha256"] == pins["afr1_dense_x"].sha256:
            raise ValueError(f"candidate {spec.proposal_id} did not change the real field")
        transition_order = tuple(
            key for key, _ in sorted(result["transition_counts"].items(), key=lambda item: (-item[1], item[0]))
        )
        g_edit = asdict(spec.g_edit) if spec.g_edit else None
        if g_edit is not None:
            g_edit["transition_order"] = list(transition_order)
        candidates.append({**asdict(spec), "g_edit": g_edit, "materialized_field": result})
        completed[spec.proposal_id] = result
        _atomic_json(
            checkpoint_path,
            {"schema": "ddm_sfp1_resume.v1", "stage": f"{spec.proposal_id}_complete", "completed": completed},
        )

    positive = {
        "control_id": "bhw2_jf2_k060000_known_rc64",
        "control_only": True,
        "eligible_for_ranking": False,
        "purpose": "prove gate-2 is consuming the intended field and physical RC64 instrument",
        "field": asdict(pins["bhw2_positive_field"]),
        "known_realized_outcome": {
            "stream": asdict(pins["bhw2_positive_stream"]),
            "physical_stream_bytes": 108_108,
            "field_edits": 8_301,
            "d_seg": "UNMEASURED",
            "d_pose": "UNMEASURED",
            "score_claim": False,
        },
        "prohibition": "do not rank, refit, or submit this prior JF field as an SFP1 proposal",
    }
    controls = {"null": null_result, "positive": positive}
    _atomic_json(output_root / "CONTROLS.json", controls)
    _atomic_json(output_root / "CANDIDATE_SET.json", {"schema": SCHEMA, "candidates": candidates})

    p01 = np.memmap(candidates[0]["materialized_field"]["path"], dtype=np.uint8, mode="r", shape=SHAPE)
    p02 = np.memmap(candidates[1]["materialized_field"]["path"], dtype=np.uint8, mode="r", shape=SHAPE)
    subset, assignments_agree, p01_count, p02_count = delta_is_subset(x, p01, p02)
    verification = {
        "schema": "ddm_sfp1_materialized_verify.v1",
        "passed": bool(subset and assignments_agree),
        "null_byte_identical": null_result["sha256"] == pins["afr1_dense_x"].sha256,
        "candidate_hashes_unique": len({row["materialized_field"]["sha256"] for row in candidates})
        == len(candidates),
        "p01_delta_is_prefix_of_p02": subset,
        "p01_p02_assignments_agree_on_prefix": assignments_agree,
        "p01_recounted_changed_sites": p01_count,
        "p02_recounted_changed_sites": p02_count,
        "proposal_documents_contain_forbidden_material": any(
            token in json.dumps(candidates, sort_keys=True).lower()
            for token in ("token_gt", "gt_argmax", "ground_truth", "exception_stream", "address_stream")
        ),
    }
    if not verification["passed"] or verification["proposal_documents_contain_forbidden_material"]:
        raise ValueError(f"materialized verification failed: {verification}")
    _atomic_json(output_root / "VERIFY.json", verification)

    handoff = {
        "schema": HANDOFF_SCHEMA,
        "verdict": "GENERATOR-READY",
        "verdict_scope": "FORMULATION",
        "score_claim": False,
        "axis": "[scorer-free projection prep]",
        "provenance": {
            "git_head_at_materialization": _git_head(),
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
                "sha256": _sha256(Path(__file__)),
            },
            "determinism": "no RNG; source-pin verification plus deterministic pair/cell ordering",
            "resume_checkpoint": str(checkpoint_path),
        },
        "denominator": {
            "law_rows_total": len(LAW_FOLD_TABLE),
            "law_rows_generated": 4,
            "law_rows_folded_or_control_only": 4,
            "ranked_proposals": len(candidates),
            "controls": 2,
            "dense_field_sites": PIXELS,
            "atlas_pairs": 600,
        },
        "boundaries": {
            "not_measured": ["RC64 bytes of any SFP1 field", "d_seg", "d_pose", "archive bytes", "exact score"],
            "no_launches": True,
            "no_scorer_runs": True,
            "no_modal": True,
            "rank_labels": "PROJECTION only",
            "coded_object": "changed dense X field; no address or exception stream",
        },
        "source_custody": str(output_root / "SOURCE_CUSTODY.json"),
        "law_fold_table": str(output_root / "LAW_FOLD_TABLE.json"),
        "proposal_schema": str(output_root / "SCHEMA.json"),
        "candidate_set": str(output_root / "CANDIDATE_SET.json"),
        "controls": str(output_root / "CONTROLS.json"),
        "verification": str(output_root / "VERIFY.json"),
        "consumption_order": [
            "verify every SOURCE_CUSTODY pin",
            "decode/compare the null control and require byte identity to afr1_dense_x",
            "run the positive control and require the retained 108108-byte RC64 outcome before trusting the instrument",
            "when RXC1 is GATE-1-PASSED, refit and byte-close p01, then p02, then p03",
            "only after real RC64 admission, send the exact changed field through the frozen Seg/Pose gate",
            "MAIN alone may compose an admitted row and fire exact contest evaluation",
        ],
        "gate_dependency": {
            "requires": "ddm_rxc1_scmdl_restartable_coder_prep verdict GATE-1-PASSED",
            "this_arm": "gate-2 scorer-free half complete",
            "fire_condition": "RXC1 GATE-1-PASSED and this handoff remains GENERATOR-READY",
        },
        "next_action": {
            "count": 1,
            "disposition": "QUEUE",
            "owner": "MAIN-selected gate-2 scorer arm",
            "consumer_store": str(output_root / "HANDOFF.json"),
            "fire_trigger": "RXC1 publishes GATE-1-PASSED against the pinned SCMDL schema",
            "action": "consume controls in order, then refit and byte-close p01 before any scorer measurement",
        },
        "own_vehicle_frontier": {
            "vehicle": "AFR1",
            "S": 0.14797617125559104,
            "archive_bytes": 180_002,
            "archive_sha256": "cbb8d9283f435204800e31250bad7880490658012e2b6d8aa196ac4666bc84f5",
            "axis": "[contest-CUDA T4 n600]",
            "pointer_moved": False,
        },
        "payloads": {
            "durable_argmax": durable_argmax,
            "null": null_result,
            "candidate_fields": [row["materialized_field"] for row in candidates],
        },
    }
    _atomic_json(output_root / "HANDOFF.json", handoff)
    completed["handoff"] = {
        "path": str(output_root / "HANDOFF.json"),
        "bytes": (output_root / "HANDOFF.json").stat().st_size,
        "sha256": _sha256(output_root / "HANDOFF.json"),
    }
    _atomic_json(
        checkpoint_path,
        {"schema": "ddm_sfp1_resume.v1", "stage": "complete", "completed": completed},
    )
    _atomic_json(
        output_root / "CLEANUP_MANIFEST.json",
        {
            "schema": "ddm_sfp1_cleanup_manifest.v1",
            "policy": "KEEP",
            "reason": "all complete fields are gate-2 inputs; no complete payload is auto-deleted",
            "rebuild_command": ".venv/bin/python experiments/ddm_sfp1_scmdl_field_proposal_prep.py --resume-from /Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/RUN_CHECKPOINT.json",
            "payloads": handoff["payloads"],
        },
    )
    return handoff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        print(json.dumps({"verified_sources": verify_sources()}, indent=2, sort_keys=True))
        return 0
    result = build(args.output_root, args.resume_from)
    print(json.dumps({"verdict": result["verdict"], "handoff": str(args.output_root / "HANDOFF.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
