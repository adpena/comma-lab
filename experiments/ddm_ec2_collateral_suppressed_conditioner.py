#!/usr/bin/env python3
"""Retained-field EC2/QS3 collateral diagnosis and sealed no-fire order.

This program never imports or executes a scorer.  It consumes hash-pinned
retained argmax/token fields, retains every derived per-cell payload, fits a
decoded-context suppression gate, and emits a MAIN-owned fire order that is
blocked unless the charter's B-preservation prediction is met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.ndimage import distance_transform_edt

from tac.optimization.ec2_collateral_suppressed_proposer import (
    NET_FLIPS_PER_BYTE,
    ORIENTED_CONTEXTS,
    CollateralSuppressedProposer,
    collateral_priced_delta,
    fit_context_counts,
    gate_from_context_counts,
    oriented_context_codes_at,
)

ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_ec2_collateral_suppressed_conditioner/seal_r1"
)
AXIS: Final = "[contest-CUDA T4 retained argmax fields, n600; macOS-CPU scorer-free analysis] COMPONENT-ONLY"
TL1_MEMO: Final = ROOT / ".omx/research/ddm_tl1_teacher_ledger_20260822.md"
TL1_SHA256: Final = "d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15"
IG1_MEMO: Final = ROOT / ".omx/research/ddm_ig1_implicit_carriage_gestalt_20260821.md"
IG1_SHA256: Final = "8ec60069b33f2d19d9a39ea30c94acee66ac299d800b5e739f411a48aa42ce8b"

BASE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1c_20260814/retained/fields/cp135_base_argmax_n600.npy"
)
BASE_SHA256: Final = "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727"
GT = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1c_20260814/retained/fields/gt_argmax_n600.npy"
)
GT_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/"
    "ddm_ec1_20260814/main_cuda/fire_inputs/decoded_tokens_n600.npy"
)
TOKENS_SHA256: Final = "03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4"
EC1 = Path("/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/ec2_oriented/argmax_n600.npy")
EC1_SHA256: Final = "803a1d8755cafcf31b03d8ad1494d49f89f6e4fb2115341423308e0db20b3a1a"
EC1_FIXED = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/bg2_postmortem_r3/"
    "decomposition/fixed_n600.bool.npy"
)
EC1_FIXED_SHA256: Final = "f1160513e85f3137b4ac4c9ddf0f31d3ae37345bcb834e85bc9cc195f211694f"
EC1_HARM = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/bg2_postmortem_r3/"
    "decomposition/introduced_n600.bool.npy"
)
EC1_HARM_SHA256: Final = "57c6ef5ff19e592388173fe980bca16ff9757ca60ca175ea87c837ff67aab150"

QS3_CANDIDATE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/retained_fields/"
    "ddm_qs1_dual_axis_20260813_r2/retained/fields/candidate_argmax_n600.npy"
)
QS3_CANDIDATE_SHA256: Final = "ad1e3dcc0a57c53f0757773a018335924afc26992f398c23ec084eecace7ed20"
QS3_DECOMPOSITION = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/GT_ATTRIBUTED_DECOMPOSITION.json")
QS3_DECOMPOSITION_SHA256: Final = "e7b9e92f7d10d3468d7f13a8a338ee396bea9827e0a256ffeb832391f52478ee"
QS3_BANK = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/full_bank_screen.jsonl")
QS3_BANK_SHA256: Final = "39dc688d75de33b628a051fca6a1b0c9c29679d2b4df7463f8b41f1af2b678c3"
QS4_MAP = Path("/Volumes/VertigoDataTier/pact/ddm_qs4_20260813/COLLATERAL_MAP.json")

CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_TO_ID: Final = {name: class_id for class_id, name in enumerate(CLASS_NAMES)}
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512


class EC2AnalysisError(RuntimeError):
    """Raised when retained-field authority or seal integrity differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise EC2AnalysisError(f"required retained file is absent: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise EC2AnalysisError(f"retained SHA differs for {path}: {digest}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if path.exists():
        if path.read_bytes() != payload:
            raise EC2AnalysisError(f"retained payload differs on resume: {path}")
        return
    if partial.exists():
        raise EC2AnalysisError(f"uncertified partial payload requires review: {partial}")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if path.exists():
        retained = np.load(path, allow_pickle=False)
        if set(retained.files) != set(arrays) or any(
            not np.array_equal(retained[key], value, equal_nan=True)
            for key, value in arrays.items()
        ):
            raise EC2AnalysisError(f"retained NPZ differs on resume: {path}")
        return
    if partial.exists():
        raise EC2AnalysisError(f"uncertified partial payload requires review: {partial}")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def load_field(path: Path, dtype: np.dtype[Any]) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (N_PAIRS, HEIGHT, WIDTH) or value.dtype != dtype:
        raise EC2AnalysisError(f"retained field geometry/dtype differs: {path}")
    return value


def boundary_mask(tokens: np.ndarray) -> np.ndarray:
    value = np.zeros(tokens.shape, dtype=np.bool_)
    value[:, 1:] |= tokens[:, 1:] != tokens[:, :-1]
    value[:, :-1] |= tokens[:, :-1] != tokens[:, 1:]
    value[1:, :] |= tokens[1:, :] != tokens[:-1, :]
    value[:-1, :] |= tokens[:-1, :] != tokens[1:, :]
    return value


def event_arrays(
    *,
    frame: int,
    mask: np.ndarray,
    outcome: int,
    tokens: np.ndarray,
    source_class: np.ndarray,
    target_class: np.ndarray,
    distance_to_b: np.ndarray,
    distance_to_base_error: np.ndarray,
    distance_to_boundary: np.ndarray,
) -> dict[str, np.ndarray]:
    flat = np.flatnonzero(mask)
    y, x = np.unravel_index(flat, (HEIGHT, WIDTH))
    frames = np.full(flat.shape, frame, dtype=np.uint16)
    codes = oriented_context_codes_at(tokens[None], np.zeros(flat.shape, dtype=np.int64), y, x)
    return {
        "frame": frames,
        "y": y.astype(np.uint16),
        "x": x.astype(np.uint16),
        "outcome": np.full(flat.shape, outcome, dtype=np.int8),
        "oriented_context": codes.astype(np.uint16),
        "center_token": tokens[y, x].astype(np.uint8),
        "scorer_source_class": source_class[y, x].astype(np.uint8),
        "scorer_target_class": target_class[y, x].astype(np.uint8),
        "scorer_transition": (source_class[y, x] * 5 + target_class[y, x]).astype(np.uint8),
        "distance_to_nearest_realized_b": distance_to_b[y, x].astype(np.float32),
        "distance_to_nearest_base_error": distance_to_base_error[y, x].astype(np.float32),
        "distance_to_semantic_boundary": distance_to_boundary[y, x].astype(np.float32),
        "pre_edit_margin": np.full(flat.shape, np.nan, dtype=np.float32),
    }


def concatenate(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise EC2AnalysisError("no per-cell events were materialized")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise EC2AnalysisError("per-cell schemas differ")
    return {key: np.concatenate([row[key] for row in rows]) for key in keys}


def extract_ec1(output: Path) -> tuple[Path, dict[str, np.ndarray]]:
    target = output / "retained/ec1_hb_cells.npz"
    if target.is_file():
        return target, dict(np.load(target, allow_pickle=False))
    base = load_field(BASE, np.dtype(np.uint8))
    gt = load_field(GT, np.dtype(np.uint8))
    ec1 = load_field(EC1, np.dtype(np.uint8))
    tokens = load_field(TOKENS, np.dtype(np.uint8))
    fixed = load_field(EC1_FIXED, np.dtype(np.bool_))
    harm = load_field(EC1_HARM, np.dtype(np.bool_))
    rows: list[dict[str, np.ndarray]] = []
    for frame in range(N_PAIRS):
        fixed_f = np.asarray(fixed[frame])
        harm_f = np.asarray(harm[frame])
        if not (fixed_f.any() or harm_f.any()):
            continue
        base_error = np.asarray(base[frame]) != np.asarray(gt[frame])
        distance_b = distance_transform_edt(~fixed_f) if fixed_f.any() else np.full(fixed_f.shape, np.inf)
        distance_error = distance_transform_edt(~base_error)
        token_f = np.asarray(tokens[frame])
        token_boundary = boundary_mask(token_f)
        distance_boundary = distance_transform_edt(~token_boundary)
        if fixed_f.any():
            rows.append(
                event_arrays(
                    frame=frame,
                    mask=fixed_f,
                    outcome=1,
                    tokens=token_f,
                    source_class=np.asarray(base[frame]),
                    target_class=np.asarray(gt[frame]),
                    distance_to_b=distance_b,
                    distance_to_base_error=distance_error,
                    distance_to_boundary=distance_boundary,
                )
            )
        if harm_f.any():
            rows.append(
                event_arrays(
                    frame=frame,
                    mask=harm_f,
                    outcome=-1,
                    tokens=token_f,
                    source_class=np.asarray(gt[frame]),
                    target_class=np.asarray(ec1[frame]),
                    distance_to_b=distance_b,
                    distance_to_base_error=distance_error,
                    distance_to_boundary=distance_boundary,
                )
            )
    arrays = concatenate(rows)
    if int(np.count_nonzero(arrays["outcome"] == 1)) != 12_075 or int(
        np.count_nonzero(arrays["outcome"] == -1)
    ) != 52_854:
        raise EC2AnalysisError("EC1 B/H denominator differs")
    atomic_npz(target, arrays)
    return target, arrays


def nearest_site_fields(sites: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site_mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    for row in sites:
        site_mask[int(row["y"]), int(row["x"])] = True
    distance, indices = distance_transform_edt(~site_mask, return_indices=True)
    return site_mask, distance, indices


def extract_qs3(output: Path) -> tuple[Path, dict[str, np.ndarray]]:
    target = output / "retained/qs3_hb_cells.npz"
    if target.is_file():
        return target, dict(np.load(target, allow_pickle=False))
    base = load_field(BASE, np.dtype(np.uint8))
    gt = load_field(GT, np.dtype(np.uint8))
    candidate = load_field(QS3_CANDIDATE, np.dtype(np.uint8))
    base_tokens = load_field(TOKENS, np.dtype(np.uint8))
    decomposition = json.loads(QS3_DECOMPOSITION.read_text())
    collateral_map = json.loads(QS4_MAP.read_text())
    records = {int(row["pair"]): row for row in collateral_map["per_proposal"]}
    bank_rows = [json.loads(line) for line in QS3_BANK.read_text().splitlines()]
    named_edges = {
        str(row["proposal_id"]): tuple(CLASS_TO_ID[name] for name in row["directed_edge"].split("->"))
        for row in bank_rows
    }
    rows: list[dict[str, np.ndarray]] = []
    for proposal_index, pair_text in enumerate(sorted(decomposition["per_pair"], key=int)):
        pair = int(pair_text)
        expected = decomposition["per_pair"][pair_text]
        proposal = records[pair]
        sites = [json.loads(line) for line in Path(proposal["site_attribution"]["path"]).read_text().splitlines()]
        site_mask, edit_distance, nearest_indices = nearest_site_fields(sites)
        proposal_tokens = np.load(proposal["candidate_tokens"]["path"], allow_pickle=False)
        if proposal_tokens.shape != (HEIGHT, WIDTH) or proposal_tokens.dtype != np.uint8:
            raise EC2AnalysisError("QS3 candidate-token geometry differs")
        base_f = np.asarray(base[pair])
        gt_f = np.asarray(gt[pair])
        candidate_f = np.asarray(candidate[pair])
        beneficial = (base_f != gt_f) & (candidate_f == gt_f)
        harmful = (base_f == gt_f) & (candidate_f != gt_f)
        if int(beneficial.sum()) != int(expected["B"]) or int(harmful.sum()) != int(expected["H"]):
            raise EC2AnalysisError(f"QS3 B/H denominator differs at pair {pair}")
        distance_b = distance_transform_edt(~beneficial)
        distance_error = distance_transform_edt(~(base_f != gt_f))
        distance_boundary = distance_transform_edt(~boundary_mask(proposal_tokens))
        for mask, outcome, source, target_class in (
            (beneficial, 1, base_f, gt_f),
            (harmful, -1, gt_f, candidate_f),
        ):
            event = event_arrays(
                frame=pair,
                mask=mask,
                outcome=outcome,
                tokens=proposal_tokens,
                source_class=source,
                target_class=target_class,
                distance_to_b=distance_b,
                distance_to_base_error=distance_error,
                distance_to_boundary=distance_boundary,
            )
            y = event["y"].astype(np.int64)
            x = event["x"].astype(np.int64)
            nearest_y = nearest_indices[0, y, x]
            nearest_x = nearest_indices[1, y, x]
            proposal_source = np.asarray(base_tokens[pair])[nearest_y, nearest_x]
            proposal_target = proposal_tokens[nearest_y, nearest_x]
            try:
                named_source, named_target = named_edges[str(proposal["proposal_id"])]
            except KeyError as exc:
                raise EC2AnalysisError(f"QS3 named proposal edge is absent at pair {pair}") from exc
            event.update(
                {
                    "proposal_index": np.full(y.shape, proposal_index, dtype=np.uint8),
                    "distance_to_nearest_edit": edit_distance[y, x].astype(np.float32),
                    "same_edit_site": site_mask[y, x].astype(np.bool_),
                    "nearest_edit_token_source_class": proposal_source.astype(np.uint8),
                    "nearest_edit_token_target_class": proposal_target.astype(np.uint8),
                    "nearest_edit_token_transition": (
                        proposal_source * 5 + proposal_target
                    ).astype(np.uint8),
                    "proposal_source_class": np.full(y.shape, named_source, dtype=np.uint8),
                    "proposal_target_class": np.full(y.shape, named_target, dtype=np.uint8),
                    "proposal_transition": np.full(
                        y.shape, named_source * 5 + named_target, dtype=np.uint8
                    ),
                }
            )
            rows.append(event)
    arrays = concatenate(rows)
    if int(np.count_nonzero(arrays["outcome"] == 1)) != 108 or int(
        np.count_nonzero(arrays["outcome"] == -1)
    ) != 76:
        raise EC2AnalysisError("QS3 B/H total differs")
    atomic_npz(target, arrays)
    return target, arrays


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    quantiles = np.quantile(finite.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(finite.size),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def class_pair_table(arrays: dict[str, np.ndarray], key: str) -> list[dict[str, Any]]:
    output = []
    for code in range(25):
        selected = arrays[key] == code
        beneficial = int(np.count_nonzero(selected & (arrays["outcome"] == 1)))
        harmful = int(np.count_nonzero(selected & (arrays["outcome"] == -1)))
        if beneficial or harmful:
            output.append(
                {
                    "source": code // 5,
                    "source_name": CLASS_NAMES[code // 5],
                    "target": code % 5,
                    "target_name": CLASS_NAMES[code % 5],
                    "B": beneficial,
                    "H": harmful,
                    "net": beneficial - harmful,
                    "beneficial_fraction_B_over_B_plus_H": beneficial / (beneficial + harmful),
                }
            )
    return sorted(output, key=lambda row: (row["net"], -row["H"]))


def gate_metrics(outcome: np.ndarray, keep: np.ndarray, name: str) -> dict[str, Any]:
    total_b = int(np.count_nonzero(outcome == 1))
    total_h = int(np.count_nonzero(outcome == -1))
    kept_b = int(np.count_nonzero((outcome == 1) & keep))
    kept_h = int(np.count_nonzero((outcome == -1) & keep))
    gross = kept_b + kept_h
    return {
        "variant": name,
        "B": kept_b,
        "H": kept_h,
        "net": kept_b - kept_h,
        "beneficial_fraction_B_over_B_plus_H": kept_b / gross if gross else 0.0,
        "B_retention": kept_b / total_b,
        "H_retention": kept_h / total_h,
        "B_suppression": 1.0 - kept_b / total_b,
        "H_suppression": 1.0 - kept_h / total_h,
    }


def crossfit_context_variant(
    arrays: dict[str, np.ndarray],
    *,
    threshold: float,
    minimum_observations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    codes = arrays["oriented_context"]
    outcome = arrays["outcome"]
    frames = arrays["frame"]
    keep = np.zeros(outcome.shape, dtype=np.bool_)
    for fold in range(5):
        test = frames % 5 == fold
        train = ~test
        counts = fit_context_counts(codes[train], outcome[train])
        gate = gate_from_context_counts(
            counts,
            minimum_beneficial_fraction=threshold,
            minimum_observations=minimum_observations,
            prior_beneficial=float(np.mean(outcome[train] == 1)),
            prior_strength=5.0,
        )
        keep[test] = gate.propose(codes[test])
    name = f"oriented_context_p{threshold:.6f}_min{minimum_observations}_fivefold"
    return gate_metrics(outcome, keep, name), keep


def build_characterization(
    ec1: dict[str, np.ndarray],
    qs3: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    result: dict[str, Any] = {
        "schema": "ddm_ec2_hb_characterization.v1",
        "axis": AXIS,
        "score_claim": False,
        "selection_mode": "full retained B/H populations; five folds by pair_id modulo 5; no prefix or sample",
        "pre_edit_margin": {
            "ec1_coverage": 0,
            "ec1_denominator": int(ec1["outcome"].size),
            "qs3_coverage": 0,
            "qs3_denominator": int(qs3["outcome"].size),
            "status": "UNAVAILABLE_NO_RETAINED_BASE_LOGITS; candidate post-edit margin is not substituted",
        },
        "feature_knowability": {
            "oriented_context": "KNOWN: decoded center and four neighbor semantic classes",
            "distance_to_semantic_boundary": "KNOWN: derived from decoded semantic tokens",
            "proposal_transition_qs3": "KNOWN: the proposal's named source-to-target semantic class pair",
            "nearest_edit_token_transition_qs3": "KNOWN but not the proposal edge: realized decoded classes at the nearest listed edit coordinate",
            "distance_to_nearest_edit_qs3": "KNOWN: proposal edit coordinates",
            "scorer_transition": "UNKNOWN: requires retained scorer argmax and GT outcome",
            "distance_to_nearest_realized_b": "UNKNOWN: hindsight outcome attribution",
            "distance_to_nearest_base_error": "UNKNOWN: requires scorer base/GT fields",
            "pre_edit_margin": "UNKNOWN AND UNRETAINED",
        },
        "objects": {},
    }
    for name, arrays in (("ec1", ec1), ("qs3", qs3)):
        result["objects"][name] = {
            "B": int(np.count_nonzero(arrays["outcome"] == 1)),
            "H": int(np.count_nonzero(arrays["outcome"] == -1)),
            "B_features": {
                key: describe(arrays[key][arrays["outcome"] == 1])
                for key in (
                    "distance_to_nearest_realized_b",
                    "distance_to_nearest_base_error",
                    "distance_to_semantic_boundary",
                    "pre_edit_margin",
                )
            },
            "H_features": {
                key: describe(arrays[key][arrays["outcome"] == -1])
                for key in (
                    "distance_to_nearest_realized_b",
                    "distance_to_nearest_base_error",
                    "distance_to_semantic_boundary",
                    "pre_edit_margin",
                )
            },
            "scorer_transition_pairs": class_pair_table(arrays, "scorer_transition"),
        }
    result["objects"]["qs3"]["proposal_transition_pairs"] = class_pair_table(qs3, "proposal_transition")
    result["objects"]["qs3"]["nearest_edit_token_transition_pairs"] = class_pair_table(
        qs3, "nearest_edit_token_transition"
    )
    result["objects"]["qs3"]["B_features"]["distance_to_nearest_edit"] = describe(
        qs3["distance_to_nearest_edit"][qs3["outcome"] == 1]
    )
    result["objects"]["qs3"]["H_features"]["distance_to_nearest_edit"] = describe(
        qs3["distance_to_nearest_edit"][qs3["outcome"] == -1]
    )

    variants = [gate_metrics(ec1["outcome"], np.ones(ec1["outcome"].shape, dtype=np.bool_), "no_suppression")]
    for threshold in (0.5, 1.0 / 1.89, 108.0 / 189.0):
        for minimum_observations in (1, 5, 20):
            row, _ = crossfit_context_variant(
                ec1,
                threshold=threshold,
                minimum_observations=minimum_observations,
            )
            row["proposal_visible"] = True
            variants.append(row)
    for radius in (0.0, 1.0, 2.0, 4.0, 8.0):
        row = gate_metrics(
            ec1["outcome"],
            ec1["distance_to_semantic_boundary"] <= radius,
            f"semantic_boundary_radius_le_{radius:g}",
        )
        row["proposal_visible"] = True
        variants.append(row)
    baseline_net = variants[0]["net"]
    eligible = [
        row
        for row in variants[1:]
        if row["net"] > baseline_net
        and row["H_suppression"] > row["B_suppression"]
        and row["variant"].startswith("oriented_context")
    ]
    selected = max(eligible, key=lambda row: (row["net"], row["B_retention"]), default=None)
    return result, variants, selected


def build_gate_payload(
    output: Path,
    ec1: dict[str, np.ndarray],
    selected: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, CollateralSuppressedProposer | None]:
    if selected is None:
        return None, None
    pieces = selected["variant"].split("_")
    threshold = float(pieces[2][1:])
    minimum_observations = int(pieces[3][3:])
    counts = fit_context_counts(ec1["oriented_context"], ec1["outcome"])
    gate = gate_from_context_counts(
        counts,
        minimum_beneficial_fraction=threshold,
        minimum_observations=minimum_observations,
        prior_beneficial=float(np.mean(ec1["outcome"] == 1)),
        prior_strength=5.0,
    )
    payload = gate.to_payload()
    repeat = gate.to_payload()
    if payload != repeat or not np.array_equal(
        CollateralSuppressedProposer.from_payload(payload).keep_by_context,
        gate.keep_by_context,
    ):
        raise EC2AnalysisError("gate payload is not deterministic and parse-back exact")
    payload_path = output / "retained/ec2_context_gate.br"
    repeat_path = output / "retained/ec2_context_gate.repeat.br"
    counts_path = output / "retained/ec2_context_counts.npz"
    atomic_bytes(payload_path, payload)
    atomic_bytes(repeat_path, repeat)
    atomic_npz(
        counts_path,
        {
            "beneficial": np.asarray(counts.beneficial),
            "harmful": np.asarray(counts.harmful),
            "keep": np.asarray(gate.keep_by_context),
        },
    )
    return (
        {
            "variant": selected["variant"],
            "payload": file_record(payload_path),
            "repeat": file_record(repeat_path),
            "counts": file_record(counts_path),
            "kept_contexts": int(np.count_nonzero(gate.keep_by_context)),
            "payload_parseback_exact": True,
            "payload_repeat_identical": True,
            "archive_delta_bytes": None,
            "archive_price_status": "UNMEASURED_UNTIL_INTEGRATED_REAL_CODER",
        },
        gate,
    )


def storage_preflight(output: Path) -> dict[str, Any]:
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    required = 512 * 1024 * 1024
    row = {
        "schema": "ddm_ec2_storage_preflight.v1",
        "tier": str(parent),
        "free_bytes": usage.free,
        "required_bytes": required,
        "passed": usage.free >= required,
        "policy": "APDataStore first; retain all per-cell payloads; certify-or-block",
    }
    if not row["passed"]:
        raise EC2AnalysisError("APDataStore free space is below the analysis reserve")
    checkpoint = output / "checkpoints/stage_00_storage_preflight.json"
    if output.exists():
        if not checkpoint.is_file():
            raise EC2AnalysisError("existing output lacks a storage-preflight checkpoint")
        retained = json.loads(checkpoint.read_text())
        if not retained.get("passed"):
            raise EC2AnalysisError("retained storage preflight did not pass")
        return retained
    output.mkdir(parents=False, exist_ok=False)
    atomic_json(checkpoint, row)
    return row


def run(output: Path) -> dict[str, Any]:
    preflight = storage_preflight(output)
    sources = {
        "tl1": file_record(TL1_MEMO, expected_sha256=TL1_SHA256),
        "ig1": file_record(IG1_MEMO, expected_sha256=IG1_SHA256),
        "base": file_record(BASE, expected_sha256=BASE_SHA256),
        "gt": file_record(GT, expected_sha256=GT_SHA256),
        "tokens": file_record(TOKENS, expected_sha256=TOKENS_SHA256),
        "ec1": file_record(EC1, expected_sha256=EC1_SHA256),
        "ec1_fixed": file_record(EC1_FIXED, expected_sha256=EC1_FIXED_SHA256),
        "ec1_harm": file_record(EC1_HARM, expected_sha256=EC1_HARM_SHA256),
        "qs3_candidate": file_record(QS3_CANDIDATE, expected_sha256=QS3_CANDIDATE_SHA256),
        "qs3_decomposition": file_record(
            QS3_DECOMPOSITION, expected_sha256=QS3_DECOMPOSITION_SHA256
        ),
        "qs3_full_bank": file_record(QS3_BANK, expected_sha256=QS3_BANK_SHA256),
        "qs4_map": file_record(QS4_MAP),
        "analysis_source": file_record(Path(__file__).resolve()),
        "proposer_source": file_record(
            ROOT / "src/tac/optimization/ec2_collateral_suppressed_proposer.py"
        ),
    }
    atomic_json(output / "checkpoints/stage_05_inputs_verified.json", sources)
    ec1_path, ec1 = extract_ec1(output)
    atomic_json(
        output / "checkpoints/stage_10_ec1_extracted.json",
        {"payload": file_record(ec1_path), "B": 12_075, "H": 52_854},
    )
    qs3_path, qs3 = extract_qs3(output)
    atomic_json(
        output / "checkpoints/stage_20_qs3_extracted.json",
        {"payload": file_record(qs3_path), "B": 108, "H": 76},
    )
    characterization, variants, selected = build_characterization(ec1, qs3)
    characterization_path = output / "H_B_CHARACTERIZATION.json"
    variants_path = output / "SUPPRESSION_VARIANTS.json"
    atomic_json(characterization_path, characterization)
    atomic_json(
        variants_path,
        {
            "schema": "ddm_ec2_suppression_variants.v1",
            "axis": AXIS,
            "variants": variants,
            "selected": selected,
            "prior_prediction_test": {
                "required_beneficial_fraction_strictly_above": 108.0 / 189.0,
                "required_B_retention": 0.95,
                "passed": bool(
                    selected is not None
                    and selected["beneficial_fraction_B_over_B_plus_H"] > 108.0 / 189.0
                    and selected["B_retention"] >= 0.95
                ),
            },
        },
    )
    gate_record, _ = build_gate_payload(output, ec1, selected)
    baseline = variants[0]
    if selected is None:
        selected_price = None
        selected_rc2_anchor_price = None
        selected_cp135_archive_price = None
    else:
        selected_price = collateral_priced_delta(
            expected_beneficial=float(selected["B"]),
            expected_harmful=float(selected["H"]),
            delta_archive_bytes=int(gate_record["payload"]["bytes"] if gate_record else 0),
        )
        selected_rc2_anchor_price = collateral_priced_delta(
            expected_beneficial=float(selected["B"]),
            expected_harmful=float(selected["H"]),
            delta_archive_bytes=1_176
            + int(gate_record["payload"]["bytes"] if gate_record else 0),
        )
        selected_cp135_archive_price = collateral_priced_delta(
            expected_beneficial=float(selected["B"]),
            expected_harmful=float(selected["H"]),
            delta_archive_bytes=1_471
            + int(gate_record["payload"]["bytes"] if gate_record else 0),
        )
    prior_passed = bool(
        selected is not None
        and selected["beneficial_fraction_B_over_B_plus_H"] > 108.0 / 189.0
        and selected["B_retention"] >= 0.95
    )
    projection = {
        "schema": "ddm_ec2_projection.v1",
        "projection_only": True,
        "current_dx2_score": 0.14821987563243377,
        "target": 0.12,
        "current_gap": 0.02821987563243377,
        "ec1_measured_B": int(baseline["B"]),
        "ec1_measured_H": int(baseline["H"]),
        "selected_fivefold_crossfit_B": None if selected is None else int(selected["B"]),
        "selected_fivefold_crossfit_H": None if selected is None else int(selected["H"]),
        "selected_gate_payload_bytes_not_archive_delta": None
        if gate_record is None
        else int(gate_record["payload"]["bytes"]),
        "selected_payload_only_delta_score_ignoring_pose_and_archive_container": None
        if selected_price is None
        else selected_price.joint_score,
        "rate_brackets": {
            "rc2_repin_adapter_anchor_bytes": 1_176,
            "cp135_original_adapter_archive_delta_bytes": 1_471,
            "gate_payload_bytes_not_archive_delta": None
            if gate_record is None
            else int(gate_record["payload"]["bytes"]),
            "crossfit_net_flips": None if selected is None else int(selected["net"]),
            "break_even_total_bytes_at_0_785_flips_per_byte": None
            if selected is None
            else float(selected["net"] / NET_FLIPS_PER_BYTE),
            "rc2_anchor_plus_raw_gate_projection_delta_score": None
            if selected_rc2_anchor_price is None
            else selected_rc2_anchor_price.joint_score,
            "cp135_archive_plus_raw_gate_projection_delta_score": None
            if selected_cp135_archive_price is None
            else selected_cp135_archive_price.joint_score,
            "boundary": "both are rate-only projections; the gate has no measured archive delta, pose, receiver closure, or realized gated field",
        },
        "selected_best_rate_bracket_projected_dx2_score_before_composability": None
        if selected_rc2_anchor_price is None
        else 0.14821987563243377 + selected_rc2_anchor_price.joint_score,
        "ideal_ec1_H_zero_seg_improvement": -12_075 * (100.0 / (N_PAIRS * HEIGHT * WIDTH)),
        "ideal_ec1_H_zero_projected_dx2_score_before_rate_pose_composability": 0.14821987563243377
        - 12_075 * (100.0 / (N_PAIRS * HEIGHT * WIDTH)),
        "sub_0_12_reached_even_in_ideal_H_zero_projection": False,
        "boundary": "EC1/QS3 fields are CP135 component objects, not receiver-closed dx2 composition; rate and pose require a fresh exact object",
    }
    projection_path = output / "SUB012_PROJECTION.json"
    atomic_json(projection_path, projection)
    fire_order = {
        "schema": "ddm_ec2_main_fire_order.v1",
        "disposition": "QUEUED-WITH-FIRE-ORDER, BLOCKED",
        "owner": "MAIN sole scorer-lane and exact-row owner",
        "consumer_store": "/Volumes/APDataStore/pact/ddm_ec2_collateral_suppressed_conditioner/main_fire/",
        "does_not_dispatch": True,
        "first_dispatch": None,
        "selected_gate": gate_record,
        "blockers": [
            "PRIOR_PREDICTION_NOT_MET_ON_RETAINED_FIELD"
            if not prior_passed
            else "NEW_CONDITIONER_NOT_TRAINED_OR_RECEIVER_CLOSED",
            "PRE_EDIT_MARGIN_UNRETAINED",
            "FRESH_SAME_OBJECT_COMPENSATION_NOT_COMPILED",
            "REAL_CODER_ARCHIVE_DELTA_UNMEASURED",
            "NO_DX2_RECEIVER_CLOSED_COMPOSITION",
        ],
        "fire_trigger": (
            "MAIN owns an idle unique contest-CUDA scorer lane; a reviewed from-scratch "
            "conditioner consumes the counted decoded-context gate inside training, prices H "
            "with the same full-field denominator as B, retains at least 95% of gross B while "
            "raising B/(B+H) strictly above 108/189 on a held-out stage, integrates a fresh "
            "same-object compensation solve, and real-coder parse-back proves negative complete "
            "projected delta before any scorer dispatch"
        ),
        "then": (
            "Seal a new request with exact source/payload/archive/runtime hashes and a non-null "
            "dispatch argv; do not reuse this blocked order as authorization"
        ),
    }
    fire_path = output / "SEALED_FIRE_ORDER.json"
    atomic_json(fire_path, fire_order)
    manifest = {
        "schema": "ddm_ec2_seal_manifest.v1",
        "axis": AXIS,
        "score_claim": False,
        "scorer_forwards_executed": 0,
        "modal_dispatched": False,
        "pointer_moved": False,
        "storage_preflight": preflight,
        "sources": sources,
        "outputs": {},
        "gate_selected": selected,
        "prior_prediction_passed": prior_passed,
        "fire_disposition": fire_order["disposition"],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json" and not path.name.startswith("._"):
            manifest["outputs"][str(path.relative_to(output))] = file_record(path)
    manifest_path = output / "MANIFEST.json"
    atomic_json(manifest_path, manifest)
    return {"manifest": file_record(manifest_path), "fire_order": file_record(fire_path)}


def verify(output: Path) -> dict[str, Any]:
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    for relative, record in manifest["outputs"].items():
        file_record(output / relative, expected_sha256=record["sha256"])
        checked += 1
    gate_record = json.loads((output / "SEALED_FIRE_ORDER.json").read_text()).get("selected_gate")
    if gate_record is not None:
        payload = Path(gate_record["payload"]["path"]).read_bytes()
        parsed = CollateralSuppressedProposer.from_payload(payload)
        if parsed.keep_by_context.shape != (ORIENTED_CONTEXTS,):
            raise EC2AnalysisError("verified gate shape differs")
    return {"verified": True, "payloads": checked, "manifest": file_record(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-seal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.output) if args.verify_seal else run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
