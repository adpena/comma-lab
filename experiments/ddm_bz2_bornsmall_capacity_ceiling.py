#!/usr/bin/env python3
"""Fit the complete HG1 born-small representation directly to registered DALI GT.

This is a CPU-only, scorer-free capacity probe.  It retains the deterministic
pair/spatial held-out controls, the full-n fit, every real-coder candidate, and
the complete receiver-parseable archive.  Native categorical disagreement is
reported only as a representation diagnostic; it is never relabelled d_seg.
The render/R/uint8/frozen-scorer terminal is emitted as a fail-closed fire order
because MAIN, not this arm, owns the single local scorer lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "experiments"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_et1_edge_topology_container_gate as et1
from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage

SCHEMA: Final = "ddm_bz2_bornsmall_capacity_ceiling.v1"
SEED: Final = 20260829
PAIR_HOLDOUT_COUNT: Final = 120
SPATIAL_HOLDOUT_MODULUS: Final = 5
RATE_DENOMINATOR: Final = 37_545_489
STRICT_TARGET: Final = 0.12
OUTPUT_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling")
GT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
GT_ARGMAX_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
SOURCE_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained")
SEMANTIC: Final = SOURCE_ROOT / "source_sections/semantic_renderer.bin"
POSE_CARRIER: Final = SOURCE_ROOT / "source_sections/inherited_pose_carrier.bin"
COMPACT_RESIDUAL: Final = SOURCE_ROOT / "source_sections/compact_residual.bin"
ZERO_RESIDUAL: Final = SOURCE_ROOT / "generators/residual_zero.raw"
ANCESTOR_TOKENS: Final = SOURCE_ROOT / "source_payloads/generated_tokens.u8"
BO2_RESULT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bo2_born_small_distortion/rows/"
    "hg1_generator_field/contest_auth_eval.json"
)
BO2_BASE_PERCLASS: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bo2_born_small_distortion/retained/perclass/"
    "perclass_base_dx2.json"
)
BO2_CANDIDATE_PERCLASS: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bo2_born_small_distortion/retained/perclass/"
    "perclass_hg1_generator.json"
)
EXPECTED_SOURCES: Final = {
    SEMANTIC: (30_856, "39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f"),
    POSE_CARRIER: (22_010, "932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12"),
    COMPACT_RESIDUAL: (96, "8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1"),
    ZERO_RESIDUAL: (20, "9239ac4d257773859b98109b5b822832b2587f92c11e6f8ca21708c15a53f9fa"),
    ANCESTOR_TOKENS: (117_964_800, "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"),
    BO2_RESULT: (36_442, "8defd0087f425a2642da38b76758b66d19dd7e2536a738e797d1d71f225a242f"),
    BO2_BASE_PERCLASS: (4_026, "ad8bc6b61a5a8bb00fa2d48d519a880bdb8559bbaa67ace96455857e0800089a"),
    BO2_CANDIDATE_PERCLASS: (4_101, "1eb6d21b587ef18668a44ad2f774972553ed2c915cafec9e1625242e4a0fdaed"),
}
MINIMUM_FREE_BYTES: Final = 5_000_000_000


class BZ2Error(RuntimeError):
    """Fail-closed source, storage, receiver, or authority refusal."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BZ2Error(f"required file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def same_fact(path: Path, expected: object) -> bool:
    if not isinstance(expected, dict) or not path.is_file():
        return False
    observed = fact(path)
    return (
        observed["bytes"] == expected.get("bytes")
        and observed["sha256"] == expected.get("sha256")
    )


def current_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def storage_preflight() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = OUTPUT_ROOT.resolve()
    allowed = Path("/Volumes/APDataStore/pact").resolve()
    if allowed not in resolved.parents:
        raise BZ2Error(f"output escaped AP custody: {resolved}")
    usage = shutil.disk_usage(OUTPUT_ROOT)
    if usage.free < MINIMUM_FREE_BYTES:
        raise BZ2Error(
            f"AP storage preflight refused: free={usage.free}, required={MINIMUM_FREE_BYTES}"
        )
    receipt = {
        "schema": f"{SCHEMA}.storage_preflight",
        "status": "PASS",
        "root": str(resolved),
        "free_bytes": usage.free,
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "retention": "certify-or-block; no fitted target, stream, coder payload, or archive is deleted",
    }
    atomic_json(OUTPUT_ROOT / "STORAGE_PREFLIGHT.json", receipt)
    return receipt


def source_preflight() -> dict[str, Any]:
    assert_gt_lineage(
        GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="ddm_bz2 direct born-small GT fit"
    )
    gt = fact(GT_ARGMAX)
    if gt["sha256"] != GT_ARGMAX_SHA256 or gt["bytes"] != 117_964_928:
        raise BZ2Error(f"registered DALI GT identity drifted: {gt}")
    sources: dict[str, Any] = {"gt_argmax": gt}
    for path, (expected_bytes, expected_sha) in EXPECTED_SOURCES.items():
        observed = fact(path)
        if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha:
            raise BZ2Error(f"frozen source identity drifted: {observed}")
        sources[path.name] = observed
    receipt = {
        "schema": f"{SCHEMA}.source_preflight",
        "status": "PASS",
        "axis": "[DALI GT lineage source identity; no score claim]",
        "git_head_at_execution": current_git_head(),
        "seed": SEED,
        "sources": sources,
    }
    atomic_json(OUTPUT_ROOT / "checkpoints/stage_00_source_preflight.json", receipt)
    return receipt


def pair_split() -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = np.random.default_rng(SEED).permutation(hg1.N_PAIRS)
    holdout = tuple(sorted(map(int, order[:PAIR_HOLDOUT_COUNT])))
    train = tuple(sorted(map(int, order[PAIR_HOLDOUT_COUNT:])))
    if len(train) != 480 or len(holdout) != 120 or set(train) & set(holdout):
        raise AssertionError("deterministic qbz1-crosswalk pair split differs")
    return train, holdout


def spatial_holdout_mask(pair_id: int) -> np.ndarray:
    y = np.arange(hg1.HEIGHT, dtype=np.int64)[:, None]
    x = np.arange(hg1.WIDTH, dtype=np.int64)[None, :]
    mixed = (x * 73_856_093) ^ (y * 19_349_663) ^ (pair_id * 83_492_791) ^ SEED
    return np.remainder(mixed, SPATIAL_HOLDOUT_MODULUS) == 0


def materialize_pair_control(gt: np.ndarray, train: tuple[int, ...], holdout: tuple[int, ...]) -> Path:
    path = OUTPUT_ROOT / "retained/targets/pair_holdout_nearest_train.u8"
    manifest_path = OUTPUT_ROOT / "retained/targets/pair_holdout_mapping.json"
    expected = int(np.prod(hg1.TOKEN_SHAPE))
    if path.is_file() and path.stat().st_size == expected:
        if not manifest_path.is_file():
            raise BZ2Error("pair-control target exists without its resume manifest")
        manifest = json.loads(manifest_path.read_text())
        if not same_fact(path, manifest.get("target")):
            raise BZ2Error("pair-control retained target drifted")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    target = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    train_array = np.asarray(train, dtype=np.int32)
    holdout_set = set(holdout)
    source_pairs: list[int] = []
    for pair_id in range(hg1.N_PAIRS):
        if pair_id in holdout_set:
            distances = np.abs(train_array - pair_id)
            source_id = int(train_array[np.argmin(distances)])
        else:
            source_id = pair_id
        source_pairs.append(source_id)
        target[pair_id] = gt[source_id]
    target.flush()
    del target
    os.replace(temporary, path)
    atomic_json(
        manifest_path,
        {
            "schema": f"{SCHEMA}.pair_control",
            "train_pairs": list(train),
            "holdout_pairs": list(holdout),
            "source_pair_for_each_output_pair": source_pairs,
            "rule": "heldout pair receives nearest absolute-id training pair, lower id wins ties",
            "target": fact(path),
        },
    )
    return path


def materialize_spatial_control(gt: np.ndarray) -> Path:
    path = OUTPUT_ROOT / "retained/targets/spatial_holdout_nearest_observed.u8"
    manifest_path = OUTPUT_ROOT / "retained/targets/spatial_holdout_contract.json"
    expected = int(np.prod(hg1.TOKEN_SHAPE))
    if path.is_file() and path.stat().st_size == expected:
        if not manifest_path.is_file():
            raise BZ2Error("spatial-control target exists without its resume manifest")
        manifest = json.loads(manifest_path.read_text())
        if not same_fact(path, manifest.get("target")):
            raise BZ2Error("spatial-control retained target drifted")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    target = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    for pair_id in range(hg1.N_PAIRS):
        frame = np.asarray(gt[pair_id])
        holdout = spatial_holdout_mask(pair_id)
        indices = ndimage.distance_transform_edt(
            holdout, return_distances=False, return_indices=True
        )
        filled = frame.copy()
        filled[holdout] = frame[indices[0][holdout], indices[1][holdout]]
        target[pair_id] = filled
    target.flush()
    del target
    os.replace(temporary, path)
    atomic_json(
        manifest_path,
        {
            "schema": f"{SCHEMA}.spatial_control",
            "seed": SEED,
            "modulus": SPATIAL_HOLDOUT_MODULUS,
            "holdout_rule": "((x*73856093) xor (y*19349663) xor (pair*83492791) xor seed) % 5 == 0",
            "fill_rule": "nearest observed spatial label by scipy Euclidean distance transform",
            "target": fact(path),
        },
    )
    return path


def fit_and_render(stage: str, target: np.ndarray) -> dict[str, Any]:
    stage_root = OUTPUT_ROOT / "retained/fits" / stage
    result_path = OUTPUT_ROOT / "checkpoints" / f"stage_{stage}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        input_path = Path(str(target.filename))
        expected_outputs = [result.get("input_target"), result.get("generated")]
        observed_paths = [input_path, Path(str(result.get("generated", {}).get("path", "")))]
        for stream in hg1.GENERATOR_STREAMS:
            expected_outputs.append(result.get("streams", {}).get(stream))
            observed_paths.append(Path(str(result.get("streams", {}).get(stream, {}).get("path", ""))))
        if not all(
            same_fact(path, expected)
            for path, expected in zip(observed_paths, expected_outputs, strict=True)
        ):
            raise BZ2Error(f"fit-stage resume bytes drifted: {stage}")
        return result
    stage_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    horizon = hg1.fit_horizon_payload(target)
    lane, lane_meta = hg1.fit_lane_payload(target)
    movable, movable_meta = hg1.fit_movable_payload(target)
    mycar, mycar_meta = hg1.fit_mycar_payload(target)
    streams = {
        "road_undrivable": horizon,
        "lane": lane,
        "movable": movable,
        "mycar": mycar,
    }
    stream_facts = {}
    for name, payload in streams.items():
        path = stage_root / f"{name}.raw"
        et1.atomic_bytes(path, payload)
        stream_facts[name] = fact(path)
    generated_path = stage_root / "generated_tokens.u8"
    hg1.render_generators(streams, generated_path)
    result = {
        "schema": f"{SCHEMA}.fit_stage",
        "stage": stage,
        "status": "COMPLETE",
        "seconds": time.monotonic() - started,
        "input_target": fact(Path(str(target.filename))),
        "streams": stream_facts,
        "generated": fact(generated_path),
        "metadata": {"lane": lane_meta, "movable": movable_meta, "mycar": mycar_meta},
    }
    atomic_json(result_path, result)
    return result


def mismatch_report(
    generated_path: Path,
    gt: np.ndarray,
    *,
    pair_ids: tuple[int, ...] | None = None,
    spatial_part: str | None = None,
) -> dict[str, Any]:
    generated = np.memmap(generated_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    ids = pair_ids if pair_ids is not None else tuple(range(hg1.N_PAIRS))
    confusion = np.zeros((5, 5), dtype=np.int64)
    misses = 0
    denominator = 0
    for pair_id in ids:
        reference = np.asarray(gt[pair_id])
        candidate = np.asarray(generated[pair_id])
        selector = np.ones(reference.shape, dtype=bool)
        if spatial_part is not None:
            holdout = spatial_holdout_mask(pair_id)
            selector = holdout if spatial_part == "holdout" else ~holdout
        ref = reference[selector]
        got = candidate[selector]
        denominator += int(ref.size)
        misses += int(np.count_nonzero(ref != got))
        confusion += np.bincount(ref.astype(np.int64) * 5 + got, minlength=25).reshape(5, 5)
    del generated
    return {
        "axis": "[native categorical representation diagnostic; NOT d_seg]",
        "pairs": len(ids),
        "positions": denominator,
        "mismatches": misses,
        "mismatch_fraction": misses / denominator,
        "confusion_gt_by_generated": confusion.tolist(),
    }


def ancestor_comparison(generated_path: Path, gt: np.ndarray) -> dict[str, Any]:
    ancestor = np.memmap(ANCESTOR_TOKENS, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    generated = np.memmap(generated_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    ancestor_misses = 0
    generated_misses = 0
    changed_sites = 0
    for pair_id in range(hg1.N_PAIRS):
        reference = np.asarray(gt[pair_id])
        old = np.asarray(ancestor[pair_id])
        new = np.asarray(generated[pair_id])
        ancestor_misses += int(np.count_nonzero(reference != old))
        generated_misses += int(np.count_nonzero(reference != new))
        changed_sites += int(np.count_nonzero(old != new))
    del ancestor, generated
    return {
        "schema": f"{SCHEMA}.native_ancestor_comparison",
        "axis": "[native categorical representation diagnostic; NOT d_seg]",
        "positions": hg1.TOTAL_POSITIONS,
        "ancestor": {
            "source": fact(ANCESTOR_TOKENS),
            "gt_mismatches": ancestor_misses,
            "gt_mismatch_fraction": ancestor_misses / hg1.TOTAL_POSITIONS,
        },
        "direct_dali_fit": {
            "source": fact(generated_path),
            "gt_mismatches": generated_misses,
            "gt_mismatch_fraction": generated_misses / hg1.TOTAL_POSITIONS,
        },
        "ancestor_to_direct_fit_changed_sites": changed_sites,
        "warning": "native categorical comparison only; no scorer or score credit",
    }


def package_full_fit(full_result: dict[str, Any], gt: np.ndarray) -> dict[str, Any]:
    package_root = OUTPUT_ROOT / "retained/full_package"
    package_root.mkdir(parents=True, exist_ok=True)
    zero_path = package_root / "residual_zero.raw"
    et1.atomic_bytes(zero_path, ZERO_RESIDUAL.read_bytes())
    races = [
        hg1.coder_race(name, Path(full_result["streams"][name]["path"]), OUTPUT_ROOT)
        for name in hg1.GENERATOR_STREAMS
    ]
    races.append(hg1.coder_race("residual", zero_path, OUTPUT_ROOT))
    packet_path = package_root / "bornsmall_gt_fit.packet"
    packet_fact = hg1.build_packet(races, packet_path)
    packet = packet_path.read_bytes()
    archive_path = package_root / "archive.zip"
    repeat_path = package_root / "archive.repeat.zip"
    semantic = SEMANTIC.read_bytes()
    carrier = POSE_CARRIER.read_bytes()
    compact = COMPACT_RESIDUAL.read_bytes()
    hg1.build_complete_archive(archive_path, packet, semantic, carrier, compact)
    hg1.build_complete_archive(repeat_path, packet, semantic, carrier, compact)
    if archive_path.read_bytes() != repeat_path.read_bytes():
        raise BZ2Error("complete archive deterministic repeat differs")
    sections, parsed_packet = hg1.parse_complete_archive(archive_path)
    if parsed_packet != packet or sections != {
        "semantic_renderer": semantic,
        "pose_carrier": carrier,
        "compact_residual": compact,
    }:
        raise BZ2Error("complete archive section parse-back differs")
    decoded_path = package_root / "archive_parseback_tokens.u8"
    decoded_fact = hg1.decode_packet_to_file(parsed_packet, decoded_path)
    generated_path = Path(full_result["generated"]["path"])
    if sha256_file(decoded_path) != sha256_file(generated_path):
        raise BZ2Error("packet parse-back field differs from fitted generator field")
    archive_bytes = archive_path.stat().st_size
    return {
        "schema": f"{SCHEMA}.full_package",
        "status": "COMPLETE",
        "axis": "[macOS-CPU scorer-free exact byte measurement; n600 DALI target fit]",
        "scorer_loaded": False,
        "races": races,
        "packet": packet_fact,
        "archive": fact(archive_path),
        "archive_repeat": fact(repeat_path),
        "deterministic_repeat_equal": True,
        "parseback": decoded_fact,
        "parseback_equals_fitted_field": True,
        "native_full_n": mismatch_report(decoded_path, gt),
        "archive_vs_fixed_distortion_cap_bytes": archive_bytes - 137_986,
        "rate_at_exact_archive_bytes": 25 * archive_bytes / RATE_DENOMINATOR,
        "realized_d_seg": None,
        "realized_d_pose": None,
        "capacity_ceiling_verdict": "QUEUED: native field and bytes close; render/R/uint8/frozen-scorer realization is unmeasured",
    }


def rederive_bo2() -> dict[str, Any]:
    bo2 = json.loads(BO2_RESULT.read_text())
    base_perclass = json.loads(BO2_BASE_PERCLASS.read_text())
    candidate_perclass = json.loads(BO2_CANDIDATE_PERCLASS.read_text())
    local_base_dseg_precise = float(base_perclass["avg_segnet_dist_recomputed"])
    local_candidate_dseg_precise = float(candidate_perclass["avg_segnet_dist_recomputed"])
    local_base_dseg = 0.0003474
    local_candidate_dseg = float(bo2["avg_segnet_dist"])
    local_candidate_dpose = float(bo2["avg_posenet_dist"])
    # BO2's matched local base pose component is recovered by subtracting the
    # published local base seg term from its retained distortion subtotal.
    local_base_distortion = 0.07308188310451118
    local_base_pose_term = local_base_distortion - 100 * local_base_dseg
    local_base_dpose = local_base_pose_term * local_base_pose_term / 10
    local_candidate_distortion = 100 * local_candidate_dseg + math.sqrt(10 * local_candidate_dpose)
    achieved_damage = local_candidate_distortion - local_base_distortion
    authority_dseg = 0.00020139
    authority_dpose = 0.00000637
    authority_distortion = 100 * authority_dseg + math.sqrt(10 * authority_dpose)
    ideal_archive_bytes = 101_128
    strict_cap = math.floor((STRICT_TARGET - authority_distortion) * RATE_DENOMINATOR / 25)
    ideal_budget = STRICT_TARGET - authority_distortion - 25 * ideal_archive_bytes / RATE_DENOMINATOR
    actual_archive_bytes = 101_150
    actual_budget = STRICT_TARGET - authority_distortion - 25 * actual_archive_bytes / RATE_DENOMINATOR
    return {
        "schema": f"{SCHEMA}.bo2_rederivation",
        "status": "PASS",
        "matched_local_pyav_measurement": {
            "base_d_seg": local_base_dseg,
            "base_d_seg_recomputed_full_precision": local_base_dseg_precise,
            "base_d_pose_recovered_from_retained_subtotal": local_base_dpose,
            "base_distortion": local_base_distortion,
            "candidate_d_seg": local_candidate_dseg,
            "candidate_d_seg_recomputed_full_precision": local_candidate_dseg_precise,
            "candidate_d_pose": local_candidate_dpose,
            "candidate_distortion": local_candidate_distortion,
            "candidate_minus_base_distortion": achieved_damage,
            "instrument_warning": "matched local PYAV advisory delta; not a contest-CUDA score delta",
        },
        "contest_cuda_fixed_distortion_envelope": {
            "d_seg": authority_dseg,
            "d_pose": authority_dpose,
            "distortion": authority_distortion,
            "strict_integer_archive_cap_bytes": strict_cap,
        },
        "bo2_ideal_arithmetic": {
            "archive_bytes": ideal_archive_bytes,
            "headroom_bytes": strict_cap - ideal_archive_bytes,
            "distortion_budget": ideal_budget,
            "refusal_multiple": achieved_damage / ideal_budget,
            "cross_instrument_boundary": "matched local achieved damage divided by contest-CUDA fixed-distortion budget; robust refusal envelope, not a same-axis score",
        },
        "retained_physical_container": {
            "archive_bytes": actual_archive_bytes,
            "headroom_bytes": strict_cap - actual_archive_bytes,
            "distortion_budget": actual_budget,
            "projected_refusal_multiple_if_local_damage_transferred": achieved_damage / actual_budget,
            "warning": "the 101150-byte physical container has no matched n600 contest-CUDA distortion receipt",
        },
        "ranking_correction": {
            "exchange_ratio": 97.25,
            "rank": "second-best of five measured regimes under BO2 convention A",
            "withdrawn_claim": "fourth-worst ranking must not be reused",
        },
        "sources": {
            "bo2_result": fact(BO2_RESULT),
            "bo2_base_perclass": fact(BO2_BASE_PERCLASS),
            "bo2_candidate_perclass": fact(BO2_CANDIDATE_PERCLASS),
        },
    }


def fire_order(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}.fire_order",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN scorer-lane router",
        "consumer_store": str(OUTPUT_ROOT / "realized_capacity_terminal"),
        "fire_trigger": "FCD3 releases the sole local scorer lane; MAIN records an active BZ2 claim; all source/archive hashes revalidate",
        "action": "Render the retained fitted field through the inherited HG1 semantic renderer, exact R, and uint8; run the frozen DALI-lineage SegNet/PoseNet terminal at real n600; retain every RGB/YUV/argmax/pose chunk and exact receipt",
        "candidate_archive": package["archive"],
        "candidate_parseback_field": package["parseback"],
        "required_chunks": "at most 120 pairs per checkpointed chunk; atomic distinct-stage checkpoints; resume from last complete chunk",
        "measured_predecessor_cost": {
            "source": fact(BO2_RESULT),
            "inflate_seconds": 380.34438875014894,
            "evaluate_seconds": 457.91787716699764,
            "total_seconds": 839.5104122920893,
            "inflated_raw_bytes": 3_662_409_600,
            "cash_cost_usd": 0,
            "scope": "BO2 local CPU exact runtime/scorer predecessor; measured cost, not a prediction of BZ2 equality",
        },
        "fail_closed": "no d_seg, d_pose, capacity ceiling, cross-object law, or score is claimed before this terminal completes",
    }


def run() -> dict[str, Any]:
    started = time.monotonic()
    storage = storage_preflight()
    sources = source_preflight()
    bo2 = rederive_bo2()
    atomic_json(OUTPUT_ROOT / "BO2_REDERIVATION.json", bo2)
    gt = np.load(GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    if gt.shape != hg1.TOKEN_SHAPE or gt.dtype != np.uint8:
        raise BZ2Error(f"DALI GT geometry differs: {gt.shape}/{gt.dtype}")
    train, holdout = pair_split()

    pair_target_path = materialize_pair_control(gt, train, holdout)
    pair_target = np.memmap(pair_target_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    pair_fit = fit_and_render("10_pair_holdout", pair_target)
    pair_metrics = {
        "train": mismatch_report(Path(pair_fit["generated"]["path"]), gt, pair_ids=train),
        "holdout": mismatch_report(Path(pair_fit["generated"]["path"]), gt, pair_ids=holdout),
    }
    del pair_target
    atomic_json(OUTPUT_ROOT / "PAIR_HOLDOUT_RESULT.json", pair_metrics)

    spatial_target_path = materialize_spatial_control(gt)
    spatial_target = np.memmap(spatial_target_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    spatial_fit = fit_and_render("20_spatial_holdout", spatial_target)
    spatial_metrics = {
        "train": mismatch_report(Path(spatial_fit["generated"]["path"]), gt, spatial_part="train"),
        "holdout": mismatch_report(Path(spatial_fit["generated"]["path"]), gt, spatial_part="holdout"),
    }
    del spatial_target
    atomic_json(OUTPUT_ROOT / "SPATIAL_HOLDOUT_RESULT.json", spatial_metrics)

    full_target_path = OUTPUT_ROOT / "retained/targets/dali_gt_full_n600.u8"
    full_target_manifest = OUTPUT_ROOT / "retained/targets/dali_gt_full_n600.json"
    if full_target_path.is_file():
        if full_target_manifest.is_file():
            manifest = json.loads(full_target_manifest.read_text())
            if not same_fact(full_target_path, manifest.get("target")):
                raise BZ2Error("retained full-n DALI target drifted")
        else:
            retained = np.memmap(
                full_target_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE
            )
            if any(not np.array_equal(retained[pair], gt[pair]) for pair in range(hg1.N_PAIRS)):
                raise BZ2Error("unmanifested retained full-n target differs from registered DALI GT")
            del retained
            atomic_json(
                full_target_manifest,
                {"schema": f"{SCHEMA}.full_target", "source": fact(GT_ARGMAX), "target": fact(full_target_path)},
            )
    else:
        full_target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = full_target_path.with_name(f".{full_target_path.name}.{os.getpid()}.tmp")
        target = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
        target[:] = gt[:]
        target.flush()
        del target
        os.replace(temporary, full_target_path)
        atomic_json(
            full_target_manifest,
            {"schema": f"{SCHEMA}.full_target", "source": fact(GT_ARGMAX), "target": fact(full_target_path)},
        )
    full_target = np.memmap(full_target_path, mode="r", dtype=np.uint8, shape=hg1.TOKEN_SHAPE)
    full_fit = fit_and_render("30_full_n600", full_target)
    del full_target
    package = package_full_fit(full_fit, gt)
    atomic_json(OUTPUT_ROOT / "FULL_PACKAGE_RESULT.json", package)
    ancestor = ancestor_comparison(Path(package["parseback"]["path"]), gt)
    atomic_json(OUTPUT_ROOT / "NATIVE_ANCESTOR_COMPARISON.json", ancestor)
    order = fire_order(package)
    atomic_json(OUTPUT_ROOT / "FIRE_ORDER.json", order)
    result = {
        "schema": SCHEMA,
        "status": "PARTIAL-PROVED-AND-QUEUED",
        "axis": "[macOS-CPU scorer-free exact byte measurement; n600 DALI target fit]",
        "seed": SEED,
        "storage": storage,
        "sources": sources,
        "qbz1_protocol_crosswalk": {
            "seed": SEED,
            "pair_split": "same seeded 480 train / 120 holdout partition",
            "spatial_split": "same xor hash and modulus-5 holdout",
            "difference": "BZ2 fits the frozen HG1 analytic representation; it does not duplicate qbz1's QBF1/qbt object",
        },
        "pair_holdout": pair_metrics,
        "spatial_holdout": spatial_metrics,
        "full_package": package,
        "native_ancestor_comparison": ancestor,
        "bo2_rederivation": bo2,
        "capacity_ceiling": None,
        "cross_object_law": {
            "status": "UNRESOLVED",
            "n_measured_ceilings": 0,
            "maximum_allowed": 2,
            "reason": "BZ2 realization is queued and qbz1's separate ceiling is not consumed; no law is fit to native proxies",
        },
        "fire_order": order,
        "elapsed_seconds": time.monotonic() - started,
        "all_materialized_payloads_retained": True,
        "no_scorer_lane_claimed": True,
        "score_claim": False,
    }
    atomic_json(OUTPUT_ROOT / "RESULT.json", result)
    inventory_path = OUTPUT_ROOT / "RETAINED_INVENTORY.json"
    inventory = [
        fact(path)
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path != inventory_path
    ]
    atomic_json(
        inventory_path,
        {"schema": f"{SCHEMA}.inventory", "files_before_manifest": inventory},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "preflight"), nargs="?", default="run")
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps({"storage": storage_preflight(), "sources": source_preflight()}, indent=2))
        return
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
