#!/usr/bin/env python3
"""Measure the physical and pair-bootstrap exchange-ratio noise floor.

The physical leg performs three complete RC64 null re-encodes through RXC1 and
retains every stream, archive, ledger, and checkpoint.  The statistical leg
resamples the 600 pair contributions from the retained JBP1 and FCD3 ledgers.
It never imports or runs a scorer: FCD3 distortion is read only from the two
already-retained full-population scorer receipts.

The bootstrap is exact-total calibrated.  Per-pair ideal codelength deltas
carry sampling variation; the sub-byte RC64 rounding residual and any fixed
container bytes remain fixed in every resample.  Consequently the original
sample sums exactly to the physical integer-byte delta without pretending that
an independently encoded bootstrap sequence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

rxc1 = importlib.import_module("experiments.ddm_rxc1_restartable_exact_coder")


STORE = Path("/Volumes/VertigoDataTier/pact/ddm_xr1_exchange_ratio_noise_floor")
RXC1_STORE = Path("/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder")
JBP1_STORE = Path("/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price")
RN1_STORE = Path("/Volumes/APDataStore/pact/ddm_rn1_n600_reopen_sweep")
FCD1_STORE = Path("/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal")
FCD3_STORE = FCD1_STORE / "fcd3_pose_screened_reselection"

PAIR_COUNT = 600
PHYSICAL_REPEATS = 3
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 20_260_903
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_NUMERATOR = 25.0
PROJECTED_OUTPUT_BYTES = 128 << 20
MANDATORY_RESERVE_BYTES = 1 << 30
AXIS = "[macOS-CPU advisory / scorer-free exact byte replay plus retained-score bootstrap]"

NULL_OVERLAY = JBP1_STORE / "retained/overlays/null.pair_planes.npz"
JBP1_BASE_LEDGER = JBP1_STORE / "retained/exact/null/bits_per_frame_exact.npy"
JBP1_CANDIDATE_LEDGER = JBP1_STORE / "retained/exact/xov1_bhw5506/bits_per_frame_exact.npy"
JBP1_BASE_RESULT = JBP1_STORE / "retained/exact/null/RESULT.json"
JBP1_CANDIDATE_RESULT = JBP1_STORE / "retained/exact/xov1_bhw5506/RESULT.json"
RN1_NEAR_WINS = RN1_STORE / "near_win_candidates.jsonl"
FCD3_BASE_LEDGER = FCD1_STORE / "reencode/work/bits_per_frame_control_600.npy"
FCD3_CANDIDATE_LEDGER = FCD3_STORE / "reencode/work/bits_per_frame_fcd3_tau_1e-6.npy"
FCD3_BASE_SCORER = FCD3_STORE / "full_scorer/base_jt21/11_batch_replay_receipt.json"
FCD3_CANDIDATE_SCORER = FCD3_STORE / "full_scorer/tau_1e-6/11_batch_replay_receipt.json"
FCD3_BASE_ARCHIVE = FCD1_STORE / "runtimes/base_jt21/archive.zip"
FCD3_BODY_ARCHIVE = FCD3_STORE / "runtimes/candidate_tau_1e-6/archive.zip"
FCD3_PUBLISHED_ARCHIVE = FCD3_STORE / "runtimes/published_tau_1e-6/archive.zip"

# Every reused physical artifact is pinned before any encode starts.  These are
# the exact charter-selected objects, not convenience substitutes.
INPUT_PINS: tuple[tuple[Path, int, str], ...] = (
    (
        REPO / ".omx/research/charters/ddm_xr1_exchange_ratio_noise_floor_20260903.md",
        5_977,
        "f811f0c5aa063e98446c82917fb32e0c8534e0cab8667f89be0f6b19cee75ab9",
    ),
    (
        REPO / ".omx/tmp/codex_runs/_common_contract.md",
        4_124,
        "eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771",
    ),
    (
        REPO / "experiments/ddm_rxc1_restartable_exact_coder.py",
        37_869,
        "115bb907520b0996bca6e3c00be34ed341266185ac3802a05cba5efe628f76e9",
    ),
    (
        REPO / "experiments/ddm_jbp1_joint_batch_price.py",
        28_204,
        "8c5b8d88cddc35a8f69f899cfa32520857adbb2ca8125c510e9607a88d8d5dc3",
    ),
    (NULL_OVERLAY, 22, "8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85"),
    (
        JBP1_BASE_LEDGER,
        4_928,
        "9954e90eb88fe5227f2899a569fe47105cf34473e12d0c018fcdc8b176722d1d",
    ),
    (
        JBP1_CANDIDATE_LEDGER,
        4_928,
        "1c0841a69f23cf7f0c706ab0d5ff3a6c4f78568ac6387d1017671a1f50082d85",
    ),
    (
        JBP1_BASE_RESULT,
        4_167,
        "0fa702befd2dce3a37a722cd1335a4abdda14e4dbb40e890098d470596b91bf9",
    ),
    (
        JBP1_CANDIDATE_RESULT,
        10_385,
        "d1d0268927b6062591bbc7756af89ccbcf309103141919824beac15f5ff8c44c",
    ),
    (
        RN1_NEAR_WINS,
        165_902,
        "9d2b20cb8861e24c23cd78f277bbb59868c8c1f3a184e4b161a538259a2c0156",
    ),
    (
        FCD3_BASE_LEDGER,
        4_928,
        "68dc35968c1f0ed1a1b15ef7e0dd65ad85f5cd23183833d773b333bd50fbd87b",
    ),
    (
        FCD3_CANDIDATE_LEDGER,
        4_928,
        "36a30f55b022fc2c68847e65898ae1ccec2656890dd13c89942e90e9f0fe5f86",
    ),
    (
        FCD3_BASE_SCORER,
        57_187,
        "4271bf2f54f3fce5f20a9c328e0da997300b20a14c3d44ff6b6d622bd42a9db4",
    ),
    (
        FCD3_CANDIDATE_SCORER,
        57_277,
        "2cfc37273cf4a6c1d465aefebe38800d59f9e0a587f1adef1869e60c9f2283cc",
    ),
    (
        FCD3_BASE_ARCHIVE,
        180_192,
        "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
    ),
    (
        FCD3_BODY_ARCHIVE,
        177_227,
        "a08ba488ba8dddf48e763420e46b43adf470baf4a2fb9df763a051639abce014",
    ),
    (
        FCD3_PUBLISHED_ARCHIVE,
        177_252,
        "a4913f44d261d5272fc2b83dffdcad1bf5e4b757c648e2d8207c3eb7f428f6ac",
    ),
)


class Xr1Error(RuntimeError):
    """A custody, population, arithmetic, or retention gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise Xr1Error(f"missing pinned input: {path}")
    fact = file_fact(path)
    if fact["bytes"] != expected_bytes or fact["sha256"] != expected_sha256:
        raise Xr1Error(f"custody pin mismatch for {path}: bytes={fact['bytes']} sha256={fact['sha256']}")
    return fact


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_commit(path: Path) -> str:
    relative = path.resolve().relative_to(REPO)
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise Xr1Error(f"runner must be committed before launch: {relative}")
    return commit


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(data, [0.025, 0.975], method="linear")
    return {
        "low": float(low),
        "high": float(high),
        "width": float(high - low),
        "half_width": float((high - low) / 2.0),
    }


def score_delta(
    *,
    base_d_seg: float,
    candidate_d_seg: float,
    base_d_pose: float,
    candidate_d_pose: float,
    delta_bytes: float,
) -> dict[str, float]:
    delta_seg = 100.0 * (candidate_d_seg - base_d_seg)
    delta_pose = math.sqrt(10.0 * candidate_d_pose) - math.sqrt(10.0 * base_d_pose)
    delta_rate = RATE_NUMERATOR * delta_bytes / RATE_DENOMINATOR_BYTES
    delta_distortion = delta_seg + delta_pose
    return {
        "delta_s_seg": delta_seg,
        "delta_s_pose": delta_pose,
        "delta_s_distortion": delta_distortion,
        "delta_s_rate": delta_rate,
        "delta_s": delta_distortion + delta_rate,
        "exchange_ratio": delta_rate / delta_distortion,
    }


def exact_total_calibrated_bootstrap(
    base_bits: np.ndarray,
    candidate_bits: np.ndarray,
    draw_indices: np.ndarray,
    *,
    exact_delta_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bootstrap pair codelength contributions while preserving exact-total calibration."""
    base = np.asarray(base_bits, dtype=np.float64)
    candidate = np.asarray(candidate_bits, dtype=np.float64)
    draws = np.asarray(draw_indices)
    if base.shape != (PAIR_COUNT,) or candidate.shape != (PAIR_COUNT,):
        raise Xr1Error("byte ledgers must each be exactly n600")
    if draws.ndim != 2 or draws.shape[1] != PAIR_COUNT:
        raise Xr1Error("bootstrap draw matrix must have 600 pair indices per resample")
    pair_delta = (candidate - base) / 8.0
    fixed_adjustment = float(exact_delta_bytes - pair_delta.sum())
    samples = pair_delta[draws].sum(axis=1) + fixed_adjustment
    if not math.isclose(float(pair_delta.sum() + fixed_adjustment), exact_delta_bytes, abs_tol=1e-9):
        raise Xr1Error("exact-total byte calibration failed")
    return pair_delta, samples, fixed_adjustment


def exact_mean_calibrated_bootstrap(
    values: np.ndarray,
    draw_indices: np.ndarray,
    *,
    exact_mean: float,
) -> tuple[np.ndarray, float]:
    """Bootstrap pair values with the retained aggregate's rounding residual fixed."""
    data = np.asarray(values, dtype=np.float64)
    draws = np.asarray(draw_indices)
    if data.shape != (PAIR_COUNT,):
        raise Xr1Error("distortion vector must be exactly n600")
    if draws.ndim != 2 or draws.shape[1] != PAIR_COUNT:
        raise Xr1Error("bootstrap draw matrix must have 600 pair indices per resample")
    fixed_adjustment = float(exact_mean - data.mean())
    samples = data[draws].mean(axis=1) + fixed_adjustment
    return samples, fixed_adjustment


def scorer_pair_vectors(receipt: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if int(receipt.get("pair_count", -1)) != PAIR_COUNT:
        raise Xr1Error("scorer receipt is not n600")
    seg = np.empty(PAIR_COUNT, dtype=np.float64)
    pose = np.empty(PAIR_COUNT, dtype=np.float64)
    seen: set[int] = set()
    for stage in receipt.get("batch_stages", []):
        start = int(stage["pair_start"])
        stop = int(stage["pair_stop_exclusive"])
        pair_ids = range(start, stop)
        seg_values = stage["d_seg_per_pair"]
        pose_values = stage["d_pose_per_pair"]
        if len(seg_values) != stop - start or len(pose_values) != stop - start:
            raise Xr1Error("per-pair scorer stage length drifted")
        for pair, seg_value, pose_value in zip(pair_ids, seg_values, pose_values, strict=True):
            if pair in seen or pair < 0 or pair >= PAIR_COUNT:
                raise Xr1Error(f"duplicate or invalid scorer pair {pair}")
            seen.add(pair)
            seg[pair] = float(seg_value)
            pose[pair] = float(pose_value)
    if seen != set(range(PAIR_COUNT)):
        raise Xr1Error("scorer receipt stages do not cover exactly pairs 0..599")
    return seg, pose


def stage_preflight() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    facts = [require_file(path, size, digest) for path, size, digest in INPUT_PINS]
    rxc1.validate_preflight_receipt(RXC1_STORE / "PREFLIGHT.json")
    free = shutil.disk_usage(STORE).free
    if free - PROJECTED_OUTPUT_BYTES < MANDATORY_RESERVE_BYTES:
        raise Xr1Error(f"storage preflight failed: {free} - {PROJECTED_OUTPUT_BYTES} < {MANDATORY_RESERVE_BYTES}")
    payload = {
        "schema": "ddm_xr1.preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "source_git_commit": source_commit(Path(__file__)),
        "runner_source": file_fact(Path(__file__)),
        "input_pins": facts,
        "storage": {
            "root": str(STORE),
            "free_bytes": free,
            "projected_output_bytes": PROJECTED_OUTPUT_BYTES,
            "mandatory_reserve_bytes": MANDATORY_RESERVE_BYTES,
            "post_projection_free_bytes": free - PROJECTED_OUTPUT_BYTES,
            "status": "PASS",
        },
        "run_spec": {
            "physical_repeats": PHYSICAL_REPEATS,
            "pair_count": PAIR_COUNT,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "percentile_interval": [0.025, 0.975],
            "quantile_method": "linear",
            "canonical_argv": [
                ".venv/bin/python",
                "-u",
                "experiments/ddm_xr1_exchange_ratio_noise_floor.py",
                "--stage",
                "all",
            ],
        },
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "metal_runs": 0,
            "contest_evaluations": 0,
            "upstream_writes": 0,
        },
    }
    receipt_path = STORE / "PREFLIGHT.json"
    if receipt_path.is_file():
        retained = json.loads(receipt_path.read_text())
        stable_keys = (
            "schema",
            "axis",
            "score_claim",
            "source_git_commit",
            "runner_source",
            "input_pins",
            "run_spec",
            "authority_boundaries",
        )
        if any(retained.get(key) != payload.get(key) for key in stable_keys):
            raise Xr1Error(f"immutable preflight custody drifted: {receipt_path}")
        return retained
    atomic_json(receipt_path, payload)
    return payload


def stage_physical_repeats() -> dict[str, Any]:
    stage_preflight()
    api = rxc1.RestartableExactCoder(store=RXC1_STORE)
    runs: list[dict[str, Any]] = []
    for repeat in range(PHYSICAL_REPEATS):
        run = api.run(
            edit_path=NULL_OVERLAY,
            run_dir=STORE / f"retained/physical_null/repeat_{repeat}",
            resume_frame=None,
        )
        if int(run["edit"]["tokens_changed"]) != 0:
            raise Xr1Error(f"null repeat {repeat} consumed a non-null edit")
        runs.append(run)
    reference_stream = Path(str(runs[0]["stream"]["path"]))
    reference_archive = Path(str(runs[0]["archive"]["path"]))
    rows = []
    for repeat, run in enumerate(runs):
        stream = Path(str(run["stream"]["path"]))
        archive = Path(str(run["archive"]["path"]))
        rows.append(
            {
                "repeat": repeat,
                "stream": file_fact(stream),
                "archive": file_fact(archive),
                "stream_vs_repeat_0": rxc1.compare_bytes(stream, reference_stream),
                "archive_vs_repeat_0": rxc1.compare_bytes(archive, reference_archive),
                "bits_per_frame_ledger": run["bits_per_frame_ledger"],
                "terminal_checkpoint": run["terminal_checkpoint"],
                "wall_seconds": float(run["wall_seconds"]),
            }
        )
    byte_counts = np.asarray([int(row["stream"]["bytes"]) for row in rows], dtype=np.int64)
    sample_sigma = float(np.std(byte_counts.astype(np.float64), ddof=1))
    spread = int(byte_counts.max() - byte_counts.min())
    if any(not row["stream_vs_repeat_0"]["byte_identical"] for row in rows):
        raise Xr1Error("physical null stream repeats are not byte-identical")
    payload = {
        "schema": "ddm_xr1.physical_byte_noise.v1",
        "axis": "[macOS-CPU advisory / scorer-free exact RC64 byte replay]",
        "object": "same null field, shipped model, shipped causal schedule",
        "repeat_unit": "one complete physical n600 RC64 re-encode",
        "repeat_count": PHYSICAL_REPEATS,
        "stream_byte_counts": byte_counts.tolist(),
        "sigma_b_sample_bytes": sample_sigma,
        "spread_max_minus_min_bytes": spread,
        "all_streams_byte_identical": True,
        "rows": rows,
    }
    atomic_json(STORE / "PHYSICAL_REPEATS.json", payload)
    return payload


def stage_bootstrap() -> dict[str, Any]:
    stage_preflight()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0,
        PAIR_COUNT,
        size=(BOOTSTRAP_RESAMPLES, PAIR_COUNT),
        dtype=np.uint16,
    )
    atomic_npy(STORE / "retained/bootstrap_draw_indices.npy", draws)

    jbp1_base = np.load(JBP1_BASE_LEDGER, allow_pickle=False)
    jbp1_candidate = np.load(JBP1_CANDIDATE_LEDGER, allow_pickle=False)
    jbp1_base_result = json.loads(JBP1_BASE_RESULT.read_text())
    jbp1_candidate_result = json.loads(JBP1_CANDIDATE_RESULT.read_text())
    if int(jbp1_base_result["stream_delta_bytes"]) != 0:
        raise Xr1Error("JBP1 retained null result is not the exact base stream")
    if int(jbp1_candidate_result["stream_delta_bytes"]) != -2_950:
        raise Xr1Error("JBP1 retained candidate no longer establishes -2,950 B")
    if int(jbp1_candidate_result["edit"]["tokens_changed"]) != 5_506:
        raise Xr1Error("JBP1 retained candidate edit count drifted")
    if len(jbp1_candidate_result["edit"]["edited_pairs"]) != 567:
        raise Xr1Error("JBP1 retained candidate edited-pair count drifted")
    jbp1_pair_bytes, jbp1_samples, jbp1_fixed = exact_total_calibrated_bootstrap(
        jbp1_base,
        jbp1_candidate,
        draws,
        exact_delta_bytes=-2_950,
    )
    atomic_npz(
        STORE / "retained/jbp1_row_a_bootstrap.npz",
        pair_delta_ideal_bytes=jbp1_pair_bytes,
        bootstrap_delta_bytes=jbp1_samples,
    )
    jbp1_interval = percentile_interval(jbp1_samples)

    fcd3_base_bits = np.load(FCD3_BASE_LEDGER, allow_pickle=False)
    fcd3_candidate_bits = np.load(FCD3_CANDIDATE_LEDGER, allow_pickle=False)
    # The body is -2,965 B; the published pose carrier adds a fixed 25 B.
    fcd3_pair_body_bytes, fcd3_body_samples, fcd3_body_fixed = exact_total_calibrated_bootstrap(
        fcd3_base_bits,
        fcd3_candidate_bits,
        draws,
        exact_delta_bytes=-2_965,
    )
    fcd3_byte_samples = fcd3_body_samples + 25.0

    base_receipt = json.loads(FCD3_BASE_SCORER.read_text())
    candidate_receipt = json.loads(FCD3_CANDIDATE_SCORER.read_text())
    base_archive_bytes = int(file_fact(FCD3_BASE_ARCHIVE)["bytes"])
    body_archive_bytes = int(file_fact(FCD3_BODY_ARCHIVE)["bytes"])
    published_archive_bytes = int(file_fact(FCD3_PUBLISHED_ARCHIVE)["bytes"])
    if body_archive_bytes - base_archive_bytes != -2_965:
        raise Xr1Error("FCD3 retained body archive no longer establishes -2,965 B")
    if published_archive_bytes - base_archive_bytes != -2_940:
        raise Xr1Error("FCD3 published archive no longer establishes -2,940 B")
    base_seg, base_pose = scorer_pair_vectors(base_receipt)
    candidate_seg, candidate_pose = scorer_pair_vectors(candidate_receipt)
    base_seg_boot, base_seg_fixed = exact_mean_calibrated_bootstrap(
        base_seg,
        draws,
        exact_mean=float(base_receipt["distortion"]["d_seg"]),
    )
    candidate_seg_boot, candidate_seg_fixed = exact_mean_calibrated_bootstrap(
        candidate_seg,
        draws,
        exact_mean=float(candidate_receipt["distortion"]["d_seg"]),
    )
    base_pose_boot, base_pose_fixed = exact_mean_calibrated_bootstrap(
        base_pose,
        draws,
        exact_mean=float(base_receipt["distortion"]["d_pose"]),
    )
    candidate_pose_boot, candidate_pose_fixed = exact_mean_calibrated_bootstrap(
        candidate_pose,
        draws,
        exact_mean=float(candidate_receipt["distortion"]["d_pose"]),
    )
    delta_d_seg = candidate_seg_boot - base_seg_boot
    delta_d_pose = candidate_pose_boot - base_pose_boot
    delta_s_seg = 100.0 * delta_d_seg
    delta_s_pose = np.sqrt(10.0 * candidate_pose_boot) - np.sqrt(10.0 * base_pose_boot)
    delta_s_distortion = delta_s_seg + delta_s_pose
    delta_s_rate = RATE_NUMERATOR * fcd3_byte_samples / RATE_DENOMINATOR_BYTES
    delta_s = delta_s_distortion + delta_s_rate
    if np.any(np.isclose(delta_s_distortion, 0.0, atol=1e-15)):
        raise Xr1Error("FCD3 bootstrap produced a zero exchange-ratio denominator")
    exchange_ratio = delta_s_rate / delta_s_distortion

    atomic_npz(
        STORE / "retained/fcd3_bootstrap.npz",
        pair_delta_body_ideal_bytes=fcd3_pair_body_bytes,
        base_d_seg_per_pair=base_seg,
        candidate_d_seg_per_pair=candidate_seg,
        base_d_pose_per_pair=base_pose,
        candidate_d_pose_per_pair=candidate_pose,
        bootstrap_delta_bytes=fcd3_byte_samples,
        bootstrap_delta_d_seg=delta_d_seg,
        bootstrap_delta_d_pose=delta_d_pose,
        bootstrap_delta_s_seg=delta_s_seg,
        bootstrap_delta_s_pose=delta_s_pose,
        bootstrap_delta_s_distortion=delta_s_distortion,
        bootstrap_delta_s_rate=delta_s_rate,
        bootstrap_delta_s=delta_s,
        bootstrap_exchange_ratio=exchange_ratio,
    )

    point = score_delta(
        base_d_seg=float(base_receipt["distortion"]["d_seg"]),
        candidate_d_seg=float(candidate_receipt["distortion"]["d_seg"]),
        base_d_pose=float(base_receipt["distortion"]["d_pose"]),
        candidate_d_pose=float(candidate_receipt["distortion"]["d_pose"]),
        delta_bytes=-2_940,
    )
    if not math.isclose(point["delta_s"], 0.0019433243907622244, abs_tol=1e-15):
        raise Xr1Error(f"FCD3 point delta drifted: {point['delta_s']}")
    fcd3_interval = percentile_interval(delta_s)
    point_radius = max(
        abs(point["delta_s"] - fcd3_interval["low"]),
        abs(fcd3_interval["high"] - point["delta_s"]),
    )
    payload = {
        "schema": "ddm_xr1.pair_bootstrap.v1",
        "axis": AXIS,
        "selection": "full n600 population; pair resampling with replacement",
        "statistical_boundary": (
            "The charter-mandated pair bootstrap treats the 600 retained pair "
            "contributions as exchangeable. It estimates sampling uncertainty for this "
            "fixed object; it does not synthesize or physically re-encode a new causal "
            "RC64 sequence, and its interval cannot transfer to another edit set."
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "pairs_per_resample": PAIR_COUNT,
            "interval": "percentile 95%",
            "quantile_method": "linear",
            "draw_indices": file_fact(STORE / "retained/bootstrap_draw_indices.npy"),
        },
        "jbp1_row_a": {
            "object": "5,506-edit XOV1 B/H/W support across 567 pairs",
            "exact_delta_bytes": -2_950,
            "fixed_rounding_adjustment_bytes": jbp1_fixed,
            "delta_bytes_interval_95": jbp1_interval,
            "prediction_narrower_than_plus_minus_600_bytes": bool(jbp1_interval["width"] < 1_200.0),
            "payload": file_fact(STORE / "retained/jbp1_row_a_bootstrap.npz"),
            "distortion": "NOT MEASURED; no exchange ratio or score interval claimed",
        },
        "fcd3": {
            "object": "published tau_1e-6 4,194-edit field plus 448-pair pose carrier",
            "exact_delta_bytes": -2_940,
            "body_exact_delta_bytes": -2_965,
            "fixed_pose_carrier_bytes": 25,
            "body_fixed_rounding_adjustment_bytes": fcd3_body_fixed,
            "aggregate_rounding_adjustments": {
                "base_d_seg": base_seg_fixed,
                "candidate_d_seg": candidate_seg_fixed,
                "base_d_pose": base_pose_fixed,
                "candidate_d_pose": candidate_pose_fixed,
            },
            "point": point,
            "delta_bytes_interval_95": percentile_interval(fcd3_byte_samples),
            "delta_d_seg_interval_95": percentile_interval(delta_d_seg),
            "delta_d_pose_interval_95": percentile_interval(delta_d_pose),
            "delta_s_distortion_interval_95": percentile_interval(delta_s_distortion),
            "delta_s_rate_interval_95": percentile_interval(delta_s_rate),
            "delta_s_interval_95": fcd3_interval,
            "exchange_ratio_interval_95": percentile_interval(exchange_ratio),
            "delta_s_95_point_radius": point_radius,
            "acceptance_rule": "ADMISSIBLE iff delta_s_interval_95.high < 0",
            "admissible": bool(fcd3_interval["high"] < 0.0),
            "interval_excludes_zero": bool(fcd3_interval["low"] > 0.0 or fcd3_interval["high"] < 0.0),
            "payload": file_fact(STORE / "retained/fcd3_bootstrap.npz"),
        },
    }
    atomic_json(STORE / "BOOTSTRAP.json", payload)
    return payload


def near_win_top20(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank RN1 scalar candidates by distance of their closest ratio to one."""

    def key(row: Mapping[str, Any]) -> tuple[float, str, int]:
        distance = min(abs(float(value) - 1.0) for value in row["ratios_le_2x"])
        return distance, str(row["source"]), int(row["line"])

    ranked = sorted(rows, key=key)[:20]
    output = []
    for rank, row in enumerate(ranked, start=1):
        source = REPO / str(row["source"])
        if not source.is_file() or sha256_file(source) != str(row["source_sha256"]):
            raise Xr1Error(f"RN1 top-20 source custody drifted: {source}")
        ratios = [float(value) for value in row["ratios_le_2x"]]
        closest = min(ratios, key=lambda value: abs(value - 1.0))
        output.append(
            {
                "rank": rank,
                "source": row["source"],
                "source_sha256": row["source_sha256"],
                "line": int(row["line"]),
                "verbatim_excerpt": row["verbatim_excerpt"],
                "authority_context": row["authority_context"],
                "ratios_le_2x": row["ratios_le_2x"],
                "closest_ratio_to_one": closest,
                "absolute_margin_from_one": abs(closest - 1.0),
                "grade": "UNGRADABLE",
                "reason": (
                    "RN1's scalar candidate row has no foreign keys to matched same-object "
                    "n600 pair_delta_bytes plus base/candidate d_seg and d_pose vectors. "
                    "FCD3's measured interval is object-specific and cannot be transferred."
                ),
            }
        )
    return output


def stage_top20() -> dict[str, Any]:
    stage_preflight()
    rows = [json.loads(line) for line in RN1_NEAR_WINS.read_text().splitlines() if line.strip()]
    if len(rows) != 300:
        raise Xr1Error(f"RN1 near-win denominator drifted: {len(rows)} != 300")
    top20 = near_win_top20(rows)
    payload = {
        "schema": "ddm_xr1.rn1_top20_regrade.v1",
        "population_rows": 300,
        "ranking": "ascending absolute distance of the nearest extracted ratio to 1.0; source path and line break ties",
        "top_k": 20,
        "grade_counts": {"ADMISSIBLE": 0, "NOT_ADMISSIBLE": 0, "UNGRADABLE": 20},
        "transfer_rule": "an interval is valid only for its same physical object and pair population",
        "rows": top20,
    }
    atomic_json(STORE / "RN1_TOP20_REGRADE.json", payload)
    return payload


def stage_manifest() -> dict[str, Any]:
    physical = stage_physical_repeats()
    bootstrap = stage_bootstrap()
    top20 = stage_top20()
    result = {
        "schema": "ddm_xr1.result.v1",
        "axis": AXIS,
        "physical_byte_noise": {
            "repeat_count": physical["repeat_count"],
            "stream_byte_counts": physical["stream_byte_counts"],
            "sigma_b_sample_bytes": physical["sigma_b_sample_bytes"],
            "spread_max_minus_min_bytes": physical["spread_max_minus_min_bytes"],
            "all_streams_byte_identical": physical["all_streams_byte_identical"],
        },
        "jbp1_row_a": bootstrap["jbp1_row_a"],
        "fcd3": bootstrap["fcd3"],
        "rn1_top20_grade_counts": top20["grade_counts"],
        "acceptance_rule": "near win admissible iff same-object pair-bootstrap 95% upper delta_s < 0",
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "metal_runs": 0,
            "contest_evaluations": 0,
            "pointer_moved": False,
        },
    }
    atomic_json(STORE / "RESULT.json", result)
    paths = sorted(
        path
        for path in STORE.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json" and "launcher" not in path.relative_to(STORE).parts
    )
    facts = [file_fact(path) for path in paths]
    payload = {
        "schema": "ddm_xr1.manifest.v1",
        "root": str(STORE),
        "artifact_count": len(facts),
        "artifact_bytes": sum(int(fact["bytes"]) for fact in facts),
        "artifacts": facts,
        "physical": file_fact(STORE / "PHYSICAL_REPEATS.json"),
        "bootstrap": file_fact(STORE / "BOOTSTRAP.json"),
        "top20": file_fact(STORE / "RN1_TOP20_REGRADE.json"),
        "result": file_fact(STORE / "RESULT.json"),
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "metal_runs": 0,
            "contest_evaluations": 0,
            "pointer_moved": False,
        },
    }
    atomic_json(STORE / "MANIFEST.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=("preflight", "physical", "bootstrap", "top20", "all"),
        default="all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        result = stage_preflight()
    elif args.stage == "physical":
        result = stage_physical_repeats()
    elif args.stage == "bootstrap":
        result = stage_bootstrap()
    elif args.stage == "top20":
        result = stage_top20()
    else:
        result = stage_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
