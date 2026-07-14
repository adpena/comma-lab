# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_throughput_frontier_math",
    ROOT / "tools/probe_throughput_frontier_math.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

FULL_R_SPEC = importlib.util.spec_from_file_location(
    "probe_pythagorean_exact_arithmetic_bitident",
    ROOT / "tools/probe_pythagorean_exact_arithmetic_bitident.py",
)
assert FULL_R_SPEC is not None and FULL_R_SPEC.loader is not None
FULL_R_PRODUCER = importlib.util.module_from_spec(FULL_R_SPEC)
FULL_R_SPEC.loader.exec_module(FULL_R_PRODUCER)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_json_source_uses_one_read_snapshot_without_is_file_probe() -> None:
    class ReadOnlySnapshot:
        def __str__(self) -> str:
            return "atomic-source.json"

        def read_bytes(self) -> bytes:
            return b'{"schema":"snapshot.v1"}'

        def is_file(self) -> bool:
            raise AssertionError("is_file TOCTOU probe must not be called")

    descriptor, payload = PROBE._read_json_source(
        ReadOnlySnapshot(),  # type: ignore[arg-type]
        required=False,
    )
    assert descriptor["exists"] is True
    assert descriptor["schema"] == "snapshot.v1"
    assert payload == {"schema": "snapshot.v1"}


def _valid_fixedpoint(
    *,
    admitted: str | None = "w8a8",
    manifest: dict | None = None,
    schema: str = "fixedpoint_scorer_forward_n600.v1",
) -> dict:
    names = ("fp32_control", "w8a8")
    arms: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for name in names:
        is_exact = name == "fp32_control" or admitted == name
        rows = [
            {
                "pair_index": pair_index,
                "split": "calibration" if pair_index < 120 else "heldout",
                "flips": 0 if is_exact else 1,
                "candidate_argmax_sha256": _digest(
                    f"fp32_control:{pair_index}" if is_exact else f"{name}:{pair_index}"
                ),
            }
            for pair_index in range(600)
        ]
        corpus_digest = PROBE._argmax_rows_digest(rows)
        heldout_digest = PROBE._argmax_rows_digest(rows[120:])
        arms[name] = {"segnet_rows": rows, "posenet_rows": []}
        summaries[name] = {
            "status": "MEASURED",
            "argmax_exact_admitted": is_exact,
            "segnet": {
                "full": {
                    "pairs": 600,
                    "flips": 0 if is_exact else 600,
                    "argmax_exact_gate": is_exact,
                    "argmax_corpus_sha256": corpus_digest,
                },
                "heldout": {
                    "pairs": 480,
                    "flips": 0 if is_exact else 480,
                    "argmax_exact_gate": is_exact,
                    "argmax_corpus_sha256": heldout_digest,
                },
            },
        }
    receipt = {
        "schema": schema,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "completed": True,
        "contract": {
            "pair_start": 0,
            "pair_count": 600,
            "native_integer_speed_claim": False,
            "accumulation": "QDQ emulation with fp32 Conv2d/Linear accumulation",
            "calibration_split": [0, 120],
            "heldout_split": [120, 600],
            "threads": {"interop": 1, "intraop": 1},
            "include_pose": False,
            "arms": [
                {"name": "fp32_control", "bits": 32, "mixed_head_fp32": False},
                {"name": "w8a8", "bits": 8, "mixed_head_fp32": False},
            ],
        },
        "custody": {
            "gt_cache_sha256": PROBE.EXPECTED_GT_CACHE_SHA256,
            "segnet_weights_sha256": PROBE.EXPECTED_SEGNET_WEIGHTS_SHA256,
            "probe_sha256": PROBE._sha256_file(PROBE.FIXEDPOINT_PRODUCER_SOURCES["probe_sha256"]),
            "module_sha256": PROBE._sha256_file(PROBE.FIXEDPOINT_PRODUCER_SOURCES["module_sha256"]),
            "calibration_digest": "3" * 64,
        },
        "fingerprint": "",
        "arms": arms,
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "arms": summaries,
            "minimum_argmax_exact_arm": admitted,
            "rung2_verdict": (
                "ARGMAX_FIXEDPOINT_FEASIBLE" if admitted is not None else "NO_ADMITTED_PRECISION_IN_LADDER"
            ),
        },
    }
    if schema == "fixedpoint_scorer_forward_n600.v2":
        receipt["contract"].update(
            {
                "activation_scale_mode": "fixed_calibration",
                "dynamic_scale_order_invariance": None,
            }
        )
    if manifest is not None:
        receipt["frontier_math_precision_manifest"] = manifest
    fingerprint_contract = {
        **receipt["contract"],
        "arms": [spec for spec in receipt["contract"]["arms"] if spec["name"] != "fp32_control"],
    }
    receipt["fingerprint"] = PROBE._canonical_sha256({"contract": fingerprint_contract, "custody": receipt["custody"]})
    return receipt


def _fixture_full_r_summary(receipt: dict) -> dict:
    authority = receipt["numpy_authority"]["summary"]
    variants = {}
    blocked = False
    for variant in PROBE.FULL_R_VARIANTS:
        rows = receipt["trials"][variant]
        measured = [row for row in rows if row["status"] == "MEASURED"]
        blockers = [row for row in rows if row["status"] == "BLOCKED_NOT_MEASURED"]
        hashes = [row["corpus_sha256"] for row in measured]
        entry = {
            "attempts": len(rows),
            "measured_processes": len(measured),
            "expected_processes": 10,
            "all_full_coverage": bool(measured and all(row["frames"] == 1200 for row in measured)),
            "unique_corpus_hashes": len(set(hashes)),
            "cross_process_identical": bool(hashes and len(set(hashes)) == 1),
            "hashes": hashes,
            "blockers": blockers,
        }
        if variant == "fixed_q15_int32_atomic":
            entry["exact_numpy_int_corpus_parity"] = bool(
                hashes and all(value == authority["integer_corpus_sha256"] for value in hashes)
            )
        variants[variant] = entry
        blocked = blocked or bool(blockers)
    complete = bool(
        authority["status"] == "MEASURED"
        and authority["coverage_exact"]
        and all(variants[variant]["measured_processes"] == 10 for variant in variants)
    )
    float_diverges = bool(complete and variants["float_atomic"]["unique_corpus_hashes"] > 1)
    integer_holds = bool(
        complete
        and variants["fixed_q15_int32_atomic"]["cross_process_identical"]
        and variants["fixed_q15_int32_atomic"]["exact_numpy_int_corpus_parity"]
        and authority["within_derived_bound"]
    )
    if blocked:
        verdict = "BLOCKED_NOT_MEASURED"
        scope = "ENVIRONMENT: no evaluated Metal device in attempted child process"
    elif not complete:
        verdict = "INCOMPLETE"
        scope = "INSTANCE: full-R real-n600 receipt coverage/process count"
    elif float_diverges and integer_holds:
        verdict = "REAL-L70-LEVER-FULL-R-N600"
        scope = (
            "n600 INSTANCE: real 0.mkv gt_f0+gt_f1, four-axis render-R VJP, "
            "Q15/int32 atomic with Q7/Q5 state schedule, this MLX/Metal host"
        )
    elif not integer_holds:
        verdict = "FULL-R-INTEGER-FORMULATION-NO-GO"
        scope = "FORMULATION: Q15/int32 atomic plus Q7/Q5 boundary schedule; not the integer family"
    else:
        verdict = "FLOAT-WALL-NOT-REPRODUCED-FULL-R"
        scope = "INSTANCE: this full-R real-n600 corpus/host/process set"
    return {
        **variants,
        "authority": authority,
        "complete": complete,
        "decisive_positive": bool(float_diverges and integer_holds),
        "overall_verdict": verdict,
        "verdict_scope": scope,
    }


def _valid_full_r(*, complete: bool = True) -> dict:
    contract = {
        "scope": "full-r-n600",
        "gt_cache": "/fixture/gt_n600.npz",
        "pair_start": 0,
        "pair_count": 600,
        "frames": 1200,
        "members": list(PROBE.FULL_R_MEMBERS),
        "gt_cache_sha256": PROBE.EXPECTED_GT_CACHE_SHA256,
        "n_processes_per_variant": 10,
        "q_weight_bits": 15,
        "state_bits_by_boundary": [7, 7, 7, 5, 5],
        "signed_requantization": ("nearest; exact half away from zero; integer division; no signed shift"),
        "cotangent": ("bilinear-down(real uint8 0.mkv frame,874x1164->384x512); clip(rint(value-127.5),-127,127)"),
        "chain": [
            "down_w_transpose_512_to_1164",
            "down_h_transpose_384_to_874",
            "up_w_transpose_1164_to_512",
            "up_h_transpose_874_to_384",
        ],
    }
    static_proof = FULL_R_PRODUCER.full_r_static_overflow_proof()
    error_bound = FULL_R_PRODUCER.derive_full_r_integer_error_bound()
    receipt = {
        "schema": "pythagorean_exact_arithmetic_full_r_n600.v2",
        "lane_id": "throughput_authority_ladder",
        "task_id": 494,
        "axis": ("[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]"),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "training": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
        "contract_fingerprint": PROBE._canonical_sha256(contract),
        "contract": contract,
        "source_custody": {
            "probe": {
                "path": "tools/probe_pythagorean_exact_arithmetic_bitident.py",
                "sha256": PROBE._sha256_file(PROBE.FULL_R_PRODUCER_SOURCES["probe"]),
            },
            "fused_r_reference": {
                "path": "src/tac/local_acceleration/metal_fused_r_operator.py",
                "sha256": PROBE._sha256_file(PROBE.FULL_R_PRODUCER_SOURCES["fused_r_reference"]),
            },
            "gt_cache": {
                "path": contract["gt_cache"],
                "sha256": PROBE.EXPECTED_GT_CACHE_SHA256,
            },
        },
        "static_integer_proof": static_proof,
        "derived_integer_error_bound": error_bound,
        "numpy_authority": {"rows": []},
        "trials": {variant: [] for variant in PROBE.FULL_R_VARIANTS},
    }
    if not complete:
        return receipt

    rows = []
    float_digest = hashlib.sha256()
    integer_digest = hashlib.sha256()
    for pair_index in range(600):
        for member in PROBE.FULL_R_MEMBERS:
            float_output = _digest(f"float:{pair_index}:{member}")
            integer_output = _digest(f"integer:{pair_index}:{member}")
            float_digest.update(f"{pair_index}:{member}:{float_output}\n".encode("ascii"))
            integer_digest.update(f"{pair_index}:{member}:{integer_output}\n".encode("ascii"))
            rows.append(
                {
                    "pair_index": pair_index,
                    "member": member,
                    "input_frame_sha256": _digest(f"input:{pair_index}:{member}"),
                    "cotangent_sha256": _digest(f"cot:{pair_index}:{member}"),
                    "clip_mask_sha256": _digest(f"clip:{pair_index}:{member}"),
                    "float_output_sha256": float_output,
                    "integer_output_sha256": integer_output,
                    "max_abs_error": 0.1,
                    "sum_squared_error": 0.01,
                    "error_elements": 1,
                    "stage_actual_max_sum_abs_contributions": [1, 1, 1, 1],
                    "stage_actual_minimum_signed_accumulator_bits": [2, 2, 2, 2],
                }
            )
    float_corpus = float_digest.hexdigest()
    integer_corpus = integer_digest.hexdigest()
    authority = {
        "status": "MEASURED",
        "coverage_exact": True,
        "frames": 1200,
        "expected_frames": 1200,
        "float_corpus_sha256": float_corpus,
        "integer_corpus_sha256": integer_corpus,
        "dequantized_max_abs_error_vs_numpy_fp32": 0.1,
        "dequantized_rmse_vs_numpy_fp32": math.sqrt(
            sum(float(row["sum_squared_error"]) for row in rows) / sum(int(row["error_elements"]) for row in rows)
        ),
        "derived_max_abs_error_bound": error_bound["final_max_abs_error_bound"],
        "within_derived_bound": True,
    }
    receipt["numpy_authority"] = {"rows": rows, "summary": authority}
    float_hashes = [_digest(f"float-trial:{index}") for index in range(10)]
    fixed_hashes = [integer_corpus] * 10
    for variant, hashes in zip(PROBE.FULL_R_VARIANTS, (float_hashes, fixed_hashes), strict=True):
        receipt["trials"][variant] = [
            {
                "variant": variant,
                "status": "MEASURED",
                "corpus_sha256": corpus,
                "frames": 1200,
                "pairs": 600,
                "pair_start": 0,
                "members": list(PROBE.FULL_R_MEMBERS),
                "trial_index": trial_index,
                "device": "Device(gpu, 0)",
                "mlx_version": "fixture-version",
                "elapsed_seconds": 1.0 + trial_index,
                "static_overflow_proofs": (
                    json.loads(json.dumps(static_proof)) if variant == "fixed_q15_int32_atomic" else None
                ),
            }
            for trial_index, corpus in enumerate(hashes)
        ]
    receipt["summary"] = _fixture_full_r_summary(receipt)
    receipt["completed"] = True
    return receipt


def _valid_full_r_in_progress(*, float_trials: int = 3) -> dict:
    receipt = _valid_full_r()
    receipt["trials"]["float_atomic"] = receipt["trials"]["float_atomic"][:float_trials]
    receipt["trials"]["fixed_q15_int32_atomic"] = []
    receipt["summary"] = _fixture_full_r_summary(receipt)
    receipt["completed"] = False
    return receipt


def _partial_fixedpoint(*, pairs: int = 100) -> dict:
    receipt = _valid_fixedpoint()
    receipt["completed"] = False
    summaries = {}
    for name, state in receipt["arms"].items():
        state["segnet_rows"] = state["segnet_rows"][:pairs]
        summaries[name] = {
            "status": "INCOMPLETE",
            "pairs": pairs,
            "unique_pair_indices": pairs,
        }
    receipt["summary"] = {
        "status": "INCOMPLETE",
        "full_real_n600": False,
        "arms": summaries,
        "minimum_argmax_exact_arm": None,
        "rung2_verdict": "INCOMPLETE",
    }
    return receipt


def _arguments(
    tmp_path: Path,
    *,
    fixedpoint: dict | None = None,
    full_r: dict | None = None,
) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    verdict_wallclock = _write(
        tmp_path / "verdict_wallclock.json",
        {
            "schema": "frozen_scorer_verdict_wallclock.v1",
            "axis": "[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE",
            "score_claim": False,
            "promotable": False,
            "means_only": True,
            "num_pairs": 96,
            "torch_threads": 1,
            "combined_verdict_s_total": 59.615,
            "per_pair_verdict_s": 0.621,
            "seg_fraction_of_verdict": 0.774,
            "pose_fraction_of_verdict": 0.226,
            "extrapolated_n600_verdict_s": 372.6,
            "extrapolated_n600_verdict_min": 6.21,
        },
    )
    pythagorean = _write(
        tmp_path / "pythagorean.json",
        {
            "schema": "pythagorean_exact_arithmetic_bitident_probe.v1",
            "numpy_static_contract": {"max_abs_integer_accumulator": 11_159_918},
            "summary": {
                "complete": True,
                "overall_verdict": "REAL-L70-LEVER",
                "verdict_scope": "INSTANCE fixture",
            },
        },
    )
    tile = _write(
        tmp_path / "tile.json",
        {
            "schema": "cheapen_real95_tile_halo_exactness.v1",
            "exact_tile_contract": {
                "exact_dependency": "FULL_FRAME_GLOBAL",
                "exact_source_area_fraction": 1.0,
                "ideal_exact_speedup_upper_bound": 1.0,
                "squeeze_excite_blocks": 23,
                "local_halo_px": 685,
                "local_receptive_field_px": 1311,
                "verdict": "NO_GO",
                "verdict_scope": "INSTANCE frozen teacher",
            },
            "n600_real_coverage": {
                "n_pairs": 600,
                "boundary_area_fraction": 0.047365976969,
                "boundary_flip_mass_share": 0.268038228210,
                "pair_ids_sha256": "7" * 64,
            },
        },
    )
    sparse = _write(
        tmp_path / "sparse.json",
        {
            "schema": "p0_sparse_adjoint_costate_vjp.v1",
            "structural_exactness": {"exact_sparse_backward_speedup_x": 1.0},
        },
    )
    fixedpoint_path = tmp_path / "fixedpoint.json"
    if fixedpoint is not None:
        _write(fixedpoint_path, fixedpoint)
    full_r_path = tmp_path / "full_r.json"
    if full_r is not None:
        _write(full_r_path, full_r)
    return argparse.Namespace(
        verdict_wallclock_receipt=verdict_wallclock,
        pythagorean_receipt=pythagorean,
        tile_halo_receipt=tile,
        sparse_adjoint_receipt=sparse,
        fixedpoint_receipt=fixedpoint_path,
        full_r_receipt=full_r_path,
        n_pairs=600,
        output_dir=tmp_path / "output",
        resume=False,
    )


def test_missing_host_receipts_preserve_static_stages_without_completion(
    tmp_path: Path,
) -> None:
    receipt = PROBE._build_receipt(_arguments(tmp_path))
    assert receipt["overall_status"] == "OWED_FIXEDPOINT_FORWARD_RECEIPT"
    assert receipt["integer_gpu_ane_backend_authority_complete"] is False
    assert receipt["full_r_training_repro_status"] == "OWED_FULL_R_TRAINING_REPRO_RECEIPT"
    assert receipt["score_claim"] is False
    assert receipt["pointer_moved"] is False
    assert receipt["measured_authority_verdict_n_pairs"] == 96
    assert receipt["derived_linear_extrapolation_n600_minutes"] == 6.21
    assert tuple(receipt["stage_files"]) == PROBE.STAGE_FILENAMES

    exact = json.loads((tmp_path / "output/stage_01_exact_number_system.json").read_text())["payload"]
    assert exact["fixed_width_certificate"]["minimum_signed_bits"] == 25
    assert exact["fixed_width_certificate"]["no_overflow"] is True
    assert exact["crt_certificate"]["symmetric_reconstruction_injective"] is True

    argmax = json.loads((tmp_path / "output/stage_02_argmax_certificate.json").read_text())["payload"]
    assert argmax["tropical_tie_fixture"]["canonical_tie_winner_is_smallest_class_index"]
    assert argmax["tropical_tie_fixture"]["permutations_checked"] == 24


def test_valid_qdq_receipt_does_not_claim_integer_backend_or_require_full_r(
    tmp_path: Path,
) -> None:
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(admitted="w8a8")))
    assert receipt["overall_status"] == ("FORWARD_QDQ_FEASIBILITY_ADMITTED__INTEGER_BACKEND_OWED")
    assert receipt["integer_gpu_ane_backend_authority_complete"] is False
    assert receipt["full_r_training_repro_status"] == "OWED_FULL_R_TRAINING_REPRO_RECEIPT"
    assert receipt["fixedpoint_forward_validation"]["admitted_arm"] == "w8a8"


def test_empty_or_incomplete_receipt_is_blocked_not_complete(tmp_path: Path) -> None:
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint={}))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert receipt["fixedpoint_forward_validation"]["valid_full_n600"] is False
    assert receipt["integer_gpu_ane_backend_authority_complete"] is False


def test_qdq_contract_must_be_explicit(tmp_path: Path) -> None:
    fixedpoint = _valid_fixedpoint()
    del fixedpoint["contract"]["accumulation"]
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert "contract_full_n600_qdq_only" in receipt["fixedpoint_forward_validation"]["failures"]


def test_fixedpoint_v2_fixed_calibration_receipt_is_supported(tmp_path: Path) -> None:
    receipt = PROBE._build_receipt(
        _arguments(
            tmp_path,
            fixedpoint=_valid_fixedpoint(schema="fixedpoint_scorer_forward_n600.v2"),
        )
    )
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")


@pytest.mark.parametrize(
    ("mode", "dynamic_invariance"),
    [
        ("dynamic_exact_absmax", "max(abs(x)) is order invariant"),
        ("fixed_calibration", "unexpected dynamic claim"),
    ],
)
def test_fixedpoint_v2_refuses_nonfixed_or_dynamic_scale_claims(
    tmp_path: Path, mode: str, dynamic_invariance: str
) -> None:
    fixedpoint = _valid_fixedpoint(schema="fixedpoint_scorer_forward_n600.v2")
    fixedpoint["contract"]["activation_scale_mode"] = mode
    fixedpoint["contract"]["dynamic_scale_order_invariance"] = dynamic_invariance
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert "contract_v2_fixed_calibration_only" in receipt["fixedpoint_forward_validation"]["failures"]


def test_well_formed_in_progress_receipt_is_owed_not_invalid(tmp_path: Path) -> None:
    fixedpoint = _valid_fixedpoint()
    fixedpoint["completed"] = False
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "OWED_FIXEDPOINT_FORWARD_COMPLETION"


def test_well_formed_partial_prefix_is_owed_not_invalid(tmp_path: Path) -> None:
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_partial_fixedpoint(pairs=100)))
    assert receipt["overall_status"] == "OWED_FIXEDPOINT_FORWARD_COMPLETION"


@pytest.mark.parametrize("corruption", ["unequal_lengths", "duplicate_index", "bad_hash"])
def test_corrupt_partial_prefix_is_invalid(tmp_path: Path, corruption: str) -> None:
    fixedpoint = _partial_fixedpoint(pairs=100)
    if corruption == "unequal_lengths":
        fixedpoint["arms"]["w8a8"]["segnet_rows"] = fixedpoint["arms"]["w8a8"]["segnet_rows"][:50]
        fixedpoint["summary"]["arms"]["w8a8"].update({"pairs": 50, "unique_pair_indices": 50})
    elif corruption == "duplicate_index":
        fixedpoint["arms"]["w8a8"]["segnet_rows"][20]["pair_index"] = 19
    else:
        fixedpoint["arms"]["w8a8"]["segnet_rows"][20]["candidate_argmax_sha256"] = "not-a-hash"
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"


def test_complete_checkpoint_with_corrupt_summary_is_invalid(tmp_path: Path) -> None:
    fixedpoint = _valid_fixedpoint()
    fixedpoint["completed"] = False
    fixedpoint["summary"]["arms"]["w8a8"]["segnet"]["full"]["argmax_corpus_sha256"] = "f" * 64
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert "w8a8.segnet_summary_custody" in receipt["fixedpoint_forward_validation"]["failures"]


def test_in_progress_receipt_with_immutable_custody_drift_is_invalid(tmp_path: Path) -> None:
    fixedpoint = _valid_fixedpoint()
    fixedpoint["completed"] = False
    fixedpoint["custody"]["probe_sha256"] = "f" * 64
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert "probe_sha256_current_source_match" in receipt["fixedpoint_forward_validation"]["failures"]


def test_valid_no_admitted_precision_is_complete_scoped_negative(tmp_path: Path) -> None:
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(admitted=None)))
    assert receipt["overall_status"] == ("FORWARD_QDQ_FEASIBILITY_NO_ADMITTED_PRECISION__INTEGER_BACKEND_OWED")


@pytest.mark.parametrize("declared", ["fp32_control", None])
def test_fixedpoint_admission_is_recomputed_from_quantized_arms(tmp_path: Path, declared: str | None) -> None:
    fixedpoint = _valid_fixedpoint(admitted="w8a8")
    if declared == "fp32_control":
        fixedpoint = _valid_fixedpoint(admitted=None)
    fixedpoint["summary"]["minimum_argmax_exact_arm"] = declared
    fixedpoint["summary"]["rung2_verdict"] = (
        "ARGMAX_FIXEDPOINT_FEASIBLE" if declared is not None else "NO_ADMITTED_PRECISION_IN_LADDER"
    )
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"
    assert "minimum_quantized_argmax_arm_consistency" in receipt["fixedpoint_forward_validation"]["failures"]


@pytest.mark.parametrize("field", ["candidate_argmax_sha256", "flips", "split"])
def test_fixedpoint_row_custody_is_rederived_not_trusted(tmp_path: Path, field: str) -> None:
    fixedpoint = _valid_fixedpoint(admitted="w8a8")
    row = fixedpoint["arms"]["w8a8"]["segnet_rows"][120]
    if field == "candidate_argmax_sha256":
        row[field] = "f" * 64
    elif field == "flips":
        row[field] = 1
    else:
        row[field] = "calibration"
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert receipt["overall_status"] == "BLOCKED_INVALID_FIXEDPOINT_FORWARD_RECEIPT"


def test_full_r_is_validated_as_orthogonal_training_repro(tmp_path: Path) -> None:
    receipt = PROBE._build_receipt(
        _arguments(
            tmp_path,
            fixedpoint=_valid_fixedpoint(admitted="w8a8"),
            full_r=_valid_full_r(),
        )
    )
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")
    assert receipt["full_r_training_repro_status"] == "FULL_R_TRAINING_REPRO_MEASURED"


def test_full_r_empty_custody_is_blocked_without_gating_forward(tmp_path: Path) -> None:
    full_r = _valid_full_r()
    full_r["source_custody"] = {}
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")


def test_full_r_base_checkpoint_without_top_summary_is_resumable_owed(
    tmp_path: Path,
) -> None:
    full_r = _valid_full_r(complete=False)
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("OWED_FULL_R_TRAINING_REPRO_COMPLETION")


def test_full_r_summary_present_in_progress_state_is_resumable_owed(
    tmp_path: Path,
) -> None:
    receipt = PROBE._build_receipt(
        _arguments(
            tmp_path,
            fixedpoint=_valid_fixedpoint(),
            full_r=_valid_full_r_in_progress(float_trials=3),
        )
    )
    assert receipt["full_r_training_repro_status"] == ("OWED_FULL_R_TRAINING_REPRO_COMPLETION")
    assert receipt["full_r_training_repro_validation"]["valid_contract"] is True


def test_full_r_summary_absent_numpy_prefix_is_validated(tmp_path: Path) -> None:
    full_r = _valid_full_r()
    rows = full_r["numpy_authority"]["rows"][:7]
    full_r["numpy_authority"] = {
        "rows": rows,
        "summary": PROBE._derive_full_r_authority_summary(
            rows,
            derived_bound=full_r["derived_integer_error_bound"]["final_max_abs_error_bound"],
        ),
    }
    full_r["trials"] = {variant: [] for variant in PROBE.FULL_R_VARIANTS}
    full_r.pop("summary")
    full_r.pop("completed")
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("OWED_FULL_R_TRAINING_REPRO_COMPLETION")


@pytest.mark.parametrize(
    "corruption",
    [
        "bad_numpy_prefix",
        "missing_raw_error",
        "bad_actual_accumulator_bits",
        "bad_trial_prefix",
    ],
)
def test_full_r_corrupt_incomplete_progress_is_blocked(tmp_path: Path, corruption: str) -> None:
    full_r = _valid_full_r_in_progress(float_trials=3)
    if corruption == "bad_numpy_prefix":
        full_r["numpy_authority"]["rows"][10]["pair_index"] = 99
    elif corruption == "missing_raw_error":
        del full_r["numpy_authority"]["rows"][10]["sum_squared_error"]
    elif corruption == "bad_actual_accumulator_bits":
        full_r["numpy_authority"]["rows"][10]["stage_actual_minimum_signed_accumulator_bits"][0] = 1
    else:
        full_r["trials"]["float_atomic"][1]["trial_index"] = 0
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")


@pytest.mark.parametrize("complete_state", [True, False])
def test_full_r_completed_flag_must_equal_rederived_summary(tmp_path: Path, complete_state: bool) -> None:
    full_r = _valid_full_r() if complete_state else _valid_full_r_in_progress()
    full_r["completed"] = not complete_state
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert "completed_summary_consistency" in receipt["full_r_training_repro_validation"]["failures"]


def test_full_r_bound_boolean_is_rederived_from_raw_numeric_rows(
    tmp_path: Path,
) -> None:
    full_r = _valid_full_r()
    row = full_r["numpy_authority"]["rows"][0]
    row["max_abs_error"] = 1.0
    row["sum_squared_error"] = 1.0
    authority = full_r["numpy_authority"]["summary"]
    authority["dequantized_max_abs_error_vs_numpy_fp32"] = 1.0
    authority["dequantized_rmse_vs_numpy_fp32"] = math.sqrt((1.0 + 1199 * 0.01) / 1200)
    authority["within_derived_bound"] = True
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")
    assert "numpy_authority_summary" in receipt["full_r_training_repro_validation"]["failures"]


def test_full_r_trials_require_exact_static_proof_not_safe_boolean(
    tmp_path: Path,
) -> None:
    full_r = _valid_full_r()
    full_r["trials"]["fixed_q15_int32_atomic"][0]["static_overflow_proofs"] = [{"stage": "fixture", "safe": True}]
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert "trial_progress" in receipt["full_r_training_repro_validation"]["failures"]


def test_malformed_full_r_is_blocked_without_gating_forward(tmp_path: Path) -> None:
    args = _arguments(tmp_path, fixedpoint=_valid_fixedpoint())
    args.full_r_receipt.write_text("{not-json", encoding="utf-8")
    receipt = PROBE._build_receipt(args)
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")


def test_full_r_complete_summary_without_raw_measurements_is_invalid(
    tmp_path: Path,
) -> None:
    full_r = _valid_full_r()
    full_r["numpy_authority"] = {"rows": [], "summary": full_r["summary"]["authority"]}
    full_r["trials"] = {variant: [] for variant in PROBE.FULL_R_VARIANTS}
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")


@pytest.mark.parametrize("corruption", ["float_wall_absent", "integer_parity_absent"])
def test_full_r_positive_verdict_is_rederived_from_raw_trials(tmp_path: Path, corruption: str) -> None:
    full_r = _valid_full_r()
    if corruption == "float_wall_absent":
        same_hash = full_r["trials"]["float_atomic"][0]["corpus_sha256"]
        for row in full_r["trials"]["float_atomic"]:
            row["corpus_sha256"] = same_hash
        full_r["summary"]["float_atomic"].update(
            {
                "hashes": [same_hash] * 10,
                "unique_corpus_hashes": 1,
                "cross_process_identical": True,
            }
        )
    else:
        wrong_hash = "e" * 64
        for row in full_r["trials"]["fixed_q15_int32_atomic"]:
            row["corpus_sha256"] = wrong_hash
        full_r["summary"]["fixed_q15_int32_atomic"].update(
            {
                "hashes": [wrong_hash] * 10,
                "unique_corpus_hashes": 1,
                "cross_process_identical": True,
                "exact_numpy_int_corpus_parity": False,
            }
        )
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")
    assert "summary_full_r_rederived_custody" in receipt["full_r_training_repro_validation"]["failures"]


def test_full_r_requires_distinct_process_identity_and_overflow_proofs(
    tmp_path: Path,
) -> None:
    full_r = _valid_full_r()
    del full_r["trials"]["float_atomic"][0]["trial_index"]
    full_r["trials"]["fixed_q15_int32_atomic"][0]["static_overflow_proofs"] = []
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=_valid_fixedpoint(), full_r=full_r))
    assert receipt["full_r_training_repro_status"] == ("BLOCKED_INVALID_FULL_R_TRAINING_REPRO_RECEIPT")


def test_normalized_precision_manifest_uses_exact_discrete_solver(tmp_path: Path) -> None:
    manifest = {
        "schema": "throughput_frontier_math_precision_manifest.v1",
        "bound_kind": "retrospective_n600_observed",
        "error_budget": 0.35,
        "layers": [
            {
                "name": "early",
                "options": [
                    {"bits": 4, "error_bound": 0.4, "measured_cost": 1.0},
                    {"bits": 8, "error_bound": 0.05, "measured_cost": 3.0},
                ],
            },
            {
                "name": "head",
                "options": [
                    {"bits": 4, "error_bound": 0.3, "measured_cost": 1.0},
                    {"bits": 8, "error_bound": 0.02, "measured_cost": 4.0},
                ],
            },
        ],
    }
    receipt = PROBE._build_receipt(
        _arguments(
            tmp_path,
            fixedpoint=_valid_fixedpoint(admitted="w8a8", manifest=manifest),
        )
    )
    assert receipt["overall_status"].startswith("FORWARD_QDQ_FEASIBILITY_ADMITTED")
    waterfill = json.loads((tmp_path / "output/stage_03_discrete_waterfill.json").read_text())["payload"]
    assert waterfill["status"] == "SOLVED_CORPUS_SCOPED"
    assert [choice["bits"] for choice in waterfill["choices"]] == [8, 4]
    assert [choice["layer"] for choice in waterfill["choices"]] == ["early", "head"]
    assert waterfill["unseen_input_certificate"] is False


def test_rigorous_bound_label_alone_is_refused() -> None:
    result = PROBE._normalized_precision_allocation(
        {
            "frontier_math_precision_manifest": {
                "schema": "throughput_frontier_math_precision_manifest.v1",
                "bound_kind": "rigorous_classwise_interval",
            }
        }
    )
    assert result["status"] == "OWED_EXTERNAL_RIGOROUS_BOUND_VALIDATOR"
    assert result["unseen_input_certificate"] is False


def test_precision_manifest_refuses_duplicate_layer_names() -> None:
    option = {"bits": 8, "error_bound": 0.1, "measured_cost": 1.0}
    with pytest.raises(ValueError, match="layer names must be unique"):
        PROBE._normalized_precision_allocation(
            {
                "frontier_math_precision_manifest": {
                    "schema": "throughput_frontier_math_precision_manifest.v1",
                    "bound_kind": "retrospective_n600_observed",
                    "error_budget": 0.2,
                    "layers": [
                        {"name": "head", "options": [option]},
                        {"name": "head", "options": [option]},
                    ],
                }
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("error_budget", True, "error_budget"),
        ("error_bound", False, "error_bound"),
        ("measured_cost", True, "measured_cost"),
        ("mode", 7, "mode"),
    ],
)
def test_precision_manifest_refuses_json_type_coercion(field: str, value: object, match: str) -> None:
    manifest = {
        "schema": "throughput_frontier_math_precision_manifest.v1",
        "bound_kind": "retrospective_n600_observed",
        "error_budget": 0.5,
        "layers": [
            {
                "name": "head",
                "options": [
                    {
                        "bits": 8,
                        "error_bound": 0.1,
                        "measured_cost": 1.0,
                        "mode": "integer",
                    }
                ],
            }
        ],
    }
    if field == "error_budget":
        manifest[field] = value
    else:
        manifest["layers"][0]["options"][0][field] = value
    with pytest.raises(ValueError, match=match):
        PROBE._normalized_precision_allocation({"frontier_math_precision_manifest": manifest})


def test_resume_verifies_every_stage_and_final_receipt(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    PROBE._build_receipt(args)
    args.resume = True
    PROBE._build_receipt(args)

    stage = tmp_path / "output/stage_02_argmax_certificate.json"
    value = json.loads(stage.read_text())
    value["payload"]["strict_tie_policy"] = "corrupt"
    stage.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        PROBE._build_receipt(args)

    # Restore the stage, then prove final-receipt bytes are independently checked.
    stage.unlink()
    PROBE._build_receipt(args)
    final = tmp_path / "output/measurement_receipt.json"
    final_value = json.loads(final.read_text())
    final_value["overall_status"] = "CORRUPT"
    final.write_text(json.dumps(final_value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final-receipt deterministic-payload drift"):
        PROBE._build_receipt(args)


def test_resume_rejects_source_drift_and_unexpected_stage(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    PROBE._build_receipt(args)
    args.resume = True
    args.pythagorean_receipt.write_text(args.pythagorean_receipt.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint drift"):
        PROBE._build_receipt(args)

    (args.output_dir / "stage_99_stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected stale stage"):
        PROBE._build_receipt(args)


def test_fingerprint_tracks_fixedpoint_producer_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    producer = _write(tmp_path / "producer.py", {"version": 1})
    module = _write(tmp_path / "module.py", {"version": 1})
    monkeypatch.setattr(
        PROBE,
        "FIXEDPOINT_PRODUCER_SOURCES",
        {"probe_sha256": producer, "module_sha256": module},
    )
    args = _arguments(tmp_path)
    first = PROBE._build_receipt(args)
    producer.write_text("changed", encoding="utf-8")
    args.output_dir = tmp_path / "output_changed"
    second = PROBE._build_receipt(args)
    assert first["fingerprint"] != second["fingerprint"]


def test_fixedpoint_and_full_r_producer_fingerprints_are_rederived(
    tmp_path: Path,
) -> None:
    fixedpoint = _valid_fixedpoint()
    fixedpoint["fingerprint"] = "4" * 64
    receipt = PROBE._build_receipt(_arguments(tmp_path, fixedpoint=fixedpoint))
    assert "producer_contract_fingerprint" in receipt["fixedpoint_forward_validation"]["failures"]

    full_r = _valid_full_r()
    full_r["contract_fingerprint"] = "5" * 64
    receipt = PROBE._build_receipt(
        _arguments(
            tmp_path / "full_r_case",
            fixedpoint=_valid_fixedpoint(),
            full_r=full_r,
        )
    )
    assert "producer_contract_fingerprint" in receipt["full_r_training_repro_validation"]["failures"]


def test_resume_refuses_corrupt_stage_envelope_even_without_final(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    PROBE._build_receipt(args)
    args.resume = True
    (args.output_dir / "measurement_receipt.json").unlink()
    stage = args.output_dir / "stage_02_argmax_certificate.json"
    envelope = json.loads(stage.read_text())
    envelope["schema"] = "corrupt"
    stage.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="envelope schema mismatch"):
        PROBE._build_receipt(args)


def test_precision_receipt_is_not_silently_coerced_to_rigorous_bound() -> None:
    result = PROBE._normalized_precision_allocation({"schema": "fixedpoint", "maximum_observed_logit_error": 0.01})
    assert result["status"] == "OWED_MEASURED_LAYER_BOUND_COST_TABLE"
    assert "not silently coerced" in result["reason"]


def test_static_sources_refuse_negative_timing_and_empty_tile_contract(
    tmp_path: Path,
) -> None:
    args = _arguments(tmp_path)
    timer = json.loads(args.verdict_wallclock_receipt.read_text())
    timer["combined_verdict_s_total"] = -1.0
    args.verdict_wallclock_receipt.write_text(json.dumps(timer), encoding="utf-8")
    with pytest.raises(ValueError, match="non-positive"):
        PROBE._build_receipt(args)

    args = _arguments(tmp_path / "tile_case")
    tile = json.loads(args.tile_halo_receipt.read_text())
    tile["exact_tile_contract"] = {}
    args.tile_halo_receipt.write_text(json.dumps(tile), encoding="utf-8")
    with pytest.raises(ValueError, match="full-frame closure contract"):
        PROBE._build_receipt(args)


def test_timer_prose_is_derived_from_validated_receipt(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    timer = json.loads(args.verdict_wallclock_receipt.read_text())
    timer.update(
        {
            "combined_verdict_s_total": 96.0,
            "per_pair_verdict_s": 1.0,
            "seg_fraction_of_verdict": 0.5,
            "pose_fraction_of_verdict": 0.5,
            "extrapolated_n600_verdict_s": 600.0,
            "extrapolated_n600_verdict_min": 10.0,
        }
    )
    args.verdict_wallclock_receipt.write_text(json.dumps(timer), encoding="utf-8")
    receipt = PROBE._build_receipt(args)
    assert "combined=96s" in receipt["premise_status"]
    assert "10min" in receipt["premise_status"]
    assert any("10-minute" in item for item in receipt["ranked_build_next"])


def test_host_wrapper_uses_content_addressed_run_directory() -> None:
    command = (ROOT / "tools/run_throughput_frontier_math_host.command").read_text()
    assert '--output-root "$OUT_ROOT"' in command
    assert "--resume" in command
    assert "RUN_KEY" not in command
    assert "shasum" not in command


def test_output_root_is_keyed_by_in_process_input_snapshot(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    args.output_root = tmp_path / "content_addressed"
    args.output_dir = None
    receipt = PROBE._build_receipt(args)
    expected = args.output_root / receipt["fingerprint"]
    assert receipt["content_addressed_output_dir"] == str(expected)
    assert (expected / "measurement_receipt.json").is_file()


def test_content_address_includes_original_source_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "content_addressed"
    first_args = _arguments(tmp_path / "a")
    first_args.output_root = output_root
    first_args.output_dir = None
    second_args = _arguments(tmp_path / "b")
    second_args.output_root = output_root
    second_args.output_dir = None

    first = PROBE._build_receipt(first_args)
    second = PROBE._build_receipt(second_args)

    assert first["fingerprint"] != second["fingerprint"]
    assert Path(first["content_addressed_output_dir"]).is_dir()
    assert Path(second["content_addressed_output_dir"]).is_dir()
