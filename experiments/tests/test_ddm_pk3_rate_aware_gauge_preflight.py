from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "ddm_pk3_rate_aware_gauge_preflight.py"
SPEC = importlib.util.spec_from_file_location("ddm_pk3_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pk3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pk3
SPEC.loader.exec_module(pk3)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory):
    retained = tmp_path_factory.mktemp("pk3_frozen") / "retained"
    copied = pk3.copy_pinned_receiver_inputs(retained, pk3.DEFAULT_ARCHIVE)
    codec, inflate = pk3.load_copied_receiver_sources(retained)
    bundle = pk3.pk2.extract_bundle(Path(copied["archive.zip"]["copied_path"]))
    baseline = pk3.baseline_state(bundle, codec)
    normalized = pk3.normalized_basis_tensor(baseline.basis, inflate)
    return codec, inflate, bundle, baseline, normalized


def test_copied_runtime_and_runner_are_exact_and_loaded_only_from_retention(frozen) -> None:
    codec, inflate, _, _, _ = frozen
    expected_root = Path(codec.__file__).resolve().parents[1] / f"pr130_runtime_{pk3.EXPECTED_INTAKE_COMMIT[:12]}"
    assert Path(codec.__file__).resolve().parent == expected_root
    assert Path(inflate.__file__).resolve().parent == expected_root
    assert pk3.sha256_file(Path(codec.__file__)) == pk3.EXPECTED_CODEC_SHA256
    assert pk3.sha256_file(Path(inflate.__file__)) == pk3.EXPECTED_INFLATE_SHA256
    runner_copy = Path(codec.__file__).resolve().parents[1] / "pk3_runner.py"
    assert pk3.sha256_file(runner_copy) == pk3.RUNNER_SOURCE_SHA256_AT_IMPORT


def test_copied_input_manifest_refuses_tamper(tmp_path: Path) -> None:
    retained = tmp_path / "retained"
    copied = pk3.copy_pinned_receiver_inputs(retained, pk3.DEFAULT_ARCHIVE)
    manifest_path = (
        retained
        / "inputs"
        / f"pr130_runtime_{pk3.EXPECTED_INTAKE_COMMIT[:12]}"
        / "MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["archive.zip"]["bytes"] += 1
    pk3.atomic_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="drifted copied-input manifest"):
        pk3.copy_pinned_receiver_inputs(
            retained, Path(copied["archive.zip"]["intake_path"])
        )


def test_sign_permutation_and_global_dc_are_receiver_gauges(frozen) -> None:
    _, inflate, _, baseline, _ = frozen
    signs = np.ones(pk3.pk2.DIM, dtype=np.int32)
    signs[[1, 5, 9]] = -1
    signed = pk3.signed_state(baseline, signs)
    assert (
        pk3.receiver_product_mse_explicit(
            signed.basis,
            signed.coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
            batch_size=16,
        )
        < 1e-14
    )

    permutation = np.arange(pk3.pk2.DIM, dtype=np.int32)[::-1]
    permuted = pk3.permuted_state(baseline, permutation)
    assert (
        pk3.receiver_product_mse_contraction(
            permuted.basis,
            permuted.coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
        )
        < 1e-12
    )

    shifts = np.zeros(pk3.pk2.DIM, dtype=np.int32)
    for dimension in (0, 4, 8):
        shifts[dimension] = 1 if baseline.basis_codes[dimension].max() < 15 else -1
    shifted = pk3.dc_shifted_state(baseline, shifts)
    assert (
        pk3.receiver_product_mse_contraction(
            shifted.basis,
            shifted.coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
        )
        < 1e-12
    )

    nonconstant = baseline.basis.copy()
    nonconstant[0, 0] += baseline.basis_scales[0]
    assert (
        pk3.receiver_product_mse_contraction(
            nonconstant,
            baseline.coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
        )
        > 0.0
    )


def test_basis_radial_is_null_but_reciprocal_coefficient_scale_is_not(frozen) -> None:
    _, inflate, _, baseline, _ = frozen
    radial = pk3.scale_state(baseline, dimension=3, exponent=5)
    assert radial.basis_scales[3] == np.ldexp(baseline.basis_scales[3], 5)
    assert np.array_equal(radial.coefficient_scales, baseline.coefficient_scales)
    radial_mse = pk3.receiver_product_mse_contraction(
        radial.basis,
        radial.coefficients,
        baseline.basis,
        baseline.coefficients,
        inflate,
    )
    assert radial_mse < 1e-10

    invalid_coeff_scales = radial.coefficient_scales.copy()
    invalid_coeff_scales[3] = np.ldexp(invalid_coeff_scales[3], -5)
    invalid = pk3.GaugeState(
        candidate_id="invalid_reciprocal_scale",
        sequence=(),
        basis_codes=radial.basis_codes,
        basis_scales=radial.basis_scales,
        coefficient_codes=radial.coefficient_codes,
        coefficient_scales=invalid_coeff_scales,
    )
    invalid_mse = pk3.receiver_product_mse_contraction(
        invalid.basis,
        invalid.coefficients,
        baseline.basis,
        baseline.coefficients,
        inflate,
    )
    assert invalid_mse > pk3.MSE_BAR


def test_raw_givens_canary_exposes_wrong_pk2_metric_and_contraction_matches_direct(
    frozen,
) -> None:
    _, inflate, _, baseline, _ = frozen
    matrix = pk3.givens_matrix(0, 7, 0.2)
    raw_coefficients = np.einsum("nd,dk->nk", baseline.coefficients.astype(np.float64), matrix, optimize=False).astype(
        np.float32
    )
    raw_basis = np.einsum(
        "ij,jchw->ichw",
        np.linalg.inv(matrix),
        baseline.basis.astype(np.float64),
        optimize=False,
    ).astype(np.float32)
    raw_mse = pk3.pk2.carrier_product_mse(
        raw_basis,
        raw_coefficients,
        baseline.basis,
        baseline.coefficients,
    )
    contracted = pk3.receiver_product_mse_contraction(
        raw_basis,
        raw_coefficients,
        baseline.basis,
        baseline.coefficients,
        inflate,
    )
    explicit = pk3.receiver_product_mse_explicit(
        raw_basis,
        raw_coefficients,
        baseline.basis,
        baseline.coefficients,
        inflate,
        batch_size=16,
    )
    assert raw_mse < 1e-12
    assert explicit > 0.0
    assert contracted == pytest.approx(explicit, rel=2e-5, abs=5e-10)


def test_normalization_aware_transform_preserves_prequant_product_then_quantizes(
    frozen,
) -> None:
    _, inflate, _, baseline, _ = frozen
    matrix = pk3.shear_matrix(2, 11, -1.0 / 256.0)
    parent_rms = pk3.centered_bicubic_rms(baseline.basis, inflate)
    exact_basis = np.einsum(
        "ij,jchw->ichw",
        matrix,
        baseline.basis.astype(np.float64) / parent_rms[:, None, None, None],
        optimize=False,
    ).astype(np.float32)
    mixed_rms = pk3.centered_bicubic_rms(exact_basis, inflate)
    exact_coefficients = (
        np.einsum(
            "nd,dk->nk",
            baseline.coefficients.astype(np.float64),
            np.linalg.inv(matrix),
            optimize=False,
        )
        * mixed_rms[None]
    )
    assert (
        pk3.receiver_product_mse_contraction(
            exact_basis,
            exact_coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
        )
        < 1e-10
    )
    quantized = pk3.transformed_state(
        baseline,
        matrix,
        {"kind": "test_normalization_aware_shear"},
        inflate,
    )
    assert quantized.basis_codes.shape == pk3.pk2.BASIS_SHAPE
    assert quantized.coefficient_codes.shape == (pk3.pk2.N, pk3.pk2.DIM)
    assert quantized.sequence[-1]["condition_number"] < 4.0
    assert "diag(r(A)^-1)" in quantized.sequence[-1]["gauge_law"]


def test_full_population_sentinels_change_hashes_and_receiver_mse(frozen) -> None:
    _, inflate, _, baseline, _ = frozen
    basis = baseline.basis.copy()
    basis.reshape(-1)[-1] += baseline.basis_scales[-1]
    coefficients = baseline.coefficients.copy()
    coefficients[-1, -1] += baseline.coefficient_scales[-1]
    assert pk3.array_sha256(basis) != pk3.array_sha256(baseline.basis)
    assert pk3.array_sha256(coefficients) != pk3.array_sha256(baseline.coefficients)
    assert (
        pk3.receiver_product_mse_contraction(
            basis,
            coefficients,
            baseline.basis,
            baseline.coefficients,
            inflate,
        )
        > 0.0
    )


def test_real_materializer_retains_payloads_and_permutation_preserves_inner_bits(frozen, tmp_path: Path) -> None:
    codec, inflate, bundle, baseline, normalized = frozen
    retained = tmp_path / "retained"
    control = pk3.materialize_candidate(
        stage="test_control",
        parent=baseline,
        state=baseline,
        bundle=bundle,
        codec=codec,
        inflate=inflate,
        baseline=baseline,
        baseline_normalized_basis=normalized,
        retained_root=retained,
    )
    carrier_path = Path(control["retained_payloads"]["carrier"]["path"])
    archive_path = Path(control["retained_payloads"]["archive"]["path"])
    assert carrier_path.read_bytes() == bundle.carrier
    assert archive_path.read_bytes() == bundle.archive_blob
    assert control["real_coder"]["receiver_parse_back"] is True
    assert control["identity"]["pk3_runner_sha256"] == pk3.RUNNER_SOURCE_SHA256_AT_IMPORT
    assert (
        control["receiver_carrier_product_mse_explicit_population_values"]
        == 600 * 3 * 384 * 512
    )

    receipt_path = carrier_path.parent / "candidate.json"
    scalar_tamper = json.loads(receipt_path.read_text(encoding="utf-8"))
    scalar_tamper["receiver_carrier_product_mse"] = 123.0
    scalar_tamper["receiver_carrier_product_mse_explicit"] = 123.0
    scalar_tamper["mse_bar_strict_explicit_pass"] = False
    pk3.atomic_json(receipt_path, scalar_tamper)
    refreshed = pk3.materialize_candidate(
        stage="test_control",
        parent=baseline,
        state=baseline,
        bundle=bundle,
        codec=codec,
        inflate=inflate,
        baseline=baseline,
        baseline_normalized_basis=normalized,
        retained_root=retained,
    )
    assert refreshed["receiver_carrier_product_mse"] == 0.0
    assert refreshed["mse_bar_strict_explicit_pass"] is True

    path_tamper = json.loads(json.dumps(refreshed))
    path_tamper["retained_payloads"]["carrier"]["path"] = refreshed[
        "retained_payloads"
    ]["archive"]["path"]
    pk3.atomic_json(receipt_path, path_tamper)
    with pytest.raises(RuntimeError, match="path escaped candidate directory"):
        pk3.materialize_candidate(
            stage="test_control",
            parent=baseline,
            state=baseline,
            bundle=bundle,
            codec=codec,
            inflate=inflate,
            baseline=baseline,
            baseline_normalized_basis=normalized,
            retained_root=retained,
        )
    pk3.atomic_json(receipt_path, refreshed)

    permutation = np.arange(pk3.pk2.DIM, dtype=np.int32)
    permutation[0], permutation[11] = permutation[11], permutation[0]
    permuted = pk3.permuted_state(baseline, permutation)
    candidate = pk3.materialize_candidate(
        stage="test_permutation",
        parent=baseline,
        state=permuted,
        bundle=bundle,
        codec=codec,
        inflate=inflate,
        baseline=baseline,
        baseline_normalized_basis=normalized,
        retained_root=retained,
    )
    assert candidate["real_coder"]["basis_bit_count"] == control["real_coder"]["basis_bit_count"]
    assert candidate["real_coder"]["coefficient_bit_count"] == control["real_coder"]["coefficient_bit_count"]

    archive_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="retained archive drift"):
        pk3.materialize_candidate(
            stage="test_control",
            parent=baseline,
            state=baseline,
            bundle=bundle,
            codec=codec,
            inflate=inflate,
            baseline=baseline,
            baseline_normalized_basis=normalized,
            retained_root=retained,
        )


def test_absolute_codes_mistaken_for_delta_codes_do_not_reproduce_baseline(frozen) -> None:
    codec, _, _, baseline, _ = frozen
    with pytest.raises(ValueError, match="coefficient code is outside the 12-bit range"):
        codec.encode_compact_carrier(
            baseline.basis_scales,
            baseline.basis_codes.reshape(-1),
            baseline.coefficient_scales,
            baseline.coefficient_codes,
        )


@pytest.mark.parametrize(
    ("saved", "mse", "expected"),
    [
        (2_000, np.nextafter(pk3.MSE_BAR, 0.0), True),
        (1_999, 0.0, False),
        (2_000, pk3.MSE_BAR, False),
    ],
)
def test_trigger_is_strict_same_row_and_uses_full_archive_bytes(saved: int, mse: float, expected: bool) -> None:
    baseline_bytes = 191_052
    receipt = {
        "identity": {
            "candidate_id": "candidate",
            "stage": "test",
            "parent_candidate_id": "parent",
            "sequence": [{"kind": "test"}],
        },
        "retained_payloads": {
            "archive": {
                "bytes": baseline_bytes - saved,
                "sha256": "a" * 64,
                "path": "/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/receiver_v6/a.zip",
            },
            "carrier": {
                "bytes": 23_054,
                "sha256": "b" * 64,
                "path": "/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/receiver_v6/c.cpr1",
            },
        },
        "real_coder": {"basis_bit_count": 104_135, "coefficient_bit_count": 79_076},
        "receiver_carrier_product_mse": mse,
        "receiver_carrier_product_mse_method": "test_explicit",
        "receiver_carrier_product_mse_explicit": mse,
        "receiver_carrier_product_mse_explicit_population_values": 600 * 3 * 384 * 512,
        "raw_stored_factor_product_mse_non_authority": 0.0,
        "mse_bar_strict_screen_pass": mse < pk3.MSE_BAR,
        "mse_bar_strict_explicit_pass": mse < pk3.MSE_BAR,
    }
    row = pk3.receipt_row(receipt, baseline_bytes)
    assert row["archive_bytes_saved"] == saved
    assert row["trigger_pass"] is expected

    screen_only = dict(receipt)
    screen_only["receiver_carrier_product_mse_explicit"] = None
    screen_only["mse_bar_strict_explicit_pass"] = None
    assert pk3.receipt_row(screen_only, baseline_bytes)["trigger_pass"] is False


def test_output_root_requires_versioned_arm_specific_ssd_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        pk3.validate_out_dir(tmp_path)
    accepted = pk3.DEFAULT_OUT
    assert pk3.validate_out_dir(accepted) == accepted
    with pytest.raises(ValueError, match="resume-from"):
        pk3.validate_resume_path(accepted, tmp_path / "progress.json")
    assert pk3.validate_resume_path(accepted, accepted / "progress.json") == (
        accepted / "progress.json"
    )
    for live_store in (
        Path("/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/pk3"),
        Path("/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/pk3"),
    ):
        with pytest.raises(ValueError, match="PK3 SSD root"):
            pk3.validate_out_dir(live_store)


def test_verification_receipt_binds_runner_test_argv_and_log(tmp_path: Path) -> None:
    out = tmp_path / "receiver"
    log_path = out / "verification" / "pytest.log"
    receipt_path = out / "verification" / "receipt.json"
    pk3.atomic_write_text(log_path, "...............\n15 passed in 1.23s\n")
    expected_argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(pk3.TEST_PATH.relative_to(pk3.REPO)),
    ]
    receipt = {
        "schema": "ddm_pk3_verification.v1",
        "created_at_utc": "2026-08-09T00:00:00Z",
        "identity": {
            "runner_sha256": pk3.RUNNER_SOURCE_SHA256_AT_IMPORT,
            "test_sha256": pk3.sha256_file(pk3.TEST_PATH),
            "test_command": pk3.VERIFICATION_TEST_COMMAND,
        },
        "actual_argv": expected_argv,
        "actual_argv_shell": "unused-by-verdict",
        "exit_code": 0,
        "passed_count": 15,
        "log": {
            "path": str(log_path),
            "bytes": log_path.stat().st_size,
            "sha256": pk3.sha256_file(log_path),
        },
    }
    pk3.atomic_json(receipt_path, receipt)
    validated = pk3.verification_receipt(out, receipt_path)
    assert validated["passed_count"] == 15

    tampered = dict(receipt)
    tampered["actual_argv"] = [*expected_argv, "--collect-only"]
    pk3.atomic_json(receipt_path, tampered)
    with pytest.raises(RuntimeError, match="argv drift"):
        pk3.verification_receipt(out, receipt_path)

    pk3.atomic_json(receipt_path, receipt)
    pk3.atomic_write_text(log_path, "15 passed in 1.23s\n1 ERROR in teardown\n")
    receipt["log"] = {
        "path": str(log_path),
        "bytes": log_path.stat().st_size,
        "sha256": pk3.sha256_file(log_path),
    }
    pk3.atomic_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="clean pytest summary"):
        pk3.verification_receipt(out, receipt_path)


def test_complete_progress_is_identity_checked_then_returned_for_full_replay(
    tmp_path: Path,
) -> None:
    out = tmp_path / "receiver"
    final_path = out / "FINAL_RECEIPT.json"
    results_path = out / "RESULTS.md"
    progress_path = out / "progress.json"
    pk3.atomic_json(final_path, {"status": "NOT_MET"})
    pk3.atomic_write_text(results_path, "result\n")
    progress = {
        "schema": "ddm_pk3_rate_aware_gauge_progress.v5",
        "complete": True,
        "status": "NOT_MET",
        "out_dir": str(out),
        "runner_sha256": pk3.RUNNER_SOURCE_SHA256_AT_IMPORT,
        "final_receipt": str(final_path),
        "final_receipt_sha256": pk3.sha256_file(final_path),
        "results": str(results_path),
        "results_sha256": pk3.sha256_file(results_path),
    }
    pk3.atomic_json(progress_path, progress)
    loaded = pk3.load_resume_progress_if_valid(progress_path, out)
    assert loaded == progress

    progress["runner_sha256"] = "0" * 64
    pk3.atomic_json(progress_path, progress)
    with pytest.raises(RuntimeError, match="runner identity drift"):
        pk3.load_resume_progress_if_valid(progress_path, out)


def test_preregistered_bank_cap_arithmetic_is_exact() -> None:
    assert 1 + 71 + 59 + 66 + 3 + 180 + 64 + 8 == 452


def test_retention_refuses_symlink_targets(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"payload")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(real)
    with pytest.raises(RuntimeError, match="symlinked retained"):
        pk3.retain_or_verify_exact_payload(linked, b"payload", "fixture")
