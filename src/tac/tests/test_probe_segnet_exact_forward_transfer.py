# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools/probe_segnet_exact_forward_transfer.py"
SPEC = importlib.util.spec_from_file_location("probe_segnet_exact_forward_transfer", TOOL)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_topology_derives_ceiling_and_candidates_from_observations() -> None:
    topology = probe.discover_thread_topology(
        torch_default=12,
        os_logical=16,
        psutil_logical=16,
        psutil_physical=8,
    )
    assert topology["label"] == "DERIVED"
    assert topology["effective_ceiling"] == 8
    assert topology["candidate_threads"] == list(range(1, 9))
    assert max(topology["candidate_threads"]) <= topology["effective_ceiling"]


def test_canary_count_is_explicitly_heuristic_and_n_pair_bounded() -> None:
    topology = {"candidate_threads": list(range(1, 7))}
    assert probe.derive_canary_count(topology, n_pairs=600) == 3
    assert probe.derive_canary_count(topology, n_pairs=2) == 2


def test_candidate_derivation_crosses_real_distinct_strategies_without_an_optimum_constant() -> None:
    topology = {"candidate_threads": [2, 4, 8]}
    arms = probe.derive_candidate_arms(topology)
    assert {(arm["strategy"], arm["threads"]) for arm in arms} == {
        ("eager_nchw_autograd", 2),
        ("eager_nchw_autograd", 4),
        ("eager_nchw_autograd", 8),
        ("eager_channels_last_autograd", 2),
        ("eager_channels_last_autograd", 4),
        ("eager_channels_last_autograd", 8),
    }
    source = TOOL.read_text()
    assert "selected_threads = 1" not in source
    assert '"threads": 1' not in source


def test_selection_key_changes_with_forward_signature_fields() -> None:
    signature = {
        "input_shape": [1, 3, 384, 512],
        "input_dtype": "torch.float32",
        "parameter_count": 99,
        "model_weights_sha256": "a" * 64,
        "torch_version": "2.test",
        "torch_config_sha256": "b" * 64,
        "dependency_sha256": "c" * 64,
    }
    arms = [{"strategy": "eager_nchw_autograd", "threads": 2}]
    first = probe.selection_key(signature, arms)
    second = probe.selection_key(signature | {"parameter_count": 100}, arms)
    assert first["sha256"] != second["sha256"]
    assert first["payload"]["forward_signature"] == signature


def test_canary_selection_rejects_fast_flipping_arm() -> None:
    rows = [
        {"strategy": "eager_channels_last_autograd", "threads": 4, "forward_ms_median": 1.0, "argmax_flip_count": 2},
        {"strategy": "eager_nchw_autograd", "threads": 4, "forward_ms_median": 3.0, "argmax_flip_count": 0},
        {"strategy": "eager_channels_last_autograd", "threads": 2, "forward_ms_median": 2.0, "argmax_flip_count": 0},
    ]
    selected = probe.select_canary_arm(rows)
    assert selected["strategy"] == "eager_channels_last_autograd"
    assert selected["threads"] == 2
    with pytest.raises(RuntimeError, match="no zero-argmax-flip"):
        probe.select_canary_arm([rows[0]])


def test_admission_requires_faster_distinct_arm_and_sha_equality() -> None:
    selected = {
        "strategy": "eager_channels_last_autograd",
        "threads": 4,
        "baseline_threads": 8,
    }
    valid = {
        "selected": selected,
        "n_pairs": 600,
        "verdict_pair_cardinality": 600,
        "baseline_median_ms": 20.0,
        "selected_median_ms": 10.0,
        "flip_count": 0,
        "reference_sha256": "a" * 64,
        "candidate_sha256": "a" * 64,
        "matched_sign_pvalue": 0.001,
        "matched_sign_alpha": 0.01,
    }
    assert probe.admit_selected_arm(**valid)
    assert not probe.admit_selected_arm(**(valid | {"candidate_sha256": "b" * 64}))
    assert not probe.admit_selected_arm(**(valid | {"flip_count": 1}))
    assert not probe.admit_selected_arm(**(valid | {"selected_median_ms": 20.0}))
    assert not probe.admit_selected_arm(**(valid | {"n_pairs": 2}))
    assert not probe.admit_selected_arm(**(valid | {"matched_sign_pvalue": 0.02}))
    baseline = selected | {"strategy": "eager_nchw_autograd", "threads": 8}
    assert not probe.admit_selected_arm(**(valid | {"selected": baseline}))


def test_matched_sign_test_is_exact_and_keeps_ties_out_of_denominator() -> None:
    result = probe.matched_sign_test([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert result["one_sided_exact_binomial_pvalue"] == 0.75
    assert result["median_gap_ms"] == 0.0


def test_compare_argmax_arrays_reports_pair_sha_and_flips() -> None:
    reference = probe.np.asarray([[0, 1, 2]], dtype=probe.np.uint8)
    candidate = reference.copy()
    result = probe.compare_argmax_arrays(reference, candidate)
    assert result["reference_pair_sha256"] == result["candidate_pair_sha256"]
    assert result["argmax_flip_count"] == 0
    candidate[0, 1] = 3
    result = probe.compare_argmax_arrays(reference, candidate)
    assert result["argmax_flip_count"] == 1
    assert result["reference_pair_sha256"] != result["candidate_pair_sha256"]


def test_storage_preflight_persists_no_argmax_bulk_and_fails_closed() -> None:
    checkpoint_dir = REPO_ROOT / "experiments/results/unit_transfer_checkpoints"
    result = probe.storage_preflight(checkpoint_dir, n_pairs=600, free_bytes=10**12)
    assert result["argmax_bulk_bytes"] == 0
    assert result["required_free_bytes"] == (
        600 * probe.CHECKPOINT_BYTES_PER_PAIR_UPPER_BOUND + probe.STORAGE_METADATA_RESERVE_BYTES
    )
    assert result["passed"]
    with pytest.raises(RuntimeError, match="storage preflight failed"):
        probe.storage_preflight(checkpoint_dir, n_pairs=600, free_bytes=1)


def _fingerprint(tmp_path: Path) -> dict:
    raw = tmp_path / "raw.bin"
    weights = tmp_path / "weights.bin"
    raw.write_bytes(b"raw")
    weights.write_bytes(b"weights")
    selection = {"sha256": "d" * 64}
    return probe.build_run_fingerprint(
        raw=raw,
        weights=weights,
        n_pairs=2,
        checkpoint_interval=1,
        canary_indices=[0, 1],
        selection=selection,
    )


def test_resume_validation_checks_fingerprint_and_vector_lengths(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path)
    state = {
        "schema": probe.CHECKPOINT_SCHEMA,
        "fingerprint_sha256": fingerprint["sha256"],
        "completed_pairs": 2,
        "baseline_ms": [2.0, 2.1],
        "selected_ms": [1.0, 1.1],
        "reference_pair_sha256": ["a" * 64, "b" * 64],
        "candidate_pair_sha256": ["a" * 64, "b" * 64],
        "pair_flip_counts": [0, 0],
    }
    assert probe.validate_resume_state(state, fingerprint) == 2
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        probe.validate_resume_state(state | {"fingerprint_sha256": "bad"}, fingerprint)
    with pytest.raises(RuntimeError, match="pair_flip_counts length mismatch"):
        probe.validate_resume_state(state | {"pair_flip_counts": [0]}, fingerprint)
    with pytest.raises(RuntimeError, match="completed-pair count"):
        probe.validate_resume_state(state | {"completed_pairs": 3}, fingerprint)
    with pytest.raises(RuntimeError, match="invalid SHA-256"):
        probe.validate_resume_state(state | {"reference_pair_sha256": ["INVALID", "b" * 64]}, fingerprint)
    with pytest.raises(RuntimeError, match="digest/flip inconsistency"):
        probe.validate_resume_state(state | {"candidate_pair_sha256": ["c" * 64, "b" * 64]}, fingerprint)
    with pytest.raises(RuntimeError, match="invalid timing"):
        probe.validate_resume_state(state | {"baseline_ms": [2.0, float("nan")]}, fingerprint)


def test_checkpoint_payload_round_trips_as_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = probe._checkpoint_payload(
        fingerprint={"sha256": "e" * 64},
        completed_pairs=2,
        baseline_ms=[4.0, 5.0],
        selected_ms=[2.0, 3.0],
        reference_pair_sha256=["a" * 64, "b" * 64],
        candidate_pair_sha256=["a" * 64, "b" * 64],
        pair_flip_counts=[0, 0],
    )
    probe._atomic_checkpoint(path, payload)
    assert json.loads(path.read_text()) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_defaults_to_full_n600_and_durable_output() -> None:
    args = probe.parse_args(
        ["--raw", "does-not-need-to-exist-at-parse.raw", "--out", "experiments/results/transfer/receipt.json"]
    )
    assert args.n_pairs == 600
    assert not hasattr(args, "canary_count")
    assert args.checkpoint_interval == 25
    assert args.out == (REPO_ROOT / "experiments/results/transfer/receipt.json").resolve()
    with pytest.raises(SystemExit):
        probe.parse_args(["--raw", "input.raw", "--out", "experiments/results/transfer/receipt"])


def test_policy_contract_is_composed_into_selection_key() -> None:
    topology = {
        "observed": {
            "torch_default": 6,
            "os_logical": 18,
            "psutil_logical": 18,
            "psutil_physical": 18,
        },
        "candidate_threads": list(range(1, 7)),
    }

    class Input:
        shape = (1, 3, 384, 512)

    contracts = probe.compile_policy_contracts(topology, Input())
    selection = probe.selection_key(
        {"input_shape": list(Input.shape)},
        [{"strategy": "eager_nchw_autograd", "threads": 1}],
        policy_contracts=contracts,
    )
    assert selection["payload"]["policy_contracts"] == contracts
    assert contracts["eager_nchw_autograd"]["verdict_pair_cardinality"] == 600


def test_exclusive_run_lock_refuses_second_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.base, "validate_durable_output", lambda path: path)
    out = tmp_path / "receipt.json"
    with (
        probe.exclusive_run_lock(out),
        pytest.raises(RuntimeError, match="another transfer probe owns"),
        probe.exclusive_run_lock(tmp_path / "receipt"),
    ):
        pass


def test_receipt_validator_rederives_terminal_verdict_and_rejects_tamper() -> None:
    n_pairs = 600
    arm = {"strategy": "eager_nchw_autograd", "threads": 1}
    topology = {"observed": {"torch_default": 6}}
    policy_contracts = {"eager_nchw_autograd": {"research_only": True}}
    selection_payload = {
        "candidate_arms": [arm],
        "forward_signature": {"thread_topology": topology},
        "policy_contracts": policy_contracts,
    }
    selection = {
        "payload": selection_payload,
        "sha256": probe.sha256_json(selection_payload),
    }
    fingerprint_payload = {
        "n_pairs": n_pairs,
        "selection_key_sha256": selection["sha256"],
    }
    fingerprint = {
        "payload": fingerprint_payload,
        "sha256": probe.sha256_json(fingerprint_payload),
    }
    pair_sha = "a" * 64
    sequence_sha = "b" * 64
    baseline_samples = [2.0] * n_pairs
    selected_samples = [1.0] * n_pairs
    state = {
        "schema": probe.CHECKPOINT_SCHEMA,
        "written_at_utc": "2026-07-13T00:00:00Z",
        "fingerprint_sha256": fingerprint["sha256"],
        "completed_pairs": n_pairs,
        "baseline_ms": baseline_samples,
        "selected_ms": selected_samples,
        "reference_pair_sha256": [pair_sha] * n_pairs,
        "candidate_pair_sha256": [pair_sha] * n_pairs,
        "pair_flip_counts": [0] * n_pairs,
    }
    baseline = probe.summarize_ms(baseline_samples)
    selected_summary = probe.summarize_ms(selected_samples)
    sign_test = probe.matched_sign_test(baseline_samples, selected_samples)
    canary_rows = [
        {
            **arm,
            "forward_ms_samples": [1.0],
            "forward_ms_median": 1.0,
            "argmax_flip_count": 0,
        }
    ]
    selected = {**canary_rows[0], "baseline_threads": 6}
    canary = {
        "schema": probe.CHECKPOINT_SCHEMA,
        "fingerprint_sha256": fingerprint["sha256"],
        "rows": canary_rows,
        "selected": selected,
    }
    canary["sha256"] = probe.sha256_json(canary)
    matched_sha = probe.sha256_json(probe.canonical_matched_state(state))
    terminal = {
        "written_at_utc": "2026-07-13T00:01:00Z",
        "fingerprint_sha256": fingerprint["sha256"],
        "canary_checkpoint_sha256": canary["sha256"],
        "matched_checkpoint_sha256": matched_sha,
        "reference_argmax_sha256": sequence_sha,
        "candidate_argmax_sha256": sequence_sha,
        "argmax_flip_count": 0,
    }
    terminal["sha256"] = probe.sha256_json(terminal)
    state["terminal_replay"] = terminal
    custody = {"sealed": True}
    receipt = {
        "schema": probe.SCHEMA,
        "verdict": "GO",
        "authority": {"score_claim": False, "pointer_moved": False, "promotion_eligible": False},
        "topology": topology,
        "policy_contracts": policy_contracts,
        "selection_key": selection,
        "selected_arm": selected,
        "canary_tournament": canary_rows,
        "measurement": {
            "n_real_pairs": n_pairs,
            "baseline": baseline,
            "selected": selected_summary,
            "matched_speedup_x": 2.0,
            "matched_sign_test": sign_test,
            "argmax_flip_count": 0,
            "argmax_flip_rate": 0.0,
            "reference_argmax_sha256": sequence_sha,
            "candidate_argmax_sha256": sequence_sha,
            "argmax_sha256_equal": True,
        },
        "resume": {
            "fingerprint": fingerprint,
            "canary_checkpoint_sha256": canary["sha256"],
            "matched_checkpoint_sha256": matched_sha,
        },
        "custody": {"start": custody, "end": custody},
    }
    probe.validate_receipt(
        receipt,
        current_custody=custody,
        latest_state=state,
        canary_state=canary,
        terminal_stage=terminal,
    )
    with pytest.raises(RuntimeError, match="verdict"):
        probe.validate_receipt(
            receipt | {"verdict": "NO-GO"},
            current_custody=custody,
            latest_state=state,
            canary_state=canary,
            terminal_stage=terminal,
        )
    forged_measurement = dict(receipt["measurement"])
    forged_measurement["baseline"] = selected_summary
    forged_measurement["selected"] = baseline
    with pytest.raises(RuntimeError, match="baseline_summary_from_checkpoint"):
        probe.validate_receipt(
            receipt | {"measurement": forged_measurement},
            current_custody=custody,
            latest_state=state,
            canary_state=canary,
            terminal_stage=terminal,
        )
