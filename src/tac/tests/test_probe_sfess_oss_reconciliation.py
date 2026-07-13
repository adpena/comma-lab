# SPDX-License-Identifier: MIT
"""Terminal-resume custody test for the SFESS OSS reconciliation probe."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_sfess_oss_reconciliation_test",
    REPO / "tools/probe_sfess_oss_reconciliation.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_arm(k: int = 1, *, best_s: float = 1.0) -> dict:
    return {
        "arm": f"clean_room_k{k}_m5",
        "k": k,
        "samples_per_gradient": 5,
        "best_mask": [1, 0],
        "best_s": best_s,
        "function_evals": 64,
        "accepted_swaps": 0,
        "padding_calls": 63,
    }


def _enriched_arm(k: int = 1, samples: int = 32, *, best_s: float = 1.0) -> dict:
    return {
        "arm": f"learned_logits_k{k}_m{samples}",
        "k": k,
        "samples_per_gradient": samples,
        "best_mask": [1, 0],
        "best_s": best_s,
        "function_evals": 64,
        "gradient_steps": 1,
        "accepted_optimizer_updates": 0,
        "rejected_optimizer_updates": 1,
        "strict_gate_calls": 1,
        "zero_variance_skips": 0,
        "padding_calls": 30,
        "min_sampled_value_spread_s": 0.25,
        "max_sampled_value_spread_s": 0.5,
        "final_logits": [0.0, 0.0],
    }


def _partial_fixture(tmp_path, monkeypatch, *, arms: list[dict]) -> tuple[dict, dict]:
    config = {"source": "sealed"}
    monkeypatch.setattr(probe, "K_VALUES", (1,))
    monkeypatch.setattr(probe, "SAMPLE_COUNTS", (32,))
    monkeypatch.setattr(probe, "_config", lambda: config)
    checkpoint = {
        "schema": probe.SCHEMA,
        "config": config,
        "clean_arms": [_clean_arm()],
        "arms": arms,
    }
    recomputed = {
        "schema": probe.SCHEMA,
        "config": config,
        "clean_arms": [_clean_arm()],
        "arms": [_enriched_arm()],
    }
    (tmp_path / "stage_checkpoint.json").write_text(
        json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        probe,
        "_run_clean_control",
        lambda *_args, **_kwargs: pytest.fail("completed clean arm was unexpectedly rerun"),
    )
    monkeypatch.setattr(
        probe,
        "_run_arm",
        lambda *_args, **_kwargs: pytest.fail("completed enriched arm was unexpectedly rerun"),
    )
    monkeypatch.setattr(
        probe,
        "_recompute_complete_checkpoint",
        lambda _output_dir, _config: copy.deepcopy(recomputed),
    )
    return checkpoint, recomputed


def _terminal_fixture(tmp_path, monkeypatch):
    config = {"source": "sealed"}
    monkeypatch.setattr(probe, "_config", lambda: config)
    clean_receipt = tmp_path / "clean_receipt.json"
    clean_receipt.write_text(
        json.dumps({"best_non_degenerate_sfess_s": 1.0, "exact_enumeration_s": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "CLEAN_RECEIPT", clean_receipt)
    checkpoint = {
        "schema": probe.SCHEMA,
        "config": config,
        "clean_arms": [
            {"arm": f"clean_room_k{k}_m5", "k": k, "best_s": 1.0}
            for k in probe.K_VALUES
        ],
        "arms": [
            {
                "arm": f"learned_logits_k{k}_m{samples}",
                "k": k,
                "samples_per_gradient": samples,
                "best_s": 1.0,
            }
            for samples in probe.SAMPLE_COUNTS
            for k in probe.K_VALUES
        ],
    }
    recomputed = copy.deepcopy(checkpoint)
    monkeypatch.setattr(
        probe,
        "_recompute_complete_checkpoint",
        lambda _output_dir, _config: copy.deepcopy(recomputed),
    )
    receipt = probe._build_receipt(
        checkpoint,
        generated_at_utc="2026-07-13T00:00:00+00:00",
        runtime={"python": "sealed", "platform": "sealed"},
    )
    checkpoint_path = tmp_path / "stage_checkpoint.json"
    terminal_path = tmp_path / "measurement_receipt.json"
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True), encoding="utf-8")
    terminal_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return checkpoint, receipt, checkpoint_path, terminal_path


def test_terminal_resume_authenticates_without_rewrite(tmp_path, monkeypatch) -> None:
    _checkpoint, receipt, checkpoint_path, terminal_path = _terminal_fixture(tmp_path, monkeypatch)
    before = (_sha256(checkpoint_path), _sha256(terminal_path))

    assert probe.run(tmp_path) == receipt
    assert (_sha256(checkpoint_path), _sha256(terminal_path)) == before


def test_terminal_resume_rejects_tampered_authority_fields(tmp_path, monkeypatch) -> None:
    _checkpoint, receipt, _checkpoint_path, terminal_path = _terminal_fixture(tmp_path, monkeypatch)
    receipt["verdict"] = "GO"
    receipt["score_claim"] = True
    receipt["pointer_moved"] = True
    terminal_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="terminal receipt content authentication failed"):
        probe.run(tmp_path)


def test_terminal_resume_rejects_tampered_checkpoint_arm(tmp_path, monkeypatch) -> None:
    checkpoint, receipt, checkpoint_path, terminal_path = _terminal_fixture(tmp_path, monkeypatch)
    checkpoint["arms"][0]["best_s"] = -999.0
    receipt["arms"][0]["best_s"] = -999.0
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True), encoding="utf-8")
    terminal_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="terminal checkpoint content authentication failed"):
        probe.run(tmp_path)


def test_first_terminal_write_rejects_forged_partial_value(tmp_path, monkeypatch) -> None:
    _partial_fixture(tmp_path, monkeypatch, arms=[_enriched_arm(best_s=-999.0)])

    with pytest.raises(RuntimeError, match="pre-terminal checkpoint content authentication failed"):
        probe.run(tmp_path)
    assert not (tmp_path / "measurement_receipt.json").exists()


def test_first_terminal_write_uses_independently_recomputed_checkpoint(
    tmp_path, monkeypatch
) -> None:
    _checkpoint, recomputed = _partial_fixture(
        tmp_path, monkeypatch, arms=[_enriched_arm()]
    )
    observed = {}

    def build(checkpoint, *, generated_at_utc, runtime):
        observed["checkpoint"] = copy.deepcopy(checkpoint)
        observed["generated_at_utc"] = generated_at_utc
        observed["runtime"] = runtime
        return {"authenticated": True}

    monkeypatch.setattr(probe, "_build_receipt", build)
    assert probe.run(tmp_path) == {"authenticated": True}
    assert observed["checkpoint"] == recomputed
    assert observed["generated_at_utc"]
    assert set(observed["runtime"]) == {"python", "platform"}
    assert json.loads((tmp_path / "measurement_receipt.json").read_text()) == {
        "authenticated": True
    }


def test_partial_checkpoint_rejects_duplicate_arm_identity(tmp_path, monkeypatch) -> None:
    arm = _enriched_arm()
    _partial_fixture(tmp_path, monkeypatch, arms=[arm, copy.deepcopy(arm)])

    with pytest.raises(RuntimeError, match="duplicate arm identity"):
        probe.run(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("k", 2), ("samples_per_gradient", 2)),
)
def test_partial_checkpoint_rejects_swapped_k_or_m(
    tmp_path, monkeypatch, field: str, value: int
) -> None:
    arm = _enriched_arm()
    arm[field] = value
    _partial_fixture(tmp_path, monkeypatch, arms=[arm])

    with pytest.raises(RuntimeError, match="k/M identity mismatch"):
        probe.run(tmp_path)


@pytest.mark.parametrize("mutation", ("missing", "extra", "nonfinite"))
def test_partial_checkpoint_rejects_malformed_payload(tmp_path, monkeypatch, mutation: str) -> None:
    arm = _enriched_arm()
    if mutation == "missing":
        arm.pop("best_s")
    elif mutation == "extra":
        arm["untrusted"] = True
    else:
        arm["best_s"] = float("nan")
    _partial_fixture(tmp_path, monkeypatch, arms=[arm])

    with pytest.raises(RuntimeError, match=r"field schema mismatch|best_s is not finite numeric"):
        probe.run(tmp_path)
    assert not (tmp_path / "measurement_receipt.json").exists()
