# SPDX-License-Identifier: MIT
"""Tests for ``--full-cpu`` mode in time-traveler L5 autonomy trainer.

Operator approved 2026-05-13. Per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #127 + #192 + #197:
``--full-cpu`` opens a non-smoke CPU path that produces ``[macOS-CPU advisory
only]`` non-promotable scores. The mode requires explicit operator
acknowledgement via ``--advisory-cpu-explicitly-waived``.

Scope:

- (R1) flag parsing + defaults
- (R1) waiver requirement (the device-or-die-gate-bypass-via-uncoupled-flags
  bug class is structurally extincted here)
- (R1) mutual exclusion with ``--smoke``
- (R1) ``--full-cpu`` requires ``--device cpu``
- (R1) dangling waiver flag refused
- (R2) ``--max-wall-clock-hours`` default + custom value
- (R2) banner written to stderr
- (R3) CLI help mentions advisory-only nature

The trainer's full _full_main path is exercised by remote dispatches; here we
test the contract surface (CLI + helper functions) that this landing adds.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_time_traveler_l5_autonomy.py"


@pytest.fixture(scope="module")
def trainer_module():
    """Load the trainer module by path (it's not a package member)."""
    spec = importlib.util.spec_from_file_location(
        "tt5l_trainer_under_test", str(TRAINER_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_val_loop_applies_same_quantized_side_info_path_as_inflate(trainer_module):
    """Validation must score the same residual-corrected frames as training/inflate."""

    class StubSubstrate:
        def eval(self):
            self.did_eval = True

        def render_pair(self, pair_idx: int):
            assert pair_idx == 0
            frame = torch.full((1, 3, 8, 8), 0.5)
            return frame.clone(), frame.clone()

    class MeanLoss:
        def __call__(
            self,
            pred_a,
            pred_b,
            gt_a,
            gt_b,
            archive_bytes_proxy,
            *,
            apply_eval_roundtrip: bool,
            noise_std: float,
            **_kwargs,
        ):
            assert apply_eval_roundtrip is True
            assert noise_std == 0.0
            assert archive_bytes_proxy.item() == 123.0
            assert gt_a.shape == (1, 3, 8, 8)
            assert gt_b.shape == (1, 3, 8, 8)
            return pred_a.mean() + pred_b.mean(), {}

    gt_tensor = torch.zeros((1, 2, 3, 8, 8))
    zero_side = torch.zeros((1, 45))
    active_side = torch.zeros((1, 45))
    active_side[0, 36:45] = 64.0

    zero_val = trainer_module._run_val_loop(
        StubSubstrate(),
        MeanLoss(),
        gt_tensor,
        [0],
        torch.tensor(123.0),
        torch.device("cpu"),
        per_pair_side_info_float=zero_side,
        int8_scale=64.0,
    )
    active_val = trainer_module._run_val_loop(
        StubSubstrate(),
        MeanLoss(),
        gt_tensor,
        [0],
        torch.tensor(123.0),
        torch.device("cpu"),
        per_pair_side_info_float=active_side,
        int8_scale=64.0,
    )

    assert active_val != zero_val


def test_state_dict_cpu_snapshot_clones_current_weights(trainer_module):
    """Checkpoint snapshots must preserve EMA-applied weights after live restore."""

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    snapshot = trainer_module._state_dict_cpu_snapshot(model)
    with torch.no_grad():
        model.weight.fill_(9.0)

    assert snapshot["weight"].device.type == "cpu"
    assert snapshot["weight"].item() == 2.0
    assert model.weight.item() == 9.0


def test_archive_byte_proxy_calibration_records_drift(trainer_module):
    proxy = trainer_module._archive_bytes_proxy_estimate(
        world_model_bytes=1000,
        n_pairs=4,
        per_pair_side_info_bytes=45,
    )
    calibration = trainer_module._byte_proxy_calibration(
        proxy_bytes=proxy,
        bin_bytes=1500,
        archive_zip_bytes=1400,
    )

    assert proxy == 1180
    assert calibration["proxy_bytes"] == 1180
    assert calibration["bin_minus_proxy_bytes"] == 320
    assert calibration["archive_zip_minus_proxy_bytes"] == 220
    assert calibration["bin_to_proxy_ratio"] == pytest.approx(1500 / 1180)


def test_full_trainer_side_info_initializes_at_quantized_lsb(trainer_module):
    """TT5L full training must not start with an archive-dead side channel."""

    generator = torch.Generator(device="cpu").manual_seed(1705)
    side_info = trainer_module._initialize_per_pair_side_info_float(
        n_pairs=8,
        per_pair_side_info_bytes=45,
        int8_scale=64.0,
        device=torch.device("cpu"),
        generator=generator,
    )

    quantized = (side_info.detach() * 64.0).round().clamp(-128, 127).to(torch.int8)

    assert side_info.requires_grad is True
    assert side_info.shape == (8, 45)
    assert torch.count_nonzero(side_info.detach()).item() > 0
    assert torch.count_nonzero(quantized).item() > 0
    assert quantized.abs().max().item() < 8


def test_export_side_info_liveness_stats_reject_all_zero_int8(trainer_module):
    """TT5L export must not silently emit another all-zero side-info packet."""

    zero_side_info = torch.zeros((8, 45), dtype=torch.int8)
    live_side_info = zero_side_info.clone()
    live_side_info[0, 0] = 1

    with pytest.raises(RuntimeError, match="tt5l_side_info_all_zero_export"):
        trainer_module._require_live_quantized_side_info_for_export(zero_side_info)

    stats = trainer_module._require_live_quantized_side_info_for_export(
        live_side_info
    )
    assert stats["checked"] is True
    assert stats["dtype"] == "torch.int8"
    assert stats["shape"] == [8, 45]
    assert stats["total_values"] == 360
    assert stats["nonzero_values"] == 1
    assert stats["nonzero_fraction"] == pytest.approx(1 / 360)
    assert stats["min"] == 0
    assert stats["max"] == 1


def test_export_side_info_liveness_rejects_empty_int8(trainer_module):
    with pytest.raises(RuntimeError, match="tt5l_side_info_empty_export"):
        trainer_module._require_live_quantized_side_info_for_export(
            torch.zeros((0, 45), dtype=torch.int8)
        )


def test_skip_archive_build_provenance_is_non_promoting(trainer_module):
    """``--skip-archive-build`` must not leave a promotable phantom packet."""

    provenance = {
        "dispatch_blockers": ["existing_blocker"],
        "score_claim": True,
        "promotion_eligible": True,
        "ready_for_exact_eval_dispatch": True,
    }

    result = trainer_module._apply_archive_build_skipped_provenance(provenance)

    assert result is provenance
    assert provenance["archive_build_skipped"] is True
    assert provenance["score_claim"] is False
    assert provenance["promotion_eligible"] is False
    assert provenance["ready_for_exact_eval_dispatch"] is False
    assert provenance["dispatch_blockers"] == [
        "existing_blocker",
        "archive_build_skipped_no_archive_zip",
        "auth_eval_skipped_archive_build_skipped",
        "no_byte_closed_packet_for_score_claim",
    ]


def test_shape_readiness_blocks_smoke_partial_and_full_cpu(trainer_module):
    smoke = trainer_module._shape_readiness_manifest(
        n_pairs=4,
        output_height=24,
        output_width=32,
        smoke=True,
        full_cpu=False,
    )
    full_cpu = trainer_module._shape_readiness_manifest(
        n_pairs=trainer_module.N_PAIRS_FULL,
        output_height=trainer_module.EVAL_HW[0],
        output_width=trainer_module.EVAL_HW[1],
        smoke=False,
        full_cpu=True,
    )
    exact_cuda = trainer_module._shape_readiness_manifest(
        n_pairs=trainer_module.N_PAIRS_FULL,
        output_height=trainer_module.EVAL_HW[0],
        output_width=trainer_module.EVAL_HW[1],
        smoke=False,
        full_cpu=False,
    )

    assert smoke["score_claim_allowed"] is False
    assert "smoke_artifact_not_score_evidence" in smoke["promotion_blockers"]
    assert "partial_pair_count_not_contest_shape" in smoke["promotion_blockers"]
    assert "renderer_resolution_not_contest_scorer_shape" in smoke["promotion_blockers"]
    assert full_cpu["exact_contest_shape"] is True
    assert full_cpu["score_claim_allowed"] is False
    assert "full_cpu_advisory_not_promotion_axis" in full_cpu["promotion_blockers"]
    assert exact_cuda["score_claim_allowed"] is True
    assert exact_cuda["promotion_blockers"] == []


def test_best_checkpoint_uses_ema_snapshot_taken_before_live_restore() -> None:
    """The full trainer must save the EMA-applied state, not restored live weights."""

    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert "ema_eval_state = _state_dict_cpu_snapshot(substrate)" in text
    assert "substrate.load_state_dict(live_state)" in text
    assert '"state_dict": ema_eval_state' in text


def test_posterior_update_passes_architecture_class_and_records_acceptance(
    trainer_module,
    monkeypatch,
    tmp_path,
):
    """TT5L posterior updates must not lose anchors via missing architecture_class."""

    calls: list[dict[str, object]] = []
    fake_continual = types.ModuleType("tac.continual_learning")

    def fake_update(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return SimpleNamespace(
            accepted=True,
            architecture_class=trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
            axis="cuda",
            evidence_tag="[contest-CUDA]",
            archive_sha256="a" * 64,
            score_value=0.19,
            posterior_n_anchors_after=7,
            refusal_reason="",
        )

    fake_continual.posterior_update_locked_from_auth_eval_json = fake_update
    monkeypatch.setitem(sys.modules, "tac.continual_learning", fake_continual)

    stages: list[str] = []
    auth_eval_json = tmp_path / "auth_eval.json"
    result = trainer_module._run_posterior_update_from_auth_eval_json(
        auth_eval_json,
        architecture_class=trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
        stage_callback=stages.append,
    )

    assert result["status"] == "accepted"
    assert result["accepted"] is True
    assert result["architecture_class"] == trainer_module.POSTERIOR_ARCHITECTURE_CLASS
    assert result["score_claim_promoted"] is False
    assert calls == [
        {
            "path": auth_eval_json,
            "architecture_class": trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
            "notes": f"{trainer_module.SUBSTRATE_TAG}_full_auth_eval",
        }
    ]
    assert stages == ["posterior_update_status_accepted"]


def test_posterior_update_records_refusal_status(trainer_module, monkeypatch, tmp_path):
    """Refused posterior updates leave explicit non-promoting artifact status."""

    fake_continual = types.ModuleType("tac.continual_learning")

    def fake_update(_path, **_kwargs):
        return SimpleNamespace(
            accepted=False,
            architecture_class=trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
            axis="cuda",
            evidence_tag="[contest-CUDA]",
            archive_sha256="b" * 64,
            score_value=0.19,
            posterior_n_anchors_after=7,
            refusal_reason="duplicate archive anchor",
        )

    fake_continual.posterior_update_locked_from_auth_eval_json = fake_update
    monkeypatch.setitem(sys.modules, "tac.continual_learning", fake_continual)

    stages: list[str] = []
    result = trainer_module._run_posterior_update_from_auth_eval_json(
        tmp_path / "auth_eval.json",
        architecture_class=trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
        stage_callback=stages.append,
    )

    assert result["status"] == "refused"
    assert result["accepted"] is False
    assert result["refusal_reason"] == "duplicate archive anchor"
    assert result["score_claim_promoted"] is False
    assert stages == ["posterior_update_status_refused"]


def test_posterior_update_records_failure_status(trainer_module, monkeypatch, tmp_path):
    """Posterior-update failures remain non-fatal but leave stage-log evidence."""

    fake_continual = types.ModuleType("tac.continual_learning")

    def fake_update(_path, **_kwargs):
        raise TypeError("boom")

    fake_continual.posterior_update_locked_from_auth_eval_json = fake_update
    monkeypatch.setitem(sys.modules, "tac.continual_learning", fake_continual)

    stages: list[str] = []
    result = trainer_module._run_posterior_update_from_auth_eval_json(
        tmp_path / "auth_eval.json",
        architecture_class=trainer_module.POSTERIOR_ARCHITECTURE_CLASS,
        stage_callback=stages.append,
    )

    assert result["status"] == "error"
    assert result["accepted"] is False
    assert result["architecture_class"] == trainer_module.POSTERIOR_ARCHITECTURE_CLASS
    assert result["error_type"] == "TypeError"
    assert result["score_claim_promoted"] is False
    assert stages == ["posterior_update_status_error_TypeError"]


# ---------------------------------------------------------------------------
# R1 — flag parsing + defaults
# ---------------------------------------------------------------------------


def test_full_cpu_flag_defaults_false(trainer_module):
    """``--full-cpu`` and ``--advisory-cpu-explicitly-waived`` default to False."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    assert args.full_cpu is False
    assert args.advisory_cpu_explicitly_waived is False


def test_max_wall_clock_hours_default(trainer_module):
    """``--max-wall-clock-hours`` defaults to 12.0 (Carmack-pessimistic bound)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    assert args.max_wall_clock_hours == 12.0


def test_full_cpu_flag_set(trainer_module):
    """``--full-cpu`` flag parses to True when passed."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    assert args.full_cpu is True
    assert args.advisory_cpu_explicitly_waived is True
    assert args.device == "cpu"


def test_max_wall_clock_hours_custom(trainer_module):
    """``--max-wall-clock-hours`` accepts a custom value (e.g. 6.0)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--max-wall-clock-hours", "6.0",
    ])
    assert args.max_wall_clock_hours == 6.0


# ---------------------------------------------------------------------------
# R1 — waiver requirement (the canonical bug class this landing extincts)
# ---------------------------------------------------------------------------


def test_full_cpu_without_waiver_raises(trainer_module):
    """``--full-cpu`` without ``--advisory-cpu-explicitly-waived`` is refused."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "advisory-cpu-explicitly-waived" in str(exc_info.value)


def test_full_cpu_with_waiver_passes(trainer_module):
    """``--full-cpu`` with the waiver passes validation."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)


def test_dangling_waiver_without_full_cpu_raises(trainer_module):
    """``--advisory-cpu-explicitly-waived`` without ``--full-cpu`` is refused.

    Per CLAUDE.md Catalog #133 + #136 pattern (no broad accept tokens): a
    dangling waiver flag could silently waive a later, unrelated check.
    """
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--advisory-cpu-explicitly-waived",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "Dangling waiver flag" in str(exc_info.value)


def test_full_cpu_and_smoke_mutually_exclusive(trainer_module):
    """``--full-cpu`` + ``--smoke`` is refused (distinct modes)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
        "--smoke",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "mutually exclusive" in str(exc_info.value)


def test_full_cpu_with_device_cuda_raises(trainer_module):
    """``--full-cpu`` with ``--device cuda`` is refused (contradiction)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cuda", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "--device cpu" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R2 — banner emission
# ---------------------------------------------------------------------------


def test_full_cpu_banner_emitted_to_stderr(trainer_module, capsys):
    """``_full_cpu_banner`` writes the loud advisory banner to stderr."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert "[macOS-CPU advisory only]" in captured.err
    assert "NON-promotable" in captured.err
    assert 'evidence_grade = "macOS-CPU-advisory"' in captured.err
    assert "score_claim" in captured.err
    assert "promotion_eligible" in captured.err
    assert "ready_for_exact_eval_dispatch" in captured.err


def test_banner_silent_without_full_cpu(trainer_module, capsys):
    """Banner is silent in non-``--full-cpu`` runs."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_banner_includes_max_wall_clock_hours(trainer_module, capsys):
    """Banner surfaces ``--max-wall-clock-hours`` so the operator sees the budget."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
        "--max-wall-clock-hours", "6.5",
    ])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert "6.5" in captured.err


# ---------------------------------------------------------------------------
# R3 — CLI help mentions advisory-only nature
# ---------------------------------------------------------------------------


def test_full_cpu_help_mentions_advisory(trainer_module):
    """``--full-cpu`` help text mentions advisory-only + non-promotable nature."""
    parser = trainer_module._build_parser()
    buf = io.StringIO()
    parser.print_help(buf)
    help_text = buf.getvalue()
    assert "--full-cpu" in help_text
    assert "advisory" in help_text.lower()
    assert "non-promotable" in help_text.lower()
    assert "advisory-cpu-explicitly-waived" in help_text.lower()


def test_full_cpu_help_mentions_wall_clock_range(trainer_module):
    """``--full-cpu`` help mentions the 2-12h wall-clock expectation."""
    parser = trainer_module._build_parser()
    buf = io.StringIO()
    parser.print_help(buf)
    help_text = buf.getvalue()
    assert "2-12h" in help_text or ("2-6" in help_text and "12" in help_text)


def test_full_cuda_auth_eval_is_fail_closed_by_canonical_claim_gate() -> None:
    """Full CUDA mode must not return rc=0 after a broken auth-eval subprocess."""

    text = TRAINER_PATH.read_text(encoding="utf-8")
    gate_text = (
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "smoke_auth_eval_gate.py"
    ).read_text(encoding="utf-8")
    assert "capture_output=True" in gate_text
    assert "contest_auth_eval.py failed" in gate_text
    assert "_canon_gate_auth_eval_call(" in text
    assert "_canon_require_contest_cuda_auth_eval_claim(" in text
    assert "auth_eval_cuda_done_valid_claim" in text


# ---------------------------------------------------------------------------
# Defense in depth — smoke + canonical CUDA path are unaffected
# ---------------------------------------------------------------------------


def test_smoke_path_unaffected_by_new_flags(trainer_module):
    """``--smoke`` without ``--full-cpu`` still works (no regression)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--smoke",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)
    assert args.smoke is True
    assert args.full_cpu is False


def test_cuda_path_unaffected_by_new_flags(trainer_module):
    """``--device cuda`` without ``--full-cpu`` still works (no regression)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1", "--device", "cuda",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)
    assert args.device == "cuda"
    assert args.full_cpu is False
