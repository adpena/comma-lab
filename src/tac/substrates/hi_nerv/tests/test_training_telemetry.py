"""NO-FAKE behavioral tests for B1 HiNeRV training telemetry.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" forbidden class 2 (tests-verify-
constants-not-behavior): every test here verifies the module actually does the
work it names on REAL inputs. The per-stage records are verified against the
REAL canonical PR95 curriculum factory (not hardcoded constants); the per-epoch
projection is verified against REAL canonical telemetry JSONL rows; the
best-checkpoint manifest is verified against REAL on-disk EMA-shadow bytes with
a REAL SHA-256 (and the test would FAIL if the manifest stopped hashing the
actual file or stopped requiring the EMA shadow).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.substrates.hi_nerv.training_telemetry import (
    MLX_RESEARCH_SIGNAL_AXIS,
    BestCheckpointManifest,
    DecomposedEpochRow,
    Pr95StageRecord,
    build_best_checkpoint_manifest,
    build_pr95_curriculum_stage_records,
    iter_decomposed_epoch_rows,
    read_decomposed_epoch_rows,
    sha256_file,
    summarize_run,
)

mlx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# 1) Per-STAGE L14/L15 rollup — verified against the REAL canonical factory.
# ---------------------------------------------------------------------------


def test_stage_records_match_canonical_factory_29650_behavior() -> None:
    """The 8 stage records must MATCH the live canonical factory verdicts.

    NO-FAKE: this does not assert hardcoded constants; it cross-checks every
    field against an INDEPENDENT call into the canonical factory. If the module
    stopped reading the factory and returned canned rows, the factory's own
    verdict would diverge and this test fails.
    """
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    records = build_pr95_curriculum_stage_records(total_epoch_budget=29650)
    assert len(records) == 8
    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=29650)
    for rec in records:
        v = factory.current_stage_verdict(rec.start_epoch)
        assert rec.descriptor_id == v.descriptor_id
        assert rec.loss_family == v.loss_family
        assert rec.qat_active == bool(v.qat_active)
        assert rec.c1a_lambda == pytest.approx(float(v.cat_lambda))
        assert rec.sigma == pytest.approx(float(v.cat_sigma))
        assert rec.muon_active == bool(v.uses_muon)


def test_stage_records_total_epochs_sum_to_budget() -> None:
    """Epochs-in-stage must sum to the requested budget (real boundary math)."""
    records = build_pr95_curriculum_stage_records(total_epoch_budget=29650)
    assert sum(r.epochs_in_stage for r in records) == 29650


def test_stage_records_muon_only_in_final_stage_faithful_policy() -> None:
    """L15: under faithful_stage8_only, ONLY stage 8 uses Muon (real factory)."""
    records = build_pr95_curriculum_stage_records(
        total_epoch_budget=29650, muon_policy="faithful_stage8_only"
    )
    muon_stages = [r.stage_index for r in records if r.muon_active]
    assert muon_stages == [8]


def test_stage_records_every_stage_policy_enables_muon_everywhere() -> None:
    """every_stage policy must flip Muon on for all stages (real factory branch)."""
    records = build_pr95_curriculum_stage_records(
        total_epoch_budget=29650, muon_policy="every_stage"
    )
    assert all(r.muon_active for r in records)


def test_stage_records_c1a_lambda_and_sigma_schedule_is_real() -> None:
    """L16 lambda 0->0.01->0.02 + L17 sigma 0.2->0.1 must come from the factory.

    NO-FAKE: verifies the actual scheduled values exist in the curriculum (not
    that the module invented them). Stage 1 has lambda 0; a late stage has
    lambda 0.02; some stage drops sigma to 0.1.
    """
    records = build_pr95_curriculum_stage_records(total_epoch_budget=29650)
    by_stage = {r.stage_index: r for r in records}
    assert by_stage[1].c1a_lambda == pytest.approx(0.0)
    assert max(r.c1a_lambda for r in records) == pytest.approx(0.02)
    sigmas = [r.sigma for r in records]
    # L17 schedule: early stages sigma=0.2, late stages drop to 0.1.
    assert any(s == pytest.approx(0.2) for s in sigmas)
    assert any(s == pytest.approx(0.1) for s in sigmas)


def test_stage_records_scaled_budget_still_8_stages() -> None:
    """A scaled (non-29650) budget must still produce 8 stages summing to it."""
    records = build_pr95_curriculum_stage_records(total_epoch_budget=80)
    assert len(records) == 8
    assert sum(r.epochs_in_stage for r in records) == 80
    # final stage still Muon under faithful policy even when scaled.
    assert records[-1].muon_active is True


def test_stage_record_as_dict_roundtrip_fields() -> None:
    """as_dict must expose every L14 field (behavioral contract for the reader)."""
    rec = Pr95StageRecord(
        stage_index=8,
        descriptor_id="pr95_stage8_muon_adamw_mlx",
        loss_family="l7_softplus_seg_loss",
        start_epoch=24650,
        end_epoch=29650,
        epochs_in_stage=5000,
        qat_active=True,
        c1a_lambda=0.02,
        sigma=0.1,
        muon_active=True,
        muon_policy="faithful_stage8_only",
    )
    d = rec.as_dict()
    for key in (
        "stage_index",
        "descriptor_id",
        "loss_family",
        "epochs_in_stage",
        "qat_active",
        "c1a_lambda",
        "sigma",
        "muon_active",
    ):
        assert key in d
    assert d["muon_active"] is True
    assert d["epochs_in_stage"] == 5000


# ---------------------------------------------------------------------------
# 2) Per-epoch decomposition projection — verified against REAL telemetry rows.
# ---------------------------------------------------------------------------


def _real_telemetry_row() -> dict:
    """A REAL canonical per-epoch telemetry row shape (seg/pose/rate present).

    Mirrors the actual fields the canonical long_training_canonical writer
    emits (verified against a live B1 smoke run). Not a toy fixture: the keys
    + nesting match the canonical PerEpochMetrics.as_dict() output exactly.
    """
    return {
        "epoch": 0,
        "stage_name": "compact_runner_pr95_hnerv_mlx_mlx_score_aware_full",
        "loss": 46.19453811645508,
        "learning_rate": 0.001,
        "wall_clock_seconds": 0.46967601776123047,
        "ema_drift_l2": 0.27156783657048783,
        "per_axis_decomposition": {
            "archive_bytes": 0.0,
            "pose": 44.70363998413086,
            "recon_aux": 0.20936709642410278,
            "seg": 1.1415866613388062,
        },
        "loss_components": {
            "gradient_global_norm_pre_clip": 0.42686012387275696,
            "pact_optimizer_uses_muon": 1.0,
            "pact_muon_tensor_count": 11.0,
            "pact_adamw_tensor_count": 18.0,
        },
    }


def test_project_epoch_row_extracts_seg_pose_rate() -> None:
    """The projection must pull seg/pose/rate from per_axis_decomposition."""
    (row,) = tuple(iter_decomposed_epoch_rows([_real_telemetry_row()]))
    assert row.seg == pytest.approx(1.1415866613388062)
    assert row.pose == pytest.approx(44.70363998413086)
    assert row.rate == pytest.approx(0.0)
    assert row.recon_aux == pytest.approx(0.20936709642410278)
    assert row.total == pytest.approx(46.19453811645508)


def test_project_epoch_row_extracts_grad_norm_and_muon() -> None:
    """grad_norm + uses_muon must come from the canonical loss_components keys."""
    (row,) = tuple(iter_decomposed_epoch_rows([_real_telemetry_row()]))
    assert row.grad_norm == pytest.approx(0.42686012387275696)
    assert row.uses_muon is True
    assert row.learning_rate == pytest.approx(0.001)
    assert row.ema_drift_l2 == pytest.approx(0.27156783657048783)


def test_project_epoch_row_handles_missing_per_axis_gracefully() -> None:
    """A row without per_axis_decomposition must project to zeros, not crash."""
    bare = {"epoch": 3, "loss": 10.0, "stage_name": "s"}
    (row,) = tuple(iter_decomposed_epoch_rows([bare]))
    assert row.epoch == 3
    assert row.seg == 0.0 and row.pose == 0.0 and row.rate == 0.0
    assert row.total == pytest.approx(10.0)
    assert row.uses_muon is False


def test_read_decomposed_epoch_rows_from_real_jsonl(tmp_path: Path) -> None:
    """Reading a JSONL file must yield projected rows in file order."""
    jsonl = tmp_path / "telemetry.jsonl"
    rows_in = []
    for i in range(3):
        r = _real_telemetry_row()
        r["epoch"] = i
        r["loss"] = 46.0 - i  # decreasing
        r["per_axis_decomposition"]["pose"] = 44.0 - i
        rows_in.append(r)
    jsonl.write_text("\n".join(json.dumps(r) for r in rows_in) + "\n")
    out = read_decomposed_epoch_rows(jsonl)
    assert [r.epoch for r in out] == [0, 1, 2]
    assert [round(r.total, 1) for r in out] == [46.0, 45.0, 44.0]


def test_read_decomposed_epoch_rows_skips_partial_trailing_line(tmp_path: Path) -> None:
    """A partially-flushed trailing JSON line (live run) must be skipped, not crash."""
    jsonl = tmp_path / "telemetry.jsonl"
    good = json.dumps(_real_telemetry_row())
    jsonl.write_text(good + "\n" + '{"epoch": 1, "loss":')  # truncated
    out = read_decomposed_epoch_rows(jsonl)
    assert len(out) == 1


def test_read_decomposed_epoch_rows_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_decomposed_epoch_rows(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# 3) Best-checkpoint manifest — verified against REAL on-disk bytes + SHA-256.
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """sha256_file must equal an independent hashlib computation (real bytes)."""
    f = tmp_path / "blob.bin"
    payload = b"\x00\x01\x02b1-hinerv-ema-shadow-bytes\xff"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


def test_build_best_checkpoint_manifest_hashes_real_ema_shadow(tmp_path: Path) -> None:
    """The manifest must hash the ACTUAL referenced EMA-shadow file.

    NO-FAKE: writes a real EMA-shadow file + a canonical-shaped meta.json that
    references it, then asserts the manifest's SHA-256 equals the real file's
    hash and the path is the EMA shadow (not the live state).
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ema_file = ckpt_dir / "best_epoch000001.ema_shadow.state.npsd"
    ema_bytes = b"EMA-SHADOW-REAL-BYTES-229K"
    ema_file.write_bytes(ema_bytes)
    live_file = ckpt_dir / "best_epoch000001.live.state.npsd"
    live_file.write_bytes(b"LIVE-WEIGHTS-DIFFERENT")
    meta = {
        "schema_version": "long_training_canonical_checkpoint.v1",
        "checkpoint_role": "best",
        "is_best": True,
        "global_epoch": 1,
        "checkpoint_selection_metric_key": "total",
        "checkpoint_selection_metric_value": 45.49714660644531,
        "checkpoint_selection_metric_mode": "min",
        "ema_shadow_state_path": str(ema_file),
        "live_state_path": str(live_file),
        "loss": 45.49714660644531,
    }
    meta_path = ckpt_dir / "best_epoch000001.meta.json"
    meta_path.write_text(json.dumps(meta))

    manifest = build_best_checkpoint_manifest(
        best_checkpoint_meta_path=meta_path,
        arch_param_count=228903,
        run_id="b1_test_run",
        git_sha="abc1234",
        config_hash="cfg5678",
        seed=7,
    )
    # The manifest hashes the EMA shadow, NOT the live weights.
    assert manifest.ema_shadow_sha256 == hashlib.sha256(ema_bytes).hexdigest()
    assert manifest.ema_shadow_sha256 != hashlib.sha256(b"LIVE-WEIGHTS-DIFFERENT").hexdigest()
    assert manifest.is_ema_shadow is True
    assert manifest.ema_shadow_checkpoint_path == str(ema_file)
    assert manifest.best_epoch == 1
    assert manifest.arch_param_count == 228903
    assert manifest.seed == 7


def test_build_best_checkpoint_manifest_rejects_meta_without_ema_shadow(
    tmp_path: Path,
) -> None:
    """Per EMA non-negotiable: a meta with no EMA-shadow path must be REFUSED.

    NO-FAKE: this is the structural protection that the best inference
    checkpoint is the EMA shadow, never live weights.
    """
    meta_path = tmp_path / "bad.meta.json"
    meta_path.write_text(
        json.dumps({"global_epoch": 1, "live_state_path": "/x/live.npsd"})
    )
    with pytest.raises(ValueError, match="EMA"):
        build_best_checkpoint_manifest(
            best_checkpoint_meta_path=meta_path,
            arch_param_count=1,
            run_id="r",
            git_sha="g",
            config_hash="c",
            seed=0,
        )


def test_best_checkpoint_manifest_carries_cite_tuple_and_nonpromotable() -> None:
    """as_dict must carry the cite-tuple + non-promotable markers (facet 5)."""
    m = BestCheckpointManifest(
        ema_shadow_checkpoint_path="/x/ema.npsd",
        ema_shadow_sha256="0" * 64,
        is_ema_shadow=True,
        best_epoch=5,
        best_stage_name="s8",
        selection_metric_key="total",
        selection_metric_value=1.23,
        selection_metric_mode="min",
        arch_param_count=228903,
        run_id="rid",
        git_sha="gsha",
        config_hash="chash",
        seed=42,
    )
    d = m.as_dict()
    assert d["cite_tuple"] == {
        "run_id": "rid",
        "git_sha": "gsha",
        "config_hash": "chash",
        "seed": 42,
        "arch_param_count": 228903,
    }
    # Non-promotable per Catalog #341.
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["ready_for_exact_eval_dispatch"] is False
    assert d["measurement_axis"] == MLX_RESEARCH_SIGNAL_AXIS
    # The proxy metric must be labeled as proxy, not a score.
    assert "selection_metric_value_proxy" in d
    assert "selection_metric_value" not in d


# ---------------------------------------------------------------------------
# 4) Run summary — verified against a REAL multi-epoch trajectory.
# ---------------------------------------------------------------------------


def test_summarize_run_computes_real_trajectory() -> None:
    """summarize_run must compute first/last/best from the actual rows."""
    rows = [
        DecomposedEpochRow(
            epoch=i,
            stage_name="s",
            seg=2.0 - i * 0.1,
            pose=44.0 - i,
            rate=0.0,
            recon_aux=0.2,
            total=46.0 - i,
            grad_norm=0.4,
            learning_rate=0.001,
            seconds=float(i + 1),
            ema_drift_l2=0.1 * i,
            uses_muon=(i >= 2),
        )
        for i in range(4)
    ]
    s = summarize_run(rows)
    assert s.epochs_observed == 4
    assert s.first_total == pytest.approx(46.0)
    assert s.last_total == pytest.approx(43.0)
    assert s.best_total == pytest.approx(43.0)
    assert s.best_pose == pytest.approx(41.0)
    assert s.muon_epochs_observed == 2  # i=2,3
    assert s.total_wall_clock_seconds == pytest.approx(4.0)
    assert s.mean_seconds_per_epoch == pytest.approx(1.0)
    assert s.loss_is_finite is True


def test_summarize_run_detects_nonfinite_loss() -> None:
    """A NaN/inf total must set loss_is_finite=False (OOM/divergence guard)."""
    rows = [
        DecomposedEpochRow(
            epoch=0,
            stage_name="s",
            seg=1.0,
            pose=1.0,
            rate=0.0,
            recon_aux=0.0,
            total=float("inf"),
            grad_norm=0.0,
            learning_rate=0.001,
            seconds=1.0,
            ema_drift_l2=0.0,
            uses_muon=False,
        )
    ]
    s = summarize_run(rows)
    assert s.loss_is_finite is False


def test_summarize_run_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize_run([])


def test_summary_as_dict_nonpromotable_markers() -> None:
    """The run summary JSON must carry the non-promotable markers (no score claim)."""
    rows = [
        DecomposedEpochRow(
            epoch=0,
            stage_name="s",
            seg=1.0,
            pose=1.0,
            rate=0.0,
            recon_aux=0.0,
            total=2.0,
            grad_norm=0.0,
            learning_rate=0.001,
            seconds=1.0,
            ema_drift_l2=0.0,
            uses_muon=False,
        )
    ]
    d = summarize_run(rows).as_dict()
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["measurement_axis"] == MLX_RESEARCH_SIGNAL_AXIS
