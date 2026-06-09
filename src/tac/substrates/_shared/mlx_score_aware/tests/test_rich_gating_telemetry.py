# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the B1 clean-relaunch RICH GATING telemetry (BLOCKER 1).

The killed off-spec B1 pilot emitted only epoch/loss/lr/ema_drift, so its
stage-1 divergence took ~20 shell probes to diagnose. The clean PR95 baseline
MUST emit per-epoch gating telemetry (stage geometry + optimizer-partition kind
+ per-step grad-norm/clip + nan/inf + loss decomposition + proxy axes) so a
diverging run is diagnosable from a single telemetry tail.

Slot EEE NO FAKE discipline (Class 2 — tests verify BEHAVIOR not constants):

* the REAL ``MlxScoreAwareAdapter.rich_gating_telemetry`` reflects the ACTUAL
  PR95 8-stage geometry per epoch (muon_active=False in stage 1, True in stage
  8; loss_form maps to the canonical per-stage loss family). A version that
  returns static constants FAILS ``test_rich_gating_stage_index_tracks_epoch``
  and ``test_rich_gating_muon_flips_only_at_stage8``.
* ``_assemble_gating_row`` augments with the per-axis proxy decomposition and an
  ARITHMETICALLY CONSISTENT ``per_axis_sum`` + proxy contest score
  (``test_assemble_gating_row_per_axis_sum_is_arithmetic`` recomputes both
  independently and would fail if the emitted numbers did not add up).
* ``run_long_training`` writes the gating fields to the telemetry JSONL at the
  row TOP level (``test_run_long_training_emits_gating_fields_to_jsonl``).
* adapters WITHOUT ``rich_gating_telemetry`` keep ``gating=None`` and emit NO
  new fields (byte-stable; ``test_gating_none_when_adapter_lacks_accessor``).

[verified-against: tac.training.long_training_canonical.PerEpochMetrics +
 _assemble_gating_row + run_long_training epoch loop]
[verified-against: tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter.rich_gating_telemetry]
[verified-against: tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum.PR95FaithfulCurriculumFactory]
"""
from __future__ import annotations

import json
import math
import shutil
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tac.training.long_training_canonical import (
    CurriculumStage,
    LongTrainingConfig,
    PerEpochMetrics,
    _assemble_gating_row,
    _contest_score_from_axes,
    run_long_training,
)

# run_long_training refuses /tmp-class paths (CLAUDE.md "Forbidden /tmp paths"),
# and pytest's ``tmp_path`` resolves under /private/tmp. Use an isolated dir
# under the canonical ephemeral-local-scratch location ``.omx/tmp/`` instead.
_REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def omx_tmp_dir() -> Iterator[Path]:
    """Isolated non-/tmp scratch dir for run_long_training output."""
    base = _REPO_ROOT / ".omx" / "tmp" / "test_rich_gating"
    run_dir = base / f"run_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield run_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Layer 0: PerEpochMetrics.as_dict flattening (unit; no MLX)
# ---------------------------------------------------------------------------


def test_per_epoch_metrics_gating_defaults_none() -> None:
    """Without gating, the row carries NO gating fields (byte-stable)."""
    row = PerEpochMetrics(epoch=0, stage_name="s", loss=1.0).as_dict()
    assert "gating" not in row
    for absent in (
        "stage_index",
        "loss_form",
        "muon_active",
        "grad_norm",
        "grad_clip_applied",
        "nan_inf_count",
        "sidecar_exported",
        "pay_rent_gate_active",
    ):
        assert absent not in row, f"{absent} must be absent when gating is None"


def test_per_epoch_metrics_gating_flattens_to_top_level() -> None:
    """Canonical gating fields surface at the row TOP level for tail-diagnosis."""
    gating = {
        "schema": "mlx_score_aware_rich_gating_telemetry.v1",
        "stage_index": 1,
        "loss_form": "ce_seg_loss",
        "muon_active": False,
        "optimizer_kind_by_group": {"matrix_decoder_weights": "adamw"},
        "grad_norm": 0.42,
        "grad_clip_applied": False,
        "nan_inf_count": 0,
        "sidecar_exported": False,
        "pay_rent_gate_active": True,
        "loss_total": 18.0,
        "loss_seg": 12.0,
        "loss_pose": 5.0,
        "loss_rate": 1.0,
        "proxy_score": 1234.5,
        "checkpoint_path": None,
    }
    row = PerEpochMetrics(
        epoch=3, stage_name="ce_seg", loss=18.0, gating=gating
    ).as_dict()
    # Nested block preserved AND flattened.
    assert row["gating"]["stage_index"] == 1
    assert row["stage_index"] == 1
    assert row["loss_form"] == "ce_seg_loss"
    assert row["muon_active"] is False
    assert row["grad_norm"] == pytest.approx(0.42)
    assert row["grad_clip_applied"] is False
    assert row["nan_inf_count"] == 0
    assert row["sidecar_exported"] is False
    assert row["pay_rent_gate_active"] is True
    assert row["proxy_score"] == pytest.approx(1234.5)
    assert row["checkpoint_path"] is None


def test_per_epoch_metrics_rejects_nan_loss() -> None:
    """A NaN loss row is refused at construction (OOM-safe runner contract)."""
    with pytest.raises(ValueError):
        PerEpochMetrics(epoch=0, stage_name="s", loss=float("nan"))


# ---------------------------------------------------------------------------
# Layer 1: _assemble_gating_row arithmetic + proxy score (unit; mock adapter)
# ---------------------------------------------------------------------------


class _FakeRichAdapter:
    """Adapter exposing a deterministic rich_gating_telemetry for unit checks."""

    def __init__(self, *, muon_active: bool = False, grad_norm: float = 0.5) -> None:
        self._muon_active = muon_active
        self._grad_norm = grad_norm

    def rich_gating_telemetry(self, global_epoch: int | None = None) -> Mapping[str, Any]:
        return {
            "schema": "mlx_score_aware_rich_gating_telemetry.v1",
            "stage_index": 8 if self._muon_active else 1,
            "loss_form": "l7_softplus_seg_loss" if self._muon_active else "ce_seg_loss",
            "muon_active": self._muon_active,
            "optimizer_kind_by_group": {
                "matrix_decoder_weights": "muon" if self._muon_active else "adamw",
                "latents": "adamw",
            },
            "grad_norm": self._grad_norm,
            "grad_clip_max_norm": 1.0,
            "grad_clip_applied": self._grad_norm > 1.0,
            "nan_inf_count": 0,
            "sidecar_exported": False,
            "pay_rent_gate_active": True,
        }


def test_assemble_gating_row_none_without_accessor() -> None:
    """An adapter lacking rich_gating_telemetry yields None (byte-stable)."""

    class _Plain:
        pass

    assert (
        _assemble_gating_row(
            adapter=_Plain(), epoch=0, total_loss=1.0, per_axis=None
        )
        is None
    )


def test_assemble_gating_row_per_axis_sum_is_arithmetic() -> None:
    """NO-FAKE: per_axis_sum equals the LITERAL sum; proxy_score the formula."""
    per_axis = {"seg": 1.25, "pose": 2.75, "archive_bytes": 0.0, "recon_aux": 0.03}
    gating = _assemble_gating_row(
        adapter=_FakeRichAdapter(),
        epoch=5,
        total_loss=18.0,
        per_axis=per_axis,
    )
    assert gating is not None
    assert gating["loss_total"] == pytest.approx(18.0)
    assert gating["loss_seg"] == pytest.approx(1.25)
    assert gating["loss_pose"] == pytest.approx(2.75)
    assert gating["loss_rate"] == pytest.approx(0.0)
    assert gating["loss_recon_aux"] == pytest.approx(0.03)
    # Independent recomputation of the literal sum (the sum-check).
    expected_sum = 1.25 + 2.75 + 0.0 + 0.03 + 0.0 + 0.0
    assert gating["per_axis_sum"] == pytest.approx(expected_sum)
    # Proxy axes mirror the per-axis decomposition.
    assert gating["proxy_d_seg"] == pytest.approx(1.25)
    assert gating["proxy_d_pose"] == pytest.approx(2.75)
    assert gating["proxy_rate"] == pytest.approx(0.0)
    # Proxy score independently recomputed from the canonical contest formula.
    expected_score = 100.0 * 1.25 + math.sqrt(10.0 * 2.75) + 25.0 * 0.0 / 37_545_489.0
    assert gating["proxy_score"] == pytest.approx(expected_score)
    assert gating["proxy_score"] == pytest.approx(
        _contest_score_from_axes(seg=1.25, pose=2.75, archive_bytes=0.0)
    )


def test_assemble_gating_row_records_checkpoint_path_and_schedule() -> None:
    """checkpoint_path = last durable path; checkpoint_save_scheduled honest."""
    gating = _assemble_gating_row(
        adapter=_FakeRichAdapter(),
        epoch=249,
        total_loss=5.0,
        per_axis={"seg": 0.1, "pose": 0.2},
        last_checkpoint_path=Path("/ssd/run/checkpoints/epoch000249.meta.json"),
        checkpoint_save_scheduled=True,
    )
    assert gating is not None
    assert gating["checkpoint_path"] == "/ssd/run/checkpoints/epoch000249.meta.json"
    assert gating["checkpoint_save_scheduled"] is True


def test_contest_score_pose_term_is_sqrt_nonlinear() -> None:
    """The pose term is the canonical sqrt(10*pose), not a linear weight."""
    s0 = _contest_score_from_axes(seg=0.0, pose=0.0, archive_bytes=0)
    s1 = _contest_score_from_axes(seg=0.0, pose=0.4, archive_bytes=0)
    assert s0 == pytest.approx(0.0)
    assert s1 == pytest.approx(math.sqrt(10.0 * 0.4))


# ---------------------------------------------------------------------------
# Layer 2: run_long_training writes gating fields to telemetry JSONL
# ---------------------------------------------------------------------------


class _DictStateModel:
    def __init__(self) -> None:
        self._params: dict[str, list[float]] = {"w_0": [0.5], "w_1": [0.6]}

    def state_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._params.items()}

    def load_state_dict(self, state: Mapping[str, list[float]]) -> None:
        for k, v in state.items():
            self._params[k] = list(v)

    def parameters(self):
        return list(self._params.values())


class _GatingRecordingAdapter:
    """Mock adapter whose rich_gating_telemetry reflects the ACTUAL epoch.

    Slot EEE NO FAKE Class 2: ``stage_index`` is a REAL function of the epoch
    (it flips to 8 at epoch >= ``stage8_epoch``), so a version that returns a
    static constant would fail ``test_run_long_training_emits_gating_fields``.
    """

    def __init__(self, *, stage8_epoch: int = 4) -> None:
        self.substrate_id = "gating_test_substrate"
        self.model = _DictStateModel()
        self.step_count = 0
        self._epoch = 0
        self._stage8_epoch = stage8_epoch

    def notify_global_epoch(self, global_epoch: int) -> None:
        self._epoch = int(global_epoch)

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        return {"batch_size": batch_size, "seed": seed}

    def loss_fn(
        self, model: Any, batch: Any, loss_weights: Mapping[str, float]
    ) -> Mapping[str, float]:
        self.step_count += 1
        base = max(0.001, 1.0 / (self.step_count + 1))
        return {"total": base, "recon": base}

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        for _k, v in model._params.items():
            for i in range(len(v)):
                v[i] = v[i] + 1e-6

    def export_state_dict(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.state_dict(), sort_keys=True))

    def export_archive(self, model: Any, output_dir: Path):
        return None

    def score_aware_components(self, model: Any, batch: Any) -> Mapping[str, float]:
        # Real per-axis decomposition values (deterministic, non-constant).
        return {
            "seg": 0.1 + 0.01 * self._epoch,
            "pose": 0.2,
            "archive_bytes": 0.0,
            "recon_aux": 0.0,
        }

    def rich_gating_telemetry(self, global_epoch: int | None = None) -> Mapping[str, Any]:
        epoch = self._epoch if global_epoch is None else int(global_epoch)
        muon = epoch >= self._stage8_epoch
        return {
            "schema": "mlx_score_aware_rich_gating_telemetry.v1",
            "stage_index": 8 if muon else 1,
            "loss_form": "l7_softplus_seg_loss" if muon else "ce_seg_loss",
            "muon_active": muon,
            "optimizer_kind_by_group": {
                "matrix_decoder_weights": "muon" if muon else "adamw",
                "latents": "adamw",
            },
            "grad_norm": 0.3,
            "grad_clip_max_norm": 1.0,
            "grad_clip_applied": False,
            "nan_inf_count": 0,
            "sidecar_exported": False,
            "pay_rent_gate_active": True,
        }


def _config(out_dir: Path, epochs: int = 6) -> LongTrainingConfig:
    return LongTrainingConfig(
        substrate_id="gating_test_substrate",
        lane_id="lane_test_gating_test_substrate_20260609",  # FAKE_LANE_OK:test_fixture_lane_token_not_a_registry_pre_registration
        epochs=epochs,
        batch_pair_indices_per_step=2,
        curriculum_stages=(
            CurriculumStage(name="only_stage", start_epoch=0, end_epoch=epochs),
        ),
        ema_decay=0.9,
        checkpoint_interval_epochs=3,
        early_stopping_patience=epochs + 1,
        learning_rate=1e-3,
        seed=0,
        output_dir=out_dir,
        device="cpu",
        evidence_grade="[advisory only]",
        notes="B1 clean-relaunch rich gating telemetry test fixture",
    )


def _read_telemetry_rows(telemetry_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in telemetry_path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_run_long_training_emits_gating_fields_to_jsonl(omx_tmp_dir: Path) -> None:
    """The telemetry JSONL carries the gating fields at the row top level."""
    adapter = _GatingRecordingAdapter(stage8_epoch=4)
    config = _config(omx_tmp_dir, epochs=6)
    artifact = run_long_training(adapter, config)

    rows = _read_telemetry_rows(artifact.telemetry_path)
    assert len(rows) == 6, f"expected 6 telemetry rows; got {len(rows)}"

    # Every row carries the canonical gating fields at the top level.
    for row in rows:
        assert "gating" in row, "gating block missing from telemetry row"
        for required in (
            "stage_index",
            "loss_form",
            "muon_active",
            "optimizer_kind_by_group",
            "grad_norm",
            "grad_clip_applied",
            "nan_inf_count",
            "sidecar_exported",
            "pay_rent_gate_active",
            "loss_total",
            "loss_seg",
            "loss_pose",
            "loss_rate",
            "proxy_d_seg",
            "proxy_d_pose",
            "proxy_score",
            "checkpoint_path",
        ):
            assert required in row, f"{required} missing from row epoch={row['epoch']}"

    # NO-FAKE Class 2: stage_index ACTUALLY tracks the epoch (flips at ep>=4).
    by_epoch = {row["epoch"]: row for row in rows}
    assert by_epoch[0]["stage_index"] == 1
    assert by_epoch[0]["muon_active"] is False
    assert by_epoch[3]["muon_active"] is False
    assert by_epoch[4]["stage_index"] == 8
    assert by_epoch[4]["muon_active"] is True
    assert by_epoch[5]["muon_active"] is True
    # If rich_gating_telemetry returned a constant, the above two pairs would be
    # identical — the assertion above is the Class-2 behavior guard.
    assert by_epoch[0]["muon_active"] != by_epoch[5]["muon_active"]

    # loss_seg tracks the per-axis decomposition (also non-constant per epoch).
    assert by_epoch[0]["loss_seg"] == pytest.approx(0.1)
    assert by_epoch[5]["loss_seg"] == pytest.approx(0.1 + 0.01 * 5)
    assert by_epoch[0]["loss_seg"] != by_epoch[5]["loss_seg"]


def test_run_long_training_gating_records_last_checkpoint_path(
    omx_tmp_dir: Path,
) -> None:
    """checkpoint_path becomes non-null AFTER the first checkpoint is written."""
    adapter = _GatingRecordingAdapter(stage8_epoch=10)
    config = _config(omx_tmp_dir, epochs=6)  # checkpoint every 3 -> writes at ep2, ep5
    artifact = run_long_training(adapter, config)
    rows = _read_telemetry_rows(artifact.telemetry_path)
    by_epoch = {row["epoch"]: row for row in rows}
    # Before the first periodic write (ep0-ep2) -> None.
    assert by_epoch[0]["checkpoint_path"] is None
    assert by_epoch[2]["checkpoint_path"] is None
    # The ep2 checkpoint (write happens AFTER the ep2 row) surfaces on ep3+.
    assert by_epoch[3]["checkpoint_path"] is not None
    assert "epoch" in by_epoch[3]["checkpoint_path"].lower() or by_epoch[3][
        "checkpoint_path"
    ].endswith(".json")
    # checkpoint_save_scheduled is the honest pre-write cadence fact.
    assert by_epoch[2]["gating"]["checkpoint_save_scheduled"] is True
    assert by_epoch[5]["gating"]["checkpoint_save_scheduled"] is True
    assert by_epoch[0]["gating"]["checkpoint_save_scheduled"] is False


def test_run_long_training_gating_none_when_adapter_lacks_accessor(
    omx_tmp_dir: Path,
) -> None:
    """Adapters without rich_gating_telemetry emit NO gating fields (byte-stable)."""

    class _NoGatingAdapter(_GatingRecordingAdapter):
        # Remove the accessor so the row stays legacy-shaped.
        rich_gating_telemetry = None  # type: ignore[assignment]

    adapter = _NoGatingAdapter()
    config = _config(omx_tmp_dir, epochs=3)
    artifact = run_long_training(adapter, config)
    rows = _read_telemetry_rows(artifact.telemetry_path)
    for row in rows:
        assert "gating" not in row
        assert "stage_index" not in row
        assert "muon_active" not in row


# ---------------------------------------------------------------------------
# Layer 3: REAL MlxScoreAwareAdapter.rich_gating_telemetry (PR95 geometry)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_pr95_adapter():
    """Real MlxScoreAwareAdapter with the PR95 faithful 3000-ep curriculum."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    import mlx.core as mx
    import mlx.nn as mlx_nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

    class TinyRenderer(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = mx.zeros((2, 3))

        def reconstruct_pair(self, batch):
            n = batch.shape[0]
            zeros = mx.zeros((n, 3, 4, 4))
            return zeros, zeros

    bundle = RendererBundle(
        model=TinyRenderer(),
        target_rgb_0=mx.zeros((4, 4, 4, 3)),
        target_rgb_1=mx.zeros((4, 4, 4, 3)),
        num_pairs=4,
        forward_convention="reconstruct_pair_nchw01",
    )
    return MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_rich_gating_pr95",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=3000,
        pr95_muon_policy="faithful_stage8_only",
        grad_clip_max_norm=1.0,
    )


def test_real_adapter_rich_gating_stage1_no_muon(real_pr95_adapter) -> None:
    """REAL PR95 geometry: epoch 0 is stage 1, ce_seg, muon OFF."""
    real_pr95_adapter.notify_global_epoch(0)
    gating = real_pr95_adapter.rich_gating_telemetry(0)
    assert gating["schema"] == "mlx_score_aware_rich_gating_telemetry.v1"
    assert gating["stage_index"] == 1
    assert gating["loss_form"] == "ce_seg_loss"
    assert gating["muon_active"] is False
    assert gating["optimizer_kind_by_group"]["matrix_decoder_weights"] == "adamw"
    assert gating["pr95_faithful_curriculum_enabled"] is True
    assert gating["pr95_muon_policy"] == "faithful_stage8_only"
    assert gating["sidecar_exported"] is False
    assert gating["pay_rent_gate_active"] is True
    assert gating["grad_clip_max_norm"] == pytest.approx(1.0)
    assert gating["authority"] == "macos_mlx_research_signal_false_authority"


def test_real_adapter_rich_gating_muon_flips_only_at_stage8(real_pr95_adapter) -> None:
    """NO-FAKE Class 2: muon_active is False for stages 1-7, True ONLY at stage 8.

    Walk the 3000-ep scaled boundaries via the REAL factory; muon must flip on
    exactly once, at the canonical stage-8 boundary. A static-constant
    implementation cannot satisfy both halves of this assertion.
    """
    factory = real_pr95_adapter._pr95_curriculum_factory
    boundaries = factory.stage_epoch_boundaries
    assert len(boundaries) == 8
    stage8_start = boundaries[-1][1]

    muon_by_stage: dict[int, bool] = {}
    loss_form_by_stage: dict[int, str] = {}
    for stage_index, start_epoch, _end_epoch in boundaries:
        real_pr95_adapter.notify_global_epoch(int(start_epoch))
        gating = real_pr95_adapter.rich_gating_telemetry(int(start_epoch))
        assert gating["stage_index"] == stage_index
        muon_by_stage[stage_index] = bool(gating["muon_active"])
        loss_form_by_stage[stage_index] = str(gating["loss_form"])

    # Stages 1-7: muon OFF. Stage 8: muon ON. Exactly one True.
    for s in range(1, 8):
        assert muon_by_stage[s] is False, f"stage {s} muon must be OFF"
    assert muon_by_stage[8] is True, "stage 8 muon must be ON"
    assert sum(1 for v in muon_by_stage.values() if v) == 1

    # Loss family progression matches PR95 source (ce -> tau -> smooth -> l7).
    assert loss_form_by_stage[1] == "ce_seg_loss"
    assert loss_form_by_stage[2] == "tau_softplus_seg_loss"
    assert loss_form_by_stage[3] == "smooth_disagreement_seg_loss"
    assert loss_form_by_stage[8] == "l7_softplus_seg_loss"

    # Stage 8 is the LAST stage (entered late in the 3000-ep budget).
    assert stage8_start > boundaries[0][2]


def test_real_adapter_rich_gating_grad_norm_after_step(real_pr95_adapter) -> None:
    """grad_norm becomes a real float after a train step appends to history."""
    # Before any step the history is empty -> grad_norm is None (honest).
    gating0 = real_pr95_adapter.rich_gating_telemetry(0)
    assert gating0["grad_norm"] is None

    real_pr95_adapter.notify_global_epoch(0)
    batch = real_pr95_adapter.sample_batch(batch_size=2, seed=0)
    real_pr95_adapter.train_step(
        batch=batch, learning_rate=1e-3, loss_weights={"recon": 1.0}
    )
    gating1 = real_pr95_adapter.rich_gating_telemetry(0)
    assert gating1["grad_norm"] is not None
    assert math.isfinite(float(gating1["grad_norm"]))
    assert gating1["nan_inf_count"] == 0
