# SPDX-License-Identifier: MIT
"""Resumable Torch runtime for V9 CGauge scorer-derived controllers.

This module owns only controller state and measurements derived from tensors the
trainer already computed.  It does not own parser defaults, schedule constants,
optimizer construction, eased-target materialization, or launch policy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
from typing import Any

import numpy as np

from tac.witness_control.birth_completion import (
    BirthCompletionController,
    birth_completion_apply_restore,
    birth_completion_restore_from_cfg,
    birth_completion_state_arrays,
)
from tac.witness_control.event_wirings import (
    EventBackstopGate,
    annulus_plateau_event,
    ladder_arms_complete,
    lane_nucleus_event,
    muon_meat_event,
)
from tac.witness_control.polyak_finisher import PolyakTailAverager
from tac.witness_control.sigma_min_plateau import (
    SigmaMinPlateauConfig,
    SigmaMinPlateauDetector,
    backstop_banked_r1_row,
    resolve_pose_finish_engage,
)
from tac.witness_curriculum.ladder_homotopy import (
    ARM_LANE,
    ARM_MOVABLE,
    homotopy_from_flags,
    perclass_lambda_proxy,
)


@dataclass(frozen=True)
class TorchControllerStep:
    """Latched controller decisions consumed for one exhaustive epoch."""

    muon_start: bool
    muon_on: bool
    lane_band_on: bool
    chroma_on: bool
    pose_finish_on: bool
    pose_banked_r1: bool
    ladder_rungs: dict[str, int]
    birth_multipliers: dict[int, float]
    telemetry: tuple[dict[str, Any], ...]


class TorchProtectedIslandSeed:
    """Separate training-only RGB seed parameter with resumable containment.

    The seed is intentionally not a model child, so it cannot enter the witness
    EMA or deploy state accidentally.  The trainer must add ``residual`` through
    :meth:`optimizer_group` and invoke :meth:`contain_grad_` before every step.
    """

    SCHEMA = "torch_protected_island_seed.v1"

    def __init__(
        self,
        residual: Any,
        mask: Any,
        *,
        mode: str,
        damp: float,
    ) -> None:
        import torch

        residual_t = torch.as_tensor(residual, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        if residual_t.ndim != 4 or residual_t.shape[-1] != 3:
            raise ValueError(
                f"seed residual must be (P,H,W,3), got {tuple(residual_t.shape)}"
            )
        if tuple(mask_t.shape) != tuple(residual_t.shape[:-1]):
            raise ValueError(
                f"seed mask {tuple(mask_t.shape)} != residual support "
                f"{tuple(residual_t.shape[:-1])}"
            )
        if mode not in {"freeze", "damp", "shield"}:
            raise ValueError(f"unsupported containment mode {mode!r}")
        if not 0.0 <= float(damp) <= 1.0:
            raise ValueError("containment damp must be in [0,1]")
        self.residual = torch.nn.Parameter(residual_t.detach().clone())
        self.mask = mask_t.detach().clone()
        self.mode = str(mode)
        self.damp = float(damp)

    def to(self, device: Any) -> "TorchProtectedIslandSeed":
        """Move before optimizer construction; preserves leaf-parameter identity."""
        import torch

        self.residual = torch.nn.Parameter(self.residual.detach().to(device))
        self.mask = self.mask.to(device)
        return self

    def parameters(self) -> tuple[Any, ...]:
        return (self.residual,)

    def optimizer_group(self, *, lr: float) -> dict[str, Any]:
        if float(lr) <= 0.0:
            raise ValueError("seed optimizer lr must be positive")
        return {"params": [self.residual], "lr": float(lr), "name": "protected_island_seed"}

    def compose(self, raw_frame1: Any, pair_indices: Sequence[int] | Any, *, weight: float) -> Any:
        import torch

        idx = torch.as_tensor(pair_indices, device=self.residual.device, dtype=torch.long)
        add = self.residual[idx] * self.mask[idx, ..., None].to(self.residual.dtype)
        return raw_frame1 + float(weight) * add

    def contain_grad_(self) -> None:
        """Apply the typed containment law in-place before the optimizer step."""
        import torch

        grad = self.residual.grad
        if grad is None:
            return
        support = self.mask[..., None]
        if self.mode == "freeze":
            grad.masked_fill_(support, 0.0)
        elif self.mode == "damp":
            grad.copy_(torch.where(support, grad * self.damp, grad))
        else:
            # Gradient descent subtracts grad. A grad with the same sign as the
            # seed shrinks its GT-appearance residual and is therefore destructive.
            sign = torch.sign(self.residual.detach())
            destructive = torch.clamp(grad * sign, min=0.0) * sign
            grad.copy_(torch.where(support, grad - destructive, grad))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "residual": self.residual.detach().cpu().clone(),
            "mask": self.mask.detach().cpu().clone(),
            "mode": self.mode,
            "damp": self.damp,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        import torch

        if state.get("schema") != self.SCHEMA:
            raise ValueError("unsupported protected-island-seed checkpoint schema")
        if str(state.get("mode")) != self.mode or float(state.get("damp")) != self.damp:
            raise ValueError("seed resume containment differs from typed configuration")
        mask = torch.as_tensor(state["mask"], dtype=torch.bool)
        residual = torch.as_tensor(state["residual"], dtype=self.residual.dtype)
        if not torch.equal(self.mask.cpu(), mask.cpu()):
            raise ValueError("seed resume mask differs from typed/GT-derived mask")
        if tuple(residual.shape) != tuple(self.residual.shape):
            raise ValueError("seed resume residual shape differs from configured seed")
        with torch.no_grad():
            self.residual.copy_(residual.to(self.residual.device))


class TorchV9ControllerRuntime:
    """One explicit, additive state owner for the V9 controller cascade.

    Decisions for epoch ``e`` consume only completed scorer rows from epochs
    ``< e``.  The in-progress GPU sensor accumulator is checkpointed too, so a
    mid-epoch resume does not discard an exhaustive-pair prefix.
    """

    SCHEMA = "torch_v9_controller_runtime"
    VERSION = 1
    _GATE_PREFIX = {"muon": "__cv9_muon_", "lane": "__cv9_lane_", "chroma": "__cv9_chroma_"}
    _POSE_PREFIX = "__cv9_pose_"
    _POLYAK_PREFIX = "__cv9_polyak_"

    def __init__(
        self,
        flags: Mapping[str, Any],
        *,
        n_classes: int,
        lane_cls: int,
        movable_cls: int,
    ) -> None:
        self.flags = dict(flags)
        self.n_classes = int(n_classes)
        if self.n_classes <= 0:
            raise ValueError("n_classes must be positive")
        self.lane_cls = int(lane_cls)
        self.movable_cls = int(movable_cls)
        if not 0 <= self.lane_cls < self.n_classes:
            raise ValueError("detected lane_cls is outside configured class range")
        if not 0 <= self.movable_cls < self.n_classes:
            raise ValueError("detected movable_cls is outside configured class range")
        if self.lane_cls == self.movable_cls:
            raise ValueError("detected lane and movable classes must be distinct")
        self.muon_gate = EventBackstopGate(
            "muon",
            "--muon-start-epoch",
            "--muon-start-event",
            self.flags.get("--muon-start-event"),
            int(self.flags["--muon-start-epoch"]),
        )
        self.lane_gate = EventBackstopGate(
            "lane_band",
            "--lane-band-start-epoch",
            "--lane-band-start-event",
            self.flags.get("--lane-band-start-event"),
            int(self.flags["--lane-band-start-epoch"]),
        )
        self.chroma_gate = EventBackstopGate(
            "seg_chroma_boundary",
            "--seg-chroma-boundary-start-epoch",
            "--seg-chroma-boundary-start-event",
            self.flags.get("--seg-chroma-boundary-start-event"),
            int(self.flags["--seg-chroma-boundary-start-epoch"]),
        )
        self.homotopy = homotopy_from_flags(
            movable_r0=float(self.flags["--ladder-movable-r0"]),
            movable_birth_epochs=int(self.flags["--ladder-movable-birth-epochs"]),
            movable_hold_epochs=int(self.flags["--ladder-movable-hold-epochs"]),
            movable_anneal_epochs=int(self.flags["--ladder-movable-anneal-epochs"]),
            movable_lambda_gate=float(self.flags["--ladder-movable-lambda-gate"]),
            lane_r0=float(self.flags["--ladder-lane-r0"]),
            lane_birth_epochs=int(self.flags["--ladder-lane-birth-epochs"]),
            lane_hold_epochs=int(self.flags["--ladder-lane-hold-epochs"]),
            lane_anneal_epochs=int(self.flags["--ladder-lane-anneal-epochs"]),
            lane_lambda_gate=float(self.flags["--ladder-lane-lambda-gate"]),
            gate_softness=float(self.flags["--ladder-gate-softness"]),
            release_coeff=float(self.flags["--ladder-release-coeff"]),
            sigma_eff=float(self.flags["--ladder-sigma-eff"]),
            lane_dash_gate=bool(self.flags["--ladder-lane-dash-gate"]),
            max_step_px=float(self.flags["--ladder-max-step-px"]),
        )
        self.ladder_lambda = {ARM_LANE: float("inf"), ARM_MOVABLE: float("inf")}
        self.ladder_prev = {ARM_LANE: 0.0, ARM_MOVABLE: 0.0}
        self.ladder_rungs = {ARM_LANE: 0, ARM_MOVABLE: 0}
        classes = tuple(
            int(x) for x in str(self.flags["--birth-completion-classes"]).split(",") if x.strip()
        )
        self.birth = BirthCompletionController(
            classes,
            tau_persist=float(self.flags["--birth-completion-tau-persist"]),
            area_band=float(self.flags["--birth-completion-area-band"]),
            ramp_epochs=int(self.flags["--birth-completion-ramp-epochs"]),
            post_level=float(self.flags["--birth-completion-post-level"]),
        )
        self.pose_detector = SigmaMinPlateauDetector(SigmaMinPlateauConfig())
        self.polyak = PolyakTailAverager(
            start_epoch=int(self.flags["--polyak-finisher-start-epoch"]),
            arm=bool(self.flags.get("--polyak-finisher-arm", False)),
        )
        self.dseg_history: list[tuple[int, float]] = []
        self.annulus_history: list[tuple[int, float]] = []
        self.latest_stats: dict[int, dict[str, float | int]] = {}
        self.sigma_rows: list[tuple[int, float]] = []
        self._sensor: dict[str, Any] | None = None
        self._active_epoch: int | None = None
        self._last_completed_epoch: int | None = None
        self._pending_telemetry: list[dict[str, Any]] = []
        self._pose_banked_r1 = False
        self._pose_finish_on = int(self.flags["--pose-finish-start-epoch"]) <= 0

    @property
    def muon_on(self) -> bool:
        return bool(self.muon_gate.fired)

    @property
    def pose_banked_r1(self) -> bool:
        return bool(self._pose_banked_r1)

    def _gate_windows(self) -> list[int]:
        return [
            int(self.flags["--ladder-lane-birth-epochs"])
            + int(self.flags["--ladder-lane-hold-epochs"])
            + int(self.flags["--ladder-lane-anneal-epochs"]),
            int(self.flags["--ladder-movable-birth-epochs"])
            + int(self.flags["--ladder-movable-hold-epochs"])
            + int(self.flags["--ladder-movable-anneal-epochs"]),
        ]

    def _resumed_step(self, epoch: int) -> TorchControllerStep:
        return TorchControllerStep(
            muon_start=False,
            muon_on=self.muon_on,
            lane_band_on=bool(self.lane_gate.fired),
            chroma_on=bool(self.chroma_gate.fired),
            pose_finish_on=self._pose_finish_on,
            pose_banked_r1=self.pose_banked_r1,
            ladder_rungs=dict(self.ladder_rungs),
            birth_multipliers=self.birth.all_multipliers(epoch),
            telemetry=(),
        )

    def begin_epoch(self, epoch: int) -> TorchControllerStep:
        epoch = int(epoch)
        if self._active_epoch is not None:
            if self._active_epoch == epoch:
                return self._resumed_step(epoch)
            raise RuntimeError(
                f"cannot begin epoch {epoch}; scorer sensor for epoch {self._active_epoch} is unfinished"
            )
        if self._last_completed_epoch is not None and epoch <= self._last_completed_epoch:
            raise ValueError("controller epochs must increase monotonically")

        telemetry = self._pending_telemetry
        self._pending_telemetry = []
        sensor_epoch = self.dseg_history[-1][0] if self.dseg_history else None
        meat = muon_meat_event(
            self.dseg_history,
            nucleation_complete=ladder_arms_complete(epoch, self._gate_windows()),
        )
        muon = self.muon_gate.update(
            epoch,
            event_fired=bool(meat["fired"]),
            sensor_data_epoch=sensor_epoch,
        )
        lane_stats = self.latest_stats.get(self.lane_cls, {})
        nucleus = lane_nucleus_event(
            float(lane_stats.get("part_frac", 0.0)),
            float(lane_stats.get("within_flip", 1.0)),
            within_flip_thresh=float(self.flags["--curriculum-nucleus-within-flip"]),
            min_part_frac=float(self.flags["--curriculum-nucleus-min-part-frac"]),
        )
        lane = self.lane_gate.update(
            epoch,
            event_fired=bool(nucleus["fired"]),
            sensor_data_epoch=sensor_epoch,
        )
        annulus = annulus_plateau_event(
            self.annulus_history,
            rel_eps=float(self.flags["--annulus-plateau-rel-eps"]),
            dwell_windows=int(self.flags["--annulus-plateau-dwell-windows"]),
            min_epochs=int(self.flags["--annulus-plateau-min-epochs"]),
        )
        chroma = self.chroma_gate.update(
            epoch,
            event_fired=bool(annulus["fired"]),
            sensor_data_epoch=(self.annulus_history[-1][0] if self.annulus_history else None),
        )
        telemetry.extend(
            step.telemetry for step in (muon, lane, chroma) if step.telemetry is not None
        )

        refresh = max(1, int(self.flags["--ladder-refresh-every"]))
        if epoch == 1 or epoch % refresh == 0:
            for arm in (ARM_LANE, ARM_MOVABLE):
                radius = self.homotopy.step_radius(
                    arm,
                    epoch,
                    self.ladder_lambda[arm],
                    self.ladder_prev[arm],
                    float(self.flags["--ladder-sigma-eff"]),
                )
                self.ladder_prev[arm] = radius
                self.ladder_rungs[arm] = max(0, int(round(radius)))

        pose_start = int(self.flags["--pose-finish-start-epoch"])
        verdict = self.pose_detector.verdict()
        if self._pose_finish_on:
            pose_on, banked = True, False
        elif self._pose_banked_r1:
            pose_on, banked = False, True
        elif pose_start <= 0:
            pose_on, banked = True, False
        else:
            pose_on, banked = resolve_pose_finish_engage(
                cond_fired=self.pose_detector.fired(),
                backstop_hit=epoch >= pose_start,
                degenerate=verdict.should_ship_banked_r1(),
            )
        self._pose_finish_on = bool(self._pose_finish_on or pose_on)
        if banked and not self._pose_banked_r1:
            self._pose_banked_r1 = True
            telemetry.append(
                backstop_banked_r1_row(
                    epoch,
                    backstop_epoch=pose_start,
                    classification=verdict.classification,
                    n_points=verdict.n_points,
                )
            )
        self._active_epoch = epoch
        self._sensor = None
        return TorchControllerStep(
            muon_start=bool(muon.just_fired),
            muon_on=bool(muon.start_reached),
            lane_band_on=bool(lane.start_reached),
            chroma_on=bool(chroma.start_reached),
            pose_finish_on=self._pose_finish_on,
            pose_banked_r1=self.pose_banked_r1,
            ladder_rungs=dict(self.ladder_rungs),
            birth_multipliers=self.birth.all_multipliers(epoch),
            telemetry=tuple(telemetry),
        )

    def observe_scorer_chunk(self, pred: Any, target: Any, margins: Any) -> None:
        """Accumulate exact argmax counts on device; no extra scorer forward."""
        import torch

        if self._active_epoch is None:
            raise RuntimeError("begin_epoch must precede scorer observations")
        p = pred.detach().reshape(-1).long()
        t = target.detach().reshape(-1).long()
        margin = margins.detach().reshape(-1)
        if p.numel() != t.numel() or p.numel() != margin.numel():
            raise ValueError("pred, target, and margins must contain the same number of pixels")
        if p.numel() == 0:
            raise ValueError("scorer sensor chunk cannot be empty")
        wrong = p != t
        ann = margin < float(self.flags["--annulus-band"])
        pred_counts = torch.bincount(p, minlength=self.n_classes)
        gt_counts = torch.bincount(t, minlength=self.n_classes)
        if pred_counts.numel() != self.n_classes or gt_counts.numel() != self.n_classes:
            raise ValueError("scorer class index exceeds configured n_classes")
        row = {
            "total": torch.as_tensor(p.numel(), device=t.device, dtype=torch.long),
            "wrong": wrong.sum(dtype=torch.long),
            "ann_total": ann.sum(dtype=torch.long),
            "ann_wrong": (ann & wrong).sum(dtype=torch.long),
            "pred": pred_counts,
            "gt": gt_counts,
            "wrong_gt": torch.bincount(t[wrong], minlength=self.n_classes),
        }
        if self._sensor is None:
            self._sensor = row
            return
        if self._sensor["total"].device != row["total"].device:
            self._sensor = {key: value.to(row["total"].device) for key, value in self._sensor.items()}
        self._sensor = {key: self._sensor[key] + value for key, value in row.items()}

    def end_epoch(self, epoch: int) -> dict[str, Any]:
        import torch

        epoch = int(epoch)
        if self._active_epoch != epoch:
            raise RuntimeError(f"ending epoch {epoch} but active controller epoch is {self._active_epoch}")
        if self._sensor is None:
            raise RuntimeError("controller epoch ended without scorer-derived sensor data")
        # One device synchronization/transfer for all scalar and per-class counters.
        packet = torch.cat(
            [
                self._sensor["total"].reshape(1),
                self._sensor["wrong"].reshape(1),
                self._sensor["ann_total"].reshape(1),
                self._sensor["ann_wrong"].reshape(1),
                self._sensor["pred"].reshape(-1),
                self._sensor["gt"].reshape(-1),
                self._sensor["wrong_gt"].reshape(-1),
            ]
        ).detach().cpu().numpy()
        total, wrong, ann_total, ann_wrong = (int(x) for x in packet[:4])
        pred = packet[4 : 4 + self.n_classes]
        gt = packet[4 + self.n_classes : 4 + 2 * self.n_classes]
        wrong_gt = packet[4 + 2 * self.n_classes :]
        dseg = wrong / max(total, 1)
        annulus_frac = ann_wrong / max(ann_total, 1)
        self.dseg_history.append((epoch, float(dseg)))
        self.annulus_history.append((epoch, float(annulus_frac)))
        total_flips = max(int(wrong_gt.sum()), 1)
        stats: dict[int, dict[str, float | int]] = {}
        for cls in range(self.n_classes):
            stats[cls] = {
                "part_frac": float(pred[cls]) / max(total, 1),
                "within_flip": float(wrong_gt[cls]) / max(int(gt[cls]), 1),
                "gt_px": int(gt[cls]),
                "pred_px": int(pred[cls]),
                "total_px": total,
                "gt_area": float(gt[cls]) / max(total, 1),
            }
        self.latest_stats = stats
        birth_rows = self.birth.observe(epoch, stats)
        self._pending_telemetry.extend(copy.deepcopy(birth_rows))
        for arm, cls in ((ARM_LANE, self.lane_cls), (ARM_MOVABLE, self.movable_cls)):
            dcls = float(wrong_gt[cls]) / max(int(gt[cls]), 1)
            share = float(wrong_gt[cls]) / total_flips
            self.ladder_lambda[arm] = perclass_lambda_proxy(dcls, share)
        self._sensor = None
        self._active_epoch = None
        self._last_completed_epoch = epoch
        return {
            "epoch": epoch,
            "d_seg": float(dseg),
            "annulus_flip_frac": float(annulus_frac),
            "per_class": copy.deepcopy(stats),
            "birth_telemetry": copy.deepcopy(birth_rows),
            "ladder_lambda": dict(self.ladder_lambda),
        }

    def observe_sigma_min(self, epoch: int, sigma_min: float) -> dict[str, Any]:
        before = self.pose_detector.n_points
        self.pose_detector.observe(epoch, sigma_min)
        if self.pose_detector.n_points > before:
            self.sigma_rows.append((int(epoch), float(sigma_min)))
        self.pose_detector.latch_if_fired(epoch)
        return self.pose_detector.verdict().to_dict()

    def observe_polyak(self, epoch: int, model: Any) -> bool:
        """Copy live state only after the typed start; canonical class enforces it."""
        return self.polyak.observe(
            epoch,
            {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
        )

    def polyak_candidate(self) -> dict[str, Any] | None:
        return self.polyak.mean_fp32()

    @staticmethod
    def _cpu_sensor(sensor: dict[str, Any] | None) -> dict[str, Any] | None:
        if sensor is None:
            return None
        return {key: value.detach().cpu().clone() for key, value in sensor.items()}

    def state_dict(self) -> dict[str, Any]:
        gate_arrays = {
            name: gate.state_arrays(self._GATE_PREFIX[name])
            for name, gate in (
                ("muon", self.muon_gate),
                ("lane", self.lane_gate),
                ("chroma", self.chroma_gate),
            )
        }
        return {
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "n_classes": self.n_classes,
            "detected_classes": {"lane": self.lane_cls, "movable": self.movable_cls},
            "epochs": {
                "active": self._active_epoch,
                "last_completed": self._last_completed_epoch,
            },
            "sensor": self._cpu_sensor(self._sensor),
            "dseg_history": list(self.dseg_history),
            "annulus_history": list(self.annulus_history),
            "latest_stats": copy.deepcopy(self.latest_stats),
            "sigma_rows": list(self.sigma_rows),
            "canonical": {
                "gates": gate_arrays,
                "pose": self.pose_detector.state_arrays(self._POSE_PREFIX),
                "birth": birth_completion_state_arrays(self.birth),
                "polyak": {
                    "scalar": self.polyak.state_arrays(self._POLYAK_PREFIX),
                    "heavy": self.polyak.heavy_state_arrays(self._POLYAK_PREFIX),
                },
            },
            "latches": {
                "muon_on": self.muon_on,
                "muon_fired_epoch": self.muon_gate.fired_epoch,
                "lane_on": bool(self.lane_gate.fired),
                "lane_fired_epoch": self.lane_gate.fired_epoch,
                "chroma_on": bool(self.chroma_gate.fired),
                "chroma_fired_epoch": self.chroma_gate.fired_epoch,
                "pose_fired_epoch": (
                    None if self.pose_detector.fired_epoch < 0 else self.pose_detector.fired_epoch
                ),
                "pose_banked_r1": self._pose_banked_r1,
                "pose_finish_on": self._pose_finish_on,
            },
            "ladder": {
                "lambda": dict(self.ladder_lambda),
                "prev": dict(self.ladder_prev),
                "rungs": dict(self.ladder_rungs),
            },
            "pending_telemetry": copy.deepcopy(self._pending_telemetry),
        }

    @staticmethod
    def _restore_gate(gate: EventBackstopGate, prefix: str, arrays: Mapping[str, Any],
                      latch: Mapping[str, Any], name: str) -> None:
        restored = gate.restore_from_cfg(prefix, dict(arrays))
        expected_on = bool(latch.get(f"{name}_on", False))
        fired_epoch = latch.get(f"{name}_fired_epoch")
        if expected_on and not restored:
            if fired_epoch is None:
                raise ValueError(f"{name} latch is on but its fired epoch is absent")
            step = gate.update(
                int(fired_epoch),
                event_fired=bool(gate.event_mode),
                sensor_data_epoch=None,
            )
            if not step.start_reached:
                raise ValueError(f"could not reconstruct {name} controller latch")
        if bool(gate.fired) != expected_on:
            raise ValueError(f"{name} canonical state disagrees with explicit latch")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA or int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported Torch V9 controller checkpoint schema")
        if int(state.get("n_classes", -1)) != self.n_classes:
            raise ValueError("controller checkpoint n_classes differs from typed configuration")
        detected = state.get("detected_classes", {})
        if (
            int(detected.get("lane", -1)) != self.lane_cls
            or int(detected.get("movable", -1)) != self.movable_cls
        ):
            raise ValueError("controller checkpoint detected classes differ from island detection")
        canonical = state.get("canonical", {})
        latches = state.get("latches", {})
        gates = canonical.get("gates", {})
        for name, gate in (
            ("muon", self.muon_gate),
            ("lane", self.lane_gate),
            ("chroma", self.chroma_gate),
        ):
            self._restore_gate(
                gate,
                self._GATE_PREFIX[name],
                gates.get(name, {}),
                latches,
                name,
            )
        self.pose_detector.restore_from_cfg(
            self._POSE_PREFIX, dict(canonical.get("pose", {}))
        )
        pose_fired_epoch = latches.get("pose_fired_epoch")
        expected_pose = None if pose_fired_epoch is None else int(pose_fired_epoch)
        actual_pose = None if self.pose_detector.fired_epoch < 0 else self.pose_detector.fired_epoch
        if actual_pose != expected_pose:
            raise ValueError("pose canonical state disagrees with explicit latch")
        restored_birth = birth_completion_restore_from_cfg(canonical.get("birth", {}))
        if restored_birth is not None:
            configured = (
                tuple(self.birth.classes),
                float(self.birth.tau_persist),
                float(self.birth.area_band),
                int(self.birth.ramp_epochs),
                float(self.birth.post_level),
            )
            persisted = (
                tuple(restored_birth.classes),
                float(restored_birth.tau_persist),
                float(restored_birth.area_band),
                int(restored_birth.ramp_epochs),
                float(restored_birth.post_level),
            )
            if configured != persisted:
                raise ValueError(
                    "birth-completion checkpoint differs from typed configuration"
                )
            birth_completion_apply_restore(self.birth, canonical.get("birth", {}))
        polyak_state = canonical.get("polyak", {})
        self.polyak.restore_from_cfg(self._POLYAK_PREFIX, dict(polyak_state.get("scalar", {})))
        heavy = polyak_state.get("heavy", {})
        stripped_heavy = {
            key[len(self._POLYAK_PREFIX) :]: value
            for key, value in heavy.items()
            if key.startswith(self._POLYAK_PREFIX)
        }
        self.polyak.restore_heavy(stripped_heavy)

        self.dseg_history = [(int(e), float(v)) for e, v in state.get("dseg_history", [])]
        self.annulus_history = [(int(e), float(v)) for e, v in state.get("annulus_history", [])]
        self.latest_stats = {
            int(cls): copy.deepcopy(row) for cls, row in state.get("latest_stats", {}).items()
        }
        self.sigma_rows = [(int(e), float(v)) for e, v in state.get("sigma_rows", [])]
        ladder = state.get("ladder", {})
        self.ladder_lambda.update({str(k): float(v) for k, v in ladder.get("lambda", {}).items()})
        self.ladder_prev.update({str(k): float(v) for k, v in ladder.get("prev", {}).items()})
        self.ladder_rungs.update({str(k): int(v) for k, v in ladder.get("rungs", {}).items()})
        epochs = state.get("epochs", {})
        active = epochs.get("active")
        completed = epochs.get("last_completed")
        self._active_epoch = None if active is None else int(active)
        self._last_completed_epoch = None if completed is None else int(completed)
        self._sensor = self._cpu_sensor(state.get("sensor"))
        if (self._active_epoch is None) != (self._sensor is None):
            raise ValueError("partial controller epoch requires both active epoch and sensor state")
        self._pending_telemetry = copy.deepcopy(state.get("pending_telemetry", []))
        self._pose_banked_r1 = bool(latches.get("pose_banked_r1", False))
        self._pose_finish_on = bool(
            latches.get(
                "pose_finish_on", int(self.flags["--pose-finish-start-epoch"]) <= 0
            )
        )
        if self._pose_banked_r1 and self._pose_finish_on:
            raise ValueError("pose checkpoint cannot be both banked-R1 and finish-engaged")


def torch_pose_jacobian_conditioning(
    pose_from_frames: Any,
    carrier: Any,
    source_nhwc: Any,
    frame1_nhwc: Any,
    pair_indices: Sequence[int],
    *,
    sigma_floor: float,
) -> dict[str, Any]:
    """Measure real six-by-six ``J_xi`` conditioning without optimizer mutation.

    Pair selection and cadence remain trainer-owned because the typed DSL owns
    ``eval_every * jacobian_basin_every`` and motion stratification.
    """
    import torch

    from tac.witness_control.jacobian_basin import aggregate_conditioning, conditioning

    idxs = [int(x) for x in pair_indices]
    if len(idxs) != int(source_nhwc.shape[0]) or len(idxs) != int(frame1_nhwc.shape[0]):
        raise ValueError("pair_indices, source frames, and frame1 frames must have equal batch size")
    conds: list[dict[str, Any]] = []
    with torch.enable_grad():
        for offset, pair_index in enumerate(idxs):
            source = source_nhwc[offset : offset + 1].detach()
            frame1 = frame1_nhwc[offset : offset + 1].detach()
            index = torch.tensor([pair_index], device=source.device, dtype=torch.long)
            xi0 = carrier.xi_effective(index).detach().reshape(6).requires_grad_(True)

            def fn(xi: Any) -> Any:
                frame0 = carrier.forward_with_xi(source, xi.reshape(1, 6))
                return pose_from_frames(frame0, frame1).reshape(6)

            jacobian = torch.autograd.functional.jacobian(
                fn, xi0, create_graph=False, vectorize=True
            )
            conds.append(conditioning(jacobian.detach().cpu().numpy()))
    aggregate = aggregate_conditioning(conds, sigma_floor=float(sigma_floor))
    return {**aggregate, "pair_indices": idxs}


__all__ = [
    "TorchControllerStep",
    "TorchProtectedIslandSeed",
    "TorchV9ControllerRuntime",
    "torch_pose_jacobian_conditioning",
]
