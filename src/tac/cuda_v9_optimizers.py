# SPDX-License-Identifier: MIT
"""Typed AdamW-to-Muon transition runtime for the V9 Torch vehicle.

Scientific values remain owned by ``spec_v9_cgauge``.  This module only maps
those values onto Torch optimizers, preserves the split optimizer state, and
applies the typed transition re-warmup / Muon tail anneal.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


class TorchMuonAdamW:
    """Muon for hidden matrices plus AdamW for every remaining parameter."""

    SCHEMA = "torch_v9_muon_adamw.v1"

    def __init__(
        self,
        muon: Any,
        adamw: Any,
        *,
        start_epoch: int,
        total_epochs: int,
        muon_lr: float,
        adamw_lr: float,
        final_frac: float,
        rewarmup_epochs: int,
        rewarmup_floor: float,
    ) -> None:
        if int(start_epoch) <= 0 or int(total_epochs) < int(start_epoch):
            raise ValueError("invalid Muon stage epoch interval")
        if not 0.0 < float(final_frac) <= 1.0:
            raise ValueError("Muon final fraction must be in (0,1]")
        if int(rewarmup_epochs) < 0 or not 0.0 < float(rewarmup_floor) <= 1.0:
            raise ValueError("invalid transition rewarmup")
        self.muon = muon
        self.adamw = adamw
        self.start_epoch = int(start_epoch)
        self.total_epochs = int(total_epochs)
        self.muon_lr = float(muon_lr)
        self.adamw_lr = float(adamw_lr)
        self.final_frac = float(final_frac)
        self.rewarmup_epochs = int(rewarmup_epochs)
        self.rewarmup_floor = float(rewarmup_floor)
        self.current_epoch = self.start_epoch
        self.set_epoch(self.start_epoch)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return list(self.muon.param_groups) + list(self.adamw.param_groups)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def set_epoch(self, epoch: int) -> dict[str, float]:
        """Apply typed linear re-warmup composed with Muon cosine annealing."""
        epoch = int(epoch)
        if epoch < self.start_epoch:
            raise ValueError("Muon optimizer cannot be scheduled before its transition")
        self.current_epoch = epoch
        offset = epoch - self.start_epoch
        if self.rewarmup_epochs > 0:
            warm = self.rewarmup_floor + (1.0 - self.rewarmup_floor) * min(
                1.0, offset / float(self.rewarmup_epochs)
            )
        else:
            warm = 1.0
        span = max(1, self.total_epochs - self.start_epoch)
        progress = min(1.0, max(0.0, offset / float(span)))
        cosine = self.final_frac + (1.0 - self.final_frac) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        muon_now = self.muon_lr * warm * cosine
        adamw_now = self.adamw_lr * warm
        for group in self.muon.param_groups:
            group["lr"] = muon_now
        for group in self.adamw.param_groups:
            group["lr"] = adamw_now
        return {
            "muon_lr": float(muon_now),
            "adamw_lr": float(adamw_now),
            "rewarmup_multiplier": float(warm),
            "muon_cosine_multiplier": float(cosine),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "config": {
                "start_epoch": self.start_epoch,
                "total_epochs": self.total_epochs,
                "muon_lr": self.muon_lr,
                "adamw_lr": self.adamw_lr,
                "final_frac": self.final_frac,
                "rewarmup_epochs": self.rewarmup_epochs,
                "rewarmup_floor": self.rewarmup_floor,
            },
            "current_epoch": self.current_epoch,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA:
            raise ValueError("checkpoint optimizer is not the typed V9 Muon/AdamW split")
        expected = self.state_dict()["config"]
        if dict(state.get("config", {})) != expected:
            raise ValueError("Muon optimizer checkpoint differs from typed configuration")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.set_epoch(int(state["current_epoch"]))


def build_torch_muon_adamw(
    model: Any,
    flags: Mapping[str, Any],
    *,
    total_epochs: int,
    start_epoch: int,
    outgoing_adamw: Any | None = None,
) -> tuple[TorchMuonAdamW, dict[str, Any]]:
    """Build the real typed split and optionally seed Muon momentum from AdamW."""
    import torch

    from tac.cuda_levelset_training import parameter_groups

    groups = parameter_groups(model)
    muon_params = list(groups["muon"])
    adamw_params = list(groups["adam"] + groups["pose"] + groups["code"])
    if not muon_params:
        raise ValueError("V9 Muon transition has no eligible hidden matrix parameters")
    if not adamw_params:
        raise ValueError("V9 Muon transition has no AdamW fallback parameters")
    muon = torch.optim.Muon(
        muon_params,
        lr=float(flags["--muon-lr"]),
        momentum=float(flags["--muon-momentum"]),
        ns_steps=int(flags["--muon-ns-steps"]),
        weight_decay=float(flags["--weight-decay"]),
    )
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=float(flags["--muon-adamw-lr"]),
        betas=(0.9, float(flags["--adam-beta2"])),
        weight_decay=float(flags["--weight-decay"]),
    )
    warm_seeded = 0
    if bool(flags.get("--muon-warm-start-momentum", False)) and outgoing_adamw is not None:
        for parameter in muon_params:
            old = outgoing_adamw.state.get(parameter, {})
            first_moment = old.get("exp_avg")
            if first_moment is not None and tuple(first_moment.shape) == tuple(parameter.shape):
                muon.state[parameter]["momentum_buffer"] = first_moment.detach().clone()
                warm_seeded += 1
    runtime = TorchMuonAdamW(
        muon,
        adamw,
        start_epoch=int(start_epoch),
        total_epochs=int(total_epochs),
        muon_lr=float(flags["--muon-lr"]),
        adamw_lr=float(flags["--muon-adamw-lr"]),
        final_frac=float(flags["--muon-lr-final-frac"]),
        rewarmup_epochs=int(flags["--stage-transition-rewarmup-epochs"]),
        rewarmup_floor=float(flags["--stage-transition-rewarmup-floor"]),
    )
    return runtime, {
        "n_muon_params": len(muon_params),
        "n_adamw_params": len(adamw_params),
        "muon_warm_seeded_leaves": int(warm_seeded),
    }


__all__ = ["TorchMuonAdamW", "build_torch_muon_adamw"]
