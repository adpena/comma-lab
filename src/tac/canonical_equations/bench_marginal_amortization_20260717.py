# SPDX-License-Identifier: MIT
"""bench_marginal_amortization_v1 — the checkpoint-cadence observer amortization law.

THE CONFOUND THIS CODIFIES (MEASURED, 2026-07-16/17; memory
``bench_lever_contaminates_measured_quantity_ckpt_every_confound_20260717``): a bench
lever that changes an OBSERVER'S FIRING CADENCE (``--ckpt-every 1`` for crash-resume
fidelity) contaminates the measured quantity (sec/epoch) whenever a default-ON observer
rides that cadence (the mod-dim ablation probe, ~1,540 s/firing on n600, run
20260716T211713Z ``span_epoch_tail_s``). The bench measured ~1,612 s/ep for a config
whose true amortized pace at the REAL cadence is ~131 s/ep.

The law (all inputs MEASURED, outputs DERIVED):

    S_amortized = S_typical + C_extra / k
    T_remaining = S_amortized * (E_total - (e_resume - 1))

where
    S_typical  = per-epoch wall WITHOUT the cadence-riding observer(s)
                 (MEASURED: median epoch_total_s of observer-OFF bench wallclock rows);
    C_extra    = checkpoint-epoch extra seconds attributed to the named observer(s) via a
                 ONE-KNOB A/B (observer-ON tail minus observer-OFF tail; the knob is
                 ``--no-mod-dim-ablation``) — never a plausibility estimate;
    k          = the REAL config's --ckpt-every (from the emitted launch.sh artifact);
    E_total    = the REAL config's --epochs; e_resume = warm-start resume epoch.

Runtime twin: ``tools/launch_witness_run.py::bench_marginal_decomposition`` (the receipt
producer). This module is the law's canonical home; both are covered by
``src/tac/tests/test_dry_start_delta_bench.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EQUATION_ID = "bench_marginal_amortization_v1"


@dataclass(frozen=True)
class MeasuredSecondsPerEpoch:
    value: float
    source_artifact: str
    status: str = "MEASURED"

    def __post_init__(self) -> None:
        if self.status != "MEASURED":
            raise ValueError("only status=MEASURED is accepted (NO-FAKE)")
        if (not isinstance(self.value, (int, float)) or isinstance(self.value, bool)
                or not math.isfinite(float(self.value)) or float(self.value) < 0.0):
            raise ValueError("value must be finite and non-negative")
        if not isinstance(self.source_artifact, str) or not self.source_artifact.strip():
            raise ValueError("source_artifact is required for measured inputs")


@dataclass(frozen=True)
class BenchMarginalAmortization:
    """One-knob-A/B amortization of a cadence-riding observer's cost."""

    typical_sec_per_ep: MeasuredSecondsPerEpoch
    observer_on_tail_s: MeasuredSecondsPerEpoch
    observer_off_tail_s: MeasuredSecondsPerEpoch
    real_ckpt_every: int
    real_epochs: int
    resume_start_epoch: int = 1
    ab_knob: str = "--no-mod-dim-ablation"
    observers: tuple[str, ...] = ("mod_dim_ablation",)

    def __post_init__(self) -> None:
        if self.real_ckpt_every <= 0:
            raise ValueError("real_ckpt_every must be a positive integer")
        if self.real_epochs <= 0:
            raise ValueError("real_epochs must be a positive integer")
        if not (1 <= self.resume_start_epoch <= self.real_epochs):
            raise ValueError("resume_start_epoch must be in 1..real_epochs")
        if not self.observers or not all(str(o).strip() for o in self.observers):
            raise ValueError("the amortized observer(s) must be NAMED")

    def compose(self) -> dict[str, object]:
        extra = max(float(self.observer_on_tail_s.value)
                    - float(self.observer_off_tail_s.value), 0.0)
        amortized = float(self.typical_sec_per_ep.value) + extra / self.real_ckpt_every
        remaining = self.real_epochs - (self.resume_start_epoch - 1)
        return {
            "equation_id": EQUATION_ID,
            "typical_sec_per_ep": float(self.typical_sec_per_ep.value),
            "ckpt_epoch_extra_s": extra,
            "amortized_sec_per_ep": amortized,
            "projected_remaining_epochs": remaining,
            "projected_remaining_wall_s": amortized * remaining,
            "observers": list(self.observers),
            "ab_knob": self.ab_knob,
            "inputs_status": "MEASURED_ONLY_ONE_KNOB_AB",
            "attribution_rule": ("ckpt_epoch_extra_s is admissible ONLY as the difference "
                                 "of two tail measurements whose sole differing knob is "
                                 f"{self.ab_knob} (attribution-requires-A/B)"),
        }


__all__ = ["EQUATION_ID", "BenchMarginalAmortization", "MeasuredSecondsPerEpoch"]
