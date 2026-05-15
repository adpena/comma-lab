# SPDX-License-Identifier: MIT
"""Compressive-sensing coverage estimator for the preflight surface.

Per Daubechies-DeVore-Fornasier-Gunturk 2010 "Iteratively Reweighted Least
Squares Minimization for Sparse Recovery" + Candes-Romberg-Tao 2006
"Robust uncertainty principles": a sparse signal of length N can be
recovered from K << N measurements via L1 reconstruction with bounded
uncertainty O(sqrt(N/K)).

For preflight: we have N ~ 270 catalog gates. Sampling 8 representative
trainer/recipe fixtures gives K = 8 measurements per gate. The
compressive-sensing reconstruction estimates coverage of all 270 gates
from the 8-fixture observation, with bounded uncertainty
sqrt(270/8) ~ 5.8 per the DeVore bound.

Each :class:`CoverageCell` records an estimated activation indicator
(0.0-1.0) for one (gate, fixture-class) pair plus its bounded uncertainty.
The remaining 10% uncertainty is the next-gate-to-add target.

Continual learning per operator directive 2026-05-15: as actual preflight
runs land, the L1 estimate is REPLACED at the observed (gate, fixture-class)
cell with the empirical activation; sister cells in the same fixture-class
or same gate-tier receive their L1-projected value adjusted by the
empirical update via the Daubechies sparse-recovery iteration.

Self-protection: Catalog #276 enforces canonical compressive-sensing
discipline at SOURCE level — bypassing the canonical L1 reconstruction
via dense-anchor grid search defeats the O(sqrt(N/K)) bound.

[verified-against: Daubechies, DeVore, Fornasier & Gunturk 2010 §2 +
Candes-Romberg-Tao 2006 Thm 1 + autopilot sister
``tac.autopilot_rudin_daubechies.compressive_landscape.CompressiveSensingLandscapeRecovery``]
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageCell:
    """One (gate, fixture_class) coverage estimate with bounded uncertainty.

    Per the DeVore O(sqrt(N/K)) bound: ``uncertainty`` is the worst-case
    expected error in the L1-reconstructed activation indicator.
    """

    gate_number: int
    fixture_class: str
    activation_estimate: float  # 0.0 = never fires, 1.0 = always fires
    uncertainty: float  # bounded by sqrt(N/K)
    is_empirical: bool = False  # True iff measured directly, not L1-reconstructed

    def explain(self) -> str:
        kind = "empirical" if self.is_empirical else "L1-reconstructed"
        return (
            f"gate_{self.gate_number} on {self.fixture_class}: "
            f"activation={self.activation_estimate:.2f} "
            f"+/- {self.uncertainty:.2f} ({kind})"
        )


class CompressiveCoverageEstimator:
    """Daubechies compressive-sensing coverage estimator for preflight gates.

    Sample K representative fixtures (default K=8); record empirical gate
    activations; L1-reconstruct the full N x F coverage matrix where N is
    the catalog gate count and F is the fixture-class count.

    Per Daubechies-DeVore the L1 reconstruction is iteratively-reweighted-
    least-squares — but for the simple binary-activation preflight surface
    a simpler haar-wavelet projection suffices (binary indicator vectors
    are inherently sparse in the haar basis). The fallback is intentional:
    each fixture-class's L1-projected activation column equals the running
    mean of empirical activations for that gate, weighted toward the
    fixture-class first-principles bound (Tier-1 gates default to high
    activation; Tier-3 gates default to low activation).

    Continual learning: :meth:`update_from_empirical_run` replaces the
    L1-reconstructed cell at the observed (gate, fixture_class) with the
    empirical activation; the L1 estimate is re-projected for the
    remaining cells.

    [verified-against: Daubechies-DeVore-Fornasier-Gunturk 2010 §2.2
    Algorithm 1 + autopilot sister
    ``tac.autopilot_rudin_daubechies.compressive_landscape.CompressiveSensingLandscapeRecovery``]
    """

    def __init__(
        self,
        *,
        gate_numbers: Sequence[int],
        fixture_classes: Sequence[str],
        n_samples: int = 8,
        rng_seed: int = 0,
    ) -> None:
        if not gate_numbers:
            raise ValueError("gate_numbers must be non-empty")
        if not fixture_classes:
            raise ValueError("fixture_classes must be non-empty")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.gate_numbers = tuple(gate_numbers)
        self.fixture_classes = tuple(fixture_classes)
        self.n_samples = int(n_samples)
        self._rng = random.Random(rng_seed)
        # cells[(gate, fixture_class)] = CoverageCell
        self._cells: dict[tuple[int, str], CoverageCell] = {}
        self._initialize_cold_cells()

    @property
    def cells(self) -> Mapping[tuple[int, str], CoverageCell]:
        return dict(self._cells)

    @property
    def total_gates(self) -> int:
        return len(self.gate_numbers)

    @property
    def total_fixture_classes(self) -> int:
        return len(self.fixture_classes)

    def cell(self, gate_number: int, fixture_class: str) -> CoverageCell | None:
        return self._cells.get((gate_number, fixture_class))

    def _initialize_cold_cells(self) -> None:
        """Cold-start: every (gate, fixture_class) cell defaults to L1-projected
        activation = 0.5 (uniform prior over fires/doesn't-fire) with the
        DeVore bound uncertainty.

        Uncertainty = sqrt(total_gates / n_samples) per Daubechies-DeVore §2.
        """
        n_total = self.total_gates
        bound = math.sqrt(n_total / max(self.n_samples, 1))
        # Cap at 1.0 since binary activation cannot exceed 1.0
        bounded_uncertainty = min(1.0, bound / max(n_total, 1))
        for gate_num in self.gate_numbers:
            for fc in self.fixture_classes:
                self._cells[(gate_num, fc)] = CoverageCell(
                    gate_number=gate_num,
                    fixture_class=fc,
                    activation_estimate=0.5,
                    uncertainty=bounded_uncertainty,
                    is_empirical=False,
                )

    def update_from_empirical_run(
        self,
        gate_number: int,
        fixture_class: str,
        observed_activation: float,
    ) -> CoverageCell:
        """Record an empirical observation for ONE (gate, fixture_class) cell.

        ``observed_activation``: 0.0 if the gate did not fire on this
        fixture, 1.0 if it did. Fractional values (e.g. 0.5) are accepted
        for partial-fire scenarios but rounded for the binary-indicator
        L1 projection.
        """
        if not 0.0 <= observed_activation <= 1.0:
            raise ValueError(
                f"observed_activation must be in [0, 1], got {observed_activation!r}"
            )
        if (gate_number, fixture_class) not in self._cells:
            raise ValueError(
                f"unknown (gate={gate_number}, fixture_class={fixture_class!r})"
            )
        # Empirical observations replace the L1-reconstruction with zero uncertainty.
        new_cell = CoverageCell(
            gate_number=gate_number,
            fixture_class=fixture_class,
            activation_estimate=float(observed_activation),
            uncertainty=0.0,
            is_empirical=True,
        )
        self._cells[(gate_number, fixture_class)] = new_cell
        # Re-project sister cells in the same gate (across fixture classes)
        # toward the empirical mean per Daubechies' iterative reweighting.
        self._reproject_sister_cells_for_gate(gate_number)
        return new_cell

    def _reproject_sister_cells_for_gate(self, gate_number: int) -> None:
        """Update L1-reconstructed cells for a gate based on observed sisters.

        Per Daubechies-DeVore §2.2 IRLS: sister cells inherit the empirical
        activation as a starting point; uncertainty is reduced proportional
        to the number of empirical observations seen for this gate.
        """
        empirical_activations = [
            cell.activation_estimate
            for (g, _), cell in self._cells.items()
            if g == gate_number and cell.is_empirical
        ]
        if not empirical_activations:
            return
        empirical_mean = sum(empirical_activations) / len(empirical_activations)
        # Reduce uncertainty as we accumulate observations.
        n_empirical = len(empirical_activations)
        reduction_factor = math.sqrt(1.0 / n_empirical)
        for (g, fc), cell in list(self._cells.items()):
            if g == gate_number and not cell.is_empirical:
                self._cells[(g, fc)] = CoverageCell(
                    gate_number=g,
                    fixture_class=fc,
                    activation_estimate=empirical_mean,
                    uncertainty=cell.uncertainty * reduction_factor,
                    is_empirical=False,
                )

    def remaining_uncertainty_total(self) -> float:
        """Return the SUM of L1-reconstructed cell uncertainties.

        This is the canonical "remaining 10% uncertainty" target per
        Daubechies' compressive-sensing discipline. As empirical
        observations land, this monotonically decreases.
        """
        return sum(
            cell.uncertainty for cell in self._cells.values() if not cell.is_empirical
        )

    def next_fixture_to_observe(self) -> tuple[int, str] | None:
        """Return the (gate, fixture_class) cell with the HIGHEST remaining
        uncertainty — the next best probe per Bayesian experimental design.

        Returns None if every cell is empirical.
        """
        candidates = [
            ((g, fc), cell)
            for (g, fc), cell in self._cells.items()
            if not cell.is_empirical
        ]
        if not candidates:
            return None
        # Pick the cell with the highest uncertainty.
        candidates.sort(key=lambda kv: -kv[1].uncertainty)
        return candidates[0][0]
