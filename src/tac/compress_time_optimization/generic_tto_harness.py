# SPDX-License-Identifier: MIT
"""GenericTTOHarness — generalize ``optimize_poses.py`` template to per-byte/
per-stream/per-pair generic TTO (test-time optimization) at compress time.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§G compress-time row 1:

  | TTO at compress time | pose only (PD-V2) | TTO on every learnable
  parameter (selector indices, palette entries, latent scales, mask CRF
  schedule) with arbitrarily many iterations | extend
  ``optimize_poses.py`` template into generic TTO harness |

The harness is the CANONICAL TEMPLATE for generic gradient-descent
optimization at compress time. It does NOT import ``optimize_poses.py``
(that file is a 2674-line CLI entry point with heavy state — pyav decode,
mask loading, scorer wiring). Instead this harness presents the abstract
loop and substrate-specific TTO passes use the harness with a custom
loss_callable.

The harness supports four target kinds (per PV-2):
  - ``parameter_tensor`` — optimize a learnable tensor (e.g. pose params)
  - ``byte_grid``        — optimize per-byte values (e.g. quant codes)
  - ``selector_indices`` — optimize K-mode selector indices (e.g. fec6)
  - ``palette``          — optimize palette entries (e.g. K=16 palette)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": this harness encodes the
substrate-agnostic LOOP. Substrate-specific GenericTTO passes provide the
loss_callable + target_kind + initial parameters. The loss_callable IS the
unique-per-method engineering surface; the harness IS the canonical
infrastructure.

Per the spec §5.5: ``deterministic=True`` required (the harness's seed is
pinned via the contract); ``scorer_free=True`` (the harness does NOT load
PoseNet/SegNet at compress — the loss_callable is a closure over substrate-
specific scorers that are already loaded at compress time per the
substrate trainer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.compress_time_optimization.contract import (
    CompressTimePassContract,
)

__all__ = [
    "GenericTTOHarness",
    "GenericTTOHarnessSpec",
    "LEGAL_TTO_TARGET_KIND",
]

LEGAL_TTO_TARGET_KIND: frozenset[str] = frozenset(
    {
        "parameter_tensor",  # e.g. pose params, latent vectors
        "byte_grid",         # e.g. quant codes, archive byte slots
        "selector_indices",  # e.g. fec6 K-mode selector
        "palette",           # e.g. K=16 palette entries
    }
)

LEGAL_TTO_OPTIMIZER: frozenset[str] = frozenset(
    {
        "adam",
        "adamw",
        "sgd",
        "muon",
    }
)


@dataclass(frozen=True)
class GenericTTOHarnessSpec:
    """Specification for a single generic TTO refinement pass.

    Frozen so spec composition is structurally immutable. The builder
    ``GenericTTOHarness`` converts this spec into a CompressTimePassContract
    that the substrate-specific TTO function then decorates with
    ``@compress_time_pass``.
    """

    pass_id: str
    target_kind: str  # one of LEGAL_TTO_TARGET_KIND
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    num_steps: int = 100
    seed: int = 42
    max_wallclock_seconds: int | None = None
    sensitivity_weighted: bool = False
    correction_resolution: str = "per_pair"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if self.target_kind not in LEGAL_TTO_TARGET_KIND:
            raise ValueError(
                f"target_kind={self.target_kind!r} not in "
                f"{sorted(LEGAL_TTO_TARGET_KIND)}"
            )
        if self.optimizer not in LEGAL_TTO_OPTIMIZER:
            raise ValueError(
                f"optimizer={self.optimizer!r} not in "
                f"{sorted(LEGAL_TTO_OPTIMIZER)}"
            )
        if self.num_steps < 1:
            raise ValueError(f"num_steps={self.num_steps} must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate={self.learning_rate} must be > 0"
            )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class GenericTTOHarness:
    """Builder for a generic TTO refinement pass contract.

    Usage::

        from tac.compress_time_optimization import (
            GenericTTOHarness, GenericTTOHarnessSpec, compress_time_pass,
        )

        spec = GenericTTOHarnessSpec(
            pass_id="tto_pose_per_pair_refinement",
            target_kind="parameter_tensor",
            optimizer="adamw",
            learning_rate=1e-3,
            num_steps=200,
            seed=42,
            sensitivity_weighted=False,
            correction_resolution="per_pair",
            description="Per-pair pose TTO at compress time (PD-V2 pattern).",
            lane_id="lane_my_substrate_20260601",
        )
        contract = GenericTTOHarness(spec=spec).build_contract()

        @compress_time_pass(contract)
        def tto_pose_per_pair_refinement(state, *, policy, seed):
            # Substrate-specific TTO inner loop: gradient descent on
            # parameter tensor with substrate's loss_callable.
            ...
            return {"archive_bytes_v1": ..., "bytes_added": delta_bytes}

    The builder does NOT execute the TTO loop — it produces the CONTRACT.
    The decorated function IS the substrate-specific loop. This separation
    is the canonical "infrastructure vs engineering" split per CLAUDE.md
    HNeRV parity discipline L7.
    """

    def __init__(self, *, spec: GenericTTOHarnessSpec) -> None:
        if not isinstance(spec, GenericTTOHarnessSpec):
            raise TypeError(
                f"spec must be GenericTTOHarnessSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> CompressTimePassContract:
        """Build the CompressTimePassContract for this TTO pass.

        Emits the canonical pattern:
          - stage_phase="compress"
          - correction_kind="refinement"
          - deterministic=True (seed pinned on contract)
          - scorer_free=True (the substrate's loss_callable closes over
            already-loaded scorers; the contract level is scorer-free)

        The contract's ``consumes`` set is intentionally MINIMAL — the
        substrate-specific decorated function may consume more keys at run
        time via the state dict. Declaring narrow consumes here lets the
        pipeline composer reason about minimal data dependencies.
        """
        consumes: frozenset[str] = frozenset({"archive_bytes_v0"})
        if self.spec.sensitivity_weighted:
            consumes = consumes | frozenset({"master_gradient"})
        return CompressTimePassContract(
            id=self.spec.pass_id,
            parent_pass_id=None,
            stage_phase="compress",
            description=(
                self.spec.description
                or (
                    f"Generic TTO harness over target_kind="
                    f"{self.spec.target_kind!r}; {self.spec.num_steps} steps "
                    f"of {self.spec.optimizer} at lr={self.spec.learning_rate} "
                    f"(seed={self.spec.seed})."
                )
            ),
            consumes=consumes,
            emits=frozenset({"archive_bytes_v1"}),
            correction_kind="refinement",
            correction_resolution=self.spec.correction_resolution,
            deterministic=True,
            scorer_free=True,
            sensitivity_weighted=self.spec.sensitivity_weighted,
            max_wallclock_seconds=self.spec.max_wallclock_seconds,
            seed=self.spec.seed,
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution=(
                "master_gradient_v1"
                if self.spec.sensitivity_weighted
                else "not_applicable_with_rationale"
            ),
            hook_pareto_constraint="rate_distortion_v1",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "compress_time_optimization_pass_outcomes_v1"
            ),
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_probe_disambiguator": (
                    "GenericTTOHarness is a canonical loop primitive; the "
                    "substrate's loss_callable IS the unique-per-method "
                    "interpretation. Disambiguation lives at the loss-callable "
                    "design surface, not here."
                ),
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Pass does not consume master_gradient; operator "
                            "opted for uniform per-target gradient descent."
                        )
                    }
                ),
                "hook_bit_allocator_class": (
                    "TTO refinement does not directly allocate bits; the "
                    "downstream codec pass owns bit allocation per Catalog "
                    "#272 distinguishing-feature integration contract."
                ),
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the gradient-descent loop infrastructure; "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the loss_callable (which "
                "is substrate-specific and provided by the decorated function)."
            ),
        )
