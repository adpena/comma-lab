# SPDX-License-Identifier: MIT
"""A1 pattern generalization — inflate-time bias correction.

Generalizes the empirically-validated A1 pause-and-diagnose exploit to ANY
substrate. The A1 anchor empirics:

A1 baseline:          score 0.192847577437 [contest-CPU GHA Linux x86_64]
A1 V1 (= PR101 bias): score 0.192847577437 [contest-CPU GHA Linux x86_64]
A1 V2 (half-mag):     score 0.194295755690 [contest-CPU GHA Linux x86_64]
A1 V7 (PR101+PR102):  score 0.192930137424 [contest-CPU GHA Linux x86_64]
A1 V0 (no PR101 bias): score 0.195213986806 [contest-CPU GHA Linux x86_64]
A1 V5 (opposite sign): score 0.198381186140 [contest-CPU GHA Linux x86_64]

Source: ``.omx/research/a1_inflate_bias_sweep_exact_cpu_review_20260509_codex
.md``.

The empirical claim: a PAUSED, fully-trained PR95-paradigm substrate can have
its head-output bytes perturbed AT INFLATE TIME (without re-training) and the
resulting archive can have a score within ±0.005 of the original. The PR101
bias block is load-bearing for A1 (removing it regresses by 0.003); reducing
its magnitude regresses by 0.0015; doubling its magnitude regresses by 0.0023.
This is a NEGATIVE-RESULT-AS-FRONTIER-EVIDENCE: A1's PR101 bias is at a local
optimum on the inflate-time bias coordinate.

The Class-shift exploit hypothesis: T4 SYMPOSIUM Priority 1 BOLT-ON-on-A1
lanes (Ballé hyperprior / PR101 entropy stack / VQ-codebook) can each ALSO
have inflate-time bias correction sweeps applied to their respective sidecar
bytes. The cost is ~$0.30 per CPU dispatch on GHA Linux x86_64 (vs $10-25 per
GPU re-train); the EIG is high because the sweep produces 6+ empirical-CPU
anchors per dollar.

`[derived]` claims:
- Inflate-time bias correction does NOT change the trained weights; it
  perturbs the EMITTED archive bytes between substrate output and contest-
  scorer input. The byte-mutation contract is auditable via Catalog #139
  (no-op detector) and Catalog #220 (operational mechanism declaration).

`[empirical:.omx/research/a1_inflate_bias_sweep_exact_cpu_review_20260509_
codex.md]` claims:
- A1 V2 half-magnitude regressed by +0.0015 from V1=baseline; the PR101 bias
  block is empirically load-bearing.
- 4-of-6 variants regressed; 1-of-6 (V1) matched baseline; 1-of-6 (V7
  PR101+PR102 stack) was within 1e-4 of baseline. Sample size 6, 95% CI
  Wilson [0.005, 0.642] for "candidate improves baseline" (data = 0/6).

`[would-need-empirical]` claims:
- Whether this generalizes from A1 to BOLT-ON-on-A1 substrates depends on
  whether the BOLT-ON head/sidecar has any free byte-coordinate that
  preserves the contest scorer's invariance. This is exactly what T4
  SYMPOSIUM Priority 1 Decision 2C operationalizes; A1 pattern generalization
  is the cheap empirical probe BEFORE committing $10-25 per BOLT-ON GPU lane.

Cargo-cult audit per assumption
───────────────────────────────
* "Inflate-time bias correction always preserves apples-to-apples decoded
  parity" — CARGO-CULTED unless the byte mutation is BELOW the renderer's
  numerical-precision threshold. For half-magnitude perturbations: the
  empirical receipt is mixed (A1 V2 measurably changed the score by 0.0015,
  so the bytes DID propagate to decoded output). The "above noise" threshold
  is substrate-specific.
* "PR101 bias correction is universally optimal" — HARD-EARNED for A1 only;
  CARGO-CULTED for any other substrate. Each consuming substrate MUST run
  its own bias-correction sweep before claiming the optimization.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Bias-correction byte-mutation arithmetic → DOCUMENTED FORK (the canonical
  byte-mutation primitive lives in ``tac.packet_compiler``; this helper is
  the substrate-archive sidecar wrapper).
* Sweep grid → UNIQUE per substrate (each substrate decides which byte
  range to sweep over; we expose the grid as caller responsibility).
* Held-out validation → ADOPT canonical
  (``experiments/contest_auth_eval.py`` via ``gate_auth_eval_call`` per
  Catalog #226).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


class InflateBiasCorrectionError(RuntimeError):
    """Raised when inflate-time bias correction invariants are violated."""


@dataclass(frozen=True)
class A1PatternBiasCorrectionPlan:
    """Plan for inflate-time bias correction sweep on a substrate.

    Args:
        substrate_id: Canonical substrate id (e.g.
            ``"a1_pr95_paradigm_v1"`` or ``"nscs01_nullspace_split_
            renderer"``).
        baseline_archive_path: Path to the substrate's baseline archive.zip
            (the un-corrected reference).
        baseline_archive_sha256: SHA-256 of the baseline archive bytes; used
            to assert apples-to-apples evidence per CLAUDE.md non-negotiable.
        sweep_offsets: Tuple of integer byte offsets into the archive at
            which to apply bias corrections. Each offset MUST be within the
            archive's byte length; this is asserted at execution time.
        sweep_deltas: Tuple of signed byte deltas to add at each offset
            (typically small values like ``+1``, ``-1``, ``+2``, ``-2``,
            ``+128``, ``-128``). The Cartesian product
            ``(offset, delta)`` defines the sweep grid.
        held_out_metric_axis: One of ``"contest-CPU"`` (Linux x86_64 GHA;
            the A1 anchor axis) or ``"contest-CUDA"``. macOS-CPU advisory is
            REFUSED as the metric axis here because the empirical anchor is
            on Linux x86_64 GHA only.
        rationale: Operator-readable rationale for the sweep (1-line).
    """

    substrate_id: str
    baseline_archive_path: str
    baseline_archive_sha256: str
    sweep_offsets: tuple[int, ...]
    sweep_deltas: tuple[int, ...]
    held_out_metric_axis: Literal["contest-CPU", "contest-CUDA"]
    rationale: str

    def __post_init__(self) -> None:
        if not self.substrate_id:
            raise InflateBiasCorrectionError("substrate_id must be non-empty")
        if not self.baseline_archive_path:
            raise InflateBiasCorrectionError(
                "baseline_archive_path must be non-empty"
            )
        if (
            not self.baseline_archive_sha256
            or len(self.baseline_archive_sha256) != 64
        ):
            raise InflateBiasCorrectionError(
                "baseline_archive_sha256 must be a 64-char hex SHA-256 string"
            )
        try:
            int(self.baseline_archive_sha256, 16)
        except ValueError as e:
            raise InflateBiasCorrectionError(
                f"baseline_archive_sha256 is not valid hex: {e}"
            ) from e
        if not self.sweep_offsets:
            raise InflateBiasCorrectionError("sweep_offsets must be non-empty")
        for off in self.sweep_offsets:
            if off < 0:
                raise InflateBiasCorrectionError(
                    f"sweep_offset {off} must be >= 0"
                )
        if not self.sweep_deltas:
            raise InflateBiasCorrectionError("sweep_deltas must be non-empty")
        for delta in self.sweep_deltas:
            if delta == 0:
                raise InflateBiasCorrectionError(
                    "sweep_delta cannot be 0 (no-op; would not change archive)"
                )
        if self.held_out_metric_axis not in {"contest-CPU", "contest-CUDA"}:
            raise InflateBiasCorrectionError(
                f"held_out_metric_axis={self.held_out_metric_axis!r} not in "
                "{'contest-CPU', 'contest-CUDA'}; macOS-CPU advisory is "
                "REFUSED here per A1 empirical anchor axis discipline"
            )
        if not self.rationale or not self.rationale.strip():
            raise InflateBiasCorrectionError(
                "rationale must be non-empty per CLAUDE.md 'Comment-only "
                "contracts are FORBIDDEN'"
            )


@dataclass(frozen=True)
class InflateBiasCorrectionVerdict:
    """Verdict for one (offset, delta) point in a sweep.

    Args:
        offset: Byte offset.
        delta: Signed byte delta applied.
        candidate_archive_sha256: SHA-256 of the corrected archive.
        candidate_score: Held-out scorer score (axis = plan.held_out_metric_
            axis); ``None`` if eval was not run yet.
        baseline_score: Held-out scorer score on the baseline archive.
        delta_score: ``candidate_score - baseline_score``; positive means
            regression. ``None`` if ``candidate_score is None``.
        verdict: One of ``"NOT_EVALUATED"`` (no eval yet),
            ``"REGRESSION"`` (delta_score > +0.0005), ``"WITHIN_NOISE"``
            (|delta_score| <= 0.0005), ``"IMPROVEMENT"`` (delta_score <
            -0.0005). Thresholds match A1's empirical noise floor; see
            CLAUDE.md "Apples-to-apples evidence discipline".
    """

    offset: int
    delta: int
    candidate_archive_sha256: str
    candidate_score: float | None
    baseline_score: float
    delta_score: float | None
    verdict: Literal[
        "NOT_EVALUATED", "REGRESSION", "WITHIN_NOISE", "IMPROVEMENT"
    ]

    def __post_init__(self) -> None:
        if (
            not self.candidate_archive_sha256
            or len(self.candidate_archive_sha256) != 64
        ):
            raise InflateBiasCorrectionError(
                "candidate_archive_sha256 must be a 64-char hex SHA-256 string"
            )
        if self.verdict not in {
            "NOT_EVALUATED",
            "REGRESSION",
            "WITHIN_NOISE",
            "IMPROVEMENT",
        }:
            raise InflateBiasCorrectionError(
                f"verdict={self.verdict!r} not in canonical set"
            )
        if self.verdict == "NOT_EVALUATED":
            if self.candidate_score is not None or self.delta_score is not None:
                raise InflateBiasCorrectionError(
                    "NOT_EVALUATED verdict requires candidate_score=None and "
                    "delta_score=None"
                )
        else:
            if self.candidate_score is None or self.delta_score is None:
                raise InflateBiasCorrectionError(
                    f"{self.verdict!r} verdict requires non-None candidate_"
                    "score AND delta_score"
                )


class GeneralizedInflateBiasCorrector:
    """Substrate-agnostic inflate-time bias correction sweep generator.

    Plans the sweep but does NOT execute it (execution requires
    ``experiments/contest_auth_eval.py`` dispatch + GHA Linux x86_64 CPU
    custody per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
    CONTEST-COMPLIANT HARDWARE" non-negotiable).

    Usage::

        from tac.training_curriculum import (
            A1PatternBiasCorrectionPlan,
            GeneralizedInflateBiasCorrector,
        )

        plan = A1PatternBiasCorrectionPlan(
            substrate_id="balle_hyperprior_bolton_a1",
            baseline_archive_path="experiments/results/.../archive.zip",
            baseline_archive_sha256="...",
            sweep_offsets=(178200, 178201, 178202),
            sweep_deltas=(+1, -1, +2, -2),
            held_out_metric_axis="contest-CPU",
            rationale="probe whether BOLT-ON sidecar admits inflate-time "
                      "bias correction at A1 pattern parity",
        )
        corrector = GeneralizedInflateBiasCorrector(plan)
        candidate_paths = corrector.materialize_candidates(
            output_dir=Path("experiments/results/.../bias_sweep/"),
        )
        # ... dispatch each candidate through contest_auth_eval ...
        # ... harvest results back ...
        verdicts = corrector.classify_results(
            results_per_candidate={...},  # {candidate_sha: score}
        )

    The actual byte-mutation execution lives in :func:`materialize_candidates`
    which delegates to the canonical
    ``tools/build_a1_inflate_time_bias_correction_sweep.py`` invocation
    pattern (NOT imported here to avoid coupling the package to a tool
    script; the substrate trainer or operator script wires the invocation).

    Args:
        plan: :class:`A1PatternBiasCorrectionPlan`.
    """

    def __init__(self, plan: A1PatternBiasCorrectionPlan) -> None:
        self.plan = plan

    def grid(self) -> tuple[tuple[int, int], ...]:
        """Return the (offset, delta) Cartesian grid as a tuple."""
        return tuple(
            (off, d)
            for off in self.plan.sweep_offsets
            for d in self.plan.sweep_deltas
        )

    def materialize_candidates(
        self,
        *,
        output_dir: Path,
    ) -> dict[tuple[int, int], Path]:
        """Materialize one corrected-archive variant per grid point.

        Args:
            output_dir: Output directory; created if missing.

        Returns:
            Dict of ``{(offset, delta): candidate_archive_path}``.

        Raises:
            :class:`InflateBiasCorrectionError` on I/O or grid validation
            failure.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = Path(self.plan.baseline_archive_path)
        if not baseline_path.is_file():
            raise InflateBiasCorrectionError(
                f"baseline_archive_path {baseline_path} does not exist"
            )
        baseline_bytes = baseline_path.read_bytes()
        archive_len = len(baseline_bytes)
        for off in self.plan.sweep_offsets:
            if off >= archive_len:
                raise InflateBiasCorrectionError(
                    f"sweep_offset {off} >= archive length {archive_len}"
                )

        candidate_paths: dict[tuple[int, int], Path] = {}
        for off, delta in self.grid():
            new_bytes = bytearray(baseline_bytes)
            new_byte = (new_bytes[off] + delta) % 256
            new_bytes[off] = new_byte
            tag = f"off_{off}_delta_{delta:+d}".replace("+", "p").replace("-", "n")
            candidate_path = output_dir / f"{self.plan.substrate_id}_{tag}.zip"
            candidate_path.write_bytes(bytes(new_bytes))
            candidate_paths[(off, delta)] = candidate_path
        return candidate_paths

    def classify_results(
        self,
        *,
        baseline_score: float,
        results_per_candidate: dict[tuple[int, int], tuple[str, float | None]],
        noise_threshold: float = 0.0005,
    ) -> dict[tuple[int, int], InflateBiasCorrectionVerdict]:
        """Classify each candidate's empirical result vs. baseline.

        Args:
            baseline_score: Empirical baseline score (axis = plan.held_out_
                metric_axis).
            results_per_candidate: Dict of ``{(offset, delta): (candidate_
                sha256, candidate_score_or_None)}``.
            noise_threshold: |delta_score| <= this is ``WITHIN_NOISE``.
                Default ``0.0005`` matches A1's empirical noise floor (PR101
                bias variants showed minimum measurable delta ~0.0001).

        Returns:
            Dict of ``{(offset, delta): InflateBiasCorrectionVerdict}``.

        Raises:
            :class:`InflateBiasCorrectionError` on grid mismatch.
        """
        expected_grid = set(self.grid())
        actual_grid = set(results_per_candidate.keys())
        if expected_grid != actual_grid:
            missing = expected_grid - actual_grid
            extra = actual_grid - expected_grid
            raise InflateBiasCorrectionError(
                f"results grid mismatch: missing={missing!r} extra={extra!r}"
            )
        if noise_threshold <= 0:
            raise InflateBiasCorrectionError(
                f"noise_threshold={noise_threshold} must be > 0"
            )

        verdicts: dict[tuple[int, int], InflateBiasCorrectionVerdict] = {}
        for (off, delta), (sha256, score) in results_per_candidate.items():
            if score is None:
                verdict_class: Literal[
                    "NOT_EVALUATED", "REGRESSION", "WITHIN_NOISE", "IMPROVEMENT"
                ] = "NOT_EVALUATED"
                delta_score = None
            else:
                delta_score = score - baseline_score
                if abs(delta_score) <= noise_threshold:
                    verdict_class = "WITHIN_NOISE"
                elif delta_score > noise_threshold:
                    verdict_class = "REGRESSION"
                else:  # delta_score < -noise_threshold
                    verdict_class = "IMPROVEMENT"
            verdicts[(off, delta)] = InflateBiasCorrectionVerdict(
                offset=off,
                delta=delta,
                candidate_archive_sha256=sha256,
                candidate_score=score,
                baseline_score=baseline_score,
                delta_score=delta_score,
                verdict=verdict_class,
            )
        return verdicts
