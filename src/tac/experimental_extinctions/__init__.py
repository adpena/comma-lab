# SPDX-License-Identifier: MIT
"""Canonical empirical-sweep helpers for the 8 remaining Wave 2C extinctions.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
canonical-vs-unique decision discipline: each module RUNS AN EMPIRICAL SWEEP
on a representative population (substrate / payload / pixel / corpus / state
JSONL) to replace a hand-picked arbitrary value with its empirically-optimal
counterpart. All results are tagged ``[macOS-CPU advisory]`` per Catalog
#192/#317; promotion to ``[contest-CPU]`` REQUIRES a paired Linux x86_64
anchor (NOT in scope for this wave).

Sister packages (DISJOINT scope; READ-ONLY from this package):
    - ``tac.contest_oracle`` — canonical contest-oracle classifier (sister
      subagent ``contest_oracle_canonical_package_20260518`` IN FLIGHT).
    - ``tac.analytical_solve_extinctions`` — sister Wave 2A analytical-solve
      helpers (LANDED commit ``8b987215a``).
    - ``tac.formula_extinctions`` — sister Wave 2B formula-derivation helpers
      (sister subagent IN FLIGHT). NOTE: Wave 2B ships
      ``canonical_early_stopping_patience`` (Prechelt slope formula); THIS
      package ships ``per_substrate_convergence_aware_early_stopping`` which
      is the sister empirical-calibration discipline (per-substrate sweep
      vs canonical-formula derivation).
    - ``tac.atom`` — canonical atom ledger surfacing arbitrariness atoms.
    - ``tac.provenance`` — canonical provenance contract (Catalog #323).
    - ``tac.optimization.macos_cpu_advisory_signal`` — canonical advisory
      manifest persistence per Catalog #192/#317.
    - ``tac.probe_outcomes_ledger`` — READ-ONLY corpus for row #6.
    - ``tac.council_continual_learning`` — READ-ONLY corpus for row #5.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer experience"):

    from tac.experimental_extinctions import (
        # Row #1 — per-substrate convergence-aware early stopping
        per_substrate_convergence_aware_early_stopping,
        ConvergenceSweepInput,
        # Row #2 — brotli quality 10 vs 11 per-payload sweep
        brotli_quality_10_vs_11_payload_sweep,
        BrotliSweepInput,
        # Row #3 — lzma vs zstd vs brotli per-payload codec sweep
        lzma_vs_zstd_vs_brotli_per_payload_sweep,
        CodecSweepInput,
        # Row #4 — SegNet boundary curvature sigma calibration
        segnet_boundary_curvature_sigma_calibration,
        SigmaCalibrationInput,
        # Row #5 — council cadence empirical calibration
        council_cadence_empirical_calibration,
        CouncilCadenceInput,
        # Row #6 — probe-outcome staleness decay calibration
        probe_outcome_staleness_decay_calibration,
        ProbeDecayInput,
        # Row #7 — negation-window FP/FN corpus sweep (Catalog #236)
        negation_window_fp_fn_corpus_sweep,
        NegationWindowInput,
        # Row #8 — memory-file category decay calibration
        memory_file_category_decay_calibration,
        MemoryDecayInput,
        # Shared result dataclass
        EmpiricalSweepResult,
    )

Each helper emits a structured ``EmpiricalSweepResult`` and optionally a
``tac.atom.Atom`` instance via ``emit_arbitrariness_atom=True`` so downstream
consumers (cathedral autopilot ranker / continual-learning posterior /
probe-disambiguator) see the canonical extinction record per Catalog #125
6-hook wire-in declaration.

Citations
---------
- Prechelt 1998 "Early Stopping — But When?" Neural Networks: Tricks of the Trade
- Alakuijala-Szabadka 2016 "Brotli Compressed Data Format" RFC 7932 (quality 0-11)
- Pavlov 1999 "LZMA Specification" (preset 0-9 = compression level)
- Collet 2016 "Zstandard Compression and the application/zstd Media Type" RFC 8478
- Mallat 1989 "A Theory for Multiresolution Signal Decomposition" (wavelet sigma)
- Cover-Thomas "Elements of Information Theory" (decay rate from posterior)
- Surowiecki 2004 "The Wisdom of Crowds" (group decision throughput)
- Kemeny-Snell 1962 "Mathematical Models in the Social Sciences" (preference)
- Wang-Rudin 2015 "Falling Rule Lists" (linguistic FP/FN tradeoff)

Lane: ``lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518``
"""

from __future__ import annotations

from tac.experimental_extinctions.per_substrate_convergence_aware_early_stopping import (  # noqa: F401
    ConvergenceSweepInput,
    EmpiricalSweepResult,
    per_substrate_convergence_aware_early_stopping,
)
from tac.experimental_extinctions.brotli_quality_10_vs_11_payload_sweep import (  # noqa: F401
    BrotliSweepInput,
    brotli_quality_10_vs_11_payload_sweep,
)
from tac.experimental_extinctions.lzma_vs_zstd_vs_brotli_per_payload_sweep import (  # noqa: F401
    CodecSweepInput,
    lzma_vs_zstd_vs_brotli_per_payload_sweep,
)
from tac.experimental_extinctions.segnet_boundary_curvature_sigma_calibration import (  # noqa: F401
    SigmaCalibrationInput,
    segnet_boundary_curvature_sigma_calibration,
)
from tac.experimental_extinctions.council_cadence_empirical_calibration import (  # noqa: F401
    CouncilCadenceInput,
    council_cadence_empirical_calibration,
)
from tac.experimental_extinctions.probe_outcome_staleness_decay_calibration import (  # noqa: F401
    ProbeDecayInput,
    probe_outcome_staleness_decay_calibration,
)
from tac.experimental_extinctions.negation_window_fp_fn_corpus_sweep import (  # noqa: F401
    NegationWindowInput,
    negation_window_fp_fn_corpus_sweep,
)
from tac.experimental_extinctions.memory_file_category_decay_calibration import (  # noqa: F401
    MemoryDecayInput,
    memory_file_category_decay_calibration,
)

__all__ = [
    # Shared result dataclass
    "EmpiricalSweepResult",
    # Row #1
    "ConvergenceSweepInput",
    "per_substrate_convergence_aware_early_stopping",
    # Row #2
    "BrotliSweepInput",
    "brotli_quality_10_vs_11_payload_sweep",
    # Row #3
    "CodecSweepInput",
    "lzma_vs_zstd_vs_brotli_per_payload_sweep",
    # Row #4
    "SigmaCalibrationInput",
    "segnet_boundary_curvature_sigma_calibration",
    # Row #5
    "CouncilCadenceInput",
    "council_cadence_empirical_calibration",
    # Row #6
    "ProbeDecayInput",
    "probe_outcome_staleness_decay_calibration",
    # Row #7
    "NegationWindowInput",
    "negation_window_fp_fn_corpus_sweep",
    # Row #8
    "MemoryDecayInput",
    "memory_file_category_decay_calibration",
]
