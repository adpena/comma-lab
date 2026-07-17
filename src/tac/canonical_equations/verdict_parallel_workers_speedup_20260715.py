# SPDX-License-Identifier: MIT
"""Canonical equation: verdict-parallel-workers wall-clock law (#509 batch 3 / m5max
mem-for-compute fire-now).

THE LAW: the ADVISORY CPU-torch verdict's scorer-forward wall divides by an efficiency-
discounted worker count when its ``--verdict-batch`` chunks fan across a thread pool
(1-thread-law torch intra-op preserved per worker; values BIT-IDENTICAL by construction
and ASSERTED on real inputs):

    wall(N) = wall(1) / (eta_N * N),   eta_8 = 0.711 (MEASURED), eta_6 = 0.803

MEASURED (2026-07-15, receipt ``experiments/results/verdict_parallel_bench_20260715T184252Z/
receipt.json``, n600 real gt frames, vbatch 32, thread-law pinned):

    w=0 (sequential): 370.64 s   ·   w=8: 65.18 s (5.686x)   ·   w=6: 76.91 s (4.819x)
    values identical across all arms (d_seg/d_pose float-equal)  ·  peak RSS 33.9 GiB
    (~2.9 GB/worker marginal — under the 5-6 GB sizing estimate)

SCOPE CAVEAT (honest): the bench measures the SCORER-FORWARD share of the verdict. The
in-run verdict epoch also renders the 600 candidate frames + runs the realized/annulus
machinery — the C0 in-run verdict wall (~2555.7 s) is dominated by those UN-parallelized
stages, so the workers lever recovers ~305 s of the ~370 s scorer share per verdict epoch
(amortized /eval-cadence-25 ≈ 12 s/ep), NOT a division of the whole verdict epoch. The
render/realized share is a SEPARATE lever (owed).

Sizing (value-provenance DERIVED): workers=8 from measured headroom (~2.9 GB/worker x 8 +
resident << the 0.70 safe-frac envelope) — compose ``VerdictParallelWorkers(8)`` into the
next n600 launch; trainer default stays 0 (byte-identical incumbent).

means != ends: a verdict-wall sec lever; ADVISORY path only (never read into training =>
score-neutral by construction); NO score claim; pointer UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "verdict_parallel_workers_speedup_v1"

_UTC = "2026-07-15T18:42:52Z"
_ADVISORY = "[macOS-CPU-torch timing] ADVISORY NON-PROMOTABLE"
_RECEIPT = "experiments/results/verdict_parallel_bench_20260715T184252Z/receipt.json"

WALL_SEQUENTIAL_S = 370.64
WALL_W8_S = 65.18
WALL_W6_S = 76.91
SPEEDUP_W8 = 5.686
SPEEDUP_W6 = 4.819
ETA_W8 = SPEEDUP_W8 / 8.0   # 0.711 pool efficiency at 8 workers
ETA_W6 = SPEEDUP_W6 / 6.0   # 0.803
PEAK_RSS_GIB = 33.86
MARGINAL_GB_PER_WORKER = 2.9
SIZED_WORKERS = 8


def derived_verdict_workers(
    available_gib: "float | None" = None, *, safe_frac: float = 0.70,
) -> int:
    """DERIVED sizing for ``VerdictParallelWorkers`` (value-provenance rung: DERIVED from the
    measured anchor constants of this law — never a hand-typed count).

    Largest ``w`` in ``[2, SIZED_WORKERS]`` whose projected pool RSS fits the safe fraction of
    available memory::

        base + w * MARGINAL_GB_PER_WORKER <= safe_frac * available_gib
        base = PEAK_RSS_GIB - SIZED_WORKERS * MARGINAL_GB_PER_WORKER   (measured intercept)

    ``SIZED_WORKERS`` (=8) is the measured ladder top — no efficiency evidence beyond w=8, so
    the derivation never extrapolates above it. ``available_gib=None`` reads live available
    memory via psutil; any failure returns the conservative floor 2 (fewest parallel workers =
    least memory risk; 0/1 sequential is composed by NOT composing the lever, per the
    off-is-orphan rule). Deterministic given ``available_gib``."""
    if available_gib is None:
        # CLASS-1 fix: reclaimable-aware basis — raw psutil .available over-trusts dirty inactive
        # anon; over-trusting here would size TOO MANY workers and OOM the box.
        try:
            try:
                from tools.mem_basis import conservative_free_gib  # noqa: PLC0415
            except Exception:
                from mem_basis import conservative_free_gib  # type: ignore # noqa: PLC0415
            cf = conservative_free_gib(default=float("nan"))
            if cf != cf:  # NaN → measurement unavailable → conservative floor
                return 2
            available_gib = cf
        except Exception:
            return 2
    base_gib = PEAK_RSS_GIB - SIZED_WORKERS * MARGINAL_GB_PER_WORKER
    budget = float(safe_frac) * float(available_gib) - base_gib
    if budget <= 0:
        return 2
    w = int(budget // MARGINAL_GB_PER_WORKER)
    return max(2, min(int(SIZED_WORKERS), w))


def verdict_wall_projection(wall_sequential_s: float, workers: int) -> float:
    """Project the scorer-forward verdict wall at N workers from the measured efficiency
    ladder (linear interpolation of eta between the measured points; eta clamps to the
    nearest measured value outside [6, 8]). Deterministic fp64."""
    n = int(workers)
    if n <= 1:
        return float(wall_sequential_s)
    if n <= 6:
        eta = ETA_W6
    elif n >= 8:
        eta = ETA_W8
    else:
        t = (n - 6) / 2.0
        eta = ETA_W6 + t * (ETA_W8 - ETA_W6)
    return float(wall_sequential_s) / (eta * n)


def build_verdict_parallel_workers_speedup_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="verdict_parallel_bench_n600_w0_w8_w6_20260715",
        measurement_utc=_UTC,
        inputs={
            "bench": "tools/bench_verdict_parallel_workers.py --workers 0 8 6",
            "pairs": 600, "vbatch": 32,
            "frames": "real gt_n600 frames (wall is content-independent; values asserted)",
            "torch_intraop_threads": "segnet_exact_forward_cpu_thread_law_20260713.SELECTED_THREADS",
        },
        predicted_output={
            "hypothesis": ("chunk fan-out divides the scorer-forward wall by ~eta*N; values "
                           "bit-identical (same chunks, ordered aggregation)"),
        },
        empirical_output={
            "wall_s": {"w0": WALL_SEQUENTIAL_S, "w8": WALL_W8_S, "w6": WALL_W6_S},
            "speedup": {"w8": SPEEDUP_W8, "w6": SPEEDUP_W6},
            "values_identical_across_arms": True,
            "peak_rss_gib": PEAK_RSS_GIB,
            "scope": ("SCORER-FORWARD share only — the in-run verdict epoch's render/realized "
                      "stages are un-parallelized by this lever (separate owed lever)"),
            "verdict_scope": "INSTANCE (this host, this thread law, vbatch 32; re-bench on change)",
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method=("paired same-process bench, untimed warmup, per-arm perf_counter; "
                            "value-equality asserted across arms on real inputs"),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIPT,
            reactivation_criteria=("re-bench when vbatch / thread law / scorer stack / host "
                                   "changes, or when the render/realized share is parallelized"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_torch",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Verdict-parallel-workers wall law: scorer-forward verdict wall(N) = wall(1)/"
              "(eta_N*N), eta_8=0.711 measured; values bit-identical; advisory path only"),
        one_line_summary=(
            "Chunk fan-out of the ADVISORY CPU verdict divides its scorer-forward wall by "
            "eta*N (measured 5.69x at 8 workers, n600, values identical; ~2.9 GB/worker)."
        ),
        latex_form=r"T_{verdict}(N)=\frac{T(1)}{\eta_N N},\quad \eta_8=0.711\ (\mathrm{meas})",
        python_callable_module_path=(
            "tac.canonical_equations.verdict_parallel_workers_speedup_20260715:"
            "verdict_wall_projection"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "v9_cgauge_*"],
            "lever": ("tac.witness_dsl.curriculum_dsl.VerdictParallelWorkers "
                      "(--verdict-parallel-workers; trainer default 0 = sequential incumbent)"),
            "measurement_axis": ["macOS-CPU-torch timing advisory"],
            "note": ("ADVISORY verdict path only (never read into training => score-neutral by "
                     "construction). Covers the scorer-forward share; render/realized share is a "
                     "separate lever. Sizing DERIVED: 8 workers from measured 2.9 GB/worker "
                     "inside the 0.70 safe-frac envelope."),
        },
        units_in={"wall_sequential_s": "seconds", "workers": "count"},
        units_out={"wall_s": "seconds"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"n600_bench_w8": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # VerdictParallelWorkers factory sizing
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        canonical_producers=(
            "tools/bench_verdict_parallel_workers.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="verdict_parallel_workers_speedup.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="apple_m5_max_cpu_torch",
        ),
    )


def populate_verdict_parallel_workers_speedup_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins). EQUATIONS leg of the
    verdict-parallel-workers lever; DSL leg = ``curriculum_dsl.VerdictParallelWorkers``;
    mechanism = the trainer's ``_verdict_dseg_dpose_chunked(workers=...)``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_verdict_parallel_workers_speedup_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="verdict_parallel_workers_speedup_20260715 (measured 5.69x @ w=8 n600; "
              "scorer-forward share; equations leg of the #509 workers lever)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "ETA_W6",
    "ETA_W8",
    "MARGINAL_GB_PER_WORKER",
    "PEAK_RSS_GIB",
    "SIZED_WORKERS",
    "SPEEDUP_W6",
    "SPEEDUP_W8",
    "WALL_SEQUENTIAL_S",
    "WALL_W6_S",
    "WALL_W8_S",
    "build_verdict_parallel_workers_speedup_v1",
    "derived_verdict_workers",
    "populate_verdict_parallel_workers_speedup_equation",
    "verdict_wall_projection",
]
