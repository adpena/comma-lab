# SPDX-License-Identifier: MIT
"""Canonical equation: the #205 n600 OOM is the ADVISORY-VERDICT batched-scorer transient spike,
NOT the accum loop or fp32 keyframes -- always CHUNK full-P scorer forwards (MEASURED 2026-07-02).

THE MEASURED FINDING (memory ``n205_oom_is_verdict_batch_spike_not_accum_loop_chunk_verdict_20260702`` +
ledger ``.omx/research/n205_oom_fix_and_relaunch_20260702T224907Z.md``): the n600 level-set witness
OOM'd (safe_run killed it at ~90 GB before the first checkpoint) because ``realized_verdict()`` runs the
CPU SegNet+PoseNet over ONE **P-wide** torch batch -> a **+66 GiB** fp32 transient
(``(600,2,3,874,1164)`` cast + EfficientNet-B2 / FastViT-T12 activations) ON TOP of the resident
~41 GiB self-orient ``cf_mx_cache``. Falsified NON-causes (do not re-chase): the accum loop (RSS FLAT
at n64), an fp32 keyframe table (gt is already uint8; the residual is (P,6)=14 KB).

THE LAW (the equation this module codifies): any full-P batched CPU-scorer forward at n600 is a
~66 GiB transient spike whose peak scales with the VERDICT BATCH, not P. Chunk it:

    verdict_transient_gib(P, vbatch) = max(FLOOR, PER_PAIR * min(vbatch, P))    for vbatch > 0
    verdict_transient_gib(P, vbatch) = PER_PAIR * P                             for vbatch <= 0 (UNCHUNKED)

with PER_PAIR = 0.11 GiB/pair, FLOOR = 6.0 GiB (measured chunked floor). The FIX is
``--verdict-batch 32`` (default-on) -> the spike collapses to ~6 GiB. Score-neutral BY CONSTRUCTION:
the verdict is advisory (outside ``value_and_grad``, never read back) and eval-mode BN uses running
stats (batch-size independent) -> d_seg BIT-IDENTICAL, d_pose ~1e-6 BLAS-tiling noise (inside 0.9997
parity; advisory-only). Self-protection: ``tools/witness_memory_preflight.py`` composes this spike law
into the full peak-RSS projection (``FIXED ~15 + cf_mx_cache(P,self_orient) + gt(P) + verdict(vbatch)``)
and ``tools/launch_witness_run.py`` REFUSES (rc=4) a config whose projected peak > 0.70 x RAM.

VERDICT (honest, NO-FAKE): a SCORE-NEUTRAL engineering / launch-safety law (not a d_seg/d_pose/rate
lever); pointer 0.19110 UNMOVED. The constants MIRROR the live guard ``tools/witness_memory_preflight.py``
(the canonical producer); the test cross-checks them so drift is caught. Advisory measurement axis
(``[macOS-CPU advisory] NON-PROMOTABLE``, frozen CPU-torch scorer, NEVER MPS).

Consumers: ``tools/launch_witness_run.py`` (the fail-closed launch gate) + ``tools/witness_memory_preflight.py``
(the full projection). Producer: the isolated micro-probe + the memory-preflight guard.
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

EQUATION_ID = "oom_verdict_batch_spike_peak_rss_v1"

_UTC = "2026-07-02T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_LEDGER = ".omx/research/n205_oom_fix_and_relaunch_20260702T224907Z.md"

# MEASURED calibration constants (2026-07-02 isolated micro-probes) -- MIRRORED from the live guard
# tools/witness_memory_preflight.py; the test cross-checks these against the guard so drift is caught.
VERDICT_PER_PAIR_GIB = 0.11          # marginal fp32-cast + EfficientNet-B2/FastViT-T12 activations per batched pair
VERDICT_FLOOR_GIB = 6.0              # measured chunked floor (vbatch in {8, 32} both ~5.6 GiB)
DEFAULT_VERDICT_BATCH = 32           # the score-neutral fix (default-on)
DEFAULT_SAFE_FRAC = 0.70             # launcher refuses projected peak > this fraction of total RAM

# MEASURED n600 verdict micro-probe (isolated resource.getrusage, N=600).
_UNCHUNKED_PEAK_TRANSIENT_GIB = 66.2   # +66 over baseline (single 600-wide batch)  -> the killer
_CHUNKED_VBATCH32_TRANSIENT_GIB = 5.6  # +5.6 over baseline (vbatch=32 fix)
_UNCHUNKED_PEAK_RSS_GIB = 70.65        # isolated-probe peak RSS (verdict path only)
_CHUNKED_PEAK_RSS_GIB = 10.03          # isolated-probe peak RSS (vbatch=32)
_DSEG_BOTH = 0.994764                  # BIT-IDENTICAL across chunked/unchunked (untrained smoke witness)
_DPOSE_BOTH = 0.096572                 # ~1e-6 BLAS-tiling noise between paths (advisory, inside 0.9997)


def verdict_transient_gib(num_pairs: int, verdict_batch: int = DEFAULT_VERDICT_BATCH) -> float:
    """Peak transient RSS (GiB) of the batched CPU-scorer verdict forward -- the #205 OOM killer law.

    ``verdict_batch <= 0`` => the UNCHUNKED N-wide batch => the full ~0.11 GiB/pair spike (this is what
    OOM'd #205 at n600: 0.11 * 600 ~= 66 GiB). ``verdict_batch > 0`` => chunked => max(6 GiB floor,
    0.11 * min(vbatch, P)). Pure + deterministic (unit-testable)."""
    p = int(num_pairs)
    if verdict_batch is None or int(verdict_batch) <= 0:
        return VERDICT_PER_PAIR_GIB * p
    eff = min(int(verdict_batch), p)
    return max(VERDICT_FLOOR_GIB, VERDICT_PER_PAIR_GIB * eff)


def unchunked_spike_gib(num_pairs: int) -> float:
    """The unchunked (pre-fix) verdict transient at ``num_pairs`` = ``0.11 * P`` GiB."""
    return verdict_transient_gib(num_pairs, verdict_batch=0)


def is_verdict_batch_launch_safe(
    projected_peak_gib: float, total_ram_gib: float, safe_frac: float = DEFAULT_SAFE_FRAC
) -> bool:
    """True iff the projected full peak RSS is <= ``safe_frac`` x total RAM (the launch-gate rule)."""
    return float(projected_peak_gib) <= float(safe_frac) * float(total_ram_gib)


def build_oom_verdict_batch_spike_peak_rss_v1() -> CanonicalEquation:
    """Build the #205 verdict-batch peak-RSS spike canonical equation."""
    pred_unchunked = unchunked_spike_gib(600)                  # 66.0
    pred_chunked = verdict_transient_gib(600, DEFAULT_VERDICT_BATCH)  # 6.0
    res_unchunked = abs(pred_unchunked - _UNCHUNKED_PEAK_TRANSIENT_GIB) / _UNCHUNKED_PEAK_TRANSIENT_GIB
    res_chunked = abs(pred_chunked - _CHUNKED_VBATCH32_TRANSIENT_GIB) / _CHUNKED_VBATCH32_TRANSIENT_GIB
    residual = float(0.5 * (res_unchunked + res_chunked))

    anchor = EmpiricalAnchor(
        anchor_id="n205_verdict_batch_spike_isolated_microprobe_n600_20260702",
        measurement_utc=_UTC,
        inputs={
            "num_pairs": 600,
            "scorer": "frozen_cpu_torch_segnet_b2_plus_posenet_fastvit_t12",
            "probe": "experiments/results/n205_oom_probe_*/verdict_mem_microprobe.py",
            "verdict_batch_a": 0,   # unchunked
            "verdict_batch_b": 32,  # chunked fix
        },
        predicted_output={
            "verdict_transient_gib_unchunked_pred": pred_unchunked,
            "verdict_transient_gib_vbatch32_pred": pred_chunked,
        },
        empirical_output={
            "verdict_transient_gib_unchunked_measured": _UNCHUNKED_PEAK_TRANSIENT_GIB,
            "verdict_transient_gib_vbatch32_measured": _CHUNKED_VBATCH32_TRANSIENT_GIB,
            "peak_rss_gib_unchunked": _UNCHUNKED_PEAK_RSS_GIB,
            "peak_rss_gib_vbatch32": _CHUNKED_PEAK_RSS_GIB,
            "d_seg_both_paths": _DSEG_BOTH,
            "d_pose_both_paths": _DPOSE_BOTH,
            "verdict": (
                "OOM = verdict-batch spike (+66 GiB unchunked -> +5.6 GiB vbatch=32); d_seg BIT-IDENTICAL, "
                "d_pose ~1e-6 noise -> score-neutral. Fix = --verdict-batch 32 (default). Always chunk "
                "full-P scorer forwards; launcher REFUSES projected peak > 0.70 x RAM."
            ),
        },
        residual=residual,
        source_artifact=_LEDGER,
        measurement_method="isolated_resource_getrusage_verdict_microprobe_n600_frozen_cpu_torch",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "re-measure the per-pair verdict constant if the scorer arch, render resolution, or "
                "torch batching changes; recalibrate PER_PAIR/FLOOR against the guard's live constants"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu_torch",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "The #205 n600 OOM is the batched-verdict CPU-scorer transient spike -- always chunk "
            "full-P scorer forwards (--verdict-batch 32); score-neutral"
        ),
        one_line_summary=(
            "verdict_transient ~= 0.11 GiB/pair; unchunked n600 = +66 GiB spike (OOM), vbatch=32 = "
            "+6 GiB (fix); score-neutral (d_seg bit-identical). Launcher refuses peak > 0.70 x RAM."
        ),
        latex_form=(
            r"\mathrm{RSS}_{verdict}(P,b) = \begin{cases} 0.11\,P & b\le 0\ (\text{unchunked, OOM}) \\ "
            r"\max(6,\,0.11\min(b,P)) & b>0\ (\text{chunked, fix})\end{cases};\ "
            r"\text{launch iff } \mathrm{peak} \le 0.70\,\mathrm{RAM}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.oom_verdict_batch_spike_peak_rss_20260702:verdict_transient_gib"
        ),
        domain_of_validity={
            "vehicle": ["levelset_witness_n600"],
            "scorer": ["frozen_cpu_torch_segnet_b2", "frozen_cpu_torch_posenet_fastvit_t12"],
            "measurement_axis": [_ADVISORY],
            "result_type": "SCORE-NEUTRAL engineering / launch-safety law (not a d_seg/d_pose/rate lever)",
            "score_neutral_because": ("verdict is advisory (outside value_and_grad, never read back) + "
                                      "eval-mode BN uses running stats (batch-size independent) => bit-identical"),
            "constants_mirror": "tools/witness_memory_preflight.py (drift-guarded by the test)",
            "untested": ["non_600_pair_counts_beyond_the_n64/n300/n600_probes",
                         "non_self_orient_cf_cache_regime"],
        },
        units_in={"num_pairs": "count", "verdict_batch": "count"},
        units_out={"verdict_transient_gib": "gibibytes_rss"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "isolated_resource_getrusage_verdict_microprobe_n600_frozen_cpu_torch": residual,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/launch_witness_run.py",
            "tools/witness_memory_preflight.py",
        ),
        canonical_producers=(
            "tools/witness_memory_preflight.py",
            "tools/register_triality_reconcile_session_20260702_equations.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="oom_verdict_batch_spike_peak_rss.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "DEFAULT_SAFE_FRAC",
    "DEFAULT_VERDICT_BATCH",
    "VERDICT_FLOOR_GIB",
    "VERDICT_PER_PAIR_GIB",
    "build_oom_verdict_batch_spike_peak_rss_v1",
    "is_verdict_batch_launch_safe",
    "unchunked_spike_gib",
    "verdict_transient_gib",
]
