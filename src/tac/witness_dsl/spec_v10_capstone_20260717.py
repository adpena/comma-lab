# SPDX-License-Identifier: MIT
"""spec_v10_capstone — THE CAPSTONE: cold start on a fully seeded, Kolmogorov-optimal program.

SPEC (canonical): ``.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md``
Charter (naming SoT): ``.omx/research/vehicle_naming_resolution_v10_capstone_20260717.md``
Task #521 · P0 ledger ``p0_v10_capstone_cold_start_seeded_20260717``.

Operator binding (verbatim, naming SoT): "after the current run and any outstanding A/B
we are doing cold start on a fully seeded Kolmogorov optimal program."

THIS MODULE IS THE FAIL-CLOSED SKELETON, mirroring the ``spec_c2_surgical_20260716``
pattern (metadata + a pure ``compile_*`` entry point; CONTAINMENT — builds/validates
only, NEVER launches; $0). It deliberately does NOT emit a flag-for-flag base config:
the v10 composition depends on seven recorded open questions (SPEC §7) and five
#518-branch post-merge levers. Emitting argv before those resolve would be a FAKE
config (NO-FAKE #4/#6) and a never-invent-flags violation. Instead,
:func:`compile_v10_capstone_launch_config` computes the full BLOCKER set and raises
:class:`SpecV10CompileBlocked` until every gate clears — the compile function is
PRESENT and honest, not a marker.

Value-provenance: every constant below is a ``(value, provenance, cite)`` triple —
MEASURED / DERIVED / ANCHOR / OPEN — never bare (SPEC §9; constants-are-poison).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── #518-branch post-merge lever dependencies (SPEC P2/P5). ────────────────────────
# These names exist on the #518 branch, NOT on this branch. They are import-probed at
# compile time and converted into typed blockers — NEVER fake-resolved, never guessed
# as trainer flags (never-invent-flags).
POST_MERGE_518_LEVERS: tuple[str, ...] = (
    "ResumeLRWarmup",       # beta2-derived warm-up (P5; constant decided by the 8-vs-27 A/B, OQ-3)
    "ForkHeadSolve",        # rank-4 exact head solve at init (P2)
    "MarginStepCap",        # boundary-law step cap (P5)
    "PoseEngageWPoseRamp",  # w_pose ramp at pose-engage boundary (P5/P6)
    "ForkEmaClearance",     # EMA clearance at fork boundary (P5)
)

# ── Launch-gate artifacts (SPEC §6). Paths are read-only probes; run dirs sacred. ──
GATE_ARTIFACTS: dict[str, dict[str, str]] = {
    "v9c2_completion": {
        "path": "experiments/results/levelset_n600_witness_20260717T113932Z/RUN_COMPLETE.json",
        "detail": "v9c2 governed-stop receipt + terminal harvest (SPEC gate 1; OQ-0/1/4/5). "
                  "Read-only existence probe — this module never writes into run dirs.",
    },
    "p0_497_curvelet_ab_verdict": {
        "path": "experiments/results/curvelet_matched_bytes_ab_20260717/verdict.json",
        "detail": "p0_497 curvelet matched-bytes through-R A/B verdict (SPEC gate 2; OQ-2 → P7 basis).",
    },
    "warmup_8v27_ab_verdict": {
        "path": "experiments/results/warmup_8v27_ab/verdict.json",
        "detail": "#518 8-vs-27 warm-up short-arm verdict (SPEC gate 3; OQ-3 → ResumeLRWarmup LawRef).",
    },
    "probe_p1_n600_band_and_terminal_decomp": {
        "path": "experiments/results/v10_probe_p1_n600_band_terminal_decomp/verdict.json",
        "detail": "P-1: n600 re-measure of the one-sided band + witness-own decomp on v9c2 terminal "
                  "frames (necessity row 2 seal).",
    },
    "probe_p2_mirror_transport_rate": {
        "path": "experiments/results/v10_probe_p2_mirror_transport/verdict.json",
        "detail": "P-2: mirror-transport term rate-vs-residual on the hood rim (necessity row 4).",
    },
    "probe_p3_chroma_plane_jacobian": {
        "path": "experiments/results/v10_probe_p3_chroma_jacobian/verdict.json",
        "detail": "P-3: per-pair chroma-plane margin-Jacobian projection at n600 (necessity row 9).",
    },
}

# ── Seed artifacts that must exist before a v10 config can be composed (SPEC §4). ──
SEED_ARTIFACTS: dict[str, dict[str, str]] = {
    "gt_cache_n600": {
        "path": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "detail": "bit-exact cached GT argmax (the measurement substrate).",
    },
    "hood_tex_seed": {
        "path": "experiments/results/necessity_dseg_calibration_20260715/hood_tex_seed.npz",
        "detail": "static hood-tex seed, 1,759 counted bytes (MEASURED: d_seg 0.04538->0.01328, "
                  "min-S 1.613; necessity_dseg_calibration_20260715.md). Path is the calibration "
                  "artifact family root; exact filename confirmed at harvest.",
    },
}

# ── SPEC constants on the value-provenance ladder (never bare). ────────────────────
V10_CONSTANTS: dict[str, dict[str, Any]] = {
    "hood_tex_seed_bytes": {
        "value": 1759, "provenance": "MEASURED",
        "cite": ".omx/research/necessity_dseg_calibration_20260715.md (min-S 1.613 knee)",
    },
    "ker_A_blind_row_fraction": {
        "value": 0.226969, "provenance": "MEASURED",
        "cite": ".omx/research/null_subspace_rate_measure_20260717.md (#519; 106/874 x 140/1164)",
    },
    "ker_A_render_energy_fraction": {
        "value": 0.524, "provenance": "MEASURED",
        "cite": "null_subspace_rate_measure_20260717.md (raw 52.42% [52.35,52.58])",
    },
    "gauge_int8_scale_refinement": {
        "value": 0.223, "provenance": "MEASURED",
        "cite": "null_subspace_rate_measure_20260717.md (scale 0.028416->0.022088; n600 confirm owed #406)",
    },
    "strict_camera_support_fraction": {
        "value": 0.0166, "provenance": "MEASURED",
        "cite": ".omx/research/necessity_solver_inverse_factorization_20260715.md",
    },
    "saddle_sub_lsb_fraction": {
        "value": 0.292, "provenance": "MEASURED",
        "cite": "necessity_solver_inverse_factorization_20260715.md (precision-only currency)",
    },
    "k_over_h": {
        "value": 0.47, "provenance": "MEASURED",
        "cite": "necessity_solver_inverse_factorization_20260715.md (the class-shift measure)",
    },
    "frame0_dseg_obligation": {
        "value": 8.5e-9, "provenance": "MEASURED",
        "cite": "frozen_scorer_exact_factorization_20260715.md S1 / DAG Unit-C",
    },
    "rate_ladder_d37_net_bytes": {
        "value": 384637, "provenance": "ANCHOR",
        "cite": "rate-law ladder D37 (ceiling, NOT a prediction; D38 #483 owed)",
    },
    "per_dash_anchor_bytes_est": {
        "value": (900, 1800), "provenance": "DERIVED",
        "cite": "L86 estimate; confirmed only at byte-close (SPEC F-review F2/F4 discipline)",
    },
}


class SpecV10CompileBlocked(RuntimeError):
    """Raised fail-closed when the v10 composition cannot honestly be compiled yet."""

    def __init__(self, blockers: list[dict[str, str]]):
        self.blockers = blockers
        lines = "\n".join(f"  - [{b['id']}] {b['detail']}" for b in blockers)
        super().__init__(
            "SPEC_v10 compile BLOCKED (fail-closed by design; SPEC §8). "
            f"{len(blockers)} blocker(s):\n{lines}\n"
            "Rule chain: naming-SoT sequencing (v9c2 + outstanding A/Bs BEFORE v10) -> "
            "never-invent-flags (#518 levers are post-merge) -> NO-FAKE (no argv for an "
            "unresolved composition). Fix: clear the named gate artifacts / merge #518, "
            "then re-run compile_v10_capstone_launch_config()."
        )


@dataclass
class SpecV10CompileReport:
    """Pure status surface (usable before gates clear, e.g. by dashboards/costate)."""

    post_merge_levers_resolved: dict[str, bool] = field(default_factory=dict)
    gate_artifacts_present: dict[str, bool] = field(default_factory=dict)
    seed_artifacts_present: dict[str, bool] = field(default_factory=dict)
    blockers: list[dict[str, str]] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return not self.blockers


def _probe_post_merge_levers(blockers: list[dict[str, str]]) -> dict[str, bool]:
    resolved: dict[str, bool] = {}
    try:
        from tac.witness_dsl import curriculum_dsl as _dsl
    except Exception as exc:  # pragma: no cover - import environment failure
        blockers.append({
            "id": "curriculum_dsl_import",
            "detail": f"cannot import tac.witness_dsl.curriculum_dsl: {exc!r}",
        })
        return dict.fromkeys(POST_MERGE_518_LEVERS, False)
    for name in POST_MERGE_518_LEVERS:
        ok = hasattr(_dsl, name)
        resolved[name] = ok
        if not ok:
            blockers.append({
                "id": f"post_merge_518:{name}",
                "detail": f"Lever factory {name!r} not present in curriculum_dsl — #518 branch "
                          "not merged. NEVER fake-resolve (SPEC P5/§8, review finding F6).",
            })
    return resolved


def _probe_artifacts(spec: dict[str, dict[str, str]], kind: str,
                     blockers: list[dict[str, str]], repo_root: Path) -> dict[str, bool]:
    present: dict[str, bool] = {}
    for key, meta in spec.items():
        p = repo_root / meta["path"]
        ok = p.exists()
        present[key] = ok
        if not ok:
            blockers.append({
                "id": f"{kind}:{key}",
                "detail": f"missing artifact {meta['path']} — {meta['detail']}",
            })
    return present


def spec_v10_status(repo_root: str | Path = ".") -> SpecV10CompileReport:
    """Compute the full blocker/readiness surface WITHOUT raising ($0, read-only)."""
    root = Path(repo_root)
    report = SpecV10CompileReport()
    report.post_merge_levers_resolved = _probe_post_merge_levers(report.blockers)
    report.gate_artifacts_present = _probe_artifacts(
        GATE_ARTIFACTS, "gate", report.blockers, root)
    report.seed_artifacts_present = _probe_artifacts(
        SEED_ARTIFACTS, "seed", report.blockers, root)
    return report


def compile_v10_capstone_launch_config(repo_root: str | Path = "."):
    """Compile the v10 capstone launch config (pure / $0; never launches).

    Fail-closed: while ANY SPEC §6 gate artifact, §4 seed artifact, or #518 post-merge
    lever is unresolved, raises :class:`SpecV10CompileBlocked` with the typed blocker
    list. When (and only when) the report is clear, the post-gate fold lands the
    flag-for-flag base here (structured init P1, solved+gauge-fixed head P2, range(A)
    render targets P3, content-priced coder P4, #518 birth laws P5, store-nothing
    pose P6, v8 carriers with the A/B-decided basis P7, harvested v9c2 seeds P8) and
    returns the launcher-facing config, exactly as ``spec_c2_surgical_20260716`` does.
    """
    report = spec_v10_status(repo_root)
    if not report.clear:
        raise SpecV10CompileBlocked(report.blockers)
    raise SpecV10CompileBlocked([{
        "id": "post_gate_fold_owed",
        "detail": "all gates report clear but the post-gate flag-for-flag fold has not been "
                  "landed/reviewed yet (SPEC §8: the base config is deferred to the fold "
                  "commit so it is composed from the MEASURED A/B outcomes, never guessed).",
    }])
