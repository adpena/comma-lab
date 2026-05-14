#!/usr/bin/env python3
"""Rank composition cells by EV/$ using posterior + alien-tech ΔS bands.

Per the floor-v3 commit ``27a7950fd`` requirement to identify highest-EV
compositions for the next dispatch wave. The output ranking informs the
next dispatch wave (operator decides which cells to fire).

Inputs (all loaded fresh from canonical sources):

1. ``tac.composition.enumerate.enumerate_cells`` — the full (substrate ×
   primitive × order) cell matrix.
2. ``.omx/state/continual_learning_posterior.json`` — per-substrate
   verified anchors (architecture_class → score_value rows). The tool
   uses anchor count + recent score deltas as a probability-the-prediction-
   holds estimate.
3. Predicted-ΔS bands per substrate name from alien-tech memos —
   defaulted to a baked-in conservative band table (operator can override
   via ``--alien-tech-bands-json``).

Per-cell EV/$ formula::

    EV_score    = abs(predicted_score_delta_band_midpoint × posterior_correction)
    p_holds     = sigmoid(anchor_count, midpoint=2, slope=1.0)
                  × (axis_multiplier_at_PR106_r2)
    EV_per_USD  = (EV_score × p_holds) / max(estimated_dispatch_cost_usd, 0.05)

Output: markdown report at the operator-specified path. Each row has cell
name + predicted ΔS band + estimated cost + EV/$ + readiness (L0/L1/L2/L3).

Per CLAUDE.md "Forbidden score claims" + "Forbidden empirical-claim-
without-evidence-tag", every emitted ranking row carries the tag
``[predicted; substrate × primitive matrix v1 × posterior reweight]`` and
``score_claim=False``. The report is operator-decision input, NOT a
score-promotion artifact.

Cross-references
----------------
- :mod:`tac.composition.enumerate` — full matrix.
- :mod:`tac.continual_learning` — posterior anchors.
- :mod:`tac.optimization.substrate_composition_matrix` — substrate
  axis taxonomy (``ScoreAxis``).
- ``tools/build_composition_ranking_json.py`` — sister tool that emits
  autopilot-consumable JSON; THIS tool emits operator-facing markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.composition.enumerate import enumerate_cells  # noqa: E402
from tac.composition.registry import (  # noqa: E402
    CompositionCell,
    canonical_primitive_inventory,
)
from tac.optimization.substrate_composition_matrix import (  # noqa: E402
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    canonical_substrate_inventory,
)


RANK_SCHEMA_VERSION = "tac_composition_cell_ev_ranking_v1"


# ── PR106-r2 axis-marginal weights (CLAUDE.md "operating-point dependent") ─


# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent",
# at PR106 r2 frontier the marginal-value ranking is POSE > SEG > RATE
# (2.71x pose-marginal vs SegNet's 1.0). The MIXED axis takes the average.
DEFAULT_AXIS_WEIGHTS_PR106_R2: dict[str, float] = {
    "pose": 2.71,
    "seg": 1.00,
    "rate": 0.50,
    "mixed": 1.50,
}


# ── Default alien-tech ΔS bands per substrate name ─────────────────────────


# These are CONSERVATIVE prior bands derived from the 2026-05-13 alien-tech
# expert-team memos (aerospace-stealth + signal-processing + ancient-elder +
# zen-state + Fields-medalist) cross-referenced against observed substrate
# anchors. The operator can override via --alien-tech-bands-json with a JSON
# mapping {substrate_id_or_token: {"low": float, "high": float, "source": str}}.
#
# Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", every band
# carries an evidence tag in its source field. None are score claims.

DEFAULT_ALIEN_TECH_BANDS: dict[str, dict[str, Any]] = {
    # HNeRV-family substrates (PR101 GOLD lineage).
    "sane_hnerv": {
        "low": -0.005, "high": -0.001,
        "source": "[predicted; sane_hnerv parity baseline]",
    },
    "balle_renderer": {
        "low": -0.012, "high": -0.003,
        "source": "[predicted; Ballé 2018 hyperprior + JSCC analog]",
    },
    "hybrid_renderer_residual": {
        "low": -0.018, "high": -0.004,
        "source": "[predicted; HNeRV + residual stack]",
    },
    "self_compress_nn": {
        "low": -0.015, "high": -0.003,
        "source": "[predicted; SC++ block-FP + Hessian quant]",
    },
    "pr101_lc_v2_clone": {
        "low": -0.008, "high": -0.001,
        "source": "[predicted; PR101 lossy-coarsening v2]",
    },
    # NeRV-family substrates.
    "tc_nerv_substrate": {
        "low": -0.006, "high": -0.001,
        "source": "[predicted; TC-NeRV temporal-conditioning]",
    },
    "block_nerv_substrate": {
        "low": -0.005, "high": -0.001,
        "source": "[predicted; Block-NeRV]",
    },
    "ff_nerv_substrate": {
        "low": -0.006, "high": -0.001,
        "source": "[predicted; FF-NeRV flow-field]",
    },
    "ds_nerv_substrate": {
        "low": -0.005, "high": -0.001,
        "source": "[predicted; DS-NeRV]",
    },
    "hi_nerv_substrate": {
        "low": -0.007, "high": -0.001,
        "source": "[predicted; Hi-NeRV hierarchical]",
    },
    # Residual basis substrates.
    "wavelet_residual": {
        "low": -0.0015, "high": -0.0002,
        "source": "[predicted; Mallat 1989 wavelet residual]",
    },
    "cool_chic_residual": {
        "low": -0.002, "high": -0.0003,
        "source": "[predicted; Cool-Chic hierarchical]",
    },
    "c3_residual": {
        "low": -0.003, "high": -0.0005,
        "source": "[predicted; C3 + temporal hyperprior]",
    },
    "siren_substrate": {
        "low": -0.004, "high": -0.0008,
        "source": "[predicted; SIREN]",
    },
    # Pose-axis sidechannels (high marginal value at PR106 r2 per axis weight).
    "raft_pose_sidecar": {
        "low": -0.025, "high": -0.005,
        "source": "[predicted; RAFT optical-flow pose sidecar]",
    },
    "lapose_atom_sidecar": {
        "low": -0.030, "high": -0.006,
        "source": "[predicted; LA-Pose atomic sidecar]",
    },
    "foveation_field_sidecar": {
        "low": -0.020, "high": -0.004,
        "source": "[predicted; foveation field]",
    },
    # Self-compression substrates.
    "scpp_substrate": {
        "low": -0.012, "high": -0.002,
        "source": "[predicted; SC++ self-compression]",
    },
    # Magic codec.
    "magic_codec_pr106_r2": {
        "low": -0.005, "high": -0.001,
        "source": "[predicted; magic codec auto-selector]",
    },
    "sar_coherent_pose_pairs_substrate": {
        "low": -0.006, "high": -0.002,
        "source": "[predicted; Lincoln Lab SAR coherent pose integration]",
    },
    "wyner_ziv_cooperative_receiver_substrate": {
        "low": -0.050, "high": -0.015,
        "source": "[predicted; Slepian-Wolf/Wyner-Ziv cooperative receiver]",
    },
}


# Default conservative band for any substrate not in the table.
GENERIC_ALIEN_TECH_BAND: dict[str, Any] = {
    "low": -0.002, "high": -0.0002,
    "source": "[predicted; generic conservative band]",
}


# Per-substrate baseline dispatch cost band (USD). The operator can override
# via --substrate-cost-json. Defaults are coarse (Modal A100 ~$0.50-2.00 for
# a smoke; full training $5-20).
DEFAULT_SUBSTRATE_COST_USD: dict[str, float] = {
    "sane_hnerv": 1.50,
    "balle_renderer": 2.00,
    "hybrid_renderer_residual": 1.80,
    "self_compress_nn": 1.20,
    "pr101_lc_v2_clone": 0.50,
    "tc_nerv_substrate": 1.50,
    "block_nerv_substrate": 1.50,
    "ff_nerv_substrate": 1.80,
    "ds_nerv_substrate": 1.50,
    "hi_nerv_substrate": 2.00,
    "wavelet_residual": 0.30,
    "cool_chic_residual": 0.40,
    "c3_residual": 0.50,
    "siren_substrate": 0.80,
    "raft_pose_sidecar": 0.30,
    "lapose_atom_sidecar": 0.40,
    "foveation_field_sidecar": 0.30,
    "scpp_substrate": 1.00,
    "magic_codec_pr106_r2": 0.20,
    "sar_coherent_pose_pairs_substrate": 1.00,
    "wyner_ziv_cooperative_receiver_substrate": 1.50,
}


GENERIC_SUBSTRATE_COST_USD = 1.00


# ── Posterior consumption ────────────────────────────────────────────────


@dataclass
class PosteriorSummary:
    """Per-substrate / per-architecture-class anchor count + recent score."""

    architecture_class: str
    n_authoritative_anchors: int
    n_advisory_anchors: int
    most_recent_authoritative_score: float | None
    most_recent_authoritative_tag: str | None


def _anchor_score_value(row: dict[str, Any]) -> float | None:
    score = row.get("score_value")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        return None
    if score_f is None or not math.isfinite(score_f):
        return None
    return score_f


def _anchor_axis(row: dict[str, Any]) -> str | None:
    for key in ("score_axis", "axis", "device"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        token = value.strip().lower().replace("-", "_")
        if token in {"cuda", "contest_cuda"}:
            return "cuda"
        if token in {"cpu", "contest_cpu"}:
            return "cpu"
    return None


def _first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        for key in keys:
            value = provenance.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _anchor_is_linux_x86_64(row: dict[str, Any]) -> bool:
    system = _first_string(
        row,
        ("platform_system", "provenance_platform_system", "os", "runner_os"),
    ).lower()
    machine = _first_string(
        row,
        (
            "platform_machine",
            "provenance_platform_machine",
            "machine",
            "arch",
            "architecture",
            "runner_arch",
        ),
    ).lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return True
    hardware = _first_string(row, ("hardware", "runner", "host", "platform")).lower()
    return "linux" in hardware and ("x86_64" in hardware or "amd64" in hardware)


def _is_authoritative_anchor_row(row: dict[str, Any]) -> bool:
    """Return True only for score-axis-explicit contest authority rows."""
    if _anchor_score_value(row) is None:
        return False
    tag = str(row.get("evidence_tag") or "")
    axis = _anchor_axis(row)
    if tag == "[contest-CUDA]":
        return axis == "cuda"
    if tag in ("[contest-CPU GHA Linux x86_64]", "[contest-CPU GHA]"):
        return axis == "cpu"
    if tag == "[contest-CPU]":
        return axis == "cpu" and _anchor_is_linux_x86_64(row)
    return False


def load_posterior_summary(
    path: Path | None,
) -> dict[str, PosteriorSummary]:
    """Load .omx/state/continual_learning_posterior.json and summarize.

    Returns a dict keyed by architecture_class. ``path`` defaulting to None
    means "use the canonical .omx/state/continual_learning_posterior.json".
    Missing file returns an empty dict (the ranker still runs with priors).
    """
    if path is None:
        path = REPO_ROOT / ".omx" / "state" / "continual_learning_posterior.json"
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    history = payload.get("accepted_anchor_history") or []
    if not isinstance(history, list):
        return {}

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        if not isinstance(row, dict):
            continue
        cls = row.get("architecture_class")
        if not isinstance(cls, str):
            continue
        by_class[cls].append(row)

    out: dict[str, PosteriorSummary] = {}
    for cls, rows in by_class.items():
        # Sort by observed_at_utc to get the most recent.
        rows_sorted = sorted(
            rows,
            key=lambda r: str(r.get("observed_at_utc") or ""),
            reverse=True,
        )
        # Find authoritative rows. Authority requires both the evidence tag
        # and a matching score-axis field; a bare tag can be stale prose.
        authoritative_score: float | None = None
        authoritative_tag: str | None = None
        n_auth = 0
        n_adv = 0
        for r in rows:
            if _is_authoritative_anchor_row(r):
                n_auth += 1
            else:
                n_adv += 1
        # Look for most-recent authoritative.
        for r in rows_sorted:
            if _is_authoritative_anchor_row(r):
                authoritative_score = _anchor_score_value(r)
                authoritative_tag = str(r.get("evidence_tag") or "")
                break

        out[cls] = PosteriorSummary(
            architecture_class=cls,
            n_authoritative_anchors=n_auth,
            n_advisory_anchors=n_adv,
            most_recent_authoritative_score=authoritative_score,
            most_recent_authoritative_tag=authoritative_tag,
        )
    return out


def _posterior_lookup(
    summary: dict[str, PosteriorSummary],
    substrate_id: str,
) -> PosteriorSummary:
    """Lookup the posterior summary for a substrate, with substring fallback.

    The continual-learning posterior keys by ``architecture_class`` (e.g.
    ``"lane_pr106_latent_sidecar_r2"``) which is not exactly the substrate
    id (e.g. ``"pr106_latent_sidecar"``). We try direct match, then prefix
    match on the substrate_id token.
    """
    if substrate_id in summary:
        return summary[substrate_id]
    for cls in summary:
        if substrate_id in cls or cls.endswith(substrate_id):
            return summary[cls]
    # No anchor found.
    return PosteriorSummary(
        architecture_class=substrate_id,
        n_authoritative_anchors=0,
        n_advisory_anchors=0,
        most_recent_authoritative_score=None,
        most_recent_authoritative_tag=None,
    )


# ── Readiness gate from lane registry ────────────────────────────────────


def load_lane_levels(
    path: Path | None = None,
) -> dict[str, int]:
    """Load .omx/state/lane_registry.json and return {substrate_id_token: level}."""
    if path is None:
        path = REPO_ROOT / ".omx" / "state" / "lane_registry.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, int] = {}
    lanes = payload.get("lanes") or []
    if not isinstance(lanes, list):
        return {}
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = lane.get("lane_id") or lane.get("id")
        level = lane.get("level")
        if isinstance(lane_id, str) and isinstance(level, int):
            out[lane_id] = level
    return out


def _lane_level_for_substrate(
    levels: dict[str, int],
    substrate_id: str,
) -> int:
    """Coarse mapping from substrate_id → max lane level.

    Returns the MAX level across any lane whose ``lane_id`` contains the
    substrate id token. 0 if no such lane exists.
    """
    best = 0
    for lane_id, lvl in levels.items():
        if substrate_id in lane_id:
            best = max(best, lvl)
    return best


# ── Ranking ───────────────────────────────────────────────────────────────


def _band_midpoint(low: float, high: float) -> float:
    return (low + high) / 2.0


def _sigmoid(x: float, midpoint: float = 2.0, slope: float = 1.0) -> float:
    """Logistic sigmoid mapping anchor_count → probability the prediction holds.

    midpoint=2 means with 2 authoritative anchors p_holds == 0.5; slope=1.0
    means linear-ish growth. Clamped to [0.05, 0.95] to prevent zero-EV
    rows from being silently dropped.
    """
    z = slope * (x - midpoint)
    p = 1.0 / (1.0 + math.exp(-z))
    return max(0.05, min(0.95, p))


def _axis_weight(
    substrate: SubstrateRow,
    weights: dict[str, float],
) -> float:
    axis_value = (
        substrate.target_axis.value
        if hasattr(substrate.target_axis, "value")
        else str(substrate.target_axis)
    )
    return float(weights.get(axis_value, 1.0))


def _alien_tech_band_for(
    substrate_id: str,
    substrate_class: SubstrateClass,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    if substrate_id in overrides:
        return overrides[substrate_id]
    if substrate_id in DEFAULT_ALIEN_TECH_BANDS:
        return DEFAULT_ALIEN_TECH_BANDS[substrate_id]
    # Substrate-class fallback band.
    cls_token = (
        substrate_class.value
        if hasattr(substrate_class, "value")
        else str(substrate_class)
    )
    if cls_token in overrides:
        return overrides[cls_token]
    return GENERIC_ALIEN_TECH_BAND


def _cost_for_substrate(
    substrate_id: str,
    overrides: dict[str, float] | None = None,
) -> float:
    overrides = overrides or {}
    if substrate_id in overrides:
        return float(overrides[substrate_id])
    return DEFAULT_SUBSTRATE_COST_USD.get(substrate_id, GENERIC_SUBSTRATE_COST_USD)


@dataclass
class RankedCellRow:
    """One ranked composition cell with EV/$ + readiness."""

    cell_id: str
    substrate_id: str
    substrate_class: str
    primitive_pipeline: list[str]
    composition_order: list[str]
    predicted_score_delta_band_low: float
    predicted_score_delta_band_high: float
    predicted_score_delta_midpoint: float
    posterior_n_authoritative_anchors: int
    posterior_correction_factor: float
    p_holds: float
    estimated_dispatch_cost_usd: float
    axis_weight: float
    ev_per_dollar: float
    lane_level: int
    readiness: str  # "L0" | "L1" | "L2" | "L3"
    alien_tech_source: str
    blockers: list[str] = field(default_factory=list)


def rank_cells(
    *,
    substrates: list[SubstrateRow] | None = None,
    cells: list[CompositionCell] | None = None,
    posterior_summary: dict[str, PosteriorSummary] | None = None,
    lane_levels: dict[str, int] | None = None,
    alien_tech_band_overrides: dict[str, dict[str, Any]] | None = None,
    substrate_cost_overrides: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
    only_compatible: bool = True,
    only_with_primitives: bool = False,
) -> list[RankedCellRow]:
    """Rank composition cells by EV/$ using posterior + alien-tech bands."""
    s_inv = substrates if substrates is not None else canonical_substrate_inventory()
    if cells is None:
        cells = enumerate_cells(substrates=s_inv, max_primitives_per_cell=2)
    posterior_summary = posterior_summary if posterior_summary is not None else {}
    lane_levels = lane_levels if lane_levels is not None else {}
    weights = axis_weights or DEFAULT_AXIS_WEIGHTS_PR106_R2

    by_id = {s.substrate_id: s for s in s_inv}

    out: list[RankedCellRow] = []
    for cell in cells:
        if only_compatible:
            if cell.compatibility_verdict not in (
                "compatible", "compatible_bare_substrate"
            ):
                continue
        if only_with_primitives and not cell.primitives:
            continue
        substrate = by_id.get(cell.substrate_id)
        if substrate is None:
            continue

        band = _alien_tech_band_for(
            cell.substrate_id, substrate.substrate_class, alien_tech_band_overrides
        )
        low = float(band["low"])
        high = float(band["high"])
        # If the cell stacks primitives on top, adjust the band by adding
        # the cell's own predicted_score_delta (which is the per-primitive
        # aggregated planning prediction). The cell's own prediction is
        # negative-for-improvement in the same convention.
        cell_extra = float(cell.predicted_score_delta)
        band_low = low + cell_extra
        band_high = high + cell_extra
        midpoint = _band_midpoint(band_low, band_high)

        ps = _posterior_lookup(posterior_summary, cell.substrate_id)
        # Posterior correction: with no anchors, factor=1.0. With anchors
        # whose recent score is BELOW the predicted band, factor goes UP.
        correction = 1.0
        if (
            ps.n_authoritative_anchors > 0
            and ps.most_recent_authoritative_score is not None
        ):
            # If we've delivered close to the predicted ΔS, bump correction
            # slightly. Conservative cap [0.5, 1.5].
            anchor = ps.most_recent_authoritative_score
            if anchor < 0.20:  # strong recent score
                correction = 1.2
            elif anchor < 0.25:
                correction = 1.1
            elif anchor > 0.30:
                correction = 0.8
        weighted_midpoint = midpoint * correction
        ev_score = abs(weighted_midpoint)
        p_holds = _sigmoid(ps.n_authoritative_anchors, midpoint=2.0)

        cost_usd = _cost_for_substrate(cell.substrate_id, substrate_cost_overrides)
        if cost_usd <= 0:
            cost_usd = 0.05  # avoid divide-by-zero
        axis_weight = _axis_weight(substrate, weights)
        ev_per_dollar = (ev_score * p_holds * axis_weight) / cost_usd

        lane_level = _lane_level_for_substrate(lane_levels, cell.substrate_id)
        readiness_label = f"L{lane_level}"

        # Blockers: pull from cell + add noise-floor advisory if midpoint
        # is below 1.6e-5 (cross-machine noise floor per F6).
        blockers = list(cell.blockers)
        if abs(weighted_midpoint) < 1.6e-5:
            blockers.append(
                "predicted_delta_below_cross_machine_noise_floor:|ΔS|<1.6e-5; "
                "F6 advisory — cell delta would be dominated by measurement noise"
            )
        if cell.semantic_compatibility_warning is not None:
            blockers.append(
                f"semantic_compatibility_warning: {cell.semantic_compatibility_warning}"
            )
        if lane_level < 2:
            blockers.append(
                f"lane_level_below_L2_exact_eval_not_dispatchable: readiness=L{lane_level}"
            )
        if ps.n_authoritative_anchors <= 0:
            blockers.append(
                "missing_authoritative_anchor_for_rank_dispatch: "
                "rank row is planning signal until a contest-CPU/CUDA anchor exists"
            )

        out.append(
            RankedCellRow(
                cell_id=cell.cell_id,
                substrate_id=cell.substrate_id,
                substrate_class=cell.substrate_class.value,
                primitive_pipeline=list(cell.primitive_ids()),
                composition_order=list(cell.composition_order),
                predicted_score_delta_band_low=band_low,
                predicted_score_delta_band_high=band_high,
                predicted_score_delta_midpoint=weighted_midpoint,
                posterior_n_authoritative_anchors=ps.n_authoritative_anchors,
                posterior_correction_factor=correction,
                p_holds=p_holds,
                estimated_dispatch_cost_usd=cost_usd,
                axis_weight=axis_weight,
                ev_per_dollar=ev_per_dollar,
                lane_level=lane_level,
                readiness=readiness_label,
                alien_tech_source=str(band.get("source") or "<unspecified>"),
                blockers=blockers,
            )
        )

    out.sort(key=lambda r: (bool(r.blockers), -r.ev_per_dollar))
    return out


# ── Markdown rendering ────────────────────────────────────────────────────


def render_markdown(
    rows: list[RankedCellRow],
    *,
    top_k: int = 20,
    notes: str | None = None,
) -> str:
    """Render the ranked-cell list as an operator-facing markdown table."""
    now = datetime.now(timezone.utc).isoformat()
    out: list[str] = []
    out.append(f"# Composition Cell EV Ranking — Top {top_k}")
    out.append("")
    out.append(f"- Generated at: {now}")
    out.append(f"- Schema: {RANK_SCHEMA_VERSION}")
    out.append(f"- Total ranked cells: {len(rows)}")
    out.append("")
    out.append("**CLAUDE.md compliance**: every row is `[predicted; substrate × "
               "primitive matrix v1 × posterior reweight]`. `score_claim=False`. "
               "This is operator decision input, NOT a score-promotion artifact.")
    out.append("")
    out.append("**Per-axis marginal weight (PR106 r2 operating point)**: "
               f"pose=2.71, seg=1.00, rate=0.50, mixed=1.50 "
               "(per CLAUDE.md operating-point-dependent rule).")
    out.append("")
    if notes:
        out.append(f"**Notes**: {notes}")
        out.append("")
    out.append(
        "| Rank | Cell | Substrate | Class | Primitives | Predicted ΔS band "
        "| Cost (USD) | EV/$ | Anchors | Readiness | Source |"
    )
    out.append(
        "|------|------|-----------|-------|------------|-------------------"
        "|------------|------|---------|-----------|--------|"
    )
    for i, r in enumerate(rows[:top_k]):
        prims = ",".join(r.primitive_pipeline) or "(bare)"
        band = (
            f"[{r.predicted_score_delta_band_low:+.5f}, "
            f"{r.predicted_score_delta_band_high:+.5f}]"
        )
        out.append(
            f"| {i + 1} "
            f"| `{r.cell_id}` "
            f"| `{r.substrate_id}` "
            f"| {r.substrate_class} "
            f"| {prims} "
            f"| {band} "
            f"| ${r.estimated_dispatch_cost_usd:.2f} "
            f"| {r.ev_per_dollar:.5f} "
            f"| {r.posterior_n_authoritative_anchors} "
            f"| {r.readiness} "
            f"| {r.alien_tech_source} |"
        )
    out.append("")
    out.append("## Methodology")
    out.append("")
    out.append("Per-cell EV/$:")
    out.append("")
    out.append("```")
    out.append("EV_score    = abs(predicted_ΔS_midpoint × posterior_correction)")
    out.append("p_holds     = sigmoid(n_authoritative_anchors, midpoint=2)")
    out.append("EV_per_$    = (EV_score × p_holds × axis_weight)")
    out.append("              / max(estimated_dispatch_cost_usd, 0.05)")
    out.append("```")
    out.append("")
    out.append("- `predicted_ΔS` band is from the alien-tech expert-team memos "
               "(2026-05-13 wave) cross-referenced against substrate anchors.")
    out.append("- `posterior_correction` ∈ [0.5, 1.5] derived from the most "
               "recent authoritative anchor for the architecture class.")
    out.append("- `p_holds` ∈ [0.05, 0.95] sigmoid in the anchor count.")
    out.append("- `axis_weight` per CLAUDE.md PR106-r2 operating point.")
    out.append("- `readiness` = max lane level across lanes whose `lane_id` "
               "contains the substrate token.")
    out.append("")
    out.append("## Rows with blockers (excerpt)")
    out.append("")
    blocked = [r for r in rows[:top_k] if r.blockers][:10]
    if not blocked:
        out.append("_(top-K rows show no blockers; full list in serialized JSON)_")
    else:
        out.append("| Cell | Blocker(s) |")
        out.append("|------|------------|")
        for r in blocked:
            out.append(f"| `{r.cell_id}` | {'; '.join(r.blockers)} |")
    out.append("")
    return "\n".join(out)


def serialize_ranked_payload(
    rows: list[RankedCellRow],
    *,
    top_k: int,
    posterior_path: Path | None,
    alien_tech_overrides_path: Path | None,
    substrate_cost_overrides_path: Path | None,
) -> dict[str, Any]:
    """JSON-safe serialization of the ranking payload (for sister tools)."""
    return {
        "schema": RANK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_total_cells": len(rows),
        "top_k": top_k,
        "axis_weights_pr106_r2": dict(DEFAULT_AXIS_WEIGHTS_PR106_R2),
        "posterior_path": str(posterior_path) if posterior_path else None,
        "alien_tech_overrides_path": (
            str(alien_tech_overrides_path) if alien_tech_overrides_path else None
        ),
        "substrate_cost_overrides_path": (
            str(substrate_cost_overrides_path)
            if substrate_cost_overrides_path else None
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "predicted_substrate_primitive_matrix_v1_posterior_reweight",
            "no_score_claim_advanced_by_this_artifact",
            "operator_decision_input_only",
            "no_tmp_paths",
        ],
        "ranked_rows": [
            {
                "rank": i + 1,
                "cell_id": r.cell_id,
                "substrate_id": r.substrate_id,
                "substrate_class": r.substrate_class,
                "primitive_pipeline": r.primitive_pipeline,
                "composition_order": r.composition_order,
                "predicted_score_delta_band_low": r.predicted_score_delta_band_low,
                "predicted_score_delta_band_high": r.predicted_score_delta_band_high,
                "predicted_score_delta_midpoint": r.predicted_score_delta_midpoint,
                "posterior_n_authoritative_anchors": r.posterior_n_authoritative_anchors,
                "posterior_correction_factor": r.posterior_correction_factor,
                "p_holds": r.p_holds,
                "estimated_dispatch_cost_usd": r.estimated_dispatch_cost_usd,
                "axis_weight": r.axis_weight,
                "ev_per_dollar": r.ev_per_dollar,
                "lane_level": r.lane_level,
                "readiness": r.readiness,
                "alien_tech_source": r.alien_tech_source,
                "blockers": list(r.blockers),
            }
            for i, r in enumerate(rows)
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rank composition cells by EV/$ using posterior + alien-tech "
            "ΔS bands. Output: markdown report (operator decision input)."
        )
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Where to write the markdown report (under reports/ or .omx/)",
    )
    p.add_argument(
        "--top-k", type=int, default=20,
        help="Top K cells to render in the markdown table (default 20)",
    )
    p.add_argument(
        "--posterior-path", type=Path, default=None,
        help="Path to continual_learning_posterior.json (default canonical)",
    )
    p.add_argument(
        "--lane-registry-path", type=Path, default=None,
        help="Path to lane_registry.json (default canonical)",
    )
    p.add_argument(
        "--alien-tech-bands-json", type=Path, default=None,
        help="Optional override JSON: {substrate_id: {low, high, source}}",
    )
    p.add_argument(
        "--substrate-cost-json", type=Path, default=None,
        help="Optional override JSON: {substrate_id: cost_usd}",
    )
    p.add_argument(
        "--max-primitives-per-cell", type=int, default=2,
        help="Cap on primitive-pipeline length per cell (default 2)",
    )
    p.add_argument(
        "--only-with-primitives", action="store_true",
        help="Drop bare-substrate cells (only show substrate+primitive cells)",
    )
    p.add_argument(
        "--also-write-json", type=Path, default=None,
        help="Optional sibling JSON path for the full ranked payload",
    )
    p.add_argument("--notes", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    posterior_summary = load_posterior_summary(args.posterior_path)
    lane_levels = load_lane_levels(args.lane_registry_path)

    band_overrides: dict[str, dict[str, Any]] | None = None
    if args.alien_tech_bands_json is not None:
        band_overrides = json.loads(
            args.alien_tech_bands_json.read_text(encoding="utf-8")
        )

    cost_overrides: dict[str, float] | None = None
    if args.substrate_cost_json is not None:
        cost_overrides = {
            k: float(v) for k, v in json.loads(
                args.substrate_cost_json.read_text(encoding="utf-8")
            ).items()
        }

    cells = enumerate_cells(max_primitives_per_cell=args.max_primitives_per_cell)
    rows = rank_cells(
        cells=cells,
        posterior_summary=posterior_summary,
        lane_levels=lane_levels,
        alien_tech_band_overrides=band_overrides,
        substrate_cost_overrides=cost_overrides,
        only_with_primitives=args.only_with_primitives,
    )

    md = render_markdown(rows, top_k=args.top_k, notes=args.notes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"WROTE: {args.output}")
    print(f"  total ranked cells: {len(rows)}")
    print(f"  top_k:              {args.top_k}")
    if rows:
        print(f"  best cell:          {rows[0].cell_id}")
        print(f"  best EV/$:          {rows[0].ev_per_dollar:.5f}")

    if args.also_write_json is not None:
        payload = serialize_ranked_payload(
            rows,
            top_k=args.top_k,
            posterior_path=args.posterior_path,
            alien_tech_overrides_path=args.alien_tech_bands_json,
            substrate_cost_overrides_path=args.substrate_cost_json,
        )
        args.also_write_json.parent.mkdir(parents=True, exist_ok=True)
        args.also_write_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  json sibling:       {args.also_write_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
