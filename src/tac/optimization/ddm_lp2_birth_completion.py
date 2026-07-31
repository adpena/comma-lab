# SPDX-License-Identifier: MIT
"""ddm_lp2 P2 — the typed ``birth_completion`` gate-key PRODUCER (scorer-free).

Consumed BETWEEN rung-1 continuation windows (gc12 §3 rung-1 / §5 seal demand): given the
window's checkpoint sequence of ``topology_per_class`` telemetry rows plus the QA91 erased-lane
inventory, decide whether the Lane birth process has PLATEAUED while real (super-nucleus) markings
remain erased — i.e. the birth WALL has been hit and the rung-1 window loop should STOP extending.

This module is the PRODUCER only. ``birth_completion`` already exists as a gate key in the trainer's
static coverage (task #364); we do NOT register a prefix. The trainer stays sealed (gc12 §5) — the
rung-1 window loop calls this tool EXTERNALLY between windows, from ``telemetry.jsonl``.

Semantics (gc12 §3 / vh1 rows 12/14 / fp1 §4 QA91):
    ``birth_completion`` fires  ⇔  (Lane betti0_realized slope ≤ ε)  ∧  (above-nucleus erasure persists)

- **slope** = OLS fit of Lane ``betti0_realized`` vs gate over the last ``window_gates`` checkpoints
  (rising during the burn: 20→476; birth_tail_slope 8.75 comp/gate at ep399 = STILL RISING).
- **ε (DERIVED, not hardcoded — provenance ladder):**
      ε = t_crit · max(SE_slope_ols, SE_slope_quant)
  where
      SE_slope_ols  = sqrt( (Σresid² / (W−2)) / S_xx )     [OLS slope standard error, window-data-driven]
      SE_slope_quant= sqrt( (1/12) / S_xx )                [integer-count rounding-noise FLOOR, var 1/12]
      S_xx          = Σ (gate_i − mean_gate)²
      t_crit        = one-sided Student-t critical value at α (dof = W−2)
  ε is the statistical band inside which the fitted birth slope is NOT significantly positive. It is
  DERIVED per-window from the data + the integer nature of a component count; the ONLY stated input is
  α (the confidence of the "slope not significantly positive" test). The OLS term adapts to the real
  early-burn count churn (e.g. 276→252→269→54); the quantization floor prevents a spuriously-tiny SE
  on a perfectly-linear-but-still-rising window from firing the key.

- **above-nucleus erasure persists:** erased_count = betti0_gt − betti0_realized_endpoint > 0 AND the
  inventory's super-nucleus AREA fraction dominates (> 0.5), so the erased Lane mass is real markings
  under the #315 nucleus law, not GT flicker. ``above_nucleus_erased_estimate`` = round(erased_count ×
  super_nucleus_area_frac) is a DERIVED-ESTIMATE; the EXACT super-nucleus-erased COUNT needs one scorer
  pass (QA91 method_note) and is out of this $0 telemetry-only producer's scope.

Determinism: pure numpy/scipy; no RNG. Fail-closed: malformed/missing telemetry raises
``BirthCompletionTelemetryError`` (refuse to emit a verdict) rather than silently passing.

Pointer 0.1910828242 [contest-CPU] UNMOVED. This is apparatus — no score claim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as _sci_stats

SCHEMA = "ddm_lp2_birth_completion_key.v1"
GATE_KEY = "birth_completion"  # already in trainer static coverage (task #364); PRODUCER, not a new prefix
CLASS_NAME = "Lane"

# Lane class index in the comma10k CANONICAL order [Road, Lane, Undrivable, Movable, MyCar].
# NOT the luma sort (CLAUDE.md class-order law). Validated at read time against the inventory's
# betti0_gt_lane (985) so a wrong index or a schema reorder fails-closed rather than silently mislabels.
LANE_CLASS_INDEX = 1

# --- MEASURED-ANCHOR constants (provenance: QA91 telemetry cadence + fp1 5-gate slope) ---
DEFAULT_WINDOW_GATES = 5  # fp1/QA91 birth_tail_slope is over the last 5 gates
DEFAULT_EPOCHS_PER_GATE = 10  # topology rows emitted every 10 epochs (verified ep 9,19,29,...,399)
MIN_WINDOW_POINTS = 3  # need dof = W-2 >= 1 for an OLS slope SE

# --- STATED-CONFIDENCE constant (provenance: class-4 stated value; re-derive trigger below) ---
# One-sided one-sigma non-significance test for a POSITIVE birth slope (Phi(1)=0.8413 -> alpha=0.1587).
# This is the sole non-derived scalar; it is a stated statistical confidence, not a tuned magic number.
# Re-derivation trigger: a change to the count-noise model, or an operator choice of a different band.
DEFAULT_ALPHA_ONE_SIDED = 0.15866

_QUANT_ROUNDING_VAR = 1.0 / 12.0  # variance of uniform rounding noise on an integer count


class BirthCompletionTelemetryError(ValueError):
    """Raised (fail-closed) on missing/malformed telemetry or inventory."""


@dataclass(frozen=True)
class Qa91Inventory:
    """The erased-lane inventory (QA91 schema ``ddm_fp1_qa91_erased_lane.v1``) this producer consumes."""

    betti0_gt_lane: int
    super_nucleus_area_frac: float
    nucleus_threshold_px: int
    source_schema: str
    lane_class_index: int = LANE_CLASS_INDEX


@dataclass(frozen=True)
class WindowFit:
    """The OLS slope fit + derived-epsilon internals over the assessment window."""

    n_points: int
    slope_comp_per_gate: float
    se_slope_ols: float
    se_slope_quant: float
    se_slope: float
    s_xx: float
    t_crit: float
    t_stat: float
    epsilon_comp_per_gate: float
    alpha_one_sided: float


@dataclass(frozen=True)
class BirthCompletionVerdict:
    """Typed, schema-versioned ``birth_completion`` gate-key row."""

    schema: str
    gate_key: str
    class_name: str
    fired: bool
    slope_le_epsilon: bool
    above_nucleus_erasure_persists: bool
    fit: WindowFit
    window_epochs: tuple[int, ...]
    window_betti0_realized: tuple[int, ...]
    epochs_per_gate: int
    betti0_gt_lane: int
    betti0_realized_endpoint: int
    erased_count: int
    super_nucleus_area_frac: float
    above_nucleus_erased_estimate: int
    nucleus_threshold_px: int
    provenance: dict[str, str]
    score_claim: bool = False
    evidence_axis: str = "[apparatus — telemetry-only, no scorer]"
    pointer: str = "0.1910828242 [contest-CPU] UNMOVED"


_PROVENANCE: dict[str, str] = {
    "slope_comp_per_gate": "DERIVED (OLS fit of betti0_realized[Lane] vs gate over the window)",
    "se_slope_ols": "DERIVED (OLS slope standard error from window residuals; dof = W-2)",
    "se_slope_quant": "DERIVED (integer-count rounding-noise floor; var 1/12 over S_xx)",
    "epsilon_comp_per_gate": "DERIVED = t_crit * max(se_slope_ols, se_slope_quant)",
    "alpha_one_sided": (
        "STATED-CONFIDENCE (class-4): one-sided one-sigma non-significance test for a positive birth "
        "slope; re-derivation trigger = count-noise-model change or operator band choice"
    ),
    "window_gates_and_epochs_per_gate": (
        "MEASURED-ANCHOR: fp1/QA91 5-gate birth_tail_slope + 10-epoch topology cadence"
    ),
    "above_nucleus_erased_estimate": (
        "DERIVED-ESTIMATE = round(erased_count * super_nucleus_area_frac); EXACT super-nucleus-erased "
        "COUNT needs one scorer pass (QA91 method_note) and is out of this $0 producer's scope"
    ),
    "betti0_gt_lane / super_nucleus_area_frac / nucleus_threshold_px": (
        "MEASURED-ANCHOR (QA91 inventory ddm_fp1_qa91_erased_lane.v1)"
    ),
    "lane_class_index": (
        "comma10k CANONICAL order [Road,Lane,Undrivable,Movable,MyCar] -> Lane=1; validated vs "
        "inventory betti0_gt_lane (fail-closed on mismatch)"
    ),
}


def load_qa91_inventory(path: str | Path) -> Qa91Inventory:
    """Load + validate the QA91 erased-lane inventory JSON (fail-closed)."""

    p = Path(path)
    if not p.is_file():
        raise BirthCompletionTelemetryError(f"QA91 inventory not found: {p}")
    try:
        raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise BirthCompletionTelemetryError(f"QA91 inventory unreadable: {p}: {exc}") from exc
    schema = str(raw.get("schema", ""))
    if not schema.startswith("ddm_fp1_qa91_erased_lane"):
        raise BirthCompletionTelemetryError(
            f"unexpected QA91 inventory schema {schema!r}; expected ddm_fp1_qa91_erased_lane.*"
        )
    endpoint = raw.get("burn_endpoint_topology")
    if not isinstance(endpoint, dict) or "betti0_gt_lane" not in endpoint:
        raise BirthCompletionTelemetryError("QA91 inventory missing burn_endpoint_topology.betti0_gt_lane")
    try:
        betti0_gt_lane = int(endpoint["betti0_gt_lane"])
        super_frac = float(raw["super_nucleus_area_frac"])
        nucleus_px = int(raw["nucleus_threshold_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BirthCompletionTelemetryError(f"QA91 inventory malformed numeric field: {exc}") from exc
    if betti0_gt_lane <= 0:
        raise BirthCompletionTelemetryError("QA91 betti0_gt_lane must be positive")
    if not (0.0 <= super_frac <= 1.0):
        raise BirthCompletionTelemetryError("QA91 super_nucleus_area_frac must be in [0,1]")
    return Qa91Inventory(
        betti0_gt_lane=betti0_gt_lane,
        super_nucleus_area_frac=super_frac,
        nucleus_threshold_px=nucleus_px,
        source_schema=schema,
    )


def read_lane_betti0_trend(
    telemetry_jsonl: str | Path,
    inventory: Qa91Inventory,
    *,
    lane_class_index: int | None = None,
) -> list[tuple[int, int]]:
    """Parse ``telemetry.jsonl`` topology rows -> sorted ``[(epoch, betti0_realized_lane), ...]``.

    Reads ONLY rows carrying a ``topology_per_class`` dict with ``betti0_gt`` + ``betti0_realized``
    5-vectors (comma10k order). Validates ``betti0_gt[Lane] == inventory.betti0_gt_lane`` so a wrong
    class index or a schema reorder fails-closed. Fail-closed on any malformed row.
    """

    idx = inventory.lane_class_index if lane_class_index is None else lane_class_index
    p = Path(telemetry_jsonl)
    if not p.is_file():
        raise BirthCompletionTelemetryError(f"telemetry.jsonl not found: {p}")
    points: dict[int, int] = {}
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BirthCompletionTelemetryError(f"telemetry line {lineno} not JSON: {exc}") from exc
            topo = row.get("topology_per_class")
            if not isinstance(topo, dict):
                continue  # non-topology telemetry row; skip
            if "epoch" not in row:
                raise BirthCompletionTelemetryError(f"topology row {lineno} missing 'epoch'")
            gt = topo.get("betti0_gt")
            realized = topo.get("betti0_realized")
            if not (isinstance(gt, list) and isinstance(realized, list)):
                raise BirthCompletionTelemetryError(
                    f"topology row {lineno} missing betti0_gt/betti0_realized lists"
                )
            if not (0 <= idx < len(gt)) or not (0 <= idx < len(realized)):
                raise BirthCompletionTelemetryError(
                    f"topology row {lineno} lane index {idx} out of range (len {len(gt)}/{len(realized)})"
                )
            gt_lane = gt[idx]
            realized_lane = realized[idx]
            if int(gt_lane) != inventory.betti0_gt_lane:
                raise BirthCompletionTelemetryError(
                    f"topology row {lineno} betti0_gt[Lane]={gt_lane} != inventory {inventory.betti0_gt_lane}; "
                    "class-index/schema mismatch (fail-closed)"
                )
            epoch = int(row["epoch"])
            rv = int(realized_lane)
            if rv < 0:
                raise BirthCompletionTelemetryError(f"topology row {lineno} negative betti0_realized {rv}")
            points[epoch] = rv  # last write wins on duplicate epoch (resume overlap)
    if not points:
        raise BirthCompletionTelemetryError(f"no topology_per_class rows in {p}")
    return sorted(points.items())


def _ols_slope_fit(gates: np.ndarray, counts: np.ndarray) -> tuple[float, float, float, float, int]:
    """Return (slope, se_slope_ols, s_xx, ssr, dof) for OLS of counts vs gates."""

    n = int(gates.size)
    dof = n - 2
    gate_mean = float(gates.mean())
    s_xx = float(((gates - gate_mean) ** 2).sum())
    if s_xx <= 0.0:
        raise BirthCompletionTelemetryError("degenerate window: all gate values identical (S_xx=0)")
    count_mean = float(counts.mean())
    s_xy = float(((gates - gate_mean) * (counts - count_mean)).sum())
    slope = s_xy / s_xx
    intercept = count_mean - slope * gate_mean
    resid = counts - (intercept + slope * gates)
    ssr = float((resid**2).sum())
    if dof >= 1:
        s2 = ssr / dof
        se_ols = float(np.sqrt(s2 / s_xx))
    else:  # pragma: no cover - guarded by MIN_WINDOW_POINTS
        se_ols = 0.0
    return slope, se_ols, s_xx, ssr, dof


def evaluate_birth_completion(
    trend_points: Sequence[tuple[int, int]],
    inventory: Qa91Inventory,
    *,
    window_gates: int = DEFAULT_WINDOW_GATES,
    epochs_per_gate: int = DEFAULT_EPOCHS_PER_GATE,
    alpha_one_sided: float = DEFAULT_ALPHA_ONE_SIDED,
) -> BirthCompletionVerdict:
    """Compute the typed ``birth_completion`` verdict from a Lane betti0 trend + QA91 inventory."""

    if epochs_per_gate <= 0:
        raise BirthCompletionTelemetryError("epochs_per_gate must be positive")
    if not (0.0 < alpha_one_sided < 0.5):
        raise BirthCompletionTelemetryError("alpha_one_sided must be in (0, 0.5)")
    if window_gates < MIN_WINDOW_POINTS:
        raise BirthCompletionTelemetryError(
            f"window_gates {window_gates} < MIN_WINDOW_POINTS {MIN_WINDOW_POINTS}"
        )

    ordered = sorted(trend_points, key=lambda t: t[0])
    epochs_all = [int(e) for e, _ in ordered]
    if len(set(epochs_all)) != len(epochs_all):
        raise BirthCompletionTelemetryError("duplicate epochs in trend_points")
    window = ordered[-window_gates:]
    if len(window) < MIN_WINDOW_POINTS:
        raise BirthCompletionTelemetryError(
            f"insufficient telemetry: {len(window)} window points < {MIN_WINDOW_POINTS} "
            "(rung-1 windows are ~120-150 ep = ~12-15 gates; refuse to emit a verdict)"
        )

    epochs = np.array([e for e, _ in window], dtype=np.float64)
    counts = np.array([c for _, c in window], dtype=np.float64)
    gates = epochs / float(epochs_per_gate)

    slope, se_ols, s_xx, _ssr, dof = _ols_slope_fit(gates, counts)
    se_quant = float(np.sqrt(_QUANT_ROUNDING_VAR / s_xx))
    se_slope = max(se_ols, se_quant)
    t_crit = float(_sci_stats.t.ppf(1.0 - alpha_one_sided, dof))
    epsilon = t_crit * se_slope
    t_stat = slope / se_slope if se_slope > 0.0 else float("inf")

    betti0_realized_endpoint = int(window[-1][1])
    erased_count = inventory.betti0_gt_lane - betti0_realized_endpoint
    if erased_count < 0:
        raise BirthCompletionTelemetryError(
            f"betti0_realized_endpoint {betti0_realized_endpoint} exceeds betti0_gt "
            f"{inventory.betti0_gt_lane} (impossible; fail-closed)"
        )
    above_estimate = round(erased_count * inventory.super_nucleus_area_frac)
    super_nucleus_dominates = inventory.super_nucleus_area_frac > 0.5
    persists = bool(erased_count > 0 and super_nucleus_dominates and above_estimate >= 1)

    slope_le_epsilon = bool(slope <= epsilon)
    fired = bool(slope_le_epsilon and persists)

    fit = WindowFit(
        n_points=len(window),
        slope_comp_per_gate=slope,
        se_slope_ols=se_ols,
        se_slope_quant=se_quant,
        se_slope=se_slope,
        s_xx=s_xx,
        t_crit=t_crit,
        t_stat=t_stat,
        epsilon_comp_per_gate=epsilon,
        alpha_one_sided=alpha_one_sided,
    )
    return BirthCompletionVerdict(
        schema=SCHEMA,
        gate_key=GATE_KEY,
        class_name=CLASS_NAME,
        fired=fired,
        slope_le_epsilon=slope_le_epsilon,
        above_nucleus_erasure_persists=persists,
        fit=fit,
        window_epochs=tuple(int(e) for e, _ in window),
        window_betti0_realized=tuple(int(c) for _, c in window),
        epochs_per_gate=epochs_per_gate,
        betti0_gt_lane=inventory.betti0_gt_lane,
        betti0_realized_endpoint=betti0_realized_endpoint,
        erased_count=erased_count,
        super_nucleus_area_frac=inventory.super_nucleus_area_frac,
        above_nucleus_erased_estimate=above_estimate,
        nucleus_threshold_px=inventory.nucleus_threshold_px,
        provenance=dict(_PROVENANCE),
    )


def verdict_to_row(verdict: BirthCompletionVerdict) -> dict[str, Any]:
    """Serialize the verdict to a canonical JSON-able dict."""

    return asdict(verdict)


def evaluate_from_paths(
    telemetry_jsonl: str | Path,
    qa91_inventory_json: str | Path,
    *,
    window_gates: int = DEFAULT_WINDOW_GATES,
    epochs_per_gate: int = DEFAULT_EPOCHS_PER_GATE,
    alpha_one_sided: float = DEFAULT_ALPHA_ONE_SIDED,
) -> BirthCompletionVerdict:
    """End-to-end convenience: load inventory + trend from disk, return the verdict."""

    inventory = load_qa91_inventory(qa91_inventory_json)
    trend = read_lane_betti0_trend(telemetry_jsonl, inventory)
    return evaluate_birth_completion(
        trend,
        inventory,
        window_gates=window_gates,
        epochs_per_gate=epochs_per_gate,
        alpha_one_sided=alpha_one_sided,
    )
