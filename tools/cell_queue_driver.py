#!/usr/bin/env python3
"""Drive an ORDERED QUEUE of sealed training cells: verify -> claim -> authorize -> launch -> read.

WHY (operator 2026-09-04): on 2026-09-04 MAIN sequenced the ng1/ng2/ng3 cells BY HAND -- seal,
re-root pins, write a bash fire script, claim two lanes, authorize through the chain driver's
functions, launch, poll milestones, adjudicate, repeat.  Three near-identical shell scripts, and
two of the three defects that cost wall-clock that day were defects OF the hand-sequencing:

  * ng2's launch died in 4 s because its sealed config's ``source_pins`` carried working-tree
    paths and the SEALED tree's ``validate_config`` compares pins as whole dicts
    (``seal_validates_only_inside_the_tree_that_fires_it_20260904``).  Every sha and byte count
    was identical; only the paths differed.
  * ng2's done-receipt name collided with ng1's (both ``DONE.json``), and the launcher refuses to
    overwrite an existing receipt.

Both are STRUCTURAL preconditions a driver can check before spending a second of Metal.  This
module is that driver.  It is deliberately narrow, exactly like the QBR1 chain driver it
generalizes: **the queue spec owns every scientific value and every child argv.**  The driver only

  1. verifies each cell's sealed config is valid FOR THE TREE THAT WILL RUN IT (content-identical
     ``source_pins``, checked by running ``verify_inputs()`` inside the sealed tree),
  2. refuses duplicate done-receipt names across the queue,
  3. asks ``tools/cell_admission.py`` whether the machine can take the cell CONCURRENTLY
     (reclaimable-aware memory headroom + measured Metal-contention throughput),
  4. places fresh lane claims, authorizes through ``ddm_qbr1_cell_chain``'s OWN functions
     (never a hand-edited JSON), launches through the canonical detached launcher,
  5. reads milestones against a NAMED CONTROL and writes one verdict row per cell against
     PRE-REGISTERED falsifiers.

MAIN only adjudicates.  ``--dry-run`` performs every verification and prints the exact plan
without placing a claim, writing an authorized config, or launching anything.

RELATION TO ``experiments/ddm_qbr1_cell_chain.py``: that driver runs ONE sealed six-cell fire
order serially.  This one runs an arbitrary ordered queue with governed concurrency.  It IMPORTS
the chain driver's claim/authorize/verify primitives rather than reimplementing them, so the two
cannot drift apart.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO, _REPO / "tools", _REPO / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cell_admission as admission  # noqa: E402
import ddm_qbr1_cell_chain as chain  # noqa: E402
import ddm_reseal_pins_inside_sealed_tree as reseal  # noqa: E402
import measured_peaks  # noqa: E402

QUEUE_SPEC_SCHEMA = "ddm_gv1_cell_queue_spec.v1"
QUEUE_PLAN_SCHEMA = "ddm_gv1_cell_queue_plan.v1"
QUEUE_VERDICT_SCHEMA = "ddm_gv1_cell_queue_verdict.v1"

DEFAULT_CLAIMS = _REPO / ".omx" / "state" / "active_lane_dispatch_claims.md"
DEFAULT_RESERVE_BYTES = 8 * 1024**3

#: Comparison operators a pre-registered falsifier may use.
_OPS = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


class QueueRefusal(RuntimeError):
    """A typed fail-closed queue stop. Mirrors ``chain.ChainRefusal``."""

    def __init__(self, reason: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ddm_gv1_cell_queue_refusal.v1",
            "status": "REFUSED",
            "reason": self.reason,
            "message": str(self),
            **self.detail,
        }


def utc_text(value: dt.datetime | None = None) -> str:
    return admission.utc_text(value)


# ── queue spec ──────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Falsifier:
    """One pre-registered kill condition, declared BEFORE the cell runs."""

    name: str
    at_step: int
    metric: str  # dotted path into a history row, e.g. "objective.seg_expected_flip_realized"
    op: str
    threshold: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Falsifier:
        try:
            op = str(payload["op"])
            if op not in _OPS:
                raise QueueRefusal(
                    "FALSIFIER_OP", f"unknown falsifier op {op!r}; expected one of {sorted(_OPS)}"
                )
            return cls(
                name=str(payload["name"]),
                at_step=int(payload["at_step"]),
                metric=str(payload["metric"]),
                op=op,
                threshold=float(payload["threshold"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueRefusal(
                "FALSIFIER_SHAPE",
                "falsifier needs name/at_step/metric/op/threshold",
                payload=dict(payload),
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def describe(self) -> str:
        return f"{self.name}: {self.metric} @{self.at_step} {self.op} {self.threshold}"


@dataclasses.dataclass(frozen=True)
class QueuedCell:
    """One sealed cell in the queue. Every scientific value comes from the spec, never the driver."""

    cell_id: str
    sealed_config: Path
    sealed_tree: Path
    authorized_config: Path
    launcher_argv: tuple[str, ...]
    done_receipt: str
    scorer_lane_prefix: str
    metal_lane_prefix: str
    #: ``None`` means "read it from the measured-peak ledger" (spec value ``"from_ledger"``).
    #: A NUMBER is a declaration, and a declaration below the family's MEASURED row is REFUSED.
    measured_peak_rss_gib: float | None
    peak_family: str | None
    control_run_dir: Path | None
    control_label: str | None
    milestones: tuple[int, ...]
    falsifiers: tuple[Falsifier, ...]
    notes: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> QueuedCell:
        try:
            argv = tuple(str(item) for item in payload["launcher_argv"])
            control = payload.get("control_run_dir")
            return cls(
                cell_id=str(payload["cell_id"]),
                sealed_config=Path(str(payload["sealed_config"])),
                sealed_tree=Path(str(payload["sealed_tree"])),
                authorized_config=Path(str(payload["authorized_config"])),
                launcher_argv=argv,
                done_receipt=str(payload["done_receipt"]),
                scorer_lane_prefix=str(payload["scorer_lane_prefix"]),
                metal_lane_prefix=str(payload["metal_lane_prefix"]),
                measured_peak_rss_gib=(
                    None
                    if str(payload["measured_peak_rss_gib"]).strip().lower() == "from_ledger"
                    else float(payload["measured_peak_rss_gib"])
                ),
                peak_family=(None if payload.get("peak_family") is None else str(payload["peak_family"])),
                control_run_dir=None if control is None else Path(str(control)),
                control_label=(
                    None if payload.get("control_label") is None else str(payload["control_label"])
                ),
                milestones=tuple(int(m) for m in payload.get("milestones", ())),
                falsifiers=tuple(
                    Falsifier.from_mapping(f) for f in payload.get("falsifiers", ())
                ),
                notes=str(payload.get("notes", "")),
            )
        except QueueRefusal:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueRefusal(
                "CELL_SPEC_SHAPE",
                "queued cell is missing a required field",
                cell_id=payload.get("cell_id"),
            ) from exc


def load_queue_spec(path: Path) -> list[QueuedCell]:
    """Read and structurally validate an ordered queue spec."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueRefusal("QUEUE_SPEC_UNREADABLE", "queue spec is not readable JSON",
                           path=str(path)) from exc
    if not isinstance(payload, dict) or payload.get("schema") != QUEUE_SPEC_SCHEMA:
        raise QueueRefusal(
            "QUEUE_SPEC_SCHEMA",
            f"queue spec must be a JSON object with schema {QUEUE_SPEC_SCHEMA}",
            path=str(path),
        )
    cells_raw = payload.get("cells")
    if not isinstance(cells_raw, list) or not cells_raw:
        raise QueueRefusal("QUEUE_SPEC_EMPTY", "queue spec has no cells", path=str(path))
    cells = [QueuedCell.from_mapping(item) for item in cells_raw]

    seen_receipts: dict[str, str] = {}
    seen_ids: set[str] = set()
    for cell in cells:
        if cell.cell_id in seen_ids:
            raise QueueRefusal("DUPLICATE_CELL_ID", "queue repeats a cell_id", cell_id=cell.cell_id)
        seen_ids.add(cell.cell_id)
        # The ng1/ng2 collision: the launcher refuses to overwrite an existing done receipt.
        if cell.done_receipt in seen_receipts:
            raise QueueRefusal(
                "DUPLICATE_DONE_RECEIPT",
                "two cells share a done-receipt name; the launcher refuses to overwrite one",
                done_receipt=cell.done_receipt,
                cells=[seen_receipts[cell.done_receipt], cell.cell_id],
            )
        seen_receipts[cell.done_receipt] = cell.cell_id
    return cells


# ── per-cell verification ───────────────────────────────────────────────────────────────────────


def verify_sealed_config_in_its_tree(cell: QueuedCell) -> dict[str, Any]:
    """THE SEAL LAW: the config's ``source_pins`` must be valid for the tree that will RUN it.

    Runs ``verify_inputs()`` inside the sealed tree's own interpreter (via the canonical re-root
    tool's helper) and refuses on any content difference OR any path difference.  A path
    difference is not fatal to the science but IS fatal to the launch -- the burn prep compares
    pins as whole dicts -- so it is reported as ``PIN_PATHS_NOT_REROOTED`` with the exact cure
    (``experiments/ddm_reseal_pins_inside_sealed_tree.py``) rather than discovered 4 s into a fire.
    """
    if not cell.sealed_config.is_file():
        raise QueueRefusal(
            "SEALED_CONFIG_MISSING", "sealed config is absent", path=str(cell.sealed_config)
        )
    if not cell.sealed_tree.is_dir():
        raise QueueRefusal(
            "SEALED_TREE_MISSING", "sealed tree is absent", path=str(cell.sealed_tree)
        )
    config = chain.load_json(cell.sealed_config, label=f"{cell.cell_id} sealed config")
    pins = config.get("source_pins")
    if not isinstance(pins, dict) or not pins:
        raise QueueRefusal(
            "SEALED_CONFIG_NO_PINS", "sealed config has no source_pins", cell_id=cell.cell_id
        )
    try:
        live = reseal.verify_inputs_inside(cell.sealed_tree)
    except reseal.ResealError as exc:
        raise QueueRefusal(
            "SEALED_TREE_VERIFY_INPUTS",
            f"verify_inputs() failed inside the sealed tree: {exc}",
            cell_id=cell.cell_id,
            sealed_tree=str(cell.sealed_tree),
        ) from exc

    missing = sorted(set(pins) - set(live))
    extra = sorted(set(live) - set(pins))
    if missing or extra:
        raise QueueRefusal(
            "PIN_KEY_SET_MISMATCH",
            "sealed config and sealed tree disagree on which inputs exist",
            cell_id=cell.cell_id,
            missing_in_tree=missing,
            extra_in_tree=extra,
        )
    content_drift = sorted(
        key
        for key in pins
        if (pins[key].get("sha256"), pins[key].get("bytes"))
        != (live[key].get("sha256"), live[key].get("bytes"))
    )
    if content_drift:
        raise QueueRefusal(
            "PIN_CONTENT_DRIFT",
            "sealed config pins differ in CONTENT from the sealed tree (not a path problem)",
            cell_id=cell.cell_id,
            drifted=content_drift,
        )
    path_drift = sorted(key for key in pins if pins[key].get("path") != live[key].get("path"))
    if path_drift:
        raise QueueRefusal(
            "PIN_PATHS_NOT_REROOTED",
            "pins are content-identical but path-rooted OUTSIDE the firing tree; the burn prep "
            "compares pins as whole dicts and will refuse this launch. Cure: "
            "experiments/ddm_reseal_pins_inside_sealed_tree.py --config-in ... --sealed-tree ...",
            cell_id=cell.cell_id,
            path_drifted=path_drift,
            pins_total=len(pins),
        )
    return {
        "cell_id": cell.cell_id,
        "sealed_config": chain.file_fact(cell.sealed_config),
        "sealed_tree": str(cell.sealed_tree),
        "pins_total": len(pins),
        "content_identical": True,
        "paths_rooted_in_firing_tree": True,
    }


def verify_launcher_argv(cell: QueuedCell) -> dict[str, Any]:
    """The launcher argv must name this cell's authorized config and its distinct done receipt."""
    argv = list(cell.launcher_argv)
    if "--done-receipt" not in argv:
        raise QueueRefusal(
            "ARGV_NO_DONE_RECEIPT", "launcher argv has no --done-receipt", cell_id=cell.cell_id
        )
    declared = argv[argv.index("--done-receipt") + 1]
    if declared != cell.done_receipt:
        raise QueueRefusal(
            "ARGV_DONE_RECEIPT_MISMATCH",
            "launcher argv's --done-receipt differs from the cell's declared receipt",
            cell_id=cell.cell_id,
            argv_value=declared,
            declared=cell.done_receipt,
        )
    if str(cell.authorized_config) not in argv:
        raise QueueRefusal(
            "ARGV_CONFIG_MISMATCH",
            "launcher argv does not reference this cell's authorized config",
            cell_id=cell.cell_id,
            authorized_config=str(cell.authorized_config),
        )
    return {
        "output_dir": str(chain.launcher_output_dir(argv)),
        "done_receipt": cell.done_receipt,
        "argv_len": len(argv),
    }


# ── THE MEASURED-PEAK LAW (ddm_gov2) ────────────────────────────────────────────────────────────


def peak_family_of(cell: QueuedCell) -> str | None:
    """The measured-peak family for this cell: its trainer entry point."""
    if cell.peak_family:
        return cell.peak_family
    for part in cell.launcher_argv:
        text = str(part)
        if text.endswith(".py") and "launch_detached_process" not in text and "safe_run" not in text:
            return Path(text).stem
    return None


def resolve_peak(cell: QueuedCell) -> dict[str, Any]:
    """``{gib, provenance, family, measured}`` -- and REFUSE a hand-typed under-declaration.

    THE DEFECT (MEASURED 2026-09-04): ng2 and ng3 were both launched under a hand-typed
    ``--measured-peak-rss-gib 2.3959503173828125`` carried over from an unrelated run, while the
    same family's measured system-availability cost is **49.572 GiB** -- a 20.7x under-declaration.
    Two cells admitted on that number drove the VM compressor to 76.978 GiB and jetsam killed
    background daemons.  A typed number is no longer allowed to be smaller than what the family has
    been MEASURED to cost.

    A family with NO measured row may still declare a number -- that is the honest bootstrap (run
    the bounded smoke first, as ng4 did), and the row it produces governs every later launch.
    """
    family = peak_family_of(cell)
    measured = measured_peaks.lookup_family(family) if family else None
    governing = None if measured is None else float(measured.get("governed_peak_gib") or 0.0)

    if cell.measured_peak_rss_gib is None:
        if governing is None or governing <= 0.0:
            raise QueueRefusal(
                "PEAK_FROM_LEDGER_BUT_NO_MEASURED_ROW",
                "the cell asks for the ledger's peak but this family has never been measured",
                cell_id=cell.cell_id,
                family=family,
                cure=(
                    "run the family's bounded smoke first (the ng4 pattern), then "
                    "`tools/measured_peaks.py harvest --root <store>`"
                ),
            )
        return {
            "gib": governing,
            "provenance": "FROM_LEDGER",
            "family": family,
            "measured_gib": governing,
            "attribution_grade": measured.get("attribution_grade"),
        }

    declared = float(cell.measured_peak_rss_gib)
    if governing is not None and declared < governing:
        raise QueueRefusal(
            "HAND_TYPED_PEAK_BELOW_MEASURED",
            "the spec declares a memory peak smaller than this family's MEASURED cost",
            cell_id=cell.cell_id,
            family=family,
            declared_peak_gib=declared,
            measured_peak_gib=governing,
            under_declaration_factor=round(governing / declared, 3) if declared > 0 else None,
            attribution_grade=measured.get("attribution_grade"),
            cure='set "measured_peak_rss_gib": "from_ledger" in the queue spec',
        )
    return {
        "gib": declared,
        "provenance": "SPEC_DECLARED_NO_MEASURED_ROW" if governing is None else "SPEC_DECLARED_AT_OR_ABOVE_MEASURED",
        "family": family,
        "measured_gib": governing,
        "attribution_grade": None if measured is None else measured.get("attribution_grade"),
    }


def launcher_argv_with_peak(argv: Sequence[str], peak_gib: float) -> tuple[str, ...]:
    """The launcher argv with ``--measured-peak-rss-gib`` set to the RESOLVED peak.

    The spec owns every scientific value; the memory declaration is the one value the GOVERNOR
    owns, because it is the value the machine's safety depends on and the one that was wrong.
    """
    parts = [str(item) for item in argv]
    for index, item in enumerate(parts):
        if item == "--measured-peak-rss-gib" and index + 1 < len(parts):
            parts[index + 1] = f"{float(peak_gib)}"
            return tuple(parts)
        if item.startswith("--measured-peak-rss-gib="):
            parts[index] = f"--measured-peak-rss-gib={float(peak_gib)}"
            return tuple(parts)
    return tuple(parts)


def admission_for(
    cell: QueuedCell,
    *,
    roots: Sequence[Path] | None = None,
    ledger_path: Path | None = None,
    margin_gib: float = admission.DEFAULT_MARGIN_GIB,
    live_cells: Sequence[admission.LiveCell] | None = None,
) -> admission.AdmissionDecision:
    """Ask the governor whether the machine can take this cell right now, concurrently.

    ``live_cells`` lets a caller planning a whole queue discover the live fleet ONCE and reuse it,
    instead of walking the SSD tiers per queued cell -- and it also makes every cell in one plan
    share a single, consistent snapshot of the machine rather than N snapshots taken seconds apart.
    """
    return admission.decide_admission(
        resolve_peak(cell)["gib"],
        margin_gib=margin_gib,
        roots=roots,
        ledger_path=ledger_path,
        live_cells=live_cells,
    )


# ── milestone reads against a named control ─────────────────────────────────────────────────────


def _dotted(row: Mapping[str, Any], path: str) -> Any:
    node: Any = row
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def read_history_at_step(run_dir: Path, step: int) -> dict[str, Any] | None:
    """The history row AT ``step``, or None when the run has not reached ``step`` yet.

    THE BUG THIS FIXES (caught 2026-09-04 by running ``verdict`` against the LIVE ng3 run, not a
    fixture).  The obvious implementation -- "the last row with ``completed_steps <= step``" --
    silently answers a milestone the run has not reached: ng3 stood at 741 steps and the step-5000
    falsifier came back ``status=EVALUATED, fired=false`` off step-741 data.  That is a FALSE
    SURVIVED: a pre-registered kill condition reported as passed before the cell had any chance to
    trip it.  A verdict apparatus that does this is worse than none.

    The cure is one extra condition: only answer when the run has actually progressed to at least
    ``step``.  Otherwise the caller gets None and the falsifier reports ``NOT_YET_REACHED``.
    """
    history = run_dir / "history.jsonl"
    if not history.is_file():
        return None
    best: dict[str, Any] | None = None
    max_completed = -1
    try:
        with history.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                completed = row.get("completed_steps")
                if not isinstance(completed, int):
                    continue
                max_completed = max(max_completed, completed)
                if completed <= step:
                    best = row
    except OSError:
        return None
    # NOT REACHED: the run's furthest progress is still short of the milestone.
    if max_completed < step:
        return None
    return best


def milestone_reads(
    cell: QueuedCell, run_dir: Path, metrics: Sequence[str]
) -> list[dict[str, Any]]:
    """Read each declared milestone for the cell and for its NAMED CONTROL, side by side.

    A milestone without a control is reported with ``control=None`` and an explicit
    ``comparable=False`` -- a number with nothing to compare it to is not a read.
    """
    rows: list[dict[str, Any]] = []
    for step in cell.milestones:
        treatment = read_history_at_step(run_dir, step)
        control = (
            read_history_at_step(cell.control_run_dir, step)
            if cell.control_run_dir is not None
            else None
        )
        entry: dict[str, Any] = {
            "at_step": step,
            "control_label": cell.control_label,
            "comparable": control is not None,
            "treatment_reached": treatment is not None,
            "metrics": {},
        }
        for metric in metrics:
            t_value = _dotted(treatment, metric) if treatment else None
            c_value = _dotted(control, metric) if control else None
            delta = (
                t_value - c_value
                if isinstance(t_value, (int, float)) and isinstance(c_value, (int, float))
                else None
            )
            entry["metrics"][metric] = {
                "treatment": t_value,
                "control": c_value,
                "delta_vs_control": delta,
            }
        rows.append(entry)
    return rows


def evaluate_falsifiers(cell: QueuedCell, run_dir: Path) -> list[dict[str, Any]]:
    """Evaluate every PRE-REGISTERED falsifier. ``fired=True`` means the cell is falsified."""
    results: list[dict[str, Any]] = []
    for falsifier in cell.falsifiers:
        row = read_history_at_step(run_dir, falsifier.at_step)
        observed = _dotted(row, falsifier.metric) if row else None
        if not isinstance(observed, (int, float)):
            results.append(
                {
                    **falsifier.as_dict(),
                    "observed": None,
                    "fired": None,
                    "status": "NOT_YET_REACHED" if row is None else "METRIC_ABSENT",
                }
            )
            continue
        fired = _OPS[falsifier.op](float(observed), falsifier.threshold)
        results.append(
            {
                **falsifier.as_dict(),
                "observed": float(observed),
                "fired": bool(fired),
                "status": "EVALUATED",
            }
        )
    return results


def cell_verdict(cell: QueuedCell, run_dir: Path, metrics: Sequence[str]) -> dict[str, Any]:
    """One verdict row: milestone reads against the control plus every falsifier's state."""
    falsifiers = evaluate_falsifiers(cell, run_dir)
    fired = [f for f in falsifiers if f.get("fired") is True]
    pending = [f for f in falsifiers if f.get("fired") is None]
    if fired:
        verdict = "FALSIFIED"
    elif pending:
        verdict = "PENDING"
    elif falsifiers:
        verdict = "SURVIVED"
    else:
        verdict = "NO_FALSIFIERS_DECLARED"
    return {
        "schema": QUEUE_VERDICT_SCHEMA,
        "cell_id": cell.cell_id,
        "evaluated_utc": utc_text(),
        "run_dir": str(run_dir),
        "control_run_dir": None if cell.control_run_dir is None else str(cell.control_run_dir),
        "control_label": cell.control_label,
        "verdict": verdict,
        "falsifiers": falsifiers,
        "falsifiers_fired": [f["name"] for f in fired],
        "milestones": milestone_reads(cell, run_dir, metrics),
        "score_claim": False,
        "promotion_eligible": False,
    }


# ── planning ────────────────────────────────────────────────────────────────────────────────────


def plan_cell(
    cell: QueuedCell,
    *,
    roots: Sequence[Path] | None,
    ledger_path: Path | None,
    margin_gib: float,
    reserve_bytes: int,
    verify_seal: bool = True,
    live_cells: Sequence[admission.LiveCell] | None = None,
) -> dict[str, Any]:
    """Everything checkable BEFORE spending Metal, as one structured plan entry."""
    entry: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "notes": cell.notes,
        "measured_peak_rss_gib": cell.measured_peak_rss_gib,
        "done_receipt": cell.done_receipt,
        "resolved_peak": None,
        "falsifiers": [f.describe() for f in cell.falsifiers],
        "milestones": list(cell.milestones),
        "control_label": cell.control_label,
        "blockers": [],
    }
    # THE MEASURED-PEAK LAW runs first: a cell whose declared peak is below its family's measured
    # cost is a blocker, not a launch, and the resolved number is what admission then charges.
    try:
        entry["resolved_peak"] = resolve_peak(cell)
    except QueueRefusal as exc:
        entry["blockers"].append({"stage": "resolve_peak", **exc.as_dict()})
        entry["ready"] = False
        return entry

    for label, check in (
        ("launcher_argv", lambda: verify_launcher_argv(cell)),
        (
            "seal",
            (lambda: verify_sealed_config_in_its_tree(cell))
            if verify_seal
            else (lambda: {"skipped": "verify_seal=False"}),
        ),
        (
            "storage",
            lambda: chain.storage_preflight(
                chain.launcher_output_dir(list(cell.launcher_argv)), reserve_bytes
            ),
        ),
    ):
        try:
            entry[label] = check()
        except (QueueRefusal, chain.ChainRefusal) as exc:
            entry[label] = exc.as_dict()
            entry["blockers"].append(f"{label}:{exc.reason}")

    decision = admission_for(
        cell,
        roots=roots,
        ledger_path=ledger_path,
        margin_gib=margin_gib,
        live_cells=live_cells,
    )
    entry["admission"] = decision.as_dict()
    if not decision.admits:
        entry["blockers"].append("admission:REFUSE")
    entry["ready"] = not entry["blockers"]
    return entry


def plan_queue(
    cells: Sequence[QueuedCell],
    *,
    roots: Sequence[Path] | None = None,
    ledger_path: Path | None = None,
    margin_gib: float = admission.DEFAULT_MARGIN_GIB,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    verify_seal: bool = True,
) -> dict[str, Any]:
    # ONE live-fleet snapshot for the whole plan: walking the SSD tiers per cell is both slow and
    # inconsistent (cell 1 would be judged against a different machine state than cell N).
    live_cells = admission.discover_live_cells(roots)
    entries = [
        plan_cell(
            cell,
            roots=roots,
            ledger_path=ledger_path,
            margin_gib=margin_gib,
            reserve_bytes=reserve_bytes,
            verify_seal=verify_seal,
            live_cells=live_cells,
        )
        for cell in cells
    ]
    ready = [entry["cell_id"] for entry in entries if entry["ready"]]
    return {
        "schema": QUEUE_PLAN_SCHEMA,
        "planned_utc": utc_text(),
        "cells_total": len(entries),
        "cells_ready": len(ready),
        "ready_cell_ids": ready,
        "next_cell_id": ready[0] if ready else None,
        "cells": entries,
        "score_claim": False,
        "actuation": "PLAN_ONLY",
    }


# ── firing (guarded; --dry-run is the tested path) ──────────────────────────────────────────────


def claim_ids(cell: QueuedCell, *, day: str) -> tuple[str, str]:
    return f"{cell.scorer_lane_prefix}_{day}", f"{cell.metal_lane_prefix}_{day}"


def place_claims(
    cell: QueuedCell, *, day: str, claims_path: Path, ttl_hours: float, agent: str
) -> tuple[str, str]:
    """Place the two fresh lane claims through the canonical claim tool (never a hand edit)."""
    scorer_id, metal_id = claim_ids(cell, day=day)
    for lane_id, platform, note in (
        (scorer_id, "local_macos_cpu", cell.notes or f"{cell.cell_id} scorer lane"),
        (metal_id, "local_mlx_metal", f"{cell.cell_id} Metal cell (governed concurrent admission)"),
    ):
        completed = subprocess.run(
            [
                sys.executable, str(_REPO / "tools" / "claim_lane_dispatch.py"), "claim",
                "--lane-id", lane_id,
                "--platform", platform,
                "--instance-job-id", f"{cell.cell_id}_{day}",
                "--agent", agent,
                "--status", "active_eval",
                "--ttl-hours", str(ttl_hours),
                "--claims-path", str(claims_path),
                "--notes", note,
            ],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise QueueRefusal(
                "CLAIM_FAILED",
                "claim_lane_dispatch refused the lane claim",
                cell_id=cell.cell_id,
                lane_id=lane_id,
                stderr=completed.stderr.strip()[-1000:],
            )
    return scorer_id, metal_id


def authorize(cell: QueuedCell, scorer_claim_id: str, metal_claim_id: str) -> dict[str, Any]:
    """Bind the fresh claim ids through the CHAIN DRIVER's own functions, never a hand edit."""
    sealed = chain.load_json(cell.sealed_config, label=f"{cell.cell_id} sealed config")
    expected = chain.authorized_config(sealed, scorer_claim_id, metal_claim_id)
    return chain.write_or_verify_authorized(cell.authorized_config, expected)


def fire_cell(
    cell: QueuedCell,
    *,
    day: str,
    claims_path: Path,
    ttl_hours: float,
    agent: str,
) -> dict[str, Any]:
    """Claim -> authorize -> launch. Callers MUST have planned the cell ready first.

    PHANTOM-CLAIM DISCIPLINE: the claims go in BEFORE the authorize and the launch, so any failure
    after that point leaves two ACTIVE claims for a cell that never ran. Claims cannot be placed
    after the launch (the launch would then be unclaimed), so instead every downstream refusal
    carries ``placed_claims`` and ``claims_need_terminal_row=True`` -- the orphan is REPORTED, never
    silent, and the caller has the exact lane ids to close with
    ``claim_lane_dispatch.py claim --force --status refused_dispatch...``.
    """
    scorer_id, metal_id = place_claims(
        cell, day=day, claims_path=claims_path, ttl_hours=ttl_hours, agent=agent
    )
    placed = {
        "placed_claims": {"scorer": scorer_id, "metal": metal_id},
        "claims_need_terminal_row": True,
        "claims_path": str(claims_path),
    }
    try:
        authorized = authorize(cell, scorer_id, metal_id)
    except (QueueRefusal, chain.ChainRefusal) as exc:
        raise QueueRefusal(
            "AUTHORIZE_FAILED_AFTER_CLAIMS",
            f"authorization failed after the lane claims were placed: {exc}",
            cell_id=cell.cell_id,
            underlying_reason=getattr(exc, "reason", None),
            **placed,
        ) from exc
    try:
        resolved = resolve_peak(cell)
    except QueueRefusal as exc:
        raise QueueRefusal(
            "PEAK_REFUSED_AFTER_CLAIMS",
            f"the measured-peak law refused this cell after the lane claims were placed: {exc}",
            cell_id=cell.cell_id,
            underlying_reason=exc.reason,
            **placed,
        ) from exc
    argv = launcher_argv_with_peak(cell.launcher_argv, resolved["gib"])
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise QueueRefusal(
            "LAUNCH_FAILED",
            "the canonical launcher refused or failed after the lane claims were placed",
            cell_id=cell.cell_id,
            stdout=completed.stdout.strip()[-2000:],
            stderr=completed.stderr.strip()[-2000:],
            **placed,
        )
    return {
        "cell_id": cell.cell_id,
        "fired_utc": utc_text(),
        "scorer_claim_id": scorer_id,
        "metal_claim_id": metal_id,
        "authorized_config": authorized,
        "resolved_peak": resolved,
        "launcher_argv": list(argv),
        "launcher_stdout": completed.stdout.strip()[-2000:],
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_plan(args: argparse.Namespace) -> int:
    cells = load_queue_spec(args.queue)
    plan = plan_queue(
        cells,
        roots=[Path(r) for r in args.root] if args.root else None,
        ledger_path=args.ledger,
        margin_gib=args.margin_gib,
        reserve_bytes=args.reserve_bytes,
        verify_seal=not args.skip_seal_verify,
    )
    _emit(plan)
    return 0 if plan["cells_ready"] else 2


def _cmd_verdict(args: argparse.Namespace) -> int:
    cells = {cell.cell_id: cell for cell in load_queue_spec(args.queue)}
    cell = cells.get(args.cell_id)
    if cell is None:
        _emit({"status": "REFUSED", "reason": "UNKNOWN_CELL", "cell_id": args.cell_id})
        return 2
    verdict = cell_verdict(cell, args.run_dir, args.metric or [])
    _emit(verdict)
    return 0


def _cmd_fire(args: argparse.Namespace) -> int:
    """THE ONE FIRE PATH (ddm_gov2).  Fire one NAMED cell, or the next ready one.

    Nothing else may launch a cell: no bespoke shell script, no hand-run launcher invocation.  The
    STRICT gate ``check_cell_launches_only_through_queue_driver`` enforces that statically; this is
    the surface it points at.  Every launch therefore gets, in order: the seal law, the duplicate-
    receipt check, storage reserve, THE MEASURED-PEAK LAW, memory + concurrency admission, the lane
    claims, the chain driver's authorize, and the canonical launcher -- with the MEASURED peak.
    """
    cells = load_queue_spec(args.queue)
    plan = plan_queue(
        cells,
        roots=[Path(r) for r in args.root] if args.root else None,
        ledger_path=args.ledger,
        margin_gib=args.margin_gib,
        reserve_bytes=args.reserve_bytes,
        verify_seal=not args.skip_seal_verify,
    )
    target_id = args.cell_id or plan["next_cell_id"]
    if target_id is None:
        _emit({**plan, "status": "REFUSED", "reason": "NO_READY_CELL"})
        return 2
    entry = next((row for row in plan["cells"] if row["cell_id"] == target_id), None)
    if entry is None:
        _emit({**plan, "status": "REFUSED", "reason": "UNKNOWN_CELL", "cell_id": target_id})
        return 2
    if not entry.get("ready"):
        _emit({**plan, "status": "REFUSED", "reason": "CELL_NOT_READY", "cell_id": target_id, "entry": entry})
        return 2
    if args.dry_run:
        _emit({**plan, "dry_run": True, "would_fire": target_id, "entry": entry})
        return 0
    target = next(cell for cell in cells if cell.cell_id == target_id)
    try:
        receipt = fire_cell(
            target,
            day=args.day or dt.datetime.now(dt.UTC).strftime("%Y%m%d"),
            claims_path=args.claims_path,
            ttl_hours=args.ttl_hours,
            agent=args.agent,
        )
    except (QueueRefusal, chain.ChainRefusal) as exc:
        _emit(exc.as_dict())
        return 2
    _emit({"schema": "ddm_gv1_cell_queue_fire.v1", "plan": plan, "fired": receipt})
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Fire the next ready cell. ``--dry-run`` stops after planning."""
    cells = load_queue_spec(args.queue)
    plan = plan_queue(
        cells,
        roots=[Path(r) for r in args.root] if args.root else None,
        ledger_path=args.ledger,
        margin_gib=args.margin_gib,
        reserve_bytes=args.reserve_bytes,
        verify_seal=not args.skip_seal_verify,
    )
    if args.dry_run:
        plan["dry_run"] = True
        plan["would_fire"] = plan["next_cell_id"]
        _emit(plan)
        return 0
    if plan["next_cell_id"] is None:
        _emit({**plan, "status": "REFUSED", "reason": "NO_READY_CELL"})
        return 2
    target = next(cell for cell in cells if cell.cell_id == plan["next_cell_id"])
    try:
        receipt = fire_cell(
            target,
            day=args.day or dt.datetime.now(dt.UTC).strftime("%Y%m%d"),
            claims_path=args.claims_path,
            ttl_hours=args.ttl_hours,
            agent=args.agent,
        )
    except (QueueRefusal, chain.ChainRefusal) as exc:
        _emit(exc.as_dict())
        return 2
    _emit({"schema": "ddm_gv1_cell_queue_fire.v1", "plan": plan, "fired": receipt})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def _shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--queue", type=Path, required=True, help="ordered queue spec JSON")
        sp.add_argument("--ledger", type=Path, default=None, help="Metal-contention ledger")
        sp.add_argument("--margin-gib", type=float, default=admission.DEFAULT_MARGIN_GIB)
        sp.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
        sp.add_argument("--root", action="append", metavar="PATH")
        sp.add_argument(
            "--skip-seal-verify",
            action="store_true",
            help="skip running verify_inputs() inside each sealed tree (fast structural plan only)",
        )

    plan = sub.add_parser("plan", help="verify every cell and report which is ready to fire")
    _shared(plan)
    plan.set_defaults(func=_cmd_plan)

    fire = sub.add_parser(
        "fire",
        help="THE ONE FIRE PATH: plan, admit, claim, authorize and launch one cell",
    )
    _shared(fire)
    fire.add_argument("--cell-id", default=None, help="cell to fire (default: the next ready one)")
    fire.add_argument("--dry-run", action="store_true")
    fire.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS)
    fire.add_argument("--ttl-hours", type=float, default=8.0)
    fire.add_argument("--agent", default="ddm_gov2_cell_queue_driver")
    fire.add_argument("--day", default=None, help="claim-id day suffix (default: today UTC)")
    fire.set_defaults(func=_cmd_fire)

    run = sub.add_parser("run", help="fire the next ready cell (use --dry-run first)")
    _shared(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS)
    run.add_argument("--ttl-hours", type=float, default=8.0)
    run.add_argument("--agent", default="ddm_gv1_cell_queue_driver")
    run.add_argument("--day", default=None, help="claim-id day suffix (default: today UTC)")
    run.set_defaults(func=_cmd_run)

    verdict = sub.add_parser("verdict", help="milestone reads + falsifier state for one cell")
    verdict.add_argument("--queue", type=Path, required=True)
    verdict.add_argument("--cell-id", required=True)
    verdict.add_argument("--run-dir", type=Path, required=True)
    verdict.add_argument("--metric", action="append", help="dotted history path (repeatable)")
    verdict.set_defaults(func=_cmd_verdict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (QueueRefusal, chain.ChainRefusal) as exc:
        _emit(exc.as_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
