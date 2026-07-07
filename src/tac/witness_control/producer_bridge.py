"""Producer bridge — the #247 de-orphaning: the costate controller as the ONE canonical consumer.

#247's mandate is "unify the cathedral autopilot + orphaned producers into ONE canonical-consumer
costate-controller (de-orphan, don't rebuild)." Before this module, the costate controller
(``tac.witness_control.shadow_controller``) consumed only its own trajectory verdicts + the
activation ledger. The cathedral autopilot, the master-gradient per-byte anchors, and the
sensitivity-map per-axis EV weights were orphaned *relative to the controller* — read (if at all) by
a separate module cluster, never by the ONE controller.

This bridge makes the controller ENUMERATE and READ every such producer into its SENSE. It does NOT
rebuild them (per the mandate) and it does NOT fabricate signals (NO-FAKE): each producer contributes
either a REAL live signal or an honest ``available=False`` with a recorded ``reason`` (exactly the
"'off' is a tracked queue, never a forgotten default" discipline — a producer with no live input is
SURFACED as awaiting-its-input, not silently dropped and not invented).

The producers + what each contributes to the controller's SENSE:

* ``sensitivity_map.axis_weights_for_named_operating_point`` — per-axis EV multipliers
  (pose/seg/rate/mixed) at the operating point. A LIVE, always-available prior the DECIDE ranker
  uses to weight costates by axis. Provenance (``operating_point_tag`` + ``basis``) is propagated
  per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".
* ``master_gradient.latest_anchor_for_archive`` — the measured ΔS-per-byte gradient anchor for a
  BYTE-CLOSED archive. During a live training run there is no byte-closed archive, so this is
  honestly ``available=False`` (reason: no archive sha) until a byte-close lands one.
* ``cathedral_autopilot.rank_candidates_via_three_metric_trichotomy`` — the campaign candidate
  RANKER. The controller does NOT invoke it per-verdict (it has side-effect-free ranking but needs a
  candidate set); instead the bridge surfaces it as available and names the composition: the
  activation ledger's ``duty_to_measure`` never-fired levers ARE its candidate set (the EIG/select
  bridge documented in DAG FEED-247loop). ``duty_to_measure_as_candidates`` exposes that label list.
* ``tac.harness_failure_ledger.sense_rows`` — ranked open/recurrent harness-failure classes (the
  Weng weakness-mining harvest 2026-07-07, ``docs/harvest_weng_harness_20260707.md``): the
  controller can weight recommendations by known-unreliable surfaces; honest ``available=False``
  when the ledger is empty.
* ``tac.witness_control.powerlaw_exit`` — the weak-KAM power-law plateau/exit detector (solver
  pack 2026-07-07; equation ``weak_kam_powerlaw_tail_exit_v1``): tail fit + extrapolated-
  remaining-meat exit verdict on the live run's trailing verdict window; honest
  ``available=False`` without a ``run_log_path`` or with too few verdict rows.

Everything here is READ-ONLY + advisory (like the whole ``witness_control`` package): it never
mutates a run, never claims a score, never auto-actuates.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProducerSignal:
    """One orphaned producer's contribution to the controller SENSE.

    ``available`` is True only when the producer returned a REAL live signal. When False, ``reason``
    records WHY (import failed / no byte-closed archive yet / awaiting campaign-select time) — never a
    fabricated value. ``provenance`` carries evidence tags (operating point, basis, axis) so any
    downstream consumer can propagate them.
    """

    name: str
    available: bool
    signal: dict | None
    reason: str
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "producer": self.name,
            "available": self.available,
            "signal": self.signal,
            "reason": self.reason,
            "provenance": self.provenance,
        }


def _sensitivity_map_signal(operating_point: str) -> ProducerSignal:
    """Per-axis EV multipliers (pose/seg/rate/mixed) — a LIVE static prior, always available."""
    try:
        from tac.sensitivity_map import axis_weights_for_named_operating_point
        aw = axis_weights_for_named_operating_point(operating_point)
        return ProducerSignal(
            name="sensitivity_map.axis_weights",
            available=True,
            signal={"pose": aw.pose, "seg": aw.seg, "rate": aw.rate, "mixed": aw.mixed},
            reason="per-axis EV multipliers at the operating point (DECIDE weights costates by axis)",
            provenance={"operating_point_tag": aw.operating_point_tag, "basis": aw.basis},
        )
    except Exception as e:  # noqa: BLE001 — advisory bridge, never breaks the report
        return ProducerSignal(
            name="sensitivity_map.axis_weights", available=False, signal=None,
            reason=f"unavailable ({type(e).__name__}: {e})",
        )


def _master_gradient_signal(archive_sha256: str | None) -> ProducerSignal:
    """The measured ΔS-per-byte anchor for a byte-closed archive. No archive during live training =>
    honestly unavailable (NOT fabricated)."""
    if not archive_sha256:
        return ProducerSignal(
            name="master_gradient.latest_anchor", available=False, signal=None,
            reason="no byte-closed archive sha (live training has no archive yet); "
                   "a byte-close lands one and this becomes available",
        )
    try:
        from tac.master_gradient import latest_anchor_for_archive
        anchor = latest_anchor_for_archive(archive_sha256)
        if anchor is None:
            return ProducerSignal(
                name="master_gradient.latest_anchor", available=False, signal=None,
                reason=f"no authoritative gradient anchor recorded for archive {archive_sha256[:12]}",
                provenance={"archive_sha256": archive_sha256},
            )
        return ProducerSignal(
            name="master_gradient.latest_anchor", available=True, signal=anchor,
            reason="measured ΔS-per-byte gradient anchor for the byte-closed archive",
            provenance={"archive_sha256": archive_sha256},
        )
    except Exception as e:  # noqa: BLE001
        return ProducerSignal(
            name="master_gradient.latest_anchor", available=False, signal=None,
            reason=f"unavailable ({type(e).__name__}: {e})",
            provenance={"archive_sha256": archive_sha256},
        )


def _cathedral_autopilot_signal() -> ProducerSignal:
    """The campaign candidate RANKER. Surfaced as available (import-verified) + the composition note;
    NOT invoked per-verdict (it ranks a candidate SET, supplied by ``duty_to_measure`` at select
    time). This de-orphans it relative to the controller WITHOUT fabricating a ranking."""
    try:
        import tac.cathedral_autopilot as _ca
        has_ranker = hasattr(_ca, "rank_candidates_via_three_metric_trichotomy")
        return ProducerSignal(
            name="cathedral_autopilot.ranker",
            available=has_ranker,
            signal=None,
            reason="campaign candidate ranker; consumes the activation-ledger duty_to_measure "
                   "never-fired levers as its candidate set (the EIG/select composition, DAG "
                   "FEED-247loop) — invoked at campaign SELECT time, not per-verdict",
            provenance={"ranker": "rank_candidates_via_three_metric_trichotomy" if has_ranker else None},
        )
    except Exception as e:  # noqa: BLE001
        return ProducerSignal(
            name="cathedral_autopilot.ranker", available=False, signal=None,
            reason=f"unavailable ({type(e).__name__}: {e})",
        )


def _harness_failure_signal() -> ProducerSignal:
    """The Weng weakness-mining failure ledger (``tac.harness_failure_ledger``, harvest
    2026-07-07): ranked open/recurrent harness-failure classes as a SENSE input. Read-only;
    the controller may weight its recommendations by known-unreliable surfaces (e.g. refuse to
    recommend a spawned-daemon measurement while the daemon-kill class is open). Honest when
    empty: no ledger rows => available=False, never a fabricated failure."""
    try:
        from tac.harness_failure_ledger import sense_rows
        rows = sense_rows(limit=10)
        if not rows:
            return ProducerSignal(
                name="harness_failure_ledger.sense_rows", available=False, signal=None,
                reason="ledger empty or absent (no recorded harness-failure classes)",
            )
        return ProducerSignal(
            name="harness_failure_ledger.sense_rows", available=True,
            signal={"ranked_failures": rows},
            reason="ranked harness-failure classes (unresolved first, recurrence desc) — "
                   "the Weng 'recurrent, addressable patterns' preference as SENSE",
            provenance={"ledger": ".omx/state/harness_failure_ledger.jsonl"},
        )
    except Exception as e:  # noqa: BLE001 — advisory bridge, never breaks the report
        return ProducerSignal(
            name="harness_failure_ledger.sense_rows", available=False, signal=None,
            reason=f"unavailable ({type(e).__name__}: {e})",
        )


def _powerlaw_exit_signal(run_log_path=None) -> ProducerSignal:
    """The weak-KAM power-law plateau/exit detector (``tac.witness_control.powerlaw_exit``,
    solver pack 2026-07-07; equation ``weak_kam_powerlaw_tail_exit_v1``): per-class (total-d_seg
    fallback while per-class telemetry is an apparatus gap) tail fits a+b*t^-alpha vs exponential
    on the CURRENT stage window of a run's verdict telemetry, plus the extrapolated-remaining-meat
    exit verdict — so the DECIDE queue can see whether the running stage still pays under the
    AIC-preferred tail model instead of an early-firing exponential window slope. Honest when no
    run log is provided or the window is too short: available=False with the reason, never a
    fabricated fit."""
    if not run_log_path:
        return ProducerSignal(
            name="powerlaw_exit.stage_tail_fit", available=False, signal=None,
            reason="no run_log_path provided (pass the live run's telemetry log to fit the "
                   "current stage window); the $0 retro-fit CLI is "
                   "tools/fit_powerlaw_plateau_detector.py",
        )
    try:
        import json as _json

        from tac.witness_control.powerlaw_exit import powerlaw_meat_exit
        rows = []
        with open(run_log_path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if r.get("stage") == "verdict" and float(r.get("epoch", 0)) > 0:
                    rows.append((float(r["epoch"]), float(r["d_seg"])))
        if len(rows) < 8:
            return ProducerSignal(
                name="powerlaw_exit.stage_tail_fit", available=False, signal=None,
                reason=f"only {len(rows)} verdict rows in {run_log_path} (need >= 8 for a "
                       "credible 3-parameter tail fit)",
                provenance={"run_log_path": str(run_log_path)},
            )
        # fit the TRAILING HALF of the verdict rows. Honest limitation (advisory signal): this
        # window MAY span a stage/optimizer transition (e.g. the tau->Muon switch) — the fitted
        # alpha then mixes regimes; window_epochs is surfaced so the consumer can see the span,
        # and the $0 CLI (tools/fit_powerlaw_plateau_detector.py) does the per-stage-window cut.
        tail = rows[len(rows) // 2:]
        verdict = powerlaw_meat_exit({"total": tail}, min_points=8)
        return ProducerSignal(
            name="powerlaw_exit.stage_tail_fit", available=True,
            signal={"exhausted": verdict["exhausted"],
                    "remaining_meat_estimate": verdict["remaining_meat_estimate"],
                    "alpha": verdict["alpha"], "ci": verdict["ci"],
                    "binding_class": verdict["binding_class"],
                    "reason": verdict["reason"],
                    "window_epochs": [tail[0][0], tail[-1][0]],
                    "per_class_available": False},
            reason="trailing-window tail fit + extrapolated-remaining-meat exit verdict "
                   "(weak_kam_powerlaw_tail_exit_v1); total-d_seg fallback — per-class d_seg "
                   "verdict telemetry is an OWED apparatus gap (duty-to-measure)",
            provenance={"run_log_path": str(run_log_path),
                        "equation": "weak_kam_powerlaw_tail_exit_v1",
                        "axis": "[macOS advisory] NON-PROMOTABLE"},
        )
    except Exception as e:  # noqa: BLE001 — advisory bridge, never breaks the report
        return ProducerSignal(
            name="powerlaw_exit.stage_tail_fit", available=False, signal=None,
            reason=f"unavailable ({type(e).__name__}: {e})",
            provenance={"run_log_path": str(run_log_path)},
        )


def read_producer_signals(
    *,
    archive_sha256: str | None = None,
    operating_point: str = "pr106_r2",
    run_log_path=None,
) -> list[dict]:
    """Read EVERY orphaned producer into the controller SENSE (the #247 de-orphaning).

    Returns one row per producer (as a plain dict for JSONL persistence). Fully fail-safe: any
    producer that errors yields an ``available=False`` row with the reason — the controller SENSE is
    never broken by a producer, and no signal is ever fabricated. ``run_log_path`` (optional) feeds
    the power-law exit detector the live run's verdict telemetry; callers that do not pass it get an
    honest available=False row (same contract as the no-archive master-gradient row).
    """
    return [
        _sensitivity_map_signal(operating_point).to_dict(),
        _master_gradient_signal(archive_sha256).to_dict(),
        _cathedral_autopilot_signal().to_dict(),
        _harness_failure_signal().to_dict(),
        _powerlaw_exit_signal(run_log_path).to_dict(),
    ]


def _lever_measurement_cost(name: str):
    """Measurement COST of firing a never-fired lever = the extra training epochs its DSL factory
    adds (``epochs_delta``, summed if the factory returns a tuple of Levers). Fail-safe: returns None
    if the factory can't be called (the ranker sinks None-cost levers last). NO fabrication."""
    try:
        from tac.witness_dsl import curriculum_dsl as _cd
        fac = getattr(_cd, name, None)
        if fac is None or not callable(fac):
            return None
        lev = fac()
        levs = lev if isinstance(lev, (list, tuple)) else [lev]
        return sum(int(getattr(x, "epochs_delta", 0) or 0) for x in levs)
    except Exception:  # noqa: BLE001
        return None


def rank_duty_to_measure(path=None) -> list[dict]:
    """The #247 EIG-bridge (the DECIDE ordering of the owed queue): RANK the activation-ledger's
    never-fired ``duty_to_measure`` levers by measurement COST ascending — the cheapest-to-measure
    lever first. This is the honest expected-information-gain-per-cost ordering UNDER AN UNINFORMATIVE
    PRIOR: a never-fired lever has NO measured costate (that is why it is owed), so every owed lever
    carries equal expected information about its own effect; dividing that equal information by the
    real per-lever cost ⇒ rank by ``1/cost`` ⇒ cheapest first. NO fabricated ΔS (a never-fired lever's
    ΔS is unknown by construction); the only real per-lever signal used is the DSL factory's
    ``epochs_delta`` cost. Levers whose cost cannot be computed sink to the end (unknown-cost tier).

    This is what the reviewer's noted composition needs: the controller does not merely LIST the owed
    levers, it RANKS which to fire next — feeding the DECIDE queue. Fail-safe (empty on any error)."""
    try:
        from tac.witness_dsl import activation_ledger as _al
        owed = _al.duty_to_measure()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for lever in owed:
        cost = _lever_measurement_cost(lever)
        rows.append({
            "candidate_lever": lever,
            "measurement_cost_epochs": cost,
            "eig_per_cost_rank_basis": ("1/cost (uninformative prior; cheapest owed lever first)"
                                        if cost is not None else "unknown-cost (sinks last)"),
            "note": "never-fired/owed; NO predicted ΔS (unmeasured by construction)",
        })
    # rank by cost ascending; unknown cost (None) sinks to the end; ties broken by lever name
    rows.sort(key=lambda r: (r["measurement_cost_epochs"] is None,
                             r["measurement_cost_epochs"] if r["measurement_cost_epochs"] is not None else 0,
                             r["candidate_lever"]))
    return rows


def duty_to_measure_as_candidates(path=None) -> list[dict]:
    """The EIG/select composition bridge (DAG FEED-247loop): expose the activation ledger's
    never-fired ``duty_to_measure`` levers as the candidate label set the cathedral autopilot ranker
    (or the findings-Lagrangian EIG selector) ranks to pick WHICH lever to fire next.

    Honest: this returns the candidate LABELS + their activation state ONLY — no fabricated score. The
    ranker assigns the ordering; once a lever fires + is measured it leaves this set and enters the
    ΔS-per-cost DECIDE queue. Fail-safe (empty list if the ledger is unavailable)."""
    try:
        from tac.witness_dsl import activation_ledger as _al
        owed = set(_al.duty_to_measure())
        return [
            {"candidate_lever": r["lever"], "activation_state": r["state"],
             "note": "never-fired/owed-measurement; ranker picks fire-order, no fabricated ΔS"}
            for r in _al.activation_report() if r["lever"] in owed
        ]
    except Exception:  # noqa: BLE001
        return []


__all__ = [
    "ProducerSignal",
    "duty_to_measure_as_candidates",
    "rank_duty_to_measure",
    "read_producer_signals",
]
