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

The three producers + what each contributes to the controller's SENSE:

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


def read_producer_signals(
    *,
    archive_sha256: str | None = None,
    operating_point: str = "pr106_r2",
) -> list[dict]:
    """Read EVERY orphaned producer into the controller SENSE (the #247 de-orphaning).

    Returns one row per producer (as a plain dict for JSONL persistence). Fully fail-safe: any
    producer that errors yields an ``available=False`` row with the reason — the controller SENSE is
    never broken by a producer, and no signal is ever fabricated.
    """
    return [
        _sensitivity_map_signal(operating_point).to_dict(),
        _master_gradient_signal(archive_sha256).to_dict(),
        _cathedral_autopilot_signal().to_dict(),
    ]


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
    "read_producer_signals",
]
