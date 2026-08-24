"""The variation gate — refuse a threshold whose quantity has not been shown to vary.

THE GENUS THIS EXTINCTS (three instances in one day, 2026-08-24)
---------------------------------------------------------------
1. MAIN's `KILL if r improves < 1.2x` — a threshold with no measured floor and no
   relative-significance number. 1.2x turned out to be 2.62x the remaining gap.
2. ddm_ds1's `proxy is sound if rank correlation >= 0.90` — the conventional
   "good correlation" value, inherited, never derived, no cost attached.
3. ddm_ds1's own R1a `two OFF seeds` design — a floor specified for a quantity
   that CANNOT VARY: wd3 restores all four RNG streams from the resume artifact
   (`_restore_rng`), so `seed_everything(config['seed'])` is overwritten and both
   arms are bit-identical. 3.4 h was authorized to measure a structural zero.

All three are ONE shape: **a threshold specified before checking whether the
quantity it measures can vary.** Instance 3 was built two messages after
instance 1 was corrected, by the agent that corrected it. **Knowing the genus did
not prevent it.** So the cure cannot be a thing to remember -- it has to be a step
that must execute. That is this module.

THE TRACE-THE-PATH TEST (step 1, and the only one that catches instance 3)
--------------------------------------------------------------------------
Before pre-registering any threshold, trace the path from the KNOB you intend to
turn to the QUANTITY you intend to measure, and name every hop. If any hop
OVERWRITES the knob's effect, the quantity cannot vary and the threshold is
vacuous. In instance 3 the path was
`config['seed'] -> seed_everything -> generator -> _restore_rng(checkpoint) -> XX`
and the last hop is an overwrite. One trace, done before launch, kills it.

WHAT THIS IS NOT
----------------
Not a text scanner over memos -- threshold-shaped prose is far too noisy to gate
on, and a noisy gate gets waived until it is inert (the campaign has that genus
too). This is a small fail-closed API a pre-registration CALLS. The refusal
arrives when the threshold is constructed, which is the moment the mistake is
made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PLACEHOLDER_TOKENS",
    "Threshold",
    "VariationClaim",
    "VariationGateError",
    "assert_can_vary",
    "trace_is_broken",
]

# Rationales that assert nothing. Rejected, per the campaign's placeholder law.
PLACEHOLDER_TOKENS = frozenset(
    {"", "tbd", "todo", "n/a", "na", "none", "unknown", "pending", "placeholder", "?", "-", "<reason>", "<rationale>"}
)

# A hop that DESTROYS an upstream effect. If one of these terminates the trace,
# the knob cannot reach the quantity and any threshold on it is vacuous.
_OVERWRITE_MARKERS = ("overwrite", "overwritten", "restore", "restored", "reset", "clobber", "ignored", "discard")

# Phrases that cite a CONVENTION as the justification for a number. Deliberately
# phrases and not bare words: `"default" in derivation` false-refuses an honest
# "measured on the default config's baseline", and a gate that cries wolf gets
# waived until it is inert.
_CONVENTION_PHRASES = (
    "conventional",
    "conventionally",
    "by convention",
    "the default value",
    "standard default",
    "usual value",
    "commonly used",
    "rule of thumb",
    "everyone uses",
)


class VariationGateError(RuntimeError):
    """Fail-closed refusal: the quantity has not been shown to vary."""


@dataclass(frozen=True)
class VariationClaim:
    """A claim that `quantity` can actually move when `knob` is turned.

    `path` is the trace: each hop from knob to quantity, in order. It is the
    load-bearing field. `evidence` must cite a file:line or a receipt -- the
    thing an author skips when they are pattern-matching instead of checking.
    """

    quantity: str
    knob: str
    path: tuple[str, ...]
    evidence: str
    observed_to_vary: bool = False

    def __post_init__(self) -> None:
        for name in ("quantity", "knob", "evidence"):
            value = str(getattr(self, name)).strip()
            if value.lower() in PLACEHOLDER_TOKENS:
                raise VariationGateError(f"VariationClaim.{name} is empty or a placeholder: {value!r}")
        if len(self.path) < 2:
            raise VariationGateError(
                "path must trace at least knob -> quantity; a one-hop 'it just varies' is the untraced claim itself"
            )
        for hop in self.path:
            if str(hop).strip().lower() in PLACEHOLDER_TOKENS:
                raise VariationGateError(f"path contains an empty or placeholder hop: {hop!r}")


@dataclass(frozen=True)
class Threshold:
    """A pre-registered threshold. Cannot be built without a passing claim."""

    name: str
    value: float
    units: str
    derivation: str
    claim: VariationClaim

    def __post_init__(self) -> None:
        for name in ("name", "units", "derivation"):
            value = str(getattr(self, name)).strip()
            if value.lower() in PLACEHOLDER_TOKENS:
                raise VariationGateError(f"Threshold.{name} is empty or a placeholder: {value!r}")
        # "0.90 because that is a good correlation" is instance 2, verbatim.
        # PHRASES, not bare words: a bare `"default" in derivation` check false-refuses
        # an honest derivation like "measured on the default config's baseline", and a
        # gate that cries wolf gets waived until it is inert -- this campaign has that
        # genus too, so the gate must not seed it.
        lowered = self.derivation.lower()
        if any(phrase in lowered for phrase in _CONVENTION_PHRASES):
            raise VariationGateError(
                f"Threshold {self.name!r} cites a convention as its derivation: {self.derivation!r}. "
                "A conventional value is an inherited guess. Derive it from a cost in the campaign's units."
            )
        assert_can_vary(self.claim)


def trace_is_broken(path: Sequence[str]) -> str | None:
    """Return the hop that destroys the knob's effect, or None if the path is live.

    This is the check that would have caught instance 3 before launch.
    """

    for hop in path[1:]:
        lowered = str(hop).lower()
        if any(marker in lowered for marker in _OVERWRITE_MARKERS):
            return str(hop)
    return None


def assert_can_vary(claim: VariationClaim) -> dict[str, Any]:
    """Refuse unless the quantity has been shown to be able to move.

    Two independent ways to pass, and BOTH are evidence, not assertion:
      * `observed_to_vary=True` -- it has actually been seen to differ, with a receipt; or
      * a live trace with no overwrite hop -- the mechanism is there in the code.
    """

    broken = trace_is_broken(claim.path)
    if broken is not None and not claim.observed_to_vary:
        raise VariationGateError(
            f"{claim.quantity!r} cannot vary with {claim.knob!r}: the hop {broken!r} destroys the knob's effect. "
            "A threshold on it would measure a structural zero. "
            "(This is ddm_ds1 R1a: config['seed'] was overwritten by _restore_rng at resume.)"
        )
    return {
        "gate": "ddm_ds1_variation_gate",
        "quantity": claim.quantity,
        "knob": claim.knob,
        "hops": len(claim.path),
        "trace_live": broken is None,
        "observed_to_vary": claim.observed_to_vary,
        "evidence": claim.evidence,
        "passed_by": "observation" if claim.observed_to_vary else "live_trace",
    }
