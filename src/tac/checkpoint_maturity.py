# SPDX-License-Identifier: MIT
"""Checkpoint MATURITY naming convention — the ``_dev`` / ``_prod`` axis.

**Source:** operator directive 2026-07-18 verbatim intent: *"Update our apparatus
to accept dev and prod checkpoints so we can keep iterating on v9c3 and keep
them if/when we run v10. Just different naming conventions, like with an
underscore and dev or prod."*

Maturity is an axis ORTHOGONAL to the VEHICLE axis (v9c2 / v9c3 / v10 — locked
in SPEC_v10 §14.1, branch ``claude/p0_521_spec_v10_capstone_20260717``). The
vehicle name says WHICH carrier; the maturity suffix says HOW PROMOTABLE its
scores are. Composed name form: ``<vehicle>_<dev|prod>[...]`` — e.g.
``v9c3_dev``, ``v10_prod_bank_20260801``.

Semantics (binding):

* ``_dev``  — free-iteration lane. Kept ALONGSIDE prod (a dev bank never
  overwrites a prod bank). Its advisory AND exact rows are NON-PROMOTABLE to
  the canonical frontier pointer BY DEFAULT — a dev exact row may be banked +
  labeled but MUST NOT move ``.omx/state/canonical_frontier_pointer.json``
  unless the operator explicitly promotes dev→prod. Sister of the CLAUDE.md
  "Frontier scores are pointer-only" + NO-FAKE #8 surrogate≠authority rules.
* ``_prod`` — the capstone lane. Only its byte-closed exact rows (operator-GO)
  are eligible to move the pointer. Prod bank dirs are treated as IMMUTABLE by
  :func:`assert_bank_dir_writable` (write a NEW dated bank dir instead).
* untagged VEHICLE-shaped names (e.g. ``v9c2_defensive_bank_20260718``) default
  to the SAFE side: treated as dev / non-promotable — never silently
  promotable.
* legacy pre-convention names with NO vehicle token (e.g.
  ``pr101_frame_exploit_selector_...``, ``levelset_n600_witness_...``) predate
  the maturity axis entirely and are GRANDFATHERED at the pointer-promotion
  surface (refusing them would clobber the standing frontier anchors, whose
  lane names predate this convention). :func:`is_pointer_promotable` remains
  strict (True ONLY for explicit ``_prod``); the grandfathering lives ONLY in
  :func:`pointer_promotion_verdict`, with the reason string recorded.

This module is a pure convention + parse helper: no side effects, no state
file, no registry. The TWO respect-points that consume it:

  A. banking — :func:`assert_bank_dir_writable` (a dev/new bank write refuses
     to clobber an existing ``_prod`` bank dir; prod banks are immutable) +
     :func:`bank_dir_name` (the composed convention).
  B. pointer promotion — ``tac.canonical_frontier_pointer.
     refresh_canonical_frontier_from_local_state`` gates every candidate
     anchor through :func:`pointer_promotion_verdict` (fail-closed: a refused
     candidate never becomes the pointer anchor; the prior anchor is kept and
     the refusal is recorded in ``refresh_provenance``).

The live run ``levelset_n600_witness_20260717T113932Z`` (v9c2 warm-start) is
implicitly dev and is NOT renamed — the convention applies to NEW run/bank
names going forward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MATURITY_DEV",
    "MATURITY_PROD",
    "MATURITY_UNKNOWN",
    "CheckpointMaturity",
    "ProdBankImmutableError",
    "parse_checkpoint_maturity",
    "is_pointer_promotable",
    "pointer_promotion_verdict",
    "bank_dir_name",
    "assert_bank_dir_writable",
]

MATURITY_DEV = "dev"
MATURITY_PROD = "prod"
MATURITY_UNKNOWN = "unknown"

#: Vehicle token per the SPEC_v10 §14.1 naming lock lineage: ``v9``, ``v9c2``,
#: ``v9c3``, ``v10``, ``v75`` / ``v7.5``, ``v8`` ... — a ``v`` followed by a
#: digit, optionally letter+digit generations (``c2``) or a dotted minor.
_VEHICLE_TOKEN_RE = re.compile(r"^v\d+(?:[a-z]\d+)*(?:\.\d+)?$")

#: Name tokenizer: underscores/hyphens/whitespace separate tokens. Dots are
#: kept INSIDE tokens (so ``v7.5`` parses as one vehicle token) except that a
#: trailing file extension is stripped before tokenizing.
_TOKEN_SPLIT_RE = re.compile(r"[_\-\s]+")

_KNOWN_EXTENSIONS = (".npz", ".json", ".pt", ".zip", ".md", ".jsonl", ".log", ".yaml", ".yml")


class ProdBankImmutableError(RuntimeError):
    """Raised when a bank write targets an existing ``_prod`` bank directory."""


@dataclass(frozen=True)
class CheckpointMaturity:
    """Parsed ``(vehicle, maturity)`` classification of a run/bank/ckpt name."""

    name: str
    vehicle: str | None
    maturity: str  # MATURITY_DEV | MATURITY_PROD | MATURITY_UNKNOWN
    reason: str


def _tokens(name: str) -> list[str]:
    base = name.strip().lower()
    for ext in _KNOWN_EXTENSIONS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return [t for t in _TOKEN_SPLIT_RE.split(base) if t]


def parse_checkpoint_maturity(name: str | Path) -> CheckpointMaturity:
    """Parse a run-dir / bank-dir / checkpoint NAME into ``(vehicle, maturity)``.

    Pure classification — no filesystem access. ``name`` may be a bare name or
    a path (only the final component is parsed; callers that want to scan a
    full path should parse each segment).

    Falling rules (safe-side by construction):

    1. explicit ``dev`` token  → ``dev`` (wins over ``prod`` if BOTH appear —
       an ambiguous name must not be silently promotable);
    2. explicit ``prod`` token → ``prod``;
    3. no maturity token       → ``unknown`` (non-promotable), with the vehicle
       token (if any) extracted for reporting.
    """

    raw = Path(name).name if isinstance(name, Path) else str(name)
    raw = raw.split("/")[-1].split("\\")[-1]
    toks = _tokens(raw)
    vehicle = next((t for t in toks if _VEHICLE_TOKEN_RE.match(t)), None)

    if MATURITY_DEV in toks:
        reason = "explicit _dev token"
        if MATURITY_PROD in toks:
            reason = "ambiguous name carries BOTH _dev and _prod tokens; dev wins (safe side)"
        return CheckpointMaturity(name=raw, vehicle=vehicle, maturity=MATURITY_DEV, reason=reason)
    if MATURITY_PROD in toks:
        return CheckpointMaturity(
            name=raw, vehicle=vehicle, maturity=MATURITY_PROD, reason="explicit _prod token"
        )
    if vehicle is not None:
        return CheckpointMaturity(
            name=raw,
            vehicle=vehicle,
            maturity=MATURITY_UNKNOWN,
            reason=f"vehicle token '{vehicle}' without _dev/_prod tag — safe-side default (treated as dev / non-promotable)",
        )
    return CheckpointMaturity(
        name=raw,
        vehicle=None,
        maturity=MATURITY_UNKNOWN,
        reason="no vehicle token and no maturity tag (legacy pre-convention name)",
    )


def is_pointer_promotable(name: str | Path) -> bool:
    """True ONLY for an explicit ``_prod`` name. Strict truth table:

    * ``v10_prod...``  → True
    * ``v9c3_dev...``  → False
    * ``v9c2_...`` (vehicle, untagged) → False (safe side)
    * legacy non-vehicle names → False (strict surface; the pointer-refresh
      grandfathering for pre-convention lanes lives in
      :func:`pointer_promotion_verdict`, never here)
    """

    return parse_checkpoint_maturity(name).maturity == MATURITY_PROD


def pointer_promotion_verdict(name: str | Path) -> tuple[bool, str]:
    """Pointer-promotion verdict for a single name: ``(allowed, reason)``.

    * explicit ``_prod``          → allowed
    * explicit ``_dev``           → REFUSED (dev rows are banked/labeled, never
      pointer-promoted, until operator dev→prod promotion)
    * vehicle-shaped, untagged    → REFUSED (safe-side default = dev)
    * legacy non-vehicle name     → allowed-with-reason (GRANDFATHERED: the
      standing frontier anchors predate the maturity convention; refusing them
      would clobber the live pointer — see module docstring)
    """

    cm = parse_checkpoint_maturity(name)
    if cm.maturity == MATURITY_PROD:
        return True, f"{cm.name}: explicit _prod — pointer-promotable"
    if cm.maturity == MATURITY_DEV:
        return False, f"{cm.name}: {cm.reason} — dev checkpoints are NON-promotable (bank + label only)"
    if cm.vehicle is not None:
        return False, f"{cm.name}: {cm.reason}"
    return True, f"{cm.name}: {cm.reason} — grandfathered (maturity axis not applicable)"


def bank_dir_name(vehicle: str, maturity: str, label: str, date_utc: str) -> str:
    """Compose the canonical bank-dir name: ``<vehicle>_<maturity>_<label>_<YYYYMMDD>``.

    Example: ``bank_dir_name("v9c3", "dev", "bank", "20260718")`` →
    ``"v9c3_dev_bank_20260718"``. Refuses an unknown maturity or a
    non-vehicle-shaped vehicle string (fail-closed on typos like ``dve``).
    """

    v = str(vehicle).strip().lower()
    m = str(maturity).strip().lower()
    if not _VEHICLE_TOKEN_RE.match(v):
        raise ValueError(f"not a vehicle token: {vehicle!r} (expected e.g. v9c3 / v10)")
    if m not in (MATURITY_DEV, MATURITY_PROD):
        raise ValueError(f"maturity must be 'dev' or 'prod', got {maturity!r}")
    parts = [v, m]
    label_s = str(label).strip().strip("_")
    if label_s:
        parts.append(label_s)
    date_s = str(date_utc).strip()
    if date_s:
        parts.append(date_s)
    return "_".join(parts)


def assert_bank_dir_writable(target_dir: str | Path) -> Path:
    """Refuse a bank write that targets an EXISTING ``_prod`` bank directory.

    Prod banks are immutable: nothing (dev iteration OR a re-run of prod
    banking) overwrites them in place — write a NEW dated bank dir instead.
    Dev/untagged bank dirs may be re-written by their own lane (dev is the
    free-iteration lane). Because dev and prod compose DIFFERENT names
    (``v9c3_dev_*`` vs ``v9c3_prod_*``), the two always coexist on disk; this
    guard closes the remaining clobber path (a caller aiming a write directly
    at a prod dir path).

    Returns the resolved ``Path`` when the write is allowed; raises
    :class:`ProdBankImmutableError` otherwise.
    """

    target = Path(target_dir)
    if target.exists():
        cm = parse_checkpoint_maturity(target.name)
        if cm.maturity == MATURITY_PROD:
            raise ProdBankImmutableError(
                f"refusing to write into existing PROD bank dir {target} "
                f"({cm.reason}); prod banks are immutable — write a new dated "
                f"bank dir (see tac.checkpoint_maturity.bank_dir_name)"
            )
    return target
