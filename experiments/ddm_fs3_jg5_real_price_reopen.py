#!/usr/bin/env python3
"""ddm_fs3 -- re-price ddm_jg3's retained configuration sweep at the REAL token rate.

WHY THIS EXISTS
---------------
``ddm_fs2`` sec.7 recorded a reusable law: the ``-log2 p`` first-order model prices
token-field edits well when they move AWAY from the model argmax (~0.88x) and
almost not at all when they move TOWARD it (~0.09x).  The charter that spawned this
arm read that row as "every jg5-class edit move carries an ~19% rate overcharge,
so moves rejected on a thin rate margin may flip sign under the real price."

That premise has to be checked against the receipts before anything is built,
because the two numbers fs2 quoted for the same ratio disagree with each other
(3.8373 / 4.718 = 0.8133, not the 0.877 written beside them), and because neither
of those numbers is necessarily the price ``ddm_jg3`` actually charged.

WHAT THE RECEIPTS SAY
---------------------
There are TWO rejection surfaces in the jg3 solve and they were priced differently.

* **Surface A -- the per-site inner gate** (``ddm_jg3_joint_solve.py:695``):
  ``if repaired * BITS_PER_SEG_CELL <= cost: continue``, where ``cost`` is the
  ``LogitPrice`` ``log2(p_old/p_new)`` of that single move.  jg3 labels that class
  "a RANKER, not a price" in its own docstring and measured it UNDER-charging by
  2.2x in aggregate (1.91 bits/token against jg2's measured 4.1379).  The rejected
  moves are dropped with a bare ``continue`` and **are not retained**, so this
  surface cannot be censused from disk at $0.

* **Surface B -- the per-configuration sweep** (``ddm_jg3_joint_solve.py:807``):
  ``cost_bits = tokens_here * RATE_PRIOR_BITS_PER_TOKEN`` with the flat
  ``4.1379`` bits/token, and the winner is ``argmin`` of ``net_delta_S`` over the
  whole ``separation x keep_fraction`` grid.  **Every rejected configuration is
  retained** in ``per_pair[].separation_sweep`` with its tokens, its measured
  repaired-cell count and its modelled net.  This surface is fully censusable at $0
  and it is where a flat-prior overcharge would live.

So this module censuses Surface B exactly, and reports Surface A as not-retained
rather than guessing at it.

WHAT "REAL" MEANS HERE
----------------------
The real price is not asserted.  It is MEASURED from ``ddm_jg4``'s retained
per-frame code-bit arrays -- the output of the ``ddm_jg2`` re-encoder whose
unedited control reproduces the shipped RC64 token stream byte-identically --
divided by jg3's own measured token count:

    B_real = (bits_candidate - bits_control)[edited pairs].sum() / tokens_changed

Everything downstream is a re-selection under that price.  The re-selection gain is
the ONLY new credit available: the flat-prior error on the *chosen* configurations
is already banked, because jg5 built the body for real and paid the real bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants, each re-derived here from the score definition rather than copied.
# ---------------------------------------------------------------------------

N_PAIRS = 600
GRID_H = 384
GRID_W = 512
PLANE = GRID_H * GRID_W

SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR
S_PER_SEG_CELL = 100.0 / (N_PAIRS * PLANE)
BITS_PER_SEG_CELL = 8.0 * S_PER_SEG_CELL / S_PER_ARCHIVE_BYTE

#: The price ddm_jg3 actually charged in its configuration sweep.
JG3_ACCEPT_BITS_PER_TOKEN = 4.1379

#: MEASURED by this arm: holding the 38 reopened pairs' 569 shipped tokens out of the
#: 455-edit field and re-encoding for real saved 189 archive bytes
#: (113,847 -> 113,658 B token stream), against a control that reproduces the base
#: stream byte-identically.  Receipt:
#: ``/Volumes/APDataStore/pact/ddm_fs3/reencode/retained/S1_encode_fs3_holdout38.json``.
#: This is the price of those pairs' EXISTING edits.  It validates the per-pair
#: attribution METHOD on exactly these pairs (predicted 2.7357, measured 2.6573,
#: 2.9% error, against a population price that was 30.1% out); it is NOT the price
#: of the additional lower-ranked tokens a re-selection would add.
MEASURED_MARGINAL_BITS_PER_TOKEN = (113_847 - 113_658) * 8.0 / 569.0

#: Leg 2 built and stat'd ONE point: reverting all 454 moved pairs' carrier
#: coefficients costs 45 archive bytes over 5,119 coefficients.  Applying it to a
#: DIFFERENT pair count is a linear extrapolation, not that measurement.
CARRIER_COMPENSATION_BYTES_PER_PAIR = 45.0 / 454.0

#: Why the emitted carrier key says DERIVED_extrapolated and not MEASURED.
#: Withdrawn by rv17 wave-3 W3-F7; superseded by a terminal measurement.  Emitted
#: beside the value so a future receipt carries the caveat without the reader
#: having to find this memo -- W3-F15 was exactly this caveat failing to travel.
CARRIER_LABEL_SUPERSEDED = (
    "carrier_MEASURED_leg2 -- WITHDRAWN as overclaimed (rv17 wave-3 W3-F7). The "
    "value is a LINEAR EXTRAPOLATION: 45 B measured over 454 pairs in ONE build, "
    "re-multiplied at 0.0991 B/pair. The ladder measured that price as NON-MONOTONE "
    "in density with a +-45 B container-search spread, so the leg's uncertainty "
    "band (+-3.00e-05 S) is LARGER than its own point estimate. SUPERSEDED by "
    "measurement: .omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md:877 -- "
    "the real build (180,625 -> 179,961 = -664 B) puts the +45 B splice on BOTH "
    "sides, so leaving the carrier unchanged makes the carrier BYTE leg EXACTLY "
    "ZERO. Caveat from the same memo: zero bytes buys a STALE carrier on the "
    "changed pairs, so the COST does not vanish even though the BYTES do."
)

#: fs1 sec.3, re-derived at the live operating point: d_pose mean shift per added
#: token, OLS over jg5's 455 kept pairs.  DERIVED cross-sectional, not causal.
POSE_D_POSE_PER_ADDED_TOKEN = 1.4636e-08
LIVE_MEAN_D_POSE = 6.365684192281523e-06

#: The live pointer body.  Quoted for the admission bar only; nothing is built here.
POINTER_ARCHIVE_BYTES = 180_456
ADMISSION_BAR_S = 3.5e-6

DEFAULT_SHARDS = [
    f"/Volumes/APDataStore/pact/ddm_jg3/retained/seg_solve_n600_wc2s{i}.json"
    for i in range(6)
]
DEFAULT_BITS_CANDIDATE = (
    "/Volumes/APDataStore/pact/ddm_jg4/retained/bits_per_frame_complete_n600.npy"
)
DEFAULT_BITS_CONTROL = (
    "/Volumes/APDataStore/pact/ddm_jg4/retained/bits_per_frame_control_600.npy"
)
DEFAULT_EDITS_NPZ = (
    "/Volumes/APDataStore/pact/ddm_jg3/retained/seg_edits_n600_complete.npz"
)
DEFAULT_KEPT_PAIRS = "/Volumes/APDataStore/pact/ddm_jg5/retained/final/kept_pairs.json"


class Fs3Error(RuntimeError):
    """Fail-closed error.  Every control in this module raises rather than warns."""


def sha256_of(path: Path) -> tuple[str, int]:
    """Content digest computed AT READ TIME.  A hardcoded sha is not a control."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True)
class Entry:
    """One configuration of one pair: a point on jg3's separation x keep grid."""

    separation: int
    keep_fraction: float
    tokens: int
    repaired: int
    flips_after: int
    net_modelled: float

    def net_at(self, bits_per_token: float) -> float:
        """Composed seg+rate delta S for this configuration at a given token price.

        The seg leg is MEASURED (realized repaired-cell counts through the
        receiver's forward model and the frozen CPU SegNet); only the rate leg is
        re-priced.
        """
        return (
            -self.repaired * S_PER_SEG_CELL
            + (self.tokens * bits_per_token / 8.0) * S_PER_ARCHIVE_BYTE
        )


@dataclass(frozen=True)
class PairSweep:
    pair: int
    chosen: Entry
    entries: tuple[Entry, ...]


def load_shards(paths: Sequence[Path]) -> tuple[list[PairSweep], dict[str, Any]]:
    """Load jg3's retained per-shard ledgers and run the three arithmetic controls.

    CONTROL 1 -- every retained ``net_delta_S`` must be reproducible from
    ``(repaired, tokens)`` and the published constants.  If it is not, this module
    does not understand the ledger and must not re-price it.

    CONTROL 2 -- the configuration jg3 recorded as chosen must be present in the
    sweep AND be the argmin of the modelled net.  A chosen row that is not the
    argmin means the selection rule was something other than the one being
    re-priced.

    CONTROL 3 -- the chosen configurations' tokens and repaired cells must sum to
    the shard totals jg3 published independently.
    """
    sweeps: list[PairSweep] = []
    provenance: list[dict[str, Any]] = []
    total_tokens_published = 0
    total_repaired_published = 0
    control1_max_residual = 0.0

    for path in paths:
        if not path.exists():
            raise Fs3Error(f"shard ledger missing: {path}")
        digest, size = sha256_of(path)
        payload = json.loads(path.read_text())
        provenance.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": size,
                "pairs": payload["pairs"],
                "tokens_changed": payload["tokens_changed"],
                "repaired": payload["repaired"],
                "break_even_yield": payload["break_even_yield"],
            }
        )
        total_tokens_published += int(payload["tokens_changed"])
        total_repaired_published += int(payload["repaired"])

        for row in payload["per_pair"]:
            entries: list[Entry] = []
            for raw in row["separation_sweep"]:
                entry = Entry(
                    separation=int(raw["accept_separation"]),
                    keep_fraction=float(raw["keep_fraction"]),
                    tokens=int(raw["tokens"]),
                    repaired=int(raw["repaired"]),
                    flips_after=int(raw["flips_after"]),
                    net_modelled=float(raw["net_delta_S"]),
                )
                rebuilt = entry.net_at(JG3_ACCEPT_BITS_PER_TOKEN)
                residual = abs(rebuilt - entry.net_modelled)
                control1_max_residual = max(control1_max_residual, residual)
                entries.append(entry)

            if not entries:
                raise Fs3Error(f"pair {row['pair']} has an empty separation sweep")

            chosen_key = (
                int(row["accept_separation_chosen"]),
                float(row["keep_fraction_chosen"]),
            )
            matches = [
                e
                for e in entries
                if (e.separation, e.keep_fraction) == chosen_key
                and e.tokens == int(row["tokens_changed"])
                and e.repaired == int(row["repaired"])
            ]
            if len(matches) != 1:
                raise Fs3Error(
                    f"pair {row['pair']}: chosen configuration {chosen_key} with "
                    f"tokens={row['tokens_changed']} repaired={row['repaired']} "
                    f"matched {len(matches)} sweep entries; refusing to re-price a "
                    "ledger whose selection cannot be located"
                )
            chosen = matches[0]
            argmin = min(entries, key=lambda e: e.net_modelled)
            if argmin.net_modelled < chosen.net_modelled - 1e-18:
                raise Fs3Error(
                    f"pair {row['pair']}: recorded choice is not the modelled argmin "
                    f"({chosen.net_modelled:.6e} vs {argmin.net_modelled:.6e}); the "
                    "selection rule is not the one this module re-prices"
                )
            sweeps.append(
                PairSweep(pair=int(row["pair"]), chosen=chosen, entries=tuple(entries))
            )

    if control1_max_residual > 1e-16:
        raise Fs3Error(
            f"CONTROL 1 FAILED: max |rebuilt - retained| net_delta_S is "
            f"{control1_max_residual:.3e}; this module's arithmetic is not the "
            "ledger's arithmetic"
        )

    tokens_chosen = sum(s.chosen.tokens for s in sweeps)
    repaired_chosen = sum(s.chosen.repaired for s in sweeps)
    if tokens_chosen != total_tokens_published:
        raise Fs3Error(
            f"CONTROL 3 FAILED: chosen tokens {tokens_chosen} != published "
            f"{total_tokens_published}"
        )
    if repaired_chosen != total_repaired_published:
        raise Fs3Error(
            f"CONTROL 3 FAILED: chosen repaired {repaired_chosen} != published "
            f"{total_repaired_published}"
        )

    controls = {
        "control_1_net_delta_S_rebuilt_from_repaired_and_tokens": {
            "max_abs_residual": control1_max_residual,
            "verdict": "PASS",
        },
        "control_2_recorded_choice_is_the_modelled_argmin": {
            "pairs_checked": len(sweeps),
            "verdict": "PASS",
        },
        "control_3_chosen_totals_match_published_shard_totals": {
            "tokens": tokens_chosen,
            "repaired": repaired_chosen,
            "verdict": "PASS",
        },
        "shards": provenance,
    }
    return sweeps, controls


def measure_real_bits_per_token(
    bits_candidate: Path,
    bits_control: Path,
    edits_npz: Path,
    tokens_changed: int,
) -> dict[str, Any]:
    """MEASURE the realised token price from retained re-encoder output.

    ``bits_candidate`` and ``bits_control`` are per-frame code-bit arrays emitted by
    ``ddm_jg2_tail_reencode``, whose unedited control reproduces the shipped token
    stream byte-identically.  This is a measurement of the coder, not a model of it.
    """
    receipts = {}
    for name, path in (
        ("bits_candidate", bits_candidate),
        ("bits_control", bits_control),
        ("edits_npz", edits_npz),
    ):
        if not path.exists():
            raise Fs3Error(f"retained array missing: {path}")
        digest, size = sha256_of(path)
        receipts[name] = {"path": str(path), "sha256": digest, "bytes": size}

    candidate = np.load(bits_candidate).astype(np.float64)
    control = np.load(bits_control).astype(np.float64)
    if candidate.shape != (N_PAIRS,) or control.shape != (N_PAIRS,):
        raise Fs3Error(
            f"per-frame bit arrays have shapes {candidate.shape} / {control.shape}, "
            f"expected ({N_PAIRS},)"
        )
    delta_bytes = (candidate - control) / 8.0

    with np.load(edits_npz) as handle:
        edited_pairs = sorted(int(k) for k in handle.files)
    edited = np.zeros(N_PAIRS, dtype=bool)
    edited[edited_pairs] = True

    bytes_edited = float(delta_bytes[edited].sum())
    bytes_bleed = float(delta_bytes[~edited].sum())

    if tokens_changed <= 0:
        raise Fs3Error("tokens_changed must be positive to form a per-token price")

    return {
        "receipts": receipts,
        "edited_pairs": int(edited.sum()),
        "tokens_changed": tokens_changed,
        "delta_bytes_edited_pairs": bytes_edited,
        "delta_bytes_unedited_pairs_context_bleed": bytes_bleed,
        "delta_bytes_all_frames": bytes_edited + bytes_bleed,
        "real_bits_per_token_edited_pairs": bytes_edited * 8.0 / tokens_changed,
        "real_bits_per_token_all_frames": (bytes_edited + bytes_bleed)
        * 8.0
        / tokens_changed,
        "jg3_accept_bits_per_token": JG3_ACCEPT_BITS_PER_TOKEN,
        "overcharge_ratio_accept_over_real": JG3_ACCEPT_BITS_PER_TOKEN
        / (bytes_edited * 8.0 / tokens_changed),
        "axis": "EXACT -- re-encoder code bits, byte-identical unedited control",
    }


def census_at_price(
    sweeps: Sequence[PairSweep],
    bits_per_token: float,
    shipping: frozenset[int] | None = None,
) -> dict[str, Any]:
    """Re-select every pair's configuration at ``bits_per_token`` and diff.

    The banked configurations were already PAID at the real price (jg5 built the
    body), so the only new credit is the RE-SELECTION delta -- both terms priced at
    the same rate.

    ``shipping`` is jg5's admitted-pair set.  A pair jg5 DROPPED reverts to the base
    render and base carrier, so re-selecting its configuration changes nothing in the
    archive.  Counting those rows in the headline would be a phantom credit, so the
    split is reported and the SHIPPING total is the one that means anything.
    """
    reopened: list[dict[str, Any]] = []
    total_gain = 0.0
    tokens_delta = 0
    repaired_delta = 0

    for sweep in sweeps:
        chosen_real = sweep.chosen.net_at(bits_per_token)
        best = min(sweep.entries, key=lambda e: e.net_at(bits_per_token))
        best_real = best.net_at(bits_per_token)
        gain = best_real - chosen_real
        if gain >= -1e-18:
            continue
        total_gain += gain
        tokens_delta += best.tokens - sweep.chosen.tokens
        repaired_delta += best.repaired - sweep.chosen.repaired
        reopened.append(
            {
                "pair": sweep.pair,
                "chosen": {
                    "separation": sweep.chosen.separation,
                    "keep_fraction": sweep.chosen.keep_fraction,
                    "tokens": sweep.chosen.tokens,
                    "repaired": sweep.chosen.repaired,
                    "net_modelled": sweep.chosen.net_modelled,
                    "net_repriced": chosen_real,
                },
                "reopened": {
                    "separation": best.separation,
                    "keep_fraction": best.keep_fraction,
                    "tokens": best.tokens,
                    "repaired": best.repaired,
                    "net_modelled": best.net_modelled,
                    "net_repriced": best_real,
                },
                "delta_S_reselect": gain,
                "delta_tokens": best.tokens - sweep.chosen.tokens,
                "delta_repaired": best.repaired - sweep.chosen.repaired,
                "ships": None if shipping is None else (sweep.pair in shipping),
            }
        )

    reopened.sort(key=lambda r: r["delta_S_reselect"])
    ship_rows = [r for r in reopened if r["ships"]] if shipping is not None else []
    drop_rows = (
        [r for r in reopened if r["ships"] is False] if shipping is not None else []
    )
    ship_gain = sum(r["delta_S_reselect"] for r in ship_rows)

    return {
        "bits_per_token": bits_per_token,
        "reopen_count": len(reopened),
        "delta_S_reselect_total": total_gain,
        "delta_tokens": tokens_delta,
        "delta_repaired": repaired_delta,
        "shipping_reopen_count": len(ship_rows) if shipping is not None else None,
        "shipping_delta_S_reselect": ship_gain if shipping is not None else None,
        "shipping_delta_tokens": (
            sum(r["delta_tokens"] for r in ship_rows) if shipping is not None else None
        ),
        "shipping_delta_repaired": (
            sum(r["delta_repaired"] for r in ship_rows)
            if shipping is not None
            else None
        ),
        "dropped_reopen_count": len(drop_rows) if shipping is not None else None,
        "dropped_delta_S_reselect_ships_nothing": (
            sum(r["delta_S_reselect"] for r in drop_rows)
            if shipping is not None
            else None
        ),
        "shipping_multiple_of_bar": (
            abs(ship_gain) / ADMISSION_BAR_S if shipping is not None else None
        ),
        "shipping_clears_admission_bar": (
            ship_gain < -ADMISSION_BAR_S if shipping is not None else None
        ),
        "rows": reopened,
    }


def highest_price_with_a_reopen(
    sweeps: Sequence[PairSweep], shipping: frozenset[int] | None = None
) -> float | None:
    """The largest token price at which ANY pair still changes its selection.

    This is the falsifier's own number: how cheap would tokens have to get before
    the flat prior's error starts costing us a configuration?  Derived by bisection
    on the retained grid, not chosen.
    """
    lo, hi = 0.0, JG3_ACCEPT_BITS_PER_TOKEN

    def count(price: float) -> int:
        cen = census_at_price(sweeps, price, shipping)
        key = "shipping_reopen_count" if shipping is not None else "reopen_count"
        return int(cen[key] or 0)

    if count(lo) == 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if count(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def price_required_to_clear_bar(
    sweeps: Sequence[PairSweep], shipping: frozenset[int]
) -> float | None:
    """The token price at which the SHIPPING re-selection gain first clears the bar.

    The falsifier for the honest close: if the real price were this low, the reopen
    would be admissible on seg+rate alone.  Compare it against the MEASURED price to
    see how far the row actually is.
    """
    lo, hi = 0.0, JG3_ACCEPT_BITS_PER_TOKEN

    def clears(price: float) -> bool:
        return bool(
            census_at_price(sweeps, price, shipping)["shipping_clears_admission_bar"]
        )

    if not clears(lo):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if clears(mid):
            lo = mid
        else:
            hi = mid
    return lo


def compose_legs(
    sweeps: Sequence[PairSweep],
    shipping: frozenset[int],
    *,
    identify_at: float,
    price_at: float,
) -> dict[str, Any]:
    """Compose the re-selection's seg+rate leg with its pose and carrier legs.

    The pair set is IDENTIFIED at ``identify_at`` (the population price, which is
    what selected the pairs the marginal price was then measured on) and SCORED at
    ``price_at`` (that measured marginal).  Identifying at the cheap price would
    apply a price measured on 38 pairs to all 573 -- the same population defect
    fs1 sec.4 refused for the pose actuator, and it is refused here.

    Each leg carries its own authority and they are NOT collapsed into a headline:

    * seg+rate -- MEASURED (jg3 realized cells; this arm's real re-encode price)
    * pose     -- DERIVED cross-sectional, an OLS slope over pairs with different
                  token counts, which is not the same object as adding tokens to a
                  GIVEN pair
    * carrier  -- DERIVED by extrapolating leg 2's single built-and-stat'd point
                  (45 B / 454 pairs) to this pair count.  The MEASURED label was
                  withdrawn (rv17 W3-F7) and the leg was later measured EXACTLY
                  ZERO when the carrier is left unchanged; see
                  ``CARRIER_LABEL_SUPERSEDED``.
    """
    identified = {
        row["pair"]
        for row in census_at_price(sweeps, identify_at, shipping)["rows"]
        if row["ships"]
    }
    by_pair = {s.pair: s for s in sweeps}
    seg_rate = 0.0
    added_tokens = 0
    pairs = 0
    for pair in sorted(identified):
        sweep = by_pair[pair]
        chosen = sweep.chosen.net_at(price_at)
        best = min(sweep.entries, key=lambda e: e.net_at(price_at))
        if best.net_at(price_at) < chosen:
            seg_rate += best.net_at(price_at) - chosen
            added_tokens += best.tokens - sweep.chosen.tokens
            pairs += 1

    new_mean = LIVE_MEAN_D_POSE + POSE_D_POSE_PER_ADDED_TOKEN * added_tokens / N_PAIRS
    pose = math.sqrt(10.0 * new_mean) - math.sqrt(10.0 * LIVE_MEAN_D_POSE)
    carrier = pairs * CARRIER_COMPENSATION_BYTES_PER_PAIR * S_PER_ARCHIVE_BYTE
    net = seg_rate + pose + carrier
    return {
        "pairs_identified_at_bits_per_token": identify_at,
        "pairs_scored_at_bits_per_token": price_at,
        "pairs": pairs,
        "added_tokens": added_tokens,
        "leg_seg_plus_rate_MEASURED": seg_rate,
        "leg_pose_DERIVED_cross_sectional": pose,
        "leg_carrier_DERIVED_extrapolated_leg2": carrier,
        "carrier_label_superseded": CARRIER_LABEL_SUPERSEDED,
        "net_delta_S": net,
        "multiple_of_bar": abs(net) / ADMISSION_BAR_S,
        "clears_admission_bar": net < -ADMISSION_BAR_S,
        "boundary": (
            "the measured price is that of these pairs' EXISTING edits; the "
            "additional lower-ranked tokens a re-selection adds are UNMEASURED, and "
            "materialising the reopened configurations needs a re-screen because "
            "jg3 did not retain its per-site candidate gains"
        ),
    }


def load_shipping_set(path: Path, sweeps: Sequence[PairSweep]) -> frozenset[int]:
    """jg5's admitted-pair set, with a control that it is a subset of the edited pairs."""
    if not path.exists():
        raise Fs3Error(f"jg5 kept-pair set missing: {path}")
    digest, _ = sha256_of(path)
    kept = frozenset(int(p) for p in json.loads(path.read_text()))
    edited = {s.pair for s in sweeps}
    stray = kept - edited
    if stray:
        raise Fs3Error(
            f"jg5 kept {len(stray)} pairs that jg3 never edited (e.g. "
            f"{sorted(stray)[:5]}); the two ledgers are not the same population"
        )
    load_shipping_set.digest = digest  # type: ignore[attr-defined]
    return kept


def run_census(args: argparse.Namespace) -> int:
    shards = [Path(p) for p in args.shards]
    sweeps, controls = load_shards(shards)
    tokens_chosen = sum(s.chosen.tokens for s in sweeps)
    shipping = load_shipping_set(Path(args.kept_pairs), sweeps)

    real = measure_real_bits_per_token(
        Path(args.bits_candidate),
        Path(args.bits_control),
        Path(args.edits_npz),
        tokens_chosen,
    )
    b_real = real["real_bits_per_token_edited_pairs"]

    prices = {
        "jg3_accept_prior_control": JG3_ACCEPT_BITS_PER_TOKEN,
        "MEASURED_marginal_of_the_reopened_38": MEASURED_MARGINAL_BITS_PER_TOKEN,
        "MEASURED_real_full_set": b_real,
        "jg5_realised_455_subset": 3.8373,
        "charter_assumed_0p88_of_prior": 0.88 * JG3_ACCEPT_BITS_PER_TOKEN,
        "fs2_ranker_mean_ratio_0p8133_of_prior": 0.8133 * JG3_ACCEPT_BITS_PER_TOKEN,
        "hm1_logit_aggregate_1p91": 1.91,
        "free_tokens_absurd_lower_bound": 0.0,
    }
    censuses = {
        name: census_at_price(sweeps, p, shipping) for name, p in prices.items()
    }

    flip = highest_price_with_a_reopen(sweeps, shipping)
    clearing = price_required_to_clear_bar(sweeps, shipping)

    report = {
        "schema": "ddm_fs3_real_price_reopen_census.v1",
        "arm": "ddm_fs3",
        "axis": (
            "seg leg MEASURED (jg3 realized repaired-cell counts through the "
            "receiver forward + frozen CPU SegNet); rate leg EXACT (jg2 re-encoder "
            "code bits, byte-identical unedited control); pose leg NOT PRICED HERE"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_archive_bytes": POINTER_ARCHIVE_BYTES,
        "admission_bar_S": ADMISSION_BAR_S,
        "admission_bar_bytes": ADMISSION_BAR_S / S_PER_ARCHIVE_BYTE,
        "S_per_archive_byte": S_PER_ARCHIVE_BYTE,
        "S_per_seg_cell": S_PER_SEG_CELL,
        "bits_per_seg_cell": BITS_PER_SEG_CELL,
        "pairs": len(sweeps),
        "sweep_entries_total": sum(len(s.entries) for s in sweeps),
        "controls": controls,
        "shipping_set": {
            "path": str(args.kept_pairs),
            "sha256": getattr(load_shipping_set, "digest", None),
            "pairs_admitted_by_jg5": len(shipping),
            "pairs_dropped_by_jg5": len(sweeps) - len(shipping),
            "note": (
                "a pair jg5 DROPPED ships base tokens and the base carrier, so "
                "re-selecting its configuration changes nothing in the archive"
            ),
        },
        "real_price_measurement": real,
        "surface_A_per_site_inner_gate": {
            "location": "experiments/ddm_jg3_joint_solve.py:695",
            "priced_with": "LogitPrice log2(p_old/p_new), per move",
            "rejected_moves_retained": False,
            "note": (
                "rejected with a bare `continue`; no per-move record exists on disk, "
                "so this surface cannot be censused at $0 and is reported as "
                "not-retained rather than estimated"
            ),
        },
        "census_by_price": censuses,
        "highest_price_with_a_shipping_reopen": flip,
        "highest_price_as_fraction_of_accept_prior": (
            flip / JG3_ACCEPT_BITS_PER_TOKEN if flip is not None else None
        ),
        "price_required_for_shipping_reopen_to_clear_bar": clearing,
        "price_required_as_fraction_of_measured_real": (
            clearing / b_real if clearing is not None else None
        ),
        "composition_at_the_measured_marginal_price": compose_legs(
            sweeps,
            shipping,
            identify_at=b_real,
            price_at=MEASURED_MARGINAL_BITS_PER_TOKEN,
        ),
        "note_on_the_cheap_price_column": (
            "the MEASURED_marginal row in census_by_price applies a price measured "
            "on 38 pairs to all 573; it is shown for shape only and is NOT a "
            "supported result. The supported number is "
            "composition_at_the_measured_marginal_price, which identifies at the "
            "population price and scores at the measured marginal."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False))

    control1 = controls["control_1_net_delta_S_rebuilt_from_repaired_and_tokens"]
    print(f"pairs={len(sweeps)} sweep_entries={report['sweep_entries_total']}")
    print(f"CONTROL 1 max residual = {control1['max_abs_residual']:.3e}")
    print(
        f"MEASURED real bits/token = {b_real:.6f} "
        f"(jg3 charged {JG3_ACCEPT_BITS_PER_TOKEN}, "
        f"overcharge {real['overcharge_ratio_accept_over_real']:.6f}x)"
    )
    print(
        f"jg5 ships {len(shipping)} of {len(sweeps)} edited pairs; a reopen on a "
        "DROPPED pair changes no archive byte"
    )
    header = (
        f"  {'price name':44s} {'bits/tok':>8s} {'all':>4s} {'SHIP':>5s} "
        f"{'SHIPPING dS':>13s} {'x bar':>7s} {'dtok':>6s}"
    )
    print(header)
    for name, cen in censuses.items():
        print(
            f"  {name:44s} {cen['bits_per_token']:8.4f} "
            f"{cen['reopen_count']:4d} {cen['shipping_reopen_count']:5d} "
            f"{cen['shipping_delta_S_reselect']:13.6e} "
            f"{cen['shipping_multiple_of_bar']:7.3f} "
            f"{cen['shipping_delta_tokens']:+6d}"
        )
    if flip is None:
        print("highest price with a shipping reopen: NONE even at 0.0 bits/token")
    else:
        print(
            f"highest price with a shipping reopen: {flip:.6f} bits/token "
            f"= {flip / JG3_ACCEPT_BITS_PER_TOKEN:.4f}x the accept prior"
        )
    if clearing is None:
        print(
            "price required for the shipping reopen to clear the bar: "
            "UNREACHABLE -- it does not clear even at 0.0 bits/token"
        )
    else:
        print(
            f"price required for the shipping reopen to clear the bar: "
            f"{clearing:.6f} bits/token = {clearing / b_real:.4f}x the MEASURED real "
            "price"
        )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    census = sub.add_parser(
        "census", help="$0 re-selection census over jg3's retained sweep"
    )
    census.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    census.add_argument("--bits-candidate", default=DEFAULT_BITS_CANDIDATE)
    census.add_argument("--bits-control", default=DEFAULT_BITS_CONTROL)
    census.add_argument("--edits-npz", default=DEFAULT_EDITS_NPZ)
    census.add_argument("--kept-pairs", default=DEFAULT_KEPT_PAIRS)
    census.add_argument(
        "--out",
        default="/Volumes/APDataStore/pact/ddm_fs3/FS3_REOPEN_CENSUS.json",
    )
    census.set_defaults(func=run_census)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
