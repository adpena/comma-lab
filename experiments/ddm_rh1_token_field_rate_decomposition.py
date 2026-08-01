"""ddm_rh1 (#853): decompose a token-field byte gap, and race the SMEVR base rule.

Two measurements, both `$0`, both scorer-free, both on real shipped archive bytes.

**A. Field-gap decomposition by SUBSTITUTION.**  Two token fields priced through
the *same* r7 coder differ by some number of bytes.  Narrative attribution of that
gap ("it is the learned structure", "it is the grammar") is exactly the error class
that let a headline stand at 2.41x understated.  ``decompose_field_gap`` instead
rebuilds controlled hybrids -- ``reconstruct_mode_delta`` lets the *mode base* of one
field be paired with the *event field* of the other -- and re-encodes each, so the
split is measured.  The two swaps are deliberately non-additive: SMEVR's residual is
``(value - base) mod levels``, so base and events are coupled and the interaction
term is reported rather than hidden.

**B. The SMEVR base rule is an unraced generic default.**
``ddm_r7_token_coder.encode_token_codes`` hardcodes ``factor_mode_delta`` -- a
per-cell *temporal mode*, chosen because it is deterministic, never raced for bytes.
The mode maximises the count of exact zeros (which the occupancy stream likes) and is
blind to the value stream's rank cost.  ``race_base_rule`` scores a derived
one-parameter-per-axis family

    base(cell) = argmin_b  sum_s hist[cell, s] * C((s - b) mod levels)
    C(r)       = 0 if r == 0 else alpha + circular_distance(r) ** p

against **real encoder bytes**, with the incumbent recovered as the ``alpha -> inf``
corner.  Losslessness is asserted at every point, so seg and pose are unchanged by
construction: this is a pure rate axis and needs no scorer slot.

Shipping a non-mode base requires a receiver change -- ``decode_token_codes`` at
``verify="canonical"`` re-encodes with ``factor_mode_delta`` and refuses anything
else.  The base is already stored and already counted, so the change is rule-118
free, but it is a format change and not a free recode.  This module measures the
prize; it does not modify the coder.

Usage::

    .venv/bin/python experiments/ddm_rh1_token_field_rate_decomposition.py \\
        --field a=/path/to/archive_a.zip --field b=/path/to/archive_b.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ddm_r7_token_coder as r7

#: Contest rate denominator, ``upstream/evaluate.py``.
RATE_DENOMINATOR: int = 37_545_489
DEFAULT_LEVELS: int = 16


class RH1Error(RuntimeError):
    """Raised when an input archive or field fails a structural precondition."""


# --------------------------------------------------------------------------- io


def load_token_field(archive: str | Path) -> np.ndarray:
    """Return the token code lattice from either shipped archive grammar.

    Handles the 6-member composed ``v3_warp`` grammar (bare ``state/tokens.dr7t``
    DR7T member) and the 2-member ``ddm_tr1_runtime_archive.v1`` packet.  The DR7T
    path decodes at ``verify="canonical"``, which re-encodes and refuses on any
    difference -- so a successful return is itself proof of the shipped codec.
    """
    path = Path(archive)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "state/tokens.dr7t" in names:
            return r7.decode_token_codes(zf.read("state/tokens.dr7t"), verify=r7.VERIFY_CANONICAL)
    from tac.optimization import ddm_tr1_runtime as tr1

    return tr1.parse_archive(path.read_bytes()).packet.token_codes


def _delta_for(values: np.ndarray, base: np.ndarray, levels: int) -> np.ndarray:
    """Residual for an arbitrary base, fail-closed on anything the format refuses.

    The lossless check alone is NOT sufficient and must not be mistaken for one:
    ``(v - b) mod levels`` reconstructs for *any* integer base, including bases
    outside the declared lattice, so that assertion passes vacuously.  The r7
    decoder refuses ``base >= levels`` ("mode base exceeds levels"), so pricing
    such a base would be costing an unshippable frame.  Both are checked here.
    """
    if base.shape != values.shape[1:]:
        raise RH1Error(f"base shape {base.shape} does not match field cells {values.shape[1:]}")
    if np.any(base >= levels) or np.any(base < 0):
        raise RH1Error(f"base leaves the declared lattice (levels={levels}); the r7 decoder would refuse it")
    delta = (values.astype(np.int16) - base[None].astype(np.int16)) % levels
    delta = np.ascontiguousarray(delta.astype(np.uint8))
    if not np.array_equal(r7.reconstruct_mode_delta(base, delta, levels), values):
        raise RH1Error("lossless reconstruction check failed for the proposed base")
    return delta


def framed_bytes(values: np.ndarray, base: np.ndarray, levels: int) -> tuple[int, int, int]:
    """Exact framed size for an arbitrary base, matching ``encode_token_codes`` layout.

    Returns ``(framed, base_stream, delta_stream)``.  With ``base`` equal to the
    deterministic mode this reproduces the shipped member byte count exactly.
    """
    delta = _delta_for(values, base, levels)
    base_stream = r7._raw_lzma_encode(r7.pack_nibbles(base))
    delta_stream = r7._encode_smevr(base, delta, levels)
    return r7.HEADER.size + len(base_stream) + len(delta_stream), len(base_stream), len(delta_stream)


def rate_delta_s(byte_delta: int) -> float:
    """Score delta on the contest rate term for a byte delta."""
    return 25.0 * byte_delta / RATE_DENOMINATOR


# ------------------------------------------------------------------ decomposition


@dataclass(frozen=True, slots=True)
class FieldGap:
    """Measured split of a token-field byte gap into base, events, and interaction."""

    left_framed: int
    right_framed: int
    left_base_bytes: int
    right_base_bytes: int
    left_event_rate: float
    right_event_rate: float
    left_const_cell_fraction: float
    right_const_cell_fraction: float
    hybrid_left_base_right_events: int
    hybrid_right_base_left_events: int

    @property
    def total_byte_delta(self) -> int:
        return self.right_framed - self.left_framed

    @property
    def event_swap_byte_delta(self) -> int:
        """Bytes moved by exchanging ONLY the event field."""
        return self.hybrid_left_base_right_events - self.left_framed

    @property
    def base_swap_byte_delta(self) -> int:
        """Bytes moved by exchanging ONLY the mode base."""
        return self.hybrid_right_base_left_events - self.left_framed

    @property
    def interaction_byte_delta(self) -> int:
        """Non-additive remainder: base and events are coupled through ``mod levels``."""
        return self.total_byte_delta - self.event_swap_byte_delta - self.base_swap_byte_delta


def decompose_field_gap(left: np.ndarray, right: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> FieldGap:
    """Attribute the byte gap between two token fields by controlled substitution."""
    if left.shape != right.shape:
        raise RH1Error(f"field geometry differs ({left.shape} vs {right.shape}); the comparison is void")

    base_l, delta_l = r7.factor_mode_delta(left, levels)
    base_r, delta_r = r7.factor_mode_delta(right, levels)
    nz_l, nz_r = delta_l != 0, delta_r != 0
    framed_l, base_bytes_l, _ = framed_bytes(left, base_l, levels)
    framed_r, base_bytes_r, _ = framed_bytes(right, base_r, levels)

    return FieldGap(
        left_framed=framed_l,
        right_framed=framed_r,
        left_base_bytes=base_bytes_l,
        right_base_bytes=base_bytes_r,
        left_event_rate=float(nz_l.mean()),
        right_event_rate=float(nz_r.mean()),
        left_const_cell_fraction=float((~nz_l.any(axis=0)).mean()),
        right_const_cell_fraction=float((~nz_r.any(axis=0)).mean()),
        hybrid_left_base_right_events=framed_bytes(
            r7.reconstruct_mode_delta(base_l, delta_r, levels), base_l, levels
        )[0],
        hybrid_right_base_left_events=framed_bytes(
            r7.reconstruct_mode_delta(base_r, delta_l, levels), base_r, levels
        )[0],
    )


# --------------------------------------------------------------------- base race


def _cell_histogram(values: np.ndarray, levels: int) -> np.ndarray:
    """Per-cell symbol counts, fail-closed on values outside the declared lattice.

    Without this check an out-of-lattice value is simply absent from every bin and the
    argmin is computed from a silently incomplete histogram -- wrong, with no symptom.
    ``framed_bytes`` would eventually catch it via the lossless assertion, but
    ``propose_base`` is public and must not have a silent path of its own.
    """
    if values.size and (int(values.max()) >= levels or int(values.min()) < 0):
        raise RH1Error(f"token values leave the declared lattice (levels={levels})")
    flat = values.reshape(values.shape[0], -1)
    return np.stack([(flat == s).sum(axis=0) for s in range(levels)], axis=1).astype(np.float64)


def propose_base(values: np.ndarray, *, alpha: float, exponent: float, levels: int) -> np.ndarray:
    """Per-cell argmin base under ``C(r) = alpha + circular_distance(r) ** exponent``.

    The reduction is written as an explicit accumulation rather than ``hist @ shift.T``
    deliberately.  The matmul form is numerically identical (verified: max abs diff 0.0,
    argmins equal to a fully manual per-cell recompute) but Apple's Accelerate BLAS
    raises spurious ``divide by zero``/``overflow``/``invalid`` RuntimeWarnings on this
    shape.  A landed instrument that prints warnings it expects to be ignored trains
    readers to ignore warnings, so the warning is removed rather than filtered.
    """
    circ = np.array([min(r, levels - r) for r in range(levels)], dtype=np.float64)
    cost = np.where(np.arange(levels) == 0, 0.0, alpha + circ**exponent)
    hist = _cell_histogram(values, levels)
    totals = np.stack(
        [(hist * np.array([cost[(s - b) % levels] for s in range(levels)])).sum(axis=1) for b in range(levels)],
        axis=1,
    )
    return totals.argmin(axis=1).astype(np.uint8).reshape(values.shape[1:])


def race_base_rule(
    values: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    exponents: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
) -> list[dict[str, float | int | str | None]]:
    """Score the derived base-rule family against real encoder bytes.

    The incumbent deterministic mode is included as ``rule="mode (incumbent)"``.
    Rows are returned sorted by framed bytes, cheapest first.

    The incumbent's ``alpha`` is reported as ``None``, not ``float("inf")``: it is the
    ``alpha -> inf`` limit of the family, and ``inf`` serialises to a bare ``Infinity``
    token that strict JSON parsers reject -- a receipt has to stay machine-readable.
    """
    base_mode, _ = r7.factor_mode_delta(values, levels)
    incumbent, _, _ = framed_bytes(values, base_mode, levels)
    rows: list[dict[str, float | int | str | None]] = [
        {
            "rule": "mode (incumbent)",
            "alpha": None,
            "exponent": None,
            "framed": incumbent,
            "byte_delta": 0,
            "rate_delta_s": 0.0,
            "cells_moved": 0,
        }
    ]
    for exponent in exponents:
        for alpha in alphas:
            base = propose_base(values, alpha=alpha, exponent=exponent, levels=levels)
            framed, _, _ = framed_bytes(values, base, levels)
            rows.append(
                {
                    "rule": f"alpha={alpha},p={exponent}",
                    "alpha": alpha,
                    "exponent": exponent,
                    "framed": framed,
                    "byte_delta": framed - incumbent,
                    "rate_delta_s": rate_delta_s(framed - incumbent),
                    "cells_moved": int((base != base_mode).sum()),
                }
            )
    rows.sort(key=lambda row: int(row["framed"]))
    return rows


def codec_sweep(values: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> list[tuple[str, int]]:
    """Framed bytes for every registered r7 codec, cheapest first."""
    out: list[tuple[str, int]] = []
    for codec in r7.CODEC_IDS:
        try:
            out.append((codec, len(r7.encode_token_codes(values, levels=levels, codec=codec))))
        except Exception as exc:
            out.append((f"{codec} [{type(exc).__name__}]", -1))
    return sorted(out, key=lambda row: (row[1] < 0, row[1]))


# -------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--field",
        action="append",
        required=True,
        metavar="NAME=ARCHIVE",
        help="labelled archive to load; pass twice to also decompose the gap between them",
    )
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--skip-base-race", action="store_true")
    parser.add_argument("--skip-codec-sweep", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    fields: dict[str, np.ndarray] = {}
    for spec in args.field:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--field expects NAME=ARCHIVE, got {spec!r}")
        fields[name] = load_token_field(path)
        print(f"[field] {name}: shape={fields[name].shape} levels<={args.levels}")

    report: dict[str, object] = {"rate_denominator": RATE_DENOMINATOR, "levels": args.levels}

    if len(fields) == 2:
        (ln, lv), (rn, rv) = fields.items()
        gap = decompose_field_gap(lv, rv, levels=args.levels)
        # asdict() carries only the stored fields; the derived legs are the numbers a
        # post-hoc consumer actually cites, so they are persisted rather than recomputed.
        report["field_gap"] = {
            "left": ln,
            "right": rn,
            **asdict(gap),
            "total_byte_delta": gap.total_byte_delta,
            "event_swap_byte_delta": gap.event_swap_byte_delta,
            "base_swap_byte_delta": gap.base_swap_byte_delta,
            "interaction_byte_delta": gap.interaction_byte_delta,
            "total_rate_delta_s": rate_delta_s(gap.total_byte_delta),
        }
        print(f"\n== gap {ln} -> {rn} ==")
        print(f"  framed           {gap.left_framed:>9,} -> {gap.right_framed:>9,}   "
              f"{gap.total_byte_delta:+,} B   dS {rate_delta_s(gap.total_byte_delta):+.7f}")
        print(f"  event rate       {gap.left_event_rate:.5f} -> {gap.right_event_rate:.5f}")
        print(f"  const cells      {gap.left_const_cell_fraction:.5f} -> {gap.right_const_cell_fraction:.5f}")
        print(f"  events-only swap {gap.event_swap_byte_delta:+,} B")
        print(f"  base-only swap   {gap.base_swap_byte_delta:+,} B")
        print(f"  interaction      {gap.interaction_byte_delta:+,} B  (base and events are coupled mod levels)")

    for name, values in fields.items():
        if not args.skip_codec_sweep:
            sweep = codec_sweep(values, levels=args.levels)
            report.setdefault("codec_sweep", {})[name] = sweep  # type: ignore[index]
            head = sweep[0]
            print(f"\n== codec sweep [{name}] == argmin {head[0]} @ {head[1]:,} B "
                  f"(next {sweep[1][0]} +{sweep[1][1] - head[1]:,})")
        if not args.skip_base_race:
            rows = race_base_rule(values, levels=args.levels)
            report.setdefault("base_race", {})[name] = rows  # type: ignore[index]
            best = rows[0]
            print(f"== base race  [{name}] == argmin {best['rule']} @ {int(best['framed']):,} B "
                  f"({int(best['byte_delta']):+,} B, dS {float(best['rate_delta_s']):+.7f})")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
