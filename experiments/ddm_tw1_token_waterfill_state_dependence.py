#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_tw1 (#869): is the waterfill's per-unit byte price a CONSTANT or a FUNCTION OF STATE?

Operator question (2026-08-01): *"It's also hard to know what the water fill at
higher rates might do with svemr and jrd"* / *"Perhaps we can actually determine
that on a token by token basis."*

The wr1 reverse-waterfill (`ddm_wr1_reverse_waterfill_20260729.md`) prices a
DESCENT: it re-encodes the whole dropped grid per tranche, so every published row
is archive-faithful.  What it never measured is the **per-unit exchange rate** —
the marginal byte price of one more drop — nor whether that price is stable as
you move along the curve.  Every allocator we might build on top of the waterfill
(a price table, a knapsack, a Lagrangian sweep) silently assumes it is stable.
That assumption is what this module measures.

SMEVR is a CONTEXT coder: `ddm_r7_token_coder.encode_token_codes` factors the
lattice into a per-cell temporal ``base`` plus a ``(value - base) mod levels``
residual, then codes the residual under contexts derived from its neighbours.  A
drop therefore changes the coding context of cells that were NOT dropped.  If
that coupling is material, a one-shot per-cell price table is a LINEARIZATION of
a non-separable function and any allocator built on it is mis-specified.

Three measurements, all `$0`, all scorer-free, all through the REAL shipped coder:

**A. `marginal_price_by_state`** — the same held-out cells priced from several
   drop states.  Reports the per-cell marginal and its spread ACROSS states.
   This is the operator's "token by token" reading of the waterfill.

**B. `additivity_defect`** — sum of singleton marginals vs the jointly measured
   saving for the same set.  A non-zero defect is the exact quantity that makes
   "price once, then allocate" wrong; its SIGN says whether the linearization
   over- or under-states the achievable saving.

**C. `coder_race_by_state`** — every r7 codec re-priced at each drop state.
   The lossless-coder race was decided at the UNDROPPED field (r7: SMEVR wins).
   Dropping ~40-80% of the residual mass changes the source statistics, so the
   winner is not guaranteed to be invariant; this re-runs the race where the
   waterfill actually leaves the stream.

Controls are mandatory and run before any reading is emitted (`run_controls`):
a positive control (re-encoding the untouched field must reproduce the shipped
member byte-for-byte), a null control (dropping the empty set must move exactly
0 bytes), and a state-reconstruction control (the k=486 / k=600 states must
reproduce wr1's published `tokens_bytes`).  A failed control aborts; readings
from an unvalidated meter are not emitted.

Nothing here is a score.  Bytes are MEASURED; d_seg and d_pose are NOT computed
by this module and are not claimed.  Axis `[macOS-CPU advisory, rate-only]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_r7_token_coder as r7

SCHEMA = "ddm_tw1_token_waterfill_state_dependence.v1"

LEVELS = 16
# upstream/evaluate.py rate term: 25 * archive_bytes / 37_545_489
UNCOMPRESSED = 37_545_489

PFS1_ARCHIVE = (
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
    "submissions/pfs1/archive.zip"
)
WR1_CELLS = "/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_cell_records.json"
WR1_RECEIPT = "/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_descent_receipt.json"

# wr1 §0: the archive members that are NOT the token stream.
ARCHIVE_FLOOR_BYTES = 12_743
# wr1 §0 / gr1 §vehicle: the shipped `state/tokens.dr7t` member.
REF_TOKENS_BYTES = 557_253


class TW1Error(RuntimeError):
    """Raised when a control fails or an input is not the expected artifact."""


def rate_term(archive_bytes: int) -> float:
    """The evaluator's rate contribution for an archive of ``archive_bytes``."""
    return 25.0 * archive_bytes / UNCOMPRESSED


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_head() -> str:
    """Best-effort git HEAD for receipt provenance; never fatal."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------- field loading


def load_token_field(archive: str | Path) -> tuple[np.ndarray, int, str]:
    """Return ``(codes, shipped_member_bytes, member_sha256)`` from a pfs1 archive.

    Decoding uses ``verify="canonical"``, which re-encodes and refuses on any
    difference, so a successful return is itself proof that the bytes on disk are
    what the shipped codec produces.
    """
    path = Path(archive)
    if not path.exists():
        raise TW1Error(f"archive not found: {path}")
    with zipfile.ZipFile(path) as zf:
        if "state/tokens.dr7t" not in set(zf.namelist()):
            raise TW1Error(f"no state/tokens.dr7t member in {path}")
        frame = zf.read("state/tokens.dr7t")
    codes = r7.decode_token_codes(frame, verify=r7.VERIFY_CANONICAL)
    return codes, len(frame), _sha(frame)


def load_cell_records(path: str | Path = WR1_CELLS) -> list[dict]:
    """Return wr1's 768 per-cell records (``cell``, ``row``, ``col``, ``band``,
    ``flip_mass``, ``residual_mass``, ``drop_rank``)."""
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or len(records) != 768:
        raise TW1Error(f"expected 768 wr1 cell records, got {type(records)}")
    return records


# ------------------------------------------------------------------ the lattice


@dataclass(frozen=True)
class Lattice:
    """The pfs1 token field factored the way the shipped SMEVR coder factors it."""

    codes: np.ndarray
    base: np.ndarray
    delta: np.ndarray
    levels: int = LEVELS

    @property
    def n_rows(self) -> int:
        return int(self.codes.shape[1])

    @property
    def n_cols(self) -> int:
        return int(self.codes.shape[2])


def factor(codes: np.ndarray, *, levels: int = LEVELS) -> Lattice:
    """Factor ``codes`` into the coder's own ``(base, delta)`` representation."""
    base, delta = r7.factor_mode_delta(codes, levels)
    return Lattice(codes=codes, base=base, delta=delta, levels=levels)


def codes_with_dropped_cells(lat: Lattice, dropped: frozenset[int] | set[int]) -> np.ndarray:
    """Return the code lattice with every cell in ``dropped`` reset to its base.

    A *cell* is a spatial column ``(row, col)``: it spans all pairs and all
    channels, which is the unit wr1 and gr1 both measured as the efficient
    coding + coarsening unit.  "Drop" means the residual goes to zero, so the
    cell decodes to its own temporal mode.  The mode itself is unchanged by this
    operation (a column of identical values has that value as its mode), so the
    base stream is invariant and the measured delta is a pure residual effect.
    """
    if not dropped:
        return lat.codes.copy()
    delta = lat.delta.copy()
    rows = np.array([c // lat.n_cols for c in sorted(dropped)], dtype=np.intp)
    cols = np.array([c % lat.n_cols for c in sorted(dropped)], dtype=np.intp)
    delta[:, rows, cols, :] = 0
    return r7.reconstruct_mode_delta(lat.base, delta, lat.levels)


def token_bytes(codes: np.ndarray, *, codec: str = "smevr", levels: int = LEVELS) -> int:
    """MEASURED bytes of one encoded token frame through the real r7 coder."""
    return len(r7.encode_token_codes(codes, levels=levels, codec=codec))


def state_bytes(
    lat: Lattice,
    dropped: frozenset[int] | set[int],
    *,
    codec: str = "smevr",
) -> int:
    """MEASURED token-stream bytes for the lattice with ``dropped`` cells zeroed."""
    return token_bytes(codes_with_dropped_cells(lat, dropped), codec=codec, levels=lat.levels)


def wr1_state(records: list[dict], k: int) -> frozenset[int]:
    """The set of cells wr1 has dropped after ``k`` tranches of its descent."""
    return frozenset(r["cell"] for r in records if r["drop_rank"] < k)


# -------------------------------------------------------------------- controls


@dataclass
class ControlResult:
    name: str
    passed: bool
    expected: int | float
    observed: int | float
    detail: str = ""


def run_controls(
    lat: Lattice,
    records: list[dict],
    shipped_bytes: int,
    *,
    state_checks: tuple[int, ...] = (486, 600),
    wr1_receipt: str | Path = WR1_RECEIPT,
) -> list[ControlResult]:
    """Prove the meter before trusting a reading (design philosophy P4).

    * positive — re-encoding the untouched field reproduces the shipped member;
    * null — dropping the empty set moves exactly zero bytes;
    * state reconstruction — wr1's published ``tokens_bytes`` at each checked
      ``k`` is reproduced from ``drop_rank`` alone, proving this module's notion
      of "the state after k tranches" is wr1's.
    """
    out: list[ControlResult] = []

    reencoded = token_bytes(lat.codes, levels=lat.levels)
    out.append(
        ControlResult(
            "positive_reencode_matches_shipped_member",
            reencoded == shipped_bytes,
            shipped_bytes,
            reencoded,
            "round-trip of the shipped tokens.dr7t through the same coder",
        )
    )

    null = state_bytes(lat, frozenset())
    out.append(
        ControlResult(
            "null_empty_drop_is_zero_delta",
            null == reencoded,
            reencoded,
            null,
            "dropping no cells must not perturb the encode path",
        )
    )

    published = {
        int(row["k_cells_dropped"]): int(row["tokens_bytes"])
        for row in json.loads(Path(wr1_receipt).read_text())["descent_rows"]
    }
    for k in state_checks:
        if k not in published:
            # A control that cannot run is NOT a control that passed.  Emitting
            # `continue` here would silently shrink the control set and leave
            # `all(passed)` True over the two weak controls above -- the
            # vacuity-reads-as-pass failure class.  Fail closed instead.
            out.append(
                ControlResult(
                    f"state_reconstruction_k{k}_matches_wr1",
                    False,
                    k,
                    -1,
                    f"REFUSED: wr1 receipt has no k={k} row, so the strong control "
                    "could not run; an unrunnable control is a failure, not a skip",
                )
            )
            continue
        observed = state_bytes(lat, wr1_state(records, k))
        out.append(
            ControlResult(
                f"state_reconstruction_k{k}_matches_wr1",
                observed == published[k],
                published[k],
                observed,
                "wr1 drop_rank ordering reproduces the published tranche bytes",
            )
        )
    return out


# ------------------------------------------------- A. marginal price by state


@dataclass
class MarginalRow:
    cell: int
    row: int
    col: int
    band: str
    flip_mass: float
    residual_mass: float
    drop_rank: int
    # state label -> marginal bytes saved by dropping this cell from that state
    marginal_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def spread_bytes(self) -> int:
        """max - min marginal across the measured states."""
        values = list(self.marginal_bytes.values())
        return max(values) - min(values) if values else 0

    @property
    def spread_ratio(self) -> float:
        """max/min marginal across states; 1.0 means a state-independent price."""
        values = list(self.marginal_bytes.values())
        lo, hi = min(values), max(values)
        if lo <= 0:
            return float("inf") if hi > 0 else 1.0
        return hi / lo


def stratified_sample(
    records: list[dict],
    *,
    exclude: frozenset[int],
    per_band: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Pick up to ``per_band`` cells from each band, deterministically.

    Cells in ``exclude`` are skipped so the SAME cells remain pricable from every
    state under test — a cell already dropped in a later state has no marginal
    there, and comparing different cells across states would confound the
    state effect with a cell effect.

    Returns ``(chosen, coverage)``.  ``coverage`` reports, per band, how many
    cells were AVAILABLE after exclusion and how many were TAKEN, for every band
    present in ``records`` — including bands that contributed zero.  Reporting
    the denominator is the point: a band silently absent from the sample (which
    is how a Knee-B-excluded sweep quietly becomes road+hood-only) must be
    visible in the receipt rather than inferred from the row count.
    """
    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    coverage: dict[str, dict[str, int]] = {}
    pool = [r for r in records if r["cell"] not in exclude]
    for band in sorted({r["band"] for r in records}):
        band_cells = sorted((r for r in pool if r["band"] == band), key=lambda r: r["cell"])
        take = min(per_band, len(band_cells))
        coverage[band] = {
            "in_records": sum(1 for r in records if r["band"] == band),
            "available_after_exclusion": len(band_cells),
            "taken": take,
        }
        if take:
            idx = rng.choice(len(band_cells), size=take, replace=False)
            chosen.extend(band_cells[int(i)] for i in sorted(idx))
    return chosen, coverage


def marginal_price_by_state(
    lat: Lattice,
    records: list[dict],
    states: dict[str, frozenset[int]],
    sample: list[dict],
    *,
    codec: str = "smevr",
    progress: bool = True,
) -> tuple[list[MarginalRow], dict[str, int]]:
    """Price each sampled cell's marginal drop from every state in ``states``.

    Returns the per-cell rows and the measured baseline bytes of each state.
    """
    baselines = {name: state_bytes(lat, cells, codec=codec) for name, cells in states.items()}
    rows = [
        MarginalRow(
            cell=r["cell"],
            row=r["row"],
            col=r["col"],
            band=r["band"],
            flip_mass=float(r["flip_mass"]),
            residual_mass=float(r["residual_mass"]),
            drop_rank=int(r["drop_rank"]),
        )
        for r in sample
    ]
    total = len(rows) * len(states)
    done = 0
    for name, cells in states.items():
        for mrow in rows:
            if mrow.cell in cells:
                raise TW1Error(
                    f"cell {mrow.cell} is already dropped in state {name}; "
                    "the sample must be held out of every state under test"
                )
            after = state_bytes(lat, cells | {mrow.cell}, codec=codec)
            mrow.marginal_bytes[name] = baselines[name] - after
            done += 1
            if progress and done % 10 == 0:
                print(f"  marginal {done}/{total}", flush=True)
    return rows, baselines


# ------------------------------------------------------- B. additivity defect


@dataclass
class AdditivityRow:
    state: str
    n_cells: int
    sum_of_singleton_marginals: int
    joint_measured_saving: int

    @property
    def defect_bytes(self) -> int:
        """Positive means the singleton sum OVERSTATES what the set really saves."""
        return self.sum_of_singleton_marginals - self.joint_measured_saving

    @property
    def defect_fraction(self) -> float:
        if self.joint_measured_saving == 0:
            return float("nan")
        return self.defect_bytes / self.joint_measured_saving


def additivity_defect(
    lat: Lattice,
    state: frozenset[int],
    cells: list[int],
    *,
    state_name: str,
    singleton_marginals: dict[int, int] | None = None,
    codec: str = "smevr",
) -> AdditivityRow:
    """Compare the sum of singleton marginals with the joint measured saving.

    ``singleton_marginals`` may be supplied to reuse measurements already taken
    (measurement A prices exactly these singletons); when absent they are
    measured here.
    """
    base = state_bytes(lat, state, codec=codec)
    if singleton_marginals is None:
        singleton_marginals = {
            c: base - state_bytes(lat, state | {c}, codec=codec) for c in cells
        }
    missing = [c for c in cells if c not in singleton_marginals]
    if missing:
        raise TW1Error(f"missing singleton marginals for cells {missing}")
    joint = base - state_bytes(lat, state | set(cells), codec=codec)
    return AdditivityRow(
        state=state_name,
        n_cells=len(cells),
        sum_of_singleton_marginals=sum(singleton_marginals[c] for c in cells),
        joint_measured_saving=joint,
    )


# ------------------------------------------------------ C. coder race by state


@dataclass
class CoderRow:
    state: str
    codec: str
    token_bytes: int
    archive_bytes: int
    rate_term: float
    encode_seconds: float

    @property
    def vs_smevr_bytes(self) -> int | None:
        return None  # filled by the caller against the smevr row of the same state


def coder_race_by_state(
    lat: Lattice,
    states: dict[str, frozenset[int]],
    codecs: tuple[str, ...],
    *,
    seconds_budget: float = 240.0,
    progress: bool = True,
) -> list[CoderRow]:
    """Re-run the lossless coder race at each drop state.

    The r7 race was decided on the UNDROPPED field.  Dropping cells removes a
    large fraction of the residual mass, which changes the source statistics the
    race was decided under, so the winner is re-measured rather than assumed.

    ``seconds_budget`` skips a codec for the remaining states once one encode has
    exceeded it, so a pathologically slow coder cannot stall the sweep.
    """
    rows: list[CoderRow] = []
    too_slow: set[str] = set()
    for name, cells in states.items():
        codes = codes_with_dropped_cells(lat, cells)
        for codec in codecs:
            if codec in too_slow:
                continue
            start = time.monotonic()
            try:
                nbytes = token_bytes(codes, codec=codec, levels=lat.levels)
            except Exception as exc:
                if progress:
                    print(f"  {name}/{codec}: REFUSED ({type(exc).__name__}: {exc})", flush=True)
                continue
            elapsed = time.monotonic() - start
            if elapsed > seconds_budget:
                too_slow.add(codec)
            archive = nbytes + ARCHIVE_FLOOR_BYTES
            rows.append(
                CoderRow(
                    state=name,
                    codec=codec,
                    token_bytes=nbytes,
                    archive_bytes=archive,
                    rate_term=rate_term(archive),
                    encode_seconds=round(elapsed, 2),
                )
            )
            if progress:
                print(f"  {name}/{codec}: {nbytes:,} B ({elapsed:.1f}s)", flush=True)
    return rows


# -------------------------------------------------------------------- runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", default=PFS1_ARCHIVE)
    parser.add_argument("--cells", default=WR1_CELLS)
    parser.add_argument("--receipt", default=WR1_RECEIPT)
    parser.add_argument("--outdir", default="/Volumes/VertigoDataTier/pact/ddm_tw1_20260801")
    parser.add_argument(
        "--per-band",
        type=int,
        default=8,
        help="cells sampled per band for the marginal-price measurement",
    )
    parser.add_argument("--seed", type=int, default=869)
    parser.add_argument(
        "--states",
        default="k0:0,kneeA:486,kneeB:600",
        help="comma-separated label:k pairs naming the drop states to price from",
    )
    parser.add_argument(
        "--codecs",
        default="smevr,brotli11,lzma1,rans_o0,kt_prev1,kt_o8_prev5_backoff,huffman_nibble",
        help="r7 codecs to race at each state (measurement C)",
    )
    parser.add_argument("--skip-race", action="store_true")
    parser.add_argument("--skip-marginal", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("loading token field ...", flush=True)
    codes, shipped_bytes, member_sha = load_token_field(args.archive)
    lat = factor(codes)
    records = load_cell_records(args.cells)
    print(
        f"  lattice {codes.shape} levels={LEVELS} shipped member {shipped_bytes:,} B "
        f"sha {member_sha[:16]}",
        flush=True,
    )

    print("controls ...", flush=True)
    controls = run_controls(lat, records, shipped_bytes, wr1_receipt=args.receipt)
    for c in controls:
        print(
            f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: "
            f"expected {c.expected:,} observed {c.observed:,}",
            flush=True,
        )
    if not all(c.passed for c in controls):
        raise TW1Error("control failed; refusing to emit readings from an unvalidated meter")

    state_spec: dict[str, frozenset[int]] = {}
    for chunk in args.states.split(","):
        label, _, k = chunk.partition(":")
        state_spec[label.strip()] = wr1_state(records, int(k))

    payload: dict = {
        "schema": SCHEMA,
        "task": "#869 / ledger QA06 follow-on",
        "evidence_axis": "[macOS-CPU advisory, rate-only] bytes MEASURED through the real r7 coder",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_archive": str(args.archive),
        "source_tokens_sha256": member_sha,
        "shipped_tokens_bytes": shipped_bytes,
        "archive_floor_bytes": ARCHIVE_FLOOR_BYTES,
        "lattice_shape": list(codes.shape),
        "levels": LEVELS,
        "controls": [asdict(c) for c in controls],
        "states": {k: len(v) for k, v in state_spec.items()},
        # provenance: a receipt that cannot regenerate its own rows is not a receipt
        "invocation": {
            "argv": list(sys.argv[1:]),
            "seed": args.seed,
            "per_band": args.per_band,
            "states_spec": args.states,
            "codecs": args.codecs,
            "git_head": _git_head(),
        },
    }

    if not args.skip_marginal:
        # The union, not the largest state: `max(..., key=len)` is the union only
        # while the states happen to be nested (they are, under `drop_rank < k`).
        # Taking the union makes the hold-out correct for any state spec instead
        # of relying on an incidental property of this one.
        excluded = frozenset().union(*state_spec.values()) if state_spec else frozenset()
        sample, coverage = stratified_sample(
            records, exclude=excluded, per_band=args.per_band, seed=args.seed
        )
        payload["sample_coverage"] = coverage
        for band, cov in coverage.items():
            print(
                f"  band {band}: {cov['taken']} taken of "
                f"{cov['available_after_exclusion']} available "
                f"({cov['in_records']} in records)",
                flush=True,
            )
        print(f"marginal price: {len(sample)} held-out cells x {len(state_spec)} states", flush=True)
        rows, baselines = marginal_price_by_state(lat, records, state_spec, sample)
        payload["state_baseline_token_bytes"] = baselines
        payload["marginal_rows"] = [
            {**asdict(r), "spread_bytes": r.spread_bytes, "spread_ratio": r.spread_ratio}
            for r in rows
        ]

        singles = {name: {r.cell: r.marginal_bytes[name] for r in rows} for name in state_spec}
        add_rows = [
            additivity_defect(
                lat,
                state_spec[name],
                [r.cell for r in rows],
                state_name=name,
                singleton_marginals=singles[name],
            )
            for name in state_spec
        ]
        payload["additivity"] = [
            {**asdict(a), "defect_bytes": a.defect_bytes, "defect_fraction": a.defect_fraction}
            for a in add_rows
        ]
        for a in add_rows:
            print(
                f"  additivity {a.state}: sum {a.sum_of_singleton_marginals:,} vs joint "
                f"{a.joint_measured_saving:,} -> defect {a.defect_bytes:,} "
                f"({a.defect_fraction:+.1%})",
                flush=True,
            )

    if not args.skip_race:
        print("coder race by state ...", flush=True)
        race = coder_race_by_state(lat, state_spec, tuple(args.codecs.split(",")))
        payload["coder_race"] = [asdict(r) for r in race]

    out = outdir / "tw1_state_dependence_receipt.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
