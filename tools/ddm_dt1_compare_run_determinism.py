# SPDX-License-Identifier: MIT
"""ddm_dt1 (#903) — run-to-run determinism comparator for TR1 trainer windows.

THE QUESTION: is ``experiments/train_tr1_partition_renderer_mlx.py`` bit-reproducible
run-to-run at fixed (seed, config, inputs)? If it is NOT, every small-delta lever
attribution in this campaign is ungrounded, because a claimed lever effect cannot be
distinguished from run-to-run noise. CLAUDE.md makes deterministic reproducibility one
of the two HARD limits, so this is a P0 apparatus question, not a curiosity.

WHAT IT COMPARES (two independent surfaces, both reported):

  1. CHECKPOINT ARRAYS — every ``.npz`` key under a run's ``checkpoints/`` dir.
     The TR1 checkpoint payload is ``param::*`` / ``ema::*`` / ``opt::*`` /
     ``meta::epoch`` / ``meta::json`` (see ``save_checkpoint``). Comparison is
     BIT-LEVEL on the raw buffer (``.tobytes()``), so a 1-ULP drift is a DIFFER,
     not a rounding-tolerant PASS. Magnitudes (max-abs, rms, max-rel, ULP) are
     reported alongside so the reader can size the drift, never to excuse it.

  2. TELEMETRY SCALARS — numeric fields of ``telemetry.jsonl`` rows, keyed by
     ``(event, epoch, field)``. This is where the LEVER-ATTRIBUTION UNITS live:
     ``realized_gate_dseg_mean`` is the quantity every d_seg lever claim is made in
     (S-units = 100 x d_seg). A spread here IS the noise floor of our A/Bs.

THE VACUITY GUARD (CLAUDE.md: "empty scope emits the same symbol as a clean full
scope"). Every verdict carries its DENOMINATOR. A comparison of 0 arrays reports
``VACUOUS``, never ``IDENTICAL``. Keys present in one run and absent in the other are
reported as ``asymmetric`` and counted separately — they are neither same nor differ.

THE POSITIVE CONTROL (CLAUDE.md L3 verdict-clearance: "a comparison that has never
been shown to fire is untrusted"). ``--self-check`` synthesizes three fixtures and
asserts the comparator's verdict on each:
    identical pair            -> IDENTICAL   (must not false-fire)
    1-ULP-perturbed pair      -> DIFFER      (must detect the smallest possible drift)
    key-asymmetric pair       -> ASYMMETRIC  (must not silently intersect-away)
    empty pair                -> VACUOUS     (must not report a bare PASS on nothing)
``--self-check`` runs automatically before any real comparison unless
``--skip-self-check`` is passed, and a self-check failure ABORTS (rc=3) rather than
degrading to an untrusted comparison.

Evidence axis: this tool makes NO score claim. It reports apparatus reproducibility.

Usage
-----
    # positive control only
    .venv/bin/python tools/ddm_dt1_compare_run_determinism.py --self-check

    # compare N run dirs (N >= 2)
    .venv/bin/python tools/ddm_dt1_compare_run_determinism.py \\
        --run-dir RUN_A --run-dir RUN_B --run-dir RUN_C --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Verdict vocabulary. VACUOUS is deliberately distinct from IDENTICAL: an empty
# comparison scope must never emit the same symbol as a clean full one.
IDENTICAL = "IDENTICAL"
DIFFER = "DIFFER"
ASYMMETRIC = "ASYMMETRIC"
VACUOUS = "VACUOUS"


@dataclass
class ArrayDelta:
    """Per-key comparison result. ``bit_identical`` is the authority; the rest sizes it."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    bit_identical: bool
    max_abs: float = 0.0
    rms: float = 0.0
    max_rel: float = 0.0
    n_elem_differ: int = 0
    n_elem: int = 0
    max_ulp: int = 0


@dataclass
class PairReport:
    """One A-vs-B comparison over one file (or one telemetry surface)."""

    label: str
    verdict: str
    n_compared: int = 0
    n_identical: int = 0
    n_differ: int = 0
    n_only_a: int = 0
    n_only_b: int = 0
    deltas: list[ArrayDelta] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)

    def denominator_line(self) -> str:
        return (f"{self.n_differ} of {self.n_compared} compared DIFFER "
                f"({self.n_identical} identical; "
                f"{self.n_only_a} only-in-A, {self.n_only_b} only-in-B)")


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Max ULP distance for float arrays; 0 for non-float. Sign-magnitude ordered."""
    if a.dtype.kind != "f":
        return 0
    try:
        width = a.dtype.itemsize
        int_t = {2: np.int16, 4: np.int32, 8: np.int64}.get(width)
        if int_t is None:
            return 0
        ia = a.view(int_t).astype(np.int64)
        ib = b.view(int_t).astype(np.int64)
        # map sign-magnitude -> monotone two's complement ordering
        sign_bit = np.int64(1) << np.int64(width * 8 - 1)
        ia = np.where(ia < 0, sign_bit - ia, ia)
        ib = np.where(ib < 0, sign_bit - ib, ib)
        d = np.abs(ia - ib)
        return int(d.max()) if d.size else 0
    except (ValueError, TypeError):
        return 0


def compare_arrays(key: str, a: np.ndarray, b: np.ndarray) -> ArrayDelta:
    """Bit-level compare of one array pair; magnitudes reported, never used to excuse."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return ArrayDelta(key=key, shape=tuple(a.shape), dtype=str(a.dtype),
                          bit_identical=False, n_elem=int(a.size), n_elem_differ=int(a.size))
    bit_identical = a.tobytes() == b.tobytes()
    d = ArrayDelta(key=key, shape=tuple(a.shape), dtype=str(a.dtype),
                   bit_identical=bit_identical, n_elem=int(a.size))
    if bit_identical or a.size == 0:
        return d
    if a.dtype.kind in "fiu":
        af = a.astype(np.float64, copy=False)
        bf = b.astype(np.float64, copy=False)
        diff = af - bf
        d.max_abs = float(np.abs(diff).max())
        d.rms = float(np.sqrt(np.mean(diff ** 2)))
        denom = np.maximum(np.abs(af), np.abs(bf))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, np.abs(diff) / denom, 0.0)
        d.max_rel = float(np.nanmax(rel)) if rel.size else 0.0
        d.n_elem_differ = int(np.count_nonzero(diff))
        d.max_ulp = _ulp_distance(a, b)
    else:
        d.n_elem_differ = int(np.count_nonzero(a != b))
    return d


def _verdict(n_compared: int, n_differ: int, n_only_a: int, n_only_b: int) -> str:
    if n_compared == 0:
        # Empty scope is NOT a pass. Report it as its own symbol so a reader
        # cannot mistake "nothing was compared" for "everything matched".
        return VACUOUS
    if n_only_a or n_only_b:
        return ASYMMETRIC if n_differ == 0 else DIFFER
    return DIFFER if n_differ else IDENTICAL


def compare_npz(label: str, path_a: Path, path_b: Path,
                exclude_keys: frozenset[str] = frozenset()) -> PairReport:
    za = np.load(path_a, allow_pickle=False)
    zb = np.load(path_b, allow_pickle=False)
    ka = {k for k in za.files if k not in exclude_keys}
    kb = {k for k in zb.files if k not in exclude_keys}
    shared = sorted(ka & kb)
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    deltas = [compare_arrays(k, za[k], zb[k]) for k in shared]
    n_differ = sum(1 for d in deltas if not d.bit_identical)
    return PairReport(
        label=label,
        verdict=_verdict(len(shared), n_differ, len(only_a), len(only_b)),
        n_compared=len(shared),
        n_identical=len(shared) - n_differ,
        n_differ=n_differ,
        n_only_a=len(only_a), n_only_b=len(only_b),
        deltas=deltas, only_a=only_a, only_b=only_b,
    )


# --------------------------------------------------------------------------
# Telemetry surface — the lever-attribution units
# --------------------------------------------------------------------------
# Volatile fields are wall-clock / pid / path-shaped: they legitimately differ
# between two processes and say NOTHING about numerical determinism. They are
# EXCLUDED from the verdict but the exclusion list is printed, so the reader sees
# exactly what was carved out (an unreported exclusion is a vacuity hole).
VOLATILE_TELEMETRY_FIELDS = frozenset({
    "wall_seconds", "gate_wall_seconds", "epoch_seconds", "elapsed_s", "elapsed_seconds",
    "t", "t_wall", "timestamp", "ts", "utc", "written_at_utc", "pid", "host", "out_dir",
    "epoch_total_s", "rss_gib", "peak_rss_gib", "seconds", "wall_minutes",
})
# ``t_wall`` is TR1's per-row monotonic clock (train_tr1_partition_renderer_mlx.py tlog).
# MEASURED 2026-08-03: it was the ONLY residual telemetry difference across three
# bit-identical MLX-CPU runs (8 of 142 fields, all t_wall) -- i.e. excluding it is what
# makes the telemetry surface agree with the checkpoint surface on a deterministic run.


def load_telemetry_scalars(path: Path,
                           volatile: frozenset[str] = VOLATILE_TELEMETRY_FIELDS,
                           ) -> dict[str, float]:
    """Flatten telemetry.jsonl into {(event|epoch|row_index).field: value} scalars."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for idx, ln in enumerate(path.read_text().splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ev = str(row.get("event", row.get("stage", "row")))
        ep = row.get("epoch", None)
        prefix = f"{ev}|ep{ep}|i{idx}" if ep is None else f"{ev}|ep{ep}"

        def walk(obj: Any, path_parts: tuple[str, ...]) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in volatile:
                        continue
                    walk(v, path_parts + (str(k),))
            elif isinstance(obj, list):
                for j, v in enumerate(obj):
                    walk(v, path_parts + (str(j),))
            elif isinstance(obj, bool):
                return  # bools carry no float drift
            elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
                out[f"{prefix}.{'.'.join(path_parts)}"] = float(obj)

        walk(row, ())
    return out


def compare_telemetry(label: str, a: dict[str, float], b: dict[str, float]) -> PairReport:
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    deltas: list[ArrayDelta] = []
    for k in shared:
        va, vb = a[k], b[k]
        same = np.float64(va).tobytes() == np.float64(vb).tobytes()
        d = ArrayDelta(key=k, shape=(), dtype="float64", bit_identical=same, n_elem=1)
        if not same:
            d.max_abs = abs(va - vb)
            d.rms = d.max_abs
            denom = max(abs(va), abs(vb))
            d.max_rel = (d.max_abs / denom) if denom > 0 else 0.0
            d.n_elem_differ = 1
            d.max_ulp = _ulp_distance(np.array([va]), np.array([vb]))
        deltas.append(d)
    n_differ = sum(1 for d in deltas if not d.bit_identical)
    return PairReport(
        label=label,
        verdict=_verdict(len(shared), n_differ, len(only_a), len(only_b)),
        n_compared=len(shared),
        n_identical=len(shared) - n_differ,
        n_differ=n_differ,
        n_only_a=len(only_a), n_only_b=len(only_b),
        deltas=deltas, only_a=only_a, only_b=only_b,
    )


# --------------------------------------------------------------------------
# Positive control
# --------------------------------------------------------------------------
def self_check() -> tuple[bool, list[str]]:
    """Prove the comparator fires on a KNOWN difference and is quiet on a KNOWN identity.

    Returns (ok, lines). A failure here means every downstream verdict is untrusted.
    """
    lines: list[str] = []
    ok = True
    rng = np.random.default_rng(20260803)
    base = {
        "param::a": rng.standard_normal((8, 5)).astype(np.float32),
        "param::b": rng.standard_normal((3,)).astype(np.float32),
        "opt::c": rng.integers(0, 255, size=(4,)).astype(np.uint8),
        "meta::epoch": np.array([7], dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pa, pb = tdp / "a.npz", tdp / "b.npz"
        np.savez(pa, **base)

        # (1) IDENTICAL — byte-equal payload must not false-fire.
        np.savez(pb, **base)
        r = compare_npz("selfcheck_identical", pa, pb)
        good = r.verdict == IDENTICAL and r.n_compared == len(base) and r.n_differ == 0
        ok &= good
        lines.append(f"  [{'PASS' if good else 'FAIL'}] identical  -> {r.verdict} "
                     f"({r.denominator_line()})")

        # (2) DIFFER at 1 ULP — the SMALLEST possible float32 drift must be caught.
        pert = dict(base)
        a2 = base["param::a"].copy()
        flat = a2.reshape(-1)
        flat[0] = np.nextafter(flat[0], np.float32(np.inf))
        pert["param::a"] = a2
        np.savez(pb, **pert)
        r2 = compare_npz("selfcheck_1ulp", pa, pb)
        hit = next((d for d in r2.deltas if d.key == "param::a"), None)
        good = (r2.verdict == DIFFER and r2.n_differ == 1
                and hit is not None and hit.max_ulp == 1)
        ok &= good
        lines.append(f"  [{'PASS' if good else 'FAIL'}] 1-ULP      -> {r2.verdict} "
                     f"({r2.denominator_line()}; max_ulp="
                     f"{hit.max_ulp if hit else 'n/a'})")

        # (3) ASYMMETRIC — a key present in only one side must not be intersected away.
        asym = {k: v for k, v in base.items() if k != "param::b"}
        np.savez(pb, **asym)
        r3 = compare_npz("selfcheck_asymmetric", pa, pb)
        good = r3.verdict == ASYMMETRIC and r3.n_only_a == 1 and r3.n_compared == len(base) - 1
        ok &= good
        lines.append(f"  [{'PASS' if good else 'FAIL'}] asymmetric -> {r3.verdict} "
                     f"({r3.denominator_line()})")

        # (4) VACUOUS — an EMPTY scope must NOT report a bare pass.
        empty = tdp / "empty_a.npz"
        empty_b = tdp / "empty_b.npz"
        np.savez(empty, **{"meta::epoch": np.array([0], dtype=np.int64)})
        np.savez(empty_b, **{"meta::epoch": np.array([0], dtype=np.int64)})
        r4 = compare_npz("selfcheck_vacuous", empty, empty_b,
                         exclude_keys=frozenset({"meta::epoch"}))
        good = r4.verdict == VACUOUS and r4.n_compared == 0
        ok &= good
        lines.append(f"  [{'PASS' if good else 'FAIL'}] empty scope-> {r4.verdict} "
                     f"({r4.denominator_line()})")

        # (5) telemetry surface: identical + smallest-drift, same two obligations.
        ta = {"gate|ep4.realized_gate_dseg_mean": 0.0052780002355575562}
        tb_same = dict(ta)
        tb_diff = {"gate|ep4.realized_gate_dseg_mean":
                   float(np.nextafter(np.float64(ta["gate|ep4.realized_gate_dseg_mean"]),
                                      np.float64(np.inf)))}
        r5 = compare_telemetry("selfcheck_tel_identical", ta, tb_same)
        r6 = compare_telemetry("selfcheck_tel_1ulp", ta, tb_diff)
        good = r5.verdict == IDENTICAL and r6.verdict == DIFFER and r6.deltas[0].max_ulp == 1
        ok &= good
        lines.append(f"  [{'PASS' if good else 'FAIL'}] telemetry  -> "
                     f"identical={r5.verdict}, 1ulp={r6.verdict} "
                     f"(ulp={r6.deltas[0].max_ulp})")

    return ok, lines


# --------------------------------------------------------------------------
# Run-dir comparison
# --------------------------------------------------------------------------
def compare_run_dirs(run_dirs: list[Path], *, exclude_meta: bool,
                     top_n: int = 12) -> dict[str, Any]:
    """Pairwise-compare every run dir against the FIRST (the reference run)."""
    exclude = frozenset({"meta::json"}) if exclude_meta else frozenset()
    ref = run_dirs[0]
    ref_ckpts = sorted((ref / "checkpoints").glob("*.npz")) if (ref / "checkpoints").is_dir() else []
    out: dict[str, Any] = {
        "reference_run": str(ref),
        "n_runs": len(run_dirs),
        "excluded_npz_keys": sorted(exclude),
        "excluded_telemetry_fields": sorted(VOLATILE_TELEMETRY_FIELDS),
        "reference_checkpoint_files": [p.name for p in ref_ckpts],
        "pairs": [],
    }
    ref_tel = load_telemetry_scalars(ref / "telemetry.jsonl")

    for other in run_dirs[1:]:
        pair: dict[str, Any] = {"run": str(other), "checkpoints": [], "telemetry": None}
        other_ck_dir = other / "checkpoints"
        for rp in ref_ckpts:
            op = other_ck_dir / rp.name
            if not op.exists():
                pair["checkpoints"].append({
                    "file": rp.name, "verdict": "MISSING_IN_B",
                    "note": "checkpoint present in reference run, absent here",
                })
                continue
            rep = compare_npz(rp.name, rp, op, exclude_keys=exclude)
            worst = sorted([d for d in rep.deltas if not d.bit_identical],
                           key=lambda d: -d.max_abs)[:top_n]
            pair["checkpoints"].append({
                "file": rp.name,
                "verdict": rep.verdict,
                "n_compared": rep.n_compared,
                "n_identical": rep.n_identical,
                "n_differ": rep.n_differ,
                "n_only_a": rep.n_only_a,
                "n_only_b": rep.n_only_b,
                "denominator": rep.denominator_line(),
                "worst_deltas": [
                    {"key": d.key, "shape": list(d.shape), "dtype": d.dtype,
                     "max_abs": d.max_abs, "rms": d.rms, "max_rel": d.max_rel,
                     "max_ulp": d.max_ulp,
                     "frac_elem_differ": (d.n_elem_differ / d.n_elem) if d.n_elem else 0.0}
                    for d in worst
                ],
            })
        other_tel = load_telemetry_scalars(other / "telemetry.jsonl")
        trep = compare_telemetry("telemetry.jsonl", ref_tel, other_tel)
        tworst = sorted([d for d in trep.deltas if not d.bit_identical],
                        key=lambda d: -d.max_abs)[:top_n]
        pair["telemetry"] = {
            "verdict": trep.verdict,
            "n_compared": trep.n_compared,
            "n_identical": trep.n_identical,
            "n_differ": trep.n_differ,
            "n_only_a": trep.n_only_a,
            "n_only_b": trep.n_only_b,
            "denominator": trep.denominator_line(),
            "worst_deltas": [
                {"key": d.key, "abs": d.max_abs, "rel": d.max_rel, "ulp": d.max_ulp}
                for d in tworst
            ],
            "dseg_rows": [
                {"key": d.key, "abs": d.max_abs, "rel": d.max_rel,
                 "bit_identical": d.bit_identical}
                for d in trep.deltas if "dseg" in d.key or "d_seg" in d.key
            ],
        }
        out["pairs"].append(pair)
    return out


def summarize(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"reference run: {report['reference_run']}")
    lines.append(f"runs compared: {report['n_runs']} "
                 f"(pairwise vs reference => {report['n_runs'] - 1} pairs)")
    if not report["reference_checkpoint_files"]:
        lines.append("  !! reference run has NO checkpoints -- checkpoint scope is VACUOUS")
    for pair in report["pairs"]:
        lines.append(f"\n-- {pair['run']}")
        for ck in pair["checkpoints"]:
            lines.append(f"   ckpt {ck['file']}: {ck['verdict']}  "
                         f"{ck.get('denominator', ck.get('note', ''))}")
            for w in ck.get("worst_deltas", [])[:5]:
                lines.append(f"       {w['key']:<34} max_abs={w['max_abs']:.6e} "
                             f"rms={w['rms']:.6e} max_rel={w['max_rel']:.3e} "
                             f"ulp={w['max_ulp']} fracdiff={w['frac_elem_differ']:.4f}")
        t = pair["telemetry"]
        lines.append(f"   telemetry: {t['verdict']}  {t['denominator']}")
        for d in t["dseg_rows"]:
            flag = "==" if d["bit_identical"] else "!="
            lines.append(f"       {flag} {d['key']:<44} abs={d['abs']:.6e} rel={d['rel']:.3e}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", default=[],
                    help="run dir (repeat >=2 times); the FIRST is the reference")
    ap.add_argument("--self-check", action="store_true",
                    help="run the positive control and exit")
    ap.add_argument("--skip-self-check", action="store_true",
                    help="DANGEROUS: skip the positive control before a real comparison")
    ap.add_argument("--keep-meta-json", action="store_true",
                    help="include meta::json in the comparison (it embeds out_dir/telemetry "
                         "tail, so it differs for non-numerical reasons; excluded by default "
                         "and the exclusion is reported)")
    ap.add_argument("--json", type=Path, default=None, help="write the full report JSON here")
    args = ap.parse_args()

    if not args.skip_self_check or args.self_check:
        ok, lines = self_check()
        print("POSITIVE CONTROL (comparator must fire on known-diff, stay quiet on known-same):")
        for ln in lines:
            print(ln)
        print(f"  => positive control {'PASSED' if ok else 'FAILED'}")
        if not ok:
            print("ABORT: comparator is untrusted; no verdict is admissible.", file=sys.stderr)
            return 3
    if args.self_check:
        return 0

    if len(args.run_dir) < 2:
        print("need >=2 --run-dir (the first is the reference)", file=sys.stderr)
        return 2
    for d in args.run_dir:
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2

    report = compare_run_dirs(args.run_dir, exclude_meta=not args.keep_meta_json)
    print("\n" + "\n".join(summarize(report)))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")

    any_differ = any(
        ck["verdict"] in (DIFFER,) for pair in report["pairs"] for ck in pair["checkpoints"]
    ) or any(pair["telemetry"]["verdict"] == DIFFER for pair in report["pairs"])
    print(f"\nVERDICT: {'NON-DETERMINISTIC (at least one pair DIFFERs)' if any_differ else 'BIT-IDENTICAL across all compared pairs'}")
    return 1 if any_differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
