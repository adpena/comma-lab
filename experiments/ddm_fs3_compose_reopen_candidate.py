#!/usr/bin/env python3
"""ddm_fs3 leg 3 -- merge the re-priced re-screen, control it, compose the candidate.

WHAT THIS DOES
--------------
``ddm_fs3_jg3_repriced_rescreen.py`` re-ran ``ddm_jg3``'s per-pair configuration
sweep for the 38 reopened pairs with the token price disarmed down to the MEASURED
2.657293 bits/token, so the winner is the reopened configuration and its
``accepted`` coordinate list is finally emitted.  This module merges those shards
and composes the candidate edit field: jg5's shipped 455-pair set with those 38
pairs' entries REPLACED.

THE TWO CONTROLS, AND WHY THEY ARE THE POINT
--------------------------------------------
The re-screen changed exactly one thing -- the price in the configuration sweep's
``argmin``.  The per-site inner gate that builds the candidate SITE POOL was left
alone.  So two things must hold, and if they do not, the re-screen measured a
different object than the census predicted and no candidate may be built from it:

**CONTROL A -- sweep reproduction.**  Every ``(separation, keep_fraction)`` entry
the re-screen emits must carry the same ``(tokens, repaired)`` as jg3's retained
entry for that pair.  This proves the solver is the same solver and only the
argmin moved.

**CONTROL B -- the winner is the predicted configuration.**  For each pair, the
configuration the re-screen chose must be the one ``FS3_RESCREEN_PREREG.json``
registered before the run finished.  A pair that lands elsewhere is reported, not
silently absorbed: the census predicted a specific object and either it arrived or
it did not.

Both controls are reported PER PAIR and in aggregate.  A pair that fails either
one is EXCLUDED from the composed candidate rather than shipped on the hope that
it is close enough, and the exclusion is counted in the realized arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

N_PAIRS = 600
GRID_H, GRID_W = 384, 512
SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR
S_PER_SEG_CELL = 100.0 / (N_PAIRS * GRID_H * GRID_W)
ADMISSION_BAR_S = 3.5e-6

#: Why the emitted carrier key says DERIVED_extrapolated and not MEASURED.
#: Withdrawn by rv17 wave-3 W3-F7; superseded by a terminal measurement.  Emitted
#: beside the value so a future receipt carries the caveat without the reader
#: having to find this memo -- W3-F15 was exactly this caveat failing to travel.
CARRIER_LABEL_SUPERSEDED = (
    "carrier_MEASURED_leg2 -- WITHDRAWN as overclaimed (rv17 wave-3 W3-F7). The "
    "value is a LINEAR EXTRAPOLATION: 45 B measured over 454 pairs in ONE build, "
    "re-multiplied by --carrier-bytes-per-pair. The ladder measured that price as "
    "NON-MONOTONE in density with a +-45 B container-search spread, so the leg's "
    "uncertainty band (+-3.00e-05 S) is LARGER than its own point estimate. "
    "SUPERSEDED by measurement: .omx/research/"
    "ddm_fs3_jg5_real_price_reopen_20260820.md:877 -- the real build (180,625 -> "
    "179,961 = -664 B) puts the +45 B splice on BOTH sides, so leaving the carrier "
    "unchanged makes the carrier BYTE leg EXACTLY ZERO. Caveat from the same memo: "
    "zero bytes buys a STALE carrier on the changed pairs, so the COST does not "
    "vanish even though the BYTES do."
)

DEFAULT_JG5_SUBSET = (
    "/Volumes/APDataStore/pact/ddm_jg5/work/waterfill_final/seg_edits_subset.npz"
)
DEFAULT_PREREG = "/Volumes/APDataStore/pact/ddm_fs3/FS3_RESCREEN_PREREG.json"
JG3_SHARDS = [
    f"/Volumes/APDataStore/pact/ddm_jg3/retained/seg_solve_n600_wc2s{i}.json"
    for i in range(6)
]


class Fs3ComposeError(RuntimeError):
    """Fail-closed error."""


def sha256_of(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_retained_sweeps() -> dict[int, dict[tuple[int, float], tuple[int, int]]]:
    """jg3's retained sweep, keyed pair -> (separation, keep) -> (tokens, repaired)."""
    out: dict[int, dict[tuple[int, float], tuple[int, int]]] = {}
    for path in JG3_SHARDS:
        payload = json.loads(Path(path).read_text())
        for row in payload["per_pair"]:
            out[int(row["pair"])] = {
                (int(e["accept_separation"]), float(e["keep_fraction"])): (
                    int(e["tokens"]),
                    int(e["repaired"]),
                )
                for e in row["separation_sweep"]
            }
    return out


def run_merge(args: argparse.Namespace) -> int:
    prereg = json.loads(Path(args.prereg).read_text())
    predicted = prereg["per_pair"]
    retained = load_retained_sweeps()

    solves = sorted(Path(args.store, "retained").glob(f"seg_solve_{args.tag}_s*.json"))
    if not solves:
        raise Fs3ComposeError(f"no re-screen solve receipts under {args.store}")

    rows: dict[int, dict[str, Any]] = {}
    receipts = []
    for path in solves:
        digest, size = sha256_of(path)
        payload = json.loads(path.read_text())
        receipts.append(
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
        for row in payload["per_pair"]:
            rows[int(row["pair"])] = row

    per_pair: list[dict[str, Any]] = []
    control_a_failures = 0
    control_b_failures = 0
    admitted: list[int] = []

    for pair_str, want in sorted(predicted.items(), key=lambda kv: int(kv[0])):
        pair = int(pair_str)
        row = rows.get(pair)
        if row is None:
            per_pair.append({"pair": pair, "status": "MISSING_FROM_RESCREEN"})
            continue

        got_sweep = {
            (int(e["accept_separation"]), float(e["keep_fraction"])): (
                int(e["tokens"]),
                int(e["repaired"]),
            )
            for e in row["separation_sweep"]
        }
        want_sweep = retained.get(pair, {})
        shared = set(got_sweep) & set(want_sweep)
        mismatched = [k for k in shared if got_sweep[k] != want_sweep[k]]
        control_a = not mismatched and len(shared) == len(want_sweep)

        got_key = (
            int(row["accept_separation_chosen"]),
            float(row["keep_fraction_chosen"]),
        )
        want_key = (int(want["predicted_sep"]), float(want["predicted_kf"]))
        control_b = got_key == want_key

        if not control_a:
            control_a_failures += 1
        if not control_b:
            control_b_failures += 1
        ok = control_a and control_b
        if ok:
            admitted.append(pair)

        per_pair.append(
            {
                "pair": pair,
                "status": "ADMITTED" if ok else "EXCLUDED",
                "control_a_sweep_reproduces_jg3": control_a,
                "control_a_entries_shared": len(shared),
                "control_a_entries_retained": len(want_sweep),
                "control_a_mismatched_entries": [list(k) for k in mismatched],
                "control_b_winner_is_predicted": control_b,
                "predicted": {
                    "separation": want["predicted_sep"],
                    "keep_fraction": want["predicted_kf"],
                    "tokens": want["predicted_tokens"],
                    "repaired": want["predicted_repaired"],
                },
                "realized": {
                    "separation": got_key[0],
                    "keep_fraction": got_key[1],
                    "tokens": int(row["tokens_changed"]),
                    "repaired": int(row["repaired"]),
                },
                "chosen_by_jg5": {
                    "tokens": want["chosen_tokens"],
                    "repaired": want["chosen_repaired"],
                },
            }
        )

    # Compose: jg5's shipped 455 with the ADMITTED pairs' planes replaced.
    subset = np.load(args.jg5_subset)
    composed = {k: subset[k] for k in subset.files}
    edit_sources = {}
    for path in sorted(Path(args.store, "retained").glob(f"seg_edits_{args.tag}_s*.npz")):
        digest, size = sha256_of(path)
        edit_sources[str(path)] = {"sha256": digest, "bytes": size}
        with np.load(path) as handle:
            for key in handle.files:
                if int(key) in admitted:
                    plane = handle[key]
                    if plane.shape != (GRID_H, GRID_W):
                        raise Fs3ComposeError(
                            f"pair {key} plane has shape {plane.shape}"
                        )
                    composed[key] = plane

    missing = [p for p in admitted if str(p) not in composed]
    if missing:
        raise Fs3ComposeError(f"admitted pairs absent from the re-screen npz: {missing}")

    out_dir = Path(args.out)
    (out_dir / "retained" / "fields").mkdir(parents=True, exist_ok=True)
    # The filename names the RUN, not the direction this module was born for.
    # A hardcoded "reopen" would have mislabelled the mirror's own field -- the
    # same label-vs-arithmetic drift rv17 wave-3 W3-F2 caught in jg3's rate_source.
    npz_path = out_dir / "retained" / "fields" / f"seg_edits_{args.tag}_composed.npz"
    np.savez_compressed(npz_path, **composed)
    npz_sha, npz_bytes = sha256_of(npz_path)

    realized_tokens = sum(
        r["realized"]["tokens"] for r in per_pair if r["status"] == "ADMITTED"
    )
    realized_repaired = sum(
        r["realized"]["repaired"] for r in per_pair if r["status"] == "ADMITTED"
    )
    chosen_tokens = sum(
        r["chosen_by_jg5"]["tokens"] for r in per_pair if r["status"] == "ADMITTED"
    )
    chosen_repaired = sum(
        r["chosen_by_jg5"]["repaired"] for r in per_pair if r["status"] == "ADMITTED"
    )

    report = {
        "schema": "ddm_fs3_reopen_compose.v1",
        "arm": "ddm_fs3",
        "leg": 3,
        "axis": "seg MEASURED (realized cells through receiver + frozen CPU SegNet); rate pending the real re-encode",
        "score_claim": False,
        "promotion_eligible": False,
        "rescreen_receipts": receipts,
        "edit_sources": edit_sources,
        "controls": {
            "control_a_sweep_reproduces_jg3_retained": {
                "pairs_failing": control_a_failures,
                "verdict": "PASS" if control_a_failures == 0 else "FAIL",
            },
            "control_b_winner_is_the_predicted_configuration": {
                "pairs_failing": control_b_failures,
                "verdict": "PASS" if control_b_failures == 0 else "PARTIAL",
            },
        },
        "pairs_predicted": len(predicted),
        "pairs_admitted": len(admitted),
        "pairs_excluded": len(predicted) - len(admitted),
        "admitted_pairs": admitted,
        "realized_vs_predicted": {
            "tokens_before": chosen_tokens,
            "tokens_after": realized_tokens,
            "delta_tokens": realized_tokens - chosen_tokens,
            "predicted_delta_tokens": prereg["predicted_totals"]["delta_tokens"],
            "repaired_before": chosen_repaired,
            "repaired_after": realized_repaired,
            "delta_repaired": realized_repaired - chosen_repaired,
            "predicted_delta_repaired": prereg["predicted_totals"]["delta_repaired"],
        },
        "seg_credit_from_the_reopen_S": -(realized_repaired - chosen_repaired)
        * S_PER_SEG_CELL,
        "composed_field": {
            "path": str(npz_path),
            "sha256": npz_sha,
            "bytes": npz_bytes,
            "pairs": len(composed),
        },
        "per_pair": per_pair,
        "next": (
            "price the composed field with ddm_jg2_tail_reencode against this arm's "
            "own byte-identical control, then carrier re-solve + splice + byte-close"
        ),
    }
    # Same label discipline as the field filename: name the RUN, not the direction
    # this module was born for (rv17 wave-3 W3-F2 class).
    out_json = out_dir / f"FS3_COMPOSE_{args.tag}.json"
    out_json.write_text(json.dumps(report, indent=2))

    print(f"re-screen receipts: {len(receipts)}; pairs in prereg: {len(predicted)}")
    print(
        f"CONTROL A (sweep reproduces jg3): "
        f"{report['controls']['control_a_sweep_reproduces_jg3_retained']['verdict']} "
        f"({control_a_failures} failing)"
    )
    print(
        f"CONTROL B (winner is predicted):  "
        f"{report['controls']['control_b_winner_is_the_predicted_configuration']['verdict']} "
        f"({control_b_failures} failing)"
    )
    print(f"admitted {len(admitted)} / {len(predicted)} pairs")
    rv = report["realized_vs_predicted"]
    print(
        f"tokens {rv['tokens_before']} -> {rv['tokens_after']} "
        f"({rv['delta_tokens']:+d}, predicted {rv['predicted_delta_tokens']:+d})"
    )
    print(
        f"cells  {rv['repaired_before']} -> {rv['repaired_after']} "
        f"({rv['delta_repaired']:+d}, predicted {rv['predicted_delta_repaired']:+d})"
    )
    print(f"seg credit {report['seg_credit_from_the_reopen_S']:.6e} S")
    print(f"composed field {npz_path} ({npz_bytes} B, sha {npz_sha[:16]})")
    print(f"wrote {out_json}")
    return 0


def run_price(args: argparse.Namespace) -> int:
    """Compose the legs once the real re-encode has priced the composed field."""
    compose = json.loads(Path(args.compose).read_text())
    encode = json.loads(Path(args.encode_receipt).read_text())
    if not encode.get("delta_trustworthy", False):
        raise Fs3ComposeError(
            "the encode receipt is not delta_trustworthy; its control did not pass"
        )

    shipped_stream = int(args.shipped_stream_bytes)
    candidate_stream = int(encode["token_stream_bytes_candidate"])
    rate_bytes = candidate_stream - shipped_stream
    added_tokens = compose["realized_vs_predicted"]["delta_tokens"]

    seg = compose["seg_credit_from_the_reopen_S"]
    rate = rate_bytes * S_PER_ARCHIVE_BYTE
    realized_bits_per_token = (rate_bytes * 8.0 / added_tokens) if added_tokens else None
    calibration = float(args.calibration_bits_per_token)
    drift = (
        abs(realized_bits_per_token - calibration) / calibration
        if realized_bits_per_token
        else None
    )

    pose = float(args.pose_leg_s)
    carrier = compose["pairs_admitted"] * float(args.carrier_bytes_per_pair) * S_PER_ARCHIVE_BYTE
    net = seg + rate + pose + carrier

    report = {
        "schema": "ddm_fs3_reopen_price.v1",
        "arm": "ddm_fs3",
        "leg": 3,
        "score_claim": False,
        "promotion_eligible": False,
        "pairs": compose["pairs_admitted"],
        "added_tokens": added_tokens,
        "token_stream": {
            "shipped": shipped_stream,
            "candidate": candidate_stream,
            "delta_bytes": rate_bytes,
        },
        "realized_bits_per_token": realized_bits_per_token,
        "calibration_bits_per_token": calibration,
        "drift_from_calibration": drift,
        "drift_flag": (drift is not None and drift > 0.10),
        "legs": {
            "seg_MEASURED": seg,
            "rate_MEASURED_real_reencode": rate,
            "pose_DERIVED": pose,
            "carrier_DERIVED_extrapolated_leg2": carrier,
        },
        # NOT inside ``legs``: every value there is printed with a numeric format,
        # and the caveat is prose.  Keeping it adjacent is the point of W3-F15.
        "carrier_label_superseded": CARRIER_LABEL_SUPERSEDED,
        "net_delta_S": net,
        "multiple_of_bar": abs(net) / ADMISSION_BAR_S,
        "clears_admission_bar": net < -ADMISSION_BAR_S,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"rate {rate_bytes:+d} B over {added_tokens:+d} tokens = "
        f"{realized_bits_per_token:.4f} bits/token "
        f"(calibration {calibration:.4f}, drift {drift:.1%}"
        f"{' FLAG' if report['drift_flag'] else ''})"
    )
    for name, value in report["legs"].items():
        print(f"  {name:34s} {value:+.6e}")
    print(
        f"  {'NET':34s} {net:+.6e}  "
        f"({report['multiple_of_bar']:.2f}x bar) clears={report['clears_admission_bar']}"
    )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    merge = sub.add_parser("merge", help="merge shards, control them, compose the field")
    merge.add_argument("--store", default="/Volumes/APDataStore/pact/ddm_fs3/rescreen38")
    merge.add_argument("--tag", default="fs3_reopen38")
    merge.add_argument("--prereg", default=DEFAULT_PREREG)
    merge.add_argument("--jg5-subset", default=DEFAULT_JG5_SUBSET)
    merge.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_fs3")
    merge.set_defaults(func=run_merge)

    price = sub.add_parser("price", help="compose the legs after the real re-encode")
    price.add_argument(
        "--compose", default="/Volumes/APDataStore/pact/ddm_fs3/FS3_REOPEN_COMPOSE.json"
    )
    price.add_argument("--encode-receipt", required=True)
    price.add_argument("--shipped-stream-bytes", type=int, default=113_847)
    price.add_argument("--calibration-bits-per-token", type=float, default=2.657293497363796)
    price.add_argument("--pose-leg-s", type=float, required=True)
    price.add_argument("--carrier-bytes-per-pair", type=float, default=45.0 / 454.0)
    price.add_argument(
        "--out", default="/Volumes/APDataStore/pact/ddm_fs3/FS3_REOPEN_PRICE.json"
    )
    price.set_defaults(func=run_price)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
