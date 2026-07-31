#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_ep2 — burn-4 ENDPOINT SELECTION on an EXPLICIT candidate list, per-class, composed-S.

WHY THIS EXISTS (and why it is not ``ddm_b4r_endpoint_extras.py`` run as-is):
``ddm_b4r_endpoint_extras.build_extras`` measures the burn's ACCEPTED TRAJECTORY (parent +
each ``window_NN``'s ``stage_seg_trunk_tau_final.npz``).  Endpoint SELECTION is a different
question: it ranks an EXPLICIT, mixed set of candidates -- a window final AND named intra-window
epochs -- against each other.  This module supplies only that driver.  The MEASUREMENT itself is
``ddm_b4r_endpoint_extras.measure_checkpoint`` **imported and called verbatim**; no rendering,
EMA, quantization, per-class or flip-matrix logic is re-implemented here (re-deriving it would
be exactly the divergence the campaign's convention exists to prevent).

THREE DISCIPLINES THIS DRIVER ENFORCES (each from a landed round-1 review finding):

1. **SELECT ON COMPOSED S, NOT d_seg.**  ``S_additive = 100*d_seg + 25*bytes/37_545_489``,
   computed through :mod:`tac.contest_score` (``seg_term`` + ``rate_term``) -- the canonical
   SoT, never hand-rolled.  Bytes vary ~5.3 kB across burn-4 candidates = ~0.0035 S, which is
   a third of the claimed selection gain; a d_seg-only ranking picks the composed-second.
   The pose term is DELIBERATELY ABSENT (not passed as ``d_pose=0``, which would read as a
   pose claim): these candidates carry no measured pose, so ``S_additive`` is a two-axis
   partial, and it is labelled as such on every row.

2. **PER-CLASS ALWAYS VISIBLE, LANE NAMED.**  The burn-4 ALARM was a per-class Lane trade; a
   total-only comparison can buy total d_seg with Lane components.  Every candidate reports all
   five classes, and the winner-vs-control per-class delta is emitted with Lane called out by
   name.  There is no single-headline collapse.

3. **CHUNK-INVARIANCE IS MEASURED, NOT ASSUMED.**  ``chunk`` is a memory knob that must not move
   the number (eval-mode BatchNorm uses running stats; argmax is per-pixel; d_seg is a per-pair
   mean).  The window_02 control has a pre-existing n600 ``full_confirm`` measured at the
   trainer's ``verdict_chunk=32``; re-measuring it here at a different chunk is a POSITIVE
   CONTROL on the instrument.  A mismatch invalidates the whole run and is reported as such
   rather than silently averaged away.

CUSTODY FACT (verified, ddm_ep2): ``window_02/checkpoints/stage_seg_trunk_tau_final.npz`` and
``intra_seg_trunk_tau_ep00805.npz`` differ in EXACTLY ONE array -- ``meta::epoch`` (806 vs 805).
All 20 EMA arrays and all weights are ``np.array_equal``.  The ALARM's rollback target and the
already-measured "ep805" state are therefore the SAME WEIGHTS, so the pre-measured control value
legitimately applies to the rollback target.

``score_claim=False``.  Every number here is ``[macOS-CPU/MLX advisory]`` on the campaign's
frozen-scorer n600 convention -- NOT an evaluator row, and it may never become one by relabelling.
Pointer 0.1910828242 [contest-CPU] UNMOVED.

Usage:
  ddm_ep2_endpoint_select.py --candidate LABEL=<ckpt path> [--candidate ...]
                             [--control-label LABEL] [--control-expected-d-seg X]
                             [--chunk 64] [--pairs 600] --output-json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_PAIRS = 600
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
LANE = 1


def _parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--candidate must be LABEL=<path>, got {spec!r}")
    label, _, raw = spec.partition("=")
    label = label.strip()
    path = Path(raw.strip()).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"--candidate has an empty LABEL: {spec!r}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"--candidate {label}: no such checkpoint: {path}")
    return label, path


def _arm_dir_for(ckpt: Path) -> Path:
    """The window dir holding ``tr1_config.json`` for this checkpoint.

    Checkpoints live at ``<window>/checkpoints/<name>.npz``; the config the measurement needs
    (render geometry, quant schedule, byte ledger coder) sits at ``<window>/tr1_config.json``.
    Fail LOUD rather than silently measuring against a neighbouring window's config -- a wrong
    config would change the render and produce a plausible, wrong number.
    """
    arm = ckpt.parent.parent
    cfg = arm / "tr1_config.json"
    if not cfg.is_file():
        raise FileNotFoundError(
            f"no tr1_config.json for {ckpt}: expected {cfg}. Refusing to measure with a "
            f"substitute config (it would silently change the render).")
    return arm


def measure_candidates(candidates: list[tuple[str, Path]], chunk: int,
                       pairs: int = N_PAIRS) -> list[dict]:
    """Serial n600 measurement, one candidate at a time (ONE full-n600 scorer job at a time)."""
    from ddm_b4r_endpoint_extras import measure_checkpoint

    rows: list[dict] = []
    for label, ckpt in candidates:
        t0 = time.time()
        try:
            m = measure_checkpoint(_arm_dir_for(ckpt), ckpt, chunk, pairs)
            m["label"] = label
            m["wall_seconds"] = time.time() - t0
            m["chunk"] = int(chunk)
        except Exception as exc:  # one bad candidate never costs the others
            m = {"label": label, "ckpt": str(ckpt), "error": repr(exc),
                 "wall_seconds": time.time() - t0, "chunk": int(chunk)}
        rows.append(m)
        print(f"[ddm_ep2] {label}: "
              + (m["error"] if "error" in m else
                 f"d_seg={m['n600_d_seg']:.9f} bytes={m['total_counted_bytes']} "
                 f"S_add={m['s_additive']:.6f} ({m['wall_seconds']:.0f}s)"),
              file=sys.stderr, flush=True)
    return rows


def _composed_s(d_seg: float, total_bytes: int) -> dict:
    """Composed S through the canonical SoT. NEVER hand-rolled.

    The pose term is absent by construction, not zeroed: these candidates carry no measured
    d_pose, so this is a two-axis partial and is labelled that way wherever it is emitted.
    """
    from tac.contest_score import rate_term, seg_term

    seg = seg_term(d_seg)
    # Rate on the campaign's counted-payload ledger, NOT archive.zip bytes: pass the ledger
    # value through the canonical rate_term so the x25 and the denominator cannot slip, and
    # carry the caveat on the row.
    rate = rate_term(total_bytes)
    return {"seg_term_s": seg, "rate_term_s": rate, "s_additive": seg + rate,
            "pose_term_s": None,
            "s_additive_is_two_axis_partial": True}


def build_selection(rows: list[dict], control_label: str | None,
                    control_expected_d_seg: float | None) -> dict:
    ok = [r for r in rows if "error" not in r]
    for r in ok:
        r["composed"] = _composed_s(r["n600_d_seg"], r["total_counted_bytes"])
        # sanity: the driver's composed S must reproduce the measurement tool's own s_additive
        r["composed"]["reproduces_tool_s_additive"] = bool(
            abs(r["composed"]["s_additive"] - r["s_additive"]) < 1e-12)

    ranked = sorted(ok, key=lambda r: r["composed"]["s_additive"])
    ranked_by_dseg = sorted(ok, key=lambda r: r["n600_d_seg"])

    # --- POSITIVE CONTROL: does a differently-chunked re-measure reproduce the banked value?
    control: dict = {"status": "not_requested"}
    if control_label is not None:
        c = next((r for r in ok if r["label"] == control_label), None)
        if c is None:
            control = {"status": "CONTROL_FAILED_TO_MEASURE", "label": control_label}
        elif control_expected_d_seg is None:
            control = {"status": "no_expected_value_supplied", "label": control_label,
                       "measured_d_seg": c["n600_d_seg"]}
        else:
            err = abs(c["n600_d_seg"] - control_expected_d_seg)
            control = {
                "status": "REPRODUCED" if err < 1e-12 else "MISMATCH",
                "label": control_label,
                "expected_d_seg_banked_at_verdict_chunk_32": control_expected_d_seg,
                "measured_d_seg": c["n600_d_seg"],
                "abs_err": err,
                "chunk_used_here": c.get("chunk"),
                "interpretation": (
                    "chunk-invariance of the n600 verdict is MEASURED, not assumed; the "
                    "instrument reproduces its banked value"
                    if err < 1e-12 else
                    "INSTRUMENT DID NOT REPRODUCE ITS BANKED VALUE. Every ranking in this "
                    "file is SUSPECT until this is explained. Do not select on it."),
            }

    # --- per-class deltas vs the control, Lane named (never collapsed to a headline)
    per_class_vs_control: list[dict] = []
    ctl = next((r for r in ok if r["label"] == control_label), None) if control_label else None
    if ctl is not None:
        for r in ok:
            if r["label"] == control_label:
                continue
            d = [r["per_class"]["s_units"][c] - ctl["per_class"]["s_units"][c] for c in range(5)]
            per_class_vs_control.append({
                "label": r["label"], "vs": control_label,
                "delta_s_per_class": dict(zip(CLASS_NAMES, d)),
                "lane_delta_s": d[LANE],
                "lane_direction": "ERODED (Lane cost)" if d[LANE] > 0 else "Lane improved",
                "eroding_classes": [CLASS_NAMES[c] for c in range(5) if d[c] > 0],
                "improving_classes": [CLASS_NAMES[c] for c in range(5) if d[c] < 0],
                "delta_seg_s": r["composed"]["seg_term_s"] - ctl["composed"]["seg_term_s"],
                "delta_rate_s": r["composed"]["rate_term_s"] - ctl["composed"]["rate_term_s"],
                "delta_composed_s": r["composed"]["s_additive"] - ctl["composed"]["s_additive"],
                "trade_read": (
                    "buys total d_seg WITH Lane components — the ALARM's exact signature; "
                    "report BOTH numbers, never the total alone"
                    if d[LANE] > 0 and (r["n600_d_seg"] < ctl["n600_d_seg"])
                    else "no Lane-for-total trade at this candidate"),
            })

    winner = ranked[0]["label"] if ranked else None
    dseg_winner = ranked_by_dseg[0]["label"] if ranked_by_dseg else None
    return {
        "schema": "ddm_ep2_endpoint_select.v1",
        "selection_rule": "argmin over composed S_additive = 100*d_seg + 25*bytes/37_545_489, "
                          "computed via tac.contest_score (canonical SoT). d_seg-only ranking "
                          "is emitted alongside ONLY to expose disagreement, never to select.",
        "winner_on_composed_s": winner,
        "winner_on_d_seg_alone": dseg_winner,
        "ranking_rules_disagree": bool(winner != dseg_winner),
        "ranking_composed_s": [
            {"label": r["label"], "s_additive": r["composed"]["s_additive"],
             "n600_d_seg": r["n600_d_seg"], "total_counted_bytes": r["total_counted_bytes"],
             "lane_s": r["per_class"]["s_units"][LANE]} for r in ranked],
        "control_reproduction": control,
        "per_class_vs_control": per_class_vs_control,
        "per_candidate": rows,
        "class_order": list(CLASS_NAMES),
        "class_order_provenance": "comma10k CANONICAL [Road, Lane, Undrivable, Movable, MyCar] "
                                  "— MEASURED (CLAUDE.md non-negotiable); NEVER luma-sorted",
        "rate_caveat_that_must_travel": "total_counted_bytes is a COUNTED-PAYLOAD LEDGER, not "
                                        "archive.zip bytes. Every composed S here is the "
                                        "campaign's ADVISORY convention. Only upstream/"
                                        "evaluate.py on the exact archive bytes is a score.",
        "n600_convention": "frozen CPU-torch SegNet, ema_shadow, gt_n600.npz; the campaign's "
                           "ADVISORY convention — never an evaluator row by relabelling",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=_parse_candidate, action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable; measured SERIALLY in the given order")
    ap.add_argument("--control-label", default=None,
                    help="label of the candidate whose n600 d_seg is already banked")
    ap.add_argument("--control-expected-d-seg", type=float, default=None,
                    help="the banked n600 d_seg for --control-label (positive control)")
    ap.add_argument("--chunk", type=int, default=64,
                    help="verdict/render chunk. Memory knob ONLY: result-neutral by "
                         "construction (eval-mode BN running stats, per-pixel argmax, "
                         "per-pair mean) and verified against the banked control value.")
    ap.add_argument("--pairs", type=int, default=N_PAIRS,
                    help="CODE-PATH SMOKE ONLY when != 600 (CLAUDE.md allergic-to-non-n600).")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    if args.pairs < 1 or args.chunk < 1:
        ap.error("--pairs and --chunk must be >= 1")
    if args.chunk > 120:
        ap.error("--chunk > 120 is refused: the unchunked/large-chunk n600 verdict is the "
                 "measured #205 OOM class (CLAUDE.md forbidden-pattern).")

    rows = measure_candidates(args.candidate, args.chunk, args.pairs)
    out = build_selection(rows, args.control_label, args.control_expected_d_seg)
    out["n_pairs"] = int(args.pairs)
    out["CODE_PATH_SMOKE_NOT_A_MEASUREMENT"] = bool(args.pairs != N_PAIRS)
    out["measurement_valid"] = bool(args.pairs == N_PAIRS)

    payload = json.dumps(out, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(args.output_json)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
