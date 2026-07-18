#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""costate_live_ingest — EXTERNAL read-only ingestion of FACTORIZED features per live
verdict (organ upgrade C, 2026-07-17).

Tails/scans the live run's ``run.log`` with the canonical event vocabulary
(``tac.witness_run_monitor.classify_line`` — the same classifier the shell monitor uses,
never a hand-rolled grep) and, for each NEW ``verdict`` event, computes the factorized
features from the run's CURRENT EMA checkpoint + the frozen scorer + the bit-exact GT
cache, then APPENDS them:

  * a FEED-426-organ block into the costate organ's triality ledger via
    ``tac.witness_control.continual_costate.append_trajectory_record`` — the
    "independent compatible trajectory" the #516 exact factorized adjoint's digest row
    says is OWED (its DERIVED lambda_Road-Lane = 2.09x validation surface);
  * a compact row into ``.omx/state/witness_factorized_snapshot.jsonl`` — the store the
    costate digest's factorized-duty section (upgrade A) recomputes rankings from.

Features per verdict (all MEASURED on real bytes, subset-labeled):
  * visible/blind ENERGY SPLIT of the witness-vs-GT camera residual through the EXACT
    ``range(A)``/``ker(A)`` support split (closed-form taps, torch-verified) — the
    fraction of the witness's residual energy the scorer cannot see;
  * per-oriented-pair FLIP-DISTANCE histogram (rank-4 head law ``m/||w_c-w_c'||``) +
    margin histograms;
  * the sample d_seg cross-checked against the verdict row's own d_seg.

SACRED-run discipline: this tool only READS the run dir; every write goes to
``.omx/state`` / ``.omx/research`` ledgers.  IDEMPOTENT: a state file records ingested
epochs per run (and the ledger's per-run_ref dedup backstops it — run_ref is suffixed
``#factorized-ep<E>`` so organ tournament records for the SAME run are never clobbered).
NEVER blocks on the live run; ``--once`` processes the newest unseen verdict and exits.
Advisory only: ``[macOS-CPU advisory] NON-PROMOTABLE``, ``score_claim=False``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from tac.witness_run_monitor import classify_line  # noqa: E402

STATE_PATH = _REPO / ".omx" / "state" / "costate_live_ingest_state.json"
SNAPSHOT_JSONL = _REPO / ".omx" / "state" / "witness_factorized_snapshot.jsonl"
DEFAULT_GT_CACHE = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n600.npz"


def read_verdict_rows(run_log: Path) -> list[dict]:
    """All parseable FULL verdict rows in run.log, in file order.

    Uses the CANONICAL monitor vocabulary (``classify_line``) as the surface filter, then
    binds on the parsed ``stage == "verdict"`` + ``d_seg`` presence.  MEASURED wart
    (2026-07-17, live run.log): full verdict rows carry the benign ``"frozen_epoch":
    false`` field, which the monitor's ordered vocabulary classifies as
    ``confound_alarm`` (alarm patterns deliberately outrank progress), so category ==
    'verdict' alone would see ONLY the thin ``verdict_async_done`` rows.  The row's own
    parsed stage is therefore the binding test; classify_line gates that the line is a
    surfaced event at all.  (Monitor-vocabulary follow-up — excluding the benign
    ``"frozen_epoch": ?false`` substring — is queued for the post-run boundary; this tool
    does not edit live-adjacent monitor vocabulary mid-run.)"""
    rows: list[dict] = []
    if not run_log.is_file():
        return rows
    for ln in run_log.read_text(errors="replace").splitlines():
        if classify_line(ln) is None:
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if (isinstance(row, dict) and row.get("stage") == "verdict"
                and "epoch" in row and isinstance(row.get("d_seg"), (int, float))):
            rows.append(row)
    return rows


def _load_state() -> dict:
    try:
        st = json.loads(STATE_PATH.read_text())
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def _save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    tmp.replace(STATE_PATH)


def ingested_epochs(state: dict, run_name: str) -> set[int]:
    return {int(e) for e in (state.get(run_name) or [])}


def compute_factorized_features(
    run_dir: Path,
    verdict: dict,
    *,
    gt_cache: Path,
    n_pairs_sample: int = 12,
    energy_split: bool = True,
    segnet_cpu=None,
) -> tuple[dict, dict]:
    """The per-verdict factorized feature computation (REAL inputs; fail-closed).

    Returns ``(organ_payload, snapshot_row)``.  The EMA checkpoint consumed is the run's
    CURRENT on-disk EMA (its ``__epoch`` is recorded next to the verdict epoch — the
    binding is RECORDED, not assumed)."""
    from tac import witness_run_artifacts as wra
    from tac.witness_control.factorized_adjoint import factorization_provenance
    from tac.witness_control.factorized_features import (
        AXIS_TAG,
        default_pair_sample,
        ker_a_zero_weight_mask,
        load_gt_slices,
        load_witness_ema,
        snapshot_witness_margins,
        utc_stamp,
        visible_energy_split,
    )

    # bind the ROLLING EMA shadow (the checkpoint the verdict measures) first; BEST is the
    # retention copy and can be many epochs stale (pose-gated selection).
    ckpt = run_dir / wra.EMA_NPZ
    if not ckpt.is_file():
        ckpt = run_dir / wra.EMA_BEST_NPZ
    manifest, _params, _code = load_witness_ema(ckpt)
    pairs = default_pair_sample(int(manifest["n_pairs"]), n_pairs_sample)

    snap = snapshot_witness_margins(
        ckpt, gt_cache, pairs, segnet_cpu=segnet_cpu,
        keep_frames=energy_split, run_ref=run_dir.name,
    )
    row = snap.summary_row()
    epoch = int(verdict["epoch"])
    row["verdict_epoch"] = epoch
    row["verdict_d_seg"] = verdict.get("d_seg")
    row["verdict_d_seg_by_class"] = verdict.get("d_seg_by_class")
    row["verdict_flip_share_by_class"] = verdict.get("flip_share_by_class")
    row["ema_ckpt"] = str(ckpt)

    energy = None
    if energy_split:
        gt = load_gt_slices(gt_cache, list(pairs), want_frames=True)
        kmask = ker_a_zero_weight_mask()
        splits = []
        for i, f1 in enumerate(snap.frames1):
            resid = f1.astype(np.float64) - gt["gt_f1"][i].astype(np.float64)
            splits.append(visible_energy_split(resid, kmask))
        vis = [s["visible_frac"] for s in splits if s["visible_frac"] is not None]
        energy = {
            "residual_energy_visible_frac_mean": float(np.mean(vis)) if vis else None,
            "residual_energy_visible_frac_min": float(np.min(vis)) if vis else None,
            "ker_zero_weight_frac": float(kmask.mean()),
            "n_pairs": len(splits),
        }
        row["visible_blind_energy"] = energy

    # the adjoint-validation surface: per-pair flip-distance quantiles beside the
    # adjoint's DERIVED Road-Lane weighting ratio (the independent compatible trajectory)
    prov = factorization_provenance()
    lam = prov["derived"]["road_lane_gain_only_lambda_ratio_vs_other_median"]
    payload = {
        "run_ref": f"{run_dir.name}#factorized-ep{epoch}",
        "generated_at": utc_stamp(),
        "kind": "factorized_features_v1",
        "verdict_epoch": epoch,
        "verdict_d_seg": verdict.get("d_seg"),
        "ema_epoch": snap.ema_epoch,
        "sample": {
            "pair_indices": list(snap.pair_indices),
            "d_seg_sample": snap.d_seg_sample,
            "n_flips": snap.n_flips,
        },
        "visible_blind_energy": energy,
        "flipdist_feat_q_by_pair": {
            k: (row["by_oriented_pair"][k] or {}).get("flipdist_feat_q")
            for k in sorted(row.get("by_oriented_pair") or {})
        },
        "lambda_validation": {
            "adjoint_derived_road_lane_ratio": lam,
            "note": ("independent compatible trajectory for the #516 factorized-adjoint "
                     "validation (per-pair measured flip-distance/margin series)"),
        },
        "n_intervals": 0,
        "prototypes": [],
        "axis_tag": AXIS_TAG,
        "score_claim": False,
    }
    return payload, row


def ingest_new_verdicts(
    run_dir: Path,
    *,
    gt_cache: Path,
    n_pairs_sample: int = 12,
    energy_split: bool = True,
    only_latest: bool = True,
    ledger_path: Path | None = None,
    snapshot_jsonl: Path | None = None,
    state_override: dict | None = None,
) -> list[int]:
    """Process unseen verdict epochs (idempotent).  Returns the epochs ingested."""
    from tac.witness_control.continual_costate import append_trajectory_record
    from tac.witness_control.factorized_features import (
        load_frozen_segnet_cpu,
        locked_append_jsonl,
    )

    run_log = run_dir / "run.log"
    verdicts = read_verdict_rows(run_log)
    if not verdicts:
        return []
    state = _load_state() if state_override is None else state_override
    seen = ingested_epochs(state, run_dir.name)
    todo = [v for v in verdicts if int(v["epoch"]) not in seen]
    if not todo:
        return []
    if only_latest:
        todo = [max(todo, key=lambda v: int(v["epoch"]))]

    segnet = load_frozen_segnet_cpu()
    done: list[int] = []
    for v in todo:
        payload, row = compute_factorized_features(
            run_dir, v, gt_cache=gt_cache, n_pairs_sample=n_pairs_sample,
            energy_split=energy_split, segnet_cpu=segnet,
        )
        append_trajectory_record(payload, ledger_path=ledger_path)
        locked_append_jsonl(snapshot_jsonl or SNAPSHOT_JSONL, row)
        ep = int(v["epoch"])
        done.append(ep)
        state.setdefault(run_dir.name, [])
        if ep not in state[run_dir.name]:
            state[run_dir.name].append(ep)
        if state_override is None:
            _save_state(state)
        print(f"[costate-live-ingest] ep{ep}: sample d_seg {row['d_seg_sample']:.6f} "
              f"(verdict {v.get('d_seg')}), flips {row['n_flips']}, "
              f"visible-energy {((row.get('visible_blind_energy') or {}).get('residual_energy_visible_frac_mean'))}")
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="live witness run dir (READ-ONLY)")
    ap.add_argument("--gt-cache", default=str(DEFAULT_GT_CACHE))
    ap.add_argument("--pairs", type=int, default=12, help="stride-sampled pairs per snapshot")
    ap.add_argument("--no-energy-split", action="store_true",
                    help="skip the camera residual energy split (avoids the gt_f1 load)")
    ap.add_argument("--all-unseen", action="store_true",
                    help="ingest EVERY unseen verdict (default: only the newest)")
    ap.add_argument("--follow", action="store_true",
                    help="poll for new verdicts every --poll-s seconds (Ctrl-C to stop)")
    ap.add_argument("--poll-s", type=float, default=300.0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")
    gt_cache = Path(args.gt_cache)

    def _once() -> list[int]:
        return ingest_new_verdicts(
            run_dir, gt_cache=gt_cache, n_pairs_sample=args.pairs,
            energy_split=not args.no_energy_split, only_latest=not args.all_unseen,
        )

    if not args.follow:
        done = _once()
        print(f"[costate-live-ingest] ingested epochs: {done or 'none (idempotent no-op)'}")
        return 0
    try:
        while True:
            done = _once()
            if done:
                print(f"[costate-live-ingest] ingested epochs: {done}")
            time.sleep(max(30.0, float(args.poll_s)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
