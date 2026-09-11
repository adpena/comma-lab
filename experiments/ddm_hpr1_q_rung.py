"""ddm_hpr1 -- price the coder's scalar q, re-solved on the REFIT prior, held out.

MAIN's question: mxo3 measured that recalibrating the shipped `q` COSTS 183.53 B held out
-- on the OLD prior. Move 47 changed every weight the mixer reads, so q's optimum may have
moved with it.

WHAT IS SCORED. `ddm_hpr1_shape_price.py --collect-q` accumulates, from the same hook the
arithmetic encoder is fed from, a 64-bin calibration table of the coded row's own
confidence (its maximum probability) against whether that argmax was the symbol actually
coded, split into two folds by frame PARITY so the folds interleave the video. This
producer scores that table exactly as mxo3 scored its own: each fold is coded under the
OTHER fold's Krichevsky-Trofimov table, with back-off to the coarser per-fold rate so a
bin the fit fold never saw returns the prior rather than a coin flip. **The KT arithmetic
is IMPORTED from mxo3, never re-derived** -- the prior negative and this measurement must
be the same instrument or the comparison is meaningless.

Nothing here is scored by a frequency fitted on itself, so the number carries no overfit
and an in-sample gain cannot pass as a win.

Axis ``[macOS-CPU advisory / scorer-free held-out calibration measurement]``;
``score_claim=false``. This prices a RUNG, not an archive: a positive held-out gain would
license building one, and a negative closes the door on this prior as mxo3 closed it on
the old one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np

from experiments.ddm_mxo3_premix_oracle import KT_ALPHA, cross_bits

#: mxo3's own measurement on the OLD prior, the number this rung is compared against.
MXO3_OLD_PRIOR_HOLDOUT_RECALIBRATION_BYTES = -183.53
ARM_STORES = ("/Volumes/VertigoDataTier/pact/ddm_hpr1",)


class QRungError(RuntimeError):
    """A q-rung input is not the table this producer scores."""


def score(table_path: Path) -> dict:
    payload = np.load(table_path, allow_pickle=False)
    counts = payload["counts"].astype(np.float64)
    hits = payload["hits"].astype(np.float64)
    binary_bits = payload["binary_bits"].astype(np.float64)
    confidence_sum = payload["confidence_sum"].astype(np.float64)
    if counts.shape[0] != 2:
        raise QRungError("the calibration table must carry exactly two folds")
    if counts.sum() <= 0:
        raise QRungError("empty calibration table")

    # The back-off prior: each fold's own overall hit rate, broadcast across its bins. A
    # bin the fit fold never saw falls back to this rather than to a coin flip, which is
    # mxo3's rule and the reason a finer partition is not punished for being finer.
    fold_rate = (hits.sum(axis=1) / np.maximum(counts.sum(axis=1), 1.0))[:, None]
    prior = np.broadcast_to(fold_rate, counts.shape).copy()

    shipped = float(binary_bits.sum())
    held_out = cross_bits(counts, hits, prior)
    gain_bits = shipped - held_out
    observed_confidence = confidence_sum.sum() / counts.sum()
    observed_rate = hits.sum() / counts.sum()
    return {
        "schema": "ddm_hpr1_q_rung.v1",
        "table": {"path": str(table_path), "bins": int(counts.shape[1]), "symbols": int(counts.sum())},
        "kt_alpha": KT_ALPHA,
        "fold_definition": "frame parity",
        "fold_symbols": [int(x) for x in counts.sum(axis=1)],
        "shipped_binary_bits": shipped,
        "heldout_recalibrated_binary_bits": held_out,
        "heldout_recalibration_gain_bits": gain_bits,
        "heldout_recalibration_gain_bytes": gain_bits / 8.0,
        "mean_shipped_confidence": float(observed_confidence),
        "mean_argmax_correct": float(observed_rate),
        "calibration_error": float(observed_confidence - observed_rate),
        "mxo3_old_prior_holdout_recalibration_bytes": MXO3_OLD_PRIOR_HOLDOUT_RECALIBRATION_BYTES,
        "verdict": (
            "POSITIVE held-out gain: recalibrating q on the refit prior saves bytes"
            if gain_bits > 0
            else "NEGATIVE held-out: recalibration costs bytes, as it did on the old prior"
        ),
        "in_sample_cannot_pass": (
            "every fold is coded under the OTHER fold's table, so no frequency scores itself"
        ),
        "axis": "[macOS-CPU advisory / scorer-free held-out calibration measurement]",
        "score_claim": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not any(str(args.out).startswith(root) for root in ARM_STORES):
        raise QRungError("out must be inside this arm's store")
    result = score(args.table.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
