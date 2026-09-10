"""ddm_gdc4 stage-00c: the adaptive-coder cost of the run-native symbol stream.

The v1/v2 packets are byte-oriented: brotli/LZMA see varint bytes, not symbols.
This probe measures what a context-adaptive arithmetic coder would actually
spend, symbol by symbol, on the same exact edit script. The accumulated cost is
``sum -log2 p_adaptive(symbol | context)`` under a sequential Krichevsky-Trofimov
style estimator, which is the honest code length of a real adaptive coder: the
model is learned online from already-decoded symbols only, so there is no
train/test leakage and no separate model to count.

The result answers the arm's decisive question: is the run-native description of
the move-44 token field reachable under the ``94,010 B`` door by coding alone,
or does it need a learned probability model?

``research_only=true`` · ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.gdc4_run_native_endpoint import (
    OP_END,
    OP_INSERT,
    OP_MATCH,
    OP_SUBST,
    plan_field_transitions,
)

FIELD_SHAPE = (600, 384, 512)
DOOR_TOTAL_B = 94_010
SHIPPED_TAIL_B = 119_969
DX_CLAMP = 12
DX_ESC = 2 * DX_CLAMP + 1


class AdaptiveModel:
    """Sequential adaptive model over a fixed alphabet, one per context."""

    __slots__ = ("alpha", "alphabet", "bits", "counts", "symbols", "totals")

    def __init__(self, alphabet: int, alpha: float = 0.35) -> None:
        self.alphabet = alphabet
        self.alpha = alpha
        self.counts: dict[object, list[float]] = {}
        self.totals: dict[object, float] = {}
        self.bits = 0.0
        self.symbols = 0

    def code(self, context: object, symbol: int) -> None:
        counts = self.counts.get(context)
        if counts is None:
            counts = [0.0] * self.alphabet
            self.counts[context] = counts
            self.totals[context] = 0.0
        total = self.totals[context]
        p = (counts[symbol] + self.alpha) / (total + self.alpha * self.alphabet)
        self.bits -= math.log2(p)
        counts[symbol] += 1.0
        self.totals[context] = total + 1.0
        self.symbols += 1

    def summary(self) -> dict[str, float | int]:
        return {
            "symbols": self.symbols,
            "bits": self.bits,
            "bytes": self.bits / 8.0,
            "bits_per_symbol": self.bits / self.symbols if self.symbols else 0.0,
            "contexts_used": len(self.counts),
        }


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=FIELD_SHAPE[0])
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    field_path = Path(args.field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(args.frames)

    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    view = np.asarray(field[:n_frames])  # SUBSET_SELECTION_OK: the codec is CAUSAL over frames (each frame's reference is the previous frame), so a contiguous prefix is the only coherent subset; --frames exists solely for local timing/round-trip smokes and every reported number in this arm is the full n600.

    m_ref = AdaptiveModel(2)
    m_lead = AdaptiveModel(6)
    m_op = AdaptiveModel(5)
    m_dx = AdaptiveModel(DX_ESC + 1)
    m_dx_esc = AdaptiveModel(1024)
    m_ins = AdaptiveModel(1024)
    m_icls = AdaptiveModel(5)
    m_scls = AdaptiveModel(5)

    dx_hist: dict[int, int] = defaultdict(int)
    op_counts = [0] * 5
    started = time.time()

    for _f, _y, ref_tag, lead_code, script, trans, ref_trans, prev_script in (
        plan_field_transitions(view)
    ):
        prev_dx = [a for op, a, _ in prev_script if op in (OP_MATCH, OP_SUBST)]
        n_ref = len(ref_trans)
        m_ref.code((min(n_ref, 8),), ref_tag)
        m_lead.code((ref_tag, min(n_ref, 8)), lead_code)

        prev_op = OP_END
        k = 0  # index among MATCH/SUBST ops in this row
        i_cur = 0
        prev_x = 0
        for op, arg0, arg1 in script:
            op_counts[op] += 1
            m_op.code((prev_op, min(k, 6), min(n_ref, 8)), op)
            if op in (OP_MATCH, OP_SUBST):
                dx = arg0
                dx_hist[dx] += 1
                ref_dx = prev_dx[k] if k < len(prev_dx) else 0
                ctx = (min(k, 6), max(-3, min(3, ref_dx)), op)
                if -DX_CLAMP <= dx <= DX_CLAMP:
                    m_dx.code(ctx, dx + DX_CLAMP)
                else:
                    m_dx.code(ctx, DX_ESC)
                    mag = abs(dx) - DX_CLAMP - 1
                    if not 0 <= mag < 1024:
                        raise ValueError(f"escape magnitude {mag} outside alphabet")
                    m_dx_esc.code((0 if dx < 0 else 1,), mag)
                if op == OP_SUBST:
                    m_scls.code((min(k, 6),), arg1)
                k += 1
                prev_x = trans[i_cur][0]
                i_cur += 1
            elif op == OP_INSERT:
                m_ins.code((min(k, 6),), max(0, min(1023, arg0 - prev_x + 512)))
                m_icls.code((min(k, 6),), arg1)
                prev_x = arg0
                i_cur += 1
            prev_op = op
        m_op.code((prev_op, min(k, 6), min(n_ref, 8)), OP_END)
        op_counts[OP_END] += 1

    elapsed = time.time() - started
    models = {
        "ref": m_ref,
        "lead": m_lead,
        "op": m_op,
        "dx": m_dx,
        "dx_escape": m_dx_esc,
        "ins": m_ins,
        "icls": m_icls,
        "scls": m_scls,
    }
    per_model = {name: model.summary() for name, model in models.items()}
    total_bits = sum(float(entry["bits"]) for entry in per_model.values())
    total_bytes = math.ceil(total_bits / 8.0)

    top_dx = sorted(dx_hist.items(), key=lambda kv: -kv[1])[:24]
    dx_total = sum(dx_hist.values())

    result = {
        "arm": "ddm_gdc4",
        "stage": "stage00c_adaptive_cost",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU byte-only n600]",
        "seed": int(args.seed),
        "frames": n_frames,
        "field": {
            "path": str(field_path),
            "sha256": sha256_file(field_path) if n_frames == FIELD_SHAPE[0] else None,
        },
        "op_counts": {
            "MATCH": op_counts[OP_MATCH],
            "SUBST": op_counts[OP_SUBST],
            "INSERT": op_counts[OP_INSERT],
            "DELETE": op_counts[3],
            "END": op_counts[OP_END],
        },
        "per_model": per_model,
        "adaptive_total_bits": total_bits,
        "adaptive_total_bytes": total_bytes,
        "door_total_bytes": DOOR_TOTAL_B,
        "shipped_tail_bytes": SHIPPED_TAIL_B,
        "vs_door_bytes": total_bytes - DOOR_TOTAL_B,
        "vs_door_ratio": total_bytes / DOOR_TOTAL_B,
        "vs_shipped_tail_ratio": total_bytes / SHIPPED_TAIL_B,
        "dx_distribution_top": [
            {"dx": int(k), "count": int(v), "share": v / dx_total} for k, v in top_dx
        ],
        "dx_total": dx_total,
        "elapsed_s": elapsed,
        "note": (
            "Adaptive cost is a code length, not a compressed file: it is what a "
            "context-adaptive arithmetic coder over these symbols would spend, "
            "with the model learned online from already-decoded symbols only."
        ),
    }

    out_path = out_dir / "RESULT.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, out_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
