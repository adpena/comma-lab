#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Synthetic non-comma example of `tac.optimizer.sweep_plugin`.

This script is the answer to "what does it look like to use the closed-loop
driver for my own rank-then-sweep workload without comma-specific knowledge?"

It defines a toy compression-style problem where:
  - candidates are configurations of a (size, accuracy) tradeoff
  - "ranking" sorts by predicted_loss = size + 0.5 * accuracy_gap
  - "dispatch" is just a local subprocess that prints + sleeps (no GPU)
  - "harvest" reads the subprocess return code and a fake score file

Run:
    .venv/bin/python examples/synthetic_sweep.py

Expected output:
    [synthetic-sweep] generated 6 candidates
    [synthetic-sweep] ranked top-3:
      ...
    [synthetic-sweep] dispatched 3 candidates (dry-run)
    [synthetic-sweep] DONE

The point: zero comma-specific imports. Plug-in pattern lifts cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimizer.sweep_plugin import (
    Candidate,
    CandidateGenerator,
    DispatchSpec,
    list_generators,
    load_generator,
    register_generator,
)


# ── Plugin definition ─────────────────────────────────────────────────────


class ToyCompressionGenerator(CandidateGenerator):
    """Synthetic generator: 6 (size, accuracy_gap) tradeoff candidates."""

    name = "toy_compression"

    def __init__(self) -> None:
        # A small grid; in a real workload these come from disk inspection.
        self._configs = [
            ("tiny",   12345,  0.40),
            ("small",  23456,  0.20),
            ("medium", 45678,  0.10),
            ("large",  98765,  0.05),
            ("xl",    198000,  0.02),
            ("xxl",   400000,  0.01),
        ]

    def __call__(self) -> list[Candidate]:
        candidates: list[Candidate] = []
        for cid, size_bytes, gap in self._configs:
            candidates.append({
                "candidate_id": f"toy_{cid}",
                "archive_bytes": size_bytes,
                "rel_err_pct": gap * 100.0,        # synthetic distortion proxy
                "lane_class": "toy_compression",
                "evidence_semantics": "synthetic_test",
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["synthetic_workload_no_real_eval"],
                "score_claim": False,
            })
        return candidates

    def build_dispatch(self, candidate: Candidate, *, label: str) -> DispatchSpec:
        # In a real workload this would launch a remote training/eval job.
        # Here we just echo the label and exit 0, suitable for a smoke test.
        cmd = [sys.executable, "-c",
               f"print('synthetic-dispatch label={label} bytes={candidate['archive_bytes']}')"]
        return DispatchSpec(label=label, cmd=cmd, estimated_cost_usd=0.0,
                            timeout_seconds=10.0)


# Register on import — symmetry with `tac.optimizer.generators.apogee_intn`.
register_generator("toy_compression", ToyCompressionGenerator)


# ── Tiny driver showing how to use the plugin from outside the loop tool ─


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Toy ranker: lower (bytes/1e5 + rel_err_pct/100) is better."""
    def key(c: Candidate) -> float:
        return c["archive_bytes"] / 1e5 + c["rel_err_pct"] / 100.0
    return sorted(candidates, key=key)


def main() -> int:
    print("[synthetic-sweep] registered generators:", list_generators())
    gen = load_generator("toy_compression")

    candidates = gen()
    print(f"[synthetic-sweep] generated {len(candidates)} candidates")

    ranked = _rank(candidates)
    print("[synthetic-sweep] ranked top-3:")
    for c in ranked[:3]:
        print(f"   {c['candidate_id']:>12s}  bytes={c['archive_bytes']:>7d}  "
              f"rel_err={c['rel_err_pct']:.2f}%")

    # In a real run we'd fan these out via the loop driver. Here we just
    # show the dispatch spec for the top candidate.
    top = ranked[0]
    spec = gen.build_dispatch(top, label="synthetic_demo_c1")
    print("[synthetic-sweep] dispatch spec for top candidate:")
    print(f"   label : {spec.label}")
    print(f"   cmd   : {' '.join(spec.cmd)}")
    print(f"   cost  : ${spec.estimated_cost_usd:.2f}")

    print("[synthetic-sweep] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
