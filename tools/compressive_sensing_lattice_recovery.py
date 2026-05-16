#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI wrapper for compressive-sensing substrate-lattice recovery.

Thin wrapper around
:class:`tac.autopilot_rudin_daubechies.SubstrateLatticeRecovery` per
operator approval 2026-05-16 + the T4 Symposium Time-Traveler verdict
``.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md``.

Mathematical foundation
-----------------------
Per Daubechies-DeVore-Fornasier-Gunturk 2010 + Candes-Tao 2006: given N
substrates and K << N empirical anchors, recover a sparse-signal posterior
over which substrates are genuinely frontier-breaking via L1 minimization
in the Daubechies db4 wavelet basis with tree-structured sparsity prior
(Baraniuk-Cevher 2010).

Observability surface (per max-observability standing directive 2026-05-16)
---------------------------------------------------------------------------
* Input snapshot: lattice JSON + per-anchor JSONL + flags
* Output snapshot: SparseSignalPosterior.to_observability_record() + diff
  against any previous posterior at --diff-against-previous
* Decision-path: which basis (Haar vs db4), which sparsity prior (tree vs
  flat), tagged in confidence_tag
* Cite-chain: explicit ``[prediction; compressive-sensing-lattice-recovery;
  K={n}; N={n}; sparsity={s}; tree_depth={d}; basis={basis}]``

Sample lattice JSON
-------------------
::

    {
      "frontier_threshold_cpu": 0.192,
      "expected_sparsity": 5,
      "nodes": [
        {"node_id": "z3_g1", "support_level": 0, "low": 0.195, "high": 0.198, "class": "plateau_adjacent"},
        {"node_id": "nscs01", "support_level": 0, "low": 0.180, "high": 0.190, "class": "plateau_adjacent"},
        {"node_id": "nscs06_v8", "support_level": 0, "low": 0.150, "high": 0.180, "class": "frontier_pursuit"},
        {"node_id": "rudin_floor", "support_level": 0, "low": 0.10, "high": 0.13, "class": "asymptotic_pursuit"}
      ]
    }

Sample anchors JSONL
--------------------
::

    {"node_id": "z3_g1", "observed_score": 0.1986, "axis": "contest_cpu"}
    {"node_id": "nscs01", "observed_score": 0.1870, "axis": "contest_cpu"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make canonical helper importable without installed package via repo layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.autopilot_rudin_daubechies import (  # noqa: E402
    FrontierPursuitClass,
    SparseSignalPosterior,
    SubstrateLatticeNode,
    SubstrateLatticeRecovery,
    diff_sparse_signal_posteriors,
)


def _parse_lattice_json(path: Path) -> tuple[list[SubstrateLatticeNode], dict[str, Any]]:
    """Parse the lattice JSON; return (nodes, config_overrides)."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"lattice JSON must be a dict, got {type(data).__name__}")
    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, list):
        raise ValueError("lattice JSON must contain a 'nodes' list")
    nodes: list[SubstrateLatticeNode] = []
    for row in nodes_raw:
        if not isinstance(row, dict):
            raise ValueError(f"node row must be dict, got {type(row).__name__}")
        node_id = row.get("node_id") or row.get("id")
        if not node_id:
            raise ValueError(f"node row missing node_id: {row!r}")
        cls_raw = row.get("class") or row.get("frontier_pursuit_class") or "plateau_adjacent"
        try:
            cls = FrontierPursuitClass(cls_raw)
        except ValueError as exc:
            raise ValueError(
                f"unknown frontier_pursuit_class {cls_raw!r} for node {node_id!r}; "
                f"expected one of {[c.value for c in FrontierPursuitClass]}"
            ) from exc
        nodes.append(
            SubstrateLatticeNode(
                node_id=str(node_id),
                parent_id=row.get("parent_id"),
                support_level=int(row.get("support_level", 0)),
                predicted_band_low=float(row.get("low", row.get("predicted_band_low", 0.18))),
                predicted_band_high=float(row.get("high", row.get("predicted_band_high", 0.20))),
                frontier_pursuit_class=cls,
            )
        )
    config = {
        "frontier_threshold_cpu": float(data.get("frontier_threshold_cpu", 0.192)),
        "expected_sparsity": int(data.get("expected_sparsity", 5)),
        "use_daubechies_db4": bool(data.get("use_daubechies_db4", True)),
        "use_tree_structured_prior": bool(data.get("use_tree_structured_prior", True)),
    }
    return nodes, config


def _parse_anchors_jsonl(path: Path) -> list[tuple[str, float]]:
    """Parse the anchors JSONL; one row per anchor."""
    anchors: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"anchors JSONL line {lineno} not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            node_id = row.get("node_id") or row.get("id")
            score = row.get("observed_score") or row.get("score")
            if node_id is None or score is None:
                continue
            anchors.append((str(node_id), float(score)))
    return anchors


def run(
    *,
    lattice_json: Path,
    anchors_jsonl: Path | None,
    output_json: Path,
    diff_against_previous: Path | None,
) -> int:
    """End-to-end recovery + write a SparseSignalPosterior observability JSON."""
    nodes, config = _parse_lattice_json(lattice_json)
    lr = SubstrateLatticeRecovery(
        use_daubechies_db4=config["use_daubechies_db4"],
        use_tree_structured_prior=config["use_tree_structured_prior"],
        expected_sparsity=config["expected_sparsity"],
        frontier_threshold_cpu=config["frontier_threshold_cpu"],
    )
    for n in nodes:
        lr.add_node(n)
    if anchors_jsonl is not None:
        for node_id, score in _parse_anchors_jsonl(anchors_jsonl):
            if node_id not in [n.node_id for n in nodes]:
                print(
                    f"warning: anchor at unknown node {node_id!r} skipped",
                    file=sys.stderr,
                )
                continue
            lr.update_from_anchor(node_id, score)
    posterior = lr.recover_sparse_signal()
    record = posterior.to_observability_record()
    record["schema"] = "compressive_sensing_lattice_recovery_v1"
    record["frontier_threshold_cpu"] = config["frontier_threshold_cpu"]
    record["expected_sparsity"] = config["expected_sparsity"]
    if diff_against_previous is not None and diff_against_previous.exists():
        with diff_against_previous.open("r", encoding="utf-8") as fh:
            prev = json.load(fh)
        record["diff_against_previous"] = _materialize_diff(prev, record)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"[compressive-sensing-lattice-recovery] K={posterior.n_anchors} "
        f"N={posterior.n_substrates} sparsity={posterior.sparsity_recovered} "
        f"basis={posterior.basis} tree_depth={posterior.tree_depth_max}",
        file=sys.stderr,
    )
    print(f"[compressive-sensing-lattice-recovery] {posterior.confidence_tag}", file=sys.stderr)
    print(f"[compressive-sensing-lattice-recovery] wrote {output_json}", file=sys.stderr)
    return 0


def _materialize_diff(prev_record: dict[str, Any], current_record: dict[str, Any]) -> dict[str, Any]:
    """Materialize a previous posterior from JSON dict back to SparseSignalPosterior
    and run diff_sparse_signal_posteriors."""

    def _from_record(rec: dict[str, Any]) -> SparseSignalPosterior:
        return SparseSignalPosterior(
            n_substrates=int(rec.get("n_substrates", 0)),
            n_anchors=int(rec.get("n_anchors", 0)),
            sparsity_recovered=int(rec.get("sparsity_recovered", 0)),
            basis=str(rec.get("basis", "haar_db1")),
            tree_depth_max=int(rec.get("tree_depth_max", 0)),
            posterior_frontier_probability=tuple(
                (r["node_id"], r["probability"])
                for r in rec.get("posterior_frontier_probability", [])
            ),
            posterior_score_band=tuple(
                (r["node_id"], r["low"], r["high"])
                for r in rec.get("posterior_score_band", [])
            ),
            recovery_uncertainty=tuple(
                (r["node_id"], r["uncertainty"])
                for r in rec.get("recovery_uncertainty", [])
            ),
            confidence_tag=str(rec.get("confidence_tag", "")),
        )

    return diff_sparse_signal_posteriors(
        _from_record(prev_record), _from_record(current_record)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lattice-json",
        type=Path,
        required=True,
        help="Path to lattice JSON describing the substrate nodes",
    )
    parser.add_argument(
        "--anchors-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL of empirical anchors to feed the recovery",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the SparseSignalPosterior observability JSON",
    )
    parser.add_argument(
        "--diff-against-previous",
        type=Path,
        default=None,
        help="Optional path to a prior posterior JSON to diff against",
    )
    args = parser.parse_args(argv)
    if not args.lattice_json.is_file():
        print(f"lattice JSON not found: {args.lattice_json}", file=sys.stderr)
        return 2
    if args.anchors_jsonl is not None and not args.anchors_jsonl.is_file():
        print(f"anchors JSONL not found: {args.anchors_jsonl}", file=sys.stderr)
        return 2
    return run(
        lattice_json=args.lattice_json,
        anchors_jsonl=args.anchors_jsonl,
        output_json=args.output_json,
        diff_against_previous=args.diff_against_previous,
    )


if __name__ == "__main__":
    sys.exit(main())
