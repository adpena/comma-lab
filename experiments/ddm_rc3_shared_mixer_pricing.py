#!/usr/bin/env python3
"""Scorer-free closed-form IHS1 context pricing; no encoder is invoked.

Research-only. All decoded bytes and tables are retained on the SSD. A completed
pricing receipt is an idempotent stage checkpoint, resumed with --resume-from.
Miller--Madow is an asymptotic estimate, never a finite-sample coding bound.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import brotli

if TYPE_CHECKING:
    import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_rc2_hpac_semistatic_depth_mixing as prior
from experiments import ddm_rc2_hpac_semistatic_mixing_codec as rc2

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor")
LIVE = Path("/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime")
PIN = "c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e"
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"


def retain(path: Path, payload: bytes) -> dict:
    """Atomic immutable stage output; never overwrite nonidentical evidence."""
    return prior.atomic_bytes(path, payload)


def save_json(path: Path, value: object) -> dict:
    return retain(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def source() -> tuple[bytes, list[int], dict]:
    """Read the pinned receiver and retain the exact body it restores."""
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < 256 * 1024**2:
        raise RuntimeError("SSD preflight refuses: less than 256 MiB available")
    archive = (LIVE / "archive.zip").read_bytes()
    assert len(archive) == 181414 and hashlib.sha256(archive).hexdigest() == PIN
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    assert pointer["effective_frontier"]["archive_sha256"] == PIN
    script = """import json,sys,torch
from pathlib import Path
torch.manual_seed(20260909); torch.set_num_threads(2)
r=Path(sys.argv[1]); sys.path[:0]=[str(r/'cpr1'),str(r)]
import inflate
from runtime.ihs2 import layout_from_runtime
l=layout_from_runtime(inflate)
print(json.dumps({'row_counts': l.row_counts, 'module_ranges':l.module_ranges}))
"""
    done = subprocess.run(
        [sys.executable, "-B", "-c", script, str(LIVE)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    layout = json.loads(done.stdout.strip().splitlines()[-1])
    counts = layout["row_counts"]
    member = prior.jg2.read_archive_member(LIVE / "archive.zip")
    sections = prior.jg2.split_member(member)
    outer = brotli.decompress(sections["hpac"])
    rider = prior.rc1.ck2_uninterleave(outer)
    spec = importlib.util.spec_from_file_location("rc3_pinned_reader", LIVE / "runtime/rc2_hpac_semistatic_mixing.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    decoded = reader.parse_rider(rider, counts)
    body = decoded["body"]
    rows, depths = rc2.unpack_rows(body, counts)
    prefix, packed, tail, _ = rc2.split_ihs1(body, counts)
    assert rc2.pack_rows(rows, depths) == packed
    facts = {}
    for name, payload in {
        "archive.zip": archive,
        "member.rx1": member,
        "hpac.br": sections["hpac"],
        "hpac.ck2": outer,
        "rider.rc2h": rider,
        "body.ihs1": body,
        "weights.i8": decoded["weights_blob"],
        "range.bin": decoded["payload"],
        "prefix.bin": prefix,
        "tail.bin": tail,
    }.items():
        facts[name] = retain(STORE / "retained/source" / name, payload)
    save_json(STORE / "retained/source/layout.json", layout)
    # Re-measure predecessor credit from retained bytes, not its scalar receipt.
    before = Path(
        "/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/retained/source/live_hpac.br"
    ).read_bytes()
    facts["rc1_predecessor_hpac"] = retain(STORE / "retained/source/rc1_predecessor.br", before)
    facts["measured_rc2_credit_bytes"] = len(before) - len(sections["hpac"])
    facts["other_sections"] = {
        k: retain(STORE / "retained/source/sections" / (k + ".bin"), sections[k])
        for k in ("semantic", "carrier", "tail")
    }
    facts["pointer"] = pointer["our_local_frontier_contest_cuda"]
    facts["free_bytes_at_start"] = free
    return body, counts, facts


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return sum(c * math.log2(n / c) for c in counts.values()) if n else 0.0


def symbol_contexts(rows: list[np.ndarray], depths: np.ndarray) -> dict:
    """Full 20,416-value census with explicit causal geometry and start states."""
    tables = {
        name: defaultdict(Counter)
        for name in (
            "depth",
            "prev",
            "position",
            "sibling_group",
            "prev_position",
            "prev_group",
            "order2",
            "order2_coarse",
            "prev_position_group",
        )
    }
    sequences = defaultdict(list)
    history = defaultdict(list)
    group, last_width = -1, None
    for row, depth in zip(rows, depths.tolist(), strict=True):
        if len(row) != last_width:
            group += 1
            last_width = len(row)
        d = int(depth)
        for pos, value in enumerate(row.tolist()):
            h = history[d]
            prev = h[-1] if h else None
            prev2 = h[-2] if len(h) > 1 else None
            quart = min(3, 4 * pos // len(row))
            sign2 = None if prev2 is None else (prev2 > 0) - (prev2 < 0)
            contexts = {
                "depth": (d,),
                "prev": (d, prev),
                "position": (d, quart),
                "sibling_group": (d, group),
                "prev_position": (d, prev, quart),
                "prev_group": (d, prev, group),
                "order2": (d, prev2, prev),
                "order2_coarse": (d, sign2, prev),
                "prev_position_group": (d, prev, quart, group),
            }
            for name, key in contexts.items():
                tables[name][key][value] += 1
            sequences[d].append(value)
            h.append(value)
    result = {}
    for name, cells in tables.items():
        plugin = sum(entropy(c) for c in cells.values())
        dof = sum(len(c) - 1 for c in cells.values())
        occupied = sum(len(c) for c in cells.values())
        supported = sum(
            sum(c.values()) for c in cells.values() if sum(c.values()) >= 5 * len(c) and min(c.values()) >= 2
        )
        result[name] = {
            "plugin_bytes": plugin / 8,
            "miller_madow_estimate_bytes": (plugin + dof / (2 * math.log(2))) / 8,
            "contexts": len(cells),
            "occupied_outcomes": occupied,
            "singleton_outcomes": sum(v == 1 for c in cells.values() for v in c.values()),
            "supported_symbols": supported,
            "total_symbols": sum(len(s) for s in sequences.values()),
            "support_rule": "context n >= 5 occupied outcomes, every outcome count >= 2",
            "is_finite_sample_bound": False,
        }
        save_json(
            STORE / "retained/counts" / (name + ".json"),
            [{"context": k, "counts": sorted(c.items())} for k, c in cells.items()],
        )
    # Exact rc2 convention: first symbol at each depth costs its marginal log loss.
    h1 = 0.0
    for seq in sequences.values():
        joint, ctx, marginal = Counter(pairwise(seq)), Counter(seq[:-1]), Counter(seq)
        h1 += entropy(joint) - entropy(ctx) + (len(joint) - len(ctx)) / (2 * math.log(2))
        h1 -= math.log2(marginal[seq[0]] / len(seq))
    result["rc2_h1_convention_bytes"] = h1 / 8
    return result


def run(resume_from: Path | None) -> dict:
    target = STORE / "ENTROPY_PRICING.json"
    if resume_from:
        if resume_from.resolve() != target.resolve():
            raise ValueError("resume-from must name this stage receipt")
        receipt = json.loads(resume_from.read_text())
        for fact in receipt["source"].values():
            if isinstance(fact, dict) and "path" in fact and "sha256" in fact:
                assert hashlib.sha256(Path(fact["path"]).read_bytes()).hexdigest() == fact["sha256"]
        return receipt
    body, counts, facts = source()
    rows, depths = rc2.unpack_rows(body, counts)
    contexts = symbol_contexts(rows, depths)
    j = facts["range.bin"]["bytes"] + facts["weights.i8"]["bytes"]
    result = {
        "schema": "ddm_rc3_entropy_pricing.v1",
        "axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "seed": 20260909,
        "source": facts,
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contexts": contexts,
        "rc2_J_bytes": j,
        "remaining_J_minus_H1_MM_bytes": j - contexts["rc2_h1_convention_bytes"],
        "coder_runs": 0,
        "interpretation": "Conditional entropy estimates are optimistic context-capacity diagnostics, not attainable lengths or rigorous bounds. Mixer forecast is a separate gate.",
        "cleanup": "All stage artifacts retained; no scratch or raw video is produced. No deletions.",
        "command": f".venv/bin/python {Path(__file__).relative_to(REPO)}",
    }
    save_json(target, result)
    return result


def supported_contexts() -> dict:
    """Price supported subsets on matching denominators; exclude zero-depth rows."""
    output = {}
    for name in ("prev_position", "prev_group", "order2", "order2_coarse", "prev_position_group"):
        cells = json.loads((STORE / "retained/counts" / f"{name}.json").read_text())
        selected = []
        parents = defaultdict(Counter)
        for c in cells:
            key = c["context"]
            counts = Counter(dict(c["counts"]))
            if key[0] == 0 or sum(counts.values()) < 5 * len(counts) or min(counts.values()) < 2:
                continue
            parent = (key[0], key[2] if name.startswith("order2") else key[1])
            selected.append(counts)
            parents[parent].update(counts)
        fine = sum(entropy(c) for c in selected)
        coarse = sum(entropy(c) for c in parents.values())
        fine_dof = sum(len(c) - 1 for c in selected)
        coarse_dof = sum(len(c) - 1 for c in parents.values())
        n = sum(sum(c.values()) for c in selected)
        output[name] = {
            "supported_coded_symbols": n,
            "coded_symbol_denominator": 17061,
            "excluded_zero_depth_symbols": 3355,
            "supported_contexts": len(selected),
            "matched_subset_plugin_gain_bytes": (coarse - fine) / 8,
            "matched_subset_MM_gain_estimate_bytes": (coarse - fine + (coarse_dof - fine_dof) / (2 * math.log(2))) / 8,
            "subset_context_plugin_bytes": fine / 8,
            "subset_context_MM_estimate_bytes": (fine + fine_dof / (2 * math.log(2))) / 8,
            "full_population_MM_bound": None,
        }
    result = {
        "axis": AXIS,
        "score_claim": False,
        "coder_runs": 0,
        "rows": output,
        "scope": "same selected observations for fine and previous-only contexts; no extrapolation to excluded sparse cells",
        "support_rule": "n >= 5 occupied outcomes and each observed outcome occurs >=2; asymptotic screening heuristic, not a confidence guarantee",
        "interpretation": "the saturated empirical conditional entropy is an oracle logloss minimum only within its specified static context family; MM is an estimate, never a rigorous entropy/coding bound",
    }
    save_json(STORE / "SUPPORTED_CONTEXTS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--supported-contexts", action="store_true")
    from comma_lab.instrument_gates import add_coder_gate_argument, check_experimental_coder

    add_coder_gate_argument(parser)
    args = parser.parse_args()
    instrument_gate = check_experimental_coder(LIVE, rationale=args.coder_differs_because)
    save_json(STORE / "INSTRUMENT_GATE.json", instrument_gate)
    if args.supported_contexts:
        print(json.dumps(supported_contexts(), indent=2))
        raise SystemExit(0)
    receipt = run(args.resume_from)
    print(
        json.dumps(
            {k: receipt[k] for k in ("contexts", "rc2_J_bytes", "remaining_J_minus_H1_MM_bytes", "coder_runs")},
            indent=2,
        )
    )
